
from collections import defaultdict


def _dist(a, b):
    """Hamming distance between equal-length plate strings."""

    if not a or not b or len(a) != len(b):
        return 99

    return sum(x != y for x, y in zip(a, b))


def _gap(a, b):
    """Seconds between two tracks' lifetimes. Zero if they overlap."""

    return max(
        0.0,
        max(a[0], b[0]) - min(a[1], b[1])
    )


def _can_merge(a, b, state, seen, window_s, max_dist):
    """
    Check whether two tracks are compatible for merging.
    """

    sa = state.get(a, {})
    sb = state.get(b, {})

    plate_a = sa.get("plate")
    plate_b = sb.get("plate")

    if not plate_a or not plate_b:
        return False

    # Plate strings must be sufficiently similar.
    distance = _dist(plate_a, plate_b)

    if distance > max_dist:
        return False

    # Never merge different confirmed plate readings.
    if (
        distance > 0
        and sa.get("status") == "CONFIRMED"
        and sb.get("status") == "CONFIRMED"
    ):
        return False

    # Tracks must be close enough in time.
    if _gap(seen[a], seen[b]) > window_s:
        return False

    return True


def group_tracks(
    best,
    state,
    seen,
    reader,
    window_s=6.0,
    max_dist=1,
    all_reads=None
):
    """
    Merge tracks that are probably the same physical vehicle.

    Improvements:
    1. Uses all OCR observations for final group consensus.
    2. Uses conservative complete-link merging.
    3. Prevents transitive chains from merging incompatible tracks.
    4. Preserves the best-quality evidence crop.
    """

    # Only tracks with a usable plate candidate can participate.
    ids = [
        tid
        for tid in best
        if state.get(tid, {}).get("plate")
    ]

    # Start with each track as its own group.
    groups = [{tid} for tid in ids]

    # --------------------------------------------------
    # CONSERVATIVE GROUP MERGING
    # --------------------------------------------------

    changed = True

    while changed:

        changed = False

        for i in range(len(groups)):

            if changed:
                break

            for j in range(i + 1, len(groups)):

                group_a = groups[i]
                group_b = groups[j]

                # Every cross-pair must be compatible.
                compatible = all(
                    _can_merge(
                        a,
                        b,
                        state,
                        seen,
                        window_s,
                        max_dist
                    )
                    for a in group_a
                    for b in group_b
                )

                if compatible:

                    groups[i] = group_a | group_b
                    groups.pop(j)

                    changed = True
                    break

    # --------------------------------------------------
    # BUILD FINAL VEHICLE GROUPS
    # --------------------------------------------------

    out = []

    for members in groups:

        members = sorted(members)

        # Pool all observations used by track-level OCR decisions.
        if all_reads is not None:

            reads = [
                observation
                for tid in members
                for observation in all_reads.get(tid, [])
            ]

        else:

            # Backward-compatible fallback.
            reads = [
                b["read"]
                for tid in members
                for b in best[tid]
            ]

        status, plate = reader.decide(reads)

        # Preserve the highest-quality crop as evidence.
        top = max(
            (
                b
                for tid in members
                for b in best[tid]
            ),
            key=lambda b: b["score"]
        )

        out.append({
            "tracks": members,
            "status": status,
            "plate": plate,
            "top": top,
            "reads": reads,
            "first_ts": min(
                seen[tid][0] for tid in members
            ),
            "last_ts": max(
                seen[tid][1] for tid in members
            )
        })

    # --------------------------------------------------
    # TRACKS WITH NO USABLE PLATE CANDIDATE
    # --------------------------------------------------

    for tid in best:

        if tid not in ids:

            top = max(
                best[tid],
                key=lambda b: b["score"]
            )

            reads = (
                all_reads.get(tid, [])
                if all_reads is not None
                else [b["read"] for b in best[tid]]
            )

            out.append({
                "tracks": [tid],
                "status": "UNREADABLE",
                "plate": None,
                "top": top,
                "reads": reads,
                "first_ts": seen[tid][0],
                "last_ts": seen[tid][1]
            })

    return out
