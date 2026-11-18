import hashlib, json, os
from datetime import datetime, timezone

def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def video_fingerprint(path):
    return {"file": os.path.basename(path), "sha256": sha256_file(path),
            "bytes": os.path.getsize(path)}

def finalize_folder(folder, updates=None):
    """Adds source info to record.json, then writes manifest.json with a SHA-256 for every file."""
    rec_path = os.path.join(folder, "record.json")
    if updates and os.path.exists(rec_path):
        rec = json.load(open(rec_path))
        rec.update(updates)
        json.dump(rec, open(rec_path, "w"), indent=2)
    files = {}
    for f in sorted(os.listdir(folder)):
        if f == "manifest.json":
            continue
        p = os.path.join(folder, f)
        files[f] = {"sha256": sha256_file(p), "bytes": os.path.getsize(p)}
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "files": files}
    json.dump(manifest, open(os.path.join(folder, "manifest.json"), "w"), indent=2)
    return manifest

