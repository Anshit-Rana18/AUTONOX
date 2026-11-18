import json
a = json.load(open("data/outputs/speed_events.json"))
b = [e for e in json.load(open("data/outputs/final2_events.json")) if e["kind"] == "OVERSPEED"]
print(f"run_speed: {len(a)} events   run_final2: {len(b)} events\n")
for e in sorted(a, key=lambda e: e["ts"]):
    m = [x for x in b if abs(x["ts"] - e["ts"]) < 1.0]
    print(f'speed  t={e["ts"]:6.2f}  track {e["track_id"]:>4}  {e["speed_kmh"]:5.1f} km/h',
          "  matched" if m else "  NOT in final2")
for x in sorted(b, key=lambda e: e["ts"]):
    if not any(abs(x["ts"] - e["ts"]) < 1.0 for e in a):
        print(f'final2 t={x["ts"]:6.2f}  track {x["track_id"]:>4}  {x["speed_kmh"]:5.1f} km/h  NOT in run_speed')

        