import json, os, sys
sys.path.insert(0, ".")
from evidence.integrity import sha256_file

folder = sys.argv[1]
m = json.load(open(os.path.join(folder, "manifest.json")))
bad = 0
for f, info in m["files"].items():
    p = os.path.join(folder, f)
    if not os.path.exists(p):
        print(f"MISSING   {f}"); bad += 1
    elif sha256_file(p) != info["sha256"]:
        print(f"CHANGED   {f}"); bad += 1
for f in os.listdir(folder):
    if f != "manifest.json" and f not in m["files"]:
        print(f"NOT IN MANIFEST  {f}"); bad += 1
print("All files match the manifest." if not bad else f"{bad} problem(s) found.")

