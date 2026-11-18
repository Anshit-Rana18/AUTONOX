import json, sys
from shapely.geometry import Polygon
from shapely.validation import explain_validity

data = json.load(open(sys.argv[1]))
for l in data["lanes"]:
    poly = Polygon(l["polygon"])
    if poly.is_valid:
        print(f"{l['name']}: ok  ({len(l['polygon'])} points, area {poly.area:.0f})")
    else:
        print(f"{l['name']}: INVALID  {explain_validity(poly)}")


