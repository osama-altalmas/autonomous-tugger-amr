#!/usr/bin/env python3

from pathlib import Path
import xml.etree.ElementTree as ET
import re
import shutil

urdf_path = Path(
    "src/tugger_description/urdf/tugger.urdf.xacro"
)
world_path = Path(
    "src/tugger_gazebo/worlds/tugger_warehouse.sdf"
)

for path in (urdf_path, world_path):
    backup = path.with_name(path.name + ".before_compact_lift")
    if not backup.exists():
        shutil.copy2(path, backup)

# --------------------------------------
# 1. Reduce lift joint travel
# --------------------------------------

text = urdf_path.read_text()

pattern = (
    r'(<joint name="lift_joint"\s+type="prismatic">)'
    r'(.*?)'
    r'(</joint>)'
)

match = re.search(pattern, text, re.S)

if not match:
    raise RuntimeError("Lift joint not found")

joint_body = match.group(2)

joint_body, count = re.subn(
    r'upper="[^"]+"',
    'upper="0.065"',
    joint_body,
    count=1,
)

if count != 1:
    raise RuntimeError("Lift upper limit not found")

text = (
    text[:match.start(2)]
    + joint_body
    + text[match.end(2):]
)

urdf_path.write_text(text)

# --------------------------------------
# 2. Lower cargo stands and boxes
# --------------------------------------

tree = ET.parse(world_path)
world = tree.getroot().find("world")

if world is None:
    raise RuntimeError("SDF world not found")

station_ids = (0, 3)

# Original:
# legs 0.35 m high
# cargo center z=0.60 m
#
# New:
# legs 0.28 m high
# cargo center z=0.53 m

updated = []

for model in world.findall("model"):
    name = model.get("name", "")

    if not name.startswith(
        tuple(f"cargo_station_{i}_" for i in station_ids)
    ):
        continue

    pose = model.find("pose")

    if pose is None:
        raise RuntimeError(f"Missing pose: {name}")

    coords = pose.text.split()

    if name.endswith("_box"):
        coords[2] = "0.53"

    elif "_leg_" in name:
        coords[2] = "0.14"

        for size in model.findall(".//box/size"):
            dims = size.text.split()
            dims[2] = "0.28"
            size.text = " ".join(dims)

    else:
        continue

    pose.text = " ".join(coords)
    updated.append(name)

if len(updated) != 10:
    raise RuntimeError(
        f"Expected 10 cargo models, found {len(updated)}"
    )

ET.indent(tree, space="  ")
tree.write(
    world_path,
    encoding="utf-8",
    xml_declaration=True,
)

print("Compact lifting geometry configured.")
print("Lift travel: 0.000 to 0.065 m")
print("Stand height: 0.280 m")
print("Cargo center height: 0.530 m")
print(f"Updated {len(updated)} cargo models.")
