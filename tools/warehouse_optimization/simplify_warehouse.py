#!/usr/bin/env python3

from pathlib import Path
from collections import defaultdict
import copy
import re
import shutil
import xml.etree.ElementTree as ET

world_file = Path("src/tugger_gazebo/worlds/tugger_warehouse.sdf")
backup_file = world_file.with_name("tugger_warehouse.before_simplification.sdf")

if not backup_file.exists():
    shutil.copy2(world_file, backup_file)

tree = ET.parse(world_file)
root = tree.getroot()
world = root.find("world")

if world is None:
    raise RuntimeError("World element not found")

rack_groups = defaultdict(list)

for model in world.findall("model"):
    name = model.get("name", "")
    match = re.match(r"^(compact_rack_\d+_\d+)_", name)

    if match:
        rack_groups[match.group(1)].append(model)

if not rack_groups:
    raise RuntimeError(
        "No detailed rack groups found. "
        "The world may already be simplified."
    )

def pose_xyz(model):
    pose = model.findtext("pose", "0 0 0 0 0 0")
    values = [float(v) for v in pose.split()]
    return values[:3]

def geometry_size(model):
    size_text = model.findtext(
        "./link/collision/geometry/box/size"
    )
    if not size_text:
        return None
    return [float(v) for v in size_text.split()]

def make_zone(name, cx, cy, sx, sy):
    height = 1.5

    model = ET.Element("model", name=name)
    ET.SubElement(model, "static").text = "true"
    ET.SubElement(model, "pose").text = (
        f"{cx:.5f} {cy:.5f} {height/2:.5f} 0 0 0"
    )

    link = ET.SubElement(model, "link", name="link")

    collision = ET.SubElement(link, "collision", name="collision")
    geometry = ET.SubElement(collision, "geometry")
    box = ET.SubElement(geometry, "box")
    ET.SubElement(box, "size").text = f"{sx:.5f} {sy:.5f} {height}"

    visual = ET.SubElement(link, "visual", name="visual")
    geometry = ET.SubElement(visual, "geometry")
    box = ET.SubElement(geometry, "box")
    ET.SubElement(box, "size").text = f"{sx:.5f} {sy:.5f} {height}"

    material = ET.SubElement(visual, "material")
    ET.SubElement(material, "ambient").text = "0.8 0.05 0.05 1"
    ET.SubElement(material, "diffuse").text = "0.9 0.08 0.08 1"

    return model

zones = []

for rack_name, models in sorted(rack_groups.items()):
    bounds = []

    for model in models:
        size = geometry_size(model)
        if size is None:
            continue

        x, y, z = pose_xyz(model)
        sx, sy, sz = size

        bounds.append((
            x - sx/2,
            x + sx/2,
            y - sy/2,
            y + sy/2,
        ))

    if not bounds:
        raise RuntimeError(f"No box geometry found for {rack_name}")

    xmin = min(b[0] for b in bounds)
    xmax = max(b[1] for b in bounds)
    ymin = min(b[2] for b in bounds)
    ymax = max(b[3] for b in bounds)

    zones.append(make_zone(
        f"forbidden_zone_{rack_name}",
        (xmin + xmax) / 2,
        (ymin + ymax) / 2,
        xmax - xmin,
        ymax - ymin,
    ))

removed = sum(len(models) for models in rack_groups.values())

for models in rack_groups.values():
    for model in models:
        world.remove(model)

for zone in zones:
    world.append(zone)

ET.indent(tree, space="  ")
tree.write(world_file, encoding="utf-8", xml_declaration=True)

print("Warehouse simplification complete")
print(f"Original rack components removed: {removed}")
print(f"Red forbidden zones created: {len(zones)}")
print(f"Original world backup: {backup_file}")
print(f"Remaining models: {len(world.findall('model'))}")
