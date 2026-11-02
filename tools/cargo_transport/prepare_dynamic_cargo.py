#!/usr/bin/env python3

from pathlib import Path
import xml.etree.ElementTree as ET
import shutil

world_file = Path(
    "src/tugger_gazebo/worlds/tugger_warehouse.sdf"
)

backup = world_file.with_name(
    "tugger_warehouse.before_dynamic_cargo.sdf"
)

if not backup.exists():
    shutil.copy2(world_file, backup)

tree = ET.parse(world_file)
root = tree.getroot()
world = root.find("world")

if world is None:
    raise RuntimeError("World element not found")

cargo_names = [
    "cargo_station_0_box",
    "cargo_station_3_box",
]

for name in cargo_names:
    model = next(
        (
            m for m in world.findall("model")
            if m.get("name") == name
        ),
        None,
    )

    if model is None:
        raise RuntimeError(f"Cargo model not found: {name}")

    static = model.find("static")
    if static is None:
        static = ET.SubElement(model, "static")

    static.text = "false"

    link = model.find("link")
    if link is None:
        raise RuntimeError(f"Link not found in {name}")

    inertial = link.find("inertial")

    if inertial is None:
        inertial = ET.SubElement(link, "inertial")

    # 10 kg rectangular cargo:
    # dimensions = 1.2 x 1.0 x 0.5 m

    for child in list(inertial):
        inertial.remove(child)

    ET.SubElement(inertial, "mass").text = "10.0"

    inertia = ET.SubElement(inertial, "inertia")

    mass = 10.0
    x, y, z = 1.2, 1.0, 0.5

    values = {
        "ixx": mass * (y*y + z*z) / 12,
        "iyy": mass * (x*x + z*z) / 12,
        "izz": mass * (x*x + y*y) / 12,
        "ixy": 0,
        "ixz": 0,
        "iyz": 0,
    }

    for key, value in values.items():
        ET.SubElement(inertia, key).text = str(value)

    print(f"Configured dynamic cargo: {name}")

ET.indent(tree, space="  ")
tree.write(
    world_file,
    encoding="utf-8",
    xml_declaration=True,
)

print("Dynamic cargo configuration complete.")
