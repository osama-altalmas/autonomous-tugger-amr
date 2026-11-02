#!/usr/bin/env python3

from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

world_path = Path("src/tugger_gazebo/worlds/tugger_warehouse.sdf")
backup_path = world_path.with_name(
    "tugger_warehouse.before_cargo_stations.sdf"
)

if not backup_path.exists():
    shutil.copy2(world_path, backup_path)

tree = ET.parse(world_path)
root = tree.getroot()
world = root.find("world")

if world is None:
    raise RuntimeError("SDF world not found")

stations = [
    (-17.5, -8.0),
    (-17.5,  8.0),
    ( 17.5, -8.0),
    ( 17.5,  8.0),
]

# Cargo geometry
cargo_length = 1.20
cargo_width = 1.00
cargo_height = 0.50

# Stand geometry
stand_height = 0.35
leg_width = 0.08

# Opening between the legs:
# length: 1.04 m
# width:  0.84 m
#
# AMR chassis collision:
# length: 0.88 m
# width:  0.58 m

def add_box_model(name, x, y, z, sx, sy, sz, color, static=True):
    model = ET.SubElement(world, "model", name=name)
    ET.SubElement(model, "static").text = (
        "true" if static else "false"
    )
    ET.SubElement(model, "pose").text = (
        f"{x} {y} {z} 0 0 0"
    )

    link = ET.SubElement(model, "link", name="link")

    collision = ET.SubElement(link, "collision", name="collision")
    geometry = ET.SubElement(collision, "geometry")
    box = ET.SubElement(geometry, "box")
    ET.SubElement(box, "size").text = f"{sx} {sy} {sz}"

    visual = ET.SubElement(link, "visual", name="visual")
    geometry = ET.SubElement(visual, "geometry")
    box = ET.SubElement(geometry, "box")
    ET.SubElement(box, "size").text = f"{sx} {sy} {sz}"

    material = ET.SubElement(visual, "material")
    ET.SubElement(material, "ambient").text = color
    ET.SubElement(material, "diffuse").text = color

    return model

# Remove previous loading pallets and generated cargo stations
for model in list(world.findall("model")):
    name = model.get("name", "")

    if (
        name.startswith("loading_pallet_")
        or name.startswith("cargo_station_")
    ):
        world.remove(model)

for i, (x, y) in enumerate(stations):

    # Four supports at the corners
    for ix, dx in enumerate((-1, 1)):
        for iy, dy in enumerate((-1, 1)):

            leg_x = x + dx * (cargo_length / 2 - leg_width / 2)
            leg_y = y + dy * (cargo_width / 2 - leg_width / 2)

            add_box_model(
                name=f"cargo_station_{i}_leg_{ix}_{iy}",
                x=leg_x,
                y=leg_y,
                z=stand_height / 2,
                sx=leg_width,
                sy=leg_width,
                sz=stand_height,
                color="0.15 0.18 0.22 1",
            )

    # Cargo box resting on the supports
    add_box_model(
        name=f"cargo_station_{i}_box",
        x=x,
        y=y,
        z=stand_height + cargo_height / 2,
        sx=cargo_length,
        sy=cargo_width,
        sz=cargo_height,
        color="0.65 0.42 0.20 1",
    )

ET.indent(tree, space="  ")
tree.write(world_path, encoding="utf-8", xml_declaration=True)

print("Cargo stations generated successfully.")
print("Stations:", len(stations))
print("Support legs:", len(stations) * 4)
print("Cargo boxes:", len(stations))
print("Opening: 1.04 m x 0.84 m")
print("Vertical clearance: 0.35 m")
print("Backup:", backup_path)
