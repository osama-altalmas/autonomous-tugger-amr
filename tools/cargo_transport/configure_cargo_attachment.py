#!/usr/bin/env python3

from pathlib import Path
import xml.etree.ElementTree as ET
import shutil

urdf_path = Path(
    "src/tugger_description/urdf/tugger.urdf.xacro"
)

world_path = Path(
    "src/tugger_gazebo/worlds/tugger_warehouse.sdf"
)

for path in (urdf_path, world_path):
    backup = path.with_name(path.name + ".before_attachment")
    if not backup.exists():
        shutil.copy2(path, backup)

tree = ET.parse(world_path)
world = tree.getroot().find("world")

if world is None:
    raise RuntimeError("World not found")

for station in (0, 3):
    name = f"cargo_station_{station}_box"

    model = next(
        (m for m in world.findall("model")
         if m.get("name") == name),
        None,
    )

    if model is None:
        raise RuntimeError(f"Missing cargo: {name}")

    static = model.find("static")

    if static is None or static.text.strip() != "false":
        raise RuntimeError(
            f"{name} is not dynamic. "
            "Run the dynamic cargo conversion first."
        )

text = urdf_path.read_text()

for station in (0, 3):
    marker = f"cargo_attachment_station_{station}"

    if marker in text:
        continue

    plugin = f'''
  <gazebo>
    <plugin
      filename="gz-sim-detachable-joint-system"
      name="gz::sim::systems::DetachableJoint">

      <parent_link>lift_platform_link</parent_link>
      <child_model>cargo_station_{station}_box</child_model>
      <child_link>link</child_link>

      <topic>/cargo/station_{station}/attach</topic>
      <detach_topic>/cargo/station_{station}/detach</detach_topic>

      <output_topic>/cargo/station_{station}/state</output_topic>
      <suppress_child_warning>true</suppress_child_warning>

    </plugin>
  </gazebo>
'''

    if "</robot>" not in text:
        raise RuntimeError("Closing robot tag not found")

    text = text.replace("</robot>", plugin + "\n</robot>", 1)

urdf_path.write_text(text)

print("Cargo attachment plugins configured.")
print("Stations: 0 (SW), 3 (NE)")
