"""Check cargo commands are bridged from ROS to Gazebo."""

import ast
from pathlib import Path
import xml.etree.ElementTree as ET


def test_cargo_bridge_has_ros_to_gazebo_command_topics():
    root = Path(__file__).resolve().parents[2]
    launch = root / 'tugger_gazebo' / 'launch' / 'simulation.launch.py'
    tree = ast.parse(launch.read_text(encoding='utf-8'))
    cargo_node = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and any(keyword.arg == 'name'
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value == 'cargo_attachment_bridge'
                for keyword in node.keywords)
    )
    arguments = next(keyword.value for keyword in cargo_node.keywords
                     if keyword.arg == 'arguments')
    topics = {node.value for node in arguments.elts}
    commands = {
        f'/cargo/station_{station}/{action}@std_msgs/msg/Empty]gz.msgs.Empty'
        for station in (0, 3) for action in ('attach', 'detach')
    }
    # Phase B4: only the lift actuator commands are bridged; the cargo state
    # and the collision scene come from the robot's own estimate, never from
    # the simulator.
    assert topics == commands


def test_cargo_plugins_reference_actual_movable_payload_models():
    root = Path(__file__).resolve().parents[2]
    robot = ET.parse(root / 'tugger_description/urdf/tugger.urdf.xacro').getroot()
    world = ET.parse(root / 'tugger_gazebo/worlds/tugger_warehouse.sdf').getroot()
    plugins = [p for p in robot.findall('./gazebo/plugin')
               if p.get('name') == 'tugger_gazebo::GuardedCargoJoint']
    assert len(plugins) == 2
    for station, plugin in zip((0, 3), plugins):
        name = plugin.findtext('child_model')
        assert name == f'cargo_station_{station}_stand'
        model = world.find(f"./world/model[@name='{name}']")
        assert model is not None and model.findtext('static') == 'false'
        assert model.find(f"./link[@name='{plugin.findtext('child_link')}']") is not None
        assert plugin.findtext('attach_topic') == f'/cargo/station_{station}/attach'
        assert plugin.findtext('detach_topic') == f'/cargo/station_{station}/detach'


def test_collision_scene_plugin_is_declared_in_world():
    root = Path(__file__).resolve().parents[2]
    world = ET.parse(root / 'tugger_gazebo/worlds/tugger_warehouse.sdf').getroot()
    plugins = world.findall("./world/plugin[@name='tugger_gazebo::CollisionScene']")
    assert len(plugins) == 1
    assert plugins[0].get('filename') == 'tugger_collision_scene'
