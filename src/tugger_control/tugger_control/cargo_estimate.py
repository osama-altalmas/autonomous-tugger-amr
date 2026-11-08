"""Cargo state and collision scene from the robot's own knowledge (Phase B4).

Replaces three simulator-only inputs with what a real AGV has:

* Which cargo is attached: an attach request, the cargo's identity read by
  the top QR reader from below just before lifting, and the lift's position
  feedback at contact height or above (from contact the lift carries it). A
  detach is the request plus the lift back at contact height or below.
* Where each cargo is: on the lift, it moves with the robot's localized pose;
  set down, it stays where the robot left it. The starting inventory is the
  site layout (the world file), as a real site's commissioning data.
* The collision scene: the static site geometry (walls, racks, stand legs)
  plus each cargo at its estimated pose, in the format the motion gate and
  panel already use. Unknown, moving obstacles are the safety monitor's job.

Pure logic; no ROS.
"""

import json
import math
import xml.etree.ElementTree as ET

from tugger_control.lift_geometry import (CONTACT_TOLERANCE_M, LIFT_CONTACT_M,
                                          cargo_lift_offset)
from tugger_control.warehouse_clearance import static_world_boxes

CARGO_IDS = (0, 3)
# A QR read older than this does not identify the cargo above the lift.
QR_MAX_AGE_S = 3.0


def cargo_model(cargo_id):
    return f'cargo_station_{cargo_id}_stand'


def initial_cargo(world_path):
    """Starting cargo poses (x, y, z, yaw) and their collision boxes, from the site layout."""
    root = ET.parse(world_path).getroot()
    poses, shapes = {}, {}
    for cargo_id in CARGO_IDS:
        model = root.find(f"./world/model[@name='{cargo_model(cargo_id)}']")
        if model is None:
            raise ValueError(f'site layout has no {cargo_model(cargo_id)}')
        x, y, z, _, _, yaw = (float(v) for v in model.findtext('pose', '0 0 0 0 0 0').split())
        poses[cargo_id] = (x, y, z, yaw)
        boxes = []
        for collision in model.findall('./link/collision'):
            size = [float(v) for v in collision.findtext('geometry/box/size').split()]
            offset = [float(v) for v in collision.findtext('pose', '0 0 0 0 0 0').split()]
            boxes.append((collision.get('name'), offset[2], size))
        shapes[cargo_id] = boxes
    return poses, shapes


class CargoEstimate:
    """Attach state and pose of every cargo, from requests, lift and QR."""

    def __init__(self, poses, station_centres=None):
        self.poses = dict(poses)
        # Where a cargo set down at a station rests (the stand centre).
        self.station_centres = dict(station_centres or {})
        self.attached = None          # cargo id on the lift, or None
        self.offset = None            # cargo pose relative to the robot while attached
        self.pending = None           # (action, cargo id) requested
        self.qr = (None, None)        # (cargo id read from below, time)

    def request(self, action, cargo_id):
        self.pending = (action, cargo_id)

    def qr_read(self, cargo_id, now):
        self.qr = (cargo_id, now)

    def update(self, robot_pose, lift, now):
        """Advance attach/detach from the lift; move an attached cargo with the robot."""
        if self.pending is not None and lift is not None:
            action, cargo_id = self.pending
            seen, at = self.qr
            # The lift carries the cargo from contact height up, and sets it
            # down on its legs at contact height and below.
            at_or_above_contact = lift >= LIFT_CONTACT_M - CONTACT_TOLERANCE_M
            if action == 'attach' and self.attached is None and at_or_above_contact:
                # Only the cargo actually identified above the lift is attached.
                if seen == cargo_id and at is not None and 0 <= now - at <= QR_MAX_AGE_S:
                    # The robot lifts only when docked within 10 mm of the
                    # station marker, which marks the cargo centre: the cargo
                    # sits centred on the lift. Only its relative heading is
                    # kept (independent of the localization error).
                    yaw = self.poses[cargo_id][3]
                    self.offset = (0.0, 0.0, yaw - robot_pose[2])
                    self.attached = cargo_id
                self.pending = None
            elif (action == 'detach' and self.attached == cargo_id and
                  lift <= LIFT_CONTACT_M + CONTACT_TOLERANCE_M):
                self._move_with(robot_pose, 0.0)
                # Set down at a stand: it rests on the stand's centre.
                x, y, _, yaw = self.poses[cargo_id]
                near = [c for c in self.station_centres.values()
                        if math.hypot(c[0] - x, c[1] - y) <= 0.10]
                if near:
                    self.poses[cargo_id] = (near[0][0], near[0][1], 0.0, yaw)
                self.attached, self.offset, self.pending = None, None, None
        if self.attached is not None and robot_pose is not None and lift is not None:
            self._move_with(robot_pose, cargo_lift_offset(lift))

    def _move_with(self, robot_pose, height):
        rx, ry, ryaw = robot_pose
        dx, dy, dyaw = self.offset
        c, s = math.cos(ryaw), math.sin(ryaw)
        yaw = math.atan2(math.sin(ryaw + dyaw), math.cos(ryaw + dyaw))
        self.poses[self.attached] = (rx + c * dx - s * dy, ry + s * dx + c * dy, height, yaw)

    def state(self, cargo_id):
        return 'attached' if self.attached == cargo_id else 'detached'


def scene_payload(static_boxes, shapes, poses, stamp):
    """Collision scene JSON: static site boxes plus each cargo at its pose."""
    boxes = []
    for box in static_boxes:
        # Axis-aligned bounds (the scene format has no yaw): conservative.
        c, s = abs(math.cos(box.yaw)), abs(math.sin(box.yaw))
        hx, hy = c * box.half_x + s * box.half_y, s * box.half_x + c * box.half_y
        boxes.append({'name': box.name, 'model': box.name.split('/', 1)[0],
                      'x': box.x, 'y': box.y, 'z': (box.z_min + box.z_max) / 2,
                      'hx': hx, 'hy': hy, 'hz': (box.z_max - box.z_min) / 2})
    for cargo_id, pose in poses.items():
        x, y, z, yaw = pose
        c, s = abs(math.cos(yaw)), abs(math.sin(yaw))
        for name, offset_z, size in shapes[cargo_id]:
            hx = c * size[0] / 2 + s * size[1] / 2
            hy = s * size[0] / 2 + c * size[1] / 2
            boxes.append({'name': f'{cargo_model(cargo_id)}/{name}', 'model': cargo_model(cargo_id),
                          'x': x, 'y': y, 'z': z + offset_z, 'hx': hx, 'hy': hy,
                          'hz': size[2] / 2})
    return json.dumps({'valid': True, 'stamp': stamp, 'boxes': boxes})


def load_site(world_path):
    """Static boxes, cargo shapes, starting poses and stand centres of the site."""
    poses, shapes = initial_cargo(world_path)
    static = static_world_boxes(world_path)
    legs = {}
    for box in static:
        model = box.name.split('/', 1)[0]
        if model.endswith('_legs'):
            legs.setdefault(model, []).append((box.x, box.y))
    centres = {model: (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))
               for model, pts in legs.items()}
    return static, shapes, poses, centres
