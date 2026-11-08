#!/usr/bin/env python3
"""Safety monitor: scanners, 3D cameras, bumper and emergency stops.

Plays the role of the safety controller on an ISO 3691-4 AGV. It fuses the
two corner safety scanners and the two depth cameras into obstacle points in
the robot frame, selects the protective and warning fields for the current
speed, steering and mode (safety_fields.py), and publishes one state at 25 Hz
on /tugger/safety/state (JSON): state, speed limit, reason and stamp. The
motion gate obeys it fail-closed: a missing or stale state stops the robot.

Emergency stops (four corner buttons, the operator panel button) and the
bumper latch until released and reset on /tugger/safety/reset. A protective
stop restarts automatically once its field has been clear for 2 s. Stale
scanner data is itself a protective stop.

Status light and sound follow the state on /tugger/signals (JSON).
"""
import json
import math

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from geometry_msgs.msg import PoseStamped
from ros_gz_interfaces.msg import Contacts
from sensor_msgs.msg import JointState, LaserScan, PointCloud2
from std_msgs.msg import Bool, Empty, Float64MultiArray, String
from tf2_ros import Buffer, TransformListener

from tugger_control.safety_fields import (
    CLEAR, EMERGENCY_STOP, PROTECTIVE_STOP, RestartTimer, WARNING, classify_crab, classify_points,
    speed_limit)
from tugger_control.turn_in_place import PIVOT_RADIUS_M, YAW_ACCELERATION, WHEELBASE_M

SCANNERS = ('/front_left/scan', '/rear_right/scan')
DEPTH_CLOUDS = ('/front_depth/points', '/rear_depth/points')
ESTOP_BUTTONS = ('front_left', 'front_right', 'rear_left', 'rear_right', 'panel')
STATION_OWNERS = ('DOCK_NAV', 'DOCK_CAMERA', 'DOCK_DEPART', 'PRECISION', 'ROTATE')
SCAN_MAX_AGE_S = 0.2           # 5 missed 25 Hz scans
DEPTH_MAX_AGE_S = 0.5
# Obstacle heights the robot can touch: unloaded it passes under anything
# above its lift top (0.2375 m), for example a stand's top plate; loaded,
# the cargo and its goods may reach up to 2 m (KMP 600-S detects 50 mm to
# 2.1 m). The robot's own raised cargo is not an obstacle.
MIN_OBSTACLE_Z = 0.05
# Unloaded: up to the chassis top (0.2225 m). The robot passes 4 cm under a
# stand's plate (0.28 m), which depth noise at grazing angles smears down to
# about 0.25 m; anything that would touch only the 1.5 cm lift edge is left
# to the bumper. People and objects rise from the floor and are still seen.
UNLOADED_MAX_Z = 0.22
LOADED_MAX_Z = 2.0
# The 1.3 m cargo square plus 5 cm.
CARGO_HALF_M = 0.70
LOADED_LIFT_M = 0.03
FIELD_RANGE_M = 5.0
LIGHTS = {CLEAR: 'GREEN', WARNING: 'YELLOW', PROTECTIVE_STOP: 'RED',
          EMERGENCY_STOP: 'RED_FLASHING'}
SOUNDS = {CLEAR: 'OFF', WARNING: 'WARNING_TONE', PROTECTIVE_STOP: 'STOP_TONE',
          EMERGENCY_STOP: 'ALARM'}


def _yaw(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


class SafetyMonitor(Node):
    def __init__(self):
        super().__init__('tugger_safety_monitor')
        self.set_parameters([Parameter('use_sim_time', value=True)])
        self._tf = Buffer()
        self._listener = TransformListener(self._tf, self)
        self._mounts = {}
        self._scan_points = {topic: None for topic in SCANNERS}
        self._scan_at = {topic: None for topic in SCANNERS}
        self._depth_points = {topic: None for topic in DEPTH_CLOUDS}
        self._depth_at = {topic: None for topic in DEPTH_CLOUDS}
        self._depth_seen = set()
        self._motion = None
        self._owner = 'STOP'
        self._pressed = {name: False for name in ESTOP_BUTTONS}
        self._bumper_at = None   # last contact (the sensor reports only contacts)
        self._lift = 0.0
        # Attached cargo (cargo_estimator): its footprint is not an obstacle.
        self._cargo_state = {0: 'detached', 3: 'detached'}
        self._cargo_pose = {0: None, 3: None}
        self._timer = RestartTimer()
        self._last = None
        for topic in SCANNERS:
            self.create_subscription(LaserScan, topic,
                                     lambda msg, t=topic: self._on_scan(t, msg), 10)
        for topic in DEPTH_CLOUDS:
            self.create_subscription(PointCloud2, topic,
                                     lambda msg, t=topic: self._on_depth(t, msg), 5)
        for name in ESTOP_BUTTONS:
            self.create_subscription(Bool, f'/tugger/estop/{name}',
                                     lambda msg, n=name: self._on_estop(n, msg), 10)
        self.create_subscription(Contacts, '/tugger/bumper/contacts', self._on_bumper, 10)
        self.create_subscription(Empty, '/tugger/safety/reset', self._on_reset, 10)
        self.create_subscription(Float64MultiArray, '/four_ws/motion_state',
                                 self._on_motion, 10)
        self.create_subscription(String, '/tugger/motion_owner', self._on_owner, 10)
        self.create_subscription(JointState, '/joint_states', self._on_joints, 10)
        for cargo in (0, 3):
            self.create_subscription(String, f'/cargo/station_{cargo}/state',
                                     lambda m, c=cargo: self._cargo_state.__setitem__(c, m.data), 10)
            self.create_subscription(PoseStamped, f'/tugger/cargo/station_{cargo}/estimated_pose',
                                     lambda m, c=cargo: self._cargo_pose.__setitem__(c, m), 10)
        self._state_pub = self.create_publisher(String, '/tugger/safety/state', 10)
        self._signal_pub = self.create_publisher(String, '/tugger/signals', 10)
        self.create_timer(0.04, self._tick)
        self.get_logger().info('safety monitor: 2 scanners, 2 depth cameras, bumper, 5 e-stops')

    def _now(self):
        return self.get_clock().now().nanoseconds / 1e9

    def _mount(self, frame):
        """Sensor frame -> base_footprint as (x, y, yaw, z), cached (static)."""
        if frame not in self._mounts:
            t = self._tf.lookup_transform('base_footprint', frame, rclpy.time.Time())
            tr = t.transform.translation
            self._mounts[frame] = (tr.x, tr.y, _yaw(t.transform.rotation), tr.z)
        return self._mounts[frame]

    def _on_scan(self, topic, msg):
        try:
            mx, my, myaw, _ = self._mount(msg.header.frame_id)
        except Exception:
            return  # TF not yet available: the scan stays stale (stop)
        ranges = np.asarray(msg.ranges, dtype=float)
        angles = msg.angle_min + msg.angle_increment * np.arange(len(ranges))
        ok = np.isfinite(ranges) & (ranges >= msg.range_min) & (ranges <= min(
            msg.range_max, FIELD_RANGE_M))
        a = angles[ok] + myaw
        r = ranges[ok]
        self._scan_points[topic] = np.column_stack((mx + r * np.cos(a), my + r * np.sin(a)))
        self._scan_at[topic] = self._now()

    def _on_depth(self, topic, msg):
        try:
            from sensor_msgs_py import point_cloud2
            mx, my, myaw, mz = self._mount(msg.header.frame_id)
        except Exception:
            return
        raw = point_cloud2.read_points_numpy(msg, field_names=('x', 'y', 'z'), skip_nans=True)
        if raw.size == 0:
            self._depth_points[topic] = np.zeros((0, 2))
        else:
            raw = raw[::4]  # every 4th point: ample for a 5 cm obstacle at 3 m
            raw = raw[np.all(np.isfinite(raw), axis=1)]  # no-return pixels are inf
            # Gazebo depth clouds use the camera link axes (x forward, y left, z up).
            x, y, z = raw[:, 0], raw[:, 1], raw[:, 2] + mz
            c, s = math.cos(myaw), math.sin(myaw)
            px, py = mx + c * x - s * y, my + s * x + c * y
            loaded = self._lift > LOADED_LIFT_M
            keep = ((z >= MIN_OBSTACLE_Z) & (z <= (LOADED_MAX_Z if loaded else UNLOADED_MAX_Z)) &
                    (np.hypot(x, y) <= FIELD_RANGE_M))
            if loaded:
                keep &= ~self._own_cargo(px, py, z)
            self._depth_points[topic] = np.column_stack((px[keep], py[keep]))
        self._depth_at[topic] = self._now()
        self._depth_seen.add(topic)

    def _on_joints(self, msg):
        if 'lift_joint' in msg.name:
            value = msg.position[msg.name.index('lift_joint')]
            if math.isfinite(value):
                self._lift = value

    def _own_cargo(self, px, py, z):
        """Points on the attached cargo, from its estimated pose (robot frame)."""
        attached = next((c for c, state in self._cargo_state.items() if state == 'attached'), None)
        pose = self._cargo_pose.get(attached) if attached is not None else None
        try:
            robot = self._tf.lookup_transform('map', 'base_footprint', rclpy.time.Time())
        except Exception:
            robot = None
        if pose is None or robot is None:
            # Unknown placement: the cargo sits on the lift, centred.
            return (np.abs(px) <= CARGO_HALF_M) & (np.abs(py) <= CARGO_HALF_M) & (z >= 0.25)
        rx, ry = robot.transform.translation.x, robot.transform.translation.y
        ryaw = _yaw(robot.transform.rotation)
        dx, dy = pose.pose.position.x - rx, pose.pose.position.y - ry
        c, s = math.cos(-ryaw), math.sin(-ryaw)
        cx, cy = c * dx - s * dy, s * dx + c * dy
        cyaw = _yaw(pose.pose.orientation) - ryaw
        cc, cs = math.cos(-cyaw), math.sin(-cyaw)
        lx = cc * (px - cx) - cs * (py - cy)
        ly = cs * (px - cx) + cc * (py - cy)
        return (np.abs(lx) <= CARGO_HALF_M) & (np.abs(ly) <= CARGO_HALF_M) & (z >= 0.25)

    def _on_estop(self, name, msg):
        self._pressed[name] = bool(msg.data)
        if msg.data:
            self._timer.trigger_emergency(f'emergency stop {name} pressed')

    def _on_bumper(self, msg):
        """Chassis contact with anything outside the robot: the bumper is hit."""
        others = [c.collision2.name if 'autonomous_tugger' in c.collision1.name
                  else c.collision1.name for c in msg.contacts]
        others = [name for name in others if 'autonomous_tugger' not in name]
        if others:
            self._bumper_at = self._now()
            self._timer.trigger_emergency(f'bumper contact with {others[0]}')

    def _on_reset(self, _msg):
        bumper = self._fresh(self._bumper_at, self._now(), 0.5)
        active = [n for n, p in self._pressed.items() if p] + (['bumper'] if bumper else [])
        if active:
            self.get_logger().warning(f'safety reset refused: still active: {active}')
            return
        if self._timer.emergency is not None:
            self.get_logger().info(f'safety reset after: {self._timer.emergency}')
        self._timer.reset()

    def _on_motion(self, msg):
        self._motion = list(msg.data)

    def _on_owner(self, msg):
        self._owner = msg.data

    def _fresh(self, stamp, now, age):
        return stamp is not None and 0 <= now - stamp <= age

    def _tick(self):
        now = self._now()
        station = self._owner in STATION_OWNERS
        reason = ''
        stale = [t for t in SCANNERS if not self._fresh(self._scan_at[t], now, SCAN_MAX_AGE_S)]
        if stale:
            measured, reason = PROTECTIVE_STOP, f'scanner data stale: {", ".join(stale)}'
        elif self._motion is None or len(self._motion) < 4:
            measured, reason = PROTECTIVE_STOP, 'motion feedback missing'
        else:
            # Depth cameras add points when they report; a camera that has
            # reported once and then goes stale stops the robot.
            dead = [t for t in self._depth_seen
                    if not self._fresh(self._depth_at[t], now, DEPTH_MAX_AGE_S)]
            if dead:
                measured, reason = PROTECTIVE_STOP, f'depth camera stale: {", ".join(dead)}'
            else:
                sources = [(t, p) for t, p in self._scan_points.items() if p is not None]
                sources += [(t, p) for t, p in self._depth_points.items()
                            if p is not None and self._fresh(self._depth_at[t], now,
                                                             DEPTH_MAX_AGE_S)]
                clouds = [p for _, p in sources]
                points = np.vstack(clouds) if clouds else np.zeros((0, 2))
                # Only points that can lie in a field: off the body, within reach.
                near = np.hypot(points[:, 0], points[:, 1]) <= 3.5 if len(points) else []
                points = points[near] if len(points) else points
                body = ((np.abs(points[:, 0]) <= 0.46) & (np.abs(points[:, 1]) <= 0.31)
                        if len(points) else [])
                points = points[~body] if len(points) else points
                v, delta = self._motion[1], self._motion[3]
                turning = len(self._motion) >= 7 and self._motion[5] > 0.5
                yaw_rate = self._motion[6] if len(self._motion) >= 7 else 0.0
                curvature = 2.0 * math.tan(delta) / WHEELBASE_M
                stop_turn = (abs(yaw_rate) * 0.17 + yaw_rate * yaw_rate / (2 * YAW_ACCELERATION))
                crabbing = len(self._motion) >= 9 and (
                    abs(self._motion[7]) > 1e-6 or abs(self._motion[8]) > 1e-6)
                if crabbing:
                    measured, hit = classify_crab(points, self._motion[7])
                else:
                    measured, hit = classify_points(
                        points, v, curvature, station=station, turning=turning,
                        turn_radius=PIVOT_RADIUS_M + 0.15 + stop_turn * PIVOT_RADIUS_M)
                if hit is not None:
                    seen_by = next((t.strip('/').split('/')[0] for t, p in sources if len(p) and
                                    np.min(np.hypot(p[:, 0] - hit[0], p[:, 1] - hit[1])) < 1e-6),
                                   'sensor')
                    reason = (f'{measured.lower().replace("_", " ")}: object at '
                              f'({hit[0]:.2f}, {hit[1]:.2f}) m in the robot frame ({seen_by})')
        state = self._timer.update(measured, now)
        if state == EMERGENCY_STOP:
            reason = self._timer.emergency
        elif state == PROTECTIVE_STOP and measured != PROTECTIVE_STOP:
            reason = 'protective field clear; restarting after 2 s'
        limit = speed_limit(state, station)
        message = String()
        message.data = json.dumps({'state': state, 'limit': limit, 'reason': reason,
                                   'station': station, 'stamp': now})
        self._state_pub.publish(message)
        signal = String()
        signal.data = json.dumps({'light': 'BLUE' if station and state == CLEAR
                                  else LIGHTS[state], 'sound': SOUNDS[state]})
        self._signal_pub.publish(signal)
        if (state, reason) != self._last:
            self._last = (state, reason)
            if state != CLEAR:
                self.get_logger().info(f'{state}: {reason}')


def main(args=None):
    rclpy.init(args=args)
    node = SafetyMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
