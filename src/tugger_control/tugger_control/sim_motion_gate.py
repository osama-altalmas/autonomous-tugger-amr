"""Gazebo-only command arbitration and conservative loaded collision guard.

The 4WS controller listens only to this node's output in the simulation
launch. This node is experimental software, not a safety-rated device.
"""
import json
import math
import time
from collections import deque
from pathlib import Path

import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped, TwistStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float64MultiArray, String
from tf2_ros import Buffer, TransformListener

from tugger_control.cargo_policy import confirmed_cargo
from tugger_control.cargo_policy import CARGO_IDS
from tugger_control.loaded_motion_guard import choose_guarded_speed, decode_scene, scene_expected_counts
from tugger_control.motion_math import speed_profile
from tugger_control.precision_scene_clearance import (
    choose_precision_speed, precision_approach_speed_limit)
from tugger_control.warehouse_clearance import validate_modeled_geometry


# Stay under the 1.2 m/s loaded guard limit. Requesting exactly the limit lets
# normal speed overshoot fail the guard and stop the robot repeatedly.
LOADED_ROUTE_REQUEST_LIMIT = 1.15


def planar_quaternion(q):
    """Return yaw, rejecting invalid or materially non-planar orientations."""
    values = (q.x, q.y, q.z, q.w)
    if not all(math.isfinite(v) for v in values):
        raise ValueError('Invalid quaternion')
    norm = math.sqrt(sum(v * v for v in values))
    if abs(norm - 1.0) > 0.02:
        raise ValueError('Invalid quaternion norm')
    x, y, z, w = (v / norm for v in values)
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    if abs(roll) > 0.05 or abs(pitch) > 0.05:
        raise ValueError('Non-planar pose is outside the guard model')
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def scene_time(payload):
    """Return the collision scene's own timestamp, or None if absent."""
    try:
        stamp = json.loads(payload).get('stamp')
    except (TypeError, ValueError, AttributeError):
        return None
    return float(stamp) if isinstance(stamp, (int, float)) and math.isfinite(stamp) else None


def payload_at_time(history, stamp, latest, maximum_gap=0.25):
    """Return the recorded pose closest to ``stamp``, or ``latest`` if none is near."""
    if not history:
        return latest
    near_stamp, pose = min(history, key=lambda item: abs(item[0] - stamp))
    return pose if abs(near_stamp - stamp) <= maximum_gap else latest


def payload_for_robot_pose(robot_pose, robot_truth, history, latest):
    """Express cargo beside the estimated robot pose, keeping its true offset.

    The estimated robot pose can lag the simulator at route speed, which looks
    like cargo shift. The robot and cargo truth poses come from one simulator
    source, so their difference is the real attachment offset; the 4 cm limit
    then judges that offset. Without robot truth, the direct comparison applies.
    """
    if robot_truth is None or latest is None:
        return latest
    stamp, truth_x, truth_y = robot_truth
    cargo = payload_at_time(history, stamp, latest)
    return (robot_pose[0] + cargo[0] - truth_x, robot_pose[1] + cargo[1] - truth_y,
            cargo[2], cargo[3])


class SimMotionGate(Node):
    def __init__(self):
        super().__init__('tugger_sim_motion_gate')
        self._tf = Buffer()
        self._listener = TransformListener(self._tf, self)
        self._output = self.create_publisher(TwistStamped, '/tugger/guarded_cmd_vel', 10)
        self._ready = self.create_publisher(Bool, '/tugger/loaded_guard/ready', 10)
        self._dock_ready = self.create_publisher(String, '/tugger/precision_guard/ready', 10)
        self._status = self.create_publisher(String, '/tugger/loaded_guard/status', 10)
        self._requests = {'NAV': None, 'PRECISION': None, 'DOCK_NAV': None}
        self._request_at = {'NAV': None, 'PRECISION': None, 'DOCK_NAV': None}
        self._owner, self._owner_at = 'STOP', None
        self._scene, self._scene_at = None, None
        self._states = {key: None for key in CARGO_IDS}
        self._states_at = {key: None for key in CARGO_IDS}
        self._payloads = {key: None for key in CARGO_IDS}
        self._payloads_at = {key: None for key in CARGO_IDS}
        self._payload_history = {key: deque(maxlen=400) for key in CARGO_IDS}
        self._aligned_payloads = {}
        self._robot_truth = None
        self._lift, self._lift_at = None, None
        self._speed, self._speed_at = None, None
        self._wheel_speed, self._wheel_speed_at = None, None
        self._last_time = None
        self._last_reason = None
        self._model_valid = False
        self._expected_counts = None
        try:
            world = Path(get_package_share_directory('tugger_gazebo')) / 'worlds/tugger_warehouse.sdf'
            robot = Path(get_package_share_directory('tugger_description')) / 'urdf/tugger.urdf.xacro'
            validate_modeled_geometry(world, robot)
            self._expected_counts = scene_expected_counts(world)
            self._model_valid = True
        except Exception as exc:
            self.get_logger().error(f'Modeled robot geometry invalid: {exc}')
        self.create_subscription(String, '/tugger/motion_owner', self._on_owner, 10)
        self.create_subscription(String, '/tugger/collision_scene', self._on_scene, 10)
        for name, topic in (('NAV', '/tugger/nav_cmd_vel'),
                            ('DOCK_NAV', '/tugger/dock_nav_cmd_vel'),
                            ('PRECISION', '/tugger/precision_cmd_vel')):
            self.create_subscription(TwistStamped, topic,
                                     lambda msg, key=name: self._on_request(key, msg), 10)
        for station in CARGO_IDS:
            self.create_subscription(String, f'/cargo/station_{station}/state',
                                     lambda msg, key=station: self._on_state(key, msg), 10)
            self.create_subscription(PoseStamped, f'/tugger/cargo/station_{station}/ground_truth_pose',
                                     lambda msg, key=station: self._on_payload(key, msg), 10)
        self.create_subscription(PoseStamped, '/tugger/ground_truth_pose',
                                 self._on_robot_truth, 10)
        self.create_subscription(JointState, '/joint_states', self._on_joint, 10)
        self.create_subscription(Float64MultiArray, '/four_ws/motion_state', self._on_motion, 10)
        self.create_subscription(Odometry, '/wheel/odom', self._on_wheel_odom, 10)
        self.create_timer(0.05, self._tick)

    def _now(self):
        return self.get_clock().now().nanoseconds / 1e9

    @staticmethod
    def _fresh(stamp, now, maximum):
        return stamp is not None and 0 <= now - stamp <= maximum

    def _on_owner(self, msg):
        owner = msg.data if msg.data in ('NAV', 'PRECISION', 'DOCK_NAV', 'DOCK_CAMERA', 'DOCK_DEPART') else 'STOP'
        source = {'DOCK_CAMERA': 'PRECISION', 'DOCK_DEPART': 'DOCK_NAV'}.get(owner, owner)
        if owner != self._owner and source in self._request_at:
            self._request_at[source] = None
        self._owner = owner
        self._owner_at = self._now()

    def _on_request(self, source, msg):
        self._requests[source] = msg
        self._request_at[source] = self._now()

    def _on_scene(self, msg):
        self._scene, self._scene_at = msg.data, self._now()

    def _on_state(self, station, msg):
        state = msg.data.strip().lower()
        if state in ('attached', 'detached'):
            self._states[station], self._states_at[station] = state, self._now()

    def _on_payload(self, station, msg):
        if msg.header.frame_id != 'map':
            return
        try:
            p = msg.pose.position
            pose = (p.x, p.y, p.z, planar_quaternion(msg.pose.orientation))
            if not all(math.isfinite(v) for v in pose):
                return
            self._payloads[station], self._payloads_at[station] = pose, self._now()
            stamp = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
            self._payload_history[station].append((stamp, pose))
        except ValueError:
            return

    def _on_robot_truth(self, msg):
        if msg.header.frame_id != 'map':
            return
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
        self._robot_truth = (stamp, msg.pose.position.x, msg.pose.position.y)

    def _on_joint(self, msg):
        try:
            value = float(msg.position[msg.name.index('lift_joint')])
            if math.isfinite(value):
                self._lift, self._lift_at = value, self._now()
        except (IndexError, ValueError, TypeError):
            pass

    def _on_motion(self, msg):
        if len(msg.data) >= 2 and math.isfinite(msg.data[1]):
            self._speed, self._speed_at = float(msg.data[1]), self._now()

    def _on_wheel_odom(self, msg):
        if msg.header.frame_id != 'odom' or msg.child_frame_id != 'base_footprint':
            return
        speed = float(msg.twist.twist.linear.x)
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
        if math.isfinite(speed) and self._fresh(stamp, self._now(), 0.3):
            self._wheel_speed, self._wheel_speed_at = speed, stamp

    def _snapshot(self, now):
        if not self._model_valid:
            raise ValueError('robot geometry not validated')
        if self.count_subscribers('/tugger/guarded_cmd_vel') < 1:
            raise ValueError('guarded controller subscriber missing')
        if self.count_publishers('/tugger/guarded_cmd_vel') != 1:
            raise ValueError('unexpected guarded command publisher')
        if not self._fresh(self._scene_at, now, 0.4):
            raise ValueError('scene feed stale')
        obstacles = decode_scene(self._scene, now, expected_counts=self._expected_counts)
        if (not self._fresh(self._lift_at, now, 1.0) or
                not self._fresh(self._speed_at, now, 0.3) or
                not self._fresh(self._wheel_speed_at, now, 0.3)):
            raise ValueError('lift or velocity feedback stale')
        if not all(self._fresh(self._payloads_at[key], now, 1.0) for key in CARGO_IDS):
            raise ValueError('cargo pose feedback stale')
        scene_stamp = scene_time(self._scene)
        for key in CARGO_IDS:
            model = f'cargo_station_{key}_stand'
            matches = [box for owner, box in obstacles if owner == model]
            if len(matches) != 1:
                raise ValueError('cargo collision geometry is ambiguous')
            # Compare with the cargo pose from the scene's own instant; at
            # route speed a few milliseconds apart already exceeds 4 cm.
            payload = (self._payloads[key] if scene_stamp is None else payload_at_time(
                getattr(self, '_payload_history', {}).get(key, ()), scene_stamp,
                self._payloads[key]))
            box = matches[0]
            if (math.hypot(box.x - payload[0], box.y - payload[1]) > 0.04 or
                    abs((box.z_min + box.z_max) / 2 - (payload[2] + 0.295)) > 0.025 or
                    min(box.half_x, box.half_y) < 0.64):
                raise ValueError('cargo pose disagrees with simulator collision scene')
        transform = self._tf.lookup_transform('map', 'base_footprint', rclpy.time.Time())
        stamp = transform.header.stamp.sec + transform.header.stamp.nanosec / 1e9
        if not self._fresh(stamp, now, 0.5):
            raise ValueError('robot transform stale')
        t = transform.transform.translation
        if not all(math.isfinite(v) for v in (t.x, t.y, t.z)) or abs(t.z) > 0.03:
            raise ValueError('robot height outside modeled ground plane')
        pose = (t.x, t.y, planar_quaternion(transform.transform.rotation))
        self._aligned_payloads = {
            key: payload_for_robot_pose(
                pose, self._robot_truth, self._payload_history[key], self._payloads[key])
            for key in CARGO_IDS}
        return pose, obstacles

    def _tick(self):
        started_wall = time.monotonic()
        now = self._now()
        if self._last_time is not None and now < self._last_time:
            self._owner_at = None
            self._request_at = {'NAV': None, 'PRECISION': None, 'DOCK_NAV': None}
            self._states_at = {key: None for key in CARGO_IDS}
            self._scene_at = None
            self._payloads_at = {key: None for key in CARGO_IDS}
            self._lift_at = self._speed_at = None
            self._wheel_speed_at = None
        self._last_time = now
        output = TwistStamped()
        output.header.stamp = self.get_clock().now().to_msg()
        output.header.frame_id = 'base_footprint'
        reason, ready, snapshot, dock_ready, owner = 'stopped', False, None, False, 'STOP'
        try:
            ages = {key: now - self._states_at[key] if self._states_at[key] is not None else None
                    for key in CARGO_IDS}
            cargo = confirmed_cargo(self._states, ages)
            try:
                snapshot = self._snapshot(now)
                ready = True
            except Exception as exc:
                reason = str(exc)
            owner = self._owner if self._fresh(self._owner_at, now, 0.3) else 'STOP'
            source = {'DOCK_CAMERA': 'PRECISION', 'DOCK_DEPART': 'DOCK_NAV'}.get(owner, owner)
            dock_ready = (
                (owner in ('DOCK_NAV', 'DOCK_CAMERA') and cargo is None and
                 snapshot is not None and -0.005 <= self._lift <= 0.005) or
                (owner == 'DOCK_DEPART' and cargo is not None and
                 snapshot is not None and 0.060 <= self._lift <= 0.070))
            if owner != 'STOP' and self._fresh(self._request_at[source], now, 0.25):
                request = self._requests[source]
                v, omega = float(request.twist.linear.x), float(request.twist.angular.z)
                if not math.isfinite(v) or not math.isfinite(omega):
                    raise ValueError('nonfinite command')
                if owner in ('DOCK_NAV', 'DOCK_CAMERA', 'PRECISION'):
                    if cargo is not None:
                        raise ValueError('unloaded precision guard cannot move attached cargo')
                    if snapshot is None:
                        raise ValueError(reason)
                    # The independent docking controller may approach at 0.4
                    # m/s in open space. Taper before fixed cargo legs; final
                    # camera corrections, including HOME, are capped at .05.
                    approach_limit = (precision_approach_speed_limit(
                        snapshot[0], snapshot[1]) if owner == 'DOCK_NAV'
                        else 0.05)
                    requested_abs = abs(v)
                    limited, hit = choose_precision_speed(
                        snapshot[0], self._lift, [box for _, box in snapshot[1]],
                        max(abs(self._speed), abs(self._wheel_speed)), v,
                        max_speed=approach_limit,
                        minimum_speed=0.10 if owner == 'DOCK_NAV' else 0.0)
                    if hit is not None:
                        raise ValueError(f'precision stopping envelope intersects {hit}')
                    omega = omega * limited / v if abs(v) > 1e-6 else 0.0
                    omega_limit = 2.0 * abs(limited) / 0.58
                    omega = max(-omega_limit, min(omega_limit, omega))
                    v = limited
                    reason = ('precision speed reduced' if
                              abs(limited) + 1e-6 < min(requested_abs, approach_limit)
                              else 'precision guarded')
                if owner == 'DOCK_DEPART' and cargo is None:
                    raise ValueError('loaded departure requires confirmed attached cargo')
                if cargo is not None:
                    if snapshot is None:
                        raise ValueError(reason)
                    if owner == 'NAV':
                        # Loaded route travel uses the same straight/curve
                        # profile as unloaded travel before clearance checks.
                        v, omega = speed_profile(
                            v, omega, straight_limit=LOADED_ROUTE_REQUEST_LIMIT)
                    # Departure shares the independent docking controller but
                    # has its own ownership mode and a stricter speed ceiling.
                    requested = (max(-0.5, min(0.5, v))
                                 if owner == 'DOCK_DEPART' else v)
                    limited, hit = choose_guarded_speed(
                        snapshot[0],
                        getattr(self, '_aligned_payloads', {}).get(cargo, self._payloads[cargo]),
                        self._lift, cargo,
                        snapshot[1], max(abs(self._speed), abs(self._wheel_speed)), requested,
                        max_speed=0.5 if owner == 'DOCK_DEPART' else 1.2)
                    if hit is not None:
                        raise ValueError(f'loaded stopping envelope intersects {hit}')
                    if abs(limited) + 1e-6 < min(abs(v), 0.1):
                        reason = 'loaded clearance reducing speed'
                    else:
                        reason = 'ready'
                    omega = omega * limited / v if abs(v) > 1e-6 else 0.0
                    v = limited
                    if owner == 'DOCK_DEPART':
                        # 45-degree counter-phase 4WS steering at 0.58 m wheelbase.
                        omega_limit = 2.0 * abs(v) / 0.58
                        omega = max(-omega_limit, min(omega_limit, omega))
                if cargo is None and owner == 'NAV':
                    v, omega = speed_profile(v, omega)
                output.twist.linear.x = v
                output.twist.angular.z = omega
                if cargo is None and owner not in ('DOCK_NAV', 'DOCK_CAMERA', 'PRECISION'):
                    reason = 'unloaded motion'
        except Exception as exc:
            dock_ready = False
            reason = str(exc)
        if time.monotonic() - started_wall > 0.15:
            output.twist.linear.x = output.twist.angular.z = 0.0
            ready, dock_ready, reason = False, False, 'guard computation exceeded 150 ms'
        self._output.publish(output)
        state = Bool()
        state.data = ready
        self._ready.publish(state)
        dock_state = String()
        dock_state.data = owner if dock_ready else 'STOP'
        self._dock_ready.publish(dock_state)
        if reason != self._last_reason:
            self._last_reason = reason
            message = String()
            message.data = reason
            self._status.publish(message)


def main(args=None):
    rclpy.init(args=args)
    node = SimMotionGate()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
