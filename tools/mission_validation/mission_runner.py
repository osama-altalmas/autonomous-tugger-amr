#!/usr/bin/env python3
"""Run real missions through OperatorBridge without Qt and record the outcome.

The same mission code, motion gate and controllers as the operator panel are
used; this script only presses the equivalent of Check/plan and RUN. It writes
one JSON result per mission so reliability can be measured across many runs.
"""
import argparse
import collections
import json
import math
import os
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import rclpy
from geometry_msgs.msg import PoseStamped, TwistStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from std_msgs.msg import Float64MultiArray, String

from tugger_control.operator_panel import OperatorBridge, OperatorPanel
from tugger_control.cargo_policy import DROPOFF_CARGO, PICKUP_CARGO

STATION_CENTERS = {
    'cargo_station_0': (-17.5, -7.0), 'cargo_dropoff_northwest': (-17.5, 7.0),
    'cargo_station_3': (17.5, 7.0), 'cargo_dropoff_southeast': (17.5, -7.0),
    'HOME': (0.0, 0.0),
}
FULL_LOCK_RAD = math.radians(40.0)
STEER_REVERSAL_RAD = math.radians(2.0)   # smaller swings are noise, not weaving
YAW_FLIP_RAD_S = 0.05


def _curvature(msg):
    v = msg.twist.linear.x
    return msg.twist.angular.z / v if abs(v) > 0.05 else 0.0


class Recorder(Node):
    """Observe commands, steering, ownership and true pose; publishes nothing."""

    def __init__(self):
        super().__init__('tugger_mission_recorder')
        self.set_parameters([Parameter('use_sim_time', value=True)])
        self.owner = 'STOP'
        self.pose = None
        self.pose_lookup = None
        # Estimate at a given stamp (map -> base_footprint at that time), to
        # compare it with the true pose of the same instant; comparing the
        # newest of each mixed in ~20 ms of age, which at 1 m/s reads as 2 cm.
        self.stamped_pose_lookup = None
        self._truth_history = collections.deque(maxlen=200)
        self.path_lookup = None     # the path being followed (nav_msgs/Path) or None
        self.reset()
        self.create_subscription(String, '/tugger/motion_owner', self._owner, 10)
        self.create_subscription(TwistStamped, '/tugger/dock_nav_cmd_vel', self._dock_cmd, 10)
        self.create_subscription(TwistStamped, '/tugger/guarded_cmd_vel', self._guarded, 10)
        self.create_subscription(TwistStamped, '/tugger/nav_cmd_vel', self._nav_cmd, 10)
        # Route follower report: lateral, heading error, path curvature,
        # commanded curvature, pure pursuit curvature, look-ahead, planned
        # speed limit, commanded speed, path ahead (curvature tracking plugin).
        self.create_subscription(Float64MultiArray, '/tugger/path_tracking', self._tracking, 10)
        self.create_subscription(Float64MultiArray, '/four_ws/motion_state', self._motion, 10)
        self.create_subscription(PoseStamped, '/tugger/ground_truth_pose', self._pose,
                                 qos_profile_sensor_data)
        self.create_subscription(String, '/tugger/loaded_guard/status', self._guard, 10)
        self.create_subscription(Odometry, '/wheel/odom', self._odom, 10)
        # Refined localization: how often its scan match is used, and why not.
        self.create_subscription(String, '/tugger/localization/refinement',
                                 self._refinement, 10)
        self.map_received = False
        self.create_subscription(
            OccupancyGrid, '/map', self._map,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))

    def reset(self):
        self.phases = {}
        self.guard = {}
        self.refinement = {}
        self.guard_events = []
        # Diagnostic trace at 2 Hz: sim time, owner, true pose, estimated
        # pose, wheel speed, wheel odometry pose. Used to explain stalls and alignment failures.
        self.trace = []
        self._trace_t = None
        self._dock_sign = 0
        self._speed = 0.0
        # Route following at 10 Hz: sim time, signed cross-track error (left
        # of the path positive), steering angle, commanded yaw rate, speed,
        # target steering, Nav2 and gate curvature, estimated cross-track
        # error, latest route follower report (or None).
        self.track = []
        self._track_t = None
        self._steer = self._omega = self._target_steer = 0.0
        self._nav_k = self._gate_k = 0.0
        self._tracking_report = None

    def _phase(self):
        return self.phases.setdefault(self.owner, {
            'dock_cmd_sign_flips': 0, 'guarded_samples': 0, 'full_lock_samples': 0,
            'steer_samples': 0, 'first_t': None, 'last_t': None, 'first_pose': None,
            'path_m': 0.0, '_last': None, 'max_abs_speed': 0.0,
            'max_estimate_error_m': 0.0, 'slow_s': 0.0, '_slow_t': None,
            'max_cross_track_m': 0.0,
            # Path-following quality ("weaving"), only while moving.
            'xte_sum_sq': 0.0, 'xte_samples': 0, 'moving_m': 0.0,
            'steer_reversals': 0, '_steer_peak': None, '_steer_dir': 0,
            'yaw_rate_flips': 0, '_yaw_sign': 0, '_path_id': None, '_path_xy': None,
            '_path_i': 0})

    def _now(self):
        return self.get_clock().now().nanoseconds / 1e9

    def _owner(self, msg):
        self.owner = msg.data

    def _dock_cmd(self, msg):
        value = msg.twist.linear.x
        sign = (value > 1e-3) - (value < -1e-3)
        if sign and self._dock_sign and sign != self._dock_sign:
            self._phase()['dock_cmd_sign_flips'] += 1
        if sign:
            self._dock_sign = sign

    def _guarded(self, msg):
        self._phase()['guarded_samples'] += 1
        self._gate_k = _curvature(msg)

    def _tracking(self, msg):
        self._tracking_report = [round(v, 4) for v in msg.data]

    def _nav_cmd(self, msg):
        self._nav_k = _curvature(msg)

    def _motion(self, msg):
        if len(msg.data) < 4:
            return
        phase = self._phase()
        phase['steer_samples'] += 1
        phase['full_lock_samples'] += abs(msg.data[3]) >= FULL_LOCK_RAD
        if len(msg.data) < 7 or abs(msg.data[1]) < 0.05:
            return
        # A reversal: the steering turns back by at least STEER_REVERSAL_RAD
        # from its last peak. Smooth curves give few; weaving gives many.
        delta = msg.data[3]
        self._steer, self._omega, self._target_steer = delta, msg.data[4], msg.data[2]
        peak, direction = phase['_steer_peak'], phase['_steer_dir']
        if peak is None:
            phase['_steer_peak'] = delta
        elif direction == 0:
            if abs(delta - peak) >= STEER_REVERSAL_RAD:
                phase['_steer_dir'], phase['_steer_peak'] = (1 if delta > peak else -1), delta
        elif direction * (delta - peak) > 0:
            phase['_steer_peak'] = delta                 # still turning the same way
        elif abs(delta - peak) >= STEER_REVERSAL_RAD:
            phase['steer_reversals'] += 1
            phase['_steer_dir'], phase['_steer_peak'] = -direction, delta
        yaw_rate = msg.data[4]
        sign = (yaw_rate > YAW_FLIP_RAD_S) - (yaw_rate < -YAW_FLIP_RAD_S)
        if sign and phase['_yaw_sign'] and sign != phase['_yaw_sign']:
            phase['yaw_rate_flips'] += 1
        if sign:
            phase['_yaw_sign'] = sign

    def _pose(self, msg):
        q = msg.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        self.pose = (msg.pose.position.x, msg.pose.position.y, yaw)
        self._truth_history.append((msg.header.stamp, self.pose))
        phase = self._phase()
        now = self._now()
        phase['first_t'] = now if phase['first_t'] is None else phase['first_t']
        phase['last_t'] = now
        phase['first_pose'] = phase['first_pose'] or self.pose
        if phase['_last'] is not None:
            step = math.hypot(self.pose[0] - phase['_last'][0], self.pose[1] - phase['_last'][1])
            phase['path_m'] += step
            if abs(self._speed) >= 0.05:
                phase['moving_m'] += step
                self._cross_track(phase)
        phase['_last'] = self.pose
        if phase['_slow_t'] is not None and abs(self._speed) < 0.3:
            phase['slow_s'] += max(0.0, now - phase['_slow_t'])
        phase['_slow_t'] = now
        estimate = None
        if self.pose_lookup is not None:
            try:
                estimate = self.pose_lookup()
                phase['max_estimate_error_m'] = max(
                    phase['max_estimate_error_m'],
                    math.hypot(estimate[0] - self.pose[0], estimate[1] - self.pose[1]))
            except Exception:
                estimate = None
        if self._trace_t is None or now - self._trace_t >= 0.5:
            self._trace_t = now
            self.trace.append([round(now, 2), self.owner,
                               *(round(v, 4) for v in self.pose),
                               *((round(v, 4) for v in estimate[:3]) if estimate else
                                 (None, None, None)),
                               round(self._speed, 3),
                               # Raw wheel odometry pose (odom frame), to measure
                               # drift through corners against the true pose. It is
                               # the latest message, about 50 ms behind the true pose
                               # beside it: shift it by that before comparing, or every
                               # 90 degree corner shows about 5 cm of false drift.
                               *((round(v, 4) for v in self._odom_pose)
                                 if getattr(self, '_odom_pose', None) else (None, None, None)),
                               # True pose ~0.2 s ago and the estimate AT ITS STAMP
                               # (same instant; TF covers it by then): indices 12-17.
                               *self._same_instant(now)])

    def _same_instant(self, now):
        empty = (None,) * 6
        if self.stamped_pose_lookup is None or not self._truth_history:
            return empty
        target = now - 0.2
        stamp, truth = min(self._truth_history, key=lambda item: abs(
            item[0].sec + item[0].nanosec * 1e-9 - target))
        try:
            estimate = self.stamped_pose_lookup(stamp)
        except Exception:
            return empty
        return (*(round(v, 4) for v in truth), *(round(v, 4) for v in estimate[:3]))

    def _refinement(self, msg):
        try:
            status = json.loads(msg.data)
        except ValueError:
            return
        key = status.get('used', '?') + ': ' + status.get('reason', '?')
        self.refinement[key] = self.refinement.get(key, 0) + 1

    def _cross_track(self, phase):
        """True distance from the path being followed (not the estimate)."""
        path = self.path_lookup() if self.path_lookup is not None else None
        if path is None or len(path.poses) < 2:
            return
        if phase['_path_id'] != id(path):
            phase['_path_id'] = id(path)
            phase['_path_xy'] = [(p.pose.position.x, p.pose.position.y) for p in path.poses]
            phase['_path_i'] = 0
        xy, best, best_i = phase['_path_xy'], None, phase['_path_i']
        start = max(phase['_path_i'] - 20, 0)
        for i in range(start, min(start + 200, len(xy) - 1)):
            (ax, ay), (bx, by) = xy[i], xy[i + 1]
            dx, dy = bx - ax, by - ay
            length = dx * dx + dy * dy
            t = 0.0 if length == 0 else max(0.0, min(1.0, ((self.pose[0] - ax) * dx +
                                                          (self.pose[1] - ay) * dy) / length))
            d = math.hypot(self.pose[0] - ax - t * dx, self.pose[1] - ay - t * dy)
            if best is None or d < best:
                best, best_i = d, i
        if best is None:
            return
        (ax, ay), (bx, by) = xy[best_i], xy[best_i + 1]
        side = (bx - ax) * (self.pose[1] - ay) - (by - ay) * (self.pose[0] - ax)
        segment = math.hypot(bx - ax, by - ay)
        if segment > 1e-9:
            # Sideways distance from the segment's line. When a new path
            # starts a few centimetres ahead of the robot, the straight-line
            # distance to its first point is an along-track gap (it read as
            # 8-14 cm "cross-track" peaks at every hand-over, 2026-11-12).
            best = abs(side) / segment
        now = self._now()
        if self.owner == 'NAV' and (self._track_t is None or now - self._track_t >= 0.1):
            self._track_t = now
            estimated = None
            if self.pose_lookup is not None:
                try:
                    ex, ey = self.pose_lookup()[:2]
                    dx, dy = bx - ax, by - ay
                    length = math.hypot(dx, dy) or 1.0
                    estimated = round(((ex - ax) * dy * -1 + (ey - ay) * dx) / length, 4)
                except Exception:
                    estimated = None
            self.track.append([round(now, 2), round(best if side >= 0 else -best, 4),
                               round(self._steer, 4), round(self._omega, 4), round(self._speed, 3),
                               round(self._target_steer, 4), round(self._nav_k, 4), round(self._gate_k, 4),
                               estimated, self._tracking_report])
        phase['_path_i'] = best_i
        phase['xte_sum_sq'] += best * best
        phase['xte_samples'] += 1
        phase['max_cross_track_m'] = max(phase['max_cross_track_m'], best)

    def _guard(self, msg):
        self.guard[msg.data] = self.guard.get(msg.data, 0) + 1
        self.guard_events.append({'t': self._now(), 'reason': msg.data,
                                  'wheel_speed': self._speed})

    def _map(self, _msg):
        self.map_received = True

    def _odom(self, msg):
        self._speed = float(msg.twist.twist.linear.x)
        q = msg.pose.pose.orientation
        self._odom_pose = (msg.pose.pose.position.x, msg.pose.pose.position.y,
                           math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)))
        phase = self._phase()
        phase['max_abs_speed'] = max(phase['max_abs_speed'], abs(self._speed))

    def summary(self):
        result = {}
        for name, data in self.phases.items():
            result[name] = {key: value for key, value in data.items()
                            if not key.startswith('_')}
            samples = max(data['steer_samples'], 1)
            result[name]['full_lock_fraction'] = data['full_lock_samples'] / samples
            per_10m = 10.0 / max(data['moving_m'], 1.0)
            result[name]['rms_cross_track_m'] = (
                math.sqrt(data['xte_sum_sq'] / data['xte_samples']) if data['xte_samples'] else None)
            result[name]['steer_reversals_per_10m'] = data['steer_reversals'] * per_10m
            result[name]['yaw_rate_flips_per_10m'] = data['yaw_rate_flips'] * per_10m
        return result


class Runner:
    def __init__(self, timeout, trial_routes=(), trial_candidates=()):
        self.timeout = timeout
        self.trial_routes = set(trial_routes)
        self.trial_candidates = set(trial_candidates)
        self.force_live = False
        self.disturb = None          # (dx, dy, dyaw): shove once, see _maybe_disturb
        self.disturb_at = None       # only during the mission to this destination
        self.disturb_owner = 'CAMERA'
        self.disturb_delay = 0.0     # seconds after that owner takes over
        self.current_destination = None
        self.confirm_warnings = False
        self._owner_since = (None, 0.0)
        self._disturbed = False
        self.reports = []
        self.finished = None
        self.planned = False
        panel = SimpleNamespace()
        panel._tools_directory = lambda: OperatorPanel._tools_directory(panel)
        panel._routes = OperatorPanel._load_routes(panel)
        self.routes = panel._routes
        docking = OperatorPanel._load_docking_routes(panel)
        self.mission_paths = OperatorPanel._load_mission_paths(panel)
        live_planner = OperatorPanel._make_live_planner(panel)
        self.entries = OperatorPanel._load_station_entries(panel)
        def approach(station, entry_pose=None):
            metadata, poses = docking.approach(station, entry_pose)
            if f"{station}__{metadata.get('entrance')}" in self.trial_candidates:
                # In memory only, for this validation run. Files stay unapproved.
                metadata = dict(metadata, execution_approved=True)
            return metadata, poses
        self.bridge = OperatorBridge(self._report, self._plan, self._done,
                                     approach_provider=approach,
                                     station_layout=panel._routes.database,
                                     live_planner=live_planner)
        self.bridge._candidate_provider = lambda destination: (
            self._with_trials(self.mission_paths.paths_to(destination)),
            self.entries.get(destination, []))
        self.recorder = Recorder()
        self.recorder.pose_lookup = self.bridge._robot_pose
        def _estimate_at(stamp, tf=self.bridge._tf):
            t = tf.lookup_transform('map', 'base_footprint', rclpy.time.Time.from_msg(stamp))
            q = t.transform.rotation
            return (t.transform.translation.x, t.transform.translation.y,
                    math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)))
        self.recorder.stamped_pose_lookup = _estimate_at
        self.recorder.path_lookup = lambda: (getattr(self.bridge, '_followed_path', None)
                                             if self.bridge._active_follow is not None else None)
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.bridge)
        self.executor.add_node(self.recorder)

    def _report(self, message):
        self.reports.append(message)
        print(f'[panel] {message}', flush=True)

    def _plan(self, message, executing):
        self.planned = True
        self._report(f'{message} (executing={executing})')

    def _done(self, succeeded, status):
        self.finished = (bool(succeeded), int(status))

    def _busy(self):
        if getattr(self.bridge, '_obstacle_hold', None) is not None:
            return True     # waiting for an object is the robot working, not idle
        bridge = self.bridge
        return (bridge._follow_goal_pending or bridge._stage_validation_pending or
                bridge._departure_pending is not None or bridge._departure_active or
                bridge._active_follow is not None or bridge._qr_alignment_active or
                bridge._cargo_task is not None)

    def _maybe_disturb(self):
        """Test only: shove the robot once in a chosen stage to force a failure.

        Attached cargo moves with the robot by the same rigid motion, as in a
        real knock. A shove that would put the robot inside a modelled
        obstacle is refused: the test must not create an impossible state.
        """
        owner = self.recorder.owner
        if owner != self._owner_since[0]:
            self._owner_since = (owner, time.monotonic())
        # 'CAMERA' means the camera stage itself, whatever the motion owner
        # (a loaded robot aligns under PRECISION, an empty one under DOCK_CAMERA).
        in_stage = (self.bridge._qr_alignment_active if self.disturb_owner == 'CAMERA'
                    else owner == self.disturb_owner)
        if (self.disturb is None or self._disturbed or self.recorder.pose is None or
                not in_stage or
                (self.disturb_at is not None and self.current_destination != self.disturb_at) or
                time.monotonic() - self._owner_since[1] < self.disturb_delay):
            return
        self._disturbed = True
        x, y, yaw = self.recorder.pose
        dx, dy = self.disturb[0], self.disturb[1]
        dyaw = self.disturb[2] if len(self.disturb) > 2 else 0.0
        target = (x + dx, y + dy, yaw + dyaw)
        try:
            cargo = self.bridge._confirmed_cargo()
        except ValueError:
            cargo = None
        if cargo is None and not self._pose_is_clear(target):
            print(f'disturbance {self.disturb} refused: target pose is not clear', flush=True)
            return
        print(f'disturbing robot by {self.disturb} during {owner} '
              f'to {self.current_destination}{" with cargo" if cargo is not None else ""}',
              flush=True)
        self._set_pose(*target)
        if cargo is not None:
            cx, cy, cz, cyaw = self.bridge._cargo_poses[cargo]
            c, s = math.cos(dyaw), math.sin(dyaw)
            rx, ry = cx - x, cy - y
            self._set_pose(x + c * rx - s * ry + dx, y + s * rx + c * ry + dy, cyaw + dyaw,
                           model=f'cargo_station_{cargo}_stand', z=cz)

    def _pose_is_clear(self, pose):
        from tugger_control.precision_scene_clearance import sampled_precision_clearance
        from tugger_control.warehouse_clearance import static_world_boxes
        from tugger_control.lift_geometry import LIFT_DOWN_M
        world = Path(__file__).resolve().parents[2] / 'src/tugger_gazebo/worlds/tugger_warehouse.sdf'
        return sampled_precision_clearance([pose, pose], static_world_boxes(world),
                                           LIFT_DOWN_M)[0]

    def spin_until(self, condition, seconds):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self.executor.spin_once(timeout_sec=0.02)
            self._maybe_disturb()
            if condition():
                return True
        return False

    def wait_ready(self, seconds=180):
        def ready():
            try:
                self.bridge._robot_pose()
            except Exception:
                return False
            return (self.bridge._follower.server_is_ready() and
                    self.bridge._dock_follower.server_is_ready() and
                    self.bridge._loaded_guard_is_ready() and
                    self.bridge._path_validator.service_is_ready() and
                    self.recorder.pose is not None and
                    self.recorder.map_received)
        return self.spin_until(ready, seconds)

    def cancel_after(self, destination, seconds, moved=None):
        """Start a mission, cancel it part-way, and report where it stopped.

        With moved, cancel once the robot is that far from where RUN was
        pressed (seconds is then only a time limit).
        """
        destination = destination.partition(':')[0]
        self.finished, self.planned = None, False
        self.bridge.execute_destination(
            destination, self.mission_paths.paths_to(destination),
            self.entries.get(destination, []))
        self.spin_until(lambda: self.planned, 60)
        self.bridge._motion_armed = True
        start = self.recorder.pose
        self.bridge.run_selected_mission_with_retry(True)
        if moved is None:
            self.spin_until(lambda: False, seconds)
        else:
            self.spin_until(lambda: self.recorder.pose is not None and math.hypot(
                self.recorder.pose[0] - start[0], self.recorder.pose[1] - start[1]) >= moved,
                seconds)
        self.bridge.cancel_navigation()
        self.spin_until(lambda: False, 5.0)
        self.bridge._cancel_requested = False
        self.bridge._motion_armed = True
        return {'destination': destination, 'outcome': 'CANCELLED_ON_PURPOSE',
                'final_pose': self.recorder.pose, 'center_error_m': 0.0, 'wall_s': seconds,
                'phases': {}, 'reports': [], 'guard_events': [], 'cargo': None,
                'lift': self.bridge._lift_position, 'docked_station': None,
                'guard_reasons': {}}

    def _set_pose(self, x, y, yaw, model='autonomous_tugger', z=0.0):
        assert os.environ.get('GZ_PARTITION', '').startswith('tugger_validation_')
        request = (f'name: "{model}" position {{x: {x} y: {y} z: {z}}} '
                   f'orientation {{z: {math.sin(yaw / 2)} w: {math.cos(yaw / 2)}}}')
        subprocess.run(['gz', 'service', '-s', '/world/tugger_test/set_pose',
                        '--reqtype', 'gz.msgs.Pose', '--reptype', 'gz.msgs.Boolean',
                        '--timeout', '5000', '--req', request], check=True,
                       capture_output=True)

    def teleport(self, x, y, yaw):
        self._set_pose(x, y, yaw)
        self.spin_until(lambda: False, 20.0)
        self.bridge.docked_station = None

    def _with_trials(self, candidates):
        """Approve --trial-route paths in memory only, wherever paths are read."""
        for key, route, _poses in candidates:
            if key in self.trial_routes or key.replace(' (reversed)', '') in self.trial_routes:
                route['execution_approved'] = True
                route['_database_execution_approved'] = True
        return candidates

    def in_place(self, destination, action):
        """ATTACH/DETACH at the station the robot is docked at, as an operator would."""
        self.current_destination = destination
        self.finished = None
        self.recorder.reset()
        wall = time.monotonic()
        self.bridge._motion_armed = True
        self.bridge.select_cargo_action(action)
        self.spin_until(lambda: self.finished is not None, 300)
        outcome = ('SUCCESS' if self.finished and self.finished[0] else
                   'NOT_STARTED' if self.finished is None else 'FAILED')
        print(f'{destination}:{action}-in-place {outcome}', flush=True)
        return {'destination': f'{destination}:{action} (in place)', 'outcome': outcome,
                'wall_s': time.monotonic() - wall, 'phases': {}, 'reports': [],
                'guard_events': [], 'cargo': None, 'final_pose': self.recorder.pose,
                'center_error_m': 0.0}

    def mission(self, destination):
        # "station:attach", "station:detach" or "station:none"; without a
        # suffix the standard choice is used (attach at a pickup station,
        # detach at a drop-off, nothing at HOME), as an operator would pick.
        destination, _, action = destination.partition(':')
        if action in ('attach', 'detach') and self.bridge.docked_station == destination:
            return self.in_place(destination, action)
        if not action:
            action = ('attach' if destination in PICKUP_CARGO else
                      'detach' if destination in DROPOFF_CARGO else 'none')
        self.current_destination = destination
        self.finished, self.planned = None, False
        self.recorder.reset()
        first_report = len(self.reports)
        wall = time.monotonic()
        result = {'destination': destination, 'outcome': 'NOT_RUN'}
        # Test only: hide preplanned paths so the live planner must run.
        candidates = [] if self.force_live else self._with_trials(
            self.mission_paths.paths_to(destination))
        self.bridge.start_mission(
            destination, candidates, self.entries.get(destination, []))
        if not self.spin_until(lambda: self.planned, 180):
            result['outcome'] = 'NO_PLAN'
        else:
            self.bridge._motion_armed = True
            if action != 'none':
                self.bridge.select_cargo_action(action)
            deadline = time.monotonic() + self.timeout
            # A loaded departure asks for a new RUN at staging, as for an operator.
            # Press RUN once, like an operator; the panel continues by itself
            # after undocking. Press again only if nothing is running.
            first = True
            while self.finished is None and time.monotonic() < deadline:
                self.planned = False
                if first or not self._busy():
                    confirm = getattr(self.bridge, '_mission_confirm', None)
                    was_confirmed = confirm is not None and confirm['confirmed']
                    self.bridge.run_selected_mission_with_retry(True)
                    confirm = getattr(self.bridge, '_mission_confirm', None)
                    if (self.confirm_warnings and confirm is not None and
                            confirm['confirmed'] and not was_confirmed):
                        # The first RUN showed a warning; confirm it as an
                        # operator would, with a second RUN.
                        print('confirming the warning with a second RUN', flush=True)
                        self.bridge.run_selected_mission_with_retry(True)
                first = False
                idle_limit = time.monotonic() + 120.0
                while (self.finished is None and not self.planned and
                       time.monotonic() < deadline):
                    self.spin_until(lambda: self.finished is not None or self.planned, 5.0)
                    if self._busy():
                        idle_limit = time.monotonic() + 120.0
                    elif time.monotonic() > idle_limit:
                        deadline = 0.0
            if self.finished is None:
                result['outcome'] = 'TIMEOUT'
                self.bridge.cancel_navigation()
                self.spin_until(lambda: False, 2.0)
            else:
                result['outcome'] = 'SUCCESS' if self.finished[0] else 'FAILED'
                result['nav2_status'] = self.finished[1]
        pose = self.recorder.pose
        center = STATION_CENTERS[destination]
        result.update({
            'wall_s': time.monotonic() - wall,
            'final_pose': pose,
            'center_error_m': math.hypot(pose[0] - center[0], pose[1] - center[1]),
            'cargo': self.bridge._confirmed_cargo(),
            'lift': self.bridge._lift_position,
            'docked_station': self.bridge.docked_station,
            'phases': self.recorder.summary(),
            'guard_reasons': dict(self.recorder.guard),
            'refinement': dict(self.recorder.refinement),
            'guard_events': list(self.recorder.guard_events),
            'trace': list(self.recorder.trace),
            'track': list(self.recorder.track),
            'reports': self.reports[first_report:],
        })
        return result


class Segment2Sweep:
    """Start real segment 2 from a chosen handover error, without cargo steps.

    Only the isolated validation world is teleported. The robot is placed at the
    staging pose plus the requested error, then the same function the mission
    calls after segment 1 takes over.
    """

    def __init__(self, runner, station):
        self.runner = runner
        self.station = station
        panel = SimpleNamespace(_routes=runner.routes)
        self.route, self.poses = OperatorPanel._load_docking_routes(panel).approach(station)
        bridge = runner.bridge
        bridge._start_cargo_operation = lambda _station: bridge._execution_done(True, 4)

    def teleport(self, x, y, yaw):
        assert os.environ.get('GZ_PARTITION', '').startswith('tugger_validation_')
        request = (f'name: "autonomous_tugger" position {{x: {x} y: {y} z: 0.0}} '
                   f'orientation {{z: {math.sin(yaw / 2)} w: {math.cos(yaw / 2)}}}')
        subprocess.run(['gz', 'service', '-s', '/world/tugger_test/set_pose',
                        '--reqtype', 'gz.msgs.Pose', '--reptype', 'gz.msgs.Boolean',
                        '--timeout', '5000', '--req', request], check=True,
                       capture_output=True)

    def trial(self, lateral, yaw_error, leading_yaw):
        runner, bridge = self.runner, self.runner.bridge
        start = self.poses[0]
        self.teleport(start[0], start[1] + lateral, leading_yaw + yaw_error)
        runner.spin_until(lambda: False, 20.0)
        runner.finished = None
        runner.recorder.reset()
        first_report = len(runner.reports)
        wall = time.monotonic()
        bridge._begin_preflight()
        bridge._precision_station = None
        bridge._camera_handoff_station = self.station
        bridge._queued_precision_station = self.station
        bridge._queued_precision_poses = self.poses
        bridge._motion_armed = True
        bridge._cancel_requested = False
        bridge._start_precision_segment()
        done = runner.spin_until(lambda: runner.finished is not None, runner.timeout)
        if not done:
            bridge.cancel_navigation()
            runner.spin_until(lambda: False, 2.0)
        pose = runner.recorder.pose
        center = STATION_CENTERS[self.station]
        return {
            'lateral_m': lateral, 'yaw_error_rad': yaw_error,
            'outcome': ('TIMEOUT' if not done else
                        'SUCCESS' if runner.finished[0] else 'FAILED'),
            'wall_s': time.monotonic() - wall, 'final_pose': pose,
            'center_error_m': math.hypot(pose[0] - center[0], pose[1] - center[1]),
            'phases': runner.recorder.summary(),
            'guard_events': list(runner.recorder.guard_events),
            'reports': runner.reports[first_report:],
        }


def sweep(runner, args):
    tool = Segment2Sweep(runner, args.sweep)
    results = []
    for item in args.offsets:
        lateral_cm, yaw_deg = (float(value) for value in item.split(':'))
        result = tool.trial(lateral_cm / 100, math.radians(yaw_deg),
                            args.leading_yaw)
        results.append(result)
        Path(args.output).write_text(json.dumps(results, indent=1))
        dock = result['phases'].get('DOCK_NAV', {})
        print(f"offset {item}: {result['outcome']} err={result['center_error_m']:.4f} "
              f"flips={dock.get('dock_cmd_sign_flips')} "
              f"lock={dock.get('full_lock_fraction', 0):.2f} "
              f"maxv={dock.get('max_abs_speed', 0):.3f} "
              f"gate_trips={sum('outside' in e['reason'] for e in result['guard_events'])}",
              flush=True)
        bridge = runner.bridge
        bridge._qr_alignment_active = False
        bridge._active_follow = None
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('destinations', nargs='*',
                        help='stations in order, each optionally :attach, :detach or :none')
    parser.add_argument('--sweep', choices=sorted(STATION_CENTERS),
                        help='segment-2 handover sweep at this station')
    parser.add_argument('--offsets', nargs='*', default=['0:0'],
                        help='lateral_cm:yaw_deg start errors for --sweep')
    parser.add_argument('--leading-yaw', type=float, default=0.0,
                        help='robot yaw at the staging pose for --sweep')
    parser.add_argument('--output', required=True)
    parser.add_argument('--timeout', type=float, default=600.0)
    parser.add_argument('--trial-route', action='append', default=[],
                        help='Allow this unapproved route in this run only')
    parser.add_argument('--trial-candidate', action='append', default=[],
                        help='Allow this unapproved STATION__gate approach in this run only')
    parser.add_argument('--cancel-first-after', type=float,
                        help='cancel the first mission after this many seconds')
    parser.add_argument('--cancel-index', type=int,
                        help='test only: cancel the mission with this index (0-based)')
    parser.add_argument('--cancel-when-moved', type=float, default=0.4,
                        help='with --cancel-index: cancel once the robot moved this far (m)')
    parser.add_argument('--confirm-warnings', action='store_true',
                        help='test only: press RUN a second time when a warning is shown')
    parser.add_argument('--teleport', nargs=3, type=float, metavar=('X', 'Y', 'YAW'),
                        help='move the robot here (isolated world only) before missions')
    parser.add_argument('--disturb-camera', nargs=3, type=float, metavar=('DX', 'DY', 'DYAW'),
                        help='test only: shove the robot once when the camera stage starts')
    parser.add_argument('--disturb-at', metavar='DESTINATION',
                        help='test only: disturb only during the mission to this destination')
    parser.add_argument('--disturb-owner', default='CAMERA',
                        help='test only: CAMERA (the camera stage) or a motion owner such as NAV')
    parser.add_argument('--disturb-delay', type=float, default=0.0,
                        help='test only: seconds after that owner takes over')
    parser.add_argument('--force-live', action='store_true',
                        help='test only: ignore preplanned paths so live planning runs')
    parser.add_argument('--turn-in-place', action='store_true',
                        help='switch the panel\'s turn-in-place capability on')
    parser.add_argument('--settle', type=float, default=30.0,
                        help='seconds to wait after the stack reports ready')
    args = parser.parse_args()
    rclpy.init()
    runner = Runner(args.timeout, args.trial_route, args.trial_candidate)
    if not runner.wait_ready():
        Path(args.output).write_text(json.dumps({'outcome': 'STACK_NOT_READY'}))
        raise SystemExit(3)
    # Action servers can be up before the global costmap has received the map,
    # and path validation then rejects a healthy route.
    runner.spin_until(lambda: False, args.settle)
    if args.sweep:
        sweep(runner, args)
        runner.bridge.destroy_node()
        runner.recorder.destroy_node()
        rclpy.shutdown()
        return
    results = []
    destinations = list(args.destinations)
    runner.force_live = args.force_live
    runner.bridge.turn_in_place_enabled = args.turn_in_place
    runner.disturb = tuple(args.disturb_camera) if args.disturb_camera else None
    runner.confirm_warnings = args.confirm_warnings
    runner.disturb_at = args.disturb_at
    runner.disturb_owner = args.disturb_owner
    runner.disturb_delay = args.disturb_delay
    if args.teleport:
        runner.teleport(*args.teleport)
    if args.cancel_first_after and destinations:
        results.append(runner.cancel_after(destinations.pop(0), args.cancel_first_after))
        print('cancelled on purpose at', runner.recorder.pose, flush=True)
    for index, destination in enumerate(destinations):
        if args.cancel_index is not None and index == args.cancel_index:
            results.append(runner.cancel_after(destination, 600.0, args.cancel_when_moved))
            print('cancelled on purpose at', runner.recorder.pose, flush=True)
            continue
        result = runner.mission(destination)
        results.append(result)
        Path(args.output).write_text(json.dumps(results, indent=1))
        # The requirement is judged on the true pose: 10 mm from the centre.
        if result['outcome'] == 'SUCCESS' and result['center_error_m'] > 0.010:
            result['outcome'] = 'FAILED'
            result['failure'] = 'centre error above the 10 mm requirement'
            Path(args.output).write_text(json.dumps(results, indent=1))
        print(destination, result['outcome'], f"{result['center_error_m']:.4f} m", flush=True)
        nav = result['phases'].get('NAV', {})
        if nav.get('rms_cross_track_m') is not None:
            print(f"  tracking: rms {nav['rms_cross_track_m'] * 100:.1f} cm, "
                  f"max {nav['max_cross_track_m'] * 100:.1f} cm, "
                  f"{nav['steer_reversals_per_10m']:.1f} steering reversals / 10 m, "
                  f"{nav['yaw_rate_flips_per_10m']:.1f} yaw flips / 10 m over "
                  f"{nav['moving_m']:.0f} m", flush=True)
        if result['outcome'] != 'SUCCESS':
            break
    runner.bridge.destroy_node()
    runner.recorder.destroy_node()
    rclpy.shutdown()
    raise SystemExit(0 if all(r['outcome'] in ('SUCCESS', 'CANCELLED_ON_PURPOSE') for r in results) else 1)


if __name__ == '__main__':
    main()
