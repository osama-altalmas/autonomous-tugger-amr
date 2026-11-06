#!/usr/bin/env python3
"""Run real missions through OperatorBridge without Qt and record the outcome.

The same mission code, motion gate and controllers as the operator panel are
used; this script only presses the equivalent of Check/plan and RUN. It writes
one JSON result per mission so reliability can be measured across many runs.
"""
import argparse
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

STATION_CENTERS = {
    'cargo_station_0': (-17.5, -7.0), 'cargo_dropoff_northwest': (-17.5, 7.0),
    'cargo_station_3': (17.5, 7.0), 'cargo_dropoff_southeast': (17.5, -7.0),
    'HOME': (0.0, 0.0),
}
FULL_LOCK_RAD = math.radians(40.0)


class Recorder(Node):
    """Observe commands, steering, ownership and true pose; publishes nothing."""

    def __init__(self):
        super().__init__('tugger_mission_recorder')
        self.set_parameters([Parameter('use_sim_time', value=True)])
        self.owner = 'STOP'
        self.pose = None
        self.pose_lookup = None
        self.reset()
        self.create_subscription(String, '/tugger/motion_owner', self._owner, 10)
        self.create_subscription(TwistStamped, '/tugger/dock_nav_cmd_vel', self._dock_cmd, 10)
        self.create_subscription(TwistStamped, '/tugger/guarded_cmd_vel', self._guarded, 10)
        self.create_subscription(Float64MultiArray, '/four_ws/motion_state', self._motion, 10)
        self.create_subscription(PoseStamped, '/tugger/ground_truth_pose', self._pose,
                                 qos_profile_sensor_data)
        self.create_subscription(String, '/tugger/loaded_guard/status', self._guard, 10)
        self.create_subscription(Odometry, '/wheel/odom', self._odom, 10)
        self.map_received = False
        self.create_subscription(
            OccupancyGrid, '/map', self._map,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))

    def reset(self):
        self.phases = {}
        self.guard = {}
        self.guard_events = []
        self._dock_sign = 0
        self._speed = 0.0

    def _phase(self):
        return self.phases.setdefault(self.owner, {
            'dock_cmd_sign_flips': 0, 'guarded_samples': 0, 'full_lock_samples': 0,
            'steer_samples': 0, 'first_t': None, 'last_t': None, 'first_pose': None,
            'path_m': 0.0, '_last': None, 'max_abs_speed': 0.0,
            'max_estimate_error_m': 0.0, 'slow_s': 0.0, '_slow_t': None,
            'max_cross_track_m': 0.0})

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

    def _guarded(self, _msg):
        self._phase()['guarded_samples'] += 1

    def _motion(self, msg):
        if len(msg.data) < 4:
            return
        phase = self._phase()
        phase['steer_samples'] += 1
        phase['full_lock_samples'] += abs(msg.data[3]) >= FULL_LOCK_RAD

    def _pose(self, msg):
        q = msg.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        self.pose = (msg.pose.position.x, msg.pose.position.y, yaw)
        phase = self._phase()
        now = self._now()
        phase['first_t'] = now if phase['first_t'] is None else phase['first_t']
        phase['last_t'] = now
        phase['first_pose'] = phase['first_pose'] or self.pose
        if phase['_last'] is not None:
            phase['path_m'] += math.hypot(self.pose[0] - phase['_last'][0],
                                          self.pose[1] - phase['_last'][1])
        phase['_last'] = self.pose
        if phase['_slow_t'] is not None and abs(self._speed) < 0.3:
            phase['slow_s'] += max(0.0, now - phase['_slow_t'])
        phase['_slow_t'] = now
        if self.pose_lookup is not None:
            try:
                estimate = self.pose_lookup()
                phase['max_estimate_error_m'] = max(
                    phase['max_estimate_error_m'],
                    math.hypot(estimate[0] - self.pose[0], estimate[1] - self.pose[1]))
            except Exception:
                pass

    def _guard(self, msg):
        self.guard[msg.data] = self.guard.get(msg.data, 0) + 1
        self.guard_events.append({'t': self._now(), 'reason': msg.data,
                                  'wheel_speed': self._speed})

    def _map(self, _msg):
        self.map_received = True

    def _odom(self, msg):
        self._speed = float(msg.twist.twist.linear.x)
        phase = self._phase()
        phase['max_abs_speed'] = max(phase['max_abs_speed'], abs(self._speed))

    def summary(self):
        result = {}
        for name, data in self.phases.items():
            result[name] = {key: value for key, value in data.items()
                            if key not in ('_last', '_slow_t')}
            samples = max(data['steer_samples'], 1)
            result[name]['full_lock_fraction'] = data['full_lock_samples'] / samples
        return result


class Runner:
    def __init__(self, timeout, trial_routes=(), trial_candidates=()):
        self.timeout = timeout
        self.trial_routes = set(trial_routes)
        self.trial_candidates = set(trial_candidates)
        self.force_live = False
        self.disturb = None          # (dx, dy): shove once when the camera stage starts
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
        self.recorder = Recorder()
        self.recorder.pose_lookup = self.bridge._robot_pose
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
        bridge = self.bridge
        return (bridge._follow_goal_pending or bridge._stage_validation_pending or
                bridge._departure_pending is not None or bridge._departure_active or
                bridge._active_follow is not None or bridge._qr_alignment_active or
                bridge._cargo_task is not None)

    def _maybe_disturb(self):
        """Test only: shove the robot once at the camera stage to force a failure."""
        if (self.disturb is None or self._disturbed or
                self.recorder.owner != 'DOCK_CAMERA' or self.recorder.pose is None):
            return
        self._disturbed = True
        x, y, yaw = self.recorder.pose
        print(f'disturbing robot by {self.disturb} at the camera stage', flush=True)
        dyaw = self.disturb[2] if len(self.disturb) > 2 else 0.0
        self._set_pose(x + self.disturb[0], y + self.disturb[1], yaw + dyaw)

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

    def cancel_after(self, destination, seconds):
        """Start a mission, cancel it part-way, and report where it stopped."""
        self.finished, self.planned = None, False
        self.bridge.execute_destination(
            destination, self.mission_paths.paths_to(destination),
            self.entries.get(destination, []))
        self.spin_until(lambda: self.planned, 60)
        self.bridge._motion_armed = True
        self.bridge.run_selected_mission_with_retry(True)
        self.spin_until(lambda: False, seconds)
        self.bridge.cancel_navigation()
        self.spin_until(lambda: False, 5.0)
        self.bridge._cancel_requested = False
        self.bridge._motion_armed = True
        return {'destination': destination, 'outcome': 'CANCELLED_ON_PURPOSE',
                'final_pose': self.recorder.pose, 'center_error_m': 0.0, 'wall_s': seconds,
                'phases': {}, 'reports': [], 'guard_events': [], 'cargo': None,
                'lift': self.bridge._lift_position, 'docked_station': None,
                'guard_reasons': {}}

    def _set_pose(self, x, y, yaw):
        assert os.environ.get('GZ_PARTITION', '').startswith('tugger_validation_')
        request = (f'name: "autonomous_tugger" position {{x: {x} y: {y} z: 0.0}} '
                   f'orientation {{z: {math.sin(yaw / 2)} w: {math.cos(yaw / 2)}}}')
        subprocess.run(['gz', 'service', '-s', '/world/tugger_test/set_pose',
                        '--reqtype', 'gz.msgs.Pose', '--reptype', 'gz.msgs.Boolean',
                        '--timeout', '5000', '--req', request], check=True,
                       capture_output=True)

    def teleport(self, x, y, yaw):
        self._set_pose(x, y, yaw)
        self.spin_until(lambda: False, 20.0)
        self.bridge.docked_station = None

    def mission(self, destination):
        self.finished, self.planned = None, False
        self.recorder.reset()
        first_report = len(self.reports)
        wall = time.monotonic()
        result = {'destination': destination, 'outcome': 'NOT_RUN'}
        # Test only: hide preplanned paths so the live planner must run.
        candidates = [] if self.force_live else self.mission_paths.paths_to(destination)
        for key, route, _poses in candidates:
            if key in self.trial_routes:
                # In memory only, for this validation run. Files stay unapproved.
                route['execution_approved'] = True
                route['_database_execution_approved'] = True
        self.bridge.start_mission(
            destination, candidates, self.entries.get(destination, []))
        if not self.spin_until(lambda: self.planned, 180):
            result['outcome'] = 'NO_PLAN'
        else:
            self.bridge._motion_armed = True
            deadline = time.monotonic() + self.timeout
            # A loaded departure asks for a new RUN at staging, as for an operator.
            # Press RUN once, like an operator; the panel continues by itself
            # after undocking. Press again only if nothing is running.
            first = True
            while self.finished is None and time.monotonic() < deadline:
                self.planned = False
                if first or not self._busy():
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
            'guard_events': list(self.recorder.guard_events),
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
    parser.add_argument('destinations', nargs='*', choices=[*sorted(STATION_CENTERS), []])
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
    parser.add_argument('--teleport', nargs=3, type=float, metavar=('X', 'Y', 'YAW'),
                        help='move the robot here (isolated world only) before missions')
    parser.add_argument('--disturb-camera', nargs=3, type=float, metavar=('DX', 'DY', 'DYAW'),
                        help='test only: shove the robot once when the camera stage starts')
    parser.add_argument('--force-live', action='store_true',
                        help='test only: ignore preplanned paths so live planning runs')
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
    runner.disturb = tuple(args.disturb_camera) if args.disturb_camera else None
    if args.teleport:
        runner.teleport(*args.teleport)
    if args.cancel_first_after and destinations:
        results.append(runner.cancel_after(destinations.pop(0), args.cancel_first_after))
        print('cancelled on purpose at', runner.recorder.pose, flush=True)
    for destination in destinations:
        result = runner.mission(destination)
        results.append(result)
        Path(args.output).write_text(json.dumps(results, indent=1))
        print(destination, result['outcome'], f"{result['center_error_m']:.4f} m", flush=True)
        if result['outcome'] != 'SUCCESS':
            break
    runner.bridge.destroy_node()
    runner.recorder.destroy_node()
    rclpy.shutdown()
    raise SystemExit(0 if all(r['outcome'] in ('SUCCESS', 'CANCELLED_ON_PURPOSE') for r in results) else 1)


if __name__ == '__main__':
    main()
