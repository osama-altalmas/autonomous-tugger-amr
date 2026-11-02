"""RQt dispatcher for offline-generated tugger routes."""

import json
import math
from pathlib import Path

import rclpy
from action_msgs.msg import GoalStatus
from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from geometry_msgs.msg import PoseStamped, TwistStamped
from nav_msgs.msg import Path as NavPath
from nav2_msgs.action import ComputePathToPose, FollowPath
from python_qt_binding.QtCore import QTimer
from python_qt_binding.QtWidgets import (
    QCheckBox, QGridLayout, QGroupBox, QLabel, QMessageBox, QPushButton,
    QTextEdit, QVBoxLayout, QWidget,
)
from qt_gui.plugin import Plugin
from rclpy.action import ActionClient
from rclpy.node import Node
from std_msgs.msg import Empty, Float64MultiArray, String
from tf2_ros import Buffer, TransformListener

from tugger_control.direction_policy import prepare_dynamic_path
from tugger_control.mission_planning import plan_is_current, select_destination_plan
from tugger_control.qr_alignment import bounded_qr_correction
from tugger_control.docking_markers import expected_marker
from tugger_control.saved_route_repository import (
    PrecisionDockingRepository,
    SavedRouteRepository,
)


LIFT_DOWN_M = 0.0
LIFT_UP_M = 0.065
QR_CONFIRMATION_MAX_AGE_S = 0.5
QR_ALIGNMENT_TIMEOUT_S = 2.0
QR_ALIGNMENT_STABLE_FRAMES = 3


def _yaw(quaternion):
    """Return planar yaw for a ROS quaternion."""
    return math.atan2(
        2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y),
        1.0 - 2.0 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z),
    )


class OperatorBridge(Node):
    """ROS work layer.  Every action result is delivered asynchronously."""

    def __init__(self, report, plan_ready, execution_done):
        super().__init__('tugger_operator_panel')
        self._report = report
        self._plan_ready = plan_ready
        self._execution_done = execution_done
        self._tf = Buffer()
        self._listener = TransformListener(self._tf, self)
        self._planner = ActionClient(self, ComputePathToPose, '/compute_path_to_pose')
        self._follower = ActionClient(self, FollowPath, '/follow_path')
        self._path_pub = self.create_publisher(NavPath, '/mission/planned_path', 10)
        self._precision_cmd_pub = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        self._lift_pub = self.create_publisher(
            Float64MultiArray, '/lift_controller/commands', 10)
        self._attach = {
            0: self.create_publisher(Empty, '/cargo/station_0/attach', 10),
            3: self.create_publisher(Empty, '/cargo/station_3/attach', 10),
        }
        self._detach = {
            0: self.create_publisher(Empty, '/cargo/station_0/detach', 10),
            3: self.create_publisher(Empty, '/cargo/station_3/detach', 10),
        }
        self._active_follow = None
        self._pending_path = None
        self._pending_pose = None
        self._plan_generation = 0
        self.docked_station = None
        self._precision_station = None
        self._camera_handoff_station = None
        self._recovery_only = False
        self._qr_identity = None
        self._qr_seen_at = None
        self._qr_expected_seen_at = None
        self._qr_alignment_active = False
        self._qr_alignment_station = None
        self._qr_aligned_frames = 0
        self._qr_alignment_started_at = None
        self._red_frame_seen_at = None
        self._red_frame_aligned = False
        self._red_frame_reported = False
        self._red_frame_error = (float('inf'), float('inf'))
        self._last_red_frame_state_report_at = None
        self.create_subscription(String, '/qr_docking/id', self._qr_id, 10)
        self.create_subscription(
            PoseStamped, '/qr_docking/frame_pose', self._red_frame_pose, 10)
        self.create_timer(0.1, self._qr_alignment_watchdog)

    def _qr_id(self, message):
        self._qr_identity = message.data
        self._qr_seen_at = self.get_clock().now()
        if (self._qr_alignment_active and
                message.data == expected_marker(self._qr_alignment_station)):
            self._qr_expected_seen_at = self._qr_seen_at

    def _expected_qr_is_fresh(self, station):
        if self._qr_seen_at is None:
            return False
        age = (self.get_clock().now() - self._qr_seen_at).nanoseconds / 1e9
        return (0.0 <= age <= QR_CONFIRMATION_MAX_AGE_S and
                self._qr_identity == expected_marker(station))

    def _publish_precision_stop(self):
        command = TwistStamped()
        command.header.stamp = self.get_clock().now().to_msg()
        self._precision_cmd_pub.publish(command)

    @staticmethod
    def _rotate(quaternion, x, y, z):
        """Rotate a point using a ROS quaternion without a TF geometry helper."""
        qx, qy, qz, qw = quaternion.x, quaternion.y, quaternion.z, quaternion.w
        tx = 2.0 * (qy * z - qz * y)
        ty = 2.0 * (qz * x - qx * z)
        tz = 2.0 * (qx * y - qy * x)
        return (
            x + qw * tx + (qy * tz - qz * ty),
            y + qw * ty + (qz * tx - qx * tz),
            z + qw * tz + (qx * ty - qy * tx),
        )

    def _red_frame_pose(self, message):
        """Use the red docking-frame centre for final positional control."""
        if not self._qr_alignment_active:
            return
        station = self._qr_alignment_station
        self._red_frame_seen_at = self.get_clock().now()
        if message.header.frame_id != 'base_footprint':
            self._stop_qr_alignment(
                f'red-frame pose must be base_footprint, got {message.header.frame_id}')
            return
        marker_x = message.pose.position.x
        marker_y = message.pose.position.y
        proposal = bounded_qr_correction(marker_y, marker_x, 0.0)
        self._red_frame_error = (marker_x, marker_y)
        self._report_red_frame_state(marker_x, marker_y, proposal['state'])
        if proposal['state'] == 'ALIGNED':
            self._qr_aligned_frames += 1
            self._publish_precision_stop()
            if self._qr_aligned_frames >= QR_ALIGNMENT_STABLE_FRAMES:
                self._red_frame_aligned = True
                self._complete_red_frame_alignment()
            return
        self._qr_aligned_frames = 0
        self._red_frame_aligned = False
        self._red_frame_reported = False
        command = TwistStamped()
        command.header.stamp = self.get_clock().now().to_msg()
        command.header.frame_id = 'base_footprint'
        command.twist.linear.x = proposal['linear_x']
        command.twist.angular.z = proposal['angular_z']
        self._precision_cmd_pub.publish(command)

    def _report_red_frame_state(self, forward_m, lateral_m, state):
        """Show the camera's final-position estimate without flooding the GUI."""
        now = self.get_clock().now()
        if self._last_red_frame_state_report_at is not None:
            elapsed = (now - self._last_red_frame_state_report_at).nanoseconds / 1e9
            if elapsed < 0.5:
                return
        self._last_red_frame_state_report_at = now
        status = 'CENTERED' if state == 'ALIGNED' else 'CORRECTING'
        self._report(
            f'RED FRAME {status}: forward error={forward_m:.3f} m, '
            f'lateral error={lateral_m:.3f} m; tolerance=0.020 m.')

    def _complete_red_frame_alignment(self):
        """Report final success after the red-frame positional gate passes."""
        if not self._qr_alignment_active:
            return
        station = self._qr_alignment_station
        marker_x, marker_y = self._red_frame_error
        self._qr_alignment_active = False
        self.docked_station = station
        self._publish_precision_stop()
        self._report(
            f'MISSION SUCCESS: red frame centered for {station}; '
            f'forward error={marker_x:.3f} m, lateral error={marker_y:.3f} m '
            f'within 20 mm for '
            f'{QR_ALIGNMENT_STABLE_FRAMES} frames.')
        self._execution_done(True, GoalStatus.STATUS_SUCCEEDED)

    def _stop_qr_alignment(self, reason):
        if not self._qr_alignment_active:
            return
        self._qr_alignment_active = False
        self._publish_precision_stop()
        self._report(f'PRECISION ALIGNMENT FAILED: {reason}. Robot stopped.')
        self._execution_done(False, GoalStatus.STATUS_ABORTED)

    def _start_qr_alignment(self, station):
        self._qr_alignment_active = True
        self._qr_alignment_station = station
        self._qr_aligned_frames = 0
        self._qr_seen_at = None
        self._qr_expected_seen_at = None
        self._red_frame_seen_at = None
        self._red_frame_aligned = False
        self._red_frame_reported = False
        self._red_frame_error = (float('inf'), float('inf'))
        self._last_red_frame_state_report_at = None
        self._qr_alignment_started_at = self.get_clock().now()
        self._report(
            f'NAV2 COARSE APPROACH COMPLETE. Red-frame alignment now owns final '
            f'motion for {station}: centering within 20 mm.  QR '
            f'{expected_marker(station)} remains an identity observation.')

    def _qr_alignment_watchdog(self):
        if not self._qr_alignment_active:
            return
        reference = self._red_frame_seen_at or self._qr_alignment_started_at
        age = (self.get_clock().now() - reference).nanoseconds / 1e9
        if age > QR_ALIGNMENT_TIMEOUT_S:
            self._stop_qr_alignment('red-frame observation became stale')

    def _pose(self, values):
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x, pose.pose.position.y, yaw = values
        pose.pose.orientation.z = math.sin(yaw / 2.0)
        pose.pose.orientation.w = math.cos(yaw / 2.0)
        return pose

    def _robot_yaw(self):
        transform = self._tf.lookup_transform(
            'map', 'base_footprint', rclpy.time.Time())
        return _yaw(transform.transform.rotation)

    def _robot_pose(self):
        transform = self._tf.lookup_transform(
            'map', 'base_footprint', rclpy.time.Time())
        translation = transform.transform.translation
        return translation.x, translation.y, _yaw(transform.transform.rotation)

    @staticmethod
    def _path_length(poses):
        return sum(math.hypot(b[0] - a[0], b[1] - a[1])
                   for a, b in zip(poses, poses[1:]))

    def _begin_preflight(self):
        """Invalidate any earlier preview before selecting a new mission."""
        self._plan_generation += 1
        self._pending_path = None
        self._pending_pose = None
        return self._plan_generation

    def execute_saved_route(self, label, saved_poses, precision_station=None,
                            camera_handoff_station=None):
        """Dispatch an offline-generated route, with only a short live connector."""
        generation = self._begin_preflight()
        try:
            x, y, yaw = self._robot_pose()
        except Exception as exc:
            self._report(f'Cannot read live robot pose: {exc}')
            return
        if precision_station:
            nearest_index = min(
                range(len(saved_poses)),
                key=lambda index: math.hypot(
                    saved_poses[index][0] - x, saved_poses[index][1] - y))
            nearest_gap = math.hypot(
                saved_poses[nearest_index][0] - x,
                saved_poses[nearest_index][1] - y)
            # A repeated Dock click may occur after Nav2 stopped within its
            # old 7 cm facility tolerance.  Rejoin only a nearby point of
            # this precision route; never create a shortcut across the stand.
            if nearest_index and nearest_gap <= 0.25:
                rejoin_index = max(0, nearest_index - 1)
                saved_poses = saved_poses[rejoin_index:]
                self._report(
                    f'{label}: rejoining saved precision route at pose '
                    f'{nearest_index}; lateral/longitudinal gap {nearest_gap:.3f} m.')
        start = saved_poses[0]
        gap = math.hypot(start[0] - x, start[1] - y)
        yaw_gap = abs(math.atan2(math.sin(start[2] - yaw), math.cos(start[2] - yaw)))
        self._route_label = label
        self._saved_poses = saved_poses
        self._recovery_only = False
        self._precision_station = precision_station
        self._camera_handoff_station = (
            camera_handoff_station or precision_station)
        if self._camera_handoff_station:
            self.docked_station = None
        # Nav2 commonly returns an empty path when a goal is only centimetres
        # away.  Position, not the route-file yaw, decides whether a connector
        # exists.  _prepare_saved_execution still checks the *live* body yaw
        # against the first travel tangent before anything is followed.
        if gap <= 0.25:
            if yaw_gap > math.radians(20):
                self._report(
                    f'{label}: start is {gap:.2f} m away; bypassing the empty '
                    'near-zero connector and applying the Phase 15 yaw gate.')
            self._prepare_saved_execution(saved_poses, 0.0, generation)
            return
        if gap > 3.0:
            self._report(
                f'{label}: route start is {gap:.2f} m away; refusing a long dynamic replacement route.')
            return
        if not self._planner.server_is_ready():
            self._report('Nav2 connector planner is unavailable; no command was sent.')
            return
        self._report(f'{label}: planning a {gap:.2f} m connector into the saved route.')
        goal = ComputePathToPose.Goal()
        goal.goal = self._pose(start)
        goal.planner_id = 'GridBased'
        goal.use_start = False
        future = self._planner.send_goal_async(goal)
        future.add_done_callback(
            lambda result, token=generation: self._on_connector_goal(result, token))

    def execute_destination(self, destination, candidates, station_targets):
        """Select a destination from the live pose without guessing a route."""
        generation = self._begin_preflight()
        try:
            pose = self._robot_pose()
            decision = select_destination_plan(
                destination, candidates, pose, destination_poses=station_targets)
        except (Exception, ValueError) as exc:
            self._report(f'Destination selection rejected: {exc}')
            return
        if decision['mode'] == 'ALREADY_AT_DESTINATION':
            self._report(f'{destination}: already at destination; no plan was sent.')
            return
        if decision['mode'] != 'LIVE_RECOVERY_REQUIRED':
            route = decision['route']
            label = f'{route["source"]} → {destination} ({decision["mode"]})'
            self._report(
                f'{label}: rejoin pose {decision["rejoin_index"]}; '
                f'gap {decision["gap_m"]:.2f} m; '
                f'leading end {decision["direction"]["leading_end"]}.')
            self.execute_saved_route(
                label, decision['poses'], camera_handoff_station=destination)
            return
        if not station_targets:
            self._report(f'{destination}: no station entry is available for live recovery.')
            return
        target = min(station_targets, key=lambda point: math.hypot(
            point[0] - pose[0], point[1] - pose[1]))
        self._report(
            f'{destination}: no safe saved-route rejoin; planning live recovery '
            f'to a facility entry point.')
        self._route_label = f'Live recovery → {destination}'
        self._saved_poses = []
        self._precision_station = None
        self._camera_handoff_station = destination
        self._recovery_only = True
        if not self._planner.server_is_ready():
            self._report('Nav2 planner is unavailable; live recovery remains preview-only.')
            return
        goal = ComputePathToPose.Goal()
        goal.goal = self._pose(target)
        goal.planner_id = 'GridBased'
        goal.use_start = False
        future = self._planner.send_goal_async(goal)
        future.add_done_callback(
            lambda result, token=generation: self._on_connector_goal(result, token))

    def return_home(self, choices):
        """Choose the nearest station-to-HOME saved-route start."""
        try:
            x, y, _ = self._robot_pose()
        except Exception as exc:
            self._report(f'Cannot read live robot pose: {exc}')
            return
        label, poses = min(
            choices,
            key=lambda item: math.hypot(item[1][0][0] - x, item[1][0][1] - y))
        self.execute_saved_route(label, poses)

    def _on_connector_goal(self, future, generation):
        if generation != self._plan_generation:
            return
        try:
            handle = future.result()
            if not handle.accepted:
                self._report('Connector planner rejected the saved-route start.')
                return
            result = handle.get_result_async()
            result.add_done_callback(
                lambda response, token=generation: self._on_connector_result(response, token))
        except Exception as exc:
            self._report(f'Connector request failed: {exc}')

    def _on_connector_result(self, future, generation):
        if generation != self._plan_generation:
            return
        try:
            response = future.result()
            connector = response.result.path
            if response.status != GoalStatus.STATUS_SUCCEEDED or len(connector.poses) < 2:
                self._report('No safe short connector to the selected saved route.')
                return
            connector_poses = [[p.pose.position.x, p.pose.position.y, 0.0]
                               for p in connector.poses]
            length = self._path_length(connector_poses)
            if self._recovery_only:
                self._recovery_only = False
                self._prepare_saved_execution(connector_poses, length, generation)
                return
            if length > 3.0:
                self._report(f'Connector is {length:.2f} m; refusing a long dynamic route.')
                return
            if math.hypot(connector_poses[-1][0] - self._saved_poses[0][0],
                          connector_poses[-1][1] - self._saved_poses[0][1]) > 0.15:
                self._report('Connector does not reach the selected saved-route start.')
                return
            merged = connector_poses[:-1] + self._saved_poses
            self._prepare_saved_execution(merged, length, generation)
        except Exception as exc:
            self._report(f'Connector result failed: {exc}')

    def _prepare_saved_execution(self, poses, connector_length, generation=None):
        if generation is not None and generation != self._plan_generation:
            return
        try:
            body_yaw = self._robot_yaw()
            preview_pose = self._robot_pose()
            prepared = prepare_dynamic_path([(pose[0], pose[1]) for pose in poses], body_yaw)
        except (Exception, ValueError) as exc:
            self._report(
                f'Phase 15 leading-end policy rejected this plan: {exc}. '
                'Nothing was executed.')
            return
        path = NavPath()
        path.header.frame_id = 'map'
        path.header.stamp = self.get_clock().now().to_msg()
        for values, heading in zip(poses, prepared['body_headings']):
            pose = self._pose(values)
            pose.pose.orientation.z = math.sin(heading / 2.0)
            pose.pose.orientation.w = math.cos(heading / 2.0)
            path.poses.append(pose)
        self._path_pub.publish(path)
        self._pending_path = path
        self._pending_pose = preview_pose
        self._plan_ready(
            f'{self._route_label}: saved route has {len(poses)} poses; '
            f'connector {connector_length:.2f} m; '
            f'initial leading end {prepared["leading_end"]}; '
            f'body motion {prepared["body_motion_direction"]}; '
            'press RUN to dispatch this exact checked path.',
            False)

    def run_selected_mission(self, armed):
        """Run only the exact path previously previewed by the operator."""
        if not armed:
            self._report('RUN blocked: arm simulator commands first.')
            return
        if self._pending_path is None:
            self._report('RUN blocked: select and validate a mission first.')
            return
        try:
            check = plan_is_current(self._pending_pose, self._robot_pose())
        except (Exception, ValueError) as exc:
            self._report(f'RUN blocked: cannot recheck the live pose: {exc}')
            return
        if not check['current']:
            self._report(
                'RUN blocked: robot pose changed since feasibility check '
                f'({check["position_error_m"]:.2f} m, '
                f'{math.degrees(check["yaw_error_rad"]):.1f} deg). '
                'Check/plan the destination again.')
            return
        if not self._follower.server_is_ready():
            self._report('FollowPath server is unavailable; plan remains preview-only.')
            return
        path = self._pending_path
        self._pending_path = None
        self._pending_pose = None
        self._follow(path)

    def _follow(self, path):
        if not self._follower.server_is_ready():
            self._report('FollowPath server is unavailable; plan remains preview-only.')
            return
        goal = FollowPath.Goal()
        goal.path = path
        goal.controller_id = (
            'PrecisionFollowPath'
            if self._precision_station else 'FollowPath')
        goal.goal_checker_id = (
            'precision_goal_checker'
            if self._precision_station else 'goal_checker')
        future = self._follower.send_goal_async(goal)
        future.add_done_callback(self._on_follow_goal)

    def _on_follow_goal(self, future):
        try:
            self._active_follow = future.result()
            if not self._active_follow.accepted:
                self._report('Nav2 rejected the trajectory; the robot remains stopped.')
                return
            self._report('Nav2 trajectory accepted.')
            result = self._active_follow.get_result_async()
            result.add_done_callback(self._on_follow_result)
        except Exception as exc:
            self._report(f'FollowPath request failed: {exc}')

    def _on_follow_result(self, future):
        try:
            status = future.result().status
            succeeded = status == GoalStatus.STATUS_SUCCEEDED
            if succeeded and self._camera_handoff_station:
                self._active_follow = None
                self._start_qr_alignment(self._camera_handoff_station)
                return
            self._execution_done(succeeded, status)
        except Exception as exc:
            self._report(f'FollowPath result failed: {exc}')
        self._active_follow = None

    def cancel_navigation(self):
        if self._qr_alignment_active:
            self._stop_qr_alignment('operator cancelled precision alignment')
            return
        if self._active_follow is None:
            self._report('No active Nav2 trajectory to cancel.')
            return
        self._active_follow.cancel_goal_async()
        self._report('Nav2 trajectory cancellation requested.')

    def lift(self, height):
        message = Float64MultiArray()
        message.data = [height]
        self._lift_pub.publish(message)
        self._report(f'Lift command published: {height:.3f} m.')

    def cargo(self, station, attach):
        (self._attach if attach else self._detach)[station].publish(Empty())
        verb = 'attach' if attach else 'detach'
        self._report(f'Cargo {verb} command sent for station {station}.')


class OperatorPanel(Plugin):
    """RQt entry point for planning missions and guarded simulator commands."""

    def __init__(self, context):
        super().__init__(context)
        self.setObjectName('OperatorPanel')
        self._routes = self._load_routes()
        self._docking_routes = self._load_docking_routes()
        self._station_entries = self._load_station_entries()
        self._widget = QWidget()
        self._widget.setWindowTitle('Tugger AMR Operator Panel')
        layout = QVBoxLayout(self._widget)
        layout.addWidget(QLabel('Tugger AMR V2 — Gazebo operator panel'))
        self._armed = QCheckBox('Arm Gazebo motion, lift, and cargo commands')
        self._armed.setToolTip('Unchecked is preview-only. Arming is required for every command.')
        self._armed.toggled.connect(self._arming_changed)
        layout.addWidget(self._armed)

        mission_box = QGroupBox('Destination missions (saved-route rejoin or live recovery)')
        mission_layout = QGridLayout(mission_box)
        buttons = [
            ('Check / plan Station 1', 'cargo_station_0'),
            ('Check / plan Station 2', 'cargo_dropoff_northwest'),
            ('Check / plan Station 3', 'cargo_station_3'),
            ('Check / plan Station 4', 'cargo_dropoff_southeast'),
        ]
        for row, (label, destination) in enumerate(buttons):
            button = QPushButton(label)
            button.clicked.connect(
                lambda _checked=False, name=destination: self._request_destination(name))
            mission_layout.addWidget(button, row, 0)
        home = QPushButton('Check / plan HOME from current pose')
        home.clicked.connect(lambda: self._request_destination('HOME'))
        mission_layout.addWidget(home, len(buttons), 0)
        run = QPushButton('RUN selected mission')
        run.clicked.connect(lambda: self._bridge.run_selected_mission(self._armed.isChecked()))
        mission_layout.addWidget(run, len(buttons) + 1, 0)
        cancel = QPushButton('Cancel active navigation')
        cancel.clicked.connect(lambda: self._bridge.cancel_navigation())
        mission_layout.addWidget(cancel, len(buttons) + 2, 0)
        layout.addWidget(mission_box)

        docking_box = QGroupBox('Precision docking (Gazebo candidate routes)')
        docking_layout = QGridLayout(docking_box)
        for row, (label, station) in enumerate((
                ('Dock beneath Station 1 — then attach cargo', 'cargo_station_0'),
                ('Dock beneath Station 3 — then attach cargo', 'cargo_station_3'),
        )):
            button = QPushButton(label)
            button.clicked.connect(
                lambda _checked=False, name=station: self._request_precision_docking(name))
            docking_layout.addWidget(button, row, 0)
        layout.addWidget(docking_box)

        cargo_box = QGroupBox('Cargo and lift (only when armed)')
        cargo_layout = QGridLayout(cargo_box)
        controls = [
            ('Attach cargo at Station 1', lambda: self._cargo(0, True)),
            ('Detach cargo at Station 2', lambda: self._cargo(0, False)),
            ('Attach cargo at Station 3', lambda: self._cargo(3, True)),
            ('Detach cargo at Station 4', lambda: self._cargo(3, False)),
            ('Lift up', lambda: self._lift(LIFT_UP_M)),
            ('Lift down', lambda: self._lift(LIFT_DOWN_M)),
        ]
        for index, (label, callback) in enumerate(controls):
            button = QPushButton(label)
            button.clicked.connect(callback)
            cargo_layout.addWidget(button, index // 2, index % 2)
        layout.addWidget(cargo_box)

        self._status = QTextEdit()
        self._status.setReadOnly(True)
        self._status.setMinimumHeight(120)
        layout.addWidget(self._status)
        context.add_widget(self._widget)
        self._bridge = OperatorBridge(self._report, self._planned, self._finished)
        self._timer = QTimer(self._widget)
        self._timer.timeout.connect(self._spin_ros)
        self._timer.start(20)
        self._report('Preview mode: dispatches offline-generated routes; only a short connector is planned live.')

    def _load_routes(self):
        try:
            directory = Path(get_package_share_directory('tugger_control')) / 'routes'
        except PackageNotFoundError:
            directory = Path(__file__).resolve().parents[3] / 'tools' / 'v1_manhattan_bezier' / 'routes'
        return SavedRouteRepository(directory)

    def _load_docking_routes(self):
        try:
            directory = Path(get_package_share_directory('tugger_control')) / 'docking'
        except PackageNotFoundError:
            directory = (Path(__file__).resolve().parents[3] / 'tools'
                         / 'v1_manhattan_bezier' / 'phase14' / 'routes_refined')
        return PrecisionDockingRepository(directory)

    def _load_station_entries(self):
        try:
            config = (Path(get_package_share_directory('tugger_control')) /
                      'config' / 'operator_stations.json')
        except PackageNotFoundError:
            config = Path(__file__).resolve().parents[1] / 'config' / 'operator_stations.json'
        data = json.loads(config.read_text(encoding='utf-8'))
        entries = {}
        for station, record in data.get('stations', {}).items():
            entries[station] = [list(candidate['pose'])
                                for candidate in record.get('candidates', [])]
        return entries

    def _spin_ros(self):
        rclpy.spin_once(self._bridge, timeout_sec=0.0)

    def _report(self, message):
        self._status.append(message)

    def _arming_changed(self, armed):
        if armed:
            response = QMessageBox.question(
                self._widget, 'Arm Gazebo commands',
                'This enables Nav2, lift, and cargo commands in the simulator. Continue?',
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No)
            if response != QMessageBox.StandardButton.Yes:
                self._armed.blockSignals(True)
                self._armed.setChecked(False)
                self._armed.blockSignals(False)
                return
        self._report('SIMULATOR COMMANDS ARMED.' if armed else 'Preview-only mode restored.')

    def _request_saved_route(self, route_key):
        try:
            route, poses = self._routes.route(route_key)
        except (OSError, ValueError, KeyError) as exc:
            self._report(f'Saved-route load rejected: {exc}')
            return
        label = f'{route["source"]} → {route["target"]}'
        self._bridge.execute_saved_route(label, poses)

    def _request_destination(self, destination):
        try:
            candidates = self._routes.routes_to(destination)
        except (OSError, ValueError, KeyError) as exc:
            self._report(f'Destination-route load rejected: {exc}')
            return
        self._bridge.execute_destination(
            destination, candidates, self._station_entries.get(destination, []))

    def _request_precision_docking(self, station):
        try:
            route, poses = self._docking_routes.approach(station)
        except (OSError, ValueError, KeyError) as exc:
            self._report(f'Precision-docking load rejected: {exc}')
            return
        label = f'{route["destination"]} precision docking'
        self._report(
            'Phase 14 precision route selected: static leg and vertical-clearance '
            'checks passed; this remains a Gazebo-only candidate.')
        self._bridge.execute_saved_route(
            label, poses, precision_station=station)

    def _return_home(self):
        choices = []
        for source in ('cargo_station_0', 'cargo_dropoff_northwest',
                       'cargo_station_3', 'cargo_dropoff_southeast'):
            try:
                route, poses = self._routes.route(f'{source}__TO__HOME')
                choices.append((f'{route["source"]} → HOME', poses))
            except (OSError, ValueError, KeyError) as exc:
                self._report(f'HOME-route load rejected: {exc}')
                return
        self._bridge.return_home(choices)

    def _lift(self, height):
        if not self._armed.isChecked():
            self._report('Lift blocked: arm simulator commands first.')
            return
        self._bridge.lift(height)

    def _cargo(self, station, attach):
        if not self._armed.isChecked():
            self._report('Cargo command blocked: arm simulator commands first.')
            return
        if attach and self._bridge.docked_station != f'cargo_station_{station}':
            self._report(
                'Cargo attachment blocked: precision docking has not confirmed '
                'the robot at this station center.')
            return
        self._bridge.cargo(station, attach)

    def _planned(self, message, executing):
        self._report(message + (' Executing.' if executing else ' Preview only.'))

    def _finished(self, succeeded, status):
        self._report('Mission successful.' if succeeded else f'Navigation ended with Nav2 status {status}.')

    def shutdown_plugin(self):
        self._timer.stop()
        self._bridge.destroy_node()
