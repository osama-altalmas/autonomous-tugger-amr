"""RQt dispatcher for offline-generated tugger routes."""

import json
import math
from pathlib import Path

import rclpy
from action_msgs.msg import GoalStatus
from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from geometry_msgs.msg import PoseStamped
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
from std_msgs.msg import Empty, Float64MultiArray
from tf2_ros import Buffer, TransformListener

from tugger_control.direction_policy import prepare_dynamic_path
from tugger_control.saved_route_repository import (
    PrecisionDockingRepository,
    SavedRouteRepository,
)


LIFT_DOWN_M = 0.0
LIFT_UP_M = 0.065


def _yaw(quaternion):
    """Return planar yaw for a ROS quaternion."""
    return math.atan2(
        2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y),
        1.0 - 2.0 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z),
    )


def _angle_error(target, current):
    return math.atan2(math.sin(target - current), math.cos(target - current))


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
        self.docked_station = None
        self._precision_station = None
        self._precision_target = None

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

    def execute_saved_route(self, label, saved_poses, execute, precision_station=None):
        """Dispatch an offline-generated route, with only a short live connector."""
        try:
            x, y, yaw = self._robot_pose()
        except Exception as exc:
            self._report(f'Cannot read live robot pose: {exc}')
            return
        precision_target = saved_poses[-1] if precision_station else None
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
        self._execute_after_plan = execute
        self._precision_station = precision_station
        self._precision_target = precision_target
        if precision_station:
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
            self._prepare_saved_execution(saved_poses, 0.0)
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
        future.add_done_callback(self._on_connector_goal)

    def return_home(self, choices, execute):
        """Choose the nearest station-to-HOME saved-route start."""
        try:
            x, y, _ = self._robot_pose()
        except Exception as exc:
            self._report(f'Cannot read live robot pose: {exc}')
            return
        label, poses = min(
            choices,
            key=lambda item: math.hypot(item[1][0][0] - x, item[1][0][1] - y))
        self.execute_saved_route(label, poses, execute)

    def _on_connector_goal(self, future):
        try:
            handle = future.result()
            if not handle.accepted:
                self._report('Connector planner rejected the saved-route start.')
                return
            result = handle.get_result_async()
            result.add_done_callback(self._on_connector_result)
        except Exception as exc:
            self._report(f'Connector request failed: {exc}')

    def _on_connector_result(self, future):
        try:
            response = future.result()
            connector = response.result.path
            if response.status != GoalStatus.STATUS_SUCCEEDED or len(connector.poses) < 2:
                self._report('No safe short connector to the selected saved route.')
                return
            connector_poses = [[p.pose.position.x, p.pose.position.y, 0.0]
                               for p in connector.poses]
            length = self._path_length(connector_poses)
            if length > 3.0:
                self._report(f'Connector is {length:.2f} m; refusing a long dynamic route.')
                return
            if math.hypot(connector_poses[-1][0] - self._saved_poses[0][0],
                          connector_poses[-1][1] - self._saved_poses[0][1]) > 0.15:
                self._report('Connector does not reach the selected saved-route start.')
                return
            merged = connector_poses[:-1] + self._saved_poses
            self._prepare_saved_execution(merged, length)
        except Exception as exc:
            self._report(f'Connector result failed: {exc}')

    def _prepare_saved_execution(self, poses, connector_length):
        try:
            body_yaw = self._robot_yaw()
            prepared = prepare_dynamic_path([(pose[0], pose[1]) for pose in poses], body_yaw)
        except (Exception, ValueError) as exc:
            self._report(
                f'Phase 15 leading-end policy rejected this plan: {exc}. '
                'Nothing was executed.')
            return
        if self._precision_station:
            # A REAR-leading docking approach intentionally keeps the body
            # 180 degrees from the route travel tangent.  The confirmation
            # must compare against that adapted final body heading, rather
            # than against the raw route-file travel yaw.
            self._precision_target = [
                poses[-1][0], poses[-1][1], prepared['body_headings'][-1]]
        path = NavPath()
        path.header.frame_id = 'map'
        path.header.stamp = self.get_clock().now().to_msg()
        for values, heading in zip(poses, prepared['body_headings']):
            pose = self._pose(values)
            pose.pose.orientation.z = math.sin(heading / 2.0)
            pose.pose.orientation.w = math.cos(heading / 2.0)
            path.poses.append(pose)
        self._path_pub.publish(path)
        self._plan_ready(
            f'{self._route_label}: saved route has {len(poses)} poses; '
            f'connector {connector_length:.2f} m; '
            f'initial leading end {prepared["leading_end"]}; '
            f'body motion {prepared["body_motion_direction"]}.',
            self._execute_after_plan)
        if self._execute_after_plan:
            self._follow(path)

    def _follow(self, path):
        if not self._follower.server_is_ready():
            self._report('FollowPath server is unavailable; plan remains preview-only.')
            return
        goal = FollowPath.Goal()
        goal.path = path
        goal.controller_id = 'FollowPath'
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
            if succeeded and self._precision_station:
                x, y, yaw = self._robot_pose()
                target = self._precision_target
                position_error = math.hypot(target[0] - x, target[1] - y)
                heading_error = abs(_angle_error(target[2], yaw))
                if position_error <= 0.02 and heading_error <= 0.035:
                    self.docked_station = self._precision_station
                    self._report(
                        f'DOCK CENTER CONFIRMED: position error {position_error:.3f} m; '
                        f'heading error {math.degrees(heading_error):.2f} deg.')
                else:
                    succeeded = False
                    self._report(
                        f'DOCK NOT ALIGNED: position error {position_error:.3f} m; '
                        f'heading error {math.degrees(heading_error):.2f} deg. '
                        'Cargo attachment remains blocked.')
            self._execution_done(succeeded, status)
        except Exception as exc:
            self._report(f'FollowPath result failed: {exc}')
        self._active_follow = None

    def cancel_navigation(self):
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
        self._widget = QWidget()
        self._widget.setWindowTitle('Tugger AMR Operator Panel')
        layout = QVBoxLayout(self._widget)
        layout.addWidget(QLabel('Tugger AMR V2 — Gazebo operator panel'))
        self._armed = QCheckBox('Arm Gazebo motion, lift, and cargo commands')
        self._armed.setToolTip('Unchecked is preview-only. Arming is required for every command.')
        self._armed.toggled.connect(self._arming_changed)
        layout.addWidget(self._armed)

        mission_box = QGroupBox('Offline-generated saved routes')
        mission_layout = QGridLayout(mission_box)
        buttons = [
            ('HOME → Station 1 — pickup', 'HOME__TO__cargo_station_0'),
            ('Station 1 → Station 2 — delivery', 'cargo_station_0__TO__cargo_dropoff_northwest'),
            ('HOME → Station 3 — pickup', 'HOME__TO__cargo_station_3'),
            ('Station 3 → Station 4 — delivery', 'cargo_station_3__TO__cargo_dropoff_southeast'),
        ]
        for row, (label, route_key) in enumerate(buttons):
            button = QPushButton(label)
            button.clicked.connect(lambda _checked=False, key=route_key: self._request_saved_route(key))
            mission_layout.addWidget(button, row, 0)
        home = QPushButton('Return HOME from current station')
        home.clicked.connect(self._return_home)
        mission_layout.addWidget(home, len(buttons), 0)
        cancel = QPushButton('Cancel active navigation')
        cancel.clicked.connect(lambda: self._bridge.cancel_navigation())
        mission_layout.addWidget(cancel, len(buttons) + 1, 0)
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
        self._bridge.execute_saved_route(label, poses, self._armed.isChecked())

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
            label, poses, self._armed.isChecked(), precision_station=station)

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
        self._bridge.return_home(choices, self._armed.isChecked())

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
        self._report('Navigation succeeded.' if succeeded else f'Navigation ended with Nav2 status {status}.')

    def shutdown_plugin(self):
        self._timer.stop()
        self._bridge.destroy_node()
