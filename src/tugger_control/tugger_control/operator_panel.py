"""RQt dispatcher for offline-generated tugger routes."""

import json
import re
import math
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from pathlib import Path

import rclpy
from action_msgs.msg import GoalStatus
from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, TwistStamped
from nav_msgs.msg import Path as NavPath
from nav_msgs.msg import Odometry
from nav2_msgs.action import ComputePathToPose, FollowPath
from nav2_msgs.srv import IsPathValid
from python_qt_binding.QtCore import QTimer
from python_qt_binding.QtWidgets import (
    QCheckBox, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QMessageBox, QPushButton,
    QTextEdit, QVBoxLayout, QWidget,
)
from qt_gui.plugin import Plugin
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Empty, Float64MultiArray, String
from tf2_ros import Buffer, TransformListener

from tugger_control.direction_policy import prepare_dynamic_path, validate_planner_motion
from tugger_control.cargo_policy import confirmed_cargo
from tugger_control.cargo_policy import (
    CARGO_IDS, CARGO_STATIONS, DROPOFF_CARGO, PICKUP_CARGO, cargo_stand_model, other_cargo)
from tugger_control.cargo_operation import CargoOperation
from tugger_control.lift_geometry import (
    LIFT_CONTACT_M, LIFT_DOWN_M, LIFT_UP_M, cargo_lift_offset, lift_is_down, lift_is_raised)
from tugger_control.loaded_motion_guard import decode_scene, scene_expected_counts
from tugger_control.mission_planning import (
    entrance_gate, home_alignment_yaw, mission_approach_rejoin, plan_is_current,
    routes_from_gate, select_destination_plan)
from tugger_control.station_departure import plan_station_departure
from tugger_control.precision_scene_clearance import sampled_precision_clearance
from tugger_control.visual_maneuver_executor import VisualManeuverExecutor
from tugger_control.visual_maneuver_planner import marker_visible, replan_visual_maneuver
from tugger_control.clearance_2_5d import collision_pairs
from tugger_control.visual_scene_clearance import PreparedVisualClearance
from tugger_control.backout_path import backout_candidates
from tugger_control.route_network import (network_edges, plan_route_chain,
                                          plan_route_chain_from)
from tugger_control.rotation_clearance import rotation_clear
from tugger_control.turn_in_place import (MAX_YAW_RATE_LOADED, MAX_YAW_RATE_UNLOADED,
                                          ROTATION_MARGIN_M, SETTLE_S,
                                          TURN_SHIFT_TOLERANCE_M, approach_axis,
                                          rotation_time_s, rotation_to_axis, wrap)
from tugger_control.station_occupancy import (
    ATTACH, DETACH, assess_destination, load_occupancy, measure_occupancy, save_occupancy,
    station_label)
from tugger_control.visual_heading_gate import relative_qr_heading_consistent
from tugger_control.visual_marker_guard import FixedMarkerGuard
from tugger_control.docking_markers import expected_marker
from tugger_control.live_route_planner import LiveRoutePlanner
from tugger_control.saved_route_repository import (
    MissionPathRepository,
    PrecisionDockingRepository,
    SavedRouteRepository,
)
from tugger_control.warehouse_clearance import (
    lift_sweep_clear, loaded_path_clear, robot_boxes, validate_modeled_geometry,
)


QR_CONFIRMATION_MAX_AGE_S = 0.5
QR_ALIGNMENT_TIMEOUT_S = 3.0
# Four-wheel lateral parking uses multiple slow arcs and stationary checkpoints.
QR_ALIGNMENT_MAX_DURATION_S = 200.0
QR_ALIGNMENT_POSITION_TOLERANCE_M = 0.010
# Guard band: the camera stage accepts and plans to 7 mm so the true centre
# error stays inside the 10 mm requirement with the camera's measurement
# uncertainty (about 1-2 mm). Accepting at the limit itself let a 9.9 mm
# camera reading end 10.1 mm off.
QR_ALIGNMENT_ACCEPT_M = 0.007
QR_ALIGNMENT_HEADING_TOLERANCE_RAD = math.radians(1.0)
QR_VISUAL_MAX_PLAN_FAILURES = 3
QR_ALIGNMENT_STABLE_FRAMES = 3
CONTINUE_ATTEMPTS = 5
QR_IDENTITY_GRACE_S = 1.5
QR_VISUAL_MAX_NO_PROGRESS = 6
QR_VISUAL_PROGRESS_M = 0.002
MAX_MISSION_RETRIES = 2
RECOVERY_DELAY_S = 1.0
RETRYABLE_CARGO_FAILURES = ('attach_rejected', 'timeout', 'cargo_moved_before_attach')
# Docked means at the station centre; farther away the robot is between stages.
DOCKED_TOLERANCE_M = 0.10
# Robot heading = floor-QR angle in the downward camera + this constant, for
# a QR placed at map yaw 0 (all station markers are). Measured in Gazebo at
# four headings (spread 0.3-0.6 degrees per frame); a mounting calibration.
QR_TO_ROBOT_YAW = math.pi / 2
QR_YAW_SAMPLES = 5
# Longest wait at a staging point for a landmark fix before entering anyway.
LANDMARK_WAIT_S = 3.0
# Heading error the car-like station exit accepts without turning in place.
ENTRY_HEADING_TOLERANCE_DEG = 15.0
GUARD_NOTE_INTERVAL_S = 15.0
# Routine guard states: normal operation, not worth a panel line.
QUIET_GUARD_STATES = ('ready', 'stopped', 'unloaded motion', 'precision guarded',
                      'precision speed reduced', 'loaded clearance reducing speed')


def guard_status_text(status):
    """Turn a motion-gate status into an operator line, or None for routine ones."""
    if status in QUIET_GUARD_STATES:
        return None
    match = re.search(r"envelope intersects \('([^']+)', '([^']+)'", status)
    if match:
        # Scoped names look like 'world::model::link::collision'; show the model.
        parts = match.group(2).split('::')
        model = parts[1] if len(parts) > 1 else parts[0].split('/')[0]
        return f'MOTION GUARD: slowing or holding near {model} ({match.group(1)}).'
    return f'MOTION GUARD: {status}.'
CONTINUE_RETRY_S = 0.5
QR_ALIGNMENT_MAX_OBSERVATION_AGE_S = 0.5
# A visual primitive is only a short proposal.  If the measured marker has
# moved materially from the snapshot used to create that proposal, stop at
# the next safe cycle and replan from fresh camera/TF feedback.
QR_VISUAL_REPLAN_MARKER_DELTA_M = 0.006
QR_STATIONARY_MAX_SPEED_M_S = 0.005
QR_STATIONARY_MAX_YAW_RATE_RAD_S = 0.02


# A running camera correction is stopped early only once the robot is well
# inside the target (and aligned): there it can settle and be accepted.
QR_EARLY_STOP_M = 0.006


def early_stop_inside_target(radial_error, anchor_error, heading_error):
    """True when a running correction may stop early and settle.

    Only well inside the 10 mm acceptance limit. Stopping anywhere within 12 mm
    trapped a robot at 10-12 mm: each new correction was cancelled on the
    next camera frame, moved 1-4 mm, and "no progress" ended the stage.
    """
    return (radial_error <= QR_EARLY_STOP_M and anchor_error <= QR_EARLY_STOP_M and
            abs(heading_error) <= QR_ALIGNMENT_HEADING_TOLERANCE_RAD)


def _compose_pose(base, local):
    """Pose ``local`` (x, y, yaw) expressed in frame ``base`` -> outer frame."""
    c, s = math.cos(base[2]), math.sin(base[2])
    yaw = base[2] + local[2]
    return (base[0] + c * local[0] - s * local[1], base[1] + s * local[0] + c * local[1],
            math.atan2(math.sin(yaw), math.cos(yaw)))


def _yaw(quaternion):
    """Return planar yaw for a ROS quaternion."""
    return math.atan2(
        2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y),
        1.0 - 2.0 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z),
    )


class ClearanceUnavailable(ValueError):
    """Clearance could not be judged (stale data), which is not a collision."""


LIFT_DATA_GAP_S = 1.0
# Inside this radius of a station centre the robot is under or beside the
# stand; the same radius keeps saved routes out of stations (route audit).
STAND_AREA_RADIUS_M = 1.5


def _marker_in_map(robot_pose, marker):
    """Map position of a marker measured in base_footprint at robot_pose."""
    x, y, yaw = robot_pose
    c, sn = math.cos(yaw), math.sin(yaw)
    return (x + c * marker[0] - sn * marker[1], y + sn * marker[0] + c * marker[1])


class OperatorBridge(Node):
    """ROS work layer.  Every action result is delivered asynchronously."""

    def __init__(self, report, plan_ready, execution_done, approach_provider=None,
                 cargo_clearance_guard=None, station_layout=None, live_planner=None):
        super().__init__('tugger_operator_panel')
        # The Gazebo scene, TF and payload feedback all use simulation time.
        # Without this, comparing their timestamps to wall time fails closed.
        self.set_parameters([Parameter('use_sim_time', value=True)])
        self._report = report
        self._plan_ready = lambda message, executing: self._announce_plan(
            plan_ready, message, executing)
        self._execution_done = lambda succeeded, status: self._finish_mission(
            execution_done, succeeded, status)
        self._mission_retry = None
        self._mission_phase = None
        self._approach_provider = approach_provider
        self._live_planner = live_planner
        self._live_plan = None
        # From the generated route database: exact centres and the stops
        # without a stand, which may be left by any gate.
        layout = station_layout or {}
        self._station_centers = dict(layout.get('station_centers', {}))
        self._pass_through = tuple(layout.get('pass_through', ()))
        self._cargo_clearance_guard = cargo_clearance_guard or self._modeled_lift_clearance
        self._cargo_task = None
        self._loaded_control_ready = False
        self._loaded_guard_seen_at = None
        self._collision_scene = None
        self._scene_sequence = 0
        self._last_guard_status = None
        self._tf = Buffer()
        self._listener = TransformListener(self._tf, self)
        self._planner = ActionClient(self, ComputePathToPose, '/compute_path_to_pose')
        self._follower = ActionClient(self, FollowPath, '/follow_path')
        self._dock_follower = ActionClient(self, FollowPath, '/docking/follow_path')
        self._path_validator = self.create_client(IsPathValid, '/is_path_valid')
        self._path_pub = self.create_publisher(NavPath, '/mission/planned_path', 10)
        self._precision_cmd_pub = self.create_publisher(
            TwistStamped, '/tugger/docking/cmd_vel', 10)
        self._motion_owner_pub = self.create_publisher(String, '/tugger/motion_owner', 10)
        self._rotate_pub = self.create_publisher(TwistStamped, '/tugger/rotate_cmd_vel', 10)
        self._initial_pose_pub = self.create_publisher(
            PoseWithCovarianceStamped, '/initialpose', 10)
        # Safety (safety_monitor.py): panel emergency stop, reset, status.
        self._estop_pub = self.create_publisher(Bool, '/tugger/estop/panel', 10)
        self._safety_reset_pub = self.create_publisher(Empty, '/tugger/safety/reset', 10)
        self._safety_state = None
        self.create_subscription(String, '/tugger/safety/state', self._on_safety_state, 10)
        self._landmark_fix_at = None
        self.create_subscription(String, '/tugger/localization/landmark_fix',
                                 self._on_landmark_fix, 10)
        # Turn in place: an optional capability (operator panel switch).
        self.turn_in_place_enabled = False
        self._pending_rotation = None
        self._rotation = None
        self._turn_recentre = None
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
        self._cargo_states = {0: None, 3: None}
        self._cargo_seen_at = {0: None, 3: None}
        self._lift_position = None
        self._lift_seen_at = None
        self._cargo_poses = {0: None, 3: None}
        self._cargo_pose_seen_at = {0: None, 3: None}
        self._cargo_pose_subscriptions = [
            self.create_subscription(
                PoseStamped, f'/tugger/cargo/station_{station}/estimated_pose',
                lambda message, station=station: self._cargo_pose(station, message), 10)
            for station in CARGO_IDS
        ]
        self._cargo_state_subscriptions = [
            self.create_subscription(
                String, f'/cargo/station_{station}/state',
                lambda message, station=station: self._cargo_state(station, message), 10)
            for station in CARGO_IDS
        ]
        self._active_follow = None
        self._follow_goal_pending = False
        self._pending_path = None
        self._pending_pose = None
        self._pending_alignment_station = None
        self._pending_precision_start = None
        self._pending_destination = None
        self._pending_execution_approved = False
        self._queued_precision_poses = None
        self._queued_precision_station = None
        self._stage_validation_pending = False
        self._dock_precision_mode = False
        self._dock_pending = None
        self._departure_pending = None
        self._pending_departure = None
        self._departure_active = False
        self._departure_target = None
        self._departure_exit_pose = None
        self._station_exit_active = False
        self._station_exit_target = None
        self._arrival_approach = None
        self._continue_after_undock = False
        self._motion_armed = False
        self._cancel_requested = False
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
        self._red_frame_stamp = None
        self._red_frame_aligned = False
        self._red_frame_reported = False
        self._red_frame_error = (float('inf'), float('inf'))
        self._last_red_frame_state_report_at = None
        self._alignment_odom = None
        self._alignment_odom_stamp = None
        # Planning is deliberately outside the ROS/Qt callback thread.  The
        # worker receives immutable pose/scene snapshots and can never publish
        # a command; its result is consumed by _qr_alignment_watchdog.
        self._visual_plan_pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix='visual-maneuver-plan')
        self._visual_plan_lock = Lock()
        self._visual_plan_future = None
        self._visual_plan_generation = 0
        self._visual_plan_result = None
        self._visual_plan_error = None
        self._visual_plan_origin = None
        self._visual_plan_marker = None
        self._visual_plan_generation_at_submit = 0
        self._visual_latest_marker = None
        self._visual_latest_camera_stamp = None
        self._visual_executor = None
        self._visual_replan_requested = False
        self._visual_plan_failures = 0
        self._visual_best_metric = None
        self._visual_no_progress = 0
        self._visual_marker_guard = None
        self._visual_camera_outliers = 0
        # Heading comes from the approved precision approach (not a guessed
        # zero-yaw marker pose). The QR vision's red-frame quaternion is identity.
        self._precision_handoff_yaw = None
        self._alignment_target_yaw = None
        self._qr_heading_angle = None
        self._qr_heading_stamp = None
        self._qr_heading_reference = None
        self.create_subscription(String, '/qr_docking/id', self._qr_id, 10)
        self.create_subscription(PoseStamped, '/qr_docking/pose',
                                 self._qr_heading_pose, 10)
        self.create_subscription(Bool, '/tugger/loaded_guard/ready', self._guard_ready, 10)
        self.create_subscription(String, '/tugger/precision_guard/ready', self._dock_guard_ready, 10)
        self.create_subscription(String, '/tugger/loaded_guard/status', self._guard_status, 10)
        self.create_subscription(String, '/tugger/collision_scene', self._scene_update, 10)
        self.create_subscription(JointState, '/joint_states', self._joint_state, 10)
        self.create_subscription(Odometry, '/wheel/odom', self._alignment_odometry, 10)
        self.create_subscription(
            PoseStamped, '/qr_docking/frame_pose', self._red_frame_received, 10)
        # Each camera frame is processed with the robot pose at its own stamp.
        self._red_frame_waiting = None
        self.create_timer(0.02, self._release_red_frame)
        self.create_timer(0.1, self._qr_alignment_watchdog)
        self.create_timer(0.1, self._advance_cargo_operation)
        self.create_timer(0.1, self._publish_motion_owner)
        self.create_timer(0.05, self._rotation_step)
        self.create_timer(0.1, self._dock_wait_watchdog)
        self.create_timer(0.2, self._poll_live_plan)

    def _guard_ready(self, message):
        self._loaded_control_ready = bool(message.data)
        self._loaded_guard_seen_at = self.get_clock().now()

    def _dock_wait_watchdog(self):
        if self._departure_pending is not None:
            started = self._departure_pending['started']
            if (self.get_clock().now() - started).nanoseconds / 1e9 > 5.0:
                self._departure_pending = None
                self._departure_target = None
                self._report('Loaded departure blocked: no fresh scene/guard handshake within 5 s.')
                self._execution_done(False, GoalStatus.STATUS_ABORTED)
        if self._dock_pending is None:
            return
        started = self._dock_pending[4]
        if (self.get_clock().now() - started).nanoseconds / 1e9 > 5.0:
            self._dock_pending = None
            self._stage_validation_pending = False
            self._dock_precision_mode = False
            self._queued_precision_poses = None
            self._report('Precision segment blocked: docking guard did not become ready. Robot stopped.')
            self._execution_done(False, GoalStatus.STATUS_ABORTED)

    def _dock_guard_ready(self, message):
        if message.data == 'DOCK_DEPART' and self._departure_pending is not None:
            self._departure_pending['guard_seen_at'] = self.get_clock().now()
            self._on_departure_guard_ready()
            return
        if message.data != 'DOCK_NAV' or self._dock_pending is None:
            return
        generation, station, path, start_pose, _ = self._dock_pending
        if generation != self._plan_generation or not self._stage_validation_pending:
            return
        try:
            if not self._motion_armed or self._cancel_requested:
                raise ValueError('simulator commands are disarmed or cancelled')
            if self._confirmed_cargo() is not None:
                raise ValueError('unloaded docking guard cannot move attached cargo')
            if not plan_is_current(start_pose, self._robot_pose())['current']:
                raise ValueError('robot pose changed while validating precision approach')
            if not self._dock_follower.server_is_ready():
                raise ValueError('independent docking controller is unavailable')
            self._dock_pending = None
            self._stage_validation_pending = False
            self._queued_precision_poses = None
            self._precision_station = station
            self._report(f'{station}: simulator precision guard ready; starting segment 2.')
            self._follow(path)
        except Exception as exc:
            self._dock_pending = None
            self._stage_validation_pending = False
            self._dock_precision_mode = False
            self._queued_precision_poses = None
            self._report(f'Precision segment blocked: {exc}. Robot stopped.')
            self._execution_done(False, GoalStatus.STATUS_ABORTED)


    def _announce_plan(self, plan_ready, message, executing):
        """Report a plan; after undocking, continue through the normal RUN checks."""
        plan_ready(message, executing)
        if self._continue_after_undock:
            self._continue_after_undock = False
            self._continue_attempt(1)

    def _mission_started(self):
        return (self._follow_goal_pending or self._stage_validation_pending or
                self._departure_pending is not None or self._departure_active or
                self._active_follow is not None or self._qr_alignment_active or
                getattr(self, '_rotation', None) is not None)

    def _report_run_block(self, message):
        """Automatic retries stay quiet; only a final failure is reported."""
        if not getattr(self, '_quiet_run_blocks', False):
            self._report(message)

    def run_selected_mission_with_retry(self, armed):
        """Operator RUN: a momentary check failure is retried quietly.

        An unusual mission (see station_occupancy) is shown as a warning on
        the first RUN; the operator's second RUN confirms it. Only this
        operator entry point confirms: automatic retries and continuations
        never do, so they cannot run an unconfirmed mission.
        """
        if not armed:
            self._report('RUN blocked: arm simulator commands first.')
            return
        confirm = getattr(self, '_mission_confirm', None)
        if confirm is not None and not confirm['confirmed']:
            confirm['confirmed'] = True
            self._report('WARNING: ' + '; '.join(confirm['warnings']) +
                         '. Press RUN again to proceed.')
            return
        self._motion_armed = True
        self._continue_attempt(1)

    def _station_occupancy(self):
        """Live station cargo state from estimated cargo poses; file as backup."""
        now = self.get_clock().now()
        fresh = all(
            self._cargo_pose_seen_at.get(key) is not None and
            0 <= (now - self._cargo_pose_seen_at[key]).nanoseconds / 1e9 <= 2.0
            for key in self._cargo_poses)
        if fresh:
            try:
                attached = self._confirmed_cargo()
            except ValueError:
                attached = None
            state = measure_occupancy(self._cargo_poses, attached,
                                      getattr(self, '_station_centers', {}))
            if state is not None:
                if state != getattr(self, '_saved_occupancy', None):
                    try:
                        save_occupancy(state)
                        self._saved_occupancy = state
                    except OSError:
                        pass
                return state
        return load_occupancy()

    def _assess_destination(self, destination, cargo):
        return assess_destination(destination, cargo, self._station_occupancy(),
                                  getattr(self, '_mission_action', None))

    def _deviation_confirmed(self, destination):
        confirm = getattr(self, '_mission_confirm', None)
        return (confirm is not None and confirm['destination'] == destination and
                confirm['confirmed'])

    def _check_destination_allowed(self, destination, cargo):
        """Hard blocks always stop; warnings need the operator's confirmation."""
        via = getattr(self, '_mission_via', None)
        if via is not None and destination == via['hop']:
            return  # a waypoint gate, not a destination
        blocks, warnings, _dock_only = self._assess_destination(destination, cargo)
        if blocks:
            raise ValueError(blocks[0])
        if warnings and not self._deviation_confirmed(destination):
            raise ValueError(f'unconfirmed: {warnings[0]}; press RUN twice to confirm')

    def _dock_only(self, station):
        """Dock without a lift action: no ATTACH or DETACH was chosen."""
        return getattr(self, '_mission_action', None) is None

    def select_cargo_action(self, action):
        """Operator ATTACH/DETACH: for the planned mission, or at once when docked.

        With a mission previewed and not yet running, the choice applies to
        it and is checked against the station memory. With no mission and the
        robot docked at a station, the action runs in place. A choice the
        station memory rules out is refused and the lift does not move.
        """
        if action not in (ATTACH, DETACH):
            raise ValueError(f'unknown cargo action {action}')
        busy = (self._follow_goal_pending or self._stage_validation_pending or
                self._departure_pending is not None or self._departure_active or
                self._active_follow is not None or self._qr_alignment_active or
                self._cargo_task is not None)
        # During a chained mission the choice is for the final station, not
        # for a waypoint on the way.
        via = getattr(self, '_mission_via', None)
        destination = (via['final'] if via is not None else
                       getattr(self, '_mission_retry', None) and
                       self._mission_retry['destination'])
        if busy:
            self._report(f'{action.upper()} not accepted: a mission is running.')
            return
        try:
            cargo = self._confirmed_cargo()
        except ValueError as exc:
            self._report(f'{action.upper()} not accepted: {exc}.')
            return
        if self._pending_destination is not None and destination is not None:
            blocks, warnings, _ = assess_destination(
                destination, cargo, self._station_occupancy(), action)
            if blocks:
                self._report(f'{action.upper()} not accepted for {station_label(destination)}: '
                             f'{blocks[0]}.')
                return
            self._mission_action = action
            if warnings:
                self._mission_confirm = {'destination': destination, 'warnings': warnings,
                                         'dock_only': False, 'confirmed': False}
            else:
                self._mission_confirm = None
            self._report(f'{action.upper()} selected for {station_label(destination)}. '
                         'Press RUN.')
            return
        self._manual_cargo_action(action, cargo)

    def _manual_cargo_action(self, action, cargo):
        """ATTACH/DETACH at the station the robot is docked at, no driving."""
        station = self.docked_station
        if station is None or station in getattr(self, '_pass_through', ()):
            self._report(f'{action.upper()} not accepted: select a station first, or dock '
                         'at a station with a stand.')
            return
        blocks, _warnings, _ = assess_destination(
            station, cargo, self._station_occupancy(), action)
        if blocks:
            self._report(f'{action.upper()} not accepted at {station_label(station)}: '
                         f'{blocks[0]}. The lift does not move.')
            return
        if not self._motion_armed:
            self._report(f'{action.upper()} not accepted: arm simulator commands first.')
            return
        self._mission_action = action
        self._mission_retry = None
        self._report(f'{action.upper()} at {station_label(station)}: starting in place.')
        self._start_cargo_operation(station)

    def _continue_attempt(self, attempt):
        """Press RUN for the planned leg; retry briefly if a check was momentary.

        A collision scene can be a few milliseconds stale at the instant of the
        check. Every retry runs the full RUN checks; nothing is bypassed.
        """
        self._quiet_run_blocks = attempt < CONTINUE_ATTEMPTS
        try:
            self.run_selected_mission(self._motion_armed)
        finally:
            self._quiet_run_blocks = False
        if self._mission_started() or self._cancel_requested or not self._motion_armed:
            return
        if attempt >= CONTINUE_ATTEMPTS:
            self._report(f'Automatic continuation stopped after {attempt} attempts; '
                         'press RUN to retry.')
            return
        timer = None

        def retry():
            timer.cancel()
            self._continue_attempt(attempt + 1)
        timer = self.create_timer(CONTINUE_RETRY_S, retry)

    def _station_partway(self, pose):
        """The cargo station whose approach the robot stands on, short of its gate."""
        for station in CARGO_STATIONS:
            try:
                approach = self._station_approach(station)
            except (OSError, ValueError, KeyError):
                continue
            if not approach:
                continue
            gate, dock = approach[0], approach[-1]
            from_dock = math.hypot(pose[0] - dock[0], pose[1] - dock[1])
            # Partway means inside the station's approach zone: off the dock,
            # and clearly closer to it than the gate (a robot stopped at a
            # gate, e.g. as a waypoint, is outside the station).
            if (from_dock <= DOCKED_TOLERANCE_M or
                    from_dock >= math.hypot(gate[0] - dock[0], gate[1] - dock[1]) - 0.30):
                continue
            if min(math.hypot(p[0] - pose[0], p[1] - pose[1]) for p in approach) <= 0.30:
                return station
        return None

    def _station_approach(self, station):
        """Return the approach the robot actually entered this station by."""
        arrival = getattr(self, '_arrival_approach', None)
        if arrival is not None and arrival[0] == station:
            return arrival[1]
        if self._approach_provider is None:
            raise ValueError('No configured station approach is available for departure')
        return self._approach_provider(station)[1]

    def _start_route_chain(self, destination, candidates, station_targets):
        """Go via a pass-through stop when no saved path leaves the entry gate.

        The saved paths form a network; a chain of them through HOME can reach
        a destination that no single path from this gate reaches. Returns True
        when the first leg of a chain was started.
        """
        station = self.docked_station
        provider = getattr(self, '_candidate_provider', None)
        pass_through = tuple(getattr(self, '_pass_through', ()))
        if (station is None or station in pass_through or provider is None or
                getattr(self, '_mission_via', None) is not None):
            return False
        gate = entrance_gate(self._station_approach(station))
        # The cheapest connection decides, direct or chained; a stop counts as
        # extra distance (route_network.STOP_PENALTY_M).
        stations = list(getattr(self, '_station_centers', {}))
        edges = network_edges(provider(name)[0] for name in stations)
        stops = plan_route_chain(edges, station, gate, destination, pass_through)
        if not stops or len(stops) < 2:
            return False  # the best connection is direct (or none exists)
        hop = stops[0]
        self._report(f'{station}: no saved route to {destination} leaves by the {gate} gate '
                     f'the robot entered by; going via {" -> ".join(stops)} '
                     f'(saved routes, one RUN; {hop} is only a waypoint at its gate: no docking).')
        self._mission_via = {'final': destination, 'candidates': candidates,
                             'targets': station_targets, 'hop': hop}
        hop_candidates, hop_targets = provider(hop)
        retry = getattr(self, '_mission_retry', None)
        if retry is not None:
            retry.update(destination=hop, candidates=hop_candidates, targets=hop_targets)
        self.execute_destination(hop, hop_candidates, hop_targets)
        return True

    def _continue_route_chain(self):
        """After reaching a chain's intermediate stop, head for the destination."""
        via = self._mission_via
        self._mission_via = None
        retry = getattr(self, '_mission_retry', None)
        if retry is not None:
            retry.update(destination=via['final'], candidates=via['candidates'],
                         targets=via['targets'])
        self._report(f'{via["hop"]} gate reached (waypoint, no docking); continuing to '
                     f'{via["final"]} automatically.')
        self._continue_after_undock = True
        timer = None

        def go():
            timer.cancel()
            self.execute_destination(via['final'], via['candidates'], via['targets'])
        timer = self.create_timer(0.5, go)

    # ---- Turn in place (optional; turn_in_place.py) -------------------------
    def _dock_gate_approaches(self, station):
        """Network edges, and each gate's approach for leaving ``station``."""
        provider = self._candidate_provider
        lists = [provider(name)[0] for name in getattr(self, '_station_centers', {})]
        approaches = {}
        for candidates in lists:
            for key, meta, poses in candidates:
                gate = meta.get('source_entrance')
                if (meta.get('source') != station or gate in approaches or not poses or
                        '(reversed)' in key):
                    continue
                start = meta.get('markers', {}).get('route_start', 0)
                try:
                    approaches[gate] = self._approach_provider(station, poses[start])[1]
                except (OSError, ValueError, KeyError, IndexError):
                    continue
        return network_edges(lists), approaches

    def _turn_clearance(self, pose, cargo):
        """Is the whole turning circle clear at the dock? Returns (clear, reason)."""
        try:
            now = self.get_clock().now()
            if self._collision_scene is None:
                raise ValueError('live simulator collision scene is unavailable')
            if (self._lift_seen_at is None or
                    not 0 <= (now - self._lift_seen_at).nanoseconds / 1e9 <= 1.0):
                raise ValueError('lift feedback is stale')
            if cargo is None and not lift_is_down(self._lift_position):
                raise ValueError('the lift is not down')
            if cargo is not None and not lift_is_raised(self._lift_position):
                raise ValueError('the cargo is not raised')
            cargo_pose = None
            if cargo is not None:
                seen = self._cargo_pose_seen_at[cargo]
                if seen is None or not 0 <= (now - seen).nanoseconds / 1e9 <= 1.0:
                    raise ValueError('estimated cargo pose is stale or missing')
                cargo_pose = self._cargo_poses[cargo]
            world = Path(get_package_share_directory('tugger_gazebo')) / 'worlds/tugger_warehouse.sdf'
            scene = decode_scene(self._collision_scene, now.nanoseconds / 1e9,
                                 expected_counts=scene_expected_counts(world))
            clear, hit = rotation_clear(tuple(pose), self._lift_position, scene, 2 * math.pi,
                                        cargo_id=cargo, cargo_pose=cargo_pose,
                                        margin=ROTATION_MARGIN_M)
            return clear, (None if clear else f'turning circle meets {hit[1]}')
        except (OSError, ValueError, KeyError, TypeError) as exc:
            return False, str(exc)

    def _maybe_turn_at_dock(self, destination, candidates, station_targets):
        """With turn in place on: turn at the dock when another gate is faster.

        Every gate the station can be left by is compared by time: the turn
        to face it (none for the gate the robot entered by) plus the saved
        route or chain from it. Other gates are offered only when the whole
        turning circle, loaded or not, is clear of the stand legs and every
        other modelled obstacle. Returns True when a turn was previewed.
        """
        if not getattr(self, 'turn_in_place_enabled', False):
            return False
        station = self.docked_station
        if (station is None or station in getattr(self, '_pass_through', ()) or
                getattr(self, '_mission_via', None) is not None or
                getattr(self, '_candidate_provider', None) is None or
                self._approach_provider is None):
            return False
        cargo = self._confirmed_cargo()
        loaded = cargo is not None
        pose = self._robot_pose()
        entry = self._station_approach(station)
        entry_gate = entrance_gate(entry)
        edges, approaches = self._dock_gate_approaches(station)
        approaches.setdefault(entry_gate, entry)
        turns = {gate: rotation_to_axis(pose[2], approach_axis(approach))
                 for gate, approach in approaches.items()}
        clear, why = self._turn_clearance(pose, cargo)
        # The car-like exit already accepts the small heading error left by
        # docking (station_departure: up to 20 degrees); no turn for that.
        entry_ok = abs(turns[entry_gate]) < math.radians(ENTRY_HEADING_TOLERANCE_DEG)
        start_gates = {entry_gate: 0.0 if entry_ok else
                       rotation_time_s(turns[entry_gate], loaded)}
        if clear:
            for gate, turn in turns.items():
                start_gates.setdefault(gate, rotation_time_s(turn, loaded))
        else:
            if len(turns) > 1:
                self._report(f'{station}: turn in place not used ({why}); leaving by the '
                             f'{entry_gate} gate it entered by.')
            return False  # the car-like exit decides whether the heading is usable
        best = plan_route_chain_from(edges, station, start_gates, destination,
                                     getattr(self, '_pass_through', ()))
        if best is None:
            return False
        stops, gate, cost = best
        if gate == entry_gate and entry_ok:
            return False  # leaving by the entry gate without a turn: unchanged
        baseline = plan_route_chain_from(edges, station, {entry_gate: 0.0}, destination,
                                         getattr(self, '_pass_through', ()))
        saving = ('' if baseline is None or gate == entry_gate else
                  f'; about {baseline[2] - cost:.0f} s faster than leaving by the '
                  f'{entry_gate} gate it entered by')
        via = '' if len(stops) < 2 else f' via {" -> ".join(stops[:-1])}'
        self._pending_rotation = {
            'station': station, 'gate': gate, 'approach': approaches[gate],
            'target': wrap(pose[2] + turns[gate]), 'turn': turns[gate], 'loaded': loaded,
            'mission': (destination, candidates, station_targets)}
        self._pending_pose = pose
        self._pending_destination = destination
        self._pending_execution_approved = True
        self._plan_ready(
            f'{station}: TURN IN PLACE {math.degrees(turns[gate]):+.0f} deg at the dock '
            f'({"loaded" if loaded else "unloaded"}; turning circle clear), then leave by '
            f'the {gate} gate to {destination}{via}{saving}. Press RUN; the mission '
            'continues automatically.', False)
        return True

    def _start_rotation(self):
        pending = self._pending_rotation
        self._pending_rotation = None
        now = self.get_clock().now().nanoseconds / 1e9
        error = abs(wrap(pending['target'] - self._robot_pose()[2]))
        self._rotation = dict(pending, started=now, best=error, progress_at=now,
                              still_since=None, start_pose=tuple(self._robot_pose()))
        self._mission_phase = 'turn'
        self._report(f'{pending["station"]}: turning in place '
                     f'{math.degrees(pending["turn"]):+.0f} deg.')

    def _send_rotation(self, omega):
        message = TwistStamped()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = 'pivot'
        message.twist.angular.z = float(omega)
        self._rotate_pub.publish(message)

    def _rotation_step(self):
        """Turn toward the target heading; settle; then continue the mission."""
        rotation = getattr(self, '_rotation', None)
        if rotation is None:
            return
        if self._cancel_requested or not self._motion_armed:
            self._end_rotation(False, 'turn in place stopped: motion disarmed or cancelled')
            return
        now = self.get_clock().now().nanoseconds / 1e9
        try:
            yaw = self._robot_pose()[2]
        except Exception as exc:  # TF unavailable: stop rather than guess
            self._end_rotation(False, f'turn in place stopped: robot pose unavailable: {exc}')
            return
        error = wrap(rotation['target'] - yaw)
        if abs(error) < rotation['best'] - math.radians(0.5):
            rotation['best'], rotation['progress_at'] = abs(error), now
        if abs(error) <= math.radians(0.4):
            self._send_rotation(0.0)
            if rotation['still_since'] is None:
                rotation['still_since'] = now
            elif now - rotation['still_since'] >= SETTLE_S:
                self._end_rotation(True, None)
            return
        rotation['still_since'] = None
        if now - rotation['progress_at'] > 6.0:
            status = getattr(self, '_last_guard_status', None)
            self._end_rotation(False, 'turn in place made no progress for 6 s'
                               + (f' ({status})' if status else ''))
            return
        limit = MAX_YAW_RATE_LOADED if rotation['loaded'] else MAX_YAW_RATE_UNLOADED
        self._send_rotation(math.copysign(min(limit, max(0.06, 0.9 * abs(error))), error))

    def _end_rotation(self, succeeded, reason):
        rotation = self._rotation
        self._rotation = None
        self._send_rotation(0.0)
        if not succeeded:
            self._report(f'{rotation["station"]}: {reason}. Robot stopped.')
            self._execution_done(False, GoalStatus.STATUS_ABORTED)
            return
        # A turn in place shifts the centre a little (wheel scrub, on a real
        # robot as in Gazebo). A small shift: the exit is planned from the
        # measured pose (3D-checked, blending onto the exit line). A larger
        # one: re-centre on the station's floor marker with the camera stage
        # (10 mm, 1 degree), as AGVs do with their downward camera, first.
        station = rotation['station']
        pose = self._robot_pose()
        shift = math.hypot(pose[0] - rotation['start_pose'][0],
                           pose[1] - rotation['start_pose'][1])
        if shift <= TURN_SHIFT_TOLERANCE_M:
            self._report(f'{station}: turned in place (centre shifted {shift * 100:.1f} cm).')
            self._continue_after_turn(rotation)
            return
        self._turn_recentre = rotation
        self._precision_handoff_yaw = rotation['target']
        self._dock_precision_mode = True
        self._report(f'{station}: turned in place; the centre shifted {shift * 100:.1f} cm, '
                     're-centring on the station marker before leaving.')
        self._start_qr_alignment(station)

    def _continue_after_turn(self, rotation):
        station, gate = rotation['station'], rotation['gate']
        self._arrival_approach = (station, rotation['approach'])
        self._report(f'{station}: turn complete; leaving by the {gate} gate. Planning the '
                     'exit from the measured pose; it continues automatically.')
        self._continue_after_undock = True
        self._mission_phase = 'turn'
        self.execute_destination(*rotation['mission'])

    def _same_gate_routes(self, station, approach, candidates, destination):
        """Restrict the next leg to routes leaving by the gate used to enter."""
        gate = entrance_gate(approach)
        routes = routes_from_gate(candidates, station, gate)
        if not routes:
            if getattr(self, '_live_planner', None) is None:
                raise ValueError(
                    f'no saved route from {station} to {destination} leaves by the '
                    f'{gate} gate the robot entered by; generate one')
            # Still leave by the entry gate; the next leg is then planned
            # live from that gate with the offline generator's rules.
            self._report(f'{station}: no saved route to {destination} leaves by the '
                         f'{gate} gate the robot entered by; it leaves by that gate '
                         'and the next leg is planned live.')
        return routes

    def _infer_pass_through_dock(self, pose, tolerance=0.02):
        """An empty robot sitting on a pass-through centre (e.g. at start) is docked."""
        for station in getattr(self, '_pass_through', ()):
            center = getattr(self, '_station_centers', {}).get(station)
            if center is not None and math.hypot(
                    pose[0] - center[0], pose[1] - center[1]) <= tolerance:
                self.docked_station = station
                self._report(f'{station}: robot is on the station centre; it leaves through a gate.')
                return

    def _pass_through_exit(self, station, current_pose, candidates):
        """Choose the gate whose approach fits the robot and the next route."""
        last_error = 'no saved route leaves this station'
        for route_key, route, poses in sorted(
                candidates, key=lambda item: item[1].get('length_m', float('inf'))):
            if route.get('source') != station or not poses:
                continue
            try:
                start = route.get('markers', {}).get('route_start', 0)
                approach = self._approach_provider(station, poses[start])[1]
                plan_station_departure(current_pose, approach)
            except (OSError, ValueError, KeyError) as exc:
                last_error = f'{route_key}: {exc}'
                continue
            return approach, routes_from_gate(candidates, station,
                                              route.get('source_entrance'))
        raise ValueError(f'no gate of {station} fits the robot: {last_error}')

    def start_mission(self, destination, candidates, station_targets):
        """Operator request: a new mission with a fresh recovery budget."""
        self._mission_retry = {'destination': destination, 'candidates': candidates,
                               'targets': station_targets, 'attempts': 0}
        self._mission_confirm = None
        self._mission_via = None
        self._mission_action = None
        self.execute_destination(destination, candidates, station_targets)

    def _finish_mission(self, report_done, succeeded, status):
        """Report the outcome, or recover from a failure and try again."""
        if (succeeded and getattr(self, '_mission_via', None) is not None and
                self._motion_armed and not self._cancel_requested):
            self._continue_route_chain()
            return
        retry = getattr(self, '_mission_retry', None)
        phase = getattr(self, '_mission_phase', None)
        if (succeeded or retry is None or status == GoalStatus.STATUS_CANCELED or
                self._cancel_requested or not self._motion_armed):
            if succeeded or status == GoalStatus.STATUS_CANCELED:
                self._mission_retry = None
            self._mission_confirm = None
            self._mission_via = None
            self._mission_action = None
            report_done(succeeded, status)
            return
        if not self._recoverable(phase):
            self._mission_retry = None
            self._mission_confirm = None
            self._mission_via = None
            self._mission_action = None
            report_done(succeeded, status)
            return
        if retry['attempts'] >= MAX_MISSION_RETRIES:
            self._report(f'MISSION FAILED: {phase} still failing after '
                         f'{MAX_MISSION_RETRIES} recovery attempts. Robot stopped.')
            self._mission_retry = None
            self._mission_confirm = None
            self._mission_via = None
            self._mission_action = None
            report_done(succeeded, status)
            return
        retry['attempts'] += 1
        self._report(f'RECOVERY {retry["attempts"]}/{MAX_MISSION_RETRIES}: the {phase} stage '
                     'did not complete; recovering automatically.')
        timer = None

        def recover():
            timer.cancel()
            try:
                self._recover(phase, retry)
            except Exception as exc:
                self._report(f'MISSION FAILED: recovery could not start: {exc}.')
                self._mission_retry = None
                report_done(False, GoalStatus.STATUS_ABORTED)
        timer = self.create_timer(RECOVERY_DELAY_S, recover)

    def _recoverable(self, phase):
        if phase in ('route', 'approach', 'camera', 'turn'):
            return True
        failure = getattr(self, '_last_cargo_failure', None)
        return (phase == 'cargo' and failure is not None and failure[0] == 'pickup' and
                failure[1] in RETRYABLE_CARGO_FAILURES)

    def _recover(self, phase, retry):
        destination = retry['destination']
        if phase in ('route', 'turn'):
            # Replan from where the robot stands: rejoin, connector or live plan
            # (after a stopped turn: the turn needed from the current heading).
            self._continue_after_undock = True
            self.execute_destination(destination, retry['candidates'], retry['targets'])
            return
        if phase == 'cargo':
            # A rejected pickup means the simulator found the robot misaligned
            # with the cargo. Lower the lift, then re-verify (and if needed
            # correct) the dock with the camera before trying again; a robot
            # that is still aligned passes in a few stationary frames.
            self.lift(LIFT_DOWN_M)
            station = self.docked_station or destination
            timer = None

            def again():
                timer.cancel()
                self._start_qr_alignment(station)
            timer = self.create_timer(3.0, again)
            return
        self._prepare_backout(destination, retry['candidates'], retry['targets'])

    def _prepare_backout(self, destination, candidates, station_targets, station=None):
        """Back out along a station's approach to its gate, then continue.

        By default the station is the destination (recovery: re-approach it).
        With another station, the robot leaves a station it is partway out of
        (for example after a cancelled departure) and then heads to destination.
        """
        station = station or destination
        arrival = getattr(self, '_arrival_approach', None)
        if arrival is not None and arrival[0] == station:
            approach = arrival[1]
        elif station != destination:
            approach = self._station_approach(station)
        else:
            raise ValueError(f'no recorded approach to {destination} to back out along')
        pose = self._robot_pose()
        index = min(range(len(approach)), key=lambda i: math.hypot(
            approach[i][0] - pose[0], approach[i][1] - pose[1]))
        if math.hypot(approach[index][0] - pose[0], approach[index][1] - pose[1]) > 0.30:
            raise ValueError('robot is not on its approach')
        cargo = self._confirmed_cargo()
        if cargo is None:
            if not lift_is_down(self._lift_position):
                raise ValueError('backing out empty requires the lift to be down')
            world = Path(get_package_share_directory('tugger_gazebo')) / 'worlds/tugger_warehouse.sdf'
            scene = decode_scene(self._collision_scene,
                                 self.get_clock().now().nanoseconds / 1e9,
                                 expected_counts=scene_expected_counts(world))
        # Follow the approach back with the robot's sideways offset blended
        # out smoothly, shortest clear blend first; a straight line to the
        # approach starts with a kink the 4WS robot cannot follow from rest.
        reversed_approach = list(reversed(approach[:index + 1]))
        if math.hypot(reversed_approach[-1][0] - pose[0],
                      reversed_approach[-1][1] - pose[1]) < 0.05:
            raise ValueError('robot is already at the approach entry')
        chosen, last_hit = None, 'no blend within the 4WS turning limit'
        for _length, positions in backout_candidates(pose, reversed_approach):
            prepared = prepare_dynamic_path(positions, pose[2])
            poses = [[x, y, heading] for (x, y), heading in
                     zip(positions, prepared['body_headings'])]
            if cargo is None:
                clear, sample, hit = sampled_precision_clearance(
                    [tuple(p) for p in poses], [box for _, box in scene], self._lift_position)
                if not clear:
                    last_hit = f'collision at sample {sample}: {hit}'
                    continue
            else:
                try:
                    self._modeled_loaded_path_clearance(poses, cargo)
                except ValueError as exc:
                    last_hit = str(exc)
                    continue
            chosen = (prepared, poses)
            break
        if chosen is None:
            raise ValueError(f'no clear back-out path: {last_hit}')
        prepared, poses = chosen
        path = NavPath()
        path.header.frame_id = 'map'
        path.header.stamp = self.get_clock().now().to_msg()
        for values in poses:
            path.poses.append(self._pose(values))
        self._precision_station = None
        self._camera_handoff_station = None
        self._queued_precision_poses = None
        self._queued_precision_station = None
        self._departure_exit_pose = tuple(poses[-1])
        self._pending_pose = pose
        self._pending_destination = destination
        self._pending_execution_approved = True
        if cargo is None:
            self._dock_precision_mode = True
            self._station_exit_active = True
            self._station_exit_target = (destination, candidates, station_targets)
            self._pending_path = path
        else:
            self._pending_departure = {
                'path': path, 'poses': poses, 'start_pose': pose,
                'cargo_id': cargo, 'station': station,
            }
            self._departure_target = (destination, candidates, station_targets)
        self._path_pub.publish(path)
        if station == destination:
            self._report(f'RECOVERY: backing out {prepared["leading_end"].lower()}-leading along '
                         f'the approach to the {destination} entry, then re-approaching.')
        else:
            self._report(f'{station}: robot is partway out of the station; backing out '
                         f'{prepared["leading_end"].lower()}-leading along its approach to the '
                         f'gate, then continuing to {destination}.')
        self._continue_attempt(1)

    def _prepare_station_exit(self, station, current_pose, destination, candidates,
                              station_targets):
        """Preview an empty robot's exit along the approach it arrived on."""
        if station in getattr(self, '_pass_through', ()):
            approach, candidates = self._pass_through_exit(station, current_pose, candidates)
        else:
            approach = self._station_approach(station)
            candidates = self._same_gate_routes(station, approach, candidates, destination)
        if not self._loaded_guard_is_ready():
            raise ValueError('Live simulator collision-scene guard is not ready')
        if not lift_is_down(self._lift_position):
            raise ValueError('Empty station exit requires the lift to be down')
        decision = plan_station_departure(current_pose, approach)
        world = Path(get_package_share_directory('tugger_gazebo')) / 'worlds/tugger_warehouse.sdf'
        if self._collision_scene is None:
            raise ValueError('simulator collision scene is unavailable')
        scene = decode_scene(self._collision_scene,
                             self.get_clock().now().nanoseconds / 1e9,
                             expected_counts=scene_expected_counts(world))
        clear, sample, hit = sampled_precision_clearance(
            [tuple(current_pose)] + [tuple(pose) for pose in decision['poses']],
            [box for _, box in scene], self._lift_position)
        if not clear:
            raise ValueError(f'station exit collision at sample {sample}: {hit}')
        path = NavPath()
        path.header.frame_id = 'map'
        path.header.stamp = self.get_clock().now().to_msg()
        for pose in decision['poses']:
            path.poses.append(self._pose(pose))
        self._precision_station = None
        self._camera_handoff_station = None
        self._queued_precision_poses = None
        self._queued_precision_station = None
        self._dock_precision_mode = True
        self._station_exit_active = True
        self._station_exit_target = (destination, candidates, station_targets)
        self._departure_exit_pose = tuple(decision['staging_pose'])
        self._pending_path = path
        self._pending_pose = current_pose
        self._pending_destination = destination
        self._pending_execution_approved = True
        self._path_pub.publish(path)
        self._plan_ready(
            f'{station}: EMPTY STATION EXIT preview to staging '
            f'({decision["length_m"]:.2f} m, {decision["leading_end"]}-leading, '
            f'{sample} clearance samples). Press RUN to exit; the route to '
            f'{destination} is planned from the measured exit pose and continues '
            'automatically.', False)

    def _complete_station_exit(self, succeeded, status):
        """Stop owning motion, then plan the next leg from the measured pose."""
        target = self._station_exit_target
        self._station_exit_active = False
        self._station_exit_target = None
        self._dock_precision_mode = False
        if not succeeded or target is None or not self._motion_armed:
            self._report('Empty station exit did not complete; robot stopped.')
            self._execution_done(False, status)
            return
        try:
            if not plan_is_current(self._departure_exit_pose, self._robot_pose(),
                                   position_tolerance_m=0.15)['current']:
                raise ValueError('robot did not reach the expected free-space staging pose')
        except Exception as exc:
            self._report(f'Empty station exit handoff blocked: {exc}')
            self._execution_done(False, GoalStatus.STATUS_ABORTED)
            return
        self.docked_station = None
        self._report('Empty station exit completed at staging. Planning the next leg '
                     'from the measured pose; it continues if every check passes.')
        self._continue_after_undock = True
        self.execute_destination(*target)

    def _on_departure_guard_ready(self):
        """Wait for a new scene after DOCK_DEPART authorization."""
        pending = self._departure_pending
        if pending is None or pending['generation'] != self._plan_generation:
            return
        if pending.get('awaiting_fresh_scene'):
            return
        pending['awaiting_fresh_scene'] = True
        pending['scene_sequence_at_guard'] = getattr(self, '_scene_sequence', 0)
        self._report(
            'Loaded departure guard ready; awaiting a new live collision scene.'
        )

    def _validate_departure_on_fresh_scene(self):
        """Authorize only against a newly consumed, still-fresh scene."""
        pending = self._departure_pending

        if (
            pending is None
            or pending['generation'] != self._plan_generation
            or not pending.get('awaiting_fresh_scene')
            or getattr(self, '_scene_sequence', 0)
                <= pending['scene_sequence_at_guard']
        ):
            return

        try:
            if not self._motion_armed or self._cancel_requested:
                raise ValueError('simulator commands disarmed or cancelled')

            guard_seen = pending.get('guard_seen_at')
            if (
                guard_seen is None
                or not 0 <= (
                    self.get_clock().now() - guard_seen
                ).nanoseconds / 1e9 <= 0.3
            ):
                return

            if self._collision_scene is None:
                return

            stamp = json.loads(self._collision_scene).get('stamp')
            now = self.get_clock().now().nanoseconds / 1e9

            if (
                not isinstance(stamp, (float, int))
                or not math.isfinite(stamp)
                or not 0 <= now - stamp <= 0.4
            ):
                return

            if not plan_is_current(
                pending['start_pose'],
                self._robot_pose(),
                position_tolerance_m=0.10
            )['current']:
                raise ValueError(
                    'robot pose changed during departure guard handshake'
                )

            if self._confirmed_cargo() != pending['cargo_id']:
                raise ValueError('attached cargo state changed')

            if not self._dock_follower.server_is_ready():
                raise ValueError('independent docking controller unavailable')

            # Preserve the full loaded robot/cargo collision check.
            self._modeled_loaded_path_clearance(
                pending['poses'], pending['cargo_id']
            )

            # The scene must still be fresh after the calculation.
            age = self.get_clock().now().nanoseconds / 1e9 - stamp
            if not 0 <= age <= 0.4:
                return

            self._departure_pending = None
            self._departure_active = True

            self._report(
                'Loaded departure: fresh scene and clearance passed; '
                'starting guarded departure.'
            )
            self._follow(pending['path'])

        except ValueError as exc:
            if str(exc) == (
                'Simulator scene is stale or from a future clock epoch'
            ):
                return
            self._fail_loaded_departure(exc)

        except Exception as exc:
            self._fail_loaded_departure(exc)

    def _fail_loaded_departure(self, exc):
        self._departure_pending = None
        self._departure_active = False
        self._departure_target = None
        self._report(f'Loaded departure blocked: {exc}. Robot stopped.')
        self._execution_done(False, GoalStatus.STATUS_ABORTED)

    def _prepare_loaded_departure(self, station, cargo_id, current_pose,
                                  destination, candidates, station_targets):
        """Prepare a loaded exit from the station's own staged approach path."""
        if self._approach_provider is None:
            raise ValueError('No configured station approach is available for departure')
        if not self._loaded_guard_is_ready():
            raise ValueError('Live simulator collision-scene guard is not ready')
        if station in getattr(self, '_pass_through', ()):
            # A pass-through stop is left by whichever gate fits the robot.
            approach_poses, candidates = self._pass_through_exit(
                station, current_pose, candidates)
        else:
            approach_meta, default_poses = self._approach_provider(station)
            if approach_meta.get('execution_approved') is not True:
                raise ValueError('Station departure source approach is not execution-approved')
            arrival = getattr(self, '_arrival_approach', None)
            approach_poses = (arrival[1] if arrival is not None and arrival[0] == station
                              else default_poses)
            candidates = self._same_gate_routes(station, approach_poses, candidates,
                                                destination)
        if candidates and not any(route.get('execution_approved') is True and
                                  route.get('_database_execution_approved') is True
                                  for _, route, _ in candidates):
            raise ValueError('No approved outgoing route exists for this destination')
        decision = plan_station_departure(current_pose, approach_poses)
        samples = self._modeled_loaded_path_clearance(decision['poses'], cargo_id)
        # Leaving the station is neither an arriving precision segment nor a
        # camera-alignment action. Clear stale state from the completed pickup.
        self._precision_station = None
        self._camera_handoff_station = None
        self._queued_precision_poses = None
        self._queued_precision_station = None
        self._dock_precision_mode = False
        path = NavPath()
        path.header.frame_id = 'map'
        path.header.stamp = self.get_clock().now().to_msg()
        for pose in decision['poses']:
            path.poses.append(self._pose(pose))
        self._pending_departure = {
            'path': path, 'poses': decision['poses'], 'start_pose': current_pose,
            'cargo_id': cargo_id, 'station': station,
        }
        self._departure_target = (destination, candidates, station_targets)
        self._departure_exit_pose = tuple(decision['staging_pose'])
        self._pending_execution_approved = True
        self._pending_pose = current_pose
        self._pending_destination = destination
        self._path_pub.publish(path)
        self._plan_ready(
            f'{station}: LOADED DEPARTURE preview to staging '
            f'({decision["length_m"]:.2f} m, {decision["leading_end"]}-leading, '
            f'{samples} loaded-clearance samples). Press RUN to depart; '
            f'the route to {destination} is planned from the measured exit pose and '
            f'continues automatically.', False)

    def _complete_loaded_departure(self):
        """Return command ownership to STOP before planning from actual exit."""
        self._departure_active = False
        target = self._departure_target
        self._departure_target = None
        if target is None or not self._motion_armed or self._cancel_requested:
            self._report('Loaded departure ended without a valid continuation.')
            self._execution_done(False, GoalStatus.STATUS_ABORTED)
            return
        try:
            current = self._robot_pose()
            if not plan_is_current(
                    self._departure_exit_pose, current,
                    position_tolerance_m=0.15)['current']:
                raise ValueError('robot did not reach the expected free-space staging pose')
            cargo = self._confirmed_cargo()
            if cargo is None:
                raise ValueError('cargo attachment was lost during departure')
            # Validation includes the measured final pose and the attached cargo.
            end = (current[0] + 0.01 * math.cos(current[2]),
                   current[1] + 0.01 * math.sin(current[2]), current[2])
            self._modeled_loaded_path_clearance([current, end], cargo)
        except Exception as exc:
            self._report(f'Loaded departure completed but navigation handoff blocked: {exc}')
            self._execution_done(False, GoalStatus.STATUS_ABORTED)
            return
        self.docked_station = None
        self._report('Loaded departure completed at staging. Planning the next leg '
                     'from the measured pose; it continues if every check passes.')
        self._continue_after_undock = True
        self.execute_destination(*target)

    def _loaded_guard_is_ready(self):
        if not self._loaded_control_ready or self._loaded_guard_seen_at is None:
            return False
        age = (self.get_clock().now() - self._loaded_guard_seen_at).nanoseconds / 1e9
        return 0 <= age <= 0.3

    def _scene_update(self, message):
        self._collision_scene = message.data
        self._scene_sequence = getattr(self, '_scene_sequence', 0) + 1
        if self._departure_pending is not None:
            self._validate_departure_on_fresh_scene()

    def _guard_status(self, message):
        if message.data == self._last_guard_status:
            return
        self._last_guard_status = message.data
        text = guard_status_text(message.data)
        if text is None:
            return
        # Slowing near the same object is reported once per interval.
        now = self.get_clock().now().nanoseconds / 1e9
        notes = getattr(self, '_guard_notes', {})
        self._guard_notes = notes
        last = notes.get(text)
        if last is not None and 0 <= now - last < GUARD_NOTE_INTERVAL_S:
            return
        notes[text] = now
        self._report(text)

    def _on_safety_state(self, message):
        try:
            state = json.loads(message.data)
        except ValueError:
            return
        previous = getattr(self, '_safety_state', None)
        self._safety_state = state
        if previous is None or previous.get('state') != state.get('state'):
            if state.get('state') in ('PROTECTIVE_STOP', 'EMERGENCY_STOP') or (
                    previous is not None and previous.get('state') in (
                        'PROTECTIVE_STOP', 'EMERGENCY_STOP')):
                self._report(f'SAFETY {state.get("state")}: {state.get("reason") or "clear"}.')

    def emergency_stop(self, pressed):
        """Panel emergency stop: latched in the safety monitor until reset."""
        message = Bool()
        message.data = bool(pressed)
        self._estop_pub.publish(message)
        if pressed:
            self._report('EMERGENCY STOP pressed on the panel.')

    def reset_safety(self):
        """Release the panel e-stop and reset; refused while a cause is active."""
        self.emergency_stop(False)
        self._safety_reset_pub.publish(Empty())
        self._report('Safety reset requested (refused while an e-stop or the bumper is active).')

    def _publish_motion_owner(self):
        owner = 'STOP'
        if self._motion_armed and not self._cancel_requested:
            if getattr(self, '_rotation', None) is not None:
                owner = 'ROTATE'
            elif self._departure_pending is not None or self._departure_active:
                owner = 'DOCK_DEPART'
            elif self._qr_alignment_active:
                owner = 'DOCK_CAMERA' if self._dock_precision_mode else 'PRECISION'
            elif self._dock_precision_mode and (self._stage_validation_pending or
                                                self._follow_goal_pending or
                                                self._active_follow is not None):
                owner = 'DOCK_NAV'
            elif self._follow_goal_pending or self._active_follow is not None:
                owner = 'NAV'
        message = String()
        message.data = owner
        self._motion_owner_pub.publish(message)

    def _qr_id(self, message):
        if (self._qr_alignment_active and
                message.data != expected_marker(self._qr_alignment_station)):
            self._qr_heading_reference = None
        self._qr_identity = message.data
        self._qr_seen_at = self.get_clock().now()
        if (self._qr_alignment_active and
                message.data == expected_marker(self._qr_alignment_station)):
            self._qr_expected_seen_at = self._qr_seen_at

    def _qr_heading_pose(self, message):
        """Record QR image yaw only if its stamped image and frame are valid."""
        if not self._qr_alignment_active:
            return
        stamp = message.header.stamp.sec + message.header.stamp.nanosec / 1e9
        age = self.get_clock().now().nanoseconds / 1e9 - stamp
        if message.header.frame_id != 'downward_camera_link' or not 0 <= age <= 0.5:
            return
        yaw = _yaw(message.pose.orientation)
        if math.isfinite(yaw) and (self._qr_heading_stamp is None or
                                   stamp > self._qr_heading_stamp):
            self._qr_heading_angle = yaw
            self._qr_heading_stamp = stamp

    def _qr_heading_agrees(self, pose, frame_stamp):
        """Require a fresh QR yaw to track the map yaw from a stable reference."""
        if (self._qr_heading_stamp is None or self._qr_heading_angle is None or
                not 0 <= frame_stamp - self._qr_heading_stamp <= 0.25):
            return False
        if self._qr_heading_reference is None:
            self._qr_heading_reference = (self._qr_heading_angle, pose[2])
            return True
        return relative_qr_heading_consistent(
            self._qr_heading_angle, self._qr_heading_reference[0],
            pose[2], self._qr_heading_reference[1])

    def _cargo_state(self, station, message):
        """Report observed simulator attachment transitions, not commands."""
        state = message.data.strip().lower()
        if station not in self._cargo_states:
            return
        if state in ('attach_rejected', 'detach_rejected'):
            self._report(f'CARGO {state.upper()}: simulator rejected the request for station {station}.')
            if self._cargo_task is not None and self._cargo_task.cargo_id == station:
                self._advance_cargo_operation(rejection=state)
            return
        if state not in ('attached', 'detached'):
            return
        self._cargo_seen_at[station] = self.get_clock().now()
        if self._cargo_states[station] == state:
            self._advance_cargo_operation()
            return
        self._cargo_states[station] = state
        label = '1' if station == 0 else '3'
        self._report(f'CARGO JOINT {state.upper()}: Station {label} cargo (cargo estimator).')
        self._advance_cargo_operation()

    def _joint_state(self, message):
        """Observe the lift; publishing a lift command is not confirmation."""
        try:
            index = message.name.index('lift_joint')
            position = float(message.position[index])
            if not math.isfinite(position):
                return
        except (ValueError, IndexError, TypeError):
            return
        self._lift_position = position
        self._lift_seen_at = self.get_clock().now()
        if not getattr(self, '_lift_hold_sent', False):
            # Hold the lift from the first reading. An uncommanded lift can be
            # shaken onto its lower limit while driving, where Gazebo latches it.
            self._lift_hold_sent = True
            self.lift(max(position, LIFT_DOWN_M))
        self._advance_cargo_operation()

    def _cargo_pose(self, station, message):
        """Record the estimated cargo pose for clearance."""
        if station not in self._cargo_poses or message.header.frame_id != 'map':
            return
        pose = message.pose
        values = (pose.position.x, pose.position.y, pose.position.z, _yaw(pose.orientation))
        if not all(math.isfinite(value) for value in values):
            return
        self._cargo_poses[station] = values
        self._cargo_pose_seen_at[station] = self.get_clock().now()

    def _modeled_lift_clearance(self, station, cargo_id, operation):
        """Check a fresh simulator collision scene and current payload poses."""
        now = self.get_clock().now()
        for key in CARGO_IDS:
            seen = self._cargo_pose_seen_at[key]
            if seen is None or not 0 <= (now - seen).nanoseconds / 1e9 <= 1.0:
                raise ClearanceUnavailable('estimated cargo pose is stale or missing')
        try:
            world = Path(get_package_share_directory('tugger_gazebo')) / 'worlds/tugger_warehouse.sdf'
            robot = Path(get_package_share_directory('tugger_description')) / 'urdf/tugger.urdf.xacro'
            validate_modeled_geometry(world, robot)
            if self._collision_scene is None:
                raise ValueError('live simulator collision scene is unavailable')
            scene = decode_scene(self._collision_scene, now.nanoseconds / 1e9,
                                 expected_counts=scene_expected_counts(world))
            obstacles = [box for model, box in scene if model not in
                         tuple(cargo_stand_model(key) for key in CARGO_IDS)]
            other = other_cargo(cargo_id)
            target = LIFT_UP_M if operation == 'pickup' else LIFT_DOWN_M
            cargo_pose = self._cargo_poses[cargo_id]
            task = getattr(self, '_cargo_task', None)
            rest = getattr(task, 'rest_z', None) if task is not None else None
            if rest is not None and task.cargo_id == cargo_id:
                # The cargo's height above its resting height follows the lift.
                # The estimate and the lift reading arrive at slightly
                # different instants; while the lift moves, combining them
                # made a few millimetres of false "penetration" into a leg.
                # The resting height, latched with the lift still, is exact.
                cargo_pose = (cargo_pose[0], cargo_pose[1],
                              rest + cargo_lift_offset(self._lift_position), cargo_pose[3])
            clear, hit = lift_sweep_clear(
                self._robot_pose(), cargo_pose, self._cargo_poses[other],
                obstacles, self._lift_position, target)
            if not clear:
                self._report(f'Modeled cargo lift clearance failed: {hit}.')
            return clear
        except (OSError, ValueError, TypeError, KeyError) as exc:
            raise ClearanceUnavailable(str(exc)) from exc

    def _modeled_loaded_path_clearance(self, poses, cargo_id):
        """Sample the known world with current payload geometry; not a sensor safety layer."""
        now = self.get_clock().now()
        for key in CARGO_IDS:
            seen = self._cargo_pose_seen_at[key]
            if seen is None or not 0 <= (now - seen).nanoseconds / 1e9 <= 1.0:
                raise ValueError('estimated cargo pose is stale or missing')
        if (self._lift_seen_at is None or
                not 0 <= (now - self._lift_seen_at).nanoseconds / 1e9 <= 1.0 or
                not lift_is_raised(self._lift_position)):
            raise ValueError('raised lift feedback is stale or outside its tolerance')
        world = Path(get_package_share_directory('tugger_gazebo')) / 'worlds/tugger_warehouse.sdf'
        robot = Path(get_package_share_directory('tugger_description')) / 'urdf/tugger.urdf.xacro'
        validate_modeled_geometry(world, robot)
        current = self._robot_pose()
        other = other_cargo(cargo_id)
        if self._collision_scene is None:
            raise ValueError('live simulator collision scene is unavailable')
        scene = decode_scene(self._collision_scene, now.nanoseconds / 1e9,
                             expected_counts=scene_expected_counts(world))
        obstacles = [box for model, box in scene if model not in
                     tuple(cargo_stand_model(key) for key in CARGO_IDS)]
        clear, sample, hit = loaded_path_clear(
            [current] + list(poses), current, self._cargo_poses[cargo_id],
            self._cargo_poses[other], obstacles, self._lift_position)
        if not clear:
            raise ValueError(f'modeled loaded path intersects {hit} at sample {sample}')
        return sample

    def _confirmed_cargo(self):
        now = self.get_clock().now()
        ages = {
            station: ((now - seen).nanoseconds / 1e9 if seen is not None else None)
            for station, seen in self._cargo_seen_at.items()
        }
        return confirmed_cargo(self._cargo_states, ages)

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

    def _hold_visual_alignment(self, reason):
        """Stop immediately and request a new measured plan after reacquisition."""
        self._qr_aligned_frames = 0
        self._red_frame_aligned = False
        self._visual_executor = None
        self._visual_replan_requested = True
        # Invalidate an already-submitted plan after any hold.
        self._visual_plan_generation += 1
        with self._visual_plan_lock:
            old_future = self._visual_plan_future
            self._visual_plan_future = None
        if old_future is not None:
            old_future.cancel()
        self._publish_precision_stop()
        # Avoid GUI floods from repeated camera observations.
        now = self.get_clock().now()
        started = getattr(self, '_qr_alignment_started_at', None)
        if (reason == 'waiting for the expected QR identity' and started is not None and
                0 <= (now - started).nanoseconds / 1e9 < QR_IDENTITY_GRACE_S):
            return  # Normal first moment of every alignment, not worth a line.
        previous = getattr(self, '_last_visual_hold_report_at', None)
        if previous is None or (now - previous).nanoseconds / 1e9 > 1.0:
            self._last_visual_hold_report_at = now
            self._report(f'Visual alignment HOLD (zero velocity): {reason}.')

    def _alignment_heading_error(self, pose=None):
        """Map TF heading against the approved final docking orientation."""
        reference = self._alignment_target_yaw
        if reference is None or not math.isfinite(reference):
            raise ValueError('no approved station-frame heading for final alignment')
        if pose is None:
            pose = self._robot_pose()
        return math.atan2(math.sin(reference - pose[2]),
                          math.cos(reference - pose[2]))

    def _red_frame_received(self, message):
        """Hold the newest camera frame until TF covers its stamp.

        The camera can run up to a few tens of ms ahead of TF. Pairing a frame
        with the newest TF instead mixes two instants, which misplaces the
        marker in the map while the robot turns (tf2 MessageFilter pattern).
        """
        if not self._qr_alignment_active:
            return
        self._red_frame_waiting = message
        self._release_red_frame()

    def _release_red_frame(self):
        message = self._red_frame_waiting
        if message is None:
            return
        stamp = rclpy.time.Time.from_msg(message.header.stamp)
        waited = (self.get_clock().now() - stamp).nanoseconds / 1e9
        if not self._tf.can_transform('map', 'base_footprint', stamp) and waited < 0.2:
            return
        # Released either with TF at its stamp or, after 0.2 s, to the
        # staleness checks of the handler, which then hold the alignment.
        self._red_frame_waiting = None
        self._red_frame_pose(message)

    def _robot_odom_pose(self):
        """Robot pose in the continuous odom frame, for short-term tracking."""
        transform = self._tf.lookup_transform('odom', 'base_footprint', rclpy.time.Time())
        translation = transform.transform.translation
        return translation.x, translation.y, _yaw(transform.transform.rotation)

    def _robot_pose_at(self, stamp):
        """Robot pose at a message stamp; fails rather than mixing instants."""
        frozen = getattr(self, '_camera_map_odom', None)
        if frozen is not None:
            transform = self._tf.lookup_transform(
                'odom', 'base_footprint', rclpy.time.Time.from_msg(stamp))
            t = transform.transform.translation
            return _compose_pose(frozen, (t.x, t.y, _yaw(transform.transform.rotation)))
        transform = self._tf.lookup_transform(
            'map', 'base_footprint', rclpy.time.Time.from_msg(stamp))
        translation = transform.transform.translation
        return translation.x, translation.y, _yaw(transform.transform.rotation)

    def _red_frame_pose(self, message):
        """Fresh red-frame centre + station yaw gate; no blind camera recovery."""
        if not self._qr_alignment_active:
            return
        station = self._qr_alignment_station
        now = self.get_clock().now()
        stamp = message.header.stamp.sec + message.header.stamp.nanosec / 1e9
        age = now.nanoseconds / 1e9 - stamp
        if not math.isfinite(stamp) or not 0 <= age <= QR_ALIGNMENT_MAX_OBSERVATION_AGE_S:
            self._hold_visual_alignment('invalid/stale red-frame timestamp')
            return
        if self._red_frame_stamp is not None and stamp <= self._red_frame_stamp:
            self._hold_visual_alignment('repeated red-frame observation')
            return
        if message.header.frame_id != 'base_footprint':
            self._stop_qr_alignment('red-frame geometry has an incorrect coordinate frame')
            return
        self._red_frame_stamp = stamp
        self._red_frame_seen_at = now
        # Camera publishes frame pose BEFORE the QR ID from each image. Hold
        # for its subsequent identified frame; never drive from anonymous red.
        if not self._expected_qr_is_fresh(station):
            self._hold_visual_alignment('waiting for the expected QR identity')
            return
        marker_x, marker_y = float(message.pose.position.x), float(message.pose.position.y)
        if not all(math.isfinite(v) for v in (marker_x, marker_y)):
            self._hold_visual_alignment('invalid frame geometry')
            return
        try:
            map_pose = self._robot_pose_at(message.header.stamp)
        except Exception as exc:
            self._hold_visual_alignment(f'no TF at the camera frame time: {exc}')
            return
        if not getattr(self, '_camera_anchored', True):
            map_pose = self._anchor_camera_frame(station, map_pose, (marker_x, marker_y))
        samples = getattr(self, '_camera_yaw_samples', None)
        if samples is not None and getattr(self, '_qr_heading_angle', None) is not None:
            map_pose = self._anchor_camera_heading(station, map_pose, message.header.stamp)
        try:
            heading_error = self._alignment_heading_error(map_pose)
        except Exception as exc:
            self._stop_qr_alignment(f'station heading unavailable: {exc}')
            return
        if not self._qr_heading_agrees(map_pose, stamp):
            self._hold_visual_alignment('QR relative yaw missing or disagrees with map TF')
            return
        if self._visual_marker_guard is None:
            self._visual_marker_guard = FixedMarkerGuard(
                map_pose, (marker_x, marker_y))
        integrity, reason = self._visual_marker_guard.check(
            map_pose, (marker_x, marker_y), self._alignment_target_yaw)
        if integrity == 'ABORT':
            self._stop_qr_alignment(reason)
            return
        if integrity == 'HOLD':
            self._visual_camera_outliers += 1
            self._hold_visual_alignment(reason)
            if self._visual_camera_outliers >= 3:
                self._stop_qr_alignment('three inconsistent camera measurements')
            return
        self._visual_camera_outliers = 0
        self._red_frame_error = (marker_x, marker_y)
        self._visual_latest_marker = (marker_x, marker_y)
        self._visual_latest_camera_stamp = stamp
        # Do not continue executing a primitive sequence against an old
        # camera target. The marker moves in the robot frame as the robot
        # moves, so compare where it is in the map: a fixed marker stays put,
        # and a map-frame change means the measured target itself changed
        # since the accepted plan. Holding here also guarantees that the next
        # plan starts only after the robot is stationary.
        if (self._visual_executor is not None and
                self._visual_plan_marker is not None and
                self._visual_plan_origin is not None and
                math.dist(_marker_in_map(map_pose, (marker_x, marker_y)),
                          _marker_in_map(self._visual_plan_origin,
                                         self._visual_plan_marker)) >
                QR_VISUAL_REPLAN_MARKER_DELTA_M):
            self._hold_visual_alignment(
                'camera target changed during maneuver; replanning')
            return
        radial_error = math.hypot(marker_x, marker_y)
        anchor_error = math.hypot(
            map_pose[0] - self._visual_marker_guard.marker[0],
            map_pose[1] - self._visual_marker_guard.marker[1])
        # Cancel the old arc sequence near the target before settling, but
        # only once the heading is aligned too. A heading correction passes
        # through the target position; cancelling it there on position alone
        # restarted it every frame, so the robot never turned.
        if (self._visual_executor is not None and
                early_stop_inside_target(radial_error, anchor_error, heading_error)):
            self._visual_executor = None
            self._visual_replan_requested = True
            self._publish_precision_stop()
            self._qr_aligned_frames = 0
            return
        if (radial_error <= QR_ALIGNMENT_ACCEPT_M and
                anchor_error <= QR_ALIGNMENT_ACCEPT_M and
                abs(heading_error) <= QR_ALIGNMENT_HEADING_TOLERANCE_RAD):
            self._publish_precision_stop()
            if not self._alignment_is_stationary(now):
                self._qr_aligned_frames = 0
                return
            self._qr_aligned_frames += 1
            if self._qr_aligned_frames >= QR_ALIGNMENT_STABLE_FRAMES:
                self._red_frame_aligned = True
                self._complete_red_frame_alignment()
            return
        self._qr_aligned_frames = 0
        self._red_frame_aligned = False
        self._red_frame_reported = False
        self._report_red_frame_state(marker_x, marker_y, 'CORRECT')
        self._visual_alignment_step(now)

    def _visual_alignment_step(self, now):
        """Guard against scene, camera and odometry loss before each command."""
        self._poll_visual_plan()
        if (not self._motion_armed or self._cancel_requested):
            self._stop_qr_alignment('motion disarmed or cancelled')
            return
        if not self._expected_qr_is_fresh(self._qr_alignment_station):
            self._hold_visual_alignment('QR identity expired')
            return
        if not self._loaded_guard_is_ready():
            self._stop_qr_alignment('collision-scene guard is not ready')
            return
        if self._visual_executor is None:
            if self._visual_plan_future is None and self._visual_replan_requested:
                if self._alignment_is_stationary(now):
                    self._request_visual_plan()
                else:
                    self._publish_precision_stop()
            else:
                self._publish_precision_stop()
            return
        try:
            pose = self._robot_pose()
            # Primitives are tracked in odom, which is continuous (REP 105);
            # the map pose can jump when localization corrects itself.
            odom_pose = self._robot_odom_pose()
            # Speed over ground: a crab move is mostly sideways.
            speed = (math.hypot(self._alignment_odom[0], self._alignment_odom[1])
                     if self._alignment_odom is not None else float('nan'))
            stamp = self._alignment_odom_stamp
            if not self._visual_clearance_current(pose):
                self._stop_qr_alignment('live footprint clearance failed')
                return
            if (stamp is None or not 0 <= now.nanoseconds / 1e9 - stamp <=
                    QR_ALIGNMENT_MAX_OBSERVATION_AGE_S):
                self._hold_visual_alignment('odometry stale; waiting to replan')
                return
            result = self._visual_executor.update(
                odom_pose, speed, now.nanoseconds / 1e9, stamp,
                self._visual_latest_camera_stamp, identity_ok=True,
                clearance_ok=True, visible=True)
        except Exception as exc:
            self._hold_visual_alignment(f'executor feedback unavailable: {exc}')
            return
        if result['state'] in ('STOP', 'WAIT_STOP', 'DONE'):
            self._publish_precision_stop()
        if result['state'] == 'STOP':
            reason = result.get('reason', 'executor stopped')

            if reason == 'odometry diverged from planned primitive':
                self._stop_qr_alignment(
                    '4WS tracking diverged; diagnostic stop without replanning '
                    f'({result.get("detail", "no detail")})'
                )
            else:
                self._hold_visual_alignment(reason)
        elif result['state'] == 'DONE':
            # A completed maneuver is NOT alignment success. Recheck real
            # stationary camera position and map heading before cargo pickup.
            self._visual_executor = None
            self._visual_replan_requested = True
        elif result['state'] == 'MOVING':
            command = TwistStamped()
            command.header.stamp = self.get_clock().now().to_msg()
            # 'crab' for a crab move (all wheels at one angle).
            command.header.frame_id = result.get('frame', 'base_footprint')
            command.twist.linear.x = result['linear_x']
            command.twist.linear.y = result.get('linear_y', 0.0)
            command.twist.angular.z = result['angular_z']
            self._precision_cmd_pub.publish(command)

    def _request_visual_plan(self):
        """Snapshot fresh geometry and run only motion-free search in a worker."""
        marker = self._visual_latest_marker
        self._publish_precision_stop()
        if marker is None:
            return
        # Plan only once the heading is anchored on the floor QR (or that has
        # timed out): a heading anchor after planning would move the marker in
        # the frame and invalidate the plan.
        if getattr(self, '_camera_yaw_samples', None) is not None:
            started = getattr(self, '_qr_alignment_started_at', None)
            if started is not None and (self.get_clock().now() - started).nanoseconds / 1e9 < 2.0:
                return
            self._camera_yaw_samples = None   # no QR heading in time: keep localization
        try:
            now = self.get_clock().now().nanoseconds / 1e9
            pose = self._robot_pose()
            heading_error = self._alignment_heading_error(pose)
            if not self._loaded_guard_is_ready():
                raise ValueError('scene guard heartbeat is stale')
            if self._collision_scene is None:
                raise ValueError('simulator collision scene unavailable')
            world = Path(get_package_share_directory('tugger_gazebo')) / 'worlds' / 'tugger_warehouse.sdf'
            scene = decode_scene(self._collision_scene, now,
                                 expected_counts=scene_expected_counts(world))
            lift = self._lift_position
            cargo = self._confirmed_cargo()
            lift_fresh = (self._lift_seen_at is not None and 0 <= (
                self.get_clock().now() - self._lift_seen_at).nanoseconds / 1e9 <= 1.0)
            if cargo is None and (lift is None or not -0.005 <= lift <= 0.005 or not lift_fresh):
                raise ValueError('fresh lowered lift feedback required')
            if cargo is not None and (not lift_is_raised(lift) or not lift_fresh):
                raise ValueError('fresh raised lift feedback required with cargo attached')
            generation = self._visual_plan_generation
            marker_x, marker_y = marker
            carried = None
            excluded = ()
            if cargo is not None:
                # Plan with the carried cargo; its own stand moves with the robot.
                carried = self._carried_offset(cargo)
                excluded = (cargo_stand_model(cargo),)
            obstacle_snapshot = tuple(box for model, box in scene if model not in excluded)
            # Indexed collision broad-phase retains EXACT oriented-box SAT
            # collision behaviour, without checking all 68 boxes every time.
            prepared = PreparedVisualClearance(obstacle_snapshot, lift, margin=0.01,
                                               carried=carried)
            c, s = math.cos(pose[2]), math.sin(pose[2])

            def clearance(local_pose):
                world_pose = (pose[0] + c * local_pose[0] - s * local_pose[1],
                              pose[1] + s * local_pose[0] + c * local_pose[1],
                              math.atan2(math.sin(pose[2] + local_pose[2]),
                                         math.cos(pose[2] + local_pose[2])))
                return prepared(world_pose)

            future = self._visual_plan_pool.submit(
                replan_visual_maneuver, (0.0, 0.0, 0.0),
                (marker_x, marker_y, heading_error), clearance,
                max_depth=8, beam_width=400, time_budget=3.0,
                position_tolerance=QR_ALIGNMENT_ACCEPT_M,
                heading_tolerance=QR_ALIGNMENT_HEADING_TOLERANCE_RAD,
                require_full_visibility=False)
            with self._visual_plan_lock:
                self._visual_plan_future = future
                self._visual_plan_error = None
                self._visual_plan_result = None
                self._visual_plan_origin = pose
                self._visual_plan_marker = marker
                self._visual_plan_generation_at_submit = generation
            self._visual_replan_requested = False
            pass  # Routine; not reported to the operator.
        except Exception as exc:
            self._stop_qr_alignment(f'visual plan preflight failed: {exc}')

    def _poll_visual_plan(self):
        with self._visual_plan_lock:
            future = self._visual_plan_future
            if future is None or not future.done():
                return
            self._visual_plan_future = None
            try:
                plan = future.result()
                error = None
            except Exception as exc:
                plan, error = None, str(exc)
        if self._visual_plan_generation_at_submit != self._visual_plan_generation:
            return
        if plan is None:
            self._visual_plan_failures += 1
            if self._visual_plan_failures >= QR_VISUAL_MAX_PLAN_FAILURES:
                self._stop_qr_alignment(
                    error or 'guarded 4WS search has no feasible camera maneuver; '
                    'Nav2 must deliver closer before visual alignment')
            else:
                self._hold_visual_alignment(
                    f'plan unavailable ({self._visual_plan_failures}/'
                    f'{QR_VISUAL_MAX_PLAN_FAILURES})')
            return
        camera_stamp = self._visual_latest_camera_stamp
        now_seconds = self.get_clock().now().nanoseconds / 1e9
        if (self._visual_latest_marker is None or camera_stamp is None or
                not 0 <= now_seconds - camera_stamp <= QR_ALIGNMENT_MAX_OBSERVATION_AGE_S or
                not self._expected_qr_is_fresh(self._qr_alignment_station)):
            self._hold_visual_alignment('plan expired while QR was unobserved')
            return
        try:
            current = self._robot_pose()
        except Exception:
            self._hold_visual_alignment('TF missing after planning')
            return
        if (self._visual_plan_origin is None or
                math.hypot(current[0] - self._visual_plan_origin[0],
                           current[1] - self._visual_plan_origin[1]) > 0.01 or
                abs(math.atan2(math.sin(current[2] - self._visual_plan_origin[2]),
                               math.cos(current[2] - self._visual_plan_origin[2]))) > 0.02 or
                self._visual_plan_marker is None or
                math.hypot(self._visual_latest_marker[0] - self._visual_plan_marker[0],
                           self._visual_latest_marker[1] - self._visual_plan_marker[1]) > 0.01):
            self._hold_visual_alignment('robot or marker moved during planning')
            return
        if not self._visual_clearance_current(current):
            self._stop_qr_alignment('scene changed or clearance lost since planning')
            return
        if not self._alignment_is_stationary(self.get_clock().now()):
            self._hold_visual_alignment('robot moved before plan acceptance')
            return
        self._visual_plan_failures = 0
        self._visual_executor = VisualManeuverExecutor(
            plan, speed=0.005, crab_speed=0.01, segment_timeout=20.0)
        if self._visual_progress_stalled(current):
            self._stop_qr_alignment(
                f'no progress after {QR_VISUAL_MAX_NO_PROGRESS} camera corrections; '
                'the heading or position is outside what camera corrections can fix')
            return

    def _carried_offset(self, cargo):
        """Attached cargo in the robot frame (forward, left, height, yaw).

        Cargo and robot are both taken from localization, so the offset is
        right even while the camera stage runs on its own anchored frame.
        """
        cx, cy, cz, cyaw = self._cargo_poses[cargo]
        transform = self._tf.lookup_transform('map', 'base_footprint', rclpy.time.Time())
        rx, ry = transform.transform.translation.x, transform.transform.translation.y
        ryaw = _yaw(transform.transform.rotation)
        dx, dy = cx - rx, cy - ry
        c, s = math.cos(ryaw), math.sin(ryaw)
        return (c * dx + s * dy, -s * dx + c * dy, cz, cyaw - ryaw)

    def _visual_clearance_current(self, pose):
        """Recheck exact current pose against a *fresh* live scene.

        Loaded (camera alignment at a drop-off), the carried cargo moves with
        the robot and its own stand is not an obstacle, as in planning.
        """
        try:
            now = self.get_clock().now().nanoseconds / 1e9
            if not self._loaded_guard_is_ready():
                return False
            world = Path(get_package_share_directory('tugger_gazebo')) / 'worlds' / 'tugger_warehouse.sdf'
            scene = decode_scene(self._collision_scene, now,
                                 expected_counts=scene_expected_counts(world))
            lift = self._lift_position
            if lift is None:
                return False
            cargo = self._confirmed_cargo()
            if cargo is None:
                if not -0.005 <= lift <= 0.005:
                    return False
                return PreparedVisualClearance((box for _, box in scene), lift)(pose)
            if not lift_is_raised(lift):
                return False
            excluded = cargo_stand_model(cargo)
            return PreparedVisualClearance(
                (box for model, box in scene if model != excluded), lift,
                carried=self._carried_offset(cargo))(pose)
        except Exception:
            return False

    def _alignment_odometry(self, message):
        """Retain measured velocity, not the requested velocity, for completion."""
        if message.header.frame_id != 'odom' or message.child_frame_id != 'base_footprint':
            return
        velocity = message.twist.twist
        values = (float(velocity.linear.x), float(velocity.linear.y), float(velocity.angular.z))
        stamp = message.header.stamp.sec + message.header.stamp.nanosec / 1e9
        if all(math.isfinite(value) for value in values):
            self._alignment_odom, self._alignment_odom_stamp = values, stamp

    def _alignment_is_stationary(self, now):
        if self._alignment_odom is None or self._alignment_odom_stamp is None:
            return False
        age = now.nanoseconds / 1e9 - self._alignment_odom_stamp
        vx, vy, yaw_rate = self._alignment_odom
        return (0.0 <= age <= QR_ALIGNMENT_MAX_OBSERVATION_AGE_S and
                math.hypot(vx, vy) <= QR_STATIONARY_MAX_SPEED_M_S and
                abs(yaw_rate) <= QR_STATIONARY_MAX_YAW_RATE_RAD_S)

    def _visual_progress_stalled(self, pose):
        """True once several accepted plans in a row have not reduced the error."""
        forward, lateral = self._red_frame_error
        try:
            heading = abs(self._alignment_heading_error(pose))
        except Exception:
            heading = 0.0
        if not all(math.isfinite(v) for v in (forward, lateral, heading)):
            return False
        # Position error plus heading error weighted to metres (6 degrees ~ 3 cm).
        metric = math.hypot(forward, lateral) + 0.3 * heading
        best = getattr(self, '_visual_best_metric', None)
        if best is None or metric < best - QR_VISUAL_PROGRESS_M:
            self._visual_best_metric = metric
            self._visual_no_progress = 0
            return False
        self._visual_no_progress = getattr(self, '_visual_no_progress', 0) + 1
        return self._visual_no_progress >= QR_VISUAL_MAX_NO_PROGRESS

    def _report_red_frame_state(self, forward_m, lateral_m, state):
        """Show the camera's estimate only when it changes, or every 5 s."""
        now = self.get_clock().now()
        last = getattr(self, '_last_red_frame_reported', None)
        if self._last_red_frame_state_report_at is not None:
            elapsed = (now - self._last_red_frame_state_report_at).nanoseconds / 1e9
            changed = (last is None or state != last[2] or
                       math.hypot(forward_m - last[0], lateral_m - last[1]) > 0.001)
            if elapsed < 0.5 or (not changed and elapsed < 5.0):
                return
        self._last_red_frame_state_report_at = now
        self._last_red_frame_reported = (forward_m, lateral_m, state)
        status = 'CENTERED' if state == 'ALIGNED' else 'CORRECTING'
        self._report(
            f'RED FRAME {status}: forward error={forward_m:.3f} m, '
            f'lateral error={lateral_m:.3f} m; '
            f'distance={math.hypot(forward_m, lateral_m):.3f} m; '
            f'accepting within {QR_ALIGNMENT_ACCEPT_M:.3f} m '
            f'(requirement {QR_ALIGNMENT_POSITION_TOLERANCE_M:.3f} m).')

    def _complete_red_frame_alignment(self):
        """Report final success after the red-frame positional gate passes."""
        if not self._qr_alignment_active:
            return
        station = self._qr_alignment_station
        marker_x, marker_y = self._red_frame_error
        self._qr_alignment_active = False
        self._release_camera_frame()
        self._dock_precision_mode = False
        self.docked_station = station
        self._publish_precision_stop()
        self._report(
            f'PRECISION POSITION REACHED: red frame centered for {station}; '
            f'forward error={marker_x:.3f} m, lateral error={marker_y:.3f} m '
            f'within 10 mm and heading within 1 degree for '
            f'{QR_ALIGNMENT_STABLE_FRAMES} stationary frames.')
        turn = getattr(self, '_turn_recentre', None)
        if turn is not None and turn['station'] == station:
            self._turn_recentre = None
            self._continue_after_turn(turn)
            return
        if station == 'HOME':
            self._execution_done(True, GoalStatus.STATUS_SUCCEEDED)
            return
        if self._dock_only(station):
            self._report(f'{station_label(station)}: docked; no cargo action was selected.')
            self._execution_done(True, GoalStatus.STATUS_SUCCEEDED)
            return
        self._start_cargo_operation(station)

    def _start_cargo_operation(self, station):
        self._mission_phase = 'cargo'
        self._start_cargo_operation_now(station)

    def _start_cargo_operation_now(self, station):
        """Begin a cargo operation only with fresh telemetry and clearance."""
        pickup = PICKUP_CARGO
        try:
            if not self._motion_armed or station not in CARGO_STATIONS:
                raise ValueError('armed, known cargo station is required')
            if not self._loaded_guard_is_ready():
                raise ValueError('loaded planner and controller clearance are not ready')
            if not self._expected_qr_is_fresh(station):
                raise ValueError('expected destination QR identity is missing or stale')
            cargo = self._confirmed_cargo()
            self._check_destination_allowed(station, cargo)
            now = self.get_clock().now()
            if (self._lift_seen_at is None or
                    not 0 <= (now - self._lift_seen_at).nanoseconds / 1e9 <= 1.0):
                raise ValueError('lift position feedback is missing or stale')
            action = getattr(self, '_mission_action', None)
            if action not in (ATTACH, DETACH):
                raise ValueError('no cargo action (ATTACH or DETACH) was selected')
            is_pickup = action == ATTACH
            if is_pickup:
                # Attach whichever cargo the station actually holds.
                occupancy = self._station_occupancy()
                cargo_id = occupancy.get(station) if occupancy else pickup.get(station)
                if cargo_id is None:
                    raise ValueError(f'{station_label(station)} has no cargo to attach')
            else:
                # Set down the cargo actually carried.
                cargo_id = cargo
                if cargo_id is None:
                    raise ValueError('no attached cargo to detach')
            pose_seen = self._cargo_pose_seen_at[cargo_id]
            if (pose_seen is None or self._cargo_poses[cargo_id] is None or
                    not 0 <= (now - pose_seen).nanoseconds / 1e9 <= 1.0):
                raise ValueError('estimated cargo pose is missing or stale')
            if is_pickup and not lift_is_down(self._lift_position):
                raise ValueError('pickup requires the lift to be down')
            if not is_pickup and not lift_is_raised(self._lift_position):
                raise ValueError('drop-off requires confirmed raised cargo')
            if self._cargo_clearance_guard is None or not self._cargo_clearance_guard(
                    station, cargo_id, 'pickup' if is_pickup else 'dropoff'):
                raise ValueError('height-aware cargo operation clearance is unavailable')
            self._lift_gap = None
            self._cargo_task = CargoOperation(
                'pickup' if is_pickup else 'dropoff', cargo_id,
                now.nanoseconds / 1e9, self._cargo_poses[cargo_id][2])
            # Lift is still here (down for pickup, raised for drop-off): the
            # cargo's resting height is its estimate minus the lift offset.
            self._cargo_task.rest_z = (self._cargo_poses[cargo_id][2] -
                                       cargo_lift_offset(self._lift_position))
            self._report(f'{station}: starting confirmed cargo operation.')
            if is_pickup:
                self._report('Allowing the vehicle to settle before requesting attachment.')
            else:
                # Lower only to contact: the cargo rests on its support first.
                self.lift(LIFT_CONTACT_M)
        except (ValueError, TypeError) as exc:
            self._report(f'MISSION INCOMPLETE: cargo operation blocked: {exc}.')
            self._execution_done(False, GoalStatus.STATUS_ABORTED)

    def _advance_cargo_operation(self, rejection=None):
        task = self._cargo_task
        if task is None:
            return
        now = self.get_clock().now()
        now_seconds = now.nanoseconds / 1e9
        state_seen = self._cargo_seen_at[task.cargo_id]
        lift_seen = self._lift_seen_at
        attachment = rejection
        if (attachment is None and state_seen is not None and
                task.phase_started <= state_seen.nanoseconds / 1e9 <= now_seconds and
                now_seconds - state_seen.nanoseconds / 1e9 <= 2.0):
            attachment = self._cargo_states[task.cargo_id]
        lift = None
        if (lift_seen is not None and
                task.phase_started <= lift_seen.nanoseconds / 1e9 <= now_seconds and
                now_seconds - lift_seen.nanoseconds / 1e9 <= 1.0):
            lift = self._lift_position
        cargo_height = None
        pose_seen = self._cargo_pose_seen_at[task.cargo_id]
        if (pose_seen is not None and self._cargo_poses[task.cargo_id] is not None and
                task.phase_started <= pose_seen.nanoseconds / 1e9 <= now_seconds and
                now_seconds - pose_seen.nanoseconds / 1e9 <= 1.0):
            cargo_height = self._cargo_poses[task.cargo_id][2]
        command = None
        if task.phase in ('WAIT_RAISE', 'WAIT_LOWER') and self._motion_armed:
            try:
                if (now_seconds - task.phase_started > 1.5 and
                        (attachment is None or lift is None or cargo_height is None)):
                    task.result, command = 'feedback_stale_during_lift', 'FAILED'
                elif not self._cargo_clearance_guard(
                        self.docked_station, task.cargo_id, task.operation):
                    task.result, command = 'clearance_lost_during_lift', 'FAILED'
                elif getattr(self, '_lift_gap', None) is not None:
                    # Fresh data again: resume toward the target held back.
                    target = self._lift_gap[1]
                    self._lift_gap = None
                    self._report('Lift resumed: clearance data is fresh again.')
                    if target is not None:
                        self.lift(target)
            except ClearanceUnavailable as exc:
                # Not a collision: hold the lift still and wait briefly for
                # fresh data instead of abandoning the pickup.
                gap = getattr(self, '_lift_gap', None)
                if gap is None:
                    self._lift_gap = (now_seconds, getattr(self, '_lift_target', None))
                    if self._lift_position is not None:
                        self.lift(self._lift_position, remember=False)
                    self._report(f'Lift paused: clearance data momentarily unavailable ({exc}).')
                    return
                if now_seconds - gap[0] <= LIFT_DATA_GAP_S:
                    return
                self._lift_gap = None
                self._report(f'Cargo clearance unavailable for over {LIFT_DATA_GAP_S:.0f} s '
                             f'during the lift: {exc}')
                task.result, command = 'clearance_unavailable_during_lift', 'FAILED'
            except Exception as exc:
                self._report(f'Cargo clearance check failed during lift: {exc}')
                task.result, command = 'clearance_unavailable_during_lift', 'FAILED'
        if command is None:
            command = task.observe(now_seconds, attachment=attachment, lift=lift,
                                   cargo_height=cargo_height, armed=self._motion_armed)
        if command in ('CONTACT', 'ATTACH', 'RAISE', 'DETACH', 'STOW'):
            try:
                if not self._expected_qr_is_fresh(self.docked_station):
                    task.result = 'destination_qr_stale'
                    command = 'FAILED'
                elif not self._cargo_clearance_guard(
                        self.docked_station, task.cargo_id, task.operation):
                    task.result = 'clearance_lost'
                    command = 'FAILED'
            except Exception as exc:
                self._report(f'Cargo clearance check failed: {exc}')
                task.result = 'clearance_unavailable'
                command = 'FAILED'
        if command == 'CONTACT':
            self.lift(LIFT_CONTACT_M)
        elif command == 'ATTACH':
            self.cargo(task.cargo_id, True)
        elif command == 'RAISE':
            self.lift(LIFT_UP_M)
        elif command == 'DETACH':
            self.cargo(task.cargo_id, False)
        elif command == 'STOW':
            self.lift(LIFT_DOWN_M)
        elif command == 'SUCCEEDED':
            self._cargo_task = None
            label = 'ATTACHED AND LIFTED' if task.operation == 'pickup' else 'LOWERED AND DETACHED'
            self._report(f'CARGO {label}: estimated cargo pose and lift feedback confirmed.')
            self._execution_done(True, GoalStatus.STATUS_SUCCEEDED)
        elif command == 'FAILED':
            self._cargo_task = None
            if (self._lift_seen_at is not None and
                    0 <= (now - self._lift_seen_at).nanoseconds / 1e9 <= 1.0):
                # Hold where it is, but never onto the latching lower limit.
                self.lift(max(self._lift_position, LIFT_DOWN_M))
            else:
                self._report('LIFT STOP UNCONFIRMED: position feedback is stale; '
                             'the previous position command may still be active.')
            self._report(f'MISSION INCOMPLETE: cargo operation failed: {task.result}.')
            self._last_cargo_failure = (task.operation, task.result)
            self._execution_done(False, GoalStatus.STATUS_ABORTED)

    def _stop_qr_alignment(self, reason):
        if not self._qr_alignment_active:
            return
        self._qr_alignment_active = False
        self._release_camera_frame()
        self._dock_precision_mode = False
        self._visual_plan_generation += 1
        with self._visual_plan_lock:
            future = self._visual_plan_future
            self._visual_plan_future = None
        if future is not None:
            future.cancel()
        self._visual_executor = None
        self._visual_replan_requested = False
        self._alignment_target_yaw = None
        self._turn_recentre = None
        self._publish_precision_stop()
        self._report(f'PRECISION ALIGNMENT FAILED: {reason}. Robot stopped.')
        self._execution_done(False, GoalStatus.STATUS_ABORTED)

    def _start_qr_alignment(self, station):
        self._mission_phase = 'camera'
        self._start_qr_alignment_now(station)

    def _start_qr_alignment_now(self, station):
        # HOME accepts either end, as the controller decides which end leads.
        # Cargo targets must use the selected approved precision segment, never
        # the yaw of an arbitrary entrance or the identity quaternion of the
        # red frame.
        try:
            target_yaw = (home_alignment_yaw(self._robot_pose()[2]) if station == 'HOME'
                          else self._precision_handoff_yaw)
        except Exception as exc:
            self._report(f'PRECISION ALIGNMENT BLOCKED for {station}: {exc}.')
            self._execution_done(False, GoalStatus.STATUS_ABORTED)
            return
        if target_yaw is None or not math.isfinite(target_yaw):
            self._report(f'PRECISION ALIGNMENT BLOCKED for {station}: no approved final heading.')
            self._execution_done(False, GoalStatus.STATUS_ABORTED)
            return
        self._alignment_target_yaw = target_yaw
        try:
            self._freeze_camera_frame()
        except Exception as exc:
            self._report(f'PRECISION ALIGNMENT BLOCKED for {station}: no map -> odom: {exc}.')
            self._execution_done(False, GoalStatus.STATUS_ABORTED)
            return
        self._qr_heading_angle = None
        self._qr_heading_stamp = None
        self._qr_heading_reference = None
        self._qr_alignment_active = True
        self._qr_alignment_station = station
        self._qr_aligned_frames = 0
        self._qr_seen_at = None
        self._qr_expected_seen_at = None
        self._red_frame_seen_at = None
        self._red_frame_stamp = None
        self._red_frame_aligned = False
        self._red_frame_reported = False
        self._red_frame_error = (float('inf'), float('inf'))
        self._last_red_frame_state_report_at = None
        self._qr_alignment_started_at = self.get_clock().now()
        self._visual_plan_generation += 1
        with self._visual_plan_lock:
            future = self._visual_plan_future
            self._visual_plan_future = None
        if future is not None:
            future.cancel()
        self._visual_plan_result = None
        self._visual_plan_error = None
        self._visual_plan_origin = None
        self._visual_plan_marker = None
        self._visual_latest_marker = None
        self._visual_latest_camera_stamp = None
        self._visual_executor = None
        self._visual_replan_requested = True
        self._visual_plan_failures = 0
        self._visual_best_metric = None
        self._visual_no_progress = 0
        self._visual_marker_guard = None
        self._visual_camera_outliers = 0
        self._last_visual_hold_report_at = None
        self._report(
            f'NAV2 COARSE APPROACH COMPLETE. Red-frame alignment now owns final '
            f'motion for {station}: 10 mm centre and 1 degree yaw. QR '
            f'{expected_marker(station)} remains an identity observation.')

    def _qr_alignment_watchdog(self):
        if not self._qr_alignment_active:
            return
        self._poll_visual_plan()
        if not self._qr_alignment_active:
            return
        self._publish_precision_stop() if (self._visual_executor is None or
                                            not self._expected_qr_is_fresh(self._qr_alignment_station)) else None
        if ((self.get_clock().now() - self._qr_alignment_started_at).nanoseconds / 1e9 >
                QR_ALIGNMENT_MAX_DURATION_S):
            self._stop_qr_alignment('alignment timed out without stationary confirmation')
            return
        reference = self._red_frame_seen_at or self._qr_alignment_started_at
        age = (self.get_clock().now() - reference).nanoseconds / 1e9
        if age > QR_ALIGNMENT_MAX_OBSERVATION_AGE_S:
            self._hold_visual_alignment('camera frame missing; stopping for reacquisition')
        if age > QR_ALIGNMENT_TIMEOUT_S:
            self._stop_qr_alignment('red-frame camera not reacquired within 3 s')
            return
        expected = self._qr_expected_seen_at or self._qr_alignment_started_at
        if (self.get_clock().now() - expected).nanoseconds / 1e9 > QR_ALIGNMENT_TIMEOUT_S:
            self._stop_qr_alignment('expected QR identity not reacquired within 3 s')

    def _pose(self, values):
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x, pose.pose.position.y, yaw = values
        pose.pose.orientation.z = math.sin(yaw / 2.0)
        pose.pose.orientation.w = math.cos(yaw / 2.0)
        return pose

    def _robot_yaw(self):
        return self._robot_pose()[2]

    def _robot_pose(self):
        frozen = getattr(self, '_camera_map_odom', None)
        if frozen is not None:
            return _compose_pose(frozen, self._robot_odom_pose())
        transform = self._tf.lookup_transform(
            'map', 'base_footprint', rclpy.time.Time())
        translation = transform.transform.translation
        return translation.x, translation.y, _yaw(transform.transform.rotation)

    def _freeze_camera_frame(self):
        """Hold map -> odom fixed for the camera stage (visual servoing frame).

        The camera stage checks millimetre consistency over a few seconds.
        Laser localization corrects itself in centimetre steps; odometry is
        smooth over that time. So during the stage every pose is the odometry
        pose placed with the map -> odom transform taken at its start.
        """
        transform = self._tf.lookup_transform('map', 'odom', rclpy.time.Time())
        t = transform.transform.translation
        self._camera_map_odom = (t.x, t.y, _yaw(transform.transform.rotation))
        self._camera_anchored = False
        self._camera_yaw_samples = []

    def _anchor_camera_heading(self, station, pose, stamp):
        """Set the frozen frame's heading from the floor QR (median of samples).

        The QR's orientation is unique, so its angle in the downward camera
        gives the absolute robot heading (QR_TO_ROBOT_YAW). The frame turns
        about the robot's current position, which stays where the marker put it.
        """
        qr_stamp = getattr(self, '_qr_heading_stamp', None)
        frame_time = stamp.sec + stamp.nanosec / 1e9
        if qr_stamp is None or abs(frame_time - qr_stamp) > 0.2:
            return pose
        self._camera_yaw_samples.append(wrap(self._qr_heading_angle + QR_TO_ROBOT_YAW))
        if len(self._camera_yaw_samples) < QR_YAW_SAMPLES:
            return pose
        samples = sorted(self._camera_yaw_samples, key=lambda a: wrap(a - pose[2]))
        heading = samples[len(samples) // 2]
        self._camera_yaw_samples = None
        # Restart the marker-consistency guard and planning in the anchored frame.
        self._visual_marker_guard = None
        self._visual_replan_requested = True
        turn = wrap(heading - pose[2])
        if abs(turn) > math.radians(5.0):
            return pose  # implausible: keep the localization heading
        frozen = self._camera_map_odom
        odom = self._robot_odom_pose()
        yaw = frozen[2] + turn
        c, s = math.cos(yaw), math.sin(yaw)
        current = self._robot_pose()
        self._camera_map_odom = (current[0] - (c * odom[0] - s * odom[1]),
                                 current[1] - (s * odom[0] + c * odom[1]), yaw)
        if abs(turn) > math.radians(0.3):
            self._report(f'{station}: camera stage heading set from the floor QR '
                         f'(localization was {math.degrees(turn):+.1f} deg off).')
        return (pose[0], pose[1], wrap(pose[2] + turn))

    def _anchor_camera_frame(self, station, pose, marker):
        """Place the frozen frame so the seen red frame sits on its station.

        ``pose`` is the robot pose in the frozen frame at the image time and
        ``marker`` the frame centre in the robot frame. The station's centre
        is known from the site layout, so the robot's map position follows
        from the camera to the millimetre (floor-marker localization); the
        heading stays the localization's.
        """
        center = getattr(self, '_station_centers', {}).get(station)
        frozen = getattr(self, '_camera_map_odom', None)
        if center is None or frozen is None:
            return pose
        c, s = math.cos(pose[2]), math.sin(pose[2])
        seen = (pose[0] + c * marker[0] - s * marker[1], pose[1] + s * marker[0] + c * marker[1])
        shift = (center[0] - seen[0], center[1] - seen[1])
        if math.hypot(*shift) > 0.30:
            return pose  # not this station's marker: never anchor on it
        self._camera_map_odom = (frozen[0] + shift[0], frozen[1] + shift[1], frozen[2])
        self._camera_anchored = True
        # Hand the exact pose to localization too, so everything placed from
        # it (the carried cargo in the scene, the motion gate) agrees with the
        # camera stage. The stage itself runs on its frozen frame and is not
        # disturbed by the resulting jump.
        fix = PoseWithCovarianceStamped()
        fix.header.stamp = self.get_clock().now().to_msg()
        fix.header.frame_id = 'map'
        current = self._robot_pose()
        fix.pose.pose.position.x, fix.pose.pose.position.y = current[0], current[1]
        fix.pose.pose.orientation.z = math.sin(current[2] / 2)
        fix.pose.pose.orientation.w = math.cos(current[2] / 2)
        covariance = [0.0] * 36
        covariance[0] = covariance[7] = 0.01 ** 2
        covariance[35] = math.radians(1.0) ** 2
        fix.pose.covariance = covariance
        self._initial_pose_pub.publish(fix)
        if math.hypot(*shift) > 0.01:
            self._report(f'{station}: camera stage anchored on the floor marker '
                         f'(localization was {math.hypot(*shift) * 100:.1f} cm off).')
        return (pose[0] + shift[0], pose[1] + shift[1], pose[2])

    def _release_camera_frame(self):
        self._camera_map_odom = None

    @staticmethod
    def _path_length(poses):
        return sum(math.hypot(b[0] - a[0], b[1] - a[1])
                   for a, b in zip(poses, poses[1:]))

    def _begin_preflight(self):
        """Invalidate any earlier preview before selecting a new mission."""
        self._plan_generation += 1
        self._pending_path = None
        self._pending_pose = None
        self._pending_alignment_station = None
        self._pending_precision_start = None
        self._pending_destination = None
        self._pending_execution_approved = False
        self._queued_precision_poses = None
        self._queued_precision_station = None
        self._stage_validation_pending = False
        self._dock_precision_mode = False
        self._dock_pending = None
        self._pending_departure = None
        self._departure_pending = None
        self._departure_target = None
        self._departure_exit_pose = None
        self._station_exit_active = False
        self._station_exit_target = None
        self._precision_handoff_yaw = None
        return self._plan_generation

    def execute_saved_route(self, label, saved_poses, precision_station=None,
                            camera_handoff_station=None, next_precision=None,
                            mission_destination=None, execution_approved=False):
        """Dispatch an offline-generated route, with only a short live connector."""
        if (self._follow_goal_pending or self._stage_validation_pending or
                self._departure_pending is not None or self._departure_active or
                self._active_follow is not None or self._qr_alignment_active or
                self._cargo_task is not None or getattr(self, '_rotation', None) is not None):
            self._report('Planning blocked: cancel the active mission first.')
            return
        generation = self._begin_preflight()
        self._pending_destination = mission_destination
        self._pending_execution_approved = execution_approved is True
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
        if next_precision is not None:
            self._queued_precision_station = camera_handoff_station
            self._queued_precision_poses = next_precision
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
                f'{label}: route start is {gap:.2f} m away; refusing a long '
                'dynamic replacement route.')
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

    def _recover_loaded_dock_station(self, cargo, pose):
        """Recover a loaded dock from live feedback after GUI restart."""
        if self._approach_provider is None:
            return

        station = f'cargo_station_{cargo}'
        _, approach = self._approach_provider(station)

        if not approach or len(approach[-1]) < 2:
            raise ValueError('Configured pickup approach unavailable')

        distance = math.hypot(
            pose[0] - approach[-1][0],
            pose[1] - approach[-1][1])

        if distance > 0.20:
            return  # Robot is no longer at the pickup dock.

        now = self.get_clock().now()
        seen = self._lift_seen_at

        if (seen is None or
                not 0 <= (now - seen).nanoseconds / 1e9 <= 1.0 or
                self._lift_position is None or
                not lift_is_raised(self._lift_position)):
            raise ValueError(
                'Cannot recover loaded dock: lift feedback stale or invalid')

        try:
            plan_station_departure(pose, approach)
        except ValueError as exc:
            raise ValueError(
                f'Cannot recover loaded dock geometry: {exc}') from exc

        self.docked_station = station
        self._report(
            f'{station}: recovered docked state from fresh cargo/lift '
            'feedback and measured robot pose after GUI restart.')

    def execute_destination(self, destination, candidates, station_targets):
        """Select a destination from the live pose without guessing a route."""
        if (self._follow_goal_pending or self._stage_validation_pending or
                self._departure_pending is not None or self._departure_active or
                self._active_follow is not None or self._qr_alignment_active or
                self._cargo_task is not None or getattr(self, '_rotation', None) is not None):
            self._report('Planning blocked: cancel the active mission first.')
            return
        generation = self._begin_preflight()
        try:
            cargo = self._confirmed_cargo()
            via = getattr(self, '_mission_via', None)
            if via is not None and destination == via['hop']:
                # A waypoint is only driven to its gate: no cargo rules apply.
                blocks, warnings, dock_only = [], [], False
            else:
                blocks, warnings, dock_only = self._assess_destination(destination, cargo)
            if blocks:
                raise ValueError(blocks[0])
            confirm = getattr(self, '_mission_confirm', None)
            if warnings:
                if confirm is None or confirm['destination'] != destination:
                    self._mission_confirm = {'destination': destination,
                                             'warnings': warnings,
                                             'dock_only': dock_only, 'confirmed': False}
                    self._report('WARNING: ' + '; '.join(warnings) +
                                 '. RUN will ask you to confirm.')
            elif (confirm is not None and confirm['destination'] != destination and
                  getattr(self, '_mission_via', None) is None):
                self._mission_confirm = None
            pose = self._robot_pose()
            center = getattr(self, '_station_centers', {}).get(self.docked_station)
            if (self.docked_station is not None and center is not None and
                    math.hypot(pose[0] - center[0], pose[1] - center[1]) > DOCKED_TOLERANCE_M):
                # Left the centre (for example a cancelled exit): no longer docked.
                self._report(f'{self.docked_station}: robot is away from the centre; '
                             'it is no longer docked there.')
                self.docked_station = None
            if self.docked_station is None:
                self._infer_pass_through_dock(pose)
            if self.docked_station is not None and self.docked_station == destination:
                if destination in getattr(self, '_pass_through', ()):
                    raise ValueError(f'the robot is already docked at {destination}')
                raise ValueError(f'the robot is already docked at {station_label(destination)}; '
                                 'use ATTACH or DETACH to act here')
            center = getattr(self, '_station_centers', {}).get(destination)
            if (destination in getattr(self, '_pass_through', ()) and center is not None and
                    self.docked_station is None and
                    math.hypot(pose[0] - center[0], pose[1] - center[1]) <= 0.25):
                # Close to a pass-through centre: camera alignment only.
                self._pending_pose = pose
                self._pending_alignment_station = destination
                self._pending_destination = destination
                self._pending_execution_approved = True
                self._report(
                    f'{destination}: already near the target; press RUN to perform '
                    'camera-guided final alignment. Preview only.')
                return
            # Recover dock identity after restarting only the GUI.
            if cargo is not None and self.docked_station is None:
                self._recover_loaded_dock_station(cargo, pose)
            # Depart a loaded dock using its own height-aware path first.
            # Never ask the ordinary 2D Nav2 global planner to start beneath
            # the stand where the static costmap marks the station occupied.
            partway = (self._station_partway(pose)
                       if cargo is not None and self.docked_station is None else None)
            if partway is not None and partway != destination:
                # Loaded and partway out of a station (e.g. a cancelled
                # departure): under a stand no route planner may start, so
                # finish leaving along the station's own approach first.
                self._prepare_backout(destination, candidates, station_targets,
                                      station=partway)
                return
            if self._maybe_turn_at_dock(destination, candidates, station_targets):
                return
            if self._start_route_chain(destination, candidates, station_targets):
                return
            if cargo is not None and self.docked_station is not None:
                # Leaves by its pickup's own approach, or through a gate of a
                # pass-through stop such as HOME, with loaded clearance checks.
                self._prepare_loaded_departure(
                    self.docked_station, cargo, pose,
                    destination, candidates, station_targets)
                return
            if (cargo is None and self.docked_station is not None and
                    self.docked_station != destination):
                self._prepare_station_exit(
                    self.docked_station, pose, destination, candidates, station_targets)
                return
            approach_rejoin = mission_approach_rejoin(candidates, pose)
            if approach_rejoin is not None:
                # Already on this destination's segment 2: continue along it.
                key, meta, remaining = approach_rejoin
                # Start it as a normal segment 2 (3D scene check and docking
                # guard), not as a route checked against the 2D costmap.
                self._pending_precision_start = (destination, remaining)
                self._pending_pose = pose
                self._pending_destination = destination
                self._pending_execution_approved = meta.get('execution_approved') is True
                self._plan_ready(
                    f'{destination}: on its segment 2 (from {key}); RUN continues the '
                    'precision approach from here.', False)
                return
            decision = select_destination_plan(
                destination, candidates, pose, destination_poses=station_targets)
        except (Exception, ValueError) as exc:
            self._report(f'Destination selection rejected: {exc}')
            return
        approach = None
        approach_approved = True
        route_meta = decision.get('route') or {}
        if decision['mode'] in ('SAVED_REJOIN', 'SHORT_CONNECTOR') and route_meta.get('markers'):
            # A single-segment mission path: its own segment 2 starts at the
            # marker; the route part before it goes to the existing route stage.
            split = route_meta['markers']['approach_start'] - decision['rejoin_index']
            approach = decision['poses'][split:]
            decision['poses'] = decision['poses'][:split + 1]
            approach_approved = route_meta.get('execution_approved') is True
        else:
            approach, approach_approved = self._station_entry_approach(
                destination, decision, pose, station_targets)
            if approach is None:
                return
        via = getattr(self, '_mission_via', None)
        if (via is not None and destination == via['hop'] and
                decision['mode'] in ('SAVED_REJOIN', 'SHORT_CONNECTOR') and
                route_meta.get('markers')):
            # A waypoint: follow the route to the waypoint's gate and stop
            # there; no segment 2 and no docking at the waypoint.
            route = decision['route']
            self.execute_saved_route(
                f'{route["source"]} → {destination} gate (waypoint)', decision['poses'],
                mission_destination=destination,
                execution_approved=(route.get('_database_execution_approved') is True and
                                    route.get('execution_approved') is True))
            return
        if decision['mode'] == 'LIVE_RECOVERY_REQUIRED' and getattr(
                self, '_live_planner', None) is not None and station_targets:
            self._start_live_plan(destination, pose, station_targets, generation,
                                  approach, approach_approved)
            return
        # Remembered so a failed short connector falls through to a live plan.
        self._connector_fallback = (
            (destination, pose, station_targets, generation, approach, approach_approved)
            if decision['mode'] == 'SHORT_CONNECTOR' and station_targets else None)
        self._dispatch_decision(destination, decision, pose, station_targets, generation,
                                approach, approach_approved)

    def _station_entry_approach(self, destination, decision, pose, station_targets):
        """Segment 2 from the configured approach repository for a gate."""
        try:
            if self._approach_provider is None:
                raise ValueError('No precision approach repository is configured')
            if decision['mode'] in ('SAVED_REJOIN', 'SHORT_CONNECTOR'):
                entry = decision['poses'][-1]
            elif station_targets:
                entry = min(station_targets, key=lambda point: math.hypot(
                    point[0] - pose[0], point[1] - pose[1]))
            else:
                raise ValueError('No station entrance is configured')
            metadata, approach = self._approach_provider(destination, entry)
            approach_approved = metadata.get('execution_approved') is True
            if (metadata.get('source') == 'isolated_entrance_candidate' and
                    not approach_approved):
                self._report(
                    f'{destination}: selected a statically sampled entrance '
                    'candidate; Gazebo behavior remains unverified.')
        except (OSError, ValueError, KeyError) as exc:
            self._report(f'{destination}: precision approach unavailable: {exc}')
            return None, False
        return approach, approach_approved

    def _start_live_plan(self, destination, pose, station_targets, generation,
                         fallback_approach, fallback_approved):
        """Plan live in the background with the offline generator's rules."""
        self._report(f'{destination}: no preplanned path within reach; planning live '
                     'with the offline generator rules.')
        future = self._visual_plan_pool.submit(
            self._live_planner.plan, tuple(pose), [list(p) for p in station_targets])
        self._live_plan = (future, destination, pose, station_targets, generation,
                           fallback_approach, fallback_approved)

    def _poll_live_plan(self):
        pending = getattr(self, '_live_plan', None)
        if pending is None or not pending[0].done():
            return
        self._live_plan = None
        future, destination, pose, station_targets, generation, approach, approved = pending
        if generation != self._plan_generation:
            return
        decision = {'mode': 'LIVE_RECOVERY_REQUIRED'}
        try:
            poses, entry = future.result()
            metadata, live_approach = self._approach_provider(destination, entry)
            label = f'Live plan → {destination}'
            self._report(f'{label}: {len(poses)} poses to the {destination} entry; '
                         'segment 2 follows.')
            self.execute_saved_route(
                label, poses, camera_handoff_station=destination,
                next_precision=live_approach, mission_destination=destination,
                execution_approved=metadata.get('execution_approved') is True)
            return
        except Exception as exc:
            self._report(f'{destination}: live planning failed ({exc}); '
                         'using the Nav2 recovery planner.')
        self._dispatch_decision(destination, decision, pose, station_targets, generation,
                                approach, approved)

    def _dispatch_decision(self, destination, decision, pose, station_targets, generation,
                           approach, approach_approved):
        if decision['mode'] == 'ALREADY_AT_DESTINATION':
            gap = math.hypot(pose[0] - approach[0][0], pose[1] - approach[0][1])
            if gap > 0.25:
                self._report(f'{destination}: no validated precision approach from this entrance.')
                return
            self.execute_saved_route(
                f'{destination} precision approach', approach,
                precision_station=destination, mission_destination=destination,
                execution_approved=approach_approved)
            return
        if decision['mode'] != 'LIVE_RECOVERY_REQUIRED':
            route = decision['route']
            if approach is not None and math.hypot(
                    decision['poses'][-1][0] - approach[0][0],
                    decision['poses'][-1][1] - approach[0][1]) > 0.03:
                self._report(
                    f'{destination}: selected route ends at an entrance without '
                    'a matching validated precision approach. No plan was sent.')
                return
            label = f'{route["source"]} → {destination} ({decision["mode"]})'
            self._report(
                f'{label}: rejoin pose {decision["rejoin_index"]}; '
                f'gap {decision["gap_m"]:.2f} m; '
                f'leading end {decision["direction"]["leading_end"]}.')
            self.execute_saved_route(
                label, decision['poses'], camera_handoff_station=destination,
                next_precision=approach, mission_destination=destination,
                execution_approved=(
                    route.get('_database_execution_approved') is True and
                    route.get('execution_approved') is True and approach_approved))
            return
        if not station_targets:
            self._report(f'{destination}: no station entry is available for live recovery.')
            return
        target = approach[0] if approach is not None else min(
            station_targets, key=lambda point: math.hypot(
                point[0] - pose[0], point[1] - pose[1]))
        self._report(
            f'{destination}: no safe saved-route rejoin; planning live recovery '
            f'to a facility entry point.')
        self._route_label = f'Live recovery → {destination}'
        self._pending_destination = destination
        self._pending_execution_approved = approach_approved
        self._saved_poses = []
        self._precision_station = None
        self._camera_handoff_station = destination
        self._queued_precision_poses = approach
        self._queued_precision_station = destination if approach is not None else None
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
                raise ValueError('connector planner rejected the saved-route start')
            result = handle.get_result_async()
            result.add_done_callback(
                lambda response, token=generation: self._on_connector_result(response, token))
        except Exception as exc:
            self._connector_failed(str(exc), generation)

    def _on_connector_result(self, future, generation):
        if generation != self._plan_generation:
            return
        try:
            response = future.result()
            connector = response.result.path
            if response.status != GoalStatus.STATUS_SUCCEEDED or len(connector.poses) < 2:
                raise ValueError('no safe short connector to the selected saved route')
            planner_poses = [[p.pose.position.x, p.pose.position.y,
                              _yaw(p.pose.orientation)] for p in connector.poses]
            planner_sign = validate_planner_motion(planner_poses)
            selected = prepare_dynamic_path(
                [(pose[0], pose[1]) for pose in planner_poses], self._robot_yaw())
            if planner_sign != selected['body_velocity_sign']:
                raise ValueError('planner direction disagrees with live heading')
            connector_poses = [[p.pose.position.x, p.pose.position.y, 0.0]
                               for p in connector.poses]
            length = self._path_length(connector_poses)
            if length > 3.0:
                raise ValueError(f'connector is {length:.2f} m; refusing a long dynamic route')
            if self._recovery_only:
                self._recovery_only = False
                self._prepare_saved_execution(connector_poses, length, generation)
                return
            if math.hypot(connector_poses[-1][0] - self._saved_poses[0][0],
                          connector_poses[-1][1] - self._saved_poses[0][1]) > 0.15:
                raise ValueError('connector does not reach the selected saved-route start')
            merged = connector_poses[:-1] + self._saved_poses
            self._prepare_saved_execution(merged, length, generation)
        except Exception as exc:
            self._connector_failed(str(exc), generation)

    def _connector_failed(self, reason, generation):
        """A short connector failed: plan the whole path live, as agreed.

        The Nav2 grid planner knows nothing of 4WS kinematics, so its short
        connector may reverse direction or miss the route. The live planner
        uses the offline generator's rules and plans one full path from the
        robot. Only the last-resort Nav2 recovery plan has no further fallback.
        """
        context = getattr(self, '_connector_fallback', None)
        self._connector_fallback = None
        if (self._recovery_only or context is None or
                getattr(self, '_live_planner', None) is None):
            self._report(f'Connector result failed: {reason}. No plan was sent.')
            return
        # The connector's generation is current (stale results return early).
        destination, pose, station_targets, _selected_at, approach, approved = context
        self._report(f'{destination}: short connector failed ({reason}); planning the '
                     'whole path live instead.')
        self._start_live_plan(destination, pose, station_targets, generation,
                              approach, approved)

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
        try:
            cargo = self._confirmed_cargo()
            clearance_note = 'Unloaded footprint checked; cargo-height clearance does not apply.'
            if cargo is not None:
                modeled_poses = [(p[0], p[1], heading)
                                 for p, heading in zip(poses, prepared['body_headings'])]
                samples = self._modeled_loaded_path_clearance(modeled_poses, cargo)
                self._report(f'Modeled loaded path clearance passed {samples} samples; '
                             'the simulator scene will be rechecked before RUN.')
                clearance_note = 'Loaded 2.5D scene sampled; runtime guard required.'
        except Exception as exc:
            self._report(f'Loaded path check failed: {exc}. RUN is not enabled.')
            return
        self._path_pub.publish(path)
        summary = (
            f'{self._route_label}: saved route has {len(poses)} poses; '
            f'connector {connector_length:.2f} m; '
            f'initial leading end {prepared["leading_end"]}; '
            f'body motion {prepared["body_motion_direction"]}. {clearance_note}')
        if not self._path_validator.service_is_ready():
            self._report('Path validation service unavailable; RUN is not enabled.')
            return
        request = IsPathValid.Request()
        request.path = path
        request.consider_unknown_as_obstacle = True
        self._report(f'{summary} Checking the path against the global costmap.')
        future = self._path_validator.call_async(request)
        token = self._plan_generation if generation is None else generation
        future.add_done_callback(
            lambda response, expected=token: self._on_path_validated(
                response, expected, path, preview_pose, summary))

    def _on_path_validated(self, future, generation, path, preview_pose, summary):
        """Make a preview runnable only after Nav2 validates the current costmap."""
        if generation != self._plan_generation:
            return
        try:
            result = future.result()
            if not result.success or not result.is_valid:
                self._report('Path validation failed or reported an obstacle; RUN is not enabled.')
                return
        except Exception as exc:
            self._report(f'Path validation failed: {exc}; RUN is not enabled.')
            return
        self._pending_path = path
        self._pending_pose = preview_pose
        self._plan_ready(
            f'{summary} Global costmap check passed; press RUN to dispatch '
            'this path.', False)

    def run_selected_mission(self, armed):
        """Run only the exact path previously previewed by the operator."""
        if not armed:
            self._report_run_block('RUN blocked: arm simulator commands first.')
            return
        if (self._follow_goal_pending or self._stage_validation_pending or
                self._departure_pending is not None or self._departure_active or
                self._active_follow is not None or self._qr_alignment_active or
                self._cargo_task is not None or getattr(self, '_rotation', None) is not None):
            self._report_run_block('RUN blocked: a navigation or precision alignment is already active.')
            return
        if (self._pending_path is None and self._pending_alignment_station is None and
                self._pending_departure is None and
                getattr(self, '_pending_precision_start', None) is None and
                getattr(self, '_pending_rotation', None) is None):
            self._report_run_block('RUN blocked: select and validate a mission first.')
            return
        if not self._pending_execution_approved:
            self._report_run_block('RUN blocked: a selected saved or precision route is not execution-approved.')
            return
        try:
            check = plan_is_current(self._pending_pose, self._robot_pose())
        except (Exception, ValueError) as exc:
            self._report_run_block(f'RUN blocked: cannot recheck the live pose: {exc}')
            return
        if not check['current']:
            self._report_run_block(
                'RUN blocked: robot pose changed since feasibility check '
                f'({check["position_error_m"]:.2f} m, '
                f'{math.degrees(check["yaw_error_rad"]):.1f} deg). '
                'Check/plan the destination again.')
            return
        try:
            cargo = self._confirmed_cargo()
            if self._pending_destination is not None:
                self._check_destination_allowed(self._pending_destination, cargo)
            if cargo is not None and not self._loaded_guard_is_ready():
                self._report_run_block('RUN blocked: loaded controller clearance monitoring is not ready.')
                return
            if cargo is not None and self._pending_path is not None:
                poses = [(p.pose.position.x, p.pose.position.y, _yaw(p.pose.orientation))
                         for p in self._pending_path.poses]
                self._modeled_loaded_path_clearance(poses, cargo)
        except (ValueError, TypeError, OSError) as exc:
            self._report_run_block(f'RUN blocked: {exc}.')
            return
        if self._pending_departure is not None:
            pending = self._pending_departure
            try:
                if self._confirmed_cargo() != pending['cargo_id']:
                    raise ValueError('cargo attachment changed since departure preview')
                if not self._loaded_guard_is_ready():
                    raise ValueError('loaded scene guard is not ready')
                self._modeled_loaded_path_clearance(pending['poses'], pending['cargo_id'])
                if not self._dock_follower.server_is_ready():
                    raise ValueError('independent docking controller unavailable')
            except Exception as exc:
                self._report_run_block(f'RUN blocked: loaded departure preflight: {exc}')
                return
            self._pending_departure = None
            self._pending_pose = None
            self._motion_armed = True
            self._cancel_requested = False
            pending['generation'] = self._plan_generation
            pending['started'] = self.get_clock().now()
            self._departure_pending = pending
            self._report('Loaded departure RUN accepted; waiting for DOCK_DEPART guard.')
            return
        if getattr(self, '_pending_rotation', None) is not None:
            self._pending_pose = None
            self._motion_armed = True
            self._cancel_requested = False
            self._start_rotation()
            return
        pending_start = getattr(self, '_pending_precision_start', None)
        if pending_start is not None:
            station, poses = pending_start
            self._pending_precision_start = None
            self._pending_pose = None
            self._motion_armed = True
            self._cancel_requested = False
            self.docked_station = None
            self._precision_station = None
            self._camera_handoff_station = station
            self._queued_precision_station = station
            self._queued_precision_poses = poses
            self._start_precision_segment()
            return
        if self._pending_alignment_station is not None:
            station = self._pending_alignment_station
            self._pending_alignment_station = None
            self._pending_pose = None
            self.docked_station = None
            self._motion_armed = True
            self._cancel_requested = False
            self._start_qr_alignment(station)
            return
        if not self._follower.server_is_ready():
            self._report('FollowPath server is unavailable; plan remains preview-only.')
            return
        path = self._pending_path
        self._pending_path = None
        self._pending_pose = None
        self._motion_armed = True
        self._cancel_requested = False
        self._follow(path)

    def _follow(self, path):
        # Save the exact selected final precision orientation before handing
        # command ownership to Nav2. A later station handoff uses this heading.
        if (self._precision_station and self._camera_handoff_station ==
                self._precision_station and path.poses):
            self._precision_handoff_yaw = _yaw(path.poses[-1].pose.orientation)
        follower = (self._dock_follower if self._dock_precision_mode or
                    self._departure_active else self._follower)
        if not follower.server_is_ready():
            self._report('FollowPath server is unavailable; plan remains preview-only.')
            return
        goal = FollowPath.Goal()
        goal.path = path
        exiting = getattr(self, '_station_exit_active', False)
        self._mission_phase = (
            'exit' if exiting or self._departure_active else
            'approach' if self._precision_station else 'route')
        goal.controller_id = (
            'DepartureFollowPath' if self._departure_active else
            'PrecisionFollowPath' if self._precision_station or exiting else 'FollowPath')
        camera_next = (self._camera_handoff_station and
                       self._queued_precision_poses is None and not exiting)
        # Segment 2 keeps its 10 mm target; routes that end directly in camera
        # alignment stop within 3 cm so the red frame stays in view.
        goal.goal_checker_id = (
            'precision_goal_checker' if self._precision_station else
            'camera_handoff_goal_checker' if camera_next else 'goal_checker')
        try:
            self._follow_goal_pending = True
            future = follower.send_goal_async(goal)
            future.add_done_callback(self._on_follow_goal)
        except Exception as exc:
            self._follow_goal_pending = False
            self._report(f'FollowPath request failed: {exc}')
            if self._departure_active:
                self._departure_active = False
                self._departure_target = None
                self._execution_done(False, GoalStatus.STATUS_ABORTED)

    def _on_follow_goal(self, future):
        self._follow_goal_pending = False
        try:
            self._active_follow = future.result()
            if not self._active_follow.accepted:
                self._active_follow = None
                self._report('Nav2 rejected the trajectory; the robot remains stopped.')
                if self._departure_active:
                    self._departure_active = False
                    self._departure_target = None
                    self._execution_done(False, GoalStatus.STATUS_ABORTED)
                return
            if not self._motion_armed or self._cancel_requested:
                self._active_follow.cancel_goal_async()
                self._report('Nav2 goal cancelled because simulator commands were disarmed.')
            self._report('Nav2 trajectory accepted.')
            result = self._active_follow.get_result_async()
            result.add_done_callback(self._on_follow_result)
        except Exception as exc:
            self._report(f'FollowPath request failed: {exc}')
            if self._departure_active:
                self._departure_active = False
                self._departure_target = None
                self._execution_done(False, GoalStatus.STATUS_ABORTED)

    def _on_follow_result(self, future):
        try:
            status = future.result().status
            if self._cancel_requested:
                self._active_follow = None
                self._queued_precision_poses = None
                self._departure_active = False
                self._departure_target = None
                self._station_exit_active = False
                self._station_exit_target = None
                self._execution_done(False, GoalStatus.STATUS_CANCELED)
                return
            succeeded = status == GoalStatus.STATUS_SUCCEEDED
            if getattr(self, '_station_exit_active', False):
                self._active_follow = None
                self._complete_station_exit(succeeded, status)
                return
            if self._departure_active:
                self._active_follow = None
                if succeeded:
                    self._complete_loaded_departure()
                else:
                    self._departure_active = False
                    self._departure_target = None
                    self._execution_done(False, status)
                return
            if succeeded and self._queued_precision_poses is not None:
                self._active_follow = None
                self._settle_then_precision()
                return
            if succeeded and self._camera_handoff_station:
                self._active_follow = None
                self._start_qr_alignment(self._camera_handoff_station)
                return
            station = self._camera_handoff_station
            if (not succeeded and self._precision_station and station and
                    self._expected_qr_is_fresh(station)):
                # Segment 2 stalled with the expected marker already in view:
                # let the downward camera finish the alignment.
                self._active_follow = None
                self._report(f'{station}: segment 2 stopped short with the marker in view; '
                             'the downward camera continues the alignment.')
                self._start_qr_alignment(station)
                return
            self._dock_precision_mode = False
            self._execution_done(succeeded, status)
        except Exception as exc:
            self._report(f'FollowPath result failed: {exc}')
        self._active_follow = None

    def _on_landmark_fix(self, message):
        try:
            self._landmark_fix_at = float(json.loads(message.data)['stamp'])
        except (ValueError, KeyError, TypeError):
            pass

    def _settle_then_precision(self):
        """Stop at the staging point until a landmark confirms the pose.

        Before driving under a stand the robot stands still for the station
        legs (or floor marker) to fix its pose exactly (marker_localization);
        it then enters on odometry from that fix. If no landmark answers
        within LANDMARK_WAIT_S it continues on laser localization alone.
        """
        start = self.get_clock().now().nanoseconds / 1e9
        timer = None

        def check():
            now = self.get_clock().now().nanoseconds / 1e9
            fixed = getattr(self, '_landmark_fix_at', None)
            if self._cancel_requested or not self._motion_armed:
                timer.cancel()
                return
            if fixed is not None and fixed >= start + 0.3:
                timer.cancel()
                self._start_precision_segment()
            elif now - start > LANDMARK_WAIT_S:
                timer.cancel()
                self._report('No landmark fix at the staging point; entering on laser '
                             'localization.')
                self._start_precision_segment()
        timer = self.create_timer(0.1, check)

    def _start_precision_segment(self):
        """Validate and dispatch segment 2 after segment 1 has stopped."""
        self._mission_phase = 'approach'
        station = self._queued_precision_station
        poses = self._queued_precision_poses
        # An empty robot leaves a station the way it came in. A re-approach
        # after a recovery follows only the remaining part of the same gate's
        # approach; keep the full one, so the exit returns to the staging point.
        previous = getattr(self, '_arrival_approach', None)
        keep = False
        try:
            keep = (previous is not None and previous[0] == station and
                    entrance_gate(previous[1]) == entrance_gate(poses) and
                    self._path_length(previous[1]) > self._path_length(poses) + 0.05)
        except (ValueError, TypeError, IndexError):
            keep = False
        if not keep:
            self._arrival_approach = (station, [list(pose) for pose in poses])
        try:
            current = self._robot_pose()
            gap = math.hypot(current[0] - poses[0][0], current[1] - poses[0][1])
            if gap > 0.25:
                raise ValueError(f'staging gap {gap:.2f} m exceeds 0.25 m')
            prepared = prepare_dynamic_path([(p[0], p[1]) for p in poses], current[2])
            cargo = self._confirmed_cargo()
            self._check_destination_allowed(station, cargo)
            if cargo is not None:
                if not self._loaded_guard_is_ready():
                    raise ValueError('loaded controller clearance monitoring is not ready')
                modeled = [(p[0], p[1], heading)
                           for p, heading in zip(poses, prepared['body_headings'])]
                self._modeled_loaded_path_clearance(modeled, cargo)
            path = NavPath()
            path.header.frame_id = 'map'
            path.header.stamp = self.get_clock().now().to_msg()
            for values, heading in zip(poses, prepared['body_headings']):
                point = self._pose(values)
                point.pose.orientation.z = math.sin(heading / 2.0)
                point.pose.orientation.w = math.cos(heading / 2.0)
                path.poses.append(point)
            if not self._motion_armed:
                raise ValueError('simulator commands are disarmed')
            if cargo is None:
                if not self._loaded_guard_is_ready():
                    raise ValueError('simulator scene guard is not ready')
                if not self._dock_follower.server_is_ready():
                    raise ValueError('independent docking controller is unavailable')
                now = self.get_clock().now()
                if (self._lift_seen_at is None or self._lift_position is None or
                        not 0 <= (now - self._lift_seen_at).nanoseconds / 1e9 <= 1.0):
                    raise ValueError('lift feedback is stale or missing')
                world = Path(get_package_share_directory('tugger_gazebo')) / 'worlds/tugger_warehouse.sdf'
                robot = Path(get_package_share_directory('tugger_description')) / 'urdf/tugger.urdf.xacro'
                validate_modeled_geometry(world, robot)
                if self._collision_scene is None:
                    raise ValueError('simulator collision scene is unavailable')
                scene = decode_scene(self._collision_scene, now.nanoseconds / 1e9,
                                     expected_counts=scene_expected_counts(world))
                modeled = [(p[0], p[1], heading)
                           for p, heading in zip(poses, prepared['body_headings'])]
                clear, sample, hit = sampled_precision_clearance(
                    [current] + modeled, [box for _, box in scene], self._lift_position)
                if not clear:
                    raise ValueError(f'precision scene collision at sample {sample}: {hit}')
                self._stage_validation_pending = True
                self._dock_precision_mode = True
                self._dock_pending = (self._plan_generation, station, path, current,
                                      self.get_clock().now())
                self._report(f'{station}: precision scene checked ({sample} samples); '
                             'waiting for the simulator docking guard.')
                return
            # The loaded path was checked above in 3D (robot, cargo, legs and
            # stand heights). A flat 2D costmap cannot represent driving
            # between the stand legs, so it checks only the open part of the
            # approach, outside the stand area; under the stand the 3D check
            # decides, as for an empty robot.
            center = getattr(self, '_station_centers', {}).get(station)
            outside = NavPath()
            outside.header = path.header
            outside.poses = [point for point in path.poses if center is None or math.hypot(
                point.pose.position.x - center[0],
                point.pose.position.y - center[1]) > STAND_AREA_RADIUS_M]
            if len(outside.poses) < 2:
                self._report(f'{station}: already inside the stand area; the 3D loaded '
                             'clearance check decides segment 2.')
                self._queued_precision_poses = None
                self._precision_station = station
                self._report(f'{station}: starting precision segment 2 automatically.')
                self._follow(path)
                return
            if not self._path_validator.service_is_ready():
                raise ValueError('path validation service unavailable')
            request = IsPathValid.Request()
            request.path = outside
            request.consider_unknown_as_obstacle = True
            self._stage_validation_pending = True
            token = self._plan_generation
            self._report(f'{station}: segment 1 complete; validating precision segment 2.')
            future = self._path_validator.call_async(request)
            future.add_done_callback(lambda response: self._on_precision_validated(
                response, token, station, path, current, checked=outside))
        except Exception as exc:
            self._stage_validation_pending = False
            self._queued_precision_poses = None
            self._report(f'Precision segment blocked: {exc}. Robot stopped.')
            self._execution_done(False, GoalStatus.STATUS_ABORTED)

    def _on_precision_validated(self, future, generation, station, path, preview_pose,
                                checked=None):
        if generation != self._plan_generation or not self._stage_validation_pending:
            return
        self._stage_validation_pending = False
        try:
            result = future.result()
            if not result.success:
                raise ValueError('segment 2 costmap validation service failed')
            if not result.is_valid:
                indices = list(result.invalid_pose_indices)
                validated = checked if checked is not None else path
                if indices and 0 <= indices[0] < len(validated.poses):
                    index = indices[0]
                    position = validated.poses[index].pose.position
                    raise ValueError(
                        f'segment 2 global costmap rejected pose {index} '
                        f'at map x={position.x:.3f}, y={position.y:.3f}; '
                        'obstacle, unknown space, or footprint conflict')
                raise ValueError('segment 2 failed the global costmap check; '
                                 'no valid rejected pose index was returned')
            if not self._motion_armed:
                raise ValueError('simulator commands are disarmed')
            if not plan_is_current(preview_pose, self._robot_pose())['current']:
                raise ValueError('robot pose changed while validating segment 2')
            self._queued_precision_poses = None
            self._precision_station = station
            self._report(f'{station}: starting precision segment 2 automatically.')
            self._follow(path)
        except Exception as exc:
            self._queued_precision_poses = None
            self._report(f'Precision segment blocked: {exc}. Robot stopped.')
            self._execution_done(False, GoalStatus.STATUS_ABORTED)

    def cancel_navigation(self):
        self._cancel_requested = True
        self._motion_armed = False
        self._continue_after_undock = False
        self._pending_rotation = None
        if getattr(self, '_rotation', None) is not None:
            self._end_rotation(False, 'operator cancelled the turn in place')
            return
        if self._cargo_task is not None:
            self._advance_cargo_operation()
            return
        if self._follow_goal_pending:
            self._queued_precision_poses = None
            self._report('Cancellation requested while waiting for Nav2 goal acceptance.')
            return
        if self._departure_pending is not None:
            self._departure_pending = None
            self._departure_target = None
            self._report('Pending loaded departure cancelled; no navigation command sent.')
            return
        if self._stage_validation_pending:
            self._stage_validation_pending = False
            self._dock_pending = None
            self._dock_precision_mode = False
            self._queued_precision_poses = None
            self._plan_generation += 1
            self._report('Pending precision segment cancelled; no command sent.')
            return
        if self._qr_alignment_active:
            self._stop_qr_alignment('operator cancelled precision alignment')
            return
        if self._active_follow is None:
            self._report('No active Nav2 trajectory to cancel.')
            return
        self._active_follow.cancel_goal_async()
        self._report('Nav2 trajectory cancellation requested.')

    def lift(self, height, remember=True):
        # Commands stay inside the lift's working range (lift_geometry).
        height = min(max(float(height), LIFT_DOWN_M), LIFT_UP_M)
        if remember:
            self._lift_target = height
        message = Float64MultiArray()
        message.data = [height]
        self._lift_pub.publish(message)
        self._report(f'Lift command published: {height:.3f} m.')

    def cargo(self, station, attach):
        (self._attach if attach else self._detach)[station].publish(Empty())
        verb = 'attach' if attach else 'detach'
        self._report(f'Cargo {verb} command sent for station {station}.')


def _status_html(message):
    """Colour the operator log: red needs attention, orange is a warning."""
    import html
    text = html.escape(message)
    upper = message.upper()
    if any(word in upper for word in ('FAILED', 'INCOMPLETE', 'BLOCKED', 'REJECTED',
                                      'UNAVAILABLE', 'STOPPED.', 'ABORT')):
        return f'<span style="color:#c62828"><b>{text}</b></span>'
    if any(word in upper for word in ('MOTION GUARD', 'HOLD', 'WARNING', 'STALE')):
        return f'<span style="color:#e65100">{text}</span>'
    return text


class OperatorPanel(Plugin):
    """RQt entry point for planning missions and guarded simulator commands."""

    def __init__(self, context):
        super().__init__(context)
        self.setObjectName('OperatorPanel')
        self._routes = self._load_routes()
        self._docking_routes = self._load_docking_routes()
        self._mission_paths = self._load_mission_paths()
        self._station_entries = self._load_station_entries()
        self._widget = QWidget()
        self._widget.setWindowTitle('Tugger AMR Operator Panel')
        layout = QVBoxLayout(self._widget)
        layout.addWidget(QLabel('Tugger AMR V2 — Gazebo operator panel'))
        self._armed = QCheckBox('Arm Gazebo motion, lift, and cargo commands')
        self._armed.setToolTip('Unchecked is preview-only. Arming is required for every command.')
        self._armed.toggled.connect(self._arming_changed)
        layout.addWidget(self._armed)
        self._turn_in_place = QCheckBox('Turn in place (54-degree steering; only where the '
                                        'turning circle is clear)')
        self._turn_in_place.setToolTip(
            'Off: car-like driving only, exactly as before. On: at a station the robot may '
            'turn in place and leave by another gate when that is faster.')
        self._turn_in_place.toggled.connect(self._turn_in_place_changed)
        layout.addWidget(self._turn_in_place)

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
        attach = QPushButton('ATTACH cargo (lift up)')
        attach.clicked.connect(lambda: self._bridge.select_cargo_action('attach'))
        detach = QPushButton('DETACH cargo (set down)')
        detach.clicked.connect(lambda: self._bridge.select_cargo_action('detach'))
        actions = QHBoxLayout()
        actions.addWidget(attach)
        actions.addWidget(detach)
        mission_layout.addLayout(actions, len(buttons) + 1, 0)
        run = QPushButton('RUN selected mission')
        run.clicked.connect(
            lambda: self._bridge.run_selected_mission_with_retry(self._armed.isChecked()))
        mission_layout.addWidget(run, len(buttons) + 2, 0)
        estop = QPushButton('EMERGENCY STOP')
        estop.setStyleSheet('background-color: #c62828; color: white; font-weight: bold;')
        estop.clicked.connect(lambda: self._bridge.emergency_stop(True))
        reset = QPushButton('Reset safety (after the cause is cleared)')
        reset.clicked.connect(lambda: self._bridge.reset_safety())
        safety_row = QHBoxLayout()
        safety_row.addWidget(estop)
        safety_row.addWidget(reset)
        layout.addLayout(safety_row)
        cancel = QPushButton('Cancel active navigation')
        cancel.clicked.connect(lambda: self._bridge.cancel_navigation())
        mission_layout.addWidget(cancel, len(buttons) + 3, 0)
        layout.addWidget(mission_box)

        # Cargo is attached and detached with ATTACH/DETACH above, which check
        # the station memory. These are service controls for the lift only.
        cargo_box = QGroupBox('Service: lift only (when armed)')
        cargo_layout = QGridLayout(cargo_box)
        controls = [
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
        self._bridge = OperatorBridge(
            self._report, self._planned, self._finished,
            approach_provider=self._docking_routes.approach,
            station_layout=self._routes.database,
            live_planner=self._make_live_planner())
        # Paths to any station, so a mission can be chained through HOME.
        self._bridge._candidate_provider = lambda destination: (
            self._mission_paths.paths_to(destination),
            self._station_entries.get(destination, []))
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

    def _tools_directory(self):
        return Path(__file__).resolve().parents[3] / 'tools' / 'v1_manhattan_bezier'

    def _load_mission_paths(self):
        return MissionPathRepository(self._tools_directory() / 'mission_paths')

    def _make_live_planner(self):
        try:
            map_yaml = (Path(get_package_share_directory('tugger_navigation')) /
                        'maps' / 'warehouse_map.yaml')
        except PackageNotFoundError:
            map_yaml = (Path(__file__).resolve().parents[3] / 'src' / 'tugger_navigation' /
                        'maps' / 'warehouse_map.yaml')
        try:
            world = (Path(get_package_share_directory('tugger_gazebo')) /
                     'worlds' / 'tugger_warehouse.sdf')
        except PackageNotFoundError:
            world = (Path(__file__).resolve().parents[3] / 'src' / 'tugger_gazebo' /
                     'worlds' / 'tugger_warehouse.sdf')
        return LiveRoutePlanner(self._tools_directory(), map_yaml, world=world)

    def _load_docking_routes(self):
        try:
            directory = Path(get_package_share_directory('tugger_control')) / 'docking'
            gazebo = Path(get_package_share_directory('tugger_gazebo'))
            description = Path(get_package_share_directory('tugger_description'))
            candidates = directory / 'entrance_candidates.json'
        except PackageNotFoundError:
            directory = (Path(__file__).resolve().parents[3] / 'tools'
                         / 'v1_manhattan_bezier' / 'phase14' / 'routes_refined')
            root = Path(__file__).resolve().parents[3]
            gazebo = root / 'src/tugger_gazebo'
            description = root / 'src/tugger_description'
            candidates = root / 'tools/mission_approaches/entrance_candidates.json'
        sources = {'world': gazebo / 'worlds/tugger_warehouse.sdf',
                   'robot': description / 'urdf/tugger.urdf.xacro',
                   'database': self._routes.directory / 'route_database.json'}
        return PrecisionDockingRepository(directory, candidates, sources)

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
        deadline = time.monotonic() + 0.008
        for _ in range(12):
            rclpy.spin_once(self._bridge, timeout_sec=0.0)
            if time.monotonic() >= deadline:
                break

    def _report(self, message):
        self._status.append(_status_html(message))

    def _turn_in_place_changed(self, enabled):
        self._bridge.turn_in_place_enabled = bool(enabled)
        self._report('Turn in place ' + ('ON: used at a station when it is faster and the '
                                         'turning circle is clear.' if enabled else
                                         'OFF: car-like driving only.'))

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
        if hasattr(self, '_bridge'):
            self._bridge._motion_armed = armed
            if not armed and (self._bridge._active_follow is not None or
                              self._bridge._qr_alignment_active or
                              self._bridge._stage_validation_pending or
                              self._bridge._cargo_task is not None):
                self._bridge.cancel_navigation()
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
            candidates = self._mission_paths.paths_to(destination)
        except (OSError, ValueError, KeyError) as exc:
            self._report(f'Destination-route load rejected: {exc}')
            return
        self._bridge.start_mission(
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
        for source in CARGO_STATIONS:
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
