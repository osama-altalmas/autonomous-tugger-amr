#!/usr/bin/env python3
"""Refined localization: AMCL's pose, sharpened by a scan-to-map match.

AMCL (tugger_navigation/config/amcl.yaml) runs as before but no longer
publishes the map -> odom transform; this node does. After every AMCL update
it fits both corner scanners' raw scans (each at its own stamp, through
odometry) to the localization map, starting from AMCL's pose
(scan_refinement.refine). A good fit replaces AMCL's transform; anything else
passes AMCL's through unchanged, so the robot is never worse off than with
AMCL alone. Measured offline on recorded runs: AMCL 3.0 cm front-to-back and
2.1 cm sideways rms; refined well under 1 cm wherever the fit is accepted.

Station zones: marker_localization raises AMCL's update thresholds near a
stand the robot works at, so the pose rides on odometry from the exact floor
marker / stand-leg fix. The refinement must not pull that pose back to the
laser either, so while AMCL's update_min_d is raised (seen on
/parameter_events) AMCL's transform is passed through untouched.
"""
import collections
import json
import math

import numpy as np
import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped
from nav_msgs.msg import OccupancyGrid
from rcl_interfaces.msg import ParameterEvent
from rcl_interfaces.srv import GetParameters
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float64, String
from tf2_ros import Buffer, TransformBroadcaster, TransformListener

from tugger_control import scan_refinement as sr

SCANS = ('/front_left/scan', '/rear_right/scan')
# Same post-dating of the transform as AMCL's transform_tolerance.
TRANSFORM_TOLERANCE_S = 0.3
# A scan is used for an AMCL update at most this far from its stamp.
MAX_SCAN_GAP_S = 0.1
# AMCL's update_min_d at or above this means marker_localization suspended
# laser updates (it sets 100 m in a station zone, 0.10 m otherwise).
SUSPENDED_MIN_D = 10.0
# Longest wait for odometry to cover an AMCL pose's stamp before the latest
# odometry is used instead (start-up only, in practice).
PENDING_TIMEOUT_S = 2.0

# Keeping AMCL near the refined pose. AMCL alone drifts 2-3 cm off (a
# map-dependent offset while driving, measured live 2026-11-14), and it is
# the fallback whenever a fit is refused. After RECENTRE_GOOD_FITS good fits
# in a row, with at least two directions pinned, AMCL is re-seeded at the
# refined pose when it is more than RECENTRE_SHIFT_M / RECENTRE_TURN_RAD
# away, at most every RECENTRE_INTERVAL_S and never in a station zone. The
# robot drives on the refined pose either way, so it feels nothing; only the
# fallback improves. In an aisle the scan pins only the cross-aisle position
# and the heading; there the refined pose is AMCL's along the aisle, and the
# re-seed keeps AMCL's own uncertainty along it (fault test 2026-11-14: a
# rack moved 20 cm pulled AMCL 18 cm across a side aisle, and with only fully
# pinned fits allowed it was never re-seeded).
RECENTRE_GOOD_FITS = 10
RECENTRE_SHIFT_M = 0.03
RECENTRE_TURN_RAD = math.radians(0.5)
RECENTRE_INTERVAL_S = 5.0
RECENTRE_POSITION_STD_M = 0.02
RECENTRE_YAW_STD_RAD = math.radians(0.6)
RECENTRE_MIN_CONSTRAINED = 2
# A fit far from AMCL (beyond scan_refinement.MAX_SHIFT_M) is still used when
# it continues the refined pose the robot was driving on a moment ago: the
# limit guards against a jump, and that is no jump; AMCL is what drifted.
CONTINUITY_S = 2.0
CONTINUITY_SHIFT_M = 0.05
CONTINUITY_TURN_RAD = math.radians(1.0)
CONTINUITY_MAX_FROM_AMCL_M = 0.50
# Watchdog: AMCL and the scan match disagreeing by more than this for
# longer than DISAGREE_S is reported (warning only; the gated pose is used).
DISAGREE_SHIFT_M = 0.10
DISAGREE_TURN_RAD = math.radians(3.0)
DISAGREE_S = 3.0
DISAGREE_REPEAT_S = 15.0


def should_recentre(good_fits, constrained, shift_m, turn_rad, since_last_s, suspended):
    """Re-seed AMCL at the refined pose? (pure; see the constants above)"""
    return (not suspended and good_fits >= RECENTRE_GOOD_FITS and
            constrained >= RECENTRE_MIN_CONSTRAINED and
            (shift_m > RECENTRE_SHIFT_M or turn_rad > RECENTRE_TURN_RAD) and
            (since_last_s is None or since_last_s >= RECENTRE_INTERVAL_S))


def continues(previous, candidate, robot_odom, since_s, from_amcl_m):
    """Does a fit refused only for its distance from AMCL continue the last fit?"""
    if previous is None or since_s is None or since_s > CONTINUITY_S:
        return False
    if from_amcl_m > CONTINUITY_MAX_FROM_AMCL_M:
        return False
    a = sr.compose(tuple(previous), tuple(robot_odom))
    b = sr.compose(tuple(candidate), tuple(robot_odom))
    turn = abs(math.atan2(math.sin(b[2] - a[2]), math.cos(b[2] - a[2])))
    return math.hypot(b[0] - a[0], b[1] - a[1]) <= CONTINUITY_SHIFT_M and turn <= CONTINUITY_TURN_RAD


def recentre_covariance(free_directions, amcl_covariance):
    """Re-seed covariance (x, y, yaw): tight where the scan pinned the pose,
    AMCL's own spread (at least the tight value) along a free direction."""
    tight = np.diag([RECENTRE_POSITION_STD_M ** 2, RECENTRE_POSITION_STD_M ** 2,
                     (RECENTRE_YAW_STD_RAD * sr.ROTATION_LENGTH_M) ** 2])
    amcl = np.array([[amcl_covariance[i * 6 + j] for j in (0, 1, 5)] for i in (0, 1, 5)])
    scale = np.diag([1.0, 1.0, sr.ROTATION_LENGTH_M])
    amcl = scale @ amcl @ scale
    cov = tight.copy()
    for v in free_directions:
        v = np.asarray(v, dtype=float)
        extra = max(float(v @ amcl @ v), float(v @ tight @ v)) - float(v @ tight @ v)
        cov += extra * np.outer(v, v)
    unscale = np.diag([1.0, 1.0, 1.0 / sr.ROTATION_LENGTH_M])
    return unscale @ cov @ unscale


# Odometry calibration: the distance driven between two fully pinned fits,
# as the scan match measures it, over the distance odometry reported. Tyres
# wear and compress under load, so the true wheel radius drifts from the
# nominal one; a 2% error put the robot 5.5 cm off at HOME's camera stage,
# where laser updates pause and odometry alone carries the approach (fault
# test 2026-11-14). The median of SCALE_SAMPLES segments of at least
# SCALE_SEGMENT_M corrects the odometry's scale (four_ws_odometry).
SCALE_SEGMENT_M = 4.0
SCALE_SAMPLES = 5
SCALE_DEADBAND = 0.002
SCALE_LIMITS = (0.95, 1.05)
SCALE_WARN = 0.03


class OdometryScale:
    """Estimate the odometry scale factor from fully pinned scan-match fits."""

    def __init__(self):
        self.scale = 1.0
        self.anchor = None
        self.samples = []

    def reset(self):
        self.anchor, self.samples = None, []

    def add(self, odom_pose, map_pose):
        """Feed one pinned fit; returns a new scale when it should change, else None."""
        if self.anchor is None:
            self.anchor = (odom_pose, map_pose)
            return None
        (ox, oy, _), (mx, my, _) = self.anchor
        odom_m = math.hypot(odom_pose[0] - ox, odom_pose[1] - oy)
        if odom_m < SCALE_SEGMENT_M:
            return None
        ratio = math.hypot(map_pose[0] - mx, map_pose[1] - my) / odom_m
        self.anchor = (odom_pose, map_pose)
        if 0.9 <= ratio <= 1.1:
            self.samples.append(ratio)
        if len(self.samples) < SCALE_SAMPLES:
            return None
        median = float(np.median(self.samples))
        self.samples = []
        if abs(median - 1.0) <= SCALE_DEADBAND:
            return None
        self.scale = min(max(self.scale * median, SCALE_LIMITS[0]), SCALE_LIMITS[1])
        self.anchor = None          # the next segment is measured on the new scale
        return self.scale


# Integrity check, once a second and also in station zones (where AMCL and
# the refinement pause): the current scan scored against the map at the pose
# in use. CHECK_BAD_COUNT poor scores in a row start a search around it
# (scan_refinement.search). A unique, clearly better pose is taken over
# (relocalized); none means the robot is lost, motion stops and RUN is
# refused until the pose is known again (operator_panel).
CHECK_PERIOD_S = 1.0
CHECK_OK_SCORE = 0.45
CHECK_BAD_COUNT = 3
RELOCALIZE_GAIN = 1.5          # the found pose must score this much better
RELOCALIZE_MIN_SCORE = 0.45
SAME_POSE_M = 0.10             # a search ending this close: just clutter
SAME_POSE_RAD = math.radians(2.0)
SEARCH_BACKOFF_S = 10.0


def judge_search(current_score, best_score, runner_up, moved_m, turned_rad):
    """'same', 'relocalize' or 'lost' for a finished search (pure)."""
    if moved_m <= SAME_POSE_M and turned_rad <= SAME_POSE_RAD:
        return 'same' if best_score >= RELOCALIZE_MIN_SCORE else 'lost'
    if (best_score >= max(RELOCALIZE_MIN_SCORE, RELOCALIZE_GAIN * current_score) and
            runner_up <= sr.SEARCH_UNIQUE_RATIO):
        return 'relocalize'
    return 'lost'


def disagrees(shift_m, turn_rad):
    return shift_m > DISAGREE_SHIFT_M or turn_rad > DISAGREE_TURN_RAD


def _yaw(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def _stamp(header):
    return header.stamp.sec + header.stamp.nanosec * 1e-9


class RefinedLocalization(Node):
    def __init__(self):
        super().__init__('tugger_refined_localization')
        self.set_parameters([Parameter('use_sim_time', value=True)])
        # Test switch: false = fits are still computed and reported but not
        # used, exactly as if every fit were refused (AMCL's pose is used).
        # Measures the fallback; never set on a working robot.
        self.declare_parameter('use_refinement', True)
        self._tf = Buffer()
        self._listener = TransformListener(self._tf, self)
        self._broadcaster = TransformBroadcaster(self)
        self._field = None
        self._mounts = {}
        self._scans = {topic: collections.deque(maxlen=8) for topic in SCANS}
        self._suspended = False
        self._map_odom = None
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(OccupancyGrid, '/localization_map', self._on_map, latched)
        for topic in SCANS:
            self.create_subscription(LaserScan, topic,
                                     lambda msg, t=topic: self._scans[t].append(msg), 10)
        # AMCL publishes amcl_pose latched (nav2 LatchedPublisherQoS): a
        # latched subscription also gets the pose sent before this node joined,
        # which at start-up, standing still, is AMCL's only one.
        amcl_qos = QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                              reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self._on_amcl, amcl_qos)
        self.create_subscription(ParameterEvent, '/parameter_events', self._on_parameters, 10)
        # A suspension set before this node listened is not in the events:
        # ask AMCL once for its current threshold.
        self._amcl_get = self.create_client(GetParameters, '/amcl/get_parameters')
        self._initial_query = None
        self._query_timer = self.create_timer(1.0, self._query_suspension)
        self._status = self.create_publisher(String, '/tugger/localization/refinement', 10)
        self._initial_pose = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)
        self._warning = self.create_publisher(String, '/tugger/localization/warning', 10)
        self._good_fits = 0
        self._last_recentre = None
        self._last_refined = None       # (time s, map -> odom) of the last used fit
        self._odometry_scale = OdometryScale()
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._scale_pub = self.create_publisher(Float64, '/tugger/odometry/scale', latched)
        self._scale_pub.publish(Float64(data=1.0))
        self._disagree_since = None
        self._last_warning = None
        # Integrity check and relocalization (see CHECK_* above).
        self._search = None
        self._bad_checks = 0
        self._search_backoff_until = 0.0
        self._relocalized_until = 0.0
        self._lost = None
        self._lost_pub = self.create_publisher(String, '/tugger/localization/lost', latched)
        self._publish_lost(None)
        self.create_timer(CHECK_PERIOD_S, self._integrity_check)
        # Keep the transform alive between AMCL updates, as AMCL does.
        self.create_timer(0.05, self._republish)
        self.get_logger().info('refined localization: AMCL + scan-to-map match on both scanners')

    def _on_map(self, msg):
        info = msg.info
        occupied = sr.occupancy_from_grid(msg.data, info.width, info.height)
        self._field = sr.DistanceField(occupied, info.resolution,
                                       (info.origin.position.x, info.origin.position.y))
        self.get_logger().info(f'localization map {info.width} x {info.height} '
                               f'at {info.resolution * 100:.0f} cm')

    def _on_parameters(self, msg):
        if msg.node.rstrip('/').split('/')[-1] != 'amcl':
            return
        for p in list(msg.changed_parameters) + list(msg.new_parameters):
            if p.name == 'update_min_d':
                self._set_suspended(p.value.double_value >= SUSPENDED_MIN_D)

    def _set_suspended(self, suspended):
        if suspended != self._suspended:
            self._suspended = suspended
            self.get_logger().info('refinement ' + ('paused: station zone'
                                                    if suspended else 'active'))

    def _query_suspension(self):
        if self._initial_query is None:
            if not self._amcl_get.service_is_ready():
                return
            request = GetParameters.Request()
            request.names = ['update_min_d']
            self._initial_query = self._amcl_get.call_async(request)
            return
        if not self._initial_query.done():
            return
        try:
            values = self._initial_query.result().values
            if values:
                self._set_suspended(values[0].double_value >= SUSPENDED_MIN_D)
            self._query_timer.cancel()
        except Exception:
            self._initial_query = None               # ask again

    def _odom_pose(self, stamp):
        t = self._tf.lookup_transform('odom', 'base_footprint', rclpy.time.Time.from_msg(stamp),
                                      timeout=Duration(seconds=0.0))
        return (t.transform.translation.x, t.transform.translation.y, _yaw(t.transform.rotation))

    def _mount(self, frame):
        if frame not in self._mounts:
            t = self._tf.lookup_transform('base_footprint', frame, rclpy.time.Time())
            self._mounts[frame] = (t.transform.translation.x, t.transform.translation.y,
                                   _yaw(t.transform.rotation))
        return self._mounts[frame]

    def _points(self, amcl_stamp):
        """Both scanners' points in odom, each scan nearest the AMCL update."""
        parts = []
        for topic in SCANS:
            near = [m for m in self._scans[topic]
                    if abs(_stamp(m.header) - amcl_stamp) <= MAX_SCAN_GAP_S]
            if not near:
                continue
            scan = min(near, key=lambda m: abs(_stamp(m.header) - amcl_stamp))
            try:
                mount = self._mount(scan.header.frame_id)
                odom = self._odom_pose(scan.header.stamp)
            except Exception:
                continue
            parts.append(sr.scan_points(scan.ranges, scan.angle_min, scan.angle_increment,
                                        mount, odom, range_min=max(scan.range_min, 0.1),
                                        range_max=min(scan.range_max, 30.0)))
        return np.concatenate(parts) if parts else None

    def _on_amcl(self, msg):
        self._pending_amcl = (msg, self.get_clock().now())
        self._process_pending()

    def _process_pending(self):
        """Handle the latest AMCL pose once odometry covers its stamp.

        At start-up AMCL's first pose (and, standing still, its only one) can
        arrive before this node's TF buffer holds odometry for its stamp.
        Dropping it left no map -> odom at all and Nav2 waited forever for the
        map frame (2026-11-14). It is kept and retried; after
        PENDING_TIMEOUT_S the latest odometry stands in.
        """
        pending = getattr(self, '_pending_amcl', None)
        if pending is None:
            return
        msg, since = pending
        try:
            robot_odom = self._odom_pose(msg.header.stamp)
        except Exception as exc:
            waited = (self.get_clock().now() - since).nanoseconds / 1e9
            if waited < PENDING_TIMEOUT_S:
                return
            try:
                robot_odom = self._odom_pose(rclpy.time.Time().to_msg())
            except Exception:
                self.get_logger().warning(f'no odometry for the AMCL pose: {exc}',
                                          throttle_duration_sec=5.0)
                return
        self._pending_amcl = None
        p = msg.pose.pose
        amcl = (p.position.x, p.position.y, _yaw(p.orientation))
        prior = sr.compose(amcl, sr.inverse(robot_odom))   # AMCL's map -> odom
        # AMCL's robot pose and, after a fit, the refined one (for the RViz
        # correction view in tools/mission_view/localization_view.py).
        status = {'stamp': _stamp(msg.header), 'used': 'amcl',
                  'amcl': [round(v, 4) for v in amcl]}
        result = None
        if self._suspended:
            status['reason'] = 'station zone'
        elif self._field is None:
            status['reason'] = 'no map yet'
        else:
            points = self._points(_stamp(msg.header))
            if points is None:
                status['reason'] = 'no scans at the AMCL stamp'
            else:
                result = sr.refine(self._field, points, prior, robot_odom)
                status.update(reason=result.reason, inliers=result.inliers,
                              fraction=round(result.inlier_fraction, 3),
                              rms_cm=round(result.rms_m * 100, 2),
                              shift_cm=round(result.shift_m * 100, 2),
                              turn_deg=round(math.degrees(result.turn_rad), 3),
                              constrained=result.constrained,
                              refined=[round(v, 4) for v in
                                       sr.compose(result.transform, robot_odom)])
                if not self.get_parameter('use_refinement').value:
                    result.accepted = False
                    result.reason = status['reason'] = 'switched off (test)'
                elif self._continues_last_fit(result, robot_odom):
                    result.accepted = True
                    result.reason = status['reason'] = (
                        f'ok, continuing the last fit ({result.reason})')
                if result.accepted:
                    status['used'] = 'refined'
                    if result.constrained == 3:
                        self._calibrate_odometry(robot_odom,
                                                 sr.compose(result.transform, robot_odom))
                    self._last_refined = (self.get_clock().now().nanoseconds / 1e9,
                                          result.transform)
        if result is not None and result.accepted:
            self._map_odom = result.transform
        elif not self._keeps_relocalized(prior, robot_odom):
            self._map_odom = prior
        self._backup_and_watchdog(msg, result, robot_odom, status)
        self._publish(msg.header.stamp)
        out = String()
        out.data = json.dumps(status)
        self._status.publish(out)

    def _backup_and_watchdog(self, msg, result, robot_odom, status):
        """Keep AMCL near the refined pose; report lasting disagreement."""
        now = self.get_clock().now().nanoseconds / 1e9
        accepted = result is not None and result.accepted
        self._good_fits = (self._good_fits + 1 if accepted and
                           result.constrained >= RECENTRE_MIN_CONSTRAINED else 0)
        if result is not None:
            since = None if self._last_recentre is None else now - self._last_recentre
            if accepted and should_recentre(self._good_fits, result.constrained, result.shift_m,
                                            result.turn_rad, since, self._suspended):
                refined = sr.compose(result.transform, robot_odom)
                self._send_initial_pose(msg.header.stamp, refined, recentre_covariance(
                    result.free_directions, msg.pose.covariance))
                self._last_recentre = now
                status['recentred'] = True
            # Watchdog: the scan match and AMCL far apart (refused for size,
            # or an accepted fit near the limits) for longer than DISAGREE_S.
            if disagrees(result.shift_m, result.turn_rad) and result.inlier_fraction >= 0.5:
                if self._disagree_since is None:
                    self._disagree_since = now
                lasting = now - self._disagree_since
                if lasting >= DISAGREE_S and (self._last_warning is None or
                                              now - self._last_warning >= DISAGREE_REPEAT_S):
                    self._last_warning = now
                    self._warn(f'laser localization and the scan match disagree by '
                               f'{result.shift_m * 100:.0f} cm and '
                               f'{math.degrees(result.turn_rad):.1f} deg for {lasting:.0f} s; '
                               f'using the {"scan match" if accepted else "AMCL"} pose '
                               f'({result.reason})')
            elif self._disagree_since is not None:
                if self._last_warning is not None:
                    self._warn('laser localization and the scan match agree again')
                self._disagree_since = self._last_warning = None

    def _calibrate_odometry(self, robot_odom, map_pose):
        scale = self._odometry_scale.add(robot_odom, map_pose)
        if scale is None:
            return
        self._scale_pub.publish(Float64(data=scale))
        self.get_logger().info(f'odometry scale corrected to {scale:.4f}')
        if abs(scale - 1.0) > SCALE_WARN:
            self._warn(f'wheel odometry measures {abs(scale - 1.0) * 100:.1f}% '
                       f'{"short" if scale > 1.0 else "long"}: check tyre wear and wheel size')

    def _continues_last_fit(self, result, robot_odom):
        """Refused only for its distance from AMCL, yet continuous with the last fit?"""
        if result.accepted or not result.reason.endswith('cm from AMCL'):
            return False
        if self._last_refined is None:
            return False
        since = self.get_clock().now().nanoseconds / 1e9 - self._last_refined[0]
        return continues(self._last_refined[1], result.transform, robot_odom, since,
                         result.shift_m)

    def _send_initial_pose(self, stamp, pose, covariance3):
        p = PoseWithCovarianceStamped()
        p.header.stamp = stamp
        p.header.frame_id = 'map'
        p.pose.pose.position.x, p.pose.pose.position.y = pose[0], pose[1]
        p.pose.pose.orientation.z = math.sin(pose[2] / 2)
        p.pose.pose.orientation.w = math.cos(pose[2] / 2)
        covariance = [0.0] * 36
        for i, a in enumerate((0, 1, 5)):
            for j, b in enumerate((0, 1, 5)):
                covariance[a * 6 + b] = float(covariance3[i][j])
        p.pose.covariance = covariance
        self._initial_pose.publish(p)

    def _warn(self, text):
        self.get_logger().warning(text)
        out = String()
        out.data = text
        self._warning.publish(out)

    def _publish(self, stamp=None):
        if self._map_odom is None:
            return
        t = TransformStamped()
        # Always stamped from now: an AMCL update can arrive after a timer
        # republish, and a transform stamped earlier than that one would not
        # be the newest in the buffer for up to a timer period.
        base = self.get_clock().now()
        if stamp is not None:
            base = max(base, rclpy.time.Time.from_msg(stamp))
        t.header.stamp = (base + Duration(seconds=TRANSFORM_TOLERANCE_S)).to_msg()
        t.header.frame_id = 'map'
        t.child_frame_id = 'odom'
        x, y, yaw = self._map_odom
        t.transform.translation.x, t.transform.translation.y = x, y
        t.transform.rotation.z, t.transform.rotation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        self._broadcaster.sendTransform(t)

    def _keeps_relocalized(self, prior, robot_odom):
        """Right after a relocalization AMCL may still report the old pose."""
        now = self.get_clock().now().nanoseconds / 1e9
        if self._map_odom is None or now > self._relocalized_until:
            return False
        a = sr.compose(tuple(prior), tuple(robot_odom))
        b = sr.compose(tuple(self._map_odom), tuple(robot_odom))
        return math.hypot(a[0] - b[0], a[1] - b[1]) > SAME_POSE_M

    def _publish_lost(self, reason):
        self._lost = reason
        self._set_state('lost' if reason is not None else 'ok', reason)

    def _set_state(self, state, reason=None):
        """ok, suspect (checking a poor fit) or lost; latched, on change only."""
        if state == getattr(self, '_state', None) and state != 'lost':
            return
        self._state = state
        out = String()
        out.data = json.dumps({'lost': state == 'lost', 'state': state,
                               'reason': reason or ''})
        self._lost_pub.publish(out)

    def _latest_points(self):
        """Newest scans' points in odom and the robot's odom pose at that stamp."""
        newest = [self._scans[t][-1] for t in SCANS if self._scans[t]]
        if not newest:
            return None, None
        stamp = min(_stamp(m.header) for m in newest)
        points = self._points(stamp)
        if points is None or len(points) < sr.MIN_INLIERS:
            return None, None
        scan = min(newest, key=lambda m: _stamp(m.header))
        try:
            robot_odom = self._odom_pose(scan.header.stamp)
        except Exception:
            return None, None
        return points, robot_odom

    def _integrity_check(self):
        if self._field is None or self._map_odom is None:
            return
        if self._search is not None:
            return                      # stepped in _republish
        points, robot_odom = self._latest_points()
        if points is None:
            return
        score = sr.fit_score(self._field, points, self._map_odom)
        now = self.get_clock().now().nanoseconds / 1e9
        if score >= CHECK_OK_SCORE:
            self._bad_checks = 0
            if self._lost is not None:
                self._publish_lost(None)
                self._warn('localization found again: the scan fits the map')
            self._set_state('ok')
            return
        self._bad_checks += 1
        if self._lost is None and now >= self._search_backoff_until:
            self._set_state('suspect', f'the scan fits the map poorly here ({score:.2f})')
        if self._bad_checks < CHECK_BAD_COUNT or now < self._search_backoff_until:
            return
        self._bad_checks = 0
        steps = sr.search_steps(self._field, points, self._map_odom, robot_odom)
        self._search = (steps, points, robot_odom, score, tuple(self._map_odom))

    def _step_search(self, budget_s=0.03):
        """Advance the search a few headings; finish it when done."""
        if self._search is None:
            return
        import time
        end = time.monotonic() + budget_s
        while time.monotonic() < end:
            try:
                step = next(self._search[0])
            except Exception as exc:
                self.get_logger().error(f'relocalization search failed: {exc}')
                self._search = None
                return
            if step is not None:
                self._finish_search(step)
                return

    def _finish_search(self, found):
        _, points, robot_odom, score, searched_from = self._search
        self._search = None
        now = self.get_clock().now().nanoseconds / 1e9
        transform, best, runner_up = found
        before = sr.compose(searched_from, robot_odom)
        after = sr.compose(transform, robot_odom)
        moved = math.hypot(after[0] - before[0], after[1] - before[1])
        turned = abs(math.atan2(math.sin(after[2] - before[2]), math.cos(after[2] - before[2])))
        verdict = judge_search(score, best, runner_up, moved, turned)
        self.get_logger().info(f'integrity search: score {score:.2f} -> {best:.2f} '
                               f'{moved * 100:.0f} cm / {math.degrees(turned):.1f} deg away, '
                               f'runner-up {runner_up:.2f}: {verdict}')
        if verdict == 'same':
            self._search_backoff_until = now + SEARCH_BACKOFF_S
            if self._lost is not None:
                self._publish_lost(None)
            self._set_state('ok')
            return
        if verdict == 'relocalize':
            result = sr.refine(self._field, points, transform, robot_odom)
            if result.accepted or result.reason.endswith('cm from AMCL'):
                transform = result.transform
            self._map_odom = transform
            self._last_refined = (now, transform)
            self._relocalized_until = now + 3.0
            pose = sr.compose(transform, robot_odom)
            self._send_initial_pose(rclpy.time.Time().to_msg(), pose, recentre_covariance(
                (), [0.0] * 36))
            self._publish(None)
            if self._lost is not None:
                self._publish_lost(None)
            self._set_state('ok')
            self._warn(f'RELOCALIZED: the robot was {moved * 100:.0f} cm and '
                       f'{math.degrees(turned):.0f} deg from where localization had it '
                       '(moved by hand?); position taken from the laser scan')
            return
        self._search_backoff_until = now + SEARCH_BACKOFF_S
        if self._lost is None:
            reason = (f'the laser scan does not fit the map here (fit {score:.2f}) and no '
                      f'single nearby position fits clearly (best {best:.2f}, '
                      f'{moved * 100:.0f} cm away)')
            self._publish_lost(reason)
            self._warn(f'LOCALIZATION LOST: {reason}')

    def _republish(self):
        self._process_pending()
        self._publish()
        self._step_search()


def main(args=None):
    rclpy.init(args=args)
    node = RefinedLocalization()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
