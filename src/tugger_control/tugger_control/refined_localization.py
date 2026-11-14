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
from std_msgs.msg import String
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
# in a row, with all three directions pinned, AMCL is re-seeded at the
# refined pose when it is more than RECENTRE_SHIFT_M / RECENTRE_TURN_RAD
# away, at most every RECENTRE_INTERVAL_S and never in a station zone. The
# robot drives on the refined pose either way, so it feels nothing; only the
# fallback improves.
RECENTRE_GOOD_FITS = 10
RECENTRE_SHIFT_M = 0.03
RECENTRE_TURN_RAD = math.radians(0.5)
RECENTRE_INTERVAL_S = 5.0
RECENTRE_POSITION_STD_M = 0.02
RECENTRE_YAW_STD_RAD = math.radians(0.6)
# Watchdog: AMCL and the scan match disagreeing by more than this for
# longer than DISAGREE_S is reported (warning only; the gated pose is used).
DISAGREE_SHIFT_M = 0.10
DISAGREE_TURN_RAD = math.radians(3.0)
DISAGREE_S = 3.0
DISAGREE_REPEAT_S = 15.0


def should_recentre(good_fits, constrained, shift_m, turn_rad, since_last_s, suspended):
    """Re-seed AMCL at the refined pose? (pure; see the constants above)"""
    return (not suspended and good_fits >= RECENTRE_GOOD_FITS and constrained == 3 and
            (shift_m > RECENTRE_SHIFT_M or turn_rad > RECENTRE_TURN_RAD) and
            (since_last_s is None or since_last_s >= RECENTRE_INTERVAL_S))


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
        self._disagree_since = None
        self._last_warning = None
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
                if result.accepted:
                    status['used'] = 'refined'
        self._map_odom = result.transform if result is not None and result.accepted else prior
        self._backup_and_watchdog(msg, result, robot_odom, status)
        self._publish(msg.header.stamp)
        out = String()
        out.data = json.dumps(status)
        self._status.publish(out)

    def _backup_and_watchdog(self, msg, result, robot_odom, status):
        """Keep AMCL near the refined pose; report lasting disagreement."""
        now = self.get_clock().now().nanoseconds / 1e9
        accepted = result is not None and result.accepted
        self._good_fits = self._good_fits + 1 if accepted and result.constrained == 3 else 0
        if result is not None:
            since = None if self._last_recentre is None else now - self._last_recentre
            if accepted and should_recentre(self._good_fits, result.constrained, result.shift_m,
                                            result.turn_rad, since, self._suspended):
                refined = sr.compose(result.transform, robot_odom)
                self._send_initial_pose(msg.header.stamp, refined)
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

    def _send_initial_pose(self, stamp, pose):
        p = PoseWithCovarianceStamped()
        p.header.stamp = stamp
        p.header.frame_id = 'map'
        p.pose.pose.position.x, p.pose.pose.position.y = pose[0], pose[1]
        p.pose.pose.orientation.z = math.sin(pose[2] / 2)
        p.pose.pose.orientation.w = math.cos(pose[2] / 2)
        covariance = [0.0] * 36
        covariance[0] = covariance[7] = RECENTRE_POSITION_STD_M ** 2
        covariance[35] = RECENTRE_YAW_STD_RAD ** 2
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

    def _republish(self):
        self._process_pending()
        self._publish()


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
