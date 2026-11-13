#!/usr/bin/env python3
"""Floor-marker localization: exact position over a station marker (Phase B5).

Laser localization (AMCL) is good to a few centimetres. Docking needs more,
so, as on floor-QR AGVs (KUKA KMP 600P: +-5 mm), the downward camera's view
of a station's red alignment frame, whose map position is known from the
site layout, sets the robot's position whenever a marker is in view. The
correction is sent to AMCL as a new pose estimate with a tight covariance;
heading stays AMCL's (accurate to about a degree; at 10 cm from the marker
centre a degree is under 2 mm).

Corrections are sent only between motions (no motion stage active, robot
standing), at most twice a second, and only when the observed frame is within 0.3 m of
a known marker (so no other red object can pull the estimate).

Station zone: within STATION_ZONE_M of a stand's centre (from its legs),
whatever the motion stage (loaded and unloaded approaches, camera
alignment, departures, turning in place), AMCL's update thresholds are raised,
so it carries the pose forward on odometry from the exact landmark fix taken
at the stop before, instead of pulling it back toward its centimetre-level
laser estimate under the stand. It keeps publishing its transform; the
normal thresholds return when the stage ends. A plain route that only passes
through a zone keeps the normal thresholds: suspending them there made AMCL
absorb metres of travel in one update and jump along the aisle when it left.
"""
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, String
from rclpy.node import Node
from rclpy.parameter import Parameter
from rcl_interfaces.msg import Parameter as ParameterMsg, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformListener

# Staging points are 2.0 m from a stand's centre.
STATION_ZONE_M = 2.2
MATCH_RADIUS_M = 0.30
MIN_CORRECTION_M = 0.005
# Fixes are taken only between motions: no motion stage active (owner STOP)
# and the robot standing (< 1 cm/s). Every fix re-seeds AMCL's particle cloud;
# a re-seed while a path is followed makes the pose jump under the follower
# (measured: exits stalled, and a slow docking end turned the robot 24 deg).
# Between motions - at a staging stop before driving in, or docked before a
# departure - a fix is exact and harmless.
MAX_SPEED_M_S = 0.01
MIN_INTERVAL_S = 0.5
# Reported to AMCL: tight enough to pull the estimate onto the marker, wide
# enough to keep particle diversity for the drive that follows.
POSITION_STD_M = 0.02
YAW_STD_RAD = 0.035


def suspend_laser_updates(near_station, route_transit):
    """Suspend only at a station the robot is working at, not one it passes."""
    return near_station and not route_transit


def zone_action(previous_station, station):
    """What to do when the station-zone state changes.

    'suspend' entering a zone, 'reseed' leaving one (re-seed AMCL at the
    current pose, then restore the normal thresholds a tick later), 'resume'
    at start-up (no zone before), None when nothing changed.
    """
    if station == previous_station:
        return None
    if station:
        return 'suspend'
    return 'reseed' if previous_station else 'resume'


def _yaw(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def marker_positions(world_path):
    """Map (x, y) of every floor marker (models named qr_*) in the site layout."""
    root = ET.parse(world_path).getroot()
    markers = {}
    for model in root.findall('./world/model'):
        name = model.get('name', '')
        if name.startswith('qr_'):
            x, y = (float(v) for v in model.findtext('pose', '0 0 0').split()[:2])
            markers[name] = (x, y)
    return markers


def station_legs(world_path):
    """Leg centres (map) of every station, from the site layout."""
    from tugger_control.warehouse_clearance import static_world_boxes
    legs = {}
    for box in static_world_boxes(world_path):
        model = box.name.split('/', 1)[0]
        if model.endswith('_legs'):
            legs.setdefault(model, []).append((box.x, box.y, max(box.half_x, box.half_y)))
    return legs


LEG_SEARCH_M = 0.15
LEG_MIN_POINTS = 3


def leg_fix(robot_pose, points, legs):
    """Robot map pose from the scan points on a station's legs, or None.

    ``points`` are scan points (x, y) in the robot frame. Each leg's predicted
    position (from the current estimate) gathers the points near it; the
    leg centre lies half a leg behind the seen faces, along the line of sight.
    With at least three legs found, the rigid transform that maps the seen
    centres onto the known ones (2D Kabsch) gives the robot pose.
    """
    import numpy as np
    x, y, yaw = robot_pose
    c, s = math.cos(yaw), math.sin(yaw)
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    seen, known = [], []
    for lx, ly, half in legs:
        dx, dy = lx - x, ly - y
        px, py = c * dx + s * dy, -s * dx + c * dy       # predicted, robot frame
        if math.hypot(px, py) > 4.0 or len(pts) == 0:
            continue
        near = pts[np.hypot(pts[:, 0] - px, pts[:, 1] - py) <= LEG_SEARCH_M]
        if len(near) < LEG_MIN_POINTS:
            continue
        centroid = near.mean(axis=0)
        sight = centroid / max(np.hypot(*centroid), 1e-9)
        seen.append(centroid + sight * half)
        known.append((lx, ly))
    if len(seen) < 3:
        return None
    a, b = np.array(seen), np.array(known)
    ca, cb = a.mean(axis=0), b.mean(axis=0)
    h = (a - ca).T @ (b - cb)
    angle = math.atan2(h[0, 1] - h[1, 0], h[0, 0] + h[1, 1])
    r = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    t = cb - r @ ca
    residual = float(np.max(np.hypot(*((r @ a.T).T + t - b).T)))
    return (float(t[0]), float(t[1]), angle), residual, len(seen)


def correction(robot_pose, frame_offset, markers):
    """Corrected robot (x, y) from an observed frame, or None if no marker matches.

    ``frame_offset`` is the frame centre in the robot frame (forward, left).
    """
    x, y, yaw = robot_pose
    c, s = math.cos(yaw), math.sin(yaw)
    seen = (x + c * frame_offset[0] - s * frame_offset[1],
            y + s * frame_offset[0] + c * frame_offset[1])
    name, known = min(markers.items(), key=lambda item: math.hypot(
        item[1][0] - seen[0], item[1][1] - seen[1]))
    error = (known[0] - seen[0], known[1] - seen[1])
    if math.hypot(*error) > MATCH_RADIUS_M:
        return None
    return name, (x + error[0], y + error[1]), math.hypot(*error)


class MarkerLocalization(Node):
    def __init__(self):
        super().__init__('tugger_marker_localization')
        self.set_parameters([Parameter('use_sim_time', value=True)])
        world = Path(get_package_share_directory('tugger_gazebo')) / 'worlds/tugger_warehouse.sdf'
        self._markers = marker_positions(world)
        self._legs = station_legs(world)
        self.create_subscription(LaserScan, '/scan', self._on_scan, 5)
        self._tf = Buffer()
        self._listener = TransformListener(self._tf, self)
        self._speed = None
        self._last_sent = None
        self._pub = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)
        # Tells the mission layer a landmark has confirmed (or fixed) the pose.
        self._fix_pub = self.create_publisher(String, '/tugger/localization/landmark_fix', 10)
        self.create_subscription(PoseStamped, '/qr_docking/frame_pose', self._on_frame, 10)
        # Frames wait until TF covers their image stamp (usually 10-30 ms).
        self._pending = None
        self.create_timer(0.02, self._retry)
        self.create_subscription(Odometry, '/wheel/odom', self._on_odom, 10)
        # The camera alignment stage measures relative to the marker and checks
        # that the marker stays fixed in the map; a fix during it would look
        # like an inconsistent measurement, so fixes pause while it runs.
        self._owner = 'STOP'
        self.create_subscription(String, '/tugger/motion_owner', self._on_owner, 10)
        self._route_transit = False
        self.create_subscription(Bool, '/tugger/localization/route_transit',
                                 self._on_route_transit, 10)
        self._amcl_params = self.create_client(SetParameters, '/amcl/set_parameters')
        self._station_mode = None
        self._resume_pending = False
        self._centres = [(sum(p[0] for p in legs) / len(legs), sum(p[1] for p in legs) / len(legs))
                         for legs in self._legs.values()]
        # HOME has no stand, but its floor marker fixes the pose exactly in the
        # camera stage; near it, recoveries ride on odometry from that fix.
        if 'qr_home' in self._markers:
            self._centres.append(self._markers['qr_home'])
        self.create_timer(0.2, self._zone_tick)
        self.get_logger().info(f'marker localization: {len(self._markers)} floor markers')

    def _on_scan(self, msg):
        """Station legs in the merged scan: position and heading at a station."""
        now = self.get_clock().now().nanoseconds / 1e9
        if (self._speed is None or self._speed > MAX_SPEED_M_S or
                self._owner != 'STOP' or
                (self._last_sent is not None and now - self._last_sent < MIN_INTERVAL_S)):
            return
        try:
            t = self._tf.lookup_transform('map', 'base_footprint', rclpy.time.Time())
        except Exception:
            return
        robot = (t.transform.translation.x, t.transform.translation.y,
                 _yaw(t.transform.rotation))
        station = min(self._legs.items(), key=lambda item: min(
            math.hypot(lx - robot[0], ly - robot[1]) for lx, ly, _ in item[1]))
        import numpy as np
        ranges = np.asarray(msg.ranges, dtype=float)
        angles = msg.angle_min + msg.angle_increment * np.arange(len(ranges))
        ok = np.isfinite(ranges) & (ranges < 4.5)
        points = np.column_stack((ranges[ok] * np.cos(angles[ok]),
                                  ranges[ok] * np.sin(angles[ok])))
        result = leg_fix(robot, points, station[1])
        if result is None:
            return
        (x, y, yaw), residual, used = result
        shift = math.hypot(x - robot[0], y - robot[1])
        turn = abs(math.atan2(math.sin(yaw - robot[2]), math.cos(yaw - robot[2])))
        if residual > 0.03 or shift > 0.30 or turn > math.radians(10):
            return
        if shift < MIN_CORRECTION_M and turn < math.radians(0.3):
            self._announce(station[0], 'legs', False)
            return
        self._send(msg.header.stamp, x, y, yaw, 0.015, math.radians(0.7))
        self._announce(station[0], 'legs', True)
        if shift > 0.02 or turn > math.radians(1):
            self.get_logger().info(f'{station[0]}: {used} legs, corrected by {shift * 100:.1f} cm '
                                   f'and {math.degrees(turn):.1f} deg')

    def _announce(self, where, kind, applied):
        message = String()
        message.data = json.dumps({'where': where, 'kind': kind, 'applied': applied,
                                   'stamp': self.get_clock().now().nanoseconds / 1e9})
        self._fix_pub.publish(message)

    def _send(self, stamp, x, y, yaw, position_std, yaw_std):
        pose = PoseWithCovarianceStamped()
        pose.header.stamp = stamp
        pose.header.frame_id = 'map'
        pose.pose.pose.position.x, pose.pose.pose.position.y = x, y
        pose.pose.pose.orientation.z = math.sin(yaw / 2)
        pose.pose.pose.orientation.w = math.cos(yaw / 2)
        covariance = [0.0] * 36
        covariance[0] = covariance[7] = position_std ** 2
        covariance[35] = yaw_std ** 2
        pose.pose.covariance = covariance
        self._pub.publish(pose)
        self._last_sent = self.get_clock().now().nanoseconds / 1e9

    def _on_owner(self, msg):
        self._owner = msg.data

    def _on_route_transit(self, msg):
        self._route_transit = bool(msg.data)

    def _zone_tick(self):
        """Station zone by position: AMCL rides on odometry near stands."""
        try:
            t = self._tf.lookup_transform('map', 'base_footprint', rclpy.time.Time())
        except Exception:
            return
        x, y = t.transform.translation.x, t.transform.translation.y
        near = any(math.hypot(x - cx, y - cy) <= STATION_ZONE_M for cx, cy in self._centres)
        station = suspend_laser_updates(near, self._route_transit)
        if getattr(self, '_resume_pending', False) and not station:
            # One tick after the re-seed: AMCL has taken it, so the normal
            # thresholds cannot apply the suspended travel as one step.
            self._resume_pending = False
            self._set_update_thresholds(station=False)
        if station != self._station_mode and self._amcl_params.service_is_ready():
            previous, self._station_mode = self._station_mode, station
            action = zone_action(previous, station)
            self.get_logger().info(f'laser updates {"suspended" if station else "active"} '
                                   f'at ({x:.2f}, {y:.2f})')
            if action == 'reseed':
                # Leaving a station zone: AMCL's filter last moved where the
                # suspension began, and its next update would apply the whole
                # departure (about 2 m) as one motion step. With noise scaled
                # by that step the particle cloud spread to tens of cm and
                # degrees, and a poor scan match there jumped the pose 15-37
                # cm at the Station 2 south gate (measured 2026-11-12). Start
                # the filter afresh at the current pose, which has ridden on
                # odometry from the exact marker fix, with the fix's spread.
                self._send(t.header.stamp, x, y, _yaw(t.transform.rotation),
                           POSITION_STD_M, YAW_STD_RAD)
                self._resume_pending = True
                self.get_logger().info('AMCL re-seeded at the current pose before resuming')
            else:
                self._resume_pending = False
                self._set_update_thresholds(station=station)

    def _set_update_thresholds(self, station):
        """Normal thresholds as in amcl.yaml; huge ones in the station zone."""
        distance, angle = (100.0, 100.0) if station else (0.10, 0.08)
        request = SetParameters.Request()
        for name, value in (('update_min_d', distance), ('update_min_a', angle)):
            request.parameters.append(ParameterMsg(
                name=name, value=ParameterValue(type=ParameterType.PARAMETER_DOUBLE,
                                                double_value=value)))
        self._amcl_params.call_async(request)

    def _on_odom(self, msg):
        self._speed = math.hypot(msg.twist.twist.linear.x, msg.twist.twist.linear.y)

    def _on_frame(self, msg):
        now = self.get_clock().now().nanoseconds / 1e9
        if self._speed is None or self._speed > MAX_SPEED_M_S:
            return
        if self._owner != 'STOP':
            return
        if self._last_sent is not None and now - self._last_sent < MIN_INTERVAL_S:
            return
        self._pending = (msg, now)
        self._retry()

    def _retry(self):
        if self._pending is None:
            return
        msg, since = self._pending
        try:
            t = self._tf.lookup_transform('map', 'base_footprint',
                                          rclpy.time.Time.from_msg(msg.header.stamp))
        except Exception:
            if self.get_clock().now().nanoseconds / 1e9 - since > 0.2:
                self._pending = None  # TF never caught up: drop, never guess
            return
        self._pending = None
        now = self.get_clock().now().nanoseconds / 1e9
        robot = (t.transform.translation.x, t.transform.translation.y,
                 _yaw(t.transform.rotation))
        result = correction(robot, (msg.pose.position.x, msg.pose.position.y), self._markers)
        if result is None:
            return
        name, (x, y), size = result
        if size < MIN_CORRECTION_M:
            self._announce(name, 'marker', False)
            return
        pose = PoseWithCovarianceStamped()
        pose.header.stamp = msg.header.stamp
        pose.header.frame_id = 'map'
        pose.pose.pose.position.x, pose.pose.pose.position.y = x, y
        pose.pose.pose.orientation = t.transform.rotation
        covariance = [0.0] * 36
        covariance[0] = covariance[7] = POSITION_STD_M ** 2
        covariance[35] = YAW_STD_RAD ** 2
        pose.pose.covariance = covariance
        self._pub.publish(pose)
        self._last_sent = now
        self._announce(name, 'marker', True)
        if size > 0.02:
            self.get_logger().info(f'{name}: position corrected by {size * 100:.1f} cm')


def main(args=None):
    rclpy.init(args=args)
    node = MarkerLocalization()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
