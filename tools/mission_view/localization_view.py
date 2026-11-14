#!/usr/bin/env python3
"""RViz view of path tracking and localization; not part of the robot software.

Publishes, for tools/mission_view/tracking.rviz:
  /viz/true_trail       where the robot really drove (simulator truth)
  /viz/estimated_trail  where the robot thought it was (map -> base_footprint)
  /viz/corrections      one line per refinement update, from AMCL's position
                        towards the scan-refined one, drawn --scale times
                        longer so millimetres show: green = used, grey =
                        refused (AMCL's pose kept); a short orange tick
                        marks AMCL updates in station zones (marker rules)
The planned route itself is /mission/planned_path (from the operator panel).

Run next to a live simulation (same terminal setup as RViz):
    python3 tools/mission_view/localization_view.py [--scale 10] [--keep 30000]
Restart it to clear the trails.
"""
import argparse
import json
import math

import rclpy
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Path
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from std_msgs.msg import ColorRGBA, String
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

TRAIL_STEP_M = 0.02          # add a trail point every 2 cm of travel
TRAIL_TURN_RAD = math.radians(2.0)


def _yaw(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def _color(r, g, b, a=1.0):
    return ColorRGBA(r=r, g=g, b=b, a=a)


def correction_segment(status, scale):
    """(kind, start, end) in the map for one refinement status, or None.

    kind: 'used', 'refused' or 'zone'. The line runs from AMCL's position in
    the direction of the refined one, ``scale`` times its true length.
    """
    amcl = status.get('amcl')
    if not amcl:
        return None
    refined = status.get('refined')
    if status.get('reason') == 'station zone' or refined is None:
        return 'zone', amcl[:2], [amcl[0], amcl[1] + 0.03]
    end = [amcl[0] + scale * (refined[0] - amcl[0]), amcl[1] + scale * (refined[1] - amcl[1])]
    return ('used' if status.get('used') == 'refined' else 'refused'), amcl[:2], end


class Trail:
    """A nav_msgs/Path that grows by distance or turn, capped in length."""

    def __init__(self, keep):
        self.path = Path()
        self.path.header.frame_id = 'map'
        self.keep = keep
        self._last = None

    def add(self, stamp, x, y, yaw):
        if self._last is not None:
            lx, ly, lyaw = self._last
            turn = abs(math.atan2(math.sin(yaw - lyaw), math.cos(yaw - lyaw)))
            if math.hypot(x - lx, y - ly) < TRAIL_STEP_M and turn < TRAIL_TURN_RAD:
                return False
        pose = PoseStamped()
        pose.header.stamp, pose.header.frame_id = stamp, 'map'
        pose.pose.position.x, pose.pose.position.y = x, y
        pose.pose.orientation.z, pose.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        self.path.poses.append(pose)
        if len(self.path.poses) > self.keep:
            del self.path.poses[:len(self.path.poses) - self.keep]
        self.path.header.stamp = stamp
        self._last = (x, y, yaw)
        return True


class LocalizationView(Node):
    def __init__(self, scale, keep):
        super().__init__('localization_view')
        self.set_parameters([Parameter('use_sim_time', value=True)])
        self.scale = scale
        self._tf = Buffer()
        self._listener = TransformListener(self._tf, self)
        self._true, self._estimated = Trail(keep), Trail(keep)
        self._segments = {'used': [], 'refused': [], 'zone': []}
        self._keep_segments = keep
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._true_pub = self.create_publisher(Path, '/viz/true_trail', latched)
        self._est_pub = self.create_publisher(Path, '/viz/estimated_trail', latched)
        self._marker_pub = self.create_publisher(MarkerArray, '/viz/corrections', latched)
        self.create_subscription(PoseStamped, '/tugger/ground_truth_pose', self._on_truth,
                                 qos_profile_sensor_data)
        self.create_subscription(String, '/tugger/localization/refinement', self._on_status, 50)
        self.create_timer(0.1, self._sample_estimate)
        self.create_timer(0.5, self._publish)
        self.get_logger().info(f'localization view: trails and corrections (x{scale}) '
                               'for tools/mission_view/tracking.rviz')

    def _on_truth(self, msg):
        self._true.add(msg.header.stamp, msg.pose.position.x, msg.pose.position.y,
                       _yaw(msg.pose.orientation))

    def _sample_estimate(self):
        try:
            t = self._tf.lookup_transform('map', 'base_footprint', rclpy.time.Time())
        except Exception:
            return
        self._estimated.add(t.header.stamp, t.transform.translation.x,
                            t.transform.translation.y, _yaw(t.transform.rotation))

    def _on_status(self, msg):
        try:
            segment = correction_segment(json.loads(msg.data), self.scale)
        except ValueError:
            return
        if segment is None:
            return
        kind, start, end = segment
        lines = self._segments[kind]
        lines.append((start, end))
        if len(lines) > self._keep_segments:
            del lines[:len(lines) - self._keep_segments]

    def _publish(self):
        self._true_pub.publish(self._true.path)
        self._est_pub.publish(self._estimated.path)
        markers = MarkerArray()
        styles = {'used': (_color(0.05, 0.6, 0.25), 0.012),
                  'refused': (_color(0.55, 0.55, 0.55), 0.012),
                  'zone': (_color(0.85, 0.45, 0.05), 0.01)}
        for i, (kind, lines) in enumerate(self._segments.items()):
            m = Marker()
            m.header.frame_id = 'map'
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns, m.id, m.type, m.action = 'corrections_' + kind, i, Marker.LINE_LIST, Marker.ADD
            m.pose.orientation.w = 1.0
            m.color, m.scale.x = styles[kind]
            for start, end in lines:
                m.points.append(Point(x=float(start[0]), y=float(start[1]), z=0.02))
                m.points.append(Point(x=float(end[0]), y=float(end[1]), z=0.02))
            if not m.points:
                m.action = Marker.DELETE
            markers.markers.append(m)
        self._marker_pub.publish(markers)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--scale', type=float, default=10.0,
                        help='draw corrections this many times longer (default 10)')
    parser.add_argument('--keep', type=int, default=30000,
                        help='trail points and correction lines kept (default 30000)')
    args = parser.parse_args()
    rclpy.init()
    node = LocalizationView(args.scale, args.keep)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
