#!/usr/bin/env python3
"""Find objects the map does not contain, from the scanners and depth cameras.

Points far enough from anything in the static map are grouped into objects and
tracked, giving each one a size and whether it moves. Publishes JSON on
/tugger/obstacles. Used for planning only; the safety monitor is independent.
"""
import json
import math
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from scipy.ndimage import distance_transform_edt, label
from sensor_msgs.msg import LaserScan, PointCloud2
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener
import yaml

MAP_TOLERANCE_M = 0.25    # closer than this to the map: part of the map (localization error)
RANGE_M = 9.0
MIN_Z, MAX_Z = 0.05, 0.22  # as the safety monitor: floor noise and carried cargo excluded
BODY_HALF = (0.52, 0.36)
CELL_M = 0.10
MIN_POINTS = 2
MATCH_M = 0.6
CONFIRM_HITS = 3
FORGET_S = 1.0
# Real travel, not the jitter of a partly seen object's centre (20-30 cm).
MOVING_DISTANCE_M = 0.6
SPEED_WINDOW_S = 2.0


def load_distance_map(yaml_path):
    """Distance (m) from every cell to the nearest occupied cell, and the map frame."""
    spec = yaml.safe_load(Path(yaml_path).read_text())
    raw = (Path(yaml_path).parent / spec['image']).read_bytes()
    fields, i = [], 0
    while len(fields) < 4:
        while raw[i:i + 1].isspace():
            i += 1
        if raw[i:i + 1] == b'#':
            while raw[i:i + 1] != b'\n':
                i += 1
            continue
        j = i
        while not raw[j:j + 1].isspace():
            j += 1
        fields.append(raw[i:j])
        i = j
    width, height = int(fields[1]), int(fields[2])
    image = np.frombuffer(raw, np.uint8, width * height, i + 1).reshape(height, width)
    resolution = float(spec['resolution'])
    return distance_transform_edt(image >= 128) * resolution, resolution, spec['origin'][:2], height


def off_map(points, distance, resolution, origin, height):
    """Mask of points farther than the tolerance from anything in the map."""
    col = ((points[:, 0] - origin[0]) / resolution).astype(int)
    row = height - 1 - ((points[:, 1] - origin[1]) / resolution).astype(int)
    inside = (col >= 0) & (col < distance.shape[1]) & (row >= 0) & (row < distance.shape[0])
    keep = np.zeros(len(points), dtype=bool)
    keep[inside] = distance[row[inside], col[inside]] > MAP_TOLERANCE_M
    return keep


def cluster(points):
    """Group points into objects: (x, y, radius) for each."""
    if len(points) == 0:
        return []
    low = points.min(axis=0)
    cells = ((points - low) / CELL_M).astype(int)
    grid = np.zeros(cells.max(axis=0) + 1, dtype=bool)
    grid[cells[:, 0], cells[:, 1]] = True
    labels, count = label(grid, structure=np.ones((3, 3)))
    ids = labels[cells[:, 0], cells[:, 1]]
    objects = []
    for k in range(1, count + 1):
        members = points[ids == k]
        if len(members) >= MIN_POINTS:
            centre = members.mean(axis=0)
            radius = float(np.max(np.hypot(*(members - centre).T))) + 0.03
            objects.append((float(centre[0]), float(centre[1]), radius))
    return objects


class Tracks:
    """Follow objects between frames to tell standing from moving."""

    def __init__(self):
        self.items, self.next_id = [], 0

    def update(self, now, objects):
        for x, y, r in objects:
            match = min(self.items, key=lambda t: math.hypot(t['x'] - x, t['y'] - y), default=None)
            if match is None or math.hypot(match['x'] - x, match['y'] - y) > MATCH_M:
                self.items.append({'id': self.next_id, 'x': x, 'y': y, 'r': r, 'hits': 1, 'seen': now,
                                   'history': [(now, x, y)], 'moving': False, 'vx': 0.0, 'vy': 0.0})
                self.next_id += 1
                continue
            match.update(x=x, y=y, r=max(r, 0.9 * match['r']), hits=match['hits'] + 1, seen=now)
            match['history'].append((now, x, y))
            old = [h for h in match['history'] if now - h[0] >= SPEED_WINDOW_S]
            if old:
                t0, x0, y0 = old[-1]
                match['vx'], match['vy'] = (x - x0) / (now - t0), (y - y0) / (now - t0)
                match['moving'] = math.hypot(x - x0, y - y0) >= MOVING_DISTANCE_M
                match['history'] = [h for h in match['history'] if h[0] >= t0]
        self.items = [t for t in self.items if now - t['seen'] <= FORGET_S]
        return [t for t in self.items if t['hits'] >= CONFIRM_HITS]


class ObstacleMonitor(Node):
    def __init__(self):
        super().__init__('tugger_obstacle_monitor')
        self.set_parameters([Parameter('use_sim_time', value=True)])
        share = Path(get_package_share_directory('tugger_navigation'))
        self.map = load_distance_map(share / 'maps' / 'localization_map.yaml')
        self.tf = Buffer()
        self.listener = TransformListener(self.tf, self)
        self.tracks = Tracks()
        self.depth = {}
        self.mounts = {}
        self.scan = None
        self.pub = self.create_publisher(String, '/tugger/obstacles', 10)
        self.create_subscription(LaserScan, '/scan', self.on_scan, qos_profile_sensor_data)
        for topic in ('/front_depth/points', '/rear_depth/points'):
            self.create_subscription(PointCloud2, topic,
                                     lambda m, t=topic: self.on_depth(t, m), qos_profile_sensor_data)
        self.create_timer(0.2, self.tick)

    def on_scan(self, msg):
        ranges = np.array(msg.ranges)
        angles = msg.angle_min + np.arange(len(ranges)) * msg.angle_increment
        good = np.isfinite(ranges) & (ranges >= msg.range_min) & (ranges <= RANGE_M)
        self.scan = np.column_stack((ranges[good] * np.cos(angles[good]),
                                     ranges[good] * np.sin(angles[good])))

    def mount(self, frame):
        if frame not in self.mounts:
            t = self.tf.lookup_transform('base_footprint', frame, rclpy.time.Time())
            q = t.transform.rotation
            self.mounts[frame] = (t.transform.translation.x, t.transform.translation.y,
                                  math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)),
                                  t.transform.translation.z)
        return self.mounts[frame]

    def on_depth(self, topic, msg):
        from sensor_msgs_py import point_cloud2
        try:
            mx, my, myaw, mz = self.mount(msg.header.frame_id)
        except Exception:
            return
        raw = point_cloud2.read_points_numpy(msg, field_names=('x', 'y', 'z'), skip_nans=True)
        raw = raw[np.all(np.isfinite(raw), axis=1)] if raw.size else np.zeros((0, 3))
        z = raw[:, 2] + mz
        raw = raw[(z >= MIN_Z) & (z <= MAX_Z) & (np.hypot(raw[:, 0], raw[:, 1]) <= RANGE_M)]
        c, s = math.cos(myaw), math.sin(myaw)
        self.depth[topic] = np.column_stack((mx + c * raw[:, 0] - s * raw[:, 1],
                                             my + s * raw[:, 0] + c * raw[:, 1]))

    def tick(self):
        now = self.get_clock().now().nanoseconds / 1e9
        try:
            t = self.tf.lookup_transform('map', 'base_footprint', rclpy.time.Time())
        except Exception:
            return
        q = t.transform.rotation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        clouds = [c for c in [self.scan, *self.depth.values()] if c is not None and len(c)]
        points = np.vstack(clouds) if clouds else np.zeros((0, 2))
        points = points[~((np.abs(points[:, 0]) <= BODY_HALF[0]) & (np.abs(points[:, 1]) <= BODY_HALF[1]))]
        c, s = math.cos(yaw), math.sin(yaw)
        world = np.column_stack((t.transform.translation.x + c * points[:, 0] - s * points[:, 1],
                                 t.transform.translation.y + s * points[:, 0] + c * points[:, 1]))
        found = cluster(world[off_map(world, *self.map)]) if len(world) else []
        confirmed = self.tracks.update(now, found)
        message = String()
        message.data = json.dumps({'stamp': now, 'obstacles': [
            {'id': o['id'], 'x': round(o['x'], 3), 'y': round(o['y'], 3), 'r': round(o['r'], 3),
             'moving': o['moving'], 'vx': round(o['vx'], 2), 'vy': round(o['vy'], 2)}
            for o in confirmed]})
        self.pub.publish(message)


def main(args=None):
    rclpy.init(args=args)
    node = ObstacleMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
