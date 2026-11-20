#!/usr/bin/env python3
"""Find objects the map does not contain, from the scanners and depth cameras.

Points far enough from anything in the static map are grouped into objects and
tracked, giving each one a size and whether it moves. Publishes JSON on
/tugger/obstacles. Used for planning only; the safety monitor is independent.
When obstacle_recognition names an object (person, pallet, cart, forklift from
the colour cameras), its class is added as 'cls' (absent when not named).
While obstacle_recognition runs, each object also carries 'looks': the frames
in which a camera had it in view (absent when recognition is not running).
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
from scipy.ndimage import binary_dilation, distance_transform_edt, label
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
# Points this close together are one object. The robot (0.66 m wide, 0.15 m
# kept each side) cannot pass through a gap under about 1 m, so joining gaps up
# to ~0.4 m changes no route; it keeps one object one object: the blocks of a
# Euro pallet (0.38 m apart), a person's legs, a cart's wheels. Split into
# pieces that merged and parted again between scans, a standing pallet's centre
# jumped 0.5 m and it was taken for a moving object (2026-11-20).
JOIN_CELLS = 2             # grown by 2 cells (0.2 m) each side: gaps up to 0.4 m join
MATCH_M = 0.6
# A scan piece this close to (or inside) a track's outline is part of that object.
OWN_MARGIN_M = 0.15
CONFIRM_HITS = 3
FORGET_S = 1.0
# Real travel, not the jitter of a partly seen object's centre (20-30 cm).
# The centre's shift less the change in the object's size: an object coming
# into view piece by piece (round a corner) grows while its centre shifts, by
# up to its size; it stands still. A walking person or a driving forklift keeps
# its size (2026-11-20: a standing pallet seen first by one corner was flagged
# as moving).
MOVING_DISTANCE_M = 0.6
SPEED_WINDOW_S = 2.0
CLASS_FRESH_S = 1.0       # a class older than this (recognition stopped) is dropped
# Each object's shape as seen: its convex outline, at most this many corners.
# An object's circle (half its longest extent) reaches past its sides: a
# 1.0 x 0.5 m cart beside the path looked 25 cm closer than it was.
OUTLINE_MAX_POINTS = 24
OUTLINE_MARGIN_M = 0.03


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


def outline(members):
    """The corners of a group of points' convex outline (all points if degenerate)."""
    from scipy.spatial import ConvexHull, QhullError
    corners = members
    if len(members) >= 3:
        try:
            corners = members[ConvexHull(members).vertices]
        except QhullError:      # all points on a line
            ends = np.argsort(members @ (members[-1] - members[0]))
            corners = members[[ends[0], ends[-1]]]
    if len(corners) > OUTLINE_MAX_POINTS:
        corners = corners[np.linspace(0, len(corners) - 1, OUTLINE_MAX_POINTS).astype(int)]
    return [(round(float(x), 3), round(float(y), 3)) for x, y in corners]


def cluster(points):
    """Group points into objects: (x, y, radius, outline) for each."""
    if len(points) == 0:
        return []
    low = points.min(axis=0)
    cells = ((points - low) / CELL_M).astype(int)
    grid = np.zeros(cells.max(axis=0) + 1, dtype=bool)
    grid[cells[:, 0], cells[:, 1]] = True
    if JOIN_CELLS:
        joined = binary_dilation(np.pad(grid, JOIN_CELLS), structure=np.ones((3, 3)),
                                 iterations=JOIN_CELLS)
        grid = joined[JOIN_CELLS:JOIN_CELLS + grid.shape[0], JOIN_CELLS:JOIN_CELLS + grid.shape[1]]
    labels, count = label(grid, structure=np.ones((3, 3)))
    ids = labels[cells[:, 0], cells[:, 1]]
    objects = []
    for k in range(1, count + 1):
        members = points[ids == k]
        if len(members) >= MIN_POINTS:
            centre = members.mean(axis=0)
            radius = float(np.max(np.hypot(*(members - centre).T))) + 0.03
            objects.append((float(centre[0]), float(centre[1]), radius, outline(members)))
    return objects


def near_outline(point, shape, margin):
    """Inside the polygon ``shape`` or within ``margin`` of its edges."""
    if not shape:
        return False
    px, py = point
    if len(shape) >= 3:
        inside = False
        for (ax, ay), (bx, by) in zip(shape, shape[1:] + shape[:1]):
            if (ay > py) != (by > py) and px < ax + (py - ay) * (bx - ax) / (by - ay):
                inside = not inside
        if inside:
            return True
    edges = list(zip(shape, shape[1:] + shape[:1])) if len(shape) > 1 else [(shape[0], shape[0])]
    for (ax, ay), (bx, by) in edges:
        dx, dy = bx - ax, by - ay
        span = dx * dx + dy * dy
        t = 0.0 if span == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / span))
        if math.hypot(px - ax - t * dx, py - ay - t * dy) <= margin:
            return True
    return False


def _area(shape):
    if not shape or len(shape) < 3:
        return 0.0
    return 0.5 * abs(sum(ax * by - bx * ay for (ax, ay), (bx, by) in zip(shape, shape[1:] + shape[:1])))


def merge_pieces(pieces):
    """One measurement (x, y, radius, outline) from the scan pieces of one object.

    A single piece is used as measured. Several pieces: the centre is their
    centres weighted by each piece's outline area (at least 1 dm^2), the
    outline the convex outline of all their corners, the radius reaches every
    corner and every piece's own circle."""
    pieces = [tuple(p) + (None,) * (4 - len(p)) for p in pieces]
    if len(pieces) == 1:
        x, y, r, shape = pieces[0][:4]
        return float(x), float(y), float(r), shape
    weights = np.array([max(_area(p[3]), 0.01) for p in pieces])
    centres = np.array([p[:2] for p in pieces], float)
    cx, cy = (centres * weights[:, None]).sum(axis=0) / weights.sum()
    corners = np.array([c for p in pieces for c in (p[3] or [p[:2]])], float)
    radius = max(math.hypot(p[0] - cx, p[1] - cy) + p[2] for p in pieces)
    radius = max(radius if not any(p[3] for p in pieces) else 0.0,
                 float(np.max(np.hypot(corners[:, 0] - cx, corners[:, 1] - cy))) + 0.03)
    return float(cx), float(cy), radius, outline(corners)


class Tracks:
    """Follow objects between frames to tell standing from moving.

    One track per real object (2026-11-20): every scan piece whose centre lies
    inside a track's last outline (or within OWN_MARGIN_M of it) belongs to that
    track, and the track's position, size and outline are measured from all its
    pieces together. A pallet whose blocks the scanners see as one group on one
    scan and as two on the next stays one track with a steady centre. A piece
    outside every outline is matched by distance (MATCH_M) as before, or starts
    a new track. Two tracks that come to overlap are merged into the older one.
    """

    def __init__(self):
        self.items, self.next_id = [], 0

    def _owner(self, piece):
        x, y = piece[0], piece[1]
        owners = [t for t in self.items if near_outline((x, y), t['outline'], OWN_MARGIN_M)]
        if owners:      # inside several outlines: the whole object (largest), then the oldest
            return max(owners, key=lambda t: (_area(t['outline']), -t['id']))
        owners = [t for t in self.items if math.hypot(t['x'] - x, t['y'] - y) <= MATCH_M]
        return min(owners, key=lambda t: math.hypot(t['x'] - x, t['y'] - y), default=None)

    def update(self, now, objects):
        owned, new = {}, []
        for piece in objects:
            track = self._owner(piece)
            if track is None:
                new.append(piece)
            else:
                owned.setdefault(track['id'], (track, []))[1].append(piece)
        for track, pieces in owned.values():
            self._measure(track, now, merge_pieces(pieces))
        for piece in new:
            x, y, r, shape = merge_pieces([piece])
            self.items.append({'id': self.next_id, 'x': x, 'y': y, 'r': r, 'outline': shape,
                               'hits': 1, 'seen': now, 'history': [(now, x, y, r)],
                               'moving': False, 'vx': 0.0, 'vy': 0.0})
            self.next_id += 1
        self._merge_overlapping(now)
        self.items = [t for t in self.items if now - t['seen'] <= FORGET_S]
        return [t for t in self.items if t['hits'] >= CONFIRM_HITS]

    @staticmethod
    def _measure(track, now, measurement):
        x, y, r, shape = measurement
        track.update(x=x, y=y, r=max(r, 0.9 * track['r']), outline=shape,
                     hits=track['hits'] + 1, seen=now)
        track['history'].append((now, x, y, r))
        old = [h for h in track['history'] if now - h[0] >= SPEED_WINDOW_S]
        if old:
            t0, x0, y0, r0 = old[-1]
            track['vx'], track['vy'] = (x - x0) / (now - t0), (y - y0) / (now - t0)
            track['moving'] = math.hypot(x - x0, y - y0) - abs(r - r0) >= MOVING_DISTANCE_M
            track['history'] = [h for h in track['history'] if h[0] >= t0]

    def _merge_overlapping(self, now):
        """A track whose centre lies inside the outline of a track measured this
        scan is part of that object: the two become one, keeping the older id
        (its camera votes and history) and this scan's measurement."""
        merged = True
        while merged:
            merged = False
            for whole in [t for t in self.items if t['seen'] == now]:
                part = next((t for t in self.items if t is not whole and
                             near_outline((t['x'], t['y']), whole['outline'], OWN_MARGIN_M)), None)
                if part is None:
                    continue
                keep, drop = (whole, part) if whole['id'] < part['id'] else (part, whole)
                if keep is not whole:
                    keep.update(x=whole['x'], y=whole['y'], r=whole['r'], outline=whole['outline'],
                                seen=now)
                    keep['history'].append((now, whole['x'], whole['y'], whole['r']))
                keep['hits'] = max(keep['hits'], drop['hits'])
                self.items.remove(drop)
                merged = True
                break


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
        self.classes, self.looks, self.pending, self.classes_at = {}, {}, set(), None
        self.create_subscription(String, '/tugger/obstacle_classes', self.on_classes, 10)
        self.create_subscription(LaserScan, '/scan', self.on_scan, qos_profile_sensor_data)
        for topic in ('/front_depth/points', '/rear_depth/points'):
            self.create_subscription(PointCloud2, topic,
                                     lambda m, t=topic: self.on_depth(t, m), qos_profile_sensor_data)
        self.create_timer(0.2, self.tick)

    def on_classes(self, msg):
        try:
            data = json.loads(msg.data)
            self.classes, self.looks = data['classes'], data.get('looks', {})
            self.pending = set(data.get('pending', []))
            self.classes_at = self.get_clock().now().nanoseconds / 1e9
        except (ValueError, KeyError, TypeError):
            pass

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
        fresh = self.classes_at is not None and now - self.classes_at <= CLASS_FRESH_S
        named = self.classes if fresh else {}
        out = []
        for o in confirmed:
            item = {'id': o['id'], 'x': round(o['x'], 3), 'y': round(o['y'], 3), 'r': round(o['r'], 3),
                    'outline': o['outline'], 'outline_margin': OUTLINE_MARGIN_M,
                    'moving': o['moving'], 'vx': round(o['vx'], 2), 'vy': round(o['vy'], 2)}
            if fresh:
                item['looks'] = int(self.looks.get(str(o['id']), 0))
                if str(o['id']) in self.pending:
                    item['pending'] = True      # being recognised: not judged yet
            if str(o['id']) in named:
                item['cls'] = named[str(o['id'])]['cls']
                item['cls_conf'] = named[str(o['id'])]['conf']
            out.append(item)
        message = String()
        message.data = json.dumps({'stamp': now, 'obstacles': out})
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
