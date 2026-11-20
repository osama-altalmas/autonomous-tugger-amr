#!/usr/bin/env python3
"""What the tracked obstacles are: person, pallet, cart or forklift (AI, AMR-V3).

The obstacle monitor finds objects with the scanners and depth cameras and
tracks whether they move; it cannot tell a person standing still from a box.
This node runs the obstacle model (YOLO11n, dataset v5: tools/ai/obstacle_dataset,
ai_data/AI_REVIEW.md) on the front and rear colour cameras and gives each
tracked object the class of the detection at its bearing and distance:

- the object's centre and size, seen from the camera, give a span of image
  columns and the row where it stands on the floor;
- a detection matches when its columns overlap that span and its bottom edge
  is near that floor row (or cut by the image edge);
- a class is kept only when it was seen at least MIN_VOTES times in the last
  VOTE_WINDOW_S, and is the most confident class there (one wrong frame does
  not relabel an object).

Publishes JSON on /tugger/obstacle_classes: {'stamp', 'classes': {id: {'cls',
'conf', 'votes'}}, 'looks': {id: frames}, 'pending': [id]}, where 'looks' counts the frames in
which a camera had the object in view (named or not) since it was first
tracked: the obstacle policy does not judge an unnamed object the cameras have
not looked at yet (look before deciding, user decision 2026-11-20), and 'pending'
lists objects with a vote but no class yet (being recognised: not judged yet). The obstacle monitor adds the class to /tugger/obstacles,
and the obstacle policy treats a person or forklift that stands still like a
moving object (user decision 2026-11-08: people and forklifts are waited for).
The safety fields do not depend on this.

The model runs with ONNX Runtime when it is installed (55 ms per image on the
development laptop) and with OpenCV's DNN module otherwise (140 ms).

Every STATUS_PERIOD_S the node logs what it did (frames looked at, frames with
detections, objects in view, objects matched, TF failures). With the
environment variable TUGGER_RECOGNITION_DEBUG=<dir> it also writes, per frame
looked at, the detections and each object's expected view to <dir>/looks.jsonl
and, while an object is in view, the frame with both drawn (at most one image
per DEBUG_IMAGE_PERIOD_S).
"""
import json
import math
import os
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np

CLASSES = ('person', 'pallet', 'cart', 'forklift')
INPUT_SIZE = 640
CONF_MIN = 0.35            # detections below this are not used
NMS_IOU = 0.7             # as Ultralytics predicts and validates (its default)
HFOV_RAD = 1.5184          # the colour cameras (tugger.urdf.xacro, recognition_camera)
MIN_RANGE_M, MAX_RANGE_M = 0.3, 10.0
COLUMN_MARGIN_PX = 12      # an object's span widened by this before matching
FLOOR_ROW_TOLERANCE = 0.35  # bottom edge within this share of the box height of the floor row
MIN_OVERLAP = 0.3          # shared columns / the narrower of the two spans
VOTE_WINDOW_S = 3.0
MIN_VOTES = 2
MODEL_FILE = 'obstacles_v5n.onnx'
STATUS_PERIOD_S = 10.0
DEBUG_IMAGE_PERIOD_S = 1.0
DEBUG_ENV = 'TUGGER_RECOGNITION_DEBUG'


MOVING_SPEED_M_S = 0.02     # slower than this the robot counts as standing


def choose_camera(in_view, direction, turn):
    """Which camera to run the model on this time (user rule 2026-11-16: the
    front camera when driving forward, the rear when reversing).

    ``in_view`` maps 'front'/'rear' to whether a tracked object is in its view,
    ``direction`` is +1 forward, -1 reversing, 0 standing, ``turn`` a counter.
    A camera with an object in view is preferred; with objects in both views,
    or none, the camera facing the driving direction; standing, they take
    turns. Then the camera that matters looks at every tick, not every other.
    """
    ahead = 'front' if direction > 0 else 'rear' if direction < 0 else None
    seeing = [name for name in ('front', 'rear') if in_view.get(name)]
    if len(seeing) == 1:
        return seeing[0]
    if ahead is not None:
        return ahead
    return ('front', 'rear')[turn % 2]


def draw_debug(image, detections, views):
    """The frame with detections (green, class and confidence) and expected
    object views (yellow column span, red floor row; cyan when matched)."""
    out = image.copy()
    for x0, y0, x1, y1, c, p in detections:
        cv2.rectangle(out, (int(x0), int(y0)), (int(x1), int(y1)), (0, 200, 0), 2)
        cv2.putText(out, f'{CLASSES[c]} {p:.2f}', (int(x0), max(12, int(y0) - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 0), 1)
    for oid, (u0, u1, floor_v), matched in views:
        colour = (255, 255, 0) if matched else (0, 220, 255)
        cv2.rectangle(out, (int(u0), 0), (int(u1), out.shape[0] - 1), colour, 1)
        cv2.line(out, (int(u0) - 15, int(floor_v)), (int(u1) + 15, int(floor_v)), (0, 0, 255), 2)
        cv2.putText(out, f'id {oid}', (int(u0), 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1)
    return out


def letterbox(image, size=INPUT_SIZE):
    """Resize keeping the aspect ratio and pad to size x size (grey 114)."""
    h, w = image.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    left, top = (size - nw) // 2, (size - nh) // 2
    padded = np.full((size, size, 3), 114, np.uint8)
    padded[top:top + nh, left:left + nw] = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return padded, scale, (left, top)


def decode(output, scale, pad, conf_min=CONF_MIN, nms_iou=NMS_IOU):
    """YOLO detect output (1, 4 + classes, anchors) to boxes in the original image:
    [(x0, y0, x1, y1, class, confidence)], after non-maximum suppression per class."""
    pred = np.asarray(output)[0].T
    scores = pred[:, 4:]
    cls = scores.argmax(axis=1)
    conf = scores[np.arange(len(pred)), cls]
    keep = conf >= conf_min
    pred, cls, conf = pred[keep], cls[keep], conf[keep]
    if not len(pred):
        return []
    cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    x0 = (cx - w / 2 - pad[0]) / scale
    y0 = (cy - h / 2 - pad[1]) / scale
    boxes = np.column_stack((x0, y0, w / scale, h / scale))
    out = []
    for c in np.unique(cls):
        idx = np.flatnonzero(cls == c)
        chosen = cv2.dnn.NMSBoxes(boxes[idx].tolist(), conf[idx].tolist(), conf_min, nms_iou)
        for i in np.array(chosen).flatten():
            x, y, bw, bh = boxes[idx[i]]
            out.append((float(x), float(y), float(x + bw), float(y + bh), int(c), float(conf[idx[i]])))
    return out


def focal_px(width, hfov=HFOV_RAD):
    return (width / 2) / math.tan(hfov / 2)


def expected_view(obstacle_cam, radius, width, height, camera_height, hfov=HFOV_RAD):
    """Where an object should appear: (column from, column to, floor row), or None
    when it is behind the camera, too close, too far or outside the view.

    ``obstacle_cam`` is the object's centre in the camera frame (x forward, y left);
    ``camera_height`` the camera above the floor.
    """
    x, y = obstacle_cam
    distance = math.hypot(x, y)
    if x <= MIN_RANGE_M or not MIN_RANGE_M <= distance <= MAX_RANGE_M:
        return None
    f = focal_px(width, hfov)
    centre_u = width / 2 - f * (y / x)
    half = f * radius / x
    u0, u1 = centre_u - half, centre_u + half
    if u1 < 0 or u0 > width:
        return None
    floor_v = height / 2 + f * camera_height / max(x - radius, MIN_RANGE_M)
    return u0, u1, floor_v


def match(view, detections, height):
    """The detection best matching an expected view, or None."""
    u0, u1, floor_v = view
    u0, u1 = u0 - COLUMN_MARGIN_PX, u1 + COLUMN_MARGIN_PX
    best, best_overlap = None, MIN_OVERLAP
    for d in detections:
        x0, y0, x1, y1 = d[:4]
        shared = min(u1, x1) - max(u0, x0)
        if shared <= 0:
            continue
        overlap = shared / max(1.0, min(u1 - u0, x1 - x0))
        cut = y1 >= height - 2
        if not cut and abs(y1 - floor_v) > FLOOR_ROW_TOLERANCE * max(y1 - y0, 20.0):
            continue
        if overlap > best_overlap:
            best, best_overlap = d, overlap
    return best


class Votes:
    """Recent detections per obstacle id; the class is the most confident one seen often enough."""

    def __init__(self, window=VOTE_WINDOW_S, min_votes=MIN_VOTES):
        self.window, self.min_votes = window, min_votes
        self.seen = defaultdict(deque)

    def add(self, now, obstacle_id, cls, conf):
        self.seen[obstacle_id].append((now, cls, conf))

    def classes(self, now, alive_ids=None):
        out = {}
        for oid in list(self.seen):
            q = self.seen[oid]
            while q and now - q[0][0] > self.window:
                q.popleft()
            if not q or (alive_ids is not None and oid not in alive_ids):
                del self.seen[oid]
                continue
            total, count = defaultdict(float), defaultdict(int)
            for _, c, p in q:
                total[c] += p
                count[c] += 1
            cls = max(total, key=total.get)
            if count[cls] >= self.min_votes:
                out[oid] = {'cls': CLASSES[cls], 'conf': round(total[cls] / count[cls], 3),
                            'votes': count[cls]}
        return out

    def pending(self, now, named):
        """Objects seen as something in the window but not named yet: a
        recognition in progress (one vote of two), not to be judged yet."""
        return sorted(oid for oid, q in self.seen.items()
                      if oid not in named and any(now - t <= self.window for t, _, _ in q))


class ObstacleModel:
    """The exported ONNX model, with ONNX Runtime if installed, else OpenCV DNN."""

    def __init__(self, path):
        try:
            import onnxruntime as ort
            self.session = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
            self.input = self.session.get_inputs()[0].name
            self.backend = 'onnxruntime'
        except ImportError:
            self.session = None
            self.net = cv2.dnn.readNetFromONNX(str(path))
            self.backend = 'opencv'

    def detect(self, image_bgr):
        padded, scale, pad = letterbox(image_bgr)
        blob = cv2.dnn.blobFromImage(padded, 1 / 255.0, (INPUT_SIZE, INPUT_SIZE), swapRB=True)
        if self.session is not None:
            output = self.session.run(None, {self.input: blob})[0]
        else:
            self.net.setInput(blob)
            output = self.net.forward()
        return decode(output, scale, pad)


def default_model_path():
    from ament_index_python.packages import get_package_share_directory
    return str(Path(get_package_share_directory('tugger_control')) / 'models' / MODEL_FILE)


def main(args=None):
    import rclpy
    from rclpy.node import Node
    import rclpy.time
    from rclpy.parameter import Parameter
    from rclpy.qos import qos_profile_sensor_data
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import Image
    from std_msgs.msg import String
    from tf2_ros import Buffer, TransformListener

    class ObstacleRecognition(Node):
        CAMERAS = (('front', '/front_rgb/image_raw', 'front_camera_link'),
                   ('rear', '/rear_rgb/image_raw', 'rear_camera_link'))

        def __init__(self):
            super().__init__('tugger_obstacle_recognition')
            self.set_parameters([Parameter('use_sim_time', value=True)])
            self.declare_parameter('model', default_model_path())
            self.declare_parameter('rate_hz', 2.5)     # images per second (see choose_camera)
            self.model = ObstacleModel(self.get_parameter('model').value)
            self.get_logger().info(f'obstacle model loaded ({self.model.backend})')
            self.tf = Buffer()
            self.listener = TransformListener(self.tf, self)
            self.images, self.obstacles, self.turn = {}, [], 0
            self.votes = Votes()
            self.looks = {}
            self.pub = self.create_publisher(String, '/tugger/obstacle_classes', 10)
            for name, topic, _ in self.CAMERAS:
                self.create_subscription(Image, topic, lambda m, n=name: self.images.__setitem__(n, m),
                                         qos_profile_sensor_data)
            self.create_subscription(String, '/tugger/obstacles', self.on_obstacles, 10)
            self.direction = 0
            self.create_subscription(Odometry, '/wheel/odom', self.on_odom, 10)
            self.create_timer(1.0 / float(self.get_parameter('rate_hz').value), self.tick)
            self.stats = {'looks': 0, 'with_detections': 0, 'in_view': 0, 'matched': 0, 'tf_fail': 0}
            self.last_tf_error = ''
            self.create_timer(STATUS_PERIOD_S, self.status)
            self.debug_dir = os.environ.get(DEBUG_ENV, '')
            self.last_debug_image = -1e9
            if self.debug_dir:
                os.makedirs(self.debug_dir, exist_ok=True)
                self.get_logger().info(f'debug output in {self.debug_dir}')

        def status(self):
            s = self.stats
            text = (f'last {STATUS_PERIOD_S:.0f} s: {s["looks"]} frames looked at, '
                    f'{s["with_detections"]} with detections, {s["in_view"]} objects in view, '
                    f'{s["matched"]} matched, {s["tf_fail"]} TF failures; '
                    f'{len(self.obstacles)} tracked, cameras {sorted(self.images)}')
            if s['tf_fail']:
                self.get_logger().warning(f'{text} (last TF error: {self.last_tf_error})')
            elif s['looks'] or self.obstacles:
                self.get_logger().info(text)
            self.stats = dict.fromkeys(s, 0)

        def on_obstacles(self, msg):
            try:
                self.obstacles = json.loads(msg.data)['obstacles']
            except (ValueError, KeyError, TypeError):
                pass

        def on_odom(self, msg):
            v = msg.twist.twist.linear.x
            self.direction = 1 if v > MOVING_SPEED_M_S else -1 if v < -MOVING_SPEED_M_S else 0

        def in_view(self, frame):
            """Whether any tracked object is in this camera's view now."""
            try:
                t = self.tf.lookup_transform(frame, 'map', rclpy.time.Time())
            except Exception:       # counted when the frame itself is looked at
                return False
            q, p = t.transform.rotation, t.transform.translation
            yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
            c, s = math.cos(yaw), math.sin(yaw)
            return any(expected_view((p.x + c * o['x'] - s * o['y'], p.y + s * o['x'] + c * o['y']),
                                     o.get('r', 0.3), 640, 480, -p.z) is not None
                       for o in self.obstacles)

        def tick(self):
            now = self.get_clock().now().nanoseconds / 1e9
            views = {name: self.in_view(frame) for name, _, frame in self.CAMERAS}
            name = choose_camera(views, self.direction, self.turn)
            frame = next(f for n, _, f in self.CAMERAS if n == name)
            self.turn += 1
            msg = self.images.get(name)
            if msg is not None and self.obstacles:
                self.look(now, msg, frame)
            alive = {o['id'] for o in self.obstacles}
            classes = self.votes.classes(now, alive)
            self.looks = {k: v for k, v in self.looks.items() if k in alive}
            out = String()
            out.data = json.dumps({'stamp': now, 'classes': {str(k): v for k, v in classes.items()},
                                   'looks': {str(k): v for k, v in self.looks.items()},
                                   'pending': [str(k) for k in self.votes.pending(now, classes)]})
            self.pub.publish(out)

        def look(self, now, msg, frame):
            try:
                # The camera's pose when the image was taken, not the latest:
                # while the robot turns they differ by degrees (15+ px), and a
                # person in view was not matched (campaign 3, 2026-11-20).
                # No waiting (one thread): a frame whose pose is not known yet
                # is skipped and counted.
                t = self.tf.lookup_transform(frame, 'map', rclpy.time.Time.from_msg(msg.header.stamp))
            except Exception as e:      # tf2 raises several kinds; counted and logged in status()
                self.stats['tf_fail'] += 1
                self.last_tf_error = f'{type(e).__name__}: {e}'
                return
            self.stats['looks'] += 1
            q, p = t.transform.rotation, t.transform.translation
            yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
            # camera height above the floor: the map origin's height in the camera frame, negated
            camera_height = -p.z
            image = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, -1)[:, :, :3]
            if msg.encoding.lower().startswith('rgb'):
                image = image[:, :, ::-1]
            detections = self.model.detect(np.ascontiguousarray(image))
            self.stats['with_detections'] += bool(detections)
            c, s = math.cos(yaw), math.sin(yaw)
            views, records = [], []
            for o in self.obstacles:
                cam = (p.x + c * o['x'] - s * o['y'], p.y + s * o['x'] + c * o['y'])
                view = expected_view(cam, o.get('r', 0.3), msg.width, msg.height, camera_height)
                records.append({'id': o['id'], 'map': [o['x'], o['y']], 'r': o.get('r'),
                                'cam': [round(cam[0], 3), round(cam[1], 3)],
                                'view': None if view is None else [round(v, 1) for v in view]})
                if view is None:
                    continue
                self.stats['in_view'] += 1
                self.looks[o['id']] = self.looks.get(o['id'], 0) + 1
                d = match(view, detections, msg.height) if detections else None
                views.append((o['id'], view, d is not None))
                if d is not None:
                    self.stats['matched'] += 1
                    self.votes.add(now, o['id'], d[4], d[5])
                    records[-1]['match'] = [CLASSES[d[4]], round(d[5], 3)]
            if self.debug_dir:
                self.write_debug(now, frame, camera_height, image, detections, views, records)

        def write_debug(self, now, frame, camera_height, image, detections, views, records):
            line = {'stamp': round(now, 3), 'frame': frame, 'camera_height': round(camera_height, 3),
                    'detections': [[round(v, 1) for v in d[:4]] + [CLASSES[d[4]], round(d[5], 3)]
                                   for d in detections],
                    'obstacles': records}
            if views and now - self.last_debug_image >= DEBUG_IMAGE_PERIOD_S:
                self.last_debug_image = now
                name = f'{frame.split("_")[0]}_{now:010.2f}.jpg'
                cv2.imwrite(os.path.join(self.debug_dir, name), draw_debug(image, detections, views))
                # the clean frame too: the drawn lines could disturb a re-run of the model
                cv2.imwrite(os.path.join(self.debug_dir, name.replace('.jpg', '_clean.png')), image)
                line['image'] = name
            try:
                with open(os.path.join(self.debug_dir, 'looks.jsonl'), 'a') as f:
                    f.write(json.dumps(line) + '\n')
            except OSError:
                pass

    rclpy.init(args=args)
    node = ObstacleRecognition()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
