#!/usr/bin/env python3
"""Dataset v2: floor markers (four AprilTags in a red frame) for YOLO-pose.

Runs against a headless Gazebo (isolated partition/domain, simulation.launch.py
only). For each sample the robot is placed over a station's marker at a random
docking-like offset and heading; the downward camera is captured twice at the
same pose:

1. clean: the four tags are detected and the plate is mapped onto the image
   (homography from all 16 tag corners). The 20 keypoints are projected from
   the plate: 4 red-frame corners, then the 4 corners of each tag (quarters
   top-left, top-right, bottom-right, bottom-left; corners TL, TR, BR, BL);
2. damaged: patches (tape, dirt, worn strips, tyre marks, paint, a whole frame
   side wiped) are laid on the marker, the light changes, floor mats vary,
   and the image is captured again with the same keypoints. A keypoint under
   a patch keeps its position but is marked covered (v=1), so the model
   learns to place hidden corners from what it still sees.

Class: the station while at least one tag is readable (its own coverage at
most TAG_READABLE), 'unknown_marker' (0) when no tag is but the frame still
shows (the robot can align but not confirm the station), no object when
neither shows; samples in between are left out. The box is the full frame.

Held-out test set (--test): damage colours and shapes, mats and light the
training never sees, to measure recognition rather than memorisation.

Usage (isolated simulation running):
  python3 tools/ai/marker_dataset/generate_pose.py --out ai_data/markers_v2 --per-marker 800
  python3 tools/ai/marker_dataset/generate_pose.py --out ai_data/markers_v2_test --test --seed 97
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src/tugger_control'))
from tugger_control.docking_markers import (  # noqa: E402
    RED_FRAME_SIZE_M, TAG_SIZE_M, detect_floor_tags, tag_detector, tag_object_corners)

WORLD = 'tugger_test'
ROBOT = 'autonomous_tugger'
MARKERS = {
    'HOME': ((0.0, 0.0), (0.0, math.pi), 1, 'TUGGER:HOME'),
    'S1': ((-17.5, -7.0), (math.pi / 2, -math.pi / 2), 2, 'TUGGER:S1'),
    'D2': ((-17.5, 7.0), (math.pi / 2, -math.pi / 2), 3, 'TUGGER:D2'),
    'S3': ((17.5, 7.0), (math.pi / 2, -math.pi / 2), 4, 'TUGGER:S3'),
    'D4': ((17.5, -7.0), (math.pi / 2, -math.pi / 2), 5, 'TUGGER:D4'),
}
CLASS_NAMES = ['unknown_marker', 'HOME', 'S1', 'D2', 'S3', 'D4']
FRAME_HALF = RED_FRAME_SIZE_M / 2
PLATE_KEYPOINTS = np.vstack([
    [[-FRAME_HALF, FRAME_HALF], [FRAME_HALF, FRAME_HALF],
     [FRAME_HALF, -FRAME_HALF], [-FRAME_HALF, -FRAME_HALF]],
    *[tag_object_corners(q)[:, :2] for q in range(4)]])          # 20 x 2
KEYPOINT_NAMES = (['frame_tl', 'frame_tr', 'frame_br', 'frame_bl'] +
                  [f'tag{q}_{c}' for q in ('tl', 'tr', 'br', 'bl')
                   for c in ('tl', 'tr', 'br', 'bl')])
TAG_READABLE = 0.15            # a tag this little covered still reads (36h11 corrects bits)
TAG_UNREADABLE = 0.35
FRAME_VISIBLE = 0.60           # share of the red ring hidden: up to this the frame counts
FRAME_GONE = 0.80
OFFSET_M = 0.06
PATCH_Z = 0.0055
PARK = (0.0, 0.0, -5.0)
COLOURS = {
    'tape': [(0.78, 0.74, 0.58), (0.85, 0.85, 0.85), (0.2, 0.35, 0.75)],
    'dirt': [(0.35, 0.27, 0.18), (0.28, 0.24, 0.20), (0.45, 0.38, 0.28)],
    'wear': [(0.55, 0.55, 0.52), (0.62, 0.62, 0.60)],
    'tyre': [(0.08, 0.08, 0.08), (0.15, 0.15, 0.15)],
    'paint': [(0.92, 0.92, 0.90), (0.95, 0.85, 0.10)],
}
COLOURS_TEST = {
    'tape': [(0.70, 0.20, 0.15), (0.95, 0.95, 0.70)],
    'dirt': [(0.20, 0.18, 0.22), (0.55, 0.50, 0.42)],
    'wear': [(0.48, 0.48, 0.46)],
    'tyre': [(0.03, 0.03, 0.05)],
    'paint': [(0.10, 0.60, 0.20), (0.95, 0.50, 0.10)],
}
MAT_COLOURS = [(0.55, 0.55, 0.52), (0.72, 0.72, 0.70), (0.0, 0.35, 0.0),
               (0.15, 0.30, 0.60), (0.85, 0.75, 0.15), (0.60, 0.52, 0.40)]
MAT_COLOURS_TEST = [(0.40, 0.18, 0.12), (0.30, 0.32, 0.36), (0.90, 0.90, 0.88)]
MAT_Z = 0.0034
# Damage levels: share of samples and how much of the plate is aimed at.
LEVELS = [('clean', 0.15, (0.0, 0.0)), ('light', 0.30, (0.03, 0.20)),
          ('medium', 0.30, (0.20, 0.45)), ('heavy', 0.25, (0.45, 0.95))]
SIDE_WIPE_SHARE = 0.20         # of damaged samples: a whole frame side wiped out


class Camera(Node):
    def __init__(self):
        super().__init__('marker_dataset_camera')
        self.bridge = CvBridge()
        self.count, self.image = 0, None
        self.create_subscription(Image, '/downward_camera/image_raw', self._on_image,
                                 qos_profile_sensor_data)

    def _on_image(self, msg):
        self.image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.count += 1

    def fresh(self, frames=3, timeout=10.0):
        start, end = self.count, time.monotonic() + timeout
        while self.count < start + frames and time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.05)
        return None if self.count < start + frames else self.image.copy()


class Gazebo:
    def __init__(self):
        from gz.transport import Node as GzNode
        self.node = GzNode()

    def _call(self, service, request, reqtype, timeout_ms=3000):
        from gz.msgs.boolean_pb2 import Boolean
        ok, _ = self.node.request(f'/world/{WORLD}/{service}', request, reqtype, Boolean,
                                  timeout_ms)
        return ok

    def set_pose(self, name, x, y, z, yaw=0.0):
        from gz.msgs.pose_pb2 import Pose
        p = Pose()
        p.name = name
        p.position.x, p.position.y, p.position.z = x, y, z
        p.orientation.z, p.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        return self._call('set_pose', p, Pose)

    def create(self, sdf):
        from gz.msgs.entity_factory_pb2 import EntityFactory
        f = EntityFactory()
        f.sdf = sdf
        return self._call('create', f, EntityFactory, 10000)

    def light(self, intensity, direction):
        from gz.msgs.light_pb2 import Light
        light = Light()
        light.name = 'sun'
        light.type = Light.DIRECTIONAL
        light.diffuse.r = light.diffuse.g = light.diffuse.b = intensity
        light.diffuse.a = 1.0
        light.specular.r = light.specular.g = light.specular.b = 0.2 * intensity
        light.direction.x, light.direction.y, light.direction.z = direction
        light.cast_shadows = False
        return self._call('light_config', light, Light)


def make_pool(rng, test=False):
    """60 thin patches of varied shape and colour, plus 4 side-wipe strips."""
    colours = COLOURS_TEST if test else COLOURS
    pool = []
    for i in range(60):
        kind = rng.choice(['tape', 'tape', 'dirt', 'dirt', 'wear', 'tyre', 'paint'])
        if kind == 'dirt':
            sx, sy = rng.uniform(0.004, 0.025), rng.uniform(0.004, 0.025)
        elif kind in ('wear', 'tyre'):
            sx, sy = rng.uniform(0.08, 0.16), rng.uniform(0.006, 0.03)
        elif kind == 'paint':
            sx, sy = rng.uniform(0.01, 0.06), rng.uniform(0.003, 0.02)
        else:
            sx, sy = rng.uniform(0.02, 0.08), rng.uniform(0.015, 0.05)
        if test:
            sx, sy = (sx * 0.8, sy * 1.3) if kind == 'tape' else (sx * 1.3, sy)
        pool.append((f'damage_{i}', kind, sx, sy, rng.choice(colours[kind])))
    for i in range(4):          # long enough to wipe a whole 12 cm side
        pool.append((f'wipe_{i}', 'wipe', 0.15, rng.uniform(0.015, 0.03),
                     rng.choice(colours['wear'] + colours['dirt'])))
    return pool


def patch_sdf(name, sx, sy, rgb, thickness=0.001):
    shape = f'<geometry><box><size>{sx} {sy} {thickness}</size></box></geometry>'
    r, g, b = rgb
    return (f"<sdf version='1.9'><model name='{name}'><static>true</static>"
            f"<pose>{PARK[0]} {PARK[1]} {PARK[2]} 0 0 0</pose><link name='l'>"
            f"<visual name='v'><cast_shadows>false</cast_shadows>{shape}<material>"
            f"<ambient>{r} {g} {b} 1</ambient><diffuse>{r} {g} {b} 1</diffuse>"
            "</material></visual></link></model></sdf>")


GRID = np.stack(np.meshgrid(np.linspace(-FRAME_HALF, FRAME_HALF, 121),
                            np.linspace(-FRAME_HALF, FRAME_HALF, 121)), -1).reshape(-1, 2)
IN_PLATE = np.all(np.abs(GRID) <= 0.05, axis=1)
IN_RING = ~IN_PLATE
IN_TAG = [np.all(np.abs(GRID - tag_object_corners(q)[:, :2].mean(axis=0)) <= TAG_SIZE_M / 2,
                 axis=1) for q in range(4)]


def hidden_mask(placed, points):
    hidden = np.zeros(len(points), dtype=bool)
    for (_, _, sx, sy, _), (x, y, yaw) in placed:
        c, s = math.cos(yaw), math.sin(yaw)
        dx, dy = points[:, 0] - x, points[:, 1] - y
        hidden |= (np.abs(c * dx + s * dy) <= sx / 2) & (np.abs(-s * dx + c * dy) <= sy / 2)
    return hidden


def coverage(placed):
    """(plate, red ring, each tag) shares hidden by the patches."""
    hidden = hidden_mask(placed, GRID)
    return (float(hidden[IN_PLATE].mean()), float(hidden[IN_RING].mean()),
            [float(hidden[t].mean()) for t in IN_TAG])


def place_damage(rng, pool, target, wipe):
    """Patches and their poses on the plate until the plate coverage reaches target."""
    placed = []
    if wipe:                    # a whole side of the frame, like a scuffed edge
        patch = next(p for p in pool if p[1] == 'wipe')
        side = rng.randrange(4)
        offset = FRAME_HALF - 0.004 + rng.uniform(-0.004, 0.004)
        x, y = [(0, offset), (offset, 0), (0, -offset), (-offset, 0)][side]
        placed.append((patch, (x, y, 0.0 if side in (0, 2) else math.pi / 2)))
    for patch in rng.sample([p for p in pool if p[1] != 'wipe'], 60):
        if coverage(placed)[0] >= target or len(placed) >= 14:
            break
        if patch[1] in ('wear', 'tyre'):
            pose = (rng.uniform(-0.05, 0.05) if rng.random() < 0.5 else 0.0,
                    rng.uniform(-0.05, 0.05),
                    rng.choice([0.0, math.pi / 2]) + rng.uniform(-0.2, 0.2))
        else:
            pose = (rng.uniform(-0.06, 0.06), rng.uniform(-0.06, 0.06), rng.uniform(0, math.pi))
        placed.append((patch, pose))
    return placed


def classify(tag_covers, ring_cover, cls):
    """Object class, None for no object, or 'skip' for an in-between sample."""
    best = min(tag_covers)
    if best <= TAG_READABLE:
        identity = cls
    elif best >= TAG_UNREADABLE:
        identity = 0
    else:
        return 'skip'
    if identity == 0:
        if ring_cover <= FRAME_VISIBLE:
            return 0
        return None if ring_cover >= FRAME_GONE else 'skip'
    return identity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--per-marker', type=int, default=800)
    parser.add_argument('--negatives', type=int, default=400)
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--val-share', type=float, default=0.15)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--mat-share', type=float, default=0.4)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    out = Path(args.out)
    splits = ('test',) if args.test else ('train', 'val')
    for split in splits:
        (out / 'images' / split).mkdir(parents=True, exist_ok=True)
        (out / 'labels' / split).mkdir(parents=True, exist_ok=True)
    rclpy.init()
    camera, gz, detector = Camera(), Gazebo(), tag_detector()
    if camera.fresh(frames=2, timeout=60.0) is None:
        sys.exit('no downward camera images: is the isolated simulation running?')
    pool = make_pool(rng, args.test)
    for name, _, sx, sy, rgb in pool:
        gz.create(patch_sdf(name, sx, sy, rgb))
    mats = [f'mat_{i}' for i in range(len(MAT_COLOURS_TEST if args.test else MAT_COLOURS))]
    for name, rgb in zip(mats, MAT_COLOURS_TEST if args.test else MAT_COLOURS):
        gz.create(patch_sdf(name, 1.3, 1.3, rgb, thickness=0.0005))
    light_range = (0.25, 1.3) if args.test else (0.35, 1.1)
    manifest, index = [], 0
    skipped = {'no_clean_label': 0, 'ambiguous': 0, 'capture': 0}
    jobs = [m for m in MARKERS for _ in range(args.per_marker)] + ['NONE'] * args.negatives
    rng.shuffle(jobs)
    started = time.monotonic()
    for n, marker in enumerate(jobs):
        for patch in pool:
            gz.set_pose(patch[0], *PARK)
        for name in mats:
            gz.set_pose(name, *PARK)
        gz.light(rng.uniform(*light_range),
                 (rng.uniform(-0.6, 0.6), rng.uniform(-0.6, 0.6), -1.0))
        mat = rng.choice(mats) if rng.random() < args.mat_share else None
        record = {'marker': marker, 'mat': mat}
        if marker == 'NONE':
            x, y = rng.uniform(-15, 15), rng.choice([rng.uniform(-3, 3), rng.uniform(5.6, 8.4)])
            gz.set_pose(ROBOT, x, y, 0.0, rng.uniform(-math.pi, math.pi))
            if mat:
                gz.set_pose(mat, x + rng.uniform(-0.3, 0.3), y + rng.uniform(-0.3, 0.3), MAT_Z,
                            rng.uniform(0, math.pi))
            image = camera.fresh()
            if image is None:
                skipped['capture'] += 1
                continue
            label = None
            record.update(level='none', wipe=False)
        else:
            (mx, my), headings, cls, payload = MARKERS[marker]
            yaw = (rng.choice(headings) + rng.uniform(-0.26, 0.26) if rng.random() < 0.8
                   else rng.uniform(-math.pi, math.pi))
            gz.set_pose(ROBOT, mx + rng.uniform(-OFFSET_M, OFFSET_M),
                        my + rng.uniform(-OFFSET_M, OFFSET_M), 0.0, yaw)
            if mat:
                gz.set_pose(mat, mx + rng.uniform(-0.3, 0.3), my + rng.uniform(-0.3, 0.3), MAT_Z,
                            rng.uniform(0, math.pi))
            clean = camera.fresh()
            if clean is None:
                skipped['capture'] += 1
                continue
            found, image_points, object_points, ids = detect_floor_tags(detector, clean)
            if found != payload or len(ids) < 4:
                skipped['no_clean_label'] += 1
                continue
            homography, _ = cv2.findHomography(object_points[:, :2], image_points)
            keypoints = cv2.perspectiveTransform(
                PLATE_KEYPOINTS.reshape(-1, 1, 2), homography).reshape(-1, 2)
            level = rng.choices([lv[0] for lv in LEVELS], [lv[1] for lv in LEVELS])[0]
            low, high = next(lv[2] for lv in LEVELS if lv[0] == level)
            wipe = level != 'clean' and rng.random() < SIDE_WIPE_SHARE
            placed = [] if level == 'clean' else place_damage(rng, pool, rng.uniform(low, high),
                                                              wipe)
            for (name, *_), (px, py, pyaw) in placed:
                gz.set_pose(name, mx + px, my + py, PATCH_Z, pyaw)
            image = camera.fresh() if placed else clean
            if image is None:
                skipped['capture'] += 1
                continue
            plate_cover, ring_cover, tag_covers = coverage(placed)
            object_class = classify(tag_covers, ring_cover, cls)
            if object_class == 'skip':
                skipped['ambiguous'] += 1
                continue
            h, w = image.shape[:2]
            hidden = hidden_mask(placed, PLATE_KEYPOINTS)
            label = None
            if object_class is not None:
                corners = keypoints[:4]
                x0, y0 = np.clip(corners.min(axis=0), 0, [w - 1, h - 1])
                x1, y1 = np.clip(corners.max(axis=0), 0, [w - 1, h - 1])
                kp = []
                for (kx, ky), covered_point in zip(keypoints, hidden):
                    inside = 0 <= kx < w and 0 <= ky < h
                    visibility = 0 if not inside else (1 if covered_point else 2)
                    kp += [kx / w if inside else 0.0, ky / h if inside else 0.0, visibility]
                label = (object_class, (x0 + x1) / 2 / w, (y0 + y1) / 2 / h,
                         (x1 - x0) / w, (y1 - y0) / h, kp)
            record.update(level=level, wipe=wipe, plate_cover=round(plate_cover, 3),
                          frame_cover=round(ring_cover, 3),
                          tag_covers=[round(c, 3) for c in tag_covers],
                          keypoints_covered=int(hidden.sum()))
        split = 'test' if args.test else ('val' if rng.random() < args.val_share else 'train')
        stem = f'{index:05d}_{marker}_{record["level"]}'
        cv2.imwrite(str(out / 'images' / split / f'{stem}.png'), image)
        text = ''
        if label is not None:
            c, bx, by, bw, bh, kp = label
            text = f'{c} {bx:.6f} {by:.6f} {bw:.6f} {bh:.6f} ' + ' '.join(
                f'{v:.6f}' if i % 3 != 2 else f'{int(v)}' for i, v in enumerate(kp)) + '\n'
        (out / 'labels' / split / f'{stem}.txt').write_text(text)
        record.update(file=f'{split}/{stem}.png', label=None if label is None else int(label[0]))
        manifest.append(record)
        index += 1
        if n % 100 == 0:
            rate = (time.monotonic() - started) / max(n, 1)
            print(f'{n}/{len(jobs)} done, {index} kept, skipped {skipped}, '
                  f'{rate:.1f} s each, ~{rate * (len(jobs) - n) / 60:.0f} min left', flush=True)
    meta = {'samples': manifest, 'skipped': skipped, 'classes': CLASS_NAMES,
            'keypoints': KEYPOINT_NAMES, 'tag_readable': TAG_READABLE,
            'tag_unreadable': TAG_UNREADABLE, 'frame_visible': FRAME_VISIBLE,
            'frame_gone': FRAME_GONE, 'held_out_test': args.test, 'seed': args.seed}
    (out / 'manifest.json').write_text(json.dumps(meta, indent=1))
    if not args.test:
        (out / 'data.yaml').write_text(
            'path: .\ntrain: images/train\nval: images/val\n'
            f'kpt_shape: [{len(KEYPOINT_NAMES)}, 3]\nnames:\n' +
            ''.join(f'  {i}: {name}\n' for i, name in enumerate(CLASS_NAMES)))
    print(f'dataset: {index} images in {out}, skipped {skipped}')


if __name__ == '__main__':
    main()
