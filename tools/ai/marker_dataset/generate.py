#!/usr/bin/env python3
"""Generate a YOLO dataset of floor markers (QR + red frame) in many conditions.

Runs against a headless Gazebo (isolated partition/domain, simulation.launch.py
only). For each sample the robot is placed over a station's floor marker at a
random docking-like offset and heading, and the downward camera is captured
twice at the same pose:

1. clean: the current OpenCV reader finds the red frame and decodes the QR;
   their image corners are the ground-truth boxes for this pose;
2. damaged: patches (tape, dirt, worn strips, tyre marks, paint) from a pool
   are laid on and around the marker, the light is changed, and the image is
   captured again with the same boxes.

Each object is labelled only while it is reasonably recognisable: covered up
to LABEL_MAX_COVER it is labelled, from UNLABEL_MIN_COVER it counts as gone
(background), and samples in between are left out so the model is not taught
either way. The trained model's confidence then falls as damage grows, which
the reader uses for its ALERT and FAULT levels.

Classes: 0 red_frame, 1 HOME, 2 S1, 3 D2, 4 S3, 5 D4 (the QR's identity).

Usage (with the simulation running in the isolated partition):
  python3 tools/ai/marker_dataset/generate.py --out ai_data/markers_v1 --per-marker 600
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
    decode_floor_marker, detect_red_alignment_frame)

WORLD = 'tugger_test'
ROBOT = 'autonomous_tugger'
# Marker centre, docking heading(s) of the robot, class id, QR payload.
MARKERS = {
    'HOME': ((0.0, 0.0), (0.0, math.pi), 1, 'TUGGER:HOME'),
    'S1': ((-17.5, -7.0), (math.pi / 2, -math.pi / 2), 2, 'TUGGER:S1'),
    'D2': ((-17.5, 7.0), (math.pi / 2, -math.pi / 2), 3, 'TUGGER:D2'),
    'S3': ((17.5, 7.0), (math.pi / 2, -math.pi / 2), 4, 'TUGGER:S3'),
    'D4': ((17.5, -7.0), (math.pi / 2, -math.pi / 2), 5, 'TUGGER:D4'),
}
CLASS_NAMES = ['red_frame', 'HOME', 'S1', 'D2', 'S3', 'D4']
QR_HALF = 0.05
FRAME_HALF = 0.06
LABEL_MAX_COVER = 0.40
UNLABEL_MIN_COVER = 0.60
OFFSET_M = 0.06
PATCH_Z = 0.0055                      # just above the QR plane (0.005 m)
PARK = (0.0, 0.0, -5.0)               # unused patches wait under the floor
COLOURS = {
    'tape': [(0.78, 0.74, 0.58), (0.85, 0.85, 0.85), (0.2, 0.35, 0.75)],
    'dirt': [(0.35, 0.27, 0.18), (0.28, 0.24, 0.20), (0.45, 0.38, 0.28)],
    'wear': [(0.55, 0.55, 0.52), (0.62, 0.62, 0.60)],
    'tyre': [(0.08, 0.08, 0.08), (0.15, 0.15, 0.15)],
    'paint': [(0.92, 0.92, 0.90), (0.95, 0.85, 0.10)],
}
# A separate test set uses damage the model never saw in training: other
# colours and shapes, other floor mats, light beyond the training range. It
# measures recognition, not memorisation (user, 2026-11-15: do not overfit).
COLOURS_TEST = {
    'tape': [(0.70, 0.20, 0.15), (0.95, 0.95, 0.70)],
    'dirt': [(0.20, 0.18, 0.22), (0.55, 0.50, 0.42)],
    'wear': [(0.48, 0.48, 0.46)],
    'tyre': [(0.03, 0.03, 0.05)],
    'paint': [(0.10, 0.60, 0.20), (0.95, 0.50, 0.10)],
}
# Floor mats laid under the marker in part of the images, so the station can
# only be told from the QR itself. The drop-offs stood on green plates and the rest
# on grey (iteration 1); the plates are removed from the world (2026-11-15), and
# varied mats keep the model from tying a station to any floor colour.
MAT_COLOURS = [(0.55, 0.55, 0.52), (0.72, 0.72, 0.70), (0.0, 0.35, 0.0),
               (0.15, 0.30, 0.60), (0.85, 0.75, 0.15), (0.60, 0.52, 0.40)]
MAT_COLOURS_TEST = [(0.40, 0.18, 0.12), (0.30, 0.32, 0.36), (0.90, 0.90, 0.88)]
MAT_Z = 0.0034                        # just under the red frame plane (0.004)
# Damage levels: share of samples and the QR coverage aimed at.
LEVELS = [('clean', 0.15, (0.0, 0.0)), ('light', 0.30, (0.03, 0.20)),
          ('medium', 0.30, (0.20, 0.45)), ('heavy', 0.25, (0.45, 0.95))]


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
        """An image taken after at least ``frames`` new frames (settled scene)."""
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
    """60 thin patches of varied shape and colour: (name, kind, sx, sy, rgb)."""
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
        if test:                         # other shapes too: squarer tape, longer paint
            sx, sy = (sx * 0.8, sy * 1.3) if kind == 'tape' else (sx * 1.3, sy)
        pool.append((f'damage_{i}', kind, sx, sy, rng.choice(colours[kind])))
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
IN_QR = np.all(np.abs(GRID) <= QR_HALF, axis=1)
IN_RING = ~IN_QR


def covered(placed):
    """Share of the QR and of the red ring hidden by patches (marker frame)."""
    hidden = np.zeros(len(GRID), dtype=bool)
    for (_, _, sx, sy, _), (x, y, yaw) in placed:
        c, s = math.cos(yaw), math.sin(yaw)
        dx, dy = GRID[:, 0] - x, GRID[:, 1] - y
        hidden |= (np.abs(c * dx + s * dy) <= sx / 2) & (np.abs(-s * dx + c * dy) <= sy / 2)
    return float(hidden[IN_QR].mean()), float(hidden[IN_RING].mean())


def place_damage(rng, pool, target):
    """Pick patches and poses (marker frame) until the QR coverage reaches target."""
    placed, order = [], rng.sample(pool, len(pool))
    for patch in order:
        if covered(placed)[0] >= target or len(placed) >= 14:
            break
        kind = patch[1]
        if kind in ('wear', 'tyre'):     # runs across, like wheel tracks
            pose = (rng.uniform(-0.05, 0.05) if rng.random() < 0.5 else 0.0,
                    rng.uniform(-0.05, 0.05), rng.choice([0.0, math.pi / 2]) +
                    rng.uniform(-0.2, 0.2))
        else:
            pose = (rng.uniform(-0.06, 0.06), rng.uniform(-0.06, 0.06), rng.uniform(0, math.pi))
        placed.append((patch, pose))
        if target <= 0.0:
            break
    if target <= 0.0:
        return []
    return placed


def yolo_box(corners, width, height):
    xs, ys = corners[:, 0], corners[:, 1]
    x0, x1 = max(xs.min(), 0), min(xs.max(), width - 1)
    y0, y1 = max(ys.min(), 0), min(ys.max(), height - 1)
    return ((x0 + x1) / 2 / width, (y0 + y1) / 2 / height,
            (x1 - x0) / width, (y1 - y0) / height)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--per-marker', type=int, default=600)
    parser.add_argument('--negatives', type=int, default=300)
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--val-share', type=float, default=0.15)
    parser.add_argument('--test', action='store_true',
                        help='held-out test set: unseen damage, mats and light; all in images/test')
    parser.add_argument('--mat-share', type=float, default=0.4)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    out = Path(args.out)
    for split in (('test',) if args.test else ('train', 'val')):
        (out / 'images' / split).mkdir(parents=True, exist_ok=True)
        (out / 'labels' / split).mkdir(parents=True, exist_ok=True)
    rclpy.init()
    camera, gz = Camera(), Gazebo()
    detector = cv2.QRCodeDetector()
    if camera.fresh(frames=2, timeout=60.0) is None:
        sys.exit('no downward camera images: is the isolated simulation running?')
    pool = make_pool(rng, args.test)
    for name, _, sx, sy, rgb in pool:
        gz.create(patch_sdf(name, sx, sy, rgb))
    mats = [(f'mat_{i}', rgb) for i, rgb in
            enumerate(MAT_COLOURS_TEST if args.test else MAT_COLOURS)]
    for name, rgb in mats:
        gz.create(patch_sdf(name, 1.3, 1.3, rgb, thickness=0.0005))
    light_range = (0.25, 1.3) if args.test else (0.35, 1.1)
    manifest, index, skipped = [], 0, {'no_clean_label': 0, 'ambiguous': 0}
    jobs = [m for m in MARKERS for _ in range(args.per_marker)] + ['NONE'] * args.negatives
    rng.shuffle(jobs)
    started = time.monotonic()
    for n, marker in enumerate(jobs):
        for patch in pool:
            gz.set_pose(patch[0], *PARK)
        for name, _ in mats:
            gz.set_pose(name, *PARK)
        gz.light(rng.uniform(*light_range),
                 (rng.uniform(-0.6, 0.6), rng.uniform(-0.6, 0.6), -1.0))
        mat = rng.choice(mats)[0] if rng.random() < args.mat_share else None
        if marker == 'NONE':             # floor away from any marker
            x, y = rng.uniform(-15, 15), rng.choice([rng.uniform(-3, 3), rng.uniform(5.6, 8.4)])
            gz.set_pose(ROBOT, x, y, 0.0, rng.uniform(-math.pi, math.pi))
            if mat:
                gz.set_pose(mat, x + rng.uniform(-0.3, 0.3), y + rng.uniform(-0.3, 0.3), MAT_Z,
                            rng.uniform(0, math.pi))
            image = camera.fresh()
            labels, qr_cover, ring_cover, level = [], 0.0, 0.0, 'none'
        else:
            (mx, my), headings, cls, payload = MARKERS[marker]
            yaw = (rng.choice(headings) + rng.uniform(-0.26, 0.26) if rng.random() < 0.8
                   else rng.uniform(-math.pi, math.pi))
            ox, oy = rng.uniform(-OFFSET_M, OFFSET_M), rng.uniform(-OFFSET_M, OFFSET_M)
            gz.set_pose(ROBOT, mx + ox, my + oy, 0.0, yaw)
            if mat:
                gz.set_pose(mat, mx + rng.uniform(-0.3, 0.3), my + rng.uniform(-0.3, 0.3), MAT_Z,
                            rng.uniform(0, math.pi))
            clean = camera.fresh()
            if clean is None:
                continue
            text, qr_corners = decode_floor_marker(detector, clean)
            frame_corners = detect_red_alignment_frame(clean)
            if text != payload or qr_corners is None or frame_corners is None:
                skipped['no_clean_label'] += 1
                continue
            level = rng.choices([lv[0] for lv in LEVELS], [lv[1] for lv in LEVELS])[0]
            low, high = next(lv[2] for lv in LEVELS if lv[0] == level)
            placed = place_damage(rng, pool, rng.uniform(low, high))
            for (name, *_), (px, py, pyaw) in placed:
                gz.set_pose(name, mx + px, my + py, PATCH_Z, pyaw)
            image = camera.fresh() if placed else clean
            if image is None:
                continue
            qr_cover, ring_cover = covered(placed)
            h, w = image.shape[:2]
            labels = []
            for box_cls, cover, corners in ((0, ring_cover, frame_corners),
                                            (cls, qr_cover, np.asarray(qr_corners).reshape(4, 2))):
                if LABEL_MAX_COVER < cover < UNLABEL_MIN_COVER:
                    labels = None
                    break
                if cover <= LABEL_MAX_COVER:
                    labels.append((box_cls, *yolo_box(corners, w, h)))
            if labels is None:
                skipped['ambiguous'] += 1
                continue
        split = 'test' if args.test else ('val' if rng.random() < args.val_share else 'train')
        stem = f'{index:05d}_{marker}_{level}'
        cv2.imwrite(str(out / 'images' / split / f'{stem}.png'), image)
        (out / 'labels' / split / f'{stem}.txt').write_text(
            ''.join(f'{c} {x:.6f} {y:.6f} {bw:.6f} {bh:.6f}\n' for c, x, y, bw, bh in labels))
        manifest.append({'file': f'{split}/{stem}.png', 'marker': marker, 'level': level,
                         'mat': mat,
                         'qr_cover': round(qr_cover, 3), 'frame_cover': round(ring_cover, 3),
                         'labels': [int(lb[0]) for lb in labels]})
        index += 1
        if n % 50 == 0:
            rate = (time.monotonic() - started) / max(n, 1)
            print(f'{n}/{len(jobs)} done, {index} kept, skipped {skipped}, '
                  f'{rate:.1f} s each, ~{rate * (len(jobs) - n) / 60:.0f} min left', flush=True)
    if args.test:
        (out / 'manifest.json').write_text(json.dumps(
            {'samples': manifest, 'skipped': skipped, 'label_max_cover': LABEL_MAX_COVER,
             'unlabel_min_cover': UNLABEL_MIN_COVER, 'classes': CLASS_NAMES,
             'held_out': 'unseen damage colours/shapes, mats and light'}, indent=1))
        print(f'test set: {index} images in {out}, skipped {skipped}')
        return
    (out / 'data.yaml').write_text(
        f'path: .\ntrain: images/train\nval: images/val\nnames:\n' +
        ''.join(f'  {i}: {name}\n' for i, name in enumerate(CLASS_NAMES)))
    (out / 'manifest.json').write_text(json.dumps(
        {'samples': manifest, 'skipped': skipped, 'label_max_cover': LABEL_MAX_COVER,
         'unlabel_min_cover': UNLABEL_MIN_COVER, 'classes': CLASS_NAMES}, indent=1))
    print(f'dataset: {index} images in {out}, skipped {skipped}')


if __name__ == '__main__':
    main()
