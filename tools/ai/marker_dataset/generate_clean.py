#!/usr/bin/env python3
"""Clean marker images, labelled automatically (dataset v3, step 1 of 2).

Only clean markers are captured here: the robot over each station's marker at
random docking-like offsets and headings, under varied light and floor mats,
plus empty floor. Each image is labelled from its own tags: the 16 tag
corners map the marker plate onto the image (a homography), and the 20
keypoints (4 frame corners, 16 tag corners) follow from the plate geometry.

All damage (tape, wiped sides, dust, smears, fading, scratches, glare, blur)
is added afterwards by augment.py from these clean images and their
homographies; nothing has to be captured again. The same labelling works
for real photographs: one clear photo per marker is enough.

Labels: one class, 'marker', with 20 keypoints. The station is not learnt;
it is read from the tags at run time (marker_identity.py), so new stations
need no retraining.

Usage (isolated Gazebo running, see run_generator.sh):
  python3 tools/ai/marker_dataset/generate_clean.py --out ai_data/markers_v3_clean --per-marker 400
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_pose import (MARKERS, MAT_COLOURS, MAT_COLOURS_TEST, MAT_Z,  # noqa: E402
                           OFFSET_M, PARK, PLATE_KEYPOINTS, KEYPOINT_NAMES, ROBOT, Camera,
                           Gazebo, detect_floor_tags, patch_sdf, tag_detector)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--per-marker', type=int, default=400)
    parser.add_argument('--negatives', type=int, default=200)
    parser.add_argument('--seed', type=int, default=21)
    parser.add_argument('--mat-share', type=float, default=0.4)
    parser.add_argument('--test', action='store_true', help='mats and light kept for testing')
    args = parser.parse_args()
    rng = random.Random(args.seed)
    out = Path(args.out)
    (out / 'images').mkdir(parents=True, exist_ok=True)
    rclpy.init()
    camera, gz, detector = Camera(), Gazebo(), tag_detector()
    if camera.fresh(frames=2, timeout=60.0) is None:
        sys.exit('no downward camera images: is the isolated simulation running?')
    colours = MAT_COLOURS_TEST if args.test else MAT_COLOURS
    mats = [f'mat_{i}' for i in range(len(colours))]
    for name, rgb in zip(mats, colours):
        gz.create(patch_sdf(name, 1.3, 1.3, rgb, thickness=0.0005))
    light_range = (0.25, 1.3) if args.test else (0.35, 1.1)
    samples, skipped = [], 0
    jobs = [m for m in MARKERS for _ in range(args.per_marker)] + ['NONE'] * args.negatives
    rng.shuffle(jobs)
    started = time.monotonic()
    for n, marker in enumerate(jobs):
        for name in mats:
            gz.set_pose(name, *PARK)
        gz.light(rng.uniform(*light_range),
                 (rng.uniform(-0.6, 0.6), rng.uniform(-0.6, 0.6), -1.0))
        mat = rng.choice(mats) if rng.random() < args.mat_share else None
        if marker == 'NONE':
            x, y = rng.uniform(-15, 15), rng.choice([rng.uniform(-3, 3), rng.uniform(5.6, 8.4)])
            yaw = rng.uniform(-math.pi, math.pi)
        else:
            (x, y), headings, _, _ = MARKERS[marker]
            yaw = (rng.choice(headings) + rng.uniform(-0.26, 0.26) if rng.random() < 0.8
                   else rng.uniform(-math.pi, math.pi))
        gz.set_pose(ROBOT, x + (0 if marker == 'NONE' else rng.uniform(-OFFSET_M, OFFSET_M)),
                    y + (0 if marker == 'NONE' else rng.uniform(-OFFSET_M, OFFSET_M)), 0.0, yaw)
        if mat:
            gz.set_pose(mat, x + rng.uniform(-0.3, 0.3), y + rng.uniform(-0.3, 0.3), MAT_Z,
                        rng.uniform(0, math.pi))
        image = camera.fresh()
        if image is None:
            skipped += 1
            continue
        record = {'marker': marker, 'mat': mat}
        if marker != 'NONE':
            payload = MARKERS[marker][3]
            found, image_points, object_points, ids = detect_floor_tags(detector, image)
            if found != payload or len(ids) < 4:
                skipped += 1
                continue
            homography, _ = cv2.findHomography(object_points[:, :2], image_points)
            keypoints = cv2.perspectiveTransform(PLATE_KEYPOINTS.reshape(-1, 1, 2),
                                                 homography).reshape(-1, 2)
            record.update(payload=payload, homography=homography.round(6).tolist(),
                          keypoints=keypoints.round(3).tolist())
        stem = f'{len(samples):05d}_{marker}'
        cv2.imwrite(str(out / 'images' / f'{stem}.png'), image)
        record['file'] = f'{stem}.png'
        samples.append(record)
        if n % 100 == 0:
            rate = (time.monotonic() - started) / max(n, 1)
            print(f'{n}/{len(jobs)} done, {len(samples)} kept, {skipped} skipped, '
                  f'~{rate * (len(jobs) - n) / 60:.0f} min left', flush=True)
    (out / 'clean.json').write_text(json.dumps(
        {'samples': samples, 'keypoints': KEYPOINT_NAMES, 'held_out_test': args.test,
         'seed': args.seed, 'image_size': [640, 480]}, indent=1))
    print(f'clean set: {len(samples)} images in {out}, {skipped} skipped')


if __name__ == '__main__':
    main()
