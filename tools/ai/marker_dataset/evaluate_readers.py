#!/usr/bin/env python3
"""Score the floor-marker readers on one held-out test set (laptop CPU).

Readers compared on the same images:
  v3       YOLO-pose, one class, keypoints in image order; station read from
           the tags (marker_identity.identify), frame fitted (marker_geometry),
           state from marker_reader.assess with the expected station.
  v2       YOLO-pose with one class per station, keypoints in marker order
           (trained without the LED ring; shown for comparison only).
  classic  the OpenCV AprilTag reader the docking uses today
           (docking_markers.detect_floor_tags).

The test set (augment.py --test) has four kinds of image, from the manifest:
  real     the station's own tags, marker still there      -> must name the station
  foreign  random tag codes in the frame (a "new station")  -> must NOT name a known station
  gone     marker damaged beyond reading (no label)         -> must not name a station
  empty    floor without a marker                           -> must find nothing

Usage: evaluate_readers.py --test ai_data/markers_v3_test --v3 v3.onnx [--v2 v2.onnx] --out dir
"""
import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src/tugger_control'))
from tugger_control.docking_markers import detect_floor_tags, tag_detector  # noqa: E402
from tugger_control.marker_geometry import fit_frame  # noqa: E402
from tugger_control.marker_identity import identify  # noqa: E402
from tugger_control.marker_reader import (  # noqa: E402
    CAMERA_HEIGHT_M, FAULT_BELOW, FRAME_SIZE_M, MarkerModel, STATION_OF_PAYLOAD, assess)

FOCAL_PX = 320 / np.tan(2.44346 / 2)    # downward camera: 640 px wide, 140 degree view
SIDE_PX = FOCAL_PX * FRAME_SIZE_M / CAMERA_HEIGHT_M   # red frame side in the image (as the node)
# The labels' frame corners come from the tags, whose plate lies 1 mm above the red
# frame in the simulation: about the image centre they are 2.0 % too far out
# (measured 2026-11-16, 248.3 / 253.4 px). The fitted frame is compared with the
# corners moved onto the red frame's plane.
PLATE_TO_FRAME = 248.34 / 253.4
IMAGE_CENTRE = np.array([319.5, 239.5])
V2_NAMES = [None, 'HOME', 'S1', 'D2', 'S3', 'D4']   # v2 classes; 0 = unknown_marker


def kind_of(sample):
    if sample['marker'] == 'NONE':
        return 'empty'
    if not sample.get('marker_present'):
        return 'gone'
    return 'foreign' if sample.get('tag_codes') is not None else 'real'


def ground_truth(label_path, w, h):
    """Label keypoints (20 x 2, image order) and their visibility (0 = outside the
    image or covered; YOLO clips those to 0, so they are not scored)."""
    rows = [r.split() for r in label_path.read_text().splitlines() if r.strip()]
    if not rows:
        return None, None
    v = np.array(rows[0][5:], float).reshape(20, 3)
    return v[:, :2] * [w, h], v[:, 2] > 0


def cyclic_error(corners, truth):
    """Mean corner distance, the four corners matched in their best cyclic turn."""
    corners, truth = np.asarray(corners), np.asarray(truth)
    return min(float(np.hypot(*(np.roll(corners, k, axis=0) - truth).T).mean()) for k in range(4))


def station_outcome(named, truth):
    """right / wrong / none for a reader that named `named` (station or None)."""
    if named is None:
        return 'none'
    return 'right' if named == truth else 'wrong'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--test', required=True)
    ap.add_argument('--v3', required=True)
    ap.add_argument('--v2')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    test, out = Path(args.test), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((test / 'manifest.json').read_text())
    samples = manifest['samples']
    clean = {c['file']: c for c in json.loads(
        (Path(manifest['clean_set']) / 'clean.json').read_text())['samples']}
    v3, v2 = MarkerModel(args.v3), MarkerModel(args.v2) if args.v2 else None
    classic = tag_detector()

    counts = defaultdict(Counter)        # (reader, kind) -> outcome counts
    by_level = defaultdict(Counter)      # (reader, level) on real markers
    states = defaultdict(Counter)        # v3 state per kind
    frame_err, tag_err, fit_err, times = [], [], [], defaultdict(list)
    rows = []
    for s in samples:
        img = cv2.imread(str(test / 'images' / s['file']))
        h, w = img.shape[:2]
        kind = kind_of(s)
        truth = STATION_OF_PAYLOAD.get(f"TUGGER:{s['marker']}")
        expected = truth if kind != 'empty' else None
        gt, seen = ground_truth(test / 'labels' / s['file'].replace('.png', '.txt'), w, h)
        # Frame corners as captured (not clipped), moved onto the red frame's plane.
        true_frame = (IMAGE_CENTRE + (np.asarray(clean[s['clean']]['keypoints'])[:4]
                                      - IMAGE_CENTRE) * PLATE_TO_FRAME
                      if s['marker'] != 'NONE' else None)
        row = {'file': s['file'], 'kind': kind, 'level': s['level'],
               'kinds': s.get('kinds')}

        t = time.perf_counter()
        det = v3.detect(img)
        found = bool(det) and det[0][1] >= FAULT_BELOW
        named, fit, ident = None, None, None
        if det:
            kp = det[0][6][:, :2]
            ident = identify(img, kp)
            fit = fit_frame(img, kp[:4], SIDE_PX)
            if ident['state'] == 'OK':
                named = STATION_OF_PAYLOAD.get(ident['payload'])
        times['v3'].append(time.perf_counter() - t)
        state = assess(det, ident, expected, fit)['state']
        states[kind][state] += 1
        if gt is not None and det:
            e = np.hypot(*(kp - gt).T)
            frame_err += list(e[:4][seen[:4]])
            tag_err += list(e[4:][seen[4:]])
            if fit['ok']:
                fit_err.append(cyclic_error(fit['corners'], true_frame))
                row['fit_error_px'] = round(fit_err[-1], 2)
        row['v3'] = {'found': found, 'score': round(det[0][1], 3) if det else 0.0,
                     'station': named, 'fit_ok': bool(fit and fit['ok']), 'state': state}
        counts['v3', kind]['found' if found else 'not found'] += 1
        counts['v3', kind][station_outcome(named, truth if kind == 'real' else None)] += 1
        if kind == 'real':
            counts['v3', kind]['frame fitted' if fit and fit['ok'] else 'frame not fitted'] += 1
            by_level['v3', s['level']][station_outcome(named, truth)] += 1

        t = time.perf_counter()
        payload = detect_floor_tags(classic, img)[0]
        times['classic'].append(time.perf_counter() - t)
        c_named = STATION_OF_PAYLOAD.get(payload)
        row['classic'] = c_named
        counts['classic', kind][station_outcome(c_named, truth if kind == 'real' else None)] += 1
        if kind == 'real':
            by_level['classic', s['level']][station_outcome(c_named, truth)] += 1

        if v2:
            t = time.perf_counter()
            d2 = v2.detect(img)
            times['v2'].append(time.perf_counter() - t)
            v2_found = bool(d2) and d2[0][1] >= FAULT_BELOW
            name2 = V2_NAMES[d2[0][0]] if v2_found else None
            v2_named = STATION_OF_PAYLOAD.get(f'TUGGER:{name2}') if name2 else None
            row['v2'] = {'found': v2_found, 'station': v2_named}
            counts['v2', kind]['found' if v2_found else 'not found'] += 1
            counts['v2', kind][station_outcome(v2_named, truth if kind == 'real' else None)] += 1
            if kind == 'real':
                by_level['v2', s['level']][station_outcome(v2_named, truth)] += 1
        rows.append(row)

    q = (lambda a: {'median_px': round(float(np.median(a)), 1),
                    'p90_px': round(float(np.percentile(a, 90)), 1)} if a else None)
    report = {
        'images': Counter(kind_of(s) for s in samples),
        'outcomes': {f'{r}/{k}': dict(c) for (r, k), c in sorted(counts.items())},
        'real_by_damage': {f'{r}/{lv}': dict(c) for (r, lv), c in sorted(by_level.items())},
        'v3_states': {k: dict(c) for k, c in states.items()},
        'v3_keypoints': {'frame_corners': q(frame_err), 'tag_corners': q(tag_err),
                         'fitted_frame': q(fit_err)},
        'cpu_ms_median': {r: round(1000 * float(np.median(t)), 1) for r, t in times.items()},
    }
    (out / 'report.json').write_text(json.dumps(report, indent=1))
    (out / 'per_image.json').write_text(json.dumps(rows, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
