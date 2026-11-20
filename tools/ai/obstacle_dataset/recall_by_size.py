#!/usr/bin/env python3
"""Recall and precision by object size: does a detector find the objects that matter to the robot?

A single mAP mixes tiny far objects (pallets deep in racks) with the near ones
the robot must react to. This scores each class per size band of the true box
height in pixels (images 640 px wide): a true box counts as found when a
prediction of the same class overlaps it with IoU >= 0.5 at confidence >= CONF.
Precision per band: of the predicted boxes in that band, the share that match
a true box of the same class.

Detectors: our own weights (.pt), or an open-vocabulary YOLO-World model given
as "world:<weights>", which is told our class names in words.

Usage: recall_by_size.py --data ai_data/loco_yolo --split test --model ai_data/models/obstacles_v3/best.pt
       recall_by_size.py --data ai_data/loco_yolo --split test --model world:yolov8l-worldv2.pt
"""
import argparse
import json
from pathlib import Path

import numpy as np

CLASSES = ['person', 'pallet', 'cart', 'forklift']
WORLD_PROMPTS = ['person', 'pallet', 'hand cart', 'forklift']   # words the open model searches for
BANDS = [(0, 50), (50, 100), (100, 200), (200, 10_000)]
CONF, IOU = 0.25, 0.5


def iou(a, b):
    x0, y0 = np.maximum(a[:, None, 0], b[None, :, 0]), np.maximum(a[:, None, 1], b[None, :, 1])
    x1, y1 = np.minimum(a[:, None, 2], b[None, :, 2]), np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area = lambda r: (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def band(h):
    return next(i for i, (lo, hi) in enumerate(BANDS) if lo <= h < hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--split', default='test')
    ap.add_argument('--model', required=True)
    ap.add_argument('--out')
    args = ap.parse_args()
    from ultralytics import YOLO
    if args.model.startswith('world:'):
        model = YOLO(args.model[6:])
        model.set_classes(WORLD_PROMPTS)
    else:
        model = YOLO(args.model)
    found = np.zeros((4, len(BANDS)), int)
    total = np.zeros((4, len(BANDS)), int)
    pred_ok = np.zeros((4, len(BANDS)), int)
    pred_all = np.zeros((4, len(BANDS)), int)
    images = sorted(Path(args.data, 'images', args.split).glob('*.jpg'))
    for n, path in enumerate(images):
        r = model.predict(str(path), conf=CONF, imgsz=640, verbose=False)[0]
        h, w = r.orig_shape
        pb, pc = r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int)
        gt = [l.split() for l in Path(str(path).replace('/images/', '/labels/')).with_suffix('.txt')
              .read_text().split('\n') if l.strip()]
        gc = np.array([int(g[0]) for g in gt], int)
        gb = np.array([[(float(g[1]) - float(g[3]) / 2) * w, (float(g[2]) - float(g[4]) / 2) * h,
                        (float(g[1]) + float(g[3]) / 2) * w, (float(g[2]) + float(g[4]) / 2) * h]
                       for g in gt]).reshape(-1, 4)
        m = iou(gb, pb) if len(gb) and len(pb) else np.zeros((len(gb), len(pb)))
        same = gc[:, None] == pc[None, :]
        hit = (m >= IOU) & same
        for i in range(len(gb)):
            b = band(gb[i, 3] - gb[i, 1])
            total[gc[i], b] += 1
            found[gc[i], b] += hit[i].any()
        for j in range(len(pb)):
            b = band(pb[j, 3] - pb[j, 1])
            pred_all[pc[j], b] += 1
            pred_ok[pc[j], b] += hit[:, j].any()
        if (n + 1) % 500 == 0:
            print(f'{n + 1}/{len(images)}', flush=True)
    res = {'model': args.model, 'data': f'{args.data}/{args.split}', 'conf': CONF, 'iou': IOU,
           'bands_px_height': BANDS, 'per_class': {}}
    for c, name in enumerate(CLASSES):
        if total[c].sum() == 0 and pred_all[c].sum() == 0:
            continue
        res['per_class'][name] = [
            {'band': f'{lo}-{hi if hi < 10_000 else "max"} px', 'objects': int(total[c, i]),
             'recall': round(found[c, i] / total[c, i], 3) if total[c, i] else None,
             'predicted': int(pred_all[c, i]),
             'precision': round(pred_ok[c, i] / pred_all[c, i], 3) if pred_all[c, i] else None}
            for i, (lo, hi) in enumerate(BANDS)]
    text = json.dumps(res, indent=1)
    print(text)
    if args.out:
        Path(args.out).write_text(text)


if __name__ == '__main__':
    main()
