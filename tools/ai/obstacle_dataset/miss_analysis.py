#!/usr/bin/env python3
"""Why a model misses objects on a held-out test: per class, by box height, cut at the
image edge, no box at all nearby, a loose box of the right class, or another class.
Also crop sheets of missed and found objects (box height >= --sheet-min px) to look at.

Written 2026-11-18 after v4n: 813 of 924 missed real forklifts had no box at all,
and 90 % of the real "forklifts" were pallet trucks (see ai_data/AI_REVIEW.md).

Usage: miss_analysis.py --model best.pt --data ai_data/loco_yolo_v5 --split test --out DIR
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

CLASSES = ['person', 'pallet', 'cart', 'forklift']
BANDS = [(0, 50), (50, 100), (100, 200), (200, 10_000)]
CONF, IOU = 0.25, 0.5


def iou(a, b):
    x0, y0 = np.maximum(a[:, None, 0], b[None, :, 0]), np.maximum(a[:, None, 1], b[None, :, 1])
    x1, y1 = np.minimum(a[:, None, 2], b[None, :, 2]), np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area = lambda r: (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def band(h):
    lo, hi = next(b for b in BANDS if b[0] <= h < b[1])
    return f'{lo}-{hi if hi < 10_000 else "max"} px'


def sheet(records, data, split, path, n=30):
    tiles = []
    for r in records[:n]:
        im = cv2.imread(str(Path(data, 'images', split, r['file'])))
        x0, y0, x1, y1 = (int(v) for v in r['box'])
        c = im[max(0, y0 - 10):y1 + 10, max(0, x0 - 10):x1 + 10]
        s = 200 / max(c.shape[:2])
        c = cv2.resize(c, (max(1, int(c.shape[1] * s)), max(1, int(c.shape[0] * s))))
        t = np.zeros((200, 200, 3), np.uint8)
        t[:c.shape[0], :c.shape[1]] = c
        tiles.append(t)
    if not tiles:
        return
    while len(tiles) % 6:
        tiles.append(np.zeros((200, 200, 3), np.uint8))
    cv2.imwrite(str(path), cv2.vconcat([cv2.hconcat(tiles[i:i + 6]) for i in range(0, len(tiles), 6)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--split', default='test')
    ap.add_argument('--out', required=True)
    ap.add_argument('--sheet-min', type=int, default=100)
    args = ap.parse_args()
    from ultralytics import YOLO
    model = YOLO(args.model)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    recs = []
    for f in sorted(Path(args.data, 'images', args.split).glob('*.jpg')):
        gt = [l.split() for l in Path(str(f).replace('/images/', '/labels/')).with_suffix('.txt')
              .read_text().split('\n') if l.strip()]
        if not gt:
            continue
        r = model.predict(str(f), conf=CONF, imgsz=640, verbose=False)[0]
        h, w = r.orig_shape
        pb, pc = r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int)
        gc = np.array([int(g[0]) for g in gt])
        gb = np.array([[(float(g[1]) - float(g[3]) / 2) * w, (float(g[2]) - float(g[4]) / 2) * h,
                        (float(g[1]) + float(g[3]) / 2) * w, (float(g[2]) + float(g[4]) / 2) * h] for g in gt])
        m = iou(gb, pb) if len(pb) else np.zeros((len(gb), 0))
        for i, b in enumerate(gb):
            same = m[i][pc == gc[i]] if len(pb) else np.zeros(0)
            other = m[i][pc != gc[i]] if len(pb) else np.zeros(0)
            found = bool(len(same) and same.max() >= IOU)
            recs.append({'file': f.name, 'cls': CLASSES[gc[i]], 'box': [round(float(v), 1) for v in b],
                         'h_px': round(float(b[3] - b[1])), 'found': found,
                         'edge_cut': bool(b[0] <= 2 or b[1] <= 2 or b[2] >= w - 2 or b[3] >= h - 2),
                         'nothing_near': bool((m[i].max() if len(pb) else 0) < 0.3),
                         'loose_same_class': bool((not found) and len(same) and 0.3 <= same.max() < IOU),
                         'other_class': (CLASSES[pc[np.flatnonzero(pc != gc[i])[other.argmax()]]]
                                         if (not found) and len(other) and other.max() >= IOU else None)})
    (out / 'objects.json').write_text(json.dumps(recs))
    report = {}
    for cls in CLASSES:
        R = [r for r in recs if r['cls'] == cls]
        if not R:
            continue
        miss = [r for r in R if not r['found']]
        rep = {'objects': len(R), 'found': len(R) - len(miss),
               'by_size': {b: f"{sum(1 for r in R if band(r['h_px']) == b and r['found'])}/"
                              f"{sum(1 for r in R if band(r['h_px']) == b)} found"
                           for b in sorted({band(r['h_px']) for r in R}, key=lambda s: int(s.split('-')[0]))},
               'missed_with_no_box_near': sum(r['nothing_near'] for r in miss),
               'missed_loose_box': sum(r['loose_same_class'] for r in miss),
               'missed_called': dict(Counter(r['other_class'] for r in miss if r['other_class'])),
               'missed_cut_at_edge': sum(r['edge_cut'] for r in miss)}
        report[cls] = rep
        rng = random.Random(1)
        big_miss = [r for r in miss if r['h_px'] >= args.sheet_min]
        big_found = [r for r in R if r['found'] and r['h_px'] >= args.sheet_min]
        rng.shuffle(big_miss)
        rng.shuffle(big_found)
        sheet(big_miss, args.data, args.split, out / f'{cls}_missed.jpg')
        sheet(big_found, args.data, args.split, out / f'{cls}_found.jpg')
    (out / 'report.json').write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
