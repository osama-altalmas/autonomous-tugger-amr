#!/usr/bin/env python3
"""Real photos from Roboflow exports (YOLO format) to our classes (v4).

Each export is checked by eye first (AI_REVIEW.md lists what was kept and why).
Per source: a class mapping to ours (person, pallet, cart, forklift; unmapped
classes are left unlabelled), the true aspect ratio when Roboflow stretched
the images to a square, and an optional near-duplicate filter for video frames
from fixed cameras. Then, as for LOCO:
- images scaled to 640 px wide, JPEG 92;
- boxes thinner than 10 px are greyed out (never left as unlabelled objects);
- missing people are added by people.PersonFinder;
- 10 % validation, the rest training (these sets are not used as tests).

Usage (Ultralytics environment, from ai_data):
  external_to_yolo.py --src external/roboflow/forklift_mustafa --out rf_forklift \
      --map Fork_Lift:forklift,signaler:person --aspect 16:9
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from people import PersonFinder

CLASSES = ['person', 'pallet', 'cart', 'forklift']
WIDTH = 640
MIN_SIDE_PX = 10
GREY = 114
VAL_SHARE = 0.10


def dhash(image, size=16):
    g = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (size + 1, size))
    return (g[:, 1:] > g[:, :-1]).flatten()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--map', required=True, help='their_class:our_class,...')
    ap.add_argument('--aspect', help='true width:height when the export was stretched')
    ap.add_argument('--dedupe-bits', type=int, default=0,
                    help='drop an image whose 256-bit dHash differs from a kept one by fewer bits')
    ap.add_argument('--seed', type=int, default=7)
    args = ap.parse_args()
    src, out = Path(args.src), Path(args.out)
    import yaml
    names = yaml.safe_load((src / 'data.yaml').read_text())['names']
    mapping = dict(m.split(':') for m in args.map.split(','))
    theirs = {i: CLASSES.index(mapping[n]) for i, n in enumerate(names) if n in mapping}
    rng = random.Random(args.seed)
    people = PersonFinder()
    counts, kept_hashes, manifest = Counter(), [], []
    images = sorted(p for p in src.glob('*/images/*') if p.suffix.lower() in ('.jpg', '.jpeg', '.png'))
    for n, path in enumerate(images):
        image = cv2.imread(str(path))
        if image is None:
            counts['unreadable'] += 1
            continue
        h = round(WIDTH * int(args.aspect.split(':')[1]) / int(args.aspect.split(':')[0])) if args.aspect \
            else round(image.shape[0] * WIDTH / image.shape[1])
        image = cv2.resize(image, (WIDTH, h), interpolation=cv2.INTER_AREA)
        if args.dedupe_bits:
            hsh = dhash(image)
            if kept_hashes and min(int(np.count_nonzero(hsh != k)) for k in kept_hashes) < args.dedupe_bits:
                counts['near duplicate'] += 1
                continue
            kept_hashes.append(hsh)
        label_file = path.parent.parent / 'labels' / (path.stem + '.txt')
        lines, persons, grey = [], [], []
        for row in (label_file.read_text().split('\n') if label_file.exists() else []):
            v = row.split()
            if len(v) != 5 or int(v[0]) not in theirs:
                continue
            cls = theirs[int(v[0])]
            x, y, w, bh = (float(t) for t in v[1:])
            box = ((x - w / 2) * WIDTH, (y - bh / 2) * h, (x + w / 2) * WIDTH, (y + bh / 2) * h)
            if min(w * WIDTH, bh * h) < MIN_SIDE_PX:
                grey.append(box)
                counts['greyed thin'] += 1
                continue
            lines.append(f'{cls} {x:.6f} {y:.6f} {w:.6f} {bh:.6f}')
            counts[f'class {CLASSES[cls]}'] += 1
            if cls == 0:
                persons.append(box)
        for x0, y0, x1, y1, _ in people.missing(image, persons):
            if min(x1 - x0, y1 - y0) < MIN_SIDE_PX:
                continue
            lines.append(f'0 {(x0 + x1) / 2 / WIDTH:.6f} {(y0 + y1) / 2 / h:.6f} '
                         f'{(x1 - x0) / WIDTH:.6f} {(y1 - y0) / h:.6f}')
            counts['class person (added)'] += 1
        for x0, y0, x1, y1 in grey:
            image[max(0, int(y0)):int(np.ceil(y1)) + 1, max(0, int(x0)):int(np.ceil(x1)) + 1] = GREY
        split = 'val' if rng.random() < VAL_SHARE else 'train'
        stem = f'{out.name}_{n:05d}'
        for d in ('images', 'labels'):
            (out / d / split).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out / 'images' / split / f'{stem}.jpg'), image, [cv2.IMWRITE_JPEG_QUALITY, 92])
        (out / 'labels' / split / f'{stem}.txt').write_text('\n'.join(lines))
        counts[f'images {split}'] += 1
        manifest.append({'file': f'{split}/{stem}.jpg', 'source': str(path.relative_to(src))})
        if (n + 1) % 200 == 0:
            print(f'{n + 1}/{len(images)}', dict(counts), flush=True)
    meta = yaml.safe_load((src / 'data.yaml').read_text()).get('roboflow', {})
    (out / 'manifest.json').write_text(json.dumps(
        {'source': meta, 'mapping': mapping, 'aspect': args.aspect, 'dedupe_bits': args.dedupe_bits,
         'people': 'labelled + missing ones added by YOLO11m and YOLO-World-L (people.py)',
         'counts': dict(counts), 'samples': manifest}, indent=1))
    print(dict(counts))


if __name__ == '__main__':
    main()
