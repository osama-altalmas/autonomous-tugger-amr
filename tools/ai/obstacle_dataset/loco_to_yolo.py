#!/usr/bin/env python3
"""Real warehouse images for the obstacle detector: LOCO (TU Munich, CC0) to YOLO.

LOCO ("Logistics Objects in Context", github.com/tum-fml/loco) has 5,097
annotated photos from five real logistics sites: pallets, pallet trucks,
forklifts, small load carriers and stillages. Mapping to our classes:
  pallet       -> pallet
  forklift     -> forklift
  pallet_truck -> forklift   (our "forklift" class includes pallet jacks)
  small_load_carrier, stillage -> not our classes, left unlabelled
People are not annotated in LOCO, but they appear in it; left unlabelled they
would teach the model that a person is background. They are labelled here by
the COCO-trained YOLO11m (class "person", confidence >= PERSON_CONF): the same
COCO weights our model starts from, so no new knowledge leaks in.

Split by site, as LOCO's own split: subsets 2, 3, 5 for training (a share of
it for validation), subsets 1 and 4 (other sites) as the real held-out test.
Images are scaled to 640 px wide (as the model sees them) and saved as JPEG.
Boxes thinner than MIN_SIDE_PX at that scale are dropped, as in the synthetic
sets.

v4 (2026-11-18, AI_REVIEW.md): dropped boxes are greyed out in the image (an
object left in the photo without a label teaches "background"), and people are
found by YOLO11m and YOLO-World-L together (people.py; YOLO-World-L found
94-98 % of the people where our v3 found 82 %). Output loco_yolo_v4; the v3
conversion (loco_yolo) is kept.

Usage: loco_to_yolo.py --loco ai_data/external/loco --out ai_data/loco_yolo_v4
(run with the Ultralytics environment: ~/.venvs/yolo/bin/python)
"""
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from people import PersonFinder

LOCO_TO_OURS = {'pallet': 1, 'forklift': 3, 'pallet_truck': 3}
# v5 (2026-11-18): subset-4 moved to training for more real pallet trucks (a 4th site);
# subset-1 stays the unseen-site test. v3/v4 used TEST_SUBSETS = ('subset-1', 'subset-4').
TRAIN_SUBSETS, TEST_SUBSETS = ('subset-2', 'subset-3', 'subset-4', 'subset-5'), ('subset-1',)
WIDTH = 640
MIN_SIDE_PX = 10
GREY = 114
VAL_SHARE = 0.12


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--loco', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=7)
    args = ap.parse_args()
    loco, out = Path(args.loco), Path(args.out)
    data = json.loads((loco / 'loco-all-v1.json').read_text())
    names = {c['id']: c['name'] for c in data['categories']}
    boxes = defaultdict(list)
    for a in data['annotations']:
        if names[a['category_id']] in LOCO_TO_OURS:
            boxes[a['image_id']].append((LOCO_TO_OURS[names[a['category_id']]], a['bbox']))

    people = PersonFinder()
    rng = random.Random(args.seed)
    counts, kept_images, manifest = Counter(), Counter(), []
    for n, img in enumerate(sorted(data['images'], key=lambda i: i['path'])):
        path = loco / img['path'].lstrip('/')
        subset = img['path'].split('/')[2]
        split = ('test' if subset in TEST_SUBSETS else
                 'val' if rng.random() < VAL_SHARE else 'train')
        image = cv2.imread(str(path))
        if image is None:
            counts['unreadable'] += 1
            continue
        scale = WIDTH / image.shape[1]
        image = cv2.resize(image, (WIDTH, round(image.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        h = image.shape[0]
        lines, grey = [], []
        for cls, (x, y, w, bh) in boxes[img['id']]:
            x, y, w, bh = x * scale, y * scale, w * scale, bh * scale
            if min(w, bh) < MIN_SIDE_PX:
                counts['greyed thin'] += 1
                grey.append((x, y, x + w, y + bh))
                continue
            lines.append(f'{cls} {(x + w / 2) / WIDTH:.6f} {(y + bh / 2) / h:.6f} {w / WIDTH:.6f} {bh / h:.6f}')
            counts[f'{split} class {cls}'] += 1
        for x0, y0, x1, y1, _ in people.find(image):
            if min(x1 - x0, y1 - y0) >= MIN_SIDE_PX:
                lines.append(f'0 {(x0 + x1) / 2 / WIDTH:.6f} {(y0 + y1) / 2 / h:.6f} '
                             f'{(x1 - x0) / WIDTH:.6f} {(y1 - y0) / h:.6f}')
                counts[f'{split} class 0 (pseudo)'] += 1
        for x0, y0, x1, y1 in grey:
            image[max(0, int(y0)):int(np.ceil(y1)) + 1, max(0, int(x0)):int(np.ceil(x1)) + 1] = GREY
        stem = f'loco_{subset}_{img["id"]}'
        for d in ('images', 'labels'):
            (out / d / split).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out / 'images' / split / f'{stem}.jpg'), image, [cv2.IMWRITE_JPEG_QUALITY, 92])
        (out / 'labels' / split / f'{stem}.txt').write_text('\n'.join(lines))
        kept_images[split] += 1
        manifest.append({'file': f'{split}/{stem}.jpg', 'loco_path': img['path'], 'subset': subset})
        if (n + 1) % 250 == 0:
            print(f'{n + 1}/{len(data["images"])}', dict(kept_images), flush=True)
    (out / 'manifest.json').write_text(json.dumps(
        {'source': 'LOCO v1 (CC0), github.com/tum-fml/loco', 'mapping': LOCO_TO_OURS,
         'person_labels': 'YOLO11m + YOLO-World-L merged (people.py)', 'thin_boxes': 'greyed out', 'images': dict(kept_images),
         'counts': dict(counts), 'samples': manifest}, indent=1))
    print(dict(kept_images)); print(dict(counts))


if __name__ == '__main__':
    main()
