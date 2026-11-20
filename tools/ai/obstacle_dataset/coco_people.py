#!/usr/bin/env python3
"""Real people from COCO for the obstacle model (v4).

Why: v3 found 82 % of near people in real photos, a COCO-trained detector
98 %. Fine-tuning replaced the COCO person head, so part of what COCO taught
was lost; keeping some COCO people in training ("rehearsal") holds it.

What: train2017 images with 1-6 people (COCO labels every person; crowd
regions are greyed out), at least one person 25 % of the image high or more,
and only images whose own licence allows any use (CC BY, CC BY-SA, no known
copyright, US government work); annotations are CC BY 4.0. Images are fetched
one by one from images.cocodataset.org, scaled to 640 px wide, JPEG 92. Only
the person class is labelled; COCO has no pallets, carts or forklifts.

Usage (from ai_data): coco_people.py --ann external/coco/annotations/instances_train2017.json \
    --out coco_people --count 3000
"""
import argparse
import json
import random
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

OPEN_LICENCES = {4, 5, 7, 8}     # CC BY, CC BY-SA, no known copyright, US government
WIDTH, MIN_SIDE_PX, GREY, VAL_SHARE = 640, 10, 114, 0.10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ann', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--count', type=int, default=3000)
    ap.add_argument('--seed', type=int, default=7)
    args = ap.parse_args()
    data = json.loads(Path(args.ann).read_text())
    licences = {l['id']: l['name'] for l in data['licenses']}
    people = defaultdict(list)
    for a in data['annotations']:
        if a['category_id'] == 1:
            people[a['image_id']].append(a)
    pick = []
    for img in data['images']:
        anns = people.get(img['id'])
        if not anns or img['license'] not in OPEN_LICENCES:
            continue
        solo = [a for a in anns if not a['iscrowd']]
        if not 1 <= len(solo) <= 6 or max(a['bbox'][3] for a in solo) < 0.25 * img['height']:
            continue
        pick.append(img)
    rng = random.Random(args.seed)
    rng.shuffle(pick)
    pick = pick[:args.count]
    out = Path(args.out)
    for s in ('train', 'val'):
        for d in ('images', 'labels'):
            (out / d / s).mkdir(parents=True, exist_ok=True)

    def one(img):
        split = 'val' if random.Random(img['id']).random() < VAL_SHARE else 'train'
        target = out / 'images' / split / f'coco_{img["id"]}.jpg'
        if target.exists():
            return split, img
        raw = urllib.request.urlopen(img['coco_url'], timeout=30).read()
        image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        s = WIDTH / image.shape[1]
        image = cv2.resize(image, (WIDTH, round(image.shape[0] * s)), interpolation=cv2.INTER_AREA)
        h = image.shape[0]
        lines = []
        for a in people[img['id']]:
            x, y, w, bh = (v * s for v in a['bbox'])
            if a['iscrowd'] or min(w, bh) < MIN_SIDE_PX:
                image[int(y):int(np.ceil(y + bh)) + 1, int(x):int(np.ceil(x + w)) + 1] = GREY
                continue
            # COCO boxes can end a hair past the image edge after scaling; clip them.
            x0, y0 = max(0.0, x), max(0.0, y)
            x1, y1 = min(WIDTH, x + w), min(h, y + bh)
            lines.append(f'0 {(x0 + x1) / 2 / WIDTH:.6f} {(y0 + y1) / 2 / h:.6f} '
                         f'{(x1 - x0) / WIDTH:.6f} {(y1 - y0) / h:.6f}')
        cv2.imwrite(str(target), image, [cv2.IMWRITE_JPEG_QUALITY, 92])
        (out / 'labels' / split / f'coco_{img["id"]}.txt').write_text('\n'.join(lines))
        return split, img

    done = []
    with ThreadPoolExecutor(16) as pool:
        for k, (split, img) in enumerate(pool.map(one, pick)):
            done.append({'file': f'{split}/coco_{img["id"]}.jpg', 'coco_url': img['coco_url'],
                         'flickr_url': img.get('flickr_url'), 'licence': licences[img['license']]})
            if (k + 1) % 500 == 0:
                print(f'{k + 1}/{len(pick)}', flush=True)
    (out / 'manifest.json').write_text(json.dumps(
        {'source': 'COCO train2017 (annotations CC BY 4.0), images with open licences only',
         'eligible_images': len(people), 'picked': len(done), 'samples': done}, indent=1))
    print(len(done), 'images')


if __name__ == '__main__':
    main()
