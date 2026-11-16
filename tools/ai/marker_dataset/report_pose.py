#!/usr/bin/env python3
"""Portfolio previews and charts for a v2 (pose) marker dataset.

Writes DATASET/previews/: stations.png, damage_levels.png, side_wipes.png,
unknown_and_empty.png (images with the frame outline and the 20 keypoints
drawn: green visible, red covered), counts_per_class.png,
counts_per_level.png, covered_keypoints.png and summary.json.

Usage: python3 tools/ai/marker_dataset/report_pose.py ai_data/markers_v2
"""
import collections
import json
import random
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SERIES = '#2a78d6'
TEXT, TEXT2, SURFACE, GRID = '#0b0b0b', '#52514e', '#fcfcfb', '#e4e3df'
LEVELS = ['clean', 'light', 'medium', 'heavy']


def draw(root, sample, classes, size=(400, 300)):
    image = cv2.imread(str(root / 'images' / sample['file']))
    h, w = image.shape[:2]
    text = (root / 'labels' / sample['file'].replace('.png', '.txt')).read_text().split()
    if text:
        kp = np.array(list(map(float, text[5:]))).reshape(-1, 3)
        frame = [(int(x * w), int(y * h)) for x, y, v in kp[:4] if v > 0]
        if len(frame) == 4:
            cv2.polylines(image, [np.array(frame)], True, (0, 200, 255), 2)
        for i, (x, y, v) in enumerate(kp):
            if v > 0:
                cv2.circle(image, (int(x * w), int(y * h)), 6 if i < 4 else 4,
                           (40, 190, 40) if v == 2 else (40, 40, 230), -1)
        caption = f"{classes[int(text[0])]}  {sample['level']}"
    else:
        caption = f"no object ({sample['marker']}, {sample['level']})"
    if sample.get('wipe'):
        caption += '  side wiped'
    if 'tag_covers' in sample:
        caption += '  tags ' + ' '.join(f'{c:.0%}' for c in sample['tag_covers'])
    image = cv2.resize(image, size)
    cv2.rectangle(image, (0, 0), (size[0], 20), (30, 30, 30), -1)
    cv2.putText(image, caption, (6, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1,
                cv2.LINE_AA)
    return image


def grid(images, columns):
    blank = np.full_like(images[0], 255)
    while len(images) % columns:
        images.append(blank)
    return np.vstack([np.hstack(images[i:i + columns]) for i in range(0, len(images), columns)])


def style(ax, title, xlabel, ylabel='images'):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc='left', fontsize=12, color=TEXT)
    ax.set_xlabel(xlabel, color=TEXT2)
    ax.set_ylabel(ylabel, color=TEXT2)
    ax.tick_params(colors=TEXT2)
    ax.grid(axis='y', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(GRID)


def bar_chart(path, names, values, title, xlabel):
    fig, ax = plt.subplots(figsize=(7, 3.6), dpi=150, facecolor=SURFACE)
    bars = ax.bar(names, values, color=SERIES, width=0.6, edgecolor=SURFACE, linewidth=2)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f'{value}', ha='center',
                va='bottom', fontsize=9, color=TEXT2)
    style(ax, title, xlabel)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def main():
    root = Path(sys.argv[1])
    out = root / 'previews'
    out.mkdir(exist_ok=True)
    meta = json.loads((root / 'manifest.json').read_text())
    samples, classes = meta['samples'], meta['classes']
    rng = random.Random(3)
    by = collections.defaultdict(list)
    for s in samples:
        by[(s['marker'], s['level'])].append(s)
    stations = ['HOME', 'S1', 'D2', 'S3', 'D4']
    picks = [draw(root, rng.choice(by[(m, lv)]), classes)
             for m in stations for lv in ('clean', 'medium') if by[(m, lv)]]
    cv2.imwrite(str(out / 'stations.png'), grid(picks, 4))
    station = max(stations, key=lambda m: sum(len(by[(m, lv)]) for lv in LEVELS))
    picks = []
    for level in LEVELS:
        pool = sorted(by[(station, level)], key=lambda s: s.get('plate_cover', 0))
        picks += [draw(root, s, classes)
                  for s in (pool[:1] + pool[len(pool) // 2:len(pool) // 2 + 1] + pool[-1:])]
    if picks:
        cv2.imwrite(str(out / 'damage_levels.png'), grid(picks, 3))
    wipes = [s for s in samples if s.get('wipe')]
    if wipes:
        cv2.imwrite(str(out / 'side_wipes.png'),
                    grid([draw(root, s, classes) for s in rng.sample(wipes, min(8, len(wipes)))], 4))
    unknown = [s for s in samples if s.get('label') == 0]
    empty = [s for s in samples if s['marker'] != 'NONE' and s.get('label') is None]
    floor = [s for s in samples if s['marker'] == 'NONE']
    picks = [draw(root, s, classes) for s in
             rng.sample(unknown, min(4, len(unknown))) + rng.sample(empty, min(2, len(empty))) +
             rng.sample(floor, min(2, len(floor)))]
    if picks:
        cv2.imwrite(str(out / 'unknown_and_empty.png'), grid(picks, 4))
    per_class = collections.Counter(s['label'] for s in samples if s.get('label') is not None)
    bar_chart(out / 'counts_per_class.png', classes, [per_class[i] for i in range(len(classes))],
              'Labelled markers per class', 'class (unknown = no tag readable, frame still there)')
    per_level = collections.Counter(s['level'] for s in samples)
    bar_chart(out / 'counts_per_level.png', ['clean', 'light', 'medium', 'heavy', 'empty floor'],
              [per_level[n] for n in LEVELS + ['none']], 'Images per damage level', 'damage level')
    covered = [s['keypoints_covered'] for s in samples if s.get('label') is not None]
    fig, ax = plt.subplots(figsize=(7, 3.6), dpi=150, facecolor=SURFACE)
    ax.hist(covered, bins=np.arange(-0.5, 21.5, 1), color=SERIES, edgecolor=SURFACE, linewidth=1)
    style(ax, 'Covered keypoints per labelled marker (of 20)', 'keypoints under damage')
    fig.tight_layout()
    fig.savefig(out / 'covered_keypoints.png', facecolor=SURFACE)
    plt.close(fig)
    split = collections.Counter(s['file'].split('/')[0] for s in samples)
    summary = {'images': len(samples), 'splits': dict(split),
               'markers_per_class': {classes[i]: per_class[i] for i in range(len(classes))},
               'images_per_level': {n: per_level[n] for n in LEVELS + ['none']},
               'side_wiped': len(wipes), 'marker_unrecognisable_no_object': len(empty),
               'empty_floor': len(floor), 'skipped_while_generating': meta['skipped'],
               'held_out_test': meta.get('held_out_test', False)}
    (out / 'summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
