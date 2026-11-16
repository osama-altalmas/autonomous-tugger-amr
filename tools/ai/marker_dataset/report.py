#!/usr/bin/env python3
"""Portfolio previews and charts for a generated marker dataset.

Writes into DATASET/previews/:
  stations.png        one damaged and one clean example per station, labels drawn
  damage_levels.png   the same station at each damage level, labels drawn
  unlabelled.png      examples the model must NOT recognise (heavily covered, empty floor)
  counts_per_class.png, counts_per_level.png, qr_coverage.png   dataset make-up
and prints a summary table (also saved as previews/summary.json).

Usage: python3 tools/ai/marker_dataset/report.py ai_data/markers_v1
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
BOX = {0: (52, 104, 235), 'qr': (214, 120, 42)}         # BGR: frame orange-red, QR blue
LEVELS = ['clean', 'light', 'medium', 'heavy']


def load(root):
    manifest = json.loads((root / 'manifest.json').read_text())
    return manifest, manifest['classes']


def draw(root, sample, classes, size=(320, 240)):
    image = cv2.imread(str(root / 'images' / sample['file']))
    h, w = image.shape[:2]
    label_file = root / 'labels' / sample['file'].replace('.png', '.txt')
    for line in label_file.read_text().splitlines():
        c, x, y, bw, bh = (float(v) for v in line.split())
        p0 = (int((x - bw / 2) * w), int((y - bh / 2) * h))
        p1 = (int((x + bw / 2) * w), int((y + bh / 2) * h))
        colour = BOX[0] if c == 0 else BOX['qr']
        cv2.rectangle(image, p0, p1, colour, 3)
        cv2.putText(image, classes[int(c)], (p0[0], max(p0[1] - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2, cv2.LINE_AA)
    image = cv2.resize(image, size)
    caption = (f"{sample['marker']}  {sample['level']}  QR {sample['qr_cover']:.0%} "
               f"frame {sample['frame_cover']:.0%}")
    cv2.rectangle(image, (0, 0), (size[0], 20), (30, 30, 30), -1)
    cv2.putText(image, caption, (6, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1,
                cv2.LINE_AA)
    return image


def grid(images, columns):
    blank = np.full_like(images[0], 255)
    while len(images) % columns:
        images.append(blank)
    rows = [np.hstack(images[i:i + columns]) for i in range(0, len(images), columns)]
    return np.vstack(rows)


def bar_chart(path, names, values, title, xlabel):
    fig, ax = plt.subplots(figsize=(7, 3.6), dpi=150, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    bars = ax.bar(names, values, color=SERIES, width=0.6, edgecolor=SURFACE, linewidth=2)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f'{value}', ha='center',
                va='bottom', fontsize=9, color=TEXT2)
    ax.set_title(title, loc='left', fontsize=12, color=TEXT)
    ax.set_xlabel(xlabel, color=TEXT2)
    ax.set_ylabel('images', color=TEXT2)
    ax.tick_params(colors=TEXT2)
    ax.grid(axis='y', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(GRID)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def coverage_chart(path, covers, low, high):
    fig, ax = plt.subplots(figsize=(7, 3.6), dpi=150, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.hist(covers, bins=np.linspace(0, 1, 41), color=SERIES, edgecolor=SURFACE, linewidth=1)
    top = ax.get_ylim()[1]
    for x, text in ((low, f'labelled up to {low:.0%}'), (high, f'unrecognisable from {high:.0%}')):
        ax.axvline(x, color=TEXT2, linewidth=1, linestyle='--')
        ax.text(x + 0.01, top * 0.92, text, fontsize=8, color=TEXT2)
    ax.set_title('How much of the QR is covered, per image', loc='left', fontsize=12,
                 color=TEXT)
    ax.set_xlabel('share of the QR hidden by damage', color=TEXT2)
    ax.set_ylabel('images', color=TEXT2)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.tick_params(colors=TEXT2)
    ax.grid(axis='y', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(GRID)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def main():
    root = Path(sys.argv[1])
    out = root / 'previews'
    out.mkdir(exist_ok=True)
    manifest, classes = load(root)
    samples = manifest['samples']
    rng = random.Random(3)
    by = collections.defaultdict(list)
    for s in samples:
        by[(s['marker'], s['level'])].append(s)
    stations = [m for m in ('HOME', 'S1', 'D2', 'S3', 'D4')]
    picks = []
    for m in stations:
        for level in ('clean', 'medium'):
            if by[(m, level)]:
                picks.append(draw(root, rng.choice(by[(m, level)]), classes))
    cv2.imwrite(str(out / 'stations.png'), grid(picks, 4))
    station = max(stations, key=lambda m: sum(len(by[(m, lv)]) for lv in LEVELS))
    picks = []
    for level in LEVELS:
        pool = sorted(by[(station, level)], key=lambda s: s['qr_cover'])
        for s in (pool[:1] + pool[len(pool) // 2:len(pool) // 2 + 1] + pool[-1:]) if pool else []:
            picks.append(draw(root, s, classes))
    cv2.imwrite(str(out / 'damage_levels.png'), grid(picks, 3))
    gone = [s for s in samples if s['marker'] != 'NONE' and not s['labels']]
    floor = [s for s in samples if s['marker'] == 'NONE']
    picks = [draw(root, s, classes) for s in rng.sample(gone, min(6, len(gone)))]
    picks += [draw(root, s, classes) for s in rng.sample(floor, min(2, len(floor)))]
    if picks:
        cv2.imwrite(str(out / 'unlabelled.png'), grid(picks, 4))
    per_class = collections.Counter(c for s in samples for c in s['labels'])
    bar_chart(out / 'counts_per_class.png', classes, [per_class[i] for i in range(len(classes))],
              'Labelled objects per class', 'class (red frame, then QR identity)')
    per_level = collections.Counter(s['level'] for s in samples)
    names = LEVELS + ['none']
    bar_chart(out / 'counts_per_level.png', ['clean', 'light', 'medium', 'heavy', 'empty floor'],
              [per_level[n] for n in names], 'Images per damage level', 'damage level')
    coverage_chart(out / 'qr_coverage.png', [s['qr_cover'] for s in samples if s['marker'] != 'NONE'],
                   manifest['label_max_cover'], manifest['unlabel_min_cover'])
    split = collections.Counter(s['file'].split('/')[0] for s in samples)
    summary = {'images': len(samples), 'train': split['train'], 'val': split['val'],
               'objects_per_class': {classes[i]: per_class[i] for i in range(len(classes))},
               'images_per_level': {n: per_level[n] for n in names},
               'unrecognisable_marker_images': len(gone), 'empty_floor_images': len(floor),
               'skipped_while_generating': manifest['skipped']}
    (out / 'summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
