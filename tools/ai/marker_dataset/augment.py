#!/usr/bin/env python3
"""Damaged training images from clean, labelled ones (dataset v3, step 2 of 2).

Input: a clean set from generate_clean.py (images + clean.json with each
image's plate-to-image homography). For every clean image, --variants
damaged versions are drawn on the marker's surface in the right perspective:
opaque damage (tape, worn strips, tyre marks, paint, a wiped frame side) and
dust and wear (surface_wear.py: dust film, specks, smears, faded print,
scratches), then camera effects (glare, blur). Labels follow from the same
plate geometry: a keypoint under dense damage is marked covered (v=1).
Keypoints are written in IMAGE order (image_order below): the marker looks
the same turned by 90 degrees, so the model learns "the corner nearest the
image's up-left first, then clockwise", which it can see; the tags then tell
the marker's turn (marker_identity.py; model v2 mixed up corners without it).

Every variant of one clean image goes to the same split, so the validation
set never holds a disguised copy of a training image. Runs anywhere with
numpy and OpenCV (also in Colab, next to the training).

Usage:
  python3 augment.py --clean ai_data/markers_v3_clean --out ai_data/markers_v3 --variants 4
  python3 augment.py --clean ai_data/markers_v3_clean_test --out ai_data/markers_v3_test \
      --variants 3 --test
"""
import argparse
import json
import math
import random
from pathlib import Path

import cv2
import numpy as np

import surface_wear as sw

FRAME_HALF, PLATE_HALF, TAG_SIZE, TAG_OFFSET = 0.06, 0.05, 0.04, 0.025
TAG_CENTRES = [(-TAG_OFFSET, TAG_OFFSET), (TAG_OFFSET, TAG_OFFSET),
               (TAG_OFFSET, -TAG_OFFSET), (-TAG_OFFSET, -TAG_OFFSET)]
DENSE = 0.6                     # opacity from which a point counts as covered
TAG_READABLE_COVER = 0.15
TAG_READABLE_CONTRAST = 0.20
FRAME_VISIBLE = 0.60
LEVELS = [('clean', 0.15), ('light', 0.30), ('medium', 0.30), ('heavy', 0.25)]
OPAQUE = {
    'tape': [(0.78, 0.74, 0.58), (0.85, 0.85, 0.85), (0.2, 0.35, 0.75)],
    'wear': [(0.55, 0.55, 0.52), (0.62, 0.62, 0.60)],
    'tyre': [(0.08, 0.08, 0.08), (0.15, 0.15, 0.15)],
    'paint': [(0.92, 0.92, 0.90), (0.95, 0.85, 0.10)],
    'dirt': [(0.35, 0.27, 0.18), (0.28, 0.24, 0.20)],
}
OPAQUE_TEST = {
    'tape': [(0.70, 0.20, 0.15), (0.95, 0.95, 0.70)],
    'wear': [(0.48, 0.48, 0.46)],
    'tyre': [(0.03, 0.03, 0.05)],
    'paint': [(0.10, 0.60, 0.20), (0.95, 0.50, 0.10)],
    'dirt': [(0.20, 0.18, 0.22)],
}


def image_order(keypoints):
    """Index order putting marker-ordered keypoints into image order.

    Same rule as tugger_control.marker_identity.image_order (kept here so this
    script runs alone in Colab; a unit test checks the two agree).
    """
    frame = np.asarray(keypoints, dtype=np.float64)[:4, :2]
    centre = frame.mean(axis=0)
    angles = np.arctan2(frame[:, 1] - centre[1], frame[:, 0] - centre[0])
    off = np.abs((angles - math.atan2(-1.0, -1.0) + math.pi) % (2 * math.pi) - math.pi)
    turn = int(np.argmin(off))
    order = [(turn + i) % 4 for i in range(4)]
    return order + [4 + 4 * q + (turn + j) % 4 for q in order for j in range(4)]


def to_canvas(x, y):
    return ((x + sw.CANVAS_M / 2) * sw.PX_PER_M, (sw.CANVAS_M / 2 - y) * sw.PX_PER_M)


def draw_rect(colour, alpha, rgb, centre, size, angle):
    (u, v), (w, h) = to_canvas(*centre), (size[0] * sw.PX_PER_M, size[1] * sw.PX_PER_M)
    box = cv2.boxPoints(((u, v), (w, h), math.degrees(angle))).astype(np.int32)
    cv2.fillPoly(colour, [box], tuple(float(c) for c in rgb[::-1]))
    cv2.fillPoly(alpha, [box], 1.0)


def opaque_layer(rng, level, test):
    """Solid damage on the canvas: colour (0..1 BGR), alpha, and whether a side was wiped."""
    palette = OPAQUE_TEST if test else OPAQUE
    colour = np.zeros((sw.CANVAS_PX, sw.CANVAS_PX, 3), np.float32)
    alpha = np.zeros((sw.CANVAS_PX, sw.CANVAS_PX), np.float32)
    count = {'clean': 0, 'light': rng.randint(0, 3), 'medium': rng.randint(2, 6),
             'heavy': rng.randint(5, 12)}[level]
    for _ in range(count):
        kind = rng.choice(['tape', 'tape', 'wear', 'tyre', 'paint', 'dirt', 'dirt'])
        if kind in ('wear', 'tyre'):
            size = (rng.uniform(0.08, 0.16), rng.uniform(0.006, 0.03))
            centre = (rng.uniform(-0.05, 0.05), rng.uniform(-0.05, 0.05))
            angle = rng.choice([0.0, math.pi / 2]) + rng.uniform(-0.2, 0.2)
        elif kind == 'dirt':
            size = (rng.uniform(0.004, 0.025), rng.uniform(0.004, 0.025))
            centre, angle = (rng.uniform(-0.06, 0.06), rng.uniform(-0.06, 0.06)), rng.uniform(0, 3)
        elif kind == 'paint':
            size = (rng.uniform(0.01, 0.06), rng.uniform(0.003, 0.02))
            centre, angle = (rng.uniform(-0.06, 0.06), rng.uniform(-0.06, 0.06)), rng.uniform(0, 3)
        else:
            size = (rng.uniform(0.02, 0.08), rng.uniform(0.015, 0.05))
            centre, angle = (rng.uniform(-0.06, 0.06), rng.uniform(-0.06, 0.06)), rng.uniform(0, 3)
        if test:
            size = (size[0] * 1.2, size[1] * 0.9)
        draw_rect(colour, alpha, rng.choice(palette[kind]), centre, size, angle)
    wiped = level != 'clean' and rng.random() < 0.2
    if wiped:
        side = rng.randrange(4)
        offset = FRAME_HALF - 0.004 + rng.uniform(-0.004, 0.004)
        centre = [(0, offset), (offset, 0), (0, -offset), (-offset, 0)][side]
        draw_rect(colour, alpha, rng.choice(palette['wear'] + palette['dirt']), centre,
                  (0.15, rng.uniform(0.015, 0.03)), 0.0 if side in (0, 2) else math.pi / 2)
    return colour, alpha, wiped


TAG_SWAP_SHARE = 0.7            # variants whose four tags get random codes


def swap_tags(image, homography, rng):
    """Paint four random AprilTag 36h11 codes into the tag positions.

    The model must learn "four tags in a red frame", not the codes of the
    stations it was trained on: a new station brings new codes. The corners
    do not move, so the labels stay exact. Returns the new image and codes.
    """
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    out = image.copy()
    h, w = image.shape[:2]
    codes = []
    # Brightness of the printed black and white as the camera sees them here.
    plate = np.asarray(homography)
    for cx, cy in TAG_CENTRES:
        code = rng.randrange(len(dictionary.bytesList))
        codes.append(code)
        tag = cv2.aruco.generateImageMarker(dictionary, code, 80)
        src = np.float32([[0, 0], [80, 0], [80, 80], [0, 80]])
        corners = np.float32([[cx - TAG_SIZE / 2, cy + TAG_SIZE / 2],
                              [cx + TAG_SIZE / 2, cy + TAG_SIZE / 2],
                              [cx + TAG_SIZE / 2, cy - TAG_SIZE / 2],
                              [cx - TAG_SIZE / 2, cy - TAG_SIZE / 2]])
        dst = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), plate).reshape(4, 2)
        warp = cv2.getPerspectiveTransform(src, dst.astype(np.float32))
        mask = cv2.warpPerspective(np.full((80, 80), 255, np.uint8), warp, (w, h)) > 127
        if not mask.any():
            continue
        # Sample the current dark and light print levels inside this tag.
        values = out[mask].mean(axis=1)
        dark, light = np.percentile(values, 10), np.percentile(values, 90)
        drawn = cv2.warpPerspective(tag, warp, (w, h), flags=cv2.INTER_LINEAR).astype(np.float32)
        level = dark + (light - dark) * drawn / 255.0
        # The plate's own tint (light cells), so the new print matches the old.
        pixels = out[mask].astype(np.float32)
        bright = pixels[values >= (dark + light) / 2]
        tint = bright.mean(axis=0) / max(float(bright.mean()), 1.0) if len(bright) else 1.0
        # Camera noise as in the rendered image (gaussian, stddev 0.007 of full scale).
        noise = np.random.default_rng(rng.randrange(1 << 30)).normal(0, 1.8, (int(mask.sum()), 1))
        out[mask] = np.clip(level[mask][:, None] * tint + noise, 0, 255).astype(np.uint8)
    return out, codes


def region(alpha, cx, cy, half):
    (u0, v0), (u1, v1) = to_canvas(cx - half, cy + half), to_canvas(cx + half, cy - half)
    return alpha[int(v0):int(v1), int(u0):int(u1)]


def led_lighting(out, rng, kinds):
    """How the camera's own LED ring can differ from the captured one.

    The clean images are lit by the ring inside the light shield; here the
    ring ages or loses an LED, the exposure differs, and a glossy marker
    reflects the LEDs.  Applied to every variant, clean ones included:
    light changes independently of wear.
    """
    h, w = out.shape[:2]
    gain = rng.uniform(0.7, 1.2)
    if rng.random() < 0.10:                     # weak ring or short exposure
        gain = rng.uniform(0.3, 0.6)
        kinds.append('dim LEDs')
    field = np.full((h, w), gain, np.float32)
    if rng.random() < 0.15:                     # one LED out: darker towards its side
        angle = rng.uniform(0, 2 * math.pi)
        yy, xx = np.mgrid[0:h, 0:w]
        along = ((xx - w / 2) * math.cos(angle) + (yy - h / 2) * math.sin(angle)) / (w / 2)
        field *= np.clip(1 - rng.uniform(0.2, 0.45) * (1 + along) / 2, 0.3, 1).astype(np.float32)
        kinds.append('LED out')
    out = out * field[..., None]
    if rng.random() < 0.2:                      # LED reflections on a glossy marker
        yy, xx = np.mgrid[0:h, 0:w]
        for _ in range(rng.randint(1, 4)):
            gx, gy = w / 2 + rng.uniform(-90, 90), h / 2 + rng.uniform(-70, 70)
            r = rng.uniform(6, 25)
            out = out + np.exp(-((xx - gx) ** 2 + (yy - gy) ** 2) / (2 * r * r))[..., None] * \
                rng.uniform(60, 200)
        kinds.append('LED reflection')
    if gain < 0.9:                              # more sensor noise when the image is darker
        noise = np.random.default_rng(rng.randrange(1 << 30)).normal(0, 2.5 / gain, out.shape)
        out = out + noise.astype(np.float32)
    return out


def augment(image, homography, keypoints, rng, test=False, level=None):
    """One damaged variant: (image, visibility per keypoint, info)."""
    level = level or rng.choices([lv for lv, _ in LEVELS], [w for _, w in LEVELS])[0]
    codes = None
    if rng.random() < TAG_SWAP_SHARE:
        image, codes = swap_tags(image, homography, rng)
    strength = {'clean': 0.0, 'light': rng.uniform(0.2, 0.5), 'medium': rng.uniform(0.4, 0.8),
                'heavy': rng.uniform(0.7, 1.0)}[level]
    o_colour, o_alpha, wiped = opaque_layer(rng, level, test)
    nprng = np.random.default_rng(rng.randrange(1 << 30))
    if strength > 0:
        w_colour, w_alpha, fade, kinds = sw.wear_layers(nprng, strength, test)
    else:
        w_colour = np.zeros_like(o_colour)
        w_alpha = np.zeros_like(o_alpha)
        fade, kinds = 0.0, []
    # Opaque damage lies on top of dust.
    alpha = o_alpha + w_alpha * (1 - o_alpha)
    colour = (o_colour * o_alpha[..., None] + w_colour * (w_alpha * (1 - o_alpha))[..., None]) \
        / np.maximum(alpha, 1e-6)[..., None]
    h, w = image.shape[:2]
    to_image = np.asarray(homography) @ sw.canvas_to_plate()
    out = image.astype(np.float32)
    if fade:
        plate = np.zeros((sw.CANVAS_PX, sw.CANVAS_PX), np.float32)
        a = int((sw.CANVAS_M / 2 - PLATE_HALF) * sw.PX_PER_M)
        plate[a:-a, a:-a] = 1.0
        plate = cv2.warpPerspective(plate, to_image, (w, h))
        dark = (out.mean(axis=2) < 110)[..., None]
        out = out + (plate * fade)[..., None] * (215.0 - out) * dark
        kinds.append('faded print')
    a_img = cv2.warpPerspective(alpha, to_image, (w, h))
    c_img = cv2.warpPerspective(colour * 255.0, to_image, (w, h))
    out = out * (1 - a_img[..., None]) + c_img * a_img[..., None]
    if level != 'clean' and rng.random() < 0.3:
        gx, gy, r = rng.uniform(0, w), rng.uniform(0, h), rng.uniform(40, 140)
        yy, xx = np.mgrid[0:h, 0:w]
        out = out + np.exp(-((xx - gx) ** 2 + (yy - gy) ** 2) / (2 * r * r))[..., None] * \
            rng.uniform(40, 120)
        kinds.append('glare')
    if level != 'clean' and rng.random() < 0.3:
        k = rng.randint(2, 5) * 2 + 1
        out = cv2.GaussianBlur(out, (k, k), 0)
        kinds.append('blur')
    out = led_lighting(out, rng, kinds)
    out = np.clip(out, 0, 255).astype(np.uint8)
    # Labels from the plate geometry.
    plate_points = np.vstack([[[-FRAME_HALF, FRAME_HALF], [FRAME_HALF, FRAME_HALF],
                               [FRAME_HALF, -FRAME_HALF], [-FRAME_HALF, -FRAME_HALF]]] +
                             [[[cx - TAG_SIZE / 2, cy + TAG_SIZE / 2],
                               [cx + TAG_SIZE / 2, cy + TAG_SIZE / 2],
                               [cx + TAG_SIZE / 2, cy - TAG_SIZE / 2],
                               [cx - TAG_SIZE / 2, cy - TAG_SIZE / 2]] for cx, cy in TAG_CENTRES])
    covered = []
    for x, y in plate_points:
        u, v = to_canvas(x, y)
        covered.append(alpha[min(int(v), sw.CANVAS_PX - 1), min(int(u), sw.CANVAS_PX - 1)] >= DENSE)
    tags = []
    for cx, cy in TAG_CENTRES:
        a_tag = region(alpha, cx, cy, TAG_SIZE / 2)
        tags.append({'covered': round(float(np.mean(a_tag >= DENSE)), 3),
                     'contrast_left': round(float(np.mean(1 - a_tag)) * (1 - fade), 3)})
    ring = region(alpha, 0, 0, FRAME_HALF)
    inner = int((FRAME_HALF - PLATE_HALF) * sw.PX_PER_M)
    ring_mask = np.ones_like(ring, bool)
    ring_mask[inner:-inner, inner:-inner] = False
    ring_cover = float(np.mean(ring[ring_mask] >= DENSE))
    readable = [t['covered'] <= TAG_READABLE_COVER and t['contrast_left'] >= TAG_READABLE_CONTRAST
                for t in tags]
    present = ring_cover <= FRAME_VISIBLE or any(readable)
    info = {'level': level, 'strength': round(strength, 2), 'wiped': wiped, 'kinds': kinds,
            'tag_codes': codes,
            'fade': round(fade, 3), 'frame_cover': round(ring_cover, 3), 'tags': tags,
            'tags_readable': int(sum(readable)), 'marker_present': bool(present)}
    return out, covered, info


def yolo_line(keypoints, covered, width, height):
    kp = np.asarray(keypoints, dtype=np.float64)
    corners = kp[:4]
    x0, y0 = np.clip(corners.min(axis=0), 0, [width - 1, height - 1])
    x1, y1 = np.clip(corners.max(axis=0), 0, [width - 1, height - 1])
    fields = [0, (x0 + x1) / 2 / width, (y0 + y1) / 2 / height, (x1 - x0) / width,
              (y1 - y0) / height]
    for (x, y), hidden in zip(kp, covered):
        inside = 0 <= x < width and 0 <= y < height
        fields += [x / width if inside else 0.0, y / height if inside else 0.0,
                   0 if not inside else (1 if hidden else 2)]
    return ' '.join(f'{v:.6f}' if isinstance(v, float) else str(v) for v in fields) + '\n'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--variants', type=int, default=4)
    parser.add_argument('--seed', type=int, default=5)
    parser.add_argument('--val-share', type=float, default=0.15)
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()
    rng = random.Random(args.seed)
    clean_dir, out = Path(args.clean), Path(args.out)
    meta = json.loads((clean_dir / 'clean.json').read_text())
    splits = ('test',) if args.test else ('train', 'val')
    for split in splits:
        (out / 'images' / split).mkdir(parents=True, exist_ok=True)
        (out / 'labels' / split).mkdir(parents=True, exist_ok=True)
    manifest = []
    for sample in meta['samples']:
        split = 'test' if args.test else ('val' if rng.random() < args.val_share else 'train')
        image = cv2.imread(str(clean_dir / 'images' / sample['file']))
        h, w = image.shape[:2]
        for k in range(args.variants if sample['marker'] != 'NONE' else 1):
            stem = f"{Path(sample['file']).stem}_v{k}"
            if sample['marker'] == 'NONE':
                result, label, info = image, '', {'level': 'none'}
            else:
                result, covered, info = augment(image, sample['homography'], sample['keypoints'],
                                                rng, args.test)
                order = image_order(sample['keypoints'])
                label = (yolo_line(np.asarray(sample['keypoints'])[order],
                                   [covered[i] for i in order], w, h)
                         if info['marker_present'] else '')
                info['keypoint_order'] = 'image'
            cv2.imwrite(str(out / 'images' / split / f'{stem}.png'), result)
            (out / 'labels' / split / f'{stem}.txt').write_text(label)
            manifest.append(dict(info, file=f'{split}/{stem}.png', marker=sample['marker'],
                                 clean=sample['file'], labelled=bool(label)))
    (out / 'manifest.json').write_text(json.dumps(
        {'samples': manifest, 'clean_set': str(clean_dir), 'variants': args.variants,
         'held_out_test': args.test, 'classes': ['marker'], 'keypoints': meta['keypoints']},
        indent=1))
    if not args.test:
        (out / 'data.yaml').write_text(
            'path: .\ntrain: images/train\nval: images/val\nkpt_shape: [20, 3]\n'
            'names:\n  0: marker\n')
    print(f'{len(manifest)} images in {out}')


if __name__ == '__main__':
    main()
