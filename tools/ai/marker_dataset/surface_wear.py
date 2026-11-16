"""Dust and wear painted onto the floor marker in a camera image (dataset v3).

Real markers rarely lose whole pieces; they get dusty, smeared, faded and
scratched. These effects are drawn on the marker's own surface (plate
coordinates, metres, x right / y up as the marker texture) and warped into
the image with the plate-to-image homography the generator already knows,
so they sit on the floor in the right perspective. Camera effects (glare,
blur, noise) are applied to the whole image afterwards.

apply_wear(image, plate_to_image, rng, test) -> (image, info): info gives,
for each tag quarter, how much of its contrast is left (1 = untouched).
"""
import math

import cv2
import numpy as np

CANVAS_M = 0.30                 # +-15 cm: the marker and the floor around it
FADE_FROM_M = 0.08              # dust thins out from here to the canvas edge (no hard border)
CANVAS_PX = 640
PX_PER_M = CANVAS_PX / CANVAS_M
DUST = [(0.62, 0.60, 0.55), (0.55, 0.50, 0.42), (0.70, 0.68, 0.64), (0.40, 0.36, 0.30)]
DUST_TEST = [(0.75, 0.72, 0.66), (0.48, 0.44, 0.40), (0.58, 0.55, 0.60)]


def canvas_to_plate():
    """3 x 3 matrix: canvas pixel (u, v) -> plate (x, y) in metres."""
    s = 1.0 / PX_PER_M
    return np.array([[s, 0, -CANVAS_M / 2], [0, -s, CANVAS_M / 2], [0, 0, 1]])


def _smooth_field(rng, scale_px, size=CANVAS_PX):
    """Smooth random field in [0, 1] (cloud-like patches)."""
    small = max(2, size // scale_px)
    field = rng.random((small, small)).astype(np.float32)
    field = cv2.resize(field, (size, size), interpolation=cv2.INTER_CUBIC)
    field = cv2.GaussianBlur(field, (0, 0), scale_px / 3)
    field -= field.min()
    return field / max(float(field.max()), 1e-6)


def wear_layers(rng, strength, test=False):
    """Colour (BGR float) and opacity layers on the canvas, and the print fade.

    strength 0..1 scales every effect; returns (colour, alpha, fade, kinds).
    """
    colours = DUST_TEST if test else DUST
    colour = np.zeros((CANVAS_PX, CANVAS_PX, 3), np.float32)
    alpha = np.zeros((CANVAS_PX, CANVAS_PX), np.float32)
    kinds = []

    def lay(mask, rgb):
        nonlocal colour, alpha
        bgr = np.array(rgb[::-1], np.float32)
        new = mask * (1 - alpha)
        colour = colour + new[..., None] * bgr
        alpha = alpha + new

    if rng.random() < 0.8:                      # dust film, patchy
        field = _smooth_field(rng, rng.integers(40, 160))
        cover = np.clip((field - rng.uniform(0.0, 0.6)) * 3, 0, 1)
        lay(cover * rng.uniform(0.2, 0.75) * strength, colours[rng.integers(len(colours))])
        kinds.append('dust film')
    if rng.random() < 0.7:                      # fine specks and grit
        mask = np.zeros((CANVAS_PX, CANVAS_PX), np.float32)
        for _ in range(int(rng.uniform(700, 10000) * strength)):     # same density on the larger canvas
            u, v = rng.integers(0, CANVAS_PX, 2)
            cv2.circle(mask, (int(u), int(v)), int(rng.integers(1, 4)), float(rng.uniform(0.5, 1)), -1)
        lay(np.clip(mask, 0, 1), colours[rng.integers(len(colours))])
        kinds.append('specks')
    if rng.random() < 0.5:                      # smears and footprints
        mask = np.zeros((CANVAS_PX, CANVAS_PX), np.float32)
        for _ in range(rng.integers(1, 4)):
            centre = tuple(int(c) for c in rng.integers(100, CANVAS_PX - 100, 2))
            axes = (int(rng.uniform(40, 220)), int(rng.uniform(15, 70)))
            cv2.ellipse(mask, centre, axes, float(rng.uniform(0, 180)), 0, 360,
                        float(rng.uniform(0.3, 0.8)), -1)
        mask = cv2.GaussianBlur(mask, (0, 0), 12)
        lay(mask * strength, colours[rng.integers(len(colours))])
        kinds.append('smears')
    if rng.random() < 0.4:                      # scratches: thin light lines
        mask = np.zeros((CANVAS_PX, CANVAS_PX), np.float32)
        for _ in range(rng.integers(2, 12)):
            p = rng.integers(0, CANVAS_PX, 2)
            angle, length = rng.uniform(0, math.pi), rng.uniform(80, 500)
            q = p + length * np.array([math.cos(angle), math.sin(angle)])
            cv2.line(mask, tuple(int(c) for c in p), tuple(int(c) for c in q),
                     float(rng.uniform(0.4, 0.9)), int(rng.integers(1, 3)))
        lay(mask * strength, (0.85, 0.85, 0.82))
        kinds.append('scratches')
    fade = float(rng.uniform(0.0, 0.55) * strength) if rng.random() < 0.4 else 0.0
    colour = colour / np.maximum(alpha, 1e-6)[..., None]
    # Dust thins out smoothly towards the canvas edge: no square of haze
    # around the marker (a shortcut a model could learn).
    # Its outline is random (centre, size, irregular edge), never the marker's.
    axis = (np.arange(CANVAS_PX) + 0.5) / PX_PER_M - CANVAS_M / 2
    xx, yy = np.meshgrid(axis, axis)
    cx, cy = rng.uniform(-0.05, 0.05, 2)
    reach = rng.uniform(FADE_FROM_M, CANVAS_M / 2 - 0.01)
    wobble = (_smooth_field(rng, 120) - 0.5) * 0.06
    r = np.hypot(xx - cx, yy - cy) + wobble
    ramp = np.clip((reach - r) / 0.05, 0, 1)
    edge = np.clip((CANVAS_M / 2 - np.maximum(np.abs(xx), np.abs(yy))) / 0.02, 0, 1)
    alpha = alpha * ramp * ramp * (3 - 2 * ramp) * edge
    return colour, np.clip(alpha, 0, 1), fade, kinds


def tag_contrast_left(alpha, fade, tag_centres_m, tag_size_m):
    """Per tag: share of its black/white contrast still visible (1 = untouched)."""
    left = []
    for cx, cy in tag_centres_m:
        u0 = int((cx - tag_size_m / 2 + CANVAS_M / 2) * PX_PER_M)
        v0 = int((CANVAS_M / 2 - cy - tag_size_m / 2) * PX_PER_M)
        n = int(tag_size_m * PX_PER_M)
        region = alpha[v0:v0 + n, u0:u0 + n]
        left.append(float(np.mean(1.0 - region)) * (1.0 - fade))
    return left


def apply_wear(image, plate_to_image, rng, tag_centres_m, tag_size_m, plate_half_m=0.05,
               strength=1.0, test=False):
    """Paint dust and wear on the marker and add camera effects."""
    colour, alpha, fade, kinds = wear_layers(rng, strength, test)
    h, w = image.shape[:2]
    to_image = plate_to_image @ canvas_to_plate()
    out = image.astype(np.float32)
    if fade:
        # The printed black fades towards the white plate: only on the plate.
        plate = np.zeros((CANVAS_PX, CANVAS_PX), np.float32)
        a = int((CANVAS_M / 2 - plate_half_m) * PX_PER_M)
        plate[a:CANVAS_PX - a, a:CANVAS_PX - a] = 1.0
        plate = cv2.warpPerspective(plate, to_image, (w, h))
        white = np.percentile(out[plate > 0.5], 95, axis=0) if (plate > 0.5).any() else 255.0
        out = out + (plate * fade)[..., None] * (white - out) * (out.mean(axis=2, keepdims=True) < white.mean() * 0.6)
    a_img = cv2.warpPerspective(alpha, to_image, (w, h))
    c_img = cv2.warpPerspective(colour * 255.0, to_image, (w, h))
    out = out * (1 - a_img[..., None]) + c_img * a_img[..., None]
    if rng.random() < 0.3:                      # glare from a ceiling light
        gx, gy = rng.uniform(0, w), rng.uniform(0, h)
        yy, xx = np.mgrid[0:h, 0:w]
        spot = np.exp(-((xx - gx) ** 2 + (yy - gy) ** 2) / (2 * rng.uniform(40, 140) ** 2))
        out = out + spot[..., None] * rng.uniform(40, 120)
        kinds.append('glare')
    if rng.random() < 0.3:                      # slight motion or focus blur
        k = int(rng.integers(2, 6)) * 2 + 1
        if rng.random() < 0.5:
            kernel = np.zeros((k, k), np.float32)
            kernel[k // 2, :] = 1.0 / k
            rot = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), float(rng.uniform(0, 180)), 1)
            out = cv2.filter2D(out, -1, cv2.warpAffine(kernel, rot, (k, k)))
        else:
            out = cv2.GaussianBlur(out, (k, k), 0)
        kinds.append('blur')
    out = np.clip(out, 0, 255).astype(np.uint8)
    info = {'kinds': kinds, 'fade': round(fade, 3),
            'tag_contrast_left': [round(v, 3) for v in
                                  tag_contrast_left(alpha, fade, tag_centres_m, tag_size_m)]}
    return out, info
