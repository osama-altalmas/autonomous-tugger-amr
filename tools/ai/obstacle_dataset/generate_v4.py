#!/usr/bin/env python3
# FROZEN copy of the v4 generator (dataset obstacles_v4, 2026-11-18). Do not edit; v5+ is generate.py.
"""Obstacle dataset from Gazebo: people, pallets, carts, forklifts (YOLO boxes).

Runs Gazebo on its own (no robot stack) with the real warehouse world, a
camera rig matching the robot's colour recognition camera (640 x 480, 87
degrees, 0.195 m above the floor, looking ahead) and a bounding-box camera
at the same place, which gives the visible 2D box of every labelled object
(occlusion included): no hand labelling. A pool of Gazebo Fuel models
(download_models.sh) is parked outside the building; for every image the rig
is put on a random free spot of the warehouse map and 0-4 objects are put at
random free spots in front of it (0.6-8 m), at random headings, under random
light. Images with no object (racks, walls, station legs and stands only)
teach the model what is not an obstacle.

Held-out test (--test): other people, pallet and cart models, another seed.

v2 (training only, after v1 failed on an unseen wooden pallet): pallets are also
built from parts (block, stringer or plastic pallets of random size, colour and
load), carts, forklifts and the grey Euro pallet also come as recoloured copies,
unlabelled distractors (boxes, cones, barrels, bins, crates) teach what is not
an obstacle class, and the world is rebuilt every count/worlds images with new
pallets and colours. --test keeps the v1 behaviour, so the v1 held-out test set
stays the benchmark.

v3 (2026-11-17, after a data audit and a literature check; v2 frozen in
generate_v2.py): pallets get wood-grain or plastic textures; forklifts and
pallet jacks are also built from parts (the Fuel Forklift and Pallet_Jack are
now test-only, so the forklift score is a real held-out score); some worlds
replace the red rack blocks by pallet racking with loaded pallets on it (rack
pallets are labelled, as every pallet must be); floor, walls, columns and
blocks change colour per world (the first world keeps the original look the
robot's simulation uses); the camera height, pitch and roll vary a little;
images get camera effects (motion blur, defocus, noise, exposure, white
balance, JPEG); boxes thinner than 10 px are dropped (fragments the camera
barely sees); more people (one lying on the floor) and two hand trucks.
The v3 test (--test) uses the held-out models in two worlds (original blocks,
and racking that carries only cartons, so every test pallet is a held-out one).

v4 (2026-11-18, after the v3 review in ai_data/AI_REVIEW.md; v3 frozen in
generate_v3.py): carts built from parts (platform carts with thin push
handles, roll cages - which count as carts - and shelf trolleys), reach
trucks, walk-behind stackers and order pickers built from parts, loads on the
forks of every truck kind; labelled objects too thin to box are greyed out
instead of left in the image unlabelled. The test also gets built roll cages,
a platform cart, reach trucks and stackers (unseen instances).

Usage:
  generate.py --out DIR --count N [--test] [--seed S] [--worlds K] [--preview]
"""
import argparse
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

WS = Path(__file__).resolve().parents[3]
WORLD_FILE = WS / 'src/tugger_gazebo/worlds/tugger_warehouse.sdf'
MAP_YAML = WS / 'src/tugger_navigation/maps/warehouse_map.yaml'
WORLD = 'tugger_test'
FUEL = 'https://fuel.gazebosim.org/1.0'

CLASSES = ['person', 'pallet', 'cart', 'forklift']
CAMERA_HEIGHT_M = 0.195        # base_footprint -> base_link 0.100 + camera 0.095
HFOV = 1.5184
WIDTH, HEIGHT = 640, 480
MIN_BOX_PX = 10               # v3: thinner boxes are fragments the camera barely sees
PARK_Y = 120.0                 # pool waits here, far outside the 12 m camera range

# Animated actors (walking people) are not used: Gazebo's set_pose does not
# move them, so they never appeared where placed (2026-11-16).
# (Fuel path, class, footprint radius m, split): 'train' models are used for the
# training and validation images, 'test' models only for the held-out test.
MODELS = [
    ('OpenRobotics/models/Standing person', 'person', 0.35, 'train'),
    ('OpenRobotics/models/Walking person', 'person', 0.35, 'train'),
    ('OpenRobotics/models/Scrubs', 'person', 0.35, 'train'),
    ('OpenRobotics/models/OpScrubs', 'person', 0.35, 'train'),
    ('OpenRobotics/models/Casual female', 'person', 0.35, 'test'),
    ('OpenRobotics/models/Euro pallet', 'pallet', 0.70, 'train'),
    ('jfrascon/models/Euro_pallet_w_stacked_boxes', 'pallet', 0.75, 'train'),
    ('OpenRobotics/models/Pallet_Standard', 'pallet', 0.75, 'test'),
    ('OpenRobotics/models/Utility Cart', 'cart', 0.60, 'train'),
    ('OpenRobotics/models/Hand_Cart', 'cart', 0.70, 'train'),
    ('OpenRobotics/models/BMWCart', 'cart', 0.60, 'train'),
    ('OpenRobotics/models/TrolleyBox1', 'cart', 0.60, 'train'),
    ('OpenRobotics/models/BPCart', 'cart', 0.60, 'test'),
    ('OpenRobotics/models/ParkingTrolleyMin', 'cart', 0.50, 'test'),
    # v3: test only (forklifts and pallet jacks for training are built from parts).
    ('OpenRobotics/models/Forklift', 'forklift', 1.40, 'test'),
    ('OpenRobotics/models/Pallet_Jack', 'forklift', 0.90, 'test'),
    # v3: more people (Survivor: lying on the floor) and hand trucks.
    ('OpenRobotics/models/MaleVisitorOnPhone', 'person', 0.35, 'train'),
    ('OpenRobotics/models/FemaleVisitor', 'person', 0.35, 'train'),
    ('OpenRobotics/models/Nurse', 'person', 0.35, 'train'),
    ('OpenRobotics/models/Survivor Male', 'person', 0.95, 'train'),
    ('OpenRobotics/models/Survivor Female', 'person', 0.95, 'test'),
    ('OpenRobotics/models/Handtruck_01', 'cart', 0.45, 'train'),
    ('OpenRobotics/models/Handtruck_02', 'cart', 0.45, 'train'),
    # v2: more carts (none resembles the held-out BPCart or ParkingTrolleyMin).
    ('OpenRobotics/models/Cart: rigid suspension', 'cart', 0.70, 'train'),
    ('OpenRobotics/models/MiRCart', 'cart', 0.60, 'train'),
    ('OpenRobotics/models/TrolleyBox2', 'cart', 0.60, 'train'),
    ('OpenRobotics/models/SurgicalTrolley', 'cart', 0.60, 'train'),
    ('OpenRobotics/models/MopCart1', 'cart', 0.60, 'train'),
    # v2: unlabelled distractors (no class: the model must learn they are none).
    ('OpenRobotics/models/Cardboard box', None, 0.40, 'both'),
    ('OpenRobotics/models/Construction Cone', None, 0.30, 'both'),
    ('OpenRobotics/models/Construction Barrel', None, 0.45, 'both'),
    ('OpenRobotics/models/TrashBin', None, 0.40, 'both'),
    ('OpenRobotics/models/Large Crate', None, 0.70, 'both'),
]
COPIES = 2                     # instances of each model, so a scene may hold two
# v2: models that also come as a recoloured copy (one copy original, one recoloured).
RECOLOUR = {'cart', 'forklift'}
RECOLOUR_PATHS = {'OpenRobotics/models/Euro pallet'}
PALLETS_PER_WORLD = 10
FORKLIFTS_PER_WORLD = 6        # v3: built from parts (training only)
PALLET_JACKS_PER_WORLD = 4
REACH_TRUCKS_PER_WORLD = 3     # v4
STACKERS_PER_WORLD = 3         # v4: walk-behind stackers and order pickers
CARTS_PER_WORLD = 8            # v4: platform carts, roll cages, shelf trolleys
IGNORE_GREY = 114              # v4: labelled objects too thin to box are greyed out
RACK_PREFIX = 'forbidden_zone_compact_rack_'
RACK_PALLET_SHARE = 0.35       # rack loads that are pallets (labelled); the rest are cartons
INDUSTRIAL = [(0.92, 0.72, 0.05), (0.95, 0.45, 0.05), (0.75, 0.10, 0.08), (0.10, 0.25, 0.65),
              (0.12, 0.45, 0.20), (0.45, 0.45, 0.48), (0.08, 0.08, 0.08), (0.90, 0.90, 0.88),
              (0.70, 0.70, 0.72), (0.30, 0.55, 0.75)]
WOOD = [(0.78, 0.62, 0.42), (0.70, 0.52, 0.33), (0.55, 0.40, 0.26), (0.45, 0.33, 0.22),
        (0.60, 0.57, 0.52), (0.82, 0.70, 0.52), (0.74, 0.45, 0.30)]
PLASTIC = [(0.10, 0.25, 0.65), (0.08, 0.08, 0.08), (0.45, 0.45, 0.48), (0.15, 0.45, 0.25),
           (0.75, 0.15, 0.12)]
CARDBOARD = [(0.72, 0.56, 0.38), (0.65, 0.50, 0.32), (0.80, 0.68, 0.50), (0.88, 0.86, 0.82)]


def entity_name(path, copy):
    return re.sub(r'[^a-z0-9]+', '_', path.split('/')[-1].lower()) + f'_{copy}'


def label_plugin(label):
    return ('<plugin filename="gz-sim-label-system" name="gz::sim::systems::Label">'
            f'<label>{label}</label></plugin>')


def recolour_models(text, name_pattern, rgb):
    """New ambient/diffuse colour for every model whose name matches."""
    def paint(m):
        body = re.sub(r'<ambient>[^<]*</ambient>', '<ambient>%.3f %.3f %.3f 1</ambient>' % rgb, m.group(0))
        return re.sub(r'<diffuse>[^<]*</diffuse>', '<diffuse>%.3f %.3f %.3f 1</diffuse>' % rgb, body)
    return re.sub(r'<model name="(%s)">.*?</model>' % name_pattern, paint, text, flags=re.S)


def capture_world(pool, out_sdf, env=None):
    """The warehouse world plus the camera rig and the parked, labelled pool.
    v3 `env`: colours of floor, walls, columns and rack blocks, or racking
    (SDF of rack models) in place of the blocks."""
    text = WORLD_FILE.read_text()
    # The collision-scene plugin serves the robot stack only.
    text = re.sub(r'<plugin filename="tugger_collision_scene"[^>]*/>', '', text)
    if env:
        for pattern, key in (('warehouse_floor', 'floor'), ('[a-z_]*wall[a-z_]*', 'walls'),
                             ('column_[a-z0-9_]+', 'columns'), (RACK_PREFIX + '[0-9_]+', 'blocks')):
            if env.get(key):
                text = recolour_models(text, pattern, env[key])
        if env.get('racks') is not None:
            text = re.sub(r'<model name="%s[0-9_]+">.*?</model>' % RACK_PREFIX, '', text, flags=re.S)
            text = text.replace('</world>', env['racks'] + '\n</world>')
    rig = f'''
    <model name="rig"><static>true</static><pose>0 0 {CAMERA_HEIGHT_M} 0 0 0</pose>
      <link name="l">
        <sensor name="rgb" type="camera"><always_on>true</always_on><update_rate>5</update_rate>
          <topic>rig/rgb</topic><camera><horizontal_fov>{HFOV}</horizontal_fov>
          <image><width>{WIDTH}</width><height>{HEIGHT}</height></image>
          <clip><near>0.10</near><far>12.0</far></clip>
          <noise><type>gaussian</type><mean>0.0</mean><stddev>0.007</stddev></noise></camera></sensor>
        <sensor name="boxes" type="boundingbox_camera"><always_on>true</always_on>
          <update_rate>5</update_rate><topic>rig/boxes</topic>
          <camera><box_type>2d</box_type><horizontal_fov>{HFOV}</horizontal_fov>
          <image><width>{WIDTH}</width><height>{HEIGHT}</height></image>
          <clip><near>0.10</near><far>12.0</far></clip></camera></sensor>
      </link></model>'''
    includes = []
    for i, item in enumerate(pool):
        pose = f'<pose>{2.0 * i - len(pool)} {PARK_Y} 0 0 0 0</pose>'
        label = label_plugin(item['label']) if item['label'] else ''
        if 'sdf' in item:            # built from parts (v2 pallets)
            includes.append(f'<model name="{item["name"]}"><static>true</static>{pose}'
                            f'{item["sdf"]}{label}</model>')
            continue
        uri = item.get('uri') or f'{FUEL}/{item["path"].replace(" ", "%20")}'
        static = '' if item['actor'] else '<static>true</static>'
        includes.append(f'<include><uri>{uri}</uri><name>{item["name"]}</name>{static}'
                        f'{pose}{label}</include>')
    text = text.replace('</world>', rig + '\n' + '\n'.join(includes) + '\n</world>')
    out_sdf.write_text(text)


def cached_model(path):
    cache = Path.home() / '.gz/fuel/fuel.gazebosim.org' / path.lower().replace(' ', '%20')
    versions = sorted(cache.iterdir(), key=lambda p: int(p.name)) if cache.exists() else []
    if not versions:
        sys.exit(f'model not downloaded: {path} (run download_models.sh)')
    return versions[-1]


def is_actor(path):
    return '<actor' in (cached_model(path) / 'model.sdf').read_text()


def material(rgb):
    r, g, b = rgb
    return (f'<material><ambient>{r:.3f} {g:.3f} {b:.3f} 1</ambient>'
            f'<diffuse>{r:.3f} {g:.3f} {b:.3f} 1</diffuse>'
            '<specular>0.15 0.15 0.15 1</specular></material>')


def jitter(rng, rgb, amount=0.08):
    k = rng.uniform(1 - amount, 1 + amount)
    return tuple(min(1.0, max(0.0, c * k + rng.uniform(-0.02, 0.02))) for c in rgb)


TEXTURE_SKIP = ('normal', 'spec', 'gloss', 'rough', 'metal', '_ao', 'occlusion', 'thumbnail')


def shift_colours(img, hue, sat, val):
    """Recolour a texture: coloured pixels get a new hue and saturation, all get a
    brightness change; grey and black parts (tyres, steel, handles) keep no hue."""
    bgr = img[..., :3]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    coloured = hsv[..., 1] > 40
    hsv[..., 0] = np.where(coloured, (hsv[..., 0] + hue) % 180, hsv[..., 0])
    hsv[..., 1] = np.where(coloured, np.clip(hsv[..., 1] * sat, 0, 255), hsv[..., 1])
    hsv[..., 2] = np.clip(hsv[..., 2] * val, 0, 255)
    out = img.copy()
    out[..., :3] = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return out


def recolour_glb(path, change):
    """Rewrite the base-colour images inside a binary glTF (.glb) in place."""
    import struct
    data = path.read_bytes()
    json_len = struct.unpack('<I', data[12:16])[0]
    gltf = json.loads(data[20:20 + json_len])
    at = 20 + json_len
    binary = bytearray(data[at + 8:at + 8 + struct.unpack('<I', data[at:at + 4])[0]])
    sources = {gltf['textures'][m['pbrMetallicRoughness']['baseColorTexture']['index']]['source']
               for m in gltf.get('materials', [])
               if 'baseColorTexture' in m.get('pbrMetallicRoughness', {})}
    for i in sorted(sources):
        image = gltf['images'][i]
        view = gltf['bufferViews'][image['bufferView']]
        start = view.get('byteOffset', 0)
        img = cv2.imdecode(np.frombuffer(bytes(binary[start:start + view['byteLength']]),
                                         np.uint8), cv2.IMREAD_UNCHANGED)
        if img is None or img.ndim < 3:
            continue
        ext = '.jpg' if image.get('mimeType') == 'image/jpeg' else '.png'
        encoded = cv2.imencode(ext, change(img))[1].tobytes()
        binary += b'\0' * (-len(binary) % 4)
        view['byteOffset'], view['byteLength'] = len(binary), len(encoded)
        binary += encoded
    binary += b'\0' * (-len(binary) % 4)
    gltf['buffers'][0]['byteLength'] = len(binary)
    text = json.dumps(gltf).encode()
    text += b' ' * (-len(text) % 4)
    path.write_bytes(struct.pack('<III', 0x46546C67, 2, 28 + len(text) + len(binary))
                     + struct.pack('<I', len(text)) + b'JSON' + text
                     + struct.pack('<I', len(binary)) + b'BIN\0' + binary)
    return len(sources)


def recoloured_copy(path, folder, rng, palette):
    """A recoloured copy of a cached Fuel model. Textured models keep their
    texture, recoloured (shift_colours): detail, tyres and steel stay. Models
    without a texture get a new flat colour per visual instead."""
    src = cached_model(path)
    if folder.exists():
        shutil.rmtree(folder)
    shutil.copytree(src, folder, ignore=shutil.ignore_patterns('thumbnails'))
    hue, sat, val = rng.uniform(15, 165), rng.uniform(0.6, 1.4), rng.uniform(0.75, 1.2)

    def change(img):
        return shift_colours(img, hue, sat, val)
    textured = 0
    for f in folder.rglob('*'):
        name = f.name.lower()
        if f.suffix.lower() == '.glb':
            textured += recolour_glb(f, change)
        elif f.suffix.lower() in ('.png', '.jpg', '.jpeg') and not any(
                k in name for k in TEXTURE_SKIP):
            img = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
            if img is not None and img.ndim == 3:
                cv2.imwrite(str(f), change(img))
                textured += 1
    # Meshes named by their Fuel address would load the cached originals: use the copies.
    sdf = re.sub(r'https://fuel\.gazebosim\.org/1\.0/[^<]*?/files/', '',
                 (src / 'model.sdf').read_text())
    if not textured:
        def paint(match):
            body = re.sub(r'<material>.*?</material>', '', match.group(0), flags=re.S)
            colour = jitter(rng, rng.choice(palette) if rng.random() < 0.8 else
                            (rng.random(), rng.random(), rng.random()))
            return body.replace('</visual>', material(colour) + '</visual>')
        sdf = re.sub(r'<visual\b.*?</visual>', paint, sdf, flags=re.S)
    (folder / 'model.sdf').write_text(sdf)
    return str(folder)


def pallet_parts(rng):
    """A pallet built from boxes: (SDF of one link, footprint radius, description).
    Kinds: block (EUR-like: blocks, cross boards, deck), stringer (US-like: three
    long stringers under deck boards) or plastic (plate on feet or runners).
    Size, board count, colour and load are random; 30 % carry a load."""
    kind = rng.choice(['block', 'block', 'stringer', 'stringer', 'plastic'])
    length, width = rng.choice([(1.2, 0.8), (1.2, 1.0), (1.219, 1.016), (0.8, 0.6),
                                (1.14, 1.14), (1.0, 1.0), (1.2, 1.2)])
    parts = []                  # (x, y, z centre, sx, sy, sz, rgb)
    if kind == 'plastic':
        base = rng.choice(PLASTIC)
        feet_h, plate = rng.uniform(0.09, 0.12), rng.uniform(0.025, 0.045)
        if rng.random() < 0.5:
            for x in (-1, 0, 1):
                for y in (-1, 0, 1):
                    parts.append((x * (length / 2 - 0.08), y * (width / 2 - 0.08), feet_h / 2,
                                  0.16, 0.16, feet_h, base))
        else:
            for y in (-1, 0, 1):
                parts.append((0, y * (width / 2 - 0.06), feet_h / 2, length, 0.12, feet_h, base))
        parts.append((0, 0, feet_h + plate / 2, length, width, plate, base))
        top = feet_h + plate
    else:
        wood = rng.choice(WOOD)
        t = rng.uniform(0.016, 0.024)                       # board thickness
        board_w = rng.uniform(0.08, 0.145)
        if kind == 'block':
            block_h, block = rng.uniform(0.075, 0.10), rng.uniform(0.10, 0.15)
            xs = [-(length / 2 - block / 2), 0.0, length / 2 - block / 2]
            ys = [-(width / 2 - block / 2), 0.0, width / 2 - block / 2]
            z = 0.0
            if rng.random() < 0.8:                            # bottom boards
                for y in ys:
                    parts.append((0, y, t / 2, length, block, t, jitter(rng, wood)))
                z = t
            for x in xs:
                for y in ys:
                    parts.append((x, y, z + block_h / 2, block, block, block_h, jitter(rng, wood)))
            z += block_h
            for x in xs:                                      # cross boards on the blocks
                parts.append((x, 0, z + t / 2, block, width, t, jitter(rng, wood)))
            z += t
        else:
            st_h, st_w = rng.uniform(0.08, 0.10), rng.uniform(0.035, 0.05)
            n_bottom = rng.choice([0, 3, 5])
            z = t if n_bottom else 0.0
            for i in range(n_bottom):
                x = -length / 2 + board_w / 2 + i * (length - board_w) / (n_bottom - 1)
                parts.append((x, 0, t / 2, board_w, width, t, jitter(rng, wood)))
            for y in (-(width / 2 - st_w / 2), 0.0, width / 2 - st_w / 2):
                parts.append((0, y, z + st_h / 2, length, st_w, st_h, jitter(rng, wood)))
            z += st_h
        # Top deck: boards along the length (block) or across it (stringer).
        n_top = rng.randint(5, 9)
        along = kind == 'block'
        span = width if along else length
        for i in range(n_top):
            c = -span / 2 + board_w / 2 + i * (span - board_w) / (n_top - 1)
            sx, sy = (length, board_w) if along else (board_w, width)
            px, py = (0, c) if along else (c, 0)
            parts.append((px, py, z + t / 2, sx, sy, t, jitter(rng, wood)))
        top = z + t
        base = wood
    load = rng.choices(['none', 'cartons', 'wrapped'], [0.7, 0.18, 0.12])[0]
    if load == 'cartons':
        nx, ny, layers = rng.randint(1, 3), rng.randint(1, 3), rng.randint(1, 4)
        cx, cy = length * 0.95 / nx, width * 0.95 / ny
        ch = rng.uniform(0.18, 0.40)
        for k in range(layers):
            for i in range(nx):
                for j in range(ny):
                    if k and rng.random() < 0.15:
                        continue
                    parts.append((-length * 0.475 + cx * (i + 0.5), -width * 0.475 + cy * (j + 0.5),
                                  top + ch * (k + 0.5), cx * 0.97, cy * 0.97, ch * 0.98,
                                  jitter(rng, rng.choice(CARDBOARD), 0.05)))
    elif load == 'wrapped':
        h = rng.uniform(0.3, 1.2)
        parts.append((0, 0, top + h / 2, length * 0.96, width * 0.96, h,
                      jitter(rng, rng.choice([(0.86, 0.86, 0.88), (0.25, 0.25, 0.27),
                                              (0.72, 0.56, 0.38)]), 0.04)))
    visuals = ''.join(
        f'<visual name="v{i}"><pose>{x:.4f} {y:.4f} {z:.4f} 0 0 0</pose><geometry><box>'
        f'<size>{sx:.4f} {sy:.4f} {sz:.4f}</size></box></geometry>{material(rgb)}</visual>'
        for i, (x, y, z, sx, sy, sz, rgb) in enumerate(parts))
    return (f'<link name="l">{visuals}</link>', math.hypot(length, width) / 2,
            dict(kind=kind, size=[length, width], load=load,
                 colour=[round(c, 2) for c in base]))


def textured(path):
    return ('<material><ambient>1 1 1 1</ambient><diffuse>1 1 1 1</diffuse>'
            f'<pbr><metal><albedo_map>{path}</albedo_map><roughness>0.9</roughness>'
            '<metalness>0.0</metalness></metal></pbr></material>')


def visuals_sdf(parts):
    """Visuals of one link from parts: dict(xyz, size, rgb or tex, shape box|cyl, rpy)."""
    out = []
    for i, p in enumerate(parts):
        x, y, z = p['xyz']
        r, pt, yw = p.get('rpy', (0, 0, 0))
        if p.get('shape') == 'cyl':
            geom = f'<cylinder><radius>{p["size"][0]:.4f}</radius><length>{p["size"][1]:.4f}</length></cylinder>'
        else:
            geom = '<box><size>%.4f %.4f %.4f</size></box>' % tuple(p['size'])
        mat = textured(p['tex']) if p.get('tex') else material(p['rgb'])
        out.append(f'<visual name="v{i}"><pose>{x:.4f} {y:.4f} {z:.4f} {r:.4f} {pt:.4f} {yw:.4f}</pose>'
                   f'<geometry>{geom}</geometry>{mat}</visual>')
    return ''.join(out)


def make_textures(rng, folder):
    """Wood-grain and plastic textures for the built pallets (v3): planks with
    grain streaks, knots, darker edges and wear; plastic with speckle. Each wood
    texture comes as along (grain along x) and across (grain along y)."""
    folder.mkdir(parents=True, exist_ok=True)
    wood, plastic = [], []
    for k in range(12):
        base = np.array(jitter(rng, rng.choice(WOOD), 0.12))[::-1] * 255     # BGR
        h, w = 64, 512
        y = np.arange(h)[:, None]
        x = np.arange(w)[None, :]
        phase = rng.uniform(0, 6.28)
        grain = (np.sin(y * rng.uniform(0.5, 1.4) + 3 * np.sin(x / rng.uniform(40, 120) + phase))
                 * rng.uniform(0.04, 0.10))
        nrng = np.random.default_rng(rng.randrange(1 << 30))
        streak = cv2.resize(nrng.normal(0, 1, (h, 16)), (w, h)) * rng.uniform(0.03, 0.08)
        shade = 1 + grain + streak + nrng.normal(0, 0.025, (h, w))
        img = np.clip(base[None, None, :] * shade[..., None], 0, 255).astype(np.uint8)
        for _ in range(rng.randint(0, 3)):                                   # knots
            cv2.ellipse(img, (rng.randrange(w), rng.randrange(h)), (rng.randint(4, 10), rng.randint(3, 6)),
                        0, 0, 360, tuple(int(c * 0.55) for c in base), -1)
        img[:2], img[-2:] = (img[:2] * 0.7).astype(np.uint8), (img[-2:] * 0.7).astype(np.uint8)
        if rng.random() < 0.4:                                               # wear, dirt
            dirt = cv2.GaussianBlur(nrng.random((h, w)), (0, 0), 6)
            img = (img * (1 - 0.35 * (dirt > 0.55)[..., None])).astype(np.uint8)
        along, across = folder / f'wood_{k}_x.png', folder / f'wood_{k}_y.png'
        cv2.imwrite(str(along), img)
        cv2.imwrite(str(across), cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE))
        wood.append((str(along), str(across)))
    for k in range(5):
        base = np.array(jitter(rng, rng.choice(PLASTIC), 0.1))[::-1] * 255
        nrng = np.random.default_rng(rng.randrange(1 << 30))
        img = np.clip(base[None, None, :] * (1 + nrng.normal(0, 0.05, (128, 128)))[..., None],
                      0, 255).astype(np.uint8)
        f = folder / f'plastic_{k}.png'
        cv2.imwrite(str(f), img)
        plastic.append(str(f))
    return {'wood': wood, 'plastic': plastic}


def pallet_v3(rng, textures, size=None, load_weights=(0.7, 0.18, 0.12)):
    """A pallet built from parts with textured boards (v3). Same kinds as
    pallet_parts (block, stringer, plastic); returns (link SDF, radius, desc)."""
    kind = rng.choice(['block', 'block', 'stringer', 'stringer', 'plastic'])
    length, width = size or rng.choice([(1.2, 0.8), (1.2, 1.0), (1.219, 1.016), (0.8, 0.6),
                                        (1.14, 1.14), (1.0, 1.0), (1.2, 1.2)])
    parts = []
    tones = rng.sample(textures['wood'], 3)

    def board(x, y, z, sx, sy, sz):
        along, across = rng.choice(tones)
        parts.append(dict(xyz=(x, y, z), size=(sx, sy, sz), tex=along if sx >= sy else across))
    if kind == 'plastic':
        tex = rng.choice(textures['plastic'])
        feet_h, plate = rng.uniform(0.09, 0.12), rng.uniform(0.025, 0.045)
        if rng.random() < 0.5:
            for x in (-1, 0, 1):
                for y in (-1, 0, 1):
                    parts.append(dict(xyz=(x * (length / 2 - 0.08), y * (width / 2 - 0.08), feet_h / 2),
                                      size=(0.16, 0.16, feet_h), tex=tex))
        else:
            for y in (-1, 0, 1):
                parts.append(dict(xyz=(0, y * (width / 2 - 0.06), feet_h / 2),
                                  size=(length, 0.12, feet_h), tex=tex))
        parts.append(dict(xyz=(0, 0, feet_h + plate / 2), size=(length, width, plate), tex=tex))
        top = feet_h + plate
    else:
        t, board_w = rng.uniform(0.016, 0.024), rng.uniform(0.08, 0.145)
        if kind == 'block':
            block_h, block = rng.uniform(0.075, 0.10), rng.uniform(0.10, 0.15)
            xs = [-(length / 2 - block / 2), 0.0, length / 2 - block / 2]
            ys = [-(width / 2 - block / 2), 0.0, width / 2 - block / 2]
            z = 0.0
            if rng.random() < 0.8:
                for y in ys:
                    board(0, y, t / 2, length, block, t)
                z = t
            for x in xs:
                for y in ys:
                    board(x, y, z + block_h / 2, block, block, block_h)
            z += block_h
            for x in xs:
                board(x, 0, z + t / 2, block, width, t)
            z += t
        else:
            st_h, st_w = rng.uniform(0.08, 0.10), rng.uniform(0.035, 0.05)
            n_bottom = rng.choice([0, 3, 5])
            z = t if n_bottom else 0.0
            for i in range(n_bottom):
                board(-length / 2 + board_w / 2 + i * (length - board_w) / (n_bottom - 1), 0, t / 2,
                      board_w, width, t)
            for y in (-(width / 2 - st_w / 2), 0.0, width / 2 - st_w / 2):
                board(0, y, z + st_h / 2, length, st_w, st_h)
            z += st_h
        n_top, along = rng.randint(5, 9), kind == 'block'
        span = width if along else length
        for i in range(n_top):
            c = -span / 2 + board_w / 2 + i * (span - board_w) / (n_top - 1)
            if along:
                board(0, c, z + t / 2, length, board_w, t)
            else:
                board(c, 0, z + t / 2, board_w, width, t)
        top = z + t
    load = rng.choices(['none', 'cartons', 'wrapped'], load_weights)[0]
    if load == 'cartons':
        nx, ny, layers = rng.randint(1, 3), rng.randint(1, 3), rng.randint(1, 4)
        cx, cy, ch = length * 0.95 / nx, width * 0.95 / ny, rng.uniform(0.18, 0.40)
        for k in range(layers):
            for i in range(nx):
                for j in range(ny):
                    if k and rng.random() < 0.15:
                        continue
                    parts.append(dict(xyz=(-length * 0.475 + cx * (i + 0.5), -width * 0.475 + cy * (j + 0.5),
                                           top + ch * (k + 0.5)), size=(cx * 0.97, cy * 0.97, ch * 0.98),
                                      rgb=jitter(rng, rng.choice(CARDBOARD), 0.05)))
    elif load == 'wrapped':
        h = rng.uniform(0.3, 1.2)
        parts.append(dict(xyz=(0, 0, top + h / 2), size=(length * 0.96, width * 0.96, h),
                          rgb=jitter(rng, rng.choice([(0.86, 0.86, 0.88), (0.25, 0.25, 0.27),
                                                      (0.72, 0.56, 0.38)]), 0.04)))
    return (f'<link name="l">{visuals_sdf(parts)}</link>', math.hypot(length, width) / 2,
            dict(kind=kind, size=[length, width], load=load))


def wheel(x, y, radius, width, rgb=(0.06, 0.06, 0.06)):
    return dict(xyz=(x, y, radius), size=(radius, width), shape='cyl', rpy=(math.pi / 2, 0, 0), rgb=rgb)


def forklift_parts(rng):
    """A counterbalance forklift built from parts (v3): body, counterweight,
    wheels, overhead guard, seat, mast with carriage and forks (raised at a
    random height); proportions and colours random. Forks point to +x."""
    body_c = jitter(rng, rng.choice(INDUSTRIAL[:6] + [(0.20, 0.45, 0.25), (0.85, 0.85, 0.80)]))
    dark = jitter(rng, rng.choice([(0.12, 0.12, 0.13), (0.25, 0.25, 0.27), (0.05, 0.05, 0.05)]), 0.05)
    L, W = rng.uniform(1.6, 2.4), rng.uniform(0.95, 1.25)
    rf, rr = rng.uniform(0.25, 0.33), rng.uniform(0.2, 0.27)
    body_h, guard_h = rng.uniform(0.75, 1.05), rng.uniform(1.95, 2.25)
    parts = [dict(xyz=(0, 0, 0.2 + body_h / 2), size=(L * 0.8, W, body_h), rgb=body_c),
             dict(xyz=(-L * 0.42, 0, 0.2 + body_h * 0.45), size=(L * 0.22, W * 1.02, body_h * 0.9),
                  rgb=jitter(rng, body_c, 0.15))]
    for sy in (-1, 1):
        parts.append(wheel(L * 0.3, sy * (W / 2 - 0.05), rf, 0.22))
        parts.append(wheel(-L * 0.32, sy * (W / 2 - 0.08), rr, 0.18))
    post = rng.uniform(0.04, 0.07)
    for px in (L * 0.18, -L * 0.25):
        for sy in (-1, 1):
            parts.append(dict(xyz=(px, sy * (W / 2 - 0.05), (0.2 + body_h + guard_h) / 2),
                              size=(post, post, guard_h - 0.2 - body_h), rgb=dark))
    for k in range(rng.randint(3, 6)):                  # guard roof bars
        parts.append(dict(xyz=(-L * 0.25 + (k + 0.5) * (L * 0.43) / 6, 0, guard_h),
                          size=(0.04, W * 0.95, 0.04), rgb=dark))
    parts.append(dict(xyz=(-L * 0.1, 0, 0.2 + body_h + 0.25), size=(0.45, 0.5, 0.5), rgb=dark))   # seat
    mast_h, mx = rng.uniform(2.0, 3.0), L * 0.4 + 0.08
    for sy in (-1, 1):
        parts.append(dict(xyz=(mx, sy * 0.33, mast_h / 2), size=(0.1, 0.12, mast_h), rgb=dark))
    for z in (0.3, mast_h * 0.55, mast_h - 0.05):
        parts.append(dict(xyz=(mx, 0, z), size=(0.08, 0.78, 0.07), rgb=dark))
    lift = rng.choice([0.05, 0.05, rng.uniform(0.1, 1.4)])
    parts.append(dict(xyz=(mx + 0.08, 0, lift + 0.25), size=(0.05, 0.9, 0.45), rgb=dark))      # carriage
    fork_l = rng.uniform(0.9, 1.25)
    for sy in (-1, 1):
        parts.append(dict(xyz=(mx + 0.1 + fork_l / 2, sy * rng.uniform(0.2, 0.32), lift + 0.02),
                          size=(fork_l, 0.11, 0.045), rgb=dark))
    loaded = rng.random() < 0.3                         # v4: a load on the forks (part of the truck)
    if loaded:
        parts += fork_load(rng, mx + 0.12 + fork_l / 2, lift + 0.045, fork_l)
    return (f'<link name="l">{visuals_sdf(parts)}</link>', math.hypot(L + fork_l, W) / 2,
            dict(kind='forklift', length=round(L, 2), mast=round(mast_h, 2), lift=round(lift, 2),
                 loaded=loaded))


def fork_load(rng, x, z, fork_l):
    """v4: what a truck carries on its forks: cartons or a wrapped block (no
    separate pallet label: the load counts as part of the truck, as in LOCO)."""
    h = rng.uniform(0.3, 1.1)
    if rng.random() < 0.5:
        return [dict(xyz=(x, 0, z + h / 2), size=(fork_l * 0.95, rng.uniform(0.75, 1.0), h),
                     rgb=jitter(rng, rng.choice([(0.86, 0.86, 0.88), (0.25, 0.25, 0.27)]), 0.04))]
    out, n = [], rng.randint(1, 3)
    for k in range(n):
        out.append(dict(xyz=(x, 0, z + h / n * (k + 0.5)), size=(fork_l * 0.9, rng.uniform(0.6, 0.95), h / n * 0.97),
                        rgb=jitter(rng, rng.choice(CARDBOARD), 0.05)))
    return out


def castor(x, y, radius, rgb=(0.06, 0.06, 0.06)):
    return [wheel(x, y, radius, radius * 0.7, rgb),
            dict(xyz=(x, y, radius * 2 + 0.02), size=(0.05, 0.05, 0.04), rgb=(0.5, 0.5, 0.52))]


def cart_parts(rng, kind=None):
    """v4 carts built from parts, random sizes and colours (v3 boxed only the deck
    of a tall-handled platform cart, and real roll cages were missing):
    platform  - flat deck on castors, thin push handle at one or both ends;
    roll_cage - deck on castors with mesh sides 1.5-1.8 m high, a side may be open;
    shelves   - two to four shelves on corner posts, with a push handle.
    Any of them may carry cartons. Front is +x."""
    kind = kind or rng.choice(['platform', 'platform', 'roll_cage', 'roll_cage', 'shelves'])
    metal = jitter(rng, rng.choice([(0.62, 0.64, 0.66), (0.45, 0.47, 0.50), (0.92, 0.72, 0.05),
                                    (0.10, 0.25, 0.65), (0.75, 0.10, 0.08), (0.15, 0.15, 0.16),
                                    (0.12, 0.45, 0.20)]), 0.06)
    deck_c = jitter(rng, rng.choice([metal, (0.15, 0.15, 0.16), (0.55, 0.40, 0.26), (0.10, 0.25, 0.65)]), 0.05)
    bar = rng.uniform(0.018, 0.03)
    if kind == 'roll_cage':
        L, W, H = rng.uniform(0.7, 0.85), rng.uniform(0.6, 0.75), rng.uniform(1.5, 1.8)
    elif kind == 'shelves':
        L, W, H = rng.uniform(0.8, 1.2), rng.uniform(0.45, 0.65), rng.uniform(0.9, 1.6)
    else:
        L, W, H = rng.uniform(0.8, 1.25), rng.uniform(0.5, 0.8), rng.uniform(0.85, 1.05)
    cr = rng.uniform(0.05, 0.08)
    deck_z = cr * 2 + 0.05
    parts = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            parts += castor(sx * (L / 2 - 0.08), sy * (W / 2 - 0.07), cr)
    parts.append(dict(xyz=(0, 0, deck_z), size=(L, W, rng.uniform(0.03, 0.06)), rgb=deck_c))
    top = deck_z + 0.03
    if kind == 'roll_cage':
        open_side = rng.choice([None, None, 'front', 'left'])
        step_v, step_h = rng.uniform(0.08, 0.15), rng.uniform(0.2, 0.4)
        sides = {'back': (-L / 2, 'y', W), 'front': (L / 2, 'y', W), 'left': (W / 2, 'x', L),
                 'right': (-W / 2, 'x', L)}
        for name, (off, along, span) in sides.items():
            if name == open_side:
                continue
            for k in range(int(span / step_v) + 1):           # vertical mesh bars
                c = -span / 2 + k * span / int(span / step_v)
                xy = (off, c) if along == 'y' else (c, off)
                parts.append(dict(xyz=(*xy, top + (H - top) / 2), size=(bar, bar, H - top), rgb=metal))
            for z in np.arange(top + step_h, H + 0.01, step_h):   # horizontal mesh bars
                size = (bar, span, bar) if along == 'y' else (span, bar, bar)
                xy = (off, 0) if along == 'y' else (0, off)
                parts.append(dict(xyz=(*xy, z), size=size, rgb=metal))
        if rng.random() < 0.3:                                 # middle shelf
            parts.append(dict(xyz=(0, 0, top + (H - top) / 2), size=(L, W, 0.02), rgb=metal))
    else:
        if kind == 'shelves':
            n = rng.randint(2, 4)
            for sx in (-1, 1):
                for sy in (-1, 1):
                    parts.append(dict(xyz=(sx * (L / 2 - 0.02), sy * (W / 2 - 0.02), (top + H) / 2),
                                      size=(bar * 1.3, bar * 1.3, H - top), rgb=metal))
            for k in range(1, n):
                parts.append(dict(xyz=(0, 0, top + (H - top) * k / (n - 1)), size=(L, W, 0.025), rgb=deck_c))
        ends = [-1] if kind == 'shelves' or rng.random() < 0.7 else [-1, 1]
        hh = H + (rng.uniform(0.0, 0.15) if kind == 'shelves' else 0)
        for sx in ends:                                        # thin push handle(s)
            hx = sx * (L / 2 + rng.uniform(0.0, 0.05))
            for sy in (-1, 1):
                parts.append(dict(xyz=(hx, sy * (W / 2 - 0.04), (top + hh) / 2), size=(bar, bar, hh - top), rgb=metal))
            parts.append(dict(xyz=(hx, 0, hh), size=(bar, W - 0.06, bar), rgb=metal))
            if rng.random() < 0.5:
                parts.append(dict(xyz=(hx, 0, (top + hh) / 2), size=(bar, W - 0.06, bar), rgb=metal))
    if rng.random() < 0.45:                                    # cartons on board
        room = (H - top) * (0.9 if kind == 'roll_cage' else 0.6)
        for k in range(rng.randint(1, 4)):
            ch = rng.uniform(0.15, 0.4)
            z0 = top + ch / 2 + (0 if kind != 'roll_cage' else rng.choice([0, 0, 0.3]))
            if z0 + ch / 2 > top + room:
                break
            parts.append(dict(xyz=(rng.uniform(-L / 4, L / 4), rng.uniform(-W / 5, W / 5), z0),
                              size=(rng.uniform(0.25, L * 0.6), rng.uniform(0.2, W * 0.6), ch),
                              rgb=jitter(rng, rng.choice(CARDBOARD), 0.05)))
    return (f'<link name="l">{visuals_sdf(parts)}</link>', math.hypot(L + 0.1, W) / 2,
            dict(kind=kind, size=[round(L, 2), round(W, 2), round(H, 2)]))


def reach_truck_parts(rng):
    """v4: a reach truck: compact body and standing operator cab with overhead
    guard at the back, two outrigger legs with load wheels reaching forward,
    a tall mast between the legs (moved forward or back), forks at any height."""
    body_c = jitter(rng, rng.choice(INDUSTRIAL[:6] + [(0.20, 0.45, 0.25), (0.85, 0.85, 0.80)]))
    dark = jitter(rng, rng.choice([(0.12, 0.12, 0.13), (0.25, 0.25, 0.27)]), 0.05)
    L, W = rng.uniform(1.0, 1.35), rng.uniform(1.0, 1.25)
    body_h = rng.uniform(1.0, 1.3)
    leg_l, leg_gap = rng.uniform(0.9, 1.3), rng.uniform(0.38, 0.5)
    parts = [dict(xyz=(-L / 2, 0, 0.1 + body_h / 2), size=(L, W, body_h), rgb=body_c),
             wheel(-L * 0.8, 0, 0.17, 0.15)]
    guard_h = rng.uniform(2.1, 2.35)
    for px in (-0.1, -L + 0.1):
        for sy in (-1, 1):
            parts.append(dict(xyz=(px, sy * (W / 2 - 0.04), (body_h + guard_h) / 2),
                              size=(0.05, 0.05, guard_h - body_h), rgb=dark))
    parts.append(dict(xyz=(-L / 2, 0, guard_h), size=(L, W * 0.95, 0.05), rgb=dark))
    for sy in (-1, 1):                                          # outrigger legs, load wheels
        parts.append(dict(xyz=(leg_l / 2, sy * leg_gap, 0.13), size=(leg_l, 0.14, 0.22), rgb=body_c))
        parts.append(wheel(leg_l - 0.12, sy * leg_gap, 0.08, 0.1))
    mast_h, mx = rng.uniform(2.3, 3.4), rng.uniform(0.15, leg_l - 0.2)
    for sy in (-1, 1):
        parts.append(dict(xyz=(mx, sy * 0.3, mast_h / 2 + 0.05), size=(0.1, 0.1, mast_h), rgb=dark))
    for z in (0.35, mast_h * 0.5, mast_h):
        parts.append(dict(xyz=(mx, 0, z), size=(0.08, 0.7, 0.07), rgb=dark))
    lift = rng.choice([0.3, rng.uniform(0.3, 2.2)])
    fork_l = rng.uniform(0.9, 1.15)
    parts.append(dict(xyz=(mx + 0.08, 0, lift + 0.22), size=(0.05, 0.8, 0.4), rgb=dark))
    for sy in (-1, 1):
        parts.append(dict(xyz=(mx + 0.1 + fork_l / 2, sy * 0.25, lift), size=(fork_l, 0.1, 0.04), rgb=dark))
    if rng.random() < 0.3:
        parts += fork_load(rng, mx + 0.12 + fork_l / 2, lift + 0.02, fork_l)
    return (f'<link name="l">{visuals_sdf(parts)}</link>', math.hypot(L + mx + fork_l, W) / 2,
            dict(kind='reach truck', mast=round(mast_h, 2), lift=round(lift, 2)))


def stacker_parts(rng, picker=None):
    """v4: a pedestrian stacker (walk-behind, mast over straddle legs, tiller) or
    an order picker (operator platform with rails that rises along the mast)."""
    picker = rng.random() < 0.35 if picker is None else picker
    c = jitter(rng, rng.choice(INDUSTRIAL[:6] + [(0.45, 0.45, 0.48), (0.12, 0.45, 0.20)]))
    dark = (0.08, 0.08, 0.08)
    W = rng.uniform(0.8, 1.0)
    bl, bh = rng.uniform(0.45, 0.8), rng.uniform(0.9, 1.4)
    parts = [dict(xyz=(-bl / 2, 0, 0.08 + bh / 2), size=(bl, W, bh), rgb=c), wheel(-bl / 2, 0, 0.12, 0.1, dark)]
    leg_l = rng.uniform(0.8, 1.15)
    for sy in (-1, 1):
        parts.append(dict(xyz=(leg_l / 2, sy * (W / 2 - 0.08), 0.06), size=(leg_l, 0.12, 0.1), rgb=c))
        parts.append(wheel(leg_l - 0.08, sy * (W / 2 - 0.08), 0.05, 0.08, dark))
    mast_h = rng.uniform(1.8, 3.2) if not picker else rng.uniform(3.0, 4.0)
    for sy in (-1, 1):
        parts.append(dict(xyz=(0.05, sy * 0.3, mast_h / 2 + 0.05), size=(0.09, 0.09, mast_h), rgb=dark))
    parts.append(dict(xyz=(0.05, 0, mast_h), size=(0.08, 0.68, 0.06), rgb=dark))
    lift = rng.choice([0.1, rng.uniform(0.1, mast_h - 0.6)])
    fork_l = rng.uniform(0.9, 1.15)
    for sy in (-1, 1):
        parts.append(dict(xyz=(0.15 + fork_l / 2, sy * 0.22, lift + 0.05), size=(fork_l, 0.1, 0.04), rgb=dark))
    if picker:                                                 # operator platform with rails
        pz = lift + 0.25
        parts.append(dict(xyz=(-0.25, 0, pz), size=(0.7, W * 0.9, 0.06), rgb=c))
        for px in (-0.55, 0.05):
            for sy in (-1, 1):
                parts.append(dict(xyz=(px, sy * W * 0.43, pz + 0.55), size=(0.04, 0.04, 1.1), rgb=c))
        parts.append(dict(xyz=(-0.25, W * 0.43, pz + 1.1), size=(0.64, 0.04, 0.04), rgb=c))
        parts.append(dict(xyz=(-0.25, -W * 0.43, pz + 1.1), size=(0.64, 0.04, 0.04), rgb=c))
    else:                                                      # tiller
        tilt, hl = rng.uniform(-0.5, 0.0), rng.uniform(0.9, 1.15)
        parts.append(dict(xyz=(-bl - math.sin(tilt) * hl / 2 + 0.1, 0, 0.6 + math.cos(tilt) * hl / 2),
                          size=(0.04, 0.04, hl), rpy=(0, tilt, 0), rgb=dark))
    if rng.random() < 0.3:
        parts += fork_load(rng, 0.17 + fork_l / 2, lift + 0.07, fork_l)
    return (f'<link name="l">{visuals_sdf(parts)}</link>', math.hypot(bl + leg_l, W) / 2,
            dict(kind='order picker' if picker else 'stacker', mast=round(mast_h, 2), lift=round(lift, 2)))


def pallet_jack_parts(rng):
    """A manual or electric pallet jack built from parts (v3): two low forks,
    a body (pump or battery box), steering wheels and a tiller handle."""
    c = jitter(rng, rng.choice([(0.92, 0.72, 0.05), (0.95, 0.45, 0.05), (0.75, 0.10, 0.08),
                                (0.10, 0.25, 0.65), (0.45, 0.45, 0.48), (0.12, 0.45, 0.20)]))
    dark = (0.08, 0.08, 0.08)
    fork_l, gap = rng.uniform(1.0, 1.2), rng.uniform(0.18, 0.28)
    electric = rng.random() < 0.35
    parts = []
    for sy in (-1, 1):
        parts.append(dict(xyz=(fork_l / 2, sy * gap, 0.05), size=(fork_l, 0.16, 0.07), rgb=c))
        parts.append(wheel(fork_l - 0.08, sy * gap, 0.04, 0.07, dark))
    if electric:
        bh = rng.uniform(0.9, 1.3)
        parts.append(dict(xyz=(-0.25, 0, bh / 2 + 0.05), size=(0.5, 0.75, bh), rgb=c))
        parts.append(dict(xyz=(-0.45, 0, bh + 0.15), size=(0.06, 0.06, 0.25), rgb=dark))
    else:
        parts.append(dict(xyz=(-0.05, 0, 0.18), size=(0.22, 0.6, 0.18), rgb=c))
        parts.append(dict(xyz=(-0.12, 0, 0.32), size=(0.09, 0.08), shape='cyl', rgb=dark))
    parts.append(wheel(-0.15, 0, 0.09, 0.08, dark))
    tilt = rng.uniform(-0.35, 0.1)
    hl = rng.uniform(1.0, 1.25)
    top_z = (0.55 if electric else 0.3)
    parts.append(dict(xyz=(-0.18 - math.sin(tilt) * hl / 2, 0, top_z + math.cos(tilt) * hl / 2),
                      size=(0.035, 0.035, hl), rpy=(0, tilt, 0), rgb=dark))
    hx, hz = -0.18 - math.sin(tilt) * hl, top_z + math.cos(tilt) * hl
    parts.append(dict(xyz=(hx, 0, hz), size=(0.04, 0.3, 0.04), rgb=dark))
    return (f'<link name="l">{visuals_sdf(parts)}</link>', math.hypot(fork_l + 0.5, 0.75) / 2,
            dict(kind='pallet jack', electric=electric))


def rack_world(rng, textures, label_pallets, original_world_text):
    """Pallet racking in place of the red rack blocks (v3): uprights, beams and
    decks in the blocks' footprints, two to four levels, each holding pallets
    (labelled 'pallet' in training worlds) or plain cartons. Returns the SDF."""
    upright = jitter(rng, rng.choice([(0.10, 0.25, 0.65), (0.30, 0.35, 0.40), (0.92, 0.72, 0.05),
                                      (0.20, 0.45, 0.25), (0.55, 0.57, 0.60)]))
    beam = jitter(rng, rng.choice([(0.95, 0.45, 0.05), (0.92, 0.72, 0.05), (0.75, 0.10, 0.08), upright]))
    deck = (0.62, 0.64, 0.66)
    models = []
    for name, x, y in re.findall(r'<model name="(%s[0-9_]+)">\s*<static>true</static>\s*'
                                  r'<pose>([-0-9.]+) ([-0-9.]+)' % RACK_PREFIX, original_world_text):
        x, y = float(x), float(y)
        L, D = 2.3, 0.9
        levels = [0.0] + [rng.uniform(1.0, 1.3)]
        while len(levels) < rng.randint(2, 4):
            levels.append(levels[-1] + rng.uniform(1.0, 1.3))
        H = levels[-1] + rng.uniform(0.3, 0.6)
        parts = []
        for sx in (-1, 1):
            for sy in (-1, 1):
                parts.append(dict(xyz=(sx * (L / 2 - 0.04), sy * (D / 2 - 0.04), H / 2),
                                  size=(0.08, 0.08, H), rgb=upright))
            for z in np.arange(0.4, H, 0.6):                 # frame bracing
                parts.append(dict(xyz=(sx * (L / 2 - 0.04), 0, z), size=(0.03, D - 0.1, 0.03), rgb=upright))
        for z in levels[1:]:
            for sy in (-1, 1):
                parts.append(dict(xyz=(0, sy * (D / 2 - 0.04), z - 0.06), size=(L, 0.05, 0.12), rgb=beam))
            parts.append(dict(xyz=(0, 0, z + 0.01), size=(L - 0.1, D - 0.1, 0.02), rgb=deck))
        models.append(f'<model name="{name}"><static>true</static><pose>{x} {y} 0 0 0 0</pose>'
                      f'<link name="l">{visuals_sdf(parts)}</link></model>')
        for li, z in enumerate(levels):
            for k, px in enumerate((-0.57, 0.57)):
                if rng.random() < 0.2:
                    continue
                pname = f'{name}_load_{li}_{k}'
                if label_pallets and rng.random() < RACK_PALLET_SHARE:
                    sdf, _, _ = pallet_v3(rng, textures, size=(rng.uniform(1.0, 1.1), 0.8),
                                          load_weights=(0.3, 0.45, 0.25))
                    models.append(f'<model name="{pname}"><static>true</static>'
                                  f'<pose>{x + px} {y} {z + 0.02} 0 0 0</pose>{sdf}'
                                  f'{label_plugin(CLASSES.index("pallet") + 1)}</model>')
                else:                                        # cartons only, no class
                    h = rng.uniform(0.3, 0.9)
                    carton = dict(xyz=(0, 0, h / 2), size=(rng.uniform(0.6, 1.0), rng.uniform(0.5, 0.8), h),
                                  rgb=jitter(rng, rng.choice(CARDBOARD), 0.05))
                    models.append(f'<model name="{pname}"><static>true</static>'
                                  f'<pose>{x + px} {y} {z + 0.02} 0 0 0</pose>'
                                  f'<link name="l">{visuals_sdf([carton])}</link></model>')
    return '\n'.join(models)


def world_env(rng, w, test, textures):
    """v3 look of capture world w: world 0 keeps the original warehouse; every
    fourth world (and test world 0) keeps rack blocks in new colours; the others
    get pallet racking. Floor, walls and columns change colour."""
    if w == 0:
        return {'desc': 'original'}
    floor = jitter(rng, rng.choice([(0.46, 0.48, 0.50), (0.35, 0.36, 0.37), (0.60, 0.60, 0.58),
                                    (0.25, 0.40, 0.30), (0.30, 0.35, 0.45), (0.55, 0.50, 0.42)]), 0.08)
    walls = jitter(rng, rng.choice([(0.73, 0.76, 0.80), (0.85, 0.85, 0.82), (0.55, 0.58, 0.62),
                                    (0.80, 0.75, 0.62), (0.45, 0.50, 0.55)]), 0.06)
    columns = jitter(rng, rng.choice([(0.13, 0.27, 0.47), (0.92, 0.72, 0.05), (0.45, 0.45, 0.48),
                                      (0.75, 0.10, 0.08), (0.85, 0.85, 0.85)]), 0.06)
    env = {'floor': floor, 'walls': walls, 'columns': columns}
    if w % 4 == 0:
        env['blocks'] = jitter(rng, rng.choice(INDUSTRIAL + [(0.9, 0.08, 0.08)]), 0.06)
        env['desc'] = 'blocks'
    else:
        env['racks'] = rack_world(rng, textures, not test, WORLD_FILE.read_text())
        env['desc'] = 'racking'
    return env


def degrade(img, rng):
    """Camera effects (v3): motion blur, defocus, exposure, white balance,
    contrast, sensor noise and JPEG compression, each with a random strength."""
    out = img.astype(np.float32)
    if rng.random() < 0.35:
        k = rng.choice([3, 5, 7, 9])
        kernel = np.zeros((k, k), np.float32)
        kernel[k // 2] = 1.0 / k
        rot = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), rng.uniform(0, 180), 1.0)
        kernel = cv2.warpAffine(kernel, rot, (k, k))
        out = cv2.filter2D(out, -1, kernel / max(kernel.sum(), 1e-6))
    if rng.random() < 0.25:
        out = cv2.GaussianBlur(out, (0, 0), rng.uniform(0.6, 1.4))
    gain = np.array([rng.uniform(0.95, 1.05) for _ in range(3)], np.float32)
    out = out * gain * rng.uniform(0.8, 1.15)
    mean = out.mean()
    out = (out - mean) * rng.uniform(0.8, 1.15) + mean
    out = 255.0 * np.clip(out / 255.0, 0, 1) ** rng.uniform(0.75, 1.35)
    nrng = np.random.default_rng(rng.randrange(1 << 30))
    out = np.clip(out + nrng.normal(0, rng.uniform(0, 6), out.shape), 0, 255).astype(np.uint8)
    if rng.random() < 0.5:
        out = cv2.imdecode(cv2.imencode('.jpg', out, [cv2.IMWRITE_JPEG_QUALITY,
                                                      rng.randint(40, 95)])[1], cv2.IMREAD_COLOR)
    return out


pool_textures = {}


def build_pool(rng, test, models_dir):
    """Every object of one capture world. v1 (and --test): COPIES of each model.
    v2 training: one original and one recoloured copy of carts, forklifts and the
    Euro pallet, PALLETS_PER_WORLD pallets built from parts, and the distractors."""
    wanted = ('test', 'both') if test else ('train', 'both')
    pool = []
    for path, cls, radius, split in MODELS:
        if split not in wanted:
            continue
        for c in range(COPIES):
            item = dict(path=path, cls=cls, radius=radius, name=entity_name(path, c),
                        label=CLASSES.index(cls) + 1 if cls else 0, actor=is_actor(path))
            if not test and c == 1 and (cls in RECOLOUR or path in RECOLOUR_PATHS):
                palette = WOOD * 2 + PLASTIC if cls == 'pallet' else INDUSTRIAL
                item['uri'] = recoloured_copy(path, models_dir / item['name'], rng, palette)
                item['recoloured'] = True
            pool.append(item)
    if test:                     # v4: the new kinds also in the test, as unseen built instances
        builders = ([('cart', lambda: cart_parts(rng, 'roll_cage'))] * 2
                    + [('cart', lambda: cart_parts(rng, 'platform'))]
                    + [('forklift', lambda: reach_truck_parts(rng))] * 2
                    + [('forklift', lambda: stacker_parts(rng))] * 2)
        for k, (cls, build) in enumerate(builders):
            sdf, radius, desc = build()
            pool.append(dict(path=f'parts/{desc["kind"].replace(" ", "_")}_{k}', cls=cls,
                             radius=radius, name=f'built_{k}', label=CLASSES.index(cls) + 1,
                             actor=False, sdf=sdf, built=desc))
    if not test:
        textures = make_textures(rng, models_dir / 'textures')
        builders = ([('pallet', lambda: pallet_v3(rng, textures))] * PALLETS_PER_WORLD
                    + [('forklift', lambda: forklift_parts(rng))] * FORKLIFTS_PER_WORLD
                    + [('forklift', lambda: pallet_jack_parts(rng))] * PALLET_JACKS_PER_WORLD
                    + [('forklift', lambda: reach_truck_parts(rng))] * REACH_TRUCKS_PER_WORLD
                    + [('forklift', lambda: stacker_parts(rng))] * STACKERS_PER_WORLD
                    + [('cart', lambda: cart_parts(rng))] * CARTS_PER_WORLD)
        for k, (cls, build) in enumerate(builders):
            sdf, radius, desc = build()
            pool.append(dict(path=f'parts/{desc["kind"].replace(" ", "_")}_{k}', cls=cls,
                             radius=radius, name=f'built_{k}', label=CLASSES.index(cls) + 1,
                             actor=False, sdf=sdf, built=desc))
        pool_textures[id(pool)] = textures
    return pool


class FreeSpace:
    """Free floor from the warehouse map, with each cell's distance to the nearest wall."""

    def __init__(self):
        import yaml
        meta = yaml.safe_load(MAP_YAML.read_text())
        img = cv2.imread(str(MAP_YAML.parent / meta['image']), cv2.IMREAD_GRAYSCALE)
        self.res, self.origin = meta['resolution'], meta['origin']
        self.h = img.shape[0]
        free = (img > 250).astype(np.uint8)
        self.clear = cv2.distanceTransform(free, cv2.DIST_L2, 5) * self.res

    def clearance(self, x, y):
        col = int((x - self.origin[0]) / self.res)
        row = self.h - 1 - int((y - self.origin[1]) / self.res)
        if not (0 <= row < self.clear.shape[0] and 0 <= col < self.clear.shape[1]):
            return 0.0
        return float(self.clear[row, col])

    def random_spot(self, rng, margin):
        rows, cols = np.nonzero(self.clear >= margin)
        k = rng.randrange(len(rows))
        x = self.origin[0] + (cols[k] + rng.random()) * self.res
        y = self.origin[1] + (self.h - 1 - rows[k] + rng.random()) * self.res
        return x, y


class Gazebo:
    def __init__(self):
        from gz.transport import Node
        from gz.msgs.image_pb2 import Image
        from gz.msgs.annotated_axis_aligned_2d_box_v_pb2 import AnnotatedAxisAligned2DBox_V
        self.node = Node()
        self.image, self.boxes = {}, {}
        self.out_now = set()            # pool entities currently placed in the warehouse
        self.node.subscribe(Image, 'rig/rgb', self._on_image)
        self.node.subscribe(AnnotatedAxisAligned2DBox_V, 'rig/boxes', self._on_boxes)

    @staticmethod
    def _stamp(msg):
        return round(msg.header.stamp.sec + msg.header.stamp.nsec * 1e-9, 3)

    def _on_image(self, msg):
        img = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3)
        self.image = {'t': self._stamp(msg), 'img': cv2.cvtColor(img, cv2.COLOR_RGB2BGR)}

    def _on_boxes(self, msg):
        self.boxes = {'t': self._stamp(msg), 'boxes': [
            (b.label, b.box.min_corner.x, b.box.min_corner.y,
             b.box.max_corner.x, b.box.max_corner.y) for b in msg.annotated_box]}

    def call(self, service, request, reqtype, timeout_ms=3000):
        from gz.msgs.boolean_pb2 import Boolean
        ok, _ = self.node.request(f'/world/{WORLD}/{service}', request, reqtype, Boolean,
                                  timeout_ms)
        return ok

    def set_pose(self, name, x, y, z, yaw, roll=0.0, pitch=0.0):
        from gz.msgs.pose_pb2 import Pose
        p = Pose()
        p.name = name
        p.position.x, p.position.y, p.position.z = x, y, z
        cr, sr = math.cos(roll / 2), math.sin(roll / 2)
        cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
        cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
        p.orientation.w = cr * cp * cy + sr * sp * sy
        p.orientation.x = sr * cp * cy - cr * sp * sy
        p.orientation.y = cr * sp * cy + sr * cp * sy
        p.orientation.z = cr * cp * sy - sr * sp * cy
        return self.call('set_pose', p, Pose)

    def light(self, intensity, direction):
        from gz.msgs.light_pb2 import Light
        light = Light()
        light.name = 'sun'
        light.type = Light.DIRECTIONAL
        light.diffuse.r = light.diffuse.g = light.diffuse.b = intensity
        light.diffuse.a = 1.0
        light.specular.r = light.specular.g = light.specular.b = 0.2 * intensity
        light.direction.x, light.direction.y, light.direction.z = direction
        light.cast_shadows = False
        return self.call('light_config', light, Light)

    def fresh(self, after, timeout=20.0):
        """An image and its boxes from the same frame, rendered after time `after`."""
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if (self.image.get('t', -1) > after and self.boxes.get('t') == self.image['t']):
                return self.image['img'].copy(), list(self.boxes['boxes']), self.image['t']
            time.sleep(0.02)
        return None


def yolo_lines(boxes, ignored=None):
    """YOLO label lines of the boxes; labelled boxes thinner than MIN_BOX_PX go to
    `ignored` (v4: they are greyed out in the image, so an object without a label
    is never shown to the model as background)."""
    lines, kept = [], []
    for label, x0, y0, x1, y1 in boxes:
        x0, y0 = max(0.0, x0), max(0.0, y0)
        x1, y1 = min(WIDTH - 1.0, x1), min(HEIGHT - 1.0, y1)
        if label < 1 or label > len(CLASSES) or x1 <= x0 or y1 <= y0:
            continue
        if x1 - x0 < MIN_BOX_PX or y1 - y0 < MIN_BOX_PX:
            if ignored is not None:
                ignored.append((int(x0), int(y0), int(math.ceil(x1)) + 1, int(math.ceil(y1)) + 1))
            continue
        cx, cy = (x0 + x1) / 2 / WIDTH, (y0 + y1) / 2 / HEIGHT
        lines.append(f'{label - 1} {cx:.6f} {cy:.6f} {(x1 - x0) / WIDTH:.6f} '
                     f'{(y1 - y0) / HEIGHT:.6f}')
        kept.append([CLASSES[label - 1], round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)])
    return lines, kept


def place_scene(rng, space, pool, gz):
    """Rig on a free spot; 0-4 objects in front of it. Returns the scene record."""
    rx, ry = space.random_spot(rng, 0.45)
    ryaw = rng.uniform(-math.pi, math.pi)
    count = rng.choices([0, 1, 2, 3, 4], [0.15, 0.30, 0.25, 0.18, 0.12])[0]
    placed, used = [], set()
    for _ in range(count * 20):
        if len(placed) == count:
            break
        # Class first, then a model of it: every class comes up equally often.
        cls = rng.choice(CLASSES)
        choices = [i for i in pool if i['cls'] == cls and i['name'] not in used]
        if not choices:
            continue
        item = rng.choice(choices)
        dist = 0.6 + item['radius'] + rng.random() ** 1.5 * 7.0
        bearing = rng.uniform(-(HFOV / 2 + 0.15), HFOV / 2 + 0.15)
        x = rx + dist * math.cos(ryaw + bearing)
        y = ry + dist * math.sin(ryaw + bearing)
        if space.clearance(x, y) < item['radius']:
            continue
        if any(math.hypot(x - p['x'], y - p['y']) < item['radius'] + p['radius'] + 0.1
               for p in placed):
            continue
        placed.append(dict(name=item['name'], cls=item['cls'], radius=item['radius'],
                           x=x, y=y, yaw=rng.uniform(-math.pi, math.pi)))
        used.add(item['name'])
    # v2: unlabelled distractors (none in the --test pool).
    distractors = [i for i in pool if i['cls'] is None]
    want = len(placed) + (rng.choices([0, 1, 2], [0.5, 0.3, 0.2])[0] if distractors else 0)
    for _ in range(40):
        if len(placed) >= want:
            break
        choices = [i for i in distractors if i['name'] not in used]
        if not choices:
            break
        item = rng.choice(choices)
        dist = 0.6 + item['radius'] + rng.random() ** 1.5 * 7.0
        bearing = rng.uniform(-(HFOV / 2 + 0.15), HFOV / 2 + 0.15)
        x = rx + dist * math.cos(ryaw + bearing)
        y = ry + dist * math.sin(ryaw + bearing)
        if space.clearance(x, y) < item['radius'] or any(
                math.hypot(x - p['x'], y - p['y']) < item['radius'] + p['radius'] + 0.1
                for p in placed):
            continue
        placed.append(dict(name=item['name'], cls=None, radius=item['radius'],
                           x=x, y=y, yaw=rng.uniform(-math.pi, math.pi)))
        used.add(item['name'])
    # Park only what was out last time and is not used now (fewer requests to Gazebo).
    for i, item in enumerate(pool):
        if item['name'] in gz.out_now and item['name'] not in used:
            gz.set_pose(item['name'], 2.0 * i - len(pool), PARK_Y, 0.0, 0.0)
    gz.out_now = set(used)
    for p in placed:
        gz.set_pose(p['name'], p['x'], p['y'], 0.0, p['yaw'])
    # v3: the camera height, pitch and roll vary a little (load, bumps, mounting).
    height = CAMERA_HEIGHT_M + rng.uniform(-0.015, 0.015)
    pitch, roll = rng.uniform(-0.05, 0.05), rng.uniform(-0.035, 0.035)
    gz.set_pose('rig', rx, ry, height, ryaw, roll, pitch)
    intensity = rng.uniform(0.35, 1.2)
    gz.light(intensity, (rng.uniform(-0.6, 0.6), rng.uniform(-0.6, 0.6), -1.0))
    return dict(rig=[round(rx, 3), round(ry, 3), round(ryaw, 4)], light=round(intensity, 3),
                camera=[round(height, 3), round(pitch, 3), round(roll, 3)],
                objects=[{k: (round(v, 3) if isinstance(v, float) else v)
                          for k, v in p.items()} for p in placed])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--count', type=int, required=True)
    ap.add_argument('--test', action='store_true')
    ap.add_argument('--seed', type=int, default=31)
    ap.add_argument('--val-share', type=float, default=0.15)
    ap.add_argument('--worlds', type=int, default=8,
                    help='training: rebuild the world (new pallets, colours) this many times')
    ap.add_argument('--preview', action='store_true', help='also save images with boxes drawn')
    args = ap.parse_args()
    rng = random.Random(args.seed)
    out = Path(args.out).resolve()     # the world file names textures and models by full path
    splits = ('test',) if args.test else ('train', 'val')
    for split in splits:
        (out / 'images' / split).mkdir(parents=True, exist_ok=True)
        (out / 'labels' / split).mkdir(parents=True, exist_ok=True)
    if args.preview:
        (out / 'preview').mkdir(parents=True, exist_ok=True)

    worlds = args.worlds
    manifest, skipped, started, pools, envs = [], 0, time.monotonic(), [], []
    space = FreeSpace()
    n = 0
    for w in range(worlds):
        pool = build_pool(rng, args.test, out / 'models' / f'world_{w}')
        pools.append([{k: v for k, v in p.items() if k not in ('sdf', 'actor')} for p in pool])
        world = out / f'capture_world_{w}.sdf'
        textures = pool_textures.get(id(pool)) or make_textures(rng, out / 'models' / f'world_{w}' / 'textures')
        env = world_env(rng, w, args.test, textures)
        envs.append({k: v for k, v in env.items() if k != 'racks'})
        capture_world(pool, world, env)
        chunk = args.count * (w + 1) // worlds - n
        print(f'world {w + 1}/{worlds} ({env["desc"]}): {len(pool)} objects, {chunk} images',
              flush=True)
        done = 0
        while done < chunk:                  # restarts after a stall continue where it stopped
            s, d = run_world(args, rng, out, world, pool, space, n + done, chunk - done,
                             manifest, started)
            skipped, done = skipped + s, done + d
        n += chunk
    (out / 'manifest.json').write_text(json.dumps(
        {'samples': manifest, 'classes': CLASSES, 'seed': args.seed,
         'held_out_test': args.test, 'models': sorted({p['path'] for p in pools[0]}),
         'pools': pools, 'worlds': envs, 'generator': 'v4'}, indent=1))
    if not args.test:
        (out / 'data.yaml').write_text(
            'path: .\ntrain: images/train\nval: images/val\nnames:\n' +
            ''.join(f'  {i}: {c}\n' for i, c in enumerate(CLASSES)))
    print(f'{len(manifest)} images in {out}, {skipped} skipped')


def run_world(args, rng, out, world, pool, space, first, count, manifest, started):
    """Start Gazebo with one capture world and save `count` images; returns skips."""
    # model://tugger_gazebo/... (the station floor markers) resolves from the install space.
    # Some Fuel meshes name their texture by file name only (Cardboard box,
    # Survivor Male): their texture folders go on the search path too.
    textures = sorted({str(cached_model(p['path']) / 'materials/textures') for p in pool
                       if not p['path'].startswith('parts/')
                       and (cached_model(p['path']) / 'materials/textures').is_dir()})
    resources = ':'.join(filter(None, [str(WS / 'install/tugger_gazebo/share'), *textures,
                                       os.environ.get('GZ_SIM_RESOURCE_PATH')]))
    env = dict(os.environ, GZ_PARTITION=f'obstacle_dataset_{os.getpid()}', GZ_IP='127.0.0.1',
               GZ_SIM_RESOURCE_PATH=resources)
    os.environ.update(GZ_PARTITION=env['GZ_PARTITION'], GZ_IP='127.0.0.1')
    log = open(out / f'gazebo_{world.stem}.log', 'w')
    sim = subprocess.Popen(['gz', 'sim', '-s', '-r', '--headless-rendering', str(world)],
                           env=env, stdout=log, stderr=subprocess.STDOUT)
    skipped = 0
    try:
        gz = Gazebo()
        if gz.fresh(-1.0, timeout=240.0) is None:
            sys.exit(f'no camera image from Gazebo (see {log.name})')
        misses = 0
        for n in range(first, first + count):
            scene = place_scene(rng, space, pool, gz)
            changed = gz.image.get('t', 0.0)
            frame = gz.fresh(changed + 0.25)
            if frame is None:
                skipped += 1
                misses += 1
                if misses >= 2:              # Gazebo stopped answering: restart it (caller)
                    print(f'Gazebo stalled at image {n}: restarting the world', flush=True)
                    return skipped, n + 1 - first
                continue
            misses = 0
            img, boxes, stamp = frame
            img = degrade(img, rng)
            ignored = []
            lines, kept = yolo_lines(boxes, ignored)
            for x0, y0, x1, y1 in ignored:
                img[y0:y1, x0:x1] = IGNORE_GREY
            split = 'test' if args.test else ('val' if rng.random() < args.val_share else 'train')
            stem = f'{n:05d}'
            # v3: JPEG (quality 95): a quarter of the size of PNG with camera noise.
            cv2.imwrite(str(out / 'images' / split / f'{stem}.jpg'), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            (out / 'labels' / split / f'{stem}.txt').write_text('\n'.join(lines))
            manifest.append(dict(scene, file=f'{split}/{stem}.jpg', boxes=kept, stamp=stamp,
                                 world=world.stem, greyed=ignored))
            if args.preview:
                vis = img.copy()
                for cls, x0, y0, x1, y1 in kept:
                    cv2.rectangle(vis, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 255), 1)
                    cv2.putText(vis, cls, (int(x0) + 2, int(y0) + 12), cv2.FONT_HERSHEY_SIMPLEX,
                                0.4, (0, 255, 255), 1)
                cv2.imwrite(str(out / 'preview' / f'{stem}.jpg'), vis)
            if (n + 1) % 50 == 0:
                rate = len(manifest) / (time.monotonic() - started)
                print(f'{n + 1}/{args.count} images, {rate:.2f}/s, {skipped} skipped', flush=True)
    finally:
        sim.terminate()
        try:
            sim.wait(timeout=20)
        except subprocess.TimeoutExpired:
            sim.kill()
        subprocess.run(['pkill', '-f', str(world)], check=False)
        log.close()
    return skipped, count


if __name__ == '__main__':
    main()
