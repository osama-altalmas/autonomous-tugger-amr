#!/usr/bin/env python3
# Frozen copy: the generator as used for obstacles_v2 (2026-11-16). v3 and later: generate.py.
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
MIN_BOX_PX = 6
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
    ('OpenRobotics/models/Forklift', 'forklift', 1.40, 'both'),
    ('OpenRobotics/models/Pallet_Jack', 'forklift', 0.90, 'both'),
    # v2: more carts (none resembles the held-out BPCart or ParkingTrolleyMin).
    ('OpenRobotics/models/Cart: rigid suspension', 'cart', 0.70, 'train'),
    ('OpenRobotics/models/MiRCart', 'cart', 0.60, 'train'),
    ('OpenRobotics/models/TrolleyBox2', 'cart', 0.60, 'train'),
    ('OpenRobotics/models/SurgicalTrolley', 'cart', 0.60, 'train'),
    ('OpenRobotics/models/MopCart1', 'cart', 0.60, 'train'),
    # v2: unlabelled distractors (no class: the model must learn they are none).
    ('OpenRobotics/models/Cardboard box', None, 0.40, 'train'),
    ('OpenRobotics/models/Construction Cone', None, 0.30, 'train'),
    ('OpenRobotics/models/Construction Barrel', None, 0.45, 'train'),
    ('OpenRobotics/models/TrashBin', None, 0.40, 'train'),
    ('OpenRobotics/models/Large Crate', None, 0.70, 'train'),
]
COPIES = 2                     # instances of each model, so a scene may hold two
# v2: models that also come as a recoloured copy (one copy original, one recoloured).
RECOLOUR = {'cart', 'forklift'}
RECOLOUR_PATHS = {'OpenRobotics/models/Euro pallet'}
PALLETS_PER_WORLD = 10
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


def capture_world(pool, out_sdf):
    """The warehouse world plus the camera rig and the parked, labelled pool."""
    text = WORLD_FILE.read_text()
    # The collision-scene plugin serves the robot stack only.
    text = re.sub(r'<plugin filename="tugger_collision_scene"[^>]*/>', '', text)
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
    if not test:
        for k in range(PALLETS_PER_WORLD):
            sdf, radius, desc = pallet_parts(rng)
            pool.append(dict(path=f'parts/pallet_{k}', cls='pallet', radius=radius,
                             name=f'built_pallet_{k}', label=CLASSES.index('pallet') + 1,
                             actor=False, sdf=sdf, built=desc))
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

    def set_pose(self, name, x, y, z, yaw):
        from gz.msgs.pose_pb2 import Pose
        p = Pose()
        p.name = name
        p.position.x, p.position.y, p.position.z = x, y, z
        p.orientation.z, p.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
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


def yolo_lines(boxes):
    lines, kept = [], []
    for label, x0, y0, x1, y1 in boxes:
        x0, y0 = max(0.0, x0), max(0.0, y0)
        x1, y1 = min(WIDTH - 1.0, x1), min(HEIGHT - 1.0, y1)
        if label < 1 or label > len(CLASSES) or x1 - x0 < MIN_BOX_PX or y1 - y0 < MIN_BOX_PX:
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
    for i, item in enumerate(pool):
        if item['name'] not in used:
            gz.set_pose(item['name'], 2.0 * i - len(pool), PARK_Y, 0.0, 0.0)
    for p in placed:
        gz.set_pose(p['name'], p['x'], p['y'], 0.0, p['yaw'])
    gz.set_pose('rig', rx, ry, CAMERA_HEIGHT_M, ryaw)
    intensity = rng.uniform(0.35, 1.2)
    gz.light(intensity, (rng.uniform(-0.6, 0.6), rng.uniform(-0.6, 0.6), -1.0))
    return dict(rig=[round(rx, 3), round(ry, 3), round(ryaw, 4)], light=round(intensity, 3),
                objects=[{k: (round(v, 3) if isinstance(v, float) else v)
                          for k, v in p.items()} for p in placed])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--count', type=int, required=True)
    ap.add_argument('--test', action='store_true')
    ap.add_argument('--seed', type=int, default=31)
    ap.add_argument('--val-share', type=float, default=0.15)
    ap.add_argument('--worlds', type=int, default=4,
                    help='training: rebuild the world (new pallets, colours) this many times')
    ap.add_argument('--preview', action='store_true', help='also save images with boxes drawn')
    args = ap.parse_args()
    rng = random.Random(args.seed)
    out = Path(args.out)
    splits = ('test',) if args.test else ('train', 'val')
    for split in splits:
        (out / 'images' / split).mkdir(parents=True, exist_ok=True)
        (out / 'labels' / split).mkdir(parents=True, exist_ok=True)
    if args.preview:
        (out / 'preview').mkdir(parents=True, exist_ok=True)

    worlds = 1 if args.test else args.worlds
    manifest, skipped, started, pools = [], 0, time.monotonic(), []
    space = FreeSpace()
    n = 0
    for w in range(worlds):
        pool = build_pool(rng, args.test, out / 'models' / f'world_{w}')
        pools.append([{k: v for k, v in p.items() if k not in ('sdf', 'actor')} for p in pool])
        world = out / f'capture_world_{w}.sdf'
        capture_world(pool, world)
        chunk = args.count * (w + 1) // worlds - n
        print(f'world {w + 1}/{worlds}: {len(pool)} objects, {chunk} images', flush=True)
        skipped += run_world(args, rng, out, world, pool, space, n, chunk, manifest, started)
        n += chunk
    (out / 'manifest.json').write_text(json.dumps(
        {'samples': manifest, 'classes': CLASSES, 'seed': args.seed,
         'held_out_test': args.test, 'models': sorted({p['path'] for p in pools[0]}),
         'pools': pools}, indent=1))
    if not args.test:
        (out / 'data.yaml').write_text(
            'path: .\ntrain: images/train\nval: images/val\nnames:\n' +
            ''.join(f'  {i}: {c}\n' for i, c in enumerate(CLASSES)))
    print(f'{len(manifest)} images in {out}, {skipped} skipped')


def run_world(args, rng, out, world, pool, space, first, count, manifest, started):
    """Start Gazebo with one capture world and save `count` images; returns skips."""
    # model://tugger_gazebo/... (the station floor markers) resolves from the install space.
    resources = ':'.join(filter(None, [str(WS / 'install/tugger_gazebo/share'),
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
        for n in range(first, first + count):
            scene = place_scene(rng, space, pool, gz)
            changed = gz.image.get('t', 0.0)
            frame = gz.fresh(changed + 0.25)
            if frame is None:
                skipped += 1
                continue
            img, boxes, stamp = frame
            lines, kept = yolo_lines(boxes)
            split = 'test' if args.test else ('val' if rng.random() < args.val_share else 'train')
            stem = f'{n:05d}'
            cv2.imwrite(str(out / 'images' / split / f'{stem}.png'), img)
            (out / 'labels' / split / f'{stem}.txt').write_text('\n'.join(lines))
            manifest.append(dict(scene, file=f'{split}/{stem}.png', boxes=kept, stamp=stamp,
                                 world=world.stem))
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
    return skipped


if __name__ == '__main__':
    main()
