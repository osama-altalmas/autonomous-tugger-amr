#!/usr/bin/env python3
"""The warehouse world with one real-looking object on a route (validation of the
obstacle recognition, AI, AMR-V3).

The scene_objects.py test objects are plain boxes; the colour cameras' model
would rightly not call those people. This puts a Gazebo Fuel model (the same
models the training images were made from) standing on the route, so the
recognition and the policy can be checked end to end:
  person  Fuel 'Standing person'  -> waited for (no detour after 5 s), then
          another way after 30 s
  pallet  Fuel 'Euro pallet'      -> waited 5 s, then driven around (as before)

Usage: obstacle_world.py OUT.sdf --kind person|pallet [--x 10 --y 7 --yaw 3.14]
       obstacle_world.py OUT.sdf --object pallet:13:7.8:0 --object person:17.5:0 ...
(HOME -> cargo_station_3 runs up x = 4.5, then east along y = 7: (10, 7) is on it.)
Then:  TUGGER_SIM_ARGS="world:=OUT.sdf" tools/mission_validation/run_one.sh OUT_DIR NAME ...
"""
import argparse
import math
from pathlib import Path

WS = Path(__file__).resolve().parents[2]
WORLD = WS / 'src/tugger_gazebo/worlds/tugger_warehouse.sdf'
# The Fuel address, resolved by Gazebo from its local cache (as generate.py does).
# A file:// path to the cache does not work: its folder names contain '%20', which
# Gazebo decodes to a space and then silently leaves the model out (seen 2026-11-20).
FUEL = 'https://fuel.gazebosim.org/1.0/OpenRobotics/models'
CACHE = Path.home() / '.gz/fuel/fuel.gazebosim.org/openrobotics/models'
MODELS = {'person': 'Standing%20person', 'pallet': 'Euro%20pallet'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--kind', choices=sorted(MODELS), help='one object (with --x --y --yaw)')
    ap.add_argument('--x', type=float, default=10.0)
    ap.add_argument('--y', type=float, default=7.0)
    ap.add_argument('--yaw', type=float, default=math.pi, help='person faces the robot coming from HOME')
    ap.add_argument('--object', action='append', default=[], metavar='KIND:X:Y[:YAW]',
                    help='several objects, e.g. --object pallet:13:7.8:0 --object person:17.5:0')
    args = ap.parse_args()
    objects = []
    if args.kind:
        objects.append((args.kind, args.x, args.y, args.yaw))
    for spec in args.object:
        kind, x, y, *yaw = spec.split(':')
        if kind not in MODELS:
            raise SystemExit(f'unknown kind {kind}: {sorted(MODELS)}')
        objects.append((kind, float(x), float(y), float(yaw[0]) if yaw else math.pi))
    if not objects:
        raise SystemExit('give --kind or at least one --object')
    includes = []
    for k, (kind, x, y, yaw) in enumerate(objects):
        if not (CACHE / MODELS[kind].lower()).exists():
            raise SystemExit(f'model not downloaded: {MODELS[kind]}')
        includes.append(f'<include><uri>{FUEL}/{MODELS[kind]}</uri><name>test_{kind}_{k}</name>'
                        f'<static>true</static><pose>{x} {y} 0 0 0 {yaw}</pose></include>')
    text = WORLD.read_text()
    marker = '</world>'
    if text.count(marker) != 1:
        raise SystemExit('unexpected world file')
    Path(args.out).write_text(text.replace(marker, '\n'.join(includes) + '\n' + marker))
    print(f'{args.out}: ' + ', '.join(f'{k} at ({x}, {y})' for k, x, y, _ in objects))


if __name__ == '__main__':
    main()
