#!/usr/bin/env python3
"""The warehouse world with a worn red frame at one station (validation).

Two opposite sides of the station's red frame are covered in floor colour, as
if worn off. The frame fit refuses that case (one side or two opposite sides
left), so the docking can only use the position from the tags
(marker_reader.frame_from_tags, 2026-11-18). Two sides keep enough red in view
for the reader's red-pixel gate (3,000 px; a full frame shows about 7,800).

Usage: worn_marker_world.py OUT.sdf [--marker qr_station_1] [--sides x|y]
Then: TUGGER_SIM_ARGS="world:=OUT.sdf" tools/mission_validation/run_one.sh ...
"""
import argparse
import re
from pathlib import Path

WS = Path(__file__).resolve().parents[2]
WORLD = WS / 'src/tugger_gazebo/worlds/tugger_warehouse.sdf'
FLOOR = '0.46 0.48 0.50 1'
FRAME, PLATE = 0.12, 0.10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--marker', default='qr_station_1')
    ap.add_argument('--sides', choices=('x', 'y'), default='x',
                    help='x: the two sides across x (left/right in the plate) are worn')
    args = ap.parse_args()
    text = WORLD.read_text()
    strip = (FRAME - PLATE) / 2 + 0.002                    # the red band plus a little floor
    off = PLATE / 2 + strip / 2 - 0.002
    patches = []
    for k, s in enumerate((-1, 1)):
        x, y, sx, sy = (s * off, 0, strip, FRAME + 0.01) if args.sides == 'x' else (0, s * off, FRAME + 0.01, strip)
        patches.append(f'<visual name="worn_{k}"><pose>{x:.4f} {y:.4f} 0.0005 0 0 0</pose>'
                       '<cast_shadows>false</cast_shadows><geometry><plane><normal>0 0 1</normal>'
                       f'<size>{sx:.4f} {sy:.4f}</size></plane></geometry><material><ambient>{FLOOR}</ambient>'
                       f'<diffuse>{FLOOR}</diffuse></material></visual>')
    pattern = r'(<model name="%s">.*?<link name="link">)' % re.escape(args.marker)
    text, n = re.subn(pattern, lambda m: m.group(1) + ''.join(patches), text, count=1, flags=re.S)
    if n != 1:
        raise SystemExit(f'marker {args.marker} not found')
    Path(args.out).write_text(text)
    print(f'{args.out}: {args.marker} with sides {args.sides} worn')


if __name__ == '__main__':
    main()
