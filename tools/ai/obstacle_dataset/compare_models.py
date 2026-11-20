#!/usr/bin/env python3
"""Score several obstacle models on the same held-out tests (laptop CPU), side by side.

For each model and test: mAP50 / mAP50-95 (Ultralytics val) and recall by box height
(recall_by_size.py, confidence 0.25, IoU 0.5). Results in one JSON file.

Usage: compare_models.py --out ai_data/models/obstacles_v4_n/compare_local.json \
    --model v3=ai_data/models/obstacles_v3/best.pt --model v4n=ai_data/models/obstacles_v4_n/.../best.pt \
    --test loco_v4=ai_data/loco_yolo_v4 --test loco_v3form=ai_data/loco_yolo ...
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

if not hasattr(np, 'trapz'):
    np.trapz = np.trapezoid
HERE = Path(__file__).resolve().parent
NAMES = {0: 'person', 1: 'pallet', 2: 'cart', 3: 'forklift'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', action='append', required=True, help='name=weights.pt')
    ap.add_argument('--test', action='append', required=True, help='name=dataset dir (split test)')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    from ultralytics import YOLO
    out = Path(args.out)
    result = json.loads(out.read_text()) if out.exists() else {}
    for m in args.model:
        mname, weights = m.split('=', 1)
        model = YOLO(weights)
        for t in args.test:
            tname, data = t.split('=', 1)
            key = f'{mname}|{tname}'
            if key in result:
                continue
            data = Path(data).resolve()
            with tempfile.NamedTemporaryFile('w', suffix='.yaml', delete=False) as y:
                y.write(f'path: {data}\ntrain: images/test\nval: images/test\ntest: images/test\n'
                        f'names: {NAMES}\n')
            v = model.val(data=y.name, split='test', imgsz=640, batch=16, device='cpu',
                          plots=False, verbose=False)
            per = {NAMES[int(c)]: round(float(v.box.maps[int(c)]), 3) for c in v.box.ap_class_index}
            size_json = out.parent / f'size_{mname}_{tname}.json'
            subprocess.run([sys.executable, str(HERE / 'recall_by_size.py'), '--data', str(data),
                            '--split', 'test', '--model', weights, '--out', str(size_json)], check=True)
            result[key] = {'mAP50': round(float(v.box.map50), 3), 'mAP50-95': round(float(v.box.map), 3),
                           'per_class_mAP50-95': per, 'by_size': json.loads(size_json.read_text())}
            out.write_text(json.dumps(result, indent=1))
            print(key, result[key]['mAP50'], result[key]['mAP50-95'], flush=True)


if __name__ == '__main__':
    main()
