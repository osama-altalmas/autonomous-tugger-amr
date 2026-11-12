#!/usr/bin/env python3
"""Write the generated staging poses into the operator station config.

operator_stations.json lists each station's entrances and their staging
poses; the approach generator and the operator panel read it. The poses are
generator output (route_database.json nodes), so they are copied from there
instead of being edited by hand. Only the poses of entrances already listed
change; stations and entrances are neither added nor removed.

Usage: sync_station_entries.py [--check]
  --check  exit 1 (and change nothing) when the config differs.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATABASE = ROOT / 'tools/v1_manhattan_bezier/routes/route_database.json'
STATIONS = ROOT / 'src/tugger_control/config/operator_stations.json'


def synced(config, database):
    """Return (new config, list of changes)."""
    nodes = database.get('nodes', {})
    changes = []
    result = json.loads(json.dumps(config))
    for station, record in result.get('stations', {}).items():
        for candidate in record.get('candidates', []):
            node = nodes.get(station, {}).get(candidate['entrance'])
            if not node or not node.get('available'):
                raise ValueError(f'{station} {candidate["entrance"]}: no available '
                                 'generated staging pose')
            pose = [float(value) for value in node['pose']]
            if any(abs(a - b) > 1e-9 for a, b in zip(pose, candidate['pose'])):
                changes.append(f'{station} {candidate["entrance"]}: '
                               f'{[round(v, 4) for v in candidate["pose"]]} -> '
                               f'{[round(v, 4) for v in pose]}')
                candidate['pose'] = pose
    return result, changes


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    config = json.loads(STATIONS.read_text(encoding='utf-8'))
    database = json.loads(DATABASE.read_text(encoding='utf-8'))
    result, changes = synced(config, database)
    for line in changes:
        print(line)
    if args.check:
        return 1 if changes else 0
    if changes:
        STATIONS.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(f'{len(changes)} staging pose(s) updated')
    return 0


if __name__ == '__main__':
    sys.exit(main())
