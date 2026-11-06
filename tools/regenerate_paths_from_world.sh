#!/usr/bin/env bash
# Rebuild the planning map and every path from the Gazebo world file.
#
# Run after changing the environment. Every step is a generator, not a hand
# edit; nothing here commands the robot. Regenerated routes and approaches are
# NOT approved for execution: review them, then approve them as before.
#
# Usage: tools/regenerate_paths_from_world.sh
set -euo pipefail
WS="$(cd "$(dirname "$0")/.." && pwd)"
cd "$WS"
echo "1/7 planning map from the world file"
python3 tools/warehouse_validation/export_static_map.py
echo "2/7 station geometry from the world and robot files"
python3 tools/v1_manhattan_bezier/audit_cargo_geometry.py
echo "3/7 route database (mission chains decide departure gates)"
python3 tools/v1_manhattan_bezier/build_route_database_v1.py \
    --map src/tugger_navigation/maps/warehouse_map.yaml \
    --geometry tools/v1_manhattan_bezier/geometry/cargo_geometry.json \
    --output tools/v1_manhattan_bezier/routes
echo "4/7 routes the forward search cannot build, by reversing their opposite"
(cd tools/v1_manhattan_bezier && python3 complete_reverse_routes.py \
    --database routes/route_database.json \
    --map ../../src/tugger_navigation/maps/warehouse_map.yaml) || true
echo "5/7 segment 2 approaches for every station gate"
NEW="$(mktemp --suffix=.json)"; rm -f "$NEW"
(cd tools/mission_approaches && python3 generate_candidates.py --output "$NEW")
mv "$NEW" tools/mission_approaches/entrance_candidates.json
echo "6/7 single-segment mission paths"
python3 tools/v1_manhattan_bezier/build_mission_paths.py
echo "7/7 3D audit of every mission path and station lift"
python3 tools/v1_manhattan_bezier/audit_mission_paths.py
echo
echo "Done. Regenerated paths are not approved for execution yet."
