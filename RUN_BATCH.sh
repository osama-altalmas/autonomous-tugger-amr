#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/Autonomous_Tugger_Project/autonomous_tugger_ws"
set +u
source /opt/ros/lyrical/setup.bash
set -u
set +u
source install/setup.bash
set -u
python3 -m py_compile tools/warehouse_navigation/prepare_graph_mission.py
python3 tools/warehouse_navigation/prepare_graph_mission.py --goal H2_1 --smoothing 0.8
printf '\nRoute generated. Keep all four simulation launch terminals running.\n'
printf 'Inspect the route in RViz using:\n'
printf 'python3 tools/warehouse_navigation/follow_warehouse_route.py --route tools/warehouse_validation/results/route_graph_mission.json\n'
printf '\nExecute ONLY after restarting at (0,0,0) and verifying path clearance:\n'
printf 'python3 tools/warehouse_navigation/follow_warehouse_route.py --route tools/warehouse_validation/results/route_graph_mission.json --execute\n'
