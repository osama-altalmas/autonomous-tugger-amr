#!/usr/bin/env bash
set -eo pipefail
cd "$HOME/Autonomous_Tugger_Project/autonomous_tugger_ws"
set +u
source /opt/ros/lyrical/setup.bash
source install/setup.bash
python3 -m py_compile tools/warehouse_navigation/live_graph_mission.py
echo "Live graph mission installed and syntax checked."
echo "Dry run: python3 tools/warehouse_navigation/live_graph_mission.py --goal H2_2"
echo "Only after clearance verification, execute with --execute"
