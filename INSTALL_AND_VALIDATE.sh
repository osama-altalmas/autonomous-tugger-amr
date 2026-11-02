#!/usr/bin/env bash
set -eo pipefail
cd "$HOME/Autonomous_Tugger_Project/autonomous_tugger_ws"
cp src/tugger_control/tugger_control/four_ws_controller.py src/tugger_control/tugger_control/four_ws_controller.py.before_smoothing
cp -f motion_update/src/tugger_control/tugger_control/four_ws_controller.py src/tugger_control/tugger_control/four_ws_controller.py
cp -f motion_update/src/tugger_control/tugger_control/motion_math.py src/tugger_control/tugger_control/motion_math.py
mkdir -p tools/motion_validation
cp -f motion_update/tools/motion_validation/test_motion_math.py tools/motion_validation/test_motion_math.py
python3 -m py_compile src/tugger_control/tugger_control/four_ws_controller.py src/tugger_control/tugger_control/motion_math.py
python3 tools/motion_validation/test_motion_math.py
set +u
source /opt/ros/lyrical/setup.bash
source install/setup.bash
colcon build --packages-select tugger_control --symlink-install
printf '\nDONE: Restart tugger_control control.launch.py to load new controller.\n'
