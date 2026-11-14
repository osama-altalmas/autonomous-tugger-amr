#!/usr/bin/env bash
# Path tracking and localization view in RViz, in one command:
# starts localization_view.py (trails and correction ticks) and RViz with the
# ready-made layout tracking.rviz; closing RViz stops both.
# Extra arguments go to localization_view.py, e.g. --scale 20.
cd "$(dirname "$0")/../.." || exit 1
set +u
source /opt/ros/lyrical/setup.bash
source install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI='<CycloneDDS><Domain><Discovery><ParticipantIndex>auto</ParticipantIndex><MaxAutoParticipantIndex>200</MaxAutoParticipantIndex></Discovery></Domain></CycloneDDS>'
python3 tools/mission_view/localization_view.py "$@" &
VIEW=$!
trap 'kill $VIEW 2>/dev/null' EXIT
rviz2 -d tools/mission_view/tracking.rviz --ros-args -p use_sim_time:=true
