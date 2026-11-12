#!/usr/bin/env bash
# Start a headless simulation stack that cannot interact with a normal session.
#
# It uses its own ROS domain and Gazebo partition, so a GUI stack on the default
# domain keeps running untouched. Only the process group started here is stopped.
#
# Usage: isolated_stack.sh up|down [domain_id]
set -u
ACTION=${1:?up or down}
DOMAIN=${2:-77}
WS="$(cd "$(dirname "$0")/../.." && pwd)"
RUN_DIR=${TUGGER_VALIDATION_DIR:-/tmp/tugger_validation}/domain_$DOMAIN
PIDFILE="$RUN_DIR/pgid"

stop_group() {
    [ -f "$PIDFILE" ] || return 0
    local pgid
    pgid=$(cat "$PIDFILE")
    kill -INT -- "-$pgid" 2>/dev/null
    for _ in $(seq 1 20); do
        kill -0 -- "-$pgid" 2>/dev/null || break
        sleep 0.5
    done
    kill -KILL -- "-$pgid" 2>/dev/null
    rm -f "$PIDFILE"
    # Killed Fast DDS processes leave shared-memory segments and ports behind,
    # which can block discovery between later processes. This removes only
    # zombie files; segments of running processes are kept.
    (set +u; source /opt/ros/lyrical/setup.bash; fastdds shm clean) >/dev/null 2>&1
}

if [ "$ACTION" = down ]; then
    stop_group
    exit 0
fi

mkdir -p "$RUN_DIR"
stop_group
rm -f "$RUN_DIR/nav2_ready"
unset AMENT_PREFIX_PATH COLCON_PREFIX_PATH CMAKE_PREFIX_PATH PYTHONPATH
set +u
source /opt/ros/lyrical/setup.bash
source "$WS/install/setup.bash"
set -u
export ROS_DOMAIN_ID=$DOMAIN GZ_PARTITION=tugger_validation_$DOMAIN
export GZ_IP=127.0.0.1 IGN_IP=127.0.0.1 ROS_LOG_DIR="$RUN_DIR/ros_log"
# Cyclone DDS, as Nav2 recommends: Fast DDS intermittently failed to match
# Nav2 servers with their clients at startup ("action server not available").
export RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}
# Cyclone DDS gives each process on a machine a participant index and by
# default allows only a few; the full stack has more processes than that, so
# a late starter (often Gazebo) fails with "no free participant index".
export CYCLONEDDS_URI=${CYCLONEDDS_URI:-'<CycloneDDS><Domain><Discovery><ParticipantIndex>auto</ParticipantIndex><MaxAutoParticipantIndex>200</MaxAutoParticipantIndex></Discovery></Domain></CycloneDDS>'}

setsid bash -c '
  echo $$ > "$0"
  ros2 launch tugger_gazebo simulation.launch.py \
      gz_extra_args:="-s --headless-rendering $3" > "$1/gazebo.log" 2>&1 &
  sleep 15
  ros2 launch tugger_control control.launch.py > "$1/control.log" 2>&1 &
  ros2 launch tugger_ground_truth simulation_localization.launch.py > "$1/truth.log" 2>&1 &
  ros2 run nav2_map_server map_server --ros-args -r __node:=map_server \
      -p "yaml_filename:=$2/src/tugger_navigation/maps/warehouse_map.yaml" \
      -p use_sim_time:=true > "$1/map.log" 2>&1 &
  for _ in $(seq 1 60); do
    ros2 lifecycle set /map_server configure >> "$1/map.log" 2>&1 && break
    sleep 2
  done
  for _ in $(seq 1 60); do
    ros2 lifecycle set /map_server activate >> "$1/map.log" 2>&1
    ros2 lifecycle get /map_server 2>/dev/null | grep -q "^active" && break
    sleep 2
  done
  # Real localization (Phase B3): AMCL on the merged safety-scanner scan.
  ros2 launch tugger_navigation localization.launch.py > "$1/amcl.log" 2>&1 &
  # Nav2 bring-up can stall on a lost lifecycle response. Retry only Nav2.
  # These three are the only Nav2 servers launched; missions call them directly.
  for attempt in 1 2 3 4; do
    ros2 launch tugger_navigation nav2.launch.py > "$1/nav2_$attempt.log" 2>&1 &
    nav2=$!
    deadline=$((SECONDS + 120))
    while [ "$SECONDS" -lt "$deadline" ]; do
      sleep 5
      ros2 lifecycle get /planner_server 2>/dev/null | grep -q "^active" &&
        ros2 lifecycle get /controller_server 2>/dev/null | grep -q "^active" &&
        ros2 lifecycle get /docking/controller_server 2>/dev/null | grep -q "^active" &&
        { echo "$attempt" > "$1/nav2_ready"; break 2; }
    done
    kill -INT "$nav2" 2>/dev/null
    sleep 8
    kill -KILL "$nav2" 2>/dev/null
  done
  wait
' "$PIDFILE" "$RUN_DIR" "$WS" "${TUGGER_GZ_EXTRA:-}" > /dev/null 2>&1 &
disown
echo "started domain=$DOMAIN logs=$RUN_DIR"
