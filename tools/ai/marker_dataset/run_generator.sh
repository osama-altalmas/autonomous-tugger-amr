#!/usr/bin/env bash
# run_generator.sh GENERATOR.py ARGS... : start Gazebo only (isolated domain/partition),
# run the dataset generator against it, then stop Gazebo.
WS="$(cd "$(dirname "$0")/../../.." && pwd)"; D=${TUGGER_DOMAIN:-77}
GEN=$1; shift
cd "$WS"; set +u; source /opt/ros/lyrical/setup.bash; source install/setup.bash
export ROS_DOMAIN_ID=$D GZ_PARTITION=tugger_validation_$D GZ_IP=127.0.0.1 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI='<CycloneDDS><Domain><Discovery><ParticipantIndex>auto</ParticipantIndex><MaxAutoParticipantIndex>200</MaxAutoParticipantIndex></Discovery></Domain></CycloneDDS>'
mkdir -p ai_data/logs
setsid ros2 launch tugger_gazebo simulation.launch.py gz_extra_args:="-s --headless-rendering" \
    > ai_data/logs/generator_gazebo.log 2>&1 &
PGID=$!
sleep 25
python3 "$GEN" "$@"
STATUS=$?
kill -INT -- -$PGID 2>/dev/null; sleep 8; kill -KILL -- -$PGID 2>/dev/null
exit $STATUS
