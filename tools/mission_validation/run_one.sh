#!/usr/bin/env bash
# One headless simulation in the isolated domain, then shut it down.
#   tools/mission_validation/run_one.sh OUT_DIR NAME DESTINATION... [runner options]
# Writes OUT_DIR/NAME.json (+ .log, .control.log, .amcl.log). Set
# TUGGER_SIM_ARGS for fault testing (see tools/fault_testing/README.md).
OUT=$(mkdir -p "$1" && cd "$1" && pwd); NAME=$2; shift 2
WS="$(cd "$(dirname "$0")/../.." && pwd)"; D=${TUGGER_DOMAIN:-77}
export TUGGER_VALIDATION_DIR=$OUT/val
cd "$WS"; set +u; source /opt/ros/lyrical/setup.bash; source install/setup.bash
export GZ_IP=127.0.0.1 QT_QPA_PLATFORM=offscreen RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI='<CycloneDDS><Domain><Discovery><ParticipantIndex>auto</ParticipantIndex><MaxAutoParticipantIndex>200</MaxAutoParticipantIndex></Discovery></Domain></CycloneDDS>'
tools/mission_validation/isolated_stack.sh down $D
tools/mission_validation/isolated_stack.sh up $D
for _ in $(seq 1 300); do [ -f "$TUGGER_VALIDATION_DIR/domain_$D/nav2_ready" ] && break; sleep 2; done
[ -f "$TUGGER_VALIDATION_DIR/domain_$D/nav2_ready" ] || { echo STACK_NOT_READY; tools/mission_validation/isolated_stack.sh down $D; exit 3; }
# Wall-clock limit for the whole run (the per-leg limit is in simulated time);
# raise TUGGER_RUN_WALL_S when a Gazebo window slows the simulation down.
ROS_DOMAIN_ID=$D GZ_PARTITION=tugger_validation_$D timeout ${TUGGER_RUN_WALL_S:-3000} python3 tools/mission_validation/mission_runner.py \
    "$@" --output "$OUT/$NAME.json" --timeout 900 > "$OUT/$NAME.log" 2>&1
echo runner_exit=$?
for f in control amcl; do cp "$TUGGER_VALIDATION_DIR/domain_$D/$f.log" "$OUT/$NAME.$f.log" 2>/dev/null; done
tools/mission_validation/isolated_stack.sh down $D
grep -E '^\S+ (SUCCESS|FAILED|TIMEOUT|NO_PLAN|CANCELLED)|tracking:' "$OUT/$NAME.log"
