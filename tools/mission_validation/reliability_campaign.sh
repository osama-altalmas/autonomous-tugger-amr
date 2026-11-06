#!/usr/bin/env bash
# Repeated full mission cycles on one isolated stack, each from a fresh stack
# and a random start near HOME. Prints one summary line per cycle.
#
# Usage: reliability_campaign.sh DOMAIN CYCLES OUTPUT_DIR [SEED]
set -u
DOMAIN=${1:?domain}
CYCLES=${2:?cycles}
OUT=${3:?output dir}
SEED=${4:-$DOMAIN}
WS="$(cd "$(dirname "$0")/../.." && pwd)"
RUN_DIR=${TUGGER_VALIDATION_DIR:-/tmp/tugger_validation}/domain_$DOMAIN
mkdir -p "$OUT"
cd "$WS"
set +u
source /opt/ros/lyrical/setup.bash
source install/setup.bash
set -u
export GZ_IP=127.0.0.1 QT_QPA_PLATFORM=offscreen
export RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}
for cycle in $(seq 1 "$CYCLES"); do
    tools/mission_validation/isolated_stack.sh down "$DOMAIN"
    rm -f "$RUN_DIR/nav2_ready"
    tools/mission_validation/isolated_stack.sh up "$DOMAIN" > "$OUT/up_${DOMAIN}_$cycle.log" 2>&1
    for _ in $(seq 1 300); do
        [ -f "$RUN_DIR/nav2_ready" ] && break
        sleep 2
    done
    if [ ! -f "$RUN_DIR/nav2_ready" ]; then
        echo "cycle $cycle domain $DOMAIN: STACK_NOT_READY"
        continue
    fi
    read -r X Y YAW < <(python3 -c "
import random; r = random.Random($SEED * 1000 + $cycle)
print(f'{r.uniform(-0.05, 0.05):.4f} {r.uniform(-0.05, 0.05):.4f} {r.uniform(-0.052, 0.052):.4f}')")
    ROS_DOMAIN_ID=$DOMAIN GZ_PARTITION=tugger_validation_$DOMAIN timeout 4200 \
        python3 tools/mission_validation/mission_runner.py \
        cargo_station_0 cargo_dropoff_northwest HOME \
        cargo_station_3 cargo_dropoff_southeast HOME \
        --teleport "$X" "$Y" "$YAW" \
        --output "$OUT/cycle_${DOMAIN}_$cycle.json" --timeout 900 \
        > "$OUT/cycle_${DOMAIN}_$cycle.log" 2>&1
    results=$(grep -E '^\S+ (SUCCESS|FAILED|TIMEOUT|NO_PLAN|CANCELLED)' \
              "$OUT/cycle_${DOMAIN}_$cycle.log" | awk '{print $2"("$3")"}' | tr '\n' ' ')
    recoveries=$(grep -c 'RECOVERY [0-9]/' "$OUT/cycle_${DOMAIN}_$cycle.log")
    bt_errors=$(cat "$RUN_DIR"/nav2_*.log 2>/dev/null | grep -c 'Error loading BT')
    echo "cycle $cycle domain $DOMAIN start=($X,$Y,$YAW): $results recoveries=$recoveries bt_errors=$bt_errors"
done
tools/mission_validation/isolated_stack.sh down "$DOMAIN"
