# Fault testing (simulation only)

Realistic faults injected into the isolated validation simulation, to check
that the robot either keeps working within limits or stops with a clear
panel message. Nothing here runs on the robot; the robot code is unchanged
by these tools.

| File | What it does |
|---|---|
| `fault_relay.py` | Sits between Gazebo's scanners/IMU and the robot: `silent`, `smudge` (dirty window), `blind` (film), `ghost` (reflections), IMU `drift`. Commands on `/fault/command`. |
| `fault_schedule.py` | Timed faults during a `mission_runner.py` run: objects, pallets, moved racks and stations, people walking, floor-marker covers, slippery patches, robot pushed by hand, a node killed and restarted. |
| `schedules/*.json` | The scenarios used in the 2026-11-14 campaign. |
| `test_fault_relay.py` | Checks of the relay's faults without ROS running. |

Run one scenario (isolated domain, headless):

```bash
export TUGGER_SIM_ARGS="sensor_faults:=true"          # add sim_wheel_radius:=0.0539 for worn tyres
tools/mission_validation/isolated_stack.sh up 77
ROS_DOMAIN_ID=77 GZ_PARTITION=tugger_validation_77 python3 tools/mission_validation/mission_runner.py \
    cargo_station_3 cargo_dropoff_southeast HOME \
    --faults tools/fault_testing/schedules/f89_pushed_sensor_node.json --keep-going \
    --output /tmp/f89.json            # also writes /tmp/f89.faults.json (fault log, safety, contacts)
tools/mission_validation/isolated_stack.sh down 77
```

`sensor_faults:=true` makes the simulation publish the scanners and IMU under
`/sim_raw/...`; the isolated stack then starts the relay. Without it the
simulation is exactly as before.
