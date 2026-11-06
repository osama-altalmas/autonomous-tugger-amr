"""Which station holds which cargo, and whether a mission makes sense.

Station state is measured, not assumed: in simulation every resting cargo's
pose is known, so a station is occupied when a cargo that is not attached to
the robot rests on its centre. The state is also saved to a file so it
survives panel restarts; the live measurement always wins when it is fresh.

assess_destination() separates hard blocks (physically impossible or
colliding: carrying cargo under a loaded stand, cargo onto cargo) from
warnings the operator may confirm with a second RUN (unusual but safe).
"""

import json
import math
from pathlib import Path

from .cargo_policy import CARGO_STATIONS, DROPOFF_CARGO, PICKUP_CARGO

STATION_LABELS = {
    'HOME': 'HOME',
    'cargo_station_0': 'Station 1',
    'cargo_dropoff_northwest': 'Station 2',
    'cargo_station_3': 'Station 3',
    'cargo_dropoff_southeast': 'Station 4',
}
OCCUPIED_TOLERANCE_M = 0.20
STATE_FILE = Path.home() / '.ros' / 'tugger_station_occupancy.json'


def station_label(station):
    return STATION_LABELS.get(station, station)


def cargo_label(cargo_id):
    """Cargo is named after the pickup station it started at."""
    home = next((s for s, k in PICKUP_CARGO.items() if k == cargo_id), None)
    return f'{station_label(home)} cargo' if home else f'cargo {cargo_id}'


def measure_occupancy(cargo_poses, attached, centers, tolerance=OCCUPIED_TOLERANCE_M):
    """Return {station: cargo id or None}; None if a cargo pose is unknown."""
    if any(cargo_poses.get(k) is None for k in cargo_poses if k != attached):
        return None
    state = {station: None for station in CARGO_STATIONS}
    for cargo_id, pose in cargo_poses.items():
        if cargo_id == attached or pose is None:
            continue
        for station in CARGO_STATIONS:
            center = centers.get(station)
            if center is not None and math.hypot(pose[0] - center[0],
                                                 pose[1] - center[1]) <= tolerance:
                state[station] = cargo_id
    return state


def save_occupancy(state, path=STATE_FILE):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({k: v for k, v in state.items()}, indent=1) + '\n')


def load_occupancy(path=STATE_FILE):
    try:
        data = json.loads(path.read_text())
        return {station: data.get(station) for station in CARGO_STATIONS}
    except (OSError, ValueError, AttributeError):
        return None


def assess_destination(destination, carried, occupancy):
    """Return (blocks, warnings, dock_only) for sending the robot there.

    blocks: reasons the mission must not run. warnings: reasons it is unusual;
    the operator confirms them with a second RUN. dock_only: the robot docks
    but performs no cargo operation there.
    """
    blocks, warnings = [], []
    if destination == 'HOME' or destination not in CARGO_STATIONS:
        return blocks, warnings, False
    name = station_label(destination)
    occupant = None if occupancy is None else occupancy.get(destination)
    if occupancy is None:
        warnings.append('station cargo state is unknown; cargo positions are not available')
    if destination in PICKUP_CARGO:
        if carried is not None:
            target = next((s for s, k in DROPOFF_CARGO.items() if k == carried), None)
            where = f' (at {station_label(target)})' if target else ''
            blocks.append(f'Robot is carrying {cargo_label(carried)}: drop it off first{where}')
            return blocks, warnings, False
        if occupancy is not None and occupant is None:
            warnings.append(f'{name} has no cargo to pick up; the robot will only dock there')
            return blocks, warnings, True
        return blocks, warnings, False
    # A drop-off station.
    if carried is None:
        if occupant is not None:
            warnings.append(f'{name} holds {cargo_label(occupant)} and the robot has no '
                            'cargo; it will only dock there')
        else:
            warnings.append(f'Robot has no cargo to drop off; it will only dock at {name}')
        return blocks, warnings, True
    if occupant is not None:
        blocks.append(f'{name} already holds {cargo_label(occupant)}; it cannot take '
                      f'{cargo_label(carried)}')
        return blocks, warnings, False
    if DROPOFF_CARGO.get(destination) != carried:
        target = next((s for s, k in DROPOFF_CARGO.items() if k == carried), None)
        warnings.append(f'{cargo_label(carried)} belongs at {station_label(target)}, '
                        f'not {name}; it will be set down at {name}')
    return blocks, warnings, False
