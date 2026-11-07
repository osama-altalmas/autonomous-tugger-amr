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

from .cargo_policy import CARGO_STATIONS, PICKUP_CARGO

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


ATTACH, DETACH = 'attach', 'detach'


def assess_destination(destination, carried, occupancy, action=None):
    """Return (blocks, warnings, dock_only) for sending the robot there.

    ``action`` is the operator's choice for this mission: ATTACH, DETACH or
    None (only dock). blocks: reasons the mission must not run. warnings:
    reasons it needs the operator's second RUN. dock_only: dock without any
    lift or cargo action.
    """
    blocks, warnings = [], []
    name = station_label(destination)
    if destination not in CARGO_STATIONS:
        if action is not None:
            blocks.append(f'{name} has no stand: ATTACH and DETACH are not available there')
        return blocks, warnings, True
    occupant = None if occupancy is None else occupancy.get(destination)
    if occupancy is None:
        warnings.append('station cargo state is unknown; cargo positions are not available')
    if carried is not None and occupant is not None:
        # Two loads under one stand would collide: physically impossible.
        blocks.append(f'Robot is carrying {cargo_label(carried)} and {name} holds '
                      f'{cargo_label(occupant)}: drop the carried cargo off first')
        return blocks, warnings, False
    if action == ATTACH:
        if carried is not None:
            blocks.append(f'Robot already carries {cargo_label(carried)}: nothing more to attach')
        elif occupancy is not None and occupant is None:
            blocks.append(f'{name} has no cargo to attach')
        return blocks, warnings, False
    if action == DETACH:
        if carried is None:
            blocks.append('Robot carries no cargo: nothing to detach')
        return blocks, warnings, False
    warnings.append('no cargo action selected (ATTACH or DETACH); the robot will only dock')
    return blocks, warnings, True
