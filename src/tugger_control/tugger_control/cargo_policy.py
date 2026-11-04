"""Simulator cargo identity and fixed pickup/drop-off mission rules."""

import math


PICKUP_CARGO = {'cargo_station_0': 0, 'cargo_station_3': 3}
DROPOFF_CARGO = {'cargo_dropoff_northwest': 0, 'cargo_dropoff_southeast': 3}


def confirmed_cargo(states, ages, *, max_age=2.0):
    """Return unloaded, cargo 0, cargo 3, or reject stale/conflicting feedback."""
    if not math.isfinite(max_age) or max_age <= 0:
        raise ValueError('Cargo feedback maximum age must be positive')
    for station in (0, 3):
        age = ages.get(station)
        if (age is None or not math.isfinite(age) or age < 0 or age > max_age or
                states.get(station) not in ('attached', 'detached')):
            raise ValueError('Cargo attachment state is unavailable or stale')
    attached = [station for station in (0, 3) if states[station] == 'attached']
    if len(attached) > 1:
        raise ValueError('Multiple cargo attachments reported')
    return attached[0] if attached else None


def check_destination(destination, cargo):
    """Enforce Station 1→2 and Station 3→4 without inferring cargo identity."""
    if cargo not in (None, 0, 3):
        raise ValueError('Unknown cargo identity')
    if destination == 'HOME':
        return
    if destination in PICKUP_CARGO:
        if cargo is not None:
            raise ValueError('A pickup requires the robot to be unloaded')
        return
    if destination in DROPOFF_CARGO:
        if cargo != DROPOFF_CARGO[destination]:
            raise ValueError('This drop-off does not match the confirmed cargo')
        return
    raise ValueError('Unknown mission destination')
