"""Simulator cargo identity and fixed pickup/drop-off mission rules."""

import math


PICKUP_CARGO = {'cargo_station_0': 0, 'cargo_station_3': 3}
DROPOFF_CARGO = {'cargo_dropoff_northwest': 0, 'cargo_dropoff_southeast': 3}
# Every cargo identity and station, derived from the maps above. Add a station
# by extending PICKUP_CARGO and DROPOFF_CARGO; other modules read these.
CARGO_IDS = tuple(sorted(set(PICKUP_CARGO.values())))
CARGO_STATIONS = tuple(PICKUP_CARGO) + tuple(DROPOFF_CARGO)


def cargo_stand_model(cargo_id):
    """Gazebo model name of the stand that carries ``cargo_id``."""
    return f'cargo_station_{cargo_id}_stand'


def other_cargo(cargo_id):
    """The single other cargo; the clearance models assume exactly two."""
    others = [key for key in CARGO_IDS if key != cargo_id]
    if len(others) != 1:
        raise ValueError('Clearance models support exactly two cargo identities')
    return others[0]


def confirmed_cargo(states, ages, *, max_age=2.0):
    """Return unloaded, cargo 0, cargo 3, or reject stale/conflicting feedback."""
    if not math.isfinite(max_age) or max_age <= 0:
        raise ValueError('Cargo feedback maximum age must be positive')
    for station in CARGO_IDS:
        age = ages.get(station)
        if (age is None or not math.isfinite(age) or age < 0 or age > max_age or
                states.get(station) not in ('attached', 'detached')):
            raise ValueError('Cargo attachment state is unavailable or stale')
    attached = [station for station in CARGO_IDS if states[station] == 'attached']
    if len(attached) > 1:
        raise ValueError('Multiple cargo attachments reported')
    return attached[0] if attached else None


def check_destination(destination, cargo):
    """Enforce Station 1→2 and Station 3→4 without inferring cargo identity."""
    if cargo is not None and cargo not in CARGO_IDS:
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
