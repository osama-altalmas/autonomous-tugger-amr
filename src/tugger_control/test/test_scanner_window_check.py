"""The safety monitor's window check: a sector that stays blind while driving."""
import math

import numpy as np

from tugger_control.safety_monitor import BLIND_TRAVEL_M, BlindSectorCheck

INC = math.radians(270.0 / 550)


def scan(blind_from_deg=None, width_deg=0.0):
    no_return = np.zeros(551, dtype=bool)
    if blind_from_deg is not None:
        first = int(math.radians(blind_from_deg + 135) / INC)
        no_return[first:first + int(math.radians(width_deg) / INC)] = True
    return no_return


def drive(check, frames, step_m=0.04):
    out = 0.0
    for no_return in frames:
        out = check.update(no_return, INC, step_m)
    return out


def test_a_film_over_120_deg_is_found_after_the_travel():
    check = BlindSectorCheck()
    n = int(BLIND_TRAVEL_M / 0.04)
    assert drive(check, [scan(-60, 120)] * (n - 2)) == 0.0     # not yet driven far enough
    assert drive(check, [scan(-60, 120)] * 5) > 100.0


def test_a_moving_gap_such_as_a_door_is_not_a_dirty_window():
    check = BlindSectorCheck()
    frames = [scan(-60 + k, 70) for k in range(0, 80)]          # the gap sweeps by
    assert drive(check, frames) == 0.0


def test_standing_still_never_concludes_and_a_small_sector_is_ignored():
    assert drive(BlindSectorCheck(), [scan(-60, 120)] * 200, step_m=0.0) == 0.0
    assert drive(BlindSectorCheck(), [scan(-10, 35)] * 200) == 0.0


def test_the_stop_holds_until_the_window_is_clean():
    check = BlindSectorCheck()
    drive(check, [scan(-60, 120)] * 60)
    assert drive(check, [scan(-60, 120)] * 20, step_m=0.0) > 100.0   # stopped, still covered
    assert drive(check, [scan()]) == 0.0                              # cleaned
