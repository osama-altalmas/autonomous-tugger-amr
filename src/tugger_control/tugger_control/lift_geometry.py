"""Lift heights for contact pickup, derived from the robot and world geometry.

The lowered lift platform top sits at 0.2375 m (base_footprint 0.100 + lift
joint 0.130 + half platform 0.0075). The cargo tabletop underside is at 0.28 m
(tabletop centre 0.295 - half thickness 0.015). The lift touches the cargo
after 0.0425 m of travel; only then may the simulator joint be created. The
cargo is then raised 0.065 m, the same carried height as before this change, so
loaded clearance checks keep their geometry.
"""

LIFT_LOWERED_TOP_M = 0.2375
CARGO_UNDERSIDE_M = 0.28
LIFT_CONTACT_M = CARGO_UNDERSIDE_M - LIFT_LOWERED_TOP_M
CARGO_RAISE_M = 0.065
# The joint's hard stops (URDF) sit outside the 0.002-0.1075 m operating range:
# a DART joint resting on a limit under a velocity limit latches there
# (gazebosim/gz-sim#2785). Commands therefore never reach a hard stop.
LIFT_DOWN_M = 0.002
LIFT_UP_M = LIFT_CONTACT_M + CARGO_RAISE_M
LIFT_TOLERANCE_M = 0.005
# Tighter band for touching the cargo: the simulator attaches only with a gap
# of 5 mm or less, so contact must be confirmed well inside that.
CONTACT_TOLERANCE_M = 0.002


# Down means 0 to 5 mm, the band every unloaded guard accepts.
DOWN_TOLERANCE_M = 0.003


def lift_is_down(lift):
    return lift is not None and abs(lift - LIFT_DOWN_M) <= DOWN_TOLERANCE_M


def lift_is_raised(lift):
    """Raised with cargo: the band the loaded guards accept."""
    return lift is not None and abs(lift - LIFT_UP_M) <= LIFT_TOLERANCE_M


def lift_at_contact(lift):
    return lift is not None and abs(lift - LIFT_CONTACT_M) <= CONTACT_TOLERANCE_M


def cargo_lift_offset(lift):
    """How far an attached cargo sits above its support at this lift height."""
    return max(0.0, lift - LIFT_CONTACT_M)
