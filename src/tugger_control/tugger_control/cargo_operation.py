"""Motion-free event state machine for confirmed simulator cargo operations."""

import math

from tugger_control.cargo_policy import CARGO_IDS
from tugger_control.lift_geometry import (
    CARGO_RAISE_M, lift_at_contact, lift_is_down, lift_is_raised)

# Cargo height change that confirms a raise or a set-down (0.065 m nominal).
RAISE_BAND_M = (CARGO_RAISE_M - 0.01, CARGO_RAISE_M + 0.01)
# The cargo must not move while the lift travels up to it, before attaching.
UNMOVED_M = 0.005
# The simulator attaches only to a platform that has been still for 0.5 s.
CONTACT_HOLD_S = 0.75
# A lift still on its lower limit this long after CONTACT is latched in Gazebo.
LATCH_DETECT_S = 3.0
LATCH_POSITION_M = 0.0005


class CargoOperation:
    """Sequence contact, attachment and lift feedback without publishing ROS commands.

    Pickup: settle, lift to contact, attach, raise.
    Drop-off: lower to contact (cargo rests on its support), detach, lower fully.
    The joint is only created or removed at contact, so the cargo is carried by
    the platform rather than held across a gap. A transition is driven by
    feedback received after its command, not by the command itself. Timeouts
    are terminal and never mean success.

    Commands returned: CONTACT, ATTACH, RAISE, DETACH, STOW, SUCCEEDED, FAILED.
    """

    def __init__(self, operation, cargo_id, now, cargo_height, timeout=10.0):
        if operation not in ('pickup', 'dropoff') or cargo_id not in CARGO_IDS:
            raise ValueError('Invalid cargo operation')
        if (not math.isfinite(now) or not math.isfinite(timeout) or timeout <= 0
                or not math.isfinite(cargo_height)):
            raise ValueError('Invalid cargo operation timeout')
        self.operation = operation
        self.cargo_id = cargo_id
        self.timeout = timeout
        self.phase = 'WAIT_ATTACH_SETTLE' if operation == 'pickup' else 'WAIT_LOWER'
        self.phase_started = now
        self.initial_cargo_height = cargo_height
        self.result = None
        self._contact_since = None

    @property
    def initial_command(self):
        return 'WAIT' if self.operation == 'pickup' else 'CONTACT'

    def _next(self, phase, now, command=None):
        self.phase = phase
        self.phase_started = now
        return command

    def _risen(self, cargo_height):
        return (cargo_height is not None and
                RAISE_BAND_M[0] <= cargo_height - self.initial_cargo_height <= RAISE_BAND_M[1])

    def _set_down(self, cargo_height):
        return (cargo_height is not None and
                RAISE_BAND_M[0] <= self.initial_cargo_height - cargo_height <= RAISE_BAND_M[1])

    def _unmoved(self, cargo_height):
        return (cargo_height is not None and
                abs(cargo_height - self.initial_cargo_height) <= UNMOVED_M)

    def observe(self, now, *, attachment=None, lift=None, cargo_height=None, armed=True):
        if self.result is not None:
            return None
        if not math.isfinite(now) or now < self.phase_started:
            self.result = 'clock_error'
            return 'FAILED'
        if not armed:
            self.result = 'disarmed'
            return 'FAILED'
        if now - self.phase_started > self.timeout:
            self.result = 'timeout'
            return 'FAILED'
        if attachment in ('attach_rejected', 'detach_rejected'):
            self.result = attachment
            return 'FAILED'
        if cargo_height is not None and not math.isfinite(cargo_height):
            self.result = 'invalid_cargo_pose'
            return 'FAILED'
        if lift is not None and not math.isfinite(lift):
            self.result = 'invalid_lift'
            return 'FAILED'
        if self.operation == 'pickup':
            return self._pickup(now, attachment, lift, cargo_height)
        return self._dropoff(now, attachment, lift, cargo_height)

    def _pickup(self, now, attachment, lift, cargo_height):
        # Give the vehicle time to settle before touching the cargo. The delay is
        # not proof of stationarity; the simulator's own guard stays authoritative.
        if self.phase == 'WAIT_ATTACH_SETTLE':
            if now - self.phase_started >= 0.75:
                return self._next('WAIT_CONTACT', now, 'CONTACT')
            return None
        if self.phase == 'WAIT_CONTACT':
            if (lift is not None and lift <= LATCH_POSITION_M and
                    now - self.phase_started >= LATCH_DETECT_S):
                self.result = 'lift_latched_in_simulator_restart_gazebo'
                return 'FAILED'
            if cargo_height is not None and not self._unmoved(cargo_height):
                self.result = 'cargo_moved_before_attach'
                return 'FAILED'
            if lift is not None and not lift_at_contact(lift):
                self._contact_since = None
            elif lift_at_contact(lift) and self._unmoved(cargo_height):
                if self._contact_since is None:
                    self._contact_since = now
                elif now - self._contact_since >= CONTACT_HOLD_S:
                    return self._next('WAIT_ATTACH', now, 'ATTACH')
            return None
        if self.phase == 'WAIT_ATTACH':
            if attachment == 'attached':
                return self._next('WAIT_RAISE', now, 'RAISE')
            return None
        if self.phase == 'WAIT_RAISE':
            if attachment == 'detached':
                self.result = 'cargo_lost'
                return 'FAILED'
            if lift_is_raised(lift) and attachment == 'attached' and self._risen(cargo_height):
                self.result = 'success'
                return 'SUCCEEDED'
        return None

    def _dropoff(self, now, attachment, lift, cargo_height):
        if self.phase == 'WAIT_LOWER':
            if attachment == 'detached':
                self.result = 'cargo_lost'
                return 'FAILED'
            if lift_at_contact(lift) and attachment == 'attached' and self._set_down(cargo_height):
                return self._next('WAIT_DETACH_SETTLE', now)
            return None
        if self.phase == 'WAIT_DETACH_SETTLE':
            if attachment == 'detached':
                self.result = 'cargo_lost'
                return 'FAILED'
            if (now - self.phase_started >= 0.75 and attachment == 'attached' and
                    lift_at_contact(lift) and self._set_down(cargo_height)):
                return self._next('WAIT_DETACH', now, 'DETACH')
            return None
        if self.phase == 'WAIT_DETACH':
            if attachment == 'detached' and self._set_down(cargo_height):
                return self._next('WAIT_STOW', now, 'STOW')
            return None
        if self.phase == 'WAIT_STOW':
            if attachment == 'attached':
                self.result = 'cargo_reattached'
                return 'FAILED'
            if lift_is_down(lift) and attachment == 'detached' and self._set_down(cargo_height):
                self.result = 'success'
                return 'SUCCEEDED'
        return None
