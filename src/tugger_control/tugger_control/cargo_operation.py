"""Motion-free event state machine for confirmed simulator cargo operations."""

import math


class CargoOperation:
    """Sequence attachment/lift feedback without publishing ROS commands.

    The caller owns arming, collision clearance, time source and command output.
    A transition is driven by feedback received after its command, not by the
    command itself. Timeouts are terminal and never mean success.
    """

    def __init__(self, operation, cargo_id, now, cargo_height, timeout=10.0):
        if operation not in ('pickup', 'dropoff') or cargo_id not in (0, 3):
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

    @property
    def initial_command(self):
        return 'WAIT' if self.operation == 'pickup' else 'LOWER'

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
        # Give the lift/vehicle time to settle before asking the simulator's
        # separate stationary guard to create or remove the joint.  The delay
        # is not itself proof of stationarity; that guard remains authoritative.
        if self.phase == 'WAIT_ATTACH_SETTLE' and now - self.phase_started >= 0.75:
            self.phase = 'WAIT_ATTACH'
            self.phase_started = now
            return 'ATTACH'
        if self.phase in ('WAIT_RAISE', 'WAIT_LOWER') and attachment == 'detached':
            self.result = 'cargo_lost'
            return 'FAILED'
        if self.phase == 'WAIT_ATTACH' and attachment == 'attached':
            self.phase = 'WAIT_RAISE'
            self.phase_started = now
            return 'RAISE'
        if self.phase == 'WAIT_RAISE' and lift is not None:
            if not math.isfinite(lift):
                self.result = 'invalid_lift'
                return 'FAILED'
            if (0.060 <= lift <= 0.070 and attachment == 'attached' and
                    cargo_height is not None and
                    0.055 <= cargo_height - self.initial_cargo_height <= 0.075):
                self.result = 'success'
                return 'SUCCEEDED'
        if self.phase == 'WAIT_LOWER' and lift is not None:
            if not math.isfinite(lift):
                self.result = 'invalid_lift'
                return 'FAILED'
            if (abs(lift) <= 0.005 and attachment == 'attached' and
                    cargo_height is not None and
                    0.055 <= self.initial_cargo_height - cargo_height <= 0.075):
                self.phase = 'WAIT_DETACH_SETTLE'
                self.phase_started = now
                return None
        if self.phase == 'WAIT_DETACH_SETTLE':
            if attachment == 'detached':
                self.result = 'cargo_lost'
                return 'FAILED'
            if (now - self.phase_started >= 0.75 and attachment == 'attached' and
                    lift is not None and abs(lift) <= 0.005 and
                    cargo_height is not None and
                    0.055 <= self.initial_cargo_height - cargo_height <= 0.075):
                self.phase = 'WAIT_DETACH'
                self.phase_started = now
                return 'DETACH'
        if (self.phase == 'WAIT_DETACH' and attachment == 'detached' and
                cargo_height is not None and
                0.055 <= self.initial_cargo_height - cargo_height <= 0.075):
            self.result = 'success'
            return 'SUCCEEDED'
        return None
