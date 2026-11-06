from tugger_control.cargo_operation import CargoOperation
from tugger_control.lift_geometry import LIFT_CONTACT_M, LIFT_UP_M

UP, TOUCH = LIFT_UP_M, LIFT_CONTACT_M


def test_pickup_touches_then_attaches_then_raises():
    task = CargoOperation('pickup', 0, 1.0, 0.0)
    assert task.initial_command == 'WAIT'
    assert task.observe(1.5, attachment='detached') is None
    assert task.observe(1.8, attachment='detached') == 'CONTACT'
    # No attachment until the platform is at contact with the cargo unmoved.
    assert task.observe(2.0, lift=0.02, cargo_height=0.0) is None
    # Within the old 5 mm band but outside the 2 mm contact band: no attach.
    assert task.observe(2.2, lift=TOUCH - 0.005, cargo_height=0.0) is None
    assert task.observe(2.5, lift=TOUCH, cargo_height=0.0) is None
    # Attach only after the platform has held at contact.
    assert task.observe(3.0, lift=TOUCH, cargo_height=0.0) is None
    assert task.observe(3.3, lift=TOUCH, cargo_height=0.0) == 'ATTACH'
    assert task.observe(3.5, attachment='attached') == 'RAISE'
    assert task.observe(4.0, attachment='attached', lift=UP - 0.02, cargo_height=0.04) is None
    assert task.observe(4.5, attachment='attached', lift=UP, cargo_height=0.0) is None
    assert task.observe(5.0, attachment='attached', lift=UP, cargo_height=0.064) == 'SUCCEEDED'


def test_contact_hold_restarts_if_the_lift_moves_away():
    task = CargoOperation('pickup', 0, 1.0, 0.0)
    assert task.observe(1.8) == 'CONTACT'
    assert task.observe(2.0, lift=TOUCH, cargo_height=0.0) is None
    assert task.observe(2.5, lift=TOUCH - 0.01, cargo_height=0.0) is None
    assert task.observe(2.9, lift=TOUCH, cargo_height=0.0) is None
    assert task.observe(3.4, lift=TOUCH, cargo_height=0.0) is None
    assert task.observe(3.7, lift=TOUCH, cargo_height=0.0) == 'ATTACH'


def test_cargo_pushed_before_attaching_fails_the_pickup():
    task = CargoOperation('pickup', 0, 1.0, 0.0)
    assert task.observe(1.8) == 'CONTACT'
    assert task.observe(2.5, lift=TOUCH, cargo_height=0.02) == 'FAILED'
    assert task.result == 'cargo_moved_before_attach'


def test_dropoff_sets_down_at_contact_then_detaches_then_stows():
    task = CargoOperation('dropoff', 3, 1.0, 0.065)
    assert task.initial_command == 'CONTACT'
    assert task.observe(2.0, attachment='attached', lift=0.08) is None
    assert task.observe(3.0, attachment='attached', lift=TOUCH, cargo_height=0.03) is None
    assert task.observe(3.5, attachment='attached', lift=TOUCH, cargo_height=0.0) is None
    assert task.observe(4.3, attachment='attached', lift=TOUCH, cargo_height=0.0) == 'DETACH'
    assert task.observe(4.4, attachment='attached') is None
    assert task.observe(5.0, attachment='detached', cargo_height=0.0) == 'STOW'
    assert task.observe(5.5, attachment='detached', lift=0.02, cargo_height=0.0) is None
    assert task.observe(6.0, attachment='detached', lift=0.0, cargo_height=0.0) == 'SUCCEEDED'


def test_no_feedback_or_disarming_never_completes_the_mission():
    missing = CargoOperation('pickup', 0, 1.0, 0.0, timeout=2.0)
    assert missing.observe(3.1) == 'FAILED'
    assert missing.result == 'timeout'
    disarmed = CargoOperation('dropoff', 0, 1.0, 0.065)
    assert disarmed.observe(1.1, armed=False) == 'FAILED'


def test_lift_confirmation_without_attachment_cannot_complete_pickup():
    task = CargoOperation('pickup', 0, 1.0, 0.0)
    assert task.observe(1.5, lift=UP) is None
    assert task.observe(2.0, attachment='detached') == 'CONTACT'
    assert task.observe(2.5, lift=TOUCH, cargo_height=0.0) is None
    assert task.observe(3.3, lift=TOUCH, cargo_height=0.0) == 'ATTACH'
    assert task.observe(3.5, attachment='attached') == 'RAISE'
    assert task.observe(4.0, attachment='detached', lift=UP) == 'FAILED'
    assert task.result == 'cargo_lost'


def test_impossible_lift_values_do_not_confirm_a_phase():
    pickup = CargoOperation('pickup', 0, 1.0, 0.0)
    assert pickup.observe(1.8) == 'CONTACT'
    assert pickup.observe(2.0, lift=1.0, cargo_height=0.0) is None
    dropoff = CargoOperation('dropoff', 0, 1.0, 0.065)
    assert dropoff.observe(2.0, attachment='attached', lift=-1.0) is None


def test_disarming_during_settling_never_sends_the_attachment_command():
    task = CargoOperation('pickup', 0, 1.0, 0.0)
    assert task.observe(1.5, armed=True) is None
    assert task.observe(1.9, armed=False) == 'FAILED'
    assert task.observe(2.1, armed=True) is None
    assert task.result == 'disarmed'


def test_dropoff_cargo_loss_during_settling_fails():
    task = CargoOperation('dropoff', 0, 1.0, 0.065)
    assert task.observe(2.0, attachment='attached', lift=TOUCH, cargo_height=0.0) is None
    assert task.observe(2.8, attachment='detached', lift=TOUCH, cargo_height=0.0) == 'FAILED'
    assert task.result == 'cargo_lost'


def test_down_band_never_exceeds_what_unloaded_guards_accept():
    from tugger_control.lift_geometry import DOWN_TOLERANCE_M, LIFT_DOWN_M, lift_is_down
    assert LIFT_DOWN_M + DOWN_TOLERANCE_M <= 0.005
    assert lift_is_down(LIFT_DOWN_M) and not lift_is_down(0.007)


def test_a_lift_stuck_on_its_limit_fails_fast_with_a_clear_reason():
    task = CargoOperation('pickup', 0, 1.0, 0.0)
    assert task.observe(1.8) == 'CONTACT'
    assert task.observe(3.0, lift=0.0, cargo_height=0.0) is None
    assert task.observe(4.9, lift=0.0, cargo_height=0.0) == 'FAILED'
    assert task.result == 'lift_latched_in_simulator_restart_gazebo'
