from tugger_control.cargo_operation import CargoOperation


def test_pickup_requires_attachment_then_confirmed_lift():
    task = CargoOperation('pickup', 0, 1.0, 0.0)
    assert task.initial_command == 'WAIT'
    assert task.observe(1.5, attachment='detached') is None
    assert task.observe(1.8, attachment='detached') == 'ATTACH'
    assert task.observe(2.0, attachment='attached') == 'RAISE'
    assert task.observe(3.0, attachment='attached', lift=0.03) is None
    assert task.observe(4.0, attachment='attached', lift=0.064) is None
    assert task.observe(4.5, attachment='attached', lift=0.064, cargo_height=0.0) is None
    assert task.observe(5.0, attachment='attached', lift=0.064, cargo_height=0.064) == 'SUCCEEDED'


def test_dropoff_lowers_before_detaching_and_waits_for_feedback():
    task = CargoOperation('dropoff', 3, 1.0, 0.065)
    assert task.initial_command == 'LOWER'
    assert task.observe(2.0, attachment='attached', lift=0.03) is None
    assert task.observe(3.0, attachment='attached', lift=0.001) is None
    assert task.observe(3.5, attachment='attached', lift=0.001, cargo_height=0.0) is None
    assert task.observe(4.3, attachment='attached', lift=0.001, cargo_height=0.0) == 'DETACH'
    assert task.observe(4.4, attachment='attached') is None
    assert task.observe(5.0, attachment='detached') is None
    assert task.observe(5.5, attachment='detached', cargo_height=0.0) == 'SUCCEEDED'


def test_no_feedback_or_disarming_never_completes_the_mission():
    missing = CargoOperation('pickup', 0, 1.0, 0.0, timeout=2.0)
    assert missing.observe(3.1) == 'FAILED'
    assert missing.result == 'timeout'
    disarmed = CargoOperation('dropoff', 0, 1.0, 0.065)
    assert disarmed.observe(1.1, armed=False) == 'FAILED'


def test_lift_confirmation_without_attachment_cannot_complete_pickup():
    task = CargoOperation('pickup', 0, 1.0, 0.0)
    assert task.observe(1.5, lift=0.065) is None
    assert task.observe(2.0, attachment='detached') == 'ATTACH'
    assert task.observe(3.0, attachment='attached') == 'RAISE'
    assert task.observe(4.0, attachment='detached', lift=0.065) == 'FAILED'
    assert task.result == 'cargo_lost'


def test_impossible_lift_values_do_not_confirm_a_phase():
    pickup = CargoOperation('pickup', 0, 1.0, 0.0)
    assert pickup.observe(1.8) == 'ATTACH'
    assert pickup.observe(2.0, attachment='attached') == 'RAISE'
    assert pickup.observe(3.0, attachment='attached', lift=1.0) is None
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
    assert task.observe(2.0, attachment='attached', lift=0.0, cargo_height=0.0) is None
    assert task.observe(2.8, attachment='detached', lift=0.0, cargo_height=0.0) == 'FAILED'
    assert task.result == 'cargo_lost'
