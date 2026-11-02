import math

from tugger_control.qr_alignment import bounded_qr_correction


def test_qr_alignment_declares_completion_inside_the_dock_gate():
    result = bounded_qr_correction(0.005, 0.005, math.radians(0.5))
    assert result['state'] == 'ALIGNED'


def test_qr_alignment_is_bounded_and_slow():
    result = bounded_qr_correction(0.1, 0.2, 0.2)
    assert result['state'] == 'CORRECT'
    assert 0.0 <= result['linear_x'] <= 0.05
    assert abs(result['angular_z']) <= 0.25


def test_qr_alignment_can_make_a_bounded_reverse_correction():
    result = bounded_qr_correction(0.0, -0.08, 0.0)
    assert result['state'] == 'CORRECT'
    assert -0.05 <= result['linear_x'] < 0.0


def test_positive_lateral_error_commands_a_positive_turn_toward_the_frame():
    result = bounded_qr_correction(0.05, 0.0, 0.0)
    assert result['state'] == 'CORRECT'
    assert result['angular_z'] > 0.0


def test_qr_alignment_stops_for_an_invalid_observation():
    result = bounded_qr_correction(float('nan'), 0.1, 0.0)
    assert result['state'] == 'STOP'
