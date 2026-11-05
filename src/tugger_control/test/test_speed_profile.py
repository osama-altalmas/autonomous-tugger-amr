import pytest

from tugger_control.motion_math import speed_profile


def test_straight_route_uses_1_2_mps_cap():
    assert speed_profile(2.0, 0.1) == pytest.approx((1.2, 0.06))


def test_curve_route_uses_0_6_mps_cap_without_changing_curvature():
    v, omega = speed_profile(1.2, 0.6)
    assert v == pytest.approx(0.6)
    assert omega == pytest.approx(0.3)
    assert omega / v == pytest.approx(0.6 / 1.2)


def test_reverse_curve_preserves_sign_and_curvature():
    v, omega = speed_profile(-1.0, 0.5)
    assert v == pytest.approx(-0.6)
    assert omega == pytest.approx(0.3)
