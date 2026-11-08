import pytest

from tugger_control.motion_math import speed_profile


def test_straight_route_uses_1_0_mps_cap():
    # Mixed traffic: 1.0 m/s maximum (Phase B).
    assert speed_profile(2.0, 0.1) == pytest.approx((1.0, 0.05))


def test_curve_route_uses_0_6_mps_cap_without_changing_curvature():
    v, omega = speed_profile(1.2, 0.6)
    assert v == pytest.approx(0.6)
    assert omega == pytest.approx(0.3)
    assert omega / v == pytest.approx(0.6 / 1.2)


def test_reverse_curve_preserves_sign_and_curvature():
    v, omega = speed_profile(-1.0, 0.5)
    assert v == pytest.approx(-0.6)
    assert omega == pytest.approx(0.3)


def test_precision_follower_tuning_stays_stable_and_below_gate_limit():
    """Guard the segment 2 tuning that was shown to abort in Gazebo."""
    from pathlib import Path

    import yaml

    config = Path(__file__).resolve().parents[2] / 'tugger_navigation/config/nav2.yaml'
    data = yaml.safe_load(config.read_text(encoding='utf-8'))
    tuning = data['controller_server']['ros__parameters']['PrecisionFollowPath']
    assert tuning['max_linear_vel'] < 0.40
    # About 0.36 s of steering slew at full speed must fit inside the look-ahead.
    assert tuning['min_lookahead_dist'] >= 0.36 * tuning['max_linear_vel']
    assert tuning['max_lookahead_dist'] >= 0.29


def test_loaded_route_request_stays_below_guard_limit():
    from tugger_control.sim_motion_gate import LOADED_ROUTE_REQUEST_LIMIT

    assert LOADED_ROUTE_REQUEST_LIMIT < 1.2
    assert speed_profile(1.2, 0.0, straight_limit=LOADED_ROUTE_REQUEST_LIMIT)[0] < 1.2


def test_camera_handoff_stop_tolerance_keeps_the_red_frame_in_view():
    from pathlib import Path

    import yaml

    config = Path(__file__).resolve().parents[2] / 'tugger_navigation/config/nav2.yaml'
    server = yaml.safe_load(config.read_text(encoding='utf-8'))['controller_server']['ros__parameters']
    assert 'camera_handoff_goal_checker' in server['goal_checker_plugins']
    # Whole 12 cm frame stays in the downward camera within about 5.3 cm forward.
    assert server['camera_handoff_goal_checker']['xy_goal_tolerance'] < 0.05
