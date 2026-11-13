"""Ackermann steering geometry and the shared turning limit."""
import math
import random
import re
from pathlib import Path

import pytest

from tugger_control import steering_geometry as g
from tugger_control.turn_in_place import drive_steering, pivot_odometry

ROOT = Path(__file__).resolve().parents[3]


def _sideways_slip(v, curvature, angles):
    """Largest wheel velocity across its own rolling direction (m/s)."""
    omega = v * curvature
    worst = 0.0
    for (x, y), angle in zip(g.WHEEL_POSITIONS, angles):
        vx, vy = v - omega * y, omega * x
        worst = max(worst, abs(-vx * math.sin(angle) + vy * math.cos(angle)))
    return worst


def test_no_wheel_slides_sideways_at_any_curvature():
    rng = random.Random(0)
    for _ in range(500):
        k = rng.uniform(-g.MAX_CURVATURE, g.MAX_CURVATURE)
        v = rng.uniform(-1.0, 1.0)
        assert _sideways_slip(v, k, g.ackermann_steering(k)) < 1e-12


def test_linked_axles_did_slide_in_a_route_corner():
    # The former steering: one angle per axle. In a 1 m corner at 0.6 m/s the
    # worst wheel slid sideways at about 4 cm/s, unseen by the odometry.
    d = g.virtual_from_curvature(1.0)
    assert _sideways_slip(0.6, 1.0, (d, d, -d, -d)) > 0.03


def test_inner_wheel_limit_sets_the_turning_radius():
    assert g.MIN_TURN_RADIUS_M == pytest.approx(0.51, abs=1e-3)
    angles = g.ackermann_steering(g.MAX_CURVATURE)
    assert max(abs(a) for a in angles) == pytest.approx(g.MAX_WHEEL_STEERING_RAD)
    # Left turn: the left (inner) wheels steer more; rear mirrors front.
    fl, fr, rl, rr = angles
    assert fl > fr > 0 and rl == pytest.approx(-fl) and rr == pytest.approx(-fr)
    assert g.ackermann_steering(0.0) == (0.0, 0.0, 0.0, 0.0)


def test_drive_pattern_is_ackermann_for_the_virtual_angle():
    for delta in (-0.4, -0.1, 0.0, 0.2, g.MAX_VIRTUAL_STEERING_RAD):
        expected = g.ackermann_steering(g.curvature_from_virtual(delta))
        assert drive_steering(delta) == pytest.approx(expected)


def test_odometry_fit_recovers_the_motion_from_ackermann_wheels():
    rng = random.Random(1)
    for _ in range(200):
        k = rng.uniform(-g.MAX_CURVATURE, g.MAX_CURVATURE)
        v = rng.uniform(-1.0, 1.0)
        angles = g.ackermann_steering(k)
        speeds = g.rolling_speeds(v, v * k, angles)
        vx, vy, omega = pivot_odometry(angles, speeds, wheel_positions=g.WHEEL_POSITIONS,
                                       wheel_radius=g.WHEEL_RADIUS_M)
        assert vx == pytest.approx(v, abs=1e-9)
        assert vy == pytest.approx(0.0, abs=1e-9)
        assert omega == pytest.approx(v * k, abs=1e-9)


def test_navigation_config_uses_the_shared_turning_limit():
    text = (ROOT / 'src/tugger_navigation/config/nav2.yaml').read_text()
    radius = round(g.MIN_TURN_RADIUS_M, 2)
    assert float(re.search(r'\n      turning_radius: ([\d.]+)', text).group(1)) == radius
    assert float(re.search(r'tracking_max_curvature: ([\d.]+)', text).group(1)) <= g.MAX_CURVATURE
    for value in re.findall(r'(?:minimum_turning_radius|regulated_linear_scaling_min_radius): '
                            r'([\d.]+)', text):
        assert float(value) == radius
    law = (ROOT / 'src/tugger_navigation/include/tugger_navigation/'
           'path_tracking_law.hpp').read_text()
    assert float(re.search(r'max_curvature = ([\d.]+);', law).group(1)) <= g.MAX_CURVATURE
    planner = (ROOT / 'src/tugger_navigation/include/tugger_navigation/'
               'rrt_star_reeds_shepp_planner.hpp').read_text()
    assert float(re.search(r'turning_radius_\{([\d.]+)\}', planner).group(1)) == radius


def test_routes_and_live_plans_stay_inside_the_turning_limit():
    text = (ROOT / 'tools/v1_manhattan_bezier/build_route_database_v1.py').read_text()
    route_limit = float(re.search(r'"--max-curvature",\s*type=float,\s*default=([\d.]+)',
                                  text).group(1))
    assert route_limit < g.MAX_CURVATURE
    from tugger_control.live_route_planner import LiveRoutePlanner
    import inspect
    default = inspect.signature(LiveRoutePlanner.__init__).parameters['max_curvature'].default
    assert default < g.MAX_CURVATURE
