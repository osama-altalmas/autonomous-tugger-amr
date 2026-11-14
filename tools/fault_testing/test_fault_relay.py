"""The fault relay's scanner and IMU faults, without ROS running."""
import math
import random
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fault_relay import apply_scan_fault, drift_imu  # noqa: E402


def scan(n=360):
    return SimpleNamespace(ranges=[5.0] * n, angle_min=-math.pi,
                           angle_increment=2 * math.pi / n, range_min=0.05, range_max=25.0)


def test_no_fault_and_silent():
    msg = scan()
    assert apply_scan_fault(msg, {'mode': 'none'}, random.Random(1)) == msg.ranges
    assert apply_scan_fault(msg, {'mode': 'silent'}, random.Random(1)) is None


def test_blind_sector_has_no_returns_only_there():
    msg = scan()
    out = apply_scan_fault(msg, {'mode': 'blind', 'start_deg': 0, 'width_deg': 30},
                           random.Random(1))
    lost = [i for i, r in enumerate(out) if math.isinf(r)]
    assert len(lost) == 30 and lost[0] == 180


def test_smudge_puts_phantoms_just_outside_the_window():
    msg = scan()
    out = apply_scan_fault(msg, {'mode': 'smudge', 'start_deg': -20, 'width_deg': 30},
                           random.Random(1))
    near = [r for r in out if r < 1.0]
    assert 5 <= len(near) <= 20
    assert all(0.05 <= r <= 0.17 for r in near)
    assert out[:150] == msg.ranges[:150]


def test_ghost_points_lie_behind_the_surface():
    rng = random.Random(3)
    behind = dropped = 0
    for _ in range(200):
        out = apply_scan_fault(scan(), {'mode': 'ghost'}, rng)
        behind += sum(1 for r in out if math.isfinite(r) and r > 5.0)
        dropped += sum(1 for r in out if math.isinf(r))
    assert behind > 0
    assert 0.01 < dropped / (200 * 360) < 0.03


def test_imu_drift_accumulates_yaw():
    q = SimpleNamespace(w=1.0, x=0.0, y=0.0, z=0.0)
    msg = SimpleNamespace(angular_velocity=SimpleNamespace(z=0.0), orientation=q)
    drift_imu(msg, 15.0, 3600.0)
    yaw = 2 * math.atan2(msg.orientation.z, msg.orientation.w)
    assert math.isclose(math.degrees(yaw), 15.0, abs_tol=1e-6)
    assert math.isclose(math.degrees(msg.angular_velocity.z) * 3600, 15.0)
