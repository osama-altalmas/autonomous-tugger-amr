"""The startup guard must construct and pick the right transitions."""
import pytest

rclpy = pytest.importorskip('rclpy')
from lifecycle_msgs.msg import State, Transition  # noqa: E402

from tugger_control.nav2_startup_guard import NEXT_TRANSITION, Nav2StartupGuard  # noqa: E402


def test_guard_constructs_with_a_client_pair_per_server():
    rclpy.init()
    try:
        guard = Nav2StartupGuard()
        assert 'docking/controller_server' in guard._lifecycle_clients
        guard.destroy_node()
    finally:
        rclpy.shutdown()


def test_guard_only_configures_or_activates():
    assert NEXT_TRANSITION[State.PRIMARY_STATE_UNCONFIGURED][0] == Transition.TRANSITION_CONFIGURE
    assert NEXT_TRANSITION[State.PRIMARY_STATE_INACTIVE][0] == Transition.TRANSITION_ACTIVATE
    assert State.PRIMARY_STATE_ACTIVE not in NEXT_TRANSITION
