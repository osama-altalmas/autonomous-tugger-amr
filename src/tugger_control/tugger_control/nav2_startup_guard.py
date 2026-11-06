#!/usr/bin/env python3
"""Complete Nav2 lifecycle transitions that a startup race left unfinished.

Nav2's lifecycle managers sometimes hang when a change_state response is lost
during service discovery on a busy machine, leaving a server unconfigured or
inactive for the whole session. After a grace period this guard finishes the
missing configure/activate transitions itself, then exits once every server is
active. It never deactivates or cleans up anything.
"""

import time

import rclpy
from lifecycle_msgs.msg import State, Transition
from lifecycle_msgs.srv import ChangeState, GetState
from rclpy.node import Node

DEFAULT_NODES = ['planner_server', 'controller_server', 'docking/controller_server']
NEXT_TRANSITION = {
    State.PRIMARY_STATE_UNCONFIGURED: (Transition.TRANSITION_CONFIGURE, 'configure'),
    State.PRIMARY_STATE_INACTIVE: (Transition.TRANSITION_ACTIVATE, 'activate'),
}


class Nav2StartupGuard(Node):
    def __init__(self):
        super().__init__('nav2_startup_guard')
        self.declare_parameter('nodes', DEFAULT_NODES)
        self.declare_parameter('grace_s', 15.0)
        self.declare_parameter('give_up_s', 180.0)
        self.nodes = list(self.get_parameter('nodes').value)
        self.grace = float(self.get_parameter('grace_s').value)
        self.give_up = float(self.get_parameter('give_up_s').value)
        self._lifecycle_clients = {name: (self.create_client(GetState, f'/{name}/get_state'),
                               self.create_client(ChangeState, f'/{name}/change_state'))
                        for name in self.nodes}

    def _call(self, client, request, timeout=5.0):
        if not client.wait_for_service(timeout_sec=timeout):
            return None
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
        return future.result() if future.done() else None

    def state(self, name):
        reply = self._call(self._lifecycle_clients[name][0], GetState.Request())
        return None if reply is None else reply.current_state.id

    def advance(self, name, state_id):
        transition_id, label = NEXT_TRANSITION[state_id]
        request = ChangeState.Request()
        request.transition.id = transition_id
        reply = self._call(self._lifecycle_clients[name][1], request, timeout=15.0)
        ok = reply is not None and reply.success
        self.get_logger().warning(
            f'{name} was stuck before {label}; completed it: {"ok" if ok else "failed"}.')

    def run(self):
        start = time.monotonic()
        time.sleep(self.grace)
        while time.monotonic() - start < self.give_up:
            pending = []
            for name in self.nodes:
                state_id = self.state(name)
                if state_id == State.PRIMARY_STATE_ACTIVE:
                    continue
                pending.append(name)
                if state_id in NEXT_TRANSITION:
                    self.advance(name, state_id)
            if not pending:
                self.get_logger().info('All Nav2 servers are active.')
                return 0
            time.sleep(3.0)
        self.get_logger().error(
            f'Nav2 servers still not active after {self.give_up:.0f} s: {pending}. '
            'Restart T5.')
        return 1


def main(args=None):
    rclpy.init(args=args)
    node = Nav2StartupGuard()
    try:
        code = node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()
    raise SystemExit(code)
