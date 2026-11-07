"""Chains of saved mission paths through pass-through stops.

The saved paths form a network: each runs from a station, left by one gate,
to another station, entered by one gate. A robot at a station with a stand
must leave by the gate it entered by; a pass-through stop such as HOME has no
stand and may be left by any gate. When no single saved path leaves the
robot's entry gate toward a destination, a shortest chain through other
stations' gates can (for example Station 3 west gate -> HOME -> Station 4).

A station used as a connector is a waypoint: the robot follows the path to
that station's gate, stops, and continues on the next path from the same gate.
It never enters the station, so there is no docking, precision stage or cargo
operation there. Pure graph search; no robot commands.
"""

import heapq


def _edge(meta, key):
    return {
        'key': key,
        'source': meta['source'], 'target': meta['target'],
        'source_entrance': meta.get('source_entrance'),
        'target_entrance': meta.get('target_entrance'),
        'length_m': float(meta.get('length_m', 0.0)),
        'approved': meta.get('execution_approved') is True,
    }


def network_edges(candidates_by_destination):
    """Edges from paths_to() results for every destination, de-duplicated."""
    edges, seen = [], set()
    for candidates in candidates_by_destination:
        for key, meta, _poses in candidates:
            if key not in seen:
                seen.add(key)
                edges.append(_edge(meta, key))
    return edges


# A waypoint costs a stop at a gate and possibly a change of driving direction
# (about 5-10 s); at route speed (about 1 m/s) that is worth roughly 10 m.
STOP_PENALTY_M = 10.0
# Chains are compared by time. Route travel averages about 1 m/s (1.2 m/s on
# straights, 0.6 m/s in curves), so a metre of route is about a second.
ROUTE_SPEED_MPS = 1.0


def route_time_s(length_m, stops=0):
    """Expected time for ``length_m`` of saved route with ``stops`` waypoints."""
    return length_m / ROUTE_SPEED_MPS + stops * STOP_PENALTY_M / ROUTE_SPEED_MPS


def plan_route_chain(edges, start_station, start_gate, destination, pass_through,
                     stop_penalty_m=STOP_PENALTY_M):
    """Shortest chain leaving by ``start_gate`` (see plan_route_chain_from)."""
    result = plan_route_chain_from(edges, start_station, {start_gate: 0.0}, destination,
                                   pass_through, stop_penalty_m=stop_penalty_m)
    return None if result is None else result[0]


def plan_route_chain_from(edges, start_station, start_gates, destination, pass_through,
                          stop_penalty_m=STOP_PENALTY_M):
    """Return ``(stops, start gate, cost)`` of the cheapest chain, or None.

    ``start_gates`` maps each gate the robot may leave by to its extra cost
    (for example the time to turn in place to face it); the gate it entered
    by costs 0, and None means a pass-through stop left by any gate.
    Dijkstra over (station, gate): the result has the lowest total cost, path
    lengths plus ``stop_penalty_m`` per intermediate stop (a metre is about a
    second, ROUTE_SPEED_MPS). A one-element ``stops`` means a direct path.
    """
    pass_through = set(pass_through)
    queue, best, counter = [], {}, 0
    for gate, extra in start_gates.items():
        start = (start_station, gate)
        if extra < best.get(start, float('inf')):
            best[start] = extra
            heapq.heappush(queue, (extra, counter, start, (), gate))
            counter += 1
    while queue:
        cost, _, (station, gate), stops, first_gate = heapq.heappop(queue)
        if stops and stops[-1] == destination:
            return list(stops), first_gate, cost
        if cost > best.get((station, gate), float('inf')):
            continue
        # Docked at a pass-through stop the robot may leave by any gate. At a
        # waypoint it only stops at the gate it arrived at (no docking), so
        # the next path must leave from that same gate.
        free_exit = station in pass_through and not stops
        for edge in edges:
            if (not edge['approved'] or edge['source'] != station or
                    (not free_exit and gate is not None and
                     edge['source_entrance'] != gate)):
                continue
            target = edge['target']
            node = (target, edge['target_entrance'])
            new_cost = cost + edge['length_m'] + (stop_penalty_m if stops else 0.0)
            if new_cost < best.get(node, float('inf')):
                best[node] = new_cost
                heapq.heappush(queue, (new_cost, counter, node, stops + (target,), first_gate))
                counter += 1
    return None
