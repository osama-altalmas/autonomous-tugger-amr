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


def plan_route_chain(edges, start_station, start_gate, destination, pass_through,
                     stop_penalty_m=STOP_PENALTY_M):
    """Return the stops of the shortest chain, ending with ``destination``.

    ``start_gate`` is the gate the robot entered its station by (None at a
    pass-through stop). Dijkstra over (station, gate): the result has the
    lowest total cost, path lengths plus ``stop_penalty_m`` per intermediate
    stop. Returns None when no chain of approved paths exists. A one-element
    result means the best connection is a direct path.
    """
    pass_through = set(pass_through)
    start = (start_station, start_gate)
    queue = [(0.0, 0, start, ())]
    best = {start: 0.0}
    counter = 1
    while queue:
        cost, _, (station, gate), stops = heapq.heappop(queue)
        if stops and stops[-1] == destination:
            return list(stops)
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
                heapq.heappush(queue, (new_cost, counter, node, stops + (target,)))
                counter += 1
    return None
