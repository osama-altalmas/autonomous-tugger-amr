#!/usr/bin/env python3
"""Put test obstacles into a running Gazebo; for demonstrations and recording.

Positions suit the mission HOME -> S3, in this order:

  block   a box in the side lane the route starts on: too little room to go
          around, so after 10 s the robot finds another path (the middle aisle)
  cross   run this as soon as the robot starts along the middle aisle: a person
          walks across it at x = 7 and leaves; the robot waits, then continues
  box     a box further along the middle aisle at x = 12, out of sensor range
          when the robot replans: it waits 10 s, then goes around it
  pace    a person keeps walking across the route (the 30 second rule)
  clear   remove everything this tool added

Walking uses simulated time, so it looks right however slowly Gazebo runs.
"""
import math
import sys
import time

from gz.msgs.boolean_pb2 import Boolean
from gz.msgs.entity_factory_pb2 import EntityFactory
from gz.msgs.entity_pb2 import Entity
from gz.msgs.pose_pb2 import Pose
from gz.msgs.world_stats_pb2 import WorldStatistics
from gz.transport import Node

WORLD = 'tugger_test'
PERSON = (0.5, 0.5, 1.7)
CROSS_X, CROSS_FROM_Y, CROSS_TO_Y = 7.0, -3.0, 3.0
WALK_SPEED = 1.0       # m/s of simulated time
OBJECTS = {'block': ((0.4, 0.4, 0.5), (4.47, 3.0)),
           'box': ((0.4, 0.4, 0.5), (12.0, 0.0))}
ALL_NAMES = ('block', 'box', 'walker')

node = Node()
sim_time = [None]
node.subscribe(WorldStatistics, '/stats', lambda m: sim_time.__setitem__(
    0, m.sim_time.sec + m.sim_time.nsec / 1e9))
time.sleep(1.5)


def spawn(name, size, x, y):
    sx, sy, sz = size
    shape = f'<geometry><box><size>{sx} {sy} {sz}</size></box></geometry>'
    request = EntityFactory()
    request.sdf = (
        f"<sdf version='1.9'><model name='{name}'><static>true</static>"
        f"<pose>{x} {y} {sz / 2} 0 0 0</pose><link name='l'>"
        f"<collision name='c'>{shape}</collision><visual name='v'>{shape}"
        "<material><ambient>0.1 0.3 0.9 1</ambient><diffuse>0.1 0.3 0.9 1</diffuse>"
        "</material></visual></link></model></sdf>")
    node.request(f'/world/{WORLD}/create', request, EntityFactory, Boolean, 5000)
    print(f'{name}: sent to Gazebo')     # the reply can be slow even when it worked


def move(name, x, y, z):
    request = Pose()
    request.name = name
    request.position.x, request.position.y, request.position.z = x, y, z
    request.orientation.w = 1.0
    node.request(f'/world/{WORLD}/set_pose', request, Pose, Boolean, 1000)


def wait_for_clock():
    end = time.monotonic() + 10.0
    while sim_time[0] is None and time.monotonic() < end:
        time.sleep(0.05)
    if sim_time[0] is None:
        sys.exit('No simulation clock: is Gazebo running with GZ_IP set?')


def walk(from_y, to_y):
    """Walk the person between two points at a steady simulated speed."""
    duration = abs(to_y - from_y) / WALK_SPEED
    start = sim_time[0]
    while True:
        elapsed = sim_time[0] - start
        fraction = min(1.0, elapsed / duration)
        move('walker', CROSS_X, from_y + (to_y - from_y) * fraction, PERSON[2] / 2)
        if fraction >= 1.0:
            return
        time.sleep(0.03)


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in (*OBJECTS, 'cross', 'pace', 'clear'):
        sys.exit(__doc__)
    what = sys.argv[1]
    if what in OBJECTS:
        size, (x, y) = OBJECTS[what]
        spawn(what, size, x, y)
    elif what == 'clear':
        for name in ALL_NAMES:
            request = Entity()
            request.name = name
            request.type = Entity.MODEL
            node.request(f'/world/{WORLD}/remove', request, Entity, Boolean, 3000)
        print('cleared')
    else:
        wait_for_clock()
        spawn('walker', PERSON, CROSS_X, CROSS_FROM_Y)
        time.sleep(1.0)
        walk(CROSS_FROM_Y, CROSS_TO_Y)
        if what == 'pace':
            print('pacing: press Ctrl+C to stop')
            try:
                while True:
                    walk(CROSS_TO_Y, CROSS_FROM_Y)
                    walk(CROSS_FROM_Y, CROSS_TO_Y)
            except KeyboardInterrupt:
                pass
        request = Entity()
        request.name = 'walker'
        request.type = Entity.MODEL
        node.request(f'/world/{WORLD}/remove', request, Entity, Boolean, 3000)
        print('the person has crossed and left')
    sys.stdout.flush()
    # Gazebo's transport threads can still be running: leave without teardown.
    import os
    os._exit(0)


if __name__ == '__main__':
    main()
