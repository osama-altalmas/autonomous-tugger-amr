"""Fault testing only: apply scheduled faults during a mission_runner run.

A schedule is a JSON file:

  {"setup":  [action, ...],                 # before the first mission
   "events": [{"leg": 2, "owner": "NAV", "after_s": 20, "do": [action, ...]},
              {"leg": 2, "after_event": 0, "after_s": 30, "do": [...]}]}

An event fires once: during that leg (1 = first destination), once the
motion owner is the given one (any owner if left out) for after_s seconds of
wall time, or after_s seconds after another event (by index) fired.

Actions (the isolated Gazebo world only):
  {"spawn": NAME, "size": [sx, sy, sz], "at": [x, y], "yaw": 0,
   "ahead": [forward_m, left_m]  (instead of "at": relative to the robot),
   "color": [r, g, b], "z": 0,    a static box resting on the floor (z = its base);
   "mu": 0.3}                     with mu: a floor patch with that friction
  {"cover": NAME, "size": [sx, sy], "at": [x, y]}  floor-coloured tape or
                                  dirt lying flat on the floor (a QR cover)
  {"move": MODEL, "to": [x, y, yaw], "z": 0}
  {"remove": NAME}
  {"walk": NAME, "path": [[x, y], ...], "speed": 1.0, "pause_s": [0, ...]}
                                  a person walking through the points
  {"walk_rel": NAME, "points": [[forward_m, left_m], ...], "speed": 1.0,
   "pause_s": [...]}              a walk given relative to the robot when it fires
  {"walk_across": NAME, "ahead_m": 4.0, "half_width_m": 3.0, "speed": 1.0,
   "stand_s": 0}                  a person crossing the robot's path ahead
                                  (stand_s: stop in the path for that long)
  {"relay": {fault relay command}}  scanner or IMU fault (fault_relay.py)
  {"push": [dx, dy, dyaw]}        the robot moved by hand (map frame)
  {"kill": "refined_localization"}  the process of that node dies
  {"start": "refined_localization"} start that node again
  {"note": "text"}
"""
import json
import math
import os
import signal
import subprocess
import threading
import time
from pathlib import Path

WORLD = 'tugger_test'
PERSON = (0.5, 0.5, 1.7)
FLOOR_RGB = (0.55, 0.55, 0.52)
RESTART = {
    'refined_localization': ['ros2', 'run', 'tugger_control', 'refined_localization',
                             '--ros-args', '-r', '__node:=tugger_refined_localization'],
}


def _gz(service, reqtype, request, timeout_ms=5000):
    assert os.environ.get('GZ_PARTITION', '').startswith('tugger_validation_')
    return subprocess.run(
        ['gz', 'service', '-s', f'/world/{WORLD}/{service}', '--reqtype', reqtype,
         '--reptype', 'gz.msgs.Boolean', '--timeout', str(timeout_ms), '--req', request],
        capture_output=True, text=True)


def spawn_box(name, size, x, y, yaw=0.0, z=0.0, color=(0.2, 0.4, 0.8), collide=True, mu=None):
    sx, sy, sz = size
    shape = f'<geometry><box><size>{sx} {sy} {sz}</size></box></geometry>'
    r, g, b = color
    surface = ('' if mu is None else
               f'<surface><friction><ode><mu>{mu}</mu><mu2>{mu}</mu2></ode></friction></surface>')
    collision = f'<collision name="c">{shape}{surface}</collision>' if collide else ''
    sdf = (f"<sdf version='1.9'><model name='{name}'><static>true</static>"
           f"<pose>{x} {y} {z + sz / 2} 0 0 {yaw}</pose><link name='l'>{collision}"
           f"<visual name='v'><cast_shadows>false</cast_shadows>{shape}<material>"
           f"<ambient>{r} {g} {b} 1</ambient><diffuse>{r} {g} {b} 1</diffuse>"
           "</material></visual></link></model></sdf>")
    escaped = sdf.replace('"', '\\"')
    return _gz('create', 'gz.msgs.EntityFactory', f'sdf: "{escaped}"', 10000)


def set_pose(name, x, y, yaw=0.0, z=0.0):
    return _gz('set_pose', 'gz.msgs.Pose',
               f'name: "{name}" position {{x: {x} y: {y} z: {z}}} '
               f'orientation {{z: {math.sin(yaw / 2)} w: {math.cos(yaw / 2)}}}', 2000)


def remove(name):
    return _gz('remove', 'gz.msgs.Entity', f'name: "{name}" type: MODEL')


def _node_pids(executable):
    """PIDs of this isolated domain's processes running that executable."""
    domain = os.environ.get('ROS_DOMAIN_ID')
    pids = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            cmd = (proc / 'cmdline').read_bytes().split(b'\0')
            env = (proc / 'environ').read_bytes().split(b'\0')
        except OSError:
            continue
        if not any(part.decode(errors='ignore').endswith(f'/{executable}') for part in cmd):
            continue
        if f'ROS_DOMAIN_ID={domain}'.encode() in env:
            pids.append(int(proc.name))
    return pids


class FaultSchedule:
    def __init__(self, runner, path):
        spec = json.loads(Path(path).read_text())
        self.runner = runner
        self.setup = spec.get('setup', [])
        self.events = spec.get('events', [])
        self.fired = {}             # event index -> wall time it fired
        self.log = []               # [wall_s, sim_s, leg, true pose, text]
        self.leg = 0
        self._owner = (None, 0.0)
        self._walkers = []
        self._children = []
        self._t0 = time.monotonic()
        from std_msgs.msg import String
        from ros_gz_interfaces.msg import Contacts
        rec = runner.recorder
        self._relay_pub = rec.create_publisher(String, '/fault/command', 10)
        # What the robot did about it: safety state changes, localization
        # warnings and any contact of the robot's bumper.
        self.safety = []            # [sim_s, state, reason, true pose]
        self.contacts = []          # [sim_s, other body, true pose]
        self._safety_last = None
        rec.create_subscription(String, '/tugger/safety/state', self._on_safety, 10)
        rec.create_subscription(String, '/tugger/localization/warning',
                                lambda m: self.note(f'LOCALIZATION WARNING: {m.data}'), 10)
        rec.create_subscription(Contacts, '/tugger/bumper/contacts', self._on_contact, 10)
        self._gz_node = None
        try:
            from gz.transport import Node as GzNode
            self._gz_node = GzNode()
        except ImportError:
            pass

    def _sim(self):
        return round(self.runner.recorder.get_clock().now().nanoseconds * 1e-9, 2)

    def _pose(self):
        pose = self.runner.recorder.pose
        return [round(v, 3) for v in pose] if pose is not None else None

    def _on_safety(self, msg):
        data = json.loads(msg.data)
        key = (data['state'], data['reason'].split(': object at')[0])
        if key != self._safety_last:
            self._safety_last = key
            self.safety.append([self._sim(), data['state'], data['reason'], self._pose()])

    def _on_contact(self, msg):
        for contact in msg.contacts:
            names = (contact.collision1.name, contact.collision2.name)
            other = next((n for n in names if 'autonomous_tugger' not in n), names[1])
            if 'ground' in other or 'floor' in other:
                continue
            if not self.contacts or self.contacts[-1][1] != other or \
                    self._sim() - self.contacts[-1][0] > 1.0:
                self.contacts.append([self._sim(), other, self._pose()])
                print(f'[fault] CONTACT with {other} at {self._pose()}', flush=True)

    # ------------------------------------------------------------------ log
    def note(self, text):
        rec = self.runner.recorder
        sim = rec.get_clock().now().nanoseconds * 1e-9
        pose = [round(v, 3) for v in rec.pose] if rec.pose is not None else None
        self.log.append([round(time.monotonic() - self._t0, 1), round(sim, 2),
                         self.leg, pose, text])
        print(f'[fault] leg {self.leg}: {text}', flush=True)

    # ------------------------------------------------------------ schedule
    def start(self):
        for action in self.setup:
            self._do(action)

    def begin_leg(self, leg):
        self.leg = leg

    def tick(self):
        owner = self.runner.recorder.owner
        if owner != self._owner[0]:
            self._owner = (owner, time.monotonic())
        now = time.monotonic()
        for index, event in enumerate(self.events):
            if index in self.fired or event.get('leg', self.leg) != self.leg:
                continue
            if 'after_event' in event:
                start = self.fired.get(event['after_event'])
                if start is None or now - start < event.get('after_s', 0.0):
                    continue
            else:
                if 'owner' in event and owner != event['owner']:
                    continue
                if now - self._owner[1] < event.get('after_s', 0.0):
                    continue
            self.fired[index] = now
            for action in event['do']:
                self._do(action)

    def finish(self):
        for walker in self._walkers:
            walker.join(timeout=30)
        for child in self._children:
            child.send_signal(signal.SIGINT)

    # ------------------------------------------------------------- actions
    def _robot(self):
        return self.runner.recorder.pose

    def _ahead(self, forward, left):
        """A point ``forward`` m along the route being driven, ``left`` m to its side.

        Along the followed path when there is one (the robot often drives
        rear first, and routes bend); else along the direction of travel.
        """
        x, y, yaw = self._robot()
        bridge = self.runner.bridge
        path = (getattr(bridge, '_followed_path', None)
                if getattr(bridge, '_active_follow', None) is not None else None)
        if path is not None and len(path.poses) >= 2:
            pts = [(p.pose.position.x, p.pose.position.y) for p in path.poses]
            i = min(range(len(pts)), key=lambda k: math.dist(pts[k], (x, y)))
            travelled = 0.0
            while i + 1 < len(pts) and travelled + math.dist(pts[i], pts[i + 1]) < forward:
                travelled += math.dist(pts[i], pts[i + 1])
                i += 1
            if i + 1 < len(pts):
                (ax, ay), (bx, by) = pts[i], pts[i + 1]
                seg = math.dist(pts[i], pts[i + 1]) or 1e-9
                f = min(1.0, (forward - travelled) / seg)
                px, py = ax + (bx - ax) * f, ay + (by - ay) * f
                heading = math.atan2(by - ay, bx - ax)
                c, s = math.cos(heading), math.sin(heading)
                return px - s * left, py + c * left, heading
        if self.runner.recorder._speed < -0.02:
            yaw += math.pi          # driving rear first
        c, s = math.cos(yaw), math.sin(yaw)
        return x + c * forward - s * left, y + s * forward + c * left, yaw

    def _do(self, action):
        if 'note' in action:
            self.note(action['note'])
        elif 'spawn' in action:
            if 'ahead' in action:
                x, y, yaw = self._ahead(*action['ahead'])
                yaw += action.get('yaw', 0.0)
            else:
                (x, y), yaw = action['at'], action.get('yaw', 0.0)
            done = spawn_box(action['spawn'], action['size'], x, y, yaw, action.get('z', 0.0),
                             action.get('color', (0.2, 0.4, 0.8)), mu=action.get('mu'))
            self.note(f"spawned {action['spawn']} {action['size']} at "
                      f"({x:.2f}, {y:.2f}, {math.degrees(yaw):.0f} deg) {done.stdout.strip()}")
        elif 'cover' in action:
            sx, sy = action['size']
            x, y = action['at']
            # 1 mm tape just above the QR plane (QR top at z = 0.005 m).
            done = spawn_box(action['cover'], (sx, sy, 0.001), x, y, action.get('yaw', 0.0),
                             0.0055, action.get('color', FLOOR_RGB), collide=False)
            self.note(f"covered ({x:.3f}, {y:.3f}) with {sx * 100:.1f} x {sy * 100:.1f} cm "
                      f"({action['cover']}) {done.stdout.strip()}")
        elif 'move' in action:
            x, y, yaw = action['to']
            done = set_pose(action['move'], x, y, yaw, action.get('z', 0.0))
            self.note(f"moved {action['move']} to ({x}, {y}, {math.degrees(yaw):.1f} deg) "
                      f"{done.stdout.strip()}")
        elif 'remove' in action:
            remove(action['remove'])
            self.note(f"removed {action['remove']}")
        elif 'walk' in action:
            self._start_walker(action['walk'], action['path'], action.get('speed', 1.0),
                               action.get('pause_s'))
        elif 'walk_across' in action:
            ahead = action.get('ahead_m', 4.0)
            half = action.get('half_width_m', 3.0)
            a = self._ahead(ahead, half)[:2]
            m = self._ahead(ahead, 0.0)[:2]
            b = self._ahead(ahead, -half)[:2]
            stand = action.get('stand_s', 0.0)
            self._start_walker(action['walk_across'], [a, m, b], action.get('speed', 1.0),
                               [0.0, stand, 0.0])
        elif 'walk_rel' in action:
            points = [self._ahead(f, l)[:2] for f, l in action['points']]
            self._start_walker(action['walk_rel'], points, action.get('speed', 1.0),
                               action.get('pause_s'))
        elif 'relay' in action:
            from std_msgs.msg import String
            self._relay_pub.publish(String(data=json.dumps(action['relay'])))
            self.note(f"sensor fault: {action['relay']}")
        elif 'push' in action:
            dx, dy, dyaw = action['push']
            x, y, yaw = self._robot()
            self.runner._set_pose(x + dx, y + dy, yaw + dyaw)
            self.note(f'robot pushed by ({dx}, {dy}) m, {math.degrees(dyaw):.0f} deg '
                      f'from ({x:.2f}, {y:.2f})')
        elif 'kill' in action:
            pids = _node_pids(action['kill'])
            for pid in pids:
                os.kill(pid, signal.SIGKILL)
            self.note(f"killed {action['kill']} (pids {pids})")
        elif 'start' in action:
            log = open(Path(os.environ.get('TUGGER_VALIDATION_DIR', '/tmp')) /
                       f"restart_{action['start']}.log", 'w')
            self._children.append(subprocess.Popen(RESTART[action['start']], stdout=log,
                                                   stderr=subprocess.STDOUT))
            self.note(f"started {action['start']} again")
        else:
            raise ValueError(f'unknown fault action {action}')

    def _fast_pose(self, name, x, y, z=0.0):
        """Move a model through gz-transport directly (a subprocess per step is slow)."""
        if self._gz_node is None:
            set_pose(name, x, y, 0.0, z)
            return
        from gz.msgs.boolean_pb2 import Boolean
        from gz.msgs.pose_pb2 import Pose
        request = Pose()
        request.name = name
        request.position.x, request.position.y, request.position.z = x, y, z
        request.orientation.w = 1.0
        self._gz_node.request(f'/world/{WORLD}/set_pose', request, Pose, Boolean, 1000)

    def _start_walker(self, name, path, speed, pauses):
        clock = self.runner.recorder.get_clock()
        sim_now = lambda: clock.now().nanoseconds * 1e-9   # noqa: E731
        pauses = pauses or [0.0] * len(path)
        x0, y0 = path[0]
        spawn_box(name, PERSON, x0, y0, color=(0.9, 0.5, 0.1))
        self.note(f'person {name} walks {[(round(p[0], 2), round(p[1], 2)) for p in path]} '
                  f'at {speed} m/s, pauses {pauses}')

        def walk():
            time.sleep(1.0)
            for (ax, ay), (bx, by), pause in zip(path, path[1:], pauses[1:] + [0.0]):
                duration = max(0.1, math.hypot(bx - ax, by - ay) / speed)
                start = sim_now()
                while True:
                    f = min(1.0, (sim_now() - start) / duration)
                    self._fast_pose(name, ax + (bx - ax) * f, ay + (by - ay) * f,
                                    PERSON[2] / 2)
                    if f >= 1.0:
                        break
                    time.sleep(0.05)
                end = sim_now() + pause
                while sim_now() < end:
                    time.sleep(0.1)
            remove(name)
        thread = threading.Thread(target=walk, daemon=True)
        thread.start()
        self._walkers.append(thread)
