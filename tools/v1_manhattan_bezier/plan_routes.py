#!/usr/bin/env python3
"""Offline V1 facility-route generator. Never sends robot motion commands."""
import argparse
import hashlib
import heapq
import json
import math
from pathlib import Path
import numpy as np
import yaml
from PIL import Image
from scipy.ndimage import distance_transform_edt

STEP = 0.025


def load_map(map_yaml, clearance):
    spec = yaml.safe_load(Path(map_yaml).read_text())
    im = np.asarray(Image.open(Path(map_yaml).parent / spec['image']).convert('L'))
    # Nav2 map_server uses lower-left map origin; PGM's first row is top.
    im = np.flipud(im)
    intensity = im.astype(float) / 255.0
    probability = intensity if spec.get('negate', 0) else (1.0 - intensity)
    occupied = (probability >= spec['occupied_thresh']) | (probability > spec['free_thresh'])
    # Intermediate/unknown pixels are not traversable.
    resolution = float(spec['resolution'])
    free_dist = distance_transform_edt(~occupied) * resolution
    safe = free_dist > clearance
    # Exclude borders: full vehicle must remain inside map.
    border = max(1, math.ceil(clearance / resolution))
    safe[:border, :] = False; safe[-border:, :] = False
    safe[:, :border] = False; safe[:, -border:] = False
    return spec, safe


def cell(p, spec, shape):
    resolution = float(spec['resolution'])
    origin = spec['origin']
    theta = origin[2] if len(origin) > 2 else 0.0
    dx, dy = p[0]-origin[0], p[1]-origin[1]
    gx = math.floor((math.cos(theta)*dx + math.sin(theta)*dy)/resolution)
    gy = math.floor((-math.sin(theta)*dx + math.cos(theta)*dy)/resolution)
    if not (0 <= gx < shape[1] and 0 <= gy < shape[0]):
        raise ValueError(f'Point outside map: {p}')
    return int(gx), int(gy)


def world(q, spec):
    r = float(spec['resolution']); ox, oy, theta = spec['origin']
    dx, dy = (q[0]+.5)*r, (q[1]+.5)*r
    return [ox+math.cos(theta)*dx-math.sin(theta)*dy,
            oy+math.sin(theta)*dx+math.cos(theta)*dy]


def cardinal_astar(start, goal, safe, penalties=None):
    sx, sy = start; gx, gy = goal
    if not safe[sy, sx] or not safe[gy, gx]:
        raise ValueError('Start or goal is not footprint-clear on the static map')
    if start == goal: return [start]
    # Direction state minimizes distance and adds a small preference for fewer corners.
    pq = [(abs(gx-sx)+abs(gy-sy), 0.0, sx, sy, -1)]
    best = {(sx,sy,-1):0.0}; parents = {}
    dirs = [(1,0),(-1,0),(0,1),(0,-1)]
    h,w=safe.shape
    terminal=None
    while pq:
        _, g, x, y, d = heapq.heappop(pq)
        state=(x,y,d)
        if g > best.get(state,float('inf'))+1e-9: continue
        if (x,y)==goal: terminal=state; break
        for nd,(dx,dy) in enumerate(dirs):
            xx,yy=x+dx,y+dy
            if not (0<=xx<w and 0<=yy<h and safe[yy,xx]): continue
            ng=g+1+(0.2 if d!=-1 and nd!=d else 0) + (float(penalties[yy,xx]) if penalties is not None else 0.0)
            nxt=(xx,yy,nd)
            if ng+1e-9 < best.get(nxt,float('inf')):
                best[nxt]=ng; parents[nxt]=state
                heapq.heappush(pq,(ng+abs(gx-xx)+abs(gy-yy),ng,xx,yy,nd))
    if terminal is None: raise ValueError('No cardinal-grid route exists')
    path=[]; s=terminal
    while True:
        path.append((s[0],s[1]))
        if s==(sx,sy,-1): break
        s=parents[s]
    return path[::-1]


def compress(points):
    if len(points)<3: return points
    out=[points[0]]
    for i in range(1,len(points)-1):
        a=np.asarray(points[i])-np.asarray(points[i-1]); b=np.asarray(points[i+1])-np.asarray(points[i])
        if abs(float(a[0]*b[1]-a[1]*b[0]))>1e-7 or float(a@b)<0: out.append(points[i])
    out.append(points[-1]); return out


def sampled_line(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float)
    n=max(1,math.ceil(float(np.linalg.norm(b-a))/STEP))
    return [(a+(b-a)*t/n).tolist() for t in range(n+1)]


# Weight of the middle control point of a rational quadratic Bezier that is an
# exact 90-degree circular arc: cos(45 degrees).
ARC_WEIGHT = math.cos(math.pi / 4.0)


def bezier_arc(a, b, u, v):
    """90-degree corner from a (heading u) to b (heading v) as a circular arc.

    A rational quadratic Bezier (control points a, the corner point, b; middle
    weight cos 45 degrees) is an exact circle: the curvature, and so the
    steering, is constant through the corner. The arc is tangent to both
    straights; its radius is the distance from a (or b) to the corner point.
    It replaced a quintic Bezier whose curvature rose, dipped and rose again
    (steering about 21 -> 12 -> 23 degrees for a 1 m corner, 2026-11-13); a
    1 m corner now holds about 16 degrees.
    """
    a = np.asarray(a, float); b = np.asarray(b, float)
    u = np.asarray(u, float); v = np.asarray(v, float)
    if abs(float(u @ v)) > 1e-5:
        raise ValueError('Circular-arc corners need a 90-degree turn')
    # Corner point where the two straights meet; radius = |corner - a|.
    corner = a + u * float(np.dot(b - a, u))
    radius = float(np.linalg.norm(corner - a))
    if radius < 1e-6 or abs(float(np.linalg.norm(b - corner)) - radius) > 1e-6:
        raise ValueError('Corner arc needs equal distances to the corner point')
    n = max(30, math.ceil(radius * math.pi / 2.0 / STEP))
    result = []
    for t in np.linspace(0, 1, n + 1):
        w0, w1, w2 = (1 - t) ** 2, 2 * t * (1 - t) * ARC_WEIGHT, t * t
        result.append(((w0 * a + w1 * corner + w2 * b) / (w0 + w1 + w2)).tolist())
    return result


def smooth(points, trim, final_straight=0.0):
    """Straight runs joined by circular-arc corners of radius up to ``trim``.

    ``final_straight`` (m): the last corner is rounded no more than leaves this
    much straight before the route's end, where the follower settles its
    heading before the station hand-over. The 1.0 m rounding otherwise took
    1.0 m of the planner's final run and left 1.475 m on two HOME routes.
    """
    if len(points)<2: raise ValueError('At least two route points required')
    out=[list(points[0])]; cursor=np.asarray(points[0],float)
    for i in range(1,len(points)-1):
        a,b,c=map(lambda x:np.asarray(x,float),points[i-1:i+2])
        v1=b-a;v2=c-b; l1=float(np.linalg.norm(v1));l2=float(np.linalg.norm(v2))
        if min(l1,l2)<1e-8: raise ValueError('Zero-length path segment')
        u=v1/l1;v=v2/l2
        if abs(float(u@v))>1e-5: raise ValueError('Only 90-degree corners allowed')
        d=min(trim,0.45*l1,0.45*l2)
        if i==len(points)-2 and final_straight>0.0 and l2-final_straight>0.05:
            d=min(d,l2-final_straight)
        entry=b-d*u;depart=b+d*v
        out+=sampled_line(cursor,entry)[1:]
        out+=bezier_arc(entry,depart,u,v)[1:]
        cursor=depart
    out+=sampled_line(cursor,points[-1])[1:]
    return [out[0]]+[p for p,q in zip(out[1:],out[:-1]) if math.dist(p,q)>1e-8]


def curvature(path):
    p=np.asarray(path,float)
    if len(p)<3: return 0.0
    a=p[1:-1]-p[:-2];b=p[2:]-p[1:-1];c=p[2:]-p[:-2]
    denom=np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)*np.linalg.norm(c,axis=1)
    cross=np.abs(a[:,0]*b[:,1]-a[:,1]*b[:,0])
    val=np.divide(2*cross,denom,out=np.zeros_like(cross),where=denom>1e-12)
    return float(val.max())


def check_samples(points,spec,safe):
    for p in points:
        try: x,y=cell(p,spec,safe.shape)
        except ValueError: return False
        if not safe[y,x]: return False
    return True


def _candidate_route(start, goal, spec, safe, trim, max_curvature, penalties):
    a=cell(start,spec,safe.shape); b=cell(goal,spec,safe.shape)
    raw=cardinal_astar(a,b,safe,penalties=penalties)
    points=compress([world(q,spec) for q in raw])
    points[0]=list(start); points[-1]=list(goal)
    if any(abs(q[0]-p[0])>1e-5 and abs(q[1]-p[1])>1e-5
           for p,q in zip(points[:-1],points[1:])):
        raise ValueError('Stations must align to cardinal grid centerlines')
    last_error='No safe smoothed candidate'
    for t in [trim, trim*.75, trim*.5, trim*.35]:
        try:
            path=smooth(points,t)
        except ValueError as e:
            last_error=str(e); continue
        k=curvature(path)
        if k<=max_curvature and check_samples(path,spec,safe):
            poses=[]
            for i,p in enumerate(path):
                q=path[i+1] if i+1<len(path) else path[i-1]
                vec=np.asarray(q)-np.asarray(p) if i+1<len(path) else np.asarray(p)-np.asarray(q)
                poses.append([round(p[0],4),round(p[1],4),round(math.atan2(vec[1],vec[0]),6)])
            return {'status':'VALIDATED_STATIC_CLEARANCE','waypoints':points,'poses':poses,
                    'max_curvature_1_per_m':round(k,5),
                    'path_length_m':round(sum(math.dist(x,y) for x,y in zip(path,path[1:])),3),
                    'corner_trim_m':t}
        last_error=f'curvature={k:.3f}, static_clear={check_samples(path,spec,safe)}'
    return points, last_error


def route(start,goal,spec,safe,trim,max_curvature,max_alternatives=8):
    """Find a cardinal route, smoothing and checking every candidate.

    On failure, disfavor previously chosen corners and search a different
    corridor. This is bounded heuristic diversification, NOT a completeness
    guarantee. Returns only paths that pass clearance + curvature checks.
    """
    if not 1 <= max_alternatives <= 40:
        raise ValueError('max_alternatives must be between 1 and 40')
    start_cell=cell(start,spec,safe.shape)
    goal_cell=cell(goal,spec,safe.shape)
    if not safe[start_cell[1], start_cell[0]] or not safe[goal_cell[1],goal_cell[0]]:
        raise ValueError('Start or goal not clear on static map')
    penalties=np.zeros(safe.shape,dtype=np.float32)
    tested=set()
    for attempt in range(max_alternatives):
        candidate=_candidate_route(start,goal,spec,safe,trim,max_curvature,penalties)
        if isinstance(candidate,dict):
            candidate['search_attempts']=attempt+1
            candidate['validation']='Conservative circular static-clearance approximation; no swept rectangular-footprint verification'
            return candidate
        corners,why=candidate
        key=tuple((round(x,4),round(y,4)) for x,y in corners)
        # Discourage *all* interior corners in this corridor, not start/goal.
        # Fixed radius of approximately a meter around corners; never modify
        # the traversable map itself or relax collision checking.
        r=max(2,int(math.ceil(0.8/float(spec['resolution']))))
        for point in corners[1:-1]:
            x,y=cell(point,spec,safe.shape)
            yy,xx=np.ogrid[max(0,y-r):min(safe.shape[0],y+r+1),
                           max(0,x-r):min(safe.shape[1],x+r+1)]
            mask=(xx-x)**2+(yy-y)**2 <= r*r
            block=penalties[max(0,y-r):min(safe.shape[0],y+r+1),
                            max(0,x-r):min(safe.shape[1],x+r+1)]
            block[mask]+=4.0
        if len(corners)<=2 or key in tested:
            # Straight path blocked or repeated route: a different route
            # may require a detour. Penalize its middle cells as well.
            x0,y0=cell(corners[0],spec,safe.shape)
            x1,y1=cell(corners[-1],spec,safe.shape)
            x,y=(x0+x1)//2,(y0+y1)//2
            penalties[max(0,y-r):min(safe.shape[0],y+r+1),
                      max(0,x-r):min(safe.shape[1],x+r+1)] += 4.0
        tested.add(key)
    raise ValueError(f'No validated smoothed route in {max_alternatives} corridor attempts; last reason: {why}')

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--map',required=True,type=Path)
    ap.add_argument('--stations',required=True,type=Path,help='JSON station_id -> [x,y]')
    ap.add_argument('--start',required=True)
    ap.add_argument('--goal',required=True)
    ap.add_argument('--out',required=True,type=Path)
    ap.add_argument('--clearance',type=float,default=0.9,help='Conservative circular footprint radius in meters')
    ap.add_argument('--corner-trim',type=float,default=1.5)
    ap.add_argument('--max-curvature',type=float,default=1.6)
    ap.add_argument('--max-alternatives',type=int,default=8)
    args=ap.parse_args()
    stations=json.loads(args.stations.read_text())
    if args.start not in stations or args.goal not in stations: ap.error('Both station IDs must exist')
    if args.clearance<=0 or args.corner_trim<=0 or args.max_curvature<=0: ap.error('Positive geometry parameters required')
    spec,safe=load_map(args.map,args.clearance)
    result=route(stations[args.start],stations[args.goal],spec,safe,args.corner_trim,args.max_curvature,args.max_alternatives)
    result.update({'start':args.start,'goal':args.goal,'map_sha256':hashlib.sha256(args.map.read_bytes()+(args.map.parent/spec['image']).read_bytes()).hexdigest(),
                   'clearance_radius_m':args.clearance,'max_curvature_limit':args.max_curvature,
                   'note':'OFFLINE ONLY: no localization, docking, runtime obstacle checking, or robot execution validated'})
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(f"Saved {args.out}: {len(result['poses'])} poses, {result['path_length_m']} m; curvature {result['max_curvature_1_per_m']} <= {args.max_curvature}")

if __name__=='__main__': main()
