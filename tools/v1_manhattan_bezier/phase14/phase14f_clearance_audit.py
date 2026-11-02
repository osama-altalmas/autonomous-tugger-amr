#!/usr/bin/env python3
"""Phase 14F: offline 2.5D footprint audit of 5 idealized docking corrections.
Reads Xacro properties, URDF collision shapes and SDF collision boxes.
No ROS dependency; NO motion commands; NOT an execution authorization.
"""
import argparse, json, math, re, xml.etree.ElementTree as ET
from pathlib import Path

def wrap(a): return math.atan2(math.sin(a),math.cos(a))
def rect(x,y,yaw,hx,hy):
 c,s=math.cos(yaw),math.sin(yaw)
 return [(x+c*dx-s*dy,y+s*dx+c*dy) for dx,dy in [(-hx,-hy),(hx,-hy),(hx,hy),(-hx,hy)]]
def overlap(a,b):
 for poly in (a,b):
  for p,q in zip(poly,poly[1:]+poly[:1]):
   axis=(p[1]-q[1],q[0]-p[0]); u=[x*axis[0]+y*axis[1] for x,y in a];v=[x*axis[0]+y*axis[1] for x,y in b]
   if max(u)<min(v) or max(v)<min(u):return False
 return True
def xyz(node,tag='origin'):
 e=node.find(tag)
 return tuple(float(n) for n in e.get('xyz','0 0 0').split()) if e is not None else (0.,0.,0.)
def sdf_pose(node):
 s=node.findtext('pose'); v=list(map(float,s.split())) if s else [0.]*6
 if len(v)!=6 or any(abs(v[i])>1e-8 for i in (3,4)):raise ValueError('Unsupported SDF pose')
 return v
def compose(a,b):
 c,s=math.cos(a[5]),math.sin(a[5]);return [a[0]+c*b[0]-s*b[1],a[1]+s*b[0]+c*b[1],a[2]+b[2],0,0,wrap(a[5]+b[5])]
def get_world_boxes(world,name):
 found=[m for m in world.findall('.//model') if m.get('name')==name]
 if len(found)!=1:raise ValueError(f'Expected unique model {name}, found {len(found)}')
 model=found[0];boxes=[]
 for link in model.findall('link'):
  lp=compose(sdf_pose(model),sdf_pose(link))
  for col in link.findall('collision'):
   b=col.find('geometry/box/size')
   if b is None or not b.text:raise ValueError(f'Unsupported collision in {name}')
   pose=compose(lp,sdf_pose(col));size=list(map(float,b.text.split()))
   boxes.append((pose,size))
 return boxes

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--robot',type=Path,required=True);ap.add_argument('--world',type=Path,required=True);ap.add_argument('--route',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
 args=ap.parse_args();root=ET.parse(args.robot).getroot();world=ET.parse(args.world).getroot();data=json.loads(args.route.read_text());name=data['destination']
 if data.get('schema')!='amr_phase14_two_segment_refined_v1' or data.get('execution_approved') is not False:raise ValueError('Unexpected route schema/authorization')
 if name!='cargo_dropoff_southeast':raise ValueError('This five-case audit is specific to the measured Southeast predock pose; other stations require their own starts')
 props={p.get('name'):float(p.get('value')) for p in root.findall('./{*}property') if re.fullmatch(r'[0-9.+-]+',p.get('value',''))}
 required=['wheelbase','wheel_y','wheel_radius','wheel_width']
 if any(k not in props for k in required):raise ValueError('Robot dimensions missing')
 base_z=xyz(root.find("./joint[@name='base_footprint_joint']"))[2];lift_joint_z=xyz(root.find("./joint[@name='lift_joint']"))[2]
 body=root.find("./link[@name='base_link']/collision");lift=root.find("./link[@name='lift_platform_link']/collision")
 def collision_box(col):
  box=col.find('geometry/box')
  if box is None:raise ValueError('Required body/lift box collision missing')
  return xyz(col),list(map(float,box.get('size').split()))
 bo,bs=collision_box(body);lo,ls=collision_box(lift)
 body_z=(base_z+bo[2]-bs[2]/2,base_z+bo[2]+bs[2]/2)
 lift_z=(base_z+lift_joint_z+lo[2]-ls[2]/2,base_z+lift_joint_z+lo[2]+ls[2]/2)
 wheel_joint=root.find(".//joint[@name='${prefix}_steering_joint']")
 if wheel_joint is None:raise ValueError('Steering wheel macro changed')
 wheel_z=base_z-0.045
 wheel_range=(wheel_z-props['wheel_radius'],wheel_z+props['wheel_radius'])
 legboxes=get_world_boxes(world,name+'_legs')
 if len(legboxes)!=4:raise ValueError('Station legs expected: four')
 table_name=name+'_stand';names={m.get('name') for m in world.findall('.//model')}
 tables=get_world_boxes(world,table_name) if table_name in names else []
 if len(tables)>1:raise ValueError('Unexpected tabletop geometry')
 upper=max(body_z[1],lift_z[1],wheel_range[1]);gap=(tables[0][0][2]-tables[0][1][2]/2)-upper if tables else None
 gx,gy,gh=data['final_docking_pose']
 starts=[]
 for dx in (-.02,-.01,0,.01,.02):
  for dy in (-.02,-.01,0,.01,.02):
   for dh in (-2,-1,0,1,2):
    starts.append((
     f'dx={dx:+.2f}_dy={dy:+.2f}_dh={dh:+.0f}',
     17.500650636+dx,
     -5.963040895+dy,
     -92.82+dh
    ))
 # True shape sweep with conservative XY wheel square overbounds.
 # Clearance margin includes 10 mm inflation on each side of every leg.
 margin=.01
 legpolys=[(rect(p[0],p[1],p[5],s[0]/2+margin,s[1]/2+margin),(p[2]-s[2]/2,p[2]+s[2]/2)) for p,s in legboxes]
 results=[]
 for label,x,y,heading in starts:
  yaw=math.radians(heading);max_lateral=0;min_gap=1e9;coll=[];steps=0;status='INCOMPLETE';peak_delta=0
  for n in range(1500):
   dx,dy=x-gx,y-gy;rem=-(dx*math.cos(gh)+dy*math.sin(gh));lat=-math.sin(gh)*dx+math.cos(gh)*dy;hd=wrap(gh-yaw);err=math.hypot(dx,dy)
   max_lateral=max(max_lateral,abs(lat))
   bodies=[('chassis',rect(x,y,yaw,bs[0]/2,bs[1]/2),body_z),('lowered_lift',rect(x,y,yaw,ls[0]/2,ls[1]/2),lift_z)]
   for fx in (-props['wheelbase']/2,props['wheelbase']/2):
    for fy in (-props['wheel_y'],props['wheel_y']):
     c,s=math.cos(yaw),math.sin(yaw);wx=x+c*fx-s*fy;wy=y+s*fx+c*fy
     radius=props['wheel_radius']+props['wheel_width']/2 # overbound (steering-independent)
     bodies.append(('articulated_wheel_overbound',rect(wx,wy,0,radius,radius),wheel_range))
   for kind,poly,zrange in bodies:
    for legpoly,legz in legpolys:
     if zrange[0]<=legz[1] and zrange[1]>=legz[0] and overlap(poly,legpoly):coll.append({'part':kind,'step':n});break
    if coll:break
   if coll:status='LEG_COLLISION';break
   if err<=.02 and abs(hd)<=math.radians(2):status='IDEAL_KINEMATIC_PASS';break
   if rem<.015 or abs(lat)>.09:status='POSE_OR_CORRIDOR_FAIL';break
   kappa=max(-.9,min(.9,4.0*hd-6.0*lat));peak_delta=max(peak_delta,abs(math.atan(kappa*props['wheelbase']/2)));ds=.002
   mid=yaw+.5*kappa*ds;x+=ds*math.cos(mid);y+=ds*math.sin(mid);yaw=wrap(yaw+kappa*ds);steps+=1
  results.append({'case':label,'status':status,'final_pose':[x,y,yaw],'position_error_m':math.hypot(x-gx,y-gy),'heading_error_deg':abs(math.degrees(wrap(gh-yaw))),'maximum_lateral_error_m':max_lateral,'peak_front_steering_deg_ideal':math.degrees(peak_delta),'detected_collisions':coll,'steps':steps})
 report={'station':name,'robot_geometry_from_xacro':{'wheelbase_m':props['wheelbase'],'wheel_offset_y_m':props['wheel_y'],'wheel_radius_m':props['wheel_radius'],'wheel_width_m':props['wheel_width'],'chassis_z_m':body_z,'lowered_lift_z_m':lift_z,'wheel_z_range_m':wheel_range},'tabletop_nominal_gap_m':gap,'leg_clearance_margin_m':margin,'tests':results,'limitations':['Nominal SDF positions are assumed unchanged; runtime world movement not measured','No timing or steering-rate dynamics simulated','Full robot collision tree, sensors, mast and hitch not modeled','Wheel XY overbound inflated; 2.5D shape model, not full 3D collision engine','Exact 4WS motion_math.py not supplied; curvature-to-steering feasibility not verified','Cargo-lift motion and loads are not checked'],'motion_approved':False}
 args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
 print('PHASE 14F — OFFLINE GEOMETRY AND TRAJECTORY AUDIT')
 print('Station:',name,'| four legs:',len(legboxes),'| tabletop:',bool(tables));print('Chassis z:',body_z,'| lift z:',lift_z,'| wheel z:',wheel_range)
 print('Nominal tabletop vertical gap:', 'N/A (no collision tabletop)' if gap is None else f'{gap*1000:.1f} mm')
 for r in results:print(f"{r['case']:18s} {r['status']:24s} pos={r['position_error_m']*100:.2f}cm heading={r['heading_error_deg']:.2f}deg peak steering={r['peak_front_steering_deg_ideal']:.2f}deg")
 print('Saved:',args.output);print('MOTION AUTHORIZED: NO');print('NOTE: audit covers only modeled chassis, lowered lift, wheels, legs and tabletop; other parts/dynamics unverified')
 if any(r['status']!='IDEAL_KINEMATIC_PASS' for r in results):raise SystemExit(2)
if __name__=='__main__':main()
