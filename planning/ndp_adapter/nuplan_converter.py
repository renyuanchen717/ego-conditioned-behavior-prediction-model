import os,sys,functools
from pathlib import Path
import numpy as np
import torch
REPO=Path(os.environ['NDPNET_ROOT']);sys.path.insert(0,str(REPO))
EXPECTED='c437f3148045b6391a030104bf16005e2d41bee2bb208382fe44fed62413335a'
def resample(q):
 q=np.asarray(q,dtype=float)[:,:2];dist=np.r_[0,np.cumsum(np.linalg.norm(np.diff(q,axis=0),axis=1))];dist,ix=np.unique(dist,return_index=True)
 if len(dist)<2:return q[:1]
 t=np.r_[np.arange(0,dist[-1],2.),dist[-1]]
 return np.column_stack([np.interp(t,dist,q[ix,k]) for k in range(2)])
APIS={}
@functools.lru_cache(maxsize=50000)
def lane_info(city,kind,uid):
 from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
 from nuplan.common.maps.maps_datatypes import SemanticMapLayer as L
 if city not in APIS:APIS[city]=get_maps_api(os.environ['NUPLAN_MAPS_ROOT'],'nuplan-maps-v1.0',city)
 ob=APIS[city].get_map_object(uid,L.LANE if kind==0 else L.LANE_CONNECTOR)
 if ob is None:raise ValueError(f'Native lane missing: {city}/{kind}/{uid}')
 bounds=[]
 for side,key in enumerate(['left_boundary','right_boundary']):
  boundary=getattr(ob,key,None)
  if boundary is not None:bounds.append((side,resample(boundary.linestring.coords)))
 relations=[(str(x.id),1) for x in ob.incoming_edges]+[(str(x.id),2) for x in ob.outgoing_edges]
 if kind==0:
  relations += [(str(x.id),3+i) for i,x in enumerate(ob.adjacent_edges) if x is not None]
 return bounds,relations
def map_graph(raw,origin):
 m=raw['map'];tl=raw['traffic_lights'];ci=int(raw['current_time_index']);polys=[];parts=[];relations=[];lookup={};unknown_speed=0
 def add(q,ptype,speed,features,key=None):
  idx=len(polys);q=np.asarray(q);v=q[1]-q[0]
  polys.append((q[0]-origin,np.arctan2(v[1],v[0]),ptype,speed))
  for coords,typ,side in features:
   coords=np.asarray(coords);delta=np.diff(coords,axis=0)
   if len(delta):parts.append((coords[:-1]-origin,np.arctan2(delta[:,1],delta[:,0]),np.linalg.norm(delta,axis=1),typ,side,idx))
  if key is not None:lookup[key]=idx
  return idx
 for j,(uid,kind) in enumerate(zip(m['ids'],m['kind'])):
  kind=int(kind)
  if kind not in [0,1,3]:continue
  lo,hi=m['point_range'][j];q=resample(m['points'][lo:hi])
  if len(q)<2:continue
  if kind in [0,1]:
   bounds,edges=lane_info(raw['map_name'],kind,str(uid));speed=float(m['speed_limit_mps'][j]);unknown_speed+=int(not np.isfinite(speed))
   # Unknown lane class / marking style stays UNDEFINED / ROAD_LINE_UNKNOWN.
   idx=add(q,0,speed*2.2369362921 if np.isfinite(speed) else 0.,[(q,12,2)]+[(b,3,side) for side,b in bounds],str(uid))
   relations.append((idx,edges))
  else:
   from shapely.geometry import Polygon,LineString
   from shapely.geometry.polygon import orient
   poly=orient(Polygon(q).buffer(0),sign=1.)
   if poly.geom_type!='Polygon' or poly.is_empty:continue
   box=np.asarray(poly.minimum_rotated_rectangle.exterior.coords)
   if np.linalg.norm(box[0]-box[1])>np.linalg.norm(box[1]-box[2]):a,b=(box[0]+box[3])/2,(box[1]+box[2])/2
   else:a,b=(box[0]+box[1])/2,(box[2]+box[3])/2
   seg=poly.intersection(LineString([a,b]))
   if seg.geom_type=='LineString':a,b=np.asarray(seg.coords)[[0,-1]]
   center=resample([a,b]);boundary=resample(poly.exterior.coords)
   if len(center)<2:continue
   for c in [center,center[::-1]]:add(c,4,0.,[(boundary,13,2),(c,12,2)])
 assert polys and parts,'No usable map geometry'
 n=len(polys);p={};p['num_nodes']=n;p['position']=torch.tensor(np.array([x[0] for x in polys]),dtype=torch.float32);p['orientation']=torch.tensor([x[1] for x in polys],dtype=torch.float32);p['type']=torch.tensor([x[2] for x in polys],dtype=torch.uint8);p['speed_limit']=torch.tensor([x[3] for x in polys],dtype=torch.float32)
 p['stop_sign_mask']=torch.zeros(n,dtype=torch.bool);p['stop_sign_position']=torch.zeros(n,2)
 p['trafic_signal_mask']=torch.zeros(n,11,dtype=torch.bool);p['traffic_signal_state']=torch.zeros(n,11,dtype=torch.uint8);p['stop_point']=torch.zeros(n,11,2)
 for j,uid in enumerate(tl['lane_ids']):
  if str(uid) not in lookup:continue
  i=lookup[str(uid)];states=np.asarray(tl['states'])[ci-10:ci+1,j];mask=np.asarray(tl['valid'])[ci-10:ci+1,j]
  # WOMD signal enum: unknown=0, stop=4, caution=5, go=6.
  p['trafic_signal_mask'][i]=torch.from_numpy(mask.copy()).bool();p['traffic_signal_state'][i]=torch.tensor([{2:4,3:5,4:6}.get(int(s),0) for s in states],dtype=torch.uint8)
  p['stop_point'][i]=torch.tensor(np.asarray(tl['stop_points'])[j]-origin,dtype=torch.float32)
 pt=dict(num_nodes=sum(len(x[0]) for x in parts),position=torch.tensor(np.concatenate([x[0] for x in parts]),dtype=torch.float32),orientation=torch.tensor(np.concatenate([x[1] for x in parts]),dtype=torch.float32),magnitude=torch.tensor(np.concatenate([x[2] for x in parts]),dtype=torch.float32),type=torch.tensor(np.concatenate([np.full(len(x[0]),x[3]) for x in parts]),dtype=torch.uint8),side=torch.tensor(np.concatenate([np.full(len(x[0]),x[4]) for x in parts]),dtype=torch.uint8))
 edges=[(lookup[uid],i,t) for i,rr in relations for uid,t in rr if uid in lookup]
 pe=torch.tensor([[a,b] for a,b,t in edges],dtype=torch.long).T if edges else torch.empty(2,0,dtype=torch.long)
 return {'map_polygon':p,'map_point':pt,('map_point','to','map_polygon'):{'edge_index':torch.stack([torch.arange(pt['num_nodes']),torch.tensor(np.concatenate([np.full(len(x[0]),x[5]) for x in parts]),dtype=torch.long)])},('map_polygon','to','map_polygon'):{'edge_index':pe,'type':torch.tensor([t for _,_,t in edges],dtype=torch.uint8)}},unknown_speed
