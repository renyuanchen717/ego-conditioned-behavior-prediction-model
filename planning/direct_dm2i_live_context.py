"""Live history adapter. No scenario or future ground truth is accessible here."""
import bootstrap
import numpy as np
import torch
from torch_geometric.data import HeteroData
from nuplan.common.maps.maps_datatypes import SemanticMapLayer as L
from ndp_adapter.nuplan_converter import map_graph, APIS

def context(inp, api, signal_history=()):
    history=inp.history; egos=list(history.ego_states); obs=list(history.observations)
    ts=np.array([e.time_point.time_s for e in egos]); desired=ts[-1]+np.arange(-10,1)*.1
    assert ts[0]<=desired[0]+.06, 'Insufficient native observation history'
    ix=np.array([np.argmin(abs(ts-t)) for t in desired]); assert np.max(abs(ts[ix]-desired))<.06
    ego=egos[-1]; origin=np.array([ego.rear_axle.x,ego.rear_axle.y])
    types={'VEHICLE':1,'PEDESTRIAN':2,'BICYCLE':3}
    ids=['ego']+sorted({o.track_token for j in ix for o in obs[j].tracked_objects.tracked_objects if o.tracked_object_type.name in types})
    lookup={s:i for i,s in enumerate(ids)}; n=len(ids)
    pos=np.zeros((n,91,2),np.float32);vel=pos.copy();head=np.zeros((n,91),np.float32);hv=np.zeros((n,11),bool);size=np.ones((n,3),np.float32);typ=np.ones(n,np.uint8)
    for t,j in enumerate(ix):
        e=egos[j];pos[0,t]=[e.center.x-origin[0],e.center.y-origin[1]];head[0,t]=e.center.heading
        v=e.dynamic_car_state.center_velocity_2d;vel[0,t]=[v.x,v.y];hv[0,t]=True
        box=e.car_footprint.oriented_box;size[0]=[box.length,box.width,box.height]
        for o in obs[j].tracked_objects.tracked_objects:
            if o.track_token not in lookup:continue
            k=lookup[o.track_token];pos[k,t]=[o.center.x-origin[0],o.center.y-origin[1]];head[k,t]=o.center.heading;vel[k,t]=[o.velocity.x,o.velocity.y];hv[k,t]=True
            size[k]=[o.box.length,o.box.width,o.box.height];typ[k]=types[o.tracked_object_type.name]

    valid=np.zeros((n,91),bool);valid[:,1:11]=hv[:,1:]&hv[:,:-1];pm=np.zeros((n,91),bool);pm[:,11:]=hv[:,-1,None]
    a=dict(observed_history_mask=torch.from_numpy(hv.copy()),num_nodes=n,av_index=0,current_observed=torch.from_numpy(hv[:,-1].copy()),position=torch.from_numpy(pos),velocity=torch.from_numpy(vel),heading=torch.from_numpy(head),valid_mask=torch.from_numpy(valid),predict_mask=torch.from_numpy(pm),type=torch.from_numpy(typ),target_mask=torch.zeros(n,dtype=torch.bool),interact_mask=torch.zeros(n,dtype=torch.bool))
    for j,key in enumerate(['length','width','height']):a[key]=torch.tensor(np.repeat(size[:,j,None],11,axis=1))
    APIS[api.map_name]=api
    near=api.get_proximal_map_objects(ego.center.point,120,[L.LANE,L.LANE_CONNECTOR,L.CROSSWALK])
    points=[];ranges=[];uids=[];kinds=[];speeds=[]
    for layer,kind in [(L.LANE,0),(L.LANE_CONNECTOR,1),(L.CROSSWALK,3)]:
        for o in near[layer]:
            q=np.asarray(o.polygon.exterior.coords) if kind==3 else np.array([[p.x,p.y] for p in o.baseline_path.discrete_path])
            if len(q)<2:continue
            start=len(points);points.extend(q[:,:2]);ranges.append([start,len(points)]);uids.append(str(o.id));kinds.append(kind);speeds.append(getattr(o,'speed_limit_mps',None) or np.nan)
    signals=list(inp.traffic_light_data);ls=[];states=[];stops=[]
    mapping={'RED':2,'YELLOW':3,'GREEN':4,'UNKNOWN':0}
    for s in signals:
        uid=str(s.lane_connector_id)
        if uid not in uids:continue
        ls.append(uid);states.append(mapping[s.status.name]);stops.append(points[ranges[uids.index(uid)][0]])
    signal=np.zeros((11,len(ls)),int);sm=np.zeros_like(signal,bool)
    signal[-1]=states;sm[-1]=True # Unknown past signals remain masked; no future signal is used.
    for observed_time, observed_signals in signal_history:
        ti=int(np.argmin(abs(desired-observed_time)))
        if abs(desired[ti]-observed_time)>.06:continue
        for status in observed_signals:
            uid=str(status.lane_connector_id)
            if uid in ls:signal[ti,ls.index(uid)]=mapping[status.status.name];sm[ti,ls.index(uid)]=True
    raw=dict(map_name=api.map_name,current_time_index=10,map=dict(ids=uids,kind=kinds,points=np.asarray(points),point_range=ranges,speed_limit_mps=speeds),traffic_lights=dict(lane_ids=ls,states=signal,valid=sm,stop_points=np.asarray(stops)))
    maps,_=map_graph(raw,origin);data=HeteroData({'agent':a,**maps})
    for key in ['agent','map_point','map_polygon']:data[key].ptr=torch.tensor([0,data[key].num_nodes])
    assert not data['agent'].position[:,11:].any() and not data['agent'].valid_mask[:,11:].any()
    return dict(data=data,metadata=dict(ordered_track_ids=ids,ego_track_id='ego',frame='world_axes_ego_rear_axle_origin')),origin
