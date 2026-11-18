"""Exact per-timestep PDM reuse and batching; no approximation or pruning."""
import contextlib, inspect, textwrap, time
from collections import OrderedDict, Counter
import numpy as np
import src.planners.pdm_planner.scoring.pdm_scorer as sm
from src.planners.pdm_planner.scoring.pdm_scorer import PDMScorer
_active=False
_cache={}
stats=Counter()

def array_key(a):
    a=np.asarray(a)
    return (a.shape,a.dtype.str,a.tobytes())

@contextlib.contextmanager
def reuse_scope():
    global _active
    _cache.clear();stats.clear();_active=True
    try:yield
    finally:_active=False;_cache.clear()

# Retain upstream collision/TTC logic verbatim; cache only TTC projections.
_original_ttc=PDMScorer._calculate_ttc
_source=textwrap.dedent(inspect.getsource(_original_ttc))
_start=_source.index('    # calculate TTC for 1s')
_end=_source.index('    # check collision for each proposal and projection')
_geometry_source=_source[_start:_end]
_geometry_ns=dict(sm.__dict__)
exec('def _geometry(self):\n'+_geometry_source+'    return future_time_idcs, speeds, polygons\n',_geometry_ns)
_make_geometry=_geometry_ns['_geometry']
def _ttc_geometry(self):
    key=('ttc',array_key(self._states),self._proposal_sampling.interval_length)
    if key not in _cache:
        _cache[key]=_make_geometry(self);stats['ttc_geometry_builds']+=1
    else:stats['ttc_geometry_hits']+=1
    return _cache[key]
_ns=dict(sm.__dict__);_ns['_ttc_geometry']=_ttc_geometry
exec(_source[:_start]+'    future_time_idcs, speeds, polygons = _ttc_geometry(self)\n'+_source[_end:],_ns)
_cached_ttc=_ns['_calculate_ttc']
def _ttc(self):
    return _cached_ttc(self) if _active else _original_ttc(self)
PDMScorer._calculate_ttc=_ttc

_original_coords=sm.state_array_to_coords_array
_original_polys=sm.coords_array_to_polygon_array
def _coords(states,params):
    if not _active:return _original_coords(states,params)
    key=('ego_coords',array_key(states),repr(params))
    if key not in _cache:_cache[key]=_original_coords(states,params)
    return _cache[key]
def _polys(coords):
    if not _active:return _original_polys(coords)
    key=('ego_polys',array_key(coords))
    if key not in _cache:_cache[key]=_original_polys(coords)
    return _cache[key]
sm.state_array_to_coords_array=_coords
sm.coords_array_to_polygon_array=_polys

# Extend background boxes once per step, and avoid expensive test-library checks
# inside the production inner loop. Shape checks remain.
import horizon_extension as h8
_original_extend=h8.extend_boxes
def _extend(coords):
    if not _active:return _original_extend(coords)
    assert coords.shape==(5,81,2),coords.shape
    key=('extension',id(coords))
    if key in _cache:
        saved,result=_cache[key];assert saved is coords
        stats['extension_hits']+=1;return result
    delta=coords[4,-1]-coords[4,-2]
    result=np.concatenate([coords,coords[:,-1:,:]+np.arange(1,11)[None,:,None]*delta],axis=1)
    assert result.shape==(5,91,2)
    _cache[key]=(coords,result);stats['extension_builds']+=1
    return result
h8.extend_boxes=_extend

def score_pool(planner, simulated, ego, observation, lights, predictions, target_ids, observation_cls, optimized):
    B,K=len(predictions),len(predictions[0]);groups=OrderedDict()
    for b in range(B):
        for k in range(K):
            pred=predictions[b][k]
            key=(None if pred is None else array_key(pred)) if optimized else (b,k)
            groups.setdefault(key,[]).append((b,k))
    multi=[None]*(B*K);weighted=[None]*(B*K);progress=np.zeros(B*K)
    collision=np.zeros(B*K);ttc=np.zeros(B*K)
    route=planner.route
    for entries in groups.values():
        b,k=entries[0]
        occ=observation_cls(planner._future_trajectory_sampling,planner.sampling,150)
        occ.update(ego,observation,lights,route._route_lane_dict,predictions[b][k],np.asarray(target_ids))
        stats['occupancy_builds']+=int(optimized)
        ids=[i for i,j in entries]
        states=simulated[b:b+1] if len(entries)==1 else simulated[ids]
        planner.scorer.score_proposals(states,ego,occ,route._centerline,route._route_lane_dict,route._drivable_area_map,planner._map_api)
        for n,(i,j) in enumerate(entries):
            pos=i*K+j
            multi[pos]=planner.scorer._multi_metrics[:,n].copy()
            weighted[pos]=planner.scorer._weighted_metrics[:,n].copy()
            progress[pos]=planner.scorer._progress_raw[n]
            collision[pos]=planner.scorer._collision_time_idcs[n]
            ttc[pos]=planner.scorer._ttc_time_idcs[n]
    return np.stack(multi,1),np.stack(weighted,1),progress,collision,ttc

def evaluate(planner,simulated,ego,observation,lights,predictions,target_ids,observation_cls,verify=False):
    reference=None;reference_s=None
    if verify:
        t=time.perf_counter()
        reference=score_pool(planner,simulated,ego,observation,lights,predictions,target_ids,observation_cls,False)
        reference_s=time.perf_counter()-t
    t=time.perf_counter()
    with reuse_scope():
        result=score_pool(planner,simulated,ego,observation,lights,predictions,target_ids,observation_cls,True)
        counts=dict(stats)
    elapsed=time.perf_counter()-t
    if verify:
        # Local group progress is transient; the planner normalizes over all 192.
        all_scores=[]
        for values in (reference,result):
            planner.scorer._multi_metrics=values[0]
            planner.scorer._weighted_metrics=values[1]
            planner.scorer._progress_raw=values[2]
            all_scores.append(planner.scorer.aggregate_scores().copy())
        for a,b in zip(reference,result):np.testing.assert_allclose(a,b,rtol=0,atol=1e-12)
        np.testing.assert_allclose(*all_scores,rtol=0,atol=1e-12)
        assert all_scores[0].reshape(len(predictions),-1).min(1).argmax()==all_scores[1].reshape(len(predictions),-1).min(1).argmax()
    planner._pool_reuse_report=dict(iteration=planner.iteration,optimized_s=elapsed,reference_s=reference_s,verified=verify,counts=counts)
    return result[:3]
