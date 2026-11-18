"""Batch candidate-independent PDM metrics; retain per-candidate collision and TTC."""
import numpy as np
from planner import FlowC0Planner
from src.planners.pdm_planner.scoring.pdm_scorer import PDMScorer
_original=PDMScorer.score_proposals
_bundle=None

def score(self,states,initial_ego_state,observation,centerline,route_lane_dict,drivable_area_map,map_api):
    global _bundle
    base=states.base
    if not isinstance(base,np.ndarray) or base.ndim!=3 or states.shape[0]!=1 or base.shape[1:]!=states.shape[1:]:
        return _original(self,states,initial_ego_state,observation,centerline,route_lane_dict,drivable_area_map,map_api)
    offset=states.ctypes.data-base.ctypes.data
    if offset%base.strides[0]:return _original(self,states,initial_ego_state,observation,centerline,route_lane_dict,drivable_area_map,map_api)
    index=offset//base.strides[0]
    if _bundle is None or _bundle[0] is not base:
        batch=PDMScorer(self._proposal_sampling)
        batch._reset(base,initial_ego_state,None,centerline,route_lane_dict,drivable_area_map,map_api)
        batch._calculate_ego_area();batch._calculate_driving_direction_compliance();batch._calculate_drivable_area_compliance();batch._calculate_progress();batch._calculate_is_comfortable()
        _bundle=(base,batch)
    batch=_bundle[1]
    self._reset(states,initial_ego_state,observation,centerline,route_lane_dict,drivable_area_map,map_api)
    self._ego_areas=batch._ego_areas[index:index+1].copy()
    self._multi_metrics=batch._multi_metrics[:,index:index+1].copy()
    self._weighted_metrics=batch._weighted_metrics[:,index:index+1].copy()
    self._progress_raw=batch._progress_raw[index:index+1].copy()
    self._calculate_no_at_fault_collision();self._calculate_ttc()
    return self.aggregate_scores()
PDMScorer.score_proposals=score
_step=FlowC0Planner.compute_planner_trajectory
def step(self,inp):
    global _bundle
    _bundle=None
    try:return _step(self,inp)
    finally:_bundle=None
FlowC0Planner.compute_planner_trajectory=step
