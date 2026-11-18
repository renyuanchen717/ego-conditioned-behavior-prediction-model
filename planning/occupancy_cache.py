"""Memoize only candidate-independent PDM object conversion within one planning step."""
import os
import src.planners.pdm_planner.observation.pdm_observation_pred as observation_module
from src.planners.pdm_planner.observation.pdm_object_manager_pred import PDMObjectManager as Original
from planner import FlowC0Planner
_cache={}
class CachedObjectManager(Original):
    def add_object(self, object, object_preds=None):
        if object_preds is not None:return super().add_object(object,object_preds)
        key=id(object)
        if key not in _cache:
            m=Original();m.add_object(object,None);_cache[key]=(object,m)
        saved,m=_cache[key]
        assert saved is object
        self._unique_objects.update(m._unique_objects)
        for kind in self._dynamic_object_tokens:
            self._dynamic_object_tokens[kind].extend(m._dynamic_object_tokens[kind])
            self._dynamic_object_coords[kind].extend(m._dynamic_object_coords[kind])
            self._dynamic_object_dxy[kind].extend(m._dynamic_object_dxy[kind])
        self._static_object_tokens.extend(m._static_object_tokens)
        self._static_object_coords.extend(m._static_object_coords)
original=FlowC0Planner.compute_planner_trajectory

def cached_step(self,inp):
    _cache.clear()
    try:return original(self,inp)
    finally:_cache.clear()
observation_module.PDMObjectManager=CachedObjectManager
FlowC0Planner.compute_planner_trajectory=cached_step
