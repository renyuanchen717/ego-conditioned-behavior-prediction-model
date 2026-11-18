"""Eight-second PDM scoring with terminal-velocity occupancy for TTC lookahead."""
from direct_dm2i_planner import FlowDirectDM2IPlanner
from planner import TrajectorySampling,PDMSimulator,PDMScorer
import direct_dm2i_planner as runtime
import numpy as np

def extend_boxes(coords):
    from horizon_extension import extend_boxes as shared_extension
    return shared_extension(coords)

class Observation8(runtime.PDMObservationPred):
    def _get_object_manager(self,*args,**kwargs):
        manager=super()._get_object_manager(*args,**kwargs)
        for kind,items in manager._dynamic_object_dxy.items():
            manager._dynamic_object_dxy[kind]=[extend_boxes(x) for x in items]
        return manager

class FlowDirectDM2IH8Planner(FlowDirectDM2IPlanner):
    def initialize(self,init):
        super().initialize(init)
        self.sampling=TrajectorySampling(num_poses=80,interval_length=.1)
        self.simulator=PDMSimulator(self.sampling)
        self.scorer=PDMScorer(self.sampling)
        self.audit_dir=runtime.bootstrap.OUTPUT/'candidate_audits'
        self.audit_dir.mkdir(parents=True,exist_ok=True)
        runtime.PDMObservationPred=Observation8
    def compute_planner_trajectory(self,inp):
        result=super().compute_planner_trajectory(inp)
        if self.iteration==1:
            import json,os
            file=self.audit_dir/f'{self.variant}_{os.getpid()}_{inp.history.current_state[0].time_point.time_us}.json'
            data=json.loads(file.read_text())
            data.update(scoring_horizon_s=8,occupancy_horizon_s=9,ttc_extension='constant terminal center velocity and heading after 8s')
            assert self.scorer._states.shape[1]==81
            assert self.scorer._observation._observation_samples==90
            data['verified_scorer_state_count']=81
            file.write_text(json.dumps(data,indent=2))
        return result
