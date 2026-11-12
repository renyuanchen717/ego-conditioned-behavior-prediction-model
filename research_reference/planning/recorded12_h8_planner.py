"""Eight-second PDM scoring with terminal-velocity occupancy for TTC lookahead."""
from recorded12_planner import FlowRecorded12Planner
from planner import TrajectorySampling,PDMSimulator,PDMScorer
import recorded12_planner as runtime
import numpy as np

def extend_boxes(coords):
    assert coords.shape==(5,81,2),coords.shape
    displacement=coords[4,-1]-coords[4,-2]
    extra=coords[:,-1:, :]+np.arange(1,11)[None,:,None]*displacement
    result=np.concatenate([coords,extra],axis=1)
    assert result.shape==(5,91,2)
    np.testing.assert_array_equal(result[:,:81],coords)
    return result

class Observation8(runtime.PDMObservationPred):
    def _get_object_manager(self,*args,**kwargs):
        manager=super()._get_object_manager(*args,**kwargs)
        for kind,items in manager._dynamic_object_dxy.items():
            manager._dynamic_object_dxy[kind]=[extend_boxes(x) for x in items]
        return manager

class FlowRecorded12H8Planner(FlowRecorded12Planner):
    def initialize(self,init):
        super().initialize(init)
        self.sampling=TrajectorySampling(num_poses=80,interval_length=.1)
        self.simulator=PDMSimulator(self.sampling)
        self.scorer=PDMScorer(self.sampling)
        self.audit_dir=runtime.bootstrap.ROOT/'recorded12_h8'/'candidate_audits'
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
