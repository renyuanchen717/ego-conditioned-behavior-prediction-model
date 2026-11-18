import bootstrap
import dataclasses,json,time,hashlib,os
import numpy as np
import torch
from flow_planner.planner import FlowPlanner
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from ndp_mode_adapter.frozen_features import FrozenFeatures
from src.planners.pdm_planner.abstract_pdm_planner import AbstractPDMPlanner
from src.planners.pdm_planner.observation.pdm_observation_pred import PDMObservationPred
from src.planners.pdm_planner.observation.pdm_observation_utils import get_drivable_area_map
from src.planners.pdm_planner.simulation.pdm_simulator import PDMSimulator
from src.planners.pdm_planner.scoring.pdm_scorer import PDMScorer
from src.planners.pdm_planner.utils.pdm_array_representation import ego_states_to_state_array
from src.planners.pdm_planner.utils.pdm_path import PDMPath

class RouteTools(AbstractPDMPlanner):
    def name(self):return 'RouteTools'
    def observation_type(self):return None
    def initialize(self,x):pass
    def compute_planner_trajectory(self,x):raise NotImplementedError

class FlowC0Planner(FlowPlanner):
    requires_scenario=False
    def __init__(self, variant='flow_c0_pdm', candidates=16, **kwargs):
        super().__init__(**kwargs);self.variant=variant;self.candidates=candidates
        self.route=RouteTools(150);self.sampling=TrajectorySampling(num_poses=40,interval_length=.1)
        self.simulator=PDMSimulator(self.sampling);self.scorer=PDMScorer(self.sampling)
        self.frozen=None;self.adapter=None
    def name(self):return self.variant
    def compute_planner_trajectory(self,inp):
        raise NotImplementedError
