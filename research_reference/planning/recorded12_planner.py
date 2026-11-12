"""Flow candidate selection with the locked twelve-mode conditional predictor."""
from planner import *
from recorded12_forecast import PlanForecast
class FlowRecorded12Planner(FlowC0Planner):
    def initialize(self, init):
        FlowPlanner.initialize(self,init)
        self.iteration=0
        self.signal_history=[]
        self.route._map_api=init.map_api
        self.route._load_route_dicts(init.route_roadblock_ids)
        if self.frozen is None:
            self.frozen=FrozenFeatures('/path/to/experiment-root/NDPNet_evaluation/checkpoints/NDPNet.ckpt','cuda')
            self.conditioner=PlanForecast()
        self.audit_dir=bootstrap.ROOT/'recorded12'/'candidate_audits'
        self.audit_dir.mkdir(parents=True,exist_ok=True)
    @torch.inference_mode()
    def compute_planner_trajectory(self,inp):
        import occupancy_cache, batch_pdm
        occupancy_cache._cache.clear();batch_pdm._bundle=None
        started=time.perf_counter();ego,observation=inp.history.current_state
        # Common noise pool for both planners, reproducible irrespective of worker order.
        seed=int(hashlib.sha256(str(ego.time_point.time_us).encode()).hexdigest()[:8],16)
        torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        data=self.planner_input_to_model_inputs(inp)
        for f in dataclasses.fields(data):
            v=getattr(data,f.name)
            if torch.is_tensor(v) and v.numel():setattr(data,f.name,v.repeat(self.candidates,*([1]*(v.ndim-1))))
        out=self.core.inference(self._planner,data,use_cfg=self.use_cfg,cfg_weight=self.cfg_weight)
        assert out.shape[0]==self.candidates and torch.isfinite(out).all()
        states=[self.outputs_to_trajectory(out[i:i+1],inp.history.ego_states) for i in range(self.candidates)]
        selected=0; scores=None;target_ids=[]; response_range=None
        if self.variant!='flow_baseline':
            self.signal_history.append((ego.time_point.time_s,list(inp.traffic_light_data)));self.signal_history=self.signal_history[-21:]
            c,origin=context(inp,self._map_api,self.signal_history)
            types=c['data']['agent'].type;present=c['data']['agent'].current_observed
            have_targets=bool(((types==1)&present)[1:].any())
            predictions=[[None]*12 for _ in range(self.candidates)]
            if have_targets:
                f=self.frozen.extract(c,max_targets=10,target_types=(1,))
                target_ids=f['metadata']['track_ids'];n=len(target_ids)
                plans=[]
                for ss in states:
                    # C0 trained with box-center ego futures, global axes relative to current rear axle.
                    future=ss[-80:]
                    plans.append(torch.tensor([[s.center.x-origin[0],s.center.y-origin[1],s.center.heading] for s in future],dtype=torch.float32))
                mu=self.conditioner(f,torch.stack(plans),c,list(inp.history.ego_states)).cpu().numpy()
                response_range=float(np.max(np.linalg.norm(mu-mu[:1],axis=-1)))
                yaw=f['yaw'].numpy();co=np.cos(yaw);si=np.sin(yaw);rot=np.stack([co,si,-si,co],-1).reshape(n,2,2)
                world=np.einsum('bnkti,nij->bnktj',mu,rot)+f['origin'].numpy()[None,:,None,None,:]+origin
                objects={o.track_token:o for o in observation.tracked_objects.tracked_objects}
                assert world.shape[2]==12
                current=np.array([[objects[t].center.x,objects[t].center.y] for t in target_ids])
                for b in range(self.candidates):
                    for k in range(12):
                        xy=world[b,:,k];full=np.concatenate([current[:,None],xy],1)
                        delta=np.diff(full,axis=1);heading=np.zeros((n,81));heading[:,0]=[objects[t].center.heading for t in target_ids]
                        for t in range(80):heading[:,t+1]=np.where(np.linalg.norm(delta[:,t],axis=-1)>.05,np.arctan2(delta[:,t,1],delta[:,t,0]),heading[:,t])
                        predictions[b][k]=np.concatenate([full,heading[...,None]],-1)
            route=self.route
            if self.iteration==0:route._route_roadblock_correction(ego)
            route._drivable_area_map=get_drivable_area_map(self._map_api,ego,150)
            lane=route._get_starting_lane(ego);route._centerline=PDMPath(route._get_discrete_centerline(lane))
            arrays=np.stack([ego_states_to_state_array(ss) for ss in states]);assert arrays.shape[1]==81,arrays.shape
            simulated=self.simulator.simulate_proposals(arrays,ego)
            if os.environ.get('FLOW_POOL_REUSE') in ('1','verify'):
                from pdm_pool_reuse import evaluate
                mm,ww,pp=evaluate(self,simulated,ego,observation,list(inp.traffic_light_data),predictions,target_ids,PDMObservationPred,verify=os.environ.get('FLOW_POOL_REUSE')=='verify')
                mult=list(mm.T);weighted=list(ww.T);progress=list(pp)
            else:
                mult=[];weighted=[];progress=[]
                for b in range(self.candidates):
                    for k in range(12):
                        occupancy=PDMObservationPred(self._future_trajectory_sampling,self.sampling,150)
                        occupancy.update(ego,observation,list(inp.traffic_light_data),route._route_lane_dict,predictions[b][k],np.asarray(target_ids))
                        self.scorer.score_proposals(simulated[b:b+1],ego,occupancy,route._centerline,route._route_lane_dict,route._drivable_area_map,self._map_api)
                        mult.append(self.scorer._multi_metrics[:,0].copy());weighted.append(self.scorer._weighted_metrics[:,0].copy());progress.append(self.scorer._progress_raw[0])
            self.scorer._multi_metrics=np.stack(mult,1);self.scorer._weighted_metrics=np.stack(weighted,1);self.scorer._progress_raw=np.asarray(progress)
            mode_scores=self.scorer.aggregate_scores().reshape(self.candidates,12)
            assert np.isfinite(mode_scores).all()
            scores=mode_scores.min(axis=1);selected=int(scores.argmax())
            assert scores[selected]==np.max(np.min(mode_scores,axis=1))
        if self.iteration==0:
            report=dict(variant=self.variant,candidates=self.candidates,selected=selected,scores=None if scores is None else scores.tolist(),target_ids=target_ids,elapsed_s=time.perf_counter()-started,conditional_mode_max_change_m=response_range,seed=seed,history_only=True,ego_conditioning='candidate box-center trajectory',scoring_horizon_s=4,forecast_horizon_s=8)
            report.update(mode_scores=mode_scores.tolist(),worst_mode_indices=mode_scores.argmin(1).tolist(),selection='argmax_i min_k PDM(i,k)',mode_combination='same raw mode index across agents; not joint modes or all combinations',progress_normalization='common across all candidate-mode pairs')
            (self.audit_dir/f'{self.variant}_{os.getpid()}_{ego.time_point.time_us}.json').write_text(json.dumps(report,indent=2))
        if self.iteration%25==0:print(json.dumps(dict(planner=self.variant,iteration=self.iteration,step_s=time.perf_counter()-started,selected=selected)),flush=True)
        occupancy_cache._cache.clear();batch_pdm._bundle=None
        self.iteration+=1
        return InterpolatedTrajectory(states[selected])
