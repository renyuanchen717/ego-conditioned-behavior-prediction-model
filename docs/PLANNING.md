# Flow Planner + D_m2i + PDM
For each planning step: produce 16 Flow Planner candidates (CFG=1.8); extract frozen NDPNet features once from observed history; condition D_m2i on each candidate; convert forecasts to world coordinates; simulate ego tracking; score using PDM over 8s; execute the winning candidate with closed-loop replanning. TTC occupancy is extended to 9s using terminal velocity. Maximum 10 nearest current vehicle targets are scored; all currently observed eligible agents remain prediction context.

Scoring variants (`DM2I_SCORING`):
- `worst12`: D_m2i, `argmax_i min_k PDM(E_i,Y_ik)`. Same raw mode index across agents, **not** calibrated joint modes or all combinations.
- `top1`: D_m2i-P, per-agent maximum probability across 12 modes; one scene hypothesis per ego candidate.
- `worst2`: D_m2i-P2, two scene hypotheses (all native Top-1 vs all new-proposal Top-1); choose max over ego candidates of min of the two scores. Not per-agent adversarial combinations.

## Dependencies
Linux/CUDA is the tested environment. Install upstream projects and their requirements:
- Flow Planner https://github.com/DiffusionAD/Flow-Planner commit `366df7654d7b7fa77f9d2577279da63127c6d2f8`
- NDPNet https://github.com/HuaHu-yizhou/NDPNet (official checkpoint)
- InterPlan https://github.com/mh0797/interPlan.git commit `a85024a48935c92ec0d7c2a8cc2e583d01514469`
- nuPlan devkit, PyG with matching CUDA extensions, flow_matching, Hydra, Shapely 2 and nuPlan map assets.
The PDM subtree used in these experiments is vendored from the Apache-2.0 BeTop checkout; retain its license. Extra unused PDM planners may require tuplan_garage. NDPNet/Flow Planner weights and native datasets are external.

```bash
export FLOW_PLANNER_ROOT=/path/to/Flow-Planner
export NDPNET_ROOT=/path/to/NDPNet
export INTERPLAN_ROOT=/path/to/interPlan
export NDPNET_CHECKPOINT=/path/to/NDPNet.ckpt
export FLOW_CHECKPOINT=/path/to/flow/model.pth
export NUPLAN_DATA_ROOT=/path/to/nuplan
export NUPLAN_MAPS_ROOT=/path/to/maps
export DM2I_OUTPUT=$PWD/outputs/interplan/worst12
export DM2I_SCORING=worst12
CUDA_VISIBLE_DEVICES=0 python planning/run_interplan.py \
 +simulation=default_interplan_benchmark planner=flow_c0 \
 scenario_filter=benchmark_scenarios worker=sequential \
 scenario_builder.max_workers=1 output_dir=$DM2I_OUTPUT seed=3407 \
 'hydra.searchpath=[pkg://interplan.planning.script.config.common,pkg://interplan.planning.script.config.simulation,pkg://interplan.planning.script.experiments,pkg://nuplan.planning.script.config.common,pkg://nuplan.planning.script.config.simulation,pkg://nuplan.planning.script.experiments]'
```
Use official InterPlan DB/filter configuration to resolve 335 augmented scenarios. A local single-scenario smoke command must additionally specify the matching DB and augmented token. Each concurrent worker needs a distinct `DM2I_OUTPUT` and `output_dir`; split scenario lists without duplicates across GPUs. This repository does not distribute DBs, map data or the training index.

Results are 335-scenario official InterPlan aggregator scores. Candidate noise is timestamp-seeded; the single-sample Flow baseline does not use the same candidate-noise draw. Results are matched by scenarios, not a perfectly noise-matched ablation. Native WOMD-conditioned predictors are transferred to nuPlan without claiming domain equivalence.
