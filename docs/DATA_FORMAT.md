# Prepared model cache
Data are not uploaded. Prepare an authorized local cache with train.jsonl and val.jsonl, each row containing `path`, `scene_id`, `split_group_id`, `backbone_hash`. Paths refer to local `.pt` samples loaded with `weights_only=True`.
A = all retained current context agents; M = up to 256 map tokens. All times use 10 Hz. History has 11 observed states; future has 80 steps. Preserve original prediction targets and validity masks; do not pad missing history as valid.

| Input | Shape per scene | Meaning |
|---|---|---|
| native | A,6,80,2 | Frozen NDPNet means in each target-local frame |
| native_prob | A,6 | Frozen native probabilities |
| agent_features | A,128 | Frozen current agent embeddings |
| map_features | M,128 | Frozen map embeddings |
| origin, velocity | A,2 | Scene-frame position / velocity |
| yaw, types | A | Local-frame heading / agent type |
| history_mask | A,11 | Observed history mask |
| ego_plan | 80,2 | Supplied conditioning future in scene axes |
| map_position, map_yaw | M,2 and M | Map geometry in scene axes |
| gt, future_valid | A,80,2 and A,80 | Training labels in target-local coordinates |

Other required fields are defined in `interaction_k12.model.INPUT_KEYS`; label fields never enter model forward. See `examples/direct_d_demo.py` for every input. `current_valid` marks the target state at t=0. Scene/root IDs, not augmented sample IDs, determine leakage checks. Non-SDC influencer augmentation changes conditioning ego while preserving scene split.
`build_scene_cache.py` supports factual native-SDC records; `convert_m2i_memory.py` supports prepared influencer contexts. These converters require licensed source records and matching frozen-feature inputs; this release does not claim to reconstruct the private curated index from WOMD alone.
