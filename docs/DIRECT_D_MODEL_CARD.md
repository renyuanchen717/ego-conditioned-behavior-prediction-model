# Direct D: frozen NDPNet + interaction-conditioned direct proposals

Direct D is **not** the legacy Recorded K=12 model. It retains six native NDPNet modes and adds six directly decoded trajectories at 10 Hz. It uses a supplied full 8-second ego plan, without a CV reference, residual trajectory addition, impact router, or learned probabilities for the new proposals.

## Released checkpoint

- File: `checkpoints/direct_d/model.pt`; SHA-256 is in the adjacent `manifest.json`.
- Variant D, seed 0, selected epoch 25 (68,150 optimizer steps). Training completed 30 epochs (81,780 steps).
- New modules trained from scratch. NDPNet was frozen, using cached backbone features.
- 10,903 eligible InterHub-derived native WOMD training scenes; 1,944 separate validation scenes. From 11,136 training and 2,000 validation source scenes, 289 failed preprocessing eligibility checks. The original native test split was not used for training or checkpoint selection.
- Batch size 4 scenes; AdamW, learning rate 0.0002, weight decay 0.01, gradient clip 2; hidden dimension 128, four attention heads, two post-conditioning agent-interaction layers, dropout 0.1.
- Objective: `L_pool + L_new + L_displacement`. No probability/NLL head or endpoint loss. See [equations and architecture](INTERACTION_K12.md).
- Checkpoint selection: minimum validation **scene/group-balanced 8s pool minADE@12**. Metrics below use target-micro or class-macro aggregation as explicitly stated, not the selection aggregation.
- The inference export omits optimizer state and machine-specific paths. Every released model tensor was checked for exact equality with the selected training checkpoint.

## Inference

```bash
python -m pip install -e .
python examples/direct_d_demo.py
python -m unittest discover -s tests -v
```

```python
import torch
from interaction_k12 import load_direct_d

model = load_direct_d('checkpoints/direct_d', device='cpu')
with torch.inference_mode():
    output = model(inputs)
new = output['new']    # [B, A, 6, 80, 2]
pool = output['pool']  # [B, A, 12, 80, 2]
assert torch.equal(pool[:, :, :6], inputs['native'])
```

The synthetic example verifies execution and preservation, not prediction accuracy. `inputs` must contain exactly `interaction_k12.model.INPUT_KEYS`; target future GT must never enter inference.

### Input contract

`B` denotes scenes/conditioning groups, `A` padded agents, `M` map polygons. Timestamps are 0.1, 0.2, ..., 8.0 seconds after the current frame; 11 observed frames cover 1 second including the current frame.

| Input | Shape | Meaning |
|---|---|---|
| `native` | B,A,6,80,2 | Frozen predictions in each target's local coordinates |
| `native_prob` | B,A,6 | Native probabilities, normalized within the original six modes |
| `native_valid` | B,A | Native prediction availability; false for ego/padding |
| `agent_features` | B,A,128 | Frozen NDPNet current agent embeddings Ea |
| `types` | B,A | 0 unknown, 1 vehicle, 2 pedestrian, 3 cyclist |
| `origin`, `velocity` | B,A,2 | Current position/velocity in common scene axes |
| `yaw` | B,A | Current agent heading in radians |
| `history_mask` | B,A,11 | True observed-frame validity, including current |
| `ego_role` | B,A | Conditioning agent identity |
| `agent_present_mask`, `context_mask`, `target_mask` | B,A | Presence, context, prediction target masks |
| `map_features` | B,M,128 | Frozen NDPNet polygon embeddings Em |
| `map_position`, `map_yaw` | B,M,2 and B,M | Map position/orientation in common scene coordinates |
| `map_mask` | B,M | Valid map tokens |
| `nearest_map` | B,A | Nearest polygon ID/index in original polygon array; -1 if none |
| `ego_plan` | B,80,2 | Supplied ego box-center future in common scene coordinates |
| `ego_plan_valid_mask` | B,80 | Must be valid throughout the supplied plan |
| `ego_current` | B,2 | Current ego box-center position in the same scene frame |

Future outputs are target-local xy positions, in meters. Raw ego plans must not be transformed into target-local coordinates before passing them to this API: the model performs that transformation. Preserve all currently observed context agents; cap map tokens to the nearest 256 polygons, as during training. The original target masks and future-valid masks remain distinct from context masks.

The original backbone checkpoint is **not bundled**. Obtain NDPNet from [its official repository](https://github.com/HuaHu-yizhou/NDPNet), subject to upstream terms. The expected checkpoint hash is in the release manifest. Generic NDPNet features or differently ordered polygon embeddings are not interchangeable with the documented extractor. See [interface audit](INTERACTION_INTERFACE_AUDIT.md) for Ea/Em extraction details.

## Model workflow

1. Encode native modes/probabilities, agent state and frozen scene/map features.
2. Encode the complete target-local ego plan and ego-versus-native-mode spatiotemporal geometry; use cross-attention to condition agent embeddings.
3. Run two agent-agent attention layers with relative state/heading/map-relation edge features.
4. Six learned queries read conditioned embeddings, ego tokens, native tokens and map tokens, then directly decode six 80-step trajectories.
5. Concatenate unchanged native modes and the six new proposals. No ranking/compression is performed.

The geometry branch currently probability-pools native modes, and the direct decoder emits the full trajectory through an MLP. The six new proposals have **no scores, probabilities or scales**. There is no joint multi-agent mode alignment. Returning all original modes preserves their oracle coverage, not top-1 accuracy or planning safety.

## Validation (selected checkpoint)

1,944 scenes, target-micro 8s metrics, meters. ADE and FDE use their respective validity masks; horizon endpoint validity is required for FDE.

| Output | minADE | minFDE |
|---|---:|---:|
| Frozen NDPNet, K=6 | 0.254414 | 0.811456 |
| D new proposals only, K=6 | 0.322462 | 0.909191 |
| D pool, K=12 | 0.213963 | 0.532412 |

Full class/diagnostic metrics: [validation.json](../results/direct_d/validation.json).

## WOMD interaction prediction diagnostic

Frozen checkpoints evaluated on the existing fixed pair-conditioned WOMD validation_interactive protocol: one official pair member supplies its recorded future; only the other member is scored. The SDC is the conditioner when it belongs to the pair; otherwise the existing seeded-random role assignment is retained. This does not establish that the predicted target is causally reactive. The full 8s conditioning plan is supplied even for 3s/5s metrics.

Of 41,595 source pairs, 1,254 overlap with existing source train/validation/test manifests and are excluded. Of the remaining 40,341, 592 have irregular timestamps and are jointly excluded. Final evaluation: **39,749 pairs**, all 150 shards completed. Predictions are sampled from 10 Hz to 2 Hz (0.5,...,8s). Reported values are masked marginal oracle errors, not official joint challenge scores or hidden-test results. Original frozen predictions were reused on exactly these matched pairs, with online native-output parity checks.

**Class-macro, meters:**

| Output | 3s minADE | 3s minFDE | 5s minADE | 5s minFDE | 8s minADE | 8s minFDE |
|---|---:|---:|---:|---:|---:|---:|
| Frozen NDPNet, K=6 | 0.240452 | 0.443451 | 0.482925 | 0.951218 | 0.946053 | 1.933524 |
| B new-only, K=6 | 0.313811 | 0.534658 | 0.623256 | 1.248234 | 1.282157 | 2.848141 |
| D new-only, K=6 | 0.329089 | 0.559853 | 0.639486 | 1.287394 | 1.303110 | 2.909236 |
| B pool, K=12 | 0.216564 | 0.369272 | 0.438877 | 0.807860 | 0.868718 | 1.663275 |
| D pool, K=12 | 0.220952 | 0.378585 | 0.442893 | 0.813497 | 0.871544 | 1.678286 |

**Vehicle-only, 8s:** D new-only minADE/minFDE@6 = 1.494778 / 3.226788; D pool minADE/minFDE@12 = 1.022201 / 1.868084. Corresponding B values are 1.527237 / 3.365113 and 1.021532 / 1.875066. D improves vehicle new-only accuracy but does not outperform B overall in class-macro evaluation. K=12 vs K=6 gains alone cannot establish the benefit of ego conditioning.

Full aggregates, denominators and all classes: [womd_interaction.json](../results/direct_d/womd_interaction.json). No per-scene data are distributed.

## Limitations and audit correction

- Single seed. No claim of statistically robust superiority or calibrated probabilities.
- Trained on factual native WOMD, not matched counterfactual responses; recorded-plan accuracy alone does not validate reactions to unobserved ego interventions.
- New proposals have duplication and temporal-jitter limitations. They are not safety-certified.
- A recorded-plan input-gradient check on 64 validation scenes found nonzero dependencies in C/D and no dependency in history-only B; this checks connectivity, not correct reaction behavior.
- The earlier `diagnose_interaction_plan.py` missed `load_state_dict`. This release fixes it. Old sensitivity outputs produced by that script do not represent trained checkpoints. Training and standard validation used the actual trained weights and are unaffected.
- InterPlan planning evaluation is ongoing separately. No completed 335-scenario planning claim is included in this release.