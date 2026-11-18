# D_m2i: Ego-conditioned Behavior Prediction and Flow Planner Integration

This repository now contains **D_m2i**, its optional probability heads, and the **Flow Planner + D_m2i + PDM** planning pipeline. Earlier Recorded K=12 and original Direct D release files have been replaced on the default branch.

**Non-commercial research use:** This model/software was made using the Waymo Open Dataset, provided by Waymo LLC under the [Waymo Dataset License Agreement for Non-Commercial Use](https://waymo.com/open/terms/); access and use are governed by those terms. Distributed model weights and derivatives remain subject to that agreement. A copy is in `licenses/WAYMO_DATASET_LICENSE.txt`.

## Quick start
```bash
python -m pip install -e .
python examples/direct_d_demo.py
python -m unittest discover -s tests -v
```
```python
from interaction_k12 import load_direct_d
model = load_direct_d('checkpoints/d_m2i', device='cuda')
# outputs = model(inputs): new [B,A,6,80,2], pool [B,A,12,80,2]
```
The API name `load_direct_d` is retained for compatibility, but this folder contains **D_m2i epoch 26**, trained on influencer-augmented WOMD InterHub data. No NDPNet checkpoint is embedded. The synthetic example checks correctness, not accuracy.

## Included
- Full proposal model source, tensor-verified inference checkpoint, training and evaluation scripts.
- D_m2i-P and D_m2i-P2 optional probability-head weights, clearly separated.
- Online history/map adapter, frozen NDPNet wrapper, Flow Planner integration and 8s PDM scoring.
- Dataset schema, dependency provenance, license notices and aggregate evaluation results.

**Training/validation data and caches are intentionally not uploaded, at the owner's request.** Training used 61,975 samples and validation 7,776, with scene-disjoint splits; 233 irregular-timestamp training samples were excluded. See [data scope](data/README.md).

## Final InterPlan results
Same 335 scenarios, scores multiplied by 100:

| Pipeline | Overall score |
|---|---:|
| Flow Planner baseline | 59.70 |
| Flow Planner + D_m2i, worst-of-12, 8s | 60.56 |
| Flow Planner + D_m2i-P, Top-1, 8s | 60.85 |
| Flow Planner + D_m2i-P2, worst-of-two Top-1, 8s | 60.67 |

These are local evaluations with the documented protocol, not the upstream paper's score. `results/` contains the underlying aggregate results. WOMD numbers are recorded-ego-future conditional diagnostics on validation, not official hidden-test or joint-interactive challenge submissions.

Read [architecture](docs/MODEL.md), [data contract](docs/DATA_FORMAT.md), [training](docs/TRAINING.md), [planning setup](docs/PLANNING.md), and [third-party notices](THIRD_PARTY_NOTICES.md).
