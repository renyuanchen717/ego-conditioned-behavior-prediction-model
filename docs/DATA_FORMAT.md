# Cache contract

The portable package operates on predictions from a separately installed, frozen NDPNet. It does not convert raw WOMD TFRecords or nuPlan DBs by itself.

Each row represents one prediction target and one conditioning ego in a shared scene. Preserve the target/conditioning identities and current timestamp in your private preprocessing metadata. Do not pass target GT into inference.

| NPZ field | Shape | Meaning |
|---|---|---|
| `n` | N×6×16×2 | Native frozen NDPNet trajectory means at 0.5,...,8s |
| `q` | N×6 | Normalized native probabilities (not logits) |
| `typ` | N | WOMD type: vehicle=1, pedestrian=2, cyclist=3 |
| `cv` | N×16×2 | Ego CV trajectory in the target-local frame |
| `rec` | N×16×2 | Recorded/candidate ego trajectory in the same frame |
| `y` | N×16×2 | Target future GT; training/evaluation only |
| `v` | N×16 | Valid GT mask |
| `sid` | N | Original scene ID as a Unicode array; training split checks |

Use float32 for geometry and probabilities. Repeated scene IDs are expected when a scene supplies multiple targets, but all such rows must stay in the same split. Each agent-local transform uses its current center and yaw. With row-vector coordinates:

```python
R = [[cos(yaw), -sin(yaw)], [sin(yaw), cos(yaw)]]
local_xy = (world_xy - target_current_xy) @ R
```

Transform native predictions, ego plans and GT consistently. Convert heading-aligned NDPNet output only once. At 10 Hz, select future indices 4,9,...,79, corresponding to 0.5,1.0,...,8.0s; do not confuse these with indices of an array containing the current state.

Frozen NDPNet's native configuration uses 1s observed history; the residual API does not add history to it. Full recorded ego future must be valid for the offline conditional protocol. Exclude unreliable conditioning inputs; mask invalid target futures only in supervised losses/metrics. Do not zero-pad missing conditioning future and present it as observed data.

Scene mapping, valid-mask generation, native NDPNet graph construction, target selection and official dataset access remain the responsibility of the external preprocessor. A generic cache evaluator cannot reproduce the reported scenario population unless these rules, exclusions and split manifests also match.
