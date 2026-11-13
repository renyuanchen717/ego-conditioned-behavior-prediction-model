# Interaction-conditioned Direct Proposals (independent version)

This version is independent of Recorded K=12. It loads none of the old anchor/5s/8s weights, uses no ego CV, residual addition, impact router or probability head. Original six NDPNet trajectories are returned unchanged; six learned queries directly decode six new 80-step trajectories. Outputs are conditional marginal pools, not learned joint modes.

See [interface audit](INTERACTION_INTERFACE_AUDIT.md) for the actual local code/data interfaces and backbone commit.

## Inputs and frames

A conditioning group is `(original scene ID, current timestamp, conditioning ego ID)`.
Inputs are batched as B groups and A padded agents. Current observation, context, target, native availability and GT masks are distinct. `native [B,A,6,80,2]` is target-local. Current origins/velocities, map polygon positions, and `ego_plan [B,80,2]` share scene coordinates. Each target's yaw/origin is used exactly once to transform scene inputs into its local frame. Future points are t=0.1,...,8s; current target position is known to be zero locally. Eleven historical frames include the current frame.

`Ea` and `Em` are the actual frozen NDPNet context/map outputs. The extractor adds true IDs, timestamp, masks and common-frame poses from the complete native scene, not from independently cached target rows. All originally predicted targets are retained. Context includes currently observed agents even if their future has no valid GT. No target future selects context. Ego is a context node with its given plan, not an unconditioned native ego future.

## Modules (default hidden dimension 128, 4 heads, dropout 0.1)

1. **FrozenContextAdapter:** project cached agent/map features; embed agent type. Each of the six native modes is encoded by a shared MLP over its complete ordered 80×2 sequence plus probability. Probability-weighted set pooling is invariant to jointly permuting native modes and probabilities. Target-local map-relative position/orientation is fused with projected map tokens and read through attention.
2. **EgoPlanEncoder:** 80 time-indexed tokens encode target-local ego position, finite-difference velocity and normalized time. Current observed ego position anchors the first velocity. Target context queries these tokens through cross-attention.
3. **Geometry:** each native mode contributes per-time synchronous relative xy/distance, soft asynchronous distance and arrival lag, plus time. A shared relation MLP and probability-weighted mode pooling add geometry to plan tokens. No native-mode index is attached to any new query. No CV difference is used.
4. **AgentInteraction:** two post-conditioning self-attention layers act within a group. Receiver-local relative position/velocity, heading, source type, history coverage and a shared-nearest-map-polygon indicator supply edge bias. Nearest-map equality is a coarse static relation, not verified lane topology. Padding and other groups cannot enter attention. Two layers permit conditional information propagation but do not establish causal chain correctness.
5. **DirectDecoder:** six learned query embeddings read conditioned context, plan tokens, the native mode-token set and map tokens. An MLP directly outputs 6×80×2 positions. A configurable 30m output-unit scale improves optimization; it adds no baseline trajectory or kinematic reference. The current point is zero for displacement supervision. No probabilities/scales are predicted for the new pool.

## Objective and masking

For a valid target, compute full-valid-trajectory ADE separately for each new proposal. Detach the winning index k*; use this same index for all displacement and optional endpoint supervision.

$$L=L_{pool}+L_{new}+\lambda_{disp}L_{disp},$$
$$L_{pool}=\operatorname{mean}_g\operatorname{mean}_{i\in g}\min(e_i^X,e_i^P),\quad
L_{new}=\operatorname{mean}_g\operatorname{mean}_{i\in g}e_i^P.$$

Invalid GT is replaced before arithmetic. Displacements are valid only when both adjacent positions are valid; the first displacement uses the known current position and its mask. Agents without adjacent valid pairs are excluded only from that term. Groups without any supervised targets are skipped safely. No NLL, classification, default endpoint loss, collision penalty or mode repulsion is added.

Default lambda_disp=1; E uses zero, and an additional D run uses five. Optional endpoint supervision uses the same full-trajectory winner and is disabled by default.

## Ablations

| Label | Configuration |
|---|---|
| A / Frozen6 | Native predictions on the exact same targets |
| B | Learned null plan tokens; ALL plan-derived geometry disabled; static agent interaction retained |
| C | Direct plan + geometry; post-conditioning interaction disabled |
| D | Full model |
| E | D with lambda_disp=0 |

The default optimizer is a single AdamW for all new modules (lr=2e-4, weight_decay=1e-2, gradient clipping=2). NDPNet is absent from the optimizer and cached-training graph. The optional online guard forces backbone eval, including dropout and normalization buffers. Epochs=30, batch size=4 scene groups, seeds=0/1/2. These are initial engineering defaults, not validated optimal values.

Selection uses **validation group-balanced 8s pool minADE**; ties keep the earlier checkpoint. Reports also include target-micro and class-macro statistics. Test data is not used. Checkpoints contain new module weights, optimizer state, config, unit normalization, commit, actual source hashes, seed, split checksums, extractor version and backbone hash. They contain no backbone weights.

## Commands

Install the project (`pip install -e .`). Cache extraction additionally needs the official Scenario protobufs and torch-geometric for existing trusted scene graph files. The source-root contract is the audited manifest format, with explicit context/feature/target/native paths; this is not a generic raw-TFRecord preprocessing replacement.

```bash
python scripts/build_scene_cache.py \
  --source-root /path/to/audited-native-manifests \
  --womd-proto-path /path/to/official-womd-python-package \
  --output /path/to/new-scene-cache --train-limit 128 --val-limit 32

# Tiny overfit on training scenes only, before substantive training.
python scripts/train_interaction_k12.py --cache /path/to/new-scene-cache \
  --output outputs/tiny --variant D --device cuda:0 \
  --overfit-groups 2 --overfit-steps 1000

# Same data/settings for B/C/D/E; use unique output directories.
python scripts/train_interaction_k12.py --cache /path/to/new-scene-cache \
  --output outputs/D_seed0 --variant D --seed 0 --device cuda:0
python scripts/train_interaction_k12.py --cache /path/to/new-scene-cache \
  --output outputs/B_seed0 --variant B --seed 0 --device cuda:1
# Repeat with --variant C, --variant E, and D --seed 1 / --seed 2.
# For the displacement grid: --variant D --lambda-disp 5.

python scripts/evaluate_interaction_k12.py --checkpoint outputs/D_seed0/best.pt \
  --manifest /path/to/new-scene-cache/val.jsonl --output outputs/D_seed0/evaluation.json --device cuda:0
python scripts/diagnose_interaction_plan.py --checkpoint outputs/D_seed0/best.pt \
  --manifest /path/to/new-scene-cache/val.jsonl --output outputs/D_seed0/conditioning.json
python -m unittest discover -s tests -v
```

Omit cache limits for full train/validation export, using a new output directory. Export does not open test scene tensors; it reads test manifest identifiers only to check split leakage. Source records with missing ego future, irregular timestamp intervals beyond 5ms tolerance, or context caps that would drop targets are explicitly excluded. No synthetic interpolation repairs them silently.

## Evaluation and interpretation

Report Frozen6/new6/pool12 minADE/minFDE at 3/5/8s on the same targets. ADE and FDE evaluation independently choose their oracle modes; the training displacement loss uses a single ADE winner. The evaluator records new-vs-native ADE win rate/gain quantiles, new-mode usage, pairwise new/new and new/native distance, duplicate fraction, best-new displacement error and speed/acceleration anomaly fractions. Diagnostic thresholds are output explicitly; they are not safety criteria.

The plan diagnostic holds native/context fixed and compares the recorded path against a slight same-path time warp. It checks kinematic/path consistency, not full traffic-rule/collision feasibility. It never assigns factual GT accuracy to the alternative plan or claims causal reaction accuracy. A pair with fully certified traffic feasibility still requires an external planner/map audit; this remains a limitation of the functional diagnostic.

Frame tests validate the new module's coordinate contract while holding frozen embeddings fixed; they do not claim perfect rigid-transform invariance of the upstream NDPNet. No PDM candidate combination or closed-loop evaluation logic is changed.

## Initial data and validation scope

The local pilot selects the first 128 training manifest scenes and 32 external validation scenes, shared by every ablation. Six training records have verified non-10Hz gaps (including 0.2s intervals) and are excluded: 122 train / 32 validation groups remain. This deterministic convenience subset is not a representative full-benchmark claim. Larger training and external evaluation are needed before concluding the model improves interactive prediction.

The first 200-step unnormalized tiny run learned slowly. Adding explicit output-unit normalization (30m, matching positional input scale) was decided from training-only tiny behavior. A separate 1,000-step two-scene run lowered new-only 8s ADE from 44.109m to 0.462m and FDE from 109.465m to 0.949m. Those are memorization/optimization checks, not held-out accuracy.
