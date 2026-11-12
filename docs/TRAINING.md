# Training and checkpoint provenance

The shipped files are byte-for-byte copies of three locked research artifacts:

| File | Original stage | Loaded fields |
|---|---|---|
| anchor.pt | cv_direct_crossfit/fold_0/selected.pt | anchor |
| five.pt | separate_horizon12/fold_0.pt | state, scale |
| eight.pt | conflict_order_response/fold_0.pt | state, scale |

The anchor artifact also contains an older correction state; the locked model **does not load or use it**. It is retained only to preserve the exact artifact hash. No native NDPNet checkpoint is embedded.

## Historical fitting

Research caches were prepared from InterHub-selected native WOMD data. They contain target rows, not one row per scene, and excluded external evaluation scene populations. Scene-disjoint cross-validation used fit, inner-selection and outer-evaluation scene groups. The released model uses fold 0, not an ensemble of folds.

1. Train an anchor residual using CV context, a frozen-plus-proposal oracle loss across 5s/8s ADE/FDE, an additional proposal error term, and a small residual penalty. Native predictions are fixed. Batch size 256, 750 steps, AdamW learning rate 0.0006, weight decay 0.002, gradient norm clipping 2; inverse-square-root class-frequency sampling; seed 272 for fold 0.
2. Freeze the anchor. Train the 8s response head using full-horizon conflict descriptors, endpoint oracle loss plus a soft-min term and residual penalty. 800 steps, batch size 256, AdamW 0.0006/weight decay 0.01, seed 6791 for fold 0. Select checkpoint and scale from 0.25/0.5/1 using inner-validation class-macro endpoint benefit over the CV pool.
3. Train a separate 5s head using geometry through 5s. Vehicle targets must have valid 5s and 8s endpoints. 800 steps, batch size 256, AdamW 0.0006 with its default weight decay, seed 590. Select checkpoint/scale using inner-validation 5s benefit over the CV pool.
4. Lock the three artifacts. The full trajectory formula combines the 5s and 8s responses without replacing native modes.

GT supplies target prediction losses only; recorded ego future supplies conditioning. No impacted-agent annotation or impact-router objective is used by this version. The original 8s response training filters for a valid endpoint; the portable recipe uses the same endpoint supervision.

## Portable training

`scripts/train_cache.py` implements the same three head architectures and objective families from a prepared train/validation pair of NPZ caches. It enforces scene-disjointness and uses validation only to select response heads/scales. It retains the anchor after the specified fixed number of steps, as in the original anchor stage. There is no test-set tuning.

This is a portable retraining recipe, not a claim of byte-identical historical reproduction: it does not recreate proprietary/private cache construction or the historical five-fold assignment. Missing classes are omitted from the validation macro average; zero response is retained if no candidate improves validation. The full published artifacts remain the reference for the reported results.

Use separate output folders for ablations. `--anchor-steps` and `--response-steps` may be reduced for synthetic smoke tests; such tests are not research evidence. A single invocation uses one selected device; independently run different folds/seeds on separate GPUs if desired. No new full training was performed solely to package this release.
