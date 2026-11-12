# Evaluation protocol and limits

## Locked comparisons

- Frozen NDPNet: original six trajectories.
- CV K=12: original six plus six learned CV-anchor proposals.
- Recorded K=12: original six plus six proposals conditioned on recorded ego future.

Increasing K improves oracle coverage even without a more informative ego plan. The Recorded-minus-CV comparison isolates the additional recorded-plan contribution much better than Recorded-minus-frozen. The observed incremental benefit is small.

## Classic validation

The locked full-trajectory evaluation used a vehicle-only, disjoint validation slice with 5,209 scenes and 14,014 eligible targets. The 5s FDE valid-endpoint count is 13,937. This is not a K=12 evaluation on all 44,097 classic validation scenarios. Earlier full-validation experiments used other model variants.

The JSON metric order is 3s ADE/FDE, 5s ADE/FDE, 8s ADE/FDE. Errors are meters and independently oracle-minimized over modes. Invalid timestamps are excluded from ADE; FDE requires a valid endpoint at the specified horizon.

## Interaction validation

40,341 pair-conditioned scenarios/pairs: use one member as the conditioning ego and predict the other. This includes designated conditioning agents that need not be the native autonomous-driving ego. The target and conditioning IDs must differ. The historical manifest determines the pair assignment and dataset exclusions.

Reported headline values are class-macro averages across vehicle, pedestrian and cyclist, not target-weighted micro averages. There are 40,338 targets with valid ADE supervision, 40,234 valid 5s endpoints and 39,134 valid 8s endpoints. A separate matched-vehicle subset supports the classic-versus-interaction gain comparison.

These are conditional marginal diagnostics, **not official joint interaction challenge results**. Recorded future conditioning is privileged information relative to the official prediction protocol. No official hidden-test score, joint mAP, or calibrated 12-mode probabilities is claimed.

## Runnable evaluator

`scripts/evaluate_npz.py` reports per-target micro-averaged minADE/minFDE for a supplied cache and all three arms. Its default aggregation is intentionally labeled micro and should not be mistaken for the published interaction class-macro table. To recreate that table, run class-specific caches and average the three class means, using the same eligible population.

The evaluator does not call the official WOMD metric implementation and does not produce MR or mAP. The result JSONs in this repository are historical aggregate artifacts, not outputs of a new full dataset run during release preparation.

## Planning is a separate experiment

Flow Planner generates ego candidates; Recorded K=12 forecasts target trajectories under each candidate; a PDM-based scorer selects an ego trajectory. InterPlan/nuPlan planning scores cannot be equated with WOMD prediction metrics. See PLANNING.md.
