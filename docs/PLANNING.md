# Flow Planner + Recorded K=12 integration

The independent planning pipeline uses the locked model as follows:

1. Flow Planner generates 16 ego candidate trajectories.
2. Frozen NDPNet predicts six modes for up to ten selected currently observed vehicle targets. This is target eligibility/selection, not ego-impact detection.
3. For each ego candidate, create the additional six conditional proposals for each target. No GT future is used during planning.
4. Evaluate each candidate under all twelve forecast mode indices.
5. Choose the candidate with the highest worst-mode PDM score:

$$i^*=\arg\max_i\min_{k\in\{1,\ldots,12\}}S_{PDM}(E_i,\mathcal P_{:,k}(E_i)).$$

The implementation combines **the same raw mode index across agents**. These twelve combinations are not learned joint modes and do not enumerate all cross-agent combinations. No learned mode probability weights the worst-case score. Additional proposals can change planning behavior even when original NDPNet predictions are preserved exactly.

The current experiment uses 8s scoring at 10 Hz. Collision/TTC occupancy extends to 9s using constant terminal center velocity and heading after 8s. Candidate execution follows the nuPlan closed-loop controller, replanning as configured by the simulator.

## Official 335-scenario InterPlan comparison

- Both arms use CFG=1.8, the aligned released Flow Planner checkpoint and the same 335 augmented scenarios.
- Baseline uses Flow Planner's single-trajectory inference; the integrated arm generates 16 candidates and scores them.
- The candidate arm retains timestamp-derived per-step random seeds. The baseline uses its original continuous RNG stream. Therefore, scenarios are matched, but generated noise is not paired between arms.
- The source of truth for scores is the official InterPlan aggregator. The evaluation is still running at release preparation; no partial score is presented as final in this repository.

## Integration boundary

`research_reference/planning` preserves the relevant original model and scoring bridge for inspection. It is not a turn-key simulator distribution. The public `RecordedK12.predict_10hz` API provides the portable prediction boundary. Supply target-local predictions and ego plans, transform its outputs back to map coordinates, and integrate with your installed simulator/PDM stack. NDPNet, Flow Planner, nuPlan, InterPlan, map data, target graph construction and PDM dependencies must be installed separately.
