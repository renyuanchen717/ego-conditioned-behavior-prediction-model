# Model card

**Name:** Recorded K=12. **Release:** locked fold-0 residual heads.

**Purpose:** investigate whether a supplied ego future improves other-agent trajectory coverage while retaining the frozen prediction pool.

**Inputs:** six NDPNet means/probabilities, target type, CV ego reference and supplied ego future in a target-local frame. **Outputs:** twelve 2D trajectories over 8s. No new mode probabilities, uncertainty calibration, agent-impact classification, map polygons or direct scene embeddings are produced by the residual head.

**Learned parts:** CV anchor correction, 5s response head, 8s response head. **Frozen parts:** external pretrained NDPNet. **Conditioning:** recorded ego future in diagnostics, candidate ego plan in planning.

**Data provenance:** research caches originating from native WOMD/InterHub selection; original group-disjoint cross-validation. The 5s head is vehicle-trained. Applying it to pedestrians/cyclists is transfer beyond that head's training population; class results must be inspected separately.

**Limitations:** no causal identification from factual logs alone; no explicit handling of indirect interaction chains; proposals can be redundant or physically infeasible; oracle gains over a six-mode baseline are partly due to larger K; recorded-plan gains over CV are small. Geometry features may have numerical boundary sensitivities inherited from upstream coordinate processing. No new rigid-transform invariance claim is made by packaging these heads.

**Preservation:** original six means are bitwise unchanged by the released concatenation path. This does not ensure preserved calibration, top-1 performance or worst-case downstream planning scores.

**Validation status:** synthetic inference/training smoke and parity checks are recorded in release audit artifacts. Historical aggregate results are included separately. Data and original cache preprocessing are not distributed, so end-to-end dataset reproduction requires external preparation and matching eligibility rules.
