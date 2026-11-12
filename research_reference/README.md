# Research reference scripts

These are provenance records for the locked experiment, not the portable public API.
Original training scripts depend on private experiment caches and other experimental modules.
The runnable package is `src/recorded_k12`; the portable cache-based training entry point is
`scripts/train_cache.py`. Planning snapshots require external nuPlan, Flow Planner, PDM,
and NDPNet feature-extraction infrastructure; they are NOT standalone benchmark launchers.
Machine paths in planning snapshots have been replaced with `/path/to/experiment-root`.
Original file hashes are recorded in `provenance.json`; hashes describe original, unsanitized files.
Trailing blank lines are normalized in the packaged snapshots; model definitions are unchanged.
No raw scene data, scene IDs, model backbone weights, or experiment logs are distributed here.
