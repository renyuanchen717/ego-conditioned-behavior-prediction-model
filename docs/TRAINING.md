# Training and evaluation
Install the package and prepare the authorized local model cache described in DATA_FORMAT.md. No dataset is distributed in this repository.

```bash
torchrun --standalone --nproc_per_node=2 scripts/train_dm2i_ddp.py \
  --cache /data/dm2i/cache --output outputs/retrain --config configs/d_m2i.json
```
Use `--resume` with the same output to continue from its last checkpoint. The historical checkpoint was trained with initial DataParallel epochs and resumed with equivalent grouped DDP updates; a fresh DDP run is not guaranteed bit-identical. The inference checkpoint excludes optimizer state.

For a single device, `scripts/train_interaction_k12.py --cache ... --output ... --config configs/d_m2i.json --variant D` uses the same architecture/objective and requires the preprocessing audit.json. `scripts/evaluate_interaction_k12.py --help` documents standalone evaluation.
Validation chooses checkpoints; test results are not used by these training scripts. Release tests use synthetic inputs and original-checkpoint tensor comparisons, not benchmark tuning.
