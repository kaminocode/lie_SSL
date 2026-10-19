# K=1 Lie-action MNIST experiments

Eight settings compare dense/block-skew generators and scalar/circle pose on rotated and blocked rotated MNIST. Every model has canonical dimension 24 and one coefficient. Train for 300 epochs, batch 256, Adam 0.001, seed 42, with observed/swap/canonical weights (1,1,1).

| Variant | Generator | Pose |
|---|---|---|
| baseline | Learned dense, Frobenius norm 1 | Unbounded scalar |
| block_skew | Twelve fixed 2×2 skew blocks, each spectral norm 1 | Unbounded scalar |
| dense_circle | Learned dense, Frobenius norm √24 | Normalized (cos, sin), then atan2 |
| block_skew_circle | Twelve fixed 2×2 skew blocks, each spectral norm 1 | Normalized (cos, sin), then atan2 |

Circle coefficients are in (-π,π]. The dense-circle action is not constrained to be periodic. Report its seam/periodicity residual. Unit-frequency 2×2 skew blocks have no continuously learned magnitude; their encoder and decoder remain learned.

The original 28×28 data protocol is preserved: stratified 54k/6k/10k splits, two fresh training rotations per epoch, fixed validation/test rotations. In the blocked dataset, a source-fixed white 7×7 block (margin 1, seed 1234) is applied after rotating the digit.

## Downstream evaluation

Frozen canonical and pose representations receive digit classification and relative-rotation probes. Circle pose features are two-dimensional; scalar pose features are one-dimensional. Probes use 100 epochs, Adam 0.001, batch 256, seeds 42–46. Rotation uses the reference 1024→1024→2 MLP on ordered pairs, trains both directions, and selects the best validation circular MAE. Classification uses a linear head and validation accuracy. There are 160 fits.

`rotation_r2` is the uniform mean of `rotation_r2_cos` and `rotation_r2_sin`, calculated from raw regression predictions over the complete test set and both directions. Predictions are not normalized before scoring. Negative R² is retained; a constant target component makes its score undefined (JSON null and blank CSV). Circular MAE/RMSE and threshold accuracies remain available. No principal-angle R² is reported.

`comparison/` contains rendering metrics, per-fit and per-digit metrics, seed summaries, an independently recomputed prediction audit, and `report.md`. Seed standard deviations measure probe-seed variability, not backbone-training uncertainty. Each run contains best/last checkpoints with optimizer state, configuration, data hashes and RNG state, CSV learning curves, prediction exports, diagnostics, and reconstruction/transfer/sweep PNG/PDFs with pixel arrays.

## LUMI execution

The launcher pins the September 29, 2026 LUMI AI Factory ROCm container, which contains all dependencies. It uses existing MNIST files read-only. No runtime pip installation or additional environment is needed. Production GPU jobs use one standard-g node with eight independent GCD workers, seven CPUs per worker (56 total), and 60 GB memory per GCD; logs are under this checkout's `logs/`. Allocation inspection selected `project_465003463` because the older default has no allocation.

```bash
python3 slurm/submit.py smoke --dry-run
python3 slurm/submit.py smoke
# After all eight smoke settings and the 32-fit smoke audit pass:
python3 slurm/submit.py production --smoke-root /scratch/project_465003463/checkpoints/baburaja/lie_SSL/smoke_TIMESTAMP
```

The launcher creates a fresh run root under `/scratch/project_465003463/checkpoints/baburaja/lie_SSL/`, snapshots source files, records hashes/environment/manifest/jobs, and refuses existing roots. Production checks that the source matches the successful smoke snapshot. Training, rendering, probe, and aggregation stages have `afterok` dependencies and one production node at a time. Training and rendering each use eight workers; the 160 probes use twenty groups of eight (`--array=0-19%1`). The final CPU aggregation remains on `small`. Pass `--partition small-g` explicitly to restore single-GCD production jobs. The dev-g submission limit is two, so two smoke jobs each run four settings sequentially. Logging uses offline W&B.

Resume a failed array element by running its recorded `jobs.json` command with `--array=INDEX` and without the old dependency. Use the saved source snapshot and original run root. Backbone and probe entrypoints resume their last checkpoints and reject incompatible configurations. If a prerequisite failed, downstream jobs are canceled by Slurm's invalid-dependency policy; resubmit the required stages from the recorded commands after recovery. Existing running jobs are never canceled automatically by this software.

The smoke exercises all variants, both probe tasks/representations, checkpoint resume, generator contracts, reference baseline parity, and R² audits. It also checks exact full-entrypoint backbone resume for the dense baseline and block-skew circle variants. Smoke metrics are only pipeline verification, not scientific results.

The existing production run was moved to standard-g using a separate, hashed launcher snapshot. Its original scientific source snapshot and output root are preserved. `latest_production.json` records the replacement job IDs and launch commands.

Requested wall-time limits: backbone training 5 hours, rendering evaluation 1 hour, each eight-probe group 24 hours, and final CPU aggregation 10 minutes.
