# Source changes

## 2026-10-16 — K=1 MNIST generator/circle comparison

- Ported the reference 28×28 encoder/decoder, loss, MNIST rendering/split protocol, and image metrics into a standalone package. `source_provenance.json` records upstream paths and SHA-256 hashes; the reference repository is unchanged.
- Added dense, fixed unit-frequency 2×2 skew-block, scalar, and normalized circle-head variants. Dense circle generators have Frobenius norm √d; baseline generators retain unit Frobenius normalization.
- Added resumable training, rendering, frozen classification/relative-rotation probes, full-dataset component R², exported-prediction audits, and seed summaries.
- Added ROCm container validation, LUMI smoke/production launchers, immutable run snapshots, and numerical/integration tests.

- Adapted dev-g smoke scheduling to the confirmed two-job submission limit: two workers cover four settings each.
- Applied the training deterministic settings to GPU numerical tests after detecting nondeterministic convolution-gradient reduction differences in the first smoke.
- Moved small generator-spectrum diagnostics to CPU LAPACK after the pinned ROCm build reported no GPU MAGMA eigenvalue support; model execution remains on GPU.

## 2026-10-16 — Move production to standard-g

- Added full-node launchers using eight independent one-GCD workers and seven CPUs each; validated exact index coverage for eight backbone/render tasks and 160 probes.
- Made standard-g the production default, with twenty probe groups and one active node. CPU aggregation stays on small. Added a runtime assertion that each worker sees exactly one ROCm GPU.
- Retained the scientific source snapshot and output root when replacing the pending small-g chain; recorded new launcher hashes and prior job IDs separately.

## 2026-10-16 — Five-hour backbone limit

- Reduced pending backbone job 22907296 to 05:00:00 and updated production training launcher defaults and current resubmission commands. Evaluation and probe wall times retain their existing values.
