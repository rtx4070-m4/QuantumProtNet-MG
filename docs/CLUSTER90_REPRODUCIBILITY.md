# Cluster-Aware Development Experiment: Reproducibility Guide

## 1. Purpose

This document describes the cluster-aware development comparison between
the QuantumProtNet-MG hybrid simulator and its classical projection comparator.
It is separate from the earlier scaffold-held-out development experiment
documented in `RESULTS_REPORT.md` and `REPRODUCIBILITY.md`.

## 2. Evaluation scope

- Evaluation split: cluster-held-out development/validation split.
- Training rows: 42,787.
- Validation rows: 10,697.
- Validation fraction: approximately 20%.
- Random seeds: 42, 123, and 2026.
- Official test set: not loaded by these runs.
- Validation metrics are not an independent final-test estimate because
  validation performance was used for checkpoint selection.

The split manifest used in the experiment was
`results/sequence_similarity_audit_v2/cluster90_cov80_cluster_heldout_split_seed20261021.csv`
in the original working environment. The sequence-level manifest is not
included in this repository package.

## 3. Split integrity and limitations

The split audit reported no cluster IDs crossing the training and validation
partitions. A separate MMseqs2 cross-partition sequence-similarity audit
nevertheless found some hits at the configured 80% identity and 80% coverage
thresholds. Therefore, the split should not be described as having zero
cross-partition sequence similarity.

The original experiment used publicly accessible TAPE stability data. Users
reproducing the work must obtain the appropriate data independently and verify
that their dataset version, row identifiers, labels, and parent identifiers
match the manifest expected by the scripts.

## 4. Models

- **Hybrid:** custom sequence encoder with a differentiable four-qubit
  PyTorch statevector simulator.
- **Classical comparator:** custom sequence encoder with four learned
  classical projection features.

Both models were evaluated on the same fixed cluster-aware split for each
seed. The simulator runs on classical hardware; no quantum hardware was used.
The experiment does not demonstrate quantum advantage.

## 5. Reported validation results

Mean and sample standard deviation across three seeds:

| Model | Metric | Mean | Sample SD |
|---|---|---:|---:|
| Hybrid | MAE | 0.370510 | 0.004363 |
| Classical | MAE | 0.372770 | 0.002272 |
| Hybrid | RMSE | 0.477165 | 0.004505 |
| Classical | RMSE | 0.478293 | 0.002120 |
| Hybrid | Pearson r | 0.670774 | 0.011106 |
| Classical | Pearson r | 0.669604 | 0.005937 |

The hybrid model had lower MAE in two of the three seeds. The mean paired
MAE improvement relative to the classical comparator was approximately 0.61%.
This is a descriptive comparison across three seeds, not a statistical
significance test. It is not evidence of quantum advantage.

See `CLUSTER90_EXPERIMENT_REPORT.md` for the experiment report and the
CSV/JSON summary files under `../results/`.

## 6. Scripts and configuration

The cluster-aware scripts are:

- `../src/train_quantumprot_hybrid_cluster90.py`
- `../src/train_classical_projection_cluster90.py`

Important: both scripts default to the older scaffold-held-out manifest when
`QP_MANIFEST` is not set. Reproduction of this cluster-aware experiment
requires explicitly setting `QP_MANIFEST` to the cluster-aware manifest.

The scripts also use these environment variables:

- `QP_MANIFEST`: path to the selected split manifest.
- `QP_SEED`: random seed.
- `QP_RESULT_DIR`: output directory for the run.

The original run manifest was stored outside this repository package. The
commands below are templates; they are not directly executable until the
dataset, manifest, Python environment, and dependencies have been configured.

## 7. Windows PowerShell command templates

Run commands from the repository root after setting `$manifest` to the
verified local path of the cluster-aware manifest and activating a compatible
Python environment. Ensure the dataset expected by the training scripts is
available before running.

```powershell
$manifest = "C:\PATH\TO\cluster90_cov80_cluster_heldout_split_seed20261021.csv"

$env:QP_MANIFEST = $manifest
$env:QP_SEED = "42"
$env:QP_RESULT_DIR = ".\results\cluster90_hybrid_seed42"
python .\src\train_quantumprot_hybrid_cluster90.py

$env:QP_SEED = "123"
$env:QP_RESULT_DIR = ".\results\cluster90_hybrid_seed123"
python .\src\train_quantumprot_hybrid_cluster90.py

$env:QP_SEED = "2026"
$env:QP_RESULT_DIR = ".\results\cluster90_hybrid_seed2026"
python .\src\train_quantumprot_hybrid_cluster90.py

$env:QP_SEED = "42"
$env:QP_RESULT_DIR = ".\results\cluster90_classical_seed42"
python .\src\train_classical_projection_cluster90.py

$env:QP_SEED = "123"
$env:QP_RESULT_DIR = ".\results\cluster90_classical_seed123"
python .\src\train_classical_projection_cluster90.py

$env:QP_SEED = "2026"
$env:QP_RESULT_DIR = ".\results\cluster90_classical_seed2026"
python .\src\train_classical_projection_cluster90.py
```

These commands create new run outputs in the specified directories. Review
existing output directories before rerunning, and do not overwrite valuable
checkpoints or logs unintentionally.

## 8. Reproducibility requirements

For a reproducible rerun:

1. Record the Python and PyTorch versions, device, and installed dependencies.
2. Obtain and document the dataset source and version.
3. Verify the manifest path, row counts, labels, and parent identifiers.
4. Confirm that the cluster-aware partition is loaded in the console output.
5. Run both models with the same three seeds and fixed split.
6. Retain run logs and per-seed metric JSON files locally.
7. Report all seeds, not only the best-performing seed.
8. Keep validation results distinct from any future independent test evaluation.

## 9. Package contents and artifact provenance

The repository package includes the two cluster-aware training scripts,
this guide, the experiment report, and aggregate CSV/JSON summaries.

The original SHA-256 manifest was generated in the experiment workspace and
references additional files that are not included in this repository package.
It is retained as an original-workspace artifact record, not as a complete
integrity manifest for this package. Generate a separate manifest for the
published package files.

## 10. Scientific reporting cautions

- The validation set was used for checkpoint selection.
- Three seeds provide limited descriptive evidence.
- Some sequence-similarity hits remain across partitions.
- The official test set was not loaded by these runs.
- The simulator is classical software, not quantum hardware.
- Neither a statistically significant improvement nor quantum advantage
  has been established by this experiment.
