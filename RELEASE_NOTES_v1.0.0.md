# QuantumProtNet-MG v1.0.0 — Release Candidate Notes

## Scope

This release candidate packages the current reproducible development comparison for QuantumProtNet-MG. It is a research prototype for protein-stability regression, not a clinically validated or production-ready predictor.

## Models

- **Hybrid prototype:** custom neural sequence encoder with a differentiable four-qubit, two-layer PyTorch statevector simulator.
- **Classical comparator:** custom neural sequence encoder with four learned classical projection features and no quantum simulation.
- Both models are custom implementations. The simulator runs on classical GPU hardware; no quantum hardware is used.

## Cluster-aware development comparison

Three random seeds (42, 123, 2026) were evaluated for each model on the same fixed cluster-aware development/validation split.

| Metric (mean ± sample SD across seeds) | Hybrid simulator | Classical comparator |
|---|---:|---:|
| Validation MAE ↓ | 0.370510 ± 0.004363 | 0.372770 ± 0.002272 |
| Validation RMSE ↓ | 0.477165 ± 0.004505 | 0.478293 ± 0.002120 |
| Validation Pearson r ↑ | 0.670774 ± 0.011106 | 0.669604 ± 0.005937 |

The hybrid model had lower validation MAE in two of three seeds. The mean paired MAE improvement was approximately 0.606% relative to the classical comparator. This is descriptive evidence across three seeds, not a statistical significance test.

## Important evaluation limitations

- The reported metrics are validation metrics. Validation performance was used to select the best epoch/checkpoint, so these scores are not an independent final-test estimate.
- The cluster manifest assigns each cluster ID to one cluster partition, but a separate sequence-similarity audit found cross-partition hits at the configured thresholds. Cluster-ID separation must not be described as zero sequence similarity.
- A raw-data audit found 26 unique exact sequences shared between the original `train.csv` and official `test.csv`. The audit did not establish that all 26 occur in the cluster-aware development-training subset. Treat the official test set as unsuitable for an untouched-test claim until the provenance and overlap are resolved.
- The manifest has both `cluster_partition` and a generic `partition` column with different assignments. The cluster-aware scripts explicitly select `cluster_partition` when it is present and validate cluster-ID separation. Use the documented cluster-aware manifest path when reproducing these results.
- No quantum advantage, statistical significance, biological-family generalization, or external generalization is established.

## Reproduction artifacts

- `docs/CLUSTER90_EXPERIMENT_REPORT.md`
- `docs/CLUSTER90_REPRODUCIBILITY.md`
- `results/cluster90_comparison_summary.json`
- `results/cluster90_comparison_seed_summary.csv`
- `results/cluster90_model_summary.csv`
- `results/cluster90_paired_seed_comparison.csv`
- `results/sequence_similarity_audit_v2/cluster90_cov80_cluster_heldout_split_seed20261021.csv`

## Environment recorded for the experiment

- Python environment used for training: PyTorch 2.11.0+cu128, CUDA build 12.8.
- GPU: NVIDIA GeForce RTX 3050 Laptop GPU, 4 GB reported VRAM.
- Seeds: 42, 123, 2026 per model.
- The public repository does not include the raw dataset; obtain it separately and verify dataset identity and manifest compatibility before reproducing.

## Recommended next research step

Before claiming independent generalization, establish a new final holdout at the sequence-cluster level, freeze the architecture/hyperparameters using development data only, and evaluate that final holdout once. If using external data, document dataset provenance, sequence/label harmonization, exclusions, and overlap checks.
