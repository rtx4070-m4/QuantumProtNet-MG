# QuantumProtNet-MG: Cluster-Aware Validation Experiment

Report generated (UTC): 2026-10-21T11:15:07+00:00

## 1. Executive summary

Six training runs were completed: a four-qubit statevector-simulator hybrid model and a classical projection ablation, each trained with seeds 42, 123, and 2026. The models used the same cluster-aware development/validation manifest. All reported scores in this report are validation metrics.

The hybrid model's mean validation MAE was **0.370510** (sample SD 0.004363), versus **0.372770** (sample SD 0.002272) for the classical model.

These descriptive results do not establish statistical significance, quantum advantage, or performance on an untouched official test set.

## 2. Models and experimental protocol

- **Hybrid:** differentiable PyTorch statevector simulator, 4 qubits, 2 quantum layers, 16 statevector amplitudes, 17,376 trainable parameters.
- **Classical ablation:** four learned classical projection features with the quantum circuit disabled, 17,360 trainable parameters.
- **Hardware recorded in training logs:** NVIDIA GeForce RTX 3050 Laptop GPU with 4 GB reported GPU memory.
- **Software recorded in training logs:** PyTorch 2.11.0+cu128, CUDA build 12.8.
- **Random seeds:** 42, 123, and 2026 for each model.
- **Training:** maximum 30 epochs with early stopping; best checkpoint selected using validation MAE.
- **Data use:** training scripts reported 42,787 development training rows and 10,697 validation rows. The official test set was not loaded by these runs.
- **Evaluation:** validation metrics were reported from the checkpoint-selection split; they are not independent final-test estimates.

## 3. Data split and similarity audit

- Manifest: `results\sequence_similarity_audit_v2\cluster90_cov80_cluster_heldout_split_seed20261021.csv`
- The split was generated using MMseqs2 clustering at 90% sequence identity and 80% coverage settings.
- The cluster-aware partition audit reported zero cluster IDs crossing between the development training and validation partitions.
- The validation proportion was approximately 20%.
- A separate cross-partition MMseqs2 search at >=80% identity and >=80% coverage found 353 hit pairs involving 212 unique validation queries (about 1.98% of validation queries). At >=90% identity and >=80% coverage, 6 hit pairs involving 6 unique validation queries were found.
- Therefore, do not describe the split as having zero sequence-similarity hits. Zero cluster-ID crossing and zero cross-partition similarity hits are different claims.

## 4. Results across seeds

| Model | Metric | Mean | Sample SD | Minimum | Maximum |
|---|---|---:|---:|---:|---:|
| Hybrid | MAE (lower is better) | 0.370510 | 0.004363 | 0.365762 | 0.374342 |
| Hybrid | RMSE (lower is better) | 0.477165 | 0.004505 | 0.472101 | 0.480731 |
| Hybrid | Pearson r (higher is better) | 0.670774 | 0.011106 | 0.658171 | 0.679128 |
| Classical | MAE (lower is better) | 0.372770 | 0.002272 | 0.370852 | 0.375280 |
| Classical | RMSE (lower is better) | 0.478293 | 0.002120 | 0.476897 | 0.480732 |
| Classical | Pearson r (higher is better) | 0.669604 | 0.005937 | 0.662756 | 0.673306 |

### Paired comparison

| Seed | Hybrid MAE | Classical MAE | Hybrid MAE improvement | Hybrid RMSE | Classical RMSE |
|---:|---:|---:|---:|---:|---:|
| 42 | 0.365762 | 0.370852 | +1.373% | 0.472101 | 0.476897 |
| 123 | 0.371425 | 0.375280 | +1.027% | 0.478662 | 0.480732 |
| 2026 | 0.374342 | 0.372179 | -0.581% | 0.480731 | 0.477249 |

Positive MAE improvement means hybrid MAE is lower than classical MAE. Negative improvement means the classical model has lower MAE.

- Mean paired difference (classical MAE minus hybrid MAE): 0.002260 (sample SD 0.003880).
- Mean relative hybrid MAE improvement: +0.606% (sample SD 1.042668 percentage points).
- Hybrid had lower MAE in 2 of 3 seeds; classical had lower MAE in 1 of 3 seeds.

## 5. Interpretation

The hybrid model has slightly better mean validation MAE, RMSE, and Pearson correlation in this three-seed comparison. The differences are small, and the hybrid model shows greater run-to-run variability across these seeds. The classical model performs better on MAE for seed 2026.

This is preliminary comparative evidence for these implementations and this fixed split only. Three seeds on the same split do not establish statistical significance or broad generalization. No quantum hardware was used; the hybrid circuit was simulated classically, so quantum advantage is not demonstrated.

## 6. Limitations and next validation requirements

1. Validation data were used to select the best checkpoint, so validation metrics may be optimistic as estimates of final generalization.
2. The cross-partition sequence-similarity hits require transparent reporting and further review if stronger sequence-family separation is a study objective.
3. This report does not present a statistical significance test or confidence interval for model differences.
4. Before making a final performance claim, specify an evaluation protocol that uses a genuinely untouched test set, while documenting the project's previous test-set inspections and any resulting limitations.
5. A fair architecture comparison should keep data, optimization protocol, parameter counts, and compute accounting as comparable as possible.

## 7. Reproducibility artifacts

- Per-run metrics, checkpoints, and training histories are stored in their respective `results/cluster90_*_seed*` directories.
- Aggregate CSV/JSON files are stored in `results/`.
- SHA-256 integrity manifest: `results\cluster90_sha256_manifest.csv`.
- Number of existing artifacts hashed: 30.

SHA-256 hashes identify the exact files present at report generation time; they do not by themselves prove scientific validity or that a run can be reproduced on different hardware/software.

## 8. Suggested reporting statement

"Across three training seeds on a fixed cluster-aware validation split, the simulator-based hybrid model showed a small improvement in mean validation metrics over the classical projection ablation, with greater run-to-run variability. The observed differences are descriptive and do not establish statistical significance or quantum advantage. Validation was used for checkpoint selection, and cross-partition sequence-similarity hits were observed in a separate MMseqs2 audit."
