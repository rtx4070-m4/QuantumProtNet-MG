# Final Release Audit — QuantumProtNet-MG

Audit date: 2026-10-21

## Release decision

**Suitable as a documented research-prototype release candidate; not suitable for claims of independent final-test performance or quantum advantage.**

## Verified artifacts and experiment results

- Six cluster-aware runs were completed: hybrid simulator and classical projection comparator, seeds 42, 123, and 2026.
- Each of the six run folders contained one metrics JSON, one checkpoint, and one log in the local reproduction audit.
- The six run-level MAE, RMSE, and Pearson correlation values matched the tracked cluster-aware seed summary exactly.
- The tracked manifest contains 53,484 rows: 42,787 `cluster_development_train` and 10,697 `cluster_heldout_validation`.
- No cluster ID crosses the `cluster_partition` assignments.
- The cluster-aware training scripts choose `cluster_partition` when that column exists and validate that cluster IDs do not cross the selected cluster partitions.
- Raw data checks confirmed that each sequence's length matches the recorded `protein_length` in the original training CSV (see the local reconciliation audit).

## Reported validation metrics

| Metric (mean ± sample SD) | Hybrid simulator | Classical comparator |
|---|---:|---:|
| MAE | 0.370510 ± 0.004363 | 0.372770 ± 0.002272 |
| RMSE | 0.477165 ± 0.004505 | 0.478293 ± 0.002120 |
| Pearson r | 0.670774 ± 0.011106 | 0.669604 ± 0.005937 |

The hybrid model had lower validation MAE in two of three seeds. The mean paired MAE improvement was approximately 0.606%. Three seeds are descriptive; this is not a significance test.

## Limitations and risks

1. **Validation reuse:** best epoch/checkpoint selection used the same validation split whose metrics are reported. These are not independent final-test estimates.
2. **Sequence similarity:** the MMseqs2 cross-partition audit found sequence-similarity hits at the configured thresholds, despite zero cluster IDs crossing the manifest's `cluster_partition` assignments.
3. **Raw train/test exact overlap:** 26 unique exact sequences were found in both the original `train.csv` and official `test.csv`. The audit did not establish how many of these appear in the cluster-aware development-training subset. Do not describe the official test set as untouched until this is resolved.
4. **Manifest column ambiguity:** the generic `partition` and `cluster_partition` columns differ on 17,039 rows. The cluster-aware scripts use `cluster_partition` when present. Downstream tools must preserve that rule.
5. **External generalization:** no fresh external evaluation has been completed.
6. **Quantum claims:** the quantum component is a classical statevector simulator. No quantum hardware or quantum advantage is demonstrated.
7. **Statistical claims:** the small descriptive comparison does not establish statistical significance.

## Recommended reporting language

“On a fixed cluster-aware development/validation split, the hybrid simulator achieved mean validation MAE 0.370510 (sample SD 0.004363) across three seeds, compared with 0.372770 (sample SD 0.002272) for the classical projection comparator. Validation was used for checkpoint selection; cross-partition sequence-similarity hits were observed, and an audit found exact sequence overlap between the original train and official test CSVs. These results are preliminary development evidence and do not establish independent generalization, statistical significance, or quantum advantage.”

## Release boundaries

This release candidate should be understood as source code, documented development metrics, and reproducibility artifacts. It does not claim an untouched official test estimate, external validation, a biological-family holdout, or a quantum-computing advantage. The raw dataset is not redistributed by this repository.

## Next step for stronger evidence

Create a new final holdout by sequence clusters before further model selection; freeze the full protocol using development data only; and evaluate the final holdout once. An external dataset is preferable if it has not influenced development.
