# QuantumProtNet-MG: Scaffold-Held-Out Development Results

## Experimental setup

- Dataset: public TAPE protein stability dataset.
- Development training rows: 42,786.
- Name-derived candidate-scaffold-held-out validation rows: 10,698.
- Shared candidate groups between partitions: 0 according to the split audit.
- Hybrid model: custom sequence encoder with a four-qubit, two-layer differentiable statevector simulator.
- Control: matching custom sequence encoder with a learned classical projection branch.
- Training seeds: 42, 123, 2026.
- Hardware: NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM).
- Simulator only; no quantum hardware or quantum-advantage claim.

## Per-run validation metrics

| Model | Seed | Best epoch | Validation MAE | Validation RMSE | Pearson r |
|---|---:|---:|---:|---:|---:|
| Classical projection | 42 | 16 | 0.34450 | 0.44454 | 0.63011 |
| Classical projection | 123 | 25 | 0.33629 | 0.43994 | 0.64047 |
| Classical projection | 2026 | 19 | 0.33840 | 0.43882 | 0.64182 |
| Hybrid simulator | 42 | 20 | 0.33628 | 0.43890 | 0.64082 |
| Hybrid simulator | 123 | 13 | 0.33488 | 0.43268 | 0.65489 |
| Hybrid simulator | 2026 | 19 | 0.33520 | 0.43831 | 0.64320 |

## Mean and sample standard deviation across seeds

| Metric | Hybrid simulator | Classical projection |
|---|---:|---:|
| Validation MAE | 0.33546 ± 0.00074 | 0.33973 ± 0.00427 |
| Validation RMSE | 0.43663 ± 0.00343 | 0.44110 ± 0.00303 |
| Pearson correlation | 0.64631 ± 0.00753 | 0.63747 ± 0.00641 |

## Interpretation

The hybrid model's mean validation MAE was 0.33546, compared with 0.33973 for the classical projection control. This corresponds to a 1.26% lower mean validation MAE for the hybrid in this experiment.

The hybrid validation MAE was lower in all three paired seeds. This is an encouraging development result, but the absolute difference is modest and does not establish statistical significance, biological-family generalization, or quantum advantage.

## Limitations and reporting cautions

- Candidate groups were derived from names and sequence metadata; they are not verified biological families.
- The validation partition was used for checkpoint selection, so these metrics are not an independent final-test estimate.
- The official test set was not used for model selection in this scaffold-held-out experiment.
- Only the latest checkpoint/history/metrics file for each model was preserved in the scaffold output directory. The per-seed logs preserve the six runs' reported metrics, but not six separate checkpoints.
- The current results support an experimental comparison only; they do not demonstrate a quantum advantage.

## Reproducibility artifacts

- `results/scaffold_experiment_summary.csv`
- `results/scaffold_experiment_logs/`
- `results/scaffold_final_archive_20261015_154912/`
- `src/train_quantumprot_hybrid_scaffold.py`
- `src/train_classical_projection_scaffold.py`
- `results/ablation_summary/scaffold_heldout_development_split_seed20261015.csv`
