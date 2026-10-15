# QuantumProtNet-MG: Reproducibility Guide

## 1. Purpose

This guide documents the experimental setup and the information needed to reproduce the reported development comparison between the hybrid simulator model and the classical projection control.

## 2. Repository

GitHub:
https://github.com/rtx4070-m4/QuantumProtNet-MG

Live demo:
https://quantumprot-net-mg-vf785atzjqj5c7vfbfgr9f.streamlit.app/

## 3. Environment

The original development environment used:
- Windows and PowerShell
- Python 3.14.3
- PyTorch 2.11.0+cu128
- NVIDIA RTX 3050 Laptop GPU with 4 GB VRAM
- Streamlit 1.66.0

The deployed demo can run on CPU. GPU availability is not required for inference.

Other environments may require compatible package versions and may produce small numerical differences.

## 4. Dataset

The project uses the public TAPE protein-stability dataset.

The original data files included training, validation, and test partitions. The development comparison used a controlled training subset and a name-derived group-held-out validation partition.

The full raw dataset is not included in this portfolio package. Obtain it from its original public source and verify its provenance before reproducing training.

## 5. Development Split

The controlled training data contained 53,484 rows after the documented overlap-control procedure.

The name-derived scaffold development manifest assigned:
- 42,786 rows to development training.
- 10,698 rows to group-held-out validation.
- 7,720 candidate groups to development training.
- 1,929 candidate groups to validation.

No candidate groups were shared between these two partitions under the implemented naming rules.

Important: these candidate groups are derived from names and metadata. They are not independently verified biological protein families.

## 6. Models

Hybrid model:
- Custom neural sequence encoder.
- Four-qubit differentiable statevector simulation.
- Two trainable circuit layers.
- Regression head.

Classical control:
- Same general sequence-encoding approach.
- Learned classical projection instead of the simulator branch.

No quantum hardware is used. The circuit is simulated using classical PyTorch operations.

## 7. Random Seeds

The three reported seeds were:
- 42
- 123
- 2026

The per-run logs are needed to inspect individual training outcomes. The currently packaged checkpoint for each model corresponds to the latest seed run, 2026, rather than a separate checkpoint for every seed.

## 8. Reported Validation Results

| Metric | Hybrid | Classical control |
|---|---:|---:|
| Mean validation MAE | 0.33546 | 0.33973 |
| Mean validation RMSE | 0.43663 | 0.44110 |
| Mean validation Pearson correlation | 0.64631 | 0.63747 |

The hybrid model's mean validation MAE was approximately 1.26% lower than the classical control.

These results are descriptive. They do not establish statistical significance or quantum advantage.

## 9. Reproduction Checklist

- [ ] Obtain and verify the original dataset.
- [ ] Record Python, PyTorch, CUDA, and GPU versions.
- [ ] Inspect the split manifest and grouping implementation.
- [ ] Verify there are no shared candidate groups between development training and validation.
- [ ] Run the hybrid model for all three seeds.
- [ ] Run the classical control for all three seeds with matching data and settings.
- [ ] Preserve a separate checkpoint, log, and metrics file for every run.
- [ ] Calculate metrics on the same validation rows.
- [ ] Report all seeds, not only the best run.
- [ ] Keep a final independent test set separate from model selection.
- [ ] Document deviations from the original environment.

## 10. Limitations and Reporting

The validation partition was used for checkpoint selection. It is not an independent final evaluation.

The official test set has previously been inspected during development and should not be presented as a pristine untouched test set.

A future independent evaluation should define grouping rules before training, verify biological relatedness where possible, and avoid using the final test set for model selection.

The model output is a regression score on the dataset's label scale. It is not a laboratory measurement.

## 11. Scientific Claims

This project demonstrates an experimental hybrid architecture and a classical control comparison.

It does not currently demonstrate:
- Quantum computational advantage.
- Statistically significant superiority.
- Reliable predictions for arbitrary unseen proteins.
- Experimental validation of predicted stability.

