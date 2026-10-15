# QuantumProtNet-MG — Model Card

## 1. Model Overview

QuantumProtNet-MG is an experimental protein-sequence regression project for predicting a stability-related score from amino-acid sequences. It combines a custom neural sequence encoder with a differentiable four-qubit statevector simulator implemented in PyTorch.

The project also includes a classical projection control model to investigate whether the simulator-based branch provides a measurable benefit under the evaluated experimental setup.

## 2. Intended Use

- Educational research in protein sequence modelling.
- Demonstration of differentiable statevector simulation integrated with a neural network.
- Portfolio demonstration of model development, controlled comparisons, and reproducible workflows.

### Out-of-Scope Uses

- Clinical or diagnostic decisions.
- Replacing laboratory protein-stability measurements.
- Making biological claims about an untested protein without independent validation.
- Claiming quantum advantage or superiority over quantum hardware methods.

## 3. Model Architecture

The hybrid model contains:
- A custom neural sequence encoder.
- A learned projection from sequence features to four qubit angles.
- A two-layer trainable circuit simulated using PyTorch statevectors.
- Rotation operations and ring-connected controlled-NOT operations.
- Quantum measurement features combined with classical sequence features.
- A regression head that outputs a scalar score.

The classical control uses the same general sequence-encoding approach with a learned classical projection instead of the simulator branch.

This is a classical computation using a quantum-circuit simulation. No quantum hardware is used.

## 4. Data

The project uses the public TAPE protein-stability dataset.

The source data contains amino-acid sequences, identifiers, parent identifiers, metadata, and regression labels. The packaged inference application accepts sequences of 43–50 residues containing the 20 standard amino-acid letters.

The model's output is on the dataset label scale. Its biological interpretation depends on the source label definition and appropriate experimental validation.

## 5. Evaluation Protocol

A name-derived candidate-scaffold grouping procedure was used to create a development training partition and a group-held-out validation partition.

- Development training rows: 42,786.
- Group-held-out validation rows: 10,698.
- Training candidate groups: 7,720.
- Validation candidate groups: 1,929.
- Shared candidate groups between these partitions: 0.
- Seeds evaluated per model: 3.

The candidate groups were derived from names and metadata. They have not been independently verified as biological families.

The validation set was used for checkpoint selection. These results are not an independent final-test evaluation. The official test set is not reported as a fresh, untouched evaluation in this experiment.

## 6. Validation Results

Mean results across three random seeds:

| Metric | Hybrid model | Classical projection control |
|---|---:|---:|
| Validation MAE | 0.33546 | 0.33973 |
| Validation RMSE | 0.43663 | 0.44110 |
| Validation Pearson correlation | 0.64631 | 0.63747 |

The hybrid model had approximately 1.26% lower mean validation MAE than the classical control.

This is a modest observed difference. It does not establish statistical significance, generalization to independent biological families, or quantum advantage.

## 7. Known Limitations

- Candidate-scaffold groups are name-derived, not independently established biological families.
- Validation results were used for model selection.
- A separate, independently designed final evaluation is still needed.
- Protein sequence length and label distributions differ across the original dataset partitions.
- The inference application does not establish experimental stability.
- The current comparison does not demonstrate quantum computational advantage.
- The hosted application may run on CPU rather than GPU.

## 8. Reproducibility

Repository:
https://github.com/rtx4070-m4/QuantumProtNet-MG

Live demo:
https://quantumprot-net-mg-vf785atzjqj5c7vfbfgr9f.streamlit.app/

The repository contains the packaged inference application, model checkpoint, source scripts, requirements, and results documentation.

## 9. Ethical and Scientific Considerations

This model is an experimental research prototype. Predictions should not be used as substitutes for experimental evidence. Users should report the evaluation split, grouping assumptions, uncertainty, and limitations when presenting results.

## 10. Project Status

Prototype implemented and deployed. Further independent validation and more comprehensive reproducibility checks are required before making strong scientific or biological claims.
