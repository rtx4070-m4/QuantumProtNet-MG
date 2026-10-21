# QuantumProtNet-MG

An experimental protein-stability prediction project comparing a custom neural sequence encoder with a small differentiable statevector-simulator branch against a classical projection control.

## Project status

- Development comparison: completed for three seeds per model.
- Hardware: NVIDIA RTX 3050 Laptop GPU, 4 GB VRAM.
- Quantum component: simulator only; no quantum hardware.
- No quantum-advantage claim is made.

## Key result

The hybrid model achieved lower mean name-derived scaffold-held-out validation MAE than the classical projection control across the three evaluated seeds. See `docs/RESULTS_REPORT.md` for all metrics and limitations.

## Cluster-aware development comparison

A separate cluster-aware validation comparison was completed using three seeds per model. Mean validation MAE was 0.370510 for the hybrid simulator and 0.372770 for the classical projection comparator. These are development/validation results, not an independent final-test estimate.

- Experiment report: `docs/CLUSTER90_EXPERIMENT_REPORT.md`
- Reproducibility guide: `docs/CLUSTER90_REPRODUCIBILITY.md`

The fixed validation set was used for checkpoint selection, and cross-partition sequence-similarity hits were observed. No quantum advantage is established.
## Experimental caution

The validation set was used for checkpoint selection and is not an independent final test set. Candidate scaffold groups are derived from names and metadata, not verified biological families. Per-seed logs preserve metrics; the current scaffold checkpoint files do not preserve a separate checkpoint for every seed.

## Run environment

Use the project's existing Windows virtual environment and PowerShell workflow. Review the data split manifest and scripts before reproducing the experiment.

## Live demo

Try the deployed Streamlit application:

https://quantumprot-net-mg-vf785atzjqj5c7vfbfgr9f.streamlit.app/

## Source code

GitHub repository: https://github.com/rtx4070-m4/QuantumProtNet-MG


