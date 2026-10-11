# Local workflow results — figure bundle

Metrics source: `/home/jovyan/work/circuits/quantum_circuit_qml_fixed/results/runs/run_20261010_191938_e4090006/actions/0001_complete_ideal_research_workflow/metrics.csv`
Dataset: Breast Cancer Wisconsin Diagnostic (scikit-learn built-in).
Split: stratified 60/20/20; standardization and PCA were fitted on training data only.
Quantum feature encoding: four PCA-derived angles represented as Qiskit QuantumCircuit objects and evaluated by Qiskit Statevector.
Qiskit: used for QuantumCircuit construction, exact Statevector simulation, circuit depth/gate counts, and OpenQASM export. The classical kernel SVM is implemented with scikit-learn.
Execution: local Qiskit Statevector only; no cloud jobs, credentials, device noise, or physical-QPU performance are claimed.
Finite-shot chart: binomial sampling of ideal Qiskit-statevector kernel probabilities only; it does not model gate or readout errors.
Genetic-algorithm results use 3 fixed seeds (42, 43, 44); the tables/figures report mean and sample standard deviation across seeds where available.
Reporting note: repeated seeds quantify search variability on one fixed train/validation/test split; they do not quantify variability across independent dataset splits. Avoid claims of superiority without additional robustness checks.

## Generated figures

- `fig1_test_performance.pdf` / `.png` — Figure 1. Held-out test accuracy, balanced accuracy, macro-F1 and malignant-class recall. Error bars show standard deviation across genetic-algorithm seeds where available. The 30-feature classical models are practical reference baselines; comparisons against four-feature quantum models are not feature-matched.
- `fig2_validation_vs_test.pdf` / `.png` — Figure 2. Validation accuracy used for model/circuit selection versus test accuracy on the held-out partition. Error bars show genetic-algorithm standard deviations where available; the diagonal indicates equal validation and test accuracy.
- `fig3_evolution_convergence.pdf` / `.png` — Figure 3. Mean best-candidate validation accuracy per generation across 3 genetic-algorithm seeds; shaded regions show ±1 standard deviation. Validation data is used for selection; it is not the final test result.
- `fig4_circuit_complexity.pdf` / `.png` — Figure 4. Gate count, two-qubit gate count, and estimated logical depth of the selected circuits. These are simulator-level structural estimates, not transpiled device metrics or measured hardware cost.
- `fig5_finite_shot_robustness.pdf` / `.png` — Figure 5. Independent finite-shot sampling repeats for the selected evolved circuits. Points represent repeats and diamonds show mean ± standard deviation; dashed horizontal segments denote ideal validation accuracy. This experiment models sampling variability only, not gate or readout noise.
- `fig6_pca_explained_variance.pdf` / `.png` — Figure 6. Individual and cumulative variance explained by the four principal components used for quantum feature encoding. The first four components explain 79.9% of variance in the training-fitted PCA transformation.
