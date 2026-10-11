"""Research experiments, cross-validation, statistical analysis, and figures."""
from __future__ import annotations

# Baselines and repeated cross-validation
import json
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
import quantum_core as common
from quantum_core import _fit_kernel_classifier, _qiskit_imports, build_genome_circuit, build_zz_feature_map_circuit, export_example_qasm, fidelity_kernel, genome_complexity, normalize_genome, random_genome, sample_kernel, sample_symmetric_kernel, states_for_data
from quantum_core import ADDITIONAL_HARD_DIGIT_DATASET_IDS, ALL_RESEARCH_DATASET_IDS, COMPLEXITY_BALANCED_ACCURACY_TOLERANCE, ExperimentCancelled, FINITE_SHOTS, FINITE_SHOT_REPEATS, GA_GENERATIONS, GA_POPULATION_SIZE, N_QUBITS, RANDOM_STATE, RESEARCH_CV_BOOTSTRAPS, RESEARCH_CV_REPEATS, RESEARCH_CV_SPLITS, RESEARCH_DATASET_IDS, RESEARCH_GA_SEED_BASE, RESEARCH_MODEL_NAMES, RESEARCH_PROTOCOL_VERSION, SCRIPT_VERSION, classification_metrics, log_message, safe_error_message, save_json, save_metrics
from quantum_core import load_dataset, load_research_dataset
from optimizer import _search_random_research_genome, _search_research_genome

def run_classical_baselines(run_dir: Path) -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    import warnings
    from sklearn.exceptions import ConvergenceWarning

    data = load_dataset(run_dir)
    y_train, y_val, y_test = data["y_train"], data["y_val"], data["y_test"]
    feature_sets = {
        "pca4": (data["X_train_pca"], data["X_val_pca"], data["X_test_pca"]),
        "original30": (data["X_train_raw"], data["X_val_raw"], data["X_test_raw"]),
    }
    rows = []
    for feature_name, (X_train, X_val, X_test) in feature_sets.items():
        models = {
            f"logistic_regression_{feature_name}": make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight=None, random_state=RANDOM_STATE)),
            f"mlp_{feature_name}": make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=1800, early_stopping=False, n_iter_no_change=40, tol=1e-4, alpha=1e-3, learning_rate_init=1e-3, random_state=RANDOM_STATE)),
        }
        for name, model in models.items():
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                model.fit(X_train, y_train)
            val_pred = model.predict(X_val)
            iteration_value = getattr(model[-1], "n_iter_", None)
            if iteration_value is None:
                first_fit_iterations = None
            else:
                iteration_array = np.asarray(iteration_value)
                first_fit_iterations = int(np.max(iteration_array)) if iteration_array.size else None
            X_fit = np.vstack([X_train, X_val])
            y_fit = np.concatenate([y_train, y_val])
            with warnings.catch_warnings(record=True) as refit_warnings:
                warnings.simplefilter("always", ConvergenceWarning)
                model.fit(X_fit, y_fit)
            test_pred = model.predict(X_test)
            row = {"experiment": name, "feature_view": feature_name, "n_train": len(y_train), "n_validation": len(y_val), "n_test": len(y_test), "test_fit_size": len(y_fit), "train_iterations_before_validation": first_fit_iterations, "convergence_warning": any(issubclass(w.category, ConvergenceWarning) for w in caught + refit_warnings)}
            row.update(classification_metrics(y_val, val_pred, "validation"))
            row.update(classification_metrics(y_test, test_pred, "test"))
            rows.append(row)
            log_message(f"{name}: validation accuracy={row['validation_accuracy']:.4f}, macro-F1={row['validation_f1_macro']:.4f}; test accuracy={row['test_accuracy']:.4f}, macro-F1={row['test_f1_macro']:.4f}" + (" [check convergence warning]" if row["convergence_warning"] else ""))
    save_metrics(run_dir, rows)

def run_qml_baseline(run_dir: Path) -> None:
    from sklearn.svm import SVC
    data = load_dataset(run_dir)
    X_train, y_train, X_val, y_val = data["X_train"], data["y_train"], data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]
    val_pred, _, _ = _fit_kernel_classifier(X_train, y_train, X_val, zz_baseline=True)
    X_fit, y_fit = np.vstack([X_train, X_val]), np.concatenate([y_train, y_val])
    fit_states, test_states = states_for_data(X_fit, zz_baseline=True), states_for_data(X_test, zz_baseline=True)
    model = SVC(kernel="precomputed", C=1.0).fit(fidelity_kernel(fit_states), y_fit)
    test_pred = model.predict(fidelity_kernel(test_states, fit_states))
    row = {"experiment": "qiskit_zz_inspired_feature_map_kernel", "feature_view": "four PCA-derived angles", "n_train": len(y_train), "n_validation": len(y_val), "n_test": len(y_test), "test_fit_size": len(y_fit), "simulator": "Qiskit QuantumCircuit + Statevector, ideal/noiseless"}
    row.update(classification_metrics(y_val, val_pred, "validation"))
    row.update(classification_metrics(y_test, test_pred, "test"))
    save_metrics(run_dir, [row])
    try:
        circuit = build_zz_feature_map_circuit(X_fit[0])
        (run_dir / "circuits").mkdir(exist_ok=True)
        (run_dir / "circuits" / "qiskit_zz_inspired_feature_map.txt").write_text(str(circuit.draw(output="text")), encoding="utf-8")
        _, _, _, qasm2 = _qiskit_imports()
        (run_dir / "circuits" / "qiskit_zz_inspired_feature_map.qasm").write_text(qasm2.dumps(circuit), encoding="utf-8")
    except Exception as exc:
        log_message(f"Could not export baseline circuit drawing: {safe_error_message(exc)}")
    log_message(f"Qiskit ZZ-inspired QML baseline: validation accuracy={row['validation_accuracy']:.4f}, macro-F1={row['validation_f1_macro']:.4f}; test accuracy={row['test_accuracy']:.4f}, macro-F1={row['test_f1_macro']:.4f}")

def run_finite_shot_evaluation(run_dir: Path) -> None:
    from sklearn.svm import SVC

    data = load_dataset(run_dir)
    candidates = [(mode, common.WORKSPACE_DIR / f"best_circuit_{mode}.json") for mode in ("accuracy", "hardware_aware")]
    candidates = [(mode, path) for mode, path in candidates if path.exists()]
    if not candidates:
        raise FileNotFoundError("No evolved circuit found. Run option 5 or 6 first.")

    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    rows = []
    rng = np.random.default_rng(RANDOM_STATE + 700)
    for mode, genome_path in candidates:
        payload = json.loads(genome_path.read_text(encoding="utf-8"))
        genome = normalize_genome(payload["genome"])
        train_states, val_states = states_for_data(X_train, genome=genome), states_for_data(X_val, genome=genome)
        K_train_ideal, K_val_ideal = fidelity_kernel(train_states), fidelity_kernel(val_states, train_states)
        ideal_model = SVC(kernel="precomputed", C=1.0).fit(K_train_ideal, y_train)
        ideal_predictions = ideal_model.predict(K_val_ideal)
        ideal_metrics = classification_metrics(y_val, ideal_predictions, "ideal_validation")

        for repeat in range(1, FINITE_SHOT_REPEATS + 1):
            K_train = sample_symmetric_kernel(K_train_ideal, FINITE_SHOTS, rng)
            K_val = sample_kernel(K_val_ideal, FINITE_SHOTS, rng)
            model = SVC(kernel="precomputed", C=1.0).fit(K_train, y_train)
            predictions = model.predict(K_val)
            row = {
                "experiment": "finite_shot_kernel_estimation", "circuit_mode": mode, "repeat": repeat,
                "shots_per_kernel_entry": FINITE_SHOTS, "n_train": len(y_train), "n_validation": len(y_val),
                "train_kernel_rmse": float(np.sqrt(np.mean((K_train - K_train_ideal) ** 2))),
                "validation_kernel_rmse": float(np.sqrt(np.mean((K_val - K_val_ideal) ** 2))),
                "sampling_model": "independent binomial shot sampling of ideal kernel entries; no gate/readout noise",
                **ideal_metrics, **classification_metrics(y_val, predictions, "finite_shot_validation"),
            }
            row["accuracy_difference"] = row["finite_shot_validation_accuracy"] - row["ideal_validation_accuracy"]
            rows.append(row)
    pd.DataFrame(rows).to_csv(run_dir / "finite_shot_repeats.csv", index=False)
    save_metrics(run_dir, rows)
    frame = pd.DataFrame(rows)
    summary = frame.groupby("circuit_mode").agg(mean_finite_shot_accuracy=("finite_shot_validation_accuracy", "mean"), std_finite_shot_accuracy=("finite_shot_validation_accuracy", "std"), mean_train_kernel_rmse=("train_kernel_rmse", "mean"), mean_validation_kernel_rmse=("validation_kernel_rmse", "mean")).reset_index()
    summary.to_csv(run_dir / "finite_shot_summary.csv", index=False)
    print(summary.to_string(index=False))
    log_message(f"Finite-shot simulations saved: {len(rows)} evaluations across {len(candidates)} evolved circuits. These are not noisy-hardware simulations.")

def run_remote_experiments_disabled(run_dir: Path) -> None:
    """Record that this version runs locally and never submits remote jobs."""
    plan = {
        "status": "disabled_local_only",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "execution_mode": "qiskit_statevector_local",
        "remote_jobs_submitted": 0,
        "message": "This build runs Qiskit circuits locally. It does not request credentials or submit jobs to a remote service.",
    }
    save_json(run_dir / "remote_execution_disabled.json", plan)
    log_message("Remote execution is disabled. No credentials were requested and zero remote jobs were submitted.")
    raise ExperimentCancelled("Remote execution is disabled in this Qiskit-local build.")

def run_local_self_tests(run_dir: Path) -> None:
    """Verify Qiskit gate unitarity, circuit execution, kernel invariants, and shot sampling."""
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator, Statevector
    tests = []
    one_qubit_gates = [("rx", lambda qc: qc.rx(0.37, 0)), ("ry", lambda qc: qc.ry(-0.62, 0)),
                       ("rz", lambda qc: qc.rz(1.11, 0)), ("h", lambda qc: qc.h(0))]
    for name, apply_gate in one_qubit_gates:
        circuit = QuantumCircuit(1)
        apply_gate(circuit)
        matrix = Operator(circuit).data
        tests.append({"test": f"qiskit_{name}_unitary", "passed": bool(np.allclose(matrix.conj().T @ matrix, np.eye(2), atol=1e-12))})
    two_qubit = QuantumCircuit(2); two_qubit.h(0); two_qubit.cx(0, 1)
    two_matrix = Operator(two_qubit).data
    tests.append({"test": "qiskit_cx_circuit_unitary", "passed": bool(np.allclose(two_matrix.conj().T @ two_matrix, np.eye(4), atol=1e-12))})
    samples = np.array([[0.1, -0.4, 0.7, 1.2], [-1.0, 0.2, 0.3, -0.5], [2.0, -1.4, 0.9, 0.1]])
    baseline_states = states_for_data(samples, zz_baseline=True)
    scalar_states = np.asarray([Statevector.from_instruction(build_zz_feature_map_circuit(row)).data for row in samples])
    tests.append({"test": "qiskit_feature_map_returns_normalized_states", "passed": bool(np.allclose(np.sum(np.abs(baseline_states) ** 2, axis=1), 1.0, atol=1e-12))})
    tests.append({"test": "qiskit_batch_wrapper_matches_scalar_statevector", "passed": bool(np.allclose(baseline_states, scalar_states, atol=1e-12))})
    sample_genome = normalize_genome(random_genome(random.Random(RANDOM_STATE), length=6))
    genome_states = states_for_data(samples, genome=sample_genome)
    tests.append({"test": "evolved_qiskit_circuit_returns_normalized_states", "passed": bool(np.allclose(np.sum(np.abs(genome_states) ** 2, axis=1), 1.0, atol=1e-12))})
    tests.append({"test": "fidelity_kernel_symmetric", "passed": bool(np.allclose(fidelity_kernel(baseline_states), fidelity_kernel(baseline_states).T, atol=1e-12))})
    tests.append({"test": "fidelity_kernel_diagonal_one", "passed": bool(np.allclose(np.diag(fidelity_kernel(baseline_states)), 1.0, atol=1e-12))})
    kernel = fidelity_kernel(baseline_states)
    tests.append({"test": "fidelity_kernel_bounded", "passed": bool(np.all((kernel >= -1e-12) & (kernel <= 1.0 + 1e-12)))} )
    sampled = sample_symmetric_kernel(kernel, 512, np.random.default_rng(RANDOM_STATE))
    tests.append({"test": "finite_shot_matrix_symmetric", "passed": bool(np.allclose(sampled, sampled.T, atol=0.0))})
    tests.append({"test": "finite_shot_diagonal_one", "passed": bool(np.allclose(np.diag(sampled), 1.0, atol=0.0))})
    tests.append({"test": "genome_complexity_matches_qiskit_structure", "passed": bool(genome_complexity(sample_genome)["gate_count"] == build_genome_circuit(samples[0], sample_genome).size())})
    tests.append({"test": "circuit_builder_returns_quantum_circuit", "passed": isinstance(build_genome_circuit(samples[0], sample_genome), QuantumCircuit)})
    report = {"timestamp": datetime.now().isoformat(timespec="seconds"), "script_version": SCRIPT_VERSION,
              "tests_passed": sum(item["passed"] for item in tests), "tests_total": len(tests), "tests": tests,
              "remote_jobs_submitted": 0, "simulation_engine": "Qiskit Statevector"}
    save_json(run_dir / "self_test_report.json", report)
    pd.DataFrame(tests).to_csv(run_dir / "self_test_report.csv", index=False)
    for item in tests:
        log_message(f"{'PASS' if item['passed'] else 'FAIL'}: {item['test']}")
    if not all(item["passed"] for item in tests):
        raise RuntimeError("One or more Qiskit self-tests failed. See self_test_report.json.")
    log_message(f"All {len(tests)} Qiskit local simulator self-tests passed. No remote jobs were submitted.")

def _prepare_outer_cv_fold(
    X_all: np.ndarray, y_all: np.ndarray, outer_train_indices: np.ndarray,
    outer_test_indices: np.ndarray, inner_seed: int,
) -> Dict[str, np.ndarray]:
    """Create an inner train/validation split and train-only preprocessing for one outer fold."""
    from sklearn.decomposition import PCA
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler, StandardScaler

    X_outer_train = X_all[outer_train_indices]
    y_outer_train = y_all[outer_train_indices]
    local_train_indices, local_val_indices = train_test_split(
        np.arange(len(outer_train_indices)), test_size=0.25, random_state=inner_seed,
        stratify=y_outer_train,
    )
    X_inner_raw, y_inner = X_outer_train[local_train_indices], y_outer_train[local_train_indices]
    X_val_raw, y_val = X_outer_train[local_val_indices], y_outer_train[local_val_indices]
    raw_scaler = StandardScaler().fit(X_inner_raw)
    X_inner_std = raw_scaler.transform(X_inner_raw)
    pca = PCA(n_components=N_QUBITS, random_state=inner_seed).fit(X_inner_std)
    X_inner_pca = pca.transform(X_inner_std)
    X_val_pca = pca.transform(raw_scaler.transform(X_val_raw))
    X_outer_pca = pca.transform(raw_scaler.transform(X_outer_train))
    X_test_raw = X_all[outer_test_indices]
    X_test_pca = pca.transform(raw_scaler.transform(X_test_raw))
    angle_scaler = MinMaxScaler(feature_range=(-np.pi, np.pi)).fit(X_inner_pca)
    return {
        "X_inner_raw": X_inner_raw, "y_inner": y_inner,
        "X_val_raw": X_val_raw, "y_val": y_val,
        "X_outer_raw": X_outer_train, "y_outer": y_outer_train,
        "X_test_raw": X_test_raw, "y_test": y_all[outer_test_indices],
        "X_inner_pca": X_inner_pca, "X_val_pca": X_val_pca,
        "X_outer_pca": X_outer_pca, "X_test_pca": X_test_pca,
        "X_inner_angles": angle_scaler.transform(X_inner_pca),
        "X_val_angles": angle_scaler.transform(X_val_pca),
        "X_outer_angles": angle_scaler.transform(X_outer_pca),
        "X_test_angles": angle_scaler.transform(X_test_pca),
        "outer_train_indices": np.asarray(outer_train_indices, dtype=int),
        "outer_test_indices": np.asarray(outer_test_indices, dtype=int),
        "pca_explained_variance_ratio": pca.explained_variance_ratio_,
    }

def _metrics_for_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return classification_metrics(np.asarray(y_true), np.asarray(y_pred), "test")

def _fast_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Equivalent binary metrics via counts, used inside bootstrap loops to avoid sklearn overhead."""
    y_true = np.asarray(y_true, dtype=np.int8).ravel()
    y_pred = np.asarray(y_pred, dtype=np.int8).ravel()
    if y_true.shape != y_pred.shape or y_true.size == 0:
        raise ValueError("y_true and y_pred must be non-empty arrays of the same shape.")
    if not np.isin(y_true, (0, 1)).all() or not np.isin(y_pred, (0, 1)).all():
        raise ValueError("Expected binary labels 0 and 1.")
    counts = np.bincount(2 * y_true + y_pred, minlength=4).astype(float)
    c00, c01, c10, c11 = counts.tolist()
    support0, support1 = c00 + c01, c10 + c11
    predicted0, predicted1 = c00 + c10, c01 + c11
    recall0 = c00 / support0 if support0 else 0.0
    recall1 = c11 / support1 if support1 else 0.0
    f1_0_denom = 2 * c00 + c01 + c10
    f1_1_denom = 2 * c11 + c01 + c10
    f1_0 = 2 * c00 / f1_0_denom if f1_0_denom else 0.0
    f1_1 = 2 * c11 / f1_1_denom if f1_1_denom else 0.0
    recalls = ([recall0] if support0 else []) + ([recall1] if support1 else [])
    class0_precision = float(c00 / predicted0 if predicted0 else 0.0)
    return {
        "test_accuracy": float((c00 + c11) / counts.sum()),
        "test_balanced_accuracy": float(np.mean(recalls)),
        "test_f1_macro": float((f1_0 + f1_1) / 2.0),
        "test_class0_precision": class0_precision,
        "test_class0_recall": float(recall0),
        "test_class1_recall": float(recall1),
        "test_malignant_precision": class0_precision,
        "test_malignant_recall": float(recall0),
        "test_benign_recall": float(recall1),
    }

def _wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    """Wilson score interval for a one-prediction-per-sample accuracy estimate."""
    if total <= 0:
        return float("nan"), float("nan")
    p = float(successes) / float(total)
    z2 = z * z
    denom = 1.0 + z2 / total
    centre = (p + z2 / (2.0 * total)) / denom
    half = (z / denom) * np.sqrt((p * (1.0 - p) / total) + z2 / (4.0 * total * total))
    return max(0.0, float(centre - half)), min(1.0, float(centre + half))

def _cluster_bootstrap_intervals(
    model_predictions: pd.DataFrame, repeats: int, n_bootstrap: int, seed: int,
) -> Dict[str, Tuple[float, float, float]]:
    """Fast sample-cluster bootstrap; all repeated predictions for a sample stay together."""
    rng = np.random.default_rng(seed)
    ordered = model_predictions.sort_values(["sample_index", "repeat"])
    sample_ids = np.asarray(sorted(ordered["sample_index"].unique()), dtype=int)
    if len(sample_ids) == 0:
        raise ValueError("Cannot bootstrap an empty prediction table.")
    y_matrix = ordered.drop_duplicates("sample_index").set_index("sample_index").loc[sample_ids, "y_true"].to_numpy(dtype=int)
    pred_pivot = ordered.pivot(index="sample_index", columns="repeat", values="y_pred").reindex(sample_ids)
    y_pred_matrix = pred_pivot.to_numpy(dtype=int)
    if y_pred_matrix.shape[1] != repeats:
        raise ValueError(f"Expected {repeats} prediction repeats per sample; found {y_pred_matrix.shape[1]}.")
    y_matrix = np.repeat(y_matrix[:, None], repeats, axis=1)
    point = _fast_binary_metrics(y_matrix.ravel(), y_pred_matrix.ravel())
    bootstrap_values: Dict[str, List[float]] = {key: [] for key in point}
    for bootstrap_number in range(n_bootstrap):
        chosen = rng.integers(0, len(sample_ids), size=len(sample_ids))
        values = _fast_binary_metrics(y_matrix[chosen].ravel(), y_pred_matrix[chosen].ravel())
        for key, value in values.items():
            bootstrap_values[key].append(float(value))
    return {
        key: (float(point[key]), float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5)))
        for key, values in bootstrap_values.items()
    }

def _paired_cluster_bootstrap_differences(
    predictions: pd.DataFrame, baseline: str, n_bootstrap: int, seed: int,
    models_to_compare: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Paired sample-cluster bootstrap (model minus baseline), optionally model-filtered."""
    rng = np.random.default_rng(seed)
    sample_ids = np.asarray(sorted(predictions["sample_index"].unique()), dtype=int)
    available_models = set(predictions["model"].astype(str))
    if models_to_compare is None:
        models = sorted(available_models - {baseline})
    else:
        models = [str(model) for model in models_to_compare if str(model) in available_models and str(model) != baseline]
    if baseline not in set(predictions["model"].astype(str)):
        raise ValueError(f"Paired-bootstrap baseline {baseline!r} is absent from predictions.")
    metric_keys = ("test_accuracy", "test_balanced_accuracy", "test_f1_macro", "test_class0_recall")

    def matrices_for(model_name: str) -> Tuple[np.ndarray, np.ndarray]:
        frame = predictions[predictions["model"].astype(str) == model_name].sort_values(["sample_index", "repeat"])
        y_true = frame.drop_duplicates("sample_index").set_index("sample_index").loc[sample_ids, "y_true"].to_numpy(dtype=int)
        y_pred = frame.pivot(index="sample_index", columns="repeat", values="y_pred").reindex(sample_ids).to_numpy(dtype=int)
        return np.repeat(y_true[:, None], y_pred.shape[1], axis=1), y_pred

    y_base, pred_base = matrices_for(baseline)
    rows = []
    for model in models:
        y_model, pred_model = matrices_for(model)
        # Prediction rows should have aligned labels/sample-repeat structure.
        if y_model.shape != y_base.shape or not np.array_equal(y_model, y_base):
            raise ValueError(f"Prediction structure does not align for paired bootstrap: {model}.")
        base_point = _fast_binary_metrics(y_base.ravel(), pred_base.ravel())
        model_point = _fast_binary_metrics(y_model.ravel(), pred_model.ravel())
        boot_diff: Dict[str, List[float]] = {key: [] for key in metric_keys}
        for _ in range(n_bootstrap):
            chosen = rng.integers(0, len(sample_ids), size=len(sample_ids))
            y_boot = y_base[chosen].ravel()
            b_values = _fast_binary_metrics(y_boot, pred_base[chosen].ravel())
            m_values = _fast_binary_metrics(y_boot, pred_model[chosen].ravel())
            for key in metric_keys:
                boot_diff[key].append(float(m_values[key] - b_values[key]))
        row: Dict[str, Any] = {"model": model, "baseline": baseline}
        for key in metric_keys:
            row[f"difference_{key}"] = float(model_point[key] - base_point[key])
            row[f"difference_{key}_ci95_low"] = float(np.percentile(boot_diff[key], 2.5))
            row[f"difference_{key}_ci95_high"] = float(np.percentile(boot_diff[key], 97.5))
        rows.append(row)
    return pd.DataFrame(rows)

def _holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    """Holm step-down family-wise-error adjustment; preserves NaN entries."""
    raw = np.asarray(list(p_values), dtype=float)
    adjusted = np.full(raw.shape, np.nan, dtype=float)
    valid_indices = np.flatnonzero(np.isfinite(raw))
    if not len(valid_indices):
        return adjusted
    valid_p = np.clip(raw[valid_indices], 0.0, 1.0)
    order = np.argsort(valid_p)
    sorted_p = valid_p[order]
    m = len(sorted_p)
    sorted_adjusted = np.maximum.accumulate((m - np.arange(m, dtype=float)) * sorted_p)
    sorted_adjusted = np.clip(sorted_adjusted, 0.0, 1.0)
    adjusted_valid = np.empty(m, dtype=float)
    adjusted_valid[order] = sorted_adjusted
    adjusted[valid_indices] = adjusted_valid
    return adjusted

def _paired_sign_flip_p_value(differences: np.ndarray, seed: int = 12345) -> Tuple[float, str]:
    """Two-sided paired sign-flip test for a mean difference; intended as exploratory at fold level."""
    differences = np.asarray(differences, dtype=float).ravel()
    differences = differences[np.isfinite(differences)]
    n = len(differences)
    if n < 2:
        return float("nan"), "insufficient_pairs"
    observed = abs(float(np.mean(differences)))
    if n <= 18:
        # Exact enumeration of all sign assignments. With 15 outer folds this is 32768 tests.
        all_codes = np.arange(1 << n, dtype=np.uint32)[:, None]
        bit_positions = np.arange(n, dtype=np.uint32)[None, :]
        signs = 1.0 - 2.0 * ((all_codes >> bit_positions) & 1).astype(float)
        null_stats = (signs @ differences) / n
        p = float(np.mean(np.abs(null_stats) >= observed - 1e-12))
        return p, f"exact_two_sided_sign_flip_{1 << n}_sign_patterns"
    rng = np.random.default_rng(seed)
    n_perm = 50000
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, n))
    null_stats = (signs @ differences) / n
    extreme = int(np.count_nonzero(np.abs(null_stats) >= observed - 1e-12))
    return float((extreme + 1) / (n_perm + 1)), f"monte_carlo_two_sided_sign_flip_{n_perm}_permutations"

def _paired_fold_complexity_comparisons(
    fold_metrics: pd.DataFrame, n_bootstrap: int = RESEARCH_CV_BOOTSTRAPS, seed: int = 73000,
) -> pd.DataFrame:
    """Paired GA-minus-random fold differences for the pre-specified primary outcome family.

    These fold-level intervals/p-values are labelled exploratory because outer-CV folds
    overlap in training data and are not fully independent experimental units.
    """
    ga = "genetic_algorithm_complexity_aware"
    rs = "random_search_complexity_aware"
    if not {"model", "repeat", "fold"}.issubset(fold_metrics.columns):
        raise ValueError("fold_metrics must contain model, repeat and fold columns.")
    metrics = ("two_qubit_gate_count", "estimated_depth", "gate_count", "test_balanced_accuracy")
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, Any]] = []
    subset = fold_metrics[fold_metrics["model"].astype(str).isin([ga, rs])].copy()
    if subset.empty or set(subset["model"].astype(str)) != {ga, rs}:
        raise ValueError("Both accuracy-constrained GA and matched random-search rows are required.")
    for metric in metrics:
        if metric not in subset.columns:
            continue
        table = subset.pivot_table(index=["repeat", "fold"], columns="model", values=metric, aggfunc="first")
        table = table.dropna(subset=[ga, rs]).sort_index()
        if len(table) < 2:
            continue
        diffs = table[ga].to_numpy(dtype=float) - table[rs].to_numpy(dtype=float)
        point = float(np.mean(diffs))
        boot_index = rng.integers(0, len(diffs), size=(int(n_bootstrap), len(diffs)))
        boot_means = diffs[boot_index].mean(axis=1)
        ci_low, ci_high = float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))
        p_value, p_method = _paired_sign_flip_p_value(diffs, seed + len(rows))
        is_balanced_accuracy = metric == "test_balanced_accuracy"
        rows.append({
            "comparison": f"{ga} minus {rs}",
            "method_a": ga,
            "method_b": rs,
            "outcome": metric,
            "difference_definition": "accuracy-constrained genetic algorithm minus matched accuracy-constrained random search",
            "mean_paired_difference": point,
            "paired_fold_bootstrap_ci95_low": ci_low,
            "paired_fold_bootstrap_ci95_high": ci_high,
            "p_value_sign_flip_two_sided_exploratory": p_value,
            "p_value_method": p_method,
            "n_paired_outer_folds": int(len(diffs)),
            "n_bootstrap": int(n_bootstrap),
            "favors_genetic_algorithm": bool(ci_high < 0) if metric in ("gate_count", "two_qubit_gate_count", "estimated_depth") else bool(ci_low > 0),
            "complexity_reduction_supported_pointwise": bool(ci_high < 0) if metric in ("gate_count", "two_qubit_gate_count", "estimated_depth") else np.nan,
            "balanced_accuracy_noninferiority_margin": 0.02 if is_balanced_accuracy else np.nan,
            "balanced_accuracy_noninferiority_supported_pointwise": bool(ci_low > -0.02) if is_balanced_accuracy else np.nan,
            "uncertainty_caveat": "Fold-level bootstrap and sign-flip results are exploratory; overlapping outer-CV training sets make folds dependent. Do not interpret these as definitive independent-replicate inference.",
        })
    result = pd.DataFrame(rows)
    if not result.empty:
        result["holm_adjusted_p_value_within_dataset_primary_family"] = _holm_adjust(
            result["p_value_sign_flip_two_sided_exploratory"].to_numpy(dtype=float)
        )
        result["multiple_comparison_family"] = "within_dataset_primary_4_outcomes"
    return result

def run_multi_dataset_research(run_dir: Path, dataset_ids: Optional[Sequence[str]] = None) -> None:
    """Run the same leakage-controlled repeated-CV protocol on selected built-in datasets."""
    selected_dataset_ids = tuple(dataset_ids) if dataset_ids is not None else ALL_RESEARCH_DATASET_IDS
    if not selected_dataset_ids:
        raise ValueError("At least one research dataset must be selected.")
    if len(set(selected_dataset_ids)) != len(selected_dataset_ids):
        raise ValueError("Dataset IDs must be unique.")
    unknown = set(selected_dataset_ids) - set(ALL_RESEARCH_DATASET_IDS)
    if unknown:
        raise ValueError(f"Unsupported dataset IDs: {sorted(unknown)}")
    dataset_root = run_dir / "datasets"
    dataset_root.mkdir(parents=True, exist_ok=True)
    result_frames: List[pd.DataFrame] = []
    metadata_rows: List[Dict[str, Any]] = []
    for index, dataset_id in enumerate(selected_dataset_ids, start=1):
        X, y, metadata = load_research_dataset(dataset_id)
        metadata_rows.append(metadata)
        dataset_run_dir = dataset_root / dataset_id
        dataset_run_dir.mkdir(parents=True, exist_ok=True)
        study_dir = dataset_run_dir / "repeated_cv"
        # Checkpoint compatibility is tied to the algorithm/protocol version. Results from
        # the previous penalty-only objective must not be silently reused as constrained runs.
        expected_protocol = {
            "protocol_version": RESEARCH_PROTOCOL_VERSION,
            "dataset_id": dataset_id,
            "cv_splits": RESEARCH_CV_SPLITS,
            "cv_repeats": RESEARCH_CV_REPEATS,
            "ga_population_size": GA_POPULATION_SIZE,
            "ga_generations": GA_GENERATIONS,
            "nominal_search_candidate_budget": GA_POPULATION_SIZE * GA_GENERATIONS,
            "complexity_selection_rule": "minimize two-qubit gates, then estimated depth, then gate count, among evaluated candidates within the fixed inner-validation balanced-accuracy tolerance",
            "matched_random_complexity_baseline": True,
            "random_search_candidate_budget_unit": "unique circuit evaluations matched to paired GA within fold",
            "paired_classical_comparisons": "all original-feature and PCA-4 classical baselines",
            "supplemental_perfect_accuracy_interval": "unique-sample majority vote Wilson score interval",
            "balanced_accuracy_tolerance": COMPLEXITY_BALANCED_ACCURACY_TOLERANCE,
        }
        protocol_path = dataset_run_dir / "research_run_protocol.json"
        existing_protocol = None
        if protocol_path.exists():
            try:
                existing_protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                existing_protocol = None
        has_existing_results = (study_dir / "fold_metrics.csv").exists() or (study_dir / "oof_predictions.csv").exists()
        if has_existing_results and existing_protocol != expected_protocol:
            import shutil
            legacy_name = f"{dataset_id}_legacy_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            legacy_dir = dataset_root / legacy_name
            shutil.move(str(dataset_run_dir), str(legacy_dir))
            log_message(f"Archived incompatible/legacy results at {legacy_dir}; starting a fresh run for the accuracy-constrained protocol.")
            dataset_run_dir.mkdir(parents=True, exist_ok=True)
            study_dir = dataset_run_dir / "repeated_cv"
        save_json(protocol_path, expected_protocol)
        log_message("\n" + "=" * 72)
        log_message(f"MULTI-DATASET STUDY {index}/{len(selected_dataset_ids)}: {metadata['dataset_name']}")
        log_message(f"Samples={len(y)}, features={X.shape[1]}, class counts={metadata['class_counts']}; class 0 is {metadata['class0_name']!r}.")
        existing_summary = study_dir / "repeated_cv_summary.csv"
        existing_figure_manifest = dataset_run_dir / "figures" / "paper" / "repeated_cv" / "figure_manifest.json"
        completed_outer_folds = 0
        if (study_dir / "fold_metrics.csv").exists():
            try:
                saved_folds = pd.read_csv(study_dir / "fold_metrics.csv", usecols=["repeat", "fold"])
                completed_outer_folds = int(saved_folds[["repeat", "fold"]].drop_duplicates().shape[0])
            except (OSError, ValueError, pd.errors.ParserError):
                completed_outer_folds = 0
        saved_circuits_complete = all(
            _saved_fold_circuits_complete(study_dir, repeat_number, fold_number)
            for repeat_number in range(1, RESEARCH_CV_REPEATS + 1)
            for fold_number in range(1, RESEARCH_CV_SPLITS + 1)
        )
        if (completed_outer_folds >= RESEARCH_CV_SPLITS * RESEARCH_CV_REPEATS
                and existing_summary.exists() and existing_figure_manifest.exists()
                and saved_circuits_complete):
            log_message(f"Using complete saved results and all fold-selected circuit genomes for {dataset_id}; no model retraining.")
        else:
            if completed_outer_folds:
                log_message(f"Found {completed_outer_folds}/{RESEARCH_CV_SPLITS * RESEARCH_CV_REPEATS} saved folds for {dataset_id}; resuming from checkpoint.")
            run_repeated_cv_research(dataset_run_dir, dataset_id=dataset_id)
        summary_path = dataset_run_dir / "repeated_cv" / "repeated_cv_summary.csv"
        summary = pd.read_csv(summary_path)
        summary.insert(0, "dataset_id", dataset_id)
        summary.insert(1, "dataset_name", metadata["dataset_name"])
        summary.insert(2, "n_samples", metadata["n_samples"])
        summary.insert(3, "n_features", metadata["n_features"])
        summary.insert(4, "class0_name", metadata["class0_name"])
        summary.insert(5, "class1_name", metadata["class1_name"])
        result_frames.append(summary)

    # Audit every fold-selected genome before claiming the multi-dataset study is complete.
    # There are four selected circuits per outer fold: two GA objectives and two matched
    # random-search controls. This is not the number of individual GA candidates evaluated.
    circuit_inventory_rows: List[Dict[str, Any]] = []
    missing_circuit_files: List[str] = []
    required_modes = (
        "performance", "complexity_aware",
        "random_search_performance", "random_search_complexity_aware",
    )
    for dataset_id in selected_dataset_ids:
        study_dir = dataset_root / dataset_id / "repeated_cv"
        for repeat_number in range(1, RESEARCH_CV_REPEATS + 1):
            for fold_number in range(1, RESEARCH_CV_SPLITS + 1):
                for mode in required_modes:
                    circuit_path = study_dir / "circuits" / f"repeat_{repeat_number}_fold_{fold_number}_{mode}.json"
                    qasm_path = circuit_path.with_suffix(".qasm")
                    payload = None
                    if circuit_path.is_file():
                        try:
                            payload = json.loads(circuit_path.read_text(encoding="utf-8"))
                        except (OSError, json.JSONDecodeError):
                            payload = None
                    genome = payload.get("genome") if isinstance(payload, dict) else None
                    valid_genome = isinstance(genome, list) and bool(genome)
                    if not valid_genome:
                        missing_circuit_files.append(str(circuit_path))
                    selection = payload.get("selection_metrics", {}) if isinstance(payload, dict) else {}
                    if not isinstance(selection, dict):
                        selection = {}
                    circuit_inventory_rows.append({
                        "dataset_id": dataset_id,
                        "repeat": repeat_number,
                        "fold": fold_number,
                        "method": mode,
                        "genome_json": str(circuit_path.resolve()),
                        "genome_json_exists_and_valid": bool(valid_genome),
                        "qasm_path": str(qasm_path.resolve()) if qasm_path.is_file() else "",
                        "qasm_exists": bool(qasm_path.is_file()),
                        "candidate_attempts": selection.get("candidate_attempts", payload.get("candidate_attempts") if isinstance(payload, dict) else None),
                        "unique_candidates_evaluated": selection.get("unique_candidates_evaluated", payload.get("unique_candidates_evaluated") if isinstance(payload, dict) else None),
                        "outer_test_used_during_selection": payload.get("outer_test_used_during_selection") if isinstance(payload, dict) else None,
                    })
    inventory_frame = pd.DataFrame(circuit_inventory_rows)
    inventory_frame.to_csv(run_dir / "cross_dataset_circuit_inventory.csv", index=False)
    expected_selected_circuits = len(selected_dataset_ids) * RESEARCH_CV_REPEATS * RESEARCH_CV_SPLITS * len(required_modes)
    saved_selected_circuits = int(inventory_frame["genome_json_exists_and_valid"].sum()) if not inventory_frame.empty else 0
    save_json(run_dir / "cross_dataset_circuit_inventory.json", {
        "dataset_ids": list(selected_dataset_ids),
        "dataset_count": len(selected_dataset_ids),
        "repeats_per_dataset": RESEARCH_CV_REPEATS,
        "folds_per_repeat": RESEARCH_CV_SPLITS,
        "selected_methods_per_fold": list(required_modes),
        "expected_fold_selected_circuit_count": expected_selected_circuits,
        "valid_saved_fold_selected_circuit_count": saved_selected_circuits,
        "missing_or_invalid_genome_json_count": len(missing_circuit_files),
        "missing_or_invalid_genome_json_paths": missing_circuit_files,
        "definition": "One selected genome per dataset/repeat/fold/method; this does not count every individual circuit evaluated inside the GA/random-search candidate pool.",
    })
    log_message(f"Fold-selected circuit audit: {saved_selected_circuits}/{expected_selected_circuits} valid genome JSON files found.")
    if missing_circuit_files:
        preview = "\n".join(missing_circuit_files[:10])
        raise RuntimeError(
            f"Circuit-saving audit failed: {len(missing_circuit_files)} of {expected_selected_circuits} "
            f"fold-selected genome files are missing or invalid. First missing files:\n{preview}"
        )

    combined = pd.concat(result_frames, ignore_index=True)
    combined_path = run_dir / "cross_dataset_model_summary.csv"
    combined.to_csv(combined_path, index=False)
    pd.DataFrame(metadata_rows).to_csv(run_dir / "cross_dataset_metadata.csv", index=False)
    audit_frames, pair_frames, classical_pair_frames, complexity_pair_frames = [], [], [], []
    for dataset_id in selected_dataset_ids:
        study_dir = dataset_root / dataset_id / "repeated_cv"
        audit_path = study_dir / "accuracy_constraint_audit.csv"
        pair_path = study_dir / "paired_cluster_bootstrap_qml_methods.csv"
        classical_pair_path = study_dir / "paired_cluster_bootstrap_qml_vs_classical.csv"
        complexity_pair_path = study_dir / "paired_fold_complexity_comparisons.csv"
        if audit_path.exists():
            frame = pd.read_csv(audit_path)
            frame["dataset_id"] = dataset_id
            audit_frames.append(frame)
        if pair_path.exists():
            frame = pd.read_csv(pair_path)
            frame["dataset_id"] = dataset_id
            pair_frames.append(frame)
        if classical_pair_path.exists():
            frame = pd.read_csv(classical_pair_path)
            if "dataset_id" not in frame:
                frame["dataset_id"] = dataset_id
            classical_pair_frames.append(frame)
        if complexity_pair_path.exists():
            frame = pd.read_csv(complexity_pair_path)
            if "dataset_id" not in frame:
                frame["dataset_id"] = dataset_id
            complexity_pair_frames.append(frame)
    if audit_frames:
        pd.concat(audit_frames, ignore_index=True).to_csv(run_dir / "cross_dataset_accuracy_constraint_audit.csv", index=False)
    if pair_frames:
        pd.concat(pair_frames, ignore_index=True).to_csv(run_dir / "cross_dataset_paired_qml_comparisons.csv", index=False)
    if classical_pair_frames:
        pd.concat(classical_pair_frames, ignore_index=True).to_csv(run_dir / "cross_dataset_paired_qml_vs_classical.csv", index=False)
    if complexity_pair_frames:
        complexity_combined = pd.concat(complexity_pair_frames, ignore_index=True)
        complexity_combined["holm_adjusted_p_value_global_primary_family"] = _holm_adjust(
            complexity_combined["p_value_sign_flip_two_sided_exploratory"].to_numpy(dtype=float)
        )
        complexity_combined["global_multiple_comparison_family"] = f"{len(selected_dataset_ids)}_datasets_x_4_prespecified_outcomes_holm"
        complexity_combined.to_csv(run_dir / "cross_dataset_paired_complexity_comparisons.csv", index=False)
        (run_dir / "multiple_comparison_plan.md").write_text(
            "# Multiple-comparison plan\n\n"
            f"Primary family: the paired comparison of accuracy-constrained genetic algorithm minus matched accuracy-constrained random search for four outcomes (two-qubit gate count, estimated depth, total gate count, and outer-test balanced accuracy) on each of the {len(selected_dataset_ids)} selected datasets. Holm correction is applied across the full {len(selected_dataset_ids) * 4}-test family in cross_dataset_paired_complexity_comparisons.csv.\n\n"
            "Pointwise paired-fold bootstrap intervals and sign-flip p-values are exploratory because repeated-CV outer folds overlap in training data and are dependent. Balanced-accuracy non-inferiority is descriptively flagged when the lower pointwise 95% fold-bootstrap interval exceeds -0.02; this is not definitive confirmatory inference. Other model/metric comparisons are secondary and exploratory, unless a separate correction plan is prespecified.\n",
            encoding="utf-8",
        )
    save_json(run_dir / "cross_dataset_study_protocol.json", {
        "datasets": metadata_rows,
        "outer_cv": f"Repeated stratified {RESEARCH_CV_SPLITS}-fold CV x {RESEARCH_CV_REPEATS} repeats per dataset",
        "preprocessing": "Scaler, PCA and angle scaling fitted on inner training partitions only, separately per outer fold",
        "primary_metrics": ["pooled_test_balanced_accuracy", "pooled_test_f1_macro", "pooled_test_class0_recall"],
        "primary_method_comparison": "accuracy-constrained genetic algorithm minus matched accuracy-constrained random search",
        "primary_complexity_outcomes": ["two_qubit_gate_count", "estimated_depth", "gate_count", "test_balanced_accuracy"],
        "multiplicity_correction": f"Holm adjustment across {len(selected_dataset_ids) * 4} prespecified dataset-by-outcome tests; other comparisons exploratory.",
        "protocol_version": RESEARCH_PROTOCOL_VERSION,
        "complexity_selection_constraint": {
            "metric": "inner-validation balanced accuracy",
            "absolute_tolerance_below_best_candidate": COMPLEXITY_BALANCED_ACCURACY_TOLERANCE,
            "selection_priority": ["minimize two-qubit gates", "minimize estimated depth", "minimize total gates", "maximize validation balanced accuracy", "maximize validation macro-F1"],
        },
        "search_budget": {"population_size": GA_POPULATION_SIZE, "generations": GA_GENERATIONS, "nominal_candidate_attempts_per_ga_search": GA_POPULATION_SIZE * GA_GENERATIONS, "random_search_budget_matching": "unique candidate circuit evaluations matched exactly to corresponding GA count where candidate space allows; attempt counts are separately recorded"},
        "paired_comparisons": "QML methods compared with all eight fixed classical baselines using paired sample-cluster bootstrap confidence intervals for accuracy, balanced accuracy, macro-F1 and class-0 recall.",
        "supplemental_accuracy_uncertainty": "Wilson score intervals for one majority-vote prediction per unique sample are reported separately from pooled repeated-CV sample-cluster bootstrap intervals.",
        "runtime_reporting": "Wall-clock seconds per search/method on the executing machine; not a quantum hardware timing estimate.",
        "qiskit_local_only": True, "remote_jobs_submitted": 0,
        "interpretation_note": f"These are {len(selected_dataset_ids)} binary benchmark tasks derived from built-in scikit-learn datasets. Digit and Iris tasks are restricted to stated class pairs; wine uses class 0 versus classes 1/2. Dataset results must be interpreted separately, not pooled as if observations were interchangeable.",
    })
    make_cross_dataset_summary_figure(combined, run_dir)
    (run_dir / "cross_dataset_analysis_notes.md").write_text(
        "# Cross-dataset analysis notes\n\n"
        "The principal method comparison is between the complexity-aware genetic algorithm and a random-search control using the same inner-validation balanced-accuracy tolerance and complexity tie-breaks. Within each outer fold, the random control targets the number of unique circuits evaluated by the corresponding GA; actual candidate attempts, unique-circuit counts and search wall-clock time are reported separately.\n\n"
        "The paired QML-versus-classical file reports model-minus-baseline differences and sample-cluster bootstrap intervals for accuracy, balanced accuracy, macro-F1 and class-0 recall. Intervals crossing zero do not establish a clear difference at the nominal 95% level. These are secondary/exploratory comparisons.\n\n"
        f"The primary paired complexity file compares the accuracy-constrained genetic algorithm against matched accuracy-constrained random search on two-qubit gate count, estimated depth, total gate count and outer-test balanced accuracy. Holm correction is applied across {len(selected_dataset_ids) * 4} selected-dataset-by-outcome tests. Fold-level intervals and sign-flip p-values are exploratory because repeated-CV folds overlap in training data; balanced-accuracy non-inferiority is flagged descriptively at a 2 percentage-point margin and is not definitive confirmatory inference.\n\n"
        "Pooled repeated-CV sample-cluster bootstrap intervals retain all predictions for each sample together. The separate Wilson interval is computed from one majority-vote prediction per unique sample and is a supplementary estimand; it does not represent the same quantity as pooled repeated-CV accuracy.\n\n"
        f"All {len(selected_dataset_ids)} tasks are binary subtasks of built-in scikit-learn datasets. No remote quantum jobs, physical QPU measurements, hardware-noise simulations, or transpiled hardware-cost estimates are included. Do not claim quantum advantage.\n",
        encoding="utf-8",
    )
    log_message(f"Multi-dataset study complete. Combined model summary: {combined_path}")
    log_message(f"Cross-dataset figure directory: {run_dir / 'figures' / 'paper' / 'cross_dataset'}")

def make_cross_dataset_summary_figure(combined: pd.DataFrame, run_dir: Path) -> None:
    """Create a compact cross-dataset primary comparison plot from already-saved per-dataset summaries."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    selected_models = [
        "logistic_regression_original30", "rbf_svm_original30",
        "genetic_algorithm_performance", "genetic_algorithm_complexity_aware",
        "random_search_performance", "random_search_complexity_aware", "qiskit_zz_inspired_feature_map_kernel",
    ]
    model_labels = {
        "logistic_regression_original30": "Logistic regression (original)",
        "rbf_svm_original30": "RBF-SVM (original)",
        "genetic_algorithm_performance": "Evolved QML (performance)",
        "genetic_algorithm_complexity_aware": "Evolved QML (accuracy-constrained complexity)",
        "random_search_performance": "Random-search QML (performance)",
        "random_search_complexity_aware": "Random-search QML (constrained complexity)",
        "qiskit_zz_inspired_feature_map_kernel": "Standard quantum kernel",
    }
    frame = combined[combined["experiment"].isin(selected_models)].copy()
    datasets = list(dict.fromkeys(frame["dataset_id"].astype(str)))
    models = [name for name in selected_models if name in set(frame["experiment"].astype(str))]
    width = 0.8 / max(1, len(models))
    fig, ax = plt.subplots(figsize=(12.0, 5.6))
    x = np.arange(len(datasets), dtype=float)
    for model_index, model in enumerate(models):
        values, lower, upper = [], [], []
        for dataset_id in datasets:
            row = frame[(frame["dataset_id"] == dataset_id) & (frame["experiment"] == model)]
            if row.empty:
                values.append(np.nan); lower.append(np.nan); upper.append(np.nan)
            else:
                r = row.iloc[0]
                val = float(r["pooled_test_balanced_accuracy"])
                values.append(val)
                lower.append(max(0.0, val - float(r["test_balanced_accuracy_ci95_low"])))
                upper.append(max(0.0, float(r["test_balanced_accuracy_ci95_high"]) - val))
        offset = (model_index - (len(models) - 1) / 2.0) * width
        ax.bar(x + offset, values, width=width, label=model_labels.get(model, model), alpha=0.84)
        ax.errorbar(x + offset, values, yerr=np.asarray([lower, upper]), fmt="none", ecolor="#222222", capsize=2, linewidth=0.7)
    label_map = {str(row.dataset_id): str(row.dataset_name) for row in combined[["dataset_id", "dataset_name"]].drop_duplicates().itertuples(index=False)}
    ax.set_xticks(x, [label_map.get(item, item) for item in datasets], rotation=12, ha="right")
    ax.set_ylim(0.0, 1.04)
    ax.set_ylabel("Balanced accuracy (outer-CV OOF predictions)")
    ax.set_title("Cross-dataset comparison of classical and quantum-kernel methods")
    ax.grid(axis="y", alpha=0.2); ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=2, fontsize=8)
    fig.tight_layout()
    out_dir = run_dir / "figures" / "paper" / "cross_dataset"
    out_dir.mkdir(parents=True, exist_ok=True)
    _save_paper_figure(fig, out_dir, "cross_dataset_primary_comparison")
    plt.close(fig)
    (out_dir / "cross_dataset_notes.md").write_text(
        "Cross-dataset balanced-accuracy comparison. Error bars are per-dataset sample-cluster bootstrap 95% confidence intervals. "
        "The constrained random-search baseline uses the same balanced-accuracy tolerance and complexity tie-breaks as the complexity-aware GA and is matched by unique candidate evaluations within fold. "
        f"The {len(datasets)} tasks are not pooled into a single score because they have different samples, features and target definitions. "
        "All tasks use the same four-qubit local Qiskit statevector simulator; no QPU or hardware-noise results are included.\n",
        encoding="utf-8",
    )

def _saved_fold_circuits_complete(study_dir: Path, repeat_number: int, fold_number: int) -> bool:
    """Return True only when all four selected fold-genome JSONs can be read."""
    modes = (
        "performance", "complexity_aware",
        "random_search_performance", "random_search_complexity_aware",
    )
    for mode in modes:
        path = study_dir / "circuits" / f"repeat_{repeat_number}_fold_{fold_number}_{mode}.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        genome = payload.get("genome") if isinstance(payload, dict) else None
        if not isinstance(genome, list) or not genome:
            return False
    return True


def run_repeated_cv_research(run_dir: Path, dataset_id: str = "breast_cancer") -> None:
    """Repeated outer stratified 5-fold CV; every outer test fold is held out from selection."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    import warnings
    from sklearn.exceptions import ConvergenceWarning

    _qiskit_imports()
    study_dir = run_dir / "repeated_cv"
    study_dir.mkdir(parents=True, exist_ok=True)
    X_all, y_all, dataset_meta = load_research_dataset(dataset_id)
    split_count = RESEARCH_CV_SPLITS * RESEARCH_CV_REPEATS
    splitter = RepeatedStratifiedKFold(n_splits=RESEARCH_CV_SPLITS, n_repeats=RESEARCH_CV_REPEATS, random_state=RANDOM_STATE)
    fold_rows: List[Dict[str, Any]] = []
    prediction_rows: List[Dict[str, Any]] = []
    history_rows: List[Dict[str, Any]] = []
    pca_variance_rows: List[Dict[str, Any]] = []
    fold_path, prediction_path = study_dir / "fold_metrics.csv", study_dir / "oof_predictions.csv"
    if fold_path.exists() and prediction_path.exists():
        try:
            fold_rows = pd.read_csv(fold_path).to_dict("records")
            prediction_rows = pd.read_csv(prediction_path).to_dict("records")
            history_path, pca_path = study_dir / "evolution_history.csv", study_dir / "fold_pca_variance.csv"
            if history_path.exists():
                history_rows = pd.read_csv(history_path).to_dict("records")
            if pca_path.exists():
                pca_variance_rows = pd.read_csv(pca_path).to_dict("records")
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            log_message(f"Checkpoint read warning for {dataset_id}; starting a fresh CV run: {safe_error_message(exc)}")
            fold_rows, prediction_rows, history_rows, pca_variance_rows = [], [], [], []
    metric_pairs = {(int(row["repeat"]), int(row["fold"])) for row in fold_rows if "repeat" in row and "fold" in row}
    completed_pairs = {
        pair for pair in metric_pairs
        if _saved_fold_circuits_complete(study_dir, pair[0], pair[1])
    }
    incomplete_pairs = metric_pairs - completed_pairs
    if incomplete_pairs:
        # If checkpoint metrics survived but one or more selected genomes did not, rerun
        # those folds cleanly rather than skipping them or duplicating stale metrics.
        fold_rows = [row for row in fold_rows if (int(row.get("repeat", -1)), int(row.get("fold", -1))) not in incomplete_pairs]
        prediction_rows = [row for row in prediction_rows if (int(row.get("repeat", -1)), int(row.get("fold", -1))) not in incomplete_pairs]
        history_rows = [row for row in history_rows if (int(row.get("repeat", -1)), int(row.get("fold", -1))) not in incomplete_pairs]
        pca_variance_rows = [row for row in pca_variance_rows if (int(row.get("repeat", -1)), int(row.get("fold", -1))) not in incomplete_pairs]
        log_message(
            f"Found {len(incomplete_pairs)} fold checkpoints for {dataset_id} with missing/invalid genome JSONs; "
            "those folds will be recomputed and checkpoint rows replaced."
        )
    if completed_pairs:
        log_message(f"Loaded {len(completed_pairs)}/{RESEARCH_CV_SPLITS * RESEARCH_CV_REPEATS} complete fold checkpoints with circuit genomes for {dataset_id}.")
    save_json(study_dir / "study_protocol.json", {
        **dataset_meta,
        "dataset": dataset_meta["dataset_name"],
        "dataset_id": dataset_id,
        "outer_cv": f"RepeatedStratifiedKFold({RESEARCH_CV_SPLITS} folds x {RESEARCH_CV_REPEATS} repeats)",
        "inner_split": "75/25 stratified split of each outer-training fold; approximately 60/20/20 overall",
        "preprocessing": "StandardScaler, PCA, and angle scaler fitted only on each fold's inner training partition; outer test data never used for preprocessing or selection",
        "evolution_selection_score": "0.5 * validation balanced accuracy + 0.5 * validation macro-F1",
        "random_search_ablation": "Two random-search baselines: a performance-selected baseline matched to the GA-performance unique-circuit count, and an accuracy-constrained complexity baseline using the same balanced-accuracy tolerance and complexity tie-breaks as the complexity-aware GA, matched to its unique-circuit count. Actual attempts, unique circuits and runtime are reported separately.",
        "complexity_aware_objective": (
            f"Search guided by the pre-specified logical-complexity penalty; final candidate selected by minimizing "
            f"two-qubit gates, then depth, then total gates among evaluated candidates whose inner-validation "
            f"balanced accuracy is within {COMPLEXITY_BALANCED_ACCURACY_TOLERANCE:.3f} absolute of the best candidate."
        ),
        "balanced_accuracy_tolerance_absolute": COMPLEXITY_BALANCED_ACCURACY_TOLERANCE,
        "protocol_version": RESEARCH_PROTOCOL_VERSION,
        "candidate_budget_nominal": GA_POPULATION_SIZE * GA_GENERATIONS,
        "candidate_budget_matching": "Random baselines target the exact number of unique candidate circuits evaluated by the corresponding GA within the same outer fold. Duplicate candidate attempts are skipped; actual attempts, unique candidates and elapsed seconds are reported separately.",
        "perfect_score_uncertainty": "In addition to pooled repeated-CV sample-cluster bootstrap intervals, report a supplemental Wilson score interval for accuracy from one majority-vote prediction per unique sample across repeats. These intervals have different estimands and must not be conflated.",
        "paired_classical_comparisons": "Paired sample-cluster bootstrap differences for each QML method versus every original-feature and PCA-four-feature classical baseline; intervals reported for accuracy, balanced accuracy, macro-F1 and class-0 recall.",
        "runtime_measurements": "Search wall-clock seconds, actual candidate attempts and unique-circuit counts per fold/method; timing is machine- and load-dependent.",
        "genetic_algorithm_population": GA_POPULATION_SIZE, "genetic_algorithm_generations": GA_GENERATIONS,
        "genetic_algorithm_runs_per_fold_and_method": 1,
        "qiskit_local_only": True, "remote_jobs_submitted": 0,
        "random_state_outer_cv": RANDOM_STATE,
        "outer_fold_count": split_count,
        "note": f"Binary benchmark task: {dataset_meta['task_description']}. Results are internal repeated-CV estimates, not independent external validation. This is a benchmark study, not a clinical system.",
    })
    log_message(f"Repeated-CV research study: {RESEARCH_CV_SPLITS}-fold x {RESEARCH_CV_REPEATS} repeats = {split_count} outer folds.")
    log_message("Within each fold, circuit selection uses only an inner validation partition; each outer test fold remains held out.")

    for split_number, (outer_train_idx, outer_test_idx) in enumerate(splitter.split(X_all, y_all)):
        repeat_index = split_number // RESEARCH_CV_SPLITS
        fold_index = split_number % RESEARCH_CV_SPLITS
        repeat_number, fold_number = repeat_index + 1, fold_index + 1
        if (repeat_number, fold_number) in completed_pairs:
            log_message(f"Skipping completed checkpoint: dataset={dataset_id}, repeat={repeat_number}, fold={fold_number}.")
            continue
        inner_seed = RANDOM_STATE + repeat_index * 100 + fold_index
        fold = _prepare_outer_cv_fold(X_all, y_all, outer_train_idx, outer_test_idx, inner_seed)
        y_test = fold["y_test"]
        context = {"repeat": repeat_number, "fold": fold_number, "n_inner_train": len(fold["y_inner"]), "n_validation": len(fold["y_val"]), "n_outer_train": len(fold["y_outer"]), "n_outer_test": len(y_test), "inner_seed": inner_seed}
        variance = np.asarray(fold["pca_explained_variance_ratio"], dtype=float)
        variance_row: Dict[str, Any] = {"repeat": repeat_number, "fold": fold_number, "inner_training_samples": len(fold["y_inner"]), "cumulative_explained_variance": float(variance.sum())}
        variance_row.update({f"pc{index + 1}_explained_variance": float(value) for index, value in enumerate(variance)})
        pca_variance_rows.append(variance_row)
        log_message(f"CV repeat {repeat_number}/{RESEARCH_CV_REPEATS}, fold {fold_number}/{RESEARCH_CV_SPLITS}: inner train={context['n_inner_train']}, validation={context['n_validation']}, outer train={context['n_outer_train']}, outer test={context['n_outer_test']}.")

        feature_views = {
            "pca4": (fold["X_inner_pca"], fold["X_val_pca"], fold["X_outer_pca"], fold["X_test_pca"]),
            "original30": (fold["X_inner_raw"], fold["X_val_raw"], fold["X_outer_raw"], fold["X_test_raw"]),
        }
        for feature_name, (X_inner, X_val, X_outer, X_test) in feature_views.items():
            models = {
                f"logistic_regression_{feature_name}": make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000, class_weight=None, random_state=inner_seed)),
                f"mlp_{feature_name}": make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=1800, early_stopping=False, n_iter_no_change=40, tol=1e-4, alpha=1e-3, learning_rate_init=1e-3, random_state=inner_seed)),
                # Fixed, pre-specified classical baselines; no outer-fold tuning.
                f"rbf_svm_{feature_name}": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, gamma="scale")),
                f"random_forest_{feature_name}": RandomForestClassifier(n_estimators=300, max_features="sqrt", min_samples_leaf=1, class_weight=None, random_state=inner_seed, n_jobs=-1),
            }
            for model_name, model in models.items():
                baseline_started = time.perf_counter()
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", ConvergenceWarning)
                    model.fit(X_inner, fold["y_inner"])
                val_prediction = model.predict(X_val)
                with warnings.catch_warnings(record=True) as refit_warnings:
                    warnings.simplefilter("always", ConvergenceWarning)
                    model.fit(X_outer, fold["y_outer"])
                test_prediction = model.predict(X_test)
                row: Dict[str, Any] = {**context, "experiment": model_name, "model": model_name,
                                       "feature_view": "four PCA scores" if feature_name == "pca4" else "original features",
                                       "selection_method": "fixed_model_no_validation_tuning",
                                       "fit_predict_elapsed_seconds": float(time.perf_counter() - baseline_started),
                                       "convergence_warning": any(issubclass(w.category, ConvergenceWarning) for w in list(caught) + list(refit_warnings))}
                row.update(classification_metrics(fold["y_val"], val_prediction, "validation"))
                row.update(classification_metrics(y_test, test_prediction, "test"))
                fold_rows.append(row)
                for sample_id, actual, predicted in zip(outer_test_idx, y_test, test_prediction):
                    prediction_rows.append({"repeat": repeat_number, "fold": fold_number, "sample_index": int(sample_id), "model": model_name, "y_true": int(actual), "y_pred": int(predicted)})

        # Standard Qiskit ZZ-inspired kernel baseline. C=1.0 is fixed before outer testing.
        qml_baseline_started = time.perf_counter()
        inner_states = states_for_data(fold["X_inner_angles"], zz_baseline=True)
        val_states = states_for_data(fold["X_val_angles"], zz_baseline=True)
        inner_kernel = fidelity_kernel(inner_states)
        val_kernel = fidelity_kernel(val_states, inner_states)
        val_classifier = SVC(kernel="precomputed", C=1.0).fit(inner_kernel, fold["y_inner"])
        val_prediction = val_classifier.predict(val_kernel)
        outer_states = states_for_data(fold["X_outer_angles"], zz_baseline=True)
        test_states = states_for_data(fold["X_test_angles"], zz_baseline=True)
        outer_kernel = fidelity_kernel(outer_states)
        test_kernel = fidelity_kernel(test_states, outer_states)
        final_classifier = SVC(kernel="precomputed", C=1.0).fit(outer_kernel, fold["y_outer"])
        test_prediction = final_classifier.predict(test_kernel)
        model_name = "qiskit_zz_inspired_feature_map_kernel"
        row = {**context, "experiment": model_name, "model": model_name, "feature_view": "four PCA-derived angles", "selection_method": "fixed_kernel_and_C1", "fit_predict_elapsed_seconds": float(time.perf_counter() - qml_baseline_started)}
        row.update(classification_metrics(fold["y_val"], val_prediction, "validation"))
        row.update(classification_metrics(y_test, test_prediction, "test"))
        fold_rows.append(row)
        for sample_id, actual, predicted in zip(outer_test_idx, y_test, test_prediction):
            prediction_rows.append({"repeat": repeat_number, "fold": fold_number, "sample_index": int(sample_id), "model": model_name, "y_true": int(actual), "y_pred": int(predicted)})

        # Run both GAs first, then matched random-search controls. Random controls are
        # allotted exactly the same number of unique candidate evaluations as their
        # corresponding GA in this outer fold.
        strategy_results = []
        ga_details_by_mode: Dict[str, Dict[str, Any]] = {}
        for mode_index, mode in enumerate(("performance", "complexity_aware")):
            search_seed = RESEARCH_GA_SEED_BASE + repeat_index * 10000 + fold_index * 100 + mode_index
            genome, selection_details, generation_history = _search_research_genome(
                fold["X_inner_angles"], fold["y_inner"], fold["X_val_angles"], fold["y_val"], mode,
                search_seed, repeat_number, fold_number,
            )
            model_name = "genetic_algorithm_performance" if mode == "performance" else "genetic_algorithm_complexity_aware"
            selection_details = dict(selection_details)
            ga_details_by_mode[mode] = selection_details
            strategy_results.append((model_name, mode, genome, selection_details, generation_history, search_seed))

        random_control_specs = (
            ("performance", "random_search_performance", "performance", 2),
            ("complexity_aware", "random_search_complexity_aware", "complexity_aware", 3),
        )
        for selection_mode, model_name, matched_ga_mode, mode_index in random_control_specs:
            search_seed = RESEARCH_GA_SEED_BASE + repeat_index * 10000 + fold_index * 100 + mode_index
            matched_unique_budget = int(ga_details_by_mode[matched_ga_mode]["unique_candidates_evaluated"])
            genome, selection_details, generation_history = _search_random_research_genome(
                fold["X_inner_angles"], fold["y_inner"], fold["X_val_angles"], fold["y_val"],
                search_seed, repeat_number, fold_number,
                target_unique_candidates=matched_unique_budget,
                selection_mode=selection_mode,
                matched_to_model=("genetic_algorithm_performance" if matched_ga_mode == "performance" else "genetic_algorithm_complexity_aware"),
            )
            strategy_results.append((model_name, model_name, genome, dict(selection_details), generation_history, search_seed))

        for model_name, mode_tag, genome, selection_details, generation_history, search_seed in strategy_results:
            history_rows.extend(generation_history)
            outer_evaluation_started = time.perf_counter()
            outer_train_states = states_for_data(fold["X_outer_angles"], genome=genome)
            outer_test_states = states_for_data(fold["X_test_angles"], genome=genome)
            evolved_train_kernel = fidelity_kernel(outer_train_states)
            evolved_test_kernel = fidelity_kernel(outer_test_states, outer_train_states)
            evolved_classifier = SVC(kernel="precomputed", C=1.0).fit(evolved_train_kernel, fold["y_outer"])
            test_prediction = evolved_classifier.predict(evolved_test_kernel)
            complexity = genome_complexity(genome)
            row = {
                **context, "experiment": model_name, "model": model_name,
                "feature_view": "four PCA-derived angles",
                "selection_method": str(selection_details.get("selection_rule", "inner_validation_only")),
                "search_seed": int(search_seed), "selected_validation_fitness": float(selection_details["fitness"]),
                "validation_selection_score": float(selection_details["validation_selection_score"]),
                "validation_accuracy": float(selection_details["validation_accuracy"]),
                "validation_balanced_accuracy": float(selection_details["validation_balanced_accuracy"]),
                "validation_f1_macro": float(selection_details["validation_f1_macro"]),
                "validation_class0_recall": float(selection_details.get("validation_class0_recall", np.nan)),
                "validation_best_balanced_accuracy_seen": float(selection_details.get("best_validation_balanced_accuracy_seen", np.nan)),
                "validation_balanced_accuracy_loss_from_best": float(selection_details.get("validation_balanced_accuracy_loss_from_best", np.nan)),
                "accuracy_constraint_tolerance": float(selection_details.get("accuracy_constraint_tolerance", COMPLEXITY_BALANCED_ACCURACY_TOLERANCE)),
                "accuracy_constraint_applicable": bool(selection_details.get("accuracy_constraint_applicable", False)),
                "accuracy_constraint_met": selection_details.get("accuracy_constraint_met", None),
                "selection_rule": str(selection_details.get("selection_rule", "not_recorded")),
                "candidate_budget_nominal": int(selection_details.get("candidate_budget_nominal", GA_POPULATION_SIZE * GA_GENERATIONS)),
                "candidate_budget_unit": str(selection_details.get("candidate_budget_unit", "nominal_population_slots")),
                "candidate_attempt_budget_nominal": (float(selection_details["candidate_attempt_budget_nominal"]) if selection_details.get("candidate_attempt_budget_nominal") is not None else np.nan),
                "candidate_attempt_safety_cap": (int(selection_details["candidate_attempt_safety_cap"]) if selection_details.get("candidate_attempt_safety_cap") is not None else np.nan),
                "candidate_attempts": int(selection_details.get("candidate_attempts", GA_POPULATION_SIZE * GA_GENERATIONS)),
                "candidate_unique_budget_target": int(selection_details.get("candidate_unique_budget_target", selection_details.get("unique_candidates_evaluated", 0))),
                "matched_to_model": str(selection_details.get("matched_to_model", "")),
                "unique_budget_matched": bool(selection_details.get("unique_budget_matched", True)),
                "unique_candidates_evaluated": int(selection_details.get("unique_candidates_evaluated", 0)),
                "search_elapsed_seconds": float(selection_details.get("search_elapsed_seconds", np.nan)),
                "outer_fit_predict_elapsed_seconds": float(time.perf_counter() - outer_evaluation_started),
                **complexity,
            }
            row.update(classification_metrics(y_test, test_prediction, "test"))
            fold_rows.append(row)
            for sample_id, actual, predicted in zip(outer_test_idx, y_test, test_prediction):
                prediction_rows.append({"repeat": repeat_number, "fold": fold_number, "sample_index": int(sample_id), "model": model_name, "y_true": int(actual), "y_pred": int(predicted)})
            (study_dir / "circuits").mkdir(parents=True, exist_ok=True)
            save_json(study_dir / "circuits" / f"repeat_{repeat_number}_fold_{fold_number}_{mode_tag}.json", {
                "repeat": repeat_number, "fold": fold_number, "mode": mode_tag, "search_seed": int(search_seed),
                "genome": genome, "selection_metrics": selection_details, "complexity": complexity,
                "outer_test_used_during_selection": False,
                "fitness_definition": selection_details.get("selection_rule", "Validation-only candidate selection"),
                "candidate_budget_nominal": selection_details.get("candidate_budget_nominal"),
                "candidate_attempt_budget_nominal": selection_details.get("candidate_attempt_budget_nominal"),
                "candidate_attempt_safety_cap": selection_details.get("candidate_attempt_safety_cap"),
                "candidate_attempts": selection_details.get("candidate_attempts"),
                "candidate_unique_budget_target": selection_details.get("candidate_unique_budget_target"),
                "unique_candidates_evaluated": selection_details.get("unique_candidates_evaluated"),
                "matched_to_model": selection_details.get("matched_to_model"),
                "unique_budget_matched": selection_details.get("unique_budget_matched"),
                "search_elapsed_seconds": selection_details.get("search_elapsed_seconds"),
                "accuracy_constraint_tolerance": selection_details.get("accuracy_constraint_tolerance"),
                "accuracy_constraint_met": selection_details.get("accuracy_constraint_met"),
            })
            try:
                export_example_qasm(study_dir / "circuits" / f"repeat_{repeat_number}_fold_{fold_number}_{mode_tag}.qasm", genome, fold["X_outer_angles"][0])
            except Exception as exc:
                log_message(f"Circuit export warning for repeat {repeat_number}, fold {fold_number}, mode {mode_tag}: {safe_error_message(exc)}")

        # Checkpoint after every fold so long runs leave recoverable outputs on disk.
        pd.DataFrame(fold_rows).to_csv(study_dir / "fold_metrics.csv", index=False)
        pd.DataFrame(prediction_rows).to_csv(study_dir / "oof_predictions.csv", index=False)
        pd.DataFrame(history_rows).to_csv(study_dir / "evolution_history.csv", index=False)
        pd.DataFrame(pca_variance_rows).to_csv(study_dir / "fold_pca_variance.csv", index=False)
        log_message(f"Completed CV repeat {repeat_number}/{RESEARCH_CV_REPEATS}, fold {fold_number}/{RESEARCH_CV_SPLITS}; checkpoint saved.")

    finalize_repeated_cv_outputs(study_dir, run_dir, fold_rows=fold_rows, prediction_rows=prediction_rows, history_rows=history_rows, pca_variance_rows=pca_variance_rows)

def finalize_repeated_cv_outputs(
    study_dir: Path, run_dir: Path,
    fold_rows: Optional[List[Dict[str, Any]]] = None,
    prediction_rows: Optional[List[Dict[str, Any]]] = None,
    history_rows: Optional[List[Dict[str, Any]]] = None,
    pca_variance_rows: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Finalize completed CV checkpoints with vectorized bootstrap; safe to rerun without training."""
    study_dir.mkdir(parents=True, exist_ok=True)
    if fold_rows is None:
        fold_path, pred_path = study_dir / "fold_metrics.csv", study_dir / "oof_predictions.csv"
        if not fold_path.exists() or not pred_path.exists():
            raise FileNotFoundError("Saved CV outputs are incomplete: fold_metrics.csv or oof_predictions.csv is missing.")
        fold_frame = pd.read_csv(fold_path)
        pred_frame = pd.read_csv(pred_path)
        history_path = study_dir / "evolution_history.csv"
        pca_path = study_dir / "fold_pca_variance.csv"
        history_frame = pd.read_csv(history_path) if history_path.exists() else pd.DataFrame()
        pca_variance_frame = pd.read_csv(pca_path) if pca_path.exists() else pd.DataFrame()
    else:
        fold_frame = pd.DataFrame(fold_rows)
        pred_frame = pd.DataFrame(prediction_rows or [])
        history_frame = pd.DataFrame(history_rows or [])
        pca_variance_frame = pd.DataFrame(pca_variance_rows or [])

    # Checkpoint raw files before expensive derived outputs; safe to interrupt after this point.
    fold_frame.to_csv(study_dir / "fold_metrics.csv", index=False)
    pred_frame.to_csv(study_dir / "oof_predictions.csv", index=False)
    history_frame.to_csv(study_dir / "evolution_history.csv", index=False)
    pca_variance_frame.to_csv(study_dir / "fold_pca_variance.csv", index=False)
    log_message(f"Finalization: saved raw checkpoint tables ({len(fold_frame)} fold/model rows; {len(pred_frame)} OOF predictions).")

    pca_columns = [column for column in pca_variance_frame.columns if "explained_variance" in column]
    pca_summary_rows = []
    for column in pca_columns:
        values = pca_variance_frame[column].to_numpy(dtype=float)
        pca_summary_rows.append({"component_or_measure": column, "mean_across_folds": float(values.mean()), "sd_across_folds": float(values.std(ddof=1)) if len(values) > 1 else 0.0, "n_outer_folds": len(values)})
    pd.DataFrame(pca_summary_rows).to_csv(study_dir / "pca_variance_summary.csv", index=False)

    repeat_rows: List[Dict[str, Any]] = []
    for (repeat_number, model_name), group in pred_frame.groupby(["repeat", "model"]):
        metrics = classification_metrics(group["y_true"].to_numpy(dtype=int), group["y_pred"].to_numpy(dtype=int), "test")
        repeat_rows.append({"repeat": int(repeat_number), "experiment": model_name, "model": model_name, "n_oof_predictions": int(len(group)), **metrics})
    repeat_frame = pd.DataFrame(repeat_rows).sort_values(["experiment", "repeat"])
    repeat_frame.to_csv(study_dir / "repeat_level_oof_metrics.csv", index=False)

    summary_rows = []
    model_groups = list(pred_frame.groupby("model"))
    log_message(f"Finalization: computing sample-cluster bootstrap intervals for {len(model_groups)} models ({RESEARCH_CV_BOOTSTRAPS} replicates each).")
    for model_index, (model_name, model_frame) in enumerate(model_groups, start=1):
        log_message(f"Finalization: bootstrap CI {model_index}/{len(model_groups)}: {model_name}.")
        metrics_across_repeats = repeat_frame[repeat_frame["model"] == model_name]
        bootstrap = _cluster_bootstrap_intervals(model_frame, RESEARCH_CV_REPEATS, RESEARCH_CV_BOOTSTRAPS, RANDOM_STATE + 9000)
        row: Dict[str, Any] = {
            "experiment": model_name, "model": model_name, "display_name": RESEARCH_MODEL_NAMES.get(model_name, model_name),
            "outer_folds": RESEARCH_CV_SPLITS * RESEARCH_CV_REPEATS, "cv_repeats": RESEARCH_CV_REPEATS,
            "oof_predictions": int(len(model_frame)), "ci_method": f"{RESEARCH_CV_BOOTSTRAPS}-replicate sample-cluster bootstrap, keeping repeated predictions for each sample together",
        }
        # A complementary, explicitly different estimand: majority vote over each
        # sample's repeated out-of-fold predictions, one vote per unique sample. This
        # avoids a degenerate 100%-100% interval for perfect sample-level results.
        vote_frame = model_frame.sort_values(["sample_index", "repeat"])
        unique_ids = np.asarray(sorted(vote_frame["sample_index"].unique()), dtype=int)
        unique_truth = vote_frame.drop_duplicates("sample_index").set_index("sample_index").loc[unique_ids, "y_true"].to_numpy(dtype=int)
        vote_matrix = vote_frame.pivot(index="sample_index", columns="repeat", values="y_pred").reindex(unique_ids).to_numpy(dtype=int)
        majority_threshold = vote_matrix.shape[1] // 2 + 1
        majority_prediction = (vote_matrix.sum(axis=1) >= majority_threshold).astype(int)
        majority_correct = int(np.sum(majority_prediction == unique_truth))
        wilson_low, wilson_high = _wilson_interval(majority_correct, len(unique_ids))
        row.update({
            "n_unique_samples": int(len(unique_ids)),
            "sample_majority_vote_accuracy": float(majority_correct / len(unique_ids)) if len(unique_ids) else float("nan"),
            "sample_majority_vote_accuracy_wilson95_low": wilson_low,
            "sample_majority_vote_accuracy_wilson95_high": wilson_high,
            "sample_majority_vote_ci_estimand": "one majority-vote prediction per unique sample across repeated out-of-fold predictions; supplemental to pooled repeated-CV bootstrap",
        })
        for metric_name in ("test_accuracy", "test_balanced_accuracy", "test_f1_macro", "test_class0_precision", "test_class0_recall", "test_class1_recall", "test_malignant_precision", "test_malignant_recall", "test_benign_recall"):
            values = metrics_across_repeats[metric_name].to_numpy(dtype=float)
            point, low, high = bootstrap[metric_name]
            row[f"mean_{metric_name}_across_repeats"] = float(np.mean(values))
            row[f"sd_{metric_name}_across_repeats"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            row[f"pooled_{metric_name}"] = point
            row[f"{metric_name}_ci95_low"] = low
            row[f"{metric_name}_ci95_high"] = high
        complexity_rows = fold_frame[fold_frame["model"] == model_name]
        for complexity_metric in (
            "gate_count", "two_qubit_gate_count", "estimated_depth", "selected_validation_fitness",
            "validation_selection_score", "validation_balanced_accuracy_loss_from_best",
            "candidate_budget_nominal", "candidate_attempt_budget_nominal", "candidate_attempts",
            "candidate_unique_budget_target", "unique_candidates_evaluated", "search_elapsed_seconds",
            "outer_fit_predict_elapsed_seconds", "fit_predict_elapsed_seconds",
        ):
            if complexity_metric in complexity_rows and complexity_rows[complexity_metric].notna().any():
                row[f"mean_{complexity_metric}"] = float(complexity_rows[complexity_metric].mean())
                row[f"sd_{complexity_metric}"] = float(complexity_rows[complexity_metric].std(ddof=1)) if complexity_rows[complexity_metric].notna().sum() > 1 else 0.0
        if "accuracy_constraint_met" in complexity_rows:
            constraint_values = complexity_rows["accuracy_constraint_met"].dropna().astype(str).str.lower().map({"true": 1.0, "false": 0.0})
            if not constraint_values.dropna().empty:
                row["accuracy_constraint_pass_rate"] = float(constraint_values.dropna().mean())
                row["accuracy_constraint_folds_evaluated"] = int(constraint_values.notna().sum())
        if "unique_budget_matched" in complexity_rows:
            match_values = complexity_rows["unique_budget_matched"].dropna().astype(str).str.lower().map({"true": 1.0, "false": 0.0})
            if not match_values.dropna().empty:
                row["unique_budget_match_rate"] = float(match_values.dropna().mean())
                row["unique_budget_folds_evaluated"] = int(match_values.notna().sum())
        summary_rows.append(row)
    summary_frame = pd.DataFrame(summary_rows).sort_values("experiment")
    summary_frame.to_csv(study_dir / "repeated_cv_summary.csv", index=False)
    log_message(f"Finalization: summary CSV saved ({len(summary_frame)} models). Computing paired intervals against the original-feature logistic-regression reference.")

    qml_methods = [
        "qiskit_zz_inspired_feature_map_kernel",
        "genetic_algorithm_performance", "genetic_algorithm_complexity_aware",
        "random_search_performance", "random_search_complexity_aware",
    ]
    paired = _paired_cluster_bootstrap_differences(
        pred_frame, "logistic_regression_original30", RESEARCH_CV_BOOTSTRAPS,
        RANDOM_STATE + 9100, models_to_compare=qml_methods,
    )
    paired.to_csv(study_dir / "paired_cluster_bootstrap_vs_logistic30.csv", index=False)

    # Paired QML-versus-classical comparisons across every fixed classical baseline.
    classical_baselines = [
        "logistic_regression_original30", "mlp_original30", "rbf_svm_original30", "random_forest_original30",
        "logistic_regression_pca4", "mlp_pca4", "rbf_svm_pca4", "random_forest_pca4",
    ]
    classical_pair_frames = []
    for baseline_index, baseline_name in enumerate(classical_baselines):
        if baseline_name not in set(pred_frame["model"].astype(str)):
            continue
        pair_frame = _paired_cluster_bootstrap_differences(
            pred_frame, baseline_name, RESEARCH_CV_BOOTSTRAPS,
            RANDOM_STATE + 9300 + baseline_index, models_to_compare=qml_methods,
        )
        if not pair_frame.empty:
            pair_frame.insert(0, "dataset_id", str((json.loads((study_dir / "study_protocol.json").read_text(encoding="utf-8")) if (study_dir / "study_protocol.json").exists() else {}).get("dataset_id", "unknown")))
            pair_frame["baseline_display_name"] = pair_frame["baseline"].map(RESEARCH_MODEL_NAMES)
            pair_frame["model_display_name"] = pair_frame["model"].map(RESEARCH_MODEL_NAMES)
            pair_frame["n_bootstrap"] = int(RESEARCH_CV_BOOTSTRAPS)
            classical_pair_frames.append(pair_frame)
    classical_pairs = pd.concat(classical_pair_frames, ignore_index=True) if classical_pair_frames else pd.DataFrame()
    classical_pairs.to_csv(study_dir / "paired_cluster_bootstrap_qml_vs_classical.csv", index=False)

    # Pre-specified comparisons, including both matched random-search controls.
    qml_pairs = [
        ("genetic_algorithm_performance", "genetic_algorithm_complexity_aware"),
        ("genetic_algorithm_performance", "random_search_performance"),
        ("genetic_algorithm_complexity_aware", "random_search_complexity_aware"),
        ("random_search_performance", "random_search_complexity_aware"),
    ]
    qml_pair_frames = []
    for pair_index, (model_a, model_b) in enumerate(qml_pairs):
        pair_frame = _paired_cluster_bootstrap_differences(
            pred_frame, model_b, RESEARCH_CV_BOOTSTRAPS, RANDOM_STATE + 9200 + pair_index,
            models_to_compare=[model_a],
        )
        if not pair_frame.empty:
            pair_frame.insert(0, "comparison", f"{model_a} minus {model_b}")
            pair_frame["method_a"] = model_a
            pair_frame["method_b"] = model_b
            qml_pair_frames.append(pair_frame)
    qml_pairwise = pd.concat(qml_pair_frames, ignore_index=True) if qml_pair_frames else pd.DataFrame()
    qml_pairwise.to_csv(study_dir / "paired_cluster_bootstrap_qml_methods.csv", index=False)

    # Paired complexity and balanced-accuracy inference for the pre-specified constrained GA vs random search.
    complexity_pairs = _paired_fold_complexity_comparisons(
        fold_frame, n_bootstrap=RESEARCH_CV_BOOTSTRAPS, seed=RANDOM_STATE + 9400,
    )
    if not complexity_pairs.empty:
        complexity_pairs.insert(0, "dataset_id", str((json.loads((study_dir / "study_protocol.json").read_text(encoding="utf-8")) if (study_dir / "study_protocol.json").exists() else {}).get("dataset_id", "unknown")))
    complexity_pairs.to_csv(study_dir / "paired_fold_complexity_comparisons.csv", index=False)

    # Transparent per-fold audit of the pre-specified accuracy-constrained selection rule.
    audit_columns = [
        "repeat", "fold", "search_seed", "model", "validation_best_balanced_accuracy_seen",
        "validation_balanced_accuracy", "validation_balanced_accuracy_loss_from_best",
        "accuracy_constraint_tolerance", "accuracy_constraint_applicable", "accuracy_constraint_met",
        "gate_count", "two_qubit_gate_count", "estimated_depth", "candidate_budget_nominal",
        "unique_candidates_evaluated", "candidate_unique_budget_target", "candidate_attempts",
        "candidate_attempt_budget_nominal", "candidate_attempt_safety_cap", "matched_to_model", "unique_budget_matched",
        "search_elapsed_seconds", "selection_rule",
    ]
    audit = fold_frame[fold_frame.get("model", pd.Series(index=fold_frame.index, dtype=object)).isin(
        ["genetic_algorithm_complexity_aware", "random_search_complexity_aware"]
    )].copy()
    for column in audit_columns:
        if column not in audit:
            audit[column] = np.nan
    audit_dataset_id = str((json.loads((study_dir / "study_protocol.json").read_text(encoding="utf-8")) if (study_dir / "study_protocol.json").exists() else {}).get("dataset_id", "unknown"))
    audit.insert(0, "dataset_id", audit_dataset_id)
    audit[["dataset_id", *audit_columns]].to_csv(study_dir / "accuracy_constraint_audit.csv", index=False)
    log_message(f"Finalization: paired intervals saved (classical reference and QML methods); accuracy-constraint audit saved. Starting figure generation.")
    log_message("REPEATED-CV SUMMARY (primary generalization estimates; pooled out-of-fold predictions)")
    compact_columns = ["display_name", "pooled_test_accuracy", "test_accuracy_ci95_low", "test_accuracy_ci95_high", "pooled_test_balanced_accuracy", "pooled_test_f1_macro", "pooled_test_class0_recall", "mean_gate_count", "mean_two_qubit_gate_count", "mean_estimated_depth"]
    print(summary_frame[[column for column in compact_columns if column in summary_frame.columns]].to_string(index=False))
    log_message(f"Saved out-of-fold predictions and repeated-CV summary under {study_dir}.")
    make_repeated_cv_paper_figures(study_dir, run_dir)

def make_repeated_cv_paper_figures(study_dir: Path, run_dir: Path) -> None:
    """Generate the primary publication figures/tables from repeated outer-CV predictions."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    summary_path = study_dir / "repeated_cv_summary.csv"
    folds_path = study_dir / "fold_metrics.csv"
    preds_path = study_dir / "oof_predictions.csv"
    repeats_path = study_dir / "repeat_level_oof_metrics.csv"
    for required_path in (summary_path, folds_path, preds_path, repeats_path):
        if not required_path.exists():
            raise FileNotFoundError(f"Repeated-CV output is incomplete: {required_path}")
    summary = pd.read_csv(summary_path)
    protocol_path = study_dir / "study_protocol.json"
    dataset_meta = json.loads(protocol_path.read_text(encoding="utf-8")) if protocol_path.exists() else {
        "dataset_name": "Breast Cancer Wisconsin Diagnostic", "class0_name": "Malignant", "class1_name": "Benign"
    }
    dataset_name = str(dataset_meta.get("dataset_name", "Binary classification dataset"))
    class0_name = str(dataset_meta.get("class0_name", "Class 0"))
    class1_name = str(dataset_meta.get("class1_name", "Class 1"))
    folds = pd.read_csv(folds_path)
    predictions = pd.read_csv(preds_path)
    repeat_metrics = pd.read_csv(repeats_path)
    figure_dir = run_dir / "figures" / "paper" / "repeated_cv"
    figure_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 120})
    captions: List[str] = []
    manifest: Dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "analysis": "Repeated stratified 5-fold outer CV x 3 repeats; inner train/validation split for model/circuit selection",
        "no_remote_jobs": True, "figures": [],
    }

    def finish_figure(fig: Any, stem: str, caption: str) -> None:
        log_message(f"Figure export starting: {stem}")
        files = _save_paper_figure(fig, figure_dir, stem)
        captions.append(caption)
        manifest["figures"].append({"stem": stem, "caption": caption, "files": files})
        log_message(f"Figure export saved: {stem} ({len(files)} files)")

    # Figure 1: primary metric with sample-cluster bootstrap confidence intervals.
    ordered = summary.sort_values("pooled_test_balanced_accuracy", ascending=False).reset_index(drop=True)
    labels = [RESEARCH_MODEL_NAMES.get(str(name), str(name)) for name in ordered["experiment"]]
    point = ordered["pooled_test_balanced_accuracy"].to_numpy(dtype=float)
    low = ordered["test_balanced_accuracy_ci95_low"].to_numpy(dtype=float)
    high = ordered["test_balanced_accuracy_ci95_high"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    y_pos = np.arange(len(ordered))
    ax.barh(y_pos, point, color=["#4477AA" if "genetic_algorithm" not in str(name) else "#CC6677" for name in ordered["experiment"]], alpha=0.88)
    ax.errorbar(point, y_pos, xerr=np.vstack([np.maximum(0, point - low), np.maximum(0, high - point)]), fmt="none", ecolor="#222222", capsize=3, linewidth=1)
    ax.set_yticks(y_pos, labels=labels)
    ax.invert_yaxis()
    ax.set_xlim(max(0.0, float(np.min(low)) - 0.05), 1.02)
    ax.set_xlabel("Balanced accuracy on pooled out-of-fold predictions")
    ax.set_title("Repeated stratified cross-validation performance")
    ax.grid(axis="x", alpha=0.2); ax.set_axisbelow(True)
    fig.tight_layout()
    finish_figure(fig, "cv_fig1_balanced_accuracy_ci", "Figure 1. Balanced accuracy measured on pooled outer-fold predictions from repeated stratified five-fold cross-validation. Error bars show 95% sample-cluster bootstrap confidence intervals that keep repeated predictions for each sample together.")

    # Figure 2: primary classification metrics.
    metric_specs = [
        ("pooled_test_balanced_accuracy", "Balanced accuracy", "test_balanced_accuracy_ci95_low", "test_balanced_accuracy_ci95_high"),
        ("pooled_test_f1_macro", "Macro-F1", "test_f1_macro_ci95_low", "test_f1_macro_ci95_high"),
        ("pooled_test_class0_recall", f"Recall ({class0_name})", "test_class0_recall_ci95_low", "test_class0_recall_ci95_high"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 5.2), sharey=True)
    x = np.arange(len(ordered))
    for ax, (metric, title, low_name, high_name) in zip(axes, metric_specs):
        values = ordered[metric].to_numpy(dtype=float)
        low_values = ordered[low_name].to_numpy(dtype=float)
        high_values = ordered[high_name].to_numpy(dtype=float)
        ax.barh(x, values, color="#4477AA", alpha=0.82)
        ax.errorbar(values, x, xerr=np.vstack([np.maximum(0, values-low_values), np.maximum(0, high_values-values)]), fmt="none", ecolor="#222222", capsize=2, linewidth=0.8)
        ax.set_title(title); ax.set_xlim(0, 1.03); ax.grid(axis="x", alpha=0.18); ax.set_axisbelow(True)
        ax.set_xlabel("Score")
    axes[0].set_yticks(x, labels=labels); axes[0].invert_yaxis()
    fig.suptitle(f"Classification quality and target-class recall: {dataset_name}", y=1.02)
    fig.tight_layout()
    finish_figure(fig, "cv_fig2_classification_metrics", f"Figure 2. Balanced accuracy, macro-F1, and recall for class 0 ({class0_name}) on repeated outer-fold predictions for {dataset_name}. Error bars are 95% sample-cluster bootstrap confidence intervals.")

    # Figure 3: outer-fold stability; a point is a fold-level held-out result.
    evolved_names = ["genetic_algorithm_performance", "genetic_algorithm_complexity_aware", "random_search_performance", "random_search_complexity_aware"]
    selected = folds[folds["model"].isin(evolved_names)].copy()
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for offset, model_name in enumerate(evolved_names):
        frame = selected[selected["model"] == model_name]
        xs = frame["test_balanced_accuracy"].to_numpy(dtype=float)
        jitter = np.linspace(-0.08, 0.08, len(xs)) if len(xs) > 1 else np.asarray([0.0])
        ax.scatter(np.arange(len(xs)) + jitter, xs, s=27, alpha=0.8, label=RESEARCH_MODEL_NAMES[model_name])
    ax.set_xlabel("Outer fold (ordered by repeat and fold)"); ax.set_ylabel("Held-out balanced accuracy")
    ax.set_title("Evolved-circuit performance across held-out folds")
    ax.grid(axis="y", alpha=0.2); ax.legend(frameon=False); fig.tight_layout()
    finish_figure(fig, "cv_fig3_evolution_fold_stability", "Figure 3. Fold-level held-out balanced accuracy for two evolutionary objectives and their candidate-budget-matched random-search controls. Each point represents one outer test partition; test labels are used only for evaluation, not for circuit selection.")

    # Figure 4: complexity/performance trade-off for evolved circuits only.
    fig, ax = plt.subplots(figsize=(7.4, 5.1))
    palette = {"genetic_algorithm_performance": "#4477AA", "genetic_algorithm_complexity_aware": "#CC6677", "random_search_performance": "#117733", "random_search_complexity_aware": "#AA4499"}
    for model_name in evolved_names:
        frame = selected[selected["model"] == model_name].dropna(subset=["estimated_depth", "two_qubit_gate_count", "test_balanced_accuracy"])
        sizes = 35 + 18 * frame["two_qubit_gate_count"].to_numpy(dtype=float)
        ax.scatter(frame["estimated_depth"], frame["test_balanced_accuracy"], s=sizes, alpha=0.72, color=palette[model_name], edgecolor="white", linewidth=0.5, label=RESEARCH_MODEL_NAMES[model_name])
        if not frame.empty:
            ax.scatter([frame["estimated_depth"].mean()], [frame["test_balanced_accuracy"].mean()], marker="D", s=70, color=palette[model_name], edgecolor="black", linewidth=0.6)
    ax.set_xlabel("Logical Qiskit circuit depth"); ax.set_ylabel("Outer-test balanced accuracy")
    ax.set_title("Accuracy–complexity trade-off (evolved circuits)")
    ax.legend(frameon=False); ax.grid(alpha=0.2); fig.tight_layout()
    finish_figure(fig, "cv_fig4_accuracy_complexity_tradeoff", "Figure 4. Outer-fold balanced accuracy versus logical circuit depth for evolved and random-search circuits. Circle size is proportional to the number of two-qubit gates; diamonds show method means. These are logical circuit metrics, not transpiled hardware cost or measured hardware performance.")

    # Figure 5: inner validation score vs outer test score for evolved circuits.
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    for model_name in evolved_names:
        frame = selected[selected["model"] == model_name].dropna(subset=["validation_balanced_accuracy", "test_balanced_accuracy"])
        ax.scatter(frame["validation_balanced_accuracy"], frame["test_balanced_accuracy"], s=30, alpha=0.78, label=RESEARCH_MODEL_NAMES[model_name], color=palette[model_name])
    ax.plot([0.4, 1.0], [0.4, 1.0], linestyle="--", color="#555555", linewidth=1)
    ax.set_xlim(0.4, 1.02); ax.set_ylim(0.4, 1.02)
    ax.set_xlabel("Inner validation balanced accuracy"); ax.set_ylabel("Outer-test balanced accuracy")
    ax.set_title("Validation-to-test generalization")
    ax.legend(frameon=False); ax.grid(alpha=0.2); fig.tight_layout()
    finish_figure(fig, "cv_fig5_validation_vs_outer_test", "Figure 5. Inner validation balanced accuracy versus held-out outer-test balanced accuracy for evolved and random-search circuits. The plot diagnoses selection optimism; the outer-test values were not used to choose the circuit.")

    # Figure 6: pre-specified reference confusion matrices, not models chosen by test score.
    matrix_models = ["logistic_regression_original30", "genetic_algorithm_complexity_aware"]
    fig, axes = plt.subplots(1, 2, figsize=(8.3, 3.8))
    for ax, model_name in zip(axes, matrix_models):
        frame = predictions[predictions["model"] == model_name]
        matrix = confusion_matrix(frame["y_true"].astype(int), frame["y_pred"].astype(int), labels=[0, 1], normalize="true")
        image = ax.imshow(matrix, vmin=0, vmax=1, cmap="Blues")
        for r in range(2):
            for c in range(2):
                ax.text(c, r, f"{matrix[r, c]:.2f}", ha="center", va="center", color="white" if matrix[r, c] > 0.55 else "black")
        ax.set_xticks([0, 1], labels=[class0_name, class1_name]); ax.set_yticks([0, 1], labels=[class0_name, class1_name])
        ax.set_xlabel("Predicted class"); ax.set_ylabel("True class")
        ax.set_title(RESEARCH_MODEL_NAMES[model_name], fontsize=8)
    fig.suptitle(f"Row-normalized outer-fold confusion matrices: {dataset_name}", y=1.02); fig.tight_layout()
    finish_figure(fig, "cv_fig6_confusion_matrices", "Figure 6. Row-normalized confusion matrices pooled over out-of-fold predictions from all repeats for two pre-specified reference models: full-feature logistic regression and the complexity-aware evolved QML circuit. Values are benchmark results, not clinical validation.")

    # Figure 7: paired cluster-bootstrap difference against the strongest classical reference.
    paired_path = study_dir / "paired_cluster_bootstrap_vs_logistic30.csv"
    if paired_path.exists():
        paired = pd.read_csv(paired_path)
        plot_rows = paired[paired["model"].isin(["qiskit_zz_inspired_feature_map_kernel", *evolved_names])].copy()
        if not plot_rows.empty:
            labels7 = [RESEARCH_MODEL_NAMES.get(str(name), str(name)) for name in plot_rows["model"]]
            values = plot_rows["difference_test_balanced_accuracy"].to_numpy(dtype=float)
            low_values = plot_rows["difference_test_balanced_accuracy_ci95_low"].to_numpy(dtype=float)
            high_values = plot_rows["difference_test_balanced_accuracy_ci95_high"].to_numpy(dtype=float)
            fig, ax = plt.subplots(figsize=(8.2, 3.9))
            y_pos = np.arange(len(plot_rows))
            ax.axvline(0, color="#555555", linestyle="--", linewidth=1)
            ax.errorbar(values, y_pos, xerr=np.vstack([np.maximum(0, values-low_values), np.maximum(0, high_values-values)]), fmt="o", color="#4477AA", ecolor="#777777", capsize=3)
            ax.set_yticks(y_pos, labels=labels7); ax.invert_yaxis()
            ax.set_xlabel("Difference in balanced accuracy vs. logistic regression (original features)")
            ax.set_title("Paired sample-cluster bootstrap comparison")
            ax.grid(axis="x", alpha=0.2); fig.tight_layout()
            finish_figure(fig, "cv_fig7_paired_difference_vs_classical", "Figure 7. Paired sample-cluster bootstrap differences in outer-fold balanced accuracy relative to full-feature logistic regression. Intervals crossing zero do not establish a difference at the corresponding nominal confidence level; repeated-CV observations are clustered by sample.")

    # Figure 8: variability in four-component variance retention across folds.
    pca_variance_path = study_dir / "fold_pca_variance.csv"
    if pca_variance_path.exists():
        pca_frame = pd.read_csv(pca_variance_path)
        component_columns = [f"pc{i}_explained_variance" for i in range(1, N_QUBITS + 1) if f"pc{i}_explained_variance" in pca_frame.columns]
        if component_columns:
            means = np.asarray([pca_frame[column].mean() for column in component_columns], dtype=float)
            stds = np.asarray([pca_frame[column].std(ddof=1) if len(pca_frame) > 1 else 0.0 for column in component_columns], dtype=float)
            cumulative = np.cumsum(means)
            fig, ax = plt.subplots(figsize=(6.6, 4.2))
            components = np.arange(1, len(component_columns) + 1)
            ax.bar(components, means, color="#4477AA", alpha=0.85, label="Mean individual variance")
            ax.errorbar(components, means, yerr=stds, fmt="none", ecolor="#222222", capsize=3, linewidth=1)
            ax.plot(components, cumulative, marker="o", color="#CC6677", label="Cumulative mean variance")
            ax.set_xticks(components); ax.set_xlabel("Principal component"); ax.set_ylabel("Explained variance ratio")
            ax.set_ylim(0, min(1.05, max(0.55, float(cumulative[-1]) + 0.15)))
            ax.set_title("PCA variance retention across outer folds")
            ax.grid(axis="y", alpha=0.2); ax.legend(frameon=False); fig.tight_layout()
            finish_figure(fig, "cv_fig8_pca_variance_stability", "Figure 8. Mean variance explained by each of the four PCA components across inner-training partitions, with fold-to-fold standard deviations. These components are fitted independently within each fold to avoid test-data leakage.")

    table_columns = ["display_name", "cv_repeats", "outer_folds", "pooled_test_accuracy", "test_accuracy_ci95_low", "test_accuracy_ci95_high", "mean_test_accuracy_across_repeats", "sd_test_accuracy_across_repeats", "pooled_test_balanced_accuracy", "test_balanced_accuracy_ci95_low", "test_balanced_accuracy_ci95_high", "pooled_test_f1_macro", "test_f1_macro_ci95_low", "test_f1_macro_ci95_high", "pooled_test_class0_recall", "test_class0_recall_ci95_low", "test_class0_recall_ci95_high", "mean_gate_count", "mean_two_qubit_gate_count", "mean_estimated_depth", "mean_candidate_budget_nominal", "mean_candidate_attempt_budget_nominal", "mean_candidate_attempt_safety_cap", "mean_candidate_attempts", "mean_candidate_unique_budget_target", "mean_unique_candidates_evaluated", "mean_search_elapsed_seconds", "mean_outer_fit_predict_elapsed_seconds", "mean_fit_predict_elapsed_seconds", "mean_validation_balanced_accuracy_loss_from_best", "accuracy_constraint_pass_rate", "accuracy_constraint_folds_evaluated", "n_unique_samples", "sample_majority_vote_accuracy", "sample_majority_vote_accuracy_wilson95_low", "sample_majority_vote_accuracy_wilson95_high"]
    summary[[column for column in table_columns if column in summary.columns]].to_csv(run_dir / "paper_results_table_repeated_cv.csv", index=False)
    save_json(figure_dir / "figure_manifest.json", {**manifest, "figure_count": len(manifest["figures"])})
    (figure_dir / "figure_captions.md").write_text("\n\n".join(f"**{caption.split('.')[0]}.**{caption.split('.', 1)[1]}" for caption in captions) + "\n", encoding="utf-8")
    notes = [
        "# Repeated-CV paper figure bundle", "",
        f"Evaluation: stratified {RESEARCH_CV_SPLITS}-fold outer CV repeated {RESEARCH_CV_REPEATS} times.",
        "Inside each outer training fold, a stratified inner train/validation split is used. Standardization, PCA and angle scaling are fitted only on inner-training data. Circuit selection uses validation labels only; outer test labels are reserved for evaluation.",
        "Performance-oriented GA and random search rank candidates by 0.5 × validation balanced accuracy + 0.5 × validation macro-F1. Complexity-aware GA and complexity-aware random search both select the least complex candidate within 0.02 absolute of the best observed inner-validation balanced accuracy, breaking ties by estimated depth, total gates and validation metrics. Each random-search baseline targets the same number of unique circuit evaluations as its paired GA within that outer fold; actual attempts, unique candidates and elapsed time are recorded separately.",
        "Pooled repeated-CV confidence intervals use sample-cluster bootstrap resampling, retaining each sample's repeated predictions together. A supplemental Wilson interval is calculated from one majority-vote prediction per unique sample and has a different estimand, so do not conflate the two. paired_cluster_bootstrap_vs_logistic30.csv compares QML methods against full-feature logistic regression; paired_cluster_bootstrap_qml_vs_classical.csv compares each QML method against all eight classical baselines; paired_cluster_bootstrap_qml_methods.csv contains the pre-specified paired search-method contrasts. accuracy_constraint_audit.csv checks the constraint for both GA and random-search complexity-aware methods. Search runtime is local wall-clock time, not physical hardware runtime.",
        "Quantum circuits are built with Qiskit QuantumCircuit and evaluated using Qiskit Statevector. There are no cloud jobs, device noise, transpiled hardware-cost measurements, or physical-QPU results.",
        f"Dataset: {dataset_name}. Binary task: {dataset_meta.get('task_description', '')}. Class 0 is {class0_name}; class 1 is {class1_name}.",
        "This is a local benchmark study, not a clinical prediction system. Do not infer quantum advantage from these results. Repeated CV improves internal robustness but is not independent external validation.", "",
        "## Figures", "",
    ]
    notes.extend(f"- `{item['stem']}.pdf` / `.png`: {item['caption']}" for item in manifest["figures"])
    (figure_dir / "paper_results_notes.md").write_text("\n".join(notes) + "\n", encoding="utf-8")
    log_message(f"Generated {len(manifest['figures'])} repeated-CV paper figures in {figure_dir} (PDF and 300-dpi PNG).")
    log_message(f"Repeated-CV paper table: {run_dir / 'paper_results_table_repeated_cv.csv'}")

def _save_paper_figure(fig: Any, figures_dir: Path, stem: str) -> List[str]:
    """Write PDF vector and 300-DPI PNG copies, with per-format failure isolation."""
    import matplotlib.pyplot as plt

    figures_dir.mkdir(parents=True, exist_ok=True)
    saved: List[str] = []
    errors: List[str] = []
    # PDF is vector-based and sufficient for publication; PNG is convenient for previews.
    # Omitting SVG avoids a redundant third export that can be slow on hosted notebook systems.
    for extension in ("pdf", "png"):
        path = figures_dir / f"{stem}.{extension}"
        try:
            if extension == "png":
                fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
            else:
                fig.savefig(path, bbox_inches="tight", facecolor="white")
            saved.append(str(path))
        except Exception as exc:
            errors.append(f"{extension}: {safe_error_message(exc)}")
    plt.close(fig)
    if not saved:
        raise RuntimeError(f"Could not export figure {stem}: {'; '.join(errors)}")
    if errors:
        log_message(f"Figure export warning for {stem}: {'; '.join(errors)}")
    return saved

def run_additional_hard_digit_benchmarks(action_dir: Path) -> None:
    """Run only two prespecified harder digit-pair tasks; the original four runs are reused later."""
    log_message("Running only the additional digits 3-vs-8 and 4-vs-9 tasks. Existing four-dataset results will not be retrained.")
    run_multi_dataset_research(action_dir, dataset_ids=ADDITIONAL_HARD_DIGIT_DATASET_IDS)

def _corrected_repeated_cv_test(
    differences: Sequence[float], test_train_ratio: float, null_value: float = 0.0,
    alternative: str = "two-sided",
) -> Dict[str, float]:
    """Nadeau-Bengio-style corrected repeated-CV t inference on paired fold differences.

    The correction inflates variance by (1/n + n_test/n_train), acknowledging overlap
    among repeated-CV training sets. This is more suitable than treating fold differences
    as independent, but still relies on approximate assumptions and is not equivalent to
    an independent external replication.
    """
    from scipy.stats import t as student_t
    values = np.asarray(list(differences), dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    mean = float(np.mean(values)) if n else float("nan")
    if n < 3:
        return {"n_paired_folds": n, "mean_difference": mean, "corrected_se": float("nan"),
                "df": max(n - 1, 0), "ci95_low": float("nan"), "ci95_high": float("nan"),
                "t_statistic": float("nan"), "p_value": float("nan")}
    variance = float(np.var(values, ddof=1))
    correction = (1.0 / n) + max(float(test_train_ratio), 0.0)
    se = float(np.sqrt(max(variance * correction, 0.0)))
    df = n - 1
    inference_status = "ok"
    if se <= 1e-15:
        # With zero observed fold variance, the corrected t statistic is not reliably
        # estimable. Do not report a misleading p=0 or a degenerate confidence interval.
        t_stat = float("nan")
        p = float("nan")
        low = high = float("nan")
        inference_status = "not_estimable_zero_observed_fold_variance"
    else:
        t_stat = (mean - null_value) / se
        critical = float(student_t.ppf(0.975, df))
        low, high = mean - critical * se, mean + critical * se
        if alternative == "less":
            p = float(student_t.cdf(t_stat, df))
        elif alternative == "greater":
            p = float(student_t.sf(t_stat, df))
        elif alternative == "two-sided":
            p = float(2.0 * student_t.sf(abs(t_stat), df))
        else:
            raise ValueError(f"Unknown alternative: {alternative}")
    return {"n_paired_folds": n, "mean_difference": mean, "corrected_se": se, "df": df,
            "ci95_low": float(low), "ci95_high": float(high), "t_statistic": float(t_stat),
            "p_value": float(np.clip(p, 0.0, 1.0)) if np.isfinite(p) else float("nan"),
            "inference_status": inference_status}

def run_corrected_repeated_cv_inference(action_dir: Path) -> None:
    """Run pre-specified corrected-CV comparisons on newest saved fold files; no retraining."""
    if common.ACTIVE_SESSION_DIR is None:
        raise RuntimeError("No active session. Resume the session containing the v3/v4 results first.")
    import math
    actions_dir = common.ACTIVE_SESSION_DIR / "actions"
    newest: Dict[str, Path] = {}
    for fold_path in actions_dir.rglob("fold_metrics.csv"):
        if fold_path.parent.name != "repeated_cv" or fold_path.parent.parent.name not in ALL_RESEARCH_DATASET_IDS:
            continue
        dataset_id = fold_path.parent.parent.name
        old = newest.get(dataset_id)
        if old is None or fold_path.stat().st_mtime > old.stat().st_mtime:
            newest[dataset_id] = fold_path
    if not newest:
        raise FileNotFoundError("No saved dataset fold_metrics.csv files were found in this session.")

    primary_rows: List[Dict[str, Any]] = []
    all_rows: List[Dict[str, Any]] = []
    ga_name, rs_name = "genetic_algorithm_complexity_aware", "random_search_complexity_aware"
    outcome_specs = {
        "two_qubit_gate_count": ("less", 0.0, "primary_complexity"),
        "estimated_depth": ("less", 0.0, "secondary_complexity"),
        "gate_count": ("less", 0.0, "secondary_complexity"),
        "test_balanced_accuracy": ("greater", -COMPLEXITY_BALANCED_ACCURACY_TOLERANCE, "noninferiority"),
    }
    for ds_index, dataset_id in enumerate(ALL_RESEARCH_DATASET_IDS):
        fold_path = newest.get(dataset_id)
        if fold_path is None:
            log_message(f"Dataset not yet available; skipped inference: {dataset_id}")
            continue
        folds = pd.read_csv(fold_path)
        if not {"model", "repeat", "fold"}.issubset(folds.columns):
            log_message(f"WARNING: required columns missing from {fold_path}; skipped {dataset_id}.")
            continue
        subset = folds[folds["model"].astype(str).isin([ga_name, rs_name])].copy()
        if subset.empty or set(subset["model"].astype(str)) != {ga_name, rs_name}:
            log_message(f"WARNING: matched methods missing in {fold_path}; skipped {dataset_id}.")
            continue
        ratio = 1.0 / max(RESEARCH_CV_SPLITS - 1, 1)
        if {"n_outer_test", "n_outer_train"}.issubset(subset.columns):
            context = subset[subset["model"].astype(str) == ga_name][["repeat", "fold", "n_outer_test", "n_outer_train"]].drop_duplicates()
            vals = context["n_outer_test"].astype(float).to_numpy() / np.maximum(context["n_outer_train"].astype(float).to_numpy(), 1.0)
            if len(vals) and np.isfinite(vals).any():
                ratio = float(np.nanmean(vals))
        for outcome, (alternative, null_value, family) in outcome_specs.items():
            if outcome not in subset.columns:
                continue
            wide = subset.pivot_table(index=["repeat", "fold"], columns="model", values=outcome, aggfunc="first")
            wide = wide.dropna(subset=[ga_name, rs_name]).sort_index()
            diffs = wide[ga_name].to_numpy(dtype=float) - wide[rs_name].to_numpy(dtype=float)
            stats = _corrected_repeated_cv_test(diffs, ratio, null_value=null_value, alternative=alternative)
            row = {
                "dataset_id": dataset_id,
                "dataset_name": str((load_research_dataset(dataset_id)[2]).get("dataset_name", dataset_id)),
                "comparison": "accuracy-constrained GA minus matched accuracy-constrained random search",
                "outcome": outcome,
                "family": family,
                "alternative": alternative,
                "null_value": float(null_value),
                "test_train_ratio_used": ratio,
                "inference_method": "Nadeau-Bengio corrected repeated-CV t approximation",
                "source_fold_metrics": str(fold_path),
                **stats,
            }
            if family == "primary_complexity":
                primary_rows.append(row)
            all_rows.append(row)
        log_message(f"Prepared corrected-CV inference for {dataset_id} using {int(subset[['repeat', 'fold']].drop_duplicates().shape[0])} paired folds.")

    if not all_rows:
        raise RuntimeError("No valid corrected-CV comparisons could be formed.")
    results = pd.DataFrame(all_rows)
    results["holm_adjusted_p_primary_two_qubit_gate_count"] = np.nan
    results["holm_adjusted_p_secondary_complexity"] = np.nan
    results["holm_adjusted_p_balanced_accuracy_noninferiority"] = np.nan
    primary_mask = results["family"] == "primary_complexity"
    results.loc[primary_mask, "holm_adjusted_p_primary_two_qubit_gate_count"] = _holm_adjust(
        results.loc[primary_mask, "p_value"].to_numpy(dtype=float)
    )
    secondary_mask = results["family"] == "secondary_complexity"
    results.loc[secondary_mask, "holm_adjusted_p_secondary_complexity"] = _holm_adjust(
        results.loc[secondary_mask, "p_value"].to_numpy(dtype=float)
    )
    noninf_mask = results["family"] == "noninferiority"
    results.loc[noninf_mask, "holm_adjusted_p_balanced_accuracy_noninferiority"] = _holm_adjust(
        results.loc[noninf_mask, "p_value"].to_numpy(dtype=float)
    )
    results["complexity_reduction_supported_after_holm"] = (
        (results["family"].isin(["primary_complexity", "secondary_complexity"]))
        & (results["mean_difference"] < 0)
        & np.where(primary_mask, results["holm_adjusted_p_primary_two_qubit_gate_count"] < 0.05,
                   results["holm_adjusted_p_secondary_complexity"] < 0.05)
    )
    results["balanced_accuracy_noninferiority_supported_after_holm"] = (
        noninf_mask
        & (results["ci95_low"] > -COMPLEXITY_BALANCED_ACCURACY_TOLERANCE)
        & (results["holm_adjusted_p_balanced_accuracy_noninferiority"] < 0.05)
    )
    out = action_dir / "cross_dataset_corrected_cv_inference.csv"
    results.to_csv(out, index=False)
    plan = action_dir / "confirmatory_inference_protocol.md"
    plan.write_text(
        "# Prespecified corrected repeated-CV inference\n\n"
        "Primary comparison: accuracy-constrained genetic algorithm minus matched accuracy-constrained random search. Primary endpoint: two-qubit gate count, with the directional alternative that the GA uses fewer gates. Holm correction is applied across datasets for this primary endpoint.\n\n"
        "Secondary endpoints are estimated depth and total gate count (one Holm family) and balanced-accuracy non-inferiority at a 0.02 absolute margin (one-sided test, one Holm family). The paired repeated-CV t approximation inflates fold-difference variance by (1/n + n_test/n_train), following the corrected resampled t-test approach; degrees of freedom are n-1. The confidence intervals are corrected-CV intervals, not ordinary independent-fold intervals.\n\n"
        "This procedure is a more dependence-aware sensitivity analysis than treating folds as independent, but it remains approximate. It does not turn reused cross-validation data into independent external validation, and conclusions should remain cautious. The datasets are analysed separately; no pooled score treats samples from different tasks as interchangeable. Gate counts/depth are logical proxies, not physical-hardware measurements.\n",
        encoding="utf-8",
    )
    cols = ["dataset_id", "outcome", "mean_difference", "ci95_low", "ci95_high", "p_value",
            "holm_adjusted_p_primary_two_qubit_gate_count", "holm_adjusted_p_secondary_complexity",
            "holm_adjusted_p_balanced_accuracy_noninferiority", "complexity_reduction_supported_after_holm",
            "balanced_accuracy_noninferiority_supported_after_holm", "n_paired_folds", "test_train_ratio_used"]
    log_message("\nCORRECTED REPEATED-CV INFERENCE (GA minus matched constrained random search)")
    print(results[[c for c in cols if c in results.columns]].to_string(index=False))
    log_message(f"Saved corrected-CV inference table: {out}")
    log_message(f"Saved prespecified inference plan: {plan}")
    log_message("No model training was performed. Corrected repeated-CV inference is approximate and is not independent external validation.")

def finalize_saved_multidataset_inference(action_dir: Path) -> None:
    """Analyse an already-completed multi-dataset run without retraining or rerunning prediction bootstraps."""
    if common.ACTIVE_SESSION_DIR is None:
        raise RuntimeError("No active session. Resume the session containing the completed v3 multi-dataset run first.")
    actions_dir = common.ACTIVE_SESSION_DIR / "actions"
    candidates = []
    if actions_dir.exists():
        for summary_path in actions_dir.rglob("cross_dataset_model_summary.csv"):
            source_run_dir = summary_path.parent
            dataset_root = source_run_dir / "datasets"
            available = [
                dataset_id for dataset_id in RESEARCH_DATASET_IDS
                if (dataset_root / dataset_id / "repeated_cv" / "fold_metrics.csv").exists()
            ]
            if available:
                candidates.append((summary_path.stat().st_mtime, source_run_dir, available))
    if not candidates:
        raise FileNotFoundError(
            "No completed multi-dataset run was found in the active session. Resume the session named v3 (or the session containing the results) and try again."
        )
    _, source_run_dir, available_datasets = max(candidates, key=lambda item: item[0])
    log_message(f"Analysing saved fold metrics under {source_run_dir}; no model training or prediction bootstrap will be run.")
    dataset_root = source_run_dir / "datasets"
    frames = []
    missing = []
    for dataset_id in RESEARCH_DATASET_IDS:
        study_dir = dataset_root / dataset_id / "repeated_cv"
        fold_path = study_dir / "fold_metrics.csv"
        if not fold_path.exists():
            missing.append(dataset_id)
            continue
        fold_frame = pd.read_csv(fold_path)
        pairs = _paired_fold_complexity_comparisons(
            fold_frame, n_bootstrap=RESEARCH_CV_BOOTSTRAPS,
            seed=RANDOM_STATE + 74000 + RESEARCH_DATASET_IDS.index(dataset_id),
        )
        if pairs.empty:
            log_message(f"WARNING: no paired complexity comparisons could be generated for {dataset_id}.")
            continue
        pairs.insert(0, "dataset_id", dataset_id)
        pairs.to_csv(study_dir / "paired_fold_complexity_comparisons.csv", index=False)
        frames.append(pairs)
        log_message(f"Complexity uncertainty computed for {dataset_id} ({len(pairs)} outcomes).")
    if not frames:
        raise FileNotFoundError("No valid paired fold-level comparisons were generated.")
    combined = pd.concat(frames, ignore_index=True)
    combined["holm_adjusted_p_value_global_primary_family"] = _holm_adjust(
        combined["p_value_sign_flip_two_sided_exploratory"].to_numpy(dtype=float)
    )
    combined["global_multiple_comparison_family"] = f"{len(combined)}_dataset_by_outcome_tests_holm"
    combined["global_p_adjustment_scope"] = "one pre-specified family across all available datasets and four outcomes"
    result_path = source_run_dir / "cross_dataset_paired_complexity_comparisons.csv"
    combined.to_csv(result_path, index=False)
    plan_path = source_run_dir / "multiple_comparison_plan.md"
    plan_path.write_text(
        "# Paired complexity uncertainty and multiplicity plan\n\n"
        "Primary comparison: accuracy-constrained genetic algorithm minus matched accuracy-constrained random search. Outcomes: two-qubit gate count, estimated depth, total gate count and outer-test balanced accuracy. Holm step-down correction is applied to the sign-flip p-values across every available dataset-by-outcome row in cross_dataset_paired_complexity_comparisons.csv.\n\n"
        "For balanced accuracy, the pointwise lower 95% paired-fold bootstrap interval is compared descriptively with a -0.02 non-inferiority margin. This is not a multiplicity-adjusted non-inferiority test. The bootstrap intervals and sign-flip tests resample/treat outer-fold differences as paired observations; because outer-CV training sets overlap, folds are dependent. Therefore these are exploratory sensitivity analyses, not definitive confirmatory inference. A confirmatory claim would require independent search replicates per fold or a more appropriate repeated-CV inferential procedure, specified before analysing results.\n\n"
        "Secondary QML-versus-classical and other metric comparisons should be treated as exploratory unless their testing family and correction are specified separately. Gate count and estimated depth are logical circuit proxies, not hardware timing or transpiled cost estimates.\n",
        encoding="utf-8",
    )
    # Write a convenient copy inside the current analysis action for upload.
    try:
        import shutil
        shutil.copy2(result_path, action_dir / result_path.name)
        shutil.copy2(plan_path, action_dir / plan_path.name)
    except OSError as exc:
        log_message(f"Could not copy convenience reports to current action directory: {exc}")
    show_cols = [
        "dataset_id", "outcome", "mean_paired_difference",
        "paired_fold_bootstrap_ci95_low", "paired_fold_bootstrap_ci95_high",
        "p_value_sign_flip_two_sided_exploratory",
        "holm_adjusted_p_value_global_primary_family",
        "complexity_reduction_supported_pointwise",
        "balanced_accuracy_noninferiority_supported_pointwise",
    ]
    print("\nPAIRED COMPLEXITY COMPARISONS (exploratory fold-level inference)")
    print(combined[[c for c in show_cols if c in combined.columns]].to_string(index=False))
    if missing:
        log_message(f"WARNING: no saved fold_metrics.csv for datasets: {', '.join(missing)}")
    log_message(f"Saved paired complexity report: {result_path}")
    log_message(f"Saved multiplicity plan: {plan_path}")
    log_message("No model training was performed. Interpret fold-level uncertainty as exploratory due to overlapping cross-validation folds.")

def finalize_latest_repeated_cv(action_dir: Path) -> None:
    """Rebuild summary/paired intervals/figures from the latest saved CV checkpoint, without retraining."""
    session_dir = action_dir.parent.parent
    actions_dir = session_dir / "actions"
    candidates = []
    if actions_dir.exists():
        for study_dir in actions_dir.rglob("repeated_cv"):
            fold_path, pred_path = study_dir / "fold_metrics.csv", study_dir / "oof_predictions.csv"
            if not (study_dir.is_dir() and fold_path.exists() and pred_path.exists()):
                continue
            try:
                fold_frame = pd.read_csv(fold_path, usecols=["repeat", "fold"])
                completed_folds = fold_frame[["repeat", "fold"]].drop_duplicates().shape[0]
                pred_frame = pd.read_csv(pred_path, usecols=["repeat", "model", "sample_index"])
                expected_models = len(pred_frame["model"].unique())
                complete = completed_folds >= RESEARCH_CV_SPLITS * RESEARCH_CV_REPEATS
                complete = complete and expected_models > 0
                if complete:
                    source_action = study_dir.parent
                    candidates.append((study_dir.stat().st_mtime, source_action, study_dir))
            except (OSError, ValueError, pd.errors.ParserError):
                continue
    if not candidates:
        raise FileNotFoundError("No saved repeated-CV fold_metrics.csv and oof_predictions.csv were found in this session.")
    _, source_action, study_dir = max(candidates, key=lambda item: item[0])
    log_message(f"Finalizing saved repeated-CV outputs from {source_action}; no model training will be run.")
    finalize_repeated_cv_outputs(study_dir, source_action)


# Reporting, figures, and saved-study analysis
import hashlib
import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
import quantum_core as common
from quantum_core import _qiskit_imports, build_genome_circuit, build_zz_feature_map_circuit, fidelity_kernel, genome_complexity, normalize_genome, states_for_data
from quantum_core import ALL_RESEARCH_DATASET_IDS, FINITE_SHOTS, GA_SEEDS, N_QUBITS, PROJECT_DIR, RANDOM_STATE, RESEARCH_CV_BOOTSTRAPS, RESEARCH_CV_REPEATS, RESEARCH_CV_SPLITS, SCRIPT_VERSION, classification_metrics, list_user_sessions, log_message, safe_error_message, save_json
from quantum_core import load_research_dataset, prepare_dataset
from optimizer import evolve_circuits

def export_saved_circuit_diagrams_pdf(action_dir: Path) -> None:
    """Export actual saved Qiskit circuits from this session into a black-and-white PDF.

    This is a rendering/export step only: it does not train models or rerun circuit searches.
    In representative mode it exports repeat 1 / fold 1 for each saved dataset/method where
    available. All-fold mode exports every saved genome/QASM circuit found in the session.
    """
    if common.ACTIVE_SESSION_DIR is None:
        raise RuntimeError("No active session. Log in and open a session before exporting circuits.")

    import shutil
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from qiskit import qasm2

    actions_root = common.ACTIVE_SESSION_DIR / "actions"
    dataset_labels = {
        "breast_cancer": "Breast Cancer Wisconsin Diagnostic",
        "iris_setosa_vs_versicolor": "Iris: Setosa vs. Versicolor",
        "wine_class0_vs_rest": "Wine: Class 0 vs. Classes 1/2",
        "digits_0_vs_1": "Handwritten Digits: 0 vs. 1",
        "digits_3_vs_8": "Handwritten Digits: 3 vs. 8",
        "digits_4_vs_9": "Handwritten Digits: 4 vs. 9",
    }

    print("\nCircuit PDF export")
    print("[1] Representative circuits: one per dataset and method (repeat 1 / fold 1 where available)")
    print("[2] All saved circuits: include every saved fold/repeat circuit")
    export_choice = input("Choose export mode [1]: ").strip() or "1"
    if export_choice not in {"1", "2"}:
        print("Invalid choice. Using representative circuits.")
        export_choice = "1"

    if not actions_root.exists():
        raise FileNotFoundError(f"No action history was found under {actions_root}")

    def read_json(path: Path) -> Optional[Dict[str, Any]]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else None
        except (OSError, json.JSONDecodeError):
            return None

    def infer_dataset_id(path: Path) -> str:
        # Multi-dataset runs include the dataset ID as a directory component.
        for part in path.parts:
            if part in ALL_RESEARCH_DATASET_IDS:
                return part
        # Single-dataset repeated-CV runs record their ID in study_protocol.json.
        for parent in path.parents:
            for protocol_name in ("study_protocol.json", "research_run_protocol.json"):
                protocol = parent / protocol_name
                if protocol.exists():
                    metadata = read_json(protocol)
                    if metadata:
                        dataset_id = str(metadata.get("dataset_id", ""))
                        if dataset_id in ALL_RESEARCH_DATASET_IDS:
                            return dataset_id
        # The legacy fixed-split workflow uses the built-in breast-cancer dataset.
        return "breast_cancer"

    def action_index(path: Path) -> int:
        parts = path.parts
        for i, part in enumerate(parts[:-1]):
            if part == "actions":
                match = re.match(r"(\d+)_", parts[i + 1])
                if match:
                    return int(match.group(1))
        return -1

    def method_label(payload: Dict[str, Any], path: Path) -> str:
        mode = str(payload.get("mode", "")).lower()
        stem = path.stem.lower()
        if mode == "random_search_performance" or "random_search_performance" in stem:
            return "Random search — performance selection"
        if mode == "random_search_complexity_aware" or "random_search_complexity_aware" in stem:
            return "Random search — complexity-aware selection"
        if mode == "complexity_aware" or "complexity_aware" in stem:
            if stem.startswith("repeat_"):
                return "Genetic algorithm — complexity-aware selection"
            return "Fixed-split evolution — complexity-aware"
        if mode == "performance" or (stem.startswith("repeat_") and stem.endswith("_performance")):
            return "Genetic algorithm — performance selection"
        if mode == "hardware_aware" or "hardware_aware" in stem:
            return "Fixed-split evolution — logical-complexity-aware"
        if mode == "accuracy" or re.match(r"best_circuit_accuracy", stem):
            return "Fixed-split evolution — accuracy-based"
        return f"Saved evolved circuit ({mode or path.stem})"

    def associated_qasm(json_path: Path, payload: Dict[str, Any]) -> Optional[Path]:
        # Repeated-CV and per-seed exports use the same stem or a *_example suffix.
        options = [json_path.with_suffix(".qasm"), json_path.with_name(json_path.stem + "_example.qasm")]
        # The aggregate fixed-split best-genome file points to the validation-selected seed.
        selection = payload.get("selection_across_seeds", {})
        if isinstance(selection, dict) and selection.get("selected_seed_by_validation_fitness") is not None:
            seed = selection["selected_seed_by_validation_fitness"]
            mode = str(payload.get("mode", "accuracy"))
            options.append(json_path.with_name(f"best_circuit_{mode}_seed_{seed}_example.qasm"))
        for option in options:
            if option.is_file():
                return option
        return None

    candidates: List[Dict[str, Any]] = []
    seen_qasm_paths = set()
    json_count = 0
    for json_path in actions_root.rglob("*.json"):
        if "circuits" not in json_path.parts:
            continue
        payload = read_json(json_path)
        if not payload or not isinstance(payload.get("genome"), list):
            continue
        json_count += 1
        dataset_id = infer_dataset_id(json_path)
        label = dataset_labels.get(dataset_id, dataset_id)
        qasm_path = associated_qasm(json_path, payload)
        if qasm_path:
            seen_qasm_paths.add(qasm_path.resolve())
        repeat = payload.get("repeat")
        fold = payload.get("fold")
        try:
            repeat_int = int(repeat) if repeat is not None else None
        except (ValueError, TypeError):
            repeat_int = None
        try:
            fold_int = int(fold) if fold is not None else None
        except (ValueError, TypeError):
            fold_int = None
        candidates.append({
            "dataset_id": dataset_id,
            "dataset_label": label,
            "method": method_label(payload, json_path),
            "source_json": json_path,
            "qasm_path": qasm_path,
            "payload": payload,
            "repeat": repeat_int,
            "fold": fold_int,
            "action_index": action_index(json_path),
            "source_kind": "saved Qiskit genome JSON",
        })

    # Include the saved standard ZZ-inspired Qiskit baseline from the fixed-split workflow.
    # Other QASM files are paired with a genome JSON above and are not duplicated here.
    for qasm_path in actions_root.rglob("qiskit_zz_inspired_feature_map.qasm"):
        try:
            resolved = qasm_path.resolve()
        except OSError:
            resolved = qasm_path
        if resolved in seen_qasm_paths:
            continue
        dataset_id = infer_dataset_id(qasm_path)
        candidates.append({
            "dataset_id": dataset_id,
            "dataset_label": dataset_labels.get(dataset_id, dataset_id),
            "method": "Standard ZZ-inspired feature-map baseline",
            "source_json": None,
            "qasm_path": qasm_path,
            "payload": {},
            "repeat": None,
            "fold": None,
            "action_index": action_index(qasm_path),
            "source_kind": "saved Qiskit OpenQASM 2 file",
        })

    if not candidates:
        raise FileNotFoundError(
            "No saved circuit genomes or QASM circuits were found in this session. "
            "First run option 14 for the four-dataset repeated-CV study, option 16 for the "
            "additional digit-pair datasets, or option 5/6 for the single-dataset evolved circuit. "
            "Then return here and choose option 18. The PDF export itself does not retrain models."
        )

    if export_choice == "1":
        # Select the most recent action's repeat-1/fold-1 circuit for each dataset/method.
        grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        for item in candidates:
            grouped.setdefault((item["dataset_id"], item["method"]), []).append(item)
        selected = []
        for items in grouped.values():
            def representative_rank(item: Dict[str, Any]) -> Tuple[int, int, int, int, str]:
                is_first_fold = int(item["repeat"] == 1 and item["fold"] == 1)
                repeat_rank = -(item["repeat"] if item["repeat"] is not None else 0)
                fold_rank = -(item["fold"] if item["fold"] is not None else 0)
                return (item["action_index"], is_first_fold, repeat_rank, fold_rank, str(item["source_json"] or item["qasm_path"]))
            selected.append(max(items, key=representative_rank))
        candidates = selected

    candidates.sort(key=lambda item: (
        item["dataset_label"].lower(), item["method"].lower(),
        item["repeat"] if item["repeat"] is not None else 0,
        item["fold"] if item["fold"] is not None else 0,
        str(item["source_json"] or item["qasm_path"]),
    ))

    pdf_path = action_dir / "figures" / "quantum_circuit_diagrams.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_rows: List[Dict[str, Any]] = []

    with PdfPages(pdf_path, metadata={
        "Title": "Saved Quantum Circuit Diagrams by Dataset",
        "Author": "Qiskit circuit export utility",
        "Subject": "Black-and-white diagrams rendered from saved Qiskit circuits",
    }) as pdf:
        for page_number, item in enumerate(candidates, start=1):
            qasm_path = item["qasm_path"]
            if qasm_path is not None:
                try:
                    circuit = qasm2.loads(qasm_path.read_text(encoding="utf-8"))
                except Exception as exc:
                    log_message(f"Could not read QASM {qasm_path}: {safe_error_message(exc)}; attempting genome fallback.")
                    circuit = None
            else:
                circuit = None

            if circuit is None:
                payload = item["payload"]
                genome = payload.get("genome") if payload else None
                if not isinstance(genome, list):
                    log_message(f"Skipping {item['method']} for {item['dataset_label']}: no usable QASM or genome.")
                    continue
                # This fallback preserves the evolved gate topology; saved QASM is preferred
                # because it records the parameter values from the original representative input.
                circuit = build_genome_circuit(np.zeros(N_QUBITS, dtype=float), genome)

            circuit_depth = int(circuit.depth())
            gate_count = int(circuit.size())
            two_qubit_count = int(sum(circuit.count_ops().get(gate, 0) for gate in ("cx", "cz", "cnot")))
            details = []
            if item["repeat"] is not None and item["fold"] is not None:
                details.append(f"Repeat {item['repeat']}, fold {item['fold']}")
            metrics = item["payload"].get("selection_metrics", {}) if item["payload"] else {}
            if isinstance(metrics, dict):
                if metrics.get("validation_balanced_accuracy") is not None:
                    try:
                        details.append(f"Validation balanced accuracy = {float(metrics['validation_balanced_accuracy']):.3f}")
                    except (ValueError, TypeError):
                        pass
                if metrics.get("validation_f1_macro") is not None:
                    try:
                        details.append(f"Validation macro-F1 = {float(metrics['validation_f1_macro']):.3f}")
                    except (ValueError, TypeError):
                        pass
            details.append(f"{circuit.num_qubits} qubits | {gate_count} gates | depth {circuit_depth}")
            title = f"{item['dataset_label']}\n{item['method']}"
            subtitle = " | ".join(details)

            try:
                fig = circuit.draw(output="mpl", style="bw", fold=-1, idle_wires=False)
                fig.suptitle(title + "\n" + subtitle, fontsize=11, fontweight="bold", y=0.99)
                fig.tight_layout(rect=(0, 0, 1, 0.90))
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
            except Exception as exc:
                plt.close("all")
                log_message(f"Could not render circuit {item['method']} / {item['dataset_label']}: {safe_error_message(exc)}")
                continue

            manifest_rows.append({
                "page": page_number,
                "dataset_id": item["dataset_id"],
                "dataset": item["dataset_label"],
                "method": item["method"],
                "repeat": item["repeat"],
                "fold": item["fold"],
                "qubits": int(circuit.num_qubits),
                "gate_count": gate_count,
                "two_qubit_gate_count": two_qubit_count,
                "depth": circuit_depth,
                "source_json": str(item["source_json"]) if item["source_json"] else "",
                "source_qasm": str(item["qasm_path"]) if item["qasm_path"] else "",
                "render_source": item["source_kind"] if item["qasm_path"] else "genome fallback with zero input angles",
            })
            print(f"Rendered page {len(manifest_rows)}: {item['dataset_label']} — {item['method']}")

    if not manifest_rows:
        raise RuntimeError("No circuit diagrams could be rendered. See the messages above for details.")

    manifest_path = action_dir / "figures" / "quantum_circuit_diagrams_manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)

    # Make the main PDF easy to locate immediately after returning to the session menu.
    session_pdf = common.ACTIVE_SESSION_DIR / "quantum_circuit_diagrams.pdf"
    shutil.copy2(pdf_path, session_pdf)
    session_manifest = common.ACTIVE_SESSION_DIR / "quantum_circuit_diagrams_manifest.csv"
    shutil.copy2(manifest_path, session_manifest)

    log_message(f"Circuit diagram PDF created: {pdf_path.resolve()}")
    log_message(f"Session-level copy: {session_pdf.resolve()}")
    log_message(f"Pages rendered: {len(manifest_rows)} from {json_count} saved genome JSON files discovered.")
    log_message(f"Circuit manifest saved: {manifest_path.resolve()}")
    if export_choice == "1":
        log_message("Representative mode selected one saved circuit per dataset/method; repeat 1/fold 1 was preferred where available.")
    else:
        log_message("All-circuits mode exported every usable saved genome/QASM circuit discovered in this session.")

def _short_model_name(name: str) -> str:
    labels = {
        "logistic_regression_pca4": "Logistic regression (4 PCA)",
        "mlp_pca4": "MLP (4 PCA)",
        "logistic_regression_original30": "Logistic regression (original features)",
        "mlp_original30": "MLP (original features)",
        "rbf_svm_pca4": "RBF-SVM (4 PCA)",
        "rbf_svm_original30": "RBF-SVM (original features)",
        "random_forest_pca4": "Random forest (4 PCA)",
        "random_forest_original30": "Random forest (original features)",
        "random_search_performance": "Random-search QML ablation",
        "qiskit_zz_inspired_feature_map_kernel": "Qiskit ZZ-inspired kernel",
        "genetic_algorithm_accuracy": "Evolved QML (accuracy)",
        "genetic_algorithm_hardware_aware": "Evolved QML (logical-complexity-aware)",
        "logistic_regression": "Logistic regression (legacy run)",
        "small_mlp": "Small MLP (legacy run)",
        "standard_zz_inspired_quantum_kernel": "Standard QML (legacy run)",
    }
    return labels.get(str(name), str(name).replace("_", " "))

def _write_paper_table(model_rows: pd.DataFrame, output_dir: Path) -> None:
    columns = [
        "experiment", "feature_view", "validation_accuracy", "validation_balanced_accuracy",
        "validation_f1_macro", "test_accuracy", "test_balanced_accuracy", "test_f1_macro",
        "test_malignant_recall", "test_accuracy_std", "test_balanced_accuracy_std", "test_f1_macro_std",
        "test_malignant_recall_std", "validation_accuracy_std", "gate_count", "gate_count_std",
        "two_qubit_gate_count", "two_qubit_gate_count_std", "estimated_depth", "estimated_depth_std", "seed_count", "seeds",
    ]
    table = model_rows[[c for c in columns if c in model_rows.columns]].copy()
    if "experiment" in table.columns:
        table.insert(0, "model", table.pop("experiment").map(_short_model_name))
    table.to_csv(output_dir / "paper_results_table.csv", index=False)

def make_paper_figures(metrics: pd.DataFrame, run_dir: Path, data_source: str) -> None:
    """Generate publication-oriented figures from one complete local workflow."""
    import matplotlib.pyplot as plt

    figures_dir = run_dir / "figures" / "paper"
    figures_dir.mkdir(parents=True, exist_ok=True)
    captions = []
    manifest = {"script_version": SCRIPT_VERSION, "created_at": datetime.now().isoformat(timespec="seconds"), "data_source": data_source, "figures": []}

    # Only model-result rows belong in the primary performance table; sampling repeats are separate.
    model_rows = metrics[~metrics["experiment"].astype(str).str.startswith("finite_shot_")].copy()
    model_rows = model_rows.drop_duplicates(subset=["experiment"], keep="last")
    model_rows = model_rows[model_rows.get("test_accuracy", pd.Series(index=model_rows.index, dtype=float)).notna()].copy()
    if model_rows.empty:
        log_message("Paper charts were not generated: no held-out test metrics were found in the selected results.")
        return
    model_rows = model_rows.sort_values("experiment").reset_index(drop=True)
    _write_paper_table(model_rows, run_dir)

    palette = {
        "test_accuracy": "#4477AA",
        "test_balanced_accuracy": "#EE6677",
        "test_f1_macro": "#228833",
        "test_malignant_recall": "#AA3377",
    }
    plt.rcParams.update({
        "font.family": "serif", "font.size": 9, "axes.titlesize": 10,
        "axes.labelsize": 9, "legend.fontsize": 8, "xtick.labelsize": 8,
        "ytick.labelsize": 8, "axes.spines.top": False, "axes.spines.right": False,
        "savefig.dpi": 300,
    })

    # Figure 1: four clinically relevant held-out metrics across models.
    metric_cols = ["test_accuracy", "test_balanced_accuracy", "test_f1_macro", "test_malignant_recall"]
    available_metrics = [col for col in metric_cols if col in model_rows.columns and model_rows[col].notna().any()]
    if available_metrics:
        labels = [_short_model_name(v) for v in model_rows["experiment"]]
        fig_height = max(4.8, 0.58 * len(model_rows) + 1.2)
        fig, ax = plt.subplots(figsize=(10.2, fig_height))
        y = np.arange(len(model_rows))
        bar_height = 0.76 / len(available_metrics)
        offsets = (np.arange(len(available_metrics)) - (len(available_metrics) - 1) / 2) * bar_height
        legend_labels = {
            "test_accuracy": "Accuracy", "test_balanced_accuracy": "Balanced accuracy",
            "test_f1_macro": "Macro-F1", "test_malignant_recall": "Malignant recall",
        }
        for offset, col in zip(offsets, available_metrics):
            values = pd.to_numeric(model_rows[col], errors="coerce").fillna(0).to_numpy(float)
            std_col = f"{col}_std"
            errors = pd.to_numeric(model_rows[std_col], errors="coerce").fillna(0).to_numpy(float) if std_col in model_rows.columns else np.zeros_like(values)
            bars = ax.barh(y + offset, values, height=bar_height * 0.88, label=legend_labels.get(col, col), color=palette.get(col), xerr=errors, error_kw={"capsize": 2, "elinewidth": 0.8})
            for bar, value in zip(bars, values):
                ax.text(min(value + 0.008, 1.015), bar.get_y() + bar.get_height() / 2, f"{value:.2f}", va="center", ha="left", fontsize=7)
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.12)
        ax.set_xlabel("Held-out test score")
        ax.set_title("Held-out classification performance")
        ax.grid(axis="x", alpha=0.22)
        ax.set_axisbelow(True)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.17), ncol=min(4, len(available_metrics)), frameon=False)
        fig.text(0.01, 0.005, "The 30-feature classical references are not feature-matched to four-qubit models.", fontsize=7.5, color="#444444")
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        files = _save_paper_figure(fig, figures_dir, "fig1_test_performance")
        caption = "Figure 1. Held-out test accuracy, balanced accuracy, macro-F1 and malignant-class recall. Error bars show standard deviation across genetic-algorithm seeds where available. The 30-feature classical models are practical reference baselines; comparisons against four-feature quantum models are not feature-matched."
        captions.append(caption)
        manifest["figures"].append({"stem": "fig1_test_performance", "caption": caption, "files": files})

    # Figure 2: validation/test agreement, which reveals selection/generalization gaps.
    if {"validation_accuracy", "test_accuracy"}.issubset(model_rows.columns):
        fig, ax = plt.subplots(figsize=(7.2, 5.8))
        x = pd.to_numeric(model_rows["validation_accuracy"], errors="coerce").to_numpy(float)
        y = pd.to_numeric(model_rows["test_accuracy"], errors="coerce").to_numpy(float)
        groups = model_rows["experiment"].astype(str).tolist()
        model_colors = {
            "logistic_regression_pca4": "#4477AA", "mlp_pca4": "#66CCEE",
            "logistic_regression_original30": "#228833", "mlp_original30": "#CCBB44",
            "qiskit_zz_inspired_feature_map_kernel": "#AA3377",
            "genetic_algorithm_accuracy": "#CC6677", "genetic_algorithm_hardware_aware": "#332288",
        }
        for index, (xv, yv, name) in enumerate(zip(x, y, groups)):
            marker = "s" if "genetic_algorithm" in name else ("^" if "standard_" in name else "o")
            color = model_colors.get(name, "#777777")
            xerr_col, yerr_col = "validation_accuracy_std", "test_accuracy_std"
            xerr = float(pd.to_numeric(pd.Series([model_rows.iloc[index].get(xerr_col, 0.0)]), errors="coerce").fillna(0).iloc[0])
            yerr = float(pd.to_numeric(pd.Series([model_rows.iloc[index].get(yerr_col, 0.0)]), errors="coerce").fillna(0).iloc[0])
            ax.errorbar(xv, yv, xerr=xerr, yerr=yerr, fmt=marker, markersize=6, color=color, ecolor=color, capsize=2, markeredgecolor="white", markeredgewidth=0.6, zorder=3, label=_short_model_name(name))
        ax.plot([0.4, 1.0], [0.4, 1.0], linestyle="--", color="#666666", linewidth=1, label="Validation = test")
        ax.set_xlim(0.4, 1.02); ax.set_ylim(0.4, 1.02)
        ax.set_xlabel("Validation accuracy (used during model selection)")
        ax.set_ylabel("Held-out test accuracy")
        ax.set_title("Validation performance versus held-out generalization")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7)
        fig.subplots_adjust(right=0.66)
        files = _save_paper_figure(fig, figures_dir, "fig2_validation_vs_test")
        caption = "Figure 2. Validation accuracy used for model/circuit selection versus test accuracy on the held-out partition. Error bars show genetic-algorithm standard deviations where available; the diagonal indicates equal validation and test accuracy."
        captions.append(caption)
        manifest["figures"].append({"stem": "fig2_validation_vs_test", "caption": caption, "files": files})

    # Figure 3: evolutionary learning curves saved by each genetic-algorithm run.
    histories = []
    for mode, label, color in (("accuracy", "Accuracy objective", "#4477AA"), ("hardware_aware", "Logical-complexity-aware objective", "#CC6677")):
        history_path = run_dir / f"evolution_history_{mode}.csv"
        if history_path.exists():
            frame = pd.read_csv(history_path)
            if "generation" in frame.columns and "best_validation_accuracy" in frame.columns and not frame.empty:
                histories.append((frame, label, color, mode))
    if histories:
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        for frame, label, color, mode in histories:
            mean_values = pd.to_numeric(frame["best_validation_accuracy"], errors="coerce").to_numpy(float)
            std_values = pd.to_numeric(frame.get("best_validation_accuracy_std", pd.Series(np.zeros(len(frame)))), errors="coerce").fillna(0).to_numpy(float)
            generations = pd.to_numeric(frame["generation"], errors="coerce").to_numpy(float)
            ax.plot(generations, mean_values, marker="o", markersize=3.5, label=label, color=color)
            ax.fill_between(generations, np.maximum(0, mean_values - std_values), np.minimum(1, mean_values + std_values), color=color, alpha=0.16, linewidth=0)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Validation accuracy of best candidate in generation")
        ax.set_ylim(0.0, 1.03)
        ax.set_title("Evolutionary search progress")
        ax.grid(alpha=0.22); ax.legend(frameon=False)
        fig.tight_layout()
        files = _save_paper_figure(fig, figures_dir, "fig3_evolution_convergence")
        caption = f"Figure 3. Mean best-candidate validation accuracy per generation across {len(GA_SEEDS)} genetic-algorithm seeds; shaded regions show ±1 standard deviation. Validation data is used for selection; it is not the final test result."
        captions.append(caption)
        manifest["figures"].append({"stem": "fig3_evolution_convergence", "caption": caption, "files": files})

    # Figure 4: logical circuit complexity for the two selected evolved circuits.
    evolutionary = model_rows[model_rows["experiment"].astype(str).isin(["genetic_algorithm_accuracy", "genetic_algorithm_hardware_aware"])].copy()
    complexity_cols = [c for c in ("gate_count", "two_qubit_gate_count", "estimated_depth") if c in evolutionary.columns and evolutionary[c].notna().any()]
    if not evolutionary.empty and complexity_cols:
        fig, axes = plt.subplots(1, len(complexity_cols), figsize=(4.0 * len(complexity_cols), 4.0), squeeze=False)
        short_names = [_short_model_name(v) for v in evolutionary["experiment"]]
        color_map = ["#4477AA" if "accuracy" in str(name) and "hardware" not in str(name) else "#CC6677" for name in evolutionary["experiment"]]
        for ax, col in zip(axes[0], complexity_cols):
            vals = pd.to_numeric(evolutionary[col], errors="coerce").to_numpy(float)
            std_col = f"{col}_std"
            errs = pd.to_numeric(evolutionary[std_col], errors="coerce").fillna(0).to_numpy(float) if std_col in evolutionary.columns else np.zeros_like(vals)
            bars = ax.bar(np.arange(len(vals)), vals, yerr=errs, capsize=3, color=color_map, width=0.62)
            ax.set_xticks(np.arange(len(vals)), ["Performance\nobjective", "Logical-complexity-aware"] if len(vals) == 2 else short_names, rotation=0)
            ax.set_ylabel(col.replace("_", " ").title())
            ax.set_title(col.replace("_", " ").title())
            ax.grid(axis="y", alpha=0.2); ax.set_axisbelow(True)
            for bar, value in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:g}", ha="center", va="bottom", fontsize=8)
        fig.suptitle("Logical complexity of selected evolved circuits", y=1.02)
        fig.tight_layout()
        files = _save_paper_figure(fig, figures_dir, "fig4_circuit_complexity")
        caption = "Figure 4. Gate count, two-qubit gate count, and estimated logical depth of the selected circuits. These are simulator-level structural estimates, not transpiled device metrics or measured hardware cost."
        captions.append(caption)
        manifest["figures"].append({"stem": "fig4_circuit_complexity", "caption": caption, "files": files})

    # Figure 5: independent finite-shot sampling repeats and kernel estimation error.
    shot_path = run_dir / "finite_shot_repeats.csv"
    if shot_path.exists():
        shots = pd.read_csv(shot_path)
        if not shots.empty and "circuit_mode" in shots.columns:
            fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.1))
            modes = [m for m in ("accuracy", "hardware_aware") if m in set(shots["circuit_mode"].astype(str))]
            labels = ["Performance objective" if m == "accuracy" else "Logical-complexity-aware" for m in modes]
            acc_col = "finite_shot_validation_accuracy"
            for index, mode in enumerate(modes):
                subset = shots[shots["circuit_mode"].astype(str) == mode]
                values = pd.to_numeric(subset[acc_col], errors="coerce").dropna().to_numpy(float)
                if len(values):
                    jitter = np.linspace(-0.08, 0.08, len(values)) if len(values) > 1 else np.array([0.0])
                    axes[0].scatter(index + jitter, values, s=20, alpha=0.75, color="#4477AA" if mode == "accuracy" else "#CC6677", edgecolor="white", linewidth=0.4)
                    axes[0].errorbar(index, float(np.mean(values)), yerr=float(np.std(values, ddof=1)) if len(values) > 1 else 0.0, fmt="D", color="#222222", capsize=3, markersize=5, zorder=4)
                    ideal_col = "ideal_validation_accuracy"
                    if ideal_col in subset.columns and subset[ideal_col].notna().any():
                        axes[0].hlines(float(subset[ideal_col].dropna().iloc[0]), index - 0.28, index + 0.28, color="#666666", linewidth=1.2, linestyles="--")
                rmse_col = "validation_kernel_rmse"
                rmse_values = pd.to_numeric(subset[rmse_col], errors="coerce").dropna().to_numpy(float) if rmse_col in subset.columns else np.array([])
                if len(rmse_values):
                    jitter = np.linspace(-0.08, 0.08, len(rmse_values)) if len(rmse_values) > 1 else np.array([0.0])
                    axes[1].scatter(index + jitter, rmse_values, s=20, alpha=0.75, color="#4477AA" if mode == "accuracy" else "#CC6677", edgecolor="white", linewidth=0.4)
                    axes[1].errorbar(index, float(np.mean(rmse_values)), yerr=float(np.std(rmse_values, ddof=1)) if len(rmse_values) > 1 else 0.0, fmt="D", color="#222222", capsize=3, markersize=5, zorder=4)
            axes[0].set_title("Finite-shot validation accuracy")
            axes[0].set_ylabel("Accuracy; diamonds show mean ± SD")
            axes[1].set_title("Validation kernel RMSE")
            axes[1].set_ylabel("RMSE against ideal kernel")
            for ax in axes:
                ax.set_xticks(np.arange(len(modes)), labels)
                ax.grid(axis="y", alpha=0.2); ax.set_axisbelow(True)
            fig.suptitle(f"Finite-shot sampling ({int(shots['shots_per_kernel_entry'].max()) if 'shots_per_kernel_entry' in shots.columns and shots['shots_per_kernel_entry'].notna().any() else FINITE_SHOTS} shots per kernel entry)", y=1.02)
            fig.tight_layout()
            files = _save_paper_figure(fig, figures_dir, "fig5_finite_shot_robustness")
            caption = "Figure 5. Independent finite-shot sampling repeats for the selected evolved circuits. Points represent repeats and diamonds show mean ± standard deviation; dashed horizontal segments denote ideal validation accuracy. This experiment models sampling variability only, not gate or readout noise."
            captions.append(caption)
            manifest["figures"].append({"stem": "fig5_finite_shot_robustness", "caption": caption, "files": files})

    # Figure 6: the retained PCA variance, documenting the four-feature encoding.
    metadata_path = run_dir / "datasets" / "dataset_metadata.json"
    if metadata_path.exists():
        try:
            meta = json.loads(metadata_path.read_text(encoding="utf-8"))
            ratios = np.asarray(meta.get("pca_explained_variance_ratio", []), dtype=float)
            if ratios.size:
                fig, ax = plt.subplots(figsize=(6.8, 4.2))
                components = np.arange(1, len(ratios) + 1)
                bars = ax.bar(components, ratios, color="#4477AA", label="Individual component")
                cumulative = np.cumsum(ratios)
                ax.plot(components, cumulative, marker="o", color="#CC6677", label="Cumulative variance")
                for bar, value in zip(bars, ratios):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.012, f"{value:.2f}", ha="center", fontsize=8)
                ax.set_xticks(components)
                ax.set_xlabel("Principal component")
                ax.set_ylabel("Explained variance ratio")
                ax.set_ylim(0, min(1.05, max(0.5, float(cumulative[-1]) + 0.15)))
                ax.set_title(f"Variance retained by four-component PCA ({cumulative[-1]:.1%})")
                ax.grid(axis="y", alpha=0.2); ax.set_axisbelow(True); ax.legend(frameon=False)
                fig.tight_layout()
                files = _save_paper_figure(fig, figures_dir, "fig6_pca_explained_variance")
                caption = f"Figure 6. Individual and cumulative variance explained by the four principal components used for quantum feature encoding. The first four components explain {cumulative[-1]:.1%} of variance in the training-fitted PCA transformation."
                captions.append(caption)
                manifest["figures"].append({"stem": "fig6_pca_explained_variance", "caption": caption, "files": files})
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            log_message(f"Skipped PCA variance chart: {safe_error_message(exc)}")

    manifest["figure_count"] = len(manifest["figures"])
    save_json(figures_dir / "figure_manifest.json", manifest)
    (figures_dir / "figure_captions.md").write_text("\n\n".join(f"**{cap.split('.')[0]}.**{cap.split('.', 1)[1]}" for cap in captions) + "\n", encoding="utf-8")
    report = [
        "# Local workflow results — figure bundle",
        "",
        f"Metrics source: `{data_source}`",
        "Dataset: Breast Cancer Wisconsin Diagnostic (scikit-learn built-in).",
        "Split: stratified 60/20/20; standardization and PCA were fitted on training data only.",
        "Quantum feature encoding: four PCA-derived angles represented as Qiskit QuantumCircuit objects and evaluated by Qiskit Statevector.",
        "Qiskit: used for QuantumCircuit construction, exact Statevector simulation, circuit depth/gate counts, and OpenQASM export. The classical kernel SVM is implemented with scikit-learn.",
        "Execution: local Qiskit Statevector only; no cloud jobs, credentials, device noise, or physical-QPU performance are claimed.",
        "Finite-shot chart: binomial sampling of ideal Qiskit-statevector kernel probabilities only; it does not model gate or readout errors.",
        f"Genetic-algorithm results use {len(GA_SEEDS)} fixed seeds ({', '.join(map(str, GA_SEEDS))}); the tables/figures report mean and sample standard deviation across seeds where available.",
        "Reporting note: repeated seeds quantify search variability on one fixed train/validation/test split; they do not quantify variability across independent dataset splits. Avoid claims of superiority without additional robustness checks.",
        "",
        "## Generated figures",
        "",
    ]
    for item in manifest["figures"]:
        report.append(f"- `{item['stem']}.pdf` / `.png` — {item['caption']}")
    (figures_dir / "paper_results_notes.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    log_message(f"Generated {manifest['figure_count']} paper-oriented figures in {figures_dir} (PDF and 300-dpi PNG).")
    log_message(f"Paper table saved to {run_dir / 'paper_results_table.csv'}")

def analyse_results(run_dir: Path) -> None:
    session_dir = common.ACTIVE_SESSION_DIR
    if session_dir is None:
        raise RuntimeError("No active session is selected.")
    frames = []
    for path in sorted((session_dir / "actions").glob("*/metrics.csv")):
        try:
            frame = pd.read_csv(path)
            if frame.empty or "experiment" not in frame.columns:
                continue
            action_name = path.parent.name
            number_match = re.match(r"(\d+)_", action_name)
            frame.insert(0, "action_number", int(number_match.group(1)) if number_match else -1)
            frame.insert(1, "action", action_name)
            frames.append(frame)
        except (OSError, pd.errors.ParserError) as exc:
            logging.warning("Skipping unreadable metrics file %s: %s", path.name, safe_error_message(exc))
    if not frames:
        log_message("No saved metrics found in this session yet. Run an experiment first.")
        return

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined.to_csv(run_dir / "aggregate_metrics_history.csv", index=False)
    model_history = combined[~combined["experiment"].astype(str).str.startswith("finite_shot_")].copy()
    session_latest = model_history.sort_values(["action_number"]).drop_duplicates(subset=["experiment"], keep="last")
    session_latest.to_csv(run_dir / "session_latest_model_comparison.csv", index=False)

    # Prefer this full workflow's metrics. Option 9 falls back to the latest complete workflow's rows.
    source_path = run_dir / "metrics.csv"
    source_frame = pd.DataFrame()
    source_label = str(source_path)
    if source_path.exists():
        try:
            candidate = pd.read_csv(source_path)
            experiments = set(candidate.get("experiment", pd.Series(dtype=str)).dropna().astype(str))
            if {"genetic_algorithm_accuracy", "genetic_algorithm_hardware_aware", "qiskit_zz_inspired_feature_map_kernel"}.issubset(experiments):
                source_frame = candidate
        except (OSError, pd.errors.ParserError):
            source_frame = pd.DataFrame()
    if source_frame.empty:
        for path in sorted((session_dir / "actions").glob("*/metrics.csv"), reverse=True):
            try:
                candidate = pd.read_csv(path)
                experiments = set(candidate.get("experiment", pd.Series(dtype=str)).dropna().astype(str))
                required = {"genetic_algorithm_accuracy", "genetic_algorithm_hardware_aware", "qiskit_zz_inspired_feature_map_kernel"}
                if required.issubset(experiments):
                    source_frame = candidate
                    source_label = str(path)
                    break
            except (OSError, pd.errors.ParserError):
                continue
    if not source_frame.empty:
        current_models = source_frame[~source_frame["experiment"].astype(str).str.startswith("finite_shot_")].copy()
        current_models = current_models.drop_duplicates(subset=["experiment"], keep="last")
        current_models.to_csv(run_dir / "latest_model_comparison.csv", index=False)
        print("\nCURRENT COMPLETE-WORKFLOW MODEL COMPARISON")
        preferred = [name for name in ("experiment", "feature_view", "validation_accuracy", "validation_balanced_accuracy", "validation_f1_macro", "test_accuracy", "test_balanced_accuracy", "test_f1_macro", "test_malignant_recall", "gate_count", "two_qubit_gate_count", "estimated_depth") if name in current_models.columns]
        print(current_models[preferred].to_string(index=False, max_rows=100))
        make_paper_figures(source_frame, run_dir, source_label)
    else:
        log_message("No complete-workflow metrics found yet; saved the session history, but skipped paper figure generation.")
    log_message(f"Full session history saved to {run_dir / 'aggregate_metrics_history.csv'}")
    log_message(f"Latest current-workflow comparison saved to {run_dir / 'latest_model_comparison.csv'}")

def run_complete_local_workflow(run_dir: Path) -> None:
    """Run the complete local Qiskit workflow, including the paper figure bundle."""
    _qiskit_imports()
    log_message("Stage 1/10: Qiskit local simulator self-tests")
    run_local_self_tests(run_dir)
    log_message("Stage 2/10: preparing the dataset")
    prepare_dataset(run_dir)
    log_message("Stage 3/10: classical baselines (four PCA features and all 30 original features)")
    run_classical_baselines(run_dir)
    log_message("Stage 4/10: Qiskit ZZ-inspired QML baseline")
    run_qml_baseline(run_dir)
    log_message("Stage 5/10: fixed-split evolutionary performance search")
    evolve_circuits(run_dir, mode="accuracy")
    log_message("Stage 6/10: fixed-split logical-complexity-aware search")
    evolve_circuits(run_dir, mode="hardware_aware")
    log_message("Stage 7/10: finite-shot sampling robustness check")
    run_finite_shot_evaluation(run_dir)
    log_message("Stage 8/10: aggregate the initial holdout results and generate exploratory figures")
    analyse_results(run_dir)
    log_message("Stage 9/10: repeated stratified 5-fold outer-CV study with inner validation across all six datasets")
    run_multi_dataset_research(run_dir, dataset_ids=ALL_RESEARCH_DATASET_IDS)
    log_message("Stage 10/10: per-dataset and cross-dataset tables, figures, protocols, and circuit inventory saved")
    log_message("Local workflow complete. Review cross_dataset_model_summary.csv, cross_dataset_circuit_inventory.csv, and each datasets/<dataset_id>/repeated_cv/ folder. No remote jobs were submitted.")

def audit_saved_runs_for_paper(action_dir: Path) -> None:
    """Inventory the active user's saved sessions and create a paper-data bundle.

    This option is read-only with respect to experiment outputs: it does not retrain,
    evolve circuits, recompute predictions, or submit hardware jobs.
    """
    if common.ACTIVE_USER is None or common.ACTIVE_SESSION_DIR is None:
        raise RuntimeError("No active user/session. Log in and resume a session first.")

    import shutil
    from collections import Counter

    project_root = PROJECT_DIR.resolve()
    bundle_dir = common.ACTIVE_SESSION_DIR / "paper_documentation"
    raw_dir = bundle_dir / "raw_artifacts"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    # Only the generated copy is refreshed; original experiment files are not modified.
    if raw_dir.exists():
        shutil.rmtree(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    supported_extensions = {
        ".json", ".jsonl", ".csv", ".tsv", ".txt", ".log", ".md", ".npz",
        ".npy", ".qasm", ".qpy", ".pdf", ".png", ".jpg", ".jpeg", ".svg",
        ".yaml", ".yml", ".toml", ".tex",
    }
    dataset_ids = list(dict.fromkeys(list(ALL_RESEARCH_DATASET_IDS) + [
        "breast_cancer", "iris_setosa_vs_versicolor", "wine_class0_vs_rest",
        "digits_0_vs_1", "digits_3_vs_8", "digits_4_vs_9",
    ]))

    # Scan all saved sessions belonging to the logged-in user, not other users.
    try:
        session_roots = list_user_sessions(common.ACTIVE_USER)
    except Exception:
        session_roots = []
    if common.ACTIVE_SESSION_DIR not in session_roots:
        session_roots.append(common.ACTIVE_SESSION_DIR)
    session_roots = sorted({p.resolve() for p in session_roots if p.exists()})

    def is_under(path: Path, parent: Path) -> bool:
        try:
            path.resolve().relative_to(parent.resolve())
            return True
        except (ValueError, OSError):
            return False

    def relative_path(path: Path) -> str:
        try:
            return str(path.resolve().relative_to(project_root))
        except (ValueError, OSError):
            return str(path)

    def category(path: Path) -> str:
        suffix, name = path.suffix.lower(), path.name.lower()
        parts = {part.lower() for part in path.parts}
        if suffix in {".png", ".jpg", ".jpeg", ".svg", ".pdf"} or "figures" in parts:
            return "figure_or_pdf"
        if "circuits" in parts or suffix in {".qasm", ".qpy"} or "genome" in name or "circuit" in name:
            return "circuit_artifact"
        if suffix in {".npz", ".npy"}:
            return "array_or_dataset"
        if suffix == ".csv":
            return "tabular_result"
        if suffix in {".log", ".txt"}:
            return "log_or_text"
        if any(word in name for word in ("protocol", "config", "metadata", "manifest", "environment")):
            return "protocol_or_configuration"
        if suffix in {".json", ".jsonl"}:
            return "json_result_or_metadata"
        return "other_documentation"

    # Exclude generated bundles (from any prior session) and this running action.
    source_files: List[Path] = []
    scan_roots = list(session_roots)
    # Include legacy project-level run folders if the user has older outputs outside sessions/.
    for legacy_root in (PROJECT_DIR / "workspace", PROJECT_DIR / "actions"):
        if legacy_root.exists() and legacy_root.is_dir():
            scan_roots.append(legacy_root.resolve())
    scan_roots = sorted(set(scan_roots))
    for root in scan_roots:
        try:
            for path in root.rglob("*"):
                if not path.is_file() or path.name.startswith(".") or "__pycache__" in path.parts:
                    continue
                if "paper_documentation" in {part.lower() for part in path.parts}:
                    continue
                if is_under(path, action_dir):
                    continue
                if path.suffix.lower() not in supported_extensions and path.name not in {"session.json", "session.log"}:
                    continue
                source_files.append(path.resolve())
        except OSError as exc:
            log_message(f"Could not fully scan {root}: {safe_error_message(exc)}")
    source_files = sorted(set(source_files), key=lambda path: relative_path(path).lower())

    def sha256(path: Path) -> str:
        try:
            if path.stat().st_size > 500 * 1024 * 1024:
                return "SKIPPED_FILE_OVER_500MB"
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            return digest.hexdigest()
        except OSError as exc:
            return "HASH_ERROR:" + type(exc).__name__

    file_rows: List[Dict[str, Any]] = []
    for src in source_files:
        error, copied, digest, dest_rel = "", False, "", ""
        try:
            stat = src.stat()
            rel = src.resolve().relative_to(project_root)
            dst = raw_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied, digest = True, sha256(src)
            dest_rel = str(dst.relative_to(bundle_dir))
            size = stat.st_size
            modified = datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds")
        except Exception as exc:
            size, modified = "", ""
            error = f"{type(exc).__name__}: {exc}"
        file_rows.append({
            "source_path": relative_path(src), "file_name": src.name,
            "extension": src.suffix.lower(), "category": category(src),
            "size_bytes": size, "modified_time_local": modified, "sha256": digest,
            "copy_status": "copied" if copied else "copy_failed",
            "bundle_path": dest_rel, "copy_error": error,
        })

    def read_json(path: Path) -> Optional[Any]:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return None

    # Compact JSON index: nested scalar metadata is extracted; genome/prediction arrays
    # remain intact in raw_artifacts but are intentionally not expanded into the index.
    json_structure_rows: List[Dict[str, Any]] = []
    json_scalar_rows: List[Dict[str, Any]] = []
    skip_keys = {"genome", "actions", "history", "predictions", "oof_predictions", "population", "folds", "tests", "figures", "per_sample", "sample_predictions"}

    def flatten_scalars(obj: Any, prefix: str, rows: List[Tuple[str, Any]], depth: int = 0) -> None:
        if len(rows) >= 300 or depth > 6:
            return
        if isinstance(obj, dict):
            for key, value in obj.items():
                if str(key).lower() in skip_keys:
                    continue
                flatten_scalars(value, f"{prefix}.{key}" if prefix else str(key), rows, depth + 1)
                if len(rows) >= 300:
                    return
        elif isinstance(obj, list):
            if len(obj) <= 25 and all(not isinstance(value, (dict, list)) for value in obj):
                rows.append((prefix, json.dumps(obj, ensure_ascii=False, default=str)))
        elif obj is None or isinstance(obj, (str, int, float, bool)):
            value = obj[:500] + "...[truncated]" if isinstance(obj, str) and len(obj) > 500 else obj
            rows.append((prefix, value))

    json_paths = [path for path in source_files if path.suffix.lower() == ".json"]
    for path in json_paths:
        obj = read_json(path)
        if obj is None:
            json_structure_rows.append({"source_path": relative_path(path), "status": "unreadable_or_invalid", "top_level_type": "unknown", "top_level_keys": "", "scalar_count": 0})
            continue
        keys = list(obj.keys()) if isinstance(obj, dict) else []
        scalar_pairs: List[Tuple[str, Any]] = []
        flatten_scalars(obj, "", scalar_pairs)
        json_structure_rows.append({
            "source_path": relative_path(path), "status": "readable",
            "top_level_type": type(obj).__name__,
            "top_level_keys": json.dumps(keys, ensure_ascii=False), "scalar_count": len(scalar_pairs),
        })
        for field, value in scalar_pairs:
            json_scalar_rows.append({"source_path": relative_path(path), "field_path": field, "value_type": type(value).__name__, "value": value})

    # CSV schema inventory and descriptive statistics by numeric column. Raw CSV copies
    # remain the authoritative values; these summaries are for navigation only.
    csv_schema_rows: List[Dict[str, Any]] = []
    numeric_rows: List[Dict[str, Any]] = []
    dataset_csv_refs: Dict[str, set] = {dataset_id: set() for dataset_id in dataset_ids}
    csv_paths = [path for path in source_files if path.suffix.lower() == ".csv"]
    for path in csv_paths:
        rel = relative_path(path)
        try:
            frame = pd.read_csv(path, low_memory=False)
            cols = list(map(str, frame.columns))
            csv_schema_rows.append({"source_path": rel, "status": "readable", "row_count": len(frame), "column_count": len(cols), "columns": json.dumps(cols, ensure_ascii=False), "error": ""})
            if "dataset_id" in frame.columns:
                values = set(frame["dataset_id"].dropna().astype(str).unique())
                for dataset_id in dataset_ids:
                    if dataset_id in values:
                        dataset_csv_refs[dataset_id].add(rel)
            for col in cols:
                try:
                    values = pd.to_numeric(frame[col], errors="coerce").dropna()
                    if values.empty:
                        continue
                    numeric_rows.append({
                        "source_path": rel, "column": col, "n_numeric": int(values.size),
                        "mean": float(values.mean()), "std": float(values.std(ddof=1)) if values.size > 1 else None,
                        "min": float(values.min()), "q25": float(values.quantile(.25)),
                        "median": float(values.median()), "q75": float(values.quantile(.75)), "max": float(values.max()),
                    })
                except Exception:
                    continue
        except Exception as exc:
            csv_schema_rows.append({"source_path": rel, "status": "unreadable", "row_count": "", "column_count": "", "columns": "", "error": f"{type(exc).__name__}: {exc}"})

    # Session/action history is joined with standalone action.json files where possible.
    action_rows: List[Dict[str, Any]] = []
    for root in session_roots:
        metadata = read_json(root / "session.json")
        metadata = metadata if isinstance(metadata, dict) else {}
        seen_numbers = set()
        recorded = metadata.get("actions", [])
        if isinstance(recorded, list):
            for item in recorded:
                if not isinstance(item, dict):
                    continue
                number = str(item.get("number", ""))
                seen_numbers.add(number)
                action_rows.append({
                    "session_path": relative_path(root), "session_name": metadata.get("name", root.name),
                    "action_number": item.get("number", ""), "option": item.get("option", ""),
                    "action_name": item.get("name", ""), "status": item.get("status", "unknown"),
                    "started_at": item.get("started_at", ""), "finished_at": item.get("finished_at", ""),
                    "output_dir": item.get("output_dir", ""), "error": item.get("error", ""),
                })
        actions_root = root / "actions"
        if actions_root.exists():
            for ap in actions_root.rglob("action.json"):
                if "paper_documentation" in {part.lower() for part in ap.parts} or is_under(ap, action_dir):
                    continue
                item = read_json(ap)
                if not isinstance(item, dict):
                    continue
                number = str(item.get("number", ap.parent.name))
                if number in seen_numbers:
                    continue
                action_rows.append({
                    "session_path": relative_path(root), "session_name": metadata.get("name", root.name),
                    "action_number": item.get("number", ""), "option": item.get("option", ""),
                    "action_name": item.get("name", ap.parent.name), "status": item.get("status", "unknown"),
                    "started_at": item.get("started_at", ""), "finished_at": item.get("finished_at", ""),
                    "output_dir": str(ap.parent.relative_to(root)), "error": item.get("error", ""),
                })
    action_rows.sort(key=lambda row: (str(row["session_path"]), str(row["action_number"])))

    def matching(*parts: str) -> List[Path]:
        needles = [word.lower() for word in parts]
        return [path for path in source_files if any(word in path.name.lower() for word in needles)]

    fold_metrics = matching("fold_metrics.csv")
    oof_predictions = matching("oof_predictions.csv")
    summaries = matching("repeated_cv_summary.csv", "cross_dataset_model_summary.csv", "paper_results_table.csv", "paper_results_table_repeated_cv.csv")
    protocols = [p for p in source_files if p.name.lower() in {"experiment_config.json", "study_protocol.json", "research_run_protocol.json", "cross_dataset_study_protocol.json", "multiple_comparison_plan.md", "confirmatory_inference_protocol.md"}]
    circuits = [p for p in source_files if ("circuits" in {part.lower() for part in p.parts} and p.suffix.lower() in {".json", ".qasm", ".qpy", ".txt"}) or p.suffix.lower() in {".qasm", ".qpy"}]
    classical = matching("paired_qml_vs_classical", "paired_cluster_bootstrap_qml_vs_classical", "paper_results_table", "classical_baseline")
    finite_shot = matching("finite_shot_summary.csv", "finite_shot_repeats.csv")
    inference = matching("cross_dataset_corrected_cv_inference.csv", "paired_cluster_bootstrap", "paired_fold_complexity_comparisons.csv", "multiple_comparison_plan.md", "confirmatory_inference_protocol.md")
    environment = matching("environment.json", "experiment_config.json")
    figure_files = [p for p in source_files if p.suffix.lower() in {".pdf", ".png", ".jpg", ".jpeg", ".svg"} or "figure_manifest.json" in p.name.lower()]
    dataset_meta = matching("dataset_metadata.json", "cross_dataset_metadata.csv", "study_protocol.json", "cross_dataset_study_protocol.json")

    dataset_rows = []
    for dataset_id in dataset_ids:
        paths = set(relative_path(p) for p in source_files if dataset_id.lower() in str(p).lower())
        paths.update(dataset_csv_refs.get(dataset_id, set()))
        dataset_rows.append({
            "dataset_id": dataset_id, "artifact_count": len(paths),
            "has_fold_metrics": any(Path(p).name.lower() == "fold_metrics.csv" for p in paths),
            "has_oof_predictions": any(Path(p).name.lower() == "oof_predictions.csv" for p in paths),
            "has_protocol_or_metadata": any(any(k in Path(p).name.lower() for k in ("protocol", "dataset_metadata")) for p in paths),
            "evidence_paths_sample": json.dumps(sorted(paths)[:12], ensure_ascii=False),
        })

    checklist = [
        ("Saved run/action history", bool(action_rows), "session.json/action.json", "Check completed, failed, cancelled, legacy and duplicate runs before selecting results."),
        ("Experiment configuration / protocol", bool(protocols), "; ".join(relative_path(p) for p in protocols[:5]), "Report seeds, search budget, preprocessing, folds and selection rules."),
        ("Fold-level repeated-CV metrics", bool(fold_metrics), "; ".join(relative_path(p) for p in fold_metrics[:5]), "Use saved fold metrics as the main source for cross-validation results."),
        ("Out-of-fold predictions", bool(oof_predictions), "; ".join(relative_path(p) for p in oof_predictions[:5]), "Use to verify prediction-derived metrics and uncertainty."),
        ("Aggregate performance summaries", bool(summaries), "; ".join(relative_path(p) for p in summaries[:8]), "Verify metric definitions and avoid treating overlapping CV folds as independent samples."),
        ("Classical/QML baseline comparisons", bool(classical), "; ".join(relative_path(p) for p in classical[:8]), "Check comparable splits and leakage-free preprocessing."),
        ("Saved circuit definitions", bool(circuits), "; ".join(relative_path(p) for p in circuits[:8]), "Prefer saved QASM when available; circuit structure alone is not performance evidence."),
        ("Circuit complexity evidence", any("depth" in r["column"].lower() or "gate" in r["column"].lower() for r in numeric_rows) or bool(circuits), "Numeric CSV gate/depth fields and/or circuit files", "Distinguish logical gate/depth proxies from transpiled or physical hardware cost."),
        ("Finite-shot robustness outputs", bool(finite_shot), "; ".join(relative_path(p) for p in finite_shot[:5]), "Do not describe finite-shot sampling as a physical noise experiment unless that was modeled."),
        ("Statistical inference / multiplicity plan", bool(inference), "; ".join(relative_path(p) for p in inference[:8]), "State test assumptions and exploratory/confirmatory status accurately."),
        ("Environment/software provenance", bool(environment), "; ".join(relative_path(p) for p in environment[:8]), "Record actual runtime package versions; configs are not a dependency lockfile."),
        ("Dataset metadata / split protocol", bool(dataset_meta), "; ".join(relative_path(p) for p in dataset_meta[:8]), "Verify sample counts, features, class definitions and preprocessing."),
        ("Candidate figure assets/manifests", bool(figure_files), "; ".join(relative_path(p) for p in figure_files[:8]), "Choose one pipeline figure for the main paper; use tables for quantitative evidence."),
        ("Physical quantum-hardware evidence", any(any(k in p.name.lower() for k in ("hardware_result", "backend_result", "job_receipt", "ibm_job")) for p in source_files), "Requires independently verifiable device/job records", "The program's documented workflow is local Qiskit simulation with remote jobs disabled; do not claim hardware results without proof."),
        ("Related-work references and bibliography", False, "Not inferable from saved run artifacts", "Add and verify scholarly references supporting background and novelty claims."),
        ("Declarations and reproducibility statement", False, "Author review required", "Confirm data/code availability, licensing, funding, conflicts and other required declarations."),
        ("Scientific claim validation", False, "Cannot be certified automatically", "Verify every reported value against the raw saved tables and protocol."),
    ]
    checklist_rows = []
    for title, found, evidence, next_step in checklist:
        if title in {"Related-work references and bibliography", "Declarations and reproducibility statement", "Scientific claim validation"}:
            status = "MANUAL REVIEW"
        elif title == "Physical quantum-hardware evidence":
            status = "FOUND - VERIFY" if found else "NOT EVIDENCED"
        elif title == "Circuit complexity evidence" and found:
            status = "PARTIAL - VERIFY METRICS"
        else:
            status = "FOUND" if found else "NOT FOUND"
        checklist_rows.append({"check": title, "status": status, "evidence": evidence, "recommended_action": next_step})

    def write_csv(name: str, rows: List[Dict[str, Any]]) -> None:
        pd.DataFrame(rows).to_csv(bundle_dir / name, index=False)

    write_csv("file_inventory.csv", file_rows)
    write_csv("json_structure_inventory.csv", json_structure_rows)
    write_csv("json_scalar_fields.csv", json_scalar_rows)
    write_csv("csv_schema_inventory.csv", csv_schema_rows)
    write_csv("csv_numeric_column_summary.csv", numeric_rows)
    write_csv("action_history.csv", action_rows)
    write_csv("dataset_coverage.csv", dataset_rows)
    write_csv("documentation_checklist.csv", checklist_rows)

    candidate_rows = []
    for path in source_files:
        name = path.name.lower()
        use = None
        if name in {"cross_dataset_model_summary.csv", "repeated_cv_summary.csv", "paper_results_table.csv", "paper_results_table_repeated_cv.csv"}:
            use = "Main performance table"
        elif "cross_dataset_metadata" in name or "dataset_metadata" in name:
            use = "Dataset summary / sample counts"
        elif "fold_metrics" in name or "repeat_level_oof_metrics" in name:
            use = "Fold-level robustness / variability"
        elif "paired_qml_vs_classical" in name or "paired_cluster_bootstrap_qml_vs_classical" in name:
            use = "Classical baseline comparison"
        elif "paired_complexity" in name or "complexity_comparisons" in name or "corrected_cv_inference" in name:
            use = "Circuit complexity / statistical inference"
        elif "finite_shot" in name:
            use = "Finite-shot robustness"
        elif "figure_manifest" in name or "circuit_diagrams_manifest" in name:
            use = "Figure/circuit manifest"
        elif name == "experiment_config.json" or "study_protocol" in name or "research_run_protocol" in name:
            use = "Methods / experimental setup"
        if use:
            candidate_rows.append({"suggested_manuscript_use": use, "source_path": relative_path(path), "bundle_copy_path": str(Path("raw_artifacts") / Path(relative_path(path))), "file_name": path.name})
    write_csv("paper_table_source_candidates.csv", candidate_rows)

    statuses = Counter(str(row.get("status", "unknown")).lower() for row in action_rows)
    checklist_counts = Counter(row["status"] for row in checklist_rows)
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "script_version": SCRIPT_VERSION,
        "active_user": common.ACTIVE_USER,
        "active_session": str(common.ACTIVE_SESSION_DIR.resolve()),
        "bundle_output": str(bundle_dir.resolve()),
        "sessions_scanned": len(session_roots),
        "scan_roots_including_legacy": [relative_path(path) for path in scan_roots],
        "source_files_indexed": len(source_files),
        "files_copied": sum(1 for row in file_rows if row["copy_status"] == "copied"),
        "json_files_indexed": len(json_paths), "csv_files_indexed": len(csv_paths),
        "actions_found": len(action_rows), "action_status_counts": dict(statuses),
        "documentation_check_counts": dict(checklist_counts),
        "fold_metrics_files_found": len(fold_metrics), "oof_prediction_files_found": len(oof_predictions),
        "aggregate_summary_files_found": len(summaries), "circuit_files_found": len(circuits),
        "finite_shot_files_found": len(finite_shot), "inference_files_found": len(inference),
        "remote_hardware_not_assumed": True, "automated_certification": False,
        "note": "Evidence inventory only; not a scientific validation or certification that the paper is complete.",
    }
    save_json(bundle_dir / "documentation_readiness.json", summary)
    try:
        shutil.copy2(Path(__file__).resolve(), bundle_dir / "source_code_snapshot.py")
    except OSError as exc:
        log_message(f"Could not save source code snapshot: {safe_error_message(exc)}")

    def bullets(paths: List[Path], limit: int = 10) -> str:
        return "\n".join(f"- `{relative_path(path)}`" for path in paths[:limit]) if paths else "- Not found."

    md = [
        "# Paper Documentation Readiness Audit", "",
        f"Generated: {summary['created_at']}",
        f"Active session: `{relative_path(common.ACTIVE_SESSION_DIR)}`",
        f"Saved sessions scanned: {len(session_roots)}",
        f"Scan roots including legacy workspaces: {len(scan_roots)}",
        f"Files indexed: {len(source_files)}; copied: {summary['files_copied']}", "",
        "## Interpretation and important cautions", "",
        "This checks evidence-file presence; it does not validate scientific correctness, reproduce experiments, or certify paper completeness. Option 19 does not rerun training or prediction.", "",
        "The application is configured for local Qiskit simulation and records remote jobs as disabled. Do not claim physical quantum-hardware evaluation unless separate device/job records verify it.", "",
        "## Checklist", "",
        "| Check | Status | Available evidence | Recommended next step |", "|---|---|---|---|",
    ]
    def mdcell(value: Any) -> str:
        return str(value if value is not None else "").replace("|", "\\|").replace("\n", " ")
    for row in checklist_rows:
        md.append("| " + " | ".join(mdcell(row[k]) for k in ("check", "status", "evidence", "recommended_action")) + " |")
    md.extend([
        "", "## Key outputs found", "", "### Fold-level metrics", bullets(fold_metrics),
        "", "### Out-of-fold predictions", bullets(oof_predictions),
        "", "### Aggregate summaries", bullets(summaries),
        "", "### Circuit files", bullets(circuits),
        "", "### Statistical inference files", bullets(inference),
        "", "## Output files", "",
        "- `file_inventory.csv`: inventory and SHA-256 for source artifacts.",
        "- `action_history.csv`: recorded action statuses across the user's saved sessions.",
        "- `dataset_coverage.csv`: dataset-level artifact coverage; presence is not validation.",
        "- `csv_schema_inventory.csv`: CSV row counts and column names.",
        "- `csv_numeric_column_summary.csv`: navigation-only descriptive statistics; not a replacement for raw result files.",
        "- `json_structure_inventory.csv` and `json_scalar_fields.csv`: JSON metadata index; genomes and large prediction arrays remain in raw copies.",
        "- `documentation_checklist.csv` and `documentation_readiness.json`: audit summaries.",
        "- `paper_table_source_candidates.csv`: likely source files for manuscript tables.",
        "- `raw_artifacts/`: copies of saved artifacts preserving their `sessions/...` paths.",
        "- `source_code_snapshot.py`: the exact running script snapshot.",
        "", "## Manual tasks before submission", "",
        "1. Verify all numbers against the raw CSVs and document which run(s) were selected.",
        "2. Exclude failed, cancelled, legacy or duplicate runs using an explicit rule.",
        "3. Confirm dataset sample counts, target/class definitions, preprocessing and split protocol from saved metadata.",
        "4. Check statistical assumptions, cross-validation dependence, multiple comparisons and baseline fairness.",
        "5. Add scholarly references, declarations, limitations, and code/data availability information.",
        "6. If no separate hardware job records exist, describe the study as simulation-only and remove unsupported hardware claims.",
        "", "## Recorded action status counts", "", json.dumps(dict(statuses), indent=2), "",
    ])
    (bundle_dir / "documentation_readiness.md").write_text("\n".join(md), encoding="utf-8")

    readme = (
        "# Paper documentation bundle\n\n"
        f"Generated by option 19 of `{Path(__file__).name}`.\n\n"
        f"- Saved sessions scanned: {len(session_roots)}\n"
        f"- Files indexed: {len(source_files)}\n"
        f"- Artifacts copied: {summary['files_copied']}\n"
        f"- Recorded actions: {len(action_rows)}\n"
        f"- CSV files: {len(csv_paths)}\n- JSON files: {len(json_paths)}\n\n"
        "Start with `documentation_readiness.md`, `documentation_checklist.csv`, `action_history.csv`, `dataset_coverage.csv`, and `paper_table_source_candidates.csv`.\n\n"
        "`raw_artifacts/` contains source copies. `source_code_snapshot.py` preserves the running script. No training, circuit evolution, prediction, or remote submission was performed. Numeric column summaries are for navigation only; use the raw outputs for calculations. File presence does not prove correctness or completeness.\n"
    )
    (bundle_dir / "README.md").write_text(readme, encoding="utf-8")

    log_message(f"Paper documentation audit finished. Bundle: {bundle_dir.resolve()}")
    log_message(f"Indexed {len(source_files)} files from {len(session_roots)} saved sessions; copied {summary['files_copied']} artifacts.")
    log_message("No model training or prediction was performed. Review documentation_readiness.md for gaps.")
    print("\nPAPER DOCUMENTATION AUDIT COMPLETE")
    print(f"Output folder: {bundle_dir.resolve()}")
    print(f"Saved sessions scanned: {len(session_roots)} | Files indexed: {len(source_files)} | Artifacts copied: {summary['files_copied']}")
    print(f"Fold metrics: {len(fold_metrics)} | OOF predictions: {len(oof_predictions)} | Summary tables: {len(summaries)}")
    print("Read paper_documentation/documentation_readiness.md and documentation_checklist.csv.")
    print("This is an evidence inventory, not certification that the project or paper is complete.")

def _find_complete_saved_qml_cv_study(user_dir: Path, dataset_id: str) -> Tuple[Path, float]:
    """Find the newest saved repeated-CV study with all four genome families and folds."""
    required_modes = (
        "performance", "complexity_aware",
        "random_search_performance", "random_search_complexity_aware",
    )
    expected_folds = [
        (repeat_number, fold_number)
        for repeat_number in range(1, RESEARCH_CV_REPEATS + 1)
        for fold_number in range(1, RESEARCH_CV_SPLITS + 1)
    ]
    candidates: List[Tuple[float, Path]] = []
    if not user_dir.exists():
        raise FileNotFoundError(f"User session directory not found: {user_dir}")
    for saved_session in user_dir.iterdir():
        actions_dir = saved_session / "actions"
        if not actions_dir.is_dir():
            continue
        for action_dir in actions_dir.iterdir():
            study_dir = action_dir / "datasets" / dataset_id / "repeated_cv"
            circuits_dir = study_dir / "circuits"
            if not circuits_dir.is_dir() or not (study_dir / "fold_metrics.csv").is_file():
                continue
            protocol_path = study_dir / "study_protocol.json"
            if protocol_path.is_file():
                try:
                    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
                    outer_cv = str(protocol.get("outer_cv", ""))
                    if outer_cv and f"{RESEARCH_CV_SPLITS} folds x {RESEARCH_CV_REPEATS} repeats" not in outer_cv:
                        continue
                except (OSError, json.JSONDecodeError):
                    continue
            complete = True
            for repeat_number, fold_number in expected_folds:
                for mode in required_modes:
                    path = circuits_dir / f"repeat_{repeat_number}_fold_{fold_number}_{mode}.json"
                    if not path.is_file():
                        complete = False
                        break
                if not complete:
                    break
            if complete:
                mtimes = [
                    p.stat().st_mtime for p in (
                        study_dir / "fold_metrics.csv",
                        study_dir / "repeated_cv_summary.csv",
                        circuits_dir,
                    ) if p.exists()
                ]
                candidates.append((max(mtimes) if mtimes else study_dir.stat().st_mtime, study_dir))
    if not candidates:
        raise FileNotFoundError(
            f"No complete saved repeated-CV circuit study for {dataset_id} was found under {user_dir}. "
            "Run the multi-dataset study (option 14) and the additional digit-pair study (option 16) first."
        )
    candidates.sort(key=lambda pair: pair[0], reverse=True)
    return candidates[0][1], candidates[0][0]

def _improvement_classical_candidates(seed: int) -> List[Tuple[str, str, Dict[str, Any]]]:
    """Compact, prespecified grid for fair inner-validation tuning of classical models."""
    specs: List[Tuple[str, str, Dict[str, Any]]] = []
    for feature_view in ("original", "pca4"):
        for C_value in (0.01, 0.1, 1.0, 10.0, 100.0):
            specs.append(("logistic_regression", feature_view, {"C": C_value}))
        for C_value in (0.1, 1.0, 10.0):
            for gamma in ("scale", 0.01, 0.1, 1.0):
                specs.append(("rbf_svm", feature_view, {"C": C_value, "gamma": gamma}))
        for hidden in ((32, 16), (64, 32)):
            for alpha in (0.001, 0.01):
                specs.append(("mlp", feature_view, {"hidden_layer_sizes": list(hidden), "alpha": alpha}))
        for max_features in ("sqrt", 0.5):
            for leaf in (1, 3):
                specs.append(("random_forest", feature_view, {"max_features": max_features, "min_samples_leaf": leaf}))
    return specs

def _make_improvement_classical_model(family: str, params: Dict[str, Any], seed: int):
    """Construct a single classical pipeline/model using only explicit grid parameters."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    if family == "logistic_regression":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=float(params["C"]), max_iter=4000, random_state=seed),
        )
    if family == "rbf_svm":
        return make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", C=float(params["C"]), gamma=params["gamma"]),
        )
    if family == "mlp":
        return make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=tuple(int(v) for v in params["hidden_layer_sizes"]),
                alpha=float(params["alpha"]), max_iter=500, early_stopping=True,
                n_iter_no_change=15, validation_fraction=0.15,
                random_state=seed,
            ),
        )
    if family == "random_forest":
        return RandomForestClassifier(
            n_estimators=150,
            max_features=params["max_features"],
            min_samples_leaf=int(params["min_samples_leaf"]),
            random_state=seed,
            n_jobs=-1,
        )
    raise ValueError(f"Unknown classical model family: {family}")

def run_qml_improvement_sweep(run_dir: Path) -> None:
    """Tune SVC C for saved quantum circuits and tune classical baselines on identical validation folds.

    No new genomes are evolved. Saved fold-specific genomes are reused, and each outer test fold
    remains untouched until all C/model-family choices have been made on that fold's validation set.
    """
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.model_selection import RepeatedStratifiedKFold

    if common.ACTIVE_USER_DIR is None:
        raise RuntimeError("No active account directory. Log in and resume your research session first.")
    C_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
    quantum_modes = (
        ("performance", "genetic_algorithm_performance"),
        ("complexity_aware", "genetic_algorithm_complexity_aware"),
        ("random_search_performance", "random_search_performance"),
        ("random_search_complexity_aware", "random_search_complexity_aware"),
        ("zz_kernel", "qiskit_zz_inspired_feature_map_kernel"),
    )
    study_out = run_dir / "improvement_sweep"
    study_out.mkdir(parents=True, exist_ok=True)
    trial_rows: List[Dict[str, Any]] = []
    fold_metric_rows: List[Dict[str, Any]] = []
    prediction_rows: List[Dict[str, Any]] = []
    source_rows: List[Dict[str, Any]] = []
    dataset_summaries: List[Dict[str, Any]] = []
    dataset_ids = tuple(ALL_RESEARCH_DATASET_IDS)

    protocol = {
        "title": "Controlled QML hyperparameter improvement sweep",
        "goal": "Test whether tuning quantum-kernel SVC regularization improves held-out predictive performance and compare against tuned classical baselines.",
        "quantum_models": [name for _, name in quantum_modes],
        "classical_families": ["logistic_regression", "rbf_svm", "mlp", "random_forest"],
        "quantum_C_grid": list(C_GRID),
        "classical_selection": "Inner-validation selection maximizing 0.5 * balanced accuracy + 0.5 * macro-F1; classical candidates include original features and four-component PCA views.",
        "outer_cv": f"RepeatedStratifiedKFold({RESEARCH_CV_SPLITS} folds x {RESEARCH_CV_REPEATS} repeats)",
        "inner_validation": "Reuse the existing study's per-fold split recipe: 75/25 stratified split within outer training data.",
        "preprocessing": "Same local dataset loaders and deterministic outer splits; scaling, PCA and angle scaling are fit using inner training data only. Outer test labels are never used for selection.",
        "saved_circuits": "Reuses the saved fold-specific GA/random-search genomes; no new genetic evolution is performed in this option.",
        "metrics": ["accuracy", "balanced_accuracy", "macro-F1"],
        "paired_uncertainty": "Per-dataset paired sample-cluster bootstrap intervals for tuned QML minus tuned classical model; exploratory across multiple datasets/models.",
        "quantum_execution": "Local Qiskit Statevector simulation only; no QPU jobs or hardware advantage is inferred.",
        "interpretation_guardrail": "A predictive score improvement under classical statevector simulation is not, by itself, a computational quantum advantage.",
    }
    save_json(study_out / "experiment_protocol.json", protocol)
    log_message("QML improvement sweep: reusing saved fold-specific circuits; no genetic algorithm rerun.")
    log_message("For each outer fold, quantum C and classical model choices are made on inner validation data only.")

    for dataset_index, dataset_id in enumerate(dataset_ids, start=1):
        try:
            source_study, source_mtime = _find_complete_saved_qml_cv_study(common.ACTIVE_USER_DIR, dataset_id)
        except FileNotFoundError as exc:
            log_message(f"Skipping dataset {dataset_id}: {safe_error_message(exc)}")
            continue
        source_rows.append({
            "dataset_id": dataset_id,
            "source_study_dir": str(source_study),
            "source_fold_metrics": str(source_study / "fold_metrics.csv"),
            "source_study_protocol": str(source_study / "study_protocol.json"),
            "source_modified_timestamp": datetime.fromtimestamp(source_mtime).isoformat(timespec="seconds"),
        })
        X_all, y_all, metadata = load_research_dataset(dataset_id)
        splitter = RepeatedStratifiedKFold(
            n_splits=RESEARCH_CV_SPLITS,
            n_repeats=RESEARCH_CV_REPEATS,
            random_state=RANDOM_STATE,
        )
        log_message(f"Improvement sweep {dataset_index}/{len(dataset_ids)}: {metadata['dataset_name']} ({len(y_all)} samples).")

        for split_number, (outer_train_idx, outer_test_idx) in enumerate(splitter.split(X_all, y_all)):
            repeat_index = split_number // RESEARCH_CV_SPLITS
            fold_index = split_number % RESEARCH_CV_SPLITS
            repeat_number, fold_number = repeat_index + 1, fold_index + 1
            inner_seed = RANDOM_STATE + repeat_index * 100 + fold_index
            fold = _prepare_outer_cv_fold(X_all, y_all, outer_train_idx, outer_test_idx, inner_seed)
            y_test = fold["y_test"]
            context = {
                "dataset_id": dataset_id,
                "dataset_name": metadata["dataset_name"],
                "repeat": repeat_number,
                "fold": fold_number,
                "n_inner_train": len(fold["y_inner"]),
                "n_validation": len(fold["y_val"]),
                "n_outer_train": len(fold["y_outer"]),
                "n_outer_test": len(y_test),
                "source_study_dir": str(source_study),
            }

            # Load the exact genome selected for this repeat/fold in the original study.
            for mode_file, base_model_name in quantum_modes:
                if mode_file == "zz_kernel":
                    genome = None
                else:
                    genome_path = source_study / "circuits" / f"repeat_{repeat_number}_fold_{fold_number}_{mode_file}.json"
                    genome_payload = json.loads(genome_path.read_text(encoding="utf-8"))
                    if genome_payload.get("outer_test_used_during_selection") is True:
                        raise ValueError(f"Saved genome indicates outer-test selection: {genome_path}")
                    genome = normalize_genome(genome_payload["genome"])

                if mode_file == "zz_kernel":
                    X_inner_states = states_for_data(fold["X_inner_angles"], zz_baseline=True)
                    X_val_states = states_for_data(fold["X_val_angles"], zz_baseline=True)
                    X_outer_states = states_for_data(fold["X_outer_angles"], zz_baseline=True)
                    X_test_states = states_for_data(fold["X_test_angles"], zz_baseline=True)
                    complexity_circuit = build_zz_feature_map_circuit(np.zeros(N_QUBITS, dtype=float))
                    counts = complexity_circuit.count_ops()
                    complexity = {
                        "gate_count": int(complexity_circuit.size()),
                        "two_qubit_gate_count": int(counts.get("cx", 0) + counts.get("cz", 0)),
                        "estimated_depth": int(complexity_circuit.depth()),
                    }
                else:
                    X_inner_states = states_for_data(fold["X_inner_angles"], genome=genome)
                    X_val_states = states_for_data(fold["X_val_angles"], genome=genome)
                    X_outer_states = states_for_data(fold["X_outer_angles"], genome=genome)
                    X_test_states = states_for_data(fold["X_test_angles"], genome=genome)
                    complexity = genome_complexity(genome)

                K_inner = fidelity_kernel(X_inner_states)
                K_val = fidelity_kernel(X_val_states, X_inner_states)
                K_outer = fidelity_kernel(X_outer_states)
                K_test = fidelity_kernel(X_test_states, X_outer_states)
                c_trials: Dict[float, Dict[str, Any]] = {}
                for C_value in C_GRID:
                    start = time.perf_counter()
                    candidate = __import__("sklearn.svm", fromlist=["SVC"]).SVC(kernel="precomputed", C=float(C_value))
                    candidate.fit(K_inner, fold["y_inner"])
                    val_pred = candidate.predict(K_val)
                    vals = classification_metrics(fold["y_val"], val_pred, "validation")
                    score = 0.5 * vals["validation_balanced_accuracy"] + 0.5 * vals["validation_f1_macro"]
                    c_trials[float(C_value)] = {**vals, "validation_selection_score": float(score), "validation_prediction": val_pred}
                    trial_rows.append({
                        **context, "model_family": base_model_name, "feature_view": "four PCA-derived quantum angles",
                        "C": float(C_value), "gamma": "precomputed fidelity kernel",
                        **{k: v for k, v in vals.items()}, "validation_selection_score": float(score),
                        "trial_elapsed_seconds": float(time.perf_counter() - start),
                    })
                best_C = max(C_GRID, key=lambda cv: (c_trials[float(cv)]["validation_selection_score"], c_trials[float(cv)]["validation_balanced_accuracy"], c_trials[float(cv)]["validation_f1_macro"]))
                best_val = c_trials[float(best_C)]
                for variant, chosen_C in ((f"qml_fixedC1_{base_model_name}", 1.0), (f"qml_tuned_{base_model_name}", float(best_C))):
                    final_model = __import__("sklearn.svm", fromlist=["SVC"]).SVC(kernel="precomputed", C=float(chosen_C))
                    final_model.fit(K_outer, fold["y_outer"])
                    test_pred = final_model.predict(K_test)
                    if variant.startswith("qml_tuned_"):
                        selection_metrics = {k: v for k, v in best_val.items() if k.startswith("validation_") and k != "validation_prediction"}
                        reported_C = float(best_C)
                    else:
                        selection_metrics = {k: v for k, v in c_trials[1.0].items() if k.startswith("validation_") and k != "validation_prediction"}
                        reported_C = 1.0
                    metric_row = {
                        **context, "model": variant, "model_type": "quantum_kernel_svc",
                        "base_quantum_method": base_model_name, "selected_C": reported_C,
                        "best_validation_C_for_method": float(best_C), **complexity,
                        **selection_metrics,
                    }
                    metric_row.update(classification_metrics(y_test, test_pred, "test"))
                    fold_metric_rows.append(metric_row)
                    for sample_id, actual, predicted in zip(outer_test_idx, y_test, test_pred):
                        prediction_rows.append({
                            "dataset_id": dataset_id, "dataset_name": metadata["dataset_name"],
                            "repeat": repeat_number, "fold": fold_number, "sample_index": int(sample_id),
                            "model": variant, "y_true": int(actual), "y_pred": int(predicted),
                        })

            # Tune classical family, hyperparameters and feature view on the same inner validation fold.
            feature_views = {
                "original": (fold["X_inner_raw"], fold["X_val_raw"], fold["X_outer_raw"], fold["X_test_raw"]),
                "pca4": (fold["X_inner_pca"], fold["X_val_pca"], fold["X_outer_pca"], fold["X_test_pca"]),
            }
            best_classical = None
            for family, feature_view, params in _improvement_classical_candidates(inner_seed):
                X_inner, X_val, _, _ = feature_views[feature_view]
                start = time.perf_counter()
                model = _make_improvement_classical_model(family, params, inner_seed)
                convergence_warning = False
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", ConvergenceWarning)
                    model.fit(X_inner, fold["y_inner"])
                convergence_warning = any(issubclass(w.category, ConvergenceWarning) for w in caught)
                val_pred = model.predict(X_val)
                vals = classification_metrics(fold["y_val"], val_pred, "validation")
                score = 0.5 * vals["validation_balanced_accuracy"] + 0.5 * vals["validation_f1_macro"]
                trial = {
                    **context, "model_family": family, "feature_view": feature_view,
                    "params_json": json.dumps(params, sort_keys=True),
                    **vals, "validation_selection_score": float(score),
                    "convergence_warning": convergence_warning,
                    "trial_elapsed_seconds": float(time.perf_counter() - start),
                }
                trial_rows.append(trial)
                ordering = (float(score), float(vals["validation_balanced_accuracy"]), float(vals["validation_f1_macro"]))
                if best_classical is None or ordering > best_classical["ordering"]:
                    best_classical = {
                        "ordering": ordering, "family": family, "feature_view": feature_view,
                        "params": params, "validation": vals,
                        "validation_selection_score": float(score), "convergence_warning": convergence_warning,
                    }

            if best_classical is None:
                raise RuntimeError(f"No classical candidates were evaluated for {dataset_id}, repeat {repeat_number}, fold {fold_number}.")
            selected_view = best_classical["feature_view"]
            X_inner, X_val, X_outer, X_test = feature_views[selected_view]
            final_classical = _make_improvement_classical_model(best_classical["family"], best_classical["params"], inner_seed)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                final_classical.fit(X_outer, fold["y_outer"])
            final_convergence_warning = any(issubclass(w.category, ConvergenceWarning) for w in caught)
            classical_test_pred = final_classical.predict(X_test)
            classical_row = {
                **context, "model": "classical_tuned_best", "model_type": best_classical["family"],
                "selected_classical_family": best_classical["family"], "selected_feature_view": selected_view,
                "selected_params_json": json.dumps(best_classical["params"], sort_keys=True),
                "selected_C": best_classical["params"].get("C", np.nan),
                "validation_selection_score": best_classical["validation_selection_score"],
                "convergence_warning": bool(best_classical["convergence_warning"] or final_convergence_warning),
            }
            classical_row.update(best_classical["validation"])
            classical_row.update(classification_metrics(y_test, classical_test_pred, "test"))
            fold_metric_rows.append(classical_row)
            for sample_id, actual, predicted in zip(outer_test_idx, y_test, classical_test_pred):
                prediction_rows.append({
                    "dataset_id": dataset_id, "dataset_name": metadata["dataset_name"],
                    "repeat": repeat_number, "fold": fold_number, "sample_index": int(sample_id),
                    "model": "classical_tuned_best", "y_true": int(actual), "y_pred": int(predicted),
                })

            log_message(
                f"{dataset_id} repeat {repeat_number}/{RESEARCH_CV_REPEATS} fold {fold_number}/{RESEARCH_CV_SPLITS}: "
                f"selected classical={best_classical['family']} ({selected_view}); "
                f"tuned quantum C values evaluated."
            )

            # Checkpoint incremental outputs so interruption does not lose completed folds.
            pd.DataFrame(fold_metric_rows).to_csv(study_out / "fold_metrics.csv", index=False)
            pd.DataFrame(prediction_rows).to_csv(study_out / "oof_predictions.csv", index=False)
            pd.DataFrame(trial_rows).to_csv(study_out / "hyperparameter_trials.csv", index=False)

    if not source_rows:
        raise FileNotFoundError(
            "No complete saved circuit studies were available. Run options 14 and 16 first, then rerun option 20."
        )
    pd.DataFrame(source_rows).to_csv(study_out / "source_studies.csv", index=False)
    fold_frame = pd.DataFrame(fold_metric_rows)
    pred_frame = pd.DataFrame(prediction_rows)
    trials_frame = pd.DataFrame(trial_rows)
    fold_frame.to_csv(study_out / "fold_metrics.csv", index=False)
    pred_frame.to_csv(study_out / "oof_predictions.csv", index=False)
    trials_frame.to_csv(study_out / "hyperparameter_trials.csv", index=False)

    # Pooled out-of-fold metrics and mean +/- SD over the three repeat-level OOF estimates.
    summary_rows = []
    for (dataset_id, model_name), group in pred_frame.groupby(["dataset_id", "model"]):
        meta_name = str(group["dataset_name"].iloc[0])
        row: Dict[str, Any] = {
            "dataset_id": dataset_id, "dataset_name": meta_name,
            "model": model_name, "n_oof_predictions": int(len(group)),
            "n_unique_samples": int(group["sample_index"].nunique()),
        }
        row.update(classification_metrics(group["y_true"].to_numpy(dtype=int), group["y_pred"].to_numpy(dtype=int), "pooled"))
        repeat_metric_rows = []
        for repeat_number, rep_group in group.groupby("repeat"):
            metrics = classification_metrics(rep_group["y_true"].to_numpy(dtype=int), rep_group["y_pred"].to_numpy(dtype=int), "repeat")
            repeat_metric_rows.append(metrics)
        for metric in ("accuracy", "balanced_accuracy", "f1_macro"):
            values = [float(item[f"repeat_{metric}"]) for item in repeat_metric_rows]
            row[f"mean_repeat_{metric}"] = float(np.mean(values))
            row[f"sd_repeat_{metric}"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        mean_c = fold_frame[(fold_frame["dataset_id"] == dataset_id) & (fold_frame["model"] == model_name)]
        if "selected_C" in mean_c and mean_c["selected_C"].notna().any():
            row["mean_selected_C"] = float(mean_c["selected_C"].dropna().mean())
            row["sd_selected_C"] = float(mean_c["selected_C"].dropna().std(ddof=1)) if mean_c["selected_C"].dropna().shape[0] > 1 else 0.0
        summary_rows.append(row)
    summary_frame = pd.DataFrame(summary_rows).sort_values(["dataset_id", "model"])
    summary_frame.to_csv(study_out / "per_dataset_summary.csv", index=False)

    # Paired sample-cluster bootstrap comparisons; these are exploratory across many models/tasks.
    paired_frames = []
    tuned_qml_names = [f"qml_tuned_{model_name}" for _, model_name in quantum_modes]
    for dataset_index, (dataset_id, group) in enumerate(pred_frame.groupby("dataset_id"), start=1):
        available = set(group["model"].astype(str))
        compare = [name for name in tuned_qml_names if name in available]
        if "classical_tuned_best" in available and compare:
            paired = _paired_cluster_bootstrap_differences(
                group[["repeat", "fold", "sample_index", "model", "y_true", "y_pred"]],
                baseline="classical_tuned_best",
                n_bootstrap=min(RESEARCH_CV_BOOTSTRAPS, 1000),
                seed=RANDOM_STATE + 12000 + dataset_index,
                models_to_compare=compare,
            )
            paired.insert(0, "dataset_id", dataset_id)
            paired.insert(1, "dataset_name", str(group["dataset_name"].iloc[0]))
            paired["inference_note"] = "Exploratory paired sample-cluster bootstrap; multiplicity across tasks/models not confirmatory."
            paired_frames.append(paired)
    if paired_frames:
        pd.concat(paired_frames, ignore_index=True).to_csv(study_out / "paired_tuned_qml_vs_tuned_classical.csv", index=False)

    # Directly test whether C tuning changed each quantum variant compared with its fixed-C=1 counterpart.
    tuning_frames = []
    tuned_comparison_pairs = [
        (f"qml_fixedC1_{name}", f"qml_tuned_{name}") for _, name in quantum_modes
    ]
    for dataset_index, (dataset_id, group) in enumerate(pred_frame.groupby("dataset_id"), start=1):
        for pair_index, (fixed_name, tuned_name) in enumerate(tuned_comparison_pairs, start=1):
            available = set(group["model"].astype(str))
            if fixed_name not in available or tuned_name not in available:
                continue
            pair = _paired_cluster_bootstrap_differences(
                group[["repeat", "fold", "sample_index", "model", "y_true", "y_pred"]],
                baseline=fixed_name,
                n_bootstrap=min(RESEARCH_CV_BOOTSTRAPS, 1000),
                seed=RANDOM_STATE + 13000 + dataset_index * 100 + pair_index,
                models_to_compare=[tuned_name],
            )
            pair.insert(0, "dataset_id", dataset_id)
            pair["comparison"] = "C-tuned minus fixed C=1"
            tuning_frames.append(pair)
    if tuning_frames:
        pd.concat(tuning_frames, ignore_index=True).to_csv(study_out / "paired_quantum_tuning_effect.csv", index=False)

    notes = [
        "# QML improvement sweep: interpretation guide",
        "",
        "This run reuses saved, fold-specific circuit genomes from the original repeated-CV studies. It does not evolve new circuit structures.",
        "For each outer fold, quantum-kernel SVC regularization C is selected from 0.01, 0.1, 1, 10 and 100 using inner-validation balanced accuracy and macro-F1. The same fixed C=1 quantum variants are evaluated alongside their tuned-C versions.",
        "Classical models are tuned on the same validation partitions over logistic regression, RBF-SVM, MLP and random forest candidates, with original-feature and PCA-4 feature views. The selected classical candidate is refit on the outer-training partition and evaluated on the untouched outer-test partition.",
        "Start with `per_dataset_summary.csv`, then examine `paired_tuned_qml_vs_tuned_classical.csv` and `paired_quantum_tuning_effect.csv`. Values are fractions in CSV files; multiply by 100 for percentages.",
        "A positive quantum-minus-classical balanced-accuracy difference whose paired 95% cluster-bootstrap interval is above zero is a candidate predictive signal, not confirmatory evidence: multiple tasks/models are screened, and the comparisons are exploratory.",
        "A high score from local statevector simulation is not, by itself, a computational quantum advantage. The current experiment tests predictive performance and tuning, not asymptotic runtime or quantum resource advantage.",
        "If one task shows a promising result, the next step is to freeze that task/model/metric and run a prespecified confirmatory experiment with an independently held-out test set and a fair, tuned classical comparator.",
    ]
    (study_out / "README_interpretation.md").write_text("\n".join(notes) + "\n", encoding="utf-8")
    log_message(f"QML improvement sweep completed. Results directory: {study_out.resolve()}")
    log_message(f"Saved {len(fold_frame)} fold/model rows, {len(pred_frame)} OOF predictions, and {len(trials_frame)} validation hyperparameter trials.")
    log_message("Review per_dataset_summary.csv and paired_tuned_qml_vs_tuned_classical.csv before drawing conclusions.")
