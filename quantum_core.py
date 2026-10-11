"""Shared settings, dataset helpers, and quantum circuit operations."""
from __future__ import annotations
import sys

# The older helpers use this shared namespace for run paths and settings.
common = sys.modules[__name__]


# Shared settings and file helpers
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple
import json
import logging
import re
import uuid
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / "results"
SESSIONS_DIR = RESULTS_DIR / "sessions"
ACCOUNTS_FILE = RESULTS_DIR / "accounts.json"
ACTIVE_USER = None
ACTIVE_USER_DIR = None
ACTIVE_SESSION_DIR = None
WORKSPACE_DIR = RESULTS_DIR / "workspace"
DATA_CACHE = WORKSPACE_DIR / "processed_dataset.npz"
LATEST_GENOME_FILE = WORKSPACE_DIR / "latest_best_genome.json"


SCRIPT_VERSION = "8.2-qiskit-circuit-pdf-paper-audit"

RANDOM_STATE = 42

N_QUBITS = 4

GA_POPULATION_SIZE = 8

GA_GENERATIONS = 8

GA_SEEDS = (42, 43, 44)

GA_ELITE_COUNT = 2

GA_MUTATION_RATE = 0.22

MAX_GENOME_LENGTH = 12

MIN_GENOME_LENGTH = 4

HARDWARE_DEPTH_PENALTY = 0.05

HARDWARE_TWO_QUBIT_PENALTY = 0.08

FINITE_SHOTS = 1024

FINITE_SHOT_REPEATS = 10

RESEARCH_CV_SPLITS = 5

RESEARCH_CV_REPEATS = 3

RESEARCH_CV_BOOTSTRAPS = 2000

RESEARCH_GA_SEED_BASE = 42000

COMPLEXITY_BALANCED_ACCURACY_TOLERANCE = 0.02

RESEARCH_PROTOCOL_VERSION = "multidataset-v5-hard-pairs-corrected-inference"

GATE_NAMES = ("RX", "RY", "RZ", "CNOT", "CZ")

ROTATION_GATES = ("RX", "RY", "RZ")

TWO_QUBIT_GATES = ("CNOT", "CZ")

ROTATION_SCALES = (0.5, 1.0, 2.0)

RESEARCH_MODEL_NAMES = {
    "logistic_regression_pca4": "Logistic regression (PCA-4)",
    "mlp_pca4": "MLP (PCA-4)",
    "logistic_regression_original30": "Logistic regression (original features)",
    "mlp_original30": "MLP (original features)",
    "rbf_svm_pca4": "RBF-SVM (PCA-4)",
    "rbf_svm_original30": "RBF-SVM (original features)",
    "random_forest_pca4": "Random forest (PCA-4)",
    "random_forest_original30": "Random forest (original features)",
    "qiskit_zz_inspired_feature_map_kernel": "Standard Qiskit quantum kernel",
    "genetic_algorithm_performance": "Evolved QML (balanced-performance objective)",
    "genetic_algorithm_complexity_aware": "Evolved QML (accuracy-constrained complexity)",
    "random_search_performance": "Random-search QML (performance selection)",
    "random_search_complexity_aware": "Random-search QML (accuracy-constrained complexity)",
}

RESEARCH_DATASET_IDS = ("breast_cancer", "iris_setosa_vs_versicolor", "wine_class0_vs_rest", "digits_0_vs_1")

ADDITIONAL_HARD_DIGIT_DATASET_IDS = ("digits_3_vs_8", "digits_4_vs_9")

ALL_RESEARCH_DATASET_IDS = RESEARCH_DATASET_IDS + ADDITIONAL_HARD_DIGIT_DATASET_IDS

class ExperimentCancelled(Exception):
    pass

class SecretRedactingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = safe_error_message(record.getMessage())
        record.args = ()
        return True

def configure_run_logger(session_dir: Path) -> None:
    logger = logging.getLogger()
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(session_dir / "session.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    handler.addFilter(SecretRedactingFilter())
    logger.addHandler(handler)

def log_message(message: str) -> None:
    print(message)
    logging.info(message)

def save_metrics(run_dir: Path, rows: Sequence[Dict[str, Any]], filename: str = "metrics.csv") -> None:
    """Append metric rows so the full-workflow option retains every stage."""
    if not rows:
        return
    path = run_dir / filename
    new_frame = pd.DataFrame(list(rows))
    if path.exists():
        try:
            old_frame = pd.read_csv(path)
            new_frame = pd.concat([old_frame, new_frame], ignore_index=True, sort=False)
        except (OSError, pd.errors.ParserError):
            pass
    new_frame.to_csv(path, index=False)

def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, default=json_default)

def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

def safe_error_message(error: Any) -> str:
    message = str(error)
    message = re.sub(r"eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+", "[REDACTED_TOKEN]", message)
    message = re.sub(r"(?i)(bearer\s+)[A-Za-z0-9._~-]+", r"\1[REDACTED_TOKEN]", message)
    return message[:1200]

def user_sessions_dir(username: str) -> Path:
    path = SESSIONS_DIR / username
    path.mkdir(parents=True, exist_ok=True)
    return path

def create_session(username: str, session_name: str) -> Path:
    safe_name = re.sub(r"[^a-zA-Z0-9_-]+", "_", session_name.lower()).strip("_") or "research_session"
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    session_dir = user_sessions_dir(username) / f"{timestamp}_{safe_name}_{uuid.uuid4().hex[:6]}"
    session_dir.mkdir(parents=True, exist_ok=False)
    (session_dir / "actions").mkdir()
    (session_dir / "logs").mkdir()
    (session_dir / "workspace").mkdir()
    metadata = {"username": username, "name": session_name, "session_id": session_dir.name, "created_at": datetime.now().isoformat(timespec="seconds"), "last_opened_at": datetime.now().isoformat(timespec="seconds"), "status": "active", "action_count": 0, "actions": []}
    save_json(session_dir / "session.json", metadata)
    (session_dir / "session.log").write_text("Session created.\n", encoding="utf-8")
    print(f"\nNew session created: {session_dir.resolve()}")
    return session_dir

def activate_session(username: str, session_dir: Path) -> None:
    global ACTIVE_USER, ACTIVE_USER_DIR, ACTIVE_SESSION_DIR, WORKSPACE_DIR, DATA_CACHE, LATEST_GENOME_FILE
    ACTIVE_USER = username
    ACTIVE_USER_DIR = SESSIONS_DIR / username
    ACTIVE_SESSION_DIR = session_dir
    WORKSPACE_DIR = session_dir / "workspace"
    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    DATA_CACHE = WORKSPACE_DIR / "processed_dataset.npz"
    LATEST_GENOME_FILE = WORKSPACE_DIR / "latest_best_genome.json"
    metadata = read_session_metadata(session_dir)
    metadata["last_opened_at"] = datetime.now().isoformat(timespec="seconds")
    metadata["status"] = "active"
    save_json(session_dir / "session.json", metadata)
    configure_run_logger(session_dir)
    log_message(f"Opened session: {metadata.get('name', session_dir.name)}")

def read_session_metadata(session_dir: Path) -> Dict[str, Any]:
    path = session_dir / "session.json"
    if not path.exists():
        raise FileNotFoundError(f"Session metadata is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))

def list_user_sessions(username: str) -> List[Path]:
    return sorted((path for path in user_sessions_dir(username).iterdir() if path.is_dir() and (path / "session.json").exists()), reverse=True)

def create_action_directory(session_dir: Path, action_number: int, label: str) -> Path:
    safe_name = re.sub(r"[^a-zA-Z0-9_-]+", "_", label.lower()).strip("_")
    action_dir = session_dir / "actions" / f"{action_number:04d}_{safe_name}"
    action_dir.mkdir(parents=True, exist_ok=False)
    for name in ("figures", "circuits", "logs", "datasets"):
        (action_dir / name).mkdir()
    return action_dir

def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, prefix: str = "") -> Dict[str, float]:
    """Binary metrics with generic class-0/class-1 names and legacy aliases."""
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score

    prefix = f"{prefix}_" if prefix else ""
    class0_precision = float(precision_score(y_true, y_pred, pos_label=0, zero_division=0))
    class0_recall = float(recall_score(y_true, y_pred, pos_label=0, zero_division=0))
    class1_recall = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    return {
        f"{prefix}accuracy": float(accuracy_score(y_true, y_pred)),
        f"{prefix}balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        f"{prefix}f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        f"{prefix}class0_precision": class0_precision,
        f"{prefix}class0_recall": class0_recall,
        f"{prefix}class1_recall": class1_recall,
        # Keep these historical columns for backward compatibility with the original breast-cancer study.
        f"{prefix}malignant_precision": class0_precision,
        f"{prefix}malignant_recall": class0_recall,
        f"{prefix}benign_recall": class1_recall,
    }


# Dataset loading and preprocessing
import json
import os
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

def run_environment_check(run_dir: Path) -> None:
    """Record system information and whether optional packages are available."""
    import importlib.util

    packages = ["qiskit", "numpy", "pandas", "sklearn", "matplotlib"]
    package_status = {name: importlib.util.find_spec(name) is not None for name in packages}
    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "packages_available": package_status,
        "note": "Reported CPU count is the OS-visible logical CPU count, not a guarantee of exclusive allocation. This program is local-only.",
    }
    save_json(run_dir / "environment.json", report)
    print(json.dumps(report, indent=2))
    log_message("Environment report saved to environment.json")

def prepare_dataset(run_dir: Path) -> None:
    """Create fixed stratified splits and save raw, PCA, and quantum-angle views."""
    from sklearn.datasets import load_breast_cancer
    from sklearn.decomposition import PCA
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler, StandardScaler

    dataset = load_breast_cancer()
    X_raw, y = np.asarray(dataset.data, dtype=float), np.asarray(dataset.target, dtype=int)
    X_train_raw, X_temp_raw, y_train, y_temp = train_test_split(X_raw, y, test_size=0.40, random_state=RANDOM_STATE, stratify=y)
    X_val_raw, X_test_raw, y_val, y_test = train_test_split(X_temp_raw, y_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_temp)

    raw_scaler = StandardScaler().fit(X_train_raw)
    X_train_std, X_val_std, X_test_std = raw_scaler.transform(X_train_raw), raw_scaler.transform(X_val_raw), raw_scaler.transform(X_test_raw)
    pca = PCA(n_components=N_QUBITS, random_state=RANDOM_STATE).fit(X_train_std)
    X_train_pca, X_val_pca, X_test_pca = pca.transform(X_train_std), pca.transform(X_val_std), pca.transform(X_test_std)

    angle_scaler = MinMaxScaler(feature_range=(-np.pi, np.pi)).fit(X_train_pca)
    X_train = angle_scaler.transform(X_train_pca)
    X_val = angle_scaler.transform(X_val_pca)
    X_test = angle_scaler.transform(X_test_pca)

    payload = {
        "schema_version": np.asarray([2], dtype=int),
        "X_train": X_train, "y_train": y_train, "X_val": X_val, "y_val": y_val, "X_test": X_test, "y_test": y_test,
        "X_train_pca": X_train_pca, "X_val_pca": X_val_pca, "X_test_pca": X_test_pca,
        "X_train_raw": X_train_raw, "X_val_raw": X_val_raw, "X_test_raw": X_test_raw,
        "feature_names": np.asarray([f"PC{i + 1}" for i in range(N_QUBITS)]),
        "raw_feature_names": np.asarray(dataset.feature_names), "target_names": np.asarray(dataset.target_names),
        "pca_explained_variance_ratio": pca.explained_variance_ratio_,
    }
    common.WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(common.DATA_CACHE, **payload)
    np.savez_compressed(run_dir / "datasets" / "processed_dataset.npz", **payload)
    metadata = {
        "dataset": "Breast Cancer Wisconsin Diagnostic (scikit-learn built-in)",
        "original_samples": int(len(y)), "original_features": int(X_raw.shape[1]), "quantum_qubits": N_QUBITS,
        "split": {"train": len(y_train), "validation": len(y_val), "test": len(y_test)},
        "class_names_by_label": {str(i): str(name) for i, name in enumerate(dataset.target_names)},
        "positive_class_for_malignancy_metrics": "malignant (label 0)",
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "pca_total_explained_variance_ratio": float(pca.explained_variance_ratio_.sum()),
        "random_state": RANDOM_STATE,
        "preprocessing": ["Stratified 60/20/20 split", "Raw features retained for full-feature classical references", "StandardScaler and PCA fitted on training data only", "PCA scores retained for controlled four-feature classical baselines", "Quantum angles scaled using training-only PCA score ranges; validation/test values are not clipped"],
        "feature_views": {"X_*_raw": "Original 30 features for practical classical references", "X_*_pca": "Four PCA scores for a feature-matched classical comparison", "X_*": "Four PCA scores mapped to rotation angles for Qiskit QuantumCircuit feature maps"},
        "schema_version": 2,
    }
    save_json(run_dir / "datasets" / "dataset_metadata.json", metadata)
    save_json(common.WORKSPACE_DIR / "dataset_metadata.json", metadata)
    log_message(f"Prepared dataset and cached it at {common.DATA_CACHE}")
    print(json.dumps(metadata, indent=2))

def load_dataset(run_dir: Optional[Path] = None) -> Dict[str, np.ndarray]:
    """Load cached data, rebuilding legacy caches that lack the current schema."""
    required = {"X_train", "X_val", "X_test", "X_train_pca", "X_val_pca", "X_test_pca", "X_train_raw", "X_val_raw", "X_test_raw", "y_train", "y_val", "y_test"}
    refresh = not common.DATA_CACHE.exists()
    if not refresh:
        try:
            with np.load(common.DATA_CACHE, allow_pickle=False) as cached:
                refresh = not required.issubset(set(cached.files))
        except (OSError, ValueError):
            refresh = True
    if refresh:
        if run_dir is None:
            raise FileNotFoundError("Processed data is missing or uses an old schema. Select option 2 to prepare the dataset.")
        log_message("Dataset cache is missing or outdated; rebuilding it with the current schema.")
        prepare_dataset(run_dir)
    with np.load(common.DATA_CACHE, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}

def load_research_dataset(dataset_id: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Load a deterministic, local scikit-learn benchmark and define class-0/class-1 semantics."""
    from sklearn.datasets import load_breast_cancer, load_digits, load_iris, load_wine

    if dataset_id == "breast_cancer":
        dataset = load_breast_cancer()
        X = np.asarray(dataset.data, dtype=float)
        y = np.asarray(dataset.target, dtype=int)  # sklearn: 0=malignant, 1=benign
        meta = {
            "dataset_id": dataset_id, "dataset_name": "Breast Cancer Wisconsin Diagnostic",
            "task_description": "malignant (class 0) versus benign (class 1)",
            "class0_name": "Malignant", "class1_name": "Benign",
            "source": "scikit-learn built-in load_breast_cancer; no external download",
        }
    elif dataset_id == "iris_setosa_vs_versicolor":
        dataset = load_iris()
        keep = np.isin(dataset.target, [0, 1])
        X = np.asarray(dataset.data[keep], dtype=float)
        y = np.asarray(dataset.target[keep], dtype=int)  # 0=setosa, 1=versicolor
        meta = {
            "dataset_id": dataset_id, "dataset_name": "Iris (setosa vs. versicolor)",
            "task_description": "Iris setosa (class 0) versus Iris versicolor (class 1); virginica excluded",
            "class0_name": "Setosa", "class1_name": "Versicolor",
            "source": "scikit-learn built-in load_iris; no external download",
        }
    elif dataset_id == "wine_class0_vs_rest":
        dataset = load_wine()
        X = np.asarray(dataset.data, dtype=float)
        y = (np.asarray(dataset.target, dtype=int) != 0).astype(int)
        meta = {
            "dataset_id": dataset_id, "dataset_name": "Wine (class 0 vs. classes 1/2)",
            "task_description": "wine cultivar class 0 (class 0) versus cultivars 1 and 2 (class 1)",
            "class0_name": "Wine class 0", "class1_name": "Wine classes 1/2",
            "source": "scikit-learn built-in load_wine; no external download",
        }
    elif dataset_id == "digits_0_vs_1":
        dataset = load_digits()
        keep = np.isin(dataset.target, [0, 1])
        X = np.asarray(dataset.data[keep], dtype=float)
        y = (np.asarray(dataset.target[keep], dtype=int) != 0).astype(int)
        meta = {
            "dataset_id": dataset_id, "dataset_name": "Handwritten digits (0 vs. 1)",
            "task_description": "handwritten digit 0 (class 0) versus handwritten digit 1 (class 1); other digits excluded",
            "class0_name": "Digit 0", "class1_name": "Digit 1",
            "source": "scikit-learn built-in load_digits; no external download",
        }
    elif dataset_id in {"digits_3_vs_8", "digits_4_vs_9"}:
        dataset = load_digits()
        pair = (3, 8) if dataset_id == "digits_3_vs_8" else (4, 9)
        keep = np.isin(dataset.target, pair)
        X = np.asarray(dataset.data[keep], dtype=float)
        y = (np.asarray(dataset.target[keep], dtype=int) == pair[1]).astype(int)
        pair_label = f"{pair[0]} vs. {pair[1]}"
        meta = {
            "dataset_id": dataset_id, "dataset_name": f"Handwritten digits ({pair_label})",
            "task_description": f"handwritten digit {pair[0]} (class 0) versus handwritten digit {pair[1]} (class 1); other digits excluded",
            "class0_name": f"Digit {pair[0]}", "class1_name": f"Digit {pair[1]}",
            "source": "scikit-learn built-in load_digits; no external download",
        }
    else:
        raise ValueError(f"Unknown research dataset {dataset_id!r}. Supported datasets: {', '.join(ALL_RESEARCH_DATASET_IDS)}")

    if X.ndim != 2 or X.shape[1] < N_QUBITS:
        raise ValueError(f"Dataset {dataset_id} must have at least {N_QUBITS} numerical features for four-component PCA.")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError(f"Dataset {dataset_id} did not yield both binary classes.")
    class_counts = {str(int(label)): int(np.sum(y == label)) for label in (0, 1)}
    meta.update({"n_samples": int(len(y)), "n_features": int(X.shape[1]), "class_counts": class_counts,
                 "n_class0": int(np.sum(y == 0)), "n_class1": int(np.sum(y == 1))})
    return X, y, meta


# Quantum circuits and kernel helpers
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

def _qiskit_imports():
    try:
        from qiskit import QuantumCircuit, qasm2
        from qiskit.quantum_info import Operator, Statevector
    except ImportError as exc:
        raise RuntimeError(
            "Qiskit is required for this QML workflow. Install it with: "
            "python -m pip install qiskit"
        ) from exc
    return QuantumCircuit, Statevector, Operator, qasm2

def build_zz_feature_map_circuit(x: np.ndarray):
    """Build the ZZ-inspired data-encoding circuit using a real Qiskit circuit."""
    QuantumCircuit, _, _, _ = _qiskit_imports()
    x = np.asarray(x, dtype=float)
    if x.shape != (N_QUBITS,):
        raise ValueError(f"Expected {N_QUBITS} encoded features, received shape {x.shape}.")
    circuit = QuantumCircuit(N_QUBITS, name="zz_inspired_feature_map")
    for qubit in range(N_QUBITS):
        circuit.h(qubit)
    for qubit in range(N_QUBITS):
        circuit.rz(2.0 * float(x[qubit]), qubit)
    for qubit in range(N_QUBITS - 1):
        angle = 2.0 * (np.pi - float(x[qubit])) * (np.pi - float(x[qubit + 1]))
        circuit.cx(qubit, qubit + 1)
        circuit.rz(angle, qubit + 1)
        circuit.cx(qubit, qubit + 1)
    return circuit

def build_genome_circuit(x: np.ndarray, genome: Sequence[Dict[str, Any]]):
    """Translate an evolved gate sequence into an executable Qiskit QuantumCircuit."""
    QuantumCircuit, _, _, _ = _qiskit_imports()
    x = np.asarray(x, dtype=float)
    if x.shape != (N_QUBITS,):
        raise ValueError(f"Expected {N_QUBITS} encoded features, received shape {x.shape}.")
    circuit = QuantumCircuit(N_QUBITS, name="evolved_qml_feature_map")
    for qubit in range(N_QUBITS):
        circuit.h(qubit)
    for operation in normalize_genome(genome):
        name, qubit = operation["gate"], int(operation["q"])
        if name in ROTATION_GATES:
            theta = float(operation["scale"]) * float(x[int(operation["feature"])])
            getattr(circuit, name.lower())(theta, qubit)
        elif name == "CNOT":
            circuit.cx(qubit, int(operation["target"]))
        elif name == "CZ":
            circuit.cz(qubit, int(operation["target"]))
        else:
            raise ValueError(f"Unsupported gate in genome: {name}")
    return circuit

def state_for_zz_feature_map(x: np.ndarray) -> np.ndarray:
    """Simulate one ZZ-inspired feature-map circuit with Qiskit Statevector."""
    _, Statevector, _, _ = _qiskit_imports()
    return np.asarray(Statevector.from_instruction(build_zz_feature_map_circuit(x)).data, dtype=complex)

def state_for_genome(x: np.ndarray, genome: Sequence[Dict[str, Any]]) -> np.ndarray:
    """Simulate one evolved Qiskit circuit and return its statevector."""
    _, Statevector, _, _ = _qiskit_imports()
    return np.asarray(Statevector.from_instruction(build_genome_circuit(x, genome)).data, dtype=complex)

def states_for_data(X: np.ndarray, genome: Optional[Sequence[Dict[str, Any]]] = None,
                    zz_baseline: bool = False) -> np.ndarray:
    """Evaluate a data matrix by executing its Qiskit circuits on Statevector."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != N_QUBITS:
        raise ValueError(f"Expected a 2D feature matrix with {N_QUBITS} columns; got {X.shape}.")
    if zz_baseline:
        states = [state_for_zz_feature_map(row) for row in X]
    else:
        if genome is None:
            raise ValueError("An evolved genome is required unless zz_baseline=True.")
        states = [state_for_genome(row, genome) for row in X]
    return np.asarray(states, dtype=complex)

def fidelity_kernel(states_a: np.ndarray, states_b: Optional[np.ndarray] = None) -> np.ndarray:
    """Compute the fidelity kernel K(x,z)=|<phi(x)|phi(z)>|^2."""
    if states_b is None:
        states_b = states_a
    overlaps = np.asarray(states_a).conj() @ np.asarray(states_b).T
    return np.clip(np.asarray(np.abs(overlaps) ** 2, dtype=float), 0.0, 1.0)

def _fit_kernel_classifier(X_train: np.ndarray, y_train: np.ndarray,
                           X_eval: np.ndarray, genome: Optional[Sequence[Dict[str, Any]]] = None,
                           zz_baseline: bool = False) -> Tuple[np.ndarray, np.ndarray, Any]:
    from sklearn.svm import SVC
    train_states = states_for_data(X_train, genome=genome, zz_baseline=zz_baseline)
    eval_states = states_for_data(X_eval, genome=genome, zz_baseline=zz_baseline)
    K_train, K_eval = fidelity_kernel(train_states), fidelity_kernel(eval_states, train_states)
    classifier = SVC(kernel="precomputed", C=1.0).fit(K_train, y_train)
    return classifier.predict(K_eval), K_eval, classifier

def random_operation(rng: random.Random) -> Dict[str, Any]:
    gate = rng.choice(GATE_NAMES)
    q = rng.randrange(N_QUBITS)
    if gate in ROTATION_GATES:
        return {"gate": gate, "q": q, "feature": rng.randrange(N_QUBITS), "scale": rng.choice(ROTATION_SCALES)}
    target = rng.randrange(N_QUBITS - 1)
    if target >= q:
        target += 1
    return {"gate": gate, "q": q, "target": target}

def random_genome(rng: random.Random, length: Optional[int] = None) -> List[Dict[str, Any]]:
    if length is None:
        length = rng.randint(MIN_GENOME_LENGTH, MAX_GENOME_LENGTH)
    return [random_operation(rng) for _ in range(length)]

def normalize_genome(genome: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Validate and normalize an operation list loaded from JSON or GA operators."""
    cleaned: List[Dict[str, Any]] = []
    for operation in genome[:MAX_GENOME_LENGTH]:
        gate = str(operation.get("gate", "RY")).upper()
        if gate not in GATE_NAMES:
            gate = "RY"
        q = int(operation.get("q", 0)) % N_QUBITS
        if gate in ROTATION_GATES:
            feature = int(operation.get("feature", 0)) % N_QUBITS
            scale = float(operation.get("scale", 1.0))
            if scale not in ROTATION_SCALES:
                scale = min(ROTATION_SCALES, key=lambda item: abs(item - scale))
            cleaned.append({"gate": gate, "q": q, "feature": feature, "scale": scale})
        else:
            target = int(operation.get("target", (q + 1) % N_QUBITS)) % N_QUBITS
            if target == q:
                target = (q + 1) % N_QUBITS
            cleaned.append({"gate": gate, "q": q, "target": target})
    while len(cleaned) < MIN_GENOME_LENGTH:
        qubit = len(cleaned) % N_QUBITS
        cleaned.append({"gate": "RY", "q": qubit, "feature": qubit, "scale": 1.0})
    return cleaned

def crossover(parent_a: Sequence[Dict[str, Any]], parent_b: Sequence[Dict[str, Any]],
              rng: random.Random) -> List[Dict[str, Any]]:
    a, b = list(parent_a), list(parent_b)
    cut_a = rng.randrange(len(a) + 1)
    cut_b = rng.randrange(len(b) + 1)
    child = [dict(op) for op in a[:cut_a] + b[cut_b:]]
    if not child:
        child = random_genome(rng)
    return normalize_genome(child)

def mutate(genome: Sequence[Dict[str, Any]], rng: random.Random,
           mutation_rate: float = 0.25) -> List[Dict[str, Any]]:
    child = [dict(op) for op in genome]
    if rng.random() < mutation_rate and len(child) < MAX_GENOME_LENGTH:
        child.insert(rng.randrange(len(child) + 1), random_operation(rng))
    if rng.random() < mutation_rate and len(child) > MIN_GENOME_LENGTH:
        del child[rng.randrange(len(child))]
    for index in range(len(child)):
        if rng.random() < mutation_rate:
            child[index] = random_operation(rng)
    return normalize_genome(child)

def genome_complexity(genome: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    """Read circuit size and depth from the actual Qiskit circuit representation."""
    circuit = build_genome_circuit(np.zeros(N_QUBITS, dtype=float), genome)
    counts = circuit.count_ops()
    two_qubit = int(counts.get("cx", 0) + counts.get("cz", 0))
    gate_count = int(circuit.size())
    return {"gate_count": gate_count, "single_qubit_gate_count": gate_count - two_qubit,
            "two_qubit_gate_count": two_qubit, "estimated_depth": int(circuit.depth())}

def tournament_select(population: Sequence[List[Dict[str, Any]]],
                      scores: Sequence[float], rng: random.Random,
                      tournament_size: int = 3) -> List[Dict[str, Any]]:
    candidates = [rng.randrange(len(population)) for _ in range(min(tournament_size, len(population)))]
    winner = max(candidates, key=lambda index: scores[index])
    return [dict(op) for op in population[winner]]

def export_example_qasm(path: Path, genome: Sequence[Dict[str, Any]], x: np.ndarray) -> None:
    """Export a real Qiskit QuantumCircuit as OpenQASM 2 and a readable text drawing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    circuit = build_genome_circuit(x, genome)
    _, _, _, qasm2 = _qiskit_imports()
    path.write_text(qasm2.dumps(circuit) + "\n", encoding="utf-8")
    path.with_suffix(".txt").write_text(str(circuit.draw(output="text")), encoding="utf-8")

def sample_kernel(kernel: np.ndarray, shots: int, rng: np.random.Generator) -> np.ndarray:
    """Sample all-zero counts from ideal kernel probabilities; no physical noise is modeled."""
    if shots <= 0:
        raise ValueError("shots must be positive")
    return rng.binomial(shots, np.clip(kernel, 0.0, 1.0)) / float(shots)

def sample_symmetric_kernel(kernel: np.ndarray, shots: int, rng: np.random.Generator) -> np.ndarray:
    """Sample each symmetric off-diagonal entry once and mirror it exactly."""
    result = np.eye(kernel.shape[0], dtype=float)
    for i in range(kernel.shape[0]):
        for j in range(i + 1, kernel.shape[1]):
            value = rng.binomial(shots, float(np.clip(kernel[i, j], 0.0, 1.0))) / float(shots)
            result[i, j] = result[j, i] = value
    return result
