"""Single entry point for the complete ideal + noisy QML research workflow.

Run ``python main.py`` from the repository root. Every invocation creates a
unique directory under ``results/runs/`` and keeps the run's outputs, logs,
configuration, workspace/cache, and stage metadata inside that directory.
"""

import argparse
import compileall
import importlib.util
import json
import logging
import platform
import sys
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parent
# Allow running from any working directory while keeping outputs project-local.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import quantum_core as common  # noqa: E402

TITLE = "Optimization of Quantum Circuit Designs Using Quantum Machine Learning"
RUNS_ROOT = common.RESULTS_DIR / "runs"


def create_run_directory(results_root: Path = RUNS_ROOT) -> Path:
    """Create a unique, self-contained result directory for one invocation."""
    results_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    while True:
        run_dir = results_root / f"run_{timestamp}_{uuid.uuid4().hex[:8]}"
        try:
            run_dir.mkdir(parents=False, exist_ok=False)
            break
        except FileExistsError:
            continue

    for directory in ("actions", "logs", "workspace"):
        (run_dir / directory).mkdir()

    now = datetime.now().isoformat(timespec="seconds")
    metadata = {
        "project": TITLE,
        "run_id": run_dir.name,
        "created_at": now,
        "finished_at": None,
        "status": "running",
        "workflow": "complete_ideal_research_then_noisy_simulation",
        "action_count": 0,
        "actions": [],
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }
    common.save_json(run_dir / "session.json", metadata)
    common.save_json(run_dir / "run_metadata.json", metadata)
    return run_dir


def activate_run_context(run_dir: Path) -> None:
    """Point legacy research helpers at this run's isolated workspace."""
    common.ACTIVE_USER = "local"
    common.ACTIVE_USER_DIR = RUNS_ROOT
    common.ACTIVE_SESSION_DIR = run_dir
    common.WORKSPACE_DIR = run_dir / "workspace"
    common.DATA_CACHE = common.WORKSPACE_DIR / "processed_dataset.npz"
    common.LATEST_GENOME_FILE = common.WORKSPACE_DIR / "latest_best_genome.json"
    common.WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    common.configure_run_logger(run_dir)


def _read_metadata(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "session.json").read_text(encoding="utf-8"))


def _write_metadata(run_dir: Path, metadata: dict[str, Any]) -> None:
    common.save_json(run_dir / "session.json", metadata)
    common.save_json(run_dir / "run_metadata.json", metadata)


def run_stage(
    run_dir: Path,
    number: int,
    label: str,
    action: Callable[[Path], None],
) -> tuple[Path, bool]:
    """Run one stage in its own folder and record success/failure consistently."""
    action_dir = common.create_action_directory(run_dir, number, label)
    started = datetime.now().isoformat(timespec="seconds")
    record: dict[str, Any] = {
        "number": number,
        "name": label,
        "started_at": started,
        "finished_at": None,
        "status": "running",
        "output_directory": str(action_dir.relative_to(run_dir)),
    }
    metadata = _read_metadata(run_dir)
    metadata["action_count"] = max(int(metadata.get("action_count", 0)), number)
    metadata.setdefault("actions", []).append(record.copy())
    _write_metadata(run_dir, metadata)

    logging.info("Starting stage %s: %s", number, label)
    print(f"\n{'=' * 78}\nSTAGE {number}: {label}\nOutput: {action_dir}\n{'=' * 78}")
    success = False
    try:
        action(action_dir)
        record["status"] = "completed"
        success = True
        logging.info("Completed stage %s: %s", number, label)
    except Exception as exc:  # Preserve trace and let the unified run finish cleanly.
        record["status"] = "failed"
        record["error"] = common.safe_error_message(exc)
        (action_dir / "error.txt").write_text(
            f"Stage failed: {label}\n\n{common.safe_error_message(exc)}\n\n"
            f"Traceback:\n{traceback.format_exc()}",
            encoding="utf-8",
        )
        logging.exception("Stage %s failed: %s", number, label)
        print(f"Stage failed. Details saved to {action_dir / 'error.txt'}")
    finally:
        record["finished_at"] = datetime.now().isoformat(timespec="seconds")
        metadata = _read_metadata(run_dir)
        for item in metadata.get("actions", []):
            if item.get("number") == number:
                item.update(record)
                break
        _write_metadata(run_dir, metadata)
    return action_dir, success


def run_ideal_workflow(action_dir: Path) -> None:
    """Run the existing complete local ideal-simulation workflow."""
    from research import run_complete_local_workflow

    run_complete_local_workflow(action_dir)


def run_noisy_simulation(action_dir: Path, genome_path: Path) -> None:
    """Compare ideal Statevector and noisy Aer kernels using the evolved genome."""
    import numpy as np
    import pandas as pd

    from quantum_core import normalize_genome
    from quantum_core import load_research_dataset
    from optimizer import (
        balanced_subsample,
        build_depolarizing_noise_model,
        compare_ideal_and_noisy_kernels,
        prepare_four_angle_split,
        validate_noise_parameters,
    )

    if not genome_path.is_file():
        raise FileNotFoundError(f"Evolved genome file was not found: {genome_path}")
    payload = json.loads(genome_path.read_text(encoding="utf-8"))
    if "genome" not in payload:
        raise ValueError(f"Genome JSON must include a top-level 'genome' array: {genome_path}")
    genome = normalize_genome(payload["genome"])

    # Fixed, recorded defaults make runs reproducible; the noise is synthetic,
    # illustrative, and not calibrated to an IBM Quantum device.
    dataset_id = "breast_cancer"
    seed = 42
    train_per_class = 12
    test_per_class = 8
    shots = 512
    chunk_size = 256
    single_qubit_error = 0.001
    two_qubit_error = 0.01
    readout_error = 0.01
    validate_noise_parameters(single_qubit_error, two_qubit_error, readout_error)

    X, y, dataset_meta = load_research_dataset(dataset_id)
    split = prepare_four_angle_split(X, y, seed=seed)
    X_train, y_train, train_indices = balanced_subsample(
        split["X_train_angles"], split["y_train"], train_per_class, seed + 10
    )
    X_test, y_test, test_indices = balanced_subsample(
        split["X_test_angles"], split["y_test"], test_per_class, seed + 20
    )
    train_indices = split["train_indices"][train_indices]
    test_indices = split["test_indices"][test_indices]

    noise_model = build_depolarizing_noise_model(
        single_qubit_error=single_qubit_error,
        two_qubit_error=two_qubit_error,
        readout_error=readout_error,
    )
    config = {
        "experiment": "ideal_statevector_vs_noisy_aer_kernel",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": dataset_meta,
        "genome_file": str(genome_path.resolve()),
        "genome_mode": payload.get("mode", "not_recorded"),
        "genome_seed": payload.get("seed"),
        "n_qubits": 4,
        "circuit_operation_count": len(genome),
        "preprocessing": "stratified 60/20/20 split; StandardScaler, PCA-4 and angle MinMaxScaler fit on training data only",
        "train_samples_per_class": train_per_class,
        "test_samples_per_class": test_per_class,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "shots_per_kernel_entry": shots,
        "chunk_size": chunk_size,
        "noise_model": {
            "type": "synthetic gate depolarization plus symmetric readout bit-flip error",
            "single_qubit_depolarizing_probability": single_qubit_error,
            "two_qubit_depolarizing_probability": two_qubit_error,
            "symmetric_readout_bit_flip_probability": readout_error,
            "T1_T2_thermal_relaxation_included": False,
            "backend_calibrated": False,
        },
        "seed": seed,
        "limitations": [
            "Exploratory noisy simulation on a small balanced subsample.",
            "Noise rates are synthetic and are not device calibrated.",
            "These results are not directly comparable to the original six-task repeated-CV results.",
            "No physical quantum hardware was used.",
        ],
    }
    common.save_json(action_dir / "run_config.json", config)
    np.savez_compressed(
        action_dir / "sample_indices.npz",
        train_indices=train_indices,
        test_indices=test_indices,
    )

    rows, artifacts, simulation_metadata = compare_ideal_and_noisy_kernels(
        X_train,
        y_train,
        X_test,
        y_test,
        genome,
        noise_model,
        shots=shots,
        seed=seed,
        chunk_size=chunk_size,
    )
    pd.DataFrame(rows).to_csv(action_dir / "metrics.csv", index=False)
    prediction_columns = {
        key: value for key, value in artifacts.items() if key.endswith("_predictions")
    }
    pd.DataFrame({"y_true": y_test, **prediction_columns}).to_csv(
        action_dir / "predictions.csv", index=False
    )
    np.savez_compressed(
        action_dir / "kernel_matrices.npz",
        **{key: value for key, value in artifacts.items() if key.endswith("_kernel")},
    )
    common.save_json(action_dir / "simulation_metadata.json", simulation_metadata)

    print("\nIDEAL VS NOISY SIMULATION")
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nNoise model is synthetic and not calibrated to quantum hardware.")


def run_dependency_check() -> int:
    """Check compilation and availability of core runtime dependencies."""
    print(f"{TITLE}\nDependency and compilation check\n")
    compile_ok = compileall.compile_dir(str(PROJECT_ROOT), quiet=1, force=True)
    print(f"Python compilation: {'PASS' if compile_ok else 'FAIL'}")
    required = ("numpy", "pandas", "sklearn", "matplotlib", "qiskit", "qiskit_aer", "pylatexenc")
    missing = []
    for module_name in required:
        found = importlib.util.find_spec(module_name) is not None
        print(f"{module_name}: {'available' if found else 'MISSING'}")
        if not found:
            missing.append(module_name)
    if missing:
        print("\nInstall project dependencies with: python -m pip install -r requirements.txt")
        return 1
    print("\nAll runtime dependencies are available.")
    return 0 if compile_ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the complete six-dataset ideal QML research workflow followed by a noisy-kernel "
            "simulation. Each invocation writes to a new results/runs folder."
        )
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="compile Python files and check runtime dependencies without running experiments",
    )
    args = parser.parse_args(argv)
    if args.check:
        return run_dependency_check()

    run_dir = create_run_directory()
    activate_run_context(run_dir)
    common.save_json(
        run_dir / "workflow_config.json",
        {
            "project": TITLE,
            "workflow": "complete_ideal_research_then_noisy_simulation",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "outputs_root": str((common.RESULTS_DIR / "runs").resolve()),
            "stages": [
                "Complete local ideal Statevector research workflow across six binary datasets",
                "Noisy Aer kernel experiment using the newly evolved circuit where available",
            ],
            "run_isolation": "Every invocation gets a unique folder; workspace and cache are run-specific.",
        },
    )
    print(f"\n{TITLE}\n")
    print("One unified workflow: ideal research pipeline, then noisy-kernel evaluation.")
    print(f"All outputs for this invocation will be saved in:\n{run_dir}\n")

    ideal_dir, ideal_ok = run_stage(
        run_dir,
        1,
        "complete_ideal_research_workflow",
        run_ideal_workflow,
    )

    genome_path = ideal_dir / "circuits" / "best_circuit_accuracy.json"
    if not genome_path.is_file():
        # Keep the two-stage workflow runnable even if ideal training failed early;
        # the run metadata records the fallback source transparently.
        genome_path = PROJECT_ROOT / "examples" / "best_circuit_accuracy.json"
        logging.warning(
            "Newly evolved genome is unavailable; noisy stage will use the bundled example genome."
        )
        common.save_json(
            run_dir / "genome_source_fallback.json",
            {
                "reason": "The ideal workflow did not produce a new accuracy-selected genome.",
                "fallback_genome": str(genome_path.resolve()),
            },
        )

    _, noisy_ok = run_stage(
        run_dir,
        2,
        "noisy_kernel_simulation",
        lambda action_dir: run_noisy_simulation(action_dir, genome_path),
    )

    metadata = _read_metadata(run_dir)
    overall_success = ideal_ok and noisy_ok
    metadata["status"] = "completed" if overall_success else "partial_failure"
    metadata["finished_at"] = datetime.now().isoformat(timespec="seconds")
    metadata["ideal_workflow_status"] = "completed" if ideal_ok else "failed"
    metadata["noisy_simulation_status"] = "completed" if noisy_ok else "failed"
    _write_metadata(run_dir, metadata)

    summary_lines = [
        f"# Run summary: {run_dir.name}",
        "",
        f"- Overall status: **{metadata['status']}**",
        f"- Ideal research workflow: **{metadata['ideal_workflow_status']}**",
        f"- Noisy simulation: **{metadata['noisy_simulation_status']}**",
        "- Ideal workflow uses local Qiskit Statevector simulation.",
        "- Noisy workflow uses a synthetic Qiskit Aer gate/readout error model.",
        "- No physical quantum hardware was used.",
        "",
        "## Outputs",
        "",
        "- `actions/0001_complete_ideal_research_workflow/` — ideal research metrics, figures, folds, predictions and circuit files.",
        "- `actions/0002_noisy_kernel_simulation/` — noisy-kernel metrics, predictions, kernels and run configuration.",
        "- `workspace/` — cached data and evolved-genome state used by this run.",
        "- `session.log` — full run log.",
        "- `run_metadata.json` — stage statuses and timestamps.",
    ]
    (run_dir / "README_run.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"\nRun finished with status: {metadata['status']}")
    print(f"All files are in: {run_dir}")
    if not overall_success:
        print("At least one stage failed. Open each failed stage's error.txt and the run's session.log.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
