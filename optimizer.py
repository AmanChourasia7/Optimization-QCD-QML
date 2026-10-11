"""Genetic search and noisy quantum-kernel evaluation."""
from __future__ import annotations

# Genetic search and circuit selection
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
import quantum_core as common
from quantum_core import crossover, export_example_qasm, fidelity_kernel, genome_complexity, mutate, normalize_genome, random_genome, states_for_data, tournament_select
from quantum_core import COMPLEXITY_BALANCED_ACCURACY_TOLERANCE, GA_ELITE_COUNT, GA_GENERATIONS, GA_MUTATION_RATE, GA_POPULATION_SIZE, GA_SEEDS, HARDWARE_DEPTH_PENALTY, HARDWARE_TWO_QUBIT_PENALTY, MAX_GENOME_LENGTH, N_QUBITS, classification_metrics, log_message, save_json, save_metrics
from quantum_core import load_dataset

def evaluate_genome(genome: Sequence[Dict[str, Any]], X_train: np.ndarray, y_train: np.ndarray,
                    X_val: np.ndarray, y_val: np.ndarray, mode: str,
                    cache: Dict[str, Tuple[float, Dict[str, Any]]]) -> Tuple[float, Dict[str, Any]]:
    from sklearn.svm import SVC

    genome = normalize_genome(genome)
    cache_key = json.dumps(genome, sort_keys=True)
    if cache_key in cache:
        return cache[cache_key]

    train_states = states_for_data(X_train, genome=genome)
    val_states = states_for_data(X_val, genome=genome)
    K_train = fidelity_kernel(train_states)
    K_val = fidelity_kernel(val_states, train_states)
    model = SVC(kernel="precomputed", C=1.0)
    model.fit(K_train, y_train)
    predictions = model.predict(K_val)
    validation_metrics = classification_metrics(y_val, predictions, "validation")
    accuracy = validation_metrics["validation_accuracy"]
    complexity = genome_complexity(genome)

    if mode == "hardware_aware":
        depth_penalty = HARDWARE_DEPTH_PENALTY * complexity["estimated_depth"] / (N_QUBITS + MAX_GENOME_LENGTH)
        two_qubit_penalty = HARDWARE_TWO_QUBIT_PENALTY * complexity["two_qubit_gate_count"] / MAX_GENOME_LENGTH
        fitness = accuracy - depth_penalty - two_qubit_penalty
    else:
        fitness = accuracy

    details = {**validation_metrics, "fitness": float(fitness), "mode": mode, **complexity}
    result = (float(fitness), details)
    cache[cache_key] = result
    return result

def evolve_circuits(run_dir: Path, mode: str) -> None:
    """Run a genetic search for each seed, then report mean and SD across seeds."""
    from sklearn.svm import SVC

    data = load_dataset(run_dir)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]
    seed_histories = []
    seed_rows = []
    selected_candidates = []
    mode_offset = 1000 if mode == "hardware_aware" else 0

    for seed in GA_SEEDS:
        rng = random.Random(seed + mode_offset)
        population = [random_genome(rng) for _ in range(GA_POPULATION_SIZE)]
        cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}
        history: List[Dict[str, Any]] = []
        best_genome: Optional[List[Dict[str, Any]]] = None
        best_details: Optional[Dict[str, Any]] = None
        best_fitness = -float("inf")

        for generation in range(GA_GENERATIONS):
            evaluated = [evaluate_genome(genome, X_train, y_train, X_val, y_val, mode, cache) for genome in population]
            scores = [item[0] for item in evaluated]
            details_list = [item[1] for item in evaluated]
            order = sorted(range(len(population)), key=lambda idx: scores[idx], reverse=True)
            generation_best = order[0]
            if scores[generation_best] > best_fitness:
                best_fitness = scores[generation_best]
                best_genome = [dict(op) for op in population[generation_best]]
                best_details = dict(details_list[generation_best])
            history.append({
                "seed": seed, "generation": generation + 1,
                "best_fitness": float(scores[generation_best]),
                "best_validation_accuracy": float(details_list[generation_best]["validation_accuracy"]),
                "mean_fitness": float(np.mean(scores)),
                "best_gate_count": int(details_list[generation_best]["gate_count"]),
                "best_two_qubit_gate_count": int(details_list[generation_best]["two_qubit_gate_count"]),
                "best_estimated_depth": int(details_list[generation_best]["estimated_depth"]),
            })
            next_population = [[dict(op) for op in population[idx]] for idx in order[:GA_ELITE_COUNT]]
            while len(next_population) < GA_POPULATION_SIZE:
                child = mutate(crossover(tournament_select(population, scores, rng), tournament_select(population, scores, rng), rng), rng, mutation_rate=GA_MUTATION_RATE)
                next_population.append(child)
            population = next_population

        assert best_genome is not None and best_details is not None
        seed_histories.extend(history)

        # The test split is only evaluated after that seed's circuit was selected using validation data.
        X_fit, y_fit = np.vstack([X_train, X_val]), np.concatenate([y_train, y_val])
        fit_states = states_for_data(X_fit, genome=best_genome)
        test_states = states_for_data(X_test, genome=best_genome)
        model = SVC(kernel="precomputed", C=1.0).fit(fidelity_kernel(fit_states), y_fit)
        test_metrics = classification_metrics(y_test, model.predict(fidelity_kernel(test_states, fit_states)), "test")
        complexity = genome_complexity(best_genome)
        seed_row = {
            "experiment": f"genetic_algorithm_{mode}", "mode": mode, "seed": int(seed),
            "random_seed_used": int(seed + mode_offset), "fitness": float(best_fitness),
            "population_size": GA_POPULATION_SIZE, "generations": GA_GENERATIONS,
            "n_train": len(y_train), "n_validation": len(y_val), "n_test": len(y_test),
            **{key: value for key, value in best_details.items() if key.startswith("validation_")},
            **test_metrics, **complexity,
        }
        seed_rows.append(seed_row)
        payload = {
            "mode": mode, "seed": int(seed), "random_seed_used": int(seed + mode_offset),
            "genome": best_genome, "selection_metrics": best_details,
            "test_metrics_after_selection": test_metrics, "complexity": complexity,
            "n_train_for_selection": len(y_train), "n_validation_for_selection": len(y_val),
            "n_test": len(y_test), "fit_size_for_final_test": len(y_fit),
            "warning": "The test partition was not used during selection; it is evaluated only after the circuit was selected using validation data.",
        }
        save_json(run_dir / "circuits" / f"best_circuit_{mode}_seed_{seed}.json", payload)
        export_example_qasm(run_dir / "circuits" / f"best_circuit_{mode}_seed_{seed}_example.qasm", best_genome, X_fit[0])
        selected_candidates.append((best_fitness, seed, best_genome, best_details, complexity, payload))
        log_message(f"{mode} evolution | seed {seed} | best validation accuracy={best_details['validation_accuracy']:.4f} | fitness={best_fitness:.4f} | test accuracy={test_metrics['test_accuracy']:.4f}")

    # Keep the candidate with the highest validation fitness for the later finite-shot study.
    # The held-out test metrics are never used to choose this circuit.
    overall_best = max(selected_candidates, key=lambda item: item[0])
    best_fitness, best_seed, best_genome, best_details, best_complexity, best_payload = overall_best
    best_payload["selection_across_seeds"] = {"seeds": list(GA_SEEDS), "selected_seed_by_validation_fitness": int(best_seed), "selection_criterion": "highest validation fitness; test metrics not used"}
    save_json(run_dir / "circuits" / f"best_circuit_{mode}.json", best_payload)
    save_json(common.WORKSPACE_DIR / f"best_circuit_{mode}.json", best_payload)
    save_json(common.LATEST_GENOME_FILE, best_payload)

    seed_frame = pd.DataFrame(seed_rows)
    seed_frame.to_csv(run_dir / f"evolution_seed_metrics_{mode}.csv", index=False)
    history_frame = pd.DataFrame(seed_histories)
    history_frame.to_csv(run_dir / f"evolution_history_{mode}_all_seeds.csv", index=False)
    history_summary = history_frame.groupby("generation").agg(
        best_validation_accuracy=("best_validation_accuracy", "mean"),
        best_validation_accuracy_std=("best_validation_accuracy", "std"),
        best_fitness=("best_fitness", "mean"), best_fitness_std=("best_fitness", "std"),
        mean_fitness=("mean_fitness", "mean"), mean_fitness_std=("mean_fitness", "std"),
    ).reset_index()
    history_summary = history_summary.fillna(0.0)
    history_summary.to_csv(run_dir / f"evolution_history_{mode}.csv", index=False)

    aggregate: Dict[str, Any] = {
        "experiment": f"genetic_algorithm_{mode}", "mode": mode,
        "feature_view": "four PCA-derived angles", "seed_count": len(GA_SEEDS),
        "seeds": ";".join(str(seed) for seed in GA_SEEDS),
        "selected_seed_by_validation_fitness": int(best_seed),
        "population_size": GA_POPULATION_SIZE, "generations": GA_GENERATIONS,
        "n_train": len(y_train), "n_validation": len(y_val), "n_test": len(y_test),
    }
    mean_columns = [
        "fitness", "validation_accuracy", "validation_balanced_accuracy", "validation_f1_macro",
        "validation_malignant_precision", "validation_malignant_recall", "validation_benign_recall",
        "test_accuracy", "test_balanced_accuracy", "test_f1_macro", "test_malignant_precision",
        "test_malignant_recall", "test_benign_recall", "gate_count", "single_qubit_gate_count",
        "two_qubit_gate_count", "estimated_depth",
    ]
    for column in mean_columns:
        if column in seed_frame:
            values = pd.to_numeric(seed_frame[column], errors="coerce")
            aggregate[column] = float(values.mean())
            aggregate[f"{column}_std"] = float(values.std(ddof=1)) if values.notna().sum() > 1 else 0.0
    aggregate["fitness_objective"] = "validation accuracy" if mode == "accuracy" else "validation accuracy minus normalized depth and two-qubit gate penalties"
    save_metrics(run_dir, [aggregate])
    log_message(f"{mode} evolution summary across {len(GA_SEEDS)} seeds: validation accuracy={aggregate['validation_accuracy']:.4f} ± {aggregate['validation_accuracy_std']:.4f}; test accuracy={aggregate['test_accuracy']:.4f} ± {aggregate['test_accuracy_std']:.4f}.")
    log_message(f"Selected circuit for finite-shot evaluation came from seed {best_seed}, chosen by validation fitness only.")

def run_accuracy_evolution(run_dir: Path) -> None:
    evolve_circuits(run_dir, mode="accuracy")

def run_hardware_aware_evolution(run_dir: Path) -> None:
    evolve_circuits(run_dir, mode="hardware_aware")

def _research_selection_score(metrics: Dict[str, float]) -> float:
    """Pre-specified score: equal weight to balanced accuracy and macro-F1."""
    return 0.5 * float(metrics["validation_balanced_accuracy"]) + 0.5 * float(metrics["validation_f1_macro"])

def _research_evaluate_genome(
    genome: Sequence[Dict[str, Any]], X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray, mode: str,
    cache: Dict[str, Tuple[float, Dict[str, Any]]],
) -> Tuple[float, Dict[str, Any]]:
    """Fitness used in the outer-CV study; fitness is based only on inner validation."""
    from sklearn.svm import SVC

    genome = normalize_genome(genome)
    cache_key = json.dumps(genome, sort_keys=True)
    if cache_key in cache:
        return cache[cache_key]
    train_states = states_for_data(X_train, genome=genome)
    val_states = states_for_data(X_val, genome=genome)
    train_kernel = fidelity_kernel(train_states)
    val_kernel = fidelity_kernel(val_states, train_states)
    classifier = SVC(kernel="precomputed", C=1.0)
    classifier.fit(train_kernel, y_train)
    validation_metrics = classification_metrics(y_val, classifier.predict(val_kernel), "validation")
    performance_score = _research_selection_score(validation_metrics)
    complexity = genome_complexity(genome)
    fitness = performance_score
    if mode == "complexity_aware":
        depth_penalty = HARDWARE_DEPTH_PENALTY * complexity["estimated_depth"] / (N_QUBITS + MAX_GENOME_LENGTH)
        two_qubit_penalty = HARDWARE_TWO_QUBIT_PENALTY * complexity["two_qubit_gate_count"] / MAX_GENOME_LENGTH
        fitness -= depth_penalty + two_qubit_penalty
    details = {
        **validation_metrics, **complexity,
        "validation_selection_score": float(performance_score),
        "fitness": float(fitness), "mode": mode,
    }
    cache[cache_key] = (float(fitness), details)
    return float(fitness), details

def _search_research_genome(
    X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
    mode: str, seed: int, repeat_number: int, fold_number: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
    """Run a fixed-budget GA; complexity-aware selection obeys a validation constraint.

    For the complexity-aware method, the GA uses the pre-specified complexity penalty
    to guide search, then selects the circuit with the fewest two-qubit gates among all
    unique candidates visited whose inner-validation balanced accuracy is no more than
    COMPLEXITY_BALANCED_ACCURACY_TOLERANCE below the best balanced accuracy observed.
    Ties are broken by lower depth, fewer total gates, then better validation metrics.
    Only the inner validation partition is used for this selection.
    """
    started = time.perf_counter()
    rng = random.Random(seed)
    population = [random_genome(rng) for _ in range(GA_POPULATION_SIZE)]
    cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}
    genome_archive: Dict[str, List[Dict[str, Any]]] = {}
    history: List[Dict[str, Any]] = []
    best_genome: Optional[List[Dict[str, Any]]] = None
    best_details: Optional[Dict[str, Any]] = None
    best_fitness = -float("inf")

    for generation in range(GA_GENERATIONS):
        scored = []
        for raw_genome in population:
            normalized = normalize_genome(raw_genome)
            cache_key = json.dumps(normalized, sort_keys=True)
            genome_archive.setdefault(cache_key, [dict(op) for op in normalized])
            scored.append(_research_evaluate_genome(normalized, X_train, y_train, X_val, y_val, mode, cache))
        scores = [item[0] for item in scored]
        details = [item[1] for item in scored]
        order = sorted(range(len(population)), key=lambda idx: scores[idx], reverse=True)
        current_best = order[0]
        if scores[current_best] > best_fitness:
            best_fitness = float(scores[current_best])
            best_genome = [dict(op) for op in population[current_best]]
            best_details = dict(details[current_best])
        history.append({
            "repeat": repeat_number, "fold": fold_number, "mode": mode, "search_seed": seed,
            "generation": generation + 1,
            "best_fitness": float(scores[current_best]),
            "best_validation_selection_score": float(details[current_best]["validation_selection_score"]),
            "best_validation_balanced_accuracy": float(details[current_best]["validation_balanced_accuracy"]),
            "best_validation_f1_macro": float(details[current_best]["validation_f1_macro"]),
            "mean_fitness": float(np.mean(scores)),
            "best_gate_count": int(details[current_best]["gate_count"]),
            "best_two_qubit_gate_count": int(details[current_best]["two_qubit_gate_count"]),
            "best_estimated_depth": int(details[current_best]["estimated_depth"]),
            "unique_genomes_evaluated": len(cache),
        })
        next_population = [[dict(op) for op in population[idx]] for idx in order[:GA_ELITE_COUNT]]
        while len(next_population) < GA_POPULATION_SIZE:
            first = tournament_select(population, scores, rng)
            second = tournament_select(population, scores, rng)
            next_population.append(mutate(crossover(first, second, rng), rng, mutation_rate=GA_MUTATION_RATE))
        population = next_population

    if best_genome is None or best_details is None:
        raise RuntimeError("Evolution failed to produce a candidate circuit.")

    best_balanced_seen = max(float(details["validation_balanced_accuracy"]) for _, details in cache.values())
    constraint_met: Optional[bool] = None
    validation_gap = max(0.0, best_balanced_seen - float(best_details["validation_balanced_accuracy"]))
    selection_rule = "highest 0.5*validation balanced accuracy + 0.5*validation macro-F1"

    if mode == "complexity_aware":
        eligible = []
        for key, (fitness, details) in cache.items():
            genome = genome_archive[key]
            if float(details["validation_balanced_accuracy"]) + 1e-12 >= best_balanced_seen - COMPLEXITY_BALANCED_ACCURACY_TOLERANCE:
                eligible.append((
                    int(details["two_qubit_gate_count"]),
                    int(details["estimated_depth"]),
                    int(details["gate_count"]),
                    -float(details["validation_balanced_accuracy"]),
                    -float(details["validation_f1_macro"]),
                    key, float(fitness), details, genome,
                ))
        if not eligible:
            raise RuntimeError("No candidate met the balanced-accuracy constraint; the best candidate should always be eligible.")
        chosen = min(eligible, key=lambda row: row[:5])
        _, _, _, _, _, chosen_key, best_fitness, chosen_details, chosen_genome = chosen
        best_genome = [dict(op) for op in chosen_genome]
        best_details = dict(chosen_details)
        validation_gap = max(0.0, best_balanced_seen - float(best_details["validation_balanced_accuracy"]))
        constraint_met = validation_gap <= COMPLEXITY_BALANCED_ACCURACY_TOLERANCE + 1e-12
        selection_rule = (
            f"minimum two-qubit gates, then depth, then total gates among evaluated candidates with "
            f"validation balanced accuracy >= best observed - {COMPLEXITY_BALANCED_ACCURACY_TOLERANCE:.3f}"
        )

    elapsed = time.perf_counter() - started
    best_details.update({
        "best_validation_balanced_accuracy_seen": float(best_balanced_seen),
        "validation_balanced_accuracy_loss_from_best": float(validation_gap),
        "accuracy_constraint_tolerance": float(COMPLEXITY_BALANCED_ACCURACY_TOLERANCE),
        "accuracy_constraint_applicable": bool(mode == "complexity_aware"),
        "accuracy_constraint_met": constraint_met,
        "selection_rule": selection_rule,
        "candidate_budget_nominal": int(GA_POPULATION_SIZE * GA_GENERATIONS),
        "candidate_budget_unit": "nominal_population_slots",
        "candidate_attempt_budget_nominal": int(GA_POPULATION_SIZE * GA_GENERATIONS),
        "candidate_attempts": int(GA_POPULATION_SIZE * GA_GENERATIONS),
        "candidate_unique_budget_target": int(len(cache)),
        "matched_to_model": "",
        "unique_budget_matched": True,
        "unique_candidates_evaluated": int(len(cache)),
        "search_elapsed_seconds": float(elapsed),
    })
    return normalize_genome(best_genome), best_details, history

def _search_random_research_genome(
    X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
    seed: int, repeat_number: int, fold_number: int,
    target_unique_candidates: Optional[int] = None,
    selection_mode: str = "performance",
    matched_to_model: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
    """Random-search control matched to a GA's number of unique circuit evaluations.

    selection_mode='performance' picks the highest pre-specified validation score.
    selection_mode='complexity_aware' applies exactly the same balanced-accuracy
    tolerance and logical-complexity tie-breaks used by the complexity-aware GA.
    Duplicate genomes do not consume the unique-candidate budget, but every attempt
    is counted. Selection sees inner-training/validation data only.
    """
    if selection_mode not in {"performance", "complexity_aware"}:
        raise ValueError("selection_mode must be 'performance' or 'complexity_aware'.")
    started = time.perf_counter()
    rng = random.Random(seed)
    cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}
    history: List[Dict[str, Any]] = []
    target = int(target_unique_candidates or (GA_POPULATION_SIZE * GA_GENERATIONS))
    target = max(1, target)
    max_attempts = max(target * 20, target + 100)
    candidate_attempts = 0
    method_name = "random_search_complexity_aware" if selection_mode == "complexity_aware" else "random_search_performance"

    while len(cache) < target and candidate_attempts < max_attempts:
        candidate_attempts += 1
        genome = normalize_genome(random_genome(rng))
        cache_key = json.dumps(genome, sort_keys=True)
        if cache_key in cache:
            continue
        fitness, details = _research_evaluate_genome(
            genome, X_train, y_train, X_val, y_val, "performance", cache
        )
        # Log the best validation-score candidate seen so far; final selection for
        # complexity-aware random search occurs after the candidate pool is complete.
        _, current_best_details = max(
            cache.values(), key=lambda item: float(item[1]["validation_selection_score"])
        )
        history.append({
            "repeat": repeat_number, "fold": fold_number, "mode": method_name,
            "search_seed": seed, "generation": len(cache),
            "candidate_attempts": candidate_attempts,
            "best_fitness": float(current_best_details["validation_selection_score"]),
            "best_validation_selection_score": float(current_best_details["validation_selection_score"]),
            "best_validation_balanced_accuracy": float(current_best_details["validation_balanced_accuracy"]),
            "best_validation_f1_macro": float(current_best_details["validation_f1_macro"]),
            "mean_fitness": float(np.mean([entry[1]["validation_selection_score"] for entry in cache.values()])),
            "best_gate_count": int(current_best_details["gate_count"]),
            "best_two_qubit_gate_count": int(current_best_details["two_qubit_gate_count"]),
            "best_estimated_depth": int(current_best_details["estimated_depth"]),
            "unique_genomes_evaluated": len(cache),
            "candidate_unique_budget_target": target,
            "matched_to_model": matched_to_model or "",
        })

    if not cache:
        raise RuntimeError("Random search failed to evaluate any unique circuit.")
    unique_budget_matched = len(cache) == target
    if not unique_budget_matched:
        log_message(
            f"WARNING: {method_name} evaluated {len(cache)}/{target} target unique circuits "
            f"after {candidate_attempts} attempts; report this budget shortfall."
        )

    best_balanced_seen = max(float(details["validation_balanced_accuracy"]) for _, details in cache.values())
    candidates = []
    for key, (_, details) in cache.items():
        genome = json.loads(key)
        candidates.append((key, genome, details))

    if selection_mode == "complexity_aware":
        eligible = [item for item in candidates if
                    float(item[2]["validation_balanced_accuracy"]) + 1e-12 >=
                    best_balanced_seen - COMPLEXITY_BALANCED_ACCURACY_TOLERANCE]
        if not eligible:
            raise RuntimeError("No random-search candidate met the balanced-accuracy constraint.")
        chosen_key, best_genome, best_details = min(
            eligible,
            key=lambda item: (
                int(item[2]["two_qubit_gate_count"]),
                int(item[2]["estimated_depth"]),
                int(item[2]["gate_count"]),
                -float(item[2]["validation_balanced_accuracy"]),
                -float(item[2]["validation_f1_macro"]),
                item[0],
            ),
        )
        validation_gap = max(0.0, best_balanced_seen - float(best_details["validation_balanced_accuracy"]))
        constraint_met: Optional[bool] = validation_gap <= COMPLEXITY_BALANCED_ACCURACY_TOLERANCE + 1e-12
        selection_rule = (
            f"random candidate pool: minimum two-qubit gates, then depth, then total gates among "
            f"candidates with validation balanced accuracy >= best observed - "
            f"{COMPLEXITY_BALANCED_ACCURACY_TOLERANCE:.3f}"
        )
    else:
        chosen_key, best_genome, best_details = max(
            candidates,
            key=lambda item: (
                float(item[2]["validation_selection_score"]),
                float(item[2]["validation_f1_macro"]),
                -int(item[2]["gate_count"]),
                item[0],
            ),
        )
        validation_gap = max(0.0, best_balanced_seen - float(best_details["validation_balanced_accuracy"]))
        constraint_met = None
        selection_rule = "highest 0.5*validation balanced accuracy + 0.5*validation macro-F1 among matched random candidate pool"

    best_details = dict(best_details)
    best_details.update({
        "mode": method_name,
        "best_validation_balanced_accuracy_seen": float(best_balanced_seen),
        "validation_balanced_accuracy_loss_from_best": float(validation_gap),
        "accuracy_constraint_tolerance": float(COMPLEXITY_BALANCED_ACCURACY_TOLERANCE),
        "accuracy_constraint_applicable": bool(selection_mode == "complexity_aware"),
        "accuracy_constraint_met": constraint_met,
        "selection_rule": selection_rule,
        "candidate_budget_nominal": int(GA_POPULATION_SIZE * GA_GENERATIONS),
        "candidate_budget_unit": "nominal_attempt_reference; unique budget matched to paired GA",
        "candidate_attempt_budget_nominal": None,
        "candidate_attempt_safety_cap": int(max_attempts),
        "candidate_attempts": int(candidate_attempts),
        "candidate_unique_budget_target": int(target),
        "matched_to_model": matched_to_model or "",
        "unique_budget_matched": bool(unique_budget_matched),
        "unique_candidates_evaluated": int(len(cache)),
        "search_elapsed_seconds": float(time.perf_counter() - started),
    })
    return normalize_genome(best_genome), best_details, history


# Noisy compute-uncompute kernel simulation

import time
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from quantum_core import build_genome_circuit, fidelity_kernel, states_for_data
from quantum_core import N_QUBITS


def validate_noise_parameters(
    single_qubit_error: float,
    two_qubit_error: float,
    readout_error: float,
) -> None:
    """Validate probability-like noise parameters without importing Qiskit Aer."""
    for name, value in (
        ("single_qubit_error", single_qubit_error),
        ("two_qubit_error", two_qubit_error),
        ("readout_error", readout_error),
    ):
        if not np.isfinite(value) or value < 0.0 or value > 1.0:
            raise ValueError(f"{name} must be a finite probability between 0 and 1.")


def build_depolarizing_noise_model(
    single_qubit_error: float = 0.001,
    two_qubit_error: float = 0.01,
    readout_error: float = 0.01,
):
    """Create a Qiskit Aer noise model with explicit, configurable error rates."""
    validate_noise_parameters(single_qubit_error, two_qubit_error, readout_error)
    try:
        from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error
    except ImportError as exc:
        raise RuntimeError(
            "Noisy simulation requires Qiskit Aer. Install project dependencies with "
            "`python -m pip install -r requirements.txt`."
        ) from exc

    noise_model = NoiseModel()
    if single_qubit_error > 0:
        one_qubit_channel = depolarizing_error(float(single_qubit_error), 1)
        noise_model.add_all_qubit_quantum_error(
            one_qubit_channel, ["h", "rx", "ry", "rz"]
        )
    if two_qubit_error > 0:
        two_qubit_channel = depolarizing_error(float(two_qubit_error), 2)
        noise_model.add_all_qubit_quantum_error(two_qubit_channel, ["cx", "cz"])
    if readout_error > 0:
        readout_channel = ReadoutError(
            [[1.0 - readout_error, readout_error],
             [readout_error, 1.0 - readout_error]]
        )
        noise_model.add_all_qubit_readout_error(readout_channel)
    return noise_model


def _build_compute_uncompute_circuit(x_a: np.ndarray, x_b: np.ndarray, genome: Sequence[Dict[str, Any]]):
    """Build U(x_b)^dagger U(x_a)|0>; all-zero probability estimates kernel fidelity."""
    try:
        from qiskit import QuantumCircuit
    except ImportError as exc:
        raise RuntimeError("Qiskit is required. Install project dependencies first.") from exc

    circuit_a = build_genome_circuit(np.asarray(x_a, dtype=float), genome)
    circuit_b = build_genome_circuit(np.asarray(x_b, dtype=float), genome)
    circuit = QuantumCircuit(N_QUBITS, name="noisy_fidelity_kernel")
    circuit.compose(circuit_a, inplace=True)
    circuit.compose(circuit_b.inverse(), inplace=True)
    circuit.measure_all()
    return circuit


def noisy_fidelity_kernel(
    X_a: np.ndarray,
    genome: Sequence[Dict[str, Any]],
    noise_model,
    shots: int = 512,
    seed: int = 42,
    X_b: Optional[np.ndarray] = None,
    chunk_size: int = 512,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Estimate an evolved-circuit fidelity kernel with noisy sampled circuits.

    If ``X_b`` is omitted, a symmetric training kernel is built. Its diagonal is
    fixed to 1 by convention and each off-diagonal pair is estimated once then
    mirrored. With ``X_b`` provided, estimates the rectangular cross-kernel
    K(X_a, X_b). Simulations are chunked to cap memory use.
    """
    try:
        from qiskit import transpile
        from qiskit_aer import AerSimulator
    except ImportError as exc:
        raise RuntimeError(
            "Noisy simulation requires qiskit and qiskit-aer. Install requirements.txt."
        ) from exc

    X_a = np.asarray(X_a, dtype=float)
    if X_a.ndim != 2 or X_a.shape[1] != N_QUBITS:
        raise ValueError(f"X_a must have shape (n_samples, {N_QUBITS}).")
    if X_b is not None:
        X_b = np.asarray(X_b, dtype=float)
        if X_b.ndim != 2 or X_b.shape[1] != N_QUBITS:
            raise ValueError(f"X_b must have shape (n_samples, {N_QUBITS}).")
    if shots < 1:
        raise ValueError("shots must be a positive integer.")
    if chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer.")

    backend = AerSimulator(noise_model=noise_model)
    start = time.perf_counter()
    records = []

    if X_b is None:
        kernel = np.eye(len(X_a), dtype=float)
        pairs = [(i, j) for i in range(len(X_a)) for j in range(i + 1, len(X_a))]
        for i, j in pairs:
            records.append((i, j, X_a[i], X_a[j]))
        symmetric = True
        out_shape = kernel.shape
    else:
        kernel = np.zeros((len(X_a), len(X_b)), dtype=float)
        records = [
            (i, j, X_a[i], X_b[j])
            for i in range(len(X_a))
            for j in range(len(X_b))
        ]
        symmetric = False
        out_shape = kernel.shape

    total = len(records)
    for start_index in range(0, total, chunk_size):
        batch = records[start_index:start_index + chunk_size]
        circuits = [_build_compute_uncompute_circuit(a, b, genome) for _, _, a, b in batch]
        compiled = transpile(circuits, backend=backend, optimization_level=0, seed_transpiler=seed + start_index)
        result = backend.run(
            compiled,
            shots=int(shots),
            seed_simulator=int(seed + start_index),
        ).result()
        for offset, record in enumerate(batch):
            i, j, _, _ = record
            counts = result.get_counts(offset)
            probability_zero = float(counts.get("0" * N_QUBITS, 0)) / float(shots)
            kernel[i, j] = probability_zero
            if symmetric:
                kernel[j, i] = probability_zero

    info = {
        "shots_per_kernel_entry": int(shots),
        "circuit_pairs_simulated": int(total),
        "symmetric_kernel": symmetric,
        "kernel_shape": list(out_shape),
        "diagonal_handling": "fixed to 1 by convention for the symmetric training kernel",
        "elapsed_seconds": float(time.perf_counter() - start),
        "seed": int(seed),
    }
    return np.clip(kernel, 0.0, 1.0), info


def prepare_four_angle_split(
    X: np.ndarray,
    y: np.ndarray,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Make a leakage-controlled stratified 60/20/20 split with four PCA angles."""
    from sklearn.decomposition import PCA
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler, StandardScaler

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    if X.ndim != 2 or len(X) != len(y) or X.shape[1] < N_QUBITS:
        raise ValueError("Expected aligned 2D features with at least four columns and labels.")

    all_indices = np.arange(len(y))
    train_idx, temporary_idx = train_test_split(
        all_indices, test_size=0.40, random_state=seed, stratify=y
    )
    val_idx, test_idx = train_test_split(
        temporary_idx, test_size=0.50, random_state=seed, stratify=y[temporary_idx]
    )

    standardizer = StandardScaler().fit(X[train_idx])
    X_train_standard = standardizer.transform(X[train_idx])
    X_val_standard = standardizer.transform(X[val_idx])
    X_test_standard = standardizer.transform(X[test_idx])
    pca = PCA(n_components=N_QUBITS, random_state=seed).fit(X_train_standard)
    X_train_pca = pca.transform(X_train_standard)
    X_val_pca = pca.transform(X_val_standard)
    X_test_pca = pca.transform(X_test_standard)
    angle_scaler = MinMaxScaler(feature_range=(-np.pi, np.pi)).fit(X_train_pca)

    return {
        "X_train_angles": angle_scaler.transform(X_train_pca),
        "X_val_angles": angle_scaler.transform(X_val_pca),
        "X_test_angles": angle_scaler.transform(X_test_pca),
        "y_train": y[train_idx],
        "y_val": y[val_idx],
        "y_test": y[test_idx],
        "train_indices": train_idx,
        "val_indices": val_idx,
        "test_indices": test_idx,
        "pca_explained_variance_ratio": pca.explained_variance_ratio_,
    }


def balanced_subsample(
    X: np.ndarray,
    y: np.ndarray,
    samples_per_class: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select an equal, reproducible number of samples from each binary class."""
    if samples_per_class < 1:
        raise ValueError("samples_per_class must be positive.")
    X = np.asarray(X)
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    selected = []
    for label in np.unique(y):
        candidates = np.flatnonzero(y == label)
        if len(candidates) < samples_per_class:
            raise ValueError(
                f"Class {label} has only {len(candidates)} samples; cannot select "
                f"{samples_per_class} per class."
            )
        selected.extend(rng.choice(candidates, samples_per_class, replace=False).tolist())
    selected = np.asarray(selected, dtype=int)
    rng.shuffle(selected)
    return X[selected], y[selected], selected


def compare_ideal_and_noisy_kernels(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    genome: Sequence[Dict[str, Any]],
    noise_model,
    shots: int = 512,
    seed: int = 42,
    chunk_size: int = 512,
) -> Tuple[list[Dict[str, Any]], Dict[str, np.ndarray], Dict[str, Any]]:
    """Fit ideal and noisy precomputed-kernel SVMs on the same subsample."""
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
    from sklearn.svm import SVC

    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    y_train, y_test = np.asarray(y_train, dtype=int), np.asarray(y_test, dtype=int)

    # Exact noiseless reference from the project's original Statevector approach.
    train_states = states_for_data(X_train, genome=genome)
    test_states = states_for_data(X_test, genome=genome)
    ideal_train = fidelity_kernel(train_states)
    ideal_test = fidelity_kernel(test_states, train_states)

    noisy_train, train_info = noisy_fidelity_kernel(
        X_train, genome, noise_model, shots=shots, seed=seed,
        X_b=None, chunk_size=chunk_size,
    )
    noisy_test, test_info = noisy_fidelity_kernel(
        X_test, genome, noise_model, shots=shots, seed=seed + 1,
        X_b=X_train, chunk_size=chunk_size,
    )

    rows = []
    all_predictions: Dict[str, np.ndarray] = {}
    for model_name, train_kernel, test_kernel in (
        ("ideal_statevector", ideal_train, ideal_test),
        ("noisy_aer", noisy_train, noisy_test),
    ):
        model = SVC(kernel="precomputed", C=1.0)
        model.fit(train_kernel, y_train)
        predictions = model.predict(test_kernel)
        all_predictions[model_name] = predictions
        rows.append({
            "model": model_name,
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "shots_per_kernel_entry": 0 if model_name == "ideal_statevector" else int(shots),
            "accuracy": float(accuracy_score(y_test, predictions)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
            "f1_macro": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
            "train_kernel_rmse_vs_ideal": 0.0 if model_name == "ideal_statevector" else float(np.sqrt(np.mean((noisy_train - ideal_train) ** 2))),
            "test_kernel_rmse_vs_ideal": 0.0 if model_name == "ideal_statevector" else float(np.sqrt(np.mean((noisy_test - ideal_test) ** 2))),
        })

    metadata = {
        "noisy_train_kernel": train_info,
        "noisy_test_kernel": test_info,
        "interpretation": (
            "Exploratory sampled-kernel classification on a balanced subsample. The noise model is synthetic, "
            "not calibrated to a specific device, and these results are not part of the original ideal repeated-CV study."
        ),
    }
    kernels = {
        "ideal_train_kernel": ideal_train,
        "ideal_test_kernel": ideal_test,
        "noisy_train_kernel": noisy_train,
        "noisy_test_kernel": noisy_test,
    }
    return rows, {**kernels, **{f"{key}_predictions": value for key, value in all_predictions.items()}}, metadata
