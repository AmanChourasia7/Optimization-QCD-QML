#!/usr/bin/env python3
"""Export saved Qiskit circuits and summarize their recorded complexity."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd

DATASETS = {
    "breast_cancer": "Breast cancer",
    "iris_setosa_vs_versicolor": "Iris (setosa vs versicolor)",
    "wine_class0_vs_rest": "Wine (class 0 vs rest)",
    "digits_0_vs_1": "Digits (0 vs 1)",
    "digits_3_vs_8": "Digits (3 vs 8)",
    "digits_4_vs_9": "Digits (4 vs 9)",
}
METHODS = {
    "complexity_aware": "Genetic algorithm - complexity-aware",
    "performance": "Genetic algorithm - performance",
    "random_search_complexity_aware": "Random search - complexity-aware",
    "random_search_performance": "Random search - performance",
    "hardware_aware": "Fixed-split GA - complexity-aware",
    "accuracy": "Fixed-split GA - accuracy",
}


def find_root(start: Path, requested: str | None) -> Path:
    if requested:
        root = Path(requested).expanduser().resolve()
        if not (root / "results").is_dir():
            raise FileNotFoundError(f"No results/ folder under {root}")
        return root
    for candidate in (start.resolve(), *start.resolve().parents):
        if (candidate / "results").is_dir():
            return candidate
    raise FileNotFoundError("Could not find project root. Run from the project or pass --root.")


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def dataset_for(path: Path) -> str:
    for part in path.parts:
        if part in DATASETS:
            return part
    for parent in path.parents:
        for name in ("study_protocol.json", "research_run_protocol.json"):
            protocol = read_json(parent / name)
            if protocol:
                ds = str(protocol.get("dataset_id", ""))
                if ds in DATASETS:
                    return ds
    return "unknown_dataset" if "repeated_cv" in path.parts else "breast_cancer"


def method_for(payload: dict[str, Any], path: Path) -> tuple[str, str]:
    mode = str(payload.get("mode", "")).lower()
    stem = path.stem.lower()
    if mode not in METHODS:
        for known in METHODS:
            if known in stem:
                mode = known
                break
    if mode in METHODS:
        return mode, METHODS[mode]
    return mode or "saved_circuit", f"Saved circuit - {mode or path.stem}"


def int_or_none(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def associated_qasm(path: Path, payload: dict[str, Any]) -> Path | None:
    options = [path.with_suffix(".qasm"), path.with_name(path.stem + "_example.qasm")]
    selection = payload.get("selection_across_seeds", {})
    if isinstance(selection, dict) and selection.get("selected_seed_by_validation_fitness") is not None:
        seed = selection["selected_seed_by_validation_fitness"]
        mode = str(payload.get("mode", "accuracy"))
        options.append(path.with_name(f"best_circuit_{mode}_seed_{seed}_example.qasm"))
    return next((p for p in options if p.is_file()), None)


def scan_circuits(root: Path, run_id: str | None = None) -> list[dict[str, Any]]:
    results = root / "results"
    if run_id:
        run_dir = results / "runs" / run_id
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
        scan_roots = [run_dir]
    else:
        scan_roots = [p for p in (results / "runs", results / "research_study", root / "sessions") if p.is_dir()]
    records: list[dict[str, Any]] = []
    seen_paths: set[Path] = set()
    associated_qasm_paths: set[Path] = set()

    for scan_root in scan_roots:
        for path in sorted(scan_root.rglob("*.json")):
            if "circuits" not in path.parts:
                continue
            payload = read_json(path)
            genome = payload.get("genome") if payload else None
            if not isinstance(genome, list):
                continue
            resolved = path.resolve()
            if resolved in seen_paths:
                continue
            seen_paths.add(resolved)
            ds = dataset_for(path)
            mode, method = method_for(payload, path)
            qasm = associated_qasm(path, payload)
            if qasm:
                associated_qasm_paths.add(qasm.resolve())
            repeat = int_or_none(payload.get("repeat"))
            fold = int_or_none(payload.get("fold"))
            metrics = payload.get("selection_metrics", {})
            complexity = payload.get("complexity", {})
            if not isinstance(metrics, dict):
                metrics = {}
            if not isinstance(complexity, dict):
                complexity = {}
            genome_key = json.dumps(genome, sort_keys=True, separators=(",", ":"))
            records.append({
                "dataset_id": ds,
                "dataset": DATASETS.get(ds, ds),
                "mode": mode,
                "method": method,
                "source_json": str(path.resolve()),
                "source_qasm": str(qasm.resolve()) if qasm else "",
                "payload": payload,
                "genome": genome,
                "genome_hash": hashlib.sha256(genome_key.encode()).hexdigest(),
                "repeat": repeat,
                "fold": fold,
                "is_fold_selection": repeat is not None and fold is not None,
                "gate_count": int_or_none(complexity.get("gate_count", metrics.get("gate_count", payload.get("gate_count")))),
                "two_qubit_gate_count": int_or_none(complexity.get("two_qubit_gate_count", metrics.get("two_qubit_gate_count", payload.get("two_qubit_gate_count")))),
                "depth": int_or_none(complexity.get("estimated_depth", metrics.get("estimated_depth", payload.get("estimated_depth")))),
                "validation_accuracy": metrics.get("validation_accuracy"),
                "validation_balanced_accuracy": metrics.get("validation_balanced_accuracy"),
                "validation_f1_macro": metrics.get("validation_f1_macro"),
                "accuracy_constraint_met": payload.get("accuracy_constraint_met", metrics.get("accuracy_constraint_met")),
                "candidate_attempts": int_or_none(payload.get("candidate_attempts", metrics.get("candidate_attempts"))),
                "unique_candidates_evaluated": int_or_none(payload.get("unique_candidates_evaluated", metrics.get("unique_candidates_evaluated"))),
                "render_source": "saved QASM" if qasm else "genome topology reconstruction (zero input angles)",
            })

        for qasm in sorted(scan_root.rglob("qiskit_zz_inspired_feature_map.qasm")):
            if qasm.resolve() in associated_qasm_paths:
                continue
            ds = dataset_for(qasm)
            records.append({
                "dataset_id": ds, "dataset": DATASETS.get(ds, ds),
                "mode": "zz_kernel_baseline", "method": "Standard ZZ-inspired feature-map baseline",
                "source_json": "", "source_qasm": str(qasm.resolve()), "payload": {}, "genome": [],
                "genome_hash": hashlib.sha256(qasm.read_bytes()).hexdigest(),
                "repeat": None, "fold": None, "is_fold_selection": False,
                "gate_count": None, "two_qubit_gate_count": None, "depth": None,
                "validation_accuracy": None, "validation_balanced_accuracy": None, "validation_f1_macro": None,
                "accuracy_constraint_met": None, "candidate_attempts": None,
                "unique_candidates_evaluated": None, "render_source": "saved QASM baseline",
            })
    records.sort(key=lambda r: (r["dataset"].lower(), r["method"].lower(), r["repeat"] or 0, r["fold"] or 0, r["source_json"] or r["source_qasm"]))
    return records


def circuit_from_genome(genome: list[dict[str, Any]]):
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(4, name="saved_evolved_circuit")
    for q in range(4):
        qc.h(q)
    gates = {"RX": qc.rx, "RY": qc.ry, "RZ": qc.rz}
    for op in genome:
        name = str(op.get("gate", "RY")).upper()
        q = int(op.get("q", 0)) % 4
        if name in gates:
            gates[name](float(op.get("scale", 1.0)) * 0.0, q)
        elif name == "CNOT":
            target = int(op.get("target", (q + 1) % 4)) % 4
            if target == q:
                target = (q + 1) % 4
            qc.cx(q, target)
        elif name == "CZ":
            target = int(op.get("target", (q + 1) % 4)) % 4
            if target == q:
                target = (q + 1) % 4
            qc.cz(q, target)
    return qc


def load_circuit(record: dict[str, Any]):
    if record["source_qasm"]:
        try:
            from qiskit import qasm2
            return qasm2.loads(Path(record["source_qasm"]).read_text(encoding="utf-8"))
        except Exception as exc:
            if not record["genome"]:
                raise RuntimeError(f"Could not load QASM {record['source_qasm']}: {exc}") from exc
    if record["genome"]:
        return circuit_from_genome(record["genome"])
    raise RuntimeError("No valid QASM or genome available")


def number_value(value: Any) -> float | None:
    try:
        v = float(value)
        return v if pd.notna(v) else None
    except (TypeError, ValueError):
        return None


def build_tables(records: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    rows = []
    for r in records:
        rows.append({k: r.get(k) for k in (
            "dataset_id", "dataset", "mode", "method", "source_json", "source_qasm", "repeat", "fold",
            "is_fold_selection", "genome_hash", "gate_count", "two_qubit_gate_count", "depth",
            "validation_accuracy", "validation_balanced_accuracy", "validation_f1_macro",
            "accuracy_constraint_met", "candidate_attempts", "unique_candidates_evaluated", "render_source",
        )})
    details = pd.DataFrame(rows)
    if details.empty:
        return details, pd.DataFrame(), pd.DataFrame(), {}

    fold_rows = details[details["is_fold_selection"] == True].copy()  # noqa: E712
    if fold_rows.empty:
        fold_rows = details.copy()
    summary = fold_rows.groupby(["dataset_id", "dataset", "mode", "method"], dropna=False).agg(
        circuit_records=("genome_hash", "count"),
        unique_genomes=("genome_hash", "nunique"),
        mean_gate_count=("gate_count", "mean"),
        mean_two_qubit_gate_count=("two_qubit_gate_count", "mean"),
        mean_depth=("depth", "mean"),
        mean_validation_accuracy=("validation_accuracy", "mean"),
        mean_validation_balanced_accuracy=("validation_balanced_accuracy", "mean"),
        mean_validation_f1_macro=("validation_f1_macro", "mean"),
    ).reset_index()

    # Keep one lowest-complexity saved fold circuit for each dataset and search method.
    selected = fold_rows[fold_rows["mode"].isin(["complexity_aware", "random_search_complexity_aware"])].copy()
    if not selected.empty:
        selected["two_qubit_gate_count"] = pd.to_numeric(selected["two_qubit_gate_count"], errors="coerce")
        selected["depth"] = pd.to_numeric(selected["depth"], errors="coerce")
        selected["gate_count"] = pd.to_numeric(selected["gate_count"], errors="coerce")
        selected["validation_balanced_accuracy"] = pd.to_numeric(selected["validation_balanced_accuracy"], errors="coerce")
        selected["constraint_rank"] = selected["accuracy_constraint_met"].map({True: 0, False: 1}).fillna(0)
        selected = selected.sort_values(
            ["dataset_id", "mode", "constraint_rank", "two_qubit_gate_count", "depth", "gate_count", "validation_balanced_accuracy"],
            ascending=[True, True, True, True, True, True, False], na_position="last",
        ).groupby(["dataset_id", "mode"], dropna=False).head(1)
        selected = selected[["dataset_id", "dataset", "mode", "method", "source_json", "repeat", "fold", "gate_count", "two_qubit_gate_count", "depth", "validation_balanced_accuracy", "accuracy_constraint_met"]]
        selected["selection_basis"] = "Lowest-complexity saved fold-selected circuit; not a single refit model"

    mean_lookup = summary.set_index(["dataset_id", "mode"]).to_dict("index")
    reductions = []
    for ds in sorted(fold_rows["dataset_id"].dropna().unique()):
        ga = mean_lookup.get((ds, "complexity_aware"), {})
        rs = mean_lookup.get((ds, "random_search_complexity_aware"), {})
        ga_mean = number_value(ga.get("mean_two_qubit_gate_count"))
        rs_mean = number_value(rs.get("mean_two_qubit_gate_count"))
        reduction = (rs_mean - ga_mean) / rs_mean * 100 if ga_mean is not None and rs_mean not in (None, 0) else None
        reductions.append({
            "dataset_id": ds,
            "dataset": DATASETS.get(ds, ds),
            "ga_complexity_aware_mean_two_qubit_gates": ga_mean,
            "random_search_complexity_aware_mean_two_qubit_gates": rs_mean,
            "observed_reduction_pct": reduction,
            "ga_mean_validation_balanced_accuracy": number_value(ga.get("mean_validation_balanced_accuracy")),
            "random_search_mean_validation_balanced_accuracy": number_value(rs.get("mean_validation_balanced_accuracy")),
            "note": "Descriptive means from saved outer-fold-selected circuits; not a significance test.",
        })
    return details, summary, pd.DataFrame(reductions), {"selected_low_complexity": selected}


def render_gallery(records: list[dict[str, Any]], pdf_path: Path, rendered_manifest_path: Path, manifest: pd.DataFrame) -> int:
    from qiskit import QuantumCircuit
    rendered = []
    with PdfPages(pdf_path, metadata={
        "Title": "All Saved Quantum Circuit Diagrams",
        "Author": "Saved circuit export utility",
        "Subject": "Diagrams reconstructed from saved genomes or loaded from QASM; no search rerun",
    }) as pdf:
        for record in records:
            try:
                circuit = load_circuit(record)
                gate_count = int(circuit.size())
                ops = circuit.count_ops()
                two_q = int(ops.get("cx", 0) + ops.get("cz", 0) + ops.get("cnot", 0))
                depth = int(circuit.depth())
                repeat = f"Repeat {record['repeat']}" if record["repeat"] is not None else ""
                fold = f"Fold {record['fold']}" if record["fold"] is not None else ""
                metrics = []
                if repeat or fold:
                    metrics.append(", ".join(x for x in (repeat, fold) if x))
                if number_value(record.get("validation_balanced_accuracy")) is not None:
                    metrics.append(f"Validation balanced accuracy: {float(record['validation_balanced_accuracy']):.3f}")
                metrics.append(f"{circuit.num_qubits} qubits | {gate_count} gates | {two_q} two-qubit gates | depth {depth}")
                if record["render_source"] != "saved QASM":
                    metrics.append("Topology diagram: rotation angles set to zero because fold input angles were not saved")
                fig = circuit.draw(output="mpl", style="bw", fold=-1, idle_wires=False)
                fig.suptitle(record["dataset"] + "\n" + record["method"] + "\n" + " | ".join(metrics),
                             fontsize=10, fontweight="bold", y=0.99)
                fig.tight_layout(rect=(0, 0, 1, 0.90))
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
                row = {k: record.get(k) for k in manifest.columns if k in record}
                row.update({"pdf_page": len(rendered) + 1, "qubits": int(circuit.num_qubits), "gate_count_rendered": gate_count,
                            "two_qubit_gate_count_rendered": two_q, "depth_rendered": depth})
                rendered.append(row)
                if len(rendered) % 25 == 0:
                    print(f"Rendered {len(rendered)} circuit diagrams...")
            except Exception as exc:
                plt.close("all")
                print(f"Skipping {record['source_json'] or record['source_qasm']}: {exc}")
    if not rendered:
        pdf_path.unlink(missing_ok=True)
        raise RuntimeError("No diagrams could be rendered. Check Qiskit and the saved circuits.")
    pd.DataFrame(rendered).to_csv(rendered_manifest_path, index=False)
    return len(rendered)


def render_efficiency_pdf(reductions: pd.DataFrame, summary: pd.DataFrame, pdf_path: Path) -> None:
    with PdfPages(pdf_path, metadata={"Title": "Saved Circuit Efficiency Comparison"}) as pdf:
        if not reductions.empty:
            d = reductions.dropna(subset=["ga_complexity_aware_mean_two_qubit_gates", "random_search_complexity_aware_mean_two_qubit_gates"])
            if not d.empty:
                fig, ax = plt.subplots(figsize=(11, max(4.5, 0.65 * len(d))))
                y = range(len(d))
                ax.barh([i - 0.18 for i in y], d["ga_complexity_aware_mean_two_qubit_gates"], height=0.34,
                        label="Genetic algorithm - complexity-aware")
                ax.barh([i + 0.18 for i in y], d["random_search_complexity_aware_mean_two_qubit_gates"], height=0.34,
                        label="Budget-matched random search")
                ax.set_yticks(list(y), d["dataset"])
                ax.set_xlabel("Mean two-qubit gates per saved fold-selected circuit")
                ax.set_title("Circuit complexity by classification task")
                ax.legend(loc="best")
                ax.grid(axis="x", alpha=0.25)
                fig.tight_layout()
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

                d = d.copy()
                fig, ax = plt.subplots(figsize=(11, max(4.5, 0.65 * len(d))))
                plot_d = d.dropna(subset=["observed_reduction_pct"])
                ax.barh(plot_d["dataset"], plot_d["observed_reduction_pct"])
                ax.axvline(0, linewidth=1)
                ax.set_xlabel("Observed reduction in mean two-qubit gate count (%)")
                ax.set_title("Descriptive GA vs budget-matched random-search comparison")
                ax.grid(axis="x", alpha=0.25)
                fig.tight_layout()
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
        if summary.empty:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.axis("off")
            ax.text(0.02, 0.95, "No fold-level circuit metadata was available for comparison.", va="top")
            pdf.savefig(fig)
            plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export figures and summaries from already-saved Qiskit circuits.")
    parser.add_argument("--root", help="Project root containing results/ (defaults to searching from this script).")
    parser.add_argument("--run-id", help="Limit scanning to one folder under results/runs/, e.g. run_20261010_120000_ab12cd34.")
    parser.add_argument("--fold-selected-only", action="store_true", help="Export only per-dataset/repeat/fold selected circuit genomes (360 for a complete six-dataset run).")
    parser.add_argument("--summary-only", action="store_true", help="Write count and metric summaries without importing Qiskit or drawing circuits.")
    args = parser.parse_args()
    root = find_root(Path(__file__).parent, args.root)
    records = scan_circuits(root, run_id=args.run_id)
    if args.fold_selected_only:
        records = [record for record in records if record["is_fold_selection"]]
    if not records:
        print(f"No saved circuit genomes found under {root / 'results'}. No experiment was run.")
        print("Check that saved results/runs/**/circuits/*.json files are present.")
        return 2

    results = root / "results"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = results / f"circuit_analysis_{stamp}"
    suffix = 1
    while out_dir.exists():
        out_dir = results / f"circuit_analysis_{stamp}_{suffix}"
        suffix += 1
    figures_dir = out_dir / "figures"
    tables_dir = out_dir / "tables"
    metadata_dir = out_dir / "metadata"
    for folder in (figures_dir, tables_dir, metadata_dir):
        folder.mkdir(parents=True, exist_ok=False)

    details, summary, reductions, extras = build_tables(records)
    details.to_csv(tables_dir / "saved_circuit_manifest.csv", index=False)
    summary.to_csv(tables_dir / "dataset_method_summary.csv", index=False)
    reductions.to_csv(tables_dir / "efficiency_comparison.csv", index=False)
    selected = extras.get("selected_low_complexity", pd.DataFrame())
    selected.to_csv(tables_dir / "lowest_complexity_saved_circuits.csv", index=False)

    fold_records = [r for r in records if r["is_fold_selection"]]
    candidate_attempts = sum(r["candidate_attempts"] or 0 for r in fold_records)
    unique_eval = sum(r["unique_candidates_evaluated"] or 0 for r in fold_records)
    stats = {
        "project_root": str(root),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "saved_circuit_records_found": len(records),
        "saved_fold_selected_circuit_records": len(fold_records),
        "unique_saved_genome_definitions": len({r["genome_hash"] for r in records if r["genome"]}),
        "datasets_represented": sorted({r["dataset_id"] for r in records}),
        "saved_circuit_records_by_dataset": dict(Counter(r["dataset_id"] for r in records)),
        "recorded_candidate_attempts_sum_over_fold_records": candidate_attempts,
        "recorded_unique_candidates_evaluated_sum_over_fold_records": unique_eval,
        "note": "Circuit PDF pages represent saved selected genome records, not every transient candidate in search populations. Candidate counts are summed only from saved fold-record metadata when present.",
    }
    (metadata_dir / "export_summary.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"Project: {root}")
    print(f"Saved circuit records: {len(records)}")
    print(f"Saved fold-selected circuit records: {len(fold_records)}")
    print(f"Unique saved genome definitions: {stats['unique_saved_genome_definitions']}")
    print(f"Datasets represented: {len(stats['datasets_represented'])} ({', '.join(stats['datasets_represented'])})")
    if candidate_attempts:
        print(f"Candidate attempts recorded in saved fold metadata (sum): {candidate_attempts}")
    if unique_eval:
        print(f"Unique candidate evaluations recorded in saved fold metadata (sum): {unique_eval}")

    (out_dir / "README.txt").write_text(
        "Saved quantum circuit export\n"
        "=============================\n\n"
        "figures/  PDF diagrams and complexity-comparison figures\n"
        "tables/   CSV manifests and dataset/method comparisons\n"
        "metadata/ export counts and source summary\n\n"
        "This export reads saved genomes and circuit files only. It does not rerun search,\n"
        "training, cross-validation, or noisy simulation.\n",
        encoding="utf-8",
    )

    if args.summary_only:
        print(f"Summaries saved to: {out_dir}")
        return 0

    try:
        import qiskit  # noqa: F401
    except ImportError:
        print("Qiskit is required for circuit drawings. Install it with: python -m pip install qiskit pylatexenc matplotlib pandas")
        return 3

    gallery_name = "fold_selected_quantum_circuit_diagrams.pdf" if args.fold_selected_only else "all_saved_quantum_circuit_diagrams.pdf"
    gallery = figures_dir / gallery_name
    efficiency_pdf = figures_dir / "circuit_efficiency_comparison.pdf"
    rendered_manifest = tables_dir / "rendered_circuit_manifest.csv"
    rendered = render_gallery(records, gallery, rendered_manifest, details)
    render_efficiency_pdf(reductions, summary, efficiency_pdf)
    stats["rendered_pdf_pages"] = rendered
    stats["gallery_pdf"] = str(gallery)
    stats["efficiency_pdf"] = str(efficiency_pdf)
    stats["rendered_manifest_csv"] = str(rendered_manifest)
    (metadata_dir / "export_summary.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"\nRendered pages: {rendered}")
    print(f"Gallery PDF: {gallery}")
    print(f"Efficiency PDF: {efficiency_pdf}")
    print(f"Tables: {tables_dir}")
    print(f"All outputs: {out_dir}")
    print("No experiment, search, training, or cross-validation was rerun.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
