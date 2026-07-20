#!/usr/bin/env python3
"""Score ASR reconciliation outputs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


from endoca.evaluation.composition import score_composition_row


SCORER_VERSION = "bibm_asr_score_v1"


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_direct_and_atoms(scored_path: Path, target_models: set[str]):
    direct: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    atoms: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in read_jsonl(scored_path):
        model_id = row.get("model_id")
        if model_id not in target_models:
            continue
        sample_id = str(row.get("sample_id"))
        if row.get("probe_kind") == "complex_direct":
            direct[model_id][sample_id] = row
        elif row.get("probe_kind") == "atomic_direct":
            atoms[model_id][sample_id].append(row)
    return direct, atoms


def is_success(row: dict[str, Any]) -> bool:
    return not row.get("error") and bool(str(row.get("revised_answer") or row.get("prediction") or "").strip())


def score_variant(base: dict[str, Any], variant: str, prediction: str, answered: bool, source: str) -> dict[str, Any]:
    row = dict(base)
    row["probe_kind"] = f"asr_{variant}"
    row["asr_variant"] = variant
    row["asr_answer_source"] = source
    row["prediction"] = prediction if answered else "insufficient support"
    row["raw_output"] = row["prediction"]
    row["error"] = "" if answered else "abstained"
    row["deployment_policy_candidate"] = True
    if answered:
        score = score_composition_row(row)
    else:
        score = {
            "parse_status": "abstained",
            "is_correct": False,
            "support_ratio": 0.0,
            "atom_evidence": [],
            "failure_reason": "abstained",
        }
    return {**row, **score, "answered": answered, "asr_scorer_version": SCORER_VERSION}


def main() -> None:
    parser = argparse.ArgumentParser(description="Score ASR outputs")
    parser.add_argument("--selected-scored-jsonl", type=Path, required=True)
    parser.add_argument("--asr-jsonl", type=Path, nargs="+", required=True)
    parser.add_argument("--out-jsonl", type=Path, required=True)
    parser.add_argument("--out-metrics", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-report", type=Path, required=True)
    parser.add_argument("--expected-model-id", action="append", default=[])
    parser.add_argument("--expected-samples-per-model", type=int, default=6000)
    args = parser.parse_args()

    target_models: set[str] = set(args.expected_model_id)
    asr_rows: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    duplicates: dict[str, Counter[str]] = defaultdict(Counter)
    failures: dict[str, list[str]] = defaultdict(list)
    for path in args.asr_jsonl:
        for row in read_jsonl(path):
            model_id = row.get("target_model_id") or row.get("model_id")
            if not model_id:
                continue
            target_models.add(model_id)
            sample_id = str(row.get("sample_id"))
            if sample_id in asr_rows[model_id]:
                duplicates[model_id][sample_id] += 1
            if not is_success(row):
                failures[model_id].append(sample_id)
            asr_rows[model_id][sample_id] = row

    direct, atoms = load_direct_and_atoms(args.selected_scored_jsonl, target_models)
    problems: list[str] = []
    validation: dict[str, dict[str, Any]] = {}
    for model_id in sorted(target_models):
        direct_ids = set(direct.get(model_id, {}))
        asr_ids = set(asr_rows.get(model_id, {}))
        missing = sorted(direct_ids - asr_ids, key=lambda x: int(x) if x.isdigit() else x)
        extra = sorted(asr_ids - direct_ids, key=lambda x: int(x) if x.isdigit() else x)
        validation[model_id] = {
            "expected_samples_per_model": args.expected_samples_per_model,
            "direct_samples": len(direct_ids),
            "asr_samples": len(asr_ids),
            "missing_asr": len(missing),
            "extra_asr": len(extra),
            "duplicate_asr": sum(duplicates.get(model_id, {}).values()),
            "generation_failures": len(set(failures.get(model_id, []))),
        }
        if len(direct_ids) != args.expected_samples_per_model:
            problems.append(f"{model_id}: expected {args.expected_samples_per_model} direct rows, got {len(direct_ids)}")
        if len(asr_ids) != args.expected_samples_per_model:
            problems.append(f"{model_id}: expected {args.expected_samples_per_model} ASR rows, got {len(asr_ids)}")
        if missing:
            problems.append(f"{model_id}: missing ASR rows, first={missing[:8]}")
        if extra:
            problems.append(f"{model_id}: extra ASR rows, first={extra[:8]}")
        if duplicates.get(model_id):
            problems.append(f"{model_id}: duplicate ASR sample ids, first={list(duplicates[model_id])[:8]}")
        if failures.get(model_id):
            problems.append(f"{model_id}: failed ASR rows, first={failures[model_id][:8]}")
    if problems:
        raise SystemExit("ASR completeness validation failed:\n- " + "\n- ".join(problems))

    variant_rows: list[dict[str, Any]] = []
    for model_id in sorted(target_models):
        sample_ids = sorted(direct[model_id], key=lambda x: int(x) if x.isdigit() else x)
        for sample_id in sample_ids:
            direct_row = direct[model_id][sample_id]
            asr_row = asr_rows[model_id][sample_id]
            atom_rows = atoms[model_id].get(sample_id, [])
            atom_total = len(atom_rows)
            atom_correct = sum(1 for atom in atom_rows if atom.get("is_correct"))
            atomic_complete = bool(atom_total and atom_correct == atom_total)
            base = {
                "sample_id": direct_row.get("sample_id"),
                "model_id": model_id,
                "model_slug": asr_row.get("model_slug"),
                "split": direct_row.get("split"),
                "dataset": direct_row.get("dataset"),
                "img_id": direct_row.get("img_id"),
                "complexity": direct_row.get("complexity"),
                "question_class": direct_row.get("question_class"),
                "question": direct_row.get("question"),
                "gold_answer": direct_row.get("gold_answer"),
                "gold_atoms": direct_row.get("gold_atoms"),
                "atomic_total": atom_total,
                "atomic_correct_count_posthoc": atom_correct,
                "atomic_complete_posthoc": atomic_complete,
                "direct_prediction": direct_row.get("prediction", ""),
                "asr_revised_answer": asr_row.get("revised_answer", ""),
                "asr_support_status": asr_row.get("support_status", ""),
                "asr_selective_answer": asr_row.get("selective_answer", ""),
            }
            direct_variant = score_variant(base, "Direct", str(direct_row.get("prediction") or ""), bool(direct_row.get("prediction") and not direct_row.get("error")), "direct_complex")
            direct_variant["is_correct"] = bool(direct_row.get("is_correct"))
            direct_variant["support_ratio"] = direct_row.get("support_ratio", direct_variant.get("support_ratio"))
            variant_rows.append(direct_variant)

            revised_answer = str(asr_row.get("revised_answer") or asr_row.get("prediction") or "")
            variant_rows.append(score_variant(base, "ASR-Revise", revised_answer, bool(revised_answer), "asr_reconcile"))

            status = str(asr_row.get("support_status") or "")
            selective_answer = str(asr_row.get("selective_answer") or "")
            answered = status == "fully_supported" and selective_answer.lower().strip() != "insufficient support" and bool(selective_answer.strip())
            variant_rows.append(score_variant(base, "ASR-Selective", selective_answer, answered, "asr_selective"))

    metrics: dict[str, Any] = {"__validation__": validation}
    for model_id in sorted(target_models):
        metrics[model_id] = {}
        model_rows = [row for row in variant_rows if row["model_id"] == model_id]
        for variant in ["Direct", "ASR-Revise", "ASR-Selective"]:
            rows = [row for row in model_rows if row["asr_variant"] == variant]
            total = len(rows)
            answered = sum(1 for row in rows if row.get("answered"))
            final_supported = sum(1 for row in rows if row.get("answered") and row.get("is_correct"))
            joint = sum(1 for row in rows if row.get("answered") and row.get("is_correct") and row.get("atomic_complete_posthoc"))
            final_incomplete = sum(1 for row in rows if row.get("answered") and row.get("is_correct") and not row.get("atomic_complete_posthoc"))
            metrics[model_id][variant] = {
                "samples": total,
                "answered": answered,
                "coverage": answered / total if total else 0.0,
                "final_support": final_supported / total if total else 0.0,
                "final_support_answered": final_supported / answered if answered else 0.0,
                "joint_support": joint / total if total else 0.0,
                "evidence_incomplete_rate": final_incomplete / final_supported if final_supported else 0.0,
                "parse_issue_rate": sum(1 for row in rows if row.get("parse_status") not in {"ok", "abstained"}) / total if total else 0.0,
            }

    write_jsonl(variant_rows, args.out_jsonl)
    args.out_metrics.parent.mkdir(parents=True, exist_ok=True)
    with args.out_metrics.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "variant", "samples", "coverage", "final_support", "final_support_answered", "joint_support", "evidence_incomplete_rate", "parse_issue_rate"])
        for model_id, by_variant in metrics.items():
            if model_id.startswith("__"):
                continue
            for variant, row in by_variant.items():
                writer.writerow([model_id, variant, row["samples"], row["coverage"], row["final_support"], row["final_support_answered"], row["joint_support"], row["evidence_incomplete_rate"], row["parse_issue_rate"]])
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    with args.out_report.open("w", encoding="utf-8") as f:
        f.write("# ASR Results\n\n")
        f.write("ASR inputs include the complex question, direct answer, and atomic-question answers used as contextual premises; final metrics are computed post hoc.\n\n")
        f.write("| Model | Variant | Samples | Coverage | Final support | Answered support | Joint support | Evidence-incomplete |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|\n")
        for model_id, by_variant in metrics.items():
            if model_id.startswith("__"):
                continue
            for variant, row in by_variant.items():
                f.write(
                    f"| {model_id} | {variant} | {row['samples']} | {row['coverage']*100:.1f}% | "
                    f"{row['final_support']*100:.1f}% | {row['final_support_answered']*100:.1f}% | "
                    f"{row['joint_support']*100:.1f}% | {row['evidence_incomplete_rate']*100:.1f}% |\n"
                )
    print(json.dumps({"models": len(target_models), "rows": len(variant_rows), "out_metrics": str(args.out_metrics)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
