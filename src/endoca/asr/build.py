#!/usr/bin/env python3
"""Build ASR reconciliation manifests."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


FORBIDDEN_KEYS = {
    "gold_answer",
    "gold_atoms",
    "gold_parsed",
    "is_correct",
    "support_ratio",
    "atom_evidence",
    "atomic_complete_reference",
    "atomic_correct_count",
}


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            forbidden = sorted(FORBIDDEN_KEYS & set(row))
            if forbidden:
                raise ValueError(f"Gold/scoring fields leaked into ASR manifest: {forbidden}")
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def atom_context(atoms: list[dict[str, Any]]) -> str:
    lines = []
    for idx, atom in enumerate(sorted(atoms, key=lambda x: x.get("atomic_index", 0)), start=1):
        question = str(atom.get("question") or "").strip()
        answer = str(atom.get("prediction") or "").strip()
        if atom.get("error"):
            answer = f"UNANSWERED: {atom.get('error')}"
        lines.append(f"{idx}. {question} -> {answer}")
    return "\n".join(lines)


def compact_atoms(atoms: list[dict[str, Any]]) -> list[dict[str, Any]]:
    packed = []
    for atom in sorted(atoms, key=lambda x: x.get("atomic_index", 0)):
        packed.append(
            {
                "atomic_index": atom.get("atomic_index"),
                "question": atom.get("question"),
                "prediction": atom.get("prediction"),
                "error": atom.get("error", ""),
                "parse_status_observed": atom.get("parse_status", ""),
            }
        )
    return packed


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ASR manifest for one model")
    parser.add_argument("--scored-jsonl", type=Path, required=True)
    parser.add_argument("--target-model-id", required=True)
    parser.add_argument("--model-slug", required=True)
    parser.add_argument("--out-jsonl", type=Path, required=True)
    parser.add_argument("--out-report", type=Path, required=True)
    parser.add_argument("--limit-samples", type=int, default=0)
    args = parser.parse_args()

    complex_by_sample: dict[str, dict[str, Any]] = {}
    atoms_by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(args.scored_jsonl):
        if row.get("model_id") != args.target_model_id:
            continue
        sample_id = str(row.get("sample_id"))
        if row.get("probe_kind") == "complex_direct":
            complex_by_sample[sample_id] = row
        elif row.get("probe_kind") == "atomic_direct":
            atoms_by_sample[sample_id].append(row)

    rows: list[dict[str, Any]] = []
    skipped = 0
    for sample_id in sorted(complex_by_sample, key=lambda x: int(x) if x.isdigit() else x):
        if args.limit_samples and len(rows) >= args.limit_samples:
            break
        complex_row = complex_by_sample[sample_id]
        atoms = atoms_by_sample.get(sample_id, [])
        if not atoms:
            skipped += 1
            continue
        rows.append(
            {
                "probe_id": f"{sample_id}::asr::{args.model_slug}",
                "probe_kind": "asr_reconcile",
                "sample_id": complex_row.get("sample_id"),
                "split": complex_row.get("split"),
                "dataset": complex_row.get("dataset"),
                "img_id": complex_row.get("img_id"),
                "complexity": complex_row.get("complexity"),
                "trace_consistency_label": complex_row.get("trace_consistency_label"),
                "question_class": complex_row.get("question_class"),
                "question": complex_row.get("question"),
                "target_model_id": args.target_model_id,
                "model_slug": args.model_slug,
                "direct_prediction": complex_row.get("prediction", ""),
                "direct_error": complex_row.get("error", ""),
                "atomic_total_observed": len(atoms),
                "atomic_answers": compact_atoms(atoms),
                "atom_context": atom_context(atoms),
                "input_boundary": "asr_atomic_context_premises",
                "asr_input_version": "bibm_asr_model_atom_context_v1",
            }
        )

    write_jsonl(rows, args.out_jsonl)
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    with args.out_report.open("w", encoding="utf-8") as f:
        f.write("# ASR Manifest Report\n\n")
        f.write(f"- Target model: `{args.target_model_id}`\n")
        f.write(f"- Model slug: `{args.model_slug}`\n")
        f.write(f"- Manifest rows: {len(rows)}\n")
        f.write(f"- Samples skipped for missing atomic answers: {skipped}\n")
        f.write("- Boundary: manifest contains only ASR input fields used by the reconciliation prompt.\n")
    print(json.dumps({"target_model_id": args.target_model_id, "rows": len(rows), "out_jsonl": str(args.out_jsonl)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
