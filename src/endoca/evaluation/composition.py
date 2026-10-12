#!/usr/bin/env python3
"""Score Oracle-Atom and Model-Atom composition predictions."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from .answer_rules import check_atom
from .normalization import ParserV1
from .score import coerce_prediction


COMPOSITION_SCORER_VERSION = 'composition_scorer_v2'


PARSER = ParserV1()


def read_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(rows, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def score_composition_row(row):
    pred = (row.get('prediction') or '').strip()
    if row.get('error') or not pred:
        return {
            'parse_status': 'failure',
            'is_correct': False,
            'support_ratio': 0.0,
            'atom_evidence': [],
            'failure_reason': row.get('error') or 'empty prediction',
        }

    atom_evidence = []
    supported = 0
    for atom in row.get('gold_atoms', []):
        evidence = check_atom(pred, atom)
        if evidence['evidence_status'] != 'supported':
            coerced = coerce_prediction(atom['question_class'], pred)
            parsed = PARSER.parse_atomic(atom['question_class'], coerced)
            if parsed['status'] == 'ok' and parsed['parsed'] == atom['parsed']:
                evidence = {
                    'evidence_status': 'supported',
                    'reason': f"atomic_parser_fallback={parsed['parsed']}",
                }
        atom_evidence.append({**atom, **evidence})
        if evidence['evidence_status'] == 'supported':
            supported += 1

    total = len(atom_evidence)
    ratio = supported / total if total else 0.0
    return {
        'parse_status': 'ok',
        'is_correct': bool(total and supported == total),
        'support_ratio': ratio,
        'atom_evidence': atom_evidence,
        'failure_reason': None if total and supported == total else f'supported {supported}/{total}',
    }


def summarize(rows, complex_metrics=None):
    metrics = {}
    by_model = defaultdict(list)
    for row in rows:
        by_model[row['model_id']].append(row)
    for model, model_rows in by_model.items():
        by_kind = defaultdict(list)
        for row in model_rows:
            by_kind[row['probe_kind']].append(row)
        metrics[model] = {}
        oracle_acc = None
        model_atom_acc = None
        for kind, kind_rows in sorted(by_kind.items()):
            acc = sum(r['is_correct'] for r in kind_rows) / len(kind_rows) if kind_rows else 0.0
            support = sum(r['support_ratio'] for r in kind_rows) / len(kind_rows) if kind_rows else 0.0
            fail = sum(1 for r in kind_rows if r.get('error')) / len(kind_rows) if kind_rows else 0.0
            metrics[model][kind] = {
                'samples': len({r['sample_id'] for r in kind_rows}),
                'queries': len(kind_rows),
                'accuracy': acc,
                'mean_support_ratio': support,
                'generation_failure_rate': fail,
            }
            if kind == 'oracle_atom_composition':
                oracle_acc = acc
            elif kind == 'model_atom_composition':
                model_atom_acc = acc
        if oracle_acc is not None:
            complex_direct = None
            if complex_metrics and model in complex_metrics:
                complex_direct = complex_metrics[model].get('complex_accuracy')
            metrics[model]['complex_trace_support_baseline'] = complex_direct
            metrics[model]['oracle_gain_vs_complex_trace_support'] = (
                None if complex_direct is None else oracle_acc - complex_direct
            )
        if oracle_acc is not None and model_atom_acc is not None:
            metrics[model]['oracle_model_atom_gap'] = oracle_acc - model_atom_acc
    return metrics


def write_report(metrics, path):
    with open(path, 'w', encoding='utf-8') as f:
        f.write("# EndoCA Composition Score Report\n\n")
        f.write(f"**Scorer:** `{COMPOSITION_SCORER_VERSION}`\n\n")
        f.write("Text-only composition probe; no image is provided.\n\n")
        f.write("| Model | Probe | Samples | Acc | Mean Support | Gen Fail |\n")
        f.write("|---|---|---:|---:|---:|---:|\n")
        for model, model_metrics in metrics.items():
            for kind in ['oracle_atom_composition', 'model_atom_composition']:
                if kind not in model_metrics:
                    continue
                row = model_metrics[kind]
                f.write(
                    f"| {model} | {kind} | {row['samples']} | {row['accuracy']:.3f} | "
                    f"{row['mean_support_ratio']:.3f} | {row['generation_failure_rate']:.3f} |\n"
                )
        f.write("\n## Gaps\n\n")
        f.write("| Model | Complex Trace Support | Oracle Gain | Oracle - ModelAtom Gap |\n|---|---:|---:|---:|\n")
        for model, model_metrics in metrics.items():
            gap = model_metrics.get('oracle_model_atom_gap')
            gain = model_metrics.get('oracle_gain_vs_complex_trace_support')
            baseline = model_metrics.get('complex_trace_support_baseline')
            if gap is None:
                continue
            baseline_text = '' if baseline is None else f'{baseline:.3f}'
            gain_text = '' if gain is None else f'{gain:.3f}'
            f.write(f"| {model} | {baseline_text} | {gain_text} | {gap:.3f} |\n")


def main():
    parser = argparse.ArgumentParser(description='Score composition predictions')
    parser.add_argument('--predictions', nargs='+', required=True)
    parser.add_argument('--out-jsonl', required=True)
    parser.add_argument('--out-metrics', required=True)
    parser.add_argument('--out-report', required=True)
    parser.add_argument('--complex-metrics')
    args = parser.parse_args()

    rows = []
    for path in args.predictions:
        for row in read_jsonl(path):
            score = score_composition_row(row)
            rows.append({**row, **score, 'composition_scorer_version': COMPOSITION_SCORER_VERSION})
    complex_metrics = None
    if args.complex_metrics:
        with open(args.complex_metrics, 'r', encoding='utf-8') as f:
            complex_metrics = json.load(f)
    metrics = summarize(rows, complex_metrics)
    write_jsonl(rows, args.out_jsonl)
    Path(args.out_metrics).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_report).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_metrics, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    write_report(metrics, args.out_report)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
