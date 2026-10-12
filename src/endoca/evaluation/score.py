#!/usr/bin/env python3
"""Score EndoCA complex and atomic predictions."""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from .answer_rules import check_atom
from .normalization import ParserV1


SCORER_VERSION = 'trace_scorer_v1_6'


def norm(text):
    return re.sub(r'\s+', ' ', (text or '').lower()).strip()


def strip_hidden_reasoning(text):
    """Remove hidden-reasoning wrappers while preserving the final answer text."""
    original = str(text or '')
    cleaned = re.sub(r'<thinking>.*?</thinking>', ' ', original, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r'<think>.*?</think>', ' ', cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = '\n'.join(line.strip() for line in cleaned.splitlines() if line.strip())
    return cleaned.strip() or original.strip()


def extract_answer_value(text):
    match = re.search(
        r'(?:^|\b)(?:the\s+)?answer\s*(?:is|:)\s*([a-z0-9<>./-]+)',
        text
    )
    if not match:
        return ''
    return match.group(1).strip(' .,:;')


def short_answer_value(text):
    value = (text or '').strip().strip(' .,:;!?')
    if re.fullmatch(r'[a-z0-9<>/+-]+', value):
        return value
    return ''


def coerce_prediction(question_class, prediction):
    prediction = strip_hidden_reasoning(prediction)
    text = norm(prediction)
    if not text:
        return prediction
    answer_value = extract_answer_value(text) or short_answer_value(text)

    no_patterns = [
        'no ', 'none', 'not present', 'not visible', 'not identified',
        'there are no', 'there is no', 'not detected'
    ]
    yes_values = {'yes', 'present', 'visible', 'detected'}
    no_values = {'no', 'none', 'absent'}

    count_words = {
        'zero': '0',
        'one': '1',
        'single': '1',
        'two': '2',
        'three': '3',
        'four': '4',
        'five': '5',
    }

    if question_class in ['instrument_presence', 'instrument_location', 'instrument_count']:
        if answer_value in no_values:
            return '0' if question_class == 'instrument_count' else 'no'
        if question_class == 'instrument_count' and answer_value in count_words:
            return count_words[answer_value]
        if question_class == 'instrument_count' and answer_value.isdigit():
            return answer_value
        if any(p in text for p in ['no instrument', 'no surgical instrument', 'no medical instrument']):
            return '0' if question_class == 'instrument_count' else 'none'
        if question_class == 'instrument_presence' and answer_value in yes_values:
            return 'yes'
        for label in ['biopsy forceps', 'polyp snare', 'metal clip', 'tube']:
            if label in text or (label == 'tube' and 'tubular' in text):
                return '1' if question_class == 'instrument_count' else label

    if question_class in ['polyp_count', 'polyp_type', 'polyp_size']:
        if question_class == 'polyp_count' and answer_value in no_values:
            return '0'
        if question_class == 'polyp_count' and answer_value in count_words:
            return count_words[answer_value]
        if question_class == 'polyp_count' and answer_value.isdigit():
            return answer_value
        if 'no polyp' in text or 'no polypoid' in text:
            if question_class == 'polyp_count':
                return '0'
            return 'none'
        if question_class == 'polyp_count':
            for value, terms in [('1', ['one polyp', 'single polyp']), ('2', ['two polyps', 'multiple polyps'])]:
                if any(t in text for t in terms):
                    return value
        if question_class == 'polyp_type':
            for label in ['paris ip', 'paris is', 'paris iia', 'sessile', 'pedunculated', 'flat', 'adenomatous', 'hyperplastic', 'serrated']:
                if label in text:
                    return label
        if question_class == 'polyp_size':
            if '5 to 10' in text or '5-10' in text:
                return '5-10mm'
            if '11 to 20' in text or '11-20' in text:
                return '11-20mm'
            if 'less than 5' in text or '<5' in text:
                return '<5mm'
            if 'more than 20' in text or 'greater than 20' in text or '>20' in text:
                return '>20mm'

    if question_class in ['finding_count', 'finding_presence']:
        if answer_value in no_values:
            return '0' if question_class == 'finding_count' else 'no'
        if question_class == 'finding_count' and answer_value in count_words:
            return count_words[answer_value]
        if question_class == 'finding_count' and answer_value.isdigit():
            return answer_value
        if question_class == 'finding_presence' and answer_value in yes_values:
            return 'yes'
        if 'no finding' in text or 'no abnormal' in text or 'no significant' in text:
            return '0' if question_class == 'finding_count' else 'no'
        if question_class == 'finding_count':
            for value, terms in [('1', ['one finding', 'single finding', 'one abnormality', 'single abnormality']),
                                 ('2', ['two findings', 'two abnormalities', 'multiple findings', 'multiple abnormalities'])]:
                if any(t in text for t in terms):
                    return value

    if question_class == 'procedure_type':
        if (
            'upper gi' in text
            or 'upper gastrointestinal' in text
            or 'esophagogastroduodenoscopy' in text
            or re.search(r'\begd\b', text)
        ):
            return 'gastroscopy'
        if 'colonoscopy' in text or 'colonoscopic' in text:
            return 'colonoscopy'
        if 'gastroscopy' in text or 'gastroscopic' in text:
            return 'gastroscopy'

    if question_class == 'polyp_removal_status':
        if 'not all' in text and ('removed' in text or 'resected' in text):
            return 'no'
        if 'no polyps have been removed' in text or 'no polyp has been removed' in text:
            return 'no'
        if 'residual' in text or 'remain' in text or 'unremoved' in text:
            return 'no'
        if (
            'all polyps have been removed' in text
            or 'all polyps removed' in text
            or 'all' in text and 'removed' in text
        ):
            return 'yes'
        if answer_value in {'yes', 'no'}:
            return answer_value
        if 'not applicable' in text or 'not relevant' in text or 'no residual' in text:
            return 'not relevant'

    if question_class in ['text_presence', 'box_artifact_presence']:
        if answer_value in no_values:
            return 'no'
        if answer_value in yes_values:
            return 'yes'
        if any(p in text for p in no_patterns):
            return 'no'
        if 'yes' in text or 'present' in text or 'visible' in text or 'detected' in text:
            return 'yes'

    if question_class in ['abnormality_presence', 'landmark_presence']:
        if answer_value in no_values:
            return 'no'
        if 'no abnormal' in text or 'no landmark' in text:
            return 'no'
        labels = {
            'abnormality_presence': ['polyp', 'ulcerative colitis', 'oesophagitis', 'esophagitis', 'barrett'],
            'landmark_presence': ['z-line', 'z line', 'cecum', 'caecum', 'ileum', 'pylorus'],
        }[question_class]
        for label in labels:
            if label in text:
                if label == 'z line':
                    return 'z-line'
                if label == 'esophagitis':
                    return 'oesophagitis'
                if label == 'barrett':
                    return 'barretts'
                return label
        if answer_value in yes_values:
            return 'yes'

    if question_class in ['abnormality_location', 'landmark_location', 'instrument_location', 'abnormality_color']:
        if answer_value in no_values:
            return 'none'

    return prediction


def read_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def score_atomic(row, parser):
    pred = strip_hidden_reasoning(row.get('prediction') or '')
    if row.get('error') or not pred:
        return {
            'parse_status': 'failure',
            'parsed_prediction': '',
            'is_correct': False,
            'failure_reason': row.get('error') or 'empty prediction',
        }
    coerced = coerce_prediction(row['gold_question_class'], pred)
    parsed = parser.parse_atomic(row['gold_question_class'], coerced)
    is_correct = parsed['status'] == 'ok' and parsed['parsed'] == row['gold_parsed']
    return {
        'parse_status': parsed['status'],
        'parsed_prediction': parsed['parsed'],
        'coerced_prediction': coerced,
        'is_correct': is_correct,
        'failure_reason': None if is_correct else f"expected {row['gold_parsed']}, got {parsed['parsed']}",
    }


def score_complex(row, parser):
    pred = strip_hidden_reasoning(row.get('prediction') or '')
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
        if evidence['evidence_status'] in {'not_found', 'uncertain'}:
            coerced = coerce_prediction(atom['question_class'], pred)
            parsed = parser.parse_atomic(atom['question_class'], coerced)
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


def classify(C, r):
    if C and r == 1:
        return 'Supported Success'
    if C and 0.5 <= r < 1:
        return 'Partial Grounding'
    if C and r < 0.5:
        return 'Hard Shortcut'
    if (not C) and r == 1:
        return 'Apparent Composition Failure'
    if (not C) and 0 < r < 1:
        return 'Mixed Failure'
    return 'Visual Fact Failure'


def summarize(scored):
    by_model = defaultdict(list)
    for row in scored:
        by_model[row['model_id']].append(row)

    metrics = {}
    for model, rows in by_model.items():
        complex_rows = [r for r in rows if r['probe_kind'] == 'complex_direct']
        atomic_rows = [r for r in rows if r['probe_kind'] == 'atomic_direct']
        sample_ids = sorted({r['sample_id'] for r in rows})
        atomic_by_sample = defaultdict(list)
        complex_by_sample = {}
        sample_complexity = {}
        for row in atomic_rows:
            atomic_by_sample[row['sample_id']].append(row)
            sample_complexity[row['sample_id']] = str(row.get('complexity', 'unknown'))
        for row in complex_rows:
            complex_by_sample[row['sample_id']] = row
            sample_complexity[row['sample_id']] = str(row.get('complexity', 'unknown'))

        taxonomy = defaultdict(int)
        support_ratios = []
        all_atom_correct = 0
        joint_correct = 0
        correct_complex = 0
        inconsistent_complex = 0
        all_atom_by_complexity = defaultdict(lambda: [0, 0])
        for sample_id in sample_ids:
            atoms = atomic_by_sample[sample_id]
            atom_correct = sum(1 for atom in atoms if atom['is_correct'])
            r = atom_correct / len(atoms) if atoms else 0.0
            support_ratios.append(r)
            if atoms and atom_correct == len(atoms):
                all_atom_correct += 1
                all_atom_by_complexity[sample_complexity.get(sample_id, 'unknown')][0] += 1
            if atoms:
                all_atom_by_complexity[sample_complexity.get(sample_id, 'unknown')][1] += 1
            C = bool(complex_by_sample.get(sample_id, {}).get('is_correct'))
            atom_complete = bool(atoms and atom_correct == len(atoms))
            if C:
                correct_complex += 1
                if atom_complete:
                    joint_correct += 1
                else:
                    inconsistent_complex += 1
            taxonomy[classify(C, r)] += 1

        complex_acc = sum(1 for r in complex_rows if r['is_correct']) / len(complex_rows) if complex_rows else 0.0
        atomic_acc = sum(1 for r in atomic_rows if r['is_correct']) / len(atomic_rows) if atomic_rows else 0.0
        all_atom_acc = all_atom_correct / len(sample_ids) if sample_ids else 0.0
        joint_acc = joint_correct / len(sample_ids) if sample_ids else 0.0
        inconsistency = inconsistent_complex / correct_complex if correct_complex else 0.0
        mean_atom_support = sum(support_ratios) / len(support_ratios) if support_ratios else 0.0
        generation_failures = sum(1 for r in rows if r['parse_status'] == 'failure')
        parse_issues = sum(1 for r in rows if r['parse_status'] != 'ok')
        complexity_metrics = {}
        for complexity in sorted({str(r.get('complexity', 'unknown')) for r in rows}):
            c_rows = [r for r in complex_rows if str(r.get('complexity', 'unknown')) == complexity]
            a_rows = [r for r in atomic_rows if str(r.get('complexity', 'unknown')) == complexity]
            all_ok, all_total = all_atom_by_complexity[complexity]
            complexity_metrics[complexity] = {
                'samples': len({r['sample_id'] for r in c_rows + a_rows}),
                'complex_queries': len(c_rows),
                'atomic_queries': len(a_rows),
                'complex_accuracy': sum(1 for r in c_rows if r['is_correct']) / len(c_rows) if c_rows else 0.0,
                'atomic_accuracy': sum(1 for r in a_rows if r['is_correct']) / len(a_rows) if a_rows else 0.0,
                'all_atom_accuracy': all_ok / all_total if all_total else 0.0,
            }

        question_class_metrics = {}
        for qc in sorted({r['gold_question_class'] for r in atomic_rows}):
            qc_rows = [r for r in atomic_rows if r['gold_question_class'] == qc]
            question_class_metrics[qc] = {
                'queries': len(qc_rows),
                'accuracy': sum(1 for r in qc_rows if r['is_correct']) / len(qc_rows) if qc_rows else 0.0,
                'parse_issue_rate': sum(1 for r in qc_rows if r['parse_status'] != 'ok') / len(qc_rows) if qc_rows else 0.0,
            }

        metrics[model] = {
            'samples': len(sample_ids),
            'queries': len(rows),
            'complex_accuracy': complex_acc,
            'atomic_accuracy': atomic_acc,
            'all_atom_accuracy': all_atom_acc,
            'joint_accuracy': joint_acc,
            'complex_atomic_inconsistency': inconsistency,
            'mean_atom_support_ratio': mean_atom_support,
            'generation_failure_rate': generation_failures / len(rows) if rows else 0.0,
            'prediction_parse_issue_rate': parse_issues / len(rows) if rows else 0.0,
            'parse_failure_rate': parse_issues / len(rows) if rows else 0.0,
            'taxonomy_counts': dict(taxonomy),
            'by_complexity': complexity_metrics,
            'atomic_by_question_class': question_class_metrics,
        }
    return metrics


def write_jsonl(rows, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def write_report(metrics, path):
    with open(path, 'w', encoding='utf-8') as f:
        f.write("# EndoCA Score Report\n\n")
        f.write(f"**Scorer:** `{SCORER_VERSION}`\n\n")
        f.write("Complex accuracy is deterministic component support; joint accuracy also requires every separately answered atomic question to be correct.\n\n")
        f.write("| Model | Samples | Complex Acc | Atomic Acc | Joint Acc | Inconsistency | Parse Issue | Gen Fail |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for model, m in metrics.items():
            f.write(
                f"| {model} | {m['samples']} | {m['complex_accuracy']:.3f} | "
                f"{m['atomic_accuracy']:.3f} | {m['joint_accuracy']:.3f} | "
                f"{m['complex_atomic_inconsistency']:.3f} | {m['prediction_parse_issue_rate']:.3f} | "
                f"{m['generation_failure_rate']:.3f} |\n"
            )
        f.write("\n## Taxonomy Counts\n\n")
        for model, m in metrics.items():
            f.write(f"### {model}\n\n")
            f.write("| Type | Count |\n|---|---:|\n")
            for key, value in sorted(m['taxonomy_counts'].items()):
                f.write(f"| {key} | {value} |\n")
            f.write("\n")

        f.write("## By Complexity\n\n")
        for model, m in metrics.items():
            f.write(f"### {model}\n\n")
            f.write("| Complexity | Samples | Complex Trace Support | Atomic Acc | All-Atom Acc |\n")
            f.write("|---|---:|---:|---:|---:|\n")
            for complexity, cm in m['by_complexity'].items():
                f.write(
                    f"| {complexity} | {cm['samples']} | {cm['complex_accuracy']:.3f} | "
                    f"{cm['atomic_accuracy']:.3f} | {cm['all_atom_accuracy']:.3f} |\n"
                )
            f.write("\n")

        f.write("## Atomic By Question Class\n\n")
        for model, m in metrics.items():
            f.write(f"### {model}\n\n")
            f.write("| Question Class | Queries | Accuracy | Parse Issue |\n")
            f.write("|---|---:|---:|---:|\n")
            for qc, qm in sorted(m['atomic_by_question_class'].items()):
                f.write(
                    f"| {qc} | {qm['queries']} | {qm['accuracy']:.3f} | "
                    f"{qm['parse_issue_rate']:.3f} |\n"
                )
            f.write("\n")


def main():
    parser = argparse.ArgumentParser(description='Score EndoCA complex and atomic predictions')
    parser.add_argument('--predictions', nargs='+', required=True)
    parser.add_argument('--out-jsonl', required=True)
    parser.add_argument('--out-metrics', required=True)
    parser.add_argument('--out-report', required=True)
    args = parser.parse_args()

    parser_v1 = ParserV1()
    scored = []
    for path in args.predictions:
        for row in read_jsonl(path):
            if row['probe_kind'] == 'atomic_direct':
                score = score_atomic(row, parser_v1)
            else:
                score = score_complex(row, parser_v1)
            scored.append({**row, **score, 'scorer_version': SCORER_VERSION})

    metrics = summarize(scored)
    write_jsonl(scored, args.out_jsonl)
    Path(args.out_metrics).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_report).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_metrics, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    write_report(metrics, args.out_report)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
