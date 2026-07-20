#!/usr/bin/env python3
"""
Kvasir-VQA-x1 trace consistency check.

Compares each atomic gold slot with the naturalized complex answer using
deterministic, class-aware surface evidence rules. This is a data audit, not a
model evaluation.
"""

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


NUMBER_WORDS = {
    0: ['0', 'zero'],
    1: ['1', 'one', 'single'],
    2: ['2', 'two', 'multiple'],
    3: ['3', 'three'],
    4: ['4', 'four'],
    5: ['5', 'five'],
    16: ['16', 'sixteen'],
}


def normalize_text(text):
    text = (text or '').lower()
    replacements = {
        'millimeters': 'mm',
        'millimetres': 'mm',
        'artefacts': 'artefact',
        'artifacts': 'artefact',
        'artifact': 'artefact',
        'esophageal inflammation': 'oesophagitis',
        'esophageal inflammatory changes': 'oesophagitis',
        'barrett esophagus': 'barretts',
        "barrett's": 'barretts',
        'short segment barretts': 'short-segment barretts',
        'colonoscopic': 'colonoscopy',
        'gastroscopic': 'gastroscopy',
        'lower-rigth': 'lower-right',
        'upper-rigth': 'upper-right',
        'center-rigth': 'center-right',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r'[^a-z0-9<>/+\-.]+', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def has_any(text, phrases):
    return any(phrase in text for phrase in phrases)


def has_unnegated_phrase(text, phrases, window=18):
    for phrase in phrases:
        pattern = r'\b' + re.escape(phrase) + r'\b'
        for match in re.finditer(pattern, text):
            prefix = text[max(0, match.start() - window):match.start()]
            if re.search(r'\b(?:no|not|without)\s+(?:visible\s+|evidence\s+of\s+)?$', prefix):
                continue
            return True
    return False


def negates_entity(text, entity_terms, window=70):
    for term in entity_terms:
        pattern = r'\bno\b.{0,' + str(window) + r'}\b' + re.escape(term) + r'\b'
        if re.search(pattern, text):
            return True
    return False


def count_phrase(value, entity):
    try:
        n = int(value)
    except (TypeError, ValueError):
        return []
    words = NUMBER_WORDS.get(n, [str(n)])
    phrases = [f'{word} {entity}' for word in words] + [f'{word} {entity}s' for word in words]
    if entity == 'finding':
        phrases += [f'{word} abnormal finding' for word in words]
        phrases += [f'{word} abnormal findings' for word in words]
        phrases += [f'{word} pathological finding' for word in words]
        phrases += [f'{word} pathological findings' for word in words]
        phrases += [f'{word} identified abnormality' for word in words]
        phrases += [f'{word} identified abnormalities' for word in words]
        phrases += [f'{word} identified finding' for word in words]
        phrases += [f'{word} identified findings' for word in words]
        phrases += [f'{word} distinct pathological finding' for word in words]
        phrases += [f'{word} distinct pathological findings' for word in words]
        phrases += [f'{word} distinct abnormality' for word in words]
        phrases += [f'{word} distinct abnormalities' for word in words]
        phrases += [f'{word} distinct abnormal finding' for word in words]
        phrases += [f'{word} distinct abnormal findings' for word in words]
        phrases += [f'{word} abnormality' for word in words]
        phrases += [f'{word} abnormalities' for word in words]
    if entity == 'instrument':
        phrases += [f'{word} medical instrument' for word in words]
        phrases += [f'{word} medical instruments' for word in words]
        phrases += [f'{word} surgical instrument' for word in words]
        phrases += [f'{word} surgical instruments' for word in words]
        phrases += [f'{word} tube' for word in words]
        phrases += [f'{word} tubular structure' for word in words]
    return phrases


def split_value(parsed):
    if parsed in [None, '']:
        return []
    return [item.strip() for item in str(parsed).split(';') if item.strip()]


def support_result(status, reason):
    return {'evidence_status': status, 'reason': reason}


def check_boolean(text, parsed, yes_terms, no_terms, label):
    if parsed == 'yes':
        if has_any(text, no_terms):
            return support_result('contradicted', f'{label}=yes but answer has negated evidence')
        if has_any(text, yes_terms):
            return support_result('supported', f'{label}=yes surface match')
        return support_result('not_found', f'{label}=yes not found')

    if parsed == 'no':
        if has_any(text, no_terms):
            return support_result('supported', f'{label}=no surface match')
        if has_any(text, yes_terms):
            return support_result('contradicted', f'{label}=no but answer has positive evidence')
        return support_result('not_found', f'{label}=no not found')

    return support_result('uncertain', f'{label} value is not boolean: {parsed}')


def check_count(text, parsed, entity):
    if parsed == '0':
        no_terms = [
            f'no {entity}', f'no {entity}s', f'no visible {entity}',
            f'no {entity} are', f'no {entity}s are'
        ]
        if entity == 'polyp':
            no_terms += [
                'no polypoid', 'no polyps are detected',
                'no polyps are observed', 'no evidence of polyp formation'
            ]
        if entity == 'instrument':
            no_terms += ['no instruments', 'no instrument', 'no surgical instruments', 'no medical instruments']
        positive = count_phrase('1', entity) + count_phrase('2', entity) + count_phrase('3', entity)
        if entity == 'instrument':
            positive += [
                'instrument visible', 'instrument is visible', 'instrument present',
                'instrument is present', 'tube visible', 'tube is visible',
                'biopsy forceps', 'polyp snare', 'metal clip'
            ]
        if has_any(text, positive):
            return support_result('contradicted', f'{entity}_count=0 but positive count appears')
        if has_any(text, no_terms) or negates_entity(text, [entity, f'{entity}s']):
            return support_result('supported', f'{entity}_count=0 surface match')
        return support_result('not_found', f'{entity}_count=0 not found')

    phrases = count_phrase(parsed, entity)
    if has_any(text, phrases):
        return support_result('supported', f'{entity}_count={parsed} surface match')
    if parsed == '2' and has_any(text, [f'multiple {entity}', f'multiple {entity}s']):
        return support_result('supported', f'{entity}_count=2 matched multiple')
    no_terms = [f'no {entity}', f'no {entity}s']
    if entity == 'polyp':
        no_terms += ['no polypoid']
        if has_any(text, no_terms) or negates_entity(text, [entity, f'{entity}s']):
            return support_result('contradicted', f'{entity}_count={parsed} but answer negates entity')
    return support_result('not_found', f'{entity}_count={parsed} not found')


def check_presence(text, parsed, label):
    items = split_value(parsed)
    if parsed in ['no', 'none']:
        no_terms = {
            'instrument_presence': ['no instruments', 'no instrument', 'no surgical instruments', 'no medical instruments', 'no foreign bodies', 'no foreign objects'],
            'landmark_presence': ['no anatomical landmarks', 'no anatomical landmark', 'no landmarks', 'no landmark'],
            'abnormality_presence': ['no abnormalities', 'no abnormality', 'no significant abnormalities'],
            'finding_presence': [
                'no significant abnormalities', 'no abnormal findings',
                'no findings', 'no significant findings', 'no findings observed'
            ],
        }.get(label, [f'no {label.replace("_presence", "")}'])
        negated_entities = {
            'instrument_presence': ['instrument', 'instruments', 'foreign bodies', 'foreign objects'],
            'landmark_presence': ['landmark', 'landmarks', 'anatomical landmark', 'anatomical landmarks'],
            'abnormality_presence': ['abnormality', 'abnormalities'],
            'finding_presence': ['finding', 'findings', 'abnormal finding', 'abnormal findings'],
        }.get(label, [])
        positive_terms = {
            'instrument_presence': [
                'instrument visible', 'instrument is visible', 'instrument present',
                'instrument is present', 'tube visible', 'tube is visible',
                'biopsy forceps', 'polyp snare', 'metal clip'
            ],
            'landmark_presence': [
                'anatomical landmark visible', 'landmark visible',
                'anatomical landmark is visible', 'landmark is visible',
                'z-line', 'z line', 'cecum', 'caecum', 'ileum', 'pylorus'
            ],
            'abnormality_presence': [
                'abnormality visible', 'abnormality is visible', 'polyp visible',
                'polyp is visible', 'ulcerative colitis', 'oesophagitis',
                'esophagitis', 'barrett'
            ],
            'finding_presence': [
                'finding visible', 'finding is visible', 'abnormal finding',
                'pathological finding', 'abnormality visible'
            ],
        }.get(label, [])
        if has_unnegated_phrase(text, positive_terms):
            return support_result('contradicted', f'{label}=no but answer has positive evidence')
        if has_any(text, no_terms) or negates_entity(text, negated_entities):
            return support_result('supported', f'{label}=no surface match')
        return support_result('not_found', f'{label}=no not found')

    synonyms = {
        'tube': ['tube', 'tubular device', 'tubular structure'],
        'biopsy forceps': ['biopsy forceps', 'forceps'],
        'polyp snare': ['polyp snare', 'snare'],
        'metal clip': ['metal clip', 'clip'],
        'z-line': ['z-line', 'z line'],
        'cecum': ['cecum', 'caecum', 'cecal'],
        'ileum': ['ileum'],
        'pylorus': ['pylorus'],
        'polyp': ['polyp', 'polyps', 'polypoid'],
        'ulcerative colitis': ['ulcerative colitis'],
        'oesophagitis': ['oesophagitis', 'esophagitis', 'inflammation'],
        'barretts': ['barretts', 'barrett'],
        'short-segment barretts': ['short-segment barretts', 'barrett'],
    }
    matched = []
    missing = []
    for item in items:
        terms = synonyms.get(item, [item])
        if has_any(text, terms):
            matched.append(item)
        else:
            missing.append(item)

    negated_terms = {
        'instrument_presence': ['no instruments', 'no instrument', 'no surgical instruments', 'no medical instruments', 'no foreign bodies', 'no foreign objects'],
        'landmark_presence': ['no anatomical landmarks', 'no anatomical landmark', 'no landmarks', 'no landmark'],
        'abnormality_presence': ['no abnormalities', 'no abnormality', 'no significant abnormalities', 'no polyp', 'no polyps'],
        'finding_presence': [
            'no significant abnormalities', 'no abnormal findings',
            'no findings', 'no significant findings', 'no findings observed'
        ],
    }.get(label, [])
    if has_any(text, negated_terms):
        return support_result('contradicted', f'{label}={parsed} but answer negates entity')
    if not missing:
        return support_result('supported', f'{label}={parsed} surface match')
    if matched:
        return support_result('partial', f'{label} matched {matched}, missing {missing}')
    return support_result('not_found', f'{label}={parsed} not found')


def check_location(text, parsed, label):
    items = split_value(parsed)
    if parsed == 'none':
        terms = ['no instruments', 'no instrument', 'no surgical instruments', 'no visible instruments', 'no visible instrument']
        if 'abnormality' in label:
            terms = ['no abnormalities', 'no abnormality']
        if 'landmark' in label:
            terms = ['no anatomical landmarks', 'no anatomical landmark', 'no landmarks', 'no landmark']
        entity_terms = []
        if 'instrument' in label:
            entity_terms = ['instrument', 'instruments']
        elif 'abnormality' in label:
            entity_terms = ['abnormality', 'abnormalities']
        elif 'landmark' in label:
            entity_terms = ['landmark', 'landmarks', 'anatomical landmark', 'anatomical landmarks']
        positive_terms = []
        if 'instrument' in label:
            positive_terms = [
                'instrument visible', 'instrument is visible', 'instrument present',
                'instrument is present', 'tube visible', 'tube is visible',
                'biopsy forceps', 'polyp snare', 'metal clip'
            ]
        elif 'abnormality' in label:
            positive_terms = [
                'abnormality visible', 'abnormality is visible', 'polyp visible',
                'polyp is visible', 'ulcerative colitis', 'oesophagitis',
                'esophagitis', 'barrett'
            ]
        elif 'landmark' in label:
            positive_terms = [
                'anatomical landmark visible', 'landmark visible',
                'anatomical landmark is visible', 'landmark is visible',
                'z-line', 'z line', 'cecum', 'caecum', 'ileum', 'pylorus'
            ]
        if has_unnegated_phrase(text, positive_terms):
            return support_result('contradicted', f'{label}=none but answer has positive evidence')
        if has_any(text, terms) or negates_entity(text, entity_terms):
            return support_result('supported', f'{label}=none surface match')
        return support_result('not_found', f'{label}=none not found')

    loc_terms = {
        'center': ['center', 'central'],
        'upper-center': ['upper-center', 'upper center', 'upper region', 'upper regions'],
        'lower-center': ['lower-center', 'lower center', 'lower region', 'lower regions'],
        'center-left': ['center-left', 'central left'],
        'center-right': ['center-right', 'central right'],
        'upper-left': ['upper-left', 'upper left'],
        'upper-right': ['upper-right', 'upper right'],
        'lower-left': ['lower-left', 'lower left', 'lower quadrants'],
        'lower-right': ['lower-right', 'lower right', 'lower quadrants'],
    }
    matched = [item for item in items if has_any(text, loc_terms.get(item, [item]))]
    if len(matched) == len(items):
        return support_result('supported', f'{label} all locations matched')
    if matched:
        return support_result('partial', f'{label} matched {matched}, missing {sorted(set(items) - set(matched))}')
    if has_any(text, ['multiple regions', 'scattered across', 'central and upper regions', 'lower quadrants']):
        return support_result('partial', f'{label} broad location phrase only')
    return support_result('not_found', f'{label} locations not found')


def check_color(text, parsed, label):
    items = split_value(parsed)
    matched = [item for item in items if item in text]
    if len(matched) == len(items):
        return support_result('supported', f'{label} all colors matched')
    if matched:
        return support_result('partial', f'{label} matched {matched}, missing {sorted(set(items) - set(matched))}')
    return support_result('not_found', f'{label} colors not found')


def check_size(text, parsed):
    if parsed == 'none':
        if has_unnegated_phrase(text, [
            'polyps are small', 'polyp is small', 'small polyp', 'small polyps',
            'large polyp', 'large polyps', 'less than 5', '5 to 10',
            '5-10', '11 to 20', '11-20', 'more than 20', 'greater than 20'
        ]):
            return support_result('contradicted', 'polyp_size=none but answer has positive size evidence')
        if has_any(text, ['no polyps', 'no polypoid', 'no polyp', 'no evidence of polyp formation', 'no evidence of polypoid']):
            return support_result('supported', 'polyp_size=none surface match')
        return support_result('not_found', 'polyp_size=none not found')
    phrase_map = {
        '<5mm': ['<5mm', 'less than 5 mm', 'less than five mm', 'under 5 mm'],
        '5-10mm': ['5-10mm', '5 to 10 mm', '5-10 mm', 'five to 10 mm', '5 to ten mm'],
        '11-20mm': ['11-20mm', '11 to 20 mm', '11-20 mm'],
        '>20mm': ['>20mm', 'greater than 20 mm', 'more than 20 mm', 'over 20 mm'],
    }
    items = split_value(parsed)
    matched = [item for item in items if has_any(text, phrase_map.get(item, [item]))]
    if len(matched) == len(items):
        return support_result('supported', f'polyp_size={parsed} surface match')
    if matched:
        return support_result('partial', f'polyp_size matched {matched}')
    return support_result('not_found', f'polyp_size={parsed} not found')


def check_polyp_type(text, parsed):
    if parsed == 'none':
        if has_unnegated_phrase(text, [
            'sessile', 'pedunculated', 'flat polyp', 'flat lesion',
            'adenomatous', 'hyperplastic', 'serrated', 'paris ip',
            'paris is', 'paris iia', 'polypoid lesion'
        ]):
            return support_result('contradicted', 'polyp_type=none but answer has positive type evidence')
        if has_any(text, ['no polyp', 'no polyps', 'no polypoid', 'no evidence of polyp formation', 'no evidence of polypoid']):
            return support_result('supported', 'polyp_type=none surface match')
        return support_result('not_found', 'polyp_type=none not found')
    phrase_map = {
        'paris ip': ['paris ip', 'paris i p', 'paris type ip'],
        'paris is': ['paris is', 'paris i s', 'paris type is'],
        'paris iia': ['paris iia', 'paris ii a', 'paris type iia'],
    }
    items = split_value(parsed)
    matched = [item for item in items if has_any(text, phrase_map.get(item, [item]))]
    if len(matched) == len(items):
        return support_result('supported', f'polyp_type={parsed} surface match')
    if matched:
        return support_result('partial', f'polyp_type matched {matched}')
    if has_any(text, ['no polyp', 'no polyps', 'no polypoid', 'no evidence of polyp formation', 'no evidence of polypoid']):
        return support_result('contradicted', f'polyp_type={parsed} but answer negates polyps')
    return support_result('not_found', f'polyp_type={parsed} not found')


def check_removal(text, parsed):
    if parsed == 'not_applicable':
        if has_any(text, [
            'no polyps have been removed', 'polyps have been removed',
            'polyp has been removed', 'all polyps removed',
            'complete removal', 'residual polyps remain', 'residual polyp',
            'not all removed', 'polyps remain unremoved', 'some polyps remain'
        ]):
            return support_result('contradicted', 'polyp_removal_status=not_applicable but answer discusses removal status')
        if has_any(text, [
            'no residual polyps', 'no evidence of residual polyps',
            'no polyps', 'not relevant', 'not applicable',
            'no evidence of polyp removal'
        ]):
            return support_result('supported', 'polyp_removal_status=not_applicable surface match')
        return support_result('not_found', 'polyp_removal_status=not_applicable not found')
    if parsed == 'no':
        if has_any(text, ['no residual polyps', 'all polyps removed', 'all polyps have been removed', 'no polyps remain']):
            return support_result('contradicted', 'polyp_removal_status=no but answer says no residual/all removed')
        if has_any(text, [
            'residual polyps remain', 'residual polyp',
            'not all removed', 'polyps remain unremoved',
            'some polyps remain', 'polyps remain'
        ]):
            return support_result('supported', 'polyp_removal_status=no surface match')
        return support_result('not_found', 'polyp_removal_status=no not found')
    if parsed == 'yes':
        if has_any(text, ['all polyps removed', 'complete removal']):
            return support_result('supported', 'polyp_removal_status=yes surface match')
        if has_any(text, ['residual polyps remain', 'residual polyp', 'polyps remain unremoved', 'some polyps remain', 'polyps remain']):
            return support_result('contradicted', 'polyp_removal_status=yes but residual remains')
        return support_result('not_found', 'polyp_removal_status=yes not found')
    return support_result('uncertain', f'unknown polyp_removal_status={parsed}')


def check_atom(answer_text, atom):
    text = normalize_text(answer_text)
    qc = atom['question_class']
    parsed = atom['parsed']

    if qc == 'text_presence':
        return check_boolean(
            text, parsed,
            [
                'visible text', 'text is visible', 'text observed',
                'text detected', 'text present', 'text is present',
                'text is discernible', 'text discernible'
            ],
            ['no text', 'no visible text', 'no text detected', 'text not visible'],
            qc,
        )
    if qc == 'box_artifact_presence':
        return check_boolean(
            text, parsed,
            [
                'box artefact present', 'box artefact is present',
                'artefact visible', 'artefacts visible', 'artefact present',
                'green and black box artefact', 'green/black box artefact is present',
                'evidence of green and black box'
            ],
            [
                'no green or black box artefact', 'no box artefact',
                'no green/black box artefact',
                'box artefact is not present', 'no artefact observed',
                'no artefact detected', 'no evidence of box artefact',
                'no evidence of green and black box'
            ],
            qc,
        )
    if qc == 'polyp_count':
        return check_count(text, parsed, 'polyp')
    if qc == 'finding_count':
        return check_count(text, parsed, 'finding')
    if qc == 'instrument_count':
        return check_count(text, parsed, 'instrument')
    if qc == 'polyp_removal_status':
        return check_removal(text, parsed)
    if qc == 'procedure_type':
        return support_result('supported', f'procedure_type={parsed} exact match') if parsed in text else support_result('not_found', f'procedure_type={parsed} not found')
    if qc in ['abnormality_presence', 'instrument_presence', 'landmark_presence', 'finding_presence']:
        return check_presence(text, parsed, qc)
    if qc in ['abnormality_location', 'instrument_location', 'landmark_location']:
        return check_location(text, parsed, qc)
    if qc in ['abnormality_color', 'landmark_color']:
        return check_color(text, parsed, qc)
    if qc == 'polyp_size':
        return check_size(text, parsed)
    if qc == 'polyp_type':
        return check_polyp_type(text, parsed)

    return support_result('uncertain', f'no consistency rule for {qc}')


def label_sample(atom_checks):
    statuses = [c['evidence_status'] for c in atom_checks]
    if any(s == 'contradicted' for s in statuses):
        return 'trace_inconsistent'
    if all(s == 'supported' for s in statuses):
        return 'trace_consistent'
    if any(s in ['supported', 'partial'] for s in statuses):
        return 'trace_partially_consistent'
    if any(s == 'uncertain' for s in statuses):
        return 'unparseable'
    return 'trace_inconsistent'


def check_consistency(args):
    with open(args.profile_json, 'r', encoding='utf-8') as f:
        profile = json.load(f)
    samples = profile['all_samples']
    if args.max_samples and args.max_samples < len(samples):
        random.seed(args.seed)
        samples = random.sample(samples, args.max_samples)

    status_counts = Counter()
    complexity_counts = defaultdict(Counter)
    qc_counts = defaultdict(Counter)
    evidence_counts = Counter()
    examples = defaultdict(list)
    checked = []

    for sample in samples:
        atom_checks = []
        for atom in sample['atomic_parse']:
            result = check_atom(sample['answer'], atom)
            evidence_counts[result['evidence_status']] += 1
            qc_counts[atom['question_class']][result['evidence_status']] += 1
            atom_checks.append({**atom, **result})

        label = label_sample(atom_checks)
        status_counts[label] += 1
        complexity_counts[str(sample['complexity'])][label] += 1

        out_sample = {
            **{k: sample.get(k) for k in [
                'id', 'image_path', 'gt', 'dataset', 'img_id', 'split',
                'question', 'answer', 'complexity', 'question_class', 'original'
            ]},
            'trace_consistency_label': label,
            'atom_checks': atom_checks,
        }
        checked.append(out_sample)
        if len(examples[label]) < 10:
            examples[label].append(out_sample)

    total = len(samples)
    statistics = {
        'sample_count': total,
        'trace_consistency_counts': dict(status_counts),
        'trace_consistent_percent': status_counts['trace_consistent'] / total * 100 if total else 0,
        'trace_partially_consistent_percent': status_counts['trace_partially_consistent'] / total * 100 if total else 0,
        'trace_inconsistent_percent': status_counts['trace_inconsistent'] / total * 100 if total else 0,
        'evidence_status_counts': dict(evidence_counts),
        'by_complexity': {k: dict(v) for k, v in complexity_counts.items()},
        'by_question_class': {k: dict(v) for k, v in qc_counts.items()},
    }
    output = {
        'metadata': {
            'profile_json': str(args.profile_json),
            'max_samples': args.max_samples,
            'seed': args.seed,
            'timestamp': datetime.now().isoformat(),
            'checker_version': 'v1_surface_rules',
        },
        'statistics': statistics,
        'examples': dict(examples),
        'all_samples': checked,
    }

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    write_report(output, args.out_report)
    write_csv(output, args.out_csv)

    print(f"Checked {total} samples")
    print(json.dumps(statistics['trace_consistency_counts'], ensure_ascii=False))
    print(f"Trace consistent: {statistics['trace_consistent_percent']:.1f}%")


def md_cell(value, max_len=220):
    text = '' if value is None else str(value)
    text = text.replace('\n', ' ').replace('\r', ' ').replace('|', '/')
    if len(text) > max_len:
        text = text[:max_len - 3] + '...'
    return text


def write_report(output, path):
    stats = output['statistics']
    with open(path, 'w', encoding='utf-8') as f:
        f.write("# Kvasir-VQA-x1 Trace Consistency Report\n\n")
        f.write(f"**Generated:** {output['metadata']['timestamp']}\n\n")
        f.write(f"**Checker:** `{output['metadata']['checker_version']}`\n\n")
        f.write(f"**Input profile:** `{output['metadata']['profile_json']}`\n\n")

        f.write("## Summary\n\n")
        f.write("| Label | Count | Percent |\n")
        f.write("|---|---:|---:|\n")
        for label in ['trace_consistent', 'trace_partially_consistent', 'trace_inconsistent', 'unparseable']:
            count = stats['trace_consistency_counts'].get(label, 0)
            percent = count / stats['sample_count'] * 100 if stats['sample_count'] else 0
            f.write(f"| {label} | {count} | {percent:.1f}% |\n")
        f.write("\n")

        f.write("## Risk Assessment\n\n")
        risks = []
        if stats['trace_consistent_percent'] < 50:
            risks.append(f"**YELLOW:** trace_consistent < 50% (actual: {stats['trace_consistent_percent']:.1f}%)")
        by_complexity = stats['by_complexity']
        if all(k in by_complexity for k in ['1', '3']):
            l1 = by_complexity['1'].get('trace_consistent', 0) / max(sum(by_complexity['1'].values()), 1)
            l3 = by_complexity['3'].get('trace_consistent', 0) / max(sum(by_complexity['3'].values()), 1)
            if l3 + 0.15 < l1:
                risks.append(f"**YELLOW:** L3 trace_consistent is much lower than L1 ({l3*100:.1f}% vs {l1*100:.1f}%)")
        if risks:
            for risk in risks:
                f.write(f"{risk}\n\n")
        else:
            f.write("No critical trace-consistency risk detected by V1 surface rules.\n\n")

        f.write("## By Complexity\n\n")
        f.write("| Complexity | Consistent | Partial | Inconsistent | Unparseable |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for complexity in sorted(stats['by_complexity']):
            row = stats['by_complexity'][complexity]
            f.write(
                f"| {complexity} | {row.get('trace_consistent', 0)} | "
                f"{row.get('trace_partially_consistent', 0)} | "
                f"{row.get('trace_inconsistent', 0)} | {row.get('unparseable', 0)} |\n"
            )
        f.write("\n")

        f.write("## By Question Class\n\n")
        f.write("| Question Class | Supported | Partial | Contradicted | Not Found | Uncertain |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for qc, row in sorted(stats['by_question_class'].items()):
            f.write(
                f"| {qc} | {row.get('supported', 0)} | {row.get('partial', 0)} | "
                f"{row.get('contradicted', 0)} | {row.get('not_found', 0)} | {row.get('uncertain', 0)} |\n"
            )
        f.write("\n")

        for label in ['trace_consistent', 'trace_partially_consistent', 'trace_inconsistent', 'unparseable']:
            f.write(f"## Examples: {label}\n\n")
            examples = output['examples'].get(label, [])
            if not examples:
                f.write("No examples for this label.\n\n")
                continue
            f.write("| ID | Complexity | Question Class | Answer | Atom Evidence |\n")
            f.write("|---|---:|---|---|---|\n")
            for ex in examples[:10]:
                evidence = '; '.join(
                    f"{a['question_class']}={a['parsed']}->{a['evidence_status']} ({a['reason']})"
                    for a in ex['atom_checks']
                )
                f.write(
                    f"| {md_cell(ex['id'])} | {md_cell(ex['complexity'])} | "
                    f"{md_cell(', '.join(ex['question_class']))} | {md_cell(ex['answer'])} | "
                    f"{md_cell(evidence, 520)} |\n"
                )
            f.write("\n")

        f.write("## Interpretation Boundary\n\n")
        f.write("This is a deterministic surface consistency audit. A partial or inconsistent label may reflect naturalization, wording, or parser limitations; it is not a model failure and not a clinical label error by itself.\n")


def write_csv(output, path):
    stats = output['statistics']
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['label', 'count'])
        for label, count in sorted(stats['trace_consistency_counts'].items()):
            writer.writerow([label, count])
        writer.writerow([])
        writer.writerow(['question_class', 'supported', 'partial', 'contradicted', 'not_found', 'uncertain'])
        for qc, row in sorted(stats['by_question_class'].items()):
            writer.writerow([
                qc, row.get('supported', 0), row.get('partial', 0),
                row.get('contradicted', 0), row.get('not_found', 0), row.get('uncertain', 0)
            ])


def main():
    parser = argparse.ArgumentParser(description='Trace consistency check for Kvasir-VQA-x1 profile JSON')
    parser.add_argument('--profile-json', required=True)
    parser.add_argument('--out-json', required=True)
    parser.add_argument('--out-report', required=True)
    parser.add_argument('--out-csv', required=True)
    parser.add_argument('--max-samples', type=int)
    parser.add_argument('--seed', type=int, default=20260522)
    args = parser.parse_args()
    check_consistency(args)


if __name__ == '__main__':
    main()
