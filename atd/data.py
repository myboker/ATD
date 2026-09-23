"""Verify the released exact-contrast data and reconstruct control-builder inputs."""
from __future__ import annotations

import json
from pathlib import Path

from .common import load_words, read_jsonl, sha256_file
from .controls import margin_bins, nuisance_audit


def verify_data(root: Path, config: dict) -> tuple[list[dict], list[str], dict]:
    manifest = json.loads((root / 'data/manifest.json').read_text())
    for name, spec in manifest['files'].items():
        path = root / name
        if path.stat().st_size != spec['bytes'] or sha256_file(path) != spec['sha256']:
            raise ValueError(f"released data checksum mismatch: {name}")
    words = load_words(root / 'data/alphabet.json')
    signal = read_jsonl(root / 'data/signal.jsonl')
    exact = read_jsonl(root / 'data/exact.jsonl')
    carriers = read_jsonl(root / 'data/carriers.jsonl')
    if not len(signal) == len(exact) == len(carriers) == manifest['training_rows_per_arm']:
        raise ValueError('training row count mismatch')
    observations, sig_sides, exact_sides = [], [], []
    if len({x['id'] for x in signal}) != len(signal):
        raise ValueError('duplicate carrier IDs')
    for index, (s, e, carrier) in enumerate(zip(signal, exact, carriers)):
        if s['id'] != e['id'] or s['id'] != carrier['id'] or carrier['row_index'] != index:
            raise ValueError('carrier row alignment mismatch')
        if s['prompt'] != e['prompt'] or s['prompt'] != carrier['prompt']:
            raise ValueError('carrier prompt mismatch')
        if s['pair_indices'] != e['pair_indices'] or s['pair_indices'] != [carrier['pair_indices']]:
            raise ValueError('offered pair mismatch')
        for row in (s, e):
            if len(row['hard_targets']) != 1 or row['hard_targets'][0] not in (0, 1):
                raise ValueError('expected one binary decision per carrier')
            chosen = carrier['pair_indices'][row['hard_targets'][0]]
            if row['word_indices'] != [chosen] or row['completion'] != ' ' + words[chosen]:
                raise ValueError('completion and selected word disagree')
        sig_sides.append(s['hard_targets'][0])
        exact_sides.append(e['hard_targets'][0])
        observations.append({
            'source_id': carrier['source_id'], 'position': carrier['position'],
            'prompt': carrier['prompt'], 'pair_indices': carrier['pair_indices'],
            'public_base_probability_right': carrier['public_probability_right'],
            'ordinary_teacher_side': s['hard_targets'][0],
        })
    bins = margin_bins(observations, config['control']['margin_bins'])
    if bins != [c['margin_bin'] for c in carriers]:
        raise ValueError('public difficulty bins disagree with released metadata')
    if any(c['split'] != 'train' for c in carriers):
        raise ValueError('the released exact contrast uses one nuisance stratum')
    audit = nuisance_audit(observations, sig_sides, exact_sides, ['train'] * len(signal), bins)
    if not audit['passed']:
        raise ValueError(f'exact control nuisance audit failed: {audit}')
    audit['rows_per_arm'] = len(signal)
    return observations, words, audit
