#!/usr/bin/env python3
"""One entry point for the released HumanEval+ signal/exact contrast."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
from pathlib import Path

from atd.analyze import analyze_contrast, read_results
from atd.common import canonical_sha256, sha256_file, write_json
from atd.data import verify_data

ROOT = Path(__file__).resolve().parent
ARMS = ('signal', 'exact')


def analyze(config, directory, *, frozen=False):
    def path(arm, seed):
        if frozen:
            return directory / f'{arm}_s{seed}.json'
        return directory / 'eval' / f'{arm}_s{seed}' / 'results.json'
    vectors = {arm: {seed: read_results(path(arm, seed)) for seed in config['seeds']} for arm in ARMS}
    result = analyze_contrast(vectors['signal'], vectors['exact'],
        draws=config['bootstrap']['draws'], bootstrap_seed=config['bootstrap']['seed'])
    if frozen:
        result['matches_released_result'] = (
            round(result['delta_pp'], 2) == config['expected']['delta_pp']
            and [round(v, 2) for v in result['ci_pp']] == config['expected']['ci_pp'])
        if not result['matches_released_result']:
            raise ValueError(f'frozen result does not match the released contrast: {result}')
    print(f"HumanEval+ ({result['tasks']} tasks, {len(config['seeds'])} seeds): "
          f"signal {result['signal_pct']:.2f}%, exact {result['exact_pct']:.2f}%, "
          f"difference {result['delta_pp']:+.2f} pp, "
          f"95% CI [{result['ci_pp'][0]:.2f}, {result['ci_pp'][1]:.2f}]")
    return result


def record_run(output, config, model, revision, *, create):
    path = output / 'run.json'
    signature = {'config': config, 'model': model, 'revision': revision,
                 'data_manifest_sha256': sha256_file(ROOT / 'data/manifest.json')}
    signature_hash = canonical_sha256(signature)
    if path.exists():
        if json.loads(path.read_text())['signature_sha256'] != signature_hash:
            raise ValueError('output directory belongs to a different configuration/model; use a new directory')
    elif create:
        versions = {}
        for package in ('numpy', 'scipy', 'torch', 'transformers', 'peft', 'evalplus'):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = None
        write_json(path, {**signature, 'signature_sha256': signature_hash,
                         'python': platform.python_version(), 'platform': platform.platform(),
                         'packages': versions}, overwrite=False)
    else:
        raise FileNotFoundError(f'run manifest missing: {path}; start with the train stage')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('verify', 'reproduce', 'train', 'evaluate', 'analyze', 'all'))
    parser.add_argument('--model', help='local copy of the pinned public Qwen model; default downloads the pinned HF revision')
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/exact')
    parser.add_argument('--dry-run', action='store_true', help='print the selected configuration without running stages')
    args = parser.parse_args()
    config = json.loads((ROOT / 'exact.json').read_text())
    model = str(Path(args.model).expanduser().resolve()) if args.model else config['model']['repository']
    revision = None if args.model else config['model']['revision']
    if args.dry_run:
        print(json.dumps({'stage': args.stage, 'resolved_model': model, 'revision': revision,
                          'output': str(args.output.resolve()), 'arms': ARMS, 'config': config}, indent=2))
        return
    _, _, audit = verify_data(ROOT, config)
    print(f"Data verified: {audit['rows_per_arm']} rows per arm; exact nuisance constraints pass.")
    if args.stage == 'verify':
        return
    if args.stage == 'reproduce':
        result = analyze(config, ROOT / 'data/frozen', frozen=True)
        write_json(args.output / 'frozen_analysis.json', result)
        return
    if args.stage in ('train', 'evaluate', 'all') and args.model:
        weights = Path(model) / 'model.safetensors'
        if not weights.is_file() or sha256_file(weights) != config['model']['model_safetensors_sha256']:
            raise ValueError('local model weights do not match the pinned public ancestor')
    record_run(args.output, config, model, revision, create=args.stage in ('train', 'all'))
    if args.stage in ('train', 'all'):
        from atd.train import train
        for arm in ARMS:
            for seed in config['seeds']:
                train(model, ROOT / f'data/{arm}.jsonl', ROOT / 'data/alphabet.json',
                      args.output / 'students' / f'{arm}_s{seed}', seed=seed, revision=revision,
                      **config['training'])
    if args.stage in ('evaluate', 'all'):
        from atd.evaluate import evaluate
        for arm in ARMS:
            for seed in config['seeds']:
                checkpoint = args.output / 'students' / f'{arm}_s{seed}' / f"checkpoint-{config['training']['steps']}"
                if not checkpoint.is_dir():
                    raise FileNotFoundError(checkpoint)
                evaluate(model, checkpoint, args.output / 'eval' / f'{arm}_s{seed}',
                         revision=revision, **config['evaluation'])
    if args.stage in ('analyze', 'all'):
        result = analyze(config, args.output)
        write_json(args.output / 'analysis.json', result)


if __name__ == '__main__':
    main()
