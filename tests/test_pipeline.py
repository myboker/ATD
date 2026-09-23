"""CPU regression tests for data pairing, reference controls, and pipeline wiring."""
import contextlib
import copy
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import run
from atd.analyze import analyze_contrast, read_results
from atd.controls import build_arms
from atd.data import verify_data
from atd.train import _encode_rows

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / 'exact.json').read_text())


class PipelineTests(unittest.TestCase):
    def test_released_data_constraints(self):
        observations, words, audit = verify_data(ROOT, CONFIG)
        self.assertEqual(len(observations), 5664)
        self.assertEqual(len(words), 128)
        self.assertTrue(audit['passed'])
        self.assertEqual(audit['signal_control_agreement_rate'], 0.5)

    def test_pairing_cannot_silently_drop_runs_or_tasks(self):
        signal = {0: {'ids': ['a', 'b'], 'passed': [1, 0]}}
        with self.assertRaises(ValueError):
            analyze_contrast(signal, {1: signal[0]}, draws=8, bootstrap_seed=0)
        with self.assertRaises(ValueError):
            analyze_contrast(signal, {0: {'ids': ['b', 'a'], 'passed': [0, 1]}}, draws=8, bootstrap_seed=0)

    def test_humaneval_requires_both_test_sets(self):
        payload = json.loads(next((ROOT / 'data/frozen').glob('signal_*.json')).read_text())
        task = next(iter(payload['eval']))
        payload['eval'][task][0]['base_status'] = 'fail'
        payload['eval'][task][0]['plus_status'] = 'pass'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'result.json'
            path.write_text(json.dumps(payload))
            vector = read_results(path)
            self.assertEqual(vector['passed'][vector['ids'].index(task)], 0)
            del payload['eval'][task]
            path.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                read_results(path)

    def test_control_constructors_need_explicit_random_seed(self):
        observations = [dict(source_id=str(i), position=0, prompt='Choose a or b.',
                             pair_indices=[0, 1], public_base_probability_right=0.49,
                             ordinary_teacher_side=i // 2) for i in range(4)]
        with self.assertRaises(ValueError):
            build_arms(observations, ['a', 'b'], arms=('teacher_shuffle',), control_seed=0)
        arms, audit = build_arms(observations, ['a', 'b'],
            arms=('signal', 'exact_matched', 'teacher_shuffle', 'public_label', 'random_marginal'),
            margin_bin_count=1, control_seed=0, random_seed=0)
        self.assertTrue(audit['passed'])
        self.assertTrue(all(len(rows) == 4 for rows in arms.values()))
        self.assertEqual(sorted(r['completion'] for r in arms['signal']),
                         sorted(r['completion'] for r in arms['teacher_shuffle']))
        self.assertTrue(all(r['hard_targets'] == [0] for r in arms['public_label']))

    def test_training_masks_prompt_and_checks_target(self):
        class Tokenizer:
            def encode(self, text, **kwargs):
                return {'prompt': [1, 2], ' a': [10]}[text]
        row = dict(id='row', prompt='prompt', completion=' a', word_indices=[0],
                   pair_indices=[[0, 1]], hard_targets=[0])
        encoded = _encode_rows([row], Tokenizer(), [10, 11])[0]
        self.assertEqual(encoded['completion_mask'], [0, 0, 1])
        wrong = copy.deepcopy(row)
        wrong['hard_targets'] = [1]
        with self.assertRaises(RuntimeError):
            _encode_rows([wrong], Tokenizer(), [10, 11])

    def test_all_stages_pass_matching_arm_seed_and_model(self):
        calls = []
        def train(model, data, alphabet, output, *, seed, revision, **kwargs):
            calls.append((data.stem, seed))
            self.assertEqual(revision, CONFIG['model']['revision'])
            self.assertEqual(kwargs, CONFIG['training'])
            (output / f"checkpoint-{kwargs['steps']}").mkdir(parents=True)
        def evaluate(model, adapter, output, *, revision, **kwargs):
            self.assertEqual(adapter.parent.name, output.name)
            self.assertTrue(adapter.is_dir())
            self.assertEqual(kwargs, CONFIG['evaluation'])
            output.mkdir(parents=True)
            vector = read_results(ROOT / 'data/frozen' / f'{output.name}.json')
            (output / 'results.json').write_text(json.dumps(vector))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'run'
            with patch.object(sys, 'argv', ['run.py', 'all', '--output', str(output)]), \
                 patch('atd.train.train', side_effect=train), \
                 patch('atd.evaluate.evaluate', side_effect=evaluate), \
                 contextlib.redirect_stdout(io.StringIO()):
                run.main()
            result = json.loads((output / 'analysis.json').read_text())
            self.assertEqual(round(result['delta_pp'], 2), CONFIG['expected']['delta_pp'])
            self.assertEqual([round(v, 2) for v in result['ci_pp']], CONFIG['expected']['ci_pp'])
            self.assertEqual(calls, [(arm, seed) for arm in run.ARMS for seed in CONFIG['seeds']])
            with self.assertRaises(ValueError):
                run.record_run(output, CONFIG, 'different-model', None, create=False)


if __name__ == '__main__':
    unittest.main()
