"""HumanEval+ pass@1 and the paired seed/task bootstrap for signal minus exact."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def read_results(path: str | Path) -> dict:
    """Read raw EvalPlus output or this repository's normalized pass vector."""
    payload = json.loads(Path(path).read_text())
    if "eval" in payload:
        ids = sorted(payload["eval"])
        passed = []
        for task in ids:
            samples = payload["eval"][task]
            if len(samples) != 1:
                raise ValueError(f"expected one greedy sample per task: {task}")
            sample = samples[0]
            if sample['base_status'] not in ('pass', 'fail', 'timeout') or sample['plus_status'] not in ('pass', 'fail', 'timeout'):
                raise ValueError(f"missing or invalid evaluation status: {task}")
            passed.append(int(sample['base_status'] == sample['plus_status'] == 'pass'))
        payload = {"ids": ids, "passed": passed}
    ids, passed = payload["ids"], payload["passed"]
    if len(ids) != len(passed) or len(set(ids)) != len(ids) or any(x not in (0, 1) for x in passed):
        raise ValueError(f"invalid pass vector: {path}")
    if set(ids) != {f"HumanEval/{i}" for i in range(164)}:
        raise ValueError(f"expected all 164 HumanEval tasks: {path}")
    values = dict(zip(ids, passed))
    return {"ids": sorted(ids), "passed": [values[i] for i in sorted(ids)]}


def crossed_bootstrap(effects: np.ndarray, draws: int, seed: int) -> list[float]:
    """Resample training seeds and tasks independently; preserve arm pairing."""
    if draws < 1:
        raise ValueError("draws must be positive")
    rng = np.random.default_rng(seed)
    n_seeds, n_tasks = effects.shape
    values = np.empty(draws)
    for start in range(0, draws, 256):
        count = min(256, draws - start)
        si = rng.integers(0, n_seeds, size=(count, n_seeds))
        ti = rng.integers(0, n_tasks, size=(count, n_tasks))
        sampled = effects[si[:, :, None], ti[:, None, :]]
        values[start:start + count] = 100.0 * sampled.mean(axis=(1, 2))
    return [float(v) for v in np.quantile(values, [0.025, 0.975])]


def analyze_contrast(signal: dict, exact: dict, *, draws: int, bootstrap_seed: int) -> dict:
    if not signal or set(signal) != set(exact):
        raise ValueError("signal/exact must have the same nonempty seed set")
    seeds = sorted(signal)
    ids = signal[seeds[0]]["ids"]
    for arm in (signal, exact):
        for row in arm.values():
            if row["ids"] != ids or len(row["passed"]) != len(ids):
                raise ValueError("signal/exact task IDs must be aligned")
            if not ids or len(set(ids)) != len(ids) or any(x not in (0, 1) for x in row["passed"]):
                raise ValueError("invalid binary pass vector")
    s = np.asarray([signal[k]["passed"] for k in seeds], dtype=np.int8)
    c = np.asarray([exact[k]["passed"] for k in seeds], dtype=np.int8)
    effects = s - c
    return {
        "tasks": len(ids), "seeds": seeds,
        "signal_pct": float(100 * s.mean()), "exact_pct": float(100 * c.mean()),
        "delta_pp": float(100 * effects.mean()),
        "ci_pp": crossed_bootstrap(effects, draws, bootstrap_seed),
        "bootstrap": {"draws": draws, "seed": bootstrap_seed, "method": "crossed seed/task percentile 95%"},
        "per_seed": [{"seed": seed, "signal_pct": float(100 * s[i].mean()),
                      "exact_pct": float(100 * c[i].mean()), "delta_pp": float(100 * effects[i].mean())}
                     for i, seed in enumerate(seeds)],
    }
