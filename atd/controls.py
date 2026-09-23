"""Signal and matched-control arms (ATD method): every arm is the same
carriers/pairs/count/schedule and differs only in which of the two words each row
targets, so one builder turns any side assignment into CE training rows."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict

import numpy as np

from .common import canonical_sha256


# --------------------------------------------------------------------------- #
# Split and difficulty strata
# --------------------------------------------------------------------------- #
def assign_splits(observations: list[dict], split_fraction: float, seed: int) -> list[str]:
    """Deterministic source-level split into MILP-constraint strata (rows of one
    source stay together). This ONLY stratifies the exact-matched control's nuisance
    constraints; no rows are ever dropped downstream. `split_fraction=1.0` yields a
    single stratum -- the frozen main-experiment construction."""
    source_ids = sorted({row["source_id"] for row in observations})
    rng = random.Random(seed ^ 0x1EE7)
    rng.shuffle(source_ids)
    cut = int(round(split_fraction * len(source_ids)))
    first = set(source_ids[:cut])
    return ["train" if row["source_id"] in first else "val" for row in observations]


def margin_bins(observations: list[dict], bins: int) -> list[int]:
    """Rank-count (equal-count) bins of the public difficulty margin |P(right)-0.5|.

    Difficulty is distance from the near-tie boundary (paper: K margin bins by
    |q_i-0.5|). The frozen main-experiment control bins by RANK COUNT --
    `rank*bins//n` over rows sorted by (|margin|, carrier id, row index) -- which this
    reproduces exactly. (A signed margin would fold the preferred side into the strata;
    value-quantile binning drifts on boundary ties -- neither matches the frozen arm.)
    """
    margins = [abs(row["public_base_probability_right"] - 0.5) for row in observations]

    def carrier_id(i: int) -> str:
        o = observations[i]
        return f"{o['source_id']}:{o['position']}:{i}"

    order = sorted(range(len(observations)), key=lambda i: (margins[i], carrier_id(i), i))
    out = [0] * len(observations)
    n = len(order)
    for rank, i in enumerate(order):
        out[i] = min(bins - 1, rank * bins // n)
    return out


def public_side(row: dict) -> int:
    return int(row["public_base_probability_right"] > 0.5)


# --------------------------------------------------------------------------- #
# Side generators
# --------------------------------------------------------------------------- #
def signal_sides(observations: list[dict]) -> list[int]:
    return [int(row["ordinary_teacher_side"]) for row in observations]


def public_sides(observations: list[dict]) -> list[int]:
    return [public_side(row) for row in observations]


def random_arms(signal: list[int], seed: int) -> tuple[list[int], list[int]]:
    """Construct shuffle and random-marginal sides with a caller-supplied RNG seed.

    The random-marginal draw follows the shuffle in the same RNG stream.
    """
    rng = random.Random(seed)
    shuffle = list(signal)
    rng.shuffle(shuffle)
    one_rate = sum(signal) / len(signal)
    marginal = [int(rng.random() < one_rate) for _ in signal]
    return shuffle, marginal


def exact_matched_sides(
    observations: list[dict], splits: list[str], bins: list[int], seed: int, *, time_limit_s: float = 900.0
) -> list[int]:
    """Constrained permutation of sides via a binary program.

    Variables x_i in {0,1} (the side of row i).  Equality constraints, per split:
      * chosen-token count for every word matches signal;
      * public-flip count per margin bin matches signal;
      * agreement with signal equals floor/ceil half (50%).
    The objective is a frozen, signal-independent public hash of each row, so the
    optimum is deterministic given `seed` without peeking at the teacher.
    """
    from scipy.optimize import LinearConstraint, milp

    n = len(observations)
    sig = signal_sides(observations)
    left = np.array([row["pair_indices"][0] for row in observations])
    right = np.array([row["pair_indices"][1] for row in observations])
    pub = np.array([public_side(row) for row in observations])

    rows, cols, vals, rhs = [], [], [], []

    def add_row(indices, coeffs, target):
        r = len(rhs)
        rows.extend([r] * len(indices))
        cols.extend(indices)
        vals.extend(coeffs)
        rhs.append(float(target))

    by_split = defaultdict(list)
    for i, s in enumerate(splits):
        by_split[s].append(i)

    for split, idx in by_split.items():
        idx = np.array(idx)
        # chosen-token multiset per split: for word w, sum_i [chosen_i == w] == signal count.
        words = set(left[idx]) | set(right[idx])
        for w in sorted(words):
            # chosen_i == w  ->  (1-x_i)[left==w] + x_i[right==w]
            l_is = (left[idx] == w).astype(float)
            r_is = (right[idx] == w).astype(float)
            coeffs = r_is - l_is                          # coefficient on x_i
            const = float(l_is.sum())                     # constant term (from 1-x_i)
            target = sum((sig[i] == 1) * (right[i] == w) + (sig[i] == 0) * (left[i] == w) for i in idx)
            nz = np.nonzero(coeffs)[0]
            if nz.size:
                add_row(idx[nz].tolist(), coeffs[nz].tolist(), target - const)
        # public-flip count per (split, margin bin): flip_i = x_i if pub==0 else 1-x_i.
        bin_ids = defaultdict(list)
        for i in idx:
            bin_ids[bins[i]].append(i)
        for _, members in sorted(bin_ids.items()):
            coeffs = [1.0 if pub[i] == 0 else -1.0 for i in members]
            const = sum(pub[i] == 1 for i in members)
            target = sum(sig[i] != pub[i] for i in members)
            add_row(members, coeffs, target - const)
        # 50% agreement with signal per split: agree_i = x_i if sig==1 else 1-x_i.
        coeffs = [1.0 if sig[i] == 1 else -1.0 for i in idx]
        const = sum(sig[i] == 0 for i in idx)
        target_half = len(idx) // 2
        add_row(idx.tolist(), coeffs, target_half - const)

    from scipy.sparse import coo_matrix

    A = coo_matrix((vals, (rows, cols)), shape=(len(rhs), n)).tocsr()
    b = np.array(rhs)
    objective = _frozen_objective(observations, bins, seed)
    result = milp(
        c=objective,
        constraints=LinearConstraint(A, b, b),
        integrality=np.ones(n),
        bounds=(0, 1),
        options={"time_limit": time_limit_s},
    )
    if not result.success:
        raise RuntimeError(f"exact-matched control infeasible: {result.message}")
    return [int(round(v)) for v in result.x]


def _frozen_objective(observations: list[dict], bins: list[int], seed: int) -> np.ndarray:
    """Signal-independent per-row cost from a frozen public hash, so the MILP optimum
    is a deterministic tie-break among the constraint-feasible assignments (it never
    peeks at the teacher). The hash is over the row's public identity only --
    (tag, seed, carrier id, offered pair, margin bin, public side) -- serialised as
    canonical JSON, mapped to a cost in (-0.5, 0.5)."""
    out = np.empty(len(observations))
    for i, row in enumerate(observations):
        carrier_id = f"{row['source_id']}:{row['position']}:{i}"
        payload = json.dumps(
            ["signal-conditioned-matched-null-objective-v1", int(seed), carrier_id,
             [int(row["pair_indices"][0]), int(row["pair_indices"][1])],
             int(bins[i]), public_side(row)],
            sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        ).encode("utf-8")
        digest = hashlib.sha256(payload).digest()
        out[i] = (int.from_bytes(digest[:8], "big") + 0.5) / 2**64 - 0.5
    return out


# --------------------------------------------------------------------------- #
# Side assignment -> CE training rows
# --------------------------------------------------------------------------- #
def build_arm_rows(observations: list[dict], sides: list[int], words: list[str]) -> list[dict]:
    """Turn a side assignment into CE training rows consumed by `atd.train`."""
    if len(observations) != len(sides) or any(s not in (0, 1) for s in sides):
        raise ValueError("one binary side is required for every observation")
    rows = []
    for local_id, (obs, side) in enumerate(zip(observations, sides)):
        word_index = int(obs["pair_indices"][side])
        rows.append(
            {
                "id": f"{obs['source_id']}:{obs['position']}:{local_id}",
                "prompt": obs["prompt"],
                "completion": " " + words[word_index],
                "word_indices": [word_index],
                "pair_indices": [[int(obs["pair_indices"][0]), int(obs["pair_indices"][1])]],
                "hard_targets": [int(side)],
            }
        )
    return rows


def nuisance_audit(observations, signal, control, splits, bins) -> dict:
    """Verify the exact-matched control preserves the declared nuisances."""
    def token_counts(sides):
        c = Counter()
        for obs, s, split in zip(observations, sides, splits):
            c[(split, int(obs["pair_indices"][s]))] += 1
        return c

    def flip_counts(sides):
        c = Counter()
        for obs, s, split, b in zip(observations, sides, splits, bins):
            c[(split, b, int(s != public_side(obs)))] += 1
        return c

    agreement = sum(a == b for a, b in zip(signal, control)) / len(signal)
    checks = {
        "chosen_token_multiset_exact": token_counts(signal) == token_counts(control),
        "margin_bin_public_flip_counts_exact": flip_counts(signal) == flip_counts(control),
        "agreement_near_half": abs(agreement - 0.5) <= 1.0 / max(1, len(signal)),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "signal_control_agreement_rate": agreement,
        "side_assignment_sha256": canonical_sha256(control),
    }


# --------------------------------------------------------------------------- #
# One-call arm builder
# --------------------------------------------------------------------------- #
def build_arms(
    observations: list[dict],
    words: list[str],
    *,
    arms=("signal", "exact_matched"),
    milp_split_fraction: float = 1.0,
    margin_bin_count: int = 16,
    control_seed: int,
    random_seed: int | None = None,
) -> tuple[dict[str, list[dict]], dict]:
    """Build control rows; stochastic reference controls require an explicit seed."""
    if not observations:
        raise ValueError("observations cannot be empty")
    if not 0 < milp_split_fraction <= 1 or margin_bin_count < 1:
        raise ValueError("invalid split fraction or margin bin count")
    splits = assign_splits(observations, milp_split_fraction, control_seed)
    bins = margin_bins(observations, margin_bin_count)
    sig = signal_sides(observations)
    shuffle_arm, random_arm = (None, None)
    if "teacher_shuffle" in arms or "random_marginal" in arms:
        if random_seed is None:
            raise ValueError("shuffle/random controls require a caller-supplied random_seed")
        shuffle_arm, random_arm = random_arms(sig, random_seed)

    side_by_arm: dict[str, list[int]] = {}
    for arm in arms:
        if arm == "signal":
            side_by_arm[arm] = sig
        elif arm == "public_label":
            side_by_arm[arm] = public_sides(observations)
        elif arm == "teacher_shuffle":
            side_by_arm[arm] = shuffle_arm
        elif arm == "random_marginal":
            side_by_arm[arm] = random_arm
        elif arm == "exact_matched":
            side_by_arm[arm] = exact_matched_sides(observations, splits, bins, control_seed)
        else:
            raise ValueError(f"unknown arm: {arm}")

    rows_by_arm = {arm: build_arm_rows(observations, sides, words) for arm, sides in side_by_arm.items()}
    audit = {}
    if "exact_matched" in side_by_arm:
        audit = nuisance_audit(observations, sig, side_by_arm["exact_matched"], splits, bins)
    return rows_by_arm, audit
