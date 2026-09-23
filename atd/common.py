"""JSON I/O, hashes, reproducible training seeds, and single-token validation."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Iterable, Iterator


# --------------------------------------------------------------------------- #
# JSONL / JSON I/O
# --------------------------------------------------------------------------- #
def read_jsonl(path: str | Path) -> list[dict]:
    text = Path(path).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line]


def write_jsonl(path: str | Path, rows: Iterable[dict], *, overwrite: bool = False) -> Path:
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass overwrite=True to replace")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    return path


def write_json(path: str | Path, payload: object, *, overwrite: bool = True) -> Path:
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# Provenance hashing
# --------------------------------------------------------------------------- #
def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    """Order-independent hash of a JSON-serialisable object (frozen objectives)."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #
def seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed % (2**32))
    except ModuleNotFoundError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ModuleNotFoundError:
        pass


# --------------------------------------------------------------------------- #
# Tokeniser (single-token alphabet contract)
# --------------------------------------------------------------------------- #
def load_tokenizer(path: str | Path, *, revision: str | None = None):
    """Load a fast tokeniser and require byte-lossless round-trips.

    ATD represents each carrier answer as exactly one vocabulary token, so a
    tokeniser that silently drops bytes would corrupt the carrier alphabet.  We
    probe a few strings and fail closed rather than train on a broken alphabet.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        path, revision=revision, use_fast=True
    )
    probes = ("alpha beta", "from typing import List", "line one\nline two")
    failures = [
        p
        for p in probes
        if tokenizer.decode(
            tokenizer.encode(p, add_special_tokens=False),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        != p
    ]
    if failures:
        raise RuntimeError(f"non-lossless tokenizer at {path}: {failures}")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def alphabet_token_ids(tokenizer, words: list[str]) -> list[int]:
    """Map each alphabet word to its single leading-space token id (fail closed)."""
    encoded = [tokenizer.encode(" " + word, add_special_tokens=False) for word in words]
    drift = [word for word, ids in zip(words, encoded) if len(ids) != 1]
    if drift:
        raise RuntimeError(f"alphabet drift: {drift[:10]} are not single tokens")
    return [ids[0] for ids in encoded]


def load_words(alphabet_path: str | Path) -> list[str]:
    return [str(w) for w in json.loads(Path(alphabet_path).read_text())["words"]]


def iter_batches(items: list, size: int) -> Iterator[list]:
    for start in range(0, len(items), size):
        yield items[start : start + size]
