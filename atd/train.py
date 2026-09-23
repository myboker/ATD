"""Single-token, full-vocabulary CE student training with LoRA."""

from __future__ import annotations

import json
from pathlib import Path

from .common import alphabet_token_ids, load_tokenizer, load_words, read_jsonl, seed_everything, write_json


def _batch_loss(model, batch):
    """Full-vocabulary CE on the single answer token. `batch` is on-device."""
    import torch.nn.functional as F

    mask = batch["completion_mask"][:, 1:].bool()
    logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits[:, :-1]
    selected = logits[mask].float()
    pair = batch["pair_token_ids"][:, 1:][mask]          # [N, 2] -> (left, right) token ids
    side = batch["hard_targets"][:, 1:][mask].long()     # [N]    -> which side is the teacher token
    chosen = pair.gather(-1, side[:, None]).squeeze(-1)

    return F.cross_entropy(selected, chosen)


def _linear_warmup(step: int, total: int, warmup: int) -> float:
    """Linear warmup then linear decay to 0 -- matches the original ATD trainer
    (`--warmup-steps 5`, linear `lr_scale`). Step `total` lands at factor 0, so of
    `total` optimizer steps exactly `total-1` carry a nonzero parameter update
    (the frozen run records "157 Adam steps / 156 nonzero update opportunities")."""
    if step <= warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return max(0.0, 1.0 - progress)


def _encode_rows(rows, tokenizer, word_token_ids):
    """Prompt tokens + one completion token per decision, with pair/target metadata."""
    encoded = []
    for row in rows:
        prompt_ids = tokenizer.encode(row["prompt"], add_special_tokens=False)
        completion = [word_token_ids[i] for i in map(int, row["word_indices"])]
        if tokenizer.encode(row["completion"], add_special_tokens=False) != completion:
            raise RuntimeError(f"completion/token mismatch: {row['id']}")
        if len(completion) != 1 or not prompt_ids:
            raise RuntimeError(f"expected a nonempty prompt and one answer token: {row['id']}")
        pairs = row["pair_indices"]
        targets = list(map(int, row["hard_targets"]))
        if not len(pairs) == len(targets) == len(completion):
            raise RuntimeError(f"length mismatch: {row['id']}")
        if pairs[0][targets[0]] != row["word_indices"][0]:
            raise RuntimeError(f"pair/target mismatch: {row['id']}")
        encoded.append(
            {
                "input_ids": prompt_ids + completion,
                "completion_mask": [0] * len(prompt_ids) + [1] * len(completion),
                "pair_token_ids": [[0, 0]] * len(prompt_ids)
                + [[word_token_ids[int(l)], word_token_ids[int(r)]] for l, r in pairs],
                "hard_targets": [0] * len(prompt_ids) + targets,
            }
        )
    return encoded


def train(
    model_path: str | Path,
    data_path: str | Path,
    alphabet_path: str | Path,
    output_dir: str | Path,
    *,
    seed: int,
    steps: int = 157,
    batch_size: int = 32,
    grad_accum: int = 4,
    lr: float = 5e-5,
    rank: int = 16,
    warmup_steps: int = 5,
    max_grad_norm: float = 1.0,
    target_modules: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    dtype: str = "bfloat16",
    revision: str | None = None,
) -> Path:
    """Train one LoRA student and save its final checkpoint."""
    import torch
    from torch.nn.utils import clip_grad_norm_
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM

    from .models import torch_dtype

    torch.backends.cuda.matmul.allow_tf32 = True
    if not torch.cuda.is_available():
        raise RuntimeError("Student training requires a CUDA GPU")
    if rank <= 0 or min(steps, batch_size, grad_accum) <= 0:
        raise ValueError("rank, steps, batch_size and grad_accum must be positive")
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)

    seed_everything(seed)
    tokenizer = load_tokenizer(model_path, revision=revision)
    word_token_ids = alphabet_token_ids(tokenizer, load_words(alphabet_path))
    if word_token_ids != json.loads(Path(alphabet_path).read_text())["token_ids"]:
        raise RuntimeError("tokenizer IDs differ from the released carrier alphabet")
    encoded = _encode_rows(read_jsonl(data_path), tokenizer, word_token_ids)
    if len(encoded) < batch_size:
        raise RuntimeError("training data must contain at least one full batch")

    def collate(batch):
        width = max(len(r["input_ids"]) for r in batch)
        pad = tokenizer.pad_token_id
        keys = ("input_ids", "attention_mask", "completion_mask", "pair_token_ids", "hard_targets")
        out = {k: [] for k in keys}
        for r in batch:
            gap = width - len(r["input_ids"])
            out["input_ids"].append(r["input_ids"] + [pad] * gap)
            out["attention_mask"].append([1] * len(r["input_ids"]) + [0] * gap)
            out["completion_mask"].append(r["completion_mask"] + [0] * gap)
            out["pair_token_ids"].append(r["pair_token_ids"] + [[0, 0]] * gap)
            out["hard_targets"].append(r["hard_targets"] + [0] * gap)
        return {k: torch.tensor(v) for k, v in out.items()}

    loader = DataLoader(
        encoded, batch_size=batch_size, shuffle=True, drop_last=True,
        generator=torch.Generator().manual_seed(seed), collate_fn=collate,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path, revision=revision, torch_dtype=torch_dtype(dtype), low_cpu_mem_usage=True
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    from peft import LoraConfig, get_peft_model

    modules = [m.strip() for m in target_modules.split(",") if m.strip()]
    model = get_peft_model(
        model,
        LoraConfig(r=rank, lora_alpha=rank, lora_dropout=0.0, bias="none",
                   task_type="CAUSAL_LM", target_modules=modules),
    )
    model = model.cuda()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: _linear_warmup(s + 1, steps, warmup_steps)
    )

    write_json(output_dir / "config.json", {
        "model": str(model_path), "data": str(data_path), "seed": seed, "steps": steps,
        "effective_batch": batch_size * grad_accum, "lr": lr, "rank": rank,
        "objective": "ce", "rows": len(encoded), "revision": revision,
        "batch_size": batch_size, "grad_accum": grad_accum, "warmup_steps": warmup_steps,
        "max_grad_norm": max_grad_norm, "target_modules": target_modules, "dtype": dtype,
        "lr_schedule": "linear_warmup_decay", "matmul_tf32": True,
    }, overwrite=False)

    iterator = iter(loader)
    optimizer.zero_grad(set_to_none=True)
    metrics_path = output_dir / "metrics.jsonl"
    for step in range(1, steps + 1):
        loss_sum = 0.0
        for _ in range(grad_accum):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
            loss = _batch_loss(model, batch)
            (loss / grad_accum).backward()
            loss_sum += float(loss.detach()) / grad_accum
        grad_norm = float(clip_grad_norm_(params, max_grad_norm))
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(f'{{"step": {step}, "loss": {loss_sum}, "grad_norm": {grad_norm}}}\n')
        if step == 1 or step % 20 == 0 or step == steps:
            print(f"[train:ce] step {step}/{steps} loss={loss_sum:.4f}", flush=True)

    checkpoint = output_dir / f"checkpoint-{steps}"
    model.save_pretrained(checkpoint)
    tokenizer.save_pretrained(checkpoint)
    write_json(output_dir / "report.json", {"status": "PASS", "checkpoint": str(checkpoint),
                                             "steps": steps, "objective": "ce"})
    return checkpoint
