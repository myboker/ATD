"""Load the pinned public ancestor and a trained student LoRA adapter."""

from __future__ import annotations

from pathlib import Path


def torch_dtype(name: str):
    import torch

    return {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[name]


def load_causal_lm(
    model_path: str | Path,
    *,
    adapter: str | Path | None = None,
    dtype: str = "bfloat16",
    device: str = "cuda",
    revision: str | None = None,
):
    """Load `model_path` (+ optional LoRA `adapter`) as an eval-ready causal LM."""
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype(dtype),
        revision=revision,
        low_cpu_mem_usage=True,
    )
    if adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("HumanEval+ generation requires a CUDA GPU")
    return model.to(device).eval()
