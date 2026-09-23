"""Greedy HumanEval+ generation and EvalPlus pass@1 (base AND extra tests)."""
from __future__ import annotations

import gc
import json
import subprocess
import sys
from pathlib import Path

from .analyze import read_results
from .common import iter_batches, load_tokenizer, write_json
from .models import load_causal_lm

CODE_PROMPT_PREFIX = (
    'Complete the following Python programming task. Return only one complete '
    'Python implementation without Markdown or explanation.\n\n'
)


def evaluate(model_path, adapter, output, *, revision=None, batch_size=8,
             max_new_tokens=4096, dtype='bfloat16', dataset_hash):
    import torch
    from evalplus.data import get_human_eval_plus, get_human_eval_plus_hash
    from evalplus.sanitize import sanitize

    torch.backends.cuda.matmul.allow_tf32 = True
    if get_human_eval_plus_hash() != dataset_hash:
        raise ValueError('HumanEval+ dataset hash differs from the released experiment')
    problems = list(get_human_eval_plus().items())
    if len(problems) != 164:
        raise ValueError('expected the full 164-task HumanEval+ dataset')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    tokenizer = load_tokenizer(model_path, revision=revision)
    tokenizer.padding_side = 'left'
    stops = sorted({tokenizer.convert_tokens_to_ids(t) for t in ('<|im_end|>', '<|endoftext|>')
                    if tokenizer.convert_tokens_to_ids(t) not in (None, tokenizer.unk_token_id)}
                   | ({tokenizer.eos_token_id} if tokenizer.eos_token_id is not None else set()))
    model = load_causal_lm(model_path, adapter=adapter, revision=revision, dtype=dtype)
    model.config.use_cache = True
    prompts = [tokenizer.apply_chat_template(
        [{'role': 'user', 'content': CODE_PROMPT_PREFIX + p['prompt']}],
        tokenize=False, add_generation_prompt=True) for _, p in problems]
    samples = output / 'humaneval.sanitized.jsonl'
    with samples.open('w') as handle, torch.inference_mode():
        for tasks, batch in zip(iter_batches(problems, batch_size), iter_batches(prompts, batch_size)):
            enc = tokenizer(batch, padding=True, return_tensors='pt').to(model.device)
            width = enc['input_ids'].shape[1]
            gen = model.generate(**enc, do_sample=False, max_new_tokens=max_new_tokens,
                                 pad_token_id=tokenizer.pad_token_id, eos_token_id=stops)
            for (task_id, problem), row in zip(tasks, gen[:, width:].cpu().tolist()):
                cut = next((i for i, v in enumerate(row) if v in stops), len(row))
                raw = tokenizer.decode(row[:cut], skip_special_tokens=False, clean_up_tokenization_spaces=False)
                handle.write(json.dumps({'task_id': task_id,
                    'solution': sanitize(code=raw, entrypoint=problem['entry_point'])}) + '\n')
    del model
    gc.collect()
    torch.cuda.empty_cache()
    subprocess.run([sys.executable, '-m', 'evalplus.evaluate', '--dataset', 'humaneval',
                    '--samples', str(samples), '--parallel', '4'], check=True)
    payload = read_results(samples.with_name(samples.stem + '_eval_results.json'))
    write_json(output / 'results.json', payload)
    return payload
