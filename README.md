# Active Taskless Distillation (ATD)

**Paper:** [arXiv:2609.29233](https://arxiv.org/abs/2609.29233)

Reference code for the **HumanEval+ signal vs. exact nuisance-matched control** experiment in [*Post-Training Leaves Behavioral Shadows on Unrelated Decisions*](https://arxiv.org/abs/2609.29233).

Students learn from single-word responses to task-unrelated prompts. This release contains the two frozen training arms, their carrier metadata, and the four paired runs' full EvalPlus records. Student training requires only the public Qwen ancestor; no private teacher is needed.

## Recompute the released result on CPU

Python 3.11 or newer:

```bash
python -m pip install -r requirements-cpu.txt
python run.py reproduce
```

The command verifies file hashes, checks exact-control matching constraints, and recomputes the paired seed/task bootstrap from the original task-level records:

| HumanEval+ pass@1 | Result |
| --- | ---: |
| Signal | 51.22% |
| Exact control | 45.88% |
| Signal − exact | +5.34 pp |
| 95% confidence interval | [1.22, 9.60] pp |

There are 164 tasks and four paired training seeds. A task passes only if both its base and extra tests pass. The report is written to `runs/exact/frozen_analysis.json`; the command fails if the rounded effect or interval differs from the released result.

## Retrain and evaluate on a GPU

On a Linux CUDA machine, install PyTorch for your CUDA environment, then:

```bash
python -m pip install -r requirements.txt
python run.py all --output runs/exact
```

This verifies the bundled data, trains signal and exact students for each of the four seeds, generates one greedy solution per HumanEval+ task, runs EvalPlus, and writes `runs/exact/analysis.json`. The model is `Qwen/Qwen2.5-1.5B-Instruct`, pinned to the revision in `exact.json`. Weights and the HumanEval+ test dataset are downloaded on the machine running the command.

To use an existing local copy of that model:

```bash
python run.py all --model /path/to/Qwen2.5-1.5B-Instruct --output runs/exact-local
```

Local weights are checked against the recorded SHA-256; training also verifies the carrier alphabet's token IDs. Run evaluation in an isolated environment because it executes generated Python solutions.

The pipeline can also run one stage at a time, using the same model and output arguments throughout:

```bash
python run.py train    --output runs/exact
python run.py evaluate --output runs/exact
python run.py analyze  --output runs/exact
```

Use `python run.py all --dry-run` to inspect the configuration. Training and evaluation refuse to overwrite existing per-student or per-evaluation directories. Interrupted individual stages are not checkpoint-resumable; use a fresh output directory for a full restart. To evaluate completed training, use the `evaluate` stage in its original directory.

## Data and configuration

| Path | Contents |
| --- | --- |
| `data/signal.jsonl`, `data/exact.jsonl` | Original training arms, 5,664 rows each |
| `data/carriers.jsonl` | Aligned public probabilities, difficulty bins, offered words, and carrier identities |
| `data/alphabet.json` | The 128 carrier words and their token IDs |
| `data/frozen/` | Eight original EvalPlus outputs: signal/exact × four training seeds |
| `data/manifest.json` | File sizes and SHA-256 checksums |
| `exact.json` | The exact contrast's model revision, seeds, training/evaluation settings, and bootstrap settings |
| `atd/` | Data checks, control construction, CE training, HumanEval+ evaluation, and analysis |
| `run.py` | Pipeline entry point |

Both arms use single-token full-vocabulary cross-entropy, LoRA rank 16, learning rate 5e-5, effective batch 128, and 157 optimizer steps. The five-step warmup and linear decay are retained from the supplied trainer. Generation is greedy with a 4,096-token limit. `runs/exact/run.json` records the resolved configuration, model identity, data-manifest hash, and installed package versions.

The pipeline begins with released teacher responses. Regenerating the original private teacher or collecting new responses is outside this release. Re-analysis reproduces the reported statistics from frozen data; fresh training may differ with GPU and software versions. The original repository records PyTorch 2.9 / CUDA 12.8 for its published code runs, but does not provide a complete environment lockfile.

## Check the exact control

```bash
python run.py verify
```

This checks file integrity, the chosen-word multiset, public-flip counts within all 16 difficulty bins, and 50% signal/control agreement. Reproduction always uses the original, checksum-verified `data/exact.jsonl`.

The binary-program constructor remains in `atd/controls.py`. A fresh solve can satisfy all matching constraints yet return a different label assignment; it is reference construction code, not a replacement for the released arm when reproducing this experiment.

## Other control implementations

`atd/controls.py` retains the teacher-label shuffle, public-label, and random-marginal constructors as reference code. Stochastic constructors require a caller-supplied seed. Their experiment configurations, historical seeds, training data, and evaluation results are not included.

## Tests

```bash
python -m unittest discover -s tests -v
```

## Citation

```bibtex
@article{zhang2026posttraining,
  title   = {Post-Training Leaves Behavioral Shadows on Unrelated Decisions},
  author  = {Zhang, Ziyang and Jing, Yubin and Zeng, Yuanhao and Li, Yuyao and Wang, Haofan and Gong, Yichen},
  journal = {arXiv preprint arXiv:2609.29233},
  year    = {2026}
}
```
