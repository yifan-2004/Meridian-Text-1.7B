# Meridian Text 1.7B

A local-first research toolkit for English STEM text completion: downloadable weights, resumable JSONL jobs, and the original evaluation code.

[Weights](https://huggingface.co/yifanouyang/Meridian-Text-1.7B) · [Local setup](docs/LOCAL_USE.md) · [Model & results](docs/MODEL.md) · [Source & license](docs/PROVENANCE.md)

## Quick start

Python 3.10+. First use downloads about 3.46 GB. CPU uses FP32 and needs substantially more RAM than the checkpoint size.

```bash
git clone https://github.com/yifan-2004/Meridian-Text-1.7B.git
cd Meridian-Text-1.7B
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts/complete.py --device cpu --prompt "Photosynthesis is the process by which"
```

Use `--device cuda` on NVIDIA GPUs, or `--device mps` on supported Apple Silicon Macs. [Hardware, offline use, troubleshooting](docs/LOCAL_USE.md).

## What this distribution adds

- **Resumable completion jobs:** JSONL in/out, duplicate-ID checks, input/settings validation, and a flushed result after each item.
- **Local execution controls:** CPU/CUDA/MPS selection, precision checks, offline loading, and the 4,096-token training-context guard.
- **Evaluation fixes:** empty-answer scoring and reasoning-marker handling, covered by offline regression tests.

```bash
python scripts/batch_complete.py --input examples/batch.jsonl --output results/completions.jsonl
# Continue an interrupted run with the same input and settings:
python scripts/batch_complete.py --input examples/batch.jsonl --output results/completions.jsonl --resume
```

Batch jobs process one prompt at a time to limit memory. This is text continuation, not a chat interface. `--help` never loads weights.

## Model and training

The checkpoint is **QVAC Genesis III Combined**, trained by **Tether AI Research**; the mirrored weights are unchanged. Yifan Ouyang maintains these launchers, tests, and documentation. It is a 1.7B Qwen3 base model, without chat or instruction tuning. Further training is described briefly in [training notes](docs/TRAINING.md); this release performs no training.

The [original paper](https://arxiv.org/abs/2609.19513) reports ARC-Easy **51.85%**, ARC-Challenge **42.71%**, GPQA Diamond **22.72%**, and MMLU STEM **30.19%**. These are upstream results, not measurements of the modified evaluation code. Full protocol and caveats: [model reference](docs/MODEL.md).

## Verification and license

```bash
python -m unittest discover -s tests -v
```

Unit tests use fake completions and pure postprocessing; they do not demonstrate model quality. Full checkpoint inference and paper evaluation have not been reproduced here.

Code and weights: Apache-2.0. Keep [LICENSE](LICENSE), [model license](docs/MODEL_LICENSE), and [attributions](docs/MODEL_ATTRIBUTIONS.md). The upstream Genesis III dataset has a separate CC-BY-NC 4.0 license and is not redistributed. [Weight hashes](docs/weights-manifest.json).
