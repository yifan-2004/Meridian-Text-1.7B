# Run locally

## 1. Install

Python 3.10+; run from the repository root. Use a virtual environment so system packages stay separate:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Windows PowerShell activation: `.venv\Scripts\Activate.ps1`. If activation is restricted, call `.venv\Scripts\python.exe` directly instead of `python`.

The default PyTorch wheel may not match an NVIDIA driver. Install the CUDA wheel appropriate for your platform using the [official PyTorch installer](https://pytorch.org/get-started/locally/), then install `requirements.txt`. The launcher checks CUDA availability before loading weights.

## 2. Choose hardware

```bash
# Broadly compatible; slow and memory intensive
python scripts/complete.py --device cpu --prompt "A prime number is"
# BF16 when the GPU supports it, otherwise FP32
python scripts/complete.py --device cuda --prompt "A prime number is"
# Explicit FP16 for compatible GPUs with limited VRAM
python scripts/complete.py --device cuda --dtype float16 --prompt "A prime number is"
# Apple Silicon: conservative FP32 default; FP16 is optional
python scripts/complete.py --device mps --prompt "A prime number is"
```

The checkpoint is about 3.44 GB in BF16; FP32 parameters alone need roughly 6.9 GB, with extra memory for loading, activations, and the KV cache. These are size estimates, not measured peak RAM/VRAM. No quantization or CPU offloading is implemented. If memory runs out, shorten `--max-new-tokens`, use a supported lower precision, or choose a machine with more memory. CUDA and MPS hardware runs have not been validated in this release.

The model is English-focused and unaligned. Give it a sentence to continue rather than a chat question. Output can be repetitive or factually wrong. Decoding is greedy; identical output across devices/library versions is not guaranteed.

## 3. Download once, run offline

```bash
hf download yifanouyang/Meridian-Text-1.7B --local-dir weights
python scripts/complete.py --model ./weights --offline --prompt "Conservation of energy means"
```

`--offline` passes `local_files_only=True` to both tokenizer and model loading. Missing files produce an error instead of a download. It can also use an already complete Hub cache. Neither launcher enables remote model Python code. For reproducibility, pass `--revision <Hub-commit>` when using a Hub ID; `main` can change. A local directory's actual content is your responsibility.

## 4. Resume JSONL jobs

Input is one JSON object per line:

```json
{"id":"biology-01","prompt":"Photosynthesis is the process by which"}
```

```bash
python scripts/batch_complete.py --input examples/batch.jsonl --output results/completions.jsonl --max-new-tokens 64
python scripts/batch_complete.py --input examples/batch.jsonl --output results/completions.jsonl --max-new-tokens 64 --resume
```

Output contains `id`, `prompt`, `completion`, and `run` settings. The launcher validates the entire input and existing output before loading weights, refuses duplicate IDs, and rejects changed prompts/model settings for completed items. It skips completed IDs and loads the model once for remaining records. Each completed record is flushed to disk; an interrupted generation is retried. This is sequential inference, not tensor-batched generation.

Use one process per output file. Do not edit output while a run is active. A crash during a write can leave a truncated final JSON line; the launcher fails explicitly rather than silently discarding it. Back up the output and remove only the incomplete final line before resuming. Flushing protects against process interruption, not all power-loss scenarios. Resume metadata records the requested revision and dtype; it does not fingerprint local weight files or pin an unversioned Hub `main` branch.

## Troubleshooting

- `CUDA/MPS is unavailable`: use `--device cpu` or install a compatible PyTorch build.
- Context limit: prompt tokens plus generated tokens must fit the 4,096-token training context. The larger positional capacity in config is not a validated context length.
- Download failure: download once with `hf download`; network/proxy settings follow Hugging Face tooling.
- `output exists`: choose another output or use `--resume` with unchanged input and generation arguments.
- Full evaluation: the inherited protocol loads a separate 32B judge and defaults to large GPU resources. Read `research/evaluation/README.md` before using it.

## Developer checks

```bash
python -m unittest discover -s tests -v
python tests/smoke_tiny_runtime.py
```

The optional integration smoke creates a temporary random tiny Qwen3 model and checks both offline CPU CLIs plus resume. It downloads nothing and does not validate Meridian weights or model quality. Tested with Transformers 4.57.6 and PyTorch 2.13.0; CUDA/MPS remain untested.
