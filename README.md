# Meridian Text 1.7B

![Model architecture and training workflow](assets/model-overview.svg)

An English STEM text-completion toolkit with a **1.7B Qwen3 base model**, local inference, resumable JSONL jobs and domain-adaptation training. Text continuation; no chat or instruction tuning.

[Weights](https://huggingface.co/yifanouyang/Meridian-Text-1.7B) · [Local / offline setup](docs/LOCAL_USE.md) · [Training](docs/TRAINING.md) · [Model details](docs/MODEL.md)

## Quick start

Python 3.10+. First use downloads about 3.44 GB; CPU FP32 needs substantially more RAM than the checkpoint size.

```bash
git clone https://github.com/yifan-2004/Meridian-Text-1.7B.git
cd Meridian-Text-1.7B
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/complete.py --device cpu --prompt "Photosynthesis is the process by which"
```

Use `--device cuda` for NVIDIA GPUs or `--device mps` for supported Apple Silicon Macs. Hardware support and offline loading are covered in the [setup guide](docs/LOCAL_USE.md).

## Resume a batch

```bash
python scripts/batch_complete.py --input examples/batch.jsonl --output results/completions.jsonl
python scripts/batch_complete.py --input examples/batch.jsonl --output results/completions.jsonl --resume
```

Each result is flushed to disk. Duplicate IDs and mismatched resume settings are rejected; jobs process one prompt at a time.

## Train on your text

```bash
python scripts/train.py --device cuda --train examples/training/train.jsonl --validation examples/training/validation.jsonl --output checkpoints/meridian-domain --max-length 512 --batch-size 1 --max-steps 20
```

Full-parameter FP32 training requires considerably more memory than inference. See the [recipe](docs/TRAINING.md) for document splits, loss masking and checkpoint loading. Training was smoke-tested on a tiny local Qwen3; the released 1.7B weights are unchanged and have not been trained or benchmarked here.

Offline checks cover batch handling, training inputs and answer postprocessing:

```bash
python -m unittest discover -s tests -v
```

## License

[Apache-2.0](LICENSE) · [Source](https://github.com/tether-ai-research/qvac-genesis-III) · [Model reference](https://huggingface.co/qvac/qvac-genesis-iii-qwen3-1.7b-combined)
