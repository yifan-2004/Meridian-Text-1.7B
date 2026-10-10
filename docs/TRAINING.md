# Training notes

This release redistributes unchanged Genesis III Combined weights; it does not claim new pretraining, fine-tuning, or evaluation improvements.

For a downstream fine-tune:

1. Create a train/validation split by source document before formatting examples; audit for leakage.
2. Use a Qwen3-compatible causal-LM trainer and this repository's Hugging Face checkpoint. This repository does not include a validated trainer or ready-to-run training recipe.
3. Keep examples within the 4,096-token training context; start with a short hardware smoke run, then evaluate a held-out set against the unchanged checkpoint.
4. Publish your configuration, data rights, training logs, adapter/base dependency, and measured results separately. Preserve original model attribution.

The upstream dataset uses CC-BY-NC 4.0, separately from the Apache-2.0 model weights. See the [original model card](UPSTREAM_MODEL_CARD.md) and [paper](https://arxiv.org/abs/2609.19513) for the authors' training description. No training was run for this toolkit update.
