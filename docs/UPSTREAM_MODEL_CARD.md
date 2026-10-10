---
license: apache-2.0
language:
  - en
library_name: transformers
pipeline_tag: text-generation
tags:
  - qvac
  - genesis-iii
  - qwen3
  - pretraining
  - research-artifact
  - ablation
  - colm-2026
---

# QVAC Genesis III Qwen3-1.7B — Combined

This repository contains the combined Failure Analysis and Option-Level
(FA+OL) checkpoint associated with the COLM 2026 paper
[QVAC Genesis III: A Large-Scale, High-Quality Open Synthetic STEM Corpus for Efficient Language Model Pre-Training](https://arxiv.org/abs/2609.19513).

> [!WARNING]
> This is an early base-model checkpoint trained only for next-token
> prediction. It has not undergone instruction or chat post-training and is
> not intended for assistant or production use. It is provided for research,
> further pretraining, and subsequent post-training.

| Resource | Link |
| -------- | ---- |
| 📄 Paper (COLM 2026) | [arXiv:2609.19513](https://arxiv.org/abs/2609.19513) |
| 📝 Blog | [huggingface.co/blog/qvac/genesis-iii](https://huggingface.co/blog/qvac/genesis-iii/) |
| 🗃️ Dataset | [huggingface.co/datasets/qvac/GenesisIII](https://huggingface.co/datasets/qvac/GenesisIII) |
| 💻 Code | [github.com/tether-ai-research/qvac-genesis-III](https://github.com/tether-ai-research/qvac-genesis-III) |
| 🤗 Collection | [huggingface.co/collections/qvac/genesis-iii](https://huggingface.co/collections/qvac/genesis-iii) |

## 🔍 What this artifact is

This model was created for a controlled data-ablation experiment. It uses the
Qwen3-1.7B architecture, initialized from random weights, and was pretrained
for one epoch on the complete 191.43B-token QVAC Genesis III corpus.

The combined corpus contains:

- Failure Analysis documents, which turn incorrect or ambiguous student
  answers into corrective educational explanations.
- Option-Level documents, which justify the correct answer and explicitly
  explain why every distractor is incorrect.

The checkpoint is released so researchers can reproduce and inspect the
paper's experiments without repeating pretraining.

## 🚫 What this artifact is not

- It is not the pretrained `Qwen/Qwen3-1.7B-Base` model or a continuation of
  those weights. Only the architecture and tokenizer family are shared.
- It is not instruction-tuned, preference-tuned, aligned, or safety-tuned.
- It is not designed as a chatbot, assistant, tutoring product, or production
  model.
- It should not be treated as a source of reliable factual, medical,
  scientific, or safety-critical advice.

## 🏋️ Training

| Setting | Value |
| ------- | ----- |
| Architecture | Qwen3-1.7B, random initialization |
| Training data | complete QVAC Genesis III FA+OL corpus |
| Training tokens | 191.43B |
| Training duration | one epoch, 22,819 optimization steps |
| Context length | 4,096 tokens |
| Precision | BF16 |
| Optimizer | AdamW |
| Learning-rate schedule | cosine decay from 2e-4 to 2e-5 with 10% warmup |
| Global batch | 2,048 sequences, approximately 8.4M tokens per step |
| Infrastructure | Megatron-Core and Megatron-Bridge on 64 NVIDIA H100 80GB GPUs |

The experiment used packed sequences with attention reset at document
boundaries. Hyperparameters were held fixed across the paper's primary
Qwen3-1.7B data ablations.

## 📊 Paper-reported evaluation

The paper evaluates free-form generations using an LLM as an answer parser.
The parser extracts the model's final multiple-choice option or abstains;
correctness is then computed against the benchmark gold label. Valid Answer
Rate (VAR) measures whether one unambiguous option can be extracted.

Reported percentages for this checkpoint:

| Benchmark | Metric | Score (%) |
| --------- | ------ | --------: |
| ARC-Easy | Accuracy | 51.85 |
| ARC-Challenge | Accuracy | 42.71 |
| GPQA Diamond | Accuracy | 22.72 |
| MMLU STEM | Accuracy | 30.19 |
| MMLU STEM | Valid Answer Rate | 92.06 |

These are point estimates under the paper's evaluation protocol and should
not be interpreted as broad measures of general-purpose model quality.

## 🚀 Basic loading

This is a base causal language model. Plain text completion is more
appropriate than chat-style prompting.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "qvac/qvac-genesis-iii-qwen3-1.7b-combined"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
)
```

The repository does not ship a chat template, since this checkpoint was not
trained for chat or instruction following. If you post-train the model, add
a chat template (for example, the Qwen3 one) as part of that process.

## 🎯 Intended use

Appropriate uses include reproducing the paper, studying synthetic
pretraining data, comparing controlled corpus ablations, and conducting
further research with suitable evaluation and safeguards.

Use in user-facing or high-stakes systems is out of scope.

## ⚠️ Limitations

This checkpoint may produce incorrect, incoherent, repetitive, biased, or
unsafe text. Evaluation is limited to the STEM-aligned multiple-choice
benchmarks described in the paper. It does not establish instruction
following, conversational ability, human learning effectiveness, factual
reliability, or safety.

## 🧪 Related checkpoints

| Checkpoint | Role | Hugging Face |
| ---------- | ---- | ------------ |
| 🔍 Failure Analysis | FA data ablation | [qvac-genesis-iii-qwen3-1.7b-fa](https://huggingface.co/qvac/qvac-genesis-iii-qwen3-1.7b-fa) |
| 🧠 Option-Level | OL data ablation | [qvac-genesis-iii-qwen3-1.7b-ol](https://huggingface.co/qvac/qvac-genesis-iii-qwen3-1.7b-ol) |
| 🔗 Combined FA+OL | Full Genesis III mix | [qvac-genesis-iii-qwen3-1.7b-combined](https://huggingface.co/qvac/qvac-genesis-iii-qwen3-1.7b-combined) 👈 **this repository** |
## ⚖️ License and citation

Licensing Information: This model which was trained as described in the
blog post (https://huggingface.co/blog/qvac/genesis-iii/) is licensed by
Tether Data, S.A. de C.V. under the Apache 2.0 license. As described in the
blog post, this model is a version of the Qwen3-1.7B
(https://huggingface.co/Qwen/Qwen3-1.7B) which is licensed under the Apache
2.0 license.

As described in the blog post, the Qwen3-1.7B was initialized from random
weights and trained using the combined Option-Level (OL) and Failure
Analysis (FA) tokens from the Genesis III dataset
(https://huggingface.co/datasets/qvac/GenesisIII) which is licensed under
the CC-BY-NC 4.0 (Creative-Commons-Attribution-Non Commercial 4.0).

Please cite the QVAC Genesis III paper when using this artifact:

```bibtex
@misc{vitabile2026qvacgenesisiii,
  title         = {QVAC Genesis III: A Large-Scale, High-Quality Open Synthetic STEM Corpus for Efficient Language Model Pre-Training},
  author        = {Davide Vitabile and Nikhil Ranjan and Akshay Nambiar and Kamal Kumar Gupta and Amril Nazir},
  year          = {2026},
  eprint        = {2609.19513},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  institution   = {Tether Data, S.A. de C.V. d.b.a. Tether AI Research},
  note          = {Accepted at the Conference on Language Modeling (COLM) 2026},
  url           = {https://arxiv.org/abs/2609.19513}
}
```
