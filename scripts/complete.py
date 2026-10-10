#!/usr/bin/env python3
"""Plain-text completion for the unchanged Genesis III Combined checkpoint.

Copyright 2026 Yifan Ouyang. Licensed under Apache-2.0.
"""
import argparse

DEFAULT_MODEL = "yifanouyang/Meridian-Text-1.7B"
TRAINING_CONTEXT = 4096


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hub model ID or local directory")
    parser.add_argument("--prompt", required=True, help="Plain text to continue; not a chat message")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    args = parser.parse_args()
    if not 1 <= args.max_new_tokens < TRAINING_CONTEXT:
        parser.error("--max-new-tokens must be between 1 and 4095")
    if not args.prompt.strip():
        parser.error("--prompt must contain text")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    encoded = tokenizer(args.prompt, return_tensors="pt")
    prompt_tokens = encoded["input_ids"].shape[1]
    if prompt_tokens + args.max_new_tokens > TRAINING_CONTEXT:
        parser.error("prompt plus generation exceeds the 4096-token training context")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16 if args.device == "cuda" else torch.float32,
        trust_remote_code=False,
    ).to(args.device)
    model.eval()
    encoded = {name: value.to(args.device) for name, value in encoded.items()}
    with torch.inference_mode():
        result = model.generate(**encoded, max_new_tokens=args.max_new_tokens, do_sample=False)
    print(tokenizer.decode(result[0][prompt_tokens:], skip_special_tokens=True))


if __name__ == "__main__":
    main()
