"""Local completion runtime; imports model libraries only when loading weights.

Copyright 2026 Yifan Ouyang. Licensed under Apache-2.0.
"""
DEFAULT_MODEL = 'yifanouyang/Meridian-Text-1.7B'
TRAINING_CONTEXT = 4096


def add_runtime_arguments(parser):
    parser.add_argument('--model', default=DEFAULT_MODEL, help='Hub ID or local model directory')
    parser.add_argument('--revision', default='main', help='Hub revision; pin a commit for reproducibility')
    parser.add_argument('--device', choices=('cpu', 'cuda', 'mps'), default='cpu')
    parser.add_argument('--dtype', choices=('auto', 'float32', 'float16', 'bfloat16'), default='auto')
    parser.add_argument('--offline', action='store_true', help='Use local files/cache only; never download')
    parser.add_argument('--max-new-tokens', type=int, default=128)


def validate_settings(args):
    if not 1 <= args.max_new_tokens < TRAINING_CONTEXT:
        raise ValueError('--max-new-tokens must be between 1 and 4095')
    if args.device == 'cpu' and args.dtype == 'float16':
        raise ValueError('CPU float16 is unsupported by this launcher; use float32')
    if args.device == 'mps' and args.dtype == 'bfloat16':
        raise ValueError('MPS bfloat16 depends on OS/PyTorch support; use float32 or float16')


def load_completion(args):
    validate_settings(args)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable; install a compatible PyTorch build or use --device cpu')
    if args.device == 'mps' and not torch.backends.mps.is_available():
        raise ValueError('MPS is unavailable; use an Apple Silicon Mac with supported PyTorch or --device cpu')
    if args.dtype == 'auto':
        dtype = torch.bfloat16 if args.device == 'cuda' and torch.cuda.is_bf16_supported() else torch.float32
    else:
        dtype = getattr(torch, args.dtype)
    if args.device == 'cuda' and dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise ValueError('This CUDA device does not support bfloat16; use --dtype float16 or float32')
    options = dict(trust_remote_code=False, local_files_only=args.offline, revision=args.revision)
    tokenizer = AutoTokenizer.from_pretrained(args.model, **options)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, **options).to(args.device)
    model.eval()

    def complete(prompt):
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError('prompt must contain text')
        encoded = tokenizer(prompt, return_tensors='pt', return_token_type_ids=False)
        count = encoded['input_ids'].shape[1]
        if count + args.max_new_tokens > TRAINING_CONTEXT:
            raise ValueError('prompt plus generation exceeds the 4096-token training context')
        encoded = {key: value.to(args.device) for key, value in encoded.items()}
        with torch.inference_mode():
            result = model.generate(**encoded, max_new_tokens=args.max_new_tokens, do_sample=False)
        return tokenizer.decode(result[0][count:], skip_special_tokens=True)
    return complete
