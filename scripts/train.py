"""Document-safe causal-LM domain adaptation. Copyright 2026 Yifan Ouyang; Apache-2.0."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random

from runtime import DEFAULT_MODEL, TRAINING_CONTEXT


def read_split(path):
    rows, ids, texts = [], set(), set()
    for number, line in enumerate(Path(path).read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f'{path}:{number}: expected a JSON object')
        doc_id, text = row.get('document_id'), row.get('text')
        if not isinstance(doc_id, str) or not doc_id.strip() or not isinstance(text, str) or not text.strip():
            raise ValueError(f'{path}:{number}: nonempty document_id and text are required')
        digest = hashlib.sha256(' '.join(text.split()).encode()).hexdigest()
        if doc_id in ids or digest in texts:
            raise ValueError(f'{path}:{number}: duplicate document ID or normalized text')
        ids.add(doc_id)
        texts.add(digest)
        rows.append(text)
    if not rows:
        raise ValueError(f'{path}: split is empty')
    return rows, ids, texts


def load_splits(train_path, validation_path):
    train, train_ids, train_texts = read_split(train_path)
    validation, validation_ids, validation_texts = read_split(validation_path)
    if train_ids & validation_ids or train_texts & validation_texts:
        raise ValueError('Train/validation overlap: shared document ID or normalized text')
    return train, validation


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', default=DEFAULT_MODEL)
    p.add_argument('--revision', default='main', help='Pin a Hub commit for reproducibility')
    p.add_argument('--train', required=True)
    p.add_argument('--validation', required=True)
    p.add_argument('--output', required=True, help='New or empty checkpoint directory')
    p.add_argument('--offline', action='store_true')
    p.add_argument('--device', choices=('cpu', 'cuda', 'mps'), default='cpu')
    p.add_argument('--max-length', type=int, default=512)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--max-steps', type=int, default=20)
    p.add_argument('--learning-rate', type=float, default=2e-5)
    p.add_argument('--weight-decay', type=float, default=0.01)
    p.add_argument('--seed', type=int, default=42)
    return p


def main():
    args = parser().parse_args()
    if not 2 <= args.max_length <= TRAINING_CONTEXT or min(args.batch_size, args.max_steps) < 1:
        raise ValueError('Use length 2..4096, batch size >=1, and steps >=1')
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0 or not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        raise ValueError('Learning rate must be finite and positive; weight decay finite and nonnegative')
    train_texts, validation_texts = load_splits(args.train, args.validation)
    output = Path(args.output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError('Output must be new or empty; existing checkpoints are never overwritten')
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA unavailable')
    if args.device == 'mps' and not torch.backends.mps.is_available():
        raise ValueError('MPS unavailable')
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    options = dict(revision=args.revision, local_files_only=args.offline, trust_remote_code=False)
    tokenizer = AutoTokenizer.from_pretrained(args.model, **options)
    if tokenizer.eos_token_id is None:
        raise ValueError('Tokenizer requires an EOS token')
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32, **options).to(args.device)
    original_use_cache = model.config.use_cache
    model.config.use_cache = False
    if args.max_length > getattr(model.config, 'max_position_embeddings', TRAINING_CONTEXT):
        raise ValueError('max-length exceeds model context')
    def encode(texts):
        encoded = []
        for text in texts:
            ids = tokenizer.encode(text, add_special_tokens=False, truncation=True, max_length=args.max_length - 1)
            if not ids:
                raise ValueError('Every training/validation text must encode to at least one token')
            encoded.append(ids + [tokenizer.eos_token_id])
        return encoded
    training, validation = encode(train_texts), encode(validation_texts)
    def batch(records):
        packed = tokenizer.pad({'input_ids': records}, padding=True, return_tensors='pt')
        labels = packed['input_ids'].clone()
        labels[packed['attention_mask'] == 0] = -100
        return {**{k: v.to(args.device) for k, v in packed.items()}, 'labels': labels.to(args.device)}
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    output.mkdir(parents=True, exist_ok=True)
    config = vars(args) | {'display_name': 'Meridian Text 1.7B', 'objective': 'causal next-token prediction', 'precision': 'float32', 'optimizer': 'AdamW', 'schedule': 'constant', 'gradient_clip_norm': 1.0, 'train_documents': len(training), 'validation_documents': len(validation), 'torch_version': torch.__version__, 'resolved_model_commit': getattr(model.config, '_commit_hash', None), 'train_sha256': hashlib.sha256(Path(args.train).read_bytes()).hexdigest(), 'validation_sha256': hashlib.sha256(Path(args.validation).read_bytes()).hexdigest()}
    (output/'training_config.json').write_text(json.dumps(config, indent=2) + '\n')
    rng = random.Random(args.seed)
    order = []
    model.train()
    with (output/'training_log.jsonl').open('w') as log:
        for step in range(1, args.max_steps + 1):
            chosen = []
            for _ in range(args.batch_size):
                if not order:
                    order = list(range(len(training)))
                    rng.shuffle(order)
                chosen.append(training[order.pop()])
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch(chosen)).loss
            if not torch.isfinite(loss):
                raise ValueError(f'Nonfinite training loss at step {step}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            row = {'step': step, 'train_loss': loss.item()}
            log.write(json.dumps(row) + '\n')
            log.flush()
            print(json.dumps(row), flush=True)
    model.eval()
    weighted_loss, tokens = 0.0, 0
    with torch.inference_mode():
        for start in range(0, len(validation), args.batch_size):
            item = batch(validation[start:start + args.batch_size])
            loss = model(**item).loss.item()
            if not math.isfinite(loss):
                raise ValueError('Nonfinite validation loss')
            count = (item['labels'][:, 1:] != -100).sum().item()
            weighted_loss += loss * count
            tokens += count
    metrics = {'validation_loss': weighted_loss / tokens, 'validation_prediction_tokens': tokens, 'steps': args.max_steps}
    (output/'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    model.config.use_cache = original_use_cache
    model.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)
    print(json.dumps(metrics), flush=True)


if __name__ == '__main__':
    main()
