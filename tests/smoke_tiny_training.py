"""Actual offline training/save/reload on random tiny Qwen3; not Meridian weights."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

ROOT = Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    weights, output = root / 'weights', root / 'trained'
    tokenizer = Tokenizer(WordLevel({'[UNK]': 0, '[PAD]': 1, '[EOS]': 2, 'science': 3, 'is': 4}, unk_token='[UNK]'))
    tokenizer.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token='[UNK]', pad_token='[PAD]', eos_token='[EOS]').save_pretrained(weights)
    Qwen3ForCausalLM(Qwen3Config(vocab_size=5, hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=4096, pad_token_id=1, eos_token_id=2)).save_pretrained(weights)
    subprocess.run([sys.executable, str(ROOT/'scripts/train.py'), '--model', str(weights), '--offline', '--device', 'cpu', '--train', str(ROOT/'examples/training/train.jsonl'), '--validation', str(ROOT/'examples/training/validation.jsonl'), '--output', str(output), '--max-steps', '1', '--max-length', '16'], check=True)
    metrics = json.loads((output/'metrics.json').read_text())
    assert metrics['steps'] == 1 and metrics['validation_prediction_tokens'] > 0
    assert (output/'model.safetensors').is_file()
    trained = Qwen3ForCausalLM.from_pretrained(output, local_files_only=True)
    original = Qwen3ForCausalLM.from_pretrained(weights, local_files_only=True)
    import torch
    assert any(not torch.equal(a, b) for a, b in zip(trained.parameters(), original.parameters()))
    PreTrainedTokenizerFast.from_pretrained(output, local_files_only=True)
print('Passed actual offline CPU training, validation, changed parameters, save and reload on random tiny Qwen3; not a 1.7B training test.')
