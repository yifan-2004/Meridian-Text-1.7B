"""Optional integration smoke test using a temporary RANDOM tiny Qwen3.

Requires requirements.txt; downloads nothing, does not test Meridian weights.
"""
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
    weights = root / 'weights'
    tokenizer = Tokenizer(WordLevel({'[UNK]': 0, '[PAD]': 1, '[EOS]': 2, 'science': 3, 'is': 4}, unk_token='[UNK]'))
    tokenizer.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token='[UNK]', pad_token='[PAD]', eos_token='[EOS]').save_pretrained(weights)
    Qwen3ForCausalLM(Qwen3Config(vocab_size=5, hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=4096, pad_token_id=1, eos_token_id=2)).save_pretrained(weights)
    arguments = ['--model', str(weights), '--offline', '--device', 'cpu', '--max-new-tokens', '2']
    subprocess.run([sys.executable, str(ROOT/'scripts/complete.py'), *arguments, '--prompt', 'science is'], check=True)
    source = root/'input.jsonl'
    output = root/'output.jsonl'
    source.write_text('{"id":"smoke","prompt":"science is"}\n')
    batch = [sys.executable, str(ROOT/'scripts/batch_complete.py'), *arguments, '--input', str(source), '--output', str(output)]
    subprocess.run(batch, check=True)
    resumed = subprocess.run([*batch, '--resume'], check=True, text=True, capture_output=True)
    assert 'Completed 0 new records' in resumed.stdout
    assert len(output.read_text().splitlines()) == 1
    assert isinstance(json.loads(output.read_text())['completion'], str)
print('Passed offline CPU CLI integration with a temporary random tiny Qwen3; not a Meridian inference or quality test.')
