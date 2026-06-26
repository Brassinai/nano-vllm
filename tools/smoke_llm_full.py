import sys
import types
from pathlib import Path

# Ensure local package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Insert a fake 'transformers' module with minimal AutoConfig and AutoTokenizer
fake_transformers = types.SimpleNamespace()

class FakeAutoConfig:
    @staticmethod
    def from_pretrained(path):
        # Return an object with required attributes
        return types.SimpleNamespace(
            max_position_embeddings=128,
            num_key_value_heads=4,
            num_hidden_layers=2,
            hidden_size=32,
            dtype=None,
        )

class FakeTokenizer:
    def __init__(self):
        self.eos_token_id = 50256
    @staticmethod
    def from_pretrained(path, use_fast=True):
        return FakeTokenizer()
    def encode(self, text):
        # simple deterministic mapping: map chars to small ints
        return [ord(c) % 256 for c in text]
    def decode(self, ids):
        # join token ids into a string for visibility
        return ''.join(f"<{i}>" for i in ids)

fake_transformers.AutoConfig = FakeAutoConfig
fake_transformers.AutoTokenizer = FakeTokenizer
sys.modules['transformers'] = fake_transformers

# Insert a fake model_runner module
fake_mod = types.ModuleType('nanovllm.engine.model_runner')
class FakeModelRunner:
    def __init__(self, config, rank, event):
        self.config = config
        self.rank = rank
        self.event = event
    def call(self, method_name, *args, **kwargs):
        if method_name == 'run':
            seqs, is_prefill = args
            # return one token id per seq (simple counter)
            return [1 for _ in seqs]
        elif method_name == 'exit':
            return None
        else:
            return None
    def exit(self):
        return None

fake_mod.ModelRunner = FakeModelRunner
sys.modules['nanovllm.engine.model_runner'] = fake_mod

# Now import and instantiate LLMEngine
try:
    from nanovllm.engine.llm_engine import LLMEngine
    from nanovllm.sampling_params import SamplingParams
    model_dir = Path(__file__).resolve().parent.parent / 'test_model'
    engine = LLMEngine(str(model_dir), tensor_parallel_size=1, enforce_eager=True)
    print('Engine instantiated')
    sp = SamplingParams()
    gen = engine.stream_generate('Hello', sp)
    print('Generator created; streaming a few chunks:')
    for i, chunk in enumerate(gen):
        print('chunk', i, repr(chunk))
        if i >= 4:
            break
except Exception as e:
    import traceback
    print('Error during smoke:', type(e).__name__, e)
    traceback.print_exc()
