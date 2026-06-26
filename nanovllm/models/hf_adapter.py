import torch
from torch import nn
from transformers import AutoTokenizer

from nanovllm.models.base import BaseModel
from nanovllm.models.registry import ModelRegistry


@ModelRegistry.register("hf_adapter", architectures=["GPT2LMHeadModel", "GPT2Model", "DistilGPT2Model", "AutoModelForCausalLM"])
class HFAdapter(BaseModel):
    """
    Lightweight HuggingFace-backed adapter used for CPU smoke tests.

    This does NOT attempt to reproduce the performance or exact weights of
    the upstream model. Instead it provides a small embedding + linear head
    to exercise the engine's data paths and streaming logic for local testing.
    """

    def __init__(self, config, quant_config=None):
        super().__init__(config, quant_config=quant_config)
        vocab_size = getattr(config, "vocab_size", 50257)
        hidden = getattr(config, "hidden_size", 256)
        self.embed = nn.Embedding(vocab_size, hidden)
        self.head = nn.Linear(hidden, vocab_size, bias=False)

    def load_from_pretrained(self, path: str):
        # Load tokenizer to ensure vocab size matches local files where possible
        try:
            tok = AutoTokenizer.from_pretrained(path, use_fast=True, local_files_only=True)
            vs = tok.vocab_size
            if vs != self.embed.num_embeddings:
                # rebuild embedding/head to match tokenizer
                hidden = self.embed.embedding_dim
                self.embed = nn.Embedding(vs, hidden)
                self.head = nn.Linear(hidden, vs, bias=False)
        except Exception:
            # Tokenizer may not be present; ignore and use defaults
            pass

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        # Accept both 1D (flattened) and 2D (batch, seq) input_ids.
        if input_ids.dim() == 1:
            # treat as a batch of single-token inputs
            emb = self.embed(input_ids)
            return emb
        elif input_ids.dim() == 2:
            # take last token embedding per sequence
            emb = self.embed(input_ids)
            return emb[:, -1, :]
        else:
            # fallback: flatten and embed
            flat = input_ids.view(-1)
            return self.embed(flat)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.head(hidden_states)
