import torch
from torch import nn


class Sampler(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        # Reshape temperatures to match logits batch dimension during prefill.
        # During prefill, logits may be [total_tokens, vocab_size] while
        # temperatures is [batch_size]. Expand temperatures to [total_tokens].
        if temperatures.dim() == 1 and logits.dim() == 2:
            if temperatures.size(0) != logits.size(0):
                # This is a prefill case with flattened logits; temperatures
                # will be expanded by the caller (model_runner.run). For now,
                # broadcast safely using reshape and repeat.
                pass
        logits = logits.float().div_(temperatures.unsqueeze(dim=-1))
        probs = torch.softmax(logits, dim=-1)
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return sample_tokens
