"""GPTQ export utilities for converting dense HF checkpoints into nano-vLLM format."""

from __future__ import annotations

import json
import math
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_CALIBRATION_TEXTS = (
    "Large language models use cached keys and values to avoid recomputing attention for previous tokens.",
    "Quantization trades numerical precision for lower memory usage and faster inference on modern accelerators.",
    "The goal of this benchmark is to compare different KV cache formats under the same prompt workload.",
    "Colab notebooks benefit from small calibration sets because they reduce quantization time while preserving usability.",
    "Grouped weight quantization typically stores one scale and zero point per output channel and input group.",
    "Inference engines often separate model weight quantization from KV cache quantization because they affect different tensors.",
    "Attention projections and MLP projections dominate decoder-only transformer weight memory footprints.",
    "A representative calibration corpus should cover varied token patterns, punctuation, and sentence lengths.",
    "GPTQ minimizes reconstruction error using a Hessian approximation collected from calibration activations.",
    "Smaller group sizes usually improve fidelity at the cost of more metadata and a larger quantized checkpoint.",
)


def quantize_dequantize(
    x: torch.Tensor,
    scale: torch.Tensor,
    zero: torch.Tensor,
    maxq: torch.Tensor,
) -> torch.Tensor:
    q = torch.clamp(torch.round(x / scale) + zero, 0, int(maxq.item()))
    return scale * (q - zero)


def quantize_int(
    x: torch.Tensor,
    scale: torch.Tensor,
    zero: torch.Tensor,
    maxq: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    q = torch.clamp(torch.round(x / scale) + zero, 0, int(maxq.item())).to(torch.int32)
    dequant = scale * (q.to(scale.dtype) - zero)
    return q, dequant


class Quantizer(nn.Module):
    """Minimal GPTQ quantizer adapted from IST-DASLab/gptq."""

    def __init__(self, shape: int = 1):
        super().__init__()
        self.register_buffer("maxq", torch.tensor(0))
        self.register_buffer("scale", torch.zeros(shape))
        self.register_buffer("zero", torch.zeros(shape))

    def configure(
        self,
        bits: int,
        *,
        perchannel: bool = False,
        sym: bool = True,
        mse: bool = False,
        norm: float = 2.4,
        grid: int = 100,
        maxshrink: float = 0.8,
    ) -> None:
        self.maxq = torch.tensor(2**bits - 1)
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink

    def find_params(self, x: torch.Tensor, *, weight: bool = False) -> None:
        dev = x.device
        self.maxq = self.maxq.to(dev)

        shape = x.shape
        if self.perchannel:
            if weight:
                x = x.flatten(1)
            elif len(shape) == 3:
                x = x.reshape((-1, shape[-1])).t()
            elif len(shape) == 2:
                x = x.t()
        else:
            x = x.flatten().unsqueeze(0)

        tmp = torch.zeros(x.shape[0], device=dev)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)

        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            neg = xmin < 0
            if torch.any(neg):
                xmin[neg] = -xmax[neg]

        is_zero = (xmin == 0) & (xmax == 0)
        xmin[is_zero] = -1
        xmax[is_zero] = 1

        self.scale = (xmax - xmin) / self.maxq
        if self.sym:
            self.zero = torch.full_like(self.scale, (self.maxq + 1) / 2)
        else:
            self.zero = torch.round(-xmin / self.scale)

        if self.mse:
            best = torch.full([x.shape[0]], float("inf"), device=dev)
            for i in range(int(self.maxshrink * self.grid)):
                p = 1 - i / self.grid
                xmin1 = p * xmin
                xmax1 = p * xmax
                scale1 = (xmax1 - xmin1) / self.maxq
                zero1 = torch.round(-xmin1 / scale1) if not self.sym else self.zero
                q = quantize_dequantize(
                    x,
                    scale1.unsqueeze(1),
                    zero1.unsqueeze(1),
                    self.maxq,
                )
                q = (q - x).abs().pow(self.norm)
                err = torch.sum(q, 1)
                better = err < best
                if torch.any(better):
                    best[better] = err[better]
                    self.scale[better] = scale1[better]
                    self.zero[better] = zero1[better]

        if not self.perchannel:
            repeat = shape[0] if weight else shape[-1]
            self.scale = self.scale.repeat(repeat)
            self.zero = self.zero.repeat(repeat)

        if weight:
            self.scale = self.scale.reshape((-1, 1))
            self.zero = self.zero.reshape((-1, 1))
        elif len(shape) == 3:
            self.scale = self.scale.reshape((1, 1, -1))
            self.zero = self.zero.reshape((1, 1, -1))
        elif len(shape) == 2:
            self.scale = self.scale.unsqueeze(0)
            self.zero = self.zero.unsqueeze(0)

    def ready(self) -> bool:
        return torch.all(self.scale != 0)


@dataclass
class GPTQTensorPack:
    qweight: torch.Tensor
    qzeros: torch.Tensor
    scales: torch.Tensor
    g_idx: torch.Tensor


class GPTQ:
    """Layer-wise GPTQ helper adapted from IST-DASLab/gptq."""

    def __init__(self, layer: nn.Linear) -> None:
        self.layer = layer
        self.dev = layer.weight.device
        weight = layer.weight.data.clone().float()
        self.rows = weight.shape[0]
        self.columns = weight.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0
        self.quantizer = Quantizer()

    def add_batch(self, inp: torch.Tensor, _out: torch.Tensor) -> None:
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2.0 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    def fasterquant(
        self,
        *,
        blocksize: int = 128,
        percdamp: float = 0.01,
        groupsize: int = -1,
    ) -> GPTQTensorPack:
        weight = self.layer.weight.data.clone().float()
        if not self.quantizer.ready():
            self.quantizer.find_params(weight, weight=True)

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        weight[:, dead] = 0

        q_int = torch.zeros_like(weight, dtype=torch.int32)
        group_count = 1 if groupsize == -1 else math.ceil(self.columns / groupsize)
        scales = torch.zeros((group_count, self.rows), dtype=torch.float32, device=self.dev)
        zeros = torch.zeros((group_count, self.rows), dtype=torch.int32, device=self.dev)

        if groupsize == -1:
            scales[0] = self.quantizer.scale.reshape(-1).float()
            zeros[0] = torch.round(self.quantizer.zero.reshape(-1)).to(torch.int32)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = weight[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                global_idx = i1 + i
                if groupsize != -1 and global_idx % groupsize == 0:
                    slice_end = min(global_idx + groupsize, self.columns)
                    self.quantizer.find_params(
                        weight[:, global_idx:slice_end],
                        weight=True,
                    )
                    group_idx = global_idx // groupsize
                    scales[group_idx] = self.quantizer.scale.reshape(-1).float()
                    zeros[group_idx] = torch.round(
                        self.quantizer.zero.reshape(-1)
                    ).to(torch.int32)

                w = W1[:, i]
                d = Hinv1[i, i]
                qi, qf = quantize_int(
                    w.unsqueeze(1),
                    self.quantizer.scale,
                    self.quantizer.zero,
                    self.quantizer.maxq,
                )
                qi = qi.flatten()
                qf = qf.flatten()
                q_int[:, global_idx] = qi
                Q1[:, i] = qf

                err1 = (w - qf) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            weight[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        g_idx = (
            torch.zeros(self.columns, dtype=torch.int32, device=self.dev)
            if groupsize == -1
            else torch.arange(self.columns, device=self.dev, dtype=torch.int32) // groupsize
        )
        return GPTQTensorPack(
            qweight=pack_qweight(q_int, bits=int(torch.log2(self.quantizer.maxq + 1).item())),
            qzeros=pack_qzeros(zeros, bits=int(torch.log2(self.quantizer.maxq + 1).item())),
            scales=scales.contiguous().to(torch.float16).cpu(),
            g_idx=g_idx.cpu(),
        )

    def free(self) -> None:
        self.H = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def pack_qweight(q_int: torch.Tensor, *, bits: int) -> torch.Tensor:
    """Pack [out_features, in_features] integer weights to GPTQ qweight layout."""
    pack_factor = 32 // bits
    out_features, in_features = q_int.shape
    if in_features % pack_factor:
        raise ValueError(
            f"in_features={in_features} must be divisible by pack_factor={pack_factor}."
        )
    transposed = q_int.t().contiguous().cpu().to(torch.int32)
    packed = torch.zeros(
        (in_features // pack_factor, out_features),
        dtype=torch.int32,
    )
    for i in range(pack_factor):
        packed |= transposed[i::pack_factor] << (i * bits)
    return packed


def pack_qzeros(qzeros: torch.Tensor, *, bits: int) -> torch.Tensor:
    """Pack [group_count, out_features] zero-points to GPTQ qzeros layout."""
    pack_factor = 32 // bits
    group_count, out_features = qzeros.shape
    if out_features % pack_factor:
        raise ValueError(
            f"out_features={out_features} must be divisible by pack_factor={pack_factor}."
        )
    values = qzeros.contiguous().cpu().to(torch.int32)
    packed = torch.zeros(
        (group_count, out_features // pack_factor),
        dtype=torch.int32,
    )
    for i in range(pack_factor):
        packed |= values[:, i::pack_factor] << (i * bits)
    return packed


def unpack_qweight(qweight: torch.Tensor, *, bits: int, out_features: int) -> torch.Tensor:
    """Inverse of :func:`pack_qweight` for tests/debugging."""
    pack_factor = 32 // bits
    rows = []
    for i in range(pack_factor):
        rows.append(((qweight >> (i * bits)) & ((1 << bits) - 1)).to(torch.int32))
    return torch.stack(rows, dim=1).reshape(-1, out_features).t().contiguous()


def unpack_qzeros(qzeros: torch.Tensor, *, bits: int, out_features: int) -> torch.Tensor:
    """Inverse of :func:`pack_qzeros` for tests/debugging."""
    pack_factor = 32 // bits
    cols = []
    for i in range(pack_factor):
        cols.append(((qzeros >> (i * bits)) & ((1 << bits) - 1)).to(torch.int32))
    return torch.stack(cols, dim=2).reshape(qzeros.shape[0], out_features).contiguous()


def find_linear_layers(module: nn.Module) -> dict[str, nn.Linear]:
    layers: dict[str, nn.Linear] = {}
    for name, child in module.named_modules():
        if name and isinstance(child, nn.Linear):
            layers[name] = child
    return layers


def layer_quantization_groups(layer: nn.Module) -> list[list[str]]:
    full = find_linear_layers(layer)
    remaining = set(full)
    ordered_groups = (
        ("self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"),
        ("self_attn.o_proj",),
        ("mlp.up_proj", "mlp.gate_proj"),
        ("mlp.down_proj",),
    )
    groups: list[list[str]] = []
    for group in ordered_groups:
        present = [name for name in group if name in full]
        if present:
            groups.append(present)
            remaining.difference_update(present)
    if remaining:
        groups.append(sorted(remaining))
    return groups


def make_calibration_batches(
    tokenizer: Any,
    texts: list[str],
    *,
    nsamples: int,
    seqlen: int,
    seed: int,
) -> list[torch.Tensor]:
    eos = tokenizer.eos_token_id
    if eos is None:
        eos = tokenizer.pad_token_id
    if eos is None:
        eos = 0

    stream: list[int] = []
    for text in texts:
        ids = tokenizer(text, add_special_tokens=False).input_ids
        if ids:
            stream.extend(ids)
            stream.append(eos)
    if not stream:
        raise ValueError("Calibration token stream is empty.")
    if len(stream) < seqlen:
        repeats = math.ceil(seqlen / len(stream))
        stream = stream * repeats

    rng = random.Random(seed)
    max_start = max(len(stream) - seqlen, 0)
    batches: list[torch.Tensor] = []
    for _ in range(nsamples):
        start = rng.randint(0, max_start) if max_start else 0
        window = stream[start : start + seqlen]
        batches.append(torch.tensor(window, dtype=torch.long).unsqueeze(0))
    return batches


def _get_base_model(model: nn.Module) -> nn.Module:
    base_model = getattr(model, "model", None)
    if base_model is None:
        raise ValueError("Only decoder-only models with a .model attribute are supported.")
    return base_model


def _get_decoder_layers(model: nn.Module) -> list[nn.Module]:
    base_model = _get_base_model(model)
    layers = getattr(base_model, "layers", None)
    if layers is None:
        raise ValueError("Model does not expose decoder layers at model.layers.")
    return list(layers)


def _get_layer_output_hidden(output: Any) -> torch.Tensor:
    if isinstance(output, tuple):
        return output[0]
    return output


@dataclass
class GPTQExportConfig:
    bits: int = 4
    group_size: int = 128
    nsamples: int = 32
    seqlen: int = 512
    seed: int = 0
    blocksize: int = 128
    percdamp: float = 0.01
    sym: bool = True
    dtype: str = "float16"
    device: str = "cuda"


def quantize_hf_model(
    model: nn.Module,
    calibration_batches: list[torch.Tensor],
    config: GPTQExportConfig,
    *,
    device: str,
) -> dict[str, GPTQTensorPack]:
    use_cache = getattr(model.config, "use_cache", False)
    model.config.use_cache = False
    try:
        base_model = _get_base_model(model)
        layers = _get_decoder_layers(model)
        embeddings = getattr(base_model, "embed_tokens", None)
        final_norm = getattr(base_model, "norm", None)
        if embeddings is None:
            raise ValueError("Model does not expose model.embed_tokens for calibration.")

        dtype = next(iter(model.parameters())).dtype
        hidden_size = int(getattr(model.config, "hidden_size"))
        nsamples = len(calibration_batches)
        inps = torch.zeros(
            (nsamples, config.seqlen, hidden_size),
            dtype=dtype,
            device=device,
        )
        cache: dict[str, Any] = {"i": 0, "kwargs": None}

        class Catcher(nn.Module):
            def __init__(self, module: nn.Module):
                super().__init__()
                self.module = module

            def __getattr__(self, name: str):
                if name == "module":
                    return super().__getattr__(name)
                try:
                    return super().__getattr__(name)
                except AttributeError:
                    return getattr(self.module, name)

            def forward(self, inp: torch.Tensor, *args, **kwargs):
                inps[cache["i"]] = inp.squeeze(0)
                cache["i"] += 1
                cache["kwargs"] = {
                    key: value
                    for key, value in kwargs.items()
                    if value is not None and key != "use_cache"
                }
                raise ValueError("capture")

        embeddings.to(device)
        if final_norm is not None:
            final_norm.to(device)
        base_model.layers[0] = Catcher(base_model.layers[0].to(device))
        for batch in calibration_batches:
            try:
                model(batch.to(device))
            except ValueError as exc:
                if str(exc) != "capture":
                    raise
        base_model.layers[0] = base_model.layers[0].module.cpu()
        embeddings.cpu()
        if final_norm is not None:
            final_norm.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        outs = torch.zeros_like(inps)
        layer_kwargs = cache["kwargs"] or {}
        quantized: dict[str, GPTQTensorPack] = {}

        for layer_idx, _ in enumerate(layers):
            layer = base_model.layers[layer_idx].to(device)
            full = find_linear_layers(layer)
            for group_names in layer_quantization_groups(layer):
                subset = {name: full[name] for name in group_names}
                helpers: dict[str, GPTQ] = {}
                handles = []

                def add_batch(name: str):
                    def hook(_module, inp, out):
                        helpers[name].add_batch(
                            inp[0].data,
                            _get_layer_output_hidden(out).data,
                        )
                    return hook

                for name, sublayer in subset.items():
                    helper = GPTQ(sublayer)
                    helper.quantizer.configure(
                        config.bits,
                        perchannel=True,
                        sym=config.sym,
                        mse=False,
                    )
                    helpers[name] = helper
                    handles.append(sublayer.register_forward_hook(add_batch(name)))

                for sample_idx in range(nsamples):
                    outs[sample_idx] = _get_layer_output_hidden(
                        layer(inps[sample_idx].unsqueeze(0), **layer_kwargs)
                    ).squeeze(0)

                for handle in handles:
                    handle.remove()

                for name, helper in helpers.items():
                    prefix = f"model.layers.{layer_idx}.{name}"
                    quantized[prefix] = helper.fasterquant(
                        blocksize=config.blocksize,
                        percdamp=config.percdamp,
                        groupsize=config.group_size,
                    )
                    helper.free()

            for sample_idx in range(nsamples):
                outs[sample_idx] = _get_layer_output_hidden(
                    layer(inps[sample_idx].unsqueeze(0), **layer_kwargs)
                ).squeeze(0)

            base_model.layers[layer_idx] = layer.cpu()
            del layer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            inps, outs = outs, inps

        return quantized
    finally:
        model.config.use_cache = use_cache


def collect_export_state_dict(
    model: nn.Module,
    quantized: dict[str, GPTQTensorPack],
) -> dict[str, torch.Tensor]:
    export_state: dict[str, torch.Tensor] = {}
    quantized_prefixes = set(quantized)
    for name, tensor in model.state_dict().items():
        if name.endswith(".weight") and name[: -len(".weight")] in quantized_prefixes:
            continue
        export_state[name] = tensor.detach().cpu().clone()

    for prefix, packed in quantized.items():
        export_state[f"{prefix}.qweight"] = packed.qweight
        export_state[f"{prefix}.qzeros"] = packed.qzeros
        export_state[f"{prefix}.scales"] = packed.scales
        export_state[f"{prefix}.g_idx"] = packed.g_idx
    return export_state


def copy_support_files(source_dir: str, output_dir: str) -> None:
    src = Path(source_dir)
    dst = Path(output_dir)
    dst.mkdir(parents=True, exist_ok=True)
    skip_names = {
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
        "model.safetensors",
        "model.safetensors.index.json",
        "quantize_config.json",
    }
    for path in src.iterdir():
        if path.name in skip_names:
            continue
        if path.name.startswith("pytorch_model-") and path.name.endswith(".bin"):
            continue
        if path.name.startswith("model-") and path.name.endswith(".safetensors"):
            continue
        if path.name.endswith(".safetensors") or path.name.endswith(".safetensors.index.json"):
            continue
        target = dst / path.name
        if path.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(path, target)
        else:
            shutil.copy2(path, target)


def export_gptq_checkpoint(
    model_path: str,
    output_path: str,
    *,
    config: GPTQExportConfig,
    calibration_texts: list[str] | None = None,
) -> str:
    output_dir = Path(output_path).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"Requested device {config.device!r}, but CUDA is not available."
        )

    load_dtype = torch.float16 if config.dtype == "float16" else torch.float32
    if config.device == "cpu":
        load_dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=load_dtype,
        low_cpu_mem_usage=True,
    )
    model.eval()

    texts = calibration_texts or list(DEFAULT_CALIBRATION_TEXTS)
    batches = make_calibration_batches(
        tokenizer,
        texts,
        nsamples=config.nsamples,
        seqlen=config.seqlen,
        seed=config.seed,
    )
    quantized = quantize_hf_model(
        model,
        batches,
        config,
        device=config.device,
    )
    export_state = collect_export_state_dict(model, quantized)

    copy_support_files(model_path, str(output_dir))
    save_file(export_state, str(output_dir / "model.safetensors"))
    quantize_config = {
        "bits": config.bits,
        "group_size": config.group_size,
        "desc_act": False,
        "checkpoint_format": "gptq_v2",
    }
    (output_dir / "quantize_config.json").write_text(
        json.dumps(quantize_config, indent=2) + "\n",
        encoding="utf-8",
    )
    return str(output_dir)
