"""AWQ export utilities for converting dense HF checkpoints into nano-vLLM format."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from nanovllm.quantization.gptq_export import (
    DEFAULT_CALIBRATION_TEXTS,
    _get_base_model,
    _get_decoder_layers,
    _get_layer_output_hidden,
    collect_export_state_dict,
    copy_support_files,
    find_linear_layers,
    layer_quantization_groups,
    make_calibration_batches,
)


@dataclass
class AWQTensorPack:
    qweight: torch.Tensor
    qzeros: torch.Tensor
    scales: torch.Tensor


@dataclass
class AWQExportConfig:
    bits: int = 4
    group_size: int = 128
    nsamples: int = 32
    seqlen: int = 512
    seed: int = 0
    clip_steps: int = 10
    min_clip_ratio: float = 0.5
    dtype: str = "float16"
    device: str = "cuda"


def pack_awq_qweight(q_int: torch.Tensor, *, bits: int) -> torch.Tensor:
    """Pack [out_features, in_features] integer weights to AWQ GEMM layout."""
    pack_factor = 32 // bits
    out_features, in_features = q_int.shape
    if out_features % pack_factor:
        raise ValueError(
            f"out_features={out_features} must be divisible by "
            f"pack_factor={pack_factor}."
        )
    transposed = q_int.t().contiguous().cpu().to(torch.int32)
    packed = torch.zeros(
        (in_features, out_features // pack_factor),
        dtype=torch.int32,
    )
    for i in range(pack_factor):
        packed |= transposed[:, i::pack_factor] << (i * bits)
    return packed


def pack_awq_qzeros(qzeros: torch.Tensor, *, bits: int) -> torch.Tensor:
    """Pack [group_count, out_features] AWQ zero-points across output channels."""
    pack_factor = 32 // bits
    group_count, out_features = qzeros.shape
    if out_features % pack_factor:
        raise ValueError(
            f"out_features={out_features} must be divisible by "
            f"pack_factor={pack_factor}."
        )
    values = qzeros.contiguous().cpu().to(torch.int32)
    packed = torch.zeros(
        (group_count, out_features // pack_factor),
        dtype=torch.int32,
    )
    for i in range(pack_factor):
        packed |= values[:, i::pack_factor] << (i * bits)
    return packed


def unpack_awq_qweight(
    qweight: torch.Tensor,
    *,
    bits: int,
    out_features: int,
) -> torch.Tensor:
    """Inverse of :func:`pack_awq_qweight` for tests/debugging."""
    pack_factor = 32 // bits
    cols = []
    for i in range(pack_factor):
        cols.append(((qweight >> (i * bits)) & ((1 << bits) - 1)).to(torch.int32))
    return torch.stack(cols, dim=2).reshape(qweight.shape[0], out_features).t().contiguous()


def unpack_awq_qzeros(
    qzeros: torch.Tensor,
    *,
    bits: int,
    out_features: int,
) -> torch.Tensor:
    """Inverse of :func:`pack_awq_qzeros` for tests/debugging."""
    pack_factor = 32 // bits
    cols = []
    for i in range(pack_factor):
        cols.append(((qzeros >> (i * bits)) & ((1 << bits) - 1)).to(torch.int32))
    return torch.stack(cols, dim=2).reshape(qzeros.shape[0], out_features).contiguous()


def _quantize_from_bounds(
    weight: torch.Tensor,
    min_val: torch.Tensor,
    max_val: torch.Tensor,
    *,
    maxq: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    scale = (max_val - min_val).clamp(min=1e-6) / maxq
    zero = torch.round(-min_val / scale).clamp(1, maxq)
    q = torch.round(weight / scale[:, None] + zero[:, None]).clamp(0, maxq)
    dequant = (q - zero[:, None]) * scale[:, None]
    return q.to(torch.int32), dequant, scale


def _search_group_quantization(
    weight: torch.Tensor,
    input_scale: torch.Tensor,
    config: AWQExportConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    maxq = (1 << config.bits) - 1
    out_features = weight.shape[0]
    ratios = torch.linspace(
        1.0,
        config.min_clip_ratio,
        steps=max(1, config.clip_steps),
        device=weight.device,
    )
    if ratios.numel() == 0:
        ratios = torch.ones(1, device=weight.device)

    best_err = torch.full((out_features,), float("inf"), device=weight.device)
    best_q = torch.zeros_like(weight, dtype=torch.int32)
    best_zero = torch.zeros(out_features, dtype=torch.int32, device=weight.device)
    best_scale = torch.ones(out_features, dtype=torch.float32, device=weight.device)
    base_min = torch.minimum(weight.min(dim=1).values, torch.zeros(out_features, device=weight.device))
    base_max = torch.maximum(weight.max(dim=1).values, torch.zeros(out_features, device=weight.device))
    input_scale = input_scale.to(weight.device).float().clamp(min=1e-6)

    for ratio in ratios:
        min_val = base_min * ratio
        max_val = base_max * ratio
        clipped = weight.clamp(min=min_val[:, None], max=max_val[:, None])
        q, dequant, scale = _quantize_from_bounds(
            clipped,
            min_val,
            max_val,
            maxq=maxq,
        )
        err = ((dequant - weight).float().square() * input_scale[None, :]).sum(dim=1)
        better = err < best_err
        if torch.any(better):
            best_err[better] = err[better]
            best_q[better] = q[better]
            best_zero[better] = torch.round(-min_val[better] / scale[better]).clamp(
                1,
                maxq,
            ).to(torch.int32)
            best_scale[better] = scale[better]

    return best_q, best_zero, best_scale


def quantize_awq_linear(
    layer: nn.Linear,
    input_scale: torch.Tensor,
    config: AWQExportConfig,
) -> AWQTensorPack:
    if config.bits != 4:
        raise ValueError("AWQ export currently supports 4-bit weights.")
    if config.group_size <= 0:
        raise ValueError(f"Unsupported AWQ group size {config.group_size}.")

    weight = layer.weight.data.float()
    out_features, in_features = weight.shape
    group_count = math.ceil(in_features / config.group_size)
    q_int = torch.zeros_like(weight, dtype=torch.int32)
    zeros = torch.zeros((group_count, out_features), dtype=torch.int32, device=weight.device)
    scales = torch.zeros((group_count, out_features), dtype=torch.float32, device=weight.device)

    input_scale = input_scale.to(weight.device).float()
    if input_scale.numel() != in_features:
        raise ValueError(
            f"AWQ input scale has {input_scale.numel()} values, expected {in_features}."
        )

    for group_idx in range(group_count):
        start = group_idx * config.group_size
        end = min(start + config.group_size, in_features)
        q_group, zero, scale = _search_group_quantization(
            weight[:, start:end],
            input_scale[start:end],
            config,
        )
        q_int[:, start:end] = q_group
        zeros[group_idx] = zero
        scales[group_idx] = scale

    return AWQTensorPack(
        qweight=pack_awq_qweight(q_int, bits=config.bits),
        qzeros=pack_awq_qzeros(zeros - 1, bits=config.bits),
        scales=scales.contiguous().to(torch.float16).cpu(),
    )


@torch.inference_mode()
def quantize_hf_model_awq(
    model: nn.Module,
    calibration_batches: list[torch.Tensor],
    config: AWQExportConfig,
    *,
    device: str,
) -> dict[str, AWQTensorPack]:
    if torch.is_grad_enabled():
        raise RuntimeError("AWQ export should run with gradients disabled.")
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
        quantized: dict[str, AWQTensorPack] = {}

        for layer_idx, _ in enumerate(layers):
            layer = base_model.layers[layer_idx].to(device)
            full = find_linear_layers(layer)
            for group_names in layer_quantization_groups(layer):
                subset = {name: full[name] for name in group_names}
                input_sums: dict[str, torch.Tensor] = {}
                input_counts: dict[str, int] = {}
                handles = []

                def add_batch(name: str):
                    def hook(_module, inp, _out):
                        x = inp[0].detach().float().reshape(-1, inp[0].shape[-1])
                        input_sums[name] = input_sums.get(
                            name,
                            torch.zeros(x.shape[-1], device=x.device),
                        ) + x.abs().sum(dim=0)
                        input_counts[name] = input_counts.get(name, 0) + x.shape[0]

                    return hook

                for name, sublayer in subset.items():
                    handles.append(sublayer.register_forward_hook(add_batch(name)))

                for sample_idx in range(nsamples):
                    outs[sample_idx] = _get_layer_output_hidden(
                        layer(inps[sample_idx].unsqueeze(0), **layer_kwargs)
                    ).squeeze(0)

                for handle in handles:
                    handle.remove()

                for name, sublayer in subset.items():
                    prefix = f"model.layers.{layer_idx}.{name}"
                    input_scale = input_sums[name] / max(input_counts[name], 1)
                    quantized[prefix] = quantize_awq_linear(
                        sublayer,
                        input_scale,
                        config,
                    )

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


def collect_awq_export_state_dict(
    model: nn.Module,
    quantized: dict[str, AWQTensorPack],
) -> dict[str, torch.Tensor]:
    export_state = collect_export_state_dict(model, {})
    quantized_prefixes = set(quantized)
    export_state = {
        name: tensor
        for name, tensor in export_state.items()
        if not (name.endswith(".weight") and name[: -len(".weight")] in quantized_prefixes)
    }
    for prefix, packed in quantized.items():
        export_state[f"{prefix}.qweight"] = packed.qweight
        export_state[f"{prefix}.qzeros"] = packed.qzeros
        export_state[f"{prefix}.scales"] = packed.scales
    return export_state


def export_awq_checkpoint(
    model_path: str,
    output_path: str,
    *,
    config: AWQExportConfig,
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
        device_map="cpu",
    )
    model.eval()
    model.requires_grad_(False)

    texts = calibration_texts or list(DEFAULT_CALIBRATION_TEXTS)
    batches = make_calibration_batches(
        tokenizer,
        texts,
        nsamples=config.nsamples,
        seqlen=config.seqlen,
        seed=config.seed,
    )
    quantized = quantize_hf_model_awq(
        model,
        batches,
        config,
        device=config.device,
    )
    export_state = collect_awq_export_state_dict(model, quantized)

    copy_support_files(model_path, str(output_dir))
    save_file(export_state, str(output_dir / "model.safetensors"))
    quantize_config = {
        "quant_method": "awq",
        "zero_point": True,
        "q_group_size": config.group_size,
        "w_bit": config.bits,
        "version": "gemm",
        "modules_to_not_convert": ["lm_head"],
    }
    (output_dir / "quantize_config.json").write_text(
        json.dumps(quantize_config, indent=2) + "\n",
        encoding="utf-8",
    )
    return str(output_dir)
