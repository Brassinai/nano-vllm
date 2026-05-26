"""GPTQ weight-only linear backend with a Triton packed-weight GEMM."""

from __future__ import annotations

import math
import os
from glob import glob
from typing import Any

import torch
import triton
import triton.language as tl
from safetensors import safe_open
from torch import nn

from nanovllm.quantization.base import (
    QuantizationConfig,
    QuantizationRegistry,
    QuantizeMethodBase,
)


@triton.jit
def _gptq_wna16_kernel(
    x_ptr,
    qweight_ptr,
    qzeros_ptr,
    scales_ptr,
    g_idx_ptr,
    bias_ptr,
    out_ptr,
    m_size,
    n_size,
    k_size,
    x_stride_m,
    x_stride_k,
    qweight_stride_k,
    qweight_stride_n,
    qzeros_stride_g,
    qzeros_stride_n,
    scales_stride_g,
    scales_stride_n,
    out_stride_m,
    out_stride_n,
    bits: tl.constexpr,
    maxq: tl.constexpr,
    pack_factor: tl.constexpr,
    zero_offset: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    group_m: tl.constexpr,
    has_bias: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(m_size, block_m)
    num_pid_n = tl.cdiv(n_size, block_n)
    group_width = group_m * num_pid_n
    first_pid_m = (pid // group_width) * group_m
    group_size_m = tl.minimum(num_pid_m - first_pid_m, group_m)
    pid_m = first_pid_m + ((pid % group_width) % group_size_m)
    pid_n = (pid % group_width) // group_size_m

    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_n = pid_n * block_n + tl.arange(0, block_n)
    offs_k = tl.arange(0, block_k)
    zero_shifts = (offs_n % pack_factor) * bits
    acc = tl.zeros((block_m, block_n), dtype=tl.float32)

    for k_start in tl.range(0, k_size, block_k):
        k = k_start + offs_k
        k_mask = k < k_size
        x = tl.load(
            x_ptr + offs_m[:, None] * x_stride_m + k[None, :] * x_stride_k,
            mask=(offs_m[:, None] < m_size) & k_mask[None, :],
            other=0.0,
        )

        packed_qweight = tl.load(
            qweight_ptr
            + (k[:, None] // pack_factor) * qweight_stride_k
            + offs_n[None, :] * qweight_stride_n,
            mask=k_mask[:, None] & (offs_n[None, :] < n_size),
            other=0,
        )
        weight_shifts = ((k % pack_factor) * bits)[:, None]
        qweight = (packed_qweight >> weight_shifts) & maxq

        group_idx = tl.load(g_idx_ptr + k, mask=k_mask, other=0)
        packed_qzeros = tl.load(
            qzeros_ptr
            + group_idx[:, None] * qzeros_stride_g
            + (offs_n[None, :] // pack_factor) * qzeros_stride_n,
            mask=k_mask[:, None] & (offs_n[None, :] < n_size),
            other=0,
        )
        qzeros = ((packed_qzeros >> zero_shifts[None, :]) & maxq) + zero_offset
        scales = tl.load(
            scales_ptr
            + group_idx[:, None] * scales_stride_g
            + offs_n[None, :] * scales_stride_n,
            mask=k_mask[:, None] & (offs_n[None, :] < n_size),
            other=0.0,
        )
        weight = ((qweight.to(tl.float32) - qzeros.to(tl.float32)) * scales).to(
            tl.float16
        )
        acc += tl.dot(x, weight)

    if has_bias:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < n_size, other=0.0)
        acc += bias[None, :]
    tl.store(
        out_ptr + offs_m[:, None] * out_stride_m + offs_n[None, :] * out_stride_n,
        acc,
        mask=(offs_m[:, None] < m_size) & (offs_n[None, :] < n_size),
    )


def _launch_gptq_wna16(
    x: torch.Tensor,
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    g_idx: torch.Tensor,
    bias: torch.Tensor | None,
    bits: int,
    zero_offset: int,
) -> torch.Tensor:
    if not x.is_cuda:
        raise RuntimeError("GPTQ Triton linear requires CUDA input tensors.")
    if x.dtype != torch.float16:
        raise TypeError(f"GPTQ Triton linear expects fp16 activations, got {x.dtype}.")

    x_2d = x.reshape(-1, x.shape[-1])
    m_size, k_size = x_2d.shape
    n_size = qweight.shape[1]
    out = torch.empty((m_size, n_size), dtype=x.dtype, device=x.device)
    pack_factor = 32 // bits
    grid = lambda meta: (
        triton.cdiv(m_size, meta["block_m"]) * triton.cdiv(n_size, meta["block_n"]),
    )
    _gptq_wna16_kernel[grid](
        x_2d,
        qweight,
        qzeros,
        scales,
        g_idx,
        bias if bias is not None else out,
        out,
        m_size,
        n_size,
        k_size,
        x_2d.stride(0),
        x_2d.stride(1),
        qweight.stride(0),
        qweight.stride(1),
        qzeros.stride(0),
        qzeros.stride(1),
        scales.stride(0),
        scales.stride(1),
        out.stride(0),
        out.stride(1),
        bits=bits,
        maxq=(1 << bits) - 1,
        pack_factor=pack_factor,
        zero_offset=zero_offset,
        block_m=16,
        block_n=64,
        block_k=32,
        group_m=8,
        has_bias=bias is not None,
        num_warps=4,
        num_stages=3,
    )
    return out.reshape(x.shape[:-1] + (n_size,))


@QuantizationRegistry.register("gptq")
class GPTQConfig(QuantizationConfig):
    """GPTQ checkpoint metadata for packed WNA16 linear layers."""

    def __init__(
        self,
        bits: int,
        group_size: int,
        desc_act: bool,
        checkpoint_format: str = "",
    ) -> None:
        if bits not in (2, 4, 8):
            raise ValueError(
                "The in-tree GPTQ Triton backend currently supports 2, 4, or 8 "
                f"bit int32-packed weights, got {bits}."
            )
        if desc_act:
            raise ValueError(
                "GPTQ desc_act checkpoints need activation-order weight handling "
                "that this Triton backend does not implement yet."
            )
        self.bits = bits
        self.group_size = group_size
        self.desc_act = desc_act
        self.checkpoint_format = checkpoint_format
        self.pack_factor = 32 // bits
        self.zero_offset = 0 if checkpoint_format == "gptq_v2" else 1
        self.quantized_modules: set[str] | None = None

    @classmethod
    def get_name(cls) -> str:
        return "gptq"

    @classmethod
    def get_config_filenames(cls) -> tuple[str, ...]:
        return ("quantize_config.json",)

    @classmethod
    def from_config(cls, raw_config: dict[str, Any]) -> "GPTQConfig":
        if cls.get_from_keys_or(raw_config, ("lm_head",), False):
            raise ValueError("GPTQ quantized lm_head checkpoints are not supported yet.")
        if cls.get_from_keys_or(raw_config, ("dynamic",), None):
            raise ValueError("GPTQ per-module dynamic overrides are not supported yet.")
        return cls(
            bits=int(cls.get_from_keys(raw_config, ("bits",))),
            group_size=int(cls.get_from_keys(raw_config, ("group_size",))),
            desc_act=bool(cls.get_from_keys_or(raw_config, ("desc_act",), False)),
            checkpoint_format=str(
                cls.get_from_keys_or(raw_config, ("checkpoint_format",), "")
            ),
        )

    def validate_runtime(self, dtype: torch.dtype) -> None:
        if dtype != torch.float16:
            raise TypeError(
                "GPTQ model quantization currently requires fp16 model activations; "
                f"resolved model dtype is {dtype}."
            )

    def update_from_model_path(self, model_path: str) -> None:
        quantized_modules = set()
        for file in glob(os.path.join(model_path, "*.safetensors")):
            with safe_open(file, "pt", "cpu") as weights:
                quantized_modules.update(
                    name.removesuffix(".qweight")
                    for name in weights.keys()
                    if name.endswith(".qweight")
                )
        if not quantized_modules:
            raise ValueError(
                "GPTQ was requested, but no packed '*.qweight' tensors were "
                f"found under {model_path!r}."
            )
        self.quantized_modules = quantized_modules

    def _source_prefixes(self, prefix: str) -> tuple[str, ...]:
        if prefix.endswith(".qkv_proj"):
            base = prefix.removesuffix("qkv_proj")
            return tuple(f"{base}{name}" for name in ("q_proj", "k_proj", "v_proj"))
        if prefix.endswith(".gate_up_proj"):
            base = prefix.removesuffix("gate_up_proj")
            return tuple(f"{base}{name}" for name in ("gate_proj", "up_proj"))
        return (prefix,)

    def _is_layer_quantized(self, prefix: str) -> bool:
        if self.quantized_modules is None:
            return True
        if prefix in self.quantized_modules:
            return True
        source_prefixes = self._source_prefixes(prefix)
        matches = tuple(name in self.quantized_modules for name in source_prefixes)
        if any(matches) and not all(matches):
            raise ValueError(
                f"GPTQ checkpoint only quantizes part of fused layer {prefix!r}: "
                f"expected {source_prefixes!r}."
            )
        return all(matches)

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> QuantizeMethodBase | None:
        if (
            getattr(layer, "supports_weight_quantization", False)
            and self._is_layer_quantized(prefix)
        ):
            return GPTQLinearMethod(self)
        return None


class GPTQLinearMethod(QuantizeMethodBase):
    def __init__(self, quant_config: GPTQConfig) -> None:
        self.quant_config = quant_config

    def create_weights(self, layer: nn.Module) -> None:
        pack_factor = self.quant_config.pack_factor
        if layer.input_size % pack_factor:
            raise ValueError(
                f"GPTQ input partition {layer.input_size} must be divisible by "
                f"pack factor {pack_factor}."
            )
        if layer.output_size % pack_factor:
            raise ValueError(
                f"GPTQ output partition {layer.output_size} must be divisible by "
                f"pack factor {pack_factor}."
            )

        group_size = (
            layer.full_input_size
            if self.quant_config.group_size == -1
            else self.quant_config.group_size
        )
        if group_size <= 0:
            raise ValueError(
                f"Unsupported GPTQ group size {self.quant_config.group_size}."
            )
        group_count = math.ceil(layer.full_input_size / group_size)

        qweight = nn.Parameter(
            torch.empty(
                layer.input_size // pack_factor,
                layer.output_size,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        qzeros = nn.Parameter(
            torch.empty(
                group_count,
                layer.output_size // pack_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        scales = nn.Parameter(
            torch.empty(
                group_count,
                layer.output_size,
                dtype=torch.get_default_dtype(),
            ),
            requires_grad=False,
        )
        global_k = torch.arange(layer.input_size, dtype=torch.int32)
        if self.quant_config.group_size != -1:
            global_k += layer.input_partition_start
            global_k //= group_size
        else:
            global_k.zero_()
        g_idx = nn.Parameter(global_k, requires_grad=False)

        for name, param in (
            ("qweight", qweight),
            ("qzeros", qzeros),
            ("scales", scales),
            ("g_idx", g_idx),
        ):
            param.weight_loader = self._make_weight_loader(layer, name)
            layer.register_parameter(name, param)

    def _make_weight_loader(self, layer: nn.Module, name: str):
        def load(
            param: nn.Parameter,
            loaded_weight: torch.Tensor,
            loaded_shard_id: str | int | None = None,
        ) -> None:
            self._load_param(
                layer,
                name,
                param,
                loaded_weight,
                loaded_shard_id,
            )

        return load

    @staticmethod
    def _copy_exact(param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        if param.shape != loaded_weight.shape:
            raise ValueError(
                f"Loaded GPTQ tensor shape {tuple(loaded_weight.shape)} does not "
                f"match parameter shape {tuple(param.shape)}."
            )
        param.data.copy_(loaded_weight)

    @staticmethod
    def _load_output_param(
        layer: nn.Module,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        target_offset: int = 0,
    ) -> None:
        target_size = loaded_weight.size(1) // layer.tp_size
        loaded_weight = loaded_weight.narrow(1, layer.tp_rank * target_size, target_size)
        param.data.narrow(1, target_offset, target_size).copy_(loaded_weight)

    def _fused_output_offset(
        self,
        layer: nn.Module,
        name: str,
        loaded_shard_id: str | int | None,
    ) -> int:
        if loaded_shard_id is None:
            return 0
        if isinstance(loaded_shard_id, str):
            shard_offset, _ = layer._qkv_shard(loaded_shard_id)
        else:
            shard_offset = sum(layer.output_sizes[:loaded_shard_id]) // layer.tp_size
        if name == "qzeros":
            return shard_offset // self.quant_config.pack_factor
        return shard_offset

    def _load_param(
        self,
        layer: nn.Module,
        name: str,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: str | int | None,
    ) -> None:
        if layer.tp_dim is None:
            self._copy_exact(param, loaded_weight)
            return

        if layer.tp_dim == 1:
            if name in ("qweight", "g_idx"):
                local_rows = param.size(0)
                param.data.copy_(
                    loaded_weight.narrow(0, layer.tp_rank * local_rows, local_rows)
                )
            else:
                self._copy_exact(param, loaded_weight)
            return

        if name == "g_idx":
            self._copy_exact(param, loaded_weight)
            return

        self._load_output_param(
            layer,
            param,
            loaded_weight,
            target_offset=self._fused_output_offset(layer, name, loaded_shard_id),
        )

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return _launch_gptq_wna16(
            x,
            layer.qweight,
            layer.qzeros,
            layer.scales,
            layer.g_idx,
            bias,
            self.quant_config.bits,
            self.quant_config.zero_offset,
        )
