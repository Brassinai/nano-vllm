"""INT4 KV cache with optional SAW-INT4/BDR rotation for nano-vllm.

The packed layout follows the SGLang INT4 path used by
togethercomputer/saw-int4: one byte stores the lower-half and upper-half
dimensions for a token/head. Plain ``int4`` stores K/V directly. BDR mode
adds block-diagonal Hadamard rotation on K before packing, and optionally V
when ``ROTATE_V=1``.
"""

from __future__ import annotations

import math
import os
from functools import lru_cache
from typing import Tuple

import torch
import triton
import triton.language as tl

from nanovllm.kvcache.base import BaseKVCache, KVCacheRegistry


MAX_HADAMARD_ORDER = 4096


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.lower() in ("1", "true", "yes", "on")


def _env_bool_alias(
    primary: str,
    fallback: str,
    default: bool = False,
) -> bool:
    if primary in os.environ:
        return _env_bool(primary, default)
    return _env_bool(fallback, default)


def _env_int_alias(primary: str, fallback: str, default: int) -> int:
    raw = os.environ.get(primary)
    if raw is None:
        raw = os.environ.get(fallback)
    if raw is not None:
        return int(raw)
    return default


def _env_bool_alias_optional(primary: str, fallback: str) -> bool | None:
    if primary in os.environ:
        return _env_bool(primary)
    if fallback in os.environ:
        return _env_bool(fallback)
    return None


def _default_hadamard_order(head_dim: int) -> int:
    del head_dim
    return _env_int_alias("NANOVLLM_SAW_INT4_HADAMARD_ORDER", "HADAMARD_ORDER", 16)


def _device_supports_bf16(device: str | torch.device) -> bool:
    if not torch.cuda.is_available():
        return False
    torch_device = torch.device(device)
    if torch_device.type != "cuda":
        return False
    index = torch_device.index
    if index is None:
        index = torch.cuda.current_device()
    major, _minor = torch.cuda.get_device_capability(index)
    return major >= 8


def validate_hadamard_order(hadamard_order: int, head_dim: int) -> None:
    if hadamard_order < 2:
        raise ValueError(f"hadamard_order must be >= 2, got {hadamard_order}")
    if hadamard_order & (hadamard_order - 1):
        raise ValueError(
            f"hadamard_order must be a power of two, got {hadamard_order}"
        )
    if hadamard_order > MAX_HADAMARD_ORDER:
        raise ValueError(
            f"hadamard_order must be <= {MAX_HADAMARD_ORDER}, got {hadamard_order}"
        )
    if head_dim % hadamard_order:
        raise ValueError(
            f"head_dim ({head_dim}) must be divisible by hadamard_order "
            f"({hadamard_order})"
        )


@lru_cache(maxsize=16)
def hadamard_matrix(hadamard_order: int) -> torch.Tensor:
    validate_hadamard_order(hadamard_order, hadamard_order)
    matrix = torch.tensor([[1.0]], dtype=torch.float32)
    base = torch.tensor([[1.0, 1.0], [1.0, -1.0]], dtype=torch.float32)
    dim = 1
    while dim < hadamard_order:
        matrix = torch.kron(matrix, base)
        dim *= 2
    return matrix / math.sqrt(hadamard_order)


@triton.jit
def _fwht_blocked_segments(x, D: tl.constexpr, LOG: tl.constexpr):
    i = tl.arange(0, D)
    for s in tl.static_range(0, LOG):
        stride = 1 << s
        partner = i ^ stride
        x_partner = tl.gather(x, partner, 0)
        is_lo = ((i >> s) & 1) == 0
        x = tl.where(is_lo, x + x_partner, x_partner - x)
    return x


@triton.jit
def store_kvcache_saw_int4_kernel(
    input_ptr,
    input_stride_token,
    input_stride_head,
    input_stride_dim,
    cache_ptr,
    scales_zeros_ptr,
    slot_mapping_ptr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    head_dim_pad: tl.constexpr,
    cache_stride_slot,
    cache_stride_head,
    cache_stride_dim,
    sz_stride_slot,
    sz_stride_head,
    sz_stride_dim,
    LOG: tl.constexpr,
    PRE_SCALE: tl.constexpr,
    BLOCK_HALF: tl.constexpr,
    HEADS_PER_PROGRAM: tl.constexpr,
    FUSE_HADAMARD: tl.constexpr,
    ROUND_TO_BF16: tl.constexpr,
):
    token_idx = tl.program_id(0)
    head_group = tl.program_id(1)
    slot = tl.load(slot_mapping_ptr + token_idx)
    if slot == -1:
        return

    for h in tl.static_range(0, HEADS_PER_PROGRAM):
        head_idx = head_group * HEADS_PER_PROGRAM + h
        if head_idx < num_heads:
            half_dim = head_dim // 2
            dim_off = tl.arange(0, BLOCK_HALF)
            dim_mask = dim_off < half_dim
            input_base = token_idx * input_stride_token + head_idx * input_stride_head

            if FUSE_HADAMARD:
                dim_full = tl.arange(0, head_dim_pad)
                full_mask = dim_full < head_dim
                x = tl.load(
                    input_ptr + input_base + dim_full * input_stride_dim,
                    mask=full_mask,
                    other=0.0,
                ).to(tl.float32)
                x = _fwht_blocked_segments(x * PRE_SCALE, head_dim_pad, LOG)
                vals1 = tl.gather(x, tl.where(dim_mask, dim_off, 0), 0)
                vals2 = tl.gather(x, tl.where(dim_mask, dim_off + half_dim, 0), 0)
            else:
                vals1 = tl.load(
                    input_ptr + input_base + dim_off * input_stride_dim,
                    mask=dim_mask,
                    other=0.0,
                ).to(tl.float32)
                vals2 = tl.load(
                    input_ptr + input_base + (dim_off + half_dim) * input_stride_dim,
                    mask=dim_mask,
                    other=0.0,
                ).to(tl.float32)

            if ROUND_TO_BF16:
                vals1 = vals1.to(tl.bfloat16).to(tl.float32)
                vals2 = vals2.to(tl.bfloat16).to(tl.float32)

            vals1_for_min = tl.where(dim_mask, vals1, float("inf"))
            vals2_for_min = tl.where(dim_mask, vals2, float("inf"))
            vals1_for_max = tl.where(dim_mask, vals1, -float("inf"))
            vals2_for_max = tl.where(dim_mask, vals2, -float("inf"))
            val_min = tl.minimum(tl.min(vals1_for_min, axis=0), tl.min(vals2_for_min, axis=0))
            val_max = tl.maximum(tl.max(vals1_for_max, axis=0), tl.max(vals2_for_max, axis=0))
            scale = tl.maximum(val_max - val_min, 1e-8) / 15.0
            zero = -val_min / scale

            q1 = tl.clamp(tl.floor(vals1 / scale + zero + 0.5), 0, 15).to(tl.uint8)
            q2 = tl.clamp(tl.floor(vals2 / scale + zero + 0.5), 0, 15).to(tl.uint8)
            packed = q1 | (q2 << 4)

            cache_base = slot * cache_stride_slot + head_idx * cache_stride_head
            tl.store(
                cache_ptr + cache_base + dim_off * cache_stride_dim,
                packed,
                mask=dim_mask,
            )

            sz_base = slot * sz_stride_slot + head_idx * sz_stride_head
            tl.store(scales_zeros_ptr + sz_base + 0 * sz_stride_dim, scale)
            tl.store(scales_zeros_ptr + sz_base + 1 * sz_stride_dim, zero)


@triton.jit
def dequantize_saw_int4_sequence_kernel(
    k_cache_ptr,
    v_cache_ptr,
    block_table_ptr,
    k_scales_zeros_ptr,
    v_scales_zeros_ptr,
    k_out_ptr,
    v_out_ptr,
    k_cache_stride_block,
    k_cache_stride_pos,
    k_cache_stride_head,
    v_cache_stride_block,
    v_cache_stride_pos,
    v_cache_stride_head,
    block_table_stride,
    k_sz_stride_block,
    k_sz_stride_pos,
    k_sz_stride_head,
    v_sz_stride_block,
    v_sz_stride_pos,
    v_sz_stride_head,
    out_stride_token,
    out_stride_head,
    seq_len,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    block_half: tl.constexpr,
):
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    if token_idx >= seq_len:
        return

    page_idx = token_idx // block_size
    page_off = token_idx % block_size
    block_num = tl.load(block_table_ptr + page_idx * block_table_stride).to(tl.int64)

    half_dim = head_dim // 2
    dim_off = tl.arange(0, block_half)
    dim_mask = dim_off < half_dim

    k_base = (
        block_num * k_cache_stride_block
        + page_off.to(tl.int64) * k_cache_stride_pos
        + tl.cast(head_idx, tl.int64) * k_cache_stride_head
    )
    v_base = (
        block_num * v_cache_stride_block
        + page_off.to(tl.int64) * v_cache_stride_pos
        + tl.cast(head_idx, tl.int64) * v_cache_stride_head
    )
    k_meta_base = (
        block_num * k_sz_stride_block
        + page_off.to(tl.int64) * k_sz_stride_pos
        + tl.cast(head_idx, tl.int64) * k_sz_stride_head
    )
    v_meta_base = (
        block_num * v_sz_stride_block
        + page_off.to(tl.int64) * v_sz_stride_pos
        + tl.cast(head_idx, tl.int64) * v_sz_stride_head
    )
    out_base = token_idx * out_stride_token + head_idx * out_stride_head

    k_packed = tl.load(k_cache_ptr + k_base + dim_off, mask=dim_mask, other=0)
    k_scale = tl.load(k_scales_zeros_ptr + k_meta_base + 0).to(tl.float32)
    k_zero = tl.load(k_scales_zeros_ptr + k_meta_base + 1).to(tl.float32)
    k_low = ((k_packed & 0x0F).to(tl.float32) - k_zero) * k_scale
    k_high = (((k_packed >> 4) & 0x0F).to(tl.float32) - k_zero) * k_scale
    tl.store(k_out_ptr + out_base + dim_off, k_low, mask=dim_mask)
    tl.store(k_out_ptr + out_base + dim_off + half_dim, k_high, mask=dim_mask)

    v_packed = tl.load(v_cache_ptr + v_base + dim_off, mask=dim_mask, other=0)
    v_scale = tl.load(v_scales_zeros_ptr + v_meta_base + 0).to(tl.float32)
    v_zero = tl.load(v_scales_zeros_ptr + v_meta_base + 1).to(tl.float32)
    v_low = ((v_packed & 0x0F).to(tl.float32) - v_zero) * v_scale
    v_high = (((v_packed >> 4) & 0x0F).to(tl.float32) - v_zero) * v_scale
    tl.store(v_out_ptr + out_base + dim_off, v_low, mask=dim_mask)
    tl.store(v_out_ptr + out_base + dim_off + half_dim, v_high, mask=dim_mask)


def dequantize_saw_int4_sequence(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    seq_len: int,
    block_table_row: torch.Tensor,
    k_scales_zeros: torch.Tensor,
    v_scales_zeros: torch.Tensor,
    out_dtype: torch.dtype,
    head_dim: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if seq_len <= 0:
        raise ValueError("SAW-INT4 dequantization expects a positive sequence length.")
    if head_dim is None:
        head_dim = k_cache.shape[-1] * 2

    num_heads = k_cache.shape[2]
    block_size = k_cache.shape[1]
    dense_k = torch.empty(
        (seq_len, num_heads, head_dim), dtype=out_dtype, device=k_cache.device
    )
    dense_v = torch.empty_like(dense_k)
    block_table_row = block_table_row.to(
        device=k_cache.device, dtype=torch.int32
    ).contiguous()

    block_half = triton.next_power_of_2(head_dim // 2)
    dequantize_saw_int4_sequence_kernel[(seq_len, num_heads)](
        k_cache,
        v_cache,
        block_table_row,
        k_scales_zeros,
        v_scales_zeros,
        dense_k,
        dense_v,
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        block_table_row.stride(0),
        k_scales_zeros.stride(0),
        k_scales_zeros.stride(1),
        k_scales_zeros.stride(2),
        v_scales_zeros.stride(0),
        v_scales_zeros.stride(1),
        v_scales_zeros.stride(2),
        dense_k.stride(0),
        dense_k.stride(1),
        seq_len,
        head_dim=head_dim,
        block_size=block_size,
        block_half=block_half,
    )
    return dense_k, dense_v


@triton.jit
def dequantize_saw_int4_batch_kernel(
    k_cache_ptr,
    v_cache_ptr,
    block_table_ptr,
    cu_seqlens_k_ptr,
    k_scales_zeros_ptr,
    v_scales_zeros_ptr,
    k_out_ptr,
    v_out_ptr,
    k_cache_stride_block,
    k_cache_stride_pos,
    k_cache_stride_head,
    v_cache_stride_block,
    v_cache_stride_pos,
    v_cache_stride_head,
    block_table_stride_batch,
    block_table_stride_block,
    k_sz_stride_block,
    k_sz_stride_pos,
    k_sz_stride_head,
    v_sz_stride_block,
    v_sz_stride_pos,
    v_sz_stride_head,
    out_stride_token,
    out_stride_head,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    block_half: tl.constexpr,
    block_tokens: tl.constexpr,
):
    token_block = tl.program_id(0)
    batch_idx = tl.program_id(1)
    head_idx = tl.program_id(2)

    seq_start = tl.load(cu_seqlens_k_ptr + batch_idx).to(tl.int64)
    seq_end = tl.load(cu_seqlens_k_ptr + batch_idx + 1).to(tl.int64)
    seq_len = seq_end - seq_start

    token_off = token_block * block_tokens + tl.arange(0, block_tokens)
    token_mask = token_off < seq_len
    page_idx = token_off // block_size
    page_off = token_off % block_size
    block_num = tl.load(
        block_table_ptr
        + batch_idx * block_table_stride_batch
        + page_idx * block_table_stride_block,
        mask=token_mask,
        other=0,
    ).to(tl.int64)

    half_dim = head_dim // 2
    dim_off = tl.arange(0, block_half)
    dim_mask = dim_off < half_dim

    k_base = (
        block_num[:, None] * k_cache_stride_block
        + page_off[:, None].to(tl.int64) * k_cache_stride_pos
        + tl.cast(head_idx, tl.int64) * k_cache_stride_head
    )
    v_base = (
        block_num[:, None] * v_cache_stride_block
        + page_off[:, None].to(tl.int64) * v_cache_stride_pos
        + tl.cast(head_idx, tl.int64) * v_cache_stride_head
    )
    k_meta_base = (
        block_num * k_sz_stride_block
        + page_off.to(tl.int64) * k_sz_stride_pos
        + tl.cast(head_idx, tl.int64) * k_sz_stride_head
    )
    v_meta_base = (
        block_num * v_sz_stride_block
        + page_off.to(tl.int64) * v_sz_stride_pos
        + tl.cast(head_idx, tl.int64) * v_sz_stride_head
    )
    out_base = (
        (seq_start + token_off)[:, None] * out_stride_token
        + tl.cast(head_idx, tl.int64) * out_stride_head
    )

    k_packed = tl.load(
        k_cache_ptr + k_base + dim_off[None, :],
        mask=token_mask[:, None] & dim_mask[None, :],
        other=0,
    )
    k_scale = tl.load(
        k_scales_zeros_ptr + k_meta_base + 0,
        mask=token_mask,
        other=1.0,
    ).to(tl.float32)
    k_zero = tl.load(
        k_scales_zeros_ptr + k_meta_base + 1,
        mask=token_mask,
        other=0.0,
    ).to(tl.float32)
    k_low = ((k_packed & 0x0F).to(tl.float32) - k_zero[:, None]) * k_scale[:, None]
    k_high = (
        ((k_packed >> 4) & 0x0F).to(tl.float32) - k_zero[:, None]
    ) * k_scale[:, None]
    tl.store(
        k_out_ptr + out_base + dim_off[None, :],
        k_low,
        mask=token_mask[:, None] & dim_mask[None, :],
    )
    tl.store(
        k_out_ptr + out_base + dim_off[None, :] + half_dim,
        k_high,
        mask=token_mask[:, None] & dim_mask[None, :],
    )

    v_packed = tl.load(
        v_cache_ptr + v_base + dim_off[None, :],
        mask=token_mask[:, None] & dim_mask[None, :],
        other=0,
    )
    v_scale = tl.load(
        v_scales_zeros_ptr + v_meta_base + 0,
        mask=token_mask,
        other=1.0,
    ).to(tl.float32)
    v_zero = tl.load(
        v_scales_zeros_ptr + v_meta_base + 1,
        mask=token_mask,
        other=0.0,
    ).to(tl.float32)
    v_low = ((v_packed & 0x0F).to(tl.float32) - v_zero[:, None]) * v_scale[:, None]
    v_high = (
        ((v_packed >> 4) & 0x0F).to(tl.float32) - v_zero[:, None]
    ) * v_scale[:, None]
    tl.store(
        v_out_ptr + out_base + dim_off[None, :],
        v_low,
        mask=token_mask[:, None] & dim_mask[None, :],
    )
    tl.store(
        v_out_ptr + out_base + dim_off[None, :] + half_dim,
        v_high,
        mask=token_mask[:, None] & dim_mask[None, :],
    )


def dequantize_saw_int4_batch(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    k_scales_zeros: torch.Tensor,
    v_scales_zeros: torch.Tensor,
    out_dtype: torch.dtype,
    max_seqlen_k: int,
    head_dim: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if head_dim is None:
        head_dim = k_cache.shape[-1] * 2
    total_tokens = int(cu_seqlens_k[-1].item())
    if total_tokens <= 0:
        raise ValueError("INT4 dequantization expects a positive token count.")

    num_heads = k_cache.shape[2]
    block_size = k_cache.shape[1]
    dense_k = torch.empty(
        (total_tokens, num_heads, head_dim),
        dtype=out_dtype,
        device=k_cache.device,
    )
    dense_v = torch.empty_like(dense_k)
    block_table = block_table.to(device=k_cache.device, dtype=torch.int32).contiguous()
    cu_seqlens_k = cu_seqlens_k.to(device=k_cache.device, dtype=torch.int32).contiguous()

    block_tokens = 32
    block_half = triton.next_power_of_2(head_dim // 2)
    grid = (triton.cdiv(max_seqlen_k, block_tokens), block_table.shape[0], num_heads)
    dequantize_saw_int4_batch_kernel[grid](
        k_cache,
        v_cache,
        block_table,
        cu_seqlens_k,
        k_scales_zeros,
        v_scales_zeros,
        dense_k,
        dense_v,
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        block_table.stride(0),
        block_table.stride(1),
        k_scales_zeros.stride(0),
        k_scales_zeros.stride(1),
        k_scales_zeros.stride(2),
        v_scales_zeros.stride(0),
        v_scales_zeros.stride(1),
        v_scales_zeros.stride(2),
        dense_k.stride(0),
        dense_k.stride(1),
        head_dim=head_dim,
        block_size=block_size,
        block_half=block_half,
        block_tokens=block_tokens,
    )
    return dense_k, dense_v


@KVCacheRegistry.register("int4")
class Int4KVCache(BaseKVCache):
    """Token-wise affine INT4 KV cache with optional BDR via ``HADAMARD=1``."""

    default_hadamard_enabled = False

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
        hadamard_enabled: bool | None = None,
        hadamard_order: int | None = None,
        rotate_v: bool | None = None,
    ):
        super().__init__(num_blocks, block_size, num_heads, head_dim, dtype, device)
        if head_dim % 2:
            raise ValueError(f"INT4 KV cache requires an even head_dim, got {head_dim}")
        if hadamard_enabled is None:
            hadamard_enabled = _env_bool_alias(
                "NANOVLLM_SAW_INT4_HADAMARD",
                "HADAMARD",
                self.default_hadamard_enabled,
            )
        self.hadamard_enabled = bool(hadamard_enabled)
        self.hadamard_order = hadamard_order or _default_hadamard_order(head_dim)
        if self.hadamard_enabled:
            validate_hadamard_order(self.hadamard_order, head_dim)
        rotate_v_enabled = (
            _env_bool_alias("NANOVLLM_SAW_INT4_ROTATE_V", "ROTATE_V", False)
            if rotate_v is None
            else rotate_v
        )
        self.rotate_v = bool(self.hadamard_enabled and rotate_v_enabled)
        self.head_dim_pad = triton.next_power_of_2(head_dim)
        self.packed_dim = head_dim // 2
        self.heads_per_program = 1 if self.head_dim_pad >= 512 else min(8, num_heads)
        fuse_hadamard = _env_bool_alias_optional(
            "NANOVLLM_SAW_INT4_FUSE_HADAMARD",
            "FUSE_HADAMARD",
        )
        if fuse_hadamard is None:
            fuse_hadamard = hasattr(tl, "gather")
        if self.hadamard_enabled and fuse_hadamard and not hasattr(tl, "gather"):
            raise RuntimeError(
                "Fused SAW-INT4 Hadamard needs triton.language.gather. "
                "Upgrade Triton or unset NANOVLLM_SAW_INT4_FUSE_HADAMARD."
            )
        self.fuse_hadamard = bool(self.hadamard_enabled and fuse_hadamard)
        self._rotation_cache: dict[
            tuple[int, int, torch.device, torch.dtype],
            torch.Tensor,
        ] = {}
        self.round_hadamard_to_bf16 = (
            self.hadamard_enabled
            and self.dtype == torch.bfloat16
            and _device_supports_bf16(self.device)
            and self.fuse_hadamard
        )

    def _get_rotation(
        self,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        key = (self.hadamard_order, self.head_dim, device, dtype)
        cached = self._rotation_cache.get(key)
        if cached is None:
            block = hadamard_matrix(self.hadamard_order).to(device=device, dtype=dtype)
            segments = self.head_dim // self.hadamard_order
            cached = torch.block_diag(*([block] * segments))
            self._rotation_cache[key] = cached
        return cached

    def _rotate(self, x: torch.Tensor) -> torch.Tensor:
        rotation = self._get_rotation(x.device, x.dtype)
        return torch.matmul(x, rotation)

    def allocate(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cache_shape = (
            self.num_blocks,
            self.block_size,
            self.num_heads,
            self.packed_dim,
        )
        meta_shape = (self.num_blocks, self.block_size, self.num_heads, 2)
        k_cache = torch.zeros(cache_shape, dtype=torch.uint8, device=self.device)
        v_cache = torch.zeros(cache_shape, dtype=torch.uint8, device=self.device)
        k_scales_zeros = torch.zeros(meta_shape, dtype=torch.float32, device=self.device)
        v_scales_zeros = torch.zeros(meta_shape, dtype=torch.float32, device=self.device)
        metadata = torch.tensor(
            [
                int(self.hadamard_enabled),
                self.hadamard_order,
                int(self.rotate_v),
                self.head_dim,
            ],
            dtype=torch.int32,
            device="cpu",
        )
        return k_cache, v_cache, k_scales_zeros, v_scales_zeros, metadata

    def _store_one(
        self,
        x: torch.Tensor,
        cache: torch.Tensor,
        scales_zeros: torch.Tensor,
        slot_mapping: torch.Tensor,
        fuse_hadamard: bool,
    ) -> None:
        num_tokens = x.shape[0]
        if num_tokens == 0:
            return
        grid = (num_tokens, triton.cdiv(self.num_heads, self.heads_per_program))
        log = int(math.log2(self.hadamard_order)) if fuse_hadamard else 0
        pre_scale = (
            1.0 / math.sqrt(float(self.hadamard_order))
            if fuse_hadamard
            else 1.0
        )
        store_kvcache_saw_int4_kernel[grid](
            x,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            cache,
            scales_zeros,
            slot_mapping,
            self.num_heads,
            self.head_dim,
            self.head_dim_pad,
            cache.stride(1),
            cache.stride(2),
            cache.stride(3),
            scales_zeros.stride(1),
            scales_zeros.stride(2),
            scales_zeros.stride(3),
            LOG=log,
            PRE_SCALE=pre_scale,
            BLOCK_HALF=triton.next_power_of_2(self.head_dim // 2),
            HEADS_PER_PROGRAM=self.heads_per_program,
            FUSE_HADAMARD=fuse_hadamard,
            ROUND_TO_BF16=fuse_hadamard and self.round_hadamard_to_bf16,
        )

    def store(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        *additional_tensors,
    ):
        if len(additional_tensors) < 3:
            raise ValueError(
                "store() requires k_scales_zeros, v_scales_zeros, metadata"
            )
        k_scales_zeros, v_scales_zeros, _metadata = additional_tensors[:3]

        num_tokens, num_heads, head_dim = key.shape
        assert num_heads == self.num_heads and head_dim == self.head_dim
        assert value.shape == key.shape
        assert key.stride(-1) == 1 and value.stride(-1) == 1
        assert key.stride(1) == head_dim and value.stride(1) == head_dim
        assert slot_mapping.numel() == num_tokens

        key_to_store = (
            key
            if not self.hadamard_enabled or self.fuse_hadamard
            else self._rotate(key)
        )
        value_to_store = (
            value
            if not self.rotate_v or self.fuse_hadamard
            else self._rotate(value)
        )

        self._store_one(
            key_to_store,
            k_cache,
            k_scales_zeros,
            slot_mapping,
            self.hadamard_enabled and self.fuse_hadamard,
        )
        self._store_one(
            value_to_store,
            v_cache,
            v_scales_zeros,
            slot_mapping,
            self.rotate_v and self.fuse_hadamard,
        )

    def retrieve(
        self,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        *additional_tensors,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError(
            "INT4 KV stores packed tensors. Use the INT4 attention backend to "
            "consume or flatten/dequantize the cache."
        )

    def needs_dequantization(self) -> bool:
        return True

    def get_cache_block_size_bytes(self) -> int:
        quant_bytes = 2 * self.block_size * self.num_heads * self.packed_dim
        metadata_bytes = 2 * self.block_size * self.num_heads * 2 * 4
        return quant_bytes + metadata_bytes


@KVCacheRegistry.register("saw_int4")
@KVCacheRegistry.register("int4_bdr")
class SawInt4KVCache(Int4KVCache):
    """BDR-on-INT4 cache; equivalent to ``int4`` with ``HADAMARD=1`` by default."""

    default_hadamard_enabled = True
