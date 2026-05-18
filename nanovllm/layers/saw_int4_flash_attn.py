"""INT4/SAW-INT4 attention backend.

Prefill follows the saw-int4 runtime shape adapted to FlashInfer: K/V are first
written to the packed paged cache, then the active pages are flattened and
dequantized into dense ragged K/V for FlashInfer prefill. Decode consumes the
packed INT4 cache directly with Triton; BDR adds the matching Q-side Hadamard
correction when enabled.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import triton
import triton.language as tl

from nanovllm.kvcache.saw_int4 import (
    dequantize_saw_int4_batch,
    hadamard_matrix,
    validate_hadamard_order,
)
from nanovllm.layers.attn_utils import normalize_decode_query
from nanovllm.layers.flash_attn_backend import (
    BaseFlashAttentionBackend,
    FlashAttentionRegistry,
)
from nanovllm.layers.flashinfer_flash_attn import FlashInferAttention


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
def _packed_decode_attention_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    k_scales_zeros_ptr,
    v_scales_zeros_ptr,
    block_table_ptr,
    seq_lens_ptr,
    out_ptr,
    q_stride_batch,
    q_stride_head,
    k_cache_stride_block,
    k_cache_stride_pos,
    k_cache_stride_head,
    v_cache_stride_block,
    v_cache_stride_pos,
    v_cache_stride_head,
    k_sz_stride_block,
    k_sz_stride_pos,
    k_sz_stride_head,
    v_sz_stride_block,
    v_sz_stride_pos,
    v_sz_stride_head,
    block_table_stride_batch,
    block_table_stride_block,
    out_stride_batch,
    out_stride_head,
    max_seq_len,
    kv_group_size,
    softmax_scale,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    block_d: tl.constexpr,
    block_half: tl.constexpr,
    block_k: tl.constexpr,
    LOG: tl.constexpr,
    PRE_SCALE: tl.constexpr,
    HADAMARD_ENABLED: tl.constexpr,
    ROTATE_V: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    kv_head = head_idx // kv_group_size
    seq_len = tl.load(seq_lens_ptr + batch_idx).to(tl.int32)

    half_dim = head_dim // 2
    d_full = tl.arange(0, block_d)
    d_full_mask = d_full < head_dim
    d_half = tl.arange(0, block_half)
    d_half_mask = d_half < half_dim

    q_base = batch_idx * q_stride_batch + head_idx * q_stride_head
    if HADAMARD_ENABLED:
        q = tl.load(q_ptr + q_base + d_full, mask=d_full_mask, other=0.0).to(tl.float32)
        q = _fwht_blocked_segments(q * PRE_SCALE, block_d, LOG)
        q_low = tl.gather(q, tl.where(d_half_mask, d_half, 0), 0)
        q_high = tl.gather(q, tl.where(d_half_mask, d_half + half_dim, 0), 0)
    else:
        q_low = tl.load(
            q_ptr + q_base + d_half,
            mask=d_half_mask,
            other=0.0,
        ).to(tl.float32)
        q_high = tl.load(
            q_ptr + q_base + d_half + half_dim,
            mask=d_half_mask,
            other=0.0,
        ).to(tl.float32)

    m_prev = -float("inf")
    l_prev = 0.0
    acc_low = tl.zeros([block_half], dtype=tl.float32)
    acc_high = tl.zeros([block_half], dtype=tl.float32)

    for start_n in tl.range(0, max_seq_len, block_k):
        kv_idx = start_n + tl.arange(0, block_k)
        kv_mask = kv_idx < seq_len
        page_idx = kv_idx // block_size
        page_off = kv_idx % block_size
        block_num = tl.load(
            block_table_ptr
            + batch_idx * block_table_stride_batch
            + page_idx * block_table_stride_block,
            mask=kv_mask,
            other=0,
        ).to(tl.int64)

        k_base = (
            block_num[:, None] * k_cache_stride_block
            + page_off[:, None].to(tl.int64) * k_cache_stride_pos
            + tl.cast(kv_head, tl.int64) * k_cache_stride_head
        )
        k_packed = tl.load(
            k_cache_ptr + k_base + d_half[None, :],
            mask=kv_mask[:, None] & d_half_mask[None, :],
            other=0,
        )
        k_meta_base = (
            block_num * k_sz_stride_block
            + page_off.to(tl.int64) * k_sz_stride_pos
            + tl.cast(kv_head, tl.int64) * k_sz_stride_head
        )
        k_scale = tl.load(
            k_scales_zeros_ptr + k_meta_base + 0,
            mask=kv_mask,
            other=1.0,
        ).to(tl.float32)
        k_zero = tl.load(
            k_scales_zeros_ptr + k_meta_base + 1,
            mask=kv_mask,
            other=0.0,
        ).to(tl.float32)
        k_low = (
            ((k_packed & 0x0F).to(tl.float32) - k_zero[:, None]) * k_scale[:, None]
        ).to(q_low.dtype)
        k_high = (
            (((k_packed >> 4) & 0x0F).to(tl.float32) - k_zero[:, None])
            * k_scale[:, None]
        ).to(q_low.dtype)
        k_low = tl.where(d_half_mask[None, :], k_low, 0.0)
        k_high = tl.where(d_half_mask[None, :], k_high, 0.0)

        scores = (
            tl.sum(q_low[None, :] * k_low, axis=1)
            + tl.sum(q_high[None, :] * k_high, axis=1)
        ) * softmax_scale
        scores = tl.where(kv_mask, scores, -float("inf"))

        m_curr = tl.max(scores, axis=0)
        m_next = tl.maximum(m_prev, m_curr)
        alpha = tl.exp(m_prev - m_next)
        probs = tl.exp(scores - m_next)

        v_base = (
            block_num[:, None] * v_cache_stride_block
            + page_off[:, None].to(tl.int64) * v_cache_stride_pos
            + tl.cast(kv_head, tl.int64) * v_cache_stride_head
        )
        v_packed = tl.load(
            v_cache_ptr + v_base + d_half[None, :],
            mask=kv_mask[:, None] & d_half_mask[None, :],
            other=0,
        )
        v_meta_base = (
            block_num * v_sz_stride_block
            + page_off.to(tl.int64) * v_sz_stride_pos
            + tl.cast(kv_head, tl.int64) * v_sz_stride_head
        )
        v_scale = tl.load(
            v_scales_zeros_ptr + v_meta_base + 0,
            mask=kv_mask,
            other=1.0,
        ).to(tl.float32)
        v_zero = tl.load(
            v_scales_zeros_ptr + v_meta_base + 1,
            mask=kv_mask,
            other=0.0,
        ).to(tl.float32)
        v_low = ((v_packed & 0x0F).to(tl.float32) - v_zero[:, None]) * v_scale[:, None]
        v_high = (
            ((v_packed >> 4) & 0x0F).to(tl.float32) - v_zero[:, None]
        ) * v_scale[:, None]

        acc_low = acc_low * alpha + tl.sum(probs[:, None] * v_low, axis=0)
        acc_high = acc_high * alpha + tl.sum(probs[:, None] * v_high, axis=0)
        l_prev = l_prev * alpha + tl.sum(probs, axis=0)
        m_prev = m_next

    out_low = acc_low / l_prev
    out_high = acc_high / l_prev
    out_base = batch_idx * out_stride_batch + head_idx * out_stride_head

    if ROTATE_V:
        low_idx = tl.where(d_full < half_dim, d_full, 0)
        high_idx = tl.where((d_full >= half_dim) & d_full_mask, d_full - half_dim, 0)
        out_full = tl.where(
            d_full < half_dim,
            tl.gather(out_low, low_idx, 0),
            tl.gather(out_high, high_idx, 0),
        )
        out_full = _fwht_blocked_segments(out_full * PRE_SCALE, block_d, LOG)
        tl.store(out_ptr + out_base + d_full, out_full, mask=d_full_mask)
    else:
        tl.store(out_ptr + out_base + d_half, out_low, mask=d_half_mask)
        tl.store(out_ptr + out_base + d_half + half_dim, out_high, mask=d_half_mask)


@FlashAttentionRegistry.register("saw_int4")
@FlashAttentionRegistry.register("int4_bdr")
class SawInt4FlashAttention(BaseFlashAttentionBackend):
    """Attention backend for packed INT4 KV cache with optional BDR metadata."""

    def __init__(self):
        self._dense_prefill = FlashInferAttention()
        self._rotation_cache: dict[tuple[int, int, torch.device, torch.dtype], torch.Tensor] = {}

    @property
    def supports_quantized_cache_inputs(self) -> bool:
        return True

    @property
    def requires_paged_prefill_cache(self) -> bool:
        return True

    def _validate_quant_tensors(
        self,
        additional_cache_tensors,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(additional_cache_tensors) < 3:
            raise ValueError(
                "SAW-INT4 attention requires k_scales_zeros, v_scales_zeros, metadata."
            )
        return additional_cache_tensors[:3]

    def _metadata(self, metadata: torch.Tensor) -> tuple[bool, int, bool, int]:
        if metadata.device.type == "cuda" and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("SAW-INT4 metadata cannot be read during graph capture.")
        if metadata.numel() >= 4:
            hadamard_enabled = bool(int(metadata[0].item()))
            hadamard_order = int(metadata[1].item())
            rotate_v = bool(int(metadata[2].item()))
            head_dim = int(metadata[3].item())
        else:
            hadamard_enabled = True
            hadamard_order = int(metadata[0].item())
            rotate_v = bool(int(metadata[1].item()))
            head_dim = int(metadata[2].item())
        return hadamard_enabled, hadamard_order, rotate_v, head_dim

    def _can_fuse_hadamard(self) -> bool:
        return hasattr(tl, "gather")

    def _get_rotation(
        self,
        hadamard_order: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        key = (hadamard_order, head_dim, device, dtype)
        cached = self._rotation_cache.get(key)
        if cached is None:
            block = hadamard_matrix(hadamard_order).to(device=device, dtype=dtype)
            segments = head_dim // hadamard_order
            cached = torch.block_diag(*([block] * segments))
            self._rotation_cache[key] = cached
        return cached

    def _rotate(
        self,
        x: torch.Tensor,
        hadamard_order: int,
    ) -> torch.Tensor:
        head_dim = x.shape[-1]
        validate_hadamard_order(hadamard_order, head_dim)
        rotation = self._get_rotation(hadamard_order, head_dim, x.device, x.dtype)
        return torch.matmul(x, rotation)

    def _prefill_from_paged_cache(
        self,
        q: torch.Tensor,
        scale: float,
        max_seqlen_q: int,
        cu_seqlens_q: torch.Tensor,
        max_seqlen_k: int,
        cu_seqlens_k: torch.Tensor,
        block_table: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        additional_cache_tensors,
    ) -> torch.Tensor:
        k_scales_zeros, v_scales_zeros, metadata = self._validate_quant_tensors(
            additional_cache_tensors
        )
        hadamard_enabled, hadamard_order, rotate_v, head_dim = self._metadata(metadata)
        if hadamard_enabled:
            validate_hadamard_order(hadamard_order, head_dim)

        dense_k, dense_v = dequantize_saw_int4_batch(
            k_cache,
            v_cache,
            block_table,
            cu_seqlens_k,
            k_scales_zeros,
            v_scales_zeros,
            q.dtype,
            max_seqlen_k,
            head_dim=head_dim,
        )
        q_for_prefill = self._rotate(q, hadamard_order) if hadamard_enabled else q
        output = self._dense_prefill.prefill(
            q_for_prefill,
            dense_k,
            dense_v,
            scale,
            max_seqlen_q,
            cu_seqlens_q,
            max_seqlen_k,
            cu_seqlens_k,
        )
        if hadamard_enabled and rotate_v:
            output = self._rotate(output, hadamard_order)
        return output

    def prefill(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        scale: float,
        max_seqlen_q: Optional[int],
        cu_seqlens_q: Optional[torch.Tensor],
        max_seqlen_k: Optional[int],
        cu_seqlens_k: Optional[torch.Tensor],
        block_table: Optional[torch.Tensor] = None,
        *additional_cache_tensors,
    ) -> torch.Tensor:
        if (
            cu_seqlens_q is None
            or cu_seqlens_k is None
            or max_seqlen_q is None
            or max_seqlen_k is None
        ):
            raise ValueError("SAW-INT4 prefill requires varlen metadata.")

        if block_table is None:
            return self._dense_prefill.prefill(
                q,
                k,
                v,
                scale,
                max_seqlen_q,
                cu_seqlens_q,
                max_seqlen_k,
                cu_seqlens_k,
            )

        return self._prefill_from_paged_cache(
            q,
            scale,
            max_seqlen_q,
            cu_seqlens_q,
            max_seqlen_k,
            cu_seqlens_k,
            block_table,
            k,
            v,
            additional_cache_tensors,
        )

    def decode(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        scale: float,
        cache_seqlens: Optional[torch.Tensor],
        block_table: Optional[torch.Tensor],
        *additional_cache_tensors,
    ) -> torch.Tensor:
        if cache_seqlens is None or block_table is None:
            raise ValueError("SAW-INT4 decode requires sequence lengths and block tables.")
        k_scales_zeros, v_scales_zeros, metadata = self._validate_quant_tensors(
            additional_cache_tensors
        )
        hadamard_enabled, hadamard_order, rotate_v, head_dim = self._metadata(metadata)
        q = normalize_decode_query(q)
        batch_size = q.shape[0]
        if batch_size == 0:
            return q.unsqueeze(1)

        seq_lens = cache_seqlens.to(device=q.device, dtype=torch.int32).contiguous()
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            max_seq_len = block_table.shape[1] * k_cache.shape[1]
        else:
            max_seq_len = int(seq_lens.max().item())
        if max_seq_len <= 0:
            return torch.zeros_like(q).unsqueeze(1)

        if hadamard_enabled:
            validate_hadamard_order(hadamard_order, head_dim)
        fuse_hadamard = hadamard_enabled and self._can_fuse_hadamard()
        if hadamard_enabled and not fuse_hadamard:
            q = self._rotate(q, hadamard_order)
        output = torch.empty_like(q)
        block_d = triton.next_power_of_2(head_dim)
        block_half = triton.next_power_of_2(head_dim // 2)
        block_k = 32
        kv_group_size = q.shape[1] // k_cache.shape[2]
        block_table_i32 = block_table.to(device=q.device, dtype=torch.int32).contiguous()
        _packed_decode_attention_kernel[(batch_size, q.shape[1])](
            q,
            k_cache,
            v_cache,
            k_scales_zeros,
            v_scales_zeros,
            block_table_i32,
            seq_lens,
            output,
            q.stride(0),
            q.stride(1),
            k_cache.stride(0),
            k_cache.stride(1),
            k_cache.stride(2),
            v_cache.stride(0),
            v_cache.stride(1),
            v_cache.stride(2),
            k_scales_zeros.stride(0),
            k_scales_zeros.stride(1),
            k_scales_zeros.stride(2),
            v_scales_zeros.stride(0),
            v_scales_zeros.stride(1),
            v_scales_zeros.stride(2),
            block_table_i32.stride(0),
            block_table_i32.stride(1),
            output.stride(0),
            output.stride(1),
            max_seq_len,
            kv_group_size,
            scale,
            head_dim=head_dim,
            block_size=k_cache.shape[1],
            block_d=block_d,
            block_half=block_half,
            block_k=block_k,
            LOG=int(math.log2(hadamard_order)) if fuse_hadamard else 0,
            PRE_SCALE=(
                1.0 / math.sqrt(float(hadamard_order))
                if fuse_hadamard
                else 1.0
            ),
            HADAMARD_ENABLED=fuse_hadamard,
            ROTATE_V=fuse_hadamard and rotate_v,
        )
        if hadamard_enabled and rotate_v and not fuse_hadamard:
            output = self._rotate(output, hadamard_order)
        return output.unsqueeze(1)


@FlashAttentionRegistry.register("int4")
class Int4FlashAttention(SawInt4FlashAttention):
    """Plain INT4 backend; BDR behavior is controlled by cache metadata."""
