# Copyright (c) Huawei Technologies Co., Ltd. 2026.
#
# NPU dispatch for miles' indexer ops (lighting indexer fwd / bwd).
#
# Contracts (match miles' GPU `*_interface` signatures exactly):
#   * npu_indexer_fwd_interface(q, kv, weights, cu_seqlen_ks, cu_seqlen_ke,
#                               clean_logits=True) -> logits [seq, seq_kv] fp32
#   * npu_indexer_bwd_interface(index_q, weights, index_k, topk_indices,
#                               grad_scores) -> (grad_q, grad_w, grad_k)
#
# Under the hood:
#   * Calls our mlir-ascend kernels in `_lighting_indexer_{fwd,bwd}_kernel`
#   * Masks invalid positions in the fwd via the cu_seqlen_* arrays (the
#     kernel itself doesn't implement varlen masking)
#   * Loops the bwd per-seq-position to dodge R-KA-14 (multi-block scatter NaN)
#   * Short-circuits the bwd atomic_addx4 scatter when grad_scores is all-zero
#     (R-KA-15 wrapper guard)
import torch

from ._lighting_indexer_fwd_kernel import lighting_indexer_fwd
from ._lighting_indexer_bwd_kernel import lighting_indexer_bwd


def _apply_clean_logits(logits: torch.Tensor, cu_seqlen_ks: torch.Tensor, cu_seqlen_ke: torch.Tensor) -> None:
    """Mask invalid kv positions per-query to -inf, in place.

    Mirrors the upstream `clean_logits_kernel`: for each query row `bx`, kv index
    `idx` outside `[cu_seqlen_ks[bx], cu_seqlen_ke[bx])` is set to -inf.
    """
    seq_len, seq_len_kv = logits.shape
    device = logits.device
    kv_idx = torch.arange(seq_len_kv, device=device)
    # broadcast: starts/ends shape [seq_len, 1]; kv_idx shape [seq_len_kv]
    starts = cu_seqlen_ks.view(seq_len, 1)
    ends = cu_seqlen_ke.view(seq_len, 1)
    mask = (kv_idx.view(1, seq_len_kv) >= starts) & (kv_idx.view(1, seq_len_kv) < ends)
    logits.masked_fill_(~mask, float("-inf"))


def _largest_pow2_divisor(n: int, cap: int) -> int:
    """Largest power-of-2 that divides n, capped at `cap`."""
    if n <= 0:
        return 1
    pw = 1
    while pw * 2 <= min(n, cap) and n % (pw * 2) == 0:
        pw *= 2
    return pw


def npu_indexer_fwd_interface(q, kv, weights, cu_seqlen_ks, cu_seqlen_ke, clean_logits=True):
    """Drop-in for miles' indexer_fwd_interface on NPU."""
    seq_len, heads, index_dim = q.shape
    seq_len_kv = kv.shape[0]

    # Kernel expects bf16 IndexQ/IndexK, fp32 Weights/Logits; ensure contiguous.
    q = q.contiguous()
    kv = kv.contiguous()
    weights = weights.contiguous()

    # Pick block_N / block_Q that divide the input dims (kernel asserts).
    block_N = _largest_pow2_divisor(seq_len_kv, cap=64)
    block_Q = _largest_pow2_divisor(seq_len, cap=max(1, 128 // heads))

    kernel = lighting_indexer_fwd(seq_len, seq_len_kv, heads, index_dim, block_N=block_N, block_Q=block_Q)
    logits = kernel(q.view(seq_len * heads, index_dim), kv, weights)

    if clean_logits:
        _apply_clean_logits(logits, cu_seqlen_ks, cu_seqlen_ke)
    return logits


def npu_indexer_bwd_interface(index_q, weights, index_k, topk_indices, grad_scores):
    """Drop-in for miles' indexer_bwd_interface on NPU.

    Loops per seq-position to avoid R-KA-14 (multi-block atomic scatter NaN);
    short-circuits at R-KA-15 (all-zero grad_scores) when applicable.
    """
    seq_len, head_num, head_dim = index_q.shape
    seq_len_kv = index_k.shape[0]
    k_top = topk_indices.shape[1]

    grad_scores = grad_scores.contiguous()
    grad_q = torch.zeros_like(index_q)
    grad_w = torch.zeros_like(weights, dtype=torch.float32)
    grad_k = torch.zeros_like(index_k, dtype=torch.float32)

    # R-KA-15 short-circuit: if grad_scores is effectively zero, the atomic_addx4
    # path would write 6e37 garbage instead of a no-op. Returning zeros is the
    # mathematically correct result.
    if grad_scores.abs().max().item() < 1e-30:
        return grad_q, grad_w, grad_k

    # R-KA-14 work-around: per-seq-position SEQ=1 call.
    block_I = _largest_pow2_divisor(k_top, cap=32)
    kernel = lighting_indexer_bwd(
        seq_len=1, seq_len_kv=seq_len_kv, heads=head_num, index_dim=head_dim, topk=k_top, block_I=block_I
    )

    # weights shape is [seq, heads] (miles squeezed in caller); ensure 2D.
    if weights.ndim == 3:
        weights = weights.squeeze(-1)

    for s in range(seq_len):
        q_row = index_q[s : s + 1].contiguous()  # [1, H, D]
        w_row = weights[s : s + 1].contiguous().to(torch.float32)  # [1, H]
        idx_row = topk_indices[s : s + 1].contiguous().to(torch.int32)  # [1, k_top]
        grad_row = grad_scores[s : s + 1].contiguous().to(torch.float32)  # [1, k_top]

        dq_row = torch.zeros_like(q_row)
        dw_row = torch.zeros_like(w_row)
        dk_local = torch.zeros_like(grad_k)  # fp32 accumulator

        kernel(q_row, index_k, w_row, idx_row, grad_row, dq_row, dw_row, dk_local)

        grad_q[s : s + 1] = dq_row
        grad_w[s : s + 1] = dw_row
        grad_k.add_(dk_local)

    return grad_q, grad_w, grad_k
