# Copyright (c) Huawei Technologies Co., Ltd. 2026.
#
# NPU dispatch for miles' sparse MLA ops (fwd / bwd).
#
# Contracts (match miles' GPU `*_interface` signatures exactly):
#   * npu_sparse_mla_fwd_interface(q, kv, indices, sm_scale=None,
#                                   return_p_sum=False, d_v=512, ...)
#       -> (out [seq, heads, d_v], lse [seq, heads])
#   * npu_sparse_mla_bwd(q, kv, o, do, indices, lse, sm_scale=None,
#                       is_casual=True, return_kernel=False, delta=None)
#       -> (dq, dkv)
#
# Adaptations vs the GPU path:
#   * Miles' kernel signatures take inputs with an implicit batch=1; we wrap to
#     match our kernel's 4D B=1 layout.
#   * Our bwd uses the R-KA-13 E5 workaround already baked into the kernel.
import torch

from ._sparse_mla_fwd_kernel import sparse_mla_fwd as _npu_sparse_mla_fwd
from ._sparse_mla_bwd_kernel import (
    sparse_mla_bwd_preprocess as _npu_preprocess,
    sparse_mla_bwd_postprocess as _npu_postprocess,
    sparse_mla_bwd_main as _npu_bwd_main,
)


def npu_sparse_mla_fwd_interface(
    q,
    kv,
    indices,
    sm_scale=None,
    return_p_sum: bool = False,
    d_v: int = 512,
    block_I: int = 64,
    num_stages: int = 2,
    threads: int = 256,
):
    """Drop-in for miles' sparse_mla_fwd_interface on NPU.

    Input shapes (miles convention):
        q:       [seq, heads, d_v + tail_dim]  fp16
        kv:      [seq_kv, kv_group, d_v + tail_dim]  fp16
        indices: [seq, kv_group, topk]  int32

    Output (squeezed batch):
        out:  [seq, heads, d_v]  fp16
        lse:  [seq, heads]       fp32
    """
    assert return_p_sum is False, "NPU path supports return_p_sum=False only"
    assert q.is_contiguous() and kv.is_contiguous() and indices.is_contiguous()

    # Inject batch dim to match our kernel's [B, S, H, ...] layout.
    q4 = q.unsqueeze(0)
    kv4 = kv.unsqueeze(0)
    idx4 = indices.unsqueeze(0)

    batch, seq_len, heads, dim_plus_tail = q4.shape
    _, seq_len_kv, kv_group, _ = kv4.shape
    assert kv4.shape[-1] == dim_plus_tail
    assert idx4.shape == (batch, seq_len, kv_group, idx4.shape[-1])
    topk = idx4.shape[-1]
    tail_dim = dim_plus_tail - d_v

    kernel = _npu_sparse_mla_fwd(
        batch=batch,
        seq_len=seq_len,
        seq_len_kv=seq_len_kv,
        heads=heads,
        dim=d_v,
        tail_dim=tail_dim,
        topk=topk,
        block_N=block_I,
        num_stages=num_stages,
    )
    out4, lse4 = kernel(q4, kv4, idx4)
    out = out4.squeeze(0)
    # lse from kernel is [B, S, H, 1]; miles' upstream lse is [B, S, H]
    lse = lse4.squeeze(0).squeeze(-1)
    return out, lse


def npu_sparse_mla_bwd(
    q,
    kv,
    o,
    do,
    indices,
    lse,
    sm_scale=None,
    is_casual: bool = True,
    return_kernel: bool = False,
    delta=None,
):
    """Drop-in for miles' sparse_mla_bwd on NPU.

    Input shapes (miles convention; batch=1 implicit):
        q:       [seq, heads, d_v + tail_dim]
        kv:      [seq_kv, kv_group, d_v + tail_dim]
        o:       [seq, heads, d_v]
        do:      [seq, heads, d_v]
        indices: [seq, kv_group, topk]
        lse:     [seq, heads]
    """
    # Inject batch dim.
    q4 = q.unsqueeze(0).contiguous()
    kv4 = kv.unsqueeze(0).contiguous()
    o4 = o.unsqueeze(0).contiguous()
    do4 = do.unsqueeze(0).contiguous()
    idx4 = indices.unsqueeze(0).contiguous()
    lse4 = lse.unsqueeze(0).unsqueeze(-1).contiguous()  # [B, S, H, 1]

    B, S, H, dim_plus_tail = q4.shape
    _, S_kv, kv_group, _ = kv4.shape
    d_v = 512
    D_tail = dim_plus_tail - d_v
    topk = idx4.shape[-1]

    # preprocess kernel: computes delta = sum_d(O * dO)
    if delta is None:
        preprocess_kernel = _npu_preprocess(B, S, H, d_v)
        # our preprocess output is [B, S, H, 1] (trailing 1 for rank parity)
        delta = preprocess_kernel(o4, do4)
    # main bwd kernel computes dq, accumulates dkv in fp32 via atomic_addx4
    bwd_kernel = _npu_bwd_main(B, S, S_kv, H, d_v, D_tail, topk, kv_group, sm_scale, is_casual)
    dkv = torch.zeros_like(kv4, dtype=torch.float32)
    dq = bwd_kernel(q4, kv4, do4, idx4, lse4, delta, dkv)
    # postprocess cast dkv fp32 -> dtype
    postprocess_kernel = _npu_postprocess(B, S_kv, d_v, D_tail, kv_group)
    dkv = postprocess_kernel(dkv)

    return dq.squeeze(0), dkv.squeeze(0)
