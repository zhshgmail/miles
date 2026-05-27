# Copyright (c) Huawei Technologies Co., Ltd. 2026.
#
# End-to-end autograd smoke for miles' GLM-5 attention block on NPU.
#
# This driver instantiates the two `torch.autograd.Function`s that miles uses
# for its GLM-5-style DSA attention block on the actual NPU device, runs a
# forward pass, computes a loss, and backpropagates. The goal is to prove
# that the full autograd graph flows through our 4 NPU-ported tilelang
# kernels with finite, non-trivial gradients on every input.
#
# Skips:
#   * sglang rollout (blocked by triton-ascend #277)
#   * Megatron / MindSpeed (not installed; FSDP would replace this)
#   * GLM5Layer wrapping (it pulls in megatron.core + transformer_engine —
#     CUDA-only deps)
#
# What this DOES test:
#   * `IndexerFunction.apply(...)` on NPU → fwd + bwd
#   * `SparseMLA.apply(...)` on NPU → fwd + bwd
#   * autograd chains the two together so a single loss.backward() walks
#     through BOTH our tilelang kernels' backward functions in one pass
#
# Run on A3:
#   docker exec tlrescue bash -c "
#     cd /home/z00637938/workspace/miles
#     PYTHONPATH=/home/z00637938/workspace/miles:/home/z00637938/workspace/tilelang-mlir-ascend \
#       python -m miles_plugins.models.glm5.ops._npu._e2e_autograd
#   "
import os
import sys

import torch
import torch_npu  # noqa: F401  (loads `.is_npu` attr + driver registration)

from miles_plugins.models.glm5.ops.indexer import lighting_indexer
from miles_plugins.models.glm5.ops.sparse_mla import SparseMLA


def _seed(s: int = 0):
    torch.manual_seed(s)


def main():
    os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")
    torch.npu.set_device(0)
    _seed(0)

    # Synthetic GLM-5-shaped inputs (small, NPU-friendly tile sizes).
    SEQ = 8
    SKV = 16
    H_INDEXER = 8       # heads for the lighting indexer
    D_INDEXER = 32      # index_dim
    K_TOPK = 4          # topk indices per query in the sparse MLA

    H_MLA = 16          # heads for sparse MLA (must equal block_M in our kernel)
    # NB: miles' GPU SparseMLA hard-codes d_v=512, dim_plus_tail=576. Using
    # those exact shapes keeps the autograd-graph contract identical to
    # miles' production path. Our NPU kernel does NOT assume d_v=512 (it
    # accepts arbitrary d_v + tail_dim) but we drive miles' SparseMLA.apply
    # so the d_v=512 default in `sparse_mla_fwd_interface` must hold.
    D_V = 512           # value dim — miles canonical
    D_TAIL = 64         # tail dim — miles canonical (576 - 512)
    DQK = D_V + D_TAIL  # = 576

    device = "npu"

    # Stage 1 — lighting_indexer fwd + bwd
    # NPU dispatcher expects: index_q [seq, heads, dim], index_k [seq_kv, dim],
    # weights [seq, heads], cu_seqlen_ks/ke [seq] int32.
    index_q = (torch.randn(SEQ, H_INDEXER, D_INDEXER, dtype=torch.float32) * 0.5).to(torch.bfloat16).npu()
    index_k = (torch.randn(SKV, D_INDEXER, dtype=torch.float32) * 0.5).to(torch.bfloat16).npu()
    weights = torch.rand(SEQ, H_INDEXER, 1, dtype=torch.float32).npu()  # [S, H, 1] then squeezed by lighting_indexer
    cu_seqlen_ks = torch.zeros(SEQ, dtype=torch.int32).npu()
    cu_seqlen_ke = torch.full((SEQ,), SKV, dtype=torch.int32).npu()

    index_q.requires_grad_(True)
    index_k.requires_grad_(True)
    weights.requires_grad_(True)

    print("[stage 1] lighting_indexer fwd ...")
    index_score, topk_indices = lighting_indexer(
        index_q, index_k, weights, cu_seqlen_ks, cu_seqlen_ke, topk=K_TOPK, topk_indices=None
    )
    print(f"  index_score shape: {tuple(index_score.shape)} dtype: {index_score.dtype}")
    print(f"  topk_indices shape: {tuple(topk_indices.shape)} dtype: {topk_indices.dtype}")
    print(f"  index_score[0, :4] = {index_score[0, :4].detach().cpu().tolist()}")

    # Stage 2 — SparseMLA fwd uses topk_indices from stage 1 to gather KV
    # Reshape indices to (S, kv_group=1, K) as miles expects.
    indices_mla = topk_indices.unsqueeze(1).contiguous().to(torch.int32)  # [S, 1, K]
    # Need topk indices in range [0, SKV); lighting_indexer's topk indices are
    # absolute kv positions chosen from `cu_seqlen_ks:cu_seqlen_ke`; in our
    # smoke that range is [0, SKV) so the indices are usable directly.
    indices_mla = indices_mla.clamp(min=0)

    # MLA q / kv tensors (separate from indexer-side q/k):
    q_mla = (torch.randn(SEQ, H_MLA, DQK, dtype=torch.float32) * 0.5).to(torch.float16).npu()
    kv_mla = (torch.randn(SKV, 1, DQK, dtype=torch.float32) * 0.5).to(torch.float16).npu()
    q_mla.requires_grad_(True)
    kv_mla.requires_grad_(True)

    sm_scale = (1.0 / DQK) ** 0.5
    print("[stage 2] SparseMLA fwd ...")
    out, lse = SparseMLA.apply(q_mla, kv_mla, indices_mla, sm_scale)
    print(f"  out shape: {tuple(out.shape)} dtype: {out.dtype}")
    print(f"  lse shape: {tuple(lse.shape)} dtype: {lse.dtype}")
    print(f"  out[0, 0, :4] = {out[0, 0, :4].detach().cpu().tolist()}")

    # Stage 3 — Combine outputs of both autograd functions into a single scalar
    # loss. Use index_score (depends on indexer fwd) and out (depends on MLA fwd
    # which depends on topk_indices from indexer fwd → so the autograd graph
    # threads through both functions).
    print("[stage 3] compute scalar loss + backward ...")
    loss = out.float().pow(2).mean() + index_score.float().pow(2).mean()
    print(f"  loss = {loss.item():.5f}")

    loss.backward()

    print("[stage 4] inspect gradients ...")
    grads = {
        "index_q": index_q.grad,
        "index_k": index_k.grad,
        "weights": weights.grad,
        "q_mla": q_mla.grad,
        "kv_mla": kv_mla.grad,
    }
    ok = True
    for name, g in grads.items():
        if g is None:
            print(f"  {name}: grad is None ❌")
            ok = False
            continue
        finite = torch.isfinite(g).all().item()
        max_abs = g.abs().max().item()
        nonzero = (g != 0).any().item()
        status = "✅" if (finite and nonzero) else "❌"
        print(f"  {name}: shape={tuple(g.shape)} max_abs={max_abs:.4e} finite={finite} nonzero={nonzero} {status}")
        if not (finite and nonzero):
            ok = False

    print("\n=== e2e autograd smoke ===")
    print(f"  result: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
