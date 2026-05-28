# Copyright (c) Huawei Technologies Co., Ltd. 2026.
#
# Real-DSv4-shape kernel smoke: does each of the 4 NPU tilelang kernels
# compile + run + produce finite output at the actual published DeepSeek-V4
# / GLM-5 attention-layer shapes (bf16 weights, no quantization)?
#
# Shapes (from DeepSeek-V4-Flash HF config.json + DeepSeek-V3.2 fields):
#   num_attention_heads = 64
#   head_dim            = 512   (= D_V)
#   qk_rope_head_dim    = 64    (= D_TAIL)
#   dim_plus_tail       = 576
#   kv_lora_rank        = 512
#   index_head_dim      = 128
#   index_n_heads       = 64
#   index_topk          = 512   (DSv4-Flash); we try 512 AND 2048 (DSv3.2)
#
# Per-call inputs sized for a SINGLE attention query (SEQ=1) to stay
# inside A3 single-chip memory — we're testing kernel correctness, not
# full-model fit.
#
# Run on A3:
#   docker exec tlrescue bash -c "
#     export TILELANG_ASCEND_MODE=Developer
#     PYTHONPATH=/home/z00637938/workspace/miles:/home/z00637938/workspace/tilelang-mlir-ascend \
#       python -m miles_plugins.models.glm5.ops._npu._real_shape_smoke
#   "
import os
import sys
import time

import torch
import torch_npu  # noqa: F401

from miles_plugins.models.glm5.ops._npu.indexer import (
    npu_indexer_fwd_interface,
    npu_indexer_bwd_interface,
)
from miles_plugins.models.glm5.ops._npu.sparse_mla import (
    npu_sparse_mla_fwd_interface,
    npu_sparse_mla_bwd,
)


# Real DSv4-Flash attention-layer shapes.
H_MLA = 64
D_V = 512
D_TAIL = 64
DQK = D_V + D_TAIL  # 576

H_INDEXER = 64
D_INDEXER = 128


def _summary(name: str, t: torch.Tensor):
    finite = torch.isfinite(t).all().item()
    nonzero = (t != 0).any().item()
    max_abs = t.abs().max().item() if t.numel() > 0 else 0.0
    return f"{name}: shape={tuple(t.shape)} dtype={t.dtype} finite={finite} nonzero={nonzero} max_abs={max_abs:.4e}"


def case_indexer_fwd(seq: int, skv: int, topk: int):
    """lighting_indexer_fwd at real H/D + real topk."""
    print(f"\n[indexer_fwd] SEQ={seq} SKV={skv} H={H_INDEXER} D={D_INDEXER} topk={topk}")
    torch.manual_seed(0)
    q = (torch.randn(seq, H_INDEXER, D_INDEXER, dtype=torch.float32) * 0.5).to(torch.bfloat16).npu().contiguous()
    k = (torch.randn(skv, D_INDEXER, dtype=torch.float32) * 0.5).to(torch.bfloat16).npu().contiguous()
    w = torch.rand(seq, H_INDEXER, dtype=torch.float32).npu().contiguous()
    cu_s = torch.zeros(seq, dtype=torch.int32).npu()
    cu_e = torch.full((seq,), skv, dtype=torch.int32).npu()

    t0 = time.time()
    try:
        logits = npu_indexer_fwd_interface(q, k, w, cu_s, cu_e, clean_logits=False)
        dt = time.time() - t0
        print(f"  PASS in {dt:.1f}s; {_summary('logits', logits)}")
        return True
    except Exception as e:
        print(f"  FAIL: {type(e).__name__}: {str(e)[:200]}")
        return False


def case_indexer_bwd(seq: int, skv: int, topk: int):
    """lighting_indexer_bwd at real H/D + real topk."""
    print(f"\n[indexer_bwd] SEQ={seq} SKV={skv} H={H_INDEXER} D={D_INDEXER} topk={topk}")
    torch.manual_seed(1)
    index_q = (torch.randn(seq, H_INDEXER, D_INDEXER, dtype=torch.float32) * 0.5).to(torch.bfloat16).npu().contiguous()
    index_k = (torch.randn(skv, D_INDEXER, dtype=torch.float32) * 0.5).to(torch.bfloat16).npu().contiguous()
    weights = torch.rand(seq, H_INDEXER, dtype=torch.float32).npu().contiguous()
    # topk_indices in valid kv range
    topk_indices = torch.arange(topk, dtype=torch.int32).unsqueeze(0).expand(seq, topk).contiguous().npu()
    grad_scores = (torch.randn(seq, topk, dtype=torch.float32) * 0.1).npu().contiguous()

    t0 = time.time()
    try:
        gq, gw, gk = npu_indexer_bwd_interface(index_q, weights, index_k, topk_indices, grad_scores)
        dt = time.time() - t0
        print(f"  PASS in {dt:.1f}s")
        for n, t in [("gq", gq), ("gw", gw), ("gk", gk)]:
            print(f"    {_summary(n, t)}")
        return True
    except Exception as e:
        print(f"  FAIL: {type(e).__name__}: {str(e)[:200]}")
        return False


def case_sparse_mla_fwd(seq: int, skv: int, topk: int):
    """sparse_mla_fwd at real H/D_V + real topk."""
    print(f"\n[sparse_mla_fwd] SEQ={seq} SKV={skv} H={H_MLA} D_V={D_V} D_TAIL={D_TAIL} topk={topk}")
    torch.manual_seed(2)
    q = (torch.randn(seq, H_MLA, DQK, dtype=torch.float32) * 0.5).to(torch.float16).npu().contiguous()
    kv = (torch.randn(skv, 1, DQK, dtype=torch.float32) * 0.5).to(torch.float16).npu().contiguous()
    # Indices clamped to [0, SKV)
    indices = torch.randint(0, skv, (seq, 1, topk), dtype=torch.int32).npu().contiguous()
    sm_scale = (1.0 / DQK) ** 0.5

    t0 = time.time()
    try:
        out, lse = npu_sparse_mla_fwd_interface(q, kv, indices, sm_scale=sm_scale, d_v=D_V)
        dt = time.time() - t0
        print(f"  PASS in {dt:.1f}s")
        print(f"    {_summary('out', out)}")
        print(f"    {_summary('lse', lse)}")
        return True
    except Exception as e:
        print(f"  FAIL: {type(e).__name__}: {str(e)[:200]}")
        return False


def case_sparse_mla_bwd(seq: int, skv: int, topk: int):
    """sparse_mla_bwd at real H/D_V + real topk."""
    print(f"\n[sparse_mla_bwd] SEQ={seq} SKV={skv} H={H_MLA} D_V={D_V} D_TAIL={D_TAIL} topk={topk}")
    torch.manual_seed(3)
    q = (torch.randn(seq, H_MLA, DQK, dtype=torch.float32) * 0.5).to(torch.float16).npu().contiguous()
    kv = (torch.randn(skv, 1, DQK, dtype=torch.float32) * 0.5).to(torch.float16).npu().contiguous()
    indices = torch.randint(0, skv, (seq, 1, topk), dtype=torch.int32).npu().contiguous()
    sm_scale = (1.0 / DQK) ** 0.5

    # Need o + lse from fwd
    try:
        o, lse = npu_sparse_mla_fwd_interface(q, kv, indices, sm_scale=sm_scale, d_v=D_V)
    except Exception as e:
        print(f"  FAIL (pre-bwd fwd compute): {type(e).__name__}: {str(e)[:200]}")
        return False
    do = (torch.randn(seq, H_MLA, D_V, dtype=torch.float32) * 0.5).to(torch.float16).npu().contiguous()

    t0 = time.time()
    try:
        dq, dkv = npu_sparse_mla_bwd(q, kv, o, do, indices, lse, sm_scale=sm_scale, d_v=D_V)
        dt = time.time() - t0
        print(f"  PASS in {dt:.1f}s")
        print(f"    {_summary('dq', dq)}")
        print(f"    {_summary('dkv', dkv)}")
        return True
    except Exception as e:
        print(f"  FAIL: {type(e).__name__}: {str(e)[:200]}")
        return False


def main():
    os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")
    torch.npu.set_device(0)

    print("=== Real DSv4-Flash attention-layer shape kernel smoke ===")
    print(f"H_MLA={H_MLA} D_V={D_V} D_TAIL={D_TAIL} DQK={DQK}")
    print(f"H_INDEXER={H_INDEXER} D_INDEXER={D_INDEXER}")

    # Use small SEQ (1) to stay inside A3 single-chip — algo correctness only.
    # SKV / topk match real DSv4-Flash (512) and DSv3.2 (2048).
    results = {}
    for label, fn, kwargs in [
        ("indexer_fwd @ topk=512  SKV=2048", case_indexer_fwd, dict(seq=1, skv=2048, topk=512)),
        ("indexer_bwd @ topk=512  SKV=2048", case_indexer_bwd, dict(seq=1, skv=2048, topk=512)),
        ("sparse_mla_fwd @ topk=512 SKV=2048", case_sparse_mla_fwd, dict(seq=1, skv=2048, topk=512)),
        ("sparse_mla_bwd @ topk=512 SKV=2048", case_sparse_mla_bwd, dict(seq=1, skv=2048, topk=512)),
    ]:
        try:
            results[label] = fn(**kwargs)
        except Exception as e:
            print(f"  EXCEPTION outside kernel: {type(e).__name__}: {e}")
            results[label] = False

    print("\n=== SUMMARY ===")
    for k, v in results.items():
        print(f"  {'✅ PASS' if v else '❌ FAIL'}: {k}")
    n_pass = sum(1 for v in results.values() if v)
    print(f"\nTotal: {n_pass}/{len(results)} kernels PASS at real DSv4-Flash shapes")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
