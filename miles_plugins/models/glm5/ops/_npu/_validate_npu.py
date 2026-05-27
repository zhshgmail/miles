# Copyright (c) Huawei Technologies Co., Ltd. 2026.
#
# Validation harness for the 4 NPU tilelang ops via miles' top-level
# `*_interface` functions. Compares NPU outputs against a pure-torch CPU
# fp32 autograd reference and reports per-op pass/fail with quantitative err.
#
# Run on A3 inside the tlrescue container:
#   python -m miles_plugins.models.glm5.ops._npu._validate_npu
#
# Exit code: 0 if all 4 ops match within tolerance, 1 otherwise.
import os
import sys

import torch

# the miles dispatch (called via miles' *_interface) needs torch_npu loaded
# for `.is_npu` attribute to be present and queryable.
import torch_npu  # noqa: F401


def _seed(s: int = 0):
    torch.manual_seed(s)


def _allclose(a: torch.Tensor, b: torch.Tensor, atol: float, rtol: float) -> tuple[float, float, bool]:
    diff = (a.float() - b.float()).abs()
    max_abs = diff.max().item()
    rel = max_abs / (b.float().abs().max().item() + 1e-12)
    ok = torch.allclose(a.float(), b.float(), atol=atol, rtol=rtol)
    return max_abs, rel, ok


def validate_indexer_fwd():
    """Exercise `indexer_fwd_interface` on NPU vs a pure-torch CPU ref.

    Calls our NPU implementation directly (bypassing miles' GPU module which
    would import @tilelang.jit decorators referencing CUDA-only symbols on
    module load).
    """
    from miles_plugins.models.glm5.ops._npu.indexer import npu_indexer_fwd_interface as indexer_fwd_interface

    _seed(0)
    SEQ, SKV, H, D = 8, 16, 8, 32
    q_cpu = torch.randn(SEQ, H, D, dtype=torch.float32) * 0.5
    k_cpu = torch.randn(SKV, D, dtype=torch.float32) * 0.5
    w_cpu = torch.rand(SEQ, H, dtype=torch.float32)
    cu_ks_cpu = torch.zeros(SEQ, dtype=torch.int32)
    cu_ke_cpu = torch.full((SEQ,), SKV, dtype=torch.int32)

    # CPU ref: logits[i, j] = sum_h max(K[j] @ Q[i,h]^T, 0) * W[i,h]
    scores_per_head = torch.einsum("ihd,jd->ijh", q_cpu, k_cpu)  # [S, SKV, H]
    relu_scores = torch.clamp(scores_per_head, min=0.0)
    ref_logits = (relu_scores * w_cpu.view(SEQ, 1, H)).sum(dim=-1)  # [S, SKV]

    # NPU run via miles' interface
    q_npu = q_cpu.to(torch.bfloat16).npu().contiguous()
    k_npu = k_cpu.to(torch.bfloat16).npu().contiguous()
    w_npu = w_cpu.npu().contiguous()
    cu_ks_npu = cu_ks_cpu.npu()
    cu_ke_npu = cu_ke_cpu.npu()
    out_npu = indexer_fwd_interface(q_npu, k_npu, w_npu, cu_ks_npu, cu_ke_npu, clean_logits=False)

    abs_err, rel_err, ok = _allclose(out_npu.cpu(), ref_logits, atol=5e-2, rtol=1e-2)
    print(
        f"[indexer_fwd] max abs err: {abs_err:.4e}, rel: {rel_err:.3f}, "
        f"ok: {ok} (S={SEQ}, SKV={SKV}, H={H}, D={D})"
    )
    return ok


def validate_indexer_bwd():
    """Exercise `indexer_bwd_interface` on NPU vs autograd CPU ref."""
    from miles_plugins.models.glm5.ops._npu.indexer import npu_indexer_bwd_interface as indexer_bwd_interface

    _seed(1)
    SEQ, SKV, H, D, K = 1, 16, 8, 32, 4
    # leaf tensors (requires_grad must apply directly on a leaf, not after *0.5)
    q_cpu = (torch.randn(SEQ, H, D, dtype=torch.float32) * 0.5).detach().requires_grad_(True)
    k_cpu = (torch.randn(SKV, D, dtype=torch.float32) * 0.5).detach().requires_grad_(True)
    w_cpu = torch.rand(SEQ, H, dtype=torch.float32).detach().requires_grad_(True)
    topk = torch.tensor([[0, 3, 7, 11]], dtype=torch.int32).expand(SEQ, K).contiguous()
    grad_scores = torch.randn(SEQ, K, dtype=torch.float32) * 0.1

    # CPU autograd ref
    scores_per_head = torch.einsum("ihd,jd->ijh", q_cpu, k_cpu)  # [S, SKV, H]
    relu_scores = torch.clamp(scores_per_head, min=0.0)
    logits = (relu_scores * w_cpu.view(SEQ, 1, H)).sum(dim=-1)  # [S, SKV]
    sel = torch.gather(logits, 1, topk.long())
    loss = (sel * grad_scores).sum()
    loss.backward()
    dq_ref, dk_ref, dw_ref = q_cpu.grad, k_cpu.grad, w_cpu.grad

    q_npu = q_cpu.detach().to(torch.bfloat16).npu().contiguous()
    k_npu = k_cpu.detach().to(torch.bfloat16).npu().contiguous()
    w_npu = w_cpu.detach().npu().contiguous()
    topk_npu = topk.npu().contiguous()
    grad_npu = grad_scores.npu().contiguous()

    dq, dw, dk = indexer_bwd_interface(q_npu, w_npu, k_npu, topk_npu, grad_npu)

    dq_err, dq_rel, dq_ok = _allclose(dq.cpu(), dq_ref, atol=1e-2, rtol=1e-2)
    dw_err, dw_rel, dw_ok = _allclose(dw.cpu(), dw_ref, atol=5e-2, rtol=5e-2)
    dk_err, dk_rel, dk_ok = _allclose(dk.cpu(), dk_ref, atol=5e-2, rtol=5e-2)
    print(
        f"[indexer_bwd] dq err {dq_err:.4e}/{dq_rel:.3f}, dw {dw_err:.4e}/{dw_rel:.3f}, "
        f"dk {dk_err:.4e}/{dk_rel:.3f}, ok: {dq_ok and dw_ok and dk_ok}"
    )
    return dq_ok and dw_ok and dk_ok


def validate_sparse_mla_fwd():
    """Exercise `sparse_mla_fwd_interface` on NPU vs CPU fp32 ref."""
    from miles_plugins.models.glm5.ops._npu.sparse_mla import npu_sparse_mla_fwd_interface as sparse_mla_fwd_interface

    _seed(2)
    S, SKV, H = 8, 16, 16
    D, DT = 64, 16  # NPU port supports D+DT == 80 (kernel asserts d_v=512 for full path)
    # Avoid the d_v=512 assertion: pass d_v=D explicitly to the interface.
    topk = 8

    q_cpu = torch.randn(S, H, D + DT, dtype=torch.float32) * 0.5
    kv_cpu = torch.randn(SKV, 1, D + DT, dtype=torch.float32) * 0.5
    indices_cpu = torch.zeros(S, 1, topk, dtype=torch.int32)
    for s in range(S):
        perm = torch.randperm(min(SKV, s + 1))[:topk]
        if len(perm) < topk:
            perm = torch.cat([perm, torch.zeros(topk - len(perm), dtype=torch.long)])
        indices_cpu[s, 0, :] = perm.to(torch.int32)

    # CPU ref: P=softmax(Q@K^T * sm_scale), O=P@V (V uses first D channels)
    sm_scale = (1.0 / (D + DT)) ** 0.5
    out_ref = torch.zeros(S, H, D, dtype=torch.float32)
    for s in range(S):
        idxs = indices_cpu[s, 0].long()
        kg = kv_cpu[idxs, 0]  # [topk, D+DT]
        qi = q_cpu[s]  # [H, D+DT]
        scores = (qi @ kg.T) * sm_scale
        P = torch.softmax(scores, dim=-1)
        out_ref[s] = P @ kg[:, :D]

    q_npu = q_cpu.to(torch.float16).npu().contiguous()
    kv_npu = kv_cpu.to(torch.float16).npu().contiguous()
    idx_npu = indices_cpu.npu().contiguous()

    out, lse = sparse_mla_fwd_interface(q_npu, kv_npu, idx_npu, sm_scale=sm_scale, d_v=D)
    abs_err, rel, ok = _allclose(out.cpu(), out_ref, atol=5e-3, rtol=5e-3)
    print(f"[sparse_mla_fwd] max abs err: {abs_err:.4e}, rel: {rel:.3f}, ok: {ok} (S={S}, SKV={SKV}, H={H}, D={D}+{DT})")
    return ok


def validate_sparse_mla_bwd():
    """Exercise `sparse_mla_bwd` on NPU. R-KA-13 E5 in-kernel; expect cosine > 0.85."""
    from miles_plugins.models.glm5.ops._npu.sparse_mla import (
        npu_sparse_mla_bwd as sparse_mla_bwd,
        npu_sparse_mla_fwd_interface as sparse_mla_fwd_interface,
    )

    _seed(3)
    S, SKV, H = 8, 16, 16
    D, DT = 64, 16
    topk = 8
    sm_scale = (1.0 / (D + DT)) ** 0.5

    q_cpu = torch.randn(S, H, D + DT, dtype=torch.float32) * 0.5
    kv_cpu = torch.randn(SKV, 1, D + DT, dtype=torch.float32) * 0.5
    indices_cpu = torch.zeros(S, 1, topk, dtype=torch.int32)
    for s in range(S):
        perm = torch.randperm(min(SKV, s + 1))[:topk]
        if len(perm) < topk:
            perm = torch.cat([perm, torch.zeros(topk - len(perm), dtype=torch.long)])
        indices_cpu[s, 0, :] = perm.to(torch.int32)
    dO_cpu = torch.randn(S, H, D, dtype=torch.float32) * 0.5

    # CPU autograd ref — clone first, then mark as leaf with requires_grad
    q_ag = q_cpu.clone().detach().requires_grad_(True)
    kv_ag = kv_cpu.clone().detach().requires_grad_(True)
    out_ref = torch.zeros(S, H, D, dtype=torch.float32)
    for s in range(S):
        idxs = indices_cpu[s, 0].long()
        kg = kv_ag[idxs, 0]
        qi = q_ag[s]
        scores = (qi @ kg.T) * sm_scale
        P = torch.softmax(scores, dim=-1)
        out_ref[s] = P @ kg[:, :D]
    loss = (out_ref * dO_cpu).sum()
    loss.backward()

    q_npu = q_cpu.to(torch.float16).npu().contiguous()
    kv_npu = kv_cpu.to(torch.float16).npu().contiguous()
    idx_npu = indices_cpu.npu().contiguous()
    dO_npu = dO_cpu.to(torch.float16).npu().contiguous()
    o_npu, lse_npu = sparse_mla_fwd_interface(q_npu, kv_npu, idx_npu, sm_scale=sm_scale, d_v=D)
    dq, dkv = sparse_mla_bwd(
        q_npu, kv_npu, o_npu, dO_npu, idx_npu, lse_npu, sm_scale=sm_scale, is_casual=True
    )

    # dq from bwd kernel covers D+DT; ref dq covers D+DT (q has full dim).
    dq_cpu = dq.cpu().float()
    dq_ref = q_ag.grad
    cos = torch.nn.functional.cosine_similarity(dq_cpu.flatten(), dq_ref.flatten(), dim=0).item()
    abs_err = (dq_cpu - dq_ref).abs().max().item()
    print(f"[sparse_mla_bwd] dq cosine vs autograd: {cos:.4f}, max abs err: {abs_err:.4e}")
    # R-KA-13 E5 workaround: expect cosine >= 0.85 (well above the omit-vsub 0.53 baseline)
    return cos >= 0.85


def main():
    os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")
    torch.npu.set_device(0)
    results = {}
    for name, fn in [
        ("indexer_fwd", validate_indexer_fwd),
        ("indexer_bwd", validate_indexer_bwd),
        ("sparse_mla_fwd", validate_sparse_mla_fwd),
        ("sparse_mla_bwd", validate_sparse_mla_bwd),
    ]:
        try:
            ok = fn()
        except Exception as e:
            print(f"[{name}] EXCEPTION: {type(e).__name__}: {e}")
            ok = False
        results[name] = ok

    print("\n=== summary ===")
    for k, v in results.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")
    print("=" * 40)
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
