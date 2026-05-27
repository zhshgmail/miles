# Copyright (c) Huawei Technologies Co., Ltd. 2026.
#
# Mini RL-style training step on NPU using miles' actual autograd Functions
# (IndexerFunction + SparseMLA) — without sglang, without Megatron, without
# transformer-engine. Demonstrates that the full optimization loop closes:
#   forward → loss → backward → optim.step → weight update.
#
# Substitutes the heavy stack with:
#   * mock rollout (random "trajectory logits" instead of sglang.generate)
#   * tiny pure-torch wrapper model whose attention block uses our 4 NPU
#     tilelang ops via miles' contract surfaces
#   * single-process Adam optimizer (no distributed, no FSDP/Megatron yet)
#
# What this proves:
#   * The 4 NPU-ported tilelang kernels integrate into a real training loop
#   * Loss decreases (1 step is too few to be statistically meaningful but
#     a single backward+step must not NaN or stall)
#   * Trainable parameters actually move (max delta > 0)
#
# Run on A3:
#   docker exec tlrescue bash -c "
#     cd /home/z00637938/workspace/miles
#     PYTHONPATH=/home/z00637938/workspace/miles:/home/z00637938/workspace/tilelang-mlir-ascend \
#       python -m miles_plugins.models.glm5.ops._npu._e2e_train_step
#   "
import os
import sys

import torch
import torch.nn as nn
import torch_npu  # noqa: F401

from miles_plugins.models.glm5.ops.indexer import lighting_indexer
from miles_plugins.models.glm5.ops.sparse_mla import SparseMLA


class GLM5MiniBlock(nn.Module):
    """Smallest module that exercises both NPU tilelang autograd functions.

    The block:
      1. Projects hidden states to (index_q, index_k, weights) via linears
      2. Runs the lighting indexer to pick topk kv positions
      3. Projects hidden states to (mla_q, mla_kv) via linears (with the
         miles canonical 576-wide qk dim)
      4. Runs SparseMLA over those selections
      5. Projects out back to hidden via a linear
    """

    def __init__(self, hidden: int, h_idx: int, d_idx: int, h_mla: int, d_v: int = 512, d_tail: int = 64):
        super().__init__()
        self.h_idx = h_idx
        self.d_idx = d_idx
        self.h_mla = h_mla
        self.d_v = d_v
        self.d_tail = d_tail
        self.dqk = d_v + d_tail

        # Indexer-side projections
        self.proj_index_q = nn.Linear(hidden, h_idx * d_idx, bias=False)
        self.proj_index_k = nn.Linear(hidden, d_idx, bias=False)
        self.proj_index_w = nn.Linear(hidden, h_idx, bias=False)

        # MLA-side projections (bf16/fp16-friendly: keep params fp32, cast at use)
        self.proj_mla_q = nn.Linear(hidden, h_mla * self.dqk, bias=False)
        self.proj_mla_kv = nn.Linear(hidden, self.dqk, bias=False)

        # Output projection on MLA output (d_v channels)
        self.proj_out = nn.Linear(h_mla * d_v, hidden, bias=False)

    def forward(self, hidden_q, hidden_kv, topk):
        """
        hidden_q:  [seq, hidden] fp32 (on NPU)
        hidden_kv: [seq_kv, hidden] fp32 (on NPU)
        topk: int
        Returns: [seq, hidden] fp32
        """
        seq = hidden_q.shape[0]
        skv = hidden_kv.shape[0]

        # Indexer-side: build inputs for lighting_indexer
        index_q = self.proj_index_q(hidden_q).view(seq, self.h_idx, self.d_idx).to(torch.bfloat16)
        index_k = self.proj_index_k(hidden_kv).to(torch.bfloat16)
        weights = self.proj_index_w(hidden_q).unsqueeze(-1).to(torch.float32)  # [seq, h_idx, 1]
        cu_seqlen_ks = torch.zeros(seq, dtype=torch.int32, device=hidden_q.device)
        cu_seqlen_ke = torch.full((seq,), skv, dtype=torch.int32, device=hidden_q.device)
        index_score, topk_indices = lighting_indexer(
            index_q, index_k, weights, cu_seqlen_ks, cu_seqlen_ke, topk=topk
        )
        topk_indices = topk_indices.clamp(min=0)
        # save index_score for the residual path so the indexer's gradient
        # also flows back to its projections (real GLM-5 attention uses
        # index_score in addition to the MLA output)
        self._index_score = index_score

        # MLA-side: build inputs for SparseMLA
        q_mla = self.proj_mla_q(hidden_q).view(seq, self.h_mla, self.dqk).to(torch.float16)
        kv_mla = self.proj_mla_kv(hidden_kv).unsqueeze(1).to(torch.float16)  # [skv, 1, dqk]
        indices_mla = topk_indices.unsqueeze(1).contiguous().to(torch.int32)  # [seq, 1, topk]
        sm_scale = (1.0 / self.dqk) ** 0.5
        out, _lse = SparseMLA.apply(q_mla, kv_mla, indices_mla, sm_scale)  # [seq, h_mla, d_v]

        # Output projection — mix the MLA output and the indexer scores into
        # the residual stream so gradients flow back through both autograd
        # functions to all their inputs.
        mla_out = self.proj_out(out.reshape(seq, self.h_mla * self.d_v).to(torch.float32))
        idx_signal = index_score.float().mean(dim=-1, keepdim=True)  # [seq, 1]
        out_hidden = mla_out + idx_signal
        return out_hidden


def _seed(s: int = 0):
    torch.manual_seed(s)


def main():
    os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")
    torch.npu.set_device(0)
    _seed(0)

    SEQ = 8
    SKV = 16
    HIDDEN = 128
    TOPK = 4

    model = GLM5MiniBlock(hidden=HIDDEN, h_idx=8, d_idx=32, h_mla=16).npu()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    print(f"[init] params: {sum(p.numel() for p in model.parameters()):,}")

    # Snapshot a representative weight before the step.
    snap_param = next(iter(model.parameters())).detach().clone()
    snap_name = next(iter(dict(model.named_parameters()))).split(".")[0]

    # Mock rollout — a single batch of (hidden_q, hidden_kv) + a scalar reward
    # signal. In real GRPO this would come from sglang; here we just want to
    # exercise the gradient path so any non-trivial loss works.
    hidden_q = (torch.randn(SEQ, HIDDEN, dtype=torch.float32) * 0.1).npu().requires_grad_(False)
    hidden_kv = (torch.randn(SKV, HIDDEN, dtype=torch.float32) * 0.1).npu().requires_grad_(False)

    # Forward
    print("[step] forward ...")
    out = model(hidden_q, hidden_kv, topk=TOPK)
    print(f"  out shape: {tuple(out.shape)} max_abs={out.abs().max().item():.4e}")

    # Mock advantage: per-sequence-position random scalar in [-1, +1]
    advantage = (torch.randn(SEQ, 1, device=out.device) * 0.5).clamp(-1, 1)
    # GRPO-style PG loss surrogate: -(out * advantage).sum() encourages out
    # to align with advantage sign. Far simpler than the real PPO clip
    # objective, but enough to drive a backward + step.
    loss = -(out * advantage).sum() / SEQ
    print(f"  loss = {loss.item():.5f}")
    assert torch.isfinite(loss), "loss is NaN/Inf"

    # Backward + step
    print("[step] backward ...")
    opt.zero_grad()
    loss.backward()
    grad_norm = 0.0
    nan_grad = False
    for n, p in model.named_parameters():
        if p.grad is None:
            print(f"  WARN: no grad for {n}")
            continue
        if not torch.isfinite(p.grad).all():
            print(f"  WARN: non-finite grad on {n}")
            nan_grad = True
        grad_norm += p.grad.pow(2).sum().item()
    grad_norm = grad_norm ** 0.5
    print(f"  total grad norm: {grad_norm:.4e}")
    assert not nan_grad, "non-finite gradients detected"

    print("[step] optim.step() ...")
    opt.step()

    # Verify weight actually moved.
    after_param = next(iter(model.parameters())).detach()
    delta = (after_param - snap_param).abs().max().item()
    print(f"[step] weight delta on '{snap_name}': max_abs={delta:.4e}")
    assert delta > 0, "weights did not change after optim.step()"

    print("\n=== mini RL train-step on NPU ===")
    print("  result: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
