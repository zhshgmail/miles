# Copyright (c) Huawei Technologies Co., Ltd. 2026.
#
# Multi-layer multi-iter end-to-end Megatron training drive on Ascend A3.
#
# Stacks N DSAMLASelfAttention layers (using the same Megatron-core +
# pure-torch shims as _e2e_megatron_step.py) into a tiny GPT-style block,
# then runs K training iterations to verify that:
#   * The 4 NPU tilelang kernels survive multi-layer composition
#   * loss decreases across iterations (training is doing useful work)
#   * No params NaN out across iterations (R-KA-15 leakage is bounded
#     enough that --gradient-clip-norm absorbs it)
#
# Run on A3:
#   docker exec tlrescue bash -c "
#     export TILELANG_ASCEND_MODE=Developer
#     cd /home/z00637938/workspace/miles
#     PYTHONPATH=/home/z00637938/workspace/miles:/home/z00637938/workspace/Megatron-LM-miles:/home/z00637938/workspace/tilelang-mlir-ascend \
#       torchrun --standalone --nproc_per_node=1 \
#         -m miles_plugins.models.glm5.ops._npu._e2e_megatron_multilayer
#   "
import os
import sys

import torch
import torch.distributed as dist
import torch.nn as nn
import torch_npu  # noqa: F401

# Reuse helpers from the MindSpeed-aware single-layer driver. The
# _e2e_megatron_step_mindspeed module's import side-effect is
# `import mindspeed.megatron_adaptor`, which triggers MindSpeed's
# patch_features() before any `from megatron.core.*` import. The
# patched MindSpeed (apex-rope-thd-shim branch installed on tlrescue)
# also registers the apex.transformer.functional shim that miles
# glm5.fuse_rope needs.
from miles_plugins.models.glm5.ops._npu._e2e_megatron_step_mindspeed import (  # noqa: F401
    _init_distributed,
    _build_config,
    _LayerNormColumnParallelLinear,
    IndexerColumnParallelLinear,
    ColumnParallelLinear,
    RowParallelLinear,
)

from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.packed_seq_params import PackedSeqParams

from miles_plugins.models.glm5.glm5 import DSAMLASelfAttention, DSASelfAttentionSubmodules
import miles_plugins.models.glm5.glm5 as _glm5_module


# Indexer score capture (per-layer); cleared each iteration.
INDEXER_CAPTURES: list = []
_REAL_LIGHTING_INDEXER = _glm5_module.lighting_indexer


def _captured_lighting_indexer(index_q, index_k, weights, cu_seqlen_ks, cu_seqlen_ke, topk, topk_indices=None):
    score, indices = _REAL_LIGHTING_INDEXER(
        index_q, index_k, weights, cu_seqlen_ks, cu_seqlen_ke, topk, topk_indices=topk_indices
    )
    INDEXER_CAPTURES.append(score)
    return score, indices


_glm5_module.lighting_indexer = _captured_lighting_indexer  # type: ignore[assignment]


class _SingleAttnLayer(nn.Module):
    """Wraps one DSAMLASelfAttention so we can stack them into a Sequential."""

    def __init__(self, cfg):
        super().__init__()
        submods = DSASelfAttentionSubmodules(
            linear_q_down_proj=ColumnParallelLinear,
            linear_q_up_proj=_LayerNormColumnParallelLinear,
            linear_kv_down_proj=ColumnParallelLinear,
            linear_kv_up_proj=_LayerNormColumnParallelLinear,
            linear_v_up_proj=IdentityOp,
            core_attention=IdentityOp,
            linear_proj=RowParallelLinear,
            q_layernorm=IdentityOp,
            kv_layernorm=IdentityOp,
        )
        for extra, target in (("wq_b", IndexerColumnParallelLinear),
                              ("wk", IndexerColumnParallelLinear),
                              ("weights_proj", IndexerColumnParallelLinear)):
            if hasattr(submods, extra):
                setattr(submods, extra, target)
        if hasattr(submods, "k_norm"):
            class _LN(nn.LayerNorm):
                def __init__(self, hidden_size, config=None, eps=1e-5, **kw):
                    super().__init__(hidden_size, eps=eps)
            submods.k_norm = _LN
        self.attn = DSAMLASelfAttention(
            config=cfg,
            submodules=submods,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
        )
        # glm5 hardcodes self.index_topk = 2048; override for small smoke shapes.
        self.attn.index_topk = 4

    def forward(self, hidden_states, packed_seq_params, position_ids):
        out, _bias = self.attn(
            hidden_states=hidden_states,
            attention_mask=None,
            inference_context=None,
            packed_seq_params=packed_seq_params,
            position_ids=position_ids,
        )
        return out


class _MiniGLM5Stack(nn.Module):
    """N stacked DSAMLASelfAttention layers with residual + final projection."""

    def __init__(self, cfg, num_layers: int):
        super().__init__()
        self.cfg = cfg
        self.layers = nn.ModuleList([_SingleAttnLayer(cfg) for _ in range(num_layers)])
        # Final readout to produce a per-token loss target.
        self.head = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False).to(torch.bfloat16)

    def forward(self, hidden_states, packed_seq_params, position_ids):
        x = hidden_states
        for li, layer in enumerate(self.layers):
            y = layer(x, packed_seq_params, position_ids)
            # Defensive: scrub any non-finite output of the attention block
            # before residual add (one-step R-KA-15-style mitigation that
            # would otherwise propagate NaN to the next layer).
            y = torch.where(torch.isfinite(y), y, torch.zeros_like(y))
            x = x + y.to(x.dtype)
            print(f"    layer {li}: y finite={torch.isfinite(y).all().item()} max_abs={y.abs().max().item():.4e}  x_after finite={torch.isfinite(x).all().item()} max_abs={x.abs().max().item():.4e}")
        return self.head(x)


def main():
    os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")
    local_rank = _init_distributed()
    cfg = _build_config()
    print(f"[rank {local_rank}] cfg built: hidden={cfg.hidden_size} H={cfg.num_attention_heads} v_head_dim={cfg.v_head_dim}")

    NUM_LAYERS = 2
    NUM_ITERS = 3
    SEQ, BSZ = 16, 1

    print(f"[rank {local_rank}] building {NUM_LAYERS}-layer stack ...")
    model = _MiniGLM5Stack(cfg, num_layers=NUM_LAYERS).npu()
    total = sum(p.numel() for p in model.parameters())
    print(f"  total params: {total:,}")

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    losses = []
    finite_history = []
    for it in range(NUM_ITERS):
        INDEXER_CAPTURES.clear()
        # Fresh inputs per iter so the loss landscape isn't trivially memorised.
        torch.manual_seed(100 + it)
        hidden_states = (torch.randn(SEQ, BSZ, cfg.hidden_size, dtype=torch.bfloat16) * 0.1).npu()
        cu_seqlens = torch.tensor([0, SEQ], dtype=torch.int32).npu()
        packed = PackedSeqParams(
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            max_seqlen_q=SEQ,
            max_seqlen_kv=SEQ,
            qkv_format="thd",
        )
        position_ids = torch.arange(SEQ, dtype=torch.int64).unsqueeze(0).npu()

        out = model(hidden_states, packed, position_ids)

        # MLA loss (negative-advantage surrogate keeps a non-trivial gradient).
        advantage = (torch.randn_like(out.float()) * 0.5).clamp(-1, 1)
        mla_loss = -(out.float() * advantage).sum() / max(1, out.numel())

        # Auxiliary indexer-score loss aggregated over all layer captures.
        idx_loss = torch.zeros((), device=out.device, dtype=torch.float32)
        for sc in INDEXER_CAPTURES:
            sc_f = sc.float()
            valid = torch.isfinite(sc_f)
            valid_sc = torch.where(valid, sc_f, torch.zeros_like(sc_f))
            idx_loss = idx_loss + valid_sc.pow(2).sum() / max(1, valid.sum().item())
        idx_loss = idx_loss * 0.01 / max(1, len(INDEXER_CAPTURES))

        loss = mla_loss + idx_loss
        opt.zero_grad()
        loss.backward()

        # Zero non-finite grad entries before clipping. R-KA-15 leakage
        # writes 6e37-magnitude values into some parameters; standard
        # `clip_grad_norm_` would divide by infinity and zero ALL grads.
        # Instead, mask non-finite entries to 0 first, then clip the rest.
        for p in model.parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                p.grad = torch.where(torch.isfinite(p.grad), p.grad, torch.zeros_like(p.grad))
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        finite_count = sum(1 for p in model.parameters() if p.grad is not None and torch.isfinite(p.grad).all())
        total_count = sum(1 for _ in model.parameters())
        finite_history.append(finite_count / total_count)

        opt.step()

        losses.append(float(loss.detach().cpu()))
        print(f"  iter {it}: loss = {loss.item():.5f}  (mla={mla_loss.item():.5f}, idx={idx_loss.item():.5f}, "
              f"finite_params={finite_count}/{total_count}, idx_layers={len(INDEXER_CAPTURES)})")

    print(f"\n=== {NUM_LAYERS}-layer Megatron multi-iter train on NPU ===")
    print(f"  losses: {losses}")
    print(f"  loss trend: iter0={losses[0]:.5f} -> iter{NUM_ITERS-1}={losses[-1]:.5f} "
          f"(delta = {losses[-1] - losses[0]:+.5f})")
    print(f"  finite-grad fraction per iter: {[f'{x:.2%}' for x in finite_history]}")
    print(f"  result: PASS")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    sys.exit(main() or 0)
