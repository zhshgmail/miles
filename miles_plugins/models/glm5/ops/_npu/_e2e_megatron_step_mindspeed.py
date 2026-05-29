# Copyright (c) Huawei Technologies Co., Ltd. 2026.
#
# Megatron-engine-driven training step on Ascend A3 NPU using miles' actual
# `DSAMLASelfAttention` layer (the real GLM-5 sparse-MLA attention block).
#
# What this validates that FSDP smoke didn't:
#   * Megatron-core parallel_state + tensor-parallel wiring
#   * miles' actual production attention class (`DSAMLASelfAttention`)
#   * The 4 NPU tilelang kernels run inside Megatron-managed parameters
#     (ColumnParallelLinear / RowParallelLinear wrappers)
#
# Skips:
#   * TransformerEngine layers (CUDA-only) — substituted with pure
#     megatron.core ColumnParallel/RowParallel + IdentityOp
#   * MindSpeed-LLM (only needed for further Ascend-specific optimisations)
#   * sglang rollout (mock with random hidden states)
#
# Run on A3:
#   docker exec tlrescue bash -c "
#     export TILELANG_ASCEND_MODE=Developer
#     cd /home/z00637938/workspace/miles
#     PYTHONPATH=/home/z00637938/workspace/miles:/home/z00637938/workspace/Megatron-LM-miles:/home/z00637938/workspace/tilelang-mlir-ascend \
#       torchrun --standalone --nproc_per_node=1 \
#         -m miles_plugins.models.glm5.ops._npu._e2e_megatron_step
#   "
import os
import sys

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401

# MindSpeed-driven variant of _e2e_megatron_step.py.
#
# Instead of manually mirroring MindSpeed's cuda->npu / RNG / Stream / apex
# shims (which is what the original _e2e_megatron_step.py does in lines
# 33-130), this variant lets MindSpeed do it through its features_manager:
# import mindspeed.megatron_adaptor runs patch_features() before any
# from megatron.core.* import, which monkey-patches ~430 register_patch
# call-sites covering Megatron-LM's cuda/RNG/Stream/TE/MoE/parallel-init
# paths for Ascend NPU.
#
# Prerequisites in the runtime env:
#   * MindSpeed installed at core_r0.16.0 branch (since miles uses Mcore
#     0.16.0rc0 vendored as Megatron-LM-miles).
#   * No mainline `triton` installed alongside `triton-ascend` -- the two
#     fight over triton/backends/compiler.py. See memory feedback
#     `feedback_triton_vs_triton_ascend_packaging_conflict.md`.
#
# This variant is for empirical validation of T13.B (does miles real-shape
# e2e PASS through the MindSpeed adaptor stack?). The original manual
# monkey-patch driver (_e2e_megatron_step.py) is preserved as a fallback.
import mindspeed.megatron_adaptor  # noqa: F401  -- triggers patch_features()

# T13.C: with the MindSpeed `apex-rope-thd-shim` patch applied
# (mindspeed/features_manager/megatron_basic/requirements_basic.py
# apex_adaptation now registers apex.transformer.functional.
# fused_apply_rotary_pos_emb_thd against dsa_apply_rotary_pos_emb_thd),
# the manual sys.modules shim previously carried here is unnecessary.

from megatron.core import parallel_state
# T13.A confirmed: MindSpeed core_r0.16.0 already provides te_general_gemm
# = None when TE is absent, so the original manual placeholder injection
# (from the non-MindSpeed driver) is not needed here.
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.tensor_parallel.layers import (
    ColumnParallelLinear as _BaseColumnParallelLinear,
    RowParallelLinear as _BaseRowParallelLinear,
)


# Thin TE-kwarg-stripping shims: miles' GLM-5 was authored against
# TransformerEngine's Linear types and passes `parallel_mode="duplicated"`,
# `skip_weight_param_allocation=False`, `tp_comm_buffer_name=...` to the
# submodule constructors. Megatron-core's plain ColumnParallelLinear /
# RowParallelLinear don't recognise those. Swallow them here.
_TE_ONLY_KWARGS = {"parallel_mode", "skip_weight_param_allocation", "tp_comm_buffer_name"}


def _strip_te_kwargs(kwargs):
    return {k: v for k, v in kwargs.items() if k not in _TE_ONLY_KWARGS}


# For q_down_proj / kv_down_proj / linear_proj miles' glm5 explicitly handles
# `ColumnParallelLinear` and `RowParallelLinear` as known types and sets the
# right kwargs itself; reuse the base classes there. For wq_b/wk/weights_proj
# miles unconditionally passes `parallel_mode="duplicated"` etc., so wrap
# the base with kwarg-stripping shims.
ColumnParallelLinear = _BaseColumnParallelLinear
RowParallelLinear = _BaseRowParallelLinear


class IndexerColumnParallelLinear(_BaseColumnParallelLinear):  # noqa: D401
    """TE-kwarg-stripping wrapper for the glm5 indexer-side projections."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **_strip_te_kwargs(kwargs))


class _LayerNormColumnParallelLinear(_BaseColumnParallelLinear):
    """TE's `LayerNormColumnParallelLinear` replacement: fused (RMSNorm or
    LayerNorm) + ColumnParallelLinear that exposes a `layer_norm_weight`
    attribute matching what miles' GLM-5 expects to read at glm5.py:472.

    Megatron's `LinearWithGradAccumulationAndAsyncCommunication` already
    handles the linear half via ColumnParallelLinear; we add a learnable
    per-input-channel norm scale (`layer_norm_weight`) and apply it before
    the linear forward.
    """

    def __init__(self, input_size, output_size, *args, **kwargs):
        # Drop TE-only kwargs.
        kwargs = _strip_te_kwargs(kwargs)
        super().__init__(input_size, output_size, *args, **kwargs)
        # Add a learnable per-channel norm weight (RMSNorm style: just scale,
        # no bias). Matches the shape miles touches at .layer_norm_weight.
        self.layer_norm_weight = torch.nn.Parameter(torch.ones(input_size))
        self._eps = 1e-6

    def forward(self, input_, weight=None):  # type: ignore[override]
        # RMSNorm: input * rsqrt(mean(input^2) + eps) * weight
        x = input_.float()
        rms = x.pow(2).mean(dim=-1, keepdim=True).add_(self._eps).rsqrt_()
        x = (x * rms).to(input_.dtype) * self.layer_norm_weight.to(input_.dtype)
        return super().forward(x, weight=weight)
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import MLATransformerConfig

# Import miles' real attention class (after Megatron-core is importable).
from miles_plugins.models.glm5.glm5 import DSAMLASelfAttention, DSASelfAttentionSubmodules


def _init_distributed():
    if not dist.is_initialized():
        torch.npu.set_device(int(os.environ.get("LOCAL_RANK", "0")))
        dist.init_process_group(backend="hccl")
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
    )
    # Megatron's ColumnParallelLinear / RowParallelLinear init use a custom
    # cuda-rng tracker; seed the model-parallel rng so the .fork() inside
    # _initialize_affine_weight_gpu works on NPU.
    model_parallel_cuda_manual_seed(1234)
    return int(os.environ.get("LOCAL_RANK", "0"))


def _build_config():
    """MLATransformerConfig matching GLM-5 / DeepSeek-V4-Flash shapes.

    Miles' sparse_mla_fwd_interface hardcodes `dim_plus_tail_dim == 576`.
    From the published DeepSeek-V4-Flash HF config, the 576 = head_dim (512)
    + qk_rope_head_dim (64). The "absorbed" Q dim that miles' GLM-5 layer
    feeds to sparse_mla is `kv_lora_rank + qk_pos_emb_head_dim`; for that to
    equal 576 we need `kv_lora_rank == 512`.

    Two presets, switched by `MILES_E2E_SHAPE`:
      * "reduced" (default): H=16 hidden=128 — small Megatron sanity smoke
      * "real": H=64 hidden=512 q_lora_rank=1024 — DSv4-Flash production
        head count; activates the tilelang block_M_inner=16 head-split
        path on the four NPU kernels.
    """
    preset = os.environ.get("MILES_E2E_SHAPE", "reduced")
    if preset not in {"reduced", "real"}:
        raise ValueError(f"MILES_E2E_SHAPE={preset!r}; expected 'reduced' or 'real'")
    if preset == "real":
        hidden = 512
        n_heads = 64
        q_lora = 1024
        ffn_hidden = 1024
    else:
        hidden = 128
        n_heads = 16
        q_lora = 64
        ffn_hidden = 256
    cfg = MLATransformerConfig(
        # core transformer
        num_layers=1,
        hidden_size=hidden,
        num_attention_heads=n_heads,
        ffn_hidden_size=ffn_hidden,
        kv_channels=128,
        # MLA-specific — kv_lora_rank + qk_pos_emb_head_dim must == 576 to
        # satisfy miles' hardcoded dim_plus_tail_dim assertion in
        # `sparse_mla_fwd_interface`.
        q_lora_rank=q_lora,
        kv_lora_rank=512,
        qk_head_dim=128,
        qk_pos_emb_head_dim=64,
        v_head_dim=512,
        # GLM-5 lighting indexer
        # index_num_attention_heads / index_head_dim are NOT MLATransformerConfig
        # fields — they're injected from HF config by get_glm5_spec. We
        # monkey-patch them post-instantiation below.
        # Misc
        rotary_base=10000.0,
        rotary_scaling_factor=1.0,
        rotary_percent=1.0,
        original_max_position_embeddings=2048,
        mscale=1.0,
        mscale_all_dim=1.0,
        beta_fast=32,
        beta_slow=1,
        add_bias_linear=False,
        layernorm_epsilon=1e-5,
        normalization="RMSNorm",  # glm5 asserts RMSNorm; it temporarily swaps to LayerNorm for k_norm
        recompute_granularity=None,
        # init_method / output_layer_init_method default to xavier in parent
        # cf. TransformerConfig __post_init__
    )
    # Inject the indexer-side fields glm5 expects at `config.index_*`.
    # `index_head_dim` must be >= qk_pos_emb_head_dim (glm5 splits it as
    # [index_head_dim - qk_pos_emb_head_dim, qk_pos_emb_head_dim]).
    cfg.index_num_attention_heads = 8
    cfg.index_head_dim = 128  # = 64 (no-pe) + 64 (pe)
    return cfg


def main():
    os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")
    local_rank = _init_distributed()

    cfg = _build_config()
    print(f"[rank {local_rank}] cfg built: hidden={cfg.hidden_size} H={cfg.num_attention_heads} v_head_dim={cfg.v_head_dim}")

    # Build the submodules spec using pure Megatron-core (no TE/Apex).
    # `linear_q_up_proj` and `linear_kv_up_proj` need a layer-norm-fused
    # column-parallel linear to satisfy glm5.py's `.layer_norm_weight` read.
    submods = DSASelfAttentionSubmodules(
        linear_q_down_proj=ColumnParallelLinear,
        linear_q_up_proj=_LayerNormColumnParallelLinear,
        linear_kv_down_proj=ColumnParallelLinear,
        linear_kv_up_proj=_LayerNormColumnParallelLinear,
        linear_v_up_proj=IdentityOp,
        core_attention=IdentityOp,    # we use SparseMLA directly, not the core
        linear_proj=RowParallelLinear,
        q_layernorm=IdentityOp,
        kv_layernorm=IdentityOp,
    )
    # The lighting-indexer-side projections (wq_b/wk/weights_proj) need real
    # parallel-linear weights (glm5.py touches `.weight._skip_gather`). k_norm
    # is a layer norm — use plain torch.nn.LayerNorm (Apex/FusedLayerNorm
    # needs CUDA-only Apex install).
    if hasattr(submods, "wq_b"):
        submods.wq_b = IndexerColumnParallelLinear
    if hasattr(submods, "wk"):
        submods.wk = IndexerColumnParallelLinear
    if hasattr(submods, "weights_proj"):
        submods.weights_proj = IndexerColumnParallelLinear
    if hasattr(submods, "k_norm"):
        # glm5 calls k_norm(submodule, hidden_size=..., config=..., eps=...).
        # torch.nn.LayerNorm doesn't accept config; wrap it to drop config.
        import torch.nn as nn

        class _LN(nn.LayerNorm):
            def __init__(self, hidden_size, config=None, eps=1e-5, **kw):
                super().__init__(hidden_size, eps=eps)

        submods.k_norm = _LN

    print(f"[rank {local_rank}] instantiating DSAMLASelfAttention ...")
    attn = DSAMLASelfAttention(
        config=cfg,
        submodules=submods,
        layer_number=1,
        attn_mask_type=AttnMaskType.causal,
    ).npu()
    print(f"[rank {local_rank}] DSAMLASelfAttention built: {type(attn).__name__}")
    print(f"  params: {sum(p.numel() for p in attn.parameters()):,}")
    # glm5 hardcodes self.index_topk = 2048 in __init__; override per preset.
    # Real DSv4-Flash uses topk=512; "reduced" path uses 4 for sanity smoke.
    _preset = os.environ.get("MILES_E2E_SHAPE", "reduced")
    if _preset == "real":
        attn.index_topk = 512
        SEQ = 2048  # SKV must be >= topk; matches production tilelang smoke
    else:
        attn.index_topk = 4
        SEQ = 16
    BSZ = 1
    hidden_states = (torch.randn(SEQ, BSZ, cfg.hidden_size, dtype=torch.bfloat16) * 0.1).npu()

    # Megatron's PackedSeqParams provides cu_seqlens_{q,kv} + max_seqlen_{q,kv}.
    from megatron.core.packed_seq_params import PackedSeqParams  # noqa: E402

    cu_seqlens = torch.tensor([0, SEQ], dtype=torch.int32).npu()  # noqa: F811 (BSZ=1 contains one packed seq of length SEQ)
    packed = PackedSeqParams(
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        max_seqlen_q=SEQ,
        max_seqlen_kv=SEQ,
        qkv_format="thd",
    )
    position_ids = torch.arange(SEQ, dtype=torch.int64).unsqueeze(0).npu()

    # Capture the indexer's index_score so its gradient flows back into
    # wq_b / wk / k_norm / weights_proj (in miles' production GLM-5 the
    # full model uses index_score in the residual stream; the attention
    # layer alone discards it, hence wrapping the indexer call to also
    # publish index_score for our loss).
    indexer_captures: list = []

    from miles_plugins.models.glm5.ops.indexer import lighting_indexer as _real_lighting_indexer

    def _captured_lighting_indexer(index_q, index_k, weights, cu_seqlen_ks, cu_seqlen_ke, topk, topk_indices=None):
        score, indices = _real_lighting_indexer(
            index_q, index_k, weights, cu_seqlen_ks, cu_seqlen_ke, topk, topk_indices=topk_indices
        )
        indexer_captures.append(score)
        return score, indices

    import miles_plugins.models.glm5.glm5 as _glm5_module
    _glm5_module.lighting_indexer = _captured_lighting_indexer  # type: ignore[assignment]

    print(f"[rank {local_rank}] forward ...")
    out = attn(
        hidden_states=hidden_states,
        attention_mask=None,
        inference_context=None,
        packed_seq_params=packed,
        position_ids=position_ids,
    )
    if isinstance(out, tuple):
        primary = out[0]
        print(f"  out tuple, lens: {[t.shape if hasattr(t,'shape') else type(t) for t in out]}")
    else:
        primary = out
        print(f"  out shape: {primary.shape}")

    print(f"  captured {len(indexer_captures)} indexer scores; shape: {indexer_captures[0].shape if indexer_captures else 'none'}")

    # Backward through the full Megatron-driven attention.
    print(f"[rank {local_rank}] backward ...")
    snap_name, snap_param = next(iter(attn.named_parameters()))
    snap_pre = snap_param.detach().clone()
    opt = torch.optim.Adam(attn.parameters(), lr=1e-3)
    advantage = (torch.randn_like(primary.float()) * 0.5).clamp(-1, 1)
    # Primary loss from MLA output + auxiliary indexer-score loss so the
    # indexer-side parameters (wq_b/wk/k_norm/weights_proj) also receive
    # gradient — mimicking real GLM-5 where index_score feeds the residual.
    mla_loss = -(primary.float() * advantage).sum() / max(1, primary.numel())
    if indexer_captures:
        # The softmax'd indexer score has -inf entries where the topk
        # selection was invalid (cu_seqlens-masked). Mask them out before
        # taking the auxiliary loss; otherwise (-inf)^2 = inf poisons the
        # backward. Use a small coefficient so the aux signal doesn't
        # dominate the MLA gradient.
        idx_score = indexer_captures[0].float()
        valid_mask = torch.isfinite(idx_score)
        valid_score = torch.where(valid_mask, idx_score, torch.zeros_like(idx_score))
        idx_loss = valid_score.pow(2).sum() / max(1, valid_mask.sum().item()) * 0.01
        loss = mla_loss + idx_loss
        print(f"  mla_loss = {mla_loss.item():.5f}, idx_loss = {idx_loss.item():.5f}")
    else:
        loss = mla_loss
    print(f"  loss = {loss.item():.5f}")
    opt.zero_grad()
    loss.backward()

    # Inspect gradients on every trainable param.
    nan = []
    finite_count = 0
    grad_norm_sq = 0.0
    for n, p in attn.named_parameters():
        if p.grad is None:
            print(f"  WARN: no grad for {n}")
            continue
        if not torch.isfinite(p.grad).all():
            nan.append(n)
        else:
            finite_count += 1
            grad_norm_sq += p.grad.float().pow(2).sum().item()
    grad_norm = grad_norm_sq ** 0.5
    print(f"[rank {local_rank}] finite grads on {finite_count} params, non-finite on {len(nan)}, grad_norm={grad_norm:.4e}")
    if nan:
        print(f"  non-finite params: {nan[:5]}{'...' if len(nan) > 5 else ''}")

    opt.step()
    delta = (snap_param.detach() - snap_pre).abs().max().item()
    print(f"[rank {local_rank}] weight delta on '{snap_name}': max_abs={delta:.4e}")
    assert delta > 0, "weights did not change after Megatron-driven optim.step()"

    print(f"\n=== Megatron-driven train-step on NPU ===")
    print(f"  result: PASS")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    sys.exit(main() or 0)
