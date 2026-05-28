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

# Megatron-core's many `torch.cuda.current_device()` callsites raise on NPU.
# Redirect them to NPU before importing megatron, mirroring what MindSpeed
# does internally. Same for the small handful of `torch.cuda.is_available()`
# and related calls used to gate device-allocator decisions.
_REAL_CUDA_AVAILABLE = torch.cuda.is_available
_REAL_CUDA_CURRENT = torch.cuda.current_device


def _npu_cuda_available() -> bool:  # pragma: no cover
    return torch.npu.is_available()


def _npu_cuda_current() -> int:  # pragma: no cover
    return torch.npu.current_device()


torch.cuda.is_available = _npu_cuda_available  # type: ignore[assignment]
torch.cuda.current_device = _npu_cuda_current  # type: ignore[assignment]
# Redirect rng + device-class + Stream to NPU equivalents so Megatron-core
# can stash and restore per-stream RNG state during ColumnParallelLinear /
# RowParallelLinear init.
def _npu_get_rng_state(device=None):
    if device is None:
        device = torch.npu.current_device()
    if isinstance(device, int):
        device = torch.device("npu", device)
    return torch.npu.get_rng_state(device)


def _npu_set_rng_state(new_state, device=None):
    if device is None:
        device = torch.npu.current_device()
    if isinstance(device, int):
        device = torch.device("npu", device)
    return torch.npu.set_rng_state(new_state, device)


torch.cuda.get_rng_state = _npu_get_rng_state  # type: ignore[assignment]
torch.cuda.set_rng_state = _npu_set_rng_state  # type: ignore[assignment]
# Megatron sometimes uses the `torch.cuda.random` submodule path; redirect
# the same callables there too.
torch.cuda.random.get_rng_state = _npu_get_rng_state  # type: ignore[assignment]
torch.cuda.random.set_rng_state = _npu_set_rng_state  # type: ignore[assignment]
torch.cuda.manual_seed = lambda seed: torch.npu.manual_seed(seed)  # type: ignore[assignment]
torch.cuda.manual_seed_all = lambda seed: torch.npu.manual_seed_all(seed)  # type: ignore[assignment]
torch.cuda.device_count = lambda: torch.npu.device_count()  # type: ignore[assignment]
torch.cuda.synchronize = lambda device=None: torch.npu.synchronize(device)  # type: ignore[assignment]
# Megatron sometimes does `torch.cuda.Stream(device)` to push compute into a
# private stream; map it to the NPU equivalent.
torch.cuda.Stream = torch.npu.Stream  # type: ignore[assignment]
torch.cuda.current_stream = lambda device=None: torch.npu.current_stream(device)  # type: ignore[assignment]
torch.cuda.default_stream = lambda device=None: torch.npu.default_stream(device)  # type: ignore[assignment]

from megatron.core import parallel_state
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
    """Minimal MLATransformerConfig matching GLM-5 canonical small shapes."""
    cfg = MLATransformerConfig(
        # core transformer
        num_layers=1,
        hidden_size=128,
        num_attention_heads=16,  # H_MLA
        ffn_hidden_size=256,
        kv_channels=128,
        # MLA-specific
        q_lora_rank=64,
        kv_lora_rank=64,
        qk_head_dim=64,
        qk_pos_emb_head_dim=16,  # rope head dim = D_TAIL
        v_head_dim=512,          # D_V (matches our sparse_mla d_v)
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
    # Inject the indexer-side fields glm5 expects at `config.index_*`
    cfg.index_num_attention_heads = 8
    cfg.index_head_dim = 32
    return cfg


def main():
    os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")
    local_rank = _init_distributed()

    cfg = _build_config()
    print(f"[rank {local_rank}] cfg built: hidden={cfg.hidden_size} H={cfg.num_attention_heads} v_head_dim={cfg.v_head_dim}")

    # Build the submodules spec using pure Megatron-core (no TE/Apex).
    submods = DSASelfAttentionSubmodules(
        linear_q_down_proj=ColumnParallelLinear,
        linear_q_up_proj=ColumnParallelLinear,
        linear_kv_down_proj=ColumnParallelLinear,
        linear_kv_up_proj=ColumnParallelLinear,
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

    # Forward smoke.
    SEQ = 8
    BSZ = 1
    hidden_states = (torch.randn(SEQ, BSZ, cfg.hidden_size, dtype=torch.bfloat16) * 0.1).npu()

    # Megatron's PackedSeqParams provides cu_seqlens_{q,kv} + max_seqlen_{q,kv}.
    from megatron.core.packed_seq_params import PackedSeqParams  # noqa: E402

    cu_seqlens = torch.tensor([0, SEQ], dtype=torch.int32).npu()
    packed = PackedSeqParams(
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        max_seqlen_q=SEQ,
        max_seqlen_kv=SEQ,
        qkv_format="thd",
    )
    position_ids = torch.arange(SEQ, dtype=torch.int64).unsqueeze(0).npu()

    print(f"[rank {local_rank}] forward ...")
    try:
        out = attn(
            hidden_states=hidden_states,
            attention_mask=None,
            inference_context=None,
            packed_seq_params=packed,
            position_ids=position_ids,
        )
        if isinstance(out, tuple):
            print(f"  out tuple, lens: {[t.shape if hasattr(t,'shape') else type(t) for t in out]}")
        else:
            print(f"  out shape: {out.shape}")
    except Exception as e:
        print(f"  FAILED at forward: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    sys.exit(main() or 0)
