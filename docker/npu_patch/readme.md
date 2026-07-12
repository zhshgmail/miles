# Miles NPU Patch Installation Guide

This bundle targets the pinned Option B training foundation. The full Miles RL
rollout stack remains a separate release gate and is not qualified by a clean
training-patch replay.

## Version Contract

| Component | Required version or immutable ref | Source |
| --- | --- | --- |
| Python | >=3.10,<3.11 | [Python](https://www.python.org/) |
| CANN | 9.0.0 | [Ascend CANN](https://www.hiascend.com/developer/download/community/result?module=cann) |
| PyTorch | 2.7.1 | [PyTorch](https://pytorch.org/) |
| torch-npu | product 26.0.0; branch v2.7.1-26.0.0; wheel 2.7.1.post4 | [Ascend PyTorch](https://gitcode.com/Ascend/pytorch) |
| torchvision | 0.22.1 | [torchvision](https://github.com/pytorch/vision) |
| Megatron-LM / Mcore | 963bf39218e8bb83a1203b40293358498322be50 (core_r0.17.0) | [NVIDIA Megatron-LM](https://github.com/NVIDIA/Megatron-LM) |
| MegatronAdaptor | 56e18624ec632cf462c079c3873b8fbf2fbd3c77 (core_r0.17.0) | [Ascend MegatronAdaptor](https://gitcode.com/ascend/MegatronAdaptor) |
| TransformerEngineNPU | cecf4a2a3ea7f31afb85cc6669f6b18adc56e5bd | [Ascend TransformerEngineNPU](https://gitcode.com/ascend/TransformerEngineNPU) |
| Megatron-Bridge | 07d61e1547a8356cc34928f7eb20226d2f9db3fa | [radixark Megatron-Bridge](https://github.com/radixark/Megatron-Bridge) |
| Miles | 551d15914c89b1229b76fe806ca5f5aa5a826309 | [radixark Miles](https://github.com/radixark/miles) |
| transformers | 5.6.0 | [Hugging Face transformers](https://github.com/huggingface/transformers) |
| huggingface-hub | 1.23.0 | [Hugging Face Hub](https://github.com/huggingface/huggingface_hub) |

Record the exact Python patch version, image digest, OS, driver, firmware, and
HDK versions as runtime evidence. Do not combine CANN 8.5 or torch-npu 2.8
artifacts with this contract.

## Base Runtime

Create a clean Python 3.10 environment on a CANN 9.0.0 host:

    conda create -n miles_option_b python=3.10
    conda activate miles_option_b
    source <CANN_PATH>/ascend-toolkit/set_env.sh

Install the exact framework packages from the package source approved for the
target host. The torch-npu distribution version is distinct from its 26.0.0
product release:

    pip install torch==2.7.1 torchvision==0.22.1
    pip install torch-npu==2.7.1.post4
    pip install transformers==5.6.0 huggingface-hub==1.23.0

Verify installed metadata and reject any resolver change to these versions.

## Source Installation

Prepare immutable source checkouts:

    mkdir <WORKSPACE> && cd <WORKSPACE>

    git clone https://github.com/NVIDIA/Megatron-LM.git
    git -C Megatron-LM checkout --detach 963bf39218e8bb83a1203b40293358498322be50

    git clone https://gitcode.com/ascend/TransformerEngineNPU.git
    git -C TransformerEngineNPU checkout --detach cecf4a2a3ea7f31afb85cc6669f6b18adc56e5bd

    git clone https://gitcode.com/ascend/MegatronAdaptor.git
    git -C MegatronAdaptor checkout --detach 56e18624ec632cf462c079c3873b8fbf2fbd3c77

    git clone https://github.com/radixark/Megatron-Bridge.git
    git -C Megatron-Bridge checkout --detach 07d61e1547a8356cc34928f7eb20226d2f9db3fa

    git clone https://github.com/radixark/miles.git
    git -C miles checkout --detach 551d15914c89b1229b76fe806ca5f5aa5a826309
    cp -r miles/docker/npu_patch .

Install in dependency order without replacing the pinned framework packages:

    pip install -e <WORKSPACE>/Megatron-LM --no-deps
    pip install -e <WORKSPACE>/TransformerEngineNPU --no-deps
    pip install -e <WORKSPACE>/MegatronAdaptor --no-deps
    pip install -e <WORKSPACE>/Megatron-Bridge --no-deps
    pip install -e <WORKSPACE>/miles --no-deps

MindSpeed is not part of this foundation. Do not install it alongside
MegatronAdaptor and do not restore the removed MindSpeed patch or aliases.

## Applying Patches

    git -C <WORKSPACE>/miles apply <WORKSPACE>/npu_patch/miles.patch
    git -C <WORKSPACE>/Megatron-LM apply <WORKSPACE>/npu_patch/megatron.patch
    git -C <WORKSPACE>/Megatron-Bridge apply <WORKSPACE>/npu_patch/megatron_bridge.patch

The Miles patch imports megatron_adaptor once at the earliest Megatron
bootstrap. There is no supported repatch(args) replacement. The Mcore patch
contains only the NPU tensor-type compatibility gap and its focused test. The
Bridge patch retains the standard Mcore TE class mappings and adapts Mcore
0.17's replicated uneven-DTensor gather back to the plain full tensor expected
by Bridge export.

## Focused Patch Tests

    cd <WORKSPACE>/miles
    MCORE_SOURCE=<WORKSPACE>/Megatron-LM \
      python -m pytest -q -o addopts='' tests/test_npu_patch_megatron_adaptor.py
    python -m pytest -q -o addopts='' tests/test_npu_patch_runtime_env.py

    cd <WORKSPACE>/Megatron-LM
    python -m pytest -q -o addopts='' \
      tests/unit_tests/transformer/test_npu_float16_module_types.py

    cd <WORKSPACE>/Megatron-Bridge
    MCORE_SOURCE=<WORKSPACE>/Megatron-LM \
    PYTHONPATH=<WORKSPACE>/Megatron-Bridge/src:<WORKSPACE>/Megatron-LM \
      python -m pytest -q -o addopts='' \
      --confcutdir=tests/unit_tests/models \
      tests/unit_tests/models/test_option_b_mcore_compat.py

The focused tests are patch-replay checks, not a substitute for a
dependency-complete import, NPU operation, reduced training, or Miles
weight-synchronization gate.

## Capability Boundaries

- FlashInfer is not patched. Mcore already marks it unavailable when import
  fails, and no Option B runtime evidence demonstrated an incompatible import
  that requires a source override.
- Paged Stashing is unsupported on Mcore 0.17. Bridge detects its absence and
  disables the optional path; this bundle does not backport it.
- SGLang, SGL Kernel NPU, Ray, rollout assets, and sglang.patch remain in the
  full RL rollout gate. Their presence in the repository does not make the
  training foundation or the full rollout release complete.
