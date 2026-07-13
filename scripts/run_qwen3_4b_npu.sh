#!/usr/bin/env bash
set -euo pipefail

: "${MILES_RUNTIME_WORKSPACE:?MILES_RUNTIME_WORKSPACE must name the attested source workspace}"
: "${MILES_SCRIPT_TRAIN_BACKEND:?MILES_SCRIPT_TRAIN_BACKEND must be set to megatron}"

export PYTHONPATH="${MILES_RUNTIME_WORKSPACE}/miles:${MILES_RUNTIME_WORKSPACE}/Megatron-Bridge/src:${MILES_RUNTIME_WORKSPACE}/Megatron-LM:${MILES_RUNTIME_WORKSPACE}/MegatronAdaptor:${MILES_RUNTIME_WORKSPACE}/TransformerEngineNPU:${MILES_RUNTIME_WORKSPACE}/mbridge:${MILES_RUNTIME_WORKSPACE}/sglang/python"

exec python3 "${MILES_RUNTIME_WORKSPACE}/miles/scripts/run_qwen3_4b_npu.py"
