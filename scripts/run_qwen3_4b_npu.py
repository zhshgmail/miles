"""Ascend NPU first-light launcher for Qwen3-4B."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import os
from pathlib import Path
import shlex


@dataclass(frozen=True)
class LauncherConfig:
    runtime_workspace: Path
    train_backend: str
    num_npus: int
    tensor_parallel_size: int
    model_dir: Path
    data_dir: Path
    model_name: str = "Qwen3-4B-Instruct-2507"

    @property
    def source_roots(self) -> tuple[str, ...]:
        workspace = self.runtime_workspace
        return tuple(
            str(path)
            for path in (
                workspace / "miles",
                workspace / "Megatron-Bridge" / "src",
                workspace / "Megatron-LM",
                workspace / "MegatronAdaptor",
                workspace / "TransformerEngineNPU",
                workspace / "mbridge",
                workspace / "sglang" / "python",
            )
        )

    @property
    def model_path(self) -> Path:
        return self.model_dir / self.model_name

    @property
    def prompt_data_path(self) -> Path:
        return self.data_dir / "dapo-math-17k" / "dapo-math-17k.jsonl"

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> LauncherConfig:
        env = os.environ if environ is None else environ
        workspace = env.get("MILES_RUNTIME_WORKSPACE", "")
        backend = env.get("MILES_SCRIPT_TRAIN_BACKEND", "")
        if not backend:
            raise ValueError("MILES_SCRIPT_TRAIN_BACKEND is required")
        if backend.lower() != "megatron":
            raise ValueError("MILES_SCRIPT_TRAIN_BACKEND must be megatron")
        workspace_path = _absolute_path(workspace, "MILES_RUNTIME_WORKSPACE")
        num_npus = int(env.get("MILES_SCRIPT_NUM_GPUS", "2"))
        tp_size = int(env.get("MILES_SCRIPT_TP_SIZE", str(num_npus)))
        if num_npus < 1:
            raise ValueError("MILES_SCRIPT_NUM_GPUS must be positive")
        if tp_size != num_npus:
            raise ValueError("MILES_SCRIPT_TP_SIZE must equal MILES_SCRIPT_NUM_GPUS")
        model_dir = _absolute_path(env.get("MILES_MODEL_DIR", "/root/models"), "MILES_MODEL_DIR")
        data_dir = _absolute_path(env.get("MILES_DATA_DIR", "/root/datasets"), "MILES_DATA_DIR")
        return cls(
            runtime_workspace=workspace_path,
            train_backend="megatron",
            num_npus=num_npus,
            tensor_parallel_size=tp_size,
            model_dir=model_dir,
            data_dir=data_dir,
        )


def _absolute_path(raw: str, name: str) -> Path:
    path = Path(raw)
    if (
        not path.is_absolute()
        or os.path.normpath(raw) != raw
        or str(path) != raw
        or any(character.isspace() for character in raw)
        or any(character in raw for character in "\0\r\n;&|`$<>\\'\"")
    ):
        raise ValueError(f"{name} must be an absolute normalized path without whitespace")
    return path


def build_train_args(config: LauncherConfig) -> str:
    model = shlex.quote(str(config.model_path))
    prompt_data = shlex.quote(str(config.prompt_data_path))
    return " ".join(
        (
            f"--hf-checkpoint {model}",
            f"--load {model}",
            f"--prompt-data {prompt_data}",
            "--input-key prompt",
            "--label-key label",
            "--apply-chat-template",
            "--rollout-shuffle",
            "--rm-type math",
            "--num-rollout 1",
            "--rollout-batch-size 2",
            "--n-samples-per-prompt 2",
            "--rollout-max-response-len 64",
            "--rollout-temperature 1",
            "--global-batch-size 2",
            "--advantage-estimator grpo",
            "--kl-loss-coef 0.00",
            "--kl-loss-type low_var_kl",
            "--kl-coef 0.00",
            "--entropy-coef 0.00",
            "--eps-clip 0.2",
            "--eps-clip-high 0.28",
            "--optimizer adam",
            "--lr 1e-6",
            "--lr-decay-style constant",
            "--weight-decay 0.1",
            "--adam-beta1 0.9",
            "--adam-beta2 0.98",
            "--rollout-num-gpus-per-engine 1",
            "--sglang-mem-fraction-static 0.6",
            "--sglang-device npu",
            "--sglang-disable-radix-cache",
            "--sglang-chunked-prefill-size 4096",
            "--sglang-max-prefill-tokens 512",
            "--train-backend megatron",
            f"--tensor-model-parallel-size {config.tensor_parallel_size}",
            "--sequence-parallel",
            "--pipeline-model-parallel-size 1",
            "--context-parallel-size 1",
            "--expert-model-parallel-size 1",
            "--expert-tensor-parallel-size 1",
            "--recompute-granularity full",
            "--recompute-method uniform",
            "--recompute-num-layers 1",
            "--attention-dropout 0.0",
            "--hidden-dropout 0.0",
            "--accumulate-allreduce-grads-in-fp32",
            "--attention-softmax-in-fp32",
            "--attention-backend flash",
            "--megatron-to-hf-mode bridge",
            "--actor-num-nodes 1",
            f"--actor-num-gpus-per-node {config.num_npus}",
            f"--rollout-num-gpus {config.num_npus}",
            "--colocate",
            "--check-weight-update-equal",
            "--no-gradient-accumulation-fusion",
            "--use-flash-attn",
        )
    )


def prepare_assets(config: LauncherConfig, utils) -> None:
    model_dir = shlex.quote(str(config.model_dir))
    data_dir = shlex.quote(str(config.data_dir))
    model_path = shlex.quote(str(config.model_path))
    utils.exec_command(f"mkdir -p {model_dir} {data_dir}")
    utils.exec_command(
        f"hf download Qwen/{config.model_name} --local-dir {model_path}"
    )
    utils.hf_download_dataset("zhuzilin/dapo-math-17k", data_dir=str(config.data_dir))


def run(config: LauncherConfig, *, utils, execute_train) -> None:
    prepare_assets(config, utils)
    execute_train(
        train_args=build_train_args(config),
        num_gpus_per_node=config.num_npus,
        megatron_model_type="qwen3-4B-Instruct-2507",
        extra_env_vars={
            "PYTHONPATH": ":".join(config.source_roots),
            "MODEL_ARGS_ROTARY_BASE": "5000000",
        },
        megatron_path=str(config.runtime_workspace / "Megatron-LM"),
    )


def main() -> None:
    from miles.utils.external_utils import command_utils as utils

    run(LauncherConfig.from_env(), utils=utils, execute_train=utils.execute_train_npu)


if __name__ == "__main__":
    main()
