from pathlib import Path
import runpy
import sys
from types import ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _launcher_namespace():
    return runpy.run_path(str(ROOT / "scripts" / "run_qwen3_4b_npu.py"))


def test_npu_patch_supplies_qwen3_launcher_entrypoints():
    assert (ROOT / "scripts" / "run_qwen3_4b_npu.py").is_file()
    assert (ROOT / "scripts" / "run_qwen3_4b_npu.sh").is_file()


def test_main_uses_command_utils_for_asset_preparation_and_npu_execution(monkeypatch):
    namespace = _launcher_namespace()
    config_type = namespace["LauncherConfig"]
    config = config_type.from_env(
        {
            "MILES_RUNTIME_WORKSPACE": "/workspace/autoport/miles-runtime",
            "MILES_SCRIPT_TRAIN_BACKEND": "megatron",
        }
    )
    events = []
    fake_utils = ModuleType("miles.utils.external_utils.command_utils")
    fake_utils.exec_command = lambda command: events.append(("command", command))
    fake_utils.hf_download_dataset = lambda dataset, data_dir: events.append(
        ("dataset", dataset, data_dir)
    )
    fake_utils.execute_train_npu = lambda **kwargs: events.append(("launch", kwargs))

    import miles.utils.external_utils as external_utils

    monkeypatch.setattr(external_utils, "command_utils", fake_utils, raising=False)
    monkeypatch.setitem(sys.modules, "miles.utils.external_utils.command_utils", fake_utils)
    monkeypatch.setattr(config_type, "from_env", classmethod(lambda cls: config))

    namespace["main"]()

    assert [event[0] for event in events] == ["command", "command", "dataset", "launch"]
    assert events[-1][1]["extra_env_vars"]["PYTHONPATH"] == ":".join(config.source_roots)


def test_shell_dispatches_the_npu_launcher_from_the_proven_runtime_workspace():
    text = (ROOT / "scripts" / "run_qwen3_4b_npu.sh").read_text(encoding="utf-8")
    workspace = "/workspace/autoport/miles-runtime"
    config = _launcher_namespace()["LauncherConfig"].from_env(
        {
            "MILES_RUNTIME_WORKSPACE": workspace,
            "MILES_SCRIPT_TRAIN_BACKEND": "megatron",
        }
    )

    assert 'MILES_RUNTIME_WORKSPACE' in text
    assert '"${MILES_RUNTIME_WORKSPACE}/miles/scripts/run_qwen3_4b_npu.py"' in text
    assert "run_qwen3_4b.py" not in text
    assert "${PYTHONPATH" not in text
    pythonpath_line = next(line for line in text.splitlines() if line.startswith("export PYTHONPATH="))
    shell_pythonpath = pythonpath_line.removeprefix('export PYTHONPATH="').removesuffix('"')
    shell_roots = tuple(shell_pythonpath.replace("${MILES_RUNTIME_WORKSPACE}", workspace).split(":"))
    assert shell_roots == config.source_roots
    assert len(shell_roots) == len(set(shell_roots)) == 7


def test_launcher_requires_canonical_megatron_backend_and_ignores_legacy_prefix():
    config_type = _launcher_namespace()["LauncherConfig"]
    env = {
        "MILES_RUNTIME_WORKSPACE": "/workspace/autoport/miles-runtime",
        "SLIME_SCRIPT_TRAIN_BACKEND": "megatron",
    }

    with pytest.raises(ValueError, match="MILES_SCRIPT_TRAIN_BACKEND"):
        config_type.from_env(env)

    env["MILES_SCRIPT_TRAIN_BACKEND"] = "fsdp"
    with pytest.raises(ValueError, match="must be megatron"):
        config_type.from_env(env)


def test_launcher_derives_two_npu_topology_and_all_source_roots_from_one_workspace():
    config_type = _launcher_namespace()["LauncherConfig"]
    config = config_type.from_env(
        {
            "MILES_RUNTIME_WORKSPACE": "/workspace/autoport/miles-runtime",
            "MILES_SCRIPT_TRAIN_BACKEND": "megatron",
        }
    )

    assert config.num_npus == 2
    assert config.tensor_parallel_size == 2
    assert config.source_roots == (
        "/workspace/autoport/miles-runtime/miles",
        "/workspace/autoport/miles-runtime/Megatron-Bridge/src",
        "/workspace/autoport/miles-runtime/Megatron-LM",
        "/workspace/autoport/miles-runtime/MegatronAdaptor",
        "/workspace/autoport/miles-runtime/TransformerEngineNPU",
        "/workspace/autoport/miles-runtime/mbridge",
        "/workspace/autoport/miles-runtime/sglang/python",
    )
    assert all(not root.startswith("/root/") for root in config.source_roots)
    assert all("sgl-kernel" not in root for root in config.source_roots)


def test_launcher_rejects_relative_workspace_and_topology_drift():
    config_type = _launcher_namespace()["LauncherConfig"]
    base = {
        "MILES_RUNTIME_WORKSPACE": "relative/workspace",
        "MILES_SCRIPT_TRAIN_BACKEND": "megatron",
    }
    with pytest.raises(ValueError, match="absolute normalized"):
        config_type.from_env(base)

    base["MILES_RUNTIME_WORKSPACE"] = "/workspace/../unattested"
    with pytest.raises(ValueError, match="absolute normalized"):
        config_type.from_env(base)

    base["MILES_RUNTIME_WORKSPACE"] = "/workspace/autoport/miles-runtime"
    base["MILES_SCRIPT_NUM_GPUS"] = "2"
    base["MILES_SCRIPT_TP_SIZE"] = "1"
    with pytest.raises(ValueError, match="must equal"):
        config_type.from_env(base)


def test_launcher_rejects_shell_metacharacters_in_asset_paths():
    config_type = _launcher_namespace()["LauncherConfig"]
    base = {
        "MILES_RUNTIME_WORKSPACE": "/workspace/autoport/miles-runtime",
        "MILES_SCRIPT_TRAIN_BACKEND": "megatron",
    }

    for field in ("MILES_MODEL_DIR", "MILES_DATA_DIR"):
        env = dict(base)
        env[field] = "/assets/owned;touch-pwned"
        with pytest.raises(ValueError, match="absolute normalized"):
            config_type.from_env(env)


def test_launcher_uses_one_model_and_dataset_contract_for_prepare_and_execute():
    namespace = _launcher_namespace()
    config = namespace["LauncherConfig"].from_env(
        {
            "MILES_RUNTIME_WORKSPACE": "/workspace/autoport/miles-runtime",
            "MILES_SCRIPT_TRAIN_BACKEND": "megatron",
            "MILES_MODEL_DIR": "/assets/models",
            "MILES_DATA_DIR": "/assets/datasets",
        }
    )
    args = namespace["build_train_args"](config)

    assert config.model_path == Path("/assets/models/Qwen3-4B-Instruct-2507")
    assert config.prompt_data_path == Path("/assets/datasets/dapo-math-17k/dapo-math-17k.jsonl")
    assert f"--hf-checkpoint {config.model_path}" in args
    assert f"--load {config.model_path}" in args
    assert f"--prompt-data {config.prompt_data_path}" in args
    assert "/root/model/" not in args
    assert "geo3k" not in args


def test_first_light_args_enable_real_compare_and_two_npu_colocation():
    namespace = _launcher_namespace()
    config = namespace["LauncherConfig"].from_env(
        {
            "MILES_RUNTIME_WORKSPACE": "/workspace/autoport/miles-runtime",
            "MILES_SCRIPT_TRAIN_BACKEND": "megatron",
        }
    )
    args = namespace["build_train_args"](config)

    assert "--check-weight-update-equal" in args
    assert "--colocate" in args
    assert "--actor-num-gpus-per-node 2" in args
    assert "--rollout-num-gpus 2" in args
    assert "--tensor-model-parallel-size 2" in args
    assert "--megatron-to-hf-mode bridge" in args
    assert "--num-rollout 1" in args
    assert "--train-backend fsdp" not in args


def test_run_prepares_matching_assets_before_launching_with_full_worker_pythonpath():
    namespace = _launcher_namespace()
    config = namespace["LauncherConfig"].from_env(
        {
            "MILES_RUNTIME_WORKSPACE": "/workspace/autoport/miles-runtime",
            "MILES_SCRIPT_TRAIN_BACKEND": "megatron",
            "MILES_MODEL_DIR": "/assets/models",
            "MILES_DATA_DIR": "/assets/datasets",
        }
    )
    events = []

    class FakeUtils:
        @staticmethod
        def exec_command(command):
            events.append(("command", command))

        @staticmethod
        def hf_download_dataset(dataset, data_dir):
            events.append(("dataset", dataset, data_dir))

    def fake_execute_train_npu(**kwargs):
        events.append(("launch", kwargs))

    namespace["run"](config, utils=FakeUtils, execute_train=fake_execute_train_npu)

    assert events[0] == ("command", "mkdir -p /assets/models /assets/datasets")
    assert events[1] == (
        "command",
        "hf download Qwen/Qwen3-4B-Instruct-2507 --local-dir /assets/models/Qwen3-4B-Instruct-2507",
    )
    assert events[2] == ("dataset", "zhuzilin/dapo-math-17k", "/assets/datasets")
    launch = events[3][1]
    assert launch["megatron_path"] == "/workspace/autoport/miles-runtime/Megatron-LM"
    assert launch["extra_env_vars"]["PYTHONPATH"] == ":".join(config.source_roots)
    assert f"--prompt-data {config.prompt_data_path}" in launch["train_args"]
