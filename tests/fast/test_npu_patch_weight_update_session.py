import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ACTOR = "miles/backends/megatron_utils/actor.py"
SGLANG_ENGINE = "miles/backends/sglang_utils/sglang_engine.py"
PATCH_BASE_BLOBS = {
    ACTOR: "54ca4b81a",
    SGLANG_ENGINE: "ee76b3872",
}


def _apply_patch(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    for relative_path, blob in PATCH_BASE_BLOBS.items():
        destination = source / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        baseline = subprocess.run(
            ["git", "cat-file", "-p", blob],
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout
        destination.write_bytes(baseline)

    subprocess.run(
        [
            "git",
            "apply",
            "--whitespace=error-all",
            "--include",
            ACTOR,
            "--include",
            SGLANG_ENGINE,
            str(ROOT / "docker" / "npu_patch" / "miles.patch"),
        ],
        cwd=source,
        check=True,
        capture_output=True,
        text=True,
    )
    return source


def test_distributed_weight_update_uses_sglang_session_boundary(tmp_path):
    source = _apply_patch(tmp_path)
    actor = (source / ACTOR).read_text(encoding="utf-8")
    engine = (source / SGLANG_ENGINE).read_text(encoding="utf-8")

    assert 'def begin_weight_update(self, selector: str = "target")' in engine
    assert 'return self._make_request("begin_weight_update", {"selector": selector})' in engine
    assert "def end_weight_update(self)" in engine
    assert 'return self._make_request("end_weight_update", {})' in engine

    begin = actor.index("engine.begin_weight_update.remote")
    update = actor.index("self.weight_updater.update_weights()", begin)
    end = actor.index("engine.end_weight_update.remote", update)
    assert begin < update < end


def test_weight_update_session_cleanup_preserves_original_failure(tmp_path):
    source = _apply_patch(tmp_path)
    actor = (source / ACTOR).read_text(encoding="utf-8")
    update_method = actor[actor.index("def update_weights(self)") :]

    assert "except Exception:" in update_method
    assert 'logger.exception("Failed to close SGLang weight-update session")' in update_method
    failure_handler = update_method[
        update_method.index("except Exception:") : update_method.index("else:", update_method.index("except Exception:"))
    ]
    assert "raise" in failure_handler
