import io
import subprocess
import sys
import tarfile
import textwrap
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MILES_BASE = "551d15914c89b1229b76fe806ca5f5aa5a826309"
PATCH_COMMIT = "1e7764421cf964e388f1d835210a4af037313448"
PATCH_SHA256 = "350fb1bad15ab3ae6a941477dffafca4d9e1db2b3f834462ac6ed63a2dd3c23d"
PROCESSORS = "miles/backends/megatron_utils/megatron_to_hf/processors"


@pytest.fixture(scope="module")
def patched_tree(tmp_path_factory):
    source = tmp_path_factory.mktemp("miles-lazy-quantizers") / "source"
    source.mkdir()
    archive = subprocess.run(
        ["git", "archive", "--format=tar", MILES_BASE],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as handle:
        handle.extractall(source, filter="data")
    subprocess.run(
        [
            "git",
            "apply",
            "--whitespace=error-all",
            str(ROOT / "docker" / "npu_patch" / "miles.patch"),
        ],
        cwd=source,
        check=True,
        capture_output=True,
        text=True,
    )
    return source


def _run_probe(source: Path, body: str) -> subprocess.CompletedProcess[str]:
    script = f"""
import importlib.abc
import importlib.util
from pathlib import Path
import sys
from types import ModuleType

source = Path({str(source)!r})
package_names = (
    "miles",
    "miles.backends",
    "miles.backends.megatron_utils",
    "miles.backends.megatron_utils.megatron_to_hf",
)
for package_name in package_names:
    package = ModuleType(package_name)
    package.__path__ = [str(source / package_name.replace(".", "/"))]
    sys.modules[package_name] = package

padding_name = (
    "miles.backends.megatron_utils.megatron_to_hf.processors.padding_remover"
)
padding = ModuleType(padding_name)
padding.remove_padding = lambda name, param, vocab_size: param
sys.modules[padding_name] = padding

processors_name = "miles.backends.megatron_utils.megatron_to_hf.processors"
processors_path = source / {PROCESSORS!r} / "__init__.py"
spec = importlib.util.spec_from_file_location(
    processors_name,
    processors_path,
    submodule_search_locations=[str(processors_path.parent)],
)
assert spec is not None and spec.loader is not None
processors = importlib.util.module_from_spec(spec)
sys.modules[processors_name] = processors
{textwrap.indent(textwrap.dedent(body), "")}
"""
    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=source,
        capture_output=True,
        text=True,
        check=False,
    )


def test_readme_pins_the_reviewed_patch_commit_and_digest():
    text = (ROOT / "docker" / "npu_patch" / "readme.md").read_text(encoding="utf-8")

    assert f"PATCH_BUNDLE_COMMIT={PATCH_COMMIT}" in text
    assert f"checkout --detach {PATCH_COMMIT}" in text
    assert f"{PATCH_SHA256}  miles.patch" in text


def test_unquantized_path_does_not_import_sglang_or_quantizer_implementations(patched_tree):
    result = _run_probe(
        patched_tree,
        """
class BlockOptionalImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "sglang" or fullname.startswith("sglang."):
            raise AssertionError(f"unexpected SGLang import: {fullname}")
        if fullname.startswith(processors_name + ".quantizer_"):
            raise AssertionError(f"unexpected quantizer import: {fullname}")
        return None

sys.meta_path.insert(0, BlockOptionalImports())
spec.loader.exec_module(processors)
payload = [("model.layers.0.weight", object())]
assert processors.quantize_params(None, "weight", payload, None) is payload
""",
    )

    assert result.returncode == 0, result.stderr


def test_quantizer_implementations_are_loaded_only_for_selected_method(patched_tree):
    result = _run_probe(
        patched_tree,
        """
spec.loader.exec_module(processors)
events = []

def install(name, function_name, marker):
    module = ModuleType(name)
    def implementation(*args, **kwargs):
        events.append(marker)
        return marker
    setattr(module, function_name, implementation)
    sys.modules[name] = module

install(processors_name + ".quantizer_fp8", "quantize_params_fp8", "fp8")
install(processors_name + ".quantizer_mxfp8", "quantize_params_mxfp8", "mxfp8")
install(
    processors_name + ".quantizer_compressed_tensors",
    "quantize_params_compressed_tensors",
    "compressed-tensors",
)

payload = [("weight", object())]
assert processors.quantize_params(None, "weight", payload, {"quant_method": "fp8"}) == "fp8"
assert events == ["fp8"]
assert processors.quantize_params(None, "weight", payload, {"quant_method": "mxfp8"}) == "mxfp8"
assert events == ["fp8", "mxfp8"]
assert (
    processors.quantize_params(
        None, "weight", payload, {"quant_method": "compressed-tensors"}
    )
    == "compressed-tensors"
)
assert events == ["fp8", "mxfp8", "compressed-tensors"]
assert processors.quantize_params(None, "weight", payload, {"quant_method": "unknown"}) is None
assert events == ["fp8", "mxfp8", "compressed-tensors"]
""",
    )

    assert result.returncode == 0, result.stderr


def test_public_quantizer_imports_remain_lazy_and_callable(patched_tree):
    result = _run_probe(
        patched_tree,
        """
class BlockOptionalImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "sglang" or fullname.startswith("sglang."):
            raise AssertionError(f"unexpected SGLang import: {fullname}")
        if fullname.startswith(processors_name + ".quantizer_"):
            raise AssertionError(f"unexpected quantizer import: {fullname}")
        return None

sys.meta_path.insert(0, BlockOptionalImports())
spec.loader.exec_module(processors)
from miles.backends.megatron_utils.megatron_to_hf.processors import (
    quantize_params_compressed_tensors,
    quantize_params_fp8,
    quantize_params_mxfp8,
)

events = []

def install(name, function_name, marker):
    module = ModuleType(name)
    def implementation(*args, **kwargs):
        events.append((marker, args, kwargs))
        return marker
    setattr(module, function_name, implementation)
    sys.modules[name] = module

install(processors_name + ".quantizer_fp8", "quantize_params_fp8", "fp8")
install(processors_name + ".quantizer_mxfp8", "quantize_params_mxfp8", "mxfp8")
install(
    processors_name + ".quantizer_compressed_tensors",
    "quantize_params_compressed_tensors",
    "compressed-tensors",
)

assert quantize_params_fp8("fp8-arg", flag=True) == "fp8"
assert quantize_params_mxfp8("mxfp8-arg") == "mxfp8"
assert quantize_params_compressed_tensors("compressed-arg") == "compressed-tensors"
assert [event[0] for event in events] == ["fp8", "mxfp8", "compressed-tensors"]
assert events[0][1:] == (("fp8-arg",), {"flag": True})
""",
    )

    assert result.returncode == 0, result.stderr
