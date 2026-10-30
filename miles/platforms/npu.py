"""Explicit Ascend registration and device/resource interfaces."""
import json
import os
import shlex

import torch
import torch_npu  # noqa: F401 -- registers the real NPU backend

from miles.platforms import Platform


class NpuPlatform(Platform):
    device_type = "npu"
    distributed_backend = "hccl"
    ray_resource = "NPU"
    nvml_numa_affinity = False

    def bind_device_before_process_group(self, local_rank: int) -> None:
        torch.npu.set_device(local_rank)

    def current_device(self):
        return torch.device("npu", torch.npu.current_device())

    def device_name(self):
        return torch.npu.get_device_name() if torch.npu.is_available() else None

    def ray_options(self, count):
        return {"num_gpus": 0, "resources": {"NPU": count}}

    def assigned_ids(self):
        import ray
        return [int(value) for value in ray.get_runtime_context().get_accelerator_ids()["NPU"]]

    def local_device_id(self, assigned):
        visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
        return visible.split(",").index(str(assigned)) if visible else int(assigned)

    def ray_start_args(self, count):
        return "--num-gpus 0 --resources " + shlex.quote(json.dumps({"NPU": count}))
