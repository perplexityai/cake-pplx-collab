"""
Copyright (c) 2026, NVIDIA CORPORATION.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

The resident clusters of a loaded AttnRes program on one CUDA device, as the driver's
``cuOccupancyMaxActiveClusters`` reports them: the program's cubin is taken from its library with ``cuobjdump -xelf``
(the CUDA toolkit the JIT build uses) and loaded into the device's primary context beside the library's own copy; the
launch is the record's block and dynamic shared memory with the compile-time cluster.  Test-only: the packages size
their grids from measured tables (``cake_jit.RESIDENT_GROUPS``).
"""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Optional


class _LaunchConfig(ctypes.Structure):
    """``CUlaunchConfig``."""

    _fields_ = [
        ("grid_x", ctypes.c_uint),
        ("grid_y", ctypes.c_uint),
        ("grid_z", ctypes.c_uint),
        ("block_x", ctypes.c_uint),
        ("block_y", ctypes.c_uint),
        ("block_z", ctypes.c_uint),
        ("shared_mem_bytes", ctypes.c_uint),
        ("stream", ctypes.c_void_p),
        ("attrs", ctypes.c_void_p),
        ("num_attrs", ctypes.c_uint),
    ]


def _cuobjdump() -> Optional[str]:
    found = shutil.which("cuobjdump")
    if found:
        return found
    for root in (os.environ.get("CUDA_HOME"), os.environ.get("CUDA_PATH")):
        if root and (Path(root) / "bin" / "cuobjdump").is_file():
            return str(Path(root) / "bin" / "cuobjdump")
    return None


def max_active_clusters(library: Path, record: dict[str, Any], device_index: int) -> tuple[Optional[int], str]:
    """``(clusters, why)``: the clusters of ``record`` (a ``MODULES`` record) in the loaded ``library`` one device holds
    at once, or None and the reason the query is not reachable."""
    tool = _cuobjdump()
    if tool is None:
        return None, "cuobjdump is not on PATH or under CUDA_HOME"
    with tempfile.TemporaryDirectory() as tmp:
        done = subprocess.run([tool, "-xelf", "all", str(library)], cwd=tmp, capture_output=True, text=True)
        if done.returncode != 0:
            return None, f"cuobjdump -xelf failed: {done.stderr[-300:]}"
        images = [path.read_bytes() for path in sorted(Path(tmp).glob("*.cubin"))]
    try:
        cuda = ctypes.CDLL("libcuda.so.1")
    except OSError as exc:
        return None, f"libcuda.so.1 does not load: {exc}"
    device, context, previous = ctypes.c_int(), ctypes.c_void_p(), ctypes.c_void_p()
    if cuda.cuDeviceGet(ctypes.byref(device), device_index) or cuda.cuDevicePrimaryCtxRetain(
        ctypes.byref(context), device
    ):
        return None, "no primary context of the device"
    cuda.cuCtxGetCurrent(ctypes.byref(previous))
    cuda.cuCtxSetCurrent(context)
    try:
        symbol = record["kernel_symbol"].encode()
        launch = record["launch"]
        for image in images:
            module, function = ctypes.c_void_p(), ctypes.c_void_p()
            if cuda.cuModuleLoadData(ctypes.byref(module), image):
                continue
            try:
                if cuda.cuModuleGetFunction(ctypes.byref(function), module, symbol):
                    continue
                config = _LaunchConfig(int(launch["cluster"][0]) * 1024, 1, 1, *map(int, launch["block"]))
                config.shared_mem_bytes = int(launch["dynamic_smem_bytes"])
                clusters = ctypes.c_int(0)
                status = cuda.cuOccupancyMaxActiveClusters(ctypes.byref(clusters), function, ctypes.byref(config))
                if status:
                    return None, f"cuOccupancyMaxActiveClusters returned {status}"
                return clusters.value, "cuOccupancyMaxActiveClusters"
            finally:
                cuda.cuModuleUnload(module)
        return None, f"{record['kernel_symbol']} is in none of the {len(images)} cubins of {library.name}"
    finally:
        cuda.cuCtxSetCurrent(previous)
        cuda.cuDevicePrimaryCtxRelease(device)
