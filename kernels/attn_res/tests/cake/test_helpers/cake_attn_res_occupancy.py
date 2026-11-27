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
launch is the record's block and dynamic shared memory with the compile-time cluster, or, for a program that launches
without a cluster (a record of cluster ``(1, 1, 1)``), a cluster-dimension launch attribute of one CTA (a program past
48 KiB of dynamic shared memory gets its limit raised first, as its binding raises it before a launch).  Test-only:
the packages size their grids from measured tables (``cake_jit.RESIDENT_GROUPS``) and, for one-CTA groups, from CTAs
per SM * SMs.
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


class _LaunchAttribute(ctypes.Structure):
    """``CUlaunchAttribute``: the id, padding to 8 bytes, then the 64-byte value union."""

    _fields_ = [("id", ctypes.c_uint), ("pad", ctypes.c_char * 4), ("value", ctypes.c_uint * 16)]


# ``CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION``.
_CLUSTER_DIMENSION_ATTRIBUTE = 4
# ``CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES`` and the dynamic shared memory a function takes without it.
_MAX_DYNAMIC_SMEM_ATTRIBUTE = 8
_DEFAULT_DYNAMIC_SMEM_BYTES = 48 * 1024


def _cuobjdump() -> Optional[str]:
    found = shutil.which("cuobjdump")
    if found:
        return found
    for root in (os.environ.get("CUDA_HOME"), os.environ.get("CUDA_PATH")):
        if root and (Path(root) / "bin" / "cuobjdump").is_file():
            return str(Path(root) / "bin" / "cuobjdump")
    return None


def unreachable() -> Optional[str]:
    """Why the query cannot run on this host (no ``cuobjdump``, no driver library), or None."""
    if _cuobjdump() is None:
        return "cuobjdump is not on PATH or under CUDA_HOME"
    try:
        ctypes.CDLL("libcuda.so.1")
    except OSError as exc:
        return f"libcuda.so.1 does not load: {exc}"
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
                smem = int(launch["dynamic_smem_bytes"])
                if smem > _DEFAULT_DYNAMIC_SMEM_BYTES:
                    status = cuda.cuFuncSetAttribute(
                        function, ctypes.c_int(_MAX_DYNAMIC_SMEM_ATTRIBUTE), ctypes.c_int(smem)
                    )
                    if status:
                        return None, f"cuFuncSetAttribute(max dynamic shared memory {smem}) returned {status}"
                ctas = int(launch["cluster"][0])
                config = _LaunchConfig(ctas * 1024, 1, 1, *map(int, launch["block"]))
                config.shared_mem_bytes = smem
                attribute = _LaunchAttribute(_CLUSTER_DIMENSION_ATTRIBUTE)
                if ctas == 1:
                    # No compile-time cluster: the query takes the cluster shape from the launch, one CTA.
                    attribute.value[0] = attribute.value[1] = attribute.value[2] = 1
                    config.attrs = ctypes.cast(ctypes.pointer(attribute), ctypes.c_void_p)
                    config.num_attrs = 1
                clusters, per_sm = ctypes.c_int(0), ctypes.c_int(0)
                status = cuda.cuOccupancyMaxActiveClusters(ctypes.byref(clusters), function, ctypes.byref(config))
                if status:
                    return None, f"cuOccupancyMaxActiveClusters returned {status}"
                status = cuda.cuOccupancyMaxActiveBlocksPerMultiprocessor(
                    ctypes.byref(per_sm), function, ctypes.c_int(int(launch["block"][0])), ctypes.c_size_t(smem)
                )
                blocks = per_sm.value if status == 0 else f"query returned {status}"
                return clusters.value, f"cuOccupancyMaxActiveClusters (CTAs per SM {blocks})"
            finally:
                cuda.cuModuleUnload(module)
        return None, f"{record['kernel_symbol']} is in none of the {len(images)} cubins of {library.name}"
    finally:
        cuda.cuCtxSetCurrent(previous)
        cuda.cuDevicePrimaryCtxRelease(device)
