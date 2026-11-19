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
"""

# Backend of ``cake_pplx.cake_gated_kv_compression_det_train``: ratio-2
# document-bounded gated KV compression, width 512, forward and backward, on
# SM100 / SM103 / SM107.  C and Z FP32 ``[T, 512]`` (row-strided views such as
# the column halves of one ``[T, 1024]`` tensor are read in place),
# ``cu_seqlens`` int32 ``[S + 1]`` on the device, out and dout BF16
# ``[G, 512]``, dC and dZ FP32 ``[T, 512]`` (writable into the halves of one
# caller buffer).  The deterministic package of the family, and its only
# package: every output element depends on its own pair or token only.  The
# registry and the launch glue live in ``cake_jit``, the checks, launches and
# autograd nodes in ``cake_backend``.  Every switch is an explicit argument;
# the package reads no environment variable.

from .cake_backend import (
    ARCHES,
    SUPPORTED_COMPUTE_CAPABILITIES,
    CakeGatedKvCompressionDetCzFunction,
    CakeGatedKvCompressionDetFunction,
    cake_gated_kv_compression_det,
    cake_gated_kv_compression_det_cz,
    cake_gated_kv_compression_det_train_backward,
    cake_gated_kv_compression_det_train_forward,
    cake_gated_kv_compression_det_train_memory_bytes,
    is_cake_gated_kv_compression_det_train_supported,
)

__all__ = [
    "ARCHES",
    "SUPPORTED_COMPUTE_CAPABILITIES",
    "CakeGatedKvCompressionDetCzFunction",
    "CakeGatedKvCompressionDetFunction",
    "cake_gated_kv_compression_det",
    "cake_gated_kv_compression_det_cz",
    "cake_gated_kv_compression_det_train_backward",
    "cake_gated_kv_compression_det_train_forward",
    "cake_gated_kv_compression_det_train_memory_bytes",
    "is_cake_gated_kv_compression_det_train_supported",
]
