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

# Backend of ``cake_pplx.cake_engram_gate_det_train``: the Engram context
# gate of four residual streams of width 5120 (forward, backward and autograd)
# on SM100 / SM103 / SM107.  X, out, key BF16 ``[T, 4, 5120]``, value BF16
# ``[T, 5120]`` (key and value may be views into one packed ``[T, 25600]``),
# q_weight and k_weight BF16 or FP32 ``[4, 5120]``, an optional bool token mask
# ``[T]``; any token count.  The deterministic package of the family.  The
# registry and the host plan live in ``cake_jit``, the loader, checks,
# launches and autograd in ``cake_backend``.  Every switch is an explicit
# keyword argument; the package reads no environment variable.

from .cake_backend import (
    ARCHES,
    HIDDEN,
    KV_WIDTH,
    STREAMS,
    CakeEngramGateDetFunction,
    CakeEngramGateDetKvFunction,
    cake_engram_gate_det,
    cake_engram_gate_det_kv,
    cake_engram_gate_det_train_backward,
    cake_engram_gate_det_train_forward,
    cake_engram_gate_det_train_workspace_bytes,
    is_cake_engram_gate_det_train_supported,
)

__all__ = [
    "ARCHES",
    "HIDDEN",
    "KV_WIDTH",
    "STREAMS",
    "CakeEngramGateDetFunction",
    "CakeEngramGateDetKvFunction",
    "cake_engram_gate_det",
    "cake_engram_gate_det_kv",
    "cake_engram_gate_det_train_backward",
    "cake_engram_gate_det_train_forward",
    "cake_engram_gate_det_train_workspace_bytes",
    "is_cake_engram_gate_det_train_supported",
]
