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

# Backend of ``cake_pplx.cake_attn_res_nondet_train``: the attention residual (AttnRes), a softmax-weighted sum of
# 1 to 9 BF16 values of width 7168 (forward, backward and autograd) on SM100 / SM103 / SM107.  The values are separate
# BF16 ``[T, 7168]`` tensors (nothing is stacked), ``w`` FP32 ``[7168]``, ``eps`` a run-time float; any token
# count.  The non-deterministic package of the family.  The registry and the host plan live in ``cake_jit``, the
# loader, checks, launches and autograd in ``cake_backend``.  Every switch is an explicit keyword argument; the package
# reads no environment variable.

from .cake_backend import (
    ARCHES,
    HIDDEN,
    MAX_VALUES,
    SUPPORTED_COMPUTE_CAPABILITIES,
    CakeAttnResNondetFunction,
    cake_attn_res_nondet,
    cake_attn_res_nondet_train_backward,
    cake_attn_res_nondet_train_forward,
    cake_attn_res_nondet_train_retained_bytes,
    cake_attn_res_nondet_train_workspace_bytes,
    is_cake_attn_res_nondet_train_supported,
)

__all__ = [
    "ARCHES",
    "HIDDEN",
    "MAX_VALUES",
    "SUPPORTED_COMPUTE_CAPABILITIES",
    "CakeAttnResNondetFunction",
    "cake_attn_res_nondet",
    "cake_attn_res_nondet_train_backward",
    "cake_attn_res_nondet_train_forward",
    "cake_attn_res_nondet_train_retained_bytes",
    "cake_attn_res_nondet_train_workspace_bytes",
    "is_cake_attn_res_nondet_train_supported",
]
