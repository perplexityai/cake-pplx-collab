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

# Backend of ``cake_pplx.cake_fmha_nondet_train``: the backward of dense
# multi-head attention, head dimension 64 or 128, non-causal or causal, on
# SM100 / SM103 / SM107.  Q, O, dO ``[B, Sq, H, D]`` and K, V ``[B, Skv, H, D]``
# (BF16, contiguous), the FP32 natural-log LSE ``[B, H, Sq]``; any head count,
# query and key lengths multiples of 128.  The ``nondet`` package of the family,
# and its only package: the dQ tiles are added into an FP32 accumulator in
# arrival order.  The entry point is ``cake_fmha_nondet_train_backward`` (a
# backward step; the family has no forward and no autograd entry).  The registry
# and the host plan live in ``cake_jit``, the loader, checks and launches in
# ``cake_backend``.  Every switch is an explicit keyword argument; the package
# reads no environment variable.

from .cake_backend import ARCHES, HEAD_DIMS, cake_fmha_nondet_train_backward

__all__ = [
    "ARCHES",
    "HEAD_DIMS",
    "cake_fmha_nondet_train_backward",
]
