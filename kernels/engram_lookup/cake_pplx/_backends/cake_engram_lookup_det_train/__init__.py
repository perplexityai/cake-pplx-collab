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

# Backend of ``cake_pplx.cake_engram_lookup_det_train``: the Engram row lookup
# on SM100 / SM103 / SM107.  Rows of a table of FP8 E4M3 values ``[R, 256]``
# with one E8M0 scale per 32 columns ``[R, 8]`` (two contiguous tensors, or two
# views into one packed ``[R, 264]`` byte buffer), gathered by int64 row ids of
# any shape and written as BF16, each value the exact product rounded once.
# The ``det`` package of the family, and its only package: a gather has no
# sums.  The entry point is ``cake_engram_lookup_det_train_forward`` (a forward
# step; the family has no backward and no autograd entry), and
# ``cake_engram_lookup_det_train_split_packed`` makes the two views of a packed
# table.  The registry and the launch grid live in ``cake_jit``, the loader,
# checks and launch in ``cake_backend``.  The package reads no environment
# variable.

from .cake_backend import (
    ARCHES,
    GROUP,
    GROUPS,
    PACKED_ROW_BYTES,
    ROW_DIM,
    cake_engram_lookup_det_train_forward,
    cake_engram_lookup_det_train_split_packed,
)

__all__ = [
    "ARCHES",
    "GROUP",
    "GROUPS",
    "PACKED_ROW_BYTES",
    "ROW_DIM",
    "cake_engram_lookup_det_train_forward",
    "cake_engram_lookup_det_train_split_packed",
]
