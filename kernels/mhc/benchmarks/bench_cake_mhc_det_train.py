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

Benchmark of the deterministic Cake four-stream mHC training package (``cake_pplx.cake_mhc_det_train``) against the
non-Cake implementations of the same function (``cake_mhc_baselines.py``): PyTorch eager FP32, ``torch.compile``
(default and ``max-autotune-no-cudagraphs``), Liger Kernel's mHC and, for the forward, TileKernels' fused kernels
after a cuBLAS FP32 or DeepGEMM hi/lo prenorm GEMM.

Per row (hidden size and token count) and operation (``fwd``, ``bwd``, ``ckpt``) every arm is timed in interleaved
groups, in eager launches and in CUDA graph replays, as described in ``cake_mhc_harness.py``.  Every arm is held to
the family's accuracy judge first (``tests/cake_naive.py``: the error against a float64 oracle within the naive
PyTorch BF16 tier's), and on each row its outputs on the row's inputs are compared with the oracle before they are
timed.  ``best_non_cake`` names the fastest non-Cake arm that meets the judge.

The same file times another release of the package: run it in an environment where that release is installed
(``--arms cake_det --label <release>``) and pair the invocations on one GPU.

Example::

    python benchmarks/bench_cake_mhc_det_train.py --json det.json
    python benchmarks/bench_cake_mhc_det_train.py --shapes 4096x16263 7168x65536 --ops bwd --modes eager
    python benchmarks/bench_cake_mhc_det_train.py --arms cake_det --aa --judge-draws 0 --label new --json new.json
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cake_mhc_harness as harness  # noqa: E402

DEFAULT_ARMS = "cake_det,eager_fp32,compile_default,compile_max_autotune,liger,tk_cublas,tk_deepgemm"

if __name__ == "__main__":
    raise SystemExit(harness.main(description=__doc__, default_arms=DEFAULT_ARMS))
