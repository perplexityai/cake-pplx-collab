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

Benchmark of the non-deterministic Cake four-stream mHC training package (``cake_pplx.cake_mhc_nondet_train``)
against the deterministic package (``cake_pplx.cake_mhc_det_train``) and the non-Cake implementations of the same
function (``cake_mhc_baselines.py``).

The rows, operations, modes, timing and accuracy checks are those of ``bench_cake_mhc_det_train.py``
(``cake_mhc_harness.py``).  The first arm, the non-deterministic package, is the reference of the per-group ratios:
``over_cake_nondet`` of the deterministic arm is ``det / nondet`` (above 1: the non-deterministic package is faster),
and of a non-Cake arm its time over the non-deterministic package's.  The non-deterministic package exports hidden
4096, 5120 and 7168 on SM100, SM103 and SM107, as the deterministic one does; a row whose hidden size an arm does not
serve on the device records that arm as unavailable and times the other arms.

Example::

    python benchmarks/bench_cake_mhc_nondet_train.py --json nondet.json
    python benchmarks/bench_cake_mhc_nondet_train.py --arms cake_nondet,cake_det --aa --judge-draws 0 --json ab.json
    python benchmarks/bench_cake_mhc_nondet_train.py --shapes 5120x16384,5120x65536 --ops hfwd,hbwd,hckpt,hcfwd,cfwd,cbwd \
        --json handoff.json
    python benchmarks/bench_cake_mhc_nondet_train.py --shapes 5120x16384,5120x65536 --ops hcfwd,cfwd --collapse-pdl off \
        --json collapse-pdl-off.json
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cake_mhc_harness as harness  # noqa: E402

DEFAULT_ARMS = "cake_nondet,cake_det,eager_fp32,compile_default,compile_max_autotune,liger,tk_cublas,tk_deepgemm"

if __name__ == "__main__":
    raise SystemExit(harness.main(description=__doc__, default_arms=DEFAULT_ARMS))
