"""Ahead-of-time JIT warm-up for the backend suites.

It replaces FlashInfer's ``flashinfer.jit.build_jit_specs`` warm-up: the specs a
backend's ``cake_jit.specs_for_arch`` returns are the package's own JIT shim
specs (``_cake_pplx_jit.JitSpec``), and each ``build()`` compiles one library
into ``$CAKE_PPLX_CACHE_DIR`` (a cached library returns at once).  The builds
run in parallel threads, as FlashInfer's ninja warm-up did.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Iterable


def build_jit_specs(specs: Iterable[Any]) -> None:
    """Build every spec of ``specs`` (``CAKE_PPLX_WARMUP_WORKERS`` threads)."""
    specs = list(specs)
    if not specs:
        return
    workers = int(
        os.environ.get("CAKE_PPLX_WARMUP_WORKERS", str(min(16, os.cpu_count() or 1)))
    )
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for _ in pool.map(lambda spec: spec.build(), specs):
            pass
