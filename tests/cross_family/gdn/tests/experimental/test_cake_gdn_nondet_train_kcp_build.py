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

Cross-family manifest check of the non-deterministic Cake packed GDN KCP training package
``cake_pplx._backends.cake_gdn_nondet_train_kcp``, moved here unchanged from
``kernels/gdn/tests/experimental/test_cake_gdn_nondet_train_kcp_build.py`` (the in-family checks, among them the
pinned bytes across the GDN packages, stay there): the loader ``cake_build.py`` and the header
``csrc/cake_ffi_checks.h`` are the pinned bytes of the KDA family's ``cake_kda_det_train`` package too
(``cake-pplx-kda``, imported through ``tests/cake/test_helpers/cake_package.py``: a missing distribution fails, never
skips).  Nothing is built or launched.
"""

from __future__ import annotations

import hashlib
import importlib
from pathlib import Path

import pytest
import torch

PACKAGE = "cake_gdn_nondet_train_kcp"


@pytest.fixture(autouse=True, scope="module")
def _few_cpu_threads():
    """The CPU emulation runs many small matmuls and elementwise ops.  On a host with hundreds of CPUs, torch's default
    intra-op pool and the oneDNN matmul path (its own pool, which torch.set_num_threads does not resize) spend far more
    time synchronizing threads than computing (see ``test_cake_gdn_nondet_train_kcp.py``), so this module runs with at most 8
    threads and oneDNN off, as that suite does."""
    threads, onednn = torch.get_num_threads(), torch.backends.mkldnn.enabled
    torch.set_num_threads(min(threads, 8))
    torch.backends.mkldnn.enabled = False
    try:
        yield
    finally:
        torch.backends.mkldnn.enabled = onednn
        torch.set_num_threads(threads)


def _package_import() -> str:
    """The import path of the package: the test-helper registry's entry (a stale entry fails here, never skips)."""
    from tests.cake.test_helpers.cake_package import PACKAGES

    return PACKAGES[PACKAGE]


def _package(sub: str | None = None):
    """The package (or one of its modules) at :func:`_package_import`'s path."""
    path = _package_import()
    return importlib.import_module(path if sub is None else f"{path}.{sub}")


PACKAGE_IMPORT = _package_import()
PACKAGE_DIR = Path(_package().__file__).resolve().parent


def _cake_build():
    """The pinned loader ``cake_build.py`` (a generated file), or a skip while the render has not delivered it."""
    if not (PACKAGE_DIR / "cake_build.py").is_file():
        pytest.skip("cake_build.py is not rendered into the package yet")
    return importlib.import_module(f"{PACKAGE_IMPORT}.cake_build")


def test_the_loader_and_header_are_the_pinned_bytes_of_every_cake_package() -> None:
    """``cake_build.py`` is byte-identical across the Cake packages (the manifest checker's rule); the binding checks
    header too."""
    _cake_build()
    if not (PACKAGE_DIR / "csrc" / "cake_ffi_checks.h").is_file():
        pytest.skip("csrc/cake_ffi_checks.h is not rendered into the package yet")
    from tests.cake.test_helpers.cake_package import cake_package

    kda_dir = Path(
        cake_package("cake_kda_det_train", "cake_build").__file__
    ).parent
    for relative in ("cake_build.py", "csrc/cake_ffi_checks.h"):
        ours = (PACKAGE_DIR / relative).read_bytes()
        theirs = (kda_dir / relative).read_bytes()
        assert hashlib.sha256(ours).hexdigest() == hashlib.sha256(theirs).hexdigest(), (
            relative
        )
