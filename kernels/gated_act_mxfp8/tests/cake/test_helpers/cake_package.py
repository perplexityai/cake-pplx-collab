"""The one place the CAKE backend test suites name their package.

Every CAKE backend suite (``tests/experimental/test_cake_<package>.py``, ``tests/norm/test_cake_rmsnorm_det_train.py``, ``tests/norm/test_cake_rmsnorm_nondet_train.py``,
``tests/mhc/test_cake_mhc_det_train.py``, ``tests/experimental/test_mok_*.py``) imports its package through :func:`cake_package` and imports nothing else from
``cake_pplx``; the public-API tests of a package live in its ``test_cake_<package>_api.py`` sibling.  A build
that vendors a package under another import path runs the backend suite unchanged after pointing the entry of
:data:`PACKAGES` at its own package.
"""

from __future__ import annotations

import importlib
from types import ModuleType

# suite / package name -> import path of the package
PACKAGES: dict[str, str] = {
    "cake_gated_act_mxfp8_det_train": "cake_pplx._backends.cake_gated_act_mxfp8_det_train",
    # the non-deterministic LM-head package; its key carries the namespace
    # the non-deterministic GQA HD256 training package; its key carries the namespace
}


def cake_package(name: str, submodule: str | None = None) -> ModuleType:
    """The package module of ``name`` (a key of :data:`PACKAGES`), or its ``submodule`` (``"cake_backend"``, ...).

    ``cake_package("cake.cake_lm_head_det_train")`` is the package itself; ``cake_package("cake.cake_lm_head_det_train", "cake_backend")``
    is ``<package>.cake_backend``.  Raises ``KeyError`` for a name the table does not list.
    """
    path = PACKAGES[name]
    if submodule:
        path = f"{path}.{submodule}"
    return importlib.import_module(path)
