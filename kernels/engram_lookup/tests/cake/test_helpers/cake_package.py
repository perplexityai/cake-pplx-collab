"""The one place the CAKE backend test suites name their package.

Every CAKE backend suite (``tests/experimental/test_cake_<package>.py``) imports its package through
:func:`cake_package` and imports nothing else from ``cake_pplx``; the public-API tests of a package live in its
``test_cake_<package>_api.py`` sibling.  A build that vendors a package under another import path runs the backend
suite unchanged after pointing the entry of :data:`PACKAGES` at its own package.
"""

from __future__ import annotations

import importlib
from types import ModuleType

# suite / package name -> import path of the package
PACKAGES: dict[str, str] = {
    # the deterministic Engram row lookup package; its key carries the namespace
    "cake.cake_engram_lookup_det_train": "cake_pplx._backends.cake_engram_lookup_det_train",
}


def cake_package(name: str, submodule: str | None = None) -> ModuleType:
    """The package module of ``name`` (a key of :data:`PACKAGES`), or its ``submodule`` (``"cake_backend"``, ...).

    ``cake_package("cake.cake_engram_lookup_det_train")`` is the package itself;
    ``cake_package("cake.cake_engram_lookup_det_train", "cake_backend")`` is ``<package>.cake_backend``.  Raises
    ``KeyError`` for a name the table does not list.
    """
    path = PACKAGES[name]
    if submodule:
        path = f"{path}.{submodule}"
    return importlib.import_module(path)
