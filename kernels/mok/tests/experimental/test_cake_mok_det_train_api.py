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

Independent single-GPU fixtures for the MoK kernel tests.
Public-API tests of the MoK facade :mod:`cake_pplx.cake_mok_det_train`.

The facade's three entry points (``prepare_mok_bf16``, ``create_mok_bf16_workspace``, ``quantize_mok_mxfp8_weights``)
are thin wrappers over ``cake_pplx._backends.cake_mok_bf16.api``: they carry no experimental marking, have the
package functions' signatures (so the documented defaults are the package defaults), re-export the package's
communication-SM tables and resolvers lazily (the facade imports no backend module when it is imported) and forward
their arguments (the package's own argument errors surface unchanged).  The kernels, the adapter, the host tables and
the configuration switches are covered by the backend suites, which import the package only through
:func:`tests.cake.test_helpers.cake_package.cake_package`.
"""

import inspect
import subprocess
import sys

import pytest
import torch

from cake_pplx import cake_mok_det_train as mok
from cake_pplx._backends.cake_mok_bf16 import api

ENTRY_POINTS = (
    "prepare_mok_bf16",
    "create_mok_bf16_workspace",
    "quantize_mok_mxfp8_weights",
)
RE_EXPORTS = (
    "COMM_SMS",
    "IMBALANCE_COMM_SMS",
    "IMBALANCE_DEVICE_THRESHOLDS",
    "IMBALANCE_THRESHOLD",
    "IMBALANCE_THRESHOLDS",
    "MOK_PRECISIONS",
    "resolve_comm_sms",
    "resolve_imbalance_comm_sms",
    "resolve_imbalance_threshold",
)


@pytest.mark.parametrize("name", ENTRY_POINTS)
def test_entry_points_mirror_the_package(name):
    """Each entry point is a plain function with the package function's signature, parameter defaults included,
    and no experimental marking."""
    wrapper, implementation = getattr(mok, name), getattr(api, name)
    assert inspect.isfunction(wrapper) and wrapper.__module__ == mok.__name__
    assert not hasattr(wrapper, "is_experimental")
    assert not (wrapper.__doc__ or "").lstrip().startswith(".. warning::")
    assert inspect.signature(wrapper) == inspect.signature(implementation)


def test_re_exports_are_the_package_objects():
    """The communication-SM tables and resolvers are reachable from the facade, are the package objects themselves
    and appear in ``dir()``; unknown names raise ``AttributeError``."""
    for name in RE_EXPORTS:
        assert getattr(mok, name) is getattr(api, name), name
    assert set(RE_EXPORTS) <= set(dir(mok))
    assert not hasattr(mok, "resolve_nothing")  # ``__getattr__`` raises AttributeError


def test_import_loads_no_backend_module():
    """Importing the facade imports no module of the backend package."""
    code = (
        "import sys, cake_pplx.cake_mok_det_train\n"
        "print(sorted(m for m in sys.modules if m.startswith('cake_pplx._backends')))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]", out.stdout


def test_argument_errors_of_the_package_surface_unchanged():
    """The facade forwards its keyword arguments to the package and raises the package's argument errors before
    any device or collective work."""
    if not torch.cuda.is_available():
        pytest.skip("The workspace factory resolves the current CUDA device")
    kwargs = dict(group=None, device=None, num_local_tokens=512, hidden_size=256, topk=2)
    with pytest.raises(ValueError, match='"auto"'):
        mok.create_mok_bf16_workspace(imbalance_comm_sms="fast", **kwargs)
    with pytest.raises(ValueError, match='"auto"'):
        mok.create_mok_bf16_workspace(imbalance_threshold="fast", **kwargs)
    with pytest.raises(ValueError, match="pair"):
        mok.create_mok_bf16_workspace(imbalance_comm_sms=(24,), **kwargs)
    with pytest.raises(ValueError, match="precision must be one of"):
        mok.create_mok_bf16_workspace(precision="fp8", **kwargs)
    with pytest.raises(ValueError, match='both be "auto"'):
        mok.create_mok_bf16_workspace(fwd_num_comm_sms="auto", bwd_num_comm_sms=40, **kwargs)
    with pytest.raises(ValueError, match="Unsupported exported"):
        mok.prepare_mok_bf16(ep_size=3, local_experts=4, topk=2)
    with pytest.raises(ValueError, match="precision must be one of"):
        mok.resolve_imbalance_threshold("fp8")
