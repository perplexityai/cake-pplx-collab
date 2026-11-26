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
    "ACTIVATIONS",
    "COMM_SMS",
    "IMBALANCE_COMM_SMS",
    "IMBALANCE_DEVICE_THRESHOLDS",
    "IMBALANCE_THRESHOLD",
    "IMBALANCE_THRESHOLDS",
    "MODE_GROUPS",
    "MOK_PRECISIONS",
    "ROUTER_WEIGHT_GRADIENTS",
    "ROUTER_WEIGHT_POSITIONS",
    "SHAPE_COMM_SMS",
    "SHARED_EXPERT_MODES",
    "SOURCE_SPLIT_COMM_SMS",
    "SOURCE_SPLIT_THRESHOLD",
    "resolve_comm_sms",
    "resolve_imbalance_comm_sms",
    "resolve_imbalance_threshold",
    "resolve_shape_comm_sms",
    "resolve_source_split_comm_sms",
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
    with pytest.raises(ValueError, match='"auto"'):
        mok.create_mok_bf16_workspace(source_split_comm_sms="fast", **kwargs)
    with pytest.raises(ValueError, match="precision must be one of"):
        mok.create_mok_bf16_workspace(precision="fp8", **kwargs)
    with pytest.raises(ValueError, match='both be "auto"'):
        mok.create_mok_bf16_workspace(fwd_num_comm_sms="auto", bwd_num_comm_sms=40, **kwargs)
    with pytest.raises(ValueError, match="Unsupported exported"):
        mok.prepare_mok_bf16(ep_size=3, local_experts=4, topk=2)
    with pytest.raises(ValueError, match="precision must be one of"):
        mok.resolve_imbalance_threshold("fp8")


@pytest.mark.parametrize(
    "position,ep,experts,topk,clamped,fp32,error,match",
    [
        ("middle", 64, 6, 6, True, True, ValueError, "router_weight_position must be one of"),
        (1, 64, 6, 6, True, True, TypeError, "router_weight_position must be one of"),
        ("after_down", 64, 6, 6, True, True, ValueError, "top-k 6 is supported only with"),
        ("after_down", 16, 16, 6, False, False, ValueError, r"only with .*'before_down' on the 384-expert layouts"),
        ("after_down", 64, 6, 8, False, False, ValueError, r"layout \(64, 6\) is supported only with"),
        ("after_down", 8, 48, 2, False, False, ValueError, r"layout \(8, 48\) is supported only with"),
        ("after_down", 3, 4, 2, False, False, ValueError, "Unsupported"),
        ("before_down", 8, 32, 8, True, True, ValueError, r"supports \(EP, local experts, top-k\)"),
        ("before_down", 64, 6, 2, True, True, ValueError, r"got \(64, 6, 2\)"),
        ("before_down", 64, 6, 6.0, True, True, TypeError, "topk must be an int, got float 6.0"),
        ("before_down", 64, 6, 6, False, True, ValueError, "clamped SwiGLU only"),
        ("before_down", 8, 48, 6, True, False, ValueError, "weight gradients in FP32 only"),
        ("after_down", 16, 16, 8.0, False, False, TypeError, "topk must be an int, got float 8.0"),
        ("after_down", 4, 4, 2.0, False, False, TypeError, "topk must be an int, got float 2.0"),
        ("after_down", 64.0, 4, 8, False, False, TypeError, "ep_size must be an int, got float 64.0"),
        ("after_down", 64, 4.0, 8, False, False, TypeError, "local_experts must be an int, got float 4.0"),
        ("after_down", True, 4, 2, False, False, TypeError, "ep_size must be an int, got bool True"),
        ("after_down", 16, 16, 8, 1, False, TypeError, "clamped_swiglu and fp32_wgrad must be booleans"),
        ("before_down", 64.0, 6, 6, True, True, TypeError, "ep_size must be an int, got float 64.0"),
        ("before_down", 64, 6.0, 6, True, True, TypeError, "local_experts must be an int, got float 6.0"),
        ("before_down", 64, 6, 6, True, 1, TypeError, "clamped_swiglu and fp32_wgrad must be booleans"),
    ],
    ids=[
        "unknown-position", "position-type", "top6-after-down", "top6-other-layout", "layout-64-6-top8",
        "layout-8-48-top2", "unsupported-layout", "before-down-layout", "before-down-topk", "before-down-topk-type",
        "before-down-plain", "before-down-bf16-wgrad", "after-down-topk-type", "after-down-top2-type",
        "after-down-ep-type", "after-down-experts-type", "after-down-ep-bool", "after-down-flag-type",
        "before-down-ep-type", "before-down-experts-type", "before-down-flag-type",
    ],
)
def test_prepare_refuses_unsupported_router_weight_combinations(
    position, ep, experts, topk, clamped, fp32, error, match
):
    """``prepare_mok_bf16`` refuses every unsupported (router-weight position, EP, local experts, top-k, SwiGLU form,
    weight-gradient accumulation) combination with a message naming it, before any device work (no GPU needed): top-k
    6 with ``"after_down"`` names the 384-expert layouts it runs on.  The types are checked first, the same in both
    positions: an EP size, local expert count or top-k that is not an ``int`` (a float equal to a supported value, or a
    ``bool``) and a SwiGLU or weight-gradient flag that is not a ``bool`` raise ``TypeError``."""
    with pytest.raises(error, match=match):
        mok.prepare_mok_bf16(
            ep_size=ep,
            local_experts=experts,
            topk=topk,
            clamped_swiglu=clamped,
            fp32_wgrad=fp32,
            router_weight_position=position,
        )


#: The keyword defaults of ``prepare_mok_bf16`` before the explicit modes, and the modes' defaults (today's behavior).
PREPARE_DEFAULTS = dict(
    ep_size=16,
    local_experts=16,
    topk=8,
    clamped_swiglu=False,
    fp32_wgrad=False,
    router_weight_position="after_down",
    router_weight_gradient="quotient",
    shared_expert="unscaled",
    activation="swiglu",
)


def test_prepare_defaults_keep_the_existing_behavior():
    """``prepare_mok_bf16`` keeps every earlier keyword and default, and each explicit mode defaults to the earlier
    behavior: the quotient router-weight gradient, the unscaled shared expert and SwiGLU (the first value of each
    mode table)."""
    parameters = inspect.signature(mok.prepare_mok_bf16).parameters
    assert {name: p.default for name, p in parameters.items()} == PREPARE_DEFAULTS
    assert all(p.kind is inspect.Parameter.KEYWORD_ONLY for p in parameters.values())
    assert mok.ROUTER_WEIGHT_GRADIENTS == ("quotient", "direct")
    assert mok.SHARED_EXPERT_MODES == ("unscaled", "scaled", "none")
    assert mok.ACTIVATIONS == ("swiglu", "clamped_swiglu", "situ_glu")
    assert mok.ROUTER_WEIGHT_POSITIONS == ("after_down", "before_down")


GROUP = dict(router_weight_gradient="direct", shared_expert="scaled", activation="swiglu")


@pytest.mark.parametrize(
    "ep,experts,topk,clamped,fp32,modes,error,match",
    [
        (8, 64, 10, False, True, dict(GROUP, router_weight_gradient=1), TypeError, "router_weight_gradient must be one of"),
        (8, 64, 10, False, True, dict(GROUP, shared_expert=None), TypeError, "shared_expert must be one of"),
        (8, 64, 10, False, True, dict(GROUP, activation=b"swiglu"), TypeError, "activation must be one of"),
        (8.0, 64, 10, False, True, GROUP, TypeError, "ep_size must be an int, got float 8.0"),
        (8, 64, 10, False, 1, GROUP, TypeError, "clamped_swiglu and fp32_wgrad must be booleans"),
        (8, 64, 10.0, False, True, dict(GROUP, shared_expert="half"), TypeError, "topk must be an int"),
        (8, 64, 10, False, True, dict(GROUP, router_weight_gradient="dot"), ValueError, "router_weight_gradient must be one of"),
        (8, 64, 10, False, True, dict(GROUP, shared_expert="half"), ValueError, "shared_expert must be one of"),
        (8, 64, 10, False, True, dict(GROUP, activation="gelu"), ValueError, "activation must be one of"),
        (64, 14, 16, False, True, dict(GROUP, activation="situ_glu"), ValueError, r"896 routed experts at top-k 16 .* runs \(EP, local experts\) \(64, 14\)"),
        (64, 14, 16, False, True, dict(GROUP, shared_expert="none"), ValueError, r"896 routed experts at top-k 16 .* activation='situ_glu'"),
        (8, 64, 8, False, False, {}, ValueError, r"512 routed experts at top-k 10 .* got \(EP, local experts\) \(8, 64\)"),
        (8, 32, 10, False, True, GROUP, ValueError, r"512 routed experts at top-k 10 .* got \(EP, local experts\) \(8, 32\)"),
        (8, 64, 10, False, True, dict(GROUP, router_weight_gradient="quotient"), ValueError, "router_weight_gradient='direct'"),
        (8, 64, 10, False, True, dict(GROUP, shared_expert="unscaled"), ValueError, "shared_expert='scaled'"),
        (8, 64, 10, False, False, GROUP, ValueError, "fp32_wgrad=True"),
        (8, 64, 10, True, True, GROUP, ValueError, "activation='swiglu'"),
        (8, 32, 8, False, True, dict(router_weight_gradient="direct"), ValueError, "512 routed experts at top-k 10"),
        (16, 16, 8, False, False, dict(shared_expert="scaled"), ValueError, "512 routed experts at top-k 10"),
        (8, 32, 8, False, False, dict(activation="situ_glu"), ValueError, "896 routed experts at top-k 16"),
        (16, 16, 8, False, False, dict(shared_expert="none"), ValueError, "896 routed experts at top-k 16"),
        (64, 14, 8, False, True, dict(GROUP, shared_expert="none", activation="situ_glu"), ValueError, r"896 routed experts at top-k 16 .* got \(EP, local experts\) \(64, 14\), topk=8"),
    ],
    ids=[
        "gradient-type", "shared-type", "activation-type", "ep-type-with-modes", "flag-type-with-modes",
        "topk-type-before-mode-value", "gradient-value", "shared-value", "activation-value", "896-scaled",
        "896-swiglu", "512-top8", "256-top10", "512-quotient", "512-unscaled", "512-bf16-wgrad", "512-clamped",
        "256-direct", "256-scaled", "256-situ", "256-none", "896-top8",
    ],
)
def test_prepare_refuses_unsupported_modes(ep, experts, topk, clamped, fp32, modes, error, match):
    """``prepare_mok_bf16`` refuses every unsupported use of the explicit modes, the 512-expert layouts and top-k 10
    before any device work (no GPU needed), with the types first: a mode that is not a ``str`` raises ``TypeError``,
    as does an ``int`` or ``bool`` argument of another type, also when a mode value is unknown; then an unknown mode
    and every configuration that uses the layout, top-k or modes of the 512-expert top-10 group or of the 896-expert
    top-16 group (SiTU-GLU, ``shared_expert="none"``) other than exactly as the group runs them raise ``ValueError``
    naming the group's configuration."""
    with pytest.raises(error, match=match):
        mok.prepare_mok_bf16(
            ep_size=ep,
            local_experts=experts,
            topk=topk,
            clamped_swiglu=clamped,
            fp32_wgrad=fp32,
            **modes,
        )
