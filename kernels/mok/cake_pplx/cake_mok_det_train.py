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

Cake MoK (Mixture of Kittens) MoE training kernels (SM100 / SM103 / SM107),
deterministic: every output repeats bit for bit from run to run (the backend
README, "Repeatability").

Thin public entry points of the backend package
``cake_pplx._backends.cake_mok_bf16``: the three functions below carry the
public signatures and documentation and forward to
``cake_pplx._backends.cake_mok_bf16.api``, which holds the bodies, the
measured communication-SM tables and their resolvers.  This module imports the
package only inside the functions (and lazily for the table and resolver
names), never at import time.

Naming rules:

1. The mode is part of the module name (``cake_mok_det_train``).  The MoK
   kernels have no non-deterministic mode.
2. No entry point takes a ``deterministic`` keyword.
"""

from ._cake_pplx_api_mok import cake_api

# Tables and resolvers that live in the package but stay reachable here
# (``cake_mok_det_train.COMM_SMS``, ``cake_mok_det_train.resolve_comm_sms(...)``, ...).
_PACKAGE_NAMES = (
    "COMM_SMS",
    "IMBALANCE_COMM_SMS",
    "IMBALANCE_DEVICE_THRESHOLDS",
    "IMBALANCE_THRESHOLD",
    "IMBALANCE_THRESHOLDS",
    "MOK_PRECISIONS",
    "ROUTER_WEIGHT_POSITIONS",
    "resolve_comm_sms",
    "resolve_imbalance_comm_sms",
    "resolve_imbalance_threshold",
)


def __getattr__(name):
    if name in _PACKAGE_NAMES:
        from ._backends.cake_mok_bf16 import api as _api

        return getattr(_api, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_PACKAGE_NAMES))


@cake_api
def prepare_mok_bf16(
    *,
    ep_size=16,
    local_experts=16,
    topk=8,
    clamped_swiglu=False,
    fp32_wgrad=False,
    router_weight_position="after_down",
):
    """Prepare the explicit Cake MoK training backend.

    Returns an adapter with ``build_schedule``, ``forward``,
    ``recompute_forward_context`` and ``backward`` methods using the public
    Mixture of Kittens functional signatures. Use
    :func:`create_mok_bf16_workspace` for caller-owned workspace and metadata.
    All compute uses the standalone CUDA sources shipped here.

    Routed experts run in BF16, or natively in MXFP8 when the routed weights
    are caller-prequantized MXFP8 tuples (see
    :func:`quantize_mok_mxfp8_weights`); the shared expert stays BF16.
    ``swiglu_limit`` (a per-call runtime value) selects clamped SwiGLU,
    ``silu(min(gate, L)) * clamp(up, -L, L)``, for routed and shared experts.
    ``clamped_swiglu`` and ``fp32_wgrad`` choose the kernels compiled now;
    call the adapter's ``prepare``/``prepare_recompute`` for any other
    variant before CUDA Graph capture. With ``fp32_wgrad=True``, ``backward``
    adds every weight gradient into caller-owned FP32 accumulators passed as
    ``weight_grad_accumulators``.

    ``router_weight_position`` (one of :data:`ROUTER_WEIGHT_POSITIONS`)
    places each route's FP32 router weight ``w``. ``"after_down"`` (default)
    weights the routed expert's output in the combine,
    ``y = S(x) + sum_k w_k E_k(x)``. ``"before_down"`` weights the SwiGLU
    activation ``a`` before the down projection, ``h = BF16(w * a)``, and sums
    the routed outputs unweighted; its router-weight gradient is
    ``dot(dh, a)`` (``dh`` the gradient of ``h``), defined at a zero weight.
    ``"before_down"`` runs clamped SwiGLU only (``clamped_swiglu=True`` and a
    ``swiglu_limit`` on every call), BF16 routed weights only, FP32
    weight-gradient accumulation only (``fp32_wgrad=True``), on sm_100a,
    sm_103a and sm_107a; ``recompute_forward_context`` then takes the
    forward's ``router_weights``.

    Prepare and warm up outside graph capture. Supported scheduler layouts
    (EP, local experts) are (1, 4) and (4, 4) for diagnostics (top-k 2 in
    both positions, top-k 8 with ``"after_down"``), 256 routed experts at EP
    4/8/16/32/64 and 288 routed experts at EP 8/32 with top-k 2 or 8
    (``"after_down"``), and 384 routed experts at EP 4/8/16/32/64 ((4, 96),
    (8, 48), (16, 24), (32, 12), (64, 6)) with top-k 6 (``"before_down"``).
    Any other combination raises before any kernel is loaded or launched:
    ``TypeError`` for a position that is not a ``str``, an ``ep_size``,
    ``local_experts`` or ``topk`` that is not an ``int`` (a ``bool``
    included) and a ``clamped_swiglu`` or ``fp32_wgrad`` that is not a
    ``bool``, in both positions; ``ValueError`` for every other value.
    Tokens, hidden/intermediate widths and per-rank source counts
    remain runtime values subject to the documented alignment constraints.
    The backend provides the MoE layer only, with no full-model or optimizer
    integration.
    """
    from ._backends.cake_mok_bf16 import api as _api

    return _api.prepare_mok_bf16(
        ep_size=ep_size,
        local_experts=local_experts,
        topk=topk,
        clamped_swiglu=clamped_swiglu,
        fp32_wgrad=fp32_wgrad,
        router_weight_position=router_weight_position,
    )


@cake_api
def create_mok_bf16_workspace(
    *,
    group,
    device,
    num_local_tokens,
    hidden_size,
    topk,
    source_capacity=None,
    fwd_num_comm_sms="auto",
    bwd_num_comm_sms="auto",
    minibatch_size=4096,
    macrobatch_size=32768,
    schedule_capacity_multiplier=3 / 16,
    precision="bf16",
    imbalance_comm_sms="auto",
    imbalance_threshold="auto",
):
    """Collectively create a caller-owned ``(config, workspace)`` pair.

    The process group must be initialized and its CUDA device current.
    Source input counts may differ across ranks, including zero. Common
    physical capacity is negotiated once; optional ``source_capacity``
    reserves future growth and must agree across ranks. Returned workspace
    exposes ``source_capacity`` and physical ``storage``. Defaults describe
    a small toy ring; see the backend README for capacity and reuse rules.
    The same workspace serves BF16 and MXFP8 routed experts.
    Call outside CUDA Graph capture; retain the workspace for every replay.
    A routed-row overflow traps the kernel: a ``schedule_capacity_multiplier``
    of at least ``ceil((min(topk, E) * T * EP + E * 255) / (T * topk)) / EP``
    (``E`` local experts, ``T`` the physical source rows, at least the
    largest rank's source rows or ``source_capacity``, ``EP`` the group
    size; about ``(EP + 1) / EP``) holds every routing; any positive
    multiplier is accepted. Every rank passes the same configuration, so
    compute the multiplier from a row bound all ranks agree on, such as
    ``source_capacity`` (the bound decreases as ``T`` grows).

    ``fwd_num_comm_sms`` / ``bwd_num_comm_sms`` are the fixed (forward, backward)
    communication-SM counts of every launch. ``"auto"`` (default, for both)
    gives each launch the measured row of the device, the launch's routed
    precision (BF16 or MXFP8 routed weights) and the EP group size
    (:func:`resolve_comm_sms`, :data:`COMM_SMS`; B300 BF16 at EP8: 24 / 40,
    GB300 BF16 at EP4-64: 20 / 24, GB300 MXFP8 at EP8/32/64: 44 / 36,
    Rubin (212 SMs) MXFP8 at EP8: 60 / 60) and the
    factory split 40 / 40 where there is no row, whatever ``precision`` the
    workspace was created with (``MoKConfig.bf16_comm_sms`` /
    ``mxfp8_comm_sms``; the returned ``fwd_num_comm_sms`` /
    ``bwd_num_comm_sms`` hold the row of ``precision``; on such a config a
    ``dataclasses.replace`` of ``fwd_num_comm_sms`` / ``bwd_num_comm_sms`` alone
    changes no launch: create the workspace with integers, or replace
    ``bf16_comm_sms`` / ``mxfp8_comm_sms`` too). A pair of integers sets the
    fixed split of every launch of both routed precisions; on a device with an
    imbalance row the imbalance split still replaces it under skew unless
    ``imbalance_comm_sms=None``. The split changes time only, never an output
    bit.
    ``imbalance_comm_sms`` is the (forward, backward) communication-SM split the
    launches switch to while the hottest rank's routed-row load is at least
    ``imbalance_threshold`` times the mean per-rank load (observed on the
    gathered expert IDs of each step without a host synchronization; the counts
    are identical on every rank, each rank applies its own latest completed
    observation; a CUDA Graph keeps the split in force at its capture, and
    layers that share one workspace share its latest observation). ``"auto"``
    (default) takes the measured row of the device for ``precision``
    (:func:`resolve_imbalance_comm_sms`, keyed by compute capability and SM
    count; B200 BF16: 24 / 28, B200 MXFP8: 44 / 36, Rubin (212 SMs) MXFP8:
    72 / 60, no row elsewhere - GB200, GB300 and B300 included - so the fixed
    split stays in force there),
    ``None`` keeps the fixed split and a pair sets it. ``precision`` selects
    that row and the threshold.
    ``imbalance_threshold`` ``"auto"`` (default) takes the measured ratio of
    the device and ``precision`` (:func:`resolve_imbalance_threshold`;
    :data:`IMBALANCE_DEVICE_THRESHOLDS`, else :data:`IMBALANCE_THRESHOLDS`):
    BF16 4, between the 3x load ratio, where its split shows no gain, and 5.6x,
    where it gains 1-2 %; MXFP8 2.7, between 2.4x (no gain) and 3x (gains 1 %);
    Rubin MXFP8 1.5, its split gaining from the measured 1.51x up. A number
    sets the ratio. Balanced steps never switch.
    ``MoKConfig.comm_split_log`` (default ``False``; ``dataclasses.replace(config,
    comm_split_log=True)`` on the returned config) prints each split change per
    rank to stderr.
    """
    from ._backends.cake_mok_bf16 import api as _api

    return _api.create_mok_bf16_workspace(
        group=group,
        device=device,
        num_local_tokens=num_local_tokens,
        hidden_size=hidden_size,
        topk=topk,
        source_capacity=source_capacity,
        fwd_num_comm_sms=fwd_num_comm_sms,
        bwd_num_comm_sms=bwd_num_comm_sms,
        minibatch_size=minibatch_size,
        macrobatch_size=macrobatch_size,
        schedule_capacity_multiplier=schedule_capacity_multiplier,
        precision=precision,
        imbalance_comm_sms=imbalance_comm_sms,
        imbalance_threshold=imbalance_threshold,
    )


@cake_api
def quantize_mok_mxfp8_weights(gate, up, down):
    """Prequantize BF16 routed expert weights for native MXFP8 MoK training.

    ``gate``/``up`` are ``[E_local, I, H]`` and ``down`` is ``[E_local, H, I]``
    BF16 tensors. Uses the MoK MXFP8 recipe (32-element UE8M0 blocks along
    the reduction dimension, E4M3 data, tensor-core scale tiles) in both
    operand orientations. Returns ``(forward_weights, backward_weights)``:
    pass ``forward_weights`` as the routed gate/up/down arguments of
    ``forward`` (and its first two entries to ``recompute_forward_context``)
    and ``backward_weights`` to ``backward``. Requantize after every weight
    update; weight gradients are returned in BF16 (or FP32 accumulators).
    """
    from ._backends.cake_mok_bf16 import api as _api

    return _api.quantize_mok_mxfp8_weights(
        gate,
        up,
        down,
    )
