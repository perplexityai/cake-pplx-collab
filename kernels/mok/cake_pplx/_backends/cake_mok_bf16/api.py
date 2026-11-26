# Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
"""Host API of the Cake MoK (Mixture of Kittens) MoE training backend.

The public, experimental-marked entry points are re-exported by
:mod:`cake_pplx.cake_mok_det_train`; this module imports nothing outside this package.
"""

from . import _kernels

MOK_PRECISIONS = ("bf16", "mxfp8")
# The router-weight positions of ``prepare_mok_bf16`` (a facade name).
ROUTER_WEIGHT_POSITIONS = _kernels.ROUTER_WEIGHT_POSITIONS
# The explicit modes of ``prepare_mok_bf16`` (facade names): the
# ``"after_down"`` router-weight gradient, the shared expert's row and the
# activation; the first value of each is the default.
ROUTER_WEIGHT_GRADIENTS = _kernels.ROUTER_WEIGHT_GRADIENTS
SHARED_EXPERT_MODES = _kernels.SHARED_EXPERT_MODES
ACTIVATIONS = _kernels.ACTIVATIONS
# The kernel groups with their own configuration (layouts, top-k, modes and
# targets), by name (a facade name).
MODE_GROUPS = _kernels.MODE_GROUPS

# Imbalance-aware communication split, keyed by device compute capability, SM
# count and the workspace's precision. Only devices with a measured gain have a row;
# "auto" resolves to None (the fixed split) elsewhere. The SM count separates
# B200 (148 SMs) from GB200 (152 SMs, also compute capability (10, 0)), where
# the B200 BF16 row measured slower than the fixed split at 8x skew and above on
# EP16-64 (GB200 NVL72, an earlier review measurement). B200 BF16 row 24 / 28: EP8
# complete training steps, two rounds on one node, step time against the fixed
# 20 / 24 on this package's backward programs. Hottest rank at 5.6x / 7.4x the
# mean routed load: -1.0 / -1.7 % and -1.2 / -1.3 %. Hottest rank at 3.0x (one
# rank's experts at twice the routing probability): +0.4 / +0.7 % (full step),
# +0.5 / +0.8 % (checkpoint step) and -0.1 ... +1.6 % with ragged source rows
# (8 cells, 7 slower) - no gain, within or just above the ~0.8 % spread of
# identical arms, so the fixed split is kept. The BF16 threshold sits between
# the measured 3.0x and 5.6x; the crossover in between is not measured, and
# with 4 ranks the ratio reaches 4 only when one rank carries every routed row.
# B200 MXFP8 row 44 / 36 against the fixed 40 / 32 (same protocol, EP8, two
# rounds on one node): hottest rank at 3.0x -1.0 % (full) / -1.0 % (checkpoint),
# at 5.6x -2.7 %; balanced steps +0.5 / +0.8 % (so they must not switch); with
# 4 ranks at 2.4x +0.8 % (one screening round). The MXFP8 threshold sits
# between the measured 2.4x (no gain) and 3.0x (gain). Balanced steps never
# switch, so the fixed split stays in force for them.
#
# Rubin (compute capability (10, 7), 212 SMs) MXFP8 row 72 / 60 over its fixed
# 60 / 60 (COMM_SMS), switching from 1.5x (IMBALANCE_DEVICE_THRESHOLDS): EP8 on
# two nodes (4 GPUs each, one NVLink domain), H 4096 / I 2048 / 288 experts /
# top 8, 27116 tokens per rank, configurations rotated on one workspace, four
# rounds, hottest-rank medians, MoE training-update proxy (5 layers x 2
# microbatches). Hottest rank at 1.51x the mean routed load: forward -6.9 %,
# update -3.8 %; 2.05x -5.7 / -2.8 %; 3.05x -10.1 / -5.1 %; 4.66x
# -11.1 / -5.5 %; 6.47x -10.6 / -5.5 %; backward within +-0.1 % (a 64-SM
# backward measured 0.2 ... 1.2 % slower). Balanced steps do not switch (+0.2 /
# 0.0 % at 27116 / 32768 tokens), where an unconditional 72 / 60 forward is
# 4.1 % slower at 27116 tokens.
IMBALANCE_COMM_SMS = {
    (10, 0): {148: {"bf16": (24, 28), "mxfp8": (44, 36)}},
    (10, 7): {212: {"mxfp8": (72, 60)}},
}
IMBALANCE_THRESHOLDS = {"bf16": 4.0, "mxfp8": 2.7}
# Measured device thresholds that replace IMBALANCE_THRESHOLDS for a device
# (compute capability, SM count) and precision. Rubin MXFP8 1.5: its split gains
# from the measured 1.51x up; balanced steps (1.0x) measured no gain.
IMBALANCE_DEVICE_THRESHOLDS = {
    (10, 7): {212: {"mxfp8": 1.5}},
}
# The BF16 threshold under the single-precision name of the first imbalance
# split revision; kept so callers written against that name keep working.
IMBALANCE_THRESHOLD = IMBALANCE_THRESHOLDS["bf16"]

# Fixed (forward, backward) communication-SM split for ``fwd_num_comm_sms="auto"``
# / ``bwd_num_comm_sms="auto"`` (the default), keyed by device compute
# capability, SM count, routed precision (of each launch's routed weights, not
# the workspace's ``precision``) and EP group size. Only devices,
# precisions and group sizes with a measured gain have a row; "auto" resolves to
# the factory split (40 / 40) elsewhere. The split never changes a bit of any
# output (see the README), only time.
#
# GB300 (compute capability (10, 3), 152 SMs), H 6144 / I 2048 / 256 experts /
# top 8 at EP4/8/16/64 and H 4096 / I 2048 / 288 experts / top 8 with clamped
# SwiGLU at EP8/32, 16384 tokens per rank, FP32 weight-gradient accumulators,
# checkpointed training step (schedule, forward, context recompute, backward)
# captured in CUDA Graphs, the row and 40 / 40 interleaved in one process on the
# same nodes, slowest-rank median of 12 steps. BF16 row 20 / 24 against the
# factory 40 / 40: balanced routing EP4 -6.6 %, EP8 -7.7 % (H 6144) and -8.8 %
# (H 4096), EP16 -8.5 %, EP32 -10.7 %, EP64 -2.6 %; ragged source counts EP32
# -9.6 %, EP64 -3.5 %; hottest rank at 1.5x the mean routed load EP32 -10.2 %,
# EP64 -6.6 %. MXFP8 row 44 / 36 at EP32 / EP64 against 40 / 40: balanced
# -3.6 % / -1.8 %, ragged -1.4 % / -1.6 %, 1.5x hot rank -1.8 % / +0.7 %; EP4
# and EP16 measured no gain (within +-0.5 %) and have no row. The GB300 MXFP8
# EP8 row 44 / 36 (against the fixed 40 / 32, two nodes, two rounds,
# hottest-rank step time): hottest rank at 3.0x the mean routed load -2.0 /
# -2.2 % (full step), -2.0 / -1.8 % (checkpoint step); at 5.6x -3.1 / -3.2 %;
# at 7.4x -3.2 / -3.3 %; balanced steps within -0.4 ... +0.4 % of 40 / 40
# (five comparisons, both shapes). B200 keeps its imbalance switch above.
# Rubin (compute capability (10, 7), 212 SMs) MXFP8 EP8 row 60 /
# 60 against the factory 40 / 40: EP8 on two nodes (4 GPUs each, one NVLink
# domain), H 4096 / I 2048 / 288 experts / top 8, configurations rotated on one
# workspace, four rounds, hottest-rank medians. Balanced routing, forward /
# backward: 8192 tokens per rank -7.4 / -4.6 %, 27116 -13.3 / -8.7 %, 32768
# -18.1 / -2.4 %; round spread at most 1.2 %. Rubin BF16 measured no consistent
# gain at EP8 (36 / 36 and 28 / 32 within the +-3 % round spread or slower).
#
# B300 ((10, 3), 148 SMs) BF16 EP8 row 24 / 40 against the factory 40 / 40: the
# forward count only (the backward keeps 40), a fixed split with no imbalance
# row, so every launch takes it whatever the routing of earlier launches (CUDA
# Graph replay, one workspace shared by layers of different skew). Chosen from
# the fixed counts 20, 24 and 28: the lowest forward geometric mean among the
# counts with no row slower than 40 / 40 in the measured configurations (backend
# README). 16 / 40 with an imbalance switch decided from an earlier observation
# measured slower under CUDA Graph replay and with shared workspaces. Like every
# row it applies to launches with routed weights of its precision (BF16): MXFP8
# launches on the same workspace take the MXFP8 key, which has no row on B300
# (40 / 40).
# B200 ((10, 0), 148 SMs) BF16 EP8 and GB200 ((10, 0), 152 SMs) BF16 EP4 keep
# 40 / 40: on B200 16 / 40 measured slower from 2.0x skew, and its imbalance
# split (24 / 28 from 4.0x) follows the workspace's precision, so it cannot
# replace the row for every workspace; on GB200 16 / 40 measured slower on its
# 16-expert top-8 configuration and 20 / 40 is not measured beyond the gating
# cells. Protocol, absolute times and the configurations not measured: backend
# README, "Fixed split per device and group size".
#
# GB300 BF16 EP4 keeps 20 / 24 (16 / 24 and 12 / 24 measured no gain).
COMM_SMS = {
    (10, 3): {
        148: {"bf16": {8: (24, 40)}},
        152: {
            "bf16": {
                4: (20, 24),
                8: (20, 24),
                16: (20, 24),
                32: (20, 24),
                64: (20, 24),
            },
            "mxfp8": {8: (44, 36), 32: (44, 36), 64: (44, 36)},
        },
    },
    (10, 7): {212: {"mxfp8": {8: (60, 60)}}},
}
_FACTORY_COMM_SMS = (40, 40)

# Fixed (forward, backward) split per launch shape for ``"auto"``: keyed like
# COMM_SMS (device compute capability, SM count, routed precision, EP group
# size), then by the launch's shape (hidden width H, routed intermediate width
# I, top-k). A forward or backward launch whose shape has a row takes its
# count instead of the COMM_SMS row or the factory split; every other launch,
# and every context recompute launch, keeps those. The split never changes a
# bit of any output, only time.
#
# GB300 ((10, 3), 152 SMs) BF16 EP8 56 / 48 for H 4096 / I 1024 / top-k 10
# (the 512-expert layout (8, 64)). Its COMM_SMS row (20 / 24) was chosen at
# I 2048 and top-k 8, where a routed row carries twice the expert GEMM work per
# byte it moves; at I 1024 the forward and backward wait on communication with
# that few SMs. Chosen by the checkpointed training step (schedule,
# context-free forward, context recompute, backward) from forward / backward
# counts measured on two 4-GPU trays of one NVLink domain, EP8, 32,768 tokens
# per rank, uniform routing and with rank 0's experts at 1.75x the mean routed
# rows, two rounds interleaved with the old row, the slowest rank's median.
# Checkpointed step, uniform / 1.75x (ms): 20 / 24 46.24 / 72.74, 48 / 48
# 33.91 / 53.72, 56 / 40 34.23 / 54.31, 56 / 48 33.49 / 53.59, 64 / 48
# 33.81 / 53.54. The context recompute launch keeps the COMM_SMS row: it
# rebuilds one macro batch, which slows as the forward count grows (0.818 ms at
# 20, 0.862 ms at 56). B300 ((10, 3), 148 SMs) keeps 24 / 40: every forward
# count measured above 24 (40 to 56, backward 40 or 48) made the checkpointed
# step faster but its backward slower, the faster forward leaving the GPU at a
# lower SM clock under its power limit (48 / 48, 32,768 tokens per rank,
# uniform: checkpointed step 41.97 -> 38.90 ms, backward 26.17 -> 27.14 ms).
# B200 ((10, 0), 148 SMs) keeps 40 / 40 (48 / 40 measured: checkpointed step
# within 0.7 % of 40 / 40) and GB200 ((10, 0), 152 SMs) keeps 40 / 40 (no other
# count measured).
# Absolute times against the previous version: backend README, "Fixed split
# per launch shape".
SHAPE_COMM_SMS = {
    (10, 3): {152: {"bf16": {8: {(4096, 1024, 10): (56, 48)}}}},
}

# Per-launch source split of the router-weight-before-down BF16 forward
# (``MoKConfig.source_split_comm_sms``): the forward communication-SM count of a
# rank whose source rows exceed ``SOURCE_SPLIT_THRESHOLD`` times the mean source
# rows of the ranks, keyed by device compute capability, SM count and EP group
# size. The forward kernel decides per launch from its own schedule (the
# owner-rank route totals every rank holds alike), so only a source-rich rank
# moves SMs from its communication program to its compute tiles, whose shared
# expert grows with its source rows while its routed rows match the other
# ranks'. Only devices and group sizes with a measured gain have a row; "auto"
# resolves to None (the split off) elsewhere. Protocol and absolute times:
# backend README, "Source split".
SOURCE_SPLIT_COMM_SMS = {
    (10, 0): {148: {8: 16}, 152: {4: 16}},
    (10, 3): {152: {4: 16}},
}
SOURCE_SPLIT_THRESHOLD = 1.5


def resolve_source_split_comm_sms(ep_size, device=None):
    """Forward communication-SM count of a source-rich rank measured for EP
    group size ``ep_size`` on ``device`` (:data:`SOURCE_SPLIT_COMM_SMS`), or
    ``None`` when the device (compute capability and SM count) or group size
    has no row (the source split stays off).
    """
    import torch

    ep_size = int(ep_size)
    if ep_size < 1:
        raise ValueError(f"ep_size must be a positive group size, got {ep_size}")
    if device is None:
        device = torch.cuda.current_device()
    capability = tuple(torch.cuda.get_device_capability(device))
    sm_count = int(torch.cuda.get_device_properties(device).multi_processor_count)
    return SOURCE_SPLIT_COMM_SMS.get(capability, {}).get(sm_count, {}).get(ep_size)


def resolve_imbalance_comm_sms(precision, device=None):
    """(forward, backward) communication-SM split used under imbalanced routing
    for ``precision`` (``"bf16"`` or ``"mxfp8"``) on ``device``, or ``None``
    when the device (compute capability and SM count) has no measured row (the
    fixed split is kept).
    """
    import torch

    if precision not in MOK_PRECISIONS:
        raise ValueError(
            f"precision must be one of {MOK_PRECISIONS}, got {precision!r}"
        )
    if device is None:
        device = torch.cuda.current_device()
    capability = tuple(torch.cuda.get_device_capability(device))
    sm_count = int(torch.cuda.get_device_properties(device).multi_processor_count)
    return IMBALANCE_COMM_SMS.get(capability, {}).get(sm_count, {}).get(precision)


def resolve_comm_sms(precision, ep_size, device=None):
    """Fixed (forward, backward) communication-SM split measured for
    ``precision`` (``"bf16"`` or ``"mxfp8"``) at EP group size ``ep_size`` on
    ``device``, or ``None`` when the device (compute capability and SM count),
    precision or group size has no measured row (the factory split is kept).
    """
    import torch

    if precision not in MOK_PRECISIONS:
        raise ValueError(
            f"precision must be one of {MOK_PRECISIONS}, got {precision!r}"
        )
    ep_size = int(ep_size)
    if ep_size < 1:
        raise ValueError(f"ep_size must be a positive group size, got {ep_size}")
    if device is None:
        device = torch.cuda.current_device()
    capability = tuple(torch.cuda.get_device_capability(device))
    sm_count = int(torch.cuda.get_device_properties(device).multi_processor_count)
    return (
        COMM_SMS.get(capability, {}).get(sm_count, {}).get(precision, {}).get(ep_size)
    )


def resolve_shape_comm_sms(precision, ep_size, device=None):
    """Fixed (forward, backward) splits per launch shape measured for
    ``precision`` (``"bf16"`` or ``"mxfp8"``) at EP group size ``ep_size`` on
    ``device`` (:data:`SHAPE_COMM_SMS`), as a new ``{(hidden, intermediate,
    topk): (forward, backward)}`` dict, or ``None`` when the device (compute
    capability and SM count), precision or group size has no shape row (every
    shape keeps the :func:`resolve_comm_sms` row or the factory split).
    """
    import torch

    if precision not in MOK_PRECISIONS:
        raise ValueError(
            f"precision must be one of {MOK_PRECISIONS}, got {precision!r}"
        )
    ep_size = int(ep_size)
    if ep_size < 1:
        raise ValueError(f"ep_size must be a positive group size, got {ep_size}")
    if device is None:
        device = torch.cuda.current_device()
    capability = tuple(torch.cuda.get_device_capability(device))
    sm_count = int(torch.cuda.get_device_properties(device).multi_processor_count)
    rows = (
        SHAPE_COMM_SMS.get(capability, {})
        .get(sm_count, {})
        .get(precision, {})
        .get(ep_size)
    )
    return dict(rows) if rows else None


def resolve_imbalance_threshold(precision, device=None):
    """Load ratio (hottest rank over the mean per-rank routed load) from which
    the imbalance split of ``precision`` (``"bf16"`` or ``"mxfp8"``) applies:
    the measured row of ``device`` in :data:`IMBALANCE_DEVICE_THRESHOLDS`
    (compute capability and SM count) when given and present, else the
    precision's :data:`IMBALANCE_THRESHOLDS` value.
    """
    if precision not in MOK_PRECISIONS:
        raise ValueError(
            f"precision must be one of {MOK_PRECISIONS}, got {precision!r}"
        )
    if device is not None:
        import torch

        capability = tuple(torch.cuda.get_device_capability(device))
        sm_count = int(torch.cuda.get_device_properties(device).multi_processor_count)
        threshold = (
            IMBALANCE_DEVICE_THRESHOLDS.get(capability, {})
            .get(sm_count, {})
            .get(precision)
        )
        if threshold is not None:
            return threshold
    return IMBALANCE_THRESHOLDS[precision]


def prepare_mok_bf16(
    *,
    ep_size=16,
    local_experts=16,
    topk=8,
    clamped_swiglu=False,
    fp32_wgrad=False,
    router_weight_position="after_down",
    router_weight_gradient="quotient",
    shared_expert="unscaled",
    activation="swiglu",
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

    ``router_weight_gradient`` (one of :data:`ROUTER_WEIGHT_GRADIENTS`),
    ``shared_expert`` (one of :data:`SHARED_EXPERT_MODES`) and ``activation``
    (one of :data:`ACTIVATIONS`) are explicit modes; the defaults
    (``"quotient"``, ``"unscaled"``, ``"swiglu"``) keep the behavior and every
    bit above. ``"quotient"`` computes the ``"after_down"`` router-weight
    gradient as ``dot(dh, h) / w`` with ``dh`` from ``BF16(w * dy)`` (zero at
    a zero weight); ``"direct"`` as ``dot(dh, h)`` with ``dh`` from the
    unscaled ``dy`` and the activation gradient ``w * dh`` (defined at a zero
    weight; the down-projection weight gradient keeps ``BF16(w * dy)``).
    ``shared_expert="scaled"`` multiplies the shared expert's output by a
    per-token FP32 scale ``s``: ``forward`` and ``backward`` take it as
    ``shared_scale`` (FP32 ``[tokens]``), ``y = s * S(x) + sum_k w_k E_k(x)``,
    and ``backward`` returns ``d_shared_scale = dot(dy, S(x))`` as a ninth
    item (defined at ``s = 0``). ``activation="clamped_swiglu"`` is
    ``clamped_swiglu=True``. ``activation="situ_glu"`` runs the routed
    activation ``beta tanh(gate / beta) sigmoid(gate) lb tanh(up / lb)``:
    ``forward``, ``recompute_forward_context`` and ``backward`` take
    ``situ_beta`` and ``situ_lb`` (finite positive floats, run-time values
    fixed inside a captured CUDA graph, as ``swiglu_limit``).
    ``shared_expert="none"`` has no shared expert,
    ``y = sum_k w_k E_k(x)``: every call takes ``None`` for the shared
    weights, the forward context's shared fields are ``None``, and
    ``backward`` takes the three routed FP32 accumulators (gate, up, down) and
    returns ``None`` for the three shared weight gradients.

    Prepare and warm up outside graph capture. Supported scheduler layouts
    (EP, local experts) are (1, 4) and (4, 4) for diagnostics (top-k 2 in
    both positions, top-k 8 with ``"after_down"``), 256 routed experts at EP
    4/8/16/32/64 and 288 routed experts at EP 8/32 with top-k 2 or 8
    (``"after_down"``), 384 routed experts at EP 4/8/16/32/64 ((4, 96),
    (8, 48), (16, 24), (32, 12), (64, 6)) with top-k 6 (``"before_down"``),
    512 routed experts at EP8 ((8, 64)) with top-k 10, which runs only
    with ``"after_down"``, plain SwiGLU, ``fp32_wgrad=True``,
    ``router_weight_gradient="direct"``, ``shared_expert="scaled"`` and BF16
    routed weights, and 896 routed experts at EP64 ((64, 14)) with top-k 16,
    which runs only with ``"after_down"``, ``activation="situ_glu"``,
    ``fp32_wgrad=True``, ``router_weight_gradient="direct"``,
    ``shared_expert="none"`` and BF16 routed weights; both on sm_100a and
    sm_103a (:data:`MODE_GROUPS`). The ``"direct"``, ``"scaled"``,
    ``"none"`` and ``"situ_glu"`` modes run with those configurations only.
    Any other combination raises before any kernel is loaded or launched:
    ``TypeError`` for a position, router-weight gradient, shared-expert mode
    or activation that is not a ``str``, an ``ep_size``, ``local_experts`` or
    ``topk`` that is not an ``int`` (a ``bool`` included) and a
    ``clamped_swiglu`` or ``fp32_wgrad`` that is not a ``bool``, in both
    positions; ``ValueError`` for every other value.
    Tokens, hidden/intermediate widths and per-rank source counts
    remain runtime values subject to the documented alignment constraints.
    This is a synthetic training experiment, with no full-model or optimizer
    integration.
    """
    from .backend import MoKFunctional

    return MoKFunctional(
        ep_size,
        local_experts,
        topk,
        clamped_swiglu=clamped_swiglu,
        fp32_wgrad=fp32_wgrad,
        router_weight_position=router_weight_position,
        router_weight_gradient=router_weight_gradient,
        shared_expert=shared_expert,
        activation=activation,
    )


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
    source_split_comm_sms="auto",
    source_split_threshold=1.5,
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
    ``mxfp8_comm_sms``), except a forward or backward launch whose shape (the
    workspace's ``hidden_size`` and ``topk``, the routed intermediate width)
    has a measured row of the device, routed precision and group size, which
    takes that row's count (:func:`resolve_shape_comm_sms`,
    :data:`SHAPE_COMM_SMS`, ``MoKConfig.shape_comm_sms``; BF16 H 4096 / I 1024
    / top-k 10 at EP8, the 512-expert layout (8, 64): GB300 56 / 48; a
    context recompute launch keeps the row above). The returned
    ``fwd_num_comm_sms`` / ``bwd_num_comm_sms`` hold the row of ``precision``;
    on such a config a ``dataclasses.replace`` of ``fwd_num_comm_sms`` /
    ``bwd_num_comm_sms`` alone changes no launch: create the workspace with
    integers, or replace ``bf16_comm_sms`` / ``mxfp8_comm_sms`` and
    ``shape_comm_sms`` too. A pair of integers sets the
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
    ``source_split_comm_sms`` is the forward communication-SM count of a rank
    whose source rows exceed ``source_split_threshold`` times the mean source
    rows of the ranks, for the router-weight-before-down BF16 forward: its
    kernel decides per launch on the device from that launch's schedule, so
    CUDA Graph replays and layers that share a workspace decide from their
    own routing, and only such a rank changes its count (when the count is
    below the forward count in force). ``"auto"`` (default) takes the measured
    row of the device and EP group size (:func:`resolve_source_split_comm_sms`,
    :data:`SOURCE_SPLIT_COMM_SMS`) and ``None`` where there is none; ``None``
    keeps the forward count and an even count sets it. The split changes time
    only, never an output bit.
    """
    from .workspace import MoKConfig
    from .backend import create_source_workspace

    if precision not in MOK_PRECISIONS:
        raise ValueError(
            f"precision must be one of {MOK_PRECISIONS}, got {precision!r}"
        )
    # Validate every argument before resolving "auto" (which reads the group
    # size and the device), so malformed arguments fail before any collective.
    auto = (fwd_num_comm_sms == "auto", bwd_num_comm_sms == "auto")
    if any(auto) and not all(auto):
        raise ValueError(
            'fwd_num_comm_sms and bwd_num_comm_sms must both be "auto" or both '
            "be communication-SM counts"
        )
    if isinstance(imbalance_comm_sms, str):
        if imbalance_comm_sms != "auto":
            raise ValueError(
                'imbalance_comm_sms must be "auto", None or a (forward, backward) pair'
            )
    elif imbalance_comm_sms is not None:
        imbalance_comm_sms = tuple(imbalance_comm_sms)
        if len(imbalance_comm_sms) != 2:
            raise ValueError(
                "imbalance_comm_sms must be a (forward, backward) pair of counts"
            )
    if isinstance(imbalance_threshold, str) and imbalance_threshold != "auto":
        raise ValueError('imbalance_threshold must be "auto" or a number')
    if isinstance(source_split_comm_sms, str) and source_split_comm_sms != "auto":
        raise ValueError('source_split_comm_sms must be "auto", None or a count')
    routed_comm_sms = {}
    shape_comm_sms = None
    if all(auto):
        import torch.distributed as dist

        ep_size = dist.get_world_size(group)
        routed_comm_sms = {
            routed: resolve_comm_sms(routed, ep_size, device) or _FACTORY_COMM_SMS
            for routed in MOK_PRECISIONS
        }
        fwd_num_comm_sms, bwd_num_comm_sms = routed_comm_sms[precision]
        shape_comm_sms = (
            tuple(
                (routed, *shape, *split)
                for routed in MOK_PRECISIONS
                for shape, split in sorted(
                    (resolve_shape_comm_sms(routed, ep_size, device) or {}).items()
                )
            )
            or None
        )
    if isinstance(imbalance_comm_sms, str):
        imbalance_comm_sms = resolve_imbalance_comm_sms(precision, device)
    if isinstance(imbalance_threshold, str):
        import torch

        imbalance_threshold = resolve_imbalance_threshold(
            precision, torch.cuda.current_device() if device is None else device
        )
    if isinstance(source_split_comm_sms, str):
        import torch.distributed as dist

        source_split_comm_sms = resolve_source_split_comm_sms(
            dist.get_world_size(group), device
        )
    config = MoKConfig(
        fwd_num_comm_sms=fwd_num_comm_sms,
        bwd_num_comm_sms=bwd_num_comm_sms,
        imbalance_comm_sms=imbalance_comm_sms,
        imbalance_threshold=imbalance_threshold,
        minibatch_size=minibatch_size,
        macrobatch_size=macrobatch_size,
        schedule_capacity_multiplier=schedule_capacity_multiplier,
        source_split_comm_sms=source_split_comm_sms,
        source_split_threshold=source_split_threshold,
        bf16_comm_sms=routed_comm_sms.get("bf16"),
        mxfp8_comm_sms=routed_comm_sms.get("mxfp8"),
        shape_comm_sms=shape_comm_sms,
    )
    return config, create_source_workspace(
        config,
        group,
        device=device,
        num_local_tokens=num_local_tokens,
        hidden_size=hidden_size,
        topk=topk,
        source_capacity=source_capacity,
    )


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
    The weights are quantized with PyTorch operations on the current stream
    (``mxfp8.py``); callers may pass their own MXFP8 weights in the same layout.
    """
    from .mxfp8 import quantize_routed_weights

    return quantize_routed_weights(gate, up, down)
