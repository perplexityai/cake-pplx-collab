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

Tests of the backend of the non-deterministic Cake packed width-4 depthwise causal convolution + SiLU training
package (``cake_pplx.cake_causal_conv1d_nondet_train``, backend
``cake_pplx._backends.cake_causal_conv1d_nondet_train``; SM100 / SM103 / SM107).

Every floating output (``y``, ``dx``, ``dweight``, ``dbias``) has no tolerance: it is judged against the naive
BF16 tier (``tests.cake.test_helpers.cake_naive``; PyTorch in BF16 on the same inputs) over 32 independent random
inputs of each case, by its error against the float64 oracle of the function (SiLU' at the unrounded preactivation
on every route and policy), in the L2 and the max norm: the median and the largest of its errors over the draws at
most 1.005 times the tier's, and 0 wherever the tier's is 0 (``tests.cake.test_helpers.cake_causal_conv1d_oracle``).
The package is not deterministic: no bit rule holds its outputs, and the suite never compares it with the
deterministic package.  ``REPEATS`` launches on each draw's inputs are all judged (the larger error of the launches
of a draw counts), and so are the fp32 policy's ``PRECISION_REPEATS`` launches at the perf size.

The workspace contract: a caller-owned zeroed workspace and the package's per-(device, stream) workspace read
back all zero after every launch, over changed shapes and routes; NaN-poisoned outputs are fully overwritten;
CUDA Graph capture / replay with and without a caller-owned workspace (each replay on a draw's inputs, judged),
one package buffer per capture and stream shared by the capture's backward calls; two streams with two workspaces at once;
the backward is ONE ordinary launch (no reduce program) and keeps its speed beside a kernel resident on another
stream.  Autograd, non-reentrant activation checkpointing (2 x forward +
backward) and gradient accumulation; torch's deterministic mode.

The opt-in ``bwd_grid_mult`` (every value of ``BWD_GRID_MULTS``) on every route, at single-tile, multi-sub-tile and
partial-last-tile channel counts and at small and large ``S``: the backward launches that many times its untrimmed
grid, its outputs are judged against the naive tier, the workspace (the same size at every multiple) reads back zero
after every launch, under CUDA Graph capture / replay and through the autograd entry (plain and checkpointed).

The narrower channel tiles of ``TILE_OVERRIDES``: each row of ``OVERRIDE_CASES`` (one per architecture, on sm_107a on
both routes that plan by it) launches the row's
tiles, on a caller-owned workspace of the public size and on the package's, with every output judged against the
naive tier and the workspace zero after the call.  The 32-vector tiles of ``NARROW_TILES`` the same way at every key of the architecture's table
(``NARROW_CASES``: the key's channel ends and an inner channel count, at token counts inside the key's range), and
the side-stream one-SM trim of a three-tile narrow row's grid.

This suite names its package once, through ``tests.cake.test_helpers.cake_package`` (key
``cake_causal_conv1d_nondet_train``).  Every test skips until the exporter has rendered the package (``build.json``
and ``csrc/``).
"""

from __future__ import annotations

import gc
import statistics
import weakref

import pytest
import torch

from tests.cake.test_helpers.cake_causal_conv1d_oracle import (
    assert_judged,
    naive_bf16,
    oracle,
    problem,
)
from tests.cake.test_helpers.cake_naive import (
    DRAWS,
    TierJudge,
    contract,
    draw_seed,
    seed_all,
)
from tests.cake.test_helpers.cake_package import cake_package

KEY = "cake_causal_conv1d_nondet_train"
pkg = cake_package(KEY)
cake_build = cake_package(KEY, "cake_build")
loader = cake_package(KEY, "cake_jit")

SUPPORTED_CAPABILITIES = ((10, 0), (10, 3), (10, 7))
WIDTH = 4
F32, BF16 = torch.float32, torch.bfloat16
PERF_FRACTIONS = (0.37, 0.29, 0.21)
SEED = 8065
# Launches per draw of the repeated-launch rows (every launch is judged).
REPEATS = 20

# (S, C, documents): one-token and short documents, empty documents before / between / after others, boundaries
# inside a sub-tile and at 64 / 65, the perf packing, 38 documents, the channel thresholds of the routed programs,
# the multi-sub-tile tiles (1280, 1536, 2560) and the partial last channel tiles (1160, 1344, 1600).
CASES = (
    (1, 128, None),
    (5, 136, (2, 0, 3)),
    (63, 384, (1, 2, 3, 57)),
    (64, 128, None),
    (65, 256, (64, 1)),
    (1000, 512, "perf"),
    (1000, 256, (27,) * 37 + (1,)),
    (4097, 1024, (0, 4097, 0)),
    (4097, 2048, (1000, 0, 3, 3094)),
    (4097, 768, (1000, 0, 3, 3094)),
    (5, 512, (2, 0, 3)),
    (5, 1024, (2, 0, 3)),
    (4097, 1280, (1000, 0, 3, 3094)),
    (5, 1280, (2, 0, 3)),
    (5, 1536, (2, 0, 3)),
    (4097, 2560, (1000, 0, 3, 3094)),
    (4097, 1160, (1000, 0, 3, 3094)),
    (4097, 1344, (1000, 0, 3, 3094)),
    (1000, 1600, "perf"),
)
ROUTE_CASES = (
    (1000, 512, "perf"),
    (4097, 1280, (1000, 0, 3, 3094)),
    (5, 1536, (2, 0, 3)),
    (65, 256, (64, 1)),
    (4097, 1344, (1000, 0, 3, 3094)),
)
# A large row per architecture class for the repeated launches (many CTAs per channel tile race on the accumulators).
REPEAT_CASES = (
    (65536, 512, "perf"),
    (65536, 1280, "perf"),
    (4097, 1160, (1000, 0, 3, 3094)),
)
PRECISION_CASES = ((250340, 512, "perf"), (250340, 1280, "perf"))
# Rows of ``TILE_OVERRIDES`` per architecture (route, S, C): the backward runs its narrower channel tiles there.  The
# sm_107a ``bwd_dt`` row serves both routes that plan by it: ``bwd_dt_bf16_taps`` (BF16 taps, through
# ``TILE_ROUTE``) and ``bwd_dt`` (FP32 taps with a BF16 ``dweight`` out).
OVERRIDE_CASES = {
    "sm_100a": (("bwd_bias_fp32g", 62585, 2048),),
    "sm_103a": (("bwd", 62585, 1536),),
    "sm_107a": (("bwd_dt_bf16_taps", 62585, 768), ("bwd_dt", 62585, 768)),
}
# The rows of ``NARROW_TILES`` (rendered with the table; empty for an architecture without rows) the narrow-tile
# test launches, per architecture (route, S, C): each key's lowest, an inner and its highest channel count at the
# key's smallest token count and one token above it (inside the delivered range; ``CASES`` stop at S = 4097), on the
# key's route and on every route that plans by its rows (``TILE_ROUTE``).
NARROW_CASES = {
    arch: tuple(
        (route, S, C)
        for (base, c_lo, c_hi), (s_min, s_max) in sorted(rows.items())
        for route in (base, *sorted(t for t, b in loader.TILE_ROUTE.items() if b == base))
        for C in sorted({int(c_lo), (int(c_lo) + int(c_hi)) // 16 * 8, int(c_hi)})
        for S in sorted({int(s_min), min(int(s_min) + 1, int(s_max))})
    )
    for arch, rows in loader.NARROW_TILES.items()
}
# Wide channel counts on every route: C = 4096 (four documents, one empty) and C = 10240 (a packed q / k / v
# projection width), at small S.
WIDE_CASES = ((4097, 4096, (1000, 0, 3, 3094)), (2048, 10240, "perf"))
# Launches per draw of the fp32 policy's perf-size rows (every launch is judged).
PRECISION_REPEATS = 5
# S x C > 2^31 elements (64-bit offsets) at a token count past the sm_107a token floor.
LARGE_CASE = (262401, 8192, "perf")
# The large case's inputs and outputs (x, dy, y, dx: 4 x 4.3 GB), its float64 oracle of y and dx (2 x 17.2 GB) and
# the naive tier's BF16 y and dx with their blocks (about 4 x 4.3 GB), with headroom.
LARGE_CASE_FREE_BYTES = 96 * 2**30
GRAPH_CASES = (
    (65, 256, (64, 1)),
    (5, 136, (2, 0, 3)),
    (1000, 512, "perf"),
    (4097, 1280, (1000, 0, 3, 3094)),
    (4097, 2560, (1000, 0, 3, 3094)),
)
# The ``bwd_grid_mult`` rows: fewer tokens than CTAs (S = 5, 65), a single-tile C, the multi-sub-tile tiles (1280,
# 1536, 2560) and the partial last channel tiles (136, 1160, 1344), at small S and at large S (65536, the perf size).
GRID_MULT_CASES = (
    (5, 136, (2, 0, 3)),
    (65, 256, (64, 1)),
    (5, 1536, (2, 0, 3)),
    (4097, 1344, (1000, 0, 3, 3094)),
    (4097, 2560, (1000, 0, 3, 3094)),
    (65536, 1160, "perf"),
    (65536, 1280, "perf"),
    (250340, 512, "perf"),
)
# The call settings that resolve to each backward route.  ``dweight_dtype`` is the dtype of the preallocated
# ``dweight`` out a call of the route passes (``None``: none, the taps' dtype): ``bwd_dt`` is FP32 taps with a BF16
# ``dweight`` out (:func:`_backward` with ``kind`` allocates a fresh one per call).
ROUTE_SETTINGS = {
    "bwd": dict(weight_dtype=F32, bias_dtype=None, policy="bf16", dweight_dtype=None),
    "bwd_dt": dict(weight_dtype=F32, bias_dtype=None, policy="bf16", dweight_dtype=BF16),
    "bwd_bias": dict(weight_dtype=F32, bias_dtype=F32, policy="bf16", dweight_dtype=None),
    "bwd_fp32g": dict(weight_dtype=F32, bias_dtype=None, policy="fp32", dweight_dtype=None),
    "bwd_bias_fp32g": dict(weight_dtype=F32, bias_dtype=F32, policy="fp32", dweight_dtype=None),
    "bwd_dt_bf16_taps": dict(weight_dtype=BF16, bias_dtype=None, policy="bf16", dweight_dtype=None),
    "bwd_bias_bf16_taps": dict(weight_dtype=BF16, bias_dtype=BF16, policy="bf16", dweight_dtype=None),
}
# The backward routes the autograd entry reaches: it returns ``dweight`` in the taps' dtype, so ``bwd_dt`` (FP32 taps
# with a BF16 ``dweight`` out) is a route of direct backward calls only.
AUTOGRAD_KINDS = tuple(
    kind for kind, settings in ROUTE_SETTINGS.items() if settings["dweight_dtype"] is None
)
FORWARD_OF = {
    "bwd": "fwd",
    "bwd_dt": "fwd",
    "bwd_bias": "fwd_bias",
    "bwd_fp32g": "fwd",
    "bwd_bias_fp32g": "fwd_bias",
    "bwd_dt_bf16_taps": "fwd_dt",
    "bwd_bias_bf16_taps": "fwd_bias",
}
# Every dtype combination a backward call takes: taps x bias (none, FP32, BF16) x SiLU' policy x the dweight /
# dbias out dtypes (each FP32 or BF16, independently of its parameter).  Each resolves to one route: BF16 taps on
# the fp32g routes, BF16 taps under the "bf16" policy (bwd_dt_bf16_taps, bwd_bias_bf16_taps, with either dweight
# dtype) and FP32 taps with a BF16 dweight (bwd_dt, and bwd_bias with a bias) are among them.
DTYPE_COMBOS = tuple(
    (weight_dtype, bias_dtype, policy, dweight_dtype, dbias_dtype)
    for policy in ("bf16", "fp32")
    for weight_dtype in (F32, BF16)
    for bias_dtype in (None, F32, BF16)
    for dweight_dtype in (F32, BF16)
    for dbias_dtype in ((None,) if bias_dtype is None else (F32, BF16))
)
# The autograd entry returns the gradients in the parameters' dtypes: taps x bias x policy.
AUTOGRAD_COMBOS = tuple(
    (weight_dtype, bias_dtype, policy)
    for policy in ("bf16", "fp32")
    for weight_dtype in (F32, BF16)
    for bias_dtype in (None, F32, BF16)
)


def _device() -> torch.device:
    return torch.device("cuda:0")


def _capability() -> tuple[int, int]:
    return torch.cuda.get_device_capability(_device())


def _supported() -> bool:
    return (
        torch.cuda.is_available()
        and _capability() in SUPPORTED_CAPABILITIES
        and pkg.is_cake_causal_conv1d_nondet_train_supported(_device())
    )


pytestmark = pytest.mark.skipif(
    not _supported(),
    reason=(
        "the non-deterministic cake_causal_conv1d_nondet_train package needs an SM100 / SM103 / SM107 device and "
        "its rendered build.json / csrc"
    ),
)


@pytest.fixture(autouse=True, scope="module")
def warmup_jit():
    if _supported():
        arch = loader.arch_for_capability(_capability())
        for spec in loader.specs_for_arch(arch):
            spec.build_and_load()
    yield


def _case_id(case) -> str:
    S, C, docs = case
    tag = "one" if docs is None else ("perf" if docs == "perf" else f"{len(docs)}docs")
    return f"S{S}-C{C}-{tag}"


def _dtype_id(dtype) -> str:
    return {None: "none", F32: "f32", BF16: "bf16"}[dtype]


def _combo_id(combo) -> str:
    weight_dtype, bias_dtype, policy, *outs = combo
    label = f"w{_dtype_id(weight_dtype)}-b{_dtype_id(bias_dtype)}-{policy}"
    if outs:
        dweight_dtype, dbias_dtype = outs
        label += f"-dw{_dtype_id(dweight_dtype)}"
        if bias_dtype is not None:
            label += f"-db{_dtype_id(dbias_dtype)}"
    return label


def _combo_route(weight_dtype, bias_dtype, policy, dweight_dtype=None) -> str:
    """The backward route a call of these dtypes runs (``dweight_dtype`` ``None``: the taps' dtype)."""

    return loader.backward_route(
        bias=bias_dtype is not None,
        fp32_grad=policy == "fp32",
        bf16_taps=weight_dtype == BF16,
        bf16_dweight=(dweight_dtype or weight_dtype) == BF16,
    )


def _arch() -> str:
    return loader.arch_for_capability(_capability())


def _route_pending(arch: str, kind: str) -> str | None:
    if kind not in loader.ROUTES.get(arch, {}):
        return f"route {kind!r} is not exported for {arch}"
    libraries = cake_build.libraries_by_name()
    for name in loader.route_modules(arch, kind):
        library = loader.library_name(name, arch)
        if library not in libraries:
            return f"route {kind!r}: {library} is not in build.json yet (awaiting the exporter's render)"
    return None


def _require_route(*kinds: str) -> None:
    for kind in kinds:
        for needed in (kind, FORWARD_OF.get(kind, kind)):
            pending = _route_pending(_arch(), needed)
            if pending:
                pytest.skip(pending)


def _backward_thresholds(arch: str, kind: str) -> tuple[int, ...]:
    """The channel thresholds of the programs route ``kind`` selects by channel count on ``arch`` (empty when the
    route ships its fallback program only there)."""

    thresholds = set()
    for name in loader.route_modules(arch, kind):
        rule = loader.select_rule(loader.MODULES[name])
        if rule is not None and arch in rule["channels"]:
            thresholds.add(int(rule["channels"][arch]))
    return tuple(sorted(thresholds))


def _token_floors(arch: str, kind: str) -> dict[str, tuple[int, int]]:
    """The programs of route ``kind`` on ``arch`` whose rule carries a token floor: name -> (channel threshold,
    token floor)."""

    floors = {}
    for name in loader.route_modules(arch, kind):
        rule = loader.select_rule(loader.MODULES[name])
        if rule is not None and arch in rule.get("tokens", {}):
            floors[name] = (int(rule["channels"][arch]), int(rule["tokens"][arch]))
    return floors


def _generator(*keys: int, seed: int = SEED) -> torch.Generator:
    """A generator of the device seeded from ``seed`` (a draw's, or :data:`SEED`) and ``keys``."""

    for key in keys:
        seed = (seed * 1_000_003 + int(key)) % (2**31 - 1)
    return torch.Generator(device=_device()).manual_seed(seed)


def doc_lengths(S: int, docs) -> tuple[int, ...]:
    if docs is None:
        return (S,)
    if docs == "perf":
        lens = []
        for fraction in PERF_FRACTIONS:
            n = int(fraction * S)
            if n % 64 == 0:
                n -= 1
            lens.append(n)
        lens.append(S - sum(lens))
        assert all(n > 0 for n in lens), f"S={S} is too short for the perf packing"
        return tuple(lens)
    lens = tuple(int(n) for n in docs)
    assert sum(lens) == S and all(n >= 0 for n in lens)
    return lens


def _cu_seqlens(S: int, docs) -> torch.Tensor:
    lens = doc_lengths(S, docs)
    return torch.tensor(
        [0, *torch.tensor(lens).cumsum(0).tolist()], dtype=torch.int32, device=_device()
    )


def _inputs(S: int, C: int, docs, *, key: int = 0, weight_dtype=F32, seed: int = SEED):
    """``(x, w, cu, dy)``: BF16 ``x`` / ``dy`` of N(0, 1), taps of N(0, 0.25) in ``weight_dtype``, drawn from
    ``seed`` (a draw's seed; :data:`SEED` is draw 0's)."""

    generator = _generator(S, C, key, seed=seed)
    x = torch.randn(
        S, C, dtype=torch.float32, device=_device(), generator=generator
    ).to(BF16)
    w = 0.5 * torch.randn(
        C, WIDTH, dtype=torch.float32, device=_device(), generator=generator
    )
    dy = torch.randn(
        S, C, dtype=torch.float32, device=_device(), generator=generator
    ).to(BF16)
    return x, w.to(weight_dtype), _cu_seqlens(S, docs), dy


def _bias_input(C: int, dtype, *, key: int = 0, seed: int = SEED) -> torch.Tensor:
    generator = _generator(C, 97, key, seed=seed)
    return (
        0.5 * torch.randn(C, dtype=torch.float32, device=_device(), generator=generator)
    ).to(dtype)


def _route_inputs(S: int, C: int, docs, kind: str, *, key: int = 0):
    """``(x, w, cu, dy, b, policy)`` resolving to backward route ``kind``."""

    settings = ROUTE_SETTINGS[kind]
    x, w, cu, dy = _inputs(S, C, docs, key=key, weight_dtype=settings["weight_dtype"])
    b = (
        None
        if settings["bias_dtype"] is None
        else _bias_input(C, settings["bias_dtype"], key=key)
    )
    return x, w, cu, dy, b, settings["policy"]


def _forward(x, w, cu, **kwargs) -> torch.Tensor:
    return pkg.cake_causal_conv1d_nondet_train_forward(x, w, cu, **kwargs)


def _backward(x, w, cu, dy, b=None, policy="bf16", *, kind=None, **kwargs):
    """``(dx, dweight, dbias_or_None)`` of one non-deterministic backward call.  With ``kind`` (a backward route of
    :data:`ROUTE_SETTINGS`) and no ``dweight`` out passed, the call writes a fresh NaN-filled ``dweight`` out of the
    route's ``dweight_dtype`` when it names one (``bwd_dt``), and the call's arguments must resolve to ``kind``."""

    if kind is not None:
        dweight_dtype = ROUTE_SETTINGS[kind]["dweight_dtype"]
        if kwargs.get("dweight") is None and dweight_dtype is not None:
            kwargs["dweight"] = torch.full(
                (w.shape[0], WIDTH), float("nan"), dtype=dweight_dtype, device=w.device
            )
        dweight = kwargs.get("dweight")
        route = _combo_route(
            w.dtype, None if b is None else b.dtype, policy, None if dweight is None else dweight.dtype
        )
        assert route == kind, f"the call resolves to route {route}, not {kind}"
    grads = pkg.cake_causal_conv1d_nondet_train_backward(
        x,
        w,
        cu,
        dy,
        bias=b,
        silu_grad_preactivation=policy,
        **kwargs,
    )
    return grads[0], grads[1], (grads[2] if b is not None else None)


def _grads(dx, dw, db) -> dict:
    """The gradients of one backward call by output name (``dbias`` only with a bias)."""

    out = {"dx": dx, "dweight": dw}
    if db is not None:
        out["dbias"] = db
    return out


def _forward_backward(x, w, cu, dy, b, policy, **kwargs) -> dict:
    """``y`` of one forward call and the gradients of one backward call, by output name."""

    y = _forward(x, w, cu, bias=b)
    return {"y": y, **_grads(*_backward(x, w, cu, dy, b, policy, **kwargs))}


def _judge(
    S: int,
    C: int,
    docs,
    call,
    *,
    weight_dtype=F32,
    bias_dtype=None,
    policy: str = "bf16",
    key: int = 0,
    label: str = "",
    perturb=None,
) -> None:
    """One problem judged over the draws: each draw's inputs (``weight_dtype`` taps, a ``bias_dtype`` bias or none;
    ``perturb(x, dy)`` returns the draw's ``x`` and ``dy`` changed) and ``call(x, w, cu, dy, b, policy)``, which
    launches and returns the outputs by name (a value, or a list of the values of several launches on the draw)."""

    def launch(_draw: int, seed: int):
        x, w, cu, dy = _inputs(S, C, docs, key=key, weight_dtype=weight_dtype, seed=seed)
        b = None if bias_dtype is None else _bias_input(C, bias_dtype, key=key, seed=seed)
        if perturb is not None:
            x, dy = perturb(x, dy)
        return call(x, w, cu, dy, b, policy), problem(x, w, cu, b, dy)

    assert_judged(
        launch,
        seed=SEED,
        label=label
        or f"S={S} C={C} docs={docs} w={_dtype_id(weight_dtype)} b={_dtype_id(bias_dtype)} {policy}",
    )


def _route_judge(S: int, C: int, docs, kind: str, call, **kwargs) -> None:
    """:func:`_judge` with the taps / bias dtypes and the policy that resolve to backward route ``kind`` (``call``
    passes ``kind=kind`` to :func:`_backward`, which writes the route's ``dweight`` out)."""

    settings = ROUTE_SETTINGS[kind]
    _judge(
        S,
        C,
        docs,
        call,
        weight_dtype=settings["weight_dtype"],
        bias_dtype=settings["bias_dtype"],
        policy=settings["policy"],
        **kwargs,
    )


def _stream_workspace(device: torch.device) -> torch.Tensor | None:
    """The package's backward workspace of the device's current stream (``None`` before its first call)."""

    key = (device.index, int(torch.cuda.current_stream(device).cuda_stream))
    return pkg._BACKWARD_WORKSPACES.get(key)


# ----------------------------------------------------------------------------------------------------------------
# Accuracy against the naive tier.


@pytest.mark.parametrize("case", CASES, ids=_case_id)
def test_forward_backward_match_reference(case) -> None:
    S, C, docs = case

    def call(x, w, cu, dy, b, policy):
        y = _forward(x, w, cu)
        assert y.dtype == BF16 and tuple(y.shape) == tuple(x.shape) and y.is_contiguous()
        dx, dw, _ = _backward(x, w, cu, dy)
        assert dx.dtype == BF16 and dw.dtype == F32 and tuple(dw.shape) == (C, WIDTH)
        return {"y": y, **_grads(dx, dw, None)}

    _judge(S, C, docs, call, label=str(case))


@pytest.mark.parametrize("combo", DTYPE_COMBOS, ids=_combo_id)
@pytest.mark.parametrize("case", ROUTE_CASES, ids=_case_id)
def test_routes_match_reference(case, combo) -> None:
    """Every backward route in every dtype combination (FP32 / BF16 taps, no / FP32 / BF16 bias, both SiLU'
    policies, FP32 / BF16 ``dweight`` and ``dbias`` outs): ``y``, ``dx``, ``dweight`` and ``dbias`` judged against
    the naive tier, the gradients written into the caller's outs."""

    weight_dtype, bias_dtype, policy, dweight_dtype, dbias_dtype = combo
    kind = _combo_route(weight_dtype, bias_dtype, policy, dweight_dtype)
    _require_route(kind)
    S, C, docs = case
    device = _device()
    label = f"{kind} {_combo_id(combo)} {case}"

    def call(x, w, cu, dy, b, policy):
        y = _forward(x, w, cu, bias=b)
        out = {
            "dweight": torch.empty(C, WIDTH, dtype=dweight_dtype, device=device),
            "dbias": None
            if b is None
            else torch.empty(C, dtype=dbias_dtype, device=device),
        }
        dx, dw, db = _backward(x, w, cu, dy, b, policy, **out)
        assert dw is out["dweight"] and db is out["dbias"], label
        return {"y": y, **_grads(dx, dw, db)}

    _judge(
        S,
        C,
        docs,
        call,
        weight_dtype=weight_dtype,
        bias_dtype=bias_dtype,
        policy=policy,
        key=1,
        label=label,
    )


# BF16 taps without a bias (bwd_dt_bf16_taps) and with a BF16 bias (bwd_bias_bf16_taps) under the "bf16" policy, at
# keys whose draws put dx past the naive tier when SiLU' is evaluated at the BF16-rounded preactivation (C 512 runs
# other programs on sm_107a than on sm_100a / sm_103a; each architecture misses on some of these keys).
BF16_TAPS_KEYS = (
    ("bwd_dt_bf16_taps", None, 33),
    ("bwd_bias_bf16_taps", BF16, 11),
    ("bwd_bias_bf16_taps", BF16, 56),
)


@pytest.mark.parametrize(
    "kind, bias_dtype, key",
    BF16_TAPS_KEYS,
    ids=[f"{kind}-b{_dtype_id(bias)}-key{key}" for kind, bias, key in BF16_TAPS_KEYS],
)
def test_bf16_taps_routes_match_reference(kind, bias_dtype, key) -> None:
    """BF16 taps under the "bf16" policy at S = 1000, C = 512 ("perf" documents): ``y``, ``dx``, ``dweight`` and
    ``dbias`` judged against the naive tier."""

    _require_route(kind)

    def call(x, w, cu, dy, b, policy):
        y = _forward(x, w, cu, bias=b)
        dx, dw, db = _backward(x, w, cu, dy, b, policy, kind=kind)
        return {"y": y, **_grads(dx, dw, db)}

    _judge(1000, 512, "perf", call, weight_dtype=BF16, bias_dtype=bias_dtype, policy="bf16", key=key,
           label=f"{kind} BF16 taps key {key}")


@pytest.mark.parametrize("kind", tuple(ROUTE_SETTINGS))
@pytest.mark.parametrize("case", WIDE_CASES, ids=_case_id)
def test_wide_channel_rows(case, kind) -> None:
    """C = 4096 and C = 10240 on every backward route (the fp32g routes included): ``y``, ``dx``, ``dweight`` and
    ``dbias`` judged against the naive tier."""

    _require_route(kind)
    S, C, docs = case

    def call(x, w, cu, dy, b, policy):
        return _forward_backward(x, w, cu, dy, b, policy, kind=kind)

    _route_judge(S, C, docs, kind, call, key=3, label=f"{kind} {case}")


@pytest.mark.parametrize("kind", tuple(ROUTE_SETTINGS))
def test_backward_programs_switch_on_channel_count(kind, monkeypatch) -> None:
    """Exactly one program of the route runs per call and every program the route ships on this architecture runs
    for some ``(S, C)``: each is judged against the naive tier on both sides of its channel threshold (and of its
    token floor, where its rule has one) on a caller-owned workspace that reads back zero after every launch."""

    _require_route(kind)
    arch, device = _arch(), _device()
    names = loader.route_modules(arch, kind)
    thresholds = _backward_thresholds(arch, kind)
    channels = sorted({C for T in thresholds for C in (T - 8, T, T + 8)} | {8192})
    rows = [
        (S, docs, C) for S, docs in ((5, (2, 0, 3)), (1000, "perf")) for C in channels
    ]
    # A token-floored program runs from its floor only: both sides of the floor at its channel threshold.
    for T, floor in _token_floors(arch, kind).values():
        rows += [(S, "perf", C) for S in (floor - 8, floor) for C in (T - 8, T, T + 8)]
    seen = []
    launch = pkg.run_module

    def record_launch(name, values, *, device_index, grid=None):
        seen.append(name)
        return launch(name, values, device_index=device_index, grid=grid)

    monkeypatch.setattr(pkg, "run_module", record_launch)
    launched = set()
    for S, docs, C in rows:
        (selected,) = loader.selected_modules(arch, kind, C=C, S=S)
        label = f"{kind} S={S} C={C} docs={docs} program={selected}"
        workspace = pkg.cake_causal_conv1d_nondet_train_backward_workspace(S, C, device)

        def call(x, w, cu, dy, b, policy, workspace=workspace, selected=selected, label=label):
            seen.clear()
            dx, dw, db = _backward(x, w, cu, dy, b, policy, kind=kind, workspace=workspace)
            assert seen == [selected], f"{label}: launched {seen}"
            assert int(torch.count_nonzero(workspace)) == 0, label
            return _grads(dx, dw, db)

        _route_judge(S, C, docs, kind, call, key=6, label=label)
        launched.add(selected)
    assert launched == set(names), f"never launched {set(names) - launched}"


@pytest.mark.parametrize("kind", tuple(ROUTE_SETTINGS))
def test_large_case_over_two_to_the_31_elements(kind) -> None:
    """``S x C > 2^31`` elements (64-bit offsets) at ``C`` 8192: ``y``, ``dx``, ``dweight`` and ``dbias`` judged
    against the naive tier, the workspace left zero."""

    _require_route(kind)
    free, _total = torch.cuda.mem_get_info(_device())
    if free < LARGE_CASE_FREE_BYTES:
        pytest.skip(
            f"needs {LARGE_CASE_FREE_BYTES >> 30} GiB free, {free >> 30} GiB available"
        )
    S, C, docs = LARGE_CASE
    assert S * C > 2**31
    device = _device()
    label = f"{kind} {_case_id(LARGE_CASE)}"
    workspace = pkg.cake_causal_conv1d_nondet_train_backward_workspace(S, C, device)

    def call(x, w, cu, dy, b, policy):
        y = _forward(x, w, cu, bias=b)
        dx, dw, db = _backward(x, w, cu, dy, b, policy, kind=kind, workspace=workspace)
        assert int(torch.count_nonzero(workspace)) == 0, label
        return {"y": y, **_grads(dx, dw, db)}

    _route_judge(S, C, docs, kind, call, key=5, label=label)
    torch.cuda.empty_cache()


@pytest.mark.parametrize("kind", tuple(ROUTE_SETTINGS))
@pytest.mark.parametrize("case", REPEAT_CASES, ids=_case_id)
def test_repeated_launches_worst_case(case, kind, record_property) -> None:
    """``REPEATS`` launches on each draw's inputs into one workspace: every launch's ``dx``, ``dweight`` and
    ``dbias`` judged against the naive tier (the worst launch of a draw counts), the workspace zero after every
    launch; the number of distinct ``dweight`` bit patterns of the first draw is recorded (the order of the
    cross-CTA sums may vary; no bound is placed on that count)."""

    _require_route(kind)
    S, C, docs = case
    workspace = pkg.cake_causal_conv1d_nondet_train_backward_workspace(S, C, _device())
    patterns = set()
    draws = []

    def call(x, w, cu, dy, b, policy):
        found = {"dx": [], "dweight": [], "dbias": []}
        for repeat in range(REPEATS):
            dx, dw, db = _backward(x, w, cu, dy, b, policy, kind=kind, workspace=workspace)
            assert int(torch.count_nonzero(workspace)) == 0, (
                f"the workspace is not zero after repeat {repeat}"
            )
            for name, value in _grads(dx, dw, db).items():
                found[name].append(value)
            if not draws:
                patterns.add(bytes(dw.contiguous().view(torch.uint8).cpu().numpy()))
        draws.append(len(draws))
        return {name: values for name, values in found.items() if values}

    _route_judge(S, C, docs, kind, call, key=3, label=f"{kind} {case}")
    print(f"{kind} S={S} C={C}: {REPEATS} launches per draw, {len(patterns)} distinct dweight bit patterns")
    record_property("repeats", REPEATS)
    record_property("distinct_dweight_patterns", len(patterns))


@pytest.mark.parametrize("own_workspace", (True, False), ids=("own-ws", "pkg-ws"))
def test_tile_override_row(own_workspace, monkeypatch, record_property) -> None:
    """Every ``OVERRIDE_CASES`` row of the architecture's ``TILE_OVERRIDES`` launches the row's narrower channel tiles
    (``ct_vec``, ``threads // ct_vec`` sub-tiles, ``ceil(C / vec / ct_vec)`` tiles) on a caller-owned workspace of
    the public size, or the package's: ``dx``, ``dweight`` and ``dbias`` judged against the naive tier, the
    workspace zero after every call."""

    arch, device = _arch(), _device()
    rows = OVERRIDE_CASES[arch]
    for kind, _, _ in rows:
        _require_route(kind)
    seen = []
    launch = pkg.run_module
    split_keys = ("ct_vec", "n_sub", "n_ch_tiles")

    def record_launch(name, values, *, device_index, grid=None):
        seen.append((name, {key: int(values[key]) for key in split_keys}))
        return launch(name, values, device_index=device_index, grid=grid)

    monkeypatch.setattr(pkg, "run_module", record_launch)
    for kind, S, C in rows:
        ct_vec = loader.tile_override(arch, kind, C, S)
        assert ct_vec is not None, f"{arch} {kind} C={C} S={S} is no TILE_OVERRIDES row"
        (name,) = loader.selected_modules(arch, kind, C=C, S=S)
        geometry = loader.MODULES[name]["grid_rule"]["geometry"]
        wanted = {
            "ct_vec": ct_vec,
            "n_sub": geometry["threads"] // ct_vec,
            "n_ch_tiles": -(-(C // geometry["vec"]) // ct_vec),
        }
        assert wanted != dict(
            zip(
                split_keys,
                loader.channel_split(C, geometry=geometry, role="bwd"),
                strict=True,
            )
        ), "the row must differ from the default split"
        label = f"{arch} {kind} S={S} C={C} ct_vec={ct_vec}"
        workspace = (
            pkg.cake_causal_conv1d_nondet_train_backward_workspace(S, C, device)
            if own_workspace
            else None
        )

        def call(x, w, cu, dy, b, policy, kind=kind, name=name, wanted=wanted, label=label, workspace=workspace):
            seen.clear()
            dx, dw, db = _backward(x, w, cu, dy, b, policy, kind=kind, workspace=workspace)
            assert seen == [(name, wanted)], f"{label}: launched {seen}"
            held = workspace if own_workspace else _stream_workspace(device)
            assert held is not None and int(torch.count_nonzero(held)) == 0, label
            return _grads(dx, dw, db)

        _route_judge(S, C, "perf", kind, call, key=9, label=label)
        record_property("override_ct_vec", ct_vec)


@pytest.mark.parametrize("own_workspace", (True, False), ids=("own-ws", "pkg-ws"))
def test_narrow_tile_row(own_workspace, monkeypatch, record_property) -> None:
    """Every ``NARROW_CASES`` row of the architecture's ``NARROW_TILES`` launches 32-vector channel tiles (``32``,
    ``threads // 32`` sub-tiles, ``ceil(C / vec / 32)`` tiles) instead of :func:`channel_split`'s single tile, on
    the grid the host plans for the call (``_call_plan``: ``launch_grid`` with the side-stream trim), on a
    caller-owned workspace of the public size or the package's: ``dx``, ``dweight`` and ``dbias`` judged against the
    naive tier, the workspace zero after every call."""

    arch, device = _arch(), _device()
    cases = NARROW_CASES.get(arch, ())
    if not cases:
        pytest.skip(f"{arch} has no NARROW_TILES row")
    seen = []
    launch = pkg.run_module

    def record_launch(name, values, *, device_index, grid=None):
        seen.append(
            (
                name,
                {key: int(values[key]) for key in ("ct_vec", "n_sub", "n_ch_tiles")},
                grid,
            )
        )
        return launch(name, values, device_index=device_index, grid=grid)

    monkeypatch.setattr(pkg, "run_module", record_launch)
    for kind, S, C in cases:
        _require_route(kind)
        label = f"{arch} {kind} S={S} C={C}"
        assert loader.narrow_tile_row(arch, kind, C, S), label
        (name,) = loader.selected_modules(arch, kind, C=C, S=S)
        geometry = loader.MODULES[name]["grid_rule"]["geometry"]
        wanted = {
            "ct_vec": 32,
            "n_sub": geometry["threads"] // 32,
            "n_ch_tiles": -(-(C // geometry["vec"]) // 32),
        }
        assert wanted != dict(
            zip(
                ("ct_vec", "n_sub", "n_ch_tiles"),
                loader.channel_split(C, geometry=geometry, role="bwd"),
                strict=True,
            )
        ), f"{label}: the row must differ from the default split"
        _, _, _, grid, _ = pkg._call_plan(
            arch,
            kind,
            S,
            C,
            len(doc_lengths(S, "perf")),
            device_index=device.index,
            policy=ROUTE_SETTINGS[kind]["policy"],
        )
        workspace = (
            pkg.cake_causal_conv1d_nondet_train_backward_workspace(S, C, device)
            if own_workspace
            else None
        )

        def call(x, w, cu, dy, b, policy, name=name, wanted=wanted, grid=grid, workspace=workspace, label=label):
            seen.clear()
            dx, dw, db = _backward(x, w, cu, dy, b, policy, kind=kind, workspace=workspace)
            torch.cuda.synchronize(device)
            assert seen == [(name, wanted, grid)], f"{label}: launched {seen}"
            held = workspace if own_workspace else _stream_workspace(device)
            assert held is not None and int(torch.count_nonzero(held)) == 0, label
            return _grads(dx, dw, db)

        _route_judge(S, C, "perf", kind, call, key=11, label=label)
    record_property("narrow_rows", len(cases))


def test_narrow_tile_row_gives_up_one_sm_at_three_tiles(monkeypatch) -> None:
    """A three-tile ``NARROW_TILES`` row: on the ``bwd`` and ``bwd_fp32g`` routes a
    single-wave launch of ``ctas_per_sm x SM count`` CTAs whose last channel-tile rank holds at least
    ``ctas_per_sm`` CTAs (``full % 3 >= ctas_per_sm``; B200: 296 % 3 = 2) launches ``ctas_per_sm`` fewer, as at a
    route-free plan of three tiles; a grid ``launch_grid`` already trimmed to whole ranks is kept.  The trimmed
    launch's outputs and those of the whole grid are judged against the naive tier, and the workspace reads back
    zero."""

    arch, device = _arch(), _device()
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    rows = [
        (kind, S, C)
        for kind, S, C in NARROW_CASES.get(arch, ())
        if kind in ("bwd", "bwd_fp32g")
    ]
    if not rows:
        pytest.skip(f"{arch} has no NARROW_TILES row on the bwd / bwd_fp32g routes")
    launched = []
    run_module = pkg.run_module

    def record_launch(name, values, *, device_index, grid=None):
        launched.append((int(values["n_ch_tiles"]), grid))
        return run_module(name, values, device_index=device_index, grid=grid)

    monkeypatch.setattr(pkg, "run_module", record_launch)
    checked = 0
    for kind, S, C in rows:
        _require_route(kind)
        (name,) = loader.selected_modules(arch, kind, C=C, S=S)
        record = loader.MODULES[name]
        rule = record["grid_rule"]
        geometry = rule["geometry"]
        tiles = -(-(C // geometry["vec"]) // 32)
        if (
            tiles != 3
            or rule["kind"] != "occupancy"
            or loader.grid_mult(record, arch) != 1
        ):
            continue
        k = int(rule["ctas_per_sm"])
        full = k * sms
        untrimmed = loader.launch_grid(record, device_index=device.index, C=C, S=S)
        cut = untrimmed[0] == full and full % tiles >= k
        expected = (full - k, 1, 1) if cut else tuple(untrimmed)
        label = f"{arch} {kind} S={S} C={C} full={full} k={k}"

        def call(x, w, cu, dy, b, policy, tiles=tiles, expected=expected, cut=cut, untrimmed=untrimmed, label=label):
            pkg.cake_causal_conv1d_nondet_train_plan_cache_clear()
            launched.clear()
            runs = [_backward(x, w, cu, dy, b, policy, kind=kind)]
            torch.cuda.synchronize(device)
            assert launched == [(tiles, expected)], f"{label}: launched {launched}"
            held = _stream_workspace(device)
            assert held is None or int(torch.count_nonzero(held)) == 0, label
            if cut:
                with monkeypatch.context() as m:
                    m.setattr(pkg, "_one_sm_trim", lambda record, n, grid, **kw: grid)
                    pkg.cake_causal_conv1d_nondet_train_plan_cache_clear()
                    launched.clear()
                    runs.append(_backward(x, w, cu, dy, b, policy, kind=kind))
                    torch.cuda.synchronize(device)
                pkg.cake_causal_conv1d_nondet_train_plan_cache_clear()
                assert launched == [(tiles, tuple(untrimmed))], f"{label}: {launched}"
            found = {}
            for run in runs:
                for out, value in _grads(*run).items():
                    found.setdefault(out, []).append(value)
            return found

        _route_judge(S, C, "perf", kind, call, key=43, label=label)
        checked += 1
    assert checked, (
        f"{arch}: no three-tile NARROW_TILES row on an occupancy single-wave record"
    )


@pytest.mark.parametrize("kind", tuple(ROUTE_SETTINGS))
@pytest.mark.parametrize("case", GRID_MULT_CASES, ids=_case_id)
def test_bwd_grid_mult_outputs_and_workspace(case, kind) -> None:
    """Every ``bwd_grid_mult`` on every route, on a caller-owned workspace (allocated at that multiple) and on the
    package's stream workspace: every launch's ``dx``, ``dweight`` and ``dbias`` judged against the naive tier (the
    worst launch of a draw counts), and the workspace read back all zero after every launch."""

    _require_route(kind)
    S, C, docs = case
    device = _device()
    workspaces = {
        mult: pkg.cake_causal_conv1d_nondet_train_backward_workspace(
            S, C, device, bwd_grid_mult=mult
        )
        for mult in pkg.BWD_GRID_MULTS
    }

    def call(x, w, cu, dy, b, policy):
        found = {}
        for mult in pkg.BWD_GRID_MULTS:
            for own in (True, False):
                kwargs = {"workspace": workspaces[mult]} if own else {}
                run = _backward(x, w, cu, dy, b, policy, kind=kind, bwd_grid_mult=mult, **kwargs)
                label = f"{kind} S={S} C={C} bwd_grid_mult={mult} own={own}"
                held = workspaces[mult] if own else _stream_workspace(device)
                assert held is not None and int(torch.count_nonzero(held)) == 0, (
                    f"workspace not zero ({label})"
                )
                for out, value in _grads(*run).items():
                    found.setdefault(out, []).append(value)
        return found

    _route_judge(S, C, docs, kind, call, key=31, label=f"{kind} {case}")


@pytest.mark.parametrize("case", PRECISION_CASES, ids=_case_id)
def test_fp32_grad_perf_size_precision(case) -> None:
    """The fp32 policy at the perf size (S = 250340): ``PRECISION_REPEATS`` launches per draw, every launch's ``dx``
    and ``dweight`` judged against the naive tier (the worst launch of a draw counts)."""

    _require_route("bwd_fp32g")
    S, C, docs = case

    def call(x, w, cu, dy, b, policy):
        found = {}
        for _ in range(PRECISION_REPEATS):
            for out, value in _grads(*_backward(x, w, cu, dy, None, "fp32")).items():
                found.setdefault(out, []).append(value)
        return found

    _judge(S, C, docs, call, policy="fp32", label=f"fp32 {case}")
    torch.cuda.empty_cache()


def test_document_isolation() -> None:
    """Perturbing one document (the 3-token document, rows 1000..1002, drawn again) leaves every document's outputs
    judged against the naive tier of the changed problem."""

    S, C, docs = 4097, 512, (1000, 0, 3, 3094)

    def perturb(x, dy):
        x = x.clone()
        x[1000:1003] = torch.randn(3, C, device=_device()).to(BF16)
        return x, dy

    _judge(S, C, docs, _forward_backward, key=5, perturb=perturb, label="perturbed document")


def test_batched_layout() -> None:
    """``[1, S, C]`` inputs give ``[1, S, C]`` outputs, judged against the naive tier."""

    S, C, docs = 1000, 512, "perf"

    def call(x, w, cu, dy, b, policy):
        y3 = _forward(x.unsqueeze(0), w, cu)
        dx3, dw3, _ = _backward(x.unsqueeze(0), w, cu, dy.unsqueeze(0))
        assert tuple(y3.shape) == (1, S, C) and tuple(dx3.shape) == (1, S, C)
        return {"y": y3, **_grads(dx3, dw3, None)}

    _judge(S, C, docs, call, label="[1, S, C]")


# ----------------------------------------------------------------------------------------------------------------
# The workspace contract.


def test_workspace_bytes_depend_on_channels_only() -> None:
    device = _device()
    arch = _arch()
    for C in (128, 512, 1160, 1280, 1344, 4096, 8192):
        nbytes = pkg.cake_causal_conv1d_nondet_train_backward_workspace_bytes(
            1, C, device
        )
        assert nbytes == pkg.cake_causal_conv1d_nondet_train_backward_workspace_bytes(
            10**6, C, device
        )
        assert nbytes == loader.backward_workspace_bytes(arch, C)
        assert nbytes % loader.WORKSPACE_ALIGN == 0
        # One accumulator set and one counter per channel tile, whatever the grid: every bwd_grid_mult, one size.
        for mult in pkg.BWD_GRID_MULTS:
            assert (
                nbytes
                == pkg.cake_causal_conv1d_nondet_train_backward_workspace_bytes(
                    1, C, device, bwd_grid_mult=mult
                )
            )
        # At least one FP32 word per tap and channel (the bias routes' owners hold five per channel).
        assert nbytes >= 4 * WIDTH * C
        ws = pkg.cake_causal_conv1d_nondet_train_backward_workspace(1, C, device)
        assert ws.dtype == torch.uint8 and ws.numel() == nbytes
        assert int(torch.count_nonzero(ws)) == 0


@pytest.mark.parametrize("kind", tuple(ROUTE_SETTINGS))
def test_changed_shapes_reuse_one_zero_workspace(kind) -> None:
    """One caller-owned workspace, sized for the largest ``C``, serves a sequence of shapes of the route (and the
    package's own stream workspace serves the same sequence): every result is judged against the naive tier and the
    buffer reads back all zero after every launch."""

    _require_route(kind)
    device = _device()
    shapes = ((4097, 1280, (1000, 0, 3, 3094)), (65, 256, (64, 1)), (1000, 1344, "perf"),
              (5, 136, (2, 0, 3)), (4097, 1160, (1000, 0, 3, 3094)), (1000, 512, "perf"))  # fmt: skip
    workspace = pkg.cake_causal_conv1d_nondet_train_backward_workspace(
        1, max(C for _, C, _ in shapes), device
    )
    for own in (True, False):
        for S, C, docs in shapes:
            label = f"{kind} S={S} C={C} own={own}"

            def call(x, w, cu, dy, b, policy, own=own, label=label):
                kwargs = {"workspace": workspace} if own else {}
                run = _backward(x, w, cu, dy, b, policy, kind=kind, **kwargs)
                held = workspace if own else _stream_workspace(device)
                assert held is not None
                assert int(torch.count_nonzero(held)) == 0, (
                    f"workspace not zero after {label}"
                )
                return _grads(*run)

            _route_judge(S, C, docs, kind, call, key=6, label=label)


@pytest.mark.parametrize("kind", tuple(ROUTE_SETTINGS))
def test_nan_poisoned_outputs_are_overwritten(kind) -> None:
    _require_route(kind)
    device = _device()
    for S, C, docs in ROUTE_CASES:

        def call(x, w, cu, dy, b, policy, C=C):
            dx = torch.full_like(x, float("nan"))
            dweight_dtype = ROUTE_SETTINGS[kind]["dweight_dtype"] or w.dtype
            dw = torch.full((C, WIDTH), float("nan"), dtype=dweight_dtype, device=device)
            db = (
                None
                if b is None
                else torch.full((C,), float("nan"), dtype=b.dtype, device=device)
            )
            y = torch.full_like(x, float("nan"))
            _forward(x, w, cu, bias=b, out=y)
            got = _backward(x, w, cu, dy, b, policy, kind=kind, dx=dx, dweight=dw, dbias=db)
            assert got[0] is dx and got[1] is dw and got[2] is db
            return {"y": y, **_grads(dx, dw, db)}

        _route_judge(S, C, docs, kind, call, key=7, label=f"NaN-poisoned {kind} S={S} C={C}")


def test_workspace_validation() -> None:
    S, C = 64, 256
    x, w, cu, dy = _inputs(S, C, None)
    nbytes = pkg.cake_causal_conv1d_nondet_train_backward_workspace_bytes(
        S, C, _device()
    )
    with pytest.raises(ValueError, match="workspace holds"):
        _backward(
            x, w, cu, dy, workspace=torch.zeros(16, dtype=torch.uint8, device=_device())
        )
    with pytest.raises(ValueError, match="uint8"):
        _backward(
            x,
            w,
            cu,
            dy,
            workspace=torch.zeros(nbytes // 4, dtype=torch.int32, device=_device()),
        )
    with pytest.raises(ValueError, match="aligned"):
        big = torch.zeros(nbytes + 16, dtype=torch.uint8, device=_device())
        _backward(x, w, cu, dy, workspace=big[1 : nbytes + 1])
    with pytest.raises(ValueError, match="1-D"):
        _backward(
            x,
            w,
            cu,
            dy,
            workspace=torch.zeros(1, nbytes, dtype=torch.uint8, device=_device()),
        )


def _held_views(workspace: torch.Tensor) -> dict | None:
    """The package's cached ``{layout key: (data_ptr, acc, work)}`` of ``workspace`` (``None`` without an entry)."""

    hit = pkg._ACCUMULATOR_VIEWS.get(id(workspace))
    return hit[1] if hit is not None and hit[0]() is workspace else None


def test_workspace_views_are_built_once_per_workspace() -> None:
    """The ``acc`` / ``work`` tensors of a workspace are built once per layout and reused by later calls; they
    alias its storage but not the tensor, follow the storage when it moves, and are dropped with the
    workspace; an entry left under the workspace's ``id`` by a dead workspace is never served.  The calls' outputs
    are judged against the naive tier."""

    S, C, docs = 1000, 1280, "perf"
    device = _device()
    nbytes = pkg.cake_causal_conv1d_nondet_train_backward_workspace_bytes(S, C, device)
    workspace = pkg.cake_causal_conv1d_nondet_train_backward_workspace(S, C, device)
    entries = len(pkg._ACCUMULATOR_VIEWS)

    def calls(count, ws):
        def call(x, w, cu, dy, b, policy):
            found = {}
            for _ in range(count):
                for out, value in _grads(*_backward(x, w, cu, dy, workspace=ws)).items():
                    found.setdefault(out, []).append(value)
            return found

        return call

    _judge(S, C, docs, calls(2, workspace), label="two calls")
    held = _held_views(workspace)
    assert held is not None and len(held) == 1
    ((ptr, acc, work),) = held.values()
    assert ptr == workspace.data_ptr()
    assert acc._base is None and work._base is None
    storage = workspace.untyped_storage().data_ptr()
    assert acc.untyped_storage().data_ptr() == storage
    assert work.untyped_storage().data_ptr() == storage
    moved = torch.zeros(nbytes, dtype=torch.uint8, device=device)
    workspace.set_(moved.untyped_storage(), 0, (nbytes,), (1,))
    del moved
    _judge(S, C, docs, calls(1, workspace), label="moved storage")
    ((ptr, acc, work),) = _held_views(workspace).values()
    assert ptr == workspace.data_ptr()
    assert acc.untyped_storage().data_ptr() == workspace.untyped_storage().data_ptr()
    assert int(torch.count_nonzero(workspace)) == 0
    # The entry of a dead workspace under this workspace's id (an id reused before the entry was dropped), with
    # views into other buffers: the call must not serve it but rebuild the views on this workspace.
    dead = torch.empty(0)
    ((key, _),) = held.items()
    pkg._ACCUMULATOR_VIEWS[id(workspace)] = (
        weakref.ref(dead),
        {key: (workspace.data_ptr(), torch.zeros_like(acc), torch.zeros_like(work))},
    )
    del dead
    assert pkg._ACCUMULATOR_VIEWS[id(workspace)][0]() is None
    _judge(S, C, docs, calls(1, workspace), label="dead entry under the id")
    ((ptr, acc, work),) = _held_views(workspace).values()
    assert ptr == workspace.data_ptr()
    assert acc.untyped_storage().data_ptr() == workspace.untyped_storage().data_ptr()
    assert work.untyped_storage().data_ptr() == workspace.untyped_storage().data_ptr()
    assert int(torch.count_nonzero(workspace)) == 0
    del workspace, held, acc, work
    gc.collect()
    assert len(pkg._ACCUMULATOR_VIEWS) == entries


# ----------------------------------------------------------------------------------------------------------------
# Graphs, streams, launches.


def _capture_and_replay(
    cases, kind: str, *, key: int, own_workspace: bool, bwd_grid_mult: int = 1
) -> int:
    """One forward + backward graph per shape (the backward at ``bwd_grid_mult``), replayed once per draw on that
    draw's inputs (copied into the captured input tensors) into NaN-filled outputs: every replay's ``y``, ``dx``,
    ``dweight`` and ``dbias`` judged against the naive tier, and a caller-owned workspace read back all zero after
    every replay.  Returns the replays."""

    device = _device()
    captured = []
    for S, C, docs in cases:
        x, w, cu, dy, b, policy = _route_inputs(S, C, docs, kind, key=key)
        y = torch.empty_like(x)
        dx = torch.empty_like(x)
        dw = torch.empty(
            C, WIDTH, dtype=ROUTE_SETTINGS[kind]["dweight_dtype"] or w.dtype, device=device
        )
        db = None if b is None else torch.empty(C, dtype=b.dtype, device=device)
        workspace = (
            pkg.cake_causal_conv1d_nondet_train_backward_workspace(
                S, C, device, bwd_grid_mult=bwd_grid_mult
            )
            if own_workspace
            else None
        )

        def step(
            x=x,
            w=w,
            cu=cu,
            dy=dy,
            b=b,
            policy=policy,
            y=y,
            dx=dx,
            dw=dw,
            db=db,
            ws=workspace,
        ):
            _forward(x, w, cu, bias=b, out=y)
            _backward(
                x,
                w,
                cu,
                dy,
                b,
                policy,
                kind=kind,
                workspace=ws,
                dx=dx,
                dweight=dw,
                dbias=db,
                bwd_grid_mult=bwd_grid_mult,
            )

        side = torch.cuda.Stream(device=device)
        side.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(side):
            step()
        torch.cuda.current_stream(device).wait_stream(side)
        torch.cuda.synchronize(device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=side):
            step()
        torch.cuda.synchronize(device)
        captured.append(((S, C, docs), graph, (x, w, cu, dy, b), (y, dx, dw, db), workspace))
    replays = 0
    for (S, C, docs), graph, inputs, (y, dx, dw, db), workspace in captured:

        def replay(xn, wn, cu, dyn, bn, policy, graph=graph, inputs=inputs, outs=(y, dx, dw, db), workspace=workspace):
            nonlocal replays
            for held, value in zip(inputs, (xn, wn, cu, dyn, bn), strict=True):
                if held is not None:
                    held.copy_(value)
            for t in outs:
                if t is not None:
                    t.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize(device)
            replays += 1
            if workspace is not None:
                assert int(torch.count_nonzero(workspace)) == 0
            return {"y": outs[0], **_grads(*outs[1:])}

        _route_judge(
            S,
            C,
            docs,
            kind,
            replay,
            key=key,
            label=f"replay {kind} S={S} C={C} bwd_grid_mult={bwd_grid_mult} own={own_workspace}",
        )
    return replays


@pytest.mark.parametrize("own_workspace", (True, False), ids=("own-ws", "pkg-ws"))
@pytest.mark.parametrize("kind", tuple(ROUTE_SETTINGS))
def test_cuda_graph_replays(kind, own_workspace) -> None:
    _require_route(kind)
    cases = GRAPH_CASES if kind == "bwd" else ROUTE_CASES[:3]
    replays = _capture_and_replay(cases, kind, key=11, own_workspace=own_workspace)
    assert replays == DRAWS * len(cases)


@pytest.mark.parametrize("bwd_grid_mult", (2, 3, 4))
@pytest.mark.parametrize("kind", tuple(ROUTE_SETTINGS))
def test_cuda_graph_replays_with_bwd_grid_mult(kind, bwd_grid_mult) -> None:
    """CUDA Graph capture / replay of a ``bwd_grid_mult`` 2-4 backward, with a caller-owned workspace and with the
    package's: every replay judged against the naive tier, the caller-owned workspace zero after every replay."""

    _require_route(kind)
    cases = ROUTE_CASES[:3]
    for own_workspace in (True, False):
        replays = _capture_and_replay(
            cases,
            kind,
            key=33,
            own_workspace=own_workspace,
            bwd_grid_mult=bwd_grid_mult,
        )
        assert replays == DRAWS * len(cases)


def test_capture_zeroes_one_backward_workspace_per_capture_and_stream(
    monkeypatch,
) -> None:
    """Without a caller workspace, the backward calls that one CUDA Graph capture records on one stream share one
    zeroed package buffer (one fill per capture and stream), and a second capture gets a buffer of its own; replays
    of both graphs on each draw's inputs into NaN-filled outputs are judged against the naive tier, and the buffer
    reads back zero after the replays."""

    pytest.importorskip("cuda.bindings")
    device = _device()
    launched = []
    run_module = pkg.run_module

    def record_launch(name, values, *, device_index, grid=None):
        launched.append(values["acc"].untyped_storage().data_ptr())
        return run_module(name, values, device_index=device_index, grid=grid)

    # the largest workspace first, so the later calls of a capture fit in the first call's buffer
    shapes = ((4097, 1280, (1000, 0, 3, 3094)), (65, 256, (64, 1)), (1000, 512, "perf"))
    problems = [_inputs(S, C, docs, key=41) for S, C, docs in shapes]
    side = torch.cuda.Stream(device=device)
    side.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(side):
        for x, w, cu, dy in problems:
            _backward(x, w, cu, dy)
    torch.cuda.current_stream(device).wait_stream(side)
    torch.cuda.synchronize(device)
    monkeypatch.setattr(pkg, "run_module", record_launch)
    captured = []
    for _capture in range(2):
        outs = [
            (
                torch.empty_like(x),
                torch.empty(x.shape[1], WIDTH, dtype=w.dtype, device=device),
            )
            for x, w, _, _ in problems
        ]
        graph = torch.cuda.CUDAGraph()
        launched.clear()
        with torch.cuda.graph(graph, stream=side):
            for (x, w, cu, dy), (dx, dw) in zip(problems, outs, strict=True):
                _backward(x, w, cu, dy, dx=dx, dweight=dw)
        torch.cuda.synchronize(device)
        assert len(launched) == len(shapes) and len(set(launched)) == 1, launched
        captured.append((graph, outs, launched[0]))
    assert captured[0][2] != captured[1][2], "two captures share a package workspace"
    for index, (graph, outs, _) in enumerate(captured):
        for i in range(len(shapes)):

            def launch(_draw, seed, graph=graph, outs=outs, i=i):
                fresh = [_inputs(S, C, docs, key=41, seed=seed) for S, C, docs in shapes]
                for held, value in zip(problems, fresh, strict=True):
                    for h, v in zip(held, value, strict=True):
                        h.copy_(v)
                for dx, dw in outs:
                    dx.fill_(float("nan"))
                    dw.fill_(float("nan"))
                graph.replay()
                torch.cuda.synchronize(device)
                x, w, cu, dy = fresh[i]
                return _grads(*outs[i], None), problem(x, w, cu, None, dy)

            assert_judged(launch, seed=SEED, label=f"capture {index} replay {shapes[i]}")
    held = pkg._CAPTURE_WORKSPACES.get((device.index, side.cuda_stream))
    assert held is not None and held[1].untyped_storage().data_ptr() == captured[1][2]
    assert int(torch.count_nonzero(held[1])) == 0


def test_two_streams_with_two_workspaces() -> None:
    """Two backward calls running at once on two streams, each with its own workspace (caller-owned, then the
    package's per-stream buffers): both judged against the naive tier on every draw, and every workspace is zero
    afterwards."""

    device = _device()
    S, C, docs = 65536, 1280, "perf"
    keys = (21, 22)
    streams = [torch.cuda.Stream(device=device) for _ in keys]
    for own in (True, False):
        workspaces = [
            pkg.cake_causal_conv1d_nondet_train_backward_workspace(S, C, device)
            if own
            else None
            for _ in keys
        ]
        for i in range(len(keys)):

            def launch(_draw, seed, i=i, workspaces=workspaces, own=own):
                problems = [_inputs(S, C, docs, key=k, seed=seed) for k in keys]
                results = [None, None]
                torch.cuda.synchronize(device)
                for j, ((x, w, cu, dy), stream) in enumerate(
                    zip(problems, streams, strict=True)
                ):
                    with torch.cuda.stream(stream):
                        kwargs = {"workspace": workspaces[j]} if own else {}
                        results[j] = _backward(x, w, cu, dy, **kwargs)
                torch.cuda.synchronize(device)
                for j, stream in enumerate(streams):
                    if own:
                        assert int(torch.count_nonzero(workspaces[j])) == 0
                    else:
                        with torch.cuda.stream(stream):
                            held = _stream_workspace(device)
                        assert held is not None and int(torch.count_nonzero(held)) == 0
                x, w, cu, dy = problems[i]
                return _grads(*results[i]), problem(x, w, cu, None, dy)

            assert_judged(launch, seed=SEED, label=f"stream {i} own={own}")


def test_backward_is_one_launch_with_zeroed_accumulators(monkeypatch) -> None:
    """A backward call is ONE ordinary launch of the program ``C`` selects (no reduce program): its arguments
    carry the ``acc`` / ``work`` views of the record's accumulator layout (FP32 ``slots x taps x C`` words as the
    rule names them, one FP32 word per sum and accumulator copy on every route; uint32 ``n_ch_tiles`` counters), zero
    at launch, and the grid of the call's plan, at least one CTA per channel tile."""

    S, C, docs = 1000, 1280, "perf"
    device = _device()
    launched = []

    def record_launch(name, values, *, device_index, grid=None):
        launched.append((name, values, grid))
        acc, work = values["acc"], values["work"]
        launched[-1] += (int(torch.count_nonzero(acc)), int(torch.count_nonzero(work)))

    monkeypatch.setattr(pkg, "run_module", record_launch)
    arch = _arch()
    for kind in ROUTE_SETTINGS:
        if _route_pending(arch, kind):
            continue
        launched.clear()
        x, w, cu, dy, b, policy = _route_inputs(S, C, docs, kind)
        _backward(x, w, cu, dy, b, policy, kind=kind)
        assert len(launched) == 1, f"{kind}: {len(launched)} launches"
        name, values, grid, acc_nonzero, work_nonzero = launched[0]
        assert name == loader.selected_modules(arch, kind, C=C, S=S)[0]
        record = loader.MODULES[name]
        assert "reduce" not in record and record["launch"]["cooperative"] is False
        assert grid == pkg._one_sm_trim(
            record,
            values["n_ch_tiles"],
            loader.launch_grid(record, device_index=device.index, C=C, S=S),
            device_index=device.index,
            arch=arch,
            bwd_grid_mult=1,
        )
        assert grid[0] >= values["n_ch_tiles"]
        layout = loader.accumulator_layout(record, C, values["n_ch_tiles"])
        acc_dtype = torch.float32
        taps = 5 if "bias" in kind else 4
        words = record["workspace_rule"]["slots"] * taps
        assert loader.accumulator_words_per_channel(record) == words
        assert values["acc"].dtype == acc_dtype and values["acc"].numel() == words * C
        assert (
            values["work"].dtype == torch.uint32
            and values["work"].numel() == values["n_ch_tiles"]
        )
        assert layout["acc_elems"] == words * C
        assert acc_nonzero == 0 and work_nonzero == 0
        assert {"x", "w", "cu", "dy", "dx", "dw", "acc", "work"} <= set(values)


@pytest.mark.parametrize("kind", tuple(ROUTE_SETTINGS))
def test_bwd_grid_mult_launches_that_many_times_the_untrimmed_grid(
    kind, monkeypatch
) -> None:
    """A backward at ``bwd_grid_mult`` k launches the program ``C`` selects once, with k times the record's
    untrimmed grid for k > 1 and the call's plan (possibly trimmed) for k = 1: the option is never a silent no-op."""

    _require_route(kind)
    device = _device()
    arch = _arch()
    launched = []
    tiles = []
    run_module = pkg.run_module

    def record_launch(name, values, *, device_index, grid=None):
        launched.append((name, grid))
        tiles.append(values["n_ch_tiles"])
        return run_module(name, values, device_index=device_index, grid=grid)

    monkeypatch.setattr(pkg, "run_module", record_launch)
    for S, C, docs in (
        (1000, 1344, "perf"),
        (65536, 1280, "perf"),
        (5, 136, (2, 0, 3)),
    ):
        x, w, cu, dy, b, policy = _route_inputs(S, C, docs, kind, key=32)
        name = loader.selected_modules(arch, kind, C=C, S=S)[0]
        record = loader.MODULES[name]
        untrimmed = loader.launch_grid(record, device_index=device.index)
        for mult in pkg.BWD_GRID_MULTS:
            launched.clear()
            tiles.clear()
            _backward(x, w, cu, dy, b, policy, kind=kind, bwd_grid_mult=mult)
            expected = (
                pkg._one_sm_trim(
                    record,
                    tiles[0],
                    loader.launch_grid(record, device_index=device.index, C=C, S=S),
                    device_index=device.index,
                    arch=arch,
                    bwd_grid_mult=1,
                )
                if mult == 1
                else (untrimmed[0] * mult, 1, 1)
            )
            assert launched == [(name, expected)], (kind, S, C, mult, launched)


@pytest.mark.parametrize("kind", tuple(ROUTE_SETTINGS))
def test_backward_grid_gives_up_one_sm_when_the_last_rank_is_partial(
    kind, monkeypatch
) -> None:
    """On the ``bwd`` and ``bwd_fp32g`` routes a default-grid backward launches ``ctas_per_sm
    x SM count`` CTAs less ``ctas_per_sm`` when the last channel-tile rank of that grid holds at least
    ``ctas_per_sm`` CTAs (so a kernel on another stream that keeps one SM's shared memory leaves the launch one
    wave), else the whole grid; the other backward routes launch ``launch_grid``'s grid.  The trimmed launch's
    outputs and those of the whole grid are judged against the naive tier and the workspace reads back zero."""

    _require_route(kind)
    device = _device()
    arch = _arch()
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    S, docs = 4097, (1000, 0, 3, 3094)
    launched = []
    run_module = pkg.run_module

    def record_launch(name, values, *, device_index, grid=None):
        launched.append(grid)
        return run_module(name, values, device_index=device_index, grid=grid)

    monkeypatch.setattr(pkg, "run_module", record_launch)
    rows = {}
    for C in range(1024, 16385, 1024):
        name = loader.selected_modules(arch, kind, C=C, S=S)[0]
        record = loader.MODULES[name]
        rule = record["grid_rule"]
        k = int(rule["ctas_per_sm"])
        full = loader.grid_mult(record, arch) * k * sms
        n = loader.plan(
            S,
            C,
            len(docs),
            geometry=rule["geometry"],
            role=record["role"],
            arch=arch,
            route=loader.record_route(record, arch),
        )["n_ch_tiles"]
        trims = (
            rule["kind"] == "occupancy"
            and loader.grid_mult(record, arch) == 1
            and full % n >= k
        )
        rows.setdefault(trims, (C, full, k))
    assert set(rows) == {True, False}, rows
    for trims, (C, full, k) in sorted(rows.items()):
        cut = trims and kind in ("bwd", "bwd_fp32g")

        def call(x, w, cu, dy, b, policy, C=C, full=full, k=k, cut=cut):
            launched.clear()
            runs = [_backward(x, w, cu, dy, b, policy, kind=kind)]
            torch.cuda.synchronize(device)
            assert launched == [(full - k if cut else full, 1, 1)], (kind, C, launched)
            held = _stream_workspace(device)
            assert held is None or int(torch.count_nonzero(held)) == 0, (kind, C)
            if cut:
                with monkeypatch.context() as m:
                    m.setattr(pkg, "_one_sm_trim", lambda record, n, grid, **kw: grid)
                    pkg.cake_causal_conv1d_nondet_train_plan_cache_clear()
                    launched.clear()
                    runs.append(_backward(x, w, cu, dy, b, policy, kind=kind))
                    torch.cuda.synchronize(device)
                pkg.cake_causal_conv1d_nondet_train_plan_cache_clear()
                assert launched == [(full, 1, 1)], (kind, C, launched)
            found = {}
            for run in runs:
                for out, value in _grads(*run).items():
                    found.setdefault(out, []).append(value)
            return found

        _route_judge(S, C, docs, kind, call, key=41, label=f"{kind} C={C}")


@pytest.mark.parametrize("kind", ("bwd_fp32g", "bwd_bias_fp32g"))
@pytest.mark.parametrize("C", (512, 8192))
def test_fp32g_accumulators_follow_the_rendered_rule(kind, C, monkeypatch) -> None:
    """An fp32g backward at ``C`` 512 and 8192 hands its program an FP32 ``acc`` (one FP32 word per sum and
    accumulator copy) of the size the rendered rule names (``slots x taps x C``, however many accumulator copies the
    program keeps) inside the public workspace; the outputs are judged against the naive tier, and the workspace
    reads back zero."""

    _require_route(kind)
    S, docs = 4097, (1000, 0, 3, 3094)
    device = _device()
    workspace = pkg.cake_causal_conv1d_nondet_train_backward_workspace(S, C, device)
    seen = []
    launch = pkg.run_module

    def record_launch(name, values, *, device_index, grid=None):
        seen.append(
            (name, values["acc"].dtype, values["acc"].numel(), values["n_ch_tiles"])
        )
        return launch(name, values, device_index=device_index, grid=grid)

    def call(x, w, cu, dy, b, policy):
        seen.clear()
        with monkeypatch.context() as m:
            m.setattr(pkg, "run_module", record_launch)
            dx, dw, db = _backward(x, w, cu, dy, b, policy, kind=kind, workspace=workspace)
        ((name, acc_dtype, acc_numel, n_ch_tiles),) = seen
        record = loader.MODULES[name]
        layout = loader.accumulator_layout(record, C, n_ch_tiles)
        assert acc_dtype == torch.float32
        assert (
            acc_numel
            == layout["acc_elems"]
            == loader.accumulator_words_per_channel(record) * C
        )
        assert workspace.numel() >= layout["total_bytes"]
        assert int(torch.count_nonzero(workspace)) == 0
        return _grads(dx, dw, db)

    _route_judge(S, C, docs, kind, call, key=11, label=f"{kind} C={C}")


def test_backward_keeps_its_speed_beside_a_resident_side_kernel(
    record_property,
) -> None:
    """One 1-thread kernel resident on another stream must not stall the
    one-launch backward (``busy <= 2 x alone``, medians of 7 after 3 warm-ups, CUDA events on the current stream);
    the outputs of backward calls launched beside it are judged against the naive tier."""

    device = _device()
    S, C, docs = 261393, 512, "perf"
    x, w, cu, dy = _inputs(S, C, docs)
    workspace = pkg.cake_causal_conv1d_nondet_train_backward_workspace(S, C, device)
    dx = torch.empty_like(x)
    dw = torch.empty(C, WIDTH, dtype=torch.float32, device=device)
    side = torch.cuda.Stream(device=device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    def timed(with_side_kernel: bool) -> float:
        torch.cuda.synchronize(device)
        if with_side_kernel:
            with torch.cuda.stream(side):
                torch.cuda._sleep(10_000_000)
        torch.cuda._sleep(1_000_000)
        start.record()
        _backward(x, w, cu, dy, workspace=workspace, dx=dx, dweight=dw)
        end.record()
        torch.cuda.synchronize(device)
        return start.elapsed_time(end)

    for _ in range(3):
        timed(False)
    alone = statistics.median(timed(False) for _ in range(7))
    busy = statistics.median(timed(True) for _ in range(7))
    print(
        f"backward alone {alone:.3f} ms, beside a resident kernel {busy:.3f} ms ({busy / alone:.2f}x)"
    )
    record_property("backward_alone_ms", f"{alone:.4f}")
    record_property("backward_beside_side_kernel_ms", f"{busy:.4f}")
    record_property("busy_over_alone", f"{busy / alone:.3f}")
    assert busy <= 2.0 * alone, (
        f"the backward stalled behind a kernel resident on another stream: {busy:.3f} ms vs {alone:.3f} ms"
    )

    def beside(x, w, cu, dy, b, policy):
        torch.cuda.synchronize(device)
        with torch.cuda.stream(side):
            torch.cuda._sleep(10_000_000)
        torch.cuda._sleep(1_000_000)
        run = _backward(x, w, cu, dy, workspace=workspace)
        torch.cuda.synchronize(device)
        return _grads(*run)

    _judge(S, C, docs, beside, label="beside the side kernel")


def test_zero_tokens_return_empty_outputs() -> None:
    C = 256
    x = torch.empty(0, C, dtype=BF16, device=_device())
    w = torch.zeros(C, WIDTH, dtype=F32, device=_device())
    b = torch.zeros(C, dtype=BF16, device=_device())
    cu = torch.zeros(2, dtype=torch.int32, device=_device())
    assert tuple(_forward(x, w, cu).shape) == (0, C)
    dx, dw, _ = _backward(x, w, cu, x)
    assert tuple(dx.shape) == (0, C) and torch.equal(dw, torch.zeros_like(dw))
    dw_out = torch.full((C, WIDTH), float("nan"), dtype=F32, device=_device())
    db_out = torch.full((C,), float("nan"), dtype=BF16, device=_device())
    dx, dw, db = _backward(x, w, cu, x, b, "fp32", dweight=dw_out, dbias=db_out)
    assert dw is dw_out and db is db_out
    assert torch.equal(dw, torch.zeros_like(dw)) and torch.equal(
        db, torch.zeros_like(db)
    )


# ----------------------------------------------------------------------------------------------------------------
# Autograd, checkpointing, accumulation.


@pytest.mark.parametrize("combo", AUTOGRAD_COMBOS, ids=_combo_id)
def test_autograd_end_to_end(combo) -> None:
    """``cake_causal_conv1d_nondet`` with FP32 / BF16 taps, no / FP32 / BF16 bias, under both SiLU' policies: ``y``
    and the gradients, in the parameters' dtypes, judged against the naive tier, with a contiguous and a
    non-contiguous (materialized) incoming gradient."""

    weight_dtype, bias_dtype, policy = combo
    kind = _combo_route(weight_dtype, bias_dtype, policy)
    _require_route(kind)
    S, C, docs = 1000, 512, "perf"

    def call(x0, w0, cu, dy, b0, policy):
        found = {}
        for incoming in (dy, dy.t().contiguous().t()):
            x = x0.detach().clone().requires_grad_(True)
            w = w0.detach().clone().requires_grad_(True)
            b = None if b0 is None else b0.detach().clone().requires_grad_(True)
            y = pkg.cake_causal_conv1d_nondet(x, w, cu, b, silu_grad_preactivation=policy)
            y.backward(incoming)
            assert x.grad.dtype == BF16
            assert w.grad.dtype == w0.dtype and (b is None or b.grad.dtype == b0.dtype)
            outs = {"y": y.detach(), **_grads(x.grad, w.grad, None if b is None else b.grad)}
            for out, value in outs.items():
                found.setdefault(out, []).append(value)
        return found

    _judge(
        S,
        C,
        docs,
        call,
        weight_dtype=weight_dtype,
        bias_dtype=bias_dtype,
        policy=policy,
        key=3,
        label=f"autograd {kind} {_combo_id(combo)}",
    )


def test_non_reentrant_checkpoint_is_two_forwards_and_a_backward() -> None:
    """``torch.utils.checkpoint(..., use_reentrant=False)`` around the autograd entry (2 x forward + backward) and
    the unchecked call: ``y`` and the gradients of both judged against the naive tier."""

    from torch.utils.checkpoint import checkpoint

    _require_route("bwd_bias")
    S, C, docs = 4097, 1280, (1000, 0, 3, 3094)

    def call(x0, w0, cu, dy, b0, policy):
        found = {}
        for ckpt in (False, True):
            x = x0.detach().clone().requires_grad_(True)
            w = w0.detach().clone().requires_grad_(True)
            b = b0.detach().clone().requires_grad_(True)

            def layer(x, w, b):
                return pkg.cake_causal_conv1d_nondet(
                    x, w, cu, b, silu_grad_preactivation=policy
                )

            y = checkpoint(layer, x, w, b, use_reentrant=False) if ckpt else layer(x, w, b)
            y.backward(dy)
            for out, value in {"y": y.detach(), **_grads(x.grad, w.grad, b.grad)}.items():
                found.setdefault(out, []).append(value)
        return found

    _route_judge(S, C, docs, "bwd_bias", call, key=13, label="checkpoint")


def test_gradient_accumulation() -> None:
    """Two backward passes accumulate into the parameters' ``.grad`` (autograd's sum of the two gradients): the sum
    judged against the naive tier's sum of its two BF16 ``dweight`` (the float64 oracle's sum of the two)."""

    S, C, docs = 4097, 512, (1000, 0, 3, 3094)
    judge = TierJudge(DRAWS)
    failures = []
    for j in range(DRAWS):
        seed = draw_seed(SEED, j)
        seed_all(seed)
        x0, w0, cu, dy = _inputs(S, C, docs, key=17, seed=seed)
        dy2 = torch.randn_like(dy.float()).to(BF16)
        w = w0.detach().clone().requires_grad_(True)
        for g in (dy, dy2):
            pkg.cake_causal_conv1d_nondet(x0, w, cu).backward(g)
        want = sum(oracle(x0, w0, cu, None, g, outputs=("dweight",))["dweight"] for g in (dy, dy2))
        naive = sum(naive_bf16(x0, w0, cu, None, g, outputs=("dweight",))["dweight"] for g in (dy, dy2))
        if message := contract("dweight", w.grad, want):
            failures.append(f"draw {j}: {message}")
            continue
        judge.add("dweight", j, w.grad, want, {"bf16": naive})
    failures += judge.failures()
    assert not failures, "\n".join(failures)


@pytest.mark.parametrize("kind", AUTOGRAD_KINDS)
def test_bwd_grid_mult_through_autograd(kind) -> None:
    """The autograd entry carries ``bwd_grid_mult`` to its backward: at every multiple, plain and under
    non-reentrant checkpointing, ``y`` and the gradients judged against the naive tier, on every route the autograd
    entry reaches."""

    from torch.utils.checkpoint import checkpoint

    _require_route(kind)
    S, C, docs = 4097, 1344, (1000, 0, 3, 3094)

    def call(x0, w0, cu, dy, b0, policy):
        found = {}
        for mult in pkg.BWD_GRID_MULTS:
            for ckpt in (False, True):
                x = x0.detach().clone().requires_grad_(True)
                w = w0.detach().clone().requires_grad_(True)
                b = None if b0 is None else b0.detach().clone().requires_grad_(True)

                def layer(x, w, b, mult=mult):
                    return pkg.cake_causal_conv1d_nondet(
                        x,
                        w,
                        cu,
                        b,
                        silu_grad_preactivation=policy,
                        bwd_grid_mult=mult,
                    )

                y = (
                    checkpoint(layer, x, w, b, use_reentrant=False)
                    if ckpt
                    else layer(x, w, b)
                )
                y.backward(dy)
                outs = {"y": y.detach(), **_grads(x.grad, w.grad, None if b is None else b.grad)}
                for out, value in outs.items():
                    found.setdefault(out, []).append(value)
        return found

    _route_judge(S, C, docs, kind, call, key=34, label=f"autograd {kind}")


# ----------------------------------------------------------------------------------------------------------------
# Torch's deterministic mode.


def test_torch_deterministic_mode_alerts_on_the_backward() -> None:
    """Under ``torch.use_deterministic_algorithms(True)`` the backward raises (``warn_only=True``: warns and runs the
    non-deterministic backward, whose outputs are judged against the naive tier); the forward runs without an alert.
    The mode is read when the backward runs: an autograd graph recorded with the mode off raises once it is on."""

    S, C, docs = 1000, 512, "perf"
    x, w, cu, dy = _inputs(S, C, docs)
    recorded = pkg.cake_causal_conv1d_nondet(x.clone().requires_grad_(True), w, cu)
    prev, prev_warn = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    try:
        torch.use_deterministic_algorithms(True)
        y = _forward(x, w, cu)
        assert y.dtype == BF16 and tuple(y.shape) == (S, C)
        with pytest.raises(
            RuntimeError, match="cake_causal_conv1d_nondet_train_backward"
        ):
            _backward(x, w, cu, dy)
        xg = x.clone().requires_grad_(True)
        y = pkg.cake_causal_conv1d_nondet(xg, w, cu)
        with pytest.raises(
            RuntimeError, match="cake_causal_conv1d_nondet_train_backward"
        ):
            y.backward(dy)
        with pytest.raises(
            RuntimeError, match="cake_causal_conv1d_nondet_train_backward"
        ):
            recorded.backward(dy)
        torch.use_deterministic_algorithms(True, warn_only=True)

        def call(x, w, cu, dy, b, policy):
            with pytest.warns(
                UserWarning, match="cake_causal_conv1d_nondet_train_backward"
            ):
                return _grads(*_backward(x, w, cu, dy))

        _judge(S, C, docs, call, label="warn_only")
    finally:
        torch.use_deterministic_algorithms(prev, warn_only=prev_warn)


def test_argument_validation() -> None:
    S, C = 64, 256
    x, w, cu, dy = _inputs(S, C, None)
    b = _bias_input(C, F32)
    with pytest.raises(ValueError, match="multiple of 8"):
        _forward(x[:, :130].contiguous(), w[:130].contiguous(), cu)
    with pytest.raises(ValueError, match="width-4"):
        _forward(x, w[:, :3].contiguous(), cu)
    with pytest.raises(ValueError, match="torch.int32"):
        _forward(x, w, cu.long())
    with pytest.raises(ValueError, match="shape"):
        _backward(x, w, cu, dy[: S - 1])
    with pytest.raises(
        ValueError, match="weight must be torch.float32 or torch.bfloat16"
    ):
        _backward(x, w.half(), cu, dy)
    with pytest.raises(ValueError, match="dbias"):
        _backward(x, w, cu, dy, dbias=torch.empty_like(b))
    with pytest.raises(ValueError, match="silu_grad_preactivation"):
        _backward(x, w, cu, dy, policy="fp16")
    for bad, error in (
        (0, ValueError),
        (5, ValueError),
        (2.0, TypeError),
        (True, TypeError),
        ("2", TypeError),
    ):
        with pytest.raises(error, match="bwd_grid_mult must be"):
            _backward(x, w, cu, dy, bwd_grid_mult=bad)
        with pytest.raises(error, match="bwd_grid_mult must be"):
            pkg.cake_causal_conv1d_nondet(x, w, cu, bwd_grid_mult=bad)
        with pytest.raises(error, match="bwd_grid_mult must be"):
            pkg.cake_causal_conv1d_nondet_train_backward_workspace(
                S, C, _device(), bwd_grid_mult=bad
            )
    with pytest.raises(ValueError, match="CUDA"):
        _forward(x.cpu(), w.cpu(), cu.cpu())


def test_fp32_dweight_must_be_16_byte_aligned() -> None:
    """The backward programs store one channel's four FP32 ``dweight`` taps with one 16-byte store: an FP32
    ``dweight`` out off the 16-byte grid is refused (``ValueError``) before anything launches; a 16-byte aligned
    view inside a larger buffer is written, and a BF16 ``dweight`` or an FP32 / BF16 ``dbias`` (stored per
    element) is accepted at any element offset; the written views are judged against the naive tier."""

    S, C, docs = 1000, 512, "perf"
    device = _device()
    x, w, cu, dy = _inputs(S, C, docs, key=8)
    b = _bias_input(C, F32, key=8)
    n = C * WIDTH
    buf = torch.zeros(n + 4, dtype=F32, device=device)
    workspace = pkg.cake_causal_conv1d_nondet_train_backward_workspace(S, C, device)
    for bias in (None, b):
        for offset in (1, 2, 3):
            misaligned = buf[offset : offset + n].view(C, WIDTH)
            assert misaligned.data_ptr() % 16 != 0
            with pytest.raises(ValueError, match="16-byte aligned"):
                _backward(x, w, cu, dy, bias, dweight=misaligned, workspace=workspace)
    torch.cuda.synchronize(device)
    assert (
        int(torch.count_nonzero(buf)) == 0 and int(torch.count_nonzero(workspace)) == 0
    )

    def aligned_view(x, w, cu, dy, b, policy):
        aligned = torch.zeros(n + 4, dtype=F32, device=device)[4:].view(C, WIDTH)
        db_view = torch.zeros(C + 1, dtype=F32, device=device)[1:]
        dx, dw, db = _backward(
            x, w, cu, dy, b, dweight=aligned, dbias=db_view, workspace=workspace
        )
        assert dw is aligned and db is db_view
        return _grads(dx, dw, db)

    _judge(S, C, docs, aligned_view, bias_dtype=F32, key=8, label="16-byte aligned dweight view")
    for dweight_offset, dbias_offset in ((4, 1), (2, 3)):

        def bf16_outs(x, w, cu, dy, b, policy, dweight_offset=dweight_offset, dbias_offset=dbias_offset):
            dw16 = torch.zeros(n + 4, dtype=BF16, device=device)[dweight_offset:][:n].view(
                C, WIDTH
            )
            db16 = torch.zeros(C + 4, dtype=BF16, device=device)[dbias_offset:][:C]
            assert dw16.data_ptr() % 16 != 0
            dx, dw, db = _backward(
                x, w, cu, dy, b, dweight=dw16, dbias=db16, workspace=workspace
            )
            assert dw is dw16 and db is db16
            return _grads(dx, dw, db)

        _judge(
            S,
            C,
            docs,
            bf16_outs,
            bias_dtype=F32,
            key=8,
            label=f"bf16 outs at element offsets {dweight_offset} / {dbias_offset}",
        )
    assert int(torch.count_nonzero(workspace)) == 0


# ----------------------------------------------------------------------------------------------------------------
# The registry against the rendered manifest.


def test_every_route_ships_on_every_architecture() -> None:
    libraries = cake_build.libraries_by_name()
    for arch in loader.ARCHES:
        assert set(loader.ROUTES[arch]) == set(loader.KINDS), arch
        for kind in loader.KINDS:
            for name in loader.route_modules(arch, kind):
                assert loader.library_name(name, arch) in libraries, (arch, kind, name)


def test_registry_records_reference_the_build_manifest() -> None:
    """Every record names its ``build.json`` library (kernel + binding sources under ``csrc/``, digests) and no
    record is left a TODO-EXPORT placeholder; the manifest names this package's import path."""

    manifest = cake_build.manifest()
    assert manifest["schema"] == "cake-build/1"
    assert (
        manifest["package"]
        == "cake_pplx._backends.cake_causal_conv1d_nondet_train"
    )
    libraries = cake_build.libraries_by_name()
    kcp_loader = cake_package(KEY, "kcp_jit")
    referenced = set()
    for registry in (loader.MODULES, kcp_loader.MODULES):
        for name, record in registry.items():
            assert "TODOEXPORT" not in name, f"{name} is still a placeholder"
            assert record["library"] == record["kernel"] == name
            assert record["kernel_symbol"] == f"kernel_{name}"
            for arch in record["arches"]:
                entry = libraries[f"{name}_{arch}"]
                assert entry["sources"] == [
                    f"csrc/{name}_kernel.cu",
                    f"csrc/{name}_binding.cu",
                ]
                referenced.add(entry["name"])
    assert referenced == set(libraries)
