"""FP64 oracles, the naive PyTorch BF16 tier, the accuracy judge, registry
readers and the memory tests' requested-bytes count shared by the suites of
the Cake RMSNorm training packages.

Accuracy has no tolerance (``tests/cake/test_helpers/cake_naive.py``).  Each
floating output -- ``y``, ``rstd`` and ``h_new`` of the forward, ``dx`` and
``dw`` of the backward -- is judged by its error against an FP64 oracle beside
the error of the naive PyTorch BF16 implementation against the same oracle,
over ``DRAWS`` (32) independent random inputs of each case: in the L2 and the
max norm, the median and the largest error over the draws at most ``NEAR_TIE``
(1.005) times the naive's, and 0 on every draw where the naive's is 0
(:class:`Draws`).

The oracles evaluate the kernels' math in float64 from the kernels' own
operands (:func:`oracle_forward`, :func:`oracle_backward`): ``rstd`` and ``y``
from the BF16 rows (the rounded ``h_new`` with a residual) and the FP32 ``eps``
the kernels take, ``h_new`` the exact sum ``x + u`` the kernels round, ``dx``
and ``dw`` at the FP32 ``rstd`` the backward takes.  The naive tier
(:func:`naive_forward`, :func:`naive_backward`) runs the same math as BF16
PyTorch code on the same inputs: ``y`` is ``torch.nn.functional.rms_norm`` (an
FP32 weight is read as stored), ``rstd`` its formula in BF16 ops, ``h_new`` the
BF16 sum, ``dx`` the backward formula with every op in BF16 (the FP32 ``rstd``
rounded to BF16 once) and ``dw`` the weight gradient of ``rms_norm``'s BF16
autograd.  An FP32 weight judged here holds a BF16 draw, so its BF16 cast is
exact: the naive tier and the FP64 oracle see the same weight
(:func:`naive_weight` refuses any other FP32 weight).

Nothing here imports a package: the registry readers take the package's
``cake_jit`` module, so a suite run against a vendored copy uses them unchanged.
"""

from __future__ import annotations

import gc
from collections.abc import Mapping

import torch
from tests.cake.test_helpers import cake_naive

#: The independent random inputs each case is judged over.
DRAWS = cake_naive.DRAWS
# The contract eps of every width.
EPS = {
    6144: 1e-5,
    5120: 1e-6,
    4096: 1e-5,
    2048: 1e-6,
    1536: 1e-6,
    1280: 1e-20,
    1024: 1e-6,
    512: 1e-6,
    256: 1e-6,
    128: 1e-20,
}
W_DTYPES = {"bf16": torch.bfloat16, "f32": torch.float32}
# Rows per block of the float64 oracles and the BF16 naive backward (a whole-T
# float64 copy of [65536, 6144] is 3.2 GB per tensor).
BLOCK_ROWS = 8192


def draw_seed(base: int, draw: int) -> int:
    """The seed of draw ``draw`` of a case whose own seed is ``base``."""
    return cake_naive.draw_seed(base, draw)


def eps32(eps: float) -> float:
    """``eps`` as the kernels evaluate it: the FP32 runtime operand."""
    return float(torch.tensor(eps, dtype=torch.float32))


def _rows2d(t: torch.Tensor) -> torch.Tensor:
    return t.reshape(-1, t.shape[-1])


def oracle_forward(
    x: torch.Tensor, w: torch.Tensor, eps: float, residual: torch.Tensor | None = None
) -> dict[str, torch.Tensor]:
    """The forward's float64 oracle over the last axis: ``y`` and ``rstd`` (the
    statistics of the BF16-rounded ``h_new`` with a residual, the FP32 ``eps``),
    and with a residual ``h_new``, the exact sum ``x + u`` the kernel rounds (the
    float64 sum of two BF16 values is exact)."""
    out = {}
    normalized = x
    if residual is not None:
        exact = x.double() + residual.double()
        normalized = exact.to(torch.bfloat16)
        out["h_new"] = exact
    xd = normalized.double()
    rstd = torch.rsqrt(xd.square().mean(dim=-1) + eps32(eps))
    out["y"] = (xd * rstd.unsqueeze(-1)) * w.double()
    out["rstd"] = rstd
    return out


def naive_weight(w: torch.Tensor) -> torch.Tensor:
    """The weight as the naive BF16 tier reads it: a BF16 weight as it is; an
    FP32 weight cast to BF16, which must be exact (the weight holds a BF16
    draw), so the naive tier and the FP64 oracle see the same weight."""
    if w.dtype == torch.bfloat16:
        return w
    wb = w.to(torch.bfloat16)
    assert torch.equal(wb.to(w.dtype), w), (
        "a weight judged against the naive BF16 tier holds BF16 values "
        f"(this {w.dtype} weight does not)"
    )
    return wb


def naive_forward(
    x: torch.Tensor, w: torch.Tensor, eps: float, residual: torch.Tensor | None = None
) -> dict[str, torch.Tensor]:
    """The naive PyTorch BF16 forward: with a residual ``h_new`` the BF16 sum
    ``x + u`` and the statistics of it; ``y`` ``torch.nn.functional.rms_norm`` of
    the rows with the weight as stored (an FP32 weight, which holds BF16 values
    (:func:`naive_weight`), stays FP32, so the product runs in FP32 by type
    promotion) cast to BF16; ``rstd`` the op's formula ``rsqrt(mean(x^2) +
    eps)`` in BF16 ops (the public op returns no reciprocal RMS)."""
    naive_weight(w)  # holds BF16 values; the forward reads it as stored
    h = x if residual is None else x + residual
    weight = w if w.dtype == torch.float32 else w.to(torch.bfloat16)
    out = {
        "y": torch.nn.functional.rms_norm(h, (h.shape[-1],), weight, eps).to(
            torch.bfloat16
        ),
        "rstd": torch.rsqrt(h.square().mean(dim=-1) + eps),
    }
    if residual is not None:
        out["h_new"] = h
    return out


def oracle_dw(g: torch.Tensor, x: torch.Tensor, rstd: torch.Tensor) -> torch.Tensor:
    """The float64 ``dw = sum(g x r)`` over every leading axis at the FP32
    ``rstd`` the backward takes, in blocks of ``BLOCK_ROWS`` rows."""
    g2, x2, r2 = _rows2d(g), _rows2d(x), rstd.reshape(-1)
    dw = torch.zeros(x2.shape[1], dtype=torch.float64, device=x.device)
    for start in range(0, x2.shape[0], BLOCK_ROWS):
        block = slice(start, min(start + BLOCK_ROWS, x2.shape[0]))
        dw += (g2[block].double() * x2[block].double() * r2[block, None].double()).sum(
            dim=0
        )
    return dw


def oracle_backward(
    g: torch.Tensor,
    x: torch.Tensor,
    w: torch.Tensor,
    rstd: torch.Tensor,
    g_h: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """The backward's float64 oracle at the FP32 ``rstd`` the backward takes,
    ``dx = r a - x r^3 c (+ g_h)`` (``a = g w``, ``c = sum(a x) / H``) per row and
    ``dw = sum(g x r)`` over every leading axis (:func:`oracle_dw`), in blocks of
    ``BLOCK_ROWS`` rows."""
    hidden = x.shape[-1]
    g2, x2 = _rows2d(g), _rows2d(x)
    gh2 = _rows2d(g_h) if g_h is not None else None
    r2 = rstd.reshape(-1)
    rows = x2.shape[0]
    wd = w.double()
    dx = torch.empty(rows, hidden, dtype=torch.float64, device=x.device)
    for start in range(0, rows, BLOCK_ROWS):
        block = slice(start, min(start + BLOCK_ROWS, rows))
        gb, xb = g2[block].double(), x2[block].double()
        r = r2[block, None].double()
        a = gb * wd
        c = (a * xb).sum(dim=-1, keepdim=True) / hidden
        dx[block] = r * a - xb * r.pow(3) * c
        if gh2 is not None:
            dx[block] += gh2[block].double()
    return {"dx": dx.reshape(x.shape), "dw": oracle_dw(g, x, rstd)}


def naive_dw(g: torch.Tensor, x: torch.Tensor, eps: float) -> torch.Tensor:
    """The naive BF16 weight gradient: ``.grad`` of a BF16 weight through
    ``torch.nn.functional.rms_norm`` over the rows of ``x`` in BF16 (the forward
    and the autograd backward in BF16, ``eps``, its own ``rstd`` from ``x``).
    The weight gradient does not depend on the weight's value: a weight of ones
    stands for it."""
    hidden = x.shape[-1]
    with torch.inference_mode(False), torch.enable_grad():
        weight = torch.ones(
            hidden, dtype=torch.bfloat16, device=x.device, requires_grad=True
        )
        y = torch.nn.functional.rms_norm(
            _rows2d(x).detach().to(torch.bfloat16), (hidden,), weight, eps
        )
        y.backward(_rows2d(g).detach().to(torch.bfloat16))
    return weight.grad


def naive_backward(
    g: torch.Tensor,
    x: torch.Tensor,
    w: torch.Tensor,
    rstd: torch.Tensor,
    eps: float,
    g_h: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """The naive PyTorch BF16 backward as BF16 tensors.  ``dx`` is
    :func:`oracle_backward`'s formula with every op's inputs and outputs in BF16
    (the FP32 ``rstd`` rounded to BF16 once; an FP32 weight, which holds BF16
    values, cast exactly: :func:`naive_weight`), in the same blocks.  ``dw`` is
    the weight gradient of ``torch.nn.functional.rms_norm`` over the rows of
    ``x`` in BF16 (``x``, the weight and ``g`` as BF16, the forward and the
    autograd backward in BF16, ``eps``): it computes its own ``rstd`` from
    ``x``; ``g_h`` adds to ``dx`` only."""
    hidden = x.shape[-1]
    g2, x2 = _rows2d(g), _rows2d(x)
    gh2 = _rows2d(g_h) if g_h is not None else None
    r_all = rstd.reshape(-1).to(torch.bfloat16)
    wb = naive_weight(w)
    rows = x2.shape[0]
    dx = torch.empty(rows, hidden, dtype=torch.bfloat16, device=x.device)
    for start in range(0, rows, BLOCK_ROWS):
        block = slice(start, min(start + BLOCK_ROWS, rows))
        gb, xb, r = g2[block], x2[block], r_all[block, None]
        a = gb * wb
        c = (a * xb).sum(dim=-1, keepdim=True) / hidden
        dx[block] = r * a - xb * r**3 * c
        if gh2 is not None:
            dx[block] += gh2[block]
    return {"dx": dx.reshape(x.shape), "dw": naive_dw(g, x, eps)}


class Draws:
    """The accuracy of one case's outputs over its draws: a
    ``cake_naive.TierJudge`` over ``DRAWS`` inputs.  :meth:`add` holds each
    output to its contract (``cake_naive.contract``) and records its error and
    the naive tier's as one draw (a second launch on a draw keeps the larger
    error); :meth:`forward` and :meth:`backward` evaluate the oracle and the
    naive tier of a launch; :meth:`check` records the verdicts on the test
    (``record_property``) and fails with every miss.  ``prefix`` names the
    outputs of one launch form among several judged in one test."""

    def __init__(self, label: str, prefix: str = "", draws: int = DRAWS) -> None:
        self.label = label
        self.prefix = prefix
        self.judge = cake_naive.TierJudge(draws)
        self.messages: list[str] = []

    @property
    def draws(self) -> int:
        return self.judge.draws

    def add(
        self,
        draw: int,
        outputs: Mapping[str, torch.Tensor],
        oracle: Mapping[str, torch.Tensor],
        naive: Mapping[str, torch.Tensor],
    ) -> None:
        for key, actual in outputs.items():
            name = f"{self.prefix}{key}"
            expected = oracle[key]
            message = cake_naive.contract(name, actual, expected)
            if message:
                self.messages.append(f"draw {draw}: {message}")
                if actual.shape != expected.shape:
                    continue
            self.judge.add(name, draw, actual, expected, {"bf16": naive[key]})

    def forward(
        self,
        draw: int,
        outputs: Mapping[str, torch.Tensor],
        x: torch.Tensor,
        w: torch.Tensor,
        eps: float,
        residual: torch.Tensor | None = None,
    ) -> None:
        """Judge a forward launch's ``outputs`` (``y``, ``rstd`` and with a
        residual ``h_new``) on the inputs of draw ``draw``."""
        self.add(
            draw,
            outputs,
            oracle_forward(x, w, eps, residual),
            naive_forward(x, w, eps, residual),
        )

    def backward(
        self,
        draw: int,
        outputs: Mapping[str, torch.Tensor],
        g: torch.Tensor,
        x: torch.Tensor,
        w: torch.Tensor,
        rstd: torch.Tensor,
        eps: float,
        g_h: torch.Tensor | None = None,
    ) -> None:
        """Judge a backward launch's ``outputs`` (``dx`` and / or ``dw``) on the
        inputs of draw ``draw`` (``x`` the normalized input, ``rstd`` the FP32
        one the backward took)."""
        oracle = oracle_backward(g, x, w, rstd, g_h)
        naive = naive_backward(g, x, w, rstd, eps, g_h)
        self.add(draw, outputs, oracle, naive)

    def failures(self) -> list[str]:
        return self.messages + self.judge.failures()

    def check(self, record_property=None) -> None:
        """Record the verdicts on the running test and fail with every contract
        message and every output that missed the naive tier."""
        if record_property is not None:
            self.judge.record(record_property)
        failures = self.failures()
        assert not failures, f"{self.label}: " + "; ".join(failures[:12])


def check_all(record_property, judges) -> None:
    """:meth:`Draws.check` of the several judges of one test: record every
    judge's verdicts, then fail with every judge's misses."""
    failures = []
    for judge in judges:
        if record_property is not None:
            judge.judge.record(record_property)
        failures += [f"{judge.label}: {message}" for message in judge.failures()]
    assert not failures, "; ".join(failures[:12])


class BackwardCalls:
    """The judge of a backward case launched many times on its own input (timed
    calls, launch grids, fresh workspaces): every call's ``dx`` and ``dw`` are
    draw 0 of a :class:`Draws` (the largest error over the calls, :meth:`add`);
    :meth:`check` launches once on each further draw (``draw_inputs(draw)`` gives
    ``(g, x, w, rstd, g_h)``, ``launch(g, x, w, rstd, g_h)`` returns ``(dx,
    dw)``), records the verdicts and fails with every miss."""

    def __init__(self, label, g, x, w, rstd, eps, g_h=None) -> None:
        self.eps = eps
        self.judge = Draws(label)
        self.oracle = oracle_backward(g, x, w, rstd, g_h)
        self.naive = naive_backward(g, x, w, rstd, eps, g_h)

    def add(self, dx: torch.Tensor, dw: torch.Tensor) -> None:
        self.judge.add(0, {"dx": dx, "dw": dw}, self.oracle, self.naive)

    def check(self, record_property, draw_inputs, launch) -> None:
        self.oracle = self.naive = None
        for draw in range(1, self.judge.draws):
            g, x, w, rstd, g_h = draw_inputs(draw)
            dx, dw = launch(g, x, w, rstd, g_h)
            self.judge.backward(
                draw, {"dx": dx, "dw": dw}, g, x, w, rstd, self.eps, g_h
            )
        self.judge.check(record_property)


def bf16_ulp_differences(a: torch.Tensor, b: torch.Tensor) -> tuple[int, int]:
    """``(differing elements, largest difference in BF16 ulps)`` of two BF16 tensors
    of one shape (the ulp distance of the ordered integer encodings; a sign change
    counts through zero)."""
    assert a.dtype == b.dtype == torch.bfloat16 and a.shape == b.shape
    ia = a.contiguous().view(torch.int16).to(torch.int32)
    ib = b.contiguous().view(torch.int16).to(torch.int32)
    differing = int((ia != ib).sum().item())
    if differing == 0:
        return 0, 0
    # monotone integer order of the encodings: negative values mirror below zero
    oa = torch.where(ia < 0, -(ia & 0x7FFF), ia)
    ob = torch.where(ib < 0, -(ib & 0x7FFF), ib)
    return differing, int((oa - ob).abs().max().item())


def fp32_ulp_differences(a: torch.Tensor, b: torch.Tensor) -> tuple[int, int]:
    """:func:`bf16_ulp_differences` for FP32 tensors."""
    assert a.dtype == b.dtype == torch.float32 and a.shape == b.shape
    ia = a.contiguous().view(torch.int32).to(torch.int64)
    ib = b.contiguous().view(torch.int32).to(torch.int64)
    differing = int((ia != ib).sum().item())
    if differing == 0:
        return 0, 0
    oa = torch.where(ia < 0, -(ia & 0x7FFFFFFF), ia)
    ob = torch.where(ib < 0, -(ib & 0x7FFFFFFF), ib)
    return differing, int((oa - ob).abs().max().item())


# --------------------------------------------------------------------------- registry readers


def admits(rule: dict | None, rows: int) -> bool:
    if rule is None:
        return True
    assert rule["kind"] == "rows_between", rule
    return rows > int(rule["above"]) and (
        rule["at_most"] is None or rows <= int(rule["at_most"])
    )


def window_records(
    loader, arch: str, hidden: int, kind: str, residual: bool, w: str, rows: int
) -> list[str]:
    """The records one call of ``rows`` tokens launches, read from ``ROUTES`` and
    the select rules (independent of the host's ``selected_modules``)."""
    key = loader.route_key(hidden, kind, residual, w)
    return [
        name
        for name in loader.ROUTES[arch][key]
        if admits(loader.MODULES[name]["select_rule"], rows)
    ]


#: The workspace kinds of a single-kernel backward window: one kernel that adds the CTAs' FP32 ``dw`` sums
#: (``claim_sum_f32``) or keeps one FP32 partial row per CTA (``claim_part_f32``).
SINGLE_KINDS = ("claim_sum_f32", "claim_part_f32")


def form(loader, names: list[str]) -> str:
    """``single`` (one ``claim_sum_f32`` or ``claim_part_f32`` kernel) or
    ``pair`` (a ``chunk_partials`` main kernel and its dw reduce) of one
    backward window's records."""
    records = [loader.MODULES[name] for name in names]
    owners = [r["workspace_rule"]["kind"] for r in records if r["workspace_rule"]]
    reduces = [r for r in records if r["role"] == "bwd_reduce"]
    if len(records) == 1 and len(owners) == 1 and owners[0] in SINGLE_KINDS:
        return "single"
    if len(records) == 2 and owners == ["chunk_partials"] and len(reduces) == 1:
        return "pair"
    raise AssertionError(f"a backward window launches {names}: neither form")


def windows(loader, arch: str, hidden: int, kind: str, residual: bool, w: str):
    """``[(above, at_most, library)]`` of the route on ``arch`` (``program_windows``)."""
    return list(loader.program_windows(arch, hidden, kind, residual, w))


def requested_bytes(device: torch.device) -> int:
    """Device bytes this process requested from the CUDA caching allocator on
    ``device`` and still holds, after a garbage collection (a tensor kept only
    by a reference cycle is freed first).  ``torch.cuda.memory_allocated``
    counts whole allocator blocks instead: every request rounded up to a
    multiple of 512 bytes, and a cached block up to 1 MiB larger served
    without splitting it, so the blocks it counts depend on what earlier tests
    left in the cache.  The count is a statistic of the native allocator:
    under another backend (``cudaMallocAsync`` or a pluggable allocator) this
    fails instead of returning a count that backend does not keep."""
    backend = torch.cuda.get_allocator_backend()
    assert backend == "native", (
        f"the requested-bytes count needs the native CUDA caching allocator, "
        f"not {backend!r} (PYTORCH_CUDA_ALLOC_CONF backend)"
    )
    gc.collect()
    return torch.cuda.memory_stats(device)["requested_bytes.all.current"]


def workspace_at_rest(loader, ws: torch.Tensor) -> bool:
    """The documented rest state after every call: every byte before the pair
    partial rows (``PAIR_ROWS_OFFSET``) is zero."""
    torch.cuda.synchronize()
    return not bool(ws[: min(loader.PAIR_ROWS_OFFSET, ws.numel())].any())
