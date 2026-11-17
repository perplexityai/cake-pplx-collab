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

CUDA-graph capture of the entry points of the Cake four-stream mHC training
package ``cake_pplx._backends.cake_mhc_nondet_train`` (hidden 4096,
5120 and 7168, 4 streams; SM100 / SM103 at the three hidden sizes, SM107 at
4096), in the
form of ``test_cake_mhc_det_train_graphs.py``.  Every case runs at each hidden
size ``H`` the device's architecture routes (``W`` is ``[H, H]``, the
workspaces are sized with ``hidden=H``); a size it does not route is skipped.

One training iteration of a layer -- prepare, an external branch ``B = U @ W``
(BF16), finish, the loss and its backward -- is captured whole with
``torch.cuda.graph`` after a warm-up on a side stream (the JIT modules and the
per-device default workspaces exist before the capture), in three forms:

* the package's direct entry points ``forward_prepare`` -> ``forward_finish``
  -> ``backward_prepare`` -> ``backward_finish`` (the branch backward
  ``dU = dB @ W^T`` in between);
* its autograd entry points ``cake_mhc_nondet_train_prepare`` /
  ``cake_mhc_nondet_train_finish`` and ``loss.backward()`` (``save_stats`` both
  ways);
* the same layer inside a non-reentrant ``torch.utils.checkpoint``
  (``preserve_rng_state=False``: the layer draws no random numbers);

and the autograd / checkpointed layer through
``torch.cuda.make_graphed_callables``.  The captured forms run with the
default workspaces and with caller-owned workspaces of the package (the
backward workspace of this package's sizing; its bytes past the zero prefix --
the ticket region and the column slab's arrival counts -- overwritten with 0xFF
before every replay, its zero prefix zero after every replay).

Each captured iteration is replayed on 32 inputs, each copied into the
captured inputs in place, after its static outputs and gradients were
overwritten with 0xFF bytes; the graphed callables get the 32 inputs as new
arguments.  Every output and gradient of a replay is judged against the naive
BF16 tier (``tests/cake_naive.py``): its error against the float64 oracle of
the layer iteration, in the L2 and the max norm, median and max over the 32
inputs, is at most 1.005 times the tier's, and zero where the tier's is zero.
The oracle evaluates the branch ``B = U @ W`` and its backward
``dU = dB @ W^T`` in float64; the tier evaluates the whole iteration with
PyTorch in BF16 (the branch on the tier's own ``U`` and ``dB``).  The package
makes no bit promise, so a replay is not compared bit for bit with an eager
call of this package or of the deterministic package.  A captured
``backward_finish`` on a caller workspace sized for a larger token count,
replayed after an eager call at the larger count on the same workspace (both
judged against the naive tier of their inputs), closes the module.
"""

from __future__ import annotations

import gc

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from tests.cake.test_helpers.cake_mhc_naive import (
    case_oracle,
    expected_and_naive,
    judge_draws,
    package_outputs,
)
from tests.cake.test_helpers.cake_package import cake_package

KEY = "cake.cake_mhc_nondet_train"
pkg = cake_package(KEY)
loader = cake_package(KEY, "cake_jit")

SUPPORTED_CAPABILITIES = ((10, 0), (10, 3), (10, 7))
STREAMS, HIDDEN, MAPPING_ROWS = 4, 4096, 24
HIDDEN_SIZES = (4096, 5120, 7168)
SEED = 2032
# The backward workspace's zero prefix: the ticket region (256 bytes) and the column slab's arrival counts (512 bytes),
# first for every token count and hidden size; every call finds it zero and leaves it zero.
ZERO_PREFIX_BYTES = 768
BF16, F32, F64 = torch.bfloat16, torch.float32, torch.float64
_OUTPUTS = ("dX", "dfn", "dscale", "dbase")
# The inputs of one layer iteration: copied into a captured iteration's static tensors before each replay.
_STATIC = ("X", "fn", "scale", "base", "W", "dY")


def _device() -> torch.device:
    return torch.device("cuda:0")


def _capability() -> tuple[int, int]:
    return tuple(torch.cuda.get_device_capability(_device()))


def _supported() -> bool:
    return (
        torch.cuda.is_available()
        and _capability() in SUPPORTED_CAPABILITIES
        and pkg.is_cake_mhc_nondet_train_supported(_device())
        and loader.route_applies(
            device_capability=_capability(), hidden=HIDDEN, kind="bwd_finish"
        )
    )


pytestmark = pytest.mark.skipif(
    not _supported(),
    reason="cake_mhc_nondet_train needs an SM100 / SM103 / SM107 device with exported routes",
)


@pytest.fixture(autouse=True, scope="module")
def warmup_jit():
    # The package's libraries of the local architecture; a prebuilt or cached library loads without compiling.
    if _supported():
        for spec in loader.specs_for_arch(_arch()):
            spec.build_and_load()
    yield


@pytest.fixture(autouse=True)
def collect_dead_graphs():
    # torch skips its gc.collect() before a capture by default (TORCH_CUDAGRAPH_GC): an earlier test's dead CUDA graph
    # that a cyclic collection destroys during a global-mode capture invalidates that capture.
    gc.collect()
    yield
    gc.collect()


def _arch() -> str:
    return loader.arch_for_capability(_capability())


@pytest.fixture(params=HIDDEN_SIZES, ids=lambda h: f"H{h}")
def hidden(request) -> int:
    """The hidden size of a case (every size of the package; skipped where the architecture has no route)."""
    arch = _arch()
    sizes = tuple(loader.supported_hidden_sizes(arch)) if arch else ()
    if request.param not in sizes:
        pytest.skip(
            f"hidden {request.param} has no exported route on {arch} (exported hidden sizes: {sizes})"
        )
    return request.param


# ----------------------------------------------------------------------------------------------------------------
# The oracle and the naive tier of one layer iteration


def _layer_expected(i: dict, names) -> tuple[dict, dict]:
    """The oracle's (float64) and the naive tier's (BF16) values of the outputs ``names`` of one layer iteration on
    ``i``: the branch ``B = U @ W`` and its backward ``dU = dB @ W^T`` in the precision of the rest (the tier's
    PyTorch BF16 matmuls on the tier's own ``U`` and ``dB``).  ``U`` and ``dB`` depend on neither ``B`` nor ``dU``, so
    a first evaluation gives them and a second one, on that ``B`` and ``dU``, every output."""
    names = tuple(names)
    parts = []
    for dtype in (F64, BF16):
        W = i["W"].to(dtype)
        zero = torch.zeros(i["B"].shape, dtype=dtype, device=_device())
        first = case_oracle({**i, "B": zero, "dU": zero}, dtype=dtype)
        B = first["U"].to(dtype) @ W
        dU = first["dB"].to(dtype) @ W.t()
        del first, zero
        parts.append(case_oracle({**i, "B": B, "dU": dU}, dtype=dtype))
    want, tier = parts
    return {k: want[k] for k in names}, {k: tier[k].to(BF16) for k in names}


# ----------------------------------------------------------------------------------------------------------------
# Inputs, steps, capture and replay


def _inputs(T: int, seed: int, hidden: int = HIDDEN) -> dict:
    g = torch.Generator(device=_device()).manual_seed(seed)

    def randn(*shape: int, dtype: torch.dtype = F32) -> torch.Tensor:
        return torch.randn(*shape, generator=g, device=_device(), dtype=F32).to(dtype)

    return {
        "X": randn(T, STREAMS, hidden, dtype=BF16),
        "fn": (randn(MAPPING_ROWS, STREAMS * hidden) * 1e-2).contiguous(),
        "scale": 1.0 + 0.1 * randn(3),
        "base": 0.1 * randn(MAPPING_ROWS),
        "W": (randn(hidden, hidden) / 64.0).to(BF16),
        "dY": randn(T, STREAMS, hidden, dtype=BF16),
        # the branch output and its gradient of the backward_finish-only capture
        "B": randn(T, hidden, dtype=BF16),
        "dU": randn(T, hidden, dtype=BF16),
    }


def _workspaces(T: int, caller_owned: bool, hidden: int = HIDDEN):
    if not caller_owned:
        return None, None
    return (
        pkg.cake_mhc_nondet_train_forward_workspace(T, _device(), hidden=hidden),
        pkg.cake_mhc_nondet_train_backward_workspace(T, _device(), hidden=hidden),
    )


def _leaves(i: dict) -> list:
    return [
        i[k].detach().clone().requires_grad_(True) for k in ("X", "fn", "scale", "base")
    ]


def _layer(W: torch.Tensor, wsf, wsb, save_stats: bool):
    def layer(X, fn, scale, base):
        U, handle = pkg.cake_mhc_nondet_train_prepare(
            X,
            fn,
            scale,
            base,
            save_stats=save_stats,
            workspace=wsf,
            backward_workspace=wsb,
        )
        return pkg.cake_mhc_nondet_train_finish(X, U @ W, handle)

    return layer


def _direct_step(i: dict, wsf, wsb):
    """One iteration through the direct entry points -> (the step, its leaves (none), its static inputs)."""
    X, fn, scale, base, W, dY = (i[k] for k in _STATIC)

    def step() -> dict:
        U, stats = pkg.cake_mhc_nondet_train_forward_prepare(
            X, fn, scale, base, workspace=wsf
        )
        B = U @ W
        Y = pkg.cake_mhc_nondet_train_forward_finish(X, B, stats)
        dB, gstats = pkg.cake_mhc_nondet_train_backward_prepare(dY, B, X, stats)
        dX, dfn, dscale, dbase = pkg.cake_mhc_nondet_train_backward_finish(
            X, dY, dB @ W.t(), B, fn, scale, base, stats, gstats, workspace=wsb,
        )  # fmt: skip
        return {
            "U": U,
            "stats": stats,
            "Y": Y,
            "dB": dB,
            "gstats": gstats,
            "dX": dX,
            "dfn": dfn,
            "dscale": dscale,
            "dbase": dbase,
        }

    return step, (), {k: i[k] for k in _STATIC}


def _autograd_step(i: dict, wsf, wsb, *, save_stats: bool, ckpt: bool):
    """One iteration through the autograd entry points (inside a checkpoint with ``ckpt``) -> (the step, its leaves,
    its static inputs: the leaves, ``W`` and the FP32 copy of ``dY`` the loss reads)."""
    layer = _layer(i["W"], wsf, wsb, save_stats)
    leaves = _leaves(i)
    dY = i["dY"].float()

    def step() -> dict:
        if ckpt:
            Y = checkpoint(
                layer, *leaves, use_reentrant=False, preserve_rng_state=False
            )
        else:
            Y = layer(*leaves)
        (Y.float() * dY).sum().backward()
        return {"Y": Y, **_grads(leaves)}

    statics = dict(zip(("X", "fn", "scale", "base"), leaves, strict=True))
    statics.update({"W": i["W"], "dY": dY})
    return step, leaves, statics


def _grads(leaves) -> dict:
    """The gradients of the leaves ``X``, ``fn``, ``scale``, ``base`` under the package's output names."""
    return {
        name: t.grad
        for name, t in zip(("dX", "dfn", "dscale", "dbase"), leaves, strict=True)
    }


def _reset(leaves) -> None:
    for t in leaves:
        t.grad = None


def _snapshot(out: dict) -> dict:
    return {k: v.detach().clone() for k, v in out.items()}


def _poison(out: dict) -> None:
    for v in out.values():
        v.detach().reshape(-1).view(torch.uint8).fill_(0xFF)


def _poisoned_outputs(T: int, hidden: int = HIDDEN) -> dict:
    """Caller-provided outputs holding NaN in every element (what the kernels must overwrite)."""
    nan = float("nan")
    return {
        "dX": torch.full((T, STREAMS, hidden), nan, dtype=BF16, device=_device()),
        "dfn": torch.full(
            (MAPPING_ROWS, STREAMS * hidden), nan, dtype=F32, device=_device()
        ),
        "dscale": torch.full((3,), nan, dtype=F32, device=_device()),
        "dbase": torch.full((MAPPING_ROWS,), nan, dtype=F32, device=_device()),
    }


def _poison_workspace(ws: torch.Tensor) -> None:
    """Every byte past the zero prefix (the ticket region and the arrival counts, first for every token count) set
    to 0xFF."""
    ws[ZERO_PREFIX_BYTES:].fill_(0xFF)


def _assert_zero_prefix(ws: torch.Tensor, what: str) -> None:
    torch.cuda.synchronize()
    nonzero = int(torch.count_nonzero(ws[:ZERO_PREFIX_BYTES]))
    assert nonzero == 0, (
        f"{what}: {nonzero} non-zero bytes in the ticket region and the arrival counts"
    )


def _capture(make):
    """``make()``'s iteration captured in a ``torch.cuda.CUDAGraph`` after three warm-up iterations on a side
    stream -> (the graph, its static outputs, its static inputs)."""
    step, leaves, statics = make()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            _reset(leaves)
            step()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    _reset(leaves)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = step()
    return graph, out, statics


def _replay(graph, out: dict, ws, what: str) -> dict:
    """One replay after the static outputs (and the bytes of ``ws`` past the zero prefix) were overwritten with
    0xFF; ``ws``'s zero prefix is zero after it."""
    _poison(out)
    if ws is not None:
        _poison_workspace(ws)
    graph.replay()
    torch.cuda.synchronize()
    if ws is not None:
        _assert_zero_prefix(ws, what)
    return _snapshot(out)


def _judge_captured(
    make, T: int, hidden: int, ws, what: str, record_property, *, seed: int
) -> None:
    """``make()``'s iteration captured once, then replayed on 32 inputs copied into its static inputs: every output
    and gradient of each replay judged against the naive tier of the layer iteration."""
    graph, out, statics = _capture(make)

    def case(j: int, s: int):
        new = _inputs(T, seed=s, hidden=hidden)
        with torch.no_grad():
            for k, t in statics.items():
                t.copy_(new[k])
        got = package_outputs(_replay(graph, out, ws, f"{what} replay {j}"))
        expected, naive = _layer_expected(new, got)
        return [got], expected, naive

    judge_draws(case, record_property, seed=seed)


def _forward(i: dict) -> dict:
    """This package's forward prepare and backward prepare of a case: the ``stats`` / ``grad_stats`` records
    ``backward_finish`` takes."""
    _U, stats = pkg.cake_mhc_nondet_train_forward_prepare(
        i["X"], i["fn"], i["scale"], i["base"]
    )
    _dB, gstats = pkg.cake_mhc_nondet_train_backward_prepare(
        i["dY"], i["B"], i["X"], stats
    )
    return {"stats": stats, "gstats": gstats}


def _nd_backward(i: dict, prep: dict, **kwargs) -> dict:
    out = pkg.cake_mhc_nondet_train_backward_finish(
        i["X"], i["dY"], i["dU"], i["B"], i["fn"], i["scale"], i["base"], prep["stats"], prep["gstats"], **kwargs,
    )  # fmt: skip
    return dict(zip(_OUTPUTS, out, strict=True))


# ----------------------------------------------------------------------------------------------------------------
# Tests


@pytest.mark.parametrize("T", (1, 129, 2049))
@pytest.mark.parametrize("caller_owned", (False, True))
def test_direct_entry_points_capture(
    T: int, caller_owned: bool, hidden: int, record_property
) -> None:
    i = _inputs(T, seed=7046 + T, hidden=hidden)
    wsf, wsb = _workspaces(T, caller_owned, hidden)
    what = f"direct T={T} H={hidden} caller_owned={caller_owned}"
    _judge_captured(
        lambda: _direct_step(i, wsf, wsb),
        T,
        hidden,
        wsb,
        what,
        record_property,
        seed=7046 + T,
    )


@pytest.mark.parametrize("T", (1, 129, 2049))
@pytest.mark.parametrize("save_stats", (True, False))
@pytest.mark.parametrize("ckpt", (False, True))
@pytest.mark.parametrize("caller_owned", (False, True))
def test_autograd_iteration_capture(
    T: int,
    save_stats: bool,
    ckpt: bool,
    caller_owned: bool,
    hidden: int,
    record_property,
) -> None:
    i = _inputs(T, seed=7146 + T, hidden=hidden)
    wsf, wsb = _workspaces(T, caller_owned, hidden)
    what = f"autograd ckpt={ckpt} save_stats={save_stats} caller_owned={caller_owned} T={T} H={hidden}"
    _judge_captured(
        lambda: _autograd_step(i, wsf, wsb, save_stats=save_stats, ckpt=ckpt),
        T,
        hidden,
        wsb,
        what,
        record_property,
        seed=7146 + T,
    )


@pytest.mark.parametrize("T", (129, 2049))
@pytest.mark.parametrize("save_stats", (True, False))
@pytest.mark.parametrize("ckpt", (False, True))
def test_make_graphed_callables(
    T: int, save_stats: bool, ckpt: bool, hidden: int, record_property
) -> None:
    """The layer through ``torch.cuda.make_graphed_callables`` (``W`` fixed: the first input set's), called on 32 new
    input sets of the same shapes (a call whose graphs did not replay would return the previous call's outputs):
    ``Y`` and the gradients of each call judged against the naive tier of the layer iteration."""
    i = _inputs(T, seed=7246 + T, hidden=hidden)
    layer = _layer(i["W"], None, None, save_stats)

    def fwd(X, fn, scale, base):
        if not ckpt:
            return layer(X, fn, scale, base)
        return checkpoint(
            layer, X, fn, scale, base, use_reentrant=False, preserve_rng_state=False
        )

    graphed = torch.cuda.make_graphed_callables(
        fwd, tuple(_leaves(i)), num_warmup_iters=3
    )

    def case(j: int, s: int):
        new = {**_inputs(T, seed=s, hidden=hidden), "W": i["W"]}
        leaves = _leaves(new)
        Y = graphed(*leaves)
        (Y.float() * new["dY"].float()).sum().backward()
        got = _snapshot({"Y": Y, **_grads(leaves)})
        expected, naive = _layer_expected(new, got)
        return [got], expected, naive

    judge_draws(case, record_property, seed=7246 + T)


@pytest.mark.parametrize(
    ("T", "t_ws"), ((129, 129), (2049, 2049), (2049, 16263), (1, 4096))
)
def test_cuda_graph_capture_and_replay(
    T: int, t_ws: int, hidden: int, record_property
) -> None:
    """This package's ``backward_finish`` with a caller workspace (sized for ``t_ws >= T``) and caller outputs,
    captured in a ``torch.cuda.CUDAGraph`` after a warm-up on a side stream, then replayed on 32 inputs copied in
    place (and, on a larger buffer, after an eager call at ``t_ws`` on new inputs on the same workspace): every replay
    and every eager call leaves the zero prefix zero, and their outputs are judged against the naive tier of their
    inputs (the eager call's under the name prefix ``eager T=<t_ws>``)."""
    dev = _device()
    ws = pkg.cake_mhc_nondet_train_backward_workspace(t_ws, dev, hidden=hidden)
    static = _inputs(T, seed=SEED + 600, hidden=hidden)
    sprep = _forward(static)
    outs = _poisoned_outputs(T, hidden)

    def step() -> None:
        pkg.cake_mhc_nondet_train_backward_finish(
            static["X"], static["dY"], static["dU"], static["B"], static["fn"], static["scale"], static["base"],
            sprep["stats"], sprep["gstats"], workspace=ws, **outs,
        )  # fmt: skip

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            step()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    torch.cuda.synchronize()

    def case(j: int, s: int):
        new = _inputs(T, seed=s, hidden=hidden)
        nprep = _forward(new)
        for k in ("X", "dY", "dU", "B", "fn", "scale", "base"):
            static[k].copy_(new[k])
        sprep["stats"].copy_(nprep["stats"])
        sprep["gstats"].copy_(nprep["gstats"])
        what = f"replay {j} at T={T} H={hidden} on a workspace of T={t_ws}"
        launch, expected, naive = {}, {}, {}
        if t_ws > T:
            big = _inputs(t_ws, seed=s + 1, hidden=hidden)
            _poison_workspace(ws)
            got = _snapshot(_nd_backward(big, _forward(big), workspace=ws))
            _assert_zero_prefix(ws, f"eager T={t_ws} before {what}")
            e, t = expected_and_naive(big, _OUTPUTS)
            prefix = f"eager T={t_ws} "
            for d, part in zip((launch, expected, naive), (got, e, t), strict=True):
                d.update({prefix + k: v for k, v in part.items()})
            del big, got, e, t
        _poison_workspace(ws)
        for v in outs.values():
            v.fill_(float("nan"))
        graph.replay()
        _assert_zero_prefix(ws, what)
        launch.update(_snapshot(outs))
        e, t = expected_and_naive(new, _OUTPUTS)
        expected.update(e)
        naive.update(t)
        return [launch], expected, naive

    judge_draws(case, record_property, seed=SEED + 610)
