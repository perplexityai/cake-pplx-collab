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

Workspace reuse across token counts of the Cake four-stream mHC training
package ``cake_pplx._backends.cake_mhc_nondet_train`` (hidden 4096,
5120 and 7168, 4 streams; SM100 / SM103 at the three hidden sizes, SM107 at
4096), in the
form of ``test_cake_mhc_det_train_workspace.py``.  Every case runs at each
hidden size the device's architecture routes, its buffers sized with
``hidden=``; a size it does not route is skipped.

The backward workspace of this package starts with a zero prefix of 768 bytes
at every token count and hidden size: the ticket words the column-slab pass
draws its work units from and the token owner's group-row form its groups from
(self-resetting ``atom.inc``; a word that is not zero at launch makes the launch
skip units or end away from zero), then the column slab's arrival counts
(``slab_cnt``: the grid's finished units and the CTAs past the wait for them;
the last CTA of a launch leaves both words zero).  The deterministic regions
follow (the token owner's records and parameter rows: the per-CTA partials, or
the group rows inside ``BWD_ROW_BANDS``), then the column slab's per-chunk
``dfn`` slots, so
the workspace is larger than the deterministic package's.  A caller-owned buffer
sized by ``cake_mhc_nondet_train_backward_workspace_bytes(T_max)`` serves
every token count up to ``T_max`` (at its ``hidden``) and CUDA Graph replays
at a smaller ``T``.
Before every call the bytes of the backward buffer past the zero prefix are
overwritten with 0xFF (no call may depend on what an earlier call left there).
The package makes no bit promise: no call is compared bit for bit with a call
on fresh workspaces or with the deterministic package.  Each test runs its
calls on 32 inputs, new inputs for every call, and judges every output of
every call with ``tests/cake_naive.py`` against the naive BF16 tier: its
error against the float64 oracle
(``tests/cake/test_helpers/cake_mhc_naive.py``), in the L2 and the max norm,
median and max over the 32 inputs, is at most 1.005 times the tier's, and zero
where the tier's is zero (the outputs of each call of a sequence under the name
prefix ``call <n> T=<T>``).  On one buffer reused over realistic token-count
sequences:

* every call finds the zero prefix of its layout zero and leaves it zero;
* a CUDA Graph captured at a smaller token count on the large buffers, its
  replays (new inputs copied in place) interleaved with eager calls at the
  large token count, keeps both properties at every replay;
* the default workspaces (``workspace=None``: one buffer per hidden size and
  power-of-two token-count bucket) across the token counts of one bucket, the
  reported sequences (7832 then 7435 tokens, and the reverse order) among
  them;
* the reported sequence on the default workspace of a fresh process, each
  sequence in its own process (its fixed inputs are not the records of a
  forward, so no oracle applies: the shapes, finite outputs and the zero
  prefix);
* no default buffer is created during a CUDA graph capture (``ValueError``,
  nothing cached), and the eager call after it creates the buffer and returns
  finite outputs of the call's shapes;
* with two devices, the capture check reads the workspace device's current
  stream, not the current device's: a capture on the workspace device raises
  while another device is current, and a capture on another device does not
  stop the call from creating the buffer and running eagerly;
* one buffer over a ragged sequence with caller outputs, and the sizes of the
  workspace and which sizing serves which backward (a buffer of the
  deterministic sizing is refused);
* the helpers' sizes per hidden size (an unrouted or unknown size is a
  ``ValueError``), and a buffer sized for H 4096 refused by an H 7168 call it
  is too small for.
"""

from __future__ import annotations

import gc
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from tests.cake.test_helpers.cake_mhc_naive import (
    expected_and_naive,
    judge_draws,
    package_outputs,
    stats_outputs,
)
from tests.cake.test_helpers.cake_package import PACKAGES, cake_package

KEY = "cake.cake_mhc_nondet_train"
pkg = cake_package(KEY)
loader = cake_package(KEY, "cake_jit")
# The deterministic package: its workspace layouts and sizes (this package's layout extends them).
det = cake_package("cake.cake_mhc_det_train")
det_loader = cake_package("cake.cake_mhc_det_train", "cake_jit")
backward_finish = pkg.cake_mhc_nondet_train_backward_finish

SUPPORTED_CAPABILITIES = ((10, 0), (10, 3), (10, 7))
STREAMS, HIDDEN, MAPPING_ROWS = 4, 4096, 24
HIDDEN_SIZES = (4096, 5120, 7168)
# The backward workspace's zero prefix: the ticket region (256 bytes) and the column slab's arrival counts (512 bytes),
# first for every token count and hidden size; every call finds it zero and leaves it zero.
ZERO_PREFIX_BYTES = 768
SLAB_CNT_WORDS = 128
BF16, F32, F64 = torch.bfloat16, torch.float32, torch.float64
SEED = 7046
# Token-count sequences of one buffer (the largest first: the buffer is sized for it).  The first two are the
# sequences of the deterministic suite's dynamic-schedule column-slab rows on sm_103a / sm_100a (this package's
# column slab draws its units from the ticket word on every architecture); the third is the per-step token count of a
# GLM-style training run.
SEQUENCES = {
    "slab_dynamic_sm103": (4096, 3000, 1024, 4096, 2049, 3000),
    "slab_dynamic_sm100": (8192, 6000, 4500, 8192, 5000),
    "glm_rows": (16263, 14928, 15566, 16263, 15288, 16164),
}
GRAPHS = ((4096, 3000), (8192, 4500), (16263, 14928))
# The reported backward token-count sequences (one 8192-token default bucket).
REPORTED_SEQUENCES = ((7832, 7435), (7435, 7832))
DEFAULT_BUCKETS = (
    (4096, 3000, 2100, 4096),
    (8192, 6000, 4500, 8192),
    *REPORTED_SEQUENCES,
)
# One caller workspace over token counts within and across the 64-token tiles, down to one token.
RAGGED_SEQUENCE = (4096, 1, 777, 2049, 4096)
OUTPUTS = ("U", "stats", "Y", "dB", "gstats", "dX", "dfn", "dscale", "dbase")
_BACKWARD = ("dX", "dfn", "dscale", "dbase")


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
    # Both packages' libraries of the local architecture (this package's, and the deterministic package's, whose
    # layouts this package's extend); a prebuilt or cached library loads without compiling.
    if _supported():
        arch = _arch()
        for spec in [*det_loader.specs_for_arch(arch), *loader.specs_for_arch(arch)]:
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


def _sm_count() -> int:
    return loader.sm_count(_device().index)


def _hidden_sizes() -> tuple[int, ...]:
    """The hidden sizes the device's architecture routes every kind of."""
    arch = _arch()
    return tuple(loader.supported_hidden_sizes(arch)) if arch else ()


@pytest.fixture(params=HIDDEN_SIZES, ids=lambda h: f"H{h}")
def hidden(request) -> int:
    """The hidden size of a case (every size of the package; skipped where the architecture has no route)."""
    if request.param not in _hidden_sizes():
        pytest.skip(
            f"hidden {request.param} has no exported route on {_arch()} "
            f"(exported hidden sizes: {_hidden_sizes()})"
        )
    return request.param


def _parts(T: int, hidden: int = HIDDEN) -> int:
    """``P``: the chunk count of the column-slab pass, the number of partials each ``dfn`` element folds."""
    parts = loader.host_plan(T, _sm_count(), _arch(), hidden)["parts"]
    assert 1 <= parts <= 32, (T, hidden, parts)
    return parts


def _round256(n: int) -> int:
    return 256 * (-(-n // 256))


# ----------------------------------------------------------------------------------------------------------------
# Inputs, steps and checks


def _make_inputs(T: int, hidden: int, seed: int) -> dict:
    g = torch.Generator(device=_device()).manual_seed(seed)

    def randn(*shape: int, dtype: torch.dtype = F32) -> torch.Tensor:
        return torch.randn(*shape, generator=g, device=_device(), dtype=F32).to(dtype)

    return {
        "X": randn(T, STREAMS, hidden, dtype=BF16),
        "fn": (randn(MAPPING_ROWS, STREAMS * hidden) * 1e-2).contiguous(),
        "scale": 1.0 + 0.1 * randn(3),
        "base": 0.1 * randn(MAPPING_ROWS),
        "B": randn(T, hidden, dtype=BF16),
        "dY": randn(T, STREAMS, hidden, dtype=BF16),
        "dU": randn(T, hidden, dtype=BF16),
    }


_INPUTS: dict[tuple[int, int], dict] = {}


def _inputs(T: int, hidden: int = HIDDEN) -> dict:
    """The fixed inputs of ``(T, hidden)`` (one pair's at a time: 16263 tokens hold ~1.3 GB at H 4096, ~2.3 GB at
    7168)."""
    if (T, hidden) not in _INPUTS:
        _INPUTS.clear()
        _INPUTS[(T, hidden)] = _make_inputs(T, hidden, SEED + T)
    return _INPUTS[(T, hidden)]


def _step(i: dict, fwd_ws, bwd_ws) -> dict:
    """One forward + backward of this package."""
    U, stats = pkg.cake_mhc_nondet_train_forward_prepare(
        i["X"], i["fn"], i["scale"], i["base"], workspace=fwd_ws
    )
    Y = pkg.cake_mhc_nondet_train_forward_finish(i["X"], i["B"], stats)
    dB, gstats = pkg.cake_mhc_nondet_train_backward_prepare(
        i["dY"], i["B"], i["X"], stats
    )
    dX, dfn, dscale, dbase = backward_finish(
        i["X"], i["dY"], i["dU"], i["B"], i["fn"], i["scale"], i["base"], stats, gstats,
        workspace=bwd_ws,
    )  # fmt: skip
    return dict(
        zip(OUTPUTS, (U, stats, Y, dB, gstats, dX, dfn, dscale, dbase), strict=True)
    )


def _snapshot(out: dict) -> dict:
    return {k: v.detach().clone() for k, v in out.items()}


def _judged(prefix: str, i: dict, got: dict) -> tuple[list, dict, dict]:
    """One call's judged outputs ``got`` (name -> tensor) under the name prefix ``prefix``, with the oracle's and the
    naive tier's values on the call's inputs ``i``: a triple of :func:`judge_draws`."""
    expected, naive = expected_and_naive(i, got)
    return (
        [{prefix + k: v for k, v in got.items()}],
        {prefix + k: v for k, v in expected.items()},
        {prefix + k: v for k, v in naive.items()},
    )


def _layout(kind: str, T: int, hidden: int) -> dict:
    return loader.workspace_layout(kind, T, device_index=_device().index, hidden=hidden)


def _workspaces(t_max: int, hidden: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Caller-owned forward and backward workspaces of this package's sizing for up to ``t_max`` tokens of hidden
    size ``hidden``."""
    return (
        pkg.cake_mhc_nondet_train_forward_workspace(t_max, _device(), hidden=hidden),
        pkg.cake_mhc_nondet_train_backward_workspace(t_max, _device(), hidden=hidden),
    )


def _nd_workspace_exact(T: int, hidden: int = HIDDEN) -> torch.Tensor:
    """A fresh zeroed backward workspace of this package of exactly ``T`` tokens' layout at ``hidden``."""
    return torch.zeros(
        loader.workspace_bytes("bwd", T, _sm_count(), _arch(), hidden),
        dtype=torch.uint8,
        device=_device(),
    )


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


def _assert_zero_words(fwd_ws, bwd_ws, T: int, hidden: int, what: str) -> None:
    """The ticket words of the forward layout (the deterministic package's layout) and the zero prefix of this
    package's backward layout for ``T`` tokens of hidden size ``hidden`` (the ticket region at offset 0, the arrival
    counts right after it) are zero."""
    torch.cuda.synchronize()
    bwd = loader.workspace_regions("bwd", T, _sm_count(), _arch(), hidden)["regions"]
    assert bwd["ticket"]["offset"] == 0 and bwd["ticket"]["bytes"] == 256
    assert bwd["slab_cnt"]["offset"] == 256
    assert bwd["slab_cnt"]["bytes"] == 4 * SLAB_CNT_WORDS == ZERO_PREFIX_BYTES - 256
    fwd = _layout("fwd", T, hidden)["regions"]["ticket"]
    for ws, kind, lo, hi in (
        (fwd_ws, "fwd ticket words", fwd["offset"], fwd["offset"] + fwd["bytes"]),
        (bwd_ws, "bwd zero prefix", 0, ZERO_PREFIX_BYTES),
    ):
        nonzero = int(torch.count_nonzero(ws[lo:hi]))
        assert nonzero == 0, (
            f"{what}: {nonzero} non-zero bytes in the {kind} of T={T} H={hidden}"
        )


# ----------------------------------------------------------------------------------------------------------------
# Tests


@pytest.mark.parametrize("name", sorted(SEQUENCES))
def test_caller_workspace_reused_across_token_counts(
    name: str, hidden: int, record_property
) -> None:
    Ts = SEQUENCES[name]
    t_max = max(Ts)
    fwd_ws, bwd_ws = _workspaces(t_max, hidden)

    def case(j: int, s: int):
        for n, T in enumerate(Ts):
            i = _make_inputs(T, hidden, s + n)
            what = (
                f"{name} H={hidden} draw {j} call {n} (T={T} on a buffer of T={t_max})"
            )
            _poison_workspace(bwd_ws)
            _assert_zero_words(fwd_ws, bwd_ws, T, hidden, what + " before the call")
            got = package_outputs(_snapshot(_step(i, fwd_ws, bwd_ws)))
            _assert_zero_words(fwd_ws, bwd_ws, T, hidden, what + " after the call")
            yield _judged(f"call {n} T={T} ", i, got)

    judge_draws(case, record_property, seed=SEED)


@pytest.mark.parametrize(("t_max", "t_small"), GRAPHS)
def test_graph_captured_at_a_smaller_token_count(
    t_max: int, t_small: int, hidden: int, record_property
) -> None:
    """A graph of one forward + backward captured at ``t_small`` on the buffers of ``t_max``; on each of 32 inputs an
    eager call at ``t_max`` on the same buffers, then a replay on new inputs copied in place: both keep the zero
    words zero, and the outputs of both are judged (the eager call's under ``eager T=<t_max>``, the replay's under
    ``replay T=<t_small>``)."""
    fwd_ws, bwd_ws = _workspaces(t_max, hidden)
    small = _make_inputs(t_small, hidden, SEED + t_small)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            _step(small, fwd_ws, bwd_ws)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = _step(small, fwd_ws, bwd_ws)
    torch.cuda.synchronize()

    def case(j: int, s: int):
        what = f"H={hidden} replay {j} at T={t_small} after an eager call at T={t_max}"
        eager = f"H={hidden} eager T={t_max} before replay {j}"
        big = _make_inputs(t_max, hidden, s + 1)
        _poison_workspace(bwd_ws)
        got = package_outputs(_snapshot(_step(big, fwd_ws, bwd_ws)))
        _assert_zero_words(fwd_ws, bwd_ws, t_max, hidden, eager)
        yield _judged(f"eager T={t_max} ", big, got)
        del big, got
        new = _make_inputs(t_small, hidden, s)
        for k, v in new.items():
            small[k].copy_(v)
        _poison_workspace(bwd_ws)
        _assert_zero_words(fwd_ws, bwd_ws, t_small, hidden, what + " (before)")
        for t in out.values():
            t.fill_(float("nan"))
        graph.replay()
        _assert_zero_words(fwd_ws, bwd_ws, t_small, hidden, what + " (after)")
        yield _judged(f"replay T={t_small} ", new, package_outputs(_snapshot(out)))

    judge_draws(case, record_property, seed=SEED + t_small)


@pytest.mark.parametrize("Ts", DEFAULT_BUCKETS)
def test_default_workspace_bucket_across_token_counts(
    Ts: tuple, hidden: int, record_property
) -> None:
    """``workspace=None`` over the token counts of one power-of-two bucket (the reported sequences among them): the
    bucket's default buffer of the hidden size (keyed by device, kind, hidden size and bucket; allocated on the first
    call) has its bytes past the zero prefix overwritten with 0xFF before every later call and its zero prefix zero
    before and after every call; every output of every call is judged over 32 inputs."""
    index = _device().index

    def case(j: int, s: int):
        for n, T in enumerate(Ts):
            i = _make_inputs(T, hidden, s + n)
            what = f"default workspace draw {j} call {n} (T={T}, H={hidden}) of {Ts}"
            key = (index, "bwd", hidden, 1 << (T - 1).bit_length())
            default = pkg._DEFAULT_WORKSPACES.get(key)
            if default is not None:
                _poison_workspace(default)
                _assert_zero_prefix(default, what + " before the call")
            got = package_outputs(_snapshot(_step(i, None, None)))
            _assert_zero_prefix(pkg._DEFAULT_WORKSPACES[key], what + " after the call")
            yield _judged(f"call {n} T={T} ", i, got)

    judge_draws(case, record_property, seed=SEED + Ts[0])


# One process per sequence: the inputs of the issue's reproduction (one CPU generator seeded 0, drawn in call order)
# at the hidden size of the case, each ``backward_finish(...)`` on the default workspace (keyed by device, kind, hidden
# size and power-of-two token bucket).  The inputs are not the records of a forward (``stats`` and ``grad_stats`` are
# random) and do not change with a seed, so no oracle applies: each call is held to its contract (the output shapes,
# finite outputs) and to the zero prefix.  The judged default-workspace calls at these token counts are
# test_default_workspace_bucket_across_token_counts's.
_REPORTED_CHILD = r"""
import importlib, json, sys
import torch
pkg = importlib.import_module(sys.argv[1])
loader = importlib.import_module(sys.argv[1] + ".cake_jit")
H = int(sys.argv[3])
dev = torch.device("cuda:0")
arch = loader.arch_for_capability(tuple(torch.cuda.get_device_capability(dev)))
sm = loader.sm_count(0)
gen = torch.Generator(device="cpu").manual_seed(0)
def r(*shape, dtype=torch.float32, scale=1.0):
    return (torch.randn(shape, generator=gen) * scale).to(dtype).to(dev)
for T in (int(t) for t in sys.argv[2].split(",")):
    args = [r(T, 4, H, dtype=torch.bfloat16), r(T, 4, H, dtype=torch.bfloat16, scale=1e-3),
            r(T, H, dtype=torch.bfloat16, scale=1e-3), None, r(24, 4 * H, scale=0.05), r(3, scale=0.1),
            r(24), r(T, 64), r(T, 32, scale=1e-3)]
    layout = loader.workspace_regions("bwd", T, sm, arch, H)
    counts = layout["regions"]["slab_cnt"]
    prefix = counts["offset"] + counts["bytes"]
    key = (0, "bwd", H, 1 << (T - 1).bit_length())
    def nonzero():
        torch.cuda.synchronize()
        ws = pkg._DEFAULT_WORKSPACES.get(key)
        return 0 if ws is None else int(torch.count_nonzero(ws[:prefix]))
    call = {"T": T, "prefix": prefix, "prefix_before": nonzero()}
    print(json.dumps(call), flush=True)
    got = [t.clone() for t in pkg.cake_mhc_nondet_train_backward_finish(*args)]
    call["prefix_after"] = nonzero()
    call["default_found"] = key in pkg._DEFAULT_WORKSPACES
    call["shapes"] = [list(t.shape) for t in got]
    call["finite"] = all(bool(torch.isfinite(t).all()) for t in got)
    print(json.dumps(call), flush=True)
"""


def _shapes(T: int, hidden: int) -> list[list[int]]:
    """The shapes of ``backward_finish``'s outputs ``dX``, ``dfn``, ``dscale``, ``dbase``."""
    return [[T, STREAMS, hidden], [MAPPING_ROWS, STREAMS * hidden], [3], [MAPPING_ROWS]]


@pytest.mark.parametrize(
    "Ts", REPORTED_SEQUENCES, ids=lambda Ts: "-".join(map(str, Ts))
)
def test_default_workspace_reported_sequence(Ts: tuple, hidden: int) -> None:
    run = subprocess.run(
        [
            sys.executable,
            "-c",
            _REPORTED_CHILD,
            PACKAGES[KEY],
            ",".join(map(str, Ts)),
            str(hidden),
        ],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=900,
    )
    calls = [
        json.loads(line) for line in run.stdout.splitlines() if line.startswith("{")
    ]
    tail = "\n".join((run.stdout + run.stderr).strip().splitlines()[-6:])
    assert run.returncode == 0, (
        f"backward_finish() over {Ts} on the default workspace: exit {run.returncode}\n{tail}"
    )
    done = [c for c in calls if "finite" in c]
    assert [c["T"] for c in done] == list(Ts), tail
    for c in done:
        what = f"default workspace, backward_finish() at T={c['T']} H={hidden} of {Ts}"
        # The zero-prefix checks read the call's default buffer, not a missing one (a vacuous zero).
        assert c["default_found"], (
            f"{what}: no default workspace under the probed key (device, kind, hidden, bucket)"
        )
        assert c["prefix"] == ZERO_PREFIX_BYTES, (what, c["prefix"])
        assert c["prefix_before"] == 0, (
            f"{what}: {c['prefix_before']} non-zero bytes in the zero prefix before the call"
        )
        assert c["prefix_after"] == 0, (
            f"{what}: {c['prefix_after']} non-zero bytes in the zero prefix after the call"
        )
        assert c["shapes"] == _shapes(c["T"], hidden), (what, c["shapes"])
        assert c["finite"], f"{what}: non-finite outputs"


# A fresh process at the hidden size of the case: ``backward_finish(...)`` on a caller-owned workspace (every module
# loaded), then the same call with ``workspace=None`` inside a CUDA graph capture, before any default buffer of its
# bucket exists, then the same call eagerly (fixed inputs that are not the records of a forward: held to the contract).
_CAPTURE_CHILD = r"""
import importlib, json, sys
import torch
pkg = importlib.import_module(sys.argv[1])
dev = torch.device("cuda:0")
T, H = int(sys.argv[2]), int(sys.argv[3])
gen = torch.Generator(device="cpu").manual_seed(0)
def r(*shape, dtype=torch.float32, scale=1.0):
    return (torch.randn(shape, generator=gen) * scale).to(dtype).to(dev)
args = [r(T, 4, H, dtype=torch.bfloat16), r(T, 4, H, dtype=torch.bfloat16, scale=1e-3),
        r(T, H, dtype=torch.bfloat16, scale=1e-3), None, r(24, 4 * H, scale=0.05), r(3, scale=0.1),
        r(24), r(T, 64), r(T, 32, scale=1e-3)]
key = (0, "bwd", H, 1 << (T - 1).bit_length())
ws = pkg.cake_mhc_nondet_train_backward_workspace(T, dev, hidden=H)
pkg.cake_mhc_nondet_train_backward_finish(*args, workspace=ws)
torch.cuda.synchronize()
out = {"cached_before": key in pkg._DEFAULT_WORKSPACES, "raised": None}
graph = torch.cuda.CUDAGraph()
try:
    with torch.cuda.graph(graph):
        pkg.cake_mhc_nondet_train_backward_finish(*args)
except ValueError as e:
    out["raised"] = str(e)
del graph
torch.cuda.synchronize()
out["cached_after_capture"] = key in pkg._DEFAULT_WORKSPACES
got = [t.clone() for t in pkg.cake_mhc_nondet_train_backward_finish(*args)]
torch.cuda.synchronize()
out["cached_after_eager"] = key in pkg._DEFAULT_WORKSPACES
out["shapes"] = [list(t.shape) for t in got]
out["finite"] = all(bool(torch.isfinite(t).all()) for t in got)
print(json.dumps(out), flush=True)
"""


def test_default_workspace_is_not_created_during_capture(hidden: int) -> None:
    T = 300
    run = subprocess.run(
        [sys.executable, "-c", _CAPTURE_CHILD, PACKAGES[KEY], str(T), str(hidden)],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=900,
    )
    tail = "\n".join((run.stdout + run.stderr).strip().splitlines()[-6:])
    assert run.returncode == 0, f"capture child: exit {run.returncode}\n{tail}"
    lines = [
        json.loads(line) for line in run.stdout.splitlines() if line.startswith("{")
    ]
    assert len(lines) == 1, tail
    out = lines[0]
    what = f"backward_finish() at T={T} H={hidden} on the default workspace"
    assert not out["cached_before"], (
        f"{what}: a default buffer existed before the capture"
    )
    assert out["raised"] is not None and "CUDA graph capture" in out["raised"], (
        f"{what}: no ValueError inside the capture ({out['raised']!r})"
    )
    assert not out["cached_after_capture"], (
        f"{what}: the capture cached a default buffer"
    )
    assert out["cached_after_eager"], f"{what}: the eager call cached no default buffer"
    assert out["shapes"] == _shapes(T, hidden), (what, out["shapes"])
    assert out["finite"], f"{what}: non-finite outputs"


# A fresh process with two devices at the hidden size of the case: ``backward_finish(...)`` on a
# caller-owned workspace of each device (every module loaded), then the call with ``workspace=None`` on cuda:1
# tensors, before any default buffer of its bucket exists, (a) inside a capture on cuda:1 while cuda:0 is the current
# device and (b) inside a relaxed-mode capture on cuda:0 (the call's stream, cuda:1's current stream, is not capturing).
_CAPTURE_DEVICE_CHILD = r"""
import importlib, json, sys
import torch
pkg = importlib.import_module(sys.argv[1])
T, H = int(sys.argv[2]), int(sys.argv[3])
dev0, dev1 = torch.device("cuda:0"), torch.device("cuda:1")
gen = torch.Generator(device="cpu").manual_seed(0)
def r(*shape, dtype=torch.float32, scale=1.0):
    return (torch.randn(shape, generator=gen) * scale).to(dtype)
host = [r(T, 4, H, dtype=torch.bfloat16), r(T, 4, H, dtype=torch.bfloat16, scale=1e-3),
        r(T, H, dtype=torch.bfloat16, scale=1e-3), None, r(24, 4 * H, scale=0.05), r(3, scale=0.1),
        r(24), r(T, 64), r(T, 32, scale=1e-3)]
def on(dev):
    return [None if t is None else t.to(dev) for t in host]
bucket = 1 << (T - 1).bit_length()
for dev in (dev0, dev1):
    ws = pkg.cake_mhc_nondet_train_backward_workspace(T, dev, hidden=H)
    pkg.cake_mhc_nondet_train_backward_finish(*on(dev), workspace=ws)
args1 = on(dev1)
torch.cuda.synchronize(dev0)
torch.cuda.synchronize(dev1)
out = {"cached_before": [(i, "bwd", H, bucket) in pkg._DEFAULT_WORKSPACES for i in (0, 1)], "raised_a": None, "raised_b": None}
# Each capture gets its own stream: torch.cuda.graph's default capture stream is created once, on the device that
# was current at the first capture, and reused by later captures on any device.
capture1 = torch.cuda.Stream(device=dev1)
with torch.cuda.device(dev1):
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph, stream=capture1):
            with torch.cuda.device(dev0):
                out["stream_a"] = [torch.cuda.current_device(), torch.cuda.current_stream(dev1).cuda_stream == capture1.cuda_stream]
                pkg.cake_mhc_nondet_train_backward_finish(*args1)
    except ValueError as e:
        out["raised_a"] = str(e)
    del graph
torch.cuda.synchronize(dev1)
out["cached_after_a"] = (1, "bwd", H, bucket) in pkg._DEFAULT_WORKSPACES
y = torch.zeros(16, device=dev0)
got = None
capture0 = torch.cuda.Stream(device=dev0)
with torch.cuda.device(dev0):
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph, stream=capture0, capture_error_mode="relaxed"):
            y.add_(1)
            with torch.cuda.device(dev1):
                capturing1 = torch.cuda.is_current_stream_capturing()
            out["stream_b"] = [torch.cuda.current_device(), torch.cuda.current_stream(dev0).cuda_stream == capture0.cuda_stream, capturing1]
            got = [t.clone() for t in pkg.cake_mhc_nondet_train_backward_finish(*args1)]
    except ValueError as e:
        out["raised_b"] = str(e)
    del graph
torch.cuda.synchronize(dev0)
torch.cuda.synchronize(dev1)
out["cached_after_b"] = [(i, "bwd", H, bucket) in pkg._DEFAULT_WORKSPACES for i in (0, 1)]
out["ran_b"] = got is not None
if got is not None:
    out["shapes"] = [list(t.shape) for t in got]
    out["finite"] = all(bool(torch.isfinite(t).all()) for t in got)
print(json.dumps(out), flush=True)
"""


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two CUDA devices")
def test_default_workspace_capture_check_uses_workspace_device(hidden: int) -> None:
    T = 300
    run = subprocess.run(
        [
            sys.executable,
            "-c",
            _CAPTURE_DEVICE_CHILD,
            PACKAGES[KEY],
            str(T),
            str(hidden),
        ],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=900,
    )
    tail = "\n".join((run.stdout + run.stderr).strip().splitlines()[-6:])
    assert run.returncode == 0, (
        f"two-device capture child: exit {run.returncode}\n{tail}"
    )
    lines = [
        json.loads(line) for line in run.stdout.splitlines() if line.startswith("{")
    ]
    assert len(lines) == 1, tail
    out = lines[0]
    what = f"backward_finish() at T={T} H={hidden} on cuda:1's default workspace"
    assert out["cached_before"] == [False, False], (
        f"{what}: a default buffer existed before the captures"
    )
    assert out.get("stream_a") == [0, True], (
        f"{what}: capture (a) did not run on cuda:1's stream with cuda:0 current ({out.get('stream_a')})"
    )
    assert out.get("stream_b") == [0, True, False], (
        f"{what}: capture (b) did not run on cuda:0's stream with cuda:1 idle ({out.get('stream_b')})"
    )
    assert out["raised_a"] is not None and "CUDA graph capture" in out["raised_a"], (
        f"{what}: no ValueError inside a capture on cuda:1 with cuda:0 current ({out['raised_a']!r})"
    )
    assert not out["cached_after_a"], (
        f"{what}: the capture on cuda:1 cached a default buffer"
    )
    assert out["raised_b"] is None, (
        f"{what}: ValueError inside a capture on cuda:0 ({out['raised_b']!r})"
    )
    assert out["ran_b"] and out["cached_after_b"] == [False, True], (
        f"{what}: the call during a capture on cuda:0 cached {out['cached_after_b']}"
    )
    assert out["shapes"] == _shapes(T, hidden), (what, out["shapes"])
    assert out["finite"], f"{what}: non-finite outputs"


# ----------------------------------------------------------------------------------------------------------------
# One buffer over a ragged sequence; the sizes


def _forward(i: dict) -> dict:
    """The forward prepare and the backward prepare of a case: the records ``backward_finish`` takes."""
    _U, stats = pkg.cake_mhc_nondet_train_forward_prepare(
        i["X"], i["fn"], i["scale"], i["base"]
    )
    _dB, gstats = pkg.cake_mhc_nondet_train_backward_prepare(
        i["dY"], i["B"], i["X"], stats
    )
    return {"stats": stats, "gstats": gstats}


def _nd_backward(i: dict, prep: dict, **kwargs) -> dict:
    out = backward_finish(
        i["X"], i["dY"], i["dU"], i["B"], i["fn"], i["scale"], i["base"], prep["stats"], prep["gstats"], **kwargs,
    )  # fmt: skip
    return dict(zip(_BACKWARD, out, strict=True))


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


def test_one_workspace_serves_a_ragged_token_sequence(
    hidden: int, record_property
) -> None:
    """One caller workspace of this package's sizing for the largest token count, its bytes past the zero prefix
    0xFF before every call, over ``RAGGED_SEQUENCE`` with caller outputs holding NaN: every call finds the zero
    prefix zero and leaves it zero, and every output of every call is judged over 32 inputs."""
    t_max = max(RAGGED_SEQUENCE)
    _fwd_ws, ws = _workspaces(t_max, hidden)
    assert ws.numel() == loader.workspace_bytes_upto(
        "bwd", t_max, _sm_count(), _arch(), hidden
    )

    def case(j: int, s: int):
        for n, T in enumerate(RAGGED_SEQUENCE):
            i = _make_inputs(T, hidden, s + n)
            prep = _forward(i)
            what = f"draw {j} call {n} (T={T} H={hidden} on a buffer of T={t_max})"
            _poison_workspace(ws)
            _assert_zero_prefix(ws, what + " before the call")
            got = _snapshot(
                _nd_backward(i, prep, workspace=ws, **_poisoned_outputs(T, hidden))
            )
            _assert_zero_prefix(ws, what + " after the call")
            yield _judged(f"call {n} T={T} ", i, got)

    judge_draws(case, record_property, seed=SEED + 1)


def test_workspace_sizes_and_which_sizing_serves_which_backward(
    hidden: int, record_property
) -> None:
    """At the hidden size ``H`` of the case, this package's backward workspace is the ticket region, the column slab's
    arrival counts (``SLAB_CNT_WORDS`` uint32 at offset 256: the zero prefix), the deterministic regions but the
    per-chunk ``dfn`` partials (their sizes and shapes, the parameter region the deterministic one: the group rows
    inside ``BWD_ROW_BANDS``) and the per-chunk ``dfn`` slots (``min(32, ceil(T / 64)) x 24 x 4H`` FP32); its
    public size is the envelope over every token count up to ``T``, above the deterministic package's; the forward
    workspace is the deterministic one.  The size helper takes no ``deterministic`` keyword (``TypeError``).  A
    workspace of the deterministic sizing (``cake_mhc_det_train_backward_workspace``) is smaller than this package's
    exact layout and refused by its backward, as is one of this package's sizing for a smaller token count; one of this
    package's sizing serves the backward (its outputs judged over 32 inputs, its zero prefix zero after each call)."""
    dev, sm, arch = _device(), _sm_count(), _arch()
    shift = ZERO_PREFIX_BYTES - 256
    for T in (1, 7, 63, 64, 65, 129, 257, 1000, 2049, 4096, 8192, 16263, 32768, 65536):
        ndl = loader.workspace_regions("bwd", T, sm, arch, hidden)
        dtl = det_loader.workspace_regions("bwd", T, sm, arch, hidden)
        names = list(dtl["regions"])
        assert names[-1] == "dfn_part", T
        assert list(ndl["regions"]) == [
            names[0],
            "slab_cnt",
            *names[1:-1],
            "dfn_slot",
        ], T
        assert ndl["regions"]["ticket"] == dtl["regions"]["ticket"], T
        assert ndl["regions"]["ticket"]["offset"] == 0
        assert ndl["regions"]["ticket"]["bytes"] == 4 * loader.TICKET_WORDS == 256
        assert ndl["regions"]["slab_cnt"] == {
            "offset": 256,
            "bytes": 4 * SLAB_CNT_WORDS,
            "dtype": "uint32",
            "shape": (SLAB_CNT_WORDS,),
        }, T
        for name in names[1:-1]:
            region = dtl["regions"][name]
            assert ndl["regions"][name] == {
                **region,
                "offset": region["offset"] + shift,
            }, (T, name)
        part, slots = dtl["regions"]["dfn_part"], ndl["regions"]["dfn_slot"]
        slot_parts = min(32, -(-T // 64))
        assert part["bytes"] == _parts(T, hidden) * MAPPING_ROWS * STREAMS * hidden * 4
        assert slots["shape"] == (slot_parts, MAPPING_ROWS, STREAMS * hidden), T
        assert slots["dtype"] == "float32" and slot_parts >= _parts(T, hidden), T
        assert ndl["total_bytes"] == (
            dtl["total_bytes"]
            - _round256(part["bytes"])
            + shift
            + _round256(slots["bytes"])
        ), T
        nd_bytes = pkg.cake_mhc_nondet_train_backward_workspace_bytes(
            T, dev, hidden=hidden
        )
        det_bytes = det.cake_mhc_det_train_backward_workspace_bytes(
            T, dev, hidden=hidden
        )
        assert nd_bytes == loader.workspace_bytes_upto("bwd", T, sm, arch, hidden), T
        assert ndl["total_bytes"] <= nd_bytes and nd_bytes % 256 == 0, T
        assert det_bytes < ndl["total_bytes"], (T, det_bytes, ndl["total_bytes"])
        assert pkg.cake_mhc_nondet_train_forward_workspace_bytes(
            T, dev, hidden=hidden
        ) == det.cake_mhc_det_train_forward_workspace_bytes(T, dev, hidden=hidden), T
    assert (
        pkg.cake_mhc_nondet_train_backward_workspace_bytes(0, dev, hidden=hidden) == 0
    )
    with pytest.raises(ValueError):
        pkg.cake_mhc_nondet_train_backward_workspace_bytes(-1, dev, hidden=hidden)
    for value in (False, True):
        with pytest.raises(TypeError, match="deterministic"):
            pkg.cake_mhc_nondet_train_backward_workspace_bytes(
                2049, dev, hidden=hidden, deterministic=value
            )
    allocated = pkg.cake_mhc_nondet_train_backward_workspace(2049, dev, hidden=hidden)
    assert allocated.dtype == torch.uint8 and allocated.device == dev
    assert allocated.numel() == pkg.cake_mhc_nondet_train_backward_workspace_bytes(
        2049, dev, hidden=hidden
    )
    assert int(torch.count_nonzero(allocated)) == 0

    T = 2049
    det_sized = det.cake_mhc_det_train_backward_workspace(T, dev, hidden=hidden)
    nd_sized = pkg.cake_mhc_nondet_train_backward_workspace(T, dev, hidden=hidden)

    def case(j: int, s: int):
        i = _make_inputs(T, hidden, s)
        got = _snapshot(_nd_backward(i, _forward(i), workspace=nd_sized))
        _assert_zero_prefix(nd_sized, f"workspace nd-sized H={hidden} draw {j}")
        return _judged("", i, got)

    judge_draws(case, record_property, seed=SEED + T)
    i = _inputs(T, hidden)
    prep = _forward(i)
    with pytest.raises(ValueError, match="workspace holds"):
        _nd_backward(i, prep, workspace=det_sized)
    small = pkg.cake_mhc_nondet_train_backward_workspace(64, dev, hidden=hidden)
    with pytest.raises(ValueError, match="workspace holds"):
        _nd_backward(i, prep, workspace=small)


def test_workspace_bytes_per_hidden_size() -> None:
    """The workspace helpers size per hidden size (``test_cake_mhc_det_train.py``'s test of the same name): at each
    routed size the request is a multiple of 256 bytes, non-decreasing in the token count and at least the exact
    layout of the call; the forward request is the deterministic package's and the backward request above it at the
    same size (the backward requests are printed per size); an unrouted or unknown hidden size is a ``ValueError``;
    the default is 4096."""
    dev = _device()
    sizes = _hidden_sizes()
    unrouted = tuple(H for H in HIDDEN_SIZES if H not in sizes)
    for kind, layout in (("forward", "fwd"), ("backward", "bwd")):
        nbytes = getattr(pkg, f"cake_mhc_nondet_train_{kind}_workspace_bytes")
        allocate = getattr(pkg, f"cake_mhc_nondet_train_{kind}_workspace")
        reference_nbytes = getattr(det, f"cake_mhc_det_train_{kind}_workspace_bytes")
        for H in sizes:
            previous = 0
            for T in (1, 7, 128, 129, 1000, 2049, 8192, 16263, 65536):
                current = nbytes(T, dev, hidden=H)
                exact = _layout(layout, T, H)["total_bytes"]
                assert current % 256 == 0 and current >= max(previous, exact), (
                    kind,
                    H,
                    T,
                    current,
                    previous,
                    exact,
                )
                want = reference_nbytes(T, dev, hidden=H)
                if kind == "forward":
                    assert current == want, (kind, H, T, current, want)
                else:
                    assert current > want, (kind, H, T, current, want)
                previous = current
        assert nbytes(129, dev) == nbytes(129, dev, hidden=4096), kind
        for bad in (6144, 0, -4096, 4097, 2 * 7168, True) + unrouted:
            with pytest.raises(ValueError, match="hidden"):
                nbytes(129, dev, hidden=bad)
        for bad in (6144,) + unrouted:
            with pytest.raises(ValueError, match="hidden"):
                allocate(129, dev, hidden=bad)
    for T in (1, 129, 2049, 8192):
        backward = {
            H: pkg.cake_mhc_nondet_train_backward_workspace_bytes(T, dev, hidden=H)
            for H in sizes
        }
        print(f"backward workspace bytes T={T}: {backward}")


@pytest.mark.parametrize("T", (129, 2049))
def test_workspace_sized_for_another_hidden_size(T: int, record_property) -> None:
    """A caller workspace serves a call of this package when it holds the call's ``(T, H)`` layout and is refused
    (``ValueError``) when it is too small (``test_cake_mhc_det_train.py``'s test of the same name): buffers sized
    for ``(T, 4096)`` used by an H 7168 call at the same ``T`` are refused where smaller than its layout and
    otherwise serve it; buffers one 256-byte block short of the H 7168 layout are refused and exact-size buffers
    serve it.  Over 32 inputs, the default-workspace call (under the name prefix ``default``) and every call a buffer
    serves (the forward statistics on a forward buffer, the backward outputs on a backward buffer, under the
    buffer's name) are judged, and the zero prefix is zero after every backward."""
    if not {4096, 7168} <= set(_hidden_sizes()):
        pytest.skip(f"hidden 4096 and 7168 are not both routed on {_arch()}")
    dev = _device()
    i = _inputs(T, 7168)
    ref = _snapshot(_step(i, None, None))
    need = {kind: _layout(kind, T, 7168)["total_bytes"] for kind in ("fwd", "bwd")}
    sized_4096 = dict(zip(("fwd", "bwd"), _workspaces(T, 4096), strict=True))
    if T == 129:
        # The N-sized split of fn alone grows by more than the token-dependent regions of 129 tokens hold.
        held = sized_4096["fwd"].numel()
        assert held < need["fwd"], (held, need)
    print(
        f"T={T}: H 4096 buffers {sized_4096['fwd'].numel()} / {sized_4096['bwd'].numel()} bytes, H 7168 layouts "
        f"{need['fwd']} / {need['bwd']} bytes"
    )
    short = {
        kind: torch.zeros(need[kind] - 256, dtype=torch.uint8, device=dev)
        for kind in need
    }
    exact = {
        kind: torch.zeros(need[kind], dtype=torch.uint8, device=dev) for kind in need
    }
    args = (i["X"], i["fn"], i["scale"], i["base"])
    prep = {"stats": ref["stats"], "gstats": ref["gstats"]}
    serving = []
    for label, buffers in (("H 4096", sized_4096), ("short", short), ("exact", exact)):
        fwd_ws, bwd_ws = buffers["fwd"], buffers["bwd"]
        if fwd_ws.numel() < need["fwd"]:
            for entry in (
                pkg.cake_mhc_nondet_train_forward_prepare,
                pkg.cake_mhc_nondet_train_forward_stats,
            ):
                with pytest.raises(ValueError, match="workspace holds"):
                    entry(*args, workspace=fwd_ws)
        else:
            serving.append((f"{label} fwd ", "fwd", fwd_ws))
        if bwd_ws.numel() < need["bwd"]:
            with pytest.raises(ValueError, match="workspace holds"):
                _nd_backward(i, prep, workspace=bwd_ws)
        else:
            serving.append((f"{label} bwd ", "bwd", bwd_ws))
    del ref, prep

    def case(j: int, s: int):
        c = _make_inputs(T, 7168, s)
        out = _snapshot(_step(c, None, None))
        yield _judged("default ", c, package_outputs(out))
        cprep = {"stats": out["stats"], "gstats": out["gstats"]}
        for prefix, kind, ws in serving:
            what = f"T={T} H=7168 on the {prefix}buffer, draw {j}"
            if kind == "fwd":
                _U, stats = pkg.cake_mhc_nondet_train_forward_prepare(
                    c["X"], c["fn"], c["scale"], c["base"], workspace=ws
                )
                got = stats_outputs(stats.clone())
            else:
                got = _snapshot(_nd_backward(c, cprep, workspace=ws))
                _assert_zero_prefix(ws, f"{what}: backward")
            yield _judged(prefix, c, got)

    judge_draws(case, record_property, seed=SEED + 7168 + T)
