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

The two-launch KCP forward of the non-deterministic package (``kcp`` module docstring, "Two-launch forward"; the
deterministic package's forward, whose tests these mirror): a main program that reads no received
history row -- the zero-history registry's single-rank ``fwd`` route program on the
rank's fragments (knob ``fwd_main`` 1) or the ``kcp_fwd`` program with every ``hist_row_off`` entry ``-1`` -- and the
``kcp_fwd_halo`` history-row program, which writes the first rows of every fragment with a history row again from the
received pool; with an exchange in flight the overlapped form (knob ``fwd_overlap`` 1) launches the main program
before the exchange's wait.  One process; the exchange is the overlap suite's recorder (a copy of the rank's receive
pool and a work object whose wait the test records; the multi entry also sees a context of two ranks, so that it
exchanges).  Every ``y`` below is judged by the naive judge (``tests/cake/test_helpers/cake_naive.py``): over 32
random draws against the float64 oracle of the unpartitioned problem and PyTorch's BF16 naive convolution.  No bit
rule holds the package: its outputs are never compared bit for bit with another launch or form, and never with the
deterministic package.  Per row of the single-GPU suites:

- every form (``fwd_main`` x ``fwd_overlap``) on both host paths: the order of the launches and the wait
  (``[main, wait(, kcp_fwd_halo)]`` overlapped, ``[wait, main(, kcp_fwd_halo)]`` waiting; the history-row program
  when the rank receives history rows and the form has two launches), ``LAST_CALL["fwd"]``, and ``y`` of each of
  ``REPEATS`` calls into a NaN-filled ``out=``;
- the main program alone (the history-row program skipped): its ``y`` judged as the zero-history forward of the
  rank's fragments as documents of their own;
- the multi entry with the device's architecture put into or taken out of ``MULTI_FWD_MAIN_ARCHS`` /
  ``MULTI_FWD_OVERLAP_ARCHS`` (knobs ``None``): the first tensor's main program before the wait, its history-row
  program after it, the other tensors' programs in the waiting form; one counter buffer per call;
- the single entry with the architecture put into or taken out of ``FWD_MAIN_ARCHS`` / ``FWD_OVERLAP_ARCHS`` (the
  multi entry's sets holding the opposite choice): the form follows the single entry's sets, and the graph-key
  variant is the form;
- the single-rank main program's token-chunk counters are zero after every call, eagerly and under graph replay
  (eager, capture, replay; ``LAST_CALL`` repeats the captured form), single and multi entry;
- ``graph="replay"`` of every two-launch form, single and multi entry: ``LAST_CALL`` holds the form on every call;
- at one rank (the package's own exchange: no work object, no recorder) ``fwd_overlap`` 1 takes the waiting form,
  single entry on both host paths and multi entry;
- a checkpoint with a recompute handle: the original forward takes the configured form, the recompute the waiting
  form on the stored pool (no exchange, no wait); ``y``, ``dx`` and ``dweight`` of the checkpointed and of the
  shipped direct pass judged;
- a steady round of every two-launch form (single and multi entry) issues no host-to-device copy, no synchronizing
  operation and no device allocation;
- the single-rank main program is the record ``cake_jit.selected_modules`` picks for the call, on both host paths:
  rows of ``C`` = 2048 and of ``C`` = 256 at 65536 tokens and below, which on sm_107a are the dynamic programs of the
  route, rows of ``C`` = 640 at local token counts inside every architecture's static range and direct range of its
  forward key (the static and the direct program, which take no counters), and a row of ``C`` = 640 above every
  architecture's token floor of the narrow forward's whole tiles and above the key;
- the multi entry with tensors of a static row and of dynamic rows in one call (one counter buffer, sized for the
  dynamic ones).

Skipped until the exporter renders the non-deterministic package.
"""

from __future__ import annotations

import pytest
import torch

import cake_pplx._backends.cake_causal_conv1d_nondet_train as pkg
from cake_pplx._backends.cake_causal_conv1d_nondet_train import cake_jit
from cake_pplx._backends.cake_causal_conv1d_nondet_train import (
    kcp as conv_kcp,
)
from tests.cake.test_helpers.cake_causal_conv1d_oracle import assert_judged, problem
from tests.experimental.test_cake_causal_conv1d_nondet_train_kcp import _ckpt_run
from tests.experimental.test_cake_causal_conv1d_nondet_train_kcp_multi import (
    judge_multi,
)
from tests.experimental.test_cake_causal_conv1d_nondet_train_kcp_overlap import (
    _recorder,
    _TwoRanks,
)
from tests.test_cake_causal_conv1d_det_train_kcp import (
    ROWS,
    PRODUCTION_ROWS,
    _problem,
    judge_kcp,
)
from tests.test_cake_causal_conv1d_det_train_kcp_multi import MROWS

REPEATS = 3
# (fwd_main, fwd_overlap): the one-launch forward, the single-rank main program waiting, the kcp_fwd program
# overlapped, the single-rank main program overlapped
FORMS = [(0, 0), (1, 0), (0, 1), (1, 1)]
# The rows of the zero-history fwd route's programs: on sm_107a C = 2048 (the c_band program), C = 256 from 65536
# tokens (the two-stage program) and below (the one-stage program); on sm_100a and sm_103a the one-stage program serves
# all three.  C = 640 at 3003 tokens lies inside the static range of every architecture's ``fwd`` key covering C 640
# (the static program), at 12003 tokens inside its direct range (the direct program); from 40003 tokens, above every
# key and every architecture's floor of the whole 40-vector tiles of the narrow forward plan, the dynamic program, so
# the main program's plan names the architecture as the single-rank forward's does.  C = 768 at 14003 tokens lies
# above the ``fwd`` route's floor of the narrow plan on sm_103a (12288: 32-vector tiles of the dynamic program there)
# and inside the direct ranges of sm_100a and sm_107a.
ROUTE_ROWS = [
    ("c2048", (5000, 1003), (1, 2, 4997, 1003), 2048),
    ("c256-s66536", (40000, 25536, 1000), (1, 2, 39997, 26536), 256),
    ("c256-s6003", (5000, 1003), (1, 2, 4997, 1003), 256),
    ("c640-s3003", (2000, 1003), (1, 2, 1997, 1003), 640),
    ("c640-s12003", (8000, 4003), (1, 2, 7997, 4003), 640),
    ("c640-s40003", (30000, 10003), (1, 2, 29997, 10003), 640),
    ("c768-s14003", (10000, 4003), (1, 2, 9997, 4003), 768),
]
# A multi-entry row whose first tensor's local token count lies inside every architecture's static range at C = 640
# (no counters) and whose other tensors run dynamic programs (C = 512, 1024): one counter buffer per call.
MROWS_KEYS = [("c7-static-mixed", (2000, 1003), (1, 2, 1997, 1003), (640, 512, 1024))]

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not pkg.is_cake_causal_conv1d_nondet_train_supported(torch.device("cuda")),
    reason=(
        "the non-deterministic cake_causal_conv1d_nondet_train package needs a supported CUDA device (SM100 / SM103 / "
        "SM107) and its rendered build.json / csrc"
    ),
)


def _reset() -> None:
    conv_kcp._set_test_knobs(None)
    conv_kcp.cake_causal_conv1d_nondet_train_kcp_host_cache_clear()
    conv_kcp.cake_causal_conv1d_nondet_train_kcp_graph_cache_clear()


def _want(ctx, *, sr: bool, overlap: bool, n: int = 1) -> list[str]:
    """The roles the forward launches and its waits, for ``n`` tensors (the multi entry overlaps its first only)."""
    main = "fwd" if sr else "kcp_fwd"
    repair = ["kcp_fwd_halo"] if int(ctx.halo.n_hist_rows) else []
    waiting = [main] + (repair if sr else [])
    if overlap:
        return [main, "wait"] + repair + waiting * (n - 1)
    return ["wait"] + waiting * n


def _record(ctx, *, sr: bool, overlap: bool) -> dict:
    """``LAST_CALL["fwd"]`` / ``LAST_CALL["fwd_multi"]`` of an eager call with tokens and an exchange in flight."""
    return {
        "main": "sr" if sr else "f",
        "form": "pair" if overlap else "wait",
        "repair": bool(int(ctx.halo.n_hist_rows)) and (sr or overlap),
    }


def _move(monkeypatch, name: str, arch: str, inside: bool) -> None:
    """Put ``arch`` into the architecture set ``name`` of the module, or take it out."""
    archs = getattr(conv_kcp, name)
    monkeypatch.setattr(
        conv_kcp, name, frozenset(archs | {arch} if inside else archs - {arch})
    )



@pytest.mark.parametrize(
    "name,lens,cuts,C", ROWS + PRODUCTION_ROWS, ids=[r[0] for r in ROWS + PRODUCTION_ROWS]
)
@pytest.mark.parametrize("host_trim", [1, 0])
def test_forward_forms_order_and_outputs(
    monkeypatch, name, lens, cuts, C, host_trim
) -> None:
    try:
        order = _recorder(monkeypatch)

        def run(p):
            found = []
            for fwd_main, fwd_overlap in FORMS:
                conv_kcp._set_test_knobs(
                    {
                        "host_trim": host_trim,
                        "fwd_main": fwd_main,
                        "fwd_overlap": fwd_overlap,
                    }
                )
                conv_kcp.cake_causal_conv1d_nondet_train_kcp_host_cache_clear()
                sr, overlap = bool(fwd_main), bool(fwd_overlap)
                want = _want(p.ctx, sr=sr, overlap=overlap)
                for repeat in range(REPEATS):
                    order.clear()
                    out = torch.full_like(p.x_local, float("nan"))
                    y, _ = conv_kcp.cake_causal_conv1d_nondet_train_kcp_forward(
                        p.x_local, p.w, p.ctx, out=out
                    )
                    torch.cuda.synchronize()
                    label = f"{name} trim={host_trim} main={fwd_main} overlap={fwd_overlap} #{repeat}"
                    assert y is out, label
                    assert order == want, (label, order)
                    assert conv_kcp.LAST_CALL["fwd"] == _record(
                        p.ctx, sr=sr, overlap=overlap
                    ), label
                    found.append(y)
            return {"y": found}

        judge_kcp(lens, cuts, C, run, label=f"{name} trim={host_trim}")
    finally:
        monkeypatch.undo()
        _reset()


@pytest.mark.parametrize("name,lens,cuts,C", ROWS, ids=[r[0] for r in ROWS])
@pytest.mark.parametrize("fwd_main", [0, 1])
def test_main_program_alone_is_the_fragments_zero_history_forward(
    monkeypatch, name, lens, cuts, C, fwd_main
) -> None:
    """With the history-row program skipped, the overlapped forward's main program (the ``kcp_fwd`` program with
    every ``hist_row_off`` entry ``-1``, or the single-rank program) writes the zero-history forward of the rank's
    fragments as documents of their own -- it reads no received row: its ``y`` judged against the naive tier of that
    problem (the rank's fragments, ``cu_seqlens_local``).  The history-row program writes the first ``min(3, len)``
    rows of the fragments with a history row; there are such rows exactly when the rank receives history rows."""
    device = torch.device("cuda")
    p = _problem(lens, cuts, C, device)
    cu = [int(v) for v in p.ctx.cu_seqlens_local_cpu]
    table = p.ctx.halo_tables["hist_row_off"].cpu()
    repaired = torch.zeros(p.x_local.shape[0], dtype=torch.bool)
    for f in range(len(cu) - 1):
        if bool((table[f] >= 0).any()):
            repaired[cu[f] : min(cu[f] + 3, cu[f + 1])] = True
    assert bool(repaired.any()) == bool(int(p.ctx.halo.n_hist_rows))
    skipped: list[int] = []

    def skip(*args, **kwargs) -> bool:
        skipped.append(1)
        return False

    try:
        order = _recorder(monkeypatch)
        monkeypatch.setattr(conv_kcp, "_fwd_repair", skip)
        conv_kcp._set_test_knobs(
            {"host_trim": 1, "fwd_main": fwd_main, "fwd_overlap": 1}
        )

        def launch(_draw: int, seed: int):
            p = _problem(lens, cuts, C, device, seed=seed)
            conv_kcp.cake_causal_conv1d_nondet_train_kcp_host_cache_clear()
            order.clear()
            skipped.clear()
            out = torch.full_like(p.x_local, float("nan"))
            y, _ = conv_kcp.cake_causal_conv1d_nondet_train_kcp_forward(
                p.x_local, p.w, p.ctx, out=out
            )
            torch.cuda.synchronize()
            assert order == ["fwd" if fwd_main else "kcp_fwd", "wait"], order
            assert skipped == [1]
            return {"y": y}, problem(p.x_local, p.w, p.ctx.cu_seqlens_local)

        assert_judged(launch, seed=0, label=f"{name} main alone")
    finally:
        monkeypatch.undo()
        _reset()


@pytest.mark.parametrize("main_in", [False, True], ids=["f", "sr"])
@pytest.mark.parametrize("overlap_in", [False, True], ids=["wait", "pair"])
@pytest.mark.parametrize(
    "name,lens,cuts,Cs", MROWS + MROWS_KEYS, ids=[r[0] for r in MROWS + MROWS_KEYS]
)
def test_multi_forward_forms_order_and_outputs(
    monkeypatch, name, lens, cuts, Cs, main_in, overlap_in
) -> None:
    """The multi entry has one host path (the trimmed one).  ``main_in`` / ``overlap_in`` put the device's
    architecture into ``MULTI_FWD_MAIN_ARCHS`` / ``MULTI_FWD_OVERLAP_ARCHS`` or take it out (knobs ``None``); every
    tensor's ``y`` of every call judged against the naive tier."""
    device = torch.device("cuda")
    arch = conv_kcp._device_arch(device)
    _reset()
    words: list[int] = []
    forward_work = conv_kcp._forward_work

    def recording_work(workspace, n_words, **kwargs):
        words.append(n_words)
        return forward_work(workspace, n_words, **kwargs)

    try:
        _move(monkeypatch, "MULTI_FWD_MAIN_ARCHS", arch, main_in)
        _move(monkeypatch, "MULTI_FWD_OVERLAP_ARCHS", arch, overlap_in)
        monkeypatch.setattr(conv_kcp, "_forward_work", recording_work)
        order = _recorder(monkeypatch)

        def run(ctx, xs, ws, dys):
            conv_kcp.cake_causal_conv1d_nondet_train_kcp_host_cache_clear()
            two = _TwoRanks(ctx)
            want = _want(ctx, sr=main_in, overlap=overlap_in, n=len(Cs))
            found = [[] for _ in xs]
            for repeat in range(REPEATS):
                order.clear()
                words.clear()
                outs = [torch.full_like(x, float("nan")) for x in xs]
                ys, _ = conv_kcp.cake_causal_conv1d_nondet_train_kcp_multi_forward(
                    xs, ws, two, outs=outs
                )
                torch.cuda.synchronize()
                label = f"{name} main={main_in} overlap={overlap_in} #{repeat}"
                assert order == want, (label, order)
                # one counter buffer per call, sized for the largest tensor (none for the kcp_fwd program)
                assert len(words) == 1 and (words[0] > 0) == main_in, (label, words)
                assert conv_kcp.LAST_CALL["fwd_multi"] == _record(
                    ctx, sr=main_in, overlap=overlap_in
                ), label
                for i, y in enumerate(ys):
                    assert y is outs[i], (label, i)
                    found[i].append(y)
            return [{"y": values} for values in found]

        judge_multi(
            lens, cuts, Cs, run, label=f"{name} main={main_in} overlap={overlap_in}"
        )
    finally:
        monkeypatch.undo()
        _reset()


@pytest.mark.parametrize("main_in", [False, True], ids=["f", "sr"])
@pytest.mark.parametrize("overlap_in", [False, True], ids=["wait", "pair"])
def test_single_forward_follows_its_architecture_sets(
    monkeypatch, main_in, overlap_in
) -> None:
    """Knobs ``None``: the single entry's form follows ``FWD_MAIN_ARCHS`` / ``FWD_OVERLAP_ARCHS`` (the device's
    architecture put in or taken out, the host and graph caches cleared), whatever the multi entry's sets hold; the
    forward's graph-key variant is that form; ``y`` judged against the naive tier."""
    device = torch.device("cuda")
    arch = conv_kcp._device_arch(device)
    name, lens, cuts, C = ROWS[0]
    p = _problem(lens, cuts, C, device)
    try:
        _move(monkeypatch, "FWD_MAIN_ARCHS", arch, main_in)
        _move(monkeypatch, "FWD_OVERLAP_ARCHS", arch, overlap_in)
        _move(monkeypatch, "MULTI_FWD_MAIN_ARCHS", arch, not main_in)
        _move(monkeypatch, "MULTI_FWD_OVERLAP_ARCHS", arch, not overlap_in)
        _reset()
        assert conv_kcp._fwd_variant(p.x_local) == (main_in, overlap_in)
        assert conv_kcp._fwd_variant(p.x_local, multi=True) == (
            not main_in,
            not overlap_in,
        )
        order = _recorder(monkeypatch)

        def run(p):
            want = _want(p.ctx, sr=main_in, overlap=overlap_in)
            found = []
            for repeat in range(REPEATS):
                order.clear()
                y, _ = conv_kcp.cake_causal_conv1d_nondet_train_kcp_forward(
                    p.x_local, p.w, p.ctx
                )
                torch.cuda.synchronize()
                label = f"{name} main={main_in} overlap={overlap_in} #{repeat}"
                assert order == want, (label, order)
                assert conv_kcp.LAST_CALL["fwd"] == _record(
                    p.ctx, sr=main_in, overlap=overlap_in
                ), label
                found.append(y)
            return {"y": found}

        judge_kcp(lens, cuts, C, run, label=f"main={main_in} overlap={overlap_in}")
    finally:
        monkeypatch.undo()
        _reset()


@pytest.mark.parametrize("fwd_overlap", [0, 1])
def test_single_rank_main_counters_are_zero_after_every_call(
    monkeypatch, fwd_overlap
) -> None:
    """The single-rank main program's ``work`` counters (the package's buffer of the device and stream, or a fresh one
    inside a graph capture) are zero after every call: eager calls, then ``graph="replay"`` (eager, capture, replay,
    replay; ``LAST_CALL["fwd"]`` keeps the captured form on a replay; the graphs of the previous draw dropped first),
    then the multi entry (one buffer per call).  ``y`` of every call judged against the naive tier."""
    name, lens, cuts, C = ROWS[0]
    buffers: list[torch.Tensor] = []
    forward_work = conv_kcp._forward_work

    def recording_work(workspace, n_words, **kwargs):
        counters = forward_work(workspace, n_words, **kwargs)
        if counters is not None:
            buffers.append(counters)
        return counters

    try:
        monkeypatch.setattr(conv_kcp, "_forward_work", recording_work)
        _recorder(monkeypatch)
        conv_kcp._set_test_knobs({"fwd_main": 1, "fwd_overlap": fwd_overlap})

        def run(p):
            torch.cuda.synchronize()
            buffers.clear()
            conv_kcp.cake_causal_conv1d_nondet_train_kcp_host_cache_clear()
            conv_kcp.cake_causal_conv1d_nondet_train_kcp_graph_cache_clear()
            form = _record(p.ctx, sr=True, overlap=bool(fwd_overlap))
            xs, ws = (p.x_local,) * 3, (p.w,) * 3
            found = []
            events = []
            for graph in ("off", "off", "replay", "replay", "replay", "replay"):
                y, _ = conv_kcp.cake_causal_conv1d_nondet_train_kcp_forward(
                    p.x_local, p.w, p.ctx, graph=graph
                )
                torch.cuda.synchronize()
                record = dict(conv_kcp.LAST_CALL["fwd"])
                if graph == "replay":
                    events.append(record.pop("graph_event"))
                label = f"{name} overlap={fwd_overlap} graph={graph} #{len(events)}"
                assert {key: record[key] for key in form} == form, (label, record)
                assert buffers, label
                assert all(not bool(b.any()) for b in buffers), label
                found.append(y.clone())
            assert events == ["eager", "capture", "replay", "replay"], events
            two = _TwoRanks(p.ctx)
            for repeat in range(2):
                buffers.clear()
                ys, _ = conv_kcp.cake_causal_conv1d_nondet_train_kcp_multi_forward(
                    xs, ws, two
                )
                torch.cuda.synchronize()
                assert len(buffers) == 1 and not bool(buffers[0].any()), repeat
                found.extend(ys)
            return {"y": found}

        judge_kcp(lens, cuts, C, run, label=f"counters overlap={fwd_overlap}")
    finally:
        torch.cuda.synchronize()
        buffers.clear()
        monkeypatch.undo()
        _reset()


@pytest.mark.parametrize("fwd_main,fwd_overlap", FORMS[1:])
def test_two_launch_forward_graph_replay_keeps_the_form(
    monkeypatch, fwd_main, fwd_overlap
) -> None:
    """``graph="replay"`` of every two-launch form, single and multi entry (eager, capture, replay, replay; the
    exchange in flight; the graphs of the previous draw dropped first): ``LAST_CALL["fwd"]`` /
    ``LAST_CALL["fwd_multi"]`` hold the form on every call (a replay repeats the captured call's) and every ``y``
    is judged against the naive tier, the overlapped ``kcp_fwd`` main program's all ``-1`` table included."""
    name, lens, cuts, C = ROWS[0]
    try:
        _recorder(monkeypatch)
        conv_kcp._set_test_knobs({"fwd_main": fwd_main, "fwd_overlap": fwd_overlap})

        def run(p):
            conv_kcp.cake_causal_conv1d_nondet_train_kcp_host_cache_clear()
            conv_kcp.cake_causal_conv1d_nondet_train_kcp_graph_cache_clear()
            form = _record(p.ctx, sr=bool(fwd_main), overlap=bool(fwd_overlap))
            xs, ws = (p.x_local,) * 3, (p.w,) * 3
            two = _TwoRanks(p.ctx)
            found = []
            for kind in ("fwd", "fwd_multi"):
                events = []
                for repeat in range(4):
                    if kind == "fwd":
                        y, _ = conv_kcp.cake_causal_conv1d_nondet_train_kcp_forward(
                            p.x_local, p.w, p.ctx, graph="replay"
                        )
                        ys = (y,)
                    else:
                        ys, _ = conv_kcp.cake_causal_conv1d_nondet_train_kcp_multi_forward(
                            xs, ws, two, graph="replay"
                        )
                    torch.cuda.synchronize()
                    record = dict(conv_kcp.LAST_CALL[kind])
                    events.append(record.pop("graph_event"))
                    label = f"{name} {kind} main={fwd_main} overlap={fwd_overlap} #{repeat}"
                    assert {key: record[key] for key in form} == form, (label, record)
                    found.extend(y.clone() for y in ys)
                assert events == ["eager", "capture", "replay", "replay"], (kind, events)
            return {"y": found}

        judge_kcp(lens, cuts, C, run, label=f"graph main={fwd_main} overlap={fwd_overlap}")
    finally:
        monkeypatch.undo()
        _reset()


@pytest.mark.parametrize("name,lens,cuts,C", ROWS, ids=[r[0] for r in ROWS])
@pytest.mark.parametrize("fwd_main", [0, 1])
def test_one_rank_forward_takes_the_waiting_form(name, lens, cuts, C, fwd_main) -> None:
    """At one rank the package's exchange returns no work object (no recorder here): with ``fwd_overlap`` 1 the single
    entry (both host paths) and the multi entry take the waiting form -- the single-rank main program and, when the
    rank's fragments read history rows, the history-row program, or the one-launch ``kcp_fwd`` forward; ``y`` into
    NaN-filled ``out=`` judged against the naive tier."""
    sr = bool(fwd_main)

    def run(p):
        assert int(p.ctx.world_size) == 1
        xs, ws = (p.x_local,) * 2, (p.w,) * 2
        want = {
            "main": "sr" if sr else "f",
            "form": "wait",
            "repair": sr and bool(int(p.ctx.halo.n_hist_rows)),
        }
        found = []
        for host_trim in (1, 0):
            conv_kcp._set_test_knobs(
                {"host_trim": host_trim, "fwd_main": fwd_main, "fwd_overlap": 1}
            )
            conv_kcp.cake_causal_conv1d_nondet_train_kcp_host_cache_clear()
            out = torch.full_like(p.x_local, float("nan"))
            y, _ = conv_kcp.cake_causal_conv1d_nondet_train_kcp_forward(
                p.x_local, p.w, p.ctx, out=out
            )
            torch.cuda.synchronize()
            label = f"{name} trim={host_trim} main={fwd_main}"
            assert y is out, label
            assert conv_kcp.LAST_CALL["fwd"] == want, (label, conv_kcp.LAST_CALL)
            found.append(y)
        outs = [torch.full_like(x, float("nan")) for x in xs]
        ys, _ = conv_kcp.cake_causal_conv1d_nondet_train_kcp_multi_forward(
            xs, ws, p.ctx, outs=outs
        )
        torch.cuda.synchronize()
        label = f"{name} multi main={fwd_main}"
        assert conv_kcp.LAST_CALL["fwd_multi"] == want, (label, conv_kcp.LAST_CALL)
        for i in range(len(xs)):
            assert ys[i] is outs[i], (label, i)
            found.append(ys[i])
        return {"y": found}

    try:
        judge_kcp(lens, cuts, C, run, label=f"{name} main={fwd_main}")
    finally:
        _reset()


@pytest.mark.parametrize("fwd_main,fwd_overlap", FORMS)
def test_recompute_handle_replays_the_waiting_form(
    monkeypatch, fwd_main, fwd_overlap
) -> None:
    """A checkpoint with a recompute handle under every form: the original forward exchanges (the recorder's exchange,
    so the overlapped form runs at one rank) and takes the form, the recompute binds the stored pool and takes the
    waiting form with no exchange and no wait; ``y``, ``dx`` and ``dweight`` of the checkpointed pass and of the
    shipped direct pass are judged against the naive tier."""
    name, lens, cuts, C = ROWS[0]
    sr, overlap = bool(fwd_main), bool(fwd_overlap)
    records: list[dict] = []
    fwd_record = conv_kcp._fwd_record

    def recording_record(kind, *args) -> None:
        fwd_record(kind, *args)
        records.append(dict(conv_kcp.LAST_CALL[kind]))

    def run(p):
        _reset()
        direct = _ckpt_run(p, ckpt=False)
        torch.cuda.synchronize()
        records.clear()
        try:
            order = _recorder(monkeypatch)
            monkeypatch.setattr(conv_kcp, "_fwd_record", recording_record)
            conv_kcp._set_test_knobs({"fwd_main": fwd_main, "fwd_overlap": fwd_overlap})
            conv_kcp.cake_causal_conv1d_nondet_train_kcp_host_cache_clear()
            handle = conv_kcp.CakeCausalConv1dNondetKcpRecomputeHandle()
            conv_kcp.LAST_CALL.pop("recompute_handle", None)
            got = _ckpt_run(p, handle=handle)
            torch.cuda.synchronize()
            assert conv_kcp.LAST_CALL["recompute_handle"] == {"fwd": "replay"}
            assert records == [
                _record(p.ctx, sr=sr, overlap=overlap),
                _record(p.ctx, sr=sr, overlap=False),
            ], records
            # the original forward's wait and the backward's; the recompute waits for nothing
            assert order.count("wait") == 2, order
        finally:
            monkeypatch.undo()
            _reset()
        return {
            "y": [direct["y"], got["y"]],
            "dx": [direct["dx"], got["dx"]],
            "dweight": [direct["dw"], got["dw"]],
        }

    judge_kcp(lens, cuts, C, run, label=f"{name} main={fwd_main} overlap={fwd_overlap}")


def test_two_launch_forward_steady_round_does_not_block_or_allocate(
    monkeypatch,
) -> None:
    """Census (``tests/experimental/test_helpers/cake_host_sync.py``) of a steady round of every two-launch form: the single entry
    and the three-tensor multi entry, each with an exchange in flight.  No pageable or pinned host-to-device copy, no
    synchronizing operation (the probe queued before the round still runs when the host returns), no device
    allocation or free once the first rounds have run.  The outputs of every form are judged by the tests above."""
    from tests.experimental.test_helpers.cake_host_sync import (
        allocator_counts,
        census,
        nonblocking,
        summary,
    )

    device = torch.device("cuda")
    alloc_ok = allocator_counts(device)
    p = _problem((5000, 1003), (1, 2, 4997, 1003), 512, device)
    xs, ws = (p.x_local,) * 3, (p.w,) * 3
    two = _TwoRanks(p.ctx)
    _reset()
    try:
        order = _recorder(monkeypatch)
        for fwd_main, fwd_overlap in FORMS[1:]:
            conv_kcp._set_test_knobs({"fwd_main": fwd_main, "fwd_overlap": fwd_overlap})

            def rnd():
                order.clear()
                y, hist = conv_kcp.cake_causal_conv1d_nondet_train_kcp_forward(
                    p.x_local, p.w, p.ctx
                )
                ys, hists = conv_kcp.cake_causal_conv1d_nondet_train_kcp_multi_forward(
                    xs, ws, two
                )
                return [y, hist, *ys, *hists]

            rnd()
            rnd()
            c = census(rnd, device)
            label = f"main={fwd_main} overlap={fwd_overlap}"
            print(f"CENSUS_CONV_NONDET_KCP_FWD {label} {summary(c)}")
            assert nonblocking(c), (label, summary(c))
            assert c["pinned"] == 0, (label, summary(c))
            if alloc_ok:
                assert (c["device_alloc"], c["device_free"], c["alloc_retries"]) == (
                    0,
                    0,
                    0,
                ), (label, summary(c))
                assert c["prof_malloc"] in (None, 0), (label, summary(c))
    finally:
        monkeypatch.undo()
        _reset()
    if not alloc_ok:
        pytest.skip(
            "the caching allocator's statistics do not count device allocations here"
        )


@pytest.mark.parametrize("arch", cake_jit.ARCHES)
def test_route_rows_select_every_program(arch) -> None:
    """The rows of ``ROUTE_ROWS`` select every program of the architecture's zero-history ``fwd`` route (a route with
    another program set needs rows of its own here): the static and the direct program at the C = 640 rows inside
    the key's ranges, which take no counters, and the dynamic programs at the others; the C = 640 row of the
    multi-entry row of ``MROWS_KEYS`` selects the static program."""
    picked = {
        name: cake_jit.selected_modules(arch, conv_kcp.FWD_MAIN_ROUTE, C=C, S=sum(lens))[0]
        for name, lens, _cuts, C in ROUTE_ROWS
    }
    assert set(picked.values()) == set(cake_jit.route_modules(arch, conv_kcp.FWD_MAIN_ROUTE)), (arch, picked)
    for name, kind in (("c640-s3003", None), ("c640-s12003", cake_jit.ROWS_KIND)):
        record = cake_jit.MODULES[picked[name]]
        assert (cake_jit.select_rule(record) or {"kind": None})["kind"] == kind, (arch, name)
        assert record["workspace_rule"] is None, (arch, name)
    for _name, lens, _cuts, Cs in MROWS_KEYS:
        (name,) = cake_jit.selected_modules(arch, conv_kcp.FWD_MAIN_ROUTE, C=Cs[0], S=sum(lens))
        assert cake_jit.select_rule(cake_jit.MODULES[name]) is None, (arch, name)


@pytest.mark.parametrize("name,lens,cuts,C", ROUTE_ROWS, ids=[r[0] for r in ROUTE_ROWS])
@pytest.mark.parametrize("fwd_overlap", [0, 1])
def test_single_rank_main_is_the_selected_route_program(
    monkeypatch, name, lens, cuts, C, fwd_overlap
) -> None:
    """The single-rank main program is the record ``cake_jit.selected_modules(arch, "fwd", C=C, S=S)`` picks for the
    call, on both host paths; ``y`` into a NaN-filled ``out=`` judged against the naive tier."""
    device = torch.device("cuda")
    arch = conv_kcp._device_arch(device)
    names: list[str] = []
    try:
        order = _recorder(monkeypatch)
        zero_history_plan = conv_kcp._zero_history_plan

        def recording(record_name, grid, device_index):
            names.append(record_name)
            return zero_history_plan(record_name, grid, device_index)

        monkeypatch.setattr(conv_kcp, "_zero_history_plan", recording)

        def run(p):
            S = int(p.x_local.shape[0])
            (expected,) = cake_jit.selected_modules(arch, conv_kcp.FWD_MAIN_ROUTE, C=C, S=S)
            found = []
            for host_trim in (1, 0):
                conv_kcp._set_test_knobs(
                    {"host_trim": host_trim, "fwd_main": 1, "fwd_overlap": fwd_overlap}
                )
                conv_kcp.cake_causal_conv1d_nondet_train_kcp_host_cache_clear()
                names.clear()
                order.clear()
                out = torch.full_like(p.x_local, float("nan"))
                y, _ = conv_kcp.cake_causal_conv1d_nondet_train_kcp_forward(
                    p.x_local, p.w, p.ctx, out=out
                )
                torch.cuda.synchronize()
                label = f"{name} {arch} trim={host_trim} overlap={fwd_overlap}"
                assert names == [expected], (label, names)
                assert order == _want(p.ctx, sr=True, overlap=bool(fwd_overlap)), (
                    label,
                    order,
                )
                found.append(y)
            return {"y": found}

        judge_kcp(lens, cuts, C, run, label=f"{name} overlap={fwd_overlap}")
    finally:
        monkeypatch.undo()
        _reset()
