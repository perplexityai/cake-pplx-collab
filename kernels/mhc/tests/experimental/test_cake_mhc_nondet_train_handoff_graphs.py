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

CUDA-graph capture and steady-state host behavior of the pre handoff and the standalone collapse of
``cake_mhc_nondet_train`` (hidden 5120, ``rms_eps`` 1e-20, ``hc_eps`` 1e-6, 20 Sinkhorn iterations), in the form of
``test_cake_mhc_nondet_train_graphs.py``.

Each iteration is captured whole with ``torch.cuda.graph`` after three warm-up iterations on a side stream:

* one handoff sublayer through the direct entry points (``handoff_forward_prepare`` -> ``forward_finish`` ->
  ``backward_prepare`` -> ``handoff_backward_finish``), on the default workspaces and on caller workspaces (poisoned
  before every replay, the backward workspace's zero prefix zero after it);
* the same sublayer through the autograd entry points (``handoff_prepare`` / ``finish``, ``save_stats`` both ways);
* two sublayers from the one-hot start through the autograd final ``collapse`` (the issue's chain, judged per sublayer on
  the inputs each received and at chain level in the median, the issue owner's answer on #79);
* the standalone collapse and its VJP.

Each captured iteration is replayed on 32 inputs copied into its static inputs, after its static outputs were
overwritten with 0xFF bytes, and every output and gradient is judged against the naive BF16 tier
(``tests/cake_naive.py``).  The steady state needs no host synchronization: after a warm-up, eager iterations of every
entry point run under ``torch.cuda.set_sync_debug_mode("error")``.
"""

from __future__ import annotations

import pytest
import torch

from tests.cake.test_helpers.cake_mhc_handoff import (
    CONSTANTS,
    HIDDEN,
    SUBLAYER_OUTPUTS,
    Api,
    ChainMedian,
    chain_expected_and_naive,
    chain_inputs,
    chain_names,
    chain_package,
    chain_sublayer_cases,
    collapse_expected_and_naive,
    make_inputs,
    sublayer_forward_backward,
)
from tests.cake.test_helpers.cake_mhc_naive import (
    expected_and_naive,
    judge_draws,
    package_outputs,
)
from tests.cake.test_helpers.cake_package import cake_package
from tests.cake_naive import DRAWS

KEY = "cake.cake_mhc_nondet_train"
pkg = cake_package(KEY)
loader = cake_package(KEY, "cake_jit")
api = Api(pkg, loader, "nondet")
SEED = 7979
ZERO_PREFIX_BYTES = 768


def _device() -> torch.device:
    return torch.device("cuda:0")


def _supported() -> bool:
    return torch.cuda.is_available() and pkg.is_cake_mhc_nondet_train_handoff_supported(
        _device(), HIDDEN
    )


pytestmark = pytest.mark.skipif(
    not _supported(),
    reason="the pre handoff of cake_mhc_nondet_train needs an SM100 / SM103 / SM107 device (hidden 5120)",
)


def _poison(out: dict) -> None:
    for v in out.values():
        v.detach().reshape(-1).view(torch.uint8).fill_(0xFF)


def _capture(step, leaves=()):
    """``step()`` captured in a ``torch.cuda.CUDAGraph`` after three warm-up iterations on a side stream -> (the
    graph, its static outputs)."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            for t in leaves:
                t.grad = None
            step()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    for t in leaves:
        t.grad = None
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = step()
    return graph, out


def _copier(statics: dict):
    """``load(new)``: copy ``new[k]`` into each static input ``statics[k]``."""

    def load(new: dict) -> None:
        for k, t in statics.items():
            t.copy_(new[k])

    return load


def _replay_judged(
    graph,
    out: dict,
    load,
    make,
    expect,
    record_property,
    *,
    seed: int,
    before=None,
    after=None,
):
    """Replays of ``graph`` on 32 inputs (``make(s)``, copied into the static inputs by ``load``), each judged by
    ``expect(i, got)`` -> ``(expected, naive)``; ``before()`` / ``after(j)`` run around each replay."""

    def case(j: int, s: int):
        new = make(s)
        with torch.no_grad():
            load(new)
        _poison(out)
        if before is not None:
            before()
        graph.replay()
        torch.cuda.synchronize()
        if after is not None:
            after(j)
        got = {k: v.detach().clone() for k, v in out.items()}
        expected, naive = expect(new, got)
        return [got], expected, naive

    judge_draws(case, record_property, seed=seed)


def _sublayer_expect(i, got):
    expected, naive = expected_and_naive(i, SUBLAYER_OUTPUTS)
    return {k: expected[k] for k in got}, {k: naive[k] for k in got}


@pytest.mark.parametrize("workspace", ("default", "caller"))
@pytest.mark.parametrize("T", (129, 2049))
def test_captured_handoff_sublayer(T: int, workspace: str, record_property) -> None:
    """One handoff sublayer through the direct entry points, captured and replayed on 32 inputs; every output of
    :data:`SUBLAYER_OUTPUTS` judged against the naive tier."""
    statics = make_inputs(T, device=_device(), seed=SEED)
    fwd_ws = bwd_ws = None
    if workspace == "caller":
        fwd_ws = pkg.cake_mhc_nondet_train_forward_workspace(
            T, _device(), hidden=HIDDEN
        )
        bwd_ws = pkg.cake_mhc_nondet_train_backward_workspace(
            T, _device(), hidden=HIDDEN
        )

    def step():
        return package_outputs(
            sublayer_forward_backward(api, statics, fwd_ws=fwd_ws, bwd_ws=bwd_ws)
        )

    graph, out = _capture(step)
    tensors = {
        k: statics[k]
        for k in ("X", "fn", "scale", "base", "B", "dY", "dU", "pre_in", "dpre_next")
    }

    def before():
        if bwd_ws is not None:
            bwd_ws[ZERO_PREFIX_BYTES:].fill_(0xFF)

    def after(j: int):
        if bwd_ws is not None:
            assert int(torch.count_nonzero(bwd_ws[:ZERO_PREFIX_BYTES])) == 0, (
                f"replay {j}: zero prefix not zero"
            )

    _replay_judged(
        graph,
        out,
        _copier(tensors),
        lambda s: make_inputs(T, device=_device(), seed=s),
        _sublayer_expect,
        record_property,
        seed=SEED + T,
        before=before,
        after=after,
    )


@pytest.mark.parametrize("save_stats", (True, False))
def test_captured_autograd_handoff_sublayer(save_stats: bool, record_property) -> None:
    """``handoff_prepare`` / ``finish`` and ``autograd.backward([U, Y, pre_next], [dU, dY, dpre_next])`` captured
    whole and replayed on 32 inputs: the forward outputs and the gradients judged against the naive tier."""
    T = 257
    statics = make_inputs(T, device=_device(), seed=SEED + 1)
    leaves = {
        "X": statics["X"].requires_grad_(True),
        "pre_in": statics["pre_in"].requires_grad_(True),
        "fn": statics["fn"].requires_grad_(True),
        "scale": statics["scale"].requires_grad_(True),
        "base": statics["base"].requires_grad_(True),
        "B": statics["B"].requires_grad_(True),
    }

    def step():
        U, pre_next, handle = pkg.cake_mhc_nondet_train_handoff_prepare(
            leaves["X"], leaves["pre_in"], leaves["fn"], leaves["scale"], leaves["base"],
            save_stats=save_stats, **CONSTANTS,
        )  # fmt: skip
        Y = pkg.cake_mhc_nondet_train_finish(leaves["X"], leaves["B"], handle)
        torch.autograd.backward(
            [U, Y, pre_next], [statics["dU"], statics["dY"], statics["dpre_next"]]
        )
        return {
            "U": U.detach(), "Y": Y.detach(), "pre_next": pre_next.detach(), "dX": leaves["X"].grad,
            "dpre_in": leaves["pre_in"].grad, "dfn": leaves["fn"].grad, "dscale": leaves["scale"].grad,
            "dbase": leaves["base"].grad, "dB": leaves["B"].grad,
        }  # fmt: skip

    graph, out = _capture(step, tuple(leaves.values()))
    tensors = {
        k: (v.detach() if k in leaves else v)
        for k, v in statics.items()
        if isinstance(v, torch.Tensor)
    }
    _replay_judged(
        graph,
        out,
        _copier(tensors),
        lambda s: make_inputs(T, device=_device(), seed=s),
        _sublayer_expect,
        record_property,
        seed=SEED + 2,
    )


def test_captured_two_sublayer_chain(record_property) -> None:
    """Two handoff sublayers from the one-hot start through the autograd final collapse (branches ``tanh(U W)``,
    ``U_final.backward(G)``) captured whole and replayed on 32 inputs, judged as the issue owner's answer on #79 asks:
    each sublayer and the final collapse on the inputs it received in the replayed chain (recorded inside the graph;
    every rule of the naive tier, ``dpre_in`` also with the FP32 gate), and the chain's outputs and gradients against
    the FP64 chain in the median (:class:`~tests.cake.test_helpers.cake_mhc_handoff.ChainMedian`)."""
    T = 129
    statics = chain_inputs(T, device=_device(), seed=SEED + 3)
    names = chain_names(2)
    sublayers = len(statics["params"])

    def step():
        out = chain_package(api, statics, capture=True)
        flat = {k: out[k] for k in names}
        for k, (inputs, outputs) in enumerate(out["sublayers"]):
            flat.update(
                {
                    f"s{k}.in.{n}": v
                    for n, v in inputs.items()
                    if isinstance(v, torch.Tensor)
                }
            )
            flat.update({f"s{k}.out.{n}": v for n, v in outputs.items()})
        inputs, outputs = out[
            "collapse"
        ]  # its dU is the static G (an input: not poisoned)
        flat.update({"c.in.X": inputs["X"], "c.in.pre_in": inputs["pre_in"]})
        flat.update({f"c.out.{n}": v for n, v in outputs.items()})
        return flat

    graph, out = _capture(step)

    def load(new: dict) -> None:
        statics["X"].copy_(new["X"])
        statics["G"].copy_(new["G"])
        for ps, qs in zip(statics["params"], new["params"], strict=True):
            for p, q in zip(ps, qs, strict=True):
                p.copy_(q)
        for W, V in zip(statics["W"], new["W"], strict=True):
            W.copy_(V)

    def part(got: dict, head: str) -> dict:
        return {n[len(head) :]: v for n, v in got.items() if n.startswith(head)}

    chain, gates = ChainMedian(), {}

    def case(j: int, s: int):
        new = chain_inputs(T, device=_device(), seed=s)
        with torch.no_grad():
            load(new)
        _poison(out)
        graph.replay()
        torch.cuda.synchronize()
        got = {k: v.detach().clone() for k, v in out.items()}
        expected, naive = chain_expected_and_naive(new)
        chain.add(j, [{k: got[k] for k in names}], expected, naive)
        del expected, naive
        rebuilt = {
            "sublayers": [
                ({**part(got, f"s{k}.in."), **CONSTANTS}, part(got, f"s{k}.out."))
                for k in range(sublayers)
            ],
            "collapse": ({**part(got, "c.in."), "dU": new["G"]}, part(got, "c.out.")),
        }
        yield from chain_sublayer_cases(rebuilt, j, gates)

    judge_draws(case, record_property, seed=SEED + 4)
    for gate in gates.values():
        gate.check(record_property, DRAWS)
    chain.check(record_property)


@pytest.mark.parametrize("T", (2049, 768, 129, 64, 34))
def test_captured_standalone_collapse(T: int, record_property) -> None:
    """``collapse_forward`` and ``collapse_backward`` captured and replayed on 32 inputs: ``U``, ``dX`` and ``dpre_in``
    judged against the naive tier (T 768 / 129 / 64 / 34: the split VJPs of the token-count bands of
    ``collapse_backward``; 34 the stage-free split VJP on sm_100a / sm_103a)."""
    statics = make_inputs(T, device=_device(), seed=SEED + 5)

    def step():
        U = pkg.cake_mhc_nondet_train_collapse_forward(statics["X"], statics["pre_in"])
        dX, dpre_in = pkg.cake_mhc_nondet_train_collapse_backward(
            statics["X"], statics["pre_in"], statics["dU"]
        )
        return {"U": U, "dX": dX, "dpre_in": dpre_in}

    graph, out = _capture(step)
    _replay_judged(
        graph,
        out,
        _copier({k: statics[k] for k in ("X", "pre_in", "dU")}),
        lambda s: make_inputs(T, device=_device(), seed=s),
        lambda i, got: collapse_expected_and_naive(i),
        record_property,
        seed=SEED + 6,
    )


@pytest.mark.parametrize("T", (129, 383, 384, 1023, 16384))
def test_handoff_steady_state_has_no_host_sync(T: int) -> None:
    """After a warm-up, eager iterations of the direct handoff entry points, the autograd chain and the standalone
    collapse run under ``torch.cuda.set_sync_debug_mode("error")`` (any synchronizing CUDA call raises)."""
    i = make_inputs(T, device=_device(), seed=SEED + 7)
    chain = chain_inputs(T, device=_device(), seed=SEED + 8)

    def iteration():
        sublayer_forward_backward(api, i)
        chain_package(api, chain)
        pkg.cake_mhc_nondet_train_collapse_forward(i["X"], i["pre_in"])
        pkg.cake_mhc_nondet_train_collapse_backward(i["X"], i["pre_in"], i["dU"])

    for _ in range(2):
        iteration()
    torch.cuda.synchronize()
    previous = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        for _ in range(3):
            iteration()
    finally:
        torch.cuda.set_sync_debug_mode(previous)
    torch.cuda.synchronize()
