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

The output backward of the non-deterministic context-parallel (KCP) KDA training route: the K2 adjoint walks
(``kcp_dhu_offset`` / ``_overlap`` / ``_vh64``, ``kcp_dhu_tape``) form the first value gradient
``dv1 = Aqk^T do`` in FP32 from the internal ``aqk_int`` rows, so ``kcp_dav`` launches only where the plan
dispatches its dav stage (:func:`cake_kcp_plan.kcp_dav_needed`: the fused-gradient records that store no dAqk read
``kcp_dav``'s).  GPU rows, one rank, on two packs (five ragged documents with partial last chunks, ``P = 1``; one
4000-token document, ``P > 1`` by the one-wave fill):

* coverage, per gate mode x norm x H {4, 8, 12, 16, 32, 64} x path (saved, replay, ``saved_rows`` intra / wy,
  checkpoint hand-off with its tape set (``saved_rows="tape"``, recorded ``"bound"``) and with the ``"auto"`` rows
  form (recorded ``"wy"``)): the shipped ``"auto"`` (twice, repeats) and the private knob ``kcp_dav="always"`` are
  judged; the dAqk buffer NaN-filled right after the ``dav`` stage reaches no output where ``kcp_dav`` is skipped (the
  fused gradient stores every dAqk row before its reader: that run is judged too, a NaN breaks its contract) and
  reaches the gradients where it runs (the negative control: an output holds a NaN or an inf); the backward's record
  says ``"launched"`` / ``"skipped"`` as the plan decides;
* forms: the private knob ``walk_form`` ``"auto"``, ``"full"`` and ``"split"`` judged at H 63 and 64 (the K2 offset's
  head rule ``cake_kcp_jit.WALK_HEADS_RULE``: ``kcp_dhu_offset`` up to 63 heads, ``kcp_dhu_offset_overlap`` above),
  saved and replay, both gate modes, the norm off and on; the full-width run launches the offset record the rule
  names.

Accuracy: this package has no bit contract, so no run is compared with another's bits.  Every output and gradient of
every run (``"<arm>.<output>"``) is judged against the FP64 reference of the pack beside the naive BF16 tier of
``tests/cake_naive.py`` over 32 draws of the operands (``cake_kda_naive.judge_arms``; the owner-accepted gaps only).
The record and launch checks run on every draw.
"""

from __future__ import annotations

import itertools

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from tests.experimental.test_cake_kda_nondet_train_kcp_l2n import _l2n_inputs
from tests.experimental.test_cake_kda_nondet_train_kcp_shared_ws import (
    _device_or_skip,
)
from tests.experimental.test_helpers.cake_kda_det_train_reference import (
    LOWER_BOUND,
    make_inputs,
    make_raw_inputs,
)
from tests.experimental.test_helpers.cake_kda_naive import judge_arms
from tests.experimental.test_helpers.cake_kda_naive_kcp import to_global
from tests.cake.test_helpers.cake_package import cake_package

kp = cake_package("cake.cake_kda_nondet_train_kcp", "cake_kcp_plan")
cake_kcp = cake_package("cake.cake_kda_nondet_train_kcp", "cake_kcp")
cake_kcp_jit = cake_package("cake.cake_kda_nondet_train_kcp", "cake_kcp_jit")

# (lengths of the one-rank pack): five documents with partial last chunks (P = 1); one document (P > 1)
PACKS = {
    "mdoc5": (100, 640, 77, 1000, 129),
    "one4k": (4000,),
}
HEADS = (4, 8, 12, 16, 32, 64)
# "handoff": checkpoint_handoff="on" with saved_rows="tape" (the hand-off's tape set); "handoff_rows": the same with
# saved_rows="auto" (an engaged hand-off keeps the "wy" rows form)
PATHS = ("saved", "replay", "rows_intra", "rows_wy", "handoff", "handoff_rows")
FORM_HEADS = (63, 64)
K2_PROGRAMS = ("kcp_dhu_offset", "kcp_dhu_tape")
# the judged operands: draw j of a case at draw_seed(SEED, j) (tests/cake_naive.py)
SEED = 0x7817


@pytest.fixture(scope="module")
def device() -> torch.device:
    return _device_or_skip()


_CTX: dict = {}


def _ctx(pack: str, device: torch.device):
    """The one-rank context of a pack (cached per device)."""
    key = (pack, device.index)
    if key not in _CTX:
        cu = torch.tensor([0, *itertools.accumulate(PACKS[pack])], dtype=torch.int64)
        part = kp.cake_kda_nondet_train_kcp_partition(cu, 1)
        _CTX[key] = (
            part,
            cu.tolist(),
            kp.cake_kda_nondet_train_kcp_build_context(part, None, 0, device),
        )
    return _CTX[key]


def _operands(
    pack: str, H: int, gate: str, l2n: bool, device, seed: int
) -> tuple[dict, dict]:
    """The operands of draw seed ``seed`` of the pack in the rank's fragment order (one rank: its fragments cover every
    token), and the whole pack in the global order (the judge's operands)."""
    lengths = PACKS[pack]
    raw = gate == "raw"
    if l2n:
        glob = _l2n_inputs(lengths, H, device, raw_gate=raw, seed=seed)
    elif raw:
        glob = make_raw_inputs(lengths, H, device, seed=seed)
    else:
        glob = make_inputs(lengths, H, device, seed=seed)
    part, cu, _ = _ctx(pack, device)
    names = ("q", "k", "v", "g", "beta", "dout")
    ops = {}
    for name in names:
        t = glob[name].to(device)
        pieces = [
            t[cu[f.doc] + f.off : cu[f.doc] + f.off + f.length]
            for f in part.fragments[0]
        ]
        ops[name] = torch.cat(pieces, 0).contiguous()
    if raw:
        ops["A_log"], ops["dt_bias"] = (
            glob["A_log"].to(device),
            glob["dt_bias"].to(device),
        )
    return ops, glob


def _global(pack: str, res: dict, device) -> dict:
    """A one-rank run's results in the global token order (``dA_log`` / ``ddt_bias`` as they are)."""
    part, cu, _ = _ctx(pack, device)
    return {
        n: t if n in ("dA_log", "ddt_bias") else to_global([part.fragments[0]], [t], cu)
        for n, t in res.items()
    }


def _nonfinite(res: dict) -> list[str]:
    """The outputs that hold a NaN or an inf."""
    return [n for n, t in res.items() if not bool(torch.isfinite(t.float()).all())]


def _run(ctx, ops, gate: str, l2n: bool, path: str, knobs: dict, poison: bool = False):
    """One forward + backward of ``path`` on a fresh workspace; returns (outputs, the backward's record, the plan's
    ``kcp_dav_needed`` seen at the ``dav`` stage).  ``poison``: the dAqk buffer NaN-filled right after the ``dav``
    stage (stream-ordered, before the fused gradient)."""
    raw = gate == "raw"
    ws = cake_kcp.CakeKDANondetKcpWorkspace(
        use_gate_in_kernel=raw, use_qk_l2norm_in_kernel=l2n
    )
    q, k, v, g, beta, dout = (ops[n] for n in ("q", "k", "v", "g", "beta", "dout"))
    params = (ops["A_log"], ops["dt_bias"]) if raw else ()
    gkw = (
        dict(
            use_gate_in_kernel=True,
            A_log=params[0],
            dt_bias=params[1],
            lower_bound=LOWER_BOUND,
        )
        if raw
        else {}
    )
    opts = {"use_qk_l2norm_in_kernel": l2n, "_kcp_test_knobs": dict(knobs)}
    seen: dict = {}
    orig = cake_kcp.KernelBackend.stage

    def staged(self, name, call, **extra):
        orig(self, name, call, **extra)
        if name == "dav":
            seen["needed"] = bool(kp.kcp_dav_needed(call.local.plan))
            if poison:
                call.local_values["dAqk"].fill_(float("nan"))

    cake_kcp.KernelBackend.stage = staged
    try:
        cake_kcp.LAST_CALL.pop("bwd", None)
        if path.startswith("handoff"):
            hkw = {"saved_rows": "tape"} if path == "handoff" else {}
            leaves = [
                x.detach().clone().requires_grad_(True)
                for x in (q, k, v, g, beta, *params)
            ]

            def layer(q_, k_, v_, g_, b_, *p_):
                pkw = (
                    dict(
                        use_gate_in_kernel=True,
                        A_log=p_[0],
                        dt_bias=p_[1],
                        lower_bound=LOWER_BOUND,
                    )
                    if raw
                    else {}
                )
                return cake_kcp.cake_kda_nondet_kcp(
                    q_,
                    k_,
                    v_,
                    g_,
                    b_,
                    ctx,
                    workspace=ws,
                    checkpoint_handoff="on",
                    **hkw,
                    **pkw,
                    **opts,
                )

            out = checkpoint(layer, *leaves, use_reentrant=False)
            out.backward(dout)
            names = ("dq", "dk", "dv", "dg", "dbeta") + (
                ("dA_log", "ddt_bias") if raw else ()
            )
            res = {"out": out.detach()}
            res.update({n: x.grad for n, x in zip(names, leaves, strict=True)})
        else:
            fkw = {"saved_rows": path[5:]} if path.startswith("rows_") else {}
            out, saved = cake_kcp.cake_kda_nondet_train_kcp_forward(
                q, k, v, g, beta, ctx, workspace=ws, **gkw, **fkw, **opts
            )
            grads = cake_kcp.cake_kda_nondet_train_kcp_backward(
                q, k, v, g, beta, ctx, dout,
                saved=None if path == "replay" else saved, workspace=ws, **gkw, **opts,
            )  # fmt: skip
            names = ("dq", "dk", "dv", "dbeta", "dg") + (
                ("dA_log", "ddt_bias") if raw else ()
            )
            res = {"out": out}
            res.update(dict(zip(names, grads, strict=True)))
        torch.cuda.synchronize()
    finally:
        cake_kcp.KernelBackend.stage = orig
    rec = dict(cake_kcp.LAST_CALL.get("bwd") or {})
    res = {n: t.detach().clone() for n, t in res.items()}
    return res, rec, seen.get("needed")


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("H", HEADS)
@pytest.mark.parametrize("l2n", (False, True), ids=("plain", "l2n"))
@pytest.mark.parametrize("gate", ("g", "raw"))
@pytest.mark.parametrize("pack", tuple(PACKS))
def test_gpu_kcp_dav_runs_only_where_its_dakq_has_a_reader(
    device, monkeypatch, record_property, pack, gate, l2n, H, path
) -> None:
    """Every draw: the record as the plan decides; ``"auto"`` (twice: repeats, ``auto``), ``kcp_dav="always"``
    (``always``) and, where ``kcp_dav`` is skipped, the run with the NaN-filled dAqk (``poisoned``: no NaN reaches an
    output) judged; where ``kcp_dav`` runs the NaN-filled dAqk reaches an output (the negative control)."""
    monkeypatch.setattr(kp, "_TEST_KNOBS_ENABLED", True)
    _, _, ctx = _ctx(pack, device)

    def draw(j, s):
        ops, glob = _operands(pack, H, gate, l2n, device, s)
        auto, rec, needed = _run(ctx, ops, gate, l2n, path, {"kcp_dav": "auto"})
        assert needed is not None, "the backward ran no dav stage"
        assert rec.get("kcp_dav") == ("launched" if needed else "skipped"), (
            rec.get("kcp_dav"),
            needed,
        )
        if path == "handoff":
            assert rec.get("checkpoint_handoff") == "bound", rec.get(
                "checkpoint_handoff"
            )
        if path == "handoff_rows":
            assert rec.get("saved_rows") == "wy", rec.get("saved_rows")
        if path.startswith("rows_"):
            assert rec.get("saved_rows") == path[5:], rec.get("saved_rows")
        again, _rec2, _ = _run(ctx, ops, gate, l2n, path, {"kcp_dav": "auto"})
        always, rec_a, needed_a = _run(ctx, ops, gate, l2n, path, {"kcp_dav": "always"})
        assert rec_a.get("kcp_dav") == "launched" and needed_a == needed
        poisoned, _rec_p, _ = _run(
            ctx, ops, gate, l2n, path, {"kcp_dav": "auto"}, poison=True
        )
        arms = {
            "auto": [_global(pack, auto, device), _global(pack, again, device)],
            "always": _global(pack, always, device),
        }
        if needed:
            # the negative control: kcp_dav's dAqk is the fused gradient's input here
            assert _nonfinite(poisoned), (
                "the NaN-filled dAqk reached no output where kcp_dav runs"
            )
        else:
            arms["poisoned"] = _global(pack, poisoned, device)
        return arms, glob

    judge_arms(
        record_property,
        draw,
        seed=SEED,
        scale=cake_kcp.SCALE,
        l2n=l2n,
        lower_bound=LOWER_BOUND,
    )


@pytest.mark.parametrize("path", ("saved", "replay"))
@pytest.mark.parametrize("H", FORM_HEADS)
@pytest.mark.parametrize("l2n", (False, True), ids=("plain", "l2n"))
@pytest.mark.parametrize("gate", ("g", "raw"))
def test_gpu_adjoint_walk_forms_judged(
    device, monkeypatch, record_property, gate, l2n, H, path
) -> None:
    """``walk_form`` ``"auto"`` / ``"full"`` / ``"split"`` across the K2 offset's head rule (63 / 64 heads), each
    judged (arm per form): every run launches both K2 walks (``walk_forms``), a forced run in the forced form only
    (every draw); the full-width offset record the registry selects at ``H`` is the one ``WALK_HEADS_RULE`` names
    (``kcp_dhu_offset`` at 63 heads, ``kcp_dhu_offset_overlap`` at 64)."""
    monkeypatch.setattr(kp, "_TEST_KNOBS_ENABLED", True)
    want = "kcp_dhu_offset" if H <= 63 else "kcp_dhu_offset_overlap"
    assert cake_kcp_jit.walk_heads_record("kcp_dhu_offset", H) == want
    assert (
        cake_kcp_jit.select_record("kcp_dhu_offset", form="full", num_heads=H) == want
    )
    pack = "one4k"
    _, _, ctx = _ctx(pack, device)

    def draw(j, s):
        ops, glob = _operands(pack, H, gate, l2n, device, s)
        arms = {}
        for form in ("auto", "full", "split"):
            res, rec, _ = _run(ctx, ops, gate, l2n, path, {"walk_form": form})
            launched = [tuple(x)[:2] for x in rec.get("walk_forms") or ()]
            k2 = [(name, f) for name, f in launched if name in K2_PROGRAMS]
            assert {name for name, _ in k2} == set(K2_PROGRAMS), (form, launched)
            if form != "auto":
                assert {f for _, f in k2} == {form}, (form, k2)
            arms[form] = _global(pack, res, device)
        return arms, glob

    judge_arms(
        record_property,
        draw,
        seed=SEED,
        scale=cake_kcp.SCALE,
        l2n=l2n,
        lower_bound=LOWER_BOUND,
    )
