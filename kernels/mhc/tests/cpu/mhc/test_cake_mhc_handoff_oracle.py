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

The test oracle of the pre handoff (``tests/cake/test_helpers/cake_mhc_naive.py``) on the CPU (no device):

* without ``pre_in`` the oracle is the standard function with its op sequence unchanged: every output equals, bit
  for bit, a verbatim copy of the standard block (float64 and the BF16 tier, one block and several);
* with ``pre_in`` it matches the requester's FP64 reference of the pre handoff (the functions of its reference
  script: coefficients in the released association ``(u fn^T) r``, the collapse with the incoming ``pre``, the
  combine; autograd of ``(U, Y, pre_next)`` for ``(dU, dY, dpre_next)``) on every output and gradient, to the FP64
  rounding of the two associations;
* the handoff's exact cases on the oracle itself: ``dpre_next`` omitted is ``dpre_next = 0``, ``dU = 0`` gives
  ``dpre_in`` exactly zero, ``dY = 0`` and ``dU = 0`` leave the coefficient VJP driven by ``dpre_next`` alone, and a
  one-hot ``pre_in`` collapses to the stream exactly.
"""

from __future__ import annotations

import pytest
import torch

from tests.cake.test_helpers import cake_mhc_naive as naive

F64, BF16 = torch.float64, torch.bfloat16
STREAMS, MAPPING_ROWS = 4, 24
RMS_EPS, HC_EPS, ITERS = 1e-20, 1e-6, 20


def _standard_block(X, fn, scale, base, B, dY, dU, *, dtype, rms_eps, hc_eps, iters) -> dict:
    """The standard block of the oracle as it stood before the pre handoff (verbatim op sequence)."""
    T, S, H = X.shape
    with torch.enable_grad():
        Xd = X.detach().to(dtype).requires_grad_(True)
        fnd = fn.detach().to(dtype).requires_grad_(True)
        scaled = scale.detach().to(dtype).requires_grad_(True)
        based = base.detach().to(dtype).requires_grad_(True)
        Bd = B.detach().to(dtype).requires_grad_(True)
        u = Xd.reshape(T, S * H)
        r = torch.rsqrt(u.square().mean(dim=-1, keepdim=True) + rms_eps)
        z = (u * r) @ fnd.T
        a_pre = z[:, 0:4] * scaled[0] + based[0:4]
        a_post = z[:, 4:8] * scaled[1] + based[4:8]
        a_logit = (z[:, 8:] * scaled[2] + based[8:]).reshape(T, STREAMS, STREAMS)
        pre = torch.sigmoid(a_pre) + hc_eps
        post = 2 * torch.sigmoid(a_post)
        C = naive._sinkhorn(a_logit, hc_eps, iters)
        U_pre = (pre.unsqueeze(-1) * Xd).sum(dim=1)
        Y_pre = post.unsqueeze(-1) * Bd.unsqueeze(1) + torch.einsum("tij,tid->tjd", C, Xd)
        dUd = dU.detach().to(dtype)
        dYd = dY.detach().to(dtype)
        dX, dfn, dscale, dbase, dB = torch.autograd.grad(
            [U_pre, Y_pre], [Xd, fnd, scaled, based, Bd], grad_outputs=[dUd, dYd]
        )
    with torch.no_grad():
        Xd, Bd = Xd.detach(), Bd.detach()
        return {
            "U": U_pre.detach(), "Y": Y_pre.detach(), "dB": dB, "dX": dX,
            "r": r.detach().reshape(T), "z": z.detach(), "pre": pre.detach(), "post": post.detach(), "C": C.detach(),
            "dpost": torch.einsum("tjd,td->tj", dYd, Bd), "dC": torch.einsum("tid,tjd->tij", Xd, dYd),
            "dfn": dfn, "dscale": dscale, "dbase": dbase,
        }  # fmt: skip


def _inputs(T: int, H: int, seed: int, *, handoff: bool = True) -> dict:
    g = torch.Generator().manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, generator=g, dtype=torch.float32)

    i = {
        "X": randn(T, STREAMS, H).to(BF16),
        "fn": (randn(MAPPING_ROWS, STREAMS * H) * 0.3).contiguous(),
        "scale": 1.0 + 0.2 * randn(3),
        "base": 0.5 * randn(MAPPING_ROWS),
        "B": randn(T, H).to(BF16),
        "dY": randn(T, STREAMS, H).to(BF16),
        "dU": randn(T, H).to(BF16),
        "rms_eps": RMS_EPS,
        "hc_eps": HC_EPS,
        "sinkhorn_iters": ITERS,
    }
    if handoff:
        i["pre_in"] = torch.softmax(randn(T, STREAMS), dim=-1)
        i["dpre_next"] = randn(T, STREAMS)
    return i


def _issue_reference(i: dict) -> dict:
    """The requester's FP64 reference of one pre-handoff sublayer (its reference script's ``coefficients``,
    ``collapse`` and ``combine`` at ``rms_eps`` 1e-20, ``hc_eps`` 1e-6, 20 iterations), with autograd of
    ``(U, Y, pre_next)`` for ``(dU, dY, dpre_next)`` with respect to ``X``, ``pre_in``, the parameters and ``B``."""
    leaves = [i[k].double().requires_grad_(True) for k in ("X", "pre_in", "fn", "scale", "base", "B")]
    X, pre_in, fn, scale, base, B = leaves
    u = X.flatten(1)
    r = torch.rsqrt(u.square().mean(-1, keepdim=True) + RMS_EPS)
    z = (u @ fn.T) * r  # the released association: mapping first, then the RMS scaling
    pre = torch.sigmoid(z[:, :4] * scale[0] + base[:4]) + HC_EPS
    post = 2 * torch.sigmoid(z[:, 4:8] * scale[1] + base[4:8])
    C = (z[:, 8:] * scale[2] + base[8:]).view(-1, 4, 4).softmax(-1) + HC_EPS
    C = C / (C.sum(-2, keepdim=True) + HC_EPS)
    for _ in range(ITERS - 1):
        C = C / (C.sum(-1, keepdim=True) + HC_EPS)
        C = C / (C.sum(-2, keepdim=True) + HC_EPS)
    U = (pre_in.unsqueeze(-1) * X).sum(1)
    Y = post.unsqueeze(-1) * B.unsqueeze(1) + torch.einsum("tij,tid->tjd", C, X)
    grads = torch.autograd.grad(
        (U, Y, pre), leaves, (i["dU"].double(), i["dY"].double(), i["dpre_next"].double())
    )
    out = dict(zip(("dX", "dpre_in", "dfn", "dscale", "dbase", "dB"), grads, strict=True))
    out.update(U=U.detach(), Y=Y.detach(), pre_next=pre.detach(), post=post.detach(), C=C.detach())
    return out


@pytest.mark.cpu_only
@pytest.mark.parametrize("dtype", (F64, BF16), ids=("fp64", "bf16"))
@pytest.mark.parametrize(("T", "H", "block_tokens"), ((3, 8, None), (7, 16, 3), (1, 64, None)))
def test_default_oracle_is_the_standard_op_sequence(dtype, T: int, H: int, block_tokens) -> None:
    """Without ``pre_in`` every output of the oracle equals the standard block's bit for bit (also over blocks)."""
    i = _inputs(T, H, seed=7 + T, handoff=False)
    got = naive.oracle(
        i["X"], i["fn"], i["scale"], i["base"], i["B"], i["dY"], i["dU"], dtype=dtype,
        rms_eps=RMS_EPS, hc_eps=HC_EPS, iters=ITERS, block_tokens=block_tokens,
    )  # fmt: skip
    bt = T if not block_tokens else block_tokens
    blocks = [
        _standard_block(
            i["X"][lo : lo + bt], i["fn"], i["scale"], i["base"], i["B"][lo : lo + bt], i["dY"][lo : lo + bt],
            i["dU"][lo : lo + bt], dtype=dtype, rms_eps=RMS_EPS, hc_eps=HC_EPS, iters=ITERS,
        )  # fmt: skip
        for lo in range(0, T, bt)
    ]
    assert set(got) == set(naive.OUTPUTS)
    for name in naive.OUTPUTS:
        if name in naive.PARAMETERS:
            want = blocks[0][name].clone()
            for b in blocks[1:]:
                want += b[name]
        else:
            want = torch.cat([b[name] for b in blocks], dim=0)
        assert torch.equal(got[name], want), name


@pytest.mark.cpu_only
@pytest.mark.parametrize(("T", "H"), ((3, 8), (5, 32)))
def test_handoff_oracle_matches_the_issue_reference(T: int, H: int) -> None:
    """With ``pre_in`` the oracle in float64 is the requester's reference of the pre handoff: ``U``, ``Y``,
    ``pre_next``, ``dX``, ``dpre_in``, ``dfn``, ``dscale``, ``dbase`` and ``dB`` agree to the FP64 rounding of the two
    associations of the mapping and the RMS scale (relative 1e-12)."""
    for seed in range(4):
        i = _inputs(T, H, seed=100 * seed + T)
        got = naive.case_oracle(i)
        want = _issue_reference(i)
        assert set(naive.HANDOFF_OUTPUTS) <= set(got)
        for name, value in want.items():
            assert got[name].dtype == F64 and got[name].shape == value.shape, name
            scale = value.abs().max().clamp_min(1e-300)
            err = ((got[name] - value).abs().max() / scale).item()
            assert err <= 1e-12, (name, seed, err)


@pytest.mark.cpu_only
@pytest.mark.parametrize("dtype", (F64, BF16), ids=("fp64", "bf16"))
def test_handoff_oracle_exact_cases(dtype) -> None:
    """``dpre_next`` omitted is ``dpre_next = 0``; ``dU = 0`` gives ``dpre_in`` exactly zero and ``dX`` without the
    collapse term; with ``dY = 0`` and ``dU = 0`` the parameter gradients come from ``dpre_next`` alone (zero when it is
    zero too); a one-hot ``pre_in`` collapses to the stream exactly."""
    T, H = 4, 16
    i = _inputs(T, H, seed=31)
    zeros = {k: torch.zeros_like(i[k]) for k in ("dpre_next", "dU", "dY")}
    run = lambda **kw: naive.case_oracle({**i, **kw}, dtype=dtype)  # noqa: E731
    omitted = naive.case_oracle({k: v for k, v in i.items() if k != "dpre_next"}, dtype=dtype)
    zero_next = run(dpre_next=zeros["dpre_next"])
    for name in naive.HANDOFF_OUTPUTS:
        assert torch.equal(omitted[name], zero_next[name]), name
    no_dU = run(dU=zeros["dU"])
    assert torch.count_nonzero(no_dU["dpre_in"]) == 0
    assert torch.count_nonzero(no_dU["dX"]) > 0
    only_next = run(dU=zeros["dU"], dY=zeros["dY"])
    assert all(torch.count_nonzero(only_next[k]) > 0 for k in ("dfn", "dscale", "dbase", "dX"))
    # dpre_next reaches the pre lanes only: rows 0..3 of dfn and dbase, entry 0 of dscale
    assert torch.count_nonzero(only_next["dfn"][4:]) == 0 and torch.count_nonzero(only_next["dbase"][4:]) == 0
    assert torch.count_nonzero(only_next["dscale"][1:]) == 0
    nothing = run(dU=zeros["dU"], dY=zeros["dY"], dpre_next=zeros["dpre_next"])
    for name in ("dX", "dpre_in", "dfn", "dscale", "dbase"):
        assert torch.count_nonzero(nothing[name]) == 0, name
    one_hot = torch.zeros(T, STREAMS)
    one_hot[:, 0] = 1.0
    hot = run(pre_in=one_hot)
    assert torch.equal(hot["U"].to(BF16), i["X"][:, 0]), "a one-hot pre_in collapses to stream 0"
