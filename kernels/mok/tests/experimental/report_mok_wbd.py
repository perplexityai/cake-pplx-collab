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

Measurements of the router weight before the down projection (``router_weight_position="before_down"``) on the
requester's inputs: a report, not a test (no pass or fail; pytest does not collect it).  The accuracy judgement is
``test_mok_accuracy.py`` (the naive tier).

Run from the directory that holds ``tests`` on an SM100a or SM103a GPU::

    python -m tests.experimental.report_mok_wbd --out report.txt

Per case (hidden 5120, intermediate 2304, top-6, 4096 tokens; 6 and 48 local experts, the production layouts' (64, 6)
and (8, 48); :func:`tests.experimental._mok_reference.production_inputs`, several seeds) the kernels run once:
forward + backward of the low-level launchers with FP32 weight-gradient accumulators from zero, ``y`` and ``dx`` summed
by the unweighted epilogue.  Three evaluations of the same layer on the same inputs:

* the stage-rounded reference of the issue: float64 with one correctly rounded BF16 rounding (:func:`round_bf16`) at
  ``g``, ``u``, ``h = w * a``, ``dh``, ``d_gate`` and ``d_up`` of every route and of the shared expert, every other
  value float64;
* the float64 oracle and the naive PyTorch BF16 implementation of :mod:`tests.experimental._mok_naive`.

For ``y``, ``dx``, ``d_weights``, the routed ``dw_gate`` / ``dw_up`` / ``dw_down`` and the shared ``ds_gate`` /
``ds_up`` / ``ds_down``: the largest absolute difference of the kernel and of the naive implementation from the
stage-rounded reference and from the oracle, and the issue's elementwise view ``|v - ref| / (0.01 + 0.01 |ref|)`` (its
largest value and the fraction of elements above 1) of the kernel against both.  Input facts: the fraction of routed
gates above the limit and of ups beyond it, the count of gates and ups exactly at it, the count of zero weights, the RMS
of ``x``.  Placement: the kernel's routed hidden of the resident macrobatch against ``w * act(g, u)`` correctly rounded
from float64 at the kernel's own saved gate and up (fraction of elements that differ, largest difference in BF16 ulp).
Every line starts with ``REPORT``.
"""

from __future__ import annotations

import argparse
import sys

import torch
import torch.nn.functional as F

from tests.cake.test_helpers.cake_package import cake_package
from tests.experimental import _mok_naive as naive
from tests.experimental._mok_reference import production_inputs, single_rank_schedule

kernels = cake_package("cake_mok_bf16", "_kernels")

HIDDEN, INTERMEDIATE, TOPK, TOKENS, LIMIT = 5120, 2304, 6, 4096, 10.0
MACRO, MINI, COMM = 32768, 4096, 8
OUTPUTS = ("y", "dx", "d_weights", "dw_gate", "dw_up", "dw_down", "ds_gate", "ds_up", "ds_down")


def round_bf16(v):
    """Float64 ``v`` rounded once to the nearest BF16 value (ties to even), as float64 (torch's float64 -> bfloat16
    cast rounds through float32 first)."""
    _, exponent = torch.frexp(v)
    ulp = torch.ldexp(torch.ones_like(v), torch.clamp(exponent - 1, min=-126) - 7)
    return torch.round(v / ulp) * ulp


def bf16_ulp(v):
    _, exponent = torch.frexp(v)
    return torch.ldexp(torch.ones_like(v), torch.clamp(exponent - 1, min=-126) - 7)


def _act(g, u):
    return F.silu(g.clamp(max=LIMIT)) * u.clamp(-LIMIT, LIMIT)


def _act_backward(g, u, da):
    """``(d_gate, d_up)`` of :func:`_act` with the inclusive masks (the gradient passes at ``g = L``, ``u = +-L``)."""
    gc, uc = g.clamp(max=LIMIT), u.clamp(-LIMIT, LIMIT)
    sig = torch.sigmoid(gc)
    dg = da * uc * sig * (1 + gc * (1 - sig))
    du = da * gc * sig
    zero = torch.zeros((), dtype=g.dtype, device=g.device)
    return torch.where(g <= LIMIT, dg, zero), torch.where((u >= -LIMIT) & (u <= LIMIT), du, zero)


def stage_reference(p):
    """The issue's stage-rounded float64 evaluation of the layer (module docstring)."""
    x, dy, w = p["x"].double(), p["dy"].double(), p["weights"].double()
    sg, su, sd = (v.double() for v in p["shared"])
    wg, wu, wd = (v.double() for v in p["routed"])
    ids = p["ids"]
    g_s, u_s = round_bf16(x @ sg.T), round_bf16(x @ su.T)
    h_s = round_bf16(_act(g_s, u_s))
    dh_s = round_bf16(dy @ sd)
    dg_s, du_s = (round_bf16(v) for v in _act_backward(g_s, u_s, dh_s))
    y = h_s @ sd.T
    dx = dg_s @ sg + du_s @ su
    out = dict(ds_gate=dg_s.T @ x, ds_up=du_s.T @ x, ds_down=dy.T @ h_s)
    dw = [torch.zeros_like(v) for v in (wg, wu, wd)]
    d_weights = torch.zeros_like(w)
    for e in range(wg.shape[0]):
        tok, slot = torch.nonzero(ids == e, as_tuple=True)
        if tok.numel() == 0:
            continue
        xe, dye, we = x[tok], dy[tok], w[tok, slot][:, None]
        g, u = round_bf16(xe @ wg[e].T), round_bf16(xe @ wu[e].T)
        a = _act(g, u)
        h = round_bf16(we * a)
        y.index_add_(0, tok, h @ wd[e].T)
        dh = round_bf16(dye @ wd[e])
        d_weights[tok, slot] = (dh * a).sum(-1)
        dg, du = (round_bf16(v) for v in _act_backward(g, u, we * dh))
        dx.index_add_(0, tok, dg @ wg[e] + du @ wu[e])
        dw[0][e], dw[1][e], dw[2][e] = dg.T @ xe, du.T @ xe, dye.T @ h
    out.update(y=y, dx=dx, d_weights=d_weights, dw_gate=dw[0], dw_up=dw[1], dw_down=dw[2])
    return out


def _segments(s):
    out, offset = [], 0
    for expert, count in enumerate(s["count_values"]):
        out.append(naive.Segment(expert, s["schedule"][offset : offset + count], offset))
        offset += count
    return out


def _tier(p, s, dtype):
    """The oracle or the naive implementation by output name."""
    r = naive.layer(
        p["x"], p["dy"], p["weights"], p["shared"], p["routed"], _segments(s), LIMIT, dtype, weight_before_down=True,
    )  # fmt: skip
    names = ("dw_gate", "dw_up", "dw_down", "ds_gate", "ds_up", "ds_down")
    out = dict(y=r["y"], dx=r["dx"], d_weights=r["d_scores"])
    out.update(zip(names, (*r["dw"][:3], *(v[0] for v in r["dw"][3:])), strict=True))
    return out


def _kernel(p, s):
    """The kernels' outputs by name, and copies of the forward context's routed gate, up and hidden of the resident
    macrobatch taken before the backward (which rewrites those ring rows while it replays later macrobatches)."""
    forward = kernels.MoKForward(True, weight_before_down=True)
    backward = kernels.MoKBackward(True, True, weight_before_down=True)
    epilogues = kernels.MoKEpilogues(TOPK, weighted=False)
    x, dy, scores, shared, routed = p["x"], p["dy"], p["weights"], p["shared"], p["routed"]
    options = dict(device=x.device, dtype=torch.bfloat16)
    schedule = (s["peers"], s["schedule"], s["num_tokens"], s["counts"])
    combine = torch.zeros(TOKENS * TOPK, HIDDEN, **options)
    dx_peer = torch.zeros_like(combine)
    ds_peer = torch.zeros_like(scores)
    accumulators = [torch.zeros(w.shape, device=x.device) for w in (*shared, *routed)]
    weights = (shared[0], routed[0], shared[1], routed[1], shared[2], routed[2])
    context = forward(
        x, [x.data_ptr()], combine, [combine.data_ptr()], *weights, *schedule,
        TOPK, LIMIT, COMM, MACRO, MINI, weight_ptrs=[scores.data_ptr()],
    )  # fmt: skip
    torch.cuda.synchronize()
    rows = min(s["actual"], MACRO)
    saved = tuple(context[index][:rows].clone() for index in (2, 4, 6))
    result = backward(
        dy, [dy.data_ptr()], dx_peer, [dx_peer.data_ptr()], scores, [scores.data_ptr()], ds_peer,
        [ds_peer.data_ptr()], *weights, *context[:7], x, [x.data_ptr()], *schedule,
        TOPK, LIMIT, COMM, MACRO, MINI, weight_grad_accumulators=accumulators,
    )  # fmt: skip
    y = epilogues.backward(context[7], combine)
    dx = epilogues.backward(result[0], dx_peer)
    torch.cuda.synchronize()
    names = ("ds_gate", "ds_up", "ds_down", "dw_gate", "dw_up", "dw_down")
    out = dict(y=y, dx=dx, d_weights=ds_peer, **dict(zip(names, accumulators, strict=True)))
    return out, saved


def _view(actual, ref):
    """Largest ``|v - ref|``, and of the issue's ``|v - ref| / (0.01 + 0.01 |ref|)`` the largest value and the fraction
    of elements above 1."""
    err = (actual.double() - ref.double()).abs()
    ratio = err / (0.01 + 0.01 * ref.double().abs())
    return err.max().item(), ratio.max().item(), (ratio > 1).double().mean().item()


def report(experts, seed, emit):
    p = production_inputs(TOKENS, HIDDEN, INTERMEDIATE, experts, TOPK, LIMIT, seed=seed)
    s = single_rank_schedule(p["ids"], experts)
    case = f"case=local{experts} seed={seed} routed_rows={s['actual']} macro={MACRO}"
    kernel, saved = _kernel(p, s)
    stage = stage_reference(p)
    oracle, tier = _tier(p, s, naive.ORACLE), _tier(p, s, naive.NAIVE)
    for name in OUTPUTS:
        k_stage = _view(kernel[name], stage[name])
        k_oracle = _view(kernel[name], oracle[name])
        n_stage, n_oracle = _view(tier[name], stage[name]), _view(tier[name], oracle[name])
        emit(
            f"REPORT {case} output={name} max_abs_kernel_vs_stage={k_stage[0]:.6e} "
            f"max_abs_naive_vs_stage={n_stage[0]:.6e} max_abs_kernel_vs_oracle={k_oracle[0]:.6e} "
            f"max_abs_naive_vs_oracle={n_oracle[0]:.6e} issue_ratio_kernel_vs_stage={k_stage[1]:.4f} "
            f"issue_frac_above_1_kernel_vs_stage={k_stage[2]:.6f} issue_ratio_kernel_vs_oracle={k_oracle[1]:.4f} "
            f"issue_frac_above_1_kernel_vs_oracle={k_oracle[2]:.6f}"
        )
    # Input facts, from the routed gate and up of every route (float64 products of the BF16 inputs).
    x, ids = p["x"].double(), p["ids"]
    above = beyond = at_limit = total = 0
    for e in range(experts):
        tok = torch.nonzero(ids == e, as_tuple=True)[0]
        g = round_bf16(x[tok] @ p["routed"][0][e].double().T)
        u = round_bf16(x[tok] @ p["routed"][1][e].double().T)
        above += (g > LIMIT).sum().item()
        beyond += (u.abs() > LIMIT).sum().item()
        at_limit += (g == LIMIT).sum().item() + (u.abs() == LIMIT).sum().item()
        total += g.numel()
    emit(
        f"REPORT {case} inputs gate_above_limit={above / total:.6f} up_beyond_limit={beyond / total:.6f} "
        f"gate_or_up_at_limit={at_limit} zero_weights={(p['weights'] == 0).sum().item()} "
        f"x_rms={x.square().mean(-1).sqrt().mean().item():.6f} pinned={p['pinned']}"
    )
    # Placement: the kernel's routed hidden of the resident macrobatch against BF16(w * act(g, u)) correctly rounded.
    rows = min(s["actual"], MACRO)
    slots = s["schedule"][:rows].long()
    valid = slots >= 0
    g, u, h = (value[valid].double() for value in saved)
    w = p["weights"].flatten()[slots[valid]].double()[:, None]
    exact = w * _act(g, u)
    nearest = round_bf16(exact)
    ulps = ((h - exact).abs() / bf16_ulp(exact)).max().item()
    emit(
        f"REPORT {case} placement rows={int(valid.sum())} "
        f"differs_from_nearest={(h != nearest).double().mean().item():.6f} max_ulp={ulps:.4f}"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Measurements of the router weight before the down projection (REPORT lines)"
    )
    parser.add_argument("--out", help="also write the REPORT lines to this file")
    parser.add_argument("--experts", type=int, nargs="+", default=[6, 48])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    args = parser.parse_args(argv)
    if not torch.cuda.is_available():
        sys.exit("A CUDA device is required")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    lines = []

    def emit(line):
        print(line, flush=True)
        lines.append(line)

    device = torch.cuda.get_device_name(), torch.cuda.get_device_capability()
    emit(f"REPORT device={device[0]} capability={device[1]} torch={torch.__version__}")
    for experts in args.experts:
        for seed in args.seeds:
            report(experts, seed, emit)
    if args.out:
        with open(args.out, "w") as handle:
            handle.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
