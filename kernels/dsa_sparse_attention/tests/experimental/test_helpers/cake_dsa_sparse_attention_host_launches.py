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

The launches of the RoPE training entries on the host: no device, no kernel.

:class:`HostLaunches` runs the package's own entries (``forward``, ``backward``, ``prepare_dsa_train`` runners) on
CPU tensors with a few stand-ins -- the architecture, the device index, the allocation site (CPU), the stage
resolver and the FFI stream context -- and records every stage launch: its stage, grid and bound values.  Each
launch is also *simulated* on its operands by a stand-in of its kernel with the kernel's addressing and an easily
predicted per-head result, so the outputs of a call show which launch wrote each query head:

- a forward launch writes ``out = q_latent``, ``o_lo = -q_latent`` (unless it is a ``with_o_lo=False`` twin) and
  ``lse = q_rope[..., 0] + TAG * n`` (``n``: the launch's ordinal among the forward launches of the harness);
- the delta launch writes, for row ``r`` of ``num_rows`` (token ``r // 64``, head ``r % 64`` of the virtual strides),
  ``delta[r] = dout[r, 0] + 2 out[r, 0] (+ 4 o_lo[r, 0])`` read at the kernel's own offsets (``dout_token_stride``,
  ``dout_head_stride``, ``dout_offset``; ``out`` / ``o_lo`` flat at ``r * 512``);
- a main launch reads ``lse`` / ``delta`` as ``[T, 64]`` (``token * 64 + lane``), and every lane ``j`` whose ``lse`` is
  finite adds ``own = b + j + 1`` -- ``b`` the first query head of its window (found by content: the launch's
  ``q_latent`` operand equals the caller's ``q_latent[:, b:b + 64]``) -- to its ``dq`` (``dq_mode`` 0: write; 1 / 2 /
  3: the FP32 partial, then ``dq = partial + own``), and adds ``(1, own)`` per such (token, lane) into the first two
  accumulator elements; a lane whose ``lse`` is ``+inf`` (a pad head) or ``-inf`` (a row without a valid slot) adds
  nothing and writes ``dq = 0``, as the kernel's ``p = exp2(s * c - lse)`` and its validity bits give;
- the cast copies (or adds, through the destination map) the first two accumulator elements.

:func:`check_forward` and :func:`check_backward` hold a call's outputs and launches to the head-window contract of
the package: one window for 64 heads, ``[H - 64, H)`` then ``[0, 64)`` for 96 and 128 heads (the heads both windows
cover come from the later one), one delta launch over ``T * H`` rows, the window statistics (``lse = +inf``,
``delta = 0`` on the heads of the first window that the second covers), one shared accumulator to which every query
head adds once per key pass, and one cast (none on a direct binding).

:func:`capture_h64_cases` records the launches of :data:`H64_CASES` (64 heads) in the form of the golden file
``cake_dsa_sparse_attention_h64_launches.json``: per case, every stage launch (stage, grid and every argument its
kernel receives: the scalar values, each tensor operand's role, shape, strides, dtype and offset) and every allocation
of the backend.  The golden holds the
launches of the 64-head entries as they were before the head-window plans; ``python -m
tests.experimental.test_helpers.cake_dsa_sparse_attention_host_launches --write PATH`` writes the capture of the
installed package.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import pytest
import torch

#: The forward's ``lse`` offset per forward-launch ordinal.
TAG = 1000.0
#: Query heads per window (the shipped kernels' head count).
WINDOW = 64
D_LATENT = 512
D_ROPE = 64
D_QK = D_LATENT + D_ROPE
GOLDEN = Path(__file__).with_name("cake_dsa_sparse_attention_h64_launches.json")


def family(stage: str) -> str:
    """The kernel family of a RoPE stage name: ``fwd``, ``delta``, ``main``, ``main_pass``, ``compact``, ``cast``."""
    if stage.startswith("fwd"):
        return "fwd"
    if stage.startswith("bwd_delta"):
        return "delta"
    if stage.startswith("bwd_main_pass"):
        return "main_pass"
    if stage.startswith("bwd_main"):
        return "main"
    if stage.startswith("bwd_compact"):
        return "compact"
    if stage.startswith("bwd_cast"):
        return "cast"
    raise ValueError(f"unknown RoPE stage {stage!r}")


@dataclass
class Launch:
    """One recorded stage launch."""

    stage: str
    grid: tuple
    values: dict = field(repr=False)
    #: first query head of the launch's head operands (forward and main launches; ``None`` otherwise)
    window: Optional[int] = None
    #: ordinal among the forward launches of the harness (forward launches only)
    ordinal: Optional[int] = None
    #: clones taken at launch: ``lse`` / ``delta`` as ``[T, 64]`` (main launches), ``acc`` (cast launches)
    snap: dict = field(default_factory=dict, repr=False)


def _flat(t: torch.Tensor, numel: int, offset: int = 0) -> torch.Tensor:
    """``numel`` elements of ``t``'s storage from its first element plus ``offset`` (the kernel's pointer view)."""
    return t.as_strided((numel,), (1,), t.storage_offset() + offset)


class HostLaunches:
    """The stand-ins of one test (see the module docstring); ``launches`` / ``allocs`` record the calls.

    ``backend`` / ``jit`` / ``launch`` are the package's ``cake_backend`` / ``cake_jit`` / ``cake_launch``; ``arch``
    the architecture the planners see; ``token_chunk`` (optional) shrinks the key-range-pass token chunk of the
    record's policy to that many tokens (``token_chunk_multiple`` and the scratch budget of exactly that chunk at
    every top-k width up to ``chunk_topk``).  ``simulate=False`` records without simulating."""

    def __init__(
        self,
        monkeypatch,
        backend,
        jit,
        launch,
        arch: str = "sm_100a",
        *,
        token_chunk: Optional[int] = None,
        chunk_topk: int = 64,
        simulate: bool = True,
    ):
        self.backend, self.jit, self.launch = backend, jit, launch
        self.arch = arch
        self.simulate = simulate
        self.launches: list[Launch] = []
        self.allocs: list[tuple] = []
        self.q_latent: Optional[torch.Tensor] = None
        self._fwd_count = 0
        cpu = torch.device("cpu")
        real_alloc = backend._alloc
        real_entry_cls = backend._StageEntry

        def alloc(shape, dtype, device, *, zero: bool = False):
            shape = (int(shape),) if isinstance(shape, int) else tuple(int(s) for s in shape)
            self.allocs.append((list(shape), str(dtype), bool(zero)))
            return real_alloc(shape, dtype, cpu, zero=zero)

        def stage_entry(stage, arch, scalars, program=None, grid=None, **_):
            if grid is None:
                grid = launch.GRID[stage](
                    scalars["num_queries"], scalars["num_kv"], scalars["topk"]
                )
            return real_entry_cls(stage, self._launcher, stage, tuple(grid))

        monkeypatch.setattr(backend, "arch_for", lambda device=None: arch)
        monkeypatch.setattr(backend, "_device_index", lambda device=None: 0)
        monkeypatch.setattr(backend, "_alloc", alloc)
        monkeypatch.setattr(backend, "_stage_entry", stage_entry)
        monkeypatch.setattr(
            backend, "_ffi_stream_context", lambda index: contextlib.nullcontext()
        )
        monkeypatch.setattr(backend.BINDING_CACHE, "_entries", type(backend.BINDING_CACHE._entries)())
        monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
        monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
        if token_chunk is not None:
            real_record = jit.record

            def record(program=jit.PROGRAM):
                rec = real_record(program)
                if program != jit.PROGRAM:
                    return rec
                rec = copy.deepcopy(rec)
                policy = rec["key_pass_policy"]
                per_token = int(
                    policy.get(
                        "dq_partial_bytes_per_token", backend.DQ_PARTIAL_BYTES_PER_TOKEN
                    )
                ) + 4 * int(chunk_topk) + 4
                policy["token_chunk_multiple"] = int(token_chunk)
                policy["workspace_budget_bytes"] = int(token_chunk) * per_token
                return rec

            monkeypatch.setattr(jit, "record", record)

    # -- recording -------------------------------------------------------------------------------------------

    def bind(self, q_latent: torch.Tensor) -> None:
        """The caller's ``q_latent`` the windows are found in (by content)."""
        self.q_latent = q_latent

    def mark(self) -> int:
        """The index of the next launch (slice ``launches[mark:]`` for the launches of the calls after it)."""
        return len(self.launches)

    def _window(self, q: torch.Tensor) -> Optional[int]:
        src = self.q_latent
        if src is None or q.ndim != 3:
            return None
        heads, width = int(src.shape[1]), int(q.shape[1])
        for b in range(heads - width + 1):
            if torch.equal(q, src[:, b : b + width]):
                return b
        raise AssertionError(
            f"a launch's q_latent operand {tuple(q.shape)} is no {width}-head window of the caller's q_latent"
        )

    def _launcher(self, run, values, grid) -> None:
        stage = str(run)
        kind = family(stage)
        rec = Launch(stage, tuple(grid), dict(values))
        if kind in ("fwd", "main", "main_pass"):
            rec.window = self._window(values["q_latent"])
        if kind == "fwd":
            rec.ordinal = self._fwd_count
            self._fwd_count += 1
        self.launches.append(rec)
        if self.simulate:
            with torch.no_grad():
                getattr(self, f"_sim_{kind}")(rec, values)

    # -- kernel stand-ins ------------------------------------------------------------------------------------

    def _sim_fwd(self, rec: Launch, v: dict) -> None:
        q, qr = v["q_latent"], v["q_rope"]
        v["out"].copy_(q)
        if v["o_lo"].numel():
            v["o_lo"].copy_(-q)
        v["lse"].copy_(qr[..., 0].float() + TAG * rec.ordinal)

    def _sim_delta(self, rec: Launch, v: dict) -> None:
        rows = int(v["num_rows"])
        ts, hs, off = int(v["dout_token_stride"]), int(v["dout_head_stride"]), int(v["dout_offset"])
        r = torch.arange(rows)
        first = off + (r // WINDOW) * ts + (r % WINDOW) * hs
        dout = v["dout"]
        span = int(first.max()) + 1 if rows else 0
        g = _flat(dout, span)[first].float() if rows else torch.zeros(0)
        o = _flat(v["out"], rows * D_LATENT).view(rows, D_LATENT)[:, 0].float()
        d = g + 2 * o
        if v["o_lo"].numel():
            d = d + 4 * _flat(v["o_lo"], rows * D_LATENT).view(rows, D_LATENT)[:, 0].float()
        _flat(v["delta"], rows).copy_(d)

    def _tokens(self, v: dict, passes: bool) -> torch.Tensor:
        if passes:
            base, n = int(v["token_base"]), int(v["num_tokens"])
            assert int(v["token_step"]) == 1, "a key-pass launch walks its tokens ascending"
            return torch.arange(base, base + n)
        return torch.arange(int(v["num_queries"]))

    def _main(self, rec: Launch, v: dict, passes: bool) -> None:
        T = int(v["num_queries"])
        lse = _flat(v["lse"], T * WINDOW).view(T, WINDOW)
        delta = _flat(v["delta"], T * WINDOW).view(T, WINDOW)
        rec.snap.update(lse=lse.clone(), delta=delta.clone())
        tokens = self._tokens(v, passes)
        b = rec.window if rec.window is not None else 0
        lanes = torch.arange(WINDOW, dtype=torch.float32) + (b + 1)
        live = torch.isfinite(lse[tokens])
        own = torch.where(live, lanes[None, :], torch.zeros(()))
        mode = int(v["dq_mode"]) if passes else 0
        dql, dqr = v["dq_latent"], v["dq_rope"]
        if mode == 0:
            dql[tokens] = own[..., None].to(dql.dtype).expand(-1, -1, dql.shape[-1])
            dqr[tokens] = own[..., None].to(dqr.dtype).expand(-1, -1, dqr.shape[-1])
        else:
            # the FP32 dQ partial row of a token: 64 heads x 576 (the kernel's ``token * 36864``)
            dqp = v["dq_partial"]
            partial = _flat(dqp, dqp.numel()).view(-1, WINDOW * D_QK)[:, :WINDOW]
            k = tokens - int(v["token_base"])
            if mode == 1:
                partial[k] = own
            elif mode == 2:
                partial[k] = partial[k] + own
            else:
                total = partial[k] + own
                dql[tokens] = total[..., None].to(dql.dtype).expand(-1, -1, dql.shape[-1])
                dqr[tokens] = total[..., None].to(dqr.dtype).expand(-1, -1, dqr.shape[-1])
        count, total = float(live.sum()), float(own.sum())
        acc = v["dkv_f32"]
        start = 0
        if rec.stage.endswith("natural") and int(v["dkv_has_map"]):
            start = int(v["dkv_dst_map"].view(-1)[0]) * int(v["dkv_stride"])
        cell = _flat(acc, start + 2)[start:]
        cell += torch.tensor([count, total], dtype=cell.dtype)

    def _sim_main(self, rec: Launch, v: dict) -> None:
        self._main(rec, v, passes=False)

    def _sim_main_pass(self, rec: Launch, v: dict) -> None:
        self._main(rec, v, passes=True)

    def _sim_compact(self, rec: Launch, v: dict) -> None:
        pass

    def _sim_cast(self, rec: Launch, v: dict) -> None:
        src = _flat(v["src_latent"], 2).clone()
        rec.snap["acc"] = src
        if int(v["accumulate"]):
            row = int(v["dst_map"].view(-1)[0]) if int(v["has_dst_map"]) else 0
            start = row * int(v["dst_row_stride"])
            _flat(v["dst_packed"], start + 2)[start:].add_(src)
        elif int(v["out_f32"]):
            _flat(v["dst_latent_f32"], 2).copy_(src)
        else:
            _flat(v["dst_latent"], 2).copy_(src.to(torch.bfloat16))


# ---------------------------------------------------------------------------------------------------------------
# Inputs and the head-window contract
# ---------------------------------------------------------------------------------------------------------------


@dataclass
class HostInputs:
    """CPU inputs of one call: BF16 ``q_latent [T, H, 512]``, ``q_rope [T, H, 64]`` (``packed``: views of one
    ``[T, H, 576]`` tensor), ``kv_latent [S, 512]``, ``k_rope [S, 64]``, int32 ``indices [T, K]`` (row 1 without a
    valid slot when ``T > 1``, ``-1`` and ``>= S`` slots elsewhere), ``topk_length [T]``, ``dout [T, H, 512]``
    (``head_major``: the ``[T, H, 512]`` view of an ``[H, T, 512]`` buffer) and a backward's ``lse [T, H]`` (finite;
    ``-inf`` on the rows without a valid slot)."""

    q_latent: torch.Tensor
    q_rope: torch.Tensor
    kv_latent: torch.Tensor
    k_rope: torch.Tensor
    indices: torch.Tensor
    topk_length: torch.Tensor
    dout: torch.Tensor
    lse: torch.Tensor

    @property
    def heads(self) -> int:
        return int(self.q_latent.shape[1])

    @property
    def total_q(self) -> int:
        return int(self.q_latent.shape[0])

    @property
    def total_k(self) -> int:
        return int(self.kv_latent.shape[0])


def host_inputs(
    T: int = 8,
    S: int = 64,
    K: int = 16,
    heads: int = WINDOW,
    *,
    seed: int = 0,
    packed: bool = False,
    head_major: bool = False,
) -> HostInputs:
    gen = torch.Generator().manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, generator=gen).to(torch.bfloat16)

    if packed:
        q = randn(T, heads, D_QK)
        q_latent, q_rope = q[..., :D_LATENT], q[..., D_LATENT:]
    else:
        q_latent, q_rope = randn(T, heads, D_LATENT), randn(T, heads, D_ROPE)
    kv_latent, k_rope = randn(S, D_LATENT), randn(S, D_ROPE)
    indices = torch.randint(0, S, (T, K), generator=gen, dtype=torch.int32)
    indices[:, -1] = -1
    if K > 2:
        indices[:, -2] = S + 3
    if T > 1:
        indices[1] = -1
    topk_length = torch.full((T,), K, dtype=torch.int32)
    dout = randn(heads, T, D_LATENT).permute(1, 0, 2) if head_major else randn(T, heads, D_LATENT)
    lse = torch.randn(T, heads, generator=gen)
    if T > 1:
        lse[1] = -math.inf
    return HostInputs(q_latent, q_rope, kv_latent, k_rope, indices, topk_length, dout, lse)


def windows(heads: int) -> list[int]:
    """The first query head of every window of an ``H``-head call, in launch order: ``[0]`` for 64 heads, else
    ``[H - 64, 0]`` (the second window runs last)."""
    return [0] if heads == WINDOW else [heads - WINDOW, 0]


def check_forward(
    launches: list[Launch], inp: HostInputs, out, lse, o_lo, *, with_o_lo: bool
) -> None:
    """The forward launches of one call and its outputs under the head-window contract (module docstring)."""
    H = inp.heads
    fwd = [rec for rec in launches if family(rec.stage) == "fwd"]
    assert [rec.window for rec in fwd] == windows(H), [rec.window for rec in fwd]
    for rec in fwd:
        assert rec.stage.endswith("no_olo") == (not with_o_lo), rec.stage
        assert rec.values["q_latent"].shape[1] == WINDOW
    tag = torch.empty(H)
    for rec in fwd:  # launch order: a later window overwrites the heads both cover
        tag[rec.window : rec.window + WINDOW] = TAG * rec.ordinal
    assert tuple(out.shape) == (inp.total_q, H, D_LATENT) and out.dtype == torch.bfloat16
    assert tuple(lse.shape) == (inp.total_q, H) and lse.dtype == torch.float32
    assert torch.equal(out, inp.q_latent), "out: a query head holds another head's result"
    want = inp.q_rope[..., 0].float() + tag[None, :]
    assert torch.equal(lse, want), (
        "lse: a query head comes from the wrong window (heads both windows cover come from the later one)"
    )
    if with_o_lo:
        assert o_lo is not None and torch.equal(o_lo, -inp.q_latent)
    else:
        assert o_lo is None


def expected_delta(inp: HostInputs, out, o_lo) -> torch.Tensor:
    """``[T, H]``: the delta stand-in of every canonical (token, head)."""
    d = inp.dout[..., 0].float() + 2 * out[..., 0].float()
    if o_lo is not None:
        d = d + 4 * o_lo[..., 0].float()
    return d


def check_backward(
    launches: list[Launch],
    inp: HostInputs,
    out,
    o_lo,
    lse,
    dq_latent,
    dq_rope,
    *,
    passes: int = 1,
    chunks: int = 1,
    direct: bool = False,
) -> dict:
    """The backward launches of one call and its ``dq`` under the head-window contract (module docstring).  Returns
    ``{"count", "sum"}``: what the accumulator must hold (every live canonical head once per key pass)."""
    H, T = inp.heads, inp.total_q
    wins = windows(H)
    delta_l = [rec for rec in launches if family(rec.stage) == "delta"]
    assert len(delta_l) == 1, [rec.stage for rec in launches]
    dl = delta_l[0]
    assert int(dl.values["num_rows"]) == T * H
    assert dl.grid[0] * 8 >= T * H and dl.grid[1:] == (1, 1), dl.grid
    assert dl.stage.endswith("no_olo") == (o_lo is None), dl.stage
    canon_delta = _flat(dl.values["delta"], T * H).view(T, H)
    assert torch.equal(canon_delta, expected_delta(inp, out, o_lo)), (
        "delta: a (token, head) row read the wrong dout / out / o_lo row"
    )
    mains = [rec for rec in launches if family(rec.stage) in ("main", "main_pass")]
    want_order = [b for _ in range(chunks) for b in wins for _ in range(passes)]
    assert [rec.window for rec in mains] == want_order, [rec.window for rec in mains]
    assert all(family(rec.stage) == ("main" if passes == 1 else "main_pass") for rec in mains)
    assert all(rec.stage.endswith("natural") == direct for rec in mains)
    if passes > 1:
        compact = [rec for rec in launches if family(rec.stage) == "compact"]
        assert len(compact) == len(mains)
        modes = [int(rec.values["dq_mode"]) for rec in mains]
        per_window = [1] + [2] * (passes - 2) + [3]
        assert modes == per_window * (chunks * len(wins)), modes
    accs = {rec.values["dkv_f32"].untyped_storage().data_ptr() for rec in mains}
    assert len(accs) == 1, "the windows add into different accumulators"
    for rec in mains:
        b = rec.window
        lse_w, delta_w = rec.snap["lse"], rec.snap["delta"]
        for j in range(WINDOW):
            h = b + j
            pad = b > 0 and h < WINDOW
            if pad:
                assert bool((lse_w[:, j] == math.inf).all()), (
                    f"window {b}: pad lane {j} (head {h}) has lse != +inf"
                )
                assert bool((delta_w[:, j] == 0).all()), f"window {b}: pad lane {j} has delta != 0"
            else:
                assert torch.equal(lse_w[:, j], lse[:, h]), f"window {b}: lane {j} lse is not head {h}'s"
                assert torch.equal(delta_w[:, j], canon_delta[:, h]), (
                    f"window {b}: lane {j} delta is not head {h}'s"
                )
    casts = [rec for rec in launches if family(rec.stage) == "cast"]
    assert len(casts) == (0 if direct else 1)
    if casts:
        assert launches[-1] is casts[0], "the cast is the last launch"
    live = torch.isfinite(lse)
    head = torch.arange(H, dtype=torch.float32) + 1
    want_dq = torch.where(live, passes * head[None, :], torch.zeros(()))
    assert torch.equal(dq_latent, want_dq[..., None].to(torch.bfloat16).expand_as(dq_latent)), (
        "dq_latent: a query head holds the wrong window's gradient (the later window must overwrite the pad heads)"
    )
    assert torch.equal(dq_rope, want_dq[..., None].to(torch.bfloat16).expand_as(dq_rope))
    total = dict(count=float(passes * live.sum()), sum=float(passes * (live * head[None, :]).sum()))
    if casts:
        acc = casts[0].snap["acc"]
        assert acc.tolist() == [total["count"], total["sum"]], (
            f"accumulator {acc.tolist()} != {total}: a query head was added more or less than once per pass"
        )
    return total


# ---------------------------------------------------------------------------------------------------------------
# The 64-head capture (golden file)
# ---------------------------------------------------------------------------------------------------------------


def _describe(value: Any, roles: list[tuple[str, torch.Tensor]]) -> Any:
    if isinstance(value, torch.Tensor):
        ptr = value.untyped_storage().data_ptr()
        role = "scratch"
        for name, t in roles:
            if t is not None and t.untyped_storage().data_ptr() == ptr and t.numel():
                role = name
                break
        return [
            role,
            list(value.shape),
            list(value.stride()),
            str(value.dtype),
            int(value.storage_offset()),
        ]
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float, str)) or value is None:
        return value
    raise TypeError(f"launch value of type {type(value).__name__}")


def describe(launches: list[Launch], roles: list[tuple[str, torch.Tensor]], launch) -> list:
    """Each launch as ``[stage, grid, {name: value}]`` over the arguments the stage's kernel receives
    (``launch.STAGE_ARGS[stage]``, the generated launcher's positional order; tensors as ``[role, shape, strides,
    dtype, offset]``)."""
    return [
        [
            rec.stage,
            list(rec.grid),
            {name: _describe(rec.values[name], roles) for name in sorted(launch.STAGE_ARGS[rec.stage])},
        ]
        for rec in launches
    ]


def _case_inputs(case: dict) -> HostInputs:
    return host_inputs(
        case.get("T", 8),
        case.get("S", 64),
        case.get("K", 16),
        case.get("heads", WINDOW),
        seed=case.get("seed", 0),
        packed=case.get("packed", False),
        head_major=case.get("head_major", False),
    )


#: The 64-head cases of the golden file: the forward and the backward of the eager entries and a prepared runner,
#: both ``with_o_lo`` modes, caller and derived lengths, packed and head-major operands, forced key passes over
#: several token chunks, the direct and the cast-accumulate paths of ``dkv_acc`` (with a repeated destination map)
#: and ``dkv_fp32``; on the three architectures where their plans differ (token order, direct rule).
H64_CASES: dict[str, dict] = {
    "fwd_derived": dict(entry="forward", lengths=False),
    "fwd_packed_lengths_no_olo": dict(entry="forward", packed=True, with_o_lo=False, derive=True),
    "bwd_single": dict(entry="backward"),
    "bwd_head_major_fp32_no_olo": dict(entry="backward", head_major=True, with_o_lo=False, dkv_fp32=True, lengths=False),
    "bwd_passes3_chunks": dict(entry="backward", T=40, S=96, key_passes=3, token_chunk=16),
    "bwd_acc_direct_map": dict(entry="backward", acc=True, dst_map=True, dkv_direct=True),
    "bwd_acc_cast_passes2": dict(entry="backward", acc=True, dkv_direct=False, key_passes=2),
    "runner_step": dict(entry="runner"),
    "runner_step_acc_no_olo": dict(entry="runner", acc=True, dst_map=True, with_o_lo=False),
}
#: The cases whose plans depend on the architecture (single-pass token order, the direct rule) also run there.
H64_ARCH_CASES = ("bwd_single", "bwd_acc_direct_map", "runner_step")
H64_ARCHES = ("sm_100a", "sm_103a", "sm_107a")


def h64_case_names() -> list[tuple[str, str]]:
    names = [(name, "sm_100a") for name in H64_CASES]
    names += [(name, arch) for name in H64_ARCH_CASES for arch in H64_ARCHES[1:]]
    return names


def _acc(inp: HostInputs, case: dict) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if not case.get("acc"):
        return None, None
    S = inp.total_k
    if case.get("dst_map"):
        dst_map = (torch.arange(S, dtype=torch.int32) % max(1, S // 3)).contiguous()
        return torch.zeros(max(1, S // 3), 640, dtype=torch.float32), dst_map
    return torch.zeros(S, 640, dtype=torch.float32), None


def run_case(h: HostLaunches, backend, case: dict, inp: Optional[HostInputs] = None) -> dict:
    """Run one case of :data:`H64_CASES` (or the same keys at another head count) under ``h``; returns the call's
    tensors by role."""
    inp = inp or _case_inputs(case)
    h.bind(inp.q_latent)
    with_o_lo = case.get("with_o_lo", True)
    lengths = inp.topk_length if case.get("lengths", True) else None
    args = (inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope, inp.indices)
    roles: dict[str, Any] = dict(
        q_latent=inp.q_latent,
        q_rope=inp.q_rope,
        kv_latent=inp.kv_latent,
        k_rope=inp.k_rope,
        indices=inp.indices,
        topk_length=lengths,
        dout=inp.dout,
        lse_in=inp.lse,
    )
    dkv_acc, dst_map = _acc(inp, case)
    roles.update(dkv_acc=dkv_acc, dkv_dst_map=dst_map)
    entry = case["entry"]
    if entry == "forward":
        out, lse, o_lo = backend.forward(
            *args,
            topk_length=lengths,
            derive_topk_length=case.get("derive", False),
            with_o_lo=with_o_lo,
        )
        roles.update(out=out, lse=lse, o_lo=o_lo)
    elif entry == "backward":
        out = inp.q_latent.clone()
        o_lo = (-inp.q_latent) if with_o_lo else None
        roles.update(out=out, o_lo=o_lo)
        grads = backend.backward(
            *args,
            out,
            o_lo,
            inp.lse,
            inp.dout,
            topk_length=lengths,
            dkv_fp32=case.get("dkv_fp32", False),
            key_passes=case.get("key_passes"),
            dkv_acc=dkv_acc,
            dkv_dst_map=dst_map,
            dkv_direct=case.get("dkv_direct"),
            with_o_lo=with_o_lo,
        )
        roles.update(zip(("dq_latent", "dq_rope", "dkv_latent", "dk_rope"), grads, strict=True))
    elif entry == "runner":
        runner = backend.prepare_dsa_train(
            *args,
            topk_length=lengths,
            dout=inp.dout,
            key_passes=case.get("key_passes"),
            dkv_acc=dkv_acc,
            dkv_dst_map=dst_map,
            dkv_direct=case.get("dkv_direct"),
            with_o_lo=with_o_lo,
        )
        grads = runner.step()
        roles.update(out=runner.out, lse=runner.lse, o_lo=runner.o_lo, runner=runner)
        roles.update(zip(("dq_latent", "dq_rope", "dkv_latent", "dk_rope"), grads, strict=True))
    else:
        raise ValueError(entry)
    roles["inputs"] = inp
    return roles


_ROLE_ORDER = (
    "q_latent",
    "q_rope",
    "kv_latent",
    "k_rope",
    "indices",
    "topk_length",
    "dout",
    "lse_in",
    "dkv_acc",
    "dkv_dst_map",
    "out",
    "lse",
    "o_lo",
    "dq_latent",
    "dq_rope",
    "dkv_latent",
    "dk_rope",
)


def capture_case(monkeypatch, backend, jit, launch, name: str, arch: str) -> dict:
    """The golden record of one 64-head case on ``arch``: ``{"launches": ..., "allocs": ...}``."""
    case = H64_CASES[name]
    h = HostLaunches(
        monkeypatch, backend, jit, launch, arch, token_chunk=case.get("token_chunk"), simulate=False
    )
    roles = run_case(h, backend, case)
    ordered = [(role, roles.get(role)) for role in _ROLE_ORDER]
    ordered = [(role, t) for role, t in ordered if isinstance(t, torch.Tensor)]
    return dict(launches=describe(h.launches, ordered, launch), allocs=h.allocs)


def capture_h64_cases(backend, jit, launch) -> dict:
    found = {}
    for name, arch in h64_case_names():
        with pytest.MonkeyPatch.context() as mp:
            found[f"{name}@{arch}"] = capture_case(mp, backend, jit, launch, name, arch)
    return found


def dumps(found: dict) -> str:
    """The golden file's text: one line per launch / allocation list, keys sorted."""
    lines = ["{"]
    items = list(found.items())
    for i, (key, rec) in enumerate(items):
        lines.append(f"  {json.dumps(key)}: {{")
        lines.append('    "launches": [')
        launches = rec["launches"]
        for j, row in enumerate(launches):
            sep = "," if j + 1 < len(launches) else ""
            lines.append("      " + json.dumps(row, sort_keys=True) + sep)
        lines.append("    ],")
        lines.append('    "allocs": ' + json.dumps(rec["allocs"]))
        lines.append("  }" + ("," if i + 1 < len(items) else ""))
    lines.append("}")
    return "\n".join(lines) + "\n"


def _package():
    from tests.cake.test_helpers.cake_package import cake_package

    name = "cake.cake_dsa_sparse_attention_nondet_train"
    return (
        cake_package(name, "cake_backend"),
        cake_package(name, "cake_jit"),
        cake_package(name, "cake_launch"),
    )


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    parser.add_argument("--write", type=Path, required=True, help="golden file to write")
    args = parser.parse_args(argv)
    backend, jit, launch = _package()
    args.write.write_text(dumps(capture_h64_cases(backend, jit, launch)))
    print(f"wrote {args.write}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
