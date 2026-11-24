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

Issue #264 at its production size on the context-parallel (KCP) route of both KDA training packages: 96 heads in the
issue's mode (the raw gate with lower bound -5, ``A_log`` ``[H]`` and ``dt_bias`` ``[H, 128]``, the in-kernel Q / K
L2 norm), 16,380 tokens per rank in contiguous fragments.  The accuracy of this mode at 96 heads is judged on the
packs of ``test_cake_kda_*_train_kcp_l2n.py`` (``WIDE_HEAD_ROWS``); these rows are larger than the judged size and
keep exact checks only, as the 64-head rows of that size do.

Rows ``(lengths, W)``: one document over every rank, and documents that start and end inside a rank (a one-token
document among them), at 4 and 8 ranks (spawned on one node, or one process per Slurm task) and at 64 ranks (one
process per Slurm task; the rows exist only in a Slurm step of 64 tasks, so a run of another size shows no skip for
them).  Every rank draws its own operands only: no rank holds the whole pack (1,048,320 tokens at 64 ranks).

Per rank, after one warm-up step of each form (a training step is two forwards and one backward under
``torch.utils.checkpoint(use_reentrant=False)``; ``plain``: no recompute handle, ``handle``: a ``"states"`` recompute
handle, the saved state of the checkpoint replay):

- the caller-held workspace is reserved once: its bytes are the workspace query's and no step changes them;
- retained state, reported apart from the workspace: after a checkpointed forward the bytes still requested beyond
  the output are 0 without a handle and the handle's own bytes with one (the saved-bytes query of the handle's tier
  is reported beside them);
- the steady steps (two plain, one with a handle) run in the host-synchronization census of
  ``test_helpers/cake_host_sync.py``: no pageable copy, no synchronizing call, the host returns while queued device
  work still runs;
- every output and gradient is finite, the cross-rank sums of the ``dA_log`` / ``ddt_bias`` partials too;
- the deterministic package: the two steady plain steps give the same bits.

The segment count ``P`` of each rank's plan is reported (the head count enters the KCP plan rule).  The workspace and
retained bytes of these rows are this route's: the KCP byte queries take ``H=96`` explicitly (their default is 64).

Fragment layouts that change between calls (two ranks spawned on one node, 16,380 tokens per rank, 96 heads in the
issue's mode, one training step per call): each pair of ``LAYOUT_PAIRS`` holds two packs whose ranks hold the same
number of fragments and the same crossing fragments at other fragment lengths.  In fresh spawned processes (no
library loaded before) the first pack of every pair runs as the warm-up, and the spy at the build layer (each
package's ``cake_build.load_library`` and ``jit_spec``, the JIT shim's builds and ``tvm_ffi.load_module``) sees its
loads on every rank (the positive control); then the second packs, layouts no earlier call had, run one after another
and build and load nothing on any rank.
"""

from __future__ import annotations

import gc
import json
import os
import tempfile
import types

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tests.cake.test_helpers.cake_package import cake_package
from tests.experimental.test_cake_kda_det_train_kcp_dist import (
    _programs_exported as _det_programs_exported,
)
from tests.experimental.test_cake_kda_nondet_train_kcp_dist import (
    _bound_rank_threads,
    _init_group,
    _programs_exported as _nondet_programs_exported,
    _rendezvous,
)
from tests.experimental.test_helpers.cake_host_sync import census, nonblocking, summary

PKGS = ("det", "nondet")
D = 128
H = 96
LOWER_BOUND = -5.0
T_RANK = 16380  # the issue's tokens per rank
SLEEP_MS = 4000.0  # the census's queued work: a whole training step's host time stays well inside a quarter of it
# (lengths, W): contiguous fragments of 16,380 tokens per rank
ROWS = [
    ((T_RANK * 4,), 4),
    ((10000, 30000, 1, 25519), 4),
    ((T_RANK * 8,), 8),
    ((20000, 1, 50000, 61039), 8),
]
SLURM_ROWS = {
    64: [
        ((T_RANK * 64,), 64),
        (
            (
                100000,
                1,
                200000,
                50000,
                150000,
                99999,
                120000,
                80000,
                100000,
                60000,
                38320,
                50000,
            ),
            64,
        ),
    ],
}
ROWS += SLURM_ROWS.get(int(os.environ.get("SLURM_NTASKS", "0") or 0), [])
# The dynamic-layout row: two ranks, 2 x 16,380 tokens, contiguous; per pair the same fragment and crossing counts per
# rank (rank 0 / rank 1: two documents with the boundary in rank 0, then in rank 1; four documents with a one-token
# document, three fragments on rank 0 and two on rank 1) at other fragment lengths
LAYOUT_W = 2
LAYOUT_PAIRS = (
    ((5000, 27760), (9001, 23759)),
    ((20000, 12760), (25001, 7759)),
    ((3000, 1, 20000, 9759), (7000, 1, 15000, 10759)),
)
SPY_PACKAGES = (
    "cake.cake_kda_det_train",
    "cake.cake_kda_nondet_train",
    "cake.cake_kda_det_train_kcp",
    "cake.cake_kda_nondet_train_kcp",
)
LEAVES = ("q", "k", "v", "g", "beta", "A_log", "dt_bias")
GRADS = ("dq", "dk", "dv", "dg", "dbeta", "dA_log", "ddt_bias")


def _mods(pkg: str) -> types.SimpleNamespace:
    """The KCP modules and entry points of package ``pkg`` (``"det"`` / ``"nondet"``)."""
    name = "Det" if pkg == "det" else "Nondet"
    kcp = cake_package(f"cake.cake_kda_{pkg}_train_kcp", "cake_kcp")
    plan = cake_package(f"cake.cake_kda_{pkg}_train_kcp", "cake_kcp_plan")
    return types.SimpleNamespace(
        kcp=kcp,
        plan=plan,
        jit=cake_package(f"cake.cake_kda_{pkg}_train", "cake_jit"),
        entry=getattr(kcp, f"cake_kda_{pkg}_kcp"),
        workspace=getattr(kcp, f"CakeKDA{name}KcpWorkspace"),
        handle=getattr(kcp, f"CakeKDA{name}KcpRecomputeHandle"),
        reduce=getattr(kcp, f"cake_kda_{pkg}_train_kcp_reduce_gate_param_grads"),
        partition=getattr(plan, f"cake_kda_{pkg}_train_kcp_partition"),
        build_context=getattr(plan, f"cake_kda_{pkg}_train_kcp_build_context"),
        workspace_bytes=getattr(plan, f"cake_kda_{pkg}_train_kcp_workspace_bytes"),
        saved_bytes=getattr(plan, f"cake_kda_{pkg}_train_kcp_saved_bytes"),
        exported=_det_programs_exported if pkg == "det" else _nondet_programs_exported,
    )


def _row_id(row) -> str:
    lengths, W = row
    docs = "1doc" if len(lengths) == 1 else f"{len(lengths)}docs"
    return f"s{sum(lengths)}-{docs}-h{H}-w{W}"


def _requested(device) -> int:
    return int(torch.cuda.memory_stats(device).get("requested_bytes.all.current", 0))


def _local_operands(T: int, device, rank: int) -> dict:
    """This rank's operands in the issue's mode, drawn on the device from a per-rank seed (raw q / k rows of a few
    magnitudes per token and head); ``A_log`` / ``dt_bias`` are the same on every rank."""
    gen = torch.Generator(device=device).manual_seed(0x264 + 7919 * rank)

    def randn(*shape):
        return torch.randn(*shape, generator=gen, device=device, dtype=torch.bfloat16)

    mag = torch.exp(randn(T, H, 1).float() * 1.5).to(torch.bfloat16)
    ops = {
        "q": (randn(T, H, D) * mag).contiguous(),
        "k": (randn(T, H, D) * mag.flip(0)).contiguous(),
        "v": randn(T, H, D),
        "g": randn(T, H, D),
        "beta": torch.sigmoid(randn(T, H).float()).to(torch.bfloat16),
        "dout": randn(T, H, D),
    }
    cpu = torch.Generator().manual_seed(0x264)
    ops["A_log"] = torch.log(torch.rand(H, generator=cpu) + 1.0).to(device)
    ops["dt_bias"] = (torch.randn(H, D, generator=cpu) * 0.5).to(device)
    return ops


def _run_rank(
    pkg: str, rank: int, W: int, row, rdzv: str | None, result_dir: str | None
) -> dict:
    """One rank of a row: the checks of the module docstring; returns the report (every rank's flags gathered)."""
    from torch.utils.checkpoint import checkpoint

    m = _mods(pkg)
    lengths, _w = row
    device = torch.device("cuda", rank % max(torch.cuda.device_count(), 1))
    torch.cuda.set_device(device)
    if W > 1 and not dist.is_initialized():
        _init_group(rank, W, device, rdzv)
    group = dist.group.WORLD if W > 1 else None
    cu = [0]
    for n in lengths:
        cu.append(cu[-1] + n)
    part = m.partition(cu, W, layout="contiguous")
    ctx = m.build_context(part, group, rank, device)
    T = int(ctx.T_r)
    ops = _local_operands(T, device, rank)
    modes = dict(use_gate_in_kernel=True, use_qk_l2norm_in_kernel=True)
    ws = m.workspace(**modes)
    reserved = ws.reserve(ctx, H)
    flags: dict = {}
    reported: dict = {
        "T_r": T,
        "n_frag": int(ctx.n_frag),
        "workspace_bytes": reserved,
        "workspace_query": m.workspace_bytes(ctx, H=H, **modes),
    }
    flags["workspace_is_the_query"] = (
        reserved == ws.bytes == reported["workspace_query"]
    )
    leaves = [ops[n].requires_grad_(True) for n in LEAVES]
    dout = ops["dout"]

    def fn(q, k, v, g, beta, A_log, dt_bias, *handle):
        h = handle[0] if handle else None
        return m.entry(
            q, k, v, g, beta, ctx, workspace=ws, recompute_handle=h,
            checkpoint_keep="states" if h is not None else None,
            A_log=A_log, dt_bias=dt_bias, lower_bound=LOWER_BOUND, **modes,
        )  # fmt: skip

    def step(form: str) -> dict:
        """One training step (two forwards and one backward under the checkpoint); its outputs and gradients, the
        parameter gradients as the cross-rank sums (a collective)."""
        for x in leaves:
            x.grad = None
        extra = (m.handle(),) if form == "handle" else ()
        out = checkpoint(fn, *leaves, *extra, use_reentrant=False)
        out.backward(dout)
        res = {"out": out.detach()}
        res.update({n: x.grad for n, x in zip(GRADS[:5], leaves[:5], strict=True)})
        res["dA_log"], res["ddt_bias"] = m.reduce(ctx, leaves[5].grad, leaves[6].grad)
        return res

    def retained(form: str) -> tuple[int, int]:
        """Bytes requested beyond the output after a checkpointed forward of ``form`` (its backward then runs), and
        the handle's own bytes (0 without one)."""
        for x in leaves:
            x.grad = None
        extra = (m.handle(),) if form == "handle" else ()
        torch.cuda.synchronize(device)
        gc.collect()
        before = _requested(device)
        out = checkpoint(fn, *leaves, *extra, use_reentrant=False)
        torch.cuda.synchronize(device)
        kept = _requested(device) - before - out.numel() * out.element_size()
        own = extra[0].nbytes() if extra else 0
        out.backward(dout)
        torch.cuda.synchronize(device)
        return kept, own

    def finite(res: dict) -> bool:
        return all(bool(torch.isfinite(t).all()) for t in res.values())

    for form in ("plain", "handle"):
        flags[f"finite_warm_{form}"] = finite(step(form))
        reported[f"P_{form}"] = m.kcp.LAST_CALL.get("fwd", {}).get("P")
    kept_plain, _ = retained("plain")
    kept_handle, own_handle = retained("handle")
    reported["retained_plain"] = kept_plain
    reported["retained_handle"] = kept_handle
    reported["handle_nbytes"] = own_handle
    for key, keep in (("handle_saved_query", "states"), ("saved_query", None)):
        try:  # reported beside the measured bytes, not compared
            reported[key] = m.saved_bytes(
                ctx, H=H, checkpoint_keep=keep, with_conv=False, **modes
            )
        except NotImplementedError as exc:
            reported[key] = f"NotImplementedError: {exc}"
    flags["retained_plain_zero"] = kept_plain == 0
    flags["retained_handle_is_the_handle"] = kept_handle == own_handle
    sleep = float(SLEEP_MS)
    steady = []
    for form in ("plain", "plain", "handle"):
        c = census(lambda form=form: step(form), device, sleep_ms=sleep)
        reported.setdefault("census", []).append({"form": form, **summary(c)})
        flags[f"nonblocking_{len(steady)}_{form}"] = nonblocking(c, sleep)
        steady.append(c["out"])
        flags[f"finite_steady_{len(steady) - 1}_{form}"] = finite(c["out"])
    flags["workspace_unchanged"] = ws.bytes == reserved
    if pkg == "det":
        a, b = steady[0], steady[1]
        flags["det_steady_bitwise"] = all(torch.equal(a[n], b[n]) for n in a)
    del steady
    flag_list: list = [None] * W
    reported_list: list = [None] * W
    if W > 1:
        dist.all_gather_object(flag_list, flags)
        dist.all_gather_object(reported_list, reported)
    else:
        flag_list, reported_list = [flags], [reported]
    report = {
        "package": pkg,
        "row": _row_id(row),
        "rank": rank,
        "world": W,
        "flags_by_rank": flag_list,
        "reported_by_rank": reported_list,
        "pass": all(all(f.values()) for f in flag_list),
        "device": torch.cuda.get_device_name(device),
        "arch": m.jit.device_arch(device.index),
        "host": os.uname().nodename,
        "job": os.environ.get("SLURM_JOB_ID"),
    }
    if rank == 0 and result_dir:
        os.makedirs(result_dir, exist_ok=True)
        with open(
            os.path.join(result_dir, f"heads-{pkg}-{_row_id(row)}.json"), "w"
        ) as fh:
            json.dump(report, fh, indent=1, default=str)
    if W > 1:
        dist.barrier()
    gc.collect()
    return report


def _spawn_entry(rank: int, pkg: str, W: int, row, rdzv: str, result_dir: str) -> None:
    try:
        _bound_rank_threads(W)
        report = _run_rank(pkg, rank, W, row, rdzv, result_dir)
        if not report["pass"]:
            raise AssertionError(f"rank {rank}: {report}")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _skip_unless_runnable(m, W: int) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    if torch.cuda.device_count() < W:
        pytest.skip(f"{W} CUDA devices required, {torch.cuda.device_count()} visible")
    if not m.exported():
        pytest.skip("the KCP programs are not exported in the package's build.json yet")
    capability = tuple(torch.cuda.get_device_capability(torch.cuda.current_device()))
    if m.jit.arch_for_capability(capability) is None:
        pytest.skip("the current device's architecture is not covered by the package")


@pytest.mark.parametrize("row", ROWS, ids=_row_id)
@pytest.mark.parametrize("pkg", PKGS)
def test_issue_rows_at_96_heads(pkg, row, record_property) -> None:
    """One row on ``W`` ranks (:func:`_run_rank`): every rank's flags hold; the reports are recorded (``report``)."""
    m = _mods(pkg)
    _lengths, W = row
    if (
        os.environ.get("SLURM_PROCID") is not None
        and int(os.environ.get("SLURM_NTASKS", "1")) == W
    ):
        # one process per Slurm task (rendezvous from MASTER_ADDR / MASTER_PORT)
        _skip_unless_runnable(m, 1)
        report = _run_rank(
            pkg,
            int(os.environ["SLURM_PROCID"]),
            W,
            row,
            None,
            os.environ.get("KCP_TEST_RESULT_DIR"),
        )
    else:
        _skip_unless_runnable(m, W)
        with tempfile.TemporaryDirectory(prefix="cake_kcp_heads_") as tmp:
            result_dir = os.environ.get("KCP_TEST_RESULT_DIR") or tmp
            mp.start_processes(
                _spawn_entry,
                args=(pkg, W, row, _rendezvous(tmp), result_dir),
                nprocs=W,
                join=True,
                start_method="spawn",
            )
            with open(
                os.path.join(result_dir, f"heads-{pkg}-{_row_id(row)}.json")
            ) as fh:
                report = json.load(fh)
    record_property(
        "report",
        json.dumps(
            {k: v for k, v in report.items() if k != "flags_by_rank"}, default=str
        ),
    )
    assert report["pass"], report["flags_by_rank"]


class _BuildLoadSpy:
    """Every library build and load at the build layer of the four KDA packages, in the order made, as ``(kind,
    name)``: each package's ``cake_build.load_library`` (what ``cake_jit.load`` / ``cake_kcp_jit.load`` call on a
    record's first use) and ``cake_build.jit_spec``, its JIT shim's ``JitSpec.build`` and ``JitSpec.build_and_load``
    (nvcc, or the cached file of an earlier build), and ``tvm_ffi.load_module``, which both the prebuilt path and the
    JIT path call.  Installed for the rest of the process (a spawned rank)."""

    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []

    def _wrap(self, owner, attr: str, kind: str, label) -> None:
        real = getattr(owner, attr)

        def spy(*args, **kwargs):
            self.events.append((kind, label(args)))
            return real(*args, **kwargs)

        setattr(owner, attr, spy)

    def install(self) -> _BuildLoadSpy:
        import tvm_ffi

        self._wrap(tvm_ffi, "load_module", "load_module", lambda a: str(a[0]))
        for name in SPY_PACKAGES:
            build = cake_package(name, "cake_build")
            shim = cake_package(name, "_cake_pplx_jit")
            self._wrap(build, "load_library", "load_library", lambda a: str(a[0]))
            self._wrap(build, "jit_spec", "jit_spec", lambda a: str(a[0]))
            self._wrap(shim.JitSpec, "build", "build", lambda a: a[0].name)
            self._wrap(
                shim.JitSpec, "build_and_load", "build_and_load", lambda a: a[0].name
            )
        return self

    def take(self) -> list[tuple[str, str]]:
        events, self.events = self.events, []
        return events


def _kinds(events) -> dict[str, int]:
    counts: dict[str, int] = {}
    for kind, _name in events:
        counts[kind] = counts.get(kind, 0) + 1
    return counts


def _layout_rank(rank: int, pkg: str, W: int, rdzv: str, result_dir: str) -> None:
    """One spawned rank of the dynamic-layout row: the spy installed before any call, the first pack of every pair
    (the warm-up), then the second packs one after another; rank 0 writes every rank's report."""
    try:
        _bound_rank_threads(W)
        m = _mods(pkg)
        device = torch.device("cuda", rank % max(torch.cuda.device_count(), 1))
        torch.cuda.set_device(device)
        _init_group(rank, W, device, rdzv)
        spy = _BuildLoadSpy().install()
        ops = _local_operands(T_RANK, device, rank)
        modes = dict(use_gate_in_kernel=True, use_qk_l2norm_in_kernel=True)
        leaves = [ops[n].requires_grad_(True) for n in LEAVES]

        def call(lengths) -> dict:
            """One training step on the layout of ``lengths``: its fragment counts, ``P`` and launch inventory."""
            cu = [0]
            for n in lengths:
                cu.append(cu[-1] + n)
            part = m.partition(cu, W, layout="contiguous")
            ctx = m.build_context(part, dist.group.WORLD, rank, device)
            for x in leaves:
                x.grad = None
            out = m.entry(
                *leaves[:5], ctx, A_log=leaves[5], dt_bias=leaves[6],
                lower_bound=LOWER_BOUND, **modes,
            )  # fmt: skip
            out.backward(ops["dout"])
            torch.cuda.synchronize(device)
            grads = [x.grad for x in leaves[:5]]
            fwd, bwd = m.kcp.LAST_CALL.get("fwd", {}), m.kcp.LAST_CALL.get("bwd", {})
            return {
                "T_r": int(ctx.T_r),
                "frags": (int(ctx.n_frag), int(ctx.n_in), int(ctx.n_out)),
                "P": fwd.get("P"),
                "inventory": [
                    list(fwd.get("sequence", ())),
                    list(bwd.get("sequence", ())),
                ],
                "finite": all(bool(torch.isfinite(t).all()) for t in (out, *grads)),
            }

        warm = [call(first) for first, _second in LAYOUT_PAIRS]
        warm_events = spy.take()
        dist.barrier()
        measured = []
        for _first, second in LAYOUT_PAIRS:
            got = call(second)
            got["events"] = spy.take()
            measured.append(got)
        flags = {
            "T_r": all(c["T_r"] == T_RANK for c in warm + measured),
            "same_fragments": all(
                a["frags"] == b["frags"] for a, b in zip(warm, measured, strict=True)
            ),
            "finite": all(c["finite"] for c in warm + measured),
            "warm_loads_seen": _kinds(warm_events).get("load_library", 0) >= 1
            and _kinds(warm_events).get("load_module", 0)
            >= _kinds(warm_events).get("load_library", 0),
            "measured_nothing_loaded": all(not c["events"] for c in measured),
        }
        mine = {
            "flags": flags,
            "warm_kinds": _kinds(warm_events),
            "warm_libraries": sorted(
                {n for k, n in warm_events if k == "load_library"}
            ),
            "warm": [{k: v for k, v in c.items() if k != "inventory"} for c in warm],
            "measured": [
                {k: v for k, v in c.items() if k != "inventory"} for c in measured
            ],
            "same_inventory": [
                a["inventory"] == b["inventory"]
                for a, b in zip(warm, measured, strict=True)
            ],
        }
        gathered: list = [None] * W
        dist.all_gather_object(gathered, mine)
        if rank == 0:
            report = {
                "package": pkg,
                "world": W,
                "by_rank": gathered,
                "pass": all(all(g["flags"].values()) for g in gathered),
                "device": torch.cuda.get_device_name(device),
                "arch": m.jit.device_arch(device.index),
                "host": os.uname().nodename,
            }
            with open(os.path.join(result_dir, f"layouts-{pkg}.json"), "w") as fh:
                json.dump(report, fh, indent=1, default=str)
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.parametrize("pkg", PKGS)
def test_changing_fragment_layouts_build_and_load_nothing(pkg, record_property) -> None:
    """The dynamic-layout row (module docstring) on two spawned ranks: on every rank the warm-up's loads are seen by
    the spy (``load_library`` and ``load_module`` events), the second pack of every pair has its first pack's
    fragment and crossing counts and 16,380 tokens per rank, every output and gradient is finite, and the second packs
    build and load nothing (no ``load_library``, ``jit_spec``, ``build``, ``build_and_load`` or ``load_module``
    event).  Whether each pair launched the same sequence of programs and collectives is reported."""
    m = _mods(pkg)
    _skip_unless_runnable(m, LAYOUT_W)
    with tempfile.TemporaryDirectory(prefix="cake_kcp_layouts_") as tmp:
        result_dir = os.environ.get("KCP_TEST_RESULT_DIR") or tmp
        os.makedirs(result_dir, exist_ok=True)
        mp.start_processes(
            _layout_rank,
            args=(pkg, LAYOUT_W, _rendezvous(tmp), result_dir),
            nprocs=LAYOUT_W,
            join=True,
            start_method="spawn",
        )
        with open(os.path.join(result_dir, f"layouts-{pkg}.json")) as fh:
            report = json.load(fh)
    record_property("report", json.dumps(report, default=str))
    assert report["pass"], [g["flags"] for g in report["by_rank"]]
