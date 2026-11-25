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

Head counts above 64 (issue #264: 96 heads, the contract 1..128) in the deterministic and the non-deterministic KDA
training packages, single rank and the context-parallel (KCP) host plan.  The accuracy at 96 and 128 heads is judged
in the packages' suites (``test_cake_kda_*_train_naive.py``, the judged rows of ``test_cake_kda_nondet_train.py`` and
the KCP ``WIDE_HEAD_ROWS`` of ``tests/cross_family``) and here on the last document of the production-size packs;
this file holds the limits, the plans and the production-size rows.

CPU rows (no CUDA device):

- the int32 bound ``R * H * 64 < 2**31`` at 96 and 128 heads: the largest packs accepted, one token or one document
  more refused before any launch;
- the KCP entries' operand checks take 1..128 heads in both gate modes (with the in-kernel Q / K L2 norm in the raw
  gate mode) and refuse 129;
- the head plans of the issue's KCP ranks (16,380 tokens of one document per rank over 4, 8 and 64 ranks) at 96 and
  128 heads: the owners of each head group, no padding, within the int32 offsets;
- the head plans of the KCP judged rows of the head ownership (``OWNERSHIP_ROWS``: two heads per owner with ranks
  that own no head, and a padded owner, over 4 and 8 ranks), every rank of each row;
- the records the KCP route's in-kernel Q / K L2 norm needs (the check of ``cake_kcp_plan.qk_l2norm_missing``, whose
  grid stops at 64 heads) are selected, delivered and exported at 65 to 128 heads in both gate modes and both forms
  of the norm (``"fused"``, the default, and ``"fallback"``);
- the single-rank workspace and saved set at 96 heads, reported apart (``memory_report``), for the single-rank packs
  of the GPU rows and a single-rank call of 16,380 tokens.  The KCP workspace and retained bytes of a rank are those
  of the KCP queries at ``H=96`` in ``tests/cross_family``'s ``test_cake_kda_train_kcp_heads.py``.

GPU rows (one device of an exported architecture), in the issue's mode (the raw gate with lower bound -5, ``A_log``
``[H]``, ``dt_bias`` ``[H, 128]``, the in-kernel Q / K L2 norm):

- single-rank packs at 96 heads (``PACKS``): the issue's 262,144 tokens, and 196,608 tokens, which fit a B200 where
  the issue's pack may not; each holds a one-token document and ends in a document of 1,000 tokens whose rows lie
  beyond 2**31 elements of a ``[S, 96, 128]`` tensor.  Through the autograd entry on a caller-held workspace, each
  training step two forwards and one backward under ``torch.utils.checkpoint(use_reentrant=False)``: the workspace's
  bytes are the query's, a checkpointed forward retains nothing beyond its output, two steady steps run under
  PyTorch's sync debug mode (any synchronizing call it routes raises) and every output and gradient is finite.  Then
  the last document is judged: on each of the judge's 32 draws its operands, ``A_log`` and ``dt_bias`` are a draw of
  a pack of that one document, the output gradient is zero on every other document (so the parameter gradients come
  from the last document alone), and its rows of the output and gradients and the parameter gradients of a step
  (plain, and under the checkpoint: ``ckpt.<name>``) are judged against the FP64 reference beside the naive BF16
  tier and FLA, computed on that document alone (``cake_kda_naive.judge_arms``);
- token counts that change between calls, in a fresh process (no library loaded before): the first call of each plan
  loads its libraries, the build layer's spy sees those loads (each package's ``cake_build.load_library``, the JIT
  shim's builds and ``tvm_ffi.load_module``), and every later call -- a token count no earlier call had, the plan of
  an earlier call -- builds and loads nothing.
"""

from __future__ import annotations

import gc
import json
import os
import tempfile
import types

import pytest
import torch
import torch.multiprocessing as mp

from tests.cake.test_helpers.cake_package import cake_package

PKGS = ("det", "nondet")
D = 128
LOWER_BOUND = -5.0
SCALE = D**-0.5  # the packages' DEFAULT_SCALE
SM = 148
ISSUE_H = 96
TAIL = 1000  # the judged last document of a single-rank pack
TAIL_SEED = 26400
# the single-rank packs at 96 heads: the issue's 262,144 tokens, and 196,608 tokens (fits a B200 with the operands,
# the gradients and the backward arena); a one-token document in each, and the judged last document of TAIL tokens
# from token 261,144 / 195,608 on (the first token whose row offset reaches 2**31 elements is 174,763)
PACKS = {
    "s262144": (40000, 61000, 1, 52000, 108143, TAIL),
    "s196608": (30000, 45000, 1, 39000, 81607, TAIL),
}
T_RANK = 16380  # the issue's tokens per KCP rank
# (Hg, owners) of 96 and 128 heads over 4, 8 and 64 ranks
KCP_OWNERS = {
    96: {4: (24, 4), 8: (12, 8), 64: (2, 48)},
    128: {4: (32, 4), 8: (16, 8), 64: (2, 64)},
}
# The KCP judged rows of the head ownership (``WIDE_HEAD_ROWS`` of ``tests/cross_family``'s
# ``test_cake_kda_*_train_kcp_l2n.py``), contiguous: ``(lengths, H, W)`` ->
# ``(Hg, n_owners, Hg per rank, padded owner, pad tiles)``
OWNERSHIP_ROWS = {
    ((3001,), 6, 4): (2, 3, (2, 2, 2, 0), -1, 0),
    ((700, 1, 1300, 999), 5, 4): (2, 3, (2, 2, 1, 0), 2, 1),
    ((4001,), 12, 8): (2, 6, (2, 2, 2, 2, 2, 2, 0, 0), -1, 0),
    ((700, 1, 1300, 999, 1000), 9, 8): (2, 5, (2, 2, 2, 2, 1, 0, 0, 0), 4, 1),
}
WIDE_HEADS = (65, 96, 127, 128)
# The dynamic-count rows at 96 heads, in the order run: no two packs of the same token count (16,378 to 131,072
# tokens; one document, a few, and more than 256)
SWEEP = (
    (16380,),
    (9999, 6382),
    (300,) * 257,
    (16379,),
    (65536,),
    (131072,),
    (40000, 25538),
    (10000, 6377, 1),
    (60000, 5530, 1, 1, 1, 1, 1, 4),
    (100001,),
    (250,) * 300,
    (70000, 61071),
)
SWEEP_MIN_MEASURED = 4  # the fewest later calls (an earlier call's plan at a new token count) the row measures
SM_OF_ARCH = {"sm_100a": 148, "sm_103a": 152, "sm_107a": 212}
SPY_PACKAGES = (
    "cake.cake_kda_det_train",
    "cake.cake_kda_nondet_train",
    "cake.cake_kda_det_train_kcp",
    "cake.cake_kda_nondet_train_kcp",
)


def _mods(pkg: str) -> types.SimpleNamespace:
    """The single-rank and KCP modules and entry points of package ``pkg`` (``"det"`` / ``"nondet"``)."""
    name = "Det" if pkg == "det" else "Nondet"
    single = f"cake.cake_kda_{pkg}_train"
    backend = cake_package(single, "cake_backend")
    kcp_plan = cake_package(f"cake.cake_kda_{pkg}_train_kcp", "cake_kcp_plan")
    return types.SimpleNamespace(
        backend=backend,
        jit=cake_package(single, "cake_jit"),
        build=cake_package(single, "cake_build"),
        kcp=cake_package(f"cake.cake_kda_{pkg}_train_kcp", "cake_kcp"),
        kcp_plan=kcp_plan,
        entry=getattr(backend, f"cake_kda_{pkg}"),
        workspace=getattr(backend, f"CakeKDA{name}Workspace"),
        partition=getattr(kcp_plan, f"cake_kda_{pkg}_train_kcp_partition"),
        build_context=getattr(kcp_plan, f"cake_kda_{pkg}_train_kcp_build_context"),
    )


# --------------------------------------------------------------------------- CPU rows


@pytest.mark.cpu_only
@pytest.mark.parametrize("pkg", PKGS)
def test_int32_limits_above_64_heads(pkg) -> None:
    """``R * H * 64 < 2**31`` (``R = 64 * NC_pad``): at 96 heads one document of 349,376 tokens and 262,144 tokens in
    1,364 documents (and the packs of the GPU rows) are accepted, one token or one document more refused; at 128 heads
    one document of 261,952 tokens is accepted, 261,953 and 262,144 refused -- by the plan's checks and by the byte
    queries, before any launch."""
    b = _mods(pkg).backend
    accepted = (
        (349376, 1, 96),
        (262144, 1364, 96),
        *((sum(lengths), len(lengths), ISSUE_H) for lengths in PACKS.values()),
        (261952, 1, 128),
    )
    refused = ((349377, 1, 96), (262144, 1365, 96), (261953, 1, 128), (262144, 1, 128))
    for S, N, H in accepted:
        _nc_max, nc_pad, rows = b._check_problem(S, N, H, SM)
        assert rows == 64 * nc_pad and rows * H * 64 < 2**31, (S, N, H)
    for S, N, H in refused:
        with pytest.raises(ValueError, match=r"R \* H \* 64 must stay below 2\*\*31"):
            b._check_problem(S, N, H, SM)
        with pytest.raises(ValueError, match="int32"):
            b.workspace_bytes(S, N, H, sm_count=SM, arch="sm_100a")


@pytest.mark.cpu_only
@pytest.mark.parametrize("pkg", PKGS)
def test_kcp_operands_take_1_to_128_heads(pkg) -> None:
    """The KCP entries' operand checks (``cake_kcp._operands``) take 1..128 heads with the activated gate and with the
    raw gate and the in-kernel Q / K L2 norm, and refuse 129 heads naming the range."""
    m = _mods(pkg)
    ctx = m.build_context(m.partition([0, 40], 1), None, 0, "cpu", 4, sm_count=SM)
    T = int(ctx.T_r)
    dtypes = dict(m.kcp.CONTRACT_DTYPES)
    for H in (1, 64, 65, 96, 128, 129):
        gen = torch.Generator().manual_seed(H)
        q, k, v = (
            torch.randn(T, H, D, generator=gen).to(torch.bfloat16) for _ in range(3)
        )
        beta = torch.rand(T, H, generator=gen).to(torch.bfloat16)
        gates = (
            ("g", -torch.rand(T, H, D, generator=gen), False),
            ("raw", torch.randn(T, H, D, generator=gen).to(torch.bfloat16), True),
        )
        for gate, g, l2n in gates:
            if H <= 128:
                res = m.kcp._operands(
                    q, k, v, g, beta, ctx, dtypes=dtypes, gate=gate, qk_l2norm=l2n
                )
                assert res[5:7] == (T, H), (H, gate)
            else:
                with pytest.raises(ValueError, match=r"take 1\.\.128 heads"):
                    m.kcp._operands(
                        q, k, v, g, beta, ctx, dtypes=dtypes, gate=gate, qk_l2norm=l2n
                    )


@pytest.mark.cpu_only
@pytest.mark.parametrize("pkg", PKGS)
@pytest.mark.parametrize("W", (4, 8, 64))
def test_kcp_head_plans_of_the_issue_ranks(pkg, W) -> None:
    """One document of 16,380 tokens per rank over ``W`` ranks (contiguous): at 96 heads 24 / 12 / 2 heads per owner
    and 4 / 8 / 48 owners (ranks 48..63 of 64 own no head), at 128 heads 32 / 16 / 2 and 4 / 8 / 64; no padding; the
    head plan of the first, a middle and the last rank is built within the int32 offsets."""
    m = _mods(pkg)
    part = m.partition([0, T_RANK * W], W, layout="contiguous")
    ctxs = {
        r: m.build_context(part, None, r, "cpu", 4, sm_count=SM)
        for r in sorted({0, 1, W // 2, W - 1})
    }
    for H, owners in KCP_OWNERS.items():
        Hg, n_owners, hg_r = m.kcp_plan.head_groups(H, W)
        assert (Hg, n_owners) == owners[W], (H, W)
        for r, ctx in ctxs.items():
            assert int(ctx.T_r) == T_RANK
            hp = ctx.heads(H)
            assert (hp.Hg, hp.n_owners, hp.Hg_r, hp.pad_tiles) == (
                Hg,
                n_owners,
                hg_r[r],
                0,
            ), (H, W, r)
            assert hp.Hg_r == (Hg if r < n_owners else 0), (H, W, r)


def _ownership_id(row) -> str:
    lengths, H, W = row
    return "s" + "_".join(map(str, lengths)) + f"-h{H}-w{W}"


@pytest.mark.cpu_only
@pytest.mark.parametrize("pkg", PKGS)
@pytest.mark.parametrize("row", list(OWNERSHIP_ROWS), ids=_ownership_id)
def test_kcp_head_plans_of_the_ownership_rows(pkg, row) -> None:
    """The head plan of every rank of each KCP judged row of the head ownership (contiguous fragments): two heads per
    owner, the owners of each row and the ranks that own no head; the single partial owner of 5 heads over 4 ranks and
    of 9 heads over 8 ranks padded by one tile per slot (``Hg_pad_by_rank``, ``pad_tiles``; the padding counted in the
    splits of the ranks that send to or receive from it, ``pad_numel``); an owner holds the merge chains of every
    document over two or more ranks, a rank without heads none."""
    m = _mods(pkg)
    lengths, H, W = row
    Hg, n_owners, hg_r, padded_owner, pad_tiles = OWNERSHIP_ROWS[row]
    assert m.kcp_plan.head_groups(H, W) == (Hg, n_owners, hg_r), row
    cu = [0]
    for n in lengths:
        cu.append(cu[-1] + n)
    part = m.partition(cu, W, layout="contiguous")
    ctxs = [m.build_context(part, None, r, "cpu", 4, sm_count=SM) for r in range(W)]
    summary = m.kcp_plan.SUMMARY_TILE
    tile = m.kcp_plan.TILE
    padded = 0
    for r, ctx in enumerate(ctxs):
        hp = ctx.heads(H)
        assert (hp.Hg, hp.n_owners, hp.Hg_r, hp.head0) == (
            Hg,
            n_owners,
            hg_r[r],
            r * Hg,
        ), (row, r)
        assert hp.Hg_by_rank == hg_r, (row, r)
        assert hp.Hg_pad_by_rank == tuple(Hg if x > 0 else 0 for x in hg_r), (row, r)
        assert (hp.padded_owner, hp.pad_tiles) == (padded_owner, pad_tiles), (row, r)
        assert hp.full_groups == (pad_tiles == 0), (row, r)
        assert hp.pad_numel["c1_send"] == ctx.n_out * pad_tiles * summary, (row, r)
        assert hp.pad_numel["c2_recv"] == ctx.n_in * pad_tiles * tile, (row, r)
        padded += sum(hp.pad_numel.values())
        crossing = tuple(d for d, chain in enumerate(ctx.chains) if len(chain) >= 2)
        if hg_r[r] > 0:
            assert hp.fwd_chain_doc == crossing and hp.adj_chain_doc == crossing, (
                row,
                r,
            )
        else:
            assert hp.fwd_chain_doc == () and hp.adj_chain_doc == (), (row, r)
    assert (padded > 0) == (pad_tiles > 0), row
    assert sum(1 for x in hg_r if x == 0) >= 1, row  # a rank that owns no head


@pytest.mark.cpu_only
@pytest.mark.parametrize("pkg", PKGS)
@pytest.mark.parametrize("gate", ("g", "raw"))
def test_l2n_records_serve_65_to_128_heads(pkg, gate) -> None:
    """The capability check of the KCP route's in-kernel Q / K L2 norm (``cake_kcp_plan._qk_l2norm_problem``, the
    check :func:`qk_l2norm_missing` runs over its grid of 4 / 16 / 64 heads) holds at 65, 96, 127 and 128 heads in
    both forms of the norm (``QK_L2NORM_FORMS``: ``"fused"``, the default, and ``"fallback"``, which the KCP judged
    rows also run): on every architecture and SM count of the check, 1 / 256 / 257 documents of 1 / 16 / 256 chunks,
    every stage of the form (and of the raw gate's parameter reduce) selects a record delivered and exported for the
    architecture, and the readers give the route's order."""
    m = _mods(pkg)
    kp = m.kcp_plan
    assert kp.QK_L2NORM_FORM_DEFAULT in kp.QK_L2NORM_FORMS
    assert "fallback" in kp.QK_L2NORM_FORMS
    libraries = m.build.libraries_by_name()
    for mode in kp.QK_L2NORM_FORMS:
        modes = {"l2n": mode} if gate == "g" else {"l2n": mode, "gate": gate}
        for arch in m.jit.ARCHES:
            for sm in kp.QK_L2NORM_SMS.get(arch, kp.QK_L2NORM_SMS_ANY):
                for docs in (1, 256, 257):
                    for heads in WIDE_HEADS:
                        for chunks in (1, 16, 256):
                            problem = (chunks * docs, docs, heads)
                            why = kp._qk_l2norm_problem(
                                mode, modes, arch, sm, problem, libraries
                            )
                            assert why is None, (mode, why)


@pytest.mark.cpu_only
@pytest.mark.parametrize("pkg", PKGS)
def test_memory_report_of_the_issue_shapes(pkg, record_property) -> None:
    """Single-rank workspace and saved set reported apart at 96 heads in the issue's mode on every architecture: the
    single-rank packs of the GPU rows (``PACKS``) and a single-rank call of 16,380 tokens in one document.  The
    backward arena (the workspace) and the saved set are both positive, the saved set carries the FP32 rstd of the L2
    norm (``8 * S * H``), and the byte query of a backward workspace is recorded beside the report.  These are
    single-rank bytes: a KCP rank's workspace and retained bytes are those of the KCP queries at ``H=96``
    (``tests/cross_family``'s ``test_cake_kda_train_kcp_heads.py``)."""
    m = _mods(pkg)
    shapes = [(sum(lengths), len(lengths)) for lengths in PACKS.values()]
    shapes.append((T_RANK, 1))
    for arch in m.jit.ARCHES:
        sm = SM_OF_ARCH[arch]
        for S, N in shapes:
            kw = {"sm_count": sm, "arch": arch, "gate": "raw", "l2n": "fused"}
            report = m.backend.memory_report(S, N, ISSUE_H, **kw)
            ws = m.backend.workspace_bytes(S, N, ISSUE_H, **kw)
            assert ws > 0 and report["arena_bwd"] > 0 and report["saved"] > 0, (arch, S)
            assert report["saved_qk_rstd"] == 8 * S * ISSUE_H, (arch, S)
            record_property(
                f"single_rank_bytes.{arch}.S{S}.N{N}.H{ISSUE_H}",
                {
                    "workspace_bytes": ws,
                    "arena_bwd": report["arena_bwd"],
                    "saved": report["saved"],
                    "bwd_total": report["bwd_total"],
                    "replay_total": report["replay_total"],
                },
            )


# --------------------------------------------------------------------------- GPU rows


@pytest.fixture(scope="module")
def device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    index = torch.cuda.current_device()
    capability = tuple(torch.cuda.get_device_capability(index))
    jit = _mods("nondet").jit
    if jit.arch_for_capability(capability) is None:
        pytest.skip(
            f"compute capability {capability} is not an exported target {jit.ARCHES}"
        )
    return torch.device("cuda", index)


def _operands(lengths, H: int, device, seed: int) -> dict:
    """The issue's mode's operands drawn on the device (raw q / k rows of a few magnitudes per token and head, the
    raw BF16 gate, ``A_log`` ``[H]``, ``dt_bias`` ``[H, 128]``) and ``cu_seqlens``."""
    S = int(sum(lengths))
    gen = torch.Generator(device=device).manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, generator=gen, device=device, dtype=torch.bfloat16)

    mag = torch.exp(randn(S, H, 1).float() * 1.5).to(torch.bfloat16)
    ops = {
        "q": (randn(S, H, D) * mag).contiguous(),
        "k": (randn(S, H, D) * mag.flip(0)).contiguous(),
        "v": randn(S, H, D),
        "g": randn(S, H, D),
        "beta": torch.sigmoid(randn(S, H).float()).to(torch.bfloat16),
        "A_log": torch.log(torch.rand(H, generator=gen, device=device) + 1.0),
        "dt_bias": randn(H, D).float() * 0.5,
        "dout": randn(S, H, D),
    }
    cu = [0]
    for n in lengths:
        cu.append(cu[-1] + int(n))
    ops["cu_seqlens"] = torch.tensor(cu, dtype=torch.int32, device=device)
    return ops


LEAVES = ("q", "k", "v", "g", "beta", "A_log", "dt_bias")
GRADS = ("dq", "dk", "dv", "dg", "dbeta", "dA_log", "ddt_bias")
PARAMS = ("A_log", "dt_bias")
PARAM_GRADS = ("dA_log", "ddt_bias")
MODE = {
    "use_gate_in_kernel": True,
    "lower_bound": LOWER_BOUND,
    "use_qk_l2norm_in_kernel": True,
}


@pytest.mark.parametrize("pack", list(PACKS))
@pytest.mark.parametrize("pkg", PKGS)
def test_issue_pack_runs_under_checkpoint_without_host_sync(
    pkg, pack, device, record_property
) -> None:
    """A single-rank pack of ``PACKS`` (96 heads) through the autograd entry on a caller-held workspace, each step
    under ``torch.utils.checkpoint(use_reentrant=False)``: the workspace's bytes are the query's; after a warm-up step
    a checkpointed forward retains nothing beyond its output; two steady steps run under the sync debug mode
    ``"error"``; every output and gradient of every step is finite.  Then the pack's last document (1,000 tokens,
    every row beyond 2**31 elements of a ``[S, 96, 128]`` tensor) is judged over the judge's draws: its operands,
    ``A_log`` and ``dt_bias`` are a draw of a pack of that document alone (``cake_kda_naive.draw_case``), the output
    gradient is zero on the other documents, and its rows of the output and gradients and the parameter gradients of
    a plain step (``<name>``) and a checkpointed step (``ckpt.<name>``) are judged against the FP64 reference beside
    the naive BF16 tier and FLA of that document alone, with the family's rule (no tolerance; a miss of both
    references fails).  A device without room for the pack (the report's bytes and the operands) skips, naming
    both."""
    from torch.utils.checkpoint import checkpoint

    from tests.experimental.test_helpers.cake_kda_naive import draw_case, judge_arms

    m = _mods(pkg)
    lengths, H = PACKS[pack], ISSUE_H
    S, N = sum(lengths), len(lengths)
    t0 = S - lengths[-1]
    assert lengths[-1] == TAIL and t0 * H * D >= 2**31, (pack, t0)
    sm = m.jit.sm_count(device.index)
    arch = m.jit.device_arch(device.index)
    kw = {"sm_count": sm, "arch": arch, "gate": "raw", "l2n": "fused"}
    report = m.backend.memory_report(S, N, H, **kw)
    io = S * H * D * 2  # one BF16 [S, H, 128] tensor
    need = 6 * io + report["saved"] + report["bwd_total"] + (1 << 30)
    gc.collect()
    torch.cuda.empty_cache()
    free, _total = torch.cuda.mem_get_info(device)
    record_property(
        "bytes",
        {
            "S": S,
            "N": N,
            "need": need,
            "free": free,
            "io": io,
            **{k: report[k] for k in ("arena_bwd", "saved", "bwd_total")},
        },
    )
    if need > free:
        pytest.skip(f"the pack needs about {need} B on the device, {free} B are free")
    ops = _operands(lengths, H, device, seed=0x264)
    cu, dout = ops["cu_seqlens"], ops["dout"]
    ws = m.workspace.for_problem(S, N, H, device, gate="raw", l2n="fused")
    assert ws.bytes == m.backend.workspace_bytes(S, N, H, **kw)
    leaves = [ops[n].requires_grad_(True) for n in LEAVES]

    def fn(q, k, v, g, beta, A_log, dt_bias):
        return m.entry(
            q, k, v, g, beta, cu, workspace=ws, A_log=A_log, dt_bias=dt_bias, **MODE
        )

    def step(ckpt: bool = True) -> dict:
        for x in leaves:
            x.grad = None
        out = checkpoint(fn, *leaves, use_reentrant=False) if ckpt else fn(*leaves)
        out.backward(dout)
        return {
            "out": out.detach(),
            **dict(zip(GRADS, (x.grad for x in leaves), strict=True)),
        }

    def finite(res: dict) -> list[str]:
        return [n for n, t in res.items() if not bool(torch.isfinite(t).all())]

    assert finite(step()) == [], "warm-up"
    # retained state: a checkpointed forward keeps only its output (the saved set goes through the checkpoint)
    for x in leaves:
        x.grad = None
    torch.cuda.synchronize(device)
    gc.collect()
    before = torch.cuda.memory_allocated(device)
    out = checkpoint(fn, *leaves, use_reentrant=False)
    torch.cuda.synchronize(device)
    kept = torch.cuda.memory_allocated(device) - before - out.untyped_storage().nbytes()
    out.backward(dout)
    del out
    record_property("retained_after_checkpointed_forward", kept)
    assert kept == 0, kept
    for i in range(2):
        torch.cuda.synchronize(device)
        torch.cuda.set_sync_debug_mode("error")
        try:
            res = step()
        finally:
            torch.cuda.set_sync_debug_mode("default")
        assert finite(res) == [], ("steady", i)
        del res

    # the judged last document: the other documents keep their operands, their output gradient is zero
    with torch.no_grad():
        dout.zero_()

    def draw(j: int, s: int):
        inputs = draw_case((TAIL,), H, "raw", True, s, device, dt_bias_2d=True)
        with torch.no_grad():
            for name, leaf in zip(LEAVES, leaves, strict=True):
                if name in PARAMS:
                    leaf.copy_(inputs[name])
                else:
                    leaf[t0:].copy_(inputs[name])
            dout[t0:].copy_(inputs["dout"])
        arms = {}
        for arm, ckpt in (("", False), ("ckpt", True)):
            res = step(ckpt)
            arms[arm] = {
                name: (t if name in PARAM_GRADS else t[t0:]).clone()
                for name, t in res.items()
            }
            del res
        return arms, inputs

    judge_arms(
        record_property,
        draw,
        seed=TAIL_SEED,
        scale=SCALE,
        l2n=True,
        lower_bound=LOWER_BOUND,
    )


class _BuildLoadSpy:
    """Every library build and load at the build layer of the four KDA packages, in the order made, as ``(kind,
    name)``: each package's ``cake_build.load_library`` (what ``cake_jit.load`` / ``cake_kcp_jit.load`` call on a
    record's first use) and ``cake_build.jit_spec``, its JIT shim's ``JitSpec.build`` and ``JitSpec.build_and_load``
    (nvcc, or the cached file of an earlier build), and ``tvm_ffi.load_module``, which both the prebuilt path and the
    JIT path call.  Installed for the rest of the process (a spawned one)."""

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


def _sweep_entry(_index: int, pkg: str, device_index: int, path: str) -> None:
    """The dynamic-count row in a fresh (spawned) process: the spy installed before any call, one call per plan of
    ``SWEEP`` (its first pack) as the warm-up, then the other packs; the report as JSON at ``path``."""
    device = torch.device("cuda", device_index)
    torch.cuda.set_device(device)
    m = _mods(pkg)
    spy = _BuildLoadSpy().install()
    H = ISSUE_H
    sm = m.jit.sm_count(device.index)
    arch = m.jit.device_arch(device.index)

    def plan_key(lengths) -> tuple:
        p = m.backend.plan(
            sum(lengths), len(lengths), H, sm, arch, gate="raw", l2n="fused"
        )
        return tuple(
            sorted(
                (route, tuple(sorted((st, r) for st, r in by.items() if r is not None)))
                for route, by in p.records.items()
            )
        )

    def call(lengths) -> None:
        ops = _operands(lengths, H, device, seed=sum(lengths))
        leaves = [ops[n].requires_grad_(True) for n in LEAVES]
        out = m.entry(
            *leaves[:5], ops["cu_seqlens"], A_log=leaves[5], dt_bias=leaves[6], **MODE
        )
        out.backward(ops["dout"])
        torch.cuda.synchronize(device)
        ok = all(
            bool(torch.isfinite(t).all()) for t in (out, *(x.grad for x in leaves))
        )
        assert ok, ("not finite", sum(lengths))

    keys = [plan_key(lengths) for lengths in SWEEP]
    first: dict[tuple, int] = {}
    for i, key in enumerate(keys):
        first.setdefault(key, i)
    warm = sorted(first.values())
    measured = [i for i in range(len(SWEEP)) if i not in set(warm)]
    for i in warm:
        call(SWEEP[i])
    warm_events = spy.take()
    per_call = []
    for i in measured:
        call(SWEEP[i])
        per_call.append({"tokens": sum(SWEEP[i]), "events": spy.take()})
    report = {
        "package": pkg,
        "arch": arch,
        "sm_count": sm,
        "distinct_plans": len(first),
        "plan_of_pack": [warm.index(first[key]) for key in keys],
        "warm_tokens": [sum(SWEEP[i]) for i in warm],
        "warm_kinds": _kinds(warm_events),
        "warm_libraries": sorted({n for k, n in warm_events if k == "load_library"}),
        "measured_tokens": [sum(SWEEP[i]) for i in measured],
        "measured_events": per_call,
    }
    with open(path, "w") as fh:
        json.dump(report, fh, indent=1)


@pytest.mark.parametrize("pkg", PKGS)
def test_changing_token_counts_build_and_load_nothing(
    pkg, device, record_property
) -> None:
    """Calls of 96 heads in the issue's mode (one training step each, the default per-call workspace) at changing token
    counts (``SWEEP``: no two of the same count, one document, a few, more than 256) in a fresh spawned process, where
    no library was loaded before.  Positive control: the warm-up (the first pack of each distinct plan, its records on
    every path) loads libraries, and the spy at the build layer sees them (``load_library`` and ``load_module``
    events).  Then every other pack -- a token count no earlier call had, the plan of a warm-up pack -- builds and
    loads nothing: no ``load_library``, ``jit_spec``, ``build``, ``build_and_load`` or ``load_module`` event.  At
    least ``SWEEP_MIN_MEASURED`` packs are measured; the number of distinct plans is reported."""
    gc.collect()
    torch.cuda.empty_cache()
    with tempfile.TemporaryDirectory(prefix="cake_kda_sweep_") as tmp:
        path = os.path.join(tmp, "report.json")
        mp.start_processes(
            _sweep_entry,
            args=(pkg, device.index, path),
            nprocs=1,
            join=True,
            start_method="spawn",
        )
        with open(path) as fh:
            report = json.load(fh)
    record_property("report", json.dumps(report))
    warm = report["warm_kinds"]
    assert warm.get("load_library", 0) >= 1, ("positive control: no load seen", report)
    assert warm.get("load_module", 0) >= warm["load_library"], (
        "positive control: a library load did not reach tvm_ffi.load_module",
        report,
    )
    assert len(report["measured_tokens"]) >= SWEEP_MIN_MEASURED, report
    loaded = [c for c in report["measured_events"] if c["events"]]
    assert loaded == [], loaded
