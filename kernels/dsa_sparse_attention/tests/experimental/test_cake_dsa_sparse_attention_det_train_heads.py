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

The deterministic package's RoPE form at 96 and 128 query heads (and its 64-head plans).

On sm_100a / sm_103a / sm_107a a call over ``H`` = 96 or 128 heads runs the head-count builds of the record
(``cake_jit.HEAD_COUNT_STAGES``): the single-pass forward ``fwd_h128``, the delta stage's heads build
``bwd_delta_heads`` (``delta [T, H]``) and the main kernel's head-window builds ``bwd_main_hw`` / ``bwd_redo_hw`` /
``bwd_main_pass_hw`` (both 64-head windows of every token in one launch: ``2 T`` work items, window selector -1).  The
key norms, the row bound and its grid, the redo preparation, the fold and the compaction run over the ``T`` tokens
unchanged.  The NoPE program serves 64 heads.

Host tests (CPU, no device; the planners see the architecture and SM count a test stands in, and every launch goes
through its generated launcher into a recorder): the records of the five builds; 64-head plans and launches equal with
and without the builds registered; above 64 heads the stage order, the grids (``fwd_h128``'s persistent grid, one CTA
per item for the main and pass builds, the redo's persistent grid over ``2 T`` items) and the launch values (head
count, window selector, item order, ``[T, H]`` operands) on every architecture; the pack / pack27 choice from
``T * topk``; the key-pass token chunk and the sm_107a key-pass rule; the workspace (``2 T`` dQ partial blocks of 36,864
FP32, ``delta [T, H]``); validation; the architecture rule, the refusals without the builds (also a registry without
the sm_107a builds) and calls without query rows.  Device tests (sm_100a / sm_103a / sm_107a): every floating output
against the FP64 oracle at the naive BF16 tier (``tests/cake_naive.py``), and bit-for-bit repeats -- eager, autograd,
the prepared runner and its CUDA-graph replay, dK / dV over 1, 2 and 3 key passes, the forward over other grids, and a
forced redo of most rows over its persistent and a per-item grid; on sm_107a the plan on the device's SM count.
"Bitwise" is bit-pattern equality (integer views).
"""

import contextlib
import functools
from types import SimpleNamespace

import pytest
import torch

from tests.cake.test_helpers.cake_package import cake_package
from tests.experimental.test_helpers import cake_dsa_sparse_attention_naive as nv
from tests.experimental.test_helpers.cake_dsa_sparse_attention_nondet_train_reference import (
    make_inputs,
)

cake_backend = cake_package("cake.cake_dsa_sparse_attention_det_train", "cake_backend")
cake_jit = cake_package("cake.cake_dsa_sparse_attention_det_train", "cake_jit")
cake_launch = cake_package("cake.cake_dsa_sparse_attention_det_train", "cake_launch")
nondet_jit = cake_package("cake.cake_dsa_sparse_attention_nondet_train", "cake_jit")
nondet_launch = cake_package(
    "cake.cake_dsa_sparse_attention_nondet_train", "cake_launch"
)

NUM_HEADS = cake_backend.NUM_HEADS
D_LATENT = cake_backend.D_LATENT
D_ROPE = cake_backend.D_ROPE
D_QK = cake_backend.D_QK
SUPPORTED_COMPUTE_CAPABILITIES = cake_backend.SUPPORTED_COMPUTE_CAPABILITIES
HEAD_COUNT_STAGES = cake_jit.HEAD_COUNT_STAGES
HEAD_COUNT_ARCHES = ("sm_100a", "sm_103a", "sm_107a")
#: The architectures whose main launches walk the items down with one key pass (``DESCENDING_TOKEN_PLANS``).
DESCENDING_ARCHES = ("sm_103a", "sm_107a")
#: FP32 elements of one dQ partial block (one 64-head work item: 64 x 576).
DQ_BLOCK = NUM_HEADS * D_QK
SEED = 20261126
REPEATS = 5
#: The softmax scale of the issue's model (192-wide query-key head dimension).
SCALE = 192**-0.5
TOPK = 2048
#: The SM count the host tests stand in.
SMS = 148
GRID_ARGS = ("grid_x", "grid_y", "grid_z")


def _bits(t: torch.Tensor) -> torch.Tensor:
    view = {2: torch.int16, 4: torch.int32, 8: torch.int64}[t.element_size()]
    return t.detach().contiguous().view(view)


def _same_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(_bits(a), _bits(b))


# ---------------------------------------------------------------------------
# Host layer (CPU)
# ---------------------------------------------------------------------------


class _Host:
    """The stand-ins of one host test.  The planners see ``arch`` and :data:`SMS` SMs; a stage's module is a recorder
    (``cake_jit.library_name`` still checks that the stage ships for the architecture), so every launch runs its
    generated launcher and lands in ``launches`` as ``(stage, {argument: value})``; allocations land on the CPU.
    ``strip`` removes record entries (the head-count builds) as if the registry did not carry them."""

    def __init__(self, monkeypatch, arch: str, *, strip: tuple[str, ...] = ()):
        self.launches: list[tuple[str, dict]] = []
        if strip:
            real_record = cake_jit.record

            def record(program=cake_jit.PROGRAM):
                rec = real_record(program)
                if program != cake_jit.PROGRAM:
                    return rec
                rec = {k: v for k, v in rec.items() if k not in strip}
                rec["stages"] = [s for s in rec["stages"] if s not in strip]
                return rec

            monkeypatch.setattr(cake_jit, "record", record)
        real_alloc = cake_backend._alloc

        def alloc(shape, dtype, device, *, zero: bool = False):
            return real_alloc(shape, dtype, torch.device("cpu"), zero=zero)

        def load(stage, arch_, program=cake_jit.PROGRAM):
            cake_jit.library_name(stage, arch_, program)
            return SimpleNamespace(run=functools.partial(self._record, stage))

        monkeypatch.setattr(cake_backend, "arch_for", lambda device=None: arch)
        monkeypatch.setattr(cake_backend, "_device_index", lambda device=None: 0)
        monkeypatch.setattr(cake_backend, "_sm_count", lambda index: SMS)
        monkeypatch.setattr(cake_backend, "_alloc", alloc)
        monkeypatch.setattr(cake_jit, "load_cake_dsa_train_module", load)
        monkeypatch.setattr(
            cake_backend, "_ffi_stream_context", lambda index: contextlib.nullcontext()
        )
        monkeypatch.setattr(cake_backend, "BINDING_CACHE", cake_backend._BindingCache())
        monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
        monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)

    def _record(self, stage: str, *args) -> None:
        names = cake_launch.STAGE_ARGS[stage]
        assert len(args) == len(names), (stage, len(args), len(names))
        self.launches.append((stage, dict(zip(names, args, strict=True))))

    def stages(self) -> list[str]:
        return [stage for stage, _ in self.launches]

    def values(self, stage: str) -> list[dict]:
        return [v for s, v in self.launches if s == stage]


def _host_inputs(T: int, S: int, topk: int, heads: int, *, seed: int = 0):
    g = torch.Generator().manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, generator=g).to(torch.bfloat16)

    return SimpleNamespace(
        q_latent=randn(T, heads, D_LATENT),
        q_rope=randn(T, heads, D_ROPE),
        kv_latent=randn(S, D_LATENT),
        k_rope=randn(S, D_ROPE),
        indices=torch.randint(0, S, (T, topk), generator=g, dtype=torch.int32),
        topk_length=torch.full((T,), topk, dtype=torch.int32),
        dout=randn(T, heads, D_LATENT),
        T=T,
        S=S,
        topk=topk,
        heads=heads,
    )


def _args(inp):
    return (inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope, inp.indices)


def _describe(value):
    if isinstance(value, torch.Tensor):
        return ("tensor", tuple(value.shape), tuple(value.stride()), str(value.dtype))
    return value


def _described(launches) -> list:
    return [
        (stage, {name: _describe(v) for name, v in values.items()})
        for stage, values in launches
    ]


def _runner_case(inp, *, key_passes=None, accumulate=False, **kw):
    acc = torch.zeros(inp.S, D_QK) if accumulate else None
    return cake_backend.prepare_dsa_train(
        *_args(inp),
        topk_length=inp.topk_length,
        dout=inp.dout,
        key_passes=key_passes,
        dkv_acc=acc,
        **kw,
    )


def test_head_count_records_are_the_builds_the_backend_launches():
    """The record registers the five head-count builds after its other stages, for sm_100a, sm_103a and sm_107a, each
    with a generated launcher and grid.  ``fwd_h128`` is the non-deterministic package's single-pass forward (the
    same kernel identity, launch shape and arguments); the head-window builds take the arguments of their 64-head
    stage plus ``num_heads`` and ``wsel``, the heads delta those of the 64-head delta plus ``num_heads``."""
    record = cake_jit.record()
    stages = cake_jit.registered_stages()
    assert HEAD_COUNT_STAGES == {
        "fwd": "fwd_h128",
        "bwd_delta": "bwd_delta_heads",
        "bwd_main": "bwd_main_hw",
        "bwd_redo": "bwd_redo_hw",
        "bwd_main_pass": "bwd_main_pass_hw",
    }
    builds = tuple(HEAD_COUNT_STAGES.values())
    assert stages[-len(builds) :] == builds
    for build in builds:
        assert cake_jit.stage_arches(build) == HEAD_COUNT_ARCHES, build
        assert build in cake_launch.LAUNCH and build in cake_launch.GRID, build
        assert record[build]["launch"]["use_pdl"] == (build == "fwd_h128"), build
    nondet = nondet_jit.record()["fwd_h128"]
    assert record["fwd_h128"]["launch"] == nondet["launch"]
    assert (
        record["fwd_h128"]["library"].rsplit("_", 1)[1]
        == nondet["library"].rsplit("_", 1)[1]
    )
    assert cake_launch.STAGE_ARGS["fwd_h128"] == nondet_launch.STAGE_ARGS["fwd_h128"]

    def body(stage):
        return [a for a in cake_launch.STAGE_ARGS[stage] if a not in GRID_ARGS]

    for base in ("bwd_main", "bwd_redo", "bwd_main_pass"):
        assert body(HEAD_COUNT_STAGES[base]) == body(base) + ["num_heads", "wsel"]
    assert body("bwd_delta_heads") == body("bwd_delta") + ["num_heads"]


H64_CASES = {
    "one_pass": dict(),
    "two_passes": dict(key_passes=2),
    "accumulate": dict(accumulate=True),
    "accumulate_three_passes": dict(accumulate=True, key_passes=3),
    "dkv_fp32": dict(dkv_fp32=True),
    "o_lo_scratch": dict(o_lo_scratch=True),
}


@pytest.mark.parametrize("arch", ("sm_100a", "sm_103a", "sm_107a"))
@pytest.mark.parametrize("name", list(H64_CASES))
def test_h64_plans_ignore_the_head_count_builds(monkeypatch, name, arch):
    """A 64-head binding plans and launches the same with the head-count builds registered and without them: the
    stage order, every grid, every launch value (tensors by shape, strides and dtype) and the workspace layout --
    prepared runner and eager entries."""
    inp = _host_inputs(8, 64, 16, NUM_HEADS)
    kw = H64_CASES[name]
    found = []
    for strip in ((), tuple(HEAD_COUNT_STAGES.values())):
        with monkeypatch.context() as mp:
            host = _Host(mp, arch, strip=strip)
            runner = _runner_case(inp, **kw)
            runner.step()
            grids = {k: tuple(v.grid) for k, v in runner.launches.items()}
            out, lse, o_lo = cake_backend.forward(
                *_args(inp), topk_length=inp.topk_length
            )
            eager_kw = {k: v for k, v in kw.items() if k != "accumulate"}
            if kw.get("accumulate"):
                eager_kw["dkv_acc"] = torch.zeros(inp.S, D_QK)
            cake_backend.backward(
                *_args(inp),
                out,
                o_lo,
                lse,
                inp.dout,
                topk_length=inp.topk_length,
                **eager_kw,
            )
            found.append(
                (
                    runner.forward_stages,
                    runner.backward_order,
                    grids,
                    runner.layout,
                    runner.storage,
                    _described(host.launches),
                )
            )
    assert found[0] == found[1]
    assert found[0][0] == ("fwd",)


def _expected_order(passes: int, accumulate: bool) -> tuple:
    order = list(cake_jit.BACKWARD_STAGES)
    if accumulate:
        order = [cake_jit.ACCUMULATE_STAGES.get(s, s) for s in order]
    if passes > 1:
        main = order.index("bwd_main")
        order[main : main + 1] = [
            (stage, i) for i in range(passes) for stage in cake_jit.KEY_PASS_STAGES
        ]
    swap = {k: v for k, v in HEAD_COUNT_STAGES.items() if k != "fwd"}
    return tuple(
        (swap.get(k[0], k[0]), k[1]) if isinstance(k, tuple) else swap.get(k, k)
        for k in order
    )


@pytest.mark.parametrize("accumulate", (False, True), ids=("fresh", "dkv_acc"))
@pytest.mark.parametrize("passes", (1, 2))
@pytest.mark.parametrize("heads", (96, 128))
@pytest.mark.parametrize("arch", HEAD_COUNT_ARCHES)
def test_head_count_plan_launches(monkeypatch, arch, heads, passes, accumulate):
    """Above 64 heads a prepared step launches ``fwd_h128`` and the 64-head backward order with the head-count
    builds in place of the delta, main, redo and pass stages.  Grids: ``fwd_h128`` the persistent single-pass grid
    ``2 * min(T, SMs // 2)``; the heads delta one CTA per token; the main and pass builds one CTA per work item
    (``2 T``); the redo ``redo_launch_grid(2 T, SMs)``; every other stage the grid of a 64-head plan of the same
    ``T``.  Values: ``num_heads = H`` and ``wsel = -1`` where the build takes them; ``num_queries`` / ``num_tokens`` =
    ``T``; the main and redo items descending from ``2 T - 1`` on sm_103a and sm_107a with one pass, else ascending
    from 0 (the compaction from token 0); ``lse`` / ``delta`` ``[T, H]``; dQ ``[T, H, *]``; the pass dQ partial
    ``2 T`` blocks of 36,864 FP32."""
    T, S, topk = 40, 96, 16
    host = _Host(monkeypatch, arch)
    inp = _host_inputs(T, S, topk, heads)
    runner = _runner_case(inp, key_passes=passes, accumulate=accumulate)
    assert runner.num_heads == heads
    assert runner.forward_stages == ("fwd_h128",)
    assert runner.backward_order == _expected_order(passes, accumulate)
    runner.step()
    assert host.stages() == [
        k[0] if isinstance(k, tuple) else k
        for k in runner.forward_stages + runner.backward_order
    ]
    grids = {k: tuple(v.grid) for k, v in runner.launches.items()}
    assert grids["fwd_h128"] == (2 * min(T, SMS // 2), 1, 1)
    assert grids["bwd_delta_heads"] == (T, 1, 1)
    assert grids["bwd_redo_hw"] == (min(2 * T, SMS), 1, 1)
    if passes == 1:
        assert grids["bwd_main_hw"] == (2 * T, 1, 1)
    else:
        for p in range(passes):
            assert grids[("bwd_main_pass_hw", p)] == (2 * T, 1, 1)
    with monkeypatch.context() as mp:
        _Host(mp, arch)
        small = _runner_case(
            _host_inputs(T, S, topk, NUM_HEADS),
            key_passes=passes,
            accumulate=accumulate,
        )
        base = {k: tuple(v.grid) for k, v in small.launches.items()}
    for key, grid in grids.items():
        stage = key[0] if isinstance(key, tuple) else key
        if stage not in cake_jit.HEAD_COUNT_STAGES.values():
            assert grid == base[key], key
    descending = passes == 1 and arch in DESCENDING_ARCHES
    order = (2 * T - 1, -1) if descending else (0, 1)
    for stage, values in host.launches:
        if "num_heads" in cake_launch.STAGE_ARGS[stage]:
            assert values["num_heads"] == heads, stage
        if "wsel" in cake_launch.STAGE_ARGS[stage]:
            assert values["wsel"] == -1, stage
        if stage in ("bwd_main_hw", "bwd_redo_hw"):
            assert (values["token_base"], values["token_step"]) == order, stage
        if stage == "bwd_main_pass_hw":
            assert (values["token_base"], values["token_step"]) == (0, 1)
            assert values["dq_partial"].numel() == 2 * T * DQ_BLOCK
        if stage == "bwd_compact":
            assert (values["token_base"], values["token_step"]) == (0, 1)
            assert values["num_tokens"] == T
        if stage in ("fwd_h128", "bwd_main_hw", "bwd_redo_hw", "bwd_main_pass_hw"):
            assert values["num_queries"] == T, stage
            assert tuple(values["lse"].shape) == (T, heads), stage
        if stage.startswith("bwd_main") or stage == "bwd_redo_hw":
            assert tuple(values["delta"].shape) == (T, heads), stage
            assert tuple(values["dq_latent"].shape) == (T, heads, D_LATENT), stage
            assert tuple(values["dq_rope"].shape) == (T, heads, D_ROPE), stage
        if stage == "bwd_delta_heads":
            assert values["num_tokens"] == T
            assert tuple(values["delta"].shape) == (T, heads)
            assert tuple(values["out"].shape) == (T, heads, D_LATENT)
            assert tuple(values["qstat"].shape)[0] == T


@pytest.mark.parametrize("arch", HEAD_COUNT_ARCHES)
def test_head_count_eager_and_autograd_entries_launch_the_builds(monkeypatch, arch):
    """The eager forward / backward and the autograd entry above 64 heads launch the same stages as a prepared step,
    with ``[T, H]`` outputs; a negative softmax scale reaches the heads delta as its magnitude (its bound terms) and
    the main build signed."""
    T, S, topk, heads = 24, 80, 16, 96
    host = _Host(monkeypatch, arch)
    inp = _host_inputs(T, S, topk, heads)
    out, lse, o_lo = cake_backend.forward(
        *_args(inp), topk_length=inp.topk_length, softmax_scale=-0.25
    )
    assert tuple(out.shape) == (T, heads, D_LATENT)
    assert tuple(lse.shape) == (T, heads)
    assert tuple(o_lo.shape) == (T, heads, D_LATENT)
    grads = cake_backend.backward(
        *_args(inp),
        out,
        o_lo,
        lse,
        inp.dout,
        topk_length=inp.topk_length,
        softmax_scale=-0.25,
    )
    assert tuple(grads[0].shape) == (T, heads, D_LATENT)
    assert tuple(grads[1].shape) == (T, heads, D_ROPE)
    assert host.stages() == ["fwd_h128"] + list(_expected_order(1, False))
    assert host.values("bwd_delta_heads")[0]["sm_scale"] == 0.25
    assert host.values("bwd_main_hw")[0]["sm_scale"] == -0.25
    mark = len(host.launches)
    leaves = [t.clone().requires_grad_() for t in _args(inp)[:4]]
    got = cake_backend.cake_dsa_sparse_attention_det(*leaves, inp.indices)
    assert tuple(got.shape) == (T, heads, D_LATENT)
    got.backward(inp.dout)
    assert [s for s, _ in host.launches[mark:]] == ["fwd_h128"] + list(
        _expected_order(1, False)
    )


@pytest.mark.parametrize("heads", (96, 128))
def test_pack_choice_reads_the_token_count(monkeypatch, heads):
    """The row bound and its grid run over the ``T`` tokens above 64 heads too: the pack27 choice reads
    ``T * topk`` (not the ``2 T`` work items), and swaps the pack27 builds into a head-count order exactly where it
    swaps them into a 64-head one (``2 ** 26 <= T * topk < 2 ** 27``)."""
    seen = []
    real = cake_backend._pack27_order

    def spy(order, program, num_queries, topk):
        seen.append((int(num_queries), int(topk)))
        return real(order, program, num_queries, topk)

    host = _Host(monkeypatch, "sm_100a")
    monkeypatch.setattr(cake_backend, "_pack27_order", spy)
    inp = _host_inputs(40, 96, 16, heads)
    runner = _runner_case(inp)
    assert seen == [(40, 16)]
    runner.step()
    assert [v["num_queries"] for v in host.values("bwd_bound")] == [40]
    assert [v["num_tokens"] for v in host.values("bwd_norm")] == [40]
    order = runner.backward_order
    for T, topk, swapped in (
        ((1 << 26) // 2048 - 1, 2048, False),
        ((1 << 26) // 2048, 2048, True),
        ((1 << 27) // 2048 - 1, 2048, True),
        ((1 << 27) // 2048, 2048, False),
        ((1 << 25) // 2048, 2048, False),
    ):
        got = real(order, cake_jit.PROGRAM, T, topk)
        want = tuple(
            cake_jit.PACK27_STAGES.get(k, k) if swapped and isinstance(k, str) else k
            for k in order
        )
        assert got == want, (T, topk)


def test_redo_grid_walks_the_items():
    """The head-window redo's persistent grid covers the ``2 T`` work items: one CTA per SM, at most one per item,
    at least one."""
    rule = cake_backend.PERSISTENT_GRIDS["bwd_redo_hw"]
    for T, sms, want in (
        (1, 148, 2),
        (40, 148, 80),
        (74, 148, 148),
        (75, 148, 148),
        (5000, 160, 160),
    ):
        assert rule(T, 7, sms) == want, (T, sms)
        assert rule(T, 7, sms) == cake_backend.redo_launch_grid(2 * T, sms)
    assert cake_backend.single_pass_grid(1, 148) == (2, 1, 1)
    assert cake_backend.single_pass_grid(74, 148) == (148, 1, 1)
    assert cake_backend.single_pass_grid(5000, 160) == (160, 1, 1)


def test_key_pass_token_chunk_counts_two_blocks_per_token(monkeypatch):
    """Above 64 heads the key-pass policy's token chunk counts two dQ partial blocks per token (2176 tokens at
    top-k 2048 against 4224 at 64 heads): a row of 2177 tokens and 131,072 keys on sm_100a takes the policy's three
    passes at 64 heads and one pass above."""
    _Host(monkeypatch, "sm_100a")
    record = cake_jit.record()
    policy = cake_backend.KeyPassPolicy.from_record(record)
    assert policy.token_chunk(TOPK) == 4224
    assert policy.token_chunk(TOPK, cake_backend.HEAD_WINDOW_ITEMS) == 2176
    stages = cake_jit.registered_stages()
    S = 131072

    def passes(T, items):
        return cake_backend.plan_key_passes(
            record, stages, T, S, TOPK, arch="sm_100a", items_per_token=items
        )

    assert passes(2177, 1) == 3
    assert passes(2177, 2) == 1
    assert passes(2176, 2) == passes(2176, 1) == 3
    for heads, T, want in ((64, 2177, 3), (96, 2177, 1), (128, 2176, 3)):
        assert cake_backend.dsa_train_workspace_size(
            T, S, TOPK, num_heads=heads
        ) == cake_backend.dsa_train_workspace_size(
            T, S, TOPK, num_heads=heads, key_passes=want
        ), (heads, T)


def test_sm107a_key_pass_rule_above_64_heads(monkeypatch):
    """On sm_107a the record's rule (two passes for a one-chunk row of at least 131,072 keys) holds above 64 heads
    within the head-window token chunk (2176 tokens at top-k 2048): 2176 tokens over 131,072 keys take two passes at
    64 heads and above, 2177 tokens two at 64 heads and one above (a row longer than the chunk), and 131,071 keys
    one."""
    _Host(monkeypatch, "sm_107a")
    record = cake_jit.record()
    stages = cake_jit.registered_stages()
    items = cake_backend.HEAD_WINDOW_ITEMS

    def passes(T, S, n):
        return cake_backend.plan_key_passes(
            record, stages, T, S, TOPK, arch="sm_107a", items_per_token=n
        )

    assert passes(2176, 131072, 1) == passes(2176, 131072, items) == 2
    assert passes(2177, 131072, 1) == 2
    assert passes(2177, 131072, items) == 1
    assert passes(2176, 131071, 1) == passes(2176, 131071, items) == 1


@pytest.mark.parametrize("passes", (1, 2))
@pytest.mark.parametrize("heads", (96, 128))
@pytest.mark.parametrize("arch", HEAD_COUNT_ARCHES)
def test_head_count_workspace(monkeypatch, arch, heads, passes):
    """The workspace above 64 heads: ``delta [T, H]`` FP32 (where it does not fit the fresh dK/dV output), and with
    key passes ``2 T`` dQ partial blocks of 36,864 FP32 (147,456 B per work item, not ``H * 2304`` B per token).
    ``dsa_train_workspace_size(num_heads=H)`` is the prepared runner's layout."""
    T, S, topk = 40, 96, 16
    _Host(monkeypatch, arch)
    inp = _host_inputs(T, S, topk, heads)
    for accumulate in (False, True):
        runner = _runner_case(inp, key_passes=passes, accumulate=accumulate)
        layout = runner.layout
        size = cake_backend.dsa_train_workspace_size(
            T, S, topk, num_heads=heads, key_passes=passes, dkv_acc=accumulate
        )
        assert size == layout["total"], (accumulate, size, layout)
        if "delta" in layout:
            assert layout["delta"][1] == T * heads * 4
        else:
            assert tuple(runner.tensors["delta"].shape) == (T, heads)
        if passes > 1:
            assert layout["dq_partial"][1] == 2 * T * DQ_BLOCK * 4
            assert tuple(runner.tensors["dq_partial"].shape) == (2 * T, DQ_BLOCK)
            assert layout["key_scratch"][1] == T * topk * 4
            assert layout["pass_counts"][1] == T * 4
        else:
            assert "dq_partial" not in layout
    assert (
        cake_backend.workspace_layout(T, S, topk, key_passes=2, num_heads=heads)[
            "dq_partial"
        ][1]
        == 2 * cake_backend.workspace_layout(T, S, topk, key_passes=2)["dq_partial"][1]
    )


def test_head_count_storage_scales_with_the_heads():
    """The dead-storage rules read the call's head count: ``o_lo`` and dQ hold ``H`` KiB per token, ``delta`` takes
    ``4 H`` B per token; at 64 heads every rule is the 64-head one."""
    T, S = 16, 1000
    for heads in (64, 96, 128):
        cap = T * heads * D_LATENT * 2
        lay = cake_backend.dq_scratch_layout(T, S, heads)
        assert lay is not None and lay["total"] <= cap
        assert (
            cake_backend.output_scratch_layout(T, S, num_heads=heads)["delta"][1]
            == T * heads * 4
        )
    R = 450
    assert not cake_backend.carve_fits(T, R, R)
    assert cake_backend.carve_fits(T, R, R, num_heads=128)
    assert cake_backend.dq_scratch_layout(T, 300000) is None
    assert cake_backend.dq_scratch_layout(T, 300000, 128) is not None
    assert cake_backend.dq_scratch_layout(T, S) == cake_backend.dq_scratch_layout(
        T, S, 64
    )
    assert cake_backend.carve_fits(T, 400, 400) == cake_backend.carve_fits(
        T, 400, 400, num_heads=64
    )


@pytest.mark.parametrize("heads", (96, 128))
def test_sm107a_serves_more_than_64_heads(monkeypatch, heads):
    """On sm_107a 96 and 128 heads run the head-count builds, as on sm_100a / sm_103a: the eager forward and
    backward, the autograd entries (flat and packed) and ``prepare_dsa_train`` plan them, the workspace size takes
    them, and a CUDA call without query rows returns its empty outputs without launching; 64 heads plan as before."""
    host = _Host(monkeypatch, "sm_107a")
    inp = _host_inputs(8, 64, 16, heads)
    backward = list(_expected_order(1, False))
    out, lse, o_lo = cake_backend.forward(*_args(inp))
    assert host.stages() == ["fwd_h128"]
    assert tuple(out.shape) == (8, heads, D_LATENT) and tuple(lse.shape) == (8, heads)
    cake_backend.backward(*_args(inp), out, o_lo, lse, inp.dout)
    assert host.stages() == ["fwd_h128"] + backward
    mark = len(host.launches)
    leaves = [t.clone().requires_grad_() for t in _args(inp)[:4]]
    got = cake_backend.cake_dsa_sparse_attention_det(*leaves, inp.indices)
    got.backward(inp.dout)
    assert host.stages()[mark:] == ["fwd_h128"] + backward
    cu_q = torch.tensor([0, 3, 8], dtype=torch.int32)
    cu_k = torch.tensor([0, 24, 64], dtype=torch.int32)
    monkeypatch.setattr(
        cake_backend, "nope_index_prep_available", lambda device=None: False
    )
    mark = len(host.launches)
    leaves = [t.clone().requires_grad_() for t in _args(inp)[:4]]
    got = cake_backend.cake_dsa_sparse_attention_det_varlen(
        *leaves, inp.indices % 24, cu_q, cu_k, causal=True
    )
    assert tuple(got.shape) == (8, heads, D_LATENT)
    assert host.stages()[mark:] == ["fwd_h128"]
    runner = cake_backend.prepare_dsa_train(*_args(inp), dout=inp.dout)
    assert runner.num_heads == heads and runner.forward_stages == ("fwd_h128",)
    assert runner.backward_order == _expected_order(1, False)
    assert cake_backend.dsa_train_workspace_size(8, 64, 16, num_heads=heads) > 0
    mark = len(host.launches)
    empty = _host_inputs(0, 64, 16, heads)
    out, lse, o_lo = cake_backend.forward(*_args(empty))
    assert tuple(out.shape) == (0, heads, D_LATENT) and tuple(lse.shape) == (0, heads)
    assert len(host.launches) == mark
    small = _host_inputs(8, 64, 16, NUM_HEADS)
    cake_backend.forward(*_args(small))
    assert host.stages()[mark:] == ["fwd"]


def _without_sm107a_builds(monkeypatch):
    """The registry as it was before the sm_107a head-count builds: each of the five builds declared for sm_100a and
    sm_103a only."""
    real_record = cake_jit.record

    def record(program=cake_jit.PROGRAM):
        rec = real_record(program)
        if program != cake_jit.PROGRAM:
            return rec
        rec = dict(rec)
        for build in HEAD_COUNT_STAGES.values():
            rec[build] = dict(rec[build], arches=["sm_100a", "sm_103a"])
        return rec

    monkeypatch.setattr(cake_jit, "record", record)


@pytest.mark.parametrize("heads", (96, 128))
def test_sm107a_refuses_more_heads_when_the_registry_lacks_its_builds(
    monkeypatch, heads
):
    """A registry that declares the head-count builds for sm_100a and sm_103a only refuses 96 and 128 heads on
    sm_107a before any launch -- eager forward and backward, the autograd entry, ``prepare_dsa_train`` and the
    workspace size -- with ``NotImplementedError`` naming the missing build; sm_100a still runs them, and 64 heads
    plan as before on sm_107a."""
    _without_sm107a_builds(monkeypatch)
    host = _Host(monkeypatch, "sm_107a")
    assert "sm_107a" not in cake_jit.stage_arches("fwd_h128")
    inp = _host_inputs(8, 64, 16, heads)
    missing = r"need the build 'fwd_h128' .* for sm_107a"
    zeros = torch.zeros(8, heads, D_LATENT, dtype=torch.bfloat16)
    lse = torch.zeros(8, heads)
    with pytest.raises(NotImplementedError, match=missing):
        cake_backend.forward(*_args(inp))
    with pytest.raises(NotImplementedError, match=missing):
        cake_backend.backward(*_args(inp), zeros, zeros, lse, inp.dout)
    leaves = [t.clone().requires_grad_() for t in _args(inp)[:4]]
    with pytest.raises(NotImplementedError, match=missing):
        cake_backend.cake_dsa_sparse_attention_det(*leaves, inp.indices)
    with pytest.raises(NotImplementedError, match=missing):
        cake_backend.prepare_dsa_train(*_args(inp), dout=inp.dout)
    with pytest.raises(NotImplementedError, match=missing):
        cake_backend.dsa_train_workspace_size(8, 64, 16, num_heads=heads)
    assert not host.launches
    cake_backend.forward(*_args(_host_inputs(8, 64, 16, NUM_HEADS)))
    assert host.stages() == ["fwd"]
    with monkeypatch.context() as mp:
        served = _Host(mp, "sm_100a")
        cake_backend.forward(*_args(inp))
        assert served.stages() == ["fwd_h128"]


def test_more_heads_need_a_head_count_arch():
    """``HEAD_COUNT_ARCHES`` (sm_100a, sm_103a, sm_107a: every supported architecture) serve 96 and 128 heads;
    another architecture, or no supported device (``None``), is refused with ``NotImplementedError`` naming them, and
    64 heads are never refused by this rule."""
    assert cake_backend.HEAD_COUNT_ARCHES == HEAD_COUNT_ARCHES
    assert set(HEAD_COUNT_ARCHES) == set(SUPPORTED_COMPUTE_CAPABILITIES.values())
    for heads in (64, 96, 128):
        for arch in HEAD_COUNT_ARCHES:
            cake_backend._check_head_arch(heads, arch)
        for arch in ("sm_90a", "sm_120a", None):
            if heads == NUM_HEADS:
                cake_backend._check_head_arch(heads, arch)
                continue
            with pytest.raises(
                NotImplementedError, match="serves 64 query heads only"
            ) as err:
                cake_backend._check_head_arch(heads, arch)
            assert "'sm_107a'" in str(err.value), str(err.value)


@pytest.mark.parametrize("arch", HEAD_COUNT_ARCHES)
def test_missing_head_count_builds(monkeypatch, arch):
    """Without the head-count builds in the registry a 96-head call is refused with an error naming the missing
    build; without only the pass build the backward plans one pass and refuses ``key_passes > 1``."""
    with monkeypatch.context() as mp:
        _Host(mp, arch, strip=tuple(HEAD_COUNT_STAGES.values()))
        inp = _host_inputs(8, 64, 16, 96)
        with pytest.raises(NotImplementedError, match="fwd_h128"):
            cake_backend.forward(*_args(inp))
        with pytest.raises(NotImplementedError, match="fwd_h128"):
            cake_backend.dsa_train_workspace_size(8, 64, 16, num_heads=96)
    with monkeypatch.context() as mp:
        _Host(mp, arch, strip=("bwd_redo_hw",))
        inp = _host_inputs(8, 64, 16, 96)
        cake_backend.forward(*_args(inp))
        with pytest.raises(NotImplementedError, match="bwd_redo_hw"):
            cake_backend.prepare_dsa_train(*_args(inp), dout=inp.dout)
    with monkeypatch.context() as mp:
        _Host(mp, arch, strip=("bwd_main_pass_hw",))
        inp = _host_inputs(8, 64, 16, 96)
        runner = cake_backend.prepare_dsa_train(*_args(inp), dout=inp.dout)
        assert runner.key_passes == 1
        with pytest.raises(NotImplementedError, match="key_passes > 1"):
            cake_backend.prepare_dsa_train(*_args(inp), dout=inp.dout, key_passes=2)


def _validate(q_heads, *, rope_heads=None, dout_heads=None):
    inp = _host_inputs(8, 16, 5, q_heads)
    q_rope = (
        inp.q_rope
        if rope_heads is None
        else torch.zeros(8, rope_heads, D_ROPE, dtype=torch.bfloat16)
    )
    dout = (
        None
        if dout_heads is None
        else torch.zeros(8, dout_heads, D_LATENT, dtype=torch.bfloat16)
    )
    return cake_backend.validate_dsa_train_inputs(
        inp.q_latent, q_rope, inp.kv_latent, inp.k_rope, inp.indices, dout=dout
    )


@pytest.mark.parametrize("heads", (64, 96, 128))
def test_validate_accepts_the_supported_head_counts(heads):
    """The RoPE form validates 64, 96 and 128 heads on the host (the architecture rule needs the device), and
    ``num_query_heads`` reads ``H``; the NoPE form keeps 64 heads and names the RoPE form for 96 and 128."""
    assert tuple(_validate(heads, dout_heads=heads)) == (8, 16, 5)
    assert cake_backend.num_query_heads(torch.zeros(8, heads, D_LATENT)) == heads
    q = torch.zeros(8, heads, D_LATENT, dtype=torch.bfloat16)
    kv = torch.zeros(16, D_LATENT, dtype=torch.bfloat16)
    idx = torch.zeros(8, 5, dtype=torch.int32)
    if heads == NUM_HEADS:
        assert tuple(cake_backend.validate_dsa_nope_inputs(q, kv, idx)) == (8, 16, 5)
    else:
        with pytest.raises(ValueError, match="NoPE program serves 64 query heads only"):
            cake_backend.validate_dsa_nope_inputs(q, kv, idx)


@pytest.mark.parametrize("heads", (1, 32, 63, 65, 80, 160, 192))
def test_validate_rejects_other_head_counts(heads):
    with pytest.raises(ValueError, match=r"H in \(64, 96, 128\)"):
        _validate(heads)


@pytest.mark.parametrize(
    "q_heads, rope_heads, dout_heads",
    [(96, 64, None), (64, 96, None), (96, 128, None), (96, 96, 64), (128, 128, 96)],
)
def test_validate_rejects_mismatched_head_counts(q_heads, rope_heads, dout_heads):
    with pytest.raises(ValueError):
        _validate(q_heads, rope_heads=rope_heads, dout_heads=dout_heads)


@pytest.mark.parametrize("heads", (96, 128))
@pytest.mark.parametrize("arch", HEAD_COUNT_ARCHES)
def test_zero_query_rows_return_empty_outputs_without_launching(
    monkeypatch, arch, heads
):
    """A call without query rows returns ``[0, H, *]`` outputs and zero key gradients without planning or
    launching."""
    host = _Host(monkeypatch, arch)
    inp = _host_inputs(0, 64, 16, heads)
    out, lse, o_lo = cake_backend.forward(*_args(inp))
    assert tuple(out.shape) == (0, heads, D_LATENT)
    assert tuple(lse.shape) == (0, heads)
    assert tuple(o_lo.shape) == (0, heads, D_LATENT)
    grads = cake_backend.backward(*_args(inp), out, o_lo, lse, inp.dout)
    assert tuple(grads[0].shape) == (0, heads, D_LATENT)
    assert tuple(grads[1].shape) == (0, heads, D_ROPE)
    assert not bool(grads[2].any()) and not bool(grads[3].any())
    assert not host.launches


# ---------------------------------------------------------------------------
# Device tests (96 and 128 heads on sm_100a / sm_103a / sm_107a)
# ---------------------------------------------------------------------------


def _device_supported() -> bool:
    return torch.cuda.is_available() and (
        torch.cuda.get_device_capability(0) in SUPPORTED_COMPUTE_CAPABILITIES
    )


def _require_heads(heads: int, *, backward: bool = True) -> None:
    """A device whose architecture serves ``heads`` query heads (skip otherwise); on ``HEAD_COUNT_ARCHES`` the
    registry must carry the head-count builds (a missing build fails the test instead of skipping it)."""
    if not _device_supported():
        pytest.skip("requires a compute capability 10.0 / 10.3 / 10.7 device")
    device = torch.device("cuda")
    if not cake_backend.generated_program_available(device, backward=backward):
        pytest.skip("generated DSA training program not registered for this device")
    arch = cake_backend.arch_for(device)
    if heads != NUM_HEADS and arch not in HEAD_COUNT_ARCHES:
        pytest.skip("96 and 128 query heads ship on sm_100a / sm_103a / sm_107a")
    cake_backend.head_count_stages(heads, arch, backward=backward)


def _globals(runner) -> list[int]:
    return [int(x) for x in runner.tensors["det_globals"].view(torch.int32).tolist()]


def _rope_outputs(forward=None, grads=None, **more) -> dict:
    found = {}
    if forward is not None:
        found.update(zip(("out", "lse", "o_lo"), forward, strict=True))
    if grads is not None:
        names = ("dq_latent", "dq_rope", "dkv_latent", "dk_rope")
        found.update((n, g) for n, g in zip(names, grads, strict=True) if g is not None)
    found.update(more)
    return found


def _some_invalid(inp) -> torch.Tensor:
    """The draw's global rows with ``-1`` and ``>= S`` slots anywhere, a run of ``-1`` and rows without a valid slot."""
    S = inp.total_k
    idx = inp.idx_global.clone()
    idx[::4, ::5] = -1
    idx[1::4, 3::7] = S + 5
    idx[2::4, 64:192] = -1
    idx[3::16] = -1
    return idx


@pytest.mark.parametrize("validity", ("all", "some"))
@pytest.mark.parametrize("heads", (96, 128))
def test_naive_tier_heads(heads, validity, record_property):
    """``out``, ``lse``, ``o_lo``, ``dq`` and the key gradients at ``heads`` query heads, scale ``192**-0.5``, top-k
    2048, judged against the FP64 oracle at the naive BF16 tier on every draw: the forward, and the backward with
    fresh BF16 key gradients and with the caller's ``dkv_acc``, each over 1 and 2 forced key passes."""
    _require_heads(heads)

    def case(s, j):
        inp = make_inputs([256], [4096], seed=s, topk=TOPK, heads=heads)
        idx = inp.idx_global if validity == "all" else _some_invalid(inp)
        args = (inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope, idx)
        fwd = cake_backend.forward(*args, softmax_scale=SCALE)
        out, lse, o_lo = fwd
        bwd = args + (out, o_lo, lse, inp.dout)
        runs = {"forward": _rope_outputs(fwd)}
        for p in (1, 2):
            grads = cake_backend.backward(*bwd, softmax_scale=SCALE, key_passes=p)
            runs[f"{p} pass"] = _rope_outputs(grads=grads)
            acc = torch.zeros(inp.total_k, D_QK, device="cuda")
            grads = cake_backend.backward(
                *bwd, softmax_scale=SCALE, key_passes=p, dkv_acc=acc
            )
            assert grads[2] is None and grads[3] is None
            runs[f"{p} pass dkv_acc"] = _rope_outputs(
                grads=grads, dkv_acc=nv.Acc(before=torch.zeros_like(acc), after=acc)
            )
        torch.cuda.synchronize()
        return nv.judged(nv.rope_problem(inp, indices=idx, scale=SCALE), runs)

    nv.judge_draws(
        record_property, case, seed=SEED + heads + (7 if validity == "some" else 0)
    )


@pytest.mark.parametrize("heads", (96, 128))
def test_heads_repeat_bit_for_bit(heads):
    """Above 64 heads every output repeats bit for bit: five autograd steps (fresh leaves), five eager backwards
    over a fresh forward (``o_lo_scratch`` alternating), and the dK / dV of 1, 2 and 3 key passes (one integer sum;
    dQ is summed through the FP32 partial between passes and follows the pass count)."""
    _require_heads(heads)
    inp = make_inputs([300, 212], [1500, 700], seed=SEED + 1, topk=256, heads=heads)
    args = (inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope, inp.idx_global)
    seen = None
    for _ in range(REPEATS):
        leaves = [t.detach().clone().requires_grad_(True) for t in args[:4]]
        out = cake_backend.cake_dsa_sparse_attention_det(
            *leaves, inp.idx_global, topk_length=inp.topk_length, softmax_scale=SCALE
        )
        out.backward(inp.dout)
        got = [out.detach()] + [leaf.grad for leaf in leaves]
        if seen is None:
            seen = got
            assert float(got[3].float().abs().sum()) > 0
        else:
            for a, b in zip(got, seen, strict=True):
                assert _same_bits(a, b)
    first = None
    for i in range(REPEATS):
        out, lse, o_lo = cake_backend.forward(
            *args, topk_length=inp.topk_length, softmax_scale=SCALE
        )
        assert _same_bits(out, seen[0])
        grads = cake_backend.backward(
            *args, out, o_lo, lse, inp.dout, topk_length=inp.topk_length,
            softmax_scale=SCALE, o_lo_scratch=bool(i % 2),
        )  # fmt: skip
        if first is None:
            first = grads
            for a, b in zip(grads, seen[1:], strict=True):
                assert _same_bits(a, b)
        for a, b in zip(grads, first, strict=True):
            assert _same_bits(a, b)
    out, lse, o_lo = cake_backend.forward(
        *args, topk_length=inp.topk_length, softmax_scale=SCALE
    )
    for p in (2, 3):
        grads = cake_backend.backward(
            *args, out, o_lo, lse, inp.dout, topk_length=inp.topk_length,
            softmax_scale=SCALE, key_passes=p,
        )  # fmt: skip
        assert _same_bits(grads[2], first[2]) and _same_bits(grads[3], first[3]), p


@pytest.mark.parametrize("heads", (96, 128))
def test_heads_runner_repeats_and_graph_replays_bit_for_bit(heads):
    """A prepared runner above 64 heads (``fwd_h128`` and the head-count backward) gives the same bits on every
    step, eagerly and replayed from a CUDA graph that captures forward and backward together; the step's bits are
    those of the eager entries."""
    _require_heads(heads)
    inp = make_inputs([256], [2048], seed=SEED + 2, topk=512, heads=heads)
    args = (inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope, inp.idx_global)
    runner = cake_backend.prepare_dsa_train(
        *args,
        topk_length=inp.topk_length,
        dout=inp.dout,
        o_lo_scratch=True,
        softmax_scale=SCALE,
    )
    assert runner.forward_stages == ("fwd_h128",) and runner.num_heads == heads
    first = [t.clone() for t in runner.step()]
    out, lse, o_lo = cake_backend.forward(
        *args, topk_length=inp.topk_length, softmax_scale=SCALE
    )
    eager = cake_backend.backward(
        *args,
        out,
        o_lo,
        lse,
        inp.dout,
        topk_length=inp.topk_length,
        softmax_scale=SCALE,
    )
    for a, b in zip(eager, first, strict=True):
        assert _same_bits(a, b)
    for _ in range(REPEATS - 1):
        for a, b in zip(runner.step(), first, strict=True):
            assert _same_bits(a, b)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        runner.step()
        with torch.cuda.graph(graph, stream=stream):
            runner.step()
    torch.cuda.current_stream().wait_stream(stream)
    for _ in range(REPEATS):
        graph.replay()
        torch.cuda.synchronize()
        grads = tuple(
            runner.tensors[n] for n in ("dq_latent", "dq_rope", "dkv_latent", "dk_rope")
        )
        for a, b in zip(grads, first, strict=True):
            assert _same_bits(a, b)


@pytest.mark.parametrize("heads", (96, 128))
def test_heads_forward_bits_do_not_depend_on_its_grid(monkeypatch, heads):
    """``fwd_h128`` runs the tokens of a persistent grid in a fixed per-token order: one cluster per token and a
    single cluster give the bits of the shipped grid (``2 * min(T, SMs // 2)`` CTAs)."""
    _require_heads(heads, backward=False)
    inp = make_inputs([300], [2048], seed=SEED + 3, topk=512, heads=heads)
    args = (inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope, inp.idx_global)
    T = inp.total_q

    def run():
        monkeypatch.setattr(cake_backend, "BINDING_CACHE", cake_backend._BindingCache())
        return cake_backend.forward(
            *args, topk_length=inp.topk_length, softmax_scale=SCALE
        )

    want = run()
    for grid in ((2 * T, 1, 1), (2, 1, 1)):
        monkeypatch.setattr(
            cake_backend, "single_pass_grid", lambda t, sms, grid=grid: grid
        )
        for a, b in zip(run(), want, strict=True):
            assert _same_bits(a, b), grid


@pytest.mark.parametrize("heads", (96, 128))
def test_heads_forced_redo_repeats_over_both_redo_grids(monkeypatch, heads):
    """With the grid 24 bits finer than the bound needs (``GRID_SHIFT = 24``, ``MIN_BUMP = 25``, not a shipped
    configuration: its accuracy is not judged) most rows overflow and the head-window redo recomputes them over both
    windows: no row is left flagged or non-finite, five steps give the same bits, and the redo's persistent grid
    gives the bits of a redo with a CTA per work item."""
    _require_heads(heads)
    inp = make_inputs([256], [1024], seed=SEED + 4, topk=256, heads=heads)
    args = (inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope, inp.idx_global)
    T = inp.total_q
    monkeypatch.setattr(cake_backend, "GRID_SHIFT", 24)
    monkeypatch.setattr(cake_backend, "MIN_BUMP", 25)

    def run():
        runner = cake_backend.prepare_dsa_train(
            *args,
            topk_length=inp.topk_length,
            dout=inp.dout,
            dkv_fp32=True,
            softmax_scale=SCALE,
        )
        runner.forward()
        return runner, [t.clone() for t in runner.backward()]

    runner, first = run()
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    assert tuple(runner.launches["bwd_redo_hw"].grid) == (min(2 * T, sms), 1, 1)
    words = _globals(runner)
    print("FORCED_REDO heads", heads, "flagged", words[cake_backend.REDO_WORD])
    assert words[cake_backend.REDO_WORD] > inp.total_k // 4
    assert words[cake_backend.NONFINITE_WORD] == 0
    assert all(torch.isfinite(t.float()).all() for t in first)
    for _ in range(REPEATS - 1):
        runner.forward()
        for a, b in zip(runner.backward(), first, strict=True):
            assert _same_bits(a, b)
    grids = dict(cake_backend.PERSISTENT_GRIDS)
    grids["bwd_redo_hw"] = lambda num_queries, num_kv, sm_count: 2 * num_queries
    monkeypatch.setattr(cake_backend, "PERSISTENT_GRIDS", grids)
    per_item, got = run()
    assert tuple(per_item.launches["bwd_redo_hw"].grid) == (2 * T, 1, 1)
    for a, b in zip(got, first, strict=True):
        assert _same_bits(a, b)


@pytest.mark.parametrize("heads", (96, 128))
def test_sm107a_device_plans_the_head_count_builds(heads):
    """On an sm_107a device a call over 96 or 128 heads binds the head-count builds on the device's SM count: the
    single-pass forward on ``2 * min(T, SMs // 2)`` CTAs, the heads delta one CTA per token, the head-window main one
    CTA per item (``2 T``) and the redo ``min(2 T, SMs)`` CTAs, both walking the items down from ``2 T - 1`` (one key
    pass), ``num_heads = H`` and ``wsel = -1``; its steps repeat bit for bit."""
    if (
        not _device_supported()
        or cake_backend.arch_for(torch.device("cuda")) != "sm_107a"
    ):
        pytest.skip("requires an sm_107a device")
    _require_heads(heads)
    inp = make_inputs([300], [2048], seed=SEED + 5, topk=512, heads=heads)
    args = (inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope, inp.idx_global)
    T = inp.total_q
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    runner = cake_backend.prepare_dsa_train(
        *args, topk_length=inp.topk_length, dout=inp.dout, softmax_scale=SCALE
    )
    assert runner.forward_stages == ("fwd_h128",) and runner.key_passes == 1
    grids = {k: tuple(v.grid) for k, v in runner.launches.items()}
    assert grids["fwd_h128"] == (2 * min(T, sms // 2), 1, 1)
    assert grids["bwd_delta_heads"] == (T, 1, 1)
    assert grids["bwd_main_hw"] == (2 * T, 1, 1)
    assert grids["bwd_redo_hw"] == (min(2 * T, sms), 1, 1)
    for stage in ("bwd_main_hw", "bwd_redo_hw"):
        values = runner.launches[stage].values
        assert (values["token_base"], values["token_step"]) == (2 * T - 1, -1), stage
        assert values["num_heads"] == heads and values["wsel"] == -1, stage
    first = [t.clone() for t in runner.step()]
    for _ in range(REPEATS - 1):
        for a, b in zip(runner.step(), first, strict=True):
            assert _same_bits(a, b)
