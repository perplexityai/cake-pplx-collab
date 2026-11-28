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

Independent single-GPU fixtures for the MoK kernel tests.
Multi-GPU MoK tests: EP1, EP4 and EP8 groups on one node, each test starting its own processes
(:func:`tests.experimental._mok_spawn.run_ranks`; run with the ``SLURM_*`` task variables unset).  An item skips when
this process sees fewer GPUs than its group has ranks (the ``ep8`` items on a node of four GPUs).

The workers live in :mod:`tests.experimental._mok_dist`: the complete training step eager and under CUDA Graph
replay, unequal source counts per rank (empty ranks included), the training features (clamped SwiGLU, recompute and
the context-free forward, FP32 weight-gradient accumulation, native MXFP8), the adapter's source-length guards and
scratch lifetime, and the router weight before the down projection through the adapter on SM100a, SM103a and
SM107a: at EP1 and the diagnostic EP4 layout, and on the requester's inputs at the 384-expert layouts (4, 96) and
(8, 48) with top-6 over changing source counts, chosen placements, a full skew at the requester's no-overflow
multiplier, the trap of a schedule past its capacity, the bytes of separate group launches and the bytes of the
forward source split; and the 512-expert top-10 group of the explicit modes (``router_weight_gradient="direct"``,
``shared_expert="scaled"``) at EP8 ``(8, 64)`` over changing source counts on SM100a, SM103a and SM107a (inputs drawn on
the CPU).
Each worker checks the contract and repeatability in every rank; the accuracy verdicts of each case (the whole
group's outputs against the float64 oracle beside the naive BF16 implementation) are recorded here and
printed as ``TIER`` lines.  The known accuracy gaps (:mod:`tests.experimental._mok_gaps`) are judged in items of their
own under a strict expected-failure marker, from the same group run: the BF16 routed weight gradients of the cases
whose experts' rows cross a macrobatch boundary and miss the tier, and the EP1 MXFP8 routed gate weight gradient with
FP32 accumulation, which misses the tier in the max norm.

Backend suite: the package is imported only through :func:`tests.cake.test_helpers.cake_package.cake_package`.
"""

import pytest
import torch

from tests.cake.test_helpers.cake_package import cake_package
from tests.experimental import _mok_dist as workers
from tests.experimental import _mok_gaps as gaps
from tests.experimental._mok_spawn import run_ranks, run_ranks_isolated

kernels = cake_package("cake_mok_bf16", "_kernels")


def _run(world, worker, *args):
    """The group run of ``worker(*args)`` over ``world`` ranks, shared by the items of one case."""
    return gaps.judged((worker.__name__, world, *args), lambda: run_ranks(world, worker, *args))


def _record(results, record_property, part=gaps.OUTPUTS, known=()):
    """Record and judge rank 0's accuracy verdicts (every rank judged the same gathered values), output names
    ``<case>:<output>``: item ``part`` of :func:`tests.experimental._mok_gaps.check`."""
    verdicts, failures = [], []
    for record in results[0]:
        if "judged" not in record:
            continue
        case_verdicts, alternatives, misses = record["judged"]
        verdicts += [dict(verdict, output=f"{record['case']}:{verdict['output']}") for verdict in case_verdicts]
        if part == gaps.OUTPUTS:
            for verdict in alternatives:
                record_property("naive_alternative", dict(verdict, output=f"{record['case']}:{verdict['output']}"))
        failures += [f"{record['case']}:{miss}" for miss in misses]
    gaps.check(record_property, part, verdicts, failures, known)


def _cases(function, world_gaps):
    """``(world, part)`` parameters of test ``function``: per world the main item, then each known-gap output under its
    marker."""
    return [
        pytest.param(world, *p.values, marks=p.marks, id=p.id)
        for world, known in world_gaps.items()
        for p in gaps.parts(function, known, f"ep{world}")
    ]


@pytest.mark.parametrize("world", [1, 4, 8], ids=["ep1", "ep4", "ep8"])
def test_training_graph(record_property, world):
    _record(_run(world, workers.training_graph), record_property)


#: The known-gap outputs of the unequal-source cases: the 0 / 513 / 769 / 8193 case's experts cross macrobatch
#: boundaries (BF16 routed dW).
UNEQUAL_GAPS = tuple(
    (f"counts-0-513-769-8193:{name}", gaps.bf16_straddle_dw)
    for name in ("dw_routed_gate", "dw_routed_up", "dw_routed_down")
)


@pytest.mark.parametrize("part", gaps.parts("test_unequal_sources", UNEQUAL_GAPS))
def test_unequal_sources(record_property, part):
    _record(_run(4, workers.unequal_sources), record_property, part, [name for name, _ in UNEQUAL_GAPS])


#: The known-gap outputs of the training features per world: at EP1 (1031 tokens, 768-row macrobatches: experts cross a
#: macrobatch boundary) the BF16 routed dW that misses the tier, the down projection of the clamped cases and all three
#: projections of the plain recompute case; and the gate projection's dW of the MXFP8 case with FP32 weight gradients,
#: which misses the max norm (its L2 norm meets the tier).  The MXFP8 score gradient of these cases meets the tier.
FEATURE_GAPS = {
    1: (
        ("bf16-clamped:dw_routed_down", gaps.bf16_straddle_dw),
        ("bf16-recompute:dw_routed_gate", gaps.bf16_straddle_dw),
        ("bf16-recompute:dw_routed_up", gaps.bf16_straddle_dw),
        ("bf16-recompute:dw_routed_down", gaps.bf16_straddle_dw),
        ("bf16-clamped-recompute:dw_routed_down", gaps.bf16_straddle_dw),
        ("mxfp8-fp32-wgrad:dw_routed_gate", gaps.mxfp8_fp32_dw_gate),
    ),
    4: (),
    8: (),
}


@pytest.mark.parametrize("world,part", _cases("test_training_features", FEATURE_GAPS))
def test_training_features(record_property, world, part):
    results = _run(world, workers.training_features)
    if part == gaps.OUTPUTS:
        assert {r["case"] for r in results[0]} == {c["name"] for c in workers.FEATURE_CASES}
    _record(results, record_property, part, [name for name, _ in FEATURE_GAPS[world]])


@pytest.mark.parametrize("world", [1, 4], ids=["ep1", "ep4"])
def test_requantized_training(world):
    """``mxfp8_transposed="requantized"`` through the adapter (:func:`tests.experimental._mok_dist.requantized_training`);
    skipped on a target without its kernels."""
    if torch.cuda.is_available():
        arch = cake_package("cake_mok_bf16", "jit").target_arch()
        if arch not in kernels.MXFP8_REQUANTIZED_TARGETS:
            pytest.skip(f"mxfp8_transposed='requantized' has no {arch} build")
    results = run_ranks(world, workers.requantized_training)
    assert [r["case"] for r in results[0]] == [
        c["name"] for c in workers.REQUANTIZED_CASES
    ]


def test_source_capacity_and_context_ownership():
    run_ranks(4, workers.source_capacity)


def test_backward_releases_launcher_scratch_before_epilogue():
    run_ranks(4, workers.backward_scratch)


def test_source_capacity_and_context_ownership_ep8():
    run_ranks(8, workers.source_capacity)


def test_backward_releases_launcher_scratch_before_epilogue_ep8():
    run_ranks(8, workers.backward_scratch)


#: The 384-expert layouts (EP, local experts) of the router weight before the down projection with peers on one node,
#: by id: top-6 over 96 local experts at EP4, over 48 at EP8.
TOP6_LAYOUTS = {"ep4-local96": (4, 96), "ep8-local48": (8, 48)}
#: The part of the error text a CUDA kernel trap raises at the next CUDA call that torch's form ("CUDA error: ...") and
#: the launchers' form ("<kernel>: cudaLaunchKernelExC failed: ...") share.
TRAP_ERROR = "unspecified launch failure"
#: Seconds the trapping group may run before its processes are stopped.
OVERFLOW_TIMEOUT = 600


def _require_before_down(world):
    """Skip before any process starts unless every rank's device has the router-weight-before-down builds
    (``BEFORE_DOWN_TARGETS``: SM100a, SM103a and SM107a have them)."""
    if torch.cuda.is_available() and torch.cuda.device_count() >= world:
        for index in range(world):
            capability = torch.cuda.get_device_capability(index)
            arch = gaps.TARGETS.get(capability, f"sm_{capability[0]}{capability[1]}")
            if arch not in kernels.BEFORE_DOWN_TARGETS:
                pytest.skip(f"The router weight before the down projection has no {arch} build")


@pytest.mark.parametrize(
    "world,local_experts", [(1, None), (4, 4), *TOP6_LAYOUTS.values()], ids=["ep1", "ep4-local4", *TOP6_LAYOUTS]
)
def test_before_down_training(record_property, world, local_experts):
    """``router_weight_position="before_down"`` through the adapter over changing source counts on one workspace
    (padded and unpadded staging, an empty source rank with peers, no kernel load after the first count) at the
    requester's no-overflow multiplier: contract, repeatability, graph replay, recompute, the context-free forward,
    another communication-SM split per rank, the unplanned launches and a test-local autograd function with the saved
    context's bytes (another router weight in the workspace between its launches); every count judged.  EP1 and the
    diagnostic EP4 layout at top-2; the 384-expert layouts at top-6 on the requester's inputs."""
    _require_before_down(world)
    results = _run(world, workers.before_down_training, local_experts)
    assert [r["case"] for r in results[0]] == [
        "bf16-clamped-before-down-counts-" + "-".join(map(str, counts)) for counts in workers.before_down_steps(world)
    ]
    _record(results, record_property)


def _record_padding(results, record_property):
    """Record, per rank and case, whether the padding rows of the routed context fields of the recompute equal the
    saved context's (reported, not required)."""
    record_property(
        "padding_rows_equal",
        {rank: {r["case"]: r["padding_rows_equal"] for r in records} for rank, records in enumerate(results)},
    )


#: ``(world, local_experts, macro)`` of the placement items: 768-row macrobatches, or one resident macrobatch (None).
PLACEMENT_CASES = [
    pytest.param(world, local, macro, id=f"{name}-{ident}")
    for name, (world, local) in TOP6_LAYOUTS.items()
    for ident, macro in (("m768", 768), ("m1", None))
]
#: ``(world, local_experts, target, macro)`` of the full-skew items: every route to rank 0's experts, or to the last
#: rank's; 768-row macrobatches, or one resident macrobatch (None, ``m1``).
SKEW_CASES = [
    pytest.param(world, local, target, macro, id=f"{name}-rank{target}{suffix}")
    for name, (world, local) in TOP6_LAYOUTS.items()
    for target in (0, world - 1)
    for suffix, macro in (("", 768), ("-m1", None))
]


@pytest.mark.parametrize("world,local_experts,macro", PLACEMENT_CASES)
def test_before_down_placement(record_property, world, local_experts, macro):
    """The requester's inputs at a 384-expert layout over chosen placements at the no-overflow multiplier, with
    768-row macrobatches (``m768``) or one resident macrobatch (``m1``): a step mixing tokens whose experts are all on
    one rank, spread over the ranks, and random; a step without routes to the last rank's experts (its routed weight
    gradients exactly zero) and without tokens on rank 0 (empty outputs, shared weight gradients exactly zero).  Each
    step: every route held once in the schedules, contract, five eager launches and five graph replays with the same
    bytes, the recompute's context against the saved one (shared fields whole, routed fields on the scheduled rows of
    the resident macrobatch) and backward from each; judged."""
    _require_before_down(world)
    results = _run(world, workers.before_down_placement, local_experts, macro)
    assert [r["case"] for r in results[0]] == [
        "bf16-clamped-before-down-placement-mixed",
        "bf16-clamped-before-down-placement-no-route-to-last-rank",
    ]
    _record_padding(results, record_property)
    _record(results, record_property)


@pytest.mark.parametrize("world,local_experts,target,macro", SKEW_CASES)
def test_before_down_full_skew(record_property, world, local_experts, target, macro):
    """Every route of the group on one rank's experts (rank 0 or the last), with the route counts per expert that pad
    the most, at the requester's no-overflow multiplier (the capacity holds the worst case, checked on the host): the
    target rank's schedule holds exactly the padded rows, no trap; contract, repeatability, graph replay, the
    recompute's context against the saved one (on the resident macrobatch: the first 768 rows, or with one resident
    macrobatch, ``m1``, the whole schedule); judged."""
    _require_before_down(world)
    results = _run(world, workers.before_down_full_skew, local_experts, target, macro)
    (record,) = results[0]
    assert record["case"] == f"bf16-clamped-before-down-full-skew-rank{target}"
    assert results[target][0]["num_tokens"] == record["padded_rows"]
    _record_padding(results, record_property)
    _record(results, record_property)


@pytest.mark.parametrize("world,local_experts", list(TOP6_LAYOUTS.values()), ids=list(TOP6_LAYOUTS))
def test_before_down_overflow_traps(record_property, world, local_experts):
    """The full skew to the last rank on a workspace at the package's default multiplier (capacity below the target's
    padded rows, checked on the host), in a process session of its own: the group fails before the timeout, the
    target rank with the kernel trap raised at the next CUDA call (inside ``build_schedule`` or at the synchronization
    after it; the text is recorded as ``trap_error``), no rank returns outputs, and the processes left waiting are
    stopped.  Then a new group step on the same GPUs passes."""
    _require_before_down(world)
    target = world - 1
    run = run_ranks_isolated(world, workers.before_down_overflow, local_experts, target, timeout=OVERFLOW_TIMEOUT)
    reports = {rank: record[0] for rank, record in enumerate(run["records"]) if record is not None}
    record_property("overflow_run", dict(returncode=run["returncode"], timed_out=run["timed_out"], reports=reports))
    assert not run["timed_out"], run["log"][-4000:]
    assert run["returncode"] != 0, run["log"][-4000:]
    assert not [rank for rank, kind in reports.items() if kind == "ok"], (reports, "a rank returned outputs")
    kind, text = run["records"][target] or (None, run["log"][-4000:])
    assert kind == "error" and workers.SCHEDULE_TRAPPED in text, text
    trap = text[text.rindex(workers.SCHEDULE_TRAPPED) :].strip()
    record_property("trap_error", trap)
    assert TRAP_ERROR in trap, text
    run_ranks(world, workers.before_down_small_step, local_experts)


@pytest.mark.parametrize("world,local_experts", list(TOP6_LAYOUTS.values()), ids=list(TOP6_LAYOUTS))
def test_before_down_run_to_run(record_property, world, local_experts):
    """Two separate group launches (new processes, workspaces and launch plans) of the placement mix and of a full
    skew at the no-overflow multiplier: every output of every rank has the same SHA-256 in both, and each launch
    repeats its bytes within its process."""
    _require_before_down(world)
    launches = [run_ranks(world, workers.before_down_digests, local_experts) for _ in range(2)]
    assert launches[0] == launches[1]
    record_property("digests", launches[0])


@pytest.mark.parametrize("world,local_experts", list(TOP6_LAYOUTS.values()), ids=list(TOP6_LAYOUTS))
def test_before_down_source_split_bitwise(record_property, world, local_experts):
    """The forward source split (``source_split_comm_sms``) on the requester's inputs at a 384-expert layout with one
    or two source-rich ranks and an empty rank, over random routes, the placement mix and a full skew: each split
    (several counts and thresholds, per-rank counts, a threshold no rank reaches) gives every rank the bytes of the
    split off, in the context and the context-free forward, and a CUDA Graph captured with the split replays them
    after new routes are loaded."""
    _require_before_down(world)
    results = run_ranks(world, workers.before_down_source_split, local_experts)
    assert [r["case"] for r in results[0]] == [
        f"bf16-clamped-before-down-source-split-{name}"
        for name in ("random", "placement-mixed", "full-skew-last")
    ]
    record_property("rich_ranks", results[0][0]["rich"])


def _require_scaled_group(world):
    """Skip before any process starts unless every rank's device has the 512-expert top-10 group's builds
    (``MODE_GROUPS["scaled_top10"].targets``: SM100a, SM103a and SM107a) and they are in this package's registry
    (``test_mok_bf16.py`` fails while they are not)."""
    group = kernels.MODE_GROUPS["scaled_top10"]
    registry = cake_package("cake_mok_bf16", "jit")._registry()
    if any(
        role not in registry
        for role in (
            "backward_fp32_direct",
            "epilogue_forward_scaled_10",
            "shared_scale_dot",
        )
    ):
        pytest.skip(
            "The 512-expert top-10 group's kernels are not in this package's registry"
        )
    if torch.cuda.is_available() and torch.cuda.device_count() >= world:
        for index in range(world):
            capability = torch.cuda.get_device_capability(index)
            arch = gaps.TARGETS.get(capability, f"sm_{capability[0]}{capability[1]}")
            if arch not in group.targets:
                pytest.skip(f"The 512-expert top-10 group has no {arch} build")


def test_scaled_group_training(record_property):
    """The 512-expert top-10 group (``router_weight_gradient="direct"``, ``shared_expert="scaled"``) through the
    adapter at EP8 ``(8, 64)`` over changing source counts on one workspace (two empty source ranks in the last step),
    on inputs drawn on the CPU: contract, repeatability, graph replay, the checkpointed step with the context step's
    bytes, no kernel load after the first count; every count's whole-group ``y``, ``dx``, router-weight gradients,
    ``d_shared_scale`` and weight gradients judged."""
    world = kernels.MODE_GROUPS["scaled_top10"].layouts[0][0]
    _require_scaled_group(world)
    results = _run(world, workers.scaled_group_training)
    assert [r["case"] for r in results[0]] == [
        "bf16-scaled-direct-top10-counts-" + "-".join(map(str, counts))
        for counts in workers.before_down_steps(world)
    ]
    _record(results, record_property)


#: The layouts of each MODE_GROUPS group after its first (one GPU: the scheduler of rank 0 alone), by id.
ADDED_GROUP_LAYOUTS = {
    f"{name}-ep{ep}-local{experts}": (ep, experts, group.topk)
    for name, group in kernels.MODE_GROUPS.items()
    for ep, experts in group.layouts[1:]
}


@pytest.mark.parametrize(
    "ep,local_experts,topk",
    list(ADDED_GROUP_LAYOUTS.values()),
    ids=list(ADDED_GROUP_LAYOUTS),
)
def test_group_layout_overflow_traps(record_property, ep, local_experts, topk):
    """The scheduler of a group's later layout as rank 0 on one GPU (the gathered ids given), at the full skew to rank
    0 with a schedule capacity one 256-row tile below rank 0's padded rows, in a process session of its own: the rows
    kernel traps and the next CUDA call raises (the text is recorded as ``trap_error``) before the timeout.  Then the
    same skew at exactly the padded rows passes and schedules every padded row."""
    registry = cake_package("cake_mok_bf16", "jit")._registry()
    for role in (f"count_{ep}_{local_experts}", f"rows_{ep}_{local_experts}"):
        if role not in registry:
            pytest.skip(f"{role} is not in this package's registry")
    if torch.cuda.is_available() and torch.cuda.device_count() >= 1:
        capability = torch.cuda.get_device_capability(0)
        arch = gaps.TARGETS.get(capability, f"sm_{capability[0]}{capability[1]}")
        if arch not in registry[f"rows_{ep}_{local_experts}"]["arches"]:
            pytest.skip(f"rows_{ep}_{local_experts} has no {arch} build")
    run = run_ranks_isolated(
        1,
        workers.group_layout_overflow,
        ep,
        local_experts,
        topk,
        1,
        timeout=OVERFLOW_TIMEOUT,
    )
    record_property(
        "overflow_run", {"returncode": run["returncode"], "timed_out": run["timed_out"]}
    )
    assert not run["timed_out"], run["log"][-4000:]
    assert run["returncode"] != 0, run["log"][-4000:]
    kind, text = run["records"][0] or (None, run["log"][-4000:])
    assert kind == "error" and workers.SCHEDULE_TRAPPED in text, text
    trap = text[text.rindex(workers.SCHEDULE_TRAPPED) :].strip()
    record_property("trap_error", trap)
    assert TRAP_ERROR in trap, text
    (fits,) = run_ranks(1, workers.group_layout_overflow, ep, local_experts, topk, 0)
    assert fits["rows"] == fits["padded"] == fits["capacity"], fits
