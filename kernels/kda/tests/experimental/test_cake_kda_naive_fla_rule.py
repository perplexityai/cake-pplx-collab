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

The verdict logic of the KDA rule on the CPU (``tests/cake_naive.py`` ``TierJudge`` and
``test_helpers/cake_kda_fla.py``): an output passes when it meets the naive BF16 tier or FLA under the same rule
(:data:`NEAR_TIE`, the max norm draw by draw, the zero rule), and fails only when it misses both; an output without an
FLA value (no FLA counterpart, or FLA unavailable) is judged against the naive tier only.

Each judged output here has two distinct values per draw (``[1, 2]`` scaled by a power of two per draw, so the draws
differ and the max norm is the norm judged draw by draw), and each side's error on a draw is its offset (one number for
both elements, or one per element) times that scale: the ratios of the errors are the ratios of the offsets.  The
offsets are dyadic, so every value is exact in FP32 and the naive values are exact in BF16.

The pin of FLA's autotuned kernels (``cake_kda_fla.FLA_PINS``) is checked here on stand-in FLA modules whose kernels
are real Triton autotuners (one under a real ``@triton.heuristics`` wrapper) over Python launches: the pin leaves each
its first config, so it launches that config with no benchmark; a first config other than its pin, or a kernel that
is not autotuned, fails the pin and changes nothing; an autotuner that chooses a config at run time fails the reference
after the call; and a forbidden variable, another FLA / Triton version or any error of the FLA call fails the reference
of a draw FLA runs (a :class:`FlaReferenceError`), while operands FLA does not run keep their naive-tier-only reason.
The record of the reference names the versions of FLA, Triton and torch, FLA's sources, the variables set and the ptxas
Triton runs; the steps of a test that calls FLA without recording them, in its own process and in a spawned rank, are
recorded for it.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import multiprocessing
import os
import platform
import sys
import types

import pytest
import torch

from tests.cake_naive import DRAWS, NO_FLA_COUNTERPART, REFERENCE, TierJudge, tier_line
from tests.experimental.test_helpers import cake_kda_fla
from tests.experimental.test_helpers import cake_kda_known_gaps as known_gaps
from tests.experimental.test_helpers.cake_kda_fla import (
    NOT_PINNED,
    PROPERTY,
    UNAVAILABLE,
    FlaReference,
    FlaReferenceError,
    FlaSteps,
    fla_line,
    fla_of,
    reference_line,
)

pytestmark = pytest.mark.cpu_only

BASE = torch.tensor([1.0, 2.0], dtype=torch.float64)


def _scale(j: int) -> float:
    return 2.0 ** (j - DRAWS // 2)


def _offset(x) -> torch.Tensor:
    return torch.tensor(x if isinstance(x, tuple) else (x, x), dtype=torch.float64)


def _judge(ours, tier, fla, name: str = "out") -> TierJudge:
    """A judge of output ``name`` over :data:`DRAWS` draws: on draw ``j`` the offsets ``ours(j)``, ``tier(j)`` and
    ``fla(j)`` (a float or a pair: FLA's offset; a string or None: FLA's reason, passed as it is) times the draw's
    scale."""
    judge = TierJudge(DRAWS)
    for j in range(DRAWS):
        s = _scale(j)
        want = BASE * s
        actual = (want + _offset(ours(j)) * s).to(torch.float32)
        naive = (want + _offset(tier(j)) * s).to(torch.bfloat16)
        assert torch.equal(naive.double(), want + _offset(tier(j)) * s), "the naive values are exact in BF16"
        other = fla(j)
        if isinstance(other, (float, tuple)):
            other = (want + _offset(other) * s).to(torch.float32)
        judge.add(name, j, actual, want, {"bf16": naive}, fla=other)
    return judge


def _const(x):
    return lambda j: x


def test_meets_the_naive_tier_only() -> None:
    judge = _judge(_const(0.0625), _const(0.125), _const(0.03125))
    (verdict,) = judge.verdicts().values()
    assert verdict["tier"] == "bf16"
    assert verdict["met"] == {"bf16": True, REFERENCE: False}
    assert verdict["wins"]["max"] == DRAWS and verdict["fla_wins"]["max"] == 0
    assert judge.failures() == []
    line = tier_line("t", verdict)
    assert line.startswith("TIER t out bf16 met=bf16 bf16=meets fla=misses draws=32 paired max 32/32 (fla 0/32) |")
    assert " bf16 " in line and " fla " in line and "(ours x0.5 / x0.5)" in line and "(ours x2 / x2)" in line


def test_meets_fla_only() -> None:
    judge = _judge(_const(0.1875), _const(0.125), _const(0.25))
    (verdict,) = judge.verdicts().values()
    assert verdict["tier"] == REFERENCE
    assert verdict["met"] == {"bf16": False, REFERENCE: True}
    assert judge.failures() == []
    line = tier_line("t", verdict)
    assert line.startswith("TIER t out fla met=fla bf16=misses fla=meets draws=32 paired max 0/32 (fla 32/32) |")
    assert "(ours x1.5 / x1.5)" in line and "(ours x0.75 / x0.75)" in line


def test_misses_both_fails_and_shows_both_references() -> None:
    judge = _judge(_const(0.375), _const(0.125), _const(0.25))
    (verdict,) = judge.verdicts().values()
    assert verdict["tier"] is None
    assert verdict["met"] == {"bf16": False, REFERENCE: False}
    (failure,) = judge.failures()
    assert failure.startswith("out: misses the naive tier and FLA over 32 inputs (draw by draw: ours within 1.005 x")
    assert "the tier on max 0 and within 1.005 x FLA on max 0 of 32 draws, at least 16 needed" in failure
    assert "bf16 " in failure and "fla " in failure
    assert "(ours x3 / x3)" in failure and "(ours x1.5 / x1.5)" in failure
    assert tier_line("t", verdict).startswith("TIER t out FAIL met=none bf16=misses fla=misses ")


def test_the_near_tie_is_the_rule_bound() -> None:
    # within 1.004 x the tier meets it; 1.006 x misses it (and ours equal to FLA's error meets FLA)
    met = _judge(_const(0.125 * 1.004), _const(0.125), _const(f"{UNAVAILABLE}: test"))
    assert next(iter(met.verdicts().values()))["tier"] == "bf16"
    missed = _judge(_const(0.125 * 1.006), _const(0.125), _const(f"{UNAVAILABLE}: test"))
    assert next(iter(missed.verdicts().values()))["tier"] is None
    equal = _judge(_const(0.125 * 1.006), _const(0.125), _const(0.125 * 1.006))
    assert next(iter(equal.verdicts().values()))["tier"] == REFERENCE


# ours and FLA with the same L2 error on every draw, ours ahead in the max norm on draws ``WIN`` and behind on the
# others: FLA is met when ours is within 1.005 x FLA's max error on at least half of the draws
P, Q = (0.1875, 0.25), (0.0, 0.3125)  # |P| = |Q| = 0.3125 (L2); max(P) = 0.25 < 0.3125 = max(Q) / 1.25


def _wins(win) -> TierJudge:
    return _judge(lambda j: P if win(j) else Q, _const(0.0625), lambda j: Q if win(j) else P)


@pytest.mark.parametrize("count", [15, 16, 17, 32])
def test_the_max_norm_is_judged_draw_by_draw_on_half_of_the_draws(count) -> None:
    assert abs(sum(x * x for x in P) - sum(x * x for x in Q)) < 1e-12
    judge = _wins(lambda j: j < count)
    (verdict,) = judge.verdicts().values()
    assert verdict["fla_wins"]["max"] == count and verdict["wins"]["max"] == 0
    assert verdict["tier"] == (REFERENCE if 2 * count >= DRAWS else None)


def test_fla_unavailable_judges_the_naive_tier_only() -> None:
    reason = f"{UNAVAILABLE}: the operands are on cpu, FLA runs on CUDA devices"
    passed = _judge(_const(0.0625), _const(0.125), _const(reason))
    (verdict,) = passed.verdicts().values()
    assert verdict["tier"] == "bf16" and verdict["met"] == {"bf16": True, REFERENCE: None}
    assert verdict["fla_reason"] == reason and REFERENCE not in verdict["norms"]["l2"]
    assert tier_line("t", verdict).endswith(f"| fla: {reason}, naive tier only")
    failed = _judge(_const(0.1875), _const(0.125), _const(reason))
    (verdict,) = failed.verdicts().values()
    assert verdict["tier"] is None
    (failure,) = failed.failures()
    assert failure.startswith("out: misses the naive tier over 32 inputs (draw by draw: ours within 1.005 x the tier")
    assert failure.endswith(f"fla: {reason}, so judged against the naive tier only")


def test_an_output_without_an_fla_counterpart_is_judged_against_the_naive_tier_only() -> None:
    passed = _judge(_const(0.0625), _const(0.125), _const(None), name="aqk_int")
    (verdict,) = passed.verdicts().values()
    assert verdict["tier"] == "bf16" and verdict["fla_reason"] == NO_FLA_COUNTERPART
    assert tier_line("t", verdict).endswith(f"| fla: {NO_FLA_COUNTERPART}, naive tier only")
    failed = _judge(_const(0.1875), _const(0.125), _const(None), name="aqk_int")
    assert next(iter(failed.verdicts().values()))["tier"] is None
    (failure,) = failed.failures()
    assert failure.startswith("aqk_int: misses the naive tier over 32 inputs")
    assert NO_FLA_COUNTERPART in failure


def test_fla_on_some_draws_only_is_not_a_reference() -> None:
    reason = f"{UNAVAILABLE}: the activated gate is outside FLA's safe gate range"
    judge = _judge(_const(0.1875), _const(0.125), lambda j: 0.25 if j else reason)
    (verdict,) = judge.verdicts().values()
    assert verdict["tier"] is None and verdict["met"][REFERENCE] is None
    assert verdict["fla_reason"] == f"{reason} (no FLA value on draws [0] of 32)"


def test_the_zero_rule_holds_for_each_reference() -> None:
    # the tier exact on draw 3 (ours not): the tier is missed whatever the other draws say; FLA meets
    judge = _judge(_const(0.0625), lambda j: 0.0 if j == 3 else 0.125, _const(0.25))
    (verdict,) = judge.verdicts().values()
    assert verdict["zero"] == {"bf16": [3], REFERENCE: []}
    assert verdict["met"] == {"bf16": False, REFERENCE: True} and verdict["tier"] == REFERENCE
    # FLA exact on draw 5 (ours not) and the tier missed: both missed, the message names both zero rules
    judge = _judge(_const(0.1875), lambda j: 0.0 if j == 7 else 0.125, lambda j: 0.0 if j == 5 else 0.25)
    (verdict,) = judge.verdicts().values()
    assert verdict["zero"] == {"bf16": [7], REFERENCE: [5]} and verdict["tier"] is None
    (failure,) = judge.failures()
    assert "tier bf16: error 0 on draws [7], ours not" in failure
    assert "fla: error 0 on draws [5], ours not" in failure
    # ours exact too on the draws where a reference is exact: no zero-rule miss
    judge = _judge(lambda j: 0.0 if j == 5 else 0.0625, _const(0.125), lambda j: 0.0 if j == 5 else 0.03125)
    (verdict,) = judge.verdicts().values()
    assert verdict["zero"] == {"bf16": [], REFERENCE: []} and verdict["tier"] == "bf16"


def test_a_non_finite_fla_value_is_not_met() -> None:
    def fla(j):
        return float("nan") if j == 2 else 0.25

    judge = _judge(_const(0.1875), _const(0.125), fla)
    (verdict,) = judge.verdicts().values()
    assert verdict["norms"]["l2"][REFERENCE] is None and verdict["met"][REFERENCE] is False
    assert verdict["tier"] is None


def test_verdicts_survive_json_and_print() -> None:
    # the KCP judge writes rank 0's verdicts as JSON; the TIER line of the read verdict is the same
    judge = _judge(_const(0.1875), _const(0.125), _const(0.25))
    (verdict,) = judge.verdicts().values()
    back = json.loads(json.dumps(verdict))
    assert tier_line("t", back) == tier_line("t", verdict)


def test_settle_passes_an_fla_met_output_and_fails_a_double_miss() -> None:
    met = _judge(_const(0.1875), _const(0.125), _const(0.25))
    known_gaps.settle(met.verdicts(), met.failures(), lengths=(64,))
    missed = _judge(_const(0.375), _const(0.125), _const(0.25))
    with pytest.raises(AssertionError, match="out: misses the naive tier and FLA over 32 inputs"):
        known_gaps.settle(missed.verdicts(), missed.failures(), lengths=(64,))


def test_fla_reference_values_and_reasons(monkeypatch) -> None:
    ddt = torch.arange(4 * 128, dtype=torch.float32)
    ref = FlaReference({"out": torch.zeros(3, 2, 128), "ddt_bias": ddt}, None, 0.5)
    assert ref.value("ddt_bias", torch.empty(4, 128)).shape == (4, 128)
    assert ref.value("out", torch.empty(3, 2, 128)).shape == (3, 2, 128)
    assert ref.value("aqk_int", torch.empty(3, 2, 128)) == "no FLA counterpart: FLA gives no aqk_int"
    gone = FlaReference(None, f"{UNAVAILABLE}: test")
    assert gone.value("out", torch.empty(1)) == f"{UNAVAILABLE}: test"
    steps = FlaSteps()
    steps.add(ref)
    steps.add(gone)
    steps.add(gone)
    # no pinned reference ran in the process: no record
    monkeypatch.setattr(cake_kda_fla, "_reference", None)
    want = {"calls": 3, "seconds": 0.5, "reasons": {f"{UNAVAILABLE}: test": 2}}
    assert steps.summary() == {**want, "reference": None}
    assert fla_line("t", steps.summary()) == f"FLA t calls=3 seconds=0.5 | {UNAVAILABLE}: test x2"
    # the record of the pinned reference that ran: in the summary and on its own line
    record = {"fla": "0.6.0", "triton": "3.8.0", "pin": "first-config x14"}
    monkeypatch.setattr(cake_kda_fla, "_reference", record)
    assert steps.summary() == {**want, "reference": record}
    assert fla_line("t", steps.summary()) == f"FLA t calls=3 seconds=0.5 | {UNAVAILABLE}: test x2"
    line = reference_line(steps.summary()["reference"])
    assert line == "FLA reference fla: 0.6.0; triton: 3.8.0; pin: first-config x14"


def _operands(device: str = "cpu", g=None) -> dict:
    S, H, D = 5, 2, 128
    return {
        "q": torch.zeros(S, H, D, dtype=torch.bfloat16, device=device),
        "k": torch.zeros(S, H, D, dtype=torch.bfloat16, device=device),
        "v": torch.zeros(S, H, D, dtype=torch.bfloat16, device=device),
        "g": torch.full((S, H, D), -1.0, device=device) if g is None else g,
        "beta": torch.zeros(S, H, dtype=torch.bfloat16, device=device),
        "dout": torch.zeros(S, H, D, dtype=torch.bfloat16, device=device),
        "cu_seqlens": torch.tensor([0, S], dtype=torch.int32, device=device),
    }


def test_fla_of_gives_reasons_where_fla_does_not_run(monkeypatch) -> None:
    # operands FLA does not run keep their reason (naive tier only), whether or not the pinned reference can run
    monkeypatch.setattr(cake_kda_fla, "_loaded", (None, f"{NOT_PINNED}: test"))
    got = fla_of(_operands(), 128**-0.5, -5.0, False)
    assert got.values is None and got.reason == f"{UNAVAILABLE}: the operands are on cpu, FLA runs on CUDA devices"
    # FLA present: operands on the CPU, and an activated gate outside FLA's safe gate range
    monkeypatch.setattr(cake_kda_fla, "_loaded", (object(), None))
    got = fla_of(_operands(), 128**-0.5, -5.0, False)
    assert got.values is None and got.reason == f"{UNAVAILABLE}: the operands are on cpu, FLA runs on CUDA devices"
    assert cake_kda_fla._check(_operands()) == got.reason
    g = torch.full((5, 2, 128), -1.0)
    g[2, 1, 7] = -6.0
    if torch.cuda.is_available():
        reason = cake_kda_fla._check(_operands("cuda", g.cuda()))
        assert reason.startswith(f"{UNAVAILABLE}: the activated gate is outside FLA's safe gate range [-5.0, 0.0]")


def test_fla_version_and_env(monkeypatch) -> None:
    # the reference is FLA 0.6.0's Triton chunk_kda with Triton 3.8.0, its 14 autotuned kernels pinned; the environment
    # of its import and calls is restored afterwards
    assert cake_kda_fla.FLA_VERSION == "0.6.0" and cake_kda_fla.TRITON_VERSION == "3.8.0"
    assert len(cake_kda_fla.FLA_PINS) == 14 and cake_kda_fla.FLA_PIN == "first-config x14"
    assert cake_kda_fla.FLA_ENV == {"FLA_FLASH_KDA": "0", "FLA_TILELANG": "0"}
    assert not set(cake_kda_fla.FORBIDDEN_ENV) & set(cake_kda_fla.FLA_ENV)
    assert not set(cake_kda_fla.FORBIDDEN_ENV) & set(cake_kda_fla.RECORDED_ENV)
    monkeypatch.setenv("FLA_FLASH_KDA", "1")
    monkeypatch.delenv("FLA_TILELANG", raising=False)
    with cake_kda_fla._environment(cake_kda_fla.FLA_ENV):
        assert os.environ["FLA_FLASH_KDA"] == "0" and os.environ["FLA_TILELANG"] == "0"
    assert os.environ["FLA_FLASH_KDA"] == "1" and "FLA_TILELANG" not in os.environ


def test_fla_of_fails_where_fla_runs_and_the_pinned_reference_cannot(monkeypatch) -> None:
    # FLA runs the operands (the operand check passes) and the pinned reference cannot: the test fails, the draw is
    # not judged against the naive tier only
    monkeypatch.setattr(cake_kda_fla, "_check", lambda inputs: None)
    monkeypatch.setattr(cake_kda_fla, "_loaded", (None, f"{NOT_PINNED}: test"))
    with pytest.raises(FlaReferenceError, match=f"^{NOT_PINNED}: test$"):
        fla_of(_operands(), 128**-0.5, -5.0, False)


def _raises(error: BaseException):
    """A stand-in for FLA's ``chunk_kda`` that raises ``error``."""

    def chunk_kda(*args, **kwargs):
        raise error

    return chunk_kda


def _out_of_resources():
    from triton.runtime.errors import OutOfResources

    return OutOfResources(262144, 232448, "shared")


# errors of the FLA call: a Triton 3.8 MLIR pass, a CUDA launch, out of memory, FLA's own input check, Triton's own
ERRORS = {
    "mlir": lambda: RuntimeError("PassManager::run failed"),
    "cuda": lambda: RuntimeError("Triton Error [CUDA]: an illegal memory access was encountered"),
    "oom": lambda: torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB"),
    "fla": lambda: ValueError("`lower_bound` must be in the safe range [-5, 0), got -6.0."),
    "triton": _out_of_resources,
}


@pytest.mark.parametrize("error", ERRORS)
def test_any_error_of_the_fla_call_fails_the_reference(monkeypatch, error) -> None:
    # FLA runs the operands (the operand check passes): an error of the call fails the test, the draw is not judged
    # against the naive tier only (the CPU operands enter a stand-in device scope)
    pytest.importorskip("triton")
    cause = ERRORS[error]()
    monkeypatch.setattr(cake_kda_fla, "_check", lambda inputs: None)
    monkeypatch.setattr(cake_kda_fla, "_loaded", (_raises(cause), None))
    monkeypatch.setattr(torch.cuda, "device", lambda device: contextlib.nullcontext())
    with pytest.raises(FlaReferenceError) as got:
        fla_of(_operands(), 128**-0.5, -5.0, False)
    assert str(got.value) == f"{NOT_PINNED}: {cake_kda_fla.FLA_ENTRY} raised {type(cause).__name__}: {cause}"
    assert got.value.__cause__ is cause


def test_operands_fla_does_not_run_keep_the_naive_tier_only(monkeypatch) -> None:
    # the operand check rejects them before any FLA call (here a call would raise): fla_of gives the reason, and the
    # judge judges the output against the naive tier only and names it
    monkeypatch.setattr(cake_kda_fla, "_loaded", (_raises(AssertionError("FLA was called")), None))
    on_gpu = types.SimpleNamespace(is_cuda=True, device="cuda:0", shape=(5, 2, 128), dim=lambda: 3)
    flat = types.SimpleNamespace(is_cuda=True, device="cuda:0", shape=(5, 256), dim=lambda: 2)
    g = torch.full((5, 2, 128), -1.0)
    g[2, 1, 7] = -6.0
    cases = {
        "the operands are on cpu, FLA runs on CUDA devices": _operands(),
        "operands of shape (5, 256), not [S, H, D]": {**_operands(), "q": flat},
        "the activated gate is outside FLA's safe gate range [-5.0, 0.0] (min -6.0, max -1.0, finite True)": {
            **_operands(g=g),
            "q": on_gpu,
        },
    }
    for want, inputs in cases.items():
        got = fla_of(inputs, 128**-0.5, -5.0, False)
        assert got.values is None and got.reason == f"{UNAVAILABLE}: {want}"
        judge = _judge(_const(0.0625), _const(0.125), _const(got.reason))
        (verdict,) = judge.verdicts().values()
        assert verdict["tier"] == "bf16" and verdict["met"] == {"bf16": True, REFERENCE: None}
        assert tier_line("t", verdict).endswith(f"| fla: {got.reason}, naive tier only")


def test_the_record_names_versions_sources_variables_and_the_ptxas_triton_runs(monkeypatch, tmp_path) -> None:
    triton = pytest.importorskip("triton")
    compiler = pytest.importorskip("triton.backends.nvidia.compiler")
    # a stand-in FLA package of two source files, a device of compute capability 10.3, and the "ptxas" Triton runs for
    # it (the Python interpreter here: its --version is one line)
    package = tmp_path / "site" / "fla"
    (package / "ops").mkdir(parents=True)
    (package / "__init__.py").write_text("__version__ = '0.6.0'\n")
    (package / "ops" / "kda.py").write_text("chunk_kda = None\n")
    (package / "ops" / "notes.txt").write_text("not a source\n")
    fla = types.ModuleType("fla")
    fla.__version__, fla.__file__ = "0.6.0", str(package / "__init__.py")
    utils = types.ModuleType("fla.utils")
    utils.get_all_max_shared_mem = lambda: [232448]
    monkeypatch.setitem(sys.modules, "fla", fla)
    monkeypatch.setitem(sys.modules, "fla.utils", utils)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (10, 3))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device=None: "Stand-in GPU")
    archs = []
    python = types.SimpleNamespace(path=sys.executable)
    monkeypatch.setattr(compiler, "get_ptxas", lambda arch: archs.append(arch) or python)
    for key in list(os.environ):
        if key.startswith(cake_kda_fla.RECORDED_PREFIXES):
            monkeypatch.delenv(key)
    monkeypatch.setenv("TRITON_CACHE_DIR", "/triton/cache")
    monkeypatch.setenv("PTXAS_STAND_IN", "1")
    monkeypatch.setenv("NOT_RECORDED", "1")
    with cake_kda_fla._environment(cake_kda_fla.FLA_ENV):
        record = cake_kda_fla._record(fla, triton)
    assert archs == [103]
    with open(sys.executable, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()
    assert record["ptxas_used"] == f"sm_103 {sys.executable} sha256 {digest} :: Python {platform.python_version()}"
    lines = "".join(
        f"{hashlib.sha256((tmp_path / 'site' / name).read_bytes()).hexdigest()}  {name}\n"
        for name in ("fla/__init__.py", "fla/ops/kda.py")
    )
    sources = hashlib.sha256(lines.encode()).hexdigest()
    assert record["fla_sources"] == f"sha256 {sources} over 2 .py files under {package}"
    assert record["torch"] == f"{torch.__version__} (CUDA {torch.version.cuda})"
    assert record["env"] == "FLA_FLASH_KDA=0 FLA_TILELANG=0 PTXAS_STAND_IN=1 TRITON_CACHE_DIR=/triton/cache"
    assert record["device"] == "Stand-in GPU sm_103" and record["max_shared_mem"] == 232448
    assert {"fla", "triton", "pin", "ptxas", "ptxas_blackwell", *cake_kda_fla.RECORDED_ENV} <= set(record)
    # a part of the record that cannot be read is recorded as such: it never fails the reference
    monkeypatch.setattr(compiler, "get_ptxas", lambda arch: types.SimpleNamespace(path=str(tmp_path / "absent")))
    assert cake_kda_fla._record(fla, triton)["ptxas_used"].startswith("not resolved (FileNotFoundError: ")


def _spawned_rank(record: dict, store: str) -> None:
    """The rank :func:`test_the_steps_of_a_test_include_its_spawned_ranks` spawns: rank 0 of a one-rank gloo group with
    the record of a pinned reference of its own, one FLA step without values (its operands are on the CPU)."""
    torch.distributed.init_process_group("gloo", init_method=f"file://{store}", rank=0, world_size=1)
    try:
        cake_kda_fla._reference = record
        fla_of(_operands(), 128**-0.5, -5.0, False)
    finally:
        torch.distributed.destroy_process_group()


def test_the_steps_of_a_test_include_its_spawned_ranks(monkeypatch, tmp_path) -> None:
    # a test that calls FLA without recording its steps: the steps of its own process and of the rank it spawned are
    # recorded for it, the record of the rank's reference among them
    monkeypatch.setattr(cake_kda_fla, "_reference", None)
    monkeypatch.setattr(cake_kda_fla, "_steps", None)
    record = {"fla": "0.6.0", "triton": "3.8.0", "pin": "first-config x14", "device": "rank 0's GPU"}
    reason = f"{UNAVAILABLE}: the operands are on cpu, FLA runs on CUDA devices"
    item = types.SimpleNamespace(user_properties=[("naive_tier", {})])
    with cake_kda_fla.steps_of_test(item):
        directory = os.environ[cake_kda_fla.STEPS_DIR]
        fla_of(_operands(), 128**-0.5, -5.0, False)
        rank = multiprocessing.get_context("spawn").Process(target=_spawned_rank, args=(record, str(tmp_path / "s")))
        rank.start()
        rank.join(600)
        assert rank.exitcode == 0
    assert not os.path.exists(directory) and os.environ.get(cake_kda_fla.STEPS_DIR) != directory
    processes = [f"pid {os.getpid()}", f"rank 0 (pid {rank.pid})"]
    steps = {"calls": 2, "seconds": 0.0, "reasons": {reason: 2}, "reference": record, "processes": processes}
    assert item.user_properties == [("naive_tier", {}), (PROPERTY, steps)]
    assert fla_line("t", steps) == f"FLA t calls=2 seconds=0.0 ({', '.join(processes)}) | {reason} x2"
    # a test that records its FLA steps itself keeps them; a test that calls no FLA gets none
    item = types.SimpleNamespace(user_properties=[])
    with cake_kda_fla.steps_of_test(item):
        own = FlaSteps()
        own.add(fla_of(_operands(), 128**-0.5, -5.0, False))
        own.record(lambda key, value: item.user_properties.append((key, value)))
    assert item.user_properties == [(PROPERTY, own.summary())]
    item = types.SimpleNamespace(user_properties=[])
    with cake_kda_fla.steps_of_test(item):
        pass
    assert item.user_properties == []


def test_the_summary_prints_the_record_of_every_process(monkeypatch) -> None:
    one, two = {"fla": "0.6.0", "device": "A"}, {"fla": "0.6.0", "device": "B"}
    processes = ["rank 0 (pid 1)", "rank 1 (pid 2)"]
    steps = {"calls": 3, "seconds": 0.5, "reasons": {}, "reference": one, "processes": processes}
    recorded = [(PROPERTY, {**steps, "other_references": [two]})]
    report = types.SimpleNamespace(when="call", nodeid="t", user_properties=recorded)
    lines = []
    reporter = types.SimpleNamespace(stats={"passed": [report]}, section=lines.append, write_line=lines.append)
    cake_kda_fla.summarize_fla(reporter)
    line = "FLA t calls=3 seconds=0.5 (rank 0 (pid 1), rank 1 (pid 2))"
    assert lines[1:] == [reference_line(one), reference_line(two), line]


@pytest.mark.parametrize("key", cake_kda_fla.FORBIDDEN_ENV)
def test_a_forbidden_variable_fails_the_reference(monkeypatch, key) -> None:
    # checked before FLA is imported: nothing of FLA runs
    monkeypatch.setattr(cake_kda_fla, "_loaded", None)
    for other in cake_kda_fla.FORBIDDEN_ENV:
        monkeypatch.delenv(other, raising=False)
    monkeypatch.setenv(key, "1")
    chunk_kda, reason = cake_kda_fla.load()
    assert chunk_kda is None
    assert reason == f"{NOT_PINNED}: RuntimeError: {key} set: each changes FLA's or Triton's code or numbers"
    monkeypatch.setattr(cake_kda_fla, "_check", lambda inputs: None)
    with pytest.raises(FlaReferenceError, match=key):
        fla_of(_operands(), 128**-0.5, -5.0, False)


@pytest.mark.parametrize("installed", [("0.5.0", "3.8.0"), ("0.6.0", "3.7.0")])
def test_another_fla_or_triton_version_fails_the_reference(monkeypatch, installed) -> None:
    triton = pytest.importorskip("triton")
    fla_version, triton_version = installed
    # stand-in FLA packages of version fla_version; the installed Triton reports triton_version
    fake = {name: types.ModuleType(name) for name in ("fla", "fla.ops", "fla.ops.kda")}
    fake["fla"].__version__ = fla_version
    fake["fla.ops.kda"].chunk_kda = object()
    for name, module in fake.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(triton, "__version__", triton_version)
    for key in cake_kda_fla.FORBIDDEN_ENV:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(cake_kda_fla, "_loaded", None)
    chunk_kda, reason = cake_kda_fla.load()
    assert chunk_kda is None
    assert reason == (
        f"{NOT_PINNED}: RuntimeError: fla {fla_version} and triton {triton_version} are installed, "
        "the reference is fla 0.6.0 with triton 3.8.0"
    )


class _Launch:
    """A stand-in for a ``@triton.jit`` kernel under an autotuner: ``fn`` is its Python function (as a JIT function's
    is), and each launch records the keyword arguments it gets (the config's among them)."""

    def __init__(self) -> None:
        self.fn, self.launches = _kernel, []

    def run(self, *args, **kwargs) -> None:
        self.launches.append(kwargs)


def _kernel(x):
    return x


def _stand_in_fla(monkeypatch, lists: dict) -> dict:
    """Stand-in FLA modules ``fla.pin_test_<name>``, each holding kernel ``<name>``: a real Triton autotuner over the
    configs ``lists[name]`` (its benchmark records the configs it times and makes the last one the fastest), the first
    under a real ``@triton.heuristics`` wrapper.  :data:`cake_kda_fla.FLA_PINS` becomes their first configs.  Returns
    ``name -> autotuner``."""
    from triton.runtime.autotuner import Autotuner, Heuristics

    class Tuner(Autotuner):
        def _bench(self, *args, config, **meta):
            self.benchmarked.append(str(config))
            return [-float(len(self.benchmarked))] * 3

    tuners, pins = {}, {}
    for i, (name, configs) in enumerate(lists.items()):
        tuner = Tuner(_Launch(), ["x"], configs, ["x"], None, None)
        tuner.benchmarked = []
        module = types.ModuleType(f"fla.pin_test_{name}")
        setattr(module, name, Heuristics(tuner, ["x"], {}) if i == 0 else tuner)
        monkeypatch.setitem(sys.modules, module.__name__, module)
        tuners[name] = tuner
        pins[(module.__name__, name)] = str(configs[0])
    monkeypatch.setattr(cake_kda_fla, "FLA_PINS", pins)
    return tuners


def _lists() -> dict:
    triton = pytest.importorskip("triton")
    return {
        "cumsum": [triton.Config({"BS": 32}, num_warps=2), triton.Config({"BS": 64}, num_warps=4)],
        "gate_bwd": [triton.Config({}, num_warps=4, num_stages=2), triton.Config({}, num_warps=8, num_stages=3)],
    }


def test_the_pin_leaves_each_fla_autotuner_its_first_config(monkeypatch) -> None:
    lists = _lists()
    tuners = _stand_in_fla(monkeypatch, lists)
    cumsum = sys.modules["fla.pin_test_cumsum"].cumsum  # the @triton.heuristics wrapper
    assert cake_kda_fla._autotuner(cumsum) is tuners["cumsum"]
    # unpinned, an autotuner times its configs and launches the fastest (the last here); the check after a call fails
    before = cake_kda_fla._cache_sizes()
    cumsum.run(1)
    assert tuners["cumsum"].benchmarked == [str(c) for c in lists["cumsum"]]
    assert tuners["cumsum"].fn.launches[-1]["BS"] == 64
    with pytest.raises(FlaReferenceError, match="chose their config at run time: fla.pin_test_cumsum.cumsum$"):
        cake_kda_fla._assert_pinned(before)
    # pinned: one config each, the cache cleared
    cake_kda_fla._pin()
    for name, tuner in tuners.items():
        assert tuner.configs == [lists[name][0]] and tuner.cache == {}
        tuner.benchmarked.clear()
    # each launches its first config with no benchmark and fills no cache; the check after the calls passes
    before = cake_kda_fla._cache_sizes()
    cumsum.run(2)
    sys.modules["fla.pin_test_gate_bwd"].gate_bwd.run(3)
    cake_kda_fla._assert_pinned(before)
    for name, tuner in tuners.items():
        launch = tuner.fn.launches[-1]
        first = lists[name][0]
        assert {k: launch[k] for k in first.all_kwargs()} == first.all_kwargs()
        assert tuner.benchmarked == [] and tuner.cache == {}


def test_the_pin_refuses_a_first_config_that_is_not_its_pin(monkeypatch) -> None:
    lists = _lists()
    tuners = _stand_in_fla(monkeypatch, lists)
    pins = dict(cake_kda_fla.FLA_PINS)
    pins[("fla.pin_test_gate_bwd", "gate_bwd")] = str(lists["gate_bwd"][1])
    pins[("fla.pin_test_cumsum", "_kernel_absent")] = str(lists["cumsum"][0])
    monkeypatch.setattr(cake_kda_fla, "FLA_PINS", pins)
    with pytest.raises(RuntimeError) as raised:
        cake_kda_fla._pin()
    assert str(raised.value) == (
        "fla.pin_test_gate_bwd.gate_bwd: the first config is "
        "'num_warps: 4, num_ctas: 1, num_stages: 2, maxnreg: None', "
        "the pin 'num_warps: 8, num_ctas: 1, num_stages: 3, maxnreg: None'; "
        "fla.pin_test_cumsum._kernel_absent is not an autotuned Triton kernel"
    )
    # no autotuner is changed when a pin fails
    for name, tuner in tuners.items():
        assert tuner.configs == lists[name]
