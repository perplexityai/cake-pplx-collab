# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION
"""The known gaps (``tests/known_gaps.py``) and what their xfail covers.  CPU only: nothing
is launched.

* The table names collected tests only, of known GPU models (the suites are collected in
  a subprocess that sees no GPU), and caps both max-norm miss ratios of each output of a
  gap.
* The xfail covers a test's known miss only, each ratio at most its cap
  (``known_gaps.covers``): the failures are those ``judge_draws`` raises on CPU tensors
  whose errors against their oracle are set by construction, and plain failures.
* The conftest's fixture and hook apply that rule: a pytest run of synthetic suites in a
  subprocess, whose device is named as a GPU model of the table (no GPU is used).
* No test checks anything after its judgment (a static check of the suites' source), so
  a miss of the naive tier never skips a check of its test.
"""

from __future__ import annotations

import ast
import math
import os
import re
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch

from tests import known_gaps
from tests.cake.test_helpers.cake_moe_judge import (
    MISS,
    NaiveTierMiss,
    judge_draws,
    raise_misses,
)
from tests.cake.test_helpers.cake_naive import NEAR_TIE

pytestmark = pytest.mark.cpu_only


def _collected() -> set[str]:
    """The node ids pytest collects under ``tests/experimental`` with no visible GPU."""
    pytest.importorskip("tvm_ffi")  # some listed suites skip at import without it
    done = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-p",
            "no:cacheprovider",
            f"--rootdir={known_gaps.ROOT}",
            "tests/experimental",
        ],
        cwd=known_gaps.ROOT,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES=""),
        capture_output=True,
        text=True,
        timeout=1800,
    )
    assert done.returncode == 0, (done.stdout[-4000:], done.stderr[-4000:])
    return {line.strip() for line in done.stdout.splitlines() if "::" in line}


def test_every_known_gap_names_a_collected_test():
    collected = _collected()
    stale = {
        model: sorted(set(gaps) - collected)
        for model, gaps in known_gaps.KNOWN_GAPS.items()
        if set(gaps) - collected
    }
    assert not stale, stale


def test_every_known_gap_model_is_a_device_model():
    assert set(known_gaps.KNOWN_GAPS) <= set(known_gaps.DEVICE_MODELS.values())


def test_every_known_gap_caps_each_of_its_outputs():
    """Every gap names an output and a reason, and each output has two caps, of its median
    and of its largest max-norm miss ratio: each a ratio of four decimals, at least
    NEAR_TIE (NEAR_TIE itself: the ratio must meet the tier), and one of them above NEAR_TIE
    (a miss of the tier has a ratio above it, so caps of NEAR_TIE only would cover
    nothing)."""
    bad = []
    for model, gaps in known_gaps.KNOWN_GAPS.items():
        for key, gap in gaps.items():
            if not gap.caps or not gap.reason:
                bad.append((model, key, dict(gap.caps), gap.reason))
            for output, caps in gap.caps.items():
                if not (
                    type(caps) is tuple
                    and len(caps) == len(known_gaps.STATISTICS)
                    and all(
                        type(cap) is float and cap >= NEAR_TIE and round(cap, 4) == cap
                        for cap in caps
                    )
                    and max(caps) > NEAR_TIE
                ):
                    bad.append((model, key, output, caps))
    assert not bad, bad


# ---------------------------------------------------------------------------
# What a known gap's xfail covers: synthetic judgments on CPU
# ---------------------------------------------------------------------------

#: A BF16 ulp in [1, 2).
_ULP = 2.0**-7
#: Elements of each synthetic output: one element one ulp off moves the output's L2 error
#: by a factor sqrt(1 + 8 / _N) < NEAR_TIE.
_N = 4096


def _draw_values(index: int, exact: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw ``index`` of a synthetic output: BF16 values in [1, 2) (the naive tier) and
    the float64 oracle a quarter ulp above them (``exact``: the values themselves), so
    the tier's error is a quarter ulp at every element (0 when ``exact``).  Element 0
    tells the draws apart."""
    steps = torch.arange(_N, dtype=torch.float64).mul(7).remainder(128)
    steps[0] = index
    naive = (1 + _ULP * steps).to(torch.bfloat16)
    oracle = 1 + _ULP * steps + (0.0 if exact else _ULP / 4)
    return oracle, naive


def _bumped(naive: torch.Tensor, *rows: int) -> torch.Tensor:
    """``naive`` with the elements ``rows`` one ulp higher (``rows`` empty: all of them)."""
    out = naive.double()
    if rows:
        out[list(rows)] += _ULP
    else:
        out += _ULP
    return out.to(torch.bfloat16)


#: The median and largest max-norm miss ratios of each synthetic output (:func:`_output`)
#: that misses the tier: three quarters of an ulp off at one element (``max_only``,
#: ``nan``) or at every element (``l2``) where the tier is a quarter ulp off, on every
#: draw; ``median_only`` is ``max_only`` with the tier 3.75 ulps off at another element on
#: draw 0, so its largest ratio is 0.75 / 3.75; and one ulp off at one element of draw 0,
#: where the tier is exact (``zero``); the tier's largest error over the draws is a
#: quarter ulp.
_RATIO = {
    "max_only": (3.0, 3.0),
    "nan": (3.0, 3.0),
    "l2": (3.0, 3.0),
    "median_only": (3.0, 0.75 / 3.75),
    "zero": (1.0, 4.0),
}


def _at_cap(*names: str) -> dict[str, tuple[float, float]]:
    """The caps of a gap of the outputs ``names``, as the table's: each ratio of a
    synthetic output's miss (a miss at its caps), NEAR_TIE where that ratio meets the tier
    and for an output with no synthetic miss."""
    return {
        name: tuple(max(ratio, NEAR_TIE) for ratio in _RATIO.get(name, (NEAR_TIE, NEAR_TIE)))
        for name in names
    }


def _output(name: str, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(actual, oracle, naive)`` of the synthetic output ``name`` on draw ``index``:
    ``meets`` is the naive tier; ``max_only`` misses the tier in the max norm only (one
    element one ulp off on every draw); ``median_only`` is ``max_only`` with the tier's
    value four ulps higher at another element on draw 0 (it misses the tier in the median
    max-norm error only); ``l2`` misses it in both norms (every element one ulp off); ``zero`` breaks the
    zero rule on draw 0 only (an exact oracle, one element off); ``nan`` is ``max_only``
    with a NaN where the oracle is finite."""
    oracle, naive = _draw_values(index, exact=name == "zero" and index == 0)
    if name == "meets":
        actual = naive.clone()
    elif name in ("max_only", "median_only", "nan"):
        actual = _bumped(naive, 1)
        if name == "nan":
            actual[2] = float("nan")
        if name == "median_only" and index == 0:
            naive = naive.double()
            naive[2] += 4 * _ULP
            naive = naive.to(torch.bfloat16)
    elif name == "l2":
        actual = _bumped(naive)
    elif name == "zero":
        actual = _bumped(naive, 1) if index == 0 else naive.clone()
    else:
        raise ValueError(name)
    return actual, oracle, naive


def _judge(names, **kwargs):
    """``judge_draws`` of the synthetic outputs ``names`` (``then`` / ``misses``)."""

    def draw(d):
        for name in names:
            d.output(name, *_output(name, d.index))

    return judge_draws(None, draw, seed=0, **kwargs)


def _failure(names, **kwargs) -> AssertionError:
    with pytest.raises(AssertionError) as info:
        _judge(names, **kwargs)
    return info.value


def test_a_listed_miss_in_the_max_norm_only_keeps_the_xfail():
    _judge(["meets"])  # the synthetic tier itself is met
    error = _failure(["meets", "max_only"])
    assert type(error) is NaiveTierMiss
    (miss,) = error.missed
    assert miss.output == "max_only" and miss.met == {"l2": True, "max": False}
    assert not miss.zero
    assert known_gaps.covers(error, _at_cap("max_only"))
    assert known_gaps.covers(error, _at_cap("max_only", "dX"))


def test_a_listed_miss_at_its_cap_keeps_the_xfail():
    error = _failure(["meets", "max_only"])
    (miss,) = error.missed
    assert miss.ratios["max"] == (3.0, 3.0)
    assert known_gaps.uncovered(error, {"max_only": (3.0, 3.0)}) == []
    assert known_gaps.covers(error, {"max_only": (3.0, 3.0)})
    # a miss in the median ratio only, at its cap; its largest ratio meets its cap NEAR_TIE
    median = _failure(["meets", "median_only"])
    (miss,) = median.missed
    assert miss.met == {"l2": True, "max": False} and not miss.zero
    assert miss.ratios["max"] == (3.0, 0.75 / 3.75)
    assert known_gaps.uncovered(median, {"median_only": (3.0, NEAR_TIE)}) == []
    assert known_gaps.covers(median, _at_cap("median_only"))


def test_a_listed_miss_above_its_cap_fails():
    """Each of the median and the largest max-norm miss ratio is held to its own cap."""
    error = _failure(["meets", "max_only"])
    for cap in (2.9999, math.nextafter(3.0, 0.0)):
        median = f"max_only: median max-norm miss ratio 3 is above its cap {cap!r}"
        largest = f"max_only: largest max-norm miss ratio 3 is above its cap {cap!r}"
        for caps, reasons in (
            ((3.0, cap), [largest]),
            ((cap, 3.0), [median]),
            ((cap, cap), [median, largest]),
        ):
            assert not known_gaps.covers(error, {"max_only": caps})
            assert known_gaps.uncovered(error, {"max_only": caps}) == reasons
    # the median ratio above its cap fails, whatever the largest ratio and its cap
    median = _failure(["meets", "median_only"])
    for caps in ((2.9999, NEAR_TIE), (2.9999, 3.0), (NEAR_TIE, 3.0)):
        assert not known_gaps.covers(median, {"median_only": caps})
        assert known_gaps.uncovered(median, {"median_only": caps}) == [
            f"median_only: median max-norm miss ratio 3 is above its cap {caps[0]!r}"
        ]
    zero = _failure(["zero"])
    (miss,) = zero.missed
    assert miss.ratios["max"] == (1.0, 4.0)
    assert known_gaps.uncovered(zero, {"zero": (NEAR_TIE, 3.9999)}) == [
        "zero: error not 0 on draws [0], where the tier's is 0",
        "zero: largest max-norm miss ratio 4 is above its cap 3.9999",
    ]


def test_a_miss_with_a_broken_contract_fails():
    error = _failure(["max_only", "nan"])
    assert type(error) is AssertionError
    assert "not finite" in str(error) and MISS in str(error)
    assert not known_gaps.covers(error, _at_cap("max_only", "nan"))


def test_a_miss_of_an_unlisted_output_fails():
    error = _failure(["max_only"])
    assert not known_gaps.covers(error, _at_cap("dX"))
    assert not known_gaps.covers(error, _at_cap())


def test_a_miss_in_the_l2_norm_fails():
    error = _failure(["l2"])
    assert type(error) is NaiveTierMiss
    assert [miss.met["l2"] for miss in error.missed] == [False]
    assert not known_gaps.covers(error, _at_cap("l2"))
    both = _failure(["max_only", "l2"])
    assert not known_gaps.covers(both, _at_cap("max_only", "l2"))


def test_a_miss_by_the_zero_rule_fails():
    error = _failure(["zero"])
    assert type(error) is NaiveTierMiss
    (miss,) = error.missed
    assert miss.zero == (0,) and miss.met["l2"]
    assert not known_gaps.covers(error, _at_cap("zero"))


def test_a_plain_failure_with_the_miss_text_fails():
    text = f"dW: {MISS} over 32 inputs"
    assert not known_gaps.covers(AssertionError(text), _at_cap("dW"))
    assert not known_gaps.covers(RuntimeError(text), _at_cap("dW"))
    assert not known_gaps.covers(None, _at_cap("dW"))
    assert not known_gaps.covers(NaiveTierMiss(text, []), _at_cap("dW"))


def test_then_runs_before_a_miss_is_raised():
    ran = []
    error = _failure(["max_only"], then=lambda: ran.append(True))
    assert ran == [True] and known_gaps.covers(error, _at_cap("max_only"))


def test_a_failure_of_then_fails_the_test():
    def then():
        raise AssertionError("bits differ (synthetic)")

    error = _failure(["max_only"], then=then)
    assert type(error) is AssertionError
    assert "bits differ" in str(error) and MISS in str(error)
    assert not known_gaps.covers(error, _at_cap("max_only"))
    alone = _failure(["meets"], then=then)
    assert type(alone) is AssertionError and "bits differ" in str(alone)

    def broken():
        raise RuntimeError("launch failed (synthetic)")

    for names in (["meets"], ["max_only"]):
        error = _failure(names, then=broken)
        assert type(error) is AssertionError and type(error.__cause__) is RuntimeError
        assert "then: RuntimeError: launch failed" in str(error)
        assert not known_gaps.covers(error, _at_cap(*names))


def test_a_skip_in_then_leaves_the_miss():
    def then():
        pytest.skip("no such build here (synthetic)")

    error = _failure(["max_only"], then=then)
    assert known_gaps.covers(error, _at_cap("max_only"))
    with pytest.raises(pytest.skip.Exception):
        _judge(["meets"], then=then)


def test_collected_misses_are_raised_last():
    misses = []
    _judge(["max_only"], misses=misses)
    _judge(["meets"], misses=misses)
    assert [type(miss) for miss in misses] == [NaiveTierMiss]
    with pytest.raises(NaiveTierMiss) as info:
        raise_misses(misses)
    assert known_gaps.covers(info.value, _at_cap("max_only"))
    raise_misses([])
    plain = _failure(["nan"], misses=[])
    assert type(plain) is AssertionError
    with pytest.raises(TypeError):
        _judge(["meets"], then=lambda: None, misses=[])


# ---------------------------------------------------------------------------
# The conftest's fixture and hook: a pytest run of synthetic suites
# ---------------------------------------------------------------------------

#: The synthetic suites: a deterministic one (strict xfail) and a non-deterministic one.
_SUITES = (
    "test_cake_moe_grouped_gemm_det_train_synthetic",
    "test_cake_moe_grouped_gemm_nondet_train_synthetic",
)
#: The synthetic gaps of each suite's tests (``test_unlisted`` has none).
_GAPS = {
    "test_miss": {"max_only": (3.0, 3.0)},
    "test_above_cap": {"max_only": (3.0, 2.9999)},
    "test_plain_failure": {"max_only": (3.0, 3.0)},
    "test_pass": {"max_only": (3.0, 3.0)},
}
#: Each synthetic test's outcome in the deterministic and the non-deterministic suite.
_OUTCOMES = {
    "test_miss": ("XFAIL", "XFAIL"),
    "test_above_cap": ("FAILED", "FAILED"),
    "test_plain_failure": ("FAILED", "FAILED"),
    "test_pass": ("FAILED", "XPASS"),
    "test_unlisted": ("FAILED", "FAILED"),
}
_SYNTHETIC_SUITE = """
from tests.test_known_gaps import _judge


def test_miss():
    _judge(["meets", "max_only"])


def test_above_cap():
    _judge(["meets", "max_only"])


def test_plain_failure():
    raise AssertionError("bits differ (synthetic)")


def test_pass():
    pass


def test_unlisted():
    _judge(["meets", "max_only"])
"""
_SYNTHETIC_CONFTEST = """
from pathlib import Path

import torch

from tests import known_gaps
from tests.conftest import _known_gap, pytest_runtest_makereport  # noqa: F401

# the fixture reads the device's name only: a B200, with no GPU
torch.cuda.is_available = lambda: True
torch.cuda.current_device = lambda: 0
torch.cuda.get_device_name = lambda device=None: "NVIDIA B200"
known_gaps.ROOT = Path(__file__).resolve().parent
known_gaps.KNOWN_GAPS = {
    "B200": {
        f"{suite}.py::{name}": known_gaps.Gap(caps, "synthetic gap")
        for suite in SUITES
        for name, caps in GAPS.items()
    }
}
known_gaps.LISTED = frozenset(known_gaps.KNOWN_GAPS["B200"])
"""


def test_the_conftest_fixture_and_hook_apply_the_rule(tmp_path):
    """The conftest's fixture and hook (``tests/conftest.py``) in a pytest run of a
    deterministic and a non-deterministic synthetic suite, on a device named as a B200 (no
    GPU is used): a listed miss at its caps is an XFAIL; a listed miss above a cap, a plain
    failure of a listed test and an unlisted test's miss fail; a listed test that passes
    fails in the deterministic suite (a strict XPASS) and is an XPASS in the
    non-deterministic one.  The failure reports say why the xfail does not cover them."""
    (tmp_path / "pytest.ini").write_text("[pytest]\n")
    (tmp_path / "conftest.py").write_text(
        f"SUITES = {_SUITES!r}\nGAPS = {_GAPS!r}\n{_SYNTHETIC_CONFTEST}"
    )
    for suite in _SUITES:
        (tmp_path / f"{suite}.py").write_text(_SYNTHETIC_SUITE)
    path = [str(known_gaps.ROOT), os.environ.get("PYTHONPATH", "")]
    done = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-rA"]
        + [f"{suite}.py" for suite in _SUITES],
        cwd=tmp_path,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=os.pathsep.join(filter(None, path))),
        capture_output=True,
        text=True,
        timeout=900,
    )
    got = {}
    for line in done.stdout.splitlines():
        if found := re.match(r"(PASSED|FAILED|ERROR|XFAIL|XPASS) (\S+)\.py::(\w+)", line):
            got[found[2], found[3]] = found[1]
    want = {
        (suite, name): outcomes[i]
        for name, outcomes in _OUTCOMES.items()
        for i, suite in enumerate(_SUITES)
    }
    log = done.stdout[-8000:] + done.stderr[-2000:]
    assert done.returncode == 1 and got == want, log
    for reason, count in (
        ("max_only: largest max-norm miss ratio 3 is above its cap 2.9999", 2),
        ("the failure is not a miss of the naive tier alone (a NaiveTierMiss)", 2),
        ("XPASS(strict)", 1),
    ):
        assert done.stdout.count(reason) >= count, (reason, log)


# ---------------------------------------------------------------------------
# No test checks anything after its judgment: a static check of the suites
# ---------------------------------------------------------------------------

#: The judgment entries of ``tests/cake/test_helpers/cake_moe_judge.py``.
_JUDGES = ("judge_draws", "raise_misses")
#: Calls a test may make after its judgment: they check nothing.
_NO_OPS = ("print", "record_property")
_DEFS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)


def _callee(call: ast.Call) -> str | None:
    func = call.func
    if isinstance(func, ast.Name):
        return func.id
    return func.attr if isinstance(func, ast.Attribute) else None


def _calls(node: ast.AST) -> Iterator[ast.Call]:
    """The calls in ``node`` outside the functions, classes and lambdas it defines."""
    for child in ast.iter_child_nodes(node):
        if isinstance(child, _DEFS):
            continue
        if isinstance(child, ast.Call):
            yield child
        yield from _calls(child)


def _blocks(stmt: ast.stmt) -> list[list[ast.stmt]]:
    """The statement blocks of a compound statement."""
    blocks = [getattr(stmt, field, None) for field in ("body", "orelse", "finalbody")]
    blocks += [handler.body for handler in getattr(stmt, "handlers", [])]
    blocks += [case.body for case in getattr(stmt, "cases", [])]
    return [block for block in blocks if block]


def _no_op(stmt: ast.stmt) -> bool:
    if isinstance(stmt, ast.Pass):
        return True
    if isinstance(stmt, ast.Return):
        return stmt.value is None or isinstance(stmt.value, (ast.Name, ast.Constant))
    if isinstance(stmt, ast.Expr):
        value = stmt.value
        if isinstance(value, ast.Constant):
            return True
        return isinstance(value, ast.Call) and _callee(value) in _NO_OPS
    return False


def _judge_entries(functions: list[ast.FunctionDef]) -> set[str]:
    """The judgment entries and every helper that calls one (by name)."""
    entries = set(_JUDGES)
    grown = True
    while grown:
        grown = False
        for fn in functions:
            if fn.name in entries or fn.name.startswith("test_"):
                continue
            if any(_callee(call) in entries for call in _calls(fn)):
                entries.add(fn.name)
                grown = True
    return entries


def _late_checks(path: Path, fn: ast.FunctionDef, entries: set[str]) -> tuple[int, list[str]]:
    """The judgment calls of ``fn`` and its problems: a statement other than a no-op after
    a judgment, a judgment in a loop that does not collect its misses (``misses=``), and
    collected misses that are not raised last (``raise_misses``)."""
    found, problems, collected = 0, [], set()
    where = f"{path.name}:{fn.name}"

    def visit(block: list[ast.stmt], loop: bool, rest: list[ast.stmt]) -> None:
        nonlocal found
        for index, stmt in enumerate(block):
            if isinstance(stmt, _DEFS):
                continue
            judged = [call for call in _calls(stmt) if _callee(call) in entries]
            if not judged:
                continue
            after = block[index + 1 :] + rest
            inner = [
                sub
                for sub in _blocks(stmt)
                if any(_callee(call) in entries for s in sub for call in _calls(s))
            ]
            if inner:
                inside = loop or isinstance(stmt, (ast.For, ast.AsyncFor, ast.While))
                for sub in inner:
                    visit(sub, inside, after)
                continue
            for call in judged:
                found += 1
                kept = [kw.value for kw in call.keywords if kw.arg == "misses"]
                if kept:
                    if isinstance(kept[0], ast.Name):
                        collected.add(kept[0].id)
                    continue
                if loop:
                    problems.append(f"{where}:{stmt.lineno}: a judgment in a loop without misses=")
                late = [s for s in after if not _no_op(s)]
                if late:
                    problems.append(
                        f"{where}:{stmt.lineno}: checks after the judgment (line {late[0].lineno})"
                    )

    visit(fn.body, False, [])
    params = {a.arg for a in [*fn.args.posonlyargs, *fn.args.args, *fn.args.kwonlyargs]}
    tail = [s for s in fn.body if not _no_op(s)]
    last = tail[-1] if tail else None
    for name in sorted(collected - params):
        if not (
            isinstance(last, ast.Expr)
            and isinstance(last.value, ast.Call)
            and _callee(last.value) == "raise_misses"
            and [ast.dump(a) for a in last.value.args] == [ast.dump(ast.Name(name, ast.Load()))]
        ):
            problems.append(f"{where}: the misses collected in {name} are not raised last")
    return found, problems


def test_no_test_checks_anything_after_its_judgment():
    """A judgment raises a miss of the naive tier only after the test's later checks ran
    (its ``then``), so in every function of the GPU suites a judgment is the last
    statement but no-ops (``return``, ``print``, ``record_property``) in each enclosing
    block, and a judgment in a loop collects its misses (``misses=``), which the test
    raises last (``raise_misses``)."""
    paths = sorted((known_gaps.ROOT / "tests" / "experimental").glob("test_*.py"))
    trees = {path: ast.parse(path.read_text(), str(path)) for path in paths}
    functions = [
        node
        for tree in trees.values()
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    entries = _judge_entries(functions)
    assert {"_judge_launch", "_judge_repeats", "_judge_spot"} <= entries, entries
    problems = []
    for path, tree in trees.items():
        found = 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                count, bad = _late_checks(path, node, entries)
                found += count
                problems += bad
        if "judge_draws" in path.read_text():
            assert found, f"{path.name}: no judgment found"
    assert not problems, "\n".join(problems)
