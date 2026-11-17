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

Host tests of the known-miss registry (``tests/experimental/test_helpers/cake_lm_head_known_misses.py``):
:func:`settle` fails a tier miss that is not listed, a listed output that meets the tier and every failure that is not
a tier miss, and records a listed miss; :func:`finish` reports a passed test with recorded misses as xfailed (its
``TIER`` lines still print), fails a passed test that never judged a listed output, leaves a failed test failed and
leaves a test that lists and records nothing as it is without reading the architecture.
The judges are fakes on the CPU (no kernel runs) and the registry a fake one, except the registry's own form check.
"""

from pathlib import Path

import pytest
import torch

from tests.cake_naive import DRAWS, TierJudge, summarize_tiers
from tests.experimental.test_helpers import cake_lm_head_known_misses as known_misses

TEST = "tests/experimental/test_fake.py::test_case[p]"
LISTED = (1.0, 1.0, 2.0, 2.0)


def _judge(miss=(), meet=(), *, draws=DRAWS):
    """A judge of the outputs ``miss`` (twice the tier's error) and ``meet`` (the oracle in FP32) over ``draws``
    draws of the 32."""
    judge = TierJudge()
    for j in range(draws):
        oracle = torch.arange(1, 9, dtype=torch.float64) * (j + 1) / 3
        tier = oracle.to(torch.bfloat16)
        for name in miss:
            judge.add(name, j, (2 * tier.double() - oracle).float(), oracle, {"bf16": tier})
        for name in meet:
            judge.add(name, j, oracle.float(), oracle, {"bf16": tier})
    return judge


def _report(judge, outcome="passed"):
    properties = []
    judge.record(lambda key, value: properties.append((key, value)))
    return pytest.TestReport(
        TEST, ("test_fake.py", 0, "test_case[p]"), {}, outcome, None, "call", user_properties=properties
    )


@pytest.fixture
def registry(monkeypatch):
    fake = {TEST: {}}
    monkeypatch.setattr(known_misses, "KNOWN_MISSES", fake)
    yield fake[TEST]
    known_misses._records.pop(TEST, None)


def _settle(judge, failures=None):
    with known_misses.running(TEST):
        known_misses.settle(judge, judge.failures() if failures is None else failures, "cpu")


def test_a_miss_that_is_not_listed_fails(registry):
    registry["a"] = {"sm_100": LISTED}  # listed on another architecture only
    judge = _judge(miss=("a",), meet=("b",))
    with pytest.raises(AssertionError, match=r"^a: misses the naive tier over 32 inputs"):
        _settle(judge)
    assert TEST not in known_misses._records


def test_a_listed_miss_is_recorded_and_the_passed_test_is_xfailed(registry, request):
    registry["a"] = {"cpu": LISTED}
    judge = _judge(miss=("a",), meet=("b",))
    _settle(judge)
    assert set(known_misses._records[TEST]["missed"]) == {("cpu", "a")}
    report = _report(judge)
    known_misses.finish(TEST, report)
    assert TEST not in known_misses._records
    measured = known_misses.ratios(judge.verdicts()["a"])
    assert all(abs(r - 2) < 0.01 for r in measured)
    assert report.skipped
    assert report.wasxfail == f"known naive-tier misses (KNOWN_MISSES): a on cpu: {known_misses._shown(measured)}"
    status = request.config.hook.pytest_report_teststatus(report=report, config=request.config)
    assert status[0] == "xfailed"
    printed = []
    reporter = type(
        "Reporter",
        (),
        {"stats": {"xfailed": [report]}, "section": lambda self, *a, **k: None, "write_line": printed.append},
    )()
    summarize_tiers(reporter)
    assert [line.split()[2:4] for line in printed] == [["a", "FAIL"], ["b", "bf16"]]


def test_a_listed_output_that_meets_the_tier_fails(registry):
    registry["a"] = {"cpu": LISTED}
    registry["b"] = {"cpu": LISTED}
    with pytest.raises(
        AssertionError,
        match=r"^b: registered as a known miss on cpu but meets the naive tier now: remove it from KNOWN_MISSES$",
    ):
        _settle(_judge(miss=("a",), meet=("b",)))
    assert TEST not in known_misses._records


def test_a_failure_that_is_not_a_tier_miss_fails(registry):
    registry["a"] = {"cpu": LISTED}
    judge = _judge(miss=("a",))
    contract = "draw 3: a: 1 of 8 elements are not finite where the oracle is, or miss its inf or NaN"
    with pytest.raises(AssertionError, match="draw 3: a: 1 of 8"):
        _settle(judge, [contract, *judge.failures()])
    short = _judge(miss=("a",), draws=DRAWS - 1)
    with pytest.raises(AssertionError, match=r"a: judged on 31 of 32 draws"):
        _settle(short)
    assert TEST not in known_misses._records


def test_a_listed_output_that_the_test_never_judged_fails(registry):
    registry["a"] = {"cpu": LISTED}
    registry["c"] = {"cpu": LISTED}
    judge = _judge(miss=("a",), meet=("b",))
    _settle(judge)
    report = _report(judge)
    known_misses.finish(TEST, report)
    assert report.failed
    assert report.longrepr == (
        "c: registered as a known miss on cpu but not judged by the test: remove it from KNOWN_MISSES"
    )


def test_a_failed_call_stays_failed_and_an_unlisted_test_is_unchanged(registry):
    registry["a"] = {"cpu": LISTED}
    judge = _judge(miss=("a",))
    _settle(judge)
    report = _report(judge, "failed")
    known_misses.finish(TEST, report)
    assert report.failed and not hasattr(report, "wasxfail") and TEST not in known_misses._records
    del registry["a"]
    clean = _report(_judge(meet=("b",)))
    known_misses.finish(TEST, clean)
    assert clean.passed and not hasattr(clean, "wasxfail")


def test_a_test_that_lists_nothing_needs_no_architecture(registry, monkeypatch):
    """A passed test that lists no output and judged none (as the plan tests that stub ``torch.cuda`` queries) is left
    as it is; :func:`finish` reads no architecture."""
    report = _report(_judge(meet=("b",)))

    def stub(device=None):
        raise AssertionError("finish read the architecture")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", stub)
    known_misses.finish(TEST, report)
    assert report.passed and not hasattr(report, "wasxfail")


def test_the_running_test_and_its_architecture(request):
    assert known_misses.current_test() == request.node.nodeid  # the conftest's
    with known_misses.running(TEST):
        assert known_misses.current_test() == TEST
        with known_misses.running(None):  # pytest's PYTEST_CURRENT_TEST
            assert known_misses.current_test() == request.node.nodeid
    assert known_misses.current_test() == request.node.nodeid
    assert known_misses.current_arch("cpu") == "cpu" and known_misses.current_arch(torch.device("cpu")) == "cpu"
    if not torch.cuda.is_available():
        assert known_misses.current_arch() == "cpu"


def test_the_registry_lists_misses_of_this_family_s_tests():
    assert known_misses.check_registry() == []
    root = Path(__file__).resolve().parents[2]
    for test in known_misses.KNOWN_MISSES:
        path, name = test.split("::", 1)
        assert f"def {name.split('[', 1)[0]}(" in (root / path).read_text(), test
    bad = {"tests/x.py::t": {"a": {"sm_90": (1.0, 1.0, 1.0, 1.1), "cpu": (1.0, 1.0, 1.0, 1.0)}}}
    assert len(known_misses.check_registry(bad)) == 2
