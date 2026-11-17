# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION
"""The suite's terminal summary: one ``TIER`` line per output a GPU test judged against the
naive BF16 tier (``tests/cake/test_helpers/cake_naive.py``), passing tests included.  The
known gaps of ``tests/known_gaps.py`` are expected failures on their GPU model."""

import pytest
import torch

from tests import known_gaps
from tests.cake.test_helpers.cake_naive import summarize_tiers

_KNOWN_GAP = pytest.StashKey[bool]()


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    summarize_tiers(terminalreporter)


@pytest.fixture(autouse=True)
def _known_gap(request):
    """Mark a test ``xfail`` where ``tests/known_gaps.py`` lists it for the current CUDA
    device's GPU model (strict in the deterministic suites)."""
    key = known_gaps.node_key(request.node.path, request.node.nodeid)
    if key not in known_gaps.LISTED or not torch.cuda.is_available():
        return
    name = torch.cuda.get_device_name(torch.cuda.current_device())
    if gap := known_gaps.known_gap(name, key):
        reason, strict = gap
        request.node.add_marker(pytest.mark.xfail(reason=reason, strict=strict))
        request.node.stash[_KNOWN_GAP] = True


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_makereport(item, call):
    """A known gap's xfail covers its miss of the naive tier only: a failure whose message
    does not report that miss stays a failure."""
    outcome = yield
    report = outcome.get_result()
    if (
        item.stash.get(_KNOWN_GAP, False)
        and hasattr(report, "wasxfail")
        and call.excinfo is not None
        and known_gaps.MISS not in str(call.excinfo.value)
    ):
        report.outcome = "failed"
        del report.wasxfail
