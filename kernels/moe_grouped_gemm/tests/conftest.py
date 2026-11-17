# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION
"""The suite's terminal summary: one ``TIER`` line per output a GPU test judged against the
naive BF16 tier (``tests/cake/test_helpers/cake_naive.py``), passing tests included.  The
known gaps of ``tests/known_gaps.py`` are expected failures on their GPU model."""

import pytest
import torch

from tests import known_gaps
from tests.cake.test_helpers.cake_naive import summarize_tiers

#: The outputs of a test's known gap on the current GPU model (set when it is marked xfail).
_KNOWN_GAP = pytest.StashKey[frozenset]()


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
        reason, strict, outputs = gap
        request.node.add_marker(pytest.mark.xfail(reason=reason, strict=strict))
        request.node.stash[_KNOWN_GAP] = outputs


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_makereport(item, call):
    """A known gap's xfail covers its known miss of the naive tier only
    (``known_gaps.covers``): any other failure of the test stays a failure."""
    outcome = yield
    report = outcome.get_result()
    outputs = item.stash.get(_KNOWN_GAP, None)
    if (
        outputs is not None
        and hasattr(report, "wasxfail")
        and call.excinfo is not None
        and not known_gaps.covers(call.excinfo.value, outputs)
    ):
        report.outcome = "failed"
        del report.wasxfail
