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

The terminal summary prints one ``TIER`` line per output the tests judged against the naive BF16 tier and FLA
(:mod:`.cake_naive`), passing and xfailed tests included, one ``FLA reference`` line per pinned FLA reference the
tests ran (its versions, pin, GPU, shared memory and ptxas) and one ``FLA`` line per test with the count and seconds
of its FLA reference steps (``experimental/test_helpers/cake_kda_fla.py``).  A test that passed with owner-accepted
gaps within their caps is reported as xfailed, its reason naming each gap; a test that passed without judging an
output listed as a gap for it fails (``experimental/test_helpers/cake_kda_known_gaps.py``).
"""

import pytest

from .cake_naive import summarize_tiers
from .experimental.test_helpers import cake_kda_known_gaps as known_gaps
from .experimental.test_helpers.cake_kda_fla import summarize_fla


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    summarize_tiers(terminalreporter)
    summarize_fla(terminalreporter)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item):
    # the running test of the gaps' settling (cake_kda_known_gaps.settle)
    with known_gaps.running(item.nodeid):
        return (yield)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item, call):
    # a passed call with gaps within their caps -> xfailed; a listed output the test never judged -> failed; a failed
    # call (a gap beyond its caps among them) stays failed
    report = yield
    if call.when == "call":
        known_gaps.finish(item.nodeid, report)
    return report
