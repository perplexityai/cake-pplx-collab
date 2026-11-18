import pytest

from .cake_naive import summarize_tiers
from .experimental.test_helpers import cake_lm_head_known_misses as known_misses


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    summarize_tiers(terminalreporter)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item):
    # the running test of the judges' known-miss settling (test_helpers/cake_lm_head_known_misses.py)
    with known_misses.running(item.nodeid):
        return (yield)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item, call):
    # a passed call with known misses within their caps -> xfailed (reason: each output, its ratios and caps); a listed
    # output the test never judged -> failed; a failed call (a miss beyond its caps among them) stays failed.  The TIER
    # lines read every report, xfailed ones too.
    report = yield
    if call.when == "call":
        known_misses.finish(item.nodeid, report)
    return report
