from .cake_naive import summarize_tiers


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    summarize_tiers(terminalreporter)
