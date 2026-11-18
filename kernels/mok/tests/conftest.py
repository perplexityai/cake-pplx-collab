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

Shared fixtures of the MoK suites and the terminal summary of the accuracy verdicts.

The summary prints one ``TIER`` line per judged output (:func:`tests.cake_naive.summarize_tiers`), then one ``ALT`` line
per output measured against a kernel-shaped alternative of the naive implementation
(:data:`tests.experimental._mok_naive.ALTERNATIVE`), then one ``GAP`` line per known-gap output that missed the tier:
its item, the architecture, and as JSON its ratios per norm it does not meet (median, largest; full precision), its
caps there and what it misses beyond them (:func:`tests.experimental._mok_gaps.check`).  The alternatives are
reported, never judged.
"""

from __future__ import annotations

import json

import pytest
import torch

from .cake_naive import summarize_tiers, tier_line


@pytest.fixture
def cuda_matmul_flags():
    """A setter for ``torch.backends.cuda.matmul`` flags; restores them at teardown.

    ``monkeypatch.setattr`` cannot be used for these flags: ``cuBLASModule`` serves them from C state through
    ``__getattr__`` / ``__setattr__``, and pytest's monkeypatch restores such an attribute with ``delattr``, which
    ``cuBLASModule`` does not support.
    """
    matmul = torch.backends.cuda.matmul
    saved: dict[str, object] = {}

    def set_flags(**flags: object) -> None:
        for name, value in flags.items():
            saved.setdefault(name, getattr(matmul, name))
            setattr(matmul, name, value)

    yield set_flags
    for name, value in reversed(saved.items()):
        setattr(matmul, name, value)


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    summarize_tiers(terminalreporter)
    lines, gap_lines = [], []
    for reports in terminalreporter.stats.values():
        for report in reports:
            if getattr(report, "when", None) != "call":
                continue
            for key, verdict in getattr(report, "user_properties", ()):
                if key == "naive_alternative":
                    lines.append("ALT " + tier_line(report.nodeid, verdict).removeprefix("TIER "))
                elif key == "known_gap":
                    record = {k: verdict[k] for k in ("ratios", "caps", "beyond")}
                    gap_lines.append(f"GAP {report.nodeid} {verdict['item']} {verdict['arch']} {json.dumps(record)}")
    if lines:
        terminalreporter.section("training accuracy: kernel-shaped alternatives of the naive tier (reported only)")
        for line in lines:
            terminalreporter.write_line(line)
    if gap_lines:
        terminalreporter.section("training accuracy: known gaps, error ratio in the norms they miss, and caps")
        for line in gap_lines:
            terminalreporter.write_line(line)
