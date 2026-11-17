# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION
"""The known-gap table (``tests/known_gaps.py``) names collected tests only, of known GPU
models.  CPU only: the suites are collected in a subprocess that sees no GPU, and nothing
is launched."""

import os
import subprocess
import sys

import pytest

from tests import known_gaps

pytestmark = pytest.mark.cpu_only

pytest.importorskip("tvm_ffi")  # some listed suites skip at import without it


def _collected() -> set[str]:
    """The node ids pytest collects under ``tests/experimental`` with no visible GPU."""
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
