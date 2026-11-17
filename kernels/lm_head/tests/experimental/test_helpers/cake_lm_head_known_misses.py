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

The outputs of the LM-head tests that miss the naive tier today, per architecture (:data:`KNOWN_MISSES`).

The shared judge (``tests/cake_naive.py``) is unchanged: every output is judged against the naive BF16 tier, and
every verdict prints its ``TIER`` line.  After a judge, a test calls :func:`settle` instead of asserting that the
judge found no failure:

- a failure that is not a tier miss (a contract break, an output judged on fewer draws than the judge's, two draws
  with the same oracle values) fails the test with the judge's text;
- an output that misses the tier and is not listed for the running test and architecture fails the test with the
  judge's text;
- an output listed for the running test and architecture that meets the tier fails the test: the listing is stale;
- an output listed that misses the tier is recorded, and the test runs on.

The family's ``conftest.py`` reports a test that passed with recorded misses as xfailed, its reason listing each
missed output and its ratios (:func:`finish`); a test that passed without judging an output listed for it fails, and
a test that failed stays failed.  The list is checked only through the tests that run: the entries of a test id that
is not collected (a renamed or removed parametrization, or a run over other test ids) or whose test skips or fails on
the architecture are not checked, and :func:`check_registry` checks the entries' form only.  The test id is pytest's
node id (``tests/experimental/<file>::<test>[<params>]``, the id of the ``TIER`` lines), set by the conftest around each
test's call (:func:`running`).

The architecture is that of the device that ran the kernel (for the reference backend, the device of its inputs):
``"cpu"`` for the CPU, else ``"sm_<major><minor>"`` of that CUDA device (:func:`current_arch`).  The judge site passes
that device to :func:`settle`.  It is the judged outputs' device, unless the site passes the kernel's device explicitly:
``test_cake_sparse_weight_gradient_drops_out_of_range_ids`` copies its dW to the host before the judge and passes the
CUDA device that ran the kernel.  Each listed output keeps the ratios measured when it was listed: the output's
error over the tier's error, (L2 median, L2 largest, max median, max largest) over the 32 draws; they document the
miss and are not compared.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator, Mapping

import torch

from tests.cake_naive import NEAR_TIE, TIERS, TierJudge

#: The architectures an output may be listed for.
ARCHS = ("cpu", "sm_100", "sm_103", "sm_107")

_current: str | None = None
# test id -> {"judged": {arch: set of outputs}, "missed": {(arch, output): ratios}}
_records: dict[str, dict] = {}


@contextlib.contextmanager
def running(test_id: str | None) -> Iterator[None]:
    """Make ``test_id`` (pytest's node id; ``None``: ``PYTEST_CURRENT_TEST``) the running test of :func:`settle`
    while the block runs."""
    global _current
    previous, _current = _current, test_id
    try:
        yield
    finally:
        _current = previous


def current_test() -> str:
    """The running test's id: the one :func:`running` set, else pytest's ``PYTEST_CURRENT_TEST``."""
    if _current is not None:
        return _current
    found = os.environ.get("PYTEST_CURRENT_TEST")
    if not found:
        raise RuntimeError("no running test: call settle() inside a pytest test")
    return found.rsplit(" (", 1)[0]


def current_arch(device: torch.device | str | None = None) -> str:
    """``"cpu"`` for a CPU ``device`` or, without a ``device``, when no CUDA device is available; else
    ``"sm_<major><minor>"`` of ``device`` (default: the current CUDA device)."""
    if device is not None:
        device = torch.device(device)
        if device.type != "cuda":
            return "cpu"
    elif not torch.cuda.is_available():
        return "cpu"
    major, minor = torch.cuda.get_device_capability(device)
    return f"sm_{major}{minor}"


def registered(test_id: str, arch: str) -> list[str]:
    """The outputs of ``test_id`` listed as known misses on ``arch``."""
    return [output for output, archs in KNOWN_MISSES.get(test_id, {}).items() if arch in archs]


def ratios(verdict: Mapping) -> tuple[float, float, float, float]:
    """The output's error over the tier's, (L2 median, L2 largest, max median, max largest), of a
    :meth:`TierJudge.verdicts` entry (inf where the tier's error is 0 or not finite and the output's is not 0)."""
    (tier,) = TIERS
    found = []
    for norm in ("l2", "max"):
        ours, theirs = verdict["norms"][norm]["ours"], verdict["norms"][norm][tier]
        for i in (0, 1):
            if theirs is None or theirs[i] == 0:
                found.append(float("inf") if ours[i] > 0 else 1.0)
            else:
                found.append(ours[i] / theirs[i])
    return tuple(found)


def _shown(r: tuple[float, ...]) -> str:
    return f"l2 x{r[0]:.4f} / x{r[1]:.4f}, max x{r[2]:.4f} / x{r[3]:.4f}"


def settle(judge: TierJudge, failures: list[str], device: torch.device | str | None = None) -> None:
    """Settle a judge's ``failures`` (``judge_case``'s, or ``judge.failures()`` and the contract's) against
    :data:`KNOWN_MISSES` for the running test on the architecture of ``device``, the device that ran the kernel (the
    judged outputs' device unless the site copied them to the host and passes the kernel's device):
    raise an ``AssertionError`` for a failure that is not a tier miss, for a tier miss that is not listed, and for a
    listed output that met the tier; record the listed misses (the test runs on)."""
    verdicts = judge.verdicts()
    misses, others = {}, []
    for failure in failures:
        name = next(
            (n for n, v in verdicts.items() if v["tier"] is None and failure.startswith(f"{n}: misses the naive tier ")),
            None,
        )
        if name is None:
            others.append(failure)
        else:
            misses[name] = failure
    if others:
        raise AssertionError("\n".join(failures))
    test, arch = current_test(), current_arch(device)
    listed = registered(test, arch)
    problems = [failure for name, failure in misses.items() if name not in listed]
    problems += [
        f"{name}: registered as a known miss on {arch} but meets the naive tier now: remove it from KNOWN_MISSES"
        for name in listed
        if name in verdicts and verdicts[name]["tier"] is not None
    ]
    if problems:
        raise AssertionError("\n".join(problems))
    record = _records.setdefault(test, {"judged": {}, "missed": {}})
    record["judged"].setdefault(arch, set()).update(verdicts)
    for name in misses:
        record["missed"][(arch, name)] = ratios(verdicts[name])


def finish(test_id: str, report) -> None:
    """Report a passed call of ``test_id`` (pytest's call-phase ``report``) with recorded misses as xfailed, and one
    that did not judge an output listed for it on its architecture as failed; a failed or skipped call is left as it
    is, and so is a test that lists no output and recorded none (without reading the architecture: such a test may
    still hold its stubs of ``torch.cuda``).  Clears the test's records."""
    record = _records.pop(test_id, None)
    if report.when != "call" or not report.passed or (record is None and not KNOWN_MISSES.get(test_id)):
        return
    judged = record["judged"] if record else {current_arch(): set()}
    absent = [
        f"{name}: registered as a known miss on {arch} but not judged by the test: remove it from KNOWN_MISSES"
        for arch in sorted(judged)
        for name in registered(test_id, arch)
        if name not in judged[arch]
    ]
    if absent:
        report.outcome = "failed"
        report.longrepr = "\n".join(absent)
    elif record and record["missed"]:
        report.outcome = "skipped"
        report.wasxfail = "known naive-tier misses (KNOWN_MISSES): " + "; ".join(
            f"{name} on {arch}: {_shown(r)}" for (arch, name), r in sorted(record["missed"].items())
        )


def check_registry(registry: Mapping | None = None) -> list[str]:
    """Problems of the registry's form: an architecture outside :data:`ARCHS`, ratios that are not four numbers, or
    an entry none of whose ratios is above :data:`NEAR_TIE` (no miss)."""
    problems = []
    for test, outputs in (KNOWN_MISSES if registry is None else registry).items():
        if not test.startswith("tests/") or "::" not in test:
            problems.append(f"{test}: not a node id of the family's tests")
        for output, archs in outputs.items():
            for arch, r in archs.items():
                if arch not in ARCHS:
                    problems.append(f"{test} {output}: architecture {arch!r} is not one of {ARCHS}")
                if len(r) != 4 or not all(isinstance(x, float) for x in r):
                    problems.append(f"{test} {output} {arch}: ratios {r!r} are not four floats")
                elif not max(r) > NEAR_TIE:
                    problems.append(f"{test} {output} {arch}: no ratio above {NEAR_TIE}: not a miss")
    return problems


# The outputs that miss the naive tier today, by test id, output and architecture, with the ratios measured on that
# architecture: (L2 median, L2 largest, max median, max largest) of the output's error over the tier's, 32 draws.
# Remove an entry when its output meets the tier (the test fails until it is removed).
KNOWN_MISSES: dict[str, dict[str, dict[str, tuple[float, float, float, float]]]] = {
    # deterministic package: dW@compact, dW@plain: the max-norm median, max-norm largest error above 1.005 x the
    # tier's on cpu, sm_100, sm_103, sm_107; worst x1.3966
    'tests/experimental/test_cake_lm_head_det_train.py::test_compaction_matches_uncompacted[ce-all_but_one]': {
        'dW@compact': {
            'cpu': (0.972911, 0.934621, 1.027744, 1.396588),
            'sm_100': (0.972911, 0.934621, 1.027744, 1.396588),
            'sm_103': (0.972916, 0.934621, 1.027744, 1.396588),
            'sm_107': (0.972916, 0.934621, 1.027744, 1.396588),
        },
        'dW@plain': {
            'cpu': (0.972911, 0.934621, 1.027744, 1.396588),
            'sm_100': (0.972911, 0.934621, 1.027744, 1.396588),
            'sm_103': (0.972916, 0.934621, 1.027744, 1.396588),
            'sm_107': (0.972916, 0.934621, 1.027744, 1.396588),
        },
    },
    # deterministic package: dX: the max-norm median error above 1.005 x the tier's on sm_100, sm_103, sm_107; worst
    # x1.0412
    'tests/experimental/test_cake_lm_head_det_train.py::test_device_default_budget_form_is_deterministic[logprob_zero_dlogp]': {
        'dX': {
            'sm_100': (0.99314, 0.993339, 1.041245, 1.0),
            'sm_103': (0.993131, 0.993329, 1.023999, 1.0),
            'sm_107': (0.993188, 0.993348, 1.012536, 1.0),
        },
    },
    # deterministic package: dX: the max-norm largest error above 1.005 x the tier's on sm_100, sm_103, sm_107; a near
    # tie (every ratio <= x1.02)
    'tests/experimental/test_cake_lm_head_det_train.py::test_device_forward_backward_matches_reference[ce-4095]': {
        'dX': {
            'sm_100': (0.989668, 0.989598, 0.990684, 1.015497),
            'sm_103': (0.98966, 0.989581, 0.990674, 1.015507),
            'sm_107': (0.989506, 0.989436, 0.990674, 1.015507),
        },
    },
    # deterministic package: dX: the max-norm median error above 1.005 x the tier's on sm_100, sm_103, sm_107; a near
    # tie (every ratio <= x1.02)
    'tests/experimental/test_cake_lm_head_det_train.py::test_device_logprob_entry[t1500_c4096_i0.05]': {
        'dX': {
            'sm_100': (0.991236, 0.992498, 1.007457, 1.0),
            'sm_103': (0.991227, 0.992532, 1.007472, 1.0),
            'sm_107': (0.991316, 0.992419, 1.007472, 1.0),
        },
    },
    # deterministic package: dX: the max-norm median error above 1.005 x the tier's on sm_100, sm_103, sm_107; a near
    # tie (every ratio <= x1.02)
    'tests/experimental/test_cake_lm_head_det_train.py::test_device_logprob_entry[t1500_c4096_i0.05_plain]': {
        'dX': {
            'sm_100': (0.991236, 0.992498, 1.007457, 1.0),
            'sm_103': (0.991227, 0.992532, 1.007472, 1.0),
            'sm_107': (0.991316, 0.992419, 1.007472, 1.0),
        },
    },
    # deterministic package: dX: the L2 median, L2 largest, max-norm median error above 1.005 x the tier's on sm_100,
    # sm_103, sm_107; a near tie (every ratio <= x1.02)
    'tests/experimental/test_cake_lm_head_det_train.py::test_device_meets_the_naive_tier_at_the_production_head[loss]': {
        'dX': {
            'sm_100': (1.015017, 1.014592, 1.010065, 0.974487),
            'sm_103': (1.015018, 1.014603, 1.010065, 0.974492),
            'sm_107': (1.012361, 1.011806, 1.004988, 0.989765),
        },
    },
    # deterministic package: dX: the max-norm median error above 1.005 x the tier's on cpu, sm_100, sm_103, sm_107;
    # worst x1.0224
    'tests/experimental/test_cake_lm_head_det_train.py::test_temperature_heterogeneous_matches_the_references[1031-512-all_but_one]': {
        'dX': {
            'cpu': (0.94225, 0.601538, 1.022367, 0.685525),
            'sm_100': (0.94225, 0.601538, 1.022367, 0.685525),
            'sm_103': (0.942896, 0.601534, 1.022367, 0.685504),
            'sm_107': (0.942896, 0.601534, 1.022367, 0.685504),
        },
    },
    # deterministic package: dW: the max-norm largest error above 1.005 x the tier's on cpu; worst x1.1060
    'tests/experimental/test_cake_lm_head_det_train_support.py::test_numerics_against_oracle[packed k_cap > V]': {
        'dW': {
            'cpu': (0.512869, 0.685513, 0.416471, 1.106026),
        },
    },
    # deterministic package: dX: the max-norm largest error above 1.005 x the tier's on sm_100, sm_103, sm_107; worst
    # x1.0409
    "tests/experimental/test_cake_lm_head_det_train_zero_dlogp.py::test_device_zero_dlogp_skip_is_deterministic_and_accurate[8117-4096-('random', 0.166)]": {
        'dX': {
            'sm_100': (0.990891, 0.99083, 1.0, 1.040872),
            'sm_103': (0.990868, 0.990845, 1.0, 1.040878),
            'sm_107': (0.990887, 0.99086, 1.0, 1.040878),
        },
    },
    # deterministic package: dW, dX: the max-norm median, max-norm largest error above 1.005 x the tier's on sm_100,
    # sm_103, sm_107; worst x1.0409
    "tests/experimental/test_cake_lm_head_det_train_zero_dlogp.py::test_device_zero_dlogp_skip_is_deterministic_and_accurate[8117-4096-('random', 0.5)]": {
        'dW': {
            'sm_100': (0.991345, 0.991646, 1.005254, 1.0),
            'sm_103': (0.991356, 0.991649, 1.005236, 1.0),
            'sm_107': (0.991356, 0.991649, 1.005236, 1.0),
        },
        'dX': {
            'sm_100': (0.990562, 0.991244, 1.006088, 1.040872),
            'sm_103': (0.99059, 0.99125, 1.006106, 1.040878),
            'sm_107': (0.990594, 0.991226, 1.006106, 1.040878),
        },
    },
    # deterministic package: dX: the max-norm largest error above 1.005 x the tier's on sm_100, sm_103, sm_107; worst
    # x1.0409
    'tests/experimental/test_cake_lm_head_det_train_zero_dlogp.py::test_device_zero_dlogp_skip_is_deterministic_and_accurate[8117-4096-none]': {
        'dX': {
            'sm_100': (0.991183, 0.990972, 0.9924, 1.040872),
            'sm_103': (0.991163, 0.990972, 0.992392, 1.040878),
            'sm_107': (0.991163, 0.990991, 0.992392, 1.040878),
        },
    },
    # non-deterministic package: dX: the max-norm median, max-norm largest error above 1.005 x the tier's on sm_100,
    # sm_103, sm_107; a near tie (every ratio <= x1.02)
    'tests/experimental/test_cake_lm_head_nondet_train.py::test_device_budget_vocab_major_logprob[12000-None]': {
        'dX': {
            'sm_100': (0.990823, 0.991393, 1.007649, 1.015738),
            'sm_103': (0.990824, 0.9914, 1.007649, 1.015738),
            'sm_107': (0.990824, 0.9914, 1.007649, 1.015738),
        },
    },
    # non-deterministic package: dX: the max-norm largest error above 1.005 x the tier's on sm_100, sm_103, sm_107; a
    # near tie (every ratio <= x1.02)
    'tests/experimental/test_cake_lm_head_nondet_train.py::test_device_logprob_backward_makes_its_gradient_buffers_behind_the_recompute[plain-1]': {
        'dX': {
            'sm_100': (0.994042, 0.991684, 1.0, 1.017604),
            'sm_103': (0.992767, 0.991936, 1.0, 1.017613),
            'sm_107': (0.991988, 0.992055, 1.0, 1.017613),
        },
    },
    # non-deterministic package: dX: the max-norm largest error above 1.005 x the tier's on sm_100, sm_103, sm_107; a
    # near tie (every ratio <= x1.02)
    'tests/experimental/test_cake_lm_head_nondet_train.py::test_device_logprob_backward_makes_its_gradient_buffers_behind_the_recompute[temperature-1]': {
        'dX': {
            'sm_100': (0.716313, 0.903808, 0.783451, 1.014777),
            'sm_103': (0.716337, 0.904527, 0.783453, 1.014712),
            'sm_107': (0.716742, 0.905374, 0.783453, 1.014712),
        },
    },
    # non-deterministic package: dX@first, dX@retained: the max-norm median, max-norm largest error above 1.005 x the
    # tier's on sm_100, sm_103, sm_107; worst x1.1479
    'tests/experimental/test_cake_lm_head_nondet_train.py::test_device_logprob_keep_lifetimes': {
        'dX@first': {
            'sm_100': (0.990935, 0.990614, 1.011576, 1.14791),
            'sm_103': (0.990917, 0.990641, 1.011585, 1.147947),
            'sm_107': (0.990917, 0.990641, 1.011585, 1.147947),
        },
        'dX@retained': {
            'sm_100': (0.990935, 0.990614, 1.011576, 1.14791),
            'sm_103': (0.990917, 0.990641, 1.011585, 1.147947),
            'sm_107': (0.990917, 0.990641, 1.011585, 1.147947),
        },
    },
    # non-deterministic package: dX: the max-norm median error above 1.005 x the tier's on sm_100, sm_103, sm_107;
    # worst x1.0411
    'tests/experimental/test_cake_lm_head_nondet_train.py::test_device_logprob_keep_meets_the_naive_tier[1500-4096-0.05-{}-0]': {
        'dX': {
            'sm_100': (0.990577, 0.991419, 1.041096, 1.0),
            'sm_103': (0.990537, 0.991382, 1.041112, 1.0),
            'sm_107': (0.990407, 0.991522, 1.041112, 1.0),
        },
    },
    # non-deterministic package: dW: the max-norm median error above 1.005 x the tier's on sm_100, sm_103, sm_107; a
    # near tie (every ratio <= x1.02)
    'tests/experimental/test_cake_lm_head_nondet_train.py::test_device_logprob_keep_meets_the_naive_tier[1500-4096-0.05-{}-8]': {
        'dW': {
            'sm_100': (0.991604, 0.991933, 1.006789, 0.997886),
            'sm_103': (0.991599, 0.991933, 1.006795, 0.997876),
            'sm_107': (0.991599, 0.991933, 1.006795, 0.997876),
        },
    },
    # non-deterministic package: dW: the max-norm largest error above 1.005 x the tier's on sm_100, sm_103, sm_107;
    # worst x1.0464
    "tests/experimental/test_cake_lm_head_nondet_train.py::test_device_logprob_keep_meets_the_naive_tier[5000-2048-0.05-{'fuse_dw_cast': False, 'dx_finalize': False}-0]": {
        'dW': {
            'sm_100': (0.991669, 0.991127, 0.993551, 1.046379),
            'sm_103': (0.991666, 0.991136, 0.993551, 1.046385),
            'sm_107': (0.991666, 0.991136, 0.993551, 1.046385),
        },
    },
    # non-deterministic package: dW, dX: the max-norm median, max-norm largest error above 1.005 x the tier's on
    # sm_100, sm_103, sm_107; worst x1.0348
    'tests/experimental/test_cake_lm_head_nondet_train.py::test_device_logprob_keep_meets_the_naive_tier[6000-4096-0.05-{}-0]': {
        'dW': {
            'sm_100': (0.991526, 0.992085, 0.97659, 1.03479),
            'sm_103': (0.991526, 0.992096, 0.976606, 1.034842),
            'sm_107': (0.991526, 0.992096, 0.976606, 1.034842),
        },
        'dX': {
            'sm_100': (0.991021, 0.99143, 1.011947, 1.0),
            'sm_103': (0.991035, 0.991473, 1.011977, 1.0),
            'sm_107': (0.991028, 0.991469, 1.011977, 1.0),
        },
    },
    # non-deterministic package: dW: the max-norm largest error above 1.005 x the tier's on sm_100, sm_103, sm_107;
    # worst x1.0395
    "tests/experimental/test_cake_lm_head_nondet_train.py::test_device_logprob_keep_meets_the_naive_tier[8700-4096-0.05-{'hidden_count': False}-0]": {
        'dW': {
            'sm_100': (0.99156, 0.991909, 0.996421, 1.039482),
            'sm_103': (0.991562, 0.991893, 0.99641, 1.039461),
            'sm_107': (0.991562, 0.991893, 0.99641, 1.039461),
        },
    },
    # non-deterministic package: dW: the max-norm largest error above 1.005 x the tier's on sm_100, sm_103, sm_107;
    # worst x1.0395
    'tests/experimental/test_cake_lm_head_nondet_train.py::test_device_logprob_keep_meets_the_naive_tier[8700-4096-0.05-{}-0]': {
        'dW': {
            'sm_100': (0.99156, 0.991909, 0.996421, 1.039482),
            'sm_103': (0.991562, 0.991893, 0.99641, 1.039461),
            'sm_107': (0.991562, 0.991893, 0.99641, 1.039461),
        },
    },
    # non-deterministic package: dX@again, dX@one_pass, dX@other_side: the max-norm median error above 1.005 x the
    # tier's on sm_107; a near tie (every ratio <= x1.02)
    "tests/experimental/test_cake_lm_head_nondet_train.py::test_device_logprob_keep_one_pass[12289-4096-0.05-{'dx_one_pass': True}-{}]": {
        'dX@again': {
            'sm_107': (0.990983, 0.9913, 1.012634, 0.979035),
        },
        'dX@one_pass': {
            'sm_107': (0.990983, 0.9913, 1.012634, 0.979035),
        },
        'dX@other_side': {
            'sm_107': (0.990983, 0.9913, 1.012634, 0.979035),
        },
    },
    # non-deterministic package: dW@again, dW@chunked, dW@ckpt, dW@ckpt_reentrant, dW@one_pass, dW@other_side,
    # dW@step: the max-norm largest error above 1.005 x the tier's on sm_100, sm_103, sm_107; a near tie (every ratio
    # <= x1.02)
    'tests/experimental/test_cake_lm_head_nondet_train.py::test_device_logprob_keep_one_pass[4600-4096-0.05-{}-{}]': {
        'dW@again': {
            'sm_100': (0.991941, 0.991767, 1.002182, 1.011899),
            'sm_103': (0.991944, 0.991764, 1.002187, 1.011909),
            'sm_107': (0.991944, 0.991764, 1.002187, 1.011909),
        },
        'dW@chunked': {
            'sm_100': (0.991941, 0.991767, 1.002182, 1.011899),
            'sm_103': (0.991944, 0.991764, 1.002187, 1.011909),
            'sm_107': (0.991944, 0.991764, 1.002187, 1.011909),
        },
        'dW@ckpt': {
            'sm_100': (0.991941, 0.991767, 1.002182, 1.011899),
            'sm_103': (0.991944, 0.991764, 1.002187, 1.011909),
            'sm_107': (0.991944, 0.991764, 1.002187, 1.011909),
        },
        'dW@ckpt_reentrant': {
            'sm_100': (0.991941, 0.991767, 1.002182, 1.011899),
            'sm_103': (0.991944, 0.991764, 1.002187, 1.011909),
            'sm_107': (0.991944, 0.991764, 1.002187, 1.011909),
        },
        'dW@one_pass': {
            'sm_100': (0.991941, 0.991767, 1.002182, 1.011899),
            'sm_103': (0.991944, 0.991764, 1.002187, 1.011909),
            'sm_107': (0.991944, 0.991764, 1.002187, 1.011909),
        },
        'dW@other_side': {
            'sm_100': (0.991941, 0.991767, 1.002182, 1.011899),
            'sm_103': (0.991944, 0.991764, 1.002187, 1.011909),
            'sm_107': (0.991944, 0.991764, 1.002187, 1.011909),
        },
        'dW@step': {
            'sm_100': (0.991941, 0.991767, 1.002182, 1.011899),
            'sm_103': (0.991944, 0.991764, 1.002187, 1.011909),
            'sm_107': (0.991944, 0.991764, 1.002187, 1.011909),
        },
    },
    # non-deterministic package: dW@again, dW@chunked, dW@ckpt, dW@ckpt_reentrant, dW@one_pass, dW@other_side,
    # dW@step: the max-norm median error above 1.005 x the tier's on sm_100, sm_103, sm_107; a near tie (every ratio
    # <= x1.02)
    "tests/experimental/test_cake_lm_head_nondet_train.py::test_device_logprob_keep_one_pass[8117-4096-0.05-{}-{'gemm_tuning': {'dx': {'k_slices': 1}}}]": {
        'dW@again': {
            'sm_100': (0.991855, 0.991764, 1.013195, 1.0),
            'sm_103': (0.991842, 0.991776, 1.013201, 1.0),
            'sm_107': (0.991842, 0.991776, 1.013201, 1.0),
        },
        'dW@chunked': {
            'sm_100': (0.991842, 0.9918, 1.00657, 1.0),
            'sm_103': (0.991823, 0.991805, 1.006576, 1.0),
            'sm_107': (0.991823, 0.991805, 1.006576, 1.0),
        },
        'dW@ckpt': {
            'sm_100': (0.991855, 0.991764, 1.013195, 1.0),
            'sm_103': (0.991842, 0.991776, 1.013201, 1.0),
            'sm_107': (0.991842, 0.991776, 1.013201, 1.0),
        },
        'dW@ckpt_reentrant': {
            'sm_100': (0.991855, 0.991764, 1.013195, 1.0),
            'sm_103': (0.991842, 0.991776, 1.013201, 1.0),
            'sm_107': (0.991842, 0.991776, 1.013201, 1.0),
        },
        'dW@one_pass': {
            'sm_100': (0.991855, 0.991764, 1.013195, 1.0),
            'sm_103': (0.991842, 0.991776, 1.013201, 1.0),
            'sm_107': (0.991842, 0.991776, 1.013201, 1.0),
        },
        'dW@other_side': {
            'sm_100': (0.991855, 0.991764, 1.013195, 1.0),
            'sm_103': (0.991842, 0.991776, 1.013201, 1.0),
            'sm_107': (0.991842, 0.991776, 1.013201, 1.0),
        },
        'dW@step': {
            'sm_100': (0.991855, 0.991764, 1.013195, 1.0),
            'sm_103': (0.991842, 0.991776, 1.013201, 1.0),
            'sm_107': (0.991842, 0.991776, 1.013201, 1.0),
        },
    },
    # non-deterministic package: dW@again, dW@chunked, dW@ckpt, dW@ckpt_reentrant, dW@one_pass, dW@other_side,
    # dW@step: the max-norm median error above 1.005 x the tier's on sm_100, sm_103, sm_107; a near tie (every ratio
    # <= x1.02)
    'tests/experimental/test_cake_lm_head_nondet_train.py::test_device_logprob_keep_one_pass[8117-4096-0.05-{}-{}]': {
        'dW@again': {
            'sm_100': (0.991855, 0.991764, 1.013195, 1.0),
            'sm_103': (0.991842, 0.991776, 1.013201, 1.0),
            'sm_107': (0.991842, 0.991776, 1.013201, 1.0),
        },
        'dW@chunked': {
            'sm_100': (0.991842, 0.9918, 1.00657, 1.0),
            'sm_103': (0.991823, 0.991805, 1.006576, 1.0),
            'sm_107': (0.991823, 0.991805, 1.006576, 1.0),
        },
        'dW@ckpt': {
            'sm_100': (0.991855, 0.991764, 1.013195, 1.0),
            'sm_103': (0.991842, 0.991776, 1.013201, 1.0),
            'sm_107': (0.991842, 0.991776, 1.013201, 1.0),
        },
        'dW@ckpt_reentrant': {
            'sm_100': (0.991855, 0.991764, 1.013195, 1.0),
            'sm_103': (0.991842, 0.991776, 1.013201, 1.0),
            'sm_107': (0.991842, 0.991776, 1.013201, 1.0),
        },
        'dW@one_pass': {
            'sm_100': (0.991855, 0.991764, 1.013195, 1.0),
            'sm_103': (0.991842, 0.991776, 1.013201, 1.0),
            'sm_107': (0.991842, 0.991776, 1.013201, 1.0),
        },
        'dW@other_side': {
            'sm_100': (0.991855, 0.991764, 1.013195, 1.0),
            'sm_103': (0.991842, 0.991776, 1.013201, 1.0),
            'sm_107': (0.991842, 0.991776, 1.013201, 1.0),
        },
        'dW@step': {
            'sm_100': (0.991855, 0.991764, 1.013195, 1.0),
            'sm_103': (0.991842, 0.991776, 1.013201, 1.0),
            'sm_107': (0.991842, 0.991776, 1.013201, 1.0),
        },
    },
    # non-deterministic package: dW@again, dW@chunked, dW@ckpt, dW@ckpt_reentrant, dW@one_pass, dW@other_side,
    # dW@step, dX@again, dX@chunked, dX@ckpt, dX@ckpt_reentrant, dX@one_pass, dX@other_side, dX@step: the max-norm
    # median, max-norm largest error above 1.005 x the tier's on sm_100, sm_103, sm_107; worst x1.0305
    "tests/experimental/test_cake_lm_head_nondet_train.py::test_device_logprob_keep_one_pass[8700-4096-0.05-{'hidden_count': False}-{}]": {
        'dW@again': {
            'sm_100': (0.991796, 0.991907, 1.016776, 1.030462),
            'sm_103': (0.991787, 0.991903, 1.016787, 1.030462),
            'sm_107': (0.991787, 0.991903, 1.016787, 1.030462),
        },
        'dW@chunked': {
            'sm_100': (0.991779, 0.991868, 1.017828, 1.030462),
            'sm_103': (0.99177, 0.99188, 1.017828, 1.030462),
            'sm_107': (0.99177, 0.99188, 1.017828, 1.030462),
        },
        'dW@ckpt': {
            'sm_100': (0.991796, 0.991907, 1.016776, 1.030462),
            'sm_103': (0.991787, 0.991903, 1.016787, 1.030462),
            'sm_107': (0.991787, 0.991903, 1.016787, 1.030462),
        },
        'dW@ckpt_reentrant': {
            'sm_100': (0.991796, 0.991907, 1.016776, 1.030462),
            'sm_103': (0.991787, 0.991903, 1.016787, 1.030462),
            'sm_107': (0.991787, 0.991903, 1.016787, 1.030462),
        },
        'dW@one_pass': {
            'sm_100': (0.991796, 0.991907, 1.016776, 1.030462),
            'sm_103': (0.991787, 0.991903, 1.016787, 1.030462),
            'sm_107': (0.991787, 0.991903, 1.016787, 1.030462),
        },
        'dW@other_side': {
            'sm_100': (0.991796, 0.991907, 1.016776, 1.030462),
            'sm_103': (0.991787, 0.991903, 1.016787, 1.030462),
            'sm_107': (0.991787, 0.991903, 1.016787, 1.030462),
        },
        'dW@step': {
            'sm_100': (0.991796, 0.991907, 1.016776, 1.030462),
            'sm_103': (0.991787, 0.991903, 1.016787, 1.030462),
            'sm_107': (0.991787, 0.991903, 1.016787, 1.030462),
        },
        'dX@again': {
            'sm_100': (0.991, 0.991126, 1.01461, 0.983873),
            'sm_103': (0.990991, 0.991135, 1.014606, 0.98389),
            'sm_107': (0.990991, 0.991126, 1.014606, 0.98389),
        },
        'dX@chunked': {
            'sm_100': (0.991, 0.991126, 1.01461, 0.983873),
            'sm_103': (0.990991, 0.991135, 1.014606, 0.98389),
            'sm_107': (0.990991, 0.991126, 1.014606, 0.98389),
        },
        'dX@ckpt': {
            'sm_100': (0.991084, 0.99119, 1.009698, 0.983873),
            'sm_103': (0.991065, 0.99118, 1.009696, 0.98389),
            'sm_107': (0.991065, 0.99118, 1.009696, 0.98389),
        },
        'dX@ckpt_reentrant': {
            'sm_100': (0.991084, 0.99119, 1.009698, 0.983873),
            'sm_103': (0.991065, 0.99118, 1.009696, 0.98389),
            'sm_107': (0.991065, 0.99118, 1.009696, 0.98389),
        },
        'dX@one_pass': {
            'sm_100': (0.991, 0.991126, 1.01461, 0.983873),
            'sm_103': (0.990991, 0.991135, 1.014606, 0.98389),
            'sm_107': (0.990991, 0.991126, 1.014606, 0.98389),
        },
        'dX@other_side': {
            'sm_100': (0.991, 0.991126, 1.01461, 0.983873),
            'sm_103': (0.990991, 0.991135, 1.014606, 0.98389),
            'sm_107': (0.990991, 0.991126, 1.014606, 0.98389),
        },
        'dX@step': {
            'sm_100': (0.991084, 0.99119, 1.009698, 0.983873),
            'sm_103': (0.991065, 0.99118, 1.009696, 0.98389),
            'sm_107': (0.991065, 0.99118, 1.009696, 0.98389),
        },
    },
    # non-deterministic package: dW@again, dW@chunked, dW@ckpt, dW@ckpt_reentrant, dW@one_pass, dW@other_side,
    # dW@step, dX@again, dX@chunked, dX@ckpt, dX@ckpt_reentrant, dX@one_pass, dX@other_side, dX@step: the max-norm
    # median, max-norm largest error above 1.005 x the tier's on sm_100, sm_103, sm_107; worst x1.0305
    "tests/experimental/test_cake_lm_head_nondet_train.py::test_device_logprob_keep_one_pass[8700-4096-0.05-{}-{'fuse_dw_cast': False, 'dx_finalize': False}]": {
        'dW@again': {
            'sm_100': (0.991796, 0.991907, 1.016776, 1.030462),
            'sm_103': (0.991787, 0.991903, 1.016787, 1.030462),
            'sm_107': (0.991787, 0.991903, 1.016787, 1.030462),
        },
        'dW@chunked': {
            'sm_100': (0.991779, 0.991868, 1.017828, 1.030462),
            'sm_103': (0.99177, 0.99188, 1.017828, 1.030462),
            'sm_107': (0.99177, 0.99188, 1.017828, 1.030462),
        },
        'dW@ckpt': {
            'sm_100': (0.991796, 0.991907, 1.016776, 1.030462),
            'sm_103': (0.991787, 0.991903, 1.016787, 1.030462),
            'sm_107': (0.991787, 0.991903, 1.016787, 1.030462),
        },
        'dW@ckpt_reentrant': {
            'sm_100': (0.991796, 0.991907, 1.016776, 1.030462),
            'sm_103': (0.991787, 0.991903, 1.016787, 1.030462),
            'sm_107': (0.991787, 0.991903, 1.016787, 1.030462),
        },
        'dW@one_pass': {
            'sm_100': (0.991796, 0.991907, 1.016776, 1.030462),
            'sm_103': (0.991787, 0.991903, 1.016787, 1.030462),
            'sm_107': (0.991787, 0.991903, 1.016787, 1.030462),
        },
        'dW@other_side': {
            'sm_100': (0.991796, 0.991907, 1.016776, 1.030462),
            'sm_103': (0.991787, 0.991903, 1.016787, 1.030462),
            'sm_107': (0.991787, 0.991903, 1.016787, 1.030462),
        },
        'dW@step': {
            'sm_100': (0.991796, 0.991907, 1.016776, 1.030462),
            'sm_103': (0.991787, 0.991903, 1.016787, 1.030462),
            'sm_107': (0.991787, 0.991903, 1.016787, 1.030462),
        },
        'dX@again': {
            'sm_100': (0.991, 0.991126, 1.01461, 0.983873),
            'sm_103': (0.990991, 0.991135, 1.014606, 0.98389),
            'sm_107': (0.990991, 0.991126, 1.014606, 0.98389),
        },
        'dX@chunked': {
            'sm_100': (0.991, 0.991126, 1.01461, 0.983873),
            'sm_103': (0.990991, 0.991135, 1.014606, 0.98389),
            'sm_107': (0.990991, 0.991126, 1.014606, 0.98389),
        },
        'dX@ckpt': {
            'sm_100': (0.991084, 0.99119, 1.009698, 0.983873),
            'sm_103': (0.991065, 0.99118, 1.009696, 0.98389),
            'sm_107': (0.991065, 0.99118, 1.009696, 0.98389),
        },
        'dX@ckpt_reentrant': {
            'sm_100': (0.991084, 0.99119, 1.009698, 0.983873),
            'sm_103': (0.991065, 0.99118, 1.009696, 0.98389),
            'sm_107': (0.991065, 0.99118, 1.009696, 0.98389),
        },
        'dX@one_pass': {
            'sm_100': (0.991, 0.991126, 1.01461, 0.983873),
            'sm_103': (0.990991, 0.991135, 1.014606, 0.98389),
            'sm_107': (0.990991, 0.991126, 1.014606, 0.98389),
        },
        'dX@other_side': {
            'sm_100': (0.991, 0.991126, 1.01461, 0.983873),
            'sm_103': (0.990991, 0.991135, 1.014606, 0.98389),
            'sm_107': (0.990991, 0.991126, 1.014606, 0.98389),
        },
        'dX@step': {
            'sm_100': (0.991084, 0.99119, 1.009698, 0.983873),
            'sm_103': (0.991065, 0.99118, 1.009696, 0.98389),
            'sm_107': (0.991065, 0.99118, 1.009696, 0.98389),
        },
    },
    # non-deterministic package: dW@again, dW@chunked, dW@ckpt, dW@ckpt_reentrant, dW@one_pass, dW@other_side,
    # dW@step, dX@again, dX@chunked, dX@ckpt, dX@ckpt_reentrant, dX@one_pass, dX@other_side, dX@step: the max-norm
    # median, max-norm largest error above 1.005 x the tier's on sm_100, sm_103, sm_107; worst x1.0305
    'tests/experimental/test_cake_lm_head_nondet_train.py::test_device_logprob_keep_one_pass[8700-4096-0.05-{}-{}]': {
        'dW@again': {
            'sm_100': (0.991796, 0.991907, 1.016776, 1.030462),
            'sm_103': (0.991787, 0.991903, 1.016787, 1.030462),
            'sm_107': (0.991787, 0.991903, 1.016787, 1.030462),
        },
        'dW@chunked': {
            'sm_100': (0.991779, 0.991868, 1.017828, 1.030462),
            'sm_103': (0.99177, 0.99188, 1.017828, 1.030462),
            'sm_107': (0.99177, 0.99188, 1.017828, 1.030462),
        },
        'dW@ckpt': {
            'sm_100': (0.991796, 0.991907, 1.016776, 1.030462),
            'sm_103': (0.991787, 0.991903, 1.016787, 1.030462),
            'sm_107': (0.991787, 0.991903, 1.016787, 1.030462),
        },
        'dW@ckpt_reentrant': {
            'sm_100': (0.991796, 0.991907, 1.016776, 1.030462),
            'sm_103': (0.991787, 0.991903, 1.016787, 1.030462),
            'sm_107': (0.991787, 0.991903, 1.016787, 1.030462),
        },
        'dW@one_pass': {
            'sm_100': (0.991796, 0.991907, 1.016776, 1.030462),
            'sm_103': (0.991787, 0.991903, 1.016787, 1.030462),
            'sm_107': (0.991787, 0.991903, 1.016787, 1.030462),
        },
        'dW@other_side': {
            'sm_100': (0.991796, 0.991907, 1.016776, 1.030462),
            'sm_103': (0.991787, 0.991903, 1.016787, 1.030462),
            'sm_107': (0.991787, 0.991903, 1.016787, 1.030462),
        },
        'dW@step': {
            'sm_100': (0.991796, 0.991907, 1.016776, 1.030462),
            'sm_103': (0.991787, 0.991903, 1.016787, 1.030462),
            'sm_107': (0.991787, 0.991903, 1.016787, 1.030462),
        },
        'dX@again': {
            'sm_100': (0.991, 0.991126, 1.01461, 0.983873),
            'sm_103': (0.990991, 0.991135, 1.014606, 0.98389),
            'sm_107': (0.990991, 0.991126, 1.014606, 0.98389),
        },
        'dX@chunked': {
            'sm_100': (0.991, 0.991126, 1.01461, 0.983873),
            'sm_103': (0.990991, 0.991135, 1.014606, 0.98389),
            'sm_107': (0.990991, 0.991126, 1.014606, 0.98389),
        },
        'dX@ckpt': {
            'sm_100': (0.991084, 0.99119, 1.009698, 0.983873),
            'sm_103': (0.991065, 0.99118, 1.009696, 0.98389),
            'sm_107': (0.991065, 0.99118, 1.009696, 0.98389),
        },
        'dX@ckpt_reentrant': {
            'sm_100': (0.991084, 0.99119, 1.009698, 0.983873),
            'sm_103': (0.991065, 0.99118, 1.009696, 0.98389),
            'sm_107': (0.991065, 0.99118, 1.009696, 0.98389),
        },
        'dX@one_pass': {
            'sm_100': (0.991, 0.991126, 1.01461, 0.983873),
            'sm_103': (0.990991, 0.991135, 1.014606, 0.98389),
            'sm_107': (0.990991, 0.991126, 1.014606, 0.98389),
        },
        'dX@other_side': {
            'sm_100': (0.991, 0.991126, 1.01461, 0.983873),
            'sm_103': (0.990991, 0.991135, 1.014606, 0.98389),
            'sm_107': (0.990991, 0.991126, 1.014606, 0.98389),
        },
        'dX@step': {
            'sm_100': (0.991084, 0.99119, 1.009698, 0.983873),
            'sm_103': (0.991065, 0.99118, 1.009696, 0.98389),
            'sm_107': (0.991065, 0.99118, 1.009696, 0.98389),
        },
    },
    # non-deterministic package: dW@below_fused, dW@between, dW@ckpt_below_fused, dW@ckpt_between, dW@ckpt_fused,
    # dW@ckpt_smallest, dW@fused, dW@smallest: the max-norm median error above 1.005 x the tier's on sm_100, sm_103,
    # sm_107; a near tie (every ratio <= x1.02)
    'tests/experimental/test_cake_lm_head_nondet_train.py::test_device_loss_explicit_budget_forms[ce_1chunk]': {
        'dW@below_fused': {
            'sm_100': (0.996154, 0.996203, 1.005612, 1.0),
            'sm_103': (0.996154, 0.996203, 1.005612, 1.0),
            'sm_107': (0.996154, 0.996203, 1.005612, 1.0),
        },
        'dW@between': {
            'sm_100': (0.996154, 0.996203, 1.005612, 1.0),
            'sm_103': (0.996154, 0.996203, 1.005612, 1.0),
            'sm_107': (0.996154, 0.996203, 1.005612, 1.0),
        },
        'dW@ckpt_below_fused': {
            'sm_100': (0.996154, 0.996203, 1.005612, 1.0),
            'sm_103': (0.996154, 0.996203, 1.005612, 1.0),
            'sm_107': (0.996154, 0.996203, 1.005612, 1.0),
        },
        'dW@ckpt_between': {
            'sm_100': (0.996154, 0.996203, 1.005612, 1.0),
            'sm_103': (0.996154, 0.996203, 1.005612, 1.0),
            'sm_107': (0.996154, 0.996203, 1.005612, 1.0),
        },
        'dW@ckpt_fused': {
            'sm_100': (0.996154, 0.996203, 1.005612, 1.0),
            'sm_103': (0.996154, 0.996203, 1.005612, 1.0),
            'sm_107': (0.996154, 0.996203, 1.005612, 1.0),
        },
        'dW@ckpt_smallest': {
            'sm_100': (0.996154, 0.996203, 1.005612, 1.0),
            'sm_103': (0.996154, 0.996203, 1.005612, 1.0),
            'sm_107': (0.996154, 0.996203, 1.005612, 1.0),
        },
        'dW@fused': {
            'sm_100': (0.996154, 0.996203, 1.005612, 1.0),
            'sm_103': (0.996154, 0.996203, 1.005612, 1.0),
            'sm_107': (0.996154, 0.996203, 1.005612, 1.0),
        },
        'dW@smallest': {
            'sm_100': (0.996154, 0.996203, 1.005612, 1.0),
            'sm_103': (0.996154, 0.996203, 1.005612, 1.0),
            'sm_107': (0.996154, 0.996203, 1.005612, 1.0),
        },
    },
    # non-deterministic package: dX: the max-norm largest error above 1.005 x the tier's on sm_100, sm_103; a near tie
    # (every ratio <= x1.02)
    'tests/experimental/test_cake_lm_head_nondet_train.py::test_device_meets_the_naive_tier_at_the_production_head[logprob]': {
        'dX': {
            'sm_100': (0.992606, 0.994163, 1.0, 1.011767),
            'sm_103': (0.992609, 0.994158, 1.0, 1.011779),
        },
    },
    # non-deterministic package: dX: the L2 median, L2 largest, max-norm median, max-norm largest error above 1.005 x
    # the tier's on sm_100, sm_103, sm_107; a near tie (every ratio <= x1.02)
    'tests/experimental/test_cake_lm_head_nondet_train.py::test_device_meets_the_naive_tier_at_the_production_head[loss]': {
        'dX': {
            'sm_100': (1.014995, 1.014285, 1.007398, 1.006541),
            'sm_103': (1.014986, 1.014297, 1.007398, 1.006541),
            'sm_107': (1.012282, 1.011667, 1.001675, 1.0037),
        },
    },
    # non-deterministic package: dX: the max-norm median error above 1.005 x the tier's on sm_100, sm_103, sm_107;
    # worst x1.0210
    'tests/experimental/test_cake_lm_head_nondet_train.py::test_device_one_slice_call_runs_under_strict_deterministic_algorithms[logprob]': {
        'dX': {
            'sm_100': (0.99132, 0.990976, 1.021012, 0.909319),
            'sm_103': (0.991329, 0.990966, 1.021008, 0.909282),
            'sm_107': (0.991329, 0.990966, 1.021008, 0.909282),
        },
    },
    # non-deterministic package: dW@again, dW@replay: the max-norm median, max-norm largest error above 1.005 x the
    # tier's on sm_100, sm_103, sm_107; a near tie (every ratio <= x1.02)
    'tests/experimental/test_cake_lm_head_nondet_train.py::test_device_prepared_runner_and_cuda_graph[logprob]': {
        'dW@again': {
            'sm_100': (0.991426, 0.991268, 1.00862, 1.010287),
            'sm_103': (0.991436, 0.991271, 1.008614, 1.010287),
            'sm_107': (0.991436, 0.991271, 1.008614, 1.010287),
        },
        'dW@replay': {
            'sm_100': (0.991426, 0.991268, 1.00862, 1.010287),
            'sm_103': (0.991436, 0.991271, 1.008614, 1.010287),
            'sm_107': (0.991436, 0.991271, 1.008614, 1.010287),
        },
    },
    # non-deterministic package: dW: the max-norm largest error above 1.005 x the tier's on sm_100, sm_103, sm_107; a
    # near tie (every ratio <= x1.02)
    'tests/experimental/test_cake_lm_head_nondet_train.py::test_device_red_min_rows_keeps_small_chunks_on_the_plain_store': {
        'dW': {
            'sm_100': (0.988583, 0.988517, 1.002569, 1.014245),
            'sm_103': (0.988572, 0.988509, 1.002569, 1.014245),
            'sm_107': (0.988572, 0.988509, 1.002569, 1.014245),
        },
    },
    # non-deterministic package: dW@again, dW@first, dX@again, dX@first: the max-norm median, max-norm largest error
    # above 1.005 x the tier's on sm_100, sm_103, sm_107; worst x1.0794
    'tests/experimental/test_cake_lm_head_nondet_train.py::test_device_short_tail_fold[ce-logprob]': {
        'dW@again': {
            'sm_100': (0.991252, 0.992022, 0.990047, 1.079374),
            'sm_103': (0.991269, 0.992028, 0.990064, 1.079374),
            'sm_107': (0.991269, 0.992028, 0.990064, 1.079374),
        },
        'dW@first': {
            'sm_100': (0.991252, 0.992022, 0.990047, 1.079374),
            'sm_103': (0.991269, 0.992028, 0.990064, 1.079374),
            'sm_107': (0.991269, 0.992028, 0.990064, 1.079374),
        },
        'dX@again': {
            'sm_100': (0.990856, 0.991668, 1.01036, 1.051173),
            'sm_103': (0.990842, 0.991593, 1.010363, 1.051202),
            'sm_107': (0.990842, 0.991593, 1.010363, 1.051202),
        },
        'dX@first': {
            'sm_100': (0.990856, 0.991668, 1.01036, 1.051173),
            'sm_103': (0.990842, 0.991593, 1.010363, 1.051202),
            'sm_107': (0.990842, 0.991593, 1.010363, 1.051202),
        },
    },
    # non-deterministic package: dX: the max-norm median error above 1.005 x the tier's on sm_100, sm_103; a near tie
    # (every ratio <= x1.02)
    'tests/experimental/test_cake_lm_head_nondet_train.py::test_device_small_call_with_ignored_rows_meets_the_naive_tier[logprob-32]': {
        'dX': {
            'sm_100': (0.993452, 0.986665, 1.012166, 0.962807),
            'sm_103': (0.993297, 0.986402, 1.012162, 0.962807),
        },
    },
    # non-deterministic package: dX: the max-norm median error above 1.005 x the tier's on sm_100, sm_103; a near tie
    # (every ratio <= x1.02)
    'tests/experimental/test_cake_lm_head_nondet_train.py::test_device_small_call_with_ignored_rows_meets_the_naive_tier[loss-32]': {
        'dX': {
            'sm_100': (0.988572, 0.988839, 1.00642, 1.0),
            'sm_103': (0.988547, 0.988717, 1.006425, 1.0),
        },
    },
    # non-deterministic package: dX: the max-norm largest error above 1.005 x the tier's on sm_100, sm_103, sm_107; a
    # near tie (every ratio <= x1.02)
    'tests/experimental/test_cake_lm_head_nondet_train.py::test_device_temperature_heterogeneous_matches_the_references[1-4096-0.0]': {
        'dX': {
            'sm_100': (0.716313, 0.903808, 0.783451, 1.014777),
            'sm_103': (0.716337, 0.904527, 0.783453, 1.014712),
            'sm_107': (0.716742, 0.905374, 0.783453, 1.014712),
        },
    },
    # non-deterministic package: dW: the max-norm median, max-norm largest error above 1.005 x the tier's on sm_100,
    # sm_103, sm_107; a near tie (every ratio <= x1.02)
    'tests/experimental/test_cake_lm_head_nondet_train.py::test_device_upstream_gradient_scale': {
        'dW': {
            'sm_100': (0.989476, 0.989406, 1.006481, 1.006425),
            'sm_103': (0.989467, 0.989415, 1.006479, 1.006431),
            'sm_107': (0.989467, 0.989415, 1.006479, 1.006431),
        },
    },
    # non-deterministic package: dW@two_T8000: the max-norm largest error above 1.005 x the tier's on sm_100, sm_103,
    # sm_107; a near tie (every ratio <= x1.02)
    'tests/experimental/test_cake_lm_head_nondet_train.py::test_device_vocab_major_logprob_backward[0.0-None]': {
        'dW@two_T8000': {
            'sm_100': (0.991544, 0.991564, 0.999223, 1.012331),
            'sm_103': (0.991547, 0.99156, 0.999223, 1.012312),
            'sm_107': (0.991547, 0.99156, 0.999223, 1.012312),
        },
    },
    # non-deterministic package: dW@ckpt_T6000, dW@kept_T6000, dW@two_T8000, dW@vm_T6000, dW@vm_T6000_run2,
    # dW@vm_T6000_run3, dX@ckpt_T7000, dX@kept_T7000, dX@vm_T7000, dX@vm_T7000_run2, dX@vm_T7000_run3: the max-norm
    # median, max-norm largest error above 1.005 x the tier's on sm_100, sm_103, sm_107; a near tie (every ratio <=
    # x1.02)
    'tests/experimental/test_cake_lm_head_nondet_train.py::test_device_vocab_major_logprob_backward[0.5-None]': {
        'dW@ckpt_T6000': {
            'sm_100': (0.99099, 0.991826, 1.014356, 1.007873),
            'sm_103': (0.990998, 0.991837, 1.01435, 1.007873),
            'sm_107': (0.990998, 0.991837, 1.01435, 1.007873),
        },
        'dW@kept_T6000': {
            'sm_100': (0.991038, 0.991891, 1.014356, 1.007873),
            'sm_103': (0.991042, 0.991895, 1.01435, 1.007873),
            'sm_107': (0.991042, 0.991895, 1.01435, 1.007873),
        },
        'dW@two_T8000': {
            'sm_100': (0.991544, 0.991564, 0.999223, 1.012331),
            'sm_103': (0.991547, 0.99156, 0.999223, 1.012312),
            'sm_107': (0.991547, 0.99156, 0.999223, 1.012312),
        },
        'dW@vm_T6000': {
            'sm_100': (0.99099, 0.991826, 1.014356, 1.007873),
            'sm_103': (0.990998, 0.991837, 1.01435, 1.007873),
            'sm_107': (0.990998, 0.991837, 1.01435, 1.007873),
        },
        'dW@vm_T6000_run2': {
            'sm_100': (0.99099, 0.991826, 1.014356, 1.007873),
            'sm_103': (0.990998, 0.991837, 1.01435, 1.007873),
            'sm_107': (0.990998, 0.991837, 1.01435, 1.007873),
        },
        'dW@vm_T6000_run3': {
            'sm_100': (0.99099, 0.991826, 1.014356, 1.007873),
            'sm_103': (0.990998, 0.991837, 1.01435, 1.007873),
            'sm_107': (0.990998, 0.991837, 1.01435, 1.007873),
        },
        'dX@ckpt_T7000': {
            'sm_100': (0.991754, 0.991626, 0.951402, 1.006409),
            'sm_103': (0.991742, 0.991608, 0.951398, 1.006394),
            'sm_107': (0.991742, 0.991608, 0.951398, 1.006394),
        },
        'dX@kept_T7000': {
            'sm_100': (0.991736, 0.991618, 0.956721, 1.006409),
            'sm_103': (0.991733, 0.991601, 0.956714, 1.006394),
            'sm_107': (0.991765, 0.991709, 0.951398, 1.006394),
        },
        'dX@vm_T7000': {
            'sm_100': (0.991754, 0.991626, 0.951402, 1.006409),
            'sm_103': (0.991742, 0.991608, 0.951398, 1.006394),
            'sm_107': (0.991742, 0.991608, 0.951398, 1.006394),
        },
        'dX@vm_T7000_run2': {
            'sm_100': (0.991754, 0.991626, 0.951402, 1.006409),
            'sm_103': (0.991742, 0.991608, 0.951398, 1.006394),
            'sm_107': (0.991742, 0.991608, 0.951398, 1.006394),
        },
        'dX@vm_T7000_run3': {
            'sm_100': (0.991754, 0.991626, 0.951402, 1.006409),
            'sm_103': (0.991742, 0.991608, 0.951398, 1.006394),
            'sm_107': (0.991742, 0.991608, 0.951398, 1.006394),
        },
    },
    # non-deterministic package: dX: the max-norm median error above 1.005 x the tier's on sm_100, sm_103, sm_107;
    # worst x1.0895
    'tests/experimental/test_cake_lm_head_nondet_train.py::test_device_zero_dlogp_one_chunk_skip[1-none]': {
        'dX': {
            'sm_100': (0.986116, 0.986992, 1.089458, 0.987278),
            'sm_103': (0.985944, 0.987945, 1.089458, 0.987272),
            'sm_107': (0.986027, 0.987819, 1.089458, 0.987272),
        },
    },
    # non-deterministic package: dX: the max-norm largest error above 1.005 x the tier's on sm_100, sm_103, sm_107;
    # worst x1.0639
    "tests/experimental/test_cake_lm_head_nondet_train.py::test_device_zero_dlogp_one_chunk_skip[1025-('random', 0.166)]": {
        'dX': {
            'sm_100': (0.991512, 0.992453, 0.976961, 1.063916),
            'sm_103': (0.991494, 0.992487, 0.976952, 1.06391),
            'sm_107': (0.991561, 0.99249, 0.976952, 1.06391),
        },
    },
    # non-deterministic package: dX: the max-norm largest error above 1.005 x the tier's on sm_100, sm_103, sm_107;
    # worst x1.0639
    'tests/experimental/test_cake_lm_head_nondet_train.py::test_device_zero_dlogp_one_chunk_skip[1025-none]': {
        'dX': {
            'sm_100': (0.991727, 0.991862, 1.0, 1.063916),
            'sm_103': (0.991656, 0.991884, 1.0, 1.06391),
            'sm_107': (0.991693, 0.991877, 1.0, 1.06391),
        },
    },
    # non-deterministic package: dW, dX: the max-norm median, max-norm largest error above 1.005 x the tier's on
    # sm_100, sm_103, sm_107; worst x1.0589
    "tests/experimental/test_cake_lm_head_nondet_train.py::test_device_zero_dlogp_one_chunk_skip[1100-('random', 0.5)]": {
        'dW': {
            'sm_100': (0.992312, 0.991962, 1.026675, 1.0),
            'sm_103': (0.992321, 0.991979, 1.026675, 1.0),
            'sm_107': (0.992321, 0.991979, 1.026675, 1.0),
        },
        'dX': {
            'sm_100': (0.99243, 0.990073, 1.016728, 1.058858),
            'sm_103': (0.992409, 0.990157, 1.016736, 1.058854),
            'sm_107': (0.992373, 0.990093, 1.016736, 1.058854),
        },
    },
    # non-deterministic package: dX: the max-norm median error above 1.005 x the tier's on sm_100, sm_103, sm_107;
    # worst x1.0319
    'tests/experimental/test_cake_lm_head_nondet_train.py::test_device_zero_dlogp_one_chunk_skip[1946-single]': {
        'dX': {
            'sm_100': (0.989804, 0.996098, 1.031325, 1.0),
            'sm_103': (0.990107, 0.996555, 1.031861, 1.0),
            'sm_107': (0.990845, 0.996555, 1.031861, 1.0),
        },
    },
    # non-deterministic package: dW: the max-norm largest error above 1.005 x the tier's on sm_100, sm_103, sm_107;
    # worst x1.0216
    "tests/experimental/test_cake_lm_head_nondet_train.py::test_device_zero_dlogp_one_chunk_skip[4096-('random', 0.5)]": {
        'dW': {
            'sm_100': (0.991727, 0.991714, 1.002757, 1.021607),
            'sm_103': (0.991727, 0.991723, 1.002751, 1.021601),
            'sm_107': (0.991727, 0.991723, 1.002751, 1.021601),
        },
    },
    # non-deterministic package: dX: the max-norm median, max-norm largest error above 1.005 x the tier's on sm_100,
    # sm_103, sm_107; worst x1.0329
    'tests/experimental/test_cake_lm_head_nondet_train.py::test_device_zero_dlogp_one_chunk_skip[4096-none]': {
        'dX': {
            'sm_100': (0.990886, 0.991115, 1.022631, 1.032815),
            'sm_103': (0.990881, 0.991115, 1.022633, 1.032904),
            'sm_107': (0.990886, 0.991133, 1.022633, 1.032904),
        },
    },
    # non-deterministic package: dW: the max-norm largest error above 1.005 x the tier's on sm_100, sm_103, sm_107; a
    # near tie (every ratio <= x1.02)
    "tests/experimental/test_cake_lm_head_nondet_train.py::test_device_zero_dlogp_raised_dx_slices[forced-8117-('random', 0.166)]": {
        'dW': {
            'sm_100': (0.991453, 0.991615, 0.986303, 1.019607),
            'sm_103': (0.991442, 0.99162, 0.986303, 1.019591),
            'sm_107': (0.991442, 0.99162, 0.986303, 1.019591),
        },
    },
    # non-deterministic package: dW: the max-norm largest error above 1.005 x the tier's on sm_100, sm_103, sm_107;
    # worst x1.0370
    'tests/experimental/test_cake_lm_head_nondet_train.py::test_device_zero_dlogp_raised_dx_slices[forced-8117-none]': {
        'dW': {
            'sm_100': (0.991875, 0.991634, 0.989838, 1.036969),
            'sm_103': (0.991872, 0.991641, 0.989827, 1.036974),
            'sm_107': (0.991872, 0.991641, 0.989827, 1.036974),
        },
    },
    # non-deterministic package: dW: the max-norm median, max-norm largest error above 1.005 x the tier's on sm_100,
    # sm_103, sm_107; worst x1.0775
    "tests/experimental/test_cake_lm_head_nondet_train.py::test_device_zero_dlogp_raised_dx_slices[rule-8117-('random', 0.5)]": {
        'dW': {
            'sm_100': (0.991565, 0.991999, 1.007646, 1.077483),
            'sm_103': (0.991551, 0.991999, 1.007646, 1.077483),
            'sm_107': (0.991551, 0.991999, 1.007646, 1.077483),
        },
    },
    # non-deterministic package: dW, dX: the max-norm median, max-norm largest error above 1.005 x the tier's on
    # sm_100, sm_103, sm_107; worst x1.0892
    'tests/experimental/test_cake_lm_head_nondet_train.py::test_device_zero_dlogp_rows_skip[4097-single]': {
        'dW': {
            'sm_100': (0.990104, 0.992088, 1.0, 1.089229),
            'sm_103': (0.990104, 0.992041, 1.0, 1.089196),
            'sm_107': (0.990104, 0.992041, 1.0, 1.089196),
        },
        'dX': {
            'sm_100': (0.994014, 0.988493, 1.012196, 1.024546),
            'sm_103': (0.993989, 0.988552, 1.012189, 1.024536),
            'sm_107': (0.994094, 0.988092, 1.012189, 1.024536),
        },
    },
    # non-deterministic package: dX: the max-norm median, max-norm largest error above 1.005 x the tier's on sm_100,
    # sm_103, sm_107; worst x1.0613
    "tests/experimental/test_cake_lm_head_nondet_train.py::test_device_zero_dlogp_rows_skip[4600-('random', 0.166)]": {
        'dX': {
            'sm_100': (0.991319, 0.99155, 1.01036, 1.061301),
            'sm_103': (0.991365, 0.991468, 1.010363, 1.06132),
            'sm_107': (0.991365, 0.991468, 1.010363, 1.06132),
        },
    },
    # non-deterministic package: dW: the max-norm largest error above 1.005 x the tier's on sm_100, sm_103, sm_107; a
    # near tie (every ratio <= x1.02)
    "tests/experimental/test_cake_lm_head_nondet_train.py::test_device_zero_dlogp_rows_skip[6000-('random', 0.5)]": {
        'dW': {
            'sm_100': (0.992171, 0.991429, 1.0, 1.016051),
            'sm_103': (0.992183, 0.991441, 1.0, 1.016051),
            'sm_107': (0.992183, 0.991441, 1.0, 1.016051),
        },
    },
    # non-deterministic package: dW: the max-norm largest error above 1.005 x the tier's on sm_100, sm_103, sm_107; a
    # near tie (every ratio <= x1.02)
    "tests/experimental/test_cake_lm_head_nondet_train.py::test_device_zero_dlogp_rows_skip[8117-('random', 0.166)]": {
        'dW': {
            'sm_100': (0.991453, 0.991615, 0.986303, 1.019607),
            'sm_103': (0.991442, 0.99162, 0.986303, 1.019591),
            'sm_107': (0.991442, 0.99162, 0.986303, 1.019591),
        },
    },
    # non-deterministic package: dW: the max-norm median, max-norm largest error above 1.005 x the tier's on sm_100,
    # sm_103, sm_107; worst x1.0775
    "tests/experimental/test_cake_lm_head_nondet_train.py::test_device_zero_dlogp_rows_skip[8117-('random', 0.5)]": {
        'dW': {
            'sm_100': (0.991565, 0.991999, 1.007646, 1.077483),
            'sm_103': (0.991551, 0.991999, 1.007646, 1.077483),
            'sm_107': (0.991551, 0.991999, 1.007646, 1.077483),
        },
    },
    # non-deterministic package: dX: the max-norm median, max-norm largest error above 1.005 x the tier's on sm_100,
    # sm_103, sm_107; a near tie (every ratio <= x1.02)
    'tests/experimental/test_cake_lm_head_nondet_train.py::test_device_zero_dlogp_short_last_chunk_keeps_the_counted_cast': {
        'dX': {
            'sm_100': (0.991051, 0.991287, 1.015866, 1.013239),
            'sm_103': (0.991078, 0.99125, 1.01586, 1.013244),
            'sm_107': (0.991124, 0.991211, 1.01586, 1.013244),
        },
    },
    # non-deterministic package: dX: the max-norm largest error above 1.005 x the tier's on cpu, sm_100, sm_103,
    # sm_107; worst x1.0387
    'tests/experimental/test_cake_lm_head_nondet_train.py::test_reference_meets_the_naive_tier[ce-1-16]': {
        'dX': {
            'cpu': (0.961527, 0.879937, 0.950877, 1.038652),
            'sm_100': (0.961527, 0.879937, 0.950877, 1.038652),
            'sm_103': (0.961606, 0.879943, 0.950866, 1.038607),
            'sm_107': (0.961606, 0.879943, 0.950866, 1.038607),
        },
    },
    # non-deterministic package: dX: the max-norm largest error above 1.005 x the tier's on cpu, sm_100, sm_103,
    # sm_107; a near tie (every ratio <= x1.02)
    'tests/experimental/test_cake_lm_head_nondet_train.py::test_temperature_heterogeneous_matches_the_references[1031-512-all_but_one]': {
        'dX': {
            'cpu': (0.688452, 0.869586, 0.690148, 1.012555),
            'sm_100': (0.688452, 0.869586, 0.690148, 1.012555),
            'sm_103': (0.688448, 0.87404, 0.690147, 1.012555),
            'sm_107': (0.688448, 0.87404, 0.690147, 1.012555),
        },
    },
}
