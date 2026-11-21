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

The FLA reference of the KDA training accuracy tests: flash-linear-attention (FLA) 0.6.0 ``fla.ops.kda.chunk_kda`` on
the operands of the judged kernels, with each autotuned FLA kernel at the first config FLA lists for it.  It is the
judge's second reference (``tests/cake_naive.py``, the KDA rule): an end-to-end output (``out`` and the gradient of
every leaf) passes when it meets the naive BF16 tier or meets FLA under the same rule, and fails only when it misses
both.

FLA runs as the benchmarks' ``fla`` arm runs (``benchmarks/cake_kda_baselines.py``), except that each autotuned FLA
kernel runs the first config FLA lists for it: the Triton ``chunk_kda`` (``FLA_FLASH_KDA=0`` and ``FLA_TILELANG=0``
while it is imported and called), ``safe_gate=True``, ``use_beta_sigmoid_in_kernel=False``, the kernels' scale, int64
``cu_seqlens``, no initial and no final state.  The activated FP32 gate goes in with ``use_gate_in_kernel=False``; the
raw gate mode with ``use_gate_in_kernel=True``, ``A_log``, ``dt_bias`` and ``lower_bound``; the in-kernel Q/K L2 norm
with ``use_qk_l2norm_in_kernel=True``.  FLA's forward and its own backward give ``out`` and every gradient
(:func:`fla_of`), keyed as ``cake_kda_naive.oracle_with_grads``.  A KCP case is judged against FLA on one GPU over the
whole packed batch, and a checkpointed arm against the same FLA step as the plain arm.

The pin.  FLA's Triton autotuner times the configs FLA lists for a kernel and runs the fastest.  The configs of FLA's
two gate cumsum kernels sum in different orders, so that choice, made by timing noise, moved FLA's values between
processes and nodes.  The reference therefore runs every autotuned FLA kernel on the judged path at the first config
FLA 0.6.0 lists for it (``configs[0]``, after FLA's own platform filters), chosen by no timing: :data:`FLA_PINS`.
:func:`load` checks that no variable of :data:`FORBIDDEN_ENV` is set, imports FLA, checks FLA :data:`FLA_VERSION` and
Triton :data:`TRITON_VERSION`, checks that the first config of each kernel of :data:`FLA_PINS` is its pin, and leaves
each that one config: Triton's autotuner then runs it with no benchmark and no autotune cache.  After each call
:func:`fla_of` checks that no FLA kernel chose its config at run time.  The pin changes FLA objects for the whole test
process; the benchmarks do not import this module.

The record.  :func:`load` records the reference of its process (:func:`_record`): the versions of FLA, Triton and
torch, a digest of FLA's installed sources, the pin, the device and its shared memory per block (which decides FLA's
config lists), every ``TRITON_*``, ``FLA_*`` and ``PTXAS_*`` variable set while FLA runs, ``TRITON_PTXAS_PATH`` and
``TRITON_PTXAS_BLACKWELL_PATH`` as set (kept, never changed), the two ptxas Triton resolves and the one it runs for the
device (path, sha256 and ``--version``).  A judge's FLA steps carry it (:meth:`FlaSteps.summary`).  The family's
conftest runs each test under :func:`steps_of_test`: every process that calls :func:`fla_of` during the test (the
pytest process and the ranks the test spawns) writes its FLA steps to the test's directory (:data:`STEPS_DIR`), and a
test that records no FLA steps itself gets theirs.  The terminal summary (:func:`summarize_fla`) so prints the record
of every reference a test ran, in the pytest process or in a spawned rank.

FLA is a reference only where it runs the case.  When the operands are not on a CUDA device or not ``[S, H, D]``, or
an activated gate is outside FLA's safe gate range (:data:`SAFE_GATE_RANGE`), FLA does not run them: :func:`fla_of`
returns the reason (``fla unavailable: ...``, :func:`_check`, before any FLA call) instead of values, the judge judges
the output against the naive tier only, and a miss of the tier stays a failure.  FLA runs every other draw, and
:func:`fla_of` raises :class:`FlaReferenceError` (``the pinned FLA reference cannot run: ...``) and the test fails when
the pinned reference gives no values for it: FLA or Triton does not import or is another version, a variable of
:data:`FORBIDDEN_ENV` is set, a first config is not its pin, the FLA call raises (a Triton build error, a CUDA error,
out of memory or an error of FLA's own), or an FLA kernel chose its config at run time.  FLA runs on every draw (not
only after a miss): after its first call in a process its step costs a small part of the FP64 reference and of the
naive tier.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator, Mapping

import torch

#: The FLA release the judge reads.
FLA_VERSION = "0.6.0"
#: The Triton release FLA's kernels are built with: their code and FLA's config lists depend on it.
TRITON_VERSION = "3.8.0"
FLA_ENTRY = "fla.ops.kda.chunk_kda"
#: FLA's Triton ``chunk_kda``: the environment while FLA is imported and called.
FLA_ENV = {"FLA_FLASH_KDA": "0", "FLA_TILELANG": "0"}
#: Each autotuned FLA kernel on the judged path, ``(module, kernel)``, and ``str()`` of the first config FLA 0.6.0 lists
#: for it (``configs[0]``), with the line of FLA 0.6.0's list.  FLA's lists depend on the device: these are the first
#: configs on an NVIDIA Blackwell device (compute capability 10.x) with at least 166912 bytes of shared memory per block
#: (FLA's ``check_shared_mem('ampere')``; sm_100a and sm_103a have 232448).  :func:`load` checks them on the device.
# fmt: off
FLA_PINS = {
    # fla/ops/kda/gate.py:401 (BS_LIST: gate.py:22)
    ("fla.ops.kda.gate", "kda_gate_chunk_cumsum_vector_kernel"): "BS: 32, num_warps: 2, num_ctas: 1, num_stages: 3, maxnreg: None",
    # fla/ops/utils/cumsum.py:80 (BS_LIST: cumsum.py:18)
    ("fla.ops.utils.cumsum", "chunk_local_cumsum_vector_kernel"): "BS: 32, num_warps: 2, num_ctas: 1, num_stages: 3, maxnreg: None",
    # fla/ops/kda/chunk_intra.py:685
    ("fla.ops.kda.chunk_intra", "chunk_kda_fwd_kernel_intra_sub_chunk"): "num_warps: 1, num_ctas: 1, num_stages: 2, maxnreg: None",
    # fla/ops/kda/chunk_intra.py:37
    ("fla.ops.kda.chunk_intra", "chunk_kda_fwd_kernel_inter_solve_fused"): "BK: 32, num_warps: 1, num_ctas: 1, num_stages: 3, maxnreg: None",
    # fla/ops/kda/wy_fast.py:24
    ("fla.ops.kda.wy_fast", "recompute_w_u_fwd_kda_kernel"): "num_warps: 2, num_ctas: 1, num_stages: 2, maxnreg: None",
    # fla/ops/common/chunk_delta_h.py:48 (num_warps [2] on Blackwell: chunk_delta_h.py:33)
    ("fla.ops.common.chunk_delta_h", "chunk_gated_delta_rule_fwd_kernel_h_blockdim64"): "BV: 32, num_warps: 2, num_ctas: 1, num_stages: 2, maxnreg: None",
    # fla/ops/gla/chunk.py:351
    ("fla.ops.gla.chunk", "chunk_gla_fwd_kernel_o"): "BK: 32, BV: 64, num_warps: 2, num_ctas: 1, num_stages: 2, maxnreg: None",
    # fla/ops/kda/chunk_bwd.py:34 (NUM_WARPS: chunk_bwd.py:28)
    ("fla.ops.kda.chunk_bwd", "chunk_kda_bwd_kernel_dAv"): "num_warps: 2, num_ctas: 1, num_stages: 2, maxnreg: None",
    # fla/ops/common/chunk_delta_h.py:359
    ("fla.ops.common.chunk_delta_h", "chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64"): "BV: 32, num_warps: 2, num_ctas: 1, num_stages: 2, maxnreg: None",
    # fla/ops/kda/chunk_bwd.py:125 (BK_LIST, BV_LIST, NUM_WARPS: chunk_bwd.py:26-28)
    ("fla.ops.kda.chunk_bwd", "chunk_kda_bwd_kernel_wy_dqkg_fused"): "BK: 32, BV: 64, num_warps: 2, num_ctas: 1, num_stages: 2, maxnreg: None",
    # fla/ops/kda/chunk_intra.py:393
    ("fla.ops.kda.chunk_intra", "chunk_kda_bwd_kernel_intra"): "num_warps: 1, num_ctas: 1, num_stages: 2, maxnreg: None",
    # fla/ops/kda/gate.py:165 (NUM_WARPS_AUTOTUNE [4, 8, 16, 32] on NVIDIA: gate.py:24)
    ("fla.ops.kda.gate", "kda_gate_bwd_kernel"): "num_warps: 4, num_ctas: 1, num_stages: 2, maxnreg: None",
    # fla/modules/norm/l2norm.py:63 (BT_LIST: l2norm.py:17; the module chunk.py:15 imports)
    ("fla.modules.norm.l2norm", "l2norm_fwd_kernel"): "BT: 8, num_warps: 1, num_ctas: 1, num_stages: 3, maxnreg: None",
    # fla/modules/norm/l2norm.py:87
    ("fla.modules.norm.l2norm", "l2norm_bwd_kernel"): "BT: 8, num_warps: 1, num_ctas: 1, num_stages: 3, maxnreg: None",
}
# fmt: on
#: The pin in the record of the reference.
FLA_PIN = f"first-config x{len(FLA_PINS)}"
#: Variables that change FLA's or Triton's code or numbers: the reference does not run with one of them set.
FORBIDDEN_ENV = (
    "FLA_USE_FAST_OPS",
    "FLA_USE_TMA",
    "FLA_CACHE_MODE",
    "TRITON_F32_DEFAULT",
    "TRITON_DEFAULT_FP_FUSION",
    "TRITON_KERNEL_OVERRIDE",
    "TRITON_OVERRIDE_ARCH",
    "PTXAS_OPTIONS",
    "DISABLE_PTXAS_OPT",
)
#: The ptxas Triton uses: recorded as set and never changed (sm_107a with CUDA 13.5 needs both set).
RECORDED_ENV = ("TRITON_PTXAS_PATH", "TRITON_PTXAS_BLACKWELL_PATH")
#: Every variable set with one of these prefixes while FLA runs is in the record of the reference.
RECORDED_PREFIXES = ("TRITON_", "FLA_", "PTXAS_")
#: The activated gates FLA's ``safe_gate=True`` takes.
SAFE_GATE_RANGE = (-5.0, 0.0)
#: The start of every reason FLA gives no values for a draw.
UNAVAILABLE = "fla unavailable"
#: The start of every reason the pinned reference cannot run (:class:`FlaReferenceError`).
NOT_PINNED = "the pinned FLA reference cannot run"
GRAD_NAMES = ("dq", "dk", "dv", "dg", "dbeta")
RAW_GRAD_NAMES = GRAD_NAMES + ("dA_log", "ddt_bias")
#: The record of the FLA steps of a judge (``record_property``), which the family's ``conftest.py`` prints.
PROPERTY = "fla_reference"
#: The directory where each process writes its FLA steps of the running test (:func:`steps_of_test`): set for each test
#: by the family's ``conftest.py``, inherited by the ranks the test spawns.
STEPS_DIR = "CAKE_KDA_FLA_STEPS_DIR"

_loaded: tuple | None = None
#: The record of the pinned reference of this process (:func:`load`), or None before FLA ran in it.
_reference: dict | None = None
#: This process's FLA steps of the running test, ``(STEPS_DIR, pid, FlaSteps)`` (:func:`_note`).
_steps: tuple | None = None


class FlaReferenceError(RuntimeError):
    """FLA runs the case and the pinned reference cannot (:data:`NOT_PINNED`): the test fails."""


@contextlib.contextmanager
def _environment(values: Mapping[str, str]) -> Iterator[None]:
    """Set the environment variables ``values`` for the block, then restore them."""
    saved = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def load() -> tuple:
    """``(chunk_kda, None)`` with every kernel of :data:`FLA_PINS` pinned, or ``(None, reason)`` when the pinned
    reference cannot run (:data:`NOT_PINNED`; cached): :func:`fla_of` then raises :class:`FlaReferenceError`."""
    global _loaded
    if _loaded is None:
        try:
            _loaded = (_load(), None)
        except Exception as exc:  # noqa: BLE001 -- fla_of raises FlaReferenceError with the reason
            _loaded = (None, f"{NOT_PINNED}: {type(exc).__name__}: {exc}"[:800])
    return _loaded


def _load():
    """FLA's ``chunk_kda`` with every kernel of :data:`FLA_PINS` pinned; sets the record of the reference."""
    global _reference
    forbidden = [key for key in FORBIDDEN_ENV if key in os.environ]
    if forbidden:
        raise RuntimeError(f"{', '.join(forbidden)} set: each changes FLA's or Triton's code or numbers")
    with _environment(FLA_ENV):
        import fla
        import triton
        from fla.ops.kda import chunk_kda

        fla_version = getattr(fla, "__version__", None)
        triton_version = getattr(triton, "__version__", None)
        if (fla_version, triton_version) != (FLA_VERSION, TRITON_VERSION):
            raise RuntimeError(
                f"fla {fla_version} and triton {triton_version} are installed, the "
                f"reference is fla {FLA_VERSION} with triton {TRITON_VERSION}"
            )
        _pin()
        _reference = _record(fla, triton)
    return chunk_kda


def _autotuner(obj):
    """The Triton autotuner of kernel ``obj`` (through ``@triton.heuristics`` wrappers), or None."""
    from triton.runtime.autotuner import Autotuner
    from triton.runtime.jit import KernelInterface

    while isinstance(obj, KernelInterface) and not isinstance(obj, Autotuner):
        obj = getattr(obj, "fn", None)
    return obj if isinstance(obj, Autotuner) else None


def _pin() -> None:
    """Leave each kernel of :data:`FLA_PINS` its first config, once the first config of every one is its pin: Triton's
    autotuner then takes its one-config path (no benchmark, no autotune cache)."""
    import importlib

    tuners, problems = [], []
    for (module, name), want in FLA_PINS.items():
        tuner = _autotuner(getattr(importlib.import_module(module), name, None))
        if tuner is None:
            problems.append(f"{module}.{name} is not an autotuned Triton kernel")
        elif str(tuner.configs[0]) != want:
            problems.append(f"{module}.{name}: the first config is '{tuner.configs[0]}', the pin '{want}'")
        else:
            tuners.append(tuner)
    if problems:
        raise RuntimeError("; ".join(problems))
    for tuner in tuners:
        tuner.configs = [tuner.configs[0]]
        tuner.cache.clear()


def _fla_autotuners() -> dict:
    """Every Triton autotuner of FLA's imported modules: ``id -> (module.kernel, autotuner)``."""
    from triton.runtime.jit import KernelInterface

    found: dict = {}
    for module_name, module in list(sys.modules.items()):
        if module_name != "fla" and not module_name.startswith("fla."):
            continue
        for name, obj in list(getattr(module, "__dict__", {}).items()):
            tuner = _autotuner(obj) if isinstance(obj, KernelInterface) else None
            if tuner is not None:
                found.setdefault(id(tuner), (f"{module_name}.{name}", tuner))
    return found


def _cache_sizes() -> dict:
    """The size of each FLA autotuner's cache (one entry per key it chose a config for at run time)."""
    return {key: len(tuner.cache) for key, (_, tuner) in _fla_autotuners().items()}


def _assert_pinned(before: Mapping) -> None:
    """Raise :class:`FlaReferenceError` when an FLA autotuner chose a config at run time since ``before``
    (:func:`_cache_sizes`): by a benchmark, from Triton's disk cache or by pruning its configs to one."""
    chose = sorted(name for key, (name, tuner) in _fla_autotuners().items() if len(tuner.cache) > before.get(key, 0))
    if chose:
        raise FlaReferenceError(f"{NOT_PINNED}: FLA kernels chose their config at run time: {', '.join(chose)}")


def _record(fla, triton) -> dict:
    """The record of the pinned reference (:data:`_reference`)."""
    from fla.utils import get_all_max_shared_mem
    from triton import knobs

    major, minor = torch.cuda.get_device_capability(0)
    record = {
        "fla": fla.__version__,
        "triton": triton.__version__,
        "pin": FLA_PIN,
        "device": f"{torch.cuda.get_device_name(0)} sm_{major}{minor}",
        # what FLA's check_shared_mem reads (device 0)
        "max_shared_mem": get_all_max_shared_mem()[0],
    }
    record.update({key: os.environ.get(key) for key in RECORDED_ENV})
    for tool in ("ptxas", "ptxas_blackwell"):
        try:
            found = getattr(knobs.nvidia, tool)
            record[tool] = f"{found.path} (release {found.version})"
        except Exception as exc:  # noqa: BLE001 -- recorded only
            record[tool] = f"not resolved ({type(exc).__name__}: {exc})"
    record["ptxas_used"] = _recorded(_ptxas_used, 10 * major + minor)
    record["torch"] = f"{torch.__version__} (CUDA {torch.version.cuda})"
    record["fla_sources"] = _recorded(_sources, fla)
    record["env"] = " ".join(f"{k}={v}" for k, v in sorted(os.environ.items()) if k.startswith(RECORDED_PREFIXES))
    return record


def _recorded(value, *args) -> str:
    """``value(*args)``, or why it is not known: an entry of the record, never a reason the reference cannot run."""
    try:
        return value(*args)
    except Exception as exc:  # noqa: BLE001 -- recorded only
        return f"not resolved ({type(exc).__name__}: {exc})"


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _ptxas_used(arch: int) -> str:
    """The ptxas Triton runs for compute capability ``arch`` (``10 * major + minor``, Triton's ``get_ptxas``): its
    path, sha256 and ``--version`` output (its lines joined by `` / ``)."""
    from triton.backends.nvidia.compiler import get_ptxas

    path = get_ptxas(arch).path
    out = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=120, check=True).stdout
    version = " / ".join(line.strip() for line in out.splitlines() if line.strip())
    return f"sm_{arch} {path} sha256 {_sha256(path)} :: {version}"


def _sources(fla) -> str:
    """The sha256 of the ``sha256sum`` lines of every ``.py`` file of the installed package ``fla``, in C order of their
    paths relative to its parent directory: ``find fla -name '*.py' | LC_ALL=C sort | xargs sha256sum | sha256sum``
    there."""
    package = os.path.dirname(fla.__file__)
    parent = os.path.dirname(package)
    names = sorted(
        os.path.relpath(os.path.join(root, name), parent)
        for root, _, files in os.walk(package)
        for name in files
        if name.endswith(".py")
    )
    lines = "".join(f"{_sha256(os.path.join(parent, name))}  {name}\n" for name in names)
    return f"sha256 {hashlib.sha256(lines.encode()).hexdigest()} over {len(names)} .py files under {package}"


class FlaReference:
    """FLA's values of one draw's outputs (:func:`fla_of`), or the reason there are none, and the step's seconds."""

    def __init__(self, values: dict[str, torch.Tensor] | None, reason: str | None, seconds: float = 0.0) -> None:
        self.values, self.reason, self.seconds = values, reason, seconds

    def value(self, name: str, actual: torch.Tensor) -> torch.Tensor | str:
        """FLA's value of output ``name`` in the shape of ``actual`` (a reshape of the same elements, as the judge reads
        the reference), or the reason there is none: the ``fla`` of ``TierJudge.add``."""
        if self.values is None:
            return self.reason
        found = self.values.get(name)
        if found is None:
            return f"no FLA counterpart: FLA gives no {name}"
        if found.shape != actual.shape and found.numel() == actual.numel():
            return found.reshape(actual.shape)
        return found


def _check(inputs: Mapping[str, torch.Tensor]) -> str | None:
    """Why FLA does not run these operands, or None."""
    q = inputs["q"]
    if not q.is_cuda:
        return f"{UNAVAILABLE}: the operands are on {q.device}, FLA runs on CUDA devices"
    if q.dim() != 3:
        return f"{UNAVAILABLE}: operands of shape {tuple(q.shape)}, not [S, H, D]"
    if "A_log" not in inputs:
        g = inputs["g"]
        lo, hi = SAFE_GATE_RANGE
        if g.numel() and not bool(((g >= lo) & (g <= hi)).all()):
            return (
                f"{UNAVAILABLE}: the activated gate is outside FLA's safe gate range [{lo}, {hi}] "
                f"(min {float(g.min())}, max {float(g.max())}, finite {bool(g.isfinite().all())})"
            )
    return None


def fla_of(inputs: Mapping[str, torch.Tensor], scale: float, lower_bound: float, l2n: bool) -> FlaReference:
    """FLA's ``out`` and gradients on the operands of a judged draw (as the FP64 reference reads them, without a batch
    axis: ``q, k, v, g, beta, dout, cu_seqlens`` and in the raw gate mode ``A_log``, ``dt_bias``), keyed as
    ``cake_kda_naive.oracle_with_grads``, with ``scale``, the raw gate's ``lower_bound`` and with ``l2n`` the in-kernel
    Q/K L2 norm; or the reason FLA does not run them (:data:`UNAVAILABLE`, :func:`_check`).  Raises
    :class:`FlaReferenceError` (:data:`NOT_PINNED`) when FLA runs them and the pinned reference gives no values: it
    cannot load, the FLA call raises any error, or an FLA kernel chose its config at run time."""
    reason = _check(inputs)
    if reason is not None:
        return _note(FlaReference(None, reason))
    chunk_kda, reason = load()
    if chunk_kda is None:
        raise FlaReferenceError(reason)
    raw = "A_log" in inputs
    names = ("q", "k", "v", "g", "beta") + (("A_log", "dt_bias") if raw else ())
    device = inputs["q"].device
    before = _cache_sizes()
    start = time.perf_counter()
    try:
        with _environment(FLA_ENV), torch.cuda.device(device), torch.enable_grad():
            # dt_bias flat [H * 128]: FLA's gradient comes back in that shape (FlaReference.value reshapes it)
            leaves = [
                (inputs[n].reshape(-1) if n == "dt_bias" else inputs[n]).detach().clone().contiguous()
                for n in names
            ]
            for leaf in leaves:
                leaf.requires_grad_(True)
            kw = {
                "scale": scale,
                "initial_state": None,
                "output_final_state": False,
                "use_qk_l2norm_in_kernel": bool(l2n),
                "use_beta_sigmoid_in_kernel": False,
                "safe_gate": True,
                "cu_seqlens": inputs["cu_seqlens"].long(),
            }
            if raw:
                kw.update(use_gate_in_kernel=True, A_log=leaves[5], dt_bias=leaves[6], lower_bound=lower_bound)
            else:
                kw["use_gate_in_kernel"] = False
            o, _ = chunk_kda(*(x.unsqueeze(0) for x in leaves[:5]), **kw)
            o = o.squeeze(0)
            grads = torch.autograd.grad(o, leaves, inputs["dout"].to(o.dtype).contiguous(), allow_unused=True)
            torch.cuda.synchronize(device)
    except Exception as exc:  # FLA runs the case (_check): any error of the call fails the reference
        raise FlaReferenceError(f"{NOT_PINNED}: {FLA_ENTRY} raised {type(exc).__name__}: {exc}"[:800]) from exc
    _assert_pinned(before)
    values = {"out": o.detach()}
    for name, grad in zip(RAW_GRAD_NAMES if raw else GRAD_NAMES, grads, strict=True):
        if grad is not None:
            values[name] = grad.detach()
    return _note(FlaReference(values, None, time.perf_counter() - start))


class FlaSteps:
    """The FLA steps of one judge: their count, seconds, the reasons of the draws without FLA values and the record of
    the pinned reference, recorded with ``record_property`` (:data:`PROPERTY`) for the conftest's summary."""

    def __init__(self) -> None:
        self.calls, self.seconds, self.reasons = 0, 0.0, {}

    def add(self, ref: FlaReference) -> FlaReference:
        self.calls += 1
        self.seconds += ref.seconds
        if ref.values is None:
            self.reasons[ref.reason] = self.reasons.get(ref.reason, 0) + 1
        return ref

    def summary(self) -> dict:
        return {
            "calls": self.calls,
            "seconds": round(self.seconds, 3),
            "reasons": dict(self.reasons),
            "reference": _reference,
        }

    def record(self, record_property) -> None:
        record_property(PROPERTY, self.summary())


def _process() -> str:
    """This process in the FLA steps of a test: its rank when it is in a process group, and its pid."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return f"rank {torch.distributed.get_rank()} (pid {os.getpid()})"
    return f"pid {os.getpid()}"


def _note(ref: FlaReference) -> FlaReference:
    """``ref``, added to this process's FLA steps of the running test, which go to ``<STEPS_DIR>/<pid>.json``."""
    global _steps
    directory = os.environ.get(STEPS_DIR)
    if not directory:
        return ref
    if _steps is None or _steps[:2] != (directory, os.getpid()):  # a new test, or a forked process
        _steps = (directory, os.getpid(), FlaSteps())
    _steps[2].add(ref)
    path = os.path.join(directory, f"{os.getpid()}.json")
    try:
        with open(path + ".part", "w") as fh:
            json.dump({**_steps[2].summary(), "process": _process()}, fh)
        os.replace(path + ".part", path)
    except OSError:  # the test's directory is gone (a process that outlived its test): nothing left to record
        pass
    return ref


def _steps_of(directory: str) -> dict | None:
    """The FLA steps the processes of a test wrote (:func:`_note`), summed (``calls``, ``seconds``, ``reasons``), the
    first record of a pinned reference among them (``reference``; any other in ``other_references``) and the processes
    (``processes``); None when no process called :func:`fla_of`."""
    found = []
    for name in sorted(os.listdir(directory)):
        if name.endswith(".json"):
            with open(os.path.join(directory, name)) as fh:
                found.append(json.load(fh))
    if not found:
        return None
    found.sort(key=lambda steps: steps["process"])
    reasons: dict = {}
    records: list = []
    for steps in found:
        for reason, count in steps["reasons"].items():
            reasons[reason] = reasons.get(reason, 0) + count
        if steps["reference"] and steps["reference"] not in records:
            records.append(steps["reference"])
    summed = {
        "calls": sum(steps["calls"] for steps in found),
        "seconds": round(sum(steps["seconds"] for steps in found), 3),
        "reasons": reasons,
        "reference": records[0] if records else None,
        "processes": [steps["process"] for steps in found],
    }
    if len(records) > 1:
        summed["other_references"] = records[1:]
    return summed


@contextlib.contextmanager
def steps_of_test(item) -> Iterator[None]:
    """Run test ``item`` with :data:`STEPS_DIR` set to a directory of its own.  Afterwards, unless the test recorded FLA
    steps itself (:data:`PROPERTY`), record the FLA steps of every process that called :func:`fla_of` during it (the
    pytest process and the ranks it spawned, :func:`_steps_of`).  The family conftest's ``pytest_runtest_call``."""
    with (
        tempfile.TemporaryDirectory(prefix="cake_kda_fla_steps_", ignore_cleanup_errors=True) as directory,
        _environment({STEPS_DIR: directory}),
    ):
        try:
            yield
        finally:
            if all(key != PROPERTY for key, _ in item.user_properties):
                steps = _steps_of(directory)
                if steps is not None:
                    item.user_properties.append((PROPERTY, steps))


def fla_line(test_id: str, steps: Mapping) -> str:
    """``FLA <test> calls=<n> seconds=<s>``, the processes that called FLA when they are recorded (:func:`_steps_of`)
    and the reasons of the draws without FLA values."""
    reasons = "; ".join(f"{reason} x{count}" for reason, count in steps.get("reasons", {}).items())
    processes = f" ({', '.join(steps['processes'])})" if steps.get("processes") else ""
    return f"FLA {test_id} calls={steps['calls']} seconds={steps['seconds']}{processes}" + (
        f" | {reasons}" if reasons else ""
    )


def reference_line(record: Mapping) -> str:
    """``FLA reference <key>: <value>; ...``: the record of a pinned reference (:meth:`FlaSteps.summary`)."""
    return "FLA reference " + "; ".join(f"{key}: {value}" for key, value in record.items())


def summarize_fla(terminalreporter) -> None:
    """The terminal summary's ``FLA`` lines: the record of each pinned reference the tests ran, then one line per test
    that called FLA.  Call it from the family conftest's ``pytest_terminal_summary``."""
    lines, records = [], []
    for reports in terminalreporter.stats.values():
        for report in reports:
            if getattr(report, "when", None) != "call":
                continue
            for key, steps in getattr(report, "user_properties", ()):
                if key == PROPERTY:
                    lines.append(fla_line(report.nodeid, steps))
                    for record in (steps.get("reference"), *steps.get("other_references", ())):
                        if record and record not in records:
                            records.append(record)
    if lines:
        terminalreporter.section("training accuracy: the FLA reference steps of each judged test")
        for record in records:
            terminalreporter.write_line(reference_line(record))
        for line in lines:
            terminalreporter.write_line(line)


__all__ = [
    "FLA_ENTRY",
    "FLA_ENV",
    "FLA_PIN",
    "FLA_PINS",
    "FLA_VERSION",
    "FORBIDDEN_ENV",
    "NOT_PINNED",
    "PROPERTY",
    "RECORDED_ENV",
    "RECORDED_PREFIXES",
    "SAFE_GATE_RANGE",
    "STEPS_DIR",
    "TRITON_VERSION",
    "UNAVAILABLE",
    "FlaReference",
    "FlaReferenceError",
    "FlaSteps",
    "fla_line",
    "fla_of",
    "load",
    "reference_line",
    "steps_of_test",
    "summarize_fla",
]
