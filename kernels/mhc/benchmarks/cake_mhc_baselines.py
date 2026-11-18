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

The reference arithmetic and the non-Cake implementations that the mHC benchmarks
(``bench_cake_mhc_det_train.py`` / ``bench_cake_mhc_nondet_train.py``) run beside the Cake packages.

The function is the packages' contract (``docs/API.md``, "mhc"): four BF16 streams ``X [T, 4, H]``, FP32 ``fn
[24, 4H]``, ``scale [3]`` and ``base [24]``, ``rms_eps = 1e-5``, ``hc_eps = 1e-6``, 20 Sinkhorn iterations::

    r = rsqrt(mean(u * u) + rms_eps),  u = float(X).reshape(T, 4H);   z = (u r) @ fn.T
    pre = sigmoid(z[:, :4] scale[0] + base[:4]) + hc_eps;   post = 2 sigmoid(z[:, 4:8] scale[1] + base[4:8])
    C = Sinkhorn(z[:, 8:] scale[2] + base[8:]):  row softmax + hc_eps, a column normalisation, 19 x (row, column)
    U = BF16(sum_i pre_i X_i);   Y[:, j] = BF16(post_j B + sum_i C_ij X_i)

and its backward from BF16 ``dY [T, 4, H]`` and ``dU [T, H]``: BF16 ``dX`` and ``dB``, FP32 ``dfn``, ``dscale``,
``dbase``.  The Sinkhorn count is the inputs' ``iters`` (:func:`make_inputs`, default 20; the packages run counts other
than 20 on their run-time-count programs), the same for every arm and for the references.

:func:`reference` evaluates the function by autograd in any dtype: float64 is the oracle (in row blocks of
:data:`BLOCK_TOKENS` tokens), bfloat16 the naive tier of the accuracy judge (``tests/cake_naive.py``: the same ops in
PyTorch BF16 on the same inputs).  Before an arm is timed, its outputs on the timed inputs are compared with the
oracle (:func:`check_outputs`, one draw: the contract and each output's error beside the naive tier's error on that
draw), and :func:`judge_arms` holds every arm to the judge itself (:data:`JUDGE_DRAWS` draws of a smaller case: in L2
and max norm, the median and largest error over the draws at most ``NEAR_TIE`` times the naive tier's).  An arm that
misses the judge is reported as a different contract and is never the fastest non-Cake implementation.

The non-Cake arms (each ``available()`` says why it cannot run on a device or software stack):

* ``eager_fp32``: the function above in PyTorch eager FP32 on the BF16 inputs (TF32 off), its backward by autograd;
* ``compile_default`` / ``compile_max_autotune``: the same Python function through ``torch.compile`` (``mode``
  default / ``"max-autotune-no-cudagraphs"``, ``fullgraph=True``, ``dynamic=False``, TF32 off).  The compiled code
  varies from compile to compile; with ``compile_trials > 1`` each trial compiles a fresh copy of the function with
  the inductor caches disabled and the fastest trial (by the median of one group of training steps) is the one timed
  (every trial's time is recorded);
* ``liger``: Liger Kernel's mHC autograd functions (``liger_kernel.ops.mhc``: the coefficient function with
  ``allow_fp32=True``, the pre-mix and the post / residual mix).  Liger's residual mix is ``out_o = sum_i h_res[o, i]
  x_i``, so it is given ``C`` transposed; its mapping matrix is a ``[4H, 24]`` parameter (Liger's layout) and the
  three scales are scalar parameters.  Its FP32 dot products follow ``TRITON_F32_DEFAULT`` (the benchmarks set
  ``tf32x3``; ``ieee`` builds no H 7168 program and does not compile on SM107);
* ``tk_cublas`` (forward only): the cuBLAS FP32 GEMM ``float(X) @ fn.T`` and the squared row norms in PyTorch, then
  TileKernels' fused pre kernel (``mhc_pre_big_fuse_fwd``: RMS scale, coefficients, Sinkhorn and ``U``) and its
  post kernel (``mhc_post_fwd``: ``Y``);
* ``tk_deepgemm`` (forward only): the same TileKernels kernels after DeepGEMM's ``tf32_hc_prenorm_gemm`` run twice,
  on the TF32-rounded ``fn`` and on its remainder ``fn - tf32(fn)`` (``X`` is exact in TF32, so the two partial
  products carry the FP32 GEMM's accuracy); the remainder's squared norms are discarded.

``tk_cublas`` and ``tk_deepgemm`` are compositions of this harness, not a library's entry point: no library ships the
whole forward on these kernels.  Each arm's :attr:`Arm.composite` says what was composed, and the reports label the
arm as a composite.

Every arm computes the whole function from the same inputs: the forward gives ``U`` and ``Y``; the backward the five
gradients.  The PyTorch, compiled and Liger arms record the autograd graph in the forward; ``bwd`` is the backward of a
recorded forward; ``ckpt`` is a forward without the graph, the forward again with it and the backward (the
recomputation of ``torch.utils.checkpoint``); ``ckpt_stats`` is the same work as their ``ckpt`` (these arms have no
statistics-only replay; the Cake arms replay only the statistics there); ``bwd_acc`` is ``bwd`` adding into the
existing parameter gradients only, like the Cake ``accumulate=True`` (``dfn``, ``dscale`` and ``dbase`` added into,
``dX`` and ``dB`` written): autograd accumulates into the existing ``.grad`` of the parameters (``fn``, ``scale``,
``base``; Liger's ``phi``, ``base`` and three scales), and the ``.grad`` of ``X`` and ``B`` is cleared before each
call, so that ``dX`` and ``dB`` are written fresh.

The comparisons (:data:`COMPARISONS`) are fixed: each one names a single non-Cake arm, the same arm on every
architecture, and never the fastest arm of a cell.  The forward has two, ``fwd-A`` (``tk_cublas``, every
architecture) and ``fwd-B`` (``tk_deepgemm``, where DeepGEMM runs; elsewhere the cell says why it is unsupported); the
backward and the checkpointed step (``ckpt`` and ``ckpt_stats``) compare with ``compile_max_autotune`` and, in a second
column, with ``liger``.

The pre handoff (operations ``hfwd``, ``hbwd``, ``hckpt``; hidden 5120, ``rms_eps`` :data:`HANDOFF_RMS_EPS` = 1e-20,
``hc_eps`` 1e-6) is one sublayer of DeepSeek-V4.1-Flash: the coefficients of ``X`` as above, the collapse with the
incoming ``pre_in`` (``U = BF16(sum_i pre_in_i X_i)``) and ``pre`` handed on as ``pre_next``; its backward from
``dY``, ``dU`` and ``dpre_next`` (the gradient of ``pre_next``) adds ``dpre_in = sum_d dU_d X_id``
(:func:`reference_handoff`).  The standalone collapse (``cfwd``, ``cbwd``) is ``U = BF16(sum_i pre_in_i X_i)`` and its
VJP ``(dX, dpre_in)`` (:func:`reference_collapse`).  :func:`make_inputs` draws ``pre_in`` (``sigmoid(N(0, 1)) +
1e-6``) and ``dpre_next`` (``sqrt(H) N(0, 1)``) after the standard inputs, which keep their values.  The autograd arms
run the handoff as the same Python function with ``pre_in`` (``eager_fp32``, ``compile_*``) and Liger as its
coefficient function, its pre-mix applied to ``pre_in`` and its post / residual mix (the coefficient function's
``h_pre`` handed on); the TileKernels arms have no handoff.  Each handoff operation compares with
``compile_max_autotune`` and with ``liger``.

Determinism class.  Each arm declares what is known about its run-to-run bits (:attr:`Arm.determinism`), and
:func:`determinism_runs` measures it: the same call on the same inputs several times, every output compared bit for
bit with the first run's.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import os
import statistics
import types
from pathlib import Path
from typing import Callable, Optional

import torch

STREAMS = 4
MAPPING_ROWS = 24
RMS_EPS = 1e-5
HC_EPS = 1e-6
SINKHORN_ITERS = 20
POST_MULT = 2.0
#: Rows per block of the float64 / BF16 references (the parameter gradients are summed over the blocks).
BLOCK_TOKENS = 8192
#: The judge's draws and the token count of its case (each hidden size of the run).
JUDGE_DRAWS = 32
JUDGE_TOKENS = 1024
# DeepGEMM 2.8.2 dispatches tf32_hc_prenorm_gemm to SM90 / SM100 kernels by major version; on sm_107a the SM100 kernel
# is built without cute's TMA support and asserts on the device.
DEEPGEMM_CAPABILITIES = ((9, 0), (10, 0), (10, 3))
FORWARD_OUTPUTS = ("U", "Y")
BACKWARD_OUTPUTS = ("dX", "dfn", "dscale", "dbase", "dB")
OUTPUTS = FORWARD_OUTPUTS + BACKWARD_OUTPUTS
PER_TOKEN_OUTPUTS = ("U", "Y", "dX", "dB")
PARAMETER_OUTPUTS = ("dfn", "dscale", "dbase")
STANDARD_OPS = ("fwd", "bwd", "ckpt", "ckpt_stats", "bwd_acc")
#: The pre-handoff operations (one sublayer) and the standalone collapse (hidden :data:`HANDOFF_HIDDEN`).
HANDOFF_OPS = ("hfwd", "hbwd", "hckpt")
COLLAPSE_OPS = ("cfwd", "cbwd")
OPS = STANDARD_OPS + HANDOFF_OPS + COLLAPSE_OPS
HANDOFF_HIDDEN = 5120
HANDOFF_RMS_EPS = 1e-20
HANDOFF_FORWARD_OUTPUTS = ("U", "Y", "pre_next")
HANDOFF_BACKWARD_OUTPUTS = ("dX", "dpre_in", "dfn", "dscale", "dbase", "dB")
HANDOFF_OUTPUTS = HANDOFF_FORWARD_OUTPUTS + HANDOFF_BACKWARD_OUTPUTS
COLLAPSE_OUTPUTS = ("U", "dX", "dpre_in")
#: The outputs of each kind of check: ``(per-token outputs, parameter outputs)``.
KIND_OUTPUTS = {
    "standard": (PER_TOKEN_OUTPUTS, PARAMETER_OUTPUTS),
    "handoff": (("U", "Y", "pre_next", "dX", "dpre_in", "dB"), PARAMETER_OUTPUTS),
    "collapse": (COLLAPSE_OUTPUTS, ()),
}


def op_kind(op: str) -> str:
    """``"standard"``, ``"handoff"`` or ``"collapse"``: the function an operation times."""
    return "handoff" if op in HANDOFF_OPS else "collapse" if op in COLLAPSE_OPS else "standard"

BF16, F32, F64 = torch.bfloat16, torch.float32, torch.float64

#: The fixed comparisons: ``(name, operations, non-Cake arm)``.  The arm is the same on every architecture; where it
#: cannot run, the cell records why (``unsupported``) instead of another arm.
COMPARISONS = (
    ("fwd-A", ("fwd",), "tk_cublas"),
    ("fwd-B", ("fwd",), "tk_deepgemm"),
    ("bwd", ("bwd", "bwd_acc"), "compile_max_autotune"),
    ("bwd-liger", ("bwd", "bwd_acc"), "liger"),
    ("ckpt", ("ckpt", "ckpt_stats"), "compile_max_autotune"),
    ("ckpt-liger", ("ckpt", "ckpt_stats"), "liger"),
    ("hfwd", ("hfwd",), "compile_max_autotune"),
    ("hfwd-liger", ("hfwd",), "liger"),
    ("hbwd", ("hbwd",), "compile_max_autotune"),
    ("hbwd-liger", ("hbwd",), "liger"),
    ("hckpt", ("hckpt",), "compile_max_autotune"),
    ("hckpt-liger", ("hckpt",), "liger"),
    ("cfwd", ("cfwd",), "compile_max_autotune"),
    ("cfwd-liger", ("cfwd",), "liger"),
    ("cbwd", ("cbwd",), "compile_max_autotune"),
    ("cbwd-liger", ("cbwd",), "liger"),
)


def comparisons_for(op: str) -> list:
    """The ``(name, arm)`` of every fixed comparison of operation ``op``."""
    return [(name, arm) for name, ops, arm in COMPARISONS if op in ops]


def cublas_version() -> Optional[str]:
    """``major.minor.patch (path)`` of the cuBLAS library loaded in this process (``cublasGetProperty``), or None
    when none is loaded yet."""
    import ctypes

    path = None
    try:
        with open("/proc/self/maps") as maps:
            for line in maps:
                name = line.split()[-1] if line.split() else ""
                if "/libcublas.so" in name:
                    path = name
                    break
    except OSError:
        return None
    if path is None:
        return None
    try:
        lib = ctypes.CDLL(path)
        parts = []
        for prop in (0, 1, 2):  # MAJOR_VERSION, MINOR_VERSION, PATCH_LEVEL
            value = ctypes.c_int()
            if lib.cublasGetProperty(prop, ctypes.byref(value)) != 0:
                return None
            parts.append(str(value.value))
        return ".".join(parts) + f" ({path})"
    except (OSError, AttributeError):
        return None


def set_ieee_fp32() -> None:
    """TF32 off for every FP32 matmul and convolution (IEEE FP32 FMA)."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def dist_version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


# ----------------------------------------------------------------------------------------------------------------
# Inputs and the reference arithmetic


def make_inputs(T: int, hidden: int, *, seed: int, device: torch.device, iters: int = SINKHORN_ITERS) -> dict:
    """``X, B, dY, dU ~ N(0, 1)`` BF16, ``fn ~ 1e-2 N(0, 1)``, ``scale ~ 1 + 0.1 N``, ``base ~ 0.1 N`` (FP32), then the
    pre-handoff inputs ``pre_in = sigmoid(N(0, 1)) + 1e-6`` and ``dpre_next = sqrt(H) N(0, 1)`` (FP32 ``[T, 4]``), from
    a device generator seeded with ``seed``; ``iters`` is the Sinkhorn count of every arm on these inputs."""
    g = torch.Generator(device=device).manual_seed(seed)

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=g, device=device, dtype=F32)

    X = randn(T, STREAMS, hidden).to(BF16)
    fn = (randn(MAPPING_ROWS, STREAMS * hidden) * 1e-2).contiguous()
    scale = 1.0 + 0.1 * randn(3)
    base = 0.1 * randn(MAPPING_ROWS)
    B = randn(T, hidden).to(BF16)
    dY = randn(T, STREAMS, hidden).to(BF16)
    dU = randn(T, hidden).to(BF16)
    pre_in = (torch.sigmoid(randn(T, STREAMS)) + 1e-6).contiguous()
    dpre_next = (randn(T, STREAMS) * hidden**0.5).contiguous()
    return {"X": X, "fn": fn, "scale": scale, "base": base, "B": B, "dY": dY, "dU": dU, "T": T, "hidden": hidden,
            "iters": int(iters), "pre_in": pre_in, "dpre_next": dpre_next}


def sinkhorn(logits: torch.Tensor, hc_eps: float = HC_EPS, iters: int = SINKHORN_ITERS) -> torch.Tensor:
    """Row softmax plus ``hc_eps``, one column normalisation, then ``iters - 1`` (row, column) normalisations."""
    C = torch.softmax(logits, dim=-1) + hc_eps
    C = C / (C.sum(dim=1, keepdim=True) + hc_eps)
    for _ in range(iters - 1):
        C = C / (C.sum(dim=2, keepdim=True) + hc_eps)
        C = C / (C.sum(dim=1, keepdim=True) + hc_eps)
    return C


def mhc_forward(X, fn, scale, base, B, *, dtype: torch.dtype, iters: int = SINKHORN_ITERS):
    """``(U, Y)`` of the function in ``dtype`` arithmetic with ``iters`` Sinkhorn iterations (an autograd graph when
    the inputs require grad)."""
    T = X.shape[0]
    Xd = X.to(dtype)
    u = Xd.reshape(T, -1)
    r = torch.rsqrt(u.square().mean(dim=-1, keepdim=True) + RMS_EPS)
    z = (u * r) @ fn.to(dtype).T
    s = scale.to(dtype)
    b = base.to(dtype)
    pre = torch.sigmoid(z[:, 0:4] * s[0] + b[0:4]) + HC_EPS
    post = POST_MULT * torch.sigmoid(z[:, 4:8] * s[1] + b[4:8])
    C = sinkhorn((z[:, 8:] * s[2] + b[8:]).reshape(T, STREAMS, STREAMS), iters=iters)
    U = (pre.unsqueeze(-1) * Xd).sum(dim=1)
    Y = post.unsqueeze(-1) * B.to(dtype).unsqueeze(1) + torch.einsum("tij,tid->tjd", C, Xd)
    return U, Y


def mhc_handoff_forward(X, pre_in, fn, scale, base, B, *, dtype: torch.dtype, iters: int = SINKHORN_ITERS,
                        rms_eps: float = HANDOFF_RMS_EPS):
    """``(U, Y, pre_next)`` of one handoff sublayer in ``dtype``: the coefficients of ``X``, ``U`` collapsed with
    ``pre_in``, this sublayer's ``pre`` as ``pre_next`` (an autograd graph when the inputs require grad)."""
    T = X.shape[0]
    Xd = X.to(dtype)
    u = Xd.reshape(T, -1)
    r = torch.rsqrt(u.square().mean(dim=-1, keepdim=True) + rms_eps)
    z = (u * r) @ fn.to(dtype).T
    s = scale.to(dtype)
    b = base.to(dtype)
    pre = torch.sigmoid(z[:, 0:4] * s[0] + b[0:4]) + HC_EPS
    post = POST_MULT * torch.sigmoid(z[:, 4:8] * s[1] + b[4:8])
    C = sinkhorn((z[:, 8:] * s[2] + b[8:]).reshape(T, STREAMS, STREAMS), iters=iters)
    U = (pre_in.to(dtype).unsqueeze(-1) * Xd).sum(dim=1)
    Y = post.unsqueeze(-1) * B.to(dtype).unsqueeze(1) + torch.einsum("tij,tid->tjd", C, Xd)
    return U, Y, pre


def collapse_forward(X, pre_in, *, dtype: torch.dtype):
    """``U = sum_i pre_in_i X_i`` in ``dtype``."""
    return (pre_in.to(dtype).unsqueeze(-1) * X.to(dtype)).sum(dim=1)


def _reference_block_handoff(i: dict, lo: int, hi: int, dtype: torch.dtype) -> dict:
    with torch.enable_grad():
        leaves = [
            t.detach().to(dtype).requires_grad_(True)
            for t in (i["X"][lo:hi], i["fn"], i["scale"], i["base"], i["B"][lo:hi], i["pre_in"][lo:hi])
        ]
        X, fn, scale, base, B, pre_in = leaves
        U, Y, pre_next = mhc_handoff_forward(X, pre_in, fn, scale, base, B, dtype=dtype,
                                             iters=i.get("iters", SINKHORN_ITERS))
        grads = torch.autograd.grad(
            [U, Y, pre_next], leaves,
            grad_outputs=[i["dU"][lo:hi].to(dtype), i["dY"][lo:hi].to(dtype), i["dpre_next"][lo:hi].to(dtype)],
        )
    out = {"U": U.detach(), "Y": Y.detach(), "pre_next": pre_next.detach()}
    out.update(zip(("dX", "dfn", "dscale", "dbase", "dB", "dpre_in"), grads))
    return out


def _reference_block_collapse(i: dict, lo: int, hi: int, dtype: torch.dtype) -> dict:
    x = i["X"][lo:hi].to(dtype)
    p = i["pre_in"][lo:hi].to(dtype)
    u = i["dU"][lo:hi].to(dtype)
    return {"U": collapse_forward(x, p, dtype=dtype), "dX": p.unsqueeze(-1) * u.unsqueeze(1),
            "dpre_in": (u.unsqueeze(1) * x).sum(dim=-1)}


def _reference_block(i: dict, lo: int, hi: int, dtype: torch.dtype) -> dict:
    """Every output of rows ``lo:hi`` in ``dtype`` (leaves in ``dtype``; the parameter gradients of these rows)."""
    with torch.enable_grad():
        leaves = [
            t.detach().to(dtype).requires_grad_(True)
            for t in (i["X"][lo:hi], i["fn"], i["scale"], i["base"], i["B"][lo:hi])
        ]
        U, Y = mhc_forward(*leaves, dtype=dtype, iters=i.get("iters", SINKHORN_ITERS))
        grads = torch.autograd.grad(
            [U, Y], leaves, grad_outputs=[i["dU"][lo:hi].to(dtype), i["dY"][lo:hi].to(dtype)]
        )
    out = {"U": U.detach(), "Y": Y.detach()}
    out.update(zip(BACKWARD_OUTPUTS, grads))
    return out


_BLOCKS = {"standard": _reference_block, "handoff": _reference_block_handoff, "collapse": _reference_block_collapse}


def reference_blocks(i: dict, dtype: torch.dtype, block_tokens: int = BLOCK_TOKENS, kind: str = "standard"):
    """``(lo, hi, outputs of rows lo:hi)`` in ``dtype``, in row blocks of ``block_tokens`` (``kind``: the function,
    :data:`KIND_OUTPUTS`)."""
    T = i["T"]
    block = _BLOCKS[kind]
    for lo in range(0, T, block_tokens):
        hi = min(T, lo + block_tokens)
        yield lo, hi, block(i, lo, hi, dtype)


def reference(i: dict, dtype: torch.dtype, block_tokens: int = BLOCK_TOKENS, kind: str = "standard") -> dict:
    """Every output in ``dtype``: per-token outputs concatenated, parameter gradients summed in block order."""
    per_token, parameters = KIND_OUTPUTS[kind]
    blocks = [b for _, _, b in reference_blocks(i, dtype, block_tokens, kind)]
    out = {k: torch.cat([b[k] for b in blocks]) for k in per_token}
    for k in parameters:
        acc = blocks[0][k].clone()
        for b in blocks[1:]:
            acc += b[k]
        out[k] = acc
    return out


def load_naive_judge():
    """The family's accuracy judge, ``tests/cake_naive.py`` next to this directory (loaded by path)."""
    path = Path(__file__).resolve().parents[1] / "tests" / "cake_naive.py"
    spec = importlib.util.spec_from_file_location("cake_mhc_benchmark_naive_judge", path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_outputs(i: dict, arm_outputs: dict, judge, block_tokens: int = BLOCK_TOKENS, kind: str = "standard") -> dict:
    """One draw: each arm's outputs (``{arm: {output: tensor}}``) against the float64 oracle, beside the naive BF16
    tier on the same inputs.  Per arm and output: the contract message (None when held), the L2 and max errors of the
    output and of the tier, and their ratios.  The oracle and the tier run in row blocks; nothing of the size of an
    output is held in float64."""
    sums = {
        arm: {name: {"l2": [0.0, 0.0], "max": [0.0, 0.0], "contract": None} for name in outs}
        for arm, outs in arm_outputs.items()
    }
    oracle_params: dict = {}
    naive_params: dict = {}

    def add(arm: str, name: str, actual, expected, naive) -> None:
        rec = sums[arm][name]
        if rec["contract"] is None:
            rec["contract"] = judge.contract(name, actual, expected)
        if actual.shape != expected.shape:
            return
        for norm in judge.NORMS:
            values, _ = judge.norm_errors(expected, (actual, naive.to(actual.dtype)), norm)
            for k in (0, 1):
                if norm == "l2":
                    rec[norm][k] += values[k] ** 2
                else:
                    rec[norm][k] = max(rec[norm][k], values[k])

    per_token, parameters = KIND_OUTPUTS[kind]
    for lo, hi, o in reference_blocks(i, F64, block_tokens, kind):
        n = _BLOCKS[kind](i, lo, hi, BF16)
        for arm, outs in arm_outputs.items():
            for name in per_token:
                if name in outs:
                    add(arm, name, outs[name][lo:hi], o[name], n[name])
        for name in parameters:
            if name in oracle_params:
                oracle_params[name] += o[name]
                naive_params[name] += n[name]
            else:
                oracle_params[name] = o[name].clone()
                naive_params[name] = n[name].clone()
        del o, n
    for arm, outs in arm_outputs.items():
        for name in parameters:
            if name in outs:
                add(arm, name, outs[name], oracle_params[name], naive_params[name])
    report: dict = {}
    for arm, by_name in sums.items():
        report[arm] = {}
        for name, rec in by_name.items():
            entry = {"contract": rec["contract"]}
            for norm in judge.NORMS:
                ours, tier = rec[norm]
                if norm == "l2":
                    ours, tier = ours**0.5, tier**0.5
                entry[norm] = {"ours": ours, "naive_bf16": tier, "ratio": (ours / tier) if tier > 0 else None}
            entry["above_naive_on_this_draw"] = any(
                entry[norm]["ratio"] is not None and entry[norm]["ratio"] > judge.NEAR_TIE for norm in judge.NORMS
            )
            report[arm][name] = entry
    return report


def judge_arms(
    arms: dict, hidden: int, *, device: torch.device, judge, draws: int = JUDGE_DRAWS,
    tokens: int = JUDGE_TOKENS, seed: int = 0, iters: int = SINKHORN_ITERS, kind: str = "standard",
) -> dict:
    """Hold each arm (``{name: arm}``, each with ``run_once``) to the naive-tier judge over ``draws`` draws of the case
    ``(tokens, hidden)``: per arm the verdict of every output, the failures (empty: the arm meets the contract) and the
    contract messages."""
    judges = {name: judge.TierJudge(draws) for name in arms}
    contract_failures: dict = {name: [] for name in arms}
    errors: dict = {}
    for j in range(draws):
        s = judge.draw_seed(seed, j)
        judge.seed_all(s)
        i = make_inputs(tokens, hidden, seed=s, device=device, iters=iters)
        oracle = reference(i, F64, kind=kind)
        naive = reference(i, BF16, kind=kind)
        for name, arm in arms.items():
            if name in errors:
                continue
            try:
                outs = arm.run_kind(i, kind)
            except Exception as exc:  # noqa: BLE001 - the arm's failure is its verdict
                errors[name] = f"{type(exc).__name__}: {exc}"[:400]
                continue
            for out_name, actual in outs.items():
                message = judge.contract(out_name, actual, oracle[out_name])
                if message:
                    contract_failures[name].append(f"draw {j}: {message}")
                    if actual.shape != oracle[out_name].shape:
                        continue
                judges[name].add(out_name, j, actual, oracle[out_name], {"bf16": naive[out_name]})
        del oracle, naive, i
    result = {}
    for name in arms:
        if name in errors:
            result[name] = {"error": errors[name], "meets_contract": False}
            continue
        verdicts = judges[name].verdicts()
        failures = contract_failures[name] + judges[name].failures()
        result[name] = {
            "meets_contract": not failures,
            "failures": failures[:20],
            "verdicts": verdicts,
            "tier_lines": judges[name].lines(f"judge[{name}] H={hidden} T={tokens}"),
        }
    return result


# ----------------------------------------------------------------------------------------------------------------
# Determinism


def _bits(t: torch.Tensor) -> torch.Tensor:
    """``t``'s storage as integers of the same width (a bit-for-bit comparison: NaN payloads and signed zeros
    included)."""
    width = {1: torch.int8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[t.element_size()]
    return t.detach().contiguous().view(width)


def determinism_runs(arm: "Arm", i: dict, runs: int, kind: str = "standard") -> dict:
    """Run ``arm.run_kind(i, kind)`` ``runs`` times on the same inputs and compare every output bit for bit with the
    first run's: ``{"runs", "bitwise", "differs": {output: [runs that differ]}}``."""
    first = {k: v.detach().clone() for k, v in arm.run_kind(i, kind).items()}
    differs: dict = {}
    for r in range(1, runs):
        outs = arm.run_kind(i, kind)
        for k, v in outs.items():
            if not torch.equal(_bits(v), _bits(first[k])):
                differs.setdefault(k, []).append(r)
        del outs
    return {"runs": runs, "bitwise": not differs, "differs": differs}


# ----------------------------------------------------------------------------------------------------------------
# Arms


class Arm:
    """One implementation of the function.  ``make(i, op)`` returns the timed callable of an operation on the
    inputs ``i`` (its buffers and state allocated once, outside the timing); ``run_once(i)`` returns every output it
    computes on ``i``.

    ``composite``: None for a library's (or the PyTorch reference's) own code path, else what this harness composed.
    ``determinism``: what the implementation declares about its run-to-run bits (the benchmarks also measure it)."""

    name = "base"
    library = ""
    ops: tuple = OPS
    cake = False
    deterministic: Optional[bool] = None
    composite: Optional[str] = None
    determinism = "not declared (measured by the benchmarks)"

    def available(self, device: torch.device, hidden: int) -> tuple[bool, str]:
        return True, ""

    def versions(self) -> dict:
        return {}

    def label(self) -> str:
        """The arm's name in reports: composites carry the suffix ``(composite)``."""
        return f"{self.name} (composite)" if self.composite else self.name

    def describe(self) -> dict:
        return {"library": self.library, "cake": self.cake, "deterministic": self.deterministic,
                "determinism_declared": self.determinism, "composite": self.composite, "label": self.label(),
                "ops": list(self.ops)}

    def prepare_shape(self, i: dict, timer=None) -> dict:
        """Per-shape setup before the check and the timing (e.g. compile trials); returns what to record."""
        return {}

    def run_once(self, i: dict) -> dict:
        raise NotImplementedError

    def run_handoff(self, i: dict) -> dict:
        """Every output of one handoff sublayer (:data:`HANDOFF_OUTPUTS`) on ``i``."""
        raise NotImplementedError

    def run_collapse(self, i: dict) -> dict:
        """The standalone collapse and its VJP (:data:`COLLAPSE_OUTPUTS`) on ``i``."""
        raise NotImplementedError

    def run_kind(self, i: dict, kind: str) -> dict:
        return {"standard": self.run_once, "handoff": self.run_handoff, "collapse": self.run_collapse}[kind](i)

    def supports(self, op: str, device: torch.device, hidden: int) -> tuple[bool, str]:
        """Whether the arm runs ``op`` at ``hidden`` on ``device`` (beyond :meth:`available`), and why not."""
        if op not in self.ops:
            return False, f"the arm has no {op}"
        if op_kind(op) != "standard" and hidden != HANDOFF_HIDDEN:
            return False, f"the pre handoff is measured at hidden {HANDOFF_HIDDEN}"
        return True, ""

    def make(self, i: dict, op: str) -> Callable[[], None]:
        raise NotImplementedError


class AutogradArm(Arm):
    """An arm whose forward records an autograd graph and whose backward is ``torch.autograd.backward``."""

    def leaves(self, i: dict) -> list:
        """The tensors the gradients are taken against, in the arm's parameter layout (aliases of the inputs)."""
        return [
            i["X"].detach().requires_grad_(True),
            i["fn"].detach().requires_grad_(True),
            i["scale"].detach().requires_grad_(True),
            i["base"].detach().requires_grad_(True),
            i["B"].detach().requires_grad_(True),
        ]

    def activation_leaves(self, leaves: list) -> list:
        """The leaves that are activations, ``X`` and ``B`` (first and last in every arm's layout); the others are the
        parameters.  ``bwd_acc`` clears the activations' ``.grad`` before each call, so their gradients are written
        fresh, as the Cake arm writes ``dX`` and ``dB``; only the parameter gradients accumulate."""
        return [leaves[0], leaves[-1]]

    def forward(self, leaves: list, iters: int):
        raise NotImplementedError

    def gradients(self, leaves: list) -> dict:
        X, fn, scale, base, B = leaves
        return {"dX": X.grad, "dfn": fn.grad, "dscale": scale.grad, "dbase": base.grad, "dB": B.grad}

    def run_once(self, i: dict) -> dict:
        leaves = self.leaves(i)
        U, Y = self.forward(leaves, i["iters"])
        torch.autograd.backward([U, Y], [i["dU"], i["dY"]])
        out = {"U": U.detach(), "Y": Y.detach()}
        out.update(self.gradients(leaves))
        return out

    def forward_handoff(self, leaves: list, pre_in: torch.Tensor, iters: int):
        """``(U, Y, pre_next)`` of one handoff sublayer (an autograd graph over ``leaves`` and ``pre_in``)."""
        raise NotImplementedError

    def forward_collapse(self, X: torch.Tensor, pre_in: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def run_handoff(self, i: dict) -> dict:
        leaves = self.leaves(i)
        pre_in = i["pre_in"].detach().requires_grad_(True)
        U, Y, pre_next = self.forward_handoff(leaves, pre_in, i["iters"])
        torch.autograd.backward([U, Y, pre_next], [i["dU"], i["dY"], i["dpre_next"]])
        out = {"U": U.detach(), "Y": Y.detach(), "pre_next": pre_next.detach(), "dpre_in": pre_in.grad}
        out.update(self.gradients(leaves))
        for t in (*leaves, pre_in):
            t.grad = None
        return out

    def run_collapse(self, i: dict) -> dict:
        X = i["X"].detach().requires_grad_(True)
        pre_in = i["pre_in"].detach().requires_grad_(True)
        U = self.forward_collapse(X, pre_in)
        U.backward(i["dU"])
        out = {"U": U.detach(), "dX": X.grad, "dpre_in": pre_in.grad}
        X.grad = pre_in.grad = None
        return out

    def _make_handoff(self, i: dict, op: str) -> Callable[[], None]:
        """The handoff (``hfwd``, ``hbwd``, ``hckpt``) and collapse (``cfwd``, ``cbwd``) operations."""
        leaves = self.leaves(i)
        pre_in = i["pre_in"].detach().requires_grad_(True)
        X = leaves[0]
        dU, dY, dpn, iters = i["dU"], i["dY"], i["dpre_next"], i["iters"]
        everything = (*leaves, pre_in)

        def clear() -> None:
            for t in everything:
                t.grad = None

        if op == "hfwd":
            return lambda: self.forward_handoff(leaves, pre_in, iters)
        if op == "hbwd":
            outs = self.forward_handoff(leaves, pre_in, iters)

            def call() -> None:
                clear()
                torch.autograd.backward(list(outs), [dU, dY, dpn], retain_graph=True)

            return call
        if op == "hckpt":

            def call() -> None:
                clear()
                with torch.no_grad():
                    self.forward_handoff(leaves, pre_in, iters)
                outs = self.forward_handoff(leaves, pre_in, iters)
                torch.autograd.backward(list(outs), [dU, dY, dpn])

            return call
        if op == "cfwd":
            return lambda: self.forward_collapse(X, pre_in)
        if op == "cbwd":
            U = self.forward_collapse(X, pre_in)

            def call() -> None:
                clear()
                U.backward(dU, retain_graph=True)

            return call
        raise ValueError(op)

    def make(self, i: dict, op: str) -> Callable[[], None]:
        if op_kind(op) != "standard":
            return self._make_handoff(i, op)
        leaves = self.leaves(i)
        dU, dY, iters = i["dU"], i["dY"], i["iters"]
        forward = self.forward

        def clear() -> None:
            for t in leaves:
                t.grad = None

        if op == "fwd":

            def call() -> None:
                forward(leaves, iters)

        elif op in ("bwd", "bwd_acc"):
            U, Y = forward(leaves, iters)
            if op == "bwd":

                def call() -> None:
                    clear()
                    torch.autograd.backward([U, Y], [dU, dY], retain_graph=True)

            else:
                # The parameter gradients accumulate into the existing ``.grad`` tensors (the first, warm-up call
                # creates them); the activation gradients ``dX`` and ``dB`` are cleared first and written fresh, as in
                # the Cake arm's ``accumulate=True``.
                activations = self.activation_leaves(leaves)

                def call() -> None:
                    for t in activations:
                        t.grad = None
                    torch.autograd.backward([U, Y], [dU, dY], retain_graph=True)

        elif op in ("ckpt", "ckpt_stats"):
            # ``ckpt_stats`` is the same work: an autograd arm has no statistics-only replay to recompute.

            def call() -> None:
                clear()
                with torch.no_grad():
                    forward(leaves, iters)
                U, Y = forward(leaves, iters)
                torch.autograd.backward([U, Y], [dU, dY])

        else:
            raise ValueError(op)
        return call


def _fp32_mhc(X, fn, scale, base, B, iters: int = SINKHORN_ITERS):
    U, Y = mhc_forward(X, fn, scale, base, B, dtype=F32, iters=iters)
    return U.to(BF16), Y.to(BF16)


def _fp32_mhc_handoff(X, fn, scale, base, B, pre_in, iters: int = SINKHORN_ITERS):
    U, Y, pre_next = mhc_handoff_forward(X, pre_in, fn, scale, base, B, dtype=F32, iters=iters)
    return U.to(BF16), Y.to(BF16), pre_next


def _fp32_collapse(X, pre_in):
    return collapse_forward(X, pre_in, dtype=F32).to(BF16)


class EagerFP32(AutogradArm):
    name = "eager_fp32"
    library = "PyTorch eager FP32 (TF32 off)"
    deterministic = None
    determinism = "not declared by PyTorch for this graph (measured by the benchmarks)"

    def versions(self) -> dict:
        return {"torch": torch.__version__, "torch_cuda": torch.version.cuda, "cublas": cublas_version()}

    def forward(self, leaves, iters):
        return _fp32_mhc(*leaves, iters)

    def forward_handoff(self, leaves, pre_in, iters):
        return _fp32_mhc_handoff(*leaves, pre_in, iters)

    def forward_collapse(self, X, pre_in):
        return _fp32_collapse(X, pre_in)


def _fresh_function(fn: Callable, tag: str) -> Callable:
    """A copy of ``fn`` with its own code object, so that ``torch.compile`` compiles it anew."""
    code = fn.__code__.replace(co_name=f"{fn.__name__}_{tag}")
    return types.FunctionType(code, fn.__globals__, f"{fn.__name__}_{tag}", fn.__defaults__, fn.__closure__)


class Compiled(AutogradArm):
    """``torch.compile`` of :func:`_fp32_mhc`; ``trials`` fresh compiles per shape, the fastest one timed."""

    determinism = "not declared: the generated Triton kernels may split reductions (measured by the benchmarks)"

    def __init__(self, name: str, mode: Optional[str], trials: int = 1) -> None:
        self.name = name
        self.mode = mode
        self.trials = max(1, int(trials))
        self.library = f"torch.compile (mode={mode or 'default'}, fullgraph, static shapes, TF32 off, no buffer donation)"
        self._fn: Optional[Callable] = None
        self._fn_handoff: Optional[Callable] = None
        self._fn_collapse: Optional[Callable] = None
        self._count = 0

    def available(self, device, hidden):
        try:
            import triton

            return True, f"triton {triton.__version__}"
        except Exception as exc:  # noqa: BLE001
            return False, f"{type(exc).__name__}: {exc}"

    def versions(self) -> dict:
        import triton

        return {"torch": torch.__version__, "torch_cuda": torch.version.cuda, "triton": triton.__version__,
                "mode": self.mode or "default", "trials": self.trials, "donated_buffer": False,
                "cublas": cublas_version()}

    def _compile(self, fn: Callable = _fp32_mhc) -> Callable:
        # The ``bwd`` operation runs the backward of one recorded forward repeatedly (``retain_graph=True``), which a
        # compiled backward that donates its saved buffers refuses: donation is turned off for the compiled arms.
        import torch._functorch.config as functorch_config

        functorch_config.donated_buffer = False
        self._count += 1
        kwargs = {"fullgraph": True, "dynamic": False}
        if self.mode:
            kwargs["mode"] = self.mode
        return torch.compile(_fresh_function(fn, f"{self.name}_{self._count}"), **kwargs)

    def forward(self, leaves, iters):
        if self._fn is None:
            self._fn = self._compile()
        return self._fn(*leaves, iters)

    def forward_handoff(self, leaves, pre_in, iters):
        if self._fn_handoff is None:
            self._fn_handoff = self._compile(_fp32_mhc_handoff)
        return self._fn_handoff(*leaves, pre_in, iters)

    def forward_collapse(self, X, pre_in):
        if self._fn_collapse is None:
            self._fn_collapse = self._compile(_fp32_collapse)
        return self._fn_collapse(X, pre_in)

    def prepare_shape(self, i: dict, timer=None) -> dict:
        """Compile for this shape; with several trials, time one group of training steps (forward + backward) of each
        fresh compile and keep the fastest.  The handoff and collapse functions compile on their first call (one
        trial)."""
        self._fn = self._fn_handoff = self._fn_collapse = None
        if self.trials == 1 or timer is None:
            self._fn = self._compile()
            return {"compile_trials": 1}
        import torch._inductor.config as inductor_config

        saved = inductor_config.force_disable_caches
        inductor_config.force_disable_caches = True
        trials = []
        try:
            for _ in range(self.trials):
                fn = self._compile()
                leaves = self.leaves(i)

                def step(fn=fn, leaves=leaves) -> None:
                    for t in leaves:
                        t.grad = None
                    U, Y = fn(*leaves, i["iters"])
                    torch.autograd.backward([U, Y], [i["dU"], i["dY"]])

                for _ in range(3):
                    step()
                torch.cuda.synchronize()
                samples = timer.times(step, timer.default_calls)
                trials.append((statistics.median(s["active_us"] for s in samples), fn))
                del leaves
        finally:
            inductor_config.force_disable_caches = saved
        best = min(range(len(trials)), key=lambda k: trials[k][0])
        self._fn = trials[best][1]
        return {"compile_trials": self.trials, "trial_step_active_us": [t for t, _ in trials], "chosen_trial": best}


class Liger(AutogradArm):
    name = "liger"
    library = "Liger Kernel mHC (liger_kernel.ops.mhc, allow_fp32=True)"
    deterministic = False
    determinism = "declared non-deterministic: its backward adds FP32 partials with atomics"

    def available(self, device, hidden):
        try:
            import liger_kernel.ops.mhc  # noqa: F401

            return True, f"liger-kernel {dist_version('liger-kernel')}, TRITON_F32_DEFAULT={os.environ.get('TRITON_F32_DEFAULT')}"
        except Exception as exc:  # noqa: BLE001
            return False, f"{type(exc).__name__}: {exc}"

    def versions(self) -> dict:
        import triton

        return {"liger-kernel": dist_version("liger-kernel"), "triton": triton.__version__, "torch": torch.__version__,
                "TRITON_F32_DEFAULT": os.environ.get("TRITON_F32_DEFAULT")}

    def leaves(self, i: dict) -> list:
        scale = i["scale"].detach()
        return [
            i["X"].detach().requires_grad_(True),
            i["fn"].detach().t().contiguous().requires_grad_(True),  # Liger's [4H, 24] layout
            i["base"].detach().requires_grad_(True),
            scale[0].clone().requires_grad_(True),
            scale[1].clone().requires_grad_(True),
            scale[2].clone().requires_grad_(True),
            i["B"].detach().requires_grad_(True),
        ]

    def gradients(self, leaves):
        X, phi, base, s0, s1, s2, B = leaves
        return {
            "dX": X.grad,
            "dfn": phi.grad.t().contiguous(),
            "dscale": torch.stack([s0.grad, s1.grad, s2.grad]),
            "dbase": base.grad,
            "dB": B.grad,
        }

    def forward(self, leaves, iters):
        from liger_kernel.ops.mhc import LigerMHCCoeffsFunction, LigerMHCPostResFunction, LigerMHCPreFunction

        X, phi, base, s0, s1, s2, B = leaves
        h_pre, h_post, h_res = LigerMHCCoeffsFunction.apply(
            X, phi, base, s0, s1, s2, True, iters, RMS_EPS, HC_EPS, HC_EPS, POST_MULT
        )
        U = LigerMHCPreFunction.apply(X, h_pre)
        Y = LigerMHCPostResFunction.apply(X, B, h_post, h_res.transpose(-1, -2))
        return U, Y

    def forward_handoff(self, leaves, pre_in, iters):
        """Liger's coefficient function at ``rms_eps`` 1e-20, its pre-mix applied to the incoming ``pre_in``, its post
        / residual mix; the coefficient function's ``h_pre`` is handed on as ``pre_next``."""
        from liger_kernel.ops.mhc import LigerMHCCoeffsFunction, LigerMHCPostResFunction, LigerMHCPreFunction

        X, phi, base, s0, s1, s2, B = leaves
        h_pre, h_post, h_res = LigerMHCCoeffsFunction.apply(
            X, phi, base, s0, s1, s2, True, iters, HANDOFF_RMS_EPS, HC_EPS, HC_EPS, POST_MULT
        )
        U = LigerMHCPreFunction.apply(X, pre_in.to(h_pre.dtype))
        Y = LigerMHCPostResFunction.apply(X, B, h_post, h_res.transpose(-1, -2))
        return U, Y, h_pre

    def forward_collapse(self, X, pre_in):
        from liger_kernel.ops.mhc import LigerMHCPreFunction

        return LigerMHCPreFunction.apply(X, pre_in)


class TileKernels(Arm):
    """TileKernels' fused pre kernel and post kernel after a prenorm GEMM (forward only): a composition of this
    harness."""

    ops = ("fwd",)
    deterministic = None
    determinism = "not declared by TileKernels or the GEMM library (measured by the benchmarks)"

    def __init__(self, gemm: str, splits: int = 16) -> None:
        if gemm not in ("cublas", "deepgemm"):
            raise ValueError(gemm)
        self.gemm = gemm
        self.splits = splits
        self.name = "tk_cublas" if gemm == "cublas" else "tk_deepgemm"
        self.library = (
            "TileKernels mhc_pre_big_fuse_fwd + mhc_post_fwd after "
            + ("cuBLAS FP32 GEMM (TF32 off)" if gemm == "cublas" else "DeepGEMM tf32_hc_prenorm_gemm hi/lo")
        )
        self.composite = (
            "harness composition: FP32 copy of X, cuBLAS FP32 GEMM float(X) @ fn.T and PyTorch squared row norms, then "
            "TileKernels mhc_pre_big_fuse_fwd and mhc_post_fwd"
            if gemm == "cublas" else
            "harness composition: fn split into its TF32 part and the remainder (PyTorch), DeepGEMM "
            "tf32_hc_prenorm_gemm on each part (16 K-splits each), then TileKernels mhc_pre_big_fuse_fwd over the 32 "
            "partials and mhc_post_fwd"
        )

    def available(self, device, hidden):
        try:
            import tile_kernels.mhc  # noqa: F401

            note = f"tile-kernels {dist_version('tile-kernels')}, tilelang {dist_version('tilelang')}"
            if self.gemm == "deepgemm":
                cap = tuple(torch.cuda.get_device_capability(device))
                if cap not in DEEPGEMM_CAPABILITIES:
                    # The device assertion would leave the CUDA context unusable for every other arm of the process.
                    dg = dist_version("deep_gemm") or dist_version("deep-gemm")
                    return False, (f"DeepGEMM has no sm_{cap[0]}{cap[1]}a support"
                                   + (f" (DeepGEMM {dg}: device assertion, TMA descriptor prefetch without "
                                      "CUTE_ARCH_TMA_SM90_ENABLED)" if dg else " (not installed on this stack)"))
                import deep_gemm

                note += f", deep-gemm {getattr(deep_gemm, '__version__', '?')}"
            return True, note
        except Exception as exc:  # noqa: BLE001
            return False, f"{type(exc).__name__}: {exc}"

    def versions(self) -> dict:
        out = {"tile-kernels": dist_version("tile-kernels"), "tilelang": dist_version("tilelang"),
               "torch": torch.__version__}
        if self.gemm == "deepgemm":
            try:
                import deep_gemm

                out["deep-gemm"] = getattr(deep_gemm, "__version__", None)
            except Exception as exc:  # noqa: BLE001
                out["deep-gemm"] = f"unavailable: {type(exc).__name__}"
            out["deep-gemm-dist"] = dist_version("deep_gemm") or dist_version("deep-gemm")
        else:
            out["cublas"] = cublas_version()
        return out

    def _buffers(self, i: dict) -> dict:
        T, H = i["T"], i["hidden"]
        dev = i["X"].device
        b = {
            "post": torch.empty(T, STREAMS, dtype=F32, device=dev),
            "comb": torch.empty(T, STREAMS * STREAMS, dtype=F32, device=dev),
            "U": torch.empty(T, H, dtype=BF16, device=dev),
            "Y": torch.empty(T, STREAMS, H, dtype=BF16, device=dev),
        }
        if self.gemm == "cublas":
            b["xf"] = torch.empty(T, STREAMS * H, dtype=F32, device=dev)
            b["mul"] = torch.empty(1, T, MAPPING_ROWS, dtype=F32, device=dev)
            b["norm"] = torch.empty(T, dtype=F32, device=dev)
            b["sq"] = torch.empty(1, T, dtype=F32, device=dev)
        else:
            s = self.splits
            b["fn_hi"] = torch.empty_like(i["fn"])
            b["fn_lo"] = torch.empty_like(i["fn"])
            b["mul"] = torch.empty(2 * s, T, MAPPING_ROWS, dtype=F32, device=dev)
            b["sq"] = torch.zeros(2 * s, T, dtype=F32, device=dev)  # the remainder half stays zero
            b["sq_lo"] = torch.empty(s, T, dtype=F32, device=dev)
        return b

    def make(self, i: dict, op: str = "fwd") -> Callable[[], None]:
        if op != "fwd":
            raise ValueError(f"{self.name} has no {op}")
        return self._make(i)[0]

    def _make(self, i: dict):
        from tile_kernels.mhc import mhc_post_fwd, mhc_pre_big_fuse_fwd

        T, H = i["T"], i["hidden"]
        X, fn, scale, base, B = i["X"], i["fn"], i["scale"], i["base"], i["B"]
        X2 = X.view(T, STREAMS * H)
        b = self._buffers(i)
        fused = dict(rms_eps=RMS_EPS, mhc_pre_eps=HC_EPS, mhc_sinkhorn_eps=HC_EPS, mhc_post_mult_value=POST_MULT,
                     sinkhorn_repeat=i["iters"])

        def tail(n_splits: int) -> None:
            mhc_pre_big_fuse_fwd(b["mul"], b["sq"], scale, base, X, b["post"], b["comb"], b["U"], **fused,
                                 n_splits=n_splits)
            mhc_post_fwd(B.view(1, T, H), X.view(1, T, STREAMS, H), b["post"].view(1, T, STREAMS, 1),
                         b["comb"].view(1, T, STREAMS, STREAMS), out=b["Y"].view(1, T, STREAMS, H))

        if self.gemm == "cublas":

            def call() -> None:
                b["xf"].copy_(X2)
                torch.mm(b["xf"], fn.t(), out=b["mul"].view(T, MAPPING_ROWS))
                torch.linalg.vector_norm(b["xf"], dim=-1, out=b["norm"])
                torch.square(b["norm"], out=b["sq"].view(T))
                tail(1)

        else:
            import deep_gemm

            s = self.splits
            bits = fn.view(torch.int32)
            hi_bits = b["fn_hi"].view(torch.int32)

            def call() -> None:
                torch.bitwise_and(bits + 0x1000, -8192, out=hi_bits)  # fn rounded to TF32 (exact in TF32)
                torch.sub(fn, b["fn_hi"], out=b["fn_lo"])
                deep_gemm.tf32_hc_prenorm_gemm(X2, b["fn_hi"], b["mul"][:s], b["sq"][:s], num_splits=s)
                deep_gemm.tf32_hc_prenorm_gemm(X2, b["fn_lo"], b["mul"][s:], b["sq_lo"], num_splits=s)
                tail(2 * s)

        return call, b

    def run_once(self, i: dict) -> dict:
        call, b = self._make(i)
        call()
        return {"U": b["U"], "Y": b["Y"]}


def non_cake_arms(names: list[str], *, compile_trials: int = 1, tk_splits: int = 16) -> list[Arm]:
    table = {
        "eager_fp32": lambda: EagerFP32(),
        "compile_default": lambda: Compiled("compile_default", None, compile_trials),
        "compile_max_autotune": lambda: Compiled("compile_max_autotune", "max-autotune-no-cudagraphs", compile_trials),
        "liger": lambda: Liger(),
        "tk_cublas": lambda: TileKernels("cublas", tk_splits),
        "tk_deepgemm": lambda: TileKernels("deepgemm", tk_splits),
    }
    unknown = [n for n in names if n not in table]
    if unknown:
        raise ValueError(f"unknown non-Cake arms {unknown}; known: {sorted(table)}")
    return [table[n]() for n in names]


NON_CAKE_ARMS = ("eager_fp32", "compile_default", "compile_max_autotune", "liger", "tk_cublas", "tk_deepgemm")
