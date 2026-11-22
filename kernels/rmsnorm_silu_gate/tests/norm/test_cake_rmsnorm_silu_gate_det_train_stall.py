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

Stall regression of the Cake SiLU-gated RMSNorm training forward and backward
beside a kernel resident on another stream (the concurrent-launch contract).

Every forward module but the tiny one (one per architecture, and one for
packed rows where routed) has the same grid rule.  Below its cut of rows per
SM of the architecture it launches one CTA per row block; from there the
fewest whole waves of ``ctas_per_sm`` CTAs per SM whose CTAs each walk at
most ``max_blocks_per_cta`` row blocks (its record's grid rule).  The cut is
``persistent_from_rows_per_sm`` on rows the kernel reads as packed and, for
the forward of rows of any layout, ``persistent_from_rows_per_sm_strided``
on other rows.  The backward is
one kernel of ``min(row blocks, ctas_per_sm * SMs)`` CTAs; past one block per
CTA its CTAs claim chunks of row blocks from a counter, so a CTA that becomes
resident late takes only the blocks left.  A backward with
``compute_grad_weight=False`` (stage ``backward_nodw``) runs the program
without the weight gradient: up to its cut of rows per SM, and on an
architecture without a claimed program at every row count, the strided
program (on the packed rows these tests use ``min(row blocks, 8 * SMs)``
128-thread CTAs over 8-row blocks, on other layouts ``min(row blocks, 4 *
SMs)`` 256-thread CTAs over 16-row blocks, each CTA walking its row blocks
with no claim and no ticket), past it the claimed program, and packed rows
from the long
modules' rows per SM the long program without the weight gradient.  A
backward that computes the weight gradient on up to 256 rows runs one CTA
(up to 16 rows) or one cluster of 16 CTAs (on a device of at least 16 SMs
that is not a MIG instance).  No kernel waits for another CTA or grid but
that cluster, whose CTAs wait for the weight-gradient words their peers
push to them: the GPU starts its 16 CTAs together, once one GPC has room
for all of them, so none waits for a CTA that is not running and no kernel
needs the whole device.  Beside a
long-resident one-thread kernel on another stream each call must keep its
speed: per token count, the median of seven event-timed calls with the side
kernel resident is at most ``MAX_RATIO`` times the median of seven calls
alone, or at most ``MAX_EXTRA_MS`` above it.  The alone and busy calls
alternate in pairs, so host launch gaps and clock drift hit both arms alike.
The bounds sit above the slowdown a resident kernel causes (the CTAs sharing
its SM run a little slower) and below the stall this test guards against, a
wait of milliseconds behind the resident kernel.  ``y``, ``dx``, ``dz`` and
``dw`` are bitwise equal across the alone and busy calls.

The forward and the backward beside a kernel resident on every SM with the
maximum-shared-memory carveout: at ``ALL_SM_TOKENS``
each starts beside it and costs at most ``MAX_ALL_SM_EXTRA_MS`` more than
alone, far below the side kernel's ``ALL_SM_SPIN_MS`` lifetime; so does the
backward that computes a BF16 or an FP32 weight gradient on
``SMALL_ROW_SHAPES`` (1 to 256 rows), which runs the program of one CTA or
of one cluster.  The forward
beside a side kernel that holds ``k`` of the ``N`` SMs (1 x 128,
16 x 640 and 32 x 640 side CTAs): busy / alone is at most
``N / (N - k)``, plus the share of one CTA's blocks in the call (a CTA whose
slot the side kernel holds walks all of its blocks once it starts), plus
``PROPORTIONAL_SLACK``, on packed rows and on rows whose ``z`` is the gate
columns of a wider projection, at the token counts that give each CTA slot
the most whole row blocks below the cut of such rows (one CTA a block,
"below"), the fewest whole waves of ``max_blocks_per_cta`` row blocks at or
above it ("first") and four times that ("last").  In these tests every
side CTA records its SM and its start time, a run counts only when the side
kernel held the intended SMs before the call started, and every output is
bitwise that of the first call alone.  The tiny forward of packed rows
(``fwd_tiny``, where routed: one CTA a block of its rule's ``rows_per_cta``
rows, no loop) at one row per SM, half its cut and its cut
(``FWD_TINY_ROWS_PER_SM``) keeps its speed beside the one-thread kernel,
starts beside the all-SM kernel with the max-shared carveout, and beside a
side kernel that holds ``k`` SMs (the item 1 side kernels and 16 CTAs of
1024 threads with the max-shared carveout) takes at most ``N / (N - k)``
times its time alone, plus the share of one CTA's blocks (one block over the
blocks a CTA slot runs, the slots counted at the most CTAs of its threads an
SM holds, and at most 1, since a call is never shorter than one block), plus
``PROPORTIONAL_SLACK``.  The thresholds are the non-deterministic suite's.
"""

from __future__ import annotations

import ctypes
import functools
import statistics

import pytest
import torch

from tests.cake.test_helpers.cake_package import cake_package

pkg = cake_package("cake_rmsnorm_silu_gate_det_train")
loader = cake_package("cake_rmsnorm_silu_gate_det_train", "cake_jit")
cake_rmsnorm_silu_gate_det_train_backward = (
    pkg.cake_rmsnorm_silu_gate_det_train_backward
)
cake_rmsnorm_silu_gate_det_train_backward_workspace = (
    pkg.cake_rmsnorm_silu_gate_det_train_backward_workspace
)
cake_rmsnorm_silu_gate_det_train_forward = pkg.cake_rmsnorm_silu_gate_det_train_forward
forward = cake_rmsnorm_silu_gate_det_train_forward
is_cake_rmsnorm_silu_gate_det_train_supported = (
    pkg.is_cake_rmsnorm_silu_gate_det_train_supported
)

D = 128
H = 48
EPS = 1e-6
# The issue's token counts and a small and a large one.
TOKENS = (256, 4096, 30193, 32737, 32388, 65536)
# Side kernel: one thread resident for about 5 ms; the main stream sleeps first
# so the side kernel is resident when the call starts.
SIDE_SLEEP_CYCLES = 10_000_000
PRE_SLEEP_CYCLES = 1_000_000
WARMUP = 3
RUNS = 7
# A row passes on either bound: the busy median over the alone median, or the
# busy median minus the alone median in ms.
MAX_RATIO = 1.5
MAX_EXTRA_MS = 0.020
SEED = 7075

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not is_cake_rmsnorm_silu_gate_det_train_supported(torch.device("cuda")),
    reason="Cake SiLU-gated RMSNorm training needs an SM100 / SM103 / SM107 GPU",
)


def _inputs(tokens: int, heads: int = H, layout: str = "packed"):
    """``(x, z, w, dy)`` of ``tokens`` tokens of ``heads`` heads, packed rows;
    with ``layout="strided"`` ``z`` the gate columns of a wider projection,
    whose rows are not packed (with more than one token)."""
    generator = torch.Generator(device="cuda").manual_seed(SEED + tokens)
    kwargs = dict(dtype=torch.float32, device="cuda", generator=generator)
    x = torch.randn(tokens, heads, D, **kwargs).to(torch.bfloat16)
    if layout == "packed":
        z = torch.randn(tokens, heads, D, **kwargs).to(torch.bfloat16)
    else:
        assert layout == "strided", layout
        wide = torch.randn(tokens, 2 * heads * D + 16, **kwargs).to(torch.bfloat16)
        z = wide[:, heads * D : 2 * heads * D].view(tokens, heads, D)
    dy = torch.randn(tokens, heads, D, **kwargs).to(torch.bfloat16)
    w = (1 + 0.5 * torch.randn(D, **kwargs)).to(torch.bfloat16)
    return x, z, w, dy


def _timed(call, *, side: torch.cuda.Stream | None):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    if side is not None:
        with torch.cuda.stream(side):
            torch.cuda._sleep(SIDE_SLEEP_CYCLES)
    torch.cuda._sleep(PRE_SLEEP_CYCLES)
    start.record()
    out = call()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end), tuple(t.clone() for t in out)


def _keeps_its_speed(alone_ms: float, busy_ms: float) -> bool:
    return busy_ms / alone_ms <= MAX_RATIO or busy_ms - alone_ms <= MAX_EXTRA_MS


@pytest.mark.parametrize("tokens", TOKENS)
@pytest.mark.parametrize("stage", ["forward", "backward", "backward_nodw"])
def test_keeps_its_speed_beside_a_resident_kernel(stage: str, tokens: int) -> None:
    x, z, w, dy = _inputs(tokens)
    if stage == "forward":
        y = torch.empty_like(x)

        def call():
            return (cake_rmsnorm_silu_gate_det_train_forward(x, z, w, EPS, out=y),)

    else:
        workspace = cake_rmsnorm_silu_gate_det_train_backward_workspace(
            tokens * H, x.device
        )
        compute = stage == "backward"

        def call():
            out = cake_rmsnorm_silu_gate_det_train_backward(
                dy, x, z, w, EPS, compute_grad_weight=compute, workspace=workspace
            )
            return out if compute else out[:2]

    side = torch.cuda.Stream()
    for _ in range(WARMUP):
        _timed(call, side=None)
    alone, busy = [], []
    for _ in range(RUNS):
        alone.append(_timed(call, side=None))
        busy.append(_timed(call, side=side))
    alone_ms = statistics.median(t for t, _ in alone)
    busy_ms = statistics.median(t for t, _ in busy)
    print(
        f"STALL stage={stage} tokens={tokens} heads={H} alone_ms={alone_ms:.4f} "
        f"busy_ms={busy_ms:.4f} ratio={busy_ms / alone_ms:.3f} extra_ms={busy_ms - alone_ms:.4f}"
    )
    reference = alone[0][1]
    for _t, out in alone[1:] + busy:
        for got, want in zip(out, reference, strict=True):
            assert torch.equal(got, want), (
                f"{stage} tokens={tokens}: results differ across calls"
            )
    assert _keeps_its_speed(alone_ms, busy_ms), (
        f"{stage} tokens={tokens}: {busy_ms:.4f} ms beside a resident kernel vs "
        f"{alone_ms:.4f} ms alone"
    )


# A side kernel whose CTAs each record their SM and start time
# and spin ``ALL_SM_SPIN_MS`` of wall time (``%globaltimer``).  A call that
# waits for it takes the rest of that lifetime; one that starts beside it costs
# at most ``MAX_ALL_SM_EXTRA_MS`` more than alone.
ALL_SM_SPIN_MS = 50.0
ALL_SM_THREADS = 128
ALL_SM_PRE_SLEEP_CYCLES = 2_000_000
ALL_SM_TOKENS = (256, 4096, 32737)
ALL_SM_RUNS = 5
ALL_SM_TRIES = 4
MAX_ALL_SM_EXTRA_MS = 1.0
# cudaSharedmemCarveoutMaxShared
CARVEOUT_MAX_SHARED = 100
# Item 1: (CTAs, threads) of side kernels that hold some SMs, the issue's
# 1 x 128, 16 x 640 and 32 x 640.
PROPORTIONAL_SIDES = ((1, 128), (16, 640), (32, 640))
PROPORTIONAL_RUNS = 7
# The noise margin of busy / alone over N / (N - k) plus one CTA's share of
# the blocks: the non-deterministic suite's, whose forward tests run forwards
# of the same design (whole waves of CTAs that each walk a few row blocks)
# beside the same side kernels.
PROPORTIONAL_SLACK = 0.02

_SPIN_KERNEL = "cake_rmsnorm_silu_gate_det_train_stall_spin"
_STAMP_KERNEL = "cake_rmsnorm_silu_gate_det_train_stall_stamp"
_SPIN_SOURCE = r"""
extern "C" __global__ void cake_rmsnorm_silu_gate_det_train_stall_spin(
    unsigned long long ns, int* smids, unsigned long long* starts) {
    unsigned long long t0, t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    if (threadIdx.x == 0) {
        unsigned int smid;
        asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
        smids[blockIdx.x] = (int)smid;
        starts[blockIdx.x] = t0;
    }
    do {
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    } while (t - t0 < ns);
}

extern "C" __global__ void cake_rmsnorm_silu_gate_det_train_stall_stamp(
    unsigned long long* stamp) {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    *stamp = t;
}
"""


def _checked(result):
    error, *values = result
    if int(error):
        raise RuntimeError(f"CUDA error {error}")
    return values[0] if values else None


@functools.cache
def _spin_cubin(major: int, minor: int) -> bytes:
    """The side kernels compiled once per architecture (NVRTC)."""
    from cuda.bindings import nvrtc

    prog = _checked(
        nvrtc.nvrtcCreateProgram(_SPIN_SOURCE.encode(), b"spin.cu", 0, [], [])
    )
    opts = [f"--gpu-architecture=sm_{major}{minor}".encode()]
    _checked(nvrtc.nvrtcCompileProgram(prog, len(opts), opts))
    size = _checked(nvrtc.nvrtcGetCUBINSize(prog))
    cubin = b" " * size
    _checked(nvrtc.nvrtcGetCUBIN(prog, cubin))
    _checked(nvrtc.nvrtcDestroyProgram(prog))
    return cubin


class _Spin:
    """A side kernel of ``ctas`` CTAs (default: one per SM) of ``threads``
    threads (driver API, so the preferred shared-memory carveout can be set on
    it; ``None`` keeps the default), and a one-thread kernel that stamps the
    time on the main stream right before a timed call."""

    def __init__(
        self,
        carveout: int | None,
        *,
        ctas: int | None = None,
        threads: int = ALL_SM_THREADS,
    ) -> None:
        from cuda.bindings import driver

        self._driver = driver
        device = torch.device("cuda", torch.cuda.current_device())
        torch.empty(1, device=device)  # primary context
        cubin = _spin_cubin(*torch.cuda.get_device_capability(device))
        ctx = _checked(driver.cuDevicePrimaryCtxRetain(device.index))
        _checked(driver.cuCtxSetCurrent(ctx))
        self._module = _checked(driver.cuModuleLoadData(cubin))
        self._func = _checked(
            driver.cuModuleGetFunction(self._module, _SPIN_KERNEL.encode())
        )
        self._stamp_func = _checked(
            driver.cuModuleGetFunction(self._module, _STAMP_KERNEL.encode())
        )
        if carveout is not None:
            _checked(
                driver.cuFuncSetAttribute(
                    self._func,
                    driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
                    carveout,
                )
            )
        self.sms = torch.cuda.get_device_properties(device).multi_processor_count
        self.ctas = self.sms if ctas is None else ctas
        self.threads = threads
        self.smids = torch.full((self.ctas,), -1, dtype=torch.int32, device=device)
        self.starts = torch.zeros(self.ctas, dtype=torch.int64, device=device)
        self.stamp_ns = torch.zeros(1, dtype=torch.int64, device=device)

    def _launch(self, func, ctas, threads, stream, args, types) -> None:
        _checked(
            self._driver.cuLaunchKernel(
                func,
                ctas,
                1,
                1,
                threads,
                1,
                1,
                0,
                self._driver.CUstream(stream.cuda_stream),
                (args, types),
                0,
            )
        )

    def launch(self, stream: torch.cuda.Stream) -> None:
        self.smids.fill_(-1)
        self.starts.fill_(torch.iinfo(torch.int64).max)
        self._launch(
            self._func,
            self.ctas,
            self.threads,
            stream,
            (
                int(ALL_SM_SPIN_MS * 1e6),
                self.smids.data_ptr(),
                self.starts.data_ptr(),
            ),
            (ctypes.c_ulonglong, ctypes.c_void_p, ctypes.c_void_p),
        )

    def stamp(self, stream: torch.cuda.Stream) -> None:
        """Record the time on ``stream`` (enqueued right before the call)."""
        self.stamp_ns.zero_()
        self._launch(
            self._stamp_func,
            1,
            1,
            stream,
            (self.stamp_ns.data_ptr(),),
            (ctypes.c_void_p,),
        )

    def held(self) -> bool:
        """The last launch's CTAs ran on ``ctas`` distinct SMs and had all
        started before the last stamp (read after a synchronize)."""
        coverage = int(torch.unique(self.smids[self.smids >= 0]).numel())
        return coverage == self.ctas and int(self.starts.max()) < int(self.stamp_ns)


# One side kernel per (device, carveout, CTAs, threads), loaded once per process.
_SPINS: dict[tuple[int, int | None, int | None, int], _Spin] = {}


def _spin(
    carveout: int | None, *, ctas: int | None = None, threads: int = ALL_SM_THREADS
) -> _Spin:
    key = (torch.cuda.current_device(), carveout, ctas, threads)
    if key not in _SPINS:
        _SPINS[key] = _Spin(carveout, ctas=ctas, threads=threads)
    return _SPINS[key]


def _spin_timed(call, spin: _Spin | None = None, side=None):
    """``(ms, cloned outputs)`` of ``call`` event-timed on the current stream,
    launched after ``spin`` started on ``side`` when given."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    if spin is not None:
        with torch.cuda.stream(side):
            spin.launch(side)
    torch.cuda._sleep(ALL_SM_PRE_SLEEP_CYCLES)
    if spin is not None:
        spin.stamp(torch.cuda.current_stream())
    start.record()
    out = call()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end), tuple(t.clone() for t in out)


def _beside(call, spin: _Spin, side: torch.cuda.Stream):
    """``call`` timed while ``spin`` holds its ``ctas`` distinct SMs from
    before the call; ``None`` when the side kernel did not in
    ``ALL_SM_TRIES`` launches."""
    for _ in range(ALL_SM_TRIES):
        timed = _spin_timed(call, spin, side)
        if spin.held():
            return timed
    return None


def _paired_medians(call, spin: _Spin, runs: int):
    """Medians of ``runs`` alone / beside-``spin`` pairs, and the pairs."""
    side = torch.cuda.Stream()
    for _ in range(WARMUP):
        _spin_timed(call)
    alone, busy = [], []
    for _ in range(runs):
        alone.append(_spin_timed(call))
        timed = _beside(call, spin, side)
        if timed is None:
            pytest.skip(
                f"the side kernel did not hold {spin.ctas} distinct SMs before "
                f"the call in {ALL_SM_TRIES} launches"
            )
        busy.append(timed)
    return (
        statistics.median(t for t, _ in alone),
        statistics.median(t for t, _ in busy),
        alone,
        busy,
    )


def _check_outputs(stage: str, tokens: int, alone, busy) -> None:
    """Every output of every call (``y``, or ``dx``, ``dz`` and, but for
    ``backward_nodw``, ``dw``) bitwise that of the first call alone."""
    names = {
        "forward": ("y",),
        "backward": ("dx", "dz", "dw"),
        "backward_nodw": ("dx", "dz"),
    }[stage]
    first = alone[0][1]
    for _t, out in alone + busy:
        for got, want, name in zip(out, first, names, strict=True):
            assert torch.equal(got, want), f"{stage} {name} tokens={tokens}"


@pytest.mark.parametrize("tokens", ALL_SM_TOKENS)
def test_forward_starts_beside_an_all_sm_kernel_with_the_max_shared_carveout(
    tokens: int,
) -> None:
    x, z, w, _dy = _inputs(tokens)
    y = torch.empty_like(x)

    def call():
        return (forward(x, z, w, EPS, out=y),)

    alone_ms, busy_ms, alone, busy = _paired_medians(
        call, _spin(CARVEOUT_MAX_SHARED), ALL_SM_RUNS
    )
    print(
        f"ALL_SM stage=forward tokens={tokens} heads={H} carveout=max_shared "
        f"alone_ms={alone_ms:.4f} busy_ms={busy_ms:.4f} "
        f"extra_ms={busy_ms - alone_ms:.4f}"
    )
    _check_outputs("forward", tokens, alone, busy)
    assert busy_ms - alone_ms <= MAX_ALL_SM_EXTRA_MS, (
        f"tokens={tokens}: forward alone {alone_ms:.4f} ms, beside an all-SM kernel "
        f"with the max-shared carveout {busy_ms:.4f} ms (at most "
        f"{MAX_ALL_SM_EXTRA_MS} ms more)"
    )


@pytest.mark.parametrize("tokens", ALL_SM_TOKENS)
@pytest.mark.parametrize("stage", ["backward", "backward_nodw"])
def test_backward_starts_beside_an_all_sm_kernel_with_the_max_shared_carveout(
    stage: str, tokens: int
) -> None:
    x, z, w, dy = _inputs(tokens)
    workspace = cake_rmsnorm_silu_gate_det_train_backward_workspace(
        tokens * H, x.device
    )
    compute = stage == "backward"

    def call():
        out = cake_rmsnorm_silu_gate_det_train_backward(
            dy, x, z, w, EPS, compute_grad_weight=compute, workspace=workspace
        )
        return out if compute else out[:2]

    alone_ms, busy_ms, alone, busy = _paired_medians(
        call, _spin(CARVEOUT_MAX_SHARED), ALL_SM_RUNS
    )
    print(
        f"ALL_SM stage={stage} tokens={tokens} heads={H} carveout=max_shared "
        f"alone_ms={alone_ms:.4f} busy_ms={busy_ms:.4f} "
        f"extra_ms={busy_ms - alone_ms:.4f}"
    )
    _check_outputs(stage, tokens, alone, busy)
    assert busy_ms - alone_ms <= MAX_ALL_SM_EXTRA_MS, (
        f"tokens={tokens}: {stage} alone {alone_ms:.4f} ms, beside an all-SM "
        f"kernel with the max-shared carveout {busy_ms:.4f} ms (at most "
        f"{MAX_ALL_SM_EXTRA_MS} ms more)"
    )


# Backwards that compute the weight gradient on up to 256 rows, as ``(T,
# H)``: one CTA up to 16 rows (1 and 16), one cluster of 16 CTAs past that
# (17, 48, 128 and 256) where the device runs clusters.
SMALL_ROW_SHAPES = ((1, 1), (1, 16), (1, 17), (1, 48), (1, 128), (2, 128))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"])
@pytest.mark.parametrize(
    "shape", SMALL_ROW_SHAPES, ids=[f"{t}x{h}" for t, h in SMALL_ROW_SHAPES]
)
def test_small_row_backward_starts_beside_an_all_sm_kernel_with_the_max_shared_carveout(
    shape: tuple[int, int], dtype: torch.dtype
) -> None:
    tokens, heads = shape
    rows = tokens * heads
    x, z, w, dy = _inputs(tokens, heads)
    index = torch.cuda.current_device()
    arch = loader.device_arch(index)
    clusters = loader.device_runs_clusters(index)
    if rows > 16 and not clusters:
        pytest.skip("this device runs no cluster program")
    role = "bwd_f32" if dtype == torch.float32 else "bwd_bf16"
    name = loader.dw_module(arch, loader.route_module(arch, role), rows, clusters)
    small = role + (loader.SOLO if rows <= 16 else loader.CLUSTER)
    assert loader.MODULES[name]["role"] == small, (rows, name)
    workspace = cake_rmsnorm_silu_gate_det_train_backward_workspace(rows, x.device)
    grad_weight = torch.empty(D, dtype=dtype, device=x.device)

    def call():
        return cake_rmsnorm_silu_gate_det_train_backward(
            dy, x, z, w, EPS, grad_weight=grad_weight, workspace=workspace
        )

    launched = []
    run_module = loader.run_module

    def spy(module, values, *, rows, device_index):
        launched.append(module)
        return run_module(module, values, rows=rows, device_index=device_index)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(loader, "run_module", spy)
        call()
    assert launched == [name], launched
    alone_ms, busy_ms, alone, busy = _paired_medians(
        call, _spin(CARVEOUT_MAX_SHARED), ALL_SM_RUNS
    )
    print(
        f"ALL_SM stage=backward_small tokens={tokens} heads={heads} "
        f"dw={'f32' if dtype == torch.float32 else 'bf16'} module={small} "
        f"carveout=max_shared alone_ms={alone_ms:.4f} busy_ms={busy_ms:.4f} "
        f"extra_ms={busy_ms - alone_ms:.4f}"
    )
    _check_outputs("backward", tokens, alone, busy)
    assert not bool(workspace.view(torch.int32).any()), "the workspace is not zero"
    assert busy_ms - alone_ms <= MAX_ALL_SM_EXTRA_MS, (
        f"{tokens}x{heads}: {small} alone {alone_ms:.4f} ms, beside an all-SM "
        f"kernel with the max-shared carveout {busy_ms:.4f} ms (at most "
        f"{MAX_ALL_SM_EXTRA_MS} ms more)"
    )


# The forward beside the item 1 side kernels: its CTA slots (``ctas_per_sm``
# per SM) get row blocks of ``rows_per_block`` rows, the most whole blocks per
# slot below the cut of rows per SM of the layout's rows on the device's
# architecture ("below": one CTA a block), the fewest whole waves of
# ``max_blocks_per_cta`` blocks per slot at or above it ("first") and four
# times that ("last"); a slot's block is 96 rows per SM (6 slots of 16 rows).
# On packed rows (the cut ``persistent_from_rows_per_sm``: 6500 rows per SM on
# sm_100a, 5600 on sm_103a, 8192 on sm_107a) that is 67, 68 and 272 blocks per
# slot on sm_100a (6432, 6528 and 26112 rows per SM), 58, 60 and 240 on
# sm_103a (5568, 5760 and 23040) and 85, 86 and 344 on sm_107a (8160, 8256 and
# 33024); on rows whose ``z`` is sliced from a projection (the cut
# ``persistent_from_rows_per_sm_strided``: 1000 rows per SM on each) 10, 12
# and 48 (960, 1152 and 4608 rows per SM) on each architecture.
FORWARD_EDGES = ("below", "first", "last")
FORWARD_LAYOUTS = ("packed", "strided")


def _forward_tokens(edge: str, layout: str = "packed") -> tuple[int, int, float]:
    """``(blocks_per_slot, T, cta_share)``: the row blocks per CTA slot of the
    forward at ``edge`` of the cut of ``layout``'s rows (``packed``: the rule's
    ``persistent_from_rows_per_sm``; ``strided``: its
    ``persistent_from_rows_per_sm_strided``) on this device's architecture,
    the fewest tokens ``T`` whose ``T * H`` rows give each slot that many,
    and the share of one CTA's blocks in the call alone (blocks a CTA walks
    over blocks a slot runs)."""
    device_index = torch.cuda.current_device()
    arch = loader.device_arch(device_index)
    name = loader.route_module(arch, "fwd")
    sms = torch.cuda.get_device_properties(device_index).multi_processor_count
    rule = loader.MODULES[name]["grid_rule"]
    slots = int(rule["ctas_per_sm"]) * sms
    rows_per_block = int(rule["rows_per_block"])
    most = int(rule["max_blocks_per_cta"])
    packed = layout == "packed"
    key = "persistent_from_rows_per_sm" + ("" if packed else "_strided")
    start = int(rule[key][arch])
    # rows per SM that give each slot of the SM one more row block
    step = int(rule["ctas_per_sm"]) * rows_per_block
    if edge == "below":
        blocks_per_slot = (start - 1) // step
        if blocks_per_slot < 1:
            pytest.skip(
                f"{arch}: no whole row block per slot below {start} rows per SM"
            )
    else:
        blocks_per_slot = -(-start // (step * most)) * most
        if edge == "last":
            blocks_per_slot *= 4
    tokens = -(-(blocks_per_slot * slots * rows_per_block) // H)
    rows = tokens * H
    assert loader.module_for_rows(arch, "fwd", rows, sms) == name
    assert (rows >= start * sms) == (edge != "below"), (edge, rows)
    (ctas, _, _) = loader.launch_grid(
        loader.MODULES[name], rows=rows, sm_count=sms, arch=arch, packed=packed
    )
    blocks = -(-rows // rows_per_block)
    return blocks_per_slot, tokens, -(-blocks // ctas) * slots / blocks


@pytest.mark.parametrize(
    "ctas,threads", PROPORTIONAL_SIDES, ids=["1x128", "16x640", "32x640"]
)
@pytest.mark.parametrize("edge", FORWARD_EDGES)
@pytest.mark.parametrize("layout", FORWARD_LAYOUTS)
def test_forward_slows_in_proportion_to_the_sms_held(
    layout: str, edge: str, ctas: int, threads: int
) -> None:
    blocks_per_slot, tokens, cta_share = _forward_tokens(edge, layout)
    x, z, w, _dy = _inputs(tokens, layout=layout)
    y = torch.empty_like(x)

    def call():
        return (forward(x, z, w, EPS, out=y),)

    spin = _spin(None, ctas=ctas, threads=threads)
    alone_ms, busy_ms, alone, busy = _paired_medians(call, spin, PROPORTIONAL_RUNS)
    ideal = spin.sms / (spin.sms - ctas)
    bound = ideal + cta_share + PROPORTIONAL_SLACK
    ratio = busy_ms / alone_ms
    print(
        f"HELD stage=forward tokens={tokens} heads={H} "
        f"blocks_per_slot={blocks_per_slot} side={ctas}x{threads} "
        f"alone_ms={alone_ms:.4f} busy_ms={busy_ms:.4f} ratio={ratio:.3f} "
        f"ideal={ideal:.3f} cta_share={cta_share:.4f} "
        f"excess={ratio - ideal - cta_share:.4f} edge={edge} layout={layout}"
    )
    _check_outputs("forward", tokens, alone, busy)
    assert ratio <= bound, (
        f"tokens={tokens} ({blocks_per_slot} row blocks per CTA slot, {edge}, "
        f"{layout}): forward alone {alone_ms:.4f} ms, beside {ctas} x {threads} "
        f"side CTAs on {ctas} SMs {busy_ms:.4f} ms: ratio {ratio:.3f} > "
        f"{bound:.3f} (proportional {ideal:.3f} + one CTA's blocks "
        f"{cta_share:.4f} + {PROPORTIONAL_SLACK})"
    )


# The tiny forward of packed rows: its probes, as rows per SM of this
# device's cut (``FWD_TINY_ROWS_PER_SM``): one row per SM, half the cut and
# the cut (the most rows it runs); and the side kernels beside it, the
# item 1 side kernels and 16 CTAs of 1024 threads with the max-shared
# carveout.
TINY_EDGES = ("one", "half", "cut")
TINY_SIDES = tuple((ctas, threads, None) for ctas, threads in PROPORTIONAL_SIDES) + (
    (16, 1024, CARVEOUT_MAX_SHARED),
)


@functools.cache
def _sm_limits(device_index: int) -> tuple[int, int]:
    """``(CTAs, threads)``: the most of each one SM of device
    ``device_index`` holds (its attributes, read once)."""
    from cuda.bindings import driver

    torch.empty(1, device=torch.device("cuda", device_index))  # primary context
    attribute = driver.CUdevice_attribute
    return (
        int(
            _checked(
                driver.cuDeviceGetAttribute(
                    attribute.CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR,
                    device_index,
                )
            )
        ),
        int(
            _checked(
                driver.cuDeviceGetAttribute(
                    attribute.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR,
                    device_index,
                )
            )
        ),
    )


def _tiny_forward(edge: str) -> tuple[int, str, float]:
    """``(T, module, cta_share)``: the most tokens ``T`` of ``H`` heads (at
    least one) within the rows per SM of ``edge`` on this device, the tiny
    forward that runs them (``forward_module`` of contiguous inputs), and the
    share of one CTA's blocks in the call alone: one block a CTA over the
    blocks a CTA slot runs, the call's CTAs over the slots, of which an SM
    has at most as many as it holds CTAs of the module's threads (its
    attributes; the share is then at its largest), and at most 1: a call of
    less than one wave still takes one block's time.  Skips where the device
    runs no tiny forward."""
    device_index = torch.cuda.current_device()
    arch = loader.device_arch(device_index)
    name = loader.ROUTES[arch].get("fwd_tiny")
    cut = loader.FWD_TINY_ROWS_PER_SM[arch] if name is not None else 0
    if cut < 1:
        pytest.skip(f"{arch} runs no tiny forward")
    rows_per_sm = {"one": 1, "half": max(1, cut // 2), "cut": cut}[edge]
    sms = torch.cuda.get_device_properties(device_index).multi_processor_count
    tokens = max(1, rows_per_sm * sms // H)
    rows = tokens * H
    tok = H * D if tokens > 1 else 0
    assert loader.forward_module(arch, rows, H, sms, tok, D, tok, D) == name, rows
    record = loader.MODULES[name]
    threads = int(record["launch"]["block"][0])
    (ctas, _, _) = loader.launch_grid(record, rows=rows, sm_count=sms, arch=arch)
    most_ctas, most_threads = _sm_limits(device_index)
    slots = min(most_ctas, most_threads // threads) * sms
    return tokens, name, min(1.0, slots / ctas)


def _launched(call) -> list[str]:
    """The modules ``call`` launches, in order."""
    launched = []
    run_module = loader.run_module

    def spy(module, values, *, rows, device_index):
        launched.append(module)
        return run_module(module, values, rows=rows, device_index=device_index)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(loader, "run_module", spy)
        call()
    return launched


@pytest.mark.parametrize("edge", TINY_EDGES)
def test_tiny_forward_keeps_its_speed_beside_a_resident_kernel(edge: str) -> None:
    tokens, name, _ = _tiny_forward(edge)
    x, z, w, _dy = _inputs(tokens)
    y = torch.empty_like(x)

    def call():
        return (forward(x, z, w, EPS, out=y),)

    assert _launched(call) == [name], (edge, tokens)
    side = torch.cuda.Stream()
    for _ in range(WARMUP):
        _timed(call, side=None)
    alone, busy = [], []
    for _ in range(RUNS):
        alone.append(_timed(call, side=None))
        busy.append(_timed(call, side=side))
    alone_ms = statistics.median(t for t, _ in alone)
    busy_ms = statistics.median(t for t, _ in busy)
    print(
        f"STALL stage=forward_tiny edge={edge} tokens={tokens} heads={H} "
        f"alone_ms={alone_ms:.4f} busy_ms={busy_ms:.4f} "
        f"ratio={busy_ms / alone_ms:.3f} extra_ms={busy_ms - alone_ms:.4f}"
    )
    _check_outputs("forward", tokens, alone, busy)
    assert _keeps_its_speed(alone_ms, busy_ms), (
        f"tiny forward tokens={tokens}: {busy_ms:.4f} ms beside a resident "
        f"kernel vs {alone_ms:.4f} ms alone"
    )


@pytest.mark.parametrize("edge", TINY_EDGES)
def test_tiny_forward_starts_beside_an_all_sm_kernel_with_the_max_shared_carveout(
    edge: str,
) -> None:
    tokens, name, _ = _tiny_forward(edge)
    x, z, w, _dy = _inputs(tokens)
    y = torch.empty_like(x)

    def call():
        return (forward(x, z, w, EPS, out=y),)

    assert _launched(call) == [name], (edge, tokens)
    alone_ms, busy_ms, alone, busy = _paired_medians(
        call, _spin(CARVEOUT_MAX_SHARED), ALL_SM_RUNS
    )
    print(
        f"ALL_SM stage=forward_tiny edge={edge} tokens={tokens} heads={H} "
        f"carveout=max_shared alone_ms={alone_ms:.4f} busy_ms={busy_ms:.4f} "
        f"extra_ms={busy_ms - alone_ms:.4f}"
    )
    _check_outputs("forward", tokens, alone, busy)
    assert busy_ms - alone_ms <= MAX_ALL_SM_EXTRA_MS, (
        f"tokens={tokens}: tiny forward alone {alone_ms:.4f} ms, beside an "
        f"all-SM kernel with the max-shared carveout {busy_ms:.4f} ms (at most "
        f"{MAX_ALL_SM_EXTRA_MS} ms more)"
    )


@pytest.mark.parametrize(
    "ctas,threads,carveout",
    TINY_SIDES,
    ids=["1x128", "16x640", "32x640", "16x1024-max_shared"],
)
@pytest.mark.parametrize("edge", TINY_EDGES)
def test_tiny_forward_slows_in_proportion_to_the_sms_held(
    edge: str, ctas: int, threads: int, carveout: int | None
) -> None:
    tokens, name, cta_share = _tiny_forward(edge)
    x, z, w, _dy = _inputs(tokens)
    y = torch.empty_like(x)

    def call():
        return (forward(x, z, w, EPS, out=y),)

    assert _launched(call) == [name], (edge, tokens)
    spin = _spin(carveout, ctas=ctas, threads=threads)
    alone_ms, busy_ms, alone, busy = _paired_medians(call, spin, PROPORTIONAL_RUNS)
    ideal = spin.sms / (spin.sms - ctas)
    bound = ideal + cta_share + PROPORTIONAL_SLACK
    ratio = busy_ms / alone_ms
    side = f"{ctas}x{threads}" + ("s" if carveout is not None else "")
    print(
        f"HELD stage=forward_tiny edge={edge} tokens={tokens} heads={H} "
        f"side={side} alone_ms={alone_ms:.4f} busy_ms={busy_ms:.4f} "
        f"ratio={ratio:.3f} ideal={ideal:.3f} cta_share={cta_share:.4f} "
        f"excess={ratio - ideal - cta_share:.4f}"
    )
    _check_outputs("forward", tokens, alone, busy)
    assert ratio <= bound, (
        f"tokens={tokens} ({edge}): tiny forward alone {alone_ms:.4f} ms, beside "
        f"{ctas} x {threads} side CTAs ({side}) on {ctas} SMs {busy_ms:.4f} ms: "
        f"ratio {ratio:.3f} > {bound:.3f} (proportional {ideal:.3f} + one CTA's "
        f"blocks {cta_share:.4f} + {PROPORTIONAL_SLACK})"
    )
