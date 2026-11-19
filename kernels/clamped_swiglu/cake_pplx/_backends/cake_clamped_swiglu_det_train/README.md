# Deterministic routed-expert clamped SwiGLU training backend (SM100 / SM103 / SM107)

Backend of `cake_pplx.cake_clamped_swiglu_det_train`: the clamped SwiGLU of a routed MoE expert in training, forward
and backward, intermediate width `I = 2304`.  The route weight multiplies the FP32 activation before the one BF16
rounding of `h`, and the route-weight gradient uses the unweighted activation, so a zero weight keeps its gradient.

The facade `cake_pplx.cake_clamped_swiglu_det_train` is a stable API.  This backend package is experimental, and its
internal names may change.  The facade imports the backend on first use.  The supported architectures are SM100
(B200 / GB200), SM103 (B300 / GB300) and SM107 (R200).  Another device raises `RuntimeError`.  The package loads its
libraries prebuilt from `aot/` when a builder compiled them.  Otherwise the package JIT-builds them from the sources
and flags that `build.json` records.  No environment variable changes a call; every control is an argument.  A JIT
build (no prebuilt `aot/` library) reads the JIT variables listed in the repository README, section JIT build and
cache.  A JIT build needs a CUDA toolkit of at least 12.8 for `sm_100a`, 12.9 for `sm_103a` and 13.5 for `sm_107a`
(`build.json` `min_cuda`); an older toolkit raises `RuntimeError`, naming the floor, when that library is built.

## Usage

```python
import torch
import torch.utils.checkpoint
from cake_pplx.cake_clamped_swiglu_det_train import cake_clamped_swiglu_det

# gate_up: BF16 [R, 4608] (gate columns [0, 2304), up [2304, 4608)), the output of the gate/up projection
# w: FP32 [R] route weights (>= 0; zeros allowed)
h = cake_clamped_swiglu_det(gate_up, weight=w, limit=10.0)  # BF16 [R, 2304], the down projection's input
h.backward(dh)  # gate_up.grad: BF16 [R, 4608]; w.grad: FP32 [R]

# two tensors, and the shared expert (no route weight)
h = cake_clamped_swiglu_det(gate, up, w, limit=10.0)
h_shared = cake_clamped_swiglu_det(gate_up_shared, limit=10.0)

# under activation checkpointing: only the inputs are kept
h = torch.utils.checkpoint.checkpoint(cake_clamped_swiglu_det, gate_up, None, w, limit=10.0, use_reentrant=False)
```

The steps without autograd, `cake_clamped_swiglu_det_train_forward` and `cake_clamped_swiglu_det_train_backward`,
take the same operands and optional preallocated outputs (`out=`, `grad_gate_up_or_gate=`, `grad_up=`,
`grad_weight=`), for a caller that fuses the activation into its own `torch.autograd.Function`.  `limit` is a
required keyword everywhere.

## Inputs and outputs

| operand | shape | dtype | notes |
|---|---|---|---|
| `gate`, `up` | `[R, 2304]` each | BF16 | or one `[R, 4608]` tensor (first argument, `up=None`); any row stride |
| `weight` | `[R]` | FP32 | route weight, `>= 0`; `None` for an unweighted expert (`w = 1`) |
| `limit` | Python number | | `L`, finite and positive, rounded to FP32; a launch argument |
| `h` | `[R, 2304]` | BF16 | new and contiguous, or the caller's `out` |
| `dh` | `[R, 2304]` | BF16 | upstream gradient |
| `d_gate`, `d_up` | `[R, 2304]` each | BF16 | or one `[R, 4608]` gradient for a merged input |
| `d_weight` | `[R]` | FP32 | `None` without `weight` |

- Every operand is on one CUDA device.  `R` is any count from 0 up to `2**31 - 257`.
- The kernels read every BF16 tensor through its row stride in elements: a row stride that is a multiple of 8, a unit
  last stride and a 16-byte aligned base are read in place (the halves of a merged tensor, row slices of a wider
  tensor).  Another input layout is copied once; a non-contiguous `weight` too.
- A caller's output buffer must already have such a layout, with rows that do not overlap (row stride at least the
  row width); otherwise it is refused (`ValueError`).  An output (`out`, `grad_gate_up_or_gate`, `grad_up`,
  `grad_weight`) must not overlap any input or another output; this is not checked.
- A wrong type, or a `limit` that is not a Python number, raises `TypeError` (a tensor `limit` would need a host
  read).  A wrong device, dtype or shape, a non-finite or non-positive `limit` or one past FP32 (an integer or a
  fraction past every Python float too), or an unfit output raises `ValueError`.  An unsupported device raises
  `RuntimeError`, also with `R = 0`.
- Every argument is checked before an input is copied or an output allocated: a call refused for its arguments copies,
  allocates and launches nothing.
- `R = 0` returns empty outputs and launches nothing.

## Programs

Each stage has two programs (`cake_jit.MODULES`, keyed by program name): a small-R program (role `fwd` or `bwd`) and
a large-R program (role `fwd_large` or `bwd_large`).  `cake_jit.PROGRAM_OF` maps a role and an architecture to its
program.  A program is compiled with the exact `-gencode` flag of each architecture it is delivered for (`build.json`).
A call runs the large-R program when `R` lies in one of the record's windows of rows per SM on the device's
architecture (`rows_per_sm_windows`): `lo * SMs <= R <= hi * SMs` for a window `[lo, hi]` (both bounds included), where
`hi` None has no upper bound.  It runs the small-R program at every other `R` (`cake_jit.route`).  The package reads
the architecture once per device and the SM count once per stage and device.  An architecture without a large-R program
of a stage runs the small-R one at every `R`.

| stage | `sm_100a` | `sm_103a` | `sm_107a` |
|---|---|---|---|
| `fwd` large-R | from 112 rows per SM | from 384 | from 384 |
| `bwd` large-R | [12, 21], [32, 41], from 48 rows per SM | [12, 21], [32, 41], from 48 | [12, 21], [35, 41], from 48 |

Outside its windows, the small-R program measures at least as fast as the large-R one on that architecture (weighted and
unweighted, merged and split, warm and cold L2, the stage alone and in a forward + backward or checkpointed step);
inside, the large-R one measures faster in one of them.  A bound runs the large-R program.  Both backward programs run 4
rows a CTA: the small-R one fits 3 CTAs on an SM (12 rows an SM a wave), the large-R one 5 (20 rows).  From 12 to 20
rows per SM and from 36 to 40, the small-R program runs twice the waves of the large-R one and is slower.  Just above 20
and 40 rows per SM, the large-R program's last wave holds a few CTAs and costs little, so those windows end at 21 and
41.  At 36 rows per SM the large-R program's unweighted backward already measures as fast or faster with warm L2
(small/large 1.003 on B200, 1.016 on GB300), and the sweeps have no point between 32 and 36, so on `sm_100a` and
`sm_103a` that window starts at 32.  On `sm_107a` the small-R program stays faster up to 35 rows per SM, and right after
a forward the large-R one turns faster between 35 and 36, so that window starts at 35.  From 48 rows per SM on, the
small-R backward alone measures 0.92 to 1.05 times the large-R one's time (by GPU, input and timer), and up to 1.5 %
slower right after a forward over the same rows (the checkpointed step, unweighted), so the large-R program runs there.

Every program is one kernel: 32 lanes own a row and a thread owns one row.  Each lane makes 9 loads of 8 BF16 values
of the gate and as many of the up (and of `dh` in the backward), at columns `(p * 32 + lane) * 8`, so every load of a
warp is contiguous.  The programs differ in their schedule:

| stage | program | architectures | CTA | a lane's 9 loads |
|---|---|---|---|---|
| `fwd` | small-R | `sm_100a`, `sm_103a`, `sm_107a` | 128 threads, 4 rows | 3 runs of 3; a run's loads are issued while the run before it computes |
| `fwd` | large-R | `sm_100a`, `sm_103a` | 256 threads, 8 rows | one run |
| `fwd` | large-R | `sm_107a` | 128 threads, 4 rows | one run; the values of each load are stored as soon as they are computed |
| `bwd` | small-R | `sm_100a`, `sm_103a`, `sm_107a` | 128 threads, 4 rows | 3 runs of 3; a run's loads are issued while the run before it computes |
| `bwd` | large-R | `sm_100a`, `sm_103a`, `sm_107a` | 128 threads, 4 rows | 3 runs of 3, one run at a time |

More runs hold fewer registers, so more warps fit on an SM.  The grid is `ceil(R / rows a CTA)` CTAs (`cake_jit.grid`,
from the launch block and `cake_jit.BUILD_OF`).  A row past `R` loads the last row and stores nothing.  The programs of
a stage compute each element with the same FP32 operations (the exact sigmoid gives the bits of the fast one wherever
the fast one may be taken, so the loads a warp's vote covers change no bit), and the `d_weight` sums take a lane's
columns in load order with the same lanes.  So both programs store the same bits for every input: the route changes no
output bit.  No program uses shared memory, a workspace, a cooperative launch or programmatic dependent launch.  The
row count, the limit, the weight flag and the strides are launch arguments, so one library per program and
architecture serves every call.

## Numerics

- `g' = min(g, L)` and `u' = clamp(u, -L, L)` from the BF16 values widened to FP32 (exact); the masks compare the raw
  inputs with the FP32 `L`, inclusively.
- `s = 1 / (1 + 2^(-g' log2 e))`: the approximate base-2 exponential (`ex2.approx.ftz`; `1 + 2^t` absorbs a flushed
  subnormal `2^t`) and the approximate reciprocal plus one round-to-nearest Newton step, which is the IEEE `rcp.rn` of
  every `1 <= d < 2^126`.  A warp with a clamped gate at or below -87 in a run of its loads computes that run with a
  branch-free exact reciprocal: `RN(1 / d)` for every `d`, subnormal results kept, +0 for an infinite `d`.
- `silu = g' s`, `a = silu u'`, `h = bf16(w a)`: one rounding.
- `d_gate = bf16(((dh w) u') (silu (1 - s) + s) [g <= L])` (one FMA for `silu (1 - s) + s`) and
  `d_up = bf16((dh w) silu [-L <= u <= L])`, each mask a factor of 1 or 0, each output rounded once.
- `d_weight = sum over c of dh a` in FP32 from the unweighted `a`: each lane keeps two FMA chains (its even and its
  odd columns) in load order and adds them, then XOR shuffles sum the row's lanes.
- Every product and FMA rounds to nearest without flushing subnormals.  Nothing is FP64.  A NaN input gives a NaN where
  the float64 operation has one (the clamps propagate NaN).

## Determinism

Every output is elementwise or a sum within one row in one fixed order, so every call gives the same bits for the same
inputs on the same architecture, whatever the row count, the layout (merged, two tensors, strided), the stream or a CUDA
graph replay.  No call alerts under `torch.use_deterministic_algorithms(True)`, and no output bit changes; no entry
point takes a `deterministic` keyword.  The family has no non-deterministic module: there is no reduction across rows
or CTAs to reorder.

With `torch.use_deterministic_algorithms(True)` and `torch.utils.deterministic.fill_uninitialized_memory` (on by
default), PyTorch fills every tensor a call allocates with NaN before the kernel writes it: one more fill launch per
new output (`h`; `d_gate_up`, or `d_gate` and `d_up`; `d_weight`, also the one an unweighted backward drops) and per
copied input.  Output buffers allocated once and passed to every call (`out=`, `grad_gate_up_or_gate=`, `grad_up=`,
`grad_weight=`) avoid the fills of those outputs.  The kernels write every element of every output, so a fill never
reaches a result.

## Workspace and retained state

- Workspace: none.  An unweighted backward allocates the `R` FP32 weight-gradient values the kernel writes and drops
  them.  An unweighted launch passes, as the weight, one cached one-element FP32 buffer of the device, set to 1, which
  the process's first unweighted call on that device makes (that call may be a CUDA graph capture).  The kernels read
  it and ignore it.
- Retained state of the autograd function: its inputs only (`gate` and `up` or the merged tensor, and `weight`),
  references with no copy: `R * 4608 * 2 + R * 4` bytes of tensors the caller already holds.  No activation is saved;
  the backward recomputes it.

## Limitations

- The intermediate width is 2304 only (a compile-time width of the kernels).
- No double backward: differentiating gradients computed with `create_graph=True` raises `RuntimeError`.  Not tested
  under `torch.compile` or `torch.jit.trace`.  No MXFP8 or FP4 output.
- A program's library builds at its first launch, and the row count picks the program (see Programs).  Before
  capturing a CUDA graph, run an eager call at each row count the graph captures, or build every library of the
  architecture with `specs_for_arch` (which leaves out a library whose `min_cuda` the local toolkit is below).
