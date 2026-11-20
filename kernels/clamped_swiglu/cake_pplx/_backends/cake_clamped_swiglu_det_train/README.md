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
| `fwd` large-R | from 112 rows per SM | from 208 | from 384 |
| `bwd` large-R | [12, 21], [32, 41], from 48 rows per SM | [12, 21], [32, 41], from 48 | [12, 21], [35, 41], from 48 |

A window changes 0.1.1's route only where forced small/large sweeps showed the other program not slower by more than
0.5 % on that architecture, with both weighted and unweighted inputs.  In these sweeps the route was forced to each
program, and both were timed at fixed row counts on B200, GB300 and VR200, with warm L2 (CUDA graph) and cold L2
(CUPTI).  The inputs were weighted, unweighted and deep-gate (every warp takes the deep-gate sigmoid, exact before
0.1.4).  The sweeps timed the backward alone and in forward + backward and checkpointed steps.  They timed the forward
alone with merged and split inputs, and in those steps only on GB300 with a split input.  They have no forward +
backward or checkpointed timing for the B200 forward change (96 to 112 rows per SM).  There only the release A/B of
0.1.2 against 0.1.1 covers those steps, and it marks none of them slower: on the graph and CUPTI timers the highest
ratio of 0.1.2's time to 0.1.1's is 1.006, with cold L2 at 111.99 rows per SM, within noise.  Below, small/large is the
small-R program's time over the large-R one's.  The bounds are whole numbers of rows per SM and run the large-R program,
as 0.1.1's cut did.  So a window also holds rows where the small-R program measures faster in some cases.  For example:
the backward alone at the bounds 12, 32 and 48 rows per SM and at 36; the forward + backward and checkpointed steps at
35 rows per SM on `sm_107a` (small/large 0.977-0.996); every measured case from about 20.25 to 21 and from 40.5 to 41
rows per SM; and the forward with the deep-gate input (small/large 0.815-0.957 inside the forward windows).  These rows
ran the large-R program in 0.1.1 too, so nothing changes for them.

Both backward programs run 4 rows a CTA: the small-R one fits 3 CTAs on an SM (12 rows per SM a wave), the large-R one 5
(20 rows).  Above 12 and up to 20 rows per SM, and above 36 and up to 40, the small-R program runs twice the waves of
the large-R one.  There it measured slower in every forward + backward and checkpointed case at 16, 20 and 40 rows per
SM (small/large 1.005-1.090).  The backward alone is mixed there: the small-R program is faster in some cases (down to
0.954) and slower in others (up to 1.083).  Just above 20 rows per SM the small-R program measured up to 3.5 % slower in
the checkpointed step (GB300, in the first A/B of this release; the forced sweeps there read up to 2.2 %).  Just above
40 it measured up to about 3 % slower in some cases (the backward alone and both steps).  From about 20.25 and 40.5 rows
per SM on, it measured as fast or faster in every measured case (small/large 0.901-0.983 at 21 * SMs + 1 rows and
0.934-0.995 at 41 * SMs + 1), so those windows end at 21 and 41.  At 36 rows per SM the large-R program's unweighted
backward with warm L2 measures as fast or faster (small/large 1.003-1.006 on B200 and 1.011-1.016 on GB300, merged and
split).  In the other measured cases there the small-R backward is faster.  The sweeps have no point between 32 and 36,
so on `sm_100a` and `sm_103a` that window begins at 32.  On `sm_107a` the small-R program is faster in both steps at 35
rows per SM (0.977-0.996), and at 36 * SMs - 1 rows the split unweighted steps read 1.005-1.007 with warm L2, so that
window begins at 35.  From 48 rows per SM on, the small-R backward alone measures 0.92 to 1.05 times the large-R one's
time (by GPU, input and timer).  Right after a forward over the same rows it measured up to 1.5 % slower: in the
checkpointed step with unweighted inputs, on GB300 at 56-64 rows per SM (with both timers) and on B200 at 60 and 112.
So the large-R program runs there.  Inside the small-R backward ranges between the windows, every swept case reads
small/large at most 1.000 (raw maximum 1.0004).  In the forward on `sm_100a`, the small-R program alone stays faster up
to 112 rows per SM.  With merged weighted and unweighted inputs it reads small/large 0.988-0.998 at 112.  For the split
layout near 112, the release A/B reads the split forward at 111.99 rows per SM at 0.978-0.995 of 0.1.1's time with
weighted and unweighted inputs on the graph and CUPTI timers.  So that window begins at 112 (0.1.1: 96).  On `sm_103a`
(GB300) the small-R forward alone is faster with a merged input up to about 384 rows per SM.  With a split input, from
224 rows per SM on, it is faster in some cases (down to 0.989) and more than 0.5 % slower in others (weighted and
unweighted inputs, warm L2: up to 1.008 at 384).  At 192 and 208 the split forward reads 0.988-0.997 alone and
0.997-1.001 in the forward + backward and checkpointed steps, so that window begins at 208 (0.1.1: 192).  On `sm_107a`
the window stays at 0.1.1's 384.  From 320 to 384 rows per SM, with weighted and unweighted inputs, the two forward
programs are within about 0.5 % of each other (small/large 0.995-1.005; the small-R forward is up to 0.53 % slower with
a merged input and cold L2).  With the deep-gate input the small-R forward is 5.6-6.2 % faster at 384.  Below 384 the
small-R forward runs, as in 0.1.1.

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
a stage compute each element with the same FP32 operations (the deep-gate sigmoid gives the fast one's bits wherever
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
  scaled reciprocal: the Newton reciprocal of `d 2^-126`, times `2^-126` with one rounding.  That is `RN(1 / d)` where
  the sigmoid is normal and within one subnormal ulp of it where it is subnormal (0.1.3 and earlier rounded it exactly,
  at the cost of more instructions per element), +0 for an infinite `d`.
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
  the backward recomputes it.  The backward reuses the forward's FP32 `limit` and checks the saved inputs again, since
  a saved-tensors hook or a `.data` assignment can change them after the forward.
- Host caches, none of which changes an output bit: per device, its architecture, its SM count, its `torch.device`
  and one launcher per stage (the route, and each program's FFI entry bound at its first launch); per process, the
  loaded libraries and the FP32 rounding of the last accepted `limit`.  The architecture is still checked at every
  call, before any allocation.

## Limitations

- The intermediate width is 2304 only (a compile-time width of the kernels).
- No double backward: differentiating gradients computed with `create_graph=True` raises `RuntimeError`.  Not tested
  under `torch.compile` or `torch.jit.trace`.  No MXFP8 or FP4 output.
- A program's library builds at its first launch, and the row count picks the program (see Programs).  Before
  capturing a CUDA graph, run an eager call at each row count the graph captures, or build every library of the
  architecture with `specs_for_arch` (which leaves out a library whose `min_cuda` the local toolkit is below).
