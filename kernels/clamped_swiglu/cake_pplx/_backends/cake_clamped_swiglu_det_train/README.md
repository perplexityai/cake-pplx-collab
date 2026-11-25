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

# activation checkpointing: checkpoint the op together with the down projection (w_down: BF16 [hidden, 2304])
def swiglu_down(gate_up, w, w_down):
    return torch.nn.functional.linear(cake_clamped_swiglu_det(gate_up, weight=w, limit=10.0), w_down)

y = torch.utils.checkpoint.checkpoint(swiglu_down, gate_up, w, w_down, use_reentrant=False)
```

The op saves only its inputs, so checkpointing it alone saves no memory: the down projection keeps `h` for its weight
gradient.  In the backward of a non-reentrant checkpoint, the recomputation runs with saved-tensors hooks set; there
`cake_clamped_swiglu_det` launches its forward kernel after autograd saved its inputs, as a built-in op does.  So with
early stop on (the torch default), the recomputation stops before that launch when nothing later in the region saves a
tensor, and runs the forward when a later operation saves one.  Anywhere else (no hooks, or not in a backward, such as
the checkpoint's own forward pass) the autograd function's `forward` launches, as up to 0.1.7.  The launch and the bits
are the same in every case.  Per checkpointed region (down projection: `F.linear(h, w_down)`, which saves `h`):

| Region | Not kept from the forward to the backward | Forward of this op in the backward |
|---|---|---|
| The op alone (down projection outside) | Nothing (the op keeps its inputs, the down projection `h`) | None (0.1.7: one) |
| Gate/up projection, then the op (down projection outside) | The projection's `[R, 4608]` output | None (0.1.7: one); the projection is recomputed |
| The op, then the down projection (above) | `h` | One, as in 0.1.7 |
| The whole block (gate/up projection, op, down projection) | The projection's output and `h` | One, as in 0.1.7 |
| Any region with `use_reentrant=True`, `early_stop=False` (or `set_checkpoint_early_stop(False)`), or a direct `CakeClampedSwiGLUDetFunction.apply` | As above | One, also when its output is dropped |

A call that Dynamo or `torch.jit.trace` records launches inside `forward`; a checkpoint's recomputation runs with Dynamo
disabled.  Not tested under `torch.compile`.

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

The forward runs stage `fwd`, a weighted backward stage `bwd` and an unweighted backward stage `bwd_unweighted`
(`cake_jit.MODULES`, keyed by program name).  Each stage has a small-R program (role `fwd`, `bwd` or `bwd_unweighted`)
and, from 0.1.9, a fine program for the fewest rows (role `fwd_fine`, `bwd_fine` or `bwd_unweighted_fine`), on all
three architectures.  `fwd` and `bwd` also have a large-R program (role `fwd_large` or `bwd_large`), and from 0.1.10
`bwd` also has a relay program (role `bwd_relay`).  `cake_jit.PROGRAM_OF` maps a role and an architecture to its
program.  A program is compiled with the exact `-gencode` flag of each architecture it is delivered for (`build.json`).
A call runs the fine program when `R <= hi * SMs` for the fine record's one window `[0, hi]` of rows per SM on the
device's architecture (`rows_per_sm_windows`, `cake_jit.fine`).  At more rows it runs a windowed program (the large-R
or the relay one) when `R` lies in one of its record's windows: `lo * SMs <= R <= hi * SMs` for a window `[lo, hi]`
(both bounds included), where `hi` None has no upper bound.  The windows of a stage's windowed programs do not overlap
or touch.  It runs the small-R program at every other `R` (`cake_jit.route`).  The package reads the architecture once
per device and the SM count once per stage and device.  Per stage and device it makes one table with an entry per row
count up to the fine window's last row count or the last edge of the windowed programs, whichever is larger.  A call
compares `R` with that table's length and reads one entry.  A call of more rows runs the windowed program whose last
window has no upper bound (`fwd`; `bwd` on `sm_100a` and `sm_107a`), and the small-R program otherwise (`bwd` on
`sm_103a`, and `bwd_unweighted`, which has no large-R or relay program).  An architecture without a fine, a large-R or
a relay program of a stage runs the small-R one in its place.  A fine window may reach at most 64 rows per SM
(`cake_jit.MAX_FINE_ROWS_PER_SM`), so the fine window adds at most `64 * SMs` entries to the table.

| stage | `sm_100a` | `sm_103a` | `sm_107a` |
|---|---|---|---|
| `fwd` fine | up to 8 rows per SM | up to 4 | up to 4 |
| `fwd` large-R | from 112 rows per SM | from 208 | from 384 |
| `bwd` fine | up to 2 rows per SM | up to 2 | up to 2 |
| `bwd` relay | [3, 5], from 25 rows per SM | [25, 647] | [3, 6], from 31 |
| `bwd` large-R | [13, 21] rows per SM | [12, 21] | [16, 21] |
| `bwd_unweighted` fine | up to 4 rows per SM | up to 4 | up to 2 |

The forward's large-R windows change 0.1.1's route only where forced small/large sweeps showed the other program not
slower by more than 0.5 % on that architecture, with both weighted and unweighted inputs.  In these sweeps the route was
forced to each program, and both were timed at fixed row counts on B200, GB300 and VR200, with warm L2 (CUDA graph) and
cold L2 (CUPTI).  The inputs were weighted, unweighted and deep-gate (every warp takes the deep-gate sigmoid, exact
before 0.1.4).  The sweeps timed the forward alone with merged and split inputs, and in forward + backward and
checkpointed steps only on GB300 with a split input.  They have no forward + backward or checkpointed timing for the
B200 forward change (96 to 112 rows per SM).  There only the release A/B of 0.1.2 against 0.1.1 covers those steps,
and it marks none of them slower: on the graph and CUPTI timers the highest ratio of 0.1.2's time to 0.1.1's is 1.006,
with cold L2 at 111.99 rows per SM, within noise.  Below, small/large is the small-R program's time over the large-R
one's.  The bounds are whole numbers of rows per SM and run the large-R program, as 0.1.1's cut did.  So a window also
holds rows where the small-R program measures faster in some cases: with the deep-gate input the forward reads
small/large 0.815-0.957 inside the forward windows.  These rows ran the large-R program in 0.1.1 too, so nothing
changes for them.  In the forward on `sm_100a`, the small-R program alone stays faster up to 112 rows per SM.  With
merged weighted and unweighted inputs it reads small/large 0.988-0.998 at 112.  For the split layout near 112, the
release A/B reads the split forward at 111.99 rows per SM at 0.978-0.995 of 0.1.1's time with weighted and unweighted
inputs on the graph and CUPTI timers.  So that window begins at 112 (0.1.1: 96).  On `sm_103a` (GB300) the small-R
forward alone is faster with a merged input up to about 384 rows per SM.  With a split input, from 224 rows per SM on,
it is faster in some cases (down to 0.989) and more than 0.5 % slower in others (weighted and unweighted inputs, warm
L2: up to 1.008 at 384).  At 192 and 208 the split forward reads 0.988-0.997 alone and 0.997-1.001 in the forward +
backward and checkpointed steps, so that window begins at 208 (0.1.1: 192).  On `sm_107a` the window stays at 0.1.1's
384.  From 320 to 384 rows per SM, with weighted and unweighted inputs, the two forward programs are within about
0.5 % of each other (small/large 0.995-1.005; the small-R forward is up to 0.53 % slower with a merged input and cold
L2).  With the deep-gate input the small-R forward is 5.6-6.2 % faster at 384.  Below 384 the small-R forward runs, as
in 0.1.1.

The `bwd` windows route weighted backward calls only: an unweighted backward runs the `bwd_unweighted` programs at
every `R`.  From 0.1.10 the `bwd` relay and large-R windows are one route, derived by a fixed rule from one forced sweep
on B200, GB300 and VR200 (at full memory clock):

- At every swept row the sweep timed each of the three backward programs (the small-R, the large-R and the relay one)
  and a second copy of the large-R program, on two GPUs (four on GB300).  A GPU's noise band is the 95th percentile of
  the two copies' time ratio, less one, over a range of rows (below 12, from 12 to 48 and from 48 rows per SM), by
  step and timer.
- The rows are `R = k * SMs` for every k from 3 to 48; `k * SMs - 1` and `k * SMs + 1` for k in 12, 16, 20, 21, 32, 35,
  36, 40, 41 and 48; `k * SMs` for k in 56, 64, 80, 96, 112 and 128; and R = 4096, 16384, 32768, 65536, 98304 and
  196608.
- Each row has 24 cases: the backward alone, a forward + backward step and a checkpointed step; warm L2 (CUDA graph)
  and cold L2 (CUPTI); merged and split inputs; normal and deep-gate inputs.
- A program may run at a swept row only if no case there reads it slower than 0.1.9's program.  A case reads slower
  when two GPUs (two of four on GB300) measure it slower beyond their band, or when a GPU has no reading.
- At each row the route takes the allowed program with the lowest geometric mean of its 24 median times.  Programs
  within 0.1 % of it count as tied, and a tie goes to 0.1.9's program, then to the small-R one.
- A program holds at most two windows on an architecture.  Where that forces a change, the row takes the allowed
  program that keeps the fewest windows, 0.1.9's program first.
- The rows between two consecutive swept rows run a program allowed at both of them.

The small-R and the large-R backward programs run 4 rows a CTA: the small-R one fits 3 CTAs on an SM (12 rows per SM a
wave), the large-R one 5 (20 rows).  The relay program runs one row a CTA and is sized for 9 CTAs on an SM (9 rows per
SM a wave).  The rule gives these windows, but for one change on `sm_103a`.  There the rule gave the relay program 12 to
13 rows per SM and the large-R one 14 to 21.  Two windows may not touch, so the rows between them (R = 1977 to 2127 on
152 SMs) ran the small-R program.  There the release A/B measured the backward alone 1.3 to 2.7 % slower than 0.1.9's
large-R program (cold L2, merged and split inputs, four GPUs).  So on `sm_103a` the large-R program keeps 0.1.9's window
from 12 rows per SM, and the relay program has no window below 25.  On `sm_103a` the relay window ends at 647 rows per
SM, the first whole number at or above 98304 / 152 SMs (646.7).  At 196608 rows there the relay program's geometric mean
is within 0.1 % of the small-R one's, so the small-R program runs above 647 rows per SM.  Every program of a stage
stores the same bits for every input, so the route changes no output bit.

At a few rows per SM, a small-R program runs each row (in `bwd_unweighted`, each run of 768 columns) on one warp, alone
on its SM sub-partition.  Its time is that warp's serial chain of loads.  A fine program splits a row over 9 tasks, one
load of each input a task, so a row's chain is 9 times shorter and takes one memory trip.  At more rows its extra
per-task instructions (`fwd`, `bwd_unweighted`) or its per-row barriers and shared-memory hand-off (`bwd`) cost more
than they save.  Each fine window ends at the largest swept multiple `j * SMs` (j in 1, 2, 4, 8, 12 and 16) at and
below which no swept row measured the fine program slower than the small-R one beyond the run's noise band.  These
sweeps forced each program and timed R = 1, 2, 3, 4, 8, 16, 32, 64, 128 and `j * SMs - 1`, `j * SMs` and
`j * SMs + 1` on B200, GB300 and VR200.  They used warm L2 (CUDA graph) and cold L2 (CUPTI), weighted and unweighted,
merged and split, and ordinary and deep-gate inputs.  Below, fine/small is the fine program's time over the small-R
one's, at the first slower swept row; each was rechecked on a second GPU of the same kind:

- `fwd`: `12 * SMs - 1` rows on `sm_100a` (1.056, warm L2); `8 * SMs - 1` on `sm_103a` (1.031) and `sm_107a`
  (1.098-1.103).  At 8 rows per SM `sm_100a` reads 0.998, so its crossover lies near the bound.
- `bwd`: `4 * SMs - 1` rows on every architecture (1.029, 1.091 and 1.122-1.165).
- `bwd_unweighted`: `8 * SMs - 1` rows on `sm_100a` (1.047) and `sm_103a` (1.075); `4 * SMs - 1` and `4 * SMs` on
  `sm_107a` (1.024-1.025, cold L2).

The sweeps have no point between two swept multiples, so a window may end below its crossover.  Each fine window ends
below its stage's first large-R or relay window.  At 1 row the fine programs measured, as a share of the small-R
program's time, 0.64-0.69 (`fwd` on B200 and GB300), 0.69-0.80 (`fwd` on VR200), 0.70-0.79 (`bwd`) and 0.77-0.94
(`bwd_unweighted`).

Every program is one kernel.  In `fwd` and `bwd`, 32 lanes own a row and a thread owns one row.  Each lane makes 9 loads
of 8 BF16 values of the gate and as many of the up (and of `dh` in the backward), at columns `(p * 32 + lane) * 8`, so
every load of a warp is contiguous.  In `bwd_unweighted`, a warp owns one (row, run) task: a run is a third of the row
(768 columns), and each lane makes that run's 3 loads of the gate, the up and `dh`, at the same columns.  The relay
program runs the same (row, run) tasks, a CTA of 3 warps per row: warp `r` owns run `r`.  In a fine program a task is
one load `p` of a row (256 columns), and each lane makes that load of the gate, the up (and `dh`), at the same columns.
The programs differ in their schedule:

| stage | program | architectures | CTA | a lane's 9 loads |
|---|---|---|---|---|
| `fwd` | small-R | `sm_100a`, `sm_103a`, `sm_107a` | 128 threads, 4 rows | 3 runs of 3; a run's loads are issued while the run before it computes |
| `fwd` | large-R | `sm_100a`, `sm_103a` | 256 threads, 8 rows | one run |
| `fwd` | large-R | `sm_107a` | 128 threads, 4 rows | one run; the values of each load are stored as soon as they are computed |
| `bwd` | small-R | `sm_100a`, `sm_103a`, `sm_107a` | 128 threads, 4 rows | 3 runs of 3; a run's loads are issued while the run before it computes |
| `bwd` | large-R | `sm_100a`, `sm_103a`, `sm_107a` | 128 threads, 4 rows | 3 runs of 3, one run at a time |
| `bwd` | relay | `sm_100a`, `sm_103a`, `sm_107a` | 96 threads (3 warps), 1 row | warp `r`: the 3 loads of run `r`; warp 0 then sums `d_weight` |
| `fwd` | fine | `sm_100a`, `sm_103a`, `sm_107a` | 128 threads, 4 (row, load) tasks | the task's load |
| `bwd` | fine | `sm_100a`, `sm_103a`, `sm_107a` | 288 threads (9 warps), 1 row | warp `p`: load `p`; warp 0 then sums `d_weight` |
| `bwd_unweighted` | small-R | `sm_100a`, `sm_103a`, `sm_107a` | 128 threads, 4 (row, run) tasks | the 3 loads of the task's run |
| `bwd_unweighted` | fine | `sm_100a`, `sm_103a`, `sm_107a` | 128 threads, 4 (row, load) tasks | the task's load |

More runs hold fewer registers, so more warps fit on an SM.  The grid is `ceil(R / rows a CTA)` CTAs, `ceil(3 R / 4)`
CTAs in the small-R `bwd_unweighted` program, `ceil(9 R / 4)` CTAs in `fwd_fine` and `bwd_unweighted_fine`, and `R` CTAs
in `bwd_fine` and `bwd_relay` (`cake_jit.grid`, from the launch block and `cake_jit.BUILD_OF`; the `bwd_unweighted`,
`fwd_fine` and `bwd_unweighted_fine` kernels count their tasks in 64 bits and stride over the blocks, and `bwd_relay`
counts its rows in 64 bits).  No routed launch has more than `2**31 - 1` CTAs (`cake_jit.MAX_GRID`): `cake_jit.fine`
refuses a fine window whose last row count needs more, and `cake_jit.grid` caps only a fine program launched at more
rows than its window holds.  `bwd_fine` counts its rows in 32 bits, so it takes at most `2**30` rows.  The route sends
it at most its window's `hi * SMs` rows: at most `64 * SMs`, and `2 * SMs` as shipped.  A row or a task past `R` loads
the last row and stores nothing.  The programs of a stage compute each element with the same FP32 operations (the
deep-gate sigmoid gives the fast one's bits wherever the fast one may be taken, so the loads a warp's vote covers change
no bit), and the `d_weight` sums take a lane's columns in load order with the same lanes.  In `bwd_fine`, warps 1 to 8
hand their `dh` words and activations to warp 0 in shared memory, and warp 0 runs each lane's `d_weight` chain over the
row's loads in load order with the same instructions; in `bwd_relay` warps 1 and 2 do the same for their runs.  So the
programs of a stage store the same bits for every input: the route changes no output bit.  `bwd_unweighted` votes once
per (row, run), the run of 768 columns that the `bwd` programs vote over, and computes each element with their
operations, taking `dh` where they take `dh w` with `w = 1` (the same value).  So it stores the `d_gate` and `d_up` bits
of the `bwd` programs with the weight flag 0, for every input; so does `bwd_unweighted_fine`, which votes once per (row,
load).  No program but `bwd_fine` and `bwd_relay` uses shared memory (their pools are static: 16 KB and 9 KB), and none
uses a workspace, a cooperative launch or programmatic dependent launch.  The row count, the limit, the weight flag
(`fwd`, `bwd`), the strides and the column offsets below are launch arguments, so one library per program and
architecture serves every call.

Each program's binding also takes the column offset of `up` (in `bwd` and `bwd_unweighted`, of `d_up` too), in BF16
elements, last before the grid, and adds it to that pointer.  A merged call passes its `[R, 4608]` tensor as both the
gate and the up with the offset 2304, and its gradient as both `d_gate` and `d_up`, so the host makes no view; a
two-tensor call passes 0.  The binding refuses an offset below 0, one that is not a multiple of 8 and one past
`size(1) - 2304` of a 2-D tensor (`ValueError`).  The kernels and their sources do not change with it.  0.1.5 and
earlier passed a view of the merged tensor's second half as the up (and `d_up`), with no offset.

## Locality domains (SM107)

An R200 has two dies, each with its own SMs and memory; default memory alternates between them every 4 KB.  With
`locality="domain"` (the steps and the autograd entry; `"off"`, the default, runs only the programs above), the forward
runs its twin, the program of the `fwd_domain` role (`sm_107a` only), when the call's operands lie in memory from
`cake_clamped_swiglu_det_train_domain_empty` (`cake_domain.py`).  That helper maps one virtual range of chunks, chunk
`k` in die `k & 1` (2 MB for `[R, 2304]`, 4 MB for `[R, 4608]`), so a row is 9 of the 4,096 slots of a chunk in either
width and tensors of one row count change die at the same rows.  The backward, weighted or unweighted, has no twin:
with `"domain"` it runs its `"off"` program (its plan's reason `arch`).

- Twin.  The forward's twin is the forward's program built with the claim loop and at least 5 resident CTAs per SM.
  It takes the forward's parameters, then the claim counters of both dies (`counters0`, `counters1`), the anchor's
  first slot and die, `chunk_shift`, `item_rows` and the SM table (by value: one int32 word per SM id with its die),
  then the column offset.  Its rows are items of `item_rows` (4) rows within one chunk; each CTA reads its die from
  the SM table and claims items of its own die's chunks first, then the other die's.  The grid is
  `min(items, ctas_per_sm * SMs)` CTAs (`cake_jit.domain_grid`); any grid gives the same bits.  The record's
  `grid_rule`, `counters` and `first_rows_per_sm` hold these facts; the row-count route and the fine window never name
  the twin.
- Route (`_domain_route`, in this order): `arch` (no twin of the stage on the device: every architecture but `sm_107a`,
  and both backward stages everywhere), `size` (fewer rows than the lower first row count of the twin's classes, or more
  than `2**27`), `sm_table` (no table on the device), `input_memory` and `input_layout` (the anchor, the merged input or
  the gate, not on a row start of a helper range), `overlap` (a caller's output buffer meets an input), then the class:
  `homed` when every BF16 operand lies in the anchor's row frame (its row 0 at most 8 slots after the anchor's), else
  `anchor`.  A class without a first row count names its first operand outside the frame: `input_phase`, `out_memory` or
  `no_out`.  Then `size` below the class's first row count, else `placed`.  The route reads the registry of helper
  ranges only: no driver call, no allocation, no launch.  Every fallback runs the `"off"` program with the same
  arguments.
- Outputs.  A new `h` is a `torch.empty` tensor (default memory), so a forward without `out` is of class `anchor`; an
  `out` from the helper keeps the class `homed`.  Autograd passes no buffer, so its forward is of class `anchor`.
- Bits.  The twin computes each row with the forward's instructions, so it stores the `"off"` bits for every input and
  every grid: the atomic claims choose only which CTA computes an item.
- State.  The helper ranges (outside torch's allocator: not in `torch.cuda.memory_*`; the last view's free synchronizes
  the device, deferred to the next helper call or `cake_clamped_swiglu_det_train_domain_release` during a capture),
  each device's SM table (probed by its first helper call through the package's JIT build of the locality probe), and
  the claim workspaces of `cake_domain.py`: one per stream for eager calls, one per capture and capture stream for CUDA
  graphs (a graph instantiated more than once shares its workspace: replay its copies one at a time), every launch
  leaving its counters zero.  The `"off"` path imports none of this.
- Plans.  `cake_clamped_swiglu_det_train_forward_domain_plan` and `..._backward_domain_plan` take their step's
  arguments without `locality` and return the route, reason, class, program, placement, grid, per-operand frame flags
  and whether each output is the caller's or new, without a launch.  The backward's plan names its `"off"` program
  with reason `arch`.
- Costs, measured on R200s of 212 SMs, where the first row counts are 23,744 rows (`homed`) and 27,136 (`anchor`).  The
  twin's route, claims and launch add 2.0 to 3.6 us of host time to an autograd step (forward and backward) and 9.0 to
  11.3 us to a checkpointed one (two forwards and a backward).  So an eager step whose host time exceeds its device time
  can take longer than with `"off"`: at 27,136 rows an autograd step took 2.1 to 4.5 % longer, and a checkpointed step
  4.8 to 6.6 % longer at 27,136 and 32,768 rows and up to 3.5 % longer at 49,152 rows with two input tensors (none was
  slower at 49,152 rows with a merged input, or at 65,536, 98,304 and 196,608 rows), while the device time of every
  measured step of the op from 27,136 rows on (CUDA graph replays, kernel timers) was 0.5 to 9.5 % shorter.  In a
  forward between the gate/up and the down projections (model width 2048) on inputs with a gate of -88 in each third of
  every row, the twin took 4.5 to 8.5 % longer than the `"off"` program on the same memory at 23,744 and 27,136 rows,
  and the whole chain 0.4 to 1.7 % longer, on the three GPUs measured whose dies have 106 SMs each (on other inputs, a
  CUDA graph replay of the chain at 27,136 rows took up to 0.5 % longer on one GPU).  At 98,304 rows such a chain took
  0.3 to 0.9 % longer on those GPUs than with `"off"` on default memory: the helper's memory makes that kernel 3 to 5 %
  slower there, and the twin wins back at most 1.7 %; on other inputs the chain took up to 1 % less time.  Below the
  first row counts a `"domain"` call runs the `"off"` program after its `locality` and size checks, with up to about 0.2
  us more host time per forward call and about 0.5 to 1 us per checkpointed autograd step; in eager timings this stayed
  within the noise but for one forward measurement, 0.5 % slower than `"off"` on default memory.

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
  `d_up = bf16((dh w) silu [-L <= u <= L])`, each mask a factor of 1 or 0, each output rounded once.  The unweighted
  backward takes `dh` for `dh w` (`w = 1`, the same value).
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
new output (`h`; `d_gate_up`, or `d_gate` and `d_up`; a weighted call's `d_weight`) and per copied input.  Output
buffers allocated once and passed to every call (`out=`, `grad_gate_up_or_gate=`, `grad_up=`, `grad_weight=`) avoid
the fills of those outputs.  The kernels write every element of every output, so a fill never reaches a result.

## Workspace and retained state

- Workspace: none with `locality="off"` (for `"domain"`, the claim workspaces above).  An unweighted backward allocates
  no weight gradient: its program takes no weight and writes no `d_weight`.  An unweighted forward passes, as the
  weight, one cached one-element FP32 buffer of the device, set to 1, which the process's first unweighted forward on
  that device makes (that call may be a CUDA graph capture).  The forward kernels read it and ignore it.
- Retained state of the autograd function: its inputs only (`gate` and `up` or the merged tensor, and `weight`),
  references with no copy: `R * 4608 * 2 + R * 4` bytes of tensors the caller already holds.  No activation is saved;
  the backward recomputes it.  So a checkpoint around the op alone saves no memory (see Usage).  The backward reuses
  the forward's FP32 `limit` and checks the saved inputs again, since a saved-tensors hook or a `.data` assignment can
  change them after the forward.
- Host caches, none of which changes an output bit: per device, its architecture, its SM count, its `torch.device`,
  the forward's twin and its first row counts (made at the first `"domain"` forward, its FFI entry bound at its first
  launch) and one launcher per stage (the route, each program's FFI entry bound at its first launch, and in `fwd` the
  ignored weight above); per process, the loaded libraries and the FP32 rounding of the last accepted `limit`.  The
  architecture is still checked at every call, before any allocation.
- Host path: a common call (plain contiguous operands and outputs of the exact dtypes and shapes on one device, the
  BF16 ones 16-byte aligned, `0 < R < 2**31 - 256`) is checked and launched by the step function itself, without the
  full checks (the autograd forward of a checkpoint's recomputation keeps that launch and runs it after `apply`
  returns, see Usage).  The step checks the first operand, the weight and `dh` in its own frame, and the `up` of a two-tensor
  input and each caller-provided output buffer with one helper call each.  So only a merged input with new outputs,
  whose limit is a Python float, or an int that a float holds exactly, equal to the float value of the last accepted
  limit, is checked and launched in exactly one Python frame per step once the host caches above are filled (any other
  limit adds one `_limit` call; a call that fills a host cache, such as the first launch of a program on a device,
  adds the calls that fill it).  The checks keep the order of the full checks: the operands and `dh`, then `limit`,
  then the output buffers.  The row strides of its contiguous tensors are constants (`4608` or `2304`; 0 for one row),
  not read.  Any other call takes the full checks, which raise every error and copy what the kernels cannot read; both
  give the same launch.  The facade binds the backend's step functions and autograd entries (`apply`; in a backward
  with saved-tensors hooks set, the held-launch entry) at their first call.
- Host path with `locality="domain"`: a common forward call is checked as above, in the same order, and routed in the
  step's own frame (a backward call with `"domain"` takes the `"off"` step after the `locality` check).  Its operands
  are contiguous, so each one's bytes and row pitch follow from its shape: the step reads each data pointer once and
  finds each placement with one `cake_domain.slot_of` call (the anchor's, then each other BF16 operand's until one lies
  outside the anchor's row frame), from what the registry keeps of each helper range.  It compares the caller's output
  buffers with the inputs only for a call whose class can take the twin, takes the claim counters as FFI tensors made
  once per workspace, and launches the twin or the `"off"` program itself.  The twin runs exactly when `_domain_route`
  places the call.  In a checkpoint's recomputation the step keeps that launch with the arguments it built (the grid
  and the SM table included); the held launch, after `apply` returns, claims the counters and calls the entry as the
  step does.  Any other call takes the full checks and `_domain_route`; both give the same launch.  A forward
  call with fewer rows than the first row count its class can reach before its placement is read (the `anchor`
  class's for a new `h`, as in the autograd forward; the lower of the two with a caller's `out`) launches the `"off"`
  program as the `"off"` step does, with no registry read, in both paths.

## Limitations

- The intermediate width is 2304 only (a compile-time width of the kernels).
- No double backward: differentiating gradients computed with `create_graph=True` raises `RuntimeError`.  Not tested
  under `torch.compile` or `torch.jit.trace`: a call that Dynamo or the tracer records launches its forward inside the
  autograd function's `forward`; a checkpoint's recomputation runs with Dynamo disabled.  No MXFP8 or FP4 output.
- A program's library builds at its first launch, and the row count picks the program (see Programs); in the backward so
  does the weight, as an unweighted backward runs its own programs.  Before capturing a CUDA graph, run an eager call of
  each kind the graph captures (weighted and unweighted) at each row count it captures, or build every library of the
  architecture with `specs_for_arch` (which leaves out a library whose `min_cuda` the local toolkit is below).
