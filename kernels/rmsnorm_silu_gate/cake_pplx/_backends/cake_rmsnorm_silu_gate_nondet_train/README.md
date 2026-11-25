# `cake_rmsnorm_silu_gate_nondet_train`: Cake SiLU-gated RMSNorm training backend, non-deterministic family (SM100 / SM103 / SM107)

Experimental backend of `RMSNorm(x) * weight * SiLU(z)` for training, on BF16
rows of 128 (`[T, H, 128]` or `[T * H, 128]`) with a weight shared by every
token and head.  It is the backend of the public module
`cake_pplx.cake_rmsnorm_silu_gate_nondet_train`
(`cake_pplx/cake_rmsnorm_silu_gate_nondet_train.py`); the deterministic
counterpart is `cake_pplx.cake_rmsnorm_silu_gate_det_train`, with the backend
`cake_pplx._backends.cake_rmsnorm_silu_gate_det_train`.

The public module is a stable API; this backend package is experimental (its
own names carry no compatibility guarantee).  It ships programs for `sm_100a`
(B200 / GB200), `sm_103a` (B300 / GB300) and `sm_107a` (R200), one
`-gencode=arch=compute_<arch>,code=sm_<arch>` build per program, minimum CUDA
12.8 (12.9 for some libraries) / 12.9 / 13.5 (`min_cuda` in `build.json`; on an
older toolkit `cake_build.jit_spec` raises and `cake_build.jit_specs` leaves
the library out).  The feature
does not take part in automatic backend selection, autotuning or trace-apply.
Its libraries are JIT-built by the package's JIT shim from `build.json`.

## What each output promises

`y`, `dx`, `dz` and an FP32 `dw` repeat bit for bit everywhere: the same
inputs on the same device give the same bits on every call, whatever the
order in which CTAs run and the work beside the call on other streams.  A
BF16 `dw` repeats bit for bit too, except in the fused band
(`bwd_bf16_fused`; see "Programs and routes"), where its last bits may vary
from run to run.  `y`, `dx` and `dz` are the same bits whatever the grid;
`dw` is too but in the repeatable bands (`bwd_bf16_rep` of a BF16 `dw`,
`bwd_f32_rep` of an FP32 `dw`, and their long programs), whose grid, and so
whose chunks of the `dw` sum, the rows and the SM count fix: there `dw` is
the same bits for the same rows and SM count, and a device with another SM
count may give other bits.  An FP32 `dw` of more than 512 rows runs a
repeatable band on every architecture, but on `sm_100a` and `sm_103a` past
64 and up to 128 rows per SM the static band (`bwd_f32`), whose `dw` is the
same bits whatever the grid.  In version 0.2.3 a call in an FP32
repeatable band changed bits from 0.2.2 (`dx`, `dz` and `dw`: they are now
the deterministic package's FP32 backward's bits in this tree), and its
`dw` is no longer the same bits whatever the grid, as it was in 0.2.2; in
the static band 0.2.3 runs 0.2.2's static program, with its bits (0.2.2 ran
it up to 256 rows per SM there).  Version 0.2.4 changes no bits: a call with
`compute_grad_weight=False` runs a program built without the weight
gradient, whose `dx` and `dz` are those of 0.2.3.  Version 0.2.5 changes no
bits: in the repeatable band (`sm_107a`) packed rows run a 128-thread build
of the strided program without the weight gradient (8-row blocks: twice the
CTAs of 16-row blocks at a small row count), other layouts the 0.2.4
256-thread one; this build and the fused band's strided program read packed
rows at their linear offsets, and a CTA of theirs with one block takes no
integer division before its loads.
Version 0.2.6 changes no kernel and no bits: it lowers the host time of a
call (each module's launch arguments and tvm-ffi function are looked up
once, the launch reads the handle of torch's current stream without making a
stream object, and a caller's workspace is checked and viewed once per
tensor); a launch still runs on torch's current stream of its tensors'
device, as in 0.2.5.
Version 0.2.7 changes nothing in this package and no bits: the deterministic
package's backward without the weight gradient runs its 128-thread strided
build on rows of any layout up to 32 rows per SM, and on packed rows up to
128 rows per SM a build of packed rows only (`sm_100a` and `sm_103a`), with
the same `dx` and `dz` bits.
Version 0.2.8 changes nothing in this package and no bits: on `sm_107a` the
deterministic package's backward without the weight gradient runs its
128-thread strided build on rows of any layout up to 96 rows per SM, and on
packed rows (and one token) up to 192 rows per SM a build of packed rows
only, with the same `dx` and `dz` bits.
Version 0.2.9 changes the `dx` and `dz` bits of some calls with
`compute_grad_weight=False` (no other call): on `sm_100a` and `sm_103a`
packed rows (and one token of head stride 128, which such a call passes as
packed rows) up to 128 rows per SM run this package's copy of the
deterministic package's build of packed rows only
(`bwd_bf16_rep_nodw_strided_packed_small`) in the place of the band's
program without the weight gradient, so their `dx` and `dz` are the
repeatable backward's bits; they were the bits of the call that computes a
BF16 `dw`, from which they now differ in their last bits in the one-cluster
bands and, on `sm_103a`, in the fused band.  Every other call, and every
call on `sm_107a`, runs the program and gives the bits of 0.2.8.
Version 0.2.10 changes nothing in this package and no bits: the
deterministic package's backward that computes the weight gradient runs, up
to 16 rows, a program of one CTA and, from 17 to 256 rows on a device of at
least 16 SMs that is not a MIG instance, a program of one cluster of 16
CTAs, neither of which uses a workspace, with the same `dx`, `dz` and `dw`
bits on a device of at least 4 SMs.
Version 0.2.11 changes nothing in this package and no bits: the
deterministic package's forward runs, on short inputs whose rows are
packed (contiguous `x` and `z`, or one token of head stride 128) up to a cut
of rows per SM (on SM107 not from 17 to 32 rows), a program of one row block
per CTA with no loop, with the same `y` bits.
Version 0.2.12 keeps the `y` bits of every call that 0.2.11 ran correctly:
on `sm_100a` and `sm_103a` the forward of packed rows (contiguous `x` and
`z`, or one token of head stride 128) up to 4 rows per SM on `sm_100a` and
12 on `sm_103a` runs a 32-lane build of the tiny forward (`fwd_l32_tiny`:
two lanes to each group of 8 elements, one 4-row block per 128-thread CTA)
with the tiny forward's sum of squares, so its `y` bits; the tiny forward
runs from there up to 36 rows per SM (past these cuts, with the rows in L2,
the 32-lane build's twice as many CTAs make it slower than the tiny
forward), and `sm_107a` runs the forward of 0.2.11.
A forward of `x` or `z` broadcast over more than one
token (token stride 0) runs the band of other layouts
(`STRIDED_FORWARD_BANDS`) where the band of packed rows runs a program that
reads packed rows at their linear offsets: 0.2.11 refused such a call in
the tiny forward's band (up to 36 rows per SM on `sm_100a` and `sm_103a` and
65 on `sm_107a`: the tiny forward takes contiguous `x` and `z` only, so a
`ValueError`), and on `sm_107a` past 280 rows per SM its pipelined forward
of packed rows read rows past the broadcast token.  At every other row
count such a call runs, as in 0.2.11, the program of packed rows, which
reads the token stride.
Version 0.2.14 changes no program and no bits, only a grid: the
deterministic package's forward and this package's pipelined forwards
(`fwd_pipe`, `fwd_pipe_packed`), which keep that forward's grid rule, launch
whole waves of CTAs from a cut of rows per SM of each architecture instead
of from 8192 rows per SM everywhere, one CTA per 16-row block below it.  On
rows the kernel reads as packed (`x` and `z` of head stride 128 and token
stride `H * 128`, the rows that get its L2 prefetches; the records'
`persistent_from_rows_per_sm`, now keyed by architecture) the cut is 5600
on `sm_103a` and, unchanged, 8192 on `sm_100a` and `sm_107a` (on B200 the
whole waves of packed rows were up to about 1 % slower per call with a cold
L2 from 6500 to 8000 rows per SM; `fwd_pipe` on `sm_100a`, whose bands of
packed rows ended at 6500 rows per SM until version 0.2.17, ran them one
CTA a block as in 0.2.13); on other rows of a program that reads any layout (strided,
broadcast over tokens or one token; a new
`persistent_from_rows_per_sm_strided`: the deterministic forward of rows of
any layout, and `fwd_pipe`), which run without the prefetches and gain from
whole waves from far fewer rows per SM, it is 500 on `sm_100a` and 400 on
`sm_103a` and `sm_107a`.  `y` is the same bits on either grid, and every
band is unchanged.
`cake_jit.launch_grid` of both packages takes the architecture (keyword
`arch`), which the records of those forwards need (without it, a
`ValueError`), and whether the kernel reads the rows as packed (keyword
`packed`, `cake_jit.forward_packed`); `run_module` computes the same grid
for the device's architecture and the call's own test, with the fewest host
operations.
Version 0.2.15 changes no bits.  The deterministic package's forward warms
in L2 the CTA's next block and the first block of the CTA a wave later, `x`
and `z` each by its own layout: packed rows as one byte range, rows of head
stride 128 and another token stride (a gate sliced from a projection, a
broadcast over tokens, one token) as one range a token, and rows of other
layouts (head-major) as one 256-byte range a row on `sm_103a` and `sm_107a`
and none on `sm_100a`; 0.2.14 warmed `x` and `z` only where both held packed
rows.  Its program names and grids are those of 0.2.14.  This package's
pipelined forwards (`fwd_pipe`, `fwd_pipe_packed`) keep the programs of
0.2.14, which warm `x` and `z` only where both hold packed rows.  Rows of other
layouts of a BF16 `dw` run the repeatable backward's long program
`bwd_bf16_rep_long` in its place from
`cake_jit.LONG_BACKWARD_STRIDED_ROWS_PER_SM` rows per SM (6145 on `sm_100a`
and `sm_103a`, 4097 on `sm_107a`), with the same bits; an FP32 `dw` and a
call with `compute_grad_weight=False` run the long programs on packed rows
only, as in 0.2.14.  `cake_jit.module_for_rows` of both packages and the
deterministic package's `cake_jit.backward_module` take
`compute_grad_weight` (default true).
Version 0.2.16 changes no bits.  On `sm_100a` and `sm_103a` the
deterministic package runs packed rows (`x` and `z` both contiguous
`[T, H, 128]`, fewer than 2^24 rows) outside its tiny forward's rows
(`cake_jit.FWD_TINY_ROWS_PER_SM`) on a second forward program
(`fwd_packed`), the forward of 0.2.14, which warms in L2 only rows packed in
both `x` and `z`, up to `cake_jit.FWD_PACKED_ROWS_PER_SM` rows per SM; its
`fwd` program, which warms each tensor by its own layout, runs the other rows
outside the tiny forward's.  This package's
programs, routes and grids are those of 0.2.15.
Version 0.2.17 changes bits only where it changes a program.  A BF16 `dw` of
up to 16 rows runs the solo backward `bwd_bf16_solo` (every architecture):
the repeatable backward built for one CTA, the deterministic package's solo
program under this package's name, so `dx`, `dz` and `dw` are the
repeatable backward's bits there (0.2.16 ran the short backward); a call with
`compute_grad_weight=False` keeps the programs and bits of 0.2.16.  On
`sm_100a` packed rows past 6500 rows per SM run the pipelined forward
`fwd_pipe` (0.2.16 ran the mid forward), whose `y` can differ in its last
bits (the programs associate a row's sum of squares differently).  On
`sm_107a` packed rows of 17 to 32 rows run the short forward `fwd_short`
(`cake_jit.FORWARD_TINY_GAP`; 0.2.16 ran the tiny forward), with the same
bits.  Four forward roles are new (six programs), builds of the pipelined
forwards that change no value (each gives the bits of `fwd_pipe` on
`sm_100a` and `sm_103a`, of `fwd_pipe_packed` on `sm_107a`):
`fwd_keep_pipe` and `fwd_keep_pipe_packed` load `x` and `z` with the L2's
default policy in place of `evict_first`, `fwd_layout_pipe` warms rows of any
layout in L2 by their layout (the deterministic package's per-layout prefetch
walk), and `fwd_layout_keep_pipe` does both (`fwd_keep_pipe` and
`fwd_layout_keep_pipe` on `sm_100a` and `sm_107a` only).  Packed rows run
`fwd_keep_pipe` on `sm_100a` from 37 to 85 rows per SM (an `x` or `z`
broadcast over more than one token keeps `fwd_pipe`) and
`fwd_keep_pipe_packed` on `sm_107a` past 8192, where 0.2.16 ran `fwd_pipe`
and `fwd_pipe_packed`, the same bits.  Rows that are not packed are classed
(`cake_jit.forward_layout`): a gate of more than one head sliced from a wider
projection with packed `x` runs `cake_jit.PROJECTION_FORWARD_BANDS` and
head-major `x` and `z` on `sm_107a` `cake_jit.HEAD_MAJOR_FORWARD_BANDS`,
whose bands run `fwd_layout_pipe` and `fwd_layout_keep_pipe` (head-major rows
`fwd_layout_keep_pipe` only, up to 100 rows per SM) where 0.2.16 ran the
forwards of `cake_jit.STRIDED_FORWARD_BANDS`, so `y` can differ there in its
last bits; other layouts, such a gate of one head among them, keep the
programs of 0.2.16.
`cake_jit.module_for_rows` takes the class (keyword `layout`).  The
deterministic package is that of 0.2.16.
A band runs a program whose `dw` may vary
only where the band-selection measurement found the checkpointed training
step (2 x forward + backward) about 3 % or more faster than with the
repeatable program (near a band's edges the gain can be smaller); every
other band repeats its bits:

| architecture | fused band of a BF16 `dw` (may vary) | repeatable band of a BF16 `dw` (repeats) |
| --- | --- | --- |
| `sm_100a` | past 512 rows, up to 1000 rows per SM | past 1000 rows per SM |
| `sm_103a` | past 256 rows, up to 1000 rows per SM | past 1000 rows per SM |
| `sm_107a` | past 64 rows per SM, up to 256 rows per SM | past 512 rows, up to 64 rows per SM; past 256 rows per SM |

The fused band runs the fused backward (its build `{"math": "ax",
"min_blocks": 4, "prefetch_share": None, "slots": 8}`) for a BF16 `dw` (a
call with `compute_grad_weight=False` runs its programs built without the
weight gradient, below): the deterministic package's fused backward's
schedule and row arithmetic (the approximate gate and `rstd`, the
next-block prefetch up to the architecture's share), so its `dx` and `dz`
are the same bits at any grid (a row's sums run on FMA chains of its own,
set per architecture, so their last bits may differ from that package's),
with the row-claims
backward's kind of weight gradient: FP32 runs of rows added into plain FP32
words, each CTA's FP32 column sums added into 8 FP32 replicas with FP32
atomic adds and the replicas summed in FP32 by the CTA that arrives last (no
FP64), so the order of the sum, and the last bits of `dw`, depend on
scheduling.  An FP32 `dw` runs no fused backward.  The repeatable band of
a BF16 `dw` runs the repeatable backward, the deterministic package's BF16
backward program under this package's name (its build `{"math": "ax",
"min_blocks": 4, "out": "bf16", "prefetch_share": None, "threads": 256}`):
the fused backward's schedule and row arithmetic, the CTAs claiming the
blocks in chunks whose rows the row count and the grid fix, each thread's
FP32 runs of at most 16 rows of a chunk added into plain FP32 words and each
CTA adding a column's FP32 word exactly into 64-bit fixed-point digits, so its `dx`,
`dz` and `dw` are the deterministic package's bits in this tree, the same
for the rows and the SM count.  An FP32 `dw` past its small-row band (512
rows, every architecture) runs the repeatable backward of an FP32 weight
gradient, `bwd_f32_rep`, the deterministic package's FP32 backward program
under this package's name (its build `{"math": "ax", "min_blocks": 4,
"out": "f32", "prefetch_share": None, "threads": 256}`): the same schedule,
chunks, FP32 runs and fixed-point digits, the exact sum rounded once to
FP32, so its `dx`, `dz` and `dw` are the deterministic package's FP32 bits
in this tree, the same for the rows and the SM count; on `sm_100a` and
`sm_103a` past 64 and up to 128 rows per SM it runs the static backward
`bwd_f32` instead, whose CTA `b` runs the chunks `b`, `b + grid`, ... of the
deterministic backward at commit 2e989556 and whose `dw` is the chunks' FP32
partials summed exactly in integer windows and rounded once to FP32, the
same bits at any grid (0.2.2's program and bits there).  In either
repeatable band a column whose terms are finite but whose FP32 run or chunk
word overflows gets an infinity (or a NaN) in `dw` where the exact sum stays
finite.  Packed rows of a repeatable band (`dy`, `x` and `z` each with head
stride 128 and token stride `H * 128`) run its long program instead,
`bwd_bf16_rep_long` or `bwd_f32_rep_long`, from 2560
rows per SM on `sm_100a` and `sm_103a` and from 1536 on `sm_107a`, and rows
of other layouts of a BF16 `dw` run `bwd_bf16_rep_long` from 6145 rows per
SM on `sm_100a` and `sm_103a` and from 4097 on `sm_107a` (one row per SM
past the default loop's L2 prefetch share of blocks per CTA times 64, up to
which the default loop warms each next block in L2; on a gate and `dy`
sliced from projections the long program is 4.2 % .. 5.3 % faster from 7000
rows per SM on B200 and GB300 and 3.5 % .. 4.0 % from 4500 on VR200, and not
faster at the share or below it, 0.1 % .. 0.5 % slower at 6144 on GB300):
the deterministic package's look-ahead program (its build
`{"lookahead": True, "math": "ax", "min_blocks": 4, "out": "bf16",
"prefetch_share": None, "threads": 256}`, with `"out": "f32"` for an FP32
`dw`), the same grid, workspace and bits.  The cuts live in one place,
`cake_jit.BACKWARD_BANDS` (the long programs' in
`cake_jit.LONG_BACKWARD_ROWS_PER_SM` and
`cake_jit.LONG_BACKWARD_STRIDED_ROWS_PER_SM`).  The row-claims programs
(`bwd_bf16_rc`, `bwd_bf16_rc_mid`; `sm_100a`, `sm_103a`), whose `dw` may vary
from run to run as well, and the BF16 static and claimed programs ship
routed, but no band runs them; no claimed program of an FP32 `dw` ships,
nor a static one on `sm_107a`.

The BF16 bands up to 512 rows follow measurements of a checkpointed training
step (a CUPTI kernel timer, cold L2), taken on version 0.1.0 of this package,
whose row-claims, fused and small-row backwards summed the weight gradient in
FP64.  On GB300 (`sm_103a`) the step
with the short backward (`bwd_bf16_short`) is 7.8 % .. 14.8 % faster than
with the faster of the exact tiny backward and the row-claims backward at
48 .. 256 rows (13.1 % .. 18.7 % faster than with the exact tiny backward),
and in version 0.1.0 the row-claims backward (which no band runs in this
version) saved 6.9 % .. 13.1 % of the step against the exact tiny backward
at 336 .. 512 rows, so `sm_103a` runs no BF16 exact tiny backward and its
next band starts at 257 rows.  That band now runs the fused backward, which
has not been measured against the exact tiny backward at 257 .. 512 rows.  On
B200 (`sm_100a`) the steps with the short and with the exact tiny backward
are within 3 % of each other at up to 256 rows, and under that
benchmark timer the short backward alone is the faster of the two at every
row count up to 256.  On R200 (`sm_107a`) the step with the exact tiny
backward takes 2.9 % .. 6.4 % longer than with the short backward at
96 .. 256 rows (the mean of 4 passes; longer in every pass) and 1.1 % less
at 48 rows, within that row count's pass-to-pass spread; at 336 .. 512 rows
the step with the row-claims backward takes 18.7 % .. 24.5 % longer than
with the exact tiny backward, so `sm_107a` runs the exact tiny backward
from 257 to 512 rows.

Every forward program takes the deterministic forward's split gate (the
reciprocal of a scaled `1 + e`: a nonzero sigmoid down to `z` about -88.72,
where `ex2.approx` overflows).  `y` is the deterministic package's bits only
from the pipelined forwards: they are the deterministic package's forward
programs with that gate, which does not flush to 0 where that package's gate
does (`z` from about -87.34 down), and they give that package's bits
elsewhere.  Each backward program repeats `dx` and `dz` bit for bit for the
same inputs on the same device; the programs' `.f32x2` FMA chains for a
row's sums differ by program and by architecture, so the last bits of `dx`
and `dz` can differ between the backward bands.  The deterministic package's
backward need not give these bits, but in the solo and the repeatable bands
`dx` and `dz` are its bits (of its BF16 or its FP32 program).  In the
small-row bands but the solo band `dw` is that program's own fixed-order FP32
sum: the same bits on every call, which need not be the deterministic
package's; in the solo band it is the deterministic package's bits in this
tree.  In the
fused band `dw` is that program's own FP32 sum of FP32 replicas, in an order
that depends on scheduling; in the repeatable bands it is the deterministic
package's bits in this tree.  In the static band of an FP32 `dw` (`sm_100a`,
`sm_103a`) `dw` is the exact sum of each chunk's FP32 partial (the chunks of
rows of the deterministic backward at commit 2e989556), rounded once in
integer arithmetic, and so it is in the static and the claimed programs of
a BF16 `dw`, which no band runs.  No program sums in FP64.

| output | how it is computed | run to run |
| --- | --- | --- |
| `y` | `rstd` with `rsqrt.approx`, the sigmoid with `ex2.approx` and `rcp.approx` (the split gate), FP32 products, one rounding to BF16 | bit for bit |
| `dx` | each backward program's FP32 row arithmetic, its `.f32x2` FMA chains for a row's sums set per program and per architecture (the last bits can differ between bands); the fused and the repeatable band run the deterministic package's row arithmetic (the fused band on FMA chains of its own) | bit for bit |
| `dz` | each backward program's FP32 row arithmetic, its `.f32x2` FMA chains for a row's sums set per program and per architecture (the last bits can differ between bands); the fused and the repeatable band run the deterministic package's row arithmetic (the fused band on FMA chains of its own) | bit for bit |
| `dw` (the static and the claimed backwards: an FP32 `dw` in the static band of `sm_100a` and `sm_103a`) | each chunk's FP32 partial (the chunks of rows of the deterministic backward at 2e989556), summed exactly in 64-bit integer windows, the exact sum rounded once to FP32 in integer arithmetic (then to BF16 for a BF16 `dw`); a column with an infinite or NaN term has the class of the exact sum (`+inf`, `-inf` or `NaN`), also when the row count is not a multiple of the row block | bit for bit |
| `dw` (the exact tiny backward's bands) | the program's FP32 weight-gradient terms, summed on chip in plain FP32 words in an order of its own, rounded once to the output dtype | bit for bit |
| `dw` (BF16 in the solo backward's band: up to 16 rows) | the deterministic package's solo program (its BF16 backward in one CTA): each thread's FP32 runs of the call's rows added into plain FP32 words, the CTA adding a column's FP32 word exactly into its own 64-bit fixed-point digits and rounding each column's exact sum once to FP32, then to BF16, so the repeatable backward's bits | bit for bit |
| `dw` (BF16 in the short backward's band) | each CTA's FP32 partial of its 16-row blocks (the row-claims backward's row arithmetic); CTA `r` adds every CTA's partials of the columns `8r` .. `8r + 7` in rank order in FP32 and rounds the sum to BF16 | bit for bit |
| `dw` (BF16 in the fused band) | each thread's FP32 runs of at most 16 rows added into plain FP32 words; each CTA adds its FP32 column sums into 8 FP32 replicas with FP32 atomic adds in an order that depends on scheduling; the CTA that arrives last sums the replicas in FP32 and rounds the sum to BF16 | its last bits may vary |
| `dw` (the repeatable bands: BF16 past the fused band, on `sm_107a` also before it; FP32 past 512 rows but in the static band) | the deterministic package's backward in this tree (its BF16 or its FP32 program): each thread's FP32 runs of at most 16 rows of a chunk added into plain FP32 words of the chunk, each CTA adding a column's FP32 word exactly into 64-bit fixed-point digits in the workspace; the CTA that arrives last rounds each column's exact sum once to FP32 (then to BF16 for a BF16 `dw`); an infinite or NaN FP32 word sets a flag of its column instead, and the column is `NaN` where a word is `NaN` or both infinities occur, else that infinity, so a column whose terms are finite but whose FP32 run or chunk word overflows gets an infinity (or a `NaN`) where the exact sum stays finite | bit for bit (the rows and the SM count fix the chunks) |

Accuracy: the tests hold every floating output of a training call to the PyTorch BF16
naive implementation of the same operation (every op in BF16, the gradients
from PyTorch autograd; a weight gradient accumulated onto a starting value is
the BF16 sum of the two), each against a float64 reference computed from the
BF16 inputs.  Over 32 random inputs of each case, in the L2 and in the max
norm of the error over the elements where the reference is finite, the
median and the largest error of the output are each at most 1.005 times the
naive implementation's, and the output's error is 0 on every input where the
naive implementation's is 0 (`tests/cake/test_helpers/cake_naive.py`, the
same file in every family that uses it).  An FP32 output is held to the BF16
naive value widened to FP32.  Every output is also finite where the reference
is finite and holds the reference's infinity or a NaN where it holds one.
No output has an error budget of its own, and the tests compare no output of
this package with the deterministic package's.

Under `torch.use_deterministic_algorithms(True)` a backward call that
computes a BF16 `dw` in the fused band (not in the repeatable band; through
`cake_rmsnorm_silu_gate_nondet_train_backward` or the autograd node's
backward) raises PyTorch's alert for an operation without a deterministic
implementation, which names the deterministic package's entry point: a
`RuntimeError`, or with `warn_only=True` a `UserWarning`, after which the call
runs.  The alert comes after the argument checks, immediately before the
launch; the call never falls back to another program.  Nothing else alerts:
the other bands, a call with `compute_grad_weight=False` (it takes the BF16
bands), a call with no rows, the forward, the workspace entry points and the
support query.  The check runs on the host, so a CUDA Graph checks it when it
is captured, not when it is replayed.

## Public entry points (`cake_pplx/cake_rmsnorm_silu_gate_nondet_train.py`)

```python
from cake_pplx.cake_rmsnorm_silu_gate_nondet_train import (
    cake_rmsnorm_silu_gate_nondet,                                  # autograd
    cake_rmsnorm_silu_gate_nondet_train_forward,
    cake_rmsnorm_silu_gate_nondet_train_backward,
    cake_rmsnorm_silu_gate_nondet_train_backward_workspace,
    cake_rmsnorm_silu_gate_nondet_train_backward_workspace_bytes,
    cake_rmsnorm_silu_gate_nondet_train_reset_workspace,
)

y = cake_rmsnorm_silu_gate_nondet(x, z, weight, eps)
dx, dz, dw = cake_rmsnorm_silu_gate_nondet_train_backward(dy, x, z, weight, eps)
ws = cake_rmsnorm_silu_gate_nondet_train_backward_workspace(T * H, x.device)
cake_rmsnorm_silu_gate_nondet_train_backward(
    dy, x, z, weight, eps, grad_weight=main_grad, accumulate_grad_weight=True,
    workspace=ws,
)
```

The mode is part of every name: no entry point takes a `deterministic`
keyword (passing one is a `TypeError`), and no call runs the deterministic
package.  This package exports the same names.

A call with no rows (`T * H == 0`) checks the last dimension (128), `weight`,
the given outputs and, in the backward, `grad_weight` and the refusal of an
FP32 workspace, then returns without a launch; it does not check the dtype,
device or layout of `x`, `z` and the upstream gradient or the size of a
`torch.uint8` workspace (the deterministic package checks a call with no
rows the same way).

## Programs and routes

`cake_jit.py` routes each call by its row count `rows = T * H`, the device's
SM count and, for the forward, whether `x` and `z` hold packed rows and, if
not, the class of their layout (`forward_layout`).  Each
program's arithmetic is fixed; every forward program takes the deterministic
forward's split gate.  The backward
programs' `.f32x2` FMA chains for a row's sums differ by program and by
architecture, so `dx`'s and `dz`'s last bits can differ between the backward
bands; each program repeats them bit for bit for the same inputs on the same
device.
The forward programs associate a row's sum of squares differently, on FP32
FMA chains (the lanes of `.f32x2` FMA chains where a program runs two or
four): the tiny and the short forward run four 2-term chains of 16 lanes
added in pairs, then folded by XOR shuffles 1, 2, 4, 8, and the 32-lane tiny
forward the same chains on its 16 lane pairs (both lanes of a pair sum the
same 8 elements), then folds them by XOR shuffles 2, 4, 8, 16: the same
additions, so the tiny forward's bits; the mid forward, the
extra-long forward but on `sm_100a` and the pipelined forwards on `sm_107a`
four 4-term chains of 8 lanes added in pairs, then folded by 1, 2, 4; the
plain forward and the pipelined forwards on `sm_100a` and `sm_103a` two
8-term chains of 8 lanes (the even and the odd elements) added, then folded
by 1, 2, 4; the long forward, and the extra-long forward on `sm_100a`,
16-term chains of 8 lanes folded by 1, 2, 4.  So `y`'s bits can differ
between the programs, and so between the forward bands of one architecture;
each program repeats `y` bit for bit for the same inputs on the same device.

| role | program | runs |
| --- | --- | --- |
| `fwd_tiny` | forward of packed rows, 128 threads, 16 lanes a row, one CTA per 8-row block (not persistent), the short forward's arithmetic and bits | its band of `FORWARD_BANDS` (packed rows), but not the rows of `FORWARD_TINY_GAP` |
| `fwd_l32_tiny` (`sm_100a`, `sm_103a`) | the tiny forward with 32 lanes a row (two lanes to each group of 8 elements, each lane the outputs of its 4), 128 threads, one CTA per 4-row block (not persistent), the tiny forward's sum of squares, so the short forward's bits | its band of `FORWARD_BANDS` (packed rows) |
| `fwd_short` | forward, 256 threads, 16 lanes a row, one CTA per 16-row block (not persistent) | its bands of `PROJECTION_FORWARD_BANDS`, `HEAD_MAJOR_FORWARD_BANDS` and `STRIDED_FORWARD_BANDS` (rows that are not packed); on `sm_107a` packed rows of 17 to 32 rows (`FORWARD_TINY_GAP`) |
| `fwd` | plain forward, 64 threads, 8 lanes a row, one CTA per 8-row block (not persistent) | the call role; its bands of `PROJECTION_FORWARD_BANDS`, `HEAD_MAJOR_FORWARD_BANDS` and `STRIDED_FORWARD_BANDS` (rows that are not packed, past the persistent forwards' bands) |
| `fwd_mid` | whole waves of 6 CTAs per SM (as many waves as the rows need), each CTA at most 2 blocks of 16 rows | its forward bands |
| `fwd_long` | whole waves of 3 CTAs per SM (as many waves as the rows need), each CTA at most 6 blocks of 32 rows | its forward bands (`sm_100a`) |
| `fwd_xlong` | the long forward's grid, also prefetching the next block with one bulk prefetch | its forward bands |
| `fwd_pipe` (`sm_100a`, `sm_103a`) | pipelined forward, the program of 0.2.14: a build of the deterministic package's forward kernel (the same grid) with a gate that does not flush (`split_gate`), which warms in L2 only rows packed in both `x` and `z`: 128 threads, 8 lanes a row, one CTA per 16-row block below a cut of rows per SM (the deterministic package's): on rows the kernel reads as packed (`x` and `z` of head stride 128 and token stride `H * 128`, which get its L2 prefetches; `persistent_from_rows_per_sm`) 8192 on `sm_100a` and 5600 on `sm_103a`, on other rows (`x` or `z` broadcast over tokens, or one token; `persistent_from_rows_per_sm_strided`) 500 on `sm_100a` and 400 on `sm_103a`; from there whole waves of 6 CTAs per SM, each CTA at most 2 blocks, a thread loading its row of the CTA's next block while it computes the current one, `x` and `z` loaded with the L2 `evict_first` policy | its forward bands |
| `fwd_pipe_packed` (`sm_107a`) | the pipelined forward of packed rows (the deterministic package's packed-rows program on `sm_107a` with a gate that does not flush: the pipelined forward's schedule with four sum chains, `rsqrt.approx.ftz` and one 32-byte store a lane; one CTA per 16-row block below 8192 rows per SM, whole waves from there) | its band of `FORWARD_BANDS` (packed rows, fewer than `PACKED_ROWS`) |
| `fwd_keep_pipe` (`sm_100a`) | `fwd_pipe` with `x` and `z` loaded with the L2's default policy in place of `evict_first` (its build and `"xz_evict_first": 0`): its grid, cuts and L2 prefetches, so its bits | its band of `FORWARD_BANDS` (packed rows; an `x` or `z` broadcast over more than one token runs `fwd_pipe` there) |
| `fwd_keep_pipe_packed` (`sm_107a`) | `fwd_pipe_packed` with `x` and `z` loaded with the L2's default policy in place of `evict_first` (its build and `"xz_evict_first": 0`): its grid, so its bits | its band of `FORWARD_BANDS` (packed rows past 8192 rows per SM, fewer than `PACKED_ROWS`) |
| `fwd_layout_pipe` | the pipelined forward of rows of any layout with the deterministic package's per-layout L2 prefetch walk (`"split_layout_prefetch": 1`): on `sm_100a` and `sm_103a` `fwd_pipe`'s build, on `sm_107a` the deterministic package's forward of rows of any layout (four sum chains, `rsqrt.approx.ftz`), each with the gate that does not flush; it warms the CTA's next block and the first block of the CTA a wave later in L2, `x` and `z` each by its own layout (as the deterministic package's forward of 0.2.15), `x` and `z` loaded with the L2 `evict_first` policy; `fwd_pipe`'s cuts (on rows the kernel reads as packed 8192 on `sm_100a` and `sm_107a` and 5600 on `sm_103a`, on other rows 500 on `sm_100a` and 400 on `sm_103a` and `sm_107a`); the bits of `fwd_pipe`, on `sm_107a` of `fwd_pipe_packed` | its bands of `PROJECTION_FORWARD_BANDS` |
| `fwd_layout_keep_pipe` (`sm_100a`, `sm_107a`) | `fwd_layout_pipe` with `x` and `z` loaded with the L2's default policy in place of `evict_first` (its build and `"xz_evict_first": 0`): its grid, cuts and L2 prefetches, so its bits | its bands of `PROJECTION_FORWARD_BANDS` and `HEAD_MAJOR_FORWARD_BANDS` |
| `bwd_bf16_solo` | solo backward (a BF16 weight gradient; every architecture): the repeatable backward `bwd_bf16_rep` built for one CTA (its build with `"cluster": 1`), the deterministic package's solo program under this package's name: one CTA of 256 threads, 16 lanes a row, that runs the call's one 16-row block as the repeatable backward runs a block and adds its FP32 column words exactly into its own fixed-point digits, rounding each column's sum once; no ticket, no claim and no workspace word (its workspace parameters unread); `bwd_bf16_rep`'s `dx`, `dz` and `dw` bits | the rows of its band (at most 16) of a call that computes the weight gradient |
| `bwd_bf16_short` | short backward (a BF16 weight gradient; every architecture): one cluster of 16 CTAs of 256 threads, 16 lanes a row, CTA `r` running the 16-row blocks `r`, `r + 16`, ... with the row-claims backward's row arithmetic; CTA `r` adds every CTA's FP32 partials of the columns `8r` .. `8r + 7` in rank order in FP32 (a fixed order), no workspace | the rows of its band (at most 256) |
| `bwd_bf16_t512` (`sm_100a`, `sm_107a`), `bwd_f32_t512` | exact tiny backward: one cluster of 16 CTAs of 4 teams of 128 threads, its FP32 weight-gradient terms summed on chip in plain FP32 words in an order of its own, no workspace | the rows of its band (at most 512) |
| `bwd_bf16_fused` | fused backward (a BF16 weight gradient; every architecture): the deterministic package's fused backward's schedule and row arithmetic (the approximate gate and `rstd`, the next-block L2 prefetch up to the architecture's share of blocks a CTA): 256 threads, 16 lanes a row, 16-row blocks, at most 4 CTAs per SM, which claim the blocks in chunks when there are more blocks than CTAs (`dx` and `dz` the same bits at any grid); each thread's FP32 runs of at most 16 rows added into plain FP32 words, each CTA's FP32 column sums added into 8 FP32 replicas with FP32 atomic adds, the CTA that arrives last summing the replicas in FP32 and rounding the sum once | its backward band, up to 1000 rows per SM (`sm_107a`: past 64, up to 256 rows per SM) |
| `bwd_bf16_rep` | repeatable backward (a BF16 weight gradient; every architecture): the deterministic package's backward program (its build `{"math": "ax", "min_blocks": 4, "out": "bf16", "prefetch_share": None, "threads": 256}`) under this package's name: 256 threads, 16 lanes a row, 16-row blocks, at most 4 CTAs per SM claiming the blocks in chunks whose rows are fixed by the row count and the grid; each thread's FP32 runs of at most 16 rows of a chunk added into plain FP32 words, each CTA adding a column's FP32 word exactly into 64-bit fixed-point digits in the workspace, the CTA that arrives last rounding the sum once (the deterministic package's `dx`, `dz` and `dw` bits, the same for the rows and the SM count) | its backward bands, past 1000 rows per SM (`sm_107a`: past 512 rows, up to 64 rows per SM, and past 256 rows per SM) |
| `bwd_bf16_rep_long` | the repeatable backward's long program (every architecture): the deterministic package's look-ahead program (its build `{"lookahead": True, "math": "ax", "min_blocks": 4, "out": "bf16", "prefetch_share": None, "threads": 256}`) under this package's name: the repeatable backward in a loop that loads a block's rows while the previous block's row math runs, with its grid, block schedule, FP32 operations in the same order and workspace, so its bits | packed rows of the last repeatable band from `LONG_BACKWARD_ROWS_PER_SM` rows per SM (2560 on `sm_100a` and `sm_103a`, 1536 on `sm_107a`); rows of other layouts of a call that computes the weight gradient from `LONG_BACKWARD_STRIDED_ROWS_PER_SM` rows per SM (6145 on `sm_100a` and `sm_103a`, 4097 on `sm_107a`) |
| `bwd_f32_rep` | repeatable backward of an FP32 weight gradient (every architecture): the deterministic package's FP32 backward program (its build `{"math": "ax", "min_blocks": 4, "out": "f32", "prefetch_share": None, "threads": 256}`) under this package's name: the schedule, grid, chunks, FP32 runs, fixed-point digits and workspace of `bwd_bf16_rep`, the CTA that arrives last rounding the sum once to FP32 (the deterministic package's FP32 `dx`, `dz` and `dw` bits, the same for the rows and the SM count) | its backward bands, past 512 rows (`sm_100a`, `sm_103a`: up to 64 rows per SM and past 128 rows per SM) |
| `bwd_f32_rep_long` | the FP32 repeatable backward's long program (every architecture): the deterministic package's FP32 look-ahead program (its build `{"lookahead": True, "math": "ax", "min_blocks": 4, "out": "f32", "prefetch_share": None, "threads": 256}`) under this package's name, so `bwd_f32_rep`'s grid, workspace and bits | packed rows only, of its last band from `LONG_BACKWARD_ROWS_PER_SM` rows per SM (2560 on `sm_100a` and `sm_103a`, 1536 on `sm_107a`) |
| `bwd_f32` (`sm_100a`, `sm_103a`) | static backward (an FP32 weight gradient): 128 threads, 8-row blocks, CTA `b` runs chunks `b`, `b + grid`, ... (the same bits at any grid), 4 copies of the weight-gradient windows and no prefetch, the CTA that arrives last reading each window by an atomic exchange with 0 (its build `{"bulk": 0, "chunks_per_step": 0, "finish": "exch", "min_blocks": 8, "out": "f32", "prefetch_ahead": 0, "schedule": "static", "slots": 4, "sparse": 0}`) | its FP32 backward band, past 64 and up to 128 rows per SM (the call role `bwd_f32` has no program on `sm_107a`) |
| `bwd_bf16` (`sm_107a`) | static backward (a BF16 weight gradient): 128 threads, 8-row blocks, CTA `b` runs chunks `b`, `b + grid`, ... (the same bits at any grid), 4 copies of the weight-gradient windows plus occupancy words (the last CTA reads only the windows some CTA added into) | no band (the call role `bwd_bf16` has no program on `sm_100a` and `sm_103a`) |
| `bwd_bf16_rc` | row-claims backward (a BF16 weight gradient; `sm_100a`, `sm_103a`): 256 threads, 16 lanes a row, 16-row blocks, the CTAs past one wave claiming blocks from the claim counter; each thread's FP32 runs of rows added into plain FP32 words, each CTA's FP32 column sums added into 4 FP32 replicas with FP32 atomic adds, the CTA that arrives last summing the replicas in FP32 and rounding the sum | no band |
| `bwd_bf16_rc_mid` | row-claims mid backward (`sm_100a`, `sm_103a`): the row-claims backward with 5 CTAs per SM, thread 0 warming the next block of packed `dy`, `x` and `z` in L2 with one bulk prefetch per tensor; the row-claims backward's FP32 words and 4 FP32 replicas | no band |
| `bwd_bf16_mid` | claimed backward (a BF16 weight gradient): CTAs claim runs of chunks at run time; on `sm_100a` and `sm_103a` thread 0 warms the CTA's next block in L2 with one bulk prefetch per tensor where `dy`, `x` and `z` hold packed rows (head stride 128 and token stride `H * 128`; other layouts keep the per-thread prefetches), on `sm_107a` it warms the block two ahead | no band |
| `bwd_bf16_short_nodw` | the short backward built without the weight gradient (its build and `"dw": False`): one cluster of 16 CTAs, no column sums and no cluster synchronization, no workspace; `bwd_bf16_short`'s `dx` and `dz` bits | `compute_grad_weight=False` in `bwd_bf16_short`'s band |
| `bwd_bf16_t512_nodw` (`sm_100a`, `sm_107a`) | the exact tiny backward of a BF16 weight gradient built without it: one cluster of 16 CTAs of 4 teams, no column sums and no cluster synchronization, no workspace; `bwd_bf16_t512`'s `dx` and `dz` bits | `compute_grad_weight=False` in `bwd_bf16_t512`'s band |
| `bwd_bf16_fused_nodw_strided` | the fused backward built without the weight gradient, strided (its build and `"dw": False, "strided": True`): CTA `b` runs the 16-row blocks `b`, `b + grid`, ... with no claim, no ticket and no workspace word; `bwd_bf16_fused`'s `dx` and `dz` bits at any grid | `compute_grad_weight=False` in the fused band up to `NODW_STRIDED_ROWS_PER_SM` (`sm_100a`, `sm_103a`; `sm_107a`: every row count of the band) |
| `bwd_bf16_fused_nodw` (`sm_100a`, `sm_103a`) | the fused backward built without the weight gradient (its build and `"dw": False`): its claims, ticket, grid and workspace; `bwd_bf16_fused`'s `dx` and `dz` bits | `compute_grad_weight=False` in the fused band past `NODW_STRIDED_ROWS_PER_SM` |
| `bwd_bf16_rep_nodw_strided` (`sm_107a`) | the repeatable backward built without the weight gradient, strided (its build and `"dw": False, "strided": True`): as `bwd_bf16_fused_nodw_strided`; `bwd_bf16_rep`'s `dx` and `dz` bits at any grid | `compute_grad_weight=False` in a repeatable band on rows that are not packed |
| `bwd_bf16_rep_nodw_strided_packed` (`sm_107a`) | the repeatable backward built without the weight gradient, strided, in 128-thread CTAs (its build with `"threads": 128, "min_blocks": 8, "dw": False, "strided": True`): 16 lanes a row as in `bwd_bf16_rep`, CTA `b` runs the 8-row blocks `b`, `b + grid`, ... with no claim, no ticket and no workspace word, at most 8 CTAs per SM; `bwd_bf16_rep`'s `dx` and `dz` bits at any grid | `compute_grad_weight=False` in a repeatable band on packed rows (`dy`, `x` and `z` of head stride 128 and token stride `H * 128`, more than one token), except past the long cut of packed rows |
| `bwd_bf16_rep_nodw_strided_packed_small` (`sm_100a`, `sm_103a`) | the repeatable backward built without the weight gradient, strided, in 128-thread CTAs of packed rows only (its build with `"threads": 128, "min_blocks": 8, "dw": False, "strided": True, "packed_rows": 1, "schedule": "joint"`: the deterministic package's build of packed rows only): 16 lanes a row as in `bwd_bf16_rep`, CTA `b` runs the 8-row blocks `b`, `b + grid`, ... with no claim, no ticket and no workspace word, no stride test, every row read at its linear offset, a row's sum of squares and dot product reduced across its lanes together, at most 8 CTAs per SM; `bwd_bf16_rep`'s `dx` and `dz` bits at any grid | `compute_grad_weight=False` on packed rows (and one token of head stride 128) up to `PACKED_SMALL_ROWS_PER_SM` (128 rows per SM), in the place of the program without the weight gradient of the band that runs the rows |
| `bwd_bf16_rep_nodw` (`sm_100a`, `sm_103a`) | the repeatable backward built without the weight gradient (its build and `"dw": False`): its claims, ticket, grid and workspace; `bwd_bf16_rep`'s `dx` and `dz` bits | `compute_grad_weight=False` in a repeatable band, except on packed rows past the long cut |
| `bwd_bf16_rep_nodw_long` | the repeatable backward's long program built without the weight gradient (its build and `"dw": False`): its look-ahead loop, claims, ticket, grid and workspace; `bwd_bf16_rep`'s `dx` and `dz` bits | `compute_grad_weight=False` on packed rows where `bwd_bf16_rep_long` runs (such a call runs no long program on other layouts) |

The programs (`ROUTES`; each name is `cake_rmsnorm_silu_gate_train_` and
the 20 hex digits shown):

| role | `sm_100a` | `sm_103a` | `sm_107a` |
| --- | --- | --- | --- |
| `fwd` | `786ee0cda6a87f643bf8` | `786ee0cda6a87f643bf8` | `786ee0cda6a87f643bf8` |
| `fwd_tiny` | `242b29d118997473fc3e` | `242b29d118997473fc3e` | `242b29d118997473fc3e` |
| `fwd_l32_tiny` | `b9af069174152aae5f62` | `b9af069174152aae5f62` | (none) |
| `fwd_short` | `7b542c846911aca66512` | `7b542c846911aca66512` | `7b542c846911aca66512` |
| `fwd_mid` | `33ef19a28735b3e470b3` | `33ef19a28735b3e470b3` | `33ef19a28735b3e470b3` |
| `fwd_long` | `0dc2a8df1f94d878ccbb` | (not routed) | (not routed) |
| `fwd_xlong` | `ff8010718cbdf7cfa830` | `ff8010718cbdf7cfa830` | `ff8010718cbdf7cfa830` |
| `fwd_pipe` | `e6e0bf4c758851cabc99` | `e6e0bf4c758851cabc99` | (none) |
| `fwd_pipe_packed` | (none) | (none) | `9cb071b2dfe75474aaa7` |
| `fwd_layout_pipe` | `f99efa1083ef8ff2189a` | `f99efa1083ef8ff2189a` | `0a7f917d9c7922f9c2d2` |
| `fwd_layout_keep_pipe` | `92eca43778aed20f8f96` | (none) | `db508dc3390bc479f749` |
| `fwd_keep_pipe` | `978d2fa31ffda703d8bd` | (none) | (none) |
| `fwd_keep_pipe_packed` | (none) | (none) | `0e472c8f137ccde90b25` |
| `bwd_bf16_solo` | `36a3f270afc1a27290b7` | `36a3f270afc1a27290b7` | `36a3f270afc1a27290b7` |
| `bwd_bf16_short` | `cb597a0bebd6145e2024` | `cb597a0bebd6145e2024` | `cb597a0bebd6145e2024` |
| `bwd_bf16_t512` | `f6a3420d9aaa20a99e8d` | (none) | `f6a3420d9aaa20a99e8d` |
| `bwd_bf16_fused` | `2ce27f38a52585400ca0` | `2ce27f38a52585400ca0` | `2ce27f38a52585400ca0` |
| `bwd_bf16_rep` | `8a6cb16beff15b6ffbfe` | `8a6cb16beff15b6ffbfe` | `8a6cb16beff15b6ffbfe` |
| `bwd_bf16_rep_long` | `fef9d1cc2cc267219992` | `fef9d1cc2cc267219992` | `fef9d1cc2cc267219992` |
| `bwd_bf16` | (none) | (none) | `6d7cf79fb3d679d07410` |
| `bwd_bf16_rc` | `58f1ac96c476c73ec5c8` | `58f1ac96c476c73ec5c8` | (none) |
| `bwd_bf16_rc_mid` | `9f2f4d1f2b85a7469a7d` | `9f2f4d1f2b85a7469a7d` | (none) |
| `bwd_bf16_mid` | `2a3bbf332a501a11ca31` | `2a3bbf332a501a11ca31` | `51408d05160a7056aad8` |
| `bwd_f32_t512` | `348a297be36a61e8f015` | `348a297be36a61e8f015` | `348a297be36a61e8f015` |
| `bwd_f32` | `421e73578399336342e3` | `421e73578399336342e3` | (none) |
| `bwd_f32_rep` | `ae2073f31d0923e61736` | `ae2073f31d0923e61736` | `ae2073f31d0923e61736` |
| `bwd_f32_rep_long` | `be861a132bbc76cb6433` | `be861a132bbc76cb6433` | `be861a132bbc76cb6433` |
| `bwd_bf16_short_nodw` | `44693c0bd2ecbeaa207b` | `44693c0bd2ecbeaa207b` | `44693c0bd2ecbeaa207b` |
| `bwd_bf16_t512_nodw` | `a374cf1736d501ae6d3f` | (none) | `a374cf1736d501ae6d3f` |
| `bwd_bf16_fused_nodw_strided` | `1b2751f09a6c95f35605` | `1b2751f09a6c95f35605` | `1b2751f09a6c95f35605` |
| `bwd_bf16_fused_nodw` | `20a7c07ccc0866350d49` | `20a7c07ccc0866350d49` | (none) |
| `bwd_bf16_rep_nodw_strided` | (none) | (none) | `cfe88bc164eeaa58821d` |
| `bwd_bf16_rep_nodw_strided_packed` | (none) | (none) | `07de4b4c9a438a52ff79` |
| `bwd_bf16_rep_nodw_strided_packed_small` | `38d04fbfa79b1461bc0c` | `38d04fbfa79b1461bc0c` | (none) |
| `bwd_bf16_rep_nodw` | `cefc51faec88b7212f3c` | `cefc51faec88b7212f3c` | (none) |
| `bwd_bf16_rep_nodw_long` | `d8dd2e7dae0ca8233241` | `d8dd2e7dae0ca8233241` | `d8dd2e7dae0ca8233241` |

The forward bands (rows per SM: a call of `rows` rows on a device of `SMs` SMs
runs the first band with `rows <= last * SMs`): `FORWARD_BANDS` where `x` and
`z` hold packed rows (row `n` at element `128 n`, as contiguous `[T, H, 128]`
and `[T * H, 128]` inputs do) and there are fewer than `PACKED_ROWS` (2^24)
rows; otherwise the table of the rows' class (`forward_layout`):
`PROJECTION_FORWARD_BANDS` where `x` holds packed rows of more than one head
and `z` the gate columns of a wider projection (head stride 128, a token
stride past `H * 128`), `HEAD_MAJOR_FORWARD_BANDS` where `x` and `z` are
head-major (token stride 128, head stride `T * 128`: a `[H, T, 128]` tensor
viewed as `[T, H, 128]`) on `sm_107a`, and `STRIDED_FORWARD_BANDS` for every
other layout (`x` and `z` both sliced, such a gate of one head, a broadcast
over heads, a mixed or padded layout, head-major rows on `sm_100a` and
`sm_103a`) and for packed rows from
`PACKED_ROWS` on.  An `x` or `z` broadcast over more than one token (token
stride 0) counts as packed rows, but where its band of packed rows runs a
program that reads the rows at their linear offsets (a tiny forward, or on
`sm_107a` a pipelined forward of packed rows) it runs the band of
`STRIDED_FORWARD_BANDS`, and where it runs `fwd_keep_pipe` it runs `fwd_pipe`
(the program of 0.2.16 there, the same bits; `fwd_keep_pipe` was measured on
contiguous rows only).  The cuts follow the checkpointed training step (two
forwards and the backward) on every architecture, measured with the
one-CTA-per-block pipelined grid of 0.2.13 (one CTA per block below 8192 rows
per SM; from 0.2.14 the pipelined forwards launch whole waves from their cuts
of each architecture, and the bands are unchanged): the pipelined forward
loads `x` and `z` with the L2 `evict_first` policy, so where the rows fit in
L2 it is faster alone but makes the step slower.  On `sm_103a` the mid forward
up to 240 rows per SM and the extra-long forward up to 600 made the step 4 %
.. 10 % faster than the pipelined forward, which alone was at most 4.5 %
faster.  On `sm_107a` the pipelined forward runs packed rows past 280 rows per
SM, where alone it was about 10 % faster than the extra-long forward and made
the step at most about 3.5 % slower (past 8192 rows per SM
`fwd_keep_pipe_packed`, below).  On `sm_100a` the pipelined forward runs
packed rows past 520 rows per SM with no end: up to 0.2.16 the mid forward ran
them past 6500, but with 0.2.17's grids the pipelined forward alone was 1 % ..
3.4 % faster there and the step 0.2 % .. 0.9 % faster (7000 to 21255 rows per
SM, CUDA Graph replays, CUPTI, cold L2).  On `sm_107a` packed rows of 17 to 32
rows run the short forward (`FORWARD_TINY_GAP`, whatever the SMs): its two
16-row blocks took 0.87 .. 0.94 of the time of the tiny forward's three or
four 8-row blocks (cold L2), the same bits.  Rows of `STRIDED_FORWARD_BANDS`
run the plain forward (`fwd`) past the persistent forwards' bands: where
measured there (from 1000 rows per SM on `sm_100a` and `sm_103a`, from 520 on
`sm_107a`; with the pipelined forward's grid of 0.2.13) it was the fastest
forward alone and made the step 1 % .. 6 % faster; no band of
`STRIDED_FORWARD_BANDS` runs a pipelined forward.  Version 0.2.17's pipelined
forwards with `x` and `z` loaded with the L2's default policy
(`fwd_keep_pipe`, `fwd_keep_pipe_packed`), measured with its grids (CUDA Graph
replays, CUPTI, cold L2) on packed rows of 1, 16 and 48 heads against the
pipelined forward, made the checkpointed step of either weight gradient faster
and were no slower alone on `sm_100a` at 59 rows per SM (the step 0.960 ..
0.980 of its time) and on `sm_107a` at 9000 and 15000 (the forward and the
step about 0.987 .. 0.990).  On `sm_103a`, where that measurement found the
step at 2000 and 3000 rows per SM 0.991 .. 0.997 of its time, the release
measurement of 0.2.17 found the forward 0.35 % .. 0.55 % slower than 0.2.16's
at 2500 (1 and 16 heads), so `sm_103a` routes no `fwd_keep_pipe`.
A gate sliced from a projection ran the bands of other layouts up to 0.2.16;
measured in 0.2.17 (16 and 48 heads, `z` sliced from a projection
`2 * H * 128 + 16` elements wide) against those forwards, on `sm_100a`
`fwd_layout_pipe` from 700 to 2000 rows per SM made the forward alone 4.5 % ..
8 % faster and `fwd_layout_keep_pipe` at 3000 the forward 3 % and the step
0.3 % .. 0.4 % faster; on `sm_103a` `fwd_layout_pipe` from 700 to 2000 the
forward 4 % .. 8 % faster; on `sm_107a` `fwd_layout_keep_pipe` from 100 to
332 rows per SM the forward 2.5 % .. 10 % faster and from 3000 the forward and
the step faster than the deterministic package's, and `fwd_layout_pipe` from
443 to 2000 every op at the deterministic package's time.  The release
measurement of 0.2.17 against 0.2.16 found `fwd_layout_keep_pipe`'s step
0.2 % .. 0.5 % slower on `sm_100a` from 4000 rows per SM and 0.2 % .. 0.7 %
slower on `sm_103a` from 2500 to 6445 (16 and 48 heads), so its `sm_100a`
band ends at 3000 and `sm_103a` runs the plain forward past 2000; on `sm_107a`
(driver 620.70) it found its step 1.1 % .. 1.9 % slower from 309 to 332 rows
per SM, so its first `sm_107a` band ends at 300 and the mid forward runs from
301 to 442, as in 0.2.16; it found a gate of one head up to 2.2 % slower with
these bands (the forward on `sm_100a` at 3000 and 4000 rows per SM, the step
on `sm_107a` at 300 and 2000), so such a gate runs the bands of other layouts.
On `sm_107a` head-major rows ran `fwd_layout_keep_pipe` at 100 rows per SM
9 % .. 10 % faster alone than the mid forward and made the step 3 % .. 4 %
faster; the release measurement of 0.2.17 against 0.2.16 (driver 620.70)
found their step with it 0.9 % .. 1.3 % slower at 120 rows per SM and with
`fwd_layout_pipe` 0.6 % .. 3.9 % slower from 442 to 1000, so
`fwd_layout_keep_pipe` runs up to 100 rows per SM and past it head-major rows
run the bands of other layouts (the mid forward up to 480, the plain forward
past it).  On `sm_100a` and `sm_103a` neither made the step of head-major rows
faster.  The tiny forwards and the short forward end where beside a kernel
holding 16 SMs they still end no later than the deterministic package's
forward as it routes the rows.  The 32-lane tiny forward (`sm_100a` and
`sm_103a`), which has that package's tiny row blocks and so twice as many
CTAs as the tiny forward, ends at 4 rows per SM on `sm_100a`, past which CUDA
Graph replays of back-to-back calls run it up to about 1.5 % slower than the
tiny forward, and at 12 on `sm_103a`, past which, with the rows in L2, it is
slower.  The bands:

| architecture | packed rows (rows per SM) | gate of more than one head sliced from a projection (rows per SM) | head-major rows (rows per SM) | other layouts (rows per SM) |
| --- | --- | --- | --- | --- |
| `sm_100a` | 1-4 `l32_tiny`, 5-36 `tiny`, 37-85 `keep_pipe`, 86-160 `mid`, 161-400 `long`, 401-520 `xlong`, past 520 `pipe` | 1-12 `short`, 13-140 `mid`, 141-400 `long`, 401-650 `xlong`, 651-2000 `layout_pipe`, 2001-3000 `layout_keep_pipe`, past 3000 `fwd` | as other layouts | 1-12 `short`, 13-140 `mid`, 141-400 `long`, 401-700 `xlong`, past 700 `fwd` |
| `sm_103a` | 1-12 `l32_tiny`, 13-36 `tiny`, 37-240 `mid`, 241-600 `xlong`, 601-6000 `pipe`, 6001-8192 `mid`, past 8192 `pipe` | 1-12 `short`, 13-250 `mid`, 251-699 `xlong`, 700-2000 `layout_pipe`, past 2000 `fwd` | as other layouts | 1-12 `short`, 13-250 `mid`, 251-700 `xlong`, past 700 `fwd` |
| `sm_107a` | 1-65 `tiny` (but 17-32 rows `short`), 66-175 `mid`, 176-280 `xlong`, 281-8192 `pipe_packed`, past 8192 `keep_pipe_packed` | 1-59 `short`, 60-300 `layout_keep_pipe`, 301-442 `mid`, 443-2999 `layout_pipe`, past 2999 `layout_keep_pipe` | 1-59 `short`, 60-100 `layout_keep_pipe`, 101-480 `mid`, past 480 `fwd` | 1-59 `short`, 60-480 `mid`, past 480 `fwd` |

The backward bands (`BACKWARD_BANDS`, in order: a call of `rows` rows on a
device of `SMs` SMs runs the first band it fits, a small-row band while
`rows` is at most its count of rows (16 for `bwd_bf16_solo`, 256 for
`bwd_bf16_short`, 512 for the exact tiny backwards), another band while
`rows <= last * SMs` (rows per SM), the last band always; a call with
`compute_grad_weight=False` takes the BF16 bands but the solo band).
A BF16 weight gradient runs the fused backward past its small-row bands up
to 1000 rows per SM and the repeatable backward past it on `sm_100a` and
`sm_103a`; on `sm_107a`, where the band-selection measurement found the
fused backward saving about 3 % or more of the step only past 64 and up to
256 rows per SM (both cuts on the repeatable
backward's waves of 64 rows per SM), the repeatable backward runs past the
small-row bands up to 64 rows per SM, the fused backward up to 256 and the
repeatable backward past it
(`SM107A_BF16_REP_LAST`, `SM107A_BF16_FUSED_LAST`), and packed rows of its
last band from `LONG_BACKWARD_ROWS_PER_SM` rows per SM (2560 on `sm_100a`
and `sm_103a`, 1536 on `sm_107a`; only the last repeatable band reaches
them), and its rows of other layouts from
`LONG_BACKWARD_STRIDED_ROWS_PER_SM` rows per SM (6145 on `sm_100a` and
`sm_103a`, 4097 on `sm_107a`; not in a call with
`compute_grad_weight=False`), run the repeatable backward's long program
`bwd_bf16_rep_long` in its place.  An FP32 weight gradient runs the repeatable backward `bwd_f32_rep`
past the exact tiny backward's 512 rows on every architecture, but on
`sm_100a` and `sm_103a` the static backward `bwd_f32` past 64 and up to 128
rows per SM (`F32_REP_LAST`, `F32_STATIC_LAST`), and packed rows of its last
band from the same `LONG_BACKWARD_ROWS_PER_SM` cuts run its long program
`bwd_f32_rep_long`.  The FP32 cuts follow the band-selection measurement of
the checkpointed training step (a CUPTI kernel timer, cold L2): the static
backward saved about 2 % .. 7 % (B200, `sm_100a`) and 5 % .. 8 % (GB300,
`sm_103a`) of the step past one and up to two of the repeatable backward's
waves of 64 rows per SM (at most 4 CTAs per SM claiming 16-row blocks,
which past one wave take a second round of blocks, where the static
backward's 8 CTAs per SM of 8-row blocks split the rows more evenly); and,
elsewhere where it measured, the repeatable backward made the step as fast
or faster and was faster alone:

| architecture | BF16 weight gradient | FP32 weight gradient |
| --- | --- | --- |
| `sm_100a` | 1-16 rows `bwd_bf16_solo`, 17-256 rows `bwd_bf16_short`, 257-512 rows `bwd_bf16_t512`, up to 1000 rows per SM `bwd_bf16_fused`, past that `bwd_bf16_rep` | 1-512 rows `bwd_f32_t512`, up to 64 rows per SM `bwd_f32_rep`, up to 128 rows per SM `bwd_f32`, past that `bwd_f32_rep` |
| `sm_103a` | 1-16 rows `bwd_bf16_solo`, 17-256 rows `bwd_bf16_short`, up to 1000 rows per SM `bwd_bf16_fused`, past that `bwd_bf16_rep` | 1-512 rows `bwd_f32_t512`, up to 64 rows per SM `bwd_f32_rep`, up to 128 rows per SM `bwd_f32`, past that `bwd_f32_rep` |
| `sm_107a` | 1-16 rows `bwd_bf16_solo`, 17-256 rows `bwd_bf16_short`, 257-512 rows `bwd_bf16_t512`, up to 64 rows per SM `bwd_bf16_rep`, up to 256 rows per SM `bwd_bf16_fused`, past that `bwd_bf16_rep` | 1-512 rows `bwd_f32_t512`, past that `bwd_f32_rep` |

For example, the FP32 static band holds 9473 to 18944 rows on a 148-SM
B200 and 9729 to 19456 rows on a 152-SM GB300; the FP32 repeatable bands
hold 513 to 9472 rows and the rows from 18945 on a 148-SM B200 and the rows
from 513 on a 212-SM R200, and packed rows run the FP32 long program from
378880 rows on a 148-SM B200 and from 325632 rows on a 212-SM R200 (as the
BF16 long program does), and other layouts of a BF16 `dw` run the BF16 long
program from 909460 rows on a 148-SM B200 and from 868564 rows on a 212-SM
R200; the BF16 fused band holds 513 to 148000 rows on a
148-SM B200 and 13569 to 54272 rows on a 212-SM R200.

A call with `compute_grad_weight=False` takes the BF16 bands (the short
backward's band from 1 row: no solo program without the weight gradient
ships) and runs, in
its band module's place, that program built without the weight gradient
(`nodw_module`; `NODW_ROLES`), whose `dx` and `dz` are the bits of the BF16
call that computes the weight gradient at every grid (but in step 2):

1. packed rows past the long cut (where `bwd_bf16_rep_long` runs):
   `bwd_bf16_rep_nodw_long`, or, where none is routed, `bwd_bf16_rep_long`
   with `weight_grad` 0;
2. else, on packed rows (`dy`, `x` and `z` of head stride 128 and token
   stride `H * 128`; the call also passes one token of head stride 128 as
   packed rows where this cut is) up to
   `PACKED_SMALL_ROWS_PER_SM[arch]["bwd_bf16"] * SMs` rows (128 rows per SM
   on `sm_100a` and `sm_103a`, no cut on `sm_107a`),
   `bwd_bf16_rep_nodw_strided_packed_small` where routed, in any band that
   routes a program without the weight gradient: its `dx` and `dz` are the
   repeatable backward's bits, which in the one-cluster bands, and on
   `sm_103a` in the fused band, can differ in their last bits from those of
   the call that computes the weight gradient (on `sm_100a` the fused
   band's are the repeatable backward's bits);
3. else `<band>_nodw_strided` where routed and `rows <=
   NODW_STRIDED_ROWS_PER_SM[arch][band] * SMs`;
4. else `<band>_nodw` where routed (the claimed program, or the
   one-cluster program of a small-row band);
5. else `<band>_nodw_strided` where routed;
6. else the band's module with `weight_grad` 0.

In steps 3 and 5 packed rows (`dy`, `x` and `z` of head stride 128 and
token stride `H * 128`, more than one token) run
`<band>_nodw_strided_packed` in the strided
program's place where it is routed: the repeatable band's 128-thread build
on `sm_107a`.  Other layouts run the 256-thread strided program.

A band has a strided cut where its architecture routes both its strided and
its claimed program.  The cuts were measured per architecture (CUDA Graph
replays, cold L2): up to them the strided program is the faster.  The
repeatable band routes only its claimed program on `sm_100a` and `sm_103a`,
and `sm_107a` routes no claimed program, so there the fused and the
repeatable bands run their strided programs at every row count (step 5):

| architecture | `bwd_bf16_fused` | `bwd_bf16_rep` |
| --- | --- | --- |
| `sm_100a` | strided up to 512 rows per SM, claimed past it | claimed |
| `sm_103a` | strided up to 448 rows per SM, claimed past it | claimed |
| `sm_107a` | strided | strided |

The chunked backwards (the static backward of an FP32 weight gradient on
`sm_100a` and `sm_103a`; the static and the claimed backward of a BF16
weight gradient, which no band runs) cut the rows into 8-row blocks and
the blocks into the chunks of the deterministic backward at 2e989556: `blocks_per_chunk =
max(1, ceil(blocks / 8192))`, `chunks = ceil(blocks / blocks_per_chunk)`, and
launch `min(chunks, 8 * SMs)` CTAs of 128 threads (15616 bytes of dynamic
shared memory each).  The claimed backward lets a CTA slowed by a kernel on
another stream take fewer chunks; the bits do not depend on which CTA runs a
chunk.  The row-claims backwards cut the rows into 16-row blocks and launch
`min(blocks, 4 * SMs)` CTAs (`5 * SMs` for `bwd_bf16_rc_mid`) of 256 threads
(12544 bytes of dynamic shared memory each); past one wave the CTAs claim
blocks from the claim counter, so a CTA that starts late takes fewer blocks.
The fused backward cuts the rows into 16-row blocks and launches
`min(blocks, 4 * SMs)` CTAs of 256 threads (12544 bytes of dynamic shared
memory each), which claim the blocks in chunks when there are more blocks
than CTAs; the bits of `dx` and `dz` do not depend on which CTA runs a block.
The repeatable backwards (BF16 and FP32) and their long programs launch the
same grid of 256-thread CTAs (18176 bytes of dynamic shared memory each),
which claim the blocks in chunks whose rows are fixed by the row count and
the grid, so their bits do not depend on which CTA runs a chunk.
A small-row backward launches one cluster of 16 CTAs whatever the rows: 512
threads each for the exact tiny backward (19584 bytes of dynamic shared
memory each), 256 threads each for the short backward (6144 bytes each); the
solo backward one CTA of 256 threads (18176 bytes), launched without a
cluster.
A strided program without the weight gradient launches `min(blocks,
ctas_per_sm * SMs)` CTAs of its own rule (`backward_ctas`), CTA `b` running
the blocks `b`, `b + grid`, ...: 256-thread CTAs over 16-row blocks, 4 per
SM, and on the repeatable band's packed rows (`sm_107a`; and the build of
packed rows only, `sm_100a` and `sm_103a`) 128-thread CTAs over 8-row
blocks, 8 per SM.  Where `dy`, `x` and `z` hold packed rows
the fused band's strided program and the 128-thread one read each row at
its linear offset, and a CTA of theirs with one block takes no integer
division before its loads; the build of packed rows only takes no stride
test at all.  The claimed, long and
one-cluster programs without the weight gradient launch their band
programs' grids.

## Saved state

The autograd function (`cake_rmsnorm_silu_gate_nondet`) saves `x`, `z` and
`weight` by reference through `save_for_backward`, and `eps` as a Python
float.  It saves no `rstd` and no other intermediate, so beyond its inputs it
keeps 0 bytes, and a non-reentrant checkpointed forward keeps nothing but its
output.  The saved tensors are the caller's own: `x`
and `z` hold `T * H * 128` BF16 values (`T * H * 256` bytes each) and
`weight` 128 BF16 values (256 bytes).  A backward's workspace (below) is not
saved state: no tensor of it is kept by the autograd node.

## Workspace

The static, the fused, the repeatable, the row-claims and the claimed
backwards need a workspace, and so do the claimed and long programs
without the weight gradient (on their band programs' layouts); the
small-row backwards (the solo backward among them) and the strided and
one-cluster programs without the
weight gradient read none, so a call in a small-row band, or with
`compute_grad_weight=False` where its band runs the strided program, uses
no workspace (a workspace passed is checked and left as it is).  The size
does not depend on `compute_grad_weight`.  A workspace is a zero-filled,
contiguous `torch.uint8` CUDA tensor of
`cake_rmsnorm_silu_gate_nondet_train_backward_workspace_bytes` bytes whatever
`T * H`: 29192 bytes on `sm_100a` and `sm_103a` and 36360 on `sm_107a` (the
largest layout among the architecture's routed backward programs), starting
at a 16-byte aligned address.  These sizes are separate from the saved state
above.  Each program lays its buffers from the start of it
(`workspace_layout`).  A chunked backward (static or claimed) lays three, with
`slots` copies of the integer windows:

| buffer | dtype | elements | byte offset |
| --- | --- | --- | --- |
| `dw_win` (integer windows: `slots` x 7 windows x 128 columns) | u64 | `slots * 896` | 0 |
| `dw_spec` (class words of infinite and NaN partials) | FP32 | 128 | `slots * 7168` |
| `dw_done` (arrival ticket, claim counter) | u32 | 2 | `slots * 7168 + 512` |

A row-claims backward lays two, 2056 bytes in all, and the fused backward
two, 4104 bytes in all (`replicas` 4 and 8):

| buffer | dtype | elements | byte offset |
| --- | --- | --- | --- |
| `dw_sums` (FP32 replicas: `replicas` x 128 columns) | FP32 | `replicas * 128` | 0 |
| `dw_done` (arrival ticket, claim counter) | u32 | 2 | `replicas * 512` |

The repeatable backwards (BF16 and FP32) and their long programs lay three,
18952 bytes in all (the deterministic package's layout):

| buffer | dtype | elements | byte offset |
| --- | --- | --- | --- |
| `acc` (fixed-point digits: 2 x 9 x 128) | u64 | 2304 | 0 |
| `flags` (a flag word a column) | u32 | 128 | 18432 |
| `done` (arrival ticket, claim counter) | u32 | 2 | 18944 |

| program | `sm_100a`, `sm_103a` | `sm_107a` |
| --- | --- | --- |
| `bwd_bf16_short` | none | none |
| `bwd_bf16_t512` | none (`sm_100a`; no program on `sm_103a`) | none |
| `bwd_f32_t512` | none | none |
| `bwd_bf16_fused` | `dw_sums` and `dw_done`, 4104 bytes | `dw_sums` and `dw_done`, 4104 bytes |
| `bwd_bf16_rep`, `bwd_bf16_rep_long`, `bwd_f32_rep`, `bwd_f32_rep_long` | `acc`, `flags` and `done`, 18952 bytes | `acc`, `flags` and `done`, 18952 bytes |
| `bwd_f32` | 4 slots, 29192 bytes | (no program) |
| `bwd_bf16` | (no program) | 5 slots (4 window copies, then the occupancy words), 36360 bytes |
| `bwd_bf16_rc` | `dw_sums` and `dw_done`, 2056 bytes | (no program) |
| `bwd_bf16_rc_mid` | `dw_sums` and `dw_done`, 2056 bytes | (no program) |
| `bwd_bf16_mid` | 4 slots, 29192 bytes | 4 slots, 29192 bytes |
| `bwd_bf16_short_nodw`, `bwd_bf16_fused_nodw_strided` | none | none |
| `bwd_bf16_t512_nodw` | none (`sm_100a`; no program on `sm_103a`) | none |
| `bwd_bf16_rep_nodw_strided`, `bwd_bf16_rep_nodw_strided_packed` | (no program) | none |
| `bwd_bf16_fused_nodw` | `dw_sums` and `dw_done`, 4104 bytes | (no program) |
| `bwd_bf16_rep_nodw` | `acc`, `flags` and `done`, 18952 bytes | (no program) |
| `bwd_bf16_rep_nodw_long` | `acc`, `flags` and `done`, 18952 bytes | `acc`, `flags` and `done`, 18952 bytes |

Every launch leaves the whole workspace zero, so one workspace serves every
later call on its stream (any shape, both weight-gradient dtypes, every
backward program of its architecture, CUDA Graph replays) without a reset.
Give each stream that may run a backward at the same time its own workspace.
A buffer that did not come from
`cake_rmsnorm_silu_gate_nondet_train_backward_workspace`, or that anything
else wrote into, must be zeroed with
`cake_rmsnorm_silu_gate_nondet_train_reset_workspace` before its next
backward.  The deterministic package's FP32 workspace is refused with a
`ValueError` that names both workspaces.

Calls without a workspace use one per device, stream and backward program
that takes one (of the size above: 29192 bytes on `sm_100a` and `sm_103a`,
36360 on `sm_107a`; on each stream at most one per such program the
architecture routes: the static, the fused `_fused`, the repeatable `_rep` and `_rep_long`, the row-claims `_rc` and
`_rc_mid` and the claimed `_mid` backwards, and the claimed and long
programs without the weight gradient `_fused_nodw`, `_rep_nodw` and
`_rep_nodw_long`, so a stream that runs calls with and without the weight
gradient in one band keeps one more), kept for the process.  A call that
runs a program without a workspace makes none and uses no reserve: its
unread workspace and weight-gradient parameters get empty tensors made
once per device.  Eager calls with a workspace but without a caller
workspace also keep a reserve per device for CUDA Graph captures: 64
zero-filled workspaces in one allocation (with 256-byte alignment, 29440 bytes
each, about 1.8 MiB, on `sm_100a` and `sm_103a`, and 36608 bytes each, about
2.2 MiB, on `sm_107a`), refilled outside any capture, at most 1024 per device.
A captured call without a workspace takes one from the reserve and keeps it,
so a replay launches no fill kernel: one for its capture and capture stream
when cuda-python is installed (it gives the capture ids), otherwise one per
captured call.  With none ready, the graph zero-fills a workspace of its own.

## Layout of this package

* `__init__.py` -- validation, the workspace views, the stream workspaces and
  the capture reserve, the launches, the autograd `Function` and the entry
  points.
* `cake_jit.py` -- the `MODULES` registry (one record per program; the
  exporter writes the derived fields), `ROUTES`, the row-count routing
  (`FORWARD_BANDS`, `PROJECTION_FORWARD_BANDS`, `HEAD_MAJOR_FORWARD_BANDS`,
  `STRIDED_FORWARD_BANDS`, `forward_layout`, `BACKWARD_BANDS`,
  `module_for_rows`; without the weight gradient `NODW_STRIDED_ROWS_PER_SM`,
  `PACKED_SMALL_ROWS_PER_SM`, `nodw_module` and the plans it reads,
  `_NODW_PLANS`, read once at import), the backward's chunks and grid (`backward_chunks`,
  `backward_ctas`, `launch_grid`), the workspace layout
  (`workspace_layout`, `workspace_bytes`) and the loader glue.
* `cake_build.py` + `build.json` -- the loader shared by every Cake-exported
  package and this package's build manifest (schema `cake-build/1`).
* `csrc/` -- generated translation units (kernel and tvm-ffi binding per
  program) and `cake_ffi_checks.h`.

The package imports only itself, `torch` and `tvm_ffi`, and, when installed,
cuda-python's `cuda.bindings.runtime` for CUDA Graph capture ids (without it,
each captured call takes its own reserve workspace); the JIT fallback of
`cake_build.py` imports the package's own JIT shim (`_cake_pplx_jit.py`).

## Tests

* `tests/norm/test_cake_rmsnorm_silu_gate_nondet_train.py` -- the package
  alone: every output judged against the PyTorch BF16 naive implementation
  over 32 inputs (see "Accuracy") and the reference's classes in every band,
  run-to-run bits on every path (a BF16 `dw` in the fused band: judged on
  every launch instead, and varying), workspaces, CUDA Graphs, routing and
  grids (the pipelined forwards' `y` bits on both of their grids, packed
  and broadcast rows; the classes of `forward_layout`; the pipelined
  forwards of 0.2.17 give the `y` bits of `fwd_pipe`, on `sm_107a` of
  `fwd_pipe_packed`, on packed, projection, head-major, one-token and
  broadcast rows), the programs without the weight gradient (the `dx`
  and `dz` bits of
  the BF16 call around their strided cut, on packed rows and other layouts
  and at small grids, no workspace for the strided and one-cluster ones,
  CUDA Graph capture included; the build of packed rows only at its cut,
  on one token and at small grids, with the repeatable backward's `dx` and
  `dz` bits and within the naive tier), the solo backward (1 to 16 rows of
  a BF16 weight gradient: the repeatable backward's `dx`, `dz` and `dw`
  bits, fresh and accumulated, in every layout, with no workspace, CUDA
  Graph capture included; a call without the weight gradient runs the
  programs of 0.2.16 at every row count, checked against the bands without
  the solo band), `torch.use_deterministic_algorithms` (the alert in the
  fused band, no alert elsewhere).
* `tests/norm/test_cake_rmsnorm_silu_gate_nondet_train_api.py` -- the public
  module: no mode keyword, the alerts, the facade and the backend, the
  workspace refusals, and each output judged against the PyTorch BF16 naive
  implementation around the forward's bands (packed rows, a gate sliced from
  a projection, head-major rows and other layouts) and the backward's bands
  and on the paths of the public module; no comparison with the deterministic
  package.
* `tests/norm/test_cake_rmsnorm_silu_gate_nondet_train_stall.py` -- the speed
  beside kernels resident on other streams (packed rows, a gate sliced from a
  projection, head-major rows and `x` and `z` both sliced).
* `benchmarks/bench_cake_rmsnorm_silu_gate_nondet_train.py` -- timing.
