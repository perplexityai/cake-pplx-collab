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
order in which CTAs run and the work beside the call on other streams, and
whatever the grid.  A BF16 `dw` repeats bit for bit too, whatever the grid
but in the repeatable band (`bwd_bf16_rep`, whose grid, and so whose bits,
the rows and the SM count fix), except in the fused band (`bwd_bf16_fused`;
see "Programs and routes"), where its last bits may vary from run to run.
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
"min_blocks": 4, "prefetch_share": None, "slots": 8}`) for a BF16 `dw` (or
`compute_grad_weight=False`): the deterministic package's fused backward's
schedule and row arithmetic (the approximate gate and `rstd`, the
next-block prefetch up to the architecture's share), so its `dx` and `dz`
are the same bits at any grid (a row's sums run on FMA chains of its own,
set per architecture, so their last bits may differ from that package's),
with the row-claims
backward's kind of weight gradient: FP32 runs of rows added into plain FP32
words, each CTA's FP32 column sums added into 8 FP32 replicas with FP32
atomic adds and the replicas summed in FP32 by the CTA that arrives last (no
FP64), so the order of the sum, and the last bits of `dw`, depend on
scheduling.  An FP32 `dw` runs no fused backward.  The
repeatable band runs the repeatable backward, the deterministic package's
BF16 backward program under this package's name (its build `{"math": "ax",
"min_blocks": 4, "out": "bf16", "prefetch_share": None, "threads": 256}`):
the fused backward's schedule and row arithmetic, the CTAs claiming the
blocks in chunks whose rows the row count and the grid fix, each chunk's
weight-gradient terms added exactly into fixed-point digits, so its `dx`,
`dz` and `dw` are the deterministic package's bits in this tree, the same
for the rows and the SM count.  Packed rows of the repeatable band (`dy`,
`x` and `z` each with head stride 128 and token stride `H * 128`) run its
long program instead, `bwd_bf16_rep_long`, from 2560 rows per SM on
`sm_100a` and `sm_103a` and from 1536 on `sm_107a`: the deterministic
package's look-ahead program (its build `{"lookahead": True, "math": "ax",
"min_blocks": 4, "out": "bf16", "prefetch_share": None, "threads": 256}`),
the same grid, workspace and bits.  The cuts live in one place,
`cake_jit.BACKWARD_BANDS` (the long program's in
`cake_jit.LONG_BACKWARD_ROWS_PER_SM`).  The row-claims programs
(`bwd_bf16_rc`, `bwd_bf16_rc_mid`; `sm_100a`, `sm_103a`), whose `dw` may vary
from run to run as well, and the BF16 static and claimed programs ship
routed, but no band runs them.

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
backward need not give these bits, but in the repeatable band `dx` and `dz`
are its bits.  Outside the small-row, the fused and the repeatable bands
`dw` is the exact sum of each chunk's FP32 partial (the chunks of rows of
the deterministic backward at commit 2e989556), rounded once in integer
arithmetic.  In the
small-row bands `dw` is that program's own fixed-order FP32 sum: the same
bits on every call, which need not be the deterministic package's.  In the
fused band `dw` is that program's own FP32 sum of FP32 replicas, in an order
that depends on scheduling; in the repeatable band it is the deterministic
package's bits in this tree.  No program sums in FP64.

| output | how it is computed | run to run |
| --- | --- | --- |
| `y` | `rstd` with `rsqrt.approx`, the sigmoid with `ex2.approx` and `rcp.approx` (the split gate), FP32 products, one rounding to BF16 | bit for bit |
| `dx` | each backward program's FP32 row arithmetic, its `.f32x2` FMA chains for a row's sums set per program and per architecture (the last bits can differ between bands); the fused and the repeatable band run the deterministic package's row arithmetic (the fused band on FMA chains of its own) | bit for bit |
| `dz` | each backward program's FP32 row arithmetic, its `.f32x2` FMA chains for a row's sums set per program and per architecture (the last bits can differ between bands); the fused and the repeatable band run the deterministic package's row arithmetic (the fused band on FMA chains of its own) | bit for bit |
| `dw` (the static and the claimed backwards: an FP32 `dw` past its small-row band) | each chunk's FP32 partial (the chunks of rows of the deterministic backward at 2e989556), summed exactly in 64-bit integer windows, the exact sum rounded once to FP32 in integer arithmetic (then to BF16 for a BF16 `dw`); a column with an infinite or NaN term has the class of the exact sum (`+inf`, `-inf` or `NaN`), also when the row count is not a multiple of the row block | bit for bit |
| `dw` (the exact tiny backward's bands) | the program's FP32 weight-gradient terms, summed on chip in plain FP32 words in an order of its own, rounded once to the output dtype | bit for bit |
| `dw` (BF16 in the short backward's band) | each CTA's FP32 partial of its 16-row blocks (the row-claims backward's row arithmetic); CTA `r` adds every CTA's partials of the columns `8r` .. `8r + 7` in rank order in FP32 and rounds the sum to BF16 | bit for bit |
| `dw` (BF16 in the fused band) | each thread's FP32 runs of at most 16 rows added into plain FP32 words; each CTA adds its FP32 column sums into 8 FP32 replicas with FP32 atomic adds in an order that depends on scheduling; the CTA that arrives last sums the replicas in FP32 and rounds the sum to BF16 | its last bits may vary |
| `dw` (BF16 in the repeatable band) | the deterministic package's backward in this tree: each thread's FP32 runs of at most 16 rows of a chunk added into plain FP32 words of the chunk, each CTA adding a column's FP32 word exactly into 64-bit fixed-point digits in the workspace; the CTA that arrives last rounds each column's exact sum once to FP32, then to BF16 | bit for bit (the rows and the SM count fix the chunks) |

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
SM count and, for the forward, whether `x` and `z` hold packed rows.  Each
program's arithmetic is fixed; every forward program takes the deterministic
forward's split gate.  The backward
programs' `.f32x2` FMA chains for a row's sums differ by program and by
architecture, so `dx`'s and `dz`'s last bits can differ between the backward
bands; each program repeats them bit for bit for the same inputs on the same
device.
The forward programs associate a row's sum of squares differently, on FP32
FMA chains (the lanes of `.f32x2` FMA chains where a program runs two or
four): the tiny and the short forward run four 2-term chains of 16 lanes
added in pairs, then folded by XOR shuffles 1, 2, 4, 8; the mid forward, the
extra-long forward but on `sm_100a` and the pipelined forward on `sm_107a`
four 4-term chains of 8 lanes added in pairs, then folded by 1, 2, 4; the
plain forward and the pipelined forward on `sm_100a` and `sm_103a` two
8-term chains of 8 lanes (the even and the odd elements) added, then folded
by 1, 2, 4; the long forward, and the extra-long forward on `sm_100a`,
16-term chains of 8 lanes folded by 1, 2, 4.  So `y`'s bits can differ
between the programs, and so between the forward bands of one architecture;
each program repeats `y` bit for bit for the same inputs on the same device.

| role | program | runs |
| --- | --- | --- |
| `fwd_tiny` | forward of packed rows, 128 threads, 16 lanes a row, one CTA per 8-row block (not persistent), the short forward's arithmetic and bits | its band of `FORWARD_BANDS` (packed rows) |
| `fwd_short` | forward, 256 threads, 16 lanes a row, one CTA per 16-row block (not persistent) | its band of `STRIDED_FORWARD_BANDS` (other layouts) |
| `fwd` | plain forward, 64 threads, 8 lanes a row, one CTA per 8-row block (not persistent) | the call role; its band of `STRIDED_FORWARD_BANDS` (other layouts, past the persistent forwards' bands) |
| `fwd_mid` | whole waves of 6 CTAs per SM (as many waves as the rows need), each CTA at most 2 blocks of 16 rows | its forward bands |
| `fwd_long` | whole waves of 3 CTAs per SM (as many waves as the rows need), each CTA at most 6 blocks of 32 rows | its forward band (`sm_100a`) |
| `fwd_xlong` | the long forward's grid, also prefetching the next block with one bulk prefetch | its forward bands |
| `fwd_pipe` (`sm_100a`, `sm_103a`) | pipelined forward: the deterministic package's forward program (the same build and grid) with a gate that does not flush (`split_gate`): 128 threads, 8 lanes a row, one CTA per 16-row block below 8192 rows per SM, from there whole waves of 6 CTAs per SM, each CTA at most 2 blocks, a thread loading its row of the CTA's next block while it computes the current one, `x` and `z` loaded with the L2 `evict_first` policy | its forward bands |
| `fwd_pipe_packed` (`sm_107a`) | the pipelined forward of packed rows (the deterministic package's packed-rows program on `sm_107a` with a gate that does not flush: the pipelined forward's schedule with four sum chains, `rsqrt.approx.ftz` and one 32-byte store a lane) | its band of `FORWARD_BANDS` (packed rows, fewer than `PACKED_ROWS`) |
| `bwd_bf16_short` | short backward (a BF16 weight gradient; every architecture): one cluster of 16 CTAs of 256 threads, 16 lanes a row, CTA `r` running the 16-row blocks `r`, `r + 16`, ... with the row-claims backward's row arithmetic; CTA `r` adds every CTA's FP32 partials of the columns `8r` .. `8r + 7` in rank order in FP32 (a fixed order), no workspace | the rows of its band (at most 256) |
| `bwd_bf16_t512` (`sm_100a`, `sm_107a`), `bwd_f32_t512` | exact tiny backward: one cluster of 16 CTAs of 4 teams of 128 threads, its FP32 weight-gradient terms summed on chip in plain FP32 words in an order of its own, no workspace | the rows of its band (at most 512) |
| `bwd_bf16_fused` | fused backward (a BF16 weight gradient; every architecture): the deterministic package's fused backward's schedule and row arithmetic (the approximate gate and `rstd`, the next-block L2 prefetch up to the architecture's share of blocks a CTA): 256 threads, 16 lanes a row, 16-row blocks, at most 4 CTAs per SM, which claim the blocks in chunks when there are more blocks than CTAs (`dx` and `dz` the same bits at any grid); each thread's FP32 runs of at most 16 rows added into plain FP32 words, each CTA's FP32 column sums added into 8 FP32 replicas with FP32 atomic adds, the CTA that arrives last summing the replicas in FP32 and rounding the sum once | its backward band, up to 1000 rows per SM (`sm_107a`: past 64, up to 256 rows per SM) |
| `bwd_bf16_rep` | repeatable backward (a BF16 weight gradient; every architecture): the deterministic package's backward program (its build `{"math": "ax", "min_blocks": 4, "out": "bf16", "prefetch_share": None, "threads": 256}`) under this package's name: 256 threads, 16 lanes a row, 16-row blocks, at most 4 CTAs per SM claiming the blocks in chunks whose rows are fixed by the row count and the grid; each chunk's weight-gradient terms added exactly into fixed-point digits in the workspace, the CTA that arrives last rounding the sum once (the deterministic package's `dx`, `dz` and `dw` bits, the same for the rows and the SM count) | its backward bands, past 1000 rows per SM (`sm_107a`: past 512 rows, up to 64 rows per SM, and past 256 rows per SM) |
| `bwd_bf16_rep_long` | the repeatable backward's long program (every architecture): the deterministic package's look-ahead program (its build `{"lookahead": True, "math": "ax", "min_blocks": 4, "out": "bf16", "prefetch_share": None, "threads": 256}`) under this package's name: the repeatable backward in a loop that loads a block's rows while the previous block's row math runs, with its grid, block schedule, FP32 operations in the same order and workspace, so its bits | packed rows of the last repeatable band from `LONG_BACKWARD_ROWS_PER_SM` rows per SM (2560 on `sm_100a` and `sm_103a`, 1536 on `sm_107a`) |
| `bwd_f32` (every architecture), `bwd_bf16` (`sm_107a`) | static backward: 128 threads, 8-row blocks, CTA `b` runs chunks `b`, `b + grid`, ... (the same bits at any grid); on `sm_100a` and `sm_103a` 4 copies of the weight-gradient windows and no prefetch, the CTA that arrives last reading each window by an atomic exchange with 0, on `sm_107a` 4 copies plus occupancy words (the last CTA reads only the windows some CTA added into) | its FP32 backward band (no BF16 band runs `bwd_bf16`; the call role `bwd_bf16` has no program on `sm_100a` and `sm_103a`) |
| `bwd_bf16_rc` | row-claims backward (a BF16 weight gradient; `sm_100a`, `sm_103a`): 256 threads, 16 lanes a row, 16-row blocks, the CTAs past one wave claiming blocks from the claim counter; each thread's FP32 runs of rows added into plain FP32 words, each CTA's FP32 column sums added into 4 FP32 replicas with FP32 atomic adds, the CTA that arrives last summing the replicas in FP32 and rounding the sum | no band |
| `bwd_bf16_rc_mid` | row-claims mid backward (`sm_100a`, `sm_103a`): the row-claims backward with 5 CTAs per SM, thread 0 warming the next block of packed `dy`, `x` and `z` in L2 with one bulk prefetch per tensor; the row-claims backward's FP32 words and 4 FP32 replicas | no band |
| `bwd_bf16_mid`, `bwd_f32_mid` | claimed backward: CTAs claim runs of chunks at run time; on `sm_100a` and `sm_103a` thread 0 warms the CTA's next block in L2 with one bulk prefetch per tensor where `dy`, `x` and `z` hold packed rows (head stride 128 and token stride `H * 128`; other layouts keep the per-thread prefetches), on `sm_107a` it warms the block two ahead | the last FP32 backward band (no BF16 band runs `bwd_bf16_mid`) |

The programs (`ROUTES`; each name is `cake_rmsnorm_silu_gate_train_` and
the 20 hex digits shown):

| role | `sm_100a` | `sm_103a` | `sm_107a` |
| --- | --- | --- | --- |
| `fwd` | `786ee0cda6a87f643bf8` | `786ee0cda6a87f643bf8` | `786ee0cda6a87f643bf8` |
| `fwd_tiny` | `242b29d118997473fc3e` | `242b29d118997473fc3e` | `242b29d118997473fc3e` |
| `fwd_short` | `7b542c846911aca66512` | `7b542c846911aca66512` | `7b542c846911aca66512` |
| `fwd_mid` | `33ef19a28735b3e470b3` | `33ef19a28735b3e470b3` | `33ef19a28735b3e470b3` |
| `fwd_long` | `0dc2a8df1f94d878ccbb` | (not routed) | (not routed) |
| `fwd_xlong` | `ff8010718cbdf7cfa830` | `ff8010718cbdf7cfa830` | `ff8010718cbdf7cfa830` |
| `fwd_pipe` | `e6e0bf4c758851cabc99` | `e6e0bf4c758851cabc99` | (none) |
| `fwd_pipe_packed` | (none) | (none) | `9cb071b2dfe75474aaa7` |
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
| `bwd_f32` | `421e73578399336342e3` | `421e73578399336342e3` | `a8cfbb391df971a9cb59` |
| `bwd_f32_mid` | `f2f608aa8d7a81906c9b` | `f2f608aa8d7a81906c9b` | `75a7f19a91e6ec05dffc` |

The forward bands (rows per SM: a call of `rows` rows on a device of `SMs`
SMs runs the first band with `rows <= last * SMs`): `FORWARD_BANDS` where `x`
and `z` hold packed rows (row `n` at element `128 n`, as contiguous
`[T, H, 128]` and `[T * H, 128]` inputs do) and there are fewer than
`PACKED_ROWS` (2^24) rows, `STRIDED_FORWARD_BANDS` otherwise.  The cuts
follow the checkpointed training step (two forwards and the backward) on
every architecture: the pipelined forward loads `x` and `z` with the L2
`evict_first` policy, so where the rows fit in L2 it is faster alone but
makes the step slower.  On `sm_103a` the mid forward up to 240 rows per SM
and the extra-long forward up to 600 make the step 4 % .. 10 % faster than
the pipelined forward, which alone is at most 4.5 % faster.  On `sm_107a`
the pipelined forward runs packed rows past 280 rows per SM, where alone it
is about 10 % faster than the extra-long forward and makes the step at most
about 3.5 % slower.  Other layouts run the plain forward (`fwd`) past the
persistent forwards' bands: where measured there (from 1000 rows per SM on
`sm_100a` and `sm_103a`, from 520 on `sm_107a`) it is the fastest forward
alone and makes the step 1 % .. 6 % faster; no band of other layouts runs a
pipelined forward.  The tiny and the short forward end where beside a kernel
holding 16 SMs they still end no later than the deterministic package's
forward.  The bands:

| architecture | packed rows (rows per SM) | other layouts (rows per SM) |
| --- | --- | --- |
| `sm_100a` | 1-38 `tiny`, 39-85 `pipe`, 86-160 `mid`, 161-400 `long`, 401-520 `xlong`, 521-6500 `pipe`, past 6500 `mid` | 1-12 `short`, 13-140 `mid`, 141-400 `long`, 401-700 `xlong`, past 700 `fwd` |
| `sm_103a` | 1-36 `tiny`, 37-240 `mid`, 241-600 `xlong`, 601-6000 `pipe`, 6001-8192 `mid`, past 8192 `pipe` | 1-12 `short`, 13-250 `mid`, 251-700 `xlong`, past 700 `fwd` |
| `sm_107a` | 1-65 `tiny`, 66-175 `mid`, 176-280 `xlong`, past 280 `pipe_packed` | 1-59 `short`, 60-480 `mid`, past 480 `fwd` |

The backward bands (`BACKWARD_BANDS`, in order: a call of `rows` rows on a
device of `SMs` SMs runs the first band it fits, a small-row band while
`rows` is at most its count of rows (256 for `bwd_bf16_short`, 512 for the
exact tiny backwards), another band while `rows <= last * SMs` (rows per SM),
the last band always; a call with `compute_grad_weight=False` takes the BF16
bands).
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
them) run the repeatable backward's long program `bwd_bf16_rep_long` in its
place; for an FP32 weight
gradient a static backward runs at
most 4 row blocks a CTA of its one-wave grid (256 rows per SM; on `sm_107a`
one wave at most) and the claimed backward runs past it:

| architecture | BF16 weight gradient | FP32 weight gradient |
| --- | --- | --- |
| `sm_100a` | 1-256 rows `bwd_bf16_short`, 257-512 rows `bwd_bf16_t512`, up to 1000 rows per SM `bwd_bf16_fused`, past that `bwd_bf16_rep` | 1-512 rows `bwd_f32_t512`, up to 256 rows per SM `bwd_f32`, past that `bwd_f32_mid` |
| `sm_103a` | 1-256 rows `bwd_bf16_short`, up to 1000 rows per SM `bwd_bf16_fused`, past that `bwd_bf16_rep` | 1-512 rows `bwd_f32_t512`, up to 256 rows per SM `bwd_f32`, past that `bwd_f32_mid` |
| `sm_107a` | 1-256 rows `bwd_bf16_short`, 257-512 rows `bwd_bf16_t512`, up to 64 rows per SM `bwd_bf16_rep`, up to 256 rows per SM `bwd_bf16_fused`, past that `bwd_bf16_rep` | 1-512 rows `bwd_f32_t512`, up to 64 rows per SM `bwd_f32`, past that `bwd_f32_mid` |

For example, the FP32 static band holds 513 to 37888 rows on a 148-SM B200
and 513 to 13568 rows on a 212-SM R200, and the BF16 fused band 513 to
148000 rows on a 148-SM B200 and 13569 to 54272 rows on a 212-SM R200.

The chunked backwards (static and claimed) cut the rows into 8-row blocks and
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
The repeatable backward and its long program launch the same grid of
256-thread CTAs (18176 bytes of dynamic shared memory each), which claim the
blocks in chunks whose rows are fixed by the row count and the grid, so
their bits do not depend on which CTA runs a chunk.
A small-row backward launches one cluster of 16 CTAs whatever the rows: 512
threads each for the exact tiny backward (19584 bytes of dynamic shared
memory each), 256 threads each for the short backward (6144 bytes each).

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
backwards need a workspace, also with `compute_grad_weight=False`; the small-row backwards
read none, so a call
in a small-row band uses no workspace.  A workspace is a zero-filled,
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

The repeatable backward lays three, 18952 bytes in all (the deterministic
package's layout):

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
| `bwd_bf16_rep`, `bwd_bf16_rep_long` | `acc`, `flags` and `done`, 18952 bytes | `acc`, `flags` and `done`, 18952 bytes |
| `bwd_bf16` | (no program) | 5 slots (4 window copies, then the occupancy words), 36360 bytes |
| `bwd_f32` | 4 slots, 29192 bytes | 5 slots (4 window copies, then the occupancy words), 36360 bytes |
| `bwd_bf16_rc` | `dw_sums` and `dw_done`, 2056 bytes | (no program) |
| `bwd_bf16_rc_mid` | `dw_sums` and `dw_done`, 2056 bytes | (no program) |
| `bwd_bf16_mid`, `bwd_f32_mid` | 4 slots, 29192 bytes | 4 slots, 29192 bytes |

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
`_rc_mid` and the claimed `_mid` backwards), kept for the process.  Eager calls without a
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
  (`FORWARD_BANDS`, `STRIDED_FORWARD_BANDS`, `BACKWARD_BANDS`,
  `module_for_rows`), the backward's chunks and grid (`backward_chunks`,
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
  grids, `torch.use_deterministic_algorithms` (the alert in the fused band,
  no alert elsewhere).
* `tests/norm/test_cake_rmsnorm_silu_gate_nondet_train_api.py` -- the public
  module: no mode keyword, the alerts, the facade and the backend, the
  workspace refusals, and each output judged against the PyTorch BF16 naive
  implementation around the forward's and the backward's bands and on the
  paths of the public module; no comparison with the deterministic
  package.
* `tests/norm/test_cake_rmsnorm_silu_gate_nondet_train_stall.py` -- the speed
  beside kernels resident on other streams.
* `benchmarks/bench_cake_rmsnorm_silu_gate_nondet_train.py` -- timing.
