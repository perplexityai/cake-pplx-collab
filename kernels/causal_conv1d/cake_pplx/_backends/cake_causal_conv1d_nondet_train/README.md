# Cake non-deterministic packed causal conv1d + SiLU training backend (SM100 / SM103 / SM107)

This package is the backend of the non-deterministic (`nondet`) packed width-4
depthwise causal convolution + SiLU training routes. It serves two routes: the single-rank route
(`cake_pplx.cake_causal_conv1d_nondet_train`) and the context-parallel (KCP)
route (`cake_pplx.cake_causal_conv1d_nondet_train_kcp`, module `kcp` of this
package). The facades are the stable APIs; this package is their backend.
Nothing else imports it; a facade imports it on first use. The package reads no
environment variable. The deterministic backend is
`cake_pplx._backends.cake_causal_conv1d_det_train` (facades
`cake_pplx.cake_causal_conv1d_det_train` and
`cake_pplx.cake_causal_conv1d_det_train_kcp`).

## Context-parallel (KCP) route

Every rank convolves its fragments of a packed batch. A fragment can continue a document. Its first three tokens then read the three document tokens before it (history rows) from the ranks that hold them.
The backward sends the derivative of those first tokens back to the
predecessor fragments. They add it to their last rows' `dx` before the one
BF16 rounding. Tokens never move otherwise: there is no activation
concatenation and no full-sequence gather. The route keeps the deterministic
KCP route's FP32-tap contract: FP32 taps, no bias, FP32 arithmetic, one BF16
rounding per output, and the SiLU derivative taken at the FP32 preactivation.

The following parts are the deterministic KCP package's:

- the context contract
- the two-launch forward (see The forward)
- the forward, history-row and head programs (this package's copies of the deterministic programs)
- the halo exchanges
- the recompute handle
- graph replay
- the multi-tensor calls

Of the programs, only the backward after the head program differs: it overlaps the halo exchange, and its `dweight`
sum is unordered.

- **Halo exchange:** one `all_to_all_single` per direction and call. The
  forward delivers the history rows into the history pool `hist`
  (`[n_hist_rows, C]` BF16, saved for the backward). The backward delivers
  the derivatives of the first tokens as `dpre_tail` (`[n_tail_rows, C]`
  FP32). With one rank no collective is issued. The single call sends its
  history rows as a slice of `x_local` when they are consecutive rows, and
  outside a CUDA Graph capture its forward exchange is a waiting collective
  (`async_op=False`).
- **Devices:** sm_100a (B200 / GB200), sm_103a (B300 / GB300) and sm_107a
  (R200). Another device raises `RuntimeError`.
- **Libraries:** load prebuilt from `aot/` when present; otherwise they are
  JIT-built from `build.json`.

### Entry points

```python
from cake_pplx.cake_causal_conv1d_nondet_train_kcp import (
    cake_causal_conv1d_nondet_kcp,          # autograd, one tensor
    cake_causal_conv1d_nondet_kcp_multi,    # autograd, n tensors of one layer (q / k / v), one exchange per direction
    CakeCausalConv1dNondetKcpRecomputeHandle,
)

y = cake_causal_conv1d_nondet_kcp(x_local, weight, ctx)                    # overlapped backward
q, k, v = cake_causal_conv1d_nondet_kcp_multi((xq, xk, xv), (wq, wk, wv), ctx)
```

| operand | shape | dtype | notes |
|---|---|---|---|
| `x_local`, `y`, `dx` | `[T_r, C]` or `[1, T_r, C]` | BF16 | contiguous, 16-byte aligned; the rank's fragments back to back in the partition's order; `C` a multiple of 8 |
| `weight`, `dweight` | `[C, 4]` | FP32 | contiguous, 16-byte aligned; `weight[c, 3]` multiplies the current token; `dweight` is the rank-local partial |
| `hist` | `[n_hist_rows, C]` | BF16 | the forward's history pool, passed back to the backward unchanged; contiguous, 16-byte aligned |
| `ctx` | | | the context of the rank's KDA step (`CakeKDADetKcpContext`, from `cake_pplx.cake_kda_det_train_kcp.cake_kda_det_train_kcp_build_context`): fragment table, halo plan, device tables, process group |

A misaligned operand raises `ValueError` before anything launches; a fresh
tensor or a `.clone()` is aligned.

Each name maps to the deterministic name with `det` in place of `nondet`. A
recompute handle (any object with an `entries` dict) serves either package.
Every rank of `ctx.group` calls the entry once per layer, in the same order,
with the same tensor count and channel counts (see Rank agreement).

The module `kcp` also holds two autograd functions and four explicit calls:

- autograd functions: `CakeCausalConv1dNondetKcpFunction` and `CakeCausalConv1dNondetKcpMultiFunction`
- explicit calls: `cake_causal_conv1d_nondet_train_kcp_forward`, `cake_causal_conv1d_nondet_train_kcp_backward`, `cake_causal_conv1d_nondet_train_kcp_multi_forward` and `cake_causal_conv1d_nondet_train_kcp_multi_backward`

The explicit calls take the keyword `graph="replay"`. It captures each call in a keyed CUDA graph and replays it; a replayed graph does no host work. Call
`cake_causal_conv1d_nondet_train_kcp_graph_cache_clear()` before
`torch.distributed.destroy_process_group()`. The eager call computes its
context-derived host work once per context and reuses it. The cache holds at most 64
contexts, and each entry keeps its context alive.
`cake_causal_conv1d_nondet_train_kcp_host_cache_clear()` empties that cache.

### Mode rules

The module name chooses the package. No entry takes a mode keyword. "Torch
deterministic mode" means `torch.use_deterministic_algorithms(True)`.

| call | torch deterministic mode off | torch deterministic mode on |
|---|---|---|
| this package's forward and checkpoint recompute | run | run (fixed-order); no alert |
| this package's backward (autograd, `cake_causal_conv1d_nondet_train_kcp_backward`, `cake_causal_conv1d_nondet_train_kcp_multi_backward`) | runs; `dweight` is non-deterministic | raises `RuntimeError` before any exchange. With `warn_only=True`, a `UserWarning`, then it runs |
| `cake_pplx.cake_causal_conv1d_det_train_kcp` (the deterministic package) | deterministic | deterministic; no warning |

- The backward reads torch's setting when it runs, including the autograd
  backward. For a given call, every rank of `ctx.group` runs the same entry points with
  the same torch deterministic setting.
- The package never falls back to the deterministic package. For
  deterministic results, call `cake_pplx.cake_causal_conv1d_det_train_kcp`.
- A checkpoint recompute (`torch.utils.checkpoint`, reentrant or not) runs the
  same package.
- `cooperative=True` raises `ValueError`. The cooperative launch form belongs
  to the deterministic KCP package (`cake_causal_conv1d_det_kcp(...,
  cooperative=True)`).

### The forward

The forward runs two programs:

1. The main program reads no received history row. It writes every row, and it writes the first rows of a
   continuing fragment as at a document start. It is the single-rank `fwd` route program: the program that this
   package's single-rank forward runs for the call's `C` and the rank's token count. It takes each of the rank's
   fragments as a document of its own and launches as the single-rank forward does: at a forward row of
   `cake_jit.FALLBACK_ROWS`, `cake_jit.DIRECT_ROWS` or `cake_jit.PAIR_ROWS` (C 512 to 1024, inside the row's token
   range; a `PAIR_ROWS` range first) the static, the direct or the pair direct program, with no counters (a direct
   or pair direct program's grid capped at its token blocks); elsewhere the dynamic program on its record's full
   co-resident grid, which takes its token chunks from zeroed counters (a multi call's tensors share one counter
   buffer, which the static, direct and pair direct programs do not read). It launches without
   programmatic dependent launch (its `enable_pdl` launch scalar is 0), in every mode.
2. The history-row program (`kcp_fwd_halo`) runs after the exchange's wait. It writes `y` of the first
   `min(3, len)` rows of every fragment with a history row again, from the received pool, with the `kcp_fwd`
   program's instructions, the instructions of the one-launch forward (the `kcp_fwd` program with the context's
   history table). The program runs only when the rank receives history rows.

The order depends on the entry and the architecture:

- Single call: on every architecture, both programs run after the exchange's wait (`kcp.FWD_OVERLAP_ARCHS` is
  empty).
- Multi call on sm_107a (`kcp.MULTI_FWD_OVERLAP_ARCHS`) with more than one rank: the first tensor's main program runs
  while the exchange is in flight, and its history-row program runs after the wait. The other tensors run both
  programs after the wait.
- Multi call on sm_100a and sm_103a: every tensor runs both programs after the wait.

The sets are the deterministic package's (`kcp.FWD_MAIN_ARCHS`, `kcp.MULTI_FWD_MAIN_ARCHS`, `kcp.FWD_OVERLAP_ARCHS`
and `kcp.MULTI_FWD_OVERLAP_ARCHS`), measured there against the one-launch forward. The exchanged rows and the
collectives are the same in every form; outside a CUDA Graph capture the single call issues its exchange as a waiting
collective (`async_op=False`), so it has no work to wait. Every forward launch is on the caller's stream. With one
rank no collective is issued.

Outside `kcp.FWD_MAIN_ARCHS` (for the multi entry, `kcp.MULTI_FWD_MAIN_ARCHS`) the main program would be the
`kcp_fwd` program: after the wait, the one-launch forward with the context's history table and no history-row
program; before the wait, with every `hist_row_off` entry `-1`, followed by the history-row program. Every supported
architecture is in both sets, so the shipped forward does not launch the `kcp_fwd` program. Its launch grid follows
the architecture (`kcp.FWD_RESERVE_SMS`): on sm_107a it is one SM short of the co-resident CTA count when that grid
holds every channel tile, and on sm_100a and sm_103a it is the full co-resident grid.

### The backward

The backward rows of `cake_jit.DIRECT_ROWS` (channel ranges within C 512 to 1024 at small S, where the single-rank
backward runs its route's direct program) and of `cake_jit.GRID_CAP_ROWS` (small-S rows on sm_107a, where the
single-rank backward launches fewer CTAs) do not apply to this route: the KCP backward runs the programs below, on
their own grids, at every `C` and `S`.

The head program writes the FP32 derivative of each continuing fragment's
first tokens (the halo adjoints) into the send layout. It matches the
deterministic route's head program. Then:

1. The halo exchange is issued.
2. The main program (`kcp_bwd_dyn`) runs before the exchange is waited, as if
   no fragment had tail rows. Its CTAs take units of tokens from lanes. Through
   Cluster Launch Control, they also take the lanes of CTAs that have not
   started. A kernel resident on another stream therefore leaves no work waiting
   for the SMs it holds. Its token arithmetic computes two adjacent channels
   per packed FP32 (`f32x2`) instruction, as the deterministic KCP backward does.
   The FP32 `dweight` runs are added into four FP32
   accumulator copies in the order the runs end. The copies are summed once per
   channel tile (with `dweight_sum="fp32_pair"`, as an FP32 (hi, lo) pair).
   On the keys of the L1o route table (below), the main program is the
   one-program backward (`kcp_bwd`) instead.
3. After the wait, on a rank that receives tail rows, the tail-row program
   (`kcp_bwd_halo`) writes `dx` of the last three rows of each such
   fragment from the received rows.

With one rank, every launch is on the caller's stream: the main program writes
every row, and the tail-row program writes the tail rows again after it.

With more than one rank, the head program, the concatenation of a multi
call's send rows, the exchange issue, the wait and the tail-row program run on
a high-priority side stream of the device. The main program starts on the
caller's stream at once, because it reads no head row. Off the L1o keys, it
leaves the last rows of each fragment that receives tail rows to the tail-row
program (its `skip_row_off` table is the context's `tail_row_off`). The two
programs therefore write disjoint rows, and the tail-row program runs as soon
as the exchange arrives, beside the main program instead of after it. On a key,
the main program (`kcp_bwd`) writes every row. The side stream then also waits
for the caller's stream, so the tail-row program writes the tail rows again
after it. In a multi call that mixes key and non-key tensors, the side stream
waits once for all main programs. The non-key tensors' tail-row programs then
also run after every main program on the caller's stream: the results are the
same, with less overlap. The caller's stream joins the side stream after the main program. The buffers that the side
stream fills are allocated on the caller's stream before the fork. The programs
and their inputs are the same, and each `dx` row comes from the program that
writes it with one rank.

The main program splits each lane into units of `M` tokens
(`kcp_jit.dyn_plan`). Lane `j` starts at token `j x NL x M` (`NL` units per
lane), and a CTA whose lane is done takes units from the ends of the other
lanes. When 64-token units leave the last lane of each channel tile empty,
`M` is 64, or 56 (`kcp_jit.DYN_IDLE_UNIT`) when 56-token units leave more
idle CTAs (the grid's CTAs less the lanes that have tokens; 64 on a tie):
the idle CTAs take units from the start, and small units balance the lanes
best. Otherwise every lane is busy and `M` is the multiple of 8 from
64 to 128 with the least lane cost: the lane bound `NL x M` tokens plus 3.2
tokens per unit, plus 0.3 x `M` tokens for the last unit, which the other CTAs
finish when an SM is busy. Only the plans whose lanes' claim counts fit the
kernel's 16-bit fields count; when none of them fits (very long sequences),
`M` doubles from 256 until a plan fits. One 10240-channel tensor of 32768
tokens per rank gets 104 on a 148-SM B200 and 64 on GB300, GB200 and VR200;
8192 channels of 16384 tokens get 64 on B200, GB300 and GB200, and 56 on the
212-SM VR200 (16 idle CTAs instead of 4).

The call's context-parallel world size selects a unit table first
(`kcp_jit.dyn_cp_unit`). It covers calls of at most 512 channels:

| World size | Tokens on the rank | `M` |
| --- | --- | --- |
| 2 | at most 4096 | 32 |
| 2 | at least 16384 | 56, unless its lanes' runs (`NL x M` tokens) are longer than with 64-token units |
| 4 | at most 4096 | 32 |
| 8 | at most 2048 | 32 |

Every other call, and a table unit whose plan does not fit the claim counts,
gets the plan above. On B200, GB300 and VR200 the table's units ran faster
than or equal to the plan above on every measured row of the table (world
sizes 2 and 4; world size 8 on B200). 56-token units ran 6-28% slower with
12288 tokens on the 212-SM VR200, where 220 units for 212 CTAs make 112-token
runs instead of 64; the lane-run check keeps such plans out of the table.

With the device's architecture as well, the table has more units on the parts
where they were measured (`kcp_jit.DYN_CP_V2_PARTS`): the 148-SM B200 at world
sizes 2, 4 and 8, and the 152-SM GB300 and 212-SM VR200 at world sizes 2 and 4,
each for at most 512 channels and at most 32768 tokens on the rank. There a
call tries these units in order (`kcp_jit.dyn_cp_units`):

| Device | World size | Tokens on the rank | `M` |
| --- | --- | --- | --- |
| VR200 | 2 or 4 | at least 16384 | 40, when a lane has at most 2 units and its run is no longer than with 64-token units |
| all three | 2 | at least 8192 | 56, unless its lanes' runs are longer than with 64-token units |
| GB300 | 4 | at least 16384 | 56, unless its lanes' runs are longer than with 64-token units |

A call there without a table unit takes 56-token units for idle lanes only
when their runs are no longer than with 64-token units: before, the idle-unit
rule took back the 56-token plan the lane-run check had refused (for example
18432 tokens on B200 and GB300 and 24576 tokens on VR200 at world size 2:
runs of 168 tokens instead of 128). Measured against the 0.2.23 plan on these
parts (backward and checkpointed backward; contiguous, zigzag, three-tensor
and CUDA Graph replay calls; 8 groups each, 10 for replay), no changed row was
slower and the faster backward rows ran at 0.86-0.95 of the time (idle-unit
check), 0.80-0.90 (40-token units), 0.92-0.97 (56-token units from 8192
tokens) and 0.92-0.95 (GB300, world size 4). Every other device, world size,
channel count or sequence length keeps the plan above.

On the keys of the L1o route table (`kcp_jit.L1O_ROUTE`, from 0.2.25) the
overlapped backward runs the one-program backward (`kcp_bwd`) in place of
`kcp_bwd_dyn`: launched before the wait, with every `tail_row_off` entry -1 and
`kcp_jit.L1O_RESERVE_SMS` SMs of its grid left free (B200 3, GB300 10, VR200
none). The tail-row program then waits for it as well as for the exchange.
`dx` is unchanged; `dweight` is summed in the one-program backward's FP32
accumulators. A key is a device, a world size, a channel count and a token
count on the rank:

| Device | World size | Channels: tokens on the rank |
| --- | --- | --- |
| B200 (148 SMs) | 2 | 256: 32768; 512: 4096, 8192, 12288, 16384, 18432, 20480, 24576, 26624, 28672, 32768; 768: 16384, 32768; 1024: 32768 |
| B200 (148 SMs) | 4 | 512: 4096, 16384, 18432, 24576, 32768 |
| B200 (148 SMs) | 8 | 512: 16384, 18432, 32768 |
| GB300 (152 SMs) | 2 | 256: 32768; 512: 4096, 6144, 8192, 12288, 16384, 18432, 20480, 24576, 28672, 32768; 768: 16384, 32768; 1024: 32768 |
| GB300 (152 SMs) | 4 | 512: 4096, 16384, 18432, 24576, 32768 |
| VR200 (212 SMs) | 2 | 512: 14336, 16384, 18432, 20480, 28672, 32768; 768: 16384, 32768 |
| VR200 (212 SMs) | 4 | 512: 4096, 16384, 32768 |

Two runs on different nodes timed the candidate keys: a probe and a
confirmation run. A key is listed when every call form timed on it in both
runs (contiguous and zigzag single-tensor, three-tensor and CUDA Graph replay
calls) met three conditions: its backward ran faster than the better of the
0.2.23 and 0.2.24 plans beyond the noise band, its checkpointed backward ran
no slower than the better of the two plans, and neither its backward nor its
checkpointed backward ran slower than det. A form timed in one run only did not
count. The route takes every call form at a key, also the forms that were not
timed there. The probe alone timed four such cases at listed keys (two forms at
the VR200 key, one form at each of the B200 and GB300 keys):

- VR200, world size 4, 512 channels, 4096 tokens (listed from its three-tensor
  form): the contiguous and zigzag single-tensor forms. With the route, all four
  readings (backward and checkpointed backward of each form) were equal to the
  0.2.24 plan within the noise band. Against det, the contiguous backward was
  slower: 1.09 times det's time (median of 8 groups, 1.01-1.14), where the
  0.2.24 plan was equal to det. The other three readings were equal to det.
- B200 and GB300, world size 2, 512 channels, 8192 tokens: the zigzag
  single-tensor form. With the route its backward was equal to the 0.2.24 plan
  within the noise band (0.90 and 0.92 of its time) and faster than det (0.79
  and 0.74 of det's time); its checkpointed backward was faster than both.

These readings come from one run, and no release gate times these forms.
Every other call keeps the overlapped pair, including
`dweight_sum="fp32_pair"` calls, the 216-SM VR200 and shapes between the keys.

Under any of these conditions, the one-program backward (`kcp_bwd`) runs after the wait instead:

- `bwd_reserve_sms >= 1`
- `C > 65536` (`kcp_jit.DYN_MAX_CHANNELS`, 32 x 2048)
- the call has no tokens

It is one ordinary launch. Its CTAs add their FP32 `dweight` rows into zeroed FP32 accumulators with unordered global reductions. The last CTA of each channel tile stores the rank-local partial and zeroes the tile's accumulators. Neither form uses a cooperative launch or a grid
barrier.

Both programs use the package's cached backward workspace for the device and
current stream. The workspace is zero before and after every launch. Inside a CUDA Graph capture the
package uses a buffer it zeroed before the capture, one per capture and stream (the graph holds it until the graph
is destroyed). Only eager calls that omit `workspace=` and take counters zero these buffers (a forward inside the
token range of a row of `FALLBACK_ROWS` / `DIRECT_ROWS` / `PAIR_ROWS` takes none), the forward's and the backward's
separately, sized to the largest need of such a call whose zeroing succeeded (one that runs out of memory keeps no
new buffer and keeps their size); a capture with no such eager call of the kind before it, or after earlier captures took
the `_POOL_DEPTH` ready buffers, or needing more, or begun before their zero fills completed
(`CUDAGraph.capture_begin` right after such an eager call with no synchronization; `torch.cuda.graph`
synchronizes first), zeroes a buffer per call (a fill node each). So does every captured call when CUDA's capture id is not available. The package keeps no reference to
such a buffer after the call. The package allocates the pre-zeroed buffers and its per-stream workspaces from a
memory pool of its own (`torch.cuda.MemPool`, one per device), so a CUDA Graph tree's warm-up does not take them into
the tree's private pool. Memory the package frees stays cached in that pool (a replaced per-stream buffer, a dropped
pre-zeroed buffer) and is never returned to the caching allocator's other pools. Per device and kind (forward
counters, backward accumulators) the live pre-zeroed buffers are at most `_POOL_DEPTH` idle ones of the pool's size
(the largest need of an eager zeroing that succeeded), those live graphs hold, and those graphs released since the
last refill: a released buffer returns to the package only at the next eager refill, also after
`cake_causal_conv1d_nondet_train_kcp_graph_cache_clear()`. A kind whose refill failed (other than out of memory)
refills no more: its idle buffers stay allocated for the life of the process, and a buffer its graphs held returns,
once released, only at a refill of the device's other kind, which keeps it idle or drops it; a dropped buffer's
memory stays cached in the package's pool. Without `torch.cuda.MemPool` the package keeps no pre-zeroed buffers and
every captured call zeroes its buffer itself.

### Rank agreement

Every rank of a call must pass a context built from the same partition. Each rank must also call the same entry (single or multi) with the same channel counts.
A check of this agreement runs on the first eager exchange of each (context, phase); the phase is forward or backward. It all-gathers one int64[8] fingerprint per rank over `ctx.group` and compares the fingerprints on the host. On a mismatch, every
rank raises the same `ValueError` naming the ranks and fields that differ.
This is the call's only collective beyond the halo exchange and its only host
sync, once per context and phase.

- No check is made by later calls on the same context and phase, by recompute with a handle (no exchange) or by graph replays.
- A first call inside a CUDA-graph capture raises: run one eager call first.
- With one rank there is no check.

The fingerprint fields are: protocol, mode, phase, entry and tensor count,
channels, world size and conv width, a partition hash, and launch form. The
layout is shared with the deterministic KCP package. Field 1 (mode) is 2 here
and 1 there.

### Determinism and accuracy, per output

The non-deterministic package makes no bit promise. No output is promised to
repeat bit for bit, run to run, or to match the deterministic package's bits.

| Output | Accuracy vs exact FP64 |
| --- | --- |
| `y` | one BF16 rounding of the FP32 result |
| `hist` (history pool) | copies of other fragments' `x_local` rows |
| `dx` | within one BF16 ulp of an FP64 oracle |
| `dweight` (rank-local FP32 partial) | an unordered sum of the CTAs' FP32 rows. By default (`dweight_sum="fp32"`) every sum is one plain FP32 word, with no compensated sums; the opt-in `"fp32_pair"` sums a tile's accumulator copies as an FP32 (hi, lo) pair. No FP64 |

The package issues no reduction collective for `dweight`. Each rank returns
its partial, and the framework sums the partials across the ranks (for example, the
data-parallel or FSDP reduction of the weight gradient).

### Keyword controls

There are no environment variables. Every setting is a keyword argument.

| Keyword | Values | Default | Effect |
| --- | --- | --- | --- |
| `recompute_handle` | `CakeCausalConv1dNondetKcpRecomputeHandle()` or any object with an `entries` dict | `None` | The checkpoint recompute runs without the forward halo exchange (see below). |
| `cooperative` | `None`, `False` | `None` | `None` and `False` mean the same. `True` raises `ValueError`. |
| `bwd_reserve_sms` | int `K`, `0 <= K <` SM count | `None` (= 0) | `K >= 1` runs the one-program backward after the exchange with a grid `K x ctas_per_sm` CTAs short. On a B200 / GB300 key of the L1o route table the default already leaves 3 / 10 SMs free before the wait, so `K` adds headroom there only above 3 / 10. Supported on SM100 / SM103 only; `ValueError` on SM107. A `bool` or a non-`int` raises `TypeError`. A value out of range raises `ValueError`. The forward and the exchanges are the same; `dweight` is summed the same way. Each rank may choose its own value. |
| `dweight_sum` | `None`, `"fp32"`, `"fp32_pair"` | `None` (= `"fp32"`) | `"fp32"`: one plain FP32 word per accumulator copy of `dweight`. `"fp32_pair"` is an opt-in: the overlapped backward's main program sums the accumulator copies of a tile as an FP32 (hi, lo) pair and rounds once, with FP32 operations only. It needs the overlapped backward: with `bwd_reserve_sms >= 1` the call raises `ValueError`, and a backward with tokens and `C > 65536` (`kcp_jit.DYN_MAX_CHANNELS`) raises `ValueError`. It never runs another program. Another string raises `ValueError`; a non-string raises `TypeError`. Each rank may choose its own value. |
| `graph` | `"replay"` | omitted (eager) | Explicit `kcp` calls only: keyed CUDA-graph capture and replay. |

### Sharing the GPU with other streams

Off the keys of the L1o route table, the overlapped backward is the default and
needs no setting. Its CTAs claim their tokens at run time.

On a key (`kcp_jit.L1O_ROUTE`), the main program is the one-program backward
(`kcp_bwd`). Its grid is one wave, and each CTA has a fixed share of the
tokens. The grid leaves `kcp_jit.L1O_RESERVE_SMS` SMs free: 3 on B200, 10 on
GB300 and none on VR200. A kernel on another stream that holds more SMs than
that leaves some CTAs without an SM. Those CTAs wait until an SM is free and
run as a second wave, so the backward takes longer. The call does not
deadlock: there is no grid barrier and no cooperative launch. The "previous
backward" column of the table below shows a program of this kind beside such a
kernel (shapes A and B are not keys). No key was timed beside a kernel on
another stream. On a key, a call can set:

- `bwd_reserve_sms=K` (SM100 / SM103 only): the one-program backward runs after
  the exchange's wait with a grid `K` SMs short, so a kernel holding up to `K`
  SMs leaves it in one wave (see below). The default already leaves 3 SMs free
  on B200 and 10 on GB300, before the wait. `K` adds headroom only when it is
  above that number: a smaller `K` gives a larger grid, and an equal `K` the
  same grid. Any `K >= 1` gives up the overlap with the exchange.
- `dweight_sum="fp32_pair"`: the call is never routed, so its main program
  claims its tokens at run time. It also changes how `dweight` is summed (see
  Keyword controls). On SM107 this is the only such setting.

A repro timed the KCP
backward at world 4 with CUDA events, alone and next to a kernel on another
stream. It used two per-rank shapes:

- **Shape A:** 32768 tokens x 10240 channels, one tensor.
- **Shape B:** 16384 tokens x 8192 channels, three tensors.

A one-thread side kernel on GB200 (sm_100a, 152 SMs) barely changes the time:

| Side kernel on another stream | shape A next-to / alone | shape B next-to / alone |
| --- | --- | --- |
| one thread, 2 ms | 0.999 | 0.999 |
| one thread, 4 ms | 0.998 | 1.000 |

The next table shows a resident kernel on another stream that holds `k` SMs. `N` is the SM count. The
ratio columns give the time next to the resident kernel divided by the time alone. The `k = 0` rows give the
unloaded backward time. The previous non-deterministic backward ran as one program after the exchange.

| GPU | `k` | ideal `N / (N - k)` | shape A | shape B | previous backward, A / B |
| --- | --- | --- | --- | --- | --- |
| GB200 (sm_100a, 152 SMs) | 0 | 1 | 360.1 us | 466.0 us | 395.0 us / 560.5 us |
| GB200 | 16 | 1.118 | 1.091 (392.9 us) | 1.183 (551.1 us) | 667.5 us / 921.1 us |
| GB200 | 32 | 1.267 | 1.210 (435.6 us) | 1.284 (598.2 us) | 687.8 us / 914.1 us |
| B200 (sm_100a, 148 SMs) | 0 | 1 | 370.1 us | 486.4 us | 423.9 us / 583.7 us |
| B200 | 16 | 1.121 | 1.159 (428.9 us) | 1.190 (578.6 us) | 738.0 us / 973.8 us |
| B200 | 32 | 1.276 | 1.289 (477.1 us) | 1.320 (642.2 us) | 743.4 us / 975.9 us |

The slowdown follows the share of SMs the resident kernel holds. For shape A on GB200 it stays below the ideal
ratio. For shape A on B200 and for shape B on both GPUs it is at most 0.07 above the ideal ratio.

`bwd_reserve_sms=K` is an opt-in alternative (SM100 / SM103 only). It runs the one-program backward
after the exchange with a grid `K x ctas_per_sm` CTAs short. A kernel holding up
to `K` SMs then does not push the top CTAs into a second wave. The unloaded
backward gives up the work of those `K` SMs. On a B200 or GB300 key of the L1o
route table, the default call already leaves 3 or 10 SMs free and runs before
the wait. There, `K` adds headroom only when it is above 3 or 10, and any
`K >= 1` gives up the overlap with the exchange. On sm_107a it raises
`ValueError`: there, the reserved grid costs up to 1.055x the default grid
unloaded (C 1024, 62585 tokens), which is above the opt-in's 1.040 limit.

### Checkpoint recompute without the halo exchange

With `recompute_handle=CakeCausalConv1dNondetKcpRecomputeHandle()`, the checkpoint
recompute of a call issues no collective:

1. The original forward exchanges as always and stores `hist` in the handle (key `"conv"`). The handle retains `n_hist_rows x C x 2` bytes per tensor.
2. Torch's recompute of that forward inside the backward binds `hist` and launches the forward programs on it (both after the binding, as in the waiting form), with no gather, no collective and no wait.
3. The backward still exchanges once.

The handle has these rules:

- One handle serves one call (single or multi) of one micro-step.
- Create it outside the outermost checkpoint region.
- Every rank passes a handle for the same call or none, at the same checkpoint nesting level. This is not checked: a mismatch hangs or corrupts data.
- A handle is refused with `graph="replay"` and inside a CUDA-graph capture.

### Multi-tensor calls

`cake_causal_conv1d_nondet_kcp_multi(xs, weights, ctx)` convolves `n >= 1`
tensors that share the context, such as a layer's q, k and v. It uses one halo exchange per
direction instead of `n`.

- Inputs: each `xs[i]` is BF16 `[T_r, C_i]` and each `weights[i]` is FP32 `[C_i, 4]`. Each `C_i` is a multiple of 8, and they may differ.
- Exchange: the tensors' send rows travel side by side in one `all_to_all_single`. After the wait, when the call uses the side stream (more than one rank, every tensor on the overlapped backward: the overlapped pair or the L1o route) and every tensor has the same width, each tensor's tail-row program reads its rows in place in the received pool, and the programs run concurrently on the side stream and streams forked from it. Otherwise the rows are copied into each tensor's own pool. When at least one tensor is on an L1o key, the side stream first waits for every main program on the caller's stream, so the other tensors' tail-row programs also run after them (the same results, less overlap).
- Programs: each tensor then runs the single call's programs. In the forward on sm_107a, the first tensor's main program runs before the wait (see The forward). Each `dweight` is that tensor's own unordered partial.
- Keywords: one `bwd_reserve_sms` value and one `dweight_sum` value apply to every tensor. One recompute handle covers the whole call; its `"conv"` entry holds the `n` history pools.
