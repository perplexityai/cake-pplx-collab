# Cake deterministic GDN chunked training backend (SM100 / SM103 / SM107)

## Mode and dispatch rules

- This package is the deterministic (`det`) route only.  Its public names
  carry the mode word: `cake_gdn_det` (the autograd entry),
  `CakeGDNDetFunction` (its autograd class), `cake_gdn_det_train_forward` and
  `cake_gdn_det_train_backward` (the low-level chain over a chunk table and a
  tape).  The package exports no other name; the planner and table helpers
  stay in `cake_chain`.  The facade `cake_pplx.cake_gdn_det_train` exports
  `cake_gdn_det`, `CakeGDNDetFunction` and the support query
  `is_cake_gdn_det_train_supported`; the low-level chain stays in this
  package because it takes the package's chunk table and tape.
- The entries take no `deterministic` keyword.  Passing one, with any value,
  raises `TypeError`.
- `torch.use_deterministic_algorithms(True)` changes nothing: every launch is
  fixed-order, so no alert is raised and the outputs and gradients are the
  same bits as without it.
- The non-deterministic route is a separate package.  Its results are not
  bitwise equal to this package's.  Nothing here routes to it.
- The forward tape carries this package's tag, `"mode": "det"`.
  `cake_gdn_det_train_backward` refuses any other tape with `ValueError`; a
  tape of the non-deterministic package carries `"mode": "nondet"`.

Experimental backend for the forward and backward of packed GDN (scalar-gated
DeltaNet) layers in training: head dimension 128, chunk 64, grouped Q/K heads
(`HV` a multiple of `H`), no context parallelism.  Every launch is a plain
stream-ordered kernel: no atomic on a value (the wavefront pair below counts
stored tiles, and the package tail counts the CTAs of a tile, with atomic
integer counters, whose order no value depends on), no
data-dependent ordering, so the outputs and every gradient repeat bit for bit
from run to run for one layout and one plan.  Some stages run as two groups of sequences on the caller's stream and a
side stream; the groups write disjoint outputs, so this holds with one or two
streams.

The facade `cake_pplx.cake_gdn_det_train` is a stable API: its entry points
are `@cake_api` functions.  This package, under `cake_pplx._backends`, is its
backend.
SM100 (B200 / GB200), SM103 (B300 / GB300) and SM107 (R200) are the supported
architectures; another device raises `RuntimeError`.  The SM107 libraries need
CUDA 13.5 or newer.  The feature does not
participate in automatic backend selection, autotuning or trace-apply.  Its
libraries are loaded prebuilt from `aot/` when a builder compiled them and
JIT-built by the package's JIT shim (`_cake_pplx_jit.py`) from the sources and
flags `build.json` records otherwise.  The package reads no environment variable.

## Opt-in

Import the entry point explicitly from the facade
`cake_pplx.cake_gdn_det_train` (it exports `cake_gdn_det`,
`CakeGDNDetFunction` and `is_cake_gdn_det_train_supported`, and forwards every
argument to this package); nothing else in `cake_pplx` imports this package.

```python
from cake_pplx.cake_gdn_det_train import cake_gdn_det

o, final_state = cake_gdn_det(
    q, k, v, g, beta,                    # [B, T, H, 128] / [B, T, HV, 128] bf16; g, beta [B, T, HV]
    scale=None,                          # 1 / sqrt(128) when None
    initial_state=None,                  # [N, HV, 128, 128] fp32 or None
    output_final_state=False,
    cu_seqlens=cu_seqlens,               # packed batch (B == 1): int32 / int64 offsets on the device
    cu_seqlens_cpu=cu_seqlens_cpu,       # the same offsets on the host: host-table route; None: device route
    use_qk_l2norm_in_kernel=True,
)
```

Without `cu_seqlens` the `B` sequences have length `T` each.  `g` is
computed in FP32; `beta` is FP32 or BF16 and its gradient comes back in its
dtype.  For a packed batch, `cu_seqlens_cpu` selects the route that builds the
chunk and segment tables on the host; without it the `meta` kernel builds
them on the device.  The device route plans its scans for a bound layout, so
the two routes can pick different plans and agree to rounding, not bitwise.

The entry point checks its inputs before the first launch and raises
`ValueError`: every tensor on the device of `q`, `k` of the shape of `q`,
`v`, `g` and `beta` over the tokens of `q`, `HV` a positive multiple of `H`,
`initial_state` of shape `[N, HV, 128, 128]`, `cu_seqlens` a 1-D int32 / int64
tensor with a batch of 1, and `cu_seqlens_cpu` with as many offsets, starting
at 0, not decreasing and ending at the tokens of `q` (an empty sequence is
allowed).  These checks read tensor metadata only; the host offsets are
checked when their layout is first planned.  The device route trusts the values
of `cu_seqlens`, since reading them would synchronize the device.

## Chain

The forward runs the prep (Q/K L2 norm, gate cumsum), the tcgen05 WY stage and
the forward scan fused with the output (`fwd_ho`).  The prep stages the Q/K
tiles by TMA with one CTA per (chunk, key head) (`prep_tma`), except where that
grid needs a second, short wave: with `slots = 6 x SMs` and
`CTAs = chunks x H`, the plain prep runs for `slots < CTAs <= 2.5 x slots`
(`sm_100a`, `sm_103a`; `sm_107a` has no plain-prep band and runs `prep_tma`
for every grid), and
for Q or K not 16-byte aligned.  Q and K that are the head views of a packed
row (strided, row-major views that `prep_tma` reads in place) run `prep_tma`
inside that band too, so the forward copies no Q / K; a `prep` pin still
decides.  Both preps give the same bits.  The WY stage
runs `wy_tc2`: the chains and outputs of `wy_tc` (the same bits) with a
64-column TMEM accumulator and packed shared-memory operands, so five CTAs fit
on an SM where `wy_tc` fits four.  A long sequence can be
split into segments: a pre-scan and a stitch then give every segment its start
state.  On `sm_107a`, a forward scan of one launch whose `segments x HV` CTAs
take at most half the SMs runs `fwd_ho_vh2`: two CTAs per segment and value
head, each the scan and output of one half of the value columns (the same
bits).  The backward runs the adjoint scan, either `dv_local` plus a register
scan or, on a serial plan, the tcgen05 scan with `dv_local` fused (`dhu_tc2`),
which a host-side cost rule selects on the host-table route.  On that route a
split `(64, 1)` plan runs the second pass of its adjoint scan on `dhu_tc2`,
from the stitched segment states, where a second cost rule prices it at most
0.95 of `dv_local` plus the register pass and, on `sm_100a`, at least 40 us
cheaper; its pre-scan then computes the `dv_local` tiles of its chunks itself,
so no `dv_local` stage runs.  Both forms give the same bits on `dhu_tc2`;
every value-head group but two (`G = HV / H`) runs `dhu_tc2_dvl32` instead on
the architectures of `DVL32_ARCHS` (below), and at `G = 3` there
`dhu_tc2_dvl32_gm` forms the segment maps that the stitch composes in place of
the pre-scan (on `sm_107a` only where its grid is larger than half the SMs).
Then the tail computes `dq`, `dk`, `dv`, `dg` and `dbeta`:

| `G` | tail | tcgen05 adjoint scan |
|---|---|---|
| 1 | the package tail (`tail` / `tail_bf16`) | `dhu_tc2_dvl32` |
| 2 | `tail_g2`, a two-CTA cluster kernel (or the wavefront pair below) | `dhu_tc2` |
| 3 | `tail_g3_f32x`, a three-CTA cluster kernel that rounds `dq` / `dk` once | `dhu_tc2_dvl32` |
| 4 or more, at least `PACKAGE_TAIL_MIN_TOKENS_PER_SEQ` tokens per sequence | the package tail | `dhu_tc2_dvl32` |
| 4 or more, fewer tokens per sequence | `dqkwg`, the WY pullback and the finalize | `dhu_tc2_dvl32` |

A `finalize` pin runs `dqkwg`, the WY pullback and that finalize for every
group but two.  The tokens per sequence are the call's mean: its tokens divided
by its sequences (`len(cu_seqlens) - 1`, or `B`).  Empty sequences count, so
repeated offsets in `cu_seqlens` lower the mean.  Both counts are host facts
on both table routes, so both routes, an eager call and its CUDA graph capture
take the same tail.

`PACKAGE_TAIL_MIN_TOKENS_PER_SEQ` is 8 on `sm_100a` and `sm_103a` and 32 on
`sm_107a`.  Each value is the smallest mean tokens per sequence from which no
backward on the tail it picks ran slower than 0.2.3 (the unfused tail on
`dhu_tc2`) beyond the noise band.  The sweep ran `G = 4` and 8, sequences of
4, 8, 16, 24, 32, 48 and 64 tokens over 4096, 16384 and 32768 (chunk, value
head) CTAs and mixed layouts, the backward and the checkpointed step, eager
and CUDA graph, on B200 (`sm_100a`, 148 SMs), GB300 (`sm_103a`, 152 SMs) and
R200 (`sm_107a`, 212 SMs).  On `sm_100a` and `sm_103a` the package tail ran
slower at `G = 4` over sequences of 4 tokens (by up to 1.9 %, most at 32768
CTAs).  On `sm_107a` it ran slower at `G = 4` up to 24 tokens per sequence,
at 16384 and 32768 CTAs (at 24 tokens on each GPU's own band).  Below each
value the package tail still ran faster than the unfused tail on some
layouts, at `G = 8` and at 4096 CTAs among them; one threshold cannot keep
those gains and drop every loss.

The package tail runs one tcgen05 kernel per (chunk, value head), with no
cluster, at every size.  At `G = 1` each CTA writes its `dq` / `dk` tile.  At
`G >= 4` each CTA writes its FP32 `dq` / `dk` partials to its own workspace
slot and adds one to the 32-bit arrival counter of its (chunk, key head) tile,
an integer increment that wraps to zero at `G`; the CTA that completes the
tile sums the group's slots in value-head order, its own included, and rounds
`dq` / `dk` to BF16 once.  The sum does not depend on which CTA completes the
tile, and `dv`, `dg` and `dbeta` have one writer each, so the tail repeats its
bits run to run.  At `G >= 4` the tail launches the chunk table in segments
of at most `TAIL_WORKSPACE_TILES // HV` chunks (4096 slots; at least one
chunk), in stream order; at `G = 1` it is one launch that uses no slot.

The workspace holds one 64 KiB FP32 slot per (chunk, value head) of a launch
and one counter per (chunk, key head).  The slots are never zeroed or
filled, also not under `torch.use_deterministic_algorithms(True)` (the chain
turns `torch.utils.deterministic.fill_uninitialized_memory` off around their
allocation only): the CTA that completes a tile reads only the slot rows of
the chunk's tokens, which the tile's CTAs write in that launch.  The counters
are created zero and every launch leaves them zero.  With `host_cache=True`
the chain keeps one workspace per (device, stream of the call), grown to the
largest call and not released: up to 256 MiB of slots (4096) once a call runs
the package tail at `G >= 4` with `chunks x HV >= 4096`, and one slot (64 KiB,
not read) after calls at `G = 1`.  Calls below the threshold take no slots.
Calls on different streams take different workspaces.  An uncached call takes
a fresh workspace.  A CUDA graph capture takes its own from the graph's memory
pool and zeroes its counters inside the graph (the slots are not filled);
replay one captured graph at a time, not concurrently with itself.

For a value-head group of three, on `sm_100a`, `sm_103a` and `sm_107a`, the
fused tail rounds `dq` and `dk` once: its three CTAs exchange the group's FP32
`dq` / `dk` partials over distributed shared memory and sum them in value-head
order before one BF16 rounding (`tail_g3_f32x`; `tail_g3` rounds each value
head's partials before their sum).  The tail writes `dq`, `dk` and `dv`
through their token-row strides, so when `q`, `k` and `v` are the head views of
one packed BF16 row that requires grad (`row.split((H * 128, H * 128, HV *
128), -1)`, each viewed as `[B, T, heads, 128]`), the row is the autograd input
and the backward writes one packed gradient row, with no concatenation and the
same bits as on the views.  The tcgen05 adjoint scan of such a call adds the
FP32 `dv_local` values to `dv2` before its one rounding (`dhu_tc2_dvl32`), on
serial plans and in the second pass of split ones.
Both are fixed-order with one writer per element, so the gradients repeat bit
for bit; they can differ from the bits of `tail_g3` on `dhu_tc2`, which a group
of three runs with `F32X_TAIL_ARCHS` and `DVL32_ARCHS` off (below).  The
register adjoint forms still round `dv_local` to BF16
first, so for such a call they give the same `o` and final state as
`dhu_tc2_dvl32`, but other bits of `dk`, `dv`, `dg`, `dbeta` and the
initial-state gradient: the cost rules and the `dhu_scan`
pin pick the bits, and on a split `(64, 1)` plan the device-table route always
runs the register pass.  Groups of two keep `tail_g2` and `dhu_tc2`.

In the second pass of a split plan of a value-head group of three on
`dhu_tc2_dvl32`, on `sm_100a`, `sm_103a` and `sm_107a` (`ADJOINT_MAP_ARCHS`),
the segment maps that the stitch composes come from `dhu_tc2_dvl32_gm` instead
of the register pre-scan: one CTA per segment and value head walks two states
in one pass with the `dhu_tc2_dvl32` scan (one from zero with `do`, one from
the identity without it) and stores the segment's map.  It skips the first
segment of each sequence, whose map the stitch does not read, and uses the
pre-scan's warm-up threshold: a segment whose gate sum falls below the
threshold before its last chunk keeps only the chunks up to there.  A map whose
bound falls in its last chunk is kept whole, where the pre-scan keeps only
those chunks (other bits there).  The maps carry the FP32 `dv_local` values, so
where a map carries state into the segment before it (slowly decaying gates)
`dk`, `dv`, `dg`, `dbeta` and the initial-state gradient take other bits than
with the pre-scan; `o`, `dq` and the final state keep theirs, and the gradients
repeat bit for bit.  The segment table, the stitch, the second pass, the tails
and the route rules are unchanged.

Where the maps' grid of one CTA per entry and value head fills at most half of
the SMs (`2 x entries x HV <= SMs`), `dhu_tc2_dvl32_gm2` stores them instead:
two CTAs per entry and value head (grid z 2), `B` and `seg_flag` on one and `M`
on the other, each with the products, roundings and orders of
`dhu_tc2_dvl32_gm` on its half, so the maps and every output keep their bits.
The rule (`split_map_launch`) reads host facts only: the entries come with the
host table, and the most entries of the two-CTA form is decided once per
architecture, value-head count and SM count.  A grid of more than one wave
keeps `dhu_tc2_dvl32_gm`.  On `sm_107a` (`ADJOINT_MAP_ONE_WAVE_PRESCAN_ARCHS`)
a call with `2 x entries x HV <= SMs` keeps the fused pre-scan instead of
either map form, as in 0.2.7.

In 0.2.8 the segment maps of a split pass 2 at `G = 3` on `dhu_tc2_dvl32` come
from `dhu_tc2_dvl32_gm` or `dhu_tc2_dvl32_gm2` (above), where 0.2.3 to 0.2.7
ran the fused pre-scan.  Where a map carries state into the segment before it,
`dk`, `dv`, `dg`, `dbeta` and the initial-state gradient of such a call take
other bits than in 0.2.7; `o`, `dq` and the final state keep them, and every
other route keeps the 0.2.7 bits.  On `sm_107a`, a call with
`2 x entries x HV <= SMs` runs the fused pre-scan, the 0.2.7 route, and keeps
the 0.2.7 bits.

In 0.2.7 the package tail and `dhu_tc2_dvl32` serve `G = 1` and `G >= 4`,
where 0.2.3 to 0.2.6 ran the unfused tail on `dhu_tc2`.  Groups of two and
three keep the 0.2.6 bits on every route; `o` and the final state keep them
everywhere.
At `G = 1` and on the package tail at `G >= 4`, `dq`, `dk`, `dv`, `dg`,
`dbeta` and the initial-state gradient may change where the tcgen05 adjoint
runs, and only `dq` / `dk` may change where the register adjoint runs.  Below
the tokens per sequence of `PACKAGE_TAIL_MIN_TOKENS_PER_SEQ` at `G >= 4`, `dk`,
`dv`, `dg`, `dbeta` and the initial-state gradient may change where the tcgen05
adjoint runs, and nothing changes where the register adjoint runs.
Determinism is run to run within one version: each version repeats its own
bits, and the bits may change between versions.

The switches are module constants of `cake_chain` with the off value `()`:
`PACKAGE_TAIL_ARCHS` (off: the unfused tail on the same adjoint scan at `G = 1`
and `G >= 4`), `DVL32_ARCHS` (off: `dhu_tc2` for every group, `G = 3`
included), `F32X_TAIL_ARCHS` (off: `tail_g3` at `G = 3`), `ADJOINT_MAP_ARCHS`
(off: the fused pre-scan forms the segment maps at `G = 3`, the 0.2.7 route
with its bits), `ADJOINT_MAP_SPLIT_ARCHS` (off: `dhu_tc2_dvl32_gm` on every
grid that takes a map form, the same bits; on `sm_107a` a grid that fits one
wave still keeps the fused pre-scan while `ADJOINT_MAP_ONE_WAVE_PRESCAN_ARCHS`
is on) and `ADJOINT_MAP_ONE_WAVE_PRESCAN_ARCHS` (off: the maps on every grid,
`dhu_tc2_dvl32_gm2` where it fits one wave, with the bits of the maps).  With
the first two off, `G = 1` and `G >= 4` run their
0.2.6 route, the unfused tail on `dhu_tc2`, with its bits.  `PACKAGE_TAIL_MIN_TOKENS_PER_SEQ` maps an
architecture to its threshold; an architecture missing from it, or 0, has
none.

On the host-table route, a call with FP32 `beta`, `use_qk_l2norm_in_kernel`,
`HV <= 3 * H` and no `prep`, `scan_config` or `segment_chunks` pin runs one of
two forwards on the A of `wy_a`, the WY A producer (A, the Q/K norms, the gate
cumsum and the normalized Q/K in one persistent kernel, one CTA per SM over
the (chunk, key head) items in a fixed order):

- On `sm_100a`, `sm_103a` and `sm_107a`, a call whose forward segment table is
  serial runs `wy_a` and `fwd_ho_a`, the forward scan and output on that A
  (u and w formed in the CTA, never in memory).  On `sm_107a`, such a call
  without plan pins keeps a serial forward table where the planner splits one,
  if the serial table prices at most 1.10 times the split table and its
  `sequences x HV` CTAs fill at least 0.65 of the SMs.
- On `sm_100a` and `sm_103a`, a call whose forward segment table splits a
  sequence, with pieces of at least 48 chunks in the host plan's one-wave piece
  table, runs the split forward: `wy_a`, then wave 1 (the first piece of every
  split sequence) beside the affine maps of the middle pieces on a side stream
  (joined before the stitch), the stitch, and wave 2 (every other piece from
  its stitched start state).  The waves run `fwd_ho_a` with `fwd_summary_a`
  (on `sm_103a` for pieces of at least 96 chunks), else the WY stage on the A
  of `wy_a` with `fwd_fused` and `fwd_summary`.  The piece length is the
  smallest one whose first wave fits one CTA per SM; it depends on the
  sequence lengths, `HV` and the SM count only.

In 0.2.5 `wy_a` and `fwd_ho_a` are built from newer kernel source under the
same program names: in `wy_a` each compute warp owns the ring slots of its
parity, and `fwd_ho_a` loads the row scales of `beta v` and `beta 2^g kn` in
batches.  Routes, launch records and output bits are those of 0.2.4.

The route is chosen from shapes, dtypes, the architecture, the SM count and the
pins, never from an address or a value: a Q, K or V that a kernel cannot read
in place, or a `g` / `beta` not at a 16-byte aligned base, is copied first.
Both forwards are fixed-order (no atomic, wave 1 and the maps write disjoint
segments), so they are bitwise run to run with one or two streams.  Their
bits differ from the prep forward's, so the outputs and gradients of these
FP32-`beta` host-table calls can differ from those of the same call under a
`prep` pin, which runs the prep forward.  Every other call (a BF16 `beta`, Q/K
not normalized, the device route, a larger value-head group, a `prep`,
`scan_config` or `segment_chunks` pin) runs the prep forward; there, Q and K
that are the head views of a packed row run `prep_tma` inside the plain-prep
band unless a `prep` pin decides (no copy; see the prep above).  The other
options (`dhu_scan`, `finalize`, `warmup_log2`, `lpt`, `streams`, `save_state`,
`host_cache`) do not change the forward route.

The forward keeps for the backward only the state of the architecture's tape
level (`T_full`: `w`, `h` and `v_new`; `u` is read only to rebuild `h` and
`v_new`, so the backward runs the WY stage again only where those are
missing).  A call that records no backward (grad mode off, or no input that
requires grad) and the first pass of a non-reentrant activation checkpoint
(`torch.utils.checkpoint` with `use_reentrant=False`) keep no tape: the forwards
on the A of `wy_a` then store no `w`, `h` or `v_new`.  The saved list keeps its
13 entries with placeholders of the same shape, dtype and device in place of
the state not kept, which the backward never reads.

The autograd forward of a host-table or equal-length call with
`host_cache=True` records its steps once its layout recurs: from the second
forward of a layout in one tape mode, a call records the steps of its class (a
layout, stream, tape mode, set of arguments, input layout (shape, strides,
dtype and 16-byte base alignment of each input) and pattern of inputs given as
one tensor, such as `k` given as `q`) while its record credit allows.  Each
replay earns `1 / FWD_PLAN_PAYBACK` of a recording and each recording spends
one, beyond a budget of `FWD_PLAN_BUDGET` recordings and one more per
`FWD_PLAN_TRICKLE` calls; with `FWD_PLAN_PAYBACK` at least the host time of a
recording over the host time a replay saves, the recordings cost no more host
time than the replays save (for the calls of one thread: calls of two threads
that overlap can add one recording each time).  A call of a layout with no
record, made while the credit holds less than one recording, runs the general
path at once and does not count among its layout's forwards.  A layout used
once is never recorded.  A record holds the call's steps: its allocations, its
takes of the
stream's pooled workspaces, the PyTorch operations it runs between its launches
(input copies and casts, the split forward's index fills, selects and
copies), its launches, its stream switches and the event records and waits
between its streams.  A checkpoint's first pass, which keeps no tape, and its
recompute have separate records.  Later calls of the class replay that
record: their inputs and the
tensors they allocate fill its slots, the pooled workspaces of the stream are
taken again by each replay, as the general path takes them, and every other
argument is a chunk table of the layout, the cached identity walk or a scalar.
A replay launches the same kernels with the same arguments, on the same
streams, in the same order as the general path, so the output, the final
state, the saved tensors and the gradients have the general path's bits (the
tests compare them bitwise); it only takes less host time.  A replay lets each
tensor of the call go after the last step that reads it (one that the side
stream reads, after the join), and the record holds none of them, so a replay
or a recording needs no more device memory than the general path.  A replay
needs the stream's pool tables and side stream, the cached tables and the
module tables and settings the record was made with; otherwise the call runs
the general path and records again.  A call whose steps the record cannot hold
runs the general path, and so do the later calls of its layout and tape mode
(a call whose recording overlapped another package's recording records
again).  These calls always run the general path: a forward inside a CUDA
graph capture (it neither replays nor records), the device route,
`host_cache=False` and a host `q`; `cake_gdn_det_train_forward` too.  A record
keeps the layout's tables and leaves with them; it holds no pooled workspace,
and a workspace eviction leaves it in place.  At most 512 records are kept
(the oldest leaves first).  `cake_chain.FWD_PLAN = False` keeps every forward
on the general path; `cake_chain.FWD_PLAN_GATE = False` records each class at
its first call.

The record gate and the recordings cost host time.  Against 0.2.8, over 15
layouts per GPU (host time of an eager forward call, fresh processes, 95 %
confidence intervals), the first sighting of a layout (the call the gate has
no entry for, which runs the general path) took 1.9 to 4.0 us more on B200
(`sm_100a`; more on 9 layouts, within the noise on 6) and 1.1 to 2.4 us more
on R200 (`sm_107a`; more on 14 layouts, within the noise on 1); on GB300
(`sm_103a`) it was within the noise of about 10 us either way on every
layout.  A recording call took 109 to 205 us more than the 0.2.8 call on
B200, 98 to 225 us more on GB300 and 60 to 119 us more on R200.

On the host-table route, where the `dhu_tc2` adjoint scan (serial, or the second
pass of a split plan) feeds the fused tail of a value-head group of two with a
BF16 `beta` on the caller's stream, and its `segments x HV` CTAs take at most
half the SMs, the backward runs the two as a wavefront pair: the scan counts
each (chunk, value head) tile in a workspace once it has stored that tile's
`dh` and `dv2`, and the tail visits the chunks in the order the scan finishes
them and waits for each count before it reads the tile.  The tail runs on the
side stream once every scan CTA has started.  The pair runs on `sm_103a` and
`sm_107a`, and on `sm_100a` with at least 32 value heads.  On `sm_100a` and
`sm_103a` the pair also runs the second pass of a split plan whose `segments x HV` CTAs
outnumber the SMs, at any value-head count, on a call of at least 393216
(3 x 2^17) tokens x value heads (below that the pair's host cost outlasts the
GPU time it saves): the pre-scan and the stitch run before the fork, and the
tail starts once the last scan CTA of the second round has started.  That
scan is `dhu_tc2_wavefront_split`, whose store warps meet at a barrier every
four chunks and let one thread fence and add the counts of those chunks;
the scan of at most half the SMs keeps `dhu_tc2_wavefront`, whose store
threads each fence every two chunks and add one per chunk (on `sm_107a` its
store warps meet every eight chunks and one thread fences and adds the counts,
as the split pass 2's scan does every four).  A
CUDA graph does not order its two branches, so under a capture the backward
records the separate launches.  The tail reads the values it reads without the pair: the
same bits.

On the host-table route the long sequences start first.  The tables list the
sequences by chunk count, longest first (ties in the caller's order); every
chunk keeps its token offset, the initial-state rows are gathered into that
order and the final state and the initial-state gradient come back in the
caller's order.  The forward scan also walks its segments longest first.  Each
CTA computes what it computes in the caller's order, so the outputs and the
gradients are bitwise the same with `lpt=False`.  The device route keeps the
caller's order.

A serial scan runs one CTA per (sequence, value head) with one CTA per SM, so
its last round can leave SMs idle.  On the host-table route such a scan can
run as two contiguous groups of sequences.  With `n` sequences, `HV` value
heads and `SMs` multiprocessors, `idle(c) = ceil(c / SMs) x SMs - c` counts the
idle slots of `c` CTAs; `k` in `[1, n)` minimizes `idle(k x HV)` (ties to the
larger `k`).  The scan is split only if `idle(k x HV) < idle(n x HV)`; a scan
with `n < 2` or `n x HV <= SMs` is never split.  The groups follow the table
order (longest first unless `lpt=False`).  The adjoint `dhu_tc2` scan of a
value-head group of two runs the first `k` sequences on the caller's stream,
then their fused tail while a side stream scans the rest.  The forward runs
`wy_tc` and `fwd_ho` of the first `n - k` sequences on the side stream; once
that `wy_tc` is done, the caller's stream runs those of the last `k`.  The side
stream is a stream of PyTorch's pool with the device's greatest priority, one
per caller stream; it is forked from the caller's stream with events and joined
before the stage returns (also when a launch raises), so the call can be
captured in a CUDA graph (warm it up first, like any captured call).
`streams=1` keeps every launch on the caller's stream; the bits are the same.

The host-table route caches device tables between calls: a layout's chunk and
segment tables, the split forward's tables, the wavefront pair's tile order and
the identity walk of the forward scan.  A table is written on the stream of the
call that builds it, and
an eager call on another stream waits for that write (an event recorded after
it) before its launches read the table.  A CUDA graph capture does not cache the
tables it builds, since they are written only when the graph replays.  It reads
the cached tables without waiting, so their writes must be done when the
capture starts (`torch.cuda.graph` synchronizes the device first), and keeps
them for the life of the process, since every replay reads them.  The device
synchronization covers only work queued before the capture: while one thread
captures calls of this package, another thread's call on the same device that
builds a table must have its stream synchronized before the capture reads that
layout.  A capture also takes its own workspaces instead of the pooled
per-stream ones.

## Plan pins (keyword-only)

| keyword | default | effect |
|---|---|---|
| `scan_config` | planner | register-scan `(value block, key split)` of both scans: `(32, 2)`, `(32, 1)` or `(64, 1)`; keeps the call on the prep forward |
| `segment_chunks` | planner | segment length in chunks of the segment-parallel scans; `0` forces the serial scan; lengths above 128 (the chunks one pre-scan CTA covers) are not supported: a call whose plan then splits a sequence raises `ValueError`; keeps the call on the prep forward |
| `finalize` | by size | finalize of the unfused tail (`HV != 2 * H`): `"finalize"` or `"finalize_mlp"` (the latter needs `HV == 3 * H`); a pin runs the unfused tail instead of `tail_g3_f32x` or the package tail (`dv`, `dg` and `dbeta` the same bits; `dq` / `dk` rounded per value head) |
| `warmup_log2` | `-10 / ln 2` | warm-up threshold of the pre-scans and of the `dhu_tc2_dvl32_gm` segment maps |
| `dhu_scan` | rule | adjoint scan of a serial plan: `"reg"` (`dv_local` + register scan) or `"tc"` (`dhu_tc2`; `dhu_tc2_dvl32` for `HV != 2 * H` on `DVL32_ARCHS`); also the second pass of a split `(64, 1)` plan on the host-table route (`"tc"` with the fused pre-scan, or at `HV == 3 * H` on `dhu_tc2_dvl32` with the `dhu_tc2_dvl32_gm` segment maps where the rule above names them); `"tc"` gives the bits of `"reg"` on `dhu_tc2`, and on `dhu_tc2_dvl32` the same `o` and final state with other gradient bits |
| `prep` | rule | forward prep: `"plain"` or `"tma"` (`prep_tma`; Q or K not 16-byte aligned still take the plain prep); keeps the call on the prep forward |
| `save_state` | `True` | keep the forward state of the tape level (`w`, `h`, `v_new`) for the backward; `False` recomputes them |
| `host_cache` | `True` | keep the chunk tables, the `dhu_tc2` selections and the per-stream workspaces between calls, the package tail's among them (a CUDA graph capture takes its own workspaces) |
| `lpt` | `True` | host-table route: dispatch the sequences and the forward segments longest first; `False` keeps the caller's order (the same bits) |
| `streams` | `2` | host-table route: `2` lets a serial scan run as two sequence groups on two streams where the rule above splits it; `1` keeps one stream (the same bits) |

## Limitations

- Head dimension 128 and chunk 64 only; `q`, `k`, `v` BF16.
- No context parallelism and no cross-rank state passing.
- `sm_100a`, `sm_103a` and `sm_107a` only (`sm_107a` with CUDA 13.5 or newer).

## Names

The entries carry the mode-word names below.  The former names were removed
with no alias: importing them raises `ImportError`.  The library
names, the program names and the kernels are unchanged.

| former name | name |
|---|---|
| `chunk_gated_delta_rule_det`, `chunk_gated_delta_rule` | `cake_gdn_det` |
| `ChunkGatedDeltaRuleDetFunction`, `ChunkGatedDeltaRuleFunction` | `CakeGDNDetFunction` |
| `gdn_chunk_det_forward`, `gdn_chunk_forward` | `cake_gdn_det_train_forward` |
| `gdn_chunk_det_backward`, `gdn_chunk_backward` | `cake_gdn_det_train_backward` |
