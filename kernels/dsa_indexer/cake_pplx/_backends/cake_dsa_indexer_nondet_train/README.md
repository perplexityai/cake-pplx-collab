# Cake DSA indexer top-k backend, non-deterministic (SM100 / SM103 / SM107)

Backend package of `cake_pplx.cake_dsa_indexer_nondet_train`. It is the
non-deterministic counterpart of `cake_pplx.cake_dsa_indexer_det_train`: the
fused DSA (sparse attention) indexer of a training stack.  One device-side
pipeline scores every visible key and selects the exact top-k with aligned
scores.  The target geometry is 32 indexer heads, head dimension 128 and
top-k 2048.

This package runs the deterministic package's scan programs of both input
forms (BF16 and the opt-in FP8 form) and its plain split merge unchanged.  It leaves out the ordering pass, which is the
deterministic package's finalize / rank finalize that sorts every row into
ascending id order.  The package implements the set-equality semantics: a
row's selected ids are a set, written in no specified slot order.  No output
is promised to repeat bit for bit between calls.

The entry points of `cake_pplx.cake_dsa_indexer_nondet_train` are stable
APIs.  This backend package under `cake_pplx/_backends/` is
experimental and has no compatibility guarantee of its own.  SM107 (Rubin) is
the primary target of the request.  SM100 (B200 / GB200) is the comparison
architecture and SM103 (B300 / GB300) is the third target.  The `sm_107a`
libraries need CUDA 13.5 or newer (`min_cuda` 13.5 in their `build.json`
entries).  The feature does
not take part in automatic backend selection, autotuning or trace-apply.  Its
libraries are loaded prebuilt from `aot/` when the package ships them.
Otherwise the package's JIT shim builds them from `build.json` (see "Layout of
this package").

## Mode and names

Every public name carries the mode word `nondet`, and no entry point takes a
`deterministic=` keyword.  The deterministic package is
`cake_pplx.cake_dsa_indexer_det_train`.  It runs the same scan programs and
writes every row in ascending id order.  The backend's own names
also carry the mode word (`prepare_dsa_indexer_nondet_topk`,
`DSAIndexerNondetRunner`, `plan_dsa_indexer_nondet_topk`,
`validate_dsa_indexer_nondet_inputs`).  The generated programs are named
`cake_dsa_indexer_nondet_topk_<20 hex>`.

**Deterministic algorithms.**  The guard applies under
`torch.use_deterministic_algorithms(True)`:

* The eager entry `cake_dsa_indexer_nondet_train_topk` raises PyTorch's
  `RuntimeError` only when the call launches (`T > 0` and `Tkv > 0`).  It
  raises after its argument checks and before it allocates or launches
  anything.  The argument checks cover shapes, dtypes, the outputs, the plan,
  the registry lookup of the call's programs, and the size and alignment of a
  caller-owned `workspace_buffer`.
* A call without queries or keys launches nothing and does not alert.
* With `warn_only=True` the call issues a `UserWarning` and then runs.
* A prepared runner (`cake_backend.prepare_dsa_indexer_nondet_topk`) checks
  the setting at the top of every `run()` that launches, before its first
  launch.
* Preparing a runner never alerts.  The workspace-size function
  `cake_dsa_indexer_nondet_train_topk_workspace_size` never alerts either.
* A CUDA graph replay runs no host code and does not alert.  The setting is
  read when a run is captured.
* Nothing falls back to the deterministic package.

## Public entry points (`cake_pplx/cake_dsa_indexer_nondet_train.py`)

```python
from cake_pplx.cake_dsa_indexer_nondet_train import (
    cake_dsa_indexer_nondet_train_topk,
    cake_dsa_indexer_nondet_train_topk_workspace_size,
)

indices, scores = cake_dsa_indexer_nondet_train_topk(q, k, w, cu_seqlens_q, cu_seqlens_k,
                                                     top_k=2048, softmax_scale=None, q_causal_offsets=None, ratio=1,
                                                     max_seqlen_q=None, max_seqlen_k=None,
                                                     workspace_buffer=None, indices=None, scores=None,
                                                     return_scores=True, k_scale=None)
nbytes = cake_dsa_indexer_nondet_train_topk_workspace_size(num_queries, num_keys, num_segments,
                                                           top_k=2048, ratio=1, device=None, dtype=torch.bfloat16)

# the FP8 form (opt-in): float8_e4m3fn q / k, per-key FP32 k_scale, every other scale folded into w
indices, scores = cake_dsa_indexer_nondet_train_topk(q_fp8, k_fp8, w, cu_seqlens_q, cu_seqlens_k, k_scale=k_scale)
nbytes = cake_dsa_indexer_nondet_train_topk_workspace_size(T, Tkv, S, device=q.device, dtype=torch.float8_e4m3fn)

# indices only: None is returned in place of the scores (they are still computed internally)
indices, _ = cake_dsa_indexer_nondet_train_topk(q, k, w, cu_seqlens_q, cu_seqlens_k, return_scores=False)
```

Every argument after `cu_seqlens_k` (and after `num_segments`) is
keyword-only.

| argument | shape and type | meaning |
|---|---|---|
| `q` | `[T, 32, 128]` BF16, contiguous | indexer queries after RoPE |
| `k` | `[Tkv, 128]` BF16 | packed indexer keys after normalization and RoPE; a row-strided view (`packed[:, :128]` of a `[Tkv, 704]` tensor) is read in place, unit inner stride, row stride a multiple of 8 elements |
| `w` | `[T, 32]` FP32, contiguous | signed head weights, already scaled by `32 ** -0.5` |
| `cu_seqlens_q`, `cu_seqlens_k` | `[S + 1]` int32, contiguous, device | independent query and key segment boundaries (`cu_seqlens_q[-1] == T`, `cu_seqlens_k[-1] == Tkv`; not read on the host) |
| `q_causal_offsets` | optional `[S]` int64, contiguous, device | per-segment query offsets |
| `top_k` | int, `1 <= top_k <= 4096` | output slots per query (default 2048) |
| `softmax_scale` | positive finite float | logit scale (default `128 ** -0.5`); `None` for the FP8 form |
| `k_scale` | optional `[Tkv]` FP32, contiguous | the FP8 form's per-key scales (see "FP8 form"); BF16 calls take none |
| `ratio` | int, `>= 1` | key compression ratio (normally 1) |
| `max_seqlen_q` | optional int | host mirror of the longest query segment; accepted, not read by any launch decision |
| `max_seqlen_k` | optional int | the caller's bound on every key segment (`max(cu_seqlens_k[1:] - cu_seqlens_k[:-1])`); read only by the segment-order rule (a packed call of equal key segments runs its scan without the longest-first order); the selected sets are identical with and without it; a value above `Tkv` or below `ceil(Tkv / S)` is rejected |
| `workspace_buffer` | optional uint8 tensor | at least `cake_dsa_indexer_nondet_train_topk_workspace_size(T, Tkv, S, top_k=..., ratio=..., device=..., dtype=q.dtype)` bytes, 8-byte aligned; allocated through the caching allocator when omitted |
| `indices`, `scores` (outputs) | `[T, top_k]` int32 and FP32, contiguous, on the device of `q` | caller-owned outputs, allocated when omitted; see "Outputs"; `scores` must be omitted with `return_scores=False` |
| `return_scores` | bool | `False`: `None` is returned in place of the scores (default `True`) |

The `q` / `k` dtype selects the input form: BF16 (the default) or the opt-in
FP8 form (both `torch.float8_e4m3fn`, see "FP8 form").  Mixed `q` / `k`
dtypes, `float8_e5m2` and every other dtype raise `ValueError`.  A device
other than compute capability 10.0, 10.3 or 10.7 raises `ValueError`.  An
architecture without registered programs of the call's form raises
`NotImplementedError`.

`T`, `Tkv`, the segment lengths and the last tile are arbitrary.  The recorded
training calls have `T = 16231` and `16172`.  Nothing needs caller-side
padding.  `T = 0` returns empty outputs, and a call without keys returns all
padding.

## Outputs

| output | contents | not specified |
|---|---|---|
| `indices [T, top_k]` int32 | segment-local key ids: per row, the `min(top_k, visible)` selected ids, unique within the row, and `-1` in every other slot | the slot order of the selected ids and the positions of the `-1` slots; both may vary between calls |
| `scores [T, top_k]` FP32 | aligned slot by slot with `indices`: the score of the id in the same slot; `-inf` in every padding slot; `None` with `return_scores=False` | the slot order, which follows `indices` |

* A caller that needs ascending id order can sort the row or use the
  deterministic package.  The following rearranges every row into ascending
  ids, padding last:

  ```python
  key = indices.masked_fill(indices < 0, torch.iinfo(torch.int32).max)
  order = key.argsort(dim=1)
  indices, scores = indices.gather(1, order), scores.gather(1, order)
  ```

* A row with no visible key is all padding (`-1` / `-inf`).

## Scoring and the reduction scheme

For query `t` and key `j` of the same segment

```
s[t, j] = sum_h w[t, h] * relu(softmax_scale * sum_d q[t, h, d] * k[j, d])
```

Products, accumulation and score arithmetic are FP32.  There is no BF16
intermediate rounding and no fast-math (IEEE round-to-nearest, no
flush-to-zero).  The scan programs are the deterministic package's BF16 scan
programs.  Every program
evaluates the score in this sequence:

1. **Dot product**: one tensor-core MMA of a fixed shape (128 keys x (queries
   of the unit x 32 heads) x K = 16).  It accumulates the 128 products of a
   (query, head, key) triple into an FP32 accumulator in eight K-steps over
   `d = 0 .. 127`, in ascending order.  The BF16 x BF16 products are exact in
   FP32.
2. **Per-key head reduction** (one thread per key row): `r_h = fl(x_h +
   |x_h|)` (exactly `2 relu(x_h)`).  Four interleaved FMA chains start from
   `+0.0`: `c_m = fma(r_h, w'_h, c_m)` for `h = m, m + 4, ..., m + 28`
   (`m = 0 .. 3`).  Then `s = fl(fl(fl(c_0 + c_2) + fl(c_1 + c_3)) * 0.5)`.
3. **Weights**: `w'_h = fl(w_h * softmax_scale)` once per query block
   (`relu(scale * x) = scale * relu(x)` for `scale > 0`).

Every head term therefore sees at most 11 roundings after the MMA.

**Signed zeros.**  `+0.0` and `-0.0` compare equal for ranking, and an
equal-score tie goes to the larger key id.  The tests hold the signed-zero
cases to the ids and valid counts of the exact selection of
`tests/experimental/test_helpers/cake_dsa_indexer_reference.py` after each row
is canonicalized (see "Tests and benchmark").

## FP8 form

The FP8 form is opt-in and selected by the input dtype: `q [T, 32, 128]` and
`k [Tkv, 128]` both `torch.float8_e4m3fn` (the same layout rules as BF16; a
row stride that is a multiple of 16 elements), `k_scale [Tkv]` FP32
(contiguous, on the device of `q`), `w [T, 32]` FP32 carrying every other
scale (the query scale, the softmax scale and the head weights), and
`softmax_scale=None` (the programs take 1.0).  Mixed q / k dtypes,
`float8_e5m2` and `q_scale` are rejected; a BF16 call keeps its default
behaviour and rejects `k_scale`.  The score is

```
s[t, j] = k_scale[j] * sum_h w[t, h] * relu(sum_d q[t, h, d] * k[j, d])
```

* **Programs**: the FP8 scan programs are the deterministic package's FP8
  scan programs under this package's prefix, and a split call runs the same
  plain split merge as BF16.  `sm_100a` / `sm_103a` register the `narrow`,
  `l6`, `split_narrow` and `split_l6` scan programs, each with its
  longest-first segment order copy (`:lsf`), and the unpaired `wide` program
  (8 queries per unit, tile unroll 2) with its `:lsf` copy; `sm_107a`
  registers `narrow`, `l6` with its snake unit order copy, `split_narrow` and
  the unpaired `wide` program (tile unroll 2), without `:lsf` copies.  Keys
  carry the form tag before the order suffix:
  `scan:<kind>:u<n>:s<0|1>:f1:fp8[:lsf]`, `s1` only for the `sm_107a` `l6`
  snake program.  Launches are non-PDL and non-cooperative, like BF16.
* **Dot product**: the e4m3 x e4m3 products are exact and accumulate in FP32
  on the tensor cores: four K = 32 steps over `d = 0 .. 127` on `sm_100a` /
  `sm_103a` (the order of DeepGEMM's `fp8_mqa_logits`), two K = 64 steps on
  `sm_107a`.
* **Head reduction**: the sum over the heads of `w[t, h] * relu(logit)` and
  its product with `k_scale[j]` are FP32; the tests judge the score like the
  BF16 form (against its FP64 value from the e4m3 inputs, beside the naive
  tier's).
* **Sets**: every FP8 program of one architecture selects the same set per
  row, in no specified slot order, like BF16.  FP8 results are not BF16
  results of the dequantized inputs.
* **Dispatch**: the architecture's FP8 record is its BF16 `POLICY` record with
  the fields of `cake_jit.POLICY_FP8[arch]` replaced
  (`cake_policy.fp8_policy_record`); `POLICY_FP8` names every architecture's
  FP8 `static_scan_kinds` itself, and the record is the deterministic
  package's FP8 record.  On `sm_100a` and `sm_103a` `POLICY_FP8` also sets
  `wide_wave_rule`, so the FP8 form chooses its wide unit by waves of the
  persistent grid (the unit geometry rule in "Host dispatch").
  `cake_policy.select_fp8_program` runs the BF16
  decision procedure under it and maps the decided kind to a registered FP8
  kind: its own where an FP8 program of it is registered, else its fallback
  (`FP8_KIND_FALLBACK`: `wide` / `pair_wide` / `pair_l6` -> `l6`,
  `pair_narrow` -> `narrow`, `split_wide` -> `split_narrow`, and `split_l6`
  -> `split_narrow` where no FP8 `split_l6` program is registered).  The
  program runs without a CTA pair, in its sampled-fit form, in the snake
  order only where the FP8 record's `snake_cap_rule` admits it and the snake
  copy is registered, and in the longest-first order where the BF16 rule
  picks it and an FP8 copy exists, or for one document of the `wide` unit
  within the FP8 record's `lsf_single_rule`.  A fallback kind takes its own
  sampling class and check period (`cake_policy.fallback_knobs`); explicit
  knobs stay.  On `sm_107a` every FP8 kind keeps the static unit enumeration
  (`static_scan_kinds`): no ticket tail, nothing zeroed.  An architecture
  without FP8 programs raises `NotImplementedError`.  A call therefore gets
  the FP8 scan program, candidate capacity, check period and split count that
  the deterministic package picks for it.
* **Workspace**: `cake_dsa_indexer_nondet_train_topk_workspace_size(...,
  dtype=torch.float8_e4m3fn)` bounds the FP8 programs' workspace; the
  default `dtype=torch.bfloat16` returns the BF16 bound.  The FP8 bound can
  differ from the BF16 bound of the same geometry (other units, no CTA pair),
  so size an FP8 buffer with this function and the FP8 dtype.
* **Prepared runners** bind `k_scale` with `q` / `k`: new values written into
  it are read by the next run.

## Visibility

Let `u` be the query's segment-local position, `j` the key's segment-local id
and `Lk` the segment's key count.  Key `j` is visible exactly when

```
0 <= j < Lk   and   j < floor((offset + u + 1) / ratio)
```

`offset` is `q_causal_offsets[s]` when supplied.  Otherwise it is `Lk - Lq`
for `ratio == 1` (queries are the tail of their key prefix) and `0` for
compressed keys (`ratio > 1`).  The floor is a true floor for negative
offsets.  Keys of other segments and keys outside the visible prefix are never
selected.  Negative offsets, empty segments and rows without visible keys are
defined: the row is all padding.

## Selection, ties, output order, padding

* Every row selects the exact global top `min(top_k, visible)` keys of its
  visible prefix.  The internal candidate gate is exact, never a fixed
  candidate margin: it is certified in-kernel, and a unit whose sampled
  threshold proves too aggressive is streamed again exactly.
* Ranking: score descending, then key id descending (the larger id wins an
  equal-score tie).  `+0.0 == -0.0`.
* Output: no specified slot order (see "Outputs").  Ids are unique.  Every slot
  that does not hold a selected entry holds `id = -1`, `score = -inf`.
* The following never change the selected ids of a row:
  * the program the host dispatches for a geometry: unit geometry, key-range
    split, CTA pair, tile-loop unroll, unit order, segment order, descending
    walk and sampled first threshold;
  * the partition knobs of the backend: `grid_ctas`, `candidate_multiplier`,
    `check_period`, `sample_tiles_max`, `sample_shift_permille` and
    `far_sample`;
  * `max_seqlen_k`.
* `far_sample` is `0` or `1`, and the architecture's record sets the default.
  Set to `1`, it makes the sampled first threshold of the ascending-tile-order
  scan programs rank the far sample tiles only.  It leaves out the sample tile
  nearest the diagonal (an attention sink's local band).  The descending-order
  and wide programs ignore it.

## Non-finite inputs

Finite inputs and finite scores are the contract's domain, and `-inf` is
reserved for padding.  The operator may get NaN, infinite or FP32-overflowing
inputs: NaN queries, `+-inf` keys, or BF16-max values whose products overflow
FP32.  It still returns, and it must not hang.  The output structure stays
intact:

* shapes and dtypes;
* unique ids inside the row's visible prefix, in no specified order;
* the rows of finite segments of the same call meet the finite contract
  exactly.

The score values inside an affected row, and the relative order of its NaN
and `+-inf` scores, are not specified.  The reference ranks NaN lowest for its
own bookkeeping only.  The kernels' loop bounds and barriers do not depend on
score comparisons, so any bit pattern terminates.

## Candidate blocks

`cake_dsa_indexer_nondet_train_topk_with_candidates` (BF16, `ratio = 1`,
`1 <= top_k <= 2047`, default 512; `sm_100a`, `sm_103a` and `sm_107a`) returns
`(indices, scores, candidate_blocks)`.  `indices` and `scores` hold the sets
of `cake_dsa_indexer_nondet_train_topk` with the same `top_k`.
`candidate_blocks` is int32 `[T, 2048]`: block `b` holds the segment-local
keys `8 b .. 8 b + 7`, a row with `n` visible keys sees `nb = ceil(n / 8)`
blocks (the newest, `nb - 1`, may be partial; its future keys do not count),
and a block's score is the largest score of its visible keys.  With
`nb <= 2048` the row holds `0 .. nb - 1`; otherwise block `nb - 1` and the
2047 blocks of `0 .. nb - 2` with the largest scores, ties broken by the
larger block id (`+0.0 == -0.0`).  A row holds its set with `-1` in the other
slots, in no specified order (all `-1` when `n == 0`), and every id of
`indices` lies in a block of its row.  Under
`torch.use_deterministic_algorithms(True)` the call raises as the plain call
does.

Any other call runs the `scan_blocks` program of its unit (role `scan_blocks`, keys
`scan_blocks:<narrow|l6>:u<n>:s0:f<fit>`): the deterministic package's
candidate-block scan program under this package's prefix.  Only the unsplit,
unpaired narrow and l6 units carry it, so the host maps the decided kind to
one of them (a wide unit or a CTA pair to its unpaired unit, a key-range split
to narrow) with `n_split = 1`, at the decided tile unroll where its program is
registered, else the smallest registered one (`sm_100a` and `sm_103a` register
`u1`; `sm_107a` also registers the narrow `u2` program).  The block entries
follow the scan's candidate buffers and ticket tail in the workspace:
`blk_cap = ceil(W / 8) - 1` entries of 8 bytes per query slot of the grid when
`ceil(W / 8) > 2048`, else none, `W` being `max_seqlen_k` when given, else
`Tkv`.  With candidate blocks `max_seqlen_k` therefore bounds every key
segment of the call: a row with more older blocks than `blk_cap` stops the
kernel with a device-side trap.

A call whose longest key segment holds at most 2048 blocks (`ceil(W / 8) <=
2048`, `W` as above) keeps every block of every row, so no block score
decides a row: it runs the plain call's programs as dispatched (any unit, CTA
pair, key-range split with its merge and unit order, with the plain
call's workspace), writing each row's `0 .. nb - 1` and `-1` tail on the device
first (stage `cb_iota`: a few small launches over the rows and one copy of
the rows, reading the bound segment boundaries and offsets on every run, no
host synchronization, graph-capturable; plan field `iota`).  A row past
2048 blocks there (a `max_seqlen_k` below one of its key segments) stops that
write with a device-side assert, where the `scan_blocks` programs trap; the
write comes first, so such a call stops before any of its programs runs.
A prepared runner launches the
write with no allocation; preparing one (the eager entry prepares on every
call) adds three small device launches for the write's scratch.  The
16 MiB table the write copies rows from is made once per device, on the
stream of its first call; a call on another stream waits for that stream's
event while the table is not complete.
`cake_dsa_indexer_nondet_train_topk_with_candidates_workspace_size` reports
the bound; the `[T, 2048]` output is not part of it.  On `sm_107a` the
`scan_blocks` programs are listed in `ticket_arg_arches` and draw from a
16-byte ticket block passed as an argument, as the plain call's programs
listed there do ("Ticket tail" under "Workspace and host behaviour"): a
prepared runner whose scan draws owns that block, outside the reported bound.
An architecture without a `scan_blocks` program raises `NotImplementedError`.

## Candidate-restricted top-k

`cake_dsa_indexer_nondet_train_topk_in_candidates` (BF16, `ratio = 1`,
`1 <= top_k <= 2048`, default 512; `sm_100a`, `sm_103a` and `sm_107a`) takes
`candidate_blocks`, int32 `[T, W]` with `W <= 2048` (the `candidate_blocks`
output of `cake_dsa_indexer_nondet_train_topk_with_candidates` fits as is), and
returns `(indices, scores)`.  For query row `t` with `n` visible keys the pool
`P_t` is the keys `j < n` whose block `j // 8` appears in its row (any order;
`-1` and ids at or past the visible blocks name no key; the non-negative ids
of a row distinct).  The row selects the top `min(top_k, |P_t|)` keys of
`P_t` by score descending, then key id descending (`+0.0 == -0.0`), scored by
the plain call's formula, each row holding its set in no specified slot order (the
program's rows as they are).  Under
`torch.use_deterministic_algorithms(True)` a call that launches raises as
the plain call does.

The call runs the `cand` program (role `cand`, key `cand`): persistent CTAs,
one per SM, draw units of four queries from a self-resetting 16-byte ticket
word; for each unit query the CTA lists its valid blocks, scores only the keys
of those blocks (8 keys per block through a block tensor map over the raw key
rows) and keeps a candidate buffer of `cand_cap` entries per query that it
settles every `check_period` steps.  The work of a row is bounded by its
candidate keys (at most `8 W`), never by its visible prefix.  `k` keeps the
family's rule: a row-strided view (row stride a multiple of 8 elements) is
read in place, by the block tensor map and by the copy of the newest partial
block alike.  The workspace is `grid x 4 x cand_cap x 8`
bytes of candidate buffers (`cand_cap = max(4096, 4 top_k)`), `grid x 64 KiB`
of block lists and the ticket word;
`cake_dsa_indexer_nondet_train_topk_in_candidates_workspace_size` reports it,
and it depends on the grid and `top_k` only.  A backend-owned workspace has its
ticket word zeroed once at prepare, a caller-owned one before every run.  A
call with `W == 0` launches nothing and returns all `-1` / `-inf`.  An
architecture without a `cand` program raises `NotImplementedError`.

## Host dispatch

One call runs one kernel or two:

* **Unsplit call**: only the persistent **scan** runs.  It does the scoring,
  the exact candidate gate and the per-row selection, and writes every row's
  selected entries and padding straight into `indices` / `scores`.  Its grid
  is `grid_ctas`: one CTA per SM by default, made even for the CTA-pair
  program.
* **Split call**: when the dispatch splits the key ranges, the scan writes the
  per-range selections to the staging area of the workspace.  The plain split
  **merge** then writes the final rows (grid = `T`).  One merge program serves
  the three architectures.

No kernel orders a row.  The deterministic package's finalize, rank finalize
and rank merge programs are not part of this package.

The scan exists in several physical programs, and the host picks one per call
from host-known integers only: `T`, `Tkv`, `S`, `ratio`, `top_k`, the optional
`max_seqlen_k` bound and the SM count.  No tensor is read.  The device's
compute capability and SM count are read once per device index and reused.
The choice goes through the registry's per-architecture `POLICY` record
(`cake_policy.DispatchPolicy`, `cake_policy.select_program`,
`cake_backend.plan_dsa_indexer_nondet_topk`; the FP8 form's record and
mapping are described in "FP8 form").  These rules are the deterministic
package's rules without its finalize and rank rules.  A BF16 call therefore
gets the same scan program, candidate capacity, check period,
split count, CTA pair and ticket plan that the deterministic package picks for
it.  One exception is on `sm_107a`: both packages' records carry the same
`sample_class_rule` and `sample_long_rule`, but on a non-wide program of the
tiny dispatch class whose mean unit holds at most the class rule's tile
count, this package's `cake_policy.sample_class_for` takes the class rule's
rank shift and keeps the tile count, where the deterministic package skips
the class rule.  So the sampled rank shift can differ (whether a call samples
does not change).  Most rules use one estimate, the
mean key tiles per work unit: `(Tkv - T / (2 ratio)) / S / 128`.

| decision | rule (record fields) |
|---|---|
| unit geometry | `block_q_wide` queries per unit first where the record sets `wide_first_rule` (sm_103a: `[512.0]`), the call is wide-eligible (at least `wide_rule[0]` wide rounds per CTA and `rounds x mean tiles >= wide_rule[1]`, `rounds = ceil(T / block_q_wide) / grid`) and the mean unit tiles reach `wide_first_rule[0]`; else `block_q_l6` (three math warpgroups) when the mean unit tiles are at most `l6_rule[0]` and `top_k >= l6_rule[1]`; else `block_q_wide` when wide-eligible; else `block_q_narrow`.  The policy also evaluates `wide_wave_rule`, which replaces both wide rules where a record sets it.  No BF16 record sets it; the FP8 records of `sm_100a` and `sm_103a` do (`POLICY_FP8`), so the FP8 form chooses its wide unit by waves of the persistent grid |
| key-range split | `n_split = max(1, min(grid // units, mean tiles // split_min_range_tiles, split_max))`.  Where `n_split` is 1, a record with `split_wave_rule` splits in two when the units are long (at least `split_wave_rule[0]` and `2 x split_min_range_tiles` mean tiles) and the split raises the share of busy CTAs over the call's rounds by at least `split_wave_rule[1]`.  Where the record sets `split_wave_scope` (sm_107a: `[["narrow"], 1, true]`), that two-way split applies only to its unit geometries, on calls of at most its segment count, and with its third item on sampled calls only.  Only a unit geometry of `split_units` (the units whose split program the architecture registers) splits; any other unit runs unsplit.  A call that this rule splits takes `split_resident_rule` (sm_100a, sm_103a): `min(grid // units, 49152 // top_k, 32, max(8, n_fit if n_fit x top_k <= 32768 else 32))` splits, at least 2.  Here `n_fit = ceil(mean unit keys / top_k)`, rounded up to whole 4096-entry merge chunks, so the merge keeps its staged entries in shared memory.  Elsewhere `split_fit_rule` raises the count to `n_fit` where that fits `split_max` and the grid |
| CTA pair | the two-CTA multicast program when the mean unit tiles lie in `pair_rule[0..1]`, `top_k >= pair_rule[2]`, the grid has at least two CTAs, the call is not split (grid made even) and the unit geometry is one of `pair_units` (the units whose pair program the architecture registers) |
| unpaired wide unit | where the record sets `wide_pair_snake_only` (sm_107a), a sampled call keeps the CTA pair of the wide unit only where the pair program walks the snake order; elsewhere it runs the unpaired wide program when that program is registered |
| unpaired three-warpgroup preference | a call whose plan does not split takes the `block_q_l6` unit without a CTA pair.  With the sampled first threshold off (`sample_tiles_max=0`), where the record sets `l6_solo_unsampled_max_queries` (sm_100a, sm_103a: 16384), this applies only to a call planned on the CTA-pair l6 unit of at most that many queries; other units keep their plan.  Otherwise it applies always where the record sets `l6_solo_rule` (sm_107a).  On a sampled call, where the record sets `l6_solo_rule`, it applies with at least `l6_solo_rule[1]` segments or at most `l6_solo_rule[0]` mean unit tiles.  In every case it applies only where the l6 unit would not split either and the program it picks is registered (`PROGRAM_KEYS[arch]`, which the backend passes to `select_program`; without it the preference stays off); else the plan of the rows above applies.  Where the record sets `snake_l6_fallback_waves` (sm_107a: 310), a sampling-off call whose snake-order l6 program is not registered runs the plain-order l6 program instead, when the planned program needs at most that many waves of its persistent grid (`ceil(ceil(T / block_q) / slots)`, slots = CTAs, or CTA pairs for a pair program) |
| unit order | snake order when `snake_rule` names the program kind and the mean tiles (the ratio-1 or the `ratio > 1` floor) and the unit-cost spread reach its floors.  On a call of two or more segments whose kind `snake_packed_rule` names (sm_107a: `[["l6"], 48.0, None, 7]`), the snake order applies only from `snake_packed_rule[1]` mean unit tiles on with the sampled first threshold on, from `snake_packed_rule[2]` with it off (None: never), or within `snake_packed_rule[3]` waves of the persistent grid.  Else the plain order |
| segment order | the longest-first copy (`:lsf`) of the decided program where the record sets `lsf_programs` (every architecture) and the call packs at least two key segments that are not all equal.  `max_seqlen_k` decides: `Tkv == S x max_seqlen_k` means equal segments; without it the copy runs.  A program registered without its copy runs in the argument order: `sm_100a` and `sm_103a` register the copy of every scan program, `sm_107a` only those of its two `l6` ticket programs (`scan:l6:u1:s0:f0`, `scan:l6:u1:s0:f1`), which keep the argument order when the longest key segment is at most 5/4 of the shortest non-empty one (same sets either way).  Every row that asks the registry for a program looks up its key without the suffix, so the copy never changes the plan |
| descending walk | the descending-walk twin of the decided program (key kind `<kind>_desc`: the same unit, split, pair, tile unroll and fit form) where `desc_walk_kinds` names the physical kind for the call's sampling mode and the twin is registered (sm_103a: the `wide` unit with the sampled first threshold off); the plan, the knobs and the workspace stay |
| tile-loop unroll | `tile_unroll_factor` when `tile_unroll_rule` names the program kind and `top_k` reaches its floor (yielding to the snake order where the rule says so) |
| sampled first threshold on buffer-fitting units | when the mean unit tiles are at most `sample_fit_max_mean_tiles`, the mean segment offset leaves fitting units and the program kind is one of `sample_fit_kinds` (the kinds whose sampled-fit program the architecture registers; the split programs, for example, run their plain form on every call) |
| sampled tiles and rank shift | tiles: `sample_tiles_tiny_units` up to `sample_dispatch_tiny_tiles_max` mean unit tiles, `sample_tiles_short_units` up to `sample_dispatch_mean_tiles_max`, else `sample_tiles_max`.  Shift: `sample_shift_rule[1]` permille where the record sets `sample_shift_rule` and `top_k >= sample_shift_rule[0]`, else `sample_shift_permille`.  Where the record sets `sample_class_rule` (sm_100a, sm_103a: `[8, 500]`; sm_107a: `[32, 500]`), the wide-unit programs take its shift on packed calls, and the tiny dispatch class (packed calls of at most 4096 queries per segment and at most `sample_dispatch_tiny_tiles_max` mean unit tiles) takes its shift, and its tile count where the mean unit holds more tiles than that count.  Where the record sets `sample_long_rule` (sm_107a: `[["l6"], 500]`), a call of more than `sample_dispatch_mean_tiles_max` mean unit tiles whose physical kind the rule names takes its shift.  On sm_107a these two rules keep every rank input of the ratio-4 pooled selection at or below its time with sampling off.  Explicit knobs override all of these.  Whether sampling is on enters the rules above.  The tile count and the shift are launch scalars only: they set how conservative the sampled first threshold is (a threshold that proves too aggressive is streamed again exactly), never the program, the capacity or the selected set.  Unsampled calls do not read the shift |
| candidate capacity | `candidate_multiplier` x `top_k` (`cand_mult_rule[1]` for long units), rounded up to whole tiles.  It is raised to `cand_cap_floor` slots, or to `cand_cap_floor_rule[1]` slots from `top_k >= cand_cap_floor_rule[0]` on where the record sets that rule.  On `sm_100a` the 16384-slot floor applies from `top_k = 1024` on and smaller `top_k` keep 8192; `sm_103a` keeps its 16384-slot floor on every `top_k`.  It is lowered to `cand_cap_ceiling` slots where the record sets one: the multiplier becomes `ceiling // top_k`, so the buffer stops growing with `top_k` once it holds the ceiling.  With the sampled first threshold off (`sample_tiles_max=0`), the `candidate_multiplier_unsampled` / `cand_mult_rule_unsampled` / `cand_cap_floor_unsampled` / `cand_cap_ceiling_unsampled` values apply instead.  An explicit `candidate_multiplier` knob replaces the multiplier |
| check period | default `min(check_period_max, (cand_cap - top_k) // 128, cand_cap // check_period_cap_divisor)`, at least 1, or the kind's `check_period_kind_overrides` value clamped to the exactness bound; an explicit `check_period` must lie in `[1, limit]` (the kernel's headroom rule) |

The dispatched programs must be registered for the device's architecture
(`cake_jit.PROGRAM_KEYS[arch]`).  This covers the scan and, on a split call,
the merge.  Both are looked up before anything is allocated.  Otherwise the
backend raises `NotImplementedError` naming the program key; it never runs a
different program.  The kernel source's own dispatch functions are the
reference for every rule.  The Cake exporter verifies the record against them
before the record is frozen.

## Workspace and host behaviour

* **Workspace bound**
  (`cake_dsa_indexer_nondet_train_topk_workspace_size(T, Tkv, S, top_k=,
  ratio=, device=)`).  It is the sum of three parts:
  * `grid_ctas x queries_per_unit x candidate_capacity(top_k) x 8` bytes of
    persistent candidate buffers for the dispatched unit geometry and
    capacity.  For example, this is about 74 MiB at `top_k = 2048` with four
    queries per unit on 148 SMs at the 16384-slot capacity floor (the floor
    `sm_100a` applies from `top_k = 1024` on), and 148 MiB with eight queries
    per unit.
  * `T x n_split x top_k x 8` bytes of staging when the dispatch splits the
    key ranges.
  * An 8320-byte tail that every plan carries: 16 int64 for the ticket word,
    then 1024 int64 for the longest-first segment order that the scan
    prologue writes.

  This is the deterministic package's bound for the same BF16 call.  The
  bound depends on the call geometry.  Size a reused buffer for the largest
  bound over the geometries it serves.  The bound is the larger of the
  sampled default's plan and the plan with the sampled first threshold off
  (`sample_tiles_max=0`), so a buffer sized by it serves both modes.  The
  backend's function of the same name also takes the `grid_ctas` and
  `candidate_multiplier` knobs; size a buffer for a call that sets them with
  the same values.  The workspace is separate from the inputs and the two
  `[T, top_k]` outputs.  A buffer below the call's bound is rejected
  (`workspace_buffer needs N bytes`), and so is a buffer that is not 8-byte
  aligned.  Computing the bound never consults
  `torch.use_deterministic_algorithms`.
* **No host synchronization**.  The segment boundaries and offsets are read
  on the device only.  The host uses `T`, `Tkv`, `S`, `ratio`, `top_k`,
  `max_seqlen_k`, the SM count and the strides.  The eager entry point
  allocates the outputs and the workspace (unless supplied) through the
  caching allocator.  A call without keys fills its outputs with padding (two
  fills) instead of launching.
* **Ticket tail**.  On all three architectures (`TICKET_TAIL_ARCHES`), the
  scan programs draw every work unit from a ticket word.  A record's
  `static_scan_kinds` names the kinds whose programs keep the static unit
  enumeration and get no tail; no BF16 record names one, and the FP8 record
  of `sm_107a` names every FP8 kind (see "FP8 form").  So a BF16 call that
  shares the GPU with another stream slows with the SMs that stream holds,
  not by a whole extra wave of work units.
  Every completed scan launch leaves the word at 0.  How the word is reset
  depends on the workspace:
  * Backend-owned workspace: the tail is zeroed once, when the runner is
    prepared.  The eager entry prepares a one-shot runner, so this happens
    once per eager call.
  * Caller-owned `workspace_buffer`: the scan binding zeroes the tail with
    `cudaMemsetAsync` on the launch stream right before every scan launch, in
    the same host call.  A CUDA graph that captures a run carries that memset
    node.
  * Static call: nothing is zeroed.  Here the shape alone bounds the work
    units by the launched grid (`cake_policy.ticket_static_path`), so the
    scan never touches the ticket word.
  * On `sm_107a`, a scan program listed in `ticket_arg_arches` draws from a
    16-byte ticket block passed as an argument instead of the tail.  A
    prepared runner owns such a block, zeroed once at prepare.  The eager
    entry passes the first 16 bytes of the workspace's tail instead, and the
    scan binding zeroes its ticket word right before the launch.  The `:lsf`
    copies of such programs still keep the segment order in the tail; every
    launch writes it before it reads it, so it needs no zeroing.
  * A run that raised while launching marks the tail dirty, and the next run
    zeroes it first.
* **Prepared runner**.  `cake_backend.prepare_dsa_indexer_nondet_topk(...)`
  validates one binding, decides its programs and allocates the missing
  outputs and the workspace.  These are the backend's only allocations.  The
  returned `DSAIndexerNondetRunner` launches with no allocation and no host
  synchronization.  It can be captured into a CUDA graph by the caller, and a
  replay reads new values written into the bound tensors.  Prepare a new
  runner when a shape, dtype, stride or tensor binding changes.  Every
  `run()` that launches checks the deterministic-algorithms setting first
  (see "Mode and names").
* **Launch structure**: the scan, then the merge on a split call (see above).
  No `[T, Tkv]` matrix is materialized.

## Layout of this package

The Cake exporter produces the kernel and launcher sources under `csrc/`, the
registry `cake_jit.py` and `build.json`.  Every program is one of the
deterministic package's BF16 or FP8 scan programs or its plain split merge.  It is
exported under this package's prefix as `cake_dsa_indexer_nondet_topk_<20 hex>`
and keeps the twenty-hex identity of the deterministic program it comes from.
Programs that differ only in a few constants share one kernel translation
unit.  Programs whose launchers differ only in the kernel symbol, the
host-shim namespace and the launch constants share one binding.  Each
`build.json` entry selects its program through its own `-D` definitions.
`build.json` records every source's digest and a closure digest per entry.
`csrc/.clang-format` disables formatting so the digests stay valid.

* `cake_jit.py` is the registry.  It holds:
  * the stage roles `scan` and `merge` only (`STAGES`) and their argument
    plans (`ARG_PLANS`, the deterministic package's, under
    `ABI = "dsa_indexer_v5"`).  The scan's `K_scale` slot holds the FP8
    form's per-key scales; a BF16 call gets a placeholder that no BF16
    program reads;
  * one record per program (`PROGRAMS`: role, the architecture-free `library`
    stem, architectures, and `ticket_arg_arches` where a scan program draws
    from its ticket block);
  * the program of every dispatch key per architecture (`PROGRAM_KEYS`):
    `scan:<kind>:u<unroll>:s<snake>:f<sampled fit>[:fp8][:lsf]` and `merge`,
    with no finalize or rank keys;
  * the per-architecture dispatch policy (`POLICY`), the FP8 form's record
    (`POLICY_FP8`), `NUMERICS` and `TICKET_TAIL_ARCHES`;
  * the loader glue: `library_name(program, arch)` gives the `build.json`
    entry `<library>_<arch>`, and `load_program(program, arch)` loads it
    through `cake_build.load_library`.

  The launch geometry of a program (block, cluster, dynamic shared memory) is
  fixed in its binding.
* `cake_build.py` + `build.json`: `cake_build.py` is the loader that every
  Cake package shares, byte for byte.  `build.json` is
  this package's build manifest (schema `cake-build/1`) and the only record of
  what the package compiles.  It has one entry per program and architecture
  with its package-relative sources, nvcc flags, closure digest, per-source
  SHA-256 and `aot/<name>.so` path.  It also has the common nvcc / link flags
  and the `tvm-ffi` requirement.  `cake_build.load_library(name)` loads the
  prebuilt library when `aot/` holds it, checked against the builder's
  `aot/build_info.json` and the library's own `build_cuda_version()`.
  Otherwise it JIT-builds the entry through the package's JIT shim
  (`_cake_pplx_jit.py`) from the same sources and flags.  The prebuilt path
  never imports the JIT shim.
* `csrc/cake_ffi_checks.h` holds the tvm-ffi / DLPack include set, the named
  tensor-argument checks and `cake_ffi_build_cuda_version()`, which the
  generated launchers use.  The package compiles from its own `csrc/`
  alone.
* `cake_policy.py` is the dependency-free dispatch policy: program choice,
  capacity, check period, split count, the ticket static path and the
  workspace bound.
* `cake_backend.py` holds validation, planning, the deterministic-algorithms
  guard, argument binding, the prepared runner and the eager entry point.
* `csrc/cake_dsa_indexer_nondet_topk/` holds the generated kernel and binding
  translation units:
  * one kernel per program, or per group of programs that differ only in
    their constants (a scan lever-twin group);
  * one binding per launcher text: the 86 scan programs share
    9 bindings, and the merge program has its own.

  A shared kernel source names its kernel `CAKE_PROGRAM_KERNEL` and reads its
  constants (tile unroll and snake order) from
  `CAKE_INDEXER_<KNOB>`.  A shared binding names the kernel, its host-shim
  namespace, block size and dynamic shared memory through
  `CAKE_KERNEL_SYMBOL`, `CAKE_PROGRAM_SHIM`, `CAKE_BINDING_THREADS` and
  `CAKE_BINDING_SMEM`.  Each of these has an `#ifndef` default naming the
  program the file was exported for.  Every other program's `build.json`
  entry passes its own values with `-D`.

## Tests and benchmark

There are seven test files:

* `tests/experimental/test_cake_dsa_indexer_nondet_train.py`: the backend
  suite;
* `tests/experimental/test_cake_dsa_indexer_nondet_train_fp8.py`: the FP8
  form;
* `tests/experimental/test_cake_dsa_indexer_nondet_train_vs_det.py`: the
  comparison with the deterministic package;
* `tests/experimental/test_cake_dsa_indexer_nondet_train_api.py`: the
  `cake_pplx.cake_dsa_indexer_nondet_train` entry points;
* `tests/experimental/test_cake_dsa_indexer_nondet_train_names.py`: the
  naming rules;
* `tests/experimental/test_cake_dsa_indexer_nondet_train_candidates.py`: the
  candidate-block call (below);
* `tests/experimental/test_cake_dsa_indexer_nondet_train_in_candidates.py`:
  the candidate-restricted call (below).

They share the reference in
`tests/experimental/test_helpers/cake_dsa_indexer_reference.py` and the
naive-tier judge in `tests/experimental/test_helpers/cake_dsa_indexer_naive.py`
(on `tests/cake_naive.py`) with the deterministic package.  Results are
compared after each row is canonicalized: valid slots sorted by id with their
scores, padding last.  The comparisons therefore check sets, not slot order.
On integer inputs every score is exact in FP32, so every call must return the
exact selection's ids and valid counts.  Over 32 draws the scores are judged
against their FP64 values beside the naive tier's (the same scores in torch
BF16), on integer inputs and, at the kernel's own keys, on Gaussian inputs.
No test compares the score bits of two calls.  Coverage:

* registry and policy host tests: no ordering or rank keys; a split plan
  runs the scan, then the merge; an unsplit plan runs the scan only;
* validation;
* the exact semantic cases, held to the exact selection;
* packed causality;
* random rows, judged against the naive tier;
* near ties;
* the recorded token counts;
* the ratio-4 pooled top-512 selection;
* `top_k = 4096`;
* five eager calls and five prepared runs, each holding the exact selection;
* split versus unsplit set equality, a forced split whose rows hold padding
  and a forced split that cuts a tie group;
* non-finite inputs;
* the workspace bound, the ticket tail and the ticket word;
* CUDA graphs and the absence of host synchronization;
* the deterministic-algorithms guard.

The FP8 suite covers the host-only key grammar, the FP8 record, the kind
mapping and the reachability of every registered FP8 key, validation and BF16
plan / workspace identity; on the GPU, the exact selection's ids and counts
on integer e4m3 inputs and the scores judged against the naive tier (integer
and amax-quantized Gaussian e4m3 inputs), the same sets from every registered
FP8 program forced on one call, the prepared runner's `k_scale` binding, the
per-form ticket decision on a poisoned workspace, the workspace bound, CUDA
graphs and, on `sm_100a` / `sm_103a` where `deep_gemm` imports, the sets
against the top-k of DeepGEMM's FP8 MQA logits.

The comparison file checks that the registry keys, policy records and host
plans of BF16 and FP8 calls are the deterministic package's, on the host, and
those of BF16 calls at the device's grid.  It does not compare the outputs of
the two packages.

The candidate-block call has its own suite,
`tests/experimental/test_cake_dsa_indexer_nondet_train_candidates.py`, with
the FP64 oracle in
`tests/experimental/test_helpers/cake_dsa_indexer_candidates.py`: host-only
registry, plans at the request's shapes on every architecture (on `sm_107a`
the expected programs), the registered program of every candidate-block plan
on a dense grid of shapes and `top_k` up to 2047, argument checks and the
refusal of an architecture whose registry lacks the programs; on the GPU, the
exact sets of `indices` and `candidate_blocks` against the oracle on
MXFP4-grid inputs (six cases, two packed layouts), sampled rows at the
request's shapes, the Gaussian rule with the scores judged against the naive
tier, containment, the repetition of the sets and the plain call's sets (also
at `top_k` 2044 and 2047), a prepared runner that draws its work units on a
poisoned caller-owned workspace, the caller-owned workspace and outputs, the
argument checks, the absence of host synchronization and the
deterministic-algorithms guard.  The benchmark's `--rows candidates` times the
call next to the plain call at the request's shapes.

The candidate-restricted call has its own suite,
`tests/experimental/test_cake_dsa_indexer_nondet_train_in_candidates.py`, with
the FP64 oracle in
`tests/experimental/test_helpers/cake_dsa_indexer_candidates.py`: host-only
registry, the plan and its workspace bound at the request's shapes, argument
checks and the refusal of an architecture whose registry lacks the program; on
the GPU, the exact sets against the oracle on MXFP4-grid inputs (six cases,
six kinds of candidate rows, two packed layouts), the unrestricted sets from
the source layer's own candidate blocks and from every visible block, keys
outside the pool that never count, sampled rows at the request's shapes, the
Gaussian rule with the scores judged against the naive tier, the repetition of
the sets, the caller-owned workspace and outputs, the argument checks, the
absence of host synchronization and the deterministic-algorithms guard.  The
benchmark's `nondet_in` / `det_in` arms on `--rows candidates` time the call
next to the plain calls.

Benchmark: `benchmarks/bench_cake_dsa_indexer_nondet_train.py` times this
package against the deterministic package on the same rows, in the BF16 form
or, with `--form fp8`, in the FP8 form on the FP8 shape set.  It checks each
package's output on its own (the structure checks, `min(top_k, visible)` valid
slots per row) and does not compare the outputs of the two packages.
