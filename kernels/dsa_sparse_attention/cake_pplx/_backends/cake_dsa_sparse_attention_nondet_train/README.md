# Cake DSA sparse-attention training backend (SM100 / SM103 / SM107)

Backend package of `cake_pplx.cake_dsa_sparse_attention_nondet_train`, the
non-deterministic package of the DeepSeek Sparse Attention training kernels
(top-k sparse MLA with absorbed queries; GLM-5.2 geometry: 64 heads, 512
latent + 64 rope query/key dimensions, 512 value dimensions, top-k 2048).
The kernels are native 64-query-head programs; the rope form also serves 96
and 128 query heads on `sm_100a` and `sm_103a`, through head-window builds
that run both 64-head windows of every token in one launch, or else on two
64-head windows of the 64-head kernels (see "Query heads above 64 (head
windows)").

The entry points of `cake_pplx.cake_dsa_sparse_attention_nondet_train` are
stable APIs; this backend package is experimental (no compatibility guarantee
for its own helpers).  SM100 (B200 / GB200) is the acceptance architecture of
the native kernels and
SM103 (B300 / GB300) is compiled from the same sources; SM107 (Rubin R200) is
the acceptance architecture of the packed / strided-layout extension.  On
`sm_107a` each forward runs its scanner-warp build: the stages `fwd_scan`,
`nope_fwd_scan` and `nope_fwd_scan_sink` ship for `sm_107a` only, and `fwd` /
`nope_fwd` / `nope_fwd_sink` ship for `sm_100a` and `sm_103a`, and so do
their `with_o_lo=False` builds (`fwd_scan_no_olo`, `nope_fwd_scan_no_olo` and
`nope_fwd_scan_sink_no_olo` for `sm_107a` only; `fwd_no_olo` / `nope_fwd_no_olo`
/ `nope_fwd_sink_no_olo` for `sm_100a` and `sm_103a`); the other stages,
the NoPE sink-gradient stage `nope_bwd_dsink` included (see "Trainable per-head
sink"), compile from the same sources for all three.  The `sm_107a` libraries need CUDA 13.5 or newer (`min_cuda` 13.5
in their `build.json` entries).  On `sm_100a` every library needs CUDA 12.9
or newer (`min_cuda` 12.9) except those of the stages `bwd_delta` /
`nope_bwd_delta` and their `with_o_lo=False` build `bwd_delta_no_olo` /
`nope_bwd_delta_no_olo`, `nope_bwd_dsink` and `index_prep`, which need 12.8.  The feature does not
participate in automatic backend selection, autotuning or trace-apply.  Its
libraries are loaded prebuilt from `aot/` when the package ships them and
JIT-built through the package's JIT shim from `build.json` otherwise (see
"Layout of this package").

## Mode and names

The module name states the mode and no entry point takes a `deterministic=`
keyword (`with_o_lo`, see "Without the output residual", chooses whether the
forward writes `o_lo` and how the backward forms `delta`; it is not a
determinism switch).  This is the `nondet` package: the backward adds the dK / dV
contributions with unordered FP32 atomics, and the package does not promise that
any output repeats bit for bit, from run to run or across routes, views,
layouts, streams or kernels running beside it.
Under `torch.use_deterministic_algorithms(True)` the forward runs and every
backward launch sequence (the autograd backward, `backward` / `backward_nope`
and the prepared runners' `backward`) raises PyTorch's `RuntimeError` before it
allocates its backward buffers and launches its first backward kernel; with
`warn_only=True` it issues a `UserWarning` and runs this package's backward.
Deterministic preparation may run before the alert: a copy of a `dout` the
TMA descriptor cannot address, the plan of a new binding (which can JIT-load
the backward stages and, without a caller `topk_length`, allocates and fills
the binding's row-length vector) and the validation of `dkv_acc` /
`dkv_dst_map`.  It never falls back to another package.  A
backward without query rows launches nothing and does not alert.  The
deterministic package `cake_pplx.cake_dsa_sparse_attention_det_train` has
the same entry points (`det` in place of `nondet`, without `with_o_lo`); its
outputs repeat bit for bit run to run.

The tests judge the accuracy of every floating output against an FP64
reference: over 32 random inputs per case, the error of each output, in the L2
and the max norm, median and largest, is at most 1.005 times that of the same
call in PyTorch BF16, and exactly 0 where that error is 0.  Index, length and
map outputs are checked exactly (see "Status").

| Before | Now |
|---|---|
| `dsa_sparse_attention` | `cake_dsa_sparse_attention_nondet` |
| `dsa_sparse_attention_varlen` | `cake_dsa_sparse_attention_nondet_varlen` |
| `dsa_sparse_attention_nope` | `cake_dsa_sparse_attention_nondet_nope` |
| `dsa_sparse_attention_nope_varlen` | `cake_dsa_sparse_attention_nondet_nope_varlen` |
| `DSASparseAttentionFunction` | `CakeDSASparseAttentionNondetFunction` |
| `DSASparseAttentionNopeFunction` | `CakeDSASparseAttentionNondetNopeFunction` |
| `DSASparseAttentionNopeVarlenFunction` | `CakeDSASparseAttentionNondetNopeVarlenFunction` |

The names before the rename are removed (no aliases); importing their modules
fails.  The environment variables `CAKE_PPLX_DSA_*`, the backend's
internal helpers (`forward`, `backward`, `prepare_dsa_train`, ...) and the
generated library, program and kernel names keep their names.

## Public entry points (`cake_pplx/cake_dsa_sparse_attention_nondet_train.py`)

```python
from cake_pplx.cake_dsa_sparse_attention_nondet_train import (
    cake_dsa_sparse_attention_nondet,
    cake_dsa_sparse_attention_nondet_varlen,
)

out = cake_dsa_sparse_attention_nondet(
    q_latent, q_rope, kv_latent, k_rope, indices,
    topk_length=None, softmax_scale=None, return_lse=False,
    dkv_acc=None, dkv_dst_map=None, num_segments=1)
out = cake_dsa_sparse_attention_nondet_varlen(
    q_latent, q_rope, kv_latent, k_rope, gather_kv_indices,
    cu_seqlens_q, cu_seqlens_k, max_seqlen_q=None, max_seqlen_k=None,
    causal=False, topk_length=None, softmax_scale=None,
    return_lse=False, dkv_acc=None, dkv_dst_map=None)
```

* `q_latent [T, H, 512]`, `q_rope [T, H, 64]` BF16, `H` = 64, 96 or 128 query
  heads (96 and 128 on `sm_100a` / `sm_103a`, see "Query heads above 64 (head
  windows)"; another `H` raises `ValueError`); `kv_latent [S, 512]` (K = V)
  and `k_rope [S, 64]` BF16.  Strided views are consumed in place -- see
  "Packed and strided inputs" below.
* `indices [T, topk]` int32 (any row stride) hold **global** key rows; `-1` or `>= S` marks an
  invalid slot anywhere in the row; `topk_length [T]` int32 optionally
  invalidates slots `>= topk_length[t]`; any positive `topk`.  Without
  `topk_length` the forward kernel derives each row's length (last valid slot
  + 1) itself while it runs and the training entry hands that vector to the
  backward, so both skip the trailing invalid key blocks of short rows; the
  skipped blocks hold only invalid slots.
* The varlen form takes per-document indices (`gather_kv_indices`, relative
  to `cu_seqlens_k[d]`) and turns them into global key rows on device.  The
  NoPE varlen entry follows the size rule of its own section below
  (`NOPE_CONVERT_SLOT_LIMIT`, `nope_varlen_index_mode`): below the
  limit, without caller lengths, the forward kernel converts the rows itself in
  its single launch, position-preserving -- the global key or `-1` at the slot
  it came from, the length the last valid slot + 1, bitwise the torch twin
  `offset_gather_kv_indices(...)`; at or above the limit, and always with a
  caller `topk_length`, one launch of the NoPE record's index-preparation
  stage right before the forward kernel (`index_prep`, `compact = 1`, the
  forward launched behind it with the programmatic-dependent-launch attribute,
  see below) compacts every row stably: its valid keys come first in ascending
  slot order, then `-1`, and the row's length is its valid count (bitwise the
  torch twin `offset_gather_kv_indices(..., compact=True)`), so rows with `-1`
  holes cost their valid slots in the forward and the backward.  The rope varlen
  entry always launches the compacting stage (the torch twin where the stage
  is not available -- host tensors, an unserved device, a non-contiguous prefix
  sum -- with bitwise the same result).  Rows whose valid slots already come
  first are unchanged bytes with the same length on every route.  A caller
  `topk_length` is a position cut applied before compaction (slots at positions
  `>= topk_length[t]` are invalid regardless of content); on the rope varlen
  entry's stage route a cut that is not a contiguous int32 `[T]` tensor on the
  rows' device raises `ValueError` before any launch.  `cu_seqlens_q`
  and `cu_seqlens_k` are independent: a query segment may be shorter than its
  key segment, in which case it is the tail of that key prefix -- query
  `local_q` of document `d` sits at key position
  `(seqlen_k[d] - seqlen_q[d]) + local_q`.  With `causal=True` the offsetting
  also drops selected keys after that position
  (`offset_gather_kv_indices(..., causal=True)` keeps a slot iff
  `0 <= idx < seqlen_k[d]` and `idx <= (seqlen_k[d] - seqlen_q[d]) + local_q`);
  with `causal=False` (the default) a query's key set is exactly its index row.
  Either way the kernels apply no positional mask of their own: the offset
  index rows define the key sets.  A zero-length query segment contributes no
  rows; a row whose every slot ends up `-1` gives `out = 0`, `lse = -inf` and
  zero gradients.  The segment count `len(cu_seqlens_k) - 1` is passed on as
  `num_segments`: the backward's whole-row key-range passes apply to
  one-segment rows only (see "Key-range passes"), and on `sm_107a` the
  direct-accumulation rule reads it (see "Direct accumulation from the main
  stage").  The flat entries and the step backwards of both forms take the
  count as the keyword `num_segments` (a Python `int`, default 1): a caller
  that passes the global rows of a packed batch passes `len(cu_seqlens_k) - 1`
  and gets the varlen entry's plan for the same rows.
* Forward: `out [T, H, 512]` BF16 and (with `return_lse=True`) the natural-log
  `lse [T, H]` FP32 over the valid keys; fully masked rows give `out = 0` and
  `lse = -inf`.  The autograd wrapper also keeps the BF16 output residual
  `o_lo = bf16(fp32(O) - bf16(O))`; the backward forms
  `delta = rowsum(dO * (O + O_lo))` in FP32 from the BF16 output and it.
  With `with_o_lo=False` no `o_lo` is computed, stored or saved and the
  backward forms `delta = rowsum(dO * O)` from the BF16 output alone (see
  "Without the output residual").
* Backward: `dq_latent`, `dq_rope` (computed once per row; above 64 heads on
  the two-window bridge the overlap heads `[H - 64, 64)` are written by both
  windows, the second write final),
  `dkv_latent [S, 512]`, `dk_rope [S, 64]` (FP32 `red.global`
  accumulation, then a cast to natural-layout BF16).  BF16 MMA operands, FP32
  accumulation throughout.  `cake_backend.backward(..., dkv_fp32=True)` (and a
  runner prepared with `dkv_fp32=True`) returns the dK/dV gradients as
  natural-layout FP32 tensors instead; the kernels' internal permuted
  accumulator layout never crosses the API boundary -- the cast stage serves
  both modes.  With a caller-provided `dkv_acc` (see "Packed fp32 dKV
  accumulation and destination mapping" below) the dK/dV gradients are
  accumulated into that packed FP32 buffer instead, no `dkv_latent` /
  `dk_rope` tensors are produced and the autograd wrapper returns no gradient
  for `kv_latent` / `k_rope`.

### Packed and strided inputs

The kernels read the query operands through TMA descriptors encoded from the
tensor's own strides and gather the key operands by row stride, so the
trainer's packed layouts are consumed in place
-- no `.contiguous()`, `cat` or copy on the host:

* `q_rope` may be the `192:256` channel slice of the pre-absorption
  `[T, H, 256]` query (head stride 256, token stride `H * 256` -- 16384 at 64
  heads --, storage offset 192 elements = 384 B), `q_latent` a view of a
  packed `[T, H, 576]` query, or both contiguous.  Rule (`cake_backend._check_head_tensor`): unit channel
  stride; head and token strides positive multiples of 8 elements (16 bytes,
  the granule of `cuTensorMapEncodeTiled`'s global strides); head stride at
  least the slice width; base address 16-byte aligned.
* `kv_latent` / `k_rope` may be the `0:512` / `512:576` column slices of a
  packed `[S, 576]` or `[S, 704]` row (a frozen 128-channel indexer key stored
  alongside).  Rule (`cake_backend._check_key_tensor`): unit column stride;
  row stride at least the slice width and a multiple of 8 elements; base
  16-byte aligned (the TMA gather descriptors and the 16-byte rope loads).
* `indices` rows stay contiguous int32 (any row stride; a column slice of a
  wider index buffer is accepted); `dout` follows the layout rules of
  `q_latent` and is read in place -- the head-major `[512, T * 512, 1]` view an
  output projection hands back included -- by the main stage (stride-carrying
  tensor map) and by `bwd_delta` (the strided delta kernel shared with the
  NoPE record: `dout_offset + token * dout_token_stride + head *
  dout_head_stride + channel`).  The eager `backward` copies a `dout` only when
  the descriptor cannot address it (base not 16-byte aligned, strides not
  multiples of 8 elements, channel stride != 1); `prepare_dsa_train` rejects
  such a view.  Above 64 heads the delta stage reads `dout` as `T * H` rows of
  64 virtual heads, which needs its token stride to be `H` head strides (a
  dense or packed-slice `dout`); any other admissible layout -- the head-major
  view included -- is copied once into the workspace by the eager `backward`
  and the prepared runner alike (see "Query heads above 64 (head windows)").
* For a strided view, only descriptor fields and pointer offsets differ from
  a contiguous tensor.

### Packed fp32 dKV accumulation and destination mapping

`cake_backend.backward(..., dkv_acc=, dkv_dst_map=)`, `prepare_dsa_train(...,
dkv_acc=, dkv_dst_map=)`, the public entries and the autograd `Function`
(`CakeDSASparseAttentionNondetFunction.apply(q_latent, q_rope, kv_latent, k_rope, indices,
topk_length, softmax_scale, key_passes, dkv_acc, dkv_dst_map, num_segments)`)
accumulate the dK/dV gradients directly into a caller-provided FP32 buffer
instead of returning `dkv_latent` / `dk_rope` (the packed dKV of a trainer, with the repeated / remapped rows of a
context-parallel window):

* `dkv_acc`: FP32 `[S_dst, >= 576]` -- the latent gradient is added into columns
  `0:512`, the rope gradient into columns `512:576`; further columns are never
  touched (a `[S_dst, 704]` buffer holding a frozen 128-channel indexer key next
  to the 576 channels is fine).  Row stride `>= 576` elements and a multiple of
  4, 16-byte-aligned base; the 576-column view of a wider buffer is accepted.
  Accumulation is `+=`: the caller zeroes the buffer when it wants fresh
  gradients, and two backward calls into the same buffer sum.
* `dkv_dst_map`: optional contiguous int32 `[S]` giving the destination row of
  every source key row (values in `[0, S_dst)`, not checked on device;
  duplicates allowed and summed).  Without a map the identity is used, which
  needs `S_dst >= S`.
* The kernel does not range-check `dkv_dst_map`: every value must lie in
  `[0, S_dst)` (the rows of `dkv_acc`), and a value outside that range would
  add into memory outside the buffer.  The keyword `check_dst_map` of the
  entry points (default `False`) validates the values on every call when
  `True` (one device synchronization) and raises `ValueError` on a violation;
  leave it off in production.  The backend reads no environment variable:
  the public wrapper `cake_pplx.cake_dsa_sparse_attention_nondet_train` maps
  `CAKE_PPLX_DSA_CHECK_DST_MAP` onto the keyword when the caller leaves
  it at `None`.
* Through the autograd entries `dkv_acc` may also be a zero-argument callable
  that returns the accumulator, or an `(accumulator, dkv_dst_map)` pair; the
  backward calls it, so the buffer need not exist at forward time (a callable's
  result is validated in the backward, where `check_dst_map` then
  synchronizes).  The autograd `Function`s save every tensor their forward
  produces for the backward -- `out`, `lse`, `o_lo`, the index rows, the
  lengths -- through `ctx.save_for_backward` (a fixed list with a zero-size
  placeholder for an absent `topk_length`, and for `o_lo` under
  `with_o_lo=False`, so non-reentrant
  `torch.utils.checkpoint` can match the recomputed tensors by position) and
  keep nothing else of size on `ctx`: after a checkpointed forward the entries
  retain ~0 MiB.  Saved-tensor hooks such as
  `torch.autograd.graph.save_on_cpu` may hand the index rows back in another
  layout (a column slice or an offset view comes back contiguous); the NoPE
  backward then plans over the rows it receives.  A tensor `dkv_acc` (and
  `dkv_dst_map`) is caller-owned and stays on the autograd node as before,
  outside the saved list: zeroing or otherwise touching it in place between
  the forward and the backward is fine.
* With `dkv_acc` the backward returns `(dq_latent, dq_rope, None, None)`, the
  autograd wrapper returns `None` gradients for `kv_latent` / `k_rope`, and
  `dkv_fp32=True` is rejected.  The kernels' permuted FP32 accumulators remain
  per-call scratch; `bwd_cast` un-permutes, remaps and adds in one pass
  (`accumulate = 1`).  With a map the adds are `red.global.add.v4.f32`, whose
  order is not fixed for repeated destination rows (an injective map is still
  exactly one add per element); without a map every 16-byte vector is a
  load-add-store, `previous + value`.
* The generated `bwd_cast` launcher takes the packed-accumulate operands
  (`dst_packed, dst_row_stride, dst_map, has_dst_map, accumulate`); a plan
  that does not accumulate passes inert placeholders with `accumulate = 0`.
  The eager plan cache keys the backward on the geometry (shape, strides,
  dtype, device, alignment) of `dkv_acc` and `dkv_dst_map` as well -- the row
  stride and the presence of a map are launch constants of the accumulating
  launch -- and the caller's buffer and map are re-supplied per call.

#### Direct accumulation from the main stage

A program may register natural-layout variants of the main stage
(`bwd_main_natural`, and `bwd_main_pass_natural` next to the key-range-pass
form): the same kernel source compiled with `-DDKV_NATURAL=1`, whose drain
transposes each key's FP32 contributions across lanes and `red.global.add`s
them at their natural position straight into the caller's packed rows
(`dkv_acc` through
`dkv_dst_map`).  Such a binding has no FP32 accumulators, no zero fill and no
`bwd_cast` launch (`dsa_train_workspace_size(..., dkv_acc=True)` is smaller by
`S * 2304` B, the runner's `dkv_direct` is set).  The lane transpose costs the
reduce-bound rows per tile, and only rows with many keys per query token pay
it back (B200 backward: `S / T >= 16` 0.5-4 % faster, `S = T` 3.7 % slower), so
the host takes the direct path by the size rule the record carries
(`dkv_direct`: `min_keys_per_query`, 4 -- direct iff `S >= 4 T`) whenever
`dkv_acc` is given.  Like `key_pass_policy`, the rule may differ per
architecture: a rule `dkv_direct.rules[arch]` maps exactly
`min_keys_per_query`, `cast_slots_per_key` and `cast_min_slots`, and
optionally the key cap `cast_max_keys` (a positive integer; no cap without
it), replaces the base count on that target and adds a work test (see
"Direct accumulation on `sm_107a`" below); an architecture without a rule takes the base, and the
planners apply the rule of the device's architecture to the row's top-k width
and segment count.  Both records carry one rule, for `sm_107a`.  The keyword
`dkv_direct` of the entry points (default
`None` = the rule) forces (`True`) or disables (`False`) it, and the resolved
value is part of the binding key (the public wrapper maps
`CAKE_PPLX_DSA_DKV_DIRECT=auto|1|0` onto the keyword when the caller
leaves it at `None`); for a packed row on an architecture whose rule reads
the segment count (`sm_107a`) the key also holds the rule's decision, so a
new document count reuses the binding of the same drain.  Both
paths add the same FP32 contributions into the same rows in another summation
order.  Without the natural stages every `dkv_acc`
call takes the cast's accumulate path.

| record | `sm_100a` | `sm_103a` | `sm_107a` |
|---|---|---|---|
| RoPE (`dkv_acc` only) | `S >= 4 T` | `S >= 4 T` | `S >= 4 T` and `T * K_eff < 112 S + 2^20` |
| NoPE (fresh `dkv` and `dkv_acc`) | `S >= T` | `S >= T` | `S >= T` and (`T * K_eff < 112 S + 2^20` or `S > 131,072`) |

`K_eff = min(K_slots, ceil(S / num_segments))`; a row outside the condition
takes the permuted accumulator and the cast.

#### Direct accumulation on `sm_107a`

On VR200 (`sm_107a`, CUDA 13.5) the natural-layout drain of the main stage is
slower than the permuted one: on the NoPE `p8949_nope` row (`T = 16,384`,
`S = 65,536`) the natural main stage takes 8.30 ms against 7.88 ms for the
permuted build.  The natural build issues local loads and stores and more
lane-transpose shuffles; its reduction traffic to L2 is the same as the
permuted build's.

The rule weighs what each path adds to the backward.  The cast path adds a
zero fill and a cast over the `S` accumulator rows plus two launches, a cost
proportional to `S`.  The rule models the natural drain's extra cost as
proportional to the main stage's work (a fit to the measured backward
crossovers, not to stage times): its `T * K_eff` (query,
selected slot) pairs, where
`K_eff = min(K_slots, ceil(S / num_segments))` bounds the keys a query can
select by the mean key count of a packed segment.  So the `sm_107a` rule of
both records keeps the base test and also takes the cast path where

    T * K_eff >= cast_slots_per_key * S + cast_min_slots    (112 and 2^20)

(on the NoPE record only while `S <= 131,072`, see below) and the natural
drain otherwise; equality takes the cast path.  Small rows,
small top-k and short-document packs have little main-stage work and keep
the natural drain whatever their `S / T`.  Backward medians of the two paths
on VR200 (8 fresh processes per path, 20 steps each, cold L2, eager):

| row | record | `T` | `S` | `K_slots` | segments | natural drain | permuted + cast | rule |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `p8949_nope` | NoPE | 16,384 | 65,536 | 2048 | 1 | 8.957 ms | 8.505 ms | cast |
| `nope_doc_65536` | NoPE | 65,536 | 65,536 | 2176 | 1 | 35.273 ms | 33.342 ms | cast |
| `nope_glm53_a` | NoPE | 16,263 | 269,564 | 2176 | 8 | 6.770 ms | 6.697 ms | natural (key cap) |
| `nope_tail_4096x65536` | NoPE | 4,096 | 65,536 | 2176 | 1 | 2.249 ms | 2.176 ms | cast |
| `p8949_rope_acc` | RoPE | 16,384 | 65,536 | 2048 | 1 | 10.277 ms | 10.015 ms | cast |
| `packed_glm_a_dstmap` | RoPE | 16,231 | 268,757 | 2048 | 8 | 8.654 ms | 8.171 ms | cast |
| `nope_glm53_a_p256` | NoPE | 16,263 | 269,564 | 1152 | 8 | 4.293 ms | 4.351 ms | natural |
| `nope_docs264x992_rank0_fixedtail` | NoPE | 16,368 | 270,072 | 2176 | 528 | 2.775 ms | 2.870 ms | natural |
| `nope_tail_1024x16384_k512` | NoPE | 1,024 | 16,384 | 512 | 1 | 0.188 ms | 0.204 ms | natural |
| `nope_short_1024_k128` | NoPE | 1,024 | 1,024 | 128 | 1 | 0.110 ms | 0.117 ms | natural |
| `nope_doc_16384_k128` | NoPE | 16,384 | 16,384 | 128 | 1 | 1.083 ms | 1.053 ms | natural |
| `nope_tail_2048x32768` | NoPE | 2,048 | 32,768 | 2176 | 1 | 1.141 ms | 1.138 ms | natural |

The constants of the work test are a fit over rows measured with both
paths (both programs, 1 to 528 segments, top-k 128 to 2176): the rule takes
the cast path on none of them where it was slower.  On eight rows measured
with both paths the work
test keeps the natural drain, the base test's choice, although the cast path measured
faster beyond the spread (natural / cast backward time - 1):
`nope_doc_16384_k128` 2.8 %, `nope_short_4096_k256` 1.3 %,
`nope_short_2048_k256` 0.5 %, `nope_tail_2048x32768` 0.3 %,
`nope_tail_4096x81920` 2.6 % and `nope_tail_4096x98304` 1.1 % (NoPE, one
segment), and `chunk_3884x267520` 1.5 % and `chunk_3943x268757` 2.1 % (RoPE
with `dkv_acc`, 8 segments).
At `K_slots` 2176 and one segment the work test takes the cast path up to
`S / T` 10.3 at `T = 1024`, 14.9 at 2048, 17.1 at 4096, 18.3 at 8192 and
18.9 at 16,384, and the natural drain above; the NoPE key cap below lowers
the bound to `S / T` 16 at `T = 8192` and 8 at `T = 16,384` (131,072 keys).

The NoPE rule also carries a key cap, `cast_max_keys` 131,072: a NoPE row
with more keys (and at least one key per query) takes the natural drain
whatever its work.  On long single-segment rows the cast path's advantage
is smaller at the largest key counts and turns into a loss: at `T = 32,768`
the cast path is faster up to 131,072 keys and slower at 163,840 keys (the
backward median in both runs, beyond the spread only in the 0.984 run); at `T = 16,384` it
is faster at 163,840 keys, at parity at 196,608 and 229,376 keys and slower
from 262,144 keys on.  The backward times below are what was measured; the
stages behind them were not timed separately.  Backward medians of the two
paths on single-segment NoPE rows at `K_slots` 2176, every one on the cast
side of the work test (VR200, 8 fresh processes per path, 20 steps each,
cold L2, eager):

| `T` | `S` | `S / T` | natural drain | permuted + cast | natural / cast |
|---:|---:|---:|---:|---:|---:|
| 32,768 | 65,536 | 2 | 18.340 ms | 17.499 ms | 1.048 |
| 32,768 | 98,304 | 3 | 18.712 ms | 17.795 ms | 1.052 |
| 32,768 | 131,072 | 4 | 19.660 ms | 19.156 ms | 1.026 |
| 32,768 | 163,840 | 5 | 19.520 ms | 19.910 ms | 0.980 |
| 16,384 | 65,536 | 4 | 9.133 ms | 8.675 ms | 1.053 |
| 16,384 | 131,072 | 8 | 9.520 ms | 9.272 ms | 1.027 |
| 16,384 | 163,840 | 10 | 9.940 ms | 9.791 ms | 1.015 |
| 16,384 | 196,608 | 12 | 10.272 ms | 10.281 ms | 0.999 |
| 16,384 | 229,376 | 14 | 10.731 ms | 10.766 ms | 0.997 |
| 16,384 | 262,144 | 16 | 10.781 ms | 11.209 ms | 0.962 |
| 16,384 | 278,528 | 17 | 11.177 ms | 11.376 ms | 0.983 |
| 16,384 | 294,912 | 18 | 11.158 ms | 11.369 ms | 0.981 |
| 16,384 | 303,104 | 18.5 | 11.259 ms | 11.542 ms | 0.975 |

The `T = 32,768` rows and the 163,840-key row at `T = 16,384` come from
later runs than the other rows, with the same protocol.  Another run of the
`T = 32,768`, 163,840-key row measured 19.600 ms and 19.916 ms (0.984), and
a second run of the 303,104-key row 11.192 ms and 11.520 ms (0.972).  The
cap is the largest key count of these rows where the cast path measured
faster at both `T` (32,768 and 16,384 query tokens).  It reads the total
`S`, not `S / num_segments`: the planners do not know the longest segment,
and a packed row (a rank of a zigzag split, say) can hold one long segment
next to short ones, which the mean underestimates.  Packed NoPE rows of
short segments above the cap measured faster on the cast path, and the cap
gives up these gains (natural / cast backward time - 1, a later run of both
paths with the same protocol; 8 segments, 247,211 to 270,336 keys):
`nope_glm53_a` 1.3 % (1.1 % in the table above), `nope_glm53_b` 1.1 %,
`nope_glm53_a_fixedtail` 1.2 %, `nope_glm53_a_fixedtail_compact` 1.3 %,
`nope_glm53_a_iid` 1.3 %, `nope_pack4_262k_rank0_fixedtail` 1.0 % and
`nope_pack4_262k_rank0_fixedtail_compact` 0.4 % (within the spread).  It
also gives up the cast path's gains on the measured NoPE rows of 131,073 to
163,840 keys (natural / cast backward time - 1): `T = 16,384`,
`S = 163,840`, one segment, 1.5 % (9.940 ms, 9.791 ms; the table above);
the same `T` and `S` in two segments (8192 query tokens over 8192 keys and
8192 over 155,648), 2.7 % (8.696 ms, 8.467 ms); and `T = 8192`,
`S = 147,456`, one segment, 1.6 % (4.813 ms, 4.736 ms; 1.9 % in a later
run, 4.763 ms and 4.675 ms).  The
RoPE rule has no key cap: its single-segment `dkv_acc` row at `T = 16,384`,
`S = 278,528`, `K_slots` 2048 measured at parity (12.368 ms natural,
12.320 ms permuted + cast), and its packed rows above the cap keep the cast
path's gains.  `sm_100a` and `sm_103a` keep their base rules.

### Explicit entry points, prepared runner and binding cache

Explicit forward / backward entry points without autograd, a prepared
allocation-free runner (`prepare_dsa_train`, CUDA-graph capturable) and the
workspace sizing helper live in `cake_backend.py`.  A binding is validated and
resolved once per input geometry (shapes, strides, dtypes, device, alignment
and the call options): the prepared runner carves its scratch from the
workspace it is given; the eager entry points keep the resolved plans in a
bounded, lock-protected cache (`BINDING_CACHE`, 64 entries by default,
`BINDING_CACHE.configure(capacity=...)` resizes it -- the public
wrapper applies `CAKE_PPLX_DSA_TRAIN_BINDING_CACHE_CAPACITY` through it
when that variable is set; least recently used first out; a plan owns only the
full-length `topk_length` vector it materializes when the caller passes none)
and, per call, allocate the outputs
and the backward scratch from the caching allocator and launch the stages
through the generated positional launchers of `cake_launch.py`.  Every launch
runs on torch's current stream of the tensors' device, a side stream or a
capture stream alike: a launch that passes torch tensors needs no stream
context, since tvm-ffi binds each binding call to that stream; the NoPE
forward's pre-bound launches (operands converted to tvm-ffi views ahead of the
stage launch) and the prepared runner issue theirs inside an explicit tvm-ffi
stream context.  During
CUDA-graph capture the eager entry points never store a plan in the cache;
they reuse a cached plan only when it owns no `topk_length` vector (the call
passes its own lengths, as the autograd entries do), since such a plan holds
no storage, and otherwise plan privately.  A cache hit of that kind also skips
the capture query.  The same cache, under its own tagged keys, holds what the
NoPE autograd backward predicts: per backward key without `dout`, the binding
the last backward ran and that `dout`'s geometry.  The NoPE autograd forward
looks it up after its launch, and a backward whose `dout` has that geometry
runs the binding with no key completion and no cache lookup; another `dout`
layout takes the lookup and replaces the prediction.  A binding planned during
capture is never remembered.  `BINDING_CACHE.clear()` drops the predictions
with the bindings, and a backward whose forward read a prediction before the
clear takes the lookup.  The rope varlen entry decides its index route on
every call; on the stage route its cached binding is keyed by the geometry of the
activations, the per-document rows and a caller cut, plus `causal` and the
scale -- not by the prefix sums, whose length (the document count) is a launch
value of each call -- and holds the forward plan with the stage's resolved
entry and the launch values that follow from that geometry -- no tensor.  The
entry looks the binding up first: a hit checks only what the key does not hold
(the structure and device of the prefix sums, and that the NoPE record is still
the registered one), any other call takes the full route decision.  Rows with
strides `(K_slots, 1)` at a vector-aligned start reach the stage as they are,
other layouts through their flat alias.  The stage and the forward are both
launched before the autograd function runs; it receives the forward's outputs
and launches nothing itself.  The bindings encode the
tensor maps by value, so a step is its
kernels -- at 64 heads and with the head-window builds above 64 (on the
two-window bridge, see "Query heads above 64 (head windows)") one launch for
the forward; three for the single-pass backward
(`bwd_delta`, `bwd_main`, `bwd_cast`); `2 + 2 x chunks x passes` for the
key-range-pass backward (`bwd_delta`, then `bwd_compact` + `bwd_main_pass` per
token chunk and pass, `bwd_cast`)
-- plus the two fills of the FP32 dK/dV accumulators; a direct binding (see
"Direct accumulation from the main stage") launches the natural-layout variant
of the main stage instead and has neither the fills nor the cast.  A call without query rows
(`T == 0`) returns empty outputs and zero gradients without launching (on
`sm_107a` a call over more than 64 heads on CUDA tensors raises
`NotImplementedError` instead, see "Query heads above 64 (head windows)");
`S == 0` is rejected.

Host cost through the autograd wrapper: the `Function.backward` runs on
PyTorch's autograd device thread, where the two thread handoffs and the Python
body add a few hundred microseconds per backward that are not in this package
-- a trivial `autograd.Function` with the same saved tensors and gradient
shapes shows the same cost, and no synchronization is involved.  In a
GPU-bound training step this is hidden behind the backward kernels (6-8 ms at
4k tokens).  Host-bound loops should call `cake_backend.forward` /
`cake_backend.backward` directly or capture the prepared runner into a CUDA
graph (`benchmarks/bench_cake_dsa_sparse_attention_nondet_train.py --host-us` reports both; `--host-calls`
/ `--host-rounds` set the sample size).

## Kernel structure of one training step

* `fwd`: one CTA per query token gathers its top-k keys once (TMA gather) and
  produces `out`, the natural-log `lse` and the BF16 output residual `o_lo`.
  On `sm_107a` its scanner-warp build `fwd_scan` (416 threads) runs in its
  place, with the same arguments, grid and outputs.  Under `with_o_lo=False`
  its twin `fwd_no_olo` (`fwd_scan_no_olo` on `sm_107a`) runs instead: same
  arguments and grid, `out` and `lse` only.  The forward writes `o_lo` with an
  L1 no-allocate / L2 evict-first cache hint: nothing reads it before the
  backward's launch.  In every build the softmax role waits for the
  shared-memory reads of its last `out` TMA stores before it exits.
* `bwd_delta`: `delta = rowsum(dO * (O + O_lo))`, one (token, head) row per warp
  (`bwd_delta_no_olo` under `with_o_lo=False`: `delta = rowsum(dO * O)`); the
  kernel splits a row into token `row / 64` and head `row % 64`, which a launch
  over more than 64 heads addresses through virtual `dout` strides (see "Query
  heads above 64 (head windows)").
* `bwd_main`: one CTA per query token (20 warps: gather, compute, reduce, MMA,
  load and metadata roles) recomputes S and P from the BF16 Q and the gathered
  K, forms dP and dS, accumulates dQ / dQ_rope in tensor memory (written once
  per row) and scatters the per-token dK/dV and dK_rope
  contributions with vectorized FP32 `red.global.add` into the accumulators.
* `bwd_compact` / `bwd_main_pass`: the key-range-pass form of the main stage
  for the DRAM regime (see below).  Per pass, `bwd_compact` (one warp per
  token) writes the token's keys inside the pass range, in slot order and
  under the main kernel's validity rules (`-1`, `>= S`, `topk_length`), to
  `key_scratch` with their count in `pass_counts`; `bwd_main_pass` is the same
  main kernel consuming that list, carrying the token's FP32 dQ / dQ_rope
  partial in `dq_partial` between passes (`dq_mode` 1: store, 2: load-add-
  store, 3: load-add and BF16 output).
* `bwd_cast`: converts the FP32 accumulators to the natural `[S, 512]` /
  `[S, 64]` BF16 outputs (or FP32 in the `dkv_fp32` mode), or adds them into
  the caller's packed FP32 rows through the optional destination-row map
  (`dkv_acc` / `dkv_dst_map`).
* `bwd_main_natural` / `bwd_main_pass_natural`: the main stage (single-pass /
  pass form) whose reduce warps add straight into the caller's packed FP32
  rows (see "Direct accumulation from the main stage"); selected instead of
  `bwd_main` / `bwd_main_pass` for a `dkv_acc` binding that meets the record's
  size rule, with no `bwd_cast` afterwards.  The reduce traffic of every main
  stage carries an L2 `evict_last` policy for the accumulator lines.

Launch grids are functions of the problem scalars in `cake_launch.py`
(`num_queries` CTAs for `fwd` / `fwd_scan`, `bwd_main`, `bwd_main_pass` and their
natural-layout variants,
`num_queries * 8` for `bwd_delta` (`ceil(T * H / 64) * 8` above 64 heads),
`ceil(num_queries / 4)` for `bwd_compact`,
`ceil(num_kv * 18 / 256)` for `bwd_cast`).

### Key-range passes for the DRAM regime

With many keys the FP32 dK/dV accumulators (2304 B per key) outgrow the L2,
and the `red.global.add` scatter of `bwd_main` runs at DRAM speed.  The host
then runs the backward in `P = ceil(S * 2304 B / 100 MiB)` passes over
disjoint key ranges (`R = ceil(S / P)` keys each), so that the accumulator
slice one pass touches stays L2-resident: `bwd_delta`, then per pass
`bwd_compact` + `bwd_main_pass` over the whole row, then `bwd_cast`.  The
policy the record carries (`key_pass_policy`: L2 budget 100 MiB, 2304 B per
key, workspace budget 640 MiB, token chunk multiple 128, at most
`max_passes = 4` passes, and one rule per architecture under `rules`) takes
the pass path only for `P > 1` and under the rule of the device's
architecture, measured in paired sweeps per target:

| target | tail rows (`2 T <= S`) | passes | floor (one-chunk row / tail row) | tail cap |
|---|---|---|---|---|
| `sm_100a` | no | `P` | - / - | - |
| `sm_103a` | yes | `P` | - / 131,072 keys | 3 |
| `sm_107a` | yes | 2 | 131,072 / 131,072 keys | - |

A row qualifies when it is one token chunk (`T <= 4224` tokens at top-k
2048: the chunk is the largest multiple of 128 whose pass scratch fits the
workspace budget) or, where the rule admits tail rows, when `2 T <= S`; a
qualifying row below its floor stays single-pass, and the count is the rule's
fixed value or `P`, capped for tail rows of more than one chunk.  So at top-k
2048 on `sm_100a`: `T <= 4224` and `S >= 45,512`, two passes at `S = 65,536`,
three at `131,072`, 4k x 4k rows and 32k-token rows single-pass; on `sm_103a`
also three passes on the 32k-token rows of 131,072 or more keys; on `sm_107a`
two passes on every qualifying row of 131,072 or more keys and one pass
below.  A count above `max_passes` runs one pass: beyond four passes the
per-pass FP32 dQ-partial round trip and pipeline fill outweigh the L2 benefit
(the `S = 225,280` / `267,520` one-chunk rows run one pass on `sm_100a` and
`sm_103a`; `sm_107a`'s two passes are under the cap).  A record without
`rules` (the first release) is the `sm_100a` rule everywhere.  Above 64
heads the head-window builds plan with the rule of the architecture and the
overrides of `cake_backend.HEAD_WINDOW_KEY_PASS_RULES`: on `sm_103a` no tail
rows, so every tail row (more than one 4,224-token chunk, `2 T <= S`, 131,072
or more keys, one segment) runs one pass there, the 16,380-token rows of
524,160 and 1,048,320 keys among them (see "Query heads above 64 (head
windows)").  The tokens of a multi-pass backward run in chunks of 4224 (every
chunk through every pass, `token_base` = the chunk's first token; 2,176 with
the head-window builds above 64 heads, see "Query heads above 64 (head
windows)"), so the pass scratch holds one chunk.  The passes split the whole key row
into equal ranges, which matches an index row whose keys spread over the whole
row (one document); in a packed multi-segment row every token's keys lie
inside its own segment, so the policy applies to one-segment rows only
(`num_segments > 1` -- the varlen entry's `len(cu_seqlens_k) - 1`, or the
flat rope entry's keyword -- plans one pass unless `key_passes` overrides).
Since the policy reads only whether the
row has more than one segment, the backward binding keys hold that class, not
the count, and on `sm_107a`, whose `dkv_direct` rule reads the count, also the
rule's decision (see "Direct accumulation from the main stage"): a training
step with a new document count reuses the backward binding of the last one
unless that decision changes.  The passes add
`min(T, 4224) * (147,456 + 4 * topk + 4)` B to the workspace (`dq_partial`,
`key_scratch`, `pass_counts`; 608 MiB at `T = 4096`, top-k 2048;
`dsa_train_workspace_size` includes them).  dQ is
still written once per row from the carried FP32 partial (its partial sums
are re-associated, so it can differ from the single pass in the last FP32
places), and the dK/dV reductions are
the same reds in another order.  `key_passes=` on the entry points overrides
the policy (`1` = single pass, `n` = that many passes); a program without the
pass stages serves the single pass only.

## Query heads above 64 (head windows)

The rope form takes `H = q_latent.shape[1]` query heads, `H` in 64, 96 and 128
(`cake_backend.SUPPORTED_HEADS`).  Every kernel of the program computes 64
heads of a token per work item.  `H = 64` runs the plan of 0.6.1 (the 64-head
release), launch for launch, whatever the registry ships.  `H = 96` and `128`
run, per phase (forward, backward), one of two plans:

| plan | where | forward | backward main stage |
|---|---|---|---|
| head-window builds | the program ships the phase's build for the device's architecture | one `fwd_hw` launch | one launch of its `*_hw` build (per token chunk and pass with key passes) |
| two-window bridge | otherwise (0.7.0) | two window launches and their copies | statistics copies, then one launch per window (per token chunk and pass) |

`cake_backend.head_window_plan(H, arch, ...)` returns the head-window stages a
call runs (`None` for a phase on the bridge).  A phase takes its build only
where the registry carries it for the architecture, among the record's stages
and the program's stage order, with its generated launcher and grid
(`cake_backend.head_window_stage`); the builds ship for `sm_100a` and
`sm_103a`.  A partial registry mixes the plans per phase.

### Head-window builds

The windows are `W0 = [0, 64)` and `W1 = [64, 128)`.  Work item `i` is window
`i & 1` of token `i >> 1`, so the two windows of a token run side by side and
the second finds the token's gathered key rows in L2.  The rows of `W1` past
`H` (heads 96 to 127 at `H = 96`) are zero-filled by the query loads and
skipped by every store.

| stage | head-window build (`cake_backend.HEAD_WINDOW_STAGES`) |
|---|---|
| `fwd` / `fwd_no_olo` | `fwd_hw` / `fwd_hw_no_olo` |
| `bwd_main` / `bwd_main_natural` | `bwd_main_hw` / `bwd_main_natural_hw` |
| `bwd_main_pass` / `bwd_main_pass_natural` | `bwd_main_pass_hw` / `bwd_main_pass_natural_hw` |

* Arguments.  A build takes the arguments of its 64-head stage plus
  `num_heads = H` and `wsel = -1` (both windows of every token), and the grid
  of its 64-head stage over `num_queries = 2 T` items.  Its query, `out`,
  `dout` and `dq` operands are the caller's whole `[T, H, *]` tensors (the
  tensor maps carry the strides, so packed `[T, H, 576]` and pre-absorption
  `[T, H, 256]` views work as at 64 heads); `lse`, `o_lo` and `delta` are
  addressed at `t * H + h`.  The host checks this contract where it binds the
  launch values (`ValueError` otherwise).
* Forward.  One `fwd_hw` launch (`fwd_hw_no_olo` under `with_o_lo=False`)
  writes `out`, `lse` and `o_lo` of every head of the call's `[T, H]` outputs:
  no scratch and no copy.  With length derivation the `W0` item of a token
  stores its length.
* Backward, in launch order:
  1. The `dout` copy, as on the bridge (step 1 below).
  2. One `bwd_delta` (or `bwd_delta_no_olo`) launch over the `T * H` rows, as
     on the bridge (step 2 below).
  3. The main stage's build over the `2 T` items, reading the call's `lse` and
     the `delta` rows directly and storing every head's `dq` once.  Its
     `token_base` / `token_step` count items and keep both items of a token
     adjacent: ascending from item 0, or descending from item `2 T - 1` where
     the 64-head plan walks its tokens descending (`sm_103a`, single pass).
     With key-range passes the backward runs per token chunk and pass:
     `bwd_compact` over the chunk's tokens (token space), then the pass build
     over the chunk's `2 n` items, ascending from item `2 * token_base`, so the
     two windows of a token read its one compaction row.
  4. One `bwd_cast` at the end (none on a direct binding).
* Key-pass chunks.  Each item has its own FP32 `dq_partial` row (147,456 B),
  so the token chunk is planned at two items per token: the largest multiple
  of 128 tokens whose `2 x 147,456 + 4 * topk + 4` B per token fit the
  640 MiB budget -- 2,176 tokens at top-k 2048 (4,224 at 64 heads).  The pass
  count follows the record's rule as at 64 heads, with the head-window
  overrides `cake_backend.HEAD_WINDOW_KEY_PASS_RULES` (only `sm_103a`: no
  tail-row passes; `head_window_key_passes_rule` says where they apply: above
  64 heads where the single-pass head-window main build ships).  On GB300 the
  single pass of the head-window build is 15-20 % faster than the record's
  three passes on every tail row timed (top-k 2048, 96 and 128 heads:
  `T = 16,380` with `S = 147,420` to `1,048,320`, `T = 8,448` and `4,352` with
  `S = 131,072`; backward 39.5 / 40.5 ms against 45.6 / 46.8 ms at 96 heads
  and `S = 524,160` / `1,048,320`); two or three forced passes over 2,176-,
  4,352- or 16,380-token chunks stay 5-15 % slower than one at `S = 524,160` /
  `1,048,320`.  A packed row of four documents plans one pass with or without
  the override.  [est] The three passes keep each pass's FP32 dKV accumulator
  slice (`S / 3 x 2,304` B) nearer the L2 budget; with head windows the extra
  launches, the FP32 dQ-partial round trip and the compaction of every pass
  cost more than that locality gains, also where the slice is close to the
  100 MiB budget (113 MB at `S = 147,420`).  The two-window bridge and the
  64-head plans keep the record's rule.
* Launches.  The forward is one launch; the single-pass backward is the
  optional `dout` copy, `bwd_delta`, one main launch and `bwd_cast`.  The
  runner's launch keys are the build names (`("fwd_hw",)` as
  `forward_stages`; `"bwd_main_hw"`, or `("bwd_main_pass_hw", chunk, pass)`
  pass keys, in `backward_order`); no `WindowLaunch` and no copy step besides
  `HeadStep("dout_copy")`.
* Workspace: no window scratch (`fwd_lse`, `fwd_o_lo`) and no window
  statistics (`window_lse`, `window_delta`); `delta` and `dout_copy` as on the
  bridge; with key passes `dq_partial` holds `2 x` the chunk's tokens rows,
  `key_scratch` and `pass_counts` one row per token.
* Numerics.  The builds give every head the same 32-row rescale groups as the
  bridge, so `out`, `lse`, `o_lo` and `dq` equal the bridge's bit for bit at
  the same key-pass count (on the `sm_103a` tail rows the head-window plan runs
  one pass where the bridge runs three, and its `dq` sums the keys in the
  single pass's order);
  dK / dV add the same terms through FP32 atomics in another order.

### Two-window bridge

The bridge runs the 64-head stages on two 64-head windows, `W1` first and then
`W0`, with no kernel, build or launcher change:

| `H` | `W1` (first) | `W0` (second) | heads both windows cover | pad heads of `W1` |
|---|---|---|---|---|
| 96 | `[32, 96)` | `[0, 64)` | `[32, 64)` | local `[0, 32)` |
| 128 | `[64, 128)` | `[0, 64)` | none | none |

* Operands.  A window's `q_latent`, `q_rope`, `out`, `dout`, `dq_latent` and
  `dq_rope` are the head slices `[:, head0 : head0 + 64]` of the call's
  tensors; the tensor maps carry the strides, so packed `[T, H, 576]` and
  pre-absorption `[T, H, 256]` views work as at 64 heads.  Its `lse`, `o_lo`
  and backward statistics are contiguous `[T, 64]` scratch.
* Forward.  Per window, in launch order, the `fwd` (or `fwd_no_olo`) launch
  writes `out[:, window]` in place and its `lse` / `o_lo` into one reused
  scratch: `fwd_lse` `[T, 64]` FP32 and, with `with_o_lo`, `fwd_o_lo`
  `[T, 64, 512]` BF16.  A device copy then moves the heads the window owns
  into the `[T, H]` outputs: `[64, H)` for `W1`, `[0, 64)` for `W0`.  `W0`
  runs last, so `out`, `lse` and `o_lo` of the overlap heads `[H - 64, 64)`
  all come from `W0`.  Both windows take the call's `topk_length` and length
  derivation (the derivation writes the same lengths twice).
* Backward, in launch order:
  1. A `dout` whose token stride is not `H` head strides
     (`dout.stride(0) != H * dout.stride(1)`, such as the head-major
     `[512, T * 512, 1]` view) is copied once into the workspace (`dout_copy`
     `[T, H, 512]` BF16).  A dense or packed-slice `dout` is read in place.
  2. One `bwd_delta` (or `bwd_delta_no_olo`) launch over the `T * H` (token,
     head) rows.  The kernel splits a row into `row / 64` and `row % 64`, so
     the launch passes the virtual `dout` strides `(64 * hs, hs)` (`hs` the
     head stride, 512 on the copy), which address row `t * H + h` exactly, and
     a grid of `ceil(T * H / 64) * 8` blocks.  It writes the `[T, H]` `delta`.
  3. Per-window statistics `window_lse` / `window_delta` `[2, T, 64]` FP32, by
     device copies: `W0` takes heads `[0, 64)`; `W1` takes heads `[H - 64, H)`,
     and its pad heads (local `[0, 128 - H)`, the heads `W0` owns) get
     `lse = +inf` and `delta = 0`.  The kernel forms `P = exp2(S * scale_log2 -
     lse)` and `dS = P (dP - delta) sm_scale`, so a pad head has `P = 0` and
     `dS = 0`: it adds exactly zero to dK / dV, and its `dq` (zero) is
     overwritten by `W0`.
  4. The main stage per window, `W1` then `W0`, both adding into the same FP32
     accumulators (zeroed once) or, on a direct binding, into `dkv_acc`.  With
     key-range passes the order is token chunk, then window, then pass: per
     chunk every pass of `W1`, then every pass of `W0`.  `dq_partial` (64
     heads per token) holds one window's partial between passes and the next
     window reuses it.
  5. One `bwd_cast` at the end (none on a direct binding).
* Numerics.  Per head they are those of the 64-head program: the windows add no
  reduction, and dK / dV sum the contributions of both windows through the
  same FP32 atomics.
* Launches.  The forward is two window launches, each followed by its copy
  step.  The single-pass backward is the optional `dout` copy, `bwd_delta`,
  the statistics copies, two main launches and `bwd_cast`.  The copies are
  torch operations on the current stream: no host synchronization, and the
  prepared runner stays CUDA-graph capturable.  The runner's launch keys are
  `cake_backend.WindowLaunch(phase, head0, key)` for a window launch (`key`
  the 64-head stage name or pass key) and `cake_backend.HeadStep(name,
  head0)` for a copy step (`"fwd_copy"`, `"dout_copy"`, `"window_stats"`).
* Workspace.  Per token, the forward adds `fwd_lse` (256 B) and, with
  `with_o_lo`, `fwd_o_lo` (64 KiB); the backward holds `window_lse` and
  `window_delta` (512 B each).  The dQ partials of a key-pass backward stay 64
  heads per token.
* Cost.  At 96 heads the forward and the main stage each process 128 heads
  (the overlap heads twice); the delta runs once over the 96 heads and the
  cast once.  The forward adds the scratch copies of `lse` and `o_lo`; the
  backward gathers the keys and drains the dK / dV reductions once per window.

### Both plans

* Workspace: per-call scratch (`workspace_layout(..., num_heads=H)`,
  `dsa_train_workspace_size(..., num_heads=H)`, which sizes the plan the
  device's architecture serves).  The backward holds `delta` (`4 * H` B per
  token) and, when the `dout` layout needs it, `dout_copy` (`H` KiB per
  token).  `dsa_train_workspace_size` sizes `dout_copy` from its keyword:
  `None` (the default) includes it for every backward over more than 64
  heads, the upper bound for any `dout` layout; `True` / `False` give the
  exact size.
* Retained state: what a training step keeps from its forward to its backward
  (`retained_layout`, `dsa_train_retained_size`).  Per token: `lse`
  (`4 * H` B) and, with `with_o_lo`, `o_lo` (`H` KiB); the flat entry without
  caller lengths also keeps the derived `topk_length` (4 B); the packed entry
  (`varlen_topk = K_slots`) keeps its compacted rows `indices`
  (`4 * K_slots` B) and their counts (4 B) on both of its routes.  The returned
  `out` (`H` KiB) is the call's output and is not counted (the autograd node
  saves it too, so it lives until the backward).  Under
  `torch.utils.checkpoint(..., use_reentrant=False)` nothing is retained.
* A call without query rows (`T = 0`) returns its empty outputs and zero key
  gradients before any plan: nothing launches.
* Not served above 64 heads: the deterministic package, the NoPE program and
  its sink form (`ValueError`; the deterministic and NoPE query checks name
  the rope form, the sink check says the sink form serves 64 query heads
  only).  On `sm_107a` an eager, autograd or step call with more than 64
  heads raises `NotImplementedError` before any kernel of the package
  launches -- also without query rows, on a CUDA tensor -- and so does
  `prepare_dsa_train`, until the window stages are validated there (a
  prepared runner refuses `T = 0` with `ValueError` at every head count).
  This holds whatever builds the registry declares for `sm_107a`.

## NoPE (zero-rope) program

A second registry record, `cake_dsa_h64_nope_train`, carries the native kernels
of the zero-rope form of the same attention (GLM-5.3 geometry: 64 heads, 512
shared latent dimensions and no rope part, softmax scale `1 / 16`, top-k slot
counts such as 2176 or 1152 -- any `K_slots >= 1`).  The score is the latent
product alone, `softmax(scale * q . kv^T) . kv`, and the backward produces one
combined gradient per key row, `dkv = dK + dV`, natural layout, FP32 `[S, 512]`.
The rope entry points above are unchanged; the NoPE entries are additive.

Slot-count rule: the host accepts every int32 `indices [T, K_slots]` with
`K_slots >= 1` (any width, odd widths included; `topk_length` may be shorter
per row) and rejects `K_slots <= 0`, a non-int32 dtype and a row stride below
`K_slots` explicitly (`validate_dsa_nope_inputs`); the kernels bound the per-row
work by `topk_length` and read no slot beyond `K_slots`.

```python
from cake_pplx.cake_dsa_sparse_attention_nondet_train import (
    cake_dsa_sparse_attention_nondet_nope,
    cake_dsa_sparse_attention_nondet_nope_varlen,
)

out = cake_dsa_sparse_attention_nondet_nope(
    q, kv, indices, topk_length=None, softmax_scale=None,
    return_lse=False, dkv_acc=None, dkv_dst_map=None, num_segments=1,
    sink=None)
out = cake_dsa_sparse_attention_nondet_nope_varlen(
    q, kv, gather_kv_indices, cu_seqlens_q, cu_seqlens_k,
    max_seqlen_q=None, max_seqlen_k=None, causal=False,
    topk_length=None, softmax_scale=None, return_lse=False,
    dkv_acc=None, dkv_dst_map=None, check_cu_seqlens=True, sink=None)
```

* `q [T, 64, 512]` BF16 (contiguous or a strided token / head view, consumed in
  place); `kv [S, 512]` BF16 or its one-head form `[S, 1, 512]` (squeezed on the
  host; the gradient reaches that leaf through the view); `indices [T, K_slots]`
  int32 global key rows, `topk_length` and the validity rules as above.
  `softmax_scale` defaults to `1 / 16`.  Without `topk_length` the flat entry
  follows the table size (`nope_flat_index_mode`): below
  `NOPE_CONVERT_SLOT_LIMIT` slots the forward alone reads the rows as given and
  derives each row's length (last valid slot + 1) into a `[T]` vector the
  backward reads with the caller's rows (one launch; the caller's tensor is
  then a saved tensor, so autograd rejects an in-place edit of it before the
  backward); at or above the limit it compacts the rows on device in one launch
  of the index-preparation stage's flat mode right before the forward (valid
  keys `0 <= idx < S` first in ascending slot order, then `-1`; the valid count
  as the row's length) into a fresh `[T, K_slots]` buffer and a `[T]` vector
  the backward reads, so rows with `-1` holes cost their valid slots and the
  caller's tensor is untouched.  With `topk_length` the rows and lengths reach
  the kernels as given.
* Rows with holes and the kernel-level entries: `forward_nope`,
  `backward_nope` and the prepared runner read rows as given (work up to the
  last valid slot, or `topk_length`).  `nope_compact_indices(indices, S,
  topk_length=None)` returns `(rows, counts)` -- one flat stage launch, any
  row stride, optional `out=` / `counts=` buffers for a capturing caller -- to
  pass as `indices` / `topk_length` to those entries and to the flat autograd
  entry with caller lengths.  (A backward over raw hole rows could instead run
  the key-range-pass form at one pass -- `nope_bwd_compact` writes each row's
  valid keys compacted, `nope_bwd_main_pass*` consumes them with `dq_mode 0` --
  which is not wired up in this round.)
* Backend entries (`cake_backend`): `forward_nope`, `backward_nope`,
  `forward_nope_varlen` (the packed forward without autograd: the public
  entry's index route; returns `(out, lse, o_lo, indices, topk_length)`, the
  global rows and lengths of that route -- position-preserving rows with last
  valid slot + 1 below the slot limit, compacted rows with their valid counts
  on the stage route -- that `backward_nope` takes with
  `num_segments = cu_seqlens_k.numel() - 1` and a caller-zeroed `dkv_acc`),
  `nope_compact_indices`, `nope_globalize_indices(..., compact=False)` (one
  stage launch over per-document rows: the position-preserving rows and last
  valid slot + 1 by default, bitwise `offset_gather_kv_indices`; `compact=True`
  the compacted rows and counts, with an optional `topk_length` cut),
  `prepare_dsa_nope_train` / `DSANopeTrainRunner`, `dsa_nope_train_workspace_size`,
  `validate_dsa_nope_inputs`, `CakeDSASparseAttentionNondetNopeFunction` (14 forward inputs,
  `sink` and `with_o_lo` last and optional; gradients for `q`, `kv` and a `sink`).  The binding cache is shared; the
  NoPE keys are tagged (`"nope_fwd"` / `"nope_bwd"`, `"nope_fwd_no_olo"` / `"nope_bwd_no_olo"` under
  `with_o_lo=False`, and `"nope_bwd_prediction"` for the
  autograd backward's predictions) and never collide with the rope keys.
* `dkv` semantics.  Fresh (no `dkv_acc`): `backward_nope` returns a new natural
  FP32 `[S, 512]` buffer -- filled by the natural-layout main stage adding into
  the zero-filled buffer when the record's direct rule of the device's
  architecture selects it (`dkv_direct`: `min_keys_per_query` 1 on `sm_100a` /
  `sm_103a`, so whenever the row has at least one key per query -- every packed
  causal problem; measured faster than the permuted drain on every row there;
  on `sm_107a` only while `T * K_eff < 112 S + 2^20` or above 131,072 keys,
  see "Direct accumulation on `sm_107a`"), else -- a row outside the rule, or
  `dkv_direct=False` -- by
  the cast stage writing the
  permuted accumulator out densely (`out_f32`).  Accumulate
  (`dkv_acc` FP32 `[S_dst, >= 512]`, optional `dkv_dst_map`): `+=` into columns
  `0:512` of the caller's rows by the same two
  paths and `None` for `dkv`; chunked backward passes of one long row add
  their FP32 contributions into one FP32 buffer (no BF16 cast in between).  Through autograd without
  `dkv_acc`, the FP32 `dkv` becomes the `kv` gradient and the autograd engine
  casts it to the leaf's dtype (BF16 for a BF16 leaf); `dkv_acc` may be the
  tensor or a zero-argument callable producing it (or an `(accumulator,
  dkv_dst_map)` pair) at backward time, as for the rope entries.  Both NoPE
  autograd `Function`s save every produced backward operand -- the saved
  outputs, `o_lo` (a zero-size placeholder under `with_o_lo=False`), the global
  index rows, the lengths -- through
  `ctx.save_for_backward` (seven tensors in a fixed order) and keep only
  options, geometry keys and the caller-owned accumulator on `ctx`, so a
  checkpointed forward retains nothing.
* `dout` may be a head-major view (strides `[512, T * 512, 1]`) or any strided
  `[T, 64, 512]` view under the layout rules of `q` (unit channel stride, token
  and head strides multiples of 8 elements, 16-byte base): it is read in place
  by every entry, the eager `backward_nope` and a prepared runner alike -- the
  main stage through a stride-carrying tensor map, the delta stage through the
  token / head strides (elements) and the element offset the launcher passes
  (`dout_token_stride`, `dout_head_stride`, `dout_offset = 0`).  No host copy.
* Varlen: per-document indices become global key rows on device; the route
  follows the table size (`nope_varlen_index_mode`).  Below
  `NOPE_CONVERT_SLOT_LIMIT` slots (both small bench tables, 1024 x 128 and
  1024 x 512, sit below it) and without caller lengths the forward
  converts the rows itself in its single launch (`convert = 2`): its scan lanes
  write the position-preserving global rows -- the global key or `-1` at the
  slot it came from -- into the fresh `indices_out` the backward reads and the
  last valid slot + 1 as the row's length (bitwise the torch twin
  `offset_gather_kv_indices(...)`), so no host gap separates a preparation
  launch from the forward; a small table with holes scans them like the
  previous version did (routing reads the table size alone: the host cannot
  see holes without a device sync).  At or above the limit, and always with a
  caller `topk_length`, one launch of the record's index-preparation stage runs
  right before the forward: `index_prep` (one warp per index row, four rows per
  128-thread CTA, any index row stride read in place) with `compact = 1`, every
  row stably compacted -- valid keys first in ascending slot order, then `-1`
  -- and its valid count as the length, after the position cut of a caller
  `topk_length` passed as `length_in`; the forward and the saved backward read
  the stage's rows and counts (bitwise the torch twin
  `offset_gather_kv_indices(..., compact=True)`).  The limit is a measured
  crossover (the largest measured table at which the one-launch forward stays
  within 0.95x of the stage route on the eager forward and 1.02x on the
  GPU-busy step, plus one, rounded up to the next power of two when the next
  measured table also stays within 1.02x on the GPU-busy step).  The forward
  stages (`fwd`, `nope_fwd` and their scanner-warp builds `fwd_scan`,
  `nope_fwd_scan`) launch with the programmatic-dependent-launch
  attribute (`use_pdl` in their launch records, a fixed attribute of their
  bindings): `index_prep` signals `launch_dependents` at entry and the forward
  kernels execute `griddepcontrol.wait` after their barrier / TMEM prologue,
  before any global read, so behind the stage the forward's prologue overlaps
  the stage; behind any other kernel the attribute changes nothing.  The host prepares
  the forward's plan, outputs and launch arguments and converts its operands
  before the first launch, then issues the forward from those pre-bound values
  inside the same stream context, so only the forward's binding call sits
  between the two launches (it hides behind the stage's kernel).  The forward
  program's conversion operands (`cu_seqlens_q`, `cu_seqlens_k`,
  `indices_out`, `num_docs`, `search_iters`, `causal`, `convert` at the end of
  its launch arguments) take three values: `convert = 0` for global rows (the
  operands never read), `convert = 2` (`NOPE_CONVERT_MASK`) for the
  position-preserving conversion of the small-table route, and `convert = 1`
  (`NOPE_CONVERT_COMPACT`) for the compacting single launch -- the same stable
  compaction and valid counts as the stage, bitwise -- which no size selects
  (`_packed_forward_nope(..., "kernel")` by name only).  A registry without the
  stage takes the position-preserving single launch at every size and no
  caller `topk_length`.  `causal=True` drops keys after the query's own
  position, query segments are tails of their key prefix.  The NoPE entries
  have no torch fallback.
* Backward host order: the autograd wrappers settle their backward at forward
  time (the binding key without `dout`, the operands autograd does not save,
  the caller's `dkv_acc` validated once, the predicted binding), so with a
  `dout` of the predicted layout the backward's host work before its first
  launch is the layout checks of the saved rows and `dout`, the operands, the
  `delta` allocation and the delta stage's values -- no key completion and no
  cache lookup.  The delta stage reads only `dout`, `out` and `o_lo` (`dout`
  and `out` under `with_o_lo=False`) and launches
  first; the `dq` allocation, the other fresh buffers, the main stage's values
  and the zero fills of the FP32 buffers the call owns follow while it runs,
  then the main (and cast) stages.  Every check precedes the delta launch, and
  the caller's `dkv_acc` is first written by the main stage, so an error after
  the delta launch (an allocation failure) leaves it unchanged.  The
  kernel-level `backward_nope` keeps the same launch order.
  Forward host order: the NoPE autograd wrappers allocate and bind the
  forward, check the accumulator arguments (an invalid `dkv_acc` tensor, with
  its `dkv_dst_map`, is rejected before any launch; a callable's result is
  checked in the backward) and launch; then, while the forward kernel runs,
  they settle what their backward needs (its binding key without `dout`, with
  the accumulator's geometry when the accumulator is a tensor or absent; the
  saved tensors, which PyTorch packs once the forward returns; the predicted
  binding when `q` or `kv` requires a gradient).
  Every call checks the structure of the prefix sums (`validate_cu_seqlens`:
  contiguous int32 `[num_docs + 1]` of one length) and never synchronizes with
  the device; `check_cu_seqlens=True` also checks their values -- start at 0,
  non-decreasing, `cu_seqlens_q[-1] == T`, `cu_seqlens_k[-1] == S` (a shorter
  or longer key window is rejected, not truncated), `seqlen_q <= seqlen_k` per
  document -- at the cost of one blocking device-to-host copy before any launch
  (skipped during CUDA-graph capture).  The rope varlen entry launches the same
  stage (compacting, caller lengths as the cut) and keeps the torch twin as its
  fallback; its structural checks are unchanged.

Stages of the record, in launch order and with the kernels they share with the
rope program: `nope_fwd` (own kernel, 320 threads; on `sm_107a` its
scanner-warp build `nope_fwd_scan`, 352 threads), `nope_bwd_delta` (the
strided delta kernel, shared with the rope program's `bwd_delta`, 256 threads:
the `[T * 64, 512]` rows of `dout` addressed as `dout_offset + token *
dout_token_stride + head * dout_head_stride + channel`, strides and offset
multiples of 8 elements),
`nope_bwd_main` /
`nope_bwd_main_natural` (own kernels, 640 threads: no rope operands, one
accumulator `dkv_f32` with `dkv_stride >= 512`, a multiple of 4, and the
destination map; a persistent token loop, so the host launches them through
`nope_bwd_launch_grid(num_queries, num_kv, sm_count, arch)`: one CTA per SM --
`min(num_queries, SMs)`, CTA `b` taking tokens `b, b + grid, ...` of the
launch's `num_queries` tokens (the whole row or the chunk) -- while the launch's
FP32 `dkv` window `num_kv x 2048 B` is within 128 MiB on SM100 / SM103 and
64 MiB on SM107, else one CTA per token;
the SM count is read once per device), `nope_bwd_compact` (the rope compaction kernel),
`nope_bwd_main_pass` / `nope_bwd_main_pass_natural` (the key-range-pass forms)
and `nope_bwd_cast` (the rope cast kernel with `rope_groups = 0`, grid
`ceil(num_kv * 16 / 256)`).  `index_prep` (own kernel, 128 threads, grid
`ceil(T / 4)`; arguments `compact`, `length_in`, `has_length_in` after
`write_out`) is the index preparation of both varlen entries (document mode,
`compact = 1`), of the flat NoPE entry without caller lengths and of
`nope_compact_indices` (flat mode: `has_docs = 0`, `flat_bound = S - 1`, the
prefix-sum operands never read); it is launched right before the forward and
is not part of a prepared runner's step.  The key-range-pass policy of the record plans one
pass on every row (`max_passes` 1, no per-architecture rules: one pass measured
fastest on every row, build and GPU of this form), so the single-pass main stage
is the default and `dq_partial` is not allocated; `key_passes=n` on the entry
points forces the pass form, which plans with 2048 B of accumulator per key and
a dQ partial of 131,072 B per token: the token chunk of a multi-pass backward is
4736 at `K_slots` 2048 / 2176, 4864 at 1152 and 4992 at 128 (the 640 MiB budget,
a multiple of 128).

`generated_program_available(..., program=cake_jit.NOPE_PROGRAM)` reports
whether the record registers the stages a call needs for the device; the NoPE
GPU tests additionally skip while `build.json` lacks a registered NoPE library
(the export delivers the libraries).

### Trainable per-head sink (NoPE)

The NoPE entries -- the flat `cake_dsa_sparse_attention_nondet_nope` and step
pair `cake_dsa_sparse_attention_nondet_train_nope_forward` / `_backward`, the
packed `cake_dsa_sparse_attention_nondet_nope_varlen` and step pair
`cake_dsa_sparse_attention_nondet_train_nope_varlen_forward` / `_backward`
(and, in this backend, `forward_nope`, `backward_nope`, `forward_nope_varlen`
and `prepare_dsa_nope_train`) -- take an optional `sink`: FP32 `[64]`,
contiguous, on the device of `q`.
Per row `t` and head `h`, over the valid slots `j`:

```text
s_j      = softmax_scale * <q[t,h], kv_j>
lse[t,h] = log(sum_j exp(s_j) + exp(sink[h]))      # the sink is not scaled
out      = BF16(sum_j exp(s_j - lse) * kv_j)       # the sink has no value term
delta    = <dout[t,h], out + o_lo>                  # <dout[t,h], out> under with_o_lo=False
dsink[h] = -sum_t exp(sink[h] - lse[t,h]) * delta[t,h]
```

* Forward: the stage `nope_fwd_sink` (the forward's sink build: the forward's
  arguments plus `sink`, its grid and launch geometry) runs in place of
  `nope_fwd`; on `sm_107a` `stage_for` resolves it to `nope_fwd_scan_sink`,
  the sink build of the scanner-warp forward.  The packed entries run the
  same stage over the packed rows on every route (in-kernel index conversion
  or index preparation); a zero-length query segment adds no row.  It folds the sink into the final normalisation of the FP32
  softmax and rounds `out` and the residual `o_lo` once from that sum (not a
  rescaled sink-free output).  `lse` includes the sink.  A row without a valid
  slot gives `out = o_lo = 0` and `lse = sink[h]` (`-inf` for a `-inf` head);
  a `-inf` head behaves as without a sink; a sink far above every score of its
  row (beyond FP32's `exp` range) gives `out = 0`, `lse = sink[h]` and finite
  gradients.
* Backward: the main stages are unchanged -- they recompute
  `P = exp(s - lse)` from the saved `lse`, which already includes the sink, so
  `dq` and `dkv` follow the formulas above on every route (single pass or
  `key_passes`, natural or permuted drain, `dkv_acc` / `dkv_dst_map`,
  head-major `dout`).  The stage `nope_bwd_dsink` (256 threads) runs once over
  all `T` rows right after `nope_bwd_delta`, outside the key passes and token
  chunks.  It splits the rows into fixed chunks of `NOPE_DSINK_CHUNK_TOKENS`
  (64) tokens, writes one FP32 partial per (chunk, head) to the workspace
  (`dsink_partial`, `ceil(T / 64) x 64` FP32); the CTA that completes the
  grid (`dsink_ticket`: a 32-bit completion counter, zero-filled by the host
  before every backward) sums the partials in one fixed order -- four
  lane-strided sums over the chunks (chunks `r, r + 4, ...`), folded in lane
  order -- so `dsink` depends on `T` only, not on the SM count or the grid.  A row without a valid slot (`delta = 0`) and a `-inf`
  head contribute exactly 0.  FP32 throughout.
* Returns: the step backwards return `(dq, dkv, dsink)` when they are given
  the `sink` their forward ran with (`(dq, dkv)` without one, as before), and
  `(dq, None, dsink)` with a tensor `dkv_acc`; the packed step backward sums
  `dsink` over all packed rows of the call.  The autograd entries return
  `dsink` as the gradient of `sink` (the flat entry's thirteenth input, the
  packed Function's sixteenth; `None` for the `with_o_lo` input after it); the prepared runner's `backward()` returns
  `(dq, dkv, dsink)`.  `lse` stays non-differentiable.  A call without query
  rows (`T = 0`) launches nothing on every architecture and gives a zero
  `dsink`.
* Checks: metadata only (dtype, shape, contiguity, the device of the other
  tensors), no value check and no device synchronization; `+inf` and NaN
  values are not supported.  The binding keys end with the sink's geometry, so
  a sink call and a call without one never share a plan, and new `T` / `S`
  values reuse the compiled stages.
* State: a call with a sink retains one more tensor for its backward -- the
  reference to the caller's `sink` (`ctx.save_for_backward`, so non-reentrant
  checkpointing releases and recomputes it like the others).  The workspace
  grows by the sink-gradient scratch, `ceil(T / 64) * 256 + 4` bytes
  before alignment and `ceil(T / 64) * 256 + 256` after it
  (`dsa_nope_train_workspace_size(..., sink=True)`, the regions
  `dsink_partial` / `dsink_ticket` of `workspace_layout_nope`); `dsink [64]`
  is an output.
* The deterministic package takes the same `sink` on its NoPE entries, with
  the same stages and the same `dsink` order.

## Without the output residual (`with_o_lo=False`)

Every entry -- the four autograd entries, the three step forwards and the
three step backwards, and in this backend `forward`, `backward`,
`forward_nope`, `backward_nope`, `forward_nope_varlen` and both `prepare_*`
runners -- takes `with_o_lo`, a keyword-only Python `bool` (default `True`;
anything else raises `ValueError`); the three autograd `Function`s take it as
their last `apply` input (see below).  No environment variable sets it: it
chooses the accuracy class of the gradients and stays visible at the call
site.

```text
with_o_lo=True   forward writes out, lse, o_lo     delta = rowsum(dout * (out + o_lo))   FP32
with_o_lo=False  forward writes out, lse           delta = rowsum(dout * out)            FP32
```

* `True` (default) is the behaviour of every earlier release: the same stages,
  libraries, binding keys and saved tensors.  `o_lo` keeps the gradients
  accurate on peaked rows (rows whose softmax puts almost all weight on one
  key).
* `False` runs the twins of `cake_jit.NO_OLO_STAGES` in place of the forward
  and the delta stage: `fwd_no_olo`, `nope_fwd_no_olo` and
  `nope_fwd_sink_no_olo` on `sm_100a` / `sm_103a` (their scanner-warp builds
  `fwd_scan_no_olo`, `nope_fwd_scan_no_olo` and `nope_fwd_scan_sink_no_olo` on
  `sm_107a`), and `bwd_delta_no_olo` / `nope_bwd_delta_no_olo` (one library
  shared by both programs, every architecture).  The forward twins take the
  default stage's arguments, grid and launch geometry and leave the `o_lo`
  operand unwritten; the delta twin reads `dout` and `out` alone, the form
  cuDNN's DSA backward uses, so on peaked rows the gradient error grows to
  that class.  The main, compaction, cast and sink-gradient stages are the
  default ones (`dsink` follows the twin's `delta`).  `out` and `lse` have
  the same accuracy in both modes.  The sink tests bound each head's `dsink`
  error by `5e-3 * max |dsink_ref|`; `with_o_lo=False` derives `delta` from
  the BF16 `out`, so its per-head `dsink` bound is 1.5 times the default
  (`7.5e-3 * max |dsink_ref|`), and every other bound and the naive BF16 tier
  are the same in both modes.
* No `o_lo` is allocated, written or saved: `H` KiB less written and saved per
  query token (64 KiB at 64 heads, 2 GiB at `T = 32,768`; above 64 heads the
  forward window scratch `fwd_o_lo` is not allocated either).  The kernels' `o_lo` operand is a
  cached zero-size BF16 placeholder chosen from the plan's mode alone, and
  the autograd `Function`s keep their saved-list length and order (the `o_lo`
  slot holds a zero-size placeholder), so non-reentrant checkpointing still
  matches by position.  At 64 heads the workspace layouts are the same in
  both modes; above 64 heads the forward's window `o_lo` scratch
  (`fwd_o_lo`) exists only with `with_o_lo=True`.
* Step entries: the step forwards return `None` in the `o_lo` slot (`(out,
  lse, None)`, the packed form `(out, lse, None, indices, topk_length)`); a
  step backward takes `o_lo=None` with `with_o_lo=False`.  `o_lo` is `None`
  exactly when `with_o_lo` is `False`: a tensor with `False`, or `None` with
  `True` on a call with query rows, raises `ValueError` before anything is
  planned, allocated or launched.  A default call without query rows
  accepts `o_lo=None` (it launches nothing).  A runner prepared with
  `with_o_lo=False` refuses an `o_lo=` tensor; its `o_lo` and the third
  element of `forward()` are `None`.
* Binding keys: the first slot gains `_no_olo` (`"fwd_no_olo"`,
  `"bwd_no_olo"`, `"rope_varlen_fwd_no_olo"`, `"nope_fwd_no_olo"`,
  `"nope_bwd_no_olo"`); every other slot keeps its position and every default
  key is unchanged.  The NoPE autograd backward's prediction keys carry the
  tag through `key[:7]`.
* Autograd `apply`: `with_o_lo` is the last input of each `Function` (the rope
  `Function`'s sixteenth after `launched`, the flat NoPE `Function`'s
  fourteenth after `sink`, the packed NoPE `Function`'s seventeenth after
  `sink`); the entries pass it only when it is `False`, so a default call
  applies the same inputs as before.  Each backward returns one gradient per
  input it received.
* A checkout whose record lacks a twin for the device's architecture raises
  `NotImplementedError` naming the twin and the architecture;
  `generated_program_available(..., with_o_lo=False)` is `False` there.

## Layout of this package

* `cake_jit.py` -- the `MODULES` registry (one record per program, the rope
  `cake_dsa_h64_train` and the NoPE `cake_dsa_h64_nope_train`, each shared by
  every architecture it compiles for; per stage the `library` stem of its
  `build.json` entries, the FFI entry and the launch geometry -- block,
  cluster, cooperative, dynamic shared memory, PDL -- that the export rewrites),
  the stage names of both programs (`STAGES` / `NOPE_STAGES`, `PROGRAMS`) and
  the loader glue: `library_name(stage, arch, program)` is the `build.json`
  entry `<library>_<arch>` that `load_cake_dsa_train_module(stage, arch,
  program)` loads through `cake_build`.  Stages of the two programs that
  share a kernel name the same library stem.  A stage entry may narrow the
  record's architectures with its own `arches` (`fwd_scan` / `nope_fwd_scan` /
  `nope_fwd_scan_sink` and their twins `fwd_scan_no_olo` /
  `nope_fwd_scan_no_olo` / `nope_fwd_scan_sink_no_olo`: `sm_107a`; `fwd` /
  `nope_fwd` / `nope_fwd_sink` and their twins `fwd_no_olo` /
  `nope_fwd_no_olo` / `nope_fwd_sink_no_olo`: `sm_100a`, `sm_103a`);
  `NO_OLO_STAGES` maps each default forward and delta stage to its
  `with_o_lo=False` twin; `SCAN_STAGES` and
  `stage_for` resolve a forward on `sm_107a` to its scanner-warp stage, and
  `library_name` returns that stage's library.
* `cake_build.py` + `build.json` -- the loader shared by every Cake-exported
  package (byte-identical copies) and this package's build manifest (schema
  `cake-build/1`, written by the export): one entry per `<library>_<arch>`
  (twenty-six libraries over twenty-two kernel sources, four of them compiled
  a second time with `-DDKV_NATURAL=1`; twenty on each architecture, sixty
  entries: the six scanner-warp forwards on `sm_107a` only, and `fwd` /
  `nope_fwd` / `nope_fwd_sink` and their `_no_olo` twins on `sm_100a` and
  `sm_103a`) with
  its translation
  units, digests, nvcc flags (`--use_fast_math` per stage, plus the natural
  variants' specialization defines), closure digest and `aot/<name>.so`
  path.  `cake_build.load_library(name)` loads the prebuilt library when
  `aot/` holds it (next to the builder's `aot/build_info.json` sidecar, which
  must agree with the library's exported `build_cuda_version()`) and otherwise
  JIT-builds the entry through the package's JIT shim (`gen_jit_spec`, sources
  digest-verified first; `sm_107a` needs CUDA 13.5 or newer, an nvcc that
  lists `compute_107`).
  `aot/` is never tracked (`.gitignore`).
* `cake_launch.py` -- generated positional launchers and grid functions over
  the kernels' own argument names, one per distinct argument plan or grid
  rule (`STAGE_ARGS` / `LAUNCH` / `GRID` keyed by stage name).
* `cake_backend.py` -- validation, workspace layout, varlen index offsetting,
  the prepared runners, the autograd Functions, the eager entry points and the
  non-determinism alert of the backward (`_alert_nondeterministic`).
* `csrc/cake_ffi_checks.h` -- the TVM-FFI helper header the bindings include
  (tensor-argument checks, `TVM_FFI_CHECK_CUDA_ERROR`, `build_cuda_version`),
  shipped with the package so that it compiles from its own `csrc/` alone.
* `csrc/cake_dsa_h64_train/` -- generated kernel and binding translation units:
  one pair per kernel (the main stage and its key-range-pass form each carry
  their natural-layout variant as the compile-line specialization
  `DKV_NATURAL=1` of the same pair, under the variant's own kernel symbol and
  binding namespace; the delta, compaction and cast pairs serve both
  programs), compiled once per library and architecture for the
  architectures of its `build.json` entries (`.clang-format` disables
  formatting: the sources are identity-checked by the digests in
  `build.json`).  The only package header they include is
  `csrc/cake_ffi_checks.h` (the binding units).

## Status

The registry holds the rope program for `sm_100a`, `sm_103a` and `sm_107a` with the forward
(on `sm_107a` its scanner-warp build `fwd_scan`),
backward preprocess, backward main (single-pass and key-range-pass form with
its compaction, each with its natural-layout direct-accumulation variant) and
cast stages plus the key-range-pass and direct-accumulation policies, exported
from the kernel snapshot named in the pull request, and the NoPE program's
record with its eighteen stages (the forward and, on `sm_107a`, its
scanner-warp build `nope_fwd_scan`; the delta, compaction and cast kernels
shared with the rope program; the trainable-sink stages `nope_fwd_sink` (on
`sm_107a` its scanner-warp build `nope_fwd_scan_sink`) and `nope_bwd_dsink`;
the five `with_o_lo=False` twins `nope_fwd_no_olo`, `nope_bwd_delta_no_olo`,
`nope_fwd_scan_no_olo`, `nope_fwd_sink_no_olo` and
`nope_fwd_scan_sink_no_olo`)
and its own key-range-pass and
direct-accumulation policies (see "NoPE (zero-rope) program").  The rope
record also carries the twins `fwd_no_olo` (`fwd_scan_no_olo` on `sm_107a`)
and `bwd_delta_no_olo` (see "Without the output residual").  The rope form
serves 96 and 128 query heads on `sm_100a` and `sm_103a`, through the
head-window builds (`fwd_hw`, `fwd_hw_no_olo` and the four `bwd_main*_hw`
builds, for those two architectures) where the record carries them and else
through the two-window bridge; `sm_107a` serves 64 heads until its window
stages are validated (see "Query heads above 64 (head windows)").

Tests: `tests/experimental/test_cake_dsa_sparse_attention_nondet_train.py` (the backend suite: imports
only this package; skips without a registered program or a compute capability
10.0 / 10.3 / 10.7 device) and `tests/experimental/test_cake_dsa_sparse_attention_nondet_train_api.py`
(the public entry points, the environment variables the wrapper
resolves into this package's keyword switches and the behaviour under
`torch.use_deterministic_algorithms`) and
`tests/experimental/test_cake_dsa_sparse_attention_nondet_train_heads.py` (96
and 128 query heads; its host tests record and check every launch of the
windows through `tests/experimental/test_helpers/cake_dsa_sparse_attention_host_launches.py`;
its `test_large_t_heads` -- T = 16380, top-k 2048, the per-rank row -- runs
only with `CAKE_PPLX_DSA_LARGE_TESTS=1` and is skipped otherwise);
the CPU-only
`tests/experimental/test_cake_dsa_sparse_attention_nondet_train_names.py`
checks the public names.  Accuracy is judged by `tests/cake_naive.py` through
`tests/experimental/test_helpers/cake_dsa_sparse_attention_naive.py` (the FP64
reference and the PyTorch BF16 form of the same call; `tests/conftest.py` prints
one `TIER` line per judged output); index, length and map outputs are checked
exactly.  Benchmarks:
`benchmarks/bench_cake_dsa_sparse_attention_nondet_train.py` and, for 96 and 64
query heads against the non-Cake baselines,
`benchmarks/bench_cake_dsa_sparse_attention_nondet_train_heads.py`.
