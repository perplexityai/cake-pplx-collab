# Cake DSA sparse-attention training backend, deterministic (SM100 / SM103 / SM107)

Backend package of `cake_pplx.cake_dsa_sparse_attention_det_train`, the
deterministic package of the native DeepSeek Sparse Attention training kernels
(top-k sparse MLA with absorbed queries; 512 latent + 64 rope query/key
dimensions, 512 value dimensions).  The rope program takes 64, 96 or 128 query
heads on every architecture (see "96 and 128 query heads"); the NoPE program
takes 64.

The entry points of `cake_pplx.cake_dsa_sparse_attention_det_train` are stable
APIs; this backend package is experimental (no compatibility guarantee for its
own helpers).  The libraries compile for `sm_100a`, `sm_103a` and `sm_107a`.
On `sm_107a` each forward runs its scanner-warp build: the stages `fwd_scan`,
`nope_fwd_scan` and `nope_fwd_scan_sink` ship for `sm_107a` only, and `fwd` /
`nope_fwd` / `nope_fwd_sink` ship for `sm_100a` and `sm_103a` (see "Layout of
this package").  The `sm_107a`
libraries need CUDA 13.5 or newer (`min_cuda` 13.5 in their `build.json`
entries).  On `sm_100a` the libraries of the stages `fwd`, `bwd_main`,
`bwd_main_pass`, `bwd_redo` and `bwd_compact`, each with its `nope_` form,
of the head-count builds `fwd_h128`, `bwd_main_hw`, `bwd_main_pass_hw` and
`bwd_redo_hw`, and of the NoPE sink build `nope_fwd_sink` need CUDA 12.9 or
newer (`min_cuda` 12.9); the other `sm_100a` libraries, the sink-gradient
stage `nope_bwd_dsink` and the heads build `bwd_delta_heads` included, need
12.8.  The feature does not participate in automatic backend
selection, autotuning or trace-apply.  Its libraries are loaded prebuilt from
`aot/` when the package ships them and JIT-built through the package's JIT
shim from `build.json` otherwise (see "Layout of this package").

## Mode and names

The module name states the mode and no entry point takes a `deterministic=`
keyword.  This is the `det` package: every output repeats bit for bit for the
same inputs, device architecture and package version.  Bits are not promised
across package versions, across architectures, or against the
non-deterministic package `cake_pplx.cake_dsa_sparse_attention_nondet_train`.
The result does not depend on the SM count, the launch grid, other work on the
device (side-stream kernels), CUDA-graph or eager execution, activation
checkpoint recompute or the saved set, repeated destination rows, or the order
of the tokens.  Nothing raises or warns under
`torch.use_deterministic_algorithms(True)` (no alert is registered), and no
entry falls back to another package.

| entry | autograd Function |
|---|---|
| `cake_dsa_sparse_attention_det` | `CakeDSASparseAttentionDetFunction` |
| `cake_dsa_sparse_attention_det_varlen` | `CakeDSASparseAttentionDetFunction` (over the compacted rows) |
| `cake_dsa_sparse_attention_det_nope` | `CakeDSASparseAttentionDetNopeFunction` |
| `cake_dsa_sparse_attention_det_nope_varlen` | `CakeDSASparseAttentionDetNopeVarlenFunction` |

The entries take the arguments of the non-deterministic entries, except `with_o_lo` (the det entries always write `o_lo`).
`key_passes` and `num_segments` select the key-range passes of the backward
(see below): the varlen entries pass `len(cu_seqlens_k) - 1`, and the flat
entries and the step backwards of both forms take the count as the keyword
`num_segments` (default 1; on the rope form a count above 1 plans one pass
unless `key_passes` overrides, as the varlen entry does for the same rows;
the NoPE policy plans one pass whatever the count).  dK / dV
do not depend on it; dQ is summed
through an FP32 partial between passes, so its bits follow the plan, and
every output repeats bit for bit for the same inputs, `key_passes` and
`num_segments`.  `dkv_direct` is accepted and validated as there and has no
effect: the accumulation into `dkv_acc` has one form.  The facade reads
`CAKE_PPLX_DSA_CHECK_DST_MAP` and
`CAKE_PPLX_DSA_TRAIN_BINDING_CACHE_CAPACITY` (as the non-deterministic
facade does, for this package's own cache); the backend reads no environment
variable.

### What repeats bit for bit

| output | how it is formed |
|---|---|
| `out`, `lse`, `o_lo` (with or without a sink) | one fixed-order forward kernel per token |
| derived `topk_length` (no caller lengths) | written by the forward kernel (last valid slot + 1) |
| varlen global rows and lengths | one index-preparation launch (or the forward's own conversion), fixed per row |
| `dq_latent`, `dq_rope` (NoPE `dq`) | summed in a fixed tile order, written once per row |
| `dkv_latent`, `dk_rope` (BF16 or `dkv_fp32`), NoPE `dkv` | integer sums on a per-row grid, one rounding into the output |
| `dkv_acc` columns `0:576` (NoPE `0:512`), with or without `dkv_dst_map` | integer sums on a per-row grid, then one FP32 addition per element |
| NoPE `dsink` | FP32 sums over fixed 64-token chunks, then one fixed-order sum over the chunks (see "Trainable per-head sink") |

## dK / dV reduction order

The backward launches every registered stage of
`cake_jit.BACKWARD_STAGES_OF[program]` on every call, in this order (the NoPE
program prefixes the names with `nope_`); no stage is chosen by a tensor value:

1. `bwd_delta`: `delta = rowsum(dout * (out + o_lo))` `[T, 64]` FP32, the
   per-query bound terms `qstat [T, 4]`
   (largest magnitudes and norms of `dout`, `delta` and the query parts,
   summed over the heads in a fixed order), and the zero fill of the global
   words.
2. `bwd_norm`: the key norms `||kv_latent[k]||` and their maximum; it zeroes
   the row-bound words.
3. `bwd_bound`: per destination row, a fixed-point upper bound `B_row` on the
   row's `sum |dK|`, `sum |dV|` over every valid (query, key) pair that maps to
   it (latent and rope parts separately), and the row's pair count `N`.  While
   `T * topk < 2 ** 26`, one reduction per pair adds the pair count and a
   12-bit term to one word, and a pair whose 12-bit term is small adds its
   full-resolution term to a second word, so `B_row` keeps the full resolution
   (global word 10 records the packed launch for `bwd_grid`); a larger launch
   uses one word per part and a count word.
4. `bwd_grid`: the row's grid exponent `e = ceil(log2 B_row) - 30 - L` with
   `L = 12` (`cake_backend.GRID_SHIFT`) and the overflow counters' skip
   threshold `2 ** (30 - ceil(log2 N))`; it zeroes the accumulator row and
   counters of every row a pair touches.
5. `bwd_main`: dQ, and each pair's contribution `x` added to the int32
   accumulator of its destination row as `n = rni(x * 2 ** -e)`
   (round-to-nearest-even; the scaling by a power of two is exact).  Integer
   addition is associative, so the sum is the same bits in any order.  Per row
   four overflow counters bound `sum |n|`; a row whose counters stay below
   `2 ** 31` is proven to have left the int32 range in no order.
6. `bwd_redo_prep` / `bwd_redo`: a row whose counters do not prove the range
   moves to a grid at least `L + 1` bits coarser
   (`cake_backend.MIN_BUMP`), is zeroed and is recomputed by the main
   kernel's redo build, which visits every token and adds only the pairs of the
   flagged rows (no dQ).  Both stages are launched on every backward; the redo
   runs one persistent CTA per SM (`cake_backend.redo_launch_grid`) and
   returns at once when no row is flagged (no host synchronization).
7. `bwd_fold`: `v = float(n) * 2 ** e` (one rounding) into the fresh outputs
   (every row; a row no pair touches is `+0.0`), or `dkv_acc[r] = prev + v`
   (one IEEE FP32 addition with subnormals kept; an element whose integer is
   zero keeps its bits, so a planted `-0.0` stays `-0.0`; a row no pair touches
   is not written).

Repeated destination rows of `dkv_dst_map` add into the same integers, so they
combine exactly before the one FP32 addition.

Key-range passes.  When the key row is much larger than the L2, the backward
splits every token's keys into `P` equal key ranges and runs, per range and in
place of `bwd_main` (NoPE: `nope_bwd_main`), the non-deterministic package's
compaction of the token's keys in the range (`bwd_compact`; the NoPE stage
`nope_bwd_compact` runs the same library) and the pass build of the main
kernel over the compacted keys (`bwd_main_pass` / `nope_bwd_main_pass`).  dQ
goes through an FP32 partial (147,456 B per token; NoPE 131,072 B) between
passes.  The integer sums and the overflow counters do not depend on the split,
so dK/dV keep the bits of one launch.  dQ is summed through the FP32 partial,
so it can differ from the dQ of one launch in the last FP32 places.  `P`
follows that package's policy (`cake_backend.plan_key_passes`); a row of more
than one token chunk (4224 tokens at top-k 2048 for rope) takes one pass here,
because every pass runs all tokens in one launch, and so does a rope backward
above 64 heads (from 0.13.1; see "96 and 128 query heads").  The NoPE
record's policy is that package's NoPE policy, which never plans a second
pass.  `key_passes` overrides `P` in both programs.  The redo stays one
launch.

Non-finite values: a pair whose contribution is NaN or infinite has no integer
image, so the counters flag its destination row, the redo flags it again, and
the fold writes every element of that row as NaN (fresh outputs and `dkv_acc`
alike); the other rows are unaffected.  The rows written as NaN:

| non-finite input | dK/dV rows written as NaN |
|---|---|
| `q_latent`, `q_rope`, `dout` or `out` of token `t` | every row token `t` selects |
| `kv_latent[k]` or `k_rope[k]` | the destination row of key `k` only |
| `lse[t]` = -inf or NaN | every row token `t` selects |
| `lse[t]` = +inf | none (the probabilities are 0) |

A finite input large enough (about `1e38`) to overflow an FP32 bound term
counts as non-finite.  Where the non-deterministic package writes NaN or
infinities element by element, this package writes NaN across the whole row.
`dq` follows the main kernel's arithmetic (a NaN input gives NaN in the rows
it reaches).

Known limitation: the redo does not compact the pairs of the flagged rows.
When any row is flagged, the redo costs about one more main-kernel pass over
all pairs (without dQ).

## Storage and the `o_lo` rule

The accumulator region of one backward holds, per destination row, the int32
accumulator (576 columns; 512 for NoPE; the row-bound words live in the row
until the grid stage reads them), 32 B of overflow counters, the grid words
(8 B; 4 B for NoPE) and the pair count (4 B).  `delta` and the global words
are never part of the region.  The rest lives in output storage that is dead
while it is used:

* `qstat` (`[T, 4]` FP32) and the key norms (4 B per key row) at the front of
  dQ (`dq_latent`; the NoPE `dq`; 64 KiB per query token) when they fit (`S`
  up to about `16380 T`) and dQ starts 16-byte aligned: `bwd_delta` and
  `bwd_norm` write them, `bwd_norm` and `bwd_bound` read them, and then the
  main stage writes every dQ element (`cake_backend.dq_scratch_layout`).
* With fresh rope dK/dV outputs, the redo mark and the redo list slot (4 B
  each per row) at the front of `dkv_latent`: `bwd_redo_prep` writes them,
  `bwd_redo` reads them, and the fold writes the output last
  (`cake_backend.redo_lists_fit`).  Otherwise (`dkv_acc`, NoPE) they are part
  of the region.
* After them, when it fits too, `delta` (`[T, 64]` FP32): `bwd_delta` writes
  it (it reads neither output), the redo is its last reader
  (`cake_backend.output_scratch_layout`, `cake_backend.delta_fits_output`:
  `2 * align(R * 4) + align(T * 256)` bytes within `dkv_latent`).  The global
  words stay in the workspace (the flagged-row and non-finite counts are read
  after the backward).
* Otherwise those parts stay in the workspace or the region.  The eager and
  autograd entries allocate per call only the global words and what stays
  there of `delta` and the region (no `topk_length` vector).

Peak memory against the non-deterministic package (same binding, allocated
bytes).  A prepared rope runner with fresh outputs holds the int32
accumulator where that package holds its FP32 accumulator (2304 B per row)
plus the counters, grid words and pair counts (44 B per destination row), and
no `delta`: it needs `align(R * 32) + align(R * 8) + align(R * 4) + 256 -
align(T * 256)` bytes more, which is less whenever `R <= ~5.8 T`.  A prepared
NoPE runner needs 48 B per destination row and the global words (one 256-byte
part) more: its fresh `dkv` is the integer accumulator through the fold, the
main stage writes dQ before the redo, and the redo reads `delta`, so no output
storage is dead while the counters, grid words, pair counts and redo lists
are live.  A backward of `P > 1` key-range passes adds the same pass scratch
as the non-deterministic package (the FP32 dQ partial, the compacted keys and
the pass counts of every token).  The NoPE autograd entry that adds into
`dkv_acc` carves the region from `o_lo` and allocates the global words (64 B)
beside `delta` per call; the global words may not alias `o_lo` (`bwd_delta`
zeroes them while it reads `o_lo`).

* The autograd entries use the storage of their saved `o_lo` (64 KiB per query
  token) for the region when it fits: `bwd_delta` is the only stage that reads
  `o_lo` (no later kernel reads `out` or `o_lo`).  The exact condition
  (`cake_backend.carve_fits`, 256-byte aligned parts) with every part in the
  region is
  `align(R * 2304) + align(R * 32) + align(R * 8) + 3 * align(R * 4) + align(S * 4) <= 65536 * T`
  for the rope program (`R` destination rows, `S` keys, `T` query tokens);
  for NoPE `align(R * 2048) + align(R * 32) + 4 * align(R * 4) +
  align(S * 4) <= 65536 * T`; the key norms and the redo lists drop out when
  they live in dQ or the output.  At `S = R` this is about `R <= 27.7 T`
  (rope) and `R <= 31.2 T` (NoPE).  A fresh NoPE `dkv` is its own integer
  accumulator (the fold rewrites it as FP32 in place), so only the per-row
  words go to the region.  A larger region is allocated per call.  Rows with
  more destination rows per query token than that therefore use more memory
  than the non-deterministic package, which adds into `dkv_acc` directly; for
  example a 4096-query backward chunk over 268757 keys mapped to 260611
  destination rows (a region of about 614 MB), a 2123-query segment over 67923
  keys (about 160 MB) and a NoPE 4096-query chunk over 269564 keys into
  `dkv_acc` (about 565 MB).
* After such a backward, `o_lo` no longer holds the forward's residual.  A
  second backward through the same graph (`retain_graph=True`) first runs the
  forward again into the saved `out`, `lse` and `o_lo` (same kernel, same
  operands: the same bits; derived lengths are derived again into a temporary).
  Under non-reentrant activation checkpointing the backward receives a
  recomputed `o_lo`.  Saved-tensor hooks such as
  `torch.autograd.graph.save_on_cpu` may hand the index rows, `q_latent` and
  `kv_latent` back in another layout (a column slice or an offset view comes
  back contiguous); the NoPE backward keys and plans by the tensors it
  receives, so the bits are those of the eager step.  A CUDA graph that
  captures forward and backward together replays correctly; a graph that holds
  only the backward and is replayed twice without a forward replay in between
  is not supported.
* The eager `backward` / `backward_nope` and the prepared runners take
  `o_lo_scratch` (default `False`: the region is allocated, or held in the
  runner's workspace; `dsa_train_workspace_size` / `dsa_nope_train_workspace_size`
  count it).  With `o_lo_scratch=True` a runner's `backward()` must follow its
  `forward()`; a second `backward()` raises outside graph capture.
* The kernels do not range-check `dkv_dst_map`; a value outside `[0, S_dst)`
  writes outside the region.  `check_dst_map=True` (or
  `CAKE_PPLX_DSA_CHECK_DST_MAP`) validates the values on every backward
  at the cost of one device synchronization; the check is skipped during CUDA
  graph capture.

No entry synchronizes with the host unless a check that says so is enabled
(`check_dst_map`, the varlen `check_cu_seqlens`).

### Measured peak memory and backward time

Measured with version 0.1.0 on GB200 (`sm_100a`), GB300 (`sm_103a`), R200
(`sm_107a`).

Peak allocated bytes of one forward + backward after a warm step (`runner`: a
prepared runner with the package defaults; `autograd`: the public entry through
autograd in the row's layout), for the rows of
`benchmarks/bench_cake_dsa_sparse_attention_det_train.py`; `+/-` is this
package's peak minus the non-deterministic package's at the same binding
(positive: this package needs more).

| row | path | sm_100a bytes | sm_100a +/- | sm_100a MiB +/- | sm_103a bytes | sm_103a +/- | sm_103a MiB +/- | sm_107a bytes | sm_107a +/- | sm_107a MiB +/- |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `doc_4096` | runner | 854,262,272 | -867,840 | -0.83 | 854,262,272 | -867,840 | -0.83 | 854,262,272 | -867,840 | -0.83 |
| `doc_4096` | autograd | 878,199,296 | -10,485,248 | -10.00 | 878,199,296 | -10,485,248 | -10.00 | 878,199,296 | -10,485,248 | -10.00 |
| `doc_16384` | runner | 3,417,047,552 | -3,472,896 | -3.31 | 3,417,047,552 | -3,472,896 | -3.31 | 3,417,047,552 | -3,472,896 | -3.31 |
| `doc_16384` | autograd | 3,512,795,648 | -41,942,528 | -40.00 | 3,512,795,648 | -41,942,528 | -40.00 | 3,512,795,648 | -41,942,528 | -40.00 |
| `doc_32768` | runner | 6,834,094,592 | -6,946,304 | -6.62 | 6,834,094,592 | -6,946,304 | -6.62 | 6,834,094,592 | -6,946,304 | -6.62 |
| `doc_32768` | autograd | 7,025,590,784 | -83,885,568 | -80.00 | 7,025,590,784 | -83,885,568 | -80.00 | 7,025,590,784 | -83,885,568 | -80.00 |
| `p8949_rope` | runner | 3,589,079,552 | -1,310,208 | -1.25 | 3,589,079,552 | -1,310,208 | -1.25 | 3,589,079,552 | -1,310,208 | -1.25 |
| `p8949_rope` | autograd | 3,569,418,752 | -155,188,736 | -148.00 | 3,569,418,752 | -155,188,736 | -148.00 | 3,569,418,752 | -155,188,736 | -148.00 |
| `doc_16231` | runner | 3,385,138,688 | -3,440,128 | -3.28 | 3,385,138,688 | -3,440,128 | -3.28 | 3,385,138,688 | -3,440,128 | -3.28 |
| `doc_16231` | autograd | 4,031,366,656 | +0 | +0.00 | 4,031,366,656 | +0 | +0.00 | 4,031,366,656 | +0 | +0.00 |
| `tail_2123x67923` | runner | 1,003,523,584 | +2,445,824 | +2.33 | 1,003,523,584 | +2,445,824 | +2.33 | 673,369,088 | +2,445,824 | +2.33 |
| `tail_2123x67923` | autograd | 1,100,545,024 | +160,027,648 | +152.61 | 1,100,545,024 | +160,027,648 | +152.61 | 770,095,616 | +91,004,928 | +86.79 |
| `packed_glm_a` | runner | 4,269,115,392 | +7,806,464 | +7.44 | 4,269,115,392 | -650,117,120 | -620.00 | 4,269,115,392 | -650,117,120 | -620.00 |
| `packed_glm_a` | autograd | 4,613,186,560 | +0 | +0.00 | 4,613,186,560 | +0 | +0.00 | 4,613,186,560 | +0 | +0.00 |
| `packed_glm_a_dstmap` | runner | 4,269,837,824 | +7,670,784 | +7.32 | 4,269,837,824 | -649,803,264 | -619.70 | 4,269,837,824 | -649,803,264 | -619.70 |
| `packed_glm_a_dstmap` | autograd | 4,594,457,088 | +0 | +0.00 | 4,594,457,088 | +0 | +0.00 | 4,594,457,088 | +0 | +0.00 |
| `chunk_4096x268757` | runner | 1,780,576,256 | +10,777,088 | +10.28 | 1,780,576,256 | +10,777,088 | +10.28 | 2,418,126,848 | +10,777,088 | +10.28 |
| `chunk_4096x268757` | autograd | 2,088,977,920 | +480,832,000 | +458.56 | 2,088,977,920 | +480,832,000 | +458.56 | 2,088,977,920 | +480,832,000 | +458.56 |
| `nope_doc_4096` | runner | 816,005,632 | +197,120 | +0.19 | 816,005,632 | +197,120 | +0.19 | 816,005,632 | +197,120 | +0.19 |
| `nope_doc_4096` | autograd | 851,461,120 | +512 | +0.00 | 851,461,120 | +512 | +0.00 | 851,461,120 | +512 | +0.00 |
| `nope_glm53_a` | runner | 3,770,834,944 | +12,939,776 | +12.34 | 3,770,834,944 | +12,939,776 | +12.34 | 3,770,834,944 | +12,939,776 | +12.34 |
| `nope_glm53_a` | autograd | 3,899,448,832 | +0 | +0.00 | 3,899,448,832 | +0 | +0.00 | 3,899,448,832 | +0 | +0.00 |
| `nope_chunk_4096x269564` | runner | 1,372,426,752 | +12,939,776 | +12.34 | 1,372,426,752 | +12,939,776 | +12.34 | 1,372,426,752 | +12,939,776 | +12.34 |
| `nope_chunk_4096x269564` | autograd | 1,960,145,408 | +565,006,848 | +538.83 | 1,960,145,408 | +565,006,848 | +538.83 | 1,960,145,408 | +565,006,848 | +538.83 |
| `p8949_nope` | runner | 3,367,043,584 | +3,146,240 | +3.00 | 3,367,043,584 | +3,146,240 | +3.00 | 3,367,043,584 | +3,146,240 | +3.00 |
| `p8949_nope` | autograd | 3,498,115,584 | +512 | +0.00 | 3,498,115,584 | +512 | +0.00 | 3,498,115,584 | +512 | +0.00 |

Rows where this package needs more memory than the non-deterministic package:
`tail_2123x67923` runner (sm_100a, sm_103a, sm_107a); `tail_2123x67923`
autograd (sm_100a, sm_103a, sm_107a); `packed_glm_a` runner (sm_100a);
`packed_glm_a_dstmap` runner (sm_100a); `chunk_4096x268757` runner (sm_100a,
sm_103a, sm_107a); `chunk_4096x268757` autograd (sm_100a, sm_103a, sm_107a);
`nope_doc_4096` runner (sm_100a, sm_103a, sm_107a); `nope_doc_4096` autograd
(sm_100a, sm_103a, sm_107a); `nope_glm53_a` runner (sm_100a, sm_103a, sm_107a);
`nope_chunk_4096x269564` runner (sm_100a, sm_103a, sm_107a);
`nope_chunk_4096x269564` autograd (sm_100a, sm_103a, sm_107a); `p8949_nope`
runner (sm_100a, sm_103a, sm_107a); `p8949_nope` autograd (sm_100a, sm_103a,
sm_107a).

Backward time of the non-deterministic package divided by this package's,
measured on version 0.1.0 (median of 3 same-GPU rounds; below 1: this package
is slower): 0.848 to 0.990 on sm_100a (13 rows), 0.882 to 0.970 on sm_103a (13
rows), 0.830 to 0.967 on sm_107a (13 rows).  This package's backward was slower
on every row.

## NoPE (zero-rope) program

`forward_nope` / `backward_nope`, `prepare_dsa_nope_train` and the NoPE
autograd entries take `q [T, 64, 512]` and the shared latent `kv [S, 512]`
(K = V), a default softmax scale of 1/16, and return one combined
natural-layout FP32 `dkv = dK + dV [S, 512]` -- fresh, or added into the
caller's `dkv_acc`.  The index routes (in-kernel length derivation below
`NOPE_CONVERT_SLOT_LIMIT` slots, the index-preparation stage at or above it,
the packed varlen conversion) are those of the non-deterministic package; the
backward reads the saved global rows and lengths on every route.  On the stage
route of `cake_dsa_sparse_attention_det_varlen` a caller `topk_length` that is
not a contiguous int32 `[T]` tensor on the rows' device raises `ValueError`
before either launch.

### Trainable per-head sink (NoPE)

The NoPE entries -- the flat `cake_dsa_sparse_attention_det_nope` and step pair
`cake_dsa_sparse_attention_det_train_nope_forward` / `_backward`, the packed
`cake_dsa_sparse_attention_det_nope_varlen` and step pair
`cake_dsa_sparse_attention_det_train_nope_varlen_forward` / `_backward` (and,
in this backend, `forward_nope`, `backward_nope`, `forward_nope_varlen` and
`prepare_dsa_nope_train`) -- take an optional `sink`: FP32 `[64]`, contiguous,
on the device of `q`.  The formulas, checks and returns are those of the
non-deterministic package (see its README, "Trainable per-head sink"):
`lse` includes `exp(sink[h])`, a row without a valid slot gives `out = 0` and
`lse = sink[h]`, and the gradient is
`dsink[h] = -sum_t exp(sink[h] - lse[t,h]) * delta[t,h]`.

* Forward: the stage `nope_fwd_sink` (`sm_100a` / `sm_103a`; on `sm_107a` its
  scanner-warp build `nope_fwd_scan_sink`) runs in place of `nope_fwd`: the
  non-deterministic package's sink build under this package's program name,
  one fixed-order kernel, so `out`, `lse` and `o_lo` repeat bit for bit.
* Backward: the stage `nope_bwd_dsink` (256 threads, every architecture of
  the record) runs once over all `T` rows right after `nope_bwd_delta`,
  outside the key-range passes and the redo.  It reads `lse`, `delta` and
  `sink` only -- none of them is scratch of a later stage -- so it is the same
  in every storage mode of the backward.  It splits the rows into fixed chunks
  of `NOPE_DSINK_CHUNK_TOKENS` (64) tokens, writes one FP32 partial per
  (chunk, head) to the workspace (`dsink_partial`, `ceil(T / 64) x 64` FP32),
  and the CTA that completes the grid (`dsink_ticket`: a 32-bit completion
  counter, zero-filled by the host before every backward) sums the partials in
  one fixed order (four lane-strided sums over the chunks, folded in lane
  order).  `dsink` therefore depends on `lse`, `delta`, `sink` and `T` only --
  not on the grid, the SM count, the key passes, `dkv_acc` or the redo -- and
  repeats bit for bit.  The other stages are unchanged: they read the
  sink-aware `lse`.
* State: a call with a sink saves one more tensor for its backward, the
  reference to the caller's `sink` (the eighth saved tensor), and a second
  backward through the same graph (`retain_graph=True`) replays the sink
  forward with it.  The workspace grows by `ceil(T / 64) * 256 + 4` bytes
  before alignment and `ceil(T / 64) * 256 + 256` after it
  (`dsa_nope_train_workspace_size(..., sink=True)`).  The
  binding keys end with the sink's geometry, so a sink call and a call
  without one never share a plan.
* A call without query rows (`T = 0`) launches nothing and gives a zero
  `dsink`.

## 96 and 128 query heads

The kernels compute 64 query heads of a token per work item.  A rope call over
`H` = 96 or 128 heads (`cake_backend.SUPPORTED_HEADS`) runs, on `sm_100a`,
`sm_103a` and `sm_107a` (`cake_backend.HEAD_COUNT_ARCHES`), the head-count
builds of the record (`cake_jit.HEAD_COUNT_STAGES`,
`cake_backend.head_count_stages`) in place of five stages; a 64-head call
launches what it launched before (same stages, grids and arguments).  Without
one of the builds for the device's architecture (or on another architecture)
such a call raises `NotImplementedError` before any launch (also without query
rows, on CUDA tensors); the NoPE program refuses 96 and 128 heads with
`ValueError` naming the rope form.

| 64-head stage | Head-count build | Launch |
|---|---|---|
| `fwd` | `fwd_h128` | the non-deterministic package's single-pass forward: one `(2, 1, 1)` cluster per token over every head, `num_queries = T`, `num_heads = H`, persistent grid `single_pass_grid(T, SMs) = 2 * max(1, min(T, SMs // 2))` |
| `bwd_delta` | `bwd_delta_heads` | `delta [T, H]` (row `t * H + h`, each row the 64-head build's bits) and `qstat [T, 4]` over the token's `H` heads; one CTA of 16 warps per token |
| `bwd_main` | `bwd_main_hw` | both 64-head windows of every token: `2 T` work items, item `i` = window `i & 1` of token `i >> 1`, `wsel = -1`, one CTA per item |
| `bwd_redo` | `bwd_redo_hw` | the redo over both windows of every flagged row's pairs, `redo_launch_grid(2 T, SMs)` |
| `bwd_main_pass` | `bwd_main_pass_hw` | a key-range pass over the `2 T` items, one FP32 dQ partial block (36,864 elements) per item |

The head-window builds read the caller's `[T, H, *]` tensors through tensor
maps whose head extent is `H`: the rows of `W1 = [64, 128)` past `H` load
zeros, store nothing and read `lse = +inf`, `delta = 0`, so they add exactly
0.  `token_base` / `token_step` count items: descending from `2 T - 1` where
`cake_backend.token_order` descends (one key pass on `sm_103a` and `sm_107a`),
else ascending from item 0.  A (token, key) pair adds one contribution per
window to its row, so the builds skip the overflow counter below `tau / 2`:
the skipped contributions add at most `N tau`, the slack of the redo
preparation's flag rule, and the counters certify the int32 range as at 64
heads.  The key norms, the row bound and its grid (over `T * topk` pair
slots: the pack / pack27 choice reads `T`), the redo preparation, the fold and
the compaction run over the `T` tokens unchanged.  The accumulator is an
integer sum, so dK / dV repeat bit for bit whatever the grid, the window
order or the key-pass count.

Storage scales with `H`: `o_lo` and dQ hold `H` KiB per query token (the
dead-storage rules of "Storage and the `o_lo` rule" read that capacity),
`delta` takes `4 H` B per token, and a key-range-pass backward keeps `2 T`
dQ partial blocks.  From 0.13.1 the policy plans one key pass above 64 heads on
every row (`plan_key_passes` with `items_per_token = HEAD_WINDOW_ITEMS`);
`key_passes` still selects the pass build `bwd_main_pass_hw`.  Before, the
record's rule planned two to four passes on the rows of one 2,176-token chunk
(two dQ partial blocks per token; top-k 2048) from 65,536 keys on `sm_100a` /
`sm_103a` and from 131,072 keys on `sm_107a`.  One pass was faster on every
such row timed (96 and 128 heads; backward, training step and checkpointed
recompute; medians of same-round ratios): B200 (`T = 2,048`, `S = 131,072`,
three passes) 6-12 %, GB300 (`T = 2,048` with `S = 65,536` / `131,072`,
`T = 1,024` with `S = 163,840`; two / three / four passes) 5-19 %, VR200
(`T = 1,024` to `2,176` with `S = 131,072` to `1,048,320`, two passes)
5-10 %; it also needs no pass scratch.  dQ takes the one-pass FP32 summation
order there; dK / dV keep their bits.
`cake_backend.dsa_train_workspace_size(..., num_heads=H)` sizes a prepared
runner.

## Layout of this package

* `cake_jit.py` -- the `MODULES` registry (one record per program, the rope
  `cake_dsa_h64_det_train` and the NoPE `cake_dsa_h64_det_nope_train`, each
  shared by every architecture it compiles for; per stage the `library` stem
  of its `build.json` entries, the FFI entry and the launch geometry that the
  export rewrites), the stage names of both programs and the loader glue
  (`library_name`, `load_cake_dsa_train_module`).  A stage entry may narrow
  the record's architectures with its own `arches`: `fwd_scan` /
  `nope_fwd_scan` / `nope_fwd_scan_sink` ship for `sm_107a`, `fwd` /
  `nope_fwd` / `nope_fwd_sink` for `sm_100a` and `sm_103a`.  `SCAN_STAGES` and
  `stage_for` resolve a forward on `sm_107a` to its scanner-warp stage, and
  `library_name` returns that stage's library.
* `cake_build.py` + `build.json` -- the loader shared by every Cake-exported
  package (byte-identical copies) and this package's build manifest (schema
  `cake-build/1`): one entry per `<library>_<arch>` with its translation units,
  digests, nvcc flags, closure digest and `aot/<name>.so` path.
  `cake_build.load_library(name)` loads the prebuilt library when `aot/` holds
  it and otherwise JIT-builds the entry through the package's JIT shim
  (`_cake_pplx_jit.py`).  `aot/` is never
  tracked (`.gitignore`).
* `cake_launch.py` -- generated positional launchers and grid functions over
  the kernels' own argument names (`STAGE_ARGS` / `LAUNCH` / `GRID` keyed by
  stage name; `GRID[stage](num_queries, num_kv, topk, num_dst)`).
* `cake_backend.py` -- validation, the storage plan of the backward, varlen
  index offsetting, the prepared runners, the autograd Functions and the eager
  entry points.
* `csrc/cake_ffi_checks.h` -- the TVM-FFI helper header the bindings include,
  shipped with the package so that it compiles from its own `csrc/` alone.
* `csrc/cake_dsa_h64_det_train/` -- generated kernel and binding translation
  units, one pair per library, compiled once per library and architecture
  (`.clang-format` disables formatting: the sources are identity-checked by
  the digests in `build.json`).  A library whose `sm_107a` binding differs
  from its other bindings (the NoPE backward's main kernel) also ships its own
  `sm_107a` binding unit.

The package imports only itself, `torch` and `tvm_ffi`; `cake_build.py`
imports the package's JIT shim (`_cake_pplx_jit.py`) only for the JIT
fallback.

## Status

Tests: `tests/experimental/test_cake_dsa_sparse_attention_det_train.py` (the
backend suite: imports only this package; skips without a registered program
or a compute capability 10.0 / 10.3 / 10.7 device),
`tests/experimental/test_cake_dsa_sparse_attention_det_train_api.py` (the
public entry points and the environment variables of the facade),
`tests/experimental/test_cake_dsa_sparse_attention_det_train_heads.py` (96 and
128 query heads: the host plans on the CPU, the naive tier and the bitwise
repeats on `sm_100a` / `sm_103a` / `sm_107a`) and the
CPU-only `tests/experimental/test_cake_dsa_sparse_attention_det_train_names.py`
and `tests/experimental/test_cake_dsa_sparse_attention_det_train_standalone.py`.
Accuracy is judged by `tests/cake_naive.py` through
`tests/experimental/test_helpers/cake_dsa_sparse_attention_naive.py` (the FP64
reference and the PyTorch BF16 form of the same call; `tests/conftest.py` prints
one `TIER` line per judged output); the run-to-run and one-feature checks of
this package are bitwise.  Benchmark: `benchmarks/bench_cake_dsa_sparse_attention_det_train.py`.
