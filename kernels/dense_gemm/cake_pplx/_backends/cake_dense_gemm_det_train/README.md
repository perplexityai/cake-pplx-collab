# Cake dense GEMM training backend, deterministic (SM100 / SM103 / SM107)

Backend of `cake_pplx.cake_dense_gemm_det_train`: the GLM-5.2 training GEMMs, the BF16 dense projection
GEMM (K1 -- BF16 operands, FP32 accumulation, BF16 or FP32 output, strided views, ragged token counts,
batched head projections) and the FP32 router-gate GEMM through split-BF16x3 tensor-core emulation with
chunked FP32 promotion (K2).

SM107 (Rubin R200) is the primary architecture; SM100 (B200 / GB200) is built
from the same sources and, for five input-gradient rows, ships the knob-free
eight-warp CTA form of the GEMM (one program, measured on GB200); SM103 (B300 /
GB300) ships its own builds of the same programs and, for six input-gradient
rows, the eight-warp CTA form in two programs; both also ship the start-path
eight-warp programs for BF16 input gradients of their 148- and 152-SM parts, on
128 x 256 and 128 x 128 pair tiles (`cake_jit.KERNELS` lists the registered architectures).  A device whose architecture the registry does not
list raises `NotImplementedError` naming the missing instance;
`is_cake_dense_gemm_det_train_supported(device)` reports whether a device is a supported architecture.  The
backend does not participate in automatic backend selection, autotuning or trace-apply.  Its libraries are
loaded prebuilt from `aot/` when the package ships them and JIT-built from `build.json`
otherwise (see "Layout of this package").

## Names and mode

The public module is `cake_pplx.cake_dense_gemm_det_train`.  It imports this package on the first
attribute access and returns the objects of `cake_backend.py` (and `ARCHES` of `cake_jit.py`) unchanged, so
`cake_pplx.cake_dense_gemm_det_train.X is cake_backend.X`.  Every public name carries the mode word `det`:

| kind | names |
|---|---|
| K1 entry points | `cake_dense_gemm_det_train_projection`, `cake_dense_gemm_det_train_projection_{forward,dgrad,wgrad}` |
| K1 plan / prepare | `cake_dense_gemm_det_train_projection_{plan,prepare,wgrad_prepare}` |
| K1 types | `CakeDenseGemmDetProjectionPlan`, `CakeDenseGemmDetProjectionLaunch` |
| K2 entry points | `cake_dense_gemm_det_train_router`, `cake_dense_gemm_det_train_router_{forward,dgrad,wgrad}` |
| K2 plan / prepare | `cake_dense_gemm_det_train_router_{plan,prepare}` |
| K2 types | `CakeDenseGemmDetRouterPlan`, `CakeDenseGemmDetRouterLaunch` |
| cache, support, constants | `cake_dense_gemm_det_train_convenience_cache_clear`, `is_cake_dense_gemm_det_train_supported`, `ARCHES`, `SUPPORTED_COMPUTE_CAPABILITIES` |

The module selects the mode: no entry point takes a mode argument.  Every program of this package runs in a
fixed order, so the same inputs, hardware and plan give bitwise the same outputs on every run.  The entry
points do not read `torch.use_deterministic_algorithms`: they never raise or warn under it (strict or
`warn_only`) and run the same programs with or without it.  The non-deterministic split-K programs are a
separate module, `cake_pplx.cake_dense_gemm_nondet_train`.  `CONVENIENCE_CACHE_SIZE` is a setting of
`cake_backend` (assign it there, not on the public module).  The backend's other helpers keep their names.

## Entry points (`cake_pplx.cake_dense_gemm_det_train`, defined in `cake_backend.py`)

```python
import cake_pplx.cake_dense_gemm_det_train as gemm

# K1: strided-view GEMM  out[l, m, n] = sum_k A[l, m, k] B[l, k, n]
gemm.cake_dense_gemm_det_train_projection(A, B, out, transposed_out=False)   # convenience path (cached plan)
gemm.cake_dense_gemm_det_train_projection_forward(X, W, out=None)               # out[T, N] = X[T, K] @ W[N, K].T
gemm.cake_dense_gemm_det_train_projection_dgrad(G, W, out=None, out_dtype=None)  # out[T, K] = G[T, N] @ W[N, K]
gemm.cake_dense_gemm_det_train_projection_wgrad(G, X, out=None, out_dtype=None)  # out[N, K] = G[T, N].T @ X[T, K]
prepared = gemm.cake_dense_gemm_det_train_projection_prepare(A, B, out, transposed_out=False)
prepared = gemm.cake_dense_gemm_det_train_projection_wgrad_prepare(G, X, out)
prepared.launch()   # allocation-free; prepared.template / .grid / .module_name / .plan

# K2: FP32 router GEMM through split-BF16x3 emulation
gemm.cake_dense_gemm_det_train_router(A, B, out, splits=None)                  # convenience path (cached plan)
gemm.cake_dense_gemm_det_train_router_forward(X, W, out=None, splits=None)       # out[T, N] = X[T, K] @ W[N, K].T
gemm.cake_dense_gemm_det_train_router_dgrad(G, W, out=None)                      # out[T, K] = G[T, N] @ W[N, K]
gemm.cake_dense_gemm_det_train_router_wgrad(G, X, out=None, splits=None)         # out[N, K] = G[T, N].T @ X[T, K]
prepared = gemm.cake_dense_gemm_det_train_router_prepare(A, B, out, splits=None)
prepared.launch()   # re-splits B, launches, reduces the split-K partials; .splits
```

### Host paths

Both paths run the same planner once per binding and bind the same generated
module; they differ in who owns the plan and the workspace.

| | prepared path (`*_prepare` + `launch()`) | convenience path (`cake_dense_gemm_det_train_{projection,router}` and their `_forward` / `_dgrad` / `_wgrad`) |
|---|---|---|
| planning | once, in `*_prepare` | once per cache key, on the first call |
| workspace (K1: stream-K slabs + slice counters; K2: BF16x3 stack, the torch-chain split's fp32 scratch, split-K partials) | owned by the prepared object | carved out of the workspace arena of the (device, stream): one growable buffer per kind shared by every cached entry of that device and stream, grown to the largest requirement seen, never per entry |
| per call | bind nothing: launch the bound views | rebind the caller's views (view objects only), resolve the workspace views of the arena (memoised until the arena grows), launch; a K1 entry point launches the binding of its key (the argument list bound once, the caller's views rebuilt from its tensors, the workspace converted once per arena growth); `out` is allocated only when the caller passes none |
| device queries | none after `*_prepare` | none after the first call |
| cache | none | per process, per kind (K1 / K2) and for the K1 entry points' bindings, `CONVENIENCE_CACHE_SIZE` = 64 entries each, least recently used first out; the arena of a (device, stream) is released with its last entry or binding and by `cake_dense_gemm_det_train_convenience_cache_clear()` |
| memory held | the prepared object's workspace | bounded by the largest geometry seen per workspace kind, however many token counts the cache has planned (a ragged T adds entries, not workspace) |
| CUDA Graphs | capture `launch()` | do not capture (a later growth replaces, an eviction frees, the workspace a graph replays) |

The orientation of `cake_dense_gemm_det_train_projection_wgrad` / `cake_dense_gemm_det_train_projection_wgrad_prepare` follows the
row's measured rule when the rule table keys the row in exactly one orientation
(the swapped `X.T @ G` form with the transposed store, or the direct `G.T @
X`), else the launcher's padded-tile-work rule (N < 256 and N = 576 swap);
`wgrad_views(G, X, arch=..., out_f32=...)` exposes the decision.  The generic
entry plans the same orientation for an unbatched call with both operands
MN-major and no transposed store (`A = G.T`, `B = X`: the weight gradient
through `cake_dense_gemm_det_train_projection`) when the rule table keys that
row in the swapped orientation alone (on `sm_107a` the bf16 6144 -> 576 row
from 6144 tokens on), so the call runs the weight-gradient entry's program
and bits.  On `sm_100a` and `sm_103a`
such a call without a rule in either orientation whose output has at most 64
columns against 4096 to 8192 rows and at least 256 tokens (the
`[d_in, d_out]` weight gradient `X.T @ G` of a narrow projection) is planned
swapped as well: its columns go to the 64-row tile of the transposed store
instead of one padded 224- / 256-column tile.  Every other call keeps its own
orientation.

The cache key is `(device index, raw stream, flags / knobs, geometry of A, B,
out)` where a view's geometry is its shape, strides, dtype, storage offset and
the 16-byte alignment residue of its first element -- never its address, so new
tensors of the same geometry hit (a training step with fresh activations) and
an entry never holds a caller's tensor.  Launches on different streams never
share a workspace (the stream is in the key and each (device, stream) has its
own arena); launches of one device and stream -- the only users of its arena --
are ordered by that stream, which is what makes one buffer per kind safe to
share: the K2 split rewrites the BF16x3 stack and its scratch in full before
every launch, the split-K partials are reduced into `out` right after the
launch, and growing a buffer replaces it while an earlier launch may still be
in flight on the old one -- safe because PyTorch's caching allocator is
stream-ordered (a freed block is handed out again only to a later allocation on
the stream it was freed on, or after a device-synchronising `cudaFree`).  The
stream-K slice counters are zero-initialised once and reused without a memset:
the kernel resets every counter it uses (the last arriver of a tail tile's slice
writes zero after the fixup), so the shared counter buffer is zero again when
each launch completes -- the same contract the Cake launcher's per-device
counters rely on.  Threads: planning and the arena bookkeeping (creation,
growth, reference counts, insertion, eviction, `cake_dense_gemm_det_train_convenience_cache_clear()`)
are serialised by a module lock (a miss re-checks the cache under it, so a key
is planned once; a growth keeps the larger buffer); the K2 hit path holds its
arena's lock from the carve of the workspace views through the split-K
reduction, because one router call is three stream operations (split, kernel,
reduction) on the shared BF16x3 stack and partials that two threads issuing on
one stream must not interleave; the K1 hit path (one self-contained kernel per
call) and the cache lookup take no lock.  The stream-ordering argument assumes
one issuer per stream at a time, as CUDA stream semantics do.  The device facts
a plan depends on (architecture tag, SM count, L2 size) are read once per
device index (`device_facts`).

### K1 contract

* `A` BF16 `[L, M, K]` (or `[M, K]`) with unit stride on `m` or `k`; `B`
  BF16 `[L, K, N]` (or `[K, N]`) with unit stride on `k` or `n`.  The other
  matrix stride and the batch stride are multiples of 8 elements (16 bytes);
  every view starts at a 16-byte aligned address.
* `out` BF16 or FP32 `[L, M, N]` / `[M, N]` with unit stride on `n` and
  16-byte aligned rows and batches, or -- with `transposed_out` -- `[L, N, M]`
  / `[N, M]` with unit stride on `m` (element `(m, n)` lands at `out[l, n,
  m]`).  The caller's view is written in place; nothing is copied or
  re-laid-out.  The row and batch strides the kernel receives (`ldo`, `out_l`,
  in elements; `rows` / `cols` = `M` / `N`, or `N` / `M` under `transposed_out`)
  satisfy `ldo >= cols`, `out_l >= cols` and one of two layouts: batch-major,
  `out_l >= rows * ldo` (each batch holds its matrix), or interleaved within a
  row, `ldo >= L * out_l` (a `[T, H, D]` projection written as `[H, T, D]`:
  `out_l = D`, `ldo = H * D`).  A batch of one passes its view's batch stride
  when that spans the matrix and the matrix span `rows * ldo` otherwise
  (`_out_strides`); any other layout is refused at launch.
* `M, K >= 1` and `N` a positive multiple of 8 are runtime values (no
  padding: TMA zero-fills out-of-bounds boxes, the epilogue masks rows `>= M`
  and columns `>= N`); `L` is arbitrary and the batch strides follow the
  layout rule above, so the token-major head views of the MLA projections
  (`[T, H, D]` permuted to `[H, T, D]`, `D` a slice of a wider slot) run
  without copies.
* The three training layouts map onto the operator: forward `X @ W.T`
  (`A` K-major, `B = W.T` K-major), input gradient `G @ W` (`B` MN-major),
  weight gradient `G.T @ X` (both MN-major, `K = T` ragged).  A weight
  gradient with fewer than 256 output rows runs the swapped `X.T @ G` with
  the transposed store (`cake_dense_gemm_det_train_projection_wgrad` / `cake_dense_gemm_det_train_projection_wgrad_prepare`).
* Numerics: BF16 x BF16 products exact in FP32, tensor-core FP32
  accumulation, one rounding at the store.  BF16 outputs match an FP64
  reference to `atol = rtol = 1e-2` elementwise; FP32 outputs carry the
  accumulation error (`rel_fro <= 5e-5`, `max_abs <= 1e-3 * max|ref|`).

### Host planning (mirror of the Cake launcher)

The host selects one traced kernel instance per (A layout, B layout, tile
width, tile height, output kind, epilogue path, raster group, TMA L2 eviction
hints); every
rule is a verbatim copy of the Cake kernel host's (each function names its
source line range).  Two device facts enter the plan: the SM count (CTA pairs:
stream-K split, raster group, wave working set) and the L2 size (hint gate);
`*_prepare` and the convenience path read both once per device index
(`device_facts`: `device_sm_count`, `device_l2_bytes` =
`torch.cuda.get_device_properties(...).L2_cache_size`, falling back to
`cuDeviceGetAttribute`), `cake_dense_gemm_det_train_projection_plan` takes them as inputs:

| choice | rule |
|---|---|
| operand layout | `operand_view`: the contraction axis contiguous -> K-major, else the M/N axis contiguous -> MN-major |
| `BLOCK_N` | `default_block_n`: 128 when `N <= 128`, 192 when `N` is a multiple of 192 but not of 256 (`N = 576`: three exact tiles), else 256.  The menu is `BLOCK_N_CHOICES = (128, 160, 192, 224, 256)`: 160 / 224 (Cake round 9) are wave-quantization fits a rule or caller selects; their 80 / 112-column warp slices take the register epilogue |
| `CTA_ROWS` | `default_cta_rows`: 128.  The 256-row tall-tile family (`cta_rows=256`, symbol `_m256`) and the 64-row Layout-B family (`cta_rows=64`, Cake round 9: 128 x `BLOCK_N` pair tiles through the M=128 `cta_group::2` MMA, four TMEM buffers; symbol `_m64`) are rule or caller selections (`CTA_ROWS_CHOICES = (64, 128, 256)`).  A caller-forced `cta_rows` outside the row's rule (or default) family drops the rule's `stages` / `slots` / `epi`; its other knobs still apply |
| epilogue | `epi_mode`: a warp slice is `epi_cols = BLOCK_N / 2` columns (`BLOCK_N / 4` on 64-row tiles; always a multiple of 16).  Transposed output -> register stores (`reg`); row-major fp32 -> TMA stores (`tma`) when the slice is whole 32-column chunks (`epi="reg"` is an opt-in float4 register path; 160 / 224-wide and 64-row 192-wide slices have no TMA path); row-major bf16 -> `tma` when `K <= 1024` and the slice is whole 64-column chunks (`BLOCK_N = 256` at any height, 128 at 128 / 256 rows), else `reg`; `quad_store` (rule-selected, `_q`) is the bf16 `reg` variant for whole-64-column slices that transposes 32-byte row segments across lane quads so each lane issues one 256-bit store per 4-row group |
| staging slots | `epi_slots`: 1 with the TMA-store epilogue (2 only when `K <= K_TWO_SLOTS = 0` and the slice's chunk count is even), 0 otherwise |
| stages | `default_stages(slots, cta_rows, block_n, b_mn)`: 7 / 6 / 5 for 0 / 1 / 2 slots at `BLOCK_N = 160 / 192 / 224 / 256`, 9 / 8 / 6 at `BLOCK_N = 128`, 4 / 4 / 3 for tall (256-row) tiles; 64-row tiles take the deepest pipeline that fits the 227 KiB opt-in beside the staging (32 KiB per slot), at most 12 (9 / 8 / 6 at `BLOCK_N = 256`, 12 / 12 / 10 at 128; `b_mn` sizes the MN-major B panels); the single-CTA form (`cta1`, round 19: its B stage streams the whole `BLOCK_N` columns per CTA, `b_stage_bytes(..., cta_group=1)`, so 48 KiB stages at `BLOCK_N = 256` and 32 KiB at 128) takes the deepest fit the same way (4 / 4 / 3 and 7 / 6 / 5).  A rule or caller `stages` is bounded by `smem_limit_for(arch)`: the 232448 B opt-in, or 334336 B on `sm_107a` (CUDA 13.4 oversized shared-memory mode, `SMEM_OVERSIZED`); the limit is not part of the instance key, so an instance has one symbol on every architecture |
| raster group | `default_group_m(a_mn, b_mn, m_tiles, pair_tiles, pairs)`: 16 CTA row tiles per cluster-launch-control raster group (a launch parameter since round 11, not an instance-key field) (even, so the two CTAs of a pair are the row halves of one 256-row tile) on every row; a 4-row group for one-to-two-wave weight-gradient rows measured no gain in the production configuration and is not applied.  A non-default group would carry the symbol suffix `_g<n>` |
| L2 hints | `default_hints(a_mn, b_mn, m_tiles, n_tiles, K, group_m, pairs, l2_bytes)`: no hint while one wave's operand panels fit the L2 (`wave_working_set(m_tiles, n_tiles, K, group_m, pairs) <= l2_bytes`); otherwise `A` streams `evict_first` when there is one column tile (`n_tiles == 1`); every other row gets no hint (reuse-ratio hints on the weight-gradient class measured 0..-4 percent in the production stream-K configuration).  Symbol suffix `_h<a><b>` (first letters): `_hen` = A evict_first, B none |
| L2 promotion, prefetch | `default_promo` = `none`, `default_pf` = 0 (measured, off); the symbol carries `_<promo>` / `_pf<n>` when set |
| CTA form | `form`: `""` (the ten-warp pipeline of every other program) or one of `W8_FORMS = ("w8r", "w8rs", "w8", "w8rel", "w8relf")`, the eight-warp CTA form of the same GEMM measured for six `sm_103a` input-gradient rows (`w8r` / `w8rs`) and five `sm_100a` input-gradient rows (`w8`, the knob-free program): four epilogue warps, a loader, a scheduler (`w8r`; `w8` = the same without the early accumulator release) or the loader's elected lane issuing the cluster-launch-control queries in its place (`w8rs`) and the MMA warp on CTA-pair 256 x 256 tiles over six 32 KiB stages, whole tiles only, the BF16 output through the TMA store, the raster quotients resolved by four fast divisors the host passes (`fd_tiles_l` / `fd_tiles_g` / `fd_group` / `fd_last`: launch arguments of these programs alone).  `w8rel` is the `w8r` program with the start path (`sm_100a` and `sm_103a`): the loader prefetches the first item's first K-step boxes into L2 in the prologue, the landing barriers work at cluster scope, each role waits on the cluster rendezvous just before it needs the peer CTA and the MMA warp allocates the TMEM itself.  `w8relf` is the same with the epilogue freeing the TMEM under the last item's store drain.  Both keep the bits of `w8r` and take no batch raster and no store hint; the token ranges of version 0.3.12 name them (part rules).  `w8relf` also runs the 128 x 128 pair tile (`block_n` 128: each CTA one 64-column B panel per K step in the same six 32 KiB stages) and, there, 32-element K steps (`block_k` 32: a CTA's A box of 128 rows by 32 columns in the 64-byte swizzle, two K-16 MMAs per stage; `instance_key` field 40, present only when the K step is not 64, so every other key keeps its 39 fields); every output element still sums its K in one FP32 accumulator in K order, the bits of the 256-column program, and the token ranges of version 0.3.14 name these programs (part rules).  Symbols `_tma1_w8r` / `_tma1_w8rs` / `_tma1_w8` / `_tma1_w8rel` / `_tma1_w8relf` (the form token is the last knob suffix; `instance_key` field 37).  A rule's form yields where the plan is not whole-tile or a caller forces a pipeline, tile, output, epilogue, tail or raster knob (the row's ten-warp program runs the plan asked for; the rule's `pd` / `pro` / `lw` / `mg` / `sw` are its program there, and the `sm_103a` 4096 -> 48 rule carries none, so the architecture default one-slot program runs there; forcing `epi` `tma`, two slots, `htail`, `lw` and `pro` 15 plans its previous half-tail program); where the form stands, those knobs are cleared from the key.  A caller-forced form is refused there (`instance_key`; the planner for stream-K units) |
| K walk | `kser`: 0 (every item walks its K blocks ascending) or P, the serpentine K walk: a whole-K item whose item index lies in an odd block of P items walks its K blocks descending, so a wave of P CTA pairs starts on the K end the previous wave left in L2 (symbol `_ks<P>` before `_pro<n>`; another FP32 summation order, fixed run to run).  The `sm_103a` T-exact rules of the 12288 -> 4096 forward (T 16172 / 16384) and the 4096 -> 12288 input gradient (T 16172 / 16231 / 16384) carry `kser` 76; it is a rule knob only (not a planner keyword); the planner drops it where the program cannot carry it and on parts whose CTA pairs are not 76 |
| promoted chunks | package-only: an `nn` BF16-output plan (the weight gradients) whose longest segment (`segment_blocks`: a whole tile, a stream-K unit's part of a tile, an exact or synchronised split part) holds more than `K1_PROMOTE_MIN_SEGMENT` (128) K blocks (8192 tokens in one accumulator, one chunk of `K1_PROMOTE_BLOCKS`) runs the promoted twin of its program (`_kc128` before `_pro<n>`, `K1_PROMOTE_BLOCKS` = 128) where the architecture registers that twin and the twin's stream-K plan is valid, except on the rows of `K1_PROMOTE_EXEMPT_ROWS`: the same plan, each segment summed in chunks of at most 128 K blocks added in a fixed order (the same bits on every run).  Up to version 0.3.5 the threshold was 256 K blocks; some BF16 weight gradients whose longest segment held 129 to 256 K blocks without promoted chunks missed the naive BF16 tier (see `K1_PROMOTE_MIN_SEGMENT`).  Ten programs ship a twin; FP32 outputs and every other program (the 256-row tall tiles, the 160- and 224-wide programs among them) never promote.  For a twin rendered from the same kernel source as its program, one chunk is expected to give the program's bits; the GPU tests check this for the two 64-row programs (`dense_proj_det_gemm_nn_n128_m64_hen_t_pro15`, `dense_proj_det_gemm_nn_n256_m64_hen_t_pro15`) only.  `promote` is a keyword of `cake_dense_gemm_det_train_projection_plan` and `cake_dense_gemm_det_train_projection_prepare`: `promote=False` keeps the program, `promote=True` runs the twin at any length (and raises where the program has no twin or the twin's stream-K plan is invalid); the weight-gradient entries take the planner's choice |
| template | `instance_symbol(instance_key(...))`, e.g. `dense_proj_det_gemm_kk_n256`, `dense_proj_det_gemm_kk_n128_hen`, `dense_proj_det_gemm_kn_n256_f32_tma1`, `dense_proj_det_gemm_nn_n256_f32_tma1`, `dense_proj_det_gemm_nn_n128_hen_t`, `dense_proj_det_gemm_kk_n256_m64`, `dense_proj_det_gemm_kk_n256_m256_ov_ht`, `dense_proj_det_gemm_nn_n160_m256_bz64`, `dense_proj_det_gemm_nn_n128_hen_t_skx`, `dense_proj_det_gemm_kn_n256_tma1_bg8`, `dense_proj_det_gemm_kn_n256_f32_v8_ef_s9`, `dense_proj_det_gemm_nn_n256_m256_sks_sb2`, `dense_proj_det_gemm_kk_n128_hen_c1`, `dense_proj_det_gemm_kn_n256_tma1_w8r`, `dense_proj_det_gemm_kn_n256_tma1_w8rs`, `dense_proj_det_gemm_kn_n256_tma1_w8`, `dense_proj_det_gemm_kn_n256_tma1_w8rel`, `dense_proj_det_gemm_kn_n256_tma1_w8relf`, `dense_proj_det_gemm_kn_n128_tma1_w8relf`, `dense_proj_det_gemm_kn_n128_k32_tma1_w8relf` (round 13: `_pk` parked / `_ov` overlapped single-TMEM-buffer / `_ht` half-height-tail-wave tall epilogues; `htail` replaces the stream-K tail plan by 2 x tail whole-K half items when they fit the CTA pairs, so `ws_f32_elems = 0` and `sk_iters = sk_units * k_blocks`; `_bz64` = 32-column MN-major B panels (`b_swz` 64, smaller stages, deeper default pipeline), `_skx` = exact p-way K split of the tail tiles (`sk_exact`, pre-empts `sk_parts` / `htail`, p travels in `sk_iters`); `_sks` = synchronised stream-K tail (round 18: `sk_sync`, the tail tiles' main units over K steps [0, s) plus collector units over the leftovers, one fp32 slab per tail tile and a deterministic two-addend fixup; `sk_iters` = the main-unit count, `iters_per_unit` = s = ceil((q K + `sk_sync_m`) / (q + 1)); yields to `sk_exact`, pre-empts `sk_parts` / `htail`); `_bg8` = eight adjacent batch entries (heads) interleaved across the CTA pairs (`batch_group`, batched MLA rows only, 0 for a batch of one); `_ef` = fp32 v8 register stores with `L1::no_allocate.L2::evict_first` (`store_ef`), `_s9` = nine pipeline stages (oversized dynamic shared memory on sm_107a); round 19: `_sb<n>` = the slab path of the synchronised fixup (`sk_slab`: 1 early arrival, 2 + the bulk slab read through the dead mainloop stages, 3 + the fp32 output as the slab - the same `_sks` plan, forced to 0 without `sk_sync`; a rule's 3 yields to 2 when a caller forces the row-major fp32 register epilogue, a rule's 2 to 0 when the eight warp slabs do not fit a caller-forced pipeline), `_c1` = the single-CTA form (`cta1`: one `CTA_ROWS` x `BLOCK_N` tile per CTA with cluster dims (1, 1, 1), so `m_tiles` is not rounded to pairs, `pair_tiles` counts CTA tiles, `sm_pairs` is the SM count (`sm_units`), the grid is `num_full` CTAs and the form carries no tail policy; 128-row family only); `_skf` directly after `_skx` = the tail-first order of the exact split (`sk_first`: pair work items `[0, sk_units)` are the K parts of the tail tiles, then the whole tiles; the same plan, counters and slabs as `_skx`), `_lw` between `_or` and `_pro` = the load warp leaving the pair prelude before the post-allocation CTA rendezvous (`lw`; pair programs whose MMA warp allocates TMEM, `pro` bit 4), `_mg` directly after `_lw` = the whole-tile tile resolve (`mg`: the load warp and the MMA warp walk whole tiles, one segment per item, with the raster quotients resolved through corrected fp32 reciprocals; whole tiles only - no batch raster, column-grouped raster, half-height tail or stream-K form, and the plan carries no stream-K unit); `_sw<n>` directly after `_mg` = the paced output stores (`sw`: each epilogue warp of the bf16 TMA-store pd-2 drain delays its bulk tensor store issue, after the staging write and its fences, by a per-warp number of steps of `n` dependent u32 multiply-adds seeded by a clock sample, so each per-item eight-store burst becomes a seven-step ramp; the stores and bytes are unchanged; needs `slots` > 0, `pd` 2 and a bf16 output; 0 = off); `_w8r` / `_w8rs` / `_w8` / `_w8rel` / `_w8relf` directly after `_pro<n>` = the eight-warp CTA form (`form`, the CTA form row above; `_w8rel` = `_w8r` with the start path, `_w8relf` = the same with the epilogue's TMEM free), `_k32` directly after `_n128` = the 32-element K steps of the 128-column `w8relf` program (`block_k`, `instance_key` field 40; absent at 64)) |
| rules | `ROW_RULES[(arch, A MN-major, B MN-major, fp32 out, transposed out, batched, N, K, M)]` (a ragged rule leaves the token count out: `M = None` for forward / input gradients, `K = None` for weight gradients; an exact key names one token count, as the `sm_107a` T-exact weight-gradient rules do at T 16231 and 16384, each in the orientation the row plans without it, and the GB T-exact forward / input-gradient rules of `sm_100a` and `sm_103a` do at `M` = T: 16 keys, each with the per-tile K partitions of the row's plan without it, so every output is bitwise equal to that plan's - the 4096 -> 48 input gradient on `sm_100a` (`_tma1_w8r`) and the 4096 -> 512 forward on both architectures and the 5120 -> 512 forward on `sm_103a` (`_ht_cg_pro15`) at T 16231 / 16384, the 6144 x 2048 input gradient (q_a / shared_gate_up) and the 2048 -> 6144 shared_down forward (`_ht_cg_pro15`) at T 16172 / 16231 on both architectures; and the GB T-exact rules of round 2: 10 keys with the same K partition property - the q_b 2048 x 16384 input gradient (`_ov_ht_cg`, the ragged rule's program with the column-grouped raster: 8-column-tile groups at T 16172 and 16-column-tile groups at T 16231 on `sm_100a`, 16-column-tile groups at both T on `sm_103a`), the 17408 -> 5120 weight gradient at T 16231 / 16384 on both architectures and the 5120 -> 17408 weight gradient at T 16231 / 16384 on `sm_103a` (`_ht_tma1`, 12-row raster groups, keyed at `K` = T in the direct orientation the row plans without them); and the GB T-exact rules of round 3: 4 keys with the same K partition property - the indexer_q 2048 -> 4096 weight gradient at T 16172 / 16231 on both architectures (`_tma1_sks_sb2`, the ragged rule's synchronised two-partial stream-K plan with its bf16 tiles through the one-slot TMA store, keyed at `K` = T in the direct orientation); and the GB T-exact rules of round 4: 4 keys with the same K partition property - the 5120 -> 17408 input gradient at T 16231 / 16384 on both architectures (`_ov_ht_tma1_cg`, the overlapped half-tail program with the column-grouped raster in 12-column-tile groups, in place of the ragged rule's row-half program), each with its bf16 tiles through the one-slot TMA store; and the GB T-exact rules of round 5: 2 keys with the same K partition property - the 12288 -> 4096 forward at T 16231 / 16384 on `sm_100a` (`_ov_ht_tma1_cg`, the ragged rule's overlapped half-tail program with the column-grouped raster in 8-column-tile groups), with its bf16 tiles through the one-slot TMA store; and the GB T-exact rules of round 6: 4 keys with the same K partition property - the 17408 -> 5120 forward at T 16231 / 16384 on both architectures (`_ov_ht_cg`, the ragged rule's own overlapped half-tail program with the column-grouped raster, in 12-column-tile groups in place of the ragged rule's 8); and the GB T-exact rules of round 7: 2 keys with the same K partition property - the 5120 -> 17408 weight gradient at T 16231 / 16384 on `sm_100a` (`_ht_tma1`, 12-row raster groups, the rule of the `sm_103a` keys of the row, keyed at `K` = T in the direct orientation the row plans without them); and the GB T-exact rules of round 8: 2 keys with the same K partition property - the 16384 -> 6144 forward at T 16231 on both architectures (`_ht_cg_or`, the half-tail program with the tall epilogue released by row half and the column-grouped raster in 8-column-tile groups, in place of the ragged rule's overlapped half-tail program `_ov_ht`), and the round-6 keys at T 16231 / 16384 on `sm_100a` re-valued to `_ht_cg_or` (the same 12-column-tile raster groups, the tall epilogue released by row half in place of the column-chunk one); and the GB T-exact rules of round 9 on `sm_103a`: the round-6 keys at T 16231 / 16384 re-valued to `_ht_cg_or` (the `sm_100a` keys' rule: the same 12-column-tile raster groups, the tall epilogue released by row half in place of the column-chunk one), and 2 keys with the same K partition property - the 12288 -> 4096 forward at T 16231 / 16384 (`_ht_cg_or` in 8-column-tile raster groups, in place of the ragged rule's overlapped half-tail program `_ov_ht_cg` in 16-column-tile groups); and the GB T-exact rules of round 10: 3 keys with the same K partition property - the 4096 -> 12288 input gradient at T 16384 on `sm_100a` and at T 16231 / 16384 on `sm_103a` (`_ht_cg_or`, the half-tail program with the tall epilogue released by row half and the column-grouped raster in 16-column-tile groups, in place of the ragged rule's overlapped half-tail program `_ov_ht_cg` in the same groups); the ragged rules keep every other T) lays per-row knobs over the defaults above, merged over `ARCH_RULE_DEFAULTS` (`pro` 7 on `sm_100a`, `sm_103a` and `sm_107a`).  Rounds 2-33 mirror the Cake table byte for byte; the shapes tranche (`sm_100a` and `sm_103a`: forward / input gradient / weight gradient rows at K 4096-17408, N 32-17408, measured at T 16172-16384 on GB200 / GB300 against cuBLAS) and its kernel-level round 2 (the half-tail column-grouped 4096 squares, 12-column raster groups on the N 5120 rows, the load-warp thin input gradients with the `l2_256b` promotion of the N 512 rows on `sm_103a`, the tail-first 16-way exact split of the 4096 x 12288 weight gradients) extend it per architecture.  A rule's `sk_first` / `lw` yield where the planned program cannot carry them (an exact split the arithmetic does not admit, the single-CTA form, a `pro` without bit 4, a caller-forced tile family).  Three `sm_103a` input-gradient rules (4096 -> 512, 5120 -> 512, the kv_a 6144 -> 576) carry `mg` with `sk: False` (the whole-tile tile resolve measured faster than the row's program above the pair spread at both T on GB300); a rule's `mg` yields where the plan carries stream-K units or exceeds the kernel's cluster-tile bound (a caller `sk` / `sk_max_units` / `sk_parts` / exact or synchronised split), where a caller forces a raster or tail form the `_mg` program excludes (`batch_group` / `cgrp` / `htail` / `sk_first` / `stt`), the single-CTA form or another tile family, and a caller-forced `mg` is refused on such a plan (`sk_units` is 0 on every `_mg` program).  Two `sm_103a` input-gradient rules (5120 -> 512, 5120 -> 1024) carry `sw` 25 (the paced output stores measured faster than the row's program above the pair spread at both T on GB300); a rule's `sw` yields where the planned program is not the bf16 TMA-store pd-2 drain (a caller-forced register epilogue or `slots` 0, another `pd`, the fp32 output) or the single-CTA form, and a caller-forced `sw` is refused there.  Six `sm_103a` input-gradient rules (4096 -> 512, 4096 -> 1024 and the kv_a 6144 -> 576 with `w8rs`; 4096 -> 48, 5120 -> 512 and 5120 -> 1024 with `w8r`) carry `form`: the eight-warp CTA form measured faster than the row's program above the pair spread at both T on GB300 (the `sm_103a` rules govern B300 as well, where no measurement exists), with raster groups of 8 row tiles on 4096 -> 1024, 12 on 5120 -> 512, 5120 -> 1024 and kv_a, and 16 on the other two; the knobs after it stay the row's program where the form yields (4096 -> 48: the architecture default program).  Five `sm_100a` input-gradient rules (4096 -> 512, 4096 -> 1024, 5120 -> 512, 5120 -> 1024 and the kv_a 6144 -> 576) carry `form` `w8`: the knob-free eight-warp program measured faster than the row's program above the pair spread at both T on GB200 (the `sm_100a` rules govern B200 as well, where no measurement exists), with raster groups of 12 row tiles on 4096 -> 512 and 12 on kv_a and 8 on the other three rows; the knobs after it stay the row's program where the form yields.  Eleven ragged `sm_107a` rules of the 212-SM tranche (measured on R200, 212 SMs, at T 16172-16384 against the plan the row had without them, both in the cold-L2 frame and under sustained back-to-back load) plan the half-tail program with the column-grouped raster (`_ht_cg_pro15`) on the 12288 -> 4096, 4096 -> 12288, 4096 -> 4096, q_b and shared_down input gradients and the 4096 -> 12288 forward, the column-grouped raster in 12-column-tile groups (`_cg_pro15`) on the 5120 -> 5120 input gradient (all of these bitwise equal to the plan without them), three K parts per tail tile (`sk_parts`) on the 12288 -> 4096 and 4096 -> 12288 weight gradients and the synchronised stream-K slab path (`_sks_sb2`, FP32 through `_f32_v8_ef`) on the 5120 -> 5120 BF16 and FP32 weight gradients; eight of them apply from a token floor on (`ROW_RULE_MIN_T`: below it the row keeps the rule it planned with before the tranche, and the FP32 5120 -> 5120 weight gradient plans with no rule up to T 257), set from the R200 A/B of every plan the tranche changed.  The `sm_100a` 5120 -> 64 / 128 BF16 weight-gradient rules (the swapped orientation) are the `sm_103a` rules of the rows from T 32768 on: the synchronised split of the 20-tile tail with margin 16, whose segments run the promoted twin (`dense_proj_det_gemm_nn_n128_hen_t_sks_kc128_pro7`; the main parts hold 264 K blocks at T 32768), measured faster than the 64-row program on B200 and GB200 at T 32768 (1.04 to 1.06 times), 65535 and 131072, within the naive BF16 tier and with the same bits run to run; their `ROW_RULE_MIN_T` tier keeps the 64-row program below T 32768 (its promoted twin where the longest segment holds more than `K1_PROMOTE_MIN_SEGMENT` K blocks).  The split was 1 to 3 % slower at T 16231; under the promotion threshold of versions up to 0.3.5 (256 K blocks) the same split with margin 0 (parts of 256 K blocks, no promoted chunks) missed the naive tier on 5120 -> 64 at T 32768, and from T 31744 down the rule's own parts held at most 256 K blocks, without promoted chunks.  Version 0.3.7 adds `ROW_RULE_MIN_T` tiers to the BF16 forward (both operands K-major) and input-gradient rules with K > 8192 of `sm_100a` (the 12288 -> 4096, 17408 -> 5120, o_proj 16384 -> 6144 and dense_down 12288 -> 6144 forwards, the q_b 2048 -> 16384, 4096 -> 12288, 5120 -> 17408 and dense_gate_up 6144 -> 12288 input gradients) and to the four such rules of `sm_103a` (it reads the other four through `RULE_PARENT`): at small token counts these rules (256-row tiles with the half-height tail wave) sum a tile's whole K, 192 to 272 K blocks, in one FP32 accumulator, and missed the naive BF16 tier on 49 of the 840 cells at T 1 - 2049 on B200, 47 of 840 on B300 and 35 of 840 on GB300 (8 rows x 21 token counts x 5 seeds of 32 draws; every miss at T 1 - 513, errors up to 1.0089 times the tier's; later passes found no other miss).  These rules met the tier on every judged cell of the 12288 -> 4096 and dense_down forwards and of the 4096 -> 12288 input gradient, and on the other rows above the token counts where they missed: there the tiers are taken for speed alone (another summation order).  Below T 257 a tier splits the tail's K steps into three units per tail tile (the q_b input gradient: four below T 513; the o_proj forward: no split tier; with K 17408 the units, of 91 K blocks, are laid in one line over the tail, so a tile sums three or four partials), then up to the key's floor (T 769, 1025 or 2049: one above the highest measured token count of the key's band; at the next measured one the tier plan was no longer faster on every measured part) it plans the 128-row program without the row's rule, with the rule's column-grouped raster on the 12288 -> 4096 and 17408 -> 5120 forwards (`dense_proj_det_gemm_kk_n256_pro7`, `dense_proj_det_gemm_kk_n256_cg_pro7`, `dense_proj_det_gemm_kn_n256_pro7`): segments of at most 128 K blocks wherever the plan before missed, every tier plan within the naive tier on every judged seed and 1.12 to 2.53 times faster than the row's rule at the measured token counts on B200, B300 and GB300, which set the bounds, and 1.04 to 2.34 times on GB200, measured afterwards (160-SM `sm_103a` parts are not measured).  A rule's `htail` names the program (`_ht`); the half-tail fit (`0 < 2 * tail <= pairs`) selects only the plan (`plan.htail_plan`): where the half items do not fit the device's pairs (the K 5120 thin input gradients: 1280 tiles on 76 pairs, tail 64; the 4096 x 12288 projections: tail 56) the `_ht` program runs the row's plain whole-tile plan (never linearised stream-K units, whatever `sk` the row or the caller asked for), as measured.  `sm_103a` registers every `sm_100a` program and its lanes measured the shapes tranche alone: a row without an `sm_103a` rule plans with the `sm_100a` rule (`RULE_PARENT`; the same knobs, so the same program, kept where the `sm_103a` registry has it), a rule `sm_103a` measured itself always wins, and the weight-gradient orientation follows the same lookup; `sm_100a` and `sm_107a` inherit nothing |
| exact-wave gate | a rule's `BLOCK_N` 224 (`EXACT_WAVE_BLOCK_N`) was measured for the whole-wave fit of its tiles on the 76-pair devices (`exact_waves`: 152 tall tiles of the 4096 x 4096 weight gradient = two waves, 76 tiles of 4096 -> 1024 = one); on a device whose pairs do not fit them in whole waves and where the default tile needs fewer waves (74 pairs of a 148-SM part) the rule's `BLOCK_N` yields to the default tile (`plan.wave_fallback`; the rule's other knobs stay, and its `wave_yield` knobs join them: the 4096 x 4096 BF16 weight-gradient rules of `sm_100a` and `sm_103a` plan the synchronised split of the 54-tile tail with slab path 2 and margin 32 there (`dense_proj_det_gemm_nn_n256_m256_sks_sb2_pro7`) from T 8192 on, measured on B200 and B300 at T 8192 - 32768; below T 8192 their `ROW_RULE_MIN_T` tier plans the rule without `wave_yield`, so the whole-tile plan stays); a caller-forced `block_n` is never touched, and the plan is the launcher's where the fit is exact or both tiles need the same waves (152 and 160 SMs).  The two `sm_107a` T-exact rules with `BLOCK_N` 224 (the 5120 x 512 weight gradient at T 16231 and 16384) were measured on the 216-SM part instead; the same gate applies to them. |
| exact-split fit | package-only: a rule's `sk_exact_fit` names the CTA pair count on which its exact split, where admitted and its `parts * tail` units exceed the pairs, takes `max(2, pairs // tail)` parts instead (one wave of units; the rule's `sk_first`, raster and other knobs stay; a caller-forced `sk_exact` keeps its parts; the rule's split stays where the fitted one is not admitted).  The 12288 -> 4096 and 4096 -> 12288 BF16 weight-gradient rules of `sm_100a` and `sm_103a` (and the `sm_100a` 4096 -> 12288 key at T 16231) name 74 pairs: on a 148-SM part their 16-way split of the 14-tile tail (224 units) becomes a 5-way split (70 units, the same `dense_proj_det_gemm_nn_n256_m256_skx_skf_pro7` program), measured faster on B200 and B300 at T 8192 - 32768; on 152 and 160 SMs the 16-way split stays |
| part rules | package-only: `ROW_RULES_BY_SM[(arch, sm_count)]` holds exact `ROW_RULES` keys that apply on a part of exactly that SM count; such a rule replaces the row's rule there alone (the architecture's other parts, `RULE_PARENT` children and other token counts keep `ROW_RULES`).  On a 148-SM `sm_100a` part the FP32 input gradients 4096 -> 32 / 48 / 64 / 128 and 5120 -> 32 / 48 / 64 at T 16231 / 16384, 5120 -> 128 at T 16231 and 6144 -> 128 at T 16172 / 16231 run the single-CTA program with 128-column tiles, two TMA-store slots and four stages (`dense_proj_det_gemm_kn_n128_f32_tma2_s4_c1`) over the raster group of the row's rule: measured 1.02 to 1.06 times faster than the row's program on B200, with the same output bits.  `ROW_RULE_RANGES_BY_SM[(arch, sm_count)]` holds token ranges of ragged `ROW_RULES` keys on a part of exactly that SM count, `{key: ((lo, hi, rule), ...)}` (inclusive, sorted and disjoint; the token count is `K` of a weight-gradient key, `M` of a forward / input-gradient key): at `lo <= T <= hi` the range's rule replaces the row's rule on that part, as an exact key at that T would (read after the part's exact keys; `ARCH_RULE_DEFAULTS` still merge under it; the weight-gradient orientation reads `ROW_RULES` alone, so these rows keep the direct orientation); the architecture's other parts, `RULE_PARENT` children and other token counts keep `ROW_RULES` and its `ROW_RULE_MIN_T` tiers.  On a 212-SM `sm_107a` part eight FP32 rows have ranges, measured on two VR200 nodes against the plan the row had before (eager launches and CUDA graph replays, faster on both nodes at each end of every range, within the naive BF16 tier, the same bits run to run): the 12288 -> 4096 and 4096 -> 12288 (T 4096 - 16384; 4096 -> 12288: T 8192 - 16231 up to version 0.3.7) weight gradients run the synchronised split of the 26-tile tail with margin 38 and slab path 2 through the FP32 v8 evict-first register epilogue (`dense_proj_det_gemm_nn_n256_f32_v8_ef_sks_sb2_pro7`), 1.05 to 1.09 times faster in the measurement of version 0.3.5, another summation order than the plan before; the 17408 -> 5120, 5120 -> 17408, 6144 -> 12288 and 12288 -> 6144 (T 4096 - 8192; 5120 -> 17408 and 12288 -> 6144: T 8192 up to version 0.3.7) weight gradients run whole 128-row tiles through the FP32 v8 register epilogue (`dense_proj_det_gemm_nn_n256_f32_v8_pro15`), 1.02 to 1.17 times faster in the measurement of version 0.3.5, with the same output bits; the 16384 -> 6144 weight gradient at T 4096 - 8192 (T 4096 and 8192 alone up to version 0.3.7) runs that program with a two-way K split of its 52 tail tiles (at T 4096 and 8192 104 units, each the whole K half of a tile), 1.06 to 1.14 times faster at T 4096 and 8192 in the measurement of version 0.3.5, another summation order; the 16384 -> 6144 input gradient at T 8192 alone runs the half-height tail wave of 128-row tiles (`dense_proj_det_gemm_kn_n256_f32_v8_ht_pro7`, 68 half items of the 34-tile tail), 1.02 to 1.06 times faster, with the same output bits.  Version 0.3.7 adds BF16 ranges for four K-major rows of the 212-SM part, measured on VR200 (5 seeds x 32 draws against the naive BF16 tier: the plan before missed it on 4 of the 560 cells at T 1 - 3072, the o_proj forward at T 16 and 33 and the 4096 -> 12288 input gradient at T 1 and 64, errors up to 1.0058 times the tier's): the o_proj 16384 -> 6144 forward runs a four-way K split of its tail tiles at T 1 - 256 (`dense_proj_det_gemm_kk_n256_pro15`, 1.81 to 2.25 times faster) and the 128-row program with the row's column-grouped raster at T 257 - 1024 (`dense_proj_det_gemm_kk_n256_cg_pro15`, 1.17 to 1.67 times faster); the 17408 -> 5120 forward and the 5120 -> 17408 input gradient (rows without a `ROW_RULES` entry on `sm_107a`, on which the plan before met the tier: taken for speed alone) a four-way split at T 16 - 256 (`dense_proj_det_gemm_kk_n256_pro15` / `dense_proj_det_gemm_kn_n256_pro7`, 1.08 to 1.21 times faster; slower at T 1, so T 1 - 15 keep their plan); the 4096 -> 12288 input gradient a four-way split at T 1 - 256 (`dense_proj_det_gemm_kn_n256_pro7`, 1.05 to 1.17 times faster from T 16 and 0.89 times as fast at T 1, where the plan before missed the tier); every range plan meets the naive tier and gives the same bits run to run; a plan that splits K in more parts than the plan before sums in another order.  Version 0.3.8 extends four of the FP32 ranges to measured token counts next to them, with the same rule, and adds three at T 12288, measured on two VR200 nodes against the plan of version 0.3.7 (11 paired rounds per node in eager launches and CUDA graph replays; a token count is taken where the rule was faster in at least 9 of the 11 rounds with a median ratio of at least 1.015 on both nodes in both modes, gave the same output on both nodes and the same bits run to run, and, where it sums in another order, met the naive BF16 tier on five seeds of 32 draws): the 4096 -> 12288 weight gradient now runs its synchronised split at T 4096 - 16384 (1.090 to 1.137 times faster at T 4096, 1.058 to 1.069 at T 16384, another summation order); 5120 -> 17408 and 12288 -> 6144 run whole 128-row tiles at T 4096 - 8192 (1.121 to 1.160 and 1.112 to 1.114 times faster at T 4096, with the same output bits); the 16384 -> 6144 weight gradient runs its two-way K split at T 4096 - 8192, where at an odd number k of K blocks a unit of ceil(k / 2) K blocks runs on into the next tail tile (103 units at k 65 - 103, 104 at k 105 - 127; 1.031 to 1.082 times faster at T 6208 and 8128, another summation order); at T 12288 the 17408 -> 5120 and 5120 -> 17408 weight gradients run the half-height tail wave of the 256-row tiles (`dense_proj_det_gemm_nn_n256_m256_f32_ht_tma1_pro7`, 636 whole tiles and 88 half items of the 44-tile tail; 1.037 to 1.045 times faster, with the same output bits) and the 16384 -> 6144 weight gradient a three-way K split of the 26 tail tiles of the 256-row tiles (`dense_proj_det_gemm_nn_n256_m256_f32_tma1_pro7`, 78 units of 64 K blocks; 1.045 to 1.052 times faster, another summation order); at the measured T 16172, 16231, 16384, 24576 and 32768 these T 12288 rules did not meet that bar, and the rows keep their plan from T 12289 on.  Version 0.3.8 also adds ranges on 148- and 152-SM `sm_103a` parts and on 152-SM `sm_100a` parts: the FP32 input gradients of the 148-SM `sm_100a` part rules run that rule's program, `dense_proj_det_gemm_kn_n128_f32_tma2_s4_c1` (in raster groups of 2 on 6144 -> 128), in place of `dense_proj_det_gemm_kn_n256_f32_tma2_pd2_pro15` (6144 -> 128: `dense_proj_det_gemm_kn_n128_f32_tma1_pro7`), with the same output bits, each range over one run of neighbouring measured token counts at which the rule met the same bar on two nodes of the part, from the lowest to the highest of them (never across a measured token count at which the rule missed the bar or was not timed): on B300 (148-SM `sm_103a`) 4096 -> 32 / 48 / 64, 5120 -> 64 and 6144 -> 128 at T 8192 - 32768, 4096 -> 128 and 5120 -> 32 at T 8192 - 16384 and 5120 -> 48 at T 16231 - 16384 (1.015 to 1.100 times faster; 5120 -> 128 and 8192 -> 128 keep their plan), on GB300 (152-SM `sm_103a`) 4096 -> 32 / 48 / 64 / 128, 5120 -> 32 / 48 / 64 / 128, 8192 -> 128 and 6144 -> 128 at T 16172 - 16384 (1.015 to 1.064 times faster), on GB200 (152-SM `sm_100a`) 4096 -> 32 / 48 / 64 / 128 and 5120 -> 32 / 48 / 64 at T 16172 - 16384, 5120 -> 128 at T 16172 and at T 16384 (two ranges of one token count each; the rule was not timed at T 16231 on this row) and 6144 -> 128 at T 16172 alone (1.021 to 1.063 times faster; 6144 -> 128 at T 16231 and 16384 and 8192 -> 128 kept their plan up to version 0.3.10).  Version 0.3.9 runs two GB300 rows of that program in raster groups of 2 row tiles, with the same output bits, measured on two GB300 nodes against the plan of version 0.3.8 (the group of 16) with the same bar: the indexer_hw input gradient 6144 -> 32 at T 16172 - 32768 (1.016 to 1.022 times faster; its plan before is its `sm_100a` rule, read through `RULE_PARENT`; at T 8192 the group met the bar in three of the four node and mode pairs and the row keeps its plan) and 8192 -> 128 at T 16231 - 16384 (1.063 to 1.071 times faster; T 16172 - 16230 keep the group of 16).  Version 0.3.10 runs that program in small raster groups on B300 (148-SM `sm_103a`) and B200 (148-SM `sm_100a`) FP32 input gradients, with the same output bits, measured at T 8192, 16231, 16384 and 32768 on two nodes of each part against the plan of version 0.3.9 with the same bar (of two groups that met it the faster, the smaller within 0.005): on B300 8192 -> 128 in groups of 2 at T 8192 - 32768 (1.027 to 1.052 times faster than `dense_proj_det_gemm_kn_n256_f32_tma2_pd2_pro15`) and the indexer_hw input gradient 6144 -> 32 in groups of 4 at T 8192 (1.015 to 1.018 times faster; 5120 -> 128 keeps its plan: at T 8192 its groups met the bar but the plan before was more than 1% off a second copy of itself in one of the four node and mode pairs, and at T 16231 - 32768 they missed it); on B200 8192 -> 128 in groups of 2 at T 8192 - 32768 (1.038 to 1.078 times faster; this range replaces its part rules of the group of 16 at T 16231 and 16384), 5120 -> 128 and the indexer_k input gradient 6144 -> 128 in groups of 2 at T 16384 - 32768 (1.020 to 1.027 and 1.015 to 1.025 times faster) and indexer_hw 6144 -> 32 in groups of 4 at T 16231 and of 2 at T 32768 (1.015 to 1.019 and 1.019 to 1.020 times faster).  Version 0.3.11 runs that program in small raster groups on GB200 (152-SM `sm_100a`) FP32 input gradients, with the same output bits (also where the program changes: the same K order per output element), measured at T 16172, 16231, 16384 and 32768 on two GB200 nodes against the plan of version 0.3.10 with the same bar (groups of 2 and 4; of two groups that met it the one with the higher lowest node and mode median, the smaller within 0.005): the indexer_hw input gradient 6144 -> 32 in groups of 2 at T 16172 - 32768 (1.030 to 1.042 times faster than the same program in groups of 16), 8192 -> 128 in groups of 4 at T 16172 and of 2 at T 16231 - 32768 (1.037 to 1.070 times faster than `dense_proj_det_gemm_kn_n256_f32_tma2_pd2_pro15`), 5120 -> 128 in groups of 2 at T 16231 (1.055 to 1.064 times faster; at T 16172 and 16384, its ranges of version 0.3.8, the groups missed the bar with medians of 1.006 to 1.019, and at T 32768 with 0.983 to 1.006: those keep their plan) and the indexer_k input gradient 6144 -> 128 in groups of 2 at T 16231 - 16384 (1.017 to 1.024 times faster than `dense_proj_det_gemm_kn_n128_f32_tma1_pro7`; at T 16172 its range of version 0.3.8 runs the group of 2 already, and at T 32768 the groups missed the bar with medians of 1.012 to 1.016, not every node and mode pair a win).  Version 0.3.12 runs the start-path eight-warp program `dense_proj_det_gemm_kn_n256_tma1_w8relf` (on GB200 5120 -> 48 at T 32768 `dense_proj_det_gemm_kn_n256_tma1_w8rel`) on the BF16 input gradients 4096 -> 32 / 48 / 64 / 128, 5120 -> 32 / 48 / 64 / 128, 8192 -> 128, indexer_hw 6144 -> 32 and indexer_k 6144 -> 128 on 148- and 152-SM `sm_100a` and `sm_103a` parts, with the same output bits, measured at T 16172, 16231, 16384 and 32768 on two nodes of each part against the plan of version 0.3.11 with the same bar (of `w8rel` and `w8relf` the one with the higher lowest node and mode median, `w8relf` within 0.005): 1.021 to 1.091 times faster on B200, 1.020 to 1.095 on GB200, 1.017 to 1.094 on B300 and 1.015 to 1.077 on GB300.  A range runs from one measured T to another and holds no measured T that missed the bar; these keep their plan: on B200 4096 -> 64 at T 16172 and 16231 and 5120 -> 128, 8192 -> 128 and indexer_k at T 32768, on GB200 4096 -> 48 and 5120 -> 128 at T 16384, on B300 4096 -> 48, 8192 -> 128 and indexer_k at T 32768, and on GB300 5120 -> 64 and 8192 -> 128 at every measured T and 4096 -> 32 and indexer_k at T 32768 (GB300 5120 -> 64 at T 16172 and 4096 -> 32 at T 32768 up to version 0.3.12, B200 4096 -> 64 at T 16172 and GB300 5120 -> 64 at the other measured T up to version 0.3.13).  On `sm_100a` a range replaces the GB T-exact rule of 4096 -> 48 at T 16231 on both parts and at T 16384 on 148 SMs.  A range's rule holds for its form: a caller that forces another form (`form`, the ten-warp program included) plans with the row's rule without the part tables, as in version 0.3.11.  Version 0.3.13 runs `dense_proj_det_gemm_kn_n256_tma1_w8relf` in raster groups of 32 or 2 row tiles on three BF16 input gradients of the 152-SM parts, each range at one measured T (the rule of version 0.3.12 with `group_m`; up to version 0.3.13), with the same output bits (also where the program changes), measured at T 16172, 16231, 16384 and 32768 on two nodes of each part against the plan of version 0.3.12 with the same bar (groups of 32 and 64 on GB300, of 2 and 4 on GB200; of two groups that met it the one with the higher lowest node and mode median, the smaller within 0.005): on GB300 (152-SM `sm_103a`) 4096 -> 32 in groups of 32 at T 16172 (1.019 to 1.029 times faster than the same program in the default group of 16; T 16173 - 16384 keep its range of version 0.3.12) and at T 32768 (1.040 to 1.050 times faster than the plan of version 0.3.11, a kept cell of version 0.3.12) and 5120 -> 64 in groups of 32 at T 16172 (1.018 to 1.023 times faster than the plan of version 0.3.11; it had no range of version 0.3.12 on this part); on GB200 (152-SM `sm_100a`) 5120 -> 32 in groups of 2 at T 32768 (1.015 to 1.021 times faster than the same program in the default group of 16; T 16172 - 32767 keep its range of version 0.3.12).  These keep their plan (all but GB300 and GB200 4096 -> 32 at T 16231 up to version 0.3.13): on GB300 4096 -> 32 at T 16231 and 16384 and 5120 -> 64 at T 16231, 16384 and 32768 (the group of 32 with medians of 1.002 to 1.029 but a paired win in two or three of the four node and mode pairs, the group of 64 0.979 to 1.020), on GB200 4096 -> 32 and indexer_hw 6144 -> 32 at every measured T and 5120 -> 32 at T 16172, 16231 and 16384 (0.999 to 1.023, a paired win in at most three of the four pairs).  Version 0.3.14 runs the start-path program `w8relf` on its 128 x 128 tile on BF16 input gradients of the 148- and 152-SM parts, each range in the raster group it names (2, 16 or 32): `dense_proj_det_gemm_kn_n128_tma1_w8relf` (64-element K steps) on 4096 -> 64 and 5120 -> 64 and `dense_proj_det_gemm_kn_n128_k32_tma1_w8relf` (32-element K steps) on 4096 -> 32, 5120 -> 32 and indexer_hw 6144 -> 32, with the same output bits (also where the plan before runs another program), measured at T 16172, 16231, 16384 and 32768 on two nodes of each part against the plan of version 0.3.13 with the same bar (groups of 16 and 32 on GB300 and B300, of 2 and 16 on GB200 and B200, of 8 too on the B200 -> 64 rows; of two groups that met it the one with the higher lowest node and mode median, the smaller within 0.005).  A range runs over neighbouring measured T with the same group (the lowest and highest node and mode medians against the plan before): on GB300 4096 -> 32 in groups of 32 at T 16172 (1.032 to 1.043) and 32768 (1.026 to 1.030) and of 16 at T 16384 (1.057 to 1.067), 4096 -> 64 in groups of 32 at T 16172 (1.032 to 1.046) and 32768 (1.018 to 1.026), and in groups of 16 at T 16172 - 32768 5120 -> 32 (1.023 to 1.042), 5120 -> 64 (1.028 to 1.062; above T 16172 the plan before is its `ROW_RULES` rule, the single-CTA program) and indexer_hw (1.022 to 1.044); on GB200 4096 -> 32 in groups of 2 at T 16172 (1.051 to 1.060) and 16384 - 32768 (1.042 to 1.065), 4096 -> 64 in groups of 16 at T 16172 (1.020 to 1.028) and 32768 (1.025 to 1.031), 5120 -> 32 in groups of 16 at T 16172 - 16384 (1.037 to 1.053) and of 2 at T 32768 (1.023 to 1.024), 5120 -> 64 in groups of 2 at T 16172 (1.017 to 1.022) and of 16 at T 16231 - 16384 (1.025 to 1.044) and indexer_hw in groups of 16 at T 16172 - 32768 (1.022 to 1.048); on B300, in groups of 16, 4096 -> 32 at T 16384 - 32768 (1.020 to 1.027), 5120 -> 32 at T 16172 - 32768 (1.018 to 1.026) and indexer_hw at T 16231 - 32768 (1.018 to 1.028); on B200 4096 -> 32 in groups of 16 at T 32768 (1.020 to 1.024), 4096 -> 64 in groups of 2 at T 16172 (1.044 to 1.057; the plan before is its `ROW_RULES` rule, the `w8r` program in groups of 8) and, in groups of 16, 5120 -> 32 at T 16172 - 32768 (1.018 to 1.031) and indexer_hw at T 16172 (1.017 to 1.019) and T 16384 - 32768 (1.017 to 1.027).  These keep their plan of version 0.3.13 (no group met the bar): on GB300 4096 -> 32 at T 16231 and 4096 -> 64 at T 16231 and 16384 (medians 1.029 to 1.063, a paired win in two or three of the four node and mode pairs), on GB200 4096 -> 32 at T 16231, 4096 -> 64 at T 16231 and 16384 and 5120 -> 64 at T 32768 (1.000 to 1.064, a paired win in at most three), on B300 4096 -> 32 at T 16172 and 16231 and indexer_hw at T 16172 (1.009 to 1.020, a paired win in at most two) and 4096 -> 64 and 5120 -> 64 at every measured T (0.956 to 1.009, no paired win), on B200 4096 -> 32 at T 16172, 16231 and 16384 and indexer_hw at T 16231 (0.989 to 1.022, a paired win in at most two), 4096 -> 64 at T 16231 (1.050 to 1.065, a paired win in two) and at T 16384 and 32768 (0.966 to 1.005, no paired win) and 5120 -> 64 at every measured T (0.958 to 1.006, no paired win).  The 208-, 216- and 224-SM `sm_107a` parts, the `sm_103a` parts of other SM counts, the `sm_100a` parts of other SM counts than 148 and 152 and every other token count plan these rows as before |
| registry fallback | a plan whose template this architecture's registry lacks is redone without the swap (the batched small-M swap or the unbatched weight-gradient swap; `plan.swap_fallback`), then without the row's measured rule (`plan.rule_fallback`: the architecture default `pro` form), then lands on the nearest generated program of the row's layout / output class (`plan.knob_fallback`): the hand-ordered knob variants first (plain TMA L2 policy, the single-CTA form, the 64-row pair family, the tall tile, the register epilogue, BLOCK_N 128, each in the three `pro` forms), then every generated program of the class, nearest symbol first, with the knobs its symbol decodes to; a caller-forced knob always wins.  Only a class without any generated program on the architecture raises (`NotImplementedError` at binding) |
| synchronised TMA-store gate | the synchronised stream-K programs that store their bf16 tiles through the TMA store (`_tma<slots>_sks`; the fp32-output `_f32_tma<slots>_sks` programs, which take the TMA store by default at every K, are served as before) were generated for measured rows whose rule names `epi="tma"` (`_serves`); a ragged synchronised row whose short K (`K <= K_TMA_BF16`) turns the epilogue default to the TMA store plans the same symbol unmeasured (two dense weight gradients on the 160-SM `sm_103a` part at T 961-1024), so the registry does not serve it there and the row keeps the registry fallback it ran before those programs were generated; a caller-forced `epi` is served |
| grid | `(num_full + sk_units) * 2` CTAs: one cluster of two CTAs per `2 * CTA_ROWS` x `BLOCK_N` output tile plus the stream-K units; under `cta1` (round 19) `num_full` CTAs, one per `CTA_ROWS` x `BLOCK_N` tile, launched with the module's recorded cluster dims (1, 1, 1) |
| stream-K | `stream_k_plan(pair_tiles, k_blocks, sm_count // 2, sk="auto")`: only when the whole problem is one partial wave of CTA pairs whose halves fit the pairs (`pair_tiles <= pairs`, `2 * tail <= pairs`) and `k_blocks >= 16`, the tiles' linearised K steps are cut into units of `iters_per_unit = ceil(k_blocks / 2)` (two K-aligned halves per tile on an even `k_blocks`), `sk_units = ceil(pair_tiles * k_blocks / iters_per_unit)` so that every unit holds a K step, with an in-kernel deterministic fixup; otherwise every work item is a whole tile (multi-wave tails stay data-parallel).  `sk=True` is the linearised-units policy (up to `pairs` units of >= 8 K steps), `sk="tiles"` the whole-tile diagnostic, `sk=False` off.  A measured `sk_parts = p` rule, or the caller knob `sk_parts`, instead splits every tail tile `p` ways when `p * tail` units fit the pairs and each keeps >= 8 K steps (`sk_parts_plan`).  The host asserts `tail_tiles * 16 <= 4096` slice counters, and `check_stream_k_plan` refuses a plan outside the kernel's launch contract: a unit without a K step on the `pd` 2 / `sk_slab` 2 / `stt` programs (they hand off the work in an item's last segment, so such a unit never finishes), synchronised collectors outside `[1, tail_tiles]` (`[1, ...)` on the other `sk_sync` programs), or a linearised plan with `2 * iters_per_unit < k_blocks` (more than `WS_PARTS` units on a tail tile) on a program that keys its workspace by tile (`tile_keyed_workspace`; its binding refuses the same plan) |

`cake_dense_gemm_det_train_projection_plan(A, B, out, sm_count=..., l2_bytes=...)` exposes
the plan (`CakeDenseGemmDetProjectionPlan`, including `wave_working_set_bytes`) without a device;
the prepared launch resolves the template
through `cake_jit.KERNELS[arch][template]`, owns its stream-K partial slabs
(`max(sk_units + tail_tiles, WS_PARTS * tail_tiles) * 2 * CTA_ROWS * BLOCK_N`
fp32, `WS_PARTS = 3`: the carried programs key a tail tile's partials by unit,
the programs rendered from the current kernel source by tile, `WS_PARTS` slabs
per tail tile of a linearised plan - `tile_keyed_workspace`; one slab per tail
tile under `sk_sync`, none under the half-tail plan), its slice counters
(`max(8192, tail_tiles * 16)` u32: the lane-spread dummy counters above index
4096 must be addressable), and binds the
module's argument plan by keyword (`bind_launch`, fails closed on an unknown
name).  Given the same device the result is bitwise identical to the Cake
launcher's.

### Three-chain FP32 weight gradient (`sm_107a`)

`CHAIN_RANGES_BY_SM[(arch, sm_count)]` maps a weight-gradient row
`(N, K, M)` (`K` the token count) to token ranges `(lo, hi, (block_n,
chains), over_exact)`.  Inside a range the FP32 weight gradient plans
`ChainGemmPlan`: one CTA pair per output tile computes the whole K
reduction, its `chains` TMEM accumulators each own a fixed contiguous range
of K blocks (`chain_bounds`), and the chains are added in a fixed order in
the epilogue (`chain_det_gemm_nn_n{block_n}_f32_t_c{chains}`).  There is no
workspace, no atomic and no split of K across CTAs, so the result is bitwise
identical from run to run on one device.  An exact `ROW_RULES` key of the
row keeps precedence unless the range sets `over_exact`; a part without an
entry (the 208-SM `sm_107a` parts, every other architecture), a caller knob
other than `transposed_out` / `sk` / `arch`, a batched or non-FP32 output
and an unregistered template all plan as before.  The ranges list the
`sm_107a` 212- and 216-SM parts for the 5120 -> 1024 and 5120 -> 512 rows.  Three ragged `sm_107a`
weight-gradient rules sit beside them: the 4096 -> 512 and 4096 -> 1024 FP32
rows plan the programs of their T 16231 keys at every token count without an
exact key (the 4096 -> 512 rule yields to the plan without it up to T 960),
and the 5120 -> 1024 BF16 row plans the 224-column tile with the
column-grouped raster (`_n224_hne_cg_pro15`).

Determinism scope: a planned launch gives the same bits on every run of one
package version on one device.  The bits are not promised across versions,
across plans (a change of `ROW_RULES`, `CHAIN_RANGES_BY_SM` or the SM count
selects a different summation order) or across devices; every plan meets
the same absolute accuracy budget against an FP64 reference.

### The swapped small-token forward

The convenience path (`cake_dense_gemm_det_train_projection`) runs a batched forward row
that the planner would swap (`swap_small_m`: `L > 1`, `M <= 256`, `N >= 2 M`)
on a program that puts the weight rows in the MMA's M and the tokens in its N
(the `swap_ab_<tile>` templates), when `A` and `B` are K-major, the output is
BF16 without `transposed_out`, `K` is a multiple of 64 and every view's row and
batch strides and start are in 16-byte units.  A CTA streams a
`tile_w x 64` weight box per stage through a deep TMA ring (64 rows over 12
stages, or 128 rows over 8) against a 64-token tile, single CTA, no split: each
output element is the one FP32 sum of its whole K, rounded once.  The tile is
the narrow one while its `ceil(N / 64) * ceil(M / 64) * L` CTAs fit one wave of
the device's SMs, the wide one while its CTAs fit two waves; a row whose wide
tiles need more than two waves runs the planner's program, which streams the
weights faster there (`swap_ab_template`, `SWAP_MAX_WAVES`).  The program
reads the caller's views through tensor maps that carry their strides and
stores at `l * out_l + m * ldo + n`, so the token-major `[T, L, K]` /
`[T, L, N]` views of a grouped projection run without a copy.  Every other row,
and every prepared launch, runs the planner's program.

Every K1 entry point (`cake_dense_gemm_det_train_projection`, the `projection_det_*`
operations and the prepared launches) takes `enable_pdl`, by default whether
the device supports programmatic dependent launch (every architecture the
package serves does), and launches its program as a programmatic dependent
launch under it: every CTA triggers its dependents at entry.  The planner's
programs wait for the predecessor grid before their first global access
(operands, output, stream-K workspace and counters), right after their
prologue, so the prologue overlaps the predecessor's tail.  The swapped
program waits the same way by default.  `weights_static=True` states that no
kernel still in flight on the stream writes `B` (a weight of the model between
updates): the swapped rows then run the `_we` programs, whose weight loads start
before `griddepcontrol.wait` while only the token loads wait, so the weight
stream overlaps the predecessor's tail.  Do not pass it right after a kernel
that writes the weights (an optimizer step).  The router (K2) and its fused
split take `enable_pdl` too and wait before their first global access.

### K2 contract

* `A`, `B`, `out` FP32 2-D views: `A [M, K]` with unit stride on `m` or `k`
  (other stride a multiple of 4), `B [K, N]` with unit stride on `k` or `n`,
  `out [M, N]` with unit inner stride and 16-byte rows; `N` a multiple of 8.
  A single-split `kk` / `kn` launch stores 16-byte vectors straight into `out`,
  so `out` must then start at a 16-byte aligned address.
* The operand layouts select the instance and the default split-K count:
  `kk` (forward, 4 splits), `kn` (input gradient, 1), both MN-major -> the
  swapped transposed-store weight gradient `nn_t` (6 splits; `out.T = B.T
  A.T`).  An MN-major `A` with a K-major `B` has no instance.
  `ROUTER_SPLIT_RULES` replaces the default count (`router_splits`, used when
  `splits` is `None`) for N = 256 rows measured faster with another count on a
  device: by architecture, SM count, layout, hidden size H and token-count
  window T 16129-16384:
  `sm_100a` at 148 / 152 SMs, weight gradient H 5120: 9 splits;
  `sm_100a` at 148 / 152 SMs, weight gradient H 6144: 3 splits;
  `sm_103a` at 148 / 152 SMs, weight gradient H 5120: 9 splits;
  `sm_103a` at 148 / 152 SMs, weight gradient H 6144: 3 splits.
* Every `launch()` re-splits the second operand into the retained
  `[3, outer, inner]` BF16 stack in the operand's own layout (exact three-way
  split, no allocation: one launch of the fused split program
  `split_fp32_det_bf16x3` when this checkout registers it for the device, else a
  five-op torch chain through an fp32 scratch -- the same bits either way),
  launches, and sums the split-K partials on the host in a fixed order.  Splits, chunk boundaries and the reduction order are
  fixed: bit-exact run to run.  The bits are a function of the split count:
  another count (a caller's `splits`, or a `ROUTER_SPLIT_RULES` count that a
  later version adds or changes) gives other bits within the same accuracy.  Error against an FP64 reference: FP32-class
  (`rel_fro` about 2e-7 to 4e-7 on the GLM-5.2 router shapes).

## Layout of this package

* `cake_jit.py` -- the registry (schema v2) and the loader glue.  Three
  literals filled by the generated-program export, each wrapped in
  `# fmt: off` / `# fmt: on` and rendered one record per line (the export
  replaces a literal from its head line up to the next `# fmt: on`):
  `ARG_PLANS` (the distinct argument plans -- the ordered `[kind, name]` pairs
  a generated entry takes -- keyed by the first 12 hex digits of the SHA-256
  of the plan's compact JSON, `arg_plan_key`; one plan serves every K1 program
  and one every K2 program), `MODULES` (one record per generated program:
  template, kernel symbol, the architecture-free `library` stem, FFI entry,
  `arg_plan` key -- `arg_plan_of` resolves it and still accepts an inline
  list -- launch geometry, `arches`) and `KERNELS` (`arch -> template ->
  module name`).  `ARCHES` names the registered architectures;
  `load_cake_dense_projection_det_gemm_module(name, arch)` loads the build
  `<library>_<arch>` through `cake_build`.
* `cake_build.py` + `build.json` -- the loader shared by every Cake-exported
  package (byte-identical copies) and this package's build manifest (schema
  `cake-build/1`, written by the export): one entry per `<library>_<arch>`
  with its translation units, digests, nvcc flags, closure digest and
  `aot/<name>.so` path.  `cake_build.load_library(name)` loads the prebuilt
  library when `aot/` holds it and otherwise JIT-builds the entry through
  the package's JIT shim `_cake_pplx_jit.py` (`gen_jit_spec`, sources
  digest-verified first); `cake_jit.specs_for_arch(arch)` lists the JIT specs
  of an architecture for ahead-of-time warm-up builds.  `cake_build.jit_spec(name)` /
  `jit_specs(arch)` gate an entry on its `min_cuda` only (`sm_107a` entries
  need CUDA 13.5): on an older toolkit `jit_spec` raises and `jit_specs`
  leaves the entry out.
* `cake_backend.py` -- view validation, instance selection, stream-K
  planning, argument-plan binding, the prepared launches and the eager entry
  points.
* `csrc/cake_dense_projection_det_gemm/` -- generated kernel and binding
  translation units (`.clang-format` disables formatting: the sources are
  identity-checked by the manifest's digests); `csrc/cake_ffi_checks.h` -- the
  tvm-ffi tensor checks the bindings include (shipped with the package).

## Status

The registries hold the generated programs of the current export lock: the
Cake exporter emits one program per template, shared by `sm_100a` and `sm_107a` wherever the generated text is the same (the per-architecture
template lists are `KERNELS` in `cake_jit.py`; `K1_TEMPLATES` of the exporter
is the source of truth and is regenerated whenever `ROW_RULES` resolve a new
instance).  `build.json` ships the builds of the architectures `cake_jit.KERNELS`
lists (`sm_100a`, `sm_103a` and `sm_107a`); the eight-warp form ships for
`sm_100a` and `sm_103a` (`_w8r`, `_w8rs`, the two batched `_w8rs` programs, the start-path `_w8rel` / `_w8relf`,
which the token ranges of version 0.3.12 name, and `_n128_tma1_w8relf` / `_n128_k32_tma1_w8relf`, the `w8relf`
program on the 128 x 128 tile in 64- and 32-element K steps, which those of version 0.3.14 name; `sm_100a` also the
knob-free `_w8`); every program the measured
`sm_100a` / `sm_103a` rules name is registered, so the registry fallback only
serves a caller that forces a template no export generated.  An entry point raises
`NotImplementedError` naming the missing instance for a template that no
export has generated and for every template on a device whose architecture the
registry does not list.

CUDA Graphs: prepare outside capture and replay the prepared launch
(`*_prepare` once, `launch` many times).  The convenience entry points
(`cake_dense_gemm_det_train_{projection,router}` and their `_forward` / `_dgrad` / `_wgrad`) reuse a cached planned
launch per (device, stream, geometry) key and the shared workspace arena of its
(device, stream); capture a prepared launch object rather than the convenience
entry points (see "Host paths").

Tests: `tests/experimental/test_cake_dense_gemm_det_train.py` and
`tests/experimental/test_cake_dense_gemm_det_train_names.py` (host-side
planning tests run everywhere; the GPU tests skip without a registered
generated instance -- a device of an architecture `cake_jit.KERNELS` lists).  Benchmark:
`benchmarks/bench_cake_dense_gemm_det_train.py` (the 11 GLM-5.2 projection
rows x fwd / dgrad / wgrad x T in {16231, 16172}, the router rows and the two
batched MLA rows, paired against `torch.matmul`; `--host-overhead` adds the
host time per call of the prepared launch and of the convenience entry point).
