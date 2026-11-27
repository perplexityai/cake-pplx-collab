# cake-pplx-mok

Cake MoK (Mixture of Kittens) training kernels: one expert-parallel MoE layer with the routed-token
scheduler, the fused forward, the context-only recompute and the fused backward. Public module:

- `cake_pplx.cake_mok_det_train`: `prepare_mok_bf16`, `create_mok_bf16_workspace`,
  `quantize_mok_mxfp8_weights`, the communication-SM tables and resolvers (`COMM_SMS`,
  `resolve_comm_sms`, ...) and `ROUTER_WEIGHT_POSITIONS`.

Install this family alone, pinned to a commit:

```bash
pip install "cake-pplx-mok @ git+https://github.com/perplexityai/cake-pplx-collab@<sha>#subdirectory=kernels/mok"
```

Routed experts run in BF16, or in MXFP8 with caller-prequantized weights; the shared expert runs in BF16.
Plain and clamped SwiGLU, BF16 or FP32 weight-gradient accumulation, and unequal (including zero) source
counts per rank are supported. `prepare_mok_bf16(router_weight_position="before_down")` applies each route's
FP32 router weight to the SwiGLU activation before the down projection (`h = BF16(w * a)`, an unweighted
combine, router-weight gradient `dot(dh, a)`): top-k 6 over 384 routed experts at EP 4/8/16/32/64 (and top-k
2 on the diagnostic layouts), clamped SwiGLU, FP32 weight gradients, BF16 routed weights, on `sm_100a`, `sm_103a`
and `sm_107a`; `recompute_forward_context` then takes the router weights. The explicit modes
`router_weight_gradient="direct"` (a router-weight gradient `dot(dh, h)` from the unscaled `dy`, defined at a zero
weight) and `shared_expert="scaled"` (a per-token FP32 shared-expert scale `s`, `y = s * S(x) + sum_k w_k E_k(x)`,
with its gradient as a ninth `backward` item) run 512 routed experts at EP8 `(8, 64)`, top-k 10, plain SwiGLU,
FP32 weight gradients and BF16 routed weights on `sm_100a` and `sm_103a`; their defaults keep every earlier bit.
With MXFP8 routed experts, two options are off by default (section MXFP8 options):
`quantize_mok_mxfp8_weights(..., blocks="32x32")` gives each 32 x 32 block of a routed weight one scale byte, so the
backward GEMMs read the forward's dequantized weight; `prepare_mok_bf16(mxfp8_transposed="requantized")` forms the
token-direction copies of the routed `x` and SwiGLU activation `h` that the routed weight gradients read from the
forward GEMMs' MXFP8 copies, `x_t = Q_tok(DQ(Q_feat(x)))` and `h_t = Q_tok(DQ(Q_feat(h)))`, on `sm_100a` and
`sm_103a`. With both off every output keeps its earlier bits; the two combine.
The backend README
([`cake_pplx/_backends/cake_mok_bf16/README.md`](cake_pplx/_backends/cake_mok_bf16/README.md)) has the
full API, contract, memory rules and measured communication-SM tables.

## Repeatability

The module is deterministic. Each output repeats bit for bit from run to run: `y`, the forward context,
the recomputed context (identical to the forward's), `dx`, `dscores` and the six weight gradients (or the
six FP32 accumulators), for BF16 and MXFP8, for plain and clamped SwiGLU, for both router-weight
positions and for the explicit modes (`d_shared_scale` included). The scope is the same inputs, weights, `MoKConfig` values other than the communication-SM split,
precision, `swiglu_limit`, `fp32_wgrad`, `router_weight_position`, EP group, package build and GPU model.
No bit depends on the communication-SM split, on CUDA Graph replay versus eager launches, or on other kernels
resident on other streams. Every reduction runs in a fixed order; atomics act on integers only.

## Requirements

| Item | Requirement |
|---|---|
| GPU | `sm_100a`, `sm_103a` or `sm_107a` (compute capability 10.0, 10.3 or 10.7), both router-weight positions; the 512-expert top-10 group (`router_weight_gradient="direct"`, `shared_expert="scaled"`) and `mxfp8_transposed="requantized"` on `sm_100a` and `sm_103a` only |
| CUDA toolkit (`nvcc`, JIT) | 12.8 for `sm_100a`, 12.9 for `sm_103a`, 13.5 for `sm_107a` |
| `torch` | with `torch.distributed` and symmetric memory (multicast) |
| Process group | Initialized, its CUDA device current; all ranks in one peer-accessible NVLink domain with multicast |
| EP sizes | 1, 4, 8, 16, 32, 64 (EP1 uses plain tensors) |
| Scheduler layouts `(EP, local experts)` | `(1, 4)`, `(4, 4)`, `(4, 64)`, `(8, 32)`, `(16, 16)`, `(32, 8)`, `(64, 4)`, `(8, 36)`, `(32, 9)`; top-k 2 or 8. `router_weight_position="before_down"`: `(4, 96)`, `(8, 48)`, `(16, 24)`, `(32, 12)`, `(64, 6)` at top-k 6, `(1, 4)` and `(4, 4)` at top-k 2. `(8, 64)` at top-k 10 with `router_weight_gradient="direct"` and `shared_expert="scaled"` |
| Schedule capacity | `schedule_capacity_multiplier` at least `ceil((min(topk, E) * T * EP + E * 255) / (T * topk)) / EP` holds every routing (`E` local experts, `T` the workspace's physical source rows, at least the largest rank's source rows or `source_capacity`); the same multiplier on every rank, computed from a row bound all ranks agree on (the bound decreases as `T` grows); no upper bound |

## Builds

`build.json` lists 285 builds of 108 kernels: 71 kernels for each of `sm_100a`, `sm_103a` and `sm_107a` (the
router-weight-before-down fused kernels, the top-k 6 epilogue, the scheduler kernels of the 384-expert layouts and
their fused schedule kernels included), two more for `sm_107a` only, and 35 more for `sm_100a` and `sm_103a` only
(the kernels of the two mode groups and the eight requantized MXFP8 kernels).

| Group | Kernels |
|---|---|
| BF16 fused | forward (plain, clamped; with or without the saved context), context recompute (plain, clamped), backward (plain, clamped; BF16 or FP32 weight gradients); router weight before the down projection: clamped forward (with or without the saved context), clamped recompute, clamped backward with FP32 weight gradients |
| MXFP8 fused | forward, context recompute (plain, clamped), backward (plain, clamped; BF16 or FP32 weight gradients); `sm_107a` only: the multi-macrobatch forward (plain, clamped); `sm_100a` and `sm_103a` only: the same eight kernels with requantized token-direction copies (`mxfp8_transposed="requantized"`) |
| Scheduler | count and rows per scheduler layout, pad per EP size; per 384-expert layout, the fused count (counts, padding, peer-rank sentinel and expert layout in one launch) |
| Layout | the expert layout, zero |
| Communication | barrier, gather, gather of int64 expert IDs |
| Epilogues | forward (weighted sum), top-k 2 and 8; backward (unweighted sum, also the router-weight-before-down forward's), top-k 2, 6 and 8 |

The locality-domain (`_domain`) builds of the MoK kernels, which split work over the two dies of an
`sm_107a` GPU, are not part of this package.

## MXFP8 weights

MXFP8 routed experts read caller-prequantized weights. `quantize_mok_mxfp8_weights(gate, up, down, *, blocks="1x32")`
(BF16, gate and up `[E, I, H]`, down `[E, H, I]`) returns `(forward_weights, backward_weights)`. Callers may pass
their own MXFP8 weights in this layout instead:

- `forward_weights` = `((gate, gate_scales), (up, up_scales), (down, down_scales))`: E4M3 data
  (`torch.float8_e4m3fn`) in the weight's own shape, quantized along its last dimension.
- `backward_weights` = `((gate, gate_scales, gate_t, gate_scales_t), (up, up_scales, up_t, up_scales_t),
  (down_t, down_scales_t))`: the `_t` pairs quantize the transposed weight (`[E, H, I]` for gate and up,
  `[E, I, H]` for down) the same way.
- Each 32 consecutive elements along the quantized dimension share one UE8M0 scale byte:
  `byte = round_up(max(amax * 0.002232142857, 1e-12))` (the FP32 exponent field, plus one when the mantissa is not
  zero), saturated to 254; the data is `satfinite_e4m3(x * 2**(127 - byte))`.
- Scales of an `[E, rows, cols]` weight are uint8 `[E * rows / 128, cols / 128, 32, 16]`: in each 128 x 128 tile, the
  byte of row `r` and 32-column block `k` is at `(r % 32) * 16 + (r // 32) * 4 + k`. Experts are concatenated along
  the first dimension.
- `blocks` (keyword, one of `MXFP8_WEIGHT_BLOCKS`) selects the elements that share a scale byte: `"1x32"` (default)
  is the layout above; `"32x32"` is in section MXFP8 options.

The bundled quantizer runs PyTorch operations on the BF16 weights (no Triton dependency). Requantize after every
weight update. Its measured cost for both block shapes is in section MXFP8 options.

## MXFP8 options

Two options change the MXFP8 operands of the routed experts. Both are off by default, and with both off every output,
saved tensor and quantized weight keeps its earlier bits. They are independent (`"requantized"` reads routed weights of
either block shape), and the tests also run them together. Their costs below were measured separately.

| Option | Keyword (default) | What changes |
|---|---|---|
| 32 x 32 weight blocks | `quantize_mok_mxfp8_weights(..., blocks="32x32")` (`"1x32"`) | The routed weights' scale bytes and E4M3 codes, both layouts. Kernels and launchers are unchanged. |
| Requantized activation copies | `prepare_mok_bf16(..., mxfp8_transposed="requantized")` (`"source"`) | The token-direction copies `x_t`, `h_t` and their scales, which only the routed weight gradients read. `sm_100a` and `sm_103a`. |

### 32 x 32 weight blocks

`"32x32"` gives each 32 x 32 block of gate, up and down one UE8M0 byte, the byte of section MXFP8 weights for the
block's largest magnitude, and converts every element of the block with it. The `_t` data are then the forward data
transposed (the same codes), so the forward and the backward GEMMs read the same dequantized weight; the scales keep
the tile layout of section MXFP8 weights, each block's byte repeated for its 32 rows (forward) or its 32 columns
(`_t`). The kernels read both as ordinary MXFP8 operands. The bytes do not depend on the quantizer's expert chunking.
A `blocks` that is not a `str` raises `TypeError`, an unknown one `ValueError`, before any tensor is read.

Measured cost of `quantize_mok_mxfp8_weights` per block shape, median of 10 calls after 3 warm-up calls (CUDA events),
median over 6 rounds in two processes with the block shapes alternating (PyTorch 2.14.1 with CUDA 13.0), with the peak
memory it allocates beyond the inputs (its outputs included):

| GPU | Routed weights (E, I, H) | `"1x32"` time | `"32x32"` time | `"1x32"` peak | `"32x32"` peak |
|---|---|---|---|---|---|
| GB300 (`sm_103a`) | 64, 2048, 6144 | 51.541 ms | 40.060 ms | 5.640 GiB | 5.372 GiB |
| GB300 (`sm_103a`) | 36, 2048, 4096 | 19.402 ms | 15.111 ms | 2.505 GiB | 2.474 GiB |
| GB300 (`sm_103a`) | 36, 1024, 4096 | 9.757 ms | 7.604 ms | 1.301 GiB | 1.284 GiB |
| GB200 (`sm_100a`) | 64, 2048, 6144 | 52.041 ms | 40.328 ms | 5.640 GiB | 5.372 GiB |
| GB200 (`sm_100a`) | 36, 2048, 4096 | 19.594 ms | 15.218 ms | 2.505 GiB | 2.474 GiB |
| GB200 (`sm_100a`) | 36, 1024, 4096 | 9.853 ms | 7.663 ms | 1.301 GiB | 1.284 GiB |

`"32x32"` takes 22.1 to 22.5 % less time and less peak memory.

Fused layer, `"32x32"` against `"1x32"` weights with one build, on GB200 (`sm_100a`) and GB300 (`sm_103a`): the
settings, timing and node pairs of the `"requantized"` cost below (EP8, H 6144 with 256 experts and H 4096 with 288
experts, the third node pair of each GPU model), with the `"1x32"` arm timed twice per round for the A/A band and calls
41-60 of each process as a second window. A row is slower (or faster) in a window when its 95 % interval excludes zero
and its mean exceeds the A/A band, or when all 6 rounds have the same sign and the median difference exceeds the largest
A/A difference (also tested against the mean of the two `"1x32"` timings). 15 of the 16 rows (forward, context
recompute, backward and checkpointed step of both configurations on both GPUs; all 8 on GB200) are neither slower nor
faster in either window: their differences in calls 1-20 are -12.4 to +49.1 us on 1.17 to 29.1 ms. The GB300 backward
at H 4096 is slower in calls 1-20 only: 11.709 ms with `"1x32"` and 11.736 ms with `"32x32"` weights, +26.3 us (+0.2 %)
[+16.2, +36.5] against an A/A band of 15.6 us; in calls 41-60 the difference is +17.0 us [-7.0, +41.1] against a band
of 30.1 us. Both node pairs failed the noise check (49.10 us on GB200, 23.70 us on GB300), so differences smaller than
that are not resolved.

### Requantized activation copies

The MXFP8 kernels quantize the routed `x` and the SwiGLU activation `h` (BF16, schedule-order rows of a macro batch,
padding rows zero) twice: along the features for the forward GEMMs (`Q_feat`, 32-element row blocks) and along the
tokens for the routed weight gradients (`Q_tok`, blocks of 32 consecutive rows). `prepare_mok_bf16(mxfp8_transposed=...)`
(one of `MXFP8_TRANSPOSED_MODES`) selects the token-direction copies:

| Mode | `x_t` | `h_t` |
|---|---|---|
| `"source"` (default) | `Q_tok(x)` | `Q_tok(h)` |
| `"requantized"` | `Q_tok(DQ(Q_feat(x)))` | `Q_tok(DQ(Q_feat(h)))` |

`DQ(code, byte) = code * 2**(byte - 127)` is the BF16 value the forward GEMMs read (exact below `2**128`; an element
within 3 % of the BF16 maximum dequantizes to `2**128`, which is infinity in BF16); `Q_tok` is the recipe above with
round to nearest. The kernels form the copies wherever they write them: the forward, the context recompute and the
backward's replay of later macro batches. The forward output, the feature-direction copies, the stored gate and up
and every other saved tensor are the default's bytes, and so are `dx`, the score gradient and the shared weight
gradients; only the routed weight gradients read other operands. With power-of-two scales an E4M3 value that is
normal at its block's scale keeps four significant bits at any scale, so the two copies hold the same value for an
element unless it is subnormal at a scale it is quantized at: its feature block's, or its token block's in either
token-direction copy (the feature rounding can land on a midpoint of the token block's subnormal grid, which then
rounds to even). Requantizing can also change a token block's scale byte (down, or up by one through an element
subnormal at its feature block's scale), which moves that block's subnormal range and gives its other values other
codes. Such an infinity gives its token block byte 254 in the `"requantized"` copy: a NaN code for it and code 0 for
the block's other elements. `"requantized"` runs MXFP8 routed weights with `router_weight_position="after_down"`
outside the mode groups, on `sm_100a` and `sm_103a`, and raises `ValueError` otherwise (before any kernel is loaded
or launched).

Measured cost of `"requantized"` against the default. Both arms run the programs this version ships, interleaved on
the same GPUs, eager, CUDA 13.4 toolkit: the default programs, which an earlier revision of the kernel sources
generated and this option leaves unchanged, and the `"requantized"` programs, generated from a later revision. Common
settings: EP8 on two 4-GPU nodes of one NVLink domain, 16,384 tokens per rank, intermediate 2048, top-8, uniform
routing, `router_weight_position="after_down"`, FP32 weight gradients, MXFP8 routed weights quantized once outside the
timed region, BF16 shared expert, macro batches of 32,768 rows (5 per rank, the last one partial). Configurations:
hidden 6144 with 256 routed experts (32 per rank) and plain SwiGLU; hidden 4096 with 288 routed experts (36 per rank)
and SwiGLU clamped at 10. Each time is the slowest rank's median of timed calls 1-20 after 3 warm-up calls in one
process, median over 6 interleaved rounds; calls 41-60 form a second window. Each round also times the default a second
time; the A/A band is the larger absolute end of the 95 % interval of the mean default-vs-default difference. The
checkpointed step times `build_schedule`, the context-saving forward, `recompute_forward_context` and `backward` in one
region. Calls 1-20:

| GPU | Configuration | Operation | Default (ms) | `"requantized"` (ms) | Difference, us (%) [95 % CI] | A/A band (us) |
|---|---|---|---|---|---|---|
| GB200 (`sm_100a`) | EP8 (8 x 32), H 6144, 256 experts, plain | forward | 8.500 | 9.236 | +723.5 (+8.5 %) [+552.2, +894.7] | 194.9 |
| GB200 (`sm_100a`) | EP8 (8 x 32), H 6144, 256 experts, plain | context recompute | 1.739 | 1.781 | +48.5 (+2.8 %) [+5.0, +92.0] | 28.3 |
| GB200 (`sm_100a`) | EP8 (8 x 32), H 6144, 256 experts, plain | backward | 19.040 | 19.551 | +511.8 (+2.7 %) [+409.1, +614.5] | 144.3 |
| GB200 (`sm_100a`) | EP8 (8 x 32), H 6144, 256 experts, plain | checkpointed step | 29.089 | 30.423 | +1325.9 (+4.6 %) [+1059.0, +1592.8] | 344.8 |
| GB200 (`sm_100a`) | EP8 (8 x 36), H 4096, 288 experts, clamped 10 | forward | 5.686 | 6.233 | +555.1 (+9.8 %) [+472.4, +637.8] | 88.8 |
| GB200 (`sm_100a`) | EP8 (8 x 36), H 4096, 288 experts, clamped 10 | context recompute | 1.321 | 1.374 | +49.5 (+3.8 %) [+25.3, +73.7] | 22.2 |
| GB200 (`sm_100a`) | EP8 (8 x 36), H 4096, 288 experts, clamped 10 | backward | 14.023 | 14.148 | +91.1 (+0.7 %) [-116.4, +298.5] | 100.2 |
| GB200 (`sm_100a`) | EP8 (8 x 36), H 4096, 288 experts, clamped 10 | checkpointed step | 20.670 | 21.636 | +904.0 (+4.4 %) [+769.3, +1038.8] | 96.5 |
| GB300 (`sm_103a`) | EP8 (8 x 32), H 6144, 256 experts, plain | forward | 7.179 | 7.965 | +783.9 (+10.9 %) [+766.0, +801.8] | 39.9 |
| GB300 (`sm_103a`) | EP8 (8 x 32), H 6144, 256 experts, plain | context recompute | 1.492 | 1.549 | +54.1 (+3.6 %) [+45.4, +62.8] | 14.5 |
| GB300 (`sm_103a`) | EP8 (8 x 32), H 6144, 256 experts, plain | backward | 15.944 | 16.600 | +658.7 (+4.1 %) [+634.0, +683.4] | 33.5 |
| GB300 (`sm_103a`) | EP8 (8 x 32), H 6144, 256 experts, plain | checkpointed step | 24.391 | 25.768 | +1385.9 (+5.7 %) [+1332.4, +1439.3] | 47.4 |
| GB300 (`sm_103a`) | EP8 (8 x 36), H 4096, 288 experts, clamped 10 | forward | 5.010 | 5.687 | +665.9 (+13.3 %) [+639.6, +692.2] | 34.4 |
| GB300 (`sm_103a`) | EP8 (8 x 36), H 4096, 288 experts, clamped 10 | context recompute | 1.168 | 1.214 | +46.6 (+4.0 %) [+28.4, +64.9] | 19.4 |
| GB300 (`sm_103a`) | EP8 (8 x 36), H 4096, 288 experts, clamped 10 | backward | 11.713 | 12.305 | +595.9 (+5.1 %) [+587.8, +604.1] | 22.1 |
| GB300 (`sm_103a`) | EP8 (8 x 36), H 4096, 288 experts, clamped 10 | checkpointed step | 17.498 | 18.743 | +1253.1 (+7.2 %) [+1218.3, +1287.9] | 69.9 |

By the rules of the fused-layer timing above, every row but the GB200 backward at H 4096 is slower in at least one
window: the forward by 8.5 to 13.3 %, the context recompute by 2.8 to 4.0 %, the backward by 2.7 to 5.1 % and the
checkpointed step by 4.4 to 7.2 % (calls 1-20). The GB200 backward at H 4096 is neither slower nor faster in either
window.

Before the rounds, each node pair timed a short BF16 forward twice per round with the default kernels, in 3 rounds; the
check asks for a 95 % band of at most 12 us in calls 1-20. Up to three pairs were tried per GPU model and all failed
(GB300 27.06, 17.78 and 23.70 us; GB200 25.33, 55.15 and 49.10 us); the third was used anyway, so differences smaller
than its band are not resolved. The GB200 `"requantized"` part at H 4096 did not fit in the third pair's time window
and ran on a fourth pair, which failed the check too (51.59 us). On GB200 the GPUs' software power cap was active in
most timed lines (per-rank median SM clocks about 1.6 to 2.1 GHz), and in the forward, backward and checkpointed step
the `"requantized"` lines ran at a higher median SM clock (2.01 to 2.06 GHz) than the default lines (1.86 to 1.93 GHz);
on GB300 the two arms' median SM clocks differ only in the H 6144 backward and checkpointed step (2.07 against 2.03
GHz).

## Precision tests

`tests/experimental/test_mok_accuracy.py` (single GPU) and `tests/experimental/test_mok_distributed.py` (EP1 and EP4)
judge every floating output of the fused kernels (`tests/cake_naive.py`):

- Oracle: the layer and its gradients in float64 on the kernel's own inputs (BF16 `x`, `dY` and weights, FP32
  scores). With MXFP8 routed experts the routed GEMMs read `x` and the weights through the MXFP8 recipe.
- Naive implementation: the standard PyTorch BF16 layer (autograd ops in BF16, one cuBLAS GEMM per expert and weight
  gradient, `d_scores = (dY * y_e).sum(-1)`); MXFP8 adds only the recipe's quantization points.
- Score gradient, in both: `y_e` is the routed expert's output as the backward holds it. With MXFP8 routed experts that
  is the SwiGLU of the gate and up the backward reads, by the backward's copy of the down weight (the forward output
  reads the forward's copy).
- Router weight before the down projection: in both, `h = s * a` is formed before the down projection (the naive
  implementation promotes the BF16 activation by the FP32 weight and casts once), `dY` reaches the down projection
  unscaled and `d_scores = (dh * a).sum(-1)`; the cases include the requester's inputs at H 5120 / I 2304 (unit-RMS
  `x`, `sqrtsoftplus` top-6 weights times 1.5 with exact zeros, gate and up past the limit 10 and exactly at it).
  `tests/experimental/report_mok_wbd.py` reports, as measurements, the errors against the issue's stage-rounded
  float64 reference on those inputs.
- Rule: over 32 random inputs per case, the kernel's error against the oracle in the L2 norm has a median and a
  largest value at most 1.005 times the naive implementation's. In the max norm, the kernel's error is at most 1.005
  times the naive error on at least 16 of the 32 inputs; 11 or fewer fail, and with 12 to 15 the case runs 96 more
  inputs and the max norm passes with at least 56 of the 128. The error is 0 wherever the naive error is 0.
  `x`'s E4M3 codes and scales are exact against the recipe.

Known gaps are judged in test items of their own. On an architecture where a gap's miss is accepted, the item carries a
strict expected-failure marker with caps (`tests/experimental/_mok_gaps.py`), one for each norm the miss is accepted in
there: the largest ratio of the kernel's median and largest error over the naive implementation's, over every recorded
run on every GPU model, rounded up to 4 decimals (at least 1.005). One item keeps caps per architecture, each from that
architecture's runs: the EP4 unequal-sources routed gate weight gradient. With inputs drawn independently of the SM
count, its three architectures give the same ratios (L2 1.03 / 1.03, max 1.30 / 1.20), so its three caps are equal. A
ratio above a cap fails the item, and so does a miss in a norm without a cap. A fix makes the item pass, and the strict
marker then fails the item until the marker is removed. On an architecture without caps the item is judged like any
output, and a miss fails it. Every other output of the same case stays judged in the case's main item. The outputs that
miss the rule, with their ratios (kernel error / naive error, median / largest over the inputs):

| Outputs | Items | `sm_100a`, `sm_103a`, `sm_107a` (B200, GB200, B300, GB300, VR200) |
|---|---|---|
| MXFP8 `d_scores`, unclamped SwiGLU | every unclamped MXFP8 case, single GPU | L2 1.03-1.04 / 1.01-1.03 |
| MXFP8 `d_scores`, clamped SwiGLU | every clamped MXFP8 case and the local-experts cases, single GPU | L2 1.15-1.16 / 1.12-1.18, max 1.09-1.21 / 1.05-1.18 |
| BF16 routed weight gradients of experts whose rows cross a macrobatch boundary | unequal sources 0 / 513 / 769 / 8193 at EP4 (gate, up, down); EP1 clamped cases (down) | L2 1.03-1.05 / 1.03-1.07, max 1.28-1.31 / 1.20-1.38 |
| same | single-GPU empty-experts cases with 768-row macrobatches; EP1 unclamped recompute (gate, up, down) | max 1.04-1.16 / 0.91-1.48 |
| MXFP8 routed gate weight gradient with FP32 weight-gradient accumulation | EP1 MXFP8 case with FP32 weight gradients (gate) | max 1.02 / 1.03 |

The five GPU models give the same ratios (single GPU, EP1 and EP4): B200, GB200, B300 and GB300 bit for bit; VR200
differs only in some MXFP8 `d_scores` L2 ratios, by less than 0.001 %.

The MXFP8 score gradient is formed as `sum(dh * hidden) / score` from the kernel's own intermediate values, and the BF16
routed weight gradient is rounded to BF16 per macrobatch. The EP1 MXFP8 routed gate weight gradient with FP32
weight-gradient accumulation misses only the max norm; its L2 norm meets the tier. Follow-up kernel changes remove these
three gaps.

The repository rules are in [`ARCHITECTURE.md`](../../ARCHITECTURE.md).
