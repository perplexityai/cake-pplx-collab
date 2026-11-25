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
| GPU | `sm_100a`, `sm_103a` or `sm_107a` (compute capability 10.0, 10.3 or 10.7), both router-weight positions; the 512-expert top-10 group (`router_weight_gradient="direct"`, `shared_expert="scaled"`) on `sm_100a` and `sm_103a` only |
| CUDA toolkit (`nvcc`, JIT) | 12.8 for `sm_100a`, 12.9 for `sm_103a`, 13.5 for `sm_107a` |
| `torch` | with `torch.distributed` and symmetric memory (multicast) |
| Process group | Initialized, its CUDA device current; all ranks in one peer-accessible NVLink domain with multicast |
| EP sizes | 1, 4, 8, 16, 32, 64 (EP1 uses plain tensors) |
| Scheduler layouts `(EP, local experts)` | `(1, 4)`, `(4, 4)`, `(4, 64)`, `(8, 32)`, `(16, 16)`, `(32, 8)`, `(64, 4)`, `(8, 36)`, `(32, 9)`; top-k 2 or 8. `router_weight_position="before_down"`: `(4, 96)`, `(8, 48)`, `(16, 24)`, `(32, 12)`, `(64, 6)` at top-k 6, `(1, 4)` and `(4, 4)` at top-k 2. `(8, 64)` at top-k 10 with `router_weight_gradient="direct"` and `shared_expert="scaled"` |
| Schedule capacity | `schedule_capacity_multiplier` at least `ceil((min(topk, E) * T * EP + E * 255) / (T * topk)) / EP` holds every routing (`E` local experts, `T` the workspace's physical source rows, at least the largest rank's source rows or `source_capacity`); the same multiplier on every rank, computed from a row bound all ranks agree on (the bound decreases as `T` grows); no upper bound |

## Builds

`build.json` lists 197 builds of 67 kernels: 65 kernels for each of `sm_100a`, `sm_103a` and `sm_107a` (the
router-weight-before-down fused kernels, the top-k 6 epilogue and the scheduler kernels of the 384-expert layouts
included) and two more for `sm_107a` only. Every build is generated from one kernel source revision.

| Group | Kernels |
|---|---|
| BF16 fused | forward (plain, clamped; with or without the saved context), context recompute (plain, clamped), backward (plain, clamped; BF16 or FP32 weight gradients); router weight before the down projection: clamped forward (with or without the saved context), clamped recompute, clamped backward with FP32 weight gradients |
| MXFP8 fused | forward, context recompute (plain, clamped), backward (plain, clamped; BF16 or FP32 weight gradients); `sm_107a` only: the multi-macrobatch forward (plain, clamped) |
| Scheduler | count and rows per scheduler layout, pad per EP size |
| Layout | the expert layout, zero |
| Communication | barrier, gather |
| Epilogues | forward (weighted sum), top-k 2 and 8; backward (unweighted sum, also the router-weight-before-down forward's), top-k 2, 6 and 8 |

The locality-domain (`_domain`) builds of the MoK kernels, which split work over the two dies of an
`sm_107a` GPU, are not part of this package.

## MXFP8 weights

MXFP8 routed experts read caller-prequantized weights. `quantize_mok_mxfp8_weights(gate, up, down)` (BF16, gate
and up `[E, I, H]`, down `[E, H, I]`) returns `(forward_weights, backward_weights)`. Callers may pass their own
MXFP8 weights in this layout instead:

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

The bundled quantizer runs PyTorch operations on the BF16 weights (no Triton dependency). Requantize after every
weight update. Measured cost of `quantize_mok_mxfp8_weights` on GB300 (`sm_103a`), median of 10 calls after 3 warm-up
calls (CUDA events), with the peak memory it allocates beyond the inputs (its outputs included):

| Routed weights (E, I, H) | Time | Peak memory beyond the inputs |
|---|---|---|
| 64, 2048, 6144 | 51.6 ms | 5.64 GiB |
| 36, 1024, 4096 | 9.76 ms | 1.30 GiB |

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
