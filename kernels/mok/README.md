# cake-pplx-mok

Cake MoK (Mixture of Kittens) training kernels: one expert-parallel MoE layer with the routed-token
scheduler, the fused forward, the context-only recompute and the fused backward. Public module:

- `cake_pplx.cake_mok_det_train`: `prepare_mok_bf16`, `create_mok_bf16_workspace`,
  `quantize_mok_mxfp8_weights`, and the communication-SM tables and resolvers (`COMM_SMS`,
  `resolve_comm_sms`, ...).

Install this family alone, pinned to a commit:

```bash
pip install "cake-pplx-mok @ git+https://github.com/perplexityai/cake-pplx-collab@<sha>#subdirectory=kernels/mok"
```

Routed experts run in BF16, or in MXFP8 with caller-prequantized weights; the shared expert runs in BF16.
Plain and clamped SwiGLU, BF16 or FP32 weight-gradient accumulation, and unequal (including zero) source
counts per rank are supported. The backend README
([`cake_pplx/_backends/cake_mok_bf16/README.md`](cake_pplx/_backends/cake_mok_bf16/README.md)) has the
full API, contract, memory rules and measured communication-SM tables.

## Repeatability

The module is deterministic. Each output repeats bit for bit from run to run: `y`, the forward context,
the recomputed context (identical to the forward's), `dx`, `dscores` and the six weight gradients (or the
six FP32 accumulators), for BF16 and MXFP8 and for plain and clamped SwiGLU. The scope is the same inputs,
weights, `MoKConfig` values other than the communication-SM split, precision, `swiglu_limit`,
`fp32_wgrad`, EP group, package build and GPU model. No bit depends on the communication-SM split, on
CUDA Graph replay versus eager launches, or on other kernels resident on other streams. Every reduction
runs in a fixed order; atomics act on integers only.

## Requirements

| Item | Requirement |
|---|---|
| GPU | `sm_100a`, `sm_103a` or `sm_107a` (compute capability 10.0, 10.3 or 10.7) |
| CUDA toolkit (`nvcc`, JIT) | 12.8 for `sm_100a`, 12.9 for `sm_103a`, 13.5 for `sm_107a` |
| `torch` | with `torch.distributed` and symmetric memory (multicast) |
| Process group | Initialized, its CUDA device current; all ranks in one peer-accessible NVLink domain with multicast |
| EP sizes | 1, 4, 8, 16, 32, 64 (EP1 uses plain tensors) |
| Scheduler layouts `(EP, local experts)` | `(1, 4)`, `(4, 4)`, `(4, 64)`, `(8, 32)`, `(16, 16)`, `(32, 8)`, `(64, 4)`, `(8, 36)`, `(32, 9)`; top-k 2 or 8 |

## Builds

`build.json` lists 152 builds of 52 kernels: 50 kernels for each of `sm_100a`, `sm_103a` and `sm_107a`, and two
more for `sm_107a` only. Every build is generated from one kernel source revision.

| Group | Kernels |
|---|---|
| BF16 fused | forward (plain, clamped; with or without the saved context), context recompute (plain, clamped), backward (plain, clamped; BF16 or FP32 weight gradients) |
| MXFP8 fused | forward, context recompute (plain, clamped), backward (plain, clamped; BF16 or FP32 weight gradients); `sm_107a` only: the multi-macrobatch forward (plain, clamped) |
| Scheduler | count and rows per scheduler layout, pad per EP size |
| Layout | the expert layout, zero |
| Communication | barrier, gather |
| Epilogues | forward and backward, top-k 2 and 8 |

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
- Rule: over 32 random inputs per case, the kernel's error against the oracle in the L2 norm has a median and a
  largest value at most 1.005 times the naive implementation's. In the max norm, the kernel's error is at most 1.005
  times the naive error on at least 16 of the 32 inputs; 11 or fewer fail, and with 12 to 15 the case runs 96 more
  inputs and the max norm passes with at least 56 of the 128. The error is 0 wherever the naive error is 0.
  `x`'s E4M3 codes and scales are exact against the recipe.

Known gaps are judged in test items of their own. On an architecture where a gap's miss is accepted, the item
carries a strict expected-failure marker with caps (`tests/experimental/_mok_gaps.py`), one for each norm the miss is
accepted in: the largest ratio measured there of the kernel's median and largest error over the naive
implementation's, rounded up to 4 decimals. A ratio above a cap fails the item, and so does a miss in a norm without a
cap. A fix makes the item pass, and the strict marker then fails the item until the marker is removed. On an
architecture without caps the item is judged like any output, and a miss fails it.
Every other output of the same case stays judged in the case's main item. The outputs that miss the rule, with their
ratios (kernel error / naive error, median / largest over the inputs):

| Outputs | Items | `sm_100a`, `sm_103a` (GB200, GB300) | `sm_107a` (VR200) |
|---|---|---|---|
| MXFP8 `d_scores` | every MXFP8 case, single GPU | L2 1.46-1.83 / 1.44-1.86, max 1.36-1.79 / 1.21-1.88 | L2 1.44-1.84 / 1.46-1.84, max 1.39-1.89 / 1.26-2.28 |
| MXFP8 `d_scores` | every MXFP8 case, EP1 and EP4 | L2 1.71-1.73 / 1.71-1.75, max 1.65-1.80 / 1.51-2.09 (EP4: GB300) | EP1: L2 1.72 / 1.70-1.72, max 1.55-1.75 / 1.45-1.83 |
| BF16 routed weight gradients of experts whose rows cross a macrobatch boundary | unequal sources 0 / 513 / 769 / 8193 at EP4 (gate, up, down); EP1 clamped cases (down) | L2 1.03 / 1.03-1.07, max 1.22-1.33 / 1.16-1.42 (unequal sources: GB300) | EP1: L2 1.03-1.04 / 1.07, max 1.23-1.29 / 1.16-1.28 |
| same | single-GPU empty-experts cases with 768-row macrobatches; EP1 plain recompute (gate, up, down) | max 1.05-1.14 / 0.96-1.24 | max 1.05-1.14 / 0.93-1.24 |

GB200 and GB300 give the same ratios (single GPU and EP1).

The MXFP8 score gradient is formed as `sum(dh * hidden) / score` from the kernel's own intermediate values, and the
BF16 routed weight gradient is rounded to BF16 per macrobatch; follow-up kernel changes remove both gaps.

The repository rules are in [`ARCHITECTURE.md`](../../ARCHITECTURE.md).
