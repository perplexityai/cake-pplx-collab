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

The locality-domain (`_domain`) builds of the MoK kernels, which split work over the two dies of an
`sm_107a` GPU, are not shipped in this package.

The repository rules are in [`ARCHITECTURE.md`](../../ARCHITECTURE.md).
