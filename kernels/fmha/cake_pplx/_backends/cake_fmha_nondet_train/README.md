# Cake non-deterministic dense multi-head attention backward (head dim 64 / 128, SM100 / SM103 / SM107)

Backend of `cake_pplx.cake_fmha_nondet_train`: the first-order backward of
dense multi-head attention, `(dq, dk, dv)` from BF16 `q [B, Sq, H, D]`,
`k`, `v [B, Skv, H, D]`, the forward's BF16 `out [B, Sq, H, D]` and FP32
natural-log `lse [B, H, Sq]`, and BF16 `dout [B, Sq, H, D]`.  Every operand
and every gradient is contiguous in that layout (the layout of FlashAttention
and of cuDNN's SDPA with `[B, S, H, D]` tensors).  `B`, `H`, `Sq` and `Skv`
are run-time values: no shape list, no recompilation, no device-to-host read,
no synchronization.

The facade `cake_pplx.cake_fmha_nondet_train` is a stable API; this backend
package is experimental (no compatibility guarantee for its internal names).
SM100 (B200 / GB200), SM103 (B300 / GB300) and SM107 (R200) are the
supported architectures; another device raises `RuntimeError`.
Its libraries are loaded prebuilt from `aot/` when a builder compiled them
and JIT-built from the sources and flags `build.json` records otherwise.
The package reads no environment variable.

The family has a backward step only: no forward kernel and no autograd entry.
The caller brings `out` and `lse` from its own forward and calls the backward
from its own `torch.autograd.Function`.

## Opt-in

```python
from cake_pplx.cake_fmha_nondet_train import cake_fmha_nondet_train_backward

# q, out, dout: BF16 [B, Sq, H, D]; k, v: BF16 [B, Skv, H, D]; lse: FP32 [B, H, Sq] (natural log)
dq, dk, dv = cake_fmha_nondet_train_backward(
    q, k, v, out, lse, dout,
    softmax_scale=None,      # default D ** -0.5; must be the forward's
    causal=False,            # True: top-left mask, needs Sq == Skv
)
```

## Variants and architectures

| Head dim | Mask | `sm_100a` | `sm_103a` | `sm_107a` | Records (pre / main / post) |
|---|---|---|---|---|---|
| 128 | none | yes | yes | yes | `bwd_pre` / `bwd_main` / `bwd_post` |
| 128 | causal | yes | yes | yes | `bwd_pre` / `bwd_main_causal` / `bwd_post` |
| 64 | none | yes | yes | yes | `bwd_pre_hd64` / `bwd_main_hd64` / `bwd_post_hd64` |
| 64 | causal | yes | yes | yes | `bwd_pre_hd64` / `bwd_main_hd64_causal` / `bwd_post_hd64` |

Every variant takes any head count `H >= 1` (the same `H` for `q` and
`k` / `v`) and query and key lengths that are positive multiples of 128.
Without the mask `Sq` and `Skv` are independent; the causal mask needs
`Sq == Skv`.  The `build.json` minimum CUDA toolkit is 12.8 for every
`sm_100a` library, 12.9 for every `sm_103a` library and 13.5 for every
`sm_107a` library; an older local toolkit raises `RuntimeError` when the
library is built.

## Kernels

One backward call launches three kernels, in this order, on the current torch
stream of the operands' device:

| Stage | Kernel |
|---|---|
| `bwd_pre` | one CTA per 128 query rows of one (head, batch): `dPsum = rowsum(out * dout)` and `LSE_log2 = lse * log2(e)` (FP32 `[B*H, Sq]`), and the FP32 dQ accumulator cleared |
| `bwd_main` | one CTA per 128 keys of one (head, batch), walking the query blocks in ascending order (with the causal mask, from the diagonal block on): S, P, dP and dS per query block, dK and dV accumulated in FP32 in tensor memory and stored once as BF16, the dQ tiles added into the FP32 accumulator by bulk reductions |
| `bwd_post` | one CTA per 128 query rows of one (head, batch): `dq = bf16(acc * softmax_scale)` in the layout of `q` |

Grids are `(Sq / 128, H, B)` for `bwd_pre` and `bwd_post` and
`(Skv / 128, H, B)` for `bwd_main`.  Numerics: FP32 accumulation for the
scores, `dP`, `dK`, `dV`, `dQ` and the row sums; `P` and `dS` are rounded to
BF16 only as MMA operands.  Every launch is a plain stream-ordered launch:
nothing cooperative, no PDL.

Each call allocates its outputs and an FP32 workspace from PyTorch's caching
allocator: `dq_accum` (`B * H * Sq * D` elements), `lse_log2` and `dpsum`
(`B * H * Sq` elements each).

## Determinism

The dQ tiles of the key blocks of one head reach the FP32 accumulator in
scheduling order, so `dq` is not bitwise reproducible run to run.  `dk`,
`dv` and the preprocess outputs have one fixed order.  No deterministic
package of these kernels exists; the module name carries `nondet` and no entry
takes a `deterministic` keyword.  Under `torch.use_deterministic_algorithms(True)`
the call raises PyTorch's `RuntimeError` (`cake_fmha_nondet_train_backward does
not have a deterministic implementation ...`) before it allocates or launches
anything; with `warn_only=True` it warns once (`UserWarning`) and runs.  A call
with `B == 0` returns empty gradients and does not alert.

## Limitations

- Backward only; no forward kernel and no autograd entry.
- Head dimension 64 or 128 only.  No grouped-query or multi-query attention
  (`k` / `v` have `q`'s head count), no variable-length segments, no dropout,
  no sliding window, no softcap, no attention bias.
- `Sq` and `Skv` must be positive multiples of 128; the causal mask needs
  `Sq == Skv`.
- `B` and `H` are each at most 65535, and `B * H * max(Sq, Skv) * D` must be
  below 2^31 (the kernels use 32-bit offsets).
- Every operand must be contiguous; the BF16 operands must start on a 16-byte
  boundary and `lse` on a 4-byte boundary.  A violation is a `ValueError`
  before any launch.
- No build for any architecture other than SM100, SM103 and SM107.
