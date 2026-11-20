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

| Head dim | Mask | `sm_100a` | `sm_103a` | `sm_107a` | Records (pre / main / post, or one launch) |
|---|---|---|---|---|---|
| 128 | none | yes | yes | yes | `bwd_pre` / `bwd_main` / `bwd_post` |
| 128 | causal | yes | yes | yes | `bwd_pre` / `bwd_main_causal` / `bwd_post` |
| 64 | none | yes | yes | yes | `bwd_pre_hd64` / `bwd_main_hd64` / `bwd_post_hd64` |
| 64 | causal | yes | yes | yes | `bwd_pre_hd64` / `bwd_main_hd64_causal` / `bwd_post_hd64` |
| 128, pair route | none | yes | yes | yes | `bwd_pre` / `bwd_main_2cta` / `bwd_post_pair` |
| 128, pair route | causal | built, not routed | yes | yes | `bwd_pre` / `bwd_main_2cta_causal` / `bwd_post_pair` |
| 128, one-launch route | none | yes | yes | yes | `bwd_fused` |
| 64, one-launch route | none | yes | yes | yes | `bwd_fused_hd64` |

Every variant takes any head count `H >= 1` (the same `H` for `q` and
`k` / `v`) and query and key lengths that are positive multiples of 128 (the
pair route: key lengths that are multiples of 256).  Without the mask `Sq`
and `Skv` are independent; the causal mask needs `Sq == Skv`.

A call takes the one-launch route when `cake_jit.fused_record` gives it a
record: no mask, and the call inside the region `FUSED_REGION[(arch, D)]` in
`cake_jit.py`, which bounds the CTAs of the one-launch grid
(`B * H * (Sq / 128 + Skv / 128)`), the longer sequence and the key length
(`Skv` at most a bound, or at most a multiple of `Sq`).  A region is set per
architecture and head dimension from measurements against the three-kernel
routes; an entry `None` routes no call.  Every other head-dimension 128 call
whose key length is a multiple of 256 and not longer than its query
length takes the pair route where a region of `PAIR_REGIONS` in `cake_jit.py`
holds its `B * H` and `Skv` for its architecture and mask (`cake_jit.variant`
picks the variant of a call); every other call takes the records of the first
four rows.  On `sm_100a` the causal pair build ships and no call takes it.  The
`build.json` minimum CUDA toolkit is 12.8 for every `sm_100a` library, 12.9
for every `sm_103a` library and 13.5 for every `sm_107a` library; an older
local toolkit raises `RuntimeError` when the library is built.

The regions of the pair route (`PAIR_REGIONS`; `Sq >= Skv` in every region):

| architecture | no mask | causal |
|---|---|---|
| `sm_100a` | `B * H` 1: `Skv >= 4096`; `B * H <= 4`: `Skv >= 8192`; `B * H >= 128`: `Skv >= 8192` | none |
| `sm_103a` | `B * H <= 4`: `Skv >= 2048`; any `B * H`: `Skv >= 8192` | `B * H <= 4`: `Skv >= 2048` |
| `sm_107a` | `B * H <= 4`: `Skv >= 2048`; any `B * H`: `Skv >= 4096` | `B * H <= 4`: `Skv >= 2048`; `B * H <= 16`: `Skv >= 4096`; any `B * H`: `Skv >= 8192` |

## Kernels

A backward call on the one-launch route launches one kernel; every other call
launches three kernels, in this order, on the current torch stream of the
operands' device:

| Stage | Kernel |
|---|---|
| `bwd_pre` | one CTA per 128 query rows of one (head, batch): `dPsum = rowsum(out * dout)` and `LSE_log2 = lse * log2(e)` (FP32 `[B*H, Sq]`), and the FP32 dQ accumulator cleared |
| `bwd_main` | one CTA per 128 keys of one (head, batch), walking the query blocks in ascending order (with the causal mask, from the diagonal block on): S, P, dP and dS per query block, dK and dV accumulated in FP32 in tensor memory and stored once as BF16, the dQ tiles added into the FP32 accumulator by bulk reductions |
| `bwd_main`, pair route | one cluster of two CTAs per 256 keys of one (head, batch), each CTA owning 128 keys: the same work as above with the MMAs issued for the pair, the peer's dS of the CTA's 64 query rows copied in through distributed shared memory for the dQ product; the dQ tiles added in the pair-row order (each CTA adds its 64 rows of a 128-row tile as one block) |
| `bwd_post` | one CTA per 128 query rows of one (head, batch): `dq = bf16(acc * softmax_scale)` in the layout of `q` |
| `bwd_post`, pair route | the same, reading the accumulator in the pair-row order |
| `bwd_fused`, one-launch route | per (head, batch), one CTA per 128 query rows (a Q-owner, which writes the dq of those rows) and one CTA per 128 keys (a KV-owner, which writes dK and dV of those keys); the kernel makes `LSE_log2 = lse * log2(e)` and `dPsum = rowsum(out * dout)` itself and stores each output element once |

Grids are `(Sq / 128, H, B)` for `bwd_pre` and `bwd_post`,
`(Skv / 128, H, B)` for `bwd_main` (on the pair route in clusters `(2, 1, 1)`:
`Skv / 128` is even) and `(Sq / 128 + Skv / 128, H, B)` for `bwd_fused`
(512 threads per CTA).  Numerics: FP32 accumulation for the
scores, `dP`, `dK`, `dV`, `dQ` and the row sums; `P` and `dS` are rounded to
BF16 only as MMA operands.  Every launch is a plain stream-ordered launch:
nothing cooperative, no PDL.

On the three-kernel routes the three launches run in one host call: each
variant has a host sequence library (`SEQUENCES` in `cake_jit.py`) whose
`build.json` entries compile the three kernel units with a host unit that
launches them in order.  Each such call allocates its outputs and an FP32
workspace from PyTorch's caching allocator: `dq_accum` (`B * H * Sq * D`
elements), `lse_log2` and `dpsum` (`B * H * Sq` elements each).  The host call
takes the workspace through the tvm-ffi environment allocator, on the current
stream.  A call on the one-launch route allocates its outputs only and needs
no workspace.

## Determinism

On the three-kernel routes the dQ tiles of the key blocks of one head reach the
FP32 accumulator in scheduling order, so `dq` may differ in its last bits run
to run.  `dk`, `dv` and the preprocess outputs have one fixed order.  On the
one-launch route `dq`, `dk` and `dv` each have one fixed order; the package
stays non-deterministic all the same.  No deterministic
package of these kernels exists; the module name carries `nondet` and no entry
takes a `deterministic` keyword.  Under `torch.use_deterministic_algorithms(True)`
the call raises PyTorch's `RuntimeError` (`cake_fmha_nondet_train_backward does
not have a deterministic implementation ...`) before it allocates or launches
anything, on every route; with `warn_only=True` it warns once (`UserWarning`)
and runs.  A call with `B == 0` returns empty gradients and does not alert.

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

## Tests

The tests are in `tests/experimental/` of the family: the backend
(`test_cake_fmha_nondet_train.py`), the public module
(`test_cake_fmha_nondet_train_api.py`) and the names
(`test_cake_fmha_nondet_train_names.py`).  Run them against the installed
distribution as [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md)
describes; without a supported device the device tests skip.

Thirty-four tests observe the kernels that one backward call launches through
`torch.profiler`: `test_the_backward_launches_the_records_of_its_variant`
and `test_the_backward_without_run_launches_the_same_records`.  The profiler
records kernels only with a CUPTI that supports the device.  On `sm_107a`,
CUPTI 13.0.85 (the one torch 2.14.1+cu130 installs) records no kernel
(`CUPTI_ERROR_INVALID_DEVICE`); CUPTI 13.5.25 records them.  Each of
these tests first profiles a plain torch operation on the device.  When the
profiler records no kernel for it, the test runs its other checks without
the profiler, then skips the kernel observation with the reason
`torch.profiler records no CUDA kernel on this device (CUPTI); install a
CUPTI that supports it`.

To observe the kernels on `sm_107a`, install CUPTI 13.5.25 into a separate
directory and preload it:

```bash
pip install --target DIR nvidia-cuda-cupti==13.5.25
LD_PRELOAD=DIR/nvidia/cu13/lib/libcupti.so.13 \
  python -m pytest -p no:cacheprovider tests/experimental/test_cake_fmha_nondet_train.py
```

Installing it into the torch environment instead replaces the CUPTI that
torch pins.
