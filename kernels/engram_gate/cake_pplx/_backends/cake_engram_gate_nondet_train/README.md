# Cake non-deterministic Engram context gate backend (four streams of width 5120, SM100 / SM103 / SM107)

Backend of `cake_pplx.cake_engram_gate_nondet_train`: the forward, the backward and the autograd entries of the
four-stream Engram context gate of width 5120. Per token `t` and stream `s`, with every quantity in FP32 and each BF16
output rounded once, at its store:

```text
h = x[t,s]   k = key[t,s]   v = value[t]   w = FP32(q_weight[s]) * FP32(k_weight[s])
r_h = rsqrt(mean(h^2) + eps)   r_k = rsqrt(mean(k^2) + eps)   P = sum_d w_d h_d k_d   dot = P r_h r_k / sqrt(5120)
gate = sigmoid(copysign(sqrt(max(|dot|, clamp_value)), dot))   (0 on a masked token)
out[t,s] = bf16(h + gate v)
```

The facade `cake_pplx.cake_engram_gate_nondet_train` is a stable API; this backend package is experimental (no
compatibility guarantee for its internal names). SM100 (B200 / GB200), SM103 (B300 / GB300) and SM107 (R200) are the
supported architectures; another device raises `RuntimeError`. Its libraries are loaded prebuilt from `aot/` when a
builder compiled them and JIT-built from the sources and flags `build.json` records otherwise. The package reads no
environment variable.

## Opt-in

```python
from cake_pplx.cake_engram_gate_nondet_train import cake_engram_gate_nondet, cake_engram_gate_nondet_kv

# key [T, 4, 5120] and value [T, 5120]: contiguous, or views into one packed kv [T, 25600]
out = cake_engram_gate_nondet(x, key, value, q_weight, k_weight, token_mask, eps=1e-20, clamp_value=1e-6)

# the packed form: the backward writes one dkv [T, 25600] (no split, no concatenation copy)
out = cake_engram_gate_nondet_kv(x, kv, q_weight, k_weight, token_mask, eps=1e-20, clamp_value=1e-6)
out.backward(dout)
```

`cake_engram_gate_nondet_train_forward` / `cake_engram_gate_nondet_train_backward` are the two halves without autograd;
the backward takes the forward's inputs (not its output) and writes caller-owned `dx`, `dkey`, `dvalue` (for example
views into one `[T, 25600]` gradient buffer) when they are given. It returns FP32 `dq_weight` and `dk_weight`; the
autograd entries return them in each parameter's dtype.

## Operands

| Operand | Shape | Dtype | Layout |
| --- | --- | --- | --- |
| `x`, `out`, `dout`, `dx` | `[T, 4, 5120]` | BF16 | contiguous, 16-byte aligned |
| `key`, `dkey` | `[T, 4, 5120]` | BF16 | unit column stride, stream stride 5120, token stride a multiple of 8, 16-byte aligned |
| `value`, `dvalue` | `[T, 5120]` | BF16 | unit column stride, token stride a multiple of 8, 16-byte aligned |
| `kv`, `dkv` | `[T, 25600]` | BF16 | unit column stride, token stride a multiple of 8, 16-byte aligned |
| `q_weight`, `k_weight` | `[4, 5120]` | BF16 or FP32 | any |
| `token_mask` | `[T]` or None | bool | any |
| `dq_weight`, `dk_weight` | `[4, 5120]` | FP32 (explicit backward) or the parameter's dtype (autograd) | contiguous |

`eps` and `clamp_value` are run-time scalars (finite, non-negative); `T >= 0` is a run-time value (no recompilation).
Output rows must not overlap.

## Kernels

| Stage | Records | Grid | Work |
| --- | --- | --- | --- |
| `fwd` | `fwd_f32`, `fwd_dot2` | `(4 * clusters, 1, 1)`, clusters of 4 CTAs | per token: the four CTAs of a cluster each sum 1280 columns of every stream, exchange the partial sums, and write `out` |
| `bwd` | `bwd_f32`, `bwd_dot2` | `(4 * clusters, 1, 1)`, clusters of 4 CTAs | per token: recompute the sums (and `sum_d g v`), write `dx`, `dkey`, `dvalue`; sum `c h k` over the cluster's tokens |
| `dw_reduce` | `dw_reduce` | over the `4 * 5120` weight-gradient columns | sum the partial rows, multiply by `k_weight` / `q_weight` |

`clusters = min(T, CTAs per SM * SM count / 4)` (`cake_jit.clusters`); a cluster walks the tokens `c, c + clusters, ...`.
The `_f32` and `_dot2` records differ only in the summation of `P`: one FP32 word, or a compensated FP32 dot product
(products split by an FMA, running sums by an error-free two-sum, the error words added once at the end). The backend
launches one choice, `cake_backend._DOT_ACC`. Every launch is a plain stream-ordered launch on the current torch
stream: nothing cooperative.

## Workspace and retained state

- Retained for the backward (autograd): the inputs only. The backward recomputes the norms and the dot product; no
  statistics are saved.
- Forward workspace: the FP32 weight product (80 KiB), plus an FP32 copy of a BF16 `q_weight` (80 KiB).
- Backward workspace: the weight product, FP32 copies of BF16 weights (80 KiB each) and one FP32 `[4, 5120]`
  weight-gradient accumulator (80 KiB, zeroed per call); `cake_engram_gate_nondet_train_workspace_bytes(T, device)`.

## Determinism

The clusters add their weight-gradient partial sums into one FP32 accumulator with float atomics, in arrival order:
`dq_weight` and `dk_weight` may differ in their last bits run to run. `out`, `dx`, `dkey` and `dvalue` have one fixed
order. Under `torch.use_deterministic_algorithms(True)` every backward call that launches raises PyTorch's
`RuntimeError`, naming the entry of `cake_pplx.cake_engram_gate_det_train` of the same role; with `warn_only=True` it
warns once and runs. The forward never alerts; a backward over no token neither launches nor alerts.

## Limitations

- Width 5120 and four streams only.
- `x`, `dout` and `dx` must be contiguous; `key`, `value` and their gradients need a unit column stride.
- No double backward.
