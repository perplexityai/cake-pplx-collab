# Cake deterministic Engram context gate backend (four streams of width 5120, SM100 / SM103 / SM107)

Backend of `cake_pplx.cake_engram_gate_det_train`: the forward, the backward and the autograd entries of the
four-stream Engram context gate of width 5120. Per token `t` and stream `s`, with every quantity in FP32 and each BF16
output rounded once, at its store:

```text
h = x[t,s]   k = key[t,s]   v = value[t]   w = FP32(q_weight[s]) * FP32(k_weight[s])
r_h = rsqrt(mean(h^2) + eps)   r_k = rsqrt(mean(k^2) + eps)   P = sum_d w_d h_d k_d   dot = P r_h r_k / sqrt(5120)
gate = sigmoid(copysign(sqrt(max(|dot|, clamp_value)), dot))   (0 on a masked token)
out[t,s] = bf16(h + gate v)
```

The facade `cake_pplx.cake_engram_gate_det_train` is a stable API; this backend package is experimental (no
compatibility guarantee for its internal names). SM100 (B200 / GB200), SM103 (B300 / GB300) and SM107 (R200) are the
supported architectures; another device raises `RuntimeError`. Its libraries are loaded prebuilt from `aot/` when a
builder compiled them and JIT-built from the sources and flags `build.json` records otherwise. The package reads no
environment variable.

## Opt-in

```python
from cake_pplx.cake_engram_gate_det_train import cake_engram_gate_det, cake_engram_gate_det_kv

# key [T, 4, 5120] and value [T, 5120]: contiguous, or views into one packed kv [T, 25600]
out = cake_engram_gate_det(x, key, value, q_weight, k_weight, token_mask, eps=1e-20, clamp_value=1e-6)

# the packed form: the backward writes one dkv [T, 25600] (no split, no concatenation copy)
out = cake_engram_gate_det_kv(x, kv, q_weight, k_weight, token_mask, eps=1e-20, clamp_value=1e-6)
out.backward(dout)

# opt-in: sum the dot product P in one FP32 word (less accurate where the products nearly cancel; see Accuracy)
out = cake_engram_gate_det_kv(x, kv, q_weight, k_weight, token_mask, dot_sum="fp32")
```

`cake_engram_gate_det_train_forward` / `cake_engram_gate_det_train_backward` are the two halves without autograd;
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
| `fwd` | `fwd_dot2`, `fwd_f32` | `(4 * clusters, 1, 1)`, clusters of 4 CTAs | per token: the four CTAs of a cluster each sum 1280 columns of every stream, exchange the partial sums, and write `out` |
| `bwd` | `bwd_dot2`, `bwd_f32` | `(4 * clusters, 1, 1)`, clusters of 4 CTAs | per token: recompute the sums (and `sum_d g v`), write `dx`, `dkey`, `dvalue`; sum `c h k` over the cluster's tokens |
| `dw_reduce` | `dw_reduce` | over the `4 * 5120` weight-gradient columns | sum the partial rows, multiply by `k_weight` / `q_weight` |

`clusters = min(T, resident clusters)` (`cake_jit.clusters`); a cluster walks the tokens `c, c + clusters, ...`. The
resident clusters are the four-CTA clusters the device holds at once for the record's CTAs per SM
(`cake_jit.CTAS_PER_SM`: four for `fwd`, three for `bwd`). A cluster stays in one GPC, so a device holds fewer of them
than `CTAs per SM * SM count / 4`. `cake_jit.RESIDENT_CLUSTERS` holds the counts `cudaOccupancyMaxActiveClusters`
gave for this family's programs at each count of CTAs per SM:

| Device | Architecture, SMs | 4 CTAs per SM (`fwd`) | 3 CTAs per SM (`bwd`) |
| --- | --- | --- | --- |
| B200 | `sm_100a`, 148 | 142 | not measured |
| GB300 | `sm_103a`, 152 | 148 | 110 |
| VR200 | `sm_107a`, 212 | 184 | 138 |

Another device or CTA count takes `0.85 * CTAs per SM * SM count / 4` clusters (`UNMEASURED_CLUSTER_FRACTION`), less
than any measured device holds. Any grid is correct. A larger grid than the device holds is slow: a cluster that is not
resident starts only after a resident one has walked all of its tokens. Every launch is a plain stream-ordered launch
on the current torch stream: nothing cooperative.

The `_dot2` and `_f32` records differ only in the sum of `P`, which every entry selects with `dot_sum`:

- `dot_sum="compensated"` (the default) launches `_dot2`: a compensated FP32 dot product (each product split by an
  FMA, each running sum by an error-free two-sum, the error words added once at the end).
- `dot_sum="fp32"` (opt-in) launches `_f32`: one FP32 word.

The autograd entries save the choice with the inputs: the backward (and a checkpoint's recomputed forward) sums `P`
as the forward did.

## Accuracy

The acceptance tests compare each output with an FP64 reference: the median and the largest relative L2 error over
25 or more draws must be at most twice those of an FP32 evaluation of the same formula. The draws are random, or
built so that the dot products nearly cancel (`|dot|` from 2e-6 to 2e-5); on those rows the gate's derivative
`1 / (2 sqrt|dot|)` magnifies any error of `P`. The default sum meets every bound. The `"fp32"` sum meets the bounds
on `out`, `dvalue` and the random `dx` and `dkey`, and misses the others. Measured on GB300 (`T = 16`, 25 draws
each), relative L2 error, median / largest:

| Output, draws | `"compensated"` | `"fp32"` | Bound |
| --- | --- | --- | --- |
| `dx`, near-cancelling, per stream | 1.7e-3 / 1.7e-3 | 1.0e-2 / 2.6e-2 | 3.3e-3 / 3.4e-3 |
| `dkey`, near-cancelling, per stream | 1.7e-3 / 1.7e-3 | 1.0e-2 / 2.6e-2 | 3.3e-3 / 3.4e-3 |
| `dq_weight`, `dk_weight`, near-cancelling, per stream | 1.4e-7 / 2.3e-7 | 9.8e-3 / 2.6e-2 | 3.2e-7 / 4.7e-7 |
| `dq_weight`, `dk_weight`, random | 1.4e-7 / 3.4e-7 | 8.0e-7 / 1.1e-4 | 1.8e-6 / 7.7e-5 |

## Workspace and retained state

- Retained for the backward (autograd): the inputs only. The backward recomputes the norms and the dot product; no
  statistics are saved.
- Forward workspace: the FP32 weight product (80 KiB), plus an FP32 copy of a BF16 `q_weight` (80 KiB).
- Backward workspace: the weight product, FP32 copies of BF16 weights (80 KiB each) and one FP32 `[4, 5120]` row of
  weight-gradient partial sums per cluster of the backward grid (80 KiB each);
  `cake_engram_gate_det_train_workspace_bytes(T, device, dot_sum=...)`.

## Determinism

Every sum has one fixed order: each cluster writes its own row of weight-gradient partial sums, and `dw_reduce` adds
the rows in one order. Every output is bitwise reproducible run to run on one device (the cluster count follows the
token count, the device and `dot_sum`). Nothing reads `torch.use_deterministic_algorithms`.

## Limitations

- Width 5120 and four streams only.
- `x`, `dout` and `dx` must be contiguous; `key`, `value` and their gradients need a unit column stride.
- No double backward.
