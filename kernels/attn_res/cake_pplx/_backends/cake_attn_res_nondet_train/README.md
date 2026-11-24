# Cake non-deterministic attention-residual backend (values of width 7168, SM100 / SM103 / SM107)

Backend of `cake_pplx.cake_attn_res_nondet_train`: the forward, the backward and the autograd entry of the attention
residual (AttnRes), a softmax-weighted sum of `n` BF16 values of width 7168, `1 <= n <= 9`. Per token `t`, with every
quantity in FP32 and each BF16 output rounded once, at its store:

```text
r_j = 1 / sqrt(mean_c v_j[c]^2 + eps)   d_j = sum_c v_j[c] w[c]   p = softmax_j(r_j d_j)   out = bf16(sum_j p_j v_j)
dp_j = sum_c dout[c] v_j[c]   ds_j = p_j (dp_j - sum_i p_i dp_i)
dv_j = bf16(p_j dout + ds_j r_j w - (ds_j d_j r_j^3 / 7168) v_j)   dw = sum over t and j of ds_j r_j v_j   (FP32)
```

The facade `cake_pplx.cake_attn_res_nondet_train` is a stable API; this backend package is experimental (no
compatibility guarantee for its internal names). SM100 (B200 / GB200), SM103 (B300 / GB300) and SM107 (R200) are the
supported architectures; another device raises `RuntimeError`. Its libraries are loaded prebuilt from `aot/` when a
builder compiled them and JIT-built from the sources and flags `build.json` records otherwise. The package reads no
environment variable.

## Opt-in

```python
from cake_pplx.cake_attn_res_nondet_train import cake_attn_res_nondet

out = cake_attn_res_nondet([v0, v1, v2], w, eps=1e-5)   # n BF16 [T, 7168] tensors, FP32 w [7168]
out.backward(dout)                                       # v0.grad, v1.grad, v2.grad (BF16), w.grad (FP32)
```

`cake_attn_res_nondet_train_forward` returns `(out, p, r, d)`; `cake_attn_res_nondet_train_backward(dout, values, w,
p, r, d)` returns `(dv, dw)` and writes caller-owned `dv` tensors when they are given.

## Operands

| Operand | Shape | Dtype | Layout |
| --- | --- | --- | --- |
| `values[j]`, `out`, `dout`, `dv[j]` | `[T, 7168]` | BF16 | contiguous, 16-byte aligned |
| `w`, `dw` | `[7168]` | FP32 | any (`w`); contiguous (`dw`) |
| `p`, `r`, `d` | `[T, n]` | FP32 | contiguous (row-major) |

`eps` is a run-time scalar (finite, non-negative); `T >= 0` and `n` are run-time values (no recompilation).

## Kernels

| Stage | Record | Grid | Work |
| --- | --- | --- | --- |
| `fwd` | `fwd` | `(cluster CTAs * groups, 1, 1)`, persistent | per token: `r`, `d`, the softmax `p` and `out`; writes `p`, `r`, `d` |
| `bwd` | `bwd` | `(cluster CTAs * groups, 1, 1)`, persistent | per token: every `dv_j`; adds the token's `dw` terms into the zeroed `dw` with `red.global.add.f32` |

`groups = min(T, resident groups)`, the groups the device holds at once (`cake_jit.resident_groups`). The host zeroes
`dw` before the backward launch. The kernels take nine value pointers; the host passes the first value in the slots
past `n`, which the kernels never read.

## Determinism

`out`, `p`, `r`, `d` and every `dv_j` have one fixed order. `dw` adds the tokens' terms in arrival order: it promises
no bits run to run. Nothing consults `torch.use_deterministic_algorithms`.

## Memory

Retained for the backward: references to the inputs and `p`, `r`, `d` (`12 * T * n` bytes). Backward workspace: none.
