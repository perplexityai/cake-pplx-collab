# Cake deterministic attention-residual backend (values of width 7168, SM100 / SM103 / SM107)

Backend of `cake_pplx.cake_attn_res_det_train`: the forward, the backward and the autograd entry of the attention
residual (AttnRes), a softmax-weighted sum of `n` BF16 values of width 7168, `1 <= n <= 9`. Per token `t`, with every
quantity in FP32 and each BF16 output rounded once, at its store:

```text
r_j = 1 / sqrt(mean_c v_j[c]^2 + eps)   d_j = sum_c v_j[c] w[c]   p = softmax_j(r_j d_j)   out = bf16(sum_j p_j v_j)
dp_j = sum_c dout[c] v_j[c]   ds_j = p_j (dp_j - sum_i p_i dp_i)
dv_j = bf16(p_j dout + ds_j r_j w - (ds_j d_j r_j^3 / 7168) v_j)   dw = sum over t and j of ds_j r_j v_j   (FP32)
```

The backward evaluates `ds_j = p_j ((dp_j - dp_c) - sum_i p_i (dp_i - dp_c))`, `c` the value of the largest `p_j` (the
first on ties): the same value, without the cancellation of `dp_c - sum_i p_i dp_i` when `p_c` rounds to 1.

The facade `cake_pplx.cake_attn_res_det_train` is a stable API; this backend package is experimental (no
compatibility guarantee for its internal names). SM100 (B200 / GB200), SM103 (B300 / GB300) and SM107 (Rubin) are the
supported architectures; another device raises `RuntimeError`. The `sm_107a` libraries pass the package tests and the
compute-sanitizer runs on VR200 and R200. Its libraries are loaded prebuilt from `aot/` when a builder
compiled them and JIT-built from the sources and flags `build.json` records otherwise. Kernel selection and launches
read no environment variable; the JIT build shim (`_cake_pplx_jit.py`) reads `CUDA_HOME` / `CUDA_PATH`, `CXX` and the
`CAKE_PPLX_*` build settings of the root README.

## Opt-in

```python
from cake_pplx.cake_attn_res_det_train import cake_attn_res_det

out = cake_attn_res_det([v0, v1, v2], w, eps=1e-5)   # n BF16 [T, 7168] tensors, FP32 w [7168]
out.backward(dout)                                       # v0.grad, v1.grad, v2.grad (BF16), w.grad (FP32)
```

`cake_attn_res_det_train_forward` returns `(out, p, r, d)`; `cake_attn_res_det_train_backward(dout, values, w,
p, r, d)` returns `(dv, dw)` and writes caller-owned `dv` tensors when they are given. A caller-owned `out` or `dv[j]`
must share no byte with an operand or another output (`ValueError`).

## Operands

| Operand | Shape | Dtype | Layout |
| --- | --- | --- | --- |
| `values[j]`, `out`, `dout`, `dv[j]` | `[T, 7168]` | BF16 | contiguous, 16-byte aligned |
| `w`, `dw` | `[7168]` | FP32 | any (`w`); contiguous (`dw`) |
| `p`, `r`, `d` | `[T, n]` | FP32 | contiguous (row-major) |

`eps` is a run-time Python float or int (finite and at least 2**-84, about 5.2e-26; a bool or a tensor is a `TypeError`); `T >= 0` and `n` are run-time values. `fwd` and `bwd` ship one program per value bound `nmax` from 2 to 9; a call of `n` values runs the program of the smallest bound at least `n` (`cake_jit.nmax_for`). The libraries are a fixed set: no shape compiles a new one.

## Kernels

| Stage | Record | Grid | Work |
| --- | --- | --- | --- |
| `fwd` | `fwd_n2` .. `fwd_n9` | `(cluster CTAs * groups, 1, 1)`, persistent | per token: `r`, `d`, the softmax `p` and `out`; writes `p`, `r`, `d`. `fwd_n2` .. `fwd_n4`: each CTA serves whole tokens (no exchange); `fwd_n5` .. `fwd_n9`: the four CTAs of a cluster split each token's columns |
| `bwd` | `bwd_n2` .. `bwd_n9` | `(cluster CTAs * groups, 1, 1)`, persistent | per token: every `dv_j`; each group writes its own FP32 row of `dw` partial sums into `dw_rows [groups, 7168]` |
| `dw_reduce` | `dw_reduce` | `(7168 / columns per CTA, 1, 1)` | `dw` = the rows of `dw_rows` summed in one fixed order |

`T = 0` launches no Cake kernel: the backward returns a zero `dw` (a `torch.zeros` fill).

`groups = min(T, resident groups)`, the groups the device holds at once (`cake_jit.resident_groups`: counts measured per launch bound on B200, GB300, VR200 and R200; GB200's counts at 3 and 4 CTAs per SM come from another family's measurements, and GB200 has no count at 2 yet, so there `fwd_n2` .. `fwd_n4` and `bwd_n7` .. `bwd_n9` take the fallback; on another device a fallback under every measured count). The kernels take nine value pointers; the host passes the first value in the slots
past `n`, which the kernels never read.

## Determinism

Every sum has one fixed order: every output repeats bit for bit for identical inputs on one device (one grid per
token count and device). Nothing consults `torch.use_deterministic_algorithms`.

## Memory

Retained for the backward: references to the inputs and `p`, `r`, `d` (`12 * T * n` bytes). Backward workspace: the
`dw_rows` partial rows, 28 KiB each, one per launch group (`cake_attn_res_det_train_workspace_bytes`). The groups follow
`T`, `n` and the device: on B200 at `T >= 142`, 142 rows for `n <= 5`, 104 for `n = 6` and 71 for `n >= 7`.
