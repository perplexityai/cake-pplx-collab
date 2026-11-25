# Cake non-deterministic attention-residual backend (values of width 7168, SM100 / SM103 / SM107)

Backend of `cake_pplx.cake_attn_res_nondet_train`: the forward, the backward and the autograd entry of the attention
residual (AttnRes), a softmax-weighted sum of `n` BF16 values of width 7168, `1 <= n <= 9`. Per token `t`, with every
quantity in FP32 and each BF16 output rounded once, at its store:

```text
r_j = 1 / sqrt(mean_c v_j[c]^2 + eps)   d_j = sum_c v_j[c] w[c]   p = softmax_j(r_j d_j)   out = bf16(sum_j p_j v_j)
dp_j = sum_c dout[c] v_j[c]   ds_j = p_j (dp_j - sum_i p_i dp_i)
dv_j = bf16(p_j dout + ds_j r_j w - (ds_j d_j r_j^3 / 7168) v_j)   dw = sum over t and j of ds_j r_j v_j   (FP32)
```

The backward evaluates `ds_j = p_j ((dp_j - dp_c) - sum_i p_i (dp_i - dp_c))`, `c` the value of the largest `p_j` (the
first on ties): the same value, without the cancellation of `dp_c - sum_i p_i dp_i` when `p_c` rounds to 1.

The facade `cake_pplx.cake_attn_res_nondet_train` is a stable API; this backend package is experimental (no
compatibility guarantee for its internal names). SM100 (B200 / GB200), SM103 (B300 / GB300) and SM107 (Rubin) are the
supported architectures; another device raises `RuntimeError`. The `sm_107a` libraries pass the package tests and the
compute-sanitizer runs on VR200 and R200. Its libraries are loaded prebuilt from `aot/` when a builder
compiled them and JIT-built from the sources and flags `build.json` records otherwise. Kernel selection and launches
read no environment variable; the JIT build shim (`_cake_pplx_jit.py`) reads `CUDA_HOME` / `CUDA_PATH`, `CXX` and the
`CAKE_PPLX_*` build settings of the root README.

## Opt-in

```python
from cake_pplx.cake_attn_res_nondet_train import cake_attn_res_nondet

out = cake_attn_res_nondet([v0, v1, v2], w, eps=1e-5)   # n BF16 [T, 7168] tensors, FP32 w [7168]
out.backward(dout)                                       # v0.grad, v1.grad, v2.grad (BF16), w.grad (FP32)
```

`cake_attn_res_nondet_train_forward` returns `(out, p, r, d)`; `cake_attn_res_nondet_train_backward(dout, values, w,
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
| `bwd` | `bwd_n2` .. `bwd_n9` | `(cluster CTAs * groups, 1, 1)`, persistent | per token: every `dv_j`. Each CTA keeps its tokens' `dw` sums in registers and adds them into the zeroed `dw` once, after its token loop, with `red.global.add.v4.f32` (four columns per instruction) |

`T = 0` launches no Cake kernel: the backward returns a zero `dw` (a `torch.zeros` fill).

`groups = min(T, resident groups)`, the groups the device holds at once (`cake_jit.resident_groups`: counts measured per launch bound on B200, GB200, GB300, VR200 and R200; on another device a fallback under every measured count). `fwd_n4` takes at most 64 groups on B200, GB200 and GB300 (`cake_jit.GROUP_CAPS`): it runs up to 7 % faster there than at all 71 or 74 resident groups. The kernels take nine value pointers; the host passes the first value in the slots
past `n`, which the kernels never read.

The zeroing of `dw` is part of the backward launch: `run` takes `dw` again as its `zero_region` argument and zeroes it
with a memset on the launch stream right before the kernel, so no host work sits between the zeroing and the kernel.

## Determinism

`out`, `p`, `r`, `d` and every `dv_j` have one fixed order. The CTAs add their `dw` sums in the order they finish, so
`dw` promises no bits run to run. Under `torch.use_deterministic_algorithms(True)` a backward of `T > 0` (explicit or
autograd) raises `RuntimeError` before the `bwd` launch, or warns (`UserWarning`) and runs under `warn_only=True`. The
forward and `T = 0` never alert. A launch captured into a CUDA graph alerts at capture; a replay does not.

## Memory

Retained for the backward: references to the inputs and `p`, `r`, `d` (`12 * T * n` bytes). Backward workspace: none.
