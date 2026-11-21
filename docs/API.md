# API reference

This page covers the public modules of all 21 kernel families. The content comes from the facade docstrings and the source. Signatures below leave out type annotations; the Inputs column gives types and shapes. Every family supports `sm_100a`, `sm_103a` and `sm_107a`, and some families limit shapes per architecture.

## Naming

| Pattern | Example | Meaning |
|---|---|---|
| `cake_pplx.cake_<op>_<mode>_train` | `cake_pplx.cake_kda_det_train` | Single-rank module |
| `cake_pplx.cake_<op>_<mode>_train_kcp` | `cake_pplx.cake_kda_det_train_kcp` | Context-parallel (KCP) module. Exists for kda, gdn and causal_conv1d. |
| `cake_<op>_<mode>[_kcp]` | `cake_kda_det`, `cake_kda_det_kcp` | Autograd entry point |
| `cake_<op>_<mode>_train_<step>` | `cake_kda_det_train_forward` | Explicit (non-autograd) step or helper |

`<mode>` is `det` or `nondet`. The module name alone selects the mode. No entry point takes a `deterministic` keyword, and passing one raises `TypeError`. Some families do not follow these patterns: mhc's autograd entries are `cake_mhc_<mode>_train_prepare` / `_finish`, gated_act_mxfp8, act_qdq, dense_gemm and dsa_indexer have no autograd entry, fmha has only a backward step (`cake_fmha_nondet_train_backward`), and engram_lookup has only a forward step (`cake_engram_lookup_det_train_forward`) and a helper that splits a packed table (`cake_engram_lookup_det_train_split_packed`). mok has no autograd entry: its entries are `prepare_mok_bf16` (an adapter with forward, recompute and backward methods), `create_mok_bf16_workspace` and `quantize_mok_mxfp8_weights`; its backend package is `cake_mok_bf16`. gated_kv_compression has a second autograd entry, `cake_gated_kv_compression_det_cz`, for one `[T, 1024]` input. Each family section notes these exceptions.

The facades load their backends (`cake_pplx._backends.*`, which are private) lazily, on the first call or attribute access. Kernels compile with `nvcc` on first use unless a prebuilt library exists. The JIT environment variables are listed in the [README](../README.md#jit-build-and-cache), and this page lists only variables that are specific to a family.

## det and nondet

- **det:** reductions run in a fixed order, so the same inputs, hardware and plan give bitwise identical outputs and gradients. Some families add conditions, such as the same token count, the same SM count or the same topology.
- **nondet:** these modules may use orders that det does not allow, for example atomics or arrival-order partial sums. Results stay within det's accuracy budgets but can differ from det in the last bits. gdn promises neither: see its row below (its tests record each output's error against an FP64 reference beside the BF16 naive implementation's). Several nondet modules still run in a fixed order today. Neither mode ever falls back to the other. fmha has no det module; its accuracy reference is an FP64 oracle beside the same math in BF16. moe_grouped_gemm has no accuracy budget; its tests judge both modules against a naive BF16 implementation, with an FP64 oracle.

| Family | nondet compared with det | nondet under `torch.use_deterministic_algorithms(True)` |
|---|---|---|
| rmsnorm | `dw` is not guaranteed bitwise reproducible. `dx` is bitwise reproducible. | Backward always alerts |
| rmsnorm_silu_gate | A BF16 `dw` in the fused band may vary in its last bits. `y` can differ from det. | Alerts only for a BF16 `dw` in the fused band |
| causal_conv1d | `dweight` / `dbias` use unordered reductions. The nondet package makes no bit promise. | Backward raises |
| kda | Bitwise reproducible. `dg` differs from det by FP32 rounding; everything else is bitwise det. | No alert |
| gdn | Fixed order today, with no run-to-run promise. Not bitwise det. As with det, the tests record each output's error against an FP64 reference beside the BF16 naive implementation's. | Runs unchanged |
| mhc | Plain FP32 sums in a fixed order today. No bit promise; not compared bitwise with det. | No alert |
| mhc_head | Same kernels as det | No alert |
| engram_gate | Same kernels as det. `out`, `dX`, `dK`, `dV` have one fixed order; no bit promise for `dq_weight` / `dk_weight`. | No alert |
| gqa | No atomics. No promise that an output repeats bit for bit, run to run or across workspace budgets and host boundaries. | No alert |
| fmha | Has no det module. `dq` sums FP32 tiles in arrival order; `dk`/`dv` have one fixed order. | Raises before launch |
| dsa_indexer | Each row holds its selected set in no specified slot order | Raises before launch |
| dsa_sparse_attention | `dkv` uses unordered FP32 atomics. No output is promised to repeat bit for bit. The opt-in `with_o_lo=False` forms `delta` from the BF16 `out` alone. | Backward raises |
| dense_gemm | Some routes add partials in arrival order. No run-to-run guarantee. | Alerts only for order-varying launches (explicit split counts) |
| moe_grouped_gemm | Every shipped launch has a fixed order. `dW` can differ from det on a split 512 tail. | No shipped launch alerts |
| lm_head | `dX` is slab-summed and row skips can change bits. No run-to-run promise for any output. | Alerts before a K-sliced `dX` reduction |
| gated_act_mxfp8 | Has no nondet module | — |
| clamped_swiglu | Has no nondet module | — |
| engram_lookup | Has no nondet module | — |
| mok | Has no nondet module | — |
| act_qdq | Has no nondet module | — |
| gated_kv_compression | Has no nondet module | — |

---

## rmsnorm

**Computes.** Fused BF16 RMSNorm for training.

- **Forward:** `r_t = (mean_j x_tj^2 + eps)^(-1/2)` in FP32 and `y = BF16((x * r) * w)`. Only the FP32 per-row `rstd [T]` is saved.
- **Backward:** one fused pass computes `dx = BF16(r * g * w - x * r^3 * mean(g * w * x))` and the FP32 `dw = sum_t g * x * r`.
- **Optional residual-add fusion:** computes `h_new = BF16(h + u)`, normalizes `h_new`, and returns one `dx` that is the gradient of both `h` and `u`.

**Architectures.** `sm_100a`, `sm_103a`, `sm_107a` (B200, B300/GB300, R200), with no family fallback. The hidden size `H` must be 128, 256, 512, 1024, 1280, 1536, 2048, 4096, 5120 or 6144 on every architecture. Residual-fused routes exist only for `H = 6144`. FP32-weight routes exist for all ten plain hidden sizes.

### `cake_pplx.cake_rmsnorm_det_train`

`dw` comes from a fixed row-chunk partition followed by a fixed-order reduction. It is bitwise reproducible for a given token count on devices with the same SM count; devices of one architecture with different SM counts can differ. `torch.use_deterministic_algorithms` has no effect. There is no KCP module. Source: [`kernels/rmsnorm/cake_pplx/cake_rmsnorm_det_train.py`](../kernels/rmsnorm/cake_pplx/cake_rmsnorm_det_train.py).

| Name | Signature | Inputs | Returns / notes |
|---|---|---|---|
| `cake_rmsnorm_det` | `(x, w, eps=1e-6, residual=None)` | `x` BF16 `[T, H]` or `[..., H]`; `w` BF16 or FP32 `[H]`; `residual` optional BF16 of `x`'s shape | Autograd. Returns BF16 `y`, or `(y, h_new)` with `residual`. `dw` is returned in `w.dtype`. |
| `cake_rmsnorm_det_train_forward` | `(x, w, eps, *, residual=None, out=None, rstd=None, residual_out=None)` | `x` BF16 `[T, H]`, or `[..., H]` with leading dims row-major contiguous with one another; `w` FP32 only on routes listed by `cake_rmsnorm_det_train_weight_dtypes`; `out`/`rstd`/`residual_out` preallocated, must not overlap one another (not checked) | Returns `(y, rstd)`, or `(y, rstd, h_new)` with `residual`. `rstd` is FP32 `1/sqrt(mean(h_new^2)+eps)`. Not autograd. A preallocated output that is not contiguous and aligned, or that shares memory with `x`, `residual` or `w`, is written through a temporary and copied after the launch. T == 1 is supported. T == 0 returns empty outputs without launching. |
| `cake_rmsnorm_det_train_backward` | `(g, x, w, rstd, *, g_residual=None, workspace=None, dx=None, dw=None)` | `g` BF16 `[T, H]`; `x` is the normalized tensor (`x`, or `h_new` in residual mode); `rstd` FP32 `[T]`; `g_residual` BF16 (residual mode); `workspace` uint8 from the factory, allocated when omitted, must not overlap `g`, `x`, `w`, `rstd` or `g_residual` (not checked); `dx`/`dw` preallocated, must not overlap `g`, `x`, `w`, `rstd`, `g_residual`, `workspace` or each other (not checked) | Returns `(dx, dw)` with FP32 `dw [H]`. The caller decides whether to cast. |

Other public names:

- `ARCHES`, `SUPPORTED_COMPUTE_CAPABILITIES = ((10, 0), (10, 3), (10, 7))`.
- `CakeRMSNormDetFunction`: the autograd Function. It saves only `(x_or_h_new, w, rstd)`.
- `cake_rmsnorm_det_train_backward_workspace(rows, hidden, device, *, residual=False)`: returns a zeroed uint8 buffer. `..._backward_workspace_bytes(...)` takes the same arguments.
- `cake_rmsnorm_det_train_program_windows(arch, hidden, kind, residual, w_dtype="bf16")`: the row-count windows of a route and the program each window launches.
- `cake_rmsnorm_det_train_route_key(hidden, kind, residual, w_dtype="bf16")`: returns `<kind>[_w32][_residual]_h<hidden>`.
- `cake_rmsnorm_det_train_weight_dtypes(arch, hidden, kind, residual)`: the weight dtypes (`'bf16'`, `'f32'`) that a route ships.
- `is_cake_rmsnorm_det_train_supported(device, hidden=None)`.

### `cake_pplx.cake_rmsnorm_nondet_train`

The forward runs the det forward programs and is bitwise reproducible.

Each backward token-count window runs one form, and `dx` is bitwise reproducible in every form. Two forms sum `dw` in scheduling order, so their `dw` is not guaranteed bitwise reproducible: the single kernel adds per-CTA `dw` sums into replicated FP32 column sums and stores their sum as `dw`, and the part-row single kernel writes one FP32 row per CTA, which helper CTAs fold. Every other window runs a fixed-order form: the one-cluster kernel, which reads no workspace, at small token counts (up to a top of T 2 to 16, or 64 rows at H = 256; the `H = 128` routes and the `H = 1280` routes other than the FP32-weight one on `sm_107a` run none), a column-task program (its CTAs sum whole `dw` columns over every row) or the deterministic pair. Their `dw` is bitwise reproducible for a given token count on devices with the same SM count. The backward always alerts under deterministic algorithms. There is no KCP module. Source: [`kernels/rmsnorm/cake_pplx/cake_rmsnorm_nondet_train.py`](../kernels/rmsnorm/cake_pplx/cake_rmsnorm_nondet_train.py).

| Name | Signature | Inputs | Returns / notes |
|---|---|---|---|
| `cake_rmsnorm_nondet` | `(x, w, eps=1e-6, residual=None)` | Same as `cake_rmsnorm_det` | Autograd. Returns `y` or `(y, h_new)`. `dw` is returned in `w.dtype`. The backward alerts and names `cake_rmsnorm_det`. |
| `cake_rmsnorm_nondet_train_forward` | `(x, w, eps, *, residual=None, out=None, rstd=None, residual_out=None)` | Same as the det forward | Returns `(y, rstd)` or `(y, rstd, h_new)`. Never alerts. Preallocated outputs follow the det forward's rules and must not overlap one another (not checked). |
| `cake_rmsnorm_nondet_train_backward` | `(g, x, w, rstd, *, g_residual=None, workspace=None, dx=None, dw=None)` | Same as the det backward. Use one workspace from `cake_rmsnorm_nondet_train_backward_workspace` per concurrently running backward. The workspace must not overlap `g`, `x`, `w`, `rstd` or `g_residual` (not checked). `dx`/`dw` must not overlap `g`, `x`, `w`, `rstd`, `g_residual`, `workspace` or each other (not checked). | Returns `(dx, dw)` with FP32 `dw`. T == 0 returns an empty `dx` and a zero `dw`. Under deterministic algorithms it raises `RuntimeError` naming `cake_rmsnorm_det_train_backward`. With `warn_only=True` it emits one `UserWarning` and runs. If `workspace` is omitted, the call allocates and zero-fills one (about 3 us fill plus about 35 us host time on B200). |

Other public names:

- `ARCHES`, `SUPPORTED_COMPUTE_CAPABILITIES`, `CakeRMSNormNondetFunction`.
- `cake_rmsnorm_nondet_train_backward_workspace(rows, hidden, device, *, residual=False)` and `..._workspace_bytes(...)`: one fixed-layout buffer, zeroed once, serves any later sequence of calls that it covers.
- `cake_rmsnorm_nondet_train_route_key(hidden, kind, residual, w_dtype="bf16")`.
- `is_cake_rmsnorm_nondet_train_supported(device, hidden=None)`: does not depend on the deterministic-algorithms setting.

**Options and environment.** Forward keywords: `residual=`, `out=`, `rstd=`, `residual_out=`. Backward keywords: `g_residual=`, `workspace=`, `dx=`, `dw=`. If other streams run resident kernels, setting `cudaFuncAttributePreferredSharedMemoryCarveout = cudaSharedmemCarveoutMaxShared` on those kernels avoids delaying the start of the backward. This family has no environment variables of its own.

**Cross-family.** None. Within the family, a backward does not check which package produced `rstd`. The det and nondet workspace layouts differ, so do not pass a workspace from one package to the other.

**Example.**

```python
import torch
from cake_pplx.cake_rmsnorm_det_train import (
    cake_rmsnorm_det_train_forward, cake_rmsnorm_det_train_backward)
from cake_pplx.cake_rmsnorm_nondet_train import cake_rmsnorm_nondet

T, H, eps = 129, 4096, 1e-6
x = torch.randn(T, H, device="cuda").to(torch.bfloat16)
g = torch.randn(T, H, device="cuda").to(torch.bfloat16)
w = torch.randn(H, device="cuda").to(torch.bfloat16)

y, rstd = cake_rmsnorm_det_train_forward(x, w, eps)      # y BF16 [T, H], rstd FP32 [T]
dx, dw = cake_rmsnorm_det_train_backward(g, x, w, rstd)  # dw FP32 [H]

xa = x.clone().requires_grad_(True)
wa = w.clone().requires_grad_(True)
ya = cake_rmsnorm_nondet(xa, wa, eps)                    # autograd, nondet dw
ya.backward(g)                                           # wa.grad in w.dtype (BF16)
```

**Caveats.**

- Only the seven hidden sizes listed above have routes. Check one with `is_cake_rmsnorm_*_train_supported(device, hidden)`.
- Inputs and outputs are BF16; `rstd` and `dw` are FP32, and the autograd wrappers cast `dw` to `w.dtype`. An FP32 `w` is meant to be the effective weight `1 + w` of a zero-centred model.
- Workspaces must be zero on first use (use the factory or `torch.zeros`). The kernels leave them at rest, so a workspace can be reused across calls and CUDA Graph replays. Give each concurrently running stream its own buffer.
- A strided input is read in place only if `stride(1) == 1`, `stride(0)` is a multiple of 8 elements and storage is aligned (32 B for the forward, 16 B for the backward). Otherwise it is copied once.
- Kernels on other streams that hold every SM with a small shared-memory carveout can delay these calls. Some SM107 programs need a whole SM; the det module docstring lists them.

---

## rmsnorm_silu_gate

**Computes.** SiLU-gated RMSNorm over a last dimension of 128: `rstd = 1/sqrt(mean(x^2) + eps)` and `y = BF16(((x * rstd) * weight) * silu(z))`. This is `RMSNorm(x) * weight * SiLU(z)`.

The backward recomputes `rstd` and the gate from `x` and `z`. With `n = x*rstd`, `s = sigmoid(z)` and `a = dy*silu(z)`:

- `dz = BF16(dy*(n*weight)*s*(1 + z*(1 - s)))`
- `dx = BF16(rstd*a*weight - x*rstd^3*mean(a*weight*x))`
- `dw = sum of a*n over all tokens and heads`, summed in FP32 (no FP64) and returned in FP32, or rounded to BF16 for a BF16 gradient. Each module section below says how its sum is ordered and rounded.

**Architectures.** `sm_100a`, `sm_103a`, `sm_107a`.

**Inputs (both modules).** `x` is BF16 `[..., 128]` (`[T, H, 128]` or `[T * H, 128]`). `z` is a BF16 gate of `x`'s shape with its own strides. `weight` is BF16 `[128]`. Inputs are read in place when the last stride is 1, the strides and offset are multiples of 16 elements, and storage is 32-byte aligned; otherwise they are copied once. `T * H` must not exceed `INT32_MAX`.

### `cake_pplx.cake_rmsnorm_silu_gate_det_train`

`dw` is summed in FP32 runs of at most 16 rows. Each run is added into a plain FP32 word of its chunk, and each CTA sums its words in a fixed order in FP32. Each CTA's FP32 column sum is added exactly in fixed point, whatever the CTA order, and the total is rounded once to FP32 (no FP64, no two-word sums). The same row count on a device with the same SM count gives the same bits. A call that computes `dw` on up to 256 rows runs one CTA (up to 16 rows) or one thread-block cluster of 16 CTAs (17 to 256 rows), and neither uses a workspace. It forms the same words and rounds their exact sum the same way, so it gives the same bits on any device of at least 4 SMs, whatever the SM count. The CTAs of the cluster wait for each other's `dw` words; the GPU starts all 16 together, once one GPC has room for them. On a MIG instance or a device of fewer than 16 SMs, 17 to 256 rows run the backward of more rows instead, which uses the workspace. Other SM partitions, such as a CUDA green context, are not detected: if no GPC of the partition can hold the 16 CTAs, the cluster launch fails with a CUDA error. No call alerts under deterministic algorithms. There is no KCP module. Source: [`kernels/rmsnorm_silu_gate/cake_pplx/cake_rmsnorm_silu_gate_det_train.py`](../kernels/rmsnorm_silu_gate/cake_pplx/cake_rmsnorm_silu_gate_det_train.py).

| Name | Signature | Inputs | Returns / notes |
|---|---|---|---|
| `cake_rmsnorm_silu_gate_det` | `(x, z, weight, eps=1e-6)` | As above | Autograd. Returns BF16 `y`. Saves only `x`, `z` and `weight`, so it works with checkpointing (`use_reentrant=False`). |
| `cake_rmsnorm_silu_gate_det_train_forward` | `(x, z, weight, eps=1e-6, *, out=None)` | As above; `out` optional BF16 of `x`'s shape | Returns BF16 `y`. Computes in FP32 with approximate rsqrt/ex2/rcp and one final BF16 rounding. Same inputs give the same bits. Short inputs whose rows are packed (`x` and `z` contiguous `[T, H, 128]`, or one token of head stride 128) run, up to a cut of rows per SM on each architecture (on SM107 not from 17 to 32 rows), a program of one row block per CTA with no loop. It computes each row with the same FP32 operations in the same order, so a row's `y` bits do not depend on which program runs it. |
| `cake_rmsnorm_silu_gate_det_train_backward` | `(grad_out, x, z, weight, eps=1e-6, *, grad_x=None, grad_z=None, grad_weight=None, accumulate_grad_weight=False, compute_grad_weight=True, workspace=None)` | `grad_out`, `x`, `z` BF16 of one shape, any strides; `grad_weight` BF16 or FP32 `[128]` (a new BF16 tensor by default); `workspace` zero-filled FP32, contiguous, 16-byte aligned | Returns `(grad_x, grad_z, grad_weight)`. The package tests hold `dx`, `dz` and `dw` to the error of the PyTorch BF16 naive implementation (against float64, 32 random draws). `accumulate_grad_weight` adds exactly before rounding and needs `grad_weight`. `compute_grad_weight=False` returns `None` for `dw` and runs a program built without the weight gradient: the same `dx` and `dz` bits as a call that computes it. On SM100 and SM103 up to 768 rows per SM, and on SM107 except where the look-ahead program runs, that program reads and writes no workspace word. A call that computes `dw` reads and writes no workspace word either up to 16 rows, and from 17 to 256 rows where the cluster runs (not on a MIG instance or a device of fewer than 16 SMs). A nondet workspace raises `ValueError`. |

Other public names:

- `ARCHES`, `SUPPORTED_COMPUTE_CAPABILITIES`.
- `CakeRMSNormSiLUGateDetFunction`: passes `compute_grad_weight=needs_input_grad[2]`.
- `cake_rmsnorm_silu_gate_det_train_backward_workspace_bytes(rows, device)`: always 18952 bytes. `..._backward_workspace(rows, device)` returns a zero-filled FP32 buffer, which every backward leaves at zero. The size does not depend on `compute_grad_weight`. A call that computes `dw` on up to 16 rows, or on 17 to 256 rows where the cluster runs (not on a MIG instance or a device of fewer than 16 SMs), checks a buffer passed to it and does not touch it.
- `cake_rmsnorm_silu_gate_det_train_reset_workspace(workspace)`: zeroes a buffer on the current stream without a host sync. Needed only for foreign or corrupted buffers. It refuses a nondet workspace.
- `is_cake_rmsnorm_silu_gate_det_train_supported(device)`.

### `cake_pplx.cake_rmsnorm_silu_gate_nondet_train`

`y`, `dx`, `dz` and an FP32 weight gradient repeat bit for bit.

A BF16 weight gradient in the **fused band** adds each CTA's FP32 column sums into FP32 replicas with FP32 atomic adds, so its last bits may vary between runs. The fused band is past 512 rows on SM100 and past 256 rows on SM103, up to 1000 rows per SM; on SM107 it is past 64 rows per SM, up to 256 rows per SM. Only that backward alerts under deterministic algorithms. A call with `compute_grad_weight=False` runs its band's program built without the weight gradient: the same `dx` and `dz` bits, no column sums, no alert. On SM100 and SM103, packed rows (and a single token) up to 128 rows per SM run the repeatable band's program built without the weight gradient instead: the `dx` and `dz` bits of the repeatable backward, whose last bits can differ from the band's. The other bands (one-cluster, static and repeatable) are bitwise repeatable; in a repeatable band (a BF16 `dw` past the fused band, on SM107 also before it; an FP32 `dw` past 512 rows except in the static band) `dw` is the same bits for the same rows and SM count. The static band is an FP32 `dw` on SM100 and SM103 past 64 and up to 128 rows per SM; there `dw` is the exact sum of each chunk's FP32 partial (integer windows), rounded once to FP32, the same bits at any grid. There is no KCP module. Source: [`kernels/rmsnorm_silu_gate/cake_pplx/cake_rmsnorm_silu_gate_nondet_train.py`](../kernels/rmsnorm_silu_gate/cake_pplx/cake_rmsnorm_silu_gate_nondet_train.py).

| Name | Signature | Inputs | Returns / notes |
|---|---|---|---|
| `cake_rmsnorm_silu_gate_nondet` | `(x, z, weight, eps=1e-6)` | As above | Autograd. Saves `x`, `z` and `weight`. The backward alerts for a BF16 `dw` in the fused band. |
| `cake_rmsnorm_silu_gate_nondet_train_forward` | `(x, z, weight, eps=1e-6, *, out=None)` | As above | Returns BF16 `y`. Repeats for the same inputs on the same device, but the bits can differ between row bands and from det. Never alerts. |
| `cake_rmsnorm_silu_gate_nondet_train_backward` | `(grad_out, x, z, weight, eps=1e-6, *, grad_x=None, grad_z=None, grad_weight=None, accumulate_grad_weight=False, compute_grad_weight=True, workspace=None)` | As det; `workspace` is a zero-filled `torch.uint8` buffer from the nondet factory | Returns `(grad_x, grad_z, grad_weight)`. One kernel is chosen by rows, SM count and `dw` dtype. Each kernel repeats `dx`/`dz` bit for bit; their last bits can differ between row bands. The package tests hold every output to the error of the PyTorch BF16 naive implementation (against float64, 32 random draws). Under deterministic algorithms, a BF16-`dw` call in the fused band raises `RuntimeError` just before launch (`UserWarning` with `warn_only`). The check runs at CUDA Graph capture, not at replay. `compute_grad_weight=False` never alerts. It gives the `dx` and `dz` bits of the call with a BF16 `dw`, except on SM100 and SM103 for packed rows (and a single token) up to 128 rows per SM, where it gives the repeatable backward's bits. A det FP32 workspace raises `ValueError`. |

Other public names:

- `ARCHES`, `SUPPORTED_COMPUTE_CAPABILITIES`, `CakeRMSNormSiLUGateNondetFunction`.
- `cake_rmsnorm_silu_gate_nondet_train_backward_workspace_bytes(rows, device)`: 29192 bytes on SM100/SM103 and 36360 bytes on SM107, whatever the row count and `compute_grad_weight`. `..._backward_workspace(rows, device)` returns a zero-filled uint8 buffer.
- `cake_rmsnorm_silu_gate_nondet_train_reset_workspace(workspace)`: refuses a det FP32 buffer.
- `is_cake_rmsnorm_silu_gate_nondet_train_supported(device)`.

**Options and environment.** All controls are keywords. This family has no environment variables of its own. Without a workspace, the nondet backward keeps one buffer per device, stream and backward kernel that uses one (a backward without the weight gradient uses none in the one-cluster bands, on SM100 and SM103 on packed rows up to 128 rows per SM, and where it runs the strided program: on SM100 in the fused band up to 512 rows per SM, on SM103 up to 448, and on SM107 in every band but where the look-ahead program runs). It also keeps a reserve of 64 zero-filled buffers per device for CUDA Graph capture (about 1.8 MiB on SM100/SM103 and 2.2 MiB on SM107).

**Cross-family.** None. The det and nondet packages never run each other's kernels, and each refuses the other's workspace (det FP32 vs nondet uint8) with `ValueError`.

**Example.**

```python
import torch
from cake_pplx.cake_rmsnorm_silu_gate_det_train import (
    cake_rmsnorm_silu_gate_det,
    cake_rmsnorm_silu_gate_det_train_forward,
    cake_rmsnorm_silu_gate_det_train_backward,
)

g = torch.Generator(device="cuda").manual_seed(11)
x, z, dy = (torch.randn(129, 48, 128, device="cuda", generator=g).bfloat16() for _ in range(3))
w = (1.0 + 0.5 * torch.randn(128, device="cuda", generator=g)).bfloat16()

y = cake_rmsnorm_silu_gate_det_train_forward(x, z, w, 1e-6)
dx, dz, dw = cake_rmsnorm_silu_gate_det_train_backward(dy, x, z, w, 1e-6)  # dw BF16 [128]

xa, za, wa = (t.clone().requires_grad_(True) for t in (x, z, w))
ya = cake_rmsnorm_silu_gate_det(xa, za, wa, 1e-6)
ya.backward(dy)  # xa.grad == dx, za.grad == dz, wa.grad == dw
```

**Caveats.**

- The normalized dimension is fixed at 128. Other dtypes or shapes raise `ValueError`.
- Approximate FP32 rsqrt/ex2/rcp means `y` can differ from the correctly rounded FP32 formula by one BF16 ulp near rounding boundaries. In the det forward, the gate is -0 for finite `z` below about -87.3.
- The default weight gradient is BF16. For FP32 main-grad accumulation, pass an FP32 `grad_weight`, optionally with `accumulate_grad_weight=True`.
- A workspace that did not come from the factory, or that anything else wrote to, must be reset before its next backward. Use one buffer per concurrently running stream.

---

## causal_conv1d

**Computes.** A packed width-4 depthwise causal convolution followed by SiLU, over BF16 `[S, C]` activations split into documents by `cu_seqlens`. Each document starts with zero history.

- **Forward:** `pre[t,c] = sum_k w[c,k]*x[t-3+k,c] + b[c]` and `y = bf16(pre*sigmoid(pre))`, computed in FP32 with one BF16 rounding.
- **Backward:** recomputes `pre` from `x` and returns BF16 `dx`, `dweight[c,k] = sum_t dpre[t,c]*x[t-3+k,c]` and, optionally, `dbias[c] = sum_t dpre[t,c]`. Under the default `silu_grad_preactivation="bf16"`, SiLU' is evaluated at `bf16(pre)` with FP32 taps and at the unrounded FP32 `pre` with BF16 taps (routes `bwd_dt_bf16_taps` / `bwd_bias_bf16_taps`). Under `"fp32"` it is evaluated at the unrounded FP32 `pre`.
- **KCP variants:** exchange 3 history rows between context-parallel ranks with one halo `all_to_all_single` per direction.

**Architectures.** `ARCHES = ("sm_100a", "sm_103a", "sm_107a")`, with no family fallback. Optional routes (bias, BF16 taps/gradients, FP32 SiLU') are checked per call and raise `ValueError` when the architecture lacks them. In nondet KCP, `bwd_reserve_sms >= 1` raises `ValueError` on `sm_107a`.

**Inputs (single rank).** `x` is BF16 `[S, C]` or `[1, S, C]`, contiguous, with `C` a multiple of 8. `weight` is FP32 or BF16 `[C, 4]`, contiguous; `weight[c,3]` multiplies the current token, and other widths are rejected. `cu_seqlens` is int32 `[num_docs+1]` on the device, with `cu[0]==0` and `cu[-1]==S`. The host reads only metadata, so there is no sync. `bias` is FP32 or BF16 `[C]`, and its dtype is independent of `weight`'s.

### `cake_pplx.cake_causal_conv1d_det_train`

The backward has two launches. A persistent main launch writes `dx` plus per-CTA FP32 partial rows. A programmatically dependent reduce launch then sums those rows in a fixed order, with no FP atomics. Gradients are bitwise reproducible for a given `(S, C, device)` and route. Source: [`kernels/causal_conv1d/cake_pplx/cake_causal_conv1d_det_train.py`](../kernels/causal_conv1d/cake_pplx/cake_causal_conv1d_det_train.py).

| Name | Signature | Inputs | Returns / notes |
|---|---|---|---|
| `cake_causal_conv1d_det` | `(x, weight, cu_seqlens, bias=None, *, silu_grad_preactivation="bf16", bwd_grid_mult=1, dweight_sum="fp32")` | As above | Autograd. Returns BF16 `y`. The backward saves only `x`, `weight`, `bias` and `cu_seqlens`. Parameter gradients come back in the parameters' dtypes. |
| `cake_causal_conv1d_det_train_forward` | `(x, weight, cu_seqlens, *, bias=None, out=None, workspace=None)` | `out` preallocated BF16; `workspace` int32 from `..._forward_workspace`, one per stream | Returns BF16 `y` of `x`'s shape. S == 0 returns an empty output without launching. Not autograd. |
| `cake_causal_conv1d_det_train_backward` | `(x, weight, cu_seqlens, dy, *, bias=None, silu_grad_preactivation="bf16", workspace=None, dx=None, dweight=None, dbias=None, bwd_grid_mult=1, dweight_sum="fp32")` | `dy` BF16, contiguous; `workspace` uint8 from `..._backward_workspace` (no zeroing needed); `dx` BF16, `dweight [C,4]` and `dbias [C]` each FP32 or BF16 (`dbias` needs `bias`); `bwd_grid_mult` 1 to 4; `dweight_sum` `"fp32"` or `"fp32_pair"` | Returns `(dx, dweight)`, plus `dbias` when `bias` is given. With FP32 taps, `"bf16"` evaluates SiLU' at the BF16-rounded preactivation, and `dweight` is within about 3e-5 rel-L2 of the exact FP64 gradient. With BF16 taps, `"bf16"` evaluates it at the unrounded FP32 preactivation. `"fp32"` evaluates it at the unrounded FP32 preactivation and is slower. By default both sum `dweight` and `dbias` in plain FP32 words, with no FP64 and no compensated sums. `dweight_sum="fp32_pair"` (opt-in, with `silu_grad_preactivation="fp32"` only) sums them as FP32 (hi, lo) pairs: a Kahan-compensated fold and a hi/lo pair row sum, rounded once, FP32 operations only. It keeps `dx`, the workspace and run-to-run bitwise reproducibility; its bits differ from the default's. `dx` is within one BF16 ulp either way. Size the workspace with the same or a larger `bwd_grid_mult`. A non-int `bwd_grid_mult` raises `TypeError`; other bad values raise `ValueError`. |

Other public names:

- `ARCHES`, `DWEIGHT_SUMS` (`("fp32", "fp32_pair")`), `SUPPORTED_COMPUTE_CAPABILITIES`.
- `CakeCausalConv1dDetFunction`.
- `cake_causal_conv1d_det_train_backward_workspace(S, C, device, *, bwd_grid_mult=1)` and `..._bytes(...)`: the size is the maximum over every backward program of every route.
- `cake_causal_conv1d_det_train_forward_workspace(S, C, device)` and `..._bytes(...)`: returns a zeroed int32 buffer of token-chunk counters. Its size depends on `C` and the package version: 0.2.4 runs narrower forward tiles at C = 520 to 568 and needs more counters there, so a buffer sized by an earlier version can be too small. A too-small buffer raises `ValueError`. 0.2.5 and 0.2.6 do not change the size: the forward rows they add run programs that take no counters (see `workspace=` below).
- `is_cake_causal_conv1d_det_train_supported(device)`: checks only the base routes.

### `cake_pplx.cake_causal_conv1d_det_train_kcp`

Supports FP32 taps only, with no bias. `dweight` is this rank's fixed-order FP32 partial. It is summed in plain FP32 words and is bitwise reproducible run to run. There is no `dweight_sum` keyword, so there is no FP32 (hi, lo) pair opt-in. Reducing `dweight` across ranks is the caller's job. Source: [`kernels/causal_conv1d/cake_pplx/cake_causal_conv1d_det_train_kcp.py`](../kernels/causal_conv1d/cake_pplx/cake_causal_conv1d_det_train_kcp.py).

| Name | Signature | Inputs | Returns / notes |
|---|---|---|---|
| `cake_causal_conv1d_det_kcp` | `(x_local, weight, ctx, *, recompute_handle=None, cooperative=None)` | `x_local` BF16 `[S_local, C]` or `[1, S_local, C]` in partition order; `weight` FP32 `[C, 4]`; `ctx` from the calling family's KCP context builder; `recompute_handle` any object with an `entries` dict; `cooperative` `None` (= `False`) or `True` | Autograd. Returns BF16 `y` of `x_local`'s shape. It equals the single-rank result on this rank's tokens. A continuing fragment reads the history rows of the preceding fragment through the halo collective. `cooperative` must be the same on every rank, and `dweight` bits differ between the two forms. |
| `cake_causal_conv1d_det_kcp_multi` | `(xs, weights, ctx, *, recompute_handle=None, cooperative=None)` | `xs`: n >= 1 BF16 tensors, each `C_i` a multiple of 8 (may differ); `weights`: FP32 `[C_i, 4]`; one handle per call (`"conv"` entry) | Autograd. Returns n BF16 outputs with one halo exchange per direction. Outputs and every gradient are bitwise equal to the single calls. Every rank must call once per layer with the same order and counts. This is not checked; a mismatch hangs or corrupts data. |

Other public names: `CakeCausalConv1dDetKcpFunction`, `CakeCausalConv1dDetKcpMultiFunction`, `CakeCausalConv1dDetKcpRecomputeHandle`.

### `cake_pplx.cake_causal_conv1d_nondet_train`

The forward runs copies of the det programs. The backward is one launch that adds `dweight`/`dbias` into zeroed accumulators with unordered global reductions. By default `dweight`/`dbias` accumulate in plain FP32 words, as the det reduce does, and their order across CTAs varies. At the rows of backend `cake_jit.SM_WIDE_ROWS` (the default `bwd` route, by architecture, `C` and `S`: C = 520 to 608 from S = 24576 on sm_100a and from S = 20480 on sm_103a, C = 520 to 680 from S = 32768 on sm_107a), that launch runs one CTA per SM with the body of the det package's SM-wide row-group program and adds its `dweight` partials with these unordered reductions instead of the det reduce launch. On sm_100a and sm_103a, at the backward rows of backend `cake_jit.FALLBACK_ROWS` (the default `bwd` route at C = 520 to 1016; from 0.2.7 also the `bwd_bias`, `bwd_bias_bf16_taps` and `bwd_dt_bf16_taps` routes at C = 520 to 1016 and small S, `bwd_dt_bf16_taps` not at C = 768 on sm_103a; each row up to its own token bound), that launch runs the route's fallback program, the one the route runs at its smallest channel counts, in place of the program its channel thresholds pick. The nondet package makes no bit promise. Under deterministic algorithms the backward raises `RuntimeError` before launch; with `warn_only=True` it emits a `UserWarning` and runs. Source: [`kernels/causal_conv1d/cake_pplx/cake_causal_conv1d_nondet_train.py`](../kernels/causal_conv1d/cake_pplx/cake_causal_conv1d_nondet_train.py).

| Name | Signature | Inputs | Returns / notes |
|---|---|---|---|
| `cake_causal_conv1d_nondet` | `(x, weight, cu_seqlens, bias=None, *, silu_grad_preactivation="bf16", bwd_grid_mult=1, dweight_sum="fp32")` | As det. `bwd_grid_mult` and `dweight_sum` are checked at the forward call. | Autograd. Returns BF16 `y`. The backward raises under deterministic algorithms. |
| `cake_causal_conv1d_nondet_train_forward` | `(x, weight, cu_seqlens, *, bias=None, out=None, workspace=None)` | As the det forward; workspace from `..._nondet_train_forward_workspace` | Returns BF16 `y`. No alert. |
| `cake_causal_conv1d_nondet_train_backward` | `(x, weight, cu_seqlens, dy, *, bias=None, silu_grad_preactivation="bf16", workspace=None, dx=None, dweight=None, dbias=None, bwd_grid_mult=1, dweight_sum="fp32")` | By default both `"bf16"` and `"fp32"` accumulate in plain FP32 words; the `"fp32"` routes spread the adds over sixteen FP32 accumulator copies, added in copy order at the end. `dweight_sum="fp32_pair"` (opt-in, with `"fp32"` only) keeps each accumulator word as an FP32 (hi, lo) pair: a Kahan-compensated fold, each add's rounding error kept in lo, four pair copies merged in copy order and rounded once. FP32 operations only; the cross-CTA order still varies. The workspace comes from `..._nondet_train_backward_workspace`; its size depends on `C` and the route, not on `S` or `bwd_grid_mult`. An FP32 `dweight` must be 16-byte aligned. | Returns `(dx, dweight)`, plus `dbias` with `bias`. S == 0 returns an empty `dx` and zero gradients. Never share a workspace between concurrent launches. |

Other public names:

- `ARCHES`, `BWD_GRID_MULTS` (`(1, 2, 3, 4)`), `DWEIGHT_SUMS` (`("fp32", "fp32_pair")`), `SUPPORTED_COMPUTE_CAPABILITIES`, `CakeCausalConv1dNondetFunction`.
- `is_cake_causal_conv1d_nondet_train_supported(device)`.
- `cake_causal_conv1d_nondet_train_backward_workspace(S, C, device, *, bwd_grid_mult=1)` and `..._bytes(...)`.
- `cake_causal_conv1d_nondet_train_forward_workspace(S, C, device)` and `..._bytes(...)`: as det, including the larger size at C = 520 to 568 from 0.2.4.

### `cake_pplx.cake_causal_conv1d_nondet_train_kcp`

The backward overlaps the halo exchange: the main program runs before the wait, without a cooperative grid. It adds FP32 `dweight` into zeroed accumulators with unordered reductions, and a tail-row program writes `dx` for the last rows of each fragment. The nondet package makes no bit promise. Under deterministic algorithms the backward raises `RuntimeError` before any exchange (`UserWarning` with `warn_only`). Source: [`kernels/causal_conv1d/cake_pplx/cake_causal_conv1d_nondet_train_kcp.py`](../kernels/causal_conv1d/cake_pplx/cake_causal_conv1d_nondet_train_kcp.py).

| Name | Signature | Inputs | Returns / notes |
|---|---|---|---|
| `cake_causal_conv1d_nondet_kcp` | `(x_local, weight, ctx, *, recompute_handle=None, cooperative=None, bwd_reserve_sms=None, dweight_sum=None)` | As det KCP. `ctx` is a `CakeKDADetKcpContext` from `cake_pplx.cake_kda_det_train_kcp.cake_kda_det_train_kcp_build_context`. `cooperative=True` raises `ValueError`. `bwd_reserve_sms`: `None` (= 0) or an int K >= 0 below the SM count. `dweight_sum`: `None` (= `"fp32"`) or `"fp32_pair"`. | Autograd. Returns BF16 `y`. The backward gives BF16 `dx` and this rank's FP32 `dweight` partial. With K >= 1, the one-program backward runs after the exchange with K x ctas_per_sm fewer CTAs. Each rank may choose its own K. A bool or non-int K raises `TypeError`; a negative or too-large K raises `ValueError`, as does K >= 1 on `sm_107a`. `dweight_sum="fp32_pair"` (opt-in) sums the overlapped main program's accumulator copies of a tile as an FP32 (hi, lo) pair and rounds once; the workspace and the exchanges are the same. It needs the overlapped backward: with K >= 1 the call raises `ValueError`, and a backward with tokens and C > 65536 raises `ValueError`. |
| `cake_causal_conv1d_nondet_kcp_multi` | `(xs, weights, ctx, *, recompute_handle=None, cooperative=None, bwd_reserve_sms=None, dweight_sum=None)` | As det multi. One `cooperative` / `bwd_reserve_sms` / `dweight_sum` value applies to all tensors. | Returns n BF16 outputs. Each tensor runs the single call's programs. Each `dweight` is its own unordered FP32 partial. |

Other public names: `CakeCausalConv1dNondetKcpRecomputeHandle`. Any object with an `entries` dict works as a handle for either package.

**Options and environment.**

- `silu_grad_preactivation` (`"bf16"` by default, or `"fp32"`): where SiLU' is evaluated. Under `"bf16"`, a call with FP32 taps evaluates it at the BF16-rounded preactivation, and a call with BF16 taps at the unrounded FP32 preactivation (routes `bwd_dt_bf16_taps` / `bwd_bias_bf16_taps`). Under `"fp32"`, every call evaluates it at the unrounded FP32 preactivation.
- `bwd_grid_mult` (1 by default, up to 4): an opt-in finer static token partition of the backward, which helps overlap with kernels on other streams. In det, the `dweight` bits change with each value and the workspace grows. In nondet, the workspace size does not change. At the nondet rows of `cake_jit.SM_WIDE_ROWS` each CTA holds most of an SM's registers and shared memory, so a kernel resident on another stream can push CTAs of that backward into a later wave. With a kernel that holds one SM's shared memory on another stream, the backward at S = 261393, C = 576 took 1.821 / 1.904 / 1.891 times its time alone on sm_100a / sm_103a / sm_107a (1.459 / 1.509 / 1.501 with the program the rules pick without `SM_WIDE_ROWS`).
- `dweight_sum` (`"fp32"` by default, or the opt-in `"fp32_pair"`): the sum form of `dweight` / `dbias`. `"fp32_pair"` runs the FP32 (hi, lo) pair programs (single rank: with `silu_grad_preactivation="fp32"` only; nondet KCP: the overlapped backward only, so no `bwd_reserve_sms` and at most 65536 channels). det KCP has no `dweight_sum` keyword and no pair opt-in. Other combinations and other strings raise `ValueError`, and a non-string raises `TypeError` (`None` is the nondet KCP default); no call falls back to another program. The default never selects a pair program.
- `workspace=`: backend `cake_jit.FALLBACK_ROWS` and `cake_jit.DIRECT_ROWS` hold the forward rows at C = 576 to 768, by architecture, route and `C`. Each row gives its own token range. From 0.2.6, every architecture and route has a row at each `C` = 576 to 768 (0.2.5 had none for sm_100a `fwd_bias` at `C` = 640, sm_103a `fwd_bias` at `C` = 576 and 712, and sm_103a `fwd` at `C` = 768). For most `C` the first range starts at S = 1. The sm_103a `fwd_bias` rows at C = 720 to 768 are in `FALLBACK_ROWS` only and start at S = 13824 (C = 736: 14336). Outside a row's range, and where an architecture, route and `C` have no row, the forward runs as in 0.2.4: it takes its counters from the workspace. Inside a row's range, the forward of either package runs a program that takes no counter buffer, eagerly or inside a CUDA Graph capture: a passed `workspace` is checked and not read, the call zeroes no buffer, and an eager call there refills none. So after eager forward calls inside these rows' token ranges only, a captured nondet forward call outside a row's range (or where an architecture, route and `C` have no row) finds no ready forward buffer and zeroes one of its own: one fill node per call (see below). Elsewhere, when omitted, the forward of either package keeps one zeroed counter buffer per (device, current stream), and the nondet backward keeps one zeroed accumulator buffer per (device, current stream). The det backward allocates a new partial-row buffer on every call, eager or captured; that buffer needs no zeroing. Inside a CUDA Graph capture, each det forward call zeroes a counter buffer of its own. nondet takes one pre-zeroed buffer per capture and stream for the forward and one for the backward, so the graph has no fill node for them. The package zeroes these buffers in refills on earlier eager calls that omit `workspace=`, separately for the forward and the backward, at the size of the largest call whose refill succeeded, and keeps at most 4 idle ones of each kind. A graph holds its buffer until the graph is destroyed and its launches have completed. The package takes a released buffer back only at its next refill on that device; until then the released buffer stays allocated. An eager call that omits `workspace=` refills its kind (forward or backward) when it needs a larger buffer than the kind's ready size on that device, which the latest of these events sets: a capture that asks for a buffer of that kind sets it to zero, a refill that succeeds sets it to the size of its buffers, and a refill that runs out of memory sets it to that call's need. A refill that runs out of memory keeps no new buffer and keeps the buffer size, but it has already taken back the device's released buffers, dropping each one that is smaller than its kind's buffer size (for the refill's own kind, that call's need if it is larger) or that would exceed 4 idle buffers of its kind. A refill that fails for any other reason turns that kind's pre-zeroed buffers off on that device for the rest of the process: its captured calls then zero buffers of their own. Replays of one graph must not overlap. Without a ready buffer of the needed kind and size, each captured nondet call zeroes a buffer of its own, one fill node per call, forward and backward alike. A buffer is ready once the zero fill that an earlier eager call queued has completed: a capture that begins right after that call without a device synchronize can find none ready (`torch.cuda.graph` synchronizes before it captures). Without `cuda.bindings` (no capture id), each captured nondet call zeroes a buffer of its own. nondet allocates its per-stream and pre-zeroed buffers from its own `torch.cuda.MemPool` per device; that memory stays cached in that pool and is not returned to the caching allocator's other pools.
- Programmatic dependent launch (PDL): nondet's single-rank forward and backward programs, except the `"fp32_pair"` programs, take it per call by architecture and by mode (eager or inside a CUDA Graph capture; backend `cake_jit.PDL_EAGER` / `PDL_GRAPH`). A launch may then begin while the stream's preceding kernel finishes; each program waits for that kernel before it reads or writes global memory. The single-rank backward launches without it, in both modes, at the rows of backend `cake_jit.PDL_OFF_ROWS` (by route, `C` and `S`): the default `bwd` route at C = 1000 to 1008 from S = 187755 and at C = 1016 from S = 155648 on sm_100a, and at C = 1000 to 1008 from S = 131073 and at C = 1016 from S = 125171 on sm_103a. From these token counts on, the measured training step and checkpointed step were not slower with that launch made without PDL. The forward keeps it at those rows. The nondet KCP programs launch without it.
- KCP: `recompute_handle` must be one per call and micro-step, created outside the outermost checkpoint region, and every rank passes one or none.
- CUDA Graph: build every library of the architecture first (backend `cake_jit.specs_for_arch`) so that no JIT build runs inside capture.
- `CAKE_CONV1D_FULL_PLAN_SWEEP=1` is a test-only variable. The KCP backend reads no environment variable, and its `_set_test_knobs` are private test hooks.

**Cross-family.** The KCP entry points use a KCP context built by another family. The det facade asks for "the calling family's KCP context builder". The nondet facade names `CakeKDADetKcpContext` from `cake_pplx.cake_kda_det_train_kcp.cake_kda_det_train_kcp_build_context`. A recompute handle may carry the layer's `"kda"` entry next to `"conv"`. The causal_conv1d `pyproject` does not list kda as a dependency; its dependencies are `torch`, `apache-tvm-ffi>=0.1.11,<0.2` and `cuda-python`.

**Example.**

```python
import torch
from cake_pplx.cake_causal_conv1d_det_train import (
    cake_causal_conv1d_det, cake_causal_conv1d_det_train_forward)

S, C = 1000, 512
x = torch.randn(S, C, device="cuda", dtype=torch.bfloat16, requires_grad=True)
w = torch.randn(C, 4, device="cuda", dtype=torch.float32, requires_grad=True)
cu = torch.tensor([0, 400, 1000], device="cuda", dtype=torch.int32)

y = cake_causal_conv1d_det(x, w, cu, None, silu_grad_preactivation="bf16")
assert torch.equal(y.detach(), cake_causal_conv1d_det_train_forward(x.detach(), w.detach(), cu))
y.backward(torch.randn_like(y))  # x.grad BF16, w.grad FP32
# nondet: from cake_pplx.cake_causal_conv1d_nondet_train import cake_causal_conv1d_nondet
```

**Caveats.**

- Only the nondet facade re-exports `BWD_GRID_MULTS`; the det facade's `__all__` does not include it.
- KCP paths accept only FP32 taps and no bias, and they follow the legacy channel tile rule.
- `is_*_supported` checks only the base routes. Optional routes can still raise `ValueError` per call.
- Do not share workspaces between launches that may run concurrently, for example on two streams.
- Ranks are not compared. A mismatch in KCP multi tensor count, channels or order, or in the recompute handle, hangs or silently corrupts data.
- The 3e-5 rel-L2 figure comes from the source docstrings. The package tests judge the L2 and max error of every output against an FP64 oracle, relative to a PyTorch BF16 implementation of the same operation, over 32 random draws.

---

## kda

**Computes.** Packed KDA (Kimi Delta Attention) chunked training forward and backward over one packed batch of documents.

- Each document starts from a zero state, and no final state is returned.
- The forward returns `out` (BF16, like `v`). The backward returns BF16 `dq`, `dk`, `dv`, `dbeta` and FP32 `dg`.
- **Raw gate mode** (`use_gate_in_kernel=True`) activates the gate in the kernel as `lower_bound * sigmoid(exp(A_log) * (g + dt_bias))`, and the backward also returns `dA_log` and `ddt_bias`.
- With `use_qk_l2norm_in_kernel=True`, the kernel normalizes `q`/`k` as `bf16(x * r)`, where `r = 1/sqrt(sum(x^2) + 1e-6)` in FP32.
- The KCP modules split the batch over a `torch.distributed` group. Collectives carry each document's state recurrence across rank boundaries.

**Architectures.** Every `build.json` declares `sm_100a`, `sm_103a` and `sm_107a`, and the JIT never adds a generic `sm_XX` target. `is_cake_kda_*_train_supported` checks for compute capability 10.0, 10.3 or 10.7.

**Inputs (single rank).**

- `q`, `k`, `v`: BF16 `[1, S, H, 128]` or `[S, H, 128]`, contiguous. The caller must L2-normalize `q`/`k` unless `use_qk_l2norm_in_kernel=True`.
- `g`: FP32 `[1, S, H, 128]`, a preactivated natural-log decay gate.
- `beta`: BF16 `[1, S, H]`, after the sigmoid.
- `cu_seqlens`: int32 `[num_docs + 1]` on the device; first value 0, last value S.
- `H` must be in 1..64.
- `scale` defaults to `128 ** -0.5` (`DEFAULT_SCALE`).
- Outputs have the same rank as the inputs.

In raw gate mode, `g` is the raw gate: BF16 `[1, S, H, 128]`. It may be a strided view with unit innermost stride, token and head strides that are multiples of 8, and 16-byte alignment. `A_log` is FP32 `[H]`, `dt_bias` is FP32 `[H*128]` or `[H, 128]`, and `lower_bound` must be finite with `-5.0 <= lower_bound < 0.0`.

With the in-kernel L2 norm, `q`/`k` may be strided views: head stride 128, token stride a multiple of 8 and >= `128*H`, 16-byte aligned.

`doc_lengths` are host ints (`len == num_docs`, `sum == S`). They are required with a batched arena.

### `cake_pplx.cake_kda_det_train`

Det, single rank. The same inputs, hardware and plan give bitwise the same outputs and gradients. No environment variable is read; every switch is a keyword argument. Source: [`kernels/kda/cake_pplx/cake_kda_det_train.py`](../kernels/kda/cake_pplx/cake_kda_det_train.py).

| Name | Signature | Inputs | Returns / notes |
|---|---|---|---|
| `cake_kda_det` | `(q, k, v, g, beta, cu_seqlens, *, scale=None, workspace=None, save_mode=None, arena=None, doc_lengths=None, use_gate_in_kernel=False, A_log=None, dt_bias=None, lower_bound=None, use_qk_l2norm_in_kernel=False)` | As above | Autograd node that works with `torch.utils.checkpoint`. Returns `out`. Gradients go to `q`, `k`, `v`, `g` and `beta`. In raw gate mode, they go to the raw BF16 `g` and to `A_log`/`dt_bias` (FP32). |
| `cake_kda_det_train_forward` | `(q, k, v, g, beta, cu_seqlens, *, scale=None, save_for_backward=True, workspace=None, save_mode=None, arena=None, doc_lengths=None, use_gate_in_kernel=False, A_log=None, dt_bias=None, lower_bound=None, use_qk_l2norm_in_kernel=False)` | As above | Returns `(out, saved)`. `saved` is a versioned `CakeKDADetChunkSaved`, or `None`. `save_mode='light'` gives saved-set version 4. `'tape'` gives version 2, which adds the state tape and `v_new` so the backward skips its whole recompute. |
| `cake_kda_det_train_backward` | `(q, k, v, g, beta, cu_seqlens, dout, *, saved=None, scale=None, workspace=None, arena=None, doc_lengths=None, use_gate_in_kernel=False, A_log=None, dt_bias=None, lower_bound=None, use_qk_l2norm_in_kernel=False)` | `dout` BF16, like `v`; `saved` is version 4 or 2. A saved set from the other gate or L2-norm mode, or with another `lower_bound`, is refused. | Returns `(dq, dk, dv, dbeta, dg)`. In raw gate mode it returns `(dq, dk, dv, dbeta, dg, dA_log, ddt_bias)`, where `dg` is BF16, `dA_log` is FP32 `[H]` and `ddt_bias` is FP32 shaped like `dt_bias`. `saved=None` recomputes the saved set (checkpoint replay). |

Other public names:

- `DEFAULT_SCALE` (`128**-0.5`).
- `cake_kda_det_train_arena`: builds a `CakeKDADetArena` lower-memory option (`lean`, `tail_passes`, `batch_tokens`, `max_docs_per_batch`). Off by default.
- `cake_kda_det_train_workspace`: allocates a caller-owned, reusable `CakeKDADetWorkspace` for `(S, num_docs, H, device)`. It takes `backward`, `save_mode`, `arena`, `doc_lengths` and the gate / L2-norm modes.
- `cake_kda_det_train_workspace_bytes`: workspace bytes for a problem (`sm_count`, `arena`, `doc_lengths`, modes).
- `cake_kda_det_train_saved_bytes`: saved-set bytes (`Aqk`/`Akk` at `256*S*H`, plus tables, per-version tensors and the L2-norm `rstd`).
- `is_cake_kda_det_train_supported`.

### `cake_pplx.cake_kda_nondet_train`

Nondet, single rank. The schedule has a fixed order, so repeated calls on the same inputs and path give bitwise the same results. `out`, `dq`, `dk`, `dv` and `dbeta` are bitwise equal to det, and `dg` differs from det only by FP32 rounding. In raw gate mode and with the in-kernel L2 norm, every gradient is bitwise equal to det. No launch is unordered, so deterministic algorithms raise no alert. Source: [`kernels/kda/cake_pplx/cake_kda_nondet_train.py`](../kernels/kda/cake_pplx/cake_kda_nondet_train.py).

| Name | Signature | Inputs | Returns / notes |
|---|---|---|---|
| `cake_kda_nondet` | `(q, k, v, g, beta, cu_seqlens, *, scale=None, workspace=None, save_mode=None, arena=None, doc_lengths=None, use_gate_in_kernel=False, A_log=None, dt_bias=None, lower_bound=None, use_qk_l2norm_in_kernel=False, checkpoint_handoff="auto")` | As above. `checkpoint_handoff` is `'auto'`, `'on'` or `'off'`; any other value raises `ValueError`. | Autograd. Returns `out`. Under `torch.utils.checkpoint`, the re-run forward hands its state tapes to the backward, which then skips the recurrence recompute. Gradients do not change. |
| `cake_kda_nondet_train_forward` | Same as `cake_kda_det_train_forward` | A det workspace or arena raises `ValueError`. | Returns `(out, saved)` with a `CakeKDANondetChunkSaved`. `out` and `saved` are bitwise det. |
| `cake_kda_nondet_train_backward` | Same as `cake_kda_det_train_backward` | A det saved set or workspace raises `ValueError`. | Returns 5 tensors, or 7 in raw gate mode. |

Other public names: `DEFAULT_SCALE`, `cake_kda_nondet_train_arena`, `cake_kda_nondet_train_workspace` (the det entries refuse it), `cake_kda_nondet_train_workspace_bytes`, `cake_kda_nondet_train_saved_bytes`, `is_cake_kda_nondet_train_supported`.

### `cake_pplx.cake_kda_det_train_kcp`

Det + KCP. The packed batch is split over a process group, and each document's state recurrence crosses rank boundaries through the plan's collectives. Results are bitwise reproducible for a given setup. The facade validates nothing and forwards every keyword. Source: [`kernels/kda/cake_pplx/cake_kda_det_train_kcp.py`](../kernels/kda/cake_pplx/cake_kda_det_train_kcp.py).

Shard contract: this rank's tokens in partition order. `q`/`k`/`v` are BF16 `[1, S_local, H, 128]`, `g` is FP32, and `beta` is BF16 `[1, S_local, H]`. In raw gate mode, `g` is the rank's raw BF16 gate, and `A_log` / `dt_bias` are the full parameters (the same tensors on every rank).

| Name | Signature | Inputs | Returns / notes |
|---|---|---|---|
| `cake_kda_det_kcp` | `(q, k, v, g, beta, ctx, scale=None, save_mode="light", boundary_mode="saved", workspace=None, *, recompute_handle=None, checkpoint_keep=None, use_qk_l2norm_in_kernel=False, **options)` | Shard; `ctx` from `build_context`; `checkpoint_keep` `'states'` or `'transfer'` | Autograd. Returns BF16 `out` for this rank's tokens. `recompute_handle` lets the checkpoint recompute skip the forward state exchange, with bitwise identical results. |
| `cake_kda_det_train_kcp_partition` | `(cu_seqlens_global_cpu, world_size, layout="contiguous", fragments=None, *, cut_points=None)` | Host int32 `[num_docs+1]` (tensor or sequence); `layout` is `'contiguous'`, `'zigzag'` or `'zigzag_per_document'`; `cut_points` (2*world_size+1 per document) applies only to `zigzag_per_document` | Returns a `CakeKDADetKcpPartition`. `fragments` gives an explicit ragged layout. |
| `cake_kda_det_train_kcp_build_context` | `(part, group, rank, device, conv_kernel_size=4, *, sm_count=None, heads=None, stream=None)` | `part` is the same on every rank; `heads` are head counts to pre-plan; `stream` is used for the async table copy | Returns `CakeKDADetKcpContext`: device tables, state and halo exchange collectives, and the segment plan. Reusable across calls with the same batch layout. The host never waits for device work. |
| `cake_kda_det_train_kcp_forward` | `(q, k, v, g, beta, ctx, *, scale=None, save_for_backward=True, save_mode="light", boundary_mode="saved", workspace=None, use_qk_l2norm_in_kernel=False, **options)` | Shard; `save_mode` `'light'` or `'tape'` | Returns `(out, saved)` with a `CakeKDADetKcpSaved`, or `None`. `boundary_mode='saved'` keeps the received boundary states. `'replay'` saves nothing, and the backward replays the forward's exchanges. |
| `cake_kda_det_train_kcp_backward` | `(q, k, v, g, beta, ctx, dout, *, saved=None, scale=None, save_mode="light", workspace=None, use_qk_l2norm_in_kernel=False, **options)` | `dout` BF16; `spec3_probe` must be the same on every rank | Returns `(dq, dk, dv, dbeta, dg)`, or 7 tensors in raw gate mode, where `dA_log`/`ddt_bias` are this rank's partial sums. `saved=None` replays the forward. |

The runtime signatures append these forwarded keyword-only options: `use_gate_in_kernel`, `A_log`, `dt_bias`, `lower_bound`, `save_transfer`, `exchange='auto'`, `head_groups=1`, `offset_class='fla'`, `debug_sequence`, `spec3_probe='off'`, `graph='off'`, `_kcp_test_knobs`.

Other public names:

- `DEFAULT_SCALE`.
- Classes, loaded lazily: `CakeKDADetKcpContext`, `CakeKDADetKcpOptions`, `CakeKDADetKcpPartition`, `CakeKDADetKcpRecomputeHandle` (one per layer and micro-step), `CakeKDADetKcpSaved`, `CakeKDADetKcpWorkspace`. The workspace's `reserve(ctxs, H)` sizes it for several contexts. Its constructor options are `local_arena='lean'`, `lean_passes`, `use_gate_in_kernel` and `use_qk_l2norm_in_kernel`.
- `cake_kda_det_train_kcp_workspace_bytes`: takes a `ctx` or a list of contexts.
- `cake_kda_det_train_kcp_saved_bytes`: can include the convolution's history rows (`with_conv`, C=8192) or a recompute-handle tier.
- `cake_kda_det_train_kcp_comm_bytes`: bytes sent and received per collective and per boundary in one step.
- `cake_kda_det_train_kcp_reduce_gate_param_grads`: sums the raw-gate `dA_log`/`ddt_bias` partials bitwise-reproducibly (all-gather, then an FP32 sum in rank order).

### `cake_pplx.cake_kda_nondet_train_kcp`

Nondet + KCP. The KCP programs and exchanges are det's, and the rank-local chunk stages are `cake_kda_nondet_train`'s. Results are bitwise reproducible at the same topology. `out`/`dq`/`dk`/`dv`/`dbeta` are bitwise det, and `dg` differs by FP32 rounding. These bitwise statements hold for the light saved set and the hand-off tape set. With `saved_rows='intra'` or `'wy'` (and `'auto'` where the hand-off engages), gradients are held to the light path's FP64 error budget, not bitwise. Raw gate and in-kernel L2-norm modes are bitwise det with the default saved set. Source: [`kernels/kda/cake_pplx/cake_kda_nondet_train_kcp.py`](../kernels/kda/cake_pplx/cake_kda_nondet_train_kcp.py).

| Name | Signature | Inputs | Returns / notes |
|---|---|---|---|
| `cake_kda_nondet_kcp` | `(q, k, v, g, beta, ctx, scale=None, save_mode="light", boundary_mode="saved", workspace=None, *, recompute_handle=None, checkpoint_keep=None, checkpoint_handoff="auto", saved_rows="auto", use_qk_l2norm_in_kernel=False, **options)` | Shard as det KCP; `ctx` from `cake_kda_nondet_train_kcp_build_context` | Autograd. Returns BF16 `out`. `checkpoint_handoff='auto'` engages only on verified devices, for H in `KCP_HANDOFF_HEADS` and `H*(total tokens // world size) >= 393216`. `saved_rows` is `'auto'`, `'none'`, `'intra'`, `'wy'` or `'tape'`. Same forwarded options as det KCP. |
| `cake_kda_nondet_train_kcp_partition` | `(cu_seqlens_global_cpu, world_size, layout="contiguous", fragments=None, *, cut_points=None)` | As det | Returns `CakeKDANondetKcpPartition`. |
| `cake_kda_nondet_train_kcp_build_context` | `(part, group, rank, device, conv_kernel_size=4, *, sm_count=None, heads=None, stream=None)` | As det | Returns `CakeKDANondetKcpContext`. |
| `cake_kda_nondet_train_kcp_forward` | `(q, k, v, g, beta, ctx, *, scale=None, save_for_backward=True, save_mode="light", boundary_mode="saved", workspace=None, use_qk_l2norm_in_kernel=False, saved_rows="auto", **options)` | `saved_rows` is `'auto'`, `'none'`, `'intra'` or `'wy'` | Returns `(out, saved)` with a `CakeKDANondetKcpSaved`. `'intra'`/`'wy'` add `256*R*H`/`768*R*H` bytes and are refused with `spec3_probe`. |
| `cake_kda_nondet_train_kcp_backward` | `(q, k, v, g, beta, ctx, dout, *, saved=None, scale=None, save_mode="light", workspace=None, use_qk_l2norm_in_kernel=False, **options)` | `dout` BF16 | Returns 5 tensors, or 7 in raw gate mode (with per-rank partials). A det saved set raises `ValueError`. |

Other public names:

- `DEFAULT_SCALE`.
- `CakeKDANondetKcp{Context,Options,Partition,RecomputeHandle,Saved,Workspace}`.
- `cake_kda_nondet_train_kcp_workspace_bytes`.
- `cake_kda_nondet_train_kcp_saved_bytes`: also takes `checkpoint_handoff` (bool) and `saved_rows`.
- `cake_kda_nondet_train_kcp_handoff_bytes`: an upper bound on how much an engaged hand-off raises the checkpoint peak.
- `cake_kda_nondet_train_kcp_handoff_stats`: counters `'save'`, `'bound'` and `'refused_<reason>'`.
- `cake_kda_nondet_train_kcp_comm_bytes`.
- `cake_kda_nondet_train_kcp_reduce_gate_param_grads`.

**Options and environment.**

- `save_mode`: `'light'` (v4) or `'tape'` (v2, 768 B more per token-head). The workspace's `save_mode` sets the default.
- Arena: `lean=True` uses lean backward layouts and cannot serve checkpoint replay. `tail_passes >= 2` splits the backward tail and implies `lean`. `batch_tokens` / `max_docs_per_batch` process groups of whole documents and need `doc_lengths`.
- The workspace must be allocated in the same gate / L2-norm modes as the call.
- KCP keywords and forwarded options are listed above. `exchange` also mentions `'ring'`, and `graph` also mentions `'replay'`.
- No environment variable is read for kernel behavior.

**Cross-family.** The causal_conv1d KCP modules take this module's `ctx`. The context exchanges the convolution history rows (`conv_kernel_size=4`). `saved_bytes`/`comm_bytes` can include those rows (`with_conv`, C=8192). `recompute_handle` accepts any object with an `entries` dict, such as the convolution's handle for the same layer. The KDA docstrings do not name the convolution module.

**Example.**

```python
import torch
from cake_pplx.cake_kda_det_train import cake_kda_det, is_cake_kda_det_train_supported
from cake_pplx.cake_kda_nondet_train import cake_kda_nondet

assert is_cake_kda_det_train_supported()  # compute capability 10.0 / 10.3 / 10.7
lengths, H, D, dev = (63, 1, 65), 2, 128, "cuda"
S = sum(lengths)

def l2n(x):  # caller-side L2 norm (FP32, eps 1e-6), stored as BF16
    return (x * torch.rsqrt((x * x).sum(-1, keepdim=True) + 1e-6)).bfloat16()

q = l2n(torch.randn(S, H, D, device=dev)).requires_grad_()
k = l2n(torch.randn(S, H, D, device=dev)).requires_grad_()
v = torch.randn(S, H, D, device=dev).bfloat16().requires_grad_()
g = (-5.0 * torch.sigmoid(torch.randn(S, H, D, device=dev))).float().requires_grad_()  # FP32 log-decay
beta = torch.sigmoid(torch.randn(S, H, device=dev)).bfloat16().requires_grad_()
cu_seqlens = torch.tensor([0, 63, 64, 129], dtype=torch.int32, device=dev)
dout = torch.randn(S, H, D, device=dev).bfloat16()

out = cake_kda_det(q, k, v, g, beta, cu_seqlens)   # BF16 [S, H, 128]
out.backward(dout)                                  # q.grad ... beta.grad (dg FP32)

# non-deterministic package, same call shape
out_nd = cake_kda_nondet(q, k, v, g, beta, cu_seqlens)
```

**Caveats.**

- The facades validate nothing; the backend does the checks.
- H must be in 1..64, and the head dim is 128. The state is zero-initialized per document, and no final state is returned or accepted.
- det and nondet refuse each other's workspaces, saved sets and arenas with `ValueError`.
- A lean arena cannot serve checkpoint replay (`saved=None`). On the segmented route, a batched arena is refused on rows with at most four (document, head) chains. Under batching, an exactly-zero `u`/`w` entry may change sign.
- KCP with `use_qk_l2norm_in_kernel=True` raises `NotImplementedError` on every rank until the device has the single-rank in-kernel normalization programs. KCP `saved_bytes` with the L2 norm on may also raise `NotImplementedError` in a gate mode where it is unavailable.
- In KCP raw gate mode, `dA_log`/`ddt_bias` are per-rank partial sums. Use `cake_kda_*_train_kcp_reduce_gate_param_grads`, because an NCCL all-reduce does not fix the summation order.
- `recompute_handle` is a collective contract. Every rank must pass a handle, or none, at the same checkpoint nesting level. A mismatch is not checked and hangs or corrupts data. The handle is refused with `graph='replay'`, inside CUDA-graph capture, with `debug_sequence`, with `spec3_probe` and with `boundary_mode='replay'`.
- Nondet KCP `checkpoint_handoff='on'` has bitwise evidence only on devices in `KCP_HANDOFF_VERIFIED` / `KCP_HANDOFF_VERIFIED_MODES`.
- In the source, the KCP entries are defined with `**options`. At run time their signatures list the forwarded options (`_forwarding_signature`).

---

## gdn

**Computes.** The GDN (Gated DeltaNet) chunked gated delta rule for training, with FLA's `chunk_gated_delta_rule` signature: head dim 128, chunk 64, and `HV` value heads that are a multiple of the `H` q/k heads.

- The forward returns `o` (like `v`) and, optionally, an FP32 final state `[N, HV, 128, 128]`.
- The backward gives gradients to `q`, `k`, `v`, `g`, `beta` and `initial_state`.
- With `use_qk_l2norm_in_kernel`, the kernel normalizes `q`/`k` as `bf16(x*r)`, where `r = 1/sqrt(sum x^2 + 1e-6)`, and backpropagates through that FP32 form.
- The KCP modules split the packed batch over a process group and pass state across ranks.

**Architectures.** All four backend `build.json` files declare `sm_100a`, `sm_103a` and `sm_107a`. `is_*_supported` checks for compute capability 10.0, 10.3 or 10.7.

**Inputs (single rank).**

- `q`, `k`: BF16 `[B, T, H, 128]`.
- `v`: BF16 `[B, T, HV, 128]`, where `HV` is a positive multiple of `H`.
- `g`: `[B, T, HV]` natural-log decay, computed in FP32.
- `beta`: `[B, T, HV]`, FP32 or BF16. Its gradient comes back in its own dtype.
- A packed batch uses `B=1` with `cu_seqlens`: 1-D int32/int64 offsets on the device. `cu_seqlens_cpu` is the same offsets on the host, and passing it selects the host-table route.
- `initial_state`: FP32 `[N, HV, 128, 128]`.
- `scale` defaults to `1/sqrt(128)`.
- All tensors must be on `q`'s device.

### `cake_pplx.cake_gdn_det_train`

Every launch has a fixed order. With the same inputs, hardware, layout and plan, outputs and gradients repeat bit for bit. `torch.use_deterministic_algorithms(True)` changes nothing. Source: [`kernels/gdn/cake_pplx/cake_gdn_det_train.py`](../kernels/gdn/cake_pplx/cake_gdn_det_train.py).

| Name | Signature | Inputs | Returns / notes |
|---|---|---|---|
| `cake_gdn_det` | `(q, k, v, g, beta, scale=None, initial_state=None, output_final_state=False, cu_seqlens=None, cu_seqlens_cpu=None, use_qk_l2norm_in_kernel=False, *, scan_config=None, segment_chunks=None, finalize=None, warmup_log2=None, save_state=True, host_cache=True, dhu_scan=None, prep=None, lpt=True, streams=2)` | As above | Autograd node that works with `torch.utils.checkpoint`. Returns `(o, final_state)`. `final_state` is FP32 `[N, HV, 128, 128]`, or `None` unless `output_final_state`. Inputs are checked before the first launch (`ValueError`). |

Other public names:

- `CakeGDNDetFunction`: loaded lazily.
- `is_cake_gdn_det_train_supported`: returns `False` without CUDA, for a non-CUDA device, or on any exception.

### `cake_pplx.cake_gdn_nondet_train`

Every launch has a fixed order today, so two runs on one device give the same bits, but the contract does not promise run-to-run bits. Results are not bitwise det. The tests measure each output's error against an FP64 reference over 32 random inputs, beside the error of PyTorch on the same BF16 inputs, and record whether it is within 1.005 times PyTorch's (L2 norm: the median and the largest over the inputs; max norm: input by input, on at least half of them) as a `TIER` line. A miss is recorded and does not fail a test ([`kernels/gdn/tests/cake_naive.py`](../kernels/gdn/tests/cake_naive.py)). Deterministic algorithms run unchanged. Source: [`kernels/gdn/cake_pplx/cake_gdn_nondet_train.py`](../kernels/gdn/cake_pplx/cake_gdn_nondet_train.py).

| Name | Signature | Inputs | Returns / notes |
|---|---|---|---|
| `cake_gdn_nondet` | Same as `cake_gdn_det`, plus `tail=None` | `tail` forces the backward tail form: `None`, `'tail'`, `'tail_g2_f32x'`, `'tail_g3_f32x'`, `'tail_g2'`, `'tail_g3'` or `'unfused'` | Autograd. Returns `(o, final_state)`. Inputs are checked before the first launch. |

Other public names:

- `CakeGDNNondetFunction`.
- `is_cake_gdn_nondet_train_supported`: imports the package first, so it raises if the package fails to import.

### `cake_pplx.cake_gdn_det_train_kcp`

Every launch has a fixed order. Results are bitwise reproducible run to run at the same topology, and the saved and replay modes agree bitwise. The backward refuses a nondet saved set or workspace (`ValueError`). Source: [`kernels/gdn/cake_pplx/cake_gdn_det_train_kcp.py`](../kernels/gdn/cake_pplx/cake_gdn_det_train_kcp.py).

Per-rank operands (`T_r` local tokens, in fragment order):

- `q`, `k`: BF16 `[1, T_r, H_qk, 128]`, either L2-normalized by the caller or raw with `use_qk_l2norm_in_kernel=True`.
- `v`: BF16 `[1, T_r, HV, 128]` with `HV % H_qk == 0`. The production shape is `(16, 48)`.
- `g`: FP32 `[1, T_r, HV]`, preactivated, with no clamp.
- `beta`: FP32 `[1, T_r, HV]`.
- Strided packed-QKV views (token stride `(2 H_qk + HV)*128`) are accepted without a copy. The batch axis is optional, and `T_r` may be 0.

| Name | Signature | Inputs | Returns / notes |
|---|---|---|---|
| `cake_gdn_det_kcp` | `(q, k, v, g, beta, ctx, *, scale=None, save_mode="light", boundary_mode="saved", workspace=None, graph="off", options=None, recompute_handle=None, use_qk_l2norm_in_kernel=False, **knobs)` | Per-rank operands | Autograd. Returns BF16 `out`, with gradients to `q`, `k`, `v`, `g` and `beta`. Every saved tensor goes through `save_for_backward`. Knobs include `reuse_recompute_tapes` (`'auto'`, `True` or `False`) and `recompute_output` (`'auto'` or `'always'`). `recompute_handle` lets the checkpoint recompute run without cross-rank communication. |
| `cake_gdn_det_train_kcp_partition` | `(cu_seqlens_global_cpu, world_size, layout="contiguous", fragments=None, *, cut_points=None)` | CPU int64 tensor or a sequence of ints, first value 0 | Returns `CakeGDNDetKcpPartition`: a per-rank fragment table (doc, off, length, prev, next) that is reused across layers. `layout` is `'contiguous'`, `'zigzag'` or `'zigzag_per_document'`. The total need not be divisible by W, and zero-length fragments are legal. |
| `cake_gdn_det_train_kcp_build_context` | `(part, group, rank, device, conv_kernel_size=4, *, sm_count=None)` | `group` is the caller's process group | This rank's view: local `cu_seqlens` (int32, on the device), fragment tables, exchange plan, convolution halo plan (at most `conv_kernel_size-1` history rows per continuing fragment) and segment plan. No tensor contents are read. |
| `cake_gdn_det_train_kcp_forward` | `(q, k, v, g, beta, ctx, *, scale=None, save_for_backward=True, workspace=None, options=None, use_qk_l2norm_in_kernel=False, **knobs)` | Per-rank operands | Returns `(out, saved)`. `out` is a fresh BF16 tensor. `saved` is `None` with `save_for_backward=False` or `boundary_mode='replay'`. |
| `cake_gdn_det_train_kcp_backward` | `(q, k, v, g, beta, ctx, dout, *, saved=None, scale=None, workspace=None, options=None, use_qk_l2norm_in_kernel=False, **knobs)` | `dout` BF16, like `v`. A saved set made with the other `use_qk_l2norm_in_kernel` value is refused. | Returns fresh `(dq, dk, dv, dbeta, dg)`: BF16 `dq`/`dk`/`dv` and FP32 `dbeta`/`dg`. The value heads' `dq`/`dk` contributions are reduced in FP32 and rounded once. `saved=None` does checkpoint replay. |

Other public names:

- `CakeGDNDetKcp{Context,Function,Options,Partition,RecomputeHandle,Saved,Workspace}`: loaded lazily.
- `cake_gdn_det_train_kcp_packed_qkv_views`: zero-copy head views of a packed `[..., T, (2 H_qk + HV)*128]` row. Needs `qkv.stride(-1) == 1`. The backward returns a packed gradient row with no concatenation.
- `cake_gdn_det_train_kcp_workspace_bytes`: for `(ctx, H_qk, HV)`.
- `cake_gdn_det_train_kcp_saved_bytes`: per layer and rank.
- `cake_gdn_det_train_kcp_comm_bytes`: bytes per collective (`c1`, `c2`, `b1`, `b2`, `halo_fwd`, `halo_bwd`) and per boundary, with C=10240 convolution channels.
- `is_cake_gdn_det_train_kcp_supported`: returns `False` on any exception.
- `DEFAULT_SCALE` (`128**-0.5`): defined at module level but not in `__all__`.

### `cake_pplx.cake_gdn_nondet_train_kcp`

Every launch has a fixed order today, but run-to-run bits are not promised. This module keeps det's contract, not det's bits. The tests record each output's error against an FP64 reference beside the BF16 naive implementation's, as for `cake_gdn_nondet_train`. An options instance from another package raises `TypeError`, and a det saved set raises `ValueError`. Use this package's own context. Source: [`kernels/gdn/cake_pplx/cake_gdn_nondet_train_kcp.py`](../kernels/gdn/cake_pplx/cake_gdn_nondet_train_kcp.py).

| Name | Signature | Inputs | Returns / notes |
|---|---|---|---|
| `cake_gdn_nondet_kcp` | `(q, k, v, g, beta, ctx, *, scale=None, save_mode=None, boundary_mode="saved", workspace=None, graph="off", options=None, recompute_handle=None, **knobs)` | Same per-rank contract as det KCP | Autograd. `save_mode=None` resolves to `'light_akk'` when `use_qk_l2norm_in_kernel=True`, and to `'light'` otherwise. `use_qk_l2norm_in_kernel` is forwarded through `**knobs` and appears in the lazily resolved `__signature__`. |
| `cake_gdn_nondet_train_kcp_partition` | `(cu_seqlens_global_cpu, world_size, layout="contiguous", fragments=None, *, cut_points=None)` | As det | Returns `CakeGDNNondetKcpPartition`. Same algorithm as det. |
| `cake_gdn_nondet_train_kcp_build_context` | `(part, group, rank, device, conv_kernel_size=4, *, sm_count=None, graph_replay=False)` | `graph_replay=True` builds the context for `graph='replay'` calls; a non-bool value raises `TypeError` | This rank's context, with this package's own head plan. |
| `cake_gdn_nondet_train_kcp_forward` | `(q, k, v, g, beta, ctx, *, scale=None, save_for_backward=True, workspace=None, options=None, **knobs)` | As det KCP | Returns `(out, saved)`. Only this package's backward can read the saved set. `save_mode` adds `'light_akk'`. |
| `cake_gdn_nondet_train_kcp_backward` | `(q, k, v, g, beta, ctx, dout, *, saved=None, scale=None, workspace=None, options=None, **knobs)` | `dout` BF16 | Returns `(dq, dk, dv, dbeta, dg)`, not bitwise det. |

Other public names:

- `CakeGDNNondetKcp{Context,Function,Options,Partition,RecomputeHandle,Saved,Workspace}`.
- `cake_gdn_nondet_train_kcp_packed_qkv_views`.
- `cake_gdn_nondet_train_kcp_workspace_bytes`: size a checkpointed `'light_akk'` layer with `save_mode='light'`.
- `cake_gdn_nondet_train_kcp_saved_bytes`.
- `cake_gdn_nondet_train_kcp_comm_bytes` (C=10240).
- `is_cake_gdn_nondet_train_kcp_supported`: raises if the package fails to import.
- `DEFAULT_SCALE`: not in `__all__`.

**Options and environment.**

- Single-rank plan pins (keyword-only `ChainOptions`):
  - `scan_config`: value block and key split.
  - `segment_chunks`: 0 means a serial scan.
  - `finalize`, `warmup_log2`.
  - `save_state`: keeps the forward tape.
  - `host_cache`.
  - `dhu_scan`: `'reg'` or `'tc'`.
  - `prep`: `'plain'` or `'tma'`.
  - `lpt`: longest-first ordering.
  - `streams`: 2 or 1.
  - Nondet adds `tail`.
- Route selection: passing `cu_seqlens_cpu` selects the host-table route. Without it, the device builds the tables and does not check the `cu_seqlens` values.
- KCP options (fields of `CakeGDN{Det,Nondet}KcpOptions` or keywords):
  - `exchange`: `'a2a'` or `'ring'`.
  - `save_transfer`.
  - `save_mode`: `'light'` or `'full'`; nondet adds `'light_akk'`.
  - `boundary_mode`: `'saved'` or `'replay'`.
  - `precision_mode`, `transfer_walk`.
  - `offset_class`: `'fla'` or `'fp32'`.
  - `layout`: `'tma'` or `'copy'`.
  - `group_reduce`: `'fused'` or `'host'`.
  - `graph`: `'off'` or `'replay'`.
  - `reuse_recompute_tapes`, `recompute_output`, `backend`, `segments`, `use_qk_l2norm_in_kernel`, `test_knobs`.
- `recompute_output='auto'` depends on the torch release being listed in `cake_kcp.RECOMPUTE_OUTPUT_TORCH_VERSIONS`. `cake_kcp.LAST_CALL['fwd']['fwd_o_reason']` records the decision.
- No environment variable controls kernel behavior.

**Cross-family.** The KCP modules state that the width-4 causal convolution with halo for the same partition is `cake_pplx.cake_causal_conv1d_det_train_kcp.cake_causal_conv1d_det_kcp(x_local, weight, ctx)`, called with this module's `ctx`. `build_context` plans the convolution halo (`conv_kernel_size=4`). One recompute handle serves this call and at most one context-parallel convolution call (key `'conv'`). `comm_bytes` counts the halo rows (C=10240). `packed_qkv_views` takes the convolution's packed output row.

**Example.**

```python
import math, torch
from cake_pplx.cake_gdn_det_train import cake_gdn_det, is_cake_gdn_det_train_supported

assert is_cake_gdn_det_train_supported()
seq_lens, (H, HV), D, dev = [100, 37], (2, 6), 128, "cuda"
T = sum(seq_lens)
q = torch.randn(1, T, H, D, device=dev).bfloat16().requires_grad_()
k = torch.randn(1, T, H, D, device=dev).bfloat16().requires_grad_()
v = torch.randn(1, T, HV, D, device=dev).bfloat16().requires_grad_()
g = (torch.rand(1, T, HV, device=dev) * -0.5).requires_grad_()   # FP32 log-decay
beta = torch.rand(1, T, HV, device=dev).requires_grad_()          # FP32 (or BF16)
cu = [0, 100, 137]
o, final_state = cake_gdn_det(
    q, k, v, g, beta,
    scale=1.0 / math.sqrt(D),
    initial_state=None,
    output_final_state=False,
    cu_seqlens=torch.tensor(cu, dtype=torch.int64, device=dev),
    cu_seqlens_cpu=torch.tensor(cu, dtype=torch.int64),   # host-table route
    use_qk_l2norm_in_kernel=True,
)
(o.float() * torch.randn_like(o, dtype=torch.float32)).sum().backward()
# cake_pplx.cake_gdn_nondet_train.cake_gdn_nondet takes the same arguments
```

**Caveats.**

- The device route (no `cu_seqlens_cpu`) trusts the `cu_seqlens` values without checking them.
- Single-rank `beta` may be FP32 or BF16. KCP `beta` and `g` must be FP32.
- Do not mix det and nondet KCP objects; their head plans and context layouts differ.
- `save_mode='light_akk'` (nondet KCP) needs `use_qk_l2norm_in_kernel=True`. Under `torch.utils.checkpoint` it runs as `'light'`.
- `recompute_handle` needs `boundary_mode='saved'` and `graph='off'`, and it implies `save_transfer=False`. A forward outside the backward that finds an entry raises. Det `reuse_recompute_tapes` is not available with `graph='replay'`.
- Plain split/view head views of a packed QKV row make the backward slower than `*_packed_qkv_views`, because torch concatenates the gradients.
- The det `is_*_supported` functions swallow exceptions and return `False`. The nondet versions raise if the package fails to import.
- The KCP function signatures are resolved lazily from the backend, so `inspect.signature` imports the package.
- The non-KCP facades do not define `DEFAULT_SCALE`.
- No H range limit is stated beyond `HV % H == 0` and head dim 128.

---

## mhc

**Computes.** A four-stream mHC (manifold-constrained hyper-connection) layer around an external branch. Arithmetic is FP32, with BF16 at the activation boundaries.

Forward:

1. `r = rsqrt(mean(u*u)+rms_eps)`, where `u = float32(X).reshape(T,4H)`.
2. `z = (u*r) @ fn.T`, shape `[T,24]`.
3. `pre = sigmoid(z[:, :4]*scale[0]+base[:4])+hc_eps`, `post = 2*sigmoid(z[:,4:8]*scale[1]+base[4:8])`, and `C` is a Sinkhorn-normalized 4x4 mixing matrix.
4. `U = BF16(sum_i pre_i X_i)` goes to the branch.
5. `Y[t,j,d] = BF16(post_j B + sum_i C_ij X_i)`.

Backward:

- `backward_prepare` runs before the branch backward. It gives `dB = BF16(sum_j post_j dY_j)` and per-token gradient statistics.
- `backward_finish` gives a single-rounded BF16 `dX` and FP32 `dfn`, `dscale` and `dbase`. The mapping, collapse and residual-mixing contributions to `dX` are summed in FP32.

**Architectures.** `sm_100a`, `sm_103a`, `sm_107a`.

- det: hidden size 4096, 5120 or 7168 on all three architectures.
- nondet: hidden size 4096, 5120 or 7168 on all three architectures. Other sizes raise `ValueError`.

**Inputs.**

- `X`: BF16 `[T, 4, H]`, contiguous.
- `fn`: FP32 `[24, 4H]`. `scale`: FP32 `[3]`. `base`: FP32 `[24]`.
- `B`: BF16 `[T, H]`, the branch output.
- `stats`: FP32 `[T, 64]`, laid out as `r | z[24] | pre[4] | post[4] | C[16] | pad[15]`.
- `grad_stats`: FP32 `[T, 32]`, laid out as `dpost[4] | dC[16] | zeros[12]`.
- All tensors must be contiguous with a 32-byte aligned base. T == 0 returns empty outputs without launching, and T == 1 is supported.

### `cake_pplx.cake_mhc_det_train`

Reductions run in a fixed order, each sum in one plain FP32 word by default (no FP64 accumulation or reduction; a few scalar epilogues compute in FP64 and round once to FP32; see **Accumulation** below for the opt-in two-word mode). `dX` and the FP32 `dfn`/`dscale`/`dbase` are bitwise reproducible for a given token count on a given device. No alert under deterministic algorithms. There is no KCP module. Source: [`kernels/mhc/cake_pplx/cake_mhc_det_train.py`](../kernels/mhc/cake_pplx/cake_mhc_det_train.py).

| Name | Signature | Inputs | Returns / notes |
|---|---|---|---|
| `cake_mhc_det_train_prepare` | `(X, fn, scale, base, *, rms_eps=1e-5, hc_eps=1e-6, sinkhorn_iters=20, accumulation="fp32", save_stats=True, workspace=None, backward_workspace=None)` | All four tensors may require grad. Size the workspaces with `hidden=X.shape[2]`. | Autograd. Returns `(U, handle)`; pass the handle to one `cake_mhc_det_train_finish`. The backward returns `dX`, `dfn`, `dscale` and `dbase`. `save_stats=False` drops `stats` after the forward and recomputes them in the backward, with bitwise the same gradients. The forward, the replay and the backward use the `accumulation` of this call; the handle carries it, so finish takes no mode. |
| `cake_mhc_det_train_finish` | `(X, B, handle)` | `X` is the tensor passed to prepare; `B` may require grad | Autograd. Returns BF16 `Y [T, 4, H]`. Its backward returns `dB` and no gradient for `X`; the residual-mixing part of `dX` is in the prepare node. |
| `cake_mhc_det_train_forward_prepare` | `(X, fn, scale, base, *, rms_eps=1e-5, hc_eps=1e-6, sinkhorn_iters=20, accumulation="fp32", out=None, stats_out=None, workspace=None)` | `sinkhorn_iters` 1..64; above 64 raises `NotImplementedError`. `out` BF16 `[T, H]`; `workspace` uint8 from `..._forward_workspace(hidden=...)` | Returns `(U, stats)`. Not autograd. |
| `cake_mhc_det_train_forward_stats` | `(X, fn, scale, base, *, rms_eps=1e-5, hc_eps=1e-6, sinkhorn_iters=20, accumulation="fp32", stats_out=None, workspace=None)` | As forward_prepare; pass the forward's `accumulation` | Returns FP32 `stats [T, 64]` for checkpoint replay, bitwise equal to forward_prepare's record in the same mode. |
| `cake_mhc_det_train_forward_finish` | `(X, B, stats, *, out=None)` | `out` BF16 `[T, 4, H]` | Returns `Y`. The FP32 sum is rounded once. |
| `cake_mhc_det_train_backward_prepare` | `(dY, B, X, stats, *, dB=None, grad_stats_out=None)` | `dY` BF16 `[T, 4, H]` | Returns `(dB, grad_stats)`. Run it before the branch backward. |
| `cake_mhc_det_train_backward_finish` | `(X, dY, dU, B, fn, scale, base, stats, grad_stats, *, rms_eps=1e-5, hc_eps=1e-6, sinkhorn_iters=20, accumulation="fp32", accumulate=False, workspace=None, dX=None, dfn=None, dscale=None, dbase=None)` | `dU` BF16 `[T, H]`; `B` may be `None` (it is validated but not read); `accumulate=True` requires `dfn`, `dscale` and `dbase` | Returns `(dX, dfn, dscale, dbase)`. `accumulate=True` adds into the caller's tensors. With T == 0, the gradients are zeroed, or left untouched under `accumulate`. |

Other public names:

- `cake_mhc_det_train_forward_workspace_bytes(T, device, *, hidden=4096)` and `..._backward_workspace_bytes(...)`: raise `ValueError` for an unsupported hidden size.
- `cake_mhc_det_train_forward_workspace(T, device, *, hidden=HIDDEN)` and `..._backward_workspace(...)`: return zeroed uint8 buffers.
- `cake_mhc_det_train_reset_workspace(workspace)`: use it after an interrupted launch, such as an asynchronous error or a cancelled capture.
- `cake_mhc_det_train_route_key(hidden, kind)`: returns `'<kind>_h<hidden>'`, where `kind` is `fwd_prepare`, `fwd_recompute`, `fwd_finish`, `bwd_prepare` or `bwd_finish`.
- `is_cake_mhc_det_train_supported(device, hidden=None)`.
- `CakeMHCDetPrepareFunction`, `CakeMHCDetFinishFunction`, `ARCHES`, `SUPPORTED_COMPUTE_CAPABILITIES`.

### `cake_pplx.cake_mhc_nondet_train`

There is no final reduction kernel. The column-slab pass adds the P <= 32 per-chunk FP32 `dfn` partials in ascending chunk order as one plain FP32 sum, and folds `dscale` and `dbase` from the token-owner rows in a fixed order. By default no sum or reduction uses FP64 or a compensation word (a few scalar epilogues compute in FP64 and round once to FP32); see **Accumulation** below for the opt-in two-word mode. The forward, the stats replay and `backward_prepare` run copies of the det programs. The sums run in a fixed order today, but nondet makes no run-to-run bit promise, and its outputs are not compared bitwise with det. Source: [`kernels/mhc/cake_pplx/cake_mhc_nondet_train.py`](../kernels/mhc/cake_pplx/cake_mhc_nondet_train.py).

| Name | Signature | Inputs | Returns / notes |
|---|---|---|---|
| `cake_mhc_nondet_train_prepare` | Same as det prepare | `backward_workspace` from `cake_mhc_nondet_train_backward_workspace` | Autograd. Returns `(U, handle)`. The forward and replay run the det programs; the backward runs the nondet `backward_finish`. |
| `cake_mhc_nondet_train_finish` | `(X, B, handle)` | A handle from another package (for example det) raises `TypeError` | Returns `Y`. |
| `cake_mhc_nondet_train_forward_prepare` | Same as det | H is 4096, 5120 or 7168 on every architecture | Returns `(U, stats)`. |
| `cake_mhc_nondet_train_forward_stats` | Same as det | As forward_prepare | Returns FP32 `[T, 64]`. |
| `cake_mhc_nondet_train_forward_finish` | `(X, B, stats, *, out=None)` | As det | Returns `Y`. |
| `cake_mhc_nondet_train_backward_prepare` | `(dY, B, X, stats, *, dB=None, grad_stats_out=None)` | As det | Returns `(dB, grad_stats)`. |
| `cake_mhc_nondet_train_backward_finish` | Same as det | `workspace` must come from the nondet factory; the det layout differs | Returns `(dX, dfn, dscale, dbase)`. `dfn` is one FP32 sum of the chunk partials, with the caller's `dfn` added last under `accumulate=True`. It is within `gamma_n (abs(start) + sum abs(x_p))` of the exact sum, with `n` the additions and `gamma_n = n u / (1 - n u)`. |

Other public names:

- `is_cake_mhc_nondet_train_supported(device, hidden=None)`.
- `cake_mhc_nondet_train_forward_workspace_bytes(T, device, *, hidden=4096)`: the forward layout is the same in both packages.
- `cake_mhc_nondet_train_backward_workspace_bytes(...)`: 256 B of tickets, plus 512 B of arrival counts, plus the det regions without the `dfn` partials, plus `dfn` slots of `min(32, ceil(T/64)) x 24 x 4H x 4` bytes.
- `cake_mhc_nondet_train_forward_workspace(...)`, `..._backward_workspace(...)`.
- `cake_mhc_nondet_train_reset_workspace(workspace)`: required before reusing a buffer that a det backward used.
- `cake_mhc_nondet_train_route_key`, `CakeMHCNondetPrepareFunction`, `CakeMHCNondetFinishFunction`, `ARCHES`, `SUPPORTED_COMPUTE_CAPABILITIES`.

### Pre handoff and standalone collapse (det and nondet)

A sublayer of [DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) collapses its streams with the `pre` of the previous sublayer and hands its own `pre` on. Both packages have entry points for this, under their own prefix (`<mode>` is `det` or `nondet`):

- Forward: `(pre_next, post, C)` are the coefficients of `X`, as above. `U = BF16(sum_i pre_in_i X_i)` uses the incoming FP32 `pre_in [T, 4]`, not `pre_next`. `pre_next` is the FP32 `pre` of the coefficient pass, bit for bit the `pre` columns of `stats`. `Y` is computed by `forward_finish` as before.
- Backward: `dpre_next [T, 4]` FP32 is the gradient of `pre_next` from its consumer. The coefficient VJP takes `dpre_next`, not a gradient derived from `dU`. `dpre_in_i = sum_d dU_d X_id` (one FP32 sum per token and stream). `dX = BF16(dX_mix + pre_in_i dU + dX_coef)` is rounded once.
- The first sublayer passes a one-hot `pre_in = [1, 0, 0, 0]`; its `U` is stream 0 bit for bit. The final collapse before the model head is the standalone collapse `(X, pre_in) -> U` with the VJP `(X, pre_in, dU) -> (dX, dpre_in)`.
- Coverage: hidden 5120 on `sm_100a`, `sm_103a` and `sm_107a` (`is_cake_mhc_<mode>_train_handoff_supported(device, hidden=None)`). Other sizes and architectures raise `ValueError`. The standard entry points are unchanged.

| Name | Signature | Returns / notes |
|---|---|---|
| `cake_mhc_<mode>_train_handoff_forward_prepare` | `(X, pre_in, fn, scale, base, *, rms_eps=1e-5, hc_eps=1e-6, sinkhorn_iters=20, accumulation="fp32", out=None, pre_next_out=None, stats_out=None, workspace=None)` | Returns `(U, pre_next, stats)`. Not autograd. The workspace is the forward workspace of the package. `Y` comes from `forward_finish(X, B, stats)`. |
| `cake_mhc_<mode>_train_handoff_backward_finish` | `(X, dY, dU, pre_in, dpre_next, B, fn, scale, base, stats, grad_stats, *, rms_eps=1e-5, hc_eps=1e-6, sinkhorn_iters=20, accumulation="fp32", accumulate=False, workspace=None, dX=None, dpre_in=None, dfn=None, dscale=None, dbase=None)` | Returns `(dX, dpre_in, dfn, dscale, dbase)`. `grad_stats` comes from `backward_prepare` as before. `accumulate` and the workspace work as in `backward_finish`. |
| `cake_mhc_<mode>_train_collapse_forward` | `(X, pre_in, *, out=None)` | Returns BF16 `U [T, H]`. No workspace and no parameters. |
| `cake_mhc_<mode>_train_collapse_backward` | `(X, pre_in, dU, *, dX=None, dpre_in=None)` | Returns BF16 `dX = BF16(pre_in_i dU)` and FP32 `dpre_in`. |
| `cake_mhc_<mode>_train_handoff_prepare` | `(X, pre_in, fn, scale, base, *, rms_eps=1e-5, hc_eps=1e-6, sinkhorn_iters=20, accumulation="fp32", save_stats=True, workspace=None, backward_workspace=None)` | Autograd. Returns `(U, pre_next, handle)`; pass the handle to the package's `finish`. The backward takes `dU` and `dpre_next` (zeros when `pre_next` reaches no loss) and returns `dX`, `dpre_in` (when `pre_in` requires grad) and the parameter gradients. It saves `pre_in` (the previous `pre_next`, no copy). |
| `cake_mhc_<mode>_train_collapse` | `(X, pre_in)` | Autograd standalone collapse. Returns `U`. It saves `X` and `pre_in`. |

- Use `rms_eps=1e-20` for DeepSeek-V4.1-Flash. `rms_eps` and `hc_eps` are launch arguments.
- Under `torch.utils.checkpoint(..., use_reentrant=False)`, a `pre_next` may cross region boundaries: a region returns `(Y, pre_next)`, and the next region takes `pre_next` as `pre_in`.
- Retained state: the handoff connection keeps what the standard connection keeps (`stats`, 256 B per token, under `save_stats=True`). `pre_next` (16 B per token) stays alive until its consumer's backward has run. The standalone collapse saves only its inputs.
- det: the bits of every output repeat for a given token count on a given device, as in the standard det entry points. nondet makes no run-to-run bit promise.
- The autograd classes are `CakeMHC<Mode>HandoffPrepareFunction` and `CakeMHC<Mode>CollapseFunction`.

```python
from cake_pplx.cake_mhc_nondet_train import (
    cake_mhc_nondet_train_collapse, cake_mhc_nondet_train_finish, cake_mhc_nondet_train_handoff_prepare,
)

pre = torch.zeros(T, 4, device=dev); pre[:, 0] = 1.0           # one-hot start
for fn, scale, base, branch in sublayers:                       # attention, then MoE, per layer
    U, pre, handle = cake_mhc_nondet_train_handoff_prepare(X, pre, fn, scale, base, rms_eps=1e-20)
    X = cake_mhc_nondet_train_finish(X, branch(U), handle)
head_input = cake_mhc_nondet_train_collapse(X, pre)            # the final collapse
```

**Options and environment.**

- `rms_eps` and `hc_eps` are launch arguments, so any finite positive value uses the same compiled programs.
- `sinkhorn_iters` can be 1..64. The default of 20 uses unrolled programs; other counts use run-time-count programs.
- `accumulation` (det and nondet `forward_prepare`, `forward_stats`, `backward_finish` and `prepare`) selects how some kernels add their sums. See **Accumulation**.
- Workspaces are uint8, sized by `*_workspace_bytes(T, device, hidden=H)`, and must be zero on first use. A buffer serves calls up to T tokens at its hidden size. A buffer that is too small is refused with `ValueError`. If you omit the workspace, a per-device default buffer is used.
- This family has no environment variables of its own.

**Accumulation.** `accumulation="fp32"` is the default: every kernel keeps each sum in one plain FP32 word. `accumulation="fp32x2"` is opt-in. It replaces the programs of some passes with two-word programs. A two-word program keeps each sum as a compensated pair of FP32 words (the running sum and its rounding error) and stores one FP32 word. It is a compensated two-word FP32 sum, not FP64.

| Package | Passes with a two-word program |
|---|---|
| det | the mapping pass (forward and replay), the backward token-owner pass, the column slab, the final reduction |
| nondet | the mapping pass (forward and replay), the backward token-owner pass, the column slab with its `dscale`/`dbase`/`dfn` folds |

- Coverage: hidden 4096 and `sinkhorn_iters=20`, on `sm_100a`, `sm_103a` and `sm_107a`. Hidden 5120 or 7168, or another `sinkhorn_iters`, raises `NotImplementedError`. Another value of `accumulation` raises `ValueError`.
- The other passes (the fn split, the coefficients, the collapse, the residual mix and `backward_prepare`) have no two-word program. They run their one-word programs in both modes.
- The workspaces, `stats` and `grad_stats` keep their layouts and sizes in both modes.
- One two-word program per pass serves every token count, in place of the token-count bands of the default programs. The mode is for accuracy studies; its speed is not tuned.
- Pass the forward's mode to `forward_stats` and `backward_finish`. The det outputs in this mode also repeat bit for bit for a given token count on a given device.

**Cross-family.** The model head after the last mHC layer is the separate family [mhc_head](#mhc_head). It takes the BF16 `[T, 4, H]` residual streams, with no shared context object. Within this family, `stats` and `grad_stats` records can be passed between det and nondet, but prepare handles and backward workspaces cannot.

**Example.**

```python
import torch
from cake_pplx.cake_mhc_det_train import cake_mhc_det_train_prepare, cake_mhc_det_train_finish

T, H, dev = 65, 4096, torch.device("cuda:0")
X = torch.randn(T, 4, H, device=dev).to(torch.bfloat16).requires_grad_(True)
fn = (torch.randn(24, 4 * H, device=dev) * 1e-2).contiguous().requires_grad_(True)
scale = (1.0 + 0.1 * torch.randn(3, device=dev)).requires_grad_(True)
base = (0.1 * torch.randn(24, device=dev)).requires_grad_(True)
Bext = torch.randn(T, H, device=dev).to(torch.bfloat16)
dY = torch.randn(T, 4, H, device=dev).to(torch.bfloat16)

U, handle = cake_mhc_det_train_prepare(X, fn, scale, base)
B = (U.float() + Bext.float()).to(torch.bfloat16)   # stand-in for the external branch
Y = cake_mhc_det_train_finish(X, B, handle)          # BF16 [T, 4, H]
(Y.float() * dY.float()).sum().backward()            # X.grad, fn.grad, scale.grad, base.grad
# nondet: cake_pplx.cake_mhc_nondet_train.cake_mhc_nondet_train_{prepare,finish}
```

**Caveats.**

- Default workspaces are one buffer per device, hidden size and power-of-two T bucket, and they are shared across streams. Pass caller-owned workspaces when you use concurrent streams.
- Nondet creates no default buffer while the stream is capturing a CUDA graph (`ValueError`). Warm up first with the same T and H.
- A reentrant checkpoint region (`use_reentrant=True`) must contain both prepare and finish, or start after prepare. It must not end between them.
- The former names (`cake_pplx.cake_mhc_train`, `CakeMHCDetTrain{Prepare,Finish}Function`, `route_key`) have no alias.
- The maximum token count is not stated in the facade docstrings.

---

## mhc_head

**Computes.** The head of the four-stream mHC model: the stream mean `Y[t, d] = BF16(((X0 + X1) + X2 + X3) * 0.25)`. The sum is FP32 in stream order with one rounding.

The backward computes `dX[t, i, d] = BF16(dY[t, d] * 0.25)` for all four streams. With `accumulate=True` it computes `dX = BF16(dX + BF16(dY * 0.25))`.

**Accuracy.** In both modes the tests judge every output over 32 inputs against a float64 oracle of the mean and its gradient. The error, in the L2 and the max norm, median and max, must be at most 1.005 times that of the naive BF16 tier (torch on the same BF16 inputs: `X.mean(dim=1)`, `dY * 0.25` on each stream and `dX0 + dY * 0.25`), and zero where the tier's error is zero. No test compares the outputs with torch bit for bit.

**Architectures.** `sm_100a`, `sm_103a` and `sm_107a` serve hidden sizes 4096, 5120 and 7168. On `sm_107a`, hidden 5120 and 7168 run the kernels built for any hidden size; no kernel is built for those two sizes there.

**Inputs.** `X` is BF16 `[T, 4, H]` and `dY` is BF16 `[T, H]`. Both must be contiguous and 32-byte aligned. T is at most `MAX_TOKENS = 2^22 - 1`.

### `cake_pplx.cake_mhc_head_det_train`

Elementwise per token, with fixed-order FP32 sums. Every output repeats bit for bit, including in checkpoint recompute. No alert under deterministic algorithms. Source: [`kernels/mhc_head/cake_pplx/cake_mhc_head_det_train.py`](../kernels/mhc_head/cake_pplx/cake_mhc_head_det_train.py).

| Name | Signature | Inputs | Returns / notes |
|---|---|---|---|
| `cake_mhc_head_det` | `(X)` | `X` may require grad | Autograd. Returns BF16 `Y [T, H]`. The node saves no tensor, so a checkpoint around it keeps nothing alive. |
| `cake_mhc_head_det_train_forward` | `(X, *, out=None)` | `out` BF16 `[T, H]`, contiguous, must not overlap `X` | Returns `Y`. Runs on the current stream and allocates nothing beyond the default `out`. It can be captured in a CUDA graph after one eager call. T == 0 returns an empty `Y`. |
| `cake_mhc_head_det_train_backward` | `(dY, *, dX=None, accumulate=False)` | `dX` BF16 `[T, 4, H]`, contiguous, required with `accumulate=True`, must not overlap `dY` | Returns `dX`; this is the caller's tensor when one is given. T == 0 launches nothing. |

Other public names:

- `is_cake_mhc_head_det_train_supported(device, hidden=None)`.
- `CakeMHCHeadDetFunction`: saves no tensor and reads a contiguous copy of a strided `dY`.
- `ARCHES`, `SUPPORTED_COMPUTE_CAPABILITIES`.

### `cake_pplx.cake_mhc_head_nondet_train`

This module is nondet by name only. The contract makes no bit promise, but the implementation uses det's kernels: fixed order, no atomics and no cross-token reduction. It raises no alert. It offers no speedup over det and exists only to complete the nondet mode next to `cake_mhc_nondet_train`. Source: [`kernels/mhc_head/cake_pplx/cake_mhc_head_nondet_train.py`](../kernels/mhc_head/cake_pplx/cake_mhc_head_nondet_train.py).

| Name | Signature | Inputs | Returns / notes |
|---|---|---|---|
| `cake_mhc_head_nondet` | `(X)` | As det | Autograd. The node saves no tensor. |
| `cake_mhc_head_nondet_train_forward` | `(X, *, out=None)` | As det | Returns `Y`. |
| `cake_mhc_head_nondet_train_backward` | `(dY, *, dX=None, accumulate=False)` | As det | Returns `dX`. |

Other public names: `is_cake_mhc_head_nondet_train_supported(device, hidden=None)`, `CakeMHCHeadNondetFunction`, `ARCHES`, `SUPPORTED_COMPUTE_CAPABILITIES`.

**Options and environment.** The keyword options are `out`, `dX` and `accumulate`; `accumulate=True` requires `dX`. This family has no environment variables of its own.

**Cross-family.** It takes the BF16 `[T, 4, H]` residual streams from the last mHC layer ([mhc](#mhc)). There is no shared context object or workspace.

**Example.**

```python
import torch
from cake_pplx.cake_mhc_head_det_train import (
    cake_mhc_head_det, cake_mhc_head_det_train_forward, cake_mhc_head_det_train_backward)

T, H, dev = 300, 4096, torch.device("cuda:0")
X = torch.randn(T, 4, H, device=dev).to(torch.bfloat16)
dY = torch.randn(T, H, device=dev).to(torch.bfloat16)

Y = cake_mhc_head_det_train_forward(X)              # mean of the 4 streams, BF16 [T, H]
dX = cake_mhc_head_det_train_backward(dY)           # BF16 [T, 4, H]
cake_mhc_head_det_train_backward(dY, dX=dX, accumulate=True)

Xg = X.detach().requires_grad_(True)
cake_mhc_head_det(Xg).backward(dY)                  # autograd; Xg.grad[:, i] = BF16(dY / 4)
# nondet: cake_pplx.cake_mhc_head_nondet_train.cake_mhc_head_nondet{,_train_forward,_train_backward}
```

**Caveats.**

- Check a hidden size with `is_cake_mhc_head_*_supported(device, hidden)`.
- Above `MAX_TOKENS`, the backend raises `ValueError`.
- Capture the forward in a CUDA graph only after one eager call has loaded the program.
- The former names (`cake_mhc_det_train_head*`, `CakeMHCDetTrainHeadFunction`, `is_cake_mhc_det_train_head_supported`, `cake_mhc_train_head*`) have no alias.

---

## engram_gate

**Computes.** The Engram context gate of the four residual streams, width 5120, with its backward. Per token `t` and stream `s`, every quantity in FP32 and each BF16 output rounded once, at its store:

```text
h = X[t,s]  k = K[t,s]  v = V[t]  w = FP32(q_weight[s]) * FP32(k_weight[s])
r_h = rsqrt(mean(h^2) + eps)  r_k = rsqrt(mean(k^2) + eps)  P = sum_d w h k  dot = P r_h r_k / sqrt(5120)
gate = sigmoid(copysign(sqrt(max(|dot|, clamp_value)), dot))   (0 where token_mask[t] is False)
out[t,s] = BF16(h + gate v)
```

The backward recomputes `r_h`, `r_k` and `P` from the inputs. It returns `dX`, `dK`, `dV` (BF16) and `dq_weight = dw * k_weight`, `dk_weight = dw * q_weight` with `dw = sum_t c h k` in FP32. The square root's derivative is 0 where `|dot| <= clamp_value`. `dot = +0` gives `+sqrt(clamp_value)` and `-0` gives `-sqrt(clamp_value)`. A masked token gives `out = X` and `dX = dOut` exactly, zero `dK` and `dV`, and no weight-gradient contribution.

**Architectures.** `sm_100a`, `sm_103a` and `sm_107a`. The forward ships two programs per architecture, one for each way to sum the dot product `P`. The backward ships two programs per architecture with the same bits and the same grid. A call without a `token_mask` of at least 4096 tokens on `sm_103a` launches the one that copies the next token's inputs into shared memory ahead of use; every other call launches the other.

**Sum of the dot product.** The autograd entries and the explicit forward take the keyword-only `dot_sum`. It selects the forward's sum of `P`.

- `dot_sum="fp32"` (the default): one FP32 word.
- `dot_sum="compensated"` (opt-in): a compensated FP32 dot product. Each product is split by an FMA, each running sum by an error-free two-sum, and the error words are added once at the end. The forward takes about 5-18% more time, depending on the GPU (`T = 16384`, FP32 weights, `token_mask=None`: 0.329 ms against 0.313 ms on GB300, 0.205 ms against 0.174 ms on VR200). Where the products nearly cancel, it rounds fewer output elements away from the correctly rounded result.

The backward is the same for both. It recomputes `P` as a compensated dot product, so the gradients do not depend on `dot_sum`. The explicit backward and the workspace query take no `dot_sum`. Another value raises `ValueError`; a value that is not a `str` raises `TypeError`.

**Accuracy.** The acceptance tests compare each output with an FP64 reference. The median and the largest relative L2 error over 25 or more draws must be at most twice those of an FP32 evaluation of the same formula. The draws are random, or built so that the dot products nearly cancel (`|dot|` from 2e-6 to 2e-5). Both `dot_sum` choices meet every bound on B200, GB200, GB300 and VR200. Measured on GB300 (`T = 16`, 25 draws of each kind), relative L2 error, median / largest:

| Output, draws | Error | Bound |
|---|---|---|
| `out`, random (either `dot_sum`) | 1.448e-3 / 1.457e-3 | 2.896e-3 / 2.914e-3 |
| `out`, near-cancelling (either `dot_sum`) | 1.661e-3 / 1.669e-3 | 3.321e-3 / 3.338e-3 |
| `dX`, `dK`, near-cancelling, per stream | 1.7e-3 / 1.7e-3 | 3.3e-3 / 3.4e-3 |
| `dq_weight`, `dk_weight`, near-cancelling, per stream | 1.3e-7 / 2.3e-7 | 3.2e-7 / 4.7e-7 |
| `dq_weight`, `dk_weight`, random | 1.4e-7 / 3.4e-7 | 1.8e-6 / 7.7e-5 |

The `out` errors of the two forwards agree to four digits. On the near-cancelling draws, the default `dot_sum="fp32"` rounds 17802 of the 8.19 million output elements away from BF16 of the FP64 result, against 117 for `dot_sum="compensated"`. An FP32 evaluation of the formula rounds 21917 of them away.

**Inputs.** `X` BF16 `[T, 4, 5120]`, contiguous. `K` BF16 `[T, 4, 5120]` and `V` BF16 `[T, 5120]`: contiguous, or views of one packed `kv [T, 25600]` (the four keys, then the value). They need a unit column stride, a stream stride of 5120 for `K` and a token stride that is a multiple of 8 elements. `q_weight` and `k_weight` are BF16 or FP32 `[4, 5120]`, each independently. `token_mask` is bool `[T]` or None. `eps` and `clamp_value` are run-time floats (1e-20 and 1e-6 by default). BF16 operands start on a 16-byte boundary. `T >= 0` is a run-time value: no recompilation for a new `T`.

### `cake_pplx.cake_engram_gate_det_train`

Every sum has one fixed order. Each cluster of the backward writes its own FP32 row of weight-gradient partial sums, and a second kernel adds the rows in one order. Every output repeats bit for bit on one device. No alert under deterministic algorithms. Source: [`kernels/engram_gate/cake_pplx/cake_engram_gate_det_train.py`](../kernels/engram_gate/cake_pplx/cake_engram_gate_det_train.py).

| Name | Signature | Inputs | Returns / notes |
|---|---|---|---|
| `cake_engram_gate_det` | `(x, key, value, q_weight, k_weight, token_mask=None, *, eps=1e-20, clamp_value=1e-6, dot_sum="fp32")` | As above; any may require grad | Autograd. Returns BF16 `out [T, 4, 5120]`. The node saves the inputs only. Weight gradients come back in each parameter's dtype. |
| `cake_engram_gate_det_kv` | `(x, kv, q_weight, k_weight, token_mask=None, *, eps=1e-20, clamp_value=1e-6, dot_sum="fp32")` | `kv` BF16 `[T, 25600]` | Autograd on the packed projection output. The backward writes one `dkv [T, 25600]` (no split, no concatenation copy). |
| `cake_engram_gate_det_train_forward` | `(x, key, value, q_weight, k_weight, token_mask=None, *, eps=1e-20, clamp_value=1e-6, dot_sum="fp32", out=None)` | `out` BF16 `[T, 4, 5120]`, contiguous | Returns `out`. Nothing is saved. |
| `cake_engram_gate_det_train_backward` | `(dout, x, key, value, q_weight, k_weight, token_mask=None, *, eps=1e-20, clamp_value=1e-6, dx=None, dkey=None, dvalue=None)` | `dout` BF16 `[T, 4, 5120]`, contiguous; `dkey`/`dvalue` may be views of one `dkv [T, 25600]` | Returns `(dx, dkey, dvalue, dq_weight, dk_weight)`, the weight gradients FP32. Takes the forward's inputs, not its output; the same after either `dot_sum`. |
| `cake_engram_gate_det_train_workspace_bytes` | `(T, device=None)` | | Device bytes one backward allocates beside its operands and outputs (see Caveats). |

Other public names: `is_cake_engram_gate_det_train_supported(device=None)`, `CakeEngramGateDetFunction`, `CakeEngramGateDetKvFunction`, `ARCHES`, `STREAMS` (4), `HIDDEN` (5120), `KV_WIDTH` (25600).

### `cake_pplx.cake_engram_gate_nondet_train`

This module is nondet by name. `out`, `dX`, `dK` and `dV` have one fixed order, and the contract makes no bit promise for `dq_weight` and `dk_weight`, but the implementation uses det's kernels: each cluster of the backward writes its own FP32 row of weight-gradient partial sums, and a second kernel adds the rows in one order. Adding the clusters' sums into one cleared accumulator with float atomics instead was slower in eager calls and faster only in some CUDA-graph calls with a token mask at 129 to 16384 tokens. It raises no alert. Source: [`kernels/engram_gate/cake_pplx/cake_engram_gate_nondet_train.py`](../kernels/engram_gate/cake_pplx/cake_engram_gate_nondet_train.py).

| Name | Signature | Inputs | Returns / notes |
|---|---|---|---|
| `cake_engram_gate_nondet` | As det | As det | Autograd. |
| `cake_engram_gate_nondet_kv` | As det | As det | Autograd on `kv`; one `dkv` written by the backward. |
| `cake_engram_gate_nondet_train_forward` | As det | As det | Returns `out`. |
| `cake_engram_gate_nondet_train_backward` | As det | As det | Returns `(dx, dkey, dvalue, dq_weight, dk_weight)`. |
| `cake_engram_gate_nondet_train_workspace_bytes` | `(T, device=None)` | | As det. |

Other public names: `is_cake_engram_gate_nondet_train_supported(device=None)`, `CakeEngramGateNondetFunction`, `CakeEngramGateNondetKvFunction`, `ARCHES`, `STREAMS`, `HIDDEN`, `KV_WIDTH`.

**Options and environment.** The keyword options are `eps`, `clamp_value`, `dot_sum`, `out`, `dx`, `dkey` and `dvalue`. This family has no environment variables of its own.

**Cross-family.** None. The four streams are the same `[T, 4, 5120]` tensors the [mhc](#mhc) family reads; no context object or workspace is shared.

**Example.**

```python
import torch
from cake_pplx.cake_engram_gate_nondet_train import cake_engram_gate_nondet_kv

T, dev = 4096, torch.device("cuda:0")
x = torch.randn(T, 4, 5120, device=dev).bfloat16().requires_grad_()
kv = torch.randn(T, 25600, device=dev).bfloat16().requires_grad_()   # four keys, then the value
q_weight = torch.ones(4, 5120, device=dev, requires_grad=True)
k_weight = torch.ones(4, 5120, device=dev, requires_grad=True)
token_mask = torch.ones(T, dtype=torch.bool, device=dev)

out = cake_engram_gate_nondet_kv(x, kv, q_weight, k_weight, token_mask, eps=1e-20, clamp_value=1e-6)
out.backward(torch.randn_like(out))      # x.grad, kv.grad [T, 25600], q_weight.grad, k_weight.grad
# det: cake_pplx.cake_engram_gate_det_train.cake_engram_gate_det_kv
# opt-in, a compensated forward sum (slower; the same backward): dot_sum="compensated"
```

**Caveats.**

- State kept for the backward: the inputs only. Workspace of a backward call: the FP32 `[4, 5120]` weight-gradient partial rows, 80 KiB each, one per cluster of the backward grid (det and nondet). The workspace query reads the device's architecture and SM count for `T > 0`: a device that is not a CUDA device raises `ValueError`, a CUDA device outside the architectures `RuntimeError`. The backward reads BF16 and FP32 weights as stored, and the autograd entries get their weight gradients in each parameter's dtype from the kernel (one rounding, as `.to(dtype)` rounds). A forward is one kernel launch: it reads BF16 and FP32 weights as stored and allocates its output alone.
- The forward and the backward run one cluster of four CTAs per token, persistently: `min(T, resident clusters)` clusters, where the resident clusters come from a table measured per architecture, SM count and CTAs per SM of the program (for example 148 for the forward on a 152-SM GB300). A device the table does not hold runs 0.85 of `CTAs per SM * SM count / 4` clusters. The grid changes the speed and, in det, the order of the weight-gradient sums (their last bits); one device uses one grid per token count.
- `X`, `dOut` and `dX` must be contiguous. Output rows must not overlap.
- No double backward.
- Capture in a CUDA graph only after one eager call has loaded the programs (of each `dot_sum` the graph's forward uses).

---

## gqa

**Computes.** Packed-varlen causal GQA attention training with head dimension 256.

- Query head `h` reads key head `h / (H_q / H_kv)`.
- Within a segment, query row `i` sees keys `j <= i + k_len - q_len` (FlashAttention bottom-right alignment). A row that sees no key gets `out = 0`, `lse = -inf` and zero gradients.
- The forward returns BF16 `out` and the FP32 natural-log LSE.
- The backward's main kernel sums `dK`/`dV` per key block in tensor memory, stores `dS^T` in a workspace, and computes `dQ` as a GEMM over it. `dK`/`dV` are returned as BF16, as FP32 window gradients, or as FP32 row-map adds into caller destinations.

**Architectures.** `sm_100a`, `sm_103a`, `sm_107a`. Other devices raise `RuntimeError`. The `build.json` minimum CUDA toolkit is 12.8 for `sm_100a`, 12.9 for `sm_103a` and 13.5 for `sm_107a`.

**Inputs.**

- `q`: BF16 `[T_q, H_q, 256]`. `k`, `v`: BF16 `[T_k, H_kv, 256]`. `H_q / H_kv` must be even.
- Strides: unit head-dim stride and 16-byte aligned token/head strides. Strided views pass without a copy.
- `cu_seqlens_q` / `cu_seqlens_k`: int32 `[B+1]` on the device, starting at 0 and non-decreasing, with `cu_q[B]=T_q` and `cu_k[B]=T_k`.
- `softmax_scale` defaults to `256**-0.5`, and `causal` must be `True`.
- `lse` is FP32 `[H_q, T_q]`.
- `cu_seqlens_*_host`: lists or 1-D CPU tensors. Pass both or neither.

### `cake_pplx.cake_gqa_det_train`

Every sum in the forward and backward has one fixed order. `out`, `lse`, `dQ` and every `dK`/`dV` form repeat bit for bit run to run for one shape, one `max_seqlen_k`, one workspace budget and the same host boundaries. A different budget or other host boundaries can move the last bits of `dQ` (not its accuracy); `out`, `lse`, `dK` and `dV` depend on neither. No alert. Source: [`kernels/gqa/cake_pplx/cake_gqa_det_train.py`](../kernels/gqa/cake_pplx/cake_gqa_det_train.py).

| Name | Signature | Inputs | Returns / notes |
|---|---|---|---|
| `cake_gqa_det` | `(q, k, v, cu_seqlens_q, cu_seqlens_k, *, softmax_scale=None, causal=True, dkv_acc=None, max_seqlen_k=None, workspace=None, return_lse=False, cu_seqlens_q_host=None, cu_seqlens_k_host=None)` | `dkv_acc` is a zero-argument callable that returns `(dk_acc, dv_acc, row_map)` (FP32 `[N_dst, H_kv, 256]`, int32 `[T_k]`) or `(dk_win, dv_win)` (FP32 `[T_k, H_kv, 256]`). `max_seqlen_k` is a host int >= every segment's key count; `None` bounds it by `T_k`. | Autograd (`CakeGQADetFunction`). Returns BF16 `out [T_q, H_q, 256]`, or `(out, lse)`. The backward gives BF16 `dq` and BF16 `dk`/`dv`. With `dkv_acc`, it writes FP32 gradients into the caller's tensors, and `k`/`v` get no gradient. |
| `cake_gqa_det_train_forward` | `(q, k, v, cu_seqlens_q, cu_seqlens_k, *, softmax_scale=None, out=None, lse=None, workspace=None)` | `out` contiguous BF16; `lse` contiguous FP32 `[H_q, T_q]`; `None` allocates | Returns `(out, lse)`. Not autograd. |
| `cake_gqa_det_train_backward` | `(q, k, v, out, lse, dout, cu_seqlens_q, cu_seqlens_k, *, softmax_scale=None, dq=None, dk=None, dv=None, dkv_dtype=torch.bfloat16, dk_acc=None, dv_acc=None, row_map=None, max_seqlen_k=None, workspace=None, cu_seqlens_q_host=None, cu_seqlens_k_host=None)` | `dout` contiguous BF16. `dk`/`dv` are contiguous `[T_k, H_kv, 256]` of `dkv_dtype` (bfloat16 or float32) and are not used with `row_map`. `dk_acc`/`dv_acc` are FP32 `[N_dst, H_kv, 256]` and need `row_map` (int32 `[T_k]`; out-of-range rows are not checked). | Returns `(dq, dk, dv)`. `dK`/`dV` are FP32 sums over the query heads of each key head. With `row_map`, `dk`/`dv` are `dk_acc`/`dv_acc`. The backward runs in chunks when `dS^T` exceeds the budget. |
| `cake_gqa_det_train_workspace` | `(device=None, *, budget_bytes=None)` | `budget_bytes` bounds one backward call's `dS^T` workspace; `None` means 16 GiB | Returns a `CakeGQADetWorkspace` (`.nbytes`, `.release()`). It grows to the largest shape seen and never shrinks. One workspace serves one stream at a time. The smallest chunk (one KV head, two key blocks) runs even past the budget. |
| `cake_gqa_det_train_workspace_bytes` | `(total_q, num_heads_q, num_segments, *, total_k=None, num_heads_kv=None, max_seqlen_k=None, row_map=False, budget_bytes=None, cu_seqlens_q_host=None, cu_seqlens_k_host=None)` | Pass `total_k` and `num_heads_kv` together; `row_map=True` needs both. Otherwise `ValueError`. | Bytes of backward workspace for one call: FP32 `[H_q/2, rows_pad, 256]` plus two `[H_q/2, rows_pad]` buffers, where `rows_pad = round_up(2 T_q + 128 B, 128)`. Adds the `dS^T` chunk and window gradients when requested. |

Other public names: `CakeGQADetFunction`, `CakeGQADetWorkspace`, `CakeGQADetDkvAccSource` (the type of `dkv_acc`).

### `cake_pplx.cake_gqa_nondet_train`

Nondet by name. The package does not promise that an output repeats bit for bit, from run to run or across workspace budgets, host boundaries, streams or kernels running beside it. The CTA pair that owns a key block sums `dK`/`dV`, and nothing is added with atomics. No alert. Source: [`kernels/gqa/cake_pplx/cake_gqa_nondet_train.py`](../kernels/gqa/cake_pplx/cake_gqa_nondet_train.py).

| Name | Signature | Inputs | Returns / notes |
|---|---|---|---|
| `cake_gqa_nondet` | Same as `cake_gqa_det` | As det. `max_seqlen_k` also sizes the `dS^T` chunks. `workspace=None` uses a per-stream default workspace with a 16 GiB budget. | Autograd (`CakeGQANondetFunction`). Same gradient forms as det. |
| `cake_gqa_nondet_train_forward` | Same as det forward | As det | Returns `(out, lse)`. Never alerts. |
| `cake_gqa_nondet_train_backward` | Same as det backward | As det | Returns `(dq, dk, dv)`. A row-map destination row `r` becomes `dst` plus the FP32 window rows `j` with `row_map[j] = r`. The budget and the host boundaries can move the rounding of the outputs, not their accuracy. |
| `cake_gqa_nondet_train_workspace` | `(device=None, *, budget_bytes=None)` | As det | Returns `CakeGQANondetWorkspace`. The `dS^T` workspace and row-map window gradients are allocated per call from PyTorch's caching allocator. |
| `cake_gqa_nondet_train_workspace_bytes` | `(total_q, num_heads_q, num_segments, *, total_k=0, num_heads_kv=0, max_seqlen_k=None, row_map=False, budget_bytes=None, cu_seqlens_q_host=None, cu_seqlens_k_host=None)` | The defaults are 0, where det's are `None` | Same formula as det. |

Other public names: `CakeGQANondetFunction`, `CakeGQANondetWorkspace`, `CakeGQANondetDkvAccSource`.

**Options and environment.** Every switch is a keyword argument, and the facades read no environment variable. The `cu_seqlens_*_host` arguments enable exact chunk planning; in det, the `dQ` bits depend on them.

**Cross-family.** None.

**Example.**

```python
import torch
from cake_pplx.cake_gqa_det_train import cake_gqa_det, cake_gqa_det_train_workspace

dev = torch.device("cuda")
D, h_q, h_kv = 256, 24, 4               # H_q / H_kv even
q_lens, k_lens = (257, 129), (513, 129)
cu = lambda lens: torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()], dtype=torch.int32, device=dev)
t_q, t_k = sum(q_lens), sum(k_lens)
q = torch.randn(t_q, h_q, D, device=dev).to(torch.bfloat16).requires_grad_()
k = torch.randn(t_k, h_kv, D, device=dev).to(torch.bfloat16).requires_grad_()
v = torch.randn(t_k, h_kv, D, device=dev).to(torch.bfloat16).requires_grad_()
dout = torch.randn(t_q, h_q, D, device=dev).to(torch.bfloat16)

ws = cake_gqa_det_train_workspace(dev)   # optional, reused across calls
out = cake_gqa_det(q, k, v, cu(q_lens), cu(k_lens), softmax_scale=0.0625, workspace=ws)
out.backward(dout)                        # BF16 q.grad, k.grad, v.grad
```

**Caveats.**

- The head dimension is fixed at 256, and only bottom-right causal alignment is supported.
- `cu_seqlens` values are read on the device only. Host copies that disagree with the device values beyond the plan fire a device-side assert, which leaves the CUDA context unusable.
- A `max_seqlen_k` smaller than a segment's key count silently leaves key blocks out. Providing a correct value is the caller's responsibility.
- Autograd holds the workspace by weak reference.
- In det, the bits of `out`, `lse`, `dK` and `dV` depend only on the shape and `max_seqlen_k`; the bits of `dQ` also depend on the workspace `budget_bytes` and the host boundaries. Nondet makes no bit promise.
- The autograd entries are named without `_train` (`cake_gqa_det`, `cake_gqa_nondet`). The non-autograd steps and workspace helpers include `_train`.

---

## fmha

**Computes.** The backward of dense multi-head attention for training, head dimension 64 or 128, non-causal or causal.

- Per batch and head, `P = softmax(softmax_scale * q k^T)` (with `causal=True`, query row `i` sees the keys `j <= i`) and `out = P v`.
- The backward returns `dv = P^T dout`, `dq = softmax_scale * dS k` and `dk = softmax_scale * dS^T q`, with `dS = P * (dout v^T - rowsum(out * dout))`.
- Three kernels run in order on the current stream. The preprocess computes `rowsum(out * dout)` and `lse * log2(e)` and clears an FP32 `dQ` accumulator. The main kernel runs one CTA per 128 keys of one head: it sums `dK`/`dV` in FP32, stores them once, and adds its `dQ` tiles into the accumulator. The postprocess writes `dq = bf16(acc * softmax_scale)`.
- The family has no forward kernel and no autograd entry. The caller brings `out` and `lse` from its own forward and calls the backward from its own `torch.autograd.Function`.

**Architectures.** `sm_100a`, `sm_103a`, `sm_107a`, with every variant (head dim 64 and 128, causal or not) on each. Other devices raise `RuntimeError`. The `build.json` minimum CUDA toolkit is 12.8 for `sm_100a`, 12.9 for `sm_103a` and 13.5 for `sm_107a`.

**Inputs.**

- `q`, `out`, `dout`: BF16 `[B, Sq, H, D]`. `k`, `v`: BF16 `[B, Skv, H, D]` with `q`'s `B`, `H` and `D` (no grouped heads).
- `lse`: FP32 `[B, H, Sq]`, the natural-log logsumexp of `softmax_scale * q k^T` over each query row with the mask applied (FlashAttention's `softmax_lse`).
- `D` is 64 or 128 and `H >= 1`. `Sq` and `Skv` are positive multiples of 128. `causal=True` needs `Sq == Skv`.
- Every operand is contiguous and on one CUDA device. BF16 operands start on a 16-byte boundary, `lse` on a 4-byte boundary.
- `B` and `H` are each at most 65535, and `B * H * max(Sq, Skv) * D` is below 2^31.

### `cake_pplx.cake_fmha_nondet_train`

The only module of the family. The `dQ` tiles of the key blocks of one head are added into an FP32 accumulator in arrival order, so `dq` may differ in its last bits run to run. `dk` and `dv` are summed in one fixed order. Under `torch.use_deterministic_algorithms(True)` the backward raises PyTorch's `RuntimeError` before it allocates or launches; with `warn_only=True` it warns once and runs. Source: [`kernels/fmha/cake_pplx/cake_fmha_nondet_train.py`](../kernels/fmha/cake_pplx/cake_fmha_nondet_train.py).

| Name | Signature | Inputs | Returns / notes |
|---|---|---|---|
| `cake_fmha_nondet_train_backward` | `(q, k, v, out, lse, dout, *, softmax_scale=None, causal=False)` | As above. `softmax_scale` defaults to `D**-0.5`, must be finite, and must be the forward's. | Returns contiguous BF16 `(dq, dk, dv)`: `dq [B, Sq, H, D]`, `dk`/`dv [B, Skv, H, D]`. Not autograd. `B == 0` returns empty gradients without a launch or an alert. A malformed call raises `ValueError` before any launch. Each call allocates its outputs and an FP32 workspace of `B * H * Sq * (D + 2)` elements. |

**Options and environment.** `softmax_scale` and `causal` are the only keywords. The facade reads no environment variable.

**Cross-family.** None.

**Example.**

```python
import torch
from cake_pplx.cake_fmha_nondet_train import cake_fmha_nondet_train_backward

B, S, H, D = 2, 1024, 8, 128
q, k, v = (torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16) for _ in range(3))
scale = D ** -0.5

# Any forward that returns out and its natural-log LSE works; here, plain PyTorch with the causal mask.
scores = torch.einsum("bqhd,bkhd->bhqk", q.float(), k.float()) * scale
scores = scores.masked_fill(torch.ones(S, S, device="cuda", dtype=torch.bool).triu(1), float("-inf"))
lse = scores.logsumexp(-1)  # FP32 [B, H, S]
out = torch.einsum("bhqk,bkhd->bqhd", scores.softmax(-1), v.float()).to(torch.bfloat16).contiguous()

dout = torch.randn_like(out)
dq, dk, dv = cake_fmha_nondet_train_backward(q, k, v, out, lse, dout, softmax_scale=scale, causal=True)
```

**Caveats.**

- Backward only: there is no forward kernel and no autograd entry.
- There is no det module, and `dq` is not bitwise reproducible run to run.
- No grouped-query or multi-query attention, variable-length segments, dropout, sliding window, softcap or attention bias.
- The causal mask is top-left and needs `Sq == Skv`.

---

## gated_kv_compression

**Computes.** Ratio-2 gated KV compression for training: each pair of adjacent tokens of a packed document becomes one row, mixed per channel by a softmax over the two gate logits. The width is 512.

- Document `s` has bounds `b_s = cu_seqlens[s]`, length `L_s` and `g_s` pairs before it. Its pair `k < floor(L_s / 2)` reads the tokens `t0 = b_s + 2k` and `t1 = t0 + 1` and writes row `r = g_s + k`. Pairs never cross a document boundary.
- **Forward:** `(p0, p1) = softmax(Z[t0, d], Z[t1, d])` and `out[r, d] = BF16(p0 * C[t0, d] + p1 * C[t1, d])`, in FP32 with one rounding.
- **Backward:** with `g = float32(dout[r, d])`, `dC[t0, d] = p0 * g`, `dC[t1, d] = p1 * g`, `dZ[t0, d] = p0 * p1 * (C[t0, d] - C[t1, d]) * g` and `dZ[t1, d] = -dZ[t0, d]`. The BF16 cast counts as the identity. The pairwise form of `dZ` stays accurate when the two values of a pair are close.
- The last token of an odd-length document belongs to no pair. It changes no output, and its `dC` and `dZ` rows are exactly zero.
- Three kernels run on the current stream: a pair table (the int32 `[S + 1]` exclusive prefix of the pair counts), the forward (one row per pair) and the backward (every token row written once). The kernels compute in FP32 only, with IEEE round to nearest and no FP64. Gate gaps of 100 and more and exact ties give finite outputs.
- The forward ships one schedule per architecture, the faster one on each: 128-bit loads of `C` and `Z` on `sm_100a` and `sm_103a`, 256-bit loads on `sm_107a`. The call picks the build of the operands' device. The backward uses 256-bit loads on all three. On `sm_100a` and `sm_103a` it finds the document of a pair in one search level for packs of at most 26 documents and in two levels for more; on `sm_107a` it uses one level. Both searches write the same bytes.

**Architectures.** `sm_100a`, `sm_103a`, `sm_107a`. Other devices raise `RuntimeError`. The `build.json` minimum CUDA toolkit is 12.8 for `sm_100a` (12.9 for the backward libraries, which use 256-bit loads and stores), 12.9 for `sm_103a` and 13.5 for `sm_107a`. The GPU suites and the benchmark ran on B200 (`sm_100a`), GB300 (`sm_103a`) and VR200 (`sm_107a`) with CUDA 13.5, against the acceptance criteria of issue #76: each `out` element is within 1 BF16 ulp of the FP64 result rounded once to BF16, or its absolute error is at most 1 BF16 ulp of `max(|C[t0, d]|, |C[t1, d]|)` (near-cancelling pairs), and `dC` and `dZ` have a relative L2 error of at most 1e-6 against FP64.

**Inputs.**

- `c`, `z`: FP32 `[T, 512]` values and gate logits. Each needs a unit feature stride, a row stride that is a multiple of 8 elements (at least 512) and a 32-byte aligned base. The column halves of one `[T, 1024]` tensor qualify and are read in place.
- `cz`: FP32 `[T, 1024]`, `C` in columns `[0, 512)` and `Z` in `[512, 1024)`, with the same layout rules (row stride at least 1024).
- `cu_seqlens`: int32 `[S + 1]` on the same device, `0 = b_0 <= ... <= b_S = T`. The host never reads it, so the caller keeps this contract.
- `dout`, `out`: BF16 `[G, 512]`, contiguous, 16-byte aligned. `dc`, `dz`: FP32 `[T, 512]` with the layout rules of `c`; the two halves of one `[T, 1024]` buffer qualify.
- `G`, the sum over documents of `floor(L_s / 2)`, comes from the caller: `num_pairs`, or the rows of a caller `out` or `dout`. `G` is at most 2^26 (67108864) and `S` at most 2^24 (16777216).
- `T`, `S` and `G` are run-time values. There is no recompilation, no device-to-host read and no synchronization.

### `cake_pplx.cake_gated_kv_compression_det_train`

The only module of the family. Every output element is one fixed chain of FP32 operations on its own pair or token, with no reduction across threads, CTAs or calls and no atomic. So `out`, `dC` and `dZ` repeat bit for bit for identical inputs, also in the recompute of an activation checkpoint. There is no nondet module, because no non-deterministic step exists that could be traded for speed. There is no KCP module, because the kernel is single-rank: under context parallelism, the caller moves a boundary token so that every rank holds whole pairs. Nothing alerts under `torch.use_deterministic_algorithms(True)`. Source: [`kernels/gated_kv_compression/cake_pplx/cake_gated_kv_compression_det_train.py`](../kernels/gated_kv_compression/cake_pplx/cake_gated_kv_compression_det_train.py).

| Name | Signature | Inputs | Returns / notes |
|---|---|---|---|
| `cake_gated_kv_compression_det` | `(c, z, cu_seqlens, num_pairs)` | `c`, `z` may require grad; `num_pairs` a Python int, at least the pair count | Autograd. Returns BF16 `out [G, 512]`. The node saves `c`, `z`, `cu_seqlens` and the int32 `[S + 1]` pair table only. Works under `torch.utils.checkpoint(..., use_reentrant=False)`. |
| `cake_gated_kv_compression_det_cz` | `(cz, cu_seqlens, num_pairs)` | `cz` FP32 `[T, 1024]` (`C` \| `Z`), may require grad; `num_pairs` a Python int, at least the pair count | Autograd. Returns BF16 `out [G, 512]`. The gradient of `cz` is one FP32 `[T, 1024]` buffer that one launch writes, with no slice-backward copy. The node saves `cz`, `cu_seqlens` and the pair table only. |
| `cake_gated_kv_compression_det_train_forward` | `(c, z, cu_seqlens, num_pairs=None, *, out=None)` | `num_pairs` or a preallocated `out` (both given must agree) | Returns `out`. Rows past the pair count are zero. `G == 0` launches nothing, and `T == 0` or `S == 0` zero-fills `out` without a kernel. Not autograd. |
| `cake_gated_kv_compression_det_train_backward` | `(dout, c, z, cu_seqlens, *, dc=None, dz=None, pair_offsets=None)` | `dout` rows `G` at least the pair count; `pair_offsets` the int32 `[S + 1]` pair table, which skips the pair-table launch | Returns FP32 `(dc, dz)`, every row written; these are the caller's tensors when given. `T == 0` launches nothing, and `G == 0` or `S == 0` zero-fills both without a kernel. |
| `cake_gated_kv_compression_det_train_memory_bytes` | `(num_tokens, num_docs)` | `T`, `S` | Returns a dict. `forward_workspace` and `backward_workspace` are `4 (S + 1)`: the pair table one explicit call allocates and frees. The autograd backward allocates no workspace, except a `1024 G`-byte copy of an incoming gradient that is not contiguous and 16-byte aligned (for example from `out.sum().backward()`), which the report leaves out. `retained_for_backward` is `4 (S + 1)`: the pair table the autograd node keeps. `retained_inputs` is `4096 T + 4 (S + 1)`: the caller's tensors the node references, with no copy. |

Other public names:

- `is_cake_gated_kv_compression_det_train_supported(device)`.
- `CakeGatedKvCompressionDetFunction`, `CakeGatedKvCompressionDetCzFunction`.
- `ARCHES`, `SUPPORTED_COMPUTE_CAPABILITIES`.

**Options and environment.** The keyword options are `out`, `dc`, `dz` and `pair_offsets`. This family has no environment variables of its own.

**Cross-family.** Its BF16 `out` feeds the [rmsnorm](#rmsnorm) family at width 512. This is a data dependency only, with no shared context object or workspace.

**Example.**

```python
import torch
from cake_pplx.cake_gated_kv_compression_det_train import (
    cake_gated_kv_compression_det_cz,
    cake_gated_kv_compression_det_train_backward,
    cake_gated_kv_compression_det_train_forward,
)

lengths = [5, 0, 8, 3]                                    # T = 16 tokens in S = 4 documents
cu_seqlens = torch.tensor([0, 5, 5, 13, 16], dtype=torch.int32, device="cuda")
G = sum(n // 2 for n in lengths)                          # 7 pairs, known on the host
cz = torch.randn(16, 1024, device="cuda", requires_grad=True)   # C | Z from one projection GEMM

out = cake_gated_kv_compression_det_cz(cz, cu_seqlens, G)      # BF16 [7, 512], autograd
out.backward(torch.randn_like(out))                             # cz.grad: FP32 [16, 1024], dC | dZ

c, z = cz.detach()[:, :512], cz.detach()[:, 512:]
out = cake_gated_kv_compression_det_train_forward(c, z, cu_seqlens, G)
grad = torch.empty(16, 1024, device="cuda")
cake_gated_kv_compression_det_train_backward(torch.randn_like(out), c, z, cu_seqlens,
                                             dc=grad[:, :512], dz=grad[:, 512:])
```

**Caveats.**

- The host never reads `cu_seqlens`. The caller keeps its contract (non-decreasing, last bound `T`) and passes `G`. A `G` below the pair count gives the first `G` rows in the forward and leaves the gradient rows of the later pairs unwritten in the backward. The autograd entries need `num_pairs` at least the pair count: their gradient buffers are not zero-filled, so with a smaller `num_pairs` the gradient rows of the tokens of the later pairs are uninitialized.
- A layout the kernels cannot read or write in place raises `ValueError`, and no call copies an operand silently. The autograd backward copies a non-contiguous or misaligned incoming gradient once.
- Capture a call in a CUDA graph only after one eager call on the device. That call loads every program of the architecture, both pair tables and both forward programs included, so a later pack of any size loads none.
- Only ratio 2 and width 512 are supported.

---

## dsa_indexer

**Computes.** Fused DeepSeek Sparse Attention indexer scoring plus exact top-k key selection for training.

- Score: `s[t, j] = sum_h w[t, h] * relu(softmax_scale * sum_d q[t, h, d] * k[j, d])`, in FP32, over the visible causal prefix of each packed segment.
- Selection: the exact global top `min(top_k, visible)` keys per query row, ranked by score descending, then key id descending. det returns them in ascending id order, and nondet leaves the slot order unspecified.
- Forward only. It returns int32 indices and FP32 scores. No autograd or backward is documented.
- Candidate blocks (`*_topk_with_candidates`): the same call also returns, per query, up to 2048 blocks of 8 keys chosen from the same scores (below).
- Candidate-restricted top-k (`*_topk_in_candidates`): the top-k over only the keys of the blocks a caller names per query, for example the candidate blocks of another layer (below).

**Architectures.** `sm_100a`, `sm_103a`, `sm_107a`, for every call, the candidate-block and candidate-restricted calls included.

**Inputs.**

- `q`: BF16 `[T, 32, 128]`, contiguous (`float8_e4m3fn` for the FP8 form of either mode).
- `k`: BF16 `[Tkv, 128]`. A row-strided view is allowed if the row stride is a multiple of 8 elements (16 for FP8). Must have the same dtype as `q`.
- `w`: FP32 `[T, 32]`, pre-scaled by `32**-0.5`.
- `cu_seqlens_q` / `cu_seqlens_k`: int32 `[S+1]` on the device, with `cu_seqlens_q[-1]==T` and `cu_seqlens_k[-1]==Tkv`. Both are required (`ValueError` if `None`).
- `top_k`: 1..4096. `softmax_scale` must be > 0 and defaults to `128**-0.5`.
- `q_causal_offsets`: int64 `[S]` on the device. `ratio` >= 1.
- `max_seqlen_k`: rejected if > `Tkv` or < `ceil(Tkv/S)`.
- `workspace_buffer`: uint8, 8-byte aligned, at least the `workspace_size` bound.
- `indices` / `scores`: caller-owned int32/float32 `[T, top_k]` outputs.

### `cake_pplx.cake_dsa_indexer_det_train`

Identical inputs give identical ids and score bits, independent of the internal partition and the dispatched program. Each row is returned in ascending key-id order. There is no KCP module. Source: [`kernels/dsa_indexer/cake_pplx/cake_dsa_indexer_det_train.py`](../kernels/dsa_indexer/cake_pplx/cake_dsa_indexer_det_train.py).

| Name | Signature | Inputs | Returns / notes |
|---|---|---|---|
| `cake_dsa_indexer_det_train_topk` | `(q, k, w, cu_seqlens_q, cu_seqlens_k, *, top_k=2048, softmax_scale=None, q_causal_offsets=None, ratio=1, max_seqlen_q=None, max_seqlen_k=None, workspace_buffer=None, indices=None, scores=None, return_scores=True, k_scale=None)` | As above. `scores` must be omitted when `return_scores=False` (`ValueError`). `k_scale` is FP32 `[Tkv]`, contiguous, FP8 form only. In the FP8 form `softmax_scale` must be `None`. | Returns `(indices, scores)`. `indices` are int32 `[T, top_k]` segment-local ids, ascending and unique, with tail `-1`. `scores` are FP32, aligned with `indices`, with tail `-inf`, or `None`. Not autograd. Never materializes `[T, Tkv]`, does not sync with the host, and can be captured in a CUDA graph. |
| `cake_dsa_indexer_det_train_topk_workspace_size` | `(num_queries, num_keys, num_segments, *, top_k=2048, ratio=1, device=None, dtype=torch.bfloat16)` | Python ints. `dtype` is `torch.bfloat16` or `torch.float8_e4m3fn`. | Workspace bound in bytes for one call geometry. It follows the scan program chosen from T, Tkv, S, `ratio`, `top_k` and the SM count; for example, 74 MiB at `top_k=2048` with 4 queries per unit on a 148-SM `sm_100a`. Raises `NotImplementedError` when no program is registered for the architecture. |
| `cake_dsa_indexer_det_train_topk_with_candidates` | `(q, k, w, cu_seqlens_q, cu_seqlens_k, *, top_k=512, softmax_scale=None, q_causal_offsets=None, ratio=1, max_seqlen_q=None, max_seqlen_k=None, workspace_buffer=None, indices=None, scores=None, candidate_blocks=None, return_scores=True)` | As `cake_dsa_indexer_det_train_topk`, BF16 only, `ratio=1`, `1 <= top_k <= 2047`. `candidate_blocks`: caller-owned int32 `[T, 2048]`, contiguous. `max_seqlen_k` must bound every key segment (it sizes the block entries). | Returns `(indices, scores, candidate_blocks)`. `indices` / `scores` as `cake_dsa_indexer_det_train_topk` with the same `top_k`, bit for bit. `candidate_blocks` holds ascending segment-local block ids with tail `-1`. Identical inputs give identical bits. |
| `cake_dsa_indexer_det_train_topk_with_candidates_workspace_size` | `(num_queries, num_keys, num_segments, *, top_k=512, ratio=1, max_seqlen_k=None, device=None)` | Python ints; the call's `max_seqlen_k`. | Workspace bound in bytes: the scan's candidate buffers and tickets plus `grid x queries_per_unit x blk_cap x 8` bytes of block entries (`blk_cap = ceil(W/8) - 1` when `ceil(W/8) > 2048`, else 0; `W` = `max_seqlen_k`, else `Tkv`). The `[T, 2048]` output is not part of it. On `sm_107a` a prepared runner whose scan draws tickets also owns a 16-byte ticket block outside the bound, as for the plain call there. |
| `cake_dsa_indexer_det_train_topk_in_candidates` | `(q, k, w, cu_seqlens_q, cu_seqlens_k, candidate_blocks, *, top_k=512, softmax_scale=None, q_causal_offsets=None, ratio=1, max_seqlen_q=None, max_seqlen_k=None, workspace_buffer=None, indices=None, scores=None, return_scores=True)` | As `cake_dsa_indexer_det_train_topk`, BF16 only, `ratio=1`, `1 <= top_k <= 2048`; a row-strided `k` is read in place, as there. `candidate_blocks`: int32 `[T, W]`, `W <= 2048`, contiguous, on the device of `q` (the `candidate_blocks` output of `*_topk_with_candidates` fits as is). | Returns `(indices, scores)`: per row the top `min(top_k, \|P_t\|)` keys of its pool (below), ascending ids with tail `-1`, scores aligned with tail `-inf`, or `None`. Identical inputs give identical bits. |
| `cake_dsa_indexer_det_train_topk_in_candidates_workspace_size` | `(num_queries, num_keys, num_segments, *, top_k=512, ratio=1, device=None)` | Python ints. | Workspace bound in bytes: `grid x 4 x cand_cap x 8` bytes of candidate buffers (`cand_cap = max(4096, 4 top_k)`), `grid x 64 KiB` of block lists and a 16-byte work ticket; `grid` is the SM count. It does not depend on T, Tkv, S or the candidates. |

The module exports only these six functions.

### `cake_pplx.cake_dsa_indexer_nondet_train`

Results are sets. Each row holds its selected ids with their scores, but the slot order and the positions of the `-1`/`-inf` padding are unspecified and may vary between calls. Under deterministic algorithms a call with queries and keys (T > 0 and Tkv > 0) raises `RuntimeError` before it allocates or launches; with `warn_only=True` it warns and runs. It takes the BF16 form and the opt-in FP8 form, as det. Source: [`kernels/dsa_indexer/cake_pplx/cake_dsa_indexer_nondet_train.py`](../kernels/dsa_indexer/cake_pplx/cake_dsa_indexer_nondet_train.py).

| Name | Signature | Inputs | Returns / notes |
|---|---|---|---|
| `cake_dsa_indexer_nondet_train_topk` | `(q, k, w, cu_seqlens_q, cu_seqlens_k, *, top_k=2048, softmax_scale=None, q_causal_offsets=None, ratio=1, max_seqlen_q=None, max_seqlen_k=None, workspace_buffer=None, indices=None, scores=None, return_scores=True, k_scale=None)` | As det. `k_scale` is FP32 `[Tkv]`, contiguous, FP8 form only. In the FP8 form `softmax_scale` must be `None`. | Returns `(indices, scores)`. Ids are unique within a row, with `-1` in unused slots; scores are `-inf` at the `-1` slots. Can be captured in a CUDA graph. |
| `cake_dsa_indexer_nondet_train_topk_workspace_size` | `(num_queries, num_keys, num_segments, *, top_k=2048, ratio=1, device=None, dtype=torch.bfloat16)` | `dtype` is `torch.bfloat16` or `torch.float8_e4m3fn`. | Same bound as det for the same call and form. Never consults the deterministic-algorithms setting. |
| `cake_dsa_indexer_nondet_train_topk_with_candidates` | `(q, k, w, cu_seqlens_q, cu_seqlens_k, *, top_k=512, softmax_scale=None, q_causal_offsets=None, ratio=1, max_seqlen_q=None, max_seqlen_k=None, workspace_buffer=None, indices=None, scores=None, candidate_blocks=None, return_scores=True)` | As det. | Returns `(indices, scores, candidate_blocks)`. Each row of `candidate_blocks` holds its block set with `-1` in the other slots, in no specified order. Raises under deterministic algorithms, as `cake_dsa_indexer_nondet_train_topk`. |
| `cake_dsa_indexer_nondet_train_topk_with_candidates_workspace_size` | `(num_queries, num_keys, num_segments, *, top_k=512, ratio=1, max_seqlen_k=None, device=None)` | As det. | Same bound as det for the same call. Never consults the deterministic-algorithms setting. |
| `cake_dsa_indexer_nondet_train_topk_in_candidates` | `(q, k, w, cu_seqlens_q, cu_seqlens_k, candidate_blocks, *, top_k=512, softmax_scale=None, q_causal_offsets=None, ratio=1, max_seqlen_q=None, max_seqlen_k=None, workspace_buffer=None, indices=None, scores=None, return_scores=True)` | As det. | Returns `(indices, scores)`. Each row holds its set with `-1` / `-inf` in the other slots, in no specified order. A call that launches (T > 0, Tkv > 0 and W > 0) raises under deterministic algorithms, as `cake_dsa_indexer_nondet_train_topk`. |
| `cake_dsa_indexer_nondet_train_topk_in_candidates_workspace_size` | `(num_queries, num_keys, num_segments, *, top_k=512, ratio=1, device=None)` | As det. | Same bound as det for the same call. Never consults the deterministic-algorithms setting. |

**Options and environment.**

- `workspace_buffer`: when omitted, the buffer is allocated through the caching allocator. On the ticket architectures, a shared buffer has its ticket word zeroed before every run.
- `max_seqlen_k`: does not change the selected ids (det: nor the score bits). It selects the rank-finalize variant (det) and, on equal-length packed calls, skips the longest-segment-first scan order. `max_seqlen_q` is accepted but not read.
- `return_scores=False` returns `None` for `scores`, but the scores are still computed internally.
- The facades read no environment variable of their own. Backend partition knobs are private.

**Candidate blocks.** Block `b` holds the segment-local keys `8b .. 8b+7`. A row with `n` visible keys sees `nb = ceil(n/8)` blocks; the newest block, `nb-1`, may be partial and its future keys do not count. A block's score is the largest score of its visible keys. With `nb <= 2048` the row holds blocks `0 .. nb-1`. Otherwise it holds block `nb-1` and the 2047 blocks of `0 .. nb-2` with the largest scores, ties broken by the larger block id (`+0.0 == -0.0`), the tie rule of `indices`. A row with `n == 0` is all `-1`. With `top_k <= 2047` every id of `indices` lies in a block of its row. The call runs the unsplit narrow or six-query scan unit, so its timing can differ from the plain top-k call of the same shape.

**Candidate-restricted top-k.** For query row `t` with `n` visible keys the pool `P_t` is the keys `j < n` whose block `j // 8` appears in `candidate_blocks[t]`. Ids may come in any order; `-1` and ids at or past the row's visible blocks name no key, and the non-negative ids of a row must be distinct: a row that repeats one gets an unspecified result for that row, without out-of-bounds access. The row selects the top `min(top_k, |P_t|)` keys of `P_t` by score descending, then key id descending (`+0.0 == -0.0`), scored by the formula of `cake_dsa_indexer_*_train_topk`. Its work is bounded by its candidate keys (at most `8 W`), not by its visible prefix: no key outside `P_t` is scored. On inputs whose FP32 scores are exact in any summation order (for example `q` and `k` on the MXFP4 grid with scales 1 or 1/2, `w` in eighths and `softmax_scale` 1/16), the `candidate_blocks` of a layer's own `*_topk_with_candidates` call and the same `top_k` give that call's top-k set; on other inputs each call meets the family's accuracy rule against FP64 on its own. A call with `W == 0` launches nothing and returns all `-1` / `-inf`.

**Cross-family.** There is no runtime contract with other families. The `k` docstring mentions the `packed[:, :128]` view of a `[Tkv, 704]` tensor, the same row layout that dsa_sparse_attention describes. Neither family consumes the other's objects.

**Example.**

```python
import torch
from cake_pplx.cake_dsa_indexer_det_train import cake_dsa_indexer_det_train_topk

seg_q, seg_k = [200, 300], [200, 2500]
T, Tkv = sum(seg_q), sum(seg_k)
q = torch.randn(T, 32, 128, device="cuda").to(torch.bfloat16)
k = torch.randn(Tkv, 128, device="cuda").to(torch.bfloat16)
w = torch.randn(T, 32, device="cuda", dtype=torch.float32) * 32 ** -0.5
cu_seqlens_q = torch.tensor([0, 200, 500], dtype=torch.int32, device="cuda")
cu_seqlens_k = torch.tensor([0, 200, 2700], dtype=torch.int32, device="cuda")
indices, scores = cake_dsa_indexer_det_train_topk(
    q, k, w, cu_seqlens_q, cu_seqlens_k, top_k=128
)
# indices: int32 [T, 128], ascending segment-local ids, -1 padding
# scores:  float32 [T, 128], -inf padding
# nondet: cake_pplx.cake_dsa_indexer_nondet_train.cake_dsa_indexer_nondet_train_topk, same call (both forms)
```

**Caveats.**

- The geometry is fixed at 32 indexer heads and head dim 128.
- FP8 is an opt-in form in both modes.
- Nondet output tensors vary between calls. Compare the per-row id sets, not the tensors.
- Operator tests skip on devices other than compute capability 10.0, 10.3 or 10.7.
- Finite inputs are the normal domain. With NaN or inf, the output structure stays intact and the kernels do not hang, but the values are not otherwise specified.
- `cu_seqlens` are never read on the host.

---

## dsa_sparse_attention

**Computes.** Differentiable top-k sparse MLA attention with absorbed queries (DeepSeek Sparse Attention) and 64 query heads.

- **Rope form:** `O = softmax(softmax_scale * (q_rope . k_rope^T + q_latent . kv_latent^T)) . kv_latent`, over the keys that each row's indices select (K = V = latent).
- **NoPE form:** `O = softmax(softmax_scale * q . kv^T) . kv`.
- **NoPE form with a sink** (every NoPE entry of both modules, flat and varlen; `sm_100a`, `sm_103a`, `sm_107a`): `lse = log(sum_j exp(s_j) + exp(sink[h]))` and `O = sum_j exp(s_j - lse) . kv_j`, with `s_j = softmax_scale * q . kv_j`. The sink is not scaled and has no value term.

The autograd entries run the forward eagerly and the backward through autograd; the `*_train*_forward` / `*_train*_backward` step entries run them as two explicit calls without autograd. The backward recomputes scores from the BF16 inputs and returns `dq` plus `dkv`, or adds `dkv` into a caller's `dkv_acc`.

**Architectures.** `sm_100a`, `sm_103a`, `sm_107a` only.

**Inputs (rope form).**

- `q_latent`: BF16 `[T, 64, 512]`, contiguous or a view of a packed `[T, 64, 576]`.
- `q_rope`: BF16 `[T, 64, 64]`, contiguous, a view of a packed `[T, 64, 576]`, or the 192:256 slice of `[T, 64, 256]`.
- `kv_latent`: BF16 `[S, 512]`, or columns 0:512 of a packed `[S, 576]` / `[S, 704]`.
- `k_rope`: BF16 `[S, 64]`, or columns 512:576 of the packed row.
- `indices`: int32 `[T, topk]` global rows. `-1` or `>= S` marks an invalid slot. Any positive `topk` is allowed.
- `topk_length`: int32 `[T]`.
- `softmax_scale`: defaults to `576**-0.5`.
- Strided layouts need unit innermost stride, head/token/key-row strides that are positive multiples of 8 elements, and a 16-byte-aligned base.
- `dkv_acc`: FP32 `[S_dst, >= 576]` with a row stride that is a multiple of 4, 16-byte aligned and zeroed by the caller. It can also be a zero-argument callable that returns the buffer or `(acc, dst_map)`.
- `dkv_dst_map`: int32 `[S]` with values in `[0, S_dst)`. Repeated destinations are allowed, and the kernels do not range-check the values.
- `backend`: only `"cake"` is accepted (`ValueError` otherwise).

**Inputs (NoPE form).**

- `q`: BF16 `[T, 64, 512]`, contiguous or strided (head stride >= 512).
- `kv`: BF16 `[S, 512]` or `[S, 1, 512]`.
- `indices`: `[T, K_slots >= 1]`.
- `softmax_scale`: defaults to `1/16`.
- `dkv_acc`: FP32 `[S_dst, >= 512]`, with a row stride that is a multiple of 4 and >= 512.
- `dout` may be head-major (strides `[512, T*512, 1]`).
- `sink`: FP32 `[64]` (one value per head), contiguous, on `q`'s device, or `None` (the default: no sink). Only the metadata is checked, so there is no device sync. `-inf` disables the sink of its head. `+inf` and NaN are not supported. Every NoPE entry of both modules takes it, flat and varlen. With a sink, `lse` includes the sink, a row with no valid slot gives `out = 0` and `lse = sink[h]`, and the backward gives `dsink = -sum_t exp(sink - lse[t]) * delta[t]` (FP32 `[64]`, summed over all rows of the call, packed rows included, in a fixed order that does not depend on the SM count): autograd returns it as the `sink` gradient, and the step backwards return it as a third value. A `-inf` head gets exactly 0.
- `num_segments`: the packed segment count of the key row, a Python `int` >= 1 (default 1; `len(cu_seqlens_k) - 1` for the global rows of a packed batch). The flat entries and step backwards of both forms take it; the varlen entries pass it themselves. It selects the plan only.
- A call with no query rows (T = 0) launches nothing. The forwards return empty `out` / `lse` / `o_lo` (`None` for `o_lo` with nondet `with_o_lo=False`). The step backwards return an empty `dq`, a zero FP32 `dkv` (`None` with `dkv_acc`) and, with a sink, a zero `dsink`. With a sink, the call still checks the sink's metadata and device.

**Varlen forms.** Indices are document-relative and are offset by `cu_seqlens_k` on the device. The rope varlen form compacts each row. The NoPE varlen form compacts rows only with a caller `topk_length` or at or above `cake_backend.NOPE_CONVERT_SLOT_LIMIT` slots; below that it converts rows in place, keeping slot positions. The query segment is the tail of its key prefix. `causal=True` also invalidates keys after the query's own position. A row with every slot invalid gives `out=0`, `lse=-inf` and zero gradients. `max_seqlen_q` / `max_seqlen_k` are accepted for parity but not read on the host. `topk_length` is a position cut that is applied before compaction.

### `cake_pplx.cake_dsa_sparse_attention_det_train`

Every output (`out`, `lse`, `dq`, `dkv` / `dkv_acc` rows) repeats bit for bit for the same inputs, architecture and package version. `dK`/`dV` contributions are summed as integers on a power-of-two grid per destination row. The result therefore does not depend on SM count, grid, CUDA graph vs eager, checkpoint recompute, repeated `dkv_dst_map` rows or token order. Bits are not promised across versions or architectures, or against nondet. No alert. There is no KCP module. Source: [`kernels/dsa_sparse_attention/cake_pplx/cake_dsa_sparse_attention_det_train.py`](../kernels/dsa_sparse_attention/cake_pplx/cake_dsa_sparse_attention_det_train.py).

| Name | Signature | Inputs | Returns / notes |
|---|---|---|---|
| `cake_dsa_sparse_attention_det` | `(q_latent, q_rope, kv_latent, k_rope, indices, *, topk_length=None, softmax_scale=None, return_lse=False, key_passes=None, backend="cake", dkv_acc=None, dkv_dst_map=None, dkv_direct=None, check_dst_map=None, num_segments=1)` | Rope form. `key_passes` is `None` or 1..S. `dkv` keeps the bits of one pass, and `dq` is summed through an FP32 partial between passes. `dkv_direct` is accepted and validated but has no effect. `num_segments`: the packed segment count of the key row, a Python `int` >= 1 (`len(cu_seqlens_k) - 1` for the global rows of a packed batch). A count above 1 plans one key-range pass unless `key_passes` overrides, as the varlen entry does for the same rows. Every output repeats bit for bit for the same inputs, `key_passes` and `num_segments`. | Autograd. Returns BF16 `out [T, 64, 512]` (zero for fully masked rows), plus FP32 `lse [T, 64]` with `return_lse=True`. Gradients go to all four inputs. With `dkv_acc`, the `kv_latent`/`k_rope` gradients are `None` and are added into the buffer instead. |
| `cake_dsa_sparse_attention_det_varlen` | `(q_latent, q_rope, kv_latent, k_rope, gather_kv_indices, cu_seqlens_q, cu_seqlens_k, max_seqlen_q=None, max_seqlen_k=None, *, causal=False, topk_length=None, softmax_scale=None, return_lse=False, key_passes=None, backend="cake", dkv_acc=None, dkv_dst_map=None, dkv_direct=None, check_dst_map=None)` | `T = cu_seqlens_q[-1]` and `S = cu_seqlens_k[-1]`. `gather_kv_indices`: int32 `[T, topk]`; `-1` or `>= seqlen_k[d]` is invalid. The two prefix sums are independent. | Packed multi-document rope form. |
| `cake_dsa_sparse_attention_det_nope` | `(q, kv, indices, *, topk_length=None, softmax_scale=None, sink=None, return_lse=False, key_passes=None, backend="cake", dkv_acc=None, dkv_dst_map=None, dkv_direct=None, check_dst_map=None, num_segments=1)` | NoPE form. `key_passes` is `None` (one pass) or 1..S. `dkv` keeps the bits of one pass, and `dq` is summed through an FP32 partial between passes. `dkv_direct` is accepted and validated but has no effect. `num_segments` is validated; the NoPE policy plans one pass at every count. `sink`: FP32 `[64]` trainable per-head sink or `None`, as in the NoPE inputs above. | Autograd. Returns `out`, plus `lse` with `return_lse=True`. `dkv = dK + dV` is FP32. Without `dkv_acc`, autograd casts it to the `kv` leaf dtype. With `sink`, autograd returns `dsink` as the `sink` gradient; it repeats bit for bit too. |
| `cake_dsa_sparse_attention_det_nope_varlen` | `(q, kv, gather_kv_indices, cu_seqlens_q, cu_seqlens_k, max_seqlen_q=None, max_seqlen_k=None, *, causal=False, topk_length=None, softmax_scale=None, sink=None, return_lse=False, key_passes=None, backend="cake", dkv_acc=None, dkv_dst_map=None, dkv_direct=None, check_dst_map=None, check_cu_seqlens=False)` | `cu_seqlens_q/k`: int32 `[num_docs+1]` of one length on `q`'s device, starting at 0 and non-decreasing, with `cu_seqlens_q[-1]==T`, `cu_seqlens_k[-1]==S` and `seqlen_q <= seqlen_k` per document | Packed NoPE form. The structure of `cu_seqlens` is checked on every call. The values are checked only with `check_cu_seqlens=True`, which costs one blocking D2H copy and is skipped during graph capture. `sink` as the det NoPE entry; `dsink` is summed over all packed rows. |
| `cake_dsa_sparse_attention_det_train_forward` | `(q_latent, q_rope, kv_latent, k_rope, indices, *, topk_length=None, softmax_scale=None, backend="cake")` | As the det rope entry | Returns `(out, lse, o_lo)`: BF16 `out`, FP32 `lse`, BF16 `o_lo` (the rounding residual of `out`). Not autograd. Bitwise the autograd entry's `out` and `lse`. |
| `cake_dsa_sparse_attention_det_train_backward` | `(q_latent, q_rope, kv_latent, k_rope, indices, out, o_lo, lse, dout, *, topk_length=None, softmax_scale=None, key_passes=None, backend="cake", dkv_acc=None, dkv_dst_map=None, dkv_direct=None, check_dst_map=None, dkv_fp32=False, num_segments=1)` | `out`, `o_lo`, `lse` from the step forward on the same inputs; `dout` BF16 `[T, 64, 512]`; `dkv_acc` a tensor only; `num_segments` as the det rope entry | Returns `(dq_latent, dq_rope, dkv_latent, dk_rope)`; the last two are `None` with `dkv_acc`, and FP32 with `dkv_fp32=True`. `o_lo` is unchanged. Bitwise the autograd entry's gradients. |
| `cake_dsa_sparse_attention_det_train_nope_forward` | `(q, kv, indices, *, topk_length=None, softmax_scale=None, sink=None, backend="cake")` | As the det NoPE entry | Returns `(out, lse, o_lo)`. Not autograd. With `sink`, `lse` includes the sink. Pass the same `sink` to the step backward. |
| `cake_dsa_sparse_attention_det_train_nope_backward` | `(q, kv, indices, out, o_lo, lse, dout, *, topk_length=None, softmax_scale=None, sink=None, key_passes=None, backend="cake", dkv_acc=None, dkv_dst_map=None, dkv_direct=None, check_dst_map=None, num_segments=1)` | `out`, `o_lo`, `lse` from the NoPE step forward; `sink`: the sink that the step forward ran with; `num_segments` as the det NoPE entry | Returns `(dq, dkv)` with FP32 `dkv [S, 512]`, or `(dq, None)` with `dkv_acc`. With `sink`, returns `(dq, dkv, dsink)`, or `(dq, None, dsink)` with `dkv_acc`. |
| `cake_dsa_sparse_attention_det_train_nope_varlen_forward` | `(q, kv, gather_kv_indices, cu_seqlens_q, cu_seqlens_k, *, causal=False, topk_length=None, softmax_scale=None, sink=None, backend="cake", check_cu_seqlens=False)` | As the det NoPE varlen entry | Returns `(out, lse, o_lo, indices, topk_length)`: the forward outputs, plus the global rows and row lengths that the forward read. With `sink`, `lse` includes the sink. |
| `cake_dsa_sparse_attention_det_train_nope_varlen_backward` | `(q, kv, indices, out, o_lo, lse, dout, cu_seqlens_k, *, topk_length, softmax_scale=None, sink=None, key_passes=None, backend="cake", dkv_acc=None, dkv_dst_map=None, dkv_direct=None, check_dst_map=None)` | `out`, `o_lo`, `lse`, `indices`, `topk_length` from the varlen step forward; `sink`: the sink that the step forward ran with | As the NoPE step backward. `dsink` is summed over all packed rows. |

Other public names:

- `CakeDSASparseAttentionDetFunction`, `CakeDSASparseAttentionDetNopeFunction`, `CakeDSASparseAttentionDetNopeVarlenFunction`: loaded lazily.
- `CHECK_DST_MAP_ENV` (`"CAKE_PPLX_DSA_CHECK_DST_MAP"`).
- `BINDING_CACHE_CAPACITY_ENV` (`"CAKE_PPLX_DSA_TRAIN_BINDING_CACHE_CAPACITY"`).

### `cake_pplx.cake_dsa_sparse_attention_nondet_train`

The backward adds `dK`/`dV` into FP32 accumulators with unordered atomics, and no output is promised to repeat bit for bit. Under deterministic algorithms, the forward runs and the backward raises `RuntimeError` before its first backward kernel (`UserWarning` and run with `warn_only=True`). There is no KCP module. Source: [`kernels/dsa_sparse_attention/cake_pplx/cake_dsa_sparse_attention_nondet_train.py`](../kernels/dsa_sparse_attention/cake_pplx/cake_dsa_sparse_attention_nondet_train.py).

**`with_o_lo`** (`bool`, keyword-only, default `True`; every nondet entry, not det). With `True` the forward also writes `o_lo`, the BF16 rounding residual of `out`, and the backward forms `delta = rowsum(dout * (out + o_lo))` in FP32. This keeps the gradients accurate on peaked rows (rows whose softmax puts almost all weight on one key). With `False` the forward neither computes nor stores `o_lo` (64 KiB less written and saved per query token), and the backward forms `delta = rowsum(dout * out)` from the BF16 `out`, the form cuDNN's DSA backward uses; on peaked rows the gradient error grows to that class. `out` and `lse` have the same accuracy in both modes. With a sink, `dsink` follows that `delta`: the sink tests bound each head's `dsink` error by `5e-3 * max |dsink_ref|`, and `with_o_lo=False`, which derives `delta` from the BF16 `out`, by 1.5 times that (`7.5e-3 * max |dsink_ref|`); every other bound and the naive BF16 tier are the same in both modes. Any value other than a `bool` raises `ValueError`; no environment variable sets it. The step forwards return `None` in the `o_lo` slot with `False`, and the step backwards take `o_lo=None` with the same `with_o_lo=False`: `o_lo` is `None` exactly when `with_o_lo` is `False`, and a mismatch raises `ValueError` before any plan or launch (a default call with no query rows accepts `o_lo=None`).

| Name | Signature | Inputs | Returns / notes |
|---|---|---|---|
| `cake_dsa_sparse_attention_nondet` | `(q_latent, q_rope, kv_latent, k_rope, indices, *, topk_length=None, softmax_scale=None, return_lse=False, key_passes=None, backend="cake", dkv_acc=None, dkv_dst_map=None, dkv_direct=None, check_dst_map=None, num_segments=1, with_o_lo=True)` | `key_passes=None` applies the per-architecture policy of the registry record (see [Key-range passes for the DRAM regime](../kernels/dsa_sparse_attention/cake_pplx/_backends/cake_dsa_sparse_attention_nondet_train/README.md#key-range-passes-for-the-dram-regime)): on one-segment rows, `ceil(S*2304 B / 100 MiB)` passes when that exceeds 1 and the row is one chunk (T <= 4224 at top-k 2048) or, on `sm_103a`/`sm_107a`, a tail row (`2T <= S`, at least 131,072 keys; at most 3 passes on `sm_103a`); `sm_107a` runs a fixed 2 passes with a 131,072-key floor; a count above 4 runs one pass. `key_passes=1` forces one pass, and a larger value (<= S) forces that many. `num_segments` (default 1) is the packed segment count of the key row: a count above 1 plans one pass unless `key_passes` overrides, as the varlen entry does for the same rows, so pass `len(cu_seqlens_k) - 1` with the global rows of a packed batch. It is a Python `int` >= 1, read on the host (no device synchronization), and selects the plan only. `dkv_direct=None` reads `CAKE_PPLX_DSA_DKV_DIRECT`; `True`/`False` select direct vs accumulate-and-cast for `dkv_acc`. The `dkv_direct` rule is 4 keys per query token; on `sm_107a` the row also needs `T * K_eff < 112 S + 2^20`, with `K_eff = min(K_slots, ceil(S / num_segments))`. | Autograd. Returns `out`, plus `lse` with `return_lse=True`. `dkv` is accumulated in FP32 with atomics, then cast to BF16 or added into `dkv_acc`. |
| `cake_dsa_sparse_attention_nondet_varlen` | `(q_latent, q_rope, kv_latent, k_rope, gather_kv_indices, cu_seqlens_q, cu_seqlens_k, max_seqlen_q=None, max_seqlen_k=None, *, causal=False, topk_length=None, softmax_scale=None, return_lse=False, key_passes=None, backend="cake", dkv_acc=None, dkv_dst_map=None, dkv_direct=None, check_dst_map=None, with_o_lo=True)` | As det varlen | Packed rope form. |
| `cake_dsa_sparse_attention_nondet_nope` | `(q, kv, indices, *, topk_length=None, softmax_scale=None, sink=None, return_lse=False, key_passes=None, backend="cake", dkv_acc=None, dkv_dst_map=None, dkv_direct=None, check_dst_map=None, num_segments=1, with_o_lo=True)` | `key_passes=None` uses one pass, and a value > 1 forces passes. For this program, the `dkv_direct` rule is 1 key per query token; on `sm_107a` the row also needs `T * K_eff < 112 S + 2^20` or more than 131,072 keys (`cast_max_keys`), with `K_eff = min(K_slots, ceil(S / num_segments))`. `sink`: FP32 `[64]` trainable per-head sink or `None`. `num_segments` (default 1): on the NoPE form, only the `sm_107a` `dkv_direct` rule reads it. | `dq` is BF16. `dkv` is FP32 via atomics, handed over through `dkv_acc` or cast by autograd. With `sink`, `lse` includes the sink, and a row with no valid slot gives `out = 0` and `lse = sink[h]`. Autograd returns `dsink = -sum_t exp(sink - lse[t]) * delta[t]` (FP32 `[64]`, summed over all rows in a fixed order that does not depend on the SM count) as the `sink` gradient. A `-inf` head gets exactly 0. |
| `cake_dsa_sparse_attention_nondet_nope_varlen` | `(q, kv, gather_kv_indices, cu_seqlens_q, cu_seqlens_k, max_seqlen_q=None, max_seqlen_k=None, *, causal=False, topk_length=None, softmax_scale=None, sink=None, return_lse=False, key_passes=None, backend="cake", dkv_acc=None, dkv_dst_map=None, dkv_direct=None, check_dst_map=None, check_cu_seqlens=False, with_o_lo=True)` | As det. `sink` as the nondet NoPE entry; `dsink` is summed over all packed rows. | Packed NoPE form. Without `check_cu_seqlens`, only structural checks run (dtype, rank, length, contiguity, device). |
| `cake_dsa_sparse_attention_nondet_train_forward` | `(q_latent, q_rope, kv_latent, k_rope, indices, *, topk_length=None, softmax_scale=None, backend="cake", with_o_lo=True)` | As det | Returns `(out, lse, o_lo)`, or `(out, lse, None)` with `with_o_lo=False`. Not autograd. |
| `cake_dsa_sparse_attention_nondet_train_backward` | `(q_latent, q_rope, kv_latent, k_rope, indices, out, o_lo, lse, dout, *, topk_length=None, softmax_scale=None, key_passes=None, backend="cake", dkv_acc=None, dkv_dst_map=None, dkv_direct=None, check_dst_map=None, dkv_fp32=False, num_segments=1, with_o_lo=True)` | As det. `dkv_direct=None` reads `CAKE_PPLX_DSA_DKV_DIRECT`. `num_segments` as the nondet rope entry. | Returns `(dq_latent, dq_rope, dkv_latent, dk_rope)`. Under deterministic algorithms it raises like the autograd backward. |
| `cake_dsa_sparse_attention_nondet_train_nope_forward` | `(q, kv, indices, *, topk_length=None, softmax_scale=None, sink=None, backend="cake", with_o_lo=True)` | As det. `sink` as the nondet NoPE entry. | Returns `(out, lse, o_lo)`, or `(out, lse, None)` with `with_o_lo=False`. With `sink`, `lse` includes the sink. Pass the same `sink` to the step backward. |
| `cake_dsa_sparse_attention_nondet_train_nope_backward` | `(q, kv, indices, out, o_lo, lse, dout, *, topk_length=None, softmax_scale=None, sink=None, key_passes=None, backend="cake", dkv_acc=None, dkv_dst_map=None, dkv_direct=None, check_dst_map=None, num_segments=1, with_o_lo=True)` | As det. `sink`: the sink that the step forward ran with. `num_segments` as the nondet NoPE entry. | Returns `(dq, dkv)`, or `(dq, None)` with `dkv_acc`. With `sink`, returns `(dq, dkv, dsink)`, or `(dq, None, dsink)` with `dkv_acc`. `dsink` is FP32 `[64]`, summed over all rows. |
| `cake_dsa_sparse_attention_nondet_train_nope_varlen_forward` | `(q, kv, gather_kv_indices, cu_seqlens_q, cu_seqlens_k, *, causal=False, topk_length=None, softmax_scale=None, sink=None, backend="cake", check_cu_seqlens=False, with_o_lo=True)` | As det. `sink` as the nondet NoPE entry. | Returns `(out, lse, o_lo, indices, topk_length)`; `o_lo` is `None` with `with_o_lo=False`. With `sink`, `lse` includes the sink. |
| `cake_dsa_sparse_attention_nondet_train_nope_varlen_backward` | `(q, kv, indices, out, o_lo, lse, dout, cu_seqlens_k, *, topk_length, softmax_scale=None, sink=None, key_passes=None, backend="cake", dkv_acc=None, dkv_dst_map=None, dkv_direct=None, check_dst_map=None, with_o_lo=True)` | As det. `sink`: the sink that the step forward ran with. | Returns `(dq, dkv)`, or `(dq, None)` with `dkv_acc`. With `sink`, returns `(dq, dkv, dsink)`, or `(dq, None, dsink)` with `dkv_acc`; `dsink` is summed over all packed rows. |

Other public names:

- `CakeDSASparseAttentionNondetFunction`, `CakeDSASparseAttentionNondetNopeFunction`, `CakeDSASparseAttentionNondetNopeVarlenFunction`.
- `DKV_DIRECT_ENV`, `CHECK_DST_MAP_ENV`, `BINDING_CACHE_CAPACITY_ENV`.

**Options and environment.**

| Variable | Modules | Effect |
|---|---|---|
| `CAKE_PPLX_DSA_CHECK_DST_MAP` | both | Used when `check_dst_map=None`. Unset, empty or `0` (the default) means no check. Any other value validates the `dkv_dst_map` values on every backward, at the cost of one device sync, and raises `ValueError` on a violation. An explicit `True`/`False` ignores the variable. The check is skipped during graph capture. |
| `CAKE_PPLX_DSA_TRAIN_BINDING_CACHE_CAPACITY` | both | A positive integer, read at module import; a bad value raises `ValueError` at import. Sets the capacity of the backend LRU binding cache (default 64). |
| `CAKE_PPLX_DSA_DKV_DIRECT` | nondet | Used when `dkv_direct=None`. `auto` (the default) applies the size rule of the device's architecture (`min_keys_per_query` 4 for rope and 1 for NoPE; on `sm_107a` the row also needs `T * K_eff < 112 S + 2^20`, with `K_eff = min(K_slots, ceil(S / num_segments))`, or, for NoPE only, more than 131,072 keys). `1` forces the direct accumulate path, and `0` forces accumulate-and-cast. Other values raise `ValueError`. |

`key_passes` splits the backward main stage into key-range passes. In det it is a scheduling choice: `dkv` keeps the bits of one pass, and `dq` is summed through an FP32 partial between passes. With `key_passes=None`, det NoPE runs one pass. In nondet it forces or overrides the pass policy.

**Cross-family.** There is no runtime contract with other families. `kv_latent` may be columns 0:512 of a packed `[S, 704]` row that also holds a 128-channel indexer key. The varlen indices are document-relative with `-1` as invalid, and with `causal=True` a top-k selector does not need to enforce causality. No dsa_indexer object is consumed.

**Example.**

```python
import torch
from cake_pplx.cake_dsa_sparse_attention_det_train import cake_dsa_sparse_attention_det

T, S, topk = 320, 960, 256
bf16 = dict(device="cuda", dtype=torch.bfloat16)
q_latent = torch.randn(T, 64, 512, **bf16).requires_grad_()
q_rope = torch.randn(T, 64, 64, **bf16).requires_grad_()
kv_latent = torch.randn(S, 512, **bf16).requires_grad_()
k_rope = torch.randn(S, 64, **bf16).requires_grad_()
# int32 [T, topk] global key rows; -1 (or >= S) marks an invalid slot
indices = torch.argsort(torch.rand(T, S, device="cuda"), dim=1)[:, :topk].to(torch.int32).contiguous()

out = cake_dsa_sparse_attention_det(q_latent, q_rope, kv_latent, k_rope, indices)  # BF16 [T, 64, 512]
out.float().square().sum().backward()  # every gradient repeats bit for bit
# nondet: from cake_pplx.cake_dsa_sparse_attention_nondet_train import cake_dsa_sparse_attention_nondet (same call)
```

**Caveats.**

- The geometry is fixed: 64 query heads, latent dim 512 and rope dim 64 (NoPE uses 512 only). All operands are BF16.
- The flat forms apply no positional mask; the index rows alone define the key set. The varlen forms mask only with `causal=True`.
- Out-of-range `dkv_dst_map` values are caught only with `check_dst_map` or the environment variable.
- In NoPE without `dkv_acc`, the FP32 `dkv` is cast to BF16. To keep FP32, use `dkv_acc` or `cake_backend.backward_nope`.
- The NoPE sink (every NoPE entry of both modules) adds one saved tensor for the backward: the reference to the caller's `sink`. The backward workspace grows by `ceil(T/64) * 256 + 256` bytes of `dsink` scratch: `ceil(T/64) * 256` bytes of FP32 partials and a 4-byte counter, each region aligned to 256 bytes. In a head whose sink is far above every score, the true `out` and `dq` are below the smallest BF16 subnormal and come out as 0.
- In nondet, deterministic preparation (the `dout` copy, the binding plan and the JIT load) may run before the alert is raised.
- Nondet `with_o_lo=False` forms `delta` from the BF16 `out` alone: on peaked rows the gradient error grows to the class of cuDNN's DSA backward. The default keeps `o_lo`.
- The binding cache keys differ. Det binds by geometry (shape, stride, dtype, alignment, scale). Nondet binds by `data_ptr`, shape, stride, dtype and scale.
- The det backward reuses the saved `o_lo` storage as its integer accumulator when `S_dst` is up to about `27.7*T` rows, and allocates otherwise. A second backward with `retain_graph=True` recomputes the forward first.
- Whether duplicate key ids are allowed within one `indices` row is not stated.

---

## dense_gemm

**Computes.** Two GEMMs.

- **K1:** the GLM-5.2 BF16 dense projection GEMM `out[l,m,n] = sum_k A[l,m,k] B[l,k,n]` over strided views, with FP32 accumulation and BF16 or FP32 output. Training layouts map onto it without copies: forward `X[T,K] @ W[N,K].T`, input gradient `G[T,N] @ W[N,K]`, and weight gradient `G[T,N].T @ X[T,K]`.
- **K2:** the FP32 router-gate GEMM `out[m,n] = sum_k A[m,k] B[k,n]`, computed through split-BF16x3 emulation, with the same forward, dgrad and wgrad forms.

The det module reduces split-K/stream-K partials in a fixed order. The nondet module adds its own programs: a K1 weight-gradient split-K program whose partials are summed in finish order (fixed order at one or two splits; three or more come only from an explicit split count), K1 stream-K and one-wave programs, and K2 stream-K, K-order and exact-FP32 SIMT programs (the SIMT and K-order programs have a fixed order).

**Architectures.** `sm_100a`, `sm_103a`, `sm_107a`. `is_*_supported` checks the device only. Shape support is checked per call: an unsupported view raises `ValueError`, and a geometry without a registered program raises `NotImplementedError`.

**View contracts.**

- **K1 (BF16):**
  - The A operand needs unit stride on m or k, and the B operand needs unit stride on k or n.
  - The other matrix stride and the batch stride must be multiples of 8 elements, and views must start 16-byte aligned.
  - `out` is BF16 or FP32 with unit n stride and 16-byte aligned rows and batches. With `transposed_out`, `out` is `[L,N,M]` with unit m stride.
  - N must be a positive multiple of 8. M and K must be >= 1; they are runtime values with no padding.
- **K2 (FP32):** 2-D views, A with unit stride on m or k and B with unit stride on k or n. `out` is FP32 `[M,N]` with unit inner stride and 16-byte rows. N must be a multiple of 8.

These are plain functions, not `torch.autograd.Function` wrappers, so the caller wires them into autograd. All names are lazy re-exports from `cake_pplx._backends.cake_dense_gemm_{det,nondet}_train`.

### `cake_pplx.cake_dense_gemm_det_train`

For the same inputs, hardware and plan, outputs are bitwise identical on every run. Nothing raises or warns under deterministic algorithms. Source: [`kernels/dense_gemm/cake_pplx/_backends/cake_dense_gemm_det_train/cake_backend.py`](../kernels/dense_gemm/cake_pplx/_backends/cake_dense_gemm_det_train/cake_backend.py) (facade: [`cake_dense_gemm_det_train.py`](../kernels/dense_gemm/cake_pplx/cake_dense_gemm_det_train.py)).

| Name | Signature | Inputs | Returns / notes |
|---|---|---|---|
| `cake_dense_gemm_det_train_projection_forward` | `(X, W, out=None, *, enable_pdl=None)` | K1 | Returns `out[T, N] = X[T, K] @ W[N, K].T`, BF16 by default. Uses the cached convenience path. |
| `cake_dense_gemm_det_train_projection_dgrad` | `(G, W, out=None, *, out_dtype=None, enable_pdl=None)` | K1; `W` is read as the MN-major B operand | Returns `out[T, K] = G[T, N] @ W[N, K]`, BF16 or FP32. |
| `cake_dense_gemm_det_train_projection_wgrad` | `(G, X, out=None, *, out_dtype=None, sk="auto", cta_rows=None, enable_pdl=None)` | K1; `G`, `X` MN-major; `sk` is a bool or `"auto"` | Returns `out[N, K] = G[T, N].T @ X[T, K]` with a ragged reduction length T. A measured row rule picks the orientation. Without a rule, the padded-tile heuristic runs the swapped `X.T @ G` with a transposed store (for example for N < 256 and N = 576). |
| `cake_dense_gemm_det_train_projection` | `(A, B, out, *, transposed_out=False, enable_pdl=None, weights_static=False)` | K1, `[L,M,K]` or `[M,K]` etc. L and the batch strides are arbitrary. | Generic K1. The convenience path keeps one plan per device, stream and view geometry, then rebinds and launches. `weights_static=True` lets the swapped small-token program load B before the PDL wait. |
| `cake_dense_gemm_det_train_projection_prepare` | `(A, B, out, *, transposed_out=False, sk="auto", block_n=None, stages=None, epi=None, slots=None, cta_rows=None, pf=None, sk_max_units=None, sk_parts=None, promo=None, hints=None, group_m=None, f32_v8=None, quad_store=None, park=None, ovl=None, htail=None, b_swz=None, sk_exact=None, batch_group=None, store_ef=None, store_hint=None, pd=None, sh=None, sk_sync=None, sk_sync_m=None, sk_slab=None, cta1=None, cgrp=None, ovr=None, stt=None, pro=None, form=None, enable_pdl=None)` | K1. Prepare a new object when a shape, dtype, stride or tensor binding changes. | Returns `CakeDenseGemmDetProjectionLaunch`, which owns the stream-K partial slabs and slice counters. `launch()` writes `out` without allocating or synchronizing, and it can be captured into a CUDA Graph from its first call. The knobs default to the Cake launcher's choices. |
| `cake_dense_gemm_det_train_projection_wgrad_prepare` | `(G, X, out, *, sk="auto", cta_rows=None, enable_pdl=None)` | K1 | Prepared wgrad. The orientation follows the row's measured rule, or else the small-N swap rule. |
| `cake_dense_gemm_det_train_router_forward` | `(X, W, out=None, *, splits=None, enable_pdl=None)` | K2 | Returns `out[T, N] = X @ W.T` in FP32. `splits` defaults to `router_splits` for the device and row (production split 4 for the kk layout). |
| `cake_dense_gemm_det_train_router_dgrad` | `(G, W, out=None, *, enable_pdl=None)` | K2 | Returns `out[T, K] = G @ W` in FP32, with one split. |
| `cake_dense_gemm_det_train_router_wgrad` | `(G, X, out=None, *, splits=None, enable_pdl=None)` | K2 | Returns `out[N, K] = G.T @ X`, run as a swapped `X.T @ G` with a transposed store. Split-K partials are reduced on the host in a fixed order. |
| `cake_dense_gemm_det_train_router` | `(A, B, out, *, splits=None, enable_pdl=None)` | K2 | Generic K2, using the convenience path. |
| `cake_dense_gemm_det_train_router_prepare` | `(A, B, out, *, splits=None, enable_pdl=None)` | K2 | Returns `CakeDenseGemmDetRouterLaunch`. It allocates the BF16x3 stack, the FP32 residual scratch and the split-K workspace. `launch()` does not allocate. |

Other public names:

- `ARCHES`.
- `SUPPORTED_COMPUTE_CAPABILITIES`: maps `(10,0)`, `(10,3)` and `(10,7)` to their arches.
- `is_cake_dense_gemm_det_train_supported(device)`.
- `cake_dense_gemm_det_train_projection_plan`: a host-only K1 planner. It requires the `sm_count` and `l2_bytes` keywords and also takes the launcher knobs and `arch='sm_100a'`. Returns `(plan, a_desc, b_desc, out3)`.
- `cake_dense_gemm_det_train_router_plan(A, B, out, *, splits=None)`: returns `(plan, A_eff, B_eff)`.
- `CakeDenseGemmDetProjectionPlan`, `CakeDenseGemmDetProjectionLaunch`, `CakeDenseGemmDetRouterPlan`, `CakeDenseGemmDetRouterLaunch`.
- `cake_dense_gemm_det_train_convenience_cache_clear`: drops every cached convenience launch and K1 binding and releases the workspace arenas.

### `cake_pplx.cake_dense_gemm_nondet_train`

Same operations, view contracts, dtypes and accuracy budgets as det, except the promoted split-K programs (`_kc128`) of long BF16 weight gradients: their output meets the naive BF16 tier (the error of PyTorch's BF16 product against FP64), and they were not measured against det's budgets. Some routes add split-K partials in arrival order, so no row carries a run-to-run bitwise guarantee. Most other rows reuse the det programs, and for those rows `*_plan` / `*_prepare` may return det objects.

Under deterministic algorithms, only an order-varying launch alerts: K1 split-K at >= 3 splits, or a K2 partition that gives an element more than 2 partials. These come only from an explicit split count passed by the caller, not from the shipped default tables. The alert is `RuntimeError`, or `UserWarning` with `warn_only`. Source: [`kernels/dense_gemm/cake_pplx/_backends/cake_dense_gemm_nondet_train/cake_backend.py`](../kernels/dense_gemm/cake_pplx/_backends/cake_dense_gemm_nondet_train/cake_backend.py).

| Name | Signature | Inputs | Returns / notes |
|---|---|---|---|
| `cake_dense_gemm_nondet_train_projection_wgrad` | `(G, X, out=None, *, out_dtype=None, sk="auto", cta_rows=None, enable_pdl=None)` | K1 | A row of `K1_WGRAD_SK_RULES_BY_SM` runs stream-K. Otherwise, where the planner takes the row, it runs split-K, which sums partials in finish order and rounds once. Every other row runs det. |
| `cake_dense_gemm_nondet_train_projection_forward` | `(X, W, out=None, *, enable_pdl=None)` | K1 | Uses fixed-order programs, so a recomputed forward reproduces the result. Rows in `K1_FORWARD_RULES_BY_SM` use other knobs: their bits differ from det but stay the same on every run. |
| `cake_dense_gemm_nondet_train_projection_dgrad` | `(G, W, out=None, *, out_dtype=None, enable_pdl=None)` | K1 | Same bits as the det plan. |
| `cake_dense_gemm_nondet_train_projection` | `(A, B, out, *, transposed_out=False, enable_pdl=None, weights_static=False)` | K1 | A weight-gradient layout runs split-K where the planner takes the row. Other rows run the table program or det. |
| `cake_dense_gemm_nondet_train_projection_prepare` | `(A, B, out, *, splits=_UNSET, chains=_UNSET, promote=_UNSET, transposed_out=_UNSET, sk=_UNSET, <det planner knobs>=_UNSET, enable_pdl=_UNSET)` | K1. Knobs the caller does not set take the det defaults. | Returns `CakeDenseGemmNondetProjectionLaunch` (weight-gradient layout), `CakeDenseGemmNondetProgramLaunch` (program-table row) or `CakeDenseGemmDetProjectionLaunch`. A prepared launch must not run concurrently with itself. `reset_counters()` re-zeroes the tickets after an incomplete launch. |
| `cake_dense_gemm_nondet_train_projection_wgrad_prepare` | `(G, X, out, *, sk="auto", cta_rows=None, enable_pdl=None)` | K1 | Prepared form of the program that `projection_wgrad` runs. |
| `cake_dense_gemm_nondet_train_router_wgrad` | `(G, X, out=None, *, splits=None, enable_pdl=None)` | K2 | `splits` comes from the router-table partition by default. Rows without an entry run det at det's split count. |
| `cake_dense_gemm_nondet_train_router_forward` | `(X, W, out=None, *, splits=None, enable_pdl=None)` | K2 | This package's program runs only with an explicit `splits` (at most 2 partials per element, so the logits repeat) or in a `ROUTER_RULES` SIMT window. Otherwise det runs. |
| `cake_dense_gemm_nondet_train_router_dgrad` | `(G, W, out=None, *, enable_pdl=None)` | K2 | Exact-FP32 SIMT program in the router-table token windows, and det with one split otherwise. |
| `cake_dense_gemm_nondet_train_router` | `(A, B, out, *, splits=None, enable_pdl=None)` | K2 | Stream-K or SIMT where the planner takes the row, and det otherwise. |
| `cake_dense_gemm_nondet_train_router_prepare` | `(A, B, out, *, splits=None, enable_pdl=None)` | K2 | Stream-K for `kk`/`nn_t`, SIMT in its windows, or det's `router_prepare`. |

Other public names:

- `ARCHES`, `is_cake_dense_gemm_nondet_train_supported(device)`.
- `cake_dense_gemm_nondet_train_projection_plan`: takes the det knobs plus `splits`, `chains` and `promote`, all `_UNSET` by default. `promote` picks the promoted split-K program (`_kc128`: each split sums its K blocks in chunks of at most 128 blocks, added in order). By default, a one-chain BF16 transposed-store plan with more than 128 K blocks in a split uses it, except `sm_107a` `5120 -> 1024` BF16 (`K1_PROMOTE_EXEMPT_ROWS`), which keeps the program without promotion. `False` never uses it. `True` uses it on every plan of this package's split-K program, at any length, and raises `ValueError` if that plan has no promoted program; a row the planner keeps on det stays on det. `promote=None` or `False` does not skip the program tables of `cake_dense_gemm_nondet_train_projection_prepare` and does not reach the det planner.
- `cake_dense_gemm_nondet_train_router_plan(A, B, out, *, splits=None)`.
- `CakeDenseGemmNondetProjectionPlan`.
- `CakeDenseGemmNondetProjectionLaunch`: owns its FP32 slots and tickets, can be captured as one kernel, and has `reset_counters()`.
- `CakeDenseGemmNondetRouterPlan`.
- `CakeDenseGemmNondetRouterLaunch`: runs a split, then an optional memset of `out`, then the program.
- `cake_dense_gemm_nondet_train_convenience_cache_clear`: clears both packages' caches and arenas.

**Options and environment.**

- `enable_pdl`: defaults to true when the device supports it. Launches as a programmatic dependent launch.
- `weights_static`: projection only.
- `sk` / `cta_rows` on wgrad and `splits` on router override the split choice. In nondet, an explicit `splits` value can select an order-varying launch.
- `CONVENIENCE_CACHE_SIZE = 64`: a backend module setting shared by both packages. Neither facade exports it.
- No environment variable selects the mode.

**Cross-family.** None. The nondet README says context parallelism does not apply, because this is a local operator with no collectives. The nondet backend lazily imports the det backend package and requires it to be installed next to it (`build.json` `common.requires.packages`).

**Example.**

```python
import torch
from cake_pplx.cake_dense_gemm_det_train import (
    cake_dense_gemm_det_train_projection_forward,
    cake_dense_gemm_det_train_router_forward,
    cake_dense_gemm_det_train_router_wgrad,
)

device = torch.device("cuda")
T, K, N = 257, 6144, 576  # kv_a forward row
X = torch.randn(T, K, device=device).to(torch.bfloat16)
W = (torch.randn(N, K, device=device) * K**-0.5).to(torch.bfloat16)
out = torch.empty(T, N, device=device, dtype=torch.bfloat16)
cake_dense_gemm_det_train_projection_forward(X, W, out)  # out = X @ W.T

# FP32 router GEMM (BF16x3 emulation)
Wr = torch.randn(256, 6144, device=device)
Xr = torch.randn(16022, 6144, device=device)
Gr = torch.randn(16022, 256, device=device)
logits = cake_dense_gemm_det_train_router_forward(Xr, Wr)  # [T, 256] fp32
dWr = cake_dense_gemm_det_train_router_wgrad(Gr, Xr)       # [256, 6144] fp32
# nondet: same calls with the _nondet_train names
```

**Caveats.**

- There is no autograd Function.
- The nondet `*_plan` / `*_prepare` functions are annotated `-> Any` and may return det objects.
- Nondet: recompute a checkpointed forward with the same package that produced the saved one. Nondet results have no run-to-run bitwise guarantee, even on rows that currently run fixed-order programs.
- Clearing the convenience cache invalidates CUDA graphs that captured convenience-path calls. A nondet prepared launch must not run concurrently on two streams.
- Device facts (SM count, L2 size) are read once per device index. Some nondet tables are keyed by SM count (148/152/208/212/216-SM parts).

---

## moe_grouped_gemm

**Computes.** A ragged BF16 MoE grouped GEMM over groups of rows. The groups are given by an int32 device tensor of cumulative end offsets (`offs [E]`, or `m_indptr [E+1]`).

- Forward: `Y[offs[e-1]:offs[e]] = X[...] @ W[e].T`.
- Data gradient: `dX = G @ W[e]`.
- Weight gradient: `dW[e] = G[e].T @ X[e]`.

All three accumulate in FP32. The weight gradient has an optional accumulate mode, `out[e] += ...`, which implements Megatron's `gradient_accumulation_fusion` into `main_grad`.

**Architectures.** `sm_100a`, `sm_103a`, `sm_107a`. `is_*_supported` checks that a program is registered for each of `fwd`, `dgrad` and `wgrad`.

**Inputs.**

- `x`: BF16 `[sum_m, K]`. `w`: BF16 `[E, N, K]`. `g`: BF16 `[sum_m, N]`.
- `offs`: int32 on the device. It is never read on the host, so there is no host sync.
- Divisibility:
  - Forward: `N % 256 == 0` and `K % 64 == 0`.
  - Dgrad: `K % 256 == 0` and `N % 64 == 0`.
  - Wgrad: `N % 256 == 0` and K a multiple of the selected k tile (256 or 512).

### `cake_pplx.cake_moe_grouped_gemm_det_train`

Every launch reduces in a fixed order. `Y`, `dX` and `dW` are bitwise reproducible run to run, including with `accumulate=True` or `fuse_wgrad_accumulation=True`. Deterministic algorithms change nothing. Source: [`kernels/moe_grouped_gemm/cake_pplx/cake_moe_grouped_gemm_det_train.py`](../kernels/moe_grouped_gemm/cake_pplx/cake_moe_grouped_gemm_det_train.py).

| Name | Signature | Inputs | Returns / notes |
|---|---|---|---|
| `cake_moe_grouped_gemm_det` | `(x, w, offs, *, fuse_wgrad_accumulation=False)` | As above | Autograd. Returns BF16 `[sum_m, N]`. The backward gives BF16 `x.grad` and `w.grad` in `w`'s dtype. With `fuse_wgrad_accumulation=True`, it adds `dW` into `w.main_grad` (FP32 or BF16) in place and returns an empty/zeros placeholder or `None` for `w`. Not supported under `torch.jit.trace`, `torch.compile` or a `create_graph` backward (`NotImplementedError`). |
| `cake_moe_grouped_gemm_det_train_forward` | `(x, w, offs, out=None)` | `offs [E]` with `offs[-1] == sum_m`, or `m_indptr [E+1]`; `out` BF16, may be row-padded | Prepares and launches in one call. Returns `out`. |
| `cake_moe_grouped_gemm_det_train_dgrad` | `(g, w, offs, out=None)` | `out` BF16 `[sum_m, K]` | Returns `dX`. `W` is read in place, with no transpose copy. |
| `cake_moe_grouped_gemm_det_train_wgrad` | `(g, x, offs, out=None, out_dtype=None, num_groups=None, *, accumulate=False)` | `num_groups=E` allows `m_indptr [E+1]`. `out_dtype` is bf16 or fp32 and defaults to `out`'s dtype, else bf16. With `accumulate`, `out` is required and must not overlap `g`, `x` or `offs` (`ValueError`). | Returns `dW [E, N, K]`. Empty groups give exact zeros. `accumulate=True` adds into `out` with AccumulateGrad's bits. With `sum_m == 0`, `out` is unchanged. |
| `cake_moe_grouped_gemm_det_train_prepare_forward` | `(x, w, offs, out=None)` | As forward. The 512-column tile is used when `N % 512 == 0` on `wide_fwd_archs`. | Returns `CakeMoEGroupedGemmDetLaunch`; call `launch()`. Works with CUDA Graph capture. |
| `cake_moe_grouped_gemm_det_train_prepare_dgrad` | `(g, w, offs, out=None)` | The 512-column tile is used when `K % 512 == 0` on `wide_dgrad_archs` | Prepared dgrad. |
| `cake_moe_grouped_gemm_det_train_prepare_wgrad` | `(g, x, offs, out=None, out_dtype=None, num_groups=None, *, accumulate=False)` | As wgrad | Prepared `dW`. With `accumulate=True`, every `launch()` and every graph replay adds again. `plan['accumulate']` is `'fused'` or `'composed'`. The composed route uses an `E*N*K`-element scratch owned by the launch. |

Other public names:

- `is_cake_moe_grouped_gemm_det_train_supported(device=None)`: loads and launches nothing.
- `CakeMoEGroupedGemmDetFunction`: `apply(x, w, offs, None, fuse_wgrad_accumulation)`. A 4th argument other than `None` raises `TypeError`. A direct `apply` always recomputes under checkpoint.
- `CakeMoEGroupedGemmDetLaunch`: `launch()` returns `out` with no host sync and stays valid when `offs` or operand values change. Exposes `plan` and `stages`.

### `cake_pplx.cake_moe_grouped_gemm_nondet_train`

Nondet by name, but every shipped launch has a fixed order. Every output repeats bit for bit for the same inputs, shapes, device SM count and configuration, and nothing alerts. `Y` and `dX` are bitwise det. `dW` is bitwise det wherever the 256 k tile runs or the 512 tile has no split tail. A split tail uses a different tail plan and group order, so it can differ from det's bits. The tests judge every output against a naive BF16 implementation, with the FP64 product of the BF16 operands as the oracle. An alert path exists only for an order-varying `unordered_tail` plan on architectures in `unordered_tail_archs`, and that set is empty. Source: [`kernels/moe_grouped_gemm/cake_pplx/cake_moe_grouped_gemm_nondet_train.py`](../kernels/moe_grouped_gemm/cake_pplx/cake_moe_grouped_gemm_nondet_train.py).

| Name | Signature | Inputs | Returns / notes |
|---|---|---|---|
| `cake_moe_grouped_gemm_nondet` | `(x, w, offs, *, fuse_wgrad_accumulation=False)` | As det | Autograd. `dX` has det bits. `dW` sums the split tail in slot order in the same launch. Same restrictions as det. |
| `cake_moe_grouped_gemm_nondet_train_forward` | `(x, w, offs, out=None)` | As det | Copies of the det programs, with det bits. |
| `cake_moe_grouped_gemm_nondet_train_dgrad` | `(g, w, offs, out=None)` | As det | Det bits. |
| `cake_moe_grouped_gemm_nondet_train_wgrad` | `(g, x, offs, out=None, out_dtype=None, num_groups=None, *, accumulate=False)` | As det | FP32 accumulation with one rounding at the store. With the 512 k tile, a split tail is summed in slot order. |
| `cake_moe_grouped_gemm_nondet_train_prepare_forward` | `(x, w, offs, out=None)` | As det | Returns `CakeMoEGroupedGemmNondetLaunch`. |
| `cake_moe_grouped_gemm_nondet_train_prepare_dgrad` | `(g, w, offs, out=None)` | As det | Prepared dgrad. |
| `cake_moe_grouped_gemm_nondet_train_prepare_wgrad` | `(g, x, offs, out=None, out_dtype=None, num_groups=None, *, accumulate=False)` | As det | When `plan['ordered_tail']` is set and `tail_splits > 1`, each unit stores an FP32 slot and the last unit sums them in slot order. |

Other public names:

- `is_cake_moe_grouped_gemm_nondet_train_supported(device=None)`.
- `CakeMoEGroupedGemmNondetFunction`.
- `CakeMoEGroupedGemmNondetLaunch`: has an `order_varying` property. The tail-slot and ticket workspaces are kept per (device, stream).

**Options and environment.**

- `fuse_wgrad_accumulation`: `w.main_grad` must have `w`'s shape and unit last-dim stride. If `w` has `grad_added_to_main_grad`, `w` gets a placeholder `torch.empty`, or `torch.zeros` when `w.zero_out_wgrad` is true. Otherwise it gets `None`.
- `plan['accumulate_reason']` explains the route choice.
- Route-selecting backend constants: `wide_fwd_archs`, `wide_dgrad_archs`, `dgrad_evict_first_archs`, `fwd_lwh_ef_archs`, `fwd_evict_first_archs`, and, for nondet only, `ordered_tail_archs` and `unordered_tail_archs`.
- This family also needs `ninja`.

**Cross-family.** None. On a context-parallel rank it runs on that rank's local rows, adds no collective and has no KCP form.

**Example.**

```python
import math, torch
import cake_pplx.cake_moe_grouped_gemm_det_train as facade

device = torch.device("cuda")
sizes = [300, 0, 1, 700]  # rows per expert
n, k = 512, 256
sum_m = sum(sizes)
offs = torch.tensor(sizes, dtype=torch.int32, device=device).cumsum(0, dtype=torch.int32)
x = (torch.randn(sum_m, k, device=device) / math.sqrt(k)).to(torch.bfloat16).requires_grad_(True)
w = (torch.randn(len(sizes), n, k, device=device) / math.sqrt(k)).to(torch.bfloat16).requires_grad_(True)
g = torch.randn(sum_m, n, device=device).to(torch.bfloat16)

y = facade.cake_moe_grouped_gemm_det(x, w, offs)   # [sum_m, n] bf16, autograd
y.backward(g)                                        # x.grad [sum_m, k], w.grad [E, n, k]
dw32 = facade.cake_moe_grouped_gemm_det_train_wgrad(g, x.detach(), offs, out_dtype=torch.float32)
# nondet: cake_pplx.cake_moe_grouped_gemm_nondet_train.cake_moe_grouped_gemm_nondet(x, w, offs)
```

**Caveats.**

- Inputs are BF16 only.
- `fuse_wgrad_accumulation` is also not supported with split backward-weight schedules. A reentrant checkpoint gives a detached `w` without `main_grad` (`ValueError`). Every backward adds again.
- A prepared accumulate launch is not idempotent: each `launch()` and each graph replay adds again. Keep the launch alive while a captured graph may replay it. Do not run one composed launch on two streams at once.
- The composed accumulate scratch takes `E*N*K` elements (`4*E*N*K` bytes for fp32).
- A nondet `dW` with a split 512 tail may differ from det bits, but it repeats for the same SM count and configuration.

---

## gated_act_mxfp8

**Computes.** Fused SwiGLU plus MXFP8 quantization. `gated_input [M, 2K]` (BF16) holds the gate columns `[0,K)` and the up columns `[K,2K)`.

- The forward quantizes `silu(gate)*up` (`[M, K]`).
- The backward quantizes the gate and up gradients (`[M, 2K]`) from `grad_output [M, K]`.
- Each value is rounded to BF16, then quantized to E4M3 per 32-element block with an RCEIL E8M0 scale, rowwise and/or colwise. Scales use the swizzled 128x4 layout.
- The kernels are deterministic: the work is elementwise plus a block max, and outputs repeat bit for bit.

**Architectures.** `sm_100a`, `sm_103a`, `sm_107a`. Other devices raise `RuntimeError("fused gated MXFP8 quantization requires SM100, SM103 or SM107")`.

### `cake_pplx.cake_gated_act_mxfp8_det_train`

This family has a det module only, with no nondet or KCP module. Nothing is summed across threads, CTAs or calls, and no call alerts. There is no autograd Function; the forward and backward are separate plain calls. Source: [`kernels/gated_act_mxfp8/cake_pplx/cake_gated_act_mxfp8_det_train.py`](../kernels/gated_act_mxfp8/cake_pplx/cake_gated_act_mxfp8_det_train.py).

| Name | Signature | Inputs | Returns / notes |
|---|---|---|---|
| `cake_gated_act_mxfp8_det_train_forward` | `(gated_input, *, rowwise=True, colwise=False)` | `gated_input` contiguous BF16 CUDA `[M, 2K]`, with M and K positive multiples of 128, 32-byte aligned, and `M*2K-1` within int32. At least one of `rowwise`/`colwise` must be `True` (`ValueError`). | Returns `(row_output, col_output, row_scales, col_scales)`:<br>- `row_output`: `float8_e4m3fn [M, K]`, row-major<br>- `col_output`: `float8_e4m3fn [M, K]`, column-major (strides `(1, M)`)<br>- `row_scales`: `float8_e8m0fnu [M, K/32]`<br>- `col_scales`: flat `float8_e8m0fnu [K*(M/32)]`, swizzled 128x4<br>A disabled orientation returns zero-size tensors. |
| `cake_gated_act_mxfp8_det_train_backward` | `(gated_input, grad_output, *, rowwise=True, colwise=False)` | `grad_output` contiguous BF16 `[M, K]` on the same device, 32-byte aligned | Same tuple with N = 2K: E4M3 `[M, 2K]`, scales `[M, 2K/32]` and flat `[2K*(M/32)]`. Columns `[0,K)` hold the gate gradient and `[K,2K)` the up gradient. No intermediate BF16 result is materialized. |

Other public names:

- `ARCHES`, `SUPPORTED_COMPUTE_CAPABILITIES`.
- `CakeGatedActMXFP8DetOutputs`: the 4-tuple type alias.
- `cake_gated_act_mxfp8_det_train_register_custom_ops`: returns `SimpleNamespace(forward, backward)` wrapping the backend with positional `(gated_input, [grad_output,] rowwise, colwise)`. In this package the registration stand-ins are no-ops, so no `torch.ops` custom op is created.
- `is_cake_gated_act_mxfp8_det_train_supported(device)`.

**Options and environment.** The only keywords are `rowwise` and `colwise`. The program is selected by `M*K` alone, and every program of a route writes the same bytes. The backend reads no environment variable for kernel behavior.

**Cross-family.** None.

**Example.**

```python
import torch
from cake_pplx.cake_gated_act_mxfp8_det_train import (
    cake_gated_act_mxfp8_det_train_forward,
    cake_gated_act_mxfp8_det_train_backward,
)

M, K = 256, 384  # positive multiples of 128
gated_input = torch.randn(M, 2 * K, device="cuda", dtype=torch.bfloat16)
grad_output = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)

row, col, row_scales, col_scales = cake_gated_act_mxfp8_det_train_forward(
    gated_input, rowwise=True, colwise=True)
drow, dcol, drow_scales, dcol_scales = cake_gated_act_mxfp8_det_train_backward(
    gated_input, grad_output, rowwise=True, colwise=True)
```

**Caveats.**

- There is no autograd wrapper, so the caller must call the forward and backward explicitly.
- `col_output` is column-major, meaning its storage is the transpose. The scale tensors are swizzled 128x4 and returned as `float8_e8m0fnu` views.
- The former names (`silu_and_mul_mxfp8_quantize[_backward]`) are removed and have no alias.

---

## clamped_swiglu

**Computes.** The clamped SwiGLU of a routed MoE expert for training, intermediate width `I = 2304`, with the FP32 route weight applied before the one BF16 rounding. Per row `r < R` and column `c < I`, from the BF16 gate `g` and up `u`, the row's FP32 route weight `w` and the FP32 limit `L`:

- **Forward:** `g' = min(g, L)`, `u' = clamp(u, -L, L)`, `a = silu(g') * u'` in FP32, and `h = BF16(w * a)`, the only rounding.
- **Backward:** with `s = sigmoid(g')`, `d_gate = BF16(dh * w * u' * s * (1 + g' (1 - s)) * [g <= L])`, `d_up = BF16(dh * w * silu(g') * [-L <= u <= L])` and the FP32 `d_weight = sum_c dh * a`. The masks are inclusive: the gradient passes at `g = L` and `u = ±L`. `d_weight` uses the unweighted `a` (no division by `w`), so a zero-weight row gets exactly 0 in `h`, `d_gate` and `d_up` and keeps its `d_weight`.
- One kernel per call (PyTorch's NaN fills under deterministic algorithms aside; see Caveats). The backward needs only `gate`, `up`, `weight` and `dh`; it recomputes the activation.
- The sigmoid uses the approximate FP32 base-2 exponential and the approximate reciprocal with one round-to-nearest Newton step (below the flushed reciprocal's range a scaled reciprocal, within one subnormal ulp). Every product rounds to nearest without flushing subnormals; nothing is FP64.

**Architectures.** `sm_100a`, `sm_103a`, `sm_107a`. Other devices raise `RuntimeError`, also for `R = 0`. The `build.json` minimum CUDA toolkit is 12.8 for `sm_100a`, 12.9 for `sm_103a` and 13.5 for `sm_107a`.

**Inputs.**

- `gate`, `up`: BF16 `[R, 2304]` each, or one `[R, 4608]` tensor with the gate in columns `[0, 2304)` and the up in `[2304, 4608)` (pass it as the first argument with `up=None`). The kernels read every BF16 tensor through its row stride, so the halves of a merged tensor and row slices of wider tensors are read in place when the row stride is a multiple of 8 elements, the last stride is 1 and the base is 16-byte aligned. Other inputs are copied once.
- `weight`: FP32 `[R]`, `>= 0` (not checked), zeros allowed; `None` for an unweighted (shared) expert, `w = 1`.
- `dh`: BF16 `[R, 2304]`.
- `limit`: `L`, a finite positive Python number, rounded to FP32. It is required (no default) and is a launch argument, as are `R`, the weight flag, the strides and the column offsets of `up` and `d_up` (a merged input and its gradient are passed as both halves, the up halves at column 2304): one library per program and architecture serves every call, with no recompilation. Each stage has a small-`R` program and a fine program for the fewest rows, and the forward and the weighted backward also have a large-`R` program. The rows per SM of the device choose the program: the fine one up to a few rows per SM, the large-`R` one in its windows, the small-`R` one elsewhere. The programs of a stage give the same bits. The unweighted backward's programs give the bits of the weighted backward's programs at `w = 1`; they take no weight and write no weight gradient.
- `R` is any count from 0 to `2**31 - 257`. Every operand is on one CUDA device.
- Every argument is checked before an input is copied or an output allocated, so a call refused for its arguments copies, allocates and launches nothing.

### `cake_pplx.cake_clamped_swiglu_det_train`

This family has a det module only, with no nondet or KCP module: every output is elementwise or a sum within one row in one fixed order (each lane's two FMA chains over its even and odd columns, added, then XOR shuffles across the row's 32 lanes). Every call gives the same bits for the same inputs on the same architecture, whatever `R`, the layout, the stream or a CUDA graph, and no call alerts under `torch.use_deterministic_algorithms(True)`. Source: [`kernels/clamped_swiglu/cake_pplx/cake_clamped_swiglu_det_train.py`](../kernels/clamped_swiglu/cake_pplx/cake_clamped_swiglu_det_train.py).

| Name | Signature | Inputs | Returns / notes |
|---|---|---|---|
| `cake_clamped_swiglu_det` | `(gate_up_or_gate, up=None, weight=None, *, limit)` | As above | Autograd. Returns contiguous BF16 `h [R, 2304]`, the down projection's input. A merged input's gradient is one `[R, 4608]` tensor; `weight=None` has no weight gradient; `limit` gets none. Saves only its inputs. In a non-reentrant checkpoint's recomputation, launches the forward after autograd saved them (see Caveats, checkpointing). |
| `cake_clamped_swiglu_det_train_forward` | `(gate_up_or_gate, up=None, weight=None, *, limit, out=None)` | `out` optional preallocated BF16 `[R, 2304]` (the layout rules above, rows not overlapping) | Returns `h` (`out` when given). Not autograd. `R == 0` returns an empty `h` without a launch. |
| `cake_clamped_swiglu_det_train_backward` | `(grad_out, gate_up_or_gate, up=None, weight=None, *, limit, grad_gate_up_or_gate=None, grad_up=None, grad_weight=None)` | `grad_out` BF16 `[R, 2304]`; optional preallocated outputs: `grad_gate_up_or_gate` (`d_gate [R, 2304]`, or the merged `[R, 4608]` gradient), `grad_up` (two-tensor form only), `grad_weight` contiguous FP32 `[R]` (needs `weight`) | Returns `(d_gate_up, None, d_weight)` for a merged input or `(d_gate, d_up, d_weight)`; `d_weight` is None without `weight`. Not autograd. Unfit output buffers raise `ValueError` (never copied). |

Other public names:

- `ARCHES`, `SUPPORTED_COMPUTE_CAPABILITIES`, `INTERMEDIATE_SIZE` (2304).
- `CakeClampedSwiGLUDetFunction`: the autograd class, `apply(gate_up_or_gate, up, weight, limit)`. A direct `apply` launches the forward inside the function's `forward`, so a checkpoint always recomputes it.
- `is_cake_clamped_swiglu_det_train_supported(device)`.

**Options and environment.** `limit` and the optional output buffers are the only keywords. The facade reads no environment variable.

**Cross-family.** None. The gate/up projections and the down projection around it are moe_grouped_gemm's: the merged `[R, 4608]` form takes the gate/up projection output as is, and `h` is the down projection's input.

**Example.**

```python
import torch
import torch.utils.checkpoint
from cake_pplx.cake_clamped_swiglu_det_train import cake_clamped_swiglu_det

R, I = 4097, 2304
gate_up = torch.randn(R, 2 * I, device="cuda", dtype=torch.bfloat16, requires_grad=True)
weight = (1.5 * torch.rand(R, device="cuda")).requires_grad_()  # FP32 route weights

h = cake_clamped_swiglu_det(gate_up, weight=weight, limit=10.0)  # BF16 [R, 2304]
h.backward(torch.randn_like(h))  # gate_up.grad BF16 [R, 4608], weight.grad FP32 [R]

# Activation checkpointing: checkpoint the op together with the down projection.
w_down = (torch.randn(4096, I, device="cuda") / I**0.5).bfloat16().requires_grad_()  # BF16 [hidden, 2304]


def swiglu_down(gate_up, weight, w_down):
    return torch.nn.functional.linear(cake_clamped_swiglu_det(gate_up, weight=weight, limit=10.0), w_down)


y = torch.utils.checkpoint.checkpoint(swiglu_down, gate_up, weight, w_down, use_reentrant=False)
y.backward(torch.randn_like(y))  # the backward recomputes h once, for the down projection's weight gradient
```

**Caveats.**

- The intermediate width is 2304 only.
- Checkpointing: the op keeps only its inputs, so checkpointing it alone saves no memory. The down projection keeps `h` for its weight gradient. Checkpoint the op together with the down projection (as in the example) so that `h` is not kept from the forward to the backward, which recomputes it once. A region of the gate/up projection and the op, with the down projection outside, does not keep the projection's `[R, 4608]` output.
- In the backward of a non-reentrant checkpoint, the recomputation runs with saved-tensors hooks set. There the forward kernel runs after autograd saved the inputs, as for a built-in op; anywhere else (no hooks, or not in a backward, such as the checkpoint's own forward pass) it runs inside the autograd function's `forward`, as up to 0.1.7. With early stop on (the torch default), the recomputation therefore stops before that launch when nothing later in the region saves a tensor. When a later operation saves one, the recomputation runs the forward. The launch and the bits are the same in every case. Per checkpointed region (down projection: `F.linear(h, w_down)`, which saves `h`):

  | Region | Not kept from the forward to the backward | Forward of this op in the backward |
  |---|---|---|
  | The op alone (down projection outside) | Nothing (the op keeps its inputs, the down projection `h`) | None (0.1.7: one) |
  | Gate/up projection, then the op (down projection outside) | The projection's `[R, 4608]` output | None (0.1.7: one); the projection is recomputed |
  | The op, then the down projection | `h` | One, as in 0.1.7 |
  | The whole block (gate/up projection, op, down projection) | The projection's output and `h` | One, as in 0.1.7 |
  | Any region with `use_reentrant=True`, `early_stop=False` (or `set_checkpoint_early_stop(False)`), or a direct `CakeClampedSwiGLUDetFunction.apply` | As above | One, also when its output is dropped |

  A call that Dynamo or `torch.jit.trace` records launches inside `forward`; a checkpoint's recomputation runs with Dynamo disabled. Not tested under `torch.compile`.
- Workspace: none. Retained state of the autograd function: its inputs only (references, no copy). An unweighted backward allocates no weight gradient. An unweighted forward passes, as the ignored weight, one cached one-element FP32 buffer per device, set to 1.
- Under `torch.use_deterministic_algorithms(True)` with `torch.utils.deterministic.fill_uninitialized_memory` (on by default), PyTorch fills every tensor a call allocates with NaN before the kernel writes it: one more fill launch per new output and per copied input. Output buffers allocated once and passed to every call avoid the fills of those outputs. The bits do not change.
- No double backward: differentiating gradients computed with `create_graph=True` raises `RuntimeError`. No MXFP8 or FP4 output.
- `limit` must be a Python number: a tensor `limit` raises `TypeError`, as reading it would synchronize the host.
- A program's library builds at its first launch. The row count picks the program, and in the backward so does the weight: an unweighted backward runs its own programs. Before capturing a CUDA graph, run an eager call of each kind the graph captures (weighted and unweighted) at each row count it captures, or build every library of the architecture (the backend's `specs_for_arch`), so that no JIT build runs inside capture.

---

## engram_lookup

**Computes.** The Engram row lookup for training: rows of a quantized table, gathered by row id and written as BF16.

- `weight` holds FP8 E4M3 values `[R, 256]` and `scale` one E8M0 scale per 32 columns `[R, 8]`. For each looked-up row `p` with `r = row_ids[p]`, group `g` in `[0, 8)` and `j` in `[0, 32)`: `out[p, 32*g + j] = BF16(float32(weight[r, 32*g + j]) * float32(scale[r, g]))`.
- Each element is the exact product rounded once to BF16 (round to nearest even), as an FP64 reference rounds it. FP32-subnormal products are kept, products past the FP32 range are `±inf`, and the E8M0 byte `0xFF` (NaN) makes its 32 columns NaN.
- One kernel runs on the current stream and reads only the looked-up rows, so time and memory follow the number of looked-up rows, not `R`.
- The family has no backward and no autograd entry. The table's gradient does not depend on its storage format; the caller sums it.

**Architectures.** `sm_100a`, `sm_103a`, `sm_107a`. Other devices raise `RuntimeError` (an empty lookup returns before that check). The `build.json` minimum CUDA toolkit is 12.8 for `sm_100a`, 12.9 for `sm_103a` and 13.5 for `sm_107a`.

**Inputs.**

- `weight`: `float8_e4m3fn [R, 256]`. `scale`: `float8_e8m0fnu [R, 8]` with `weight`'s `R`.
- Each of `weight` and `scale` has a unit column stride, a row stride that is a multiple of 8 elements and a start on an 8-byte boundary. They are two contiguous tensors, or two views into one packed `uint8 [R, 264]` buffer (row stride 264), as rows fetched from other ranks arrive. Both layouts give the same output bytes.
- `row_ids`: contiguous int64 of any shape, fewer than 2^31 - 8 elements: `[T, 24]` for a direct lookup, `[M]` for deduplicated or routed rows. Values in `[0, R)`; repeats are allowed. An id outside `[0, R)` (a negative id included) stops the launch with a device-side trap before its row is read: the call or the next synchronizing CUDA call raises, and the CUDA context is unusable afterwards. The host checks no id (that would synchronize).
- All operands are on one CUDA device.

### `cake_pplx.cake_engram_lookup_det_train`

This family has a det module only, with no nondet or KCP module: a gather has no sums, so det is the only mode, and the lookup is a single-rank local lookup. Every output element is one product rounded once, so the output repeats bit for bit, for both layouts and every `row_ids` shape. No call reads torch's deterministic-algorithms setting, and no call alerts under `torch.use_deterministic_algorithms(True)`. Source: [`kernels/engram_lookup/cake_pplx/cake_engram_lookup_det_train.py`](../kernels/engram_lookup/cake_pplx/cake_engram_lookup_det_train.py).

| Name | Signature | Inputs | Returns / notes |
|---|---|---|---|
| `cake_engram_lookup_det_train_forward` | `(weight, scale, row_ids)` | As above. | Returns a new contiguous BF16 `[*row_ids.shape, 256]` that does not require a gradient. Not autograd; it works inside `torch.utils.checkpoint(..., use_reentrant=False)`. An empty `row_ids` (`T == 0` or `M == 0`) returns an empty output without a launch. Workspace: none. The call allocates only the output (`row_ids.numel() * 512` bytes) and retains no state. Two programs per architecture serve every `R`, `T` and `M` and both layouts, with no recompilation and no host synchronization: a call of at most `SMALL_MAX_POSITIONS` looked-up rows (the backend's `cake_jit.py`) launches the small one, a larger call the main one, and both write the same bytes. The first call on each device loads both programs and launches both on its operands (the one its row count does not pick first), so that a CUDA graph capture after it loads nothing, whichever program the captured call takes; make that first call outside a capture (a call captured before it launches its own program only, and the next call outside a capture loads both). A malformed call raises `ValueError` before any launch. |
| `cake_engram_lookup_det_train_split_packed` | `(buf)` | `buf`: `uint8 [R, 264]` with a unit column stride, on any device | Returns `(weight, scale)`: `buf[:, :256]` as `float8_e4m3fn` and `buf[:, 256:]` as `float8_e8m0fnu`. Views that share `buf`'s storage and row stride; nothing is copied. Another dtype, rank, width or column stride raises `ValueError`. |

**Options and environment.** The entries take no keyword argument. The facade reads no environment variable.

**Cross-family.** None.

**Example.**

```python
import torch
from cake_pplx.cake_engram_lookup_det_train import (
    cake_engram_lookup_det_train_forward,
    cake_engram_lookup_det_train_split_packed,
)

R, T = 1_000_000, 4096
# A local shard: E4M3 values and one E8M0 scale per 32 columns (the byte 127 is the scale 1.0).
weight = (torch.randn(R, 256, device="cuda") * 8).clamp(-448, 448).to(torch.float8_e4m3fn)
scale = torch.randint(120, 135, (R, 8), device="cuda", dtype=torch.uint8).view(torch.float8_e8m0fnu)
row_ids = torch.randint(0, R, (T, 24), device="cuda")
rows = cake_engram_lookup_det_train_forward(weight, scale, row_ids)  # BF16 [T, 24, 256]
x = rows.flatten(-2)                                                 # BF16 [T, 6144]

# Rows fetched from other ranks: one packed uint8 [M, 264] buffer.
M = 2048
fetched = torch.empty(M, 264, device="cuda", dtype=torch.uint8)
fetched[:, :256] = weight[:M].view(torch.uint8)
fetched[:, 256:] = scale[:M].view(torch.uint8)
w, s = cake_engram_lookup_det_train_split_packed(fetched)
out = cake_engram_lookup_det_train_forward(w, s, torch.arange(M, device="cuda"))  # BF16 [M, 256]
```

**Caveats.**

- Forward only: there is no backward and no autograd entry.
- The row width is 256 with 8 scales per row.
- `row_ids` must lie in `[0, R)`; see Inputs for what an id outside that range does.
- `weight` and `scale` need a unit column stride, a row stride that is a multiple of 8 elements and an 8-byte aligned start. A packed `[R, 264]` buffer from the caching allocator meets these rules.

---

## mok

**Computes.** One expert-parallel MoE layer for training (Mixture of Kittens): the routed-token scheduler, the fused forward (dispatch, gate/up GEMMs, SwiGLU, down GEMM, combine), the context-only recompute and the fused backward (`dx`, router-score gradient, six expert-weight gradients).

- Routed experts run in BF16, or in MXFP8 with caller-prequantized weights. The shared expert runs in BF16.
- `swiglu_limit` (a per-call value) selects clamped SwiGLU, `silu(min(gate, L)) * clamp(up, -L, L)`.
- With `fp32_wgrad=True`, the backward adds every weight gradient into six caller-owned FP32 accumulators.
- `router_weight_position="before_down"` applies each route's FP32 router weight `w` to the SwiGLU activation `a` before the down projection, `h = BF16(w * a)`, and sums the routed outputs unweighted; the router-weight gradient is `dot(dh, a)` (`dh` the gradient of `h`), defined at a zero weight. It runs top-k 6 over 384 routed experts (and top-k 2 on the diagnostic layouts) with clamped SwiGLU, FP32 weight gradients and BF16 routed weights. The default `"after_down"` weights each routed output in the combine.
- Every output repeats bit for bit from run to run, within the scope in the backend README (same inputs, weights, configuration other than the communication-SM split, precision, router-weight position, EP group, build and GPU model).

**Architectures.** `sm_100a`, `sm_103a`, `sm_107a`; `router_weight_position="before_down"`: `sm_100a` and `sm_103a`. Minimum CUDA toolkit: 12.8, 12.9 and 13.5. The MXFP8 schedule is 256 x 256 tiles on `sm_100a` / `sm_103a` and 256 x 512 tiles on `sm_107a`.

**Distributed requirements.** The caller passes an initialized `torch.distributed` process group with its CUDA device current. All ranks share one peer-accessible NVLink domain with multicast (PyTorch symmetric memory). EP sizes: 1, 4, 8, 16, 32 and 64. EP1 uses plain tensors.

### `cake_pplx.cake_mok_det_train`

This family has a det module only. The backend package is `cake_pplx._backends.cake_mok_bf16`. Source: [`kernels/mok/cake_pplx/cake_mok_det_train.py`](../kernels/mok/cake_pplx/cake_mok_det_train.py).

| Name | Signature | Returns / notes |
|---|---|---|
| `prepare_mok_bf16` | `(*, ep_size=16, local_experts=16, topk=8, clamped_swiglu=False, fp32_wgrad=False, router_weight_position="after_down")` | An adapter with `prepare`, `prepare_recompute`, `create_workspace`, `build_schedule`, `forward`, `recompute_forward_context` and `backward`. It compiles one BF16 variant at once; call `prepare` / `prepare_recompute` for any other variant before CUDA Graph capture. With `router_weight_position="before_down"`, `recompute_forward_context` takes `router_weights` (the forward's). An unsupported combination raises before any kernel is loaded or launched: `TypeError` for a position that is not a `str`, an `ep_size`, `local_experts` or `topk` that is not an `int` (a `bool` included) and a `clamped_swiglu` or `fp32_wgrad` that is not a `bool`, in both positions; `ValueError` for every other value. |
| `create_mok_bf16_workspace` | `(*, group, device, num_local_tokens, hidden_size, topk, source_capacity=None, fwd_num_comm_sms="auto", bwd_num_comm_sms="auto", minibatch_size=4096, macrobatch_size=32768, schedule_capacity_multiplier=3/16, precision="bf16", imbalance_comm_sms="auto", imbalance_threshold="auto")` | Collective. Returns `(config, workspace)`. Call it outside CUDA Graph capture. |
| `quantize_mok_mxfp8_weights` | `(gate, up, down)` | `(forward_weights, backward_weights)` of BF16 routed weights `gate`/`up [E_local, I, H]`, `down [E_local, H, I]` in the MXFP8 recipe. PyTorch operations; requantize after every weight update. |

Other public names: `COMM_SMS`, `IMBALANCE_COMM_SMS`, `IMBALANCE_DEVICE_THRESHOLDS`, `IMBALANCE_THRESHOLD`, `IMBALANCE_THRESHOLDS`, `MOK_PRECISIONS`, `ROUTER_WEIGHT_POSITIONS`, `resolve_comm_sms`, `resolve_imbalance_comm_sms`, `resolve_imbalance_threshold` (loaded from the backend on first access).

**Contract.** Scheduler layouts `(EP, local experts)`: `(1, 4)`, `(4, 4)`, `(4, 64)`, `(8, 32)`, `(16, 16)`, `(32, 8)`, `(64, 4)`, `(8, 36)`, `(32, 9)`; top-k 2 or 8. With `router_weight_position="before_down"`: `(4, 96)`, `(8, 48)`, `(16, 24)`, `(32, 12)`, `(64, 6)` at top-k 6, `(1, 4)` and `(4, 4)` at top-k 2; `clamped_swiglu=True` and a `swiglu_limit` on every call, `fp32_wgrad=True`, BF16 routed weights, router weights `>= 0`. A `schedule_capacity_multiplier` of at least `ceil((min(topk, E) * T * EP + E * 255) / (T * topk)) / EP` (`E` local experts, `T` the workspace's physical source rows, at least the largest rank's source rows or `source_capacity`) holds every routing; no upper bound is enforced. Every rank passes the same multiplier, so compute it from a row bound all ranks agree on, such as `source_capacity`: the bound decreases as `T` grows. Hidden and intermediate widths are multiples of 256 (MXFP8: hidden a multiple of 512). Source counts may differ across ranks, including zero. Communication-SM counts are positive and even. The locality-domain (`_domain`) builds are not shipped. The backend README (`kernels/mok/cake_pplx/_backends/cake_mok_bf16/README.md`) holds the full contract, the memory and recompute rules and the measured communication-SM tables.

**Options and environment.** The options are the keywords of `prepare_mok_bf16` and `create_mok_bf16_workspace` above. The facade reads no family-specific environment variable; the JIT variables in the README apply.

**Cross-family.** None.

**Example.**

```python
from cake_pplx.cake_mok_det_train import (
    create_mok_bf16_workspace,
    prepare_mok_bf16,
)

config, workspace = create_mok_bf16_workspace(
    group=group, device=device, num_local_tokens=n, hidden_size=6144, topk=8,
)
backend = prepare_mok_bf16(ep_size=16, local_experts=16, topk=8)
schedule = backend.build_schedule(workspace, config, ids, num_local_experts=16)
y, context = backend.forward(config, workspace, schedule, x, scores, *weights)
grads = backend.backward(config, workspace, schedule, context, dy, x, scores, *weights)
```

**Caveats.**

- This is the MoE layer only. Optimizer steps, router-network backpropagation and the all-reduce of shared-expert gradients are outside the adapter; shared-expert weight gradients stay local to each source rank.
- All ranks must make matching collective calls. Keep workspaces and peer mappings alive while the adapter or a captured graph can use them, and use a separate workspace and adapter for concurrent executions.
- Prepare and warm up before CUDA Graph capture. Pass the matching context and schedule to `backward`.
- The MXFP8 forward always saves its context and rejects `save_context=False`.
- Schedule capacity must hold every padded route; an overflow traps.
- A context serves one `backward`: when the routed rows exceed one macrobatch, `backward` replays the later macrobatches into the context's routed ring tensors, in either router-weight position. Do not read a context after its `backward`; call `recompute_forward_context` again before another `backward`.
- `router_weight_position="before_down"` has no `sm_107a` build, no plain SwiGLU, no MXFP8 routed weights and no BF16 weight-gradient accumulation; the package ships no autograd wrapper (a checkpointing caller wraps forward, recompute and backward in its own `torch.autograd.Function`).

---

## lm_head

**Computes.** A chunked LM-head projection `z = X @ W^T` over token chunks, with the per-row selected-token log-probability `logp_t = z[t, y_t] - logsumexp_v z[t, v]`. The objective is one of:

- cross-entropy: `-sum(logp[valid]) / loss_div`
- a clipped policy objective: `-sum(loss_weights * min(exp(logp - infer_logp), 2))`

No vocab-sized tensor larger than `chunk_size` rows is ever materialized.

- The loss forward already forms the FP32 gradient accumulators. The backward applies the scalar grad and casts.
- The logprob backward recomputes the logits per chunk.
- The family also computes log-probabilities, and the unreduced forward KL from a detached teacher, over a ragged per-token candidate support, with an optional per-row temperature.

**Architectures.** `sm_100a`, `sm_103a`, `sm_107a`. The `build.json` minimum CUDA toolkit is 12.9 for `sm_100a`/`sm_103a` and 13.5 for `sm_107a`.

**Inputs.**

- `X`: BF16 `[T, H]`, row-major with any leading stride. H must be a multiple of 256, and a pitch that is not 16-byte aligned is copied to contiguous.
- `W`: BF16 `[V, H]`, contiguous. V must be a multiple of 256.
- `labels`: int64 `[T]`; `-100` marks an ignored row.
- `backend`: only `'cake'` is accepted (`ValueError` otherwise).

### `cake_pplx.cake_lm_head_det_train`

Chunks run in a fixed sequential order with no atomics, so the loss, `logp`, `dX` and `dW` are bitwise reproducible run to run. Deterministic algorithms change nothing and never alert. There is no KCP module. Source: [`kernels/lm_head/cake_pplx/cake_lm_head_det_train.py`](../kernels/lm_head/cake_pplx/cake_lm_head_det_train.py).

| Name | Signature | Inputs | Returns / notes |
|---|---|---|---|
| `cake_lm_head_det_train_loss` | `(X, W, labels, *, objective="ce", loss_div=None, infer_logp=None, loss_weights=None, chunk_size=4096, return_logp=False, grad_weight_dtype=torch.bfloat16, backend="cake", compact_rows=None, fuse_dw_cast=None, dx_finalize=None, dw_stream=None, hidden_count=None, gemm_tuning=None, memory_mode=None, num_valid_rows=None, check_num_valid_rows=False)` | `'ce'` requires a positive scalar `loss_div`. `'policy'` requires `infer_logp` FP32 `[T]` and `loss_weights` FP32 `[T]`. `grad_weight_dtype` is bfloat16 or float32 but must equal `W.dtype` through autograd. `memory_mode` is `None` (automatic), `'fast'` or `'budget'`; `'keep_logits'` raises `ValueError`. `num_valid_rows` is a host int (a CUDA tensor is rejected). `check_num_valid_rows=True` without `num_valid_rows` raises `ValueError`. | Autograd. Returns the FP32 scalar loss, plus a detached FP32 `logp [T]` with `return_logp=True`. Gradients go to `X` and `W` (BF16, FP32 accumulation, one cast), only for inputs that require them. T == 0 gives loss 0. |
| `cake_lm_head_det_train_logprob` | `(X, W, labels, *, chunk_size=4096, backend="cake", compact_rows=None, fuse_dw_cast=None, dx_finalize=None, dw_stream=None, hidden_count=None, gemm_tuning=None, memory_mode=None, num_valid_rows=None, check_num_valid_rows=False, temperature=None, check_temperature=False)` | `memory_mode='keep_logits'` keeps the BF16 logits of valid rows (`2*T_v*V` bytes) from forward to backward and takes no temperature. `temperature` is `None` (= 1.0), a positive number, or a float32 `[T]` tensor on `X`'s device (any stride, not `requires_grad`). Every value must be positive and finite, including on ignored rows. | Autograd. Returns FP32 `logp [T]`, with 0 on label `-100` rows. With temperature: `logp_t = z[t,y_t]/tau_t - logsumexp_v(z[t,v]/tau_t)`, defined while `abs(z/tau_t) < 3e38`. The backward recomputes each chunk's logits (four GEMMs per chunk) and returns BF16 `dX` and `dW`. A bad temperature tensor triggers a device-side assertion, unless `check_temperature=True`, which raises `ValueError` instead. |
| `cake_lm_head_det_train_preload` | `(device=None, *, temperature=False)` | A CUDA device, its index, or `None` | Does part of a process's one-time first-call work ahead of time. It loads the loss and logprob libraries (building them with JIT if none are prebuilt), creates the per-device side stream, count cells and constants, runs the valid-row count PyTorch ops once, and synchronizes. It launches no package kernels. The support-entry libraries still load at their first call. `temperature=True` also loads the temperature libraries. |
| `cake_lm_head_det_train_support_logprob` | `(X, W, support_ids, support_offsets, labels=None, *, k_cap, temperature=1.0, chunk_size=None, check_inputs=False, backend="cake")` | V <= 1572864. `support_ids` int32/int64 `[M]`, unique within a row. `support_offsets` int32/int64 `[T+1]`, starting at 0, non-decreasing, ending at M, with at most `k_cap` per row. A non-ignored label must be one of its row's candidates. `temperature` is a finite positive number or FP32 `[T]`, not differentiable. `chunk_size` defaults to 4096. | Autograd. `logp_j = z_j/tau_t - logsumexp` over row t's candidates, 0 on rows with <= 1 candidate. With `labels=None`, returns the packed FP32 `logp [M]` in caller order; otherwise returns the selected FP32 `logp [T]`. Bitwise reproducible. `check_inputs=False` queues an async device assertion; `True` reads back once and raises `ValueError`. |
| `cake_lm_head_det_train_support_kl` | `(X, W, support_ids, support_offsets, teacher_logprobs, *, k_cap, temperature=1.0, return_logprobs=False, chunk_size=None, check_inputs=False, backend="cake")` | `teacher_logprobs` FP32 `[M]` aligned with `support_ids`, `requires_grad=False`; `-inf` entries drop out | Returns FP32 `kl [T]` with `KL_t = sum_j Q_j (lq_j - logp_j)` and `Q_j = exp(lq_j)`. With `return_logprobs=True`, returns `(kl, logp)` with the differentiable packed student `logp [M]`. Weighting, reduction and any `tau**2` factor are the caller's job. |

The module exports exactly these five functions. The backend `cake_pplx._backends.cake_lm_head_det_train.cake_backend`, which is not exported by the facade, also provides:

- `set_gradient_placement`: a process setting, default `'auto'`.
- `memory_report`: peak temporary memory, reported separately from weights, outputs and accumulators.
- `BINDING_CACHE` (`.configure(...)`).
- `forward_loss` / `backward_loss`: the explicit pair. This is the documented way to get an FP32 `dW` for a BF16 `W`.
- `SUPPORTED_COMPUTE_CAPABILITIES`.

### `cake_pplx.cake_lm_head_nondet_train`

Same computation, arguments and precision boundaries as det, with these differences:

- A sliced `dX` GEMM stores K slices into FP32 slabs that are summed in slice order. That fixed order is a design note, not a run-to-run promise, and `dX` bits may differ from det's.
- Both packages leave zero-weight policy rows out of the gradient path and skip the GEMM work of zero-`dlogp` rows. Nondet adds no loss term for a zero-weight row (det's default `"exact"` loss mode keeps the loss bits of the unsplit call), and its zero-`dlogp` skip starts at 1946 valid rows on `sm_107a` (det: 4097), so loss and `dW` bits may differ from det. Both modules are judged against the BF16 naive tier.
- A one-row CE call at H=6144 can run one-pass programs.
- The module makes no run-to-run bit-equality promise for any output.
- Under deterministic algorithms, the backend raises PyTorch's alert before a K-sliced `dX` reduction (`RuntimeError`, or `UserWarning` with `warn_only`).
- There is no automatic memory form: `memory_mode=None` is the fastest form.

There is no KCP module. Source: [`kernels/lm_head/cake_pplx/cake_lm_head_nondet_train.py`](../kernels/lm_head/cake_pplx/cake_lm_head_nondet_train.py).

| Name | Signature | Inputs | Returns / notes |
|---|---|---|---|
| `cake_lm_head_nondet_train_loss` | Same as the det loss | As det. `dx_finalize` here only selects the one-pass scatter of the compacted `dX` output. | Returns the FP32 loss, plus `logp` with `return_logp=True`. |
| `cake_lm_head_nondet_train_logprob` | Same as the det logprob | As det. `None`, or a number whose FP32 value is 1.0, runs without temperature. A bool temperature raises `ValueError`. With `'keep_logits'`, a `hidden_count` left at `None` is passed down as `None` unless `CAKE_PPLX_LM_HEAD_LOSS_HIDDEN_COUNT` is `'0'`. | Returns FP32 `logp [T]`. |
| `cake_lm_head_nondet_train_preload` | `(device=None, *, temperature=False)` | As det | Per-thread state (side streams, events, count cells) belongs to the calling thread. |
| `cake_lm_head_nondet_train_support_logprob` | Same as det | As det | No run-to-run bit promise for any output. The backward alerts before a dense `dX` of more than one K slice. |
| `cake_lm_head_nondet_train_support_kl` | Same as det | As det | Gradients behave as in `support_logprob`. |

The module exports exactly these five functions. The nondet backend also provides `set_gradient_placement`, `set_one_pass_rows`, `set_kept_logits_form` (which decides `hidden_count` for `'keep_logits'` when it is left at `None`), `memory_report`, `BINDING_CACHE` and `SUPPORTED_COMPUTE_CAPABILITIES`.

**Options and environment.** The switch keywords on loss and logprob are `compact_rows`, `fuse_dw_cast`, `dx_finalize`, `hidden_count`, `dw_stream` (`True`, `False` or `'auto'`) and `gemm_tuning` (`{'logits': {...}, 'dx': {...}, 'dw': {...}}`). An explicit keyword overrides the environment variable.

| Variable | Effect |
|---|---|
| `CAKE_PPLX_LM_HEAD_LOSS_COMPACT_ROWS` | `'0'` turns valid-row compaction off |
| `CAKE_PPLX_LM_HEAD_LOSS_FUSE_DW_CAST` | `'0'` turns off the fused last-chunk weight-gradient cast |
| `CAKE_PPLX_LM_HEAD_LOSS_DX_FINALIZE` | `'0'` turns off the fused `dX` finalize |
| `CAKE_PPLX_LM_HEAD_LOSS_DW_STREAM` | `0/false/off/no` gives `False`, `1/true/on/yes` gives `True`, and anything else gives `'auto'`. The side stream runs the `dW` GEMM; `'auto'` uses it for calls with 3 or more chunks. |
| `CAKE_PPLX_LM_HEAD_LOSS_HIDDEN_COUNT` | `'0'` turns off the hidden device-side valid-row count |
| `CAKE_PPLX_LM_HEAD_LOSS_GEMM_TUNING` | A JSON object of per-GEMM knobs. Invalid JSON or a non-object raises `ValueError`. |
| `CAKE_PPLX_LM_HEAD_LOSS_BINDING_CACHE` | `'0'` disables both binding caches. Read at import and applied on the first call. |
| `CAKE_PPLX_LM_HEAD_LOSS_BINDING_CACHE_CAPACITY` | A positive int capacity (default 64). A bad value raises `ValueError` at import. |

`memory_mode` has no environment variable. `num_valid_rows` lets the caller supply the valid-row count and avoid a host wait. The knob parsing lives in the private module `cake_pplx._cake_lm_head_train_knobs`.

**Cross-family.** None.

**Example.**

```python
import torch
from cake_pplx.cake_lm_head_det_train import cake_lm_head_det_train_loss, cake_lm_head_det_train_logprob

T, H, V = 1024, 4096, 32768  # H and V multiples of 256
X = (0.5 * torch.randn(T, H, device="cuda")).to(torch.bfloat16).requires_grad_(True)
W = (0.02 * torch.randn(V, H, device="cuda")).to(torch.bfloat16).requires_grad_(True)
labels = torch.randint(0, V, (T,), device="cuda", dtype=torch.int64)
labels[:16] = -100  # ignored rows

loss, logp = cake_lm_head_det_train_loss(
    X, W, labels, objective="ce", loss_div=float((labels >= 0).sum()),
    chunk_size=4096, return_logp=True,
)
loss.backward()  # X.grad, W.grad: BF16

logp2 = cake_lm_head_det_train_logprob(X.detach().requires_grad_(True), W, labels)
logp2.sum().backward()
```

**Caveats.**

- Through the autograd entry, `grad_weight_dtype` must equal `W.dtype`. To get an FP32 `dW` for a BF16 `W`, use the backend's `forward_loss` / `backward_loss`.
- A wrong `num_valid_rows` value ends the CUDA context through a device-side assertion, unless `check_num_valid_rows=True`. An `sm_103a` call whose library was built with CUDA 13.0 keeps the host-count schedule and reads the count back once.
- An invalid temperature tensor or support layout also ends the CUDA context through a device assertion, unless `check_temperature` / `check_inputs` is `True`.
- `memory_mode='budget'` (or an automatic budget form) releases the saved accumulators after a graph-freeing backward, so a second backward raises.
- Nondet: the deterministic-algorithms alert is part of the contract. Its zero-weight loss sum and its `sm_107a` zero-`dlogp` floor differ from det's, so loss and `dW` bits can differ from det; both modules are judged against the BF16 naive tier.
- The binding-cache environment variables are read once at import, so a malformed capacity fails at import.
- Each backend package's README (`cake_pplx/_backends/<pkg>/README.md`) holds the memory-margin and toolchain notes that the docstrings refer to.

---

## act_qdq

**Computes.** A block-scaled quantize-dequantize (QDQ) of BF16 activations back to BF16, forward only. Each group of `G` consecutive elements of the last dimension gets one scale `s`, then `y = bf16(q * s)`, where `q` is the element format's round-to-nearest-even value of `clamp(x / s, -Q, Q)`, with the sign of zero kept. `amax` is the group's largest magnitude.

| `format` | Elements | `G` | Scale `s` | `amax` floor | `Q` |
|---|---|---|---|---|---|
| `"fp4_e4m3_g16"` | E2M1 | 16 | E4M3 of `max(amax, 6*2^-9) / 6`, round to nearest even, saturating at 448 | `6*2^-9` | 6 |
| `"mxfp4_e8m0_g32"` | E2M1 | 32 | `2^ceil(log2(max(amax, 6*2^-126) / 6))` | `6*2^-126` | 6 |
| `"mxfp8_e8m0_g32"` | E4M3 | 32 | `2^ceil(log2(max(amax, fp32(1e-4)) / 448))` | `fp32(1e-4)` | 448 |

- The outputs equal this rule evaluated in FP64, bit for bit, signed zeros included. The rule's quotient is a true division. With an E4M3 scale the kernels never take the plain product with a rounded reciprocal: the `sm_100a` and `sm_103a` programs correct it by one FMA residual step, and the `sm_107a` programs round it toward zero. The kernels use no FP64; the backend README gives the argument that their FP32 and BF16 steps give the rule's outputs.
- The optional second output is the FP32 scale of each group, `[..., D/G]`.
- One kernel per call reads `x` once and writes `y` (and the scales) once, on the current stream. The input may be any rank; the groups run along the flat buffer.

**Architectures.** `sm_100a`, `sm_103a`, `sm_107a`. Other devices raise `RuntimeError("activation QDQ requires SM100, SM103 or SM107")`. The `build.json` minimum CUDA toolkit is 12.9 for `sm_100a` and `sm_103a` (the kernels use the three-input `max` of PTX ISA 8.8) and 13.5 for `sm_107a`. Validated on B200 (`sm_100a`), GB300 (`sm_103a`) and VR200 (`sm_107a`).

### `cake_pplx.cake_act_qdq_det_train`

This family has a det module only, with no nondet or KCP module. Each group is quantized by its own threads, its maximum is exact in any order, and nothing is summed, so a nondet module would have no non-deterministic step; the op works on each token alone, so there is nothing to split across ranks. No call alerts under `torch.use_deterministic_algorithms(True)`. There is no autograd Function. Source: [`kernels/act_qdq/cake_pplx/cake_act_qdq_det_train.py`](../kernels/act_qdq/cake_pplx/cake_act_qdq_det_train.py).

| Name | Signature | Inputs | Returns / notes |
|---|---|---|---|
| `cake_act_qdq_det_train_forward` | `(x, *, format, out=None, scales=None, return_scales=False)` | `x` contiguous BF16 CUDA `[..., D]`, 16-byte aligned, `D` a multiple of `G`, at most `8*(2^31-256)` elements. `out`: `x` itself (in place) or a contiguous, 16-byte aligned BF16 tensor of `x`'s shape that does not overlap `x`. `scales`: a contiguous FP32 `[..., D/G]` buffer that overlaps neither. | Returns `y` (BF16, `x`'s shape; `out` when given, so `out=x` returns `x`), or `(y, scales)` when `return_scales=True` or `scales` is given. Not autograd. Any number of rows: a tensor with no elements returns empty outputs without a launch. A wrong dtype raises `TypeError`, any other bad argument `ValueError`, before any launch. |
| `cake_act_qdq_det_train_forward_workspace_bytes` | `(shape, *, format)` | The input shape and format | Returns 0: no workspace. A call keeps no state between calls either. Raises `ValueError` for a shape or format the forward refuses. |

Other public names:

- `ARCHES`, `SUPPORTED_COMPUTE_CAPABILITIES = ((10, 0), (10, 3), (10, 7))`.
- `FORMATS = ("fp4_e4m3_g16", "mxfp4_e8m0_g32", "mxfp8_e8m0_g32")`, `GROUP_SIZES` (format to `G`).
- `is_cake_act_qdq_det_train_supported(device)`.

**Options and environment.** `format`, `out`, `scales` and `return_scales` are the only keywords. A call runs one program of its format and scale output, chosen by `n_vecs = numel / 8` and by whether the call is in place: the small program (256-thread CTAs, tiles of 512 vectors, 8 resident CTAs per SM, 4 on `sm_107a`) out of place for 65,537 to 524,288 vectors on `sm_100a` and `sm_103a` (to 131,072 for MXFP8 without scales) and for 1 to 262,144 on `sm_107a`, and in place for 1 to 262,144 on `sm_107a` only; for MXFP8 without scales on `sm_100a` and `sm_103a`, the wide program (128-thread CTAs, tiles of 512 vectors) out of place from 131,073 vectors to 2,147,483,136 and in place from 262,145 to 33,554,431; for MXFP4 and MXFP8 on `sm_100a` and `sm_103a`, the large program (1024-thread CTAs, tiles of 2,048 vectors) in place from 4,194,304 vectors (from 33,554,432 for MXFP8 without scales) and out of place from 8,388,608 for MXFP8 with scales only, all to 2,147,481,600; else the main program (64-thread CTAs, tiles of 128 vectors, 256 on `sm_107a`, 32 resident CTAs per SM, 16 on `sm_107a`). Every program gives the same bytes. The grid is one CTA per tile once the tiles pass 2.5 waves of resident CTAs, else the resident CTAs, at most one per tile, walk the tiles; the wide and the large programs always launch one CTA per tile. Every library serves any shape or row count; a call builds or loads its program's library on first use, so before a CUDA graph capture build the architecture's libraries (`cake_jit.specs_for_arch`) or make one eager call per program the capture uses. The facade reads no environment variable.

**Cross-family.** None.

**Example.**

```python
import torch
from cake_pplx.cake_act_qdq_det_train import cake_act_qdq_det_train_forward

kv = torch.randn(16384, 512, device="cuda", dtype=torch.bfloat16)
q = torch.randn(16384, 32, 128, device="cuda", dtype=torch.bfloat16)
o = torch.randn(16384, 8, 4096, device="cuda", dtype=torch.bfloat16)

kv_qdq, kv_scales = cake_act_qdq_det_train_forward(kv, format="fp4_e4m3_g16", return_scales=True)
cake_act_qdq_det_train_forward(q, format="mxfp4_e8m0_g32", out=q)  # in place
o_qdq = cake_act_qdq_det_train_forward(o, format="mxfp8_e8m0_g32")
```

**Caveats.**

- Forward only: there is no backward and no autograd entry. For training through the rounding, wrap the call in your own `torch.autograd.Function` (for example a straight-through estimator). Outputs carry no autograd history; a buffer the call writes (`out`, `x` in place, `scales`) must not require grad while grad mode is on, and the call advances its version counter as a PyTorch in-place op does.
- Non-finite inputs (NaN, ±Inf) are outside the rule; their outputs are unspecified.
- Finite inputs can give `±inf`: with MXFP8 an element with `|x| >= 248*2^120`, with MXFP4 one with `|x| >= 224*2^120` (the rule's own BF16 rounding of `q * s`).
- Only the dequantized BF16 values and the FP32 scales are returned, not the packed FP4/FP8 codes.
