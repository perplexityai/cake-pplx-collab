# cake-pplx-route-weight-scale

Cake `route_weight_scale` training kernels: route-weight row scaling of a routed MoE expert. Each row of the down
projection's BF16 output is multiplied by its FP32 route weight before the combine, forward and backward, at hidden
width 4096 or 3584, on `sm_100a`, `sm_103a` and `sm_107a`. Public module:

- `cake_pplx.cake_route_weight_scale_det_train`

The family has a deterministic module only (no `nondet` module and no KCP module): nothing sums across rows, and the
one sum, `dw` within a row, has one fixed order.

| Item | Contract |
|---|---|
| Entries | `cake_route_weight_scale_det(x, w) -> y` (autograd); `cake_route_weight_scale_det_train_forward(x, w, *, out=None) -> y`; `cake_route_weight_scale_det_train_backward(x, dy, w, *, dx=None, dw=None) -> (dx, dw)` |
| Forward | `y = BF16(w * x)`: the FP32 product, rounded once. Bit for bit `(w[:, None] * x.float()).bfloat16()`. |
| Backward | `dx = BF16(w * dy)`, bit for bit `(w[:, None] * dy.float()).bfloat16()`; `dw = sum over c of dy * x`, one FP32 word per row, no division by `w` |
| `x`, `dy`, `y`, `dx` | BF16 `[rows, H]`, `H` 4096 or 3584. Unit column stride, a row stride that is a multiple of 8 elements, a 16-byte aligned base: row slices of wider or longer buffers are read and written in place. |
| `w`, `dw` | FP32 `[rows]`, contiguous |
| `rows` | 0 to 1,048,576 and beyond (fewer than `2**29`). Row offsets are 64-bit, so one call may span more than 2^31 elements. `rows = 0` launches nothing. |
| Workspace | 0 bytes. The forward allocates only `y`, the backward only `dx` and `dw` (none when passed in). No `[rows, H]` FP32 buffer. |
| Retained state | The autograd entry saves `x` and `w` only (references, no copy). |
| Shapes | `rows` and the strides are run-time values: one library per program and architecture, no recompilation. |
| Host | No synchronization, no device-to-host read; runs on the current stream of the operands' device |
| Checkpointing | Works under `torch.utils.checkpoint(..., use_reentrant=False)`; the recomputation launches the forward once. |
| CUDA graphs | Run one eager call of each stage, width and row count the graph captures first, so that no JIT build happens during capture. The forward and the backward at both widths pick a program and a grid by the row count, with the same bits. |
| Architectures | `sm_100a`, `sm_103a`, `sm_107a`; the JIT build needs the CUDA toolkit that `build.json` names for each |
| Determinism | Bitwise reproducible run to run. No alert under `torch.use_deterministic_algorithms(True)`. |
| Modes | `det` only: there is no step that sums across rows, so `nondet` and KCP are not applicable. |

Install this family alone, pinned to a commit:

```bash
pip install "cake-pplx-route-weight-scale @ git+https://github.com/perplexityai/cake-pplx-collab@<sha>#subdirectory=kernels/route_weight_scale"
```

The repository rules are in [`ARCHITECTURE.md`](../../ARCHITECTURE.md).
