# cake-pplx-engram-lookup

Cake `engram_lookup` training kernel: the Engram row lookup. It gathers rows of a quantized table (FP8 E4M3 values,
one E8M0 scale per 32 columns) by row id and writes them as BF16, on `sm_100a`, `sm_103a` and `sm_107a`. Public
module:

- `cake_pplx.cake_engram_lookup_det_train`

The family has a deterministic module only and a forward step only (no backward, no autograd entry).

| Item | Contract |
|---|---|
| Entry | `cake_engram_lookup_det_train_forward(weight, scale, row_ids) -> out` |
| `weight` | `float8_e4m3fn [R, 256]`: unit column stride, row stride a multiple of 8 elements, 8-byte aligned start |
| `scale` | `float8_e8m0fnu [R, 8]`, one scale per 32 columns: same stride and alignment rules |
| Layouts | Two contiguous tensors, or two views into one packed `uint8 [R, 264]` buffer (row stride 264; `cake_engram_lookup_det_train_split_packed(buf)` makes the views). Both give the same bytes. |
| `row_ids` | Contiguous `int64` of any shape, `[T, 24]` and `[M]` included; repeats allowed; values in `[0, R)` |
| `out` | New contiguous BF16 `[*row_ids.shape, 256]` |
| Computes | `out[p, 32*g + j] = BF16(float32(weight[r, 32*g + j]) * float32(scale[r, g]))` with `r = row_ids[p]`: the exact product rounded once to BF16, FP32 subnormals kept, `±inf` past the FP32 range, NaN for an E8M0 byte `0xFF` (its 32 columns only) |
| Workspace | 0 bytes. Each call allocates only `out` (`P * 512` bytes for `P = row_ids.numel()` looked-up rows). |
| Retained state | None |
| Empty inputs | `T = 0` or `M = 0` returns an empty `out` and launches nothing |
| Shapes | `R`, `T` and `M` are run-time values: no recompilation. Two programs per architecture: a call of at most `SMALL_MAX_POSITIONS` looked-up rows (the backend's `cake_jit.py`) launches the small one, a larger call the main one; both write the same bytes. |
| Host | No synchronization, no device-to-host read; runs on the current stream of the operands' device |
| Architectures | `sm_100a` (CUDA 12.8 or newer), `sm_103a` (12.9), `sm_107a` (13.5) |
| Determinism | Bitwise reproducible run to run: every output element is one product rounded once, and nothing is summed. No alert under `torch.use_deterministic_algorithms(True)`. |
| Modes | `det` only. There is no `nondet` module: a gather has no sums, so det is the only mode. |
| KCP | No KCP form: the lookup is a single-rank local lookup. |
| Row ids | In `[0, R)`. An id outside (a negative id included) stops the launch with a device-side trap before its row is read: the call or the next synchronizing CUDA call raises, and the CUDA context is unusable afterwards. The host checks no id (that would synchronize). |

Install this family alone, pinned to a commit:

```bash
pip install "cake-pplx-engram-lookup @ git+https://github.com/perplexityai/cake-pplx-collab@<sha>#subdirectory=kernels/engram_lookup"
```

The repository rules are in [`ARCHITECTURE.md`](../../ARCHITECTURE.md).
