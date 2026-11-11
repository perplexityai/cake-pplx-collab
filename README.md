# cake-pplx-collab

This repository holds Cake training kernels for NVIDIA Blackwell GPUs (SM100, SM103 and SM107). Most families provide a forward, a backward and an autograd entry point. The kernels moved here from FlashInfer. [flashinfer-ai/flashinfer#9326](https://github.com/flashinfer-ai/flashinfer/pull/9326), a draft pull request, removes them from FlashInfer. Their names follow [flashinfer-ai/flashinfer#8928](https://github.com/flashinfer-ai/flashinfer/issues/8928). The source commit is [flashinfer-ai/flashinfer@c5c39b0e87f027011c86881aa2935a83e1ce53e7](https://github.com/flashinfer-ai/flashinfer/commit/c5c39b0e87f027011c86881aa2935a83e1ce53e7). There are 14 kernel families, and each one is a separate pip distribution. You can pin, upgrade or roll back one family without changing the others. FlashInfer is not required.

## Kernel families

All modules live under `cake_pplx.`. `{det,nondet}` means that both a deterministic module and a non-deterministic module exist.

| Family | Distribution | Public modules | Computes | API |
|---|---|---|---|---|
| rmsnorm | `cake-pplx-rmsnorm` | `cake_rmsnorm_{det,nondet}_train` | Fused BF16 RMSNorm, optional fused residual add | [API](docs/API.md#rmsnorm) |
| rmsnorm_silu_gate | `cake-pplx-rmsnorm-silu-gate` | `cake_rmsnorm_silu_gate_{det,nondet}_train` | RMSNorm(x) * weight * SiLU(z) over a last dim of 128 | [API](docs/API.md#rmsnorm_silu_gate) |
| causal_conv1d | `cake-pplx-causal-conv1d` | `cake_causal_conv1d_{det,nondet}_train[_kcp]` | Packed width-4 depthwise causal conv + SiLU | [API](docs/API.md#causal_conv1d) |
| kda | `cake-pplx-kda` | `cake_kda_{det,nondet}_train[_kcp]` | Kimi Delta Attention chunked training | [API](docs/API.md#kda) |
| gdn | `cake-pplx-gdn` | `cake_gdn_{det,nondet}_train[_kcp]` | Gated DeltaNet chunked gated delta rule | [API](docs/API.md#gdn) |
| mhc | `cake-pplx-mhc` | `cake_mhc_{det,nondet}_train` | Four-stream mHC hyper-connection layer | [API](docs/API.md#mhc) |
| mhc_head | `cake-pplx-mhc-head` | `cake_mhc_head_{det,nondet}_train` | Stream mean at the mHC model head | [API](docs/API.md#mhc_head) |
| gqa | `cake-pplx-gqa` | `cake_gqa_{det,nondet}_train` | Packed-varlen causal GQA attention, head dim 256 | [API](docs/API.md#gqa) |
| dsa_indexer | `cake-pplx-dsa-indexer` | `cake_dsa_indexer_{det,nondet}_train` | DSA indexer scores + exact top-k (forward only) | [API](docs/API.md#dsa_indexer) |
| dsa_sparse_attention | `cake-pplx-dsa-sparse-attention` | `cake_dsa_sparse_attention_{det,nondet}_train` | Top-k sparse MLA attention, 64 heads | [API](docs/API.md#dsa_sparse_attention) |
| dense_gemm | `cake-pplx-dense-gemm` | `cake_dense_gemm_{det,nondet}_train` | BF16 projection GEMM, FP32 router GEMM | [API](docs/API.md#dense_gemm) |
| moe_grouped_gemm | `cake-pplx-moe-grouped-gemm` | `cake_moe_grouped_gemm_{det,nondet}_train` | Ragged BF16 MoE grouped GEMM | [API](docs/API.md#moe_grouped_gemm) |
| gated_act_mxfp8 | `cake-pplx-gated-act-mxfp8` | `cake_gated_act_mxfp8_det_train` | Fused SwiGLU + MXFP8 quantization | [API](docs/API.md#gated_act_mxfp8) |
| lm_head | `cake-pplx-lm-head` | `cake_lm_head_{det,nondet}_train` | Chunked LM-head loss, log-probs and support KL | [API](docs/API.md#lm_head) |

lm_head is still in review and is not on `main` yet. Its install line works after that pull request merges.

Each family's source is in `kernels/<family>/`. Distribution names use hyphens, and the subdirectory names use underscores. pip treats `-` and `_` in distribution names as the same character.

## Install one family

To install one family pinned to a commit:

```bash
pip install "cake-pplx-kda @ git+https://github.com/perplexityai/cake-pplx-collab@<sha>#subdirectory=kernels/kda"
```

In `requirements.txt`, give each family its own line. In this example the two families come from different commits:

```text
cake-pplx-kda @ git+https://github.com/perplexityai/cake-pplx-collab@<sha-a>#subdirectory=kernels/kda
cake-pplx-rmsnorm @ git+https://github.com/perplexityai/cake-pplx-collab@<sha-b>#subdirectory=kernels/rmsnorm
```

To upgrade or downgrade one family, change the SHA on its line. The other families stay as they are.

All families install into one namespace package, `cake_pplx` (PEP 420). Neither `cake_pplx/` nor `cake_pplx/_backends/` contains an `__init__.py`, so each distribution only adds its own modules. Several families can be installed side by side, each from a different commit.

Every family is at `VERSION` 0.1.0. Per-family release tags of the form `<family>/vX.Y.Z` are planned but do not exist yet. Until they do, pin by commit SHA.

## Use

```python
import torch
from cake_pplx.cake_rmsnorm_det_train import cake_rmsnorm_det

x = torch.randn(129, 4096, device="cuda", dtype=torch.bfloat16, requires_grad=True)
w = torch.randn(4096, device="cuda", dtype=torch.bfloat16, requires_grad=True)
y = cake_rmsnorm_det(x, w, 1e-6)  # BF16 [129, 4096]
y.backward(torch.randn_like(y))   # x.grad BF16, w.grad in w.dtype
```

The first call compiles the kernels (see [JIT build and cache](#jit-build-and-cache)). Each family section in [docs/API.md](docs/API.md) includes a fuller example.

### Migrating from FlashInfer

Only the import prefix changes:

| FlashInfer | cake-pplx-collab |
|---|---|
| `flashinfer.<module>` | `cake_pplx.<module>` |
| `flashinfer.experimental.cake.<module>` | `cake_pplx._backends.<module>` |

For example, `from flashinfer.cake_kda_det_train import cake_kda_det` becomes `from cake_pplx.cake_kda_det_train import cake_kda_det`. The `cake_pplx` namespace does not re-export anything, so import each module by its full name. Names that FlashInfer had already removed have no alias here. The caveats for each family in docs/API.md list them.

## Requirements

| Item | Requirement |
|---|---|
| GPU | Compute capability 10.0, 10.3 or 10.7 (`sm_100a`, `sm_103a`, `sm_107a`). Some families limit shapes per architecture; see docs/API.md. |
| CUDA toolkit | `nvcc` must be available at run time, because kernels compile on first use. Each library in `build.json` sets its own minimum: 12.8 for `sm_100a` (12.9 for lm_head), 12.9 for `sm_103a` and 13.4 for `sm_107a`. Some moe_grouped_gemm libraries need 13.4 on every architecture. An older toolkit raises `RuntimeError` when that library is built. |
| `torch` | Required by every family. |
| `apache-tvm-ffi` | `>=0.1.11,<0.2`, required by every family. |
| `ninja` | moe_grouped_gemm only. |
| `cuda-python` | causal_conv1d, dense_gemm and rmsnorm_silu_gate only. |

## JIT build and cache

Kernels are compiled just in time (JIT) with `nvcc` on first use. The build reads each backend's `build.json` and compiles for exactly the arch flags in that manifest. Compiled libraries are cached under a key that is a digest of the sources, the flags and the CUDA version, so switching commits never reuses a stale library.

| Variable | Effect |
|---|---|
| `CAKE_PPLX_CACHE_DIR` | Library cache directory. Default: `~/.cache/cake_pplx`. |
| `CAKE_PPLX_NVCC` | Path of the `nvcc` that compiles. Default: `<toolkit root>/bin/nvcc`, with the root from the next row. The CUDA version check and the cache key still use `<toolkit root>/bin/nvcc`, so point both at the same toolkit. |
| `CUDA_HOME` / `CUDA_PATH` | CUDA toolkit root. Otherwise `nvcc` on `PATH`, or `/usr/local/cuda`. |
| `CXX` | Host compiler. Default: `c++`. |
| `CAKE_PPLX_JIT_WORKERS` | Number of parallel compile jobs. Default: `min(8, cpu_count)`. |
| `CAKE_PPLX_DISABLE_JIT` or `FLASHINFER_DISABLE_JIT` | Disables JIT. Only prebuilt `aot/` libraries load, and a missing library raises `RuntimeError`. |

Any non-empty value of `CAKE_PPLX_DISABLE_JIT` or `FLASHINFER_DISABLE_JIT`, including `0`, disables the JIT. Exception: moe_grouped_gemm also builds a small queue-split probe with `tvm_ffi.cpp.load_inline` (needs `ninja` and `nvcc`), outside `build.json` and these variables. If that build fails, the kernel uses the even split and the output bits do not change.

Environment variables named `FLASHINFER_*` that kernels read (for example `FLASHINFER_CAKE_DSA_*` and `FLASHINFER_CAKE_LM_HEAD_LOSS_*`) keep their names for now. Before you capture a CUDA graph, run one eager call so that no JIT build happens during capture.

## Cross-family combinations

The KCP (context-parallel) route of causal_conv1d uses the KCP context object that kda or gdn creates. The causal_conv1d distribution does not declare kda or gdn as a dependency, so you need to install both families yourself:

- Pin causal_conv1d and kda (or gdn) to compatible commits. The tested combination is listed in the pull requests.
- A protocol version check (`KCP_CONTEXT_PROTOCOL`) is planned but not implemented yet. Today the contract is duck-typed.
- Cross-family tests are in `tests/cross_family/<family>/`. That directory has the same layout as `kernels/<family>/` (`tests/` and `benchmarks/`).

## Status

| Item | State |
|---|---|
| Validated | NVIDIA GB300 (`sm_103a`), driver 580.159.03, CUDA 13.4.92, torch 2.14.1+cu130, apache-tvm-ffi 0.1.15 |
| Supported by the code, not yet validated here | `sm_100a`, `sm_107a` |
| Not run | Tests that need more than 4 GPUs |
| Release tags | Planned, not created yet |

## Development

[ARCHITECTURE.md](ARCHITECTURE.md) describes the repository layout, the isolation rules, the upstream sync and the release procedure.

To run a family's tests, first install the family with its `test` extra. Then copy its `tests`, `benchmarks` and `pyproject.toml` to a directory outside the source tree, and run pytest there:

```bash
# from the repository root
pip install "./kernels/kda[test]"
mkdir -p /tmp/kda-tests
cp -r kernels/kda/tests kernels/kda/benchmarks kernels/kda/pyproject.toml /tmp/kda-tests/
cd /tmp/kda-tests && python -m pytest -p no:cacheprovider tests
```

The "Running tests" section of ARCHITECTURE.md has the details. CI runs the whole suite of each changed family on a CPU runner, where GPU tests are skipped.
