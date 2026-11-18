# cake-pplx-kda

Cake `kda` training kernels. Public modules:

- `cake_pplx.cake_kda_det_train`
- `cake_pplx.cake_kda_det_train_kcp`
- `cake_pplx.cake_kda_nondet_train`
- `cake_pplx.cake_kda_nondet_train_kcp`

Install this family alone, pinned to a commit:

```bash
pip install "cake-pplx-kda @ git+https://github.com/perplexityai/cake-pplx-collab@<sha>#subdirectory=kernels/kda"
```

The version is in [`VERSION`](VERSION). Since 0.2.0 every `sm_107a` library needs CUDA 13.5 or newer.

The tests judge every floating output of the four packages by its error against an FP64 reference, beside the
error of a naive BF16 PyTorch implementation on the same inputs, over many random draws (`tests/cake_naive.py`).
They set no tolerance, and they never compare `det` with `nondet`. A `det` package is also checked for bitwise
identical results from run to run.

The benchmarks `benchmarks/bench_cake_kda_baselines.py` (one GPU) and `benchmarks/bench_cake_kda_kcp_baselines.py`
(KCP, `torchrun`) time both packages against the cuDNN frontend's KDA (plan `kda_frost`, `batch_invariant=True` for
`det` and `False` for `nondet`) and, on one GPU, against flash-linear-attention's `chunk_kda`.

The repository rules are in [`ARCHITECTURE.md`](../../ARCHITECTURE.md).
