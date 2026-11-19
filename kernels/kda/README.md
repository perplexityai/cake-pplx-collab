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

The tests judge every floating output of the four packages by its error against an FP64 reference over 32 random
draws (`tests/cake_naive.py`). Two references run on the same inputs: a naive BF16 PyTorch implementation, and
flash-linear-attention 0.6.0 `chunk_kda` (FLA) for the layer's output and gradients. This is the KDA rule: an output
passes when it meets either reference, and fails only when it misses both. To meet a reference, its median and
largest L2 error over the draws are at most 1.005 times the reference's, its max-norm error is within 1.005 times
the reference's on at least half of the draws, and its error is 0 on every draw where the reference's is 0. An
output with no FLA counterpart, or a case that FLA cannot run (for example FLA is not installed: install the `fla`
extra), is judged against the naive implementation only. Each `TIER` line of the test summary names the reference
met and shows the errors against both. The tests set no tolerance, and they never compare `det` with `nondet`. A
`det` package is also checked for bitwise identical results from run to run.

The benchmarks `benchmarks/bench_cake_kda_baselines.py` (one GPU) and `benchmarks/bench_cake_kda_kcp_baselines.py`
(KCP, `torchrun`) time both packages against the cuDNN frontend's KDA (plan `kda_frost`, `batch_invariant=True` for
`det` and `False` for `nondet`) and, on one GPU, against flash-linear-attention's `chunk_kda`.

The repository rules are in [`ARCHITECTURE.md`](../../ARCHITECTURE.md).
