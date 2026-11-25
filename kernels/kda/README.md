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

The version is in [`VERSION`](VERSION). Since 0.2.0 every `sm_107a` library needs CUDA 13.5 or newer. Since 0.2.2 the
single-rank packages also read the pack's token count when they choose the segments per document of the segmented
route, on the architectures and SM counts that `cake_jit.TOKEN_SEGMENTS` lists (the token-aware segment rule; see
[Kernel structure of one training step](cake_pplx/_backends/cake_kda_det_train/README.md#kernel-structure-of-one-training-step)).
The context-parallel packages keep the occupancy rule. The token-aware rule changes the segment plan of some
single-rank packs of 4, 8 and 16 heads, so on those packs the outputs of the deterministic single-rank package can
differ in their bits from those of 0.2.1; within one version it gives the same bits from run to run. Since 0.2.2 the
four packages also take 65 to 128 heads (before: 1..64, a limit the KCP entries enforced). That change alone changes no
program, plan or record of a call of at most 64 heads: at a given version, such a call runs the same plan and programs
with or without it. Above 64 heads the int32 offsets bound the tokens: see [docs/API.md](../../docs/API.md#kda).

The tests judge every floating output of the four packages by its error against an FP64 reference over 32 random
draws (`tests/cake_naive.py`). Two references run on the same inputs: a naive BF16 PyTorch implementation, and
flash-linear-attention 0.6.0 `chunk_kda` (FLA) for the layer's output and gradients. This is the KDA rule: an output
passes when it meets either reference, and fails only when it misses both. To meet a reference, its median and
largest L2 error over the draws are at most 1.005 times the reference's, its max-norm error is within 1.005 times
the reference's on at least half of the draws, and its error is 0 on every draw where the reference's is 0. An
output with no FLA counterpart, or a case that FLA does not run (operands that are not on a GPU or not `[S, H, D]`,
an activated gate outside FLA's safe range), is judged against the naive implementation only. The owner reviewed the
misses of both references measured on B200, GB300 and VR200 with the pinned FLA reference below and accepted a listed
set of them, each capped at its measured values (`tests/cake_naive_accepted.json`): a listed output of its listed test
passes only when FLA was judged and each part of the rule against its listed reference is met or within its cap, and its
`TIER` line reads `accepted`; a value beyond a cap fails, and any other miss of both references fails. No entry is
marked `temporary`. Each `TIER` line of the test
summary names the reference met and shows the errors against both. The tests set no tolerance, and they never compare
`det` with `nondet`. A `det` package is also checked for bitwise identical results from run to run.

FLA runs with Triton 3.8.0, and each autotuned FLA kernel runs the first config FLA lists for it, not a config that
Triton's autotuner picks by timing (`tests/experimental/test_helpers/cake_kda_fla.py`, `FLA_PINS`). A GPU test fails
when FLA runs its case and this reference gives no values: FLA 0.6.0 is not installed (install the `fla` extra), the
installed Triton is not 3.8.0, a first config differs from its pin, a variable that changes FLA's or Triton's code or
numbers is set (`FORBIDDEN_ENV`), or the FLA call raises an error. Triton comes with torch, and each torch release
requires one Triton minor release (torch 2.14.1 requires `triton~=3.8.0`). The `fla` extra therefore does not pin
Triton, as the package does not pin torch. When the installed Triton is not 3.8.0, these GPU tests fail. With a torch
that requires `triton~=3.8.0` (torch 2.14.1), which also admits a later 3.8.x release, install `triton==3.8.0` alongside
it; with a torch that requires another Triton minor release, install a torch that requires `triton~=3.8.0`. The
`FLA reference` line of the test summary records the versions of FLA, Triton and torch, a digest of FLA's sources, the
pin, the GPU and its shared memory, the `TRITON_*`, `FLA_*` and `PTXAS_*` variables that are set, the variables in the
key of Triton's compile cache that are set, and the ptxas that Triton runs (its path, sha256 and version).

The benchmarks `benchmarks/bench_cake_kda_baselines.py` (one GPU) and `benchmarks/bench_cake_kda_kcp_baselines.py`
(KCP, `torchrun`) time both packages against the cuDNN frontend's KDA (plan `kda_frost`, `batch_invariant=True` for
`det` and `False` for `nondet`) and, on one GPU, against flash-linear-attention's `chunk_kda`.

For the memory and time trade-off between `save_mode="light"` (the default) and `"tape"` in the single-rank
packages, see [Choosing `save_mode`](cake_pplx/_backends/cake_kda_det_train/README.md#choosing-save_mode); the KCP
packages support `"light"` only.

The repository rules are in [`ARCHITECTURE.md`](../../ARCHITECTURE.md).
