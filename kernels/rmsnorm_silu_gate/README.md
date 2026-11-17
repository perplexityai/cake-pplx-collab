# cake-pplx-rmsnorm-silu-gate

Cake `rmsnorm_silu_gate` training kernels. Public modules:

- `cake_pplx.cake_rmsnorm_silu_gate_det_train`
- `cake_pplx.cake_rmsnorm_silu_gate_nondet_train`

Install this family alone, pinned to a commit:

```bash
pip install "cake-pplx-rmsnorm-silu-gate @ git+https://github.com/perplexityai/cake-pplx-collab@<sha>#subdirectory=kernels/rmsnorm_silu_gate"
```

Requirements: CUDA 12.8 or newer for `sm_100a` (12.9 for some of
its libraries), 12.9 for `sm_103a` and 13.5 for `sm_107a`: the
`min_cuda` of each library in the backends' `build.json`.  On an older toolkit
`cake_build.jit_spec` raises and `cake_build.jit_specs` leaves the library out.

The repository rules are in [`ARCHITECTURE.md`](../../ARCHITECTURE.md).
