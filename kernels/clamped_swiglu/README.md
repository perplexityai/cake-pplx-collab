# cake-pplx-clamped-swiglu

Cake `clamped_swiglu` training kernels: the clamped SwiGLU of a routed MoE expert, forward and backward, with the FP32
route weight applied before the one BF16 rounding (intermediate width 2304), on `sm_100a`, `sm_103a` and `sm_107a`.
Public module:

- `cake_pplx.cake_clamped_swiglu_det_train`

The family has a deterministic module only (no `nondet` module): every output is elementwise or a sum within one row.

On an R200 (`sm_107a`), `locality="domain"` (opt-in; `"off"` is the default) runs a locality-domain twin of the
forward, whose CTAs work first on their own die's rows, for a call whose first input lies in memory from
`cake_clamped_swiglu_det_train_domain_empty` and whose row count reaches the twin's first rows per SM for the call's
class; every other call runs the default programs, and the bits are the same.  The backward has no twin: with
`"domain"` it runs its default programs.
The helper needs cuda-python: install the `locality` extra (`cake-pplx-clamped-swiglu[locality]`).  See
[docs/API.md](../../docs/API.md#clamped_swiglu).  On an R200 the locality tests need the extra too: install
`"./kernels/clamped_swiglu[test,locality]"`; without cuda-python the SM107 tests of
`tests/experimental/test_cake_clamped_swiglu_det_train_locality.py` fail.

The kernels compile on first use: a JIT build needs a CUDA toolkit of at least 12.8 for `sm_100a`, 12.9 for `sm_103a`
and 13.5 for `sm_107a`.

Install this family alone, pinned to a commit:

```bash
pip install "cake-pplx-clamped-swiglu @ git+https://github.com/perplexityai/cake-pplx-collab@<sha>#subdirectory=kernels/clamped_swiglu"
```

The repository rules are in [`ARCHITECTURE.md`](../../ARCHITECTURE.md).
