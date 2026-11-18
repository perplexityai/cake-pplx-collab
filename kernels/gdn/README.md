# cake-pplx-gdn

Cake `gdn` training kernels. Public modules:

- `cake_pplx.cake_gdn_det_train`
- `cake_pplx.cake_gdn_det_train_kcp`
- `cake_pplx.cake_gdn_nondet_train`
- `cake_pplx.cake_gdn_nondet_train_kcp`

Install this family alone, pinned to a commit:

```bash
pip install "cake-pplx-gdn @ git+https://github.com/perplexityai/cake-pplx-collab@<sha>#subdirectory=kernels/gdn"
```

The version is in [`VERSION`](VERSION). Since 0.2.0 every `sm_107a` library needs CUDA 13.5 or newer.

The repository rules are in [`ARCHITECTURE.md`](../../ARCHITECTURE.md).
