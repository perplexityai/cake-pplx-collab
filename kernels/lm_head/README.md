# cake-pplx-lm-head

Cake `lm_head` training kernels. Public modules:

- `cake_pplx.cake_lm_head_det_train`
- `cake_pplx.cake_lm_head_nondet_train`

Install this family alone, pinned to a commit:

```bash
pip install "cake-pplx-lm-head @ git+https://github.com/perplexityai/cake-pplx-collab@<sha>#subdirectory=kernels/lm_head"
```

The version is in [`VERSION`](VERSION). Since 0.3.0 `dX` folds each row's FP32 label residual (what the BF16 `dlogits` store drops from the label column's FP32 gradient), so its bits differ from 0.2.1's in both modules; the deterministic module's other outputs keep their 0.2.1 bits on the shapes checked (its backend README, "FP32 label residual").

The repository rules are in [`ARCHITECTURE.md`](../../ARCHITECTURE.md).
