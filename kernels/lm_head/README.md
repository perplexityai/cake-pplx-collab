# cake-pplx-lm-head

Cake `lm_head` training kernels. Public modules:

- `cake_pplx.cake_lm_head_det_train`
- `cake_pplx.cake_lm_head_nondet_train`

Install this family alone, pinned to a commit:

```bash
pip install "cake-pplx-lm-head @ git+https://github.com/perplexityai/cake-pplx-collab@<sha>#subdirectory=kernels/lm_head"
```

The log-probability entries of both modules take an opt-in `memory_mode="keep_logits"` (see each backend README): the forward keeps its BF16 logits for the backward, which then runs three GEMMs instead of four, with the same output bits. The default is unchanged. Measured at `T` = 4097 (3892 valid rows, chunk 4096; `H` 6144 / `V` 154880 and `H` 5120 / `V` 129280) against the default call: a forward + backward step is 21.1-24.0 % faster on B200, 19.7-22.3 % on GB300 and 19.2-25.5 % on R200 for the non-deterministic module (the deterministic module's default call already keeps the logits there at this shape); the forward alone is 1.3-1.9 % slower on B200, -1.8 to +2.6 % on GB300 and 2.4-5.2 % slower on R200. The kept logits hold `2 * T_v * V` bytes from the forward to the backward (1.206 GB at `V` 154880, 1.007 GB at `V` 129280), the kept backward has no zero-gradient row skip, and a checkpointed call is not faster (it runs the kept forward twice).

The version is in [`VERSION`](VERSION). Since 0.3.0 the `dX` of the loss and log-probability entries folds each row's FP32 label residual (what the BF16 `dlogits` store drops from the label column's FP32 gradient), so its bits differ from 0.2.1's in both modules; the candidate-support entries fold nothing. The deterministic module's other outputs keep their 0.2.1 bits on the shapes checked (its backend README, "FP32 label residual").

The repository rules are in [`ARCHITECTURE.md`](../../ARCHITECTURE.md).
