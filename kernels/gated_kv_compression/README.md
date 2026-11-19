# cake-pplx-gated-kv-compression

Cake `gated_kv_compression` training kernels: ratio-2 gated KV compression, width 512. Each pair of adjacent tokens
of a packed document becomes one row, mixed per channel by a softmax over the two gate logits. Forward, backward and
autograd entries on `sm_100a`, `sm_103a` and `sm_107a`. Public module:

- `cake_pplx.cake_gated_kv_compression_det_train`

The family has a deterministic module only: every output depends on its own token pair, so no non-deterministic step
exists. It has no KCP module: the kernel is single-rank.

Install this family alone, pinned to a commit:

```bash
pip install "cake-pplx-gated-kv-compression @ git+https://github.com/perplexityai/cake-pplx-collab@<sha>#subdirectory=kernels/gated_kv_compression"
```

The repository rules are in [`ARCHITECTURE.md`](../../ARCHITECTURE.md).
