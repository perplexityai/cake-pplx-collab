# cake-pplx-collab

Cake training kernels for Perplexity, one independent package per kernel family.

Each family lives in `kernels/<family>/` and installs alone, so you can pin every family to its own commit:

```text
cake-pplx-kda @ git+https://github.com/perplexityai/cake-pplx-collab@<sha>#subdirectory=kernels/kda
cake-pplx-rmsnorm @ git+https://github.com/perplexityai/cake-pplx-collab@<other-sha>#subdirectory=kernels/rmsnorm
```

Imports keep the FlashInfer names with a new prefix:

```python
from cake_pplx.cake_kda_nondet_train import cake_kda_nondet  # was flashinfer.cake_kda_nondet_train
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the layout, the isolation rules, and how to release one family.
