# Kernelized Path Gradient

> Reference upstream repository: https://github.com/longinyu/ksivi

This is the official repository for *Semi-Implicit Variational Inference via Kernelized Path Gradient Descent* (Pielok et al., 2026).

This repository provides the KPG files that should be overlaid onto
an official KSIVI ([Cheng et al., 2024](https://proceedings.mlr.press/v235/cheng24l.html)) checkout.

Simulation scripts in this package are derived from KSIVI experiment scripts, with minimal edits for KPG integration.

## How to use

1. Clone the official KSIVI repository:

```bash
git clone https://github.com/longinyu/ksivi
```

2. Overwrite your local KSIVI checkout using the helper script from this repository:

```bash
python scripts/overwrite_ksivi.py --ksivi-dir /path/to/ksivi
```

The script copies all files from this overlay package (except `README.md` and `scripts/*`)
into the KSIVI checkout, preserving relative paths.

Alternative: you can copy the files manually if you prefer not to use the script.

## Important note

- Users must obtain KSIVI directly from its official repository.
- This overlay package is provided for reproducibility.

## References

- Pielok, T., Bischl, B., and Rügamer, D. (2026). *Semi-Implicit Variational Inference via Kernelized Path Gradient Descent*. AISTATS 2026.
- Cheng, L., Lin, Y., Mroueh, Y., and Bühlmann, P. (2024). Kernel Semi-Implicit Variational Inference. *Proceedings of the 41st International Conference on Machine Learning (ICML 2024)*. https://proceedings.mlr.press/v235/cheng24l.html
