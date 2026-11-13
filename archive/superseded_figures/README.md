# Superseded figures

Hand-exported from the original `GraphsFinal.ipynb` before the audit. They are kept only
so the earlier version of the paper still compiles if anyone needs to diff against it.

**Do not use them.** They plot quality against each codec's own quality setting rather
than against bits per pixel, and they were produced from the pre-correction SMoE rate
(the size of the reconstructed PNG rather than the model). See [`docs/AUDIT.md`](../../docs/AUDIT.md).

Current figures live in `paper/figs/generated/` and are regenerated from `results/metrics/*.csv`
by `scripts/06_make_figures.py`.
