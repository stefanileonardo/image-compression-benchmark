# Legacy notebooks

These are the **original** notebooks, kept verbatim as the historical record of how the
study was first run. They are not part of the pipeline and are not maintained.

| Notebook | Superseded by |
|---|---|
| `JPEG.ipynb` | `src/codecs/jpeg.py` + `scripts/02_run_classical.py` |
| `JPEG2000.ipynb` | `src/codecs/jpeg2000.py` + `scripts/02_run_classical.py` |
| `AVIF.ipynb` | `src/codecs/avif.py` + `scripts/02_run_classical.py` |
| `SMoE.ipynb` | `src/smoe/tiling.py` + `scripts/03_run_smoe.py` |
| `GraphsFinal.ipynb` | `src/analysis/` + `scripts/05_aggregate.py`, `06_make_figures.py` |

## They will not run as-is

This is intentional — they are an archive, not a fallback.

- **Hardcoded absolute paths.** Every notebook points at
  `C:\Users\Leonardo\...\Project\Images\...`. That directory no longer exists under that
  name, so the paths are stale even on the original machine. The notebooks fail *soft*
  (`if not path.exists(): print("Path not found"); continue`), so a re-run produces empty
  plots rather than an error — which is how several gaps went unnoticed.
- **`SMoE.ipynb` is Kaggle-only.** 47 references to `/kaggle/input` and `/kaggle/working`,
  plus `!cp`, `!mkdir`, `!ls`, `sed -i`, `!git clone` and seven `%cd` magics. It also
  patches `smoe.py` and `train_model.py` in place at runtime; the vendored copy under
  `src/smoe/vendor/tf_smoe/` already has those patches applied and committed.
- **`GraphsFinal.ipynb` hardcodes its data.** Roughly 350 lines of JPEG / JPEG 2000 /
  AVIF metrics are embedded as Python literals, plus an inline fallback copy of a timing
  table. Figures could drift silently from the code that claimed to produce them. The
  replacement reads `results/metrics/*.csv` and nothing else.

## Known defects preserved here

Reading these alongside `docs/AUDIT.md` shows what changed and why. In particular:

- SMoE rate taken as the size of the reconstructed PNG (B1)
- no byte count recorded during the quality sweeps, so no rate axis (M1)
- `.convert('RGB')` applied to grayscale sources (M2)
- `_box_filter_same` off by one, returning `(H-1, W-1)` (m1)
- `speed = 6` defined but never passed to the AVIF encoder (m2)
- single unwarmed `perf_counter()` timings at the 40 ms scale (M3)

Every one of these is fixed in `src/`.
