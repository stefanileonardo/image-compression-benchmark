# Correctness Audit

> **Status: closed for release.** This document records an audit of the *original*
> version of the study. Every finding below has been either fixed in this repository or
> explicitly documented as a limitation — see the status table immediately below. It is
> kept in full because the corrections change what the study concludes, and that record
> should be public rather than quietly rewritten.

## Status summary

| ID | Finding | Status in this repository |
|---|---|---|
| **B1** | SMoE rate taken from the reconstructed PNG | **Fixed.** `src/smoe/rate.py` derives rate from the parameter count; paper and README rewritten around it. Regression-tested. |
| **B2** | Two SMoE folders mislabelled by kernel budget | **Fixed.** Renamed to verified counts (`K016`/`K032`/`K096`); `scripts/04_smoe_metrics.py` re-checks at runtime; `tests/test_smoe_rate.py` guards it. |
| **B3** | Two configurations produced identical reconstructions | **Fixed.** Duplicate quarantined in `results/smoe_params/_disputed_second_K016_run/`, loaded by nothing. Three budgets reported, not four. |
| **B4** | Infinite PSNR fed into `np.mean` / `np.max` | **Fixed.** `src/analysis/aggregate.py` states and applies an explicit exclusion policy; lossless points reported separately. Tested. |
| **B5** | 96k *Mixed* run incomplete but reported | **Fixed.** Excluded from aggregates, flagged by a runtime warning, and disclosed in the paper's Limitations. |
| **M1** | No rate axis anywhere | **Fixed.** Byte count recorded at every operating point; all figures now plot against bits per pixel. |
| **M2** | Grayscale encoded as RGB | **Fixed.** Codecs encode single-channel natively. Tested. |
| **M3** | Single unwarmed timing measurements | **Fixed.** Median of N runs after a warm-up, configurable. Cross-hardware caveat stated in the paper. |
| **M4** | Non-monotonic SSIM in the tables | **Resolved.** The tables were regenerated from code; the dips do not reproduce. Monotonicity is now a test. |
| **m1** | `_box_filter_same` off by one | **Fixed.** Corrected and verified against `scikit-image` to ~1e-15. |
| **m2** | AVIF `speed` never passed to the encoder | **Fixed.** |
| **m3** | Docs named the wrong AVIF library | **Fixed.** |
| **m4** | Metrics scraped from notebook output cells | **Fixed.** Metrics written to CSV at compute time; the scraper is gone. |
| **m5** | Duplicate `Graphs.ipynb` | **Fixed.** Superseded notebooks archived under `notebooks/legacy/`. |
| **m6** | Figures hardcoded their own data | **Fixed.** `src/analysis/figures.py` reads only `results/metrics/*.csv`. |
| **P1** | Stale absolute Windows paths | **Fixed.** All paths resolve from the repository root via `src/config.py`. Tested. |
| **P2** | SMoE notebook was Kaggle-only | **Fixed.** `src/smoe/tiling.py` is a portable port; upstream vendored pre-patched. |
| **P3** | Runtime environment mutation | **Fixed.** No `pip install`, `git clone` or `sed -i` at runtime. |
| **P4** | No environment specification | **Fixed.** Pinned `requirements.txt`, `ENVIRONMENT.md`, CI workflow. |
| **P5** | Lock files, `__pycache__`, four vendor copies | **Fixed.** Removed; one vendored copy; `.gitignore` added. |
| **P6** | 4,232 files / 2.1 GB of intermediates | **Fixed.** 76 MB, ~390 files. Provenance in `results/PROVENANCE.md`. |

**Open, by choice:** the $K=64$ run was never performed and is not reconstructable without
hours of GPU time; parameter quantisation and entropy coding for SMoE are not implemented.
Both are disclosed in the paper's Limitations section and in the README.

---

**Scope:** all code, results and write-up in the original `Project V1` folder, as of 8 August 2026.
**Method:** static review of every notebook and script; inspection of stored artefacts
(reconstructions, checkpoints, parameter pickles, logs); small numerical spot-checks. The heavy
SMoE training was **not** re-run — findings about it are derived from the stored parameter files,
which is sufficient to establish the rate and configuration errors below.

**Verdict:** the paper's qualitative conclusion — SMoE is slower and lower-fidelity than JPEG,
JPEG 2000 and AVIF — is well supported and survives the audit. Several specific quantitative
claims do not, and the file-size argument is wrong in *direction*, not just magnitude.

---

## B1 — SMoE "compressed size" is the decoder output, not the bitstream

**Severity: blocking. This invalidates a section of the paper.**

Table `tab:perimage_filesizes` reports a SMoE column of 266 / 331 / 752 / 544 / 331 KB. Those
values match, byte for byte, the PNG files in
`Code/Final/SMoE/96k_100i_43212s/tf-smoe-master/results/`:

| Image | Paper's "SMoE size" | `*_reconstructed.png` on disk |
|---|---|---|
| Gradient | 266 KB | 266.2 KB |
| Grass | 331 KB | 331.3 KB |
| Nature | 544 KB | 544.8 KB |
| Sharp Edges | 331 KB | 331.4 KB |
| Mixed | 752 KB | **file does not exist** (see B5) |

A PNG of the *decoded* image measures the entropy of a blurry reconstruction. It is not what a
SMoE decoder would need to receive. The actual payload is the fitted model: per kernel,
$\pi$ (1) + $\boldsymbol{\mu}$ (2) + $\mathbf{A}$ (4) + $\boldsymbol{\gamma}_e$ (2) + $\nu_e$ (1)
= **10 float32 values**.

Counting them from `params_best.pkl` across all tiles:

| Config | Image | Tiles | Parameters | bpp @ fp32 | bpp @ 8-bit* |
|---|---|---|---|---|---|
| `32k_…` | Gradient | 9 | 23,040 | 0.45 | 0.11 |
| `32k_…` | Nature | 9 | 23,040 | 0.51 | 0.13 |
| `64k_…` | Gradient | 9 | 92,160 | 1.80 | 0.45 |
| `64k_…` | Nature | 9 | 92,090 | 2.05 | 0.51 |
| `96k_…` | Gradient | 9 | 799,200 | **15.61** | 3.90 |
| `96k_…` | Grass | 9 | 729,630 | **14.39** | 3.60 |
| `96k_…` | Nature | 9 | 671,510 | **14.92** | 3.73 |
| `96k_…` | Sharp Edges | 16 | 1,389,040 | **20.13** | 5.03 |

\* hypothetical — **no quantisation or entropy coding of the SMoE parameters is implemented
anywhere in this project.** The 8-bit column is a lower bound offered for context only.

Reference points: the uncompressed original is **8.00 bpp**. AVIF at its top setting stores
*Nature* in 855 KB = **4.86 bpp** at roughly 59 dB PSNR. SMoE at $K\approx96$ needs **14.92 bpp**
as stored, for roughly 29 dB.

So at the operating point the paper actually reports, SMoE costs **almost twice the uncompressed
image** and delivers 30 dB less than AVIF. The claims

> "SMoE files are smaller in size than JPEG and JPEG2000"
> "the sizes suggest a rather competitive compression compared to established transform codecs"
> "SMoE fails the efficiency criterion … but not the size criterion"

are the opposite of what the data show. The RQ2 answer and the "Practical Implications" section
both depend on this and need rewriting.

**Fix:** compute SMoE rate from the parameter count. State explicitly that no quantisation or
entropy coder was implemented, and present the fp32 figure as the honest measured rate with the
8-bit figure as an optimistic bound.

---

## B2 — Two SMoE result folders are mislabelled

**Severity: blocking.** Kernel counts recovered from `params_best.pkl` (`pis` array length =
number of surviving kernels):

| Folder | Kernels per tile | Implied $K$ | Folder claims |
|---|---|---|---|
| `16k_100i_3535s` | 256 | **16** | 16 ✅ |
| `32k_100i_7899s` | 256 | **16** | 32 ❌ |
| `64k_100i_25201s` | 1024 | **32** | 64 ❌ |
| `96k_100i_43212s` | 8642–9017 | **≈96** (from 9216, $\ell_1$-pruned) | 96 ✅ |

256 = $16^2$ and 1024 = $32^2$ exactly, with no pruning, so these are unambiguous.

Consequences: the $x$-axis of every "PSNR/SSIM vs. number of kernels" plot is wrong at two of four
points, `kernel_configs` in `SMoE.ipynb` cell 21 and `smoe_configs_for_si` in `GraphsFinal.ipynb`
cell 4 both encode the wrong values, and **the $K=64$ run was never performed**. Only three
budgets exist: 16, 32, 96.

---

## B3 — The two lowest SMoE configurations are the same run

**Severity: blocking.** MD5 of the final stitched reconstructions:

| Image | `16k_…` | `32k_…` | Identical? |
|---|---|---|---|
| Gradient | `737cb4b1ff050bb6` | `737cb4b1ff050bb6` | **yes** |
| Grass | `73c0077329b47c81` | `73c0077329b47c81` | **yes** |
| Mixed | `b508febe041b8014` | `b508febe041b8014` | **yes** |
| Nature | `88e4ba549a29378d` | `88e4ba549a29378d` | **yes** |
| Sharp Edges | `6d3c4d7f58e220bc` | `6d3c4d7f58e220bc` | **yes** |

Byte-identical across all five images — impossible for two independent stochastic training runs.
Corroborating evidence: `16k_100i_3535s/…/results_tiles/` contains only three of the five images
(Gradient, Grass, Mixed), so its top-level `results/` cannot have been produced by its own tile
data; and B2 shows both folders hold $K=16$ parameters.

So the SMoE quality curve has **three real points plotted as four**, with the lowest one counted
twice. Every aggregate min/mean/max over SMoE is weighted accordingly.

The two folders nonetheless carry different wall-clock times (3535 s vs 7899 s), so the *timing*
runs and the *stored reconstructions* disagree. At least one of the two is misattributed.

---

## B4 — Infinite PSNR at the JPEG 2000 lossless point

**Severity: blocking.** Verified directly: `pil.save(buf, 'JPEG2000', irreversible=False)` with no
`quality_layers` produces a truly lossless file — measured MSE exactly `0.0`, therefore
$\mathrm{PSNR} = \infty$.

`GraphsFinal.ipynb` cell 3 hardcodes this: `{'quality': 100, 'psnr': float('inf'), 'ssim': 1.0000}`
for all five images. Cell 2 aggregates with `np.mean` and `np.max` and contains no `np.isfinite`,
`np.nanmax` or filtering of the top quality point.

Any mean or max over that array is `inf`. The paper nonetheless reports a JPEG 2000 mean PSNR of
**42.7 dB** and standard deviation **9.49** — numbers the committed code cannot produce. Either a
different, uncommitted computation generated the figure, or the values were entered manually.

**Fix:** adopt an explicit policy — exclude the lossless point from PSNR aggregates, or cap it at a
stated ceiling — apply it in code, and state it in the paper caption. The same applies to the
`ssim: 1.0000` entries.

---

## B5 — The $K\approx96$ run for *Mixed* is incomplete

**Severity: blocking.** Evidence:

- `96k_100i_43212s/…/results_tiles/Mixed_sq_gray/` has **13 tiles**; every other configuration has
  **25** for the same image.
- `96k_100i_43212s/…/results/Mixed_sq_gray_reconstructed.png` **does not exist**.
- `computationaltime/96ktime.txt` records `**(log incomplete)**` for the Mixed row.

Yet the paper's size table lists 752 KB for Mixed/SMoE. No artefact in the repository has that
size. The folder-name total of 43212 s also exceeds the sum of the four logged 96k timings
(33,083 s) by 10,129 s — an unlogged residual attributed to the incomplete run.

**Fix:** either finish the run or mark the cell as not available.

---

## M1 — No rate–distortion analysis

**Severity: major.** Every figure plots quality against a "quality setting" whose meaning changes
per codec:

| Codec | What the $x$-axis actually is |
|---|---|
| JPEG | Quantisation-table quality factor, 10–100 |
| JPEG 2000 | An arbitrary label 10–100 mapped to PSNR targets 28–60 dB |
| AVIF | libaom quality, 10–100 |
| SMoE | Kernels per dimension, 16/32/96 |

These are not commensurable — a JPEG quality of 50 and an AVIF quality of 50 are unrelated
operating points. Codec comparison is conventionally done as PSNR (or SSIM) versus **bits per
pixel**, which places every codec on one physically meaningful axis and makes BD-rate computable.

The quality sweeps in `JPEG.ipynb`, `JPEG2000.ipynb` and `AVIF.ipynb` **never record the encoded
byte count**, even though the data is available for free (`buf.tell()` on the `BytesIO` they
already have). Only the single top-quality file per codec is written to disk and measured.

**Fix:** record `buf.tell()` at every sweep point and re-plot on a bpp axis. This is a small change
that would substantially strengthen the paper — and it is what makes B1's SMoE rate comparable.

---

## M2 — Grayscale images encoded as RGB

**Severity: major.** All five sources are `mode='L'`. Every notebook calls `.convert('RGB')` before
encoding, so all three classical codecs encode three identical channels.

Measured on `SharpEdges_sq_gray.png` (Pillow, 4:4:4, `optimize=True`):

| Quality | RGB-encoded | Native grayscale | Overhead |
|---|---|---|---|
| 10 | 40.5 KB | 23.5 KB | **+72.5 %** |
| 50 | 70.2 KB | 53.2 KB | **+32.0 %** |
| 90 | 125.6 KB | 108.6 KB | **+15.7 %** |

PSNR and SSIM are unaffected — averaging three identical channels changes nothing. **Rate is
materially affected**, and in the direction that penalises the classical codecs in the size table.
JPEG 2000 and AVIF both have native single-channel modes and would benefit further.

**Fix:** drop `.convert('RGB')` and encode `mode='L'` directly. Handle the decoded array as 2-D in
the metric functions (they already support it).

---

## M3 — Timing methodology

**Severity: major.** Encode times are single unwarmed `time.perf_counter()` measurements at the
0.04–0.2 s scale, with no repetition, no warm-up and no median. The hardcoded tables show the
expected symptom — non-monotonic noise:

- JPEG *Gradient*: Q40 = 0.0461 s but Q70 = 0.0357 s (higher quality, faster).
- JPEG *Mixed*: Q20 = 0.2059 s, Q30 = 0.1580 s, Q70 = 0.2324 s, Q100 = 0.1775 s.

The 10⁴–10⁵× SMoE gap is far larger than this noise, so the headline conclusion is safe. But the
per-quality timing curves are not interpretable as presented.

**Fix:** `timeit`-style repetition (≥ 5 runs after a warm-up), report the median, and state
whether decode time is included. Also note that SMoE ran on a Tesla P100 while classical codecs ran
on an i7-1165G7 CPU — a cross-hardware comparison that should be stated in the figure captions,
not only in the setup section.

---

## M4 — Non-monotonic SSIM

**Severity: major — needs investigation.** In the hardcoded JPEG table, SSIM drops at every fourth
quality step and recovers:

| Image | Q50 | Q60 | Q70 | Q80 |
|---|---|---|---|---|
| Grass | 0.9788 | 0.9823 | **0.9211** | 0.9905 |
| Nature | 0.9445 | 0.9530 | **0.9346** | 0.9739 |
| Sharp Edges | 0.9884 | 0.9902 | **0.9481** | 0.9945 |
| Gradient | 0.9886 | 0.9897 | **0.9686** | 0.9937 |

The same dip appears at Q40 across images. SSIM should rise monotonically with JPEG quality, and
the corresponding PSNR values *do* rise monotonically — so PSNR and SSIM disagree systematically at
exactly Q40 and Q70, on every image. That pattern is too regular to be content-dependent and
points at a bug in how those specific rows were produced or transcribed rather than a real effect.

**Fix:** regenerate the table from code and confirm the dips disappear. If they persist,
cross-check against `skimage.metrics.structural_similarity` at the same settings.

---

## m1 — `_box_filter_same` is off by one

**Severity: minor.** For an $H\times W$ input the function returns $(H-1)\times(W-1)$, and the
window is shifted one pixel. Verified on a $64\times64$ ramp: output shape `(63, 63)`, and
`out[19,19]` equals the true centred $11\times11$ mean at pixel `(20,20)`.

The correct sliding-window identity is $\sum_{k=i}^{i+w-1} x_k = C_{i+w-1} - C_{i-1}$, which needs
the cumulative sum prepended with a zero row/column; the code uses `tmp[w:] - tmp[:-w]`, giving
$C_{i+w} - C_i$.

Both images pass through the same filter, so the SSIM map stays self-consistent and the effect on
the mean is small — one boundary row and column are dropped. It is still a correctness bug and the
name `_same` is wrong.

Separately: the window is **uniform**, not the Gaussian $\sigma = 1.5$ of Wang et al. Measured on
*Sharp Edges*:

| Quality | This project | `skimage` Gaussian | `skimage` uniform |
|---|---|---|---|
| 10 | 0.96587 | 0.96058 | 0.95758 |
| 50 | 0.98837 | 0.98403 | 0.98288 |
| 90 | 0.99733 | 0.99580 | 0.99570 |

Consistently high by 0.003–0.008. Legitimate as a variant, but it must be stated in the paper, and
comparison against any externally published SSIM number is invalid without it.

---

## m2–m6 — Code quality

| ID | Finding |
|---|---|
| **m2** | `AVIF.ipynb` defines `speed = 6` with the comment "AVIF speed/quality tradeoff" and never passes it to `save()`. All AVIF encodes ran at the library default, so the AVIF timings do not reflect the documented configuration. |
| **m3** | `AVIF.ipynb` markdown claims `pillow-avif-plugin`; the code installs and uses `pillow-heif`. These are different libraries with different defaults. |
| **m4** | `extract_smoe_metrics.py` recovers PSNR/SSIM by regex-scraping *saved notebook output cells*. It breaks if the notebook is re-run or cleared, and its hardcoded path (`…\Project\Code\Final\SMoE\SMoE.ipynb`) no longer resolves. Metrics belong in CSV written at compute time. |
| **m5** | `Graphs/Graphs.ipynb` is superseded by `Graphs/GraphsFinal.ipynb`. Two notebooks that produce overlapping figures is a correctness hazard. |
| **m6** | `GraphsFinal.ipynb` cell 3 hardcodes ~350 lines of JPEG / JPEG 2000 / AVIF metrics as Python literals, plus an inline `fallback_tables` copy of the 16k timing table (because `computationaltime/16ktime.txt` is **0 bytes**). Figures can therefore drift silently from the code that claims to produce them. |

One thing done **right**: `algo_styles` is defined in cell 2 and used in cells 3–4, so
`GraphsFinal.ipynb` has no cell-ordering hazard on a clean top-to-bottom run.

---

## P1–P6 — Portability

The project as committed **runs only on the original machine**, and arguably not even there.

### P1 — Stale absolute Windows paths

Every notebook hardcodes paths of the form:

```
C:\Users\Leonardo\Documents\Università\Erasmus\Subjects\Quantitative Research\Project\Images\…
```

Found in `JPEG.ipynb`, `JPEG2000.ipynb`, `AVIF.ipynb`, `Graphs.ipynb`, `GraphsFinal.ipynb`,
`SMoE.ipynb` and `extract_smoe_metrics.py`. The directory has since been renamed to
**`Project V1`**, so these paths are broken even on Leonardo's own machine. The notebooks fail
soft — `if not path.exists(): print("Path not found"); continue` — so a re-run would silently
produce empty plots rather than an error.

`GraphsFinal.ipynb` has a partial fallback to the SMoE `data/` folder, which is the only reason it
would still produce anything.

### P2 — Kaggle-only SMoE notebook

`SMoE.ipynb` contains 47 references to `/kaggle/input` and `/kaggle/working`, plus `!cp`,
`!mkdir -p`, `!ls`, `sed -i`, `!git clone` and 7 `%cd` magics. None of it runs on Windows or on a
plain local Jupyter install. Cell 1 is boilerplate from the Kaggle template and does nothing.

### P3 — Runtime environment mutation

The notebooks install packages (`!pip install`, `%pip install`), clone upstream (`!git clone`), and
rewrite `smoe.py` and `train_model.py` in place with string replacement and `sed -i` to force
TensorFlow 1.x compatibility. Cell 16 even prints a warning when its patch target is not found and
asks the user to edit the file by hand — the pipeline is not deterministic.

### P4 — No environment specification

No `requirements.txt`, `environment.yml`, `.gitignore`, `LICENSE` or `README` exists at the root.
Versions of Python, TensorFlow, Pillow, `pillow-heif`, libaom and OpenJPEG are unrecorded.

This matters more than usual here: **codec output is version-dependent.** A different libaom or
OpenJPEG build produces different bytes and different PSNR at the same nominal quality, so the
results are not reproducible without pins.

### P5 — Repository hygiene

- `~$Notes.docx`, `~$search Proposal.docx` — Word lock files, committed by accident.
- `ops/__pycache__/*.pyc` — 20 compiled files.
- Four near-identical copies of the upstream `tf-smoe` repository, one per configuration.
- `Context/` — 24 timestamped scratch Markdown files plus `codexcontext.txt`.
- `computationaltime/16ktime.txt` is 0 bytes, with its content living inline in a notebook instead.

### P6 — Bulk

| Type | Count | Nature |
|---|---|---|
| `.png` | 1,721 | Mostly per-iteration reconstructions and loss plots |
| `.pkl` | 861 | Per-tile parameter snapshots |
| `.index` / `.data-00000-of-00001` / `.meta` | 433 / 433 / 432 | TensorFlow checkpoints |
| `.ipynb` | 9 | |
| `.py` | 22 | Mostly duplicated across the four vendored copies |

Roughly 4,200 files, the overwhelming majority of them regenerable intermediates. These should not
enter Git history — once committed, they cannot be removed without a rewrite.

---

## What remains

Everything in the status table above is closed. Two things are deliberately left open,
because neither can be resolved without new computation:

1. **The $K=64$ run.** The original ladder claimed four kernel budgets; only three were
   ever executed. Filling the gap between $K=32$ and $K=96$ needs roughly 7 GPU-hours
   (`python scripts/03_run_smoe.py --kernels 64`). Until then the study reports three
   budgets and says so.
2. **SMoE parameter quantisation and entropy coding.** The reference implementation
   stores `float32` with no coding, so the measured rates are what that produces. A
   purpose-built SMoE codec would do better. The repository reports an 8-bit bound
   alongside the `float32` figure and states plainly that the bound is not achieved.
   This is the single change most likely to alter the conclusions, and it is named as
   such in the paper's Future Work.

## What holds up

Worth stating plainly, because the list above is long:

- The **core conclusion is correct and robust**. SMoE is slower by three to five orders of
  magnitude and lower in fidelity at every budget tested. Fixing B1 makes this case *stronger*, not
  weaker — the rate story turns out to favour the classical codecs too.
- The **PSNR implementation is correct**, including the `inf` guard at zero MSE.
- The **SI implementation is correct** — proper Sobel kernels, `std` of gradient magnitude, matching
  ITU-T P.910.
- The **tiling and Hann-window blending are sound**: correct edge clamping, de-duplicated
  coordinates, proper normalisation by accumulated weight with an $\varepsilon$ guard.
- The **experimental design is well motivated** — five images chosen for structural diversity
  (smooth ramp, stochastic texture, hard edges, natural detail, mixed) is a sensible answer to the
  constraint that SMoE takes hours per image.
- The **paper is clearly written** and honest about SMoE's weaknesses; the Research Ethics section
  is a genuine strength.
