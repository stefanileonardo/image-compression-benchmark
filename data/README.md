# Dataset

Five images chosen for **structural diversity** rather than sample size. SMoE training
takes hours per image, which bounded the scale, so the set deliberately spans very
different content types: a smooth ramp with almost no high-frequency energy, dense
stochastic texture, hard geometric edges, natural photographic detail, and a mixture.

## `processed/` — the benchmark set

All four codecs see exactly these pixels. Square, 8-bit, single channel (`mode='L'`),
because the reference SMoE implementation requires square grayscale input.

| File | Resolution | Character |
|---|---|---|
| `Gradient_sq_gray.png` | 1280 x 1280 | Smooth low-frequency ramp |
| `Grass_sq_gray.png` | 1274 x 1274 | Dense stochastic texture |
| `Mixed_sq_gray.png` | 2300 x 2300 | Mixed structure and texture |
| `Nature_sq_gray.png` | 1200 x 1200 | Natural photograph, fine detail |
| `SharpEdges_sq_gray.png` | 1486 x 1486 | Hard geometric edges |

## `raw/` — originals

The uncropped captures, kept so preprocessing is reproducible rather than assumed.
`scripts/01_preprocess.py` regenerates `processed/` from these: centre-crop to a square,
flatten any alpha onto white, convert to 8-bit grayscale.

```bash
python scripts/01_preprocess.py
```

## Why grayscale matters for the results

Because these are `mode='L'`, the codecs encode a **single channel**. The original study
called `.convert('RGB')` first, which encoded three identical channels and inflated the
compressed size by 16-73 % — biasing the size comparison against the classical codecs
while leaving PSNR/SSIM untouched. See `docs/AUDIT.md`, finding M2.

## Licensing

These images were assembled by the author as a test set for this study. `Gradient`,
`SharpEdges` and `Mixed` are synthetic or composed; `Grass` and `Nature` are
photographic. They are distributed with this repository under the same MIT licence as
the code, for the purpose of reproducing the benchmark.

If you intend to reuse the photographic images outside that purpose, or if you fork this
repository for publication elsewhere, substitute a standard test set (Kodak, CLIC) and
re-run the pipeline:

```bash
# drop replacements into data/raw/, then
python scripts/01_preprocess.py
python scripts/02_run_classical.py
python scripts/05_aggregate.py && python scripts/06_make_figures.py
```

Note that the shipped SMoE parameters are tied to *these* images: swapping the dataset
invalidates `results/smoe_params/` and requires retraining (`scripts/03_run_smoe.py`).
