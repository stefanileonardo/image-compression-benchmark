# Environment for the committed CSVs

`results/metrics/classical.csv` was produced on the environment below.
**Codec output is build-dependent** -- a different libjpeg-turbo, OpenJPEG or
libaom will give different bytes at the same nominal quality, so regenerating
on your machine will shift the rate figures slightly. That is expected.

| Component | Version |
|---|---|
| Python | 3.10.12 |
| Platform | Linux x86_64 |
| numpy | 2.2.6 |
| pandas | 2.3.3 |
| Pillow | 12.2.0 |
| libjpeg (via Pillow) | 6.2 |
| OpenJPEG (via Pillow) | 2.5.4 |
| pillow-heif | 1.5.0 |

## Sweep invocation

```bash
python scripts/02_run_classical.py --repeats 3 --images Gradient_sq_gray Grass_sq_gray
python scripts/02_run_classical.py --repeats 3 --append --images Nature_sq_gray SharpEdges_sq_gray
python scripts/02_run_classical.py --repeats 1 --append --images Mixed_sq_gray
```

`Mixed_sq_gray` used `--repeats 1` because it is 2300x2300; its `encode_seconds`
column is therefore a single unwarmed measurement and less reliable than the rest.
Rate and distortion for that image are unaffected.
