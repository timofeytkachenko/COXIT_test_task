# Research: alternative approaches to room segmentation

This folder is a workspace for trying other ways to get room polygons out of a
3D floor-plan render, and for showing — on the images in `data/`, with figures
— how each one compares against the pipeline that ships in `floorplan_seg/`.

It is deliberately **separate from the product**. Nothing here is imported by
`floorplan_seg`, nothing here changes its behaviour, and `docker compose run
segment …` produces exactly the same output as before this folder existed. The
dependency goes one way: research code imports the production pipeline, runs it
as the baseline, and compares against it.

## Approaches

| Approach | What it replaces | Result on `data/` |
|---|---|---|
| [`medial-axis-doorways/`](medial-axis-doorways/ALGORITHM.md) | the h-maxima markers of step 5 | baseline 9/9/10 regions → 9/8/12 (limestone / heritage / highlandlux). Same partition on the clean render, one spurious region removed on the second, floor area recovered on the third that the baseline drops; 3 regions of total movement over a ±35 % parameter sweep against 8 for the baseline, and one bathroom wrongly cut in two. |
| [`superpixel-finish-split/`](superpixel-finish-split/ALGORITHM.md) | the *cut* part of the optional VLM stage (step 10) | splits the open-plan region on 2 of 3 renders with no network — 42 % → 31 %+11 %, 50 % → 33 %+16 % — and correctly declines on the render whose living room and kitchen share one plank floor. |

Both documents apply the method to the three renders in `data/`, embed the
generated figures, and state what got worse as well as what got better.

## Layout

```
research/
  README.md              this file
  common.py              shared baseline runs, figures and metrics
  run_all.py             regenerate every approach
  <approach-slug>/
    ALGORITHM.md         the method, applied to real images, compared to the baseline
    approach.py          the method itself: pure functions + a frozen config dataclass
    run.py               CLI that regenerates assets/ and prints a summary table
    assets/              generated figures (.webp) and metrics.json
```

`assets/` is committed so the documents can be read on GitHub without running
anything. Every file in it is reproducible from `run.py`; none of them is a
stock image.

## Regenerating everything

```bash
uv sync                                  # once; Python 3.13, no extra dependencies
uv run python research/run_all.py        # ~20 s for both approaches on three renders
```

Per approach, with options:

```bash
uv run python research/medial-axis-doorways/run.py --help
uv run python research/medial-axis-doorways/run.py --image limestone --markers cores
uv run python research/superpixel-finish-split/run.py --min-region-frac 0.12 --max-cut-ratio 99
```

Each run writes `<approach>/assets/*.webp` and `<approach>/assets/metrics.json`
and prints its summary table as markdown, ready to paste into the document.
Runs are deterministic: where an upstream function has a random component
(`skimage.morphology.medial_axis` breaks ties at random — see
[that approach's note](medial-axis-doorways/ALGORITHM.md#6-reproducibility-a-random-tie-break-and-how-to-avoid-it)),
the seed is pinned in the config.

The research code is covered by `tests/test_research.py`, which runs offline
against the synthetic render in `tests/conftest.py`:

```bash
uv run pytest tests/test_research.py
```

## Conventions

An approach directory is expected to hold:

1. **`ALGORITHM.md`** — the method *and* its application: what it changes
   against the baseline, the steps with the code references, the figures on at
   least two of the three renders, a table of numbers, and an explicit "what
   got worse" section. Negative results are the point of a research folder.
2. **`approach.py`** — the method as importable functions with a frozen
   `*Config` dataclass, type annotations and NumPy-style docstrings, matching
   the style of `floorplan_seg/`. No I/O, no argument parsing, no figures.
3. **`run.py`** — everything I/O: it runs the baseline and the approach over
   `data/`, writes the assets and the metrics, and prints a summary. It is the
   single command named in the document's "regenerating" section.
4. **`assets/`** — at minimum the input crop, the baseline overlay, the new
   overlay, a captioned side-by-side and a difference view, per render, plus
   whatever intermediate figure explains the method. WebP at quality 88 keeps a
   1140 × 855 overlay near 150 kB.

Two rules that make the comparisons mean something:

- **Change one stage.** An approach that replaces the marker stage runs its
  output through the *production* merge/absorb tail
  (`common.postprocess_like_baseline`), so any difference comes from the stage
  under test. This is why `research/common.py` imports a few private helpers of
  `floorplan_seg.seeds`; `tests/test_research.py` fails loudly if they are
  renamed.
- **Measure, then claim.** Region counts alone are not evidence. `common.py`
  provides the metrics used in both documents: matched IoU and surviving-pixel
  agreement against the baseline (`agreement`), the geometry of the cuts a
  partition makes (`cut_stats`, `cut_ratio`), and per-region area tables
  (`region_table`).

## Shared tooling (`common.py`)

| Function | Use |
|---|---|
| `sample_images()`, `baseline(path)` | the three renders and the production result for one of them |
| `free_space(seg)`, `distance_map(free)` | the masks every approach starts from |
| `postprocess_like_baseline(...)` | the production merge/absorb/renumber tail |
| `overlay`, `palette`, `matched_palette` | region overlays; the matched palette gives the same room the same colour in both panels of a comparison |
| `difference_view`, `changed_pixels` | what moved between two partitions |
| `row`, `caption`, `crop_to`, `save` | figure composition and WebP output |
| `agreement`, `cut_stats`, `cut_ratio`, `region_table` | the numbers |
| `load_approach(slug)` | import an approach module from a hyphenated directory |

## Ideas not explored here

Kept as a list rather than as empty directories:

- **Felzenszwalb or normalised-cut over the whole plan** instead of a
  watershed. Cheap to try with the tooling here; the reason it was not is that
  the barrier already disconnects most rooms (see
  [the measurement](medial-axis-doorways/ALGORITHM.md#1-why-touch-the-marker-stage-at-all)),
  so a different region grower has little left to decide.
- **Superpixel graph cut for the *wall* mask**, replacing the per-pixel Lab
  threshold. This is where the low-contrast render actually fails, upstream of
  everything both approaches here touch.
- **Line-topology polygonalisation**: detect wall lines, build an axis-aligned
  arrangement, label cells, and report rectilinear polygons at wall
  centrelines. It would attack the systematic ~10 % area shortfall that the
  root README documents, which neither approach here addresses.
- **Learned segmentation** distilled from pseudo-labels, as sketched in the
  root README's next steps. Out of scope for an offline, weight-free folder.
