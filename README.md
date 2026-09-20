# Room layout from a 3D floor-plan render

Exploratory prototype (4–6 h scope) that takes **one 3D apartment floor-plan render**
and approximates its 2D room layout with classical computer vision: room-like regions,
their polygon boundaries, and relative pixel-based areas.

The same pipeline runs in two modes.

### Default — classical CV only, no ML, no network

Regions are numbered and labelled with their share of the total room area.

```bash
docker compose run --rm segment "data/limestone ranch_santa fe_625sq.webp"
```

![geometric result](output/limestone%20ranch_santa%20fe_625sq.png)

### With `--semantics` — rooms get names, open-plan gets split

A vision-language model names every region and cuts the open-plan area that geometry
cannot separate: the single 42 % region above becomes **living room 23 % + kitchen 19 %**.
Same polygons, same JSON, only `"name"` is filled in. Needs `OPENAI_API_KEY`;
see [Optional semantic stage](#optional-semantic-stage).

```bash
docker compose run --rm segment-semantic "data/limestone ranch_santa fe_625sq.webp"
```

![semantic result](output/semantic/limestone%20ranch_santa%20fe_625sq.png)

---

## Quick start (Docker)

```bash
# build once
docker compose build

# run the geometric pipeline on a render; writes output/<stem>.png and <stem>.json
docker compose run --rm segment "data/limestone ranch_santa fe_625sq.webp"

# any other file, plus intermediate masks for inspection
docker compose run --rm segment "data/heritage towers_a1_1 Bed 1 Bath 593 Sq. Ft..webp" --debug

# with room names and open-plan splitting (needs OPENAI_API_KEY in .env)
docker compose run --rm segment-semantic "data/limestone ranch_santa fe_625sq.webp"

# all flags
docker compose run --rm segment --help
```

`./data` is mounted read-only, `./output` is mounted for results. The container runs as
uid 1000, so results are not root-owned; if your user has a different uid, run
`HOST_UID=$(id -u) HOST_GID=$(id -g) docker compose run …` (or put both in `.env`). The `segment-semantic`
service is the same image plus `.env`, and its entrypoint already carries `--semantics`
and `--output-dir output/semantic`, so geometric and named results never overwrite
each other.

### Local (uv)

```bash
uv sync --no-dev                          # runtime deps only
uv sync                                   # + jupyter/matplotlib/pytest (dev group is on by default)

uv run python -m floorplan_seg "data/limestone ranch_santa fe_625sq.webp" --debug

# with room names; the CLI loads .env itself
uv run python -m floorplan_seg "data/limestone ranch_santa fe_625sq.webp" \
    -o output/semantic --semantics
```

Python 3.13, dependencies: `opencv-python-headless`, `scikit-image`, `scipy` (+ `numpy`),
`openai`, `pydantic`, `python-dotenv`.

Tests (no network, synthetic images): `uv run pytest`.

---

## Deliverables

| # | Item | Where |
|---|---|---|
| 1 | Annotated images | `output/*.png` (geometric), `output/semantic/*.png` (with VLM names) |
| 2 | JSON with polygons and relative areas | `output/*.json`, `output/semantic/*.json` |
| 3 | Source, `Dockerfile`, `docker-compose.yml` | this repo |
| 4 | README: approach, assumptions, limitations, next steps | this file |
| 5 | Loom walkthrough | link to be added by the author |

The three renders in `data/` were all processed; results are committed so they can be
inspected without running anything.

### JSON schema

```jsonc
{
  "image": "limestone ranch_santa fe_625sq.webp",
  "image_size": {"width": 1140, "height": 855},
  "plan_area_px": 416787,            // footprint incl. walls
  "rooms_area_px": 305365,           // sum of room pixels
  "advertised_total_sqft": 625.0,    // parsed from the file name, may be null
  "units": { ... },                  // what each number means
  "rooms": [
    {
      "id": 3,
      "name": null,                  // filled by the optional semantic stage
      "area_px": 129628,
      "relative_area": 0.4245,       // area_px / rooms_area_px  <- the required metric
      "share_of_plan": 0.311,        // area_px / plan_area_px
      "area_sqft_estimate": 265.3,   // only when advertised_total_sqft is known
      "centroid": [445, 423],
      "polygon": [[x, y], ...]       // outer boundary, pixel coords, origin top-left
    }
  ]
}
```

---

## Approach

The renders are near-orthographic top-down views with **extruded walls**: each wall shows a
bright top face plus a shaded vertical face. Rooms are therefore *floor areas enclosed by
wall-coloured structure*, and the pipeline is built around that.

```
render ─► plan mask ─► wall mask ─► barrier ─► distance-transform watershed ─► merge ─► polygons/JSON
          (flood fill)  (adaptive   (dilated)   (h-maxima markers)             (open boundaries,
                         Lab ΔE)                                                small regions)
```

1. **Plan mask** (`preprocess.plan_mask`). Flood-fill the white page from the four corners,
   keep the largest foreground component (drops the disclaimer text), fill interior pockets
   that do not touch the image border.
2. **Wall colour, per image** (`preprocess.estimate_wall_lab`). The median CIELAB colour of a
   thin ring just inside the plan outline. The outline is always exterior wall, so this is a
   reliable prior and it adapts to each provider's palette. A fixed global threshold worked
   on two of the three samples and failed on the third (beige floor almost as bright as walls).
3. **Wall mask → barrier** (`preprocess.wall_mask`). Pixels within distance 18 of the wall colour
   in OpenCV's 8-bit Lab encoding (≈ 7 CIELAB L\* units — not a CIE ΔE, see ALGORITHM.md),
   opened 3×3, then connected components that are neither large nor elongated are dropped
   (white sanitary ware, light furniture). The full wall network is normally one large
   component, so this filter keeps structure and drops objects. The result is dilated with a
   3×3 kernel (one pixel on each side) so the shaded vertical face also acts as a barrier.
4. **Regions** (`seeds.region_labels`). Euclidean distance transform of the free space
   (plan minus barrier), smoothed σ=2, markers from an **h-maxima** transform (h=10),
   watershed on the negated distance. This splits free space at its narrowest points, which
   is where doorways are. Two corrections follow:
   - adjacent regions whose shared boundary is wider than 16 px are merged — a doorway is
     narrow, a boundary that merely cuts across an open area is not;
   - regions under 0.4 % of the plan are absorbed into the neighbour they touch most; an
     enclosed scrap that touches nothing is attached to the nearest room, so no area is lost.
5. **Export** (`export`). Largest external contour per room, Douglas–Peucker simplified at
   0.4 % of perimeter; relative area = room px / all room px; annotated PNG with tinted
   regions, outlines and share labels.

Every stage is a pure function on NumPy arrays; `--debug` writes `plan.png`, `wall.png`,
`barrier.png`, `regions.png` next to the result.

### Results on the three samples

| Render | Regions found | True rooms (approx.) | Notes |
|---|---|---|---|
| limestone ranch | 9 | ~10 | all rooms separated; kitchen+living+entry are one open-plan region |
| heritage towers | 9 | ~9 | same open-plan merge; bedroom 111.5 sq ft vs 126 from the printed `11'9" x 10'9"` (−12 %) |
| highlandlux | 10 | ~7 | warm beige palette; walls under-detected, some spurious splits |

Relative areas sum to 1.0 by construction. Where the file name carries the advertised total
(e.g. `593 Sq. Ft.`), rooms are also reported in square feet, which is a useful sanity check
(a "bedroom" of 4 sq ft means the segmentation broke).

### What was tried and rejected

- **Local-peak watershed markers** fired several times per room (30/27/28 regions);
  h-maxima gave 12/12/13 before merging.
- **SAM 2** (`sam2.1-hiera-large`, one point prompt per region). No mask candidate behaved
  like a room: the tight one grabbed the rug or bed under the point, the loose one flooded
  through walls (28–48 % of its area on top of walls). Dropped; walls are respected by
  construction with the watershed.
- **Gradient-driven split of open-plan zones** (hoping the cut would follow the change in
  floor finish). Furniture outlines form closed ridges and one basin floods the region
  (zones of 730 px vs 92 596 px). Replaced by a geodesic Voronoi split.

Details and figures are in `segmentation.ipynb`.

---

## Assumptions

- **Top-down, near-orthographic view.** Measured dominant line directions are within ±6° of
  0°/90°; no perspective rectification is done. Strongly oblique isometric renders would
  need it.
- **Walls are the brightest, least saturated large structure** and the plan's outer boundary
  is wall. Both hold for typical marketing renders (3DPlans.com style).
- **Floor and wall colours are distinguishable** (distance ≳ 18 in OpenCV 8-bit Lab, about
  7 L\* units). The third sample is close
  to this limit and shows what happens when it is violated.
- **Limited occlusion.** Extruded walls hide a strip of floor along their far side; furniture
  covers floor but does not enclose it. Neither is compensated.
- **White page background**, and any caption/disclaimer is disconnected from the plan.
- **File-name convention** `<n> Sq. Ft.` / `<n>sq` is optional and used only for the
  square-foot estimate.

### Manual correction (documented)

No manual preprocessing was applied to the samples. The CLI exposes the few parameters that
matter as *light manual correction* when a render is off-distribution:

| Flag | Effect | Try when |
|---|---|---|
| `--wall-delta-e` | wall colour tolerance | walls broken (raise) / floor eaten (lower) |
| `--wall-dilate` | barrier thickness | rooms leaking through door frames (raise) |
| `--h-maxima` | marker significance | too many rooms (raise) / rooms merged (lower) |
| `--merge-width` | doorway width cut-off | open areas fragmented (raise) |
| `--min-area` | small-region cut-off | closets lost (lower) |

---

## Limitations

- **Open-plan areas stay one region.** Kitchen, dining and living with no wall between them
  have no geometric boundary; only semantics (furniture) can split them. This is the largest
  single error in all three samples.
- **Areas run low by ~10 %.** Boundaries follow visible floor, not the wall centreline;
  extruded walls hide floor along their shaded side.
- **White fixtures adjacent to walls** (bathtubs, kitchen runs, tiled splashbacks) can merge
  into the wall network and survive the compactness filter, biting into bathrooms and kitchens.
- **Palette sensitivity.** Low wall/floor contrast degrades everything downstream (third sample).
- **No room types** in the geometric stage — regions are numbered only.
- **Polygons are outer contours**; interior holes (islands, white furniture) are not
  represented.
- Evaluated on **three images**, without ground-truth masks. Numbers above are indications.

---

## Optional semantic stage

`--semantics` sends the render plus the numbered overlay to a vision-language model
(set-of-mark prompting, structured output). The model names each region and, for a region
holding several functional zones, returns anchor points; the region is then split into
geodesic Voronoi cells around them. This stage needs `OPENAI_API_KEY` in `.env` (never
committed; see `.gitignore`). If it fails — no key, API error, refusal — the geometric
result is still written (with `"name": null`) and the exit code is 1.

```bash
# Docker: the service mounts .env and already passes --semantics
docker compose run --rm segment-semantic "data/limestone ranch_santa fe_625sq.webp"

# local: pass the flag yourself; the CLI loads .env on its own
uv run python -m floorplan_seg "data/…webp" -o output/semantic --semantics
```

Results are committed in `output/semantic/`.

![semantic result](output/semantic/limestone%20ranch_santa%20fe_625sq.png)

What changed against the geometric run on `limestone ranch` — 9 regions became 10 named
rooms, because the open-plan area was cut in two:

| Geometric region | With `--semantics` |
|---|---|
| 42.4 % (265.3 sq ft) — one open-plan region | **living room** 23.4 % (146.0 sq ft) + **kitchen** 19.1 % (119.3 sq ft) |
| 26.1 % | **bedroom** (162.8 sq ft) |
| 12.2 % | **bathroom** (75.9 sq ft) |
| 7.3 % | **walk-in closet** (45.7 sq ft) |
| 4.0 / 3.1 / 2.4 / 1.5 / 1.1 % | storage, entry, laundry, balcony, closet |

In the JSON the only difference is that `"name"` is no longer `null`; `polygon`,
`area_px` and `relative_area` keep the same meaning. Room ids are renumbered when a
region is split, so they are not comparable between the two runs.

Accuracy is uneven and this is a demonstration, not a finished component:

- `limestone ranch` — in the committed run all ten names are plausible, but the
  living/kitchen boundary is a straight Voronoi line between two anchor points, not the
  real edge of the floor finish.
- **Runs disagree even at `temperature=0`.** An earlier run of the same image split the
  open-plan area 30.9 / 11.6 % instead of 23.4 / 19.1 %, and called the 6.8 sq ft niche
  *dining room* rather than *closet*. The zone anchors the model returns are imprecise and
  GPT-4o is not fully deterministic.
- `heritage towers` and `highlandlux` — several labels are wrong; the beige palette of the
  third render already degrades the geometry the model is asked to interpret.

---

## Next steps

1. **Close the open-plan gap without an API**: detect kitchen fixtures (hob, sink, cabinet
   run) and seating with a small detector, use them as zone anchors for the Voronoi split.
2. **Wall centreline recovery**: estimate wall thickness from the barrier mask and grow rooms
   half a wall inward so areas match printed dimensions; validate against the `13'3" x 13'3"`
   captions via OCR.
3. **Low-contrast palettes**: add a texture/shading cue (walls are flat, floors are textured)
   alongside colour, or a two-class GMM on the ring vs interior instead of a single ΔE.
4. **Scale up and distil**: run the pipeline over a few hundred renders, review the output as
   pseudo-labels, and train one instance-segmentation model (Mask2Former / YOLO-seg) that
   also learns room types — removing both the palette sensitivity and the VLM dependency.
5. **Evaluation set**: hand-annotate 20–30 renders to replace the indications above with
   IoU / room-count metrics.

---

## Repository layout

```
floorplan_seg/
  cli.py          command-line entry (python -m floorplan_seg)
  config.py       dataclasses with every tunable and its rationale
  preprocess.py   plan mask, adaptive wall colour, wall/barrier masks
  seeds.py        h-maxima markers, watershed, boundary-width merge
  export.py       polygons, relative areas, JSON, annotated image
  semantics.py    optional VLM naming + open-plan split
  pipeline.py     orchestration, Room/Segmentation dataclasses
  viz.py          overlays
segmentation.ipynb  exploration notebook: measurements, rejected ideas, figures
data/               the three sample renders
output/             committed results (geometric); output/semantic/ (with VLM)
Dockerfile, docker-compose.yml
```

## Loom walkthrough (outline)

1. Setup: `docker compose build && docker compose run --rm segment <file>`.
2. Code structure: `cli → pipeline → preprocess → seeds → export`; config as documentation.
3. Pipeline live with `--debug`: plan → wall → barrier → regions → annotated result.
4. Results: what is right (rooms, doorways, relative areas), what is not (open plan, −10 %
   areas, beige sample), and why SAM 2 and gradient splitting were rejected.
