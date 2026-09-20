# Splitting open-plan areas by floor finish, offline

**What it adds:** a purely local, network-free replacement for the part of
step 10 that *cuts* an open-plan region ([root ALGORITHM.md, step 10](../../ALGORITHM.md#step-10-the-semantic-layer---semantics)).
The production pipeline can only split a kitchen from a living room by asking a
vision-language model for anchor points. This approach asks the floor instead:
where the material changes, the function usually changes too.

It sits **after** the baseline, consumes its regions, and touches nothing else.

**Result on the three samples in `data/`:**

| Render | Baseline | This approach | Decision on the open-plan region | Baseline partition kept |
|---|---|---|---|---|
| limestone ranch | 9 regions | **10** | split 42.2 % → 31.4 % + 10.9 % | 89 % |
| highlandlux (beige) | 10 regions | **11** | split 49.8 % → 33.5 % + 16.3 % | 84 % |
| heritage towers | 9 regions | **9** | declined — the cut wrapped the furniture | 100 % |

The last column is the share of baseline pixels that stay together in one new
region: it drops exactly by the size of the zone that was cut away, and
everything outside the open-plan region is untouched.

Two of the three renders get a plausible kitchen/living boundary with no API
call. The third is correctly left alone, and that is the part that took the
most work: the naive version happily cut bedrooms into "bed" and "floor".

---

## Contents

1. [The gap this fills](#1-the-gap-this-fills)
2. [The idea](#2-the-idea)
3. [Step by step](#3-step-by-step)
4. [The two guards](#4-the-two-guards)
5. [Applied to the three renders](#5-applied-to-the-three-renders)
6. [How much of this is the features? An ablation](#6-how-much-of-this-is-the-features-an-ablation)
7. [Against the committed vision-language run](#7-against-the-committed-vision-language-run)
8. [What improved, what got worse](#8-what-improved-what-got-worse)
9. [Limitations](#9-limitations)
10. [Regenerating the assets](#10-regenerating-the-assets)

---

## 1. The gap this fills

Every sample in this repository has one region that swallows a third to a half
of the flat: 42 % on `limestone ranch`, 50 % on `highlandlux`, 53 % on
`heritage towers`. It is the kitchen, the dining area and the living room with
no wall between them. The README calls it "the largest single error in all
three samples", and the shipped answer is `--semantics`: send the render to
GPT-4o, get anchor points back, cut geodesic Voronoi cells around them.

That answer costs an API key, 3–5 s per image, and — measured in the root
document — is not reproducible between runs at `temperature=0`.

But the information is often right there in the render. Kitchens are tiled or
planked, living rooms are carpeted; the boundary between the two materials is
drawn in the image, sharp and straight, and it is usually where a floor plan
would put the room boundary as well.

![the limestone ranch render](assets/limestone-ranch-01-input.webp)

*`limestone ranch`. The 42 % region is everything on the left, from the sofa at
the top to the hob at the bottom: light floor above, dark plank below, and the
material changes on a straight line just south of the island.*

## 2. The idea

```mermaid
flowchart TD
    A["one baseline region<br/>≥ 22 % of the plan"] --> B["SLIC superpixels"]
    B --> C["appearance vector per superpixel<br/><i>chroma, relative contrast, coherence, damped L</i>"]
    C --> D["Ward clustering → 4 finishes"]
    D --> E{"large patch<br/>+ runs along the walls?"}
    E -- no --> F["not floor:<br/>rug, sofa, counter"]
    E -- yes --> G["floor seed"]
    G --> H["keep the 2 largest finishes"]
    H --> I["flat watershed inside the region<br/><i>geodesic Voronoi from extended seeds</i>"]
    I --> J{"is the cut short and straight?"}
    J -- no --> K["decline: leave the region whole"]
    J -- yes --> L["two zones"]
```

Three decisions carry the method, and each was forced by a failure:

- **superpixels, not pixels** — a photoreal floor is noisy, and clustering raw
  pixels produces confetti. SLIC gives ~100–170 regions per open-plan area,
  each already respecting the material edges.
- **"is it floor?" before "is it different?"** — the strongest appearance
  boundary in a living room is the sofa, not the flooring, so a cluster has to
  earn the right to be a seed.
- **at most two zones, then a shape test on the cut** — lighting splits one
  material into two clusters. On `limestone ranch` the lit part of the living
  floor and its shaded part are 41 units of OpenCV's 8-bit `L` apart, while
  that floor and the plank floor are 40 apart: *within one material, lighting
  spans the same distance as the material change itself*. The defences that
  work are keeping only the two largest floor seeds and testing the shape of
  the cut; the feature engineering meant to solve it does not, which is
  measured in [§6](#6-how-much-of-this-is-the-features-an-ablation).

## 3. Step by step

Code: [`approach.py`](approach.py). Defaults:
`FinishSplitConfig(min_region_frac=0.22, superpixel_px=900, compactness=12, n_finishes=4, max_zones=2, max_cut_ratio=1.8)`.

### 3.1 Over-segment the region

`slic(..., mask=region, compactness=12, n_segments=area/900)` — 138, 108 and
169 superpixels on the three candidate regions.

![SLIC superpixels inside the open-plan region](assets/limestone-ranch-06-superpixels.webp)

### 3.2 Describe each superpixel

Five numbers, standardised over the region and then weighted:

| Feature | Weight | Intended role |
|---|---|---|
| `L` (median lightness) | 0.3 | carries the material, but also the lighting — damped, not dropped |
| `a`, `b` (median chroma) | 1.0 | warm plank vs neutral tile vs beige carpet |
| mean \|∇L\| / mean `L` | 1.0 | texture energy, **divided by brightness**, so a multiplicative change in illumination cancels |
| structure-tensor coherence | 0.5 | plank flooring has one dominant orientation; carpet has none |

The design intent was that the normalised contrast makes a sunbeam harmless:
brightening a surface scales its gradients by the same factor and the ratio
does not move. It does behave that way — but [§6](#6-how-much-of-this-is-the-features-an-ablation)
shows it is not what produces the results below.

![appearance clusters](assets/limestone-ranch-07-finish-clusters.webp)

*Four Ward clusters on the open-plan region of `limestone ranch`, with their
median 8-bit `L`: orange the shaded light floor (166, 27 % of the region),
magenta the plank floor (126, 31 %), green everything bright — the sunlit floor
near the windows together with the white counters (207, 33 %), blue the sofa
(59, 9 %). The sunlit floor being its own cluster is exactly the failure mode
the feature weighting was meant to prevent, and it happens anyway; what saves
the result is that its patches lose on wall contact and on size in the next
two steps.*

### 3.3 Decide which clusters are floor

A cluster becomes a seed through one of its connected patches when the patch is

- **large** — at least 6 % of the region, and
- **wall-bound** — at least 15 % of its own outline runs along the barrier.

The second test is what rejects furniture. A floor finish reaches the walls all
around its zone; a rug touches nothing, a sofa touches one wall over a short
stretch.

![accepted finish seeds](assets/highlandlux-08-finish-seeds.webp)

*`highlandlux`: the white kitchen tile (23 % of the region, 59 % of its outline
against a wall) and the beige living floor (47 %, 24 %). Both accepted.*

On `limestone ranch` the bright cluster above puts forward four patches and
loses three of them here — one on wall contact (10.6 % of the region but only
8.5 % of its outline against a wall: that is the white counter run) and two on
size (3.0 % and 5.3 %). The sofa cluster is rejected outright at 1.7 % wall
contact.

Only the two largest surviving finishes are kept (`max_zones=2`), which
removes the bright cluster's last patch: with 10.1 % of the region it comes
third behind the shaded floor (24.6 %) and the plank floor (16.1 %). The third
cluster, on all three renders, is a lighting artefact or a furniture group:
with the cap lifted (`--max-zones 4`) `limestone ranch` comes back as three
zones, 51 % / 24 % / 26 %, the extra boundary running along the edge of the
sunlit strip.

### 3.4 Grow the seeds

```python
watershed(np.zeros(region.shape, np.float32), markers=seeds, mask=region)
```

Flat terrain, so the watershed degenerates into geodesic Voronoi cells: every
pixel joins the nearest seed, measured *inside* the region. Because the seeds
are extended patches rather than single points, the boundary lands on the
material edge instead of halfway between two guessed anchors. Zones under 18 %
of the region are merged back into their largest neighbour.

## 4. The two guards

Without them the method is unusable. Run with the guards disabled
(`--min-region-frac 0.12 --max-cut-ratio 99`), every region above 12 % of the
plan gets cut:

| Render | Region | Share of plan | Proposed zones | Cut ratio | What it is | Verdict at defaults |
|---|---|---|---|---|---|---|
| limestone ranch | #3 | 31 % | 74 / 26 | **0.22** | open plan | **split** |
| limestone ranch | #4 | 20 % | 54 / 44 | 2.06 | bedroom | size guard (cut guard would also reject) |
| highlandlux | #2 | 32 % | 67 / 33 | **1.53** | open plan | **split** |
| highlandlux | #7 | 15 % | 69 / 31 | 1.63 | bedroom | size guard only |
| heritage towers | #3 | 13 % | 38 / 62 | 1.02 | bedroom | size guard only |
| heritage towers | #6 | 38 % | 43 / 56 | 3.04 | open plan, one material | **cut-shape guard** |

**The size guard** (`min_region_frac = 0.22`). On these renders every genuine
open-plan area covers 31–38 % of the footprint and the largest single room
stops at 20 %, so the threshold sits between the two populations. It is fitted
on three images and the 20 % bedroom is uncomfortably close to the line — which
is why the second guard matters. A region below the threshold is never
inspected at all, so the cut ratios in the table for those rows come from the
relaxed run, not from the default one.

**The cut-shape guard** (`max_cut_ratio = 1.8`). Measure the length of the
proposed cut in units of `sqrt(region area)`
([`research.common.cut_ratio`](../common.py)). A straight cut across a space
scores around 1, because its length is roughly the width of the space. A cut
that goes around an object has to travel out and back, and scores 2 and up:

![the cut rejected on heritage towers](assets/heritage-towers-09-rejected-cut.webp)

*What the method wanted to do on `heritage towers`: orange is the plank floor,
blue is rug + sofa + kitchen counters. There is only one flooring material in
this region, so the strongest appearance boundary available is the furniture,
and the cut wraps it — ratio 3.04, rejected.*

Neither guard is sufficient alone. Two bedrooms produce a cut that looks
perfectly respectable (1.63 and 1.02) and only the size guard stops them; the
38 % region on `heritage towers` passes the size guard comfortably and only the
cut shape gives it away. Both are needed, and both were fitted on six
candidate regions — with margins of 0.02 in share of plan and 0.26 in cut
ratio, which is thin.

## 5. Applied to the three renders

### limestone ranch — the clean case

![baseline vs finish split, limestone ranch](assets/limestone-ranch-04-side-by-side.webp)

The 42.2 % region becomes 31.4 % (light floor: living, entry and the strip in
front of the closets) and 10.9 % (plank floor: kitchen and the hall in front of
it). Cut ratio 0.22 — the shortest cut in the whole study, because the two
materials meet on a straight line across the narrowest part of the space.

![difference, limestone ranch](assets/limestone-ranch-05-difference.webp)

*Red: the pixels that leave the old region. Blue: baseline boundaries; red
lines: the new ones. Everything outside the open-plan region is untouched.*

The boundary follows the material change, so it runs *south* of the kitchen
island — the island itself ends up with the living zone. A floor plan would
probably draw the line north of it. This is the intrinsic limit of the cue:
it finds where the flooring changes, which is close to, but not identical
with, where the function changes.

### highlandlux — right answer, ragged boundary

![baseline vs finish split, highlandlux](assets/highlandlux-04-side-by-side.webp)

The 49.8 % region becomes 33.5 % (beige living and dining) and 16.3 % (white
tiled kitchen). The tile zone also claims a strip along the bottom wall, where
the beige floor is bright enough under direct light to cluster with the tile —
visible as the thin blue band at the bottom of the seed figure above. Cut ratio
1.53: higher than `limestone ranch` because the kitchen is an L and the cut has
to follow it.

### heritage towers — correctly declined

![baseline vs finish split, heritage towers](assets/heritage-towers-04-side-by-side.webp)

Identical to the baseline, by design. The living room, the hall and the kitchen
of this flat are laid with the *same* dark plank floor, so there is no material
boundary to find, and the method says so rather than inventing one. The
rejected proposal is the figure in [§4](#4-the-two-guards).

This is the case where the vision-language stage is genuinely irreplaceable:
only knowing that a hob means kitchen can divide this space.

## 6. How much of this is the features? An ablation

The appearance vector of [§3.2](#32-describe-each-superpixel) was designed
first and justified by an argument about illumination. Turning its channels off
one at a time says something less flattering:

| Configuration | limestone ranch | highlandlux | heritage towers |
|---|---|---|---|
| **default** (damped `L` + chroma + contrast + coherence) | split 74/26, cut 0.22 | split 67/33, cut 1.53 | declined, cut 3.04 |
| colour only, damped `L` | split 56/44, cut 1.13 | **declined**, cut 2.92 | declined, cut 3.19 |
| colour only, full `L` | split 74/26, cut 0.22 | split 67/33, **cut 1.25** | declined, cut 3.19 |
| texture only (no `L`, no chroma) | **declined**: fewer than two wall-bound finishes | **declined**: finishes too similar | declined |
| default, cap lifted (`--max-zones 4`) | **3 zones** 51/24/26 | split 67/33 | declined |

Reproduce any row with the flags shown in [§10](#10-regenerating-the-assets).

Read honestly:

- **Plain CIELAB with full lightness weight does everything the designed
  vector does** — same two splits, same cut on `limestone ranch`, a slightly
  *shorter* cut on `highlandlux`. On this data the texture channels buy
  nothing.
- **Damping lightness only works because the texture channels compensate for
  it.** Turn them off and keep the damping and the method loses `highlandlux`.
  The pair is self-consistent; neither half is independently justified here.
- **Texture alone is useless**: the finishes stop being separable at all.
- **What actually suppresses the lighting artefact is `max_zones = 2`**, not
  the features. With the cap lifted the sunlit strip becomes its own zone
  regardless of which feature set is used.

The default is left as it is — the illumination-invariance argument is sound,
the richer vector is not worse anywhere, and three renders cannot settle it —
but the honest summary is that the two guards and the two-zone cap carry this
method, and a plain Lab clustering would serve equally well until a render
appears where a sunlit strip outgrows the second material.

## 7. Against the committed vision-language run

`output/semantic/*.json` holds a GPT-4o run of the same three images, so the
two approaches can be compared on the same regions. Shares are of the total
room area, as everywhere in this repository.

| Render | Baseline region | This approach | `--semantics` (GPT-4o) |
|---|---|---|---|
| limestone ranch | 42.2 % | 31.4 % + 10.9 % | living room 23.4 % + kitchen 19.1 % |
| highlandlux | 49.8 % | 33.5 % + 16.3 % | dining room 32.3 % + living room 17.5 % |
| heritage towers | 53.3 % | declined | not split either — named *entry* |

Three observations:

- **They cut different things.** On `highlandlux` the model separated *dining*
  from *living* — both on the same beige carpet — and gave the name *kitchen*
  to a 0.7 % region elsewhere. This approach separated the tiled kitchen from
  everything else. The model is reasoning about furniture; the method here is
  reasoning about materials. Neither is the other's approximation.
- **The model is not obviously better where both fire.** On `limestone ranch`
  its boundary is a straight Voronoi line between two guessed anchor points,
  which the root document already flags as "not the real edge of the floor
  finish". The finish boundary *is* the real edge — of the flooring, at least.
- **On the hardest region both fail the same way.** Neither splits the
  53 % open plan on `heritage towers`; the model only mislabels it.

What the model gives that this cannot: names. This approach produces two
anonymous zones. A plausible combination is to use the finish boundary as the
geometry and the model — or a small furniture detector — only for the labels.

## 8. What improved, what got worse

**Improved**

- The open-plan region is split on 2 of 3 renders **with no API key, no
  network, no weights**, in under a second.
- It is deterministic: SLIC, Ward linkage and the watershed have no random
  component here, so the same image gives the same cut every time. The shipped
  VLM path does not (documented in the root README).
- The boundary is placed on visible evidence — a material edge — not on a line
  between two coordinates a model guessed.
- It refuses to answer when the evidence is absent, which is the behaviour that
  makes it safe to run by default.

**Got worse / new risks**

- **Cuts on material, reports as function.** The kitchen island on
  `limestone ranch` ends up on the living side. Where a flat has continuous
  flooring under a functional boundary, the cut is simply wrong or absent.
- **Two thresholds fitted on three images.** `min_region_frac = 0.22` and
  `max_cut_ratio = 1.8` separate the six candidate regions cleanly, but with
  margins of 0.02 and 0.26 respectively. A fourth render could easily land
  between them.
- **A bright floor can pass for tile** (the bottom strip on `highlandlux`),
  so the zone boundary is ragged where the lighting is strong.
- **The appearance features are not earning their place.** Plain CIELAB
  produces the same two splits ([§6](#6-how-much-of-this-is-the-features-an-ablation)),
  so the extra two channels are cost without demonstrated benefit on this
  data, and the lighting robustness the method actually relies on comes from
  the two-zone cap.
- It costs 0.4–1.0 s per image on top of the baseline's 0.2–0.6 s, which is
  2–3× the geometric pipeline, almost all of it in SLIC and in the per
  superpixel statistics.
- Zones are unnamed, and the id numbering of the whole plan shifts when a
  region is split — the same caveat the semantic stage carries.

## 9. Limitations

- Three renders, six candidate regions, no ground truth. Every number here is
  an observation, not a measurement against an annotation.
- Only one split per region (`max_zones = 2`). A kitchen + dining + living with
  three materials would come back as two zones.
- The method inherits every upstream error: it can only cut regions the
  baseline produced, and on `highlandlux` those regions are already dirty.
- The wall-contact test needs the barrier mask to be reasonable. On a render
  where walls are badly detected, "runs along a wall" stops meaning anything.

**Next steps, in the order I would try them**

1. Replace the two hand-set guards with one test on the *evidence*: how much of
   the proposed cut coincides with a strong, straight appearance edge in the
   image. That subsumes both guards and has a single threshold.
2. Snap the cut to the dominant plan directions. The material edge is straight
   in the render but the Voronoi boundary wobbles around furniture; projecting
   it onto the nearest axis-aligned line would clean up the polygons.
3. Detect the obvious kitchen fixtures (hob, sink, cabinet run) with a small
   template or colour model and use them to *name* the zone the finish split
   already found — the cheap half of what the VLM does.
4. Run the whole thing over a few hundred renders and check how often the
   guards fire, which is the only way to know whether 0.22 and 1.8 mean
   anything outside these three images.

## 10. Regenerating the assets

```bash
uv sync                                                      # once
uv run python research/superpixel-finish-split/run.py        # all three renders
uv run python research/superpixel-finish-split/run.py --image limestone

# the guard inventory in section 4
uv run python research/superpixel-finish-split/run.py \
    --min-region-frac 0.12 --max-cut-ratio 99 --no-assets

# the ablation rows in section 6
uv run python research/superpixel-finish-split/run.py --no-assets \
    --contrast-weight 0 --coherence-weight 0                       # colour only, damped L
uv run python research/superpixel-finish-split/run.py --no-assets \
    --lightness-weight 1 --contrast-weight 0 --coherence-weight 0  # colour only, full L
uv run python research/superpixel-finish-split/run.py --no-assets \
    --lightness-weight 0 --chroma-weight 0                         # texture only
uv run python research/superpixel-finish-split/run.py --no-assets --max-zones 4
```

Every figure in this document and `assets/metrics.json` come from the first
command; the run prints its summary table as markdown. Nothing outside
`research/superpixel-finish-split/assets/` is written.
