# Raster2Seq: Autoregressive polygon sequences for raster→vector floorplans

**Date:** 2026-09-20  
**Paper:** Raster2Seq: Polygon Sequence Generation for Floorplan Reconstruction  
**Venue:** SIGGRAPH 2026  
**Authors:** Hao Phung, Hadar Averbuch-Elor (Cornell)  
**Links:** [Project](https://cornell-vailab.github.io/Raster2Seq/) · [arXiv](https://arxiv.org/abs/2602.09016) · [Code](https://github.com/Cornell-VAILab/Raster2Seq) · [Checkpoints](https://huggingface.co/haopt/Raster2Seq)  
**Benchmarks:** Structured3D-B / density, CubiCasa5K, Raster2Graph; zero-shot style transfer to WAFFLE

---

## What it does

Raster2Seq turns a **raster floorplan image** into a **vector floorplan**: an ordered sequence of labeled polygons (rooms with types, plus doors and windows). Instead of predicting a fixed set of room queries in one shot (RoomFormer, PolyRoom), it **autoregressively emits corner tokens**—each token is `(x, y, semantic)`—so the number of rooms and corners can grow with scene complexity.

Core pieces:

1. **Labeled corner sequence.** Polygons are sorted left-to-right. Each room is a variable-length corner list delimited by special tokens; door/window are extra semantic classes.
2. **Anchor-based autoregressive decoder.** Next-corner prediction conditions on image features, prior tokens, and **learnable spatial anchors** that steer attention to informative regions (walls, junctions).
3. **Token-level semantic loss.** Per-corner classification keeps room type (and door/window) aligned with geometry during generation—RoomFormer-style models often lose 2–5 points on geometry when semantics are added; Raster2Seq reports little or no such drop on Structured3D density maps.

Reported Room F1 (paper tables): **99.6** on Structured3D-B, **88.7** on CubiCasa5K, **97.0** on Raster2Graph—ahead of RoomFormer / FRI-Net / PolyRoom / HEAT on those raster-to-vector setups. CubiCasa5K-trained models also look stronger than RoomFormer on messy real-world WAFFLE internet plans.

---

## Why it matters for COXIT_test_task

Our deliverable is exactly **room polygons + relative areas (+ optional names)** from a single top-down image. Raster2Seq's native output is closer to that schema than FloorSAM's mask-then-vectorize path or classical watershed:

| Need in this repo | Raster2Seq | Current pipeline |
|---|---|---|
| Outer room polygons | Direct sequence of corners | Watershed masks → contours |
| Room names | Built into tokens (CubiCasa-style classes) | Separate VLM `--semantics` stage |
| Variable room count | Autoregressive, no fixed query budget | h-maxima + merge heuristics |
| Doors / windows | Explicit classes | Implicit (doorways = narrow passes) |
| CPU / no-weights default | No — needs GPU + HF checkpoint | Yes (115–290 ms) |

### Transferable ideas (even if we do not ship the full model)

- **Polygon-as-sequence decoding** as an alternative to "segment then contour." Matches next-step #4 in the README (pseudo-label → instance / structured model) better than another mask head alone.
- **Joint geometry + semantics** in one pass could replace the brittle VLM open-plan split for kitchen/living naming—if the domain gap is closed.
- **Complexity scaling:** gains grow as polygon/corner count rises; marketing apartments with 7–12 rooms sit in the regime where query-budget methods degrade.

### Concrete experiment path

1. **Zero-shot smoke test.** Run the official CubiCasa5K checkpoint (`hf:cubicasa5k`, Room F1 88.7 on in-domain test) on `data/*.webp` via [Cornell-VAILab/Raster2Seq](https://github.com/Cornell-VAILab/Raster2Seq). Expect domain shock: CubiCasa is line-drawing / CAD-like raster, our inputs are photoreal 3D marketing renders with furniture and shading.
2. **Preprocess bridge (cheap).** Before inference, feed a **structure-emphasized** image instead of raw RGB—e.g. inverted free-space / wall ΔE map from `preprocess.wall_mask`, or a sketch-like edge map—so the appearance is closer to CubiCasa binary/line plans (same spirit as FloorSAM's density-map → SAM idea).
3. **Map outputs into our JSON.** Convert predicted room polygons to `rooms[].polygon`, compute `relative_area` as today, map CubiCasa labels → our `name` field; drop or ignore door/window polys for the geometric deliverable.
4. **Compare on the three samples.** Room count vs README table (~10 / ~9 / ~7), open-plan split quality vs watershed + VLM, latency vs classical baseline.
5. **Only if zero-shot is hopeless:** fine-tune on Structured3D top-down renders + a handful of hand-labeled marketing plans (README next-step #4/#5), using Raster2Seq as the structured head instead of Mask2Former/YOLO-seg.

---

## Gaps and caveats

1. **Domain gap (largest risk).** Training data are architectural rasters and density maps, not textured 3DPlans-style renders. Furniture, rugs, and wall extrusion shading are out of distribution. WAFFLE generalization helps for "wild" *2D* plans, not photoreal 3D top-downs.
2. **Weight and latency.** Contradicts the default "classical CV only, no ML, no network" mode. Fine as an optional `--ml-vectorize` path or research baseline, not a replacement for the Docker default.
3. **Manhattan / drawing bias.** Anchors and training favor clean wall drawings; low wall–floor ΔE (highlandlux) may still break structure cues the encoder expects.
4. **Open-plan semantics ≠ open-plan geometry.** CubiCasa labels one polygon per room type when walls exist; merged living+kitchen with no wall may still become one polygon—semantics help naming more than inventing missing barriers.
5. **License / dependency surface.** PyTorch ops build (`models/ops`, `diff_ras`), HF download, GPU. Heavier than FloorSAM-style "prompt our barrier into SAM" experiments.

---

## Relation to the FloorSAM note

| | [FloorSAM](2026-09-20-floorsam-sam-guided-room-segmentation.md) | Raster2Seq (this note) |
|---|---|---|
| Input family | LiDAR → BEV density | Raster floorplan drawing / density |
| Segmentation style | SAM masks + filter | Autoregressive labeled polygons |
| Semantics | Optional fusion | First-class in the sequence |
| Best COXIT hook | Reuse barrier + h-maxima as SAM prompts | Structured polygon+name head after domain adaptation |

They are complementary: FloorSAM is a **drop-in segmentation backend** for our existing preprocess; Raster2Seq is a **end-to-end vector+label** alternative if we invest in closing the render↔drawing gap.

---

## Summary

Raster2Seq is the strongest recent **raster→labeled vector floorplan** method with public code and checkpoints. For this repo it is the natural candidate for README next-step #4 (learn a structured model that emits polygons and room types together), with a clear zero-shot / preprocess-bridge experiment on `data/*.webp` before any fine-tuning. Keep classical watershed as the default path until the domain gap is measured.
