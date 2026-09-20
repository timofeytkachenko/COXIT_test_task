# CAGE: Continuity-Aware edGE network for vector floorplans

**Date:** 2026-09-20  
**Paper:** CAGE: Continuity-Aware edGE Network Unlocks Robust Floorplan Reconstruction  
**Venue:** NeurIPS 2025  
**Authors:** Yiyi Liu, Chunyang Liu, Bohan Wang, Weiqin Jiao, Bojian Wu, Lubin Fan, Yuwei Chen, Fashuai Li, Biao Xiong  
**Links:** [arXiv](https://arxiv.org/abs/2509.15459) · [PDF](https://arxiv.org/pdf/2509.15459) · [Project](https://ee-liu.github.io/CAGE_page/) · [Code](https://github.com/ee-Liu/CAGE) · [Checkpoints](https://drive.google.com/drive/folders/1jajjRamJ7SVgCWB-Tihp-ToqPsv0GmE7)  
**Benchmarks:** Structured3D, SceneCAD (ScanNet RGB-D → density maps)

    10|---

## What it does

CAGE reconstructs **watertight vector floorplans** from **2D point-cloud density maps**. Instead of predicting corner sequences (RoomFormer, PolyRoom) or assembling lines after the fact (FRI-Net), it uses a **native edge-centric** representation: each wall segment is a directed, geometrically continuous edge whose endpoints need not land exactly on polygon vertices.

Core pieces:

1. **Edge tokens.** Each room is an ordered sequence of up to *N* directed edges; each edge is two endpoints in normalized 2D plus a validity class. Invalid edges pad unused slots; an all-invalid room is padding.
2. **Image backbone + transformer encoder.** Density map (typically 256×256) → ResNet-50 or Swin Transformer V2 features → standard deformable-attention-style encoder.
    20|3. **Dual-query decoder (DN-DETR style).** **Latent** edge queries produce the final polygons; **perturbed** queries (noisy GT endpoints + optional label flips) get denoising losses so the model learns to recover clean walls from corrupted structure. At inference only latent queries run.
4. **Iterative refinement.** Across decoder layers, edge geometry and room polygons tighten; intersecting / chaining predicted edges yields closed room polygons without a separate integer-programming stage.

Reported Structured3D F1 (no post-process): **99.1% rooms**, **91.7% corners**, **89.3% angles** — ahead of HEAT / RoomFormer / FRI-Net / PolyRoom / PolyGraph on those fine-grained geometry metrics, at ~0.01 s/image (same ballpark as RoomFormer). Cross-dataset (train Structured3D → test SceneCAD) stays competitive without heavy post-optimization; PolyDiffuse refinement adds only marginal gains, which the authors take as evidence the edge prior already enforces topology.

---

## Why it matters for COXIT_test_task

Our deliverable is **room polygons + relative areas** from one top-down image. CAGE's output is vector rooms, but its training domain is **structure-emphasizing density maps**, not textured marketing renders — same family of gap as FloorSAM and Raster2Seq, with a different inductive bias.
    30|
| Need in this repo | CAGE | Current pipeline |
|---|---|---|
| Outer room polygons | Edge sequences → closed rooms | Watershed masks → contours |
| Topology / watertightness | Explicit edge continuity + denoising | Merge-by-passage-width heuristics |
| Occlusion / weak walls | Designed for missing density | Low ΔE walls (highlandlux) still break barrier |
| Room names | Optional semantic-rich extension in upstream RoomFormer lineage; not the main claim | Separate VLM `--semantics` |
| CPU / no-weights default | No — PyTorch, ops build, GPU checkpoint | Yes (115–290 ms) |

### Transferable ideas (even without shipping the full model)
    40|
- **Edges over corners for fragile walls.** Marketing renders often lose thin or low-contrast wall tops; corner detectors fragment, then polygons go non-watertight. An edge prior that allows endpoints *along* a wall segment matches how our `barrier` is a dilated ribbon, not a crisp centerline.
- **Denoising as a training trick for classical→learned hybrids.** The dual-query idea (corrupt GT structure, force recovery) is reusable if we ever fine-tune a polygon head on noisy pseudo-labels from our watershed pipeline (README next-step #4).
- **Density / barrier preprocess bridge.** Same cheap experiment as FloorSAM / Raster2Seq: feed `wall_mask` / inverted free-space / edge sketch into a Structured3D-trained checkpoint and measure room-count vs the three `data/*.webp` samples before any fine-tune.

### Concrete experiment path

1. **Install + checkpoint.** Follow [ee-Liu/CAGE](https://github.com/ee-Liu/CAGE): conda env, compile `models/ops` + `diff_ras`, download Google Drive weights.
2. **Build a fake density map** from each marketing render (invert free space, or normalize the barrier/distance map to `[0,1]` grayscale 256×256) so appearance is closer to Structured3D projections.
3. **Zero-shot eval** with `./tools/eval_stru3d.sh`-style entry (`eval.py`) on those maps; export predicted polygons to our JSON schema (`rooms[].polygon`, recompute `relative_area`).
    50|4. **Compare** room count and open-plan splits vs watershed + VLM on limestone / heritage / highlandlux; note latency and failure modes (furniture as "walls", double-wall vs centerline).
5. **Only if promising:** fine-tune with Structured3D density + a few hand-labeled marketing plans, optionally enabling `semantic_classes` for joint naming.

---

## Gaps and caveats

1. **Domain gap (largest risk).** Input is LiDAR/RGB-D **density**, not photoreal RGB with furniture and shadows. A wall-highlighted preprocess helps but does not invent missing structure cues.
2. **Weight and build cost.** Deformable-attention CUDA ops + differentiable rasterization + Swin/ResNet checkpoints. Contradicts the Docker default "classical CV only." Fine as an optional research baseline, not a replacement.
3. **Fixed query budget (*M* rooms × *N* edges).** Autoregressive Raster2Seq scales corner count more naturally; CAGE inherits DETR-style caps. Large open-plan apartments with many corners may need higher *N* or post-merge.
    60|4. **Open-plan living+kitchen.** Edge continuity recovers walls that exist in the density map; it does not invent a partition where the render has none — VLM / semantics still needed for that split.
5. **Single-room bias on SceneCAD.** Strong SceneCAD numbers partly reflect single-room ScanNet layouts; multi-room marketing plans are closer to Structured3D, where CAGE already leads.

---

## Relation to existing notes

| | [FloorSAM](2026-09-20-floorsam-sam-guided-room-segmentation.md) | [Raster2Seq](2026-09-20-raster2seq-polygon-sequence-floorplan.md) | CAGE (this note) |
|---|---|---|---|
| Representation | SAM masks → contours | Autoregressive labeled corners | Directed continuous **edges** |
    70|| Input family | LiDAR → BEV density | Raster drawing / density | Point cloud → density |
| Topology story | Mask filter + Manhattan snap | Sequence closure via tokens | Edge continuity + denoising |
| Best COXIT hook | Barrier + h-maxima as SAM prompts | Structured polygon+name head | Edge-prior vector head after density bridge |

Complementary stack: FloorSAM for **mask segmentation** on a wall map, CAGE for **edge-based vectorization** of that structure map, Raster2Seq when **joint room-type labels** matter more than geometric edge priors.

---

## Summary

    80|CAGE is the strongest recent **edge-centric, end-to-end density→vector floorplan** method with public code and checkpoints (NeurIPS 2025). For this repo it is the natural counterpart to Raster2Seq's corner sequences: try it when watershed polygons are fragmented or non-watertight under weak walls, using the same structure-emphasized preprocess bridge. Keep classical watershed as the default until the density-map domain gap is measured on `data/*.webp`.
