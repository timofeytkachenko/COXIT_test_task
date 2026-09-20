# FloorSAM: SAM-guided room segmentation from BEV density maps

**Date:** 2026-09-20  
**Paper:** FloorSAM: SAM-Guided Floorplan Reconstruction with Semantic-Geometric Fusion  
**Authors:** arXiv:2509.15750, September 2025  
**Links:** [arXiv](https://arxiv.org/abs/2509.15750) | [PDF](https://arxiv.org/pdf/2509.15750) | [Code](https://github.com/Silentbarber/FloorSAM)  
**Datasets:** Giblayout, ISPRS

---

## What it does

FloorSAM reconstructs vector floorplans from **LiDAR point clouds** via a multi-stage pipeline: it generates a bird's-eye-view (BEV) density map from near-ceiling points, applies adaptive image enhancement, then uses Segment Anything Model (SAM) in zero-shot mode with adaptive prompt points to produce room masks. These masks are filtered through a multistage refinement process and combined with geometry-based contour extraction and regularization to recover room topology as structured vector plans.

The key insight: instead of training a domain-specific segmentation model, the method transforms 3D point-cloud data into a 2D wall/structure-emphasizing top-down map, then leverages SAM's strong zero-shot generalization to segment rooms without training data.

---

## Pipeline sketch

1. **Point cloud processing**  
   Grid and filter near-ceiling points (ceiling height typically 2.4–3.0 m) from the LiDAR scan.

2. **BEV density map generation**  
   Project filtered points into a top-down 2D grid with adaptive resolution based on point cloud density. Denser regions (walls, columns) appear brighter; empty floor space is dark.

3. **Image enhancement**  
   Apply contrast enhancement and noise reduction to emphasize structural boundaries in the density map.

4. **Adaptive prompt generation**  
   Compute candidate room center points via distance transform + h-maxima or clustering methods. Unlike classical markers for watershed, these serve as SAM prompts.

5. **SAM multi-mask room segmentation**  
   Feed the density map and prompt points into Segment Anything Model. SAM returns multiple mask candidates per prompt; select masks based on size, shape, and overlap constraints.

6. **Multistage mask filtering**  
   Filter and merge masks through:
   - Invalid geometry rejection (too small, too large, extreme aspect ratios)
   - Overlap resolution (IoU-based merging or suppression)
   - Semantic-geometric fusion (optional: refine boundaries using structural cues from the density map)

7. **Contour extraction and regularization**  
   Convert filtered masks to vector contours. Apply Douglas–Peucker simplification and Manhattan-world regularization (snap angles to 0°/90°, align collinear segments) to produce clean room polygons.

8. **Room topology recovery**  
   Infer door positions and adjacency graph from mask boundaries and density map gaps. Output structured floorplan representation (rooms, walls, doors).

---

## Why it matters for COXIT_test_task

Our input is a **marketing-style photorealistic top-down render** (textured RGB with furniture, lighting, and shadows), not LiDAR point clouds. The modality is fundamentally different. However, FloorSAM demonstrates a transferable architectural idea:

### Transferable concept
Build a wall/structure-emphasizing 2D map (analogous to their BEV density map), then use **SAM with adaptive prompts + mask filtering** as a room segmentation backend instead of (or as a fallback to) classical watershed.

### Concrete relevance
- Our current pipeline (ALGORITHM.md) already extracts a `barrier` mask (dilated walls) and computes a distance transform with h-maxima markers for watershed.
- FloorSAM suggests: instead of watershed, feed a wall-highlighted map into SAM with those same h-maxima centroids as point prompts.
- SAM's training on diverse natural images might generalize better to non-Manhattan layouts, open-plan spaces with subtle floor-finish boundaries, and rooms with unusual shapes than hand-tuned watershed + merge heuristics.

### Potential advantages over current watershed approach
- **Open-plan robustness:** SAM may learn to respect subtle visual cues (floor finish changes, furniture arrangement patterns) that watershed ignores, potentially splitting kitchen/living areas more reliably than geodesic Voronoi + VLM.
- **Fewer hand-tuned parameters:** The current pipeline has `wall_delta_e`, `wall_dilate`, `h_maxima`, `passage_merge_width`, `min_region_area_frac` (see ALGORITHM.md §13). SAM-based segmentation could reduce reliance on merge-width thresholds and open-boundary heuristics.
- **Non-Manhattan geometry:** Curved walls, angled rooms, and L-shaped spaces challenge watershed's assumption that doorways are narrow passes; SAM learns shape priors from training data.

---

## Gaps and caveats

### 1. Input modality mismatch
- **FloorSAM input:** sparse LiDAR points → BEV density map (structure information, minimal texture)
- **COXIT input:** dense RGB render with photorealistic materials, furniture clutter, shadows, and reflections
- The optimal preprocessing for SAM prompts differs. LiDAR density naturally emphasizes structure; RGB renders may need Lab ΔE wall highlighting (as in our current `preprocess.wall_mask`) or inverted free-space masks to achieve similar emphasis.

### 2. Model deployment cost
- SAM adds ~400 MB model weights (SAM ViT-B) to ~900 MB (SAM ViT-H), plus GPU memory and inference time.
- Our current pipeline is classical CV, runs in **115–290 ms CPU-only** (ALGORITHM.md §14), no network connection, no model weights.
- Acceptable for batch offline processing; may be prohibitive for real-time or edge deployment.

### 3. Prompt engineering for RGB renders
- FloorSAM's adaptive prompts assume the density map encodes ceiling structure reliably.
- In photorealistic renders, furniture, rugs, and lighting gradients introduce visual clutter. Centroid prompts from h-maxima on the distance transform may land on furniture rather than room centers.
- May need prompt augmentation: multiple prompts per room, or negative prompts on detected furniture.

### 4. Doorway and furniture occlusion
- SAM's zero-shot masks may leak through open doorways (no density discontinuity in our renders like there is in LiDAR ceiling-plane gaps).
- Furniture clusters (sofas, beds) might be segmented as separate instances rather than part of the room.
- Likely need post-filtering similar to FloorSAM's multistage mask refinement, adapted to RGB clutter patterns.

### 5. Output format compatibility
- FloorSAM outputs vector floorplans with wall geometries and room topology.
- COXIT requires `polygon` + `relative_area` in the existing JSON schema (README.md §3.2).
- SAM masks are binary rasters; must convert to polygons via `cv.findContours` + Douglas–Peucker (same as current `export.room_polygon`), no additional complexity.

### 6. No public evaluation on photorealistic renders
- FloorSAM is validated on **Giblayout** and **ISPRS** (both LiDAR-derived datasets).
- Transferability to 3DPlans.com-style marketing renders is unproven. Would need empirical validation on our `data/*.webp` samples.

---

## Concrete next experiment

To test FloorSAM-inspired segmentation on COXIT data **without re-implementing the full pipeline**, use our existing preprocessing as the SAM input:

### Option A: Feed existing `barrier` / free-space mask to SAM
1. **Preprocessing:** Use current `preprocess.wall_mask` to generate the barrier, or invert it to get a free-space mask. Alternatively, create a wall-highlighted map by computing Lab ΔE against the wall color and thresholding (visualizing wall prominence).
2. **Prompts:** Extract h-maxima centroids from the distance transform (current `seeds.py:157-159`) as point prompts.
3. **SAM inference:** Pass the preprocessed map + prompts to `sam2.1-hiera-large` or `SAM ViT-B`, collect multi-mask output per prompt.
4. **Filtering:** Apply FloorSAM-style rules:
   - Reject masks smaller than 0.4% or larger than 50% of plan area
   - Merge masks with IoU > 0.7
   - Drop masks with >30% overlap with the barrier (leaking through walls)
5. **Polygon export:** Convert final masks to polygons via existing `export.py` code.

### Option B: Compare room IoU / count vs watershed
- Run Option A on the three `data/*.webp` samples.
- Compare against committed watershed results in `output/*.json`:
  - Room count accuracy (ground truth: ~10 / ~9 / ~7 for limestone / heritage / highlandlux)
  - Per-room IoU if manual annotations are created
  - Qualitative: does SAM split the open-plan kitchen/living better than geodesic Voronoi?
- Measure latency overhead (SAM inference time vs 115–290 ms classical baseline).

### Option C: Fallback architecture
- Keep classical watershed as default (fast, zero dependencies).
- Invoke SAM only when heuristic quality gates fail:
  - Any room >40% of total area (probable open-plan merge)
  - Room count <5 or >15 (likely over/under-segmentation)
- If SAM improves IoU by >10% on failures, integrate as fallback; otherwise document as explored but not cost-effective.

---

## Related work (brief)

- **SpatialGen** (arXiv:2509.14981, September 2025): Layout-to-3D scene generation. Opposite direction (layout → rendering), not a segmentation method. Useful context for understanding layout representation formats, but not a drop-in alternative for our room recovery task.

- **CubiCasa5K, DeepFloorplan** (prior work, noted in ALGORITHM.md): Trained on 2D architectural drawings (line-based wall representations). Transfer poorly to photorealistic 3D renders. FloorSAM's zero-shot approach via SAM sidesteps the domain gap by not training on floorplan-specific data.

---

## Summary

FloorSAM offers a **methodological blueprint**: preprocess domain-specific input into a structure-emphasizing 2D map, then apply a strong zero-shot vision model (SAM) with adaptive prompts instead of hand-tuned classical segmentation. For COXIT's photorealistic render input, the core transferable step is using SAM + mask filtering as a watershed alternative, with existing preprocessing (wall masks, distance transforms, h-maxima) supplying the prompts. The main unknowns are RGB-domain prompt robustness, computational cost vs accuracy tradeoff, and whether SAM's learned priors help with open-plan splitting. A focused experiment (Option A/B above) on the three existing samples would answer these questions without full re-architecture.
