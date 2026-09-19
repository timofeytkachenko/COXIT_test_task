"""Tunable parameters for the room-segmentation pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PreprocessConfig:
    """Parameters for plan extraction and wall detection.

    Attributes
    ----------
    bg_flood_tol
        Grey-level tolerance used when flood-filling the page background from
        the image corners.
    wall_ring_inner, wall_ring_outer
        Erosion depths (px) delimiting the ring just inside the plan outline.
        The ring is assumed to be wall and provides the per-image wall colour.
    wall_delta_e
        CIELAB distance below which a pixel is considered wall-coloured.
    wall_min_component_frac
        Wall-coloured blobs smaller than this fraction of the largest such blob
        are discarded unless they are elongated. The full wall network is
        normally a single large component, so this removes white furniture and
        sanitary ware while keeping the structure.
    wall_min_elongation
        Aspect ratio of the minimum-area rectangle above which a small blob is
        kept anyway, so that short isolated wall stubs survive.
    wall_dilate
        Dilation (px) applied to the wall skeleton so that the shaded vertical
        face of a wall, which is not wall-coloured, still acts as a barrier.
    """

    bg_flood_tol: int = 6
    wall_ring_inner: int = 2
    wall_ring_outer: int = 9
    wall_delta_e: float = 18.0
    wall_min_component_frac: float = 0.02
    wall_min_elongation: float = 4.0
    wall_dilate: int = 3


@dataclass(frozen=True)
class SeedConfig:
    """Parameters for turning free space into candidate room regions.

    Attributes
    ----------
    smooth_sigma
        Gaussian smoothing of the distance transform before marker extraction,
        which suppresses ripples caused by furniture outlines.
    h_maxima
        Dynamic-range threshold for the h-maxima transform used to place one
        marker per room. Measured on the sample renders, this produced far
        fewer spurious markers than local-peak detection.
    passage_merge_width
        Two adjacent regions are merged when the free space along their shared
        boundary is wider than this (px). A doorway is narrow; a boundary that
        merely cuts across an open area is wide.
    min_region_area_frac
        Regions smaller than this fraction of the plan area are merged into
        the neighbour they touch most.
    """

    smooth_sigma: float = 2.0
    h_maxima: float = 10.0
    passage_merge_width: float = 16.0
    min_region_area_frac: float = 0.004


@dataclass(frozen=True)
class SemanticsConfig:
    """Parameters for the vision-language labelling step.

    Regions are labelled with set-of-mark prompting: the numbered overlay is
    sent to the model, which names each number and, for a region holding
    several functional zones, returns an anchor point per zone.

    Attributes
    ----------
    enabled
        When ``False`` the pipeline returns geometric regions without labels
        and without splitting open-plan areas.
    model
        Chat-completions model identifier.
    api_key_env
        Environment variable holding the API key. The key is never logged.
    max_image_side
        Longest side (px) of the overlay sent to the API.
    split_open_plan
        Whether to subdivide a region that the model reports as holding
        several functional zones.
    temperature
        Sampling temperature. Kept at zero because repeated calls at the
        default temperature disagreed with each other on small regions.
    request_timeout_s
        Per-request timeout in seconds.
    """

    enabled: bool = True
    model: str = "gpt-4o"
    api_key_env: str = "OPENAI_API_KEY"
    max_image_side: int = 1280
    split_open_plan: bool = True
    temperature: float = 0.0
    request_timeout_s: float = 90.0


@dataclass(frozen=True)
class PipelineConfig:
    """Top-level configuration."""

    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    seeds: SeedConfig = field(default_factory=SeedConfig)
    semantics: SemanticsConfig = field(default_factory=SemanticsConfig)
