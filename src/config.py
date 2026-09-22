"""
Central configuration for the QueryEarth DLPK.

Every path here is resolved relative to this file's location (the DLPK
root) rather than hardcoded to a drive letter, so the package works
wherever ArcGIS Pro unpacks it.
"""

import glob
import os
import torch

# ------------------------------------------------------------------
# Layout
# ------------------------------------------------------------------
EMBEDDINGS_DIR = rf"/mnt/sdc1/susanket/embeddings"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _find_first(directory: str, patterns) -> str:
    """Return the first file matching any of `patterns` (globs) under
    `directory`, or None. Lets the DLPK pick up whatever filename the
    checkpoint/weights happen to have instead of hardcoding one."""
    if not os.path.isdir(directory):
        return None
    for pattern in patterns:
        hits = sorted(glob.glob(os.path.join(directory, pattern)))
        if hits:
            return hits[0]
    return None


# ------------------------------------------------------------------
# Demography assets
# ------------------------------------------------------------------
DEMO_PARQUET_PATH = os.path.join(EMBEDDINGS_DIR, "demography-emb.parquet")

# Trained TabularTextCLIP checkpoint. Expected in weights/, e.g.
# weights/demo_clip.pth — auto-detected so you don't have to hardcode a name.
DEMO_CLIP_CKPT_PATH = os.path.join(EMBEDDINGS_DIR, "demo_embedder.pth")

# Local, offline text embedder for demo-text queries. Smallest well-supported general
# text embedder available (~23M params, 384-dim, CPU-friendly, no network).
TEXT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TEXT_EMBED_DIM = 384  # all-MiniLM-L6-v2 output dim. Update this if you swap the embedder later.

# ------------------------------------------------------------------
# Vision assets
# ------------------------------------------------------------------
VISION_INDEX_DIRS = {
    "low": rf"{EMBEDDINGS_DIR}/turboquant/vision-low-2026",
    "high": rf"{EMBEDDINGS_DIR}/turboquant/vision-high-2026",
}
DEFAULT_RESOLUTION = "high"
RESOLUTION_DIST_MAP = {
    "low": 2900,
    "high": 290
}

# Time-period labels for change detection. Each maps to a year whose
# vision index folder is named vision-{resolution}-{year}.
VISION_YEARS = {
    "past":    2014,
    "recent":  2020,
    "present": 2026,
}

# Per-year vision index directories, keyed as {year: {resolution: path}}.
VISION_YEAR_INDEX_DIRS = {
    year: {
        res: rf"{EMBEDDINGS_DIR}/turboquant/vision-{res}-{year}"
        for res in ("low", "high")
    }
    for year in VISION_YEARS.values()
}

# Maximum distance (in degrees ~ metres) between two points to consider
# them the "same location" across time periods for change detection.
# ~0.001 ≈ 111 m at the equator.
CHANGE_DISTANCE_THRESHOLD = 0.001

# Text encoder used to embed vision *queries* into the same space the offline
# image embeddings were computed in. This must match whatever model produced
# the embeddings baked into embeddings/{lowres,highres}-vision — unrelated to
# TurboQuant itself, which only compresses the already-computed image side.
CLIP_VISION_MODEL_NAME = "ViT-H-14"
CLIP_VISION_PRETRAINED = "laion2b_s32b_b79k"
CLIP_VLM_PATH = rf"{EMBEDDINGS_DIR}/RS5M_ViT-H-14.pt"

# ------------------------------------------------------------------
# OSM assets
# ------------------------------------------------------------------
OSM_EMBEDDING_DIR = rf"{EMBEDDINGS_DIR}/osm"
OSM_YEARS = ["2014", "2026"]
OSM_DEFAULT_YEAR = "2026"
OSM_CATEGORY_FILES = [
    'landuse.parquet',
    'pois.parquet',
    'roads.parquet',
    'waterway.parquet',
]

# ------------------------------------------------------------------
# Search defaults
# ------------------------------------------------------------------
VISION_NPROBE_DEFAULT = 24  # IVF clusters probed per global (non-spatially-filtered) vision search
MAX_RESULTS = 100  # Maximum features any single step can return
AUTOMATIC_BUFFER_RADIUS = 1