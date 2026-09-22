"""
Visual similarity search, backed by a TurboQuant-compressed embedding index
per resolution (see turboquant_index.TurboQuantSearchIndex).

Replaces the previous raw-NPZ memory-mapped brute-force GPU scan: the index
is already built (rotated, IVF-clustered, bit-packed) ahead of time by
build_turboquant_index.py, so a query here only ever decompresses and scores
a small fraction of the index (nprobe clusters for global search, or the
region-filtered candidate rows for spatially-restricted search).
"""
from typing import Optional
import numpy as np

import geopandas as gpd
import asyncio
from shapely.geometry import Point

import config
from schema import empty_gdf, from_geometries
from operations.threshold import compute_threshold


GROUND=False

def search_vision(target: str,
                   region: Optional[gpd.GeoDataFrame],
                   vision_encoder: "models.VisionEncoder",
                   vision_grounder: "models.SpatialGrounder",
                   turbo_index: Optional["turboquant_index.TurboQuantSearchIndex"],
                   resolution: str = config.DEFAULT_RESOLUTION,
                   nprobe: int = config.VISION_NPROBE_DEFAULT) -> gpd.GeoDataFrame:
    """Find the top-N tile locations whose visual embedding best matches a
    text query, optionally restricted to `region`."""
    if turbo_index is None:
        print(f"No vision index loaded for resolution '{resolution}'.")
        return empty_gdf()

    query_vector = vision_encoder.encode_text(target)  # (1, D), normalized, on DEVICE
    query_np = query_vector.squeeze(0).detach().cpu().numpy()

    if region is not None and not region.empty:
        print(f"[{resolution}] Region-filtered TurboQuant search...")
    else:
        print(f"[{resolution}] Global IVF-routed TurboQuant search (nprobe={nprobe})...")

    scores, lat, lon = turbo_index.search(query_np, region=region, nprobe=nprobe)

    # Adaptive Thresholding
    mask = compute_threshold(scores)
    scores = scores[mask == 1]
    lat = lat[mask == 1]
    lon = lon[mask == 1]

    if GROUND:
        grounded_targets = vision_grounder.ground_locations(
            locations=list(zip(lat, lon)),
            query=target,
            # radius_meters=200,
        )

        if len(scores) == 0:
            print(f"No tiles at resolution '{resolution}' matched (or none fall inside the given region).")
            return empty_gdf()

        # Unpack detections into flat lists
        points = []
        detection_scores = []
        tile_scores = []
        queries = []

        for tile_score, tile_detections in zip(scores, grounded_targets):
            for det in tile_detections:
                # Create Shapely Point from grounded coordinates
                points.append(Point(det["lon"], det["lat"]))
                detection_scores.append(det["confidence"])
                tile_scores.append(tile_score)
                queries.append(det["query"])

        # Fallback if vector search found tiles, but GroundingDINO detected 0 matching objects
        if not points:
            print(f"No grounded objects matching '{target}' were found inside candidate tiles.")
            return empty_gdf()

        # Build GeoDataFrame from extracted detections
        gdf = gpd.GeoDataFrame(
            {
                "target": queries,
                "detection_score": detection_scores,
                "tile_score": tile_scores,
            },
            geometry=points,
            crs="EPSG:4326"
        )

    else:
        if len(scores) == 0:
            print(f"No tiles at resolution '{resolution}' matched (or none fall inside the given region).")
            return empty_gdf()

        geometries = gpd.points_from_xy(lon, lat)
        gdf = from_geometries(geometries, scores=scores)

        # # Region Masking
        # mask = mask_region(gdf, region)
        # if mask.any():
        #     mask_indices = np.where(mask)[0]
        #     gdf = gdf.iloc[mask_indices].copy().reset_index(drop=True)

    return gdf

