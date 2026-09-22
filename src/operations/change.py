"""
Change-detection search.

Compares visual embeddings from two different time periods to find areas
that have changed.  The user provides a query (e.g. "buildings"),
a from-time and to-time (past/recent/present mapping to 2014/2020/2026),
and a mode ("new" or "removed").

- mode="new": finds features that APPEARED (not in from_time, but in to_time)
- mode="removed": finds features that DISAPPEARED (in from_time, but not in to_time)

Use from_time="recent", to_time="present" for changes in the past 5 years.
Use from_time="past", to_time="present" for long-term changes over 10 years.
"""

from typing import Optional, Dict, List, Tuple, Union
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import geopandas as gpd
from scipy.spatial import cKDTree
from shapely.geometry import Point

import config
from schema import empty_gdf, from_geometries
from operations.threshold import compute_threshold


def _nearest_match_coords(
    from_coords: np.ndarray,
    to_coords: np.ndarray,
    threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Finds nearest neighbor within threshold using KDTree.
    
    Returns array pairs of matching indices (to_indices, from_indices).
    """
    if len(from_coords) == 0 or len(to_coords) == 0:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)

    tree = cKDTree(from_coords)
    dists, from_indices = tree.query(to_coords, distance_upper_bound=threshold)

    valid_mask = dists <= threshold
    to_indices = np.where(valid_mask)[0]
    matched_from_indices = from_indices[valid_mask].astype(int)

    return to_indices, matched_from_indices


def _parse_queries(query: Union[str, List[str]]) -> Tuple[str, str]:
    """Parse query input into (from_query, to_query).
    
    Supports:
    - Single string: "buildings" -> ("buildings", "buildings")
    - Comma-separated: "forests,buildings" -> ("forests", "buildings")
    - List of 1-2 strings: ["forests", "buildings"] -> ("forests", "buildings")
    """
    if isinstance(query, list):
        if len(query) == 0:
            raise ValueError("query list cannot be empty")
        elif len(query) == 1:
            return query[0], query[0]
        else:
            return query[0], query[1]
    
    # String input - check for comma separation
    if "," in query:
        parts = [q.strip() for q in query.split(",") if q.strip()]
        if len(parts) >= 2:
            return parts[0], parts[1]
        return parts[0], parts[0]
    
    return query, query


def change(
    query: Union[str, List[str]],
    from_time: str,
    to_time: str,
    mode: str = "new",
    region: Optional[gpd.GeoDataFrame] = None,
    vision_encoder=None,
    vision_year_indices: Optional[Dict[int, Dict[str, "TurboQuantSearchIndex"]]] = None,
    resolution: str = config.DEFAULT_RESOLUTION,
    nprobe: int = config.VISION_NPROBE_DEFAULT,
) -> gpd.GeoDataFrame:
    """Detect changes between *from_time* and *to_time* for *query*.
    
    Args:
        query: 1 or 2 queries (comma-separated or list)
        from_time: start time period ("past", "recent", "present")
        to_time: end time period ("past", "recent", "present")
        mode: "new" to find features that appeared, "removed" to find features that disappeared
    
    For mode="removed", the function internally swaps from_time and to_time
    to reuse the "new" detection logic.
    """
    if vision_year_indices is None:
        print("No year-specific vision indices loaded.")
        return empty_gdf()

    # For "removed" mode, swap time periods to reuse "new" logic
    print(from_time, to_time)
    if mode == "removed":
        from_time, to_time = to_time, from_time

    from_query, to_query = _parse_queries(query)
    
    from_year = config.VISION_YEARS[from_time]
    to_year = config.VISION_YEARS[to_time]

    from_index = vision_year_indices.get(from_year, {}).get(resolution)
    to_index = vision_year_indices.get(to_year, {}).get(resolution)

    if from_index is None:
        print(f"No vision index for from_time='{from_time}' (year {from_year}, resolution '{resolution}').")
        return empty_gdf()
    if to_index is None:
        print(f"No vision index for to_time='{to_time}' (year {to_year}, resolution '{resolution}').")
        return empty_gdf()

    # --- Encode queries and search both time periods ---
    from_query_vector = vision_encoder.encode_text(from_query)
    from_query_np = from_query_vector.squeeze(0).detach().cpu().numpy()
    
    to_query_vector = vision_encoder.encode_text(to_query)
    to_query_np = to_query_vector.squeeze(0).detach().cpu().numpy()

    # --- Early index filtering ---
    from_thresh = None
    to_thresh = 0.2

    print(f"[{resolution}] Searching {from_time} ({from_year}, query='{from_query}') and {to_time} ({to_year}, query='{to_query}') indices in parallel...")
    with ThreadPoolExecutor(max_workers=2) as ex:
        from_future = ex.submit(from_index.search, from_query_np, region=region, nprobe=nprobe, confidence_thresh=from_thresh)
        to_future = ex.submit(to_index.search, to_query_np, region=region, nprobe=nprobe, confidence_thresh=to_thresh)
        from_scores, from_lat, from_lon = from_future.result()
        to_scores, to_lat, to_lon = to_future.result()

    if len(from_scores) == 0 and len(to_scores) == 0:
        print("[change] No results in either time period.")
        return empty_gdf()

    # Convert search results directly into contiguous NumPy arrays
    from_scores = np.asarray(from_scores, dtype=np.float64)
    from_coords = np.column_stack((from_lat, from_lon))

    to_scores = np.asarray(to_scores, dtype=np.float64)
    to_coords = np.column_stack((to_lat, to_lon))

    # --- Match points across time periods ---
    to_idx, from_idx = _nearest_match_coords(from_coords, to_coords, config.CHANGE_DISTANCE_THRESHOLD)

    if len(to_idx) == 0:
        print(f"[{resolution}] No '{mode}' changes detected.")
        return empty_gdf()

    m_to_scores = to_scores[to_idx]
    m_from_scores = from_scores[from_idx]
    minus_scores = m_to_scores - m_from_scores

    # --- Vectorized filtering ---
    # Detect "new" features: low confidence in from_time, high confidence in to_time
    mask = (m_from_scores < 0.18) & (m_to_scores > 0.2) if from_query == to_query else (m_from_scores > 0.2) & (m_to_scores > 0.2)
    res_scores = minus_scores[mask]
    res_time = f"{from_time}->{to_time}"

    if not np.any(mask):
        print(f"[{resolution}] No '{mode}' changes detected.")
        return empty_gdf()

    matched_to_idx = to_idx[mask]
    matched_lons = to_coords[matched_to_idx, 1]
    matched_lats = to_coords[matched_to_idx, 0]

    # Batch create Shapely Point objects via high-speed vectorized constructor
    result_points = gpd.points_from_xy(matched_lons, matched_lats)

    gdf = from_geometries(list(result_points), scores=res_scores.tolist())
    gdf["time"] = res_time
    print(f"[{resolution}] {mode}: {len(gdf)} feature(s) detected.")
    return gdf