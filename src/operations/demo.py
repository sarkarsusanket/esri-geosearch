"""Demographic similarity search against the TabularTextCLIP embedding space."""
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import geopandas as gpd

import config
from models import LocalTextEmbedder
from schema import SCORE_COL, ensure_crs


def search_demographics(target: Optional[str],
                         region: Optional[gpd.GeoDataFrame],
                         demo_gdf: gpd.GeoDataFrame,
                         ae_embeddings: torch.Tensor,
                         clip_model: torch.nn.Module,
                         text_embedder: torch.nn.Module,
                        ) -> gpd.GeoDataFrame:
    """Rank demographic polygons by similarity to a free-text query, optionally
    restricted to a prior region.

    If `target` is None/empty, the entire (optionally region-restricted)
    demographic layer is returned unranked.
    """
    candidates = demo_gdf
    candidate_embeddings = ae_embeddings

    if region is not None and not region.empty:
        # Ensure region contains valid Polygon / MultiPolygon geometries before taking unary_union
        polygons = region.geometry[region.geometry.type.isin(["Polygon", "MultiPolygon"])]
        if not polygons.empty:
            region_union = polygons.unary_union
        else:
            # Fallback to buffer(0) or original unary_union if type filtering isn't enough
            region_union = region.geometry.unary_union.buffer(0)

        # Convert GeometryCollection to MultiPolygon if unary_union produced mixed types
        if region_union.geom_type == "GeometryCollection":
            from shapely.geometry import MultiPolygon
            poly_list = [g for g in region_union.geoms if g.geom_type in ["Polygon", "MultiPolygon"]]
            region_union = MultiPolygon(poly_list) if poly_list else region_union

        # Spatial-index-accelerated candidate lookup
        mask_indices = demo_gdf.sindex.query(region_union, predicate="intersects")
        if len(mask_indices) > 0:
            masked_gdf = demo_gdf.iloc[mask_indices].copy().reset_index(drop=True)
            masked_embeddings = ae_embeddings[mask_indices]
            
            # Perform clip on the filtered region_union
            candidates = gpd.clip(masked_gdf, region_union)
            candidate_embeddings = masked_embeddings[candidates.index.to_numpy()]
        else:
            print("Demographic layer does not intersect the given region; searching globally instead.")
    if not target:
        result = candidates.copy()
        if SCORE_COL not in result.columns:
            result[SCORE_COL] = 1.0
        return ensure_crs(result)

    text_embedding = text_embedder.encode(target)
    text_tensor = torch.from_numpy(np.array([text_embedding], dtype=np.float32)).to(config.DEVICE)

    with torch.no_grad():
        tabular_latents = F.normalize(clip_model.geo_projector(candidate_embeddings), p=2, dim=-1).float()
        text_latent = F.normalize(clip_model.text_projector(text_tensor), p=2, dim=-1).float()
        scores = torch.matmul(tabular_latents, text_latent.t()).squeeze(-1).cpu().numpy()

    scores = _min_max_normalize(scores)

    above_threshold_indices = np.where(scores > 0.90)[0]
    sorted_above_indices = above_threshold_indices[np.argsort(scores[above_threshold_indices])[::-1]]

    matched = candidates.iloc[sorted_above_indices].copy()
    matched[SCORE_COL] = scores[sorted_above_indices]
    return ensure_crs(matched)


def _min_max_normalize(scores: np.ndarray) -> np.ndarray:
    s_min, s_max = scores.min(), scores.max()
    if s_max - s_min > 0:
        return (scores - s_min) / (s_max - s_min)
    return np.ones_like(scores)