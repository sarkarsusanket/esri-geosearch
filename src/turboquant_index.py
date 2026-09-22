"""
Inference-only loader/searcher for a TurboQuant index folder (as produced by
build_turboquant_index.py). No training/building logic lives here — this is
the lightweight half meant to be loaded once per DLPK session and queried
many times with low latency.

Two search modes, matching how operations/vision.py actually calls this:

  - Global search (no region): IVF-routed. Only `nprobe` of `nlist` clusters
    are decompressed and scored — the rest of the 7.7M-row index is never
    touched.
  - Region-filtered search: a fast vectorized bbox pre-filter over the
    (already in-memory) lat/lon arrays, refined with an exact shapely
    `.within()` test on that much-smaller candidate set, then those
    candidate rows are decompressed and scored directly (IVF is skipped —
    the spatial filter already does the narrowing, and skipping IVF here
    avoids the risk of a real match falling outside the probed clusters).

The big per-vector arrays (packed_codes, packed_signs, residual_scale_q,
lat, lon, sorted_to_original) are memory-mapped rather than eagerly loaded,
so opening the index is fast and only the rows actually touched by a given
query get paged in.
"""

import json
import os
from typing import Optional, Tuple

import numpy as np
import torch
import geopandas as gpd

import config


def unpack_bits_rows(packed_rows: np.ndarray, num_bits: int, d: int) -> torch.Tensor:
    """Inverse of build_turboquant_index.py's pack_bits_rows, for an arbitrary
    (possibly non-contiguous) subset of rows. Returns (n_rows, D) int64."""
    n_rows = packed_rows.shape[0]
    bits = np.unpackbits(packed_rows, axis=1)[:, : d * num_bits].reshape(n_rows, d, num_bits)
    values = np.zeros((n_rows, d), dtype=np.uint32)
    for i in range(num_bits):
        values |= bits[:, :, num_bits - 1 - i].astype(np.uint32) << i
    return torch.from_numpy(values.astype(np.int64))


class TurboQuantSearchIndex:
    """Load a single index folder (e.g. embeddings/highres-vision/) and serve
    both global and spatially-filtered nearest-neighbor search."""

    def __init__(self, folder: str):
        self.folder = folder
        with open(os.path.join(folder, "meta.json")) as f:
            meta = json.load(f)
        self.dim = meta["dim"]
        self.num_bits = meta["num_bits"]
        self.clip_val = meta["clip_val"]
        self.scale = meta["scale"]
        self.residual_scale_max = meta["residual_scale_max"]
        self.nlist = meta["nlist"]
        self.num_vectors = meta["num_vectors"]

        self.rotation_matrix = torch.from_numpy(
            np.load(os.path.join(folder, "rotation_matrix.npy"))
        ).float().to(config.DEVICE)
        self.centroids = torch.from_numpy(
            np.load(os.path.join(folder, "centroids.npy"))
        ).float().to(config.DEVICE)
        self.cluster_offsets = np.load(os.path.join(folder, "cluster_offsets.npy"))

        # Memory-mapped: reading a subset of rows only pages in that subset,
        # not the whole 7.7M-row array.
        self.packed_codes = np.load(os.path.join(folder, "packed_codes.npy"), mmap_mode="r")
        self.packed_signs = np.load(os.path.join(folder, "packed_signs.npy"), mmap_mode="r")
        self.residual_scale_q = np.load(os.path.join(folder, "residual_scale_q.npy"), mmap_mode="r")
        self.lat = np.load(os.path.join(folder, "lat.npy"), mmap_mode="r")
        self.lon = np.load(os.path.join(folder, "lon.npy"), mmap_mode="r")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _rotate(self, query_vec: np.ndarray) -> torch.Tensor:
        q = torch.from_numpy(np.asarray(query_vec, dtype=np.float32)).view(1, -1)
        return q @ self.rotation_matrix  # (1, dim)

    def _score_rows(self, storage_rows: np.ndarray, rotated_query: torch.Tensor) -> np.ndarray:
        """Decompress ONLY these rows and score them. No full-index reconstruction."""
        if len(storage_rows) == 0:
            return np.empty(0, dtype=np.float32)
        codes_packed = np.ascontiguousarray(self.packed_codes[storage_rows])
        signs_packed = np.ascontiguousarray(self.packed_signs[storage_rows])
        rs_q = np.ascontiguousarray(self.residual_scale_q[storage_rows])

        codes = unpack_bits_rows(codes_packed, self.num_bits, self.dim).float().to(config.DEVICE)
        signs_pm1 = unpack_bits_rows(signs_packed, 1, self.dim).float().to(config.DEVICE) * 2 - 1
        residual_scale = (torch.from_numpy(rs_q.astype(np.float32)) / 255.0 * self.residual_scale_max).to(config.DEVICE)

        q = rotated_query.to(config.DEVICE).view(1, -1)
        stage1 = self.scale * (codes @ q.T).squeeze(1) - self.clip_val * q.sum()
        stage2 = residual_scale * (signs_pm1 @ q.T).squeeze(1)
        return (stage1 + stage2).cpu().numpy()

    def _spatial_candidates(self, region: gpd.GeoDataFrame) -> np.ndarray:
        """Bbox pre-filter (fast, vectorized) + exact polygon refine (on the
        much smaller candidate set) -> storage row indices."""
        minx, miny, maxx, maxy = region.total_bounds
        lon_all = np.asarray(self.lon)
        lat_all = np.asarray(self.lat)
        bbox_mask = (lon_all >= minx) & (lon_all <= maxx) & (lat_all >= miny) & (lat_all <= maxy)
        candidate_rows = np.where(bbox_mask)[0]
        print(f"[DEBUG] region_bounds={minx, miny, maxx, maxy}, "
            f"index_lat=[{lat_all.min():.4f}, {lat_all.max():.4f}], "
            f"index_lon=[{lon_all.min():.4f}, {lon_all.max():.4f}]")
        if len(candidate_rows) == 0:
            return candidate_rows

        pts = gpd.GeoDataFrame(
            {"row": candidate_rows},
            geometry=gpd.points_from_xy(lon_all[candidate_rows], lat_all[candidate_rows]),
            crs="EPSG:4326",
        )
        region_ll = region.to_crs("EPSG:4326") if region.crs is not None else region.set_crs("EPSG:4326")
        # Make all geometries valid before union
        valid_geometries = region_ll.geometry.make_valid()
        region_union = valid_geometries.unary_union
        exact_mask = pts.within(region_union).values
        return candidate_rows[exact_mask]

    # ------------------------------------------------------------------
    # Public search API
    # ------------------------------------------------------------------
    def search(
        self,
        query_vec: np.ndarray,
        top_k: Optional[int] = None,
        confidence_thresh: Optional[float] = None,
        region: Optional[gpd.GeoDataFrame] = None,
        nprobe: int = 24,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (scores, lat, lon) for the top_k matches, best-first."""
        rotated_query = self._rotate(query_vec)

        if region is not None and not region.empty:
            candidate_rows = self._spatial_candidates(region)
            if len(candidate_rows) == 0:
                return np.empty(0), np.empty(0), np.empty(0)
            raw_scores = self._score_rows(candidate_rows, rotated_query)
            rows_pool = candidate_rows
        else:
            centroid_sims = (rotated_query @ self.centroids.T).squeeze(0).cpu().numpy()
            nprobe = min(nprobe, self.nlist)
            probe_clusters = np.argpartition(-centroid_sims, nprobe - 1)[:nprobe]

            all_rows = []
            for c in probe_clusters:
                lo, hi = int(self.cluster_offsets[c]), int(self.cluster_offsets[c + 1])
                if hi == lo:
                    continue
                all_rows.append(np.arange(lo, hi))
            if not all_rows:
                return np.empty(0), np.empty(0), np.empty(0)
            rows_pool = np.concatenate(all_rows)
            raw_scores = self._score_rows(rows_pool, rotated_query)

        if top_k: 
            k = min(top_k, len(raw_scores))
            top_idx = np.argpartition(-raw_scores, k - 1)[:k]
        elif confidence_thresh: 
            top_idx = np.where(raw_scores >= confidence_thresh)[0]
        else: 
            k = len(raw_scores)
            top_idx = np.argpartition(-raw_scores, k - 1)[:k]
        
        top_idx = top_idx[np.argsort(-raw_scores[top_idx])]  # best-first
        chosen_rows = rows_pool[top_idx]

        scores = raw_scores[top_idx]
        lat = np.asarray(self.lat[chosen_rows])
        lon = np.asarray(self.lon[chosen_rows])
        return scores, lat, lon
