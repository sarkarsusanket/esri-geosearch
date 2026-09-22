"""
Shared GeoDataFrame conventions.

Every pipeline step (geocode / demo / vision / tool) returns a GeoDataFrame
that always has a `geometry` column and, where applicable, a `score`
column. Standardizing on this shape means tool operations (buffer, union,
intersection, difference) can operate on the output of *any* step without
needing to know which operation produced it.
"""
import geopandas as gpd
import config

GEOMETRY_COL = "geometry"
SCORE_COL = "score"
CRS = "EPSG:4326"


def empty_gdf() -> gpd.GeoDataFrame:
    """Return a well-formed, empty GeoDataFrame matching the pipeline schema."""
    return gpd.GeoDataFrame(columns=[GEOMETRY_COL, SCORE_COL], geometry=GEOMETRY_COL, crs=CRS)


def from_geometries(geometries, scores=None) -> gpd.GeoDataFrame:
    """Build a standard-schema GeoDataFrame from a list of geometries (+ optional scores)."""
    data = {GEOMETRY_COL: geometries}
    if scores is not None:
        data[SCORE_COL] = scores
    return gpd.GeoDataFrame(data, geometry=GEOMETRY_COL, crs=CRS)


def ensure_crs(gdf: gpd.GeoDataFrame, crs: str = CRS) -> gpd.GeoDataFrame:
    """Make sure a GeoDataFrame is tagged with the pipeline's working CRS."""
    if gdf.crs is None:
        return gdf.set_crs(crs)
    if str(gdf.crs) != crs:
        return gdf.to_crs(crs)
    return gdf


def buffer_points_if_needed(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """If any geometries in gdf are Points, buffer them by config.AUTOMATIC_BUFFER_RADIUS kilometers."""
    if gdf is None or gdf.empty:
        return gdf

    gdf = ensure_crs(gdf)

    # Check if any geometries are points
    point_mask = gdf.geometry.type.isin(["Point", "MultiPoint"])
    if not point_mask.any():
        return gdf

    # If all geometries are points, buffer them all
    # If mixed, only buffer the points
    metric = gdf.to_crs(gdf.estimate_utm_crs())
    buffered = metric.copy()
    buffered.loc[point_mask, GEOMETRY_COL] = metric.loc[point_mask].geometry.buffer(config.AUTOMATIC_BUFFER_RADIUS * 1000)
    return ensure_crs(buffered.to_crs(CRS))
