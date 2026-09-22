"""
Geocoding: resolve a place name to its administrative boundary polygon
(falling back to a bounding box), via Nominatim with an Esri fallback.

Results are cached in-process by normalized place name (functools.lru_cache
on _geocode_cached): geocode() sits on the critical path of nearly every
query, place names repeat constantly both across queries in a session and
across independent branches of the same multi-filter plan, and a repeat
lookup has no reason to pay for another network round trip.
"""
import functools

import geopandas as gpd
import requests
from shapely import wkb
from shapely.geometry import Polygon, shape
from schema import empty_gdf, from_geometries


def geocode(target: str) -> gpd.GeoDataFrame:
    """Resolve a place name to its EXACT administrative boundary polygon.

    Falls back to a bounding box if exact polygon geometry is unavailable.

    Thin wrapper around the cached implementation below: GeoDataFrames
    aren't safe to cache directly (they're mutable, and callers may mutate
    results downstream), so the cache stores WKB bytes — immutable and
    hashable — and this function rebuilds a fresh GeoDataFrame from those
    bytes on every call.
    """
    if not target:
        return empty_gdf()

    wkb_bytes = _geocode_cached(target.strip().lower())
    if wkb_bytes is None:
        return empty_gdf()
    return from_geometries([wkb.loads(wkb_bytes)])


@functools.lru_cache(maxsize=256)
def _geocode_cached(cache_key: str):
    """Does the actual network lookup, cached by normalized place name.
    Returns WKB bytes, or None if nothing was found."""
    poly = _geocode_uncached(cache_key)
    return poly.wkb if poly is not None else None


def _geocode_uncached(target: str):
    """Runs the real Nominatim (then Esri fallback) network lookup and
    returns a shapely geometry, or None if both fail. Never called
    directly — always go through geocode()/_geocode_cached() above."""
    headers = {
        "User-Agent": "QueryEarthPipeline/1.0 (contact: admin@queryearth.local)"
    }

    # 1. Try Nominatim search with polygon_geojson enabled
    nominatim_url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": target,
        "format": "json",
        "polygon_geojson": 1,
        "limit": 1,
    }

    try:
        res = requests.get(
            nominatim_url, params=params, headers=headers, timeout=10
        )

        if res.status_code == 200:
            data = res.json()
            if data and len(data) > 0:
                item = data[0]
                geojson_geom = item.get("geojson")

                # Ensure geometry is a valid Polygon or MultiPolygon
                if geojson_geom and geojson_geom.get("type") in [
                    "Polygon",
                    "MultiPolygon",
                ]:
                    return shape(geojson_geom)

                # If no polygon geometry, fallback to bounding box from Nominatim
                if "boundingbox" in item:
                    # Nominatim returns [south, north, west, east]
                    s, n, w, e = map(float, item["boundingbox"])
                    return Polygon([(w, s), (e, s), (e, n), (w, n), (w, s)])

    except Exception as e:
        print(f"Nominatim polygon lookup failed for '{target}': {e}")

    # 2. Fallback to Esri Geocoding Bounding Box if Nominatim fails
    try:
        esri_url = "https://geocode.arcgis.com/arcgis/rest/services/World/GeocodeServer/findAddressCandidates"
        esri_params = {
            "SingleLine": target,
            "f": "json",
            "maxLocations": 1,
            "outFields": "extent",
        }
        res = requests.get(esri_url, params=esri_params, timeout=10)
        if res.status_code == 200:
            data = res.json()
            candidates = data.get("candidates", [])
            if candidates and "extent" in candidates[0]:
                ext = candidates[0]["extent"]
                return Polygon(
                    [
                        (ext["xmin"], ext["ymin"]),
                        (ext["xmax"], ext["ymin"]),
                        (ext["xmax"], ext["ymax"]),
                        (ext["xmin"], ext["ymax"]),
                        (ext["xmin"], ext["ymin"]),
                    ]
                )
    except Exception as e:
        print(f"Esri bbox fallback failed for '{target}': {e}")

    return None