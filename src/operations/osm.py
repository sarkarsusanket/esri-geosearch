"""OpenStreetMap (OSM) search module.

Provides unified OSM keyword search with spatial filtering and fuzzy term
normalization (plural/stem handling).
"""

import os
import re
from typing import Dict, List, Optional, Tuple

import shapely
import geopandas as gpd
import pandas as pd

import config
from operations.tool import shapely_overlay
from schema import GEOMETRY_COL, SCORE_COL, empty_gdf, ensure_crs

# Maps mode -> (parquet filename, primary category column, has_name_col)
MODE_REGISTRY = {
    "roads": ("roads.parquet", "highway", False),
    "waterways": ("waterway.parquet", "waterway", False),
    "landuse": ("landuse.parquet", "landuse", False),
    "pois": ("pois.parquet", "amenity", True),
}

SUPPORTED_MODES = set(MODE_REGISTRY.keys())


def _category_cols_for(mode: str, gdf: gpd.GeoDataFrame) -> List[str]:
    """Primary category column for `mode`, plus any secondary POI tag
    columns (leisure, shop, tourism, ...) that are actually present in the
    loaded data. Roads/waterways/landuse always get just their one fixed
    column back.
    """
    _, primary_col, _ = MODE_REGISTRY[mode]
    cols = [primary_col]
    if mode == "pois":
        cols.extend(c for c in POI_SECONDARY_CATEGORY_COLS if c in gdf.columns)
    return cols


# "pois" mode only ever checked `amenity`, but plenty of common POI queries
# are tagged under a *different* OSM key entirely — parks are `leisure=park`,
# not `amenity=park`. Searching only `amenity` meant those queries found no
# category match and fell through to the much less reliable name search
# (which is how "park" ended up matching "Parking Lot" by name). If your
# pois.parquet has any of these columns, they get checked too. Confirm which
# of these actually exist in your data with `inspect_unique_categories`.
POI_SECONDARY_CATEGORY_COLS = ("leisure", "shop", "tourism", "office", "craft")


def _stem(word: str) -> str:
    """Basic English lemmatizer/stemmer to strip common plurals.

    Handles cases like 'rivers' -> 'river', 'beaches' -> 'beach', 'cities' ->
    'city'.
    """
    if len(word) <= 3:
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith(("shes", "ches", "sses", "boxes", "faxes")):
        return word[:-2]
    # Guard against words that end in "-us" (bus, campus, focus, status) or
    # "-ss" (glass) being treated as plurals. Only strip a bare trailing "s"
    # when it's genuinely a plural marker.
    if word.endswith("s") and not word.endswith(("ss", "us")):
        return word[:-1]
    return word


def _normalize(query: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    return re.sub(r"[^a-z0-9]+", " ", query.lower()).strip()


def _tokenize(query: str) -> List[str]:
    return [_normalize(token) for token in _normalize(query).split() if token]


def _resolve_mode_and_query(
    arg1: Optional[str],
    arg2: Optional[str],
) -> Tuple[str, Optional[str]]:
    """Determine which argument is the mode and which is the query."""
    if arg1 is None and arg2 is None:
        return "roads", None

    if arg1 is None:
        if arg2 and arg2.lower() in SUPPORTED_MODES:
            return arg2.lower(), None
        return "roads", arg2

    if arg2 is None:
        if arg1.lower() in SUPPORTED_MODES:
            return arg1.lower(), None
        return "roads", arg1

    arg1_lower = arg1.lower()
    arg2_lower = arg2.lower()

    arg1_is_mode = arg1_lower in SUPPORTED_MODES
    arg2_is_mode = arg2_lower in SUPPORTED_MODES

    if arg1_is_mode and not arg2_is_mode:
        return arg1_lower, arg2
    elif arg2_is_mode and not arg1_is_mode:
        return arg2_lower, arg1
    elif arg1_is_mode and arg2_is_mode:
        return arg1_lower, arg2
    else:
        return arg2_lower, arg1


# Conservative, high-confidence synonyms for the "pois" (amenity) mode only.
# These map common phrasing to the *actual* Geofabrik/OSM amenity tag before
# we do the strict exact-match lookup below. Deliberately small: a wrong
# synonym here silently returns the wrong category, so only add pairs you've
# confirmed exist as real amenity values in your data (see
# `inspect_unique_categories` below to check). NOT used for roads/waterways/
# landuse, since those already use the fixed vocab from MODE_REGISTRY.
POI_SYNONYMS = {
    "gas station": "fuel",
    "gas stations": "fuel",
    "petrol station": "fuel",
    "petrol stations": "fuel",
    "grocery": "supermarket",
    "grocery store": "supermarket",
    "atm": "atm",
    "cash machine": "atm",
    "cash machines": "atm",
    "drugstore": "pharmacy",
    "drug store": "pharmacy",
}


def inspect_unique_categories(
    osm_data: Dict[str, gpd.GeoDataFrame],
) -> Dict[str, List[str]]:
    """Print + return the actual unique category values present in each
    loaded mode's data. Use this against your real Geofabrik parquet files
    to build accurate alias/synonym tables instead of guessing — amenity
    values in particular vary by extract/vendor.
    """
    out = {}
    for mode, (_, category_col, _) in MODE_REGISTRY.items():
        gdf = osm_data.get(mode)
        if gdf is None or gdf.empty or category_col not in gdf.columns:
            out[mode] = []
            continue
        vals = sorted(gdf[category_col].dropna().astype(str).unique().tolist())
        out[mode] = vals
        print(f"[{mode}] {category_col} ({len(vals)} unique): {vals[:50]}")
    return out


def _category_hits(query: str, gdf: gpd.GeoDataFrame, category_col: str) -> List[str]:
    """Find category tags that match the query using exact compound term/subtag comparison.

    Handles `;` multi-value tags and `_`-separated class names.
    """
    if category_col not in gdf.columns:
        return []

    # Sub-queries split by comma, normalized (e.g. "car_rental" -> "car rental")
    sub_queries = [q.strip() for q in query.split(",") if q.strip()]
    
    # Generate query target terms (both raw normalized and stemmed)
    target_terms = set()
    for sq in sub_queries:
        norm_sq = _normalize(sq)
        target_terms.add(norm_sq)
        # Apply stem to individual words in the query phrase (e.g. "car rentals" -> "car rental")
        stemmed_sq = " ".join([_stem(w) for w in norm_sq.split()])
        target_terms.add(stemmed_sq)

    hits = []
    unique_categories = gdf[category_col].dropna().astype(str).unique()

    for tag in unique_categories:
        # Split multi-value tags like "bench;bicycle_parking" into individual category values
        subtags = [s.strip() for s in tag.split(";") if s.strip()]
        
        for subtag in subtags:
            # Normalize tag ("car_rental" -> "car rental")
            norm_subtag = _normalize(subtag)
            stemmed_subtag = " ".join([_stem(w) for w in norm_subtag.split()])

            # Exact match against the full tag phrase or stemmed tag phrase
            if norm_subtag in target_terms or stemmed_subtag in target_terms:
                hits.append(tag)
                break

    return hits


def _name_hits(query: str, gdf: gpd.GeoDataFrame) -> List[int]:
    """Vectorized row matching where name contains ALL query tokens (AND).

    Previously this matched if the name contained ANY token, so a
    two-word query like "bus parking" matched every row whose name
    merely contained "parking" (or even "bus"), regardless of the
    other word. Multi-token queries are almost always meant as a
    single compound phrase intent, not a loose OR of keywords, so we
    require every token (or its stem) to be found before a row counts
    as a hit.
    """
    qtokens = _tokenize(query)
    if not qtokens or "name" not in gdf.columns:
        return []

    name_series = gdf["name"].fillna("").astype(str).str.lower()
    mask = pd.Series(True, index=gdf.index)

    for token in qtokens:
        variants = {token}
        if len(token) > 3:
            variants.add(_stem(token))
        # \b word-boundary is required here: without it, "park" as a plain
        # substring matches inside "Parking Lot", "Parking Garage", etc.
        # This is exactly why "find me parks" was returning parking lots.
        pattern = "|".join(rf"\b{re.escape(v)}\b" for v in variants)
        mask &= name_series.str.contains(pattern, case=False, regex=True, na=False)
        if not mask.any():
            return []

    return gdf.index[mask].tolist()


def _keyword_search(
    query: str,
    gdf: gpd.GeoDataFrame,
    category_cols: List[str],
    has_name: bool,
) -> Tuple[Optional[gpd.GeoDataFrame], List[str]]:
    """Keyword search across one or more category columns plus name field.

    category_cols is a list because a single "mode" (especially "pois") can
    map to several distinct OSM tag keys — e.g. parks are `leisure=park`,
    not `amenity=park`. Checking multiple columns means a query like "park"
    gets a real category match instead of falling through to the much
    looser name search.
    """
    sub_queries = [q.strip() for q in query.split(",") if q.strip()]
    if query == "conservation": sub_queries = ['protected area', 'forest', 'conservation']

    all_cat_hits: Dict[str, List[str]] = {col: [] for col in category_cols}
    all_name_indices = []

    for sq in sub_queries:
        norm = _normalize(sq)

        for col in category_cols:
            # Amenity tags are open-vocabulary, so a small, conservative
            # synonym table can resolve common phrasing ("gas station" ->
            # "fuel") to the real tag before we do the strict exact-match
            # lookup. Only applies to the amenity column.
            cat_query = POI_SYNONYMS.get(norm, norm) if col == "amenity" else norm
            all_cat_hits[col].extend(_category_hits(cat_query, gdf, col))

        if has_name:
            all_name_indices.extend(_name_hits(norm, gdf))

    for col in all_cat_hits:
        all_cat_hits[col] = list(dict.fromkeys(all_cat_hits[col]))
    all_name_indices = list(dict.fromkeys(all_name_indices))

    cat_mask = pd.Series(False, index=gdf.index)
    matched_pairs: List[str] = []
    for col, hits in all_cat_hits.items():
        if hits and col in gdf.columns:
            cat_mask |= gdf[col].isin(hits)
            matched_pairs.extend(f"{col}={v}" for v in hits)

    name_mask = (
        gdf.index.isin(all_name_indices)
        if all_name_indices
        else pd.Series(False, index=gdf.index)
    )

    combined_mask = cat_mask | name_mask

    if combined_mask.any():
        return gdf[combined_mask].copy(), matched_pairs

    return None, []


def _trim(
    result: gpd.GeoDataFrame,
    region: Optional[gpd.GeoDataFrame],
    score: float,
    extra_cols: Optional[List[str]] = None,
) -> Optional[gpd.GeoDataFrame]:
    """Clips result geometries to their exact spatial intersection with region."""
    if result.empty:
        return None

    if region is not None and not region.empty:
        region_clean = ensure_crs(region)
        result = ensure_crs(result)

        # Overlay intersection trims/clips geometries and joins region columns
        result = shapely_overlay(
            result,
            region_clean,
            how="intersection",
        )

    result = result.copy()
    result[SCORE_COL] = float(score)

    keep_cols = [GEOMETRY_COL, SCORE_COL]

    # Preserve region attributes alongside extra_cols
    if region is not None and not region.empty:
        region_cols = [c for c in region.columns if c != region._geometry_column_name]
        keep_cols.extend(c for c in region_cols if c in result.columns and c not in keep_cols)

    if extra_cols:
        keep_cols.extend(c for c in extra_cols if c in result.columns and c not in keep_cols)

    return ensure_crs(result[keep_cols])

def load_osm_data(
    osm_dir: str = config.OSM_EMBEDDING_DIR,
    year: str = "latest",
) -> Dict[str, gpd.GeoDataFrame]:
    """Load all OSM parquet files for a given year into a dict keyed by mode."""
    year_dir = os.path.join(osm_dir, year)
    if not os.path.isdir(year_dir):
        print(f"OSM year directory not found: {year_dir}")
        return {}

    data = {}
    for mode, (filename, _, _) in MODE_REGISTRY.items():
        path = os.path.join(year_dir, filename)
        if os.path.isfile(path):
            print(f"Loading OSM {mode} from {path}...")
            data[mode] = gpd.read_parquet(path)
        else:
            print(f"OSM file not found for mode '{mode}': {path}")
    return data


def search_osm(
    mode: str,
    query: Optional[str],
    region: Optional[gpd.GeoDataFrame],
    osm_data: Dict[str, gpd.GeoDataFrame],
) -> gpd.GeoDataFrame:
    """Search OSM features by mode, term, and bounding region."""
    if mode not in SUPPORTED_MODES:
        print(
            f"OSM search: unsupported mode '{mode}'. Directing to POI search."
        )
        mode = "pois"

    if mode not in osm_data or osm_data[mode] is None or osm_data[mode].empty:
        print(f"OSM search: no data loaded for mode '{mode}'.")
        return empty_gdf()

    gdf = osm_data[mode]
    filename, category_col, has_name = MODE_REGISTRY[mode]
    category_cols = _category_cols_for(mode, gdf)

    extra = list(category_cols)
    if has_name:
        extra.append("name")

    if not query:
        res = _trim(gdf, region, score=1.0, extra_cols=extra)
        return res if res is not None else empty_gdf()

    # Spatially pre-filter to the region BEFORE keyword matching, not after.
    # `_name_hits` runs a regex `.str.contains()` over every row's `name`
    # field — for a country/global-scale "buildings" or "pois" table, that
    # scan used to run on the FULL table even when the query only cares
    # about one small region, with `_trim` only clipping to the region
    # afterward. sindex.query() is a fast candidate lookup, so this shrinks
    # the table the regex has to scan before doing any string work, rather
    # than doing the string work first and throwing most of it away.
    search_gdf = gdf
    if region is not None and not region.empty:
        region_union = region.geometry.unary_union
        cand_idx = gdf.sindex.query(region_union, predicate="intersects")
        if len(cand_idx) > 0:
            search_gdf = gdf.iloc[cand_idx]
        else:
            search_gdf = gdf.iloc[[]]

    cand, cat_hits = _keyword_search(query, search_gdf, category_cols, has_name)
    if cand is not None:
        res = _trim(cand, region, score=1.0, extra_cols=extra)
        if res is not None:
            if cat_hits:
                print(
                    f"OSM [{mode}] query {query!r} matched category(es): {cat_hits[:10]}"
                )
            else:
                print(f"OSM [{mode}] query {query!r} matched by name.")
            return res

    # Fallback: if no matches in the requested mode, search POIs as well
    if mode != "pois" and "pois" in osm_data and osm_data["pois"] is not None and not osm_data["pois"].empty:
        print(f"OSM [{mode}] no keyword matches for {query!r}, falling back to POI search...")
        pois_gdf = osm_data["pois"]
        pois_search_gdf = pois_gdf
        if region is not None and not region.empty:
            region_union = region.geometry.unary_union
            pois_cand_idx = pois_gdf.sindex.query(region_union, predicate="intersects")
            pois_search_gdf = pois_gdf.iloc[pois_cand_idx] if len(pois_cand_idx) > 0 else pois_gdf.iloc[[]]
        pois_has_name = MODE_REGISTRY["pois"][2]
        pois_cat_cols = _category_cols_for("pois", pois_gdf)
        pois_cand, pois_cat_hits = _keyword_search(query, pois_search_gdf, pois_cat_cols, pois_has_name)
        if pois_cand is not None:
            pois_extra = list(pois_cat_cols)
            if pois_has_name:
                pois_extra.append("name")
            res = _trim(pois_cand, region, score=1.0, extra_cols=pois_extra)
            if res is not None:
                if pois_cat_hits:
                    print(f"OSM [pois] fallback query {query!r} matched category(es): {pois_cat_hits[:10]}")
                else:
                    print(f"OSM [pois] fallback query {query!r} matched by name.")
                return res

    print(f"OSM [{mode}] no keyword matches for {query!r}")
    return empty_gdf()