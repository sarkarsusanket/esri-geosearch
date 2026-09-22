"""
Display-artifact classification and result explanation.

Turns an already-executed QueryPlan into:

  1. a small set of DisplayArtifact objects — one per pipeline step that is
     an ancestor of the final result — each tagged with a `role`
     ("final" / "context" / "intermediate") and a `default_visible` flag,
     so the UI knows what to show in the global "context layers" list and
     what to hide by default.

  2. a per-feature "why this result?" explanation for every row of the
     final GeoDataFrame, listing which context layers it satisfies and,
     for distance-based filters, which specific named feature it's
     closest to (e.g. "UCLA Hospital — 2.3 mi").

Nothing here calls the router LLM or asks it to decide what to show.
Every decision is derived purely from the DAG shape (PipelineStep.operation
/ .inputs / .parameters) that the executor already walks, plus the
GeoDataFrames the executor already computed. See `classify()`'s docstring
for the exact rules and their rationale.
"""
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import shapely
from shapely.geometry import mapping

from query_parser import QueryPlan, PipelineStep
from schema import ensure_crs

# Step operations that pull directly from a named target (as opposed to
# "tool" steps, which just combine/reshape prior outputs).
LEAF_OPS = {"geocode", "demo", "vision", "osm", "change"}

# "tool" actions that purely combine two existing layers (set algebra).
# These never get their own chip in the UI — there's no single clean name
# for "the AND of four other things" — they're the glue, not a concept.
GLUE_TOOL_ACTIONS = {"intersection", "union", "difference", "add", "get_centroid"}

KM_PER_MILE = 1.60934

ROLE_FINAL = "final"
ROLE_CONTEXT = "context"
ROLE_INTERMEDIATE = "intermediate"


@dataclass
class DisplayArtifact:
    id: str                        # == step.output_variable, stable within a plan
    label: str                     # human-friendly name, built from the step's own parameters
    geometry: gpd.GeoDataFrame     # the full result for this step (kept, not just a shape,
                                    # so per-feature attributes like `name` survive for explanations)
    source_node: str               # the producing PipelineStep's step_id, as a string
    type: str                      # "geocode" | "demo" | "vision" | "osm" | "change" | "tool:<action>"
    role: str                      # ROLE_FINAL | ROLE_CONTEXT | ROLE_INTERMEDIATE
    default_visible: bool
    filter_kind: Optional[str] = None       # "region" | "distance" | None
    distance_miles: Optional[float] = None  # only set when filter_kind == "distance"
    concept_label: Optional[str] = None     # e.g. "Hospitals", for phrasing "within 5 mi of {concept_label}"
    explains_source: Optional[str] = None   # id of the raw (pre-buffer) artifact, used to name the
                                             # nearest specific feature in a distance explanation


def _label_for(step: PipelineStep, artifacts_so_far: Dict[str, DisplayArtifact]) -> str:
    """Build a human-readable label for a step, purely from its own
    parameters (+ the already-built label of its input, for buffer)."""
    params = step.parameters

    if step.operation == "geocode":
        return params.get("target") or "Region"

    if step.operation in ("demo", "vision", "osm"):
        target = params.get("target") or "results"
        return target[:1].upper() + target[1:]

    if step.operation == "change":
        target = params.get("target") or "features"
        from_time = params.get("from_time", "")
        to_time = params.get("to_time", "")
        mode = params.get("mode", "new")
        verb = "New" if mode == "new" else "Removed"
        return f"{verb} {target} ({from_time} to {to_time})"

    if step.operation == "tool" and params.get("target") == "buffer":
        dist_km = params.get("buffer_distance_km") or 0
        miles = round(dist_km / KM_PER_MILE, 1)
        src_var = step.inputs[0] if step.inputs else None
        concept = artifacts_so_far[src_var].label if src_var in artifacts_so_far else "area"
        return f"{miles:g} mi radius around {concept}"

    return step.output_variable


def _ancestors_of(final_var: str, steps_by_var: Dict[str, PipelineStep]) -> set:
    """Backward reachability from the final variable, so steps that were
    computed but never actually feed the result don't get a chip."""
    seen = {final_var}
    frontier = [final_var]
    while frontier:
        var = frontier.pop()
        step = steps_by_var.get(var)
        if step is None:
            continue
        for inp in step.inputs:
            if inp not in seen:
                seen.add(inp)
                frontier.append(inp)
    return seen


def classify(plan: QueryPlan, variables: Dict[str, gpd.GeoDataFrame]) -> Dict[str, DisplayArtifact]:
    """Tag every ancestor step of the plan's final output with a display role.

    Rules, and why:

    - The terminal step -> "final". Always shown; this *is* the result list.

    - Any leaf search step (geocode/demo/vision/osm/change) -> "context",
      shown by default. Each one represents a single named concept the
      user asked about ("Los Angeles", "wildfire-prone areas",
      "hospitals") — regardless of *how* it's later used (as a region
      scope passed straight into another search, or as one side of an
      intersection/difference), it's a standalone, nameable layer.

      Exception: if a leaf has exactly the same (operation, target) as the
      final step, it's almost certainly the *same* entity search re-used
      pre-filter (e.g. an unfiltered "houses" search that later gets
      intersected down to the final "houses") — showing it as a separate
      chip next to the final result list would just be a second, larger
      copy of the same thing. That gets demoted to "intermediate".

    - A unary buffer step -> "context", shown by default. "5 mi radius
      around hospitals" is exactly the kind of derived-but-nameable layer
      a user wants to see and toggle, same as a raw search result.

    - Any binary set op (intersection/union/difference/add) or
      get_centroid -> "intermediate", hidden by default. These are pure
      combination glue — there's no single good name for "the AND of
      three other layers" — but they still run and their output is kept
      available (role is still tracked) in case a future UI wants a
      debug view.

    - A leaf whose *only* consumer is a single downstream buffer step
      gets hidden by default too (`default_visible=False`) so the UI
      doesn't show both "Hospitals" and "5 mi radius around Hospitals" —
      but it keeps role="context" and is linked via `explains_source` so
      per-feature explanations can still name the specific nearest
      hospital.

    This is a structural heuristic, not a semantic understanding of the
    query — it's designed to match the DSL patterns actually produced by
    the router (see the few-shot examples in query_parser.py), not to be
    provably correct for every conceivable plan shape.
    """
    steps_by_var = {s.output_variable: s for s in plan.steps}
    final_var = plan.final_variable
    final_step = steps_by_var.get(final_var)
    final_signature = (
        (final_step.operation, final_step.parameters.get("target"))
        if final_step is not None else None
    )

    ancestors = _ancestors_of(final_var, steps_by_var)

    # consumers[var] = step_ids that take `var` as one of their inputs
    consumers: Dict[str, List[int]] = {}
    for s in plan.steps:
        for inp in s.inputs:
            consumers.setdefault(inp, []).append(s.step_id)

    artifacts: Dict[str, DisplayArtifact] = {}

    for step in plan.steps:
        var = step.output_variable
        if var not in ancestors:
            continue  # computed, but never actually feeds the result
        gdf = variables.get(var)
        if gdf is None:
            continue

        filter_kind = None
        concept_label = None
        explains_source = None

        if var == final_var:
            role, default_visible = ROLE_FINAL, True

        elif step.operation in LEAF_OPS:
            same_as_final = final_signature == (step.operation, step.parameters.get("target"))
            if same_as_final:
                role, default_visible = ROLE_INTERMEDIATE, False
            else:
                role, default_visible = ROLE_CONTEXT, True
                filter_kind = "region"

        elif step.operation == "tool" and step.parameters.get("target") == "buffer":
            role, default_visible = ROLE_CONTEXT, True
            filter_kind = "distance"
            dist_km = step.parameters.get("buffer_distance_km") or 0
            src_var = step.inputs[0] if step.inputs else None
            src_step = steps_by_var.get(src_var) if src_var else None
            concept_label = artifacts[src_var].label if src_var in artifacts else "target"
            if src_step is not None and src_step.operation in LEAF_OPS:
                explains_source = src_var
            distance_miles = round(dist_km / KM_PER_MILE, 1)

        else:  # intersection / union / difference / add / get_centroid
            role, default_visible = ROLE_INTERMEDIATE, False

        op_type = step.operation if step.operation != "tool" else f"tool:{step.parameters.get('target')}"

        artifacts[var] = DisplayArtifact(
            id=var,
            label=_label_for(step, artifacts),
            geometry=gdf,
            source_node=str(step.step_id),
            type=op_type,
            role=role,
            default_visible=default_visible,
            filter_kind=filter_kind,
            distance_miles=distance_miles if filter_kind == "distance" else None,
            concept_label=concept_label,
            explains_source=explains_source,
        )

    # Collapse: a leaf whose *only* consumer is a single buffer step
    # shouldn't get its own visible chip — the buffer chip already speaks
    # for it (e.g. hide "Hospitals", keep "5 mi radius around Hospitals").
    # Its `filter_kind` is cleared too: it's no longer an independent
    # condition to check in explain_feature (a point almost never literally
    # intersects another point), it's now purely a name/distance data
    # source for the buffer artifact's `explains_source` lookup.
    for step in plan.steps:
        if step.operation == "tool" and step.parameters.get("target") == "buffer" and step.inputs:
            src_var = step.inputs[0]
            if (
                src_var in artifacts
                and artifacts[src_var].role == ROLE_CONTEXT
                and consumers.get(src_var) == [step.step_id]
            ):
                artifacts[src_var].default_visible = False
                artifacts[src_var].filter_kind = None

    return artifacts


def explain_feature(
    feature_geom,
    artifacts: Dict[str, DisplayArtifact],
) -> List[dict]:
    """Single-feature convenience wrapper around the batched check
    functions below — checks one geometry against every filter-type
    context artifact. `build_display_bundle` does NOT use this for the
    real result set (it batches across all features at once, see there);
    this exists for one-off/ad hoc use and tests.
    """
    single = gpd.GeoDataFrame({"geometry": [feature_geom]}, crs="EPSG:4326")
    checks = []
    for art in artifacts.values():
        if art.role != ROLE_CONTEXT or art.filter_kind is None:
            continue
        if art.filter_kind == "region":
            passed, geoms = _region_check_batch(single, art.geometry)
            checks.append({"label": f"In {art.label}", "passed": passed[0], "detail": None, "geometry": geoms[0]})
        elif art.filter_kind == "distance":
            src = artifacts.get(art.explains_source) if art.explains_source else None
            src_gdf = src.geometry if src is not None else None
            passed, details, geoms = _distance_check_batch(single, art.geometry, src_gdf)
            concept = (art.concept_label or "target").rstrip("s").lower() if art.concept_label else "target"
            checks.append({
                "label": f"Within {art.distance_miles:g} mi of {concept}",
                "passed": passed[0], "detail": details[0], "geometry": geoms[0],
            })
    return checks


def _region_check_batch(final_gdf: gpd.GeoDataFrame, ctx_gdf: gpd.GeoDataFrame):
    """For every row in final_gdf, does it intersect ctx_gdf (any row)?
    One spatial-index-backed join for the *whole* feature set at once,
    instead of a fresh `.intersects(...).any()` scan of ctx_gdf repeated
    once per feature (the original per-feature approach) — this is the
    main region-check cost driver once you have more than a handful of
    results, and sjoin uses ctx_gdf's spatial index rather than a linear
    scan per call.

    Returns (passed: list[bool], matched_geometry: list[geometry|None]),
    aligned with final_gdf's row order.
    """
    n = len(final_gdf)
    passed = [False] * n
    matched = [None] * n

    ctx_gdf = ensure_crs(ctx_gdf)
    if ctx_gdf.empty:
        return passed, matched

    left = ensure_crs(final_gdf)[["geometry"]].reset_index(drop=True)
    left["_fidx"] = range(n)
    right = ctx_gdf[["geometry"]].reset_index(drop=True)
    right["_cidx"] = range(len(right))

    joined = gpd.sjoin(left, right, how="left", predicate="intersects")
    for _, r in joined.iterrows():
        fidx = int(r["_fidx"])
        cidx = r.get("_cidx")
        if pd.notna(cidx):
            passed[fidx] = True
            if matched[fidx] is None:
                matched[fidx] = right.geometry.iloc[int(cidx)]

    return passed, matched


def _distance_check_batch(
    final_gdf: gpd.GeoDataFrame,
    buffer_gdf: gpd.GeoDataFrame,
    source_gdf: Optional[gpd.GeoDataFrame],
):
    """Pass/fail (within the buffer polygon) + nearest-named-source-feature detail.

    Ultra-fast exact minimum distance to LineString/Polygon/Point boundaries
    using direct C-level shapely.STRtree indexing and vectorized operations.
    """
    n = len(final_gdf)
    passed, _ = _region_check_batch(final_gdf, buffer_gdf)
    details = [None] * n
    geoms = [None] * n

    if source_gdf is None or source_gdf.empty:
        return passed, details, geoms

    source_gdf = ensure_crs(source_gdf)
    final_gdf = ensure_crs(final_gdf)

    # 1. Project to UTM once
    metric_crs = source_gdf.estimate_utm_crs()
    src_geoms = source_gdf.geometry.to_crs(metric_crs).to_numpy()
    feat_geoms = final_gdf.geometry.to_crs(metric_crs).to_numpy()

    # 2. Query GEOS C-based STRtree directly (bypasses GeoPandas sjoin overhead)
    tree = shapely.STRtree(src_geoms)
    nearest_src_idxs = tree.nearest(feat_geoms)

    # 3. Vectorized exact C-level boundary distance calculation
    dists_m = shapely.distance(feat_geoms, src_geoms[nearest_src_idxs])
    dists_miles = dists_m / 1609.34

    # 4. Extract names and geometries via vectorized Pandas slicing
    name_col = next((c for c in ("name", "target") if c in source_gdf.columns), None)
    if name_col is not None:
        names_series = source_gdf[name_col].fillna("").astype(str).to_numpy()
        matched_names = names_series[nearest_src_idxs]
    else:
        matched_names = np.full(n, "", dtype=object)

    src_orig_geoms = source_gdf.geometry.to_numpy()

    # 5. Fast memory block assignment without iterrows
    for i in range(n):
        src_idx = nearest_src_idxs[i]
        name = matched_names[i].strip() or "Nearest match"
        details[i] = f"{name} — {dists_miles[i]:.1f} mi"
        geoms[i] = src_orig_geoms[src_idx]

    return passed, details, geoms
def build_display_bundle(plan: QueryPlan, variables: Dict[str, gpd.GeoDataFrame]) -> dict:
    """Convenience entry point: classify the plan and attach a per-feature
    'why this result' explanation to every row of the final GeoDataFrame.

    `context` here is *global* context only — layers that mean the same
    thing for every result (a region boundary, a wildfire-risk polygon):
    shown once, in the toggle list. Distance-based filters (a buffer
    around hospitals) are inherently *local* — each result has its own
    nearest hospital and its own distance — so they never appear as a
    single shared layer here; they only show up per-feature, inside that
    feature's own `checks`.

    Returns:
        {
            "final": DisplayArtifact,
            "context": [DisplayArtifact, ...],   # filter_kind == "region" only
            "features": [
                {"index": 0, "label": ..., "score": ...,
                 "checks": [{"label", "passed", "detail", "geometry"}, ...]},
                ...
            ],
        }
    """
    artifacts = classify(plan, variables)
    final_var = plan.final_variable
    final_artifact = artifacts.get(final_var)

    global_context = [a for a in artifacts.values() if a.role == ROLE_CONTEXT and a.filter_kind == "region"]
    distance_context = [a for a in artifacts.values() if a.role == ROLE_CONTEXT and a.filter_kind == "distance"]

    features = []
    if final_artifact is not None:
        final_gdf = ensure_crs(final_artifact.geometry).reset_index(drop=True)
        n = len(final_gdf)

        # Compute each context artifact's check ONCE for the whole result
        # set, rather than once per feature per artifact.
        region_results = {a.id: _region_check_batch(final_gdf, a.geometry) for a in global_context}
        distance_results = {}
        for a in distance_context:
            src = artifacts.get(a.explains_source) if a.explains_source else None
            src_gdf = src.geometry if src is not None else None
            distance_results[a.id] = _distance_check_batch(final_gdf, a.geometry, src_gdf)

        for i, row in final_gdf.iterrows():
            label = None
            for col in ("name", "target"):
                if col in final_gdf.columns and pd.notna(row.get(col)):
                    label = str(row[col])
                    break

            checks = []
            for a in global_context:
                passed_list, geom_list = region_results[a.id]
                checks.append({
                    "label": f"In {a.label}",
                    "passed": passed_list[i],
                    "detail": None,
                    "geometry": geom_list[i],
                })
            for a in distance_context:
                passed_list, detail_list, geom_list = distance_results[a.id]
                concept = (a.concept_label or "target").rstrip("s").lower() if a.concept_label else "target"
                checks.append({
                    "label": f"Within {a.distance_miles:g} mi of {concept}",
                    "passed": passed_list[i],
                    "detail": detail_list[i],
                    "geometry": geom_list[i],
                })

            features.append({
                "index": i,
                "label": label or f"Result {i + 1}",
                "score": float(row["score"]) if "score" in final_gdf.columns and pd.notna(row.get("score")) else None,
                "checks": checks,
            })

    return {
        "final": final_artifact,
        "context": global_context,
        "features": features,
    }


def _geom_to_geojson(geom):
    if geom is None:
        return None
    return mapping(geom)


def _artifact_to_dict(art: DisplayArtifact) -> dict:
    gdf = ensure_crs(art.geometry)
    return {
        "id": art.id,
        "label": art.label,
        "type": art.type,
        "role": art.role,
        "default_visible": art.default_visible,
        "source_node": art.source_node,
        "feature_count": int(len(gdf)),
        "geojson": json.loads(gdf.to_json()),
    }


def serialize_bundle(bundle: dict) -> dict:
    """JSON-safe version of build_display_bundle()'s output: DisplayArtifacts
    become plain dicts with GeoJSON geometry, and each check's shapely
    `geometry` becomes a GeoJSON geometry dict, instead of live
    GeoDataFrame/shapely objects."""
    features = []
    for feat in bundle.get("features", []):
        checks = [
            {**{k: v for k, v in c.items() if k != "geometry"}, "geometry": _geom_to_geojson(c.get("geometry"))}
            for c in feat["checks"]
        ]
        features.append({**feat, "checks": checks})

    return {
        "final": _artifact_to_dict(bundle["final"]) if bundle.get("final") is not None else None,
        "context": [_artifact_to_dict(a) for a in bundle.get("context", [])],
        "features": features,
    }
