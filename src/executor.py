"""
DAG executor.

Walks a `QueryPlan` (an ordered list of `PipelineStep`s) and executes each
step against an in-memory variable table, resolving `inputs` references to
prior steps' outputs as it goes. Every step's result is a GeoDataFrame in
the standard schema, so tool operations can consume the output of *any*
prior step regardless of which operation produced it.

Steps run in dependency-topological LEVELS, not plan order: every step
whose inputs are already resolved runs in the same level, concurrently, via
a thread pool. Level N+1 only starts once level N has fully resolved.
Independent branches — e.g. `demo(a, "elderly")`, `osm(a, "hospitals")`,
`osm(a, "roads")` that only later get ANDed together via intersection — are
extremely common in router-generated plans (see query_parser.py's few-shot
examples) and cost `sum(branch latencies)` under strict sequential
execution for no reason; running same-level steps concurrently turns that
into `max(branch latencies)` per level. This is safe because steps within
a level never depend on each other by construction (that's the definition
of "same level"), each step only touches its own slice of `self.variables`
by writing to its own `output_variable` key, and the heavy per-step work
(GPU embedding search, geopandas spatial ops, `requests`-based geocoding)
all releases the GIL during its actual C/CUDA compute, so threads offer
real concurrency here despite the GIL — including full parallelism for
independent network-bound geocode() calls.
"""

from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

import geopandas as gpd

import config
from query_parser import QueryPlan, PipelineStep
from operations import geocode as geocode_op
from operations import demo as demo_op
from operations import vision as vision_op
from operations import osm as osm_op
from operations import change as change_op
from operations.tool import TOOL_DISPATCH
from schema import buffer_points_if_needed


class PipelineContext:
    """Holds the long-lived assets (models, embeddings, vision indices)
    needed across steps, so they're loaded once per DLPK session rather
    than once per step or once per query."""

    def __init__(self, demo_gdf, ae_embeddings, demo_model, text_embedder, vision_encoder, vision_indices,
                 grounder=None, vision_year_indices=None, osm_data=None, osm_category_embeddings=None):
        self.demo_gdf = demo_gdf
        self.ae_embeddings = ae_embeddings
        self.demo_model = demo_model
        self.text_embedder = text_embedder
        self.vision_encoder = vision_encoder
        self.vision_indices = vision_indices  # dict: resolution -> TurboQuantSearchIndex
        self.grounder = grounder
        self.vision_year_indices = vision_year_indices or {}  # {year: {resolution: TurboQuantSearchIndex}}
        self.osm_data = osm_data or {}  # {mode: GeoDataFrame}
        self.osm_category_embeddings = osm_category_embeddings or {}  # {mode: DataFrame with category+embedding}


class PipelineExecutor:
    """Executes a QueryPlan step by step against a shared variable table."""

    def __init__(self, context: PipelineContext):
        self.context = context
        self.variables: Dict[str, gpd.GeoDataFrame] = {}

    def _resolve(self, variable_name: str) -> gpd.GeoDataFrame:
        if variable_name not in self.variables:
            raise KeyError(f"Step references undefined variable '{variable_name}'.")
        return self.variables[variable_name]

    def _single_input(self, step: PipelineStep) -> gpd.GeoDataFrame:
        """Most geocode/demo/vision steps take zero or one input region."""
        if not step.inputs:
            return None
        return self._resolve(step.inputs[0])

    def run_step(self, step: PipelineStep) -> gpd.GeoDataFrame:
        params = step.parameters

        if step.operation == "geocode":
            result = geocode_op.geocode(params.get("target"))

        elif step.operation == "demo":
            region = self._single_input(step)
            if region is not None:
                region = buffer_points_if_needed(region)
            result = demo_op.search_demographics(
                target=params.get("target"),
                region=region,
                demo_gdf=self.context.demo_gdf,
                ae_embeddings=self.context.ae_embeddings,
                clip_model=self.context.demo_model,
                text_embedder=self.context.text_embedder
            )

        elif step.operation == "vision":
            region = self._single_input(step)
            if region is not None:
                region = buffer_points_if_needed(region)
            resolution = params.get("resolution")
            time_period = params.get("time")
            if time_period and time_period in config.VISION_YEARS:
                year = config.VISION_YEARS[time_period]
                turbo_index = self.context.vision_year_indices.get(year, {}).get(resolution)
            else:
                turbo_index = self.context.vision_indices.get(resolution)
            result = vision_op.search_vision(
                target=params.get("target"),
                region=region,
                vision_encoder=self.context.vision_encoder,
                vision_grounder=self.context.grounder,
                turbo_index=turbo_index,
                resolution=resolution,
            )

        elif step.operation == "osm":
            region = self._single_input(step)
            if region is not None:
                region = buffer_points_if_needed(region)
            result = osm_op.search_osm(
                mode=params.get("osm_mode"),
                query=params.get("target"),
                region=region,
                osm_data=self.context.osm_data,
            )

        elif step.operation == "change":
            region = self._single_input(step)
            if region is not None:
                region = buffer_points_if_needed(region)
            resolution = params.get("resolution")
            result = change_op.change(
                query=params.get("target"),
                from_time=params.get("from_time"),
                to_time=params.get("to_time"),
                mode=params.get("mode", "new"),
                region=region,
                vision_encoder=self.context.vision_encoder,
                vision_year_indices=self.context.vision_year_indices,
                resolution=resolution,
            )

        elif step.operation == "tool":
            result = self._run_tool_step(step)

        else:
            raise ValueError(
                f"Unknown operation '{step.operation}' in step {step.step_id}."
            )

        # Fallback to empty GeoDataFrame if operation returned None
        if result is None:
            result = gpd.GeoDataFrame()

        self.variables[step.output_variable] = result
        return result
    
    def _run_tool_step(self, step: PipelineStep) -> gpd.GeoDataFrame:
        action = step.parameters.get("target")
        handler = TOOL_DISPATCH.get(action)
        if handler is None:
            raise ValueError(
                f"Unsupported tool action '{action}' in step {step.step_id}."
            )

        inputs = [self._resolve(name) for name in step.inputs]

        if action == "buffer":
            if len(inputs) != 1:
                raise ValueError(
                    f"'buffer' expects exactly 1 input, got {len(inputs)} in step {step.step_id}."
                )
            return handler(inputs[0], step.parameters.get("buffer_distance_km"))

        if action == "get_centroid":
            if len(inputs) != 1:
                raise ValueError(
                    f"'get_centroid' expects exactly 1 input, got {len(inputs)} in step {step.step_id}."
                )
            return handler(inputs[0])

        if len(inputs) != 2:
            raise ValueError(
                f"'{action}' expects exactly 2 inputs, got {len(inputs)} in step {step.step_id}."
            )
        return handler(inputs[0], inputs[1])

    def _topological_levels(self, plan: QueryPlan) -> List[List[PipelineStep]]:
        """Group steps into levels: level 0 has no unresolved dependencies,
        level 1 depends only on level 0's outputs, etc. Steps within a level
        are mutually independent by construction and can run concurrently."""
        steps_by_var = {s.output_variable: s for s in plan.steps}
        resolved = set()
        remaining = list(plan.steps)
        levels: List[List[PipelineStep]] = []

        while remaining:
            ready = [
                s for s in remaining
                if all(inp in resolved or inp not in steps_by_var for inp in s.inputs)
            ]
            if not ready:
                # Shouldn't happen for a well-formed DAG (no cycles), but
                # don't hang forever if the router ever produces one — run
                # whatever's left in one level instead of looping forever.
                ready = remaining

            ready_ids = {s.step_id for s in ready}
            levels.append(ready)
            resolved.update(s.output_variable for s in ready)
            remaining = [s for s in remaining if s.step_id not in ready_ids]

        return levels

    def run_plan(self, plan: QueryPlan, verbose: bool = True) -> gpd.GeoDataFrame:
        levels = self._topological_levels(plan)
        max_workers = max((len(level) for level in levels), default=1)

        with ThreadPoolExecutor(max_workers=max(max_workers, 1)) as pool:
            for level in levels:
                if verbose:
                    names = ", ".join(f"{s.step_id}:{s.operation}" for s in level)
                    print(f"[level, {len(level)} step(s) in parallel] {names}")

                if len(level) == 1:
                    # No thread-pool overhead for the (very common) single-
                    # step level.
                    self.run_step(level[0])
                else:
                    futures = {pool.submit(self.run_step, s): s for s in level}
                    for future in futures:
                        future.result()  # re-raises any step's exception here

                if verbose:
                    for step in level:
                        result = self.variables[step.output_variable]
                        print(f"  -> '{step.output_variable}': {len(result)} feature(s)")

        final_result = self.variables[plan.final_variable]
        if not final_result.empty and len(final_result) > config.MAX_RESULTS:
            final_result = final_result.head(config.MAX_RESULTS).copy()
        return final_result
