"""
QueryEarth DLPK entry point.

Loads all long-lived assets once (demographic embeddings + model, vision
text encoder, TurboQuant vision indices, local text embedder, local GGUF
query router), then serves `predict(query)` calls cheaply against them.

    from queryearth import QueryEarth
    qe = QueryEarth()                 # heavy, one-time load
    fs = qe.find("find golf courses near wealthy suburbs in Texas")
"""
import logging

# Silence Fiona's environment logger before importing fiona/geopandas
logging.getLogger("fiona._env").setLevel(logging.CRITICAL)
logging.getLogger("fiona").setLevel(logging.CRITICAL)

import json
import os
import time
import warnings
warnings.filterwarnings("ignore")

import geopandas as gpd
from geopandas import GeoDataFrame
from shapely.geometry.polygon import orient

import config
import models
import query_parser
import artifacts as artifacts_module
from executor import PipelineContext, PipelineExecutor
from turboquant_index import TurboQuantSearchIndex
from schema import GEOMETRY_COL, SCORE_COL
from operations.grounding import SpatialGrounder
from operations.osm import load_osm_data


# ------------------------------------------------------------------
# QueryEarth
# ------------------------------------------------------------------
class QueryEarth:
    """Loads every long-lived asset once at construction time; `predict()`
    is the cheap, repeatable per-query call."""

    def __init__(self, **kwargs):
        self.name = "QueryEarth"
        self.description = "Natural-language geospatial query pipeline."
        

    def initialize(self, **kwargs):
        demo_gdf, ae_embeddings, demo_model, text_embedder = models.load_demographic_assets()
        vision_encoder = models.VisionEncoder()

        grounder = SpatialGrounder()

        vision_indices = {}
        for resolution, folder in config.VISION_INDEX_DIRS.items():
            if os.path.isdir(folder) and os.path.exists(os.path.join(folder, "meta.json")):
                vision_indices[resolution] = TurboQuantSearchIndex(folder)
            else:
                print(f"No vision index found for resolution '{resolution}' at {folder} (skipping).")

        vision_year_indices = {}
        for year, res_map in config.VISION_YEAR_INDEX_DIRS.items():
            vision_year_indices[year] = {}
            for resolution, folder in res_map.items():
                if os.path.isdir(folder) and os.path.exists(os.path.join(folder, "meta.json")):
                    vision_year_indices[year][resolution] = TurboQuantSearchIndex(folder)
                else:
                    print(f"No vision index for year {year}, resolution '{resolution}' at {folder} (skipping).")

        # Load OSM data for the default year
        osm_data = load_osm_data(config.OSM_EMBEDDING_DIR, config.OSM_DEFAULT_YEAR)

        self.context = PipelineContext(
            demo_gdf, ae_embeddings, demo_model, text_embedder,
            vision_encoder, vision_indices, grounder,
            vision_year_indices=vision_year_indices,
            osm_data=osm_data,
        )
        self.executor = PipelineExecutor(self.context)

    def clean_duplicates(self, gdf:GeoDataFrame):
        gdf["geometry"] = gdf["geometry"].set_precision(1e-6)
        gdf_clean = gdf.drop_duplicates(subset=["geometry"])

        return gdf_clean


    def find(self, query: str, save_gdf = False, **kwargs):
        """Given a natural-language query string, parse it into a pipeline
        plan, execute it, and return the result as an arcgis FeatureSet."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("QueryEarth.predict expects a non-empty query string.")
        
        begin = time.time()

        plan = query_parser.parse_query(query)
        result_gdf = self.executor.run_plan(plan)
        result_gdf = self.clean_duplicates(result_gdf)

        print(f"Found the objects in {(time.time() - begin)} seconds.")

        # if save_gdf:
        #     os.makedirs(rf"results\{query.replace(" ", "_")}", exist_ok=True)
        #     result_gdf.to_file(rf"results\{query.replace(" ", "_")}\output .shp")

        return result_gdf

    def find_with_context(self, query: str, **kwargs):
        """Like `find()`, but also returns the display bundle needed to
        render global context layers + per-feature "why this result?"
        explanations, with no LLM involved in deciding what to show —
        role/visibility is derived purely from the plan's DAG shape (see
        artifacts.classify).

        Returns:
            (result_gdf, bundle) where `bundle` is the dict produced by
            artifacts.build_display_bundle(): {"final", "context", "features"}.
            Call artifacts.serialize_bundle(bundle) to get a JSON/GeoJSON-safe
            version for a web front end.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("QueryEarth.find_with_context expects a non-empty query string.")

        begin = time.time()

        plan = query_parser.parse_query(query)
        result_gdf = self.executor.run_plan(plan)
        result_gdf = self.clean_duplicates(result_gdf)

        # executor.variables holds every step's output GeoDataFrame keyed by
        # its DSL variable name — exactly what classify() needs, already
        # computed, no extra work. Swap in the deduped/truncated final gdf
        # so it's the one both returned as the FeatureSet and explained.
        variables = dict(self.executor.variables)
        variables[plan.final_variable] = result_gdf
        bundle = artifacts_module.build_display_bundle(plan, variables)

        print(f"Found the objects in {(time.time() - begin)} seconds.")
        return result_gdf, bundle


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ESRI Earth Search Engine")
    parser.add_argument("-query", type=str, required=True, help="The query to search for")
    args = parser.parse_args()

    qe = QueryEarth()
    qe.initialize()
    result = qe.find(args.query)