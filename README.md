# ESRI Earth Search Engine

GeoSearch is a geospatial query pipeline that converts natural-language requests into executable spatial programs. It searches across multiple modalities — demographic data, satellite imagery, OpenStreetMap features, and temporal change detection — to return geographically grounded results.


## Search Modalities

| Modality | Description | Data Source |
|----------|-------------|-------------|
| `geocode` | Resolves place names to boundary polygons | Nominatim / Esri |
| `demo` | Demographic/socioeconomic similarity search | TabularTextCLIP + census embeddings |
| `vision-low` | Large objects and land-use patterns from satellite imagery | TurboQuant (low-res) |
| `vision-high` | Small/fine-grained objects from satellite imagery | TurboQuant (high-res) |
| `osm` | Structured categorical data (roads, waterways, POIs) | OpenStreetMap Parquet |
| `change-low` / `change-high` | Temporal change detection across time periods | Multi-year vision indices |
| `buffer` | Spatial proximity filtering | Shapely |
| `intersection` / `union` / `difference` | Set algebra over GeoDataFrames | Shapely |

## Project Structure

```
src/
├── queryearth.py          # Main entry point — QueryEarth class
├── query_parser.py        # Natural language → DSL plan (Gemini router)
├── executor.py            # DAG executor with topological-level concurrency
├── config.py              # Central configuration (paths, model settings)
├── models.py              # TabularTextCLIP, VisionEncoder, LocalTextEmbedder
├── schema.py              # Shared GeoDataFrame conventions (geometry + score)
├── turboquant_index.py    # TurboQuant compressed index loader/searcher
├── artifacts.py           # Display classification and per-feature explanations
└── operations/
    ├── geocode.py         # Geocoding (Nominatim + Esri fallback)
    ├── demo.py            # Demographic similarity search
    ├── vision.py          # Visual similarity search
    ├── osm.py             # OpenStreetMap keyword search
    ├── change.py          # Temporal change detection
    ├── tool.py            # Spatial operations (buffer, union, etc.)
    ├── grounding.py       # GroundingDINO object detection pipeline
    └── threshold.py       # Adaptive score thresholding
```

## Getting Started

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended for inference speed)
- Gemini API key (set as `GEMINI_API_KEY` environment variable)

### Installation

```bash
pip install -r requirements.txt
```

### Data Assets

Place the following under `E:\Data\query-earth\embeddings\` (configurable in `config.py`):

```
embeddings/
├── demography-emb.parquet          # Demographic GeoParquet with embeddings
├── demo_embedder.pth               # Trained TabularTextCLIP checkpoint
├── RS5M_ViT-H-14.pt               # CLIP vision model weights
├── turboquant/
│   ├── vision-low-2026/            # Low-res TurboQuant index
│   ├── vision-high-2026/           # High-res TurboQuant index
│   ├── vision-low-2014/            # Past period indices
│   ├── vision-high-2014/
│   ├── vision-low-2020/
│   └── vision-high-2020/
└── osm/
    ├── 2026/                       # Current OSM data (Parquet)
    └── 2014/                       # Historical OSM data
```

### Usage

```python
from queryearth import QueryEarth

qe = QueryEarth()
qe.initialize()

# Simple search
results = qe.find("find golf courses near wealthy suburbs in Texas")

# Search with context (returns explanation bundle for UI rendering)
results, bundle = qe.find_with_context(
    "find baseball fields in low-income neighborhoods in Los Angeles"
)
```

### CLI

```bash
python -m src.queryearth -query "find hospitals in areas with many elderly residents"
```

## Query Examples

| Query | Generated Plan |
|-------|---------------|
| "Find primary highways in California" | `geocode → osm` |
| "Find swimming pools in wealthy neighborhoods" | `demo → vision-high` |
| "Where were forests cleared between 2014 and 2026?" | `change-low("forests", "past", "present", "removed")` |
| "Find new buildings near fire stations in wildfire-prone areas" | `geocode → osm → buffer → demo → intersection → change-high` |
| "Find illegal-looking construction inside protected wildlife areas" | `vision-low → change-high → buffer → vision-high` |

## Key Design Decisions

### DAG-Based Execution

Queries are parsed into a directed acyclic graph of pipeline steps. Steps at the same dependency level execute concurrently via a thread pool, reducing latency when independent branches (e.g., `demo(a, "elderly")` and `osm(a, "hospitals")`) run in parallel.

### TurboQuant Indices

Vision search uses pre-compressed embedding indices (7.7M+ vectors). IVF routing probes only a fraction of clusters; spatially-filtered queries skip IVF entirely and use a bounding-box pre-filter followed by exact polygon refinement.

### Multi-Modal Routing

The Gemini router distinguishes between:
- **Structured data** (OSM) — categorical features like road types, waterway classes, named POIs
- **Visual appearance** (Vision) — physical objects visible in imagery but absent from databases
- **Demographic properties** (Demo) — population, income, age distributions

Concepts that exist in both modalities (e.g., "parking lots") can use multiple operations when the user's intent requires it.

### Adaptive Thresholding

Vision scores are filtered using a dynamic threshold (`max(0.2, max_score - 0.07)`) rather than a fixed cutoff, adapting to the score distribution of each query.

## Configuration

All settings live in `src/config.py`. Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DEVICE` | CUDA if available | Compute device for model inference |
| `DEFAULT_RESOLUTION` | `"high"` | Default vision index resolution |
| `MAX_RESULTS` | `500` | Maximum features per pipeline step |
| `AUTOMATIC_BUFFER_RADIUS` | `1 km` | Auto-buffer radius for point geometries |
| `VISION_NPROBE_DEFAULT` | `24` | IVF clusters probed in global vision search |
| `CHANGE_DISTANCE_THRESHOLD` | `0.001°` | Max distance for cross-temporal matching |


-------------------------------------------------------------------------------------------------------------