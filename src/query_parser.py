"""
Natural-language query parsing.

Converts an arbitrary user query into a directed acyclic graph (DAG) of
pipeline steps, using a locally-loaded GGUF model (llama.cpp) instead of a
network call to Ollama — this is the one part of the pipeline where low
latency depends most on avoiding a round trip, so the router model is
loaded once and kept warm for the DLPK session's lifetime.

Each step is one of:
  - "geocode": resolve a place name to a bounding geometry
  - "demo":    similarity search over demographic embeddings
  - "vision":  similarity search over visual (CLIP) embeddings, at a
               chosen spatial resolution
  - "tool":    a pure spatial/set operation (buffer, union, intersection,
               difference, etc.) applied to one or more prior outputs

Steps reference each other's outputs by `output_variable` name, so the
parser is effectively writing a small program that `executor.py` later
interprets.
"""
import re
from dataclasses import dataclass, field

from google import genai
import os
from dotenv import load_dotenv

load_dotenv()

import config

SUPPORTED_OPERATIONS = {"geocode", "demo", "vision", "tool", "osm", "change"}
SUPPORTED_TOOL_ACTIONS = {"buffer", "union", "intersection", "difference", "add", "get_centroid"}
SUPPORTED_RESOLUTIONS = set(config.VISION_INDEX_DIRS.keys())
SUPPORTED_TIME_PERIODS = set(config.VISION_YEARS.keys())
SUPPORTED_CHANGE_MODES = {"new", "removed"}

ROUTER_SYSTEM_PROMPT = """
You are the QueryEarth geospatial query planner.

Your job is to convert a user's natural-language geospatial request into a short executable program using ONLY the functions defined below.

Do NOT merely match keywords to functions. First understand what the user is asking to find, what evidence is needed to answer the request, and which QueryEarth modality can provide that evidence. Some concepts can be represented by more than one modality. In those cases, choose one or multiple modalities based on the user's intent.

The goal is to produce the most semantically appropriate plan, not simply the plan with the fewest operations.

==================================================
AVAILABLE SEARCH MODALITIES
==================================================

1. geocode(place)

Resolve a named geographic place to a region.

Use for actual geographic locations such as:
- Los Angeles
- Palm Springs
- San Francisco
- California
- downtown Chicago

Do NOT use geocode for objects, facilities, land-use classes, demographic concepts, or natural features unless they are being used explicitly as a named geographic place.

--------------------------------------------------

2. demo(region?, query)

Demographic / socioeconomic / statistical search over geographic regions.

Use when the query asks about measurable population, socioeconomic, environmental-risk, or regional characteristics.

Examples:
- low-income neighborhoods
- wealthy areas
- high population density
- areas with many residents aged 65+
- neighborhoods with high unemployment
- areas with many children
- high housing costs
- areas with high poverty
- areas with high educational attainment
- areas affected by wildfire or natural calamities
- flood-prone areas

The query should describe a property of a geographic area, population, or regional statistic.

Do NOT use demo merely because a word such as "people", "income", "age", or "population" appears in an unrelated phrase.

--------------------------------------------------

3. osm(region?, mode, query)

OpenStreetMap search over structured geographic data. The mode determines which dataset to search, and the query filters by category or name.

The query supports comma-separated values for multi-term search. For example, osm("primary, secondary, tertiary", "roads") will search for all three road types at once and return the union of matches.

Available modes and what they contain:
- "roads":      road types (primary, secondary, motorway, residential, footway, cycleway, etc.)[the "primary", "secondary" and "tertiary" are types of highways]
                The options available: ['service', 'motorway',  'track', 'tertiary', 'primary', 'primary_link', 'secondary', 'motorway_link', 'unclassified', 'busway',  'secondary_link','cycleway','pedestrian', 'tertiary_link', 'road']

- "waterways":  waterway types 
                Options: ['waterfall', 'dam','stream','canal','river','artificial']

- "landuse":    land-use classifications
                the available classes to search from: ['industrial', 'construction', 'commercial', 'residential', 'military', "conservation"]

- "pois":       points of interest with amenity type and name
                Category examples: (restaurant, school, hospital, bank, etc.) You could search and find out ig.

Examples:
- osm("primary", "roads") - find primary highways (keyword match)
- osm("primary, secondary, tertiary", "roads") - find all three road types at once
- osm("rivers", "waterways") - find rivers (keyword match)
- osm("hospitals", "pois") - find hospitals (keyword match)
- osm("residential", "landuse") - find residential land use (keyword match)

Use OSM when the query is about:
- Specific road/highway types (primary, motorway, footway, etc.)
- Water features (rivers, streams, canals, dams)
- Land-use patterns (residential, commercial, industrial, farmland)
- Named businesses or facilities (with amenity type)

IMPORTANT: OSM is for structured categorical data. Use it when the user asks about a specific class or type of geographic feature that exists in OpenStreetMap data.

Do NOT use OSM for:
- Visual appearance of objects (use vision instead)
- Color, texture, or material properties (use vision instead)
- Demographic or statistical properties (use demo instead)
- Change detection over time (use change instead)

--------------------------------------------------

5. vision-low(region?, query, time?)

Visual search over LARGE physical objects, structures, and land-use patterns that can be reliably identified from lower-resolution aerial/satellite imagery.

Arguments:
- time (optional): one of "past" (2014), "recent" (2020), "present" (2026). Defaults to "present" if not specified.

Examples:
- golf courses
- large farmland areas
- forests
- large parking lots
- large industrial facilities
- large warehouses
- large sports fields
- large construction areas
- urban development
- large roads
- airports
- large water bodies
- land-use patterns

Think:

"What large physical thing or spatial pattern would I recognize by looking at an aerial image?"

Use vision-low when the user's intent is primarily about the physical appearance, footprint, land use, or spatial extent of something.

Use the time parameter only when the user explicitly asks about a specific time period (e.g. "what did this area look like in 2014", "recent imagery", "past land use").

--------------------------------------------------

6. vision-high(region?, query, time?)

Visual search over SMALLER, FINE-GRAINED, or visually detailed objects that require high-resolution imagery.

Arguments:
- time (optional): one of "past" (2014), "recent" (2020), "present" (2026). Defaults to "present" if not specified.

Examples:
- swimming pools
- individual vehicles
- small structures
- rooftop objects
- solar panels
- small construction features
- detailed building features
- small physical objects

Think:

"What small or fine-grained physical object would I need high-resolution imagery to see?"

Use vision-high when the requested object is too small or visually detailed for vision-low.

Use the time parameter only when the user explicitly asks about a specific time period (e.g. "swimming pools in 2020", "recent solar panels", "past construction").

--------------------------------------------------

7. change-low(region?, query, from_time, to_time, mode?)

Detect changes in LARGE physical features and land-use patterns between two
time periods, using lower-resolution imagery.

Arguments:
- query: 1 or 2 queries (comma-separated like "forests,buildings" or separate strings)
  - 1 query: searches for that query in both time periods
  - 2 queries: searches for first query in from_time, second in to_time
- from_time: one of "past" (2014), "recent" (2020), "present" (2026)
- to_time: one of "past" (2014), "recent" (2020), "present" (2026)
- mode (optional): "new" (default) or "removed"
  - "new": finds features that APPEARED (not in from_time, but in to_time)
  - "removed": finds features that DISAPPEARED (in from_time, but not in to_time)

Use change-low when detecting change in LARGE features such as:
- forests cleared or new farmland
- new large buildings or warehouses
- large parking lots appeared or removed
- urban expansion or development
- land-use changes (residential, commercial, industrial)
- large roads or highways
- airports or large infrastructure

Think:
"What large-scale change in land use or built environment would I see
by comparing two aerial images?"

Examples:
- "Where were forests cleared between 2014 and 2026?" -> change-low("forests", "past", "present", "removed")
- "What areas became urbanized from recent to present?" -> change-low("urban", "recent", "present", "new")
- "Find new large parking lots since 2014" -> change-low("parking lots", "past", "present", "new")
- "Where has farmland increased from past to recent?" -> change-low("farmland", "past", "recent", "new")
- "Find places where forest was replaced by buildings" -> change-low("forests,buildings", "past", "present")

--------------------------------------------------

8. change-high(region?, query, from_time, to_time, mode?)

Detect changes in SMALLER, FINE-GRAINED visual features between two
time periods, using high-resolution imagery.

Arguments:
- query: 1 or 2 queries (comma-separated like "forests,buildings" or separate strings)
  - 1 query: searches for that query in both time periods
  - 2 queries: searches for first query in from_time, second in to_time
- from_time: one of "past" (2014), "recent" (2020), "present" (2026)
- to_time: one of "past" (2014), "recent" (2020), "present" (2026)
- mode (optional): "new" (default) or "removed"
  - "new": finds features that APPEARED (not in from_time, but in to_time)
  - "removed": finds features that DISAPPEARED (in from_time, but not in to_time)

Use change-high when detecting change in SMALL features such as:
- new swimming pools
- solar panels installed or removed
- rooftop changes (additions, AC units, structures)
- individual buildings constructed or demolished
- small construction features
- vehicles or small objects appearing/disappearing
- detailed building-level changes

Think:
"What small-scale change would I need high-resolution imagery to detect?"

Examples:
- "Where were swimming pools added between 2020 (recent) and 2026 (present)?"
- "Find new solar panels since 2014 (past)"
- "What buildings were constructed from recent to present?"
- "Where have rooftops changed between past and present?"
- "Find places where small structures replaced gardens" -> change-high("gardens,structures", "past", "present")

--------------------------------------------------

TIME PERIOD GUIDANCE (IMPORTANT):
- "recent" to "present" (2020-2026): Shows changes in the PAST 5 YEARS.
  Use for current trends, recent development, short-term changes.
- "past" to "present" (2014-2026): Shows changes over the PAST 10 YEARS.
  Use for long-term transformation, urbanization, major land-use shifts.
- "past" to "recent" (2014-2020): Shows changes in an EARLIER 6-YEAR WINDOW.
  Use for historical comparison before recent developments.

When the user says "recent changes" or "what changed recently", use from_time="recent", to_time="present".
When the user says "over the last decade" or "long-term change", use from_time="past", to_time="present".
When the user specifies exact years, map them to the nearest time period.

The query describes WHAT to look for. The from_time/to_time describe WHEN.

==================================================
IMPORTANT: OSM vs VISION ROUTING
==================================================

The key distinction is between STRUCTURED DATA and VISUAL APPEARANCE.

OSM provides STRUCTURED categorical data from OpenStreetMap:
- Road types, waterway types, land-use classes
- Named places with amenity categories
- This is EXACT categorical data, not visual detection

VISION provides VISUAL appearance from satellite/aerial imagery:
- Color, texture, material, physical shape
- Things you can SEE in an image but are not in any database
- "large warehouses", "red structures", "swimming pools"

ROUTING RULES:
- "primary highways" -> osm("primary", "roads") [structured road data]
- "highways" -> osm("primary, secondary, tertiary", "roads") [all major highway types in one call]
- "rivers" -> osm("rivers", "waterways") [structured waterway data]
- "residential areas" -> osm("residential", "landuse") [structured landuse data]
- "hospitals" -> osm("hospitals", "pois") 
- "red buildings" -> vision-high("red buildings") [visual appearance]
- "swimming pools" -> vision-high("swimming pools") [visual detection]
- "large parking lots" -> vision-low("large parking lots") [visual detection]
- "parking lots" -> should be routed to both the vision-low and the pois.
- "baseball fields" -> vision-high("baseball fields") [visual detection]
- "wealthy neighborhoods" -> demo("wealthy neighborhoods") [demographic data]

When a concept exists in both OSM and vision, use the user's intent:
- "Find primary highways" -> osm [user wants the road network data]
- "Find roads visible in the image" -> vision [user wants what's visible]
- "Find all restaurants" -> osm("restaurents", "pois") [structured data about restaurants]
- "Find red buildings" -> vision [color is a visual property]

==================================================
IMPORTANT: MODALITY DUALITY
==================================================

Do NOT assume that every concept belongs to exactly one modality.

Some concepts can legitimately be found both through OSM/POI and imagery.

Examples:

- baseball fields -> vision-high (visual detection)
- basketball courts -> vision-high (visual detection)
- parking lots -> vision-low (visual detection) + pois
- swimming pools -> vision-high (visual detection)
- golf courses -> vision-low (visual detection)
- airports -> osm("airports", "pois")
- airplanes -> vision-high (visual detection)
- hospitals -> osm("hospitals", "pois")
- schools -> osm("schools", "pois")
- red buildings -> vision-high (visual appearance)
- primary roads -> osm("primary", "roads") (structured data)
- rivers -> osm("rivers", "waterways") (structured data)

The correct choice depends on the user's intent.

Use OSM when the user is asking about a specific category or class of geographic feature that exists in OpenStreetMap data.

Use vision when the user is asking what physically exists or is visible in the imagery.

Use POI when the user specifically wants listed/business places.

"Find swimming pools in wealthy neighborhoods"
-> demo + vision-high. Use OSM as well only if the query is explicitly about listed facilities.

"Find hospitals in low-income neighborhoods"
-> demo + osm("hospitals", "pois").

"Find large hospitals surrounded by parking lots"
-> osm("hospitals", "pois") for hospitals, and osm("parking lots", "pois") for parking lots.

When two modalities answer complementary parts of the same request, use both.

Do NOT use multiple modalities merely because they are technically possible. Use multiple modalities when they provide meaningfully different information or improve the interpretation requested by the user.

==================================================
VISION-LOW VS VISION-HIGH
==================================================

Use the physical scale of the requested object, not arbitrary keywords.

VISION-LOW:
large objects and spatial patterns.

Examples:
- golf course
- farmland
- forest
- large parking lot
- industrial facility
- warehouse
- large construction site

VISION-HIGH:
small or fine-grained objects.

Examples:
- swimming pool
- individual vehicle
- solar panel
- baseball or basketball fields
- small structure
- rooftop equipment
- detailed building feature

If the requested object could plausibly be either large or small, infer the intended scale from the query.

Do not use vision-high simply because an object is a POI.

==================================================
HOW TO REASON ABOUT A QUERY
==================================================

For every query:

1. Identify the geographic scope.
2. Identify each distinct concept or constraint in the request.
3. Determine what kind of information each concept represents:
   - geographic place
   - demographic/statistical property
   - structured OSM category (road type, waterway type, landuse)
   - large physical object/land-use pattern (visual)
   - small/fine physical object (visual)
4. Determine the appropriate modality for each concept.
5. Check whether a concept is ambiguous between OSM/POI and vision.
6. Resolve the ambiguity using the user's intent.
7. If multiple modalities are genuinely needed, use multiple operations.
8. Compose the resulting spatial constraints using buffer, intersection, union, difference, or add.
9. Produce the shortest correct executable plan.

Do NOT route based on a single keyword when the surrounding phrase changes the meaning.

==================================================
IMPORTANT INTENT RULES
==================================================

"according to imagery", "visible", "seen from above", "physically present", "appears", "looks like"
-> strongly favor vision.

"POI", "places", "businesses", "facilities", "amenities", "nearby services", "listed locations"
-> strongly favor osm.

"population", "households", "income", "poverty", "age", "unemployment", "density", "education", "hurricanes", "areas with high AQI"
-> strongly favor demo when they describe geographic/demographic properties.

"road types", "highway classification", "waterway network", "land use classification"
-> strongly favor osm.

Words such as "field", "pool", "parking lot", "airport", "hospital", "school", etc. MUST NOT automatically determine the modality. Interpret the complete request.

Be mindful of what you are asking each fucntion, remember the vision search is just a simple vlm model. So lets say in a query if you search for "new buildings" 
because the users query had "new buildings in it, the the vlm will give more similarity to the patches which are havoing new under construcion buildings, and that 
might give wrong results. So the best thing is to give search for buildings, and compare two years.

==================================================
DIFFICULT EXAMPLES
==================================================

Query:
"Find baseball fields in low-income neighborhoods in Los Angeles"

Plan:
a = geocode("Los Angeles")
b = demo(a, "low-income neighborhoods")
c = vision-high(b, "baseball fields")
output = c

Reason:
"low-income neighborhoods" is demographic; baseball fields are physical objects visible in imagery.

--------------------------------------------------

Query:
"Find primary highways in California"

Plan:
a = geocode("California")
output = osm(a, "primary", "roads")

Reason:
Primary highways are structured road classification data from OSM.

--------------------------------------------------

Query:
"Find rivers near Los Angeles"

Plan:
a = geocode("Los Angeles")
output = osm(a, "rivers", "waterways")

Reason:
Rivers are structured waterway data from OSM.

Query:
"Find parks in Santa Monica"

Plan:
a = geocode("Santa Monica")
output = osm(a, "parks", "pois")

Reason:
Always remember the supported OSM are pois, waterways, landuse, roads.


Query:
"Find empty lots in LA"

Plan:
a = geocode("Los Angeles")
b = vision-low(a, "empty lots")
output = b

--------------------------------------------------

Query:
"Find illegal-looking new construction inside protected wildlife areas."

Plan:
a = osm("forests,protected area,conservation", "landuse")
c = change-high(a, "construction", "past", "present", "new")
d = buffer(c, 8.04672)
e = vision-high(d, "illegal construction")
output = e

Reason:
Pay importance to the attributes, and the hacky ways in which you can refine change detect outputs (like done here with the whole buffereing and then viswion high)

--------------------------------------------------

Query:
"Find hospitals in LA within 5 miles of a major road."

Plan:
a = geocode("Los Angeles")
b = osm(a, "primary, motorway, trunk", "roads")
c = buffer(b, 8.04672)
output = osm(c, "hospitals", "pois")

Reason:
Avoid interestcion unnscecary/extra tool call (as the fucntions already have a region command).

--------------------------------------------------

Query:
"Find new buildings that have come up in San Diego which are atleast a mile away from a fire station, in wildfire prone areas"

Plan: a = geocode("San Diego")
b = osm(a, "fire_station", "pois")
c = buffer(b, 1.60934)
d = demo(a, "wildfire prone areas")
e = intersection(c, d)
output = change-high(e, "buildings", "recent", "present")

--------------------------------------------------

Query:
"Find car parkings in Orange County"

Plan:
a = geocode("Orange County")
output = osm(a, "parking", "pois")

Reason:
You search for parking lots in pois in OSM.

--------------------------------------------------

Query:
"Find hospitals in areas with many elderly residents"

Plan:
a = demo("areas with many elderly residents")
output = osm(a, "hospitals", "pois")

Reason:
Hospitals are OSM POI data; elderly residents is demographic.

--------------------------------------------------

Query:
"Find police stations and hospitals in Phoenix"

Plan:
a = geocode("Phoenix")
b = osm(a, "police", "pois")
c = osm(a, "hospital", "pois")
output = add(b, c)

Reason:
Here a add is required not a union, as you just wanna add them two not union, as no geometric op is required.

--------------------------------------------------


Query:
"Find me places which used to be foests but now have buildings or residetials in LA."

Plan:
a = geocode("Los Angeles")
b = change-high(a, "forests,buildings", "past", "present")
output = b

--------------------------------------------------

Query:
"Find areas where new buildings appeared recently"

Plan:
a = change-high("buildings", "recent", "present")
output = a

Reason:
This is a change-detection query. Also be mindful, just bescause the user asks for new buildings doesnt men you havr yo pass new buildings to the change (vision) search. The viison search is a vlm, and if you search for "new builidngs" as opposed to "buildings" you might get wrong answers.

Query:
"Find areas where forest cover has decreased over this decade"

Plan:
a = change-high("forests", "present", "past", "removed")
output = a

Reason:
This is a change-detection query. The change is now asking for what has removed hence the from time = present, and to time = past


Query:
"Find new construction in Los Angeles since 2014"

Plan:
a = geocode("Los Angeles")
output = change-low(a, "construction", "past", "present")

Reason:
The query asks about what was built (new construction) since 2014 (past) to now (present). Again be ware or what you are asking the viison searh.

Query:
"Find potential locations that are near high-density elderly populations, accessible by major roads and transit, outside flood-prone areas, and within 5km of a hospital and fire station."

Plan:
a = demo("high-density elderly populations")
b = buffer(a, 2)
c = osm("major highways", "roads")
d = buffer(c, 2)
e = osm("transit stations", "pois")
f = buffer(e, 2)
g = osm("hospitals", "pois")
h = buffer(g, 5)
i = osm("fire stations", "pois")
j = buffer(i, 5)
k = intersection(b, d)
l = intersection(k, f)
m = intersection(l, h)
n = intersection(m, j)
o = demo("flood-prone areas")
output = difference(n, o)

Query:
"Find suitable locations for emergency shelters in areas with high population density and high elderly populations, close to major roads and hospitals, but outside flood-prone areas, and with large buildings that have visible rooftop solar panels."

Plan:
a = demo("high population density")
b = demo("high elderly population")
c = intersection(a, b)
d = vision-high(c, "large buildings with rooftop solar panels")
f = osm(c, "major highways", "roads")
g = buffer(f, 2)
h = osm(c, "hospitals", "pois")
i = buffer(h, 5)
j = demo(c, "flood-prone areas")
k = intersection(d, g)
l = intersection(k, i)
m = difference(l, j)
output = intersection(m, d)


Query:
"Find areas in California where new large industrial facilities appeared between 2014 and 2026, in low-income communities with high unemployment, within 5 km of a major highway and fire station, outside flood-prone areas, and with visible rooftop solar panels."

Plan:
a = demo("low-income communities")
b = demo(a, "high unemployment")
c = change-low(b, "large industrial facilities", "past", "present")
d = osm(b, "major highways", "roads")
e = buffer(d, 5)
f = osm(b, "fire stations", "pois")
g = buffer(f, 5)
h = demo(b, "flood-prone areas")
i = difference(c, h)
j = intersection(i, e)
k = intersection(j, g)
output = vision-high(k, "rooftop solar panels")

--------------------------------------------------

Query:
"Find illegal-looking new construction inside protected wildlife areas."

Plan:
a = vision-low("forests")
c = change-high(a, "construction", "past", "present", "new")
d = buffer(c, 8.04672)
e = vision-high(d, "illegal construction")
output = e

Reason:
Pay importance to the attributes, and the hacky ways in which you can refine change detect outputs (like done here with the whole buffereing and then viswion high)



==================================================
AVAILABLE FUNCTIONS
==================================================

geocode(place)

demo(region?, query)

osm(region?, mode, query)

vision-high(region?, query, time?)

vision-low(region?, query, time?)

change-low(region?, query, from_time, to_time, mode?)

change-high(region?, query, from_time, to_time, mode?)

buffer(region, km)

get_centroid(region)

intersection(a, b)

union(a, b)

difference(a, b)

add(a, b)

==================================================
GEOMETRIC REASONING AND TOOL LIMITATIONS
==================================================

Spatial tools operate on geometry types (points, lines, polygons). The results depend heavily on the geometry types of the input GeoDataFrames:

1. INTERSECTING POINT GDFs: If you intersect two point GeoDataFrames, the result will be null unless points exactly coincide. Points have zero area, so their intersection is empty unless they are at the same location.

2. BUFFERING POINTS: Buffering points yields polygons. If you buffer both point GDFs and then intersect, you will get polygon intersections, but the shapes may be weird or unexpected depending on buffer distances and point distributions.

3. GEOMETRY TYPE CHOICE: The LLM is responsible for choosing appropriate geometry types based on the user's intent:
   - For queries about "neighbourhoods", "areas", or "regions", polygons are usually appropriate
   - For queries about "locations", "sites", or "points of interest", points (centroids) may be appropriate
   - Either points or polygons can be acceptable depending on context

4. USING get_centroid: Use get_centroid to convert polygon geometries to point geometries (centroids). This is useful when:
   - You need point inputs for operations that require points
   - You want to represent polygons as their central point
   - You need to perform point-based operations on polygon results

5. GEOMETRY CONVERSION STRATEGY: Before performing spatial operations, consider whether the geometry types are compatible:
   - Point + Point intersection: likely null result
   - Polygon + Polygon intersection: standard polygon overlap
   - Point + Polygon intersection: points that fall within polygons
   - Use get_centroid to convert polygons to points when needed

The LLM must reason about geometry types and choose operations that will produce meaningful results for the user's query.

==================================================
OSM MODES REFERENCE
==================================================

roads:      highway column - primary, secondary, tertiary, motorway, residential, footway, cycleway, path, service, track, etc.
waterways:  waterway column - river, stream, canal, dam, waterfall, dock, drain, ditch, etc.
landuse:    landuse column - residential, commercial, industrial, farm, forest, grass, farmland, etc.
pois:       amenity column + name - restaurant, school, hospital, bank, pharmacy, cafe, etc.

==================================================
SYNTAX
==================================================

- One statement per line:
  varname = function(args)

- Variable names are short bare identifiers:
  a, b, c, region1, etc.

- String arguments use double quotes.

- Variable arguments are bare identifiers.

- Arguments are comma-separated.

- The final line MUST assign to exactly:
  output

- Write ONLY the program.

- No markdown fences.
- No comments.
- No explanation.
- No prose.
- No trailing text.
- No intercalling of fucntions inside fucntions like func1(func2) not evben tools. Each fucn is a different line.

Only use the functions listed above.

Your primary objective is semantic correctness. Do not blindly choose POI simply because a concept has a POI category. Decide whether the user wants a known/listed place, a structured OSM category, or the physical thing visible in imagery, and use multiple modalities when the query genuinely requires them.
"""


@dataclass
class PipelineStep:
    step_id: int
    operation: str
    parameters: dict
    inputs: list
    output_variable: str

    def __post_init__(self):
        if self.operation not in SUPPORTED_OPERATIONS:
            raise ValueError(
                f"Unsupported operation '{self.operation}' in step {self.step_id}"
            )
        if self.operation == "tool":
            action = self.parameters.get("target")
            if action not in SUPPORTED_TOOL_ACTIONS:
                raise ValueError(
                    f"Unsupported tool action '{action}' in step {self.step_id}"
                )
        if self.operation == "vision":
            resolution = self.parameters.get("resolution") or config.DEFAULT_RESOLUTION
            if resolution not in SUPPORTED_RESOLUTIONS:
                raise ValueError(
                    f"Unsupported resolution '{resolution}' in step {self.step_id}"
                )
            self.parameters["resolution"] = resolution
        if self.operation == "change":
            resolution = self.parameters.get("resolution") or config.DEFAULT_RESOLUTION
            if resolution not in SUPPORTED_RESOLUTIONS:
                raise ValueError(
                    f"Unsupported resolution '{resolution}' in step {self.step_id}"
                )
            self.parameters["resolution"] = resolution
            from_time = self.parameters.get("from_time")
            to_time = self.parameters.get("to_time")
            mode = self.parameters.get("mode", "new")
            if from_time not in SUPPORTED_TIME_PERIODS:
                raise ValueError(
                    f"Unsupported from_time '{from_time}' in step {self.step_id}. "
                    f"Must be one of {SUPPORTED_TIME_PERIODS}."
                )
            if to_time not in SUPPORTED_TIME_PERIODS:
                raise ValueError(
                    f"Unsupported to_time '{to_time}' in step {self.step_id}. "
                    f"Must be one of {SUPPORTED_TIME_PERIODS}."
                )
            if mode not in SUPPORTED_CHANGE_MODES:
                raise ValueError(
                    f"Unsupported change mode '{mode}' in step {self.step_id}. "
                    f"Must be one of {SUPPORTED_CHANGE_MODES}."
                )


@dataclass
class QueryPlan:
    steps: list  # list[PipelineStep], in execution order
    raw: dict = field(default_factory=dict)

    @property
    def final_variable(self) -> str:
        return self.steps[-1].output_variable if self.steps else None


# ------------------------------------------------------------------
# DSL parsing
#
# Lines look like:
#   a = geocode("Los Angeles")
#   b = demo(a, "poorer regions")
#   c = vision-high(a, "houses with pools")
#   output = intersection(c, b)
#
# Each function name maps to an internal (operation, resolution) pair that
# PipelineStep/executor.py already understand — the DSL is just a friendlier
# surface syntax over the same step model used before.
# ------------------------------------------------------------------

# function name -> (operation, resolution or None, tool-action or None)
_FUNC_MAP = {
    "geocode": ("geocode", None, None),
    "demo": ("demo", None, None),
    "osm": ("osm", None, None),
    "vision-high": ("vision", "high", None),
    "vision-low": ("vision", "low", None),
    "change-high": ("change", "high", None),
    "change-low": ("change", "low", None),
    "buffer": ("tool", None, "buffer"),
    "intersection": ("tool", None, "intersection"),
    "union": ("tool", None, "union"),
    "difference": ("tool", None, "difference"),
    "add": ("tool", None, "add"),
    "get_centroid": ("tool", None, "get_centroid"),
}

# Accept both `=` and `:` as the assignment separator so a model writing
# `output: a = ...` or `output = ...` both parse.
_LINE_RE = re.compile(
    r'^\s*(?P<var>[A-Za-z_]\w*)\s*[=:]\s*(?P<func>[A-Za-z_][A-Za-z_\-]*)\s*\((?P<args>.*)\)\s*$'
)
# Matches the head of a statement anywhere in the raw output (mid-prose,
# bullets, trailing junk, etc.). The full statement is extracted by
# `_scan_balanced_paren` so strings containing ')' are handled correctly.
_STMT_HEAD_RE = re.compile(r'([A-Za-z_]\w*)\s*[=:]\s*([A-Za-z_][A-Za-z_\-]*)\s*\(')
_NUMBER_RE = re.compile(r'^-?\d+(\.\d+)?$')

# Fuzz guard: how many arguments each function may legitimately take. Anything
# more is a strong signal the model garbled the statement, so reject it rather
# than silently feeding wrong inputs downstream. Keyed by (operation, action).
_FUZZ_MAX_ARGS = {
    ("geocode", None): 1,
    ("demo", None): 2,
    ("osm", None): 4,
    ("vision", None): 3,
    ("change", None): 6,
    ("change", "high"): 6,
    ("change", "low"): 6,
    ("tool", "buffer"): 2,
    ("tool", "get_centroid"): 1,
}


def _scan_balanced_paren(text: str, open_paren: int):
    """Scan forward from an opening paren, respecting quotes and nesting, and
    return (args_str, close_index) at the matching close paren. Returns
    (None, None) if the paren is never closed."""
    depth = 0
    quote = None
    for j in range(open_paren, len(text)):
        ch = text[j]
        if quote:
            if ch == quote:
                quote = None
        elif ch in ('"', "'"):
            quote = ch
        elif ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
            if depth == 0:
                return text[open_paren + 1:j], j
    return None, None


def _find_statements(text: str):
    """Single-capture pass: locate every `var = func(args)` statement anywhere
    in the raw output, reconstructing each as a clean line so the parser never
    depends on the model's line discipline."""
    statements = []
    i, n = 0, len(text)
    while i < n:
        m = _STMT_HEAD_RE.search(text, i)
        if not m:
            break
        open_paren = m.end() - 1
        args, close = _scan_balanced_paren(text, open_paren)
        if args is None:
            i = m.end()
            continue
        statements.append(f"{m.group(1).strip()} = {m.group(2).strip()}({args.strip()})")
        i = close + 1
    return statements


def _split_args(arg_str: str):
    """Split a DSL argument list on top-level commas, respecting quotes.
    Arguments never nest function calls in this grammar, so we only need to
    track whether we're inside a quoted string."""
    args, current, quote = [], [], None
    for ch in arg_str:
        if quote:
            current.append(ch)
            if ch == quote:
                quote = None
        elif ch in ('"', "'"):
            quote = ch
            current.append(ch)
        elif ch == ',':
            args.append(''.join(current))
            current = []
        else:
            current.append(ch)
    if current:
        args.append(''.join(current))
    return [a.strip() for a in args if a.strip()]


def _repair_token(token: str) -> str:
    """Repair pass: drop trailing punctuation that may have glued onto an arg
    (e.g. `"chicago",` -> `"chicago"`, `5.` -> `5`, `a,` -> `a`)."""
    token = token.strip()
    stripped = token.rstrip('.,;:!?') 
    return stripped if stripped else token


def _classify_arg(token: str):
    """Return ('text'|'number'|'var'|'null', value) for a single DSL argument, or
    None for junk that is neither a quoted string, a number, nor a bare
    identifier."""
    token = _repair_token(token)

    if len(token) >= 2 and token[0] == token[-1] and token[0] in ('"', "'"):
        return ('text', token[1:-1])

    # Auto-close an unclosed leading quote (model dropped the closing mark),
    # but only if it's the sole quote in the token.
    if token[0] in ('"', "'") and token.count(token[0]) == 1:
        return ('text', token[1:])

    if _NUMBER_RE.match(token):
        num = float(token)
        return ('number', int(num) if num.is_integer() else num)

    # Treat None/null as a special null value, not a variable reference
    if token.lower() in ('none', 'null'):
        return ('null', None)

    if re.fullmatch(r'[A-Za-z_]\w*', token):
        return ('var', token)

    return None


def _parse_dsl_line(line: str, step_id: int) -> PipelineStep:
    match = _LINE_RE.match(line)
    if not match:
        raise ValueError(f"Could not parse line as 'var = func(args)': {line!r}")

    var_name = match.group("var")
    func_name = match.group("func").lower()
    if func_name not in _FUNC_MAP:
        raise ValueError(f"Unsupported function '{func_name}' in line: {line!r}")

    operation, resolution, tool_action = _FUNC_MAP[func_name]
    # Drop junk tokens (neither quoted string, number, nor identifier).
    classified = [
        c for c in (_classify_arg(tok) for tok in _split_args(match.group("args")))
        if c is not None
    ]

    # Fuzz guard: reject statements with far more arguments than the function
    # allows, a strong sign the model garbled the line.
    fuzz_limit = _FUZZ_MAX_ARGS.get((operation, tool_action), 2)
    if len(classified) > fuzz_limit:
        raise ValueError(
            f"{func_name}() got {len(classified)} arguments, expected at most "
            f"{fuzz_limit}: {line!r}"
        )

    var_args = [v for kind, v in classified if kind == 'var']
    text_args = [v for kind, v in classified if kind == 'text']
    number_args = [v for kind, v in classified if kind == 'number']

    parameters = {"target": None, "resolution": None, "buffer_distance_km": None}

    if operation == "geocode":
        if not text_args:
            raise ValueError(f"geocode() needs a string place name: {line!r}")
        parameters["target"] = text_args[0]
        inputs = []

    elif operation == "change":
        # change(region?, query, from_time, to_time, mode?) or
        # change(region?, query1, query2, from_time, to_time, mode?)
        # The LLM may quote time/mode ("past", "present", "new", "removed") so they land in
        # text_args, or leave them bare so they land in var_args.
        # Strategy: pull region from var_args (if any), then find the
        # time periods and mode from text_args, and use remaining text_args as query/queries.
        if len(text_args) < 3:
            raise ValueError(
                f"change() needs query, from_time, to_time as strings: {line!r}"
            )
        
        # Find which text_args are time periods or mode
        time_args = [i for i, arg in enumerate(text_args) if arg.lower() in SUPPORTED_TIME_PERIODS]
        mode_args = [i for i, arg in enumerate(text_args) if arg.lower() in SUPPORTED_CHANGE_MODES]
        
        if len(time_args) < 2:
            raise ValueError(
                f"change() needs at least 2 time period arguments (from_time, to_time): {line!r}"
            )
        
        # The last two time args are from_time and to_time
        from_time_idx = time_args[-2]
        to_time_idx = time_args[-1]
        
        # Mode is optional, defaults to "new"
        if mode_args:
            mode_idx = mode_args[0]
            parameters["mode"] = text_args[mode_idx].lower()
        else:
            parameters["mode"] = "new"
        
        # Everything before from_time_idx is query(s)
        query_args = text_args[:from_time_idx]
        
        if len(query_args) == 0:
            raise ValueError(
                f"change() needs at least one query string: {line!r}"
            )
        elif len(query_args) == 1:
            # Single query - check for comma separation
            parameters["target"] = query_args[0]
        else:
            # Multiple queries - join with comma for the parser
            parameters["target"] = ",".join(query_args)
        
        parameters["from_time"] = text_args[from_time_idx]
        parameters["to_time"] = text_args[to_time_idx]
        inputs = var_args[:1] if var_args else []

    elif operation == "demo":
        if not text_args:
            raise ValueError(f"{func_name}() needs a string query: {line!r}")
        parameters["target"] = text_args[0]
        inputs = var_args  # zero or one region variable

    elif operation == "vision":
        if not text_args:
            raise ValueError(f"{func_name}() needs a string query: {line!r}")
        parameters["target"] = text_args[0]
        parameters["resolution"] = resolution
        # Optional time parameter (last text arg if it's a known time period)
        if len(text_args) >= 2:
            candidate = text_args[-1].lower()
            if candidate in SUPPORTED_TIME_PERIODS:
                parameters["time"] = candidate
        inputs = var_args  # zero or one region variable

    elif operation == "osm":
        # osm(region?, mode, query)
        # The LLM may write:
        #   osm("primary", "roads")           -> (query="primary", mode="roads")
        #   osm("roads", "primary")           -> (query="primary", mode="roads")
        #   osm(a, "rivers", "waterways")     -> region=a, query="rivers", mode="waterways"
        #
        # Strategy: detect which text_arg is the mode by matching against
        # SUPPORTED_MODES. The remaining text args become query.

        from operations.osm import SUPPORTED_MODES

        if len(text_args) < 1:
            raise ValueError(f"osm() needs at least a query string: {line!r}")

        # Separate region variable (if any) from text args
        inputs = var_args[:1] if var_args else []

        # Among text_args, find which one is the mode
        mode_idx = None
        for i, arg in enumerate(text_args):
            if arg.lower() in SUPPORTED_MODES:
                mode_idx = i
                break

        if mode_idx is not None:
            # Found a recognized mode
            parameters["osm_mode"] = text_args[mode_idx].lower()
            # Remaining text_args (excluding mode) are query and optional method
            remaining = [a for i, a in enumerate(text_args) if i != mode_idx]
            parameters["target"] = remaining[0] if remaining else None
            parameters["osm_method"] = remaining[1] if len(remaining) > 1 else "keyword"
        else:
            # No recognized mode found - use smart detection from osm.py
            # Assume (query, mode) order, let _resolve_mode_and_query sort it out
            if len(text_args) >= 2:
                parameters["target"] = text_args[0]
                parameters["osm_mode"] = text_args[1]
                parameters["osm_method"] = text_args[2] if len(text_args) > 2 else "keyword"
            else:
                # Only one arg - treat as query, default to roads mode
                parameters["target"] = text_args[0]
                parameters["osm_mode"] = "roads"
                parameters["osm_method"] = "keyword"

    elif operation == "tool" and tool_action == "buffer":
        if not var_args:
            raise ValueError(f"buffer() needs a region variable: {line!r}")
        if not number_args:
            raise ValueError(f"buffer() needs a distance in km: {line!r}")
        parameters["target"] = "buffer"
        parameters["buffer_distance_km"] = number_args[0]
        inputs = var_args[:1]

    elif operation == "tool" and tool_action == "get_centroid":
        if len(var_args) != 1:
            raise ValueError(f"get_centroid() needs exactly 1 variable argument: {line!r}")
        parameters["target"] = "get_centroid"
        inputs = var_args

    else:  # intersection / union / difference / add
        if len(var_args) != 2:
            raise ValueError(f"{func_name}() needs exactly 2 variable arguments: {line!r}")
        parameters["target"] = tool_action
        inputs = var_args

    return PipelineStep(
        step_id=step_id,
        operation=operation,
        parameters=parameters,
        inputs=inputs,
        output_variable=var_name,
    )


def _extract_dsl(raw_content: str) -> str:
    """Line-level filter. Keep only lines that look like a DSL statement,
    dropping prose, fences, comments, bullets, and blank lines regardless of
    where the model put them."""
    raw_content = raw_content.strip()
    fence_match = re.search(r"```(?:\w+)?\s*(.*?)\s*```", raw_content, flags=re.DOTALL)
    if fence_match:
        raw_content = fence_match.group(1).strip()
    return "\n".join(_find_statements(raw_content))


def _coerce_plan(dsl_text: str) -> QueryPlan:
    lines = [ln for ln in dsl_text.splitlines() if ln.strip()]
    if not lines:
        raise ValueError("Parsed plan contains no steps.")

    # Per-line error tolerance: skip malformed lines instead of failing the
    # whole plan, so one bad statement doesn't discard the good ones.
    steps = []
    for i, line in enumerate(lines):
        try:
            steps.append(_parse_dsl_line(line, step_id=i + 1))
        except Exception as e:
            print(f"Skipping unparsable DSL line {i + 1} ({e}): {line!r}")

    if not steps:
        # Graceful drop-down: only fall back if NO line produced a valid step.
        raise ValueError("No valid DSL lines survived filtering.")

    if steps[-1].output_variable.lower() != "output":
        # Be lenient: rename the last step's output rather than failing outright.
        steps[-1].output_variable = "output"
    steps[-1].output_variable = "FINAL_ANSWER"

    return QueryPlan(steps=steps, raw={"dsl": dsl_text})


def _fallback_plan(user_query: str) -> QueryPlan:
    """A degenerate single-step plan used if the local router fails entirely:
    treat the whole query as a vision search over the full study area."""
    step = PipelineStep(
        step_id=1,
        operation="vision",
        parameters={
            "target": user_query,
            "resolution": config.DEFAULT_RESOLUTION,
            "buffer_distance_km": None,
        },
        inputs=[],
        output_variable="FINAL_ANSWER",
    )
    return QueryPlan(steps=[step], raw={})


# ------------------------------------------------------------------
# Gemini router — lazy-initialized client + cached system instruction
# ------------------------------------------------------------------

_genai_client = None
_router_cache = None


def _ensure_genai():
    """Lazily create the GenAI client and cache the system instruction.
    Called on first router_lm() invocation, after the API key is in env."""
    global _genai_client, _router_cache
    if _genai_client is not None:
        return
    _genai_client = genai.Client()
    _create_router_cache()


def _create_router_cache():
    """(Re)create the cached system instruction. The cache has a 1-hour TTL
    (Google's minimum-billing-friendly window) — for a server that stays up
    longer than that, the cached_content reference goes stale and Google's
    API rejects it. Call this both on first init and again whenever a
    cached-content request comes back with an error, so the cache silently
    refreshes instead of taking every query down after an hour of uptime."""
    global _router_cache
    try:
        _router_cache = _genai_client.caches.create(
            model="gemini-3.1-flash-lite",
            config={"system_instruction": ROUTER_SYSTEM_PROMPT, "ttl": "3600s"},
        )
        print(f"[router] System instruction cached: {_router_cache.name}")
    except Exception as e:
        print(f"[router] Could not cache system instruction ({e}), falling back to per-request.")
        _router_cache = None


def router_lm(user_query: str):
    _ensure_genai()

    def _uncached_call():
        return _genai_client.models.generate_content(
            model="gemini-3.1-flash-lite",
            config={"system_instruction": ROUTER_SYSTEM_PROMPT},
            contents=f"QUERY: {user_query}",
        )

    if _router_cache is None:
        return _uncached_call().text

    try:
        response = _genai_client.models.generate_content(
            model="gemini-3.1-flash-lite",
            config={"cached_content": _router_cache.name},
            contents=f"QUERY: {user_query}",
        )
        return response.text
    except Exception as e:
        # Most likely cause: the cache's 1-hour TTL expired mid-session.
        # Refresh it once and retry via the (now-fresh) cache; if that
        # somehow fails too, fall through to an uncached call so this
        # single query still succeeds rather than taking the whole router
        # down until a restart.
        print(f"[router] Cached-content call failed ({e}); refreshing cache and retrying.")
        _create_router_cache()
        if _router_cache is not None:
            try:
                response = _genai_client.models.generate_content(
                    model="gemini-3.1-flash-lite",
                    config={"cached_content": _router_cache.name},
                    contents=f"QUERY: {user_query}",
                )
                return response.text
            except Exception as e2:
                print(f"[router] Retry with refreshed cache also failed ({e2}); falling back to uncached call.")
        return _uncached_call().text

def parse_query(user_query: str) -> QueryPlan:
    """Parse a natural-language geospatial query into an executable QueryPlan."""
    raw_content = router_lm(user_query)

    print(raw_content)

    dsl_text = _extract_dsl(raw_content)
    if not dsl_text:
        print(f"Router produced no parseable program. Falling back to a single vision step. Raw output:\n{raw_content}")
        return _fallback_plan(user_query)

    try:
        return _coerce_plan(dsl_text)
    except Exception as e:
        print(f"Failed to parse DSL plan ({e}). Falling back to a single vision step.")
        return _fallback_plan(user_query)