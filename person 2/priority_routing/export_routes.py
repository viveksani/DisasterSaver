import geopandas as gpd
import networkx as nx
import json
import os

from shapely.geometry import LineString, Point
from pyproj import Transformer


# ============================================================
# DisasterSaver - Level 6.5
# Export Final Hazard-Aware Routes
# ============================================================

print("=" * 65)
print("DisasterSaver - Level 6.5")
print("Exporting Final Routes")
print("=" * 65)


CALCULATED_ROUTES_FILE = (
    r"C:\DisasterSaver\outputs\calculated_routes.json"
)

GRAPH_FILE = (
    r"C:\DisasterSaver\outputs\road_network.graphml"
)

OUTPUT_FILE = (
    r"C:\DisasterSaver\outputs\routes.geojson"
)


# ============================================================
# LOAD CALCULATED ROUTES
# ============================================================

print("\nLoading calculated routes...")

if not os.path.exists(
    CALCULATED_ROUTES_FILE
):

    raise SystemExit(
        "ERROR: calculated_routes.json not found."
    )


with open(
    CALCULATED_ROUTES_FILE,
    "r",
    encoding="utf-8"
) as f:

    calculated_routes = json.load(f)


print(
    f"Calculated routes loaded: "
    f"{len(calculated_routes)}"
)


# ============================================================
# LOAD GRAPH
# ============================================================

print("\nLoading road network...")

if not os.path.exists(
    GRAPH_FILE
):

    raise SystemExit(
        "ERROR: road_network.graphml not found."
    )


G = nx.read_graphml(
    GRAPH_FILE
)


print(
    f"Nodes: {G.number_of_nodes()}"
)

print(
    f"Edges: {G.number_of_edges()}"
)


# ============================================================
# GRAPH CRS
# ============================================================

# Level 6.3 creates the graph in EPSG:32644.
GRAPH_CRS = "EPSG:32644"

# Final GeoJSON must use EPSG:4326.
OUTPUT_CRS = "EPSG:4326"


print(
    f"\nGraph CRS: {GRAPH_CRS}"
)

print(
    f"Output CRS: {OUTPUT_CRS}"
)


# ============================================================
# COORDINATE TRANSFORMATION
# ============================================================

transformer = Transformer.from_crs(
    GRAPH_CRS,
    OUTPUT_CRS,
    always_xy=True
)


# ============================================================
# NODE COORDINATES
# ============================================================

node_coordinates = {}

for node in G.nodes:

    x = float(
        G.nodes[node]["x"]
    )

    y = float(
        G.nodes[node]["y"]
    )

    # Convert UTM -> longitude/latitude
    lon, lat = transformer.transform(
        x,
        y
    )

    node_coordinates[node] = (
        lon,
        lat
    )


# ============================================================
# EXPORT ROUTES
# ============================================================

print("\n" + "=" * 65)
print("EXPORTING ROUTES")
print("=" * 65)


records = []


for route in calculated_routes:

    route_id = str(
        route["route_id"]
    )

    from_id = str(
        route["from_id"]
    )

    to_id = str(
        route["to_id"]
    )

    distance_km = float(
        route["distance_km"]
    )

    route_risk_score = float(
        route["route_risk_score"]
    )

    estimated_time_min = float(
        route["estimated_time_min"]
    )

    flooded_edges_count = int(
        route["flooded_edges_count"]
    )

    path = route.get(
        "path",
        []
    )


    # --------------------------------------------------------
    # Validate path
    # --------------------------------------------------------

    if len(path) < 2:

        print(
            f"WARNING: {route_id} "
            f"has an invalid path."
        )

        continue


    # --------------------------------------------------------
    # Convert graph nodes to WGS84 coordinates
    # --------------------------------------------------------

    coordinates = []

    for node in path:

        if node not in node_coordinates:

            raise SystemExit(
                f"ERROR: Node {node} "
                f"not found in graph."
            )

        coordinates.append(
            node_coordinates[node]
        )


    # --------------------------------------------------------
    # Create LineString
    # --------------------------------------------------------

    geometry = LineString(
        coordinates
    )


    # --------------------------------------------------------
    # Add final record
    # --------------------------------------------------------

    records.append(
        {
            "route_id":
                route_id,

            "from_id":
                from_id,

            "to_id":
                to_id,

            "distance_km":
                round(
                    distance_km,
                    2
                ),

            "route_risk_score":
                round(
                    route_risk_score,
                    2
                ),

            "estimated_time_min":
                round(
                    estimated_time_min,
                    1
                ),

            "flooded_edges_count":
                flooded_edges_count,

            "geometry":
                geometry
        }
    )


    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print(
        f"{route_id} | "
        f"{from_id} -> {to_id} | "
        f"{distance_km:.2f} km | "
        f"risk={route_risk_score:.2f} | "
        f"time={estimated_time_min:.1f} min | "
        f"flooded={flooded_edges_count}"
    )


# ============================================================
# CHECK
# ============================================================

if not records:

    raise SystemExit(
        "ERROR: No valid routes generated."
    )


# ============================================================
# CREATE GEODATAFRAME
# ============================================================

routes = gpd.GeoDataFrame(
    records,
    geometry="geometry",
    crs=OUTPUT_CRS
)


# ============================================================
# EXACT OUTPUT COLUMN ORDER
# ============================================================

routes = routes[
    [
        "route_id",
        "from_id",
        "to_id",
        "distance_km",
        "route_risk_score",
        "estimated_time_min",
        "flooded_edges_count",
        "geometry"
    ]
]


# ============================================================
# SAVE
# ============================================================

os.makedirs(
    os.path.dirname(
        OUTPUT_FILE
    ),
    exist_ok=True
)


routes.to_file(
    OUTPUT_FILE,
    driver="GeoJSON"
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 65)
print("FINAL ROUTES")
print("=" * 65)


print(
    routes[
        [
            "route_id",
            "from_id",
            "to_id",
            "distance_km",
            "route_risk_score",
            "estimated_time_min",
            "flooded_edges_count"
        ]
    ].to_string(
        index=False
    )
)


print("\nSaved:")
print(OUTPUT_FILE)

print(
    "\nLevel 6.5 completed successfully."
)