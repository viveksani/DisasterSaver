import geopandas as gpd
import networkx as nx
import os

from shapely.geometry import LineString


# ============================================================
# DisasterSaver - Level 6.3
# Building Hazard-Aware Road Network
# ============================================================

print("=" * 65)
print("DisasterSaver - Level 6.3")
print("Building Hazard-Aware Road Network")
print("=" * 65)


# ============================================================
# FILE PATHS
# ============================================================

INPUT_FILE = r"C:\DisasterSaver\outputs\routing_roads.geojson"

OUTPUT_FILE = r"C:\DisasterSaver\outputs\road_network.graphml"


# ============================================================
# LOAD ROUTING ROADS
# ============================================================

print("\nLoading routing roads...")

if not os.path.exists(INPUT_FILE):

    raise SystemExit(
        f"\nERROR: Input file not found:\n{INPUT_FILE}"
    )


roads = gpd.read_file(INPUT_FILE)

print(
    f"Road segments loaded: {len(roads)}"
)


# ============================================================
# CHECK CRS
# ============================================================

print("\nChecking CRS...")

print(
    f"Input CRS: {roads.crs}"
)

if roads.crs is None:

    raise SystemExit(
        "\nERROR: Input roads do not have a CRS."
    )


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = [
    "road_id",
    "highway_type",
    "road_length_m",
    "routing_cost",
    "flooded",
    "flood_risk_score",
    "risk_class",
    "geometry"
]

print("\nChecking required columns...")


missing_columns = [
    column
    for column in required_columns
    if column not in roads.columns
]


if missing_columns:

    print("\nERROR: Missing columns:")

    for column in missing_columns:
        print(f" - {column}")

    print("\nAvailable columns:")
    print(list(roads.columns))

    raise SystemExit(
        "\nRequired columns are missing."
    )


print(
    "All required columns found."
)


# ============================================================
# CHECK GEOMETRIES
# ============================================================

print("\nChecking road geometries...")


if roads.geometry.isnull().any():

    raise SystemExit(
        "\nERROR: Some road geometries are empty."
    )


if (~roads.geometry.is_valid).any():

    raise SystemExit(
        "\nERROR: Some road geometries are invalid."
    )


print(
    "All road geometries are valid."
)


# ============================================================
# PROJECT TO METRIC CRS
# ============================================================

print("\nProjecting road network to EPSG:32644...")

roads_metric = roads.to_crs(
    "EPSG:32644"
)

print(
    f"Metric CRS: {roads_metric.crs}"
)


# ============================================================
# CREATE NETWORKX GRAPH
# ============================================================

print("\nCreating NetworkX graph...")


G = nx.Graph()


# ============================================================
# PROCESS EACH ROAD
# ============================================================

for _, road in roads_metric.iterrows():

    road_id = str(
        road["road_id"]
    )

    highway_type = str(
        road["highway_type"]
    )

    routing_cost = float(
        road["routing_cost"]
    )

    flooded = int(
        road["flooded"]
    )

    flood_risk_score = float(
        road["flood_risk_score"]
    )

    risk_class = str(
        road["risk_class"]
    )


    # --------------------------------------------------------
    # ROAD GEOMETRY
    # --------------------------------------------------------

    geometry = road.geometry


    if geometry.geom_type != "LineString":

        print(
            f"WARNING: {road_id} is not a LineString."
        )

        continue


    coordinates = list(
        geometry.coords
    )


    if len(coordinates) < 2:

        print(
            f"WARNING: {road_id} has insufficient coordinates."
        )

        continue


    # --------------------------------------------------------
    # ACTUAL ROAD LENGTH FROM GEOMETRY
    #
    # Because roads_metric is EPSG:32644,
    # geometry.length is measured in METERS.
    # --------------------------------------------------------

    total_geometry_length = (
        geometry.length
    )


    if total_geometry_length <= 0:

        print(
            f"WARNING: {road_id} has zero geometry length."
        )

        continue


    # --------------------------------------------------------
    # SPLIT LINESTRING INTO INDIVIDUAL GRAPH EDGES
    # --------------------------------------------------------

    for i in range(
        len(coordinates) - 1
    ):

        start = coordinates[i]

        end = coordinates[i + 1]


        # ----------------------------------------------------
        # UNIQUE NODE IDs
        # ----------------------------------------------------

        start_node = (
            f"{start[0]:.3f},"
            f"{start[1]:.3f}"
        )

        end_node = (
            f"{end[0]:.3f},"
            f"{end[1]:.3f}"
        )


        # ----------------------------------------------------
        # SEGMENT GEOMETRY
        # ----------------------------------------------------

        segment = LineString(
            [
                start,
                end
            ]
        )


        # ----------------------------------------------------
        # ACTUAL SEGMENT LENGTH IN METERS
        # ----------------------------------------------------

        segment_length = (
            segment.length
        )


        if segment_length <= 0:

            continue


        # ----------------------------------------------------
        # PROPORTION OF ORIGINAL ROAD
        # ----------------------------------------------------

        proportion = (
            segment_length /
            total_geometry_length
        )


        # ----------------------------------------------------
        # ALLOCATE ROUTING COST
        # ----------------------------------------------------

        segment_routing_cost = (
            routing_cost *
            proportion
        )


        # ----------------------------------------------------
        # ADD START NODE
        # ----------------------------------------------------

        G.add_node(

            start_node,

            x=float(start[0]),

            y=float(start[1])
        )


        # ----------------------------------------------------
        # ADD END NODE
        # ----------------------------------------------------

        G.add_node(

            end_node,

            x=float(end[0]),

            y=float(end[1])
        )


        # ----------------------------------------------------
        # ADD ROAD EDGE
        # ----------------------------------------------------

        G.add_edge(

            start_node,

            end_node,

            road_id=road_id,

            highway_type=highway_type,

            length_m=float(
                segment_length
            ),

            routing_cost=float(
                segment_routing_cost
            ),

            flooded=int(
                flooded
            ),

            flood_risk_score=float(
                flood_risk_score
            ),

            risk_class=risk_class
        )


# ============================================================
# NETWORK STATISTICS
# ============================================================

print("\n" + "=" * 65)
print("NETWORK STATISTICS")
print("=" * 65)


print(
    f"Nodes : {G.number_of_nodes()}"
)

print(
    f"Edges : {G.number_of_edges()}"
)


# ============================================================
# CONNECTED COMPONENTS
# ============================================================

components = list(
    nx.connected_components(G)
)


print(
    f"Connected components : "
    f"{len(components)}"
)


if len(components) > 1:

    print("\nWARNING:")

    print(
        "The road network is disconnected."
    )


    for i, component in enumerate(
        components,
        start=1
    ):

        print(
            f"Component {i}: "
            f"{len(component)} node(s)"
        )


else:

    print(
        "\nRoad network is connected."
    )


# ============================================================
# NETWORK EDGES
# ============================================================

print("\n" + "=" * 65)
print("NETWORK EDGES")
print("=" * 65)


for u, v, data in G.edges(
    data=True
):

    print(
        f"{data['road_id']} | "
        f"{data['highway_type']} | "
        f"length="
        f"{data['length_m']:.2f} m | "
        f"risk="
        f"{data['flood_risk_score']:.2f} | "
        f"flooded="
        f"{data['flooded']} | "
        f"cost="
        f"{data['routing_cost']:.2f}"
    )


# ============================================================
# SAVE GRAPH
# ============================================================

print("\nSaving graph...")


os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)


nx.write_graphml(
    G,
    OUTPUT_FILE
)


# ============================================================
# FINISHED
# ============================================================

print("\nGraph saved:")

print(
    OUTPUT_FILE
)


print(
    "\nLevel 6.3 completed successfully."
)