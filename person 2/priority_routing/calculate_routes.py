import geopandas as gpd
import networkx as nx
import os
import math


# ============================================================
# DisasterSaver - Level 6.4
# Hazard-Aware Route Calculation
# ============================================================

print("=" * 60)
print("DisasterSaver - Level 6.4")
print("Hazard-Aware Route Calculation")
print("=" * 60)


GRAPH_FILE = (
    r"C:\DisasterSaver\outputs\road_network.graphml"
)

BASES_FILE = (
    r"C:\DisasterSaver\data\raw\emergency_bases.geojson"
)

PLACES_FILE = (
    r"C:\DisasterSaver\outputs\priority_places.geojson"
)

OUTPUT_FILE = (
    r"C:\DisasterSaver\outputs\calculated_routes.json"
)


# ============================================================
# LOAD GRAPH
# ============================================================

print("\nLoading network...")


if not os.path.exists(GRAPH_FILE):

    raise SystemExit(
        f"ERROR: Graph not found:\n{GRAPH_FILE}"
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
# LOAD BASES
# ============================================================

print("\nLoading emergency bases...")


bases = gpd.read_file(
    BASES_FILE
)


print(
    f"Bases: {len(bases)}"
)


# ============================================================
# LOAD PLACES
# ============================================================

print("\nLoading priority places...")


places = gpd.read_file(
    PLACES_FILE
)


print(
    f"Places: {len(places)}"
)


# ============================================================
# PROJECT BASES AND PLACES
# TO SAME CRS AS GRAPH
# ============================================================

print(
    "\nProjecting bases and places "
    "to EPSG:32644..."
)


bases_metric = bases.to_crs(
    "EPSG:32644"
)


places_metric = places.to_crs(
    "EPSG:32644"
)


# ============================================================
# SELECT ROUTING PRIORITY
# ============================================================

priority_classes = [
    "CRITICAL",
    "HIGH",
    "MEDIUM"
]


places_metric = places_metric[
    places_metric["priority_class"].isin(
        priority_classes
    )
].copy()


print(
    f"Routing priority places: "
    f"{len(places_metric)}"
)


# ============================================================
# SELECT PRIMARY BASE
# ============================================================

primary_base = bases_metric.loc[
    bases_metric["team_count"].idxmax()
]


PRIMARY_BASE_ID = str(
    primary_base["base_id"]
)


print(
    f"\nPrimary response base: "
    f"{PRIMARY_BASE_ID}"
)


print(
    f"Team count: "
    f"{primary_base['team_count']}"
)


# ============================================================
# GRAPH NODE COORDINATES
# ============================================================

node_coordinates = {}


for node in G.nodes:

    node_coordinates[node] = (

        float(
            G.nodes[node]["x"]
        ),

        float(
            G.nodes[node]["y"]
        )
    )


# ============================================================
# NEAREST GRAPH NODE
# ============================================================

def nearest_node(point):

    px = point.x
    py = point.y

    nearest = None

    minimum_distance = float(
        "inf"
    )


    for node, coords in (
        node_coordinates.items()
    ):

        x, y = coords


        distance = math.sqrt(

            (px - x) ** 2 +

            (py - y) ** 2

        )


        if distance < minimum_distance:

            minimum_distance = distance

            nearest = node


    return nearest


# ============================================================
# ROUTE METRICS
# ============================================================

def calculate_metrics(path):

    distance_m = 0.0

    time_min = 0.0

    risk_weighted_sum = 0.0

    risk_length_total = 0.0

    flooded_edges_count = 0


    SPEED_KMH = {

        "primary": 50.0,

        "secondary": 40.0,

        "tertiary": 35.0,

        "residential": 30.0,

        "service": 20.0
    }


    for u, v in zip(
        path[:-1],
        path[1:]
    ):

        edge = G.get_edge_data(
            u,
            v
        )


        if edge is None:
            continue


        length_m = float(
            edge["length_m"]
        )


        highway_type = str(
            edge.get(
                "highway_type",
                "tertiary"
            )
        )


        risk = float(
            edge.get(
                "flood_risk_score",
                0.0
            )
        )


        flooded = int(
            edge.get(
                "flooded",
                0
            )
        )


        distance_m += length_m


        speed_kmh = SPEED_KMH.get(
            highway_type,
            30.0
        )


        if flooded:

            flooded_edges_count += 1

            speed_kmh *= 0.75


        time_min += (

            length_m / 1000.0

        ) / speed_kmh * 60.0


        # ----------------------------------------------------
        # Length-weighted route risk
        # ----------------------------------------------------

        risk_weighted_sum += (
            risk * length_m
        )

        risk_length_total += length_m


    if risk_length_total > 0:

        route_risk_score = (

            risk_weighted_sum /
            risk_length_total

        )

    else:

        route_risk_score = 0.0


    return (

        distance_m,

        route_risk_score,

        time_min,

        flooded_edges_count

    )


# ============================================================
# CALCULATE ROUTES
# ============================================================

print("\n" + "=" * 60)
print("CALCULATING ROUTES")
print("=" * 60)


routes = []


route_counter = 1


for _, place in places_metric.iterrows():

    place_id = str(
        place["place_id"]
    )


    # --------------------------------------------------------
    # Find graph nodes
    # --------------------------------------------------------

    base_node = nearest_node(
        primary_base.geometry
    )


    place_node = nearest_node(
        place.geometry
    )


    print(
        f"\n{PRIMARY_BASE_ID} -> {place_id}"
    )


    print(
        f"Base graph node: "
        f"{base_node}"
    )


    print(
        f"Place graph node: "
        f"{place_node}"
    )


    # --------------------------------------------------------
    # Check whether base/place mapped
    # --------------------------------------------------------

    if base_node is None:

        print(
            "ERROR: Could not map base."
        )

        continue


    if place_node is None:

        print(
            "ERROR: Could not map place."
        )

        continue


    # --------------------------------------------------------
    # Check connectivity
    # --------------------------------------------------------

    if not nx.has_path(
        G,
        base_node,
        place_node
    ):

        print(
            "NO PATH"
        )

        continue


    # --------------------------------------------------------
    # Hazard-aware shortest path
    # --------------------------------------------------------

    path = nx.shortest_path(

        G,

        source=base_node,

        target=place_node,

        weight="routing_cost"

    )


    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    (
        distance_m,
        route_risk_score,
        estimated_time_min,
        flooded_edges_count
    ) = calculate_metrics(
        path
    )


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    route = {

        "route_id":
            f"RT{route_counter:06d}",

        "from_id":
            PRIMARY_BASE_ID,

        "to_id":
            place_id,

        "distance_km":
            round(
                distance_m / 1000.0,
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

        "path":
            path
    }


    routes.append(
        route
    )


    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print(
        f"Route: "
        f"{route['route_id']}"
    )


    print(
        f"Distance: "
        f"{route['distance_km']:.2f} km"
    )


    print(
        f"Risk: "
        f"{route['route_risk_score']:.2f}"
    )


    print(
        f"Time: "
        f"{route['estimated_time_min']:.1f} min"
    )


    print(
        f"Flooded edges: "
        f"{route['flooded_edges_count']}"
    )


    route_counter += 1


# ============================================================
# SAVE
# ============================================================

import json


with open(
    OUTPUT_FILE,
    "w"
) as f:

    json.dump(
        routes,
        f,
        indent=2
    )


print("\nTemporary route data saved:")

print(
    OUTPUT_FILE
)


print(
    "\nLevel 6.4 completed successfully."
)