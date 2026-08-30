import geopandas as gpd
import os


# ============================================================
# DisasterSaver - Level 6.1
# Hazard-Aware Road Preparation
# ============================================================

print("=" * 65)
print("DisasterSaver - Level 6.1")
print("Hazard-Aware Road Preparation")
print("=" * 65)


from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

ROADS_FILE = BASE_DIR / "data" / "raw" / "roads.geojson"
FLOOD_FILE = BASE_DIR / "data" / "raw" / "predicted_flood.geojson"
OUTPUT_FILE = BASE_DIR / "outputs" / "hazard_aware_roads.geojson"


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading roads...")

if not os.path.exists(ROADS_FILE):
    raise SystemExit(
        f"ERROR: Roads file not found:\n{ROADS_FILE}"
    )

roads = gpd.read_file(ROADS_FILE)

print(
    f"Roads loaded: {len(roads)}"
)


print("\nLoading predicted flood zones...")

if not os.path.exists(FLOOD_FILE):
    raise SystemExit(
        f"ERROR: Flood file not found:\n{FLOOD_FILE}"
    )

floods = gpd.read_file(FLOOD_FILE)

print(
    f"Flood zones loaded: {len(floods)}"
)


# ============================================================
# CHECK CRS
# ============================================================

if roads.crs is None:
    raise SystemExit(
        "ERROR: Roads have no CRS."
    )

if floods.crs is None:
    raise SystemExit(
        "ERROR: Flood zones have no CRS."
    )


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_road_columns = [
    "road_id",
    "highway_type",
    "road_length_m",
    "geometry"
]

required_flood_columns = [
    "zone_id",
    "risk_score",
    "risk_class",
    "geometry"
]


for column in required_road_columns:

    if column not in roads.columns:

        raise SystemExit(
            f"ERROR: Missing road column: {column}"
        )


for column in required_flood_columns:

    if column not in floods.columns:

        raise SystemExit(
            f"ERROR: Missing flood column: {column}"
        )


# ============================================================
# PROJECT TO METRIC CRS
# ============================================================

print("\nConverting to metric CRS...")

roads_metric = roads.to_crs(
    "EPSG:32644"
)

floods_metric = floods.to_crs(
    "EPSG:32644"
)


# ============================================================
# PREPARE OUTPUT COLUMNS
# ============================================================

roads_metric["flooded_length_m"] = 0.0

roads_metric["flood_exposure_ratio"] = 0.0

roads_metric["flood_risk_score"] = 0.0

roads_metric["risk_class"] = "SAFE"

roads_metric["flood_zone_id"] = "NONE"

roads_metric["flooded"] = 0


# ============================================================
# FLOOD EXPOSURE CALCULATION
# ============================================================

print("\nCalculating flood exposure...")


for road_index, road in roads_metric.iterrows():

    road_geometry = road.geometry

    road_length = road_geometry.length


    if road_length <= 0:
        continue


    best_risk = 0.0

    best_risk_class = "SAFE"

    best_zone = "NONE"

    total_flooded_length = 0.0


    # --------------------------------------------------------
    # Check every flood zone
    # --------------------------------------------------------

    for _, flood in floods_metric.iterrows():

        flood_geometry = flood.geometry


        if not road_geometry.intersects(
            flood_geometry
        ):
            continue


        intersection = (
            road_geometry.intersection(
                flood_geometry
            )
        )


        # ----------------------------------------------------
        # Intersection may be empty
        # ----------------------------------------------------

        if intersection.is_empty:
            continue


        intersection_length = (
            intersection.length
        )


        if intersection_length <= 0:
            continue


        total_flooded_length += (
            intersection_length
        )


        risk = float(
            flood["risk_score"]
        )


        # Keep highest-risk zone
        if risk > best_risk:

            best_risk = risk

            best_risk_class = str(
                flood["risk_class"]
            )

            best_zone = str(
                flood["zone_id"]
            )


    # --------------------------------------------------------
    # Cap flooded length at road length
    # --------------------------------------------------------

    total_flooded_length = min(
        total_flooded_length,
        road_length
    )


    # --------------------------------------------------------
    # Flood exposure ratio
    # --------------------------------------------------------

    exposure_ratio = (
        total_flooded_length /
        road_length
    )


    # --------------------------------------------------------
    # Store results
    # --------------------------------------------------------

    roads_metric.at[
        road_index,
        "flooded_length_m"
    ] = total_flooded_length


    roads_metric.at[
        road_index,
        "flood_exposure_ratio"
    ] = exposure_ratio


    roads_metric.at[
        road_index,
        "flood_risk_score"
    ] = best_risk


    roads_metric.at[
        road_index,
        "risk_class"
    ] = best_risk_class


    roads_metric.at[
        road_index,
        "flood_zone_id"
    ] = best_zone


    if total_flooded_length > 0:

        roads_metric.at[
            road_index,
            "flooded"
        ] = 1


# ============================================================
# CONVERT BACK TO EPSG:4326
# ============================================================

roads_output = roads_metric.to_crs(
    "EPSG:4326"
)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 65)
print("HAZARD-AWARE ROAD RESULTS")
print("=" * 65)


for _, road in roads_output.iterrows():

    print(
        f"{road['road_id']} | "
        f"type={road['highway_type']} | "
        f"length={road['road_length_m']:.1f} m | "
        f"flooded_length="
        f"{road['flooded_length_m']:.1f} m | "
        f"exposure="
        f"{road['flood_exposure_ratio']:.2f} | "
        f"risk="
        f"{road['flood_risk_score']:.2f} | "
        f"zone={road['flood_zone_id']}"
    )


# ============================================================
# SAVE
# ============================================================

os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)


roads_output.to_file(
    OUTPUT_FILE,
    driver="GeoJSON"
)


print("\nSaved:")
print(OUTPUT_FILE)

print(
    "\nLevel 6.1 completed successfully."
)