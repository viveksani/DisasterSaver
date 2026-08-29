import geopandas as gpd
import os


# ============================================================
# DisasterSaver - Level 6.2
# Hazard-Aware Routing Cost
# ============================================================

print("=" * 60)
print("DisasterSaver - Level 6.2")
print("Hazard-Aware Routing Cost")
print("=" * 60)


INPUT_FILE = r"C:\DisasterSaver\outputs\hazard_aware_roads.geojson"

OUTPUT_FILE = r"C:\DisasterSaver\outputs\routing_roads.geojson"


# ============================================================
# LOAD
# ============================================================

print("\nLoading hazard-aware roads...")


if not os.path.exists(INPUT_FILE):

    raise SystemExit(
        f"ERROR: Input file not found:\n{INPUT_FILE}"
    )


roads = gpd.read_file(
    INPUT_FILE
)


print(
    f"Roads loaded: {len(roads)}"
)


# ============================================================
# CHECK
# ============================================================

required_columns = [
    "road_id",
    "highway_type",
    "road_length_m",
    "flooded",
    "flood_risk_score",
    "flood_exposure_ratio",
    "risk_class",
    "geometry"
]


print("\nChecking required columns...")


missing = [
    column
    for column in required_columns
    if column not in roads.columns
]


if missing:

    print("\nERROR: Missing columns:")

    for column in missing:
        print(f" - {column}")

    raise SystemExit(
        "Required columns are missing."
    )


print(
    "All required columns found."
)


# ============================================================
# SPEED VALUES
# ============================================================

SPEED_KMH = {

    "primary": 50.0,

    "secondary": 40.0,

    "tertiary": 35.0,

    "residential": 30.0,

    "service": 20.0
}


# ============================================================
# HAZARD PENALTY
# ============================================================

RISK_MULTIPLIER = {

    "SAFE": 1.0,

    "LOW": 1.5,

    "MEDIUM": 2.5,

    "HIGH": 5.0,

    "VERY_HIGH": 10.0
}


# ============================================================
# CALCULATE COST
# ============================================================

print("\nCalculating routing costs...")


routing_costs = []


for _, road in roads.iterrows():

    road_length_m = float(
        road["road_length_m"]
    )


    highway_type = str(
        road["highway_type"]
    )


    risk_class = str(
        road["risk_class"]
    )


    flood_risk = float(
        road["flood_risk_score"]
    )


    exposure_ratio = float(
        road["flood_exposure_ratio"]
    )


    # --------------------------------------------------------
    # Speed
    # --------------------------------------------------------

    speed_kmh = SPEED_KMH.get(
        highway_type,
        30.0
    )


    # --------------------------------------------------------
    # Base travel time
    # --------------------------------------------------------

    travel_time_min = (

        road_length_m / 1000.0

    ) / speed_kmh * 60.0


    # --------------------------------------------------------
    # Hazard multiplier
    # --------------------------------------------------------

    hazard_multiplier = (
        RISK_MULTIPLIER.get(
            risk_class,
            1.0
        )
    )


    # --------------------------------------------------------
    # Exposure-aware hazard penalty
    # --------------------------------------------------------

    hazard_penalty = (

        1.0 +

        (
            exposure_ratio *
            (hazard_multiplier - 1.0)
        )

    )


    # --------------------------------------------------------
    # Routing cost
    # --------------------------------------------------------

    routing_cost = (

        travel_time_min *
        hazard_penalty

    )


    routing_costs.append(
        routing_cost
    )


roads["routing_cost"] = (
    routing_costs
)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 60)
print("ROUTING COST RESULTS")
print("=" * 60)


for _, road in roads.iterrows():

    print(

        f"{road['road_id']} | "

        f"type={road['highway_type']} | "

        f"length="
        f"{road['road_length_m']:.1f} m | "

        f"exposure="
        f"{road['flood_exposure_ratio']:.2f} | "

        f"risk="
        f"{road['flood_risk_score']:.2f} | "

        f"flooded="
        f"{road['flooded']} | "

        f"speed="
        f"{SPEED_KMH.get(road['highway_type'], 30.0):.1f} km/h | "

        f"time="
        f"{road_length_m / 1000.0 / SPEED_KMH.get(road['highway_type'], 30.0) * 60.0:.2f} min | "

        f"cost="
        f"{road['routing_cost']:.2f}"
    )


# ============================================================
# SAVE
# ============================================================

roads.to_file(
    OUTPUT_FILE,
    driver="GeoJSON"
)


print("\n" + "=" * 60)
print("OUTPUT")
print("=" * 60)

print(
    "Routing roads saved:"
)

print(
    OUTPUT_FILE
)


print(
    "\nLevel 6.2 completed successfully."
)