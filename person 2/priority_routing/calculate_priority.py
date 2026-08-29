import geopandas as gpd
import numpy as np
from pathlib import Path


# ============================================
# DisasterSaver - Level 5
# Calculate Priority Places
# ============================================

# -----------------------------
# File paths
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

CRITICAL_PLACES_FILE = BASE_DIR / "data" / "raw" / "critical_places.geojson"
FLOOD_FILE = BASE_DIR / "data" / "raw" / "predicted_flood.geojson"

OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_FILE = OUTPUT_DIR / "priority_places.geojson"


# -----------------------------
# Place criticality
# -----------------------------

PLACE_CRITICALITY = {
    "hospital": 1.00,
    "fire_station": 0.90,
    "shelter": 0.85,
}


# -----------------------------
# Load data
# -----------------------------

print("=" * 50)
print("DisasterSaver - Level 5")
print("Priority Place Calculation")
print("=" * 50)

print("\nLoading critical places...")
places = gpd.read_file(CRITICAL_PLACES_FILE)

print("Loading predicted flood zones...")
floods = gpd.read_file(FLOOD_FILE)

print("\nCritical places:", len(places))
print("Flood zones:", len(floods))


# Project data to a metric CRS for accurate distance calculations
METRIC_CRS = "EPSG:32644"

places_metric = places.to_crs(METRIC_CRS)
floods_metric = floods.to_crs(METRIC_CRS)


# -----------------------------
# Validate CRS
# -----------------------------

if places.crs is None:
    raise ValueError("Critical places have no CRS.")

if floods.crs is None:
    raise ValueError("Flood zones have no CRS.")

print("\nCRS:")
print("Places:", places.crs)
print("Floods:", floods.crs)


# -----------------------------
# Calculate flood exposure
# -----------------------------

print("\nCalculating flood exposure...")


def calculate_flood_exposure(place, flood_zones):

    # First check whether the place is inside a flood zone
    containing_zones = flood_zones[flood_zones.geometry.contains(place.geometry)]

    if not containing_zones.empty:

        # If multiple zones contain the place,
        # use the highest risk score.
        return containing_zones["risk_score"].max()

    # If not inside a flood zone,
    # use distance to nearest flood zone.

    distances = flood_zones.geometry.distance(place.geometry)

    nearest_index = distances.idxmin()

    nearest_zone = flood_zones.loc[nearest_index]

    nearest_distance = distances.loc[nearest_index]

    # Convert distance into a simple exposure value.
    # Within approximately 0.01 degrees -> significant exposure.
    #
    # This is a prototype geographic exposure measure.
    exposure = nearest_zone["risk_score"] * max(
        0,
        1 - (nearest_distance / 0.01)
    )

    return exposure


places["flood_exposure"] = places_metric.apply(
    lambda row: calculate_flood_exposure(row, floods_metric),
    axis=1
)


# -----------------------------
# Calculate place criticality
# -----------------------------

print("Calculating place criticality...")

places["criticality"] = places["type"].map(
    PLACE_CRITICALITY
)

# Unknown types receive a moderate default
places["criticality"] = places["criticality"].fillna(0.50)


# -----------------------------
# Normalize factors
# -----------------------------

print("Normalizing factors...")


def min_max_normalize(series):

    min_value = series.min()
    max_value = series.max()

    if max_value == min_value:
        return np.ones(len(series))

    return (series - min_value) / (max_value - min_value)


places["flood_norm"] = min_max_normalize(
    places["flood_exposure"]
)

places["criticality_norm"] = min_max_normalize(
    places["criticality"]
)


# -----------------------------
# Entropy weighting
# -----------------------------

print("Calculating entropy weights...")


def entropy_weights(data):

    data = np.asarray(data, dtype=float)

    # Avoid division by zero
    column_sums = data.sum(axis=0)

    # If a column contains only zero values
    # assign equal probability.
    probabilities = np.zeros_like(data)

    for j in range(data.shape[1]):

        if column_sums[j] == 0:
            probabilities[:, j] = 1 / data.shape[0]

        else:
            probabilities[:, j] = (
                data[:, j] / column_sums[j]
            )

    n = data.shape[0]

    if n <= 1:
        return np.ones(data.shape[1]) / data.shape[1]

    k = 1 / np.log(n)

    entropy = []

    for j in range(data.shape[1]):

        p = probabilities[:, j]

        entropy_value = -k * np.sum(
            p[p > 0] * np.log(p[p > 0])
        )

        entropy.append(entropy_value)

    entropy = np.array(entropy)

    divergence = 1 - entropy

    if divergence.sum() == 0:
        return np.ones(len(divergence)) / len(divergence)

    weights = divergence / divergence.sum()

    return weights


factor_columns = [
    "flood_norm",
    "criticality_norm"
]

factor_data = places[factor_columns].values

weights = entropy_weights(factor_data)


print("\nEntropy weights:")

for column, weight in zip(factor_columns, weights):
    print(f"{column}: {weight:.4f}")


# -----------------------------
# Calculate priority score
# -----------------------------

places["priority_score"] = (
    places["flood_norm"] * weights[0]
    +
    places["criticality_norm"] * weights[1]
)

# Convert to 0-100
places["priority_score"] = (
    places["priority_score"] * 100
)


# -----------------------------
# Priority classification
# -----------------------------

def classify_priority(score):

    if score >= 75:
        return "CRITICAL"

    elif score >= 50:
        return "HIGH"

    elif score >= 25:
        return "MEDIUM"

    else:
        return "LOW"


places["priority_class"] = places["priority_score"].apply(
    classify_priority
)


# -----------------------------
# Select final columns
# -----------------------------

result = places[
    [
        "place_id",
        "name",
        "type",
        "flood_exposure",
        "criticality",
        "priority_score",
        "priority_class",
        "geometry"
    ]
].copy()


# -----------------------------
# Save output
# -----------------------------

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

result.to_file(
    OUTPUT_FILE,
    driver="GeoJSON"
)


# -----------------------------
# Display result
# -----------------------------

print("\n" + "=" * 50)
print("PRIORITY CALCULATION COMPLETE")
print("=" * 50)

print("\nResults:")

print(
    result[
        [
            "place_id",
            "name",
            "type",
            "flood_exposure",
            "criticality",
            "priority_score",
            "priority_class"
        ]
    ].to_string(index=False)
)

print("\nOutput:")
print(OUTPUT_FILE)

print("\nLevel 5 completed successfully.")