    #          Your dataset
    #               ↓
    #       ┌───────────────┐
    #       │ validate_ids  │
    #       └───────┬───────┘
    #               ↓
    #    ┌─────────────────────┐
    #    │ Missing IDs?         │
    #    │ Duplicate IDs?       │
    #    │ Wrong format?        │
    #    │ Empty IDs?           │
    #    └──────────┬──────────┘
    #               ↓
    #          PASS / FAIL



import geopandas as gpd
import re


# ==========================================
# File paths
# ==========================================

ROADS_FILE = "C:\\DisasterSaver\\data\\raw\\roads.geojson"
CRITICAL_PLACES_FILE = "C:\\DisasterSaver\\data\\raw\\critical_places.geojson"
EMERGENCY_BASES_FILE = "C:\\DisasterSaver\\data\\raw\\emergency_bases.geojson"
PREDICTED_FLOOD_FILE = "C:\\DisasterSaver\\data\\raw\\predicted_flood.geojson"


# ==========================================
# Load data
# ==========================================

roads = gpd.read_file(ROADS_FILE)
critical_places = gpd.read_file(CRITICAL_PLACES_FILE)
emergency_bases = gpd.read_file(EMERGENCY_BASES_FILE)
predicted_flood = gpd.read_file(PREDICTED_FLOOD_FILE)


# ==========================================
# ID Validation Function
# ==========================================

def validate_ids(data, dataset_name, id_column, expected_prefix=None):

    print("\n========================================")
    print(dataset_name)
    print("========================================")

    # Check whether ID column exists
    if id_column not in data.columns:
        print(f"ID column '{id_column}' NOT FOUND")
        print("Available columns:", data.columns.tolist())
        return

    print(f"ID column: {id_column}")

    # Total records
    total_records = len(data)

    # Missing IDs
    missing_ids = data[id_column].isna().sum()

    # Duplicate IDs
    duplicate_ids = data[id_column].duplicated().sum()

    # Unique IDs
    unique_ids = data[id_column].nunique()

    # ID format validation
    invalid_format = 0

    if expected_prefix is not None:

        pattern = rf"^{expected_prefix}\d+$"

        for value in data[id_column].dropna():

            if not re.match(pattern, str(value)):
                invalid_format += 1

    # Print results
    print(f"Total records       : {total_records}")
    print(f"Unique IDs          : {unique_ids}")
    print(f"Missing IDs         : {missing_ids}")
    print(f"Duplicate IDs       : {duplicate_ids}")
    print(f"Invalid ID format   : {invalid_format}")

    # Final result
    if (
        missing_ids == 0
        and duplicate_ids == 0
        and invalid_format == 0
    ):
        print("\nID VALIDATION: PASSED")
    else:
        print("\nID VALIDATION: FAILED")


# ==========================================
# Validate datasets
# ==========================================

validate_ids(
    roads,
    "ROADS",
    "road_id",
    "R"
)

validate_ids(
    critical_places,
    "CRITICAL PLACES",
    "place_id",
    "P"
)

validate_ids(
    emergency_bases,
    "EMERGENCY BASES",
    "base_id",
    "B"
)

validate_ids(
    predicted_flood,
    "PREDICTED FLOOD",
    "zone_id",
    "F"
)


# ==========================================
# Completed
# ==========================================

print("\n========================================")
print("ID validation completed!")
print("========================================")