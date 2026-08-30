import geopandas as gpd
import pandas as pd
import os


# ==========================================
# File paths
# ==========================================

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

ROADS_FILE = BASE_DIR / "data" / "raw" / "roads.geojson"
CRITICAL_PLACES_FILE = BASE_DIR / "data" / "raw" / "critical_places.geojson"
EMERGENCY_BASES_FILE = BASE_DIR / "data" / "raw" / "emergency_bases.geojson"
PREDICTED_FLOOD_FILE = BASE_DIR / "data" / "raw" / "predicted_flood.geojson"

REGISTRY_FILE = BASE_DIR / "data" / "processed" / "id_registry.csv"


# ==========================================
# Load datasets
# ==========================================

roads = gpd.read_file(ROADS_FILE)
critical_places = gpd.read_file(CRITICAL_PLACES_FILE)
emergency_bases = gpd.read_file(EMERGENCY_BASES_FILE)
predicted_flood = gpd.read_file(PREDICTED_FLOOD_FILE)


# ==========================================
# Create ID registry
# ==========================================

registry = []


# Roads
for entity_id in roads["road_id"]:
    registry.append({
        "entity_type": "road",
        "id_column": "road_id",
        "entity_id": entity_id,
        "status": "LOCKED"
    })


# Critical places
for entity_id in critical_places["place_id"]:
    registry.append({
        "entity_type": "place",
        "id_column": "place_id",
        "entity_id": entity_id,
        "status": "LOCKED"
    })


# Emergency bases
for entity_id in emergency_bases["base_id"]:
    registry.append({
        "entity_type": "base",
        "id_column": "base_id",
        "entity_id": entity_id,
        "status": "LOCKED"
    })


# Predicted flood zones
for entity_id in predicted_flood["zone_id"]:
    registry.append({
        "entity_type": "flood_zone",
        "id_column": "zone_id",
        "entity_id": entity_id,
        "status": "LOCKED"
    })


# ==========================================
# Convert to DataFrame
# ==========================================

registry_df = pd.DataFrame(registry)


# ==========================================
# Create processed folder if needed
# ==========================================

os.makedirs(
    "C:\\DisasterSaver\\data\\processed",
    exist_ok=True
)


# ==========================================
# Save registry
# ==========================================

registry_df.to_csv(
    REGISTRY_FILE,
    index=False
)


# ==========================================
# Display result
# ==========================================

print("========================================")
print("DisasterSaver - ID Registry")
print("========================================")

print("\nLocked IDs:")
print(registry_df.to_string(index=False))

print("\n========================================")
print(f"Registry saved to:")
print(REGISTRY_FILE)
print("========================================")

print("\nID LOCKING COMPLETED")