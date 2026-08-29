import geopandas as gpd


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
# Basic information
# ==========================================

print("================================")
print("DisasterSaver - Data Inspection")
print("================================")


# ==========================================
# Roads
# ==========================================

print("\n\n========== ROADS ==========")

print("\nColumns:")
print(roads.columns.tolist())

print("\nData:")
print(roads)

print("\nGeometry types:")
print(roads.geometry.geom_type.tolist())

print("\nCRS:")
print(roads.crs)


# ==========================================
# Critical Places
# ==========================================

print("\n\n========== CRITICAL PLACES ==========")

print("\nColumns:")
print(critical_places.columns.tolist())

print("\nData:")
print(critical_places)

print("\nGeometry types:")
print(critical_places.geometry.geom_type.tolist())

print("\nCRS:")
print(critical_places.crs)


# ==========================================
# Emergency Bases
# ==========================================

print("\n\n========== EMERGENCY BASES ==========")

print("\nColumns:")
print(emergency_bases.columns.tolist())

print("\nData:")
print(emergency_bases)

print("\nGeometry types:")
print(emergency_bases.geometry.geom_type.tolist())

print("\nCRS:")
print(emergency_bases.crs)


# ==========================================
# Predicted Flood
# ==========================================

print("\n\n========== PREDICTED FLOOD ==========")

print("\nColumns:")
print(predicted_flood.columns.tolist())

print("\nData:")
print(predicted_flood)

print("\nGeometry types:")
print(predicted_flood.geometry.geom_type.tolist())

print("\nCRS:")
print(predicted_flood.crs)


print("\n\n================================")
print("Inspection completed!")
print("================================")