import geopandas as gpd
import os
import pandas as pd


# ==========================================
# DisasterSaver - Affected Road Detection
# Level 4.4: Calculate Flood Overlap
# ==========================================

ROADS_FILE = r"C:\\DisasterSaver\\data\\raw\\roads.geojson"
FLOOD_FILE = r"C:\\DisasterSaver\\data\\raw\\predicted_flood.geojson"

OUTPUT_DIR = r"C:\DisasterSaver\data\processed"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "affected_roads.geojson")

print("========================================")
print("DisasterSaver - Affected Road Detection")
print("========================================")

# ==========================================
# 1. LOAD DATA
# ==========================================

roads = gpd.read_file(ROADS_FILE)
flood_zones = gpd.read_file(FLOOD_FILE)

print("\nData loaded successfully!")
print("Number of roads:", len(roads))
print("Number of flood zones:", len(flood_zones))

# ==========================================
# 2. CRS
# ==========================================

print("\n========== CRS ==========")

print("Road CRS:", roads.crs)
print("Flood CRS:", flood_zones.crs)

if roads.crs != flood_zones.crs:
    flood_zones = flood_zones.to_crs(roads.crs)

print("CRS STATUS: READY")

# ==========================================
# 3. VALIDATE GEOMETRIES
# ==========================================

print("\n========== GEOMETRY VALIDATION ==========")

if roads.geometry.isna().any() or roads.geometry.is_empty.any():
    print("ERROR: Roads contain missing/empty geometry.")
    exit()

if flood_zones.geometry.isna().any() or flood_zones.geometry.is_empty.any():
    print("ERROR: Flood zones contain missing/empty geometry.")
    exit()

if (~roads.geometry.is_valid).any():
    print("ERROR: Roads contain invalid geometry.")
    exit()

if (~flood_zones.geometry.is_valid).any():
    print("ERROR: Flood zones contain invalid geometry.")
    exit()

print("Geometry validation: PASSED")

# ==========================================
# 4. PROJECT TO METRIC CRS
# ==========================================

print("\n========== PROJECTED CRS ==========")

# EPSG:32644 = UTM Zone 44N
# Suitable for the Chennai-area sample coordinates.

roads_metric = roads.to_crs(epsg=32644)
flood_metric = flood_zones.to_crs(epsg=32644)

print("Metric CRS:", roads_metric.crs)

# ==========================================
# 5. CALCULATE ORIGINAL ROAD LENGTH
# ==========================================

roads_metric["original_length_m"] = roads_metric.geometry.length

# ==========================================
# 6. FIND FLOOD INTERSECTIONS
# ==========================================

print("\n========== FLOOD INTERSECTION ==========")

affected_records = []

for road_index, road in roads_metric.iterrows():

    road_geometry = road.geometry

    total_road_length = road_geometry.length

    best_overlap_length = 0
    best_flood_zone = None
    best_flood_risk = None

    # Check this road against every flood zone
    for flood_index, flood in flood_metric.iterrows():

        flood_geometry = flood.geometry

        if road_geometry.intersects(flood_geometry):

            intersection = road_geometry.intersection(flood_geometry)

            overlap_length = intersection.length

            # Keep the flood zone producing the largest overlap
            if overlap_length > best_overlap_length:

                best_overlap_length = overlap_length
                best_flood_zone = flood["zone_id"]
                best_flood_risk = flood["risk_score"]

    # ======================================
    # 7. KEEP ONLY AFFECTED ROADS
    # ======================================

    if best_overlap_length > 0:

        overlap_percentage = (
            best_overlap_length / total_road_length
        ) * 100

        affected_records.append({
            "road_index": road_index,
            "road_id": road["road_id"],
            "zone_id": best_flood_zone,
            "flood_overlap_pct": round(overlap_percentage, 2),
            "flood_risk_score": best_flood_risk
        })

# ==========================================
# 8. CREATE RESULT TABLE
# ==========================================

affected_info = pd.DataFrame(affected_records)

# Get original road attributes
road_attributes = roads_metric[
    [
        "road_id",
        "highway_type",
        "road_length_m",
        "geometry"
    ]
].copy()

# Merge calculated flood information
affected_roads = road_attributes.merge(
    affected_info[
        [
            "road_id",
            "flood_overlap_pct",
            "flood_risk_score"
        ]
    ],
    on="road_id",
    how="inner"
)

# ==========================================
# 9. FINAL COLUMNS
# ==========================================

affected_roads = affected_roads[
    [
        "road_id",
        "flood_overlap_pct",
        "flood_risk_score",
        "highway_type",
        "road_length_m",
        "geometry"
    ]
]

# ==========================================
# 10. RETURN TO EPSG:4326
# ==========================================

affected_roads = affected_roads.to_crs(epsg=4326)

# ==========================================
# 11. SAVE
# ==========================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

affected_roads.to_file(
    OUTPUT_FILE,
    driver="GeoJSON"
)

# ==========================================
# 12. DISPLAY RESULTS
# ==========================================

print("\n========== AFFECTED ROADS ==========")

print(
    affected_roads[
        [
            "road_id",
            "flood_overlap_pct",
            "flood_risk_score",
            "highway_type",
            "road_length_m"
        ]
    ].to_string(index=False)
)

print("\n========== OUTPUT ==========")

print("Affected roads:", len(affected_roads))
print("Saved to:")
print(OUTPUT_FILE)

print("\n========================================")
print("LEVEL 4.4 COMPLETE")
print("========================================")