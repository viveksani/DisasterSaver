import geopandas as gpd
import os


print("=" * 70)
print("DisasterSaver - Level 7")
print("Creating Response Zones")
print("=" * 70)


# ============================================================
# FILE PATHS
# ============================================================

PRIORITY_PLACES_FILE = "outputs/priority_places.geojson"
ROUTES_FILE = "outputs/routes.geojson"
BASES_FILE = "data/raw/emergency_bases.geojson"

OUTPUT_FILE = "outputs/response_zones.geojson"


# ============================================================
# SETTINGS
# ============================================================

# Maximum distance between threatened places for them
# to be considered part of the same response zone.
GROUPING_DISTANCE_METERS = 2500

# A place must have some flood exposure to be considered
# for the current flood-response zone.
MIN_FLOOD_EXPOSURE = 0.01


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading priority places...")
places = gpd.read_file(PRIORITY_PLACES_FILE)
print(f"Priority places loaded: {len(places)}")


print("\nLoading routes...")
routes = gpd.read_file(ROUTES_FILE)
print(f"Routes loaded: {len(routes)}")


print("\nLoading emergency bases...")
bases = gpd.read_file(BASES_FILE)
print(f"Emergency bases loaded: {len(bases)}")


# ============================================================
# VALIDATE COLUMNS
# ============================================================

required_place_columns = [
    "place_id",
    "priority_score",
    "priority_class",
    "flood_exposure",
    "criticality",
    "geometry"
]

required_route_columns = [
    "route_id",
    "from_id",
    "to_id",
    "route_risk_score",
    "geometry"
]

required_base_columns = [
    "base_id",
    "team_count",
    "geometry"
]


missing_places = [
    c for c in required_place_columns
    if c not in places.columns
]

missing_routes = [
    c for c in required_route_columns
    if c not in routes.columns
]

missing_bases = [
    c for c in required_base_columns
    if c not in bases.columns
]


if missing_places:
    print("\nERROR: Missing columns in priority_places.geojson:")
    print(missing_places)
    raise SystemExit(1)


if missing_routes:
    print("\nERROR: Missing columns in routes.geojson:")
    print(missing_routes)
    raise SystemExit(1)


if missing_bases:
    print("\nERROR: Missing columns in emergency_bases.geojson:")
    print(missing_bases)
    raise SystemExit(1)


# ============================================================
# CHECK CRS
# ============================================================

print("\nInput CRS:")
print(f"Priority places : {places.crs}")
print(f"Routes          : {routes.crs}")
print(f"Emergency bases : {bases.crs}")


if places.crs is None:
    raise SystemExit("ERROR: Priority places have no CRS.")

if routes.crs is None:
    raise SystemExit("ERROR: Routes have no CRS.")

if bases.crs is None:
    raise SystemExit("ERROR: Emergency bases have no CRS.")


# ============================================================
# PROJECT TO EPSG:32644
# ============================================================

print("\nProjecting data to EPSG:32644...")

places_m = places.to_crs("EPSG:32644")
routes_m = routes.to_crs("EPSG:32644")
bases_m = bases.to_crs("EPSG:32644")


# ============================================================
# IDENTIFY RESPONSE PLACES
# ============================================================

print("\nIdentifying flood-exposed priority places...")


response_places = places_m[
    places_m["flood_exposure"].astype(float)
    >= MIN_FLOOD_EXPOSURE
].copy()


print(
    f"Flood-exposed priority places: "
    f"{len(response_places)}"
)


if response_places.empty:

    print("\nNo flood-exposed priority places found.")

    empty = gpd.GeoDataFrame(
        columns=[
            "response_zone_id",
            "priority_level",
            "critical_places_count",
            "recommended_team_count",
            "nearest_base_id",
            "zone_risk_score",
            "geometry"
        ],
        geometry="geometry",
        crs="EPSG:4326"
    )

    os.makedirs("outputs", exist_ok=True)

    empty.to_file(
        OUTPUT_FILE,
        driver="GeoJSON"
    )

    print(f"\nEmpty output created: {OUTPUT_FILE}")

    raise SystemExit(0)


# ============================================================
# GROUP RESPONSE PLACES
# ============================================================

print("\nGrouping nearby response places...")


response_places["group_id"] = -1

next_group = 0


for index in response_places.index:

    if response_places.loc[index, "group_id"] != -1:
        continue

    response_places.loc[index, "group_id"] = next_group

    changed = True

    while changed:

        changed = False

        current_members = response_places[
            response_places["group_id"] == next_group
        ]

        for candidate_index in response_places.index:

            if response_places.loc[
                candidate_index,
                "group_id"
            ] != -1:
                continue

            candidate_geometry = response_places.loc[
                candidate_index,
                "geometry"
            ]

            for member_index in current_members.index:

                member_geometry = response_places.loc[
                    member_index,
                    "geometry"
                ]

                distance = candidate_geometry.distance(
                    member_geometry
                )

                if distance <= GROUPING_DISTANCE_METERS:

                    response_places.loc[
                        candidate_index,
                        "group_id"
                    ] = next_group

                    changed = True
                    break

        if changed:
            current_members = response_places[
                response_places["group_id"] == next_group
            ]

    next_group += 1


number_of_zones = response_places[
    "group_id"
].nunique()


print(
    f"Response zones identified: "
    f"{number_of_zones}"
)


# ============================================================
# PRIORITY RANK
# ============================================================

priority_rank = {
    "LOW": 1,
    "MEDIUM": 2,
    "HIGH": 3,
    "CRITICAL": 4
}


def calculate_priority_level(group):

    ranks = []

    for value in group["priority_class"]:

        ranks.append(
            priority_rank.get(
                str(value).upper(),
                1
            )
        )

    highest_rank = max(ranks)

    if highest_rank == 4:
        return "CRITICAL"

    if highest_rank == 3:
        return "HIGH"

    if highest_rank == 2:
        return "MEDIUM"

    return "LOW"


# ============================================================
# TEAM COUNT CALCULATION
# ============================================================

def calculate_recommended_team_count(group):

    total = 0

    for _, place in group.iterrows():

        priority = str(
            place["priority_class"]
        ).upper()

        if priority == "CRITICAL":
            total += 2

        elif priority == "HIGH":
            total += 2

        elif priority == "MEDIUM":
            total += 1

        else:
            total += 1

    return max(1, total)


# ============================================================
# CREATE RESPONSE ZONES
# ============================================================

print("\nCalculating response-zone attributes...")


zone_records = []


for group_number in sorted(
    response_places["group_id"].unique()
):

    group = response_places[
        response_places["group_id"] == group_number
    ].copy()


    # --------------------------------------------------------
    # RESPONSE ZONE ID
    # --------------------------------------------------------

    response_zone_id = (
        f"Z{group_number + 1:06d}"
    )


    # --------------------------------------------------------
    # COUNT RESPONSE PLACES
    # --------------------------------------------------------

    critical_places_count = len(group)


    # --------------------------------------------------------
    # PRIORITY LEVEL
    # --------------------------------------------------------

    priority_level = calculate_priority_level(
        group
    )


    # --------------------------------------------------------
    # RECOMMENDED TEAM COUNT
    # --------------------------------------------------------

    recommended_team_count = (
        calculate_recommended_team_count(
            group
        )
    )


    # --------------------------------------------------------
    # PLACE IDS
    # --------------------------------------------------------

    place_ids = set(
        group["place_id"].astype(str)
    )


    # --------------------------------------------------------
    # FIND ROUTES SERVING THESE PLACES
    # --------------------------------------------------------

    serving_routes = routes_m[
        routes_m["to_id"]
        .astype(str)
        .isin(place_ids)
    ].copy()


    # --------------------------------------------------------
    # DETERMINE BASE
    # --------------------------------------------------------

    if not serving_routes.empty:

        route_base_ids = (
            serving_routes["from_id"]
            .astype(str)
            .dropna()
            .unique()
            .tolist()
        )

        candidate_bases = bases_m[
            bases_m["base_id"]
            .astype(str)
            .isin(route_base_ids)
        ].copy()

    else:

        candidate_bases = bases_m.copy()


    # --------------------------------------------------------
    # FALLBACK
    # --------------------------------------------------------

    if candidate_bases.empty:

        candidate_bases = bases_m.copy()


    # --------------------------------------------------------
    # ZONE CENTROID
    # --------------------------------------------------------

    place_union = group.geometry.union_all()

    zone_center = place_union.centroid


    # --------------------------------------------------------
    # BASE DISTANCE
    # --------------------------------------------------------

    candidate_bases["zone_distance"] = (
        candidate_bases.geometry.distance(
            zone_center
        )
    )


    nearest_base = candidate_bases.loc[
        candidate_bases["zone_distance"].idxmin()
    ]


    nearest_base_id = nearest_base["base_id"]


    # --------------------------------------------------------
    # FLOOD RISK
    # --------------------------------------------------------

    flood_exposure = group[
        "flood_exposure"
    ].astype(float)


    priority_score = group[
        "priority_score"
    ].astype(float)


    if priority_score.sum() > 0:

        weighted_flood_risk = (
            (
                flood_exposure *
                priority_score
            ).sum()
            /
            priority_score.sum()
        )

    else:

        weighted_flood_risk = (
            flood_exposure.mean()
        )


    # --------------------------------------------------------
    # ROUTE RISK
    # --------------------------------------------------------

    if not serving_routes.empty:

        route_risk = (
            serving_routes[
                "route_risk_score"
            ]
            .astype(float)
            .mean()
        )

    else:

        route_risk = 0.0


    # --------------------------------------------------------
    # ZONE RISK
    # --------------------------------------------------------

    zone_risk_score = (
        0.60 * weighted_flood_risk
        +
        0.40 * route_risk
    )


    zone_risk_score = round(
        float(zone_risk_score),
        2
    )


    # --------------------------------------------------------
    # CREATE OPERATIONAL POLYGON
    # --------------------------------------------------------

    zone_geometry = place_union.buffer(
        GROUPING_DISTANCE_METERS
    )


    # --------------------------------------------------------
    # STORE ZONE
    # --------------------------------------------------------

    zone_records.append(
        {
            "response_zone_id":
                response_zone_id,

            "priority_level":
                priority_level,

            "critical_places_count":
                critical_places_count,

            "recommended_team_count":
                recommended_team_count,

            "nearest_base_id":
                nearest_base_id,

            "zone_risk_score":
                zone_risk_score,

            "geometry":
                zone_geometry
        }
    )


# ============================================================
# CREATE GEODATAFRAME
# ============================================================

response_zones = gpd.GeoDataFrame(
    zone_records,
    geometry="geometry",
    crs="EPSG:32644"
)


# ============================================================
# CONVERT TO EPSG:4326
# ============================================================

response_zones = response_zones.to_crs(
    "EPSG:4326"
)


# ============================================================
# EXACT COLUMN ORDER
# ============================================================

response_zones = response_zones[
    [
        "response_zone_id",
        "priority_level",
        "critical_places_count",
        "recommended_team_count",
        "nearest_base_id",
        "zone_risk_score",
        "geometry"
    ]
]


# ============================================================
# SAVE
# ============================================================

os.makedirs(
    "outputs",
    exist_ok=True
)


response_zones.to_file(
    OUTPUT_FILE,
    driver="GeoJSON"
)


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 70)
print("RESPONSE ZONE CREATION COMPLETE")
print("=" * 70)


print(
    f"\nResponse zones created: "
    f"{len(response_zones)}"
)


for _, zone in response_zones.iterrows():

    print("\n----------------------------------------")

    print(
        f"Response Zone ID:       "
        f"{zone['response_zone_id']}"
    )

    print(
        f"Priority Level:         "
        f"{zone['priority_level']}"
    )

    print(
        f"Critical Places Count:  "
        f"{zone['critical_places_count']}"
    )

    print(
        f"Recommended Teams:      "
        f"{zone['recommended_team_count']}"
    )

    print(
        f"Nearest Base:           "
        f"{zone['nearest_base_id']}"
    )

    print(
        f"Zone Risk Score:        "
        f"{zone['zone_risk_score']}"
    )


print("\nOutput file:")
print(f"  {OUTPUT_FILE}")

print("\nOutput CRS:")
print(f"  {response_zones.crs}")

print("\nLevel 7 completed successfully!")

print("=" * 70)