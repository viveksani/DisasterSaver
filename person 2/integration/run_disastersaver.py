import subprocess
import sys
from pathlib import Path
from datetime import datetime


# ============================================================
# DisasterSaver - Master Integration Pipeline
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
SCRIPT_DIR = BASE_DIR / "priority_routing"
OUTPUT_DIR = BASE_DIR / "outputs"
PROCESSED_DIR = BASE_DIR / "data" / "processed"


# ============================================================
# PIPELINE STAGES
# ============================================================

STAGES = [
    {
        "name": "Level 3 - ID Validation",
        "script": "validate_ids.py",
        "outputs": [],
    },

    {
        "name": "Level 4.4 - Affected Road Detection",
        "script": "detect_affected_roads.py",
        "outputs": [
            PROCESSED_DIR / "affected_roads.geojson"
        ],
    },

    {
        "name": "Level 5 - Priority Place Calculation",
        "script": "calculate_priority.py",
        "outputs": [
            OUTPUT_DIR / "priority_places.geojson"
        ],
    },

    {
        "name": "Level 6.1 - Hazard-Aware Routing",
        "script": "hazard_aware_routing.py",
        "outputs": [
            OUTPUT_DIR / "hazard_aware_roads.geojson"
        ],
    },

    {
        "name": "Level 6.2 - Routing Cost Calculation",
        "script": "calculate_route_cost.py",
        "outputs": [
            OUTPUT_DIR / "routing_roads.geojson"
        ],
    },

    {
        "name": "Level 6.3 - Road Network Construction",
        "script": "build_network.py",
        "outputs": [
            OUTPUT_DIR / "road_network.graphml"
        ],
    },

    {
        "name": "Level 6.4 - Hazard-Aware Route Calculation",
        "script": "calculate_routes.py",
        "outputs": [
            OUTPUT_DIR / "calculated_routes.json"
        ],
    },

    {
        "name": "Level 6.5 - Final Route Export",
        "script": "export_routes.py",
        "outputs": [
            OUTPUT_DIR / "routes.geojson"
        ],
    },

    {
        "name": "Level 7 - Response Zone Creation",
        "script": "create_response_zones.py",
        "outputs": [
            OUTPUT_DIR / "response_zones.geojson"
        ],
    },
]


# ============================================================
# HEADER
# ============================================================

print("=" * 75)
print("DISASTERSAVER - MASTER INTEGRATION PIPELINE")
print("=" * 75)

start_time = datetime.now()

print(f"\nProject directory:")
print(BASE_DIR)

print(f"\nPipeline started:")
print(start_time.strftime("%Y-%m-%d %H:%M:%S"))


# ============================================================
# ENVIRONMENT
# ============================================================

print("\n" + "=" * 75)
print("ENVIRONMENT CHECK")
print("=" * 75)

print(f"Python executable:")
print(sys.executable)

print(f"\nPython version:")
print(sys.version.split()[0])


# ============================================================
# DIRECTORIES
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# RUN STAGES
# ============================================================

completed_stages = []
failed_stage = None


for number, stage in enumerate(STAGES, start=1):

    stage_name = stage["name"]
    script_name = stage["script"]

    script_path = SCRIPT_DIR / script_name

    print("\n")
    print("#" * 75)
    print(f"STAGE {number}/{len(STAGES)}")
    print(stage_name)
    print("#" * 75)

    # --------------------------------------------------------
    # Check script
    # --------------------------------------------------------

    if not script_path.exists():

        print("\nERROR: Script not found:")
        print(script_path)

        failed_stage = stage_name
        break

    print(f"\nRunning:")
    print(script_path)

    # --------------------------------------------------------
    # Execute script
    # --------------------------------------------------------

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(BASE_DIR),
    )

    # --------------------------------------------------------
    # Check return code
    # --------------------------------------------------------

    if result.returncode != 0:

        print("\n" + "!" * 75)
        print("STAGE FAILED")
        print("!" * 75)

        print(f"\nStage:")
        print(stage_name)

        print(f"\nScript:")
        print(script_name)

        print(f"\nReturn code:")
        print(result.returncode)

        failed_stage = stage_name
        break

    # --------------------------------------------------------
    # Check expected outputs
    # --------------------------------------------------------

    missing_outputs = []

    for output_file in stage["outputs"]:

        if not output_file.exists():

            missing_outputs.append(
                output_file
            )

    if missing_outputs:

        print("\n" + "!" * 75)
        print("STAGE FAILED - OUTPUT MISSING")
        print("!" * 75)

        for output_file in missing_outputs:

            print(
                f"\nMissing output:\n{output_file}"
            )

        failed_stage = stage_name
        break

    # --------------------------------------------------------
    # Success
    # --------------------------------------------------------

    print("\n" + "-" * 75)
    print("STAGE COMPLETED SUCCESSFULLY")
    print("-" * 75)

    if stage["outputs"]:

        print("\nGenerated outputs:")

        for output_file in stage["outputs"]:

            print(
                f"  [OK] {output_file}"
            )

    completed_stages.append(stage_name)


# ============================================================
# FINAL SUMMARY
# ============================================================

end_time = datetime.now()

duration = end_time - start_time


print("\n")
print("=" * 75)
print("DISASTERSAVER PIPELINE SUMMARY")
print("=" * 75)

print(
    f"\nCompleted stages: "
    f"{len(completed_stages)}/{len(STAGES)}"
)

print(
    f"Execution time: "
    f"{duration}"
)


# ============================================================
# SUCCESS
# ============================================================

if failed_stage is None:

    print("\n" + "=" * 75)
    print("ALL PIPELINE STAGES COMPLETED SUCCESSFULLY")
    print("=" * 75)

    print("\nFinal outputs:")
    print()

    final_outputs = [
        OUTPUT_DIR / "priority_places.geojson",
        OUTPUT_DIR / "hazard_aware_roads.geojson",
        OUTPUT_DIR / "routing_roads.geojson",
        OUTPUT_DIR / "road_network.graphml",
        OUTPUT_DIR / "calculated_routes.json",
        OUTPUT_DIR / "routes.geojson",
        OUTPUT_DIR / "response_zones.geojson",
        PROCESSED_DIR / "affected_roads.geojson",
    ]

    for output_file in final_outputs:

        if output_file.exists():

            size_kb = (
                output_file.stat().st_size / 1024
            )

            print(
                f"[OK] {output_file.name:<30} "
                f"{size_kb:.2f} KB"
            )

    print("\nDisasterSaver integration complete.")

    sys.exit(0)


# ============================================================
# FAILURE
# ============================================================

print("\n" + "=" * 75)
print("DISASTERSAVER PIPELINE FAILED")
print("=" * 75)

print(
    f"\nFailed stage:"
)

print(
    failed_stage
)

print(
    "\nFix the failed stage and run the master pipeline again."
)

sys.exit(1)