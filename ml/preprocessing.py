"""
DisasterSaver — Flood Prediction Pipeline (Person 1)
Module: ml/preprocessing.py
Description: Feature Matrix Assembly & Spatial Preprocessing

Functions & Methods:
- compute_flow_accumulation: D8 topographic flow accumulation down steepest slopes.
  Note: This implementation routes flow down steepest descent gradients.
  Unresolved DEM depressions/flats act as sinks (flow terminates). It does NOT
  perform full hydrological sink filling or pit breaching.
- densify_linestring_segment_points: Interpolates points along LineStrings at step_m intervals.
- map_drains_to_grid: Calculates minimum distance from cell centers to drain geometries
  using a 10m LineString vertex densification approximation (nearest-point lookup via cKDTree).
  The 10m sampling step bounds spatial distance approximation error to <= 5.0m (half step).
  Coordinate conversion uses a local equirectangular (geographic-to-metric) approximation,
  NOT true projected UTM coordinates.
- assemble_feature_matrix: Assembles 2D feature matrix X (total_cells x 7) in strict
  FEATURE_NAMES column order.
- save_feature_artifacts: Exports X, y, and feature_names to compressed .npz archive.

Target Environment: Google Colab (/content/ml/preprocessing.py) / Python 3.8+
"""

import os
import json
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from typing import Tuple, Dict, Any, List
from scipy.spatial import cKDTree

from ml.rainfall_runoff import rational_runoff, overload_ratio, DEFAULT_OVERLAND_FLOW_CAPACITY_M3_S

FEATURE_NAMES = [
    "elevation",        # Elevation Z (meters)
    "slope",            # Slope S (degrees)
    "tpi",              # Topographic Position Index (meters)
    "dist_to_drain",    # Distance to nearest drain geometry (meters, 10m densification approximation)
    "q_cap",            # Local drainage conveyance capacity Q_cap (m³/s)
    "q_demand",         # Surface runoff demand Q_demand (m³/s)
    "r_overload"        # Drainage overload ratio R_overload = Q_demand / Q_cap
]


def compute_flow_accumulation(
    elevation: np.ndarray,
    pixel_size_m: float = 30.83
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes D8 topographic flow direction and catchment accumulation area down steepest
    elevation descent gradients.

    Algorithm Note:
    ---------------
    Flow routes from each cell to its single steepest downhill neighbor among the 8 surrounding cells.
    Cells with no downhill neighbor (flats, local depressions, or boundary pits) receive flow from
    upstream but do not route flow further (acting as local sinks). This is a standard D8 flow routing
    model without depression-filling or pit-breaching post-processing.

    Parameters:
        elevation: 2D float32 array of elevation values (height x width)
        pixel_size_m: Grid cell spatial resolution in meters (~30.83m)

    Returns:
        accum_count: 2D float32 array of upstream contributing cell counts (min value 1.0)
        catchment_area_m2: 2D float32 array of contributing catchment areas in m²
    """
    height, width = elevation.shape
    cell_area_m2 = pixel_size_m * pixel_size_m

    # 8-neighbor directional offsets (dy, dx, distance_multiplier)
    neighbors = [
        (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
        (-1, -1, 1.41421356), (-1, 1, 1.41421356), (1, -1, 1.41421356), (1, 1, 1.41421356)
    ]

    pad_z = np.pad(elevation, 1, mode='edge')

    max_slope = np.zeros((height, width), dtype=np.float32)
    flow_to = np.full((height, width), -1, dtype=np.int32)

    for idx, (dy, dx, dist_mult) in enumerate(neighbors):
        neighbor_z = pad_z[1+dy:height+1+dy, 1+dx:width+1+dx]
        dz = elevation - neighbor_z
        slope = dz / (dist_mult * pixel_size_m)

        better = slope > max_slope
        max_slope[better] = slope[better]
        flow_to[better] = idx

    # Calculate in-degree for topological accumulation
    in_degree = np.zeros((height, width), dtype=np.int32)
    downstream_r = np.full((height, width), -1, dtype=np.int32)
    downstream_c = np.full((height, width), -1, dtype=np.int32)

    for r in range(height):
        for c in range(width):
            dir_idx = flow_to[r, c]
            if dir_idx >= 0:
                dy, dx, _ = neighbors[dir_idx]
                nr, nc = r + dy, c + dx
                if 0 <= nr < height and 0 <= nc < width:
                    downstream_r[r, c] = nr
                    downstream_c[r, c] = nc
                    in_degree[nr, nc] += 1

    # Topological D8 Flow Accumulation using Kahn's algorithm queue
    accum = np.ones((height, width), dtype=np.float32)

    flat_in_deg = in_degree.ravel().copy()
    flat_accum = accum.ravel()
    flat_down_r = downstream_r.ravel()
    flat_down_c = downstream_c.ravel()

    # Start queue with zero in-degree cells (ridge tops)
    zero_deg = np.where(flat_in_deg == 0)[0]
    queue = list(zero_deg)

    head = 0
    while head < len(queue):
        curr = queue[head]
        head += 1

        nr = flat_down_r[curr]
        nc = flat_down_c[curr]

        if nr >= 0 and nc >= 0:
            target_idx = nr * width + nc
            flat_accum[target_idx] += flat_accum[curr]
            flat_in_deg[target_idx] -= 1
            if flat_in_deg[target_idx] == 0:
                queue.append(target_idx)

    catchment_area_m2 = accum * cell_area_m2
    return accum.astype(np.float32), catchment_area_m2.astype(np.float32)


def densify_linestring_segment_points(coords: List[List[float]], step_m: float = 10.0) -> List[List[float]]:
    """
    Interpolates points along a LineString geometry at ~10m intervals.

    Approximation Note:
    -------------------
    Calculating distance to densified line vertices approximates true point-to-segment distance.
    With a 10m step size, the maximum orthogonal distance approximation error is bounded by
    step_m / 2 = 5.0 meters, which is well below the DEM grid resolution (~30.83m).
    """
    if len(coords) < 2:
        return coords

    dense_pts = [coords[0]]
    for i in range(len(coords) - 1):
        p1 = coords[i]
        p2 = coords[i+1]

        dx = (p2[0] - p1[0]) * 111000.0 * 0.97
        dy = (p2[1] - p1[1]) * 111000.0
        seg_dist = float(np.sqrt(dx * dx + dy * dy))

        if seg_dist > step_m:
            num_steps = int(np.ceil(seg_dist / step_m))
            for k in range(1, num_steps):
                t = k / num_steps
                interp_lon = p1[0] + t * (p2[0] - p1[0])
                interp_lat = p1[1] + t * (p2[1] - p1[1])
                dense_pts.append([interp_lon, interp_lat])
        dense_pts.append(p2)

    return dense_pts


def parse_all_drain_vertices_from_kml(kml_path: str, df_drains: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parses LineString geometries in drains.kml, densifying segments every 10m (~291,000 vertices).
    Associates each densified point with the hydraulic Manning capacity Q_cap of its drain feature.
    """
    tree = ET.parse(kml_path)
    root = tree.getroot()
    ns = {'kml': 'http://www.opengis.net/kml/2.2'}

    w = df_drains['width_m'].values if 'width_m' in df_drains.columns else np.full(len(df_drains), 0.5)
    d = df_drains['depth_m'].values if 'depth_m' in df_drains.columns else np.full(len(df_drains), 0.5)
    s = df_drains['slope_m_m'].values if 'slope_m_m' in df_drains.columns else np.full(len(df_drains), 0.001)
    n = df_drains['mannings_n'].values if 'mannings_n' in df_drains.columns else np.full(len(df_drains), 0.015)

    area = w * d
    p = w + 2.0 * d
    r_hyd = area / np.maximum(1e-4, p)
    q_caps = (1.0 / n) * area * (r_hyd ** (2.0 / 3.0)) * np.sqrt(s)

    all_vertex_coords = []
    all_vertex_capacities = []

    placemarks = root.findall('.//kml:Placemark', ns)
    num_drains = min(len(placemarks), len(q_caps))

    for idx in range(num_drains):
        pm = placemarks[idx]
        cap = q_caps[idx]
        coords_elem = pm.find('.//kml:coordinates', ns)
        if coords_elem is not None and coords_elem.text:
            tokens = coords_elem.text.strip().split()
            pts = []
            for t in tokens:
                parts = t.split(',')
                if len(parts) >= 2:
                    try:
                        pts.append([float(parts[0]), float(parts[1])])
                    except ValueError:
                        continue
            
            dense_pts = densify_linestring_segment_points(pts, step_m=10.0)
            for d_pt in dense_pts:
                all_vertex_coords.append(d_pt)
                all_vertex_capacities.append(cap)

    if len(all_vertex_coords) == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float32)

    return np.array(all_vertex_coords, dtype=np.float64), np.array(all_vertex_capacities, dtype=np.float32)


def map_drains_to_grid(
    drains_kml_path: str,
    drainage_csv_path: str,
    grid_bounds: Dict[str, float],
    grid_shape: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes distance to drain geometries and assigns conveyance capacity.

    Coordinate Transformation Note:
    -------------------------------
    Uses a local equirectangular geographic-to-metric approximation:
        dx_m = (lon - ref_lon) * 111000 * cos(mid_lat)
        dy_m = (ref_lat - lat) * 111000
    This metric scaling is a local planar approximation centered on Chennai, NOT full projected UTM Zone 44N.

    Capacity Assignment Note:
    -------------------------
    Cells <= 100m from a drain receive the nearest channel capacity Q_cap.
    Cells > 100m receive DEFAULT_OVERLAND_FLOW_CAPACITY_M3_S (0.05 m³/s) as an ambient sheet-flow fallback.
    """
    height, width = grid_shape
    left, bottom, right, top = grid_bounds['left'], grid_bounds['bottom'], grid_bounds['right'], grid_bounds['top']
    ref_lon, ref_lat = left, top
    mid_lat_rad = np.radians((top + bottom) / 2.0)
    cos_lat = float(np.cos(mid_lat_rad))

    df_drains = pd.read_csv(drainage_csv_path)
    vertex_coords, vertex_caps = parse_all_drain_vertices_from_kml(drains_kml_path, df_drains)

    if len(vertex_coords) == 0:
        return np.full(grid_shape, 500.0, dtype=np.float32), np.full(grid_shape, DEFAULT_OVERLAND_FLOW_CAPACITY_M3_S, dtype=np.float32)

    # Local equirectangular metric conversion
    x_m = (vertex_coords[:, 0] - ref_lon) * 111000.0 * cos_lat
    y_m = (top - vertex_coords[:, 1]) * 111000.0
    drain_metric_pts = np.column_stack([x_m, y_m]).astype(np.float32)

    drain_tree = cKDTree(drain_metric_pts)

    lons = np.linspace(left, right, width, dtype=np.float32)
    lats = np.linspace(top, bottom, height, dtype=np.float32)

    grid_x = (lons - ref_lon) * 111000.0 * cos_lat
    grid_y = (top - lats) * 111000.0

    mesh_x, mesh_y = np.meshgrid(grid_x, grid_y)
    grid_metric_pts = np.column_stack([mesh_x.ravel(), mesh_y.ravel()])

    dists_m, nearest_indices = drain_tree.query(grid_metric_pts, k=1)

    dist_to_drain_m = dists_m.reshape(grid_shape).astype(np.float32)
    nearest_capacities = vertex_caps[nearest_indices].reshape(grid_shape)

    q_cap_grid = np.where(dist_to_drain_m <= 100.0, nearest_capacities, DEFAULT_OVERLAND_FLOW_CAPACITY_M3_S).astype(np.float32)

    return dist_to_drain_m, q_cap_grid


def assemble_feature_matrix(
    terrain_npz_path: str,
    drains_kml_path: str,
    drainage_csv_path: str,
    grid_metadata_path: str,
    rainfall_intensity_mm_hr: float = 80.0,
    runoff_coeff: float = 0.75
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Assembles the 2D feature matrix X (num_cells x 7) in strict FEATURE_NAMES order.
    """
    with open(grid_metadata_path, 'r') as f:
        meta = json.load(f)

    height, width = meta['height'], meta['width']
    grid_shape = (height, width)
    pixel_size_m = float(meta.get('pixel_size_m', 30.83))

    terrain = np.load(terrain_npz_path)
    elevation = terrain['elevation'].astype(np.float32)
    slope = terrain['slope'].astype(np.float32)
    tpi = terrain['tpi'].astype(np.float32)

    # 1. D8 Flow Accumulation & Catchment Area
    _, catchment_area_m2 = compute_flow_accumulation(elevation, pixel_size_m=pixel_size_m)

    # 2. LineString Segment Drain Distance & Capacity Mapping
    dist_to_drain, q_cap_grid = map_drains_to_grid(
        drains_kml_path, drainage_csv_path, meta['bounds'], grid_shape
    )

    # 3. Rational Runoff Demand Q_demand (m³/s)
    q_demand_grid = rational_runoff(
        rainfall_intensity_mm_hr=rainfall_intensity_mm_hr,
        catchment_area_m2=catchment_area_m2,
        runoff_coeff=runoff_coeff
    ).astype(np.float32)

    # 4. Drainage Overload Ratio R_overload
    r_overload_grid = overload_ratio(
        q_demand=q_demand_grid,
        q_cap=q_cap_grid
    ).astype(np.float32)

    # Assemble 2D Feature Matrix in strict FEATURE_NAMES order
    X_2d = np.column_stack([
        elevation.ravel(),
        slope.ravel(),
        tpi.ravel(),
        dist_to_drain.ravel(),
        q_cap_grid.ravel(),
        q_demand_grid.ravel(),
        r_overload_grid.ravel()
    ]).astype(np.float32)

    summary_metadata = {
        "feature_names": FEATURE_NAMES,
        "grid_shape": list(grid_shape),
        "total_cells": int(X_2d.shape[0]),
        "num_features": int(X_2d.shape[1]),
        "rainfall_intensity_mm_hr": float(rainfall_intensity_mm_hr),
        "runoff_coeff": float(runoff_coeff),
        "dtype": "float32"
    }

    return X_2d, summary_metadata


def save_feature_artifacts(
    X: np.ndarray,
    ground_truth_mask_path: str,
    output_npz_path: str
) -> Dict[str, Any]:
    """
    Saves feature matrix X and target y into compressed npz archive.
    """
    y_mask = np.load(ground_truth_mask_path).astype(np.uint8)
    y_flat = y_mask.ravel()

    assert X.shape[0] == y_flat.shape[0], (
        f"Shape mismatch! X rows ({X.shape[0]}) != y elements ({y_flat.shape[0]})"
    )

    np.savez_compressed(
        output_npz_path,
        X=X,
        y=y_flat,
        feature_names=np.array(FEATURE_NAMES)
    )

    file_size_mb = os.path.getsize(output_npz_path) / (1024 * 1024)
    print(f"[OK] Feature matrix saved to {output_npz_path} ({file_size_mb:.2f} MB)")
    
    return {
        "X_shape": list(X.shape),
        "y_shape": list(y_flat.shape),
        "positive_flood_class_count": int(np.sum(y_flat == 1)),
        "negative_dry_class_count": int(np.sum(y_flat == 0)),
        "file_size_mb": round(file_size_mb, 2)
    }
