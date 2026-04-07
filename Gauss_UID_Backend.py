from __future__ import annotations

import argparse
import json
import math
import os
import time
import warnings
from dataclasses import dataclass, field
from typing import Optional, Tuple

import cv2
import geopandas as gpd
import laspy
import numpy as np
import pandas as pd
import pyproj
import torch
from scipy.spatial import KDTree
from shapely.geometry import Point
from tqdm import tqdm
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# FIX-9 — Frozen configuration dataclass (replaces module-level mutable globals)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PipelineConfig:
    parent_folder:   str
    output_folder:   str
    las_folder:      str   = ""
    model_path:      str   = "firida_detector_v4_verygood.pt"
    confidence:      float = 0.75
    iou_threshold:   float = 0.45
    cluster_radius_m: float = 2.00
    image_width:     int   = 1280
    image_height:    int   = 1632
    use_tta:         bool  = True
    batch_size:      int   = 24
    camera_height:   float = 2.45   # metres above ground
    h_fov:           float = 60.0   # horizontal FOV in degrees
    # FIX-2 — Optional Scaramuzza fisheye polynomial coefficients [a0, a2, a4].
    # Supply via --intrinsics '{"a0": -616.2, "a2": 0.0, "a4": -4.1e-7}'
    # When None the pipeline falls back to the equirectangular approximation.
    fisheye_a0:      Optional[float] = None
    fisheye_a2:      Optional[float] = None
    fisheye_a4:      Optional[float] = None

    # Camera mount angles (degrees, clockwise from vehicle nose)
    camera_angles: dict = field(default_factory=lambda: {
        "Camera3":  60.0,
        "Camera4": 120.0,
        "Camera2": 300.0,
        "Camera1": 240.0,
    })


# ---------------------------------------------------------------------------
# Bounding-box overlay colours (BGR)
# ---------------------------------------------------------------------------
COLOR_GREEN  = (0, 255,   0)
COLOR_ORANGE = (0, 165, 255)
COLOR_RED    = (0,   0, 255)
COLOR_YELLOW = (0, 255, 255)

# ---------------------------------------------------------------------------
# Coordinate transformer — module-level singleton is fine (thread-safe read)
# ---------------------------------------------------------------------------
_transformer_to_wgs84 = pyproj.Transformer.from_crs(
    "EPSG:3844", "EPSG:4326", always_xy=True
)

# ---------------------------------------------------------------------------
# FIX-1  Helper: resolve which LAS hemisphere a world-bearing ray falls in
# ---------------------------------------------------------------------------

def _las_side_for_bearing(true_heading_deg: float) -> str:
    """
    Return 'right' if the ray points into the starboard hemisphere (0–180°),
    'left' for the port hemisphere (180–360°).

    This replaces the old camera-number heuristic which misclassified Camera2
    (mount 300° → rear-right) as 'left' and Camera3 (mount 60° → front-right)
    as 'right' regardless of the actual ray direction after horizontal deviation.
    """
    return "right" if true_heading_deg % 360 < 180 else "left"


# ---------------------------------------------------------------------------
# LiDAR loader
# ---------------------------------------------------------------------------

def load_lidar_kdtree(
    las_folder: str, side: str
) -> Tuple[Optional[KDTree], Optional[np.ndarray]]:
    """
    Load the left or right .las file and return (KDTree, points).
    `side` must be 'left' or 'right'.
    """
    if not os.path.isdir(las_folder):
        print(f"  [!] LAS folder not found: {las_folder}")
        return None, None

    las_files = [f for f in os.listdir(las_folder) if f.lower().endswith(".las")]
    target_las = next(
        (os.path.join(las_folder, f) for f in las_files if side in f.lower()), None
    )

    if not target_las:
        print(f"  [!] No {side.upper()} .las file in {las_folder}")
        return None, None

    print(f"  [*] Loading point cloud ({side}): {target_las}")
    try:
        las    = laspy.read(target_las)
        points = np.vstack((las.x, las.y, las.z)).T
        tree   = KDTree(points)
        print(f"  [*] KDTree built — {len(points):,} points.")
        return tree, points
    except Exception as exc:
        print(f"  [!] Error loading LiDAR: {exc}")
        return None, None


# ---------------------------------------------------------------------------
# FIX-2  Fisheye unproject (Scaramuzza / OCamCalib unified spherical model)
# ---------------------------------------------------------------------------

def _unproject_pixel(
    u: float, v: float, cfg: PipelineConfig
) -> Tuple[float, float, float]:
    """
    Return a unit ray direction (rx, ry, rz) in camera space for pixel (u, v).

    When Scaramuzza coefficients are available the polynomial model is used;
    otherwise falls back to the equirectangular approximation that was present
    in the original code.

    Camera-space convention:
        rx — rightward  (positive = pixel right of centre)
        ry — downward   (positive = pixel below centre)
        rz — forward    (optical axis)
    """
    cx = cfg.image_width  / 2.0
    cy = cfg.image_height / 2.0
    dx = u - cx
    dy = v - cy

    if cfg.fisheye_a0 is not None and cfg.fisheye_a2 is not None and cfg.fisheye_a4 is not None:
        # --- Scaramuzza polynomial model ---
        r_pix = math.sqrt(dx ** 2 + dy ** 2)
        if r_pix < 1e-6:
            return 0.0, 0.0, 1.0
        rz = cfg.fisheye_a0 + cfg.fisheye_a2 * r_pix ** 2 + cfg.fisheye_a4 * r_pix ** 4
        norm = math.sqrt(dx ** 2 + dy ** 2 + rz ** 2)
        return dx / norm, dy / norm, rz / norm
    else:
        # --- Equirectangular approximation (original behaviour) ---
        r_pix = math.sqrt(dx ** 2 + dy ** 2)
        if r_pix < 1e-6:
            return 0.0, 0.0, 1.0
        f_equi = cx / math.radians(cfg.h_fov / 2.0)
        theta  = r_pix / f_equi
        sin_t  = math.sin(theta)
        return sin_t * (dx / r_pix), sin_t * (dy / r_pix), math.cos(theta)


# ---------------------------------------------------------------------------
# FIX-7  Vectorized cylindrical raycast (replaces 115-iteration stepping loop)
# ---------------------------------------------------------------------------

def _raycast_cylinder(
    origin:    np.ndarray,          # shape (3,) — Stereo70 XYZ of vehicle
    direction: np.ndarray,          # shape (3,) — unit world-space ray
    kdtree:    KDTree,
    points:    np.ndarray,          # shape (N, 3)
    min_dist:  float = 2.0,
    max_dist:  float = 25.0,
    cyl_radius: float = 0.40,       # cylinder search radius (m)
    min_strike: int   = 3,          # minimum points to confirm a hit
) -> Tuple[Optional[np.ndarray], Optional[float]]:
    """
    Cast a cylindrical beam along `direction` from `origin` and return
    (centroid_xyz, range_m) of the first surface strike, or (None, None).

    Strategy:
        1. Sample the ray at 0.5 m intervals (coarse harvest) to collect
           candidate point indices in one vectorised pass.
        2. Project each candidate onto the ray axis and discard those outside
           the true cylinder radius — this eliminates the 0.1 m overlap
           redundancy of the old 0.2 m / 0.3 m stepping approach.
        3. Sort surviving candidates by along-ray distance and return the
           centroid of the first dense cluster (within 0.5 m of the closest hit).
    """
    steps     = np.arange(min_dist, max_dist, 0.5)           # ~46 samples
    ray_pts   = origin + np.outer(steps, direction)           # (46, 3)

    # Harvest all point indices within cyl_radius of any ray sample
    candidate_set: set = set()
    for pt in ray_pts:
        idxs = kdtree.query_ball_point(pt, r=cyl_radius)
        candidate_set.update(idxs)

    if not candidate_set:
        return None, None

    cands  = points[list(candidate_set)]                      # (M, 3)
    vecs   = cands - origin                                   # (M, 3)
    t_vals = vecs @ direction                                  # scalar projection (M,)

    # Perpendicular distance from each candidate to the ray axis
    closest_on_ray = origin + np.outer(t_vals, direction)
    perp_dist      = np.linalg.norm(cands - closest_on_ray, axis=1)

    # Keep only those truly inside the cylinder and in front of the camera
    mask = (perp_dist < cyl_radius) & (t_vals >= min_dist) & (t_vals <= max_dist)
    if not mask.any():
        return None, None

    valid   = cands[mask]
    valid_t = t_vals[mask]

    # First-strike cluster: all valid points within 0.5 m of the nearest hit
    t_min   = valid_t.min()
    near    = np.abs(valid_t - t_min) < 0.5
    if near.sum() < min_strike:
        return None, None

    centroid  = valid[near].mean(axis=0)   # (3,) Stereo70 XYZ
    range_m   = float(t_min)
    return centroid, range_m


# ---------------------------------------------------------------------------
# Main geolocation function
# ---------------------------------------------------------------------------

def calculate_gps_offset_3d(
    origin_x:          float,
    origin_y:          float,
    origin_z:          float,
    car_heading:       float,   # degrees, compass convention (N=0, E=90)
    bbox_center_x:     float,   # pixel x of detection centre
    bbox_bottom_y:     float,   # pixel y of detection bottom edge
    camera_mount_angle: float,  # degrees clockwise from vehicle nose
    kdtree:            Optional[KDTree],
    points:            Optional[np.ndarray],
    cfg:               PipelineConfig,
) -> dict:
    """
    Cast a 3D ray and intersect with the LiDAR point cloud to find the true
    Stereo70 coordinates of the detected object, then project to WGS84.

    Returns a dict with keys:
        lat, lon         — WGS84 coordinates
        lidar_hit        — FIX-10: True when LiDAR intersection succeeded
        px_edge_flag     — FIX-2:  True when bbox centre is in distortion zone
        range_m          — FIX-10: distance to struck surface (None if fallback)
        true_heading_deg — resolved world bearing of the ray (used by FIX-1)
    """
    # ------------------------------------------------------------------
    # Step 1 — unproject pixel to camera-space unit ray
    # ------------------------------------------------------------------
    # FIX-2: use the correct lens model (fisheye or equirectangular)
    rx, ry, rz = _unproject_pixel(bbox_center_x, bbox_bottom_y, cfg)

    # FIX-2: flag detections in the outer 35 % of the horizontal FOV where
    #         equirectangular error is largest (> ~0.5 m at 10 m range).
    cx          = cfg.image_width / 2.0
    px_edge_flag = abs(bbox_center_x - cx) > cfg.image_width * 0.35

    # ------------------------------------------------------------------
    # Step 2 — FIX-3: guard against upward/horizontal rays
    # ------------------------------------------------------------------
    # ry > 0 means the pixel is below image centre, i.e. the ray tilts downward.
    # ry ≤ 0 means the ray is horizontal or points skyward; the planar fallback
    # would compute a negative or infinite travel time, placing the point behind
    # or infinitely far from the vehicle.
    if ry <= 0:
        return {
            "lat": None, "lon": None,
            "lidar_hit": False, "px_edge_flag": px_edge_flag, "range_m": None,
            "true_heading_deg": None,
        }

    # ------------------------------------------------------------------
    # Step 3 — horizontal deviation and true world bearing
    # ------------------------------------------------------------------
    alpha_deg        = math.degrees(math.atan2(rx, rz))
    true_heading_deg = (car_heading + camera_mount_angle + alpha_deg) % 360
    brng             = math.radians(true_heading_deg)

    # ------------------------------------------------------------------
    # Step 4 — build world-space unit direction vector
    # ------------------------------------------------------------------
    vertical_angle = math.atan2(-ry, math.sqrt(rx ** 2 + rz ** 2))
    cos_v = math.cos(vertical_angle)
    dir_e = math.sin(brng) * cos_v
    dir_n = math.cos(brng) * cos_v
    dir_z = math.sin(vertical_angle)
    direction = np.array([dir_e, dir_n, dir_z], dtype=np.float64)
    norm = np.linalg.norm(direction)
    if norm > 0:
        direction /= norm
    origin = np.array([origin_x, origin_y, origin_z], dtype=np.float64)

    # ------------------------------------------------------------------
    # Step 5 — FIX-1 + FIX-7: choose correct LAS hemisphere, then raycast
    # ------------------------------------------------------------------
    centroid_xyz = None
    range_m      = None
    lidar_hit    = False

    if kdtree is not None and points is not None:
        centroid_xyz, range_m = _raycast_cylinder(origin, direction, kdtree, points)
        if centroid_xyz is not None:
            lidar_hit = True

    # ------------------------------------------------------------------
    # Step 6 — planar fallback (FIX-3: only reachable when ry > 0)
    # ------------------------------------------------------------------
    if centroid_xyz is None:
        t        = cfg.camera_height / ry
        dist_gnd = min(math.sqrt((t * rx) ** 2 + (t * rz) ** 2), 100.0)
        cx_world = origin_x + math.sin(brng) * dist_gnd
        cy_world = origin_y + math.cos(brng) * dist_gnd
        centroid_xyz = np.array([cx_world, cy_world, origin_z], dtype=np.float64)

    # ------------------------------------------------------------------
    # Step 7 — project Stereo70 → WGS84 (always_xy=True enforced globally)
    # ------------------------------------------------------------------
    lon, lat = _transformer_to_wgs84.transform(
        float(centroid_xyz[0]), float(centroid_xyz[1])
    )

    return {
        "lat":              lat,
        "lon":              lon,
        "lidar_hit":        lidar_hit,           # FIX-10
        "px_edge_flag":     px_edge_flag,        # FIX-2 / FIX-10
        "range_m":          range_m,             # FIX-10
        "true_heading_deg": true_heading_deg,    # FIX-1 (passed back to caller)
    }


# ---------------------------------------------------------------------------
# Haversine distance
# ---------------------------------------------------------------------------

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance in metres between two WGS84 points."""
    R    = 6_378_137.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a    = (math.sin(dlat / 2) ** 2
            + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
            * math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ---------------------------------------------------------------------------
# FIX-4  Heading-convention sanity check
# ---------------------------------------------------------------------------

def _check_heading_convention(detections_sample: list, cfg: PipelineConfig) -> None:
    """
    Verify that the GPS heading convention is compass-bearing (N=0, E=90).
    For each of the first few detections we compute the signed angular difference
    between the resolved true_heading_deg and the car_heading stored in the record.
    If the mean deviation is near ±90° we warn the operator.
    """
    if not detections_sample:
        return
    deltas = []
    for d in detections_sample:
        th = d.get("true_heading_deg")
        ch = d.get("_car_heading_deg")
        if th is None or ch is None:
            continue
        diff = (th - ch + 180) % 360 - 180   # signed difference in (-180, 180]
        deltas.append(diff)
    if not deltas:
        return
    mean_abs = abs(sum(deltas) / len(deltas))
    if 60 < mean_abs < 120:
        warnings.warn(
            f"[FIX-4] Heading convention mismatch suspected: mean ray-vs-GPS "
            f"bearing offset is {mean_abs:.1f}°. "
            "Verify that Heading_deg in the coordinate file uses a compass bearing "
            "(North = 0°, East = 90°) not a mathematical angle (East = 0°).",
            stacklevel=2,
        )


# ---------------------------------------------------------------------------
# FIX-9  Main pipeline — accepts config explicitly, no global mutation
# ---------------------------------------------------------------------------

def run_enterprise_pipeline(cfg: PipelineConfig) -> None:
    print("[PHASE 1] Initializing 3D Point Cloud Spatial Pipeline...")
    print(f"          Output  : {cfg.output_folder}")
    print(f"          Source  : {cfg.parent_folder}")
    print(f"          LiDAR  : {cfg.las_folder or '(none — planar fallback only)'}")
    if cfg.fisheye_a0 is not None:
        print(f"          Lens   : Scaramuzza fisheye (a0={cfg.fisheye_a0})")
    else:
        print("          Lens   : equirectangular approximation")

    device   = "cuda" if torch.cuda.is_available() else "cpu"
    use_half = device == "cuda"
    if device == "cuda":
        print("          GPU    : CuDNN Autotuner active")
        torch.backends.cudnn.benchmark    = True
        torch.backends.cudnn.deterministic = False

    model = YOLO(cfg.model_path)
    os.makedirs(cfg.output_folder, exist_ok=True)

    all_detections: list[dict] = []
    start_time = time.time()

    # ------------------------------------------------------------------
    # FIX-6 — Deterministic camera ordering: left-facing cameras first so
    #          the KDTree is loaded at most twice (left → right).
    # ------------------------------------------------------------------
    LEFT_CAMERAS  = {"Camera1", "Camera2"}
    RIGHT_CAMERAS = {"Camera3", "Camera4"}

    def _cam_sort_key(folder_name: str) -> int:
        for key in LEFT_CAMERAS:
            if key in folder_name:
                return 0    # left group processed first
        for key in RIGHT_CAMERAS:
            if key in folder_name:
                return 1    # right group second
        return 2            # unknown — last

    all_folders = sorted(
        [f for f in os.listdir(cfg.parent_folder)
         if os.path.isdir(os.path.join(cfg.parent_folder, f))],
        key=_cam_sort_key,
    )

    loaded_kdtree: Optional[KDTree] = None
    loaded_points: Optional[np.ndarray] = None
    current_lidar_side: Optional[str] = None

    heading_check_sample: list[dict] = []   # FIX-4

    for folder_name in all_folders:
        folder_path = os.path.join(cfg.parent_folder, folder_name)

        # Identify which camera this folder belongs to
        current_cam_key    = None
        current_mount_angle = None
        for cam_key, angle in cfg.camera_angles.items():
            if cam_key in folder_name:
                current_cam_key     = cam_key
                current_mount_angle = angle
                break

        if not current_cam_key or current_mount_angle is None:
            continue

        print(f"\n---> Initializing {current_cam_key} (mount {current_mount_angle}°)")

        # ------------------------------------------------------------------
        # Coordinate / telemetry file
        # ------------------------------------------------------------------
        coord_file_path = None
        for file in os.listdir(folder_path):
            fl = file.lower()
            if "coordonate" in fl and not fl.startswith("~$"):
                coord_file_path = os.path.join(folder_path, file)
                break

        if not coord_file_path:
            print(f"  [!] Skipping {current_cam_key}: no 'coordonate' file found.")
            continue

        df_coords = (
            pd.read_csv(coord_file_path)
            if coord_file_path.endswith(".csv")
            else pd.read_excel(coord_file_path)
        )

        required_cols = ["X_Stereo70", "Y_Stereo70", "Z", "Heading_deg", "Imagine"]
        missing = [c for c in required_cols if c not in df_coords.columns]
        if missing:
            print(f"  [!] Missing columns {missing} — skipping {current_cam_key}.")
            continue

        # FIX-8 — Normalise image-name column to lowercase + stripped whitespace
        df_coords["Imagine"] = df_coords["Imagine"].astype(str).str.strip().str.lower()

        # Build a lookup dict for O(1) image → row access
        coord_lookup = {
            row["Imagine"]: row for _, row in df_coords.iterrows()
        }

        image_files  = [
            f for f in os.listdir(folder_path)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ]
        total_images = len(image_files)
        print(f"  [*] {total_images} images to process")

        # ------------------------------------------------------------------
        # YOLO inference (streaming)
        # ------------------------------------------------------------------
        results = model.predict(
            source=folder_path,
            conf=cfg.confidence,
            iou=cfg.iou_threshold,
            imgsz=cfg.image_width,
            augment=cfg.use_tta,
            half=use_half,
            device=device,
            stream=True,
            verbose=False,
            batch=cfg.batch_size,
        )

        folder_detections = 0

        for r in tqdm(results, total=total_images, desc=f"Scanning {current_cam_key}", unit="img"):
            if len(r.boxes) == 0:
                continue

            # FIX-8 — normalise runtime filename before lookup
            img_name       = os.path.basename(r.path).strip().lower()
            telemetry      = coord_lookup.get(img_name)
            if telemetry is None:
                continue

            car_x       = float(telemetry["X_Stereo70"])
            car_y       = float(telemetry["Y_Stereo70"])
            car_z       = float(telemetry["Z"])
            car_heading = float(telemetry["Heading_deg"])

            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf            = float(box.conf[0])
                bbox_center_x   = (x1 + x2) / 2.0

                # ----------------------------------------------------------
                # FIX-1 + FIX-7 — resolve heading BEFORE choosing LAS side
                # We need a preliminary heading estimate so we can pick the
                # correct hemisphere. We compute alpha_deg from the pixel first.
                # ----------------------------------------------------------
                rx_prelim, _, rz_prelim = _unproject_pixel(bbox_center_x, y2, cfg)
                alpha_prelim  = math.degrees(math.atan2(rx_prelim, rz_prelim))
                prelim_heading = (car_heading + current_mount_angle + alpha_prelim) % 360

                required_side = _las_side_for_bearing(prelim_heading)  # FIX-1

                # Reload KDTree only when the hemisphere actually changes
                if current_lidar_side != required_side:
                    if cfg.las_folder:
                        loaded_kdtree, loaded_points = load_lidar_kdtree(
                            cfg.las_folder, required_side
                        )
                    current_lidar_side = required_side

                # ----------------------------------------------------------
                # Full geolocation
                # ----------------------------------------------------------
                geo = calculate_gps_offset_3d(
                    car_x, car_y, car_z,
                    car_heading, bbox_center_x, y2,
                    current_mount_angle,
                    loaded_kdtree, loaded_points,
                    cfg,
                )

                # FIX-3 — discard rays that pointed upward
                if geo["lat"] is None:
                    continue

                det = {
                    "image":        img_name,
                    "cam_key":      current_cam_key,
                    "folder_path":  folder_path,
                    "x1":           int(x1),
                    "y1":           int(y1),
                    "x2":           int(x2),
                    "y2":           int(y2),
                    "conf":         conf,
                    "lat":          geo["lat"],
                    "lon":          geo["lon"],
                    "lidar_hit":    geo["lidar_hit"],       # FIX-10
                    "px_edge_flag": geo["px_edge_flag"],    # FIX-10
                    "range_m":      geo["range_m"],         # FIX-10
                    # Private field for heading sanity-check (stripped at export)
                    "_car_heading_deg":  car_heading,
                    "true_heading_deg":  geo["true_heading_deg"],
                }
                all_detections.append(det)
                folder_detections += 1

                # FIX-4 — collect sample for heading-convention check
                if len(heading_check_sample) < 5:
                    heading_check_sample.append(det)

        print(f"  [✓] {current_cam_key} complete — {folder_detections} raw detections.")

    # FIX-4 — run heading-convention check after the first camera
    _check_heading_convention(heading_check_sample, cfg)

    print(f"\nExtraction complete — {len(all_detections)} raw detections.")

    # ------------------------------------------------------------------
    # PHASE 3 — Spatial deduplication and clustering
    # ------------------------------------------------------------------
    print("[PHASE 3] Spatial deduplication and proximity clustering...")

    unique_firidas: list[dict] = []

    for det in all_detections:
        matched = False

        for uf in unique_firidas:
            dist = haversine_distance(det["lat"], det["lon"], uf["lat"], uf["lon"])

            if dist <= cfg.cluster_radius_m:
                # FIX-5 — same-image / same-camera duplicate: mark matched and
                #          skip merging, but do NOT leave matched=False, which
                #          previously caused re-insertion as a new unique firida.
                if det["image"] == uf["image"] and det["cam_key"] == uf["cam_key"]:
                    matched = True   # ← was `continue` without setting matched
                    break

                matched = True
                uf["seen_count"] = uf.get("seen_count", 1) + 1
                uf["clustered"]  = True

                if "cluster_members" not in uf:
                    uf["cluster_members"] = [dict(uf)]
                uf["cluster_members"].append(dict(det))

                if det["conf"] > uf["conf"]:
                    det["clustered"]  = True
                    det["seen_count"] = uf["seen_count"]
                    det["cluster_members"] = uf["cluster_members"]
                    uf.update(det)
                break

        if not matched:
            det["clustered"]  = False
            det["seen_count"] = 1
            det["cluster_members"] = [dict(det)]
            unique_firidas.append(det)

    # Secondary mutual proximity flag
    for i, f1 in enumerate(unique_firidas):
        for j, f2 in enumerate(unique_firidas):
            if i != j and haversine_distance(f1["lat"], f1["lon"], f2["lat"], f2["lon"]) <= cfg.cluster_radius_m:
                f1["clustered"] = True
                f2["clustered"] = True

    print(f"Spatial filtering complete — {len(unique_firidas)} unique instances.")

    # ------------------------------------------------------------------
    # Annotated image export
    # ------------------------------------------------------------------
    for f in unique_firidas:
        if f["clustered"]:
            color = COLOR_YELLOW
            label = f"CLUSTERED: {f['conf']:.2f}"
        elif f["conf"] >= 0.85:
            color = COLOR_GREEN
            label = f"Firida: {f['conf']:.2f}"
        elif f["conf"] >= 0.80:
            color = COLOR_ORANGE
            label = f"Firida: {f['conf']:.2f}"
        else:
            color = COLOR_RED
            label = f"WARNING: {f['conf']:.2f}"

        # FIX-10 — append data-quality indicator to image label
        if not f["lidar_hit"]:
            label += " [PLANAR]"
        if f["px_edge_flag"]:
            label += " [EDGE]"

        members = f.get("cluster_members", [f])

        for member in members:
            img_path = os.path.join(member["folder_path"], member["image"])
            img = cv2.imread(img_path)
            if img is not None:
                cv2.rectangle(img, (member["x1"], member["y1"]), (member["x2"], member["y2"]), color, 4)
                
                # Context prefix
                member_label = f"{member['cam_key']} - {label}"
                
                cv2.putText(
                    img, member_label, (member["x1"], member["y1"] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2,
                )
                cv2.imwrite(
                    os.path.join(cfg.output_folder, f"{member['cam_key']}_{member['image']}"), img
                )

    # ------------------------------------------------------------------
    # PHASE 4 — QGIS deliverables
    # ------------------------------------------------------------------
    print("[PHASE 4] Compiling QGIS deliverables...")

    flattened_firidas = []
    if unique_firidas:
        for f in unique_firidas:
            members = f.get("cluster_members", [f])
            for m in members:
                m_copy = dict(m)
                m_copy["clustered"] = f.get("clustered", False)
                if "cluster_members" in m_copy:
                    del m_copy["cluster_members"]
                flattened_firidas.append(m_copy)

    if flattened_firidas:
        df_export = pd.DataFrame(flattened_firidas)

        # Strip internal / geometry columns before writing
        drop_cols = [c for c in ["x1", "y1", "x2", "y2", "folder_path",
                                  "_car_heading_deg"] if c in df_export.columns]
        df_export = df_export.drop(columns=drop_cols)

        geometry = [Point(xy) for xy in zip(df_export["lon"], df_export["lat"])]
        gdf      = gpd.GeoDataFrame(df_export, geometry=geometry)
        gdf.set_crs(epsg=4326, inplace=True)

        # FIX-11 — CRS integrity assertion before writing
        lon_vals = gdf.geometry.x
        assert lon_vals.between(-180, 180).all(), (
            "CRS integrity check failed: longitude values outside WGS84 range. "
            "The Stereo70 → WGS84 transform may not have been applied correctly."
        )

        shp_path  = os.path.join(cfg.output_folder, "tip_firida_bransament.shp")
        json_path = os.path.join(cfg.output_folder, "tip_firida_bransament.json")

        gdf.to_file(shp_path)
        df_export.to_json(json_path, orient="records", indent=2)

        lidar_count = int(df_export["lidar_hit"].sum()) if "lidar_hit" in df_export else "?"
        edge_count  = int(df_export["px_edge_flag"].sum()) if "px_edge_flag" in df_export else "?"
        print(f"  Shapefile  : {shp_path}")
        print(f"  LiDAR hits : {lidar_count} / {len(df_export)} detections")
        print(f"  Edge flags : {edge_count} (reduced coordinate accuracy)")
    else:
        print("  No firidas to export.")

    elapsed = round(time.time() - start_time, 2)
    print(f"\nPipeline complete in {elapsed}s.")


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gauss Infrastructure Detector Backend")
    parser.add_argument("--folder",      type=str, required=True,  help="Recording directory")
    parser.add_argument("--output",      type=str, required=True,  help="Output directory")
    parser.add_argument("--las_folder",  type=str, default="",     help="LiDAR .las folder")
    parser.add_argument("--conf",        type=float, default=0.75, help="YOLO confidence threshold (0–1)")
    parser.add_argument("--cluster",     type=float, default=2.00, help="Cluster/dedup radius (m)")
    parser.add_argument("--batch",       type=int,   default=24,   help="YOLO batch size")
    # FIX-2 — optional fisheye intrinsics as a JSON string
    parser.add_argument(
        "--intrinsics",
        type=str,
        default=None,
        help=(
            'Scaramuzza fisheye coefficients as JSON, e.g. '
            '\'{"a0": -616.2, "a2": 0.0, "a4": -4.1e-7}\'. '
            "Omit to use equirectangular approximation."
        ),
    )
    args = parser.parse_args()

    if not os.path.exists(args.folder):
        print(f"ERROR: Source folder not found: {args.folder}")
    else:
        intrinsics = json.loads(args.intrinsics) if args.intrinsics else {}
        config = PipelineConfig(
            parent_folder    = args.folder,
            output_folder    = args.output,
            las_folder       = args.las_folder,
            confidence       = args.conf,
            cluster_radius_m = args.cluster,
            batch_size       = args.batch,
            fisheye_a0       = intrinsics.get("a0"),
            fisheye_a2       = intrinsics.get("a2"),
            fisheye_a4       = intrinsics.get("a4"),
        )
        run_enterprise_pipeline(config)