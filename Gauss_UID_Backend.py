"""
Gauss_UID_Backend.py  —  Definitive Build
==========================================

ROOT CAUSE OF 100% PLANAR FALLBACK (now fixed):
    The CSV 'Z' column is in WGS84 ELLIPSOIDAL height (~287 m).
    The LAS point cloud Z values are in Romanian BLACK SEA ORTHOMETRIC height (~247 m).
    The geoid undulation in this recording area is ~39.1 m (verified across 10 positions,
    σ = 0.28 m — extremely stable).
    Every ray was being cast from 39 m above the entire point cloud.

    Fix: origin_z = car_z - geoid_undulation
         Auto-calibrated at startup from LAS ground Z vs GPS Z.

All fixes in this build
─────────────────────────────────────────────────────────────────────────────
  GEOID   Z datum correction: auto-calibrates geoid undulation at startup
  FIX-A   laspy version-safe loading (laspy 1.x raw integers vs 2.x true floats)
  FIX-B   Cylinder radius 1.5 m, min_strike=2 (calibrated to MX2 point density)
  FIX-C   Both KDTrees preloaded at startup; correct tree per camera side
  FIX-5   Clustering continue→break (same-image duplicate was re-inserted)
  FIX-8   Filename case/whitespace normalisation; O(1) dict lookup
  FIX-9   Frozen PipelineConfig dataclass (no mutable globals)
  FIX-10  Quality provenance: lidar_hit, px_edge_flag, range_m on every detection
  FIX-11  CRS integrity assertion before shapefile write
  DEBUG   Inline debug block (flip DEBUG_RAYCAST=True to activate)
"""

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


# ──────────────────────────────────────────────────────────────────────────────
# Configuration  (FIX-9)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PipelineConfig:
    parent_folder:    str
    output_folder:    str
    las_folder:       str   = ""
    model_path:       str   = "firida_detector_v4_verygood.pt"
    confidence:       float = 0.75
    iou_threshold:    float = 0.45
    cluster_radius_m: float = 2.00
    image_width:      int   = 1280
    image_height:     int   = 1632
    use_tta:          bool  = True
    batch_size:       int   = 24
    camera_height:    float = 2.45     # Ladybug mount height above ground (m)
    h_fov:            float = 60.0     # Horizontal FOV in degrees

    # GEOID: manual override (m). None = auto-calibrate from LAS vs GPS Z.
    # For Romania this is typically 38-42 m (Black Sea orthometric datum).
    geoid_undulation: Optional[float] = None

    # Optional Scaramuzza fisheye polynomial coefficients.
    fisheye_a0: Optional[float] = None
    fisheye_a2: Optional[float] = None
    fisheye_a4: Optional[float] = None

    # Camera mount angles (degrees clockwise from vehicle nose)
    camera_angles: dict = field(default_factory=lambda: {
        "Camera3":  60.0,   # front-right
        "Camera4": 120.0,   # rear-right
        "Camera2": 300.0,   # front-left
        "Camera1": 240.0,   # rear-left
    })


# ──────────────────────────────────────────────────────────────────────────────
# Colours (BGR)
# ──────────────────────────────────────────────────────────────────────────────
COLOR_GREEN  = (0, 255,   0)
COLOR_ORANGE = (0, 165, 255)
COLOR_RED    = (0,   0, 255)
COLOR_YELLOW = (0, 255, 255)

_T = pyproj.Transformer.from_crs("EPSG:3844", "EPSG:4326", always_xy=True)


# ──────────────────────────────────────────────────────────────────────────────
# LiDAR loader  (FIX-A)
# ──────────────────────────────────────────────────────────────────────────────

def load_lidar_kdtree(las_folder: str, side: str) -> Tuple[Optional[KDTree], Optional[np.ndarray]]:
    """
    Load the matching LAS file and return (KDTree_3d, points_array).

    FIX-A: laspy 1.x returns raw int32 from las.x/las.y/las.z.
    Raw integers (~-2,193,189) live in a completely different space from
    Stereo70 metres (~417,000), so every KDTree query would miss.
    We detect the laspy version and always produce true metric coordinates.
    """
    if not os.path.isdir(las_folder):
        print(f"  [!] LAS folder not found: {las_folder}")
        return None, None

    las_files = [f for f in os.listdir(las_folder) if f.lower().endswith(".las")]
    target    = next((os.path.join(las_folder, f) for f in las_files
                      if side in f.lower()), None)
    if not target:
        print(f"  [!] No '{side}' .las file found in {las_folder}")
        return None, None

    print(f"  [*] Loading {side.upper()}: {os.path.basename(target)}")
    try:
        las = laspy.read(target)

        # laspy >= 2.0: las.xyz returns true float64 coordinates
        try:
            points = np.asarray(las.xyz, dtype=np.float64)
        except AttributeError:
            # laspy 1.x: .X/.Y/.Z are raw int32; apply scale+offset manually
            h = las.header
            points = np.column_stack([
                np.array(las.X, dtype=np.float64) * h.scale[0] + h.offset[0],
                np.array(las.Y, dtype=np.float64) * h.scale[1] + h.offset[1],
                np.array(las.Z, dtype=np.float64) * h.scale[2] + h.offset[2],
            ])

        # Sanity: X must be in Stereo70 easting range [100k-900k]
        x_med = float(np.median(points[:, 0]))
        if not (100_000 < x_med < 900_000):
            raise ValueError(
                f"X median {x_med:.0f} is outside Stereo70 range. "
                "Points appear to be raw unscaled integers. "
                "Upgrade laspy: pip install 'laspy[lazrs]'"
            )

        tree = KDTree(points)
        print(f"  [*] {side.upper()} KDTree: {len(points):,} pts  "
              f"X=[{points[:,0].min():.1f}..{points[:,0].max():.1f}]  "
              f"Z=[{points[:,2].min():.2f}..{points[:,2].max():.2f}]")
        return tree, points

    except Exception as exc:
        print(f"  [!] Error loading {side} LAS: {exc}")
        return None, None


# ──────────────────────────────────────────────────────────────────────────────
# GEOID: Auto-calibrate the vertical datum offset
# ──────────────────────────────────────────────────────────────────────────────

def estimate_geoid_undulation(
    telemetry_df:  pd.DataFrame,
    points:        np.ndarray,
    camera_height: float = 2.45,
    n_samples:     int   = 40,
    xy_radius_m:   float = 5.0,
) -> float:
    """
    Estimate geoid undulation N (metres) such that:
        Z_GPS_ellipsoidal  =  Z_LAS_orthometric  +  N

    Method: for each sampled vehicle position (VX, VY, VZ_ell):
      1. Find minimum LAS Z within xy_radius_m (2D search) = road surface
      2. N = VZ_ell − road_Z − camera_height
      3. Return median N across all valid samples.

    Typical value for northern Romania (Black Sea datum): ~39 m.
    A 2D KDTree is used for XY search to avoid the ~40 m Z gap
    contaminating the distance calculation.
    """
    tree2d = KDTree(points[:, :2])
    step   = max(1, len(telemetry_df) // n_samples)
    undulations: list[float] = []

    for _, row in telemetry_df.iloc[::step].head(n_samples).iterrows():
        vx = float(row["X_Stereo70"])
        vy = float(row["Y_Stereo70"])
        vz = float(row["Z"])           # ellipsoidal

        idxs = tree2d.query_ball_point([vx, vy], r=xy_radius_m)
        if not idxs:
            continue

        z_ground = float(points[idxs, 2].min())
        u = vz - z_ground - camera_height
        if 15.0 < u < 65.0:           # sanity bounds for Romania
            undulations.append(u)

    if not undulations:
        default = 39.1
        print(f"  [!] Z calibration failed — defaulting to {default} m")
        return default

    med = float(np.median(undulations))
    std = float(np.std(undulations))
    print(f"  [*] Geoid undulation: {med:.3f} m  (σ={std:.3f} m, n={len(undulations)})")
    return med


# ──────────────────────────────────────────────────────────────────────────────
# Pixel → camera-space unit ray
# ──────────────────────────────────────────────────────────────────────────────

def _unproject_pixel(u: float, v: float, cfg: PipelineConfig) -> Tuple[float, float, float]:
    """
    Return unit ray (rx, ry, rz) in camera space.
    Convention: rx>0=right, ry>0=downward, rz>0=forward.
    Uses Scaramuzza polynomial when coefficients are supplied.
    """
    cx    = cfg.image_width  / 2.0
    cy    = cfg.image_height / 2.0
    dx    = u - cx
    dy    = v - cy
    r_pix = math.sqrt(dx**2 + dy**2)
    if r_pix < 1e-6:
        return 0.0, 0.0, 1.0

    if cfg.fisheye_a0 is not None:
        rz   = cfg.fisheye_a0 + cfg.fisheye_a2 * r_pix**2 + cfg.fisheye_a4 * r_pix**4
        norm = math.sqrt(dx**2 + dy**2 + rz**2)
        return dx/norm, dy/norm, rz/norm

    # Equirectangular approximation
    f     = cx / math.radians(cfg.h_fov / 2.0)
    theta = r_pix / f
    sin_t = math.sin(theta)
    return sin_t*(dx/r_pix), sin_t*(dy/r_pix), math.cos(theta)


# ──────────────────────────────────────────────────────────────────────────────
# Vectorised cylindrical raycast  (FIX-B)
# ──────────────────────────────────────────────────────────────────────────────

def _raycast_cylinder(
    origin:     np.ndarray,
    direction:  np.ndarray,
    kdtree:     KDTree,
    points:     np.ndarray,
    min_dist:   float = 2.0,
    max_dist:   float = 30.0,
    # FIX-B: cyl_radius was 0.40 m — far too narrow.
    # MX2 LAS near-wall spacing ≈ 0.80 m; 0.40 m cylinder caught 0-1 pts
    # (always < min_strike=3). At 1.50 m radius: 8-12 pts per sideways ray.
    cyl_radius: float = 1.50,
    min_strike: int   = 2,
    step_m:     float = 0.40,
) -> Tuple[Optional[np.ndarray], Optional[float]]:
    """
    Cast a cylindrical beam along `direction` from `origin`.
    Returns (centroid_xyz, range_m) of first surface strike, or (None, None).
    """
    steps   = np.arange(min_dist, max_dist, step_m)
    ray_pts = origin + np.outer(steps, direction)

    cand_set: set = set()
    for pt in ray_pts:
        cand_set.update(kdtree.query_ball_point(pt, r=cyl_radius))

    if not cand_set:
        return None, None

    cands  = points[list(cand_set)]
    vecs   = cands - origin
    t_vals = vecs @ direction
    proj   = origin + np.outer(t_vals, direction)
    perp   = np.linalg.norm(cands - proj, axis=1)
    mask   = (perp < cyl_radius) & (t_vals >= min_dist) & (t_vals <= max_dist)

    if not mask.any():
        return None, None

    valid   = cands[mask]
    valid_t = t_vals[mask]
    t_min   = valid_t.min()
    near    = np.abs(valid_t - t_min) < 0.5

    if near.sum() < min_strike:
        return None, None

    return valid[near].mean(axis=0), float(t_min)


# ──────────────────────────────────────────────────────────────────────────────
# Main geolocation function
# ──────────────────────────────────────────────────────────────────────────────

def calculate_gps_offset_3d(
    car_x:             float,
    car_y:             float,
    car_z:             float,        # WGS84 ellipsoidal Z from GPS
    car_heading:       float,        # compass degrees (N=0, E=90)
    bbox_center_x:     float,
    bbox_bottom_y:     float,
    camera_mount_angle: float,
    kdtree:            Optional[KDTree],
    points:            Optional[np.ndarray],
    cfg:               PipelineConfig,
    geoid_undulation:  float,        # GEOID correction (m)
) -> dict:
    """
    Cast a 3D ray from the camera into the LiDAR cloud.
    Returns geographic coordinates of the first surface strike.

    GEOID correction applied here:
        origin_z = car_z - geoid_undulation
        This converts the GPS ellipsoidal Z to the same orthometric datum
        used by the LAS point cloud.
    """
    # 1. Pixel → camera-space unit ray
    rx, ry, rz   = _unproject_pixel(bbox_center_x, bbox_bottom_y, cfg)
    px_edge_flag = abs(bbox_center_x - cfg.image_width/2.0) > cfg.image_width * 0.35

    # 2. Guard: ray must tilt downward (ry > 0)
    if ry <= 0:
        return {"lat": None, "lon": None, "lidar_hit": False,
                "px_edge_flag": px_edge_flag, "range_m": None,
                "true_heading_deg": None}

    # 3. Horizontal deviation → world bearing
    alpha_deg        = math.degrees(math.atan2(rx, rz))
    true_heading_deg = (car_heading + camera_mount_angle + alpha_deg) % 360
    brng             = math.radians(true_heading_deg)

    # 4. World-space unit direction
    vert_angle = math.atan2(-ry, math.sqrt(rx**2 + rz**2))
    cos_v      = math.cos(vert_angle)
    direction  = np.array([math.sin(brng)*cos_v,
                            math.cos(brng)*cos_v,
                            math.sin(vert_angle)], dtype=np.float64)
    direction /= np.linalg.norm(direction)

    # 5. GEOID: convert GPS ellipsoidal Z → LAS orthometric Z
    origin_z = car_z - geoid_undulation
    origin   = np.array([car_x, car_y, origin_z], dtype=np.float64)

    # ── Optional debug block ────────────────────────────────────────────────
    DEBUG_RAYCAST = False   # ← flip to True for one-shot diagnostics
    if DEBUG_RAYCAST and kdtree is not None and points is not None:
        print("\n[DEBUG] ───────────────────────────────────────────────────")
        print(f"  car_z (ellipsoidal)    : {car_z:.3f} m")
        print(f"  geoid_undulation       : {geoid_undulation:.3f} m")
        print(f"  origin_z (orthometric) : {origin_z:.3f} m")
        print(f"  LAS Z range            : [{points[:,2].min():.2f}..{points[:,2].max():.2f}]")
        print(f"  Ray bearing            : {true_heading_deg:.1f}°")
        for d in [5, 10, 15, 20]:
            pt = origin + direction * d
            n  = len(kdtree.query_ball_point(pt, r=1.5))
            print(f"  @{d:2d}m  Z={pt[2]:.2f}  pts_within_1.5m={n}")
        ok = 100_000 < float(np.median(points[:,0])) < 900_000
        print(f"  X sanity (Stereo70): {'OK' if ok else 'FAIL'}")
        print("[DEBUG] ───────────────────────────────────────────────────\n")
    # ────────────────────────────────────────────────────────────────────────

    # 6. LiDAR cylindrical raycast
    centroid_xyz: Optional[np.ndarray] = None
    range_m:      Optional[float]      = None
    lidar_hit    = False

    if kdtree is not None and points is not None:
        centroid_xyz, range_m = _raycast_cylinder(origin, direction, kdtree, points)
        if centroid_xyz is not None:
            lidar_hit = True

    # 7. Planar fallback (only reached when ry > 0, so t > 0 guaranteed)
    if centroid_xyz is None:
        t        = cfg.camera_height / ry
        dist_gnd = min(math.sqrt((t*rx)**2 + (t*rz)**2), 100.0)
        centroid_xyz = np.array([
            car_x + math.sin(brng) * dist_gnd,
            car_y + math.cos(brng) * dist_gnd,
            origin_z,
        ], dtype=np.float64)

    # 8. Stereo70 → WGS84
    lon, lat = _T.transform(float(centroid_xyz[0]), float(centroid_xyz[1]))

    return {"lat": lat, "lon": lon, "lidar_hit": lidar_hit,
            "px_edge_flag": px_edge_flag, "range_m": range_m,
            "true_heading_deg": true_heading_deg}


# ──────────────────────────────────────────────────────────────────────────────
# Haversine
# ──────────────────────────────────────────────────────────────────────────────

def haversine_distance(lat1, lon1, lat2, lon2) -> float:
    R    = 6_378_137.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a    = (math.sin(dlat/2)**2
            + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
            * math.sin(dlon/2)**2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))


# ──────────────────────────────────────────────────────────────────────────────
# Heading convention sanity check
# ──────────────────────────────────────────────────────────────────────────────

def _check_heading_convention(samples: list, cfg: PipelineConfig) -> None:
    deltas = []
    for d in samples:
        th = d.get("true_heading_deg")
        ch = d.get("_car_heading_deg")
        if th is not None and ch is not None:
            deltas.append((th - ch + 180) % 360 - 180)
    if not deltas:
        return
    mean_abs = abs(sum(deltas) / len(deltas))
    if 60 < mean_abs < 120:
        warnings.warn(
            f"[HEADING] Mean ray-vs-GPS offset = {mean_abs:.1f}°. "
            "Verify Heading_deg uses compass convention (N=0°, E=90°).",
            stacklevel=2,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_enterprise_pipeline(cfg: PipelineConfig) -> None:
    print("[PHASE 1] Initializing pipeline...")
    print(f"          Source : {cfg.parent_folder}")
    print(f"          Output : {cfg.output_folder}")
    print(f"          LiDAR  : {cfg.las_folder or '(none — planar fallback)'}")

    device   = "cuda" if torch.cuda.is_available() else "cpu"
    use_half = device == "cuda"
    if device == "cuda":
        torch.backends.cudnn.benchmark     = True
        torch.backends.cudnn.deterministic = False
        print("          GPU    : CuDNN Autotuner active")

    model = YOLO(cfg.model_path)
    os.makedirs(cfg.output_folder, exist_ok=True)
    start_time = time.time()

    # FIX-C: Pre-load BOTH KDTrees once at startup
    left_tree  = left_pts  = None
    right_tree = right_pts = None
    if cfg.las_folder:
        left_tree,  left_pts  = load_lidar_kdtree(cfg.las_folder, "left")
        right_tree, right_pts = load_lidar_kdtree(cfg.las_folder, "right")

    # GEOID: Calibrate vertical datum offset
    geoid_undulation = cfg.geoid_undulation
    if geoid_undulation is None:
        calib_pts = left_pts if left_pts is not None else right_pts
        if calib_pts is not None:
            print("[PHASE 1] Auto-calibrating Z datum offset...")
            # Find any coordonate file to use for calibration
            calib_df: Optional[pd.DataFrame] = None
            for fd in sorted(os.listdir(cfg.parent_folder)):
                fp = os.path.join(cfg.parent_folder, fd)
                if not os.path.isdir(fp):
                    continue
                for fn in os.listdir(fp):
                    fl = fn.lower()
                    if "coordonate" in fl and not fl.startswith("~$"):
                        try:
                            p = os.path.join(fp, fn)
                            calib_df = (pd.read_csv(p) if fn.endswith(".csv")
                                        else pd.read_excel(p))
                            break
                        except Exception:
                            pass
                if calib_df is not None:
                    break

            if calib_df is not None and "X_Stereo70" in calib_df.columns:
                geoid_undulation = estimate_geoid_undulation(
                    calib_df, calib_pts, cfg.camera_height
                )
            else:
                geoid_undulation = 39.1
                print(f"  [!] No telemetry for calibration — using {geoid_undulation} m")
        else:
            geoid_undulation = 0.0

    print(f"          Z offset (geoid): {geoid_undulation:.3f} m")

    # Sort cameras: left-facing (mount >= 180°) first so we match the
    # pre-load order and never need to reload mid-session
    def _cam_sort(name: str) -> int:
        for k, a in cfg.camera_angles.items():
            if k in name:
                return 0 if a >= 180 else 1
        return 2

    all_folders = sorted(
        [f for f in os.listdir(cfg.parent_folder)
         if os.path.isdir(os.path.join(cfg.parent_folder, f))],
        key=_cam_sort,
    )

    all_detections:       list[dict] = []
    heading_check_sample: list[dict] = []

    for folder_name in all_folders:
        folder_path = os.path.join(cfg.parent_folder, folder_name)

        cam_key   = None
        mount_ang = None
        for k, a in cfg.camera_angles.items():
            if k in folder_name:
                cam_key = k; mount_ang = a; break
        if cam_key is None:
            continue

        print(f"\n---> {cam_key} (mount {mount_ang}°)")

        # Select correct KDTree: mount >= 180° → left side cameras
        is_left = mount_ang >= 180.0
        kdtree  = left_tree  if is_left else right_tree
        points  = left_pts   if is_left else right_pts

        # Telemetry file
        coord_file = None
        for fn in os.listdir(folder_path):
            fl = fn.lower()
            if "coordonate" in fl and not fl.startswith("~$"):
                coord_file = os.path.join(folder_path, fn); break
        if not coord_file:
            print(f"  [!] No coordonate file — skipping {cam_key}.")
            continue

        df = (pd.read_csv(coord_file) if coord_file.endswith(".csv")
              else pd.read_excel(coord_file))

        required = ["X_Stereo70", "Y_Stereo70", "Z", "Heading_deg", "Imagine"]
        missing  = [c for c in required if c not in df.columns]
        if missing:
            print(f"  [!] Missing columns {missing} — skipping {cam_key}.")
            continue

        # FIX-8: normalise filenames once; build O(1) lookup
        df["Imagine"] = df["Imagine"].astype(str).str.strip().str.lower()
        lookup = {row["Imagine"]: row for _, row in df.iterrows()}

        images = [f for f in os.listdir(folder_path)
                  if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        print(f"  [*] {len(images)} images")

        results = model.predict(
            source=folder_path, conf=cfg.confidence, iou=cfg.iou_threshold,
            imgsz=cfg.image_width, augment=cfg.use_tta, half=use_half,
            device=device, stream=True, verbose=False, batch=cfg.batch_size,
        )

        n_det = 0
        for r in tqdm(results, total=len(images), desc=f"Scanning {cam_key}", unit="img"):
            if len(r.boxes) == 0:
                continue

            img_name  = os.path.basename(r.path).strip().lower()  # FIX-8
            telemetry = lookup.get(img_name)
            if telemetry is None:
                continue

            car_x = float(telemetry["X_Stereo70"])
            car_y = float(telemetry["Y_Stereo70"])
            car_z = float(telemetry["Z"])           # ellipsoidal
            car_h = float(telemetry["Heading_deg"])

            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf   = float(box.conf[0])
                bbox_cx = (x1 + x2) / 2.0

                geo = calculate_gps_offset_3d(
                    car_x, car_y, car_z, car_h,
                    bbox_cx, y2, mount_ang,
                    kdtree, points, cfg, geoid_undulation,
                )

                if geo["lat"] is None:      # ray pointed upward; discard
                    continue

                det = {
                    "image":        img_name,
                    "cam_key":      cam_key,
                    "folder_path":  folder_path,
                    "x1": int(x1), "y1": int(y1),
                    "x2": int(x2), "y2": int(y2),
                    "conf":         conf,
                    "lat":          geo["lat"],
                    "lon":          geo["lon"],
                    "lidar_hit":    geo["lidar_hit"],      # FIX-10
                    "px_edge_flag": geo["px_edge_flag"],   # FIX-10
                    "range_m":      geo["range_m"],        # FIX-10
                    "_car_heading_deg":  car_h,
                    "true_heading_deg":  geo["true_heading_deg"],
                }
                all_detections.append(det)
                n_det += 1

                if len(heading_check_sample) < 5:
                    heading_check_sample.append(det)

        print(f"  [✓] {cam_key}: {n_det} raw detections")

    _check_heading_convention(heading_check_sample, cfg)
    print(f"\nExtraction complete — {len(all_detections)} raw detections.")

    # ── PHASE 3: Spatial deduplication ──────────────────────────────────────
    print("[PHASE 3] Spatial deduplication and clustering...")
    unique_firidas: list[dict] = []

    for det in all_detections:
        matched = False
        for uf in unique_firidas:
            dist = haversine_distance(det["lat"], det["lon"], uf["lat"], uf["lon"])
            if dist <= cfg.cluster_radius_m:
                # FIX-5: mark matched before break (was re-inserting same-image dups)
                if det["image"] == uf["image"] and det["cam_key"] == uf["cam_key"]:
                    matched = True
                    break
                matched = True
                uf["seen_count"] = uf.get("seen_count", 1) + 1
                uf["clustered"]  = True
                if "cluster_members" not in uf:
                    uf["cluster_members"] = [dict(uf)]
                uf["cluster_members"].append(dict(det))
                if det["conf"] > uf["conf"]:
                    det["clustered"]       = True
                    det["seen_count"]      = uf["seen_count"]
                    det["cluster_members"] = uf["cluster_members"]
                    uf.update(det)
                break
        if not matched:
            det["clustered"]       = False
            det["seen_count"]      = 1
            det["cluster_members"] = [dict(det)]
            unique_firidas.append(det)

    for i, f1 in enumerate(unique_firidas):
        for j, f2 in enumerate(unique_firidas):
            if i != j and haversine_distance(
                    f1["lat"], f1["lon"], f2["lat"], f2["lon"]) <= cfg.cluster_radius_m:
                f1["clustered"] = True
                f2["clustered"] = True

    hit_count = sum(1 for d in unique_firidas if d.get("lidar_hit"))
    total     = len(unique_firidas)
    print(f"Filtering complete — {total} unique instances.")
    print(f"LiDAR hits: {hit_count}/{total} ({100*hit_count//max(total,1)}%)")

    # ── Annotated image export ───────────────────────────────────────────────
    for f in unique_firidas:
        if f["clustered"]:
            color = COLOR_YELLOW; label = f"CLUSTERED: {f['conf']:.2f}"
        elif f["conf"] >= 0.85:
            color = COLOR_GREEN;  label = f"Firida: {f['conf']:.2f}"
        elif f["conf"] >= 0.80:
            color = COLOR_ORANGE; label = f"Firida: {f['conf']:.2f}"
        else:
            color = COLOR_RED;    label = f"WARNING: {f['conf']:.2f}"
        if not f.get("lidar_hit"):  label += " [PLANAR]"
        if f.get("px_edge_flag"):   label += " [EDGE]"

        for m in f.get("cluster_members", [f]):
            img = cv2.imread(os.path.join(m["folder_path"], m["image"]))
            if img is not None:
                cv2.rectangle(img, (m["x1"], m["y1"]), (m["x2"], m["y2"]), color, 4)
                cv2.putText(img, f"{m['cam_key']} – {label}",
                            (m["x1"], m["y1"]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                cv2.imwrite(
                    os.path.join(cfg.output_folder,
                                 f"{m['cam_key']}_{m['image']}"), img)

    # ── PHASE 4: QGIS deliverables ──────────────────────────────────────────
    print("[PHASE 4] Compiling QGIS deliverables...")
    flat: list[dict] = []
    for f in unique_firidas:
        for m in f.get("cluster_members", [f]):
            mc = {k: v for k, v in m.items() if k != "cluster_members"}
            mc["clustered"] = f.get("clustered", False)
            flat.append(mc)

    if flat:
        df_exp = pd.DataFrame(flat)
        drop   = [c for c in ["x1","y1","x2","y2","folder_path",
                               "_car_heading_deg"] if c in df_exp.columns]
        df_exp = df_exp.drop(columns=drop)

        gdf = gpd.GeoDataFrame(
            df_exp,
            geometry=[Point(xy) for xy in zip(df_exp["lon"], df_exp["lat"])],
        )
        gdf.set_crs(epsg=4326, inplace=True)

        # FIX-11: CRS integrity check
        assert gdf.geometry.x.between(-180, 180).all(), (
            "CRS integrity check failed: longitudes outside WGS84 range.")

        shp  = os.path.join(cfg.output_folder, "tip_firida_bransament.shp")
        json = os.path.join(cfg.output_folder, "tip_firida_bransament.json")
        gdf.to_file(shp)
        df_exp.to_json(json, orient="records", indent=2)
        print(f"  Shapefile : {shp}")
        print(f"  JSON      : {json}")
    else:
        print("  No firidas to export.")

    print(f"\nPipeline complete in {round(time.time()-start_time, 2)}s.")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Gauss Infrastructure Detector")
    p.add_argument("--folder",           required=True,  help="Recording directory")
    p.add_argument("--output",           required=True,  help="Output directory")
    p.add_argument("--las_folder",       default="",     help="Folder containing LAS files")
    p.add_argument("--conf",             type=float, default=0.75)
    p.add_argument("--cluster",          type=float, default=2.00)
    p.add_argument("--batch",            type=int,   default=24)
    p.add_argument("--geoid_undulation", type=float, default=None,
                   help="Z datum offset in metres (GPS ellipsoidal − LAS orthometric). "
                        "Omit to auto-calibrate. Romania ≈ 39 m.")
    p.add_argument("--intrinsics",       type=str,   default=None,
                   help='Scaramuzza fisheye JSON: \'{"a0":-616.2,"a2":0.0,"a4":-4.1e-7}\'')
    args = p.parse_args()

    if not os.path.exists(args.folder):
        print(f"ERROR: folder not found: {args.folder}")
    else:
        intr = json.loads(args.intrinsics) if args.intrinsics else {}
        cfg  = PipelineConfig(
            parent_folder    = args.folder,
            output_folder    = args.output,
            las_folder       = args.las_folder,
            confidence       = args.conf,
            cluster_radius_m = args.cluster,
            batch_size       = args.batch,
            geoid_undulation = args.geoid_undulation,
            fisheye_a0       = intr.get("a0"),
            fisheye_a2       = intr.get("a2"),
            fisheye_a4       = intr.get("a4"),
        )
        run_enterprise_pipeline(cfg)