from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import time
import warnings
from dataclasses import dataclass, field
from typing import Optional, Tuple

# Restrict threading at the C/C++ level BEFORE importing heavy numerical libraries
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"


import cv2
import geopandas as gpd
import laspy
import numpy as np
import pandas as pd
import pyproj
import torch
import hashlib
import tempfile
import scipy.interpolate
from scipy.spatial import KDTree
from shapely.geometry import Point
from tqdm import tqdm
from ultralytics import YOLO

# Limit PyTorch and OpenCV threads to reduce excessive CPU usage on high-core-count processors (like i9 13900K)
torch.set_num_threads(4)
cv2.setNumThreads(4)
# ---
# Config
# ---

@dataclass(frozen=True)
class PipelineConfig:
    # required
    parent_folder:    str
    output_folder:    str

    # paths
    las_folder:       str   = ""
    model_path:       str   = "firida_detector_v4_verygood.pt"

    # detection thresholds
    confidence:       float = 0.75
    iou_threshold:    float = 0.45
    cluster_radius_m:       float = 2.00   # tight same-camera dedup radius (metres)
    cross_camera_radius_m:  float = 8.00   # wider cross-camera merge radius (metres).
                                            # rays from different camera angles land a few metres
                                            # apart even for the same firida, so we need a looser
                                            # threshold here than the same-camera pass.

    # camera / image settings
    image_width:      int   = 1280
    image_height:     int   = 1632
    use_tta:          bool  = True   # test-time augmentation, helps catch edge cases
    batch_size:       int   = 24
    camera_height:    float = 2.45   # ladybug sits about 2.45m off the ground
    h_fov:            float = 60.0   # horizontal field of view per camera in degrees

    # vertical datum correction — if None we calculate it automatically from the data.
    # for this recording area in Romania it ends up being ~39.1m.
    # you can hardcode it here if you already know it for a given recording area.
    geoid_undulation: Optional[float] = None

    # Ladybug hardware calibration file (.cal).  When set, the precise per-lens
    # pinhole intrinsics + extrinsic rotation matrices from the file are used for
    # unprojection using the .cal B-Spline mesh for all active lenses.
    ladybug_cal_path: Optional[str] = None

    # cam_key → Ladybug camera ID mapping derived from angular geometry of the rig.
    # Override if the physical wiring differs from this default.
    camera_lb_ids: dict = field(default_factory=lambda: {
        "Camera1": 3,   # rear-left   (~216° CW)
        "Camera2": 4,   # front-left  (~288° CW)
        "Camera3": 1,   # front-right (~72°  CW)
        "Camera4": 2,   # rear-right  (~144° CW)
    })


    # how each camera is rotated relative to the car's forward direction (clockwise degrees)
    camera_angles: dict = field(default_factory=lambda: {
        "Camera3":  60.0,   # front-right
        "Camera4": 120.0,   # rear-right
        "Camera2": 300.0,   # front-left
        "Camera1": 240.0,   # rear-left
    })


# bounding box colors in BGR because OpenCV is BGR for some reason
COLOR_GREEN  = (0, 255,   0)   # high confidence
COLOR_ORANGE = (0, 165, 255)   # medium confidence
COLOR_RED    = (0,   0, 255)   # low confidence, worth reviewing
COLOR_YELLOW = (0, 255, 255)   # clustered — seen from multiple angles

# single shared transformer, no need to recreate it every call


# ---
# LiDAR loading
# ---

def load_lidar_kdtree(las_folder: str, side: str) -> Tuple[Optional[KDTree], Optional[np.ndarray]]:
    """
    Loads a left or right LAS file and builds a KDTree from it.
    Returns (tree, points) or (None, None) if something goes wrong.

    One thing to watch out for: laspy 1.x returns raw integers from las.x/las.y/las.z,
    not actual coordinates. Those integers are things like -2,193,189 instead of 417,248.
    We handle both versions — laspy 2.x has las.xyz which just works, and for 1.x we
    manually apply the scale and offset from the file header.
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

        try:
            # laspy 2.x — this just gives us the real coordinates directly
            points = np.asarray(las.xyz, dtype=np.float64)
        except AttributeError:
            # laspy 1.x — have to do it manually
            h = las.header
            points = np.column_stack([
                np.array(las.X, dtype=np.float64) * h.scale[0] + h.offset[0],
                np.array(las.Y, dtype=np.float64) * h.scale[1] + h.offset[1],
                np.array(las.Z, dtype=np.float64) * h.scale[2] + h.offset[2],
            ])

        # quick sanity check — Stereo70 easting should be somewhere between 100k and 900k.
        # if we're getting something wildly outside that range we probably got raw integers
        # instead of real coordinates and the KDTree will be useless.
        x_med = float(np.median(points[:, 0]))
        if not (100_000 < x_med < 900_000):
            raise ValueError(
                f"X median is {x_med:.0f}, which is outside the expected Stereo70 range. "
                "Looks like we got raw unscaled integers from laspy. "
                "Try upgrading: pip install 'laspy[lazrs]'"
            )

        tree = KDTree(points)
        print(f"  [*] {side.upper()} KDTree ready — {len(points):,} points  "
              f"X=[{points[:,0].min():.1f}..{points[:,0].max():.1f}]  "
              f"Z=[{points[:,2].min():.2f}..{points[:,2].max():.2f}]")
        return tree, points

    except Exception as exc:
        print(f"  [!] Failed to load {side} LAS: {exc}")
        return None, None


# ---
# Z datum calibration
# ---

def estimate_geoid_undulation(
    telemetry_df:  pd.DataFrame,
    points:        np.ndarray,
    camera_height: float = 2.45,
    n_samples:     int   = 40,
    xy_radius_m:   float = 5.0,
) -> float:
    """
    Figures out the vertical offset between the GPS altitude in the CSV
    and the Z values in the LAS file.

    The CSV gives us WGS84 ellipsoidal height. The LAS gives us orthometric
    height above the Black Sea. Those are different things and the gap between
    them (the geoid undulation) needs to be subtracted before we can raycast.

    We measure it empirically: for each vehicle position we find the lowest
    LAS point within a few metres, subtract the camera height, and compare
    to the GPS altitude. The median across ~40 samples is our correction value.

    We use a 2D KDTree (XY only) for the search here because if we searched
    in 3D the ~39m Z gap would mean we'd never find any nearby points.
    """
    tree2d = KDTree(points[:, :2])
    step   = max(1, len(telemetry_df) // n_samples)
    undulations: list[float] = []

    for _, row in telemetry_df.iloc[::step].head(n_samples).iterrows():
        vx = float(row["X_Stereo70"])
        vy = float(row["Y_Stereo70"])
        vz = float(row["Z"])   # this is the ellipsoidal GPS altitude

        idxs = tree2d.query_ball_point([vx, vy], r=xy_radius_m)
        if not idxs:
            continue

        # lowest Z in the neighbourhood = road surface
        z_ground = float(points[idxs, 2].min())
        u = vz - z_ground - camera_height

        # 15–65m is the sane range for Romania, anything outside is probably noise
        if 15.0 < u < 65.0:
            undulations.append(u)

    if not undulations:
        # shouldn't happen with a valid LAS file, but just in case
        default = 39.1
        print(f"  [!] Couldn't calibrate Z offset automatically, falling back to {default}m")
        return default

    med = float(np.median(undulations))
    std = float(np.std(undulations))
    print(f"  [*] Z offset: {med:.3f}m  (std={std:.3f}m across {len(undulations)} samples)")
    return med


# ---
# Ray math
# ---
#
# Ladybug 5+ calibration constants — these are properties of the camera hardware
# and never change for a given physical camera serial number. The .cal file
# encodes intrinsics and distortion for a 2448x2048 sensor frame per camera.
LADYBUG_SENSOR_W      = 2448
LADYBUG_SENSOR_H      = 2048
LADYBUG_SPLINE_DEGREE = 3      # cubic, indicated by 4-fold repeated end knots

# Two module-level caches, both keyed in a way that survives across function calls
# but is bounded — there's only one .cal file in practice, and a small fixed set
# of cameras within it.
_LB_CAL_CACHE: dict = {}  # path -> {'cameras': {...}, 'warps': {...}}
_LB_LUT_CACHE: dict = {}  # (path, lb_id) -> ndarray of shape (W, H, 3) in working frame


def _parse_ladybug_cal(path: str) -> dict:
    """
    Parse a Ladybug .cal file fully — cameras AND warp blocks — and validate
    everything aggressively. Cached so this only happens once per run.

    Parses:
      - Per-camera intrinsics:  focalLength, Center (principal point)
      - Per-camera extrinsics:  CamToLadybugEulerZYX
      - Per-camera warp IDs:    RectifiedSpline, DistortedSpline
      - Tensor-product B-spline warp blocks: knots, coefficient grids

    Validates (raises ValueError if anything is off):
      - File exists and is readable
      - Every camera has all required fields
      - Focal lengths are non-zero (catches corrupt cal files)
      - Rotation matrices are orthonormal (det == 1, R @ R.T == I)
      - Warp ID references resolve to defined warp blocks
      - Knot count matches the cubic spline relation N_knots == N_coefs + degree + 1

    Returns:
        {
          'cameras': { lb_id (int) -> {
              'fl_x', 'fl_y', 'cx_n', 'cy_n', 'R',
              'rect_warp_u_id', 'rect_warp_v_id',
              'dist_warp_u_id', 'dist_warp_v_id',
          }, ... },
          'warps': { warp_id (int) -> {
              'knots_x', 'knots_y', 'coefs_2d', 'kx', 'ky',
          }, ... },
        }

    Cam frame convention:    X=right, Y=down, Z=forward (standard pinhole)
    Ladybug body convention: X=forward, Y=left, Z=up
    Rotation R takes camera frame → Ladybug body frame.
    """
    if path in _LB_CAL_CACHE:
        return _LB_CAL_CACHE[path]

    if not path or not os.path.isfile(path):
        raise ValueError(f"Ladybug calibration file not found: {path!r}")

    cameras: dict  = {}
    warps:   dict  = {}
    cur_cam: dict  = {}
    cur_warp: dict = {}
    cur_block      = None    # 'camera' | 'warp' | None
    reading_coefs  = False   # True while parsing the multi-line Coefs payload

    with open(path, 'r') as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            tok = line.split()
            key = tok[0]

            # Section delimiters
            if key == 'BeginCamera':
                cur_cam = {}
                cur_block = 'camera'
                continue
            if key == 'EndCamera':
                if 'id' in cur_cam:
                    cameras[cur_cam['id']] = cur_cam
                cur_cam   = {}
                cur_block = None
                continue
            if key == 'BeginWarp':
                cur_warp      = {'_coef_buffer': []}
                cur_block     = 'warp'
                reading_coefs = False
                continue
            if key == 'EndWarp':
                if 'id' in cur_warp:
                    nx, ny = cur_warp.get('num_coefs', (0, 0))
                    buf    = cur_warp.pop('_coef_buffer')
                    if nx == 0 or ny == 0:
                        raise ValueError(f"Warp {cur_warp['id']}: NumberCoefs missing")
                    if len(buf) != nx * ny:
                        raise ValueError(
                            f"Warp {cur_warp['id']}: expected {nx*ny} coefs "
                            f"({nx} x {ny}), got {len(buf)}"
                        )
                    # FITPACK row-major: c[i*ny + j], achieved by reshape(nx, ny).flatten()
                    cur_warp['coefs_2d'] = np.array(buf, dtype=np.float64).reshape(nx, ny)
                    cur_warp['kx']       = LADYBUG_SPLINE_DEGREE
                    cur_warp['ky']       = LADYBUG_SPLINE_DEGREE
                    warps[cur_warp['id']] = cur_warp
                cur_warp      = {}
                cur_block     = None
                reading_coefs = False
                continue

            # Camera fields
            if cur_block == 'camera':
                if   key == 'Id':                     cur_cam['id']   = int(tok[1])
                elif key == 'focalLength':            cur_cam['fl_x'] = float(tok[1]); cur_cam['fl_y'] = float(tok[2])
                elif key == 'Center':                 cur_cam['cx_n'] = float(tok[4]); cur_cam['cy_n'] = float(tok[5])
                elif key == 'RectifiedSpline':        cur_cam['rect_warp_u_id'] = int(tok[1]); cur_cam['rect_warp_v_id'] = int(tok[2])
                elif key == 'DistortedSpline':        cur_cam['dist_warp_u_id'] = int(tok[1]); cur_cam['dist_warp_v_id'] = int(tok[2])
                elif key == 'CamToLadybugEulerZYX':
                    cur_cam['euler'] = (float(tok[1]), float(tok[2]), float(tok[3]))
                continue

            # Warp fields
            if cur_block == 'warp':
                if key == 'Id':
                    cur_warp['id'] = int(tok[1])
                    reading_coefs = False
                elif key == 'NumberKnots':
                    cur_warp['num_knots'] = int(tok[1])
                    reading_coefs = False
                elif key == 'KnotsX':
                    cur_warp['knots_x'] = np.array([float(x) for x in tok[1:]], dtype=np.float64)
                    reading_coefs = False
                elif key == 'KnotsY':
                    cur_warp['knots_y'] = np.array([float(x) for x in tok[1:]], dtype=np.float64)
                    reading_coefs = False
                elif key == 'NumberCoefs':
                    cur_warp['num_coefs'] = (int(tok[1]), int(tok[2]))
                    reading_coefs = False
                elif key == 'Coefs':
                    reading_coefs = True
                    cur_warp['_coef_buffer'].extend(float(x) for x in tok[1:])
                elif reading_coefs:
                    # Continuation lines for the Coefs block — pure numbers
                    cur_warp['_coef_buffer'].extend(float(x) for x in tok)
                continue

    # ---
    # Validate cameras and build rotation matrices with orthonormality checks
    # ---
    if not cameras:
        raise ValueError(f"No camera blocks parsed from {path}")

    for cam_id, cam in cameras.items():
        for req in ('fl_x', 'fl_y', 'cx_n', 'cy_n', 'euler',
                    'rect_warp_u_id', 'rect_warp_v_id'):
            if req not in cam:
                raise ValueError(f"Camera {cam_id}: missing required field '{req}'")

        if cam['fl_x'] == 0.0 or cam['fl_y'] == 0.0:
            raise ValueError(f"Camera {cam_id}: focal length is zero — corrupt cal file")

        rx_e, ry_e, rz_e = cam['euler']
        cX, sX = math.cos(rx_e), math.sin(rx_e)
        cY, sY = math.cos(ry_e), math.sin(ry_e)
        cZ, sZ = math.cos(rz_e), math.sin(rz_e)
        mat_Rx = np.array([[1, 0, 0], [0, cX, -sX], [0, sX, cX]], dtype=np.float64)
        mat_Ry = np.array([[cY, 0, sY], [0, 1, 0], [-sY, 0, cY]], dtype=np.float64)
        mat_Rz = np.array([[cZ, -sZ, 0], [sZ, cZ, 0], [0, 0, 1]], dtype=np.float64)
        R = mat_Rz @ mat_Ry @ mat_Rx

        # Orthonormality guardrails — catch corrupted Euler angles before they
        # produce drifted-but-plausible coordinates downstream.
        det = float(np.linalg.det(R))
        if abs(det - 1.0) > 1e-6:
            raise ValueError(
                f"Camera {cam_id}: rotation matrix not orthonormal "
                f"(det={det:.6f}, expected 1.0)"
            )
        if not np.allclose(R @ R.T, np.eye(3), atol=1e-6):
            raise ValueError(
                f"Camera {cam_id}: rotation matrix is not orthogonal (R @ R.T != I)"
            )
        cam['R'] = R

    # ---
    # Validate warps and the camera → warp references
    # ---
    for cam_id, cam in cameras.items():
        for ref_key in ('rect_warp_u_id', 'rect_warp_v_id'):
            wid = cam[ref_key]
            if wid not in warps:
                raise ValueError(
                    f"Camera {cam_id}: references undefined warp ID {wid} ({ref_key})"
                )

    for warp_id, warp in warps.items():
        for req in ('knots_x', 'knots_y', 'num_coefs', 'coefs_2d'):
            if req not in warp:
                raise ValueError(f"Warp {warp_id}: missing '{req}'")
        kx, ky = warp['kx'], warp['ky']
        nx, ny = warp['num_coefs']
        # Cubic B-spline relation: N_knots = N_coefs + degree + 1
        if len(warp['knots_x']) != nx + kx + 1:
            raise ValueError(
                f"Warp {warp_id}: knots_x has {len(warp['knots_x'])} entries, "
                f"expected {nx + kx + 1} (num_coefs_x + kx + 1)"
            )
        if len(warp['knots_y']) != ny + ky + 1:
            raise ValueError(
                f"Warp {warp_id}: knots_y has {len(warp['knots_y'])} entries, "
                f"expected {ny + ky + 1} (num_coefs_y + ky + 1)"
            )

    result = {'cameras': cameras, 'warps': warps}
    _LB_CAL_CACHE[path] = result
    return result


def _lut_disk_path(cal_path: str, lb_id: int) -> str:
    """
    Path on disk where a per-camera LUT is cached. The filename includes a hash
    of the .cal file's contents so that if the cal file ever changes (it won't,
    but defensive), the cache is automatically invalidated.
    """
    with open(cal_path, 'rb') as fh:
        cal_hash = hashlib.md5(fh.read()).hexdigest()[:12]
    cache_dir = os.path.join(tempfile.gettempdir(), 'gauss_ladybug_lut_cache')
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"lut_{cal_hash}_cam{lb_id}.npy")


def _build_ladybug_lut(cal_path: str, lb_id: int) -> np.ndarray:
    """
    Build (or load from disk cache) a (sensor_w, sensor_h, 3) lookup table
    that maps every DISTORTED sensor pixel to its corresponding unit ray
    vector in our working frame: (rx=right, ry=down, rz=forward).

    The LUT eliminates the pinhole-approximation drift by evaluating the full
    B-spline distortion model once per pixel, ahead of time. After this, every
    detection just bilinear-interpolates the LUT, which is O(1) and exact.

    Steps for each sensor pixel (u, v):
      1. Normalise (u, v) to (0..1) over the sensor dimensions
      2. Evaluate the RectifiedSpline B-spline at that normalised location to
         get the corresponding rectified normalised position (u_r, v_r). This
         is the proper distortion model — undoes the fisheye barrel.
      3. Apply linear pinhole math on the rectified coords using the cal file's
         focal length and principal point to get a ray in camera frame.
      4. Rotate camera frame → Ladybug body frame using R from CamToLadybugEulerZYX.
      5. Convert Ladybug body frame (X=forward, Y=left, Z=up) to our working
         frame (rx=right, ry=down, rz=forward), which is the (-Y, -Z, +X) mapping.
      6. Normalise.

    The result is stored as float32 — 60MB per camera at 2448x2048x3, vs 120MB
    for float64. The precision difference is irrelevant for ray vectors.
    """
    cache_key = (cal_path, lb_id)
    if cache_key in _LB_LUT_CACHE:
        return _LB_LUT_CACHE[cache_key]

    # Try disk cache first — saves ~5–10 seconds per camera on every startup
    disk_path = _lut_disk_path(cal_path, lb_id)
    if os.path.isfile(disk_path):
        try:
            lut = np.load(disk_path)
            if lut.shape == (LADYBUG_SENSOR_W, LADYBUG_SENSOR_H, 3):
                _LB_LUT_CACHE[cache_key] = lut
                return lut
        except Exception:
            pass  # corrupt cache file; fall through and rebuild

    data = _parse_ladybug_cal(cal_path)
    cam  = data['cameras'].get(lb_id)
    if cam is None:
        raise ValueError(
            f"Camera id {lb_id} not present in calibration file {cal_path}"
        )

    # We use RectifiedSpline — the spline that takes DISTORTED normalised
    # coordinates and outputs RECTIFIED normalised coordinates. This is the
    # rectification (anti-distortion) map.
    warp_u = data['warps'][cam['rect_warp_u_id']]
    warp_v = data['warps'][cam['rect_warp_v_id']]

    # scipy expects (tx, ty, c, kx, ky); c is flat in FITPACK row-major order
    tck_u = (
        warp_u['knots_x'], warp_u['knots_y'],
        warp_u['coefs_2d'].flatten(),
        warp_u['kx'], warp_u['ky'],
    )
    tck_v = (
        warp_v['knots_x'], warp_v['knots_y'],
        warp_v['coefs_2d'].flatten(),
        warp_v['kx'], warp_v['ky'],
    )

    # Build the grid of normalised distorted pixel positions
    sensor_w, sensor_h = LADYBUG_SENSOR_W, LADYBUG_SENSOR_H
    u_grid_n = np.arange(sensor_w, dtype=np.float64) / (sensor_w - 1)  # (W,)
    v_grid_n = np.arange(sensor_h, dtype=np.float64) / (sensor_h - 1)  # (H,)

    # Evaluate the rectification spline — bisplev returns shape (len(x), len(y))
    print(f"        evaluating rectification spline for camera {lb_id}...")
    u_rect_grid = scipy.interpolate.bisplev(u_grid_n, v_grid_n, tck_u)  # (W, H)
    v_rect_grid = scipy.interpolate.bisplev(u_grid_n, v_grid_n, tck_v)  # (W, H)

    # Sanity check on the spline output — rectified normalised coords should
    # be roughly in [-0.5..1.5] (some pixels can map slightly outside [0..1]
    # because the rectified FOV is larger than the distorted FOV). If we see
    # values like 1e6 or -1e6, the FITPACK coefficient ordering is wrong.
    if (np.abs(u_rect_grid) > 5).any() or (np.abs(v_rect_grid) > 5).any():
        raise ValueError(
            f"Camera {lb_id}: rectification spline output out of plausible range "
            f"(u in [{u_rect_grid.min():.2f}, {u_rect_grid.max():.2f}], "
            f"v in [{v_rect_grid.min():.2f}, {v_rect_grid.max():.2f}]). "
            "FITPACK coefficient ordering may be wrong, or RectifiedSpline / "
            "DistortedSpline naming is swapped in this cal version."
        )

    # Pinhole: rectified normalised coords → ray in camera frame
    fl_x, fl_y = cam['fl_x'], cam['fl_y']
    cx_n, cy_n = cam['cx_n'], cam['cy_n']
    dx_cam = (u_rect_grid - cx_n) / fl_x   # (W, H)
    dy_cam = (v_rect_grid - cy_n) / fl_y   # (W, H)

    rays_cam = np.stack(
        [dx_cam, dy_cam, np.ones_like(dx_cam)],
        axis=-1,
    )                                         # (W, H, 3)
    norms    = np.linalg.norm(rays_cam, axis=-1, keepdims=True)
    rays_cam = rays_cam / np.maximum(norms, 1e-12)

    # Rotate camera frame → Ladybug body frame.
    # Batched form: for each ray r, want R @ r. Doing this for an (W,H,3) tensor
    # is rays_cam @ R.T (this is the standard "right-multiply by transpose" trick).
    R       = cam['R']
    rays_lb = rays_cam @ R.T                  # (W, H, 3)

    # Ladybug body (X=fwd, Y=left, Z=up)  →  working (rx=right, ry=down, rz=fwd)
    rays_work = np.stack([
        -rays_lb[..., 1],   # rx = -Y_lb
        -rays_lb[..., 2],   # ry = -Z_lb
         rays_lb[..., 0],   # rz = +X_lb
    ], axis=-1)
    # Re-normalise once more — FP error from the multiply can drift slightly.
    norms     = np.linalg.norm(rays_work, axis=-1, keepdims=True)
    rays_work = rays_work / np.maximum(norms, 1e-12)

    # Optional sanity check: the centre pixel of camera 0 (front-facing) should
    # produce a ray that points roughly forward in the working frame. For other
    # cameras this won't be true since they're rotated. We still log it.
    cu, cv     = sensor_w // 2, sensor_h // 2
    center_ray = rays_work[cu, cv]
    print(f"        camera {lb_id} centre-pixel ray: "
          f"({center_ray[0]:+.3f}, {center_ray[1]:+.3f}, {center_ray[2]:+.3f})")

    # Convert to float32 to halve the memory footprint
    rays_work = rays_work.astype(np.float32)

    # Cache to disk for next run
    try:
        np.save(disk_path, rays_work)
    except Exception as e:
        print(f"        [warn] couldn't write LUT cache to {disk_path}: {e}")

    _LB_LUT_CACHE[cache_key] = rays_work
    return rays_work


def get_ray_for_pixel(u: float, v: float, cfg: PipelineConfig, cam_key: str) -> Tuple[float, float, float]:
    """
    Returns a unit ray vector (rx, ry, rz) in the working frame for a pixel
    in the current processing resolution.

    Steps:
      1. Scale (u, v) from (image_width, image_height) → (sensor_w, sensor_h).
      2. Bilinear-interpolate the camera's pre-built LUT at that location.
      3. Re-normalise (averaging unit vectors mildly shortens them).

    No silent fallbacks — if the cal isn't usable, this raises ValueError.
    """
    if not cfg.ladybug_cal_path:
        raise ValueError(
            "get_ray_for_pixel called but no ladybug_cal_path configured in cfg"
        )

    lb_id = cfg.camera_lb_ids.get(cam_key)
    if lb_id is None:
        raise ValueError(
            f"No Ladybug camera ID mapped for camera key '{cam_key}'. "
            f"Configured keys: {list(cfg.camera_lb_ids.keys())}"
        )

    lut       = _build_ladybug_lut(cfg.ladybug_cal_path, lb_id)
    sensor_w  = LADYBUG_SENSOR_W
    sensor_h  = LADYBUG_SENSOR_H
    img_w     = float(cfg.image_width)
    img_h     = float(cfg.image_height)

    # Scale processing-resolution pixel → sensor pixel.
    # NOTE: this assumes a direct linear scale between (1280×1632) and (2448×2048).
    # If the processing pipeline actually rotates the sensor frame by 90° before
    # resizing, the user must add that rotation here. The aspect ratios don't
    # quite match a pure resize, so a 90° rotation followed by a slightly
    # non-uniform resize is the more plausible interpretation — but only the
    # owner of the processing pipeline can say for sure.
    u_sensor = u * (sensor_w - 1) / max(img_w - 1, 1.0)
    v_sensor = v * (sensor_h - 1) / max(img_h - 1, 1.0)

    # Clamp to LUT bounds — boxes can sit on the very edge of the image
    u_sensor = max(0.0, min(float(sensor_w - 1), u_sensor))
    v_sensor = max(0.0, min(float(sensor_h - 1), v_sensor))

    # Bilinear interpolation
    u0 = int(np.floor(u_sensor))
    v0 = int(np.floor(v_sensor))
    u1 = min(u0 + 1, sensor_w - 1)
    v1 = min(v0 + 1, sensor_h - 1)
    fu = u_sensor - u0
    fv = v_sensor - v0

    r00 = lut[u0, v0]
    r01 = lut[u0, v1]
    r10 = lut[u1, v0]
    r11 = lut[u1, v1]

    r_v0 = r00 * (1.0 - fu) + r10 * fu
    r_v1 = r01 * (1.0 - fu) + r11 * fu
    ray  = r_v0 * (1.0 - fv) + r_v1 * fv

    # Re-normalise — averaging unit vectors can mildly shorten the result
    n = float(np.linalg.norm(ray))
    if n < 1e-9:
        raise ValueError(
            f"Interpolated ray vanished at sensor pixel ({u_sensor:.1f}, {v_sensor:.1f}). "
            "This shouldn't be possible — LUT may be corrupted."
        )
    return float(ray[0] / n), float(ray[1] / n), float(ray[2] / n)


def precompute_ladybug_luts(cfg: PipelineConfig) -> None:
    """
    Force-build the LUT for every camera configured in cfg.camera_lb_ids before
    the main loop starts. This pays the spline-evaluation cost once up-front
    instead of as a stall on the first detection per camera.
    """
    if not cfg.ladybug_cal_path:
        return
    t0 = time.time()
    print("[PHASE 1] Loading Ladybug calibration...")
    data = _parse_ladybug_cal(cfg.ladybug_cal_path)
    print(f"        parsed in {time.time()-t0:.2f}s — "
          f"{len(data['cameras'])} cameras, {len(data['warps'])} warp blocks")

    seen: set = set()
    for cam_key, lb_id in cfg.camera_lb_ids.items():
        if lb_id in seen or lb_id not in data['cameras']:
            continue
        seen.add(lb_id)
        t1 = time.time()
        _build_ladybug_lut(cfg.ladybug_cal_path, lb_id)
        print(f"        LUT ready for {cam_key} (lb_id {lb_id}) "
              f"in {time.time()-t1:.2f}s")


def _unproject_pixel(u: float, v: float, cfg: PipelineConfig, cam_key: str = "") -> Tuple[float, float, float]:
    """
    Converts a pixel coordinate to a unit direction vector in the working frame.
    Returns (rx, ry, rz) where rx>0=right, ry>0=down, rz>0=forward.

    If the Ladybug .cal calibration is configured, uses the LUT-backed
    `get_ray_for_pixel` for full B-spline-accurate unprojection (no pinhole
    approximation). Failures here raise ValueError — we deliberately do NOT
    silently fall back to equirectangular, because that produced "looks plausible
    but is wrong" results when the cal was misconfigured.

    The equirectangular branch is only entered when no .cal is configured at all.
    """
    if cfg.ladybug_cal_path:
        if not cam_key:
            raise ValueError(
                "_unproject_pixel: ladybug_cal_path is set but no cam_key supplied. "
                "Every call site must pass cam_key when calibration is active."
            )
        return get_ray_for_pixel(u, v, cfg, cam_key)

    # No calibration configured → equirectangular approximation (legacy behaviour)
    W, H  = float(cfg.image_width), float(cfg.image_height)
    cx    = W / 2.0
    cy    = H / 2.0
    dx    = u - cx
    dy    = v - cy
    r_pix = math.sqrt(dx**2 + dy**2)
    if r_pix < 1e-6:
        return 0.0, 0.0, 1.0
    f     = cx / math.radians(cfg.h_fov / 2.0)
    theta = r_pix / f
    sin_t = math.sin(theta)
    return sin_t*(dx/r_pix), sin_t*(dy/r_pix), math.cos(theta)


def _raycast_cylinder(
    origin:     np.ndarray,
    direction:  np.ndarray,
    kdtree:     KDTree,
    points:     np.ndarray,
    min_dist:   float = 2.0,
    max_dist:   float = 30.0,
    cyl_radius: float = 1.50,   # 1.5m works well for MX2 point density (~0.8m wall spacing)
    min_strike: int   = 2,      # need at least 2 points to call it a real hit
    step_m:     float = 0.40,
) -> Tuple[Optional[np.ndarray], Optional[float]]:
    """
    Casts a ray and checks if it hits anything in the point cloud.
    Returns (centroid_xyz, distance_m) of the first surface it hits, or (None, None).

    The approach: sample points along the ray at regular intervals, collect all
    cloud points that fall within cyl_radius of any sample, then filter those
    down to ones that are actually inside the cylinder (not just near a sample point).
    Take the closest cluster that has at least min_strike points in it.

    On the radius choice: we measured the MX2 LAS files for this project and the
    average nearest-neighbour spacing on walls is about 0.8m. A 0.4m radius cylinder
    would miss most of the time. 1.5m catches things reliably while still being
    narrow enough to give a decent centroid position.
    """
    steps   = np.arange(min_dist, max_dist, step_m)
    ray_pts = origin + np.outer(steps, direction)

    # collect candidate point indices from all sample positions along the ray
    cand_set: set = set()
    for pt in ray_pts:
        cand_set.update(kdtree.query_ball_point(pt, r=cyl_radius))

    if not cand_set:
        return None, None

    # now filter to points actually inside the cylinder, not just near a sample
    cands  = points[list(cand_set)]
    vecs   = cands - origin
    t_vals = vecs @ direction                          # distance along ray axis
    proj   = origin + np.outer(t_vals, direction)     # closest point on ray to each candidate
    perp   = np.linalg.norm(cands - proj, axis=1)     # perpendicular distance to ray axis

    mask   = (perp < cyl_radius) & (t_vals >= min_dist) & (t_vals <= max_dist)
    if not mask.any():
        return None, None

    valid   = cands[mask]
    valid_t = t_vals[mask]
    t_min   = valid_t.min()

    # grab everything within 0.5m of the first hit — that's our surface cluster
    near = np.abs(valid_t - t_min) < 0.5
    if near.sum() < min_strike:
        return None, None

    return valid[near].mean(axis=0), float(t_min)


# ---
# Geolocation
# ---

def calculate_gps_offset_3d(
    car_x:              float,
    car_y:              float,
    car_z:              float,   # GPS altitude, WGS84 ellipsoidal
    car_heading:        float,   # degrees, compass (0=north, 90=east)
    bbox_center_x:      float,
    bbox_bottom_y:      float,   # we use the bottom of the YOLO box as the reference point
    camera_mount_angle: float,
    kdtree:             Optional[KDTree],
    points:             Optional[np.ndarray],
    cfg:                PipelineConfig,
    geoid_undulation:   float,
    cam_key:            str = "",
) -> dict:
    """
    Given a YOLO detection in an image, figures out where that firida actually
    is on the ground in WGS84 lat/lon.

    Steps:
      1. Convert the pixel position to a direction vector in camera space
      2. Rotate that into world space using the car heading + camera mount angle
      3. Correct the Z so it's in the same datum as the LAS file
      4. Cast the ray into the point cloud and find the first thing it hits
      5. If the cloud misses, fall back to flat-ground geometry
      6. Convert the Stereo70 result to WGS84

    Returns a dict with lat, lon, and some quality flags.
    """
    rx, ry, rz = _unproject_pixel(bbox_center_x, bbox_bottom_y, cfg, cam_key)

    px_edge_flag = abs(bbox_center_x - cfg.image_width / 2.0) > cfg.image_width * 0.35

    # if ry is zero or negative the ray is pointing up or sideways, which means
    # the flat-ground fallback would give a nonsensical result (negative time).
    # just skip it.
    if ry <= 0:
        return {"x": None, "y": None, "z": None, "lidar_hit": False,
                "px_edge_flag": px_edge_flag, "range_m": None,
                "true_heading_deg": None}

    # figure out which direction in world space this pixel is pointing
    alpha_deg        = math.degrees(math.atan2(rx, rz))
    true_heading_deg = (car_heading + camera_mount_angle + alpha_deg) % 360
    brng             = math.radians(true_heading_deg)

    vert_angle = math.atan2(-ry, math.sqrt(rx**2 + rz**2))
    cos_v      = math.cos(vert_angle)
    direction  = np.array([math.sin(brng)*cos_v,
                            math.cos(brng)*cos_v,
                            math.sin(vert_angle)], dtype=np.float64)
    direction /= np.linalg.norm(direction)

    # the key correction: GPS gives ellipsoidal altitude, LAS uses orthometric.
    # subtract the geoid undulation to get them into the same reference system.
    origin_z = car_z - geoid_undulation
    origin   = np.array([car_x, car_y, origin_z], dtype=np.float64)

    # debug block — flip this to True if you want to see exactly what the ray
    # is doing and whether it's landing in the right Z range
    DEBUG_RAYCAST = False
    if DEBUG_RAYCAST and kdtree is not None and points is not None:
        print("\n[DEBUG] -------------------------------------------------")
        print(f"  GPS Z (ellipsoidal):  {car_z:.3f}m")
        print(f"  Geoid undulation:     {geoid_undulation:.3f}m")
        print(f"  Ray origin Z (LAS):   {origin_z:.3f}m")
        print(f"  LAS Z range:          [{points[:,2].min():.2f}..{points[:,2].max():.2f}]")
        print(f"  Bearing:              {true_heading_deg:.1f}°")
        for d in [5, 10, 15, 20]:
            pt  = origin + direction * d
            n   = len(kdtree.query_ball_point(pt, r=1.5))
            print(f"  @{d:2d}m  Z={pt[2]:.2f}  nearby_pts={n}")
        x_ok = 100_000 < float(np.median(points[:,0])) < 900_000
        print(f"  X coordinates look valid: {x_ok}")
        print("[DEBUG] -------------------------------------------------\n")

    # try to hit the point cloud first
    centroid_xyz: Optional[np.ndarray] = None
    range_m:      Optional[float]      = None
    lidar_hit    = False

    if kdtree is not None and points is not None:
        centroid_xyz, range_m = _raycast_cylinder(origin, direction, kdtree, points)
        if centroid_xyz is not None:
            lidar_hit = True

    # if the cloud didn't give us anything, estimate using flat-ground geometry.
    # this assumes the firida is on flat terrain at camera height — not great but
    # better than nothing, and it's flagged as [PLANAR] in the output.
    if centroid_xyz is None:
        t        = cfg.camera_height / ry
        dist_gnd = min(math.sqrt((t*rx)**2 + (t*rz)**2), 100.0)
        centroid_xyz = np.array([
            car_x + math.sin(brng) * dist_gnd,
            car_y + math.cos(brng) * dist_gnd,
            origin_z,
        ], dtype=np.float64)

    return {"x": float(centroid_xyz[0]), "y": float(centroid_xyz[1]), "z": float(centroid_xyz[2]), "lidar_hit": lidar_hit,
            "px_edge_flag": px_edge_flag, "range_m": range_m,
            "true_heading_deg": true_heading_deg}


# ---
# Helpers
# ---

def euclidean_distance(x1, y1, x2, y2) -> float:
    """Straight-line 2D distance in metres between two points."""
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def _check_heading_convention(samples: list, cfg: PipelineConfig) -> None:
    """
    Sanity check on the heading values. If the mean offset between the resolved
    ray bearing and the car heading is around 90°, that usually means the CSV
    is using math angles (east=0) instead of compass angles (north=0).
    Logs a warning rather than crashing — easy to miss otherwise.
    """
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
            f"[HEADING] Average ray-vs-GPS bearing difference is {mean_abs:.1f}°. "
            "If positions look rotated 90°, check that Heading_deg in the CSV "
            "uses compass convention (north=0°, east=90°).",
            stacklevel=2,
        )


# ---
# Main pipeline
# ---

def run_enterprise_pipeline(cfg: PipelineConfig) -> None:
    print("[PHASE 1] Starting up...")
    print(f"          Source : {cfg.parent_folder}")
    print(f"          Output : {cfg.output_folder}")
    print(f"          LiDAR  : {cfg.las_folder or '(none — will use planar fallback)'}")

    device   = "cuda" if torch.cuda.is_available() else "cpu"
    use_half = device == "cuda"
    if device == "cuda":
        torch.backends.cudnn.benchmark     = True
        torch.backends.cudnn.deterministic = False
        print("          GPU    : CUDA available, using half precision")

    model = YOLO(cfg.model_path)
    os.makedirs(cfg.output_folder, exist_ok=True)
    start_time = time.time()

    # build the Ladybug LUTs up-front so the first detection per camera isn't
    # blocked by a multi-second spline-evaluation stall.
    precompute_ladybug_luts(cfg)

    # load both LAS files up front so we don't have to reload them per camera.
    # Camera1/2 are left-facing, Camera3/4 are right-facing.
    left_tree  = left_pts  = None
    right_tree = right_pts = None
    if cfg.las_folder:
        left_tree,  left_pts  = load_lidar_kdtree(cfg.las_folder, "left")
        right_tree, right_pts = load_lidar_kdtree(cfg.las_folder, "right")

    # figure out the Z correction needed for this recording.
    # if the user passed --geoid_undulation we use that directly,
    # otherwise we calculate it from the data.
    geoid_undulation = cfg.geoid_undulation
    if geoid_undulation is None:
        calib_pts = left_pts if left_pts is not None else right_pts
        if calib_pts is not None:
            print("[PHASE 1] Calculating Z datum offset from LAS vs GPS...")
            calib_df: Optional[pd.DataFrame] = None

            # just grab the first coordonate file we can find for calibration
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
                print(f"  [!] Couldn't find telemetry for calibration, using {geoid_undulation}m default")
        else:
            geoid_undulation = 0.0   # no LAS loaded, correction doesn't matter

    print(f"          Z correction: {geoid_undulation:.3f}m")

    # process left cameras first so the sort is deterministic and matches our preloaded trees
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
                cam_key   = k
                mount_ang = a
                break
        if cam_key is None or mount_ang is None:
            continue

        print(f"\n---> {cam_key} (mounted at {mount_ang}° from nose)")

        # cameras with mount angle >= 180° face left, the rest face right
        is_left = mount_ang >= 180.0
        kdtree  = left_tree  if is_left else right_tree
        points  = left_pts   if is_left else right_pts

        # find the coordonate file for this camera folder
        coord_file = None
        for fn in os.listdir(folder_path):
            fl = fn.lower()
            if "coordonate" in fl and not fl.startswith("~$"):
                coord_file = os.path.join(folder_path, fn)
                break
        if not coord_file:
            print(f"  [!] No coordonate file found, skipping {cam_key}.")
            continue

        df = (pd.read_csv(coord_file) if coord_file.endswith(".csv")
              else pd.read_excel(coord_file))

        required = ["X_Stereo70", "Y_Stereo70", "Z", "Heading_deg", "Imagine"]
        missing  = [c for c in required if c not in df.columns]
        if missing:
            print(f"  [!] Missing columns {missing}, skipping {cam_key}.")
            continue

        # lowercase everything once here so we don't have to worry about
        # case mismatches when looking up image names later
        df["Imagine"] = df["Imagine"].astype(str).str.strip().str.lower()
        lookup = {row["Imagine"]: row for _, row in df.iterrows()}

        images = [f for f in os.listdir(folder_path)
                  if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        print(f"  [*] {len(images)} images to process")

        results = model.predict(
            source=folder_path, conf=cfg.confidence, iou=cfg.iou_threshold,
            imgsz=cfg.image_width, augment=cfg.use_tta, half=use_half,
            device=device, stream=True, verbose=False, batch=cfg.batch_size,
            workers=0,
        )

        n_det = 0
        for r in tqdm(results, total=len(images), desc=f"Scanning {cam_key}", unit="img"):
            if len(r.boxes) == 0:
                continue

            img_name  = os.path.basename(r.path).strip().lower()
            telemetry = lookup.get(img_name)
            if telemetry is None:
                continue   # image not in the coordonate file, skip it

            car_x = float(telemetry["X_Stereo70"])
            car_y = float(telemetry["Y_Stereo70"])
            car_z = float(telemetry["Z"])
            car_h = float(telemetry["Heading_deg"])

            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf    = float(box.conf[0])
                bbox_cx = (x1 + x2) / 2.0

                effective_mount = 0.0 if cfg.ladybug_cal_path else mount_ang
                geo = calculate_gps_offset_3d(
                    car_x, car_y, car_z, car_h,
                    bbox_cx, y2, effective_mount,
                    kdtree, points, cfg, geoid_undulation,
                    cam_key=cam_key,
                )

                if geo["x"] is None:
                    # ray was pointing upward, nothing we can do with this one
                    continue

                det = {
                    "image":        img_name,
                    "cam_key":      cam_key,
                    "folder_path":  folder_path,
                    "x1": int(x1), "y1": int(y1),
                    "x2": int(x2), "y2": int(y2),
                    "conf":         conf,
                    "x":            geo["x"],
                    "y":            geo["y"],
                    "z":            geo["z"],
                    "lidar_hit":    geo["lidar_hit"],     # True = sub-metre accuracy
                    "px_edge_flag": geo["px_edge_flag"],  # True = near frame edge, less accurate
                    "range_m":      geo["range_m"],       # how far the ray travelled to hit something
                    "_car_heading_deg":  car_h,           # kept for the heading sanity check, not exported
                    "true_heading_deg":  geo["true_heading_deg"],
                }
                all_detections.append(det)
                n_det += 1

                if len(heading_check_sample) < 5:
                    heading_check_sample.append(det)

        print(f"  [✓] {cam_key} done — {n_det} detections")

    _check_heading_convention(heading_check_sample, cfg)
    print(f"\nAll cameras done — {len(all_detections)} total raw detections.")

    # ---
    # Phase 3: merge detections that are pointing at the same firida
    # ---
    print("[PHASE 3] Deduplicating and clustering nearby detections...")
    unique_firidas: list[dict] = []

    for det in all_detections:
        matched = False
        for uf in unique_firidas:
            dist = euclidean_distance(det["x"], det["y"], uf["x"], uf["y"])
            if dist <= cfg.cluster_radius_m:
                # same image, same camera — definitely a duplicate, just skip it
                if det["image"] == uf["image"] and det["cam_key"] == uf["cam_key"]:
                    matched = True
                    break
                # close enough to count as the same firida
                matched = True
                uf["seen_count"] = uf.get("seen_count", 1) + 1
                uf["clustered"]  = True
                if "cluster_members" not in uf:
                    uf["cluster_members"] = [dict(uf)]
                uf["cluster_members"].append(dict(det))
                # keep whichever detection has the higher confidence score
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

    # Cross-camera merge pass.
    #
    # The first pass above uses a tight 2m radius which works great for the same
    # camera seeing the same firida multiple times as the car drives past. But when
    # two different cameras both spot the same firida, their ray angles are different
    # and the calculated coordinates can land 3-8m apart even for the same object.
    # The 2m threshold misses those matches.
    #
    # This pass uses a wider radius and, crucially, actually merges the cluster_members
    # lists so the gallery ends up with all the photos together in one group. The old
    # version only set clustered=True without combining anything, which is why you'd
    # see two separate yellow entries for the same firida.
    #
    # We loop until nothing changes because merging two groups can create new pairs.
    did_merge = True
    while did_merge:
        did_merge = False
        i = 0
        while i < len(unique_firidas):
            j = i + 1
            while j < len(unique_firidas):
                fi = unique_firidas[i]
                fj = unique_firidas[j]
                dist = euclidean_distance(fi["x"], fi["y"], fj["x"], fj["y"])
                if dist <= cfg.cross_camera_radius_m:
                    # merge fj into fi — combine their photo lists
                    if "cluster_members" not in fi:
                        fi["cluster_members"] = [dict(fi)]
                    fi["cluster_members"].extend(fj.get("cluster_members", [dict(fj)]))
                    fi["seen_count"] = fi.get("seen_count", 1) + fj.get("seen_count", 1)
                    fi["clustered"]  = True

                    # if fj had the better confidence, promote it as the representative
                    # but keep the merged member list we just built
                    if fj["conf"] > fi["conf"]:
                        best = dict(fj)
                        best["cluster_members"] = fi["cluster_members"]
                        best["seen_count"]      = fi["seen_count"]
                        best["clustered"]       = True
                        unique_firidas[i]       = best
                        fi = unique_firidas[i]

                    unique_firidas.pop(j)
                    did_merge = True
                    # don't increment j — the list just got shorter
                else:
                    j += 1
            i += 1

    hit_count = sum(1 for d in unique_firidas if d.get("lidar_hit"))
    total     = len(unique_firidas)
    print(f"Done — {total} unique firidas found.")
    print(f"LiDAR positioned: {hit_count}/{total} ({100*hit_count//max(total,1)}%)")

    # save annotated images so you can review what got detected
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
        if not f.get("lidar_hit"):
            label += " [PLANAR]"
        if f.get("px_edge_flag"):
            label += " [EDGE]"

        for m in f.get("cluster_members", [f]):
            img = cv2.imread(os.path.join(m["folder_path"], m["image"]))
            if img is not None:
                cv2.rectangle(img, (m["x1"], m["y1"]), (m["x2"], m["y2"]), color, 4)
                cv2.putText(img, f"{m['cam_key']} - {label}",
                            (m["x1"], m["y1"]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                cv2.imwrite(
                    os.path.join(cfg.output_folder,
                                 f"{m['cam_key']}_{m['image']}"), img)

    # ---
    # Phase 4: export GeoJSON — Hardware-Certified GeoJSON (EPSG:3844)
    # Schema: nr_imobil + tip_firida_bransament only (no raw detection metadata)
    # ---
    print("[PHASE 4] Exporting results...")
    flat: list[dict] = []
    for f in unique_firidas:
        for m in f.get("cluster_members", [f]):
            mc = {k: v for k, v in m.items() if k != "cluster_members"}
            mc["clustered"] = f.get("clustered", False)
            flat.append(mc)

    if flat:
        df_exp = pd.DataFrame(flat)

        # Round coordinates to 4 decimal places
        for col in ("x", "y", "z"):
            if col in df_exp.columns:
                df_exp[col] = df_exp[col].round(4)

        # Ensure schema columns exist
        if "nr_imobil" not in df_exp.columns:
            df_exp["nr_imobil"] = "FN"
        if "tip_firida_bransament" not in df_exp.columns:
            df_exp["tip_firida_bransament"] = ""

        geometry = gpd.points_from_xy(df_exp["x"], df_exp["y"], z=df_exp["z"])
        gdf = gpd.GeoDataFrame(
            df_exp[["nr_imobil", "tip_firida_bransament"]],
            geometry=geometry,
            crs="EPSG:3844",
        )

        geo = os.path.join(cfg.output_folder, "tip_firida_bransament.geojson")
        jsn = os.path.join(cfg.output_folder, "tip_firida_bransament.json")
        gdf.to_file(pathlib.Path(geo), driver="GeoJSON")
        df_exp.to_json(jsn, orient="records", indent=2)
        print(f"  GeoJSON   : {geo}")
        print(f"  JSON      : {jsn}")
    else:
        print("  No firidas to export.")

    print(f"\nFinished in {round(time.time()-start_time, 2)}s.")


# ---
# Entry point
# ---

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Gauss firida detector")
    p.add_argument("--folder",           required=True,  help="Recording directory (contains Camera1-4 subfolders)")
    p.add_argument("--output",           required=True,  help="Where to save results")
    p.add_argument("--las_folder",       default="",     help="Folder with the left/right .las files")
    p.add_argument("--conf",             type=float, default=0.75, help="YOLO confidence threshold")
    p.add_argument("--cluster",              type=float, default=2.00, help="Same-camera dedup radius in metres")
    p.add_argument("--cross_camera_radius",  type=float, default=8.00,
                   help="Cross-camera merge radius in metres. Wider than --cluster because "
                        "rays from different angles land a few metres apart for the same firida.")
    p.add_argument("--batch",            type=int,   default=24,   help="YOLO batch size")
    p.add_argument("--geoid_undulation", type=float, default=None,
                   help="Z offset in metres between GPS altitude and LAS orthometric height. "
                        "Leave blank to calculate automatically. For this area of Romania it's ~39m.")
    p.add_argument("--cal",              type=str,   default=None,
                   help='Path to the Ladybug .cal hardware calibration file (e.g. ladybug15295546.cal)')
    args = p.parse_args()

    if not os.path.exists(args.folder):
        print(f"Error: folder not found: {args.folder}")
    else:
        cfg  = PipelineConfig(
            parent_folder         = args.folder,
            output_folder         = args.output,
            las_folder            = args.las_folder,
            confidence            = args.conf,
            cluster_radius_m      = args.cluster,
            cross_camera_radius_m = args.cross_camera_radius,
            batch_size            = args.batch,
            geoid_undulation      = args.geoid_undulation,
            ladybug_cal_path      = args.cal,
        )
        run_enterprise_pipeline(cfg)