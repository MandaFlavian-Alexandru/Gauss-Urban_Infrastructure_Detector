from ultralytics import YOLO
import torch
import time
import os
import math
import cv2
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from tqdm import tqdm  
import numpy as np
import laspy
from scipy.spatial import KDTree
import pyproj

# --- Configuration ---
MODEL_PATH = "firida_detector_v4_verygood.pt"            
PARENT_FOLDER = ""    
LAS_FOLDER = ""
OUTPUT_FOLDER = "Results_Master_Fusion_Map" 

# --- Inference Settings ---
CONFIDENCE = 0.75         
IOU_THRESHOLD = 0.45      
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 1632       
USE_TTA = True
BATCH_SIZE = 24

# --- Hardware Orientation ---
CAMERA_HEIGHT = 2.45      
H_FOV = 60.0              
V_FOV = H_FOV * (IMAGE_HEIGHT / IMAGE_WIDTH)  

# Camera mount offsets relative to vehicle orientation
CAMERA_ANGLES = {
    "Camera3": 60.0,    
    "Camera4": 120.0,   
    "Camera2": 300.0,   
    "Camera1": 240.0    
}

# --- Spatial Parameters ---
DEDUPE_RADIUS_M = 5.00
CLUSTER_RADIUS_M = 2.00

# --- Bounding Box Colors ---
COLOR_GREEN = (0, 255, 0)
COLOR_ORANGE = (0, 165, 255)
COLOR_RED = (0, 0, 255)
COLOR_YELLOW = (0, 255, 255)

# --- Coordinate Transformations ---
# EPSG:3844 (Stereo 70 Romania) to EPSG:4326 (WGS84)
transformer_to_wgs84 = pyproj.Transformer.from_crs("EPSG:3844", "EPSG:4326", always_xy=True)

def load_lidar_kdtree(las_folder: str, is_left: bool):
    """Loads matching Left or Right .las file and returns a KDTree and points array"""
    if not os.path.isdir(las_folder):
        print(f"  [!] LAS Folder not found: {las_folder}")
        return None, None
        
    keyword = "left" if is_left else "right"
    las_files = [f for f in os.listdir(las_folder) if f.lower().endswith('.las')]
    
    target_las = None
    for lf in las_files:
        if keyword in lf.lower():
            target_las = os.path.join(las_folder, lf)
            break
            
    if not target_las:
        print(f"  [!] No matching LAS file found for {keyword.upper()} facing camera in {las_folder}")
        return None, None
        
    print(f"  [*] Loading Point Cloud for intersection: {target_las}")
    try:
        las = laspy.read(target_las)
        points = np.vstack((las.x, las.y, las.z)).transpose()
        # Subsample to keep KDTree reasonable memory (e.g. 1 in 10 points) or use as is if small
        # But we'll try to use as is. If memory is an issue, uncomment below:
        # if len(points) > 5000000: points = points[::5] 
        tree = KDTree(points)
        print(f"  [*] KDTree built with {len(points)} points.")
        return tree, points
    except Exception as e:
        print(f"  [!] Error loading LiDAR file: {e}")
        return None, None

def calculate_gps_offset_3d(origin_x, origin_y, origin_z, car_heading, bbox_center_x, bbox_bottom_y, camera_mount_angle, kdtree, points):
    """Cast a 3D ray into the LiDAR KDTree to find object physical location in Stereo70, then project to WGS84."""
    
    center_x = IMAGE_WIDTH / 2.0
    center_y = IMAGE_HEIGHT / 2.0
    
    dx = bbox_center_x - center_x
    dy = bbox_bottom_y - center_y
    if dy <= 0: dy = 0.001
        
    r = math.sqrt(dx**2 + dy**2)
    if r == 0: r = 0.001

    f_equi = center_x / math.radians(H_FOV / 2.0)
    theta = r / f_equi
    
    sin_theta = math.sin(theta)
    ray_x = sin_theta * (dx / r)
    ray_y = sin_theta * (dy / r)  # Positive is down visually
    ray_z = math.cos(theta)       # Positive is forward visually
    
    # Horizontal deviation
    alpha_deg = math.degrees(math.atan2(ray_x, ray_z))
    true_heading = (car_heading + camera_mount_angle + alpha_deg) % 360
    brng = math.radians(true_heading)
    
    # Vertical tilt of the ray
    vertical_angle = math.atan2(-ray_y, math.sqrt(ray_x**2 + ray_z**2))
    
    # World normalized direction vector
    dir_e = math.sin(brng) * math.cos(vertical_angle)
    dir_n = math.cos(brng) * math.cos(vertical_angle)
    dir_z = math.sin(vertical_angle)
    
    centroid_x, centroid_y = None, None
    
    # Raycast Search Sequence
    if kdtree is not None and points is not None:
        # Finer ray step resolution for exact surface impact
        for d in np.arange(2.0, 25.0, 0.5):
            pt = [origin_x + dir_e * d, origin_y + dir_n * d, origin_z + dir_z * d]
            # Narrower search radius prevents grabbing foreground tree leaves
            idxs = kdtree.query_ball_point(pt, r=0.6)
            
            # Strike detection: Require at least 5 points to consider it a solid object/wall
            if len(idxs) >= 5:
                cluster = points[idxs]
                # We calculate the centroid ONLY on this specific concentrated strike plate
                centroid_x = float(np.mean(cluster[:, 0]))
                centroid_y = float(np.mean(cluster[:, 1]))
                break
                
    # Fallback to 2D planar math if LiDAR isn't available or ray hit empty air
    if centroid_x is None:
        t = CAMERA_HEIGHT / ray_y
        x_gnd = t * ray_x
        z_gnd = t * ray_z
        dist_gnd = math.sqrt(x_gnd**2 + z_gnd**2)
        if dist_gnd > 100.0: dist_gnd = 100.0
        
        centroid_x = origin_x + math.sin(brng) * dist_gnd
        centroid_y = origin_y + math.cos(brng) * dist_gnd

    # Transform Stereo70 Easting,Northing to WGS84 Lon,Lat
    lon, lat = transformer_to_wgs84.transform(centroid_x, centroid_y)
    
    return lat, lon

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculates distance in meters between two WGS84 points."""
    R = 6378137.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

def run_enterprise_pipeline():
    print("[PHASE 1] Initializing 3D Point Cloud Spatial Pipeline...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print("Hardware Configuration: PyTorch CuDNN Autotuner Active")
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
    use_half = True if device == 'cuda' else False
    
    model = YOLO(MODEL_PATH)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    all_detections = []
    
    start_time = time.time()
    
    loaded_kdtree = None
    loaded_points = None
    current_lidar_side = None

    for folder_name in os.listdir(PARENT_FOLDER):
        folder_path = os.path.join(PARENT_FOLDER, folder_name)
        if not os.path.isdir(folder_path):
            continue
            
        current_cam_key = None
        current_mount_angle = None
        for cam_key, angle in CAMERA_ANGLES.items():
            if cam_key in folder_name:
                current_cam_key = cam_key
                current_mount_angle = angle
                break
                
        if not current_cam_key:
            continue
            
        print(f"\n---> Initializing {current_cam_key} (Angle: {current_mount_angle}°)")
        
        # Determine Left or Right face
        is_left_face = current_cam_key in ["Camera1", "Camera2"]
        required_side = "left" if is_left_face else "right"
        
        # Only reload KDTree if crossing from left cameras to right cameras
        if current_lidar_side != required_side:
            loaded_kdtree, loaded_points = load_lidar_kdtree(LAS_FOLDER, is_left=is_left_face)
            current_lidar_side = required_side
            
        coord_file_path = None
        for file in os.listdir(folder_path):
            if "coordonate" in file.lower() and not file.lower().startswith('~$'):
                coord_file_path = os.path.join(folder_path, file)
                break
                
        if not coord_file_path:
            print(f"  [!] Skipping {current_cam_key}: No 'coordonate' file found.")
            continue
            
        if coord_file_path.endswith('.csv'):
            df_coords = pd.read_csv(coord_file_path)
        else:
            df_coords = pd.read_excel(coord_file_path)

        # Ensure required Stereo70 3D spatial columns exist
        required_cols = ['X_Stereo70', 'Y_Stereo70', 'Z', 'Heading_deg', 'Imagine']
        missing = [c for c in required_cols if c not in df_coords.columns]
        if missing:
             print(f"  [!] Missing critical 3D coordinate columns in tracking file: {missing}")
             print("      Aborting processing for this camera.")
             continue

        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        total_images = len(image_files)
                
        print(f"  [*] Initialized Camera - Iterating {total_images} images")

        results = model.predict(
            source=folder_path, conf=CONFIDENCE, iou=IOU_THRESHOLD, 
            imgsz=IMAGE_WIDTH, augment=USE_TTA, half=use_half, device=device, 
            stream=True, verbose=False, batch=BATCH_SIZE
        )

        folder_detections = 0
        
        for r in tqdm(results, total=total_images, desc=f"Scanning {current_cam_key}", unit="img"):
            if len(r.boxes) == 0:
                continue
                
            img_name = os.path.basename(r.path)
            car_data = df_coords[df_coords['Imagine'] == img_name]
            if car_data.empty:
                continue
                
            car_x = car_data.iloc[0]['X_Stereo70']
            car_y = car_data.iloc[0]['Y_Stereo70']
            car_z = car_data.iloc[0]['Z']
            car_heading = car_data.iloc[0]['Heading_deg']

            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                
                center_x = (x1 + x2) / 2.0
                firida_lat, firida_lon = calculate_gps_offset_3d(
                    car_x, car_y, car_z, car_heading, center_x, y2, current_mount_angle, loaded_kdtree, loaded_points
                )
                
                all_detections.append({
                    'image': img_name, 'cam_key': current_cam_key, 'folder_path': folder_path,
                    'x1': int(x1), 'y1': int(y1), 'x2': int(x2), 'y2': int(y2),
                    'conf': conf, 'lat': firida_lat, 'lon': firida_lon
                })
                folder_detections += 1
                
        print(f"  [✓] {current_cam_key} Complete. Found {folder_detections} raw detections.")

    print(f"Extraction complete. {len(all_detections)} total raw detections recorded.")
    print("[PHASE 3] Running spatial deduplication and proximity clustering...")
    
    unique_firidas = []
    for det in all_detections:
        matched = False
        for uf in unique_firidas:
            dist = haversine_distance(det['lat'], det['lon'], uf['lat'], uf['lon'])
            
            if dist <= DEDUPE_RADIUS_M:
                if det['image'] == uf['image'] and det['cam_key'] == uf['cam_key']:
                    continue 
                matched = True
                if det['conf'] > uf['conf']:
                    uf.update(det) 
                break
        
        if not matched:
            det['clustered'] = False
            unique_firidas.append(det)

    for i, f1 in enumerate(unique_firidas):
        for j, f2 in enumerate(unique_firidas):
            if i != j:
                dist = haversine_distance(f1['lat'], f1['lon'], f2['lat'], f2['lon'])
                if dist <= CLUSTER_RADIUS_M:
                    f1['clustered'] = True
                    f2['clustered'] = True

    print(f"Spatial filtering complete. Final dataset contains {len(unique_firidas)} unique instances.")

    for f in unique_firidas:
        if f['clustered']:
            color = COLOR_YELLOW
            label = f"CLUSTERED: {f['conf']:.2f}"
        elif f['conf'] >= 0.85:
            color = COLOR_GREEN
            label = f"Firida: {f['conf']:.2f}"
        elif f['conf'] >= 0.80:
            color = COLOR_ORANGE
            label = f"Firida: {f['conf']:.2f}"
        else:
            color = COLOR_RED
            label = f"WARNING: {f['conf']:.2f}"

        img_path = os.path.join(f['folder_path'], f['image'])
        img = cv2.imread(img_path)
        if img is not None:
            cv2.rectangle(img, (f['x1'], f['y1']), (f['x2'], f['y2']), color, 4)
            cv2.putText(img, label, (f['x1'], f['y1'] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            save_filename = f"{f['cam_key']}_{f['image']}"
            save_path = os.path.join(OUTPUT_FOLDER, save_filename)
            cv2.imwrite(save_path, img)

    print("[PHASE 4] Compiling QGIS deliverables...")
    if len(unique_firidas) > 0:
        df_export = pd.DataFrame(unique_firidas)
        df_export = df_export.drop(columns=['x1', 'y1', 'x2', 'y2', 'folder_path'])
        
        geometry = [Point(xy) for xy in zip(df_export['lon'], df_export['lat'])]
        gdf = gpd.GeoDataFrame(df_export, geometry=geometry)
        gdf.set_crs(epsg=4326, inplace=True)
        
        shp_path = os.path.join(OUTPUT_FOLDER, "tip_firida_bransament.shp")
        gdf.to_file(shp_path)
        
        json_path = os.path.join(OUTPUT_FOLDER, "tip_firida_bransament.json")
        df_export.to_json(json_path, orient="records")
        print(f"Shapefile successfully generated: {shp_path}")
    else:
        print("No Firidas to export to map.")

    end_time = time.time()
    print(f"Pipeline executed successfully in {round(end_time - start_time, 2)} seconds.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, default=PARENT_FOLDER)
    parser.add_argument("--las_folder", type=str, default="")
    parser.add_argument("--conf", type=float, default=CONFIDENCE)
    parser.add_argument("--cluster", type=float, default=CLUSTER_RADIUS_M)
    parser.add_argument("--batch", type=int, default=BATCH_SIZE)
    parser.add_argument("--output", type=str, required=True) 
    args = parser.parse_args()

    PARENT_FOLDER = args.folder
    LAS_FOLDER = args.las_folder
    CONFIDENCE = args.conf
    CLUSTER_RADIUS_M = args.cluster
    BATCH_SIZE = args.batch
    OUTPUT_FOLDER = args.output 

    if not os.path.exists(PARENT_FOLDER):
        print(f"ERROR: Cannot reach image folder {PARENT_FOLDER}.")
    else:
        run_enterprise_pipeline()