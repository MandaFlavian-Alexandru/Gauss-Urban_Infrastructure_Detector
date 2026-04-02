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

# --- Configuration ---
MODEL_PATH = "firida_detector_v4_verygood.pt"            
PARENT_FOLDER = ""    
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

def calculate_gps_offset(lat, lon, car_heading, bbox_center_x, bbox_bottom_y, camera_mount_angle):
    """Calculates the WGS84 coordinate using a 2D equidistant fisheye ray-casting projection."""
    center_x = IMAGE_WIDTH / 2.0
    center_y = IMAGE_HEIGHT / 2.0
    
    # Pixel offsets from optical center
    dx = bbox_center_x - center_x
    dy = bbox_bottom_y - center_y
    
    # Mathematical stabilization at the exact dead center
    if dy <= 0:
        dy = 0.001
        
    r = math.sqrt(dx**2 + dy**2)
    if r == 0:
        r = 0.001

    # Equidistant focal length estimation
    # H_FOV is 60, meaning the very edge of the sensor (1280/2 = 640px) is an optical 30 degree angle.
    # Therefore under the r = f * theta model, f = dx / theta
    f_equi = center_x / math.radians(H_FOV / 2.0)
    
    # Ray angle in radians departing the optical center
    theta = r / f_equi
    
    # Constructing a 3D ray in camera coordinate system (Z forward, X right, Y down)
    sin_theta = math.sin(theta)
    ray_x = sin_theta * (dx / r)
    ray_y = sin_theta * (dy / r)
    ray_z = math.cos(theta)
    
    # Intersect the ray with the physical ground plane (Y = CAMERA_HEIGHT meters)
    t = CAMERA_HEIGHT / ray_y
    
    # The physical horizontal vector on the ground surface
    x_gnd = t * ray_x
    z_gnd = t * ray_z
    
    dist_gnd = math.sqrt(x_gnd**2 + z_gnd**2)
    # Safety cap: limit maximum detection distance to 100 meters to avoid infinite ground projections near the horizon
    if dist_gnd > 100.0:
        dist_gnd = 100.0
        
    # Angle deviation from camera's immediate forward direction
    alpha_deg = math.degrees(math.atan2(x_gnd, z_gnd))
    
    R = 6378137.0
    # Final geographic bearing factoring in car heading + camera mount twist + object's deviation angle
    true_heading = (car_heading + camera_mount_angle + alpha_deg) % 360
    brng = math.radians(true_heading)
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    
    new_lat_rad = math.asin(math.sin(lat_rad) * math.cos(dist_gnd / R) +
                            math.cos(lat_rad) * math.sin(dist_gnd / R) * math.cos(brng))
    new_lon_rad = lon_rad + math.atan2(math.sin(brng) * math.sin(dist_gnd / R) * math.cos(lat_rad),
                                       math.cos(dist_gnd / R) - math.sin(lat_rad) * math.sin(new_lat_rad))
    
    return math.degrees(new_lat_rad), math.degrees(new_lon_rad)

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculates distance in meters between two GPS points."""
    R = 6378137.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

def run_enterprise_pipeline():
    print("Initiating Object Detection & Spatial Mapping Pipeline...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print("Hardware Configuration: Enabling PyTorch CuDNN Auto-Tuner")
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
    use_half = True if device == 'cuda' else False
    
    model = YOLO(MODEL_PATH)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    all_detections = []
    
    start_time = time.time()

    # THE MASTER LOOP: Search the Parent Folder for Camera folders
    for folder_name in os.listdir(PARENT_FOLDER):
        folder_path = os.path.join(PARENT_FOLDER, folder_name)
        
        if not os.path.isdir(folder_path):
            continue
            
        # Identify which camera this is
        current_cam_key = None
        current_mount_angle = None
        for cam_key, angle in CAMERA_ANGLES.items():
            if cam_key in folder_name:
                current_cam_key = cam_key
                current_mount_angle = angle
                break
                
        if not current_cam_key:
            continue # Skip folders that don't match our Camera list
            
        print(f"\n---> Initializing {current_cam_key} (Angle: {current_mount_angle}°)")
        
        # Auto-Detect Coordinate File inside this specific Camera's folder
        coord_file_path = None
        for file in os.listdir(folder_path):
            if "coordonate" in file.lower():
                coord_file_path = os.path.join(folder_path, file)
                break
                
        if not coord_file_path:
            print(f"  [!] Skipping {current_cam_key}: No 'coordonate' file found.")
            continue
            
        if coord_file_path.endswith('.csv'):
            df_coords = pd.read_csv(coord_file_path)
        else:
            df_coords = pd.read_excel(coord_file_path)

        # Count the total images in the folder so the progress bar knows when it's at 100%
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        total_images = len(image_files)
                
        print(f"\n---> Initializing {current_cam_key} (Angle: {current_mount_angle}°) - Found {total_images} images")
        
        # ... (coordinate loading stays exactly the same) ...

        # Run AI on this specific folder
        results = model.predict(
            source=folder_path, conf=CONFIDENCE, iou=IOU_THRESHOLD, 
            imgsz=IMAGE_WIDTH, augment=USE_TTA, half=use_half, device=device, 
            stream=True, verbose=False, batch=BATCH_SIZE
        )

        folder_detections = 0
        
        # WRAP THE LOOP IN TQDM FOR THE PROGRESS BAR
        for r in tqdm(results, total=total_images, desc=f"Scanning {current_cam_key}", unit="img"):
            if len(r.boxes) == 0:
                continue
            
            # ... (the rest of the bounding box and GPS math stays exactly the same) ...
                
            img_name = os.path.basename(r.path)
            car_data = df_coords[df_coords['Imagine'] == img_name]
            if car_data.empty:
                continue
                
            car_lat = car_data.iloc[0]['Lat_WGS84']
            car_lon = car_data.iloc[0]['Lon_WGS84']
            car_heading = car_data.iloc[0]['Heading_deg']

            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                
                # Calculate passing the specific camera's angle and center X pixel
                center_x = (x1 + x2) / 2.0
                firida_lat, firida_lon = calculate_gps_offset(car_lat, car_lon, car_heading, center_x, y2, current_mount_angle)
                
                all_detections.append({
                    'image': img_name, 'cam_key': current_cam_key, 'folder_path': folder_path,
                    'x1': int(x1), 'y1': int(y1), 'x2': int(x2), 'y2': int(y2),
                    'conf': conf, 'lat': firida_lat, 'lon': firida_lon
                })
                folder_detections += 1
                
        print(f"  [✓] {current_cam_key} Complete. Found {folder_detections} raw detections.")

    print(f"Extraction complete. {len(all_detections)} total raw detections recorded.")
    
    print("Running spatial deduplication and proximity clustering...")
    
    unique_firidas = []
    
    for det in all_detections:
        matched = False
        for uf in unique_firidas:
            dist = haversine_distance(det['lat'], det['lon'], uf['lat'], uf['lon'])
            
            if dist <= DEDUPE_RADIUS_M:
                # OVERRIDE: If from the exact same photo, they are side-by-side boxes
                if det['image'] == uf['image'] and det['cam_key'] == uf['cam_key']:
                    continue 
                
                # Cross-Camera Merging: Keep the absolute best shot, regardless of which camera took it!
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

    print("Generating annotated visuals...")

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
        
        cv2.rectangle(img, (f['x1'], f['y1']), (f['x2'], f['y2']), color, 4)
        cv2.putText(img, label, (f['x1'], f['y1'] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        # Save filename prefixed with Camera name to prevent overwriting images with the same frame number
        save_filename = f"{f['cam_key']}_{f['image']}"
        save_path = os.path.join(OUTPUT_FOLDER, save_filename)
        cv2.imwrite(save_path, img)

    print("Visual generation complete. Compiling QGIS deliverables...")

    if len(unique_firidas) > 0:
        df_export = pd.DataFrame(unique_firidas)
        # Keep 'cam_key' in the dataframe so engineers know which camera took the shot
        df_export = df_export.drop(columns=['x1', 'y1', 'x2', 'y2', 'folder_path'])
        
        geometry = [Point(xy) for xy in zip(df_export['lon'], df_export['lat'])]
        gdf = gpd.GeoDataFrame(df_export, geometry=geometry)
        gdf.set_crs(epsg=4326, inplace=True)
        
        shp_path = os.path.join(OUTPUT_FOLDER, "tip_firida_bransament.shp")
        gdf.to_file(shp_path)
        
        # Export as JSON for API ingestion
        json_path = os.path.join(OUTPUT_FOLDER, "tip_firida_bransament.json")
        df_export.to_json(json_path, orient="records")
        
        print(f"Shapefile successfully generated: {shp_path}")
    else:
        print("No Firidas to export to map.")

    end_time = time.time()
    print(f"Pipeline executed successfully in {round(end_time - start_time, 2)} seconds.")
    print(f"Artifacts preserved in: {OUTPUT_FOLDER}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, default=PARENT_FOLDER)
    parser.add_argument("--conf", type=float, default=CONFIDENCE)
    parser.add_argument("--cluster", type=float, default=CLUSTER_RADIUS_M)
    parser.add_argument("--batch", type=int, default=BATCH_SIZE)
    # 1. ADD THE NEW OUTPUT ARGUMENT
    parser.add_argument("--output", type=str, required=True) 
    args = parser.parse_args()

    PARENT_FOLDER = args.folder
    CONFIDENCE = args.conf
    CLUSTER_RADIUS_M = args.cluster
    BATCH_SIZE = args.batch
    # 2. OVERRIDE THE HARDCODED FOLDER
    OUTPUT_FOLDER = args.output 

    if not os.path.exists(PARENT_FOLDER):
        print(f"ERROR: Cannot reach {PARENT_FOLDER}.")
    else:
        run_enterprise_pipeline()