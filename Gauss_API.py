import os
import sys
import json
import logging
import subprocess
import threading
import re
import zipfile
import glob
import tempfile
import signal
import time
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, FileResponse
from pydantic import BaseModel

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Gauss_FastAPI")

app = FastAPI(title="Gauss Infrastructure Detector API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- NEW: Centralized State Management for Clustered Workers ---
STATE_FILE = os.path.join(tempfile.gettempdir(), "gauss_state.json")

def load_state():
    try:
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {
            "is_running": False,
            "progress": 0,
            "camera_progress": {"Camera1": 0, "Camera2": 0, "Camera3": 0, "Camera4": 0},
            "logs": [],
            "results_ready": False,
            "current_output_dir": "", 
            "source_folder": "",
            "process_id": None
        }

def save_state(state):
    try:
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f)
    except Exception:
        pass

class AnalysisRequest(BaseModel):
    folder_path: str
    min_confidence: float
    cluster_radius: float

class DeleteRequest(BaseModel):
    image_name: str
    cam_key: str

class FinalExportRequest(BaseModel):
    results: list

def run_subprocess(folder_path, conf, cluster, output_dir):
    """Executes the Gauss backend pipeline as a subprocess and monitors its stdout."""
    for f in glob.glob(os.path.join(output_dir, "tip_firida_bransament.*")): 
        try: 
            os.remove(f)
        except OSError as e:
            logger.warning(f"Could not remove old file {f}: {e}")

    st = load_state()
    st["is_running"] = True
    st["logs"] = [f"Initializing AI Analysis in: {folder_path}"]
    st["progress"] = 0
    st["camera_progress"] = {"Camera1": 0, "Camera2": 0, "Camera3": 0, "Camera4": 0}
    st["results_ready"] = False
    
    cmd = [
        sys.executable, "Gauss_UID_Backend.py",  # <--- CHANGE THIS LINE
        "--folder", folder_path,
        "--conf", str(conf / 100.0),
        "--cluster", str(cluster),
        "--output", output_dir 
    ]
    
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    st["process_id"] = process.pid
    save_state(st)
    
    for line in iter(process.stdout.readline, ''):
        line = line.strip()
        if line:
            st = load_state() # Refresh state before updating
            st["logs"].append(line)
            # Keep logs manageable
            if len(st["logs"]) > 50: 
                st["logs"] = st["logs"][-50:]
                
            cam_m = re.search(r'Scanning (Camera[1-4]).*?(\d+)%', line)
            if cam_m:
                st["camera_progress"][cam_m.group(1)] = int(cam_m.group(2))
                
            if "PHASE 1" in line: st["progress"] = 5
            elif "Scanning Camera" in line: st["progress"] += 15
            elif "PHASE 2" in line: st["progress"] = 70
            elif "PHASE 3" in line: st["progress"] = 80
            elif "PHASE 4" in line: st["progress"] = 95
            save_state(st)
            
    if process.stdout:
        process.stdout.close()
    process.wait()
    
    st = load_state()
    st["progress"] = 100
    st["camera_progress"] = {"Camera1": 100, "Camera2": 100, "Camera3": 100, "Camera4": 100}
    st["is_running"] = False
    st["results_ready"] = True
    st["process_id"] = None
    save_state(st)

@app.post("/api/analyze")
def start_analysis(req: AnalysisRequest):
    st = load_state()
    if st["is_running"]:
        return {"status": "already_running"}
    
    st["source_folder"] = req.folder_path
    
    dynamic_output_dir = os.path.join(tempfile.gettempdir(), f"Gauss_App_Staging_{int(time.time())}")
    os.makedirs(dynamic_output_dir, exist_ok=True)
    
    st["current_output_dir"] = dynamic_output_dir
    save_state(st)
    
    thread = threading.Thread(target=run_subprocess, args=(req.folder_path, req.min_confidence, req.cluster_radius, dynamic_output_dir))
    thread.start()
    return {"status": "started"}

@app.get("/api/status")
def get_status():
    st = load_state()
    return {
        "is_running": st["is_running"],
        "progress": min(st["progress"], 100),
        "camera_progress": st["camera_progress"],
        "logs": st["logs"][-20:],
        "results_ready": st["results_ready"]
    }

@app.get("/images/{image_name}")
def get_image(image_name: str):
    """Serves annotated images from the active inference staging directory."""
    st = load_state()
    if not st.get("current_output_dir"):
        return Response(status_code=404)
    
    image_path = os.path.join(st["current_output_dir"], image_name)
    if os.path.exists(image_path):
        return FileResponse(image_path)
    return Response(status_code=404)

@app.get("/api/results")
def get_results():
    st = load_state()
    if not st["results_ready"] or not st["current_output_dir"]:
        return {"status": "not_ready"}
    try:
        json_path = os.path.join(st["current_output_dir"], "tip_firida_bransament.json")
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        return {"error": str(e)}
    return []

@app.post("/api/delete_result")
def delete_result(req: DeleteRequest):
    st = load_state()
    if not st["current_output_dir"]:
        return {"status": "error", "message": "No active directory"}

    json_path = os.path.join(st["current_output_dir"], "tip_firida_bransament.json")
    shp_path = os.path.join(st["current_output_dir"], "tip_firida_bransament.shp")
    
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        new_data = [d for d in data if not (d['image'] == req.image_name and d['cam_key'] == req.cam_key)]
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(new_data, f)
            
        df = pd.DataFrame(new_data)
        if not df.empty:
            geometry = [Point(xy) for xy in zip(df['lon'], df['lat'])]
            gdf = gpd.GeoDataFrame(df, geometry=geometry)
            gdf.set_crs(epsg=4326, inplace=True)
            gdf.to_file(shp_path)
        else:
            if os.path.exists(shp_path):
                for ext in ['.shp', '.shx', '.dbf', '.cpg', '.prj']:
                    f_to_rem = shp_path.replace('.shp', ext)
                    if os.path.exists(f_to_rem):
                        os.remove(f_to_rem)
        
        return {"status": "success", "remaining": len(new_data)}
    return {"status": "error", "message": "JSON not found"}

@app.post("/api/generate_final_export")
def generate_final_export(req: FinalExportRequest):
    st = load_state()
    if not st["current_output_dir"]:
        return {"status": "error", "message": "No active directory"}
    
    # Filter only verified and classified results
    final_data = [d for d in req.results if d.get('verified') and d.get('classification')]
    
    if not final_data:
        # Still generate an empty dataframe for zip if requested, or return error. Returning error is safer to avoid creating malformed shapefiles.
        return {"status": "error", "message": "No verified detections to export."}
        
    df = pd.DataFrame(final_data)
    geometry = [Point(xy) for xy in zip(df['lon'], df['lat'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry)
    gdf.set_crs(epsg=4326, inplace=True)
    
    base_name = "tip_firida_bransament"
    
    # Overwrite the output files with the final data
    json_path = os.path.join(st["current_output_dir"], f"{base_name}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(final_data, f)
        
    shp_path = os.path.join(st["current_output_dir"], f"{base_name}.shp")
    # if it previously existed and we update, Geopandas handles overwriting cleanly mostly, but let's just use to_file
    gdf.to_file(shp_path)
    
    return {"status": "success"}

@app.get("/api/download_shapefile")
def download_shapefile():
    st = load_state()
    target_dir = st["current_output_dir"]
    base_name = "tip_firida_bransament"
    
    if not target_dir or not os.path.exists(os.path.join(target_dir, f"{base_name}.shp")):
        return Response(content="Shapefile not generated yet", status_code=404)
        
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    final_output_dir = os.path.join(desktop_path, f"Gauss_Results_{int(time.time())}")
    os.makedirs(final_output_dir, exist_ok=True)
    
    zip_path = os.path.join(final_output_dir, f"{base_name}.zip")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for ext in ['.shp', '.shx', '.dbf', '.prj', '.cpg', '.json']:
            file_path = os.path.join(target_dir, f"{base_name}{ext}")
            if os.path.exists(file_path):
                import shutil
                shutil.copy2(file_path, os.path.join(final_output_dir, f"{base_name}{ext}"))
                zip_file.write(file_path, arcname=f"{base_name}{ext}")
                
    # --- FIXED: Added headers to force the browser to trigger a download window ---
    return FileResponse(
        path=zip_path, 
        filename=f"{base_name}.zip", 
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={base_name}.zip"}
    )

@app.post("/api/cancel")
def cancel_analysis():
    st = load_state()
    pid = st.get("process_id")
    
    if pid:
        try:
            # Terminate the background process across workers
            os.kill(pid, signal.SIGTERM) 
        except Exception:
            pass
            
    st["is_running"] = False
    st["camera_progress"] = {"Camera1": 0, "Camera2": 0, "Camera3": 0, "Camera4": 0}
    st["logs"].append("ERROR: Analysis cancelled by user.")
    st["process_id"] = None
    save_state(st)
    return {"status": "cancelled"}