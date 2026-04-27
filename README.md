# Gauss Urban Infrastructure Detector

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-00a393?style=for-the-badge&logo=fastapi)
![Next.js](https://img.shields.io/badge/Next.js-14%2B-black?style=for-the-badge&logo=next.js)
![LiDAR](https://img.shields.io/badge/LiDAR-KDTree_Engine-red?style=for-the-badge)
![License](https://img.shields.io/badge/License-Proprietary-darkred?style=for-the-badge)

An enterprise-grade computer vision and geospatial mapping suite engineered to process Ladybug 5 panoramic imagery and automatically detect, classify, and geographically anchor electrical distribution boxes (*firidas*).

Designed for high-precision surveying, the platform fuses AI object detection with dense LiDAR point clouds to extract sub-centimeter 3D spatial intelligence, exporting directly to strict enterprise GIS schemas.

---

## 🏗️ Architecture Overview

The Gauss Infrastructure Detector operates on a dual-stack architecture designed for robust, asynchronous processing and elegant human-in-the-loop verification.

### AI & Geospatial Backend (Python / FastAPI)
The core engine relies on a fine-tuned YOLOv8 model (`firida_detector_v4_verygood.pt`) for high-resolution optical detection. Its true power, however, lies in its spatial processing capabilities:
- **3D LiDAR Raycasting:** Unprojects 2D bounding boxes into 3D vectors and casts virtual cylinders into a KDTree of the LiDAR point cloud to extract precise XYZ coordinates. 
- **Planar Fallback:** Intelligently drops down to flat-earth trigonometric estimation if the point cloud is occluded or too sparse.
- **Euclidean Clustering:** Deduplicates infrastructure seen from multiple camera angles using planar mathematics natively in the Stereo70 coordinate system.
- **Strict GIS Export:** Generates QGIS-ready shapefiles (PointZ) locked to the EPSG:3844 (Pulkovo 1942 / Stereo 70) projection, rigorously enforcing a 24-column enterprise metadata schema.

### Verification Console (Next.js / React)
A modern, responsive web application serving as the operator's dashboard:
- **Live Telemetry:** Tracks running Python subprocesses and multi-camera scanning progress in real-time.
- **Interactive Triage:** Allows operators to rapidly verify AI detections, assign metadata (like house numbers), and soft-delete false positives using a streamlined Lightbox gallery.
- **Dynamic Cartography:** Utilizes `proj4` and `react-leaflet` to seamlessly re-project Stereo70 payloads into WGS84 on the fly for OpenStreetMap visualization without compromising the backend's strict metric schema.

---

## 📚 Technical Documentation

For a deep dive into the mathematics and algorithms powering our spatial engine, refer to our architecture documentation:

- 📖 [The 3D LiDAR Raycasting Engine](docs/technical/01_lidar_3d_raycasting.md) - *Explains camera unprojection, geoid calibration, and KDTree intersection.*
- 📖 [Planar Fallback Geometry](docs/technical/02_planar_fallback_geometry.md) - *Explains the flat-earth trigonometric failover for occluded targets.*

---

## 🚀 Quick Start Guide

Ensure you have Python 3.10+ and Node.js v18+ installed on your system.

### 1. Initialize the Python Backend
The backend utilizes isolated virtual environments to prevent system-level package conflicts.

```bash
# Clone the repository and setup the environment
git clone https://github.com/MandaFlavian-Alexandru/Gauss-Urban_Infrastructure_Detector.git
cd Gauss-Urban_Infrastructure_Detector

# Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\activate

# Install core dependencies
pip install -r requirements.txt

# Launch the FastAPI cluster
uvicorn Gauss_API:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Initialize the Next.js Frontend
Open a **new terminal window**, leaving the Python backend running in the background.

```bash
cd frontend

# Install UI dependencies (including Leaflet and Proj4)
npm install

# Launch the development server
npm run dev
```

Navigate to `http://localhost:3000` in your web browser to access the Verification Console.

---

## 📡 Core API Capabilities

The backend exposes a suite of endpoints designed to orchestrate the LiDAR engine:

- `POST /api/analyze` - Spawns an asynchronous inference worker, feeding Ladybug 5 imagery and telemetry through the YOLO and KDTree pipelines.
- `GET /api/status` - Streams real-time execution telemetry and granular camera progress back to the Next.js dashboard.
- `GET /images/{image_name}` - Dynamically serves annotated validation imagery out of temporary staging directories.
- `POST /api/generate_final_export` - Receives verified targets from the frontend, enforces the 24-column enterprise schema, and compiles the definitive spatial dataset.
- `GET /api/download_shapefile` - Packages the finalized `.shp`, `.shx`, `.dbf`, `.cpg`, `.prj`, and `.json` deliverables into a downloadable ZIP archive.

---

*Engineered by Gauss for superior urban intelligence.*
