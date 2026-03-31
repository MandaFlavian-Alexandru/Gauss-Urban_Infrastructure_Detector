# Gauss Firida Detector

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-00a393)
![License](https://img.shields.io/badge/License-Proprietary-red)

Enterprise computer vision mapping tool designed to process Ladybug 5 panoramic imagery and automatically detect electrical distribution boxes (firidas).

## Architecture Overview
The system relies on a fine-tuned YOLOv8 model for object detection within high-resolution panoramic frames. 

**Model & Training Data**
- The production weights (`firida_detector_v4_verygood.pt`) are included directly in the repository for immediate deployment.
- The raw dataset used for fine-tuning this model is archived externally on [Google Drive](https://drive.google.com/drive/folders/1uw5cL-kKW_8aHGDqBC2TvdWZFL8Ai68P).

The core innovation is the capability to translate 2D bounding boxes into precise real-world 3D WGS84 coordinates. 

The backend is built as a FastAPI multi-worker cluster (via Uvicorn) that manages long-running inference jobs as background subprocesses, communicating through a centralized state file. Spatial filtering, including clustering and deduplication, is performed using Haversine heuristics before the final vector data is exported as QGIS-ready shapefiles.

## Installation

Ensure you have Python 3.10+ installed.

1. Clone the repository
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Server

Start the API with Uvicorn in multi-worker mode:
```bash
python -m uvicorn Gauss_API:app --host 0.0.0.0 --port 8000 --workers 4
```

## API Endpoints

- `POST /api/analyze` - Starts a multi-threaded inference subprocess given a target source directory containing Ladybug 5 imagery and telemetry data.
- `GET /api/status` - Returns real-time execution telemetry, localized camera progress, and logs for the currently active background job.
- `GET /images/{image_name}` - Serves bounding-box annotated images generated during the pipeline execution for frontend review.
- `GET /api/results` - Retrieves the live JSON spatial dataset containing all unique, deduplicated detections.
- `POST /api/delete_result` - Dynamically purges false-positive detections from both the JSON and the active shapefile.
- `GET /api/download_shapefile` - Streams a ZIP archive back to the client containing the finalized QGIS `.shp`, `.shx`, `.dbf`, `.cpg`, and `.prj` deliverables.
- `POST /api/cancel` - Broadcasts a termination signal to abruptly halt the running inference worker.
