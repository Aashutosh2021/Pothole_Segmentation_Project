from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
from ultralytics import YOLO
import cv2
import os
import numpy as np
import random
import uuid
import json
import base64
from typing import Optional

app = FastAPI(title="CityFix Pothole Detection API")

# Add CORS middleware to allow frontend to access the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = YOLO(r"C:\Users\gaash\Downloads\Pothole_Segmentation_Project\Project\backend\best.pt")

DB_FILE = "database.json"

def load_db():
    if not os.path.exists(DB_FILE):
        default_db = {
            "public_complaints": [
                
            ],
            "ai_detections": [
               
            ]
        }
        with open(DB_FILE, "w") as f:
            json.dump(default_db, f, indent=4)
        return default_db
    with open(DB_FILE, "r") as f:
        return json.load(f)

def save_db(db_data):
    with open(DB_FILE, "w") as f:
        json.dump(db_data, f, indent=4)

@app.get("/api/v1/stats/summary")
async def get_summary():
    db = load_db()
    ai_count = len(db["ai_detections"])
    pub_count = len(db["public_complaints"])
    
    return {
        "ai_detections": ai_count,
        "public_complaints": pub_count,
        "total_reports": ai_count + pub_count,
        "pending": sum(1 for x in db["public_complaints"] if x["status"] == "Pending"),
        "in_progress": sum(1 for x in db["public_complaints"] if x["status"] == "In Progress"),
        "resolved": sum(1 for x in db["public_complaints"] if x["status"] == "Resolved")
    }

@app.get("/api/v1/public_complaints")
async def get_public_complaints():
    db = load_db()
    return db["public_complaints"]

@app.get("/api/v1/ai_detections")
async def get_ai_detections():
    db = load_db()
    return db["ai_detections"]

@app.post("/api/v1/run_ai_analysis")
async def process_images(background_tasks: BackgroundTasks):
    source = 0  # Use webcam as source. Change to a video file path if needed.
    background_tasks.add_task(analyze_and_store, source)
    return {"message": "AI Background Analysis Started"}

class ComplaintCreate(BaseModel):
    title: str
    location: str
    gps: str
    description: str
    image_base64: Optional[str] = None

class ComplaintUpdate(BaseModel):
    status: str
    repaired_image_base64: Optional[str] = None
    worker_gps: Optional[str] = None

class ImagePayload(BaseModel):
    image_base64: str

def determine_severity_from_base64(image_base64: str) -> str:
    severity = "LOW"
    try:
        if image_base64 and "," in image_base64:
            header, encoded = image_base64.split(",", 1)
            file_bytes = base64.b64decode(encoded)
            np_arr = np.frombuffer(file_bytes, np.uint8)
            img_cv2 = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if img_cv2 is not None:
                results = model.predict(source=img_cv2)
                for r in results:
                    if r.masks is not None:
                        areas = [np.sum(mask) for mask in r.masks.data.cpu().numpy()]
                        if len(areas) > 0:
                            max_area = max(areas)
                            if max_area > 20000:
                                severity = "CRITICAL"
                            elif max_area > 10000:
                                severity = "HIGH"
                            elif max_area > 5000:
                                severity = "MEDIUM"
                            else:
                                severity = "LOW"
    except Exception as e:
        print(f"Error analyzing image: {e}")
    return severity

@app.post("/api/v1/analyze_image")
async def analyze_image_endpoint(payload: ImagePayload):
    severity = determine_severity_from_base64(payload.image_base64)
    return {"severity": severity}

@app.post("/api/v1/public_complaints")
async def create_complaint(complaint: ComplaintCreate):
    db = load_db()
    
    severity = determine_severity_from_base64(complaint.image_base64)

    new_complaint = {
        "id": f"PUB-{len(db['public_complaints']) + 1:03d}",
        "title": complaint.title,
        "location": complaint.location,
        "gps": complaint.gps,
        "severity": severity,
        "status": "Pending",
        "description": complaint.description,
        "image": complaint.image_base64 if complaint.image_base64 else "https://images.unsplash.com/photo-1449824913935-59a10b8d2000?w=400&h=220&fit=crop"
    }
    db["public_complaints"].insert(0, new_complaint)
    save_db(db)
    return new_complaint

@app.put("/api/v1/public_complaints/{complaint_id}")
async def update_complaint(complaint_id: str, update_data: ComplaintUpdate):
    import re
    db = load_db()
    for complaint in db["public_complaints"]:
        if complaint["id"] == complaint_id:
            if update_data.status:
                complaint["status"] = update_data.status
            if update_data.repaired_image_base64:
                if not update_data.repaired_image_base64.startswith("data:image/"):
                    return {"error": "Invalid image format. Must be a base64 data URI."}
                complaint["repaired_image"] = update_data.repaired_image_base64
            if update_data.worker_gps:
                if not re.match(r"^-?\d+(\.\d+)?,\s*-?\d+(\.\d+)?$", update_data.worker_gps):
                    return {"error": "Invalid GPS format. Must be 'lat, lon'."}
                complaint["worker_gps"] = update_data.worker_gps
            save_db(db)
            return complaint
    return {"error": "Complaint not found"}

def analyze_and_store(source):
    if not os.path.exists(source):
        return

    results = model.predict(source=source, stream=True)
    count = 0
    for r in results:
        if r.masks is not None:
            # For each detected mask, mock a GPS coordinate and store it
            areas = [np.sum(mask) for mask in r.masks.data.cpu().numpy()]
            mask_count = len(areas)
            
            # Since an image/frame could have multiple potholes, let's store them as 1 detection event
            # or separate events. Let's do separate events for simplicity.
            for area in areas:
                count += 1
                severity = "Low"
                if area > 10000: severity = "Medium"
                if area > 20000: severity = "High"
                
                lat = 40.7128 + random.uniform(-0.05, 0.05)
                lon = 74.0060 + random.uniform(-0.05, 0.05)
                
                db = load_db()
                db["ai_detections"].insert(0, {
                    "id": f"AI-NEW-{count}-{uuid.uuid4().hex[:4]}",
                    "location": "Detected via Video Analysis",
                    "gps": f"{lat:.4f}° N, {abs(lon):.4f}° W",
                    "severity": severity.upper(),
                    "status": "AI Logged",
                    "potholes_count": 1
                })
                save_db(db)

# Mount the frontend directory to serve HTML files directly
app.mount("/", StaticFiles(directory=r"C:\Users\gaash\Downloads\Pothole_Segmentation_Project\Project\frontend", html=True), name="frontend")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)