from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import shutil
import cv2
import os
import requests
import base64
from typing import List, Optional
from report import generate_report

app = FastAPI(title="AutoSpect - AI Vehicle Inspection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Roboflow API configuration
ROBOFLOW_API_KEY = "bUF0vK5fXo62uixEN4PN"
ROBOFLOW_MODEL_ID = "car-damage-detection-t0g92"
ROBOFLOW_VERSION = "3"
ROBOFLOW_API_URL = f"https://detect.roboflow.com/{ROBOFLOW_MODEL_ID}/{ROBOFLOW_VERSION}"

# Single folder structure - everything in same directory
UPLOAD_DIR = "uploads"
STATIC_DIR = "static"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

REPORT_PATH = os.path.join(STATIC_DIR, "inspection_report.pdf")

CLASS_MAPPING = {
    "bonnet": "Bonnet",
    "bumper": "Bumper",
    "dickey": "Dickey",
    "door": "Door",
    "fender": "Fender",
    "light": "Light",
    "windshield": "Windshield"
}

# Global state for live detection captures
captured_frames = []
captured_defect_types = set()
captured_all_defects = []  # NEW: Store all defects with confidence for report generation


def detect_damage(image_path: str):
    """Run Roboflow inference on an image"""
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")
    
    response = requests.post(
        ROBOFLOW_API_URL,
        params={
            "api_key": ROBOFLOW_API_KEY,
            "confidence": 40,
            "overlap": 30
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data=image_data
    )
    
    if response.status_code != 200:
        raise Exception(f"Roboflow API error: {response.text}")
    
    return response.json()


@app.post("/detect-live")
async def detect_live(file: UploadFile = File(...)):
    """
    Live detection endpoint - analyzes a single frame from iVCam,
    returns detected defects, and auto-captures if new defect found.
    """
    global captured_frames, captured_defect_types, captured_all_defects
    
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    temp_path = os.path.join(UPLOAD_DIR, "temp_frame.jpg")
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        result = detect_damage(temp_path)
        predictions = result.get("predictions", [])
        
        frame = cv2.imread(temp_path)
        defects = []
        new_defect_found = False
        
        for pred in predictions:
            class_name = pred["class"].lower().strip()
            display_name = CLASS_MAPPING.get(class_name, class_name.capitalize())
            confidence = round(pred["confidence"] * 100, 1)
            
            defects.append({
                "class": display_name,
                "confidence": confidence
            })
            
            if class_name not in captured_defect_types:
                captured_defect_types.add(class_name)
                new_defect_found = True
        
        annotated_frame = frame.copy()
        for pred in predictions:
            x = int(pred["x"])
            y = int(pred["y"])
            w = int(pred["width"])
            h = int(pred["height"])
            
            class_name = pred["class"].lower().strip()
            display_name = CLASS_MAPPING.get(class_name, class_name.capitalize())
            confidence = round(pred["confidence"] * 100, 1)
            
            cv2.rectangle(
                annotated_frame,
                (x - w // 2, y - h // 2),
                (x + w // 2, y + h // 2),
                (0, 0, 255),
                4
            )
            
            label_text = f"{display_name} {confidence}%"
            (text_width, text_height), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
            )
            
            cv2.rectangle(
                annotated_frame,
                (x - w // 2, y - h // 2 - text_height - 10),
                (x - w // 2 + text_width + 10, y - h // 2),
                (0, 0, 255),
                -1
            )
            
            cv2.putText(
                annotated_frame,
                label_text,
                (x - w // 2 + 5, y - h // 2 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
        
        # NEW: If new defect found, capture this frame AND store defects
        if new_defect_found and len(predictions) > 0:
            capture_filename = f"capture_{len(captured_frames)}.jpg"
            capture_path = os.path.join(STATIC_DIR, capture_filename)
            cv2.imwrite(capture_path, annotated_frame)
            captured_frames.append(capture_path)
            
            # NEW: Store the defects for this frame
            for pred in predictions:
                class_name = pred["class"].lower().strip()
                display_name = CLASS_MAPPING.get(class_name, class_name.capitalize())
                confidence = round(pred["confidence"] * 100, 1)
                captured_all_defects.append((display_name, confidence))
        
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return JSONResponse({
            "success": True,
            "defects": defects,
            "count": len(defects),
            "new_capture": new_defect_found and len(predictions) > 0,
            "total_captures": len(captured_frames),
            "unique_defects": len(captured_defect_types),
            "annotated_frame": frame_base64
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/finalize-live-detection")
async def finalize_live_detection(
    vin: Optional[str] = Form(None),
    make: Optional[str] = Form(None),
    model: Optional[str] = Form(None),
    year: Optional[str] = Form(None),
    mileage: Optional[str] = Form(None)
):
    """
    Generate report from captured frames during live detection.
    """
    global captured_frames, captured_defect_types, captured_all_defects
    
    if len(captured_frames) == 0:
        raise HTTPException(status_code=400, detail="No frames captured during live detection")
    
    # NEW: Use the defects that were already collected during live detection
    # This prevents re-running inference on already processed frames
    all_defects = captured_all_defects.copy()
    
    vehicle_info = {
        "vin": vin or "Not Provided",
        "make": make or "Not Provided",
        "model": model or "Not Provided",
        "year": year or "Not Provided",
        "mileage": mileage or "Not Provided"
    }
    
    try:
        generate_report(all_defects, captured_frames, REPORT_PATH, vehicle_info)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate report: {str(e)}")
    
    unique_defect_types = len(captured_defect_types)
    annotated_image_paths = [f"static/{os.path.basename(p)}" for p in captured_frames]
    
    # Reset global state
    captured_frames = []
    captured_defect_types = set()
    captured_all_defects = []  # NEW: Reset stored defects
    
    return {
        "message": "Live detection report generated",
        "image_count": len(annotated_image_paths),
        "total_defects_detected": len(all_defects),
        "unique_defect_types": unique_defect_types,
        "defects_detected": all_defects,
        "annotated_images": annotated_image_paths
    }


@app.post("/reset-live-detection")
async def reset_live_detection():
    """Reset live detection state."""
    global captured_frames, captured_defect_types, captured_all_defects
    
    for frame_path in captured_frames:
        if os.path.exists(frame_path):
            try:
                os.remove(frame_path)
            except:
                pass
    
    captured_frames = []
    captured_defect_types = set()
    captured_all_defects = []  # NEW: Reset stored defects
    
    return {"message": "Live detection reset"}


@app.post("/inspect")
async def inspect_vehicle(
    files: List[UploadFile] = File(...),
    vin: Optional[str] = Form(None),
    make: Optional[str] = Form(None),
    model: Optional[str] = Form(None),
    year: Optional[str] = Form(None),
    mileage: Optional[str] = Form(None)
):
    """
    Full inspection endpoint - processes uploaded images, creates annotations, and generates report.
    """
    if not files or len(files) == 0:
        raise HTTPException(status_code=400, detail="At least one image file is required")

    for file in files:
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="All uploaded files must be images")

    all_defects = []
    annotated_image_paths = []
    saved_annotated_paths = []

    for idx, file in enumerate(files):
        input_path = os.path.join(UPLOAD_DIR, f"input_{idx}.jpg")
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        try:
            result = detect_damage(input_path)
        except Exception as e:
            if os.path.exists(input_path):
                os.remove(input_path)
            raise HTTPException(status_code=500, detail=f"Inference failed on image {idx+1}: {str(e)}")

        image = cv2.imread(input_path)
        if image is None:
            os.remove(input_path)
            raise HTTPException(status_code=500, detail=f"Failed to load image {idx+1}")

        predictions = result.get("predictions", [])

        for pred in predictions:
            class_name = pred["class"].lower().strip()
            display_name = CLASS_MAPPING.get(class_name, class_name.capitalize())
            confidence = round(pred["confidence"] * 100, 1)
            all_defects.append((display_name, confidence))

        for pred in predictions:
            x = int(pred["x"])
            y = int(pred["y"])
            w = int(pred["width"])
            h = int(pred["height"])

            cv2.rectangle(
                image,
                (x - w // 2, y - h // 2),
                (x + w // 2, y + h // 2),
                (0, 0, 255),
                4
            )

            label_text = f"{CLASS_MAPPING.get(pred['class'].lower().strip(), pred['class'].capitalize())} {round(pred['confidence']*100, 1)}%"
            (text_width, text_height), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
            )

            cv2.rectangle(
                image,
                (x - w // 2, y - h // 2 - text_height - 10),
                (x - w // 2 + text_width + 10, y - h // 2),
                (0, 0, 255),
                -1
            )

            cv2.putText(
                image,
                label_text,
                (x - w // 2 + 5, y - h // 2 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

        # NEW: Only save annotated image if defects were detected
        if len(predictions) > 0:
            annotated_filename = f"annotated_{idx}.jpg"
            annotated_abs_path = os.path.join(STATIC_DIR, annotated_filename)
            cv2.imwrite(annotated_abs_path, image)

            annotated_image_paths.append(f"static/{annotated_filename}")
            saved_annotated_paths.append(annotated_abs_path)

        if os.path.exists(input_path):
            os.remove(input_path)

    vehicle_info = {
        "vin": vin or "Not Provided",
        "make": make or "Not Provided",
        "model": model or "Not Provided",
        "year": year or "Not Provided",
        "mileage": mileage or "Not Provided"
    }

    try:
        generate_report(all_defects, saved_annotated_paths, REPORT_PATH, vehicle_info)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate report: {str(e)}")

    unique_defect_types = len({component.lower() for component, _ in all_defects})

    return {
        "message": "Inspection complete",
        "image_count": len(files),
        "total_defects_detected": len(all_defects),
        "unique_defect_types": unique_defect_types,
        "defects_detected": all_defects,
        "annotated_images": annotated_image_paths
    }


@app.get("/report")
def get_report():
    if not os.path.exists(REPORT_PATH):
        raise HTTPException(status_code=404, detail="Report not generated yet. Please inspect a vehicle first.")

    headers = {
        "Content-Disposition": "inline; filename=AutoSpect_Vehicle_Inspection_Report.pdf"
    }
    return FileResponse(
        REPORT_PATH,
        media_type="application/pdf",
        headers=headers,
        filename="AutoSpect_Vehicle_Inspection_Report.pdf"
    )


# Root endpoint - serves index.html
@app.get("/")
async def read_root():
    return FileResponse("index.html")

# NEW: Serve app.js file
@app.get("/app.js")
async def get_app_js():
    return FileResponse("app.js", media_type="application/javascript")

# Mount static directories
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")


# For running with uvicorn programmatically (used by Render)
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
