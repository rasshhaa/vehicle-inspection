from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import shutil
import cv2
import os
import asyncio
from typing import List, Optional
from inference_sdk import InferenceHTTPClient
from report import generate_report
import base64

app = FastAPI(title="AI Vehicle Inspection API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Roboflow client
# Read API key and model id from environment variables for production safety
ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY", "bUF0vK5fXo62uixEN4PN")
ROBOFLOW_MODEL_ID = os.environ.get("ROBOFLOW_MODEL_ID", "car-damage-detection-t0g92/3")

CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=ROBOFLOW_API_KEY
)

# File paths
UPLOAD_DIR = "../uploads"
STATIC_DIR = "../static"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

REPORT_PATH = os.path.join(STATIC_DIR, "inspection_report.pdf")

# Standardized class names
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
captured_all_defects = []  # Store all defects with confidence for report generation


@app.post("/detect-live")
async def detect_live(file: UploadFile = File(...)):
    """
    Live detection endpoint - analyzes a single frame from iVCam,
    returns detected defects, and auto-captures if new defect found.
    """
    global captured_frames, captured_defect_types, captured_all_defects
    
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Save frame temporarily
    temp_path = os.path.join(UPLOAD_DIR, "temp_frame.jpg")
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # Run inference
        result = CLIENT.infer(
            temp_path,
            model_id="car-damage-detection-t0g92/3"
        )
        
        predictions = result.get("predictions", [])
        
        # Load image for annotation
        frame = cv2.imread(temp_path)
        
        # Extract defect information and check for new defects
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
            
            # Check if this is a new defect type
            if class_name not in captured_defect_types:
                captured_defect_types.add(class_name)
                new_defect_found = True
        
        # Draw annotations on frame
        annotated_frame = frame.copy()
        for pred in predictions:
            x = int(pred["x"])
            y = int(pred["y"])
            w = int(pred["width"])
            h = int(pred["height"])
            
            class_name = pred["class"].lower().strip()
            display_name = CLASS_MAPPING.get(class_name, class_name.capitalize())
            confidence = round(pred["confidence"] * 100, 1)
            
            # Bounding box
            cv2.rectangle(
                annotated_frame,
                (x - w // 2, y - h // 2),
                (x + w // 2, y + h // 2),
                (0, 0, 255),
                4
            )
            
            # Label
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
        
        # If new defect found, capture this frame
        if new_defect_found and len(predictions) > 0:
            # Save captured frame
            capture_filename = f"capture_{len(captured_frames)}.jpg"
            capture_path = os.path.join(STATIC_DIR, capture_filename)
            cv2.imwrite(capture_path, annotated_frame)
            captured_frames.append(capture_path)
            
            # Store the defects for this frame
            for pred in predictions:
                class_name = pred["class"].lower().strip()
                display_name = CLASS_MAPPING.get(class_name, class_name.capitalize())
                confidence = round(pred["confidence"] * 100, 1)
                captured_all_defects.append((display_name, confidence))
        
        # Encode annotated frame as base64 for preview
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
        # Clean up temp file
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
    
    # Use the defects that were already collected during live detection
    all_defects = captured_all_defects.copy()
    
    # Prepare vehicle info
    vehicle_info = {
        "vin": vin or "Not Provided",
        "make": make or "Not Provided",
        "model": model or "Not Provided",
        "year": year or "Not Provided",
        "mileage": mileage or "Not Provided"
    }
    
    # Generate report
    try:
        generate_report(all_defects, captured_frames, REPORT_PATH, vehicle_info)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate report: {str(e)}")
    
    unique_defect_types = len(captured_defect_types)
    
    # Prepare annotated image paths for response
    annotated_image_paths = [f"static/{os.path.basename(p)}" for p in captured_frames]
    
    # Reset global state
    captured_frames = []
    captured_defect_types = set()
    captured_all_defects = []
    
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
    
    # Clean up captured frames
    for frame_path in captured_frames:
        if os.path.exists(frame_path):
            try:
                os.remove(frame_path)
            except:
                pass
    
    captured_frames = []
    captured_defect_types = set()
    captured_all_defects = []
    
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

    # Validate all files are images
    for file in files:
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="All uploaded files must be images")

    all_defects = []
    annotated_image_paths = []
    saved_annotated_paths = []

    for idx, file in enumerate(files):
        # Save original uploaded image temporarily
        input_path = os.path.join(UPLOAD_DIR, f"input_{idx}.jpg")
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        try:
            # Run inference
            result = CLIENT.infer(
                input_path,
                model_id="car-damage-detection-t0g92/3"
            )
        except Exception as e:
            if os.path.exists(input_path):
                os.remove(input_path)
            raise HTTPException(status_code=500, detail=f"Inference failed on image {idx+1}: {str(e)}")

        # Load image for annotation
        image = cv2.imread(input_path)
        if image is None:
            os.remove(input_path)
            raise HTTPException(status_code=500, detail=f"Failed to load image {idx+1}")

        predictions = result.get("predictions", [])

        # Collect defects
        for pred in predictions:
            class_name = pred["class"].lower().strip()
            display_name = CLASS_MAPPING.get(class_name, class_name.capitalize())
            confidence = round(pred["confidence"] * 100, 1)
            all_defects.append((display_name, confidence))

        # Draw bounding boxes and labels
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

        # Save annotated image only if defects detected
        if len(predictions) > 0:
            annotated_filename = f"annotated_{idx}.jpg"
            annotated_abs_path = os.path.join(STATIC_DIR, annotated_filename)
            cv2.imwrite(annotated_abs_path, image)

            annotated_image_paths.append(f"static/{annotated_filename}")
            saved_annotated_paths.append(annotated_abs_path)

        if os.path.exists(input_path):
            os.remove(input_path)

    # Prepare vehicle info
    vehicle_info = {
        "vin": vin or "Not Provided",
        "make": make or "Not Provided",
        "model": model or "Not Provided",
        "year": year or "Not Provided",
        "mileage": mileage or "Not Provided"
    }

    # Generate report
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
        "Content-Disposition": "inline; filename=AI_Vehicle_Inspection_Report.pdf"
    }
    return FileResponse(
        REPORT_PATH,
        media_type="application/pdf",
        headers=headers,
        filename="AI_Vehicle_Inspection_Report.pdf"
    )


# Static file mounts (must be after routes)
app.mount("/static", StaticFiles(directory="../static"), name="static")
app.mount("/", StaticFiles(directory="../frontend", html=True), name="frontend")