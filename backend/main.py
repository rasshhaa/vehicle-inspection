from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
import cv2
import os
import subprocess
import re
import time
import tempfile
import traceback
from typing import List, Optional
from inference_sdk import InferenceHTTPClient
from report import generate_report, generate_ai_analysis  # ✅ FIXED: removed dot
import base64
from dotenv import load_dotenv

import firebase_admin
from firebase_admin import credentials, firestore
load_dotenv()

# ✅ Firebase Initialization
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)
db = firestore.client()
print("✅ Firebase Connected!")

app = FastAPI(title="AI Vehicle Inspection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=os.getenv("ROBOFLOW_API_KEY", "bUF0vK5fXo62uixEN4PN")
)

UPLOAD_DIR = "uploads"
STATIC_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

REPORT_PATH = os.path.join(STATIC_DIR, "inspection_report.pdf")
VEHICLE_MODEL_ID = os.getenv("ROBOFLOW_MODEL_ID", "car-damage-detection-t0g92/3")

CLASS_MAPPING = {
    "bonnet": "Bonnet", "bumper": "Bumper", "dickey": "Dickey",
    "door": "Door", "fender": "Fender", "light": "Light", "windshield": "Windshield",
}

DEFECT_TYPE_LABELS = {
    "windshield": {"minor": "Surface Chip/Crack", "moderate": "Windshield Crack", "severe": "Shattered/Major Crack"},
    "bonnet":     {"minor": "Surface Scratch",    "moderate": "Bonnet Dent",       "severe": "Crumple/Deep Impact"},
    "bumper":     {"minor": "Scuff/Scratch",       "moderate": "Bumper Dent/Crack", "severe": "Bumper Collapse"},
    "fender":     {"minor": "Surface Scratch",    "moderate": "Fender Dent",       "severe": "Deep Dent/Crease"},
    "door":       {"minor": "Paint Scratch/Scuff","moderate": "Door Dent",         "severe": "Deep Dent/Panel Damage"},
    "dickey":     {"minor": "Surface Scratch",    "moderate": "Boot Dent",         "severe": "Deep Dent/Structural"},
    "light":      {"minor": "Cover Scratch",      "moderate": "Cracked Cover",     "severe": "Broken/Shattered"},
}


def get_defect_type_label(part_key: str, confidence: float) -> str:
    tier = "severe" if confidence >= 80 else "moderate" if confidence >= 55 else "minor"
    return DEFECT_TYPE_LABELS.get(part_key, {}).get(tier, f"{part_key.capitalize()} Damage")


def draw_annotation(image, pred):
    x, y = int(pred["x"]), int(pred["y"])
    w, h = int(pred["width"]), int(pred["height"])
    cn = pred["class"].lower().strip()
    cf = round(pred["confidence"] * 100, 1)
    dtype = get_defect_type_label(cn, cf)
    label_text = f"{dtype} {cf}%"
    x1, y1 = x - w // 2, y - h // 2
    x2, y2 = x + w // 2, y + h // 2
    box_color = (0, 0, 220) if cf >= 80 else (0, 120, 255) if cf >= 55 else (0, 200, 255)
    cv2.rectangle(image, (x1, y1), (x2, y2), box_color, 3)
    font, font_scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2
    (tw, th), _ = cv2.getTextSize(label_text, font, font_scale, thickness)
    cv2.rectangle(image, (x1, y1 - th - 10), (x1 + tw + 10, y1), box_color, -1)
    cv2.putText(image, label_text, (x1 + 5, y1 - 6), font, font_scale, (255, 255, 255), thickness)


captured_frames = []
captured_defect_types = set()
captured_all_defects = []
_engine_extractor = _engine_model = _engine_labels = None
_engine_target_sr = 16000

KNOCK_LABEL_KEYS = {"knock", "knocking", "engine_knock", "defective", "fault", "faulty"}
CLEAN_LABEL_KEYS = {"no_knock", "no knock", "clean", "healthy", "normal", "ok", "good"}


def _label_is_knock(label: str) -> bool:
    l = label.lower().replace("-", "_").replace(" ", "_")
    for c in CLEAN_LABEL_KEYS:
        if c.replace(" ", "_") in l: return False
    for k in KNOCK_LABEL_KEYS:
        if k.replace(" ", "_") in l: return True
    return "knock" in l and not l.startswith("no")


def _get_engine_model():
    global _engine_extractor, _engine_model, _engine_labels, _engine_target_sr
    if _engine_model is not None:
        return _engine_extractor, _engine_model, _engine_labels, _engine_target_sr
    try:
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
        import torch
        _engine_extractor = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        _engine_model = AutoModelForAudioClassification.from_pretrained("cxlrd/revix-AST-engine-knock")
        _engine_model.eval()
        _engine_labels = _engine_model.config.id2label
        _engine_target_sr = getattr(_engine_extractor, "sampling_rate", 16000)
        return _engine_extractor, _engine_model, _engine_labels, _engine_target_sr
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Engine model unavailable: {str(e)}")


def _convert_to_wav(input_path: str, target_sr: int) -> str:
    wav_path = input_path + "_converted.wav"
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", input_path, "-ar", str(target_sr), "-ac", "1", "-f", "wav", wav_path],
        capture_output=True, timeout=60,
    )
    if result.returncode != 0:
        raise HTTPException(status_code=500, detail=f"Audio conversion failed: {result.stderr.decode(errors='replace')[-600:]}")
    return wav_path


# ── /ai-analysis ──────────────────────────────────────────────────────────────
class AIAnalysisRequest(BaseModel):
    defects_detected: list
    unique_defect_types: int = 0
    vehicle_info: dict = {}
    engine_result: Optional[dict] = None
    overall_status: str = ""

@app.post("/ai-analysis")
async def ai_analysis_endpoint(req: AIAnalysisRequest):
    defects_norm = []
    for d in req.defects_detected:
        if isinstance(d, (list, tuple)) and len(d) >= 2:
            defects_norm.append((str(d[0]), float(d[1])))
        elif isinstance(d, dict):
            name = d.get("label") or d.get("class") or d.get("name") or "Unknown"
            defects_norm.append((name, float(d.get("confidence", 0))))
    overall_status = req.overall_status
    if not overall_status:
        ut = req.unique_defect_types or len({d[0].lower() for d in defects_norm})
        overall_status = "PASS" if ut == 0 else "ATTENTION" if ut <= 2 else "FAIL"
    try:
        result = generate_ai_analysis(defects=defects_norm, vehicle_info=req.vehicle_info,
                                       engine_result=req.engine_result, overall_status=overall_status)
        return {"success": True, "ai_analysis": result, "overall_status": overall_status}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI analysis failed: {str(e)}")


# ── /analyze-engine ───────────────────────────────────────────────────────────
@app.post("/analyze-engine")
async def analyze_engine(audio: UploadFile = File(...)):
    try:
        import librosa, torch
    except ImportError:
        raise HTTPException(status_code=503, detail="Audio libraries not installed.")
    extractor, model, labels, target_sr = _get_engine_model()
    suffix = os.path.splitext(audio.filename or "audio.webm")[-1].lower() or ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(audio.file, tmp)
        raw_path = tmp.name
    wav_path = None
    try:
        try: wav_path = _convert_to_wav(raw_path, target_sr)
        except FileNotFoundError: wav_path = raw_path
        waveform, _ = librosa.load(wav_path, sr=target_sr, mono=True)
        waveform = waveform.astype("float32")
        duration_s = round(len(waveform) / target_sr, 2)
        inputs = extractor(waveform, sampling_rate=target_sr, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
        scores = {labels[i]: float(probs[i]) for i in range(len(probs))}
        top_label = max(scores, key=scores.get)
        return {
            "verdict": top_label, "is_knock": _label_is_knock(top_label),
            "confidence": round(scores[top_label] * 100, 2),
            "scores": [{"label": k, "score": round(v, 6)} for k, v in scores.items()],
            "model": "cxlrd/revix-AST-engine-knock", "sample_rate": target_sr,
            "audio_file": audio.filename, "duration_s": duration_s,
        }
    except HTTPException: raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio analysis failed: {str(e)}")
    finally:
        for path in (raw_path, wav_path):
            if path and os.path.exists(path):
                try: os.remove(path)
                except: pass


# ── /detect-live ──────────────────────────────────────────────────────────────
@app.post("/detect-live")
async def detect_live(file: UploadFile = File(...)):
    global captured_frames, captured_defect_types, captured_all_defects
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    temp_path = os.path.join(UPLOAD_DIR, "temp_frame.jpg")
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    try:
        result = CLIENT.infer(temp_path, model_id=VEHICLE_MODEL_ID)
        predictions = result.get("predictions", [])
        frame = cv2.imread(temp_path)
        defects, new_defect = [], False
        for pred in predictions:
            cn = pred["class"].lower().strip()
            dn = CLASS_MAPPING.get(cn, cn.capitalize())
            cf = round(pred["confidence"] * 100, 1)
            defects.append({"class": dn, "confidence": cf})
            if cn not in captured_defect_types:
                captured_defect_types.add(cn); new_defect = True
        ann = frame.copy()
        for pred in predictions: draw_annotation(ann, pred)
        if new_defect and predictions:
            cp = os.path.join(STATIC_DIR, f"capture_{len(captured_frames)}.jpg")
            cv2.imwrite(cp, ann); captured_frames.append(cp)
            for pred in predictions:
                cn = pred["class"].lower().strip()
                captured_all_defects.append((CLASS_MAPPING.get(cn, cn.capitalize()), round(pred["confidence"] * 100, 1)))
        _, buf2 = cv2.imencode(".jpg", ann)
        return JSONResponse({
            "success": True, "defects": defects, "count": len(defects),
            "new_capture": new_defect and bool(predictions),
            "total_captures": len(captured_frames), "unique_defects": len(captured_defect_types),
            "annotated_frame": base64.b64encode(buf2).decode("utf-8"),
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")
    finally:
        if os.path.exists(temp_path): os.remove(temp_path)


# ── /finalize-live-detection ──────────────────────────────────────────────────
@app.post("/finalize-live-detection")
async def finalize_live_detection(
    vin: Optional[str] = Form(None), make: Optional[str] = Form(None),
    model: Optional[str] = Form(None), year: Optional[str] = Form(None),
    mileage: Optional[str] = Form(None), engine_verdict: Optional[str] = Form(None),
    engine_is_knock: Optional[str] = Form(None), engine_confidence: Optional[str] = Form(None),
    engine_duration: Optional[str] = Form(None),
):
    global captured_frames, captured_defect_types, captured_all_defects
    if not captured_frames:
        raise HTTPException(status_code=400, detail="No frames captured")
    all_defects = captured_all_defects.copy()
    vehicle_info = {"vin": vin or "Not Provided", "make": make or "Not Provided",
                    "model": model or "Not Provided", "year": year or "Not Provided", "mileage": mileage or "Not Provided"}
    engine_result = None
    if engine_verdict:
        engine_result = {"verdict": engine_verdict, "is_knock": (engine_is_knock or "").lower() == "true",
                         "confidence": float(engine_confidence or 0), "duration_s": float(engine_duration or 0)}
    ut = len(captured_defect_types)
    overall_status = "PASS" if ut == 0 else "ATTENTION" if ut <= 2 else "FAIL"
    ai_analysis = generate_ai_analysis(all_defects, vehicle_info, engine_result, overall_status)
    try:
        generate_report(all_defects, captured_frames, REPORT_PATH, vehicle_info,
                        engine_result=engine_result, ai_analysis=ai_analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report failed: {str(e)}")
    try:
        db.collection("inspections").add({
            "vehicle_info": vehicle_info, "defects": [{"part": d[0], "confidence": d[1]} for d in all_defects],
            "engine_result": engine_result, "overall_status": overall_status,
            "ai_analysis": ai_analysis, "createdAt": firestore.SERVER_TIMESTAMP,
        })
    except Exception as e:
        print(f"⚠️ Firebase save failed: {str(e)}")
    ann = [f"static/{os.path.basename(p)}" for p in captured_frames]
    captured_frames = []; captured_defect_types = set(); captured_all_defects = []
    return {"message": "Live detection report generated", "image_count": len(ann),
            "total_defects_detected": len(all_defects), "unique_defect_types": ut,
            "defects_detected": all_defects, "annotated_images": ann,
            "engine_result": engine_result, "ai_analysis": ai_analysis}


# ── /reset-live-detection ─────────────────────────────────────────────────────
@app.post("/reset-live-detection")
async def reset_live_detection():
    global captured_frames, captured_defect_types, captured_all_defects
    for p in captured_frames:
        if os.path.exists(p):
            try: os.remove(p)
            except: pass
    captured_frames = []; captured_defect_types = set(); captured_all_defects = []
    return {"message": "Live detection reset"}


# ── /inspect ──────────────────────────────────────────────────────────────────
@app.post("/inspect")
async def inspect_vehicle(
    files: List[UploadFile] = File(...),
    vin: Optional[str] = Form(None),
    make: Optional[str] = Form(None),
    model: Optional[str] = Form(None),
    year: Optional[str] = Form(None),
    mileage: Optional[str] = Form(None),
    inspection_type: Optional[str] = Form(None),        # ← ADD THIS LINE
    engine_verdict: Optional[str] = Form(None),
    engine_is_knock: Optional[str] = Form(None),
    engine_confidence: Optional[str] = Form(None),
    engine_duration: Optional[str] = Form(None),
):
    if not files: raise HTTPException(status_code=400, detail="At least one image file is required")
    for f in files:
        if not f.content_type or not f.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="All uploaded files must be images")
    all_defects = []; annotated_image_paths = []
    for idx, file in enumerate(files):
        input_path = os.path.join(UPLOAD_DIR, f"input_{idx}.jpg")
        with open(input_path, "wb") as buf: shutil.copyfileobj(file.file, buf)
        try:
            result = CLIENT.infer(input_path, model_id=VEHICLE_MODEL_ID)
        except Exception as e:
            if os.path.exists(input_path): os.remove(input_path)
            raise HTTPException(status_code=500, detail=f"Inference failed on image {idx + 1}: {str(e)}")
        image = cv2.imread(input_path)
        if image is None:
            os.remove(input_path)
            raise HTTPException(status_code=500, detail=f"Failed to load image {idx + 1}")
        predictions = result.get("predictions", [])
        for pred in predictions:
            cn = pred["class"].lower().strip()
            all_defects.append((CLASS_MAPPING.get(cn, cn.capitalize()), round(pred["confidence"] * 100, 1)))
        for pred in predictions: draw_annotation(image, pred)
        if predictions:
            ap = os.path.join(STATIC_DIR, f"annotated_{idx}.jpg")
            cv2.imwrite(ap, image); annotated_image_paths.append(f"static/annotated_{idx}.jpg")
        os.remove(input_path)
    vehicle_info = {"vin": vin or "Not Provided", "make": make or "Not Provided",
                    "model": model or "Not Provided", "year": year or "Not Provided", "mileage": mileage or "Not Provided"}
    engine_result = None
    if engine_verdict and engine_verdict.strip():
        engine_result = {"verdict": engine_verdict.strip(), "is_knock": (engine_is_knock or "").strip().lower() == "true",
                         "confidence": float(engine_confidence or 0), "duration_s": float(engine_duration or 0)}
    unique_defect_types = len({c.lower() for c, _ in all_defects})
    try:
        overall_status = "PASS" if unique_defect_types == 0 else "ATTENTION" if unique_defect_types <= 2 else "FAIL"
        db.collection("inspections").add({
            "vehicle_info": vehicle_info, "defects": [{"part": d[0], "confidence": d[1]} for d in all_defects],
            "engine_result": engine_result, "overall_status": overall_status, "createdAt": firestore.SERVER_TIMESTAMP,
        })
    except Exception as e:
        print(f"⚠️ Firebase save failed: {str(e)}")
    return {"message": "Inspection complete", "image_count": len(files),
            "total_defects_detected": len(all_defects), "unique_defect_types": unique_defect_types,
            "defects_detected": all_defects, "annotated_images": annotated_image_paths, "engine_result": engine_result}


# ── /generate-report ──────────────────────────────────────────────────────────
class _EngineResult(BaseModel):
    verdict: str = ""; is_knock: bool = False; confidence: float = 0.0; duration_s: float = 0.0

class _VehicleInfo(BaseModel):
    vin: str = ""; make: str = ""; model: str = ""; year: str = ""; mileage: str = ""

class GenerateReportRequest(BaseModel):
    defects_detected: list; annotated_images: List[str]; image_count: int = 0
    unique_defect_types: int = 0; vehicle_info: _VehicleInfo = _VehicleInfo()
    engine_result: Optional[_EngineResult] = None

@app.post("/generate-report")
async def generate_report_from_data(req: GenerateReportRequest):
    defects_normalised = []
    for d in req.defects_detected:
        if isinstance(d, (list, tuple)) and len(d) >= 2: defects_normalised.append((str(d[0]), float(d[1])))
        elif isinstance(d, dict):
            name = d.get("label") or d.get("class") or d.get("name") or "Unknown"
            defects_normalised.append((name, float(d.get("confidence", 0))))
    image_paths = []
    for rel in req.annotated_images:
        basename = os.path.basename(rel.lstrip("/"))
        candidate = os.path.join(STATIC_DIR, basename)
        if os.path.isfile(candidate): image_paths.append(candidate)
        elif os.path.isfile(rel.lstrip("/")): image_paths.append(rel.lstrip("/"))
    vehicle_info = req.vehicle_info.dict()
    engine_result = req.engine_result.dict() if req.engine_result else None
    unique_types = len({d[0].lower() for d in defects_normalised})
    overall_status = "PASS" if unique_types == 0 else "ATTENTION" if unique_types <= 2 else "FAIL"
    ai_analysis = generate_ai_analysis(defects=defects_normalised, vehicle_info=vehicle_info,
                                        engine_result=engine_result, overall_status=overall_status)
    try:
        generate_report(defects=defects_normalised, image_paths=image_paths, output_path=REPORT_PATH,
                        vehicle_info=vehicle_info, engine_result=engine_result, ai_analysis=ai_analysis)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(e)}")
    return {"message": "Report generated with AI analysis", "image_count": req.image_count,
            "total_defects_detected": len(defects_normalised), "unique_defect_types": unique_types,
            "defects_detected": [[d[0], d[1]] for d in defects_normalised],
            "annotated_images": req.annotated_images, "engine_result": engine_result, "ai_analysis": ai_analysis}


# ── /save-claim (NEW - Insurance) ─────────────────────────────────────────────
class InsuranceClaimRequest(BaseModel):
    ownerId: str = ""; ownerName: str = ""; ownerEmail: str = ""; ownerPhone: str = ""
    insurerId: str = ""; insurerName: str = ""; policyNumber: str = ""
    vehiclePlate: str = ""; vehicleMake: str = ""; vehicleModel: str = ""; vehicleYear: str = ""
    incidentDate: str = ""; description: str = ""; status: str = "pending"

@app.post("/save-claim")
async def save_claim(req: InsuranceClaimRequest):
    try:
        _, doc_ref = db.collection("claims").add({
            **req.dict(), "createdAt": firestore.SERVER_TIMESTAMP, "updatedAt": firestore.SERVER_TIMESTAMP,
        })
        return {"success": True, "claimId": doc_ref.id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save claim: {str(e)}")


# ── /get-insurance-companies (NEW) ────────────────────────────────────────────
@app.get("/get-insurance-companies")
async def get_insurance_companies():
    try:
        docs = db.collection("users").where("role", "==", "insurance").stream()
        return {"companies": [{"uid": d.id, **{k: v for k, v in d.to_dict().items() if k in ("companyName", "email")}} for d in docs]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Static routes ─────────────────────────────────────────────────────────────
@app.get("/")
async def read_root():
    frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend", "index.html")
    if os.path.exists(frontend_path): return FileResponse(frontend_path)
    return JSONResponse(status_code=404, content={"error": "index.html not found"})

@app.get("/report")
def get_report():
    if not os.path.exists(REPORT_PATH):
        raise HTTPException(status_code=404, detail="Report not generated yet.")
    return FileResponse(REPORT_PATH, media_type="application/pdf",
                        headers={"Content-Disposition": "inline; filename=AI_Vehicle_Inspection_Report.pdf"})

@app.get("/get-captured-images")
async def get_captured_images():
    return {"images": [{"path": f"static/{os.path.basename(p)}", "defects": [], "approved": True} for p in captured_frames]}

@app.get("/test-firebase")
def test_firebase():
    db.collection("test").document("demo").set({"message": "Hello from FastAPI!", "status": "connected"})
    return {"message": "Data written to Firebase!"}

# ── NEW: Extract UAE Mulkiya (Vehicle License) ─────────────────────────────
@app.post("/extract-mulkiya")
async def extract_mulkiya(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files allowed")

    # Save temporarily
    temp_path = os.path.join(UPLOAD_DIR, f"mulkiya_{int(time.time())}.jpg")
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        import easyocr
        reader = easyocr.Reader(['en', 'ar'], gpu=False)  # English + Arabic
        results = reader.readtext(temp_path, detail=0, paragraph=True)

        full_text = " ".join(results).upper()

        # Smart parsing for UAE Mulkiya
        data = {
            "traffic_plate": None,
            "tc_no": None,           # T.C. No. = Traffic Code (often used as VIN)
            "place_of_issue": None,
            "owner_name": None,
            "nationality": None,
            "reg_date": None,
            "exp_date": None,
        }

        # Traffic Plate No.
        plate_match = re.search(r'TRAFFIC PLATE NO\.\s*[:.]?\s*([0-9\s/]+)', full_text)
        if plate_match:
            data["traffic_plate"] = plate_match.group(1).strip()

        # T.C. No.
        tc_match = re.search(r'T\.?\s*C\.?\s*NO\.\s*[:.]?\s*([0-9]+)', full_text)
        if tc_match:
            data["tc_no"] = tc_match.group(1).strip()

        # Owner Name
        owner_match = re.search(r'OWNER\s*[:.]?\s*([A-Z\s]+?)(?=NATIONALITY|INDIAN|EXP|REG)', full_text)
        if owner_match:
            data["owner_name"] = owner_match.group(1).strip().title()

        # Nationality
        nat_match = re.search(r'NATIONALITY\s*[:.]?\s*(INDIAN|PAKISTANI|EGYPTIAN|FILIPINO|ARAB|UAE)', full_text)
        if nat_match:
            data["nationality"] = nat_match.group(1).title()

        # Reg. Date
        reg_match = re.search(r'REG\.\s*DATE\s*[:.]?\s*([0-9]{2}-[A-Z]{3}-[0-9]{2,4})', full_text)
        if reg_match:
            data["reg_date"] = reg_match.group(1)

        # Place of Issue
        if "SHARJAH" in full_text:
            data["place_of_issue"] = "Sharjah"
        elif "DUBAI" in full_text:
            data["place_of_issue"] = "Dubai"
        elif "ABU DHABI" in full_text:
            data["place_of_issue"] = "Abu Dhabi"

        return {
            "success": True,
            "extracted": data,
            "raw_text": full_text
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR failed: {str(e)}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")