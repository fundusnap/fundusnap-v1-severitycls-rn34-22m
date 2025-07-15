from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from ultralytics import YOLO
import os
import shutil
from typing import List, Dict, Any, Optional
import logging
import io
import numpy as np
import cv2


# ?  SETUP
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Fundusnap-AI API",
    description="API for detecting retinal artefacts",
    version="1.0"
)

# ?  MODEL PATHS
MODEL_PATH_FUNDUS_ARTIFACTS = "models/fundus_artifacts.pt"

models = {}

try:
    if os.path.exists(MODEL_PATH_FUNDUS_ARTIFACTS):
        models["fundus_artifacts"] = YOLO(MODEL_PATH_FUNDUS_ARTIFACTS)
        logger.info(f"Fundus Artifacts (Detection) model loaded from: {MODEL_PATH_FUNDUS_ARTIFACTS}")
    else:
        logger.warning(f"Fundus Artifacts model not found at: {MODEL_PATH_FUNDUS_ARTIFACTS}")
except Exception as e:
    logger.error(f"Failed to load Fundus Artifacts model: {e}")


# ?  RESPONSE MODELS

# ? For Object Detection (Fundus Artifacts)
class DetectionBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int

class DetectionResult(BaseModel):
    class_name: str
    confidence: float
    box: DetectionBox


class DetectionResponse(BaseModel):
    filename: str
    model_used: str
    detections: List[DetectionResult]


# ?  HELPER funcs

# ? For drawing detection boxes
def draw_detections(image: np.ndarray, detections: List[DetectionResult]) -> np.ndarray:
    for detection in detections:
        box = detection.box
        label = f"{detection.class_name}: {detection.confidence:.2f}"
        cv2.rectangle(image, (box.x1, box.y1), (box.x2, box.y2), color=(0, 0, 255), thickness=2)
        cv2.putText(image, label, (box.x1, box.y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return image



# ?  INFERENCE PROCESSING

# ? For Object Detection models
def process_detection_inference(model: YOLO, image: np.ndarray) -> List[DetectionResult]:
    results = model(image)
    detections = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            detections.append(DetectionResult(
                class_name=model.names[cls_id],
                confidence=round(conf, 2),
                box=DetectionBox(x1=x1, y1=y1, x2=x2, y2=y2)
            ))
    return detections




# ?  API ENDPOINTS

# ? Fundus Artifacts (DETECTION)
@app.post("/inspect/fundus-artifacts/", response_model=DetectionResponse, tags=["JSON Inspection"])
async def inspect_fundus_artifacts(file: UploadFile = File(...)):
    model_key = "fundus_artifacts"
    if model_key not in models:
        raise HTTPException(status_code=500, detail=f"Model '{model_key}' is not available.")
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    detections = process_detection_inference(models[model_key], img)
    return DetectionResponse(filename=file.filename, model_used=model_key, detections=detections)

# ? Fundus Artifacts (VISUALIZATION)
@app.post("/visualize/fundus-artifacts/", summary="Visualize Fundus Artifacts Detections", tags=["Visual Inspection"])
async def visualize_fundus_artifacts(file: UploadFile = File(...)):
    model_key = "fundus_artifacts"
    if model_key not in models:
        raise HTTPException(status_code=500, detail=f"Model '{model_key}' is not available.")
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    detections = process_detection_inference(models[model_key], img)
    img_with_detections = draw_detections(img, detections)
    is_success, buffer = cv2.imencode(".jpg", img_with_detections)
    if not is_success:
        raise HTTPException(status_code=500, detail="Failed to encode visualized image.")
    return StreamingResponse(io.BytesIO(buffer), media_type="image/jpeg")



# ? Root/Health Check Endpoint
@app.get("/", summary="Health Check", tags=["General"])
def read_root():
    return {
        "status": "Fundusnap AI API is running!",
        "loaded_models": list(models.keys()),
    }
