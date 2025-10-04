import re
import io
import uuid
import torch
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
from typing import Dict
from pydantic import BaseModel
from pathlib import Path

# Import our database module
import database

# --- 1. Configuration and Setup ---
# The platform provides a persistent '/data' directory
IMAGE_STORAGE_PATH = Path("data/uploaded_images")

app = FastAPI(title="Persistent Invoice Extractor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def on_startup():
    """Create directories and initialize the database on server startup."""
    IMAGE_STORAGE_PATH.mkdir(exist_ok=True)
    database.init_db()

# --- 2. Pydantic Models ---
class InvoiceData(BaseModel):
    company_name: str | None = None
    invoice_number: str | None = None
    date: str | None = None
    total: str | None = None

class CorrectionPayload(BaseModel):
    image_id: str
    ai_prediction: InvoiceData
    user_correction: InvoiceData

# --- 3. AI Model and Pre-processing ---
print("Loading Florence-2-base model...")
model_id = 'microsoft/Florence-2-base'
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    trust_remote_code=True, 
    attn_implementation="eager" # Important for CPU
)
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
device = torch.device("cpu") # Forcing CPU for the free tier
model.to(device)
print(f"✅ Model loaded and ready to run on CPU.")

def deskew(image):
    if cv2.countNonZero(image) == 0: return image
    coords = np.column_stack(np.where(image > 0))
    if len(coords) < 10: return image
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45: angle = -(90 + angle)
    else: angle = -angle
    if abs(angle) < 0.1: return image
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos, sin = np.abs(M[0, 0]), np.abs(M[0, 1])
    new_w, new_h = int((h * sin) + (w * cos)), int((h * cos) + (w * sin))
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    return cv2.warpAffine(image, M, (new_w, new_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

def preprocess_image_advanced(image_bytes: bytes) -> Image.Image:
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    original_for_warp = img.copy()
    gray_for_contours = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray_for_contours, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4 and cv2.contourArea(approx) > (img.shape[0] * img.shape[1] * 0.2):
                pts, rect = approx.reshape(4, 2), np.zeros((4, 2), dtype="float32")
                s = pts.sum(axis=1); rect[0], rect[2] = pts[np.argmin(s)], pts[np.argmax(s)]
                diff = np.diff(pts, axis=1); rect[1], rect[3] = pts[np.argmin(diff)], pts[np.argmax(diff)]
                (tl, tr, br, bl) = rect
                widthA, widthB = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2)), np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
                maxWidth = max(int(widthA), int(widthB))
                heightA, heightB = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2)), np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
                maxHeight = max(int(heightA), int(heightB))
                dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
                M = cv2.getPerspectiveTransform(rect, dst)
                img = cv2.warpPerspective(original_for_warp, M, (maxWidth, maxHeight))
                break
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dilated = cv2.dilate(gray, np.ones((7, 7), np.uint8))
    bg_img = cv2.medianBlur(dilated, 21)
    diff = 255 - cv2.absdiff(gray, bg_img)
    norm_img = cv2.normalize(diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    upscaled = cv2.resize(norm_img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    denoised = cv2.fastNlMeansDenoising(upscaled, None, h=10, templateWindowSize=7, searchWindowSize=21)
    thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    final_image = cv2.bitwise_not(deskew(thresh))
    return Image.fromarray(final_image).convert("RGB")

# --- 4. OCR and Extraction Functions ---
def get_ocr_text(image: Image.Image) -> str:
    prompt = "<OCR>"
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    generated_ids = model.generate(input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"], max_new_tokens=4096)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

def extract_details_smarter(text: str) -> Dict[str, str]:
    text = text.replace('\n', ' ').strip()
    patterns = { "company_name": [r"([\w\s.&]+(?:Co\.|STORE|Pvt\.\s*Ltd\.))"], "invoice_number": [r"Invoice No\s*:?\s*([\w\/-]+)"], "date": [r"Date\s*:?\s*(\d{2}\/\d{2}\/\d{4})", r"Dated\s*:?\s*(\d{1,2}-[A-Za-z]{3}-\d{2})"], "total": [r"Grand Total\s*:*\s*\S?\s*([\d,]+\.\d{2})", r"Net Amount\s*:?\s*([\d,]+\.\d{2})"] }
    data = {}
    for key, pattern_list in patterns.items():
        found = False
        for pattern in pattern_list:
            if match := re.search(pattern, text, re.IGNORECASE):
                data[key], found = match.group(1).strip(), True
                break
        if not found: data[key] = None
    return data

# --- 5. API Endpoints ---
@app.post("/process_invoice/")
async def process_invoice(file: UploadFile = File(...)):
    contents = await file.read()
    image_id = f"{uuid.uuid4()}.jpg"
    save_path = IMAGE_STORAGE_PATH / image_id
    with open(save_path, "wb") as f: f.write(contents)
    cleaned_image = preprocess_image_advanced(contents)
    ocr_text = get_ocr_text(cleaned_image)
    extracted_data = extract_details_smarter(ocr_text)
    return {"image_id": image_id, "extracted_data": extracted_data, "full_ocr_text": ocr_text}

@app.post("/log_correction/")
async def log_user_correction(payload: CorrectionPayload):
    try:
        database.log_correction(image_id=payload.image_id, ai_prediction=payload.ai_prediction.dict(), user_correction=payload.user_correction.dict())
        return {"status": "success", "message": "Correction logged."}
    except Exception as e:
        return {"status": "error", "message": str(e)}