import cv2
import pytesseract
import numpy as np
import re
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

# Path to Tesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

app = FastAPI(title="Advanced Invoice OCR Service")


# -------------------- IMAGE PREPROCESSING --------------------
def deskew(image: np.ndarray) -> np.ndarray:
    """Correct skew using the minimum area rectangle method."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    gray = cv2.bitwise_not(gray)
    coords = np.column_stack(np.where(gray > 0))
    if len(coords) == 0:
        return image

    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle

    (h, w) = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Enhance image for OCR by removing background and improving text clarity."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Reduce background noise
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    # Adaptive threshold to isolate text
    binary = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 15, 8
    )

    # Morphological closing to strengthen characters
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Deskew image
    final = deskew(morph)

    return final


# -------------------- TEXT CLEANING --------------------
def clean_text(text: str) -> str:
    """Clean OCR text by removing unwanted symbols and formatting."""
    text = re.sub(r'[^\x20-\x7E\n]', '', text)  # Remove non-ASCII characters
    text = re.sub(r'\n+', '\n', text)           # Collapse multiple newlines
    text = re.sub(r'[|]+', '', text)            # Remove stray pipes
    return text.strip()


# -------------------- STRUCTURED DATA EXTRACTION --------------------
def extract_invoice_details(text: str) -> dict:
    """Extract key fields like invoice number, date, and total."""
    details = {}

    # Invoice Number
    match_invoice = re.search(r'Invoice\s*No[:\-]?\s*([A-Z0-9\-]+)', text, re.IGNORECASE)
    details["invoice_number"] = match_invoice.group(1) if match_invoice else None

    # Invoice Date
    match_date = re.search(r'Invoice\s*Date[:\-]?\s*(\d{2}/\d{2}/\d{4})', text, re.IGNORECASE)
    details["invoice_date"] = match_date.group(1) if match_date else None

    # Party Name
    match_party = re.search(r'Party[:\-]?\s*([A-Za-z\s]+)', text, re.IGNORECASE)
    details["party_name"] = match_party.group(1).strip() if match_party else None

    # Net Amount
    match_total = re.search(r'Net\s*Payable[:\-]?\s*(\d+\.\d{2}|\d+)', text, re.IGNORECASE)
    details["net_payable"] = match_total.group(1) if match_total else None

    return details


# -------------------- FASTAPI ENDPOINT --------------------
@app.post("/upload/")
async def ocr_and_extract(file: UploadFile = File(...)):
    image_bytes = await file.read()

    # Preprocess image
    processed_image = preprocess_image(image_bytes)

    # OCR configuration
    config = r'--oem 3 --psm 6'
    raw_text = pytesseract.image_to_string(processed_image, lang='eng', config=config)

    # Clean extracted text
    cleaned_text = clean_text(raw_text)

    # Extract structured data
    invoice_data = extract_invoice_details(cleaned_text)

    return JSONResponse(content={
        "filename": file.filename,
        "cleaned_text": cleaned_text,
        "invoice_data": invoice_data
    })
