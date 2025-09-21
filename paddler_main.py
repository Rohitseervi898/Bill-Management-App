import cv2
import numpy as np
import sys
import os
import json
from pathlib import Path
from datetime import datetime
from paddleocr import PaddleOCR

# Initialize PaddleOCR
ocr_model = PaddleOCR(use_textline_orientation=True, lang='en')  # updated

# ---------- IMAGE PREPROCESSING ----------
def preprocess_image(image_path):
    """
    Preprocess image before OCR, but keep text intact.
    """
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError(f"Image not found or unreadable: {image_path}")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Light denoising
    gray = cv2.fastNlMeansDenoising(gray, None, 15, 7, 21)

    # Instead of hard threshold, try OTSU
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return th


# ---------- PADDLE OCR EXTRACTION ----------
def extract_text_paddle(image_path):
    """
    Run PaddleOCR and return raw result (coords + text + conf).
    """
    print(f"🔹 Running PaddleOCR on {os.path.basename(image_path)} ...")

    preprocessed_img = preprocess_image(image_path)
    temp_path = "temp_processed.png"
    cv2.imwrite(temp_path, preprocessed_img)

    # use `predict` instead of `ocr` to avoid DeprecationWarning
    result = ocr_model.ocr(temp_path)

    return result  # return raw OCR structure directly


# ---------- CREATE LABEL STUDIO JSON ----------
def create_labelstudio_json(image_path, extracted_data, output_dir="paddle_output"):
    """
    Save raw OCR result (not cleaned) in JSON format.
    """
    os.makedirs(output_dir, exist_ok=True)

    file_upload_id = int(datetime.now().timestamp())

    json_data = {
        "data": {
            "filename": os.path.basename(image_path),
            "ocr_raw": extracted_data  # keep full OCR output
        },
        "file_upload_id": file_upload_id
    }

    json_filename = os.path.splitext(os.path.basename(image_path))[0] + ".json"
    json_path = os.path.join(output_dir, json_filename)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    print(f"✅ PaddleOCR JSON saved: {json_path}")
    return json_path


# ---------- PROCESS MULTIPLE IMAGES ----------
def process_invoices_with_paddle(input_path, output_dir="paddle_output"):
    """
    Process multiple invoices using PaddleOCR.
    """
    input_path = Path(input_path)

    # If single image provided
    if input_path.is_file():
        files = [input_path]
    else:
        # Allow only image formats
        files = [f for f in input_path.glob("*") if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]

    if not files:
        print("❌ No images found to process!")
        return

    print(f"📂 Found {len(files)} invoice(s) to process with PaddleOCR...\n")

    for img_file in files:
        try:
            text = extract_text_paddle(str(img_file))
            create_labelstudio_json(str(img_file), text, output_dir)
        except Exception as e:
            print(f"❌ Error processing {img_file.name}: {e}")


# ---------- CLI ENTRY POINT ----------
def main():
    """
    CLI Usage:
    python main.py <path_to_image_or_folder> [output_folder]
    """
    if len(sys.argv) < 2:
        print("Usage: python main.py <image_or_folder_path> [output_json_folder]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_folder = sys.argv[2] if len(sys.argv) > 2 else "paddle_output"

    process_invoices_with_paddle(input_path, output_folder)


if __name__ == "__main__":
    main()
