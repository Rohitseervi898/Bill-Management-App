import cv2
import pytesseract
from PIL import Image
import numpy as np
import sys
import os
import json
from datetime import datetime
from pathlib import Path

# If using Windows, set Tesseract executable path
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# ---------- IMAGE PREPROCESSING ----------
def preprocess_image(image_path):
    """
    Preprocess the image to improve OCR accuracy.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        raise ValueError(f"Image not found or unreadable: {image_path}")

    # 1. Noise Removal
    img = cv2.fastNlMeansDenoising(img, None, 30, 7, 21)

    # 2. Adaptive Thresholding
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 31, 2)

    # 3. Morphological Cleaning
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    # cv2.imwrite("debug_prevg.png", img)

    return img


# ---------- OCR TEXT EXTRACTION ----------
def extract_text(image_path):
    """
    Extract raw text from the invoice image using Tesseract OCR.
    """
    processed_img = preprocess_image(image_path)

    # OCR Configuration
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(processed_img, config=custom_config)

    # Clean the text
    clean_text = "\n".join([line.strip() for line in text.splitlines() if line.strip()])
    return clean_text


# ---------- CREATE LABEL STUDIO JSON ----------
def create_labelstudio_json(image_path, extracted_text, output_dir="output_json"):
    """
    Create a JSON file compatible with Label Studio for annotation.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Generate unique file_upload_id
    file_upload_id = int(datetime.now().timestamp())

    json_data = {
        "data": {
            "filename": os.path.basename(image_path),
            "text": extracted_text
        },
        "file_upload_id": file_upload_id
    }

    # Save JSON with the same name as the image
    json_filename = os.path.splitext(os.path.basename(image_path))[0] + ".json"
    json_path = os.path.join(output_dir, json_filename)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    print(f"✅ JSON saved for {os.path.basename(image_path)} → {json_path}")
    return json_path


# ---------- PROCESS MULTIPLE IMAGES ----------
def process_invoices(input_path, output_dir="output_json"):
    """
    Process multiple invoice images from a folder or single image.
    """
    input_path = Path(input_path)

    # If single file provided
    if input_path.is_file():
        files = [input_path]
    else:
        # Support common image extensions
        files = [f for f in input_path.glob("*") if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]

    if not files:
        print("❌ No images found to process!")
        return

    print(f"📂 Found {len(files)} invoice(s) to process...\n")

    for img_file in files:
        try:
            print(f"🔹 Processing: {img_file.name}")
            text = extract_text(str(img_file))
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
    output_folder = sys.argv[2] if len(sys.argv) > 2 else "output_json"

    process_invoices(input_path, output_folder)


if __name__ == "__main__":
    main()
