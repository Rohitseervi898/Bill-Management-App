import cv2
import pytesseract
import numpy as np
import sys
import os
import json
from pathlib import Path
from datetime import datetime

# If using Windows, set Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def deskew(image):
    """
    Detects and corrects the skew of a binary image without cropping.
    """
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    if abs(angle) < 0.1:
        return image
    (h, w) = image.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    deskewed = cv2.warpAffine(image, M, (new_w, new_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return deskewed


def preprocess_image(image_path, debug=False):
    """
    An advanced pipeline that handles uneven illumination and shadows.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- 1. Illumination Normalization ---
    # This is the key step to remove shadows
    dilated_img = cv2.dilate(gray, np.ones((7, 7), np.uint8))
    bg_img = cv2.medianBlur(dilated_img, 21)
    diff_img = 255 - cv2.absdiff(gray, bg_img)
    norm_img = diff_img.copy()
    cv2.normalize(diff_img, norm_img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    
    # --- 2. Thresholding and Deskewing ---
    thresh = cv2.threshold(norm_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    deskewed = deskew(thresh)
    final_image = cv2.bitwise_not(deskewed)

    if debug:
        cv2.imwrite("debug_normalized_illumination.png", norm_img)
        cv2.imwrite("debug_final_ocr_input.png", final_image)

    return final_image


# In your main.py, replace only this function.

# In your main.py, please replace only the extract_text function with this stable version.

def extract_text(image_path):
    """
    Extracts text using the final, stable pipeline and Tesseract configuration.
    """
    processed_img = preprocess_image(image_path, debug=True)
    
    # --- CORRECTED: Reverted to the most robust and reliable PSM mode ---
    custom_config = r'--oem 3 --psm 3'
    
    text = pytesseract.image_to_string(processed_img, config=custom_config)
    
    # Clean the text
    clean_text = "\n".join([line.strip() for line in text.splitlines() if line.strip()])
    return clean_text


def create_labelstudio_json(image_path, extracted_text, output_dir="output_json1"):
    os.makedirs(output_dir, exist_ok=True)
    file_upload_id = int(datetime.now().timestamp())
    json_data = {
        "data": { "filename": os.path.basename(image_path), "text": extracted_text },
        "file_upload_id": file_upload_id
    }
    json_filename = os.path.splitext(os.path.basename(image_path))[0] + ".json"
    json_path = os.path.join(output_dir, json_filename)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"✅ JSON saved: {json_path}")


def process_invoices(input_path, output_dir="output_json1"):
    input_path = Path(input_path)
    if input_path.is_file(): files = [input_path]
    else: files = [f for f in input_path.glob("*") if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]

    if not files:
        print("❌ No images found!")
        return

    print(f"📂 Found {len(files)} invoice(s) to process...\n")
    for img_file in files:
        try:
            print(f"🔹 Processing: {img_file.name}")
            text = extract_text(str(img_file))
            create_labelstudio_json(str(img_file), text, output_dir)
        except Exception as e:
            print(f"❌ Error processing {img_file.name}: {e}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <image_or_folder_path> [output_folder]")
        sys.exit(1)
    input_path = sys.argv[1]
    output_folder = sys.argv[2] if len(sys.argv) > 2 else "output_json1"
    process_invoices(input_path, output_folder)

if __name__ == "__main__":
    main()