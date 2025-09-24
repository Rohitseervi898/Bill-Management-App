import cv2
import pytesseract
import numpy as np
import sys
import os
import json
from pathlib import Path
from datetime import datetime

# If using Windows, set Tesseract executable path
# Make sure to update this if your Tesseract installation is in a different location
try:
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
except Exception:
    print("Warning: Tesseract is not installed or not in the specified path.")


def deskew(image):
    """
    Detects and corrects the skew of a binary image without cropping.
    This function expects a binary image (white text on a black background).
    """
    # Failsafe for blank images
    if cv2.countNonZero(image) == 0:
        return image

    coords = np.column_stack(np.where(image > 0))
    
    # Failsafe for images with very few points
    if len(coords) < 10:
        return image
        
    angle = cv2.minAreaRect(coords)[-1]

    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    
    if abs(angle) < 0.1:
        return image

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    
    deskewed = cv2.warpAffine(
        image, M, (new_w, new_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    return deskewed


def preprocess_image(image_path, debug=False):
    """
    The definitive, multi-step pipeline to prepare any invoice image for Tesseract.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found or unreadable: {image_path}")

    # --- Step 1: Perspective Correction (with Failsafe) ---
    original_for_warp = img.copy()
    gray_for_contours = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray_for_contours, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
        screen_contour = None
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                screen_contour = approx
                break

        if screen_contour is not None and cv2.contourArea(screen_contour) > (img.shape[0] * img.shape[1] * 0.2):
            pts = screen_contour.reshape(4, 2)
            rect = np.zeros((4, 2), dtype="float32")
            s = pts.sum(axis=1); rect[0] = pts[np.argmin(s)]; rect[2] = pts[np.argmax(s)]
            diff = np.diff(pts, axis=1); rect[1] = pts[np.argmin(diff)]; rect[3] = pts[np.argmax(diff)]
            (tl, tr, br, bl) = rect
            widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
            widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
            maxWidth = max(int(widthA), int(widthB))
            heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
            heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
            maxHeight = max(int(heightA), int(heightB))
            dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
            M = cv2.getPerspectiveTransform(rect, dst)
            img = cv2.warpPerspective(original_for_warp, M, (maxWidth, maxHeight))
    
    # --- Step 2: Illumination Normalization ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dilated = cv2.dilate(gray, np.ones((7, 7), np.uint8))
    bg_img = cv2.medianBlur(dilated, 21)
    diff = 255 - cv2.absdiff(gray, bg_img)
    norm_img = cv2.normalize(diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

    # --- Step 3: Upscaling and Denoising ---
    upscaled = cv2.resize(norm_img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    denoised = cv2.fastNlMeansDenoising(upscaled, None, h=10, templateWindowSize=7, searchWindowSize=21)

    # --- Step 4: Binarization ---
    thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    
    # --- Step 5: Deskewing ---
    deskewed = deskew(thresh)
    final_image = cv2.bitwise_not(deskewed)

    if debug:
        # Create a debug folder if it doesn't exist
        debug_folder = Path("debug_images")
        debug_folder.mkdir(exist_ok=True)
        # Get the original image filename to create a unique debug filename
        original_filename = Path(image_path).stem
        cv2.imwrite(str(debug_folder / f"{original_filename}_final.png"), final_image)

    return final_image


def extract_text(image_path):
    """
    Extracts text using the definitive preprocessing pipeline.
    """
    processed_img = preprocess_image(image_path, debug=True)
    custom_config = r'--oem 3 --psm 3'
    text = pytesseract.image_to_string(processed_img, config=custom_config)
    clean_text = "\n".join([line.strip() for line in text.splitlines() if line.strip()])
    return clean_text


def create_labelstudio_json(image_path, extracted_text, output_dir="output_json"):
    """
    Creates a JSON file compatible with Label Studio.
    """
    os.makedirs(output_dir, exist_ok=True)
    file_upload_id = int(datetime.now().timestamp())
    json_data = {
        "data": {"filename": os.path.basename(image_path), "text": extracted_text},
        "file_upload_id": file_upload_id
    }
    json_filename = Path(image_path).stem + ".json"
    json_path = Path(output_dir) / json_filename
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"✅ JSON saved: {json_path}")


def process_invoices(input_path, output_dir="output_json"):
    """
    Processes a single image or all images in a folder.
    """
    input_path = Path(input_path)
    if input_path.is_file():
        files = [input_path]
    else:
        files = [f for f in input_path.glob("*") if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]

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
        print("Usage: python tesseract_main.py <image_or_folder_path> [output_json_folder]")
        sys.exit(1)
    input_path = sys.argv[1]
    output_folder = sys.argv[2] if len(sys.argv) > 2 else "output_json"
    process_invoices(input_path, output_folder)


if __name__ == "__main__":
    main()

