import os
import json
import cv2
import pytesseract
import numpy as np
import re
from pdf2image import convert_from_path


# ================================
# SET TESSERACT PATH
# ================================
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# ================================
# IMAGE PREPROCESSING
# ================================
def preprocess_image(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    thresh = cv2.threshold(
        blur, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]

    return thresh


# ================================
# PDF → IMAGES
# ================================
def pdf_to_images(path):

    pages = convert_from_path(path, dpi=300)

    images = []

    for page in pages:
        img = np.array(page)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        images.append(img)

    return images


# ================================
# OCR
# ================================
def ocr_image(img):

    processed = preprocess_image(img)

    text = pytesseract.image_to_string(processed)

    return text


# ================================
# SMART EXTRACTION
# ================================
def extract_fields(text):

    data = {
        "invoice_number": "",
        "date": "",
        "seller": "",
        "buyer": "",
        "total_amount": "",
        "tax": ""
    }

    patterns = {
        "invoice_number": r"invoice\s*number\s*([A-Z0-9\-]+)",
        "date": r"(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},\s+\d{4}",
        "seller": r"from:\s*(.*)",
        "buyer": r"to:\s*(.*)",
        "total_amount": r"total\s*\|?\s*\$?([\d\.]+)",
        "tax": r"tax\s*\|?\s*\$?([\d\.]+)"
    }

    for key, pattern in patterns.items():

        match = re.search(pattern, text, re.IGNORECASE)

        if match:
            data[key] = match.group(1).strip()

    return data


# ================================
# MAIN
# ================================
def main():

    input_folder = "../input"
    output_folder = "../output"

    files = os.listdir(input_folder)

    if not files:
        print("❌ No input file found")
        return

    file_path = os.path.join(input_folder, files[0])

    print("📂 Input file:", files[0])

    full_text = ""


    # -------- PDF --------
    if file_path.lower().endswith(".pdf"):

        print("📄 Processing PDF...")

        images = pdf_to_images(file_path)

        for i, img in enumerate(images):
            print(f"🔍 OCR page {i+1}")
            text = ocr_image(img)
            full_text += text + "\n"


    # -------- IMAGE --------
    elif file_path.lower().endswith((".jpg", ".png", ".jpeg")):

        print("📷 Processing image...")

        img = cv2.imread(file_path)

        if img is None:
            print("❌ Cannot read image")
            return

        full_text = ocr_image(img)

    else:
        print("❌ Unsupported file type")
        return


    print("\n📄 OCR TEXT:\n")
    print(full_text)


    print("\n🔍 Extracting fields...")

    final_data = extract_fields(full_text)


    # ================================
    # SAVE JSON
    # ================================
    output_path = os.path.join(output_folder, "result_ai.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_data, f, indent=4)


    print("\n✅ Extraction Complete")
    print("📁 Saved:", output_path)

    print("\n📄 FINAL RESULT:\n")
    print(json.dumps(final_data, indent=4))


# ================================
if __name__ == "__main__":
    main()
