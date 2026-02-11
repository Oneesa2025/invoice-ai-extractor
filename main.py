import os
import re
import json
import pdfplumber
from transformers import pipeline


# Load local AI model
print("⏳ Loading AI model...")
extractor = pipeline("text2text-generation", model="google/flan-t5-base")
print("✅ Model loaded")


# -------- PDF READER --------
def read_pdf(path):
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text


# -------- RULE EXTRACTION --------
def rule_extract(text):

    data = {
        "invoice_number": "",
        "date": "",
        "seller": "",
        "buyer": "",
        "total_amount": "",
        "tax": ""
    }

    patterns = {
        "invoice_number": r"invoice\s*(no|number)?\s*[:\-]?\s*(\S+)",
        "date": r"date\s*[:\-]?\s*(.+)",
        "seller": r"seller\s*[:\-]?\s*(.+)",
        "buyer": r"buyer\s*[:\-]?\s*(.+)",
        "total_amount": r"total\s*[:\-]?\s*(.+)",
        "tax": r"tax\s*[:\-]?\s*(.+)"
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            if key == "invoice_number":
                data[key] = match.group(2).strip()
            else:
                data[key] = match.group(1).strip()

    return data


# -------- AI EXTRACTION (Backup) --------
def ai_extract(text):

    prompt = f"""
Extract invoice information:

Invoice Number:
Date:
Seller:
Buyer:
Total Amount:
Tax:

Invoice:
{text}
"""

    result = extractor(prompt, max_length=256)
    return result[0]["generated_text"]


def parse_ai(text):

    data = {
        "invoice_number": "",
        "date": "",
        "seller": "",
        "buyer": "",
        "total_amount": "",
        "tax": ""
    }

    patterns = {
        "invoice_number": r"invoice\s*number\s*:\s*(.*)",
        "date": r"date\s*:\s*(.*)",
        "seller": r"seller\s*:\s*(.*)",
        "buyer": r"buyer\s*:\s*(.*)",
        "total_amount": r"total\s*amount\s*:\s*(.*)",
        "tax": r"tax\s*:\s*(.*)"
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            data[key] = match.group(1).strip()

    return data


def merge_data(rule, ai):
    final = {}
    for key in rule:
        final[key] = rule[key] if rule[key] else ai[key]
    return final


# -------- MAIN --------
def main():

    input_folder = "../input"
    output_folder = "../output"

    files = [f for f in os.listdir(input_folder) if f.endswith(".pdf")]

    if not files:
        print("❌ No PDF invoice found")
        return

    path = os.path.join(input_folder, files[0])

    print("📄 Reading PDF invoice...")
    text = read_pdf(path)

    rule_data = rule_extract(text)

    print("🤖 Running AI backup...")
    ai_raw = ai_extract(text)
    ai_data = parse_ai(ai_raw)

    final_data = merge_data(rule_data, ai_data)

    output_path = os.path.join(output_folder, "result_ai.json")

    with open(output_path, "w") as f:
        json.dump(final_data, f, indent=4)

    print("✅ Extraction Complete")
    print(json.dumps(final_data, indent=4))


if __name__ == "__main__":
    main()
