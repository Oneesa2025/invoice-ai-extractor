import os
import re
import json
from transformers import pipeline


# Load model
print("⏳ Loading local AI model...")
extractor = pipeline("text2text-generation", model="google/flan-t5-base")
print("✅ Model loaded")


def read_invoice(path):
    with open(path, "r") as f:
        return f.read()


# Rule-based extractor (always works for simple invoices)
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
        "invoice_number": r"invoice\s*no\s*[:\-]?\s*(\S+)",
        "date": r"date\s*[:\-]?\s*(.+)",
        "seller": r"seller\s*[:\-]?\s*(.+)",
        "buyer": r"buyer\s*[:\-]?\s*(.+)",
        "total_amount": r"total\s*[:\-]?\s*(.+)",
        "tax": r"tax\s*[:\-]?\s*(.+)"
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            data[key] = match.group(1).strip()

    return data


# AI extractor (backup)
def ai_extract(text):

    prompt = f"""
Extract invoice info and write like this:

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


# Parse AI output
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
        if rule[key]:
            final[key] = rule[key]
        else:
            final[key] = ai[key]

    return final


def main():

    input_folder = "../input"
    output_folder = "../output"

    files = os.listdir(input_folder)

    if not files:
        print("❌ No invoice found")
        return

    path = os.path.join(input_folder, files[0])

    text = read_invoice(path)

    print("📄 Reading invoice...")

    # Step 1: Rule-based
    rule_data = rule_extract(text)

    # Step 2: AI backup
    print("🤖 Running AI backup...")
    ai_raw = ai_extract(text)
    ai_data = parse_ai(ai_raw)

    # Step 3: Merge
    final_data = merge_data(rule_data, ai_data)

    output_path = os.path.join(output_folder, "result_ai.json")

    with open(output_path, "w") as f:
        json.dump(final_data, f, indent=4)

    print("✅ Final Output Saved")
    print("\n📊 Result:")
    print(json.dumps(final_data, indent=4))


if __name__ == "__main__":
    main()
