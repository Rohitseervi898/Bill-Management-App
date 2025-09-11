import re
import json
import sys
import argparse
from pathlib import Path
import doctr.io as io
import doctr.models as models

# ---------- OCR + Regex Extractor ----------

def extract_company(text):
    lines = [line.strip() for line in text.split("\n") if line.strip()]

    # Priority: look for "For. ..." footer
    for line in reversed(lines):  # check from bottom
        match = re.search(r"For[ .:]*\s*(.*STORE.*)", line, re.IGNORECASE)
        if match:
            company = match.group(1).strip()
            return re.sub(r'[^A-Za-z0-9 &.,-]', '', company).title()

    # Fallback: first match containing keywords
    for line in lines:
        if re.search(r"(STORE|PVT|LTD|SHOP|MART|TRADERS|ENTERPRISES)", line, re.IGNORECASE):
            return re.sub(r'[^A-Za-z0-9 &.,-]', '', line).title()

    return None


def extract_amount(text):
    # First priority: explicit Net Payable
    match = re.search(r"(?:Net\s*Payable|Net\s*Amount)\D*([\d,]+\.\d{2})", text, re.IGNORECASE)
    if match:
        return match.group(1).replace(",", "")
    
    # Fallback: pick the largest number in the invoice
    numbers = re.findall(r"\d{2,7}\.\d{2}", text)
    if numbers:
        return max(numbers, key=lambda x: float(x))
    
    return None


def extract_fields(text):
    fields = {}

    # Invoice number
    match = re.search(r"(?:Invoice\s*No|Tax\s*Invoice\s*No)[:\s]*([A-Z0-9\-]+)", text, re.IGNORECASE)
    fields["invoice_number"] = match.group(1).strip() if match else None

    # Date
    match = re.search(r"(?:Invoice\s*Date)[:\s]*([\d/.-]+)", text, re.IGNORECASE)
    fields["date"] = match.group(1).strip() if match else None

    # Amount
    fields["amount"] = extract_amount(text)

    # Company
    fields["company"] = extract_company(text)

    return fields


def process_document(paths, merge=False):
    """Process one or many files (merge=True treats them as one bill)."""
    model = models.ocr_predictor(pretrained=True)

    if merge:
        # Multi-page document from multiple images/PDFs
        docs = []
        for p in paths:
            if p.suffix.lower() == ".pdf":
                docs.append(io.DocumentFile.from_pdf(p))
            else:
                docs.append(io.DocumentFile.from_images(p))
        # flatten into one multi-page DocumentFile
        doc = sum(docs[1:], docs[0])
        result = model(doc)

        text = "\n".join(
            [" ".join([w.value for w in line.words])
             for page in result.pages
             for block in page.blocks
             for line in block.lines]
        )

        return extract_fields(text)

    else:
        results = {}
        for p in paths:
            if p.suffix.lower() == ".pdf":
                doc = io.DocumentFile.from_pdf(p)
            else:
                doc = io.DocumentFile.from_images(p)

            result = model(doc)

            text = "\n".join(
                [" ".join([w.value for w in line.words])
                 for page in result.pages
                 for block in page.blocks
                 for line in block.lines]
            )

            results[p.name] = extract_fields(text)
        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="File or folder containing invoices")
    parser.add_argument("--merge", action="store_true", help="Treat all files in folder as one multi-page bill")
    args = parser.parse_args()

    path = Path(args.path)

    if path.is_file():
        fields = process_document([path], merge=True)  # single file = single doc
        print(json.dumps(fields, indent=2))

    elif path.is_dir():
        files = sorted([p for p in path.iterdir() if p.suffix.lower() in [".jpg", ".png", ".pdf"]])
        if not files:
            print("No invoice files found in folder.")
            sys.exit(1)

        if args.merge:
            fields = process_document(files, merge=True)
            print(json.dumps(fields, indent=2))
        else:
            results = process_document(files, merge=False)
            print(json.dumps(results, indent=2))
