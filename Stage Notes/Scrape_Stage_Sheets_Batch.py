import pandas as pd
import numpy as np
import cv2
import pytesseract
import os
import string
import time
from pathlib import Path
from pdf2image import convert_from_path

# ==========================================
# CONFIG
# ==========================================

pytesseract.pytesseract.tesseract_cmd = r"C:\Users\Joel\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
POPPLER_PATH = r"C:\Program Files (x86)\poppler\Library\bin"

INPUT_DIR = Path("Stage Notes")      # folder containing PDFs
OUTPUT_FILE = Path("combined_output.xlsx")

DPI = 300
MIN_COL_WIDTH = 5
CONF_THRESHOLD = 60

ALLOW_LIST = string.ascii_letters + string.digits + string.punctuation + " "

# ==========================================
# GRID DETECTION
# ==========================================

def preprocess(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        15, 10
    )
    return thresh

def cluster_positions(positions, tolerance=15):
    if not positions:
        return []
    positions = sorted(positions)
    clusters = [[positions[0]]]
    for p in positions[1:]:
        if abs(p - clusters[-1][-1]) < tolerance:
            clusters[-1].append(p)
        else:
            clusters.append([p])
    return [int(np.mean(c)) for c in clusters]

def detect_horizontal_lines(image, horizontal_size=180):
    thresh = preprocess(image)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
    horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h_positions = []
    h, w = horizontal.shape

    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        if cw > w * 0.5:
            h_positions.append(y)

    return cluster_positions(h_positions)

def detect_vertical_lines(image, vertical_size=220):
    thresh = preprocess(image)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_size))
    vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    v_positions = []
    h, w = vertical.shape

    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        if ch > h * 0.5:
            v_positions.append(x)

    return cluster_positions(v_positions)

# ==========================================
# OCR
# ==========================================

def ocr_cell(cell_img):
    config = f"--psm 6 --oem 1 -c tessedit_char_whitelist={ALLOW_LIST}"
    text = pytesseract.image_to_string(cell_img, config=config).strip()
    return text

# ==========================================
# TABLE EXTRACTION
# ==========================================

def extract_table(image, v_lines, h_lines):
    img = np.array(image)

    rows = []

    for i in range(len(h_lines) - 1):
        y1, y2 = h_lines[i], h_lines[i + 1]
        if y2 - y1 < 10:
            continue

        row_texts = []
        j = 0

        while j < len(v_lines) - 1:
            x1 = v_lines[j]
            x2 = v_lines[j + 1]

            # Detect merged column
            if j < len(v_lines) - 2 and (v_lines[j + 1] - v_lines[j] <= MIN_COL_WIDTH):
                x2 = v_lines[j + 2]
                j += 1

            cell_img = img[y1:y2, x1:x2]

            # OCR merged cell as single region
            text = ocr_cell(cell_img)

            # First column numeric guard
            if j == 0:
                if not text.replace('.', '', 1).isdigit():
                    text = ""

            row_texts.append(text)
            j += 1

        rows.append(row_texts)

    if not rows:
        return None

    max_cols = max(len(r) for r in rows)
    df = pd.DataFrame(rows, columns=[f"col_{i}" for i in range(max_cols)])

    return df

# ==========================================
# MAIN BATCH PROCESS
# ==========================================

def process_all_pdfs():

    all_tables = []

    pdf_files = list(INPUT_DIR.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDFs")

    for pdf_path in pdf_files:
        print(f"\nProcessing: {pdf_path.name}")

        try:
            pages = convert_from_path(pdf_path, dpi=DPI, poppler_path=POPPLER_PATH)

            for page_number, page in enumerate(pages, start=1):

                v_lines = detect_vertical_lines(page)
                h_lines = detect_horizontal_lines(page)

                # Only process if BOTH detected
                if len(v_lines) > 2 and len(h_lines) > 2:
                    print(f"  Page {page_number}: grid detected")

                    df = extract_table(page, v_lines, h_lines)

                    if df is not None and not df.empty:
                        df["page"] = page_number
                        df["source_filename"] = pdf_path.name
                        all_tables.append(df)

                else:
                    print(f"  Page {page_number}: skipped (no full grid)")

        except Exception as e:
            print(f"Error processing {pdf_path.name}: {e}")
            continue

    if all_tables:
        final_df = pd.concat(all_tables, ignore_index=True)
        final_df.to_excel(OUTPUT_FILE, index=False)
        print(f"\nSaved combined Excel: {OUTPUT_FILE}")
    else:
        print("No tables detected.")

# ==========================================
# RUN
# ==========================================

if __name__ == "__main__":
    start = time.time()
    process_all_pdfs()
    print(f"\nFinished in {round(time.time() - start, 2)} seconds")