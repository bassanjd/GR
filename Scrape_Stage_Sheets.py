import fitz
import pytesseract
import pandas as pd
import numpy as np
import cv2
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# ==============================
# CONFIG
# ==============================

INPUT_DIR = Path(".\\Stage Notes")
OUTPUT_DIR = Path(".\\Stage Notes Output")
DPI = 300

pytesseract.pytesseract.tesseract_cmd = r"C:\Users\Joel\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"


# ==============================
# PDF RENDER
# ==============================

def render_pdf_to_images(pdf_path, dpi=300):
    doc = fitz.open(pdf_path)
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)

    images = []

    for page in doc:
        pix = page.get_pixmap(matrix=mat)
        img = np.frombuffer(pix.samples, dtype=np.uint8)
        img = img.reshape(pix.height, pix.width, pix.n)

        if pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        images.append(img)

    return images


# ==============================
# TABLE GRID DETECTION
# ==============================

def detect_table_cells(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        15, 8
    )

    # Detect horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    detect_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel)

    # Detect vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    detect_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel)

    # Combine
    grid = cv2.add(detect_horizontal, detect_vertical)

    # Find contours
    contours, _ = cv2.findContours(grid, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cells = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # Filter small noise boxes
        if w > 50 and h > 20:
            cells.append((x, y, w, h))

    # Sort top-to-bottom, left-to-right
    cells = sorted(cells, key=lambda b: (b[1], b[0]))

    return cells


# ==============================
# OCR EACH CELL
# ==============================

def ocr_cell(image, box):
    x, y, w, h = box
    cell_img = image[y:y+h, x:x+w]

    text = pytesseract.image_to_string(
        cell_img,
        config="--oem 3 --psm 6"
    )

    return text.strip()


def extract_table_from_page(image):

    cells = detect_table_cells(image)

    if not cells:
        return None

    # Cluster rows by Y coordinate
    rows = {}
    for (x, y, w, h) in cells:
        row_key = y // 20
        rows.setdefault(row_key, []).append((x, y, w, h))

    table = []

    for row_key in sorted(rows.keys()):
        row_cells = sorted(rows[row_key], key=lambda b: b[0])
        row_text = [ocr_cell(image, box) for box in row_cells]
        table.append(row_text)

    return pd.DataFrame(table)


# ==============================
# PROCESS STAGE
# ==============================

def process_stage(pdf_path):

    print(f"Processing {pdf_path.name}")

    images = render_pdf_to_images(pdf_path, dpi=DPI)

    stage_tables = []

    for img in images:
        df = extract_table_from_page(img)
        if df is not None:
            stage_tables.append(df)

    if not stage_tables:
        return None

    return pd.concat(stage_tables, ignore_index=True)


# ==============================
# MAIN
# ==============================

def main():

    OUTPUT_DIR.mkdir(exist_ok=True)

    stage_files = sorted(INPUT_DIR.glob("*.pdf"))

    workers = max(1, multiprocessing.cpu_count() - 1)
    print(f"Using {workers} workers")

    master_tables = []

    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(process_stage, stage_files))

    for pdf_file, df in zip(stage_files, results):
        if df is None:
            continue

        stage_name = pdf_file.stem

        df.to_csv(OUTPUT_DIR / f"{stage_name}_FULL_TABLE.csv", index=False)
        df.to_parquet(OUTPUT_DIR / f"{stage_name}_FULL_TABLE.parquet", index=False)

        master_tables.append(df)

    if master_tables:
        master = pd.concat(master_tables, ignore_index=True)
        master.to_csv(OUTPUT_DIR / "Great_Race_2025_FULL_MASTER.csv", index=False)
        master.to_parquet(OUTPUT_DIR / "Great_Race_2025_FULL_MASTER.parquet", index=False)

    print("Done.")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()