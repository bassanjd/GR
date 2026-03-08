import fitz
import pytesseract
import pandas as pd
import numpy as np
import cv2
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import traceback
import gc

# ==============================
# CONFIG
# ==============================

INPUT_DIR = Path(".\\Stage Notes")
OUTPUT_DIR = Path(".\\Stage Notes Output")
DPI = 300

pytesseract.pytesseract.tesseract_cmd = r"C:\Users\Joel\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"


# ==============================
# SAFE PDF RENDER
# ==============================

def render_pdf_to_images(pdf_path, dpi=300):
    images = []
    try:
        doc = fitz.open(pdf_path)
        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)

        for page in doc:
            pix = page.get_pixmap(matrix=mat)
            img = np.frombuffer(pix.samples, dtype=np.uint8)
            img = img.reshape(pix.height, pix.width, pix.n)

            if pix.n == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            images.append(img)

        doc.close()

    except Exception as e:
        print(f"⚠ Render failed: {pdf_path.name}")
        raise e

    return images


# ==============================
# TABLE DETECTION
# ==============================

def detect_table_cells(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        15, 8
    )

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))

    detect_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel)
    detect_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel)

    grid = cv2.add(detect_horizontal, detect_vertical)

    contours, _ = cv2.findContours(grid, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cells = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 50 and h > 20:
            cells.append((x, y, w, h))

    cells = sorted(cells, key=lambda b: (b[1], b[0]))
    return cells


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
# ISOLATED STAGE PROCESSOR
# ==============================

def process_stage_safe(pdf_path):

    try:
        print(f"Processing {pdf_path.name}")

        images = render_pdf_to_images(pdf_path, dpi=DPI)

        stage_tables = []

        for img in images:
            df = extract_table_from_page(img)
            if df is not None:
                stage_tables.append(df)

            # aggressively free memory
            del img
            gc.collect()

        if not stage_tables:
            return {"file": pdf_path.name, "status": "no_table", "error": None}

        combined = pd.concat(stage_tables, ignore_index=True)

        stage_name = pdf_path.stem

        combined.to_csv(OUTPUT_DIR / f"{stage_name}_FULL_TABLE.csv", index=False)
        combined.to_parquet(OUTPUT_DIR / f"{stage_name}_FULL_TABLE.parquet", index=False)

        del combined
        del stage_tables
        gc.collect()

        return {"file": pdf_path.name, "status": "success", "error": None}

    except MemoryError:
        return {
            "file": pdf_path.name,
            "status": "memory_error",
            "error": "Out of memory"
        }

    except Exception as e:
        return {
            "file": pdf_path.name,
            "status": "error",
            "error": traceback.format_exc()
        }


# ==============================
# MAIN
# ==============================

def main():

    OUTPUT_DIR.mkdir(exist_ok=True)

    stage_files = sorted(INPUT_DIR.glob("*.pdf"))

    if not stage_files:
        print("No PDFs found.")
        return

    workers = max(1, multiprocessing.cpu_count() - 1)

    print(f"Using {workers} workers\n")

    results_log = []

    with ProcessPoolExecutor(max_workers=workers) as executor:

        futures = {
            executor.submit(process_stage_safe, pdf): pdf
            for pdf in stage_files
        }

        for future in as_completed(futures):
            result = future.result()
            results_log.append(result)

            if result["status"] == "success":
                print(f"✓ {result['file']} completed")
            else:
                print(f"⚠ {result['file']} → {result['status']}")

    # Save processing log
    log_df = pd.DataFrame(results_log)
    log_df.to_csv(OUTPUT_DIR / "processing_log.csv", index=False)

    print("\nProcessing complete.")
    print("Log saved to processing_log.csv")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()