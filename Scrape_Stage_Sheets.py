import fitz  # PyMuPDF
import pytesseract
import pandas as pd
import numpy as np
import cv2
import re
from pathlib import Path

# ==============================
# CONFIG
# ==============================

INPUT_DIR = Path(".\\Stage Notes")
OUTPUT_DIR = Path(".\\Stage Notes Output")
DPI = 300
ROW_BIN_PIXELS = 40

# Uncomment if needed (Windows)
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\Joel\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

# ==============================
# PDF RENDER (NO POPPLER)
# ==============================

def render_pdf_to_images(pdf_path, dpi=300):
    doc = fitz.open(pdf_path)
    zoom = dpi / 72  # 72 is default PDF DPI
    mat = fitz.Matrix(zoom, zoom)

    images = []

    for page in doc:
        pix = page.get_pixmap(matrix=mat)
        img = np.frombuffer(pix.samples, dtype=np.uint8)
        img = img.reshape(pix.height, pix.width, pix.n)

        if pix.n == 4:  # remove alpha channel
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        images.append(img)

    return images


# ==============================
# OCR + EXTRACTION
# ==============================

def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 2
    )
    return thresh


def time_to_seconds(time_str):
    match = re.match(r"(\d+)m([\d\.]+)s", str(time_str))
    if not match:
        return None
    return int(match.group(1)) * 60 + float(match.group(2))


def extract_page(image):

    h, w = image.shape

    # Column layout tuned to your stage sheet
    col_A = (0, int(w * 0.45))
    col_C = (int(w * 0.45), int(w * 0.80))

    data = pytesseract.image_to_data(
        image,
        output_type=pytesseract.Output.DATAFRAME,
        config="--oem 3 --psm 6"
    )

    data = data.dropna(subset=["text"])
    data = data[data["text"].str.strip() != ""]
    data["row_bin"] = (data["top"] // ROW_BIN_PIXELS).astype(int)

    rows = []

    for _, group in data.groupby("row_bin"):
        group = group.sort_values("left")

        colA_words = group[
            (group["left"] >= col_A[0]) &
            (group["left"] < col_A[1])
        ]["text"].tolist()

        colC_words = group[
            (group["left"] >= col_C[0]) &
            (group["left"] < col_C[1])
        ]["text"].tolist()

        colA_text = " ".join(colA_words)
        colC_text = " ".join(colC_words)

        leg_match = re.search(r"\b(\d{1,2})\b", colA_text)
        times = re.findall(r"\d+m\d+\.\d+s", colC_text)

        if leg_match and len(times) >= 1:
            leg_time = times[0]
            cumulative_time = times[1] if len(times) > 1 else None

            rows.append({
                "Leg": int(leg_match.group(1)),
                "Instruction": colA_text.strip(),
                "Leg_Time": leg_time,
                "Leg_Time_sec": time_to_seconds(leg_time),
                "Cumulative_Time": cumulative_time,
                "Cumulative_Time_sec": time_to_seconds(cumulative_time) if cumulative_time else None
            })

    return rows


# ==============================
# STAGE PROCESSOR
# ==============================

def extract_stage(pdf_path):

    print(f"\nProcessing: {pdf_path.name}")

    images = render_pdf_to_images(pdf_path, dpi=DPI)

    all_rows = []

    for i, img in enumerate(images):
        print(f"  OCR page {i+1}")
        processed = preprocess(img)
        rows = extract_page(processed)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows).drop_duplicates()

    if df.empty:
        print("  ⚠ No data extracted.")
        return None

    df = df.sort_values("Leg").reset_index(drop=True)

    # Validate cumulative monotonicity
    if df["Cumulative_Time_sec"].notna().all():
        if not df["Cumulative_Time_sec"].is_monotonic_increasing:
            print("  ⚠ Warning: cumulative time not monotonic.")

    return df


# ==============================
# BATCH PROCESSOR
# ==============================

def main():

    OUTPUT_DIR.mkdir(exist_ok=True)

    stage_files = sorted(INPUT_DIR.glob("*.pdf"))

    if not stage_files:
        print("No PDFs found in input folder.")
        return

    master_df_list = []

    for pdf_file in stage_files:
        df = extract_stage(pdf_file)

        if df is None:
            continue

        stage_name = pdf_file.stem
        df["Stage"] = stage_name

        # Save per stage
        csv_path = OUTPUT_DIR / f"{stage_name}_clean.csv"
        parquet_path = OUTPUT_DIR / f"{stage_name}_clean.parquet"

        df.to_csv(csv_path, index=False)
        df.to_parquet(parquet_path, index=False)

        print(f"  Saved: {csv_path.name}")
        print(f"  Saved: {parquet_path.name}")

        master_df_list.append(df)

    # Combine master
    if master_df_list:
        master_df = pd.concat(master_df_list, ignore_index=True)

        master_csv = OUTPUT_DIR / "Great_Race_2025_master.csv"
        master_parquet = OUTPUT_DIR / "Great_Race_2025_master.parquet"

        master_df.to_csv(master_csv, index=False)
        master_df.to_parquet(master_parquet, index=False)

        print("\nMaster files saved:")
        print(master_csv)
        print(master_parquet)


if __name__ == "__main__":
    main()
