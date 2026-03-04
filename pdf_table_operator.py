import streamlit as st
import pandas as pd
import numpy as np
import cv2
import pytesseract
import psutil
import os
import gc
from pdf2image import convert_from_path
from pathlib import Path
from PIL import Image

# ==========================================
# CONFIG
# ==========================================

pytesseract.pytesseract.tesseract_cmd = r"C:\Users\Joel\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

BASE_DIR = Path("data/projects")
BASE_DIR.mkdir(parents=True, exist_ok=True)

DPI = 300

st.set_page_config(layout="wide")
st.title("PDF Grid Table Operator")

# ==========================================
# LINE DETECTION
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

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (horizontal_size, 1)
    )

    horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(
        horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    h_positions = []
    h, w = horizontal.shape

    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        if cw > w * 0.5:
            h_positions.append(y)

    return cluster_positions(h_positions)


def detect_vertical_lines(image, vertical_size=180, crop_top=0, crop_bottom=None):
    thresh = preprocess(image)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, vertical_size)
    )

    vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(
        vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    v_positions = []
    h, w = vertical.shape

    if crop_bottom is None:
        crop_bottom = h

    crop_height = crop_bottom - crop_top

    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)

        if ch > crop_height * 0.5:
            v_positions.append(x)

    v_positions = cluster_positions(v_positions)

    # Enforce exactly 6 vertical boundaries
    if len(v_positions) >= 6:
        v_positions = sorted(v_positions)
        # Keep 6 most evenly spaced
        indices = np.linspace(0, len(v_positions)-1, 6).astype(int)
        v_positions = [v_positions[i] for i in indices]
    elif len(v_positions) >= 2:
        left = min(v_positions)
        right = max(v_positions)
        v_positions = list(np.linspace(left, right, 6).astype(int))
    else:
        v_positions = list(np.linspace(50, w-50, 6).astype(int))

    return sorted(v_positions)

# ==========================================
# GRID OVERLAY
# ==========================================

def overlay_grid(image, v_lines, h_lines, crop_top, crop_bottom):
    img = np.array(image).copy()

    cv2.line(img, (0, crop_top), (img.shape[1], crop_top), (255,0,0), 2)
    cv2.line(img, (0, crop_bottom), (img.shape[1], crop_bottom), (255,0,0), 2)

    for x in v_lines:
        cv2.line(img, (x, crop_top), (x, crop_bottom), (255,0,0), 2)

    for y in h_lines:
        if crop_top <= y <= crop_bottom:
            cv2.line(img, (v_lines[0], y), (v_lines[-1], y), (255,0,0), 2)

    return img

# ==========================================
# OCR
# ==========================================

def extract_table(image, v_lines, h_lines, crop_top, crop_bottom, ocr_config):
    img = np.array(image)

    rows = []
    h_lines = [y for y in h_lines if crop_top <= y <= crop_bottom]
    h_lines = sorted(h_lines)

    for i in range(len(h_lines)-1):
        y1 = h_lines[i]
        y2 = h_lines[i+1]

        if y2 - y1 < 10:
            continue

        row = []

        for j in range(5):
            x1 = v_lines[j]
            x2 = v_lines[j+1]

            cell = img[y1:y2, x1:x2]
            text = pytesseract.image_to_string(
                cell, config=ocr_config
            ).strip()

            row.append(text)

        rows.append(row)

    return pd.DataFrame(rows, columns=[0,1,2,3,4])

# ==========================================
# LAYOUT
# ==========================================

left_col, middle_col, right_col = st.columns([1,1.5,1.5])

# ==========================================
# LEFT COLUMN
# ==========================================

with left_col:

    st.header("PDF & Page Range")

    project_name = st.text_input("Project Name")
    uploaded = st.file_uploader("Upload PDF", type="pdf")

    if project_name and uploaded:

        project_dir = BASE_DIR / project_name
        project_dir.mkdir(parents=True, exist_ok=True)

        pdf_path = project_dir / "original.pdf"

        if not pdf_path.exists():
            with open(pdf_path, "wb") as f:
                f.write(uploaded.read())

        if "pages" not in st.session_state:
            poppler_path = r"C:\Program Files (x86)\poppler\Library\bin"
            st.session_state.pages = convert_from_path(
                pdf_path, dpi=DPI, poppler_path=poppler_path
            )

        pages = st.session_state.pages
        total_pages = len(pages)

        page_range = st.slider(
            "Select Page Range",
            1, total_pages,
            (1, min(3, total_pages))
        )

        start_page, end_page = page_range

        st.markdown(f"**Processing Pages:** {start_page} → {end_page}")

        preview_page = st.slider(
            "Preview Page",
            start_page, end_page, start_page
        )

        image = pages[preview_page-1]
        st.image(image, use_container_width=True)

# ==========================================
# MIDDLE COLUMN
# ==========================================

with middle_col:

    st.header("Grid Controls")

    if "pages" in st.session_state:

        img_h, img_w = np.array(image).shape[:2]

        crop_top = st.slider("Crop Top", 0, img_h, 0)
        crop_bottom = st.slider("Crop Bottom", 0, img_h, img_h)

        vertical_size = st.slider("Vertical Sensitivity", 50, 400, 220)
        horizontal_size = st.slider("Horizontal Sensitivity", 50, 400, 180)

        v_lines = detect_vertical_lines(
            image, vertical_size, crop_top, crop_bottom
        )

        h_lines = detect_horizontal_lines(
            image, horizontal_size
        )

        overlay = overlay_grid(
            image, v_lines, h_lines, crop_top, crop_bottom
        )

        st.image(overlay, use_container_width=True)


# ==========================================
# RIGHT COLUMN
# ==========================================

with right_col:
    st.subheader("OCR Settings")
    psm = st.selectbox("PSM Mode", [3,4,6,11], index=2)
    oem = st.selectbox("OEM Mode", [0,1,2,3], index=1)
    whitelist = st.text_input("Whitelist (optional)", "")
    ocr_config = f"--psm {psm} --oem {oem}"
    if whitelist:
        ocr_config += f' -c tessedit_char_whitelist="{whitelist}"'
    if st.button("Process Selected Pages"):
        all_dfs = []
        for page_num in range(start_page, end_page+1):
            img_page = pages[page_num-1]
            v_lines_page = detect_vertical_lines(
                img_page, vertical_size, crop_top, crop_bottom
            )
            h_lines_page = detect_horizontal_lines(
                img_page, horizontal_size
            )
            df = extract_table(
                img_page,
                v_lines_page,
                h_lines_page,
                crop_top,
                crop_bottom,
                ocr_config
            )
            df["page"] = page_num
            all_dfs.append(df)
        if all_dfs:
            final_df = pd.concat(all_dfs, ignore_index=True)
            st.session_state["df"] = final_df
        gc.collect()

    st.header("Extracted Table")

    if "df" in st.session_state:

        df = st.session_state["df"]
        edited = st.data_editor(df, num_rows="dynamic")

        if project_name and st.button("Export Combined"):
            parquet_path = BASE_DIR / project_name / "combined.parquet"
            edited.to_parquet(parquet_path, index=False)
            st.success("Saved combined parquet")

    st.markdown("---")
    st.write(
        "Memory (MB):",
        round(psutil.Process(os.getpid()).memory_info().rss / 1e6, 2)
    )