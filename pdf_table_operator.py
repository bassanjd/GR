import streamlit as st
import pandas as pd
import numpy as np
import cv2
import pytesseract
import psutil
import os
import string
from pdf2image import convert_from_path
from pathlib import Path
from PIL import Image
import time

# ==========================================
# CONFIG
# ==========================================
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\Joel\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
BASE_DIR = Path("data/projects")
BASE_DIR.mkdir(parents=True, exist_ok=True)
DPI = 300
MIN_COL_WIDTH = 5  # px threshold to detect missing vertical line

# ==========================================
# UTILITY FUNCTIONS
# ==========================================
def preprocess(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255,
                                   cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 15, 10)
    return thresh

def cluster_positions(positions, tolerance=15):
    if not positions: return []
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
        if cw > w*0.5:
            h_positions.append(y)
    return cluster_positions(h_positions)

def detect_vertical_lines(image, vertical_size=220, crop_top=0, crop_bottom=None):
    thresh = preprocess(image)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_size))
    vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    v_positions = []
    h, w = vertical.shape
    if crop_bottom is None:
        crop_bottom = h
    crop_height = crop_bottom - crop_top
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        if ch > crop_height*0.5:
            v_positions.append(x)
    v_positions = cluster_positions(v_positions)
    # Enforce 5 fixed columns (6 lines)
    if len(v_positions) < 6:
        v_positions = list(np.linspace(50, w-50, 6).astype(int))
    elif len(v_positions) > 6:
        v_positions = sorted(v_positions)
        indices = np.linspace(0, len(v_positions)-1, 6).astype(int)
        v_positions = [v_positions[i] for i in indices]
    return sorted(v_positions)

def overlay_grid(image, v_lines, h_lines, crop_top, crop_bottom):
    img = np.array(image).copy()
    for x in v_lines:
        cv2.line(img,(x,crop_top),(x,crop_bottom),(0,0,255),2)
    for y in h_lines:
        if crop_top<=y<=crop_bottom:
            cv2.line(img,(v_lines[0],y),(v_lines[-1],y),(0,0,255),2)
    return img

# OCR for one cell
def ocr_cell(cell_img, allow_list):
    config = f"--psm 6 --oem 1 -c tessedit_char_whitelist={allow_list}"
    text = pytesseract.image_to_string(cell_img, config=config).strip()
    conf = 80
    return text, conf

# Extract table for one page
def extract_table(image, v_lines, h_lines, crop_top, crop_bottom, allow_list):
    img = np.array(image)
    rows, confs = [], []
    h_lines = [y for y in h_lines if crop_top <= y <= crop_bottom]
    h_lines = sorted(h_lines)
    for i in range(len(h_lines)-1):
        y1, y2 = h_lines[i], h_lines[i+1]
        if y2-y1 < 10:
            continue
        row, row_conf = [], []
        # Detect if col2 & col3 are merged
        col2_merge = (v_lines[3]-v_lines[2] <= MIN_COL_WIDTH)
        for j in range(5):
            if col2_merge and j == 2:
                # Merge columns 2+3 for OCR
                x1, x2 = v_lines[2], v_lines[4]
            elif col2_merge and j == 3:
                row.append("")
                continue
            else:
                x1, x2 = v_lines[j], v_lines[j+1]
            cell_img = img[y1:y2, x1:x2]
            text, conf = ocr_cell(cell_img, allow_list)
            if j == 0:
                text = text if text.replace('.', '', 1).isdigit() else ""
            row.append(text)
            row_conf.append(conf)
        rows.append(row)
        confs.append(np.mean(row_conf))
    df = pd.DataFrame(rows, columns=[0,1,2,3,4])
    df["row_confidence"] = confs
    return df

# ==========================================
# STREAMLIT LAYOUT
# ==========================================
left_col, middle_col, right_col = st.columns([1,1.5,1.5])

# LEFT COLUMN: PDF + page range
with left_col:
    st.header("PDF & Page Range")
    project_name = st.text_input("Project Name")
    uploaded = st.file_uploader("Upload PDF", type="pdf")
    if project_name and uploaded:
        project_dir = BASE_DIR / project_name
        project_dir.mkdir(exist_ok=True)
        pdf_path = project_dir / "original.pdf"
        if not pdf_path.exists():
            with open(pdf_path,"wb") as f: f.write(uploaded.read())
        if "pages" not in st.session_state:
            poppler_path = r"C:\Program Files (x86)\poppler\Library\bin"
            st.session_state.pages = convert_from_path(pdf_path, dpi=DPI, poppler_path=poppler_path)
        pages = st.session_state.pages
        total_pages = len(pages)
        page_range = st.slider("Select Page Range", 1, total_pages, (1, min(3, total_pages)))
        start_page, end_page = page_range
        st.markdown(f"**Processing Pages:** {start_page} → {end_page}")
        preview_page = st.slider("Preview Page for Grid", start_page, end_page, start_page)

# MIDDLE COLUMN: grid preview + crop
with middle_col:
    st.header("Grid Preview")
    if "pages" in st.session_state:
        image = st.session_state.pages[preview_page-1]
        img_h, img_w = np.array(image).shape[:2]
        crop_top = st.slider("Crop Top", 0, img_h, 0)
        crop_bottom = st.slider("Crop Bottom", 0, img_h, img_h)
        vertical_size = st.slider("Vertical Sensitivity", 50, 400, 220)
        horizontal_size = st.slider("Horizontal Sensitivity", 50, 400, 180)
        v_lines = detect_vertical_lines(image, vertical_size, crop_top, crop_bottom)
        h_lines = detect_horizontal_lines(image, horizontal_size)
        overlay = overlay_grid(image, v_lines, h_lines, crop_top, crop_bottom)
        st.image(overlay, use_container_width=True)

# RIGHT COLUMN: OCR + table
with right_col:
    st.header("OCR & Extracted Table")
    CONF_THRESHOLD = st.slider("Low Confidence Threshold", 20, 100, 60)
    default_allow = string.ascii_letters + string.digits + string.punctuation + " "
    allow_list = st.text_area("Allow-list (whitelist)", default_allow, height=100)
    disallow_list = st.text_area("Disallow-list (blacklist)", "", height=100)
    if st.button("Process Selected Pages"):
        all_dfs = []
        progress_bar = st.progress(0)
        total = end_page - start_page + 1
        pages = st.session_state.pages
        for idx, page_num in enumerate(range(start_page, end_page+1)):
            img_page = pages[page_num-1]
            v_lines_page = detect_vertical_lines(img_page, vertical_size, crop_top, crop_bottom)
            h_lines_page = detect_horizontal_lines(img_page, horizontal_size)
            df = extract_table(img_page, v_lines_page, h_lines_page, crop_top, crop_bottom, allow_list)
            # Re-OCR low confidence rows
            low_conf_rows = df[df['row_confidence'] < CONF_THRESHOLD].index
            for r in low_conf_rows:
                y1, y2 = h_lines_page[r], h_lines_page[r+1] if r+1 < len(h_lines_page) else h_lines_page[-1]
                col2_merge = (v_lines_page[3]-v_lines_page[2] <= MIN_COL_WIDTH)
                for j in range(5):
                    if col2_merge and j == 2:
                        x1, x2 = v_lines_page[2], v_lines_page[4]
                    elif col2_merge and j == 3:
                        df.at[r,j] = ""
                        continue
                    else:
                        x1, x2 = v_lines_page[j], v_lines_page[j+1]
                    cell_img = np.array(img_page)[y1:y2, x1:x2]
                    text, conf = ocr_cell(cell_img, allow_list)
                    if j == 0:
                        text = text if text.replace('.', '', 1).isdigit() else ""
                    df.at[r,j] = text
                df.at[r,'row_confidence'] = np.mean([df.at[r,j]!="" for j in range(5)])*100
            df["page"] = page_num
            all_dfs.append(df)
            progress_bar.progress((idx+1)/total)
            time.sleep(0.05)
        if all_dfs:
            final_df = pd.concat(all_dfs, ignore_index=True)
            st.session_state["df"] = final_df
            st.subheader("Extracted Table")
            edited = st.data_editor(final_df, num_rows="dynamic")
            if project_name and st.button("Export Combined"):
                parquet_path = BASE_DIR / project_name / "combined.parquet"
                edited.to_parquet(parquet_path, index=False)
                st.success("Saved combined parquet")

st.markdown("---")
st.write("Memory (MB):", round(psutil.Process(os.getpid()).memory_info().rss/1e6, 2))