import streamlit as st
import pandas as pd
import numpy as np
import cv2
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image
import io
import string

# -----------------------------
# CONFIG PATHS
# -----------------------------
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\Joel\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
POPPLER_PATH = r"C:\Program Files (x86)\poppler\Library\bin"

ALLOW_LIST = string.ascii_letters + string.digits + string.punctuation + " "

st.set_page_config(layout="wide")
st.title("PDF Table Extractor with 5 Columns and Merged Column Handling")

# -----------------------------
# SIDEBAR CONFIGURATION
# -----------------------------
st.sidebar.header("Configuration")
DPI = st.sidebar.slider("Render DPI", 100, 600, 300, 50)
MIN_COL_WIDTH = st.sidebar.slider("Min Column Width (px)", 1, 20, 5)
HORIZONTAL_KERNEL_SIZE = st.sidebar.slider("Horizontal Line Kernel Width", 50, 300, 180, 10)
VERTICAL_KERNEL_SIZE = st.sidebar.slider("Vertical Line Kernel Height", 50, 300, 220, 10)

# -----------------------------
# GRID DETECTION FUNCTIONS
# -----------------------------
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

def detect_vertical_lines_per_row(image, y_top, y_bottom, vertical_size=220):
    img_row = np.array(image)[y_top:y_bottom, :]
    thresh = preprocess(img_row)
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

def draw_grid_overlay(img, h_lines, v_lines):
    overlay = np.array(img).copy()
    for y in h_lines:
        cv2.line(overlay, (0, y), (overlay.shape[1], y), (0, 0, 255), 1)  # Red horizontal
    for x in v_lines:
        cv2.line(overlay, (x, 0), (x, overlay.shape[0]), (0, 0, 255), 1)  # Red vertical
    return overlay

# -----------------------------
# OCR & TABLE EXTRACTION (5 columns, merged 2+3)
# -----------------------------
def ocr_cell(cell_img):
    config = f"--psm 6 --oem 1 -c tessedit_char_whitelist={ALLOW_LIST}"
    text = pytesseract.image_to_string(cell_img, config=config).strip()
    return text

def extract_table_fixed_5(image, h_lines, min_col_width):
    img_np = np.array(image)
    rows_data = []

    y_top = h_lines[0]
    y_bottom = h_lines[-1]

    for i in range(len(h_lines)-1):
        y1 = max(h_lines[i], y_top)
        y2 = min(h_lines[i+1], y_bottom)
        if y2 - y1 < 10:
            continue

        # Vertical lines per row
        v_lines = detect_vertical_lines_per_row(image, y1, y2, vertical_size=VERTICAL_KERNEL_SIZE)
        if len(v_lines) < 6:  # Need 6 boundaries for 5 columns
            # fallback: evenly divide width into 5
            width = img_np.shape[1]
            v_lines = [int(width * i / 5) for i in range(6)]

        x_left = v_lines[0]
        x_right = v_lines[-1]

        row_cells = []
        j = 0
        while j < len(v_lines)-1 and len(row_cells)<5:
            x1 = max(v_lines[j], x_left)
            x2 = min(v_lines[j+1], x_right)

            # Detect merge of column 2+3
            if j == 1 and (v_lines[j+1]-v_lines[j]) <= min_col_width:
                # Merge col2+col3
                if j+2 < len(v_lines):
                    x2 = v_lines[j+2]
                    cell_img = img_np[y1:y2, x1:x2]
                    text = ocr_cell(cell_img)
                    row_cells.append(text)
                    row_cells.append("")  # column 3 left blank
                    j += 2
                    continue

            cell_img = img_np[y1:y2, x1:x2]
            text = ocr_cell(cell_img)
            row_cells.append(text)
            j += 1

        # Fill to 5 columns
        while len(row_cells) < 5:
            row_cells.append("")

        rows_data.append(row_cells)

    if not rows_data:
        return None
    df = pd.DataFrame(rows_data, columns=[f"col_{i}" for i in range(5)])
    return df

# -----------------------------
# STREAMLIT UI
# -----------------------------
uploaded_file = st.file_uploader("Upload scanned PDF", type="pdf")

if uploaded_file:
    pages = convert_from_bytes(uploaded_file.read(), dpi=DPI, poppler_path=POPPLER_PATH)
    st.subheader("PDF Pages and Extracted Tables (5 Columns, Merged 2+3)")

    for i, page in enumerate(pages):
        h_lines = detect_horizontal_lines(page, horizontal_size=HORIZONTAL_KERNEL_SIZE)
        if len(h_lines)<2:
            st.warning(f"Page {i+1}: Not enough horizontal lines.")
            continue

        df_page = extract_table_fixed_5(page, h_lines, MIN_COL_WIDTH)
        overlay = draw_grid_overlay(page, h_lines, detect_vertical_lines_per_row(page, h_lines[0], h_lines[-1], vertical_size=VERTICAL_KERNEL_SIZE))

        col1, col2 = st.columns([1,2])
        with col1:
            overlay_pil = Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
            st.image(overlay_pil, caption=f"Page {i+1} Grid Overlay", use_container_width=True)

        with col2:
            if df_page is not None:
                st.dataframe(df_page)
            else:
                st.write("No table extracted for this page.")