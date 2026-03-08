import streamlit as st
import pandas as pd
import numpy as np
import cv2
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image
import string

# Optional: install EasyOCR if not present
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

# -----------------------------
# CONFIG PATHS
# -----------------------------
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\Joel\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
POPPLER_PATH = r"C:\Program Files (x86)\poppler\Library\bin"

ALLOW_LIST = string.ascii_letters + string.digits + string.punctuation + " "
HEADER_ALLOW = " ABCD"

st.set_page_config(layout="wide")
st.title("PDF Table Extractor – Enhanced OCR with EasyOCR Option")

# -----------------------------
# SIDEBAR CONFIGURATION
# -----------------------------
st.sidebar.header("Configuration")
DPI = st.sidebar.slider("Render DPI", 100, 600, 300, 50)
MIN_COL_WIDTH = st.sidebar.slider("Min Column Width (px)", 1, 20, 5)
HORIZONTAL_KERNEL_SIZE = st.sidebar.slider("Horizontal Line Kernel Width", 50, 300, 180, 10)
VERTICAL_KERNEL_SIZE = st.sidebar.slider("Vertical Line Kernel Height", 50, 300, 220, 10)
BOTTOM_FRACTION_COL0 = st.sidebar.slider("Column 0 OCR Fraction", 0.1, 0.5, 0.3, 0.05)
use_easyocr = st.sidebar.checkbox("Use EasyOCR (if available)", value=False)

if use_easyocr and EASYOCR_AVAILABLE:
    reader = easyocr.Reader(['en'], gpu=False)
elif use_easyocr:
    st.sidebar.warning("EasyOCR not installed, falling back to Tesseract")
    use_easyocr = False

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
        cv2.line(overlay, (0, y), (overlay.shape[1], y), (0, 0, 255), 1)
    for x in v_lines:
        cv2.line(overlay, (x, 0), (x, overlay.shape[0]), (0, 0, 255), 1)
    return overlay

# -----------------------------
# OCR FUNCTIONS
# -----------------------------
def ocr_tesseract(cell_img, merged=False, whitelist=ALLOW_LIST):
    gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    if h < 40:
        gray = cv2.resize(gray, (w*2, h*2), interpolation=cv2.INTER_CUBIC)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    gray = cv2.filter2D(gray, -1, kernel)
    gray = cv2.bitwise_not(gray)
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 11, 2)
    kernel = np.ones((1,1), np.uint8)
    gray = cv2.dilate(gray, kernel, iterations=1)
    gray = cv2.erode(gray, kernel, iterations=1)
    psm_mode = 7 if merged else 6
    config = f"--psm {psm_mode} --oem 1 -c tessedit_char_whitelist={whitelist}"
    text = pytesseract.image_to_string(gray, config=config).strip()
    return text

def ocr_easyocr(cell_img):
    # EasyOCR expects RGB
    cell_rgb = cv2.cvtColor(cell_img, cv2.COLOR_BGR2RGB)
    result = reader.readtext(cell_rgb, detail=0)
    return " ".join(result)

def ocr_cell(cell_img, merged=False, whitelist=ALLOW_LIST):
    if use_easyocr:
        return ocr_easyocr(cell_img)
    else:
        return ocr_tesseract(cell_img, merged=merged, whitelist=whitelist)

def ocr_column0_bottom(cell_img, fraction=0.3):
    h = cell_img.shape[0]
    y_start = int(h*(1-fraction))
    bottom_img = cell_img[y_start:h, :]
    return ocr_cell(bottom_img, merged=False, whitelist="0123456789")

def ocr_header_row(cell_img):
    return ocr_cell(cell_img, merged=False, whitelist=HEADER_ALLOW)

# -----------------------------
# TABLE EXTRACTION (5 columns, merged 2+3)
# -----------------------------
def extract_table_fixed_5_auto(image, h_lines, min_col_width, bottom_fraction=0.3):
    img_np = np.array(image)
    rows_data = []
    y_top = h_lines[0]
    y_bottom = h_lines[-1]

    for row_index in range(len(h_lines)-1):
        y1 = max(h_lines[row_index], y_top)
        y2 = min(h_lines[row_index+1], y_bottom)
        if y2 - y1 < 10:
            continue

        v_lines = detect_vertical_lines_per_row(image, y1, y2, vertical_size=VERTICAL_KERNEL_SIZE)
        if len(v_lines) < 6:
            width = img_np.shape[1]
            v_lines = [int(width*i/5) for i in range(6)]
        x_left = v_lines[0]
        x_right = v_lines[-1]

        row_cells = []
        col_index = 0
        while col_index < 5:
            x1 = v_lines[col_index]
            x2 = v_lines[col_index+1]

            # Header row
            if row_index == 0:
                cell_img = img_np[y1:y2, x1:x2]
                row_cells.append(ocr_header_row(cell_img))
                col_index += 1
                continue

            # Column 0 bottom OCR
            if col_index == 0:
                cell_img = img_np[y1:y2, x1:x2]
                row_cells.append(ocr_column0_bottom(cell_img, fraction=bottom_fraction))
                col_index += 1
                continue

            # Columns 2+3 merged
            if col_index == 2 and (v_lines[3]-v_lines[2] <= min_col_width):
                cell_img = img_np[y1:y2, x1:v_lines[4]]
                row_cells.append(ocr_cell(cell_img, merged=True))
                row_cells.append("")  # col3 blank
                col_index += 2
                continue

            # Other columns normal
            cell_img = img_np[y1:y2, x1:x2]
            row_cells.append(ocr_cell(cell_img))
            col_index += 1

        # Ensure exactly 5 columns
        while len(row_cells)<5:
            row_cells.append("")
        rows_data.append(row_cells)

    df = pd.DataFrame(rows_data, columns=[f"col_{i}" for i in range(5)])
    return df

# -----------------------------
# STREAMLIT UI
# -----------------------------
uploaded_file = st.file_uploader("Upload scanned PDF", type="pdf")

if uploaded_file:
    pages = convert_from_bytes(uploaded_file.read(), dpi=DPI, poppler_path=POPPLER_PATH)
    st.subheader("PDF Pages with Enhanced OCR")

    for i, page in enumerate(pages):
        h_lines = detect_horizontal_lines(page, horizontal_size=HORIZONTAL_KERNEL_SIZE)
        if len(h_lines)<2:
            st.warning(f"Page {i+1}: Not enough horizontal lines.")
            continue

        df_page = extract_table_fixed_5_auto(page, h_lines, MIN_COL_WIDTH, bottom_fraction=BOTTOM_FRACTION_COL0)
        overlay = draw_grid_overlay(page, h_lines,
                                    detect_vertical_lines_per_row(page, h_lines[0], h_lines[-1], vertical_size=VERTICAL_KERNEL_SIZE))

        col1, col2 = st.columns([1,2])
        with col1:
            overlay_pil = Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
            st.image(overlay_pil, caption=f"Page {i+1} Grid Overlay", use_container_width=True)

        with col2:
            st.dataframe(df_page)