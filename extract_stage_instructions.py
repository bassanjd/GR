"""
extract_stage_instructions.py

Processes all Great Race stage notes PDFs and extracts a structured instruction
table suitable for variable importance analysis.

Outputs:
    Output/stage_instructions.parquet   – one row per grid instruction
    Output/leg_characteristics.parquet  – one row per (Stage, Leg) with
                                          aggregated features for joining to
                                          long_format_times.parquet

Usage:
    python extract_stage_instructions.py [--stages 1 2 3] [--dpi 250]

Dependencies:
    pdf2image, easyocr, pandas, pyarrow, opencv-python, pillow, numpy
"""

import re
import sys
import json
import logging
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from pdf2image import convert_from_path
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent
STAGE_NOTES_DIR = BASE / "Stage Notes"
OUTPUT_DIR = BASE / "Output"
POPPLER_PATH = r"C:\Program Files (x86)\poppler\Library\bin"

INSTRUCTIONS_PARQUET = OUTPUT_DIR / "stage_instructions.parquet"
LEG_PARQUET = OUTPUT_DIR / "leg_characteristics.parquet"
CHECKPOINT_JSON = OUTPUT_DIR / "_extract_checkpoint.json"  # resume support

# ─── Stage PDF mapping (prefer "Clean" versions) ──────────────────────────────
def build_pdf_map(stages=None):
    pdfs = {}
    for s in range(0, 10):
        if stages and s not in stages:
            continue
        clean = STAGE_NOTES_DIR / f"2025 Great Race Stage {s} - Clean.pdf"
        regular = STAGE_NOTES_DIR / f"2025 Great Race Stage {s}.pdf"
        trophy = STAGE_NOTES_DIR / "2025 Great Race Trophy Run.pdf"
        if s == 0 and trophy.exists():
            pdfs[0] = trophy
        elif clean.exists():
            pdfs[s] = clean
        elif regular.exists():
            pdfs[s] = regular
    return pdfs

# ─── EasyOCR ──────────────────────────────────────────────────────────────────
_reader = None

def get_reader():
    global _reader
    if _reader is None:
        import easyocr
        log.info("Initializing EasyOCR (first run may take ~30s)…")
        _reader = easyocr.Reader(['en'], gpu=False)
    return _reader

# ─── Grid detection ───────────────────────────────────────────────────────────

def preprocess_for_lines(image_pil):
    gray = cv2.cvtColor(np.array(image_pil), cv2.COLOR_BGR2GRAY)
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 10
    )


def cluster(positions, tol=12):
    if not positions:
        return []
    positions = sorted(set(positions))
    groups = [[positions[0]]]
    for p in positions[1:]:
        if p - groups[-1][-1] < tol:
            groups[-1].append(p)
        else:
            groups.append([p])
    return [int(np.mean(g)) for g in groups]


def detect_hlines(image_pil, kernel_w=200):
    thresh = preprocess_for_lines(image_pil)
    h, w = thresh.shape
    kern = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 1))
    hmap = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kern)
    contours, _ = cv2.findContours(hmap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ys = [int(cv2.boundingRect(c)[1]) for c in contours
          if cv2.boundingRect(c)[2] > w * 0.45]
    return cluster(ys)


def detect_vlines(image_pil, y1, y2, kernel_h=200):
    img = np.array(image_pil)[y1:y2, :]
    thresh = preprocess_for_lines(Image.fromarray(img))
    h, w = thresh.shape
    kern = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_h))
    vmap = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kern)
    contours, _ = cv2.findContours(vmap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    xs = [int(cv2.boundingRect(c)[0]) for c in contours
          if cv2.boundingRect(c)[3] > h * 0.45]
    return cluster(xs)


def fallback_vlines(img_w, n_cols=5):
    """Divide page into equal column strips as a fallback."""
    return [int(img_w * i / n_cols) for i in range(n_cols + 1)]

# ─── OCR helpers ──────────────────────────────────────────────────────────────

def ocr_cell(img_np, x1, y1, x2, y2, pad=4):
    """OCR a cell rectangle; return list of (bbox_top, text, confidence)."""
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(img_np.shape[1], x2 + pad)
    y2 = min(img_np.shape[0], y2 + pad)
    cell = img_np[y1:y2, x1:x2]
    if cell.size == 0 or cell.shape[0] < 8 or cell.shape[1] < 8:
        return []
    results = get_reader().readtext(cell, detail=1, paragraph=False)
    # Normalise to (relative_y_top, text, confidence)
    cell_h = cell.shape[0]
    out = []
    for bbox, text, conf in results:
        top_y = min(pt[1] for pt in bbox) / max(cell_h, 1)
        out.append((top_y, text.strip(), conf))
    return out


def cell_text(detections):
    """Join all detection texts, sorted top-to-bottom."""
    return " | ".join(d[1] for d in sorted(detections, key=lambda d: d[0]) if d[1])


def bottom_number(detections, threshold=0.70):
    """
    Extract the instruction number from the bottom portion of the A-column cell.
    It appears as a 1–3 digit number in roughly the bottom 30% of the cell.
    """
    candidates = []
    for top_y, text, conf in detections:
        if top_y >= threshold:
            nums = re.findall(r'\b(\d{1,3})\b', text)
            for n in nums:
                v = int(n)
                if 1 <= v <= 999:
                    candidates.append(v)
    if candidates:
        return min(candidates)   # smallest number = most likely row number
    return None

# ─── Text parsing helpers ──────────────────────────────────────────────────────

_TIME_RE  = re.compile(r'(\d+)\s*m\s*(\d+(?:\.\d+)?)\s*s', re.I)
_SPEED_RE = re.compile(r'(\d+)\s*MPH', re.I)
_SPD_LIM  = re.compile(r'speed\s*limit\s+(\d+)', re.I)
_COMPASS  = re.compile(r'\b(North|South|East|West|NE|NW|SE|SW)\b', re.I)
_HIGHWAY  = re.compile(r'\b(I-?\d+|US\s*\d+|SR\s*\d+|Exit\s+\d|I90|I94|I35)', re.I)
_COUNTY   = re.compile(r'\b(County|Co\.?\s*\d+|CR\s*\d+)', re.I)
_ODOMETER = re.compile(r'0\s*[|I]\s*(\d)\s*[|I]\s*(\d)\s*[|I]\s*0')
_CDT_TIME = re.compile(r'\b(\d{1,2}):(\d{2}):(\d{2})\b')
_SIGNED   = re.compile(r'(?<![:\d])([+\-]\s*\d+(?:\.\d+)?)(?![:\d])')
_HW_SPEED = re.compile(r'(\d+)\s*MPH\s+FOR\s+(\d+)\s*SEC', re.I)
_SIT_TIME = re.compile(r'0\s*MPH\s+[Ss][iI][Tt]', re.I)
_INTERVAL_PARENS = re.compile(r'\((\d+)m(\d+)s\)')   # e.g. (6m00s) = rest break
_LEG_TOTAL = re.compile(r'^(\d+)m(\d{2})s$')          # e.g. 26m00s at checkpoint


def parse_times(text):
    return [float(m.group(1)) * 60 + float(m.group(2))
            for m in _TIME_RE.finditer(text)]


def parse_speeds(text):
    return [int(m.group(1)) for m in _SPEED_RE.finditer(text)]


def parse_speed_limit(text):
    m = _SPD_LIM.search(text)
    return int(m.group(1)) if m else None


def parse_odometer_code(text):
    """Returns digits as int: '0|2|8|0' → 280."""
    m = _ODOMETER.search(text)
    return int(f"{m.group(1)}{m.group(2)}") if m else None


def parse_compass(text):
    m = _COMPASS.search(text)
    return m.group(1).title() if m else None


def parse_hw_error(text):
    """Return the most prominent signed correction value, e.g. -3.7."""
    hits = _SIGNED.findall(text)
    if hits:
        try:
            return float(hits[0].replace(' ', ''))
        except ValueError:
            pass
    return None


def parse_hw_speed_note(text):
    m = _HW_SPEED.search(text)
    return f"{m.group(1)} MPH FOR {m.group(2)} SEC" if m else None


def parse_checkpoint_time(text):
    """Extract handwritten checkpoint time from D column: 'CkPT2 3:08:06'."""
    m = _CDT_TIME.search(text)
    return m.group(0) if m else None


def classify_col_a(text):
    upper = text.upper()
    return {
        'has_stop':         bool(re.search(r'\bSTOP\b', upper)),
        'has_railroad':     bool(re.search(r'\bR\s*R\b|railroad|crossing', upper, re.I)),
        'has_yield':        bool(re.search(r'\bYIELD\b', upper)),
        'has_speed_limit':  bool(_SPD_LIM.search(text)),
        'speed_limit_mph':  parse_speed_limit(text),
        'is_highway':       bool(_HIGHWAY.search(text)),
        'is_county_road':   bool(_COUNTY.search(text)),
        'road_direction':   parse_compass(text),
        'has_distance':     bool(re.search(r'\d+\s*(mile|miles|1\/4|1\/2|3\/4)', text, re.I)),
        'has_sit_stop':     bool(_SIT_TIME.search(text)),
    }


def classify_col_c(text):
    """
    Column C timing structure per instruction:
      Regular:     interval_s | cumulative_s
      Leg start:   target_mph | leg_duration_s | '*' | 0m00.0s
      Checkpoint:  interval_s | cumulative_s | leg_total_s (no decimal)
    """
    times = parse_times(text)
    speeds = parse_speeds(text)

    # Is this the leg-start row? Detected by presence of '0m00.0s' / '*'
    is_leg_start = bool(re.search(r'0\s*m\s*00\.0\s*s', text, re.I)) or \
                   bool(re.search(r'\*\s*0\s*m\s*00', text))

    # CDT clock time from C column (appears at leg start in some formats)
    cdt_m = _CDT_TIME.search(text)

    # Leg total time at checkpoint (last time value with no decimal, Nm00s)
    leg_total = None
    for m in _LEG_TOTAL.finditer(text):
        leg_total = float(m.group(1)) * 60 + float(m.group(2))

    # Rest/break window (e.g. "(6m00s)")
    break_m = _INTERVAL_PARENS.search(text)
    break_window = None
    if break_m:
        break_window = int(break_m.group(1)) * 60 + int(break_m.group(2))

    return {
        'target_speed_mph':   speeds[0] if speeds else None,
        'speed_sequence':     speeds,          # all speeds mentioned
        'interval_seconds':   times[0] if len(times) >= 1 else None,
        'cumulative_seconds': times[1] if len(times) >= 2 else None,
        'leg_total_seconds':  leg_total,
        'is_leg_start_c':     is_leg_start,
        'cdt_clock_time':     cdt_m.group(0) if cdt_m else None,
        'break_window_s':     break_window,
    }


def classify_col_b(text):
    code = parse_odometer_code(text)
    is_cdt  = bool(re.search(r'\bCDT\b', text, re.I)) or (code is not None and code > 50)
    is_ckpt = bool(re.search(r'hourglass|\b0\s*[|I]\s*0\s*[|I]\s*[5-9]\s*[|I]\s*0', text, re.I))
    hw_speed = parse_hw_speed_note(text)
    hw_error = parse_hw_error(text)
    return {
        'odometer_code':    code,
        'is_cdt_b':         is_cdt,
        'is_checkpoint_b':  is_ckpt,
        'hw_speed_note_b':  hw_speed,
        'hw_error_b':       hw_error,
    }


def classify_col_d(text):
    lower = text.lower()
    hw_error = parse_hw_error(text)
    ckpt_time = parse_checkpoint_time(text)
    return {
        'is_quick':           'comes quick' in lower and 'very' not in lower,
        'is_very_quick':      'comes very quick' in lower,
        'caution_noted':      'caution' in lower,
        'merge_note':         bool(re.search(r'\bmerge\b', lower)),
        'hw_error_d':         hw_error,
        'hw_checkpoint_time': ckpt_time,
        'has_hw_ckpt':        ckpt_time is not None,
    }

# ─── Page classification ──────────────────────────────────────────────────────

def is_grid_page(page_img):
    """Return True if this PDF page is a GRIID navigation table page."""
    img = np.array(page_img)
    h, w = img.shape[:2]

    # Check bottom strip for "Page N of M" marker
    bottom = img[int(h * 0.90):, :]
    results = get_reader().readtext(bottom, detail=0, paragraph=False)
    bot_text = " ".join(results)
    if re.search(r'Page\s+\d+\s+of\s+\d+', bot_text, re.I):
        return True

    # Check top strip for A B C D column headers
    top = img[:int(h * 0.10), :]
    results_top = get_reader().readtext(top, detail=0, paragraph=False)
    top_text = " ".join(results_top)
    return bool(re.search(r'\bA\b.*\bB\b.*\bC\b.*\bD\b', top_text))


def get_grid_page_number(page_img):
    """Extract (page_num, total_pages) from 'Page N of M' footer."""
    img = np.array(page_img)
    h, w = img.shape[:2]
    bottom = img[int(h * 0.88):, :]
    results = get_reader().readtext(bottom, detail=0, paragraph=False)
    text = " ".join(results)
    m = re.search(r'Page\s+(\d+)\s+of\s+(\d+)', text, re.I)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


def get_page_top_hw_number(page_img):
    """
    Extract the handwritten speedometer/target number at the top of some pages
    (e.g. '40', '35', '50').  These appear above the grid on some stage notes.
    """
    img = np.array(page_img)
    h, w = img.shape[:2]
    # Top-centre strip, above the grid
    strip = img[:int(h * 0.09), int(w * 0.2):int(w * 0.8)]
    results = get_reader().readtext(strip, detail=0, paragraph=False)
    text = " ".join(results)
    m = re.search(r'\b(\d{2,3})\b', text)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return None

# ─── Per-page extraction ──────────────────────────────────────────────────────

# Fixed column ratios (fraction of page width) — empirically determined from
# examining multiple stages.  These work well as a fallback when line detection
# is noisy.
COL_RATIOS = {
    'a_start': 0.00, 'a_end': 0.38,
    'b_start': 0.38, 'b_end': 0.52,
    'c_start': 0.52, 'c_end': 0.72,
    'd_start': 0.72, 'd_end': 1.00,
}


def get_column_bounds(v_lines, img_w):
    """
    Derive column pixel boundaries from detected vertical lines.
    Falls back to fixed ratios if lines look wrong.
    """
    # We expect ~5 lines defining 4 data columns (A B C D).
    # The leftmost column strip is the row number embedded in A.
    if len(v_lines) >= 5:
        # Pick the 5 most evenly-spaced lines
        lines = sorted(v_lines)
        # Try to find lines at roughly expected positions
        expected = [int(img_w * r) for r in [0.00, 0.38, 0.52, 0.72, 1.00]]
        picked = []
        for exp in expected:
            closest = min(lines, key=lambda x: abs(x - exp))
            picked.append(closest)
        if len(set(picked)) == 5 and picked[-1] - picked[0] > img_w * 0.5:
            return dict(
                a_start=picked[0], a_end=picked[1],
                b_start=picked[1], b_end=picked[2],
                c_start=picked[2], c_end=picked[3],
                d_start=picked[3], d_end=picked[4],
            )
    # Fallback
    return {k: int(v * img_w) for k, v in COL_RATIOS.items()}


def extract_page_instructions(page_img, stage, leg_counter, grid_page_num):
    """
    Extract all instruction rows from one grid page.
    Returns (list_of_records, updated_leg_counter).
    """
    img_np = np.array(page_img)
    h, w = img_np.shape[:2]

    # Detect horizontal grid lines
    h_lines = detect_hlines(page_img, kernel_w=max(100, int(w * 0.4)))
    if len(h_lines) < 3:
        log.warning(f"    Too few horizontal lines ({len(h_lines)}), skipping page")
        return [], leg_counter

    # Detect vertical column lines
    v_lines = detect_vlines(page_img, h_lines[0], h_lines[-1], kernel_h=max(80, int((h_lines[-1]-h_lines[0])*0.4)))
    col_bounds = get_column_bounds(v_lines, w)

    # Page-level handwritten speedometer number
    hw_page_speed = get_page_top_hw_number(page_img)

    records = []
    # Skip row 0 (header row with A B C D labels)
    row_start = 1

    for row_idx in range(row_start, len(h_lines) - 1):
        y1 = h_lines[row_idx]
        y2 = h_lines[row_idx + 1]
        if y2 - y1 < 15:
            continue

        # OCR each column with bounding boxes
        dets_a = ocr_cell(img_np, col_bounds['a_start'], y1, col_bounds['a_end'], y2)
        dets_b = ocr_cell(img_np, col_bounds['b_start'], y1, col_bounds['b_end'], y2)
        dets_c = ocr_cell(img_np, col_bounds['c_start'], y1, col_bounds['c_end'], y2)
        dets_d = ocr_cell(img_np, col_bounds['d_start'], y1, col_bounds['d_end'], y2)

        # Instruction number from bottom of column A
        instr_num = bottom_number(dets_a, threshold=0.70)

        # Raw text per column
        raw_a = cell_text(dets_a)
        raw_b = cell_text(dets_b)
        raw_c = cell_text(dets_c)
        raw_d = cell_text(dets_d)

        # Skip completely empty rows
        if not any([raw_a, raw_b, raw_c, raw_d]):
            continue

        # Feature classification
        feat_a = classify_col_a(raw_a)
        feat_b = classify_col_b(raw_b)
        feat_c = classify_col_c(raw_c)
        feat_d = classify_col_d(raw_d)

        # Leg boundary detection
        # CDT leg start: odometer_code in B (>50) OR '0m00.0s' / '*' in C
        is_leg_start = feat_b['is_cdt_b'] or feat_c['is_leg_start_c']

        # Checkpoint: hourglass (hard to detect) OR leg_total_seconds present in C
        # OR odometer code 0..50 in B (small codes = checkpoint codes)
        is_checkpoint = (feat_c['leg_total_seconds'] is not None) or feat_b['is_checkpoint_b']

        if is_leg_start:
            leg_counter += 1

        # Consolidate handwritten error from any column
        hw_error = feat_b['hw_error_b'] or feat_d['hw_error_d'] or \
                   parse_hw_error(raw_c)

        # Consolidate hw speed note from B or D
        hw_speed = feat_b['hw_speed_note_b'] or parse_hw_speed_note(raw_d)

        # Road name: first text fragment in column A that isn't the row number
        # and isn't a pure compass word
        road_name_candidates = [
            d[1] for d in sorted(dets_a, key=lambda x: x[0])
            if not re.fullmatch(r'\d{1,3}', d[1].strip())
            and len(d[1].strip()) > 1
        ]
        road_name = road_name_candidates[0] if road_name_candidates else ""

        record = {
            # ── Identity ──────────────────────────────────────────────────
            'stage':            stage,
            'grid_page':        grid_page_num,
            'instruction_num':  instr_num,
            'leg_num':          leg_counter,

            # ── Raw OCR text ───────────────────────────────────────────────
            'col_a_text': raw_a,
            'col_b_text': raw_b,
            'col_c_text': raw_c,
            'col_d_text': raw_d,

            # ── Leg boundary ───────────────────────────────────────────────
            'is_leg_start':   is_leg_start,
            'is_checkpoint':  is_checkpoint,

            # ── Timing (from column C) ─────────────────────────────────────
            'target_speed_mph':    feat_c['target_speed_mph'],
            'interval_seconds':    feat_c['interval_seconds'],
            'cumulative_seconds':  feat_c['cumulative_seconds'],
            'leg_total_seconds':   feat_c['leg_total_seconds'],
            'cdt_clock_time':      feat_c['cdt_clock_time'],
            'break_window_s':      feat_c['break_window_s'],

            # ── Column B features ──────────────────────────────────────────
            'odometer_code':   feat_b['odometer_code'],

            # ── Road / diagram features (from column A text) ───────────────
            'road_name':       road_name,
            'road_direction':  feat_a['road_direction'],
            'has_stop':        feat_a['has_stop'],
            'has_railroad':    feat_a['has_railroad'],
            'has_yield':       feat_a['has_yield'],
            'has_speed_limit': feat_a['has_speed_limit'],
            'speed_limit_mph': feat_a['speed_limit_mph'],
            'is_highway':      feat_a['is_highway'],
            'is_county_road':  feat_a['is_county_road'],
            'has_distance':    feat_a['has_distance'],
            'has_sit_stop':    feat_a['has_sit_stop'],

            # ── Column D attention flags ───────────────────────────────────
            'is_quick':          feat_d['is_quick'],
            'is_very_quick':     feat_d['is_very_quick'],
            'caution_noted':     feat_d['caution_noted'],
            'merge_note':        feat_d['merge_note'],
            'hw_checkpoint_time': feat_d['hw_checkpoint_time'],
            'has_hw_ckpt':       feat_d['has_hw_ckpt'],

            # ── Navigator handwritten annotations ─────────────────────────
            'hw_error_seconds':  hw_error,
            'hw_speed_note':     hw_speed,
            'hw_page_speedometer': hw_page_speed,
            'hw_has_note':       bool(hw_error or hw_speed or feat_d['hw_checkpoint_time']),
        }
        records.append(record)

    return records, leg_counter

# ─── Per-stage processing ─────────────────────────────────────────────────────

def process_stage(stage_num, pdf_path, dpi=250):
    log.info(f"Stage {stage_num}: {pdf_path.name}")
    pages = convert_from_path(str(pdf_path), dpi=dpi, poppler_path=POPPLER_PATH)
    log.info(f"  {len(pages)} pages total")

    all_records = []
    leg_counter = 0  # resets per stage; 0 = pre-timing

    for page_idx, page in enumerate(pages):
        # Quick check: skip non-grid pages
        if not is_grid_page(page):
            log.info(f"    Page {page_idx+1}: not a grid page, skipping")
            continue

        grid_pnum, grid_total = get_grid_page_number(page)
        log.info(f"    Page {page_idx+1} → grid page {grid_pnum}/{grid_total}")

        records, leg_counter = extract_page_instructions(
            page, stage_num, leg_counter, grid_pnum
        )
        all_records.extend(records)
        log.info(f"      {len(records)} instructions extracted  (leg_counter={leg_counter})")

    log.info(f"  Stage {stage_num} done: {len(all_records)} total instructions, "
             f"{leg_counter} legs detected")
    return all_records

# ─── Post-processing and derived features ─────────────────────────────────────

def clean_dataframe(df):
    """Enforce types and fill obvious gaps."""
    int_cols = ['stage', 'grid_page', 'instruction_num', 'leg_num', 'odometer_code']
    float_cols = [
        'target_speed_mph', 'interval_seconds', 'cumulative_seconds',
        'leg_total_seconds', 'break_window_s', 'speed_limit_mph',
        'hw_error_seconds', 'hw_page_speedometer',
    ]
    bool_cols = [
        'is_leg_start', 'is_checkpoint',
        'has_stop', 'has_railroad', 'has_yield',
        'has_speed_limit', 'is_highway', 'is_county_road',
        'has_distance', 'has_sit_stop',
        'is_quick', 'is_very_quick', 'caution_noted', 'merge_note',
        'has_hw_ckpt', 'hw_has_note',
    ]
    for c in int_cols:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors='coerce').astype('Int64')
    for c in float_cols:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors='coerce').astype('float64')
    for c in bool_cols:
        if c in df:
            df[c] = df[c].astype(bool)

    # Sort
    df = df.sort_values(['stage', 'instruction_num'], na_position='last').reset_index(drop=True)
    return df


def build_leg_characteristics(df):
    """
    Aggregate instruction-level features to (stage, leg_num) level.
    This table can be directly joined to long_format_times.parquet on
    (Stage, Leg) keys.
    """
    # Propagate target speed forward within each leg (speed stays constant until changed)
    df = df.sort_values(['stage', 'instruction_num'])
    df['speed_filled'] = (
        df.groupby(['stage', 'leg_num'])['target_speed_mph']
        .transform(lambda s: s.ffill())
    )

    agg = df.groupby(['stage', 'leg_num']).agg(
        # Count features
        instruction_count    = ('instruction_num', 'count'),
        stop_count           = ('has_stop', 'sum'),
        railroad_count       = ('has_railroad', 'sum'),
        yield_count          = ('has_yield', 'sum'),
        speed_limit_count    = ('has_speed_limit', 'sum'),
        highway_count        = ('is_highway', 'sum'),
        county_road_count    = ('is_county_road', 'sum'),
        sit_stop_count       = ('has_sit_stop', 'sum'),
        quick_count          = ('is_quick', 'sum'),
        very_quick_count     = ('is_very_quick', 'sum'),
        caution_count        = ('caution_noted', 'sum'),
        merge_count          = ('merge_note', 'sum'),
        hw_note_count        = ('hw_has_note', 'sum'),

        # Speed statistics
        speed_mean           = ('speed_filled', 'mean'),
        speed_max            = ('speed_filled', 'max'),
        speed_min            = ('speed_filled', lambda s: s.dropna().min() if s.notna().any() else None),
        speed_std            = ('speed_filled', 'std'),

        # Leg timing
        leg_total_seconds    = ('leg_total_seconds', 'max'),
        leg_cumulative_max   = ('cumulative_seconds', 'max'),

        # Handwritten corrections (sum of absolute errors the navigator noted)
        hw_error_sum_abs     = ('hw_error_seconds', lambda s: s.dropna().abs().sum()),
        hw_error_mean        = ('hw_error_seconds', 'mean'),
        hw_error_count       = ('hw_error_seconds', lambda s: s.notna().sum()),
    ).reset_index()

    # Derived ratios
    agg['stop_density']    = agg['stop_count']    / agg['instruction_count'].clip(lower=1)
    agg['quick_density']   = agg['quick_count']   / agg['instruction_count'].clip(lower=1)
    agg['highway_density'] = agg['highway_count'] / agg['instruction_count'].clip(lower=1)

    # Speed-change count: number of instructions where speed differs from previous
    # (proxy for complexity)
    def speed_change_count(group):
        speeds = group['target_speed_mph'].dropna()
        return (speeds != speeds.shift()).sum() - 1  # subtract the initial "change"

    changes = df.groupby(['stage', 'leg_num']).apply(speed_change_count).reset_index()
    changes.columns = ['stage', 'leg_num', 'speed_change_count']
    agg = agg.merge(changes, on=['stage', 'leg_num'], how='left')

    # Rename to match long_format_times.parquet keys (Stage, Leg)
    agg = agg.rename(columns={'stage': 'Stage', 'leg_num': 'Leg'})

    return agg

# ─── Checkpoint save/load (resume support) ────────────────────────────────────

def load_checkpoint():
    if CHECKPOINT_JSON.exists():
        with open(CHECKPOINT_JSON) as f:
            return json.load(f)
    return {}


def save_checkpoint(done_stages):
    with open(CHECKPOINT_JSON, 'w') as f:
        json.dump({'done_stages': done_stages}, f)

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Extract Great Race stage instructions")
    parser.add_argument('--stages', nargs='+', type=int,
                        help="Specific stage numbers to process (default: all found)")
    parser.add_argument('--dpi', type=int, default=250,
                        help="Render DPI for PDF pages (default: 250)")
    parser.add_argument('--resume', action='store_true',
                        help="Skip already-processed stages (uses checkpoint file)")
    parser.add_argument('--no-save-intermediate', action='store_true',
                        help="Don't save intermediate parquet after each stage")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)

    pdf_map = build_pdf_map(stages=args.stages)
    if not pdf_map:
        log.error("No stage PDF files found in Stage Notes directory.")
        sys.exit(1)
    log.info(f"Found PDFs for stages: {sorted(pdf_map.keys())}")

    # Resume support
    done_stages = []
    existing_records = []
    if args.resume:
        cp = load_checkpoint()
        done_stages = cp.get('done_stages', [])
        if done_stages and INSTRUCTIONS_PARQUET.exists():
            existing_df = pd.read_parquet(INSTRUCTIONS_PARQUET)
            existing_records = existing_df.to_dict('records')
            log.info(f"Resuming: already done stages {done_stages}, "
                     f"loaded {len(existing_records)} existing records")

    all_records = list(existing_records)

    for stage_num in sorted(pdf_map.keys()):
        if stage_num in done_stages:
            log.info(f"Stage {stage_num}: skipping (already done)")
            continue

        pdf_path = pdf_map[stage_num]
        try:
            records = process_stage(stage_num, pdf_path, dpi=args.dpi)
            all_records.extend(records)
            done_stages.append(stage_num)

            if not args.no_save_intermediate:
                # Save intermediate result
                df_tmp = clean_dataframe(pd.DataFrame(all_records))
                df_tmp.to_parquet(INSTRUCTIONS_PARQUET, index=False)
                save_checkpoint(done_stages)
                log.info(f"  Intermediate save: {len(df_tmp)} rows total")

        except Exception as e:
            log.error(f"Stage {stage_num} failed: {e}", exc_info=True)

    # Final assembly
    df = clean_dataframe(pd.DataFrame(all_records))
    df.to_parquet(INSTRUCTIONS_PARQUET, index=False)
    log.info(f"\nSaved {len(df)} instructions → {INSTRUCTIONS_PARQUET}")

    # Build leg-level characteristics
    df_legs = build_leg_characteristics(df)
    df_legs.to_parquet(LEG_PARQUET, index=False)
    log.info(f"Saved {len(df_legs)} leg summaries → {LEG_PARQUET}")

    # Clean up checkpoint
    if CHECKPOINT_JSON.exists():
        CHECKPOINT_JSON.unlink()

    # ── Summary report ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EXTRACTION SUMMARY")
    print("=" * 60)
    print(f"Total instructions extracted : {len(df):,}")
    print(f"Stages processed            : {sorted(df['stage'].dropna().unique().astype(int).tolist())}")
    print(f"Total legs detected         : {df['leg_num'].max()}")
    print(f"Leg starts found            : {df['is_leg_start'].sum()}")
    print(f"Checkpoints found           : {df['is_checkpoint'].sum()}")
    print()
    print("Column fill rates (instruction table):")
    check_cols = [
        'instruction_num', 'target_speed_mph', 'interval_seconds',
        'cumulative_seconds', 'road_direction', 'hw_error_seconds',
        'hw_speed_note', 'hw_page_speedometer',
    ]
    for col in check_cols:
        if col in df:
            n = df[col].notna().sum()
            print(f"  {col:<30} {n:>5}/{len(df)} ({n/len(df)*100:>4.0f}%)")
    print()
    print("Leg characteristics table:")
    for col in ['stop_count', 'speed_change_count', 'quick_count',
                'hw_error_count', 'speed_mean', 'speed_std']:
        if col in df_legs:
            valid = df_legs[col].notna().sum()
            print(f"  {col:<30} {valid:>4}/{len(df_legs)} legs have values")
    print("=" * 60)

    print(f"\nJoin key for variable importance:")
    print(f"  stage_instructions.parquet  → (stage, leg_num)")
    print(f"  leg_characteristics.parquet → (Stage, Leg)  ← matches long_format_times.parquet")


if __name__ == "__main__":
    main()
