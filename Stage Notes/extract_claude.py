"""
extract_claude.py

Extracts structured data from Great Race stage notes PDFs using Claude's
vision capability.  Results are cached per-page as JSON so the expensive
API call is made only once per page — subsequent runs assemble from cache
in seconds.

Outputs
-------
Output/claude_cache/stage_N_page_M.json  – raw Claude response per page
Output/stage_instructions.parquet        – one row per grid instruction
Output/leg_characteristics.parquet       – one row per (Stage, Leg)

Usage
-----
    python extract_claude.py                        # all stages
    python extract_claude.py --stages 1 2           # specific stages
    python extract_claude.py --stages 1 --pages 1 6 # specific pages
    python extract_claude.py --assemble-only        # rebuild parquets from cache

Future daily-coaching use
-------------------------
    python extract_claude.py --scan path/to/scan.jpg --stage 3
"""

import argparse
import base64
import io
import json
import logging
import re
import sys
from pathlib import Path

# Load .env file if present (before any API clients are initialized)
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

import pandas as pd
from pdf2image import convert_from_path
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE          = Path(__file__).parent
STAGE_DIR     = BASE / "Stage Notes"
OUTPUT_DIR    = BASE / "Output"
CACHE_DIR     = OUTPUT_DIR / "claude_cache"
INST_PARQUET  = OUTPUT_DIR / "stage_instructions.parquet"
LEG_PARQUET   = OUTPUT_DIR / "leg_characteristics.parquet"
POPPLER_PATH  = r"C:\Program Files (x86)\poppler\Library\bin"
DPI           = 200      # lower than before — Claude reads images well at lower res
MAX_PX        = 1400     # max image dimension sent to API (saves tokens)

# ─── PDF map ──────────────────────────────────────────────────────────────────
def build_pdf_map(stages=None):
    pdfs = {}
    for s in range(0, 10):
        if stages and s not in stages:
            continue
        clean   = STAGE_DIR / f"2025 Great Race Stage {s} - Clean.pdf"
        regular = STAGE_DIR / f"2025 Great Race Stage {s}.pdf"
        trophy  = STAGE_DIR / "2025 Great Race Trophy Run.pdf"
        if   s == 0 and trophy.exists():  pdfs[0] = trophy
        elif clean.exists():              pdfs[s] = clean
        elif regular.exists():            pdfs[s] = regular
    return pdfs

# ─── Image helpers ────────────────────────────────────────────────────────────
def pil_to_b64(img: Image.Image, max_px=MAX_PX) -> str:
    """Resize if needed, encode as base64 JPEG."""
    w, h = img.size
    if max(w, h) > max_px:
        scale = max_px / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.standard_b64encode(buf.getvalue()).decode()


def is_grid_page_quick(img: Image.Image) -> bool:
    """
    Fast heuristic: grid pages have dense black lines forming a table.
    Count horizontal black pixel runs across the middle of the image.
    """
    import numpy as np
    gray = img.convert("L")
    arr  = np.array(gray)
    mid  = arr[arr.shape[0]//2, :]
    dark = (mid < 80).sum()
    return dark > img.width * 0.03   # at least 3% dark pixels in middle row

# ─── Claude API call ──────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a precise data-extraction assistant for motorsport navigation notes.
You will receive an image of one page from a Great Race navigation instruction booklet.
Extract the data EXACTLY as described — no interpretation, no invention.
Return ONLY valid JSON, no markdown, no commentary."""

PAGE_PROMPT = """This is Stage {stage}, grid page {grid_page} of the Great Race 2025 navigation booklet.

The page contains a table with columns: [row#] | A | B | C | D
- Column A: navigation diagram (arrow drawings) + road/location text + instruction number at bottom-left
- Column B: timing markers (CDT clock = leg start; hourglass = checkpoint; handwritten speed corrections)
- Column C: timing data (target speed, interval time, cumulative time; leg total at checkpoint)
- Column D: navigation warnings, handwritten error corrections

DIAGRAM TYPES for col_a_diagram (pick the best match):
STRAIGHT, TURN_LEFT, TURN_RIGHT, BEAR_LEFT, BEAR_RIGHT,
HIGHWAY_ENTER, HIGHWAY_EXIT, MERGE, STOP_SIGN, TRAFFIC_LIGHT,
RAILROAD, ROUNDABOUT, YIELD_SIGN, SPEED_LIMIT, CURVE_WARNING,
CONSTRUCTION, GAS_STATION, RESTROOM, LUNCH_BREAK, HOTEL,
FINISH_FLAG, INFO_BOX, TEXT_BOX, OTHER

CDT LEG START row: Column B has a speedometer/tire icon + 4-digit code like "0|2|8|0".
  Column C has: target speed (e.g. "50 MPH"), leg duration (e.g. "34m00s"), asterisk "*", and "0m00.0s"
CHECKPOINT row: Column B has an hourglass icon + 4-digit code like "0|0|7|0".
  Column C has: interval time, cumulative time, and leg TOTAL time (integer seconds like "26m00s" with NO decimal).
  The leg total is always of the form "NNm00s" (whole minutes, zero seconds).

SIT STOP sequence in C: "0 MPH  sit  0m15s  N.N  45 MPH"
  - "sit" = navigator abbreviation for a stop-and-go
  - The decimal number (N.N) after "0m15s" is the navigator's HANDWRITTEN observed sit time in seconds
  - Extract it as sit_time_observed_s

HANDWRITTEN content is usually at an angle, rougher font, or circled/boxed.
PRINTED content is clean typeset.

Extract every visible instruction row. Skip non-instruction rows (large text boxes, QR codes, etc.) but note them in page_notes.

Return JSON matching this EXACT schema:
{{
  "stage": <int>,
  "grid_page": <int or null>,
  "page_top_hw_number": <float or null>,   // handwritten number at top of page (speedometer tracking)
  "page_notes": "<string>",                // any whole-page notes (e.g. "Time allowance submission box after instr 152")
  "instructions": [
    {{
      "num": <int or null>,                // instruction number (bottom-left of col A)
      "a_road": "<string>",                // road name / landmark text in col A (clean text only, no diagram description)
      "a_diagram": "<DIAGRAM_TYPE>",       // diagram icon type from list above
      "a_turn": "<LEFT|RIGHT|STRAIGHT|BEAR_LEFT|BEAR_RIGHT|U_TURN|null>",  // arrow direction
      "b_printed": "<string>",             // printed text in col B
      "b_handwritten": "<string>",         // handwritten text in col B
      "b_odometer": <int or null>,         // 4-digit code as integer: "0|2|8|0" -> 280
      "b_is_cdt": <bool>,                  // true if CDT/tire icon present (leg start)
      "b_is_checkpoint": <bool>,           // true if hourglass icon present
      "c_target_mph": <float or null>,     // target speed in MPH
      "c_leg_duration_s": <float or null>, // leg total allowed seconds (at leg start row only)
      "c_interval_s": <float or null>,     // segment interval seconds (e.g. "1m28.9s" -> 88.9)
      "c_cumulative_s": <float or null>,   // cumulative leg seconds (e.g. "4m25.4s" -> 265.4)
      "c_leg_total_s": <float or null>,    // leg total allowed seconds at CHECKPOINT (e.g. "26m00s" -> 1560)
      "c_handwritten": "<string>",         // any handwritten content in col C
      "c_break_s": <float or null>,        // rest break window e.g. "(6m00s)" -> 360
      "c_is_leg_start": <bool>,            // true if this row has the CDT start markers in C
      "sit_time_observed_s": <float or null>, // navigator's handwritten observed sit time
      "d_printed": "<string>",
      "d_handwritten": "<string>",
      "d_is_quick": <bool>,                // "comes quick" in D
      "d_is_very_quick": <bool>,           // "comes very quick" in D
      "d_caution": <bool>,
      "d_merge": <bool>,
      "d_hw_error_s": <float or null>,     // signed error value e.g. -3.7
      "d_hw_ckpt_time": "<string or null>",// handwritten checkpoint time e.g. "3:08:06"
      "row_highlighted": <bool>,           // yellow/orange highlight band visible
      "hw_speed_correction": "<string or null>"  // e.g. "42 MPH FOR 20 SEC" from B or D
    }}
  ]
}}"""


def call_claude(img_b64: str, stage: int, grid_page: int) -> dict:
    """Send one page image to Claude, return parsed JSON dict."""
    import anthropic
    client = anthropic.Anthropic()

    prompt = PAGE_PROMPT.format(stage=stage, grid_page=grid_page)

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": img_b64,
                    },
                },
                {"type": "text", "text": prompt},
            ],
        }],
    )

    raw = response.content[0].text.strip()
    # Strip any markdown code fences if present
    raw = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.MULTILINE)
    raw = re.sub(r'\s*```$', '', raw, flags=re.MULTILINE)
    return json.loads(raw)

# ─── Cache helpers ────────────────────────────────────────────────────────────
def cache_path(stage: int, grid_page: int) -> Path:
    return CACHE_DIR / f"stage_{stage}_page_{grid_page:03d}.json"


def load_cache(stage: int, grid_page: int) -> dict | None:
    p = cache_path(stage, grid_page)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


def save_cache(stage: int, grid_page: int, data: dict):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p = cache_path(stage, grid_page)
    with open(p, "w") as f:
        json.dump(data, f, indent=2)

# ─── Stage processing ─────────────────────────────────────────────────────────
def infer_grid_page(page_idx: int) -> int | None:
    """
    Rough heuristic: pages 0,1,2 are cover/start-order/emergency (skip).
    Grid pages start at 3.  Returns 1-based grid page number.
    """
    if page_idx < 3:
        return None
    return page_idx - 2   # page 3 → grid 1


def get_page_count(pdf_path: Path) -> int:
    """Return total page count without loading image data."""
    from pdf2image.pdf2image import pdfinfo_from_path
    info = pdfinfo_from_path(str(pdf_path), poppler_path=POPPLER_PATH)
    return info["Pages"]


def process_stage(stage: int, pdf_path: Path, dpi: int = DPI,
                  page_filter: list[int] | None = None,
                  no_grid_filter: bool = False):
    """Extract all grid pages from one stage PDF, using cache where available.
    Pages are loaded one at a time to keep memory usage low."""
    log.info(f"Stage {stage}: {pdf_path.name}")
    total_pages = get_page_count(pdf_path)
    log.info(f"  {total_pages} PDF pages")

    results = []
    for pdf_idx in range(total_pages):
        grid_page = infer_grid_page(pdf_idx)
        if grid_page is None:
            log.info(f"  PDF page {pdf_idx+1}: skipping (cover/start-order/emergency)")
            continue

        if page_filter and grid_page not in page_filter:
            continue

        # Check cache before loading the image at all
        cached = load_cache(stage, grid_page)
        if cached is not None:
            log.info(f"  PDF page {pdf_idx+1} (grid {grid_page}): loaded from cache")
            results.append(cached)
            continue

        # Load just this one page (1-based for pdf2image)
        page_pil = convert_from_path(
            str(pdf_path), dpi=dpi, poppler_path=POPPLER_PATH,
            first_page=pdf_idx + 1, last_page=pdf_idx + 1
        )[0]

        # Quick visual check — skip obviously non-grid pages (e.g. solid-colour cover)
        if not no_grid_filter and not is_grid_page_quick(page_pil):
            log.info(f"  PDF page {pdf_idx+1} (grid {grid_page}): looks non-grid, skipping")
            continue

        # Call Claude
        log.info(f"  PDF page {pdf_idx+1} (grid {grid_page}): calling Claude API…")
        img_b64 = pil_to_b64(page_pil)
        del page_pil  # free memory immediately after encoding
        try:
            data = call_claude(img_b64, stage, grid_page)
            data["stage"]     = stage
            data["grid_page"] = grid_page
            save_cache(stage, grid_page, data)
            results.append(data)
            log.info(f"    → {len(data.get('instructions', []))} instructions")
        except Exception as e:
            log.error(f"    Claude call failed: {e}")
            save_cache(stage, grid_page, {"stage": stage, "grid_page": grid_page,
                                          "error": str(e), "instructions": []})

    return results


def process_scan(scan_path: Path, stage: int) -> dict:
    """Process a single scanned image (daily coaching workflow)."""
    img = Image.open(scan_path).convert("RGB")
    img_b64 = pil_to_b64(img)
    grid_page = 99  # placeholder
    log.info(f"Processing scan: {scan_path.name}")
    data = call_claude(img_b64, stage, grid_page)
    data["stage"] = stage
    data["grid_page"] = grid_page
    # Cache under scan filename
    out = CACHE_DIR / f"scan_{scan_path.stem}.json"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(data, f, indent=2)
    log.info(f"Cached → {out}")
    return data

# ─── Assemble parquets from cache ─────────────────────────────────────────────

def assemble_instructions(stage_filter=None) -> pd.DataFrame:
    """Read all cache JSON files and flatten into instruction-level DataFrame."""
    rows = []
    leg_counter: dict[int, int] = {}  # stage → current leg number

    files = sorted(CACHE_DIR.glob("stage_*_page_*.json"))
    for fp in files:
        m = re.match(r"stage_(\d+)_page_(\d+)", fp.stem)
        if not m:
            continue
        stage = int(m.group(1))
        if stage_filter and stage not in stage_filter:
            continue

        with open(fp) as f:
            page_data = json.load(f)

        if "error" in page_data and not page_data.get("instructions"):
            log.warning(f"Skipping errored cache: {fp.name}")
            continue

        if stage not in leg_counter:
            leg_counter[stage] = 0

        hw_page_speed = page_data.get("page_top_hw_number")

        for instr in page_data.get("instructions", []):
            # Advance leg counter when we hit a CDT start.
            # Claude sometimes labels a CDT row as b_is_checkpoint when the icon
            # is ambiguous — use odometer code as tiebreaker: code > 50 means CDT.
            odo = instr.get("b_odometer")
            odo_says_cdt = (odo is not None and odo > 50)
            is_leg_start = (
                instr.get("b_is_cdt", False)
                or instr.get("c_is_leg_start", False)
                or (instr.get("b_is_checkpoint", False) and odo_says_cdt)
            )
            if is_leg_start:
                leg_counter[stage] += 1

            row = {
                # Identity
                "stage":           stage,
                "grid_page":       page_data.get("grid_page"),
                "instruction_num": instr.get("num"),
                "leg_num":         leg_counter[stage],

                # Column A
                "road_name":    instr.get("a_road", ""),
                "diagram_type": instr.get("a_diagram", ""),
                "turn_dir":     instr.get("a_turn"),

                # Column B
                "col_b_printed":     instr.get("b_printed", ""),
                "col_b_handwritten": instr.get("b_handwritten", ""),
                "odometer_code":     instr.get("b_odometer"),
                "b_is_cdt":          instr.get("b_is_cdt", False),
                "b_is_checkpoint":   instr.get("b_is_checkpoint", False),
                "hw_speed_correction": instr.get("hw_speed_correction"),

                # Column C (timing)
                "target_speed_mph":  instr.get("c_target_mph"),
                "leg_duration_s":    instr.get("c_leg_duration_s"),
                "interval_seconds":  instr.get("c_interval_s"),
                "cumulative_seconds": instr.get("c_cumulative_s"),
                "leg_total_seconds": instr.get("c_leg_total_s"),
                "col_c_handwritten": instr.get("c_handwritten", ""),
                "break_window_s":    instr.get("c_break_s"),
                "is_leg_start":      is_leg_start,
                "sit_time_observed_s": instr.get("sit_time_observed_s"),

                # Column D
                "col_d_printed":     instr.get("d_printed", ""),
                "col_d_handwritten": instr.get("d_handwritten", ""),
                "is_quick":          instr.get("d_is_quick", False),
                "is_very_quick":     instr.get("d_is_very_quick", False),
                "caution_noted":     instr.get("d_caution", False),
                "merge_note":        instr.get("d_merge", False),
                "hw_error_seconds":  instr.get("d_hw_error_s"),
                "hw_checkpoint_time": instr.get("d_hw_ckpt_time"),

                # Row-level flags
                "is_checkpoint":     instr.get("b_is_checkpoint", False),
                "row_highlighted":   instr.get("row_highlighted", False),
                "hw_page_speedometer": hw_page_speed,
                "hw_has_note": bool(
                    instr.get("d_hw_error_s") or
                    instr.get("d_hw_ckpt_time") or
                    instr.get("b_handwritten") or
                    instr.get("hw_speed_correction")
                ),
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        log.warning("No instruction data found in cache.")
        return df

    # Type enforcement
    int_cols   = ["stage", "grid_page", "instruction_num", "leg_num", "odometer_code"]
    float_cols = ["target_speed_mph", "leg_duration_s", "interval_seconds",
                  "cumulative_seconds", "leg_total_seconds", "break_window_s",
                  "hw_error_seconds", "hw_page_speedometer", "sit_time_observed_s"]
    bool_cols  = ["b_is_cdt", "b_is_checkpoint", "is_leg_start", "is_checkpoint",
                  "is_quick", "is_very_quick", "caution_noted", "merge_note",
                  "row_highlighted", "hw_has_note"]

    for c in int_cols:
        if c in df: df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    for c in float_cols:
        if c in df: df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    for c in bool_cols:
        if c in df: df[c] = df[c].fillna(False).astype(bool)

    df = df.sort_values(["stage", "instruction_num"], na_position="last").reset_index(drop=True)
    return df


def build_leg_characteristics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate instruction-level features to (Stage, Leg) level for variable
    importance analysis.  Join key matches long_format_times.parquet on
    Stage and Leg columns.
    """
    if df.empty:
        return pd.DataFrame()

    # Forward-fill target speed within each leg (speed holds until next change)
    df = df.sort_values(["stage", "instruction_num"])
    df["speed_filled"] = (
        df.groupby(["stage", "leg_num"])["target_speed_mph"]
        .transform(lambda s: s.ffill())
    )

    # Diagram type dummies
    diagram_types = [
        "TURN_LEFT", "TURN_RIGHT", "BEAR_LEFT", "BEAR_RIGHT",
        "STOP_SIGN", "TRAFFIC_LIGHT", "RAILROAD", "ROUNDABOUT",
        "YIELD_SIGN", "SPEED_LIMIT", "CURVE_WARNING", "CONSTRUCTION",
        "HIGHWAY_ENTER", "HIGHWAY_EXIT", "MERGE", "STRAIGHT",
    ]
    for dt in diagram_types:
        col = f"diag_{dt.lower()}"
        df[col] = (df["diagram_type"] == dt)

    # Speed change count per leg
    def speed_changes(grp):
        s = grp["target_speed_mph"].dropna()
        return int((s != s.shift()).sum()) - 1 if len(s) > 0 else 0

    sc = df.groupby(["stage", "leg_num"]).apply(speed_changes).reset_index()
    sc.columns = ["stage", "leg_num", "speed_change_count"]

    agg_dict = {
        "instruction_count":   ("instruction_num", "count"),
        # Road features
        "stop_count":          ("diag_stop_sign", "sum"),
        "traffic_light_count": ("diag_traffic_light", "sum"),
        "railroad_count":      ("diag_railroad", "sum"),
        "roundabout_count":    ("diag_roundabout", "sum"),
        "yield_count":         ("diag_yield_sign", "sum"),
        "curve_warning_count": ("diag_curve_warning", "sum"),
        "highway_enter_count": ("diag_highway_enter", "sum"),
        "highway_exit_count":  ("diag_highway_exit", "sum"),
        "merge_count_diag":    ("diag_merge", "sum"),
        "straight_count":      ("diag_straight", "sum"),
        "turn_left_count":     ("diag_turn_left", "sum"),
        "turn_right_count":    ("diag_turn_right", "sum"),
        # Attention flags
        "quick_count":         ("is_quick", "sum"),
        "very_quick_count":    ("is_very_quick", "sum"),
        "caution_count":       ("caution_noted", "sum"),
        "highlighted_count":   ("row_highlighted", "sum"),
        "sit_stop_count":      ("sit_time_observed_s", lambda x: x.notna().sum()),
        # Speed stats
        "speed_mean":          ("speed_filled", "mean"),
        "speed_max":           ("speed_filled", "max"),
        "speed_min":           ("speed_filled", lambda x: x.dropna().min() if x.notna().any() else None),
        "speed_std":           ("speed_filled", "std"),
        # Timing
        "leg_duration_s":      ("leg_duration_s", "max"),
        "leg_total_seconds":   ("leg_total_seconds", "max"),
        # Navigator corrections
        "hw_error_count":      ("hw_error_seconds", lambda x: x.notna().sum()),
        "hw_error_sum_abs":    ("hw_error_seconds", lambda x: x.dropna().abs().sum()),
        "hw_error_mean":       ("hw_error_seconds", "mean"),
        "hw_note_count":       ("hw_has_note", "sum"),
        "sit_time_total_s":    ("sit_time_observed_s", lambda x: x.dropna().sum()),
    }

    agg = df.groupby(["stage", "leg_num"]).agg(**agg_dict).reset_index()
    agg = agg.merge(sc, on=["stage", "leg_num"], how="left")

    # Derived densities
    n = agg["instruction_count"].clip(lower=1)
    agg["stop_density"]        = agg["stop_count"] / n
    agg["turn_density"]        = (agg["turn_left_count"] + agg["turn_right_count"]) / n
    agg["quick_density"]       = agg["quick_count"] / n
    agg["highway_density"]     = (agg["highway_enter_count"] + agg["highway_exit_count"]) / n
    agg["speed_change_density"]= agg["speed_change_count"] / n

    # Rename to match long_format_times.parquet join keys
    agg = agg.rename(columns={"stage": "Stage", "leg_num": "Leg"})
    return agg

# ─── Reporting ────────────────────────────────────────────────────────────────

def print_quality_report(df: pd.DataFrame):
    print("\n" + "=" * 65)
    print("EXTRACTION QUALITY REPORT")
    print("=" * 65)
    print(f"Total instructions : {len(df):,}")
    if df.empty:
        return
    print(f"Stages             : {sorted(df['stage'].dropna().unique().astype(int).tolist())}")
    leg_max = int(df['leg_num'].max()) if df['leg_num'].notna().any() else 0
    print(f"Legs detected      : {leg_max}")
    print(f"Leg starts         : {df['is_leg_start'].sum()}")
    print(f"Checkpoints        : {df['is_checkpoint'].sum()}")
    print()
    print("Key column fill rates:")
    check = [
        "instruction_num", "target_speed_mph", "interval_seconds",
        "cumulative_seconds", "diagram_type", "turn_dir",
        "hw_error_seconds", "hw_speed_correction", "hw_page_speedometer",
    ]
    for c in check:
        if c not in df.columns:
            continue
        valid = df[c].notna().sum() if df[c].dtype != bool else (df[c] != "").sum()
        pct   = valid / len(df) * 100
        bar   = "#" * int(pct / 5)
        print(f"  {c:<30} {pct:>5.0f}%  {bar}")
    print()
    print("Diagram type distribution:")
    if "diagram_type" in df.columns:
        top = df["diagram_type"].value_counts().head(15)
        for dt, cnt in top.items():
            print(f"  {dt:<25} {cnt:>4}")
    print("=" * 65)

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Extract Great Race stage notes via Claude")
    parser.add_argument("--stages",       nargs="+", type=int)
    parser.add_argument("--pages",        nargs="+", type=int, help="Specific grid page numbers")
    parser.add_argument("--dpi",          type=int, default=DPI)
    parser.add_argument("--assemble-only",action="store_true",
                        help="Skip API calls; rebuild parquets from existing cache")
    parser.add_argument("--force",        action="store_true",
                        help="Re-call Claude even if cache exists")
    parser.add_argument("--scan",         type=str, help="Path to a single scanned image")
    parser.add_argument("--no-grid-filter", action="store_true",
                        help="Disable non-grid page heuristic; send every uncached page to Claude")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)
    CACHE_DIR.mkdir(exist_ok=True)

    # ── Single-scan daily coaching mode ───────────────────────────────────────
    if args.scan:
        if not args.stages or len(args.stages) != 1:
            log.error("--scan requires exactly one --stages N argument")
            sys.exit(1)
        data = process_scan(Path(args.scan), args.stages[0])
        print(json.dumps(data, indent=2))
        return

    # ── Assemble-only mode ────────────────────────────────────────────────────
    if not args.assemble_only:
        pdf_map = build_pdf_map(stages=args.stages)
        if not pdf_map:
            log.error("No stage PDF files found.")
            sys.exit(1)
        log.info(f"PDFs to process: stages {sorted(pdf_map.keys())}")

        if args.force:
            # Delete matching cache files so they get re-generated
            for stage in pdf_map:
                for fp in CACHE_DIR.glob(f"stage_{stage}_page_*.json"):
                    fp.unlink()

        for stage, pdf_path in sorted(pdf_map.items()):
            process_stage(stage, pdf_path, dpi=args.dpi, page_filter=args.pages,
                          no_grid_filter=args.no_grid_filter)

    # ── Build parquets from cache ──────────────────────────────────────────────
    log.info("Assembling parquets from cache…")
    df = assemble_instructions(stage_filter=args.stages)

    if df.empty:
        log.warning("No data assembled — check cache directory.")
        return

    df.to_parquet(INST_PARQUET, index=False)
    log.info(f"Saved {len(df):,} instructions → {INST_PARQUET}")

    df_legs = build_leg_characteristics(df)
    df_legs.to_parquet(LEG_PARQUET, index=False)
    log.info(f"Saved {len(df_legs)} leg summaries → {LEG_PARQUET}")

    print_quality_report(df)

    print(f"\nVariable importance join:")
    print(f"  leg_characteristics.parquet (Stage, Leg)")
    print(f"  long_format_times.parquet   (Stage, Leg)")


if __name__ == "__main__":
    main()
