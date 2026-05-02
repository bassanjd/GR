"""
Normalize navigator chart timing data to a single Excel table.

Reads:  2026 Great Race Charts April 29th.xlsx
Writes: navigator_chart_normalized.xlsx

Three test types over a fixed measured course:
  straight_speed  — flying start and finish (baseline)
  start_speed     — standing start to flying finish
  speed_stop      — flying start to full stop

Time entry format in source data:
  "MM:SS:cs"  colon before centiseconds (common entry typo)
  "MM:SS.cs"  dot before centiseconds (intended format)
  Both encode: MM minutes, SS whole seconds, cs hundredths of a second.
  Example: "01:10:68" and "01:10.68" both mean 1 min 10.68 sec = 70.68 s.
"""
import datetime
import re
from pathlib import Path

import openpyxl
import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.table import Table, TableStyleInfo

EXCEL_IN = Path(__file__).parent / "2026 Great Race Charts April 29th.xlsx"
EXCEL_OUT = Path(__file__).parent / "navigator_chart_normalized.xlsx"
PARQUET_OUT = Path(__file__).parent / "navigator_chart_calibration_runs.parquet"

SPEEDS_MPH = [15, 20, 25, 30, 35, 40, 45, 50]
DIRECTIONS_ALT = ['E', 'W', 'E', 'W', 'E', 'W', 'E', 'W']


# ── Parsing helpers ───────────────────────────────────────────────────────────

def parse_time_str(s):
    """Parse 'MM:SS:cs' or 'MM:SS.cs' string to decimal seconds.

    The colon variant (MM:SS:cs) is a common entry typo for the dot variant.
    Last two digits are always hundredths of a second in both formats.
    """
    if not isinstance(s, str):
        return None
    m = re.match(r'^(\d{1,2}):(\d{2})[:.](\d{2})$', s.strip())
    if m:
        mm, ss, cs = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return mm * 60 + ss + cs / 100
    return None


def time_obj_to_s(t):
    """Convert datetime.time to decimal seconds."""
    if not isinstance(t, datetime.time):
        return None
    return t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1_000_000


def fmt_mm_ss_cs(total_s):
    """Format decimal seconds as 'MM:SS.cs' (normalized dot form)."""
    if total_s is None:
        return ""
    mm = int(total_s // 60)
    rem = total_s % 60
    ss = int(rem)
    cs = round((rem - ss) * 100)
    if cs >= 100:
        cs -= 100
        ss += 1
    return f"{mm:02d}:{ss:02d}.{cs:02d}"


# ── Data extraction ───────────────────────────────────────────────────────────

def extract_rows():
    """Return a list of dicts, one per timing run."""
    wb_f = openpyxl.load_workbook(EXCEL_IN, data_only=False)
    wb_v = openpyxl.load_workbook(EXCEL_IN, data_only=True)
    rows = []

    def append(date, test_type, mph, run_num, direction, t_s, raw, notes):
        if t_s is None:
            return
        rows.append({
            "date": date,
            "test_type": test_type,
            "target_mph": mph,
            "run_number": run_num,
            "direction": direction,
            "time_raw": raw,
            "time_s": round(t_s, 2),
            "notes": notes,
        })

    # ── Straight Speed ────────────────────────────────────────────────────────

    # 2024/2025 data (rows 3–10, datetime.time values, cols B–H)
    # Column dates from row 2: B–F = 2024-04-12, G–H = 2025-04-19
    ws = wb_v["Straight Speed"]
    col_date_ss = {2: "2024-04-12", 3: "2024-04-12", 4: "2024-04-12",
                   5: "2024-04-12", 6: "2024-04-12",
                   7: "2025-04-19", 8: "2025-04-19"}
    for row, mph in zip(range(3, 11), SPEEDS_MPH):
        for col in range(2, 9):          # B–H = runs 1–7
            val = ws.cell(row, col).value
            if not isinstance(val, datetime.time):
                continue
            t_s = time_obj_to_s(val)
            append(col_date_ss[col], "straight_speed", mph,
                   col - 1, "", t_s, fmt_mm_ss_cs(t_s), "short course")

    # 2026 data (rows 21–28, raw "MM:SS:cs" strings, cols B–G = runs 1–6)
    ws_f = wb_f["Straight Speed"]
    for row, mph in zip(range(21, 29), SPEEDS_MPH):
        for col in range(2, 8):          # B–G = runs 1–6
            val = ws_f.cell(row, col).value
            t_s = parse_time_str(val)
            append("2026-04-29", "straight_speed", mph,
                   col - 1, DIRECTIONS_ALT[col - 2], t_s,
                   str(val) if val else "", "1-mile course")

    # ── Speed Stop ────────────────────────────────────────────────────────────

    # 2025 data (rows 3–10, datetime.time, cols B–H = runs 1–7)
    ws = wb_v["Speed Stop"]
    for row, mph in zip(range(3, 11), SPEEDS_MPH):
        for col in range(2, 9):
            val = ws.cell(row, col).value
            if not isinstance(val, datetime.time):
                continue
            t_s = time_obj_to_s(val)
            append("2025-04-19", "speed_stop", mph,
                   col - 1, "", t_s, fmt_mm_ss_cs(t_s), "short course")

    # 2026 data (rows 33–40, raw strings, cols B–G = runs 1–6; H is AVG)
    ws_f = wb_f["Speed Stop"]
    for row, mph in zip(range(33, 41), SPEEDS_MPH):
        for col in range(2, 8):
            val = ws_f.cell(row, col).value
            t_s = parse_time_str(val)
            append("2026-04-29", "speed_stop", mph,
                   col - 1, DIRECTIONS_ALT[col - 2], t_s,
                   str(val) if val else "", "1-mile course")

    # ── Start Speed ───────────────────────────────────────────────────────────

    # 2025 data (rows 3–10, datetime.time, cols B–G = runs 1–6; H is AVG)
    ws = wb_v["Start Speed"]
    for row, mph in zip(range(3, 11), SPEEDS_MPH):
        for col in range(2, 8):
            val = ws.cell(row, col).value
            if not isinstance(val, datetime.time):
                continue
            t_s = time_obj_to_s(val)
            append("2025-04-12", "start_speed", mph,
                   col - 1, "", t_s, fmt_mm_ss_cs(t_s), "short course")

    # 2026 data (rows 34–41, raw strings, cols B–G = runs 1–6; H is AVG)
    ws_f = wb_f["Start Speed"]
    for row, mph in zip(range(34, 42), SPEEDS_MPH):
        for col in range(2, 8):
            val = ws_f.cell(row, col).value
            t_s = parse_time_str(val)
            append("2026-04-29", "start_speed", mph,
                   col - 1, DIRECTIONS_ALT[col - 2], t_s,
                   str(val) if val else "", "1-mile course")

    return rows


# ── Derived calculations ──────────────────────────────────────────────────────

COURSE_MI = 1.0  # 2026 measured course length (highway mile markers)


def compute_summary(df):
    """Per (date, test_type, target_mph): average time and std dev."""
    agg = (df.groupby(["date", "test_type", "target_mph"])["time_s"]
             .agg(avg_time_s="mean", std_time_s="std")
             .reset_index())
    return agg


def compute_calibration(df):
    """2026 timing data: actual vs indicated speed,
    acceleration loss, and deceleration loss per indicated MPH."""
    df26 = df[df["date"] == "2026-04-29"]
    agg = (df26.groupby(["test_type", "target_mph"])["time_s"]
               .mean().unstack("test_type").reset_index())
    agg.columns.name = None

    agg["actual_mph"] = COURSE_MI / (agg["straight_speed"] / 3600)
    agg["speed_error_mph"] = agg["actual_mph"] - agg["target_mph"]
    agg["speed_error_pct"] = agg["speed_error_mph"] / agg["target_mph"] * 100
    agg["accel_loss_s"] = agg["start_speed"] - agg["straight_speed"]
    agg["decel_loss_s"] = agg["speed_stop"] - agg["straight_speed"]
    agg = agg.rename(columns={
        "straight_speed": "straight_avg_s",
        "start_speed": "start_avg_s",
        "speed_stop": "stop_avg_s",
    })
    return agg[["target_mph", "straight_avg_s", "start_avg_s", "stop_avg_s",
                "actual_mph", "speed_error_mph", "speed_error_pct",
                "accel_loss_s", "decel_loss_s"]]


# ── Excel output ──────────────────────────────────────────────────────────────

HEADER_FILL = PatternFill(start_color="1F4E79", end_color="1F4E79", fill_type="solid")
HEADER_FONT = Font(color="FFFFFF", bold=True)
ALT_FILL = PatternFill(start_color="D6E4F0", end_color="D6E4F0", fill_type="solid")


def _write_table(ws, df, table_name, col_specs, start_row=1):
    """Write a DataFrame as a formatted Excel table.

    col_specs: list of (col_name, header_label, width, num_fmt)
    """
    n_cols = len(col_specs)
    for c_idx, (_, label, width, _) in enumerate(col_specs, start=1):
        cell = ws.cell(start_row, c_idx, label)
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = Alignment(horizontal="center")
        ws.column_dimensions[get_column_letter(c_idx)].width = width

    for r_idx, row in enumerate(df.itertuples(index=False), start=start_row + 1):
        for c_idx, (col_name, _, _, num_fmt) in enumerate(col_specs, start=1):
            val = getattr(row, col_name)
            cell = ws.cell(r_idx, c_idx, val)
            if num_fmt:
                cell.number_format = num_fmt

    end_row = start_row + len(df)
    ref = f"A{start_row}:{get_column_letter(n_cols)}{end_row}"
    tbl = Table(displayName=table_name, ref=ref)
    tbl.tableStyleInfo = TableStyleInfo(
        name="TableStyleMedium2",
        showRowStripes=True, showColumnStripes=False,
    )
    ws.add_table(tbl)
    ws.freeze_panes = ws.cell(start_row + 1, 1)


def write_excel(raw_rows, df_raw):
    wb = openpyxl.Workbook()

    # ── Sheet 1: all raw runs ────────────────────────────────────────────────
    ws1 = wb.active
    ws1.title = "Calibration Runs"
    col_specs_raw = [
        ("date",        "Date",        12, None),
        ("test_type",   "Test Type",   18, None),
        ("target_mph",  "Target MPH",  12, "0"),
        ("run_number",  "Run #",       8,  "0"),
        ("direction",   "Direction",   10, None),
        ("time_raw",    "Time (raw)",  12, None),
        ("time_s",      "Time (s)",    10, "0.00"),
        ("notes",       "Notes",       18, None),
    ]
    _write_table(ws1, df_raw, "CalibrationRuns", col_specs_raw)

    # ── Sheet 2: 2026 timing data ────────────────────────────────────────────
    ws2 = wb.create_sheet("2026 Timing Data")
    cal = compute_calibration(df_raw)
    col_specs_cal = [
        ("target_mph",      "Indicated MPH",   14, "0"),
        ("straight_avg_s",  "Straight Avg (s)",16, "0.00"),
        ("start_avg_s",     "Start Avg (s)",   14, "0.00"),
        ("stop_avg_s",      "Stop Avg (s)",    14, "0.00"),
        ("actual_mph",      "Actual MPH",      12, "0.00"),
        ("speed_error_mph", "Error (mph)",     12, "0.00"),
        ("speed_error_pct", "Error (%)",       10, "0.00"),
        ("accel_loss_s",    "Accel Loss (s)",  14, "0.00"),
        ("decel_loss_s",    "Decel Loss (s)",  14, "0.00"),
    ]
    _write_table(ws2, cal, "TimingData2026", col_specs_cal)

    wb.save(EXCEL_OUT)
    return cal


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    raw_rows = extract_rows()
    df_raw = pd.DataFrame(raw_rows)

    print(f"\nExtracted {len(df_raw)} timing runs\n")
    print(df_raw.groupby(["date", "test_type"]).size().to_string())

    cal = write_excel(raw_rows, df_raw)

    print(f"\nWrote {EXCEL_OUT.name}")
    try:
        df_raw.to_parquet(PARQUET_OUT, index=False)
        print(f"Wrote {PARQUET_OUT.name}")
    except Exception as e:
        print(f"Warning: failed to write parquet {PARQUET_OUT}: {e}")
    print("\n2026 Timing Data:")
    print(cal.to_string(index=False, float_format="%.2f"))
