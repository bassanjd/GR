"""
Build reference navigation charts from calibration run data.

Reads:  navigator_chart_calibration_runs.parquet
Writes: navigator_chart_normalized.xlsx

Four matrices, all using 2026 1-mile-course data:

  1. Speed Transition Time (s)
  2. Available Pause at Stop Sign (s)
  3. Turn Time Lost vs 15↔15 reference (s)
  4. Turn Time Lost vs 20↔20 reference (s)
"""
from navigator_chart_helpers import (
    build_reference_workbook,
    compute_losses,
    load_calibration_runs,
    losses_to_dicts,
)
from pathlib import Path

NORMALIZED_XLSX = Path(__file__).parent / "navigator_charts.xlsx"

def build_reference_charts():
    df = load_calibration_runs()
    losses = compute_losses(df)
    accel, decel = losses_to_dicts(losses)

    if accel is None:
        print("ERROR: Could not compute losses from 2026 data.")
        return

    wb = build_reference_workbook(accel, decel)
    wb.save(NORMALIZED_XLSX)
    print(f"Wrote {NORMALIZED_XLSX.name}")

    mph_keys = [int(m) for m in losses["MPH"]]
    print(f"\nAccel losses (s): { {k: round(accel[k], 3) for k in [0] + mph_keys} }")
    print(f"Decel losses (s): { {k: round(decel[k], 3) for k in [0] + mph_keys} }")


if __name__ == "__main__":
    build_reference_charts()
