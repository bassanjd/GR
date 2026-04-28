"""Convert an AIM XRK datalog file to a Parquet file.

All channels (data, GPS, GPS-raw) are exported and aligned to a common time
axis via outer-join + linear interpolation.  SI units are converted to US
customary (°F, mph, ft, mi, psi) automatically.


https://github.com/laz-/xrk


"""

import sys
from pathlib import Path

# xrk.py lives alongside this script
sys.path.insert(0, str(Path(__file__).parent))

import xrk
import numpy as np
import pandas as pd

AIM_DIR     = Path(__file__).parent.parent / "DataLogs" / "AIM"
DATAPARQUET = Path(__file__).parent.parent / "DataParquet"

TARGET_HZ = 25  # output sample rate — change to 25 or 100 if you need higher resolution

# Multiplicative conversions: source_unit -> (factor, us_unit)
_MUL = {
    # Speed
    "km/h":  (0.621371,         "mph"),
    "kph":   (0.621371,         "mph"),
    "m/s":   (2.23694,          "mph"),
    # Distance — meters → feet; kilometers → miles; centimeters → inches
    "m":     (3.28084,          "ft"),
    "km":    (0.621371,         "mi"),
    "cm":    (0.393701,         "in"),
    # Acceleration
    "m/s^2": (1.0 / 9.80665,   "g"),
    "m/s2":  (1.0 / 9.80665,   "g"),
    # Pressure
    "bar":   (14.5038,          "psi"),
    "mbar":  (0.0145038,        "psi"),
    "kPa":   (0.145038,         "psi"),
    "Pa":    (0.000145038,      "psi"),
}

# Affine conversions (value * factor + offset): source_unit -> (factor, offset, us_unit)
_AFF = {
    "C":     (9 / 5, 32, "°F"),
    "°C":    (9 / 5, 32, "°F"),
    "deg C": (9 / 5, 32, "°F"),
}


def convert_units(values: np.ndarray, units: str) -> tuple[np.ndarray, str]:
    """Return (converted_values, new_unit_string), or the originals if no rule matches."""
    key = units.strip()
    if key in _MUL:
        factor, new_unit = _MUL[key]
        return values * factor, new_unit
    if key in _AFF:
        factor, offset, new_unit = _AFF[key]
        return values * factor + offset, new_unit
    return values, units


def channels_to_dataframe(myxrk: xrk.XRK, hz: int = TARGET_HZ) -> pd.DataFrame:
    """Resample all channels onto a uniform time grid at `hz` samples/second."""
    session_end = sum(dur for _, dur in myxrk.lap_info)
    grid = np.arange(0.0, session_end, 1.0 / hz)

    session_start = pd.Timestamp(myxrk.datetime)
    combined = pd.DataFrame({
        "Elapsed time (sec)": grid,
        "Datetime": session_start + pd.to_timedelta(grid, unit="s"),
    })
    for name, ch in myxrk.channels.items():
        try:
            times, values = ch.samples(xtime=True, xabsolute=True)
            raw_units = ch.units()
            # np.interp clamps to boundary values outside the channel's range
            interp_values = np.interp(grid, times, values)
            converted, display_units = convert_units(interp_values, raw_units)
            col = f"{name} ({display_units})" if display_units else name
            combined[col] = converted
            suffix = f"  [{raw_units} -> {display_units}]" if display_units != raw_units else f"  [{raw_units}]"
            print(f"  {name}: {len(times)} samples{suffix}")
        except Exception as exc:
            print(f"  {name}: SKIPPED — {exc}")

    return combined


def main():
    xrk_files = sorted(AIM_DIR.glob("*.xrk"))
    if not xrk_files:
        print(f"No .xrk files found in {AIM_DIR}")
        return
    for xrk_file in xrk_files:
        parquet_file = DATAPARQUET / xrk_file.with_suffix(".parquet").name
        print(f"Opening {xrk_file.name} …")
        myxrk = xrk.XRK(str(xrk_file))
        print(myxrk.summary())
        print(f"Channels ({len(myxrk.channels)}):")
        df = channels_to_dataframe(myxrk)
        df.to_parquet(parquet_file, index=False)
        print(f"\nWrote {len(df):,} rows x {len(df.columns)} columns -> {parquet_file.name}")
        print(df.head())
        myxrk.close()


if __name__ == "__main__":
    main()
