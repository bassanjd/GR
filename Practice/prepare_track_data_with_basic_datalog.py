"""
prepare_track_data_with_basic_datalog.py
Converts speed-tracker CSV/.numbers files to privacy-filtered parquet files,
integrating temperature readings from datestamp CSV datalogger files.

Datestamp files are grouped by date:
  - Speed tracker CSV (standard encoding, has "Elapsed time (sec)" column)
    named: YYYY-MM-DD *.csv
  - Speed tracker Numbers (Apple .numbers, LSA-GPS format with Lat/Lon/Speed)
    named: *M-D-YY*.numbers  (e.g. "LSA-GPS 3-10-26.numbers")
  - Temp logger CSV  (UTF-16 encoding, has CH1-CH4 temperature columns)
    named: YYYY-MM-DD *.csv

Temperature readings are aligned to speed tracker rows by nearest timestamp.
Rows north of LAT_MAX are dropped to hide the starting/home location.

Usage:
    python prepare_track_data_with_basic_datalog.py              # process all datestamp files
    python prepare_track_data_with_basic_datalog.py --force      # re-generate even if parquet exists
"""
import sys
import re
from io import StringIO
from pathlib import Path
import pandas as pd

DATALOGS = Path(__file__).parent / "DataLogs"
LAT_MAX = 29.66716  # drop everything north of this latitude
DATE_PREFIX = re.compile(r'^\d{4}-\d{2}-\d{2} ')
DATE_MDY = re.compile(r'(\d{1,2})-(\d{1,2})-(\d{2})\b')  # M-D-YY in filename

# Per-date channel label overrides: {date_key: {old_col: new_col}}
CHANNEL_LABELS: dict[str, dict[str, str]] = {
    "2026-03-07": {
        "CH1": "Mechanical fuel pump",
        "CH2": "Fuel line near the engine",
        "CH3": "Top of carburetors",
        "CH4": "Exhaust manifold",
    },
}


def find_datestamp_files(datalogs: Path) -> dict:
    """Return {date_key: [Path, ...]} for all datestamp CSVs and .numbers files."""
    groups: dict[str, list[Path]] = {}
    for f in sorted(datalogs.glob("*.csv")):
        if DATE_PREFIX.match(f.name):
            date_key = f.name[:10]   # "YYYY-MM-DD"
            groups.setdefault(date_key, []).append(f)
    for f in sorted(datalogs.glob("*.numbers")):
        m = DATE_MDY.search(f.name)
        if m:
            month, day, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
            date_key = f"20{year:02d}-{month:02d}-{day:02d}"
            groups.setdefault(date_key, []).append(f)
    return groups


def classify_file(path: Path) -> str:
    """Return 'temp_logger' if UTF-16 BOM detected, 'numbers_tracker' for .numbers, else 'speed_tracker'."""
    if path.suffix == '.numbers':
        return 'numbers_tracker'
    bom = path.read_bytes()[:2]
    return 'temp_logger' if bom in (b'\xff\xfe', b'\xfe\xff') else 'speed_tracker'


def parse_speed_tracker(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    df = df.sort_values("Elapsed time (sec)")
    df = df.drop_duplicates(subset="Elapsed time (sec)", keep="first")

    # Parse wall-clock datetime for merging.
    # Date: "07/03/26" (DD/MM/YY), Time: "11:23 AM GMT-6"
    time_clean = df["Time"].str.replace(r'\s+GMT[+-]\d+', '', regex=True).str.strip()
    dt_str = df["Date"].str.strip() + " " + time_clean
    df["_datetime"] = pd.to_datetime(dt_str, format="%d/%m/%y %I:%M %p")
    return df


def _haversine_mi(lat1, lon1, lat2, lon2):
    """Return distance in miles between two lat/lon points."""
    import math
    R = 3958.8  # Earth radius in miles
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))


def parse_numbers_tracker(numbers_path: Path) -> pd.DataFrame:
    """Parse an LSA-GPS .numbers file into the same shape as parse_speed_tracker."""
    from numbers_parser import Document
    import numpy as np
    doc = Document(numbers_path)
    rows = list(doc.sheets[0].tables[0].rows(values_only=True))
    df = pd.DataFrame(rows[1:], columns=rows[0])

    df['_datetime'] = pd.to_datetime(df['Position_Date'], format='mixed')
    df = df.sort_values('_datetime')
    df = df.drop_duplicates(subset='_datetime', keep='first').reset_index(drop=True)

    # Compute elapsed seconds so downstream code can sort/dedup consistently.
    t0 = df['_datetime'].iloc[0]
    df['Elapsed time (sec)'] = (df['_datetime'] - t0).dt.total_seconds()

    # Normalize column names to match speed-tracker conventions.
    df = df.rename(columns={'Lat': 'Latitude', 'Lon': 'Longitude', 'Speed': 'Speed (mph)'})

    # Compute cumulative distance in miles via haversine.
    lats = df['Latitude'].to_numpy()
    lons = df['Longitude'].to_numpy()
    dists = np.zeros(len(df))
    for i in range(1, len(df)):
        dists[i] = dists[i-1] + _haversine_mi(lats[i-1], lons[i-1], lats[i], lons[i])
    df['Distance (mi)'] = dists

    return df


def parse_temp_logger(csv_path: Path) -> pd.DataFrame:
    content = csv_path.read_bytes().decode('utf-16')
    lines = content.splitlines()

    # Locate the tab-separated header row that starts with "No."
    try:
        header_idx = next(
            i for i, line in enumerate(lines)
            if line.split('\t')[0].strip() == 'No.'
        )
    except StopIteration:
        raise ValueError(f"Could not find data header in {csv_path}")

    data_str = '\n'.join(lines[header_idx:])
    df = pd.read_csv(StringIO(data_str), sep='\t')
    df.columns = df.columns.str.strip()

    # Parse wall-clock datetime: "07-03-26" (DD-MM-YY) + "11:23:13" (HH:MM:SS)
    dt_str = df['DD-MM-YY'].str.strip() + ' ' + df['HH:MM:SS'].str.strip()
    df['_datetime'] = pd.to_datetime(dt_str, format='%d-%m-%y %H:%M:%S')

    return df[['_datetime', 'CH1', 'CH2', 'CH3', 'CH4']].copy()


def prepare(date_key: str, files: list, force: bool = False) -> None:
    speed_files   = [f for f in files if classify_file(f) == 'speed_tracker']
    numbers_files = [f for f in files if classify_file(f) == 'numbers_tracker']
    temp_files    = [f for f in files if classify_file(f) == 'temp_logger']

    all_track_files = speed_files + numbers_files
    if not all_track_files:
        print(f"  skip  {date_key}  (no speed tracker file found)")
        return

    for speed_path in all_track_files:
        if speed_path.suffix == '.numbers':
            pq_path = speed_path.parent / f"{date_key} {speed_path.stem}.parquet"
        else:
            pq_path = speed_path.with_suffix(".parquet")
        if pq_path.exists() and not force:
            print(f"  skip  {pq_path.name}  (already exists — use --force to regenerate)")
            continue

        if speed_path.suffix == '.numbers':
            df = parse_numbers_tracker(speed_path)
        else:
            df = parse_speed_tracker(speed_path)

        before = len(df)
        df = df[df["Latitude"] <= LAT_MAX].reset_index(drop=True)
        removed = before - len(df)

        # Merge temperature channels if a temp logger file is available.
        temp_note = " (no temp logger found)"
        if temp_files:
            if len(temp_files) > 1:
                print(f"  warn  {date_key}: {len(temp_files)} temp logger files; using {temp_files[0].name}")
            temp_df = parse_temp_logger(temp_files[0]).sort_values('_datetime')
            df = df.sort_values('_datetime')
            df = pd.merge_asof(
                df, temp_df,
                on='_datetime',
                direction='nearest',
                tolerance=pd.Timedelta('30s'),
            )
            temp_note = ", +4 temp channels (CH1-CH4)"

        df = df.drop(columns=['_datetime'])

        # Apply per-date channel label overrides.
        if date_key in CHANNEL_LABELS:
            df = df.rename(columns=CHANNEL_LABELS[date_key])

        df.to_parquet(pq_path, index=False)
        print(f"  wrote {pq_path.name}  ({len(df):,} rows kept, {removed:,} removed north of {LAT_MAX}{temp_note})")


if __name__ == "__main__":
    force = "--force" in sys.argv
    groups = find_datestamp_files(DATALOGS)
    if not groups:
        print(f"No datestamp files found in {DATALOGS}")
        sys.exit(1)
    print(f"Processing {len(groups)} date(s) in {DATALOGS}")
    for date_key, files in sorted(groups.items()):
        prepare(date_key, files, force=force)
    print("Done.")
