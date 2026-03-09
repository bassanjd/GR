"""
prepare_track_data_with_basic_datalog.py
Converts speed-tracker CSV files to privacy-filtered parquet files,
integrating temperature readings from datestamp CSV datalogger files.

Datestamp files (YYYY-MM-DD *.csv) are grouped by date:
  - Speed tracker CSV (standard encoding, has "Elapsed time (sec)" column)
  - Temp logger CSV  (UTF-16 encoding, has CH1-CH4 temperature columns)

Temperature readings are aligned to speed tracker rows by nearest timestamp.
Rows north of LAT_MAX are dropped to hide the starting/home location.

Usage:
    python prepare_track_data_with_basic_datalog.py              # process all datestamp CSVs
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
    """Return {date_key: [Path, ...]} for all YYYY-MM-DD-prefixed CSVs."""
    groups: dict[str, list[Path]] = {}
    for csv in sorted(datalogs.glob("*.csv")):
        if DATE_PREFIX.match(csv.name):
            date_key = csv.name[:10]   # "YYYY-MM-DD"
            groups.setdefault(date_key, []).append(csv)
    return groups


def classify_file(csv_path: Path) -> str:
    """Return 'temp_logger' if UTF-16 BOM detected, else 'speed_tracker'."""
    bom = csv_path.read_bytes()[:2]
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
    speed_files = [f for f in files if classify_file(f) == 'speed_tracker']
    temp_files  = [f for f in files if classify_file(f) == 'temp_logger']

    if not speed_files:
        print(f"  skip  {date_key}  (no speed tracker CSV found)")
        return

    for speed_path in speed_files:
        pq_path = speed_path.with_suffix(".parquet")
        if pq_path.exists() and not force:
            print(f"  skip  {pq_path.name}  (already exists — use --force to regenerate)")
            continue

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
        print(f"No datestamp CSV files found in {DATALOGS}")
        sys.exit(1)
    print(f"Processing {len(groups)} date(s) in {DATALOGS}")
    for date_key, files in sorted(groups.items()):
        prepare(date_key, files, force=force)
    print("Done.")
