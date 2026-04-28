"""
prepare_track_data.py
Converts speed-tracker CSV files to privacy-filtered parquet files.
Rows north of LAT_MAX are dropped to hide the starting/home location.

Usage:
    python prepare_track_data.py              # process all CSVs in DataLogs/SpeedTracker Phone GPS/
    python prepare_track_data.py --force      # re-generate even if parquet exists
"""
import sys
import pandas as pd
from pathlib import Path

DATALOGS = Path(__file__).parent.parent / "DataLogs" / "SpeedTracker Phone GPS"
DATAPARQUET = Path(__file__).parent.parent / "DataParquet"
LAT_MAX = 29.66716  # drop everything north of this latitude


def prepare(csv_path: Path, force: bool = False) -> Path:
    pq_path = DATAPARQUET / csv_path.with_suffix(".parquet").name
    if pq_path.exists() and not force:
        print(f"  skip  {pq_path.name}  (already exists — use --force to regenerate)")
        return pq_path

    df = pd.read_csv(csv_path, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    df = df.sort_values("Elapsed time (sec)")
    df = df.drop_duplicates(subset="Elapsed time (sec)", keep="first")

    before = len(df)
    df = df[df["Latitude"] <= LAT_MAX].reset_index(drop=True)
    removed = before - len(df)

    df.to_parquet(pq_path, index=False)
    print(f"  wrote {pq_path.name}  ({len(df):,} rows kept, {removed:,} removed north of {LAT_MAX})")
    return pq_path


if __name__ == "__main__":
    force = "--force" in sys.argv
    csvs = sorted(DATALOGS.glob("*.csv"))
    if not csvs:
        print(f"No CSV files found in {DATALOGS}")
        sys.exit(1)
    print(f"Processing {len(csvs)} file(s) in {DATALOGS}")
    for csv in csvs:
        prepare(csv, force=force)
    print("Done.")
