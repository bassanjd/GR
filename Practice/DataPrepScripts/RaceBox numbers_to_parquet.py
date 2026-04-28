"""Convert .numbers files in this directory to .parquet files.

Handles two formats:
  - LSA-GPS: header in row 0, data from row 1
  - RaceBox: metadata rows 0-11, header in row 12, data from row 13
"""

import numbers_parser
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

NUMBERS_DIR = Path(__file__).parent.parent / "DataLogs" / "numbers"
DATAPARQUET = Path(__file__).parent.parent / "DataParquet"
LAT_MAX = 29.66716  # exclude rows north of this latitude


def _is_fresh(src: Path, out: Path) -> bool:
    """True if out exists and is newer than src — skip conversion."""
    return out.exists() and out.stat().st_mtime >= src.stat().st_mtime


def table_to_df(table) -> pd.DataFrame:
    rows = list(table.iter_rows())
    values = [[c.value for c in row] for row in rows]

    # Detect RaceBox format: row 12 has 'Record' as first cell
    if len(values) > 12 and str(values[12][0]) == "Record":
        header = [str(v) if v is not None else f"col_{i}" for i, v in enumerate(values[12])]
        data = values[13:]
    else:
        header = [str(v) if v is not None else f"col_{i}" for i, v in enumerate(values[0])]
        data = values[1:]

    df = pd.DataFrame(data, columns=header)

    # Drop fully-None rows
    df = df.dropna(how="all")

    # Try to infer better dtypes (single pass: compare NaN counts before/after)
    for col in df.columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.isna().sum() == df[col].isna().sum():
            df[col] = converted

    # Filter out rows north of LAT_MAX
    lat_col = next((c for c in df.columns if c.lower() == "latitude"), None)
    if lat_col is not None:
        df = df[pd.to_numeric(df[lat_col], errors="coerce") <= LAT_MAX]

    return df


def _load_df(path: Path) -> pd.DataFrame:
    """Load a single .numbers file into a DataFrame (worker-safe, no print)."""
    doc = numbers_parser.Document(str(path))
    return table_to_df(doc.sheets[0].tables[0])


def convert_numbers_to_parquet(numbers_path: Path) -> Path:
    df = _load_df(numbers_path)
    out_path = numbers_path.with_suffix(".parquet")
    df.to_parquet(out_path, index=False)
    return out_path


def extract_date(stem: str) -> str | None:
    """Extract DD-MM-YYYY date from a filename stem, or None if not found."""
    import re
    m = re.search(r"\d{2}-\d{2}-\d{4}", stem)
    return m.group() if m else None


def output_name(stem: str, date: str) -> str:
    """Build output stem: strip 'Track Sessionon' and time, keep prefix + date."""
    import re
    cleaned = stem.replace("Track Sessionon", "").strip()
    cleaned = re.sub(r"\s+" + re.escape(date) + r".*", f" {date}", cleaned)
    return cleaned


def _process_group(date: str, day_files: list[Path], out_dir: Path) -> str:
    """Convert one date-group of .numbers files to a single parquet. Returns status line."""
    out_stem = output_name(day_files[0].stem, date)
    out_path = out_dir / f"{out_stem}.parquet"
    # Skip if all source files are older than existing parquet
    if all(_is_fresh(f, out_path) for f in day_files):
        return f"Skipping {out_path.name} (up to date)"
    names = ", ".join(f.name for f in day_files)
    frames = [_load_df(f) for f in day_files]
    pd.concat(frames, ignore_index=True).to_parquet(out_path, index=False)
    return f"Merged {names} -> {out_path.name}"


def _process_undated(f: Path) -> str:
    """Convert a single undated .numbers file. Returns status line."""
    out_path = DATAPARQUET / f.with_suffix(".parquet").name
    if _is_fresh(f, out_path):
        return f"Skipping {f.name} (up to date)"
    df = _load_df(f)
    df.to_parquet(out_path, index=False)
    return f"Converted {f.name} -> {out_path.name}"


def main():
    files = sorted(NUMBERS_DIR.glob("RaceBox*.numbers"))
    if not files:
        print("No .numbers files found.")
        return

    # Group files by date
    from collections import defaultdict
    groups: dict[str, list[Path]] = defaultdict(list)
    undated: list[Path] = []
    for f in files:
        date = extract_date(f.stem)
        if date:
            groups[date].append(f)
        else:
            undated.append(f)

    # Build list of tasks: (callable, args)
    tasks_dated = [(date, day_files) for date, day_files in sorted(groups.items())]
    total = len(tasks_dated) + len(undated)
    if total == 0:
        print("Nothing to process.")
        return

    # Run conversions in parallel (numbers_parser is CPU-bound)
    with ProcessPoolExecutor() as pool:
        futures = {}
        for date, day_files in tasks_dated:
            fut = pool.submit(_process_group, date, day_files, DATAPARQUET)
            futures[fut] = None
        for f in undated:
            fut = pool.submit(_process_undated, f)
            futures[fut] = None

        for fut in as_completed(futures):
            print(fut.result())


if __name__ == "__main__":
    main()
