"""
Generate two GPX files for the practice track:
  - track_eastbound.gpx  (car runs east: west trap → east trap)
  - track_westbound.gpx  (car runs west: east trap → west trap)

Each file contains:
  - Two waypoints: east trap and west trap (start/finish)
  - One track segment with GPS points from a representative run

Run:  python make_track_gpx.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
from xml.etree.ElementTree import Element, SubElement, tostring
import xml.dom.minidom

HERE = Path(__file__).parent

# ── Trap geometry (from racebox_tracker_app.py) ───────────────────────────────
TRAP_EAST_LAT =  29.337290
TRAP_EAST_LON = -95.63952
TRAP_WEST_LAT =  29.33716
TRAP_WEST_LON = -95.65612
LAT_THRESH    =  29.34

# ── Load & clean data ─────────────────────────────────────────────────────────
def load(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["_ts"] = pd.to_datetime(df["Time"])
    df = df.sort_values("_ts").reset_index(drop=True)
    df.loc[df["Latitude"] == 0, "Latitude"] = np.nan
    df.loc[df["Longitude"] == 0, "Longitude"] = np.nan
    df = df.dropna(subset=["Latitude", "Longitude"])
    df["elapsed"] = (df["_ts"] - df["_ts"].iloc[0]).dt.total_seconds()
    return df.reset_index(drop=True)


def find_runs(df: pd.DataFrame, min_speed_mph: float = 8.0,
              min_rows: int = 100) -> list[dict]:
    """Extract qualifying east- and west-bound runs from the track zone."""
    track = df[
        (df["Latitude"] < LAT_THRESH) &
        (df["Longitude"] >= TRAP_WEST_LON - 0.01) &
        (df["Longitude"] <= TRAP_EAST_LON + 0.01)
    ].copy().reset_index(drop=True)

    if track.empty:
        return []

    # Smooth longitude to derive direction without noise
    track["lon_smooth"] = track["Longitude"].rolling(25, center=True, min_periods=1).mean()
    track["lon_vel"] = track["lon_smooth"].diff() / track["elapsed"].diff()
    track["going_west"] = track["lon_vel"] < 0
    track["is_moving"] = track["Speed"] > min_speed_mph

    # New segment on direction flip or coming to a stop
    track["seg"] = (
        (track["going_west"] != track["going_west"].shift()) |
        (~track["is_moving"] & track["is_moving"].shift())
    ).cumsum()

    runs = []
    for _, grp in track.groupby("seg"):
        moving = grp[grp["is_moving"]]
        if len(moving) < min_rows:
            continue
        lon_span = grp["Longitude"].max() - grp["Longitude"].min()
        if grp["Speed"].max() < 10 or lon_span < 0.005:
            continue
        lon_delta = grp["Longitude"].iloc[-1] - grp["Longitude"].iloc[0]
        runs.append({
            "direction": "West" if lon_delta < 0 else "East",
            "lon_span": lon_span,
            "max_speed": grp["Speed"].max(),
            "data": grp.reset_index(drop=True),
        })
    return runs


def best_run(runs: list[dict], direction: str) -> pd.DataFrame | None:
    """Pick the run with the largest lon span (most complete) for a direction."""
    candidates = [r for r in runs if r["direction"] == direction]
    if not candidates:
        return None
    return max(candidates, key=lambda r: r["lon_span"])["data"]


# ── GPX builder ───────────────────────────────────────────────────────────────
GPX_NS = "http://www.topografix.com/GPX/1/1"
XSI_NS = "http://www.w3.org/2001/XMLSchema-instance"
XSI_LOC = (
    "http://www.topografix.com/GPX/1/1 "
    "http://www.topografix.com/GPX/1/1/gpx.xsd"
)


def make_gpx(name: str, waypoints: list[dict], track_pts: pd.DataFrame,
             downsample: int = 5) -> str:
    """
    Build a GPX XML string.

    waypoints: list of {"lat":, "lon":, "name":, "sym":}
    track_pts: DataFrame with Latitude, Longitude, Altitude (opt), Speed (opt)
    downsample: keep every Nth GPS point (25 Hz → ~5 Hz at downsample=5)
    """
    root = Element("gpx", attrib={
        "version": "1.1",
        "creator": "make_track_gpx.py",
        "xmlns": GPX_NS,
        "xmlns:xsi": XSI_NS,
        "xsi:schemaLocation": XSI_LOC,
    })

    meta = SubElement(root, "metadata")
    SubElement(meta, "name").text = name

    # Waypoints (start then finish)
    for wp in waypoints:
        wpt = SubElement(root, "wpt", lat=f"{wp['lat']:.6f}", lon=f"{wp['lon']:.6f}")
        SubElement(wpt, "name").text = wp["name"]
        if "sym" in wp:
            SubElement(wpt, "sym").text = wp["sym"]

    # Track
    trk = SubElement(root, "trk")
    SubElement(trk, "name").text = name
    trkseg = SubElement(trk, "trkseg")

    pts = track_pts.iloc[::downsample]
    for _, row in pts.iterrows():
        trkpt = SubElement(trkseg, "trkpt",
                           lat=f"{row['Latitude']:.7f}",
                           lon=f"{row['Longitude']:.7f}")
        if "Altitude" in track_pts.columns and not np.isnan(row.get("Altitude", float("nan"))):
            SubElement(trkpt, "ele").text = f"{row['Altitude']:.1f}"
        if "Speed" in track_pts.columns:
            SubElement(trkpt, "speed").text = f"{row['Speed'] * 0.44704:.3f}"  # mph → m/s

    raw = tostring(root, encoding="unicode", xml_declaration=False)
    dom = xml.dom.minidom.parseString(raw)
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + dom.toprettyxml(indent="  ")[23:]


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # Prefer the most recent file with the most runs
    parquet_files = sorted(HERE.glob("RaceBox*.parquet"), reverse=True)
    if not parquet_files:
        raise FileNotFoundError("No RaceBox*.parquet files found in " + str(HERE))

    # Try files until we get both directions
    all_runs: list[dict] = []
    for pf in parquet_files:
        print(f"Loading {pf.name} …")
        df = load(pf)
        all_runs.extend(find_runs(df))

    west_run = best_run(all_runs, "West")
    east_run = best_run(all_runs, "East")

    if west_run is None:
        raise ValueError("No westbound runs found")
    if east_run is None:
        raise ValueError("No eastbound runs found")

    print(f"Westbound run: {len(west_run)} GPS points, max {west_run['Speed'].max():.1f} mph")
    print(f"Eastbound run: {len(east_run)} GPS points, max {east_run['Speed'].max():.1f} mph")

    east_wpt = {"lat": TRAP_EAST_LAT, "lon": TRAP_EAST_LON,
                "name": "East Trap", "sym": "Flag, Blue"}
    west_wpt = {"lat": TRAP_WEST_LAT, "lon": TRAP_WEST_LON,
                "name": "West Trap", "sym": "Flag, Red"}

    # Westbound: start = east trap, finish = west trap
    west_gpx = make_gpx(
        "Practice Track – Westbound",
        waypoints=[
            {**east_wpt, "name": "Start (East Trap)"},
            {**west_wpt, "name": "Finish (West Trap)"},
        ],
        track_pts=west_run,
    )
    out_west = HERE / "track_westbound.gpx"
    out_west.write_text(west_gpx, encoding="utf-8")
    print(f"Written: {out_west}")

    # Eastbound: start = west trap, finish = east trap
    east_gpx = make_gpx(
        "Practice Track – Eastbound",
        waypoints=[
            {**west_wpt, "name": "Start (West Trap)"},
            {**east_wpt, "name": "Finish (East Trap)"},
        ],
        track_pts=east_run,
    )
    out_east = HERE / "track_eastbound.gpx"
    out_east.write_text(east_gpx, encoding="utf-8")
    print(f"Written: {out_east}")


if __name__ == "__main__":
    main()
