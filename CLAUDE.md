# CLAUDE.md — Great Race (GR) Codebase Guide

This file is the authoritative reference for AI assistants working in this repository.

---

## Project Overview

This is a **motorsport data analysis toolkit** for the Great Race — a cross-country rally competition run in 10 stages. Competitors navigate timed stages with precision; the goal is hitting exact target times at checkpoints. The codebase covers:

- Driver **practice and coaching** (reaction time, GPS telemetry, acceleration metrics)
- **Competition results** analysis and team comparison
- **Stage notes** extraction from PDFs using OCR and Claude vision API
- **Navigator reference charts** — speedometer calibration and speed-transition time matrices for in-car use

All user-facing tools are **Streamlit web applications**. There is no REST API, backend server, or database — data flows through Parquet files and in-memory Pandas DataFrames.

---

## Repository Structure

```
GR/
├── Practice/                    # Driver practice & coaching apps
│   ├── driver_performance_app.py        # Primary practice analysis app (all sources)
│   ├── car_performance_app.py           # Multi-source channel browser / data explorer
│   ├── stopwatch_repeatability_app.py   # Reaction time consistency tester
│   ├── telemetry.py                     # GPS/physics analysis module (see below)
│   ├── loaders.py                       # Schema-translating parquet loaders (see below)
│   ├── test_loaders.py                  # pytest tests for loaders.py
│   ├── test_telemetry.py                # pytest tests for telemetry.py
│   ├── make_track_gpx.py                # GPX file generator for east/west track routes
│   ├── prepare_track_data_with_basic_datalog.py
│   └── DataParquet/                     # Telemetry parquet files (RaceBox, speed-tracker, Generic)
│
├── Results/                     # Competition results analysis
│   ├── team_analysis_app.py             # Team performance dashboard
│   ├── advanced_pivot_explorer1_app.py  # Pivot tables + Monte Carlo BI
│   ├── group_sum_montecarlo_explorer_app.py  # PCA/clustering explorer
│   ├── GreatRaceResultsScrape.py        # HTML scraper for results pages
│   ├── Results Variation.ipynb          # Jupyter analysis notebook
│   └── long_format_times.parquet        # Primary results dataset
│
├── Stage Notes/                 # Rally stage instruction extraction
│   ├── extract_claude.py                # Claude vision API extractor
│   ├── extract_stage_instructions.py    # EasyOCR extractor
│   ├── field_comparison_app.py          # Stage field comparison app
│   ├── pdf_table_operator_app.py        # Manual PDF table extraction
│   ├── Scrape_Stage_Sheets.py           # HTML scraper
│   ├── Scrape_Stage_Sheets_Batch.py     # Batch scraper
│   ├── stage_instructions.parquet       # Extracted stage instructions
│   ├── leg_characteristics.parquet      # Per-leg metadata
│   └── great_race_all_stages.parquet    # All stages combined
│
├── NavigatorCharts/             # Speedometer calibration & navigator reference charts
│   ├── navigator_chart_helpers.py       # Shared matrix math, data loading, Excel utilities
│   ├── navigator_chart_app.py           # Interactive Streamlit app (calibration runs + export)
│   ├── make_navigator_charts.py         # Batch script: build reference charts from calibration data
│   ├── normalize_calibration_data.py    # Parse raw timing Excel → normalized parquet/xlsx
│   ├── test_navigator_chart_helpers.py  # pytest tests for navigator_chart_helpers.py
│   └── navigator_chart_calibration_runs.parquet  # Normalized calibration run data
│
├── .devcontainer/
│   └── devcontainer.json                # Docker dev environment config
├── requirements.txt                     # Core Python dependencies
├── Regulations.pdf                      # Competition rulebook
└── README.md                            # Currently empty
```

---

## Technology Stack

| Category | Technology |
|---|---|
| Language | Python 3.11 |
| UI Framework | Streamlit |
| Data Manipulation | Pandas, NumPy |
| Visualization | Plotly Express / Graph Objects |
| ML / Stats | scikit-learn, scipy |
| PDF Processing | pdf2image, PIL/Pillow |
| OCR | EasyOCR, OpenCV (cv2) |
| AI Extraction | Anthropic Claude API (vision) |
| Data Storage | Parquet files |
| Dev Environment | Docker devcontainer (Python 3.11 Bookworm) |

**Runtime dependencies** (`requirements.txt`):
```
streamlit, pandas, numpy, plotly, pathlib
```

Heavy dependencies (OCR, ML, PDF processing) are installed on demand and are not in `requirements.txt`.

---

## Development Setup

### Devcontainer (secondary workflow)

The repository ships with a devcontainer. It:
1. Uses `mcr.microsoft.com/devcontainers/python:1-3.11-bookworm`
2. Runs `pip3 install --user -r requirements.txt` on container create
3. Forwards port **8501**

### Local setup (primary workflow)

```bash
pip install -r requirements.txt
streamlit run Practice/driver_performance_app.py
```

### Environment variables

Create a `.env` file in the repository root for API keys (`.gitignore` excludes it):

```
ANTHROPIC_API_KEY=sk-ant-...
```

`Stage Notes/extract_claude.py` loads this automatically via `python-dotenv`.

---

## Running the Applications

All apps run on port 8501 (only one at a time):

```bash
# Practice apps
streamlit run Practice/driver_performance_app.py    # primary: multi-tab practice analysis
streamlit run Practice/car_performance_app.py       # channel browser / data explorer
streamlit run Practice/stopwatch_repeatability_app.py

# Results apps
streamlit run Results/team_analysis_app.py
streamlit run Results/advanced_pivot_explorer1_app.py
streamlit run Results/group_sum_montecarlo_explorer_app.py

# Stage Notes apps
streamlit run "Stage Notes/field_comparison_app.py"
streamlit run "Stage Notes/pdf_table_operator_app.py"

# NavigatorCharts app
streamlit run NavigatorCharts/navigator_chart_app.py
```

### Data extraction scripts (not Streamlit apps)

```bash
# Extract stage instructions from PDFs using Claude vision
python "Stage Notes/extract_claude.py"                      # all stages
python "Stage Notes/extract_claude.py" --stages 1 2         # specific stages
python "Stage Notes/extract_claude.py" --assemble-only      # rebuild from cache

# EasyOCR alternative extractor
python "Stage Notes/extract_stage_instructions.py"

# Generate GPX files for the practice track
python Practice/make_track_gpx.py                           # produces track_eastbound.gpx, track_westbound.gpx

# Scrape results from HTML
python Results/GreatRaceResultsScrape.py
python "Stage Notes/Scrape_Stage_Sheets.py"
python "Stage Notes/Scrape_Stage_Sheets_Batch.py"

# Navigator charts: normalize raw calibration data then build reference charts
python NavigatorCharts/normalize_calibration_data.py        # parse raw Excel → navigator_chart_normalized.xlsx
python NavigatorCharts/make_navigator_charts.py             # build reference matrices from normalized data
```

---

## Practice Analysis Architecture

### `loaders.py` — schema translation layer

Converts source-specific parquet formats to the **standard internal schema**. Apps call `loaders.load_any(path)` wrapped with `@st.cache_data`; they never read parquets directly.

**Standard internal schema** — every loader produces these columns:
- `Elapsed time (sec)` — time axis
- `Distance (mi)` — cumulative GPS odometer (haversine)
- `Speed (mph)` — vehicle speed
- `Latitude`, `Longitude` — GPS position (decimal degrees)
- `Time` — timestamp string (HH:MM:SS)
- `Date` — date string (YYYY-MM-DD)

Extra source-specific columns are preserved and available to apps.

**Three source formats handled:**
- `load_racebox(path)` — raw RaceBox logger: parses ISO-8601 `Time`, renames `Speed` → `Speed (mph)`, computes haversine `Distance (mi)`, preserves GForce channels
- `load_speed_tracker(path)` — pre-processed speed-tracker output: already in standard schema, just sorts and deduplicates; supports `.parquet` and `.csv`
- `load_generic(path)` — proprietary datalogger: renames `GPS Speed (mph)` → `Speed (mph)`, `GPS Latitude/Longitude (deg)` → `Latitude/Longitude`, converts `Datetime` → `Time`/`Date` strings, computes haversine distance; preserves IMU/RPM/temperature channels

**Dispatch**: `load_any(path)` routes by filename — `RaceBox*` → `load_racebox`, `*speed_tracker*` → `load_speed_tracker`, everything else → `load_generic`.

### `telemetry.py` — analysis and physics module

All data-source-agnostic telemetry logic lives here. Schema translation is in `loaders.py`; `telemetry.py` imports `haversine_cumulative_mi` is the only function used by loaders. Apps import analysis functions directly from `telemetry`.

**`TrapConfig` dataclass** — all geometric state for the measured section:
- `east_lon/lat`, `west_lon/lat` — gate endpoint coordinates
- `dist_mi` — nominal trap length (default 1 mile / 5280 ft)
- `flying_window_mi` — half-width of speed-sampling window around gate
- Derived fields (`ux`, `uy`, `chord_mi`, `mi_per_deg_lon`) computed in `__post_init__`

**Signal processing**: `smooth_series`, `compute_derivatives` (adds `speed_smooth`, `accel_ft_s2`, `accel_g`, `jerk_ft_s3`)

**GPS geometry**: `haversine_cumulative_mi`, `find_crossing_dist` (oblique trap gate crossing)

**Run segmentation**: `segment_exercise_runs` — isolates runs in the exercise zone (lat ≈ 29.33715, east-west track), splits on time gaps > 30 s, detects target speed from mode of in-trap speeds

**Run classification**: `run_start_type`, `run_finish_type` — Flying vs Standing based on speed in window around trap gates; `get_run_timing_refs` — returns `(entry_d, exit_d)` odometer distances appropriate for each start/finish combination

### `driver_performance_app.py` — primary practice app

Multi-tab Streamlit app supporting all parquet sources (RaceBox, speed-tracker, Generic). Loads all files from `DataParquet/` via `loaders.load_any()`.

**Tabs:**
1. **Runs Overview** — summary table with timing/speed/accel stats; speed overlay chart; trap-aligned speed comparison; acceleration & jerk distributions
2. **Run Detail** — single-run deep dive: 5-panel subplot (cumulative time error, speed, acceleration, jerk, speed residual) vs ideal profile; actual vs ideal comparison table
3. **Accel/Decel** — standing-start acceleration curves and standing-finish deceleration curves, plotted vs distance and vs speed (for shift-consistency analysis)
4. **GPS vs Accel** — compares GPS-derived acceleration against RaceBox GForce channels (X/Y/Z or horizontal magnitude); auto-selects best-correlated axis; unity plot and residual histogram
5. **Settings** — smoothing window, standing-detection window, trap gate lat/lon coordinates (number inputs, 0.00001° step ≈ 3 ft), at-rest point map
6. **Track Map** — full-session GPS track colored by zone (Transit / Exercise Area) with run overlays and trap gate lines; street or ESRI satellite basemap

**Sidebar filters** (above file checkboxes): target speed, start type, finish type — all reset when the file selection changes.

**Trap coordinates** are hardcoded defaults at the top of the app and overridden via Settings tab `st.session_state` keys (`cfg_east_ctr_lat`, etc.).

### `make_track_gpx.py` — GPX export utility

Reads the most recent RaceBox parquet, finds a representative eastbound and westbound run, and writes `track_eastbound.gpx` / `track_westbound.gpx` with the trap waypoints and GPS track. Used to load the course into navigation devices or mapping apps.

---

## Navigator Charts Architecture

### `navigator_chart_helpers.py` — shared matrix math and Excel utilities

Constants, pure matrix functions, loss computation, and openpyxl writing helpers shared by both the batch script and the Streamlit app.

**Constants**: `SPEEDS = [0, 15, 20, 25, 30, 35, 40, 45, 50]`, `STOP_SIGN_SECONDS = 15.0`, `COURSE_MI = 1.0`, `DATE_2026`

**Matrix functions** (all return a `len(SPEEDS) × len(SPEEDS)` list-of-lists):
- `matrix_transition(accel, decel)` — extra seconds for any speed change; diagonal is `BLANK`
- `matrix_stop_go(accel, decel)` — standstill seconds remaining inside a mandatory stop
- `matrix_turn_loss(accel, decel, ref_mph)` — time lost vs arriving/leaving at `ref_mph`

**Data pipeline**:
- `load_calibration_runs()` — reads the *Calibration Runs* sheet from `navigator_chart_normalized.xlsx`
- `compute_losses(df)` — pivots 2026 rows into per-speed accel/decel loss table; returns `None` if data incomplete
- `losses_to_dicts(losses)` — converts loss DataFrame to `(accel, decel)` dicts keyed by MPH; missing speeds fill with `NaN`

**Excel writing**: `write_matrix(...)`, `write_reference_charts_to_sheet(...)`, `build_reference_workbook(...)` — write four labeled, color-scaled matrices to openpyxl worksheets.

### `normalize_calibration_data.py` — raw data normalization

Reads `2026 Great Race Charts April 29th.xlsx` (three worksheets: *Straight Speed*, *Speed Stop*, *Start Speed*), parses timing strings in both `MM:SS.cs` and `MM:SS:cs` formats, and writes:
- `navigator_chart_normalized.xlsx` — *Calibration Runs* sheet (all raw runs) + *2026 Calibration* sheet (per-speed averages, actual MPH, error %, accel/decel losses)
- `navigator_chart_calibration_runs.parquet` — same calibration runs as Parquet

### `make_navigator_charts.py` — batch reference chart generator

Calls `load_calibration_runs()` → `compute_losses()` → `losses_to_dicts()` → `build_reference_workbook()` and saves the result to `navigator_chart_normalized.xlsx` (*Reference Charts* sheet).

### `navigator_chart_app.py` — interactive Streamlit app

Three-column layout (no tabs). Left column: raw calibration runs table with per-row *Exclude* checkboxes. Middle/right columns: strip charts and loss line charts computed from all data vs. filtered data. Two download buttons export in-memory Excel workbooks with the four reference matrices.

---

## Data Architecture

### Primary data stores (Parquet files)

| File | Location | Schema |
|---|---|---|
| `long_format_times.parquet` | `Results/` | One row per team per stage leg; columns: `RANK`, `CAR`, `YEAR`, `FACTOR`, `ScYR`, `DIV`, `CREW`, `Stage`, `Leg`, `Time`, `Early`, `Discarded`, `Actual_Time` |
| `stage_instructions.parquet` | `Stage Notes/` | One row per grid instruction (stage, leg, position, direction) |
| `leg_characteristics.parquet` | `Stage Notes/` | One row per (Stage, Leg) with terrain/type metadata |
| `great_race_all_stages.parquet` | `Stage Notes/` | Combined stage data |
| `RaceBox *.parquet` | `Practice/DataParquet/` | Raw RaceBox GPS logger output (Time, Latitude, Longitude, Speed, GForceX/Y/Z, …) |
| `*speed_tracker*.parquet` | `Practice/DataParquet/` | Pre-processed speed-tracker output (already in standard schema); may include attached datalogger temperature channels |
| `*Generic*.parquet` | `Practice/DataParquet/` | Proprietary datalogger output (GPS Speed/Lat/Lon/Altitude, Datetime, IMU channels InlineAcc/LateralAcc/VerticalAcc, RPM, temperatures) |
| `navigator_chart_calibration_runs.parquet` | `NavigatorCharts/` | Normalized calibration run data (date, test_type, target_mph, run_number, direction, time_s) |

### Caching conventions

- Claude API responses are cached as JSON files in `Stage Notes/claude_cache/stage_N_page_M.json` — **never delete these unless re-extracting**
- Streamlit uses `@st.cache_data` on all expensive loads (parquet reads, data joins)
- `driver_performance_app.py` caches per-run derivative DataFrames on the run dict (`r["_grp_with_derivs"]`) to avoid recomputation across tabs

### Excluded from git

`.gitignore` excludes:
- `DataLogs/`, `raw_data/` — raw GPS/telemetry source data
- `claude_cache/` — cached Claude API responses
- Python caches (`__pycache__`, `*.pyc`, `*.pyo`)
- Virtual environments (`.venv`, `env/`)
- Streamlit secrets (`.streamlit/secrets.toml`)

---

## Code Conventions

### Naming

- **Files**: `snake_case_app.py` for Streamlit apps, `snake_case.py` for utilities/modules
- **Functions**: `snake_case`
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `MPH_S_TO_FT_S2`, `DATALOGS`, `PARQUET`)
- **DataFrame columns**: match source data exactly (e.g., `"Speed (mph)"`, `"Elapsed time (sec)"`)

### Streamlit app structure

Every app follows this layout:

```python
"""Module docstring explaining the app."""

# Standard imports
import streamlit as st
import pandas as pd
...

# st.set_page_config() — always first Streamlit call
st.set_page_config(page_title="...", layout="wide", page_icon="...")

# Constants / paths
PARQUET = Path(__file__).parent / "data.parquet"

# ── Helpers ───────────────────────────────────────────────────────────────────
# Pure functions, @st.cache_data loaders, domain logic

# ── State init ────────────────────────────────────────────────────────────────
if "key" not in st.session_state:
    st.session_state.key = default_value

# ── Main UI ───────────────────────────────────────────────────────────────────
# Sidebar, main content, charts
```

ASCII section dividers (`# ── Section ───`) mark logical blocks.

### Session state pattern

Always guard initialization:

```python
if "phase" not in st.session_state:
    st.session_state.phase = "setup"
```

Use `st.session_state.pop(key, None)` for cleanup/reset rather than setting to `None`.

### Data loading pattern

```python
@st.cache_data
def load_data():
    df = pd.read_parquet(PARQUET)
    return df[~df["Discarded"]].copy()

df = load_data()
```

Always call `load_data()` at module level so the cache is populated on startup.

### Visualization pattern

Use Plotly for all charts (never matplotlib). Prefer `plotly.express` for simple charts, `plotly.graph_objects` for composite or custom charts. Pass `use_container_width=True` to `st.plotly_chart()`.

### Path handling

Use `pathlib.Path` exclusively. Always resolve paths relative to `__file__`:

```python
BASE = Path(__file__).parent
DATA = BASE / "data.parquet"
```

Never hardcode absolute paths (there is one legacy Windows path in `extract_claude.py` for Poppler — leave it but note it needs updating for Linux).

---

## Domain Knowledge

### Great Race basics

- 10-stage cross-country rally (Stage 0 is a Trophy Run)
- Competitors drive a set route with hidden checkpoints
- Goal: arrive at each checkpoint at the exact target time (zero seconds early/late = perfect)
- Penalty is the absolute deviation from the target time in seconds
- `Discarded` rows in results data mark invalid/DSQ legs
- `FACTOR` = age-based score multiplier; `DIV` = division (S/E/G/R/X)

### Telemetry / standard schema columns

Practice apps operate on the standard schema produced by `loaders.py`:
- `Speed (mph)` — vehicle speed
- `Elapsed time (sec)` — time since first record in the file
- `Distance (mi)` — cumulative haversine GPS odometer
- `Latitude`, `Longitude` — GPS position
- `Time` — original timestamp string
- `Date` — date string (YYYY-MM-DD)

Computed derivative columns (added by `compute_derivatives`):
- `speed_smooth` — rolling-mean smoothed speed
- `accel_ft_s2` — longitudinal acceleration (ft/s²)
- `accel_g` — acceleration in g
- `jerk_ft_s3` — jerk (ft/s³)

Physics constants in `telemetry.py`:
```python
MPH_S_TO_FT_S2 = 5280.0 / 3600.0   # 1 mph/s → ft/s²
G_FT_S2        = 32.174              # 1 g in ft/s²
```

### Stage instruction extraction

Stage notes are PDFs containing grids of navigation instructions. `extract_claude.py` converts each PDF page to an image at 200 DPI and sends it to the Claude API with a structured prompt to extract the grid as JSON. The output feeds `stage_instructions.parquet`.

---

## Claude API Usage

`Stage Notes/extract_claude.py` uses `claude-sonnet-4-6` for vision extraction. Key patterns:

```python
client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY from environment

response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=4096,
    messages=[{
        "role": "user",
        "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
            {"type": "text", "text": EXTRACTION_PROMPT},
        ]
    }]
)
```

- Cache responses before any post-processing to avoid re-billing on parse errors
- Use `--assemble-only` to rebuild Parquet from existing cache without new API calls

---

## Testing

### Automated tests

Run all tests from the repo root:

```bash
pytest Practice/test_loaders.py Practice/test_telemetry.py NavigatorCharts/test_navigator_chart_helpers.py
```

#### `Practice/test_loaders.py`

Covers all three loaders and the `load_any()` dispatcher:

- **Individual loader tests** — schema correctness, sort/dedup, source-specific column presence (GForce channels, datalogger channels, GPS rename)
- **`load_any` dispatch tests** — verifies routing by filename prefix
- **Parametrized contract tests** (run over every `.parquet` in `DataParquet/`) — required schema columns, monotonic elapsed time, no duplicate timestamps, non-negative speed, non-decreasing distance, lat/lon in valid range, non-empty result

When adding a new loader or modifying schema translation, run the full suite and ensure all parametrized contract tests pass for every file in `DataParquet/`.

#### `Practice/test_telemetry.py`

Covers all pure analysis functions in `telemetry.py`: `TrapConfig` geometry, `haversine_cumulative_mi`, `smooth_series`, `compute_derivatives`, `find_crossing_dist`, `run_start_type`/`run_finish_type`, `get_run_timing_refs`, and `segment_exercise_runs`.

#### `NavigatorCharts/test_navigator_chart_helpers.py`

Covers all pure functions in `navigator_chart_helpers.py` and the time-parsing helpers in `normalize_calibration_data.py`: matrix math (`matrix_transition`, `matrix_stop_go`, `matrix_turn_loss`), loss pipeline (`compute_losses`, `losses_to_dicts`), Excel style helpers, `build_reference_workbook`, `parse_time_str`, `time_obj_to_s`, `fmt_mm_ss_cs`.

### Manual verification

For Streamlit app changes, visual verification remains necessary:

1. Run the target app
2. Load a known data file
3. Verify charts and tables match expectations

---

## Git Workflow

- **Branch**: `main`
- **No CI/CD** — no GitHub Actions workflows exist
- Commit message style: `feat:`, `fix:`, `chore:`, `docs:`

```bash
git push -u origin <branch-name>
```

---

## Common Tasks

### Add a new Streamlit app

1. Create `<directory>/<name>_app.py`
2. Follow the module structure above (docstring → imports → `set_page_config` → constants → helpers → state init → UI)
3. Use `@st.cache_data` for any Parquet/file load
4. Reference data files relative to `Path(__file__).parent`

### Add a new data source to the practice app

1. Write `load_<source>(path: str) -> pd.DataFrame` in `loaders.py` that produces the standard internal schema (see `loaders.py` docstring)
2. Add a detection branch in `loaders.load_any()` based on the filename pattern
3. The analysis layer (`segment_exercise_runs`, `compute_derivatives`, etc.) and both Streamlit apps require no changes — they consume the standard schema via `load_any()`
4. Add contract tests in `test_loaders.py` for the new source and confirm the parametrized suite passes

### Add a new Parquet dataset

1. Generate it with a Python script (not a Streamlit app)
2. Save to the relevant directory using `df.to_parquet(path, index=False)`
3. Load in apps with `@st.cache_data` + `pd.read_parquet()`
4. Add to `.gitignore` only if it's derived/regeneratable (otherwise commit it)

### Modify the Claude extraction prompt

Edit the `EXTRACTION_PROMPT` constant near the top of `Stage Notes/extract_claude.py`. Delete the relevant `claude_cache/` JSON files for the affected pages, then re-run the extractor.

### Update results data

Run `Results/GreatRaceResultsScrape.py` pointing at the official results HTML, then verify `long_format_times.parquet` columns match what `team_analysis_app.py` expects.

---

## What NOT to Do

- **Do not add a database** — Parquet + Pandas is intentional and sufficient
- **Do not add authentication** — these are local single-user tools
- **Do not use matplotlib** — all charts use Plotly
- **Do not hardcode absolute paths** — always use `Path(__file__).parent`
- **Do not call the Claude API in a Streamlit app** — API calls belong in batch scripts; apps read from Parquet cache
- **Do not delete `claude_cache/` JSON files** unless you intend to re-run the expensive extraction
- **Do not use `st.experimental_rerun()`** — use `st.rerun()` (Streamlit ≥1.27)
- **Do not duplicate telemetry logic in apps** — GPS/physics/signal processing belongs in `telemetry.py`; schema translation belongs in `loaders.py`
- **Do not add loaders to `telemetry.py`** — schema translation lives in `loaders.py`; `telemetry.py` is analysis-only
