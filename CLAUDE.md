# CLAUDE.md — Great Race (GR) Codebase Guide

This file is the authoritative reference for AI assistants working in this repository.

---

## Project Overview

This is a **motorsport data analysis toolkit** for the Great Race — a cross-country rally competition run in 10 stages. Competitors navigate timed stages with precision; the goal is hitting exact target times at checkpoints. The codebase covers:

- Driver **practice and coaching** (reaction time, GPS telemetry, acceleration metrics)
- **Competition results** analysis and team comparison
- **Stage notes** extraction from PDFs using OCR and Claude vision API

All user-facing tools are **Streamlit web applications**. There is no REST API, backend server, or database — data flows through Parquet files and in-memory Pandas DataFrames.

---

## Repository Structure

```
GR/
├── Practice/                    # Driver practice & coaching apps
│   ├── stopwatch_repeatability_app.py   # Reaction time consistency tester
│   ├── practice_tracker_app.py          # GPS/speed telemetry analysis
│   ├── racebox_tracker_app.py           # RaceBox GPS logger analysis (25Hz)
│   ├── car_performance_app.py           # Generic performance channel browser
│   ├── prepare_track_data.py            # Data prep utility
│   ├── prepare_track_data_with_basic_datalog.py
│   └── *.parquet                        # Cached telemetry data files
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

### Devcontainer (primary workflow)

The repository ships with a devcontainer. It:
1. Uses `mcr.microsoft.com/devcontainers/python:1-3.11-bookworm`
2. Runs `pip3 install --user -r requirements.txt` on container create
3. Auto-starts `streamlit run Practice/stopwatch_repeatability_app.py` on attach
4. Forwards port **8501** and opens it as a preview

### Local setup

```bash
pip install -r requirements.txt
streamlit run Practice/stopwatch_repeatability_app.py
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
streamlit run Practice/stopwatch_repeatability_app.py
streamlit run Practice/practice_tracker_app.py
streamlit run Practice/racebox_tracker_app.py
streamlit run Practice/car_performance_app.py

# Results apps
streamlit run Results/team_analysis_app.py
streamlit run Results/advanced_pivot_explorer1_app.py
streamlit run Results/group_sum_montecarlo_explorer_app.py

# Stage Notes apps
streamlit run "Stage Notes/field_comparison_app.py"
streamlit run "Stage Notes/pdf_table_operator_app.py"
```

### Data extraction scripts (not Streamlit apps)

```bash
# Extract stage instructions from PDFs using Claude vision
python "Stage Notes/extract_claude.py"                      # all stages
python "Stage Notes/extract_claude.py" --stages 1 2         # specific stages
python "Stage Notes/extract_claude.py" --assemble-only      # rebuild from cache

# EasyOCR alternative extractor
python "Stage Notes/extract_stage_instructions.py"

# Scrape results from HTML
python Results/GreatRaceResultsScrape.py
python "Stage Notes/Scrape_Stage_Sheets.py"
python "Stage Notes/Scrape_Stage_Sheets_Batch.py"
```

---

## Data Architecture

### Primary data stores (Parquet files)

| File | Location | Schema |
|---|---|---|
| `long_format_times.parquet` | `Results/` | One row per team per stage leg; columns include `Team`, `Stage`, `Leg`, penalty info, `Discarded` flag |
| `stage_instructions.parquet` | `Stage Notes/` | One row per grid instruction (stage, leg, position, direction) |
| `leg_characteristics.parquet` | `Stage Notes/` | One row per (Stage, Leg) with terrain/type metadata |
| `great_race_all_stages.parquet` | `Stage Notes/` | Combined stage data |
| `*.parquet` | `Practice/` | GPS/speed telemetry (Time, Speed (mph), Lat/Lon, Altitude, Acceleration, Jerk) |

### Caching conventions

- Claude API responses are cached as JSON files in `Stage Notes/claude_cache/stage_N_page_M.json` — **never delete these unless re-extracting**
- Streamlit uses `@st.cache_data` on all expensive loads (parquet reads, data joins)
- Telemetry Parquet files in `Practice/` are pre-processed from raw GPS data logs

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

- **Files**: `snake_case_app.py` for Streamlit apps, `snake_case.py` for utilities
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

### Telemetry columns

Practice apps use these GPS columns:
- `Speed (mph)` — vehicle speed
- `Elapsed time (sec)` — time since segment start
- `Latitude`, `Longitude` — GPS position
- `Altitude (ft)` — elevation
- `Acceleration (ft/s²)` — computed derivative of speed
- `Jerk (ft/s³)` — computed derivative of acceleration

Physics constants used throughout:
```python
MPH_S_TO_FT_S2 = 5280 / 3600  # convert mph/s → ft/s²
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

There is **no automated test suite**. Verification is manual:

1. Run the target Streamlit app
2. Load a known data file
3. Visually verify charts and tables match expectations

When modifying data transformation logic, verify against known-good Parquet files before committing.

---

## Git Workflow

- **Branch**: `claude/add-claude-documentation-GsOMh` (current feature branch)
- **Remote**: `origin` at the local proxy
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
