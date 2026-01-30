# Waste & Flood Impact Evaluation

PhD UCL project: evaluating the relationship between flood exposure, waste, and population displacement using Facebook population data, WorldPop, flood rasters, and waste/SVI point data.

---

## Overview

This repository builds a **quadkey-level analysis dataset** that combines:

- **Displacement metrics** — population outflow during a flood event (Facebook population change, scaled to WorldPop)
- **Flood metrics** — zonal statistics from a flood depth/exposure raster
- **Waste metrics** — spatially aggregated waste points and SVI (Street View Imagery) coverage
- **Population metrics** — WorldPop estimates and scaling factors

The final dataset is used in **spatial regression models** (OLS, Spatial Lag, Spatial Error) to study how flood and waste relate to displacement, with optional population controls. Outputs include model results, summary tables, and figures for the paper.

**Final analysis layer:** `Data/4/4_flood_waste_metrics_quadkey.gpkg` (one row per quadkey).

---

## Workflow

```
External data (FB, WorldPop, flood raster, waste/SVI)
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 1: 1y_compute_accumulated_outflow.py                       │
│  Continuous outflow: baseline vs crisis → shortfall per cell     │
│  Output: Data/1/                                                  │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 2: 2y_detect_csat_anomalies.py                              │
│  CSAT (modified Z-score) anomaly detection → max outflow         │
│  Output: Data/2/                                                  │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 3: 3scale_to_worldpop.py                                    │
│  Scale FB displacement to WorldPop; zonal WorldPop per quadkey    │
│  Output: Data/3/                                                  │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 4: 4flood_waste.py                                          │
│  Zonal flood stats + waste/SVI aggregation → final quadkey GPKG   │
│  Output: Data/4/4_flood_waste_metrics_quadkey.gpkg               │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│  Models: model_flood, model_wasteflood, model_floodpop,           │
│          model_wastefloodpop                                      │
│  OLS + Moran's I + SLM/SEM + impacts → Output/flood, etc.         │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│  summarize_outputs.py + Figure&Table/*.py                         │
│  Table 1, supplementary tables, figures → Output/summary, Figure/  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Repository structure

| Path | Description |
|------|-------------|
| `1y_compute_accumulated_outflow.py` | Step 1: continuous outflow (shortfall) per quadkey |
| `2y_detect_csat_anomalies.py` | Step 2: CSAT anomaly detection, max outflow |
| `3scale_to_worldpop.py` | Step 3: WorldPop scaling and zonal stats |
| `4flood_waste.py` | Step 4: flood + waste metrics → final GPKG |
| `model_flood.py` | Flood-only regression (displacement ~ flood) |
| `model_wasteflood.py` | Flood + waste (displacement ~ flood + waste) |
| `model_floodpop.py` | Flood-only with population control |
| `model_wastefloodpop.py` | Flood + waste + population |
| `multicollinearity_check.py` | VIF and correlation diagnostics |
| `multicollinearity_utils.py` | Shared multicollinearity helpers |
| `summarize_outputs.py` | Merge model outputs into paper tables |
| `Data/1/` … `Data/4/` | Intermediate and final pipeline outputs |
| `Figure&Table/` | Scripts for figures and tables (create_table_*.py, figure_*.py) |
| `Figure/` | Generated figures |
| `Output/` | Model results (flood, wasteflood, floodpop, wastefloodpop, summary) |
| `SCRIPT_OUTPUTS_SUMMARY.md` | Column-level documentation of the final dataset |

---

## Requirements

### External data (paths are set inside each script)

You need to have these files and point the scripts to them (edit the path constants at the top of each script if your layout differs):

| Data | Purpose |
|------|--------|
| Facebook population (wide) | `FB_32737_wide.gpkg` (layer `population_change`) — baseline vs crisis week |
| WorldPop raster | Clipped to study area (e.g. `WorldPop_clipped_aoi.tif`) |
| Flood raster | Flood depth/exposure (e.g. `model8.tif`) |
| Waste points | `aoi_waste.gpkg` (EPSG:4326) |
| SVI points | `aoi_all_svi.gpkg` (EPSG:4326) |

Default paths used in the scripts point under something like:
`/Users/wenlanzhang/Downloads/PhD_UCL/Data/Waste_flood/` and `.../Data/Waste/`.

### Python environment

Suggested stack:

- **Python:** 3.10+
- **Core:** `geopandas`, `pandas`, `numpy`, `rasterio`, `rasterstats`, `matplotlib`, `seaborn`
- **Stats:** `statsmodels`, `scipy`
- **Spatial:** `libpysal`, `esda`, `spreg`

Optional: `contextily` (basemaps in figures).

Example (Conda):

```bash
conda create -n waste_flood python=3.10 -y
conda activate waste_flood
conda install -c conda-forge geopandas pandas numpy rasterio rasterstats matplotlib seaborn statsmodels scipy libpysal esda
pip install spreg
```

---

## How to run

From the project root (`Waste_Flood_out`).

### 1. Data pipeline (run in order)

```bash
python 1y_compute_accumulated_outflow.py
python 2y_detect_csat_anomalies.py
python 3scale_to_worldpop.py
python 4flood_waste.py
```

Step 3 reads from `Data/1/` and `Data/2/`; Step 4 reads from `Data/3/`. Step 4 produces `Data/4/4_flood_waste_metrics_quadkey.gpkg`, which all models use.

### 2. Models (after Step 4)

```bash
python model_flood.py
python model_wasteflood.py
python model_floodpop.py
python model_wastefloodpop.py
```

Outputs go to `Output/flood/`, `Output/wasteflood/`, `Output/floodpop/`, `Output/wastefloodpop/`.

### 3. Summary tables and figures

```bash
python summarize_outputs.py
python Figure&Table/create_table_1.py
python Figure&Table/create_table_s1.py   # optional: supplementary tables
python Figure&Table/create_table_s2.py
# ... create_table_s3.py, create_table_s4.py as needed
python Figure&Table/figure_1_study_design.py
# ... figure_2_spatial_distribution.py, figure_3_spatial_dependence.py (some use R)
```

Table and figure scripts expect the model outputs in `Output/`; see docstrings in each script for exact inputs.

### 4. Multicollinearity checks (optional)

```bash
python multicollinearity_check.py
```

Uses the same GPKG and variable naming as the models; can be run after the pipeline and models.

---

## Main outputs

| Output | Description |
|--------|-------------|
| `Data/4/4_flood_waste_metrics_quadkey.gpkg` | Final analysis layer (quadkey + displacement, flood, waste, population) |
| `Output/flood/` | Flood-only OLS/SLM coefficients, Moran's I, diagnostics |
| `Output/wasteflood/` | Flood + waste OLS/SLM/SEM and diagnostics |
| `Output/wastefloodpop/` | Flood + waste + population models |
| `Output/summary/` | Merged Table 1 and supplementary tables (after `summarize_outputs.py`) |
| `Figure/` | Study design, spatial distribution, spatial dependence figures |
| `SCRIPT_OUTPUTS_SUMMARY.md` | Full column documentation for the final dataset |

---

## Column naming conventions

- **Script 1 (`1y_`):** `1_outflow_accumulated`, `1_outflow_accumulated_hour0`, `1_outflow_max`, `1_outflow_max_hour0`, etc.
- **Script 2 (`2y_`):** `2_outflow_max`, `2_displaced_excess_max`, etc.
- **Script 3 (`3scale_`):** `3_worldpop`, `3_estimated_outflow_pop_from_2_outflow_max`, `3_flood_p95`, etc.
- **Script 4 (`4flood_`):** `4_flood_p95`, `4_flood_mean`, `4_flood_max`, `4_waste_count`, `4_waste_per_population`, `4_waste_per_svi_count`, `4_svi_count`, etc.

For a full list and definitions, see **SCRIPT_OUTPUTS_SUMMARY.md**.

---

## License and contact

PhD UCL project. For details on data sources and licensing, see the paper and data providers (Facebook/Meta, WorldPop, etc.).
