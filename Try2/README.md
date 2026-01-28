# Try2: Streamlined Analysis Pipeline

This directory contains a streamlined version of the flood-waste-displacement analysis.

## Scripts

1. **01_load_and_prepare_data.py**: Loads the final dataset and prepares variables
2. **02_run_regression.py**: Runs regression models with different specifications
3. **run_all.py**: Runs all scripts in sequence

## Usage

Run in your Python environment (e.g., geo_env_LLM):

```bash
cd Try2
python 02_run_regression.py
```

Or run both scripts:
```bash
python 01_load_and_prepare_data.py
python 02_run_regression.py
```

## What it does

1. Loads `Data/4/4_flood_waste_metrics_quadkey.gpkg`
2. Tests multiple variable specifications:
   - Different Y variables (Step 1 vs Step 2 displacement)
   - Different waste variables (count vs per_population vs per_svi)
3. Runs OLS regression with robust standard errors
4. Tests for spatial autocorrelation (Moran's I)
5. Runs spatial models (SLM/SEM) if spatial autocorrelation is detected
6. Displays results (no files saved)

## Variable Specifications Tested

1. **Main Specification**:
   - Y: `3_estimated_outflow_pop_from_1_outflow_accumulated_hour0` (Step 1)
   - Flood: `4_flood_p95`
   - Waste: `4_waste_per_population`

2. **CSAT Method**:
   - Y: `3_estimated_outflow_pop_from_2_outflow_max` (Step 2)
   - Flood: `4_flood_p95`
   - Waste: `4_waste_per_population`

3. **Waste Count**:
   - Y: `3_estimated_outflow_pop_from_1_outflow_accumulated_hour0`
   - Flood: `4_flood_p95`
   - Waste: `4_waste_count` (not normalized)

4. **Waste per SVI**:
   - Y: `3_estimated_outflow_pop_from_1_outflow_accumulated_hour0`
   - Flood: `4_flood_p95`
   - Waste: `4_waste_per_svi_count`

## Expected Output

The script will display:
- Model coefficients and significance
- R² and adjusted R²
- Moran's I test results
- Spatial model results (if spatial autocorrelation detected)
- Summary table comparing all specifications
