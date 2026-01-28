#! /Users/wenlanzhang/miniconda3/envs/geo_env_LLM/bin/python
"""
01_load_and_prepare_data.py

Load final dataset and prepare variables for modeling.
This script loads the combined flood-waste-displacement data and prepares
clean variables for regression analysis.

Output: Clean dataframe ready for modeling
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd

# ============================================================================
# PATHS
# ============================================================================
BASE_DIR = Path("/Users/wenlanzhang/PycharmProjects/Waste_Flood_out")
DATA_DIR = BASE_DIR / "Data" / "4"
GPKG_PATH = DATA_DIR / "4_flood_waste_metrics_quadkey.gpkg"
LAYER_NAME = "4_flood_waste_metrics_quadkey"

# ============================================================================
# LOAD DATA
# ============================================================================
print("="*80)
print("LOADING DATA")
print("="*80)
print(f"Loading from: {GPKG_PATH}")
print(f"Layer: {LAYER_NAME}")

gdf = gpd.read_file(GPKG_PATH, layer=LAYER_NAME)
print(f"✓ Loaded {len(gdf):,} quadkeys")
print(f"  CRS: {gdf.crs}")
print(f"  Columns: {len(gdf.columns)}")

# ============================================================================
# EXPLORE AVAILABLE VARIABLES
# ============================================================================
print("\n" + "="*80)
print("AVAILABLE VARIABLES")
print("="*80)

# Group by prefix
prefixes = ['1_', '2_', '3_', '4_']
for prefix in prefixes:
    cols = [c for c in gdf.columns if c.startswith(prefix)]
    if cols:
        print(f"\n{prefix} prefix ({len(cols)} variables):")
        for col in sorted(cols)[:10]:  # Show first 10
            print(f"  - {col}")
        if len(cols) > 10:
            print(f"  ... and {len(cols) - 10} more")

# ============================================================================
# PREPARE MODELING VARIABLES
# ============================================================================
print("\n" + "="*80)
print("PREPARING MODELING VARIABLES")
print("="*80)

# Create a clean dataframe for modeling
model_df = gdf.copy()

# Handle NODATA values (-9999)
NODATA = -9999.0
numeric_cols = model_df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if col in model_df.columns:
        model_df[col] = pd.to_numeric(model_df[col], errors='coerce')
        model_df.loc[model_df[col] == NODATA, col] = np.nan

# ============================================================================
# DEFINE VARIABLE OPTIONS
# ============================================================================

# Y Variable Options (Displacement)
Y_OPTIONS = {
    'step1_accumulated': '3_estimated_outflow_pop_from_1_outflow_accumulated_hour0',
    'step1_max': '3_estimated_outflow_pop_from_1_outflow_max_hour0',
    'step2_csat': '3_estimated_outflow_pop_from_2_outflow_max',
    'step1_pct': '3_pct_outflow_worldpop_from_1_outflow_accumulated_hour0',
    'step2_pct': '3_pct_outflow_worldpop_from_2_outflow_max',
}

# Flood Variable Options
FLOOD_OPTIONS = {
    'p95': '4_flood_p95',
    'p95_log': None,  # Will create log version
}

# Waste Variable Options
WASTE_OPTIONS = {
    'count': '4_waste_count',
    'per_area': '4_waste_per_quadkey_area',
    'per_pop': '4_waste_per_population',
    'per_svi': '4_waste_per_svi_count',
}

# ============================================================================
# CREATE TRANSFORMED VARIABLES
# ============================================================================
print("\nCreating transformed variables...")

# Log transforms (log1p handles zeros)
for col in ['4_flood_p95', '4_waste_count']:
    if col in model_df.columns:
        model_df[f'{col}_log'] = np.log1p(model_df[col].fillna(0))

# Check which variables exist
print("\nChecking variable availability:")
print("\nY Variables (Displacement):")
for name, col in Y_OPTIONS.items():
    exists = col in model_df.columns
    n_valid = model_df[col].notna().sum() if exists else 0
    print(f"  {name:20s} ({col:50s}): {'✓' if exists else '✗'} ({n_valid:,} valid)")

print("\nFlood Variables:")
for name, col in FLOOD_OPTIONS.items():
    if col:
        exists = col in model_df.columns
        n_valid = model_df[col].notna().sum() if exists else 0
        print(f"  {name:20s} ({col:50s}): {'✓' if exists else '✗'} ({n_valid:,} valid)")

print("\nWaste Variables:")
for name, col in WASTE_OPTIONS.items():
    exists = col in model_df.columns
    n_valid = model_df[col].notna().sum() if exists else 0
    print(f"  {name:20s} ({col:50s}): {'✓' if exists else '✗'} ({n_valid:,} valid)")

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================
print("\n" + "="*80)
print("DATA SUMMARY")
print("="*80)

# Key variables for summary
key_vars = [
    '3_estimated_outflow_pop_from_1_outflow_accumulated_hour0',
    '3_estimated_outflow_pop_from_2_outflow_max',
    '4_flood_p95',
    '4_waste_count',
    '4_waste_per_population',
    '3_worldpop',
]

print("\nSummary statistics for key variables:")
for var in key_vars:
    if var in model_df.columns:
        valid = model_df[var].dropna()
        if len(valid) > 0:
            print(f"\n{var}:")
            print(f"  Valid observations: {len(valid):,}")
            print(f"  Mean:   {valid.mean():.2f}")
            print(f"  Median: {valid.median():.2f}")
            print(f"  Min:    {valid.min():.2f}")
            print(f"  Max:    {valid.max():.2f}")
            print(f"  Std:    {valid.std():.2f}")

# ============================================================================
# SAVE CLEAN DATA
# ============================================================================
print("\n" + "="*80)
print("DATA PREPARATION COMPLETE")
print("="*80)
print(f"Total quadkeys: {len(model_df):,}")
print(f"Variables prepared: {len(model_df.columns)}")

# Store in global namespace for next script
import sys
sys.modules['__main__'].model_df = model_df
sys.modules['__main__'].Y_OPTIONS = Y_OPTIONS
sys.modules['__main__'].FLOOD_OPTIONS = FLOOD_OPTIONS
sys.modules['__main__'].WASTE_OPTIONS = WASTE_OPTIONS

print("\n✓ Data ready for modeling!")
print("  Run: 02_run_regression.py")
