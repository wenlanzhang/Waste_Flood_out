#!/usr/bin/env python3
"""
Quick test script - minimal version to verify data loading works
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd

print("="*80)
print("QUICK TEST: Data Loading")
print("="*80)

BASE_DIR = Path("/Users/wenlanzhang/PycharmProjects/Waste_Flood_out")
DATA_DIR = BASE_DIR / "Data" / "4"
GPKG_PATH = DATA_DIR / "4_flood_waste_metrics_quadkey.gpkg"
LAYER_NAME = "4_flood_waste_metrics_quadkey"

print(f"Loading: {GPKG_PATH}")
gdf = gpd.read_file(GPKG_PATH, layer=LAYER_NAME)
print(f"✓ Loaded {len(gdf):,} quadkeys")
print(f"  Columns: {len(gdf.columns)}")

# Check key variables
key_vars = [
    '3_estimated_outflow_pop_from_1_outflow_accumulated_hour0',
    '3_estimated_outflow_pop_from_2_outflow_max',
    '4_flood_p95',
    '4_waste_count',
    '4_waste_per_population',
]

print("\nKey variables:")
for var in key_vars:
    if var in gdf.columns:
        n_valid = gdf[var].notna().sum()
        print(f"  ✓ {var}: {n_valid:,} valid values")
    else:
        print(f"  ✗ {var}: NOT FOUND")

# Quick stats
print("\nQuick statistics:")
for var in ['3_estimated_outflow_pop_from_1_outflow_accumulated_hour0', 
            '4_flood_p95', '4_waste_per_population']:
    if var in gdf.columns:
        valid = gdf[var].dropna()
        if len(valid) > 0:
            print(f"\n{var}:")
            print(f"  Mean: {valid.mean():.2f}")
            print(f"  Median: {valid.median():.2f}")
            print(f"  Min: {valid.min():.2f}")
            print(f"  Max: {valid.max():.2f}")

print("\n" + "="*80)
print("Data loading successful! Ready to run regression.")
print("="*80)
