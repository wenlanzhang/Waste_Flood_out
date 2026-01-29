#!/usr/bin/env python
"""
Combine flood and waste metrics into quadkey polygons.

This script:
1. Loads quadkey polygons from step 3 output (3_displacement_scaled_to_worldpop.gpkg)
2. Computes 95th percentile, mean, and max flood values per quadkey using zonal statistics
3. Aggregates waste point data and SVI image points to quadkey polygons
4. Computes waste metrics (count, per area, per population, per SVI)
   - All "per" metrics are scaled by 1000 for readability (units: per 1000)
5. Adds all metrics to quadkey polygons

Output:
- GeoPackage with quadkey polygons, flood metrics, and waste metrics
- Saved to: Data/4/4_flood_waste_metrics_quadkey.gpkg

Flood metric units:
- 4_flood_p95: 95th percentile flood value per quadkey
- 4_flood_mean: Mean flood value per quadkey (original/average flood value)
- 4_flood_max: Maximum flood value per quadkey

Waste metric units:
- 4_waste_count: raw count of waste points
- 4_waste_per_population: waste points per 1000 people
- 4_waste_per_svi_count: waste points per 1000 SVI images
"""

from pathlib import Path
import geopandas as gpd
import pandas as pd
import numpy as np
from rasterstats import zonal_stats
import rasterio
import warnings
warnings.filterwarnings('ignore')

# ---------- User inputs ----------
# Base directories
BASE_DIR = Path("/Users/wenlanzhang/PycharmProjects/Waste_Flood_out/")
SOURCE_DIR = BASE_DIR / "Data/3"
OUTPUT_DIR = BASE_DIR / "Data/4"
WASTE_DIR = Path("/Users/wenlanzhang/Downloads/PhD_UCL/Data/Waste_flood/Waste")

# Input files
DISPLACEMENT_GPKG = SOURCE_DIR / "3_displacement_scaled_to_worldpop.gpkg"  # Reference quadkey polygons
DISPLACEMENT_LAYER = "3_displacement_scaled"
FLOOD_RASTER = Path("/Users/wenlanzhang/Downloads/PhD_UCL/Data/Waste/FastFlood/model8.tif")
WASTE_POINTS = WASTE_DIR / "aoi_waste.gpkg"  # Point data in EPSG:4326
SVI_POINTS = WASTE_DIR / "aoi_all_svi.gpkg"  # SVI image points in EPSG:4326

# Output file
OUT_GPKG = OUTPUT_DIR / "4_flood_waste_metrics_quadkey.gpkg"

# Parameters
PERCENTILE = 95  # 95th percentile for flood
NODATA_VALUE = -9999.0  # Nodata value for waste metrics where no SVI data

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# PART 1: FLOOD METRICS
# ============================================================================
print("="*60)
print("PART 1: Computing Flood Metrics")
print("="*60)

# ---------- Check input files ----------
if not DISPLACEMENT_GPKG.exists():
    raise FileNotFoundError(f"Displacement GPKG not found: {DISPLACEMENT_GPKG}")
if not FLOOD_RASTER.exists():
    raise FileNotFoundError(f"Flood raster not found: {FLOOD_RASTER}")

# ---------- Load quadkey polygons (reference geometry) ----------
print("\nLoading quadkey polygons from displacement file...")
quadkey_gdf = gpd.read_file(DISPLACEMENT_GPKG, layer=DISPLACEMENT_LAYER)
print(f"  Loaded {len(quadkey_gdf):,} quadkey polygons")
print(f"  CRS: {quadkey_gdf.crs}")

# Ensure quadkey column exists
if 'quadkey' not in quadkey_gdf.columns:
    raise ValueError("Displacement GPKG must have 'quadkey' column.")

# ---------- Read flood raster properties ----------
print("\nReading flood raster properties...")
with rasterio.open(FLOOD_RASTER) as flood_src:
    flood_crs = flood_src.crs
    flood_transform = flood_src.transform
    flood_width = flood_src.width
    flood_height = flood_src.height
    flood_bounds = flood_src.bounds
    flood_nodata = flood_src.nodata
    flood_dtype = flood_src.dtypes[0]
    flood_count = flood_src.count
    
print(f"  Flood raster:")
print(f"    CRS: {flood_crs}")
print(f"    Resolution: {flood_transform[0]:.2f} x {abs(flood_transform[4]):.2f} meters")
print(f"    Shape: {flood_height} x {flood_width} pixels")
print(f"    Bounds: {flood_bounds}")
print(f"    Nodata: {flood_nodata}")
print(f"    Data type: {flood_dtype}")
print(f"    Bands: {flood_count}")

if flood_count > 1:
    print(f"\n⚠️  Warning: Flood raster has {flood_count} bands. Processing first band only.")

# ---------- Reproject quadkey polygons to flood raster CRS if needed ----------
if quadkey_gdf.crs != flood_crs:
    print(f"\n🔄 Reprojecting quadkey polygons to flood raster CRS: {flood_crs}")
    quadkey_gdf_reproj = quadkey_gdf.to_crs(flood_crs)
    print(f"  Reprojected {len(quadkey_gdf_reproj):,} polygons")
else:
    quadkey_gdf_reproj = quadkey_gdf.copy()
    print(f"\n✅ Quadkey polygons already in flood raster CRS")

# ---------- Compute zonal statistics for flood ----------
print(f"\nComputing {PERCENTILE}th percentile, mean, and max flood values per quadkey...")
print(f"  This may take a few minutes for large datasets...")

# Compute zonal statistics (percentile, mean, and max)
zs = zonal_stats(
    vectors=quadkey_gdf_reproj.geometry,
    raster=str(FLOOD_RASTER),
    stats=[f'percentile_{PERCENTILE}', 'mean', 'max'],
    nodata=flood_nodata,
    all_touched=True,
    geojson_out=False
)

# Extract percentile values
flood_p95_values = np.array([
    v[f'percentile_{PERCENTILE}'] if v[f'percentile_{PERCENTILE}'] is not None else np.nan
    for v in zs
])

# Extract mean values (original flood value)
flood_mean_values = np.array([
    v['mean'] if v['mean'] is not None else np.nan
    for v in zs
])

# Extract max values (maximum flood value)
flood_max_values = np.array([
    v['max'] if v['max'] is not None else np.nan
    for v in zs
])

# Add to quadkey_gdf (in original CRS)
quadkey_gdf[f'4_flood_p{PERCENTILE}'] = flood_p95_values
quadkey_gdf['4_flood_mean'] = flood_mean_values
quadkey_gdf['4_flood_max'] = flood_max_values

print(f"  ✅ Computed flood percentiles, mean, and max values for {len(quadkey_gdf):,} quadkeys")

# ---------- Summary statistics for flood ----------
valid_flood_p95 = quadkey_gdf[quadkey_gdf[f'4_flood_p{PERCENTILE}'].notna()]
valid_flood_mean = quadkey_gdf[quadkey_gdf['4_flood_mean'].notna()]
print(f"\nFlood {PERCENTILE}th Percentile:")
print(f"  Quadkeys with valid flood data: {len(valid_flood_p95):,}")
print(f"  Quadkeys with no flood data: {(quadkey_gdf[f'4_flood_p{PERCENTILE}'].isna()).sum():,}")

if len(valid_flood_p95) > 0:
    print(f"  Min: {valid_flood_p95[f'4_flood_p{PERCENTILE}'].min():.4f}")
    print(f"  Max: {valid_flood_p95[f'4_flood_p{PERCENTILE}'].max():.4f}")
    print(f"  Mean: {valid_flood_p95[f'4_flood_p{PERCENTILE}'].mean():.4f}")
    print(f"  Median: {valid_flood_p95[f'4_flood_p{PERCENTILE}'].median():.4f}")

print(f"\nFlood Mean (Original Value):")
print(f"  Quadkeys with valid flood data: {len(valid_flood_mean):,}")
print(f"  Quadkeys with no flood data: {(quadkey_gdf['4_flood_mean'].isna()).sum():,}")

if len(valid_flood_mean) > 0:
    print(f"  Min: {valid_flood_mean['4_flood_mean'].min():.4f}")
    print(f"  Max: {valid_flood_mean['4_flood_mean'].max():.4f}")
    print(f"  Mean: {valid_flood_mean['4_flood_mean'].mean():.4f}")
    print(f"  Median: {valid_flood_mean['4_flood_mean'].median():.4f}")

valid_flood_max = quadkey_gdf[quadkey_gdf['4_flood_max'].notna()]
print(f"\nFlood Max (Maximum Value):")
print(f"  Quadkeys with valid flood data: {len(valid_flood_max):,}")
print(f"  Quadkeys with no flood data: {(quadkey_gdf['4_flood_max'].isna()).sum():,}")

if len(valid_flood_max) > 0:
    print(f"  Min: {valid_flood_max['4_flood_max'].min():.4f}")
    print(f"  Max: {valid_flood_max['4_flood_max'].max():.4f}")
    print(f"  Mean: {valid_flood_max['4_flood_max'].mean():.4f}")
    print(f"  Median: {valid_flood_max['4_flood_max'].median():.4f}")

# ============================================================================
# PART 2: WASTE METRICS
# ============================================================================
print("\n" + "="*60)
print("PART 2: Computing Waste Metrics")
print("="*60)

# ---------- Check input files ----------
if not WASTE_POINTS.exists():
    raise FileNotFoundError(f"Waste points file not found: {WASTE_POINTS}")
if not SVI_POINTS.exists():
    raise FileNotFoundError(f"SVI points file not found: {SVI_POINTS}")

# Store original CRS
original_crs = quadkey_gdf.crs
print(f"  Original CRS: {original_crs}")

# Target CRS for aggregation (use original CRS)
target_crs = original_crs
print(f"\n  Target CRS for aggregation: {target_crs}")

# ---------- Load waste points ----------
print("\nLoading waste points...")
waste_gdf = gpd.read_file(WASTE_POINTS)
print(f"  Loaded {len(waste_gdf):,} waste points")
print(f"  Original CRS: {waste_gdf.crs}")

# Reproject to target CRS
if waste_gdf.crs != target_crs:
    print(f"  Reprojecting from {waste_gdf.crs} to {target_crs}...")
    waste_gdf = waste_gdf.to_crs(target_crs)
    print(f"  ✅ Reprojected to {target_crs}")
else:
    print(f"  ✅ Already in target CRS")

print(f"  Waste points bounds: {waste_gdf.total_bounds}")

# ---------- Load SVI points ----------
print("\nLoading SVI image points...")
svi_gdf = gpd.read_file(SVI_POINTS)
print(f"  Loaded {len(svi_gdf):,} SVI image points")
print(f"  Original CRS: {svi_gdf.crs}")

# Reproject to target CRS
if svi_gdf.crs != target_crs:
    print(f"  Reprojecting from {svi_gdf.crs} to {target_crs}...")
    svi_gdf = svi_gdf.to_crs(target_crs)
    print(f"  ✅ Reprojected to {target_crs}")
else:
    print(f"  ✅ Already in target CRS")

print(f"  SVI points bounds: {svi_gdf.total_bounds}")

# ---------- Count waste points per quadkey ----------
print(f"\nCounting waste points per quadkey...")
print(f"  Performing spatial join (this may take a moment)...")

# Spatial join to count points per quadkey
waste_in_quadkeys = gpd.sjoin(waste_gdf, quadkey_gdf, how='inner', predicate='within')
print(f"  Found {len(waste_in_quadkeys):,} waste points within quadkey boundaries")

# Count points per quadkey
waste_counts = waste_in_quadkeys.groupby('quadkey').size().reset_index(name='4_waste_count')
print(f"  Quadkeys with waste: {len(waste_counts):,}")

# Merge counts back to quadkey_gdf
quadkey_gdf = quadkey_gdf.merge(waste_counts, on='quadkey', how='left')
quadkey_gdf['4_waste_count'] = quadkey_gdf['4_waste_count'].fillna(0).astype(int)

total_waste = quadkey_gdf['4_waste_count'].sum()
print(f"  Total waste points counted: {total_waste:,}")
print(f"  Quadkeys with waste > 0: {(quadkey_gdf['4_waste_count'] > 0).sum():,}")

# ---------- Count SVI images per quadkey ----------
print(f"\nCounting SVI images per quadkey...")
print(f"  Performing spatial join (this may take a moment)...")

# Spatial join to count SVI points per quadkey
svi_in_quadkeys = gpd.sjoin(svi_gdf, quadkey_gdf, how='inner', predicate='within')
print(f"  Found {len(svi_in_quadkeys):,} SVI images within quadkey boundaries")

# Count SVI images per quadkey
svi_counts = svi_in_quadkeys.groupby('quadkey').size().reset_index(name='4_svi_count')
print(f"  Quadkeys with SVI images: {len(svi_counts):,}")

# Merge SVI counts back to quadkey_gdf
quadkey_gdf = quadkey_gdf.merge(svi_counts, on='quadkey', how='left')
quadkey_gdf['4_svi_count'] = quadkey_gdf['4_svi_count'].fillna(0).astype(int)

total_svi = quadkey_gdf['4_svi_count'].sum()
quadkeys_with_svi = (quadkey_gdf['4_svi_count'] > 0).sum()
quadkeys_without_svi = (quadkey_gdf['4_svi_count'] == 0).sum()
print(f"  Total SVI images counted: {total_svi:,}")
print(f"  Quadkeys with SVI > 0: {quadkeys_with_svi:,}")
print(f"  Quadkeys without SVI: {quadkeys_without_svi:,}")

# ---------- Compute waste metrics ----------
print(f"\nComputing waste metrics...")
print(f"  Logic: no SVI data → nodata (-9999), SVI but no waste → 0")

# All metrics follow the same logic:
# - If no SVI data (svi_count == 0) → nodata (-9999)
# - If SVI data exists (svi_count > 0) → calculate metric (can be 0 if no waste)

# 1. Waste count (already computed, but set nodata where no SVI)
quadkey_gdf['4_waste_count_final'] = np.where(
    quadkey_gdf['4_svi_count'] == 0,
    NODATA_VALUE,
    quadkey_gdf['4_waste_count'].astype(float)
)

# 2. Waste per WorldPop population (waste_count / worldpop) - scaled by 1000 for readability
# Units: waste points per 1000 people
# Check if worldpop column exists (from script 3, it's 3_worldpop)
if '3_worldpop' in quadkey_gdf.columns:
    quadkey_gdf['4_waste_per_population'] = np.where(
        quadkey_gdf['4_svi_count'] == 0,
        NODATA_VALUE,  # No SVI data → nodata (-9999), don't multiply
        np.where(
            (quadkey_gdf['3_worldpop'] > 0) & 
            (quadkey_gdf['3_worldpop'].notna()) &
            (quadkey_gdf['4_waste_count'].notna()),
            (quadkey_gdf['4_waste_count'] / quadkey_gdf['3_worldpop']) * 1000,
            0.0  # Valid SVI but invalid worldpop → 0, don't multiply
        )
    )
else:
    print("  ⚠️  Warning: '3_worldpop' column not found, skipping waste_per_population")
    quadkey_gdf['4_waste_per_population'] = NODATA_VALUE

# 3. Waste per SVI count (waste_count / svi_count) - scaled by 1000 for readability
# Units: waste points per 1000 SVI images
quadkey_gdf['4_waste_per_svi_count'] = np.where(
    quadkey_gdf['4_svi_count'] == 0,
    NODATA_VALUE,  # No SVI data → nodata (-9999), don't multiply
    np.where(
        (quadkey_gdf['4_svi_count'] > 0) &
        (quadkey_gdf['4_waste_count'].notna()),
        (quadkey_gdf['4_waste_count'] / quadkey_gdf['4_svi_count']) * 1000,
        0.0  # Valid SVI but invalid waste_count → 0, don't multiply
    )
)

print(f"  ✅ Metrics computed")

# ============================================================================
# SUMMARY AND SAVE
# ============================================================================
print("\n" + "="*60)
print("Summary Statistics")
print("="*60)

# Flood summary
valid_flood_p95 = quadkey_gdf[quadkey_gdf[f'4_flood_p{PERCENTILE}'].notna()]
valid_flood_mean = quadkey_gdf[quadkey_gdf['4_flood_mean'].notna()]
valid_flood_max = quadkey_gdf[quadkey_gdf['4_flood_max'].notna()]
print(f"\nFlood {PERCENTILE}th Percentile:")
print(f"  Quadkeys with valid flood data: {len(valid_flood_p95):,}")
print(f"  Quadkeys with no flood data: {(quadkey_gdf[f'4_flood_p{PERCENTILE}'].isna()).sum():,}")

print(f"\nFlood Mean (Original Value):")
print(f"  Quadkeys with valid flood data: {len(valid_flood_mean):,}")
print(f"  Quadkeys with no flood data: {(quadkey_gdf['4_flood_mean'].isna()).sum():,}")

print(f"\nFlood Max (Maximum Value):")
print(f"  Quadkeys with valid flood data: {len(valid_flood_max):,}")
print(f"  Quadkeys with no flood data: {(quadkey_gdf['4_flood_max'].isna()).sum():,}")

# Waste summary
valid_waste = quadkey_gdf[quadkey_gdf['4_waste_count_final'] != NODATA_VALUE]
print(f"\nWaste Count (where SVI data exists):")
print(f"  Total waste points: {valid_waste['4_waste_count'].sum():,}")
print(f"  Quadkeys with waste > 0: {(valid_waste['4_waste_count'] > 0).sum():,}")
print(f"  Quadkeys with waste = 0: {(valid_waste['4_waste_count'] == 0).sum():,}")
print(f"  Quadkeys with no SVI data (nodata): {(quadkey_gdf['4_svi_count'] == 0).sum():,}")

# Waste per SVI count
valid_waste_per_svi = quadkey_gdf[quadkey_gdf['4_waste_per_svi_count'] != NODATA_VALUE]
if len(valid_waste_per_svi) > 0:
    print(f"\nWaste per SVI Count:")
    print(f"  Min: {valid_waste_per_svi['4_waste_per_svi_count'].min():.6f}")
    print(f"  Max: {valid_waste_per_svi['4_waste_per_svi_count'].max():.6f}")
    print(f"  Mean: {valid_waste_per_svi['4_waste_per_svi_count'].mean():.6f}")
    print(f"  Median: {valid_waste_per_svi['4_waste_per_svi_count'].median():.6f}")

# ---------- Save output ----------
print("\n" + "="*60)
print("Saving Output")
print("="*60)

print(f"\nSaving {len(quadkey_gdf):,} quadkeys with flood and waste metrics to: {OUT_GPKG}")

# Save to GeoPackage
quadkey_gdf.to_file(OUT_GPKG, driver='GPKG', layer='4_flood_waste_metrics_quadkey')
print(f"✅ Saved to: {OUT_GPKG}")

print("\n" + "="*60)
print("Done!")
print("="*60)
print(f"\nOutput file: {OUT_GPKG}")
print(f"  Total quadkeys: {len(quadkey_gdf):,}")
print(f"  Quadkeys with flood p95 data: {len(valid_flood_p95):,}")
print(f"  Quadkeys with flood mean data: {len(valid_flood_mean):,}")
print(f"  Quadkeys with flood max data: {len(valid_flood_max):,}")
print(f"  Quadkeys with waste: {(quadkey_gdf['4_waste_count'] > 0).sum():,}")
print(f"  Quadkeys with SVI data: {quadkeys_with_svi:,}")
