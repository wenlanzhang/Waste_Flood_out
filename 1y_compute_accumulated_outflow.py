#!/usr/bin/env python3

"""
================================================================================
PROJECT: PhD UCL - Waste & Flood Impact Evaluation
OUTPUT DIRECTORY: /Users/wenlanzhang/PycharmProjects/Waste_Flood_out/Data/1/

APPROACH: Continuous Outflow Detection
1. Establish "normal" for each place and time: median baseline population per quadkey-hour
2. Compare observed to expected: shortfall = max(0, baseline_median - observed)
3. Accumulate outflow across flood event: sum shortfall across all timestamps

PRIMARY OUTPUTS:
1. agg_outflow_accumulated.gpkg      : Final spatial output
   - 'outflow_accumulated'            : Total shortfall accumulated across flood event (all hours)
   - 'outflow_accumulated_hour0'      : Total shortfall accumulated at hour=0 only (for world pop association)
   - 'outflow_max'                    : Maximum shortfall at any single timestamp (all hours)
   - 'outflow_max_hour0'              : Maximum shortfall at hour=0 only
   - 'n_timestamps'                   : Number of timestamps with outflow
   - 'n_valid_timestamps_hour0'       : Number of valid timestamps at hour=0
   
2. rows_with_outflow.csv             : All timestamps with shortfall values
3. baseline_medians_per_cell_hour.csv : Historical baseline medians per quadkey-hour
================================================================================
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path

# ---------- PARAMETERS ----------
FB_WIDE_PATH = '/Users/wenlanzhang/Downloads/PhD_UCL/Data/Waste_flood/Meta/FB_32737_wide.gpkg'
FB_WIDE_LAYER = 'population_change'
OUT_DIR = Path('/Users/wenlanzhang/PycharmProjects/Waste_Flood_out/Data/1/')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Analysis choices
MIN_BASELINE_SAMPLES = 3              # minimum samples needed to compute reliable baseline median
# Crisis week timestamps (these will be excluded from baseline computation, but used for outflow accumulation)
CRISIS_WEEK_PREFIXES = [
    '20240430_1600','20240501_0000','20240502_0800',
    '20240502_1600','20240503_0000'
]
BASELINE_SUFFIX = '_n_baseline'
CRISIS_SUFFIX = '_n_crisis'

# ---------- Helper functions ----------
def extract_timestamp_prefix(col):
    """Extract timestamp prefix from column name (e.g., '20240430_1600_n_baseline' -> '20240430_1600')."""
    try:
        return str(col).split('_n_')[0]
    except Exception:
        return None

def parse_prefix_to_dt(prefix):
    """Parse timestamp prefix to datetime (e.g., '20240430_1600' -> datetime)."""
    try:
        return pd.to_datetime(prefix, format='%Y%m%d_%H%M')
    except Exception:
        return pd.NaT

# ---------- wide -> long paired ----------
def wide_to_long_paired(fb_wide, baseline_suffix='_n_baseline', crisis_suffix='_n_crisis', exclude_prefixes=None):
    """
    Convert wide format to long format with paired baseline/crisis columns.
    Returns DataFrame with columns: quadkey, ts_prefix, dt, hour, baseline, crisis
    """
    cols = fb_wide.columns.tolist()
    baseline_cols = [c for c in cols if str(c).endswith(baseline_suffix)]
    crisis_cols = [c for c in cols if str(c).endswith(crisis_suffix)]
    baseline_map = {extract_timestamp_prefix(c): c for c in baseline_cols}
    crisis_map = {extract_timestamp_prefix(c): c for c in crisis_cols}
    matching_prefixes = sorted(set(baseline_map.keys()) & set(crisis_map.keys()))
    records = []
    exclude_set = set(exclude_prefixes) if exclude_prefixes else set()
    for prefix in matching_prefixes:
        if prefix in exclude_set:
            continue
        bcol = baseline_map[prefix]
        ccol = crisis_map[prefix]
        dt = parse_prefix_to_dt(prefix)
        hour = int(dt.hour) if pd.notna(dt) else None
        bl = pd.to_numeric(fb_wide[bcol], errors='coerce')
        cr = pd.to_numeric(fb_wide[ccol], errors='coerce')
        tmp = pd.DataFrame({
            'quadkey': fb_wide['quadkey'],
            'ts_prefix': prefix,
            'dt': dt,
            'hour': hour,
            'baseline': bl,
            'crisis': cr
        })
        records.append(tmp)
    if len(records) == 0:
        return pd.DataFrame(columns=['quadkey','ts_prefix','dt','hour','baseline','crisis'])
    long_df = pd.concat(records, ignore_index=True)
    # Ensure hour is nullable integer type
    long_df['hour'] = long_df['hour'].astype('Int64')
    return long_df

# ---------- Compute baseline medians per quadkey-hour ----------
def compute_baseline_medians_per_cell_hour(
    fb_wide,
    min_baseline_samples=3,
    crisis_week_prefixes=None,
    baseline_suffix='_n_baseline',
    crisis_suffix='_n_crisis'
):
    """
    Step 1: Establish "normal" for each place and time.
    For each quadkey and hour, compute median baseline population from BASELINE WEEK ONLY.
    Uses only the 'baseline' column values from baseline week timestamps (excludes crisis week).
    
    Returns DataFrame with columns: quadkey, hour, baseline_median, n_samples, low_confidence
    """
    print("Computing baseline medians per quadkey-hour from BASELINE WEEK ONLY...")
    
    # Build sample from baseline week only (exclude crisis week timestamps)
    paired = wide_to_long_paired(fb_wide, baseline_suffix, crisis_suffix, exclude_prefixes=crisis_week_prefixes)
    if paired.empty:
        raise ValueError("No matching baseline/crisis pairs found in fb_wide (after excluding crisis week).")
    
    # Filter to valid baseline values
    # IMPORTANT: We use the 'baseline' column values from baseline week only
    paired_valid = paired.dropna(subset=['baseline', 'hour']).copy()
    
    print(f"Using {len(paired_valid)} baseline week observations to compute medians")
    
    # Group by quadkey and hour, compute median baseline
    # Ensure hour is integer for grouping
    paired_valid['hour'] = paired_valid['hour'].astype('Int64')  # Nullable integer type
    
    rows = []
    grouped = paired_valid.groupby(['quadkey', 'hour'], sort=False)
    for (qk, hr), grp in grouped:
        baseline_vals = grp['baseline'].values.astype(float)
        n_samples = int(np.sum(~np.isnan(baseline_vals)))
        
        if n_samples >= min_baseline_samples:
            baseline_median = np.nanmedian(baseline_vals)
            low_confidence = False
        else:
            baseline_median = np.nan
            low_confidence = True
        
        rows.append({
            'quadkey': qk,
            'hour': int(hr) if pd.notna(hr) else None,  # Ensure integer
            '1_baseline_median': float(baseline_median) if pd.notna(baseline_median) else np.nan,
            '1_n_samples': n_samples,
            '1_low_confidence': low_confidence
        })
    
    baseline_df = pd.DataFrame(rows)
    # Ensure baseline_median is numeric dtype
    baseline_df['1_baseline_median'] = pd.to_numeric(baseline_df['1_baseline_median'], errors='coerce')
    baseline_df['hour'] = baseline_df['hour'].astype('Int64')
    
    print(f"Computed baselines for {len(baseline_df)} quadkey-hour combinations")
    print(f"  - High confidence: {(~baseline_df['1_low_confidence']).sum()}")
    print(f"  - Low confidence: {baseline_df['1_low_confidence'].sum()}")
    return baseline_df

# ---------- Compute shortfall for all timestamps ----------
def compute_shortfall_all_timestamps(fb_wide, baseline_medians_df,
                                     crisis_week_prefixes=None,
                                     baseline_suffix='_n_baseline',
                                     crisis_suffix='_n_crisis',
                                     enforce_confidence=False):
    """
    Step 2: Compare observed population to what is normally expected.
    For each timestamp, compute shortfall = max(0, baseline_median - observed).
    This is continuous, not binary - every timestamp gets a value.
    
    Also marks which timestamps are from crisis week vs baseline week.
    
    Returns DataFrame with columns: quadkey, ts_prefix, dt, hour, baseline, crisis, 
                                     baseline_median, shortfall, low_confidence, is_crisis_week
    """
    print("Computing shortfall for all timestamps...")
    
    # Use ALL rows (both baseline and crisis week)
    paired = wide_to_long_paired(fb_wide, baseline_suffix, crisis_suffix, exclude_prefixes=None)
    if paired.empty:
        raise ValueError("No matching baseline/crisis pairs found in fb_wide.")
    
    # Mark which timestamps are from crisis week
    crisis_set = set(crisis_week_prefixes) if crisis_week_prefixes else set()
    paired['is_crisis_week'] = paired['ts_prefix'].isin(crisis_set)
    
    print(f"Total timestamps: {len(paired)}")
    print(f"  - Baseline week: {(~paired['is_crisis_week']).sum()}")
    print(f"  - Crisis week: {paired['is_crisis_week'].sum()}")
    
    # Ensure hour is integer for merging
    merged = paired.copy()
    merged['hour'] = merged['hour'].astype('Int64')
    baseline_medians_df['hour'] = baseline_medians_df['hour'].astype('Int64')
    
    # Merge with baseline medians
    merged = merged.merge(baseline_medians_df[['quadkey', 'hour', '1_baseline_median', '1_low_confidence']], 
                         how='left', on=['quadkey', 'hour'])
    
    # Fill missing low_confidence with True (no baseline found)
    if '1_low_confidence' in merged.columns:
        mask = merged['1_low_confidence'].isna()
        merged.loc[mask, '1_low_confidence'] = True
        merged['1_low_confidence'] = merged['1_low_confidence'].astype(bool)
    else:
        merged['1_low_confidence'] = True
    
    # Ensure baseline_median is numeric
    merged['1_baseline_median'] = pd.to_numeric(merged['1_baseline_median'], errors='coerce')
    
    # Step 3: Compute shortfall as continuous quantity
    # shortfall = max(0, baseline_median - observed)
    # where observed is the 'crisis' column value (observed population at that timestamp)
    # This represents how many fewer people are here than normal
    # 
    # Default behavior (enforce_confidence=False): Set shortfall to NaN if:
    #   - baseline_median is missing/NaN (includes low-confidence baselines, which have NaN baseline_median)
    #   - crisis is missing/NaN
    # This exposes uncertainty rather than hiding it with zeros.
    # 
    # If enforce_confidence=True: Also exclude low-confidence baselines even if baseline_median exists
    # (defensive programming, though low_confidence should always imply NaN baseline_median)
    
    # Initialize shortfall as NaN (default for all rows)
    merged['1_shortfall'] = np.nan
    
    # Compute shortfall only when both baseline_median and crisis are valid (not NaN)
    valid_mask = (merged['1_baseline_median'].notna()) & (merged['crisis'].notna())
    
    if enforce_confidence:
        # If enforcing confidence, also exclude low-confidence baselines from computation
        # (Note: low_confidence should always imply NaN baseline_median, but this is defensive)
        valid_mask = valid_mask & (~merged['1_low_confidence'])
    
    # Compute shortfall for valid cases only
    # All other cases remain NaN (missing data or low confidence)
    merged.loc[valid_mask, '1_shortfall'] = np.maximum(
        0.0, 
        merged.loc[valid_mask, '1_baseline_median'] - merged.loc[valid_mask, 'crisis']
    )
    
    # Count affected quadkeys
    n_quadkeys_with_low_confidence = merged[merged['1_low_confidence']]['quadkey'].nunique()
    n_quadkeys_with_missing_baseline = merged[merged['1_baseline_median'].isna()]['quadkey'].nunique()
    n_quadkeys_with_missing_crisis = merged[merged['crisis'].isna()]['quadkey'].nunique()
    
    print(f"Computed shortfall for {len(merged)} timestamp-cell combinations")
    print(f"  - Valid shortfall values: {merged['1_shortfall'].notna().sum()}")
    print(f"  - NaN shortfall (missing baseline_median): {merged['1_baseline_median'].isna().sum()}")
    print(f"  - NaN shortfall (missing crisis): {merged['crisis'].isna().sum()}")
    print(f"  - NaN shortfall (low-confidence baseline): {merged[merged['1_low_confidence'] & merged['1_baseline_median'].notna()]['1_shortfall'].isna().sum()}")
    print(f"  - Quadkeys affected by low-confidence baselines: {n_quadkeys_with_low_confidence}")
    print(f"  - Quadkeys with missing baseline_median: {n_quadkeys_with_missing_baseline}")
    print(f"  - Quadkeys with missing crisis data: {n_quadkeys_with_missing_crisis}")
    
    # Only sum non-NaN values for reporting
    valid_shortfall = merged['1_shortfall'].dropna()
    if len(valid_shortfall) > 0:
        print(f"Total shortfall (valid values only): {valid_shortfall.sum():.2f}")
        crisis_week_valid = merged[merged['is_crisis_week']]['1_shortfall'].dropna()
        if len(crisis_week_valid) > 0:
            print(f"Total shortfall from crisis week (valid values only): {crisis_week_valid.sum():.2f}")
    
    return merged

# ---------- Accumulate outflow across flood event ----------
def accumulate_outflow_per_quadkey(shortfall_df):
    """
    Step 4: Accumulate outflow across the flood event.
    Sum all shortfall values across CRISIS WEEK timestamps only for each quadkey.
    This represents total population deficit experienced during the crisis event.
    
    IMPORTANT: Only accumulates from crisis week, not baseline week.
    NaN shortfall values are excluded from aggregation (they represent missing/uncertain data).
    
    Returns aggregated DataFrame with columns: 
        - quadkey: Grid cell identifier
        - outflow_accumulated: Total shortfall across all crisis week timestamps (all hours)
        - outflow_accumulated_hour0: Total shortfall across crisis week timestamps at hour=0 only
        - outflow_max: Maximum shortfall at any single timestamp (all hours)
        - outflow_max_hour0: Maximum shortfall at hour=0 only
        - outflow_mean, outflow_p95: Statistics across all hours
        - n_timestamps, n_valid_timestamps: Counts for all hours
        - n_valid_timestamps_hour0: Count of valid timestamps at hour=0
        - Other diagnostic columns
    """
    print("Accumulating outflow across flood event (CRISIS WEEK ONLY)...")
    
    # Filter to crisis week only
    if 'is_crisis_week' not in shortfall_df.columns:
        raise ValueError("shortfall_df must contain 'is_crisis_week' column")
    
    crisis_week_df = shortfall_df[shortfall_df['is_crisis_week']].copy()
    print(f"Filtering to {len(crisis_week_df)} crisis week timestamps (out of {len(shortfall_df)} total)")
    
    # Helper function to compute percentile safely (guard against empty arrays)
    def safe_percentile(x, p):
        """Compute percentile, returning NaN if array is empty or all NaN."""
        x_clean = x.dropna()
        if len(x_clean) == 0:
            return np.nan
        try:
            return np.nanpercentile(x_clean, p)
        except (ValueError, IndexError):
            return np.nan
    
    # Group by quadkey and aggregate (only from crisis week)
    # Use custom aggregation to handle NaN values properly
    agg_list = []
    grouped = crisis_week_df.groupby('quadkey', sort=False)
    
    for qk, grp in grouped:
        shortfall_vals = grp['1_shortfall'].dropna()  # Exclude NaN values
        
        # Counts
        n_timestamps = len(grp)  # Total timestamps (including NaN)
        n_valid_timestamps = len(shortfall_vals)  # Valid (non-NaN) shortfall values
        
        # Aggregations (only on valid values) - ALL HOURS
        if n_valid_timestamps > 0:
            outflow_accumulated = shortfall_vals.sum()
            outflow_max = shortfall_vals.max()
            outflow_mean = shortfall_vals.mean()
            outflow_p95 = safe_percentile(shortfall_vals, 95)
            n_timestamps_with_outflow = (shortfall_vals > 0).sum()
        else:
            outflow_accumulated = 0.0
            outflow_max = np.nan
            outflow_mean = np.nan
            outflow_p95 = np.nan
            n_timestamps_with_outflow = 0
        
        # Aggregations for HOUR=0 ONLY
        grp_hour0 = grp[grp['hour'] == 0]
        shortfall_vals_hour0 = grp_hour0['1_shortfall'].dropna()
        n_valid_timestamps_hour0 = len(shortfall_vals_hour0)
        
        if n_valid_timestamps_hour0 > 0:
            outflow_accumulated_hour0 = shortfall_vals_hour0.sum()
            outflow_max_hour0 = shortfall_vals_hour0.max()
        else:
            outflow_accumulated_hour0 = 0.0
            outflow_max_hour0 = np.nan
        
        # Other statistics (on all values, including NaN)
        baseline_median_mean = grp['1_baseline_median'].mean()
        crisis_mean = grp['crisis'].mean()
        n_low_confidence = grp['1_low_confidence'].sum()
        
        agg_list.append({
            'quadkey': qk,
            '1_outflow_accumulated': outflow_accumulated,
            '1_outflow_accumulated_hour0': outflow_accumulated_hour0,
            '1_outflow_max': outflow_max,
            '1_outflow_max_hour0': outflow_max_hour0,
            '1_outflow_mean': outflow_mean,
            '1_outflow_p95': outflow_p95,
            '1_n_timestamps': n_timestamps,
            '1_n_valid_timestamps': n_valid_timestamps,
            '1_n_valid_timestamps_hour0': n_valid_timestamps_hour0,
            '1_n_timestamps_with_outflow': n_timestamps_with_outflow,
            '1_baseline_median_mean': baseline_median_mean,
            '1_crisis_mean': crisis_mean,
            '1_n_low_confidence': n_low_confidence
        })
    
    agg = pd.DataFrame(agg_list)
    
    # Compute additional statistics
    agg['1_outflow_fraction'] = np.where(
        agg['1_n_valid_timestamps'] > 0,
        agg['1_n_timestamps_with_outflow'] / agg['1_n_valid_timestamps'],
        np.nan
    )
    
    # Data quality metrics
    agg['1_valid_fraction'] = agg['1_n_valid_timestamps'] / agg['1_n_timestamps']
    
    print(f"Aggregated outflow for {len(agg)} quadkeys")
    print(f"  - Quadkeys with valid shortfall data: {(agg['1_n_valid_timestamps'] > 0).sum()}")
    print(f"  - Quadkeys with no valid shortfall data: {(agg['1_n_valid_timestamps'] == 0).sum()}")
    print(f"  - Total accumulated outflow (crisis week, all hours, valid values only): {agg['1_outflow_accumulated'].sum():.2f}")
    print(f"  - Total accumulated outflow (crisis week, hour=0 only, valid values only): {agg['1_outflow_accumulated_hour0'].sum():.2f}")
    print(f"  - Maximum single-timestamp outflow (all hours): {agg['1_outflow_max'].max():.2f}")
    print(f"  - Maximum single-timestamp outflow (hour=0 only): {agg['1_outflow_max_hour0'].max():.2f}")
    print(f"  - Mean valid timestamps per quadkey (all hours): {agg['1_n_valid_timestamps'].mean():.2f}")
    print(f"  - Mean valid timestamps per quadkey (hour=0 only): {agg['1_n_valid_timestamps_hour0'].mean():.2f}")
    
    return agg

# ---------- Main pipeline orchestration ----------
def run_pipeline_continuous_outflow():
    """
    Main pipeline for continuous outflow detection:
    1. Establish baseline medians per quadkey-hour from historical data
    2. Compute shortfall for all timestamps (continuous, not binary)
    3. Accumulate outflow across flood event
    """
    print("=" * 80)
    print("Continuous Outflow Detection Pipeline")
    print("=" * 80)
    
    # Load data
    print("\nStep 0: Loading fb_wide...")
    fb_wide = gpd.read_file(FB_WIDE_PATH, layer=FB_WIDE_LAYER)
    if 'quadkey' not in fb_wide.columns:
        raise ValueError("fb_wide must contain a 'quadkey' column")
    print(f"Loaded {len(fb_wide)} grid cells")
    
    # Step 1: Establish baseline medians per quadkey-hour (BASELINE WEEK ONLY)
    print("\n" + "=" * 80)
    print("Step 1: Establishing baseline medians per quadkey-hour (BASELINE WEEK ONLY)")
    print("=" * 80)
    baseline_medians_df = compute_baseline_medians_per_cell_hour(
        fb_wide=fb_wide,
        min_baseline_samples=MIN_BASELINE_SAMPLES,
        crisis_week_prefixes=CRISIS_WEEK_PREFIXES,
        baseline_suffix=BASELINE_SUFFIX,
        crisis_suffix=CRISIS_SUFFIX
    )
    
    # Save baseline medians
    baseline_medians_df.to_csv(OUT_DIR / '1_baseline_medians_per_cell_hour.csv', index=False)
    print(f"Saved baseline medians: 1_baseline_medians_per_cell_hour.csv")
    
    # Step 2 & 3: Compute shortfall for all timestamps
    print("\n" + "=" * 80)
    print("Step 2 & 3: Computing shortfall for all timestamps (continuous outflow)")
    print("Shortfall = max(0, baseline_median - observed)")
    print("=" * 80)
    shortfall_df = compute_shortfall_all_timestamps(
        fb_wide=fb_wide,
        baseline_medians_df=baseline_medians_df,
        crisis_week_prefixes=CRISIS_WEEK_PREFIXES,
        baseline_suffix=BASELINE_SUFFIX,
        crisis_suffix=CRISIS_SUFFIX,
        enforce_confidence=False
    )
    
    # Save all rows with shortfall
    shortfall_df.to_csv(OUT_DIR / '1_rows_with_outflow.csv', index=False)
    print(f"Saved shortfall data: 1_rows_with_outflow.csv")
    
    # Step 4: Accumulate outflow across flood event
    print("\n" + "=" * 80)
    print("Step 4: Accumulating outflow across flood event")
    print("=" * 80)
    agg_df = accumulate_outflow_per_quadkey(shortfall_df)
    
    # Add geometry from fb_wide if available
    if 'geometry' in fb_wide.columns:
        geom_map = fb_wide[['quadkey','geometry']].drop_duplicates(subset='quadkey').set_index('quadkey')['geometry']
        agg_df['geometry'] = agg_df['quadkey'].map(geom_map)
        agg_gdf = gpd.GeoDataFrame(agg_df, geometry='geometry', crs=fb_wide.crs)
        gpkg_path = OUT_DIR / '1_agg_outflow_accumulated.gpkg'
        agg_gdf.to_file(gpkg_path, layer='1_outflow_accumulated', driver='GPKG')
        print(f"Saved GeoPackage: {gpkg_path}")
    else:
        agg_df.to_csv(OUT_DIR / '1_agg_outflow_accumulated.csv', index=False)
        print(f"Saved CSV: 1_agg_outflow_accumulated.csv")
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("Summary Statistics")
    print("=" * 80)
    print(f"Total quadkeys: {len(agg_df)}")
    print(f"Quadkeys with valid shortfall data: {(agg_df['1_n_valid_timestamps'] > 0).sum()}")
    print(f"Quadkeys with outflow > 0 (all hours): {(agg_df['1_outflow_accumulated'] > 0).sum()}")
    print(f"Quadkeys with outflow > 0 (hour=0 only): {(agg_df['1_outflow_accumulated_hour0'] > 0).sum()}")
    print(f"Total accumulated outflow (all hours, valid values only): {agg_df['1_outflow_accumulated'].sum():.2f}")
    print(f"Total accumulated outflow (hour=0 only, valid values only): {agg_df['1_outflow_accumulated_hour0'].sum():.2f}")
    print(f"Maximum single-timestamp outflow (all hours): {agg_df['1_outflow_max'].max():.2f}")
    print(f"Maximum single-timestamp outflow (hour=0 only): {agg_df['1_outflow_max_hour0'].max():.2f}")
    print(f"95th percentile outflow: {agg_df['1_outflow_p95'].quantile(0.95):.2f}")
    print(f"Mean accumulated outflow (where > 0, all hours): {agg_df[agg_df['1_outflow_accumulated'] > 0]['1_outflow_accumulated'].mean():.2f}")
    print(f"Mean accumulated outflow (where > 0, hour=0 only): {agg_df[agg_df['1_outflow_accumulated_hour0'] > 0]['1_outflow_accumulated_hour0'].mean():.2f}")
    print(f"Mean valid timestamps per quadkey (all hours): {agg_df['1_n_valid_timestamps'].mean():.2f}")
    print(f"Mean valid timestamps per quadkey (hour=0 only): {agg_df['1_n_valid_timestamps_hour0'].mean():.2f}")
    print(f"Quadkeys with low-confidence baselines: {(agg_df['1_n_low_confidence'] > 0).sum()}")
    
    print("\n" + "=" * 80)
    print("Pipeline finished. Outputs in:", OUT_DIR)
    print("=" * 80)

if __name__ == '__main__':
    run_pipeline_continuous_outflow()
