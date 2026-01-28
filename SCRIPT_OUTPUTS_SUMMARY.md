# Final Dataset: Column Documentation

This document describes all columns in the **final analysis dataset**: `Data/4/4_flood_waste_metrics_quadkey.gpkg`

**Layer Name:** `4_flood_waste_metrics_quadkey`

**Structure:** One row per quadkey (grid cell) polygon with all metrics aggregated/computed at the quadkey level.

---

## Dataset Overview

The final dataset combines:
- **Displacement metrics** from Steps 1-3 (Facebook population change analysis)
- **Flood metrics** from Step 4 (zonal statistics on flood raster)
- **Waste metrics** from Step 4 (spatial aggregation of waste points)
- **Population metrics** from Step 3 (WorldPop scaling)

---

## Column Categories

### 1. **Identifier Columns**

#### `quadkey`
- **Type:** String
- **Description:** Unique identifier for each grid cell (quadkey)
- **Source:** Facebook population data grid system
- **Usage:** Primary key for joining and spatial operations

#### `geometry`
- **Type:** Polygon (GeoPandas geometry)
- **Description:** Spatial polygon geometry for each quadkey
- **CRS:** EPSG:32737 (UTM Zone 37S) or as specified
- **Usage:** For spatial operations, mapping, and visualization

---

### 2. **Displacement Metrics (Step 1: Continuous Outflow Detection)**

These columns come from Step 1 (`1y_compute_accumulated_outflow.py`), which computes continuous shortfall (population deficit) during flood events.

#### `1_outflow_accumulated`
- **Type:** Float
- **Description:** **Total accumulated shortfall** across all crisis week timestamps (all hours)
- **Calculation:** Sum of `shortfall = max(0, baseline_median - observed_crisis)` across all crisis week timestamps
- **Units:** Facebook population units (people)
- **Interpretation:** Total population deficit experienced during the entire flood event
- **Note:** Only accumulates from crisis week, excludes baseline week

#### `1_outflow_accumulated_hour0`
- **Type:** Float
- **Description:** **Total accumulated shortfall** at hour=0 only (midnight)
- **Calculation:** Sum of shortfall values at hour=0 across crisis week timestamps
- **Units:** Facebook population units (people)
- **Usage:** Used for WorldPop scaling (hour=0 is reference hour)
- **Interpretation:** Population deficit at midnight during crisis week

#### `1_outflow_max`
- **Type:** Float
- **Description:** **Maximum shortfall** at any single timestamp (all hours)
- **Calculation:** Maximum value of shortfall across all crisis week timestamps
- **Units:** Facebook population units (people)
- **Interpretation:** Peak single-timestamp population deficit

#### `1_outflow_max_hour0`
- **Type:** Float
- **Description:** **Maximum shortfall** at hour=0 only
- **Calculation:** Maximum shortfall value at hour=0 across crisis week timestamps
- **Units:** Facebook population units (people)

#### `1_outflow_mean`
- **Type:** Float
- **Description:** Mean shortfall across all crisis week timestamps
- **Calculation:** Average of all shortfall values (excluding NaN)
- **Units:** Facebook population units (people)

#### `1_outflow_p95`
- **Type:** Float
- **Description:** 95th percentile shortfall value
- **Calculation:** 95th percentile of shortfall values across crisis week
- **Units:** Facebook population units (people)

#### `1_n_timestamps`
- **Type:** Integer
- **Description:** Total number of timestamps in crisis week for this quadkey
- **Usage:** Data quality indicator

#### `1_n_valid_timestamps`
- **Type:** Integer
- **Description:** Count of valid (non-NaN) shortfall values (all hours)
- **Usage:** Data quality indicator

#### `1_n_valid_timestamps_hour0`
- **Type:** Integer
- **Description:** Count of valid shortfall values at hour=0
- **Usage:** Data quality indicator

#### `1_n_timestamps_with_outflow`
- **Type:** Integer
- **Description:** Count of timestamps with shortfall > 0
- **Usage:** Indicates how many timestamps showed population deficit

#### `1_outflow_fraction`
- **Type:** Float
- **Description:** Fraction of valid timestamps with outflow > 0
- **Calculation:** `1_n_timestamps_with_outflow / 1_n_valid_timestamps`
- **Range:** 0.0 to 1.0

#### `1_valid_fraction`
- **Type:** Float
- **Description:** Fraction of timestamps with valid data
- **Calculation:** `1_n_valid_timestamps / 1_n_timestamps`
- **Range:** 0.0 to 1.0

#### `1_baseline_median_mean`
- **Type:** Float
- **Description:** Mean baseline median across crisis week timestamps
- **Usage:** Diagnostic metric

#### `1_crisis_mean`
- **Type:** Float
- **Description:** Mean crisis population across crisis week timestamps
- **Usage:** Diagnostic metric

#### `1_n_low_confidence`
- **Type:** Integer
- **Description:** Count of timestamps with low-confidence baselines (< MIN_BASELINE_SAMPLES)
- **Usage:** Data quality indicator

---

### 3. **Displacement Metrics (Step 2: CSAT Anomaly Detection)**

These columns come from Step 2 (`2y_detect_csat_anomalies.py`), which uses Modified Z-Score method to detect statistical anomalies.

#### `2_outflow_max`
- **Type:** Float
- **Description:** **Maximum outflow displacement** across all timestamps at reference hour (hour=0)
- **Calculation:** Maximum of displacement values at reference hour
- **Units:** Facebook population units (people)
- **Method:** CSAT (Modified Z-Score) anomaly detection
- **Usage:** Primary displacement metric for scaling to WorldPop

#### `2_displaced_excess_max`
- **Type:** Float
- **Description:** **Maximum excess displacement** exceeding -3.5 Z-score threshold
- **Calculation:** Maximum displacement value where modified_zscore < -3.5
- **Units:** Facebook population units (people)
- **Interpretation:** Extreme displacement events (statistical outliers)

#### `2_n_crisis_rows`
- **Type:** Integer
- **Description:** Number of crisis week rows at reference hour
- **Usage:** Data quality indicator

#### `2_n_obs`
- **Type:** Integer
- **Description:** Total observations at reference hour
- **Usage:** Data quality indicator

#### `2_n_low_confidence`
- **Type:** Integer
- **Description:** Count of low-confidence thresholds (< MIN_BASELINE_SAMPLES)
- **Usage:** Data quality indicator

#### `2_change_mean`, `2_change_std`, `2_change_min`, `2_change_max`
- **Type:** Float
- **Description:** Statistics of change metric (modified_zscore, relative_percent, absolute_diff, or log_change)
- **Usage:** Diagnostic metrics

#### `2_change_range`
- **Type:** Float
- **Description:** Range of change values (`2_change_max - 2_change_min`)
- **Usage:** Diagnostic metric

#### `2_change_iqr`
- **Type:** Float
- **Description:** Interquartile range (75th percentile - 25th percentile) of change values
- **Usage:** Robust measure of spread

#### `2_change_cv`
- **Type:** Float
- **Description:** Coefficient of variation (std/mean) of change values
- **Calculation:** `2_change_std / abs(2_change_mean)`
- **Usage:** Normalized measure of variability

---

### 4. **Population Scaling Metrics (Step 3: WorldPop Scaling)**

These columns come from Step 3 (`3scale_to_worldpop.py`), which scales Facebook displacement metrics to WorldPop population estimates.

#### `3_worldpop`
- **Type:** Float
- **Description:** **Zonal sum of WorldPop raster** per quadkey
- **Calculation:** Sum of WorldPop pixel values within quadkey polygon
- **Units:** People (WorldPop population estimate)
- **Source:** WorldPop raster data
- **Usage:** Population denominator for waste metrics and scaling reference

#### `3_fb_baseline_median`
- **Type:** Float
- **Description:** Median Facebook baseline population per quadkey (from reference hour rows)
- **Calculation:** Median of baseline week population values at reference hour
- **Units:** Facebook population units (people)
- **Usage:** Denominator for computing scaling ratio

#### `3_scaling_ratio`
- **Type:** Float
- **Description:** **WorldPop / FB baseline ratio** for scaling displacement metrics
- **Calculation:** `3_worldpop / 3_fb_baseline_median` (with safety caps and fallbacks)
- **Logic:**
  - If `fb_baseline_median < MIN_FB_BASELINE_FOR_UNCAPPED` (50) AND `per_cell_ratio > global_ratio`: use global fallback ratio
  - Otherwise: use per-cell ratio (capped at MAX_RATIO=50.0 for low FB coverage)
- **Usage:** Multiplier to convert FB displacement to WorldPop population units

#### `3_used_global_ratio`
- **Type:** Integer (0 or 1)
- **Description:** Flag indicating cells that used global fallback ratio
- **Values:** 1 = used global fallback, 0 = used per-cell ratio
- **Usage:** Data quality indicator

#### `3_estimated_outflow_pop_from_2_outflow_max`
- **Type:** Float
- **Description:** **Scaled outflow in population units** from CSAT method (Step 2)
- **Calculation:** `2_outflow_max × 3_scaling_ratio`
- **Units:** People (WorldPop-scaled)
- **Usage:** Primary displacement metric for modeling (most commonly used)

#### `3_estimated_outflow_pop_from_1_outflow_accumulated_hour0`
- **Type:** Float
- **Description:** **Scaled accumulated outflow** from continuous method (Step 1, hour=0)
- **Calculation:** `1_outflow_accumulated_hour0 × 3_scaling_ratio`
- **Units:** People (WorldPop-scaled)
- **Usage:** Alternative displacement metric for modeling

#### `3_estimated_outflow_pop_from_1_outflow_max_hour0`
- **Type:** Float
- **Description:** **Scaled maximum outflow** from continuous method (Step 1, hour=0)
- **Calculation:** `1_outflow_max_hour0 × 3_scaling_ratio`
- **Units:** People (WorldPop-scaled)
- **Usage:** Alternative displacement metric for modeling

#### `3_estimated_excess_displacement_pop`
- **Type:** Float
- **Description:** **Scaled excess displacement** (extreme events exceeding -3.5 Z-score)
- **Calculation:** `2_displaced_excess_max × 3_scaling_ratio`
- **Units:** People (WorldPop-scaled)
- **Usage:** Extreme displacement metric

#### `3_pct_outflow_fb_from_2_outflow_max`
- **Type:** Float
- **Description:** Facebook displacement as percentage of FB baseline
- **Calculation:** `(2_outflow_max / 3_fb_baseline_median) × 100`
- **Units:** Percentage
- **Usage:** Relative displacement metric (FB scale)

#### `3_pct_outflow_fb_from_1_outflow_accumulated_hour0`
- **Type:** Float
- **Description:** Facebook accumulated outflow as percentage of FB baseline
- **Calculation:** `(1_outflow_accumulated_hour0 / 3_fb_baseline_median) × 100`
- **Units:** Percentage

#### `3_pct_outflow_fb_from_1_outflow_max_hour0`
- **Type:** Float
- **Description:** Facebook maximum outflow as percentage of FB baseline
- **Calculation:** `(1_outflow_max_hour0 / 3_fb_baseline_median) × 100`
- **Units:** Percentage

#### `3_pct_outflow_worldpop_from_2_outflow_max`
- **Type:** Float
- **Description:** WorldPop displacement as percentage of WorldPop population
- **Calculation:** `(3_estimated_outflow_pop_from_2_outflow_max / 3_worldpop) × 100`
- **Units:** Percentage
- **Usage:** Relative displacement metric (WorldPop scale)

#### `3_pct_outflow_worldpop_from_1_outflow_accumulated_hour0`
- **Type:** Float
- **Description:** WorldPop accumulated outflow as percentage of WorldPop population
- **Calculation:** `(3_estimated_outflow_pop_from_1_outflow_accumulated_hour0 / 3_worldpop) × 100`
- **Units:** Percentage

#### `3_pct_outflow_worldpop_from_1_outflow_max_hour0`
- **Type:** Float
- **Description:** WorldPop maximum outflow as percentage of WorldPop population
- **Calculation:** `(3_estimated_outflow_pop_from_1_outflow_max_hour0 / 3_worldpop) × 100`
- **Units:** Percentage

#### `3_ratio_was_capped`
- **Type:** Integer (deprecated)
- **Description:** Legacy flag (kept for compatibility, always 0)
- **Note:** Deprecated - use `3_used_global_ratio` instead

---

### 5. **Flood Metrics (Step 4: Zonal Statistics)**

These columns come from Step 4 (`4flood_waste.py`), computed using zonal statistics on the flood raster.

#### `4_flood_p95`
- **Type:** Float
- **Description:** **95th percentile flood value** per quadkey
- **Calculation:** 95th percentile of flood raster pixel values within quadkey polygon
- **Method:** Zonal statistics (rasterstats)
- **Units:** Flood depth/exposure units (as per flood raster)
- **Usage:** Primary flood exposure metric (captures extreme flood values)
- **Interpretation:** High values indicate areas with severe flooding

#### `4_flood_mean`
- **Type:** Float
- **Description:** **Mean flood value** per quadkey (average flood exposure)
- **Calculation:** Mean of flood raster pixel values within quadkey polygon
- **Method:** Zonal statistics (rasterstats)
- **Units:** Flood depth/exposure units (as per flood raster)
- **Usage:** Alternative flood metric (captures average flood exposure)
- **Interpretation:** Average flood level across the quadkey

#### `4_flood_max`
- **Type:** Float
- **Description:** **Maximum flood value** per quadkey (highest flood exposure)
- **Calculation:** Maximum of flood raster pixel values within quadkey polygon
- **Method:** Zonal statistics (rasterstats)
- **Units:** Flood depth/exposure units (as per flood raster)
- **Usage:** Alternative flood metric (captures peak flood exposure)
- **Interpretation:** Highest flood level within the quadkey

---

### 6. **Waste Metrics (Step 4: Spatial Aggregation)**

These columns come from Step 4 (`4flood_waste.py`), computed by spatially aggregating waste point data and SVI image points to quadkey polygons.

#### `4_waste_count`
- **Type:** Integer
- **Description:** **Raw count of waste points** per quadkey
- **Calculation:** Spatial join (points within polygon) to count waste points
- **Source:** Waste point GeoPackage (`aoi_waste.gpkg`)
- **Units:** Count (number of waste points)
- **Usage:** Primary waste metric
- **Note:** Set to 0 where no waste points found (not nodata)

#### `4_waste_count_final`
- **Type:** Float
- **Description:** Waste count with nodata flagging where no SVI coverage exists
- **Calculation:** 
  - If `4_svi_count == 0`: set to `-9999` (nodata)
  - Otherwise: same as `4_waste_count`
- **Units:** Count (or -9999 for nodata)
- **Usage:** Data quality indicator (distinguishes "no waste" from "no SVI coverage")

#### `4_waste_per_quadkey_area`
- **Type:** Float
- **Description:** **Waste points per 1000 m²** (density metric)
- **Calculation:** `(4_waste_count / 4_quadkey_area_m2) × 1000`
- **Units:** Waste points per 1000 m²
- **Logic:**
  - If `4_svi_count == 0`: set to `-9999` (nodata)
  - If `4_quadkey_area_m2 > 0`: calculate density
  - Otherwise: set to `0.0`
- **Usage:** Normalized waste metric accounting for quadkey size
- **Interpretation:** Higher values indicate more waste per unit area

#### `4_waste_per_population`
- **Type:** Float
- **Description:** **Waste points per 1000 people** (population-normalized metric)
- **Calculation:** `(4_waste_count / 3_worldpop) × 1000`
- **Units:** Waste points per 1000 people
- **Logic:**
  - If `4_svi_count == 0`: set to `-9999` (nodata)
  - If `3_worldpop > 0`: calculate per-population rate
  - Otherwise: set to `0.0`
- **Usage:** Normalized waste metric accounting for population
- **Interpretation:** Higher values indicate more waste relative to population

#### `4_waste_per_svi_count`
- **Type:** Float
- **Description:** **Waste points per 1000 SVI images** (SVI-normalized metric)
- **Calculation:** `(4_waste_count / 4_svi_count) × 1000`
- **Units:** Waste points per 1000 SVI images
- **Logic:**
  - If `4_svi_count == 0`: set to `-9999` (nodata)
  - If `4_svi_count > 0`: calculate per-SVI rate
  - Otherwise: set to `0.0`
- **Usage:** Normalized waste metric accounting for SVI coverage
- **Interpretation:** Higher values indicate more waste relative to SVI image coverage (accounts for observation bias)

#### `4_svi_count`
- **Type:** Integer
- **Description:** **Count of SVI (Street View Imagery) images** per quadkey
- **Calculation:** Spatial join to count SVI image points within quadkey polygon
- **Source:** SVI point GeoPackage (`aoi_all_svi.gpkg`)
- **Units:** Count (number of SVI images)
- **Usage:** 
  - Denominator for `4_waste_per_svi_count`
  - Data quality indicator (no SVI = no waste observation capability)
- **Interpretation:** Higher values indicate better street view coverage

#### `4_quadkey_area_m2`
- **Type:** Float
- **Description:** **Area of quadkey polygon** in square meters
- **Calculation:** Polygon area computed in UTM CRS for accuracy
- **Units:** Square meters (m²)
- **Usage:** 
  - Denominator for `4_waste_per_quadkey_area`
  - Spatial reference metric

---

## Data Quality Notes

### Missing Data Handling

1. **Flood Metrics:**
   - Missing values (NaN) occur where flood raster has no data or quadkey is outside raster extent

2. **Waste Metrics:**
   - **Nodata (-9999):** Assigned where `4_svi_count == 0` (no SVI coverage = cannot observe waste)
   - **Zero (0):** Assigned where SVI coverage exists but no waste points found
   - **Distinction:** Important to distinguish "no waste observed" (0) from "cannot observe waste" (-9999)

3. **Displacement Metrics:**
   - Missing values (NaN) occur where:
     - Insufficient baseline data (< MIN_BASELINE_SAMPLES)
     - Missing crisis week data
     - Low-confidence baselines

### Recommended Usage

- **For modeling:** Use columns with `3_estimated_*` prefix (WorldPop-scaled) as dependent variables
- **For flood exposure:** Use `4_flood_p95` (primary) or `4_flood_mean`/`4_flood_max` (alternatives)
- **For waste exposure:** Use `4_waste_per_svi_count` (recommended, accounts for observation bias) or `4_waste_per_population` (population-normalized)
- **For population:** Use `3_worldpop` as control variable or denominator

---

## Script Outputs Summary

For details on intermediate outputs from each script, see the original documentation structure below.

---

## Original Script-by-Script Documentation

<details>
<summary>Click to expand script-by-script output documentation</summary>

[The original script-by-script documentation would go here if needed for reference]

</details>
