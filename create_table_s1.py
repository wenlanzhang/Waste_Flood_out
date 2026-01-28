#!/usr/bin/env python3
"""
create_table_s1.py

Creates Table S1: Alternative displacement measures (Outcome robustness)
For each alternative dependent variable, runs Flood+Waste OLS model and extracts:
- Waste coefficient sign and significance
- Flood coefficient sign and significance
- Adj. R²
- Moran's I (OLS residuals)

Requirements:
  geopandas, pandas, numpy, statsmodels, libpysal, esda
"""
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import statsmodels.api as sm
from scipy import stats

# spatial libraries
from libpysal import weights
from esda.moran import Moran

# -------------------- USER PARAMETERS --------------------
GPKG_PATH = Path("/Users/wenlanzhang/PycharmProjects/Waste_Flood_out/Data/4/4_flood_waste_metrics_quadkey.gpkg")
LAYER_NAME = "4_flood_waste_metrics_quadkey"

OUT_DIR = Path("/Users/wenlanzhang/PycharmProjects/Waste_Flood_out/Output/summary")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Fixed independent variables (same for all models)
FLOOD_VAR = '4_flood_p95'
WASTE_VAR = '4_waste_count'

# Alternative dependent variables to test
# Format: (variable_name, display_name)
ALTERNATIVE_DVARS = [
    ('2_outflow_max', '2_outflow_max (FB users)'),
    ('1_outflow_accumulated_hour0', '1_outflow_accumulated_hour0'),
    ('1_outflow_max_hour0', '1_outflow_max_hour0'),
    # Percentage versions (computed in script 3)
    ('3_pct_outflow_fb_from_2_outflow_max', '% outflow (FB)'),
    ('3_pct_outflow_worldpop_from_2_outflow_max', '% outflow (WorldPop)'),
    # Also check for other percentage versions
    ('3_pct_outflow_fb_from_1_outflow_accumulated_hour0', '% outflow (FB) from accumulated'),
    ('3_pct_outflow_worldpop_from_1_outflow_accumulated_hour0', '% outflow (WorldPop) from accumulated'),
    ('3_pct_outflow_fb_from_1_outflow_max_hour0', '% outflow (FB) from max'),
    ('3_pct_outflow_worldpop_from_1_outflow_max_hour0', '% outflow (WorldPop) from max'),
]

MIN_OBS = 30
N_PERM = 999

# -------------------- Helper functions --------------------
def load_gpkg(path: Path, layer: str):
    """Load GeoPackage."""
    if not path.exists():
        raise FileNotFoundError(f"GPKG not found: {path}")
    gdf = gpd.read_file(path, layer=layer)
    return gdf

def replace_nodata_with_nan(df, numeric_cols, nodata=-9999.0):
    """Replace NODATA values with NaN."""
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
            df.loc[df[c] == nodata, c] = np.nan
    return df

def safe_moran(series, w_obj, permutations=N_PERM):
    """Calculate Moran's I safely."""
    try:
        mi = Moran(series.values, w_obj, permutations=permutations)
        return mi
    except Exception as e:
        print(f"Moran's I error: {e}")
        return None

def get_significance_stars(pval):
    """Get significance stars for p-value."""
    if pd.isna(pval):
        return ""
    if pval < 0.001:
        return "***"
    elif pval < 0.01:
        return "**"
    elif pval < 0.05:
        return "*"
    return ""

# def get_coef_sign(coef):
#     """Get sign of coefficient."""
#     if pd.isna(coef):
#         return ""
#     return "+" if coef > 0 else "-"

# def format_coef_sign_sig(coef, pval):
#     """Format coefficient sign with significance stars."""
#     sign = get_coef_sign(coef)
#     stars = get_significance_stars(pval)
#     if sign:
#         return f"{sign}{stars}"
#     return ""

def format_coef_sign_sig(coef, pval):
    """
    Paper-aligned logic:
    - show + / - ONLY if statistically significant
    - otherwise show 0
    """
    if pd.isna(coef) or pd.isna(pval):
        return ""
    if pval < 0.05:
        sign = "+" if coef > 0 else "-"
        return f"{sign}{get_significance_stars(pval)}"
    return "0"

# -------------------- STEP 1: Load data --------------------
print("="*80)
print("STEP 1: Loading data...")
print("="*80)

gdf = load_gpkg(GPKG_PATH, LAYER_NAME)
print(f"Loaded {len(gdf):,} rows")

# Clean data
numeric_cols = gdf.select_dtypes(include=[np.number]).columns.tolist()
gdf = replace_nodata_with_nan(gdf, numeric_cols)

# Check available columns
print("\nChecking for alternative dependent variables...")
all_disp_vars = [col for col in gdf.columns if any(x in col.lower() for x in 
    ['outflow', 'displacement', 'displace', 'excess'])]
print(f"Found {len(all_disp_vars)} potential displacement variables")
print("Sample:", all_disp_vars[:10])

# -------------------- STEP 2: Find all alternative dependent variables --------------------
print("\n" + "="*80)
print("STEP 2: Identifying alternative dependent variables...")
print("="*80)

# Check which variables exist
available_dvars = []
for var_name, display_name in ALTERNATIVE_DVARS:
    if var_name in gdf.columns:
        available_dvars.append((var_name, display_name))
        print(f"✓ Found: {var_name}")
    else:
        print(f"✗ Missing: {var_name}")

# Also look for any other percentage versions that might exist
for col in gdf.columns:
    if 'outflow' in col.lower() and ('percent' in col.lower() or 'pct' in col.lower() or '%' in col.lower()):
        if col not in [v[0] for v in available_dvars]:
            # Try to determine if it's FB or WorldPop based on column name
            if 'fb' in col.lower() or 'facebook' in col.lower():
                # Extract the base metric name if possible
                if '2_outflow_max' in col:
                    display_name = "% outflow (FB)"
                elif '1_outflow_accumulated' in col:
                    display_name = "% outflow (FB) from accumulated"
                elif '1_outflow_max' in col:
                    display_name = "% outflow (FB) from max"
                else:
                    display_name = f"% outflow (FB) - {col}"
            elif 'worldpop' in col.lower():
                if '2_outflow_max' in col:
                    display_name = "% outflow (WorldPop)"
                elif '1_outflow_accumulated' in col:
                    display_name = "% outflow (WorldPop) from accumulated"
                elif '1_outflow_max' in col:
                    display_name = "% outflow (WorldPop) from max"
                else:
                    display_name = f"% outflow (WorldPop) - {col}"
            else:
                display_name = col
            available_dvars.append((col, display_name))
            print(f"✓ Found additional percentage variable: {col} -> {display_name}")

# If we don't have percentage versions, we might need to compute them
# For now, let's proceed with what we have

print(f"\nTotal alternative dependent variables to test: {len(available_dvars)}")

# Verify independent variables exist
if FLOOD_VAR not in gdf.columns:
    raise ValueError(f"Flood variable '{FLOOD_VAR}' not found in data.")
if WASTE_VAR not in gdf.columns:
    raise ValueError(f"Waste variable '{WASTE_VAR}' not found in data.")

print(f"\nUsing fixed independent variables:")
print(f"  Flood: {FLOOD_VAR}")
print(f"  Waste: {WASTE_VAR}")

# -------------------- STEP 3: Run models for each alternative dependent variable --------------------
print("\n" + "="*80)
print("STEP 3: Running Flood+Waste OLS models for each alternative dependent variable...")
print("="*80)

results = []

for dvar_name, display_name in available_dvars:
    print(f"\n--- Testing: {display_name} ({dvar_name}) ---")
    
    # Prepare data
    model_cols = ['quadkey', 'geometry', dvar_name, FLOOD_VAR, WASTE_VAR]
    model_gdf = gdf[model_cols].dropna().reset_index(drop=True)
    
    if len(model_gdf) < MIN_OBS:
        print(f"  ⚠️  Skipping: Only {len(model_gdf)} observations (need {MIN_OBS})")
        continue
    
    print(f"  Observations: {len(model_gdf)}")
    
    # Prepare variables
    y = model_gdf[dvar_name].astype(float)
    X_df = pd.DataFrame({
        FLOOD_VAR: model_gdf[FLOOD_VAR].astype(float),
        WASTE_VAR: model_gdf[WASTE_VAR].astype(float)
    })
    X = sm.add_constant(X_df, has_constant='add')
    
    # Fit OLS
    try:
        ols = sm.OLS(y, X).fit(cov_type='HC3')
    except Exception as e:
        print(f"  ⚠️  Error fitting OLS: {e}")
        continue
    
    # Extract coefficients
    waste_coef = ols.params[WASTE_VAR]
    waste_pval = ols.pvalues[WASTE_VAR]
    flood_coef = ols.params[FLOOD_VAR]
    flood_pval = ols.pvalues[FLOOD_VAR]
    adj_r2 = ols.rsquared_adj
    
    print(f"  Waste coef: {waste_coef:.4f} (p={waste_pval:.4f})")
    print(f"  Flood coef: {flood_coef:.4f} (p={flood_pval:.4f})")
    print(f"  Adj. R²: {adj_r2:.4f}")
    
    # Calculate Moran's I on residuals
    model_gdf['residuals'] = ols.resid
    
    # Build spatial weights
    if model_gdf.geometry.is_valid.all() is False:
        model_gdf.geometry = model_gdf.geometry.buffer(0)
    
    try:
        w = weights.contiguity.Queen.from_dataframe(model_gdf)
        w.transform = "r"
        
        mi = safe_moran(model_gdf['residuals'], w, permutations=N_PERM)
        
        if mi is not None:
            moran_i = mi.I
            moran_p = mi.p_sim
            moran_stars = get_significance_stars(moran_p)
            print(f"  Moran's I: {moran_i:.4f}{moran_stars} (p={moran_p:.4g})")
        else:
            moran_i = None
            moran_p = None
            moran_stars = ""
            print(f"  ⚠️  Could not calculate Moran's I")
    except Exception as e:
        print(f"  ⚠️  Error calculating Moran's I: {e}")
        moran_i = None
        moran_p = None
        moran_stars = ""
    
    # Store results
    results.append({
        'Dependent variable': display_name,
        'Waste coef (OLS)': format_coef_sign_sig(waste_coef, waste_pval),
        'Flood coef (OLS)': format_coef_sign_sig(flood_coef, flood_pval),
        'Adj. R²': f"{adj_r2:.4f}" if not pd.isna(adj_r2) else "",
        "Moran's I (OLS resid.)": f"{moran_i:.4f}{moran_stars}" if moran_i is not None else "",
        # Store raw values for sorting/filtering
        '_waste_coef': waste_coef,
        '_flood_coef': flood_coef,
        '_adj_r2': adj_r2,
        '_moran_i': moran_i,
        '_moran_p': moran_p,
        '_nobs': len(model_gdf)
    })

if not results:
    raise RuntimeError("No models successfully fit. Check data availability.")

# -------------------- STEP 4: Create Table S1 --------------------
print("\n" + "="*80)
print("STEP 4: Creating Table S1...")
print("="*80)

# Create DataFrame
table_s1 = pd.DataFrame(results)

# Select and reorder columns for final table
table_s1_final = table_s1[[
    'Dependent variable',
    'Waste coef (OLS)',
    'Flood coef (OLS)',
    'Adj. R²',
    "Moran's I (OLS resid.)"
]].copy()

# Sort by dependent variable name (or you could sort by adj R²)
table_s1_final = table_s1_final.sort_values('Dependent variable').reset_index(drop=True)

# Save table
output_file = OUT_DIR / "S1_table_alternative_displacement.csv"
table_s1_final.to_csv(output_file, index=False)

print(f"\nTable S1 saved to: {output_file}")
print("\n" + "="*80)
print("TABLE S1: Alternative displacement measures")
print("="*80)
print(table_s1_final.to_string(index=False))
print("="*80)

# # Also save detailed results with raw values
# detailed_file = OUT_DIR / "S1_table_alternative_displacement_detailed.csv"
# table_s1.to_csv(detailed_file, index=False)
# print(f"\nDetailed results (with raw values) saved to: {detailed_file}")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Successfully tested {len(results)} alternative dependent variables")
print(f"Table S1 saved to: {output_file}")
print("="*80)
