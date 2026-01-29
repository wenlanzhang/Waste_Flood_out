#!/usr/bin/env python3
"""
create_table_s2.py

Creates Table S2: Alternative waste normalizations (Proxy robustness)

For each alternative waste measure, runs Flood + Waste OLS model and extracts:
- Waste coefficient sign and significance
- Flood coefficient sign and significance
- Adjusted R²
- Moran's I (OLS residuals, with p-value)
- Sample size (N)

This table is intended for Supplementary Results (npj Urban Sustainability).
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import statsmodels.api as sm

# spatial libraries
from libpysal import weights
from esda.moran import Moran

# -------------------- USER PARAMETERS --------------------
GPKG_PATH = Path("/Users/wenlanzhang/PycharmProjects/Waste_Flood_out/Data/4/4_flood_waste_metrics_quadkey.gpkg")
LAYER_NAME = "4_flood_waste_metrics_quadkey"

OUT_DIR = Path("/Users/wenlanzhang/PycharmProjects/Waste_Flood_out/Output/summary")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Fixed variables
DVAR = "3_estimated_outflow_pop_from_2_outflow_max"
FLOOD_VAR = "4_flood_p95"

# Alternative waste measures to test (ordered)
ALTERNATIVE_WASTE_VARS = [
    ("4_waste_count", "Raw count"),
    ("4_waste_per_population", "Per population"),
    ("4_waste_per_svi_count", "Per SVI coverage"),
]

MIN_OBS = 30
N_PERM = 999
NODATA = -9999.0

# -------------------- Helper functions --------------------
def load_gpkg(path: Path, layer: str):
    if not path.exists():
        raise FileNotFoundError(f"GPKG not found: {path}")
    return gpd.read_file(path, layer=layer)

def replace_nodata_with_nan(df, nodata=NODATA):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df.loc[df[c] == nodata, c] = np.nan
    return df

def safe_moran(series, w_obj, permutations=N_PERM):
    try:
        return Moran(series.values, w_obj, permutations=permutations)
    except Exception:
        return None

def significance_stars(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return ""

def format_coef_significance(coef, pval):
    """
    Paper-aligned logic:
    - show + / - ONLY if statistically significant
    - otherwise show 0
    """
    if pd.isna(coef) or pd.isna(pval):
        return ""
    if pval < 0.05:
        sign = "+" if coef > 0 else "-"
        return f"{sign}{significance_stars(pval)}"
    return "0"

# -------------------- Load & clean data --------------------
print("=" * 80)
print("Loading data...")
print("=" * 80)

gdf = load_gpkg(GPKG_PATH, LAYER_NAME)
gdf = replace_nodata_with_nan(gdf)

if DVAR not in gdf.columns or FLOOD_VAR not in gdf.columns:
    raise ValueError("Dependent or flood variable missing from dataset.")

results = []

# -------------------- Run robustness models --------------------
print("\nRunning Flood + Waste OLS models for alternative waste measures...")

for waste_var, display_name in ALTERNATIVE_WASTE_VARS:

    if waste_var not in gdf.columns:
        print(f"Skipping missing waste variable: {waste_var}")
        continue

    print(f"\nTesting waste measure: {display_name}")

    model_gdf = gdf[
        ["quadkey", "geometry", DVAR, FLOOD_VAR, waste_var]
    ].dropna().reset_index(drop=True)

    nobs = len(model_gdf)
    if nobs < MIN_OBS:
        print(f"  Skipped (N={nobs})")
        continue

    y = model_gdf[DVAR].astype(float)
    X_df = pd.DataFrame({
        FLOOD_VAR: model_gdf[FLOOD_VAR].astype(float),
        waste_var: model_gdf[waste_var].astype(float),
    })
    X = sm.add_constant(X_df, has_constant="add")

    ols = sm.OLS(y, X).fit(cov_type="HC3")

    waste_coef = ols.params[waste_var]
    waste_p = ols.pvalues[waste_var]

    flood_coef = ols.params[FLOOD_VAR]
    flood_p = ols.pvalues[FLOOD_VAR]

    adj_r2 = ols.rsquared_adj

    # Moran's I on residuals
    model_gdf["residuals"] = ols.resid

    if not model_gdf.geometry.is_valid.all():
        model_gdf.geometry = model_gdf.geometry.buffer(0)

    w = weights.contiguity.Queen.from_dataframe(model_gdf)
    w.transform = "r"

    mi = safe_moran(model_gdf["residuals"], w)

    if mi is not None:
        moran_i = mi.I
        moran_p = mi.p_sim
        moran_str = f"{moran_i:.3f} (p={moran_p:.3f})"
    else:
        moran_str = ""

    results.append({
        "Waste measure": display_name,
        "Waste coef (OLS)": format_coef_significance(waste_coef, waste_p),
        "Flood coef (OLS)": format_coef_significance(flood_coef, flood_p),
        "Adj. R²": f"{adj_r2:.3f}",
        "Moran's I (OLS resid.)": moran_str,
        "N": nobs,
        # raw values retained for traceability
        "_waste_coef": waste_coef,
        "_waste_p": waste_p,
        "_adj_r2": adj_r2,
        "_moran_i": moran_i if mi is not None else np.nan,
        "_moran_p": moran_p if mi is not None else np.nan,
    })

# -------------------- Create final Table S2 --------------------
print("\nCreating Table S2...")

table = pd.DataFrame(results)

# enforce preferred order
order = {name: i for i, (_, name) in enumerate(ALTERNATIVE_WASTE_VARS)}
table["_order"] = table["Waste measure"].map(order)
table = table.sort_values("_order").drop(columns="_order")

table_final = table[
    [
        "Waste measure",
        "Waste coef (OLS)",
        "Flood coef (OLS)",
        "Adj. R²",
        "Moran's I (OLS resid.)",
        "N",
    ]
]

# save outputs
out_main = OUT_DIR / "Table_S2_alternative_waste_normalizations.csv"
# out_full = OUT_DIR / "Table_S2_alternative_waste_normalizations_detailed.csv"

table_final.to_csv(out_main, index=False)
# table.to_csv(out_full, index=False)

print("\n" + "=" * 80)
print("TABLE S2 | Alternative waste normalizations")
print("=" * 80)
print(table_final.to_string(index=False))
print("=" * 80)

print(f"\nSaved final table to: {out_main}")
# print(f"Saved detailed table to: {out_full}")
print("\nDone.")
