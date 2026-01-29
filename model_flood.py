#!/usr/bin/env python3
"""
model_floodonly.py

Flood-only model pipeline (displacement ~ flood; no waste, no population):
  1) Read geopackage with displacement and flood vars (no waste columns used)
  2) Sanity checks + cleaning
  3) Model search: displacement ~ flood (no waste, no interaction)
  4) OLS diagnostics (HC3 robust SE) and Moran's I test on residuals
  5) Fit Spatial Lag Model (SLM) if spatial autocorrelation present
  6) Compute SLM impacts (direct/indirect/total) and pseudo-R^2
  7) Save model results and outputs

COLUMN NAMING CONVENTIONS (updated):
  - Script 1 (1y_): 1_outflow_accumulated, 1_outflow_accumulated_hour0, 1_outflow_max, 1_outflow_max_hour0
  - Script 2 (2y_): 2_outflow_max, 2_displaced_excess_max
  - Script 3 (3scale_): 3_worldpop, 3_estimated_outflow_pop_from_2_outflow_max, 
                       3_estimated_outflow_pop_from_1_outflow_accumulated_hour0, etc.
  - Script 4 (4flood_): 4_flood_p95, 4_flood_mean, 4_flood_max (flood only; no waste)

Requirements:
  geopandas, pandas, numpy, statsmodels, libpysal, esda, spreg (pysal/spreg)
"""
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import statsmodels.api as sm
from numpy.linalg import inv
from scipy import stats
import matplotlib.pyplot as plt

# spatial libraries
from libpysal import weights
from esda.moran import Moran

# spreg import with fallback locations
try:
    from spreg import ML_Lag, ML_Error
except Exception:
    try:
        # older packaging variation
        from spreg import ml_lag as ML_Lag, ml_error as ML_Error
    except Exception as e:
        raise ImportError("spreg not found. Install with: conda/pip install pysal spreg. Error: " + str(e))


# -------------------- USER PARAMETERS --------------------
GPKG_PATH = Path("/Users/wenlanzhang/PycharmProjects/Waste_Flood_out/Data/4/4_flood_waste_metrics_quadkey.gpkg")
LAYER_NAME = "4_flood_waste_metrics_quadkey"

OUT_DIR = Path("/Users/wenlanzhang/PycharmProjects/Waste_Flood_out/Output/flood")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FIGURE_DIR = Path("/Users/wenlanzhang/PycharmProjects/Waste_Flood_out/Figure/flood")
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

MIN_OBS = 30                 # minimum observations to consider a model
MAX_TOP_MODELS = 10         # keep top N models by adj. R^2
N_PERM = 999                # permutations for Moran's I

# Updated to match new column naming conventions (1_, 2_, 3_, 4_ prefixes)
DISP_SUBS = ['1_outflow', '2_outflow', '3_estimated_outflow', '1_outflow_accumulated', '1_outflow_max', 
             '2_displaced_excess', '3_estimated_excess', 'displace', 'displacement', 'outflow', 
             'excess_displacement', 'estimated_outflow', 'estimated_excess']
FLOOD_SUBS = ['4_flood', '4_flood_p', '4_flood_p95', '4_flood_mean', '4_flood_max', 'flood', 'flood_p', 'flood_p95', 'flood_mean', 'flood_max', 'flood_exposure', 'flood_risk']
# No WASTE_SUBS: flood-only model uses displacement + flood only

NODATA = -9999.0

# Optional: Set specific variables to use
YCOL = '3_estimated_outflow_pop_from_2_outflow_max'  # e.g., '3_estimated_outflow_pop_from_2_outflow_max' or None to search
FLOOD_VAR = '4_flood_p95'  # e.g., '4_flood_p95' or None to search

# YCOL = None
# FLOOD_VAR = None

# -------------------- Helper functions --------------------
def find_candidates(subs, cols):
    found = [c for c in cols if any(s.lower() in c.lower() for s in subs)]
    return sorted(list(set(found)))

def load_gpkg(path: Path, layer: str):
    if not path.exists():
        raise FileNotFoundError(f"GPKG not found: {path}")
    gdf = gpd.read_file(path, layer=layer)
    return gdf

def replace_nodata_with_nan(df, numeric_cols):
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
            df.loc[df[c] == NODATA, c] = np.nan
    return df

def safe_moran(series, w_obj, permutations=N_PERM):
    try:
        mi = Moran(series.values, w_obj, permutations=permutations)
        return mi
    except Exception as e:
        print("Moran's I error:", e)
        return None

def parse_impacts_from_summary(summary_text):
    """
    Parse impacts (Direct, Indirect, Total) from spreg's SLM summary text.
    Returns a list of dictionaries with 'variable', 'direct', 'indirect', 'total' keys.
    
    The summary format is:
        SPATIAL LAG MODEL IMPACTS
        Impacts computed using the 'simple' method.
            Variable         Direct        Indirect          Total
         4_flood_p95        12.1738          8.2833         20.4571
    """
    impacts = []
    lines = summary_text.split('\n')
    
    # Find the "SPATIAL LAG MODEL IMPACTS" section
    in_impacts_section = False
    header_found = False
    
    for line in lines:
        if 'SPATIAL LAG MODEL IMPACTS' in line:
            in_impacts_section = True
            continue
        
        if in_impacts_section:
            # Skip the "Impacts computed using..." line
            if 'Impacts computed using' in line:
                continue
            
            # Look for the header row with Variable, Direct, Indirect, Total
            if 'Variable' in line and 'Direct' in line and 'Indirect' in line and 'Total' in line:
                header_found = True
                continue
            
            # Parse data rows (skip lines that are just separators or empty)
            if header_found and line.strip() and not line.strip().startswith('=') and not line.strip().startswith('END'):
                # Split by whitespace and extract values
                # The format uses fixed-width columns, but we can parse by splitting on whitespace
                parts = line.split()
                if len(parts) >= 4:
                    # Try to parse the last 3 as floats (direct, indirect, total)
                    try:
                        total = float(parts[-1])
                        indirect = float(parts[-2])
                        direct = float(parts[-3])
                        # Everything before the last 3 parts is the variable name
                        variable = ' '.join(parts[:-3]).strip()
                        
                        if variable:  # Only add if we got a valid variable name
                            impacts.append({
                                'variable': variable,
                                'direct': direct,
                                'indirect': indirect,
                                'total': total
                            })
                    except (ValueError, IndexError):
                        # Skip lines that don't parse correctly
                        continue
    
    return impacts


# -------------------- STEP 1: Read & checks --------------------
print("\nSTEP 1: Reading GeoPackage and doing basic checks...")
print(f"Loading from: {GPKG_PATH}")
print(f"Layer: {LAYER_NAME}")
gdf = load_gpkg(GPKG_PATH, LAYER_NAME)
print(f"Loaded layer '{LAYER_NAME}' with {len(gdf):,} rows. CRS: {gdf.crs}")
if 'quadkey' not in gdf.columns:
    raise ValueError("Input must contain 'quadkey' column.")

# Show available columns with prefixes (1_, 2_, 3_, 4_)
print("\nColumn prefixes found:")
prefix_cols = {prefix: [c for c in gdf.columns if c.startswith(prefix)] for prefix in ['1_', '2_', '3_', '4_']}
for prefix, cols in prefix_cols.items():
    if cols:
        print(f"  {prefix}: {len(cols)} columns (sample: {cols[:3]})")

n_missing_geom = gdf.geometry.isna().sum()
geom_types = gdf.geometry.geom_type.value_counts().to_dict()
print(f"\nGeometry types sample: {geom_types}; missing geometries: {n_missing_geom}")

# -------------------- STEP 2: Prepare modeling DataFrame --------------------
print("\n" + "="*80)
print("STEP 2: Preparing cleaned modeling table...")
print("="*80)
numeric_cols = gdf.select_dtypes(include=[np.number]).columns.tolist()
gdf = replace_nodata_with_nan(gdf, numeric_cols)

# Find candidate variables (displacement Y, flood X only; no waste)
disp_candidates = find_candidates(DISP_SUBS, numeric_cols)
flood_candidates = find_candidates(FLOOD_SUBS, numeric_cols)

if not disp_candidates:
    raise RuntimeError("No displacement candidate variables found.")
if not flood_candidates:
    raise RuntimeError("No flood candidate variables found.")

# Create modeling dataframe (flood-only: no waste columns)
model_cols = ['quadkey', 'geometry'] + sorted(set(disp_candidates + flood_candidates))
dfg = gdf[model_cols].copy()
dfg = dfg[dfg.geometry.notna()].reset_index(drop=True)
df_clean = dfg.drop(columns='geometry').copy()

missing_summary = df_clean.isna().mean().round(3)
print("\nProportion missing per candidate column (rows with valid geometry):")
print(missing_summary[missing_summary > 0].sort_values(ascending=False).to_string())
print("="*80)

# -------------------- STEP 3B: Flood-Only Model Search --------------------
print("\n" + "="*80)
print("STEP 3B: FLOOD-ONLY MODEL SEARCH")
print("="*80)

# Check if variables are manually set
if YCOL is not None and FLOOD_VAR is not None:
    print(f"Using manually specified variables:")
    print(f"  Dependent variable: {YCOL}")
    print(f"  Flood variable:     {FLOOD_VAR}")
    print("="*80)
    
    # Validate variables exist
    if YCOL not in df_clean.columns:
        raise ValueError(f"Specified YCOL '{YCOL}' not found in data.")
    if FLOOD_VAR not in df_clean.columns:
        raise ValueError(f"Specified FLOOD_VAR '{FLOOD_VAR}' not found in data.")
    
    ycol = YCOL
    flood_var = FLOOD_VAR
    
    # Create a single result entry
    subdf = df_clean[[ycol, flood_var]].dropna()
    nobs = len(subdf)
    if nobs < MIN_OBS:
        raise RuntimeError(f"Only {nobs} observations available, need at least {MIN_OBS}.")
    
    y = subdf[ycol].astype(float)
    X = pd.DataFrame({flood_var: subdf[flood_var].astype(float)}, index=subdf.index)
    X = sm.add_constant(X, has_constant='add')
    model = sm.OLS(y, X).fit(cov_type='HC3')
    
    best_flood = {
        'y': ycol,
        'flood': flood_var,
        'nobs': nobs,
        'r2': float(model.rsquared),
        'adjr2': float(model.rsquared_adj),
        'aic': float(model.aic),
        'bic': float(model.bic)
    }
    
    print(f"\n{'='*80}")
    print("SELECTED MODEL:")
    print(f"{'='*80}")
    print(f"  Dependent variable: {best_flood['y']}")
    print(f"  Flood variable:     {best_flood['flood']}")
    print(f"  Observations:       {best_flood['nobs']}")
    print(f"  R²:                 {best_flood['r2']:.6f}")
    print(f"  Adj. R²:            {best_flood['adjr2']:.6f}")
    print(f"  AIC:                {best_flood['aic']:.2f}")
    print(f"  BIC:                {best_flood['bic']:.2f}")
    print(f"{'='*80}")
    
    # Save enhanced CSV with R2, flood variable (name + coefficient + p-value)
    flood_coef = model.params.iloc[1]  # flood variable coefficient
    flood_pval = model.pvalues.iloc[1]  # flood variable p-value
    
    enhanced_result = pd.DataFrame([{
        'y': ycol,
        'flood_var': flood_var,
        'flood_coefficient': float(flood_coef),
        'flood_p_value': float(flood_pval),
        'r2': best_flood['r2'],
        'adjr2': best_flood['adjr2'],
        'nobs': nobs,
        'aic': best_flood['aic'],
        'bic': best_flood['bic']
    }])
    enhanced_result.to_csv(OUT_DIR / "flood_model_search_with_coefficients.csv", index=False)
    print(f"Enhanced model results (with coefficients and p-values) saved to: {OUT_DIR / 'flood_model_search_with_coefficients.csv'}")
    
else:
    print("Searching models: displacement ~ flood (no waste, no interaction)")
    print("="*80)
    
    results_flood_only = []
    for ycol_search in disp_candidates:
        for fcol_search in flood_candidates:
            subdf = df_clean[[ycol_search, fcol_search]].dropna()
            nobs = len(subdf)
            if nobs < MIN_OBS:
                continue
            y = subdf[ycol_search].astype(float)
            X = pd.DataFrame({fcol_search: subdf[fcol_search].astype(float)}, index=subdf.index)
            X = sm.add_constant(X, has_constant='add')
            try:
                model = sm.OLS(y, X).fit(cov_type='HC3')
            except Exception:
                continue
            results_flood_only.append({
                'y': ycol_search, 
                'flood': fcol_search, 
                'nobs': nobs,
                'r2': float(model.rsquared), 
                'adjr2': float(model.rsquared_adj),
                'aic': float(model.aic), 
                'bic': float(model.bic), 
                'model_obj': model
            })
    
    if not results_flood_only:
        print("⚠️  WARNING: No flood-only models fit. Check data availability.")
        raise RuntimeError("No flood-only models fit. Check data availability.")
    
    res_flood_df = pd.DataFrame([{k:v for k,v in r.items() if k != 'model_obj'} for r in results_flood_only])
    res_flood_df = res_flood_df.sort_values(['adjr2', 'r2'], ascending=[False, False]).reset_index(drop=True)
    n_show = min(MAX_TOP_MODELS, len(res_flood_df))
    print(f"\nFound {len(res_flood_df):,} flood-only candidate models; showing top {n_show}:")
    print(res_flood_df.head(n_show).to_string(index=False))
    
    # Create enhanced CSV with R2, flood variable (name + coefficient + p-value) — single search output
    enhanced_results = []
    for r in results_flood_only:
        model = r['model_obj']
        # Extract coefficients and p-values
        # Model has: const, flood_var (in that order)
        flood_coef = model.params.iloc[1]  # flood variable coefficient
        flood_pval = model.pvalues.iloc[1]  # flood variable p-value
        
        enhanced_results.append({
            'y': r['y'],
            'flood_var': r['flood'],
            'flood_coefficient': float(flood_coef),
            'flood_p_value': float(flood_pval),
            'r2': r['r2'],
            'adjr2': r['adjr2'],
            'nobs': r['nobs'],
            'aic': r['aic'],
            'bic': r['bic']
        })
    
    enhanced_df = pd.DataFrame(enhanced_results)
    enhanced_df = enhanced_df.sort_values(['adjr2', 'r2'], ascending=[False, False]).reset_index(drop=True)
    enhanced_df.to_csv(OUT_DIR / "flood_model_search_with_coefficients.csv", index=False)
    print(f"\nModel search results (with coefficients and p-values) saved to: {OUT_DIR / 'flood_model_search_with_coefficients.csv'}")
    
    # Show best model
    best_flood = res_flood_df.iloc[0]
    ycol = best_flood['y']
    flood_var = best_flood['flood']
    
    print(f"\n{'='*80}")
    print("BEST FLOOD-ONLY MODEL:")
    print(f"{'='*80}")
    print(f"  Dependent variable: {best_flood['y']}")
    print(f"  Flood variable:     {best_flood['flood']}")
    print(f"  Observations:       {best_flood['nobs']}")
    print(f"  R²:                 {best_flood['r2']:.6f}")
    print(f"  Adj. R²:            {best_flood['adjr2']:.6f}")
    print(f"  AIC:                {best_flood['aic']:.2f}")
    print(f"  BIC:                {best_flood['bic']:.2f}")
    print(f"{'='*80}")

# -------------------- Flood-only OLS --------------------

# Build dataframe
flood_df = gdf[[ycol, flood_var, 'geometry']].dropna().copy()
print(f"\nObservations: {len(flood_df)}")

# OLS
y = flood_df[ycol].astype(float)
X = sm.add_constant(flood_df[[flood_var]].astype(float))
ols_flood = sm.OLS(y, X).fit(cov_type="HC3")

print("\nFlood-only OLS results:")
print(ols_flood.summary())

# Save OLS summary to file
with open(OUT_DIR / "flood_only_ols_summary.txt", "w") as f:
    f.write(str(ols_flood.summary()))

# -------------------- Moran's I Test --------------------
# Attach residuals
flood_df['resid_ols_flood'] = ols_flood.resid

# Spatial weights (Queen, same as before)
w_flood = weights.Queen.from_dataframe(flood_df)
w_flood.transform = "r"

mi_flood = Moran(flood_df['resid_ols_flood'], w_flood, permutations=999)

print("\nMoran's I for flood-only OLS residuals:")
print(f"I = {mi_flood.I:.4f}")
print(f"z = {mi_flood.z_norm:.3f}")
print(f"p_perm = {mi_flood.p_sim:.4g}")

# Moran's I results (saved combined with SLM/SEM later)
moran_ols = pd.DataFrame({
    'model': ['OLS'] * 3,
    'statistic': ['Moran\'s I', 'z-score', 'p-value'],
    'value': [mi_flood.I, mi_flood.z_norm, mi_flood.p_sim]
})

# -------------------- Create and save residual map --------------------
print("\nCreating residual map...")
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
flood_df.plot(column='resid_ols_flood', ax=ax, scheme='quantiles', k=5, legend=True,
             cmap='RdBu', vmin=-flood_df['resid_ols_flood'].abs().max(), 
             vmax=flood_df['resid_ols_flood'].abs().max(), edgecolor='none')
ax.set_title("OLS Residuals (HC3) - Flood-Only Model")
ax.axis('off')
fig.tight_layout()
residual_map_path = FIGURE_DIR / "flood_only_residual_map.png"
fig.savefig(residual_map_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Residual map saved to: {residual_map_path}")

# -------------------- Create and save Moran's I scatter plot --------------------
print("Creating Moran's I scatter plot...")
resid_values = flood_df['resid_ols_flood'].values
lag_resid = weights.spatial_lag.lag_spatial(w_flood, resid_values)

fig, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.scatter(resid_values, lag_resid, s=10, alpha=0.7)
m, b = np.polyfit(resid_values, lag_resid, 1)
xs = np.linspace(resid_values.min(), resid_values.max(), 100)
ax.plot(xs, m*xs + b, color='red', lw=1.5, label=f"slope={m:.3f}")
ax.axvline(0, color='grey', lw=0.8)
ax.axhline(0, color='grey', lw=0.8)
ax.set_xlabel("OLS Residual")
ax.set_ylabel("Spatial Lag of Residual (W * resid)")
ax.legend()
ax.set_title(f"Moran Scatterplot of OLS Residuals\nI={mi_flood.I:.4f}, p={mi_flood.p_sim:.4g}")
fig.tight_layout()
moran_map_path = FIGURE_DIR / "flood_only_moran_scatter.png"
fig.savefig(moran_map_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Moran's I scatter plot saved to: {moran_map_path}")

# -------------------- Spatial Lag Model (SLM) --------------------
# Prepare arrays
y_arr = np.array(flood_df[ycol].astype(float)).reshape((-1, 1))
X_arr = np.array(flood_df[[flood_var]].astype(float))

# Fit Spatial Lag Model
slm_flood = ML_Lag(
    y_arr,
    X_arr,
    w=w_flood,
    name_y=ycol,
    name_x=[flood_var],
    name_w="w_flood"
)

print("\nFlood-only Spatial Lag Model (SLM):")
print(slm_flood.summary)

# Save SLM summary to file
with open(OUT_DIR / "flood_only_slm_summary.txt", "w") as f:
    f.write(str(slm_flood.summary))

# -------------------- Spatial Error Model (SEM) --------------------
# Fit Spatial Error Model
print("\nFitting Spatial Error Model (SEM)...")
try:
    sem_flood = ML_Error(
        y_arr,
        X_arr,
        w=w_flood,
        name_y=ycol,
        name_x=[flood_var],
        name_w="w_flood"
    )
except TypeError:
    sem_flood = ML_Error(
        y_arr,
        X_arr,
        w=w_flood,
        name_y=ycol,
        name_x=[flood_var],
        name_w="w_flood"
    )
print("SEM fitted.")

print("\nFlood-only Spatial Error Model (SEM):")
print(sem_flood.summary)

# Save SEM summary to file
with open(OUT_DIR / "flood_only_sem_summary.txt", "w") as f:
    f.write(str(sem_flood.summary))

# -------------------- Compute residual Moran's I for spatial models --------------------
# Compute residuals for SLM and SEM
def get_pred_resid_flood(spreg_obj, y_obs_arr):
    try:
        pred = np.asarray(spreg_obj.predy).flatten()
        resid = y_obs_arr - pred
        return pred, resid
    except Exception:
        return None, None

y_obs_arr = y_arr.flatten()
slm_pred_flood, slm_resid_flood = get_pred_resid_flood(slm_flood, y_obs_arr)
sem_pred_flood, sem_resid_flood = get_pred_resid_flood(sem_flood, y_obs_arr)

if slm_resid_flood is not None:
    mi_slm_flood = safe_moran(pd.Series(slm_resid_flood), w_flood, permutations=999)
    print(f"\nMoran's I on SLM residuals: I={mi_slm_flood.I:.4f}, p_perm={mi_slm_flood.p_sim:.4g}, z={mi_slm_flood.z_norm:.3f}")
    moran_slm = pd.DataFrame({
        'model': ['SLM'] * 4,
        'statistic': ['Moran\'s I', 'Expected I', 'z-score', 'p-value'],
        'value': [mi_slm_flood.I, mi_slm_flood.EI, mi_slm_flood.z_norm, mi_slm_flood.p_sim]
    })
else:
    moran_slm = pd.DataFrame(columns=['model', 'statistic', 'value'])

if sem_resid_flood is not None:
    mi_sem_flood = safe_moran(pd.Series(sem_resid_flood), w_flood, permutations=999)
    print(f"Moran's I on SEM residuals: I={mi_sem_flood.I:.4f}, p_perm={mi_sem_flood.p_sim:.4g}, z={mi_sem_flood.z_norm:.3f}")
    moran_sem = pd.DataFrame({
        'model': ['SEM'] * 4,
        'statistic': ['Moran\'s I', 'Expected I', 'z-score', 'p-value'],
        'value': [mi_sem_flood.I, mi_sem_flood.EI, mi_sem_flood.z_norm, mi_sem_flood.p_sim]
    })
else:
    moran_sem = pd.DataFrame(columns=['model', 'statistic', 'value'])

# Save combined Moran's I (OLS + SLM + SEM)
moran_combined = pd.concat([moran_ols, moran_slm, moran_sem], ignore_index=True)
moran_combined.to_csv(OUT_DIR / "flood_only_moran_i_results.csv", index=False)
print(f"Moran's I results (OLS + SLM + SEM) saved to: {OUT_DIR / 'flood_only_moran_i_results.csv'}")

# -------------------- Pseudo R^2 for flood-only SLM --------------------
y_hat = slm_flood.predy.flatten()

ssr = np.sum((y_obs_arr - y_hat)**2)
sst = np.sum((y_obs_arr - y_obs_arr.mean())**2)

pseudo_r2_flood = 1 - ssr / sst
print(f"\nFlood-only SLM pseudo R^2: {pseudo_r2_flood:.4f}")

# -------------------- Compute flood-only impacts --------------------
# Read impacts directly from spreg's canonical summary output
print("\nExtracting flood-only SLM impacts from summary text...")
summary_text = str(slm_flood.summary)
impacts_list = parse_impacts_from_summary(summary_text)

if impacts_list:
    impacts_df = pd.DataFrame(impacts_list)
    impacts_df = impacts_df.set_index('variable')
    print("\nFlood-only SLM impacts (from spreg summary):")
    print(impacts_df.round(4).to_string())
    row = impacts_list[0]
    direct_val, indirect_val, total_val = row['direct'], row['indirect'], row['total']
else:
    print("⚠️  WARNING: Could not parse impacts from summary text. Falling back to manual computation.")
    rho_flood = float(slm_flood.rho)
    W = w_flood.full()[0]
    I = np.eye(W.shape[0])
    A = inv(I - rho_flood * W)
    diagA = np.diag(A)
    rowSums = A.sum(axis=1)
    beta_flood = slm_flood.betas[1][0]
    direct_val = np.mean(diagA) * beta_flood
    total_val = np.mean(rowSums) * beta_flood
    indirect_val = total_val - direct_val
    print("\nFlood-only SLM impacts (manual):")
    print(f"Direct: {direct_val:.2f}, Indirect: {indirect_val:.2f}, Total: {total_val:.2f}")

# Save SLM pseudo R² + impacts in one file
slm_metrics_df = pd.DataFrame([
    {'metric': 'pseudo_r2', 'value': pseudo_r2_flood},
    {'metric': 'Direct', 'value': direct_val},
    {'metric': 'Indirect', 'value': indirect_val},
    {'metric': 'Total', 'value': total_val}
])
slm_metrics_df.to_csv(OUT_DIR / "flood_only_slm_metrics.csv", index=False)
print(f"SLM metrics (pseudo R² + impacts) saved to: {OUT_DIR / 'flood_only_slm_metrics.csv'}")

# -------------------- Save model objects and data --------------------
# Save flood_df with residuals
flood_df.to_file(OUT_DIR / "flood_only_model_data.gpkg", driver="GPKG", layer="flood_only_data")

# Build OLS coefficients (z_stat = NaN for OLS)
coef_ols = pd.DataFrame({
    'model': ['OLS'] * 2,
    'variable': ['const', flood_var],
    'coefficient': [ols_flood.params.iloc[0], ols_flood.params.iloc[1]],
    'std_err': [ols_flood.bse.iloc[0], ols_flood.bse.iloc[1]],
    'z_stat': [np.nan, np.nan],
    'p_value': [ols_flood.pvalues.iloc[0], ols_flood.pvalues.iloc[1]]
})

# Save SLM coefficients
# Extract rho and betas robustly (handling different spreg versions)
try:
    rho_val = float(np.asarray(slm_flood.rho).flatten()[0])
except Exception:
    rho_val = float(slm_flood.rho)

# Betas may be (k+1)x1 including constant
betas = np.asarray(slm_flood.betas).flatten()
beta_const = betas[0]
beta_flood_coef = betas[1]

# Access std_err, z_stat - convert to arrays and flatten
std_err_arr = np.asarray(slm_flood.std_err).flatten()
z_stat_arr = np.asarray(slm_flood.z_stat).flatten()

# Ensure we have 3 values (const, flood_var, rho)
if len(std_err_arr) < 3 or len(z_stat_arr) < 3:
    raise ValueError(f"Expected 3 coefficients but got {len(std_err_arr)} std_err and {len(z_stat_arr)} z_stat values")

# Compute p-values from z-statistics (two-tailed test)
# This ensures we always have the correct number of p-values
p_value_arr = 2 * (1 - stats.norm.cdf(np.abs(z_stat_arr)))

coef_slm = pd.DataFrame({
    'model': ['SLM'] * 3,
    'variable': ['const', flood_var, 'rho'],
    'coefficient': [beta_const, beta_flood_coef, rho_val],
    'std_err': [std_err_arr[0], std_err_arr[1], std_err_arr[2]],
    'z_stat': [z_stat_arr[0], z_stat_arr[1], z_stat_arr[2]],
    'p_value': [p_value_arr[0], p_value_arr[1], p_value_arr[2]]
})

# Save SEM coefficients
try:
    # SEM uses lambda (lam) instead of rho for spatial error parameter
    lam_val = float(np.asarray(sem_flood.lam).flatten()[0])
except Exception:
    try:
        lam_val = float(sem_flood.lam)
    except Exception:
        lam_val = None

sem_betas_arr = np.asarray(sem_flood.betas).flatten()
sem_std_err_arr = np.asarray(sem_flood.std_err).flatten()
sem_z_stat_arr = np.asarray(sem_flood.z_stat).flatten()
sem_p_value_arr = 2 * (1 - stats.norm.cdf(np.abs(sem_z_stat_arr)))

# SEM has const, flood_var, and lambda (lam)
sem_vars = ['const', flood_var]
if lam_val is not None:
    sem_vars.append('lambda')
    coef_sem = pd.DataFrame({
        'model': ['SEM'] * 3,
        'variable': sem_vars,
        'coefficient': [sem_betas_arr[0], sem_betas_arr[1], lam_val],
        'std_err': [sem_std_err_arr[0], sem_std_err_arr[1], sem_std_err_arr[2]],
        'z_stat': [sem_z_stat_arr[0], sem_z_stat_arr[1], sem_z_stat_arr[2]],
        'p_value': [sem_p_value_arr[0], sem_p_value_arr[1], sem_p_value_arr[2]]
    })
else:
    coef_sem = pd.DataFrame({
        'model': ['SEM'] * 2,
        'variable': sem_vars,
        'coefficient': [sem_betas_arr[0], sem_betas_arr[1]],
        'std_err': [sem_std_err_arr[0], sem_std_err_arr[1]],
        'z_stat': [sem_z_stat_arr[0], sem_z_stat_arr[1]],
        'p_value': [sem_p_value_arr[0], sem_p_value_arr[1]]
    })

# Save combined coefficients (OLS + SLM + SEM)
coef_combined = pd.concat([coef_ols, coef_slm, coef_sem], ignore_index=True)
coef_combined.to_csv(OUT_DIR / "flood_only_coefficients.csv", index=False)
print(f"Coefficients (OLS + SLM + SEM) saved to: {OUT_DIR / 'flood_only_coefficients.csv'}")

print(f"\n{'='*80}")
print("All model results saved to:", OUT_DIR)
print("Figure directory ready at:", FIGURE_DIR)
print(f"{'='*80}")
