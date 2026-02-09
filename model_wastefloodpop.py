#!/usr/bin/env python3
"""
model_wastefloodpop.py

Flood + Waste + Population model pipeline (displacement ~ flood + waste + population):
  1) Read geopackage with displacement, flood, waste, and population vars
  2) Sanity checks + cleaning
  3) Model search: displacement ~ flood + waste + population (no interaction)
  4) OLS diagnostics (HC3 robust SE) and Moran's I test on residuals
  5) Fit Spatial Lag Model (SLM) and Spatial Error Model (SEM) if spatial autocorrelation present
  6) Compute SLM impacts (direct/indirect/total) and pseudo-R^2
  7) Save model results and outputs

COLUMN NAMING CONVENTIONS (updated):
  - Script 1 (1y_): 1_outflow_accumulated, 1_outflow_accumulated_hour0, 1_outflow_max, 1_outflow_max_hour0
  - Script 2 (2y_): 2_outflow_max, 2_displaced_excess_max
  - Script 3 (3scale_): 3_worldpop, 3_estimated_outflow_pop_from_2_outflow_max, 
                       3_estimated_outflow_pop_from_1_outflow_accumulated_hour0, etc.
  - Script 4 (4flood_): 4_flood_p95, 4_flood_mean, 4_flood_max, 4_waste_count, 4_waste_per_population, 4_waste_per_svi_count

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

from multicollinearity_utils import run_multicollinearity_diagnostics


# -------------------- USER PARAMETERS --------------------
GPKG_PATH = Path("/Users/wenlanzhang/PycharmProjects/Waste_Flood_out/Data/4/4_flood_waste_metrics_quadkey.gpkg")
LAYER_NAME = "4_flood_waste_metrics_quadkey"

OUT_DIR = Path("/Users/wenlanzhang/PycharmProjects/Waste_Flood_out/Output/wastefloodpop")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FIGURE_DIR = Path("/Users/wenlanzhang/PycharmProjects/Waste_Flood_out/Figure/wastefloodpop")
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

MIN_OBS = 30                 # minimum observations to consider a model
MAX_TOP_MODELS = 10         # keep top N models by adj. R^2
N_PERM = 999                # permutations for Moran's I

# Updated to match new column naming conventions (1_, 2_, 3_, 4_ prefixes)
DISP_SUBS = ['1_outflow', '2_outflow', '3_estimated_outflow', '1_outflow_accumulated', '1_outflow_max', 
             '2_displaced_excess', '3_estimated_excess', 'displace', 'displacement', 'outflow', 
             'excess_displacement', 'estimated_outflow', 'estimated_excess']
FLOOD_SUBS = ['4_flood', '4_flood_p', '4_flood_p95', '4_flood_mean', '4_flood_max', 'flood', 'flood_p', 'flood_p95', 'flood_mean', 'flood_max', 'flood_exposure', 'flood_risk']
WASTE_SUBS = ['4_waste', '4_waste_count', '4_waste_per', '4_waste_per_population', 
              '4_waste_per_svi', 'waste', 'waste_count', 'waste_per', 
              'waste_per_population', 'waste_per_svi']
# POP_SUBS = ['3_worldpop', '3_population_sum', 'population', 'worldpop', 'pop_sum', 'total_population']
POP_SUBS = ['3_worldpop']
FB_BASELINE_SUBS = ['3_fb_baseline_median', '3_fb_baseline', 'fb_baseline', 'facebook_baseline', 'baseline_median']

NODATA = -9999.0

# Optional: Restrict to rows with sufficient Facebook baseline (for scaling reliability)
# Set to None to use all rows; set to a number (e.g. 50) to keep only 3_fb_baseline_median >= that and not NaN
FB_BASELINE_MIN = 50  # default: keep only cells with baseline >= 50; None = no filter
BASELINE_COL = "3_fb_baseline_median"

# Optional: Set specific variables to use (set to None to search for best fit)
# Priority: user-defined > auto-search
YCOL = '2_outflow_max' # e.g., '3_estimated_outflow_pop_from_2_outflow_max' or None to search
FLOOD_VAR = '4_flood_p95'  # e.g., '4_flood_p95' or None to search
WASTE_VAR = '4_waste_count'  # e.g., '4_waste_count' or None to search
POP_VAR = '3_worldpop'  # e.g., '3_worldpop' or '3_fb_baseline_median' or None to search

# YCOL = None  # e.g., '3_estimated_outflow_pop_from_2_outflow_max' or None to search
# FLOOD_VAR = None  # e.g., '4_flood_p95' or None to search
# WASTE_VAR = None  # e.g., '4_waste_count' or None to search
# POP_VAR = None  # e.g., '3_worldpop' or '3_fb_baseline_median' or None to search

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
                # The format uses fixed-width columns with spaces
                # Example: "         4_flood_p95        12.1738          8.2833         20.4571"
                # Split by whitespace - variable name comes first, then three numeric values
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
                                'Direct': direct,    # Use exact capitalization from spreg
                                'Indirect': indirect,
                                'Total': total
                            })
                    except (ValueError, IndexError):
                        # Skip lines that don't parse correctly
                        continue
    
    return impacts

def compute_vif(X_df):
    """
    Compute Variance Inflation Factor (VIF) for each variable in X_df.
    VIF > 10 indicates potential multicollinearity issues.
    VIF > 5 is sometimes used as a more conservative threshold.
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    
    # Add constant for VIF calculation
    X_with_const = sm.add_constant(X_df)
    
    vif_data = []
    for i, col in enumerate(X_df.columns):
        try:
            vif = variance_inflation_factor(X_with_const.values, i + 1)  # +1 because constant is at index 0
            vif_data.append({'variable': col, 'VIF': vif})
        except Exception as e:
            print(f"Warning: Could not compute VIF for {col}: {e}")
            vif_data.append({'variable': col, 'VIF': np.nan})
    
    vif_df = pd.DataFrame(vif_data)
    return vif_df

def print_model_comparison(ols, slm=None, sem=None):
    """
    Print comparison of OLS, SLM, and SEM models.
    """
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    
    models = [('OLS', ols)]
    if slm is not None:
        models.append(('SLM', slm))
    if sem is not None:
        models.append(('SEM', sem))
    
    comparison = []
    for name, model in models:
        if name == 'OLS':
            r2 = model.rsquared_adj
            aic = model.aic
            bic = model.bic
            loglik = model.llf
        else:
            # For spatial models, try to extract these metrics
            try:
                r2 = getattr(model, 'pr2', getattr(model, 'pseudo_r2', np.nan))
                aic = getattr(model, 'aic', np.nan)
                bic = getattr(model, 'schwarz', getattr(model, 'bic', np.nan))
                loglik = getattr(model, 'logll', getattr(model, 'llf', np.nan))
            except:
                r2 = aic = bic = loglik = np.nan
        
        comparison.append({
            'Model': name,
            'Adj. R²': r2,
            'AIC': aic,
            'BIC': bic,
            'Log-Likelihood': loglik
        })
    
    comp_df = pd.DataFrame(comparison)
    print(comp_df.to_string(index=False))
    print("="*80)
    
    return comp_df


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

# Optional filter: keep only rows with 3_fb_baseline_median >= FB_BASELINE_MIN and not NaN
if FB_BASELINE_MIN is not None:
    if BASELINE_COL not in gdf.columns:
        raise ValueError(f"Column '{BASELINE_COL}' not found; cannot apply baseline filter.")
    n_before = len(gdf)
    gdf = gdf[gdf[BASELINE_COL].notna() & (gdf[BASELINE_COL] >= FB_BASELINE_MIN)].copy()
    gdf = gdf.reset_index(drop=True)
    n_after = len(gdf)
    print(f"\nFilter applied: {BASELINE_COL} >= {FB_BASELINE_MIN} and not NaN.")
    print(f"  Rows before filter: {n_before:,}")
    print(f"  Rows after filter:  {n_after:,} (removed {n_before - n_after:,})")

# Find candidate variables using search strings
disp_candidates = find_candidates(DISP_SUBS, numeric_cols)
flood_candidates = find_candidates(FLOOD_SUBS, numeric_cols)
waste_candidates = [c for c in find_candidates(WASTE_SUBS, numeric_cols) if c not in ('4_waste_per_quadkey_area', '4_waste_count_final')]
pop_candidates = [c for c in find_candidates(POP_SUBS, numeric_cols) if c != '3_worldpop_safe']
fb_baseline_candidates = [c for c in find_candidates(FB_BASELINE_SUBS, numeric_cols) if c != '3_fb_baseline_safe']

# Combine population and fb_baseline candidates for population search
# User can specify either worldpop or fb_baseline
all_pop_candidates = pop_candidates + fb_baseline_candidates

# print("\nDisplacement candidates found:", disp_candidates)
# print("Flood candidates found:", flood_candidates)
# print("Waste candidates found:", waste_candidates)
# print("Population candidates found:", pop_candidates)
# print("Facebook baseline candidates found:", fb_baseline_candidates)
# print("All population candidates (worldpop + fb_baseline):", all_pop_candidates)

if not disp_candidates:
    raise RuntimeError("No displacement candidate variables found.")
if not flood_candidates:
    raise RuntimeError("No flood candidate variables found.")
if not waste_candidates:
    raise RuntimeError("No waste candidate variables found.")
if not all_pop_candidates:
    raise RuntimeError("No population candidate variables found (neither worldpop nor fb_baseline).")

# Create modeling dataframe
model_cols = ['quadkey', 'geometry'] + sorted(set(disp_candidates + flood_candidates + waste_candidates + all_pop_candidates))
dfg = gdf[model_cols].copy()
dfg = dfg[dfg.geometry.notna()].reset_index(drop=True)
df_clean = dfg.drop(columns='geometry').copy()

missing_summary = df_clean.isna().mean().round(3)
print("\nProportion missing per candidate column (rows with valid geometry):")
print(missing_summary[missing_summary > 0].sort_values(ascending=False).to_string())
print("="*80)

# -------------------- STEP 3: Flood + Waste + Population Model Search --------------------
print("\n" + "="*80)
print("STEP 3: FLOOD + WASTE + POPULATION MODEL SEARCH (NO INTERACTION)")
print("="*80)

# Determine which variables to use (prioritize user-defined, then search)
use_ycol = YCOL if YCOL is not None else None
use_flood = FLOOD_VAR if FLOOD_VAR is not None else None
use_waste = WASTE_VAR if WASTE_VAR is not None else None
use_pop = POP_VAR if POP_VAR is not None else None

# Validate user-defined variables if provided
if use_ycol is not None and use_ycol not in df_clean.columns:
    raise ValueError(f"Specified YCOL '{use_ycol}' not found in data.")
if use_flood is not None and use_flood not in df_clean.columns:
    raise ValueError(f"Specified FLOOD_VAR '{use_flood}' not found in data.")
if use_waste is not None and use_waste not in df_clean.columns:
    raise ValueError(f"Specified WASTE_VAR '{use_waste}' not found in data.")
if use_pop is not None and use_pop not in df_clean.columns:
    raise ValueError(f"Specified POP_VAR '{use_pop}' not found in data.")

# If all variables are set, use them directly
if all([use_ycol, use_flood, use_waste, use_pop]):
    print(f"Using manually specified variables:")
    print(f"  Dependent variable: {use_ycol}")
    print(f"  Flood variable:     {use_flood}")
    print(f"  Waste variable:     {use_waste}")
    print(f"  Population variable: {use_pop}")
    print("="*80)
    
    ycol = use_ycol
    fcol = use_flood
    wcol = use_waste
    pcol = use_pop
    
    # Create a single result entry
    subdf = df_clean[[ycol, fcol, wcol, pcol]].dropna()
    nobs = len(subdf)
    if nobs < MIN_OBS:
        raise RuntimeError(f"Only {nobs} observations available, need at least {MIN_OBS}.")
    
    y = subdf[ycol].astype(float)
    X = pd.DataFrame({fcol: subdf[fcol].astype(float), wcol: subdf[wcol].astype(float), pcol: subdf[pcol].astype(float)}, index=subdf.index)
    X = sm.add_constant(X, has_constant='add')
    model = sm.OLS(y, X).fit(cov_type='HC3')
    
    best_model_info = {
        'y': ycol,
        'flood': fcol,
        'waste': wcol,
        'population': pcol,
        'nobs': nobs,
        'r2': float(model.rsquared),
        'adjr2': float(model.rsquared_adj),
        'aic': float(model.aic),
        'bic': float(model.bic)
    }
    
    print(f"\n{'='*80}")
    print("SELECTED MODEL:")
    print(f"{'='*80}")
    print(f"  Dependent variable: {best_model_info['y']}")
    print(f"  Flood variable:     {best_model_info['flood']}")
    print(f"  Waste variable:     {best_model_info['waste']}")
    print(f"  Population variable: {best_model_info['population']}")
    print(f"  Observations:       {best_model_info['nobs']}")
    print(f"  R²:                 {best_model_info['r2']:.6f}")
    print(f"  Adj. R²:            {best_model_info['adjr2']:.6f}")
    print(f"  AIC:                {best_model_info['aic']:.2f}")
    print(f"  BIC:                {best_model_info['bic']:.2f}")
    print(f"{'='*80}")
    
    # Save enhanced CSV with R2, flood variable (name + coefficient + p-value), waste variable (name + coefficient + p-value), population variable (name + coefficient + p-value)
    flood_coef = model.params.iloc[1]  # flood variable coefficient
    flood_pval = model.pvalues.iloc[1]  # flood variable p-value
    waste_coef = model.params.iloc[2]  # waste variable coefficient
    waste_pval = model.pvalues.iloc[2]  # waste variable p-value
    pop_coef = model.params.iloc[3]  # population variable coefficient
    pop_pval = model.pvalues.iloc[3]  # population variable p-value
    
    enhanced_result = pd.DataFrame([{
        'y': ycol,
        'flood_var': fcol,
        'flood_coefficient': float(flood_coef),
        'flood_p_value': float(flood_pval),
        'waste_var': wcol,
        'waste_coefficient': float(waste_coef),
        'waste_p_value': float(waste_pval),
        'population_var': pcol,
        'population_coefficient': float(pop_coef),
        'population_p_value': float(pop_pval),
        'r2': best_model_info['r2'],
        'adjr2': best_model_info['adjr2'],
        'nobs': nobs,
        'aic': best_model_info['aic'],
        'bic': best_model_info['bic']
    }])
    enhanced_result.to_csv(OUT_DIR / "wastefloodpop_model_search_with_coefficients.csv", index=False)
    print(f"Enhanced model results (with coefficients and p-values) saved to: {OUT_DIR / 'wastefloodpop_model_search_with_coefficients.csv'}")
    
else:
    print("Searching models: displacement ~ flood + waste + population (no interaction)")
    print("="*80)
    if use_ycol:
        print(f"  Using specified YCOL: {use_ycol}")
    if use_flood:
        print(f"  Using specified FLOOD_VAR: {use_flood}")
    if use_waste:
        print(f"  Using specified WASTE_VAR: {use_waste}")
    if use_pop:
        print(f"  Using specified POP_VAR: {use_pop}")
    print("="*80)
    
    results = []
    ycols_to_search = [use_ycol] if use_ycol else disp_candidates
    floods_to_search = [use_flood] if use_flood else flood_candidates
    wastes_to_search = [use_waste] if use_waste else waste_candidates
    pops_to_search = [use_pop] if use_pop else all_pop_candidates
    
    for ycol_search in ycols_to_search:
        for fcol_search in floods_to_search:
            for wcol_search in wastes_to_search:
                for pcol_search in pops_to_search:
                    # Skip if any columns are the same
                    if len(set([fcol_search, wcol_search, pcol_search])) < 3:
                        continue
                    subdf = df_clean[[ycol_search, fcol_search, wcol_search, pcol_search]].dropna()
                    nobs = len(subdf)
                    if nobs < MIN_OBS:
                        continue
                    y = subdf[ycol_search].astype(float)
                    X = pd.DataFrame({
                        fcol_search: subdf[fcol_search].astype(float), 
                        wcol_search: subdf[wcol_search].astype(float),
                        pcol_search: subdf[pcol_search].astype(float)
                    }, index=subdf.index)
                    X = sm.add_constant(X, has_constant='add')
                    try:
                        model = sm.OLS(y, X).fit(cov_type='HC3')
                    except Exception:
                        continue
                    results.append({
                        'y': ycol_search, 
                        'flood': fcol_search, 
                        'waste': wcol_search, 
                        'population': pcol_search,
                        'nobs': nobs,
                        'r2': float(model.rsquared), 
                        'adjr2': float(model.rsquared_adj),
                        'aic': float(model.aic), 
                        'bic': float(model.bic), 
                        'model_obj': model
                    })
    
    if not results:
        raise RuntimeError("No candidate models fit. Try lowering MIN_OBS or verifying variables.")
    
    res_df = pd.DataFrame([{k:v for k,v in r.items() if k != 'model_obj'} for r in results])
    res_df = res_df.sort_values(['adjr2', 'r2'], ascending=[False, False]).reset_index(drop=True)
    n_show = min(MAX_TOP_MODELS, len(res_df))
    print(f"\nFound {len(res_df):,} candidate models; showing top {n_show}:")
    print(res_df.head(n_show).to_string(index=False))
    
    # Create enhanced CSV with R2, flood variable (name + coefficient + p-value), waste variable (name + coefficient + p-value), population variable (name + coefficient + p-value)
    enhanced_results = []
    for r in results:
        model = r['model_obj']
        # Extract coefficients and p-values
        # Model has: const, flood_var, waste_var, pop_var (in that order)
        flood_coef = model.params.iloc[1]  # flood variable coefficient
        flood_pval = model.pvalues.iloc[1]  # flood variable p-value
        waste_coef = model.params.iloc[2]  # waste variable coefficient
        waste_pval = model.pvalues.iloc[2]  # waste variable p-value
        pop_coef = model.params.iloc[3]  # population variable coefficient
        pop_pval = model.pvalues.iloc[3]  # population variable p-value
        
        enhanced_results.append({
            'y': r['y'],
            'flood_var': r['flood'],
            'flood_coefficient': float(flood_coef),
            'flood_p_value': float(flood_pval),
            'waste_var': r['waste'],
            'waste_coefficient': float(waste_coef),
            'waste_p_value': float(waste_pval),
            'population_var': r['population'],
            'population_coefficient': float(pop_coef),
            'population_p_value': float(pop_pval),
            'r2': r['r2'],
            'adjr2': r['adjr2'],
            'nobs': r['nobs'],
            'aic': r['aic'],
            'bic': r['bic']
        })
    
    enhanced_df = pd.DataFrame(enhanced_results)
    enhanced_df = enhanced_df.sort_values(['adjr2', 'r2'], ascending=[False, False]).reset_index(drop=True)
    enhanced_df.to_csv(OUT_DIR / "wastefloodpop_model_search_with_coefficients.csv", index=False)
    print(f"Enhanced model search results (with coefficients and p-values) saved to: {OUT_DIR / 'wastefloodpop_model_search_with_coefficients.csv'}")
    
    # Get best model
    best_row = res_df.iloc[0]
    best = next(r for r in results if (r['y'] == best_row['y'] and r['flood'] == best_row['flood'] 
                                       and r['waste'] == best_row['waste'] and r['population'] == best_row['population']))
    best_model_info = {k:v for k,v in best.items() if k != 'model_obj'}
    
    ycol = best_model_info['y']
    fcol = best_model_info['flood']
    wcol = best_model_info['waste']
    pcol = best_model_info['population']
    
    print(f"\n{'='*80}")
    print("BEST MODEL:")
    print(f"{'='*80}")
    print(f"  Dependent variable: {best_model_info['y']}")
    print(f"  Flood variable:     {best_model_info['flood']}")
    print(f"  Waste variable:     {best_model_info['waste']}")
    print(f"  Population variable: {best_model_info['population']}")
    print(f"  Observations:       {best_model_info['nobs']}")
    print(f"  R²:                 {best_model_info['r2']:.6f}")
    print(f"  Adj. R²:            {best_model_info['adjr2']:.6f}")
    print(f"  AIC:                {best_model_info['aic']:.2f}")
    print(f"  BIC:                {best_model_info['bic']:.2f}")
    print(f"{'='*80}")

# -------------------- Create file identifier from variables --------------------
def create_file_identifier(y_var, f_var, w_var, p_var):
    """
    Create a clean, readable identifier from variable names for use in filenames.
    Removes prefixes and special characters to make filenames shorter and readable.
    """
    def clean_var_name(var):
        # Remove common prefixes
        var = var.replace('3_estimated_outflow_pop_from_', 'outflow_')
        var = var.replace('3_estimated_excess_displacement_pop', 'excess_disp')
        var = var.replace('1_outflow_accumulated_hour0', 'outflow_acc')
        var = var.replace('1_outflow_max_hour0', 'outflow_max')
        var = var.replace('2_outflow_max', 'outflow_max')
        var = var.replace('2_displaced_excess_max', 'excess_max')
        var = var.replace('4_flood_p95', 'flood_p95')
        var = var.replace('4_waste_count', 'waste_count')
        var = var.replace('4_waste_per_population', 'waste_per_pop')
        var = var.replace('4_waste_per_svi_count', 'waste_per_svi')
        var = var.replace('3_worldpop', 'worldpop')
        var = var.replace('3_fb_baseline_median', 'fb_baseline')
        var = var.replace('3_fb_baseline_safe', 'fb_baseline_safe')
        # Remove remaining prefixes
        var = var.replace('3_', '').replace('4_', '').replace('1_', '').replace('2_', '')
        return var
    
    y_clean = clean_var_name(y_var)
    f_clean = clean_var_name(f_var)
    w_clean = clean_var_name(w_var)
    p_clean = clean_var_name(p_var)
    
    # Create identifier: Y_yvar_F_fvar_W_wvar_P_pvar
    identifier = f"Y_{y_clean}_F_{f_clean}_W_{w_clean}_P_{p_clean}"
    # Limit length to avoid filesystem issues
    if len(identifier) > 150:
        # Use shorter version
        identifier = f"Y_{y_clean[:20]}_F_{f_clean[:15]}_W_{w_clean[:15]}_P_{p_clean[:15]}"
    return identifier

# Create file identifier
FILE_ID = create_file_identifier(ycol, fcol, wcol, pcol)
print(f"\n{'='*80}")
print(f"FILE IDENTIFIER: {FILE_ID}")
print(f"{'='*80}")

# -------------------- Multicollinearity diagnostics (figure folder) --------------------
# Predictors: selected flood + waste + pop if fixed, else all flood + waste + pop candidates
all_vars_set = (YCOL is not None and FLOOD_VAR is not None and WASTE_VAR is not None and POP_VAR is not None)
pred_cols_wfp = [fcol, wcol, pcol] if all_vars_set else (flood_candidates + waste_candidates + all_pop_candidates)
print("\nMulticollinearity diagnostics (predictors for this step)...")
run_multicollinearity_diagnostics(pred_cols_wfp, df_clean, FIGURE_DIR, OUT_DIR, file_id=FILE_ID)

# -------------------- STEP 4: VIF Analysis and OLS --------------------
print("\nSTEP 4: Running VIF analysis and OLS diagnostics...")
model_gdf = dfg[['quadkey', 'geometry', ycol, fcol, wcol, pcol]].dropna().reset_index(drop=True)
print(f"Observations used in model: {len(model_gdf)}")

y = model_gdf[ycol].astype(float)
X_df = pd.DataFrame({fcol: model_gdf[fcol].astype(float), wcol: model_gdf[wcol].astype(float), pcol: model_gdf[pcol].astype(float)})

# VIF Analysis
print("\n" + "="*80)
print("VIF ANALYSIS (Multicollinearity Check)")
print("="*80)
print("VIF > 10 indicates potential multicollinearity issues")
print("VIF > 5 is a more conservative threshold")
vif_df = compute_vif(X_df)
print(vif_df.to_string(index=False))

high_vif = vif_df[vif_df['VIF'] > 10]
if len(high_vif) > 0:
    print(f"\n⚠️  WARNING: {len(high_vif)} variable(s) with VIF > 10:")
    print(high_vif.to_string(index=False))
    print("Consider: centering variables, removing highly correlated predictors, or using regularization")
else:
    moderate_vif = vif_df[vif_df['VIF'] > 5]
    if len(moderate_vif) > 0:
        print(f"\n⚠️  Note: {len(moderate_vif)} variable(s) with VIF > 5 (moderate multicollinearity):")
        print(moderate_vif.to_string(index=False))
    else:
        print("\n✅ No multicollinearity issues detected (all VIF < 5)")

# Save VIF results
vif_file = OUT_DIR / f"vif_analysis_{FILE_ID}.csv"
vif_df.to_csv(vif_file, index=False)
print(f"\nVIF results saved to: {vif_file}")

# OLS Estimation
X = sm.add_constant(X_df, has_constant='add')
ols = sm.OLS(y, X).fit(cov_type='HC3')
print("\n" + "="*80)
print("OLS SUMMARY (HC3 Robust Standard Errors)")
print("="*80)
print(ols.summary())

# Save OLS summary
ols_summary_file = OUT_DIR / f"ols_summary_{FILE_ID}.txt"
with open(ols_summary_file, "w") as f:
    f.write(f"OLS Model Summary\n")
    f.write(f"{'='*80}\n")
    f.write(f"Dependent variable: {ycol}\n")
    f.write(f"Flood variable:     {fcol}\n")
    f.write(f"Waste variable:     {wcol}\n")
    f.write(f"Population variable: {pcol}\n")
    f.write(f"{'='*80}\n\n")
    f.write(str(ols.summary()))
print(f"OLS summary saved to: {ols_summary_file}")

model_gdf['residuals'] = ols.resid

# OLS coefficients (saved combined with SLM/SEM in spatial branch, or alone in no-spatial branch)
coef_ols = pd.DataFrame({
    'model': ['OLS'] * 4,
    'variable': ['const', fcol, wcol, pcol],
    'coefficient': [ols.params.iloc[0], ols.params.iloc[1], ols.params.iloc[2], ols.params.iloc[3]],
    'std_err': [ols.bse.iloc[0], ols.bse.iloc[1], ols.bse.iloc[2], ols.bse.iloc[3]],
    'z_stat': [np.nan, np.nan, np.nan, np.nan],
    'p_value': [ols.pvalues.iloc[0], ols.pvalues.iloc[1], ols.pvalues.iloc[2], ols.pvalues.iloc[3]]
})

# -------------------- STEP 5: Spatial weights & Moran's I --------------------
print("\nSTEP 5: Building spatial weights (Queen contiguity) and testing residual spatial autocorrelation...")
if model_gdf.geometry.is_valid.all() is False:
    model_gdf.geometry = model_gdf.geometry.buffer(0)

w = weights.contiguity.Queen.from_dataframe(model_gdf)
w.transform = "r"
print("Weights built. n:", w.n, "components:", w.n_components)
mi = safe_moran(model_gdf['residuals'], w, permutations=N_PERM)
if mi is None:
    raise RuntimeError("Moran's I calculation failed.")
print(f"\nMoran's I on OLS residuals: I={mi.I:.4f}, Expected={mi.EI:.4f}, z={mi.z_norm:.3f}, p_perm={mi.p_sim:.4g}")

# Moran's I for OLS (saved combined with SLM/SEM in spatial branch)
moran_ols = pd.DataFrame({
    'model': ['OLS'] * 4,
    'statistic': ['Moran\'s I', 'Expected I', 'z-score', 'p-value'],
    'value': [mi.I, mi.EI, mi.z_norm, mi.p_sim]
})

# -------------------- STEP 6: Spatial Models --------------------
spatial_threshold_p = 0.05
if mi.p_sim < spatial_threshold_p:
    print(f"\nSignificant spatial autocorrelation detected (p < {spatial_threshold_p}). Fitting spatial models...")
    
    # Prepare arrays: spreg expects y as (n,1) and X without constant
    y_arr = np.array(model_gdf[ycol].astype(float)).reshape((-1, 1))
    X_arr = np.array(X_df.astype(float))  # includes flood, waste, population; do NOT include constant
    
    print("\nFitting Spatial Lag Model (ML)...")
    try:
        slm = ML_Lag(y_arr, X_arr, w=w, name_y=ycol, name_x=list(X_df.columns), spat_diag=True, name_w='w')
    except TypeError:
        slm = ML_Lag(y_arr, X_arr, w=w, name_y=ycol, name_x=list(X_df.columns), name_w='w')
    print("SLM fitted.")
    
    print("\nFitting Spatial Error Model (ML)...")
    try:
        sem = ML_Error(y_arr, X_arr, w=w, name_y=ycol, name_x=list(X_df.columns), spat_diag=True, name_w='w')
    except TypeError:
        sem = ML_Error(y_arr, X_arr, w=w, name_y=ycol, name_x=list(X_df.columns), name_w='w')
    print("SEM fitted.")
    
    # Compute residual Moran's I for spatial models
    def get_pred_resid(spreg_obj):
        try:
            pred = np.asarray(spreg_obj.predy).flatten()
            resid = np.asarray(y_arr).flatten() - pred
            return pred, resid
        except Exception:
            return None, None
    
    slm_pred, slm_resid = get_pred_resid(slm)
    sem_pred, sem_resid = get_pred_resid(sem)
    
    if slm_resid is not None:
        mi_slm = safe_moran(pd.Series(slm_resid), w, permutations=N_PERM)
        print(f"\nMoran's I on SLM residuals: I={mi_slm.I:.4f}, p_perm={mi_slm.p_sim:.4g}, z={mi_slm.z_norm:.3f}")
        moran_slm = pd.DataFrame({
            'model': ['SLM'] * 4,
            'statistic': ['Moran\'s I', 'Expected I', 'z-score', 'p-value'],
            'value': [mi_slm.I, mi_slm.EI, mi_slm.z_norm, mi_slm.p_sim]
        })
    else:
        moran_slm = pd.DataFrame(columns=['model', 'statistic', 'value'])
    if sem_resid is not None:
        mi_sem = safe_moran(pd.Series(sem_resid), w, permutations=N_PERM)
        print(f"Moran's I on SEM residuals: I={mi_sem.I:.4f}, p_perm={mi_sem.p_sim:.4g}, z={mi_sem.z_norm:.3f}")
        moran_sem = pd.DataFrame({
            'model': ['SEM'] * 4,
            'statistic': ['Moran\'s I', 'Expected I', 'z-score', 'p-value'],
            'value': [mi_sem.I, mi_sem.EI, mi_sem.z_norm, mi_sem.p_sim]
        })
    else:
        moran_sem = pd.DataFrame(columns=['model', 'statistic', 'value'])
    moran_combined = pd.concat([moran_ols, moran_slm, moran_sem], ignore_index=True)
    moran_file = OUT_DIR / f"moran_i_results_{FILE_ID}.csv"
    moran_combined.to_csv(moran_file, index=False)
    print(f"Moran's I results (OLS + SLM + SEM) saved to: {moran_file}")
    
    # Save SLM/SEM summaries
    slm_summary_file = OUT_DIR / f"slm_summary_{FILE_ID}.txt"
    with open(slm_summary_file, "w") as f:
        f.write(f"Spatial Lag Model (SLM) Summary\n")
        f.write(f"{'='*80}\n")
        f.write(f"Dependent variable: {ycol}\n")
        f.write(f"Flood variable:     {fcol}\n")
        f.write(f"Waste variable:     {wcol}\n")
        f.write(f"Population variable: {pcol}\n")
        f.write(f"{'='*80}\n\n")
        f.write(str(getattr(slm, 'summary', slm)))
    print(f"SLM summary saved to: {slm_summary_file}")
    
    sem_summary_file = OUT_DIR / f"sem_summary_{FILE_ID}.txt"
    with open(sem_summary_file, "w") as f:
        f.write(f"Spatial Error Model (SEM) Summary\n")
        f.write(f"{'='*80}\n")
        f.write(f"Dependent variable: {ycol}\n")
        f.write(f"Flood variable:     {fcol}\n")
        f.write(f"Waste variable:     {wcol}\n")
        f.write(f"Population variable: {pcol}\n")
        f.write(f"{'='*80}\n\n")
        f.write(str(getattr(sem, 'summary', sem)))
    print(f"SEM summary saved to: {sem_summary_file}")
    
    # -------------------- STEP 7: Compute SLM impacts --------------------
    print("\nSTEP 7: Extracting SLM impacts from summary text and computing pseudo-R^2...")
    
    # Read impacts directly from spreg's canonical summary output
    summary_text = str(getattr(slm, 'summary', slm))
    impacts_list = parse_impacts_from_summary(summary_text)
    
    if impacts_list:
        # Use spreg's impacts table exactly as provided - no modifications
        # The parse function already uses correct column names (Direct, Indirect, Total)
        impacts_df = pd.DataFrame(impacts_list)
        impacts_df = impacts_df.set_index('variable')
        
        print("\nSLM impacts (from spreg's canonical summary output):")
        print(impacts_df.round(4).to_string())
        
        # Save impacts exactly as spreg provides them (Direct, Indirect, Total columns)
        impacts_file = OUT_DIR / f"slm_impacts_spreg_summary_{FILE_ID}.csv"
        impacts_df.to_csv(impacts_file)
        print(f"SLM impacts saved to: {impacts_file}")
    else:
        print("⚠️  WARNING: Could not parse impacts from summary text. Falling back to manual computation.")
        # Fallback to manual computation
        def extract_params(spreg_obj):
            rho = getattr(spreg_obj, 'rho', None)
            if rho is None:
                rho = getattr(spreg_obj, 'lam', None)
            try:
                rho_val = float(np.asarray(rho).flatten()[0])
            except Exception:
                rho_val = float(rho)
            # betas may be (k+1)x1 including constant
            betas = np.asarray(getattr(spreg_obj, 'betas', getattr(spreg_obj, 'beta', getattr(spreg_obj, 'b', None))))
            betas = betas.flatten()
            return rho_val, betas
        
        rho, betas = extract_params(slm)
        print(f"Estimated rho: {rho:.5f}")
        
        # Determine varnames
        try:
            varnames = list(slm.name_x)
        except Exception:
            varnames = list(X_df.columns)
        
        # Align betas: many spreg versions have betas with constant first
        if len(betas) == len(varnames) + 1:
            beta_no_const = betas[1:]
        elif len(betas) == len(varnames):
            beta_no_const = betas
        else:
            beta_no_const = betas[:len(varnames)]
        
        beta_series = pd.Series(beta_no_const, index=varnames)
        
        W_full = w.full()[0]
        n = W_full.shape[0]
        I_mat = np.eye(n)
        try:
            A = np.linalg.inv(I_mat - rho * W_full)
        except np.linalg.LinAlgError as e:
            raise RuntimeError("Matrix inversion failed for SLM impacts. Check rho and W. Error: " + str(e))
        diagA = np.diag(A)
        row_sums = A.sum(axis=1)
        
        effects = []
        for nm in varnames:
            beta_j = float(beta_series.get(nm))
            direct = np.mean(diagA) * beta_j
            total = np.mean(row_sums) * beta_j
            indirect = total - direct
            effects.append({'variable': nm, 'beta': beta_j, 'direct_mean': direct, 'indirect_mean': indirect, 'total_mean': total})
        
        effects_df = pd.DataFrame(effects).set_index('variable')
        print("\nSLM point-estimate impacts (manual computation fallback):")
        print(effects_df.round(4).to_string())
        
        # Save impacts with different name to indicate manual computation
        impacts_file = OUT_DIR / f"slm_impacts_manual_computation_{FILE_ID}.csv"
        effects_df.to_csv(impacts_file)
        print(f"⚠️  Manual computation impacts saved to: {impacts_file}")
    
    # Pseudo R^2 and SLM metrics (pseudo R² + impacts in one file)
    pseudo_r2 = np.nan
    try:
        y_obs = np.asarray(y).flatten()
        yhat = np.asarray(slm.predy).flatten()
        ssr = np.sum((y_obs - yhat)**2)
        sst = np.sum((y_obs - y_obs.mean())**2)
        pseudo_r2 = 1 - ssr / sst
        print(f"\nPseudo R^2 (SLM): {pseudo_r2:.4f}")
    except Exception:
        print("Could not compute pseudo R^2 for SLM (predictions unavailable).")
    
    # Save SLM metrics (pseudo R² + impacts)
    slm_metrics_rows = [{'metric': 'pseudo_r2', 'value': pseudo_r2}]
    if impacts_list:
        for row in impacts_list:
            v = row['variable']
            slm_metrics_rows.append({'metric': f'{v}_Direct', 'value': row['Direct']})
            slm_metrics_rows.append({'metric': f'{v}_Indirect', 'value': row['Indirect']})
            slm_metrics_rows.append({'metric': f'{v}_Total', 'value': row['Total']})
    slm_metrics_file = OUT_DIR / f"slm_metrics_{FILE_ID}.csv"
    pd.DataFrame(slm_metrics_rows).to_csv(slm_metrics_file, index=False)
    print(f"SLM metrics (pseudo R² + impacts) saved to: {slm_metrics_file}")
    
    # Save SLM coefficients
    # NOTE: For publication, report:
    #   - ρ (rho) and its p-value
    #   - β for waste and its p-value
    #   - β for flood and its p-value (note if not significant)
    #   - β for population and its p-value
    #   - Direct/indirect/total impacts for waste (primary), flood (secondary), and population
    #   Use slm_coefficients_*.csv and slm_impacts_spreg_summary_*.csv for tables
    try:
        rho_val = float(np.asarray(slm.rho).flatten()[0])
    except Exception:
        rho_val = float(slm.rho)
    
    betas_arr = np.asarray(slm.betas).flatten()
    std_err_arr = np.asarray(slm.std_err).flatten()
    z_stat_arr = np.asarray(slm.z_stat).flatten()
    p_value_arr = 2 * (1 - stats.norm.cdf(np.abs(z_stat_arr)))
    
    # Ensure we have 4 coefficients (const, flood, waste, pop) + rho
    if len(betas_arr) < 4 or len(std_err_arr) < 5:
        raise ValueError(f"Expected at least 4 betas and 5 std_err values, got {len(betas_arr)} and {len(std_err_arr)}")
    
    slm_coef_df = pd.DataFrame({
        'model': ['SLM'] * 5,
        'variable': ['const', fcol, wcol, pcol, 'rho'],
        'coefficient': [betas_arr[0], betas_arr[1], betas_arr[2], betas_arr[3], rho_val],
        'std_err': [std_err_arr[0], std_err_arr[1], std_err_arr[2], std_err_arr[3], std_err_arr[4]],
        'z_stat': [z_stat_arr[0], z_stat_arr[1], z_stat_arr[2], z_stat_arr[3], z_stat_arr[4]],
        'p_value': [p_value_arr[0], p_value_arr[1], p_value_arr[2], p_value_arr[3], p_value_arr[4]]
    })
    
    # Save SEM coefficients
    try:
        # SEM uses lambda (lam) instead of rho for spatial error parameter
        lam_val = float(np.asarray(sem.lam).flatten()[0])
    except Exception:
        try:
            lam_val = float(sem.lam)
        except Exception:
            lam_val = None
    
    sem_betas_arr = np.asarray(sem.betas).flatten()
    sem_std_err_arr = np.asarray(sem.std_err).flatten()
    sem_z_stat_arr = np.asarray(sem.z_stat).flatten()
    sem_p_value_arr = 2 * (1 - stats.norm.cdf(np.abs(sem_z_stat_arr)))
    
    # SEM has const, flood, waste, pop, and lambda (lam)
    sem_vars = ['const', fcol, wcol, pcol]
    if lam_val is not None:
        sem_vars.append('lambda')
        # Ensure we have 5 coefficients (const, flood, waste, pop) + lambda
        if len(sem_betas_arr) < 4 or len(sem_std_err_arr) < 5:
            raise ValueError(f"Expected at least 4 betas and 5 std_err values for SEM, got {len(sem_betas_arr)} and {len(sem_std_err_arr)}")
        sem_coef_df = pd.DataFrame({
            'model': ['SEM'] * len(sem_vars),
            'variable': sem_vars,
            'coefficient': [sem_betas_arr[0], sem_betas_arr[1], sem_betas_arr[2], sem_betas_arr[3], lam_val],
            'std_err': [sem_std_err_arr[0], sem_std_err_arr[1], sem_std_err_arr[2], sem_std_err_arr[3], sem_std_err_arr[4]],
            'z_stat': [sem_z_stat_arr[0], sem_z_stat_arr[1], sem_z_stat_arr[2], sem_z_stat_arr[3], sem_z_stat_arr[4]],
            'p_value': [sem_p_value_arr[0], sem_p_value_arr[1], sem_p_value_arr[2], sem_p_value_arr[3], sem_p_value_arr[4]]
        })
    else:
        sem_coef_df = pd.DataFrame({
            'model': ['SEM'] * len(sem_vars),
            'variable': sem_vars,
            'coefficient': [sem_betas_arr[0], sem_betas_arr[1], sem_betas_arr[2], sem_betas_arr[3]],
            'std_err': [sem_std_err_arr[0], sem_std_err_arr[1], sem_std_err_arr[2], sem_std_err_arr[3]],
            'z_stat': [sem_z_stat_arr[0], sem_z_stat_arr[1], sem_z_stat_arr[2], sem_z_stat_arr[3]],
            'p_value': [sem_p_value_arr[0], sem_p_value_arr[1], sem_p_value_arr[2], sem_p_value_arr[3]]
        })
    # Save combined coefficients (OLS + SLM + SEM)
    coef_combined = pd.concat([coef_ols, slm_coef_df, sem_coef_df], ignore_index=True)
    coef_file = OUT_DIR / f"coefficients_{FILE_ID}.csv"
    coef_combined.to_csv(coef_file, index=False)
    print(f"Coefficients (OLS + SLM + SEM) saved to: {coef_file}")
    
    # Model Comparison
    comp_df = print_model_comparison(ols, slm=slm, sem=sem)
    comp_file = OUT_DIR / f"model_comparison_{FILE_ID}.csv"
    comp_df.to_csv(comp_file, index=False)
    print(f"Model comparison saved to: {comp_file}")
    
    print(f"\nSaved SLM/SEM summaries and SLM impacts to output folder: {OUT_DIR}")
    
else:
    print(f"\nNo significant spatial autocorrelation detected (p >= {spatial_threshold_p}). OLS may be adequate with robust SEs.")
    print("Still consider reporting the Moran's I test and the OLS model diagnostics.")
    moran_file = OUT_DIR / f"moran_i_results_{FILE_ID}.csv"
    moran_ols.to_csv(moran_file, index=False)
    print(f"Moran's I results (OLS only) saved to: {moran_file}")
    coef_file = OUT_DIR / f"coefficients_{FILE_ID}.csv"
    coef_ols.to_csv(coef_file, index=False)
    print(f"Coefficients (OLS only) saved to: {coef_file}")
    # Save OLS-only comparison
    comp_df = print_model_comparison(ols)
    comp_file = OUT_DIR / f"model_comparison_{FILE_ID}.csv"
    comp_df.to_csv(comp_file, index=False)
    print(f"Model comparison saved to: {comp_file}")

# Save model data with residuals
model_data_file = OUT_DIR / f"model_data_{FILE_ID}.gpkg"
model_gdf.to_file(model_data_file, driver="GPKG", layer="model_data")
print(f"Model data with residuals saved to: {model_data_file}")

print(f"\n{'='*80}")
print("All model results saved to:", OUT_DIR)
print("Figure directory ready at:", FIGURE_DIR)
print(f"{'='*80}")
