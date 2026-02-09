#!/usr/bin/env python3
"""
model_wasteflood_remove_low_baseline.py

Same as model_wasteflood (displacement ~ flood + waste) but:
  - Optional filter: keep only rows where 3_fb_baseline_median >= FB_BASELINE_MIN and not NaN
    (default FB_BASELINE_MIN=50; set to None to use all rows).
  - Uses the same variable selection: Y=2_outflow_max, F=4_flood_p95, W=4_waste_count.
  - Optional flood*waste interaction: set USE_INTERACTION = True to fit
    displacement ~ flood + waste + flood*waste.
  - Saves all results to Output/wasteflood_remove_low_baseline/ and figures to
    Figure/wasteflood_remove_low_baseline/.

Pipeline: same as model_wasteflood (OLS, Moran's I, SLM, SEM, impacts, etc.).

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
        from spreg import ml_lag as ML_Lag, ml_error as ML_Error
    except Exception as e:
        raise ImportError("spreg not found. Install with: conda/pip install pysal spreg. Error: " + str(e))

from multicollinearity_utils import run_multicollinearity_diagnostics


# -------------------- USER PARAMETERS --------------------
GPKG_PATH = Path("/Users/wenlanzhang/PycharmProjects/Waste_Flood_out/Data/4/4_flood_waste_metrics_quadkey.gpkg")
LAYER_NAME = "4_flood_waste_metrics_quadkey"

# Output folder: remove_low_baseline (rows with 3_fb_baseline_median < 50 or NaN excluded)
OUT_DIR = Path("/Users/wenlanzhang/PycharmProjects/Waste_Flood_out/Output/wasteflood_remove_low_baseline")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FIGURE_DIR = Path("/Users/wenlanzhang/PycharmProjects/Waste_Flood_out/Figure/wasteflood_remove_low_baseline")
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# Optional: Restrict to rows with sufficient Facebook baseline (for scaling reliability)
# Set to None to use all rows; set to a number (e.g. 50) to keep only 3_fb_baseline_median >= that and not NaN
FB_BASELINE_MIN = 50  # default: keep only cells with baseline >= 50; None = no filter
BASELINE_COL = "3_fb_baseline_median"

MIN_OBS = 30
MAX_TOP_MODELS = 10
N_PERM = 999

# Same variable selection as model_wasteflood
DISP_SUBS = ['1_outflow', '2_outflow', '3_estimated_outflow', '1_outflow_accumulated', '1_outflow_max',
             '2_displaced_excess', '3_estimated_excess', 'displace', 'displacement', 'outflow',
             'excess_displacement', 'estimated_outflow', 'estimated_excess']
FLOOD_SUBS = ['4_flood', '4_flood_p', '4_flood_p95', '4_flood_mean', '4_flood_max', 'flood', 'flood_p', 'flood_p95', 'flood_mean', 'flood_max', 'flood_exposure', 'flood_risk']
WASTE_SUBS = ['4_waste', '4_waste_count', '4_waste_per', '4_waste_per_population',
              '4_waste_per_svi', 'waste', 'waste_count', 'waste_per',
              'waste_per_population', 'waste_per_svi']

NODATA = -9999.0

# Fixed variables (same as model_wasteflood)
YCOL = '2_outflow_max'
FLOOD_VAR = '4_flood_p95'
WASTE_VAR = '4_waste_count'

# Include flood * waste interaction term when True
USE_INTERACTION = True  # set to False for additive model only


# -------------------- Helper functions (same as model_wasteflood) --------------------
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
    impacts = []
    lines = summary_text.split('\n')
    in_impacts_section = False
    header_found = False
    for line in lines:
        if 'SPATIAL LAG MODEL IMPACTS' in line:
            in_impacts_section = True
            continue
        if in_impacts_section:
            if 'Impacts computed using' in line:
                continue
            if 'Variable' in line and 'Direct' in line and 'Indirect' in line and 'Total' in line:
                header_found = True
                continue
            if header_found and line.strip() and not line.strip().startswith('=') and not line.strip().startswith('END'):
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        total = float(parts[-1])
                        indirect = float(parts[-2])
                        direct = float(parts[-3])
                        variable = ' '.join(parts[:-3]).strip()
                        if variable:
                            impacts.append({'variable': variable, 'Direct': direct, 'Indirect': indirect, 'Total': total})
                    except (ValueError, IndexError):
                        continue
    return impacts

def compute_vif(X_df):
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    X_with_const = sm.add_constant(X_df)
    vif_data = []
    for i, col in enumerate(X_df.columns):
        try:
            vif = variance_inflation_factor(X_with_const.values, i + 1)
            vif_data.append({'variable': col, 'VIF': vif})
        except Exception as e:
            print(f"Warning: Could not compute VIF for {col}: {e}")
            vif_data.append({'variable': col, 'VIF': np.nan})
    return pd.DataFrame(vif_data)

def print_model_comparison(ols, slm=None, sem=None):
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
            r2, aic, bic, loglik = model.rsquared_adj, model.aic, model.bic, model.llf
        else:
            try:
                r2 = getattr(model, 'pr2', getattr(model, 'pseudo_r2', np.nan))
                aic = getattr(model, 'aic', np.nan)
                bic = getattr(model, 'schwarz', getattr(model, 'bic', np.nan))
                loglik = getattr(model, 'logll', getattr(model, 'llf', np.nan))
            except Exception:
                r2 = aic = bic = loglik = np.nan
        comparison.append({'Model': name, 'Adj. R²': r2, 'AIC': aic, 'BIC': bic, 'Log-Likelihood': loglik})
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

print("\nColumn prefixes found:")
prefix_cols = {prefix: [c for c in gdf.columns if c.startswith(prefix)] for prefix in ['1_', '2_', '3_', '4_']}
for prefix, cols in prefix_cols.items():
    if cols:
        print(f"  {prefix}: {len(cols)} columns (sample: {cols[:3]})")

n_missing_geom = gdf.geometry.isna().sum()
geom_types = gdf.geometry.geom_type.value_counts().to_dict()
print(f"\nGeometry types sample: {geom_types}; missing geometries: {n_missing_geom}")

# -------------------- STEP 2: Prepare modeling DataFrame + baseline filter --------------------
print("\n" + "="*80)
print("STEP 2: Preparing cleaned modeling table (with baseline filter)...")
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

# Find candidate variables
disp_candidates = find_candidates(DISP_SUBS, numeric_cols)
flood_candidates = find_candidates(FLOOD_SUBS, numeric_cols)
waste_candidates = [c for c in find_candidates(WASTE_SUBS, numeric_cols) if c not in ('4_waste_per_quadkey_area', '4_waste_count_final')]

if not disp_candidates:
    raise RuntimeError("No displacement candidate variables found.")
if not flood_candidates:
    raise RuntimeError("No flood candidate variables found.")
if not waste_candidates:
    raise RuntimeError("No waste candidate variables found.")

model_cols = ['quadkey', 'geometry'] + sorted(set(disp_candidates + flood_candidates + waste_candidates))
dfg = gdf[model_cols].copy()
dfg = dfg[dfg.geometry.notna()].reset_index(drop=True)
df_clean = dfg.drop(columns='geometry').copy()

missing_summary = df_clean.isna().mean().round(3)
print("\nProportion missing per candidate column (rows with valid geometry, after baseline filter):")
print(missing_summary[missing_summary > 0].sort_values(ascending=False).to_string())
print("="*80)

# -------------------- STEP 3: Fixed variables (same as model_wasteflood) --------------------
print("\n" + "="*80)
print("STEP 3: FLOOD + WASTE MODEL (fixed variables, after baseline filter)")
print("="*80)
print(f"  Dependent variable: {YCOL}")
print(f"  Flood variable:     {FLOOD_VAR}")
print(f"  Waste variable:     {WASTE_VAR}")
print(f"  Interaction (flood*waste): {USE_INTERACTION}")
print("="*80)

if YCOL not in df_clean.columns:
    raise ValueError(f"Specified YCOL '{YCOL}' not found in data.")
if FLOOD_VAR not in df_clean.columns:
    raise ValueError(f"Specified FLOOD_VAR '{FLOOD_VAR}' not found in data.")
if WASTE_VAR not in df_clean.columns:
    raise ValueError(f"Specified WASTE_VAR '{WASTE_VAR}' not found in data.")

ycol, fcol, wcol = YCOL, FLOOD_VAR, WASTE_VAR
# Interaction term name (used in X and outputs)
interaction_col = f"{fcol}_x_{wcol}" if USE_INTERACTION else None

if USE_INTERACTION:
    subdf = df_clean[[ycol, fcol, wcol]].copy()
    subdf[interaction_col] = subdf[fcol].astype(float) * subdf[wcol].astype(float)
    subdf = subdf[[ycol, fcol, wcol, interaction_col]].dropna()
else:
    subdf = df_clean[[ycol, fcol, wcol]].dropna()

nobs = len(subdf)
if nobs < MIN_OBS:
    raise RuntimeError(f"Only {nobs} observations available, need at least {MIN_OBS}.")

y = subdf[ycol].astype(float)
if USE_INTERACTION:
    X = pd.DataFrame({
        fcol: subdf[fcol].astype(float),
        wcol: subdf[wcol].astype(float),
        interaction_col: subdf[interaction_col].astype(float)
    }, index=subdf.index)
else:
    X = pd.DataFrame({fcol: subdf[fcol].astype(float), wcol: subdf[wcol].astype(float)}, index=subdf.index)
X = sm.add_constant(X, has_constant='add')
model = sm.OLS(y, X).fit(cov_type='HC3')

best_model_info = {
    'y': ycol, 'flood': fcol, 'waste': wcol, 'nobs': nobs,
    'r2': float(model.rsquared), 'adjr2': float(model.rsquared_adj),
    'aic': float(model.aic), 'bic': float(model.bic)
}
print(f"\nSELECTED MODEL (after baseline filter):")
print(f"  Dependent variable: {best_model_info['y']}")
print(f"  Flood variable:     {best_model_info['flood']}")
print(f"  Waste variable:     {best_model_info['waste']}")
if USE_INTERACTION:
    print(f"  Interaction:        {interaction_col}")
print(f"  Observations:       {best_model_info['nobs']}")
print(f"  R²:                 {best_model_info['r2']:.6f}")
print(f"  Adj. R²:            {best_model_info['adjr2']:.6f}")
print(f"  AIC:                {best_model_info['aic']:.2f}")
print(f"  BIC:                {best_model_info['bic']:.2f}")

flood_coef = model.params.iloc[1]
flood_pval = model.pvalues.iloc[1]
waste_coef = model.params.iloc[2]
waste_pval = model.pvalues.iloc[2]
enhanced_result_dict = {
    'y': ycol, 'flood_var': fcol, 'flood_coefficient': float(flood_coef), 'flood_p_value': float(flood_pval),
    'waste_var': wcol, 'waste_coefficient': float(waste_coef), 'waste_p_value': float(waste_pval),
    'r2': best_model_info['r2'], 'adjr2': best_model_info['adjr2'], 'nobs': nobs,
    'aic': best_model_info['aic'], 'bic': best_model_info['bic']
}
if USE_INTERACTION:
    enhanced_result_dict['interaction_var'] = interaction_col
    enhanced_result_dict['interaction_coefficient'] = float(model.params.iloc[3])
    enhanced_result_dict['interaction_p_value'] = float(model.pvalues.iloc[3])
enhanced_result = pd.DataFrame([enhanced_result_dict])
enhanced_result.to_csv(OUT_DIR / "wasteflood_model_search_with_coefficients.csv", index=False)
print(f"Model search result saved to: {OUT_DIR / 'wasteflood_model_search_with_coefficients.csv'}")

# Multicollinearity diagnostics (add interaction to df_clean if needed)
pred_cols_wf = [fcol, wcol] if not USE_INTERACTION else [fcol, wcol, interaction_col]
if USE_INTERACTION:
    df_clean[interaction_col] = df_clean[fcol].astype(float) * df_clean[wcol].astype(float)
print("\nMulticollinearity diagnostics (predictors)...")
run_multicollinearity_diagnostics(pred_cols_wf, df_clean, FIGURE_DIR, OUT_DIR)

# -------------------- STEP 4: VIF and OLS --------------------
print("\nSTEP 4: Running VIF analysis and OLS diagnostics...")
if USE_INTERACTION:
    dfg[interaction_col] = dfg[fcol].astype(float) * dfg[wcol].astype(float)
    model_gdf = dfg[['quadkey', 'geometry', ycol, fcol, wcol, interaction_col]].dropna().reset_index(drop=True)
else:
    model_gdf = dfg[['quadkey', 'geometry', ycol, fcol, wcol]].dropna().reset_index(drop=True)
print(f"Observations used in model: {len(model_gdf)}")

y = model_gdf[ycol].astype(float)
if USE_INTERACTION:
    X_df = pd.DataFrame({
        fcol: model_gdf[fcol].astype(float),
        wcol: model_gdf[wcol].astype(float),
        interaction_col: model_gdf[interaction_col].astype(float)
    })
else:
    X_df = pd.DataFrame({fcol: model_gdf[fcol].astype(float), wcol: model_gdf[wcol].astype(float)})

print("\nVIF ANALYSIS (Multicollinearity Check)")
vif_df = compute_vif(X_df)
print(vif_df.to_string(index=False))
high_vif = vif_df[vif_df['VIF'] > 10]
if len(high_vif) > 0:
    print(f"\n⚠️  WARNING: {len(high_vif)} variable(s) with VIF > 10:")
    print(high_vif.to_string(index=False))
else:
    moderate_vif = vif_df[vif_df['VIF'] > 5]
    if len(moderate_vif) > 0:
        print(f"\n⚠️  Note: {len(moderate_vif)} variable(s) with VIF > 5")
    else:
        print("\n✅ No multicollinearity issues detected (all VIF < 5)")
vif_df.to_csv(OUT_DIR / "vif_analysis.csv", index=False)

X = sm.add_constant(X_df, has_constant='add')
ols = sm.OLS(y, X).fit(cov_type='HC3')
print("\nOLS SUMMARY (HC3 Robust Standard Errors)")
print(ols.summary())
with open(OUT_DIR / "ols_summary.txt", "w") as f:
    f.write(str(ols.summary()))
model_gdf['residuals'] = ols.resid

# OLS coefficients: const + flood + waste [+ interaction]
ols_vars = ['const', fcol, wcol]
if USE_INTERACTION:
    ols_vars.append(interaction_col)
coef_ols = pd.DataFrame({
    'model': ['OLS'] * len(ols_vars),
    'variable': ols_vars,
    'coefficient': list(ols.params.iloc[:len(ols_vars)]),
    'std_err': list(ols.bse.iloc[:len(ols_vars)]),
    'z_stat': [np.nan] * len(ols_vars),
    'p_value': list(ols.pvalues.iloc[:len(ols_vars)])
})

# -------------------- STEP 5: Spatial weights & Moran's I --------------------
print("\nSTEP 5: Building spatial weights and Moran's I on residuals...")
if model_gdf.geometry.is_valid.all() is False:
    model_gdf.geometry = model_gdf.geometry.buffer(0)
w = weights.contiguity.Queen.from_dataframe(model_gdf)
w.transform = "r"
print("Weights built. n:", w.n, "components:", w.n_components)
mi = safe_moran(model_gdf['residuals'], w, permutations=N_PERM)
if mi is None:
    raise RuntimeError("Moran's I calculation failed.")
print(f"Moran's I on OLS residuals: I={mi.I:.4f}, Expected={mi.EI:.4f}, z={mi.z_norm:.3f}, p_perm={mi.p_sim:.4g}")

moran_ols = pd.DataFrame({
    'model': ['OLS'] * 4,
    'statistic': ['Moran\'s I', 'Expected I', 'z-score', 'p-value'],
    'value': [mi.I, mi.EI, mi.z_norm, mi.p_sim]
})

# -------------------- STEP 6: Spatial Models --------------------
spatial_threshold_p = 0.05
if mi.p_sim < spatial_threshold_p:
    print(f"\nSignificant spatial autocorrelation (p < {spatial_threshold_p}). Fitting SLM and SEM...")
    y_arr = np.array(model_gdf[ycol].astype(float)).reshape((-1, 1))
    X_arr = np.array(X_df.astype(float))
    try:
        slm = ML_Lag(y_arr, X_arr, w=w, name_y=ycol, name_x=list(X_df.columns), spat_diag=True, name_w='w')
    except TypeError:
        slm = ML_Lag(y_arr, X_arr, w=w, name_y=ycol, name_x=list(X_df.columns), name_w='w')
    try:
        sem = ML_Error(y_arr, X_arr, w=w, name_y=ycol, name_x=list(X_df.columns), spat_diag=True, name_w='w')
    except TypeError:
        sem = ML_Error(y_arr, X_arr, w=w, name_y=ycol, name_x=list(X_df.columns), name_w='w')

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
        model_gdf['slm_residuals'] = slm_resid

    if slm_resid is not None:
        mi_slm = safe_moran(pd.Series(slm_resid), w, permutations=N_PERM)
        print(f"Moran's I on SLM residuals: I={mi_slm.I:.4f}, p_perm={mi_slm.p_sim:.4g}")
        moran_slm = pd.DataFrame({
            'model': ['SLM'] * 4,
            'statistic': ['Moran\'s I', 'Expected I', 'z-score', 'p-value'],
            'value': [mi_slm.I, mi_slm.EI, mi_slm.z_norm, mi_slm.p_sim]
        })
    else:
        moran_slm = pd.DataFrame(columns=['model', 'statistic', 'value'])
    if sem_resid is not None:
        mi_sem = safe_moran(pd.Series(sem_resid), w, permutations=N_PERM)
        print(f"Moran's I on SEM residuals: I={mi_sem.I:.4f}, p_perm={mi_sem.p_sim:.4g}")
        moran_sem = pd.DataFrame({
            'model': ['SEM'] * 4,
            'statistic': ['Moran\'s I', 'Expected I', 'z-score', 'p-value'],
            'value': [mi_sem.I, mi_sem.EI, mi_sem.z_norm, mi_sem.p_sim]
        })
    else:
        moran_sem = pd.DataFrame(columns=['model', 'statistic', 'value'])
    moran_combined = pd.concat([moran_ols, moran_slm, moran_sem], ignore_index=True)
    moran_combined.to_csv(OUT_DIR / "moran_i_results.csv", index=False)

    with open(OUT_DIR / "slm_summary.txt", "w") as f:
        f.write(str(getattr(slm, 'summary', slm)))
    with open(OUT_DIR / "sem_summary.txt", "w") as f:
        f.write(str(getattr(sem, 'summary', sem)))

    summary_text = str(getattr(slm, 'summary', slm))
    impacts_list = parse_impacts_from_summary(summary_text)
    pseudo_r2_val = np.nan
    if impacts_list:
        impacts_df = pd.DataFrame(impacts_list).set_index('variable')
        print("\nSLM impacts:")
        print(impacts_df.round(4).to_string())
        impacts_for_metrics = impacts_list
    else:
        def extract_params(spreg_obj):
            rho = getattr(spreg_obj, 'rho', None) or getattr(spreg_obj, 'lam', None)
            try:
                rho_val = float(np.asarray(rho).flatten()[0])
            except Exception:
                rho_val = float(rho)
            betas = np.asarray(getattr(spreg_obj, 'betas', getattr(spreg_obj, 'beta', None))).flatten()
            return rho_val, betas
        rho, betas = extract_params(slm)
        varnames = list(getattr(slm, 'name_x', X_df.columns))
        beta_no_const = betas[1:] if len(betas) == len(varnames) + 1 else betas[:len(varnames)]
        beta_series = pd.Series(beta_no_const, index=varnames)
        W_full = w.full()[0]
        n = W_full.shape[0]
        A = np.linalg.inv(np.eye(n) - rho * W_full)
        diagA, row_sums = np.diag(A), A.sum(axis=1)
        impacts_for_metrics = []
        for nm in varnames:
            beta_j = float(beta_series.get(nm))
            direct = np.mean(diagA) * beta_j
            total = np.mean(row_sums) * beta_j
            impacts_for_metrics.append({'variable': nm, 'direct': direct, 'indirect': total - direct, 'total': total})
    try:
        y_obs = np.asarray(y).flatten()
        yhat = np.asarray(slm.predy).flatten()
        pseudo_r2_val = 1 - np.sum((y_obs - yhat)**2) / np.sum((y_obs - y_obs.mean())**2)
        print(f"\nPseudo R^2 (SLM): {pseudo_r2_val:.4f}")
    except Exception:
        pass
    slm_metrics_rows = [{'metric': 'pseudo_r2', 'value': pseudo_r2_val}]
    for row in impacts_for_metrics:
        v = row['variable']
        slm_metrics_rows.append({'metric': f'{v}_Direct', 'value': row.get('Direct', row.get('direct'))})
        slm_metrics_rows.append({'metric': f'{v}_Indirect', 'value': row.get('Indirect', row.get('indirect'))})
        slm_metrics_rows.append({'metric': f'{v}_Total', 'value': row.get('Total', row.get('total'))})
    pd.DataFrame(slm_metrics_rows).to_csv(OUT_DIR / "slm_metrics.csv", index=False)

    try:
        rho_val = float(np.asarray(slm.rho).flatten()[0])
    except Exception:
        rho_val = float(slm.rho)
    betas_arr = np.asarray(slm.betas).flatten()
    std_err_arr = np.asarray(slm.std_err).flatten()
    z_stat_arr = np.asarray(slm.z_stat).flatten()
    p_value_arr = 2 * (1 - stats.norm.cdf(np.abs(z_stat_arr)))
    # Rows: const, flood, waste, [interaction], rho
    n_slm = 5 if USE_INTERACTION else 4
    slm_vars = ['const', fcol, wcol]
    if USE_INTERACTION:
        slm_vars.append(interaction_col)
    slm_vars.append('rho')
    n_reg = n_slm - 1  # const + regressors, then rho last
    slm_coef_vals = [betas_arr[i] if i < len(betas_arr) else np.nan for i in range(n_reg)] + [rho_val]
    slm_coef_df = pd.DataFrame({
        'model': ['SLM'] * n_slm,
        'variable': slm_vars,
        'coefficient': slm_coef_vals,
        'std_err': list(std_err_arr[:n_slm]) if len(std_err_arr) >= n_slm else list(std_err_arr) + [np.nan] * (n_slm - len(std_err_arr)),
        'z_stat': list(z_stat_arr[:n_slm]) if len(z_stat_arr) >= n_slm else list(z_stat_arr) + [np.nan] * (n_slm - len(z_stat_arr)),
        'p_value': list(p_value_arr[:n_slm]) if len(p_value_arr) >= n_slm else list(p_value_arr) + [np.nan] * (n_slm - len(p_value_arr))
    })
    try:
        lam_val = float(np.asarray(sem.lam).flatten()[0])
    except Exception:
        lam_val = float(sem.lam) if hasattr(sem, 'lam') else None
    sem_betas_arr = np.asarray(sem.betas).flatten()
    sem_std_err_arr = np.asarray(sem.std_err).flatten()
    sem_z_stat_arr = np.asarray(sem.z_stat).flatten()
    sem_p_value_arr = 2 * (1 - stats.norm.cdf(np.abs(sem_z_stat_arr)))
    sem_vars = ['const', fcol, wcol]
    if USE_INTERACTION:
        sem_vars.append(interaction_col)
    if lam_val is not None:
        sem_vars.append('lambda')
    n_sem = len(sem_vars)
    sem_coef_vals = list(sem_betas_arr[:n_sem]) if len(sem_betas_arr) >= n_sem else list(sem_betas_arr) + [np.nan] * (n_sem - len(sem_betas_arr))
    sem_coef_df = pd.DataFrame({
        'model': ['SEM'] * n_sem, 'variable': sem_vars,
        'coefficient': sem_coef_vals,
        'std_err': list(sem_std_err_arr[:n_sem]) if len(sem_std_err_arr) >= n_sem else list(sem_std_err_arr) + [np.nan] * (n_sem - len(sem_std_err_arr)),
        'z_stat': list(sem_z_stat_arr[:n_sem]) if len(sem_z_stat_arr) >= n_sem else list(sem_z_stat_arr) + [np.nan] * (n_sem - len(sem_z_stat_arr)),
        'p_value': list(sem_p_value_arr[:n_sem]) if len(sem_p_value_arr) >= n_sem else list(sem_p_value_arr) + [np.nan] * (n_sem - len(sem_p_value_arr))
    })
    coef_combined = pd.concat([coef_ols, slm_coef_df, sem_coef_df], ignore_index=True)
    coef_combined.to_csv(OUT_DIR / "coefficients.csv", index=False)
    comp_df = print_model_comparison(ols, slm=slm, sem=sem)
    comp_df.to_csv(OUT_DIR / "model_comparison.csv", index=False)
    if impacts_list:
        pd.DataFrame(impacts_list).to_csv(OUT_DIR / "slm_impacts_spreg_summary.csv", index=False)
else:
    print(f"\nNo significant spatial autocorrelation (p >= {spatial_threshold_p}). OLS only.")
    moran_ols.to_csv(OUT_DIR / "moran_i_results.csv", index=False)
    coef_ols.to_csv(OUT_DIR / "coefficients.csv", index=False)
    comp_df = print_model_comparison(ols)
    comp_df.to_csv(OUT_DIR / "model_comparison.csv", index=False)

model_gdf.to_file(OUT_DIR / "model_data.gpkg", driver="GPKG", layer="model_data")
print(f"\n{'='*80}")
print("All results saved to:", OUT_DIR)
print("Figures to:", FIGURE_DIR)
print(f"{'='*80}")
