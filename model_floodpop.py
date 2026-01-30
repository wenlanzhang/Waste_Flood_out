#!/usr/bin/env python3
"""
model_floodpop.py

Flood + Population (no waste) model pipeline (displacement ~ flood + population):
  1) Read geopackage with displacement, flood, and population vars (no waste)
  2) Sanity checks + cleaning
  3) Model search: displacement ~ flood + population (no interaction)
  4) OLS diagnostics (HC3 robust SE) and Moran's I test on residuals
  5) Fit Spatial Lag Model (SLM) and Spatial Error Model (SEM) if spatial autocorrelation present
  6) Compute SLM impacts (direct/indirect/total) and pseudo-R^2
  7) Save model results and outputs

COLUMN NAMING CONVENTIONS:
  - Script 1 (1y_): 1_outflow_accumulated, 1_outflow_max, etc.
  - Script 2 (2y_): 2_outflow_max, 2_displaced_excess_max, etc.
  - Script 3 (3scale_): 3_worldpop, 3_fb_baseline_median, 3_estimated_outflow_pop_*, etc.
  - Script 4 (4flood_): 4_flood_p95, 4_flood_mean, 4_flood_max, etc.

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
from scipy import stats

from libpysal import weights
from esda.moran import Moran

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

OUT_DIR = Path("/Users/wenlanzhang/PycharmProjects/Waste_Flood_out/Output/floodpop")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FIGURE_DIR = Path("/Users/wenlanzhang/PycharmProjects/Waste_Flood_out/Figure/floodpop")
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

MIN_OBS = 30
MAX_TOP_MODELS = 10
N_PERM = 999

DISP_SUBS = ['1_outflow', '2_outflow', '3_estimated_outflow', '1_outflow_accumulated', '1_outflow_max',
             '2_displaced_excess', '3_estimated_excess', 'displace', 'displacement', 'outflow',
             'excess_displacement', 'estimated_outflow', 'estimated_excess']
FLOOD_SUBS = ['4_flood', '4_flood_p', '4_flood_p95', '4_flood_mean', '4_flood_max', 'flood', 'flood_p',
              'flood_p95', 'flood_mean', 'flood_max', 'flood_exposure', 'flood_risk']
POP_SUBS = ['3_worldpop']
FB_BASELINE_SUBS = ['3_fb_baseline_median', '3_fb_baseline', 'fb_baseline', 'facebook_baseline', 'baseline_median']

NODATA = -9999.0

# # Optional: set to use specific variables; set to None to search
# YCOL = None
# FLOOD_VAR = None
# POP_VAR = None

# YCOL = '3_estimated_outflow_pop_from_2_outflow_max' 
YCOL = '2_outflow_max'
FLOOD_VAR = '4_flood_p95' 
POP_VAR = '3_worldpop'  

def find_candidates(subs, cols):
    found = [c for c in cols if any(s.lower() in c.lower() for s in subs)]
    return sorted(list(set(found)))


def load_gpkg(path: Path, layer: str):
    if not path.exists():
        raise FileNotFoundError(f"GPKG not found: {path}")
    return gpd.read_file(path, layer=layer)


def replace_nodata_with_nan(df, numeric_cols):
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
            df.loc[df[c] == NODATA, c] = np.nan
    return df


def safe_moran(series, w_obj, permutations=N_PERM):
    try:
        return Moran(series.values, w_obj, permutations=permutations)
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
            r2 = getattr(model, 'pr2', getattr(model, 'pseudo_r2', np.nan))
            aic = getattr(model, 'aic', np.nan)
            bic = getattr(model, 'schwarz', getattr(model, 'bic', np.nan))
            loglik = getattr(model, 'logll', getattr(model, 'llf', np.nan))
        comparison.append({'Model': name, 'Adj. R²': r2, 'AIC': aic, 'BIC': bic, 'Log-Likelihood': loglik})
    comp_df = pd.DataFrame(comparison)
    print(comp_df.to_string(index=False))
    print("="*80)
    return comp_df


# -------------------- STEP 1: Read & checks --------------------
print("\nSTEP 1: Reading GeoPackage and doing basic checks...")
print(f"Loading from: {GPKG_PATH}")
gdf = load_gpkg(GPKG_PATH, LAYER_NAME)
print(f"Loaded layer '{LAYER_NAME}' with {len(gdf):,} rows. CRS: {gdf.crs}")
if 'quadkey' not in gdf.columns:
    raise ValueError("Input must contain 'quadkey' column.")

# -------------------- STEP 2: Prepare modeling DataFrame --------------------
print("\n" + "="*80)
print("STEP 2: Preparing cleaned modeling table...")
print("="*80)
numeric_cols = gdf.select_dtypes(include=[np.number]).columns.tolist()
gdf = replace_nodata_with_nan(gdf, numeric_cols)

disp_candidates = find_candidates(DISP_SUBS, numeric_cols)
flood_candidates = find_candidates(FLOOD_SUBS, numeric_cols)
pop_candidates = [c for c in find_candidates(POP_SUBS, numeric_cols) if c != '3_worldpop_safe']
fb_baseline_candidates = [c for c in find_candidates(FB_BASELINE_SUBS, numeric_cols) if c != '3_fb_baseline_safe']
all_pop_candidates = pop_candidates + fb_baseline_candidates

if not disp_candidates:
    raise RuntimeError("No displacement candidate variables found.")
if not flood_candidates:
    raise RuntimeError("No flood candidate variables found.")
if not all_pop_candidates:
    raise RuntimeError("No population candidate variables found.")

model_cols = ['quadkey', 'geometry'] + sorted(set(disp_candidates + flood_candidates + all_pop_candidates))
dfg = gdf[model_cols].copy()
dfg = dfg[dfg.geometry.notna()].reset_index(drop=True)
df_clean = dfg.drop(columns='geometry').copy()

# -------------------- STEP 3: Flood + Population Model Search --------------------
print("\n" + "="*80)
print("STEP 3: FLOOD + POPULATION MODEL SEARCH (NO INTERACTION)")
print("="*80)

if YCOL is not None and FLOOD_VAR is not None and POP_VAR is not None:
    print(f"Using manually specified variables:")
    print(f"  Dependent variable: {YCOL}")
    print(f"  Flood variable:     {FLOOD_VAR}")
    print(f"  Population variable: {POP_VAR}")
    print("="*80)
    if YCOL not in df_clean.columns or FLOOD_VAR not in df_clean.columns or POP_VAR not in df_clean.columns:
        raise ValueError("Specified variable(s) not found in data.")
    ycol, fcol, pcol = YCOL, FLOOD_VAR, POP_VAR
    subdf = df_clean[[ycol, fcol, pcol]].dropna()
    nobs = len(subdf)
    if nobs < MIN_OBS:
        raise RuntimeError(f"Only {nobs} observations available, need at least {MIN_OBS}.")
    y = subdf[ycol].astype(float)
    X = pd.DataFrame({fcol: subdf[fcol].astype(float), pcol: subdf[pcol].astype(float)}, index=subdf.index)
    X = sm.add_constant(X, has_constant='add')
    model = sm.OLS(y, X).fit(cov_type='HC3')
    best_model_info = {
        'y': ycol, 'flood': fcol, 'population': pcol, 'nobs': nobs,
        'r2': float(model.rsquared), 'adjr2': float(model.rsquared_adj),
        'aic': float(model.aic), 'bic': float(model.bic)
    }
    print(f"\nSELECTED MODEL: y={ycol}, flood={fcol}, pop={pcol}, nobs={nobs}, Adj.R²={best_model_info['adjr2']:.4f}")
    enhanced_result = pd.DataFrame([{
        'y': ycol, 'flood_var': fcol, 'flood_coefficient': float(model.params.iloc[1]), 'flood_p_value': float(model.pvalues.iloc[1]),
        'population_var': pcol, 'population_coefficient': float(model.params.iloc[2]), 'population_p_value': float(model.pvalues.iloc[2]),
        'r2': best_model_info['r2'], 'adjr2': best_model_info['adjr2'], 'nobs': nobs, 'aic': best_model_info['aic'], 'bic': best_model_info['bic']
    }])
    enhanced_result.to_csv(OUT_DIR / "floodpop_model_search_with_coefficients.csv", index=False)
else:
    print("Searching models: displacement ~ flood + population (no interaction)")
    print("="*80)
    results = []
    for ycol_search in disp_candidates:
        for fcol_search in flood_candidates:
            for pcol_search in all_pop_candidates:
                if fcol_search == pcol_search:
                    continue
                subdf = df_clean[[ycol_search, fcol_search, pcol_search]].dropna()
                nobs = len(subdf)
                if nobs < MIN_OBS:
                    continue
                y = subdf[ycol_search].astype(float)
                X = pd.DataFrame({fcol_search: subdf[fcol_search].astype(float), pcol_search: subdf[pcol_search].astype(float)}, index=subdf.index)
                X = sm.add_constant(X, has_constant='add')
                try:
                    model = sm.OLS(y, X).fit(cov_type='HC3')
                except Exception:
                    continue
                results.append({
                    'y': ycol_search, 'flood': fcol_search, 'population': pcol_search, 'nobs': nobs,
                    'r2': float(model.rsquared), 'adjr2': float(model.rsquared_adj), 'aic': float(model.aic), 'bic': float(model.bic),
                    'model_obj': model
                })
    if not results:
        raise RuntimeError("No candidate models fit.")
    res_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'model_obj'} for r in results])
    res_df = res_df.sort_values(['adjr2', 'r2'], ascending=[False, False]).reset_index(drop=True)
    n_show = min(MAX_TOP_MODELS, len(res_df))
    print(f"\nFound {len(res_df):,} candidate models; showing top {n_show}:")
    print(res_df.head(n_show).to_string(index=False))
    enhanced_results = []
    for r in results:
        m = r['model_obj']
        enhanced_results.append({
            'y': r['y'], 'flood_var': r['flood'], 'flood_coefficient': float(m.params.iloc[1]), 'flood_p_value': float(m.pvalues.iloc[1]),
            'population_var': r['population'], 'population_coefficient': float(m.params.iloc[2]), 'population_p_value': float(m.pvalues.iloc[2]),
            'r2': r['r2'], 'adjr2': r['adjr2'], 'nobs': r['nobs'], 'aic': r['aic'], 'bic': r['bic']
        })
    pd.DataFrame(enhanced_results).sort_values(['adjr2', 'r2'], ascending=[False, False]).reset_index(drop=True).to_csv(
        OUT_DIR / "floodpop_model_search_with_coefficients.csv", index=False)
    best_row = res_df.iloc[0]
    best = next(r for r in results if r['y'] == best_row['y'] and r['flood'] == best_row['flood'] and r['population'] == best_row['population'])
    best_model_info = {k: v for k, v in best.items() if k != 'model_obj'}
    ycol = best_model_info['y']
    fcol = best_model_info['flood']
    pcol = best_model_info['population']
    print(f"\nBEST MODEL: y={ycol}, flood={fcol}, pop={pcol}, Adj.R²={best_model_info['adjr2']:.4f}")

# -------------------- Multicollinearity diagnostics (figure folder) --------------------
# Predictors: selected flood + pop if fixed, else all flood + pop candidates
pred_cols_fp = [fcol, pcol] if (FLOOD_VAR is not None and POP_VAR is not None) else (flood_candidates + all_pop_candidates)
print("\nMulticollinearity diagnostics (predictors for this step)...")
run_multicollinearity_diagnostics(pred_cols_fp, df_clean, FIGURE_DIR, OUT_DIR)

# -------------------- STEP 4: VIF and OLS --------------------
print("\nSTEP 4: VIF analysis and OLS diagnostics...")
model_gdf = dfg[['quadkey', 'geometry', ycol, fcol, pcol]].dropna().reset_index(drop=True)
print(f"Observations used in model: {len(model_gdf)}")

y = model_gdf[ycol].astype(float)
X_df = pd.DataFrame({fcol: model_gdf[fcol].astype(float), pcol: model_gdf[pcol].astype(float)})

vif_df = compute_vif(X_df)
print("\nVIF ANALYSIS")
print(vif_df.to_string(index=False))
vif_df.to_csv(OUT_DIR / "vif_analysis.csv", index=False)

X = sm.add_constant(X_df, has_constant='add')
ols = sm.OLS(y, X).fit(cov_type='HC3')
print("\nOLS SUMMARY (HC3 Robust SE)")
print(ols.summary())
with open(OUT_DIR / "ols_summary.txt", "w") as f:
    f.write(str(ols.summary()))

model_gdf['residuals'] = ols.resid

coef_ols = pd.DataFrame({
    'model': ['OLS'] * 3,
    'variable': ['const', fcol, pcol],
    'coefficient': [ols.params.iloc[0], ols.params.iloc[1], ols.params.iloc[2]],
    'std_err': [ols.bse.iloc[0], ols.bse.iloc[1], ols.bse.iloc[2]],
    'z_stat': [np.nan, np.nan, np.nan],
    'p_value': [ols.pvalues.iloc[0], ols.pvalues.iloc[1], ols.pvalues.iloc[2]]
})

# -------------------- STEP 5: Spatial weights & Moran's I --------------------
print("\nSTEP 5: Building spatial weights and Moran's I on residuals...")
if model_gdf.geometry.is_valid.all() is False:
    model_gdf.geometry = model_gdf.geometry.buffer(0)
w = weights.contiguity.Queen.from_dataframe(model_gdf)
w.transform = "r"
mi = safe_moran(model_gdf['residuals'], w, permutations=N_PERM)
if mi is None:
    raise RuntimeError("Moran's I calculation failed.")
print(f"Moran's I on OLS residuals: I={mi.I:.4f}, p_perm={mi.p_sim:.4g}")

moran_ols = pd.DataFrame({
    'model': ['OLS'] * 4,
    'statistic': ["Moran's I", 'Expected I', 'z-score', 'p-value'],
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

    moran_slm = pd.DataFrame(columns=['model', 'statistic', 'value'])
    moran_sem = pd.DataFrame(columns=['model', 'statistic', 'value'])
    if slm_resid is not None:
        mi_slm = safe_moran(pd.Series(slm_resid), w, permutations=N_PERM)
        print(f"Moran's I on SLM residuals: I={mi_slm.I:.4f}, p_perm={mi_slm.p_sim:.4g}")
        moran_slm = pd.DataFrame({'model': ['SLM'] * 4, 'statistic': ["Moran's I", 'Expected I', 'z-score', 'p-value'],
                                  'value': [mi_slm.I, mi_slm.EI, mi_slm.z_norm, mi_slm.p_sim]})
    if sem_resid is not None:
        mi_sem = safe_moran(pd.Series(sem_resid), w, permutations=N_PERM)
        print(f"Moran's I on SEM residuals: I={mi_sem.I:.4f}, p_perm={mi_sem.p_sim:.4g}")
        moran_sem = pd.DataFrame({'model': ['SEM'] * 4, 'statistic': ["Moran's I", 'Expected I', 'z-score', 'p-value'],
                                 'value': [mi_sem.I, mi_sem.EI, mi_sem.z_norm, mi_sem.p_sim]})

    moran_combined = pd.concat([moran_ols, moran_slm, moran_sem], ignore_index=True)
    moran_combined.to_csv(OUT_DIR / "moran_i_results.csv", index=False)
    print(f"Moran's I results saved to: {OUT_DIR / 'moran_i_results.csv'}")

    with open(OUT_DIR / "slm_summary.txt", "w") as f:
        f.write(str(getattr(slm, 'summary', slm)))
    with open(OUT_DIR / "sem_summary.txt", "w") as f:
        f.write(str(getattr(sem, 'summary', sem)))

    # STEP 7: SLM impacts and pseudo R²
    summary_text = str(getattr(slm, 'summary', slm))
    impacts_list = parse_impacts_from_summary(summary_text)
    pseudo_r2_val = np.nan
    impacts_for_metrics = impacts_list if impacts_list else []
    if impacts_list:
        print("\nSLM impacts:")
        print(pd.DataFrame(impacts_list).set_index('variable').round(4).to_string())
    else:
        print("⚠️  Could not parse impacts; using manual computation.")
        rho = float(np.asarray(slm.rho).flatten()[0])
        betas = np.asarray(slm.betas).flatten()
        varnames = list(getattr(slm, 'name_x', X_df.columns))
        if len(betas) == len(varnames) + 1:
            beta_no_const = betas[1:]
        else:
            beta_no_const = betas[:len(varnames)]
        W_full = w.full()[0]
        n = W_full.shape[0]
        A = np.linalg.inv(np.eye(n) - rho * W_full)
        diagA = np.diag(A)
        row_sums = A.sum(axis=1)
        for i, nm in enumerate(varnames):
            beta_j = float(beta_no_const[i]) if i < len(beta_no_const) else np.nan
            direct = np.mean(diagA) * beta_j
            total = np.mean(row_sums) * beta_j
            indirect = total - direct
            impacts_for_metrics.append({'variable': nm, 'Direct': direct, 'Indirect': indirect, 'Total': total})
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
    print(f"SLM metrics saved to: {OUT_DIR / 'slm_metrics.csv'}")

    # SLM coefficients
    rho_val = float(np.asarray(slm.rho).flatten()[0])
    betas_arr = np.asarray(slm.betas).flatten()
    std_err_arr = np.asarray(slm.std_err).flatten()
    z_stat_arr = np.asarray(slm.z_stat).flatten()
    p_value_arr = 2 * (1 - stats.norm.cdf(np.abs(z_stat_arr)))
    slm_coef_df = pd.DataFrame({
        'model': ['SLM'] * 4,
        'variable': ['const', fcol, pcol, 'rho'],
        'coefficient': [betas_arr[0], betas_arr[1], betas_arr[2], rho_val],
        'std_err': [std_err_arr[0], std_err_arr[1], std_err_arr[2], std_err_arr[3]],
        'z_stat': [z_stat_arr[0], z_stat_arr[1], z_stat_arr[2], z_stat_arr[3]],
        'p_value': [p_value_arr[0], p_value_arr[1], p_value_arr[2], p_value_arr[3]]
    })

    # SEM coefficients
    try:
        lam_val = float(np.asarray(sem.lam).flatten()[0])
    except Exception:
        lam_val = None
    sem_betas_arr = np.asarray(sem.betas).flatten()
    sem_std_err_arr = np.asarray(sem.std_err).flatten()
    sem_z_stat_arr = np.asarray(sem.z_stat).flatten()
    sem_p_value_arr = 2 * (1 - stats.norm.cdf(np.abs(sem_z_stat_arr)))
    sem_vars = ['const', fcol, pcol]
    if lam_val is not None:
        sem_vars.append('lambda')
        sem_coef_df = pd.DataFrame({
            'model': ['SEM'] * 4,
            'variable': sem_vars,
            'coefficient': [sem_betas_arr[0], sem_betas_arr[1], sem_betas_arr[2], lam_val],
            'std_err': [sem_std_err_arr[0], sem_std_err_arr[1], sem_std_err_arr[2], sem_std_err_arr[3]],
            'z_stat': [sem_z_stat_arr[0], sem_z_stat_arr[1], sem_z_stat_arr[2], sem_z_stat_arr[3]],
            'p_value': [sem_p_value_arr[0], sem_p_value_arr[1], sem_p_value_arr[2], sem_p_value_arr[3]]
        })
    else:
        sem_coef_df = pd.DataFrame({
            'model': ['SEM'] * 3,
            'variable': sem_vars,
            'coefficient': [sem_betas_arr[0], sem_betas_arr[1], sem_betas_arr[2]],
            'std_err': [sem_std_err_arr[0], sem_std_err_arr[1], sem_std_err_arr[2]],
            'z_stat': [sem_z_stat_arr[0], sem_z_stat_arr[1], sem_z_stat_arr[2]],
            'p_value': [sem_p_value_arr[0], sem_p_value_arr[1], sem_p_value_arr[2]]
        })
    coef_combined = pd.concat([coef_ols, slm_coef_df, sem_coef_df], ignore_index=True)
    coef_combined.to_csv(OUT_DIR / "coefficients.csv", index=False)
    print(f"Coefficients (OLS + SLM + SEM) saved to: {OUT_DIR / 'coefficients.csv'}")

    comp_df = print_model_comparison(ols, slm=slm, sem=sem)
    comp_df.to_csv(OUT_DIR / "model_comparison.csv", index=False)
else:
    print(f"\nNo significant spatial autocorrelation (p >= {spatial_threshold_p}). OLS only.")
    moran_ols.to_csv(OUT_DIR / "moran_i_results.csv", index=False)
    coef_ols.to_csv(OUT_DIR / "coefficients.csv", index=False)
    comp_df = print_model_comparison(ols)
    comp_df.to_csv(OUT_DIR / "model_comparison.csv", index=False)

model_gdf.to_file(OUT_DIR / "model_data.gpkg", driver="GPKG", layer="model_data")
print(f"\nAll model results saved to: {OUT_DIR}")
print("Done.")
