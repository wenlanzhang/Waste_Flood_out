#! /Users/wenlanzhang/miniconda3/envs/geo_env_LLM/bin/python
"""
02_run_regression.py

Run regression models: displacement ~ flood + waste

This script:
1. Loads prepared data
2. Tests different variable specifications
3. Runs OLS regression with robust standard errors
4. Tests for spatial autocorrelation (Moran's I)
5. Runs spatial models (SLM/SEM) if needed
6. Displays results

No results saved - just displays output for comparison
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import statsmodels.api as sm
from scipy import stats

# Spatial libraries
from libpysal import weights
from esda.moran import Moran

try:
    from spreg import ML_Lag, ML_Error
except ImportError:
    try:
        from spreg import ml_lag as ML_Lag, ml_error as ML_Error
    except ImportError:
        print("Warning: spreg not available. Spatial models will be skipped.")
        ML_Lag = None
        ML_Error = None

# ============================================================================
# CONFIGURATION
# ============================================================================

# Variable selection (can be changed)
Y_VAR = '3_estimated_outflow_pop_from_1_outflow_accumulated_hour0'  # Step 1 accumulated
FLOOD_VAR = '4_flood_p95'
WASTE_VAR = '4_waste_per_population'  # Normalized by population

# Alternative specifications to test
ALTERNATIVE_SPECS = [
    {
        'name': 'Main Specification',
        'y': '3_estimated_outflow_pop_from_1_outflow_accumulated_hour0',
        'flood': '4_flood_p95',
        'waste': '4_waste_per_population',
    },
    {
        'name': 'CSAT Method (Step 2)',
        'y': '3_estimated_outflow_pop_from_2_outflow_max',
        'flood': '4_flood_p95',
        'waste': '4_waste_per_population',
    },
    {
        'name': 'Waste Count (not normalized)',
        'y': '3_estimated_outflow_pop_from_1_outflow_accumulated_hour0',
        'flood': '4_flood_p95',
        'waste': '4_waste_count',
    },
    {
        'name': 'Waste per SVI',
        'y': '3_estimated_outflow_pop_from_1_outflow_accumulated_hour0',
        'flood': '4_flood_p95',
        'waste': '4_waste_per_svi_count',
    },
]

MIN_OBS = 30
N_PERM = 999

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def safe_moran(series, w_obj, permutations=N_PERM):
    """Calculate Moran's I safely."""
    try:
        mi = Moran(series.values, w_obj, permutations=permutations)
        return mi
    except Exception as e:
        print(f"  Moran's I error: {e}")
        return None

def print_coefficients(model, var_names):
    """Print coefficients in a clean format."""
    print("\n" + "-"*80)
    print("COEFFICIENTS:")
    print("-"*80)
    print(f"{'Variable':<30} {'Coefficient':>15} {'Std Err':>15} {'P-value':>15} {'Signif':>10}")
    print("-"*80)
    
    for i, var in enumerate(['const'] + var_names):
        if i < len(model.params):
            coef = model.params.iloc[i]
            se = model.bse.iloc[i]
            pval = model.pvalues.iloc[i]
            
            # Significance stars
            if pval < 0.001:
                sig = "***"
            elif pval < 0.01:
                sig = "**"
            elif pval < 0.05:
                sig = "*"
            else:
                sig = ""
            
            print(f"{var:<30} {coef:>15.4f} {se:>15.4f} {pval:>15.4f} {sig:>10}")
    
    print("-"*80)
    print(f"R²: {model.rsquared:.4f}")
    print(f"Adj. R²: {model.rsquared_adj:.4f}")
    print(f"AIC: {model.aic:.2f}")
    print(f"BIC: {model.bic:.2f}")
    print(f"N: {model.nobs:,}")

# ============================================================================
# LOAD DATA
# ============================================================================
print("="*80)
print("LOADING DATA")
print("="*80)

BASE_DIR = Path("/Users/wenlanzhang/PycharmProjects/Waste_Flood_out")
DATA_DIR = BASE_DIR / "Data" / "4"
GPKG_PATH = DATA_DIR / "4_flood_waste_metrics_quadkey.gpkg"
LAYER_NAME = "4_flood_waste_metrics_quadkey"

gdf = gpd.read_file(GPKG_PATH, layer=LAYER_NAME)
print(f"✓ Loaded {len(gdf):,} quadkeys")

# Handle NODATA
NODATA = -9999.0
numeric_cols = gdf.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if col in gdf.columns:
        gdf[col] = pd.to_numeric(gdf[col], errors='coerce')
        gdf.loc[gdf[col] == NODATA, col] = np.nan

# ============================================================================
# RUN MODELS FOR EACH SPECIFICATION
# ============================================================================

results_summary = []

for spec in ALTERNATIVE_SPECS:
    print("\n" + "="*80)
    print(f"MODEL: {spec['name']}")
    print("="*80)
    print(f"  Y:   {spec['y']}")
    print(f"  Flood: {spec['flood']}")
    print(f"  Waste: {spec['waste']}")
    
    # Check variables exist
    missing = []
    for var_type, var_name in [('Y', spec['y']), ('Flood', spec['flood']), ('Waste', spec['waste'])]:
        if var_name not in gdf.columns:
            missing.append(f"{var_type}: {var_name}")
    
    if missing:
        print(f"  ✗ Missing variables: {', '.join(missing)}")
        continue
    
    # Prepare data
    model_cols = ['quadkey', 'geometry', spec['y'], spec['flood'], spec['waste']]
    model_gdf = gdf[model_cols].dropna().reset_index(drop=True)
    
    if len(model_gdf) < MIN_OBS:
        print(f"  ✗ Insufficient observations: {len(model_gdf)} (need {MIN_OBS})")
        continue
    
    print(f"  ✓ {len(model_gdf):,} observations")
    
    # Prepare variables
    y = model_gdf[spec['y']].astype(float)
    X_df = pd.DataFrame({
        spec['flood']: model_gdf[spec['flood']].astype(float),
        spec['waste']: model_gdf[spec['waste']].astype(float)
    })
    X = sm.add_constant(X_df, has_constant='add')
    
    # ========================================================================
    # OLS REGRESSION
    # ========================================================================
    print("\n" + "-"*80)
    print("OLS REGRESSION (HC3 Robust Standard Errors)")
    print("-"*80)
    
    ols = sm.OLS(y, X).fit(cov_type='HC3')
    
    var_names = [spec['flood'], spec['waste']]
    print_coefficients(ols, var_names)
    
    # Store residuals for spatial test
    model_gdf['residuals'] = ols.resid
    
    # ========================================================================
    # SPATIAL AUTOCORRELATION TEST
    # ========================================================================
    print("\n" + "-"*80)
    print("SPATIAL AUTOCORRELATION TEST (Moran's I on OLS Residuals)")
    print("-"*80)
    
    # Fix geometries if needed
    if not model_gdf.geometry.is_valid.all():
        model_gdf.geometry = model_gdf.geometry.buffer(0)
    
    # Build spatial weights
    try:
        w = weights.contiguity.Queen.from_dataframe(model_gdf)
        w.transform = "r"
        print(f"  Spatial weights: {w.n} observations, {w.n_components} components")
        
        # Test Moran's I
        mi = safe_moran(model_gdf['residuals'], w, permutations=N_PERM)
        if mi:
            print(f"  Moran's I: {mi.I:.4f}")
            print(f"  Expected I: {mi.EI:.4f}")
            print(f"  z-score: {mi.z_norm:.4f}")
            print(f"  p-value: {mi.p_sim:.4f}")
            
            if mi.p_sim < 0.05:
                print(f"  ⚠ Significant spatial autocorrelation detected (p < 0.05)")
                print(f"    → Spatial models recommended")
                run_spatial = True
            else:
                print(f"  ✓ No significant spatial autocorrelation")
                run_spatial = False
        else:
            run_spatial = False
            
    except Exception as e:
        print(f"  ✗ Could not build spatial weights: {e}")
        run_spatial = False
    
    # ========================================================================
    # SPATIAL MODELS (if needed)
    # ========================================================================
    if run_spatial and ML_Lag is not None:
        print("\n" + "-"*80)
        print("SPATIAL MODELS")
        print("-"*80)
        
        # Prepare arrays for spreg
        y_arr = np.array(y).reshape((-1, 1))
        X_arr = np.array(X_df.astype(float))  # No constant
        
        # Spatial Lag Model (SLM)
        print("\nSpatial Lag Model (SLM):")
        try:
            slm = ML_Lag(y_arr, X_arr, w=w, name_y=spec['y'], 
                        name_x=list(X_df.columns), name_w='w')
            
            # Extract coefficients
            try:
                rho = float(np.asarray(slm.rho).flatten()[0])
            except:
                rho = float(slm.rho)
            
            betas = np.asarray(slm.betas).flatten()
            std_errs = np.asarray(slm.std_err).flatten()
            z_stats = np.asarray(slm.z_stat).flatten()
            p_values = 2 * (1 - stats.norm.cdf(np.abs(z_stats)))
            
            print(f"  ρ (spatial lag): {rho:.4f} (p={p_values[-1]:.4f})")
            print(f"  Flood coefficient: {betas[1]:.4f} (p={p_values[1]:.4f})")
            print(f"  Waste coefficient: {betas[2]:.4f} (p={p_values[2]:.4f})")
            
            # Pseudo R²
            try:
                y_obs = np.asarray(y).flatten()
                yhat = np.asarray(slm.predy).flatten()
                ssr = np.sum((y_obs - yhat)**2)
                sst = np.sum((y_obs - y_obs.mean())**2)
                pseudo_r2 = 1 - ssr / sst
                print(f"  Pseudo R²: {pseudo_r2:.4f}")
            except:
                pass
                
        except Exception as e:
            print(f"  ✗ SLM failed: {e}")
        
        # Spatial Error Model (SEM)
        print("\nSpatial Error Model (SEM):")
        try:
            sem = ML_Error(y_arr, X_arr, w=w, name_y=spec['y'],
                          name_x=list(X_df.columns), name_w='w')
            
            try:
                lam = float(np.asarray(sem.lam).flatten()[0])
            except:
                lam = float(sem.lam)
            
            betas = np.asarray(sem.betas).flatten()
            std_errs = np.asarray(sem.std_err).flatten()
            z_stats = np.asarray(sem.z_stat).flatten()
            p_values = 2 * (1 - stats.norm.cdf(np.abs(z_stats)))
            
            print(f"  λ (spatial error): {lam:.4f} (p={p_values[-1]:.4f})")
            print(f"  Flood coefficient: {betas[1]:.4f} (p={p_values[1]:.4f})")
            print(f"  Waste coefficient: {betas[2]:.4f} (p={p_values[2]:.4f})")
            
        except Exception as e:
            print(f"  ✗ SEM failed: {e}")
    
    # ========================================================================
    # STORE RESULTS SUMMARY
    # ========================================================================
    results_summary.append({
        'specification': spec['name'],
        'y_var': spec['y'],
        'flood_var': spec['flood'],
        'waste_var': spec['waste'],
        'n_obs': len(model_gdf),
        'r2': ols.rsquared,
        'adj_r2': ols.rsquared_adj,
        'flood_coef': ols.params[spec['flood']],
        'flood_pval': ols.pvalues[spec['flood']],
        'waste_coef': ols.params[spec['waste']],
        'waste_pval': ols.pvalues[spec['waste']],
        'moran_i': mi.I if mi else np.nan,
        'moran_p': mi.p_sim if mi else np.nan,
    })

# ============================================================================
# SUMMARY TABLE
# ============================================================================
print("\n" + "="*80)
print("SUMMARY: ALL SPECIFICATIONS")
print("="*80)

summary_df = pd.DataFrame(results_summary)

if len(summary_df) > 0:
    print("\n" + summary_df.to_string(index=False))
    
    print("\n" + "-"*80)
    print("KEY FINDINGS:")
    print("-"*80)
    
    for _, row in summary_df.iterrows():
        print(f"\n{row['specification']}:")
        print(f"  Waste effect: {row['waste_coef']:.4f} (p={row['waste_pval']:.4f})")
        print(f"  Flood effect: {row['flood_coef']:.4f} (p={row['flood_pval']:.4f})")
        print(f"  R²: {row['adj_r2']:.4f}")
        if not pd.isna(row['moran_i']):
            print(f"  Moran's I: {row['moran_i']:.4f} (p={row['moran_p']:.4f})")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
