#!/usr/bin/env python3
"""
create_table_s3.py

Creates Table S3: Controlling for baseline population (Exposure robustness)

Compares Flood + Waste OLS vs Flood + Waste + Population OLS to show that
waste is not just a proxy for population density.

Requirements:
  pandas, numpy, pathlib
"""
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import pandas as pd
import numpy as np

# -------------------- USER PARAMETERS --------------------
OUTPUT_BASE = Path("/Users/wenlanzhang/PycharmProjects/Waste_Flood_out/Output")
SUMMARY_OUT_DIR = Path("/Users/wenlanzhang/PycharmProjects/Waste_Flood_out/Output/summary")
SUMMARY_OUT_DIR.mkdir(parents=True, exist_ok=True)

# Model directories
WASTEFLOOD_DIR = OUTPUT_BASE / "wasteflood"
WASTEFLOODPOP_DIR = OUTPUT_BASE / "wastefloodpop"
FLOODPOP_DIR = OUTPUT_BASE / "floodpop"

# -------------------- Helper functions --------------------
def load_coefficients(file_path):
    """Load coefficient file."""
    if not file_path.exists():
        return None
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def load_moran_results(file_path, model_filter=None):
    """Load Moran's I results. If model_filter is set and CSV has 'model' column, use only that model's rows."""
    if not file_path.exists():
        return None
    try:
        df = pd.read_csv(file_path)
        if model_filter is not None and 'model' in df.columns:
            df = df[df['model'] == model_filter]
        moran_dict = {}
        if 'statistic' in df.columns and 'value' in df.columns:
            for _, row in df.iterrows():
                moran_dict[row['statistic']] = row['value']
        return moran_dict if moran_dict else None
    except Exception as e:
        print(f"Error loading Moran's I from {file_path}: {e}")
        return None

def load_model_comparison(file_path):
    """Load model comparison file."""
    if not file_path.exists():
        return None
    try:
        df = pd.read_csv(file_path)
        results = {}
        if 'Model' in df.columns and 'Adj. R²' in df.columns:
            for _, row in df.iterrows():
                results[row['Model']] = row['Adj. R²']
        return results
    except Exception as e:
        print(f"Error loading model comparison from {file_path}: {e}")
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

def format_coef_significance(coef, pval):
    """
    Format coefficient sign with significance stars.
    Shows + / - ONLY if statistically significant, otherwise shows 0.
    """
    if pd.isna(coef) or pd.isna(pval):
        return "—"
    if pval < 0.05:
        sign = "+" if coef > 0 else "-"
        return f"{sign}{get_significance_stars(pval)}"
    return "0"

def format_moran_i_stars(moran_dict):
    """Format Moran's I as just significance stars."""
    if not moran_dict:
        return "—"
    p_val = moran_dict.get("p-value", None)
    if p_val is None or pd.isna(p_val):
        return "—"
    try:
        p_val = float(p_val)
        return get_significance_stars(p_val)
    except:
        return "—"

# -------------------- Load data --------------------
print("="*80)
print("Creating Table S3: Controlling for baseline population")
print("="*80)

# Load Flood + Waste coefficients (consolidated: coefficients.csv with model column)
wasteflood_ols_coefs = None
wasteflood_slm_coefs = None
if (WASTEFLOOD_DIR / "coefficients.csv").exists():
    try:
        _wf = pd.read_csv(WASTEFLOOD_DIR / "coefficients.csv")
        if "model" in _wf.columns:
            _ols = _wf[_wf["model"] == "OLS"].drop(columns=["model"], errors="ignore")
            _slm = _wf[_wf["model"] == "SLM"].drop(columns=["model"], errors="ignore")
            wasteflood_ols_coefs = _ols if len(_ols) > 0 else None
            wasteflood_slm_coefs = _slm if len(_slm) > 0 else None
    except Exception:
        pass
if wasteflood_ols_coefs is None:
    wasteflood_ols_coefs = load_coefficients(WASTEFLOOD_DIR / "ols_coefficients.csv")  # fallback
if wasteflood_ols_coefs is not None:
    print("Found Flood + Waste OLS coefficients (coefficients.csv)")
if wasteflood_slm_coefs is not None:
    print("Found Flood + Waste SLM coefficients (coefficients.csv)")

# Load Flood + Waste + Population coefficients (consolidated: coefficients_*.csv; prefer fb_baseline)
wastefloodpop_coef_files = [f for f in WASTEFLOODPOP_DIR.glob("coefficients_*.csv") if 'fb_baseline' in f.name]
if not wastefloodpop_coef_files:
    wastefloodpop_coef_files = list(WASTEFLOODPOP_DIR.glob("coefficients_*.csv"))
wastefloodpop_ols_coefs = None
wastefloodpop_slm_coefs = None
if wastefloodpop_coef_files:
    try:
        _wfp = pd.read_csv(wastefloodpop_coef_files[0])
        if "model" in _wfp.columns:
            _ols = _wfp[_wfp["model"] == "OLS"].drop(columns=["model"], errors="ignore")
            _slm = _wfp[_wfp["model"] == "SLM"].drop(columns=["model"], errors="ignore")
            wastefloodpop_ols_coefs = _ols if len(_ols) > 0 else None
            wastefloodpop_slm_coefs = _slm if len(_slm) > 0 else None
        if wastefloodpop_ols_coefs is not None:
            print(f"Found Flood + Waste + Population coefficients: {wastefloodpop_coef_files[0].name}")
    except Exception:
        pass

# Load Moran's I (consolidated files have model column; filter to OLS or SLM as needed)
wasteflood_moran = load_moran_results(WASTEFLOOD_DIR / "moran_i_results.csv", model_filter="OLS")
wasteflood_moran_slm = load_moran_results(WASTEFLOOD_DIR / "moran_i_results.csv", model_filter="SLM")
if wasteflood_moran is None:
    wasteflood_moran = load_moran_results(WASTEFLOOD_DIR / "moran_i_results.csv")  # no model column

wastefloodpop_moran_files = [f for f in WASTEFLOODPOP_DIR.glob("moran_i_results_*.csv") if 'fb_baseline' in f.name]
if not wastefloodpop_moran_files:
    wastefloodpop_moran_files = list(WASTEFLOODPOP_DIR.glob("moran_i_results_*.csv"))
wastefloodpop_moran_ols = None
wastefloodpop_moran_slm = None
if wastefloodpop_moran_files:
    wastefloodpop_moran_ols = load_moran_results(wastefloodpop_moran_files[0], model_filter="OLS")
    wastefloodpop_moran_slm = load_moran_results(wastefloodpop_moran_files[0], model_filter="SLM")
    if wastefloodpop_moran_ols is None:
        wastefloodpop_moran_ols = load_moran_results(wastefloodpop_moran_files[0])
    if wastefloodpop_moran_slm is None:
        wastefloodpop_moran_slm = load_moran_results(wastefloodpop_moran_files[0])  # legacy single-file
    if wastefloodpop_moran_ols or wastefloodpop_moran_slm:
        print(f"Found Flood + Waste + Population Moran's I: {wastefloodpop_moran_files[0].name}")

wasteflood_comp = load_model_comparison(WASTEFLOOD_DIR / "model_comparison.csv")
# Load Flood + Waste SLM pseudo R²
wasteflood_slm_pseudo_r2 = None
if (WASTEFLOOD_DIR / "slm_metrics.csv").exists():
    try:
        _sm = pd.read_csv(WASTEFLOOD_DIR / "slm_metrics.csv")
        _row = _sm[_sm["metric"] == "pseudo_r2"]
        if len(_row) > 0:
            wasteflood_slm_pseudo_r2 = float(_row["value"].iloc[0])
            print("Found Flood + Waste SLM pseudo R²")
    except Exception:
        pass
wastefloodpop_comp = None
wastefloodpop_comp_files = [f for f in WASTEFLOODPOP_DIR.glob("model_comparison_*.csv") if 'fb_baseline' in f.name]
if not wastefloodpop_comp_files:
    wastefloodpop_comp_files = list(WASTEFLOODPOP_DIR.glob("model_comparison_*.csv"))
if wastefloodpop_comp_files:
    wastefloodpop_comp = load_model_comparison(wastefloodpop_comp_files[0])
    if wastefloodpop_comp:
        print(f"Found Flood + Waste + Population model comparison: {wastefloodpop_comp_files[0].name}")

# Load pseudo R² for SLM (consolidated: slm_metrics_*.csv; fallback slm_pseudo_r2_*.csv)
wastefloodpop_slm_pseudo_r2 = None
slm_metrics_files = [f for f in WASTEFLOODPOP_DIR.glob("slm_metrics_*.csv") if 'fb_baseline' in f.name]
if not slm_metrics_files:
    slm_metrics_files = list(WASTEFLOODPOP_DIR.glob("slm_metrics_*.csv"))
if slm_metrics_files:
    try:
        _sm = pd.read_csv(slm_metrics_files[0])
        _row = _sm[_sm["metric"] == "pseudo_r2"]
        if len(_row) > 0:
            wastefloodpop_slm_pseudo_r2 = float(_row["value"].iloc[0])
            print(f"Found Flood + Waste + Population SLM pseudo R²: {wastefloodpop_slm_pseudo_r2:.4f}")
    except Exception:
        pass
if wastefloodpop_slm_pseudo_r2 is None:
    pseudo_files = list(WASTEFLOODPOP_DIR.glob("slm_pseudo_r2_*.csv"))
    if pseudo_files:
        try:
            df = pd.read_csv(pseudo_files[0])
            if 'pseudo_r2' in df.columns:
                wastefloodpop_slm_pseudo_r2 = float(df['pseudo_r2'].iloc[0])
        except Exception:
            pass

# Load Flood + Population (no waste) coefficients and stats
floodpop_ols_coefs = None
floodpop_slm_coefs = None
floodpop_moran_ols = None
floodpop_moran_slm = None
floodpop_comp = None
floodpop_slm_pseudo_r2 = None
if FLOODPOP_DIR.exists() and (FLOODPOP_DIR / "coefficients.csv").exists():
    try:
        _fp = pd.read_csv(FLOODPOP_DIR / "coefficients.csv")
        if "model" in _fp.columns:
            _ols = _fp[_fp["model"] == "OLS"].drop(columns=["model"], errors="ignore")
            _slm = _fp[_fp["model"] == "SLM"].drop(columns=["model"], errors="ignore")
            floodpop_ols_coefs = _ols if len(_ols) > 0 else None
            floodpop_slm_coefs = _slm if len(_slm) > 0 else None
        if floodpop_ols_coefs is not None:
            print("Found Flood + Population coefficients (coefficients.csv)")
    except Exception:
        pass
if FLOODPOP_DIR.exists() and (FLOODPOP_DIR / "moran_i_results.csv").exists():
    floodpop_moran_ols = load_moran_results(FLOODPOP_DIR / "moran_i_results.csv", model_filter="OLS")
    floodpop_moran_slm = load_moran_results(FLOODPOP_DIR / "moran_i_results.csv", model_filter="SLM")
if FLOODPOP_DIR.exists() and (FLOODPOP_DIR / "model_comparison.csv").exists():
    floodpop_comp = load_model_comparison(FLOODPOP_DIR / "model_comparison.csv")
if FLOODPOP_DIR.exists() and (FLOODPOP_DIR / "slm_metrics.csv").exists():
    try:
        _sm = pd.read_csv(FLOODPOP_DIR / "slm_metrics.csv")
        _row = _sm[_sm["metric"] == "pseudo_r2"]
        if len(_row) > 0:
            floodpop_slm_pseudo_r2 = float(_row["value"].iloc[0])
    except Exception:
        pass

# -------------------- Extract coefficients --------------------
def get_coef_from_df(df, var_name):
    """Extract coefficient and p-value for a variable."""
    if df is None:
        return None, None
    subset = df[df['variable'].str.contains(var_name, case=False, na=False)]
    if len(subset) > 0:
        row = subset.iloc[0]
        coef = row['coefficient']
        pval = row.get('p_value', np.nan)
        return coef, pval
    return None, None

# Extract coefficients
flood_fw_coef, flood_fw_p = get_coef_from_df(wasteflood_ols_coefs, 'flood')
waste_fw_coef, waste_fw_p = get_coef_from_df(wasteflood_ols_coefs, 'waste')
# Extract Flood + Waste SLM coefficients
flood_fw_slm_coef, flood_fw_slm_p = get_coef_from_df(wasteflood_slm_coefs, 'flood') if wasteflood_slm_coefs is not None else (None, None)
waste_fw_slm_coef, waste_fw_slm_p = get_coef_from_df(wasteflood_slm_coefs, 'waste') if wasteflood_slm_coefs is not None else (None, None)

# Extract OLS coefficients (Flood + Waste + Population)
flood_fwp_ols_coef, flood_fwp_ols_p = get_coef_from_df(wastefloodpop_ols_coefs, 'flood') if wastefloodpop_ols_coefs is not None else (None, None)
waste_fwp_ols_coef, waste_fwp_ols_p = get_coef_from_df(wastefloodpop_ols_coefs, 'waste') if wastefloodpop_ols_coefs is not None else (None, None)
pop_fwp_ols_coef, pop_fwp_ols_p = get_coef_from_df(wastefloodpop_ols_coefs, 'baseline') if wastefloodpop_ols_coefs is not None else (None, None)
if (pop_fwp_ols_coef, pop_fwp_ols_p) == (None, None) and wastefloodpop_ols_coefs is not None:
    pop_fwp_ols_coef, pop_fwp_ols_p = get_coef_from_df(wastefloodpop_ols_coefs, 'worldpop')  # e.g. 3_worldpop

# Extract SLM coefficients (Flood + Waste + Population)
flood_fwp_slm_coef, flood_fwp_slm_p = get_coef_from_df(wastefloodpop_slm_coefs, 'flood') if wastefloodpop_slm_coefs is not None else (None, None)
waste_fwp_slm_coef, waste_fwp_slm_p = get_coef_from_df(wastefloodpop_slm_coefs, 'waste') if wastefloodpop_slm_coefs is not None else (None, None)
pop_fwp_slm_coef, pop_fwp_slm_p = get_coef_from_df(wastefloodpop_slm_coefs, 'baseline') if wastefloodpop_slm_coefs is not None else (None, None)
if (pop_fwp_slm_coef, pop_fwp_slm_p) == (None, None) and wastefloodpop_slm_coefs is not None:
    pop_fwp_slm_coef, pop_fwp_slm_p = get_coef_from_df(wastefloodpop_slm_coefs, 'worldpop')

# Extract Flood + Population coefficients (no waste)
flood_fp_ols_coef, flood_fp_ols_p = get_coef_from_df(floodpop_ols_coefs, 'flood') if floodpop_ols_coefs is not None else (None, None)
pop_fp_ols_coef, pop_fp_ols_p = get_coef_from_df(floodpop_ols_coefs, 'baseline') if floodpop_ols_coefs is not None else (None, None)
if (pop_fp_ols_coef, pop_fp_ols_p) == (None, None) and floodpop_ols_coefs is not None:
    pop_fp_ols_coef, pop_fp_ols_p = get_coef_from_df(floodpop_ols_coefs, 'worldpop')
flood_fp_slm_coef, flood_fp_slm_p = get_coef_from_df(floodpop_slm_coefs, 'flood') if floodpop_slm_coefs is not None else (None, None)
pop_fp_slm_coef, pop_fp_slm_p = get_coef_from_df(floodpop_slm_coefs, 'baseline') if floodpop_slm_coefs is not None else (None, None)
if (pop_fp_slm_coef, pop_fp_slm_p) == (None, None) and floodpop_slm_coefs is not None:
    pop_fp_slm_coef, pop_fp_slm_p = get_coef_from_df(floodpop_slm_coefs, 'worldpop')

# Get Adj R² / Pseudo R²
adj_r2_fw = wasteflood_comp.get('OLS') if wasteflood_comp else None
adj_r2_fw_slm = wasteflood_slm_pseudo_r2 if wasteflood_slm_pseudo_r2 is not None else (wasteflood_comp.get('SLM') if wasteflood_comp else None)
adj_r2_fwp_ols = wastefloodpop_comp.get('OLS') if wastefloodpop_comp else None
adj_r2_fwp_slm = wastefloodpop_slm_pseudo_r2 if wastefloodpop_slm_pseudo_r2 is not None else (wastefloodpop_comp.get('SLM') if wastefloodpop_comp else None)
adj_r2_fp_ols = floodpop_comp.get('OLS') if floodpop_comp else None
adj_r2_fp_slm = floodpop_slm_pseudo_r2 if floodpop_slm_pseudo_r2 is not None else (floodpop_comp.get('SLM') if floodpop_comp else None)

# -------------------- Build table --------------------
# Column headers are specification names only; OLS/SLM shown in a "Model" row.
# Order: OLS columns first (Flood + Waste, Flood + Population, Flood + Waste + Population), then SLM in same order.
table_data = {
    'Variable': [
        'Model',
        'Flood exposure',
        'Waste accumulation',
        'Baseline population',
        'Adj. R²',
        "Moran's I (resid.)"
    ],
    'Flood + Waste': [
        'OLS',
        format_coef_significance(flood_fw_coef, flood_fw_p) if flood_fw_coef is not None else "—",
        format_coef_significance(waste_fw_coef, waste_fw_p) if waste_fw_coef is not None else "—",
        "—",
        f"{adj_r2_fw:.3f}" if adj_r2_fw is not None else "—",
        format_moran_i_stars(wasteflood_moran)
    ],
    'Flood + Population': [
        'OLS',
        format_coef_significance(flood_fp_ols_coef, flood_fp_ols_p) if flood_fp_ols_coef is not None else "—",
        "—",
        format_coef_significance(pop_fp_ols_coef, pop_fp_ols_p) if pop_fp_ols_coef is not None else "—",
        f"{adj_r2_fp_ols:.3f}" if adj_r2_fp_ols is not None else "—",
        format_moran_i_stars(floodpop_moran_ols) if floodpop_moran_ols else "—"
    ],
    'Flood + Waste + Population': [
        'OLS',
        format_coef_significance(flood_fwp_ols_coef, flood_fwp_ols_p) if flood_fwp_ols_coef is not None else "—",
        format_coef_significance(waste_fwp_ols_coef, waste_fwp_ols_p) if waste_fwp_ols_coef is not None else "—",
        format_coef_significance(pop_fwp_ols_coef, pop_fwp_ols_p) if pop_fwp_ols_coef is not None else "—",
        f"{adj_r2_fwp_ols:.3f}" if adj_r2_fwp_ols is not None else "—",
        format_moran_i_stars(wastefloodpop_moran_ols) if wastefloodpop_moran_ols else "—"
    ],
    'Flood + Waste.1': [
        'SLM',
        format_coef_significance(flood_fw_slm_coef, flood_fw_slm_p) if flood_fw_slm_coef is not None else "—",
        format_coef_significance(waste_fw_slm_coef, waste_fw_slm_p) if waste_fw_slm_coef is not None else "—",
        "—",
        f"{adj_r2_fw_slm:.3f}" if adj_r2_fw_slm is not None else "—",
        format_moran_i_stars(wasteflood_moran_slm) if wasteflood_moran_slm else "—"
    ],
    'Flood + Population.1': [
        'SLM',
        format_coef_significance(flood_fp_slm_coef, flood_fp_slm_p) if flood_fp_slm_coef is not None else "—",
        "—",
        format_coef_significance(pop_fp_slm_coef, pop_fp_slm_p) if pop_fp_slm_coef is not None else "—",
        f"{adj_r2_fp_slm:.3f}" if adj_r2_fp_slm is not None else "—",
        format_moran_i_stars(floodpop_moran_slm) if floodpop_moran_slm else "—"
    ],
    'Flood + Waste + Population.1': [
        'SLM',
        format_coef_significance(flood_fwp_slm_coef, flood_fwp_slm_p) if flood_fwp_slm_coef is not None else "—",
        format_coef_significance(waste_fwp_slm_coef, waste_fwp_slm_p) if waste_fwp_slm_coef is not None else "—",
        format_coef_significance(pop_fwp_slm_coef, pop_fwp_slm_p) if pop_fwp_slm_coef is not None else "—",
        f"{adj_r2_fwp_slm:.3f}" if adj_r2_fwp_slm is not None else "—",
        format_moran_i_stars(wastefloodpop_moran_slm) if wastefloodpop_moran_slm else "—"
    ]
}

# Create DataFrame
table_s3 = pd.DataFrame(table_data)
# Column headers: OLS block then SLM block (same spec order)
table_s3.columns = [
    'Variable',
    'Flood + Waste',
    'Flood + Population',
    'Flood + Waste + Population',
    'Flood + Waste',
    'Flood + Population',
    'Flood + Waste + Population'
]

# Save table
output_file = SUMMARY_OUT_DIR / "Table_S3_controlling_baseline_population.csv"
table_s3.to_csv(output_file, index=False)

print("\n" + "="*80)
print("TABLE S3 | Controlling for baseline population")
print("="*80)
print(table_s3.to_string(index=False))
print("="*80)

print(f"\nTable S3 saved to: {output_file}")
print("\nDone.")
