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

def load_moran_results(file_path):
    """Load Moran's I results."""
    if not file_path.exists():
        return None
    try:
        df = pd.read_csv(file_path)
        moran_dict = {}
        if 'statistic' in df.columns and 'value' in df.columns:
            for _, row in df.iterrows():
                moran_dict[row['statistic']] = row['value']
        return moran_dict
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

# Load coefficients
wasteflood_ols_coefs = load_coefficients(WASTEFLOOD_DIR / "ols_coefficients.csv")
wastefloodpop_ols_coefs = None

# Find wastefloodpop coefficients files (with Facebook baseline - fb_baseline)
wastefloodpop_ols_files = [f for f in WASTEFLOODPOP_DIR.glob("ols_coefficients_*.csv") if 'fb_baseline' in f.name]
wastefloodpop_slm_files = [f for f in WASTEFLOODPOP_DIR.glob("slm_coefficients_*.csv") if 'fb_baseline' in f.name]

wastefloodpop_ols_coefs = None
wastefloodpop_slm_coefs = None

if wastefloodpop_ols_files:
    wastefloodpop_ols_coefs = load_coefficients(wastefloodpop_ols_files[0])
    print(f"Found wastefloodpop OLS coefficients: {wastefloodpop_ols_files[0].name}")
else:
    print("Warning: No wastefloodpop OLS coefficients found (with fb_baseline)")

if wastefloodpop_slm_files:
    wastefloodpop_slm_coefs = load_coefficients(wastefloodpop_slm_files[0])
    print(f"Found wastefloodpop SLM coefficients: {wastefloodpop_slm_files[0].name}")
else:
    print("Warning: No wastefloodpop SLM coefficients found (with fb_baseline)")

# Load statistics
wasteflood_moran = load_moran_results(WASTEFLOOD_DIR / "moran_i_results.csv")
wastefloodpop_moran = None

# Find wastefloodpop Moran's I files (with fb_baseline)
wastefloodpop_moran_ols_files = [f for f in WASTEFLOODPOP_DIR.glob("moran_i_results_*.csv") if 'fb_baseline' in f.name]
wastefloodpop_moran_slm_files = [f for f in WASTEFLOODPOP_DIR.glob("slm_moran_i_results_*.csv") if 'fb_baseline' in f.name]

wastefloodpop_moran_ols = None
wastefloodpop_moran_slm = None

if wastefloodpop_moran_ols_files:
    wastefloodpop_moran_ols = load_moran_results(wastefloodpop_moran_ols_files[0])
    print(f"Found wastefloodpop OLS Moran's I: {wastefloodpop_moran_ols_files[0].name}")

if wastefloodpop_moran_slm_files:
    wastefloodpop_moran_slm = load_moran_results(wastefloodpop_moran_slm_files[0])
    print(f"Found wastefloodpop SLM Moran's I: {wastefloodpop_moran_slm_files[0].name}")

wasteflood_comp = load_model_comparison(WASTEFLOOD_DIR / "model_comparison.csv")
wastefloodpop_comp = None

# Find wastefloodpop model comparison (with fb_baseline)
wastefloodpop_comp_files = [f for f in WASTEFLOODPOP_DIR.glob("model_comparison_*.csv") if 'fb_baseline' in f.name]
if wastefloodpop_comp_files:
    wastefloodpop_comp = load_model_comparison(wastefloodpop_comp_files[0])
    print(f"Found wastefloodpop model comparison: {wastefloodpop_comp_files[0].name}")

# Load pseudo R² for SLM
wastefloodpop_slm_pseudo_r2_files = [f for f in WASTEFLOODPOP_DIR.glob("slm_pseudo_r2_*.csv") if 'fb_baseline' in f.name]
wastefloodpop_slm_pseudo_r2 = None
if wastefloodpop_slm_pseudo_r2_files:
    try:
        df = pd.read_csv(wastefloodpop_slm_pseudo_r2_files[0])
        if 'pseudo_r2' in df.columns:
            wastefloodpop_slm_pseudo_r2 = float(df['pseudo_r2'].iloc[0])
            print(f"Found wastefloodpop SLM pseudo R²: {wastefloodpop_slm_pseudo_r2:.4f}")
    except:
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

# Extract OLS coefficients (Flood + Waste + Population)
flood_fwp_ols_coef, flood_fwp_ols_p = get_coef_from_df(wastefloodpop_ols_coefs, 'flood') if wastefloodpop_ols_coefs is not None else (None, None)
waste_fwp_ols_coef, waste_fwp_ols_p = get_coef_from_df(wastefloodpop_ols_coefs, 'waste') if wastefloodpop_ols_coefs is not None else (None, None)
pop_fwp_ols_coef, pop_fwp_ols_p = get_coef_from_df(wastefloodpop_ols_coefs, 'baseline') if wastefloodpop_ols_coefs is not None else (None, None)

# Extract SLM coefficients (Flood + Waste + Population)
flood_fwp_slm_coef, flood_fwp_slm_p = get_coef_from_df(wastefloodpop_slm_coefs, 'flood') if wastefloodpop_slm_coefs is not None else (None, None)
waste_fwp_slm_coef, waste_fwp_slm_p = get_coef_from_df(wastefloodpop_slm_coefs, 'waste') if wastefloodpop_slm_coefs is not None else (None, None)
pop_fwp_slm_coef, pop_fwp_slm_p = get_coef_from_df(wastefloodpop_slm_coefs, 'baseline') if wastefloodpop_slm_coefs is not None else (None, None)

# Get Adj R² / Pseudo R²
adj_r2_fw = wasteflood_comp.get('OLS') if wasteflood_comp else None
adj_r2_fwp_ols = wastefloodpop_comp.get('OLS') if wastefloodpop_comp else None
adj_r2_fwp_slm = wastefloodpop_slm_pseudo_r2 if wastefloodpop_slm_pseudo_r2 is not None else (wastefloodpop_comp.get('SLM') if wastefloodpop_comp else None)

# -------------------- Build table --------------------
table_data = {
    'Variable': [
        'Flood exposure',
        'Waste accumulation',
        'Baseline population',
        'Adj. R²',
        "Moran's I (resid.)"
    ],
    'Flood + Waste (OLS)': [
        format_coef_significance(flood_fw_coef, flood_fw_p) if flood_fw_coef is not None else "—",
        format_coef_significance(waste_fw_coef, waste_fw_p) if waste_fw_coef is not None else "—",
        "—",
        f"{adj_r2_fw:.3f}" if adj_r2_fw is not None else "—",
        format_moran_i_stars(wasteflood_moran)
    ],
    'Flood + Waste + Population (OLS)': [
        format_coef_significance(flood_fwp_ols_coef, flood_fwp_ols_p) if flood_fwp_ols_coef is not None else "—",
        format_coef_significance(waste_fwp_ols_coef, waste_fwp_ols_p) if waste_fwp_ols_coef is not None else "—",
        format_coef_significance(pop_fwp_ols_coef, pop_fwp_ols_p) if pop_fwp_ols_coef is not None else "—",
        f"{adj_r2_fwp_ols:.3f}" if adj_r2_fwp_ols is not None else "—",
        format_moran_i_stars(wastefloodpop_moran_ols) if wastefloodpop_moran_ols else "—"
    ],
    'Flood + Waste + Population (SLM)': [
        format_coef_significance(flood_fwp_slm_coef, flood_fwp_slm_p) if flood_fwp_slm_coef is not None else "—",
        format_coef_significance(waste_fwp_slm_coef, waste_fwp_slm_p) if waste_fwp_slm_coef is not None else "—",
        format_coef_significance(pop_fwp_slm_coef, pop_fwp_slm_p) if pop_fwp_slm_coef is not None else "—",
        f"{adj_r2_fwp_slm:.3f}" if adj_r2_fwp_slm is not None else "—",
        format_moran_i_stars(wastefloodpop_moran_slm) if wastefloodpop_moran_slm else "—"
    ]
}

# Create DataFrame
table_s3 = pd.DataFrame(table_data)

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
