#!/usr/bin/env python3
"""
create_table_1.py

Creates Table 1: Spatial lag model impacts (Mechanism & spillovers)

This table shows coefficients and statistics for:
- Flood-only OLS
- Flood + Waste OLS  
- Flood + Waste SLM

Requirements:
  pandas, numpy, pathlib
"""
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import pandas as pd
import numpy as np
import re

# -------------------- USER PARAMETERS --------------------
OUTPUT_BASE = Path("/Users/wenlanzhang/PycharmProjects/Waste_Flood_out/Output")
SUMMARY_OUT_DIR = Path("/Users/wenlanzhang/PycharmProjects/Waste_Flood_out/Output/summary")
SUMMARY_OUT_DIR.mkdir(parents=True, exist_ok=True)

# Model directories
FLOOD_DIR = OUTPUT_BASE / "flood"
WASTEFLOOD_DIR = OUTPUT_BASE / "wasteflood"

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
        if 'Model' in df.columns and 'AIC' in df.columns:
            for _, row in df.iterrows():
                results[f"{row['Model']}_AIC"] = row['AIC']
        return results
    except Exception as e:
        print(f"Error loading model comparison from {file_path}: {e}")
        return None

def load_pseudo_r2(file_path):
    """Load pseudo R² from file."""
    if not file_path.exists():
        return None
    try:
        df = pd.read_csv(file_path)
        if 'pseudo_r2' in df.columns:
            return float(df['pseudo_r2'].iloc[0])
        return None
    except Exception as e:
        print(f"Error loading pseudo R² from {file_path}: {e}")
        return None

def parse_adj_r2_from_summary(summary_file):
    """Parse Adj R² from OLS summary text file."""
    if not summary_file.exists():
        return None
    try:
        with open(summary_file, 'r') as f:
            content = f.read()
            match = re.search(r'Adj\. R-squared:\s+([-\d.]+)', content)
            if match:
                return float(match.group(1))
    except:
        pass
    return None

def parse_nobs_from_summary(summary_file):
    """Parse number of observations from OLS summary text file."""
    if not summary_file.exists():
        return None
    try:
        with open(summary_file, 'r') as f:
            content = f.read()
            match = re.search(r'No\. Observations:\s+(\d+)', content)
            if match:
                return int(match.group(1))
    except:
        pass
    return None

def parse_aic_from_summary(summary_file):
    """Parse AIC from OLS summary text file."""
    if not summary_file.exists():
        return None
    try:
        with open(summary_file, 'r') as f:
            content = f.read()
            match = re.search(r'AIC:\s+([\d.]+)', content)
            if match:
                return float(match.group(1))
    except:
        pass
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

def format_coefficient(coef, se, pval):
    """Format coefficient with SE in parentheses and significance stars."""
    if pd.isna(coef) or pd.isna(se):
        return "-"
    stars = get_significance_stars(pval)
    return f"{coef:.2f}{stars} ({se:.2f})"

def format_moran_i(moran_dict, include_ns=True):
    """Format Moran's I with significance stars."""
    if not moran_dict:
        return "-"
    i_val = moran_dict.get("Moran's I", None)
    p_val = moran_dict.get("p-value", None)
    if i_val is None or pd.isna(i_val):
        return "-"
    try:
        i_val = float(i_val)
        if p_val is not None and not pd.isna(p_val):
            p_val = float(p_val)
            stars = get_significance_stars(p_val)
            if stars == "" and include_ns:
                return f"{i_val:.4f} (n.s.)"
            return f"{i_val:.4f}{stars}"
        return f"{i_val:.4f}"
    except:
        return str(i_val)

# -------------------- Load data --------------------
print("="*80)
print("Creating Table 1: Spatial lag model impacts")
print("="*80)

# Load coefficients (combined file: filter OLS for flood-only OLS)
flood_ols_coefs = None
if (FLOOD_DIR / "flood_only_coefficients.csv").exists():
    try:
        _flood_coefs = pd.read_csv(FLOOD_DIR / "flood_only_coefficients.csv")
        if "model" in _flood_coefs.columns:
            _ols = _flood_coefs[_flood_coefs["model"] == "OLS"].drop(columns=["model"], errors="ignore")
        else:
            _ols = _flood_coefs
        flood_ols_coefs = _ols if len(_ols) > 0 else None
    except Exception:
        flood_ols_coefs = None
# Wasteflood: combined coefficients.csv (filter OLS and SLM)
wasteflood_ols_coefs = None
wasteflood_slm_coefs = None
if (WASTEFLOOD_DIR / "coefficients.csv").exists():
    try:
        _wf_coefs = pd.read_csv(WASTEFLOOD_DIR / "coefficients.csv")
        if "model" in _wf_coefs.columns:
            _ols = _wf_coefs[_wf_coefs["model"] == "OLS"].drop(columns=["model"], errors="ignore")
            _slm = _wf_coefs[_wf_coefs["model"] == "SLM"].drop(columns=["model"], errors="ignore")
            wasteflood_ols_coefs = _ols if len(_ols) > 0 else None
            wasteflood_slm_coefs = _slm if len(_slm) > 0 else None
    except Exception:
        pass

# Load statistics (combined Moran file: use OLS rows for flood-only)
flood_moran = None
if (FLOOD_DIR / "flood_only_moran_i_results.csv").exists():
    try:
        _flood_moran_df = pd.read_csv(FLOOD_DIR / "flood_only_moran_i_results.csv")
        if len(_flood_moran_df) > 0:
            if "model" in _flood_moran_df.columns:
                _ols = _flood_moran_df[_flood_moran_df["model"] == "OLS"]
                if len(_ols) > 0:
                    flood_moran = dict(zip(_ols["statistic"], _ols["value"]))
            if flood_moran is None and "statistic" in _flood_moran_df.columns and "value" in _flood_moran_df.columns:
                flood_moran = dict(zip(_flood_moran_df["statistic"], _flood_moran_df["value"]))
    except Exception:
        pass
# Wasteflood: combined moran_i_results.csv (filter by model for OLS and SLM)
wasteflood_moran_ols = None
wasteflood_moran_slm = None
if (WASTEFLOOD_DIR / "moran_i_results.csv").exists():
    try:
        _wf_moran = pd.read_csv(WASTEFLOOD_DIR / "moran_i_results.csv")
        if "model" in _wf_moran.columns and "statistic" in _wf_moran.columns and "value" in _wf_moran.columns:
            _ols = _wf_moran[_wf_moran["model"] == "OLS"]
            _slm = _wf_moran[_wf_moran["model"] == "SLM"]
            if len(_ols) > 0:
                wasteflood_moran_ols = dict(zip(_ols["statistic"], _ols["value"]))
            if len(_slm) > 0:
                wasteflood_moran_slm = dict(zip(_slm["statistic"], _slm["value"]))
        else:
            wasteflood_moran_ols = load_moran_results(WASTEFLOOD_DIR / "moran_i_results.csv")
    except Exception:
        pass

wasteflood_comp = load_model_comparison(WASTEFLOOD_DIR / "model_comparison.csv")
# Wasteflood: pseudo R² from slm_metrics.csv (metric/value)
wasteflood_slm_pseudo_r2 = None
if (WASTEFLOOD_DIR / "slm_metrics.csv").exists():
    try:
        _sm = pd.read_csv(WASTEFLOOD_DIR / "slm_metrics.csv")
        _row = _sm[_sm["metric"] == "pseudo_r2"]
        if len(_row) > 0:
            wasteflood_slm_pseudo_r2 = float(_row["value"].iloc[0])
    except Exception:
        pass
if wasteflood_slm_pseudo_r2 is None:
    wasteflood_slm_pseudo_r2 = load_pseudo_r2(WASTEFLOOD_DIR / "slm_pseudo_r2.csv")  # fallback

# Parse from summary files
flood_ols_r2 = parse_adj_r2_from_summary(FLOOD_DIR / "flood_only_ols_summary.txt")
flood_nobs = parse_nobs_from_summary(FLOOD_DIR / "flood_only_ols_summary.txt")
flood_aic = parse_aic_from_summary(FLOOD_DIR / "flood_only_ols_summary.txt")

wasteflood_nobs = parse_nobs_from_summary(WASTEFLOOD_DIR / "ols_summary.txt")

# -------------------- Extract coefficients --------------------
def get_coef_from_df(df, var_name):
    """Extract coefficient, SE, and p-value for a variable."""
    if df is None:
        return None, None, None
    subset = df[df['variable'].str.contains(var_name, case=False, na=False)]
    if len(subset) > 0:
        row = subset.iloc[0]
        coef = row['coefficient']
        se = row.get('std_err', np.nan)
        pval = row.get('p_value', np.nan)
        return coef, se, pval
    return None, None, None

def get_exact_coef_from_df(df, var_name):
    """Extract coefficient for exact variable name match."""
    if df is None:
        return None, None, None
    subset = df[df['variable'] == var_name]
    if len(subset) > 0:
        row = subset.iloc[0]
        coef = row['coefficient']
        se = row.get('std_err', np.nan)
        pval = row.get('p_value', np.nan)
        return coef, se, pval
    return None, None, None

# Flood coefficients
flood_fo_coef, flood_fo_se, flood_fo_p = get_coef_from_df(flood_ols_coefs, 'flood')
flood_fw_ols_coef, flood_fw_ols_se, flood_fw_ols_p = get_coef_from_df(wasteflood_ols_coefs, 'flood')
flood_fw_slm_coef, flood_fw_slm_se, flood_fw_slm_p = get_coef_from_df(wasteflood_slm_coefs, 'flood')

# Waste coefficients
waste_fw_ols_coef, waste_fw_ols_se, waste_fw_ols_p = get_coef_from_df(wasteflood_ols_coefs, 'waste')
waste_fw_slm_coef, waste_fw_slm_se, waste_fw_slm_p = get_coef_from_df(wasteflood_slm_coefs, 'waste')

# Rho (spatial lag)
rho_coef, rho_se, rho_p = get_exact_coef_from_df(wasteflood_slm_coefs, 'rho')

# Constants
const_fo_coef, const_fo_se, const_fo_p = get_exact_coef_from_df(flood_ols_coefs, 'const')
const_fw_ols_coef, const_fw_ols_se, const_fw_ols_p = get_exact_coef_from_df(wasteflood_ols_coefs, 'const')
const_fw_slm_coef, const_fw_slm_se, const_fw_slm_p = get_exact_coef_from_df(wasteflood_slm_coefs, 'const')

# -------------------- Build table --------------------
table_data = {
    'Variable': [
        'Flood exposure (p95)',
        'Waste accumulation',
        'Spatial lag (ρ)',
        'Constant',
        'Observations',
        'Adj. R² / Pseudo R²',
        "Moran's I (residuals)",
        'AIC'
    ],
    'Flood-only OLS': [
        format_coefficient(flood_fo_coef, flood_fo_se, flood_fo_p) if flood_fo_coef is not None else "-",
        "-",
        "-",
        format_coefficient(const_fo_coef, const_fo_se, const_fo_p) if const_fo_coef is not None else "-",
        str(flood_nobs) if flood_nobs else "-",
        f"{flood_ols_r2:.4f}" if flood_ols_r2 is not None else "-",
        format_moran_i(flood_moran),
        f"{flood_aic:.2f}" if flood_aic else "-"
    ],
    'Flood + Waste OLS': [
        format_coefficient(flood_fw_ols_coef, flood_fw_ols_se, flood_fw_ols_p) if flood_fw_ols_coef is not None else "-",
        format_coefficient(waste_fw_ols_coef, waste_fw_ols_se, waste_fw_ols_p) if waste_fw_ols_coef is not None else "-",
        "-",
        format_coefficient(const_fw_ols_coef, const_fw_ols_se, const_fw_ols_p) if const_fw_ols_coef is not None else "-",
        str(wasteflood_nobs) if wasteflood_nobs else "-",
        f"{wasteflood_comp.get('OLS', ''):.4f}" if wasteflood_comp and 'OLS' in wasteflood_comp else "-",
        format_moran_i(wasteflood_moran_ols),
        f"{wasteflood_comp.get('OLS_AIC', ''):.2f}" if wasteflood_comp and 'OLS_AIC' in wasteflood_comp else "-"
    ],
    'Flood + Waste SLM': [
        format_coefficient(flood_fw_slm_coef, flood_fw_slm_se, flood_fw_slm_p) if flood_fw_slm_coef is not None else "-",
        format_coefficient(waste_fw_slm_coef, waste_fw_slm_se, waste_fw_slm_p) if waste_fw_slm_coef is not None else "-",
        format_coefficient(rho_coef, rho_se, rho_p) if rho_coef is not None else "-",
        format_coefficient(const_fw_slm_coef, const_fw_slm_se, const_fw_slm_p) if const_fw_slm_coef is not None else "-",
        str(wasteflood_nobs) if wasteflood_nobs else "-",
        f"{wasteflood_slm_pseudo_r2:.4f}" if wasteflood_slm_pseudo_r2 is not None else (f"{wasteflood_comp.get('SLM', ''):.4f}" if wasteflood_comp and 'SLM' in wasteflood_comp else "-"),
        format_moran_i(wasteflood_moran_slm, include_ns=True),
        f"{wasteflood_comp.get('SLM_AIC', ''):.2f}" if wasteflood_comp and 'SLM_AIC' in wasteflood_comp else "-"
    ]
}

# Create DataFrame
table_1 = pd.DataFrame(table_data)

# Save table
output_file = SUMMARY_OUT_DIR / "Table_1_spatial_lag_impacts.csv"
table_1.to_csv(output_file, index=False)

print("\n" + "="*80)
print("TABLE 1: Spatial lag model impacts (Mechanism & spillovers)")
print("="*80)
print(table_1.to_string(index=False))
print("="*80)

print(f"\nTable 1 saved to: {output_file}")
print("\nDone.")
