#!/usr/bin/env python3
"""
summarize_outputs.py

Post-processing script to create paper-ready tables from model outputs:
1. Merged coefficient table (Table 1)
2. Robustness summary tables (Supplementary Tables)
3. Outputs manifest (metadata)

This script reads outputs from:
- Output/flood/ (flood-only models)
- Output/wasteflood/ (flood + waste models)
- Output/wastefloodpop/ (flood + waste + population models)

Requirements:
  pandas, numpy, pathlib
"""
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
from glob import glob

# -------------------- USER PARAMETERS --------------------
OUTPUT_BASE = Path("/Users/wenlanzhang/PycharmProjects/Waste_Flood_out/Output")
SUMMARY_OUT_DIR = Path("/Users/wenlanzhang/PycharmProjects/Waste_Flood_out/Output/summary")
SUMMARY_OUT_DIR.mkdir(parents=True, exist_ok=True)

# Significance threshold
SIG_THRESHOLD = 0.05

# -------------------- Helper functions --------------------
def load_coefficients(file_path, model_name, model_type="OLS"):
    """
    Load coefficient file and standardize format.
    Returns DataFrame with columns: variable, coefficient, std_err, p_value, model_name, model_type
    """
    if not file_path.exists():
        return None
    
    try:
        df = pd.read_csv(file_path)
        
        # Standardize column names
        if 'z_stat' in df.columns:
            # SLM/SEM files have z_stat, use p_value if available
            if 'p_value' not in df.columns:
                # Compute p-value from z_stat if missing
                from scipy import stats
                df['p_value'] = 2 * (1 - stats.norm.cdf(np.abs(df['z_stat'])))
        elif 'p_value' not in df.columns:
            print(f"Warning: No p_value column in {file_path}")
            df['p_value'] = np.nan
        
        # Ensure required columns exist
        required_cols = ['variable', 'coefficient', 'std_err', 'p_value']
        for col in required_cols:
            if col not in df.columns:
                print(f"Warning: Missing column {col} in {file_path}")
                df[col] = np.nan
        
        # Add metadata
        df['model_name'] = model_name
        df['model_type'] = model_type
        
        # Select and reorder columns
        df = df[['variable', 'coefficient', 'std_err', 'p_value', 'model_name', 'model_type']].copy()
        
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def load_moran_results(file_path):
    """Load Moran's I results and return as dict."""
    if not file_path.exists():
        return None
    try:
        df = pd.read_csv(file_path)
        # Convert to dict with statistic as key
        moran_dict = {}
        if 'statistic' in df.columns and 'value' in df.columns:
            for _, row in df.iterrows():
                moran_dict[row['statistic']] = row['value']
        return moran_dict
    except Exception as e:
        print(f"Error loading Moran's I from {file_path}: {e}")
        return None

def load_model_comparison(file_path):
    """Load model comparison file and extract adj R²."""
    if not file_path.exists():
        return None
    try:
        df = pd.read_csv(file_path)
        # Extract adj R² for each model type
        results = {}
        if 'Model' in df.columns and 'Adj. R²' in df.columns:
            for _, row in df.iterrows():
                results[row['Model']] = row['Adj. R²']
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

def format_coefficient(coef, se, pval, sig_threshold=SIG_THRESHOLD, format_type="combined"):
    """
    Format coefficient for table display.
    format_type: "combined" (coef*** (SE)), "separate" (coef***, SE), "latex" (coef$^{***}$ (SE))
    """
    if pd.isna(coef) or pd.isna(se):
        return "", ""
    
    # Determine significance stars
    stars = ""
    if not pd.isna(pval):
        if pval < 0.001:
            stars = "***"
        elif pval < 0.01:
            stars = "**"
        elif pval < 0.05:
            stars = "*"
    
    if format_type == "separate":
        return f"{coef:.2f}{stars}", f"({se:.2f})"
    elif format_type == "latex":
        star_latex = "$^{***}$" if stars == "***" else ("$^{**}$" if stars == "**" else ("$^{*}$" if stars == "*" else ""))
        return f"{coef:.2f}{star_latex} ({se:.2f})", ""
    else:  # combined
        return f"{coef:.2f}{stars} ({se:.2f})", ""

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

# -------------------- STEP 1: Load all coefficient files --------------------
print("="*80)
print("STEP 1: Loading coefficient files from all models...")
print("="*80)

all_coefs = []

# Flood-only models
flood_dir = OUTPUT_BASE / "flood"
if flood_dir.exists():
    ols_file = flood_dir / "flood_only_ols_coefficients.csv"
    slm_file = flood_dir / "flood_only_slm_coefficients.csv"
    sem_file = flood_dir / "flood_only_sem_coefficients.csv"
    
    df_ols = load_coefficients(ols_file, "Flood-only", "OLS")
    if df_ols is not None:
        all_coefs.append(df_ols)
        print(f"Loaded: {ols_file.name}")
    
    df_slm = load_coefficients(slm_file, "Flood-only", "SLM")
    if df_slm is not None:
        all_coefs.append(df_slm)
        print(f"Loaded: {slm_file.name}")
    
    df_sem = load_coefficients(sem_file, "Flood-only", "SEM")
    if df_sem is not None:
        all_coefs.append(df_sem)
        print(f"Loaded: {sem_file.name}")

# Flood + Waste models
wasteflood_dir = OUTPUT_BASE / "wasteflood"
if wasteflood_dir.exists():
    ols_file = wasteflood_dir / "ols_coefficients.csv"
    slm_file = wasteflood_dir / "slm_coefficients.csv"
    sem_file = wasteflood_dir / "sem_coefficients.csv"
    
    df_ols = load_coefficients(ols_file, "Flood+Waste", "OLS")
    if df_ols is not None:
        all_coefs.append(df_ols)
        print(f"Loaded: {ols_file.name}")
    
    df_slm = load_coefficients(slm_file, "Flood+Waste", "SLM")
    if df_slm is not None:
        all_coefs.append(df_slm)
        print(f"Loaded: {slm_file.name}")
    
    df_sem = load_coefficients(sem_file, "Flood+Waste", "SEM")
    if df_sem is not None:
        all_coefs.append(df_sem)
        print(f"Loaded: {sem_file.name}")

# Flood + Waste + Population models (may have FILE_ID suffixes)
wastefloodpop_dir = OUTPUT_BASE / "wastefloodpop"
if wastefloodpop_dir.exists():
    # Find all coefficient files (they have FILE_ID suffixes)
    ols_files = list(wastefloodpop_dir.glob("ols_coefficients_*.csv"))
    slm_files = list(wastefloodpop_dir.glob("slm_coefficients_*.csv"))
    sem_files = list(wastefloodpop_dir.glob("sem_coefficients_*.csv"))
    
    # For now, use the first one found (or you can process all)
    if ols_files:
        df_ols = load_coefficients(ols_files[0], "Flood+Waste+Pop", "OLS")
        if df_ols is not None:
            all_coefs.append(df_ols)
            print(f"Loaded: {ols_files[0].name}")
    
    if slm_files:
        df_slm = load_coefficients(slm_files[0], "Flood+Waste+Pop", "SLM")
        if df_slm is not None:
            all_coefs.append(df_slm)
            print(f"Loaded: {slm_files[0].name}")
    
    if sem_files:
        df_sem = load_coefficients(sem_files[0], "Flood+Waste+Pop", "SEM")
        if df_sem is not None:
            all_coefs.append(df_sem)
            print(f"Loaded: {sem_files[0].name}")

if not all_coefs:
    raise RuntimeError("No coefficient files found. Run model scripts first.")

# Combine all coefficients
coefs_all = pd.concat(all_coefs, ignore_index=True)
print(f"\nTotal coefficient rows loaded: {len(coefs_all)}")

# -------------------- STEP 2: Create merged coefficient table --------------------
print("\n" + "="*80)
print("STEP 2: Creating merged coefficient table...")
print("="*80)

# Create a wide-format table: Variable x Model columns
# Models: Flood-only OLS, Flood+Waste OLS, Flood+Waste SLM, Flood+Waste+Pop OLS, Flood+Waste+Pop SLM

# Filter to main models for Table 1 (exclude SEM for main table)
main_models = coefs_all[
    ((coefs_all['model_name'] == 'Flood-only') & (coefs_all['model_type'] == 'OLS')) |
    ((coefs_all['model_name'] == 'Flood+Waste') & (coefs_all['model_type'] == 'OLS')) |
    ((coefs_all['model_name'] == 'Flood+Waste') & (coefs_all['model_type'] == 'SLM')) |
    ((coefs_all['model_name'] == 'Flood+Waste+Pop') & (coefs_all['model_type'] == 'OLS')) |
    ((coefs_all['model_name'] == 'Flood+Waste+Pop') & (coefs_all['model_type'] == 'SLM'))
].copy()

# Create model identifier column
main_models['model_id'] = main_models['model_name'] + " " + main_models['model_type']

# Get unique variables and models
variables = sorted(main_models['variable'].unique())
models = sorted(main_models['model_id'].unique())

# Create wide-format table with separate coefficient and SE columns
table_rows = []
for var in variables:
    row = {'Variable': var}
    for model_id in models:
        subset = main_models[
            (main_models['variable'] == var) & 
            (main_models['model_id'] == model_id)
        ]
        if len(subset) > 0:
            coef = subset.iloc[0]['coefficient']
            se = subset.iloc[0]['std_err']
            pval = subset.iloc[0]['p_value']
            coef_str, se_str = format_coefficient(coef, se, pval, format_type="separate")
            row[f"{model_id}_Coef"] = coef_str
            row[f"{model_id}_SE"] = se_str
        else:
            row[f"{model_id}_Coef"] = ""
            row[f"{model_id}_SE"] = ""
    table_rows.append(row)

table_main = pd.DataFrame(table_rows)

# Also create a combined format version (for display)
table_rows_combined = []
for var in variables:
    row = {'Variable': var}
    for model_id in models:
        subset = main_models[
            (main_models['variable'] == var) & 
            (main_models['model_id'] == model_id)
        ]
        if len(subset) > 0:
            coef = subset.iloc[0]['coefficient']
            se = subset.iloc[0]['std_err']
            pval = subset.iloc[0]['p_value']
            coef_str, _ = format_coefficient(coef, se, pval, format_type="combined")
            row[model_id] = coef_str
        else:
            row[model_id] = ""
    table_rows_combined.append(row)

table_main_combined = pd.DataFrame(table_rows_combined)

# Save main table (separate columns version - better for Excel/Google Sheets)
main_table_file = SUMMARY_OUT_DIR / "table_main_regression.csv"
table_main.to_csv(main_table_file, index=False)
print(f"\nMain regression table (separate columns) saved to: {main_table_file}")

# Save combined version (for LaTeX/display)
main_table_combined_file = SUMMARY_OUT_DIR / "table_main_regression_combined.csv"
table_main_combined.to_csv(main_table_combined_file, index=False)
print(f"Main regression table (combined format) saved to: {main_table_combined_file}")

print("\nPreview (combined format):")
print(table_main_combined.head(10).to_string(index=False))

# Also create a long-format version for easier analysis
long_format = main_models[['variable', 'model_id', 'coefficient', 'std_err', 'p_value']].copy()
long_format = long_format.rename(columns={
    'variable': 'Variable',
    'model_id': 'Model',
    'coefficient': 'Coefficient',
    'std_err': 'Std_Err',
    'p_value': 'P_Value'
})
long_format_file = SUMMARY_OUT_DIR / "table_main_regression_long.csv"
long_format.to_csv(long_format_file, index=False)
print(f"\nLong-format table saved to: {long_format_file}")

# -------------------- STEP 3: Create robustness summary tables --------------------
print("\n" + "="*80)
print("STEP 3: Creating robustness summary tables...")
print("="*80)

def create_robustness_summary(coefs_df, moran_dict, model_comp_dict, model_name, model_type):
    """Create a summary row for robustness table."""
    # Extract key coefficients
    summary = {
        'Model': f"{model_name} {model_type}",
        'Flood_Sign': "",
        'Flood_Sig': "",
        'Waste_Sign': "",
        'Waste_Sig': "",
        'Pop_Sign': "",
        'Pop_Sig': "",
        'Adj_R2': "",
        'Moran_I': "",
        'Moran_P': ""
    }
    
    # Get flood coefficient (exclude rho, lambda, const)
    flood_coefs = coefs_df[
        coefs_df['variable'].str.contains('flood', case=False, na=False) &
        ~coefs_df['variable'].isin(['const', 'rho', 'lambda'])
    ]
    if len(flood_coefs) > 0:
        flood_coef = flood_coefs.iloc[0]
        summary['Flood_Sign'] = "+" if flood_coef['coefficient'] > 0 else "-"
        summary['Flood_Sig'] = get_significance_stars(flood_coef['p_value'])
    
    # Get waste coefficient (exclude rho, lambda, const)
    waste_coefs = coefs_df[
        coefs_df['variable'].str.contains('waste', case=False, na=False) &
        ~coefs_df['variable'].isin(['const', 'rho', 'lambda'])
    ]
    if len(waste_coefs) > 0:
        waste_coef = waste_coefs.iloc[0]
        summary['Waste_Sign'] = "+" if waste_coef['coefficient'] > 0 else "-"
        summary['Waste_Sig'] = get_significance_stars(waste_coef['p_value'])
    
    # Get population coefficient (exclude rho, lambda, const)
    pop_coefs = coefs_df[
        (coefs_df['variable'].str.contains('pop|baseline|worldpop|fb_baseline', case=False, na=False)) &
        ~coefs_df['variable'].isin(['const', 'rho', 'lambda'])
    ]
    if len(pop_coefs) > 0:
        pop_coef = pop_coefs.iloc[0]
        summary['Pop_Sign'] = "+" if pop_coef['coefficient'] > 0 else "-"
        summary['Pop_Sig'] = get_significance_stars(pop_coef['p_value'])
    
    # Get adj R²
    if model_comp_dict:
        adj_r2 = model_comp_dict.get(model_type, "")
        summary['Adj_R2'] = f"{adj_r2:.4f}" if isinstance(adj_r2, (int, float)) else str(adj_r2)
    
    # Get Moran's I
    if moran_dict:
        moran_i = moran_dict.get("Moran's I", "")
        moran_p = moran_dict.get("p-value", "")
        summary['Moran_I'] = f"{moran_i:.4f}" if isinstance(moran_i, (int, float)) else str(moran_i)
        summary['Moran_P'] = f"{moran_p:.4g}" if isinstance(moran_p, (int, float)) else str(moran_p)
    
    return summary

# Collect robustness summaries
robustness_rows = []

# Flood-only
flood_coefs = coefs_all[(coefs_all['model_name'] == 'Flood-only')].copy()
if len(flood_coefs) > 0:
    moran_file = flood_dir / "flood_only_moran_i_results.csv"
    moran_dict = load_moran_results(moran_file)
    
    # For flood-only, extract adj R² from OLS summary or pseudo R² files
    comp_dict = {}
    # Try to read OLS summary for adj R² (would need parsing, but for now use pseudo R² for SLM)
    slm_pseudo_r2_file = flood_dir / "flood_only_slm_pseudo_r2.csv"
    slm_pseudo_r2 = load_pseudo_r2(slm_pseudo_r2_file)
    if slm_pseudo_r2:
        comp_dict['SLM'] = slm_pseudo_r2
    
    for model_type in ['OLS', 'SLM', 'SEM']:
        subset = flood_coefs[flood_coefs['model_type'] == model_type]
        if len(subset) > 0:
            # For SLM/SEM, try to get residual Moran's I
            if model_type == 'SLM':
                slm_moran_file = flood_dir / "flood_only_slm_moran_i_results.csv"
                moran_dict = load_moran_results(slm_moran_file) or moran_dict
            elif model_type == 'SEM':
                sem_moran_file = flood_dir / "flood_only_sem_moran_i_results.csv"
                moran_dict = load_moran_results(sem_moran_file) or moran_dict
            
            summary = create_robustness_summary(subset, moran_dict, comp_dict, "Flood-only", model_type)
            robustness_rows.append(summary)

# Flood+Waste
wasteflood_coefs = coefs_all[(coefs_all['model_name'] == 'Flood+Waste')].copy()
if len(wasteflood_coefs) > 0:
    moran_file = wasteflood_dir / "moran_i_results.csv"
    comp_file = wasteflood_dir / "model_comparison.csv"
    moran_dict = load_moran_results(moran_file)
    comp_dict = load_model_comparison(comp_file)
    
    for model_type in ['OLS', 'SLM', 'SEM']:
        subset = wasteflood_coefs[wasteflood_coefs['model_type'] == model_type]
        if len(subset) > 0:
            # For SLM/SEM, use residual Moran's I
            if model_type == 'SLM':
                slm_moran_file = wasteflood_dir / "slm_moran_i_results.csv"
                moran_dict = load_moran_results(slm_moran_file) or moran_dict
            elif model_type == 'SEM':
                sem_moran_file = wasteflood_dir / "sem_moran_i_results.csv"
                moran_dict = load_moran_results(sem_moran_file) or moran_dict
            
            summary = create_robustness_summary(subset, moran_dict, comp_dict, "Flood+Waste", model_type)
            robustness_rows.append(summary)

# Flood+Waste+Pop
wastefloodpop_coefs = coefs_all[(coefs_all['model_name'] == 'Flood+Waste+Pop')].copy()
if len(wastefloodpop_coefs) > 0:
    # Find moran and comparison files (they have FILE_ID suffixes)
    moran_files = list(wastefloodpop_dir.glob("moran_i_results_*.csv"))
    comp_files = list(wastefloodpop_dir.glob("model_comparison_*.csv"))
    
    moran_dict = load_moran_results(moran_files[0]) if moran_files else None
    comp_dict = load_model_comparison(comp_files[0]) if comp_files else None
    
    for model_type in ['OLS', 'SLM', 'SEM']:
        subset = wastefloodpop_coefs[wastefloodpop_coefs['model_type'] == model_type]
        if len(subset) > 0:
            # For SLM/SEM, use residual Moran's I
            if model_type == 'SLM':
                slm_moran_files = list(wastefloodpop_dir.glob("slm_moran_i_results_*.csv"))
                if slm_moran_files:
                    moran_dict = load_moran_results(slm_moran_files[0])
            elif model_type == 'SEM':
                sem_moran_files = list(wastefloodpop_dir.glob("sem_moran_i_results_*.csv"))
                if sem_moran_files:
                    moran_dict = load_moran_results(sem_moran_files[0])
            
            summary = create_robustness_summary(subset, moran_dict, comp_dict, "Flood+Waste+Pop", model_type)
            robustness_rows.append(summary)

if robustness_rows:
    robustness_df = pd.DataFrame(robustness_rows)
    robustness_file = SUMMARY_OUT_DIR / "supp_robustness_summary.csv"
    robustness_df.to_csv(robustness_file, index=False)
    print(f"\nRobustness summary saved to: {robustness_file}")
    print("\nPreview:")
    print(robustness_df.to_string(index=False))
else:
    print("Warning: No robustness summaries created.")

# -------------------- STEP 4: Create outputs manifest --------------------
print("\n" + "="*80)
print("STEP 4: Creating outputs manifest...")
print("="*80)

manifest_lines = [
    "# Outputs Manifest",
    "# Generated by summarize_outputs.py",
    "",
    "## Main Tables",
    "",
    "### Table 1: Main Regression Results",
    f"- `{SUMMARY_OUT_DIR.name}/table_main_regression.csv` - Wide format coefficient table (separate Coef/SE columns)",
    f"- `{SUMMARY_OUT_DIR.name}/table_main_regression_combined.csv` - Combined format (coef*** (SE))",
    f"- `{SUMMARY_OUT_DIR.name}/table_main_regression_long.csv` - Long format for analysis",
    "- Source files:",
    "  - `Output/flood/flood_only_ols_coefficients.csv`",
    "  - `Output/wasteflood/ols_coefficients.csv`",
    "  - `Output/wasteflood/slm_coefficients.csv`",
    "  - `Output/wastefloodpop/ols_coefficients_*.csv`",
    "  - `Output/wastefloodpop/slm_coefficients_*.csv`",
    "",
    "### Table S1: Robustness Summary",
    f"- `{SUMMARY_OUT_DIR.name}/supp_robustness_summary.csv` - Sign, significance, adj R², Moran's I across specifications",
    "",
    "## Model Outputs by Directory",
    "",
    "### Output/flood/ (Flood-only models)",
    "- `flood_only_ols_coefficients.csv` - OLS coefficients",
    "- `flood_only_slm_coefficients.csv` - Spatial Lag Model coefficients",
    "- `flood_only_sem_coefficients.csv` - Spatial Error Model coefficients",
    "- `flood_only_moran_i_results.csv` - Moran's I test on OLS residuals",
    "- `flood_only_slm_moran_i_results.csv` - Moran's I test on SLM residuals",
    "- `flood_only_slm_impacts.csv` - SLM direct/indirect/total impacts",
    "- `flood_only_slm_pseudo_r2.csv` - SLM pseudo R²",
    "",
    "### Output/wasteflood/ (Flood + Waste models)",
    "- `ols_coefficients.csv` - OLS coefficients",
    "- `slm_coefficients.csv` - Spatial Lag Model coefficients",
    "- `sem_coefficients.csv` - Spatial Error Model coefficients",
    "- `moran_i_results.csv` - Moran's I test on OLS residuals",
    "- `slm_moran_i_results.csv` - Moran's I test on SLM residuals",
    "- `slm_impacts_spreg_summary.csv` - SLM direct/indirect/total impacts",
    "- `slm_pseudo_r2.csv` - SLM pseudo R²",
    "- `vif_analysis.csv` - Variance Inflation Factor analysis",
    "- `model_comparison.csv` - OLS vs SLM vs SEM comparison",
    "",
    "### Output/wastefloodpop/ (Flood + Waste + Population models)",
    "- Files use FILE_ID suffixes (e.g., `*_Y_outflow_outflow_max_F_flood_p95_W_waste_count_P_fb_baseline.csv`)",
    "- `ols_coefficients_*.csv` - OLS coefficients",
    "- `slm_coefficients_*.csv` - Spatial Lag Model coefficients",
    "- `sem_coefficients_*.csv` - Spatial Error Model coefficients",
    "- `moran_i_results_*.csv` - Moran's I test on OLS residuals",
    "- `slm_moran_i_results_*.csv` - Moran's I test on SLM residuals",
    "- `slm_impacts_spreg_summary_*.csv` - SLM direct/indirect/total impacts",
    "- `slm_pseudo_r2_*.csv` - SLM pseudo R²",
    "- `vif_analysis_*.csv` - Variance Inflation Factor analysis",
    "- `model_comparison_*.csv` - OLS vs SLM vs SEM comparison",
    "",
    "## Figures",
    "",
    "### Figure/flood/",
    "- `flood_only_residual_map.png` - OLS residual map",
    "- `flood_only_moran_scatter.png` - Moran's I scatterplot",
    "",
    "### Figure/wasteflood/",
    "- (Add figure descriptions here)",
    "",
    "### Figure/wastefloodpop/",
    "- (Add figure descriptions here)",
    "",
    "## Notes",
    "",
    "- Significance levels: *** p<0.001, ** p<0.01, * p<0.05",
    "- All models use HC3 robust standard errors for OLS",
    "- Spatial models use Maximum Likelihood estimation",
    "- Moran's I tests use 999 permutations",
    "",
]

manifest_file = SUMMARY_OUT_DIR / "outputs_manifest.md"
with open(manifest_file, 'w') as f:
    f.write('\n'.join(manifest_lines))

print(f"\nOutputs manifest saved to: {manifest_file}")

# -------------------- STEP 5: Create coefficient + statistics table --------------------
print("\n" + "="*80)
print("STEP 5: Creating coefficient + statistics table...")
print("="*80)

def create_coef_stats_table():
    """
    Create table with coefficients and model statistics:
    Rows: Flood exposure (p95), Waste accumulation, Spatial lag (ρ), Constant,
          Observations, Adj. R² / Pseudo R², Moran's I (residuals), AIC
    Columns: Flood-only OLS, Flood + Waste OLS, Flood + Waste SLM
    """
    table_rows = []
    
    # Helper to get coefficient value
    def get_coef_value(coefs_df, model_name, model_type, var_name):
        """Extract coefficient value for a variable."""
        subset = coefs_df[
            (coefs_df['model_name'] == model_name) & 
            (coefs_df['model_type'] == model_type) &
            (coefs_df['variable'].str.contains(var_name, case=False, na=False))
        ]
        if len(subset) > 0:
            coef = subset.iloc[0]['coefficient']
            se = subset.iloc[0]['std_err']
            pval = subset.iloc[0]['p_value']
            stars = get_significance_stars(pval)
            return f"{coef:.2f}{stars}", f"({se:.2f})"
        return "", ""
    
    # Helper to get rho value
    def get_rho_value(coefs_df, model_name, model_type):
        """Extract rho (spatial lag) coefficient."""
        subset = coefs_df[
            (coefs_df['model_name'] == model_name) & 
            (coefs_df['model_type'] == model_type) &
            (coefs_df['variable'] == 'rho')
        ]
        if len(subset) > 0:
            coef = subset.iloc[0]['coefficient']
            se = subset.iloc[0]['std_err']
            pval = subset.iloc[0]['p_value']
            stars = get_significance_stars(pval)
            return f"{coef:.2f}{stars}", f"({se:.2f})"
        return "", ""
    
    # Helper to get constant
    def get_const_value(coefs_df, model_name, model_type):
        """Extract constant coefficient."""
        subset = coefs_df[
            (coefs_df['model_name'] == model_name) & 
            (coefs_df['model_type'] == model_type) &
            (coefs_df['variable'] == 'const')
        ]
        if len(subset) > 0:
            coef = subset.iloc[0]['coefficient']
            se = subset.iloc[0]['std_err']
            pval = subset.iloc[0]['p_value']
            stars = get_significance_stars(pval)
            return f"{coef:.2f}{stars}", f"({se:.2f})"
        return "", ""
    
    # Get flood coefficient
    flood_coef_fo, flood_se_fo = get_coef_value(coefs_all, 'Flood-only', 'OLS', 'flood')
    flood_coef_fw, flood_se_fw = get_coef_value(coefs_all, 'Flood+Waste', 'OLS', 'flood')
    flood_coef_fw_slm, flood_se_fw_slm = get_coef_value(coefs_all, 'Flood+Waste', 'SLM', 'flood')
    
    # Get waste coefficient
    waste_coef_fw, waste_se_fw = get_coef_value(coefs_all, 'Flood+Waste', 'OLS', 'waste')
    waste_coef_fw_slm, waste_se_fw_slm = get_coef_value(coefs_all, 'Flood+Waste', 'SLM', 'waste')
    
    # Get rho
    rho_coef_fw_slm, rho_se_fw_slm = get_rho_value(coefs_all, 'Flood+Waste', 'SLM')
    
    # Get constants
    const_coef_fo, const_se_fo = get_const_value(coefs_all, 'Flood-only', 'OLS')
    const_coef_fw, const_se_fw = get_const_value(coefs_all, 'Flood+Waste', 'OLS')
    const_coef_fw_slm, const_se_fw_slm = get_const_value(coefs_all, 'Flood+Waste', 'SLM')
    
    # Get model statistics
    # Observations - count from coefficient files (number of non-null coefficients)
    def get_nobs(coefs_df, model_name, model_type):
        """Estimate nobs from coefficient file (count non-null coefficients)."""
        subset = coefs_df[
            (coefs_df['model_name'] == model_name) & 
            (coefs_df['model_type'] == model_type)
        ]
        # For now, we'll need to read from model data files or summaries
        # Try to get from model comparison or summary files
        return None
    
    # Load model statistics
    flood_moran = load_moran_results(flood_dir / "flood_only_moran_i_results.csv")
    flood_slm_pseudo_r2 = load_pseudo_r2(flood_dir / "flood_only_slm_pseudo_r2.csv")
    
    # Flood+Waste OLS and SLM
    wasteflood_comp = load_model_comparison(wasteflood_dir / "model_comparison.csv")
    wasteflood_moran_ols = load_moran_results(wasteflood_dir / "moran_i_results.csv")
    wasteflood_moran_slm = load_moran_results(wasteflood_dir / "slm_moran_i_results.csv")
    wasteflood_slm_pseudo_r2 = load_pseudo_r2(wasteflood_dir / "slm_pseudo_r2.csv")
    
    # Get AIC from model comparison
    def get_aic_from_comparison(file_path, model_type):
        """Get AIC from model comparison file."""
        if not file_path.exists():
            return ""
        try:
            df = pd.read_csv(file_path)
            if 'Model' in df.columns and 'AIC' in df.columns:
                subset = df[df['Model'] == model_type]
                if len(subset) > 0:
                    aic_val = subset.iloc[0]['AIC']
                    return f"{float(aic_val):.2f}" if pd.notna(aic_val) else ""
        except:
            pass
        return ""
    
    # Get observations from model data files
    def get_nobs_from_model_data(model_dir, prefix=""):
        """Get number of observations from model data file."""
        model_data_file = model_dir / f"{prefix}model_data.gpkg"
        if model_data_file.exists():
            try:
                import geopandas as gpd
                gdf = gpd.read_file(model_data_file)
                return len(gdf)
            except:
                pass
        return None
    
    nobs_flood = get_nobs_from_model_data(flood_dir, "flood_only_")
    nobs_wasteflood = get_nobs_from_model_data(wasteflood_dir)
    
    # Format Adj R² / Pseudo R²
    def format_r2(r2_val):
        """Format R² value."""
        if r2_val is None or pd.isna(r2_val):
            return ""
        try:
            return f"{float(r2_val):.4f}"
        except:
            return str(r2_val)
    
    # Format Moran's I with significance
    def format_moran_i(moran_dict):
        """Format Moran's I with significance stars."""
        if not moran_dict:
            return ""
        i_val = moran_dict.get("Moran's I", None)
        p_val = moran_dict.get("p-value", None)
        if i_val is None or pd.isna(i_val):
            return ""
        try:
            i_val = float(i_val)
            if p_val is not None and not pd.isna(p_val):
                p_val = float(p_val)
                stars = get_significance_stars(p_val)
                return f"{i_val:.4f}{stars}"
            return f"{i_val:.4f}"
        except:
            return str(i_val)
    
    # Get flood-only OLS adj R² - parse from OLS summary file
    def parse_adj_r2_from_summary(summary_file):
        """Parse Adj R² from OLS summary text file."""
        if not summary_file.exists():
            return ""
        try:
            with open(summary_file, 'r') as f:
                content = f.read()
                # Look for "Adj. R-squared:" pattern
                import re
                match = re.search(r'Adj\. R-squared:\s+([-\d.]+)', content)
                if match:
                    return f"{float(match.group(1)):.4f}"
        except:
            pass
        return ""
    
    def parse_nobs_from_summary(summary_file):
        """Parse number of observations from OLS summary text file."""
        if not summary_file.exists():
            return None
        try:
            with open(summary_file, 'r') as f:
                content = f.read()
                # Look for "No. Observations:" pattern
                import re
                match = re.search(r'No\. Observations:\s+(\d+)', content)
                if match:
                    return int(match.group(1))
        except:
            pass
        return None
    
    def parse_aic_from_summary(summary_file):
        """Parse AIC from OLS summary text file."""
        if not summary_file.exists():
            return ""
        try:
            with open(summary_file, 'r') as f:
                content = f.read()
                # Look for "AIC:" pattern
                import re
                match = re.search(r'AIC:\s+([\d.]+)', content)
                if match:
                    return f"{float(match.group(1)):.2f}"
        except:
            pass
        return ""
    
    flood_ols_r2 = parse_adj_r2_from_summary(flood_dir / "flood_only_ols_summary.txt")
    flood_nobs = parse_nobs_from_summary(flood_dir / "flood_only_ols_summary.txt")
    if flood_nobs:
        nobs_flood = flood_nobs
    
    # Also try to get wasteflood nobs from summary
    wasteflood_nobs = parse_nobs_from_summary(wasteflood_dir / "ols_summary.txt")
    if wasteflood_nobs:
        nobs_wasteflood = wasteflood_nobs
    
    # Build table
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
            f"{flood_coef_fo} {flood_se_fo}".strip() if flood_coef_fo else "",
            "",  # No waste in flood-only
            "",  # No rho in OLS
            f"{const_coef_fo} {const_se_fo}".strip() if const_coef_fo else "",
            str(nobs_flood) if nobs_flood else "",
            flood_ols_r2 if flood_ols_r2 else "",
            format_moran_i(flood_moran),
            parse_aic_from_summary(flood_dir / "flood_only_ols_summary.txt")
        ],
        'Flood + Waste OLS': [
            f"{flood_coef_fw} {flood_se_fw}".strip() if flood_coef_fw else "",
            f"{waste_coef_fw} {waste_se_fw}".strip() if waste_coef_fw else "",
            "",  # No rho in OLS
            f"{const_coef_fw} {const_se_fw}".strip() if const_coef_fw else "",
            str(nobs_wasteflood) if nobs_wasteflood else "",
            format_r2(wasteflood_comp.get('OLS') if wasteflood_comp else None),
            format_moran_i(wasteflood_moran_ols),
            get_aic_from_comparison(wasteflood_dir / "model_comparison.csv", "OLS")
        ],
        'Flood + Waste SLM': [
            f"{flood_coef_fw_slm} {flood_se_fw_slm}".strip() if flood_coef_fw_slm else "",
            f"{waste_coef_fw_slm} {waste_se_fw_slm}".strip() if waste_coef_fw_slm else "",
            f"{rho_coef_fw_slm} {rho_se_fw_slm}".strip() if rho_coef_fw_slm else "",
            f"{const_coef_fw_slm} {const_se_fw_slm}".strip() if const_coef_fw_slm else "",
            str(nobs_wasteflood) if nobs_wasteflood else "",
            format_r2(wasteflood_slm_pseudo_r2 if wasteflood_slm_pseudo_r2 else (wasteflood_comp.get('SLM') if wasteflood_comp else None)),
            format_moran_i(wasteflood_moran_slm),
            get_aic_from_comparison(wasteflood_dir / "model_comparison.csv", "SLM")
        ]
    }
    
    # Clean up formatting
    for col in ['Flood-only OLS', 'Flood + Waste OLS', 'Flood + Waste SLM']:
        table_data[col] = [val.strip() if val else "" for val in table_data[col]]
    
    table_coef_stats = pd.DataFrame(table_data)
    
    # Save table
    coef_stats_file = SUMMARY_OUT_DIR / "T1_table_coefficients_statistics.csv"
    table_coef_stats.to_csv(coef_stats_file, index=False)
    print(f"\nCoefficient + statistics table saved to: {coef_stats_file}")
    print("\nPreview:")
    print(table_coef_stats.to_string(index=False))
    
    return table_coef_stats

# Create the table
table_coef_stats = create_coef_stats_table()

# -------------------- Summary --------------------
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"\nAll summary files saved to: {SUMMARY_OUT_DIR}")
print("\nFiles created:")
print(f"  1. {SUMMARY_OUT_DIR.name}/table_main_regression.csv - Main coefficient table (wide format)")
print(f"  2. {SUMMARY_OUT_DIR.name}/table_main_regression_combined.csv - Main coefficient table (combined format)")
print(f"  3. {SUMMARY_OUT_DIR.name}/table_main_regression_long.csv - Main coefficient table (long format)")
print(f"  4. {SUMMARY_OUT_DIR.name}/table_coefficients_statistics.csv - Coefficients + statistics table")
print(f"  5. {SUMMARY_OUT_DIR.name}/supp_robustness_summary.csv - Robustness summary table")
print(f"  6. {SUMMARY_OUT_DIR.name}/outputs_manifest.md - Outputs manifest/metadata")
print("\n" + "="*80)

