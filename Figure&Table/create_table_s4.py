#!/usr/bin/env python3
"""
create_table_s4.py

Creates Table S4: Spatial lag model impacts (Mechanism & spillovers)

Shows Direct, Indirect, and Total impacts from the Spatial Lag Model (SLM)
to demonstrate that waste effects propagate through spatial spillovers.

Requirements:
  pandas, pathlib
"""
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import pandas as pd

# -------------------- USER PARAMETERS --------------------
OUTPUT_BASE = Path("/Users/wenlanzhang/PycharmProjects/Waste_Flood_out/Output")
SUMMARY_OUT_DIR = Path("/Users/wenlanzhang/PycharmProjects/Waste_Flood_out/Output/summary")
SUMMARY_OUT_DIR.mkdir(parents=True, exist_ok=True)

# Model directories
WASTEFLOOD_DIR = OUTPUT_BASE / "wasteflood"
WASTEFLOODPOP_DIR = OUTPUT_BASE / "wastefloodpop"
FLOODPOP_DIR = OUTPUT_BASE / "floodpop"

# Variable name mappings for display (include all flood/waste variants used by model search)
VAR_DISPLAY_NAMES = {
    '4_flood_p95': 'Flood exposure',
    '4_flood_mean': 'Flood exposure',
    '4_flood_max': 'Flood exposure',
    '4_waste_count': 'Waste accumulation',
    '4_waste_per_population': 'Waste accumulation',
    '4_waste_per_svi_count': 'Waste accumulation',
    '3_fb_baseline_median': 'Baseline population',
    '3_worldpop': 'Baseline population',
}

# -------------------- Helper functions --------------------
def load_impacts(file_path):
    """Load SLM impacts file (variable, Direct, Indirect, Total)."""
    if not file_path.exists():
        return None
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error loading impacts from {file_path}: {e}")
        return None


def load_impacts_from_slm_metrics(metrics_path):
    """
    Build impacts DataFrame from consolidated slm_metrics.csv (metric, value rows).
    Expects rows like: {var}_Direct, {var}_Indirect, {var}_Total.
    Returns DataFrame with columns: variable, Direct, Indirect, Total.
    """
    if not metrics_path.exists():
        return None
    try:
        df = pd.read_csv(metrics_path)
        if 'metric' not in df.columns or 'value' not in df.columns:
            return None
        # Build dict: metric -> value (skip pseudo_r2)
        metrics = {}
        for _, row in df.iterrows():
            m = str(row['metric']).strip()
            if m == 'pseudo_r2':
                continue
            metrics[m] = row['value']
        rows = []
        for m, direct in metrics.items():
            if not m.endswith('_Direct'):
                continue
            var = m[:-7]  # drop '_Direct'
            indirect = metrics.get(var + '_Indirect')
            total = metrics.get(var + '_Total')
            if indirect is not None and total is not None:
                rows.append({'variable': var, 'Direct': direct, 'Indirect': indirect, 'Total': total})
        if not rows:
            return None
        return pd.DataFrame(rows)
    except Exception as e:
        print(f"Error building impacts from {metrics_path}: {e}")
        return None

def format_impact(value):
    """Format impact value to one decimal place."""
    if pd.isna(value):
        return "—"
    try:
        return f"{float(value):.1f}"
    except:
        return str(value)

# -------------------- Load data --------------------
print("="*80)
print("Creating Table S4 and S41: Spatial lag model impacts")
print("="*80)

# Load Flood + Waste SLM impacts (no baseline)
# Prefer slm_impacts_spreg_summary.csv; fall back to slm_metrics.csv (consolidated output)
print("\nLoading Flood + Waste SLM impacts...")
wasteflood_impacts_file = WASTEFLOOD_DIR / "slm_impacts_spreg_summary.csv"
wasteflood_impacts_df = load_impacts(wasteflood_impacts_file)
if wasteflood_impacts_df is None:
    wasteflood_metrics_file = WASTEFLOOD_DIR / "slm_metrics.csv"
    wasteflood_impacts_df = load_impacts_from_slm_metrics(wasteflood_metrics_file)
    if wasteflood_impacts_df is not None:
        print(f"Using impacts from consolidated file: {wasteflood_metrics_file.name}")
if wasteflood_impacts_df is None:
    raise FileNotFoundError(
        f"No Flood + Waste SLM impacts found. Looked for: {wasteflood_impacts_file.name} or slm_metrics.csv in {WASTEFLOOD_DIR}"
    )
print(f"Found impacts: {len(wasteflood_impacts_df)} variables")

# Load Flood + Waste + Population SLM impacts (prefer Facebook baseline; fall back to any or slm_metrics)
print("\nLoading Flood + Waste + Population SLM impacts...")
wastefloodpop_impacts_files = [f for f in WASTEFLOODPOP_DIR.glob("slm_impacts_spreg_summary_*.csv") if 'fb_baseline' in f.name]
if not wastefloodpop_impacts_files:
    wastefloodpop_impacts_files = list(WASTEFLOODPOP_DIR.glob("slm_impacts_spreg_summary_*.csv"))
wastefloodpop_impacts_df = None
if wastefloodpop_impacts_files:
    wastefloodpop_impacts_df = load_impacts(wastefloodpop_impacts_files[0])
    if wastefloodpop_impacts_df is not None:
        print(f"Found impacts file: {wastefloodpop_impacts_files[0].name}")
if wastefloodpop_impacts_df is None:
    # Fall back to slm_metrics_*.csv (consolidated output); prefer fb_baseline
    metrics_files = [f for f in WASTEFLOODPOP_DIR.glob("slm_metrics_*.csv") if 'fb_baseline' in f.name]
    if not metrics_files:
        metrics_files = list(WASTEFLOODPOP_DIR.glob("slm_metrics_*.csv"))
    if metrics_files:
        wastefloodpop_impacts_df = load_impacts_from_slm_metrics(metrics_files[0])
        if wastefloodpop_impacts_df is not None:
            print(f"Using impacts from consolidated file: {metrics_files[0].name}")
if wastefloodpop_impacts_df is None:
    raise FileNotFoundError(
        f"No Flood + Waste + Population SLM impacts found in {WASTEFLOODPOP_DIR} "
        "(looked for slm_impacts_spreg_summary_*.csv or slm_metrics_*.csv)"
    )
print(f"Found impacts: {len(wastefloodpop_impacts_df)} variables")

# Verify required columns exist for both
required_cols = ['variable', 'Direct', 'Indirect', 'Total']
for df, name in [(wasteflood_impacts_df, "Flood + Waste"), (wastefloodpop_impacts_df, "Flood + Waste + Population")]:
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in {name} impacts file: {missing_cols}")

# -------------------- Helper function to prepare impacts table --------------------
def _display_name_for_variable(var):
    """Map variable name to display name; fallback for unknown flood/waste/baseline vars."""
    if pd.isna(var):
        return var
    var = str(var).strip()
    if var in VAR_DISPLAY_NAMES:
        return VAR_DISPLAY_NAMES[var]
    var_lower = var.lower()
    if 'flood' in var_lower:
        return 'Flood exposure'
    if 'waste' in var_lower:
        return 'Waste accumulation'
    if 'baseline' in var_lower or 'worldpop' in var_lower:
        return 'Baseline population'
    return var


def prepare_impacts_table(impacts_df, model_name, include_baseline=True):
    """Prepare impacts DataFrame with display names and ordering."""
    df = impacts_df.copy()
    df['Display_Name'] = df['variable'].apply(_display_name_for_variable)
    
    # Filter to variables we want to show (exact list + partial match for model-specific names)
    if include_baseline:
        vars_to_show = ['4_flood_p95', '4_flood_mean', '4_flood_max', '4_waste_count', '3_fb_baseline_median', '3_worldpop']
        pattern = 'flood|waste|baseline|worldpop'
    else:
        vars_to_show = ['4_flood_p95', '4_flood_mean', '4_flood_max', '4_waste_count']
        pattern = 'flood|waste'
    
    df_filtered = df[df['variable'].isin(vars_to_show)].copy()
    if len(df_filtered) == 0:
        df_filtered = df[df['variable'].str.contains(pattern, case=False, na=False)].copy()
    
    # Order: Flood exposure (0), Waste accumulation (1), Baseline population (2)
    var_order = {'Flood exposure': 0, 'Waste accumulation': 1, 'Baseline population': 2}
    df_filtered['_order'] = df_filtered['Display_Name'].map(var_order).fillna(99)
    df_filtered = df_filtered.sort_values('_order').reset_index(drop=True)
    
    # Create table
    table = pd.DataFrame({
        'Variable': df_filtered['Display_Name'],
        'Direct': df_filtered['Direct'].apply(format_impact),
        'Indirect': df_filtered['Indirect'].apply(format_impact),
        'Total': df_filtered['Total'].apply(format_impact)
    })
    
    return table

# -------------------- Create Table S41 (Flood + Waste SLM, no baseline) --------------------
print("\n" + "="*80)
print("Creating Table S41: Flood + Waste SLM impacts")
print("="*80)

table_s41 = prepare_impacts_table(wasteflood_impacts_df, "Flood + Waste", include_baseline=False)

# Save Table S41
output_file_s41 = SUMMARY_OUT_DIR / "Table_S4_spatial_lag_impacts.csv"
table_s41.to_csv(output_file_s41, index=False)

print(table_s41.to_string(index=False))
print("="*80)
print(f"\nTable S41 saved to: {output_file_s41}")

# -------------------- Create Table S5 (Flood + Waste + Population SLM impacts only) --------------------
print("\n" + "="*80)
print("Creating Table S5: Flood + Waste + Population SLM impacts")
print("="*80)

table_s5 = prepare_impacts_table(wastefloodpop_impacts_df, "Flood + Waste + Population", include_baseline=True)

# Save Table S5 (Flood + Waste + Population only; no F+P)
output_file_s5 = SUMMARY_OUT_DIR / "Table_S5_spatial_lag_impacts.csv"
table_s5.to_csv(output_file_s5, index=False)

print(table_s5.to_string(index=False))
print("="*80)
print(f"\nTable S5 saved to: {output_file_s5}")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("Notes:")
print("- Table S4: Impacts from Flood + Waste SLM (no baseline population)")
print("- Table S5: Impacts from Flood + Waste + Population SLM (Direct/Indirect/Total only)")
print("- Values are point estimates from SLM impacts (spreg / slm_metrics)")
print("- Purpose: 'Waste effects propagate through spatial spillovers.'")
print("\nDone.")
