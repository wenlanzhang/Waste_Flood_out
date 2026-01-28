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

# Variable name mappings for display
VAR_DISPLAY_NAMES = {
    '4_flood_p95': 'Flood exposure',
    '4_waste_count': 'Waste accumulation',
    '3_fb_baseline_median': 'Baseline population',
    '3_worldpop': 'Baseline population',
}

# -------------------- Helper functions --------------------
def load_impacts(file_path):
    """Load SLM impacts file."""
    if not file_path.exists():
        return None
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error loading impacts from {file_path}: {e}")
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
print("\nLoading Flood + Waste SLM impacts...")
wasteflood_impacts_file = WASTEFLOOD_DIR / "slm_impacts_spreg_summary.csv"
wasteflood_impacts_df = load_impacts(wasteflood_impacts_file)
if wasteflood_impacts_df is None:
    raise FileNotFoundError(f"No Flood + Waste SLM impacts file found: {wasteflood_impacts_file}")
print(f"Found impacts file: {wasteflood_impacts_file.name}")

# Load Flood + Waste + Population SLM impacts (with Facebook baseline)
print("\nLoading Flood + Waste + Population SLM impacts (with Facebook baseline)...")
wastefloodpop_impacts_files = [f for f in WASTEFLOODPOP_DIR.glob("slm_impacts_spreg_summary_*.csv") if 'fb_baseline' in f.name]
if not wastefloodpop_impacts_files:
    raise FileNotFoundError(f"No Flood + Waste + Population SLM impacts file found (with fb_baseline) in {WASTEFLOODPOP_DIR}")

wastefloodpop_impacts_df = load_impacts(wastefloodpop_impacts_files[0])
if wastefloodpop_impacts_df is None:
    raise RuntimeError("Could not load Flood + Waste + Population impacts data")
print(f"Found impacts file: {wastefloodpop_impacts_files[0].name}")

# Verify required columns exist for both
required_cols = ['variable', 'Direct', 'Indirect', 'Total']
for df, name in [(wasteflood_impacts_df, "Flood + Waste"), (wastefloodpop_impacts_df, "Flood + Waste + Population")]:
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in {name} impacts file: {missing_cols}")

# -------------------- Helper function to prepare impacts table --------------------
def prepare_impacts_table(impacts_df, model_name, include_baseline=True):
    """Prepare impacts DataFrame with display names and ordering."""
    df = impacts_df.copy()
    df['Display_Name'] = df['variable'].map(VAR_DISPLAY_NAMES).fillna(df['variable'])
    
    # Filter to variables we want to show
    if include_baseline:
        vars_to_show = ['4_flood_p95', '4_waste_count', '3_fb_baseline_median', '3_worldpop']
    else:
        vars_to_show = ['4_flood_p95', '4_waste_count']
    
    df_filtered = df[df['variable'].isin(vars_to_show)].copy()
    
    if len(df_filtered) == 0:
        # If exact matches don't work, try partial matching
        if include_baseline:
            pattern = 'flood|waste|baseline|worldpop'
        else:
            pattern = 'flood|waste'
        df_filtered = df[df['variable'].str.contains(pattern, case=False, na=False)].copy()
    
    # Define preferred order
    var_order = {
        'Flood exposure': 0,
        'Waste accumulation': 1,
        'Baseline population': 2
    }
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
output_file_s41 = SUMMARY_OUT_DIR / "Table_S41_spatial_lag_impacts.csv"
table_s41.to_csv(output_file_s41, index=False)

print(table_s41.to_string(index=False))
print("="*80)
print(f"\nTable S41 saved to: {output_file_s41}")

# -------------------- Create Table S4 (Flood + Waste + Population SLM, with baseline) --------------------
print("\n" + "="*80)
print("Creating Table S4: Flood + Waste + Population SLM impacts")
print("="*80)

table_s4 = prepare_impacts_table(wastefloodpop_impacts_df, "Flood + Waste + Population", include_baseline=True)

# Save Table S4
output_file_s4 = SUMMARY_OUT_DIR / "Table_S4_spatial_lag_impacts.csv"
table_s4.to_csv(output_file_s4, index=False)

print(table_s4.to_string(index=False))
print("="*80)
print(f"\nTable S4 saved to: {output_file_s4}")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("Notes:")
print("- Table S41: Impacts from Flood + Waste SLM (no baseline)")
print("- Table S4: Impacts from Flood + Waste + Population SLM (with Facebook baseline)")
print("- Values are point estimates from canonical spreg summaries")
print("- Purpose: 'Waste effects propagate through spatial spillovers.'")
print("\nDone.")
