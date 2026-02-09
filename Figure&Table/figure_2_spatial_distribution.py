#!/usr/bin/env python3
"""
figure_2_spatial_distribution.py

FIGURE 2 (Main text, most important map figure)
Spatial distribution of displacement, flood exposure, and waste accumulation

Purpose: Visually demonstrate why flood alone is insufficient and why waste matters.

Panel A: Displacement intensity (choropleth)
Panel B: Flood exposure (p95 inundation)
Panel C: Waste accumulation (raw count)

FB baseline filter: set FB_BASELINE_MIN = 50 (default) to keep only cells with
3_fb_baseline_median >= 50 and not NaN; set to None to use all rows.

Requirements:
  geopandas, matplotlib, seaborn, numpy, pandas, pathlib

USAGE:
  Run this script in your Python environment with the required packages installed:

  pip install geopandas matplotlib seaborn numpy pandas

  Then execute:
  python figure_2_spatial_distribution.py

OUTPUT:
  Figure/wasteflood/figure_2_spatial_distribution.png
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.colors import Normalize, LogNorm
from matplotlib.colorbar import ColorbarBase
from mpl_toolkits.axes_grid1 import make_axes_locatable

# -------------------- USER PARAMETERS --------------------
OUTPUT_BASE = Path("/Users/wenlanzhang/PycharmProjects/Waste_Flood_out/Output")
MODEL_DATA_PATH = OUTPUT_BASE / "wasteflood" / "model_data.gpkg"
MODEL_DATA_LAYER = "model_data"

FLOOD_WASTE_DATA_PATH = Path("/Users/wenlanzhang/PycharmProjects/Waste_Flood_out/Data/4/4_flood_waste_metrics_quadkey.gpkg")
FLOOD_WASTE_LAYER = "4_flood_waste_metrics_quadkey"

# FB baseline filter: keep only rows with 3_fb_baseline_median >= this and not NaN (match baseline-controlled runs)
# Set to None to use all rows; default 50
FB_BASELINE_MIN = 50
BASELINE_COL = "3_fb_baseline_median"

FIGURE_DIR = Path("/Users/wenlanzhang/PycharmProjects/Waste_Flood_out/Figure/wasteflood")
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# Figure settings
FIG_SIZE = (18, 8)  # Width x Height for three-panel figure (increased height for colorbars)
DPI = 300
FONT_SIZE = 12

# Color schemes for different variables
DISPLACEMENT_CMAP = 'YlOrRd'  # Yellow-Orange-Red for displacement intensity
FLOOD_CMAP = 'Blues'          # Blues for flood depth
WASTE_CMAP = 'YlOrBr'         # Yellow-Orange-Brown for waste accumulation

# Panel labels (more analytical)
PANEL_A_LABEL = 'A. Flood-induced FB user displacement intensity'
PANEL_B_LABEL = 'B. Flood exposure (95th percentile inundation depth)'
PANEL_C_LABEL = 'C. Persistent waste accumulation'

# Data columns
# DISPLACEMENT_COL = '3_estimated_outflow_pop_from_2_outflow_max'  # Should be in flood-waste data
DISPLACEMENT_COL = '2_outflow_max'
FLOOD_COL = '4_flood_p95'                                        # From flood-waste data
WASTE_COL = '4_waste_count'                                      # From flood-waste data

# -------------------- Helper functions --------------------

def load_spatial_data(model_path, model_layer, flood_waste_path, flood_waste_layer):
    """Load and merge model data with flood-waste data."""

    print("Loading flood-waste data (contains all variables)...")
    if not flood_waste_path.exists():
        raise FileNotFoundError(f"Flood-waste data GPKG not found: {flood_waste_path}")

    try:
        flood_gdf = gpd.read_file(flood_waste_path, layer=flood_waste_layer)
        print(f"  Loaded {len(flood_gdf):,} flood-waste features from layer '{flood_waste_layer}'")
    except Exception as e:
        print(f"  Failed to load layer '{flood_waste_layer}': {e}")
        print("  Attempting to load without specifying layer...")
        try:
            flood_gdf = gpd.read_file(flood_waste_path)
            print(f"  Loaded {len(flood_gdf):,} flood-waste features (no layer specified)")
        except Exception as e2:
            raise RuntimeError(f"Cannot load flood-waste data: {e2}")

    # Check if displacement data is already in flood-waste data
    if DISPLACEMENT_COL in flood_gdf.columns:
        print(f"  Found displacement data ({DISPLACEMENT_COL}) in flood-waste file")
        merged_gdf = flood_gdf
    else:
        print(f"  Displacement data ({DISPLACEMENT_COL}) not found in flood-waste file")
        print("  Attempting to load from model data...")

        if not model_path.exists():
            raise FileNotFoundError(f"Model data GPKG not found: {model_path}. "
                                   "Run 'python model_wasteflood.py' to generate it.")

        try:
            model_gdf = gpd.read_file(model_path, layer=model_layer)
            print(f"  Loaded {len(model_gdf):,} model features from layer '{model_layer}'")
        except Exception as e:
            print(f"  Failed to load layer '{model_layer}': {e}")
            print("  Attempting to load without specifying layer...")
            try:
                model_gdf = gpd.read_file(model_path)
                print(f"  Loaded {len(model_gdf):,} model features (no layer specified)")
            except Exception as e2:
                print(f"  Model data loading failed: {e2}")
                print("  Continuing with flood-waste data only (displacement data will be missing)")
                merged_gdf = flood_gdf
            else:
                # Merge model data with flood-waste data
                print("  Merging datasets on quadkey...")
                merge_cols = ['quadkey', FLOOD_COL, WASTE_COL, DISPLACEMENT_COL]
                if BASELINE_COL in flood_gdf.columns:
                    merge_cols.append(BASELINE_COL)
                merged_gdf = model_gdf.merge(flood_gdf[merge_cols],
                                            on='quadkey', how='left')
                print(f"  Merged dataset has {len(merged_gdf):,} features")
        else:
            # Merge model data with flood-waste data
            merge_cols = ['quadkey', FLOOD_COL, WASTE_COL]
            if BASELINE_COL in flood_gdf.columns:
                merge_cols.append(BASELINE_COL)
            merged_gdf = model_gdf.merge(flood_gdf[merge_cols],
                                        on='quadkey', how='left')
            print(f"  Merged dataset has {len(merged_gdf):,} features")

    return merged_gdf

def create_variable_map(ax, gdf, var_col, title, cmap, colorbar_label, show_colorbar=True, log_scale=False):
    """Create a choropleth map for a variable with colorbar."""

    # Handle missing data
    valid_data = gdf[var_col].dropna()
    if len(valid_data) == 0:
        print(f"Warning: No valid data for {var_col}")
        # Plot empty map
        gdf.plot(ax=ax, color='lightgrey', edgecolor='none')
        return ax

    # Apply quantile clipping for better visualization (98th percentile)
    if log_scale:
        # For log scale, clip at 98th percentile to avoid extreme outliers
        vmax = valid_data.quantile(0.98)
        vmin = valid_data[valid_data > 0].min()
        # Ensure we have valid data after clipping
        clipped_data = valid_data[valid_data <= vmax]
        if len(clipped_data) == 0:
            clipped_data = valid_data
            vmax = valid_data.max()
    else:
        # For linear scale, still clip at 98th percentile
        vmax = valid_data.quantile(0.98)
        vmin = valid_data.min()
        clipped_data = valid_data[valid_data <= vmax]
        if len(clipped_data) == 0:
            clipped_data = valid_data
            vmax = valid_data.max()

    # Determine color normalization
    if log_scale and (clipped_data > 0).any():
        # Log scale for variables with large range and positive values
        norm = LogNorm(vmin=vmin, vmax=vmax)
    else:
        # Linear scale
        norm = Normalize(vmin=vmin, vmax=vmax)

    # Plot the data
    gdf_plot = gdf.plot(column=var_col,
                       ax=ax,
                       cmap=cmap,
                       norm=norm,
                       legend=False,  # We'll add our own colorbar
                       edgecolor='none',
                       linewidth=0.1,
                       missing_kwds={
                           "color": "lightgrey",
                           "label": "No data"
                       })

    # Add colorbar below the axis
    if show_colorbar:
        # Create colorbar axis below the main axis
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("bottom", size="3%", pad=0.1)

        # Create colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, cax=cax, orientation='horizontal')

        # Style the colorbar
        cbar.ax.tick_params(labelsize=FONT_SIZE-2)
        cbar.set_label(colorbar_label, fontsize=FONT_SIZE-1, labelpad=2)

        # Format tick labels appropriately
        if log_scale:
            # For log scale, show actual values
            import matplotlib.ticker as ticker
            cbar.ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        # For linear scale (flood depths), use matplotlib's default formatting

    # Remove axis labels and ticks for cleaner look
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')

    # Add title
    ax.set_title(title, fontsize=FONT_SIZE+4, fontweight='bold', pad=10)

    return ax

def create_figure_2(gdf):
    """Create Figure 2: Spatial distribution of key variables."""

    print("\n" + "="*70)
    print("Creating Figure 2: Spatial distribution of displacement, flood, and waste")
    print("="*70)

    # Create figure with three panels
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=FIG_SIZE)

    # Panel A: Displacement intensity
    print(f"Creating Panel A: Displacement ({DISPLACEMENT_COL})")
    create_variable_map(ax1, gdf, DISPLACEMENT_COL, PANEL_A_LABEL,
                       DISPLACEMENT_CMAP, "Population outflow (log scale)", show_colorbar=True, log_scale=True)

    # Panel B: Flood exposure
    print(f"Creating Panel B: Flood exposure ({FLOOD_COL})")
    create_variable_map(ax2, gdf, FLOOD_COL, PANEL_B_LABEL,
                       FLOOD_CMAP, "95th percentile flood depth (m)", show_colorbar=True, log_scale=False)

    # Panel C: Waste accumulation
    print(f"Creating Panel C: Waste accumulation ({WASTE_COL})")
    create_variable_map(ax3, gdf, WASTE_COL, PANEL_C_LABEL,
                       WASTE_CMAP, "Waste count (log scale)", show_colorbar=True, log_scale=True)

    # Adjust layout
    plt.tight_layout()

    # Save figure
    output_path = FIGURE_DIR / "figure_2_spatial_distribution.png"
    print(f"\nSaving figure to: {output_path}")
    fig.savefig(output_path, dpi=DPI, bbox_inches='tight')

    print("✅ Figure 2 saved successfully!")

    return fig

# -------------------- Main execution --------------------

def main():
    print("="*80)
    print("FIGURE 2: Spatial distribution of displacement, flood, and waste")
    print("="*80)

    try:
        # Load and merge spatial data
        gdf = load_spatial_data(MODEL_DATA_PATH, MODEL_DATA_LAYER,
                               FLOOD_WASTE_DATA_PATH, FLOOD_WASTE_LAYER)

        # FB baseline filter: keep only rows with sufficient baseline (match baseline-controlled runs)
        if FB_BASELINE_MIN is not None and BASELINE_COL in gdf.columns:
            n_before = len(gdf)
            gdf = gdf[gdf[BASELINE_COL].notna() & (gdf[BASELINE_COL] >= FB_BASELINE_MIN)].copy()
            gdf = gdf.reset_index(drop=True)
            n_after = len(gdf)
            print(f"\nFB baseline filter: {BASELINE_COL} >= {FB_BASELINE_MIN} and not NaN.")
            print(f"  Rows before: {n_before:,}  after: {n_after:,}  (removed {n_before - n_after:,})")

        # Check for required columns
        required_cols = [DISPLACEMENT_COL, FLOOD_COL, WASTE_COL]
        missing_cols = [col for col in required_cols if col not in gdf.columns]

        if missing_cols:
            print(f"❌ Missing required columns: {missing_cols}")
            print("Available columns:")
            for col in sorted(gdf.columns):
                print(f"  - {col}")
            raise RuntimeError(f"Missing required columns: {missing_cols}")

        # Basic statistics
        print(f"\nData summary:")
        for col in required_cols:
            valid = gdf[col].dropna()
            print(f"  {col}: {len(valid):,} valid values, "
                  f"range: {valid.min():.2f} - {valid.max():.2f}, "
                  f"mean: {valid.mean():.2f}")

        # Create Figure 2
        fig = create_figure_2(gdf)

        print("\n" + "="*80)
        print("Figure 2 completed successfully!")
        print("="*80)
        print(f"Output: {FIGURE_DIR / 'figure_2_spatial_distribution.png'}")
        print("\nPurpose: Visually demonstrate why flood alone is insufficient and why waste matters")
        print("Panel A: Displacement intensity shows where population outflow occurred")
        print("Panel B: Flood exposure shows flood risk areas")
        print("Panel C: Waste accumulation shows environmental degradation hotspots")

    except Exception as e:
        print(f"❌ Error: {e}")
        raise

if __name__ == "__main__":
    main()