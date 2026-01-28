#!/usr/bin/env python3
"""
figure_3_spatial_dependence.py

FIGURE 3 (Main text)
Spatial dependence in displacement and effect of spatial lag modelling

Purpose: Justify why you used spatial econometrics.

Panel A: OLS residuals (Flood + Waste OLS)
Panel B: SLM residuals (Flood + Waste SLM)

Caption message: "Spatial lag models substantially reduce residual clustering compared to OLS."

Requirements:
  geopandas, matplotlib, seaborn, numpy, pandas, pathlib

USAGE:
  Run this script in your Python environment with the required packages installed:

  pip install geopandas matplotlib seaborn numpy pandas

  Then execute:
  python figure_3_spatial_dependence.py

ALTERNATIVE APPROACH:
  If you cannot install packages, use QGIS or ArcGIS to create residual maps:
  1. Load Output/wasteflood/model_data.gpkg
  2. Create choropleth maps using the 'residuals' column (OLS) and computed SLM residuals
  3. Use RdYlBu color scheme (red=positive, blue=negative)
  4. Export as high-resolution PNG (300 DPI)
"""

# Check for required packages
try:
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
    from matplotlib.colors import Normalize
    from matplotlib.colorbar import ColorbarBase
    PACKAGES_AVAILABLE = True
except ImportError as e:
    PACKAGES_AVAILABLE = False
    MISSING_PACKAGE = str(e).split("'")[1] if "'" in str(e) else "unknown"

# -------------------- USER PARAMETERS --------------------
OUTPUT_BASE = Path("/Users/wenlanzhang/PycharmProjects/Waste_Flood_out/Output")
MODEL_DATA_PATH = OUTPUT_BASE / "wasteflood" / "model_data.gpkg"
MODEL_DATA_LAYER = "model_data"

FIGURE_DIR = Path("/Users/wenlanzhang/PycharmProjects/Waste_Flood_out/Figure/wasteflood")
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# Figure settings
FIG_SIZE = (16, 6)  # Width x Height for two-panel figure
DPI = 300
FONT_SIZE = 12

# Color scheme for residuals
CMAP = 'RdYlBu_r'  # Red-Yellow-Blue reversed (red=positive, blue=negative)
CMAP_LABEL = 'Residual Value'

# Panel labels
PANEL_A_LABEL = 'A. OLS Residuals\n(Flood + Waste OLS)'
PANEL_B_LABEL = 'B. SLM Residuals\n(Flood + Waste SLM)'

# -------------------- Helper functions --------------------

def load_model_data(gpkg_path, layer_name):
    """Load the model data GeoPackage."""
    if not gpkg_path.exists():
        raise FileNotFoundError(f"Model data GPKG not found: {gpkg_path}")

    print(f"Loading model data from: {gpkg_path}")
    gdf = gpd.read_file(gpkg_path, layer=layer_name)
    print(f"  Loaded {len(gdf):,} features")
    print(f"  CRS: {gdf.crs}")
    return gdf

def check_residual_columns(gdf):
    """Check for and identify residual columns."""
    all_cols = gdf.columns.tolist()
    residual_cols = [col for col in all_cols if 'resid' in col.lower()]

    print(f"\nAll columns: {all_cols}")
    print(f"Residual columns found: {residual_cols}")

    return residual_cols

# Removed get_residual_column function - now using exact column names

def create_residual_map(ax, gdf, resid_col, title, cmap=CMAP, show_colorbar=True):
    """Create a residual map on the given axis."""

    # Plot the residuals with explicit missing data handling
    gdf.plot(column=resid_col,
             ax=ax,
             cmap=cmap,
             legend=show_colorbar,
             edgecolor='none',
             linewidth=0.1,
             missing_kwds={
                 "color": "lightgrey",
                 "label": "No data"
             })

    # Remove axis labels and ticks for cleaner look
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')

    # Add title
    ax.set_title(title, fontsize=FONT_SIZE+2, fontweight='bold', pad=10)

    return ax

def create_figure_3(gdf, ols_resid_col, slm_resid_col):
    """Create Figure 3: Spatial dependence in displacement."""

    print("\n" + "="*60)
    print("Creating Figure 3: Spatial dependence in displacement")
    print("="*60)

    # Create figure with two panels
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIG_SIZE)

    # Panel A: OLS residuals
    print(f"Creating Panel A: OLS residuals ({ols_resid_col})")
    create_residual_map(ax1, gdf, ols_resid_col, PANEL_A_LABEL, show_colorbar=False)

    # Panel B: SLM residuals
    print(f"Creating Panel B: SLM residuals ({slm_resid_col})")
    create_residual_map(ax2, gdf, slm_resid_col, PANEL_B_LABEL, show_colorbar=False)

    # Add a single colorbar for both panels
    # Get the combined range for colorbar
    ols_resids = gdf[ols_resid_col].dropna()
    slm_resids = gdf[slm_resid_col].dropna()
    all_resids = pd.concat([ols_resids, slm_resids])

    if len(all_resids) > 0:
        # Use symmetric color scale for residuals (critical for visual interpretation)
        abs_max = np.nanmax(np.abs(all_resids))
        vmin, vmax = -abs_max, abs_max
        norm = Normalize(vmin=vmin, vmax=vmax)

        # Add colorbar axis
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
        cbar = ColorbarBase(cbar_ax, cmap=plt.cm.get_cmap(CMAP), norm=norm,
                           orientation='vertical')
        cbar.set_label(CMAP_LABEL, fontsize=FONT_SIZE)
        cbar.ax.tick_params(labelsize=FONT_SIZE-1)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 0.9, 1])  # Leave space for colorbar

    # Save figure
    output_path = FIGURE_DIR / "figure_3_spatial_dependence.png"
    print(f"\nSaving figure to: {output_path}")
    fig.savefig(output_path, dpi=DPI, bbox_inches='tight')

    print("✅ Figure 3 saved successfully!")

    return fig

# -------------------- Main execution --------------------

# Removed approximation function - now using actual saved SLM residuals

def main():
    print("="*80)
    print("FIGURE 3: Spatial dependence in displacement")
    print("="*80)

    if not PACKAGES_AVAILABLE:
        print("\n❌ ERROR: Required package missing:")
        print(f"   {MISSING_PACKAGE}")
        print("\n📦 INSTALLATION INSTRUCTIONS:")
        print("   pip install geopandas matplotlib seaborn numpy pandas")
        print("\n🗺️  ALTERNATIVE: Use GIS software")
        print("   1. Open Output/wasteflood/model_data.gpkg in QGIS/ArcGIS")
        print("   2. Create choropleth maps using 'residuals' column (OLS)")
        print("   3. Compute SLM residuals or use approximation")
        print("   4. Use RdYlBu color scheme (red=positive, blue=negative)")
        print("   5. Export panels as figure_3_panel_a.png and figure_3_panel_b.png")
        return

    print("Starting script execution...")
    try:
        # Load model data
        gdf = load_model_data(MODEL_DATA_PATH, MODEL_DATA_LAYER)

        # Check for residual columns
        residual_cols = check_residual_columns(gdf)

        # Use exact column names (no more searching/patterns)
        ols_resid_col = 'residuals'
        slm_resid_col = 'slm_residuals'

        print(f"\nUsing OLS residuals: {ols_resid_col}")
        print(f"Using SLM residuals: {slm_resid_col}")

        # Fail loudly if required columns are missing
        if ols_resid_col not in gdf.columns:
            raise RuntimeError(f"OLS residuals column '{ols_resid_col}' not found in model_data.gpkg. "
                             "Regenerate model data with OLS residuals saved.")

        if slm_resid_col not in gdf.columns:
            raise RuntimeError(f"SLM residuals column '{slm_resid_col}' not found in model_data.gpkg. "
                             "Regenerate model data with SLM residuals saved (see model_wasteflood.py).")

        # Basic statistics
        ols_mean = gdf[ols_resid_col].mean()
        ols_std = gdf[ols_resid_col].std()
        slm_mean = gdf[slm_resid_col].mean()
        slm_std = gdf[slm_resid_col].std()

        print(f"\nOLS residuals - Mean: {ols_mean:.4f}, Std: {ols_std:.4f}")
        print(f"SLM residuals - Mean: {slm_mean:.4f}, Std: {slm_std:.4f}")

        # Create Figure 3
        print("Creating figure...")
        fig = create_figure_3(gdf, ols_resid_col, slm_resid_col)
        print("Figure created successfully.")

        print("\n" + "="*80)
        print("Figure 3 completed successfully!")
        print("="*80)
        print(f"Output: {FIGURE_DIR / 'figure_3_spatial_dependence.png'}")
        print("\nCaption suggestion:")
        print('"Spatial lag models substantially reduce residual clustering compared to OLS."')

    except Exception as e:
        print(f"Error: {e}")
        raise

if __name__ == "__main__":
    main()