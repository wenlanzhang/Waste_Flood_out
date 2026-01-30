#!/usr/bin/env python3
"""
plot_model_variable_distributions.py

Publication-quality distribution plots for a list of columns from the
flood-waste GPKG. Layout: 2 or 3 panels per row, as many rows as needed.

Edit COLS_TO_PLOT to choose which columns to plot. Optionally set
COL_LABELS and COL_LOG_X to customize display name and log-scale per column.

Requirements: geopandas, pandas, numpy, matplotlib, seaborn

OUTPUT: Figure/description/model_variable_distributions.png (.pdf)
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import math
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------- Parameters --------------------
GPKG_PATH = Path("/Users/wenlanzhang/PycharmProjects/Waste_Flood_out/Data/4/4_flood_waste_metrics_quadkey.gpkg")
LAYER_NAME = "4_flood_waste_metrics_quadkey"
NODATA = -9999.0

# List of column names to plot (same names as in the GPKG)
COLS_TO_PLOT = [
    "2_outflow_max",
    "4_flood_p95",
    "4_waste_count",
    "3_worldpop",
    "3_fb_baseline_median",
]

# Panels per row (2 or 3)
COLS_PER_ROW = 3

# Optional: custom display label and log-scale per column (defaults: label = column name, log_x = False)
COL_LABELS = {
    "2_outflow_max": "Outflow (max)",
    "4_flood_p95": "Flood exposure (p95)",
    "4_waste_count": "Waste count",
    "3_worldpop": "Population (WorldPop)",
}
COL_LOG_X = {
    "2_outflow_max": True,
    "4_flood_p95": False,
    "4_waste_count": True,
    "3_worldpop": True,
}

FIGURE_DIR = Path("/Users/wenlanzhang/PycharmProjects/Waste_Flood_out/Figure/description")
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH_PNG = FIGURE_DIR / "model_variable_distributions.png"
OUT_PATH_PDF = FIGURE_DIR / "model_variable_distributions.pdf"  # vector for publication

# Publication figure settings (300 DPI, readable fonts)
PANEL_WIDTH = 3.5
PANEL_HEIGHT = 3.0
DPI = 300
FONT_SIZE = 11
TITLE_FONT_SIZE = 12
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.titlesize"] = TITLE_FONT_SIZE
plt.rcParams["axes.labelsize"] = FONT_SIZE
plt.rcParams["xtick.labelsize"] = FONT_SIZE - 1
plt.rcParams["ytick.labelsize"] = FONT_SIZE - 1
plt.rcParams["legend.fontsize"] = FONT_SIZE - 1
plt.rcParams["figure.dpi"] = 100  # screen; savefig uses DPI


def build_var_config(cols_to_plot, col_labels=None, col_log_x=None):
    """Build list of {col, label, xlabel, log_x} from COLS_TO_PLOT and optional dicts."""
    col_labels = col_labels or {}
    col_log_x = col_log_x or {}
    config = []
    for col in cols_to_plot:
        label = col_labels.get(col, col.replace("_", " ").strip())
        config.append({
            "col": col,
            "label": label,
            "xlabel": label,
            "log_x": col_log_x.get(col, False),
        })
    return config


def replace_nodata_with_nan(df, cols):
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
            out.loc[out[c] == NODATA, c] = np.nan
    return out


def load_analysis_sample(path, layer, cols):
    if not path.exists():
        raise FileNotFoundError(f"GPKG not found: {path}")
    gdf = gpd.read_file(path, layer=layer)
    for c in cols:
        if c not in gdf.columns:
            raise ValueError(f"Column '{c}' not found in {path} layer '{layer}'.")
    gdf = replace_nodata_with_nan(gdf, cols)
    df = gdf[cols].copy()
    df = df.dropna(how="any")
    return df


def main():
    cols = list(COLS_TO_PLOT)
    if not cols:
        raise ValueError("COLS_TO_PLOT is empty.")
    var_config = build_var_config(cols, COL_LABELS, COL_LOG_X)

    print(f"Loading data (analysis sample: rows with all {len(cols)} variables non-missing)...")
    df = load_analysis_sample(GPKG_PATH, LAYER_NAME, cols)
    n = len(df)
    print(f"  N = {n:,}")

    n_axes = len(cols)
    ncols = min(COLS_PER_ROW, n_axes)
    nrows = math.ceil(n_axes / ncols)
    fig_width = ncols * PANEL_WIDTH
    fig_height = nrows * PANEL_HEIGHT
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height), constrained_layout=True)
    axes_flat = np.atleast_1d(axes).ravel() if n_axes > 1 else [axes]

    for idx, (ax, cfg) in enumerate(zip(axes_flat, var_config)):
        col = cfg["col"]
        label = cfg["label"]
        xlabel = cfg["xlabel"]
        log_x = cfg["log_x"]
        s = df[col].astype(float)

        # For log scale, drop non-positive if needed
        if log_x and (s <= 0).any():
            s_plot = s[s > 0]
            if len(s_plot) < 2:
                s_plot = s
                log_x = False
        else:
            s_plot = s

        if log_x:
            s_plot = np.log10(s_plot + 1)
            xlabel_use = f"{xlabel} (log₁₀(x+1))"
        else:
            xlabel_use = xlabel

        nbins = min(60, max(25, len(s_plot) // 25))
        ax.hist(s_plot, bins=nbins, color="steelblue", alpha=0.7, edgecolor="white", linewidth=0.4, density=True, label="Histogram")
        sns.kdeplot(s_plot, ax=ax, color="navy", linewidth=2, label="KDE")
        ax.set_xlabel(xlabel_use)
        ax.set_ylabel("Density")
        ax.set_title(f"{label}\n(n = {len(s_plot):,})", fontweight="bold")
        ax.legend(loc="upper right", frameon=True, fancybox=False, edgecolor="gray")
        ax.set_ylim(bottom=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Hide unused subplots (when len(cols) is not a full grid)
    for j in range(len(var_config), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Distribution of selected variables (analysis sample)", fontsize=TITLE_FONT_SIZE + 2, fontweight="bold", y=1.02)
    plt.savefig(OUT_PATH_PNG, dpi=DPI, bbox_inches="tight")
    plt.savefig(OUT_PATH_PDF, bbox_inches="tight")  # vector for journals
    plt.close()
    print(f"Saved: {OUT_PATH_PNG}")
    print(f"Saved: {OUT_PATH_PDF}")


if __name__ == "__main__":
    main()
