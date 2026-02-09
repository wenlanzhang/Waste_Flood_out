#!/usr/bin/env python3
"""
map_all_columns.py

Maps all mappable (numeric) columns from the final analysis dataset as choropleth maps,
so you can inspect the spatial distribution of every variable.

Input:  Data/4/4_flood_waste_metrics_quadkey.gpkg (layer: 4_flood_waste_metrics_quadkey)
Output: Figure/all_columns_maps/<column_name>.png (one map per column)

Optional: Set FB_BASELINE_MIN = 50 to restrict to cells with sufficient FB baseline; set to None to use all rows.

Requirements: geopandas, matplotlib, numpy, pandas, pathlib
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

# -------------------- USER PARAMETERS --------------------
GPKG_PATH = Path("/Users/wenlanzhang/PycharmProjects/Waste_Flood_out/Data/4/4_flood_waste_metrics_quadkey.gpkg")
LAYER_NAME = "4_flood_waste_metrics_quadkey"
OUT_DIR = Path("/Users/wenlanzhang/PycharmProjects/Waste_Flood_out/Figure/description/all_columns_maps")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# FB baseline filter: keep only rows with 3_fb_baseline_median >= this (set None to use all rows)
FB_BASELINE_MIN = None
BASELINE_COL = "3_fb_baseline_median"

# Columns to skip (identifiers, geometry, or redundant/internal)
SKIP_COLUMNS = {"quadkey", "geometry"}

# Optional: skip redundant columns (per SCRIPT_OUTPUTS_SUMMARY.md)
SKIP_REDUNDANT = True
REDUNDANT_PATTERNS = ("_safe", "_final", "3_ratio_was_capped", "3_used_global_ratio")

# Figure settings
FIG_SIZE_SINGLE = (10, 8)
DPI = 150
FONT_SIZE = 11
QUANTILE_CLIP = 0.98  # clip color scale at this quantile to reduce outlier influence

# Default colormap and scale; can override per-column below
DEFAULT_CMAP = "viridis"
DEFAULT_LOG = False

# Per-column overrides: (short_title, cmap, log_scale). None = use default.
COLUMN_CONFIG = {
    # Displacement (Step 1–2)
    "1_outflow_accumulated": ("Outflow accumulated (all hours)", "YlOrRd", True),
    "1_outflow_accumulated_hour0": ("Outflow accumulated (hour 0)", "YlOrRd", True),
    "1_outflow_max": ("Outflow max (all hours)", "YlOrRd", True),
    "1_outflow_max_hour0": ("Outflow max (hour 0)", "YlOrRd", True),
    "1_outflow_mean": ("Outflow mean", "YlOrRd", True),
    "1_outflow_p95": ("Outflow 95th pct", "YlOrRd", True),
    "2_outflow_max": ("Displacement max (CSAT)", "YlOrRd", True),
    "2_displaced_excess_max": ("Excess displacement max", "YlOrRd", True),
    # Scaled displacement (Step 3)
    "3_estimated_outflow_pop_from_2_outflow_max": ("Est. outflow pop (from 2_outflow_max)", "YlOrRd", True),
    "3_estimated_outflow_pop_from_1_outflow_accumulated_hour0": ("Est. outflow pop (accum hour0)", "YlOrRd", True),
    "3_estimated_outflow_pop_from_1_outflow_max_hour0": ("Est. outflow pop (max hour0)", "YlOrRd", True),
    "3_estimated_excess_displacement_pop": ("Est. excess displacement pop", "YlOrRd", True),
    # Percentages
    "3_pct_outflow_fb_from_2_outflow_max": ("% outflow FB (2_outflow_max)", "YlOrRd", False),
    "3_pct_outflow_fb_from_1_outflow_accumulated_hour0": ("% outflow FB (accum hour0)", "YlOrRd", False),
    "3_pct_outflow_fb_from_1_outflow_max_hour0": ("% outflow FB (max hour0)", "YlOrRd", False),
    "3_pct_outflow_worldpop_from_2_outflow_max": ("% outflow WorldPop (2_outflow_max)", "YlOrRd", False),
    "3_pct_outflow_worldpop_from_1_outflow_accumulated_hour0": ("% outflow WorldPop (accum hour0)", "YlOrRd", False),
    "3_pct_outflow_worldpop_from_1_outflow_max_hour0": ("% outflow WorldPop (max hour0)", "YlOrRd", False),
    # Population / scaling
    "3_worldpop": ("WorldPop population", "Blues", True),
    "3_fb_baseline_median": ("FB baseline median", "Blues", True),
    "3_scaling_ratio": ("Scaling ratio (WorldPop/FB)", "Purples", False),
    # Flood (Step 4)
    "4_flood_p95": ("Flood 95th percentile", "Blues", False),
    "4_flood_mean": ("Flood mean", "Blues", False),
    "4_flood_max": ("Flood max", "Blues", False),
    # Waste
    "4_waste_count": ("Waste count", "YlOrBr", True),
    "4_waste_per_population": ("Waste per 1000 pop", "YlOrBr", False),
    "4_waste_per_svi_count": ("Waste per 1000 SVI", "YlOrBr", False),
    "4_svi_count": ("SVI image count", "Greens", True),
    # Counts / diagnostics (Step 1–2)
    "1_n_timestamps": ("N timestamps", "Greys", False),
    "1_n_valid_timestamps": ("N valid timestamps", "Greys", False),
    "1_n_valid_timestamps_hour0": ("N valid timestamps hour0", "Greys", False),
    "1_n_timestamps_with_outflow": ("N timestamps with outflow", "Greys", False),
    "1_outflow_fraction": ("Outflow fraction", "YlOrRd", False),
    "1_valid_fraction": ("Valid fraction", "Greys", False),
    "1_baseline_median_mean": ("Baseline median mean", "Blues", False),
    "1_crisis_mean": ("Crisis mean", "Blues", False),
    "1_n_low_confidence": ("N low confidence", "Reds", False),
    "2_n_crisis_rows": ("N crisis rows", "Greys", False),
    "2_n_obs": ("N obs", "Greys", False),
    "2_n_low_confidence": ("N low confidence", "Reds", False),
}


def get_mappable_columns(gdf):
    """Return list of numeric column names to map (excluding geometry, quadkey, and optional redundant)."""
    exclude = set(SKIP_COLUMNS)
    if SKIP_REDUNDANT:
        for col in gdf.columns:
            if col in exclude:
                continue
            if any(p in col for p in REDUNDANT_PATTERNS):
                exclude.add(col)
    mappable = []
    for col in gdf.columns:
        if col in exclude:
            continue
        if not np.issubdtype(gdf[col].dtype, np.number):
            continue
        mappable.append(col)
    return sorted(mappable)


def create_choropleth(ax, gdf, col, title, cmap=DEFAULT_CMAP, log_scale=DEFAULT_LOG):
    """Draw a choropleth for column col; use quantile clipping for robust color scale."""
    valid = gdf[col].replace([np.inf, -np.inf], np.nan).dropna()
    if len(valid) == 0:
        gdf.plot(ax=ax, color="lightgrey", edgecolor="none")
        ax.set_title(f"{title}\n(no valid data)", fontsize=FONT_SIZE)
        return

    vmax = float(valid.quantile(QUANTILE_CLIP))
    vmin = float(valid.min())
    # For log scale, avoid zero
    if log_scale and vmin <= 0:
        positive = valid[valid > 0]
        vmin = float(positive.min()) if len(positive) > 0 else 1e-6
    if vmax <= vmin:
        vmax = float(valid.max())
        if vmax <= vmin:
            vmax = vmin + 1

    if log_scale and vmin > 0:
        norm = LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = Normalize(vmin=vmin, vmax=vmax)

    gdf.plot(
        column=col,
        ax=ax,
        cmap=cmap,
        norm=norm,
        legend=False,
        edgecolor="none",
        linewidth=0.05,
        missing_kwds={"color": "lightgrey", "label": "No data"},
    )

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="4%", pad=0.15)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.ax.tick_params(labelsize=FONT_SIZE - 2)
    cbar.set_label(col, fontsize=FONT_SIZE - 1)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=FONT_SIZE + 1, fontweight="bold")


def main():
    print("Loading dataset...")
    if not GPKG_PATH.exists():
        raise FileNotFoundError(f"GPKG not found: {GPKG_PATH}")
    try:
        gdf = gpd.read_file(GPKG_PATH, layer=LAYER_NAME)
    except Exception:
        gdf = gpd.read_file(GPKG_PATH)
    print(f"  Loaded {len(gdf):,} rows, {len(gdf.columns)} columns")

    if FB_BASELINE_MIN is not None and BASELINE_COL in gdf.columns:
        n_before = len(gdf)
        gdf = gdf[gdf[BASELINE_COL].notna() & (gdf[BASELINE_COL] >= FB_BASELINE_MIN)].copy()
        gdf = gdf.reset_index(drop=True)
        print(f"  FB baseline filter ({BASELINE_COL} >= {FB_BASELINE_MIN}): {n_before:,} -> {len(gdf):,} rows")

    cols = get_mappable_columns(gdf)
    print(f"  Mappable columns: {len(cols)}")

    for col in cols:
        cfg = COLUMN_CONFIG.get(col)
        if cfg:
            title, cmap, log_scale = cfg
        else:
            title = col.replace("_", " ").title()
            cmap = DEFAULT_CMAP
            log_scale = DEFAULT_LOG
            # heuristic: log for counts and large-scale values
            if any(x in col.lower() for x in ("count", "worldpop", "outflow", "accumulated", "max", "n_")) and "pct" not in col.lower() and "fraction" not in col.lower():
                log_scale = True

        fig, ax = plt.subplots(1, 1, figsize=FIG_SIZE_SINGLE)
        create_choropleth(ax, gdf, col, title, cmap=cmap, log_scale=log_scale)
        plt.tight_layout()
        safe_name = col.replace("/", "_").replace(" ", "_")
        out_path = OUT_DIR / f"{safe_name}.png"
        fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path.name}")

    print(f"\nDone. All maps saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
