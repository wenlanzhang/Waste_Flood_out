#!/usr/bin/env python3
"""
map_bivariate_worldpop_waste.py

Bivariate map: 3_worldpop (WorldPop population) × 4_waste_count (waste count).

Each variable is classified into tertiles (low / medium / high). The map shows
the 9 combinations with a 3×3 color palette and a 2D legend.

Optional: USE_BASEMAP = True adds a satellite/imagery basemap below and draws
the choropleth at 50% transparency. Requires contextily (and xyzservices for
Esri World Imagery). Install: pip install contextily

Input:  Data/4/4_flood_waste_metrics_quadkey.gpkg (layer: 4_flood_waste_metrics_quadkey)
Output: Figure/bivariate_worldpop_waste.png

Requirements: geopandas, matplotlib, numpy, pandas, pathlib; optional: contextily
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
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap, BoundaryNorm

# -------------------- USER PARAMETERS --------------------
GPKG_PATH = Path("/Users/wenlanzhang/PycharmProjects/Waste_Flood_out/Data/4/4_flood_waste_metrics_quadkey.gpkg")
LAYER_NAME = "4_flood_waste_metrics_quadkey"
OUT_DIR = Path("/Users/wenlanzhang/PycharmProjects/Waste_Flood_out/Figure")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "bivariate_worldpop_waste.png"

COL_WORLDPOP = "3_worldpop"
COL_WASTE = "4_waste_count"

# FB baseline filter (set None to use all rows)
FB_BASELINE_MIN = None
BASELINE_COL = "3_fb_baseline_median"

# Classification: number of bins per variable (N_BINS × N_BINS = number of classes)
N_BINS = 3

# 3×3 bivariate palette: [row][col] = row = WorldPop tertile (0=low, 2=high), col = Waste tertile (0=low, 2=high)
# Order: (worldpop_low, waste_low) -> (worldpop_high, waste_high). Light to dark.
BIVAR_PALETTE_3X3 = [
    ["#e8f4f8", "#a8d4e0", "#4a9ec2"],   # worldpop low:   waste low -> high (light blue -> blue)
    ["#b8e0c8", "#7bc090", "#3d8058"],   # worldpop mid:   waste low -> high (light green -> green)
    ["#f5d976", "#e8a838", "#c85a5a"],   # worldpop high:  waste low -> high (yellow -> orange -> red)
]
# Flatten row-major: class = worldpop_bin * 3 + waste_bin
BIVAR_COLORS = [BIVAR_PALETTE_3X3[r][c] for r in range(3) for c in range(3)]

FIG_SIZE = (12, 10)
DPI = 200
FONT_SIZE = 11
NODATA_COLOR = "lightgrey"

# Basemap: True = add satellite/imagery tile below and draw choropleth at 50% transparency
USE_BASEMAP = True
CHOROPLETH_ALPHA = 0.5
# Satellite-style source: use Esri.WorldImagery if available, else OSM
try:
    import contextily as cx
    try:
        BASEMAP_SOURCE = cx.providers.Esri.WorldImagery
    except AttributeError:
        try:
            import xyzservices.providers as xyz
            BASEMAP_SOURCE = xyz.Esri.WorldImagery
        except Exception:
            BASEMAP_SOURCE = None  # fallback to no basemap
    HAS_CONTEXTILY = True
except ImportError:
    HAS_CONTEXTILY = False
    BASEMAP_SOURCE = None


def load_data():
    """Load GPKG and optional baseline filter."""
    if not GPKG_PATH.exists():
        raise FileNotFoundError(f"GPKG not found: {GPKG_PATH}")
    try:
        gdf = gpd.read_file(GPKG_PATH, layer=LAYER_NAME)
    except Exception:
        gdf = gpd.read_file(GPKG_PATH)
    for c in (COL_WORLDPOP, COL_WASTE):
        if c not in gdf.columns:
            raise ValueError(f"Column not found: {c}")
    if FB_BASELINE_MIN is not None and BASELINE_COL in gdf.columns:
        gdf = gdf[gdf[BASELINE_COL].notna() & (gdf[BASELINE_COL] >= FB_BASELINE_MIN)].copy()
    return gdf


def classify_bivariate(gdf):
    """
    Classify 3_worldpop and 4_waste_count into N_BINS each; return class index 0..(N_BINS²-1).
    Uses quantile-based bins. NaN/invalid -> class -1 (no data).
    """
    wp = pd.to_numeric(gdf[COL_WORLDPOP], errors="coerce").replace(0, np.nan)
    waste = pd.to_numeric(gdf[COL_WASTE], errors="coerce")
    # Replace NODATA sentinel if present
    if (waste == -9999).any():
        waste = waste.replace(-9999, np.nan)
    valid = wp.notna() & waste.notna()
    out = np.full(len(gdf), -1, dtype=int)
    if not valid.any():
        return out
    wp_valid = wp.loc[valid]
    waste_valid = waste.loc[valid]
    # Quantile-based bins (0, 1, 2 for N_BINS=3)
    wp_q = wp_valid.quantile(np.linspace(0, 1, N_BINS + 1)[1:-1])
    waste_q = waste_valid.quantile(np.linspace(0, 1, N_BINS + 1)[1:-1])
    wp_bin = np.digitize(wp_valid, wp_q.values)  # 0, 1, or 2 (or 3 at max edge; clip to 2)
    waste_bin = np.digitize(waste_valid, waste_q.values)
    wp_bin = np.clip(wp_bin, 0, N_BINS - 1)
    waste_bin = np.clip(waste_bin, 0, N_BINS - 1)
    class_idx = wp_bin * N_BINS + waste_bin
    out[valid] = class_idx
    return out


def add_bivariate_legend(ax_legend, gdf, class_col):
    """Add a 3×3 legend: rows = WorldPop tertile, cols = Waste tertile."""
    # Get approximate break values from data for legend labels
    wp = gdf[COL_WORLDPOP].dropna()
    waste = gdf[COL_WASTE].replace(-9999, np.nan).dropna()
    wp_q = wp.quantile([0, 1/3, 2/3, 1]).values
    waste_q = waste.quantile([0, 1/3, 2/3, 1]).values
    ax_legend.set_xlim(0, N_BINS)
    ax_legend.set_ylim(0, N_BINS)
    ax_legend.set_aspect("equal")
    dx, dy = 1.0, 1.0
    for r in range(N_BINS):
        for c in range(N_BINS):
            idx = r * N_BINS + c
            color = BIVAR_COLORS[idx] if idx < len(BIVAR_COLORS) else "gray"
            rect = plt.Rectangle((c, N_BINS - 1 - r), dx, dy, facecolor=color, edgecolor="black", linewidth=0.5)
            ax_legend.add_patch(rect)
    ax_legend.set_xticks(np.arange(N_BINS) + 0.5)
    ax_legend.set_yticks(np.arange(N_BINS) + 0.5)
    ax_legend.set_xticklabels([f"Low\n(≤{waste_q[1]:.0f})", f"Medium\n({waste_q[1]:.0f}–{waste_q[2]:.0f})", f"High\n(>{waste_q[2]:.0f})"], fontsize=FONT_SIZE - 2)
    ax_legend.set_yticklabels([f"High\n(>{wp_q[2]:.0f})", f"Medium\n({wp_q[1]:.0f}–{wp_q[2]:.0f})", f"Low\n(≤{wp_q[1]:.0f})"], fontsize=FONT_SIZE - 2)
    ax_legend.set_xlabel("Waste count (4_waste_count)", fontsize=FONT_SIZE)
    ax_legend.set_ylabel("WorldPop (3_worldpop)", fontsize=FONT_SIZE)
    ax_legend.set_title("Bivariate legend", fontsize=FONT_SIZE + 1, fontweight="bold")


def main():
    print("Loading data...")
    gdf = load_data()
    print(f"  Rows: {len(gdf):,}")

    print("Classifying bivariate (WorldPop × Waste count)...")
    classes = classify_bivariate(gdf)
    n_valid = (classes >= 0).sum()
    print(f"  Valid cells: {n_valid:,} (no data: {len(gdf) - n_valid:,})")

    gdf = gdf.copy()
    gdf["_bivar_class"] = classes

    # For basemap tiles use Web Mercator (EPSG:3857)
    gdf_3857 = gdf.to_crs(epsg=3857)
    gdf["_plot_class"] = gdf["_bivar_class"] + 1  # 0 = no data, 1..9 = classes
    gdf_3857["_plot_class"] = gdf["_plot_class"]

    fig = plt.figure(figsize=FIG_SIZE)
    ax_map = fig.add_axes([0.05, 0.25, 0.7, 0.7])   # main map
    ax_leg = fig.add_axes([0.78, 0.35, 0.2, 0.5])   # legend

    # Set extent from data so basemap knows what to fetch
    xmin, ymin, xmax, ymax = gdf_3857.total_bounds
    ax_map.set_xlim(xmin, xmax)
    ax_map.set_ylim(ymin, ymax)
    ax_map.set_aspect("equal")

    # 1) Draw basemap first (below), then 2) choropleth on top with alpha
    if USE_BASEMAP and HAS_CONTEXTILY and BASEMAP_SOURCE is not None:
        try:
            cx.add_basemap(ax_map, source=BASEMAP_SOURCE, crs=gdf_3857.crs, zoom="auto", alpha=1)
        except Exception as e:
            print(f"  Basemap failed ({e}); drawing choropleth only.")
    elif USE_BASEMAP and not HAS_CONTEXTILY:
        print("  contextily not installed; skipping basemap. pip install contextily for satellite basemap.")

    n_cmap = 1 + len(BIVAR_COLORS)
    cmap = ListedColormap([NODATA_COLOR] + BIVAR_COLORS)
    bounds = np.arange(-0.5, n_cmap + 0.5, 1.0)
    norm = BoundaryNorm(bounds, cmap.N)
    gdf_3857.plot(
        column="_plot_class",
        ax=ax_map,
        cmap=cmap,
        norm=norm,
        legend=False,
        edgecolor="none",
        linewidth=0.05,
        alpha=CHOROPLETH_ALPHA if (USE_BASEMAP and HAS_CONTEXTILY and BASEMAP_SOURCE) else 1,
        missing_kwds={"color": NODATA_COLOR},
    )

    ax_map.set_xticks([])
    ax_map.set_yticks([])
    title = "Bivariate map: WorldPop (3_worldpop) × Waste count (4_waste_count)"
    if USE_BASEMAP and HAS_CONTEXTILY and BASEMAP_SOURCE:
        title += f" — basemap + {int(CHOROPLETH_ALPHA * 100)}% transparency"
    ax_map.set_title(title, fontsize=FONT_SIZE + 2, fontweight="bold")

    add_bivariate_legend(ax_leg, gdf, "_bivar_class")

    plt.savefig(OUT_PATH, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
