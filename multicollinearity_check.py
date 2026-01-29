#!/usr/bin/env python3
"""
multicollinearity_check.py

Publication-ready multicollinearity diagnostics for predictors from 4flood_waste.py output.

Uses: Data/4/4_flood_waste_metrics_quadkey.gpkg

Predictors:
  - Default (PREDICTOR_COLS = None): all numeric columns from steps 1–4 (prefix 1_, 2_, 3_, 4_).
  - Manual: set PREDICTOR_COLS to a list of column names to check only those.

Outputs:
  - Output/multicollinearity/: correlation matrix (CSV), VIF and tolerance (CSV), condition number, summary text
  - Figure/multicollinearity/: correlation heatmap, VIF bar chart (publication quality, 300 DPI)

Interpretation:
  - VIF > 10: severe multicollinearity; VIF > 5: moderate (conservative)
  - Condition number (scaled X) > 30: ill-conditioning; > 100: severe
  - |r| > 0.7: high correlation between predictors

Requirements: geopandas, pandas, numpy, matplotlib (no statsmodels or seaborn).
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
import matplotlib.colors as mcolors

# -------------------- USER PARAMETERS --------------------
BASE_DIR = Path("/Users/wenlanzhang/PycharmProjects/Waste_Flood_out/")
GPKG_PATH = BASE_DIR / "Data/4/4_flood_waste_metrics_quadkey.gpkg"
LAYER_NAME = "4_flood_waste_metrics_quadkey"

OUT_DIR = BASE_DIR / "Output/multicollinearity"
FIG_DIR = BASE_DIR / "Figure/multicollinearity"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

NODATA = -9999.0
DPI = 300
FONT_SIZE = 11

# Predictor columns to check:
#   None = use ALL numeric columns from steps 1–4 (prefix 1_, 2_, 3_, 4_) present in the GPKG.
PREDICTOR_COLS = None

#   Or set to a list of column names:
# PREDICTOR_COLS = ["4_flood_p95", "4_waste_count", "3_worldpop"]

# Short labels for figures (publication-ready). Columns not listed use get_label() shortening.
LABEL_MAP = {
    "4_flood_p95": "Flood (95th %ile)",
    "4_flood_mean": "Flood (mean)",
    "4_flood_max": "Flood (max)",
    "4_waste_count": "Waste count",
    "4_waste_per_population": "Waste per pop",
    "4_waste_per_svi_count": "Waste per SVI",
    "3_worldpop": "Population",
}

# Max characters for axis labels in figures (longer labels truncated with "…").
MAX_LABEL_LEN = 22


def _is_numeric_column(gdf, col):
    """True if column can be used as numeric (coerce to float, at least one non-nan)."""
    try:
        s = pd.to_numeric(gdf[col], errors="coerce")
        return s.notna().any()
    except Exception:
        return False


def get_predictor_columns(gdf):
    """
    Return list of predictor columns to check.
    If PREDICTOR_COLS is set (non-empty list), use only those that exist.
    Else use all columns from steps 1–4 (prefix 1_, 2_, 3_, 4_) that are numeric.
    """
    if PREDICTOR_COLS is not None and len(PREDICTOR_COLS) > 0:
        return sorted([c for c in PREDICTOR_COLS if c in gdf.columns])
    # Auto: all 1_, 2_, 3_, 4_ columns that are numeric; exclude geometry/quadkey and vars not used as X or Y in modelling
    skip = {
        "geometry", "quadkey", "4_waste_per_quadkey_area", "4_quadkey_area_m2",
        "3_scaling_ratio", "3_used_global_ratio", "3_ratio_was_capped",
        "2_n_low_confidence", "2_n_obs",
        "3_worldpop_safe",  # same as 3_worldpop except 0→NaN; keep 3_worldpop only
        "3_fb_baseline_safe",  # same as 3_fb_baseline_median except 0→NaN; keep 3_fb_baseline_median only
        "4_waste_count_final",  # same as 4_waste_count but -9999 where no SVI; keep 4_waste_count only
    }
    prefix_cols = [
        c for c in gdf.columns
        if c not in skip and (c.startswith("1_") or c.startswith("2_") or c.startswith("3_") or c.startswith("4_"))
    ]
    numeric = [c for c in prefix_cols if _is_numeric_column(gdf, c)]
    return sorted(numeric)


# Patterns to shorten long column names (applied after stripping 1_/2_/3_/4_).
# Order matters: longer/more specific first.
_SHORTEN_PATTERNS = [
    ("estimated_outflow_pop_from_1_outflow_accumulated_hour0", "Outflow pop (1 acc)"),
    ("estimated_outflow_pop_from_1_outflow_max_hour0", "Outflow pop (1 max)"),
    ("estimated_outflow_pop_from_2_outflow_max", "Outflow pop (2 max)"),
    ("estimated_excess_displacement_pop", "Excess disp. pop"),
    ("pct_outflow_worldpop_from_1_outflow_accumulated_hour0", "Pct WP (1 acc)"),
    ("pct_outflow_worldpop_from_1_outflow_max_hour0", "Pct WP (1 max)"),
    ("pct_outflow_worldpop_from_2_outflow_max", "Pct WP (2 max)"),
    ("pct_outflow_fb_from_1_outflow_accumulated_hour0", "Pct FB (1 acc)"),
    ("pct_outflow_fb_from_1_outflow_max_hour0", "Pct FB (1 max)"),
    ("pct_outflow_fb_from_2_outflow_max", "Pct FB (2 max)"),
    ("outflow_accumulated_hour0", "Outflow acc h0"),
    ("outflow_max_hour0", "Outflow max h0"),
    ("displaced_excess_max", "Disp. excess max"),
    ("outflow_max", "Outflow max"),
    ("fb_baseline_median", "FB baseline"),
    ("scaling_ratio", "Scaling ratio"),
    ("ratio_was_capped", "Ratio capped"),
    ("used_global_ratio", "Global ratio"),
    ("svi_count", "SVI count"),
    ("n_crisis_rows", "N crisis"),
    ("n_low_confidence", "N low conf"),
    ("n_obs", "N obs"),
    ("change_cv", "Ch. CV"),
    ("change_iqr", "Ch. IQR"),
    ("change_max", "Ch. max"),
    ("change_mean", "Ch. mean"),
    ("change_min", "Ch. min"),
    ("change_range", "Ch. range"),
    ("change_std", "Ch. std"),
]


def get_label(col):
    """Short, publication-ready label. Uses LABEL_MAP, else pattern shortening, else strip prefix + replace _."""
    if col in LABEL_MAP:
        return LABEL_MAP[col]
    # Strip step prefix
    rest = col
    for prefix in ("1_", "2_", "3_", "4_"):
        if rest.startswith(prefix):
            rest = rest[len(prefix):]
            break
    # Apply shorten patterns (exact match on rest)
    for pattern, replacement in _SHORTEN_PATTERNS:
        if rest == pattern:
            return replacement
    # Default: replace underscores with spaces
    return rest.replace("_", " ")


def _fig_label(label):
    """Truncate label for figure axis if longer than MAX_LABEL_LEN."""
    if len(label) <= MAX_LABEL_LEN:
        return label
    return label[: MAX_LABEL_LEN - 1] + "…"


def replace_nodata(df, cols):
    """Replace NODATA with NaN for given columns."""
    out = df[cols].copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
        out.loc[out[c] == NODATA, c] = np.nan
    return out


def _vif_one_j(X, j):
    """VIF for column j: 1 / (1 - R^2) from regressing X[:, j] on other columns."""
    n, p = X.shape
    if p < 2:
        return 1.0
    y = X[:, j]
    idx = [i for i in range(p) if i != j]
    X_other = X[:, idx]
    X_other = np.column_stack([np.ones(n), X_other])
    try:
        beta, _, _, _ = np.linalg.lstsq(X_other, y, rcond=None)
        yhat = X_other @ beta
        ss_res = np.sum((y - yhat) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1.0 - (ss_res / (ss_tot + 1e-20))
        return 1.0 / (1.0 - r2) if r2 < 1.0 else np.inf
    except Exception:
        return np.nan


def compute_vif(X_df):
    """VIF for each column. X_df should not contain constant. Uses numpy only."""
    X = np.asarray(X_df.astype(float), dtype=float)
    vif_list = []
    for j, col in enumerate(X_df.columns):
        vif = _vif_one_j(X, j)
        vif_list.append({
            "variable": col,
            "VIF": vif,
            "Tolerance": 1.0 / vif if np.isfinite(vif) and vif > 0 else np.nan,
        })
    return pd.DataFrame(vif_list)


def condition_number(X_df):
    """Condition number of design matrix (after standardizing columns)."""
    X = np.asarray(X_df, dtype=float)
    X = (X - np.nanmean(X, axis=0)) / (np.nanstd(X, axis=0) + 1e-12)
    X = np.nan_to_num(X, nan=0.0)
    _, s, _ = np.linalg.svd(X, full_matrices=False)
    return s[0] / (s[-1] + 1e-12)


def main():
    print("=" * 60)
    print("Multicollinearity check (4flood_waste main output)")
    print("=" * 60)

    # Load data
    if not GPKG_PATH.exists():
        raise FileNotFoundError(f"GPKG not found: {GPKG_PATH}")
    gdf = gpd.read_file(GPKG_PATH, layer=LAYER_NAME)
    pred_cols = get_predictor_columns(gdf)
    if len(pred_cols) < 2:
        raise ValueError("Need at least 2 predictor columns for multicollinearity check.")

    print(f"\nPredictors included: {pred_cols}")
    df = replace_nodata(gdf, pred_cols)
    df = df[pred_cols].dropna(how="any")
    n_obs = len(df)
    print(f"Complete cases (after dropping NaN/NODATA): {n_obs}")

    X = df.astype(float)

    # ----- 1. Correlation matrix -----
    corr = X.corr(method="pearson")
    corr.to_csv(OUT_DIR / "correlation_matrix.csv")
    print(f"\nSaved: {OUT_DIR / 'correlation_matrix.csv'}")

    # ----- 2. VIF and tolerance -----
    vif_df = compute_vif(X)
    vif_df.to_csv(OUT_DIR / "vif_tolerance.csv", index=False)
    print(f"Saved: {OUT_DIR / 'vif_tolerance.csv'}")

    # ----- 3. Condition number -----
    cond = condition_number(X)
    cond_df = pd.DataFrame([{"Condition_number": cond}])
    cond_df.to_csv(OUT_DIR / "condition_number.csv", index=False)
    print(f"Condition number (scaled design matrix): {cond:.2f}")
    print(f"Saved: {OUT_DIR / 'condition_number.csv'}")

    # ----- 4. Summary text with interpretation -----
    high_vif = vif_df[vif_df["VIF"] > 10]
    mod_vif = vif_df[(vif_df["VIF"] > 5) & (vif_df["VIF"] <= 10)]
    interpretation = []
    if len(high_vif) > 0:
        interpretation.append(
            f"  {len(high_vif)} predictor(s) with VIF > 10 (severe): "
            + ", ".join(high_vif["variable"].tolist())
        )
        interpretation.append("  Recommendation: do not include all of these in the same model; use one per construct (e.g. one flood, one waste, one population).")
    if len(mod_vif) > 0:
        interpretation.append(
            f"  {len(mod_vif)} predictor(s) with 5 < VIF <= 10 (moderate): "
            + ", ".join(mod_vif["variable"].tolist())
        )
    if cond > 100:
        interpretation.append(f"  Condition number {cond:.1f} indicates severe ill-conditioning; avoid using all predictors jointly.")
    elif cond > 30:
        interpretation.append(f"  Condition number {cond:.1f} indicates ill-conditioning.")
    if not interpretation:
        interpretation.append("  No severe multicollinearity detected among predictors (all VIF <= 10, condition number acceptable).")

    lines = [
        "Multicollinearity diagnostics",
        "Data: 4_flood_waste_metrics_quadkey.gpkg",
        f"Predictors: {', '.join(pred_cols)}",
        f"N (complete cases): {n_obs}",
        "",
        "VIF (Variance Inflation Factor):",
        "  VIF > 10: severe multicollinearity; consider dropping or combining predictors.",
        "  VIF > 5:  moderate multicollinearity (conservative threshold).",
        "",
        vif_df.to_string(index=False),
        "",
        f"Condition number (scaled X): {cond:.2f}",
        "  > 30:  ill-conditioning; > 100: severe.",
        "",
        "Interpretation:",
        *interpretation,
        "",
        "Outputs: correlation_matrix.csv, vif_tolerance.csv, condition_number.csv",
        "Figures: Figure/multicollinearity/ (correlation heatmap, VIF bar chart, two-panel).",
    ]
    summary_path = OUT_DIR / "multicollinearity_summary.txt"
    with open(summary_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved: {summary_path}")

    # ----- 5. Publication figures -----
    n_pred = len(corr.columns)
    plt.rcParams["font.size"] = FONT_SIZE
    # Scale figure and tick font when many predictors
    heatmap_size = (max(8, n_pred * 0.4), max(6.5, n_pred * 0.4))
    tick_fontsize = 7 if n_pred > 20 else (8 if n_pred > 12 else 9)
    vif_fig_height = max(4, n_pred * 0.2)

    # Figure 1: Correlation heatmap (matplotlib only); use short labels for axes
    labels_heatmap = [_fig_label(get_label(c)) for c in corr.columns]
    corr_renamed = corr.copy()
    corr_renamed.columns = labels_heatmap
    corr_renamed.index = labels_heatmap
    C = corr_renamed.values
    mask_upper = np.triu(np.ones_like(C, dtype=bool), k=1)
    C_masked = np.where(mask_upper, np.nan, C)  # show lower triangle + diagonal

    fig1, ax1 = plt.subplots(figsize=heatmap_size)
    im = ax1.imshow(C_masked, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    for i in range(len(labels_heatmap)):
        for j in range(len(labels_heatmap)):
            if not mask_upper[i, j]:
                val = C[i, j]
                color = "white" if abs(val) > 0.5 else "black"
                ax1.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=tick_fontsize - 1, color=color)
    ax1.set_xticks(range(len(labels_heatmap)))
    ax1.set_yticks(range(len(labels_heatmap)))
    ax1.set_xticklabels(labels_heatmap, rotation=45, ha="right", fontsize=tick_fontsize)
    ax1.set_yticklabels(labels_heatmap, fontsize=tick_fontsize)
    cbar = plt.colorbar(im, ax=ax1, shrink=0.8)
    cbar.set_label("Pearson r")
    ax1.set_title("Predictor correlation matrix (Pearson)")
    plt.tight_layout()
    fig1.savefig(FIG_DIR / "multicollinearity_correlation_heatmap.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig1)
    print(f"Saved: {FIG_DIR / 'multicollinearity_correlation_heatmap.png'}")

    # Figure 2: VIF bar chart (cap display at VIF_MAX so extreme values don't dominate)
    VIF_MAX_DISPLAY = 15  # cap for readability; full values in CSV
    vif_plot = vif_df.copy()
    vif_plot["label"] = vif_plot["variable"].map(lambda c: _fig_label(get_label(c)))
    vif_display = np.minimum(vif_plot["VIF"].values, VIF_MAX_DISPLAY)

    fig2, ax2 = plt.subplots(figsize=(7, vif_fig_height))
    colors = ["#c0392b" if v > 10 else "#e67e22" if v > 5 else "#27ae60" for v in vif_plot["VIF"]]
    ax2.barh(vif_plot["label"], vif_display, color=colors, edgecolor="gray", linewidth=0.5)
    ax2.tick_params(axis="y", labelsize=tick_fontsize)
    ax2.axvline(5, color="gray", linestyle="--", linewidth=1, label="VIF = 5")
    ax2.axvline(10, color="gray", linestyle=":", linewidth=1, label="VIF = 10")
    ax2.set_xlabel("Variance Inflation Factor")
    ax2.set_title("Multicollinearity: VIF by predictor")
    if (vif_plot["VIF"] > VIF_MAX_DISPLAY).any():
        ax2.set_xlim(0, VIF_MAX_DISPLAY)
        ax2.text(0.98, 0.02, "VIF > 15 truncated; see CSV for full values", transform=ax2.transAxes, fontsize=8, ha="right", va="bottom")
    else:
        ax2.set_xlim(left=0)
    ax2.legend(loc="lower right")
    plt.tight_layout()
    fig2.savefig(FIG_DIR / "multicollinearity_vif_bars.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved: {FIG_DIR / 'multicollinearity_vif_bars.png'}")

    # Optional: single two-panel figure for supplement (horizontal: A left, B right)
    _w = max(20, n_pred * 0.7)   # wider (longer in x) for readability
    _h = max(6, n_pred * 0.28)
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(_w, _h))

    im3 = ax3a.imshow(C_masked, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    for i in range(len(labels_heatmap)):
        for j in range(len(labels_heatmap)):
            if not mask_upper[i, j]:
                val = C[i, j]
                color = "white" if abs(val) > 0.5 else "black"
                ax3a.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=tick_fontsize - 1, color=color)
    ax3a.set_xticks(range(len(labels_heatmap)))
    ax3a.set_yticks(range(len(labels_heatmap)))
    ax3a.set_xticklabels(labels_heatmap, rotation=45, ha="right", fontsize=tick_fontsize)
    ax3a.set_yticklabels(labels_heatmap, fontsize=tick_fontsize)
    plt.colorbar(im3, ax=ax3a, shrink=0.8, label="Pearson r")
    ax3a.set_title("A. Predictor correlations")

    # Panel B: vertical bar chart (labels at bottom, bars go up)
    colors2 = ["#c0392b" if v > 10 else "#e67e22" if v > 5 else "#27ae60" for v in vif_plot["VIF"]]
    x_pos = np.arange(len(vif_plot["label"]))
    ax3b.bar(x_pos, vif_display, color=colors2, edgecolor="gray", linewidth=0.5)
    ax3b.set_xticks(x_pos)
    ax3b.set_xticklabels(vif_plot["label"], rotation=45, ha="right", fontsize=tick_fontsize)
    ax3b.axhline(5, color="gray", linestyle="--", linewidth=1, label="VIF = 5")
    ax3b.axhline(10, color="gray", linestyle=":", linewidth=1, label="VIF = 10")
    ax3b.set_ylabel("VIF")
    ax3b.set_title("B. Variance Inflation Factor")
    if (vif_plot["VIF"] > VIF_MAX_DISPLAY).any():
        ax3b.set_ylim(0, VIF_MAX_DISPLAY)
    else:
        ax3b.set_ylim(bottom=0)
    ax3b.legend(loc="upper right")
    plt.tight_layout()
    fig3.savefig(FIG_DIR / "multicollinearity_two_panel.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig3)
    print(f"Saved: {FIG_DIR / 'multicollinearity_two_panel.png'}")

    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
