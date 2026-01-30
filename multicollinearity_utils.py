"""
multicollinearity_utils.py

Shared multicollinearity diagnostics (VIF, correlation matrix, condition number)
and publication figures. Used by model_flood, model_wasteflood, model_floodpop,
model_wastefloodpop.

Outputs:
  - CSV and .txt -> output_dir / "multicollinearity" (e.g. Output/flood/multicollinearity/)
  - Figures (PNG) -> figure_dir (e.g. Figure/flood/)

Usage:
  run_multicollinearity_diagnostics(
      pred_cols=[...],
      df=df_clean,
      figure_dir=FIGURE_DIR,
      output_dir=OUT_DIR,   # CSV/txt go to output_dir/multicollinearity/
      file_id=None
  )

If len(pred_cols) < 2, only a short note is written; no VIF/figures.
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DPI = 300
FONT_SIZE = 11
MAX_LABEL_LEN = 22
VIF_MAX_DISPLAY = 15

LABEL_MAP = {
    "4_flood_p95": "Flood (95th %ile)",
    "4_flood_mean": "Flood (mean)",
    "4_flood_max": "Flood (max)",
    "4_waste_count": "Waste count",
    "4_waste_per_population": "Waste per pop",
    "4_waste_per_svi_count": "Waste per SVI",
    "3_worldpop": "Population",
    "3_fb_baseline_median": "FB baseline",
}

_SHORTEN_PATTERNS = [
    ("estimated_outflow_pop_from_2_outflow_max", "Outflow pop (2 max)"),
    ("outflow_max", "Outflow max"),
    ("fb_baseline_median", "FB baseline"),
    ("flood_p95", "Flood p95"),
    ("waste_count", "Waste count"),
    ("worldpop", "WorldPop"),
]


def get_label(col):
    """Short, publication-ready label for a column name."""
    if col in LABEL_MAP:
        return LABEL_MAP[col]
    rest = col
    for prefix in ("1_", "2_", "3_", "4_"):
        if rest.startswith(prefix):
            rest = rest[len(prefix) :]
            break
    for pattern, replacement in _SHORTEN_PATTERNS:
        if rest == pattern:
            return replacement
    return rest.replace("_", " ")


def _fig_label(label):
    if len(label) <= MAX_LABEL_LEN:
        return label
    return label[: MAX_LABEL_LEN - 1] + "…"


def _vif_one_j(X, j):
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
    """VIF for each column (numpy only; X_df has no constant)."""
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
    """Condition number of design matrix (columns standardized)."""
    X = np.asarray(X_df, dtype=float)
    X = (X - np.nanmean(X, axis=0)) / (np.nanstd(X, axis=0) + 1e-12)
    X = np.nan_to_num(X, nan=0.0)
    _, s, _ = np.linalg.svd(X, full_matrices=False)
    return s[0] / (s[-1] + 1e-12)


def run_multicollinearity_diagnostics(pred_cols, df, figure_dir, output_dir, file_id=None):
    """
    Run multicollinearity diagnostics. CSV and .txt -> output_dir/multicollinearity/;
    figures (PNG) -> figure_dir. Uses complete cases of df for pred_cols.

    pred_cols: list of predictor column names (must be in df).
    df: dataframe with at least pred_cols.
    figure_dir: Path; figures saved here.
    output_dir: Path; CSV and txt saved to output_dir / "multicollinearity".
    file_id: optional string (e.g. from wastefloodpop); appended to filenames.
    """
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path(output_dir) / "multicollinearity"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{file_id}" if file_id else ""

    # Restrict to columns that exist
    pred_cols = [c for c in pred_cols if c in df.columns]
    if len(pred_cols) < 2:
        note_path = out_dir / f"multicollinearity_note{suffix}.txt"
        with open(note_path, "w") as f:
            f.write(f"Multicollinearity: only {len(pred_cols)} predictor(s) ({', '.join(pred_cols)}). VIF/figures require ≥2 predictors.\n")
        print(f"Multicollinearity: <2 predictors; note saved to {note_path}")
        return

    X = df[pred_cols].dropna(how="any").astype(float)
    n_obs = len(X)
    if n_obs < 2:
        note_path = out_dir / f"multicollinearity_note{suffix}.txt"
        with open(note_path, "w") as f:
            f.write("Multicollinearity: insufficient complete cases for predictor set.\n")
        print(f"Multicollinearity: insufficient complete cases; note saved to {note_path}")
        return

    # ----- Correlation matrix -----
    corr = X.corr(method="pearson")
    corr.to_csv(out_dir / f"correlation_matrix{suffix}.csv")
    print(f"Saved: {out_dir / f'correlation_matrix{suffix}.csv'}")

    # ----- VIF and tolerance -----
    vif_df = compute_vif(X)
    vif_df.to_csv(out_dir / f"vif_tolerance{suffix}.csv", index=False)
    print(f"Saved: {out_dir / f'vif_tolerance{suffix}.csv'}")

    # ----- Condition number -----
    cond = condition_number(X)
    pd.DataFrame([{"Condition_number": cond}]).to_csv(out_dir / f"condition_number{suffix}.csv", index=False)
    print(f"Condition number (scaled X): {cond:.2f}")
    print(f"Saved: {out_dir / f'condition_number{suffix}.csv'}")

    # ----- Summary text -----
    high_vif = vif_df[vif_df["VIF"] > 10]
    mod_vif = vif_df[(vif_df["VIF"] > 5) & (vif_df["VIF"] <= 10)]
    interpretation = []
    if len(high_vif) > 0:
        interpretation.append(
            f"  {len(high_vif)} predictor(s) with VIF > 10 (severe): "
            + ", ".join(high_vif["variable"].tolist())
        )
        interpretation.append("  Recommendation: do not include all of these in the same model; use one per construct.")
    if len(mod_vif) > 0:
        interpretation.append(
            f"  {len(mod_vif)} predictor(s) with 5 < VIF <= 10 (moderate): "
            + ", ".join(mod_vif["variable"].tolist())
        )
    if cond > 100:
        interpretation.append(f"  Condition number {cond:.1f} indicates severe ill-conditioning.")
    elif cond > 30:
        interpretation.append(f"  Condition number {cond:.1f} indicates ill-conditioning.")
    if not interpretation:
        interpretation.append("  No severe multicollinearity detected (all VIF <= 10, condition number acceptable).")

    lines = [
        "Multicollinearity diagnostics (model step)",
        f"Predictors: {', '.join(pred_cols)}",
        f"N (complete cases): {n_obs}",
        "",
        "VIF (Variance Inflation Factor):",
        "  VIF > 10: severe; VIF > 5: moderate (conservative).",
        "",
        vif_df.to_string(index=False),
        "",
        f"Condition number (scaled X): {cond:.2f}",
        "  > 30: ill-conditioning; > 100: severe.",
        "",
        "Interpretation:",
        *interpretation,
    ]
    summary_path = out_dir / f"multicollinearity_summary{suffix}.txt"
    with open(summary_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved: {summary_path}")

    # ----- Publication figures -----
    plt.rcParams["font.size"] = FONT_SIZE
    n_pred = len(corr.columns)
    heatmap_size = (max(8, n_pred * 0.4), max(6.5, n_pred * 0.4))
    tick_fontsize = 7 if n_pred > 20 else (8 if n_pred > 12 else 9)
    vif_fig_height = max(4, n_pred * 0.2)

    labels_heatmap = [_fig_label(get_label(c)) for c in corr.columns]
    C = corr.values
    mask_upper = np.triu(np.ones_like(C, dtype=bool), k=1)
    C_masked = np.where(mask_upper, np.nan, C)

    # Figure 1: Correlation heatmap
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
    plt.colorbar(im, ax=ax1, shrink=0.8, label="Pearson r")
    ax1.set_title("Predictor correlation matrix (Pearson)")
    plt.tight_layout()
    fig1.savefig(figure_dir / f"multicollinearity_correlation_heatmap{suffix}.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig1)
    print(f"Saved: {figure_dir / f'multicollinearity_correlation_heatmap{suffix}.png'}")

    # Figure 2: VIF bar chart
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
    fig2.savefig(figure_dir / f"multicollinearity_vif_bars{suffix}.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved: {figure_dir / f'multicollinearity_vif_bars{suffix}.png'}")

    # Figure 3: Two-panel (heatmap + VIF)
    _w = max(20, n_pred * 0.7)
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
    x_pos = np.arange(len(vif_plot["label"]))
    colors2 = ["#c0392b" if v > 10 else "#e67e22" if v > 5 else "#27ae60" for v in vif_plot["VIF"]]
    ax3b.bar(x_pos, vif_display, color=colors2, edgecolor="gray", linewidth=0.5)
    ax3b.set_xticks(x_pos)
    ax3b.set_xticklabels(vif_plot["label"], rotation=45, ha="right", fontsize=tick_fontsize)
    ax3b.axhline(5, color="gray", linestyle="--", linewidth=1, label="VIF = 5")
    ax3b.axhline(10, color="gray", linestyle=":", linewidth=1, label="VIF = 10")
    ax3b.set_ylabel("VIF")
    ax3b.set_title("B. Variance Inflation Factor")
    ax3b.set_ylim(0, VIF_MAX_DISPLAY if (vif_plot["VIF"] > VIF_MAX_DISPLAY).any() else None)
    if not (vif_plot["VIF"] > VIF_MAX_DISPLAY).any():
        ax3b.set_ylim(bottom=0)
    ax3b.legend(loc="upper right")
    plt.tight_layout()
    fig3.savefig(figure_dir / f"multicollinearity_two_panel{suffix}.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig3)
    print(f"Saved: {figure_dir / f'multicollinearity_two_panel{suffix}.png'}")

    print("Multicollinearity CSV/txt saved to:", out_dir)
    print("Multicollinearity figures saved to:", figure_dir)
