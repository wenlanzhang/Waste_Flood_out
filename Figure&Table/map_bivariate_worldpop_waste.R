#!/usr/bin/env Rscript
#
# map_bivariate_worldpop_waste.R
#
# Bivariate map: 3_worldpop (WorldPop population) x 4_waste_count (waste count).
# Uses the 'biscale' package for classification, built-in palettes, and 2D legend.
#
# Optional: satellite/street basemap below the choropleth (50% transparency).
#   Set use_basemap = TRUE. For basemap tiles install: ggspatial and prettymapr
#   install.packages(c("ggspatial", "prettymapr"))
#   Basemap is drawn first, then choropleth on top with alpha = 0.5.
#
# Requirements: sf, ggplot2, dplyr, biscale, cowplot; optional: ggspatial, prettymapr (for basemap)
#   install.packages(c("sf", "ggplot2", "dplyr", "biscale", "cowplot", "ggspatial", "prettymapr"))
#
# Usage: Rscript map_bivariate_worldpop_waste.R
#
# Output: Figure/bivariate_worldpop_waste.png (map)
#         Figure/bivariate_worldpop_waste_scatter.png (correlation scatter, all points)
#         Figure/bivariate_worldpop_waste_scatter_nozero.png (correlation scatter, waste_count > 0 only)
#
# Note: style_bivar = "equal" avoids "breaks are not unique" when many cells
#       have the same value (e.g. zero waste). Use "quantile" if your data
#       has enough variation for equal-count classes.

library(sf)
library(ggplot2)
library(dplyr)
library(biscale)
library(cowplot)
if (requireNamespace("ggspatial", quietly = TRUE)) {
  library(ggspatial)
  HAS_GGSPATIAL <- TRUE
} else {
  HAS_GGSPATIAL <- FALSE
}

# -------------------- USER PARAMETERS --------------------
gpkg_path   <- "/Users/wenlanzhang/PycharmProjects/Waste_Flood_out/Data/4/4_flood_waste_metrics_quadkey.gpkg"
layer_name  <- "4_flood_waste_metrics_quadkey"
out_dir     <- "/Users/wenlanzhang/PycharmProjects/Waste_Flood_out/Figure"
out_file    <- file.path(out_dir, "bivariate_worldpop_waste.png")
out_scatter <- file.path(out_dir, "bivariate_worldpop_waste_scatter.png")
out_scatter_nozero <- file.path(out_dir, "bivariate_worldpop_waste_scatter_nozero.png")

col_worldpop <- "3_worldpop"
col_waste    <- "4_waste_count"

# R sometimes prefixes column names with "X" when reading; we resolve below
fb_baseline_min <- NULL   # set to 50 to filter 3_fb_baseline_median >= 50
baseline_col    <- "3_fb_baseline_median"

# Bivariate options
dim_bivar <- 3L          # 3x3 bivariate legend
# "equal" = equal intervals (avoids "breaks are not unique" when many zeros/ties)
# "quantile" = equal counts per class (use if data has enough variation)
style_bivar <- "equal"
pal_bivar   <- "DkBlue"   # e.g. "DkBlue", "GrPink", "DkViolet"

fig_width  <- 12
fig_height <- 10
dpi        <- 200

# Basemap: TRUE = add tile layer below choropleth and draw choropleth at 50% transparency
use_basemap   <- TRUE
choropleth_alpha <- 0.5
# Tile type for ggspatial: "osm" (street), "cartodark", "cartolight", etc. (see rosm::osm.types())
# For satellite-like imagery you need a tile URL that serves imagery; "osm" is street map.
basemap_type  <- "osm"

# Optional overlay shapefile (e.g. slum/settlement boundaries). Set to NULL to skip.
slum_shp_path <- "/Users/wenlanzhang/Downloads/PhD_UCL/Data/Waste/Angela/slumaps_nairobi_sett/slumaps_nairobi_sett.shp"
slum_fill_alpha <- 0.3   # 70% transparency (0.3 = 30% opaque)
slum_color     <- "black"
slum_linetype  <- "dashed"
slum_linewidth <- 0.6

# -------------------- Load data --------------------
cat("Loading data...\n")
if (!file.exists(gpkg_path)) stop("GPKG not found: ", gpkg_path)

gdf <- tryCatch(
  st_read(gpkg_path, layer = layer_name, quiet = TRUE),
  error = function(e) st_read(gpkg_path, quiet = TRUE)
)

# Resolve column names (R may add "X" prefix for names starting with a digit)
names(gdf)[names(gdf) == "X3_worldpop"]       <- "3_worldpop"
names(gdf)[names(gdf) == "X4_waste_count"]   <- "4_waste_count"
names(gdf)[names(gdf) == "X3_fb_baseline_median"] <- "3_fb_baseline_median"

if (!(col_worldpop %in% names(gdf))) stop("Column not found: ", col_worldpop)
if (!(col_waste %in% names(gdf)))    stop("Column not found: ", col_waste)

# Optional baseline filter
if (!is.null(fb_baseline_min) && baseline_col %in% names(gdf)) {
  gdf <- gdf %>%
    filter(!is.na(.data[[baseline_col]]), .data[[baseline_col]] >= fb_baseline_min)
  cat("  Filtered to", nrow(gdf), "rows (", baseline_col, ">=", fb_baseline_min, ")\n")
}

# Replace NODATA sentinel for waste if present
gdf[[col_waste]] <- replace(gdf[[col_waste]], gdf[[col_waste]] == -9999, NA_real_)

# Drop rows with NA in either variable for bivariate classification
gdf <- gdf %>%
  filter(!is.na(.data[[col_worldpop]]), !is.na(.data[[col_waste]]))

cat("  Rows for bivariate map:", nrow(gdf), "\n")

# -------------------- Bivariate classification (biscale) --------------------
# biscale expects unquoted column names (backticks for names starting with a digit)
gdf <- bi_class(gdf,
                x = `3_worldpop`,
                y = `4_waste_count`,
                style = style_bivar,
                dim = dim_bivar)

# -------------------- Transform to Web Mercator for basemap tiles --------------------
gdf_3857 <- st_transform(gdf, 3857)

# -------------------- Map --------------------
# If use_basemap: draw tile layer first (below), then choropleth with alpha
p_map <- ggplot(gdf_3857) +
  coord_sf(crs = 3857, expand = FALSE)

draw_basemap <- FALSE
if (use_basemap && HAS_GGSPATIAL) {
  tryCatch({
    p_map <- p_map +
      annotation_map_tile(type = basemap_type, zoomin = 0, alpha = 1, progress = "none") +
      geom_sf(aes(fill = bi_class), color = NA, linewidth = 0, alpha = choropleth_alpha)
    draw_basemap <- TRUE
  }, error = function(e) {
    cat("  Basemap skipped (", conditionMessage(e), "). Install: install.packages('prettymapr')\n")
  })
}
if (!draw_basemap) {
  if (use_basemap && !HAS_GGSPATIAL)
    cat("  ggspatial not installed; skipping basemap. install.packages('ggspatial') for basemap.\n")
  p_map <- p_map + geom_sf(aes(fill = bi_class), color = NA, linewidth = 0)
}

p_map <- p_map +
  bi_scale_fill(pal = pal_bivar, dim = dim_bivar) +
  bi_theme() +
  theme(
    plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
    plot.subtitle = element_text(size = 11, hjust = 0.5)
  ) +
  labs(
    title = "Bivariate map: WorldPop (3_worldpop) x Waste count (4_waste_count)",
    subtitle = paste0(style_bivar, " breaks, ", dim_bivar, "x", dim_bivar, " — ", pal_bivar,
                      if (draw_basemap) paste0(" (basemap + ", choropleth_alpha * 100, "% transparency)") else "")
  )

# -------------------- Optional overlay: slum/settlement shapefile --------------------
if (!is.null(slum_shp_path) && file.exists(slum_shp_path)) {
  tryCatch({
    slum_sf <- st_read(slum_shp_path, quiet = TRUE)
    slum_3857 <- st_transform(slum_sf, st_crs(gdf_3857))
    p_map <- p_map +
      geom_sf(data = slum_3857, fill = "gray40", alpha = slum_fill_alpha,
              color = slum_color, linetype = slum_linetype, linewidth = slum_linewidth,
              inherit.aes = FALSE)
    cat("  Overlay: slum/settlement shapefile added (", nrow(slum_3857), " features, ", round(slum_fill_alpha * 100), "% transparent, dashed).\n")
  }, error = function(e) {
    cat("  Overlay shapefile skipped:", conditionMessage(e), "\n")
  })
} else if (!is.null(slum_shp_path)) {
  cat("  Overlay shapefile not found:", slum_shp_path, "\n")
}

# -------------------- Legend --------------------
p_legend <- bi_legend(
  pal = pal_bivar,
  dim = dim_bivar,
  xlab = "Higher waste count",
  ylab = "Higher WorldPop",
  size = 9
)

# -------------------- Correlation scatter plot --------------------
# Use non-geometry columns for scatter (same rows as map)
dat <- st_drop_geometry(gdf)
wp  <- dat[[col_worldpop]]
wa  <- dat[[col_waste]]
ok  <- is.finite(wp) & is.finite(wa) & wp > 0 & wa >= 0
wp  <- wp[ok]
wa  <- wa[ok]
n   <- length(wp)
if (n >= 3L) {
  ct_pearson  <- cor.test(wp, wa, method = "pearson", exact = FALSE)
  ct_spearman <- cor.test(wp, wa, method = "spearman", exact = FALSE)
  r_pearson   <- ct_pearson$estimate
  p_pearson   <- ct_pearson$p.value
  r_spearman  <- ct_spearman$estimate
  p_spearman  <- ct_spearman$p.value
  cat("  Correlation (n = ", n, "): Pearson r = ", round(r_pearson, 4), ", p = ", format.pval(p_pearson, digits = 3),
      "; Spearman rho = ", round(r_spearman, 4), ", p = ", format.pval(p_spearman, digits = 3), "\n")
  df_scatter <- data.frame(worldpop = wp, waste_count = wa)
  p_scatter <- ggplot(df_scatter, aes(x = worldpop, y = waste_count)) +
    geom_point(alpha = 0.5, size = 2) +
    geom_smooth(method = "lm", se = TRUE, color = "darkblue", linewidth = 0.8) +
    scale_x_continuous(trans = "log10") +
    scale_y_continuous(trans = "log10") +
    labs(
      x = "WorldPop (3_worldpop)",
      y = "Waste count (4_waste_count)",
      title = "Correlation: WorldPop vs Waste count",
      subtitle = sprintf("Pearson r = %.3f (p %s)  |  Spearman rho = %.3f (p %s)",
                         r_pearson, format.pval(p_pearson, digits = 2),
                         r_spearman, format.pval(p_spearman, digits = 2))
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
      plot.subtitle = element_text(size = 10, hjust = 0.5),
      panel.grid.minor = element_blank()
    )
  ggsave(out_scatter, plot = p_scatter, width = 8, height = 6, dpi = dpi, bg = "white")
  cat("Saved:", out_scatter, "\n")
} else {
  cat("  Too few valid points (n = ", n, ") for correlation scatter; skipping.\n")
}

# -------------------- Correlation scatter plot (waste_count > 0 only) --------------------
wp_all <- dat[[col_worldpop]]
wa_all <- dat[[col_waste]]
ok_nozero <- is.finite(wp_all) & is.finite(wa_all) & wp_all > 0 & wa_all > 0
wp_nz <- wp_all[ok_nozero]
wa_nz <- wa_all[ok_nozero]
n_nz <- length(wp_nz)
if (n_nz >= 3L) {
  ct_pearson_nz  <- cor.test(wp_nz, wa_nz, method = "pearson", exact = FALSE)
  ct_spearman_nz <- cor.test(wp_nz, wa_nz, method = "spearman", exact = FALSE)
  r_pearson_nz   <- ct_pearson_nz$estimate
  p_pearson_nz   <- ct_pearson_nz$p.value
  r_spearman_nz  <- ct_spearman_nz$estimate
  p_spearman_nz  <- ct_spearman_nz$p.value
  cat("  Correlation, waste > 0 only (n = ", n_nz, "): Pearson r = ", round(r_pearson_nz, 4), ", p = ", format.pval(p_pearson_nz, digits = 3),
      "; Spearman rho = ", round(r_spearman_nz, 4), ", p = ", format.pval(p_spearman_nz, digits = 3), "\n")
  df_scatter_nz <- data.frame(worldpop = wp_nz, waste_count = wa_nz)
  p_scatter_nz <- ggplot(df_scatter_nz, aes(x = worldpop, y = waste_count)) +
    geom_point(alpha = 0.5, size = 2) +
    geom_smooth(method = "lm", se = TRUE, color = "darkblue", linewidth = 0.8) +
    scale_x_continuous(trans = "log10") +
    scale_y_continuous(trans = "log10") +
    labs(
      x = "WorldPop (3_worldpop)",
      y = "Waste count (4_waste_count)",
      title = "Correlation: WorldPop vs Waste count (waste > 0 only)",
      subtitle = sprintf("n = %d  |  Pearson r = %.3f (p %s)  |  Spearman rho = %.3f (p %s)",
                         n_nz, r_pearson_nz, format.pval(p_pearson_nz, digits = 2),
                         r_spearman_nz, format.pval(p_spearman_nz, digits = 2))
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
      plot.subtitle = element_text(size = 10, hjust = 0.5),
      panel.grid.minor = element_blank()
    )
  ggsave(out_scatter_nozero, plot = p_scatter_nz, width = 8, height = 6, dpi = dpi, bg = "white")
  cat("Saved:", out_scatter_nozero, "\n")
} else {
  cat("  Too few valid points with waste > 0 (n = ", n_nz, "); skipping no-zero scatter.\n")
}

# -------------------- Combine map + legend --------------------
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
final <- ggdraw() +
  draw_plot(p_map, 0, 0, 1, 1) +
  draw_plot(p_legend, 0.72, 0.35, 0.24, 0.28)

ggsave(out_file, plot = final, width = fig_width, height = fig_height, dpi = dpi, bg = "white")
cat("Saved:", out_file, "\n")
