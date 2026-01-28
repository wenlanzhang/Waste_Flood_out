#!/usr/bin/env Rscript
"
figure_3_spatial_dependence.R

FIGURE 3 (Main text)
Spatial dependence in displacement and effect of spatial lag modelling

Purpose: Justify why you used spatial econometrics.

Panel A: OLS residuals (Flood + Waste OLS)
Panel B: SLM residuals (Flood + Waste SLM)

Caption: Spatial lag models substantially reduce residual clustering compared to OLS.

Requirements:
  sf, ggplot2, dplyr, patchwork, RColorBrewer

USAGE:
  Install required packages:
  install.packages(c('sf', 'ggplot2', 'dplyr', 'patchwork', 'RColorBrewer'))

  Then run:
  Rscript figure_3_spatial_dependence.R
"

# Load required libraries
library(sf)
library(ggplot2)
library(dplyr)
library(patchwork)
library(RColorBrewer)

# -------------------- USER PARAMETERS --------------------

# File paths
model_data_path <- "/Users/wenlanzhang/PycharmProjects/Waste_Flood_out/Output/wasteflood/model_data.gpkg"
model_data_layer <- "model_data"

output_dir <- "/Users/wenlanzhang/PycharmProjects/Waste_Flood_out/Figure/wasteflood"
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

# Figure settings
fig_width <- 16
fig_height <- 6
dpi <- 300

# Residual column names
ols_resid_col <- "residuals"
slm_resid_col <- "slm_residuals"

# Panel labels
panel_a_label <- "A. OLS Residuals\n(Flood + Waste OLS)"
panel_b_label <- "B. SLM Residuals\n(Flood + Waste SLM)"

# Color scheme for residuals (diverging: red for positive, blue for negative)
residual_palette <- "RdYlBu"

# -------------------- Helper functions --------------------

load_model_data <- function() {
  "Load the model data GeoPackage."

  if (!file.exists(model_data_path)) {
    stop("Model data GPKG not found: ", model_data_path)
  }

  cat("Loading model data from:", model_data_path, "\n")

  tryCatch({
    gdf <- st_read(model_data_path, layer = model_data_layer, quiet = TRUE)
    cat("  Loaded", nrow(gdf), "features from layer '", model_data_layer, "'\n")
    cat("  CRS:", st_crs(gdf)$input, "\n")
  }, error = function(e) {
    cat("  Failed to load layer '", model_data_layer, "': ", conditionMessage(e), "\n")
    cat("  Attempting to load without specifying layer...\n")
    gdf <- st_read(model_data_path, quiet = TRUE)
    cat("  Loaded", nrow(gdf), "features (no layer specified)\n")
  })

  return(gdf)
}

create_residual_map <- function(gdf, resid_col, title) {
  "Create a residual map."

  # Check if column exists
  if (!resid_col %in% colnames(gdf)) {
    cat("Warning: Column '", resid_col, "' not found. Creating empty plot.\n", sep = "")
    p <- ggplot() +
      theme_void() +
      ggtitle(title) +
      annotate("text", x = 0.5, y = 0.5, label = paste("Column", resid_col, "not found"),
               size = 6, hjust = 0.5, vjust = 0.5)
    return(p)
  }

  # Get residual data
  residuals <- gdf[[resid_col]]

  # Calculate symmetric color scale limits
  abs_max <- max(abs(residuals), na.rm = TRUE)
  limits <- c(-abs_max, abs_max)

  # Create breaks for colorbar
  breaks <- seq(-abs_max, abs_max, length.out = 7)
  labels <- sprintf("%.3f", breaks)

  # Create the plot
  p <- ggplot(gdf) +
    geom_sf(aes(fill = residuals), color = NA) +
    scale_fill_distiller(
      palette = residual_palette,
      limits = limits,
      breaks = breaks,
      labels = labels,
      guide = guide_colorbar(
        title = "Residual Value",
        direction = "vertical",
        barheight = unit(4, "cm"),
        barwidth = unit(0.5, "cm"),
        title.position = "top",
        label.position = "right",
        title.hjust = 0.5
      )
    ) +
    theme_minimal() +
    theme(
      axis.text = element_blank(),
      axis.ticks = element_blank(),
      axis.title = element_blank(),
      panel.grid = element_blank(),
      legend.position = "right",
      legend.title = element_text(size = 11, face = "bold"),
      legend.text = element_text(size = 9),
      plot.title = element_text(size = 14, face = "bold", hjust = 0.5, margin = margin(b = 10))
    ) +
    ggtitle(title)

  return(p)
}

# -------------------- Main execution --------------------

cat("================================================================================\n")
cat("FIGURE 3: Spatial dependence in displacement\n")
cat("================================================================================\n")

tryCatch({
  # Load model data
  gdf <- load_model_data()

  # Check for residual columns
  residual_cols <- colnames(gdf)[grepl("resid", colnames(gdf), ignore.case = TRUE)]
  cat("\nResidual columns found:", paste(residual_cols, collapse = ", "), "\n")

  # Use exact column names
  cat("\nUsing OLS residuals:", ols_resid_col, "\n")
  cat("Using SLM residuals:", slm_resid_col, "\n")

  # Fail loudly if required columns are missing
  if (!ols_resid_col %in% colnames(gdf)) {
    stop("OLS residuals column '", ols_resid_col, "' not found in model_data.gpkg. ",
         "Regenerate model data with OLS residuals saved.")
  }

  if (!slm_resid_col %in% colnames(gdf)) {
    stop("SLM residuals column '", slm_resid_col, "' not found in model_data.gpkg. ",
         "Regenerate model data with SLM residuals saved (see model_wasteflood.py).")
  }

  # Basic statistics
  ols_mean <- mean(gdf[[ols_resid_col]], na.rm = TRUE)
  ols_std <- sd(gdf[[ols_resid_col]], na.rm = TRUE)
  slm_mean <- mean(gdf[[slm_resid_col]], na.rm = TRUE)
  slm_std <- sd(gdf[[slm_resid_col]], na.rm = TRUE)

  cat("\nOLS residuals - Mean:", sprintf("%.4f", ols_mean), ", Std:", sprintf("%.4f", ols_std), "\n")
  cat("SLM residuals - Mean:", sprintf("%.4f", slm_mean), ", Std:", sprintf("%.4f", slm_std), "\n")

  # Create the two panels
  cat("\nCreating panels...\n")

  cat("Creating Panel A: OLS residuals (", ols_resid_col, ")\n", sep = "")
  panel_a <- create_residual_map(gdf, ols_resid_col, panel_a_label)

  cat("Creating Panel B: SLM residuals (", slm_resid_col, ")\n", sep = "")
  panel_b <- create_residual_map(gdf, slm_resid_col, panel_b_label)

  # Combine panels side by side with shared legend
  combined_plot <- panel_a + panel_b +
    plot_layout(ncol = 2, guides = "collect") +
    plot_annotation(
      title = "Spatial Dependence in Displacement",
      subtitle = "Effect of Spatial Lag Modelling",
      theme = theme(
        plot.title = element_text(size = 16, face = "bold", hjust = 0.5),
        plot.subtitle = element_text(size = 14, hjust = 0.5, margin = margin(b = 10))
      )
    ) &
    theme(legend.position = "right")

  # Save the figure
  output_path <- file.path(output_dir, "figure_3R_spatial_dependence.png")
  cat("\nSaving figure to:", output_path, "\n")

  ggsave(output_path, combined_plot, width = fig_width, height = fig_height,
         dpi = dpi, bg = "white")

  cat("✅ Figure 3 saved successfully!\n")

  cat("\nCaption suggestion:\n")
  cat("\"Spatial lag models substantially reduce residual clustering compared to OLS.\"\n")

}, error = function(e) {
  cat("❌ Error:", conditionMessage(e), "\n")
})