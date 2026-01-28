#!/usr/bin/env Rscript
"
figure_2_spatial_distribution.R

FIGURE 2 (Main text, most important map figure)
Spatial distribution of displacement, flood exposure, and waste accumulation

Purpose: Visually demonstrate why flood alone is insufficient and why waste matters.

Panel A: Displacement intensity (choropleth)
Panel B: Flood exposure (95th percentile inundation)
Panel C: Waste accumulation (raw count)

Requirements:
  sf, ggplot2, dplyr, patchwork, viridis, RColorBrewer, scales

USAGE:
  Install required packages:
  install.packages(c('sf', 'ggplot2', 'dplyr', 'patchwork', 'viridis', 'RColorBrewer', 'scales'))

  Then run:
  Rscript figure_2_spatial_distribution.R
"

# Load required libraries
library(sf)
library(ggplot2)
library(dplyr)
library(patchwork)
library(viridis)
library(RColorBrewer)
library(scales)

# -------------------- USER PARAMETERS --------------------

# File paths
model_data_path <- "/Users/wenlanzhang/PycharmProjects/Waste_Flood_out/Output/wasteflood/model_data.gpkg"
model_data_layer <- "model_data"

flood_waste_path <- "/Users/wenlanzhang/PycharmProjects/Waste_Flood_out/Data/4/4_flood_waste_metrics_quadkey.gpkg"
flood_waste_layer <- "4_flood_waste_metrics_quadkey"

output_dir <- "/Users/wenlanzhang/PycharmProjects/Waste_Flood_out/Figure/wasteflood"
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

# Figure settings
fig_width <- 18
fig_height <- 8
dpi <- 300

# Color schemes
displacement_palette <- "YlOrRd"  # Yellow-Orange-Red
flood_palette <- "Blues"          # Blues
waste_palette <- "YlOrBr"         # Yellow-Orange-Brown

# Data columns
displacement_col <- "X3_estimated_outflow_pop_from_2_outflow_max"
flood_col <- "X4_flood_p95"
waste_col <- "X4_waste_count"

# Panel labels
panel_a_label <- "A. Flood-induced displacement intensity"
panel_b_label <- "B. Flood exposure (95th percentile inundation depth)"
panel_c_label <- "C. Persistent waste accumulation"

# -------------------- Helper functions --------------------

load_spatial_data <- function() {
  "Load and merge model data with flood-waste data."

  cat("Loading flood-waste data (contains all variables)...\n")
  if (!file.exists(flood_waste_path)) {
    stop("Flood-waste data GPKG not found: ", flood_waste_path)
  }

  # Try to load flood-waste data
  tryCatch({
    flood_gdf <- st_read(flood_waste_path, layer = flood_waste_layer, quiet = TRUE)
    cat("  Loaded", nrow(flood_gdf), "flood-waste features from layer '", flood_waste_layer, "'\n")
  }, error = function(e) {
    cat("  Failed to load layer '", flood_waste_layer, "': ", conditionMessage(e), "\n")
    cat("  Attempting to load without specifying layer...\n")
    flood_gdf <- st_read(flood_waste_path, quiet = TRUE)
    cat("  Loaded", nrow(flood_gdf), "flood-waste features (no layer specified)\n")
  })

  # Check if displacement data is already in flood-waste data
  if (displacement_col %in% colnames(flood_gdf)) {
    cat("  Found displacement data (", displacement_col, ") in flood-waste file\n")
    merged_gdf <- flood_gdf
  } else {
    cat("  Displacement data (", displacement_col, ") not found in flood-waste file\n", sep = "")
    cat("  Attempting to load from model data...\n")

    if (!file.exists(model_data_path)) {
      stop("Model data GPKG not found: ", model_data_path,
           ". Run 'python model_wasteflood.py' to generate it.")
    }

    # Try to load model data
    tryCatch({
      model_gdf <- st_read(model_data_path, layer = model_data_layer, quiet = TRUE)
      cat("  Loaded", nrow(model_gdf), "model features from layer '", model_data_layer, "'\n")
    }, error = function(e) {
      cat("  Failed to load layer '", model_data_layer, "': ", conditionMessage(e), "\n")
      cat("  Attempting to load without specifying layer...\n")
      model_gdf <- st_read(model_data_path, quiet = TRUE)
      cat("  Loaded", nrow(model_gdf), "model features (no layer specified)\n")
    })

    # Merge datasets
    cat("  Merging datasets on quadkey...\n")
    merged_gdf <- model_gdf %>%
      left_join(st_drop_geometry(flood_gdf), by = "quadkey")
    cat("  Merged dataset has", nrow(merged_gdf), "features\n")
  }

  return(merged_gdf)
}

create_variable_map <- function(gdf, var_col, title, color_palette, log_scale = FALSE) {
  "Create a choropleth map for a variable."

  # Get valid data and apply clipping
  valid_data <- gdf[[var_col]][!is.na(gdf[[var_col]])]

  if (length(valid_data) == 0) {
    cat("Warning: No valid data for", var_col, "\n")
    # Return empty plot
    p <- ggplot() + theme_void() + ggtitle(title)
    return(p)
  }

  # Apply 98th percentile clipping for better visualization
  if (log_scale) {
    # For log scale, clip at 98th percentile
    vmax <- quantile(valid_data, 0.98, na.rm = TRUE)
    vmin <- min(valid_data[valid_data > 0], na.rm = TRUE)
    plot_data <- pmin(gdf[[var_col]], vmax)
  } else {
    # For linear scale, also clip at 98th percentile
    vmax <- quantile(valid_data, 0.98, na.rm = TRUE)
    vmin <- min(valid_data, na.rm = TRUE)
    plot_data <- pmin(gdf[[var_col]], vmax)
  }

  # Create breaks for colorbar
  if (log_scale) {
    breaks <- 10^seq(log10(vmin), log10(vmax), length.out = 5)
    labels <- scales::scientific(breaks, digits = 1)
  } else {
    breaks <- seq(vmin, vmax, length.out = 5)
    labels <- round(breaks, 1)
  }

  # Create the plot
  p <- ggplot(gdf) +
    geom_sf(aes(fill = plot_data), color = NA) +
    scale_fill_distiller(
      palette = color_palette,
      na.value = "lightgrey",
      limits = c(vmin, vmax),
      breaks = breaks,
      labels = labels,
      guide = guide_colorbar(
        title = NULL,
        direction = "horizontal",
        barheight = unit(0.3, "cm"),
        barwidth = unit(8, "cm"),
        title.position = "top",
        label.position = "bottom",
        label.hjust = 0.5
      )
    ) +
    theme_minimal() +
    theme(
      axis.text = element_blank(),
      axis.ticks = element_blank(),
      axis.title = element_blank(),
      panel.grid = element_blank(),
      legend.position = "bottom",
      legend.title = element_blank(),
      plot.title = element_text(size = 14, face = "bold", hjust = 0.5, margin = margin(b = 10))
    ) +
    ggtitle(title)

  # Add colorbar label
  if (log_scale) {
    colorbar_label <- paste0(gsub(".*\\.", "", var_col), " (log scale)")
  } else {
    colorbar_label <- paste0(gsub(".*\\.", "", var_col), " (linear)")
  }

  p <- p + labs(fill = colorbar_label)

  return(p)
}

# -------------------- Main execution --------------------

cat("================================================================================\n")
cat("FIGURE 2: Spatial distribution of displacement, flood, and waste\n")
cat("================================================================================\n")

tryCatch({
  # Load spatial data
  gdf <- load_spatial_data()

  # Check for required columns
  required_cols <- c(displacement_col, flood_col, waste_col)
  missing_cols <- required_cols[!required_cols %in% colnames(gdf)]

  if (length(missing_cols) > 0) {
    cat("❌ Missing required columns:", paste(missing_cols, collapse = ", "), "\n")
    cat("Available columns:\n")
    for (col in sort(colnames(gdf))) {
      cat("  - ", col, "\n", sep = "")
    }
    stop("Missing required columns: ", paste(missing_cols, collapse = ", "))
  }

  # Basic statistics
  cat("\nData summary:\n")
  for (col in required_cols) {
    valid <- !is.na(gdf[[col]])
    cat("  ", col, ": ", sum(valid), " valid values, ",
        "range: ", round(min(gdf[[col]], na.rm = TRUE), 2), " - ",
        round(max(gdf[[col]], na.rm = TRUE), 2), ", ",
        "mean: ", round(mean(gdf[[col]], na.rm = TRUE), 2), "\n", sep = "")
  }

  # Create the three panels
  cat("\nCreating panels...\n")

  cat("Creating Panel A: Displacement (", displacement_col, ")\n", sep = "")
  panel_a <- create_variable_map(gdf, displacement_col, panel_a_label,
                                displacement_palette, log_scale = TRUE)

  cat("Creating Panel B: Flood exposure (", flood_col, ")\n", sep = "")
  panel_b <- create_variable_map(gdf, flood_col, panel_b_label,
                                flood_palette, log_scale = FALSE)

  cat("Creating Panel C: Waste accumulation (", waste_col, ")\n", sep = "")
  panel_c <- create_variable_map(gdf, waste_col, panel_c_label,
                                waste_palette, log_scale = TRUE)

  # Combine panels side by side
  combined_plot <- panel_a + panel_b + panel_c +
    plot_layout(ncol = 3, widths = c(1, 1, 1)) +
    plot_annotation(
      title = "Spatial Distribution of Key Variables",
      theme = theme(
        plot.title = element_text(size = 16, face = "bold", hjust = 0.5, margin = margin(b = 20))
      )
    )

  # Save the figure
  output_path <- file.path(output_dir, "figure_2R_spatial_distribution.png")
  cat("\nSaving figure to:", output_path, "\n")

  ggsave(output_path, combined_plot, width = fig_width, height = fig_height,
         dpi = dpi, bg = "white")

  cat("✅ Figure 2 saved successfully!\n")

  cat("\nPurpose: Visually demonstrate why flood alone is insufficient and why waste matters\n")
  cat("Panel A: Displacement intensity shows where population outflow occurred\n")
  cat("Panel B: Flood exposure shows flood risk areas\n")
  cat("Panel C: Waste accumulation shows environmental degradation hotspots\n")

}, error = function(e) {
  cat("❌ Error:", conditionMessage(e), "\n")
})