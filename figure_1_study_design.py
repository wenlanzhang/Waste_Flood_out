#!/usr/bin/env python3
"""
figure_1_study_design.py

FIGURE 1 (Main text)
Study design and data integration framework

Purpose: Orient the reader quickly. Shows this is a systems integration paper.

Panel A: Study area map (Nairobi metro boundary + quadkey grid)
Panel B: Data layers aligned to grid
Panel C: Analytical workflow

Requirements:
  geopandas, matplotlib, numpy, pandas, pathlib, contextily (optional for basemap)

USAGE:
  pip install geopandas matplotlib numpy pandas contextily
  python figure_1_study_design.py

OUTPUT:
  Figure/wasteflood/figure_1_study_design.png
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
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
from matplotlib import gridspec

# -------------------- USER PARAMETERS --------------------
OUTPUT_BASE = Path("/Users/wenlanzhang/PycharmProjects/Waste_Flood_out/Output")
FLOOD_WASTE_DATA_PATH = Path("/Users/wenlanzhang/PycharmProjects/Waste_Flood_out/Data/4/4_flood_waste_metrics_quadkey.gpkg")
FLOOD_WASTE_LAYER = "4_flood_waste_metrics_quadkey"

FIGURE_DIR = Path("/Users/wenlanzhang/PycharmProjects/Waste_Flood_out/Figure/wasteflood")
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# Figure settings
FIG_SIZE = (16, 10)  # Width x Height for three-panel figure
DPI = 300
FONT_SIZE = 11

# Colors for different data types
DISPLACEMENT_COLOR = '#e74c3c'  # Red
FLOOD_COLOR = '#3498db'        # Blue
WASTE_COLOR = '#e67e22'        # Orange
BOUNDARY_COLOR = '#2c3e50'     # Dark blue-gray
GRID_COLOR = '#95a5a6'         # Light gray

# Panel labels
PANEL_A_LABEL = 'A. Study Area & Spatial Grid'
PANEL_B_LABEL = 'B. Data Integration Framework'
PANEL_C_LABEL = 'C. Analytical Workflow'

# -------------------- Helper functions --------------------

def create_study_area_map(ax, gdf):
    """Create Panel A: Study area map with quadkey grid."""

    # Plot the quadkey grid
    gdf.plot(ax=ax, color='none', edgecolor=GRID_COLOR, linewidth=0.3, alpha=0.7)

    # Add some sample quadkeys in different colors to show data availability
    # (This is conceptual - in reality we'd show actual data density)
    sample_size = min(500, len(gdf))  # Show up to 500 sample quadkeys
    sample_gdf = gdf.sample(sample_size, random_state=42)

    # Color code based on data availability (conceptual)
    colors = []
    for idx, row in sample_gdf.iterrows():
        if pd.notna(row.get('4_flood_p95', None)) and pd.notna(row.get('4_waste_count', None)):
            colors.append('#2ecc71')  # Green - complete data
        elif pd.notna(row.get('4_flood_p95', None)):
            colors.append(FLOOD_COLOR)  # Blue - flood only
        elif pd.notna(row.get('4_waste_count', None)):
            colors.append(WASTE_COLOR)  # Orange - waste only
        else:
            colors.append('#ecf0f1')  # Light gray - no data

    sample_gdf.plot(ax=ax, color=colors, edgecolor='none', alpha=0.8)

    # Remove axes for clean map appearance
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')

    # Add title
    ax.set_title(PANEL_A_LABEL, fontsize=FONT_SIZE+6, fontweight='bold', pad=20)

    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor='#2ecc71', label='Complete data', alpha=0.8),
        mpatches.Patch(facecolor=FLOOD_COLOR, label='Flood only', alpha=0.8),
        mpatches.Patch(facecolor=WASTE_COLOR, label='Waste only', alpha=0.8),
        mpatches.Patch(facecolor='#ecf0f1', label='No data', alpha=0.8),
        plt.Line2D([0], [0], color=GRID_COLOR, linewidth=1, label='Quadkey grid')
    ]

    ax.legend(handles=legend_elements, loc='lower left', fontsize=FONT_SIZE-1,
              frameon=True, facecolor='white', edgecolor='none')

    # Add scale and location info
    ax.text(0.02, 0.98, 'Nairobi Metropolitan Area\nQuadkey Resolution: ~4.8km²',
            transform=ax.transAxes, fontsize=FONT_SIZE,
            verticalalignment='top', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9))

    return ax

def create_data_layers_panel(ax):
    """Create Panel B: Data integration framework schematic."""

    # Clear the axis
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Title
    ax.set_title(PANEL_B_LABEL, fontsize=FONT_SIZE+6, fontweight='bold', pad=20)

    # Central quadkey grid representation
    grid_x, grid_y = 5, 4
    grid_size = 2

    # Draw a sample quadkey (simplified representation)
    quadkey = FancyBboxPatch((grid_x-0.8, grid_y-0.8), 1.6, 1.6,
                            boxstyle="round,pad=0.1",
                            facecolor='lightgray', edgecolor='black', linewidth=1)
    ax.add_patch(quadkey)

    # Add grid lines to represent quadkey structure
    for i in range(2):
        for j in range(2):
            sub_quadkey = FancyBboxPatch((grid_x-0.8 + i*0.8, grid_y-0.8 + j*0.8), 0.8, 0.8,
                                       boxstyle="round,pad=0.02",
                                       facecolor='white', edgecolor='gray', linewidth=0.5)
            ax.add_patch(sub_quadkey)

    ax.text(grid_x, grid_y+1.2, 'Spatial Unit\n(Quadkey)', ha='center', va='center',
            fontsize=FONT_SIZE, fontweight='bold')

    # Data source arrows and boxes
    data_sources = [
        {'name': 'Facebook\nMobility', 'desc': 'Population displacement\n& outflow estimation',
         'color': DISPLACEMENT_COLOR, 'pos': (2, 6.5)},
        {'name': 'Flood\nRaster', 'desc': '95th percentile\ninundation depth',
         'color': FLOOD_COLOR, 'pos': (8, 6.5)},
        {'name': 'Waste\nPoints', 'desc': 'Aerial imagery\ndetection & aggregation',
         'color': WASTE_COLOR, 'pos': (5, 1.5)}
    ]

    # Draw data source boxes and arrows
    for source in data_sources:
        # Data source box
        box = FancyBboxPatch((source['pos'][0]-1.2, source['pos'][1]-0.8), 2.4, 1.6,
                           boxstyle="round,pad=0.3",
                           facecolor=source['color'], edgecolor='black', linewidth=1, alpha=0.8)
        ax.add_patch(box)

        ax.text(source['pos'][0], source['pos'][1], f"{source['name']}\n{source['desc']}",
                ha='center', va='center', fontsize=FONT_SIZE-1, fontweight='bold', color='white')

        # Arrow to central quadkey
        arrow = ConnectionPatch(source['pos'], (grid_x, grid_y), "data", "data",
                              arrowstyle="->", shrinkA=20, shrinkB=20,
                              mutation_scale=15, fc=source['color'], color=source['color'],
                              linewidth=2, alpha=0.7)
        ax.add_patch(arrow)

    # Add integration explanation
    ax.text(5, 7.5, 'Data Integration & Alignment',
            ha='center', va='center', fontsize=FONT_SIZE+2, fontweight='bold')

    ax.text(5, 0.5, 'All data aggregated to\nuniform quadkey grid\n(~4.8 km² resolution)',
            ha='center', va='center', fontsize=FONT_SIZE-1,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

    return ax

def create_workflow_panel(ax):
    """Create Panel C: Analytical workflow."""

    # Clear the axis
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Title
    ax.set_title(PANEL_C_LABEL, fontsize=FONT_SIZE+6, fontweight='bold', pad=20)

    # Workflow steps
    steps = [
        {'name': 'Baseline\nPopulation', 'desc': 'Facebook\nconnectivity\nbaseline',
         'x': 2, 'y': 6, 'color': '#34495e'},
        {'name': 'Flood-Only\nModel', 'desc': 'Displacement ~\nFlood exposure',
         'x': 6, 'y': 6, 'color': FLOOD_COLOR},
        {'name': 'Flood +\nWaste Model', 'desc': 'Displacement ~\nFlood + Waste',
         'x': 10, 'y': 6, 'color': WASTE_COLOR},
        {'name': 'Spatial Lag\nModel', 'desc': 'Controls for\nspatial\nautocorrelation',
         'x': 8, 'y': 3, 'color': '#9b59b6'},
        {'name': 'Robustness\nChecks', 'desc': 'Population\ncontrols +\nalternative\nspecifications',
         'x': 6, 'y': 1.5, 'color': '#1abc9c'}
    ]

    # Draw workflow steps
    for step in steps:
        # Step box
        box = FancyBboxPatch((step['x']-1.5, step['y']-0.8), 3, 1.6,
                           boxstyle="round,pad=0.3",
                           facecolor=step['color'], edgecolor='black', linewidth=1, alpha=0.9)
        ax.add_patch(box)

        ax.text(step['x'], step['y'], f"{step['name']}\n{step['desc']}",
                ha='center', va='center', fontsize=FONT_SIZE-2, fontweight='bold', color='white')

    # Draw workflow arrows
    arrows = [
        ((2, 6), (6, 6)),    # Baseline -> Flood-only
        ((6, 6), (10, 6)),   # Flood-only -> Flood+Waste
        ((10, 6), (8, 3)),   # Flood+Waste -> Spatial Lag
        ((8, 3), (6, 1.5))   # Spatial Lag -> Robustness
    ]

    for start, end in arrows:
        arrow = ConnectionPatch(start, end, "data", "data",
                              arrowstyle="->", shrinkA=30, shrinkB=30,
                              mutation_scale=20, fc='black', color='black',
                              linewidth=2, alpha=0.8)
        ax.add_patch(arrow)

    # Add workflow explanation
    ax.text(6, 7.3, 'Progressive Model Development',
            ha='center', va='center', fontsize=FONT_SIZE+2, fontweight='bold')

    ax.text(1, 3.5, 'Research Question:\nDoes waste accumulation\nexplain displacement\nbeyond flood exposure?',
            ha='left', va='center', fontsize=FONT_SIZE,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))

    return ax

def create_figure_1(gdf):
    """Create Figure 1: Study design and data integration framework."""

    print("\n" + "="*80)
    print("Creating Figure 1: Study design and data integration framework")
    print("="*80)

    # Create figure with custom grid layout
    fig = plt.figure(figsize=FIG_SIZE)

    # Create grid specification for panels
    gs = gridspec.GridSpec(2, 2, figure=fig,
                          width_ratios=[1.2, 1], height_ratios=[1, 1],
                          hspace=0.3, wspace=0.3)

    # Panel A: Study area map (top-left, spans 1 row 1 col)
    ax1 = fig.add_subplot(gs[0, 0])
    create_study_area_map(ax1, gdf)

    # Panel B: Data layers (top-right)
    ax2 = fig.add_subplot(gs[0, 1])
    create_data_layers_panel(ax2)

    # Panel C: Workflow (bottom span)
    ax3 = fig.add_subplot(gs[1, :])
    create_workflow_panel(ax3)

    # Overall title
    fig.suptitle('Study Design and Data Integration Framework',
                fontsize=FONT_SIZE+8, fontweight='bold', y=0.98)

    # Save figure
    output_path = FIGURE_DIR / "figure_1_study_design.png"
    print(f"\nSaving figure to: {output_path}")
    fig.savefig(output_path, dpi=DPI, bbox_inches='tight')

    print("✅ Figure 1 saved successfully!")

    return fig

# -------------------- Main execution --------------------

def main():
    print("="*80)
    print("FIGURE 1: Study design and data integration framework")
    print("="*80)

    try:
        # Load flood-waste data (contains the spatial grid)
        print("Loading spatial data for study area visualization...")
        if not FLOOD_WASTE_DATA_PATH.exists():
            raise FileNotFoundError(f"Flood-waste data GPKG not found: {FLOOD_WASTE_DATA_PATH}")

        try:
            gdf = gpd.read_file(FLOOD_WASTE_DATA_PATH, layer=FLOOD_WASTE_LAYER)
            print(f"  Loaded {len(gdf):,} quadkeys from layer '{FLOOD_WASTE_LAYER}'")
        except Exception as e:
            print(f"  Failed to load layer '{FLOOD_WASTE_LAYER}': {e}")
            print("  Attempting to load without specifying layer...")
            try:
                gdf = gpd.read_file(FLOOD_WASTE_DATA_PATH)
                print(f"  Loaded {len(gdf):,} quadkeys (no layer specified)")
            except Exception as e2:
                raise RuntimeError(f"Cannot load spatial data: {e2}")

        # Create Figure 1
        fig = create_figure_1(gdf)

        print("\n" + "="*80)
        print("Figure 1 completed successfully!")
        print("="*80)
        print(f"Output: {FIGURE_DIR / 'figure_1_study_design.png'}")
        print("\nPurpose: Orient readers to study design and data integration")
        print("Panel A: Shows the spatial scope and quadkey grid")
        print("Panel B: Illustrates how different data sources are integrated")
        print("Panel C: Demonstrates the analytical workflow progression")

    except Exception as e:
        print(f"❌ Error: {e}")
        raise

if __name__ == "__main__":
    main()