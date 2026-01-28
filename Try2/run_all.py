#!/usr/bin/env python3
"""
run_all.py

Run the complete analysis pipeline.
This script runs all steps in sequence.
"""

import subprocess
import sys
import os
from pathlib import Path

# Change to Try2 directory
script_dir = Path(__file__).parent
os.chdir(script_dir)

print("="*80)
print("TRY2: COMPLETE ANALYSIS PIPELINE")
print("="*80)

# Step 1: Load and prepare data
print("\n" + "="*80)
print("STEP 1: Loading and preparing data")
print("="*80)
exec(open('01_load_and_prepare_data.py').read())

# Step 2: Run regression
print("\n" + "="*80)
print("STEP 2: Running regression models")
print("="*80)
exec(open('02_run_regression.py').read())

print("\n" + "="*80)
print("ALL STEPS COMPLETE!")
print("="*80)
