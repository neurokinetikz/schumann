#!/usr/bin/env python3
"""
Regenerate NC4 figure with fixed Panel B visualization
"""

import pandas as pd
import numpy as np
import sys
from scipy import stats

sys.path.insert(0, 'lib')

# Import visualization function from NC4
from null_control_4_hybrid import create_four_panel_figure

# Load existing results
print("Loading data...")
observed_df = pd.read_csv('data/SIE.csv')
random_df = pd.read_csv('null_control_4_random.csv')

# Map column names for consistency
observed_df = observed_df.rename(columns={
    'sr1_z_max': 'sr_z_max',
    'msc_sr1': 'msc_7p83_v',
    'plv_sr1': 'plv_mean_pm5'
})

# Add phi ratios to observed_df for visualization
if 'sr3/sr1' in observed_df.columns:
    observed_df['phi_31'] = observed_df['sr3/sr1']
    observed_df['phi_51'] = observed_df['sr5/sr1']
    observed_df['phi_53'] = observed_df['sr5/sr3']

# Rebuild results dict for statistical annotation
print("Computing statistics...")
METRICS = ['sr_score', 'sr_z_max', 'msc_7p83_v', 'plv_mean_pm5', 'HSI']
results = {}

for metric in METRICS:
    if metric not in observed_df.columns or metric not in random_df.columns:
        continue

    obs_values = observed_df[metric].dropna().values
    rand_values = random_df[metric].dropna().values

    if len(obs_values) == 0 or len(rand_values) == 0:
        continue

    u_stat, p_value = stats.mannwhitneyu(obs_values, rand_values, alternative='two-sided')
    pooled_std = np.sqrt((obs_values.std()**2 + rand_values.std()**2) / 2)
    cohens_d = (obs_values.mean() - rand_values.mean()) / pooled_std if pooled_std > 0 else 0

    combined = np.concatenate([obs_values, rand_values])
    obs_median = np.median(obs_values)
    percentile = stats.percentileofscore(combined, obs_median)

    results[metric] = {
        'p_value': p_value,
        'cohens_d': cohens_d,
        'percentile': percentile
    }

# Create the updated figure
print("Creating figure...")
create_four_panel_figure(observed_df, random_df, results, out_path='null_control_4_results_fixed.png')
print("✓ Figure regenerated: null_control_4_results_fixed.png")
