# toroidal_phase

## Overview

Toroidal Phase–Torus Analysis (fs=128)

**Module Statistics:**
- Total Functions: 12
- Public Functions: 2
- Private Functions: 10

## Main Analysis

| Function | Parameters | Description |
|----------|------------|-------------|
| `run_toroidal_phase_analysis` | `(RECORDS, time_col='Timestamp', ...)` | *No description* |

## Detection

| Function | Parameters | Description |
|----------|------------|-------------|
| `find_channel_series` | `(records, ch_name)` | *No description* |

## Private Helpers

| Function | Description |
|----------|-------------|
| `_get_fs` | *Helper function* |
| `_bandpass` | *Helper function* |
| `_mean_reference` | *Helper function* |
| `_circ_R` | *Helper function* |
| `_circ_corr` | *Helper function* |
| `_two_nn_id` | Levina–Bickel TwoNN intrinsic dimension estimate on points (... |
| `_joint_phase_entropy` | *Helper function* |
| `_winding` | *Helper function* |
| `_phase_scramble` | *Helper function* |
| `_compute_torus_metrics` | *Helper function* |
