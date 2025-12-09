# surface_cuts

## Overview

RT-Style Multi-Seed Surfaces — Multiway Cuts Across Subsystems (fs=128)

**Module Statistics:**
- Total Functions: 13
- Public Functions: 5
- Private Functions: 8

## Main Analysis

| Function | Parameters | Description |
|----------|------------|-------------|
| `run_multi_seed_surface_cuts` | `(RECORDS, ignition_windows, ...)` | *No description* |

## Plotting

| Function | Parameters | Description |
|----------|------------|-------------|
| `plot_multicut_deltas` | `(df, shuffle)` | *No description* |

## Detection

| Function | Parameters | Description |
|----------|------------|-------------|
| `find_channel_series` | `(records, ch_name)` | *No description* |

## Utilities

| Function | Parameters | Description |
|----------|------------|-------------|
| `default_clusters` | `(electrodes)` | Map label-> indices into electrodes list for broad subsystems F, P, O, T. |
| `adj_for_windows` | `(windows, f1, f2)` | *No description* |

## Private Helpers

| Function | Description |
|----------|-------------|
| `_get_fs` | *Helper function* |
| `_autoelectrodes` | *Helper function* |
| `_bandpass` | *Helper function* |
| `_pseudo_wpli` | *Helper function* |
| `_set_set_mincut_capacity` | Given a Gomory–Hu tree GH (capacities on edges), compute min... |
| `_multi_seed_capacity` | Build graph → GH tree → sum of pairwise set–set mincut capac... |
| `_weight_permute_surrogate` | *Helper function* |
| `_degree_rewire_surrogate` | Threshold to given density; rewire with double_edge_swap; re... |
