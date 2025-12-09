# ignition_rebound

## Overview

Ignition vs Rebound Power Plots

**Module Statistics:**
- Total Functions: 6
- Public Functions: 6
- Private Functions: 0

## Plotting

| Function | Parameters | Description |
|----------|------------|-------------|
| `plot_bandpower` | `(rows, kind='rel', groupby='electrode')` | *No description* |
| `plot_bandpower_conditions` | `(long, focus_electrodes=None)` | *No description* |
| `plot_topomap_grid` | `(df, bands=...)` | Plot a grid of topomaps for multiple bands in one figure. |
| `plot_topomap_from_power` | `(df, band='rel_Alpha')` | Plot a scalp topomap from a per-electrode power table. |
| `plot_topomap_grid_from_power` | `(df, bands=None)` | Plot a grid of topomaps for multiple bands from a per-electrode power table. |

## Utilities

| Function | Parameters | Description |
|----------|------------|-------------|
| `bandpower_to_long` | `(rows, kind='rel')` | *No description* |
