# psd_waterfall

## Overview

Across‑Session Ignition PSD Collector & Grand Waterfall

**Module Statistics:**
- Total Functions: 24
- Public Functions: 9
- Private Functions: 15

## Plotting

| Function | Parameters | Description |
|----------|------------|-------------|
| `plot_heatmap` | `(title=None, cmap='turbo', ...)` | *No description* |
| `plot_grand_waterfall` | `(title=None, cmap='turbo', ...)` | *No description* |
| `plot_waterfall_sr` | `(freqs, Z, meta, title=None, ...)` | *No description* |
| `plot_ignition_psd_waterfall` | `(csv_or_df, windows, fs, ...)` | Backward-compatible function that accepts a DataFrame **or** CSV path. |

## Computation

| Function | Parameters | Description |
|----------|------------|-------------|
| `compute_psd_by_window_df` | `(df, windows, fs, channels=None, ...)` | *No description* |

## Utilities

| Function | Parameters | Description |
|----------|------------|-------------|
| `add_session` | `(RECORDS, windows, fs, session_id, ...)` | *No description* |
| `add_precomputed` | `(freqs, Z, session_id, windows=None, ...)` | *No description* |
| `freqs` | `()` | *No description* |
| `to_dataframe` | `()` | Return a tidy DataFrame with columns: row_id, freq, value, plus metadata. |

## Private Helpers

| Function | Description |
|----------|-------------|
| `_guess_time_col` | *Helper function* |
| `_auto_channels` | *Helper function* |
| `_sec_to_idx` | *Helper function* |
| `_bandpass_notch` | *Helper function* |
| `_welch_psd` | *Helper function* |
| `_aggregate_psd` | *Helper function* |
| `__init__` | sort_by: None \| 'session' \| 'duration' \| 'max' \| ('sr', f0) |
| `_row_df` | *Helper function* |
| `_sort_indices` | *Helper function* |
| `_add_sr_lines_2d` | *Helper function* |
| `_sr_markers_2d` | *Helper function* |
| `_resolve_view` | *Helper function* |
| `_add_sr_curtains` | *Helper function* |
| `_add_sr_markers` | *Helper function* |
| `_plot_waterfall_any` | *Helper function* |
