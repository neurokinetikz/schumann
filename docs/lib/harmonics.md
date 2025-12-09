# harmonics

## Overview

Schumann Spike Detector — Morlet Wavelet (fs=128)

**Module Statistics:**
- Total Functions: 57
- Public Functions: 26
- Private Functions: 31

## Main Analysis

| Function | Parameters | Description |
|----------|------------|-------------|
| `run_overlap_coherence_etas` | `(records, fused, electrodes=None, ...)` | *No description* |

## Plotting

| Function | Parameters | Description |
|----------|------------|-------------|
| `plot_harmonic_heatmap` | `(t, z_spec, f0, title='')` | *No description* |
| `plot_piano_roll` | `(t, events, f0, title='')` | *No description* |
| `plot_sai` | `(t, sai, title='')` | *No description* |
| `plot_overlap_series` | `(t, overlap, title='')` | *No description* |
| `plot_overlap_hist` | `(overlap, title='')` | *No description* |
| `plot_overlap_intervals` | `(t, intervals, title='')` | *No description* |

## Computation

| Function | Parameters | Description |
|----------|------------|-------------|
| `compute_overlap_series` | `(z_ridge, z_thresh)` | Count of harmonics with z >= z_thresh at each time point. |
| `compute_and_plot_overlap_from_fused` | `(fused, z_thresh=None, ...)` | Compute and (optionally) plot overlap score from a fused micro-grid result. |
| `estimate_session_sr_harmonics` | `(records, electrodes, fs, ...)` | Estimate average Schumann Resonance (SR) frequencies for canonical harmonic v... |
| `estimate_sr_harmonics` | `(records, sr_channel='EEG.F4', ...)` | Estimate peaks near canonical Schumann harmonics using Welch PSD. |

## Detection

| Function | Parameters | Description |
|----------|------------|-------------|
| `find_channel_series` | `(records, ch_name)` | *No description* |
| `detect_schumann_spikes_wavelet` | `(records, signal_col, ...)` | *No description* |
| `detect_and_plot_schumann_wavelet` | `(records, signal_col, ...)` | *No description* |
| `detect_and_plot_schumann_microgrid_with_heatmaps` | `(records, signal_col, ...)` | *No description* |
| `detect_and_plot_schumann_microgrid_with_global_tf` | `(records, signal_col, ...)` | *No description* |

## Utilities

| Function | Parameters | Description |
|----------|------------|-------------|
| `group_coincident` | `(events, tol_sec=0.1)` | *No description* |
| `schumann_activity_index` | `(z_spec)` | *No description* |
| `group_coincident` | `(events, tol_sec=0.1)` | *No description* |
| `schumann_activity_index` | `(z)` | *No description* |
| `group_coincident` | `(events, tol_sec=0.1)` | *No description* |
| `schumann_activity_index` | `(z)` | *No description* |
| `summarize_overlap_intervals` | `(t, overlap, n_harm, min_len_sec, fs)` | For K=2..n_harm, create intervals where overlap>=K, enforcing min length. |
| `eta` | `(y)` | *No description* |
| `eta_c` | `(y)` | *No description* |
| `bandplot` | `(ax, y, ci, label, color)` | *No description* |

## Private Helpers

| Function | Description |
|----------|-------------|
| `_get_fs` | *Helper function* |
| `_smooth` | *Helper function* |
| `_rolling_median_mad` | *Helper function* |
| `_get_channel_vector` | *Helper function* |
| `_morlet_kernel` | Build a complex Morlet wavelet centered at f0 (Hz). |
| `_cwt_morlet` | Compute CWT magnitudes for requested freqs via FFT convoluti... |
| `_get_fs` | *Helper function* |
| `_smooth` | *Helper function* |
| `_rolling_median_mad` | *Helper function* |
| `_morlet_kernel` | *Helper function* |
| `_cwt_grid_morlet` | *Helper function* |
| `_find_intervals` | *Helper function* |
| `_get_fs` | *Helper function* |
| `_smooth` | *Helper function* |
| `_rolling_median_mad` | *Helper function* |
| `_morlet_kernel` | *Helper function* |
| `_cwt_grid_morlet` | *Helper function* |
| `_find_intervals` | *Helper function* |
| `_find_intervals` | *Helper function* |
| `_get_fs` | *Helper function* |
| `_autoelectrodes` | *Helper function* |
| `_bandpass` | *Helper function* |
| `_plv_mean_block` | *Helper function* |
| `_mincut_wpli_block` | *Helper function* |
| `_pac_mi_block` | *Helper function* |
| `_beta_welch_block` | *Helper function* |
| `_eta_time_series` | *Helper function* |
| `_infer_fs` | *Helper function* |
| `_as_list` | Return x as a Python list if x is an iterable of channel nam... |
| `_get_channel_array` | Stack multiple channel vectors into a 2D array (n_channels, ... |
| `_peak_near` | *Helper function* |
