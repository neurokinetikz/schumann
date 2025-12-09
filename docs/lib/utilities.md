# utilities

## Overview

Core utility functions for EEG data loading, filtering, PSD computation, and visualization.

**Module Statistics:**
- Total Functions: 34
- Public Functions: 29
- Private Functions: 5

## Main Analysis

| Function | Parameters | Description |
|----------|------------|-------------|
| `run_event_detection_pipeline` | `(df, electrodes, fs, bands=None, ...)` | High-level helper that computes band GFP and detects events. |

## Plotting

| Function | Parameters | Description |
|----------|------------|-------------|
| `plot_channel_overview` | `(df, electrode='AF3', seconds=10)` | *No description* |
| `plot_stacked_relpower` | `(rp_df, bands=list(RANGES.keys()))` | *No description* |
| `plot_stacked_relpower_timeseries` | `(df, electrodes=ELECTRODES, ...)` | For each electrode, plot the stacked relative power time series and the PSD i... |
| `animate_theta_alpha_psd` | `(df, electrode, fs=128, ...)` | Create a timelapse animation of the combined theta+alpha PSD for a single ele... |
| `plot_gfp_and_theta_alpha` | `(df, electrodes, fs=128, bands=None, ...)` | Plot Global Field Power (GFP) and theta/alpha band power over the entire reco... |
| `plot_pps_mountains` | `(df, electrodes, fs, start_sec=0.0, ...)` | Plot band-limited GFP for theta/alpha over a 10-minute window and mark |
| `plot_aperiodic_slope_timeseries` | `(df, electrodes, fs, start_sec=0.0, ...)` | Plot the aperiodic 1/f slope (beta exponent) over time for each electrode. |
| `plot_eeg_timeline_grid` | `(df, electrodes, ranges=RANGES, ...)` | Create a timeline plot for each frequency band by electrode (grid of subplots). |

## Computation

| Function | Parameters | Description |
|----------|------------|-------------|
| `compute_psd_multitaper` | `(sig, fs=FS, fmin=1, fmax=45)` | Return freqs (Hz) and PSD using MNE multitaper on a 1D numpy array. |
| `compute_relpower_table` | `(df, electrodes=ELECTRODES, ...)` | *No description* |
| `compute_iaf` | `(df, electrodes, fs=128, ...)` | Compute Individual Alpha Frequency (IAF) per electrode and ROI summary. |
| `compute_gfp` | `(data)` | Compute Global Field Power (GFP) across electrodes. |
| `compute_gfp_multichannel` | `(X)` | Compute Global Field Power (GFP) over time. |
| `compute_band_gfp` | `(df, electrodes, fs, bands, ...)` | Compute band-limited GFP time series per band. |
| `compute_aperiodic_slope_timeseries` | `(df, electrodes, fs, start_sec=0.0, ...)` | Compute time series of the aperiodic 1/f slope (beta exponent) per electrode. |

## Detection

| Function | Parameters | Description |
|----------|------------|-------------|
| `detect_power_spike_events` | `(df, electrodes, fs=128, bands=..., ...)` | Detect meditation-specific spectral bursts defined as epochs where GFP exceed... |
| `detect_power_spike_events` | `(band_gfp, fs, baseline_slice=None, ...)` | Detect power spike events where GFP exceeds baseline mean+z*std |

## Data Processing

| Function | Parameters | Description |
|----------|------------|-------------|
| `bandpass` | `(x, fs, f1, f2, order=4)` | *No description* |
| `load_eeg_csv` | `(csv_path, electrodes=ELECTRODES, ...)` | Load CSV as in user's snippet and return a pre-processed DataFrame. |

## Utilities

| Function | Parameters | Description |
|----------|------------|-------------|
| `butter_highpass` | `(sig, cutoff_hz, fs=FS, order=2)` | *No description* |
| `butter_bandpass` | `(sig, f_lo, f_hi, fs=FS, order=4)` | *No description* |
| `zscore` | `(x)` | *No description* |
| `butter_highpass` | `(sig, cutoff_hz, fs=FS, order=2)` | *No description* |
| `butter_bandpass` | `(sig, f_lo, f_hi, fs=FS, order=4)` | *No description* |
| `bandpowers_from_psd` | `(freqs, psd, ranges=RANGES)` | Integrate PSD over frequency bands (absolute) and compute relative shares. |
| `binary_lzc` | `(series)` | *No description* |
| `series_entropy` | `(x)` | Sample entropy wrapper (returns NaN if package missing). |
| `graph_eeg_timeline` | `(df, electrodes, ranges=RANGES, ...)` | Create a timeline plot of EEG power for each frequency band by electrode. |

## Private Helpers

| Function | Description |
|----------|-------------|
| `_moving_average` | *Helper function* |
| `_update` | *Helper function* |
| `_butter_bandpass` | *Helper function* |
| `_moving_average` | *Helper function* |
| `_fit_mask` | *Helper function* |
