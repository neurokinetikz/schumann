# utilities

**Total functions:** 34 (29 public, 5 private)

## Public Functions

### `butter_highpass(sig, cutoff_hz, fs, order)`

### `butter_bandpass(sig, f_lo, f_hi, fs, order)`

### `bandpass(x: np.ndarray, fs: float, f1: float, f2: float, order: int) -> np.ndarray`

### `zscore(x: np.ndarray) -> np.ndarray`

### `load_eeg_csv(csv_path, electrodes, device, fs, header)`
> Load CSV as in user's snippet and return a pre-processed DataFrame.

### `butter_highpass(sig, cutoff_hz, fs, order)`

### `butter_bandpass(sig, f_lo, f_hi, fs, order)`

### `compute_psd_multitaper(sig, fs, fmin, fmax)`
> Return freqs (Hz) and PSD using MNE multitaper on a 1D numpy array.

### `bandpowers_from_psd(freqs, psd, ranges)`
> Integrate PSD over frequency bands (absolute) and compute relative shares.

### `binary_lzc(series)`

### `series_entropy(x)`
> Sample entropy wrapper (returns NaN if package missing).

### `plot_channel_overview(df, electrode, seconds)`

### `compute_relpower_table(df, electrodes, bands)`

### `plot_stacked_relpower(rp_df, bands)`

### `plot_stacked_relpower_timeseries(df, electrodes, bands, start_sec, end_sec, ...)`
> For each electrode, plot the stacked relative power time series and the PSD in one row (two subplots).

### `compute_iaf(df, electrodes, fs, start_sec, end_sec, ...)`
> Compute Individual Alpha Frequency (IAF) per electrode and ROI summary.

### `animate_theta_alpha_psd(df, electrode, fs, start_sec, end_sec, ...)`
> Create a timelapse animation of the combined theta+alpha PSD for a single electrode.

### `plot_gfp_and_theta_alpha(df, electrodes, fs, bands, smooth_window)`
> Plot Global Field Power (GFP) and theta/alpha band power over the entire recording.

### `compute_gfp(data)`
> Compute Global Field Power (GFP) across electrodes.

### `detect_power_spike_events(df, electrodes, fs, bands, threshold, ...)`
> Detect meditation-specific spectral bursts defined as epochs where GFP exceeds baseline mean by >3 SDs

### `compute_gfp_multichannel(X)`
> Compute Global Field Power (GFP) over time.

### `compute_band_gfp(df, electrodes, fs, bands, use_existing_cols)`
> Compute band-limited GFP time series per band.

### `detect_power_spike_events(band_gfp, fs, baseline_slice, z_thresh, min_bands, ...)`
> Detect power spike events where GFP exceeds baseline mean+z*std

### `run_event_detection_pipeline(df, electrodes, fs, bands, baseline_slice, ...)`
> High-level helper that computes band GFP and detects events.

### `plot_pps_mountains(df, electrodes, fs, start_sec, duration_sec, ...)`
> Plot band-limited GFP for theta/alpha over a 10-minute window and mark

### `compute_aperiodic_slope_timeseries(df, electrodes, fs, start_sec, end_sec, ...)`
> Compute time series of the aperiodic 1/f slope (beta exponent) per electrode.

### `plot_aperiodic_slope_timeseries(df, electrodes, fs, start_sec, end_sec, ...)`
> Plot the aperiodic 1/f slope (beta exponent) over time for each electrode.

### `graph_eeg_timeline(df, electrodes, ranges, time_col, start_time, ...)`
> Create a timeline plot of EEG power for each frequency band by electrode.

### `plot_eeg_timeline_grid(df, electrodes, ranges, time_col, start_time, ...)`
> Create a timeline plot for each frequency band by electrode (grid of subplots).

## Private/Helper Functions

- `_moving_average(x, k)`
- `_update(i)`
- `_butter_bandpass(sig, fs, f_lo, ...)`
- `_moving_average(x, win)`
- `_fit_mask(freqs)`
