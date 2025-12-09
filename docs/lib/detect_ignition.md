# detect_ignition

## Overview

Detection of neural ignition events based on Schumann-band power and coherence thresholds.

**Module Statistics:**
- Total Functions: 59
- Public Functions: 20
- Private Functions: 39

## Plotting

| Function | Parameters | Description |
|----------|------------|-------------|
| `plot_psd_pre_peak_post` | `(RECORDZ, events_df, event_index, ...)` | Make the hero PSD overlay (baseline, crest, afterglow) for a single event wit... |
| `plot_harmonic_rbp_bar` | `(RECORDZ, events_df, event_index, ...)` | Barplot of Relative Band Power for each harmonic at crest vs baseline for one... |
| `animate_rbp` | `(RECORDZ, eeg_channels=None, ...)` | Create an animation of the stacked Relative Band Power over time for a select... |
| `plot_delta_spectrogram` | `(RECORDZ, eeg_channels=None, ...)` | Fast view of *which delta frequencies* (0.5–4 Hz) show surges. |
| `animate_delta_psd` | `(RECORDZ, eeg_channels=None, ...)` | Animate delta-band PSD over time for selected electrodes. |
| `animate_psd_stacked` | `(RECORDZ, eeg_channels=None, ...)` | Animate a *stacked area* of **absolute band power** (integrated PSD) over tim... |
| `plot_phase_delay_wtc_bico` | `(x, y, fs, bands=None, ...)` | Small diagnostic panel that: |

## Detection

| Function | Parameters | Description |
|----------|------------|-------------|
| `detect_ignitions_session` | `(RECORDZ, sr_channel='EEG.F4', ...)` | *No description* |

## Data Processing

| Function | Parameters | Description |
|----------|------------|-------------|
| `bandpass_safe` | `(x, fs, f1, f2, order=4)` | *No description* |

## Utilities

| Function | Parameters | Description |
|----------|------------|-------------|
| `ensure_timestamp_column` | `(df, time_col='Timestamp', ...)` | *No description* |
| `infer_fs` | `(df, time_col)` | *No description* |
| `get_series` | `(df, col)` | *No description* |
| `fmt_iqr` | `(x)` | *No description* |
| `col_vals` | `(name, fill=np.nan)` | *No description* |
| `make_ignition_hero_figures` | `(RECORDZ, events_csv_path, ...)` | Convenience wrapper: generates the PSD overlay and the harmonic RBP barplot f... |
| `delta_peaks_for_event` | `(RECORDZ, t0_net, eeg_channels=None, ...)` | Return up to top_n delta-frequency peaks at the ignition crest with z vs base... |
| `summarize_delta_hotspots` | `(RECORDZ, events_df, ...)` | Scan all events for delta crest peaks, aggregate, and plot KDE+hist of hotspots. |
| `cluster_delta_hotspots_meanshift` | `(DF, z_thresh=2.0, ...)` | Cluster crest delta peaks (Hz) using MeanShift with robust bandwidth fallback, |
| `boot_ci` | `(vals, weights)` | *No description* |
| `phase_wtc_bico_from_df` | `(RECORDZ, x_col, y_col, ...)` | Convenience wrapper: slice a time window from RECORDZ and call `plot_phase_de... |

## Private Helpers

| Function | Description |
|----------|-------------|
| `_ensure_dir` | *Helper function* |
| `_merge_intervals_int` | *Helper function* |
| `_safe_band` | *Helper function* |
| `_sr_envelope_z_series` | Envelope z-score for a monaural SR band. |
| `_scalar_bandwidth` | *Helper function* |
| `_msc_f0_series` | Return a short-window MSC time series at f0 for signals x,y. |
| `_ssd_weights` | *Helper function* |
| `_plv_weights` | *Helper function* |
| `_pca_reference` | *Helper function* |
| `_build_virtual_sr` | *Helper function* |
| `_msc_per_channel_vs_median` | Compute mean MSC across channels using chart-style method: |
| `_spectral_slope_during_event` | Compute mean 1/f spectral slope across channels during event... |
| `_frequency_specificity_index` | Compute Frequency Specificity Index (FSI). |
| `_msc_bandwidth_specificity` | Compute MSC Bandwidth Specificity Ratio (BSR). |
| `_kuramoto_R_timeseries` | *Helper function* |
| `_detect_t0_from_R` | *Helper function* |
| `_channel_latencies` | Return per-channel latency estimates (and optional diagnosti... |
| `_granger_bivariate_matrix` | Pairwise (time-domain) Granger causality using simple VAR fi... |
| `_directed_flow_scores` | Compute per-channel directed flow (in/out) using bivariate G... |
| `_phase_gradient_directionality` | *Helper function* |
| `_harmonic_stack_index_flexible` | Compute Harmonic Stack Index (HSI) and MaxH (strongest overt... |
| `_get_sr_env_z` | *Helper function* |
| `_bandwidth_for_harmonic` | *Helper function* |
| `_compute_seed_score` | Composite seed score combining temporal, causal, signal stre... |
| `_slice_idx` | *Helper function* |
| `_extract_eeg_matrix` | Return X (n_ch,n_samp), time vector t, fs, and channel list ... |
| `_welch_psd` | Welch PSD for a 1-D signal x. |
| `_band_power_from_psd` | *Helper function* |
| `_slice` | *Helper function* |
| `_slice` | *Helper function* |
| `_compute_rbp_timeseries` | Compute sliding-window Relative Band Power (RBP) time series... |
| `_left_index` | *Helper function* |
| `_draw_frame` | *Helper function* |
| `_parabolic_peak_refine` | Quadratic (parabolic) interpolation around bin i (use log po... |
| `_init` | *Helper function* |
| `_update` | *Helper function* |
| `_draw_frame` | *Helper function* |
| `_fit_group_delay` | Fit a line to unwrapped phase(f) over `fit_range` to estimat... |
| `_band_center` | *Helper function* |
