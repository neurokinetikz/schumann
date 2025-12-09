# spatial_source_harmonics

**Total functions:** 14 (12 public, 2 private)

## Public Functions

### `ensure_dir(d)`

### `ensure_timestamp_column(df, time_col, default_fs)`

### `infer_fs(df, time_col)`

### `phase_series(x, fs, f0, half)`

### `detect_eeg_channels(df, prefix)`

### `hpli_topography(RECORDS, sr_channel, harmonics, time_col, half_bw, ...)`
> Compute per‑electrode H‑PLI_k maps + p‑values for each harmonic, and HCS map.

### `wins_to_seg(wins, N)`

### `plv_networks(RECORDS, channels, harmonics, time_col, half_bw, ...)`
> Compute PLV networks at each harmonic; save adjacency heatmaps & graph plots; return stats (modularity, min‑cut, path length).

### `wins_mask(N)`

### `lcmv_sources_at_lines(RECORDS, eeg_channels, sr, fwd, noise_cov, ...)`
> If MNE forward model & noise covariance are provided, compute LCMV source power maps

### `connectome_overlap(source_vec, SC, k_modes)`
> Project a source map (N nodes) onto Laplacian eigenmodes of SC; return variance explained per mode.

### `analyze_spatial_and_source(RECORDS, sr_channel, windows, harmonics, half_bw, ...)`
> Run spatial topographies (H‑PLI, HCS), GFS bars, (optional) PLV networks and LCMV sources.

## Private/Helper Functions

- `_bandpass(x, fs, lo, ...)`
- `_plot_topo(values_dict, title, fname)`
