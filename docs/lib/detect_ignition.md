# detect_ignition

**Total functions:** 59 (20 public, 39 private)

## Public Functions

### `ensure_timestamp_column(df: pd.DataFrame, time_col: str, default_fs: float) -> str`

### `infer_fs(df: pd.DataFrame, time_col: str) -> float`

### `get_series(df: pd.DataFrame, col: str) -> np.ndarray`

### `bandpass_safe(x: np.ndarray, fs: float, f1: float, f2: float, order) -> np.ndarray`

### `detect_ignitions_session(RECORDZ: pd.DataFrame, sr_channel: Optional[str], eeg_channels: Optional[List[str]], time_col: str, out_dir: str, ...) -> Tuple[Dict[str, object], List[Tuple[int, int]]]`

### `fmt_iqr(x: np.ndarray) -> str`

### `col_vals(name: str, fill: float) -> np.ndarray`

### `plot_psd_pre_peak_post(RECORDZ: pd.DataFrame, events_df: pd.DataFrame, event_index: int, eeg_channels: Optional[List[str]], center_hz: float, ...) -> str`
> Make the hero PSD overlay (baseline, crest, afterglow) for a single event with harmonic annotations.

### `plot_harmonic_rbp_bar(RECORDZ: pd.DataFrame, events_df: pd.DataFrame, event_index: int, eeg_channels: Optional[List[str]], center_hz: float, ...) -> str`
> Barplot of Relative Band Power for each harmonic at crest vs baseline for one event.

### `make_ignition_hero_figures(RECORDZ: pd.DataFrame, events_csv_path: str, event_index: int, eeg_channels: Optional[List[str]], center_hz: float, ...) -> Dict[str, str]`
> Convenience wrapper: generates the PSD overlay and the harmonic RBP barplot for one event.

### `animate_rbp(RECORDZ: pd.DataFrame, eeg_channels: Optional[List[str]], combine: str, time_col: str, t_range: Optional[Tuple[float, float]], ...)`
> Create an animation of the stacked Relative Band Power over time for a selected electrode or

### `plot_delta_spectrogram(RECORDZ: pd.DataFrame, eeg_channels: Optional[List[str]], combine: str, time_col: str, t_range: Optional[Tuple[float, float]], ...)`
> Fast view of *which delta frequencies* (0.5–4 Hz) show surges.

### `delta_peaks_for_event(RECORDZ: pd.DataFrame, t0_net: float, eeg_channels: Optional[List[str]], combine: str, time_col: str, ...) -> pd.DataFrame`
> Return up to top_n delta-frequency peaks at the ignition crest with z vs baseline.

### `summarize_delta_hotspots(RECORDZ: pd.DataFrame, events_df: pd.DataFrame, eeg_channels: Optional[List[str]], combine: str, time_col: str, ...) -> Tuple[pd.DataFrame, str]`
> Scan all events for delta crest peaks, aggregate, and plot KDE+hist of hotspots.

### `cluster_delta_hotspots_meanshift(DF: pd.DataFrame, z_thresh: float, bandwidth_quantile: float, fallback_bw: float, B: int, ...) -> pd.DataFrame`
> Cluster crest delta peaks (Hz) using MeanShift with robust bandwidth fallback,

### `boot_ci(vals, weights)`

### `animate_delta_psd(RECORDZ: pd.DataFrame, eeg_channels: Optional[List[str]], combine: str, time_col: str, t_range: Optional[Tuple[float, float]], ...)`
> Animate delta-band PSD over time for selected electrodes.

### `animate_psd_stacked(RECORDZ: pd.DataFrame, eeg_channels: Optional[List[str]], combine: str, time_col: str, t_range: Optional[Tuple[float, float]], ...)`
> Animate a *stacked area* of **absolute band power** (integrated PSD) over time across bands.

### `plot_phase_delay_wtc_bico(x: np.ndarray, y: np.ndarray, fs: float, bands: Optional[List[Tuple[str, Tuple[float, float]]]], freq_fit: Tuple[float, float], ...)`
> Small diagnostic panel that:

### `phase_wtc_bico_from_df(RECORDZ: pd.DataFrame, x_col: str, y_col: str, time_col: str, t_range: Optional[Tuple[float, float]], ...)`
> Convenience wrapper: slice a time window from RECORDZ and call `plot_phase_delay_wtc_bico`.

## Private/Helper Functions

- `_ensure_dir(p: str)`
- `_merge_intervals_int(it: List[Tuple[int, int]])`
- `_safe_band(f_lo, f_hi, fs, ...)`
- `_sr_envelope_z_series(y: np.ndarray, fs: float, f0: float, ...)`
- `_scalar_bandwidth(val, default: float)`
- `_msc_f0_series(x: np.ndarray, y: np.ndarray, fs: float, ...)`
- `_ssd_weights(X: np.ndarray, fs: float, f0: float, ...)`
- `_plv_weights(X: np.ndarray, fs: float, f_lo: float, ...)`
- `_pca_reference(X: np.ndarray, fs: float, f_lo: float, ...)`
- `_build_virtual_sr(X: np.ndarray, fs: float, f0: float, ...)`
- `_msc_per_channel_vs_median(X: np.ndarray, fs: float, freqs: List[float], ...)`
- `_spectral_slope_during_event(X: np.ndarray, fs: float, band: Tuple[float, float], ...)`
- `_frequency_specificity_index(X: np.ndarray, fs: float, sr_freqs: List[float], ...)`
- `_msc_bandwidth_specificity(X: np.ndarray, fs: float, sr_freqs: List[float], ...)`
- `_kuramoto_R_timeseries(X, fs, f_lo, ...)`
- `_detect_t0_from_R(times: np.ndarray, R: np.ndarray, thresh: float)`
- `_channel_latencies(X: np.ndarray, fs: float, f_lo: float, ...)`
- `_granger_bivariate_matrix(X: np.ndarray, maxlag: int)`
- `_directed_flow_scores(X: np.ndarray, fs: float, f_lo: float, ...)`
- `_phase_gradient_directionality(X: np.ndarray, fs: float, f_lo: float, ...)`
- `_harmonic_stack_index_flexible(x: np.ndarray, fs: float, base_hz: float, ...)`
- `_get_sr_env_z(f0: float, bw: float)`
- `_bandwidth_for_harmonic(idx: int)`
- `_compute_seed_score(idx: int)`
- `_slice_idx(t0, left, right)`
- `_extract_eeg_matrix(RECORDZ: pd.DataFrame, eeg_channels: Optional[List[str]])`
- `_welch_psd(x: np.ndarray, fs: float, nperseg: Optional[int], ...)`
- `_band_power_from_psd(f: np.ndarray, Pxx: np.ndarray, f_lo: float, ...)`
- `_slice(left, right)`
- `_slice(left, right)`
- `_compute_rbp_timeseries(RECORDZ: pd.DataFrame, eeg_channels: Optional[List[str]], time_col: str, ...)`
- `_left_index(k)`
- `_draw_frame(k)`
- `_parabolic_peak_refine(f: np.ndarray, y: np.ndarray, i: int)`
- `_init()`
- `_update(i)`
- `_draw_frame(k)`
- `_fit_group_delay(f: np.ndarray, phase: np.ndarray, fit_range)`
- `_band_center(blo, bhi)`
