# test

**Total functions:** 128 (53 public, 75 private)

## Public Functions

### `p2_min_dur(self) -> float`

### `slice(self, t0: float, t1: float) -> 'BaseProvider'`

### `t(self) -> np.ndarray`

### `z_fund(self) -> np.ndarray`

### `z_h2(self) -> np.ndarray`

### `z_h3(self) -> np.ndarray`

### `plv_fund(self) -> np.ndarray`

### `hsi(self) -> np.ndarray`

### `beta(self) -> Optional[np.ndarray]`

### `ridge_is_fund(self) -> Optional[np.ndarray]`

### `bic_7_7_15(self) -> Optional[np.ndarray]`

### `bic_7_15_23(self) -> Optional[np.ndarray]`

### `pac_mvl(self) -> Optional[np.ndarray]`

### `spectrogram(self) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]`

### `slice(self, t0: float, t1: float) -> 'PackProvider'`

### `t(self)`

### `z_fund(self)`

### `z_h2(self)`

### `z_h3(self)`

### `plv_fund(self)`

### `hsi(self)`

### `beta(self)`

### `ridge_is_fund(self)`

### `bic_7_7_15(self)`

### `bic_7_15_23(self)`

### `pac_mvl(self)`

### `spectrogram(self)`

### `spectrogram_for_window(self, t0, t1)`

### `window_spec_median(records, window)`
> Robust spectrogram inside `window`:

### `patch_pack_with_hsi_v3_for_windows(pack, records, windows)`
> Compute a per-window spectrogram from time-domain median, derive HSI_v3,

### `repl(match)`

### `hsi_from_spec_v2(spec, ladder, win_half_hz, smooth_hz)`
> spec = (tS, fS, S) with S shape (F,T), linear power.

### `hsi_v3_from_window_spec(tW, fW, SW)`
> HSI_v3(t): lower = tighter harmonics.

### `sanity(pack)`

### `piano_roll_from_spec(spec_by_window)`
> spec_by_window: (tW, fW, SW) from your per-window spectrogram

### `bandtrace_from_spec(tW, fW, SW, f0, bw)`

### `plot_ignition_window_report(_records, provider, electrodes)`

### `z_norm(y)`

### `z_for_display(t, y)`

### `z_for_display(t_vec, y_vec, s)`

### `compute_session_spectrogram(_records)`
> Return a robust session spectrogram as (t_spec_abs, f_spec, Sxx_med).

### `build_ignition_feature_pack(_records: pd.DataFrame, windows: List[Tuple[float, float]]) -> Dict[str, np.ndarray]`

### `to_abs(ts, ys)`

### `interp_to_raw(t_src, y_src)`

### `robust_z(x)`

### `smooth_sec(t, y, sec)`

### `annotate_phases(ax, phases: Dict[str, Any], ymin: float, ymax: float) -> None`
> Annotate phases on plot. Supports both old 5-phase (P0-P4) and new 6-phase models.

### `six_panel(records, electrodes, ign_win, ign_out, ladder, ...)`

### `estimate_sr_peaks(records, fs, ign_win, session_harmonics, search_band)`
> Get a simple list of estimated SR harmonic frequencies from ignition window EEG (all channels).

### `six_panel_2(records, electrodes, ign_win, ign_out, ladder, ...)`

### `sr_signature_panel(records, electrodes, ign_win, ign_out, ladder, ...)`

### `ignition_signature_panel(records, electrodes, ign_win, ign_out, ladder, ...)`

### `six_panel_3(records, electrodes, ign_win, ign_out, ladder, ...)`

## Private/Helper Functions

- `_resolve_palette(name: str)`
- `__init__(self, pack: Dict[str, Any], sl: slice)`
- `_get(self, k: str, default)`
- `_get_any(self)`
- `_slice_spec_to_window(spec, window, min_cols)`
- `_spec_db_rowz(SW)`
- `_as_float_1d(x)`
- `_normalize_channel_label(label: Optional[str])`
- `_resolve_seed_channel_index(seed_ch: Optional[str], electrodes: Sequence[str])`
- `_match_ignition_event_row(ign_out: Any, ign_win: Tuple[float, float])`
- `_format_numeric_labels(labels: Sequence[str], decimals: int)`
- `_extract_seed_channel(ign_out: Any, ign_win: Tuple[float, float])`
- `_bp_hilbert_env_z(X, fs, f0, ...)`
- `_plv_7p8(X, fs, f0, ...)`
- `_spec_median(X, fs, band, ...)`
- `_hsi_from_spec(tS, fS, S, ...)`
- `_first_onset(mask: np.ndarray, t: np.ndarray, min_dur: float)`
- `_collect_runs(mask: np.ndarray, t: np.ndarray, min_dur: float, ...)`
- `_band_mask(t: np.ndarray, lo: float, hi: float)`
- `_clip_seed_to_window(seed_t: float | None, t0: float, t1: float, ...)`
- `_robust_z(y: np.ndarray)`
- `_winsor_robust_z(y: np.ndarray, p_lo: float, p_hi: float)`
- `_rising_over_tau(y: np.ndarray, t: np.ndarray, tau_s: float, ...)`
- `_bridge(mask: np.ndarray, t: np.ndarray, bridge_sec: float)`
- `_spectral_slope_series(t_spec: np.ndarray, f_spec: np.ndarray, Sxx: np.ndarray, ...)`
- `_avalanche_size_duration(signal: np.ndarray, t: np.ndarray, thresh: float, ...)`
- `_kuramoto_order_series(X: np.ndarray, fs: float, center_hz: float, ...)`
- `_msc_channel_to_reference(ch_signal: np.ndarray, ref_signal: np.ndarray)`
- `_msc_matrix(X: np.ndarray, fs: float, freqs: Sequence[float], ...)`
- `_plv_matrix(X: np.ndarray, fs: float, f0: float, ...)`
- `_mode_metrics(power: np.ndarray)`
- `_interp_safe(x_new: np.ndarray, xp: np.ndarray, fp: np.ndarray)`
- `_te_matrix(X: np.ndarray, fs: float, lead_sec: float)`
- `_transfer_entropy_proxy(theta_env: np.ndarray, gamma_env: np.ndarray, fs: float, ...)`
- `_sample_entropy(signal: np.ndarray, m: int, r: float)`
- `_phi(mm: int)`
- `_complexity_series(signal: np.ndarray, t: np.ndarray, win_sec: float, ...)`
- `_hurst_exponent(signal: np.ndarray, scales: Sequence[int])`
- `_lempel_ziv_complexity(signal: np.ndarray)`
- `_lz_complexity_series(signal: np.ndarray, t: np.ndarray, win_sec: float, ...)`
- `_baseline_slice(records: pd.DataFrame, time_col: str, window: Tuple[float, float], ...)`
- `_infer_fs(df: pd.DataFrame, time_col: str)`
- `_looks_like_eeg_col(name: str)`
- `_auto_channels(df: pd.DataFrame, time_col: str)`
- `_get_matrix(df: pd.DataFrame, channels: Sequence[str])`
- `_fir_bandpass(f0: float, bw: float, fs: float, ...)`
- `_fir_lowpass(fc: float, fs: float, numtaps: int)`
- `_sliding_windows(n: int, fs: float, win_sec: float, ...)`
- `_plv_across_channels(phases: np.ndarray)`
- `_plv_timecourse(X: np.ndarray, fs: float, f0: float, ...)`
- `_msc_timecourse(X: np.ndarray, fs: float, f0: float, ...)`
- `_narrowband_envelope_z(X, fs, f0, ...)`
- `_hsi_timecourse(X, fs, win_sec, ...)`
- `_pac_tort_mi_timecourse(X, fs, theta_band, ...)`
- `_pac_mvl_timecourse(X, fs)`
- `_bicoherence_triads_timecourse(X, fs, triads, ...)`
- `_detect_ignition_phases(t: np.ndarray, z_fund: np.ndarray, plv_fund: np.ndarray, ...)`
- `_np_percentile(arr: np.ndarray, p: float, default: float)`
- `_robust_z(arr: np.ndarray, idx: Optional[int])`
- `_sigmoid(x: float)`
- `_first_run(mask: np.ndarray, min_sec: float, start_idx: int)`
- `_gauss_smooth(y: np.ndarray, sigma_sec: float)`
- `_snap_event(ev: Dict[str, Any])`
- `_detect_six_phase_evolution(t: np.ndarray, z_fund: np.ndarray, plv_fund: np.ndarray, ...)`
- `_smooth(y: np.ndarray, sigma_sec: float)`
- `_find_phase_boundary(start_idx: int, direction: int, criterion_fn, ...)`
- `_sigmoid(x: float)`
- `_phase_confidence(start_idx: int, end_idx: int, criteria: dict)`
- `_time_to_idx(time: float)`
- `_annotate_six_phases(ax, phases: Dict[str, Any], ymin: float, ...)`
- `_annotate_five_phases_legacy(ax, phases: Dict[str, Any], ymin: float, ...)`
- `_get_event(name: str)`
- `_safe_envelope(center_hz: float)`
- `_to_z(env)`
- `_apply_sunrise_style(ax)`
