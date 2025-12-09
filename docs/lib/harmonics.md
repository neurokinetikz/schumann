# harmonics

**Total functions:** 57 (26 public, 31 private)

## Public Functions

### `find_channel_series(records: pd.DataFrame, ch_name: str) -> Optional[pd.Series]`

### `detect_schumann_spikes_wavelet(records: pd.DataFrame, signal_col: str, time_col: str, f0: float, n_harmonics: int, ...) -> Dict[str, object]`

### `group_coincident(events: List[List[Dict[str, float]]], tol_sec: float) -> List[Dict[str, object]]`

### `schumann_activity_index(z_spec: np.ndarray) -> np.ndarray`

### `plot_harmonic_heatmap(t: np.ndarray, z_spec: np.ndarray, f0: float, title: str) -> None`

### `plot_piano_roll(t: np.ndarray, events: List[List[Dict[str, float]]], f0: float, title: str) -> None`

### `plot_sai(t: np.ndarray, sai: np.ndarray, title: str) -> None`

### `detect_and_plot_schumann_wavelet(records: pd.DataFrame, signal_col: str, time_col: str, f0: float, n_harmonics: int, ...) -> Dict[str, object]`

### `group_coincident(events: List[List[Dict[str, float]]], tol_sec: float) -> List[Dict[str, object]]`

### `schumann_activity_index(z: np.ndarray) -> np.ndarray`

### `detect_and_plot_schumann_microgrid_with_heatmaps(records: pd.DataFrame, signal_col: str, time_col: str, f0: float, n_harmonics: int, ...) -> Dict[str, object]`

### `group_coincident(events: List[List[Dict[str, float]]], tol_sec: float) -> List[Dict[str, object]]`

### `schumann_activity_index(z: np.ndarray) -> np.ndarray`

### `detect_and_plot_schumann_microgrid_with_global_tf(records: pd.DataFrame, signal_col: str, time_col: str, f0: float, n_harmonics: int, ...) -> Dict[str, object]`

### `compute_overlap_series(z_ridge: np.ndarray, z_thresh: float) -> np.ndarray`
> Count of harmonics with z >= z_thresh at each time point.

### `summarize_overlap_intervals(t: np.ndarray, overlap: np.ndarray, n_harm: int, min_len_sec: float, fs: float) -> Dict[int, List[Tuple[int, int, float, float]]]`
> For K=2..n_harm, create intervals where overlap>=K, enforcing min length.

### `plot_overlap_series(t: np.ndarray, overlap: np.ndarray, title: str) -> None`

### `plot_overlap_hist(overlap: np.ndarray, title: str) -> None`

### `plot_overlap_intervals(t: np.ndarray, intervals: Dict[int, List[Tuple[int, int, float, float]]], title: str) -> None`

### `compute_and_plot_overlap_from_fused(fused: Dict[str, object], z_thresh: float | None, min_len_sec: float, show: bool) -> Dict[str, object]`
> Compute and (optionally) plot overlap score from a fused micro-grid result.

### `run_overlap_coherence_etas(records: pd.DataFrame, fused: Dict[str, object], electrodes: Optional[List[str]], time_col: str, K: int, ...) -> Dict[str, object]`

### `eta(y)`

### `eta_c(y)`

### `bandplot(ax, y, ci, label, color)`

### `estimate_session_sr_harmonics(records, electrodes, fs, canonical_harmonics, search_band)`
> Estimate average Schumann Resonance (SR) frequencies for canonical harmonic values across an EEG session.

### `estimate_sr_harmonics(records, sr_channel, fs, f_can, search_halfband, ...)`
> Estimate peaks near canonical Schumann harmonics using Welch PSD.

## Private/Helper Functions

- `_get_fs(records: pd.DataFrame, time_col: str)`
- `_smooth(x: np.ndarray, fs: float, smooth_sec: float)`
- `_rolling_median_mad(x: np.ndarray, win: int)`
- `_get_channel_vector(records, ch: str)`
- `_morlet_kernel(fs: float, f0: float, w: float, ...)`
- `_cwt_morlet(x: np.ndarray, fs: float, freqs: np.ndarray, ...)`
- `_get_fs(records: pd.DataFrame, time_col: str)`
- `_smooth(x: np.ndarray, fs: float, smooth_sec: float)`
- `_rolling_median_mad(x: np.ndarray, win: int)`
- `_morlet_kernel(fs: float, f0: float, w: float, ...)`
- `_cwt_grid_morlet(x: np.ndarray, fs: float, grid: np.ndarray, ...)`
- `_find_intervals(mask: np.ndarray)`
- `_get_fs(records: pd.DataFrame, time_col: str)`
- `_smooth(x: np.ndarray, fs: float, smooth_sec: float)`
- `_rolling_median_mad(x: np.ndarray, win: int)`
- `_morlet_kernel(fs: float, f0: float, w: float, ...)`
- `_cwt_grid_morlet(x: np.ndarray, fs: float, grid: np.ndarray, ...)`
- `_find_intervals(mask: np.ndarray)`
- `_find_intervals(mask: np.ndarray)`
- `_get_fs(records: pd.DataFrame, time_col: str)`
- `_autoelectrodes(records: pd.DataFrame, time_col: str)`
- `_bandpass(X: np.ndarray, fs: float, f1: float, ...)`
- `_plv_mean_block(Xb: np.ndarray)`
- `_mincut_wpli_block(Xb: np.ndarray)`
- `_pac_mi_block(x_phase: np.ndarray, x_amp: np.ndarray, nbins: int)`
- `_beta_welch_block(x: np.ndarray, fs: float, fmin: float, ...)`
- `_eta_time_series(onsets: np.ndarray, tvec: np.ndarray, y: np.ndarray, ...)`
- `_infer_fs(df: pd.DataFrame, time_col: str)`
- `_as_list(x)`
- `_get_channel_array(records, channels)`
- `_peak_near(f0, half)`
