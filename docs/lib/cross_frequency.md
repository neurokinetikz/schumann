# cross_frequency

**Total functions:** 30 (25 public, 5 private)

## Public Functions

### `infer_fs_from_records(RECORDS: pd.DataFrame, time_col: str) -> float`

### `pac_tort_mi(phase: np.ndarray, amp: np.ndarray, nbins: int) -> float`

### `pac_glm_r2(phase: np.ndarray, amp: np.ndarray) -> float`

### `pac_surrogate_z(phase: np.ndarray, amp: np.ndarray, method: str, n: int) -> Tuple[float, float]`

### `pac_comodulogram_array(x: np.ndarray, fs: float, phase_bands: List[Tuple[float, float]], amp_bands: List[Tuple[float, float]], method: str, ...) -> Tuple[np.ndarray, Optional[np.ndarray]]`

### `pac_event_windows_records(RECORDS: pd.DataFrame, ch_name: str, windows: List[Tuple[float, float]], phase_bands: List[Tuple[float, float]], amp_bands: List[Tuple[float, float]], ...) -> Tuple[np.ndarray, Optional[np.ndarray], List[str], List[str]]`

### `bicoherence_array(x: np.ndarray, fs: float, nperseg: int, noverlap: int, fmax: Optional[float])`

### `bicoherence_event_windows_records(RECORDS: pd.DataFrame, ch_name: str, windows: List[Tuple[float, float]], nperseg: int, noverlap: int, ...)`

### `waveform_shape_metrics_array(x: np.ndarray, fs: float, band: Tuple[float, float], neighborhood_ms: float)`

### `sharpness(idxs)`

### `waveform_shape_event_windows_records(RECORDS: pd.DataFrame, ch_name: str, windows: List[Tuple[float, float]], band: Tuple[float, float], time_col: str) -> Dict[str, float]`

### `run_crossfreq_suite_records(RECORDS: pd.DataFrame, ignition_windows: List[Tuple[float, float]], rebound_windows: Optional[List[Tuple[float, float]]], sensor_phase_ch: str, sensor_amp_chs: Tuple[str, ...], ...) -> Dict[str, object]`

### `infer_fs(RECORDS: pd.DataFrame, time_col: str) -> float`

### `get_series(RECORDS: pd.DataFrame, name: str) -> np.ndarray`
> Return a numeric signal array. Accepts 'EEG.O1' or bare 'O1' (will try 'EEG.O1').

### `bandpass(x: np.ndarray, fs: float, f1: float, f2: float, order: int) -> np.ndarray`

### `slice_epoch(x: np.ndarray, idx0: int, idx1: int) -> Optional[np.ndarray]`

### `detect_schumann_bursts(RECORDS: pd.DataFrame, sr_channel: str, time_col: str, center_hz: float, half_bw_hz: float, ...) -> Dict[str, object]`
> Detect Schumann bursts on a reference signal by thresholding the narrowband envelope.

### `pac_mi_phase_amp(x_phase: np.ndarray, x_amp: np.ndarray, nbins: int) -> float`
> Tort MI: KL divergence of phase-binned amplitude from uniform.

### `epochwise_pac_timecourse(RECORDS: pd.DataFrame, eeg_channels: List[str], time_col: str, onsets_sec: List[float], win_sec: Tuple[float, float], ...) -> Dict[str, object]`
> Build trial x time PAC(t) around onsets, averaged over channels.

### `cluster_permutation_1d(mean_tc: np.ndarray, trials_tc: np.ndarray, alpha: float, n_perm: int, rng_seed: int) -> Dict[str, object]`
> Simple 1D cluster-based permutation along time for ERPAC curve.

### `run_schumann_locked_erpac(RECORDS: pd.DataFrame, sr_channel: str, eeg_channels: List[str], time_col: str, detect_params: Dict, ...) -> Dict[str, object]`
> Full ERPAC:

### `segment_fft(sig: np.ndarray, fs: float, nperseg: int, noverlap: int) -> np.ndarray`
> Return STFT-like complex spectra array (n_seg, n_freq) using Hann windows.

### `cross_bicoherence(RECORDS: pd.DataFrame, x_sr: str, y_eeg: str, z_eeg: Optional[str], time_col: str, ...) -> Dict[str, object]`
> Compute cross-bicoherence b_xy(f1,f2) predicting Z at f1+f2:

### `idx_of(freq)`

### `plot_bicoherence(out: Dict[str, object], i_f1: int, title: Optional[str]) -> None`
> Heatmap of cross-bicoherence at a fixed f1 index across f2_grid.

## Private/Helper Functions

- `_butter_bandpass(x: np.ndarray, fs: float, f1: float, ...)`
- `_phase_amp(x: np.ndarray, fs: float, f_phase: Tuple[float, float], ...)`
- `_find_channel_series(RECORDS: pd.DataFrame, ch_name: str)`
- `_sanitize_band(band: Tuple[float, float], fs: float, min_bw: float)`
- `_sanitize_band_list(bands: List[Tuple[float, float]], fs: float, label: str)`
