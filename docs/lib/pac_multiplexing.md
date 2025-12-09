# pac_multiplexing

**Total functions:** 21 (17 public, 4 private)

## Public Functions

### `bandpass(x: np.ndarray, fs: float, f1: float, f2: float, order: int) -> np.ndarray`

### `sliding_windows(n: int, win: int, step: int) -> List[Tuple[int, int]]`

### `circ_shift(arr: np.ndarray, shift: int) -> np.ndarray`

### `pac_mi_single(x_phase: np.ndarray, x_amp: np.ndarray, nbins: int) -> float`
> Tort-style Modulation Index: MI = (KL divergence of phase-binned amp distribution)/log(nbins).

### `compute_pac_timeseries(X: np.ndarray, fs: float, pairs: Dict[str, Tuple[Tuple[float, float], Tuple[float, float]]], win_sec: float, step_sec: float, ...) -> Tuple[np.ndarray, Dict[str, np.ndarray]]`
> Compute PAC(t) per pair with sliding windows; returns time vector and PAC dict.

### `find_channel_series(records: pd.DataFrame, ch_name: str) -> Optional[pd.Series]`

### `align_series_to_pac_timebase(pac_t: np.ndarray, sch_t: np.ndarray, sai: np.ndarray, overlap: np.ndarray) -> Tuple[np.ndarray, np.ndarray]`
> Interpolate SAI and overlap to the PAC timebase.

### `run_pac_vs_schumann(records: pd.DataFrame, fused: Dict[str, object], electrodes: Optional[List[str]], time_col: str, pac_pairs: Optional[Dict[str, Tuple[Tuple[float, float], Tuple[float, float]]]], ...) -> Dict[str, object]`

### `plot_pac_timeseries(t: np.ndarray, pac_ts: Dict[str, np.ndarray], sai: np.ndarray) -> None`

### `plot_pac_overlap_etas(t: np.ndarray, pac_ts: Dict[str, np.ndarray], overlap: np.ndarray, k_tiers: List[int]) -> None`
> Event-triggered averages of PAC around overlap K≥tiers onsets.

### `onsets(mask)`

### `bandpass(x: np.ndarray, fs: float, f1: float, f2: float, order: int) -> np.ndarray`

### `pac_mi_single(x_phase: np.ndarray, x_amp: np.ndarray, nbins: int) -> float`

### `sliding_windows(n: int, win: int, step: int) -> List[Tuple[int, int]]`

### `compute_pac_ts_custom(X: np.ndarray, fs: float, electrodes: List[str], phase_band: Tuple[float, float], amp_band: Tuple[float, float], ...) -> Tuple[np.ndarray, np.ndarray]`

### `xcorr_peak(y: np.ndarray, x: np.ndarray, max_lag: int) -> Tuple[int, float, np.ndarray, np.ndarray]`
> Return (best_lag_samples, peak_r, lags, r) for cross-correlation of y vs x within ±max_lag.

### `run_ridge_pac_coupling(RECORDS: pd.DataFrame, fused: Dict[str, object], electrodes: Optional[List[str]], time_col: str, pac_pairs: Optional[Dict[str, Tuple[Tuple[float, float], Tuple[float, float]]]], ...) -> Dict[str, object]`

## Private/Helper Functions

- `_get_fs(records: pd.DataFrame, time_col: str)`
- `_smooth(x: np.ndarray, fs: float, smooth_sec: float)`
- `_get_fs(RECORDS: pd.DataFrame, time_col: str)`
- `_smooth_ts(y: np.ndarray, fs_eff: float, smooth_sec: float)`
