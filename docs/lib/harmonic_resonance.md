# harmonic_resonance

**Total functions:** 11 (10 public, 1 private)

## Public Functions

### `detect_time_col(df, candidates) -> Optional[str]`

### `ensure_timestamp_column(df: pd.DataFrame, time_col: Optional[str], default_fs: float, out_name: str) -> str`

### `infer_fs(df: pd.DataFrame, time_col: str) -> float`

### `get_series(df: pd.DataFrame, name: str) -> np.ndarray`

### `slice_concat(x: np.ndarray, fs: float, wins: Optional[List[Tuple[float, float]]]) -> np.ndarray`

### `bandpass(x, fs, f1, f2, order)`

### `welch_psd(x: np.ndarray, fs: float, nperseg_sec: float) -> Tuple[np.ndarray, np.ndarray]`

### `harmonic_zscores(f: np.ndarray, p: np.ndarray, harmonics, half_bw: float, side_bw: float) -> Dict[str, float]`
> For each target harmonic h, compute z = (P(h) - median(side)) / MAD(side),

### `spatial_mode_8hz(X: np.ndarray, fs: float, f0, half) -> Dict[str, object]`
> X: (n_ch, T) — band-pass 7.83±half, compute covariance → PC1 variance ratio,

### `run_harmonic_resonance_spectral_modes(RECORDS: pd.DataFrame, eeg_channels: List[str], time_col: str, ignition_windows: Optional[List[Tuple[float, float]]], nperseg_sec: float, ...) -> Dict[str, object]`
> High-resolution spectral harmonic test + spatial mode at 7–8 Hz.

## Private/Helper Functions

- `_ensure_dir(d)`
