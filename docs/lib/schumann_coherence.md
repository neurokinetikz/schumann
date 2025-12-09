# schumann_coherence

**Total functions:** 13 (12 public, 1 private)

## Public Functions

### `infer_fs(df: pd.DataFrame, time_col) -> float`

### `get_series(df: pd.DataFrame, name: str) -> np.ndarray`

### `slice_concat(x: np.ndarray, fs: float, wins: Optional[List[Tuple[float, float]]])`

### `bandpass(x, fs, f1, f2, order)`

### `msc_harmonics_table(df, eeg_channels, sr_channel, wins, time_col, ...) -> pd.DataFrame`

### `wavelet_coherence_tf(df, x_name, y_name, time_col, fmin, ...) -> Dict[str, object]`

### `cwt_linear(sig: np.ndarray) -> np.ndarray`

### `smooth(A: np.ndarray, wlen: int) -> np.ndarray`

### `sliding_coherence_f0(df, eeg_channel, sr_channel, ignition_windows, f0, ...)`

### `compute_coherence_at_f0(xe, xs, fs, f0, half)`
> Magnitude-squared coherence between signals xe and xs at target frequency f0.

### `build_null_threshold(coh, n_null, method, block_len, alpha, ...)`
> Estimate a null threshold for a sliding coherence trace by resampling the

### `run_eeg_schumann_coherence(RECORDS: pd.DataFrame, eeg_channels: List[str], sr_channel: str, ignition_windows: Optional[List[Tuple[float, float]]], baseline_windows: Optional[List[Tuple[float, float]]], ...) -> Dict[str, object]`

## Private/Helper Functions

- `_ensure_dir(d)`
