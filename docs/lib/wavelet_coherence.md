# wavelet_coherence

**Total functions:** 10 (9 public, 1 private)

## Public Functions

### `infer_fs(df: pd.DataFrame, time_col) -> float`

### `get_series(df: pd.DataFrame, name: str) -> np.ndarray`

### `slice_concat(x: np.ndarray, fs: float, wins: Optional[List[Tuple[float, float]]])`

### `bandpass(x, fs, f1, f2, order)`

### `msc_harmonics_table(df, eeg_channels, sr_channel, wins, time_col, ...) -> pd.DataFrame`

### `cwt_linear(freqs, sig: np.ndarray, w0, N, fs) -> np.ndarray`

### `smooth(A: np.ndarray, wlen: int) -> np.ndarray`

### `wavelet_coherence_tf(df, x_name, y_name, time_col, fmin, ...) -> Dict[str, object]`

### `plot_sr_ignition_wtc_strip(RECORDS, eeg_channel: str, sr_channel: str, ignition_windows: list, time_col: str, ...)`

## Private/Helper Functions

- `_ensure_dir(d)`
