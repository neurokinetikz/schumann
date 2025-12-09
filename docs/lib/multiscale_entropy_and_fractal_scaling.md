# multiscale_entropy_and_fractal_scaling

**Total functions:** 15 (14 public, 1 private)

## Public Functions

### `detect_time_col(df, candidates) -> Optional[str]`

### `ensure_timestamp_column(df: pd.DataFrame, time_col: Optional[str], default_fs: float, out_name: str) -> str`

### `infer_fs(df: pd.DataFrame, time_col: str) -> float`

### `get_series(df: pd.DataFrame, name: str) -> np.ndarray`

### `slice_concat(x: np.ndarray, fs: float, wins: Optional[List[Tuple[float, float]]]) -> np.ndarray`

### `zscore(x)`

### `coarse_grain(x: np.ndarray, scale: int) -> np.ndarray`
> Non-overlapping average; drops remainder.

### `sampen(x: np.ndarray, m: int, r_ratio: float) -> float`
> Sample Entropy (m,r) with Chebyshev metric.

### `count_matches(X, tol)`

### `mse_curve(x: np.ndarray, fs: float, max_scale_sec: float, m: int, r_ratio: float, ...) -> Tuple[np.ndarray, np.ndarray]`
> Compute MSE over integer coarse-grain scales up to max_scale_sec.

### `dfa_alpha(x: np.ndarray, fs: float, min_win_sec: float, max_win_sec: float, n_win: int) -> Dict[str, object]`
> Detrended Fluctuation Analysis on z-scored signal (integrated profile).

### `phase_randomize(x: np.ndarray) -> np.ndarray`

### `run_mse_dfa_multiscale(RECORDS: pd.DataFrame, eeg_channels: List[str], ignition_windows: Optional[List[Tuple[float, float]]], baseline_windows: Optional[List[Tuple[float, float]]], time_col: str, ...) -> Dict[str, object]`
> MSE + DFA with surrogate validation, for ignition and baseline windows.

### `build_drive(wins)`

## Private/Helper Functions

- `_ensure_dir(d)`
