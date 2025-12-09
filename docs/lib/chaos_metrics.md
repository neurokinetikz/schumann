# chaos_metrics

**Total functions:** 18 (17 public, 1 private)

## Public Functions

### `detect_time_col(df, candidates) -> Optional[str]`

### `ensure_timestamp_column(df: pd.DataFrame, time_col: Optional[str], default_fs: float, out_name: str) -> str`

### `infer_fs(df: pd.DataFrame, time_col: str) -> float`

### `get_series(df: pd.DataFrame, name: str) -> np.ndarray`

### `slice_concat(x: np.ndarray, fs: float, wins: Optional[List[Tuple[float, float]]]) -> np.ndarray`

### `zscore(x)`

### `estimate_delay_tau(x: np.ndarray, fs: float, max_lag_sec: float, method) -> int`

### `takens_embedding(x: np.ndarray, m: int, tau: int) -> np.ndarray`

### `false_nearest_neighbors(x: np.ndarray, tau: int, m_list: List[int], theiler: int) -> pd.DataFrame`

### `lyapunov_rosenstein(x: np.ndarray, m: int, tau: int, fs: float, theiler: int, ...) -> Dict[str, object]`

### `correlation_dimension_gp(X: np.ndarray, r_min_quant: float, r_max_quant: float, n_r: int) -> Dict[str, object]`

### `recurrence_matrix(X: np.ndarray, eps: float, theiler: int) -> np.ndarray`

### `rqa_metrics(R: np.ndarray, lmin: int, vmin: int) -> Dict[str, float]`

### `phase_randomize(x: np.ndarray) -> np.ndarray`

### `run_rqa_chaos_metrics(RECORDS: pd.DataFrame, eeg_channels: List[str], ignition_windows: Optional[List[Tuple[float, float]]], baseline_windows: Optional[List[Tuple[float, float]]], time_col: str, ...) -> Dict[str, object]`
> RQA + Chaos metrics with surrogate validation for ignition & baseline windows.

### `build_drive(wins)`

### `pval(obs, arr, greater)`

## Private/Helper Functions

- `_ensure_dir(d)`
