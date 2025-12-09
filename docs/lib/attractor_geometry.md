# attractor_geometry

**Total functions:** 16 (15 public, 1 private)

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

### `phase_randomize(x: np.ndarray) -> np.ndarray`

### `recurrence_plot(X: np.ndarray, eps_quant: float) -> Dict[str, object]`

### `run_tda_attractor_topology(RECORDS: pd.DataFrame, eeg_channels: List[str], ignition_windows: Optional[List[Tuple[float, float]]], time_col: str, out_dir: str, ...) -> Dict[str, object]`
> Takens embedding + persistent homology (with surrogates) to test torus-like topology.

### `max_persistence(dgm)`

### `maxP(dgm)`

### `count_sig(dgm, thr)`

## Private/Helper Functions

- `_ensure_dir(d)`
