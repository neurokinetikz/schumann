# dynamic_connectivity_metastability

**Total functions:** 18 (17 public, 1 private)

## Public Functions

### `detect_time_col(df, candidates) -> Optional[str]`

### `ensure_timestamp_column(df: pd.DataFrame, time_col: Optional[str], default_fs: float, out_name) -> str`

### `infer_fs(df: pd.DataFrame, time_col: str) -> float`

### `get_series(df: pd.DataFrame, name: str) -> np.ndarray`

### `slice_concat(x: np.ndarray, fs: float, wins: Optional[List[Tuple[float, float]]]) -> np.ndarray`

### `zscore(x)`

### `bandpass(x, fs, f1, f2, order)`

### `analytic_phase(x, fs, f1, f2)`

### `pli_window(Xb: np.ndarray) -> np.ndarray`
> PLI on analytic phases; Xb: (n_ch, W) bandpassed.

### `imagcoh_window(Xb: np.ndarray) -> np.ndarray`
> Imag coherency; Xb: (n_ch, W) bandpassed.

### `sliding_windows(X: np.ndarray, fs: float, win_sec: float, step_sec: float)`

### `kuramoto_R(Xb: np.ndarray) -> float`

### `phase_randomize(x: np.ndarray) -> np.ndarray`

### `build_surrogate_matrix(X: np.ndarray) -> np.ndarray`

### `run_dynamic_connectivity_metastability(RECORDS: pd.DataFrame, eeg_channels: List[str], ignition_windows: Optional[List[Tuple[float, float]]], baseline_windows: Optional[List[Tuple[float, float]]], band: Tuple[float, float], ...) -> Dict[str, object]`
> Dynamic connectivity & metastability with simple graphs and tests.

### `build_state_matrix(wins)`

### `dyn_conn_for_state(X: np.ndarray, names: List[str], label: str)`

## Private/Helper Functions

- `_ensure_dir(d)`
