# cross_frequency_region_coupling

**Total functions:** 24 (22 public, 2 private)

## Public Functions

### `detect_time_col(df, candidates) -> Optional[str]`

### `ensure_timestamp_column(df: pd.DataFrame, time_col: Optional[str], default_fs: float, out_name) -> str`

### `infer_fs(df: pd.DataFrame, time_col: str) -> float`

### `get_series(df: pd.DataFrame, name: str) -> np.ndarray`

### `slice_concat(x: np.ndarray, fs: float, wins: Optional[List[Tuple[float, float]]]) -> np.ndarray`

### `zscore(x)`

### `bandpass(x, fs, f1, f2, order)`

### `analytic_phase_amp(x, fs, f1, f2)`

### `pac_mi_phase_amp(phase: np.ndarray, amp: np.ndarray, nbins: int) -> float`

### `n_m_plv(phi1: np.ndarray, phi2: np.ndarray, n: int, m: int) -> float`

### `comodulogram_pair(x_phase: np.ndarray, y_amp: np.ndarray, fs: float, slow_range, slow_bw, ...)`
> Phase@x vs Amp@y PAC with an explicit upper clamp for fast frequencies.

### `bandpass(x, f1, f2, order)`

### `analytic_phase_amp(x, f1, f2)`

### `nm_plv_grid(x1: np.ndarray, x2: np.ndarray, fs: float, f1_range, f1_bw, ...)`
> Phase locking grid with an explicit upper clamp for f2 frequencies.

### `bandpass(x, f1, f2, order)`

### `analytic_phase(x, f1, f2)`

### `run_cfc_cross_region(RECORDS, pairs, ignition_windows, baseline_windows, time_col, ...)`
> Same orchestrator as before, but enforces a ≤limit_high_hz scan in PAC/PLV.

### `ensure_timestamp_column(df, time_col, default_fs: float, out_name)`

### `infer_fs(df, time_col: str) -> float`

### `get_series(df, name: str) -> np.ndarray`

### `zscore(x)`

### `get_sig(name, wins)`

## Private/Helper Functions

- `_ensure_dir(d)`
- `_ensure_dir(d)`
