# attractor_topology

**Total functions:** 19 (18 public, 1 private)

## Public Functions

### `infer_fs(RECORDS: pd.DataFrame, time_col: str) -> float`

### `get_series(RECORDS: pd.DataFrame, name: str) -> np.ndarray`

### `slice_concat(x: np.ndarray, fs: float, windows: Optional[List[Tuple[float, float]]]) -> np.ndarray`

### `zscore(x)`

### `detect_time_col(RECORDS: pd.DataFrame, candidates) -> str | None`

### `ensure_timestamp_column(RECORDS: pd.DataFrame, time_col: str | None, default_fs: float, out_name: str) -> str`
> Ensure RECORDS[out_name] exists as numeric seconds (t=0 at first sample).

### `estimate_delay_tau(x: np.ndarray, fs: float, max_lag_sec: float, method: str) -> int`
> Pick τ from the first time where autocorrelation falls below 1/e (default) or crosses 0.

### `takens_embedding(x: np.ndarray, m: int, tau: int) -> np.ndarray`
> Return (N_eff, m) embedded matrix: [x_t, x_{t+τ}, ..., x_{t+(m-1)τ}]

### `false_nearest_neighbors(x: np.ndarray, tau: int, m_list: List[int], theiler: int) -> pd.DataFrame`
> Simple FNN percentage vs m. If sklearn is present, uses KDTree; else brute force sample.

### `correlation_dimension_gp(X: np.ndarray, r_min_quant: float, r_max_quant: float, n_r: int, max_pairs: int) -> Dict[str, object]`
> Grassberger-Procaccia correlation sum C(r) and slope (D2) over a mid-range.

### `lyapunov_rosenstein(x: np.ndarray, m: int, tau: int, fs: float, theiler: int, ...) -> Dict[str, object]`
> Largest Lyapunov exponent (Rosenstein et al.).

### `recurrence_plot(X: np.ndarray, eps_quant: float) -> Dict[str, object]`
> Binary RP thresholded at eps = quantile(eps_quant) of distances.

### `persistent_homology_summary(X: np.ndarray, maxdim: int) -> Dict[str, object]`
> If ripser is installed, compute persistence and return diagrams + simple counts.

### `count_persistent(dgm, thr)`

### `phase_randomize(x: np.ndarray) -> np.ndarray`

### `time_shuffle(x: np.ndarray) -> np.ndarray`

### `metric_vs_surrogates(metric_func, x: np.ndarray, n_surr: int, kind: str) -> Tuple[float, float]`
> Compute metric on x, build null from surrogates (phase or shuffle). Return (value, p-value).

### `run_attractor_topology(RECORDS: pd.DataFrame, eeg_channels: List[str], ignition_windows: Optional[List[Tuple[float, float]]], baseline_windows: Optional[List[Tuple[float, float]]], time_col: str, ...) -> Dict[str, object]`
> Build attractor embeddings and tests for Ignition and Baseline.

## Private/Helper Functions

- `_ensure_dir(d)`
