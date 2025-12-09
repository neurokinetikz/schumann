# microstate_segmentation

**Total functions:** 18 (17 public, 1 private)

## Public Functions

### `detect_time_col(df, candidates) -> Optional[str]`

### `ensure_timestamp_column(df: pd.DataFrame, time_col: Optional[str], default_fs: float, out_name: str) -> str`

### `infer_fs(df: pd.DataFrame, time_col: str) -> float`

### `get_series(df: pd.DataFrame, name: str) -> np.ndarray`

### `slice_concat(x: np.ndarray, fs: float, wins: Optional[List[Tuple[float, float]]]) -> np.ndarray`

### `zscore(x)`

### `bandpass(x, fs, f1, f2, order)`

### `gfp(X: np.ndarray) -> np.ndarray`
> Global Field Power across channels per time (std). X: (n_ch, T)

### `pick_gfp_peaks(G: np.ndarray, skip: int) -> np.ndarray`
> Pick timepoints at local maxima of GFP with a refractory 'skip' in samples.

### `normalize_maps(X: np.ndarray) -> np.ndarray`
> Zero-mean & L2-normalize topographies columnwise. X: (n_ch, Nmaps)

### `kmeans_microstates(Xmaps: np.ndarray, k: int, n_init: int, seed: int) -> Tuple[np.ndarray, np.ndarray]`
> KMeans on normalized maps (channels×N) → centers (channels×k), labels (N,)

### `backfit_sequence(X: np.ndarray, centers: np.ndarray) -> Tuple[np.ndarray, np.ndarray]`
> Assign each timepoint to the map with max |corr| (polarity-invariant).

### `smooth_labels(labels: np.ndarray, fs: float, min_dur_ms: float) -> np.ndarray`
> Enforce minimum segment duration by merging short runs into neighbors.

### `microstate_metrics(labels: np.ndarray, fs: float, k: int) -> Dict[str, object]`
> Mean duration (ms), coverage, occurrence rate (/s), transition matrix, sequence entropy (bits).

### `gev_score(GFP: np.ndarray, corr_abs: np.ndarray) -> float`
> Global Explained Variance: sum(GFP^2 * corr^2)/sum(GFP^2).

### `run_microstate_segmentation(RECORDS: pd.DataFrame, eeg_channels: List[str], ignition_windows: Optional[List[Tuple[float, float]]], baseline_windows: Optional[List[Tuple[float, float]]], band: Optional[Tuple[float, float]], ...) -> Dict[str, object]`
> Microstate maps & metrics with surrogate validation; Ignition/Baseline comparison.

### `build_X(wins)`

## Private/Helper Functions

- `_ensure_dir(d)`
