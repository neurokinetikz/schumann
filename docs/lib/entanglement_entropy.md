# entanglement_entropy

**Total functions:** 16 (15 public, 1 private)

## Public Functions

### `infer_fs(RECORDS: pd.DataFrame, time_col: str) -> float`

### `get_series(RECORDS: pd.DataFrame, name: str) -> np.ndarray`

### `slice_concat(x: np.ndarray, fs: float, windows: Optional[List[Tuple[float, float]]]) -> np.ndarray`

### `bandpass(x: np.ndarray, fs: float, f1: float, f2: float, order) -> np.ndarray`

### `zscore(x: np.ndarray) -> np.ndarray`

### `plv_matrix(RECORDS, channels, band, windows, time_col) -> np.ndarray`

### `laplacian_spectral_entropy(A: np.ndarray) -> float`

### `gaussian_entropies(X: np.ndarray) -> Dict[str, float]`
> X: (n_ch, T) z-scored

### `pca_first_component(X: np.ndarray) -> np.ndarray`

### `lz_complexity_binary(seq: np.ndarray) -> float`
> LZ76 complexity of a binary sequence (0/1), normalized by n/log2(n).

### `permutation_entropy(x: np.ndarray, m: int, tau: int) -> float`
> Band-limited x; simple permutation entropy of order m (permutation count m! bins).

### `circular_shift_null(xmat: np.ndarray, n_surr: int) -> List[np.ndarray]`
> Circularly shift each channel independently; returns list of surrogates (n_ch, T).

### `run_integration_analogs(RECORDS: pd.DataFrame, eeg_channels: List[str], band: Tuple[float, float], ignition_windows: Optional[List[Tuple[float, float]]], baseline_windows: Optional[List[Tuple[float, float]]], ...) -> Dict[str, object]`
> Compute integration/complexity measures and simple tests; produce figures + CSV summary.

### `compute_state(wins, state_name)`

### `pval(obs, null)`

## Private/Helper Functions

- `_ensure_dir(d)`
