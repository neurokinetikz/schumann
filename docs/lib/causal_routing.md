# causal_routing

**Total functions:** 24 (22 public, 2 private)

## Public Functions

### `infer_fs(RECORDS: pd.DataFrame, time_col: str) -> float`

### `get_series(RECORDS: pd.DataFrame, name: str) -> np.ndarray`

### `zscore(x: np.ndarray) -> np.ndarray`

### `slice_concat(x: np.ndarray, fs: float, windows: Optional[List[Tuple[float, float]]]) -> np.ndarray`

### `bandpass(x: np.ndarray, fs: float, f1: float, f2: float, order) -> np.ndarray`

### `pick_posterior_sr(RECORDS: pd.DataFrame) -> str`

### `make_roi_series(RECORDS: pd.DataFrame, eeg_channels: List[str], roi_map: Optional[Dict[str, List[str]]], windows: Optional[List[Tuple[float, float]]], time_col: str) -> Tuple[np.ndarray, List[str], float]`
> Return (X, names, fs) where X is (n_nodes, T) z-scored per node.

### `fit_var_robust(X: np.ndarray, fs: float, order_max: int, trend: str, ridge: float) -> Tuple[object, Optional[int], Optional[np.ndarray]]`
> Robust VAR(p) fit with manual BIC selection and PD enforcement on Σ_u.

### `A_of_f(A: np.ndarray, f: np.ndarray, fs: float) -> np.ndarray`
> A(f) = I − Σ_k A_k e^{−i2πfk/fs};  returns (n_f, n, n).

### `H_of_f(Af: np.ndarray) -> np.ndarray`
> Transfer matrix H(f) = A(f)^{-1}, per frequency.

### `spectral_dtf_pdc(A: np.ndarray, fs: float, fmin: float, fmax: float, n_freq: int) -> Dict[str, np.ndarray]`
> DTF_{i<-j}(f) = |H_ij| / sqrt(Σ_k |H_ik|^2)   (row-normalized)

### `band_average(M: np.ndarray, f: np.ndarray, band: Tuple[float, float]) -> np.ndarray`
> Mean over frequency indices in band; M shape (n,n,n_f).

### `fit_var(X: np.ndarray, order_max: int, criterion: str)`
> Fit VAR to (n_nodes, T) → statsmodels VAR on (T, n_nodes).

### `A_of_f(A: np.ndarray, f: np.ndarray, fs: float) -> np.ndarray`
> A(f) = I − Σ_k A_k e^{−i2πfk/fs};  returns (n_f, n, n).

### `H_of_f(Af: np.ndarray) -> np.ndarray`
> Transfer matrix H(f) = A(f)^{-1}, per frequency.

### `spectral_dtf_pdc(A: np.ndarray, fs: float, fmin: float, fmax: float, n_freq: int) -> Dict[str, np.ndarray]`
> DTF_{i<-j}(f) = |H_ij| / sqrt(Σ_k |H_ik|^2)   (row-normalized)

### `band_average(M: np.ndarray, f: np.ndarray, band: Tuple[float, float]) -> np.ndarray`
> Mean over frequency indices in band; M shape (n,n,n_f).

### `circular_shift_sr_null(RECORDS: pd.DataFrame, sr_channel: str, fs: float, windows: List[Tuple[float, float]], nodes: List[str], ...) -> float`
> Build null distribution for SR→ROI DTF at ~f0 by circularly shifting SR and refitting VAR.

### `run_directed_connectivity_routing(RECORDS: pd.DataFrame, eeg_channels: List[str], ignition_windows: Optional[List[Tuple[float, float]]], baseline_windows: Optional[List[Tuple[float, float]]], roi_map: Optional[Dict[str, List[str]]], ...) -> Dict[str, object]`
> Map directed connectivity inside the brain and between brain and field (SR).

### `pca_reduce_nodes(X: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]`
> X: (n_nodes, T) z-scored. Returns (Z, U) where

### `granger_bivariate_matrix(X: np.ndarray, maxlag: int) -> np.ndarray`
> X: (n_nodes, T) z-scored.

### `fit_var_safeguarded(X: np.ndarray, fs: float, order_max: int, ridge: float) -> Tuple[object, Optional[int], Optional[np.ndarray]]`
> Adaptive lag cap from data length; manual BIC; PD enforcement on Sigma_u.

## Private/Helper Functions

- `_ensure_dir(d)`
- `_align(A, B)`
