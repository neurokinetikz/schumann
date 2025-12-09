# information_flow

**Total functions:** 25 (17 public, 8 private)

## Public Functions

### `infer_fs(RECORDS: pd.DataFrame, time_col: str) -> float`

### `get_series(RECORDS: pd.DataFrame, name: str) -> np.ndarray`
> Return numeric signal. Accepts 'EEG.O1' or bare 'O1' (tries 'EEG.O1').

### `slice_concat(x: np.ndarray, fs: float, windows: Optional[List[Tuple[float, float]]]) -> np.ndarray`

### `stack_channels(RECORDS: pd.DataFrame, channels: List[str], fs: float, windows: Optional[List[Tuple[float, float]]], demean: bool) -> np.ndarray`

### `fit_var_model(X: np.ndarray, order_max: int, crit: str) -> Dict[str, object]`
> Fit VAR to (n_ch, L) array X.T using statsmodels.

### `spectral_dtf_pdc(A: np.ndarray, Sigma_u: np.ndarray, fs: float, fmin: float, fmax: float, ...) -> Dict[str, np.ndarray]`
> Compute DTF and PDC spectra from VAR(A, Sigma_u).

### `summarize_dtf_pdc_at_harmonics(spec: Dict[str, np.ndarray], harm: List[float]) -> pd.DataFrame`

### `run_freq_granger_pdc_dtf(RECORDS: pd.DataFrame, channels: List[str], windows: Optional[List[Tuple[float, float]]], time_col: str, order_max: int, ...) -> Dict[str, object]`
> Fit VAR, compute DTF/PDC spectra, and report harmonic values.

### `transfer_entropy_knn(x: np.ndarray, y: np.ndarray, lag: int, k_embed_x: int, k_embed_y: int, ...) -> float`
> TE X→Y at a given *positive* lag (samples): predicts y_{t+lag} from [y_t^(k_embed_y), x_t^(k_embed_x)].

### `embed(sig, kdim)`

### `run_transfer_entropy(RECORDS: pd.DataFrame, x_channel: str, y_channel: str, windows: Optional[List[Tuple[float, float]]], time_col: str, ...) -> Dict[str, object]`
> Compute TE(X→Y) and TE(Y→X) across lags (ms). Returns arrays and surrogate 95% thresholds.

### `kalman_rls_tvar_ar(X: np.ndarray, order: int, lam: float) -> Dict[str, object]`
> Track time-varying AR coefficients for a multivariate series X (n_ch, L)

### `phi_t(t_idx)`

### `dtf_at_freq_from_A(A: np.ndarray, fs: float, f0: float) -> np.ndarray`
> DTF at a single frequency f0 from AR matrices A (p,n,n).

### `run_tvar_dtf(RECORDS: pd.DataFrame, channels: List[str], windows: Optional[List[Tuple[float, float]]], time_col: str, order: int, ...) -> Dict[str, object]`
> Time-varying AR (Kalman-RLS) and DTF(t, i<-j) at f0.

### `transfer_entropy_knn(x: np.ndarray, y: np.ndarray, lag: int, k_embed_x: int, k_embed_y: int, ...) -> float`
> TE X→Y at positive sample lag.

### `plot_dtf_grid_bidir_like_single(tv_ign: Dict[str, object], tv_base: Optional[Dict[str, object]], src_channel: str, smooth_sec: float, n_cols: int, ...) -> None`
> Grid of DTF(t, target <- source) for all targets (all electrodes except source).

## Private/Helper Functions

- `_lb_last_pvalue(x: np.ndarray, max_lag: int)`
- `_A_of_f(A: np.ndarray, f: np.ndarray, fs: float)`
- `_H_of_f(Af: np.ndarray)`
- `_knn_entropy(points: np.ndarray, k: int)`
- `_knn_entropy(points: np.ndarray, k: int)`
- `_embed(sig: np.ndarray, kdim: int)`
- `_moving_average(x: np.ndarray, win: int)`
- `_mean_in_seconds(x: np.ndarray, fs: float, windows: Optional[List[Tuple[float, float]]])`
