# informational_geometry

**Total functions:** 18 (17 public, 1 private)

## Public Functions

### `infer_fs(RECORDS: pd.DataFrame, time_col: str) -> float`

### `get_series(RECORDS: pd.DataFrame, name: str) -> np.ndarray`

### `bandpass(x: np.ndarray, fs: float, f1: float, f2: float, order) -> np.ndarray`

### `zscore(x: np.ndarray) -> np.ndarray`

### `schumann_envelope(sr: np.ndarray, fs: float, center, half_bw) -> np.ndarray`

### `slice_windows(RECORDS: pd.DataFrame, time_col: str, fs: float, win_sec: float, step_sec: float) -> List[Tuple[int, int, float]]`
> Return list of index windows (s,e, t_center).

### `in_any_window(t: float, windows: List[Tuple[float, float]]) -> bool`

### `plv_graph_entropy(RECORDS, eeg_channels, fs, s, e, ...)`
> Build PLV adjacency on [s:e] and return Laplacian spectral entropy.

### `window_features(RECORDS: pd.DataFrame, eeg_channels: List[str], sr_channel: str, time_col: str, win_sec: float, ...) -> pd.DataFrame`
> Build a feature vector per sliding window:

### `embed_states(F: pd.DataFrame, method: str, n_neighbors: int, n_components: int, random_state: int) -> Dict[str, np.ndarray]`
> Embed feature matrix (T×D) to 2D/3D. Returns {'X':coords, 'method':..., 'components':...}

### `trust_continuity(D_high: np.ndarray, D_low: np.ndarray, k: int) -> Tuple[float, float]`
> Trustworthiness & Continuity (Tenenbaum / van der Maaten).

### `ranks(D)`

### `geodesic_stress(X_high: np.ndarray, X_low: np.ndarray, k: int) -> float`
> Geodesic stress between k-NN geodesics (from high-D feature space) and Euclidean in embedding.

### `curvature_proxy(X_high: np.ndarray, X_low: np.ndarray, k: int) -> float`
> Mean relative geodesic stretch over k-NN: mean_i mean_{j in N_i} (d_geo - d_euc)/d_euc.

### `entropy_2d_embed(Z: np.ndarray, bins: int) -> float`
> Entropy of the embedded distribution via 2D histogram (Shannon, base e).

### `logeuclidean_spread(cov_list: List[np.ndarray]) -> float`
> Spread on SPD manifold (Log-Euclidean): mean pairwise ||log(S_i) − log(S_j)||_F.

### `run_info_geometry_state_manifolds(RECORDS: pd.DataFrame, eeg_channels: List[str], ignition_windows: List[Tuple[float, float]], baseline_windows: List[Tuple[float, float]], sr_channel: Optional[str], ...) -> Dict[str, object]`
> Build state vectors → embeddings → info-geom metrics, with simple graphs + tests.

## Private/Helper Functions

- `_ensure_dir(d)`
