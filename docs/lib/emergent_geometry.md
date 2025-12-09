# emergent_geometry

**Total functions:** 16 (9 public, 7 private)

## Public Functions

### `find_channel_series(records: pd.DataFrame, ch_name: str) -> Optional[pd.Series]`

### `plv_distance_matrix(X: np.ndarray, fs: float, band: Tuple[float, float]) -> np.ndarray`
> Return D = 1 − |PLV| for band-limited analytic phases across channels.

### `trustworthiness_continuity(D_high: np.ndarray, D_low: np.ndarray, k: int) -> Tuple[float, float]`
> Compute trustworthiness & continuity from distance matrices.

### `geodesic_stress(D_high: np.ndarray, X_low: np.ndarray, k: int) -> float`
> Compute normalized stress between high‑D geodesic distances and low‑D Euclidean distances.

### `embed_distance_matrix(D: np.ndarray, method: str, n_neighbors: int, n_components: int, random_state: int) -> np.ndarray`

### `run_phase_embedding_emergent_geometry(RECORDS: pd.DataFrame, ignition_windows: List[Tuple[float, float]], rebound_windows: Optional[List[Tuple[float, float]]], control_windows: Optional[List[Tuple[float, float]]], time_col: str, ...) -> Dict[str, object]`

### `analyze_state(blocks: List[np.ndarray], do_surrogates: bool) -> Dict[str, object]`
> Build PLV distance D, embed to low-D (Isomap/UMAP), and compute quality metrics.

### `plot_phase_embedding_quality(res: Dict[str, object]) -> None`

### `embed_distance_matrix(D: np.ndarray, method: str, n_neighbors: int, n_components: int, random_state: int) -> np.ndarray`
> Embed a precomputed distance matrix D.

## Private/Helper Functions

- `_get_fs(RECORDS: pd.DataFrame, time_col: str)`
- `_autoelectrodes(RECORDS: pd.DataFrame, time_col: str)`
- `_bandpass(X: np.ndarray, fs: float, f1: float, ...)`
- `_slice_blocks(RECORDS: pd.DataFrame, time_col: str, X: np.ndarray, ...)`
- `_fourier_phase_randomize_1d(x: np.ndarray, rng: np.random.Generator)`
- `_make_surrogate_concat(concat: np.ndarray, rng: np.random.Generator)`
- `_rank_matrix(D: np.ndarray)`
