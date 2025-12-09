# surface_cuts

**Total functions:** 13 (5 public, 8 private)

## Public Functions

### `find_channel_series(records: pd.DataFrame, ch_name: str) -> Optional[pd.Series]`

### `default_clusters(electrodes: List[str]) -> Dict[str, List[int]]`
> Map label-> indices into electrodes list for broad subsystems F, P, O, T.

### `run_multi_seed_surface_cuts(RECORDS: pd.DataFrame, ignition_windows: List[Tuple[float, float]], rebound_windows: Optional[List[Tuple[float, float]]], time_col: str, bands: Optional[Dict[str, Tuple[float, float]]], ...) -> Dict[str, object]`

### `adj_for_windows(windows: List[Tuple[float, float]], f1: float, f2: float) -> np.ndarray`

### `plot_multicut_deltas(df: pd.DataFrame, shuffle: Optional[pd.DataFrame]) -> None`

## Private/Helper Functions

- `_get_fs(RECORDS: pd.DataFrame, time_col: str)`
- `_autoelectrodes(RECORDS: pd.DataFrame, time_col: str)`
- `_bandpass(X: np.ndarray, fs: float, f1: float, ...)`
- `_pseudo_wpli(Xb: np.ndarray)`
- `_set_set_mincut_capacity(GH: nx.Graph, A: List[int], B: List[int])`
- `_multi_seed_capacity(A: np.ndarray, clusters_idx: Dict[str, List[int]])`
- `_weight_permute_surrogate(A: np.ndarray, rng: np.random.Generator)`
- `_degree_rewire_surrogate(A: np.ndarray, density: float, rng: np.random.Generator, ...)`
