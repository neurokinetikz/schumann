# network_graph_hubs

**Total functions:** 20 (19 public, 1 private)

## Public Functions

### `detect_time_col(df, candidates) -> Optional[str]`

### `ensure_timestamp_column(df: pd.DataFrame, time_col: Optional[str], default_fs: float, out_name: str) -> str`

### `infer_fs(df: pd.DataFrame, time_col: str) -> float`

### `get_series(df: pd.DataFrame, name: str) -> np.ndarray`

### `slice_concat(x: np.ndarray, fs: float, wins: Optional[List[Tuple[float, float]]]) -> np.ndarray`

### `zscore(x)`

### `bandpass(x, fs, f1, f2, order)`

### `pli_connectivity(X: np.ndarray, fs: float, f1: float, f2: float) -> np.ndarray`
> PLI from band-passed analytic phases. X: (n_ch, T)

### `imagcoh_connectivity(X: np.ndarray, fs: float, f1: float, f2: float) -> np.ndarray`
> Imag coherency from analytic signals. Robust to zero-lag.

### `threshold_by_density(W: np.ndarray, density: float) -> np.ndarray`
> Keep top fraction of weights (upper triangle) to reach target density.

### `graph_from_weighted(W: np.ndarray) -> nx.Graph`

### `global_efficiency_weighted(G: nx.Graph) -> float`
> Weighted global efficiency: mean of 1/d_ij on finite shortest paths (length attr).

### `char_path_length_weighted(G: nx.Graph) -> float`
> Weighted characteristic path length using 'length' as distance.

### `clustering_weighted(G: nx.Graph) -> float`

### `modularity_greedy(G: nx.Graph) -> Tuple[Dict[int, int], float]`
> Greedy modularity communities (weighted). Returns membership dict and Q.

### `participation_coeff(W: np.ndarray, memb: Dict[int, int]) -> np.ndarray`
> Weighted participation coefficient: 1 - sum_s (k_is/k_i)^2

### `small_world_sigma(G: nx.Graph, n_rewire: int) -> Tuple[float, float, float]`
> Small-world index σ = (C/C_rand)/(L/L_rand) using degree-preserving nulls.

### `compute_connectivity(RECORDS: pd.DataFrame, channels: List[str], wins, band: Tuple[float, float], method: str, ...) -> Tuple[np.ndarray, List[str], float]`
> Returns (W, chan_names, fs) — symmetric connectivity matrix in [0,1].

### `run_graph_metrics_hubs(RECORDS: pd.DataFrame, eeg_channels: List[str], ignition_windows: Optional[List[Tuple[float, float]]], baseline_windows: Optional[List[Tuple[float, float]]], bands: Dict[str, Tuple[float, float]], ...) -> Dict[str, object]`
> Build functional graphs per band and state; compute small-worldness, clustering,

## Private/Helper Functions

- `_ensure_dir(d)`
