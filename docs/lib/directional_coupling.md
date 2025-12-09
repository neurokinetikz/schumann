# directional_coupling

**Total functions:** 11 (8 public, 3 private)

## Public Functions

### `find_channel_series(records: pd.DataFrame, ch_name: str) -> Optional[pd.Series]`

### `default_clusters() -> Dict[str, List[str]]`
> Default sensor clusters (10–20ish). Feel free to override.

### `cluster_signal(RECORDS: pd.DataFrame, time_col: str, names: List[str]) -> np.ndarray`

### `dpli_block(x_src: np.ndarray, x_tgt: np.ndarray, fs: float, f1: float, f2: float) -> float`
> Directed PLI (src→tgt) in [f1,f2]. Returns fraction in [0,1], 0.5 ~ no direction.

### `granger_block(x_src: np.ndarray, x_tgt: np.ndarray, order: int) -> float`
> Pairwise GC advantage: F(src→tgt) − F(tgt→src). Requires statsmodels; else returns nan.

### `slice_windows_idx(t: np.ndarray, fs: float, windows: List[Tuple[float, float]]) -> List[Tuple[int, int]]`

### `run_directional_coupling_rdlfpc_sensory(RECORDS: pd.DataFrame, ignition_windows: List[Tuple[float, float]], rebound_windows: Optional[List[Tuple[float, float]]], control_windows: Optional[List[Tuple[float, float]]], time_col: str, ...) -> Dict[str, object]`

### `plot_directional_deltas(df: pd.DataFrame) -> None`

## Private/Helper Functions

- `_get_fs(RECORDS: pd.DataFrame, time_col: str)`
- `_bandpass(x: np.ndarray, fs: float, f1: float, ...)`
- `_find_series(RECORDS: pd.DataFrame, ch: str)`
