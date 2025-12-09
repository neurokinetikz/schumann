# network_geometry

**Total functions:** 19 (16 public, 3 private)

## Public Functions

### `infer_fs_from_records(RECORDS: pd.DataFrame, time_col: str) -> float`

### `find_channel_series(RECORDS: pd.DataFrame, ch_name: str) -> Optional[pd.Series]`

### `bandpass_guard(x: np.ndarray, fs: float, f1: float, f2: float, order: int) -> np.ndarray`

### `compute_wpli_records(X: np.ndarray, sf: float, fmin: float, fmax: float) -> np.ndarray`
> wPLI on (n_channels, n_times). Multitaper via mne_connectivity if available.

### `graph_entropy(adj: np.ndarray) -> float`

### `minimal_cut_weight(adj: np.ndarray) -> float`

### `connectome_harmonics(adj: np.ndarray, n_modes: int) -> Tuple[np.ndarray, np.ndarray]`

### `run_network_geometry_suite_records(RECORDS: pd.DataFrame, ignition_windows: List[Tuple[float, float]], rebound_windows: Optional[List[Tuple[float, float]]], fband: Tuple[float, float], electrodes: Optional[List[str]], ...) -> Dict[str, object]`

### `run_state_space_embedding_records(RECORDS: pd.DataFrame, ignition_windows: List[Tuple[float, float]], baseline_windows: Optional[List[Tuple[float, float]]], electrodes: Optional[List[str]], n_components: int, ...) -> Dict[str, np.ndarray]`

### `session_report_records(RECORDS: pd.DataFrame, ignition_windows: List[Tuple[float, float]], rebound_windows: Optional[List[Tuple[float, float]]], fband: Tuple[float, float], electrodes: Optional[List[str]], ...) -> Dict[str, object]`

### `run_multi_band_geometry_records(RECORDS: pd.DataFrame, ignition_windows: List[Tuple[float, float]], rebound_windows: Optional[List[Tuple[float, float]]], bands: Optional[Dict[str, Tuple[float, float]]], electrodes: Optional[List[str]], ...) -> pd.DataFrame`
> Run network geometry over multiple frequency bands and return a tidy DataFrame.

### `save_session_report_csv_json(report: Dict[str, object], base_path: str) -> Tuple[str, str]`
> Write report CSV and JSON to `base_path` (without extension). Returns paths.

### `save_embedding_plot(emb: Dict[str, np.ndarray], filename: str) -> str`
> Save a 2D/3D scatter plot of embeddings returned by run_state_space_embedding_records.

### `animate_embedding_over_time_records(RECORDS: pd.DataFrame, window_sec: float, step_sec: float, electrodes: Optional[List[str]], method: str, ...) -> Dict[str, object]`
> Create an embedding over sliding windows and save frames (and optional GIF).

### `run_full_session_with_bands_and_exports(RECORDS: pd.DataFrame, ignition_windows: List[Tuple[float, float]], rebound_windows: Optional[List[Tuple[float, float]]], electrodes: Optional[List[str]], embed_method: str, ...) -> Dict[str, object]`
> Run single‑band report, multi‑band table, embedding PNG, and write CSV/JSON.

### `animate_embedding_over_time_records(RECORDS, window_sec: float, step_sec: float, electrodes: List[str] | None, method: str, ...) -> Dict[str, object]`

## Private/Helper Functions

- `_discover_electrodes(RECORDS: pd.DataFrame, time_col: str)`
- `_slice_windows(data: np.ndarray, t: np.ndarray, wins: List[Tuple[float, float]])`
- `_slice(wins: List[Tuple[float, float]])`
