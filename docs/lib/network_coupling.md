# network_coupling

**Total functions:** 12 (12 public, 0 private)

## Public Functions

### `infer_fs(RECORDS: pd.DataFrame, time_col: str) -> float`

### `get_series(RECORDS: pd.DataFrame, name: str) -> np.ndarray`
> Return a numeric signal array. Accepts 'EEG.O1' or bare 'O1'.

### `slice_concat(x: np.ndarray, fs: float, windows: Optional[List[Tuple[float, float]]]) -> np.ndarray`

### `bandpass(x: np.ndarray, fs: float, f1: float, f2: float, order: int) -> np.ndarray`

### `plv_matrix(RECORDS: pd.DataFrame, channels: List[str], band: Tuple[float, float], windows: Optional[List[Tuple[float, float]]], time_col: str) -> np.ndarray`
> Pairwise PLV matrix (N×N) within 'band'. Uses analytic phases from Hilbert

### `msc_vs_sr(RECORDS: pd.DataFrame, channels: List[str], sr_channel: str, windows: Optional[List[Tuple[float, float]]], time_col: str, ...) -> pd.DataFrame`
> Per-channel magnitude-squared coherence with SR at harmonics

### `laplacian_entropy(adj: np.ndarray) -> float`
> Shannon entropy of positive Laplacian eigenvalues (normalized).

### `global_mincut(adj: np.ndarray) -> float`
> Stoer–Wagner global min-cut on weighted undirected graph.

### `cross_domain_graph_alignment(RECORDS: pd.DataFrame, eeg_channels: List[str], sr_channel: str, band: Tuple[float, float], harmonics: List[float], ...) -> Dict[str, object]`
> Build EEG PLV graph in 'band'. Weight edges by geometric mean of the nodes'

### `symmetric_orthogonalize(ts: np.ndarray) -> np.ndarray`
> Symmetric orthogonalization (Colclough et al., 2015) for leakage reduction.

### `roi_time_series(RECORDS: pd.DataFrame, roi_map: Dict[str, List[str]], windows: Optional[List[Tuple[float, float]]], time_col: str, orthogonalize: bool) -> Tuple[np.ndarray, List[str], float]`
> Build (n_roi, T) ROI matrix by averaging available channels per ROI,

### `roi_plv_msc_vs_sr(RECORDS: pd.DataFrame, roi_map: Dict[str, List[str]], sr_channel: str, windows: Optional[List[Tuple[float, float]]], time_col: str, ...) -> Dict[str, object]`
> ROI-level PLV (phase_band) and MSC (harmonics) vs SR with circular-shift surrogates.
