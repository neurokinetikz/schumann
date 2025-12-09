# toroidal_phase

**Total functions:** 12 (2 public, 10 private)

## Public Functions

### `find_channel_series(records: pd.DataFrame, ch_name: str) -> Optional[pd.Series]`

### `run_toroidal_phase_analysis(RECORDS: pd.DataFrame, time_col: str, ignition_windows: List[Tuple[float, float]], rebound_windows: Optional[List[Tuple[float, float]]], ref_electrodes: Optional[List[str]], ...) -> Dict[str, object]`

## Private/Helper Functions

- `_get_fs(RECORDS: pd.DataFrame, time_col: str)`
- `_bandpass(x: np.ndarray, fs: float, f1: float, ...)`
- `_mean_reference(RECORDS: pd.DataFrame, time_col: str, electrodes: List[str])`
- `_circ_R(phi: np.ndarray)`
- `_circ_corr(phi1: np.ndarray, phi2: np.ndarray)`
- `_two_nn_id(pts: np.ndarray)`
- `_joint_phase_entropy(phi1: np.ndarray, phi2: np.ndarray, n_bins: int)`
- `_winding(phi: np.ndarray)`
- `_phase_scramble(x: np.ndarray, rng: np.random.Generator)`
- `_compute_torus_metrics(sig: np.ndarray, fs: float, band1: Tuple[float, float], ...)`
