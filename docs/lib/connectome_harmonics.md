# connectome_harmonics

**Total functions:** 19 (11 public, 8 private)

## Public Functions

### `find_channel_series(records: pd.DataFrame, ch_name: str) -> Optional[pd.Series]`

### `project_to_harmonics(X: np.ndarray, H: np.ndarray, orthonormal: bool, ridge: float) -> np.ndarray`
> Project sensor data X (n_elec × n_times) onto spatial harmonics H (n_elec × n_modes).

### `mode_power(A: np.ndarray) -> np.ndarray`
> Return per-mode average power across time: P_k = mean_t A_k(t)^2.

### `spectral_entropy(P: np.ndarray) -> float`

### `participation_ratio(P: np.ndarray) -> float`

### `top_decile_mass(P: np.ndarray) -> float`

### `run_connectome_harmonics_breadth(RECORDS: pd.DataFrame, H: np.ndarray, electrodes: List[str], ignition_windows: List[Tuple[float, float]], rebound_windows: Optional[List[Tuple[float, float]]], ...) -> Dict[str, object]`

### `analyze_state(blocks: List[np.ndarray]) -> Dict[str, object]`

### `plot_harmonics_power_spectra(spectra: Dict[str, np.ndarray], top_k: int) -> None`

### `plot_harmonics_breadth_deltas(delta_df: pd.DataFrame) -> None`

### `build_functional_harmonics_from_baseline(RECORDS: pd.DataFrame, electrodes: List[str], ignition_windows: Optional[List[Tuple[float, float]]], time_col: str, fband: Tuple[float, float], ...) -> np.ndarray`
> Build sensor-space functional harmonics from baseline data.

## Private/Helper Functions

- `_get_fs(RECORDS: pd.DataFrame, time_col: str)`
- `_slice_blocks(RECORDS: pd.DataFrame, time_col: str, X: np.ndarray, ...)`
- `_fourier_phase_randomize(x: np.ndarray, rng: np.random.Generator)`
- `_make_surrogate_block(data: np.ndarray, rng: np.random.Generator)`
- `_get_fs(RECORDS: pd.DataFrame, time_col: str)`
- `_slice_baseline_mask(RECORDS: pd.DataFrame, time_col: str, fs: float, ...)`
- `_bandpass(X: np.ndarray, fs: float, f1: float, ...)`
- `_pseudo_wpli(Xb: np.ndarray)`
