# criticality

**Total functions:** 13 (12 public, 1 private)

## Public Functions

### `bandpass(x: np.ndarray, fs: float, f1: float, f2: float, order: int) -> np.ndarray`

### `slice_blocks(RECORDS: pd.DataFrame, time_col: str, X: np.ndarray, fs: float, winlist: List[Tuple[float, float]]) -> List[np.ndarray]`

### `find_channel_series(records: pd.DataFrame, ch_name: str) -> Optional[pd.Series]`

### `welch_beta(x: np.ndarray, fs: float, fmin: float, fmax: float) -> float`
> Robust 1/f slope β via log–log linear regression on Welch PSD between fmin–fmax.

### `dfa_alpha(x: np.ndarray, fs: float, scales_sec: np.ndarray) -> float`
> Detrended fluctuation analysis (DFA) exponent α on band-limited amplitude envelope.

### `avalanche_events(env: np.ndarray, thresh: float) -> List[Tuple[int, int, float]]`
> Return [(start_idx, end_idx, size)] where size is area under envelope above threshold.

### `avalanche_stats(envs: List[np.ndarray], fs: float, thresh_mode: str) -> Dict[str, object]`

### `run_criticality_analysis(RECORDS: pd.DataFrame, ignition_windows: List[Tuple[float, float]], rebound_windows: Optional[List[Tuple[float, float]]], control_windows: Optional[List[Tuple[float, float]]], electrodes: Optional[List[str]], ...) -> Dict[str, object]`

### `analyze_state(blocks: List[np.ndarray]) -> Dict[str, object]`

### `plot_criticality_deltas(delta_df: pd.DataFrame) -> None`

### `plot_avalanche_ccdf(aval_dict: Dict[str, Dict[str, np.ndarray]]) -> None`
> Plot complementary CDFs of avalanche sizes/durations per state (log–log).

### `fit_powerlaw_tail(x, xmin)`

## Private/Helper Functions

- `_get_fs(RECORDS: pd.DataFrame, time_col: str)`
