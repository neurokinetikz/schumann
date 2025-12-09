# temporal_holography

**Total functions:** 8 (3 public, 5 private)

## Public Functions

### `find_channel_series(records: pd.DataFrame, ch_name: str) -> Optional[pd.Series]`

### `run_temporal_holography_multiplexed(RECORDS: pd.DataFrame, event_onsets: List[float], labels: Optional[List], time_col: str, electrodes: Optional[List[str]], ...) -> Dict[str, object]`

### `trial_segment(on)`

## Private/Helper Functions

- `_get_fs(RECORDS: pd.DataFrame, time_col: str)`
- `_bandpass(x: np.ndarray, fs: float, f1: float, ...)`
- `_get_series_matrix(RECORDS, electrodes, time_col)`
- `_phase_at_time(ref_sig: np.ndarray, tvec: np.ndarray, fs: float, ...)`
- `_pac_mi_single(x_phase: np.ndarray, x_amp: np.ndarray, nbins: int)`
