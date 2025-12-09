# psd_waterfall

**Total functions:** 24 (9 public, 15 private)

## Public Functions

### `compute_psd_by_window_df(df: pd.DataFrame, windows: Sequence[Tuple[float, float]], fs: float, channels: Optional[Sequence[str]], band: Optional[Tuple[float, float]], ...) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]`

### `add_session(self, RECORDS: pd.DataFrame, windows: Sequence[Tuple[float, float]], fs: float, session_id: str, ...) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]`

### `add_precomputed(self, freqs: np.ndarray, Z: np.ndarray, session_id: str, windows: Optional[Sequence[Tuple[float, float]]], ...) -> None`

### `freqs(self) -> np.ndarray`

### `to_dataframe(self) -> pd.DataFrame`
> Return a tidy DataFrame with columns: row_id, freq, value, plus metadata.

### `plot_heatmap(self, title: Optional[str], cmap: str, annotate: bool, vmin: Optional[float], ...) -> Tuple[plt.Figure, Dict[str, Any]]`

### `plot_grand_waterfall(self, title: Optional[str], cmap: str, view_preset: str, heatmap_panel: bool, ...) -> Tuple[plt.Figure, np.ndarray, np.ndarray, Dict[str, Any]]`

### `plot_waterfall_sr(freqs: np.ndarray, Z: np.ndarray, meta: Dict[str, Any], title: Optional[str], cmap: str, ...) -> Tuple[plt.Figure, Dict]`

### `plot_ignition_psd_waterfall(csv_or_df: Union[str, pd.DataFrame], windows: Sequence[Tuple[float, float]], fs: float, channels: Optional[Sequence[str]], band: Optional[Tuple[float, float]], ...) -> Tuple[plt.Figure, np.ndarray, np.ndarray, Dict]`
> Backward-compatible function that accepts a DataFrame **or** CSV path.

## Private/Helper Functions

- `_guess_time_col(df: pd.DataFrame)`
- `_auto_channels(df: pd.DataFrame)`
- `_sec_to_idx(df: pd.DataFrame, s: float, e: float, ...)`
- `_bandpass_notch(X: np.ndarray, fs: float, band: Optional[Tuple[float, float]], ...)`
- `_welch_psd(X: np.ndarray, fs: float, nperseg_sec, ...)`
- `_aggregate_psd(P: np.ndarray, mode)`
- `__init__(self, freq_range: Tuple[float, float], sort_by: Optional[Union[str, Tuple[str, float]]], ...)`
- `_row_df(self)`
- `_sort_indices(self)`
- `_add_sr_lines_2d(self, ax)`
- `_sr_markers_2d(self, ax, Z_sorted)`
- `_resolve_view(view_preset: Optional[str], elev: Optional[float], azim: Optional[float])`
- `_add_sr_curtains(ax, freqs: np.ndarray, N: int, ...)`
- `_add_sr_markers(ax, freqs: np.ndarray, Z: np.ndarray, ...)`
- `_plot_waterfall_any(freqs: np.ndarray, Z: np.ndarray, title: Optional[str], ...)`
