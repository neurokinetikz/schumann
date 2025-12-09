# resonant_modes

**Total functions:** 15 (14 public, 1 private)

## Public Functions

### `infer_fs(RECORDS: pd.DataFrame, time_col: str) -> float`

### `get_series(RECORDS: pd.DataFrame, name: str) -> np.ndarray`

### `slice_concat(x: np.ndarray, fs: float, windows: Optional[List[Tuple[float, float]]]) -> np.ndarray`

### `bandpass(x: np.ndarray, fs: float, f1: float, f2: float, order) -> np.ndarray`

### `zscore(x)`

### `plv_adj(RECORDS, channels, band, windows, time_col) -> np.ndarray`

### `laplacian_eigendecomp(W: np.ndarray, n_modes: int) -> Tuple[np.ndarray, np.ndarray]`
> Return first n_modes Laplacian eigenvalues & eigenvectors (columns).

### `project_to_harmonics(X: np.ndarray, H: np.ndarray) -> np.ndarray`
> X: (n_ch, T), H: (n_ch, K) columns orthonormal → A: (K, T).

### `mode_band_power(A: np.ndarray, fs: float, fband: Tuple[float, float]) -> np.ndarray`
> A: (K, T) → band power per mode via band-pass + RMS.

### `mode_welch_power(A: np.ndarray, fs: float, nperseg: Optional[int]) -> Tuple[np.ndarray, np.ndarray]`
> Return (f, Pk(f)) where Pk is (K, n_f).

### `msc_mode_to_sr(A: np.ndarray, sr: np.ndarray, fs: float, harmonics: List[float], nperseg: Optional[int]) -> pd.DataFrame`

### `schumann_envelope(sr: np.ndarray, fs: float, center_hz: float, half_bw: float) -> np.ndarray`

### `run_connectome_harmonics_resonance(RECORDS: pd.DataFrame, eeg_channels: List[str], ignition_windows: Optional[List[Tuple[float, float]]], baseline_windows: Optional[List[Tuple[float, float]]], sr_channel: Optional[str], ...) -> Dict[str, object]`
> Build harmonic basis (connectome W_conn if provided; else functional PLV) and test resonance:

### `schumann_envelope(sig: np.ndarray, fs: float, center: float, half: float) -> np.ndarray`
> Compatibility wrapper: accepts center/half or center_hz/half_bw.

## Private/Helper Functions

- `_ensure_dir(d)`
