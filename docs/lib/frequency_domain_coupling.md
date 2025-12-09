# frequency_domain_coupling

**Total functions:** 25 (23 public, 2 private)

## Public Functions

### `infer_fs(RECORDS: pd.DataFrame, time_col: str) -> float`

### `get_series(RECORDS: pd.DataFrame, name: str) -> np.ndarray`
> Return signal column. Accepts 'EEG.O1' or bare 'O1' (will try 'EEG.O1').

### `slice_concat(x: np.ndarray, fs: float, windows: Optional[List[Tuple[float, float]]]) -> np.ndarray`
> Concatenate [t0,t1] windows (sec) from x; if windows is None/empty, return x.

### `run_multitaper_msc_harmonics(RECORDS: pd.DataFrame, x_channels: List[str], y_channel: str, windows: Optional[List[Tuple[float, float]]], time_col: str, ...) -> Dict[str, object]`
> MSC between mean of x_channels and y_channel over 'windows'.

### `run_wavelet_coherence(RECORDS: pd.DataFrame, x_channel: str, y_channel: str, time_col: str, fmin: float, ...) -> Dict[str, object]`
> Morlet wavelet coherence with cluster-based permutation correction.

### `cwt(sig: np.ndarray) -> np.ndarray`

### `smooth(A: np.ndarray, wlen: int) -> np.ndarray`

### `plot_msc_harmonics_table(msc_table: pd.DataFrame, ax: Optional[plt.Axes], label: str, color: str, title: Optional[str]) -> plt.Axes`
> Plot MSC with jackknife CIs at harmonic lines from a single run_multitaper_msc_harmonics result.

### `plot_msc_harmonics_compare(msc_ign: Dict[str, object], msc_base: Dict[str, object], title: str) -> None`
> Side-by-side bars with jackknife CIs comparing ignition vs baseline.

### `infer_fs(RECORDS: pd.DataFrame, time_col: str) -> float`

### `get_series(RECORDS: pd.DataFrame, name: str) -> np.ndarray`
> Return a numeric signal array. Accepts 'EEG.O1' or bare 'O1'.

### `slice_concat(x: np.ndarray, fs: float, windows: Optional[List[Tuple[float, float]]]) -> np.ndarray`
> Concatenate [t0,t1] windows in seconds; if None/empty, return full signal.

### `bandpass(x: np.ndarray, fs: float, f1: float, f2: float, order: int) -> np.ndarray`

### `plv_and_mean_phase(eeg: np.ndarray, sr: np.ndarray, fs: float, center_hz: float, half_bw_hz: float) -> Tuple[float, float]`
> Compute PLV and circular mean phase difference between EEG and SR at a narrow band centered at center_hz.

### `run_plv_harmonics_topography(RECORDS: pd.DataFrame, eeg_channels: List[str], sr_channel: str, harmonics: List[float], half_bw_hz: float, ...) -> Dict[str, object]`
> Compute PLV and mean phase for each EEG channel vs Schumann reference at Schumann harmonics.

### `plot_plv_topography(plv_table: pd.DataFrame, chan_pos: Dict[str, Tuple[float, float]], freq: float, vmin: float, vmax: float, ...) -> None`
> Simple 2D scatter topography for a single harmonic:

### `xcorr_envelopes_peaklag(eeg: np.ndarray, sr: np.ndarray, fs: float, center_hz: float, half_bw_hz: float, ...) -> Tuple[float, np.ndarray, np.ndarray]`
> Band-limit both to [center±half_bw], take Hilbert envelopes and compute normalized cross-correlogram.

### `bootstrap_peaklag_ci(eeg: np.ndarray, sr: np.ndarray, fs: float, center_hz: float, half_bw_hz: float, ...) -> Dict[str, object]`
> Circular-shift bootstrap null for peak lag. Returns {'peak_ms':..., 'ci':(lo,hi), 'lags_ms':..., 'xcorr':...}.

### `scf_cyclic_periodogram(x: np.ndarray, fs: float, alpha_hz: float, nperseg: int, noverlap: int) -> Tuple[np.ndarray, np.ndarray]`
> Simple cyclic periodogram SCF estimator at cyclic frequency alpha (Hz):

### `scf_at_harmonics(RECORDS: pd.DataFrame, channel: str, harmonics: List[float], windows: Optional[List[Tuple[float, float]]], time_col: str, ...) -> Dict[str, object]`
> Compute SCF magnitude integrated over frequency for each Schumann cyclic frequency alpha.

### `scf_cyclic_periodogram_demod(x: np.ndarray, fs: float, alpha_hz: float, nperseg: int, noverlap: int, ...) -> Tuple[np.ndarray, np.ndarray]`
> Cyclostationary spectral correlation via complex demodulation:

### `scf_at_harmonics(RECORDS: pd.DataFrame, channel: str, harmonics: List[float], windows: Optional[List[Tuple[float, float]]], time_col: str, ...) -> Dict[str, object]`
> Compute SCF magnitude integrated over f for each Schumann cyclic α using demodulation SCF.

### `scf_cyclic_periodogram_demod(x: np.ndarray, fs: float, alpha_hz: float, nperseg: int, noverlap: int, ...) -> Tuple[np.ndarray, np.ndarray]`
> Cyclostationary spectral correlation via complex demodulation:

## Private/Helper Functions

- `_mtm_cross_spectra(x: np.ndarray, y: np.ndarray, fs: float, ...)`
- `_mtm_cross_spectra(x: np.ndarray, y: np.ndarray, fs: float, ...)`
