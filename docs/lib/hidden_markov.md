# hidden_markov

**Total functions:** 18 (18 public, 0 private)

## Public Functions

### `infer_fs(RECORDS: pd.DataFrame, time_col: str) -> float`

### `get_series(RECORDS: pd.DataFrame, name: str) -> np.ndarray`

### `bandpass(x: np.ndarray, fs: float, f1: float, f2: float, order: int) -> np.ndarray`

### `slice_epoch(x: np.ndarray, i0: int, i1: int) -> Optional[np.ndarray]`

### `detect_schumann_bursts(RECORDS: pd.DataFrame, sr_channel: str, time_col: str, center_hz: float, half_bw_hz: float, ...) -> Dict[str, object]`

### `morlet_cwt(sig: np.ndarray, fs: float, freqs: np.ndarray, w0: float) -> np.ndarray`
> Complex Morlet CWT via FFT-convolution.

### `erp_ersp_itc(RECORDS: pd.DataFrame, eeg_channels: List[str], sr_channel: str, time_col: str, win_sec: Tuple[float, float], ...) -> Dict[str, object]`
> Build ERP/ERSP/ITC time-locked to Schumann bursts and run simple cluster-perm tests.

### `max_cluster_mass(mask, value_map)`

### `circ_shift_2d(A, sh0, sh1)`

### `bandpower_features(RECORDS: pd.DataFrame, eeg_channels: List[str], time_col: str, bands: Dict[str, Tuple[float, float]], win_sec: float, ...) -> Dict[str, object]`
> Sliding-window mean band power per band, averaged over EEG channels.

### `schumann_amplitude(RECORDS: pd.DataFrame, sr_channel: str, time_col: str, center_hz: float, half_bw_hz: float, ...) -> Dict[str, object]`
> Sliding-window Schumann envelope mean in the same grid as bandpower_features.

### `hmm_states_gmm(features: np.ndarray, K: int, random_state: int) -> Dict[str, object]`
> Fit GaussianMixture as a simple HMM surrogate; decode state(t).

### `eta_state_occupancy(states: np.ndarray, T: np.ndarray, event_times: np.ndarray, span_sec: float, n_states: int) -> Dict[str, object]`
> Event-triggered state occupancy around event_times (ETA).

### `logistic_state_transition_vs_amp(states: np.ndarray, amp: np.ndarray) -> Dict[str, float]`
> Binary transition Y: 1 if state changes at t+1; regress on Schumann amp(t).

### `run_hmm_state_tests(RECORDS: pd.DataFrame, eeg_channels: List[str], sr_channel: str, time_col: str, K: int, ...) -> Dict[str, object]`
> 1) Build band-power features; fit GMM(K) → state(t).

### `erp_ersp_itc_safe(RECORDS, eeg_channels, sr_channel, time_col, win_sec, ...)`
> Schumann-locked ERP/ERSP/ITC with edge padding and TF cluster-perm.

### `take_segment(x, i_on)`

### `max_cluster_mass(mask, val)`
