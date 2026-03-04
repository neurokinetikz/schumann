# Library API Reference

## Quick Module Finder

| Analysis Goal | Module | Key Function |
|--------------|--------|--------------|
| Load EEG data | utilities | `load_eeg_csv()` |
| Filter signals | utilities | `butter_bandpass()`, `butter_highpass()` |
| Compute PSD | utilities | `compute_psd_multitaper()` |
| Individual alpha freq | utilities | `compute_iaf()` |
| Detect SR spikes | harmonics | `detect_schumann_spikes_wavelet()` |
| Estimate harmonics | harmonics | `estimate_sr_harmonics()` |
| FOOOF peak detection | fooof_harmonics | `detect_harmonics_fooof()` |
| Detect ignitions | detect_ignition | `detect_ignitions_session()` |
| PAC comodulogram | cross_frequency | `pac_comodulogram_array()` |
| Bicoherence | cross_frequency | `bicoherence_array()` |
| Full cross-freq suite | cross_frequency | `run_crossfreq_suite_records()` |
| Network/graph metrics | network_geometry | `run_network_geometry_suite_records()` |
| UMAP embedding | network_geometry | `run_state_space_embedding_records()` |
| Granger/PDC/DTF | information_flow | `run_freq_granger_pdc_dtf()` |
| Transfer entropy | information_flow | `run_transfer_entropy()` |
| 1/f slope, DFA | criticality | `run_criticality_analysis()` |
| Avalanche stats | criticality | `avalanche_stats()` |
| HMM state detection | hidden_markov | `run_hmm_state_tests()` |
| Wavelet coherence | wavelet_coherence | `wavelet_coherence_tf()` |
| Phase locking | harmonic_locking | `analyze_locking()` |
| Microstate analysis | microstate_segmentation | `run_microstate_segmentation()` |
| TDA/topology | attractor_topology | `run_tda_attractor_topology()` |
| RQA/chaos | chaos_metrics | `run_rqa_chaos_metrics()` |
| Multiscale entropy | multiscale_entropy_and_fractal_scaling | `run_mse_dfa_multiscale()` |

---

## Core Constants

```python
# In utilities.py
FS = 128  # Sampling rate (Hz)

ELECTRODES = ['AF3','AF4','F7','F8','F3','F4','FC5','FC6','P7','P8','T7','T8','O1','O2']

BRAINWAVES = ['Delta','Theta','Alpha','BetaL','BetaH','Gamma']

RANGES = {
    'Delta': [1, 4],
    'Theta': [4, 8],
    'Alpha': [8, 12],
    'BetaL': [12, 16],
    'BetaH': [16, 25],
    'Gamma': [25, 45]
}
```

---

## Standard Return Structure

Most `run_*` functions return a dictionary with consistent keys:

```python
{
    'summary': pd.DataFrame,       # Key metrics table
    'delta_table': pd.DataFrame,   # Ignition-baseline differences
    'params': Dict,                # Parameters used
    'ignition': Dict,              # Ignition-specific results
    'baseline': Dict,              # Baseline results (if provided)
    'rebound': Dict,               # Rebound results (if provided)
    'plots': List[Figure],         # Generated figures (if show=True)
    'data': Dict[str, np.ndarray], # Raw arrays
}
```

---

## Module Reference by Domain

### 1. Core Utilities (utilities.py)

**Data Loading**
```python
RECORDS = load_eeg_csv(
    filename: str,
    electrodes: List[str] = ELECTRODES,
    device: str = "emotiv",  # or "muse"
    fs: int = 128,
    resample_to: int = None
) -> pd.DataFrame
```
Returns DataFrame with columns: `Timestamp`, `EEG.<electrode>`, `EEG.<electrode>.FILTERED`, etc.

**Filtering**
```python
filtered = butter_bandpass(data, lowcut, highcut, fs, order=4) -> np.ndarray
filtered = butter_highpass(data, cutoff, fs, order=4) -> np.ndarray
filtered = bandpass(data, f1, f2, fs, order=4) -> np.ndarray  # Safe Nyquist clamping
```

**Spectral Analysis**
```python
freqs, psd = compute_psd_multitaper(data, fs, fmin=1, fmax=45, bandwidth=2.0)
iaf = compute_iaf(psd, freqs, search_range=(7, 14)) -> float
```

**Visualization**
```python
plot_stacked_relpower_timeseries(RECORDS, electrodes, bands, time_col='Timestamp')
```

---

### 2. Schumann Detection (harmonics.py)

**Wavelet-Based Spike Detection**
```python
events = detect_schumann_spikes_wavelet(
    signal: np.ndarray,
    fs: float = 128,
    f_harmonics: Tuple = (7.83, 14.3, 20.8, 27.3, 33.8),
    n_cycles: int = 6,
    z_thresh: float = 2.5,
    min_duration_s: float = 0.5,
    merge_gap_s: float = 0.2
) -> List[Dict]
# Returns: [{'start_idx', 'end_idx', 'f_hz', 'peak_z', 'duration'}, ...]
```

**Harmonic Estimation**
```python
harmonics = estimate_sr_harmonics(
    RECORDS: pd.DataFrame,
    sr_channel: str = 'EEG.F4',
    f_can: Tuple = (7.83, 14.3, 20.8, 27.3, 33.8),
    search_halfband: float = 0.5,
    fs: float = 128
) -> Tuple[float, ...]
# Returns session-specific harmonic frequencies
```

**Activity Index**
```python
sai = schumann_activity_index(
    signal, fs=128,
    harmonics=(7.83, 14.3, 20.8, 27.3, 33.8),
    half_bw=0.5
) -> np.ndarray
# Returns time series of summed z-scored harmonic power
```

**Visualization Functions**
```python
fused = detect_and_plot_schumann_microgrid_with_global_tf(
    RECORDS, signal_col='EEG.F4', f0=7.83, n_harmonics=5, show=True
)

detect_and_plot_schumann_microgrid_with_heatmaps(
    RECORDS, signal_col='EEG.F4', ...
)
```

---

### 3. FOOOF/SpecParam (fooof_harmonics.py)

**FOOOF-Based Detection**
```python
result = detect_harmonics_fooof(
    psd: np.ndarray,
    freqs: np.ndarray,
    freq_range: Tuple = (1, 45),
    peak_width_limits: Tuple = (0.5, 4.0),
    max_n_peaks: int = 10,
    aperiodic_mode: str = 'fixed'
) -> FOOOFHarmonicResult
# Returns: FOOOFHarmonicResult(harmonics, aperiodic_exponent, offset)
```

**Multi-Channel Detection**
```python
results = detect_harmonics_fooof_multichannel(
    RECORDS, eeg_channels, fs=128, method='per_channel'
) -> Dict[str, FOOOFHarmonicResult]
```

**Peak Matching**
```python
matched = match_peaks_to_canonical(
    detected_peaks: List[Tuple],
    canonical: Tuple = (7.83, 14.3, 20.8, 27.3, 33.8),
    tolerance: float = 1.0
) -> List[Optional[Tuple]]
```

---

### 4. Ignition Detection (detect_ignition.py)

**Main Detection Function**
```python
output, IGNITION_WINDOWS = detect_ignitions_session(
    RECORDS: pd.DataFrame,
    sr_channel: str = 'EEG.F4',
    eeg_channels: List[str] = None,
    center_hz: float = 7.83,
    half_bw_hz: float = 0.4,
    z_thresh: float = 3.0,
    harmonics_hz: Tuple = None,
    harmonic_bw_hz: float = 0.3,
    gamma_band: Tuple = (30, 45),
    enable_directionality: bool = True,
    max_n_peaks: int = None,
    out_dir: str = None,
    show: bool = True
) -> Tuple[Dict, List[Tuple[float, float]]]
```

**Output Dict Keys**:
- `summary`: DataFrame with per-event metrics
- `times`: Array of event onset times
- `event_details`: List of dicts with all computed metrics
- `baseline_summary`: Baseline stats (if baseline_windows provided)

**Per-Event Metrics**:
- `sr_z_max`, `sr_z_mean_pm5`, `sr_z_mean_post5`: SR envelope z-scores
- `kuramoto_R`: Phase synchrony across channels
- `latencies`: Per-channel peak timing relative to t₀
- `granger_matrix`: Bivariate Granger causality
- `pgd_f0`: Phase gradient directionality at f₀
- `harmonic_stack_index`: Sum of harmonic envelope power
- `spectral_slope`: 1/f exponent
- `freq_specificity`: MSC bandwidth concentration

---

### 5. Cross-Frequency Coupling (cross_frequency.py)

**PAC Comodulogram**
```python
pac = pac_comodulogram_array(
    signal: np.ndarray,
    fs: float,
    phase_freqs: np.ndarray = np.arange(2, 20, 1),
    amp_freqs: np.ndarray = np.arange(20, 100, 2),
    method: str = 'tort_mi'  # or 'mvl', 'plv'
) -> np.ndarray  # Shape: (n_phase_freqs, n_amp_freqs)
```

**Bicoherence**
```python
bic = bicoherence_array(
    signal: np.ndarray,
    fs: float,
    nfft: int = 256,
    freq_range: Tuple = (1, 45)
) -> Tuple[np.ndarray, np.ndarray]  # (bicoherence, freqs)
```

**Full Suite**
```python
results = run_crossfreq_suite_records(
    RECORDS: pd.DataFrame,
    ignition_windows: List[Tuple] = None,
    baseline_windows: List[Tuple] = None,
    eeg_channels: List[str] = None,
    sr_channels: List[str] = None,
    skip_bicoherence: bool = False,
    show: bool = True
) -> Dict
```

---

### 6. Network Geometry (network_geometry.py)

**Full Suite**
```python
results = run_network_geometry_suite_records(
    RECORDS: pd.DataFrame,
    ignition_windows: List[Tuple] = None,
    baseline_windows: List[Tuple] = None,
    eeg_channels: List[str] = None,
    bands: Dict = BAND_DEFS,
    show: bool = True
) -> Dict
```

**State-Space Embedding**
```python
results = run_state_space_embedding_records(
    RECORDS: pd.DataFrame,
    eeg_channels: List[str] = None,
    method: str = 'umap',  # or 'tsne', 'pca'
    n_components: int = 2,
    window_sec: float = 2.0,
    step_sec: float = 0.5
) -> Dict
# Returns: {'embedding': np.ndarray, 'times': np.ndarray, ...}
```

**Multi-Band Analysis**
```python
results = run_multi_band_geometry_records(
    RECORDS, ignition_windows, bands=BAND_DEFS
) -> Dict
```

**Animation**
```python
animate_embedding_over_time_records(
    RECORDS, embedding, times, out_path='embedding.mp4'
)
```

---

### 7. Information Flow (information_flow.py)

**VAR/PDC/DTF**
```python
results = run_freq_granger_pdc_dtf(
    RECORDS: pd.DataFrame,
    eeg_channels: List[str],
    ignition_windows: List[Tuple] = None,
    max_order: int = 10,
    fs: float = 128,
    freq_range: Tuple = (1, 45)
) -> Dict
# Returns: {'pdc': np.ndarray, 'dtf': np.ndarray, 'granger': np.ndarray, 'freqs': np.ndarray}
```

**Transfer Entropy**
```python
results = run_transfer_entropy(
    RECORDS: pd.DataFrame,
    eeg_channels: List[str],
    ignition_windows: List[Tuple] = None,
    lag_range: Tuple = (1, 20),
    k_neighbors: int = 4
) -> Dict
# Returns: {'te_matrix': np.ndarray, 'lags': np.ndarray, ...}
```

**Time-Varying DTF**
```python
results = run_tvar_dtf(
    RECORDS, eeg_channels, window_sec=2.0, step_sec=0.5
) -> Dict
```

---

### 8. Criticality (criticality.py)

**Full Analysis**
```python
results = run_criticality_analysis(
    RECORDS: pd.DataFrame,
    ignition_windows: List[Tuple],
    baseline_windows: List[Tuple] = None,
    electrodes: List[str] = None,
    bands: Dict = {'theta': (4, 8), 'alpha': (8, 13)},
    dfa_scales_sec: np.ndarray = np.geomspace(0.25, 8.0, 14),
    avalanche_thresh: str = 'p95'
) -> Dict
```

**Individual Metrics**
```python
beta = welch_beta(signal, fs, freq_range=(1, 30)) -> float  # 1/f exponent
alpha = dfa_alpha(signal, scales) -> float  # DFA exponent
events = avalanche_events(binary_signal, fs) -> List[Dict]
stats = avalanche_stats(events) -> Dict  # tau, zeta exponents
```

---

### 9. Hidden Markov Models (hidden_markov.py)

**HMM State Detection**
```python
results = run_hmm_state_tests(
    RECORDS: pd.DataFrame,
    eeg_channels: List[str],
    sr_channel: str,
    ignition_windows: List[Tuple],
    K: int = 3,  # Number of states
    features: str = 'power'  # or 'wavelet'
) -> Dict
# Returns: {'states': np.ndarray, 'transition_matrix': np.ndarray, ...}
```

**Burst Detection**
```python
bursts = detect_schumann_bursts(
    signal, fs, center_hz=7.83, K=2
) -> List[Tuple[float, float]]
```

---

### 10. Wavelet Coherence (wavelet_coherence.py)

**Time-Frequency Coherence**
```python
wtc, freqs, times = wavelet_coherence_tf(
    signal1: np.ndarray,
    signal2: np.ndarray,
    fs: float,
    freq_range: Tuple = (1, 45),
    n_cycles: int = 6
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]
```

**MSC at Harmonics**
```python
table = msc_harmonics_table(
    RECORDS, sr_channel, eeg_channels,
    harmonics=(7.83, 14.3, 20.8, 27.3, 33.8),
    n_surrogates: int = 100
) -> pd.DataFrame
```

---

### 11. Phase Analysis (harmonic_locking.py, toroidal_phase.py)

**Harmonic Locking**
```python
results = analyze_locking(
    RECORDS: pd.DataFrame,
    sr_channel: str,
    eeg_channels: List[str],
    harmonics: Tuple = (7.83, 14.3, 20.8, 27.3)
) -> Dict
# Returns PLV, preferred phase, locking strength per harmonic
```

**Toroidal Phase**
```python
results = run_toroidal_phase_analysis(
    RECORDS, ignition_windows,
    band1=(4, 8), band2=(8, 13)
) -> Dict
# Returns joint phase entropy, winding number
```

---

### 12. Attractor/Topology (attractor_topology.py, chaos_metrics.py)

**TDA Analysis**
```python
results = run_tda_attractor_topology(
    RECORDS, ignition_windows,
    embedding_dim: int = 3,
    delay_samples: int = None
) -> Dict
# Returns Betti numbers, persistence diagrams
```

**RQA Chaos Metrics**
```python
results = run_rqa_chaos_metrics(
    RECORDS, ignition_windows,
    embedding_dim: int = 3,
    recurrence_threshold: float = 0.1
) -> Dict
# Returns recurrence rate, determinism, laminarity
```

---

### 13. Entropy (multiscale_entropy_and_fractal_scaling.py, entanglement_entropy.py)

**Multiscale Entropy**
```python
results = run_mse_dfa_multiscale(
    RECORDS, ignition_windows,
    scales_sec: np.ndarray = np.arange(0.1, 8.0, 0.1)
) -> Dict
```

**Integration Analogs**
```python
results = run_integration_analogs(
    RECORDS, ignition_windows
) -> Dict
# Returns Phi (integrated information), geometric II
```

---

### 14. Microstate Analysis (microstate_segmentation.py)

```python
results = run_microstate_segmentation(
    RECORDS: pd.DataFrame,
    ignition_windows: List[Tuple],
    n_states: int = 4,
    min_segment_ms: float = 30
) -> Dict
# Returns state labels, transition probabilities, mean durations
```

---

### 15. PSD Waterfall (psd_waterfall.py)

**Collector Class**
```python
collector = IgnitionPsdCollector()
collector.add_session(RECORDS, IGNITION_WINDOWS, label='session1')
collector.add_session(RECORDS2, IGNITION_WINDOWS2, label='session2')

# Visualization
collector.plot_grand_waterfall(view='sr_alignment')
collector.plot_heatmap()

# Export
df = collector.to_dataframe()
```

---

## Common Patterns

### Channel Name Handling
Functions accept both bare names and prefixed names:
```python
# All equivalent:
eeg_channels = ['F4', 'O1', 'P7']
eeg_channels = ['EEG.F4', 'EEG.O1', 'EEG.P7']
```

### Window Format
```python
# All windows are (start_sec, end_sec) tuples
IGNITION_WINDOWS = [(178.0, 189.0), (280.0, 291.0), (561.0, 572.0)]
```

### Sampling Rate Inference
```python
def infer_fs(RECORDS, time_col='Timestamp'):
    t = RECORDS[time_col].values
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    return 1.0 / np.median(dt)
```

### Safe Band Filtering
```python
def safe_bandpass(data, f1, f2, fs, order=4):
    ny = 0.5 * fs
    f1 = max(1e-6, min(f1, ny * 0.99))
    f2 = max(f1 + 1e-6, min(f2, ny * 0.999))
    b, a = butter(order, [f1/ny, f2/ny], btype='band')
    return filtfilt(b, a, data)
```

---

## Workflow Example

```python
import sys; sys.path.insert(0, './lib')
import utilities
import harmonics
import detect_ignition
import cross_frequency
import criticality
import network_geometry

# 1. Load data
RECORDS = utilities.load_eeg_csv("data/session.csv", device="emotiv", fs=128)

# 2. Estimate session harmonics
HARMONICS = harmonics.estimate_sr_harmonics(
    RECORDS, sr_channel='EEG.F4',
    f_can=(7.83, 14.3, 20.8, 27.3, 33.8)
)

# 3. Detect ignition events
out, IGNITION_WINDOWS = detect_ignition.detect_ignitions_session(
    RECORDS, sr_channel='EEG.F4',
    eeg_channels=['EEG.O1', 'EEG.O2', 'EEG.P7', 'EEG.P8'],
    center_hz=HARMONICS[0], half_bw_hz=0.4, z_thresh=3
)

# 4. Define baseline (e.g., 30s before each ignition)
baseline_windows = [(max(0, s-30), s) for s, e in IGNITION_WINDOWS]

# 5. Run analyses
xfreq = cross_frequency.run_crossfreq_suite_records(
    RECORDS, ignition_windows=IGNITION_WINDOWS,
    baseline_windows=baseline_windows
)

crit = criticality.run_criticality_analysis(
    RECORDS, ignition_windows=IGNITION_WINDOWS,
    baseline_windows=baseline_windows
)

net = network_geometry.run_network_geometry_suite_records(
    RECORDS, ignition_windows=IGNITION_WINDOWS,
    baseline_windows=baseline_windows
)

# 6. Review results
print("Ignition Summary:")
print(out['summary'])

print("\nCriticality Changes:")
print(crit['delta_table'])
```
