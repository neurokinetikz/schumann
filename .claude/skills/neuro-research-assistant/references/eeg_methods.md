# EEG Methods Reference

## Frequency Bands and Functional Significance

### Standard Band Definitions

| Band | Frequency | Primary Functions |
|------|-----------|-------------------|
| Delta | 1-4 Hz | Deep sleep (N3), homeostatic processes, attention modulation |
| Theta | 4-8 Hz | Working memory, spatial navigation, meditation, hippocampal rhythm |
| Alpha | 8-13 Hz | Relaxed wakefulness, inhibition, sensory gating, IAF varies by individual |
| Beta | 13-30 Hz | Active cognition, motor preparation, sustained attention |
| Gamma | 30-100 Hz | Perceptual binding, conscious awareness, often artifact-contaminated |

### Sub-Band Distinctions

**Low vs High Alpha (8-10 Hz vs 10-13 Hz)**
- Low alpha: general arousal, attention
- High alpha: semantic processing, memory

**Low vs High Beta (13-20 Hz vs 20-30 Hz)**
- Low beta (SMR, 12-15 Hz): sensorimotor integration
- High beta: anxiety, active problem-solving

**Low vs High Gamma (30-60 Hz vs 60-100 Hz)**
- Low gamma: feature binding, attention
- High gamma: often muscle artifact, requires careful verification

### Individual Alpha Frequency (IAF)

```python
def compute_iaf(psd, freqs, search_range=(7, 14)):
    """Find peak alpha frequency in individual."""
    mask = (freqs >= search_range[0]) & (freqs <= search_range[1])
    peak_idx = np.argmax(psd[mask])
    return freqs[mask][peak_idx]
```

- IAF varies ~8-12 Hz across individuals
- Adjust band boundaries relative to IAF for precision
- IAF correlates with cognitive speed, age

---

## Schumann Resonance

### Physical Basis
- Electromagnetic resonance of Earth-ionosphere cavity
- Excited by global lightning activity (~50 strikes/second)
- Global phenomenon, detectable worldwide with sensitive equipment

### Canonical Frequencies

| Mode | Frequency | EEG Band Overlap |
|------|-----------|------------------|
| f₀ | 7.83 Hz | Theta/Alpha boundary |
| f₁ | 14.3 Hz | Low Beta |
| f₂ | 20.8 Hz | Beta |
| f₃ | 27.3 Hz | High Beta |
| f₄ | 33.8 Hz | Low Gamma |
| f₅ | 40.3 Hz | Gamma |

**Sub-harmonics**: f₀/2 = 3.915 Hz, f₀/3 = 2.61 Hz, f₀/4 = 1.96 Hz, f₀/5 = 1.57 Hz

### Frequency Variation
- SR frequencies vary ±0.5 Hz with ionospheric conditions
- Diurnal variation, seasonal effects
- Use `estimate_sr_harmonics()` to find session-specific peaks

### Detection Methods

**Wavelet Ridge Tracking**
```python
import harmonics

events = harmonics.detect_schumann_spikes_wavelet(
    signal, fs=128,
    f_harmonics=(7.83, 14.3, 20.8, 27.3, 33.8),
    n_cycles=6,        # Morlet wavelet cycles
    z_thresh=2.5,      # Detection threshold
    min_duration_s=0.5 # Minimum event duration
)
```

**FOOOF/SpecParam Separation**
```python
import fooof_harmonics

result = fooof_harmonics.detect_harmonics_fooof(
    psd, freqs,
    freq_range=(1, 45),
    peak_width_limits=(0.5, 4.0),
    max_n_peaks=10
)
# Returns peaks separated from 1/f background
```

### Brain-Field Coherence Hypothesis

The research question: Do EEG signals show phase-locking to Schumann frequencies during specific conscious states?

**Analysis Approach**:
1. Detect SR harmonic peaks in EEG PSD
2. Compute magnitude-squared coherence (MSC) at harmonic frequencies
3. Compare coherence during "ignition" events vs baseline
4. Use surrogate testing (phase shuffle) for significance

---

## Preprocessing

### Filtering

**High-Pass Filtering** (remove DC drift)
```python
from scipy.signal import butter, filtfilt

def highpass(data, cutoff, fs, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, cutoff/nyq, btype='high')
    return filtfilt(b, a, data)

# Standard: 1 Hz high-pass for ERP, 0.1 Hz for oscillatory
```

**Band-Pass Filtering**
```python
def bandpass(data, low, high, fs, order=4):
    nyq = 0.5 * fs
    # Clamp to Nyquist
    low = max(1e-6, min(low, nyq * 0.99))
    high = max(low + 1e-6, min(high, nyq * 0.999))
    b, a = butter(order, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, data)
```

**Filter Edge Effects**
- Problem: Transients at signal boundaries
- Solution 1: Pad signal (reflect or constant)
- Solution 2: Use longer recording, trim edges
- Solution 3: Use FIR with explicit padding in MNE

**Phase Distortion**
- IIR filters (butter) cause phase distortion
- `filtfilt` applies forward-backward for zero phase
- For causal real-time: accept phase lag or use FIR

### Artifact Types

| Artifact | Frequency | Characteristics | Detection |
|----------|-----------|-----------------|-----------|
| Eye blinks | 0-4 Hz | High amplitude, frontal | Threshold + topography |
| Eye movements | 0-8 Hz | Horizontal gradient | EOG correlation |
| Muscle (EMG) | >20 Hz | Broadband, high frequency | High-freq power spike |
| Line noise | 50/60 Hz | Narrow peak | Notch filter |
| Movement | Broadband | Sudden transients | Amplitude threshold |
| Electrode pop | Broadband | Single channel spike | Per-channel variance |

### Artifact Handling Strategies

**Rejection** (conservative)
- Mark bad segments, exclude from analysis
- Preserves signal integrity
- Loses data

**Correction** (ICA/regression)
- Remove artifact components, keep signal
- Preserves more data
- Risk of removing real signal

```python
import mne

# ICA for eye artifacts
ica = mne.preprocessing.ICA(n_components=15, random_state=42)
ica.fit(raw)
ica.exclude = [0, 1]  # Components identified as eye artifacts
raw_clean = ica.apply(raw.copy())
```

### Re-Referencing

| Scheme | Use Case | Formula |
|--------|----------|---------|
| Average | Standard for high-density | x - mean(all) |
| Linked mastoids | Standard for low-density | x - 0.5*(M1+M2) |
| Laplacian | Spatial filtering | x - mean(neighbors) |
| REST | Theoretical zero reference | Inverse source model |

```python
# MNE re-referencing
raw.set_eeg_reference('average')  # Average reference
raw.set_eeg_reference(['M1', 'M2'])  # Linked mastoids
```

---

## Spectral Analysis Methods

### FFT (numpy.fft)

**Pros**: Fast, exact for periodic signals
**Cons**: Assumes stationarity, frequency resolution = 1/T

```python
from scipy.fft import fft, fftfreq

N = len(signal)
yf = fft(signal)
xf = fftfreq(N, 1/fs)[:N//2]
psd = 2.0/N * np.abs(yf[:N//2])**2
```

### Welch's Method (scipy.signal.welch)

**Pros**: Variance reduction via averaging segments
**Cons**: Tradeoff between frequency resolution and variance

```python
from scipy.signal import welch

freqs, psd = welch(signal, fs=128, nperseg=256, noverlap=128)
# nperseg=256 at 128 Hz → 2 sec segments → 0.5 Hz resolution
```

**Parameter Selection**:
- `nperseg`: Longer = better frequency resolution, worse temporal
- `noverlap`: Typically 50% (nperseg//2)
- `window`: 'hann' (default) good for most cases

### Multitaper Method

**Pros**: Optimal for short segments, controls spectral leakage
**Cons**: More computation, bandwidth tradeoff

```python
from mne.time_frequency import psd_array_multitaper

psd, freqs = psd_array_multitaper(signal, sfreq=128, fmin=1, fmax=45,
                                   bandwidth=2.0, adaptive=True)
```

**Bandwidth Selection**:
- bandwidth = 2 × half-bandwidth
- More tapers = smoother but lower frequency resolution
- For narrow peaks (SR harmonics): use low bandwidth (1-2 Hz)

### Wavelet Transform

**Pros**: Time-frequency resolution, good for transients
**Cons**: Scale-dependent resolution (coarse at low freq)

```python
from mne.time_frequency import tfr_array_morlet

power = tfr_array_morlet(signal[np.newaxis, np.newaxis, :],
                         sfreq=128, freqs=np.arange(1, 45, 0.5),
                         n_cycles=6, output='power')
```

**n_cycles Selection**:
- Higher n_cycles = better frequency resolution, worse temporal
- Rule: n_cycles ≥ 3 for lowest frequency
- Common: n_cycles = freqs / 2 (adaptive)

### Method Selection Guide

| Goal | Recommended Method |
|------|-------------------|
| Overall spectrum | Welch or Multitaper |
| Narrow peaks (SR) | Multitaper (low bandwidth) |
| Time-varying power | Wavelets |
| Short segments | Multitaper |
| Real-time | FFT with windowing |

---

## Connectivity Measures

### Coherence (Magnitude-Squared Coherence)

```python
from scipy.signal import coherence

f, coh = coherence(x, y, fs=128, nperseg=256)
```

**Interpretation**:
- Range: 0-1
- Measures linear relationship in amplitude AND phase
- Sensitive to volume conduction (can be spuriously high)

### Phase Locking Value (PLV)

```python
def plv(phase1, phase2):
    """Phase Locking Value between two phase time series."""
    return np.abs(np.mean(np.exp(1j * (phase1 - phase2))))
```

**Interpretation**:
- Range: 0-1
- Pure phase synchrony (ignores amplitude)
- Still sensitive to volume conduction

### Weighted Phase Lag Index (wPLI)

```python
def wpli(x, y, fs, band):
    """wPLI robust to volume conduction."""
    # Filter to band
    x_filt = bandpass(x, band[0], band[1], fs)
    y_filt = bandpass(y, band[0], band[1], fs)

    # Hilbert transform for phase
    phase_x = np.angle(hilbert(x_filt))
    phase_y = np.angle(hilbert(y_filt))

    # Cross-spectral imaginary part
    dphi = phase_x - phase_y
    imag_part = np.sin(dphi)

    # wPLI
    return np.abs(np.mean(np.abs(imag_part) * np.sign(imag_part))) / np.mean(np.abs(imag_part))
```

**Interpretation**:
- Range: 0-1
- Ignores zero-lag (volume conduction)
- Preferred for source-level claims

### Granger Causality

```python
from statsmodels.tsa.stattools import grangercausalitytests

# Returns F-test for X → Y causality
results = grangercausalitytests(np.column_stack([y, x]), maxlag=10)
```

**Interpretation**:
- Tests if past of X improves prediction of Y
- Requires stationarity
- Sensitive to model order selection

### Partial Directed Coherence (PDC) / DTF

```python
import information_flow

results = information_flow.run_freq_granger_pdc_dtf(
    RECORDS, eeg_channels=['EEG.F4', 'EEG.O1'],
    max_order=10, fs=128
)
```

**Interpretation**:
- Frequency-resolved directional connectivity
- PDC: direct connections only
- DTF: direct + indirect pathways

### Measure Selection Guide

| Question | Measure |
|----------|---------|
| General synchrony | Coherence |
| Phase-only coupling | PLV |
| Robust to volume conduction | wPLI, imaginary coherence |
| Directional influence | Granger, PDC, DTF |
| Nonlinear coupling | Transfer entropy |

---

## Event Detection

### Z-Score Thresholding

```python
def detect_events_zscore(signal, fs, z_thresh=3.0, min_dur_s=0.1):
    """Detect events exceeding z-score threshold."""
    z = (signal - np.mean(signal)) / np.std(signal)
    above_thresh = np.abs(z) > z_thresh

    # Find contiguous regions
    events = []
    in_event = False
    start = 0

    for i, above in enumerate(above_thresh):
        if above and not in_event:
            in_event = True
            start = i
        elif not above and in_event:
            in_event = False
            duration = (i - start) / fs
            if duration >= min_dur_s:
                events.append((start/fs, i/fs, np.max(z[start:i])))

    return events
```

### Coincidence Detection

For multi-metric events (e.g., ignition):
1. Detect threshold crossings for each metric
2. Find temporal overlap
3. Require N of M metrics to co-occur

```python
def coincidence_events(event_lists, min_overlap_s=0.5):
    """Find events that overlap across multiple metrics."""
    # Merge overlapping events that span multiple metrics
    # ... implementation depends on specific requirements
```

### Baseline Selection

**Strategies**:
1. **Pre-event baseline**: Fixed window before each event
2. **Random windows**: Matched duration from non-event periods
3. **Surrogate baseline**: Phase-shuffled signal preserves spectrum

```python
# Pre-event baseline
baseline_windows = [(start - baseline_dur, start) for start, end in event_windows]

# Random windows (avoiding events)
def random_baseline_windows(total_duration, event_windows, n_windows, window_dur):
    # Generate random starts, check no overlap with events
    ...
```

---

## Time-Frequency Analysis

### Short-Time Fourier Transform (STFT)

```python
from scipy.signal import stft

f, t, Zxx = stft(signal, fs=128, nperseg=64, noverlap=48)
power = np.abs(Zxx)**2
```

### Wavelet Time-Frequency

```python
from mne.time_frequency import tfr_array_morlet

# Shape: (n_epochs, n_channels, n_times)
data = signal[np.newaxis, np.newaxis, :]
freqs = np.arange(1, 45, 0.5)

power = tfr_array_morlet(data, sfreq=128, freqs=freqs,
                         n_cycles=freqs/2, output='power')
```

### Baseline Normalization

| Method | Formula | Use Case |
|--------|---------|----------|
| Absolute | power | Raw comparison |
| Relative | power / baseline | Percent change |
| dB | 10 * log10(power / baseline) | Log scale |
| Z-score | (power - mean) / std | Standardized |
| Percent | 100 * (power - baseline) / baseline | Intuitive |

```python
def baseline_normalize(tfr, baseline_times, times, method='zscore'):
    bl_mask = (times >= baseline_times[0]) & (times <= baseline_times[1])
    bl_mean = tfr[..., bl_mask].mean(axis=-1, keepdims=True)
    bl_std = tfr[..., bl_mask].std(axis=-1, keepdims=True)

    if method == 'zscore':
        return (tfr - bl_mean) / bl_std
    elif method == 'db':
        return 10 * np.log10(tfr / bl_mean)
    elif method == 'percent':
        return 100 * (tfr - bl_mean) / bl_mean
```

---

## Source Localization Concepts

### Forward Problem
- Given source configuration, compute scalp potentials
- Requires head model (geometry, conductivity)
- Deterministic

### Inverse Problem
- Given scalp potentials, estimate sources
- Ill-posed (infinite solutions)
- Requires constraints/priors

### Common Methods

| Method | Type | Assumptions |
|--------|------|-------------|
| Dipole fitting | Parametric | Few focal sources |
| Beamforming (LCMV) | Spatial filter | Point sources, known location |
| MNE/dSPM | Distributed | Smooth current distribution |
| sLORETA | Distributed | Zero localization error |
| eLORETA | Distributed | Exact, more computation |

### For Low-Density EEG (14 channels)

Source localization is limited with few electrodes:
- Use spatial filtering (Laplacian) for local enhancement
- Group electrodes into ROIs (frontal, parietal, occipital)
- Focus on sensor-level analysis with cautious interpretation
