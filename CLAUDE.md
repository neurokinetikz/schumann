# CLAUDE.md - Schumann Resonance EEG Analysis

## Project Overview

This repository implements advanced EEG signal processing and analysis focused on **Schumann Resonance** detection and brain-field coherence. The project analyzes EEG data to detect "ignition" events (transient neural activations) and their relationship to Earth's Schumann Resonance frequencies (~7.83 Hz fundamental + harmonics).

### Core Hypothesis
The project investigates whether EEG signals show coherence with Schumann Resonance frequencies during specific conscious states ("ignition" events), suggesting potential brain-field coupling.

## Quick Start

```python
# Standard imports in notebooks
import sys; sys.path.insert(0, './lib')
import utilities
import harmonics
import detect_ignition
# ... other lib modules

# Key constants
FS = 128  # Sampling rate (Hz)
ELECTRODES = ['AF3','AF4','F7','F8','F3','F4','FC5','FC6','P7','P8','T7','T8','O1','O2']
RANGES = {'Delta':[1,4],'Theta':[4,8],'Alpha':[8,12],'BetaL':[12,16],'BetaH':[16,25],'Gamma':[25,45]}

# Load EEG data
RECORDS = utilities.load_eeg_csv("data/file.csv", electrodes=ELECTRODES)

# Estimate Schumann harmonics from data
HARMONICS = harmonics.estimate_sr_harmonics(RECORDS, sr_channel='EEG.F4', fs=128,
    f_can=(7.83, 14.3, 20.8, 27.3, 33.8), search_halfband=0.5)

# Detect ignition windows
out, IGNITION_WINDOWS = detect_ignition.detect_ignitions_session(RECORDS, ...)
```

## Directory Structure

```
schumann/
├── lib/                      # Core analysis modules
│   ├── utilities.py          # Data loading, filtering, basic analysis
│   ├── harmonics.py          # Schumann spike detection (Morlet wavelet)
│   ├── detect_ignition.py    # Ignition event detection
│   ├── cross_frequency.py    # PAC, bicoherence, waveform shape
│   ├── criticality.py        # 1/f slope, DFA, avalanches
│   ├── network_geometry.py   # State-space embeddings (UMAP)
│   ├── information_flow.py   # VAR, PDC, DTF, transfer entropy
│   ├── harmonic_coherence.py # Schumann-EEG coherence signatures
│   └── ... (40+ specialized modules)
├── data/                     # EEG CSV files (gitignored)
├── exports*/                 # Analysis outputs (gitignored)
├── Basics.ipynb              # Main analysis notebook
├── Six Panel.ipynb           # Dashboard visualizations
└── Ignition.ipynb            # Ignition-focused analysis
```

## Data Format

### Input: EEG CSV Files
- **Emotiv format**: Columns like `EEG.AF3`, `EEG.F4`, etc.
- **Muse format**: Columns `eeg_1`-`eeg_4` mapped to `EEG.AF7`, `EEG.AF8`, `EEG.TP9`, `EEG.TP10`
- **Timestamp**: Either explicit `Timestamp` column or inferred at 128 Hz

### RECORDS DataFrame Columns (after loading)
```
Timestamp                     # Time in seconds
EEG.<electrode>               # Raw EEG values
EEG.<electrode>.FILTERED      # 1 Hz high-pass filtered
EEG.<electrode>.FILTERED.POW  # Squared power
EEG.<electrode>.<band>        # Band-passed signal (e.g., EEG.F4.Alpha)
POW.EEG.<electrode>.<band>    # Band power
POW.EEG.<electrode>.<band>.REL  # Relative band power
```

## Key Schumann Resonance Frequencies

| Harmonic | Canonical (Hz) | Description |
|----------|---------------|-------------|
| f₀ | 7.83 | Fundamental |
| f₁ | 14.3 | 2nd harmonic |
| f₂ | 20.8 | 3rd harmonic |
| f₃ | 27.3 | 4th harmonic |
| f₄ | 33.8 | 5th harmonic |
| f₅ | 40.3 | 6th harmonic |

Subharmonics: 3.915 (f₀/2), 2.61 (f₀/3), 1.9575 (f₀/4), 1.566 (f₀/5)

## Core Analysis Pipeline

### 1. Load & Preprocess
```python
RECORDS = utilities.load_eeg_csv(filename, electrodes=ELECTRODES, device="emotiv", fs=128)
```

### 2. Estimate Session-Specific Harmonics
```python
HARMONICS = harmonics.estimate_sr_harmonics(RECORDS, sr_channel='EEG.F4',
    f_can=(7.83, 14.3, 20.8, 27.3, 33.8))
```

### 3. Detect Ignition Windows
```python
out, IGNITION_WINDOWS = detect_ignition.detect_ignitions_session(
    RECORDS, sr_channel='EEG.F4', eeg_channels=['EEG.O1','EEG.O2',...],
    center_hz=HARMONICS[0], half_bw_hz=0.4, z_thresh=3)
```

### 4. Run Specialized Analyses
```python
# Schumann wavelet detection with heatmaps
fused = harmonics.detect_and_plot_schumann_microgrid_with_global_tf(
    RECORDS, signal_col='EEG.F4', f0=7.83, n_harmonics=5, show=True)

# Cross-frequency coupling
res = cross_frequency.run_crossfreq_suite_records(RECORDS, ignition_windows=IGNITION_WINDOWS)

# Criticality analysis
crit = criticality.run_criticality_analysis(RECORDS, ignition_windows=IGNITION_WINDOWS)
```

## Module Reference

### utilities.py (Core)
- `load_eeg_csv()` - Load and preprocess EEG data
- `butter_bandpass()`, `butter_highpass()` - Signal filtering
- `compute_psd_multitaper()` - Power spectral density
- `compute_iaf()` - Individual Alpha Frequency
- `plot_stacked_relpower_timeseries()` - Band power visualization

### harmonics.py (Schumann Detection)
- `detect_schumann_spikes_wavelet()` - Morlet wavelet spike detection
- `detect_and_plot_schumann_microgrid_with_heatmaps()` - Ridge tracking
- `detect_and_plot_schumann_microgrid_with_global_tf()` - Global TF panel
- `estimate_sr_harmonics()` - Estimate harmonics from PSD peaks
- `schumann_activity_index()` - Sum of z-scored harmonic activity

### detect_ignition.py
- `detect_ignitions_session()` - Full session ignition detection
- Returns `(output_dict, IGNITION_WINDOWS)` list of (start_sec, end_sec) tuples

### cross_frequency.py
- `pac_comodulogram_array()` - Phase-amplitude coupling
- `bicoherence_array()` - Bicoherence calculation
- `run_crossfreq_suite_records()` - Full PAC/bicoherence pipeline

### criticality.py
- `run_criticality_analysis()` - 1/f slope, DFA, avalanche statistics
- `welch_beta()` - 1/f exponent estimation
- `dfa_alpha()` - Detrended fluctuation analysis

### network_geometry.py
- `run_full_session_with_bands_and_exports()` - UMAP embeddings
- `run_multi_band_geometry_records()` - Multi-band graph analysis

### information_flow.py
- `run_freq_granger_pdc_dtf()` - VAR/PDC/DTF analysis
- `run_transfer_entropy()` - Transfer entropy
- `run_tvar_dtf()` - Time-varying DTF

## Common Patterns & Conventions

### Function Signatures
Most analysis functions follow this pattern:
```python
def run_<analysis>_<type>(
    RECORDS: pd.DataFrame,              # Main data
    ignition_windows: List[Tuple],      # [(start_sec, end_sec), ...]
    baseline_windows: List[Tuple]=None, # Optional baseline
    electrodes: List[str]=None,         # None = auto-detect EEG.* columns
    time_col: str='Timestamp',
    out_dir: str=None,                  # Export directory
    show: bool=True,                    # Show plots
    **kwargs
) -> Dict[str, object]:
```

### Channel Naming
- Raw: `'F4'`, `'O1'`, `'AF3'`
- With prefix: `'EEG.F4'`, `'EEG.O1'`
- Most functions accept either form

### Windows Format
```python
IGNITION_WINDOWS = [(178, 189), (280, 291), (561, 572)]  # (start_sec, end_sec)
```

### Return Values
Most functions return dicts with:
- `'summary'` or `'delta_table'` - Key metrics as DataFrame
- `'params'` - Parameters used
- State-specific results: `'ignition'`, `'baseline'`, `'rebound'`

## Important Constants

```python
# In utilities.py
FS = 128  # Sampling rate
ELECTRODES = ['AF3','AF4','F7','F8','F3','F4','FC5','FC6','P7','P8','T7','T8','O1','O2']
BRAINWAVES = ['Delta','Theta','Alpha','BetaL','BetaH','Gamma']
RANGES = {'Delta':[1,4],'Theta':[4,8],'Alpha':[8,12],'BetaL':[12,16],'BetaH':[16,25],'Gamma':[25,45]}
```

## Development Guidelines

### Adding New Analysis Modules
1. Place in `lib/` directory
2. Include `infer_fs_from_records()` or similar for fs inference
3. Support both bare channel names and `EEG.` prefixed
4. Return structured dict with `summary` DataFrame
5. Include `show=True` parameter for optional plotting

### Common Helper Patterns
```python
# Infer sampling rate
def _get_fs(RECORDS, time_col='Timestamp'):
    t = RECORDS[time_col].values
    dt = np.diff(t); dt = dt[np.isfinite(dt) & (dt > 0)]
    return 1.0 / np.median(dt)

# Find channel flexibly
def find_channel_series(records, ch_name):
    patterns = ("EEG.{ch}", "eeg.{ch}", "{ch}")
    for pat in patterns:
        col = pat.format(ch=ch_name)
        if col in records.columns:
            return records[col]
    return None
```

### Band Filtering Safety
Always clamp frequencies to Nyquist:
```python
ny = 0.5 * fs
f1 = max(1e-6, min(f1, ny * 0.99))
f2 = max(f1 + 1e-6, min(f2, ny * 0.999))
```

## Testing & Validation

- Run analyses on known meditation EEG sessions
- Compare ignition windows across different detection parameters
- Validate Schumann harmonics against known values (7.83, 14.3, 20.8...)
- Use surrogate controls (phase shuffling) for statistical validation

## Dependencies

```
numpy
pandas
scipy
matplotlib
seaborn
mne (for time-frequency analysis)
networkx (for graph metrics)
scikit-learn (for UMAP, clustering)
```

Optional:
- `entropy` (sample entropy, permutation entropy)
- `lempel_ziv_complexity`
- `statsmodels` (for Granger causality)

## File Ignores

Per `.gitignore`:
- `data/*` - Raw EEG files
- `exports*`, `session_exports/*` - Analysis outputs
- `*.png`, `*.csv`, `*.npy` - Generated files
- `embedding_frames*` - Animation frames
