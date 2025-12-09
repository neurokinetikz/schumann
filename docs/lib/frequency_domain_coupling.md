# frequency_domain_coupling

## Overview

Frequency-domain coupling (core): Multi-taper MSC + Wavelet Coherence (WTC)
fs is inferred from RECORDS[time_col] (or override easily).

**Module Statistics:**
- Total Functions: 25
- Public Functions: 23
- Private Functions: 2

## Main Analysis

| Function | Parameters | Description |
|----------|------------|-------------|
| `run_multitaper_msc_harmonics` | `(RECORDS, x_channels, y_channel, ...)` | MSC between mean of x_channels and y_channel over 'windows'. |
| `run_wavelet_coherence` | `(RECORDS, x_channel, y_channel, ...)` | Morlet wavelet coherence with cluster-based permutation correction. |
| `run_plv_harmonics_topography` | `(RECORDS, eeg_channels, sr_channel, ...)` | Compute PLV and mean phase for each EEG channel vs Schumann reference at Schu... |

## Plotting

| Function | Parameters | Description |
|----------|------------|-------------|
| `plot_msc_harmonics_table` | `(msc_table, ax=None, label='state', ...)` | Plot MSC with jackknife CIs at harmonic lines from a single run_multitaper_ms... |
| `plot_msc_harmonics_compare` | `(msc_ign, msc_base, title=...)` | Side-by-side bars with jackknife CIs comparing ignition vs baseline. |
| `plot_plv_topography` | `(plv_table, chan_pos, freq, vmin=0.0, ...)` | Simple 2D scatter topography for a single harmonic: |

## Data Processing

| Function | Parameters | Description |
|----------|------------|-------------|
| `slice_concat` | `(x, fs, windows)` | Concatenate [t0,t1] windows (sec) from x; if windows is None/empty, return x. |
| `slice_concat` | `(x, fs, windows)` | Concatenate [t0,t1] windows in seconds; if None/empty, return full signal. |
| `bandpass` | `(x, fs, f1, f2, order=4)` | *No description* |

## Utilities

| Function | Parameters | Description |
|----------|------------|-------------|
| `infer_fs` | `(RECORDS, time_col='Timestamp')` | *No description* |
| `get_series` | `(RECORDS, name)` | Return signal column. Accepts 'EEG.O1' or bare 'O1' (will try 'EEG.O1'). |
| `cwt` | `(sig)` | *No description* |
| `smooth` | `(A, wlen=7)` | *No description* |
| `infer_fs` | `(RECORDS, time_col='Timestamp')` | *No description* |
| `get_series` | `(RECORDS, name)` | Return a numeric signal array. Accepts 'EEG.O1' or bare 'O1'. |
| `plv_and_mean_phase` | `(eeg, sr, fs, center_hz, half_bw_hz=0.6)` | Compute PLV and circular mean phase difference between EEG and SR at a narrow... |
| `xcorr_envelopes_peaklag` | `(eeg, sr, fs, center_hz=7.83, ...)` | Band-limit both to [center±half_bw], take Hilbert envelopes and compute norma... |
| `bootstrap_peaklag_ci` | `(eeg, sr, fs, center_hz=7.83, ...)` | Circular-shift bootstrap null for peak lag. Returns {'peak_ms':..., 'ci':(lo,... |
| `scf_cyclic_periodogram` | `(x, fs, alpha_hz, nperseg=2048, ...)` | Simple cyclic periodogram SCF estimator at cyclic frequency alpha (Hz): |
| `scf_at_harmonics` | `(RECORDS, channel, harmonics=..., ...)` | Compute SCF magnitude integrated over frequency for each Schumann cyclic freq... |
| `scf_cyclic_periodogram_demod` | `(x, fs, alpha_hz, nperseg=4096, ...)` | Cyclostationary spectral correlation via complex demodulation: |
| `scf_at_harmonics` | `(RECORDS, channel, harmonics=..., ...)` | Compute SCF magnitude integrated over f for each Schumann cyclic α using demo... |
| `scf_cyclic_periodogram_demod` | `(x, fs, alpha_hz, nperseg=4096, ...)` | Cyclostationary spectral correlation via complex demodulation: |

## Private Helpers

| Function | Description |
|----------|-------------|
| `_mtm_cross_spectra` | Thomson multi-taper auto/cross spectra with jackknife (leave... |
| `_mtm_cross_spectra` | Thomson multi-taper auto/cross spectra with jackknife CIs. |
