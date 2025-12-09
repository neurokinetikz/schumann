# shape_vs_resonance

## Overview

Disentangling waveform shape vs. true multi‑mode resonance (0.1–60 Hz)

**Module Statistics:**
- Total Functions: 21
- Public Functions: 18
- Private Functions: 3

## Main Analysis

| Function | Parameters | Description |
|----------|------------|-------------|
| `analyze_shape_vs_resonance` | `(RECORDS, eeg_channel, sr_channel, ...)` | Run morphology, IRASA, and (cross‑)bicoherence tests; save figures and a summ... |

## Utilities

| Function | Parameters | Description |
|----------|------------|-------------|
| `ensure_dir` | `(d)` | *No description* |
| `ensure_timestamp_column` | `(df, time_col='Timestamp', ...)` | *No description* |
| `infer_fs` | `(df, time_col='Timestamp')` | *No description* |
| `get_series` | `(df, name)` | *No description* |
| `cycles_morphology` | `(x, fs, f0=7.83, half=1.0, sharp_win=0.02)` | Compute cycle features from bandpassed x around f0±half. |
| `next_idx` | `(arr, i)` | *No description* |
| `irasa_psd` | `(x, fs, hset=..., nperseg=None, fmax=60.0)` | Approximate IRASA: resample by h and 1/h, compute PSDs, map freqs back by /h ... |
| `res_psd` | `(y, up, down)` | *No description* |
| `bicoherence_discrete_auto` | `(x, fs, f_list, nperseg=None, step=None)` | Auto‑bicoherence on a discrete frequency list f_list (Hz). Returns matrix B[i... |
| `bin_idx` | `(f)` | *No description* |
| `bicoherence_discrete_cross` | `(sr, eeg, fs, f_list, nperseg=None, ...)` | Cross‑bicoherence variant: B_sse(f1,f2) = <S(f1) S(f2) E*(f1+f2)> / sqrt(<\|S(... |
| `bin_idx` | `(f)` | *No description* |
| `at` | `(freq, bw=0.4)` | *No description* |
| `sur_cross` | `(nr=200)` | *No description* |
| `sur_auto` | `(nr=200)` | *No description* |
| `nearest_idx` | `(arr, val)` | *No description* |
| `heat` | `(M, thr, title, fname)` | *No description* |

## Private Helpers

| Function | Description |
|----------|-------------|
| `_bandpass` | *Helper function* |
| `_lowpass` | *Helper function* |
| `_fft_segments` | *Helper function* |
