# hidden_markov

## Overview

Event-related & HMM approaches — simple validity tests & graphs

**Module Statistics:**
- Total Functions: 18
- Public Functions: 18
- Private Functions: 0

## Main Analysis

| Function | Parameters | Description |
|----------|------------|-------------|
| `run_hmm_state_tests` | `(RECORDS, eeg_channels, sr_channel, ...)` | 1) Build band-power features; fit GMM(K) → state(t). |

## Detection

| Function | Parameters | Description |
|----------|------------|-------------|
| `detect_schumann_bursts` | `(RECORDS, sr_channel, ...)` | *No description* |

## Data Processing

| Function | Parameters | Description |
|----------|------------|-------------|
| `bandpass` | `(x, fs, f1, f2, order=4)` | *No description* |
| `slice_epoch` | `(x, i0, i1)` | *No description* |

## Utilities

| Function | Parameters | Description |
|----------|------------|-------------|
| `infer_fs` | `(RECORDS, time_col='Timestamp')` | *No description* |
| `get_series` | `(RECORDS, name)` | *No description* |
| `morlet_cwt` | `(sig, fs, freqs, w0=6.0)` | Complex Morlet CWT via FFT-convolution. |
| `erp_ersp_itc` | `(RECORDS, eeg_channels, sr_channel, ...)` | Build ERP/ERSP/ITC time-locked to Schumann bursts and run simple cluster-perm... |
| `max_cluster_mass` | `(mask, value_map)` | *No description* |
| `circ_shift_2d` | `(A, sh0, sh1)` | *No description* |
| `bandpower_features` | `(RECORDS, eeg_channels, ...)` | Sliding-window mean band power per band, averaged over EEG channels. |
| `schumann_amplitude` | `(RECORDS, sr_channel, ...)` | Sliding-window Schumann envelope mean in the same grid as bandpower_features. |
| `hmm_states_gmm` | `(features, K=3, random_state=0)` | Fit GaussianMixture as a simple HMM surrogate; decode state(t). |
| `eta_state_occupancy` | `(states, T, event_times, ...)` | Event-triggered state occupancy around event_times (ETA). |
| `logistic_state_transition_vs_amp` | `(states, amp)` | Binary transition Y: 1 if state changes at t+1; regress on Schumann amp(t). |
| `erp_ersp_itc_safe` | `(RECORDS, eeg_channels, sr_channel, ...)` | Schumann-locked ERP/ERSP/ITC with edge padding and TF cluster-perm. |
| `take_segment` | `(x, i_on)` | *No description* |
| `max_cluster_mass` | `(mask, val)` | *No description* |
