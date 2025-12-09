# criticality

## Overview

Criticality Signatures — 1/f, DFA, Avalanches vs Conscious Mode (fs=128)

**Module Statistics:**
- Total Functions: 13
- Public Functions: 12
- Private Functions: 1

## Main Analysis

| Function | Parameters | Description |
|----------|------------|-------------|
| `run_criticality_analysis` | `(RECORDS, ignition_windows, ...)` | *No description* |
| `analyze_state` | `(blocks)` | *No description* |

## Plotting

| Function | Parameters | Description |
|----------|------------|-------------|
| `plot_criticality_deltas` | `(delta_df)` | *No description* |
| `plot_avalanche_ccdf` | `(aval_dict)` | Plot complementary CDFs of avalanche sizes/durations per state (log–log). |

## Detection

| Function | Parameters | Description |
|----------|------------|-------------|
| `find_channel_series` | `(records, ch_name)` | *No description* |

## Data Processing

| Function | Parameters | Description |
|----------|------------|-------------|
| `bandpass` | `(x, fs, f1, f2, order=4)` | *No description* |
| `slice_blocks` | `(RECORDS, time_col, X, fs, winlist)` | *No description* |

## Utilities

| Function | Parameters | Description |
|----------|------------|-------------|
| `welch_beta` | `(x, fs, fmin=1.0, fmax=40.0)` | Robust 1/f slope β via log–log linear regression on Welch PSD between fmin–fmax. |
| `dfa_alpha` | `(x, fs, scales_sec)` | Detrended fluctuation analysis (DFA) exponent α on band-limited amplitude env... |
| `avalanche_events` | `(env, thresh)` | Return [(start_idx, end_idx, size)] where size is area under envelope above t... |
| `avalanche_stats` | `(envs, fs, thresh_mode='p95')` | *No description* |
| `fit_powerlaw_tail` | `(x, xmin=None)` | *No description* |

## Private Helpers

| Function | Description |
|----------|-------------|
| `_get_fs` | *Helper function* |
