# harmonic_groups

## Overview

Schumann harmonic group analysis and summarization.

**Module Statistics:**
- Total Functions: 10
- Public Functions: 8
- Private Functions: 2

## Plotting

| Function | Parameters | Description |
|----------|------------|-------------|
| `plot_sr_group_adaptive` | `(records, eeg_channel, sr_channel, ...)` | Compute & plot sliding z-coherence for all f0 in the chosen group with |

## Computation

| Function | Parameters | Description |
|----------|------------|-------------|
| `build_null_threshold` | `(coh, n_null=200, method='block', ...)` | Estimate a null threshold for a sliding coherence trace by resampling the |
| `compute_coherence_at_f0` | `(xe, xs, fs, f0, half)` | Magnitude-squared coherence between signals xe and xs at target frequency f0. |

## Utilities

| Function | Parameters | Description |
|----------|------------|-------------|
| `sr_groups` | `()` | *No description* |
| `win_for_f0` | `(f0, cycles=8, min_win=8.0, max_win=120.0)` | Adaptive window: ensure ≥cycles of f0, clipped to [min_win, max_win]. |
| `half_bw_for_win` | `(win_sec, mult=2.5, min_bw=0.1)` | Choose half-bandwidth from spectral resolution Δf≈2/win_sec, scaled by mult. |
| `sliding_coherence_f0` | `(df, eeg_channel, sr_channel, ...)` | *No description* |
| `summarize_sr_groups` | `(records, eeg_channel, sr_channel, ...)` | *No description* |

## Private Helpers

| Function | Description |
|----------|-------------|
| `_maybe_smoother` | *Helper function* |
| `_build_null_for_series` | *Helper function* |
