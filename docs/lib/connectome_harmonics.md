# connectome_harmonics

## Overview

Connectome Harmonics Engagement Breadth (fs=128)

**Module Statistics:**
- Total Functions: 19
- Public Functions: 11
- Private Functions: 8

## Main Analysis

| Function | Parameters | Description |
|----------|------------|-------------|
| `run_connectome_harmonics_breadth` | `(RECORDS, H, electrodes, ...)` | *No description* |
| `analyze_state` | `(blocks)` | *No description* |

## Plotting

| Function | Parameters | Description |
|----------|------------|-------------|
| `plot_harmonics_power_spectra` | `(spectra, top_k=60)` | *No description* |
| `plot_harmonics_breadth_deltas` | `(delta_df)` | *No description* |

## Computation

| Function | Parameters | Description |
|----------|------------|-------------|
| `build_functional_harmonics_from_baseline` | `(RECORDS, electrodes, ...)` | Build sensor-space functional harmonics from baseline data. |

## Detection

| Function | Parameters | Description |
|----------|------------|-------------|
| `find_channel_series` | `(records, ch_name)` | *No description* |

## Utilities

| Function | Parameters | Description |
|----------|------------|-------------|
| `project_to_harmonics` | `(X, H, orthonormal=True, ridge=0.001)` | Project sensor data X (n_elec × n_times) onto spatial harmonics H (n_elec × n... |
| `mode_power` | `(A)` | Return per-mode average power across time: P_k = mean_t A_k(t)^2. |
| `spectral_entropy` | `(P)` | *No description* |
| `participation_ratio` | `(P)` | *No description* |
| `top_decile_mass` | `(P)` | *No description* |

## Private Helpers

| Function | Description |
|----------|-------------|
| `_get_fs` | *Helper function* |
| `_slice_blocks` | *Helper function* |
| `_fourier_phase_randomize` | *Helper function* |
| `_make_surrogate_block` | *Helper function* |
| `_get_fs` | *Helper function* |
| `_slice_baseline_mask` | *Helper function* |
| `_bandpass` | *Helper function* |
| `_pseudo_wpli` | Pseudo-wPLI (Hilbert analytic) on band-limited data. |
