# directed_connectivity

## Overview

DLPFC → Sensory Top-Down Connectivity (Source-space)

**Module Statistics:**
- Total Functions: 8
- Public Functions: 5
- Private Functions: 3

## Main Analysis

| Function | Parameters | Description |
|----------|------------|-------------|
| `run_topdown_ignition_pipeline` | `(records, electrodes, fs, raw, ...)` | Run SENSOR or SOURCE pipeline. |

## Utilities

| Function | Parameters | Description |
|----------|------------|-------------|
| `df_to_raw` | `(records, ch_names, sfreq, ...)` | Convert a (Timestamp + EEG.<ch>.FILTERED) DataFrame into MNE RawArray. |
| `phase_transfer_entropy_stub` | `()` | Stub for PTE (wire IDTxl/JIDT here if desired). |
| `sensor_directed_connectivity` | `(raw, windows, bands=BANDS, ...)` | Compute dPLI + (pairwise) Granger from F4→posterior sensors per event window ... |
| `source_directed_connectivity` | `(raw, windows, subjects_dir, subject, ...)` | Compute dPLI + conditional Granger: DLPFC_R→(occipital/temporal/parietal) per... |

## Private Helpers

| Function | Description |
|----------|-------------|
| `_compute_dpli` | dPLI(source→target) for a 2×T array (row0=src, row1=tgt). |
| `_conditional_granger` | Conditional Granger: does src cause tgt given other series? ... |
| `_pick` | *Helper function* |
