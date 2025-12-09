# test

## Overview

Test utilities and experimental functions.

**Module Statistics:**
- Total Functions: 128
- Public Functions: 53
- Private Functions: 75

## Plotting

| Function | Parameters | Description |
|----------|------------|-------------|
| `plot_ignition_window_report` | `(_records, provider, electrodes)` | *No description* |

## Computation

| Function | Parameters | Description |
|----------|------------|-------------|
| `compute_session_spectrogram` | `(_records)` | Return a robust session spectrogram as (t_spec_abs, f_spec, Sxx_med). |
| `build_ignition_feature_pack` | `(_records, windows)` | *No description* |
| `estimate_sr_peaks` | `(records, fs, ign_win, ...)` | Get a simple list of estimated SR harmonic frequencies from ignition window E... |

## Utilities

| Function | Parameters | Description |
|----------|------------|-------------|
| `p2_min_dur` | `()` | *No description* |
| `slice` | `(t0, t1)` | *No description* |
| `t` | `()` | *No description* |
| `z_fund` | `()` | *No description* |
| `z_h2` | `()` | *No description* |
| `z_h3` | `()` | *No description* |
| `plv_fund` | `()` | *No description* |
| `hsi` | `()` | *No description* |
| `beta` | `()` | *No description* |
| `ridge_is_fund` | `()` | *No description* |
| `bic_7_7_15` | `()` | *No description* |
| `bic_7_15_23` | `()` | *No description* |
| `pac_mvl` | `()` | *No description* |
| `spectrogram` | `()` | *No description* |
| `slice` | `(t0, t1)` | *No description* |
| `t` | `()` | *No description* |
| `z_fund` | `()` | *No description* |
| `z_h2` | `()` | *No description* |
| `z_h3` | `()` | *No description* |
| `plv_fund` | `()` | *No description* |
| `hsi` | `()` | *No description* |
| `beta` | `()` | *No description* |
| `ridge_is_fund` | `()` | *No description* |
| `bic_7_7_15` | `()` | *No description* |
| `bic_7_15_23` | `()` | *No description* |
| `pac_mvl` | `()` | *No description* |
| `spectrogram` | `()` | *No description* |
| `spectrogram_for_window` | `(t0, t1)` | *No description* |
| `window_spec_median` | `(records, window)` | Robust spectrogram inside `window`: |
| `patch_pack_with_hsi_v3_for_windows` | `(pack, records, windows)` | Compute a per-window spectrogram from time-domain median, derive HSI_v3, |
| `repl` | `(match)` | *No description* |
| `hsi_from_spec_v2` | `(spec, ladder=..., win_half_hz=0.6, ...)` | spec = (tS, fS, S) with S shape (F,T), linear power. |
| `hsi_v3_from_window_spec` | `(tW, fW, SW)` | HSI_v3(t): lower = tighter harmonics. |
| `sanity` | `(pack)` | *No description* |
| `piano_roll_from_spec` | `(spec_by_window)` | spec_by_window: (tW, fW, SW) from your per-window spectrogram |
| `bandtrace_from_spec` | `(tW, fW, SW, f0, bw=0.8)` | *No description* |
| `z_norm` | `(y)` | *No description* |
| `z_for_display` | `(t, y)` | *No description* |
| `z_for_display` | `(t_vec, y_vec, s=0.45)` | *No description* |
| `to_abs` | `(ts, ys)` | *No description* |
| `interp_to_raw` | `(t_src, y_src)` | *No description* |
| `robust_z` | `(x)` | *No description* |
| `smooth_sec` | `(t, y, sec=0.15)` | *No description* |
| `annotate_phases` | `(ax, phases, ymin, ymax)` | Annotate phases on plot. Supports both old 5-phase (P0-P4) and new 6-phase mo... |
| `six_panel` | `(records, electrodes, ign_win, ...)` | *No description* |
| `six_panel_2` | `(records, electrodes, ign_win, ...)` | *No description* |
| `sr_signature_panel` | `(records, electrodes, ign_win, ...)` | *No description* |
| `ignition_signature_panel` | `(records, electrodes, ign_win, ...)` | *No description* |
| `six_panel_3` | `(records, electrodes, ign_win, ...)` | *No description* |

## Private Helpers

| Function | Description |
|----------|-------------|
| `_resolve_palette` | *Helper function* |
| `__init__` | *Helper function* |
| `_get` | *Helper function* |
| `_get_any` | *Helper function* |
| `_slice_spec_to_window` | Return (tW, fW, SW) for the window. If the mask is too small... |
| `_spec_db_rowz` | 10*log10 then row-wise robust z (median/MAD). |
| `_as_float_1d` | *Helper function* |
| `_normalize_channel_label` | *Helper function* |
| `_resolve_seed_channel_index` | *Helper function* |
| `_match_ignition_event_row` | *Helper function* |
| `_format_numeric_labels` | *Helper function* |
| `_extract_seed_channel` | *Helper function* |
| `_bp_hilbert_env_z` | *Helper function* |
| `_plv_7p8` | *Helper function* |
| `_spec_median` | *Helper function* |
| `_hsi_from_spec` | *Helper function* |
| `_first_onset` | *Helper function* |
| `_collect_runs` | *Helper function* |
| `_band_mask` | *Helper function* |
| `_clip_seed_to_window` | *Helper function* |
| `_robust_z` | *Helper function* |
| `_winsor_robust_z` | Winsorized robust‑z (MAD∨IQR) to avoid explosive scales in q... |
| `_rising_over_tau` | *Helper function* |
| `_bridge` | Bridge short False gaps (morphological closing) to avoid mic... |
| `_spectral_slope_series` | *Helper function* |
| `_avalanche_size_duration` | *Helper function* |
| `_kuramoto_order_series` | *Helper function* |
| `_msc_channel_to_reference` | *Helper function* |
| `_msc_matrix` | *Helper function* |
| `_plv_matrix` | *Helper function* |
| `_mode_metrics` | *Helper function* |
| `_interp_safe` | *Helper function* |
| `_te_matrix` | *Helper function* |
| `_transfer_entropy_proxy` | *Helper function* |
| `_sample_entropy` | *Helper function* |
| `_phi` | *Helper function* |
| `_complexity_series` | *Helper function* |
| `_hurst_exponent` | *Helper function* |
| `_lempel_ziv_complexity` | *Helper function* |
| `_lz_complexity_series` | *Helper function* |
| `_baseline_slice` | *Helper function* |
| `_infer_fs` | *Helper function* |
| `_looks_like_eeg_col` | *Helper function* |
| `_auto_channels` | *Helper function* |
| `_get_matrix` | *Helper function* |
| `_fir_bandpass` | *Helper function* |
| `_fir_lowpass` | *Helper function* |
| `_sliding_windows` | *Helper function* |
| `_plv_across_channels` | PLV across channels for one time point given phase per-chann... |
| `_plv_timecourse` | Correct: compute channel resultant \|mean(exp(i*phi_ch(t)))\| ... |
| `_msc_timecourse` | Sliding magnitude-squared coherence between the broadband ch... |
| `_narrowband_envelope_z` | *Helper function* |
| `_hsi_timecourse` | *Helper function* |
| `_pac_tort_mi_timecourse` | *Helper function* |
| `_pac_mvl_timecourse` | *Helper function* |
| `_bicoherence_triads_timecourse` | *Helper function* |
| `_detect_ignition_phases` | Phase-aware ignition detector returning P0-P3 events and con... |
| `_np_percentile` | *Helper function* |
| `_robust_z` | *Helper function* |
| `_sigmoid` | *Helper function* |
| `_first_run` | Return the first contiguous run of `True` values long enough... |
| `_gauss_smooth` | *Helper function* |
| `_snap_event` | *Helper function* |
| `_detect_six_phase_evolution` | Six-phase temporal evolution detection for SR ignitions. |
| `_smooth` | *Helper function* |
| `_find_phase_boundary` | Find phase boundary by searching in direction until criterio... |
| `_sigmoid` | *Helper function* |
| `_phase_confidence` | Compute confidence based on signal characteristics in phase. |
| `_time_to_idx` | *Helper function* |
| `_annotate_six_phases` | Annotate six-phase temporal evolution model. |
| `_annotate_five_phases_legacy` | Legacy five-phase annotation (P0-P4). |
| `_get_event` | *Helper function* |
| `_safe_envelope` | *Helper function* |
| `_to_z` | *Helper function* |
| `_apply_sunrise_style` | *Helper function* |
