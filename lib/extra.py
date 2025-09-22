
# def _plv_timecourse(X, fs, f0, bw, win_sec, step_sec):
#     n = X.shape[1]
#     w = int(round(win_sec*fs)); s = int(round(step_sec*fs))
#     idx = []; i = 0
#     while i + w <= n:
#         idx.append((i, i+w)); i += s
#     if not idx: idx.append((0, n))
#     b = firwin(801, [max(0.1, f0-bw), f0+bw], pass_zero=False, fs=fs)
#     Xb = filtfilt(b, [1.0], X, axis=-1, padlen=min(2400, X.shape[-1]-1))
#     ph = np.angle(hilbert(Xb, axis=-1))
#     R_t = np.abs(np.nanmean(np.exp(1j*ph), axis=0))
#     t_mid = []; plv = []
#     for i0,i1 in idx:
#         t_mid.append((i0+i1)/2/fs)
#         plv.append(float(np.nanmean(R_t[i0:i1])))
#     return np.asarray(t_mid), np.asarray(plv)


# def _hsi_timecourse(X: np.ndarray, fs: float, win_sec: float, step_sec: float, ladder: Sequence[float], ladder_bw: float = 0.6, band=(2,60)) -> Tuple[np.ndarray, np.ndarray]:
#     """Harmonic Spread Index over time (lower = tighter).
#     We form a ladder template L(f) as Gaussian bumps at each harmonic, then
#     compute concentration C = <P(f), L(f)> / <P(f), 1_band(f)>. HSI = 1 - C.
#     """
#     n = X.shape[1]
#     idx = _sliding_windows(n, fs, win_sec, step_sec)
#     t_mid = []
#     hsi = []
#     for i0, i1 in idx:
#         seg = X[:, i0:i1]
#         # median Welch across channels for robustness
#         f, P = welch(seg, fs=fs, nperseg=min(int(fs*2), seg.shape[-1]), axis=-1)
#         Pm = np.nanmedian(P, axis=0)
#         m = (f >= band[0]) & (f <= band[1])
#         fB, PB = f[m], Pm[m]
#         # template
#         L = np.zeros_like(fB)
#         for hk in ladder:
#             L += np.exp(-0.5*((fB - hk)/ladder_bw)**2)
#         L /= (L.max() + 1e-12)
#         C = float(np.sum(PB * L) / (np.sum(PB) + 1e-12))
#         H = 1.0 - C
#         t_mid.append((i0 + i1)/2 / fs)
#         hsi.append(H)
#     return np.asarray(t_mid), np.asarray(hsi)

# def _hsi_timecourse(X: np.ndarray, fs: float, win_sec: float, step_sec: float, ladder, ladder_bw: float = 1.0, band=(2,60)):
#     n = X.shape[1]
#     w = int(round(win_sec*fs)); s = int(round(step_sec*fs))
#     idx = []
#     i = 0
#     while i + w <= n:
#         idx.append((i, i+w))
#         i += s
#     if not idx:
#         idx.append((0, n))

#     t_mid = []
#     hsi = []
#     for i0, i1 in idx:
#         seg = X[:, i0:i1]
#         # Robust PSD (median across channels)
#         f, P = welch(seg, fs=fs, nperseg=min(int(fs*2), seg.shape[-1]), axis=-1)
#         Pm = np.nanmedian(P, axis=0)
#         band_mask = (f >= band[0]) & (f <= band[1])
#         fB, PB = f[band_mask], Pm[band_mask]
#         # Ladder template as sum of Gaussians, then **area normalize**
#         L = np.zeros_like(fB)
#         for hk in ladder:
#             L += np.exp(-0.5*((fB - hk)/ladder_bw)**2)
#         L_sum = np.sum(L)
#         if L_sum > 0:
#             L /= L_sum
#         # Concentration of power on the ladder vs total in the band
#         PB_sum = np.sum(PB) + 1e-12
#         C = float(np.sum(PB * L) / PB_sum)
#         H = 1.0 - C
#         t_mid.append((i0+i1)/2/fs)
#         hsi.append(H)
#     return np.asarray(t_mid), np.asarray(hsi)

# def _hsi_timecourse(X, fs, win_sec, step_sec, ladder, ladder_bw=1.0, band=(2,60)):
#     n = X.shape[1]
#     w = int(round(win_sec*fs)); s = int(round(step_sec*fs))
#     idx = []; i = 0
#     while i + w <= n:
#         idx.append((i, i+w)); i += s
#     if not idx: idx.append((0, n))
#     ts = []; hs = []
#     for i0,i1 in idx:
#         seg = X[:, i0:i1]
#         f, P = welch(seg, fs=fs, nperseg=min(int(fs*2), seg.shape[-1]), axis=-1)
#         Pm = np.nanmedian(P, axis=0)
#         m = (f>=band[0]) & (f<=band[1])
#         fB, PB = f[m], Pm[m]
#         L = np.zeros_like(fB)
#         for hk in ladder:
#             L += np.exp(-0.5*((fB - hk)/ladder_bw)**2)
#         Lsum = np.sum(L)
#         L = L / Lsum if Lsum>0 else L
#         C = float(np.sum(PB * L) / (np.sum(PB)+1e-12))
#         hs.append(1.0 - C); ts.append((i0+i1)/2/fs)
#     return np.asarray(ts), np.asarray(hs)


# def _pac_mvl_timecourse(X: np.ndarray, fs: float, theta_band=(7.0,8.0), gamma_band=(40.0,100.0), win_sec: float = 4.0, step_sec: float = 0.25) -> Tuple[np.ndarray, np.ndarray]:
#     """Compute MVL(t) for θ phase (7–8 Hz) → γ amplitude.

#     Bands are auto‑adjusted to be Nyquist‑safe:
#       • θ: min_bw=0.5 Hz,   γ: min_bw=5 Hz, hi clamped to 0.45*fs.
#       • If requested γ is infeasible at this fs, fallback to γ=(30, min(55, 0.45*fs)).
#       • If still infeasible, returns t and NaNs for MVL.
#     """
#     # Build analysis windows on the full record
#     n = X.shape[1]
#     w = int(round(win_sec * fs)); s = int(round(step_sec * fs))
#     idx = []
#     i = 0
#     while i + w <= n:
#         idx.append((i, i + w))
#         i += s
#     if not idx:
#         idx.append((0, n))

#     # Safe bands
#     th_band = _safe_passband(theta_band, fs, min_bw=0.5, max_frac=0.45, lo_floor=0.5)
#     gm_band = _safe_passband(gamma_band, fs, min_bw=5.0,  max_frac=0.45, lo_floor=5.0)
#     if gm_band is None:
#         gm_fallback = (30.0, min(55.0, 0.45 * fs))
#         gm_band = _safe_passband(gm_fallback, fs, min_bw=5.0, max_frac=0.45, lo_floor=5.0)
#     if th_band is None or gm_band is None:
#         # Not enough bandwidth at this fs — return NaNs (graceful degrade)
#         t_mid = np.array([(a+b)/2/fs for (a,b) in idx], dtype=float)
#         return t_mid, np.full_like(t_mid, np.nan, dtype=float)

#     # Filters
#     b_th = firwin(801, th_band, pass_zero=False, fs=fs)
#     b_gm = firwin(801, gm_band, pass_zero=False, fs=fs)

#     # Filter across channels, analytic signals
#     Xth = filtfilt(b_th, [1.0], X, axis=-1, padlen=min(2400, X.shape[-1]-1))
#     Xgm = filtfilt(b_gm, [1.0], X, axis=-1, padlen=min(2400, X.shape[-1]-1))
#     ph = np.angle(hilbert(Xth, axis=-1))
#     amp = np.abs(hilbert(Xgm, axis=-1))

#     # Reduce across channels
#     ph_med = np.angle(np.nanmean(np.exp(1j*ph), axis=0))
#     amp_med = np.nanmedian(amp, axis=0)

#     # Sliding MVL
#     t_mid = []
#     mvl = []
#     for i0, i1 in idx:
#         t_mid.append((i0 + i1) / 2 / fs)
#         z = amp_med[i0:i1] * np.exp(1j * ph_med[i0:i1])
#         mvl.append(np.abs(np.sum(z)) / (np.sum(amp_med[i0:i1]) + 1e-12))
#     return np.asarray(t_mid), np.asarray(mvl)






# def _pac_mvl_timecourse(X: np.ndarray, fs: float,
#                         theta_band=(7.0,8.0), gamma_band=(40.0,100.0),
#                         win_sec: float = 4.0, step_sec: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     MVL(t): θ phase (median across channels) → γ amplitude.
#     Safe bands: clamp γ to <= 0.45*fs and ensure >=5 Hz width.
#     Unweighted MVL so values live ~0–0.4, not ~0.9.
#     """
#     # clamp gamma
#     gm_hi = min(gamma_band[1], 0.45*fs)
#     gm_lo = max(gamma_band[0], 5.0)
#     if gm_hi - gm_lo < 5.0:
#         c = 0.5*(gm_lo+gm_hi); gm_lo, gm_hi = c-2.5, c+2.5
#     gamma_band = (gm_lo, gm_hi)

#     b_th = firwin(801, theta_band, pass_zero=False, fs=fs)
#     b_gm = firwin(801, gamma_band, pass_zero=False, fs=fs)

#     Xth = filtfilt(b_th, [1.0], X, axis=-1, padlen=min(2400, X.shape[-1]-1))
#     Xgm = filtfilt(b_gm, [1.0], X, axis=-1, padlen=min(2400, X.shape[-1]-1))

#     ph  = np.angle(hilbert(Xth, axis=-1))          # (ch, n)
#     amp = np.abs(hilbert(Xgm, axis=-1))            # (ch, n)

#     # reduce across channels
#     ph_med  = np.angle(np.nanmean(np.exp(1j*ph), axis=0))  # (n,)
#     amp_med = np.nanmedian(amp, axis=0)                     # for masking, not weighting

#     n = X.shape[1]
#     idx = _sliding_windows(n, fs, win_sec, step_sec)
#     t_mid, mvl = [], []
#     for i0, i1 in idx:
#         t_mid.append((i0+i1)/2/fs)
#         ph_seg = ph_med[i0:i1]
#         # unweighted MVL
#         mvl.append(float(np.abs(np.nanmean(np.exp(1j*ph_seg)))))
#     return np.asarray(t_mid), np.asarray(mvl)



# def _bicoherence_triads_timecourse(X: np.ndarray, fs: float, 
#                                    triads: Sequence[Tuple[float,float,float]], 
#                                    bw: float, win_sec: float, 
#                                    step_sec: float) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
#     """Approximate focused bicoherence using narrowband phases: R = |E[e^{i(φ1+φ2−φ3)}]|.
#     We median across channels for robustness.
#     Returns: t, {label: series}
#     """
#     n = X.shape[1]
#     idx = _sliding_windows(n, fs, win_sec, step_sec)
#     # prefilter all involved bands once
#     centers = sorted(set([f for tri in triads for f in tri]))
#     phases = {}
#     for f0 in centers:
#         b = _fir_bandpass(f0, bw, fs)
#         Xb = filtfilt(b, [1.0], X, axis=-1, padlen=3*len(b))
#         phases[f0] = np.angle(hilbert(Xb, axis=-1))  # (n_ch, n)
#     t_mid = []
#     out = {f"({f1},{f2}->{f3})": [] for (f1,f2,f3) in triads}
#     for i0, i1 in idx:
#         t_mid.append((i0 + i1)/2 / fs)
#         for (f1,f2,f3) in triads:
#             # average phase across the window per channel
#             # p1 = np.unwrap(phases[f1][:, i0:i1], axis=-1).mean(axis=-1)
#             # p2 = np.unwrap(phases[f2][:, i0:i1], axis=-1).mean(axis=-1)
#             # p3 = np.unwrap(phases[f3][:, i0:i1], axis=-1).mean(axis=-1)

#             p1 = np.angle(np.mean(np.exp(1j*phases[f1][:, i0:i1]), axis=-1))
#             p2 = np.angle(np.mean(np.exp(1j*phases[f2][:, i0:i1]), axis=-1))
#             p3 = np.angle(np.mean(np.exp(1j*phases[f3][:, i0:i1]), axis=-1))


#             z = np.exp(1j*(p1 + p2 - p3))
#             R = np.abs(np.nanmean(z))
#             out[f"({f1},{f2}->{f3})"].append(R)
#     for k in out:
#         out[k] = np.asarray(out[k])
#     return np.asarray(t_mid), out



# def build_pack_and_make_reports(_records: pd.DataFrame, windows: List[Tuple[float,float]], *, cfg: FeaturePackCfg = FeaturePackCfg(), save_dir: Optional[str] = None):
#     pack = build_ignition_feature_pack(_records, windows, cfg=cfg)
#     # reuse your earlier reporter utilities
#     try:
#         from ignition_reports import make_reports_for_windows
#     except Exception:
#         raise ImportError("Import the previously created ignition_reports module before calling this.")
#     return make_reports_for_windows(pack, windows, save_dir=save_dir)



# def _prep_spec_for_window(spec, window, mode="db-rowz", pct=(2,98)):
#     """
#     spec: (t_spec, f_spec, Sxx) with linear power
#     window: (t0,t1)
#     mode:
#       - 'db-rowz': 10*log10, then row-wise (per frequency) median/MAD z-score
#       - 'db-rel' : 10*log10, no z, but color limits set to robust percentiles
#     """
#     tS, fS, S = spec
#     m = (tS >= window[0]) & (tS <= window[1])
#     if not np.any(m):      # fall back to the nearest slice
#         k = np.argmin(np.abs(tS - np.mean(window)))
#         m = slice(max(0, k-1), min(len(tS), k+1))

#     Sw = S[:, m]                          # (F, Tw)
#     SdB = 10.0 * np.log10(Sw + 1e-20)

#     if mode == "db-rowz":
#         # subtract per-frequency median, scale by MAD (robust z)
#         med = np.median(SdB, axis=1, keepdims=True)
#         mad = np.median(np.abs(SdB - med), axis=1, keepdims=True) + 1e-9
#         Z = (SdB - med) * (1.0 / mad)
#         return tS[m], fS, Z, (-3, 3), "z"
#     else:  # 'db-rel'
#         vmin, vmax = np.percentile(SdB, pct)
#         return tS[m], fS, SdB, (float(vmin), float(vmax)), "Power (dB)"



# def process_window(pack, records, win, eeg_cols, fs, time_col="Timestamp"):
#     t0,t1 = win; t = pack['t']; w = (t>=t0)&(t<=t1)
#     # 1) per-window spec + HSI_v3
#     tW,fW,SW = window_spec_median(records, win, channels=eeg_cols, fs=fs, time_col=time_col,
#                                   band=(2,60), win_sec=1.0, overlap=0.80)
#     tH,H = hsi_v3_from_window_spec(tW,fW,SW, in_bw=0.5, ring_offset=1.5, ring_bw=0.8, smooth_hz=6.0)
#     pack['hsi'][w] = np.interp(t[w], tH, H)
#     # 2) robust-z + light smoothing for envelopes
#     for k in ['z_7p83','z_15p6','z_23p4']:
#         z = robust_z(pack[k][w]); pack[k][w] = smooth_sec(t[w], z, 0.15)
#     # 3) adaptive thresholds
#     z95  = float(np.nanpercentile(pack['z_7p83'][w], 95))
#     plv60= float(np.nanpercentile(pack['plv_7p83'][w], 60))
#     h10  = float(np.nanpercentile(pack['hsi'][w], 10))
#     # p = PhaseParams(
#     #     z_p0=0.6, plv_p0=0.45,
#     #     z_p1=max(1.0, 0.9*z95), plv_p1=plv60,
#     #     hsi_tight=h10, hsi_release=max(h10+0.10, 0.76),
#     #     plv_release=plv60-0.02,
#     #     min_p0_dur=0.10, min_p1_dur=0.10, min_p2_cycles=1.0,
#     #     rel_h2=0.10, rel_h3=0.10, bic_7_7_15=0.0, bic_7_15_23=0.0
#     # )
#     # # 4) detect & plot
#     # fig, phases, traces = plot_ignition_window_report(records,PackProvider(pack).slice(t0,t1),eeg_cols,
#     #                                                   params=p, title=f"Ignition {t0:.2f}–{t1:.2f}s")
#     # return {'window': win, 'phases': phases, 'fig': fig, 'params': p}



# def _narrowband_envelope_z(X, fs, f0, bw):
#     b = firwin(801, [max(0.1, f0-bw), f0+bw], pass_zero=False, fs=fs)
#     Xb = filtfilt(b, [1.0], X, axis=-1, padlen=min(2400, X.shape[-1]-1))
#     H = hilbert(Xb, axis=-1)
#     amp = np.abs(H)
#     amp_med = np.nanmedian(amp, axis=0)
#     m = np.nanmean(amp_med); s = np.nanstd(amp_med)
#     env_z = (amp_med - m) / (s + 1e-9)
#     phase_mean = np.angle(np.nanmean(np.exp(1j*np.angle(H)), axis=0))
#     return env_z.astype(float), phase_mean.astype(float)


# def _hsi_from_window_spec(tW, fW, SW, ladder=(7.83, 15.6, 23.4, 31.2, 39.0, 46.8, 54.6),
#                           win_half_hz=0.6, smooth_hz=6.0):
#     """HSI(t) ∈ [0,1], lower = tighter. Flattens 1/f, then measures excess power on SR lines."""
#     df = float(np.median(np.diff(fW)))
#     W = max(5, int(np.ceil(smooth_hz/df)));  W += (W % 2 == 0)  # odd length
#     logS = np.log(SW + 1e-20)
#     bg   = savgol_filter(logS, window_length=W, polyorder=2, axis=0, mode='interp')
#     R    = np.exp(logS - bg)  # ratio >1 where peaks exceed 1/f

#     M = np.zeros_like(fW, float)
#     for hk in ladder:
#         M += (np.abs(fW - hk) <= win_half_hz).astype(float)

#     C = (R * M[:, None]).sum(axis=0) / (R.sum(axis=0) + 1e-12)
#     H = 1.0 - C
#     return tW, H

# def _prettify(ax):
#     ax.tick_params(axis='both', which='major', length=5, width=0.8)
#     ax.tick_params(axis='both', which='minor', length=3, width=0.6)
#     for spine in ("top","right"):
#         ax.spines[spine].set_visible(False)
        
# def _runlen_first_onset(mask: np.ndarray, t: np.ndarray, min_dur: float) -> Optional[int]:
#     if mask is None or t is None or len(mask) == 0:
#         return None
#     mask = np.asarray(mask, dtype=bool)
#     t = np.asarray(t, dtype=float)
#     dm = np.diff(mask.astype(int), prepend=0, append=0)
#     starts = np.where(dm == 1)[0]
#     ends   = np.where(dm == -1)[0] - 1
#     for s, e in zip(starts, ends):
#         if t[e] - t[s] >= min_dur:
#             return int(s)
#     return None

# def _sustained(mask: np.ndarray, t: np.ndarray, min_dur: float) -> np.ndarray:
#     mask = np.asarray(mask, bool)
#     keep = np.zeros_like(mask)
#     dm = np.diff(mask.astype(int), prepend=0, append=0)
#     starts = np.where(dm == 1)[0]
#     ends   = np.where(dm == -1)[0] - 1
#     for s, e in zip(starts, ends):
#         if t[e] - t[s] >= min_dur:
#             keep[s:e+1] = True
#     return keep


# def _quick_sanity(pack):
#     return {
#         'z_7p83_std': float(np.nanstd(pack['z_7p83'])),
#         'plv_median': float(np.nanmedian(pack['plv_7p83'])),
#         'hsi_range': (float(np.nanmin(pack['hsi'])), float(np.nanmax(pack['hsi']))),
#     }


# def _safe_passband(band: Sequence[float], fs: float, *, min_bw: float, max_frac: float = 0.45, lo_floor: float = 0.5) -> Optional[Tuple[float,float]]:
#     """Clamp (lo, hi) to Nyquist‑safe range and ensure minimum bandwidth.

#     Parameters
#     ----------
#     band : (lo, hi) desired band in Hz
#     fs : sampling rate in Hz
#     min_bw : minimum allowable bandwidth in Hz
#     max_frac : keep hi <= max_frac*fs (default 0.45)
#     lo_floor : keep lo >= lo_floor Hz (default 0.5)

#     Returns
#     -------
#     (lo, hi) if feasible, else None
#     """
#     lo, hi = float(band[0]), float(band[1])
#     if not np.isfinite(lo) or not np.isfinite(hi):
#         return None
#     if lo > hi:
#         lo, hi = hi, lo
#     hi = min(hi, max_frac * fs)
#     lo = max(lo, lo_floor)
#     # widen around center if too narrow
#     if hi - lo < min_bw:
#         c = 0.5 * (lo + hi)
#         lo = max(lo_floor, c - 0.5 * min_bw)
#         hi = min(max_frac * fs, c + 0.5 * min_bw)
#     if hi - lo < min_bw:
#         return None
#     return (lo, hi)

# def events_df_to_windows(df, *, t_start_col: str = 't_start', t_end_col: str = 't_end') -> List[Tuple[float,float]]:
#     """Extract (t0,t1) ignition windows from an events DataFrame.

#     Accepts a pandas DataFrame with at least 't_start' and 't_end' columns.
#     Returns a list of (float t0, float t1) pairs.
#     """
#     import pandas as pd  # type: ignore
#     if not isinstance(df, pd.DataFrame):
#         raise TypeError("events_df_to_windows expects a pandas DataFrame")
#     if t_start_col not in df.columns or t_end_col not in df.columns:
#         raise ValueError(f"DataFrame must have '{t_start_col}' and '{t_end_col}' columns; has: {list(df.columns)}")
#     t0 = df[t_start_col].to_numpy(dtype=float)
#     t1 = df[t_end_col].to_numpy(dtype=float)
#     return [(float(a), float(b)) for a, b in zip(t0, t1)]

# def _zscore(x: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
#     m = np.nanmean(x, axis=axis, keepdims=True)
#     s = np.nanstd(x, axis=axis, keepdims=True)
#     return (x - m) / (s + 1e-9)



# def attach_spectrogram(_records,pack,ign_windows):
#     # 1) Attach a session spectrogram to the pack (2–60 Hz, Hann 2s, 75% overlap)
#     pack['spec'] = compute_session_spectrogram(
#         _records, channels='auto', time_col='Timestamp', fs=128,
#         band=(2,60), win_sec=2.0, overlap=0.75,
#     )
    
#     tH, H = hsi_v3_from_window_spec(tW, fW, SW, in_bw=0.5, ring_offset=1.5, ring_bw=0.8, smooth_hz=6.0)
#     print("HSI_v3 window min/max:", float(H.min()), float(H.max()))
    
#     # resample to the report time base:
#     # patch the pack for that window so provider.hsi() returns the good curve
#     m = (pack['t'] >= 556) & (pack['t'] <= 581)
#     pack['hsi'][m] = np.interp(pack['t'][m], tH, H)
    
    
#     # do this for each ignition window (example: 556–581)
#     # for (t0, t1) in [ign_win]:  # or IGNITION_WINDOWS
#     m = (pack['t'] >= ign_win[0]) & (pack['t'] <= ign_win[1])
#     for k in ['z_7p83','z_15p6','z_23p4']:
#         pack[k][m] = robust_z(pack[k][m])
    
#     # sanity in the window you’re plotting:
#     m = (pack['t']>=556)&(pack['t']<=581)
#     print("std z@7.83:", float(np.nanstd(pack['z_7p83'][m])))
#     # should be ~1.0; the plot y-axis should lose the 1e-12 annotation
    
#     z95  = float(np.nanpercentile(pack['z_7p83'][m], 95))
#     plv60= float(np.nanpercentile(pack['plv_7p83'][m], 60))
#     h10  = float(np.nanpercentile(pack['hsi'][m], 10))
#     print("suggested: z_p1≈", round(0.9*z95,2), "plv_p1≈", round(plv60,3), "hsi_tight≈", round(h10,3))
    
    
#     # you already have tH, H from hsi_v3_from_window_spec(...)
#     m = (pack['t'] >= 556) & (pack['t'] <= 581)
#     pack['hsi'][m] = np.interp(pack['t'][m], tH, H)
    
    
#     # compute adaptive thresholds from this window (so you don't guess)
#     z95   = float(np.nanpercentile(pack['z_7p83'][w], 95))  # e.g., ~1.6–1.8 after robust z
#     plv60 = float(np.nanpercentile(pack['plv_7p83'][w], 60))# ~0.55
#     h10   = float(np.nanpercentile(pack['hsi'][w], 10))     # ~0.53–0.57 with HSI_v3
    
#     # slice the window and smooth
#     twin = pack['t']
#     w = (twin >= 556) & (twin <= 581)                    # <-- or loop all IGNITION_WINDOWS
#     zf_s  = smooth_sec(twin[w], pack['z_7p83'][w], 0.15) # 150 ms
#     z2_s  = smooth_sec(twin[w], pack['z_15p6'][w], 0.15)
#     z3_s  = smooth_sec(twin[w], pack['z_23p4'][w], 0.15)
    
#     # OPTIONAL: write back just for this window so the provider sees the smoothed series
#     pack['z_7p83'][w] = zf_s
#     pack['z_15p6'][w] = z2_s
#     pack['z_23p4'][w] = z3_s
    
#     # params = PhaseParams(
#     #     z_p0=0.6,  plv_p0=0.45,         # easy prelude
#     #     z_p1=max(1.0, 0.9*z95),         # ~10% below the 95th pct of z
#     #     plv_p1=plv60,                   # sits on your PLV level
#     #     hsi_tight=h10,                  # sits on your HSI dip
#     #     hsi_release=0.80, plv_release=plv60,
#     #     min_p0_dur=0.10,                # 100 ms (was 250 ms)
#     #     min_p1_dur=0.10,
#     #     min_p2_cycles=1.0,              # ~1 cycle at 7.83 Hz ≈ 128 ms
#     #     rel_h2=0.05, rel_h3=0.05,       # relaxed while validating
#     #     bic_7_7_15=0.0, bic_7_15_23=0.0 # re-enable later
#     # )
    
    
#     # params.rel_h2 = 0.10
#     # params.rel_h3 = 0.10
    
#     # params.hsi_release = max(params.hsi_tight + 0.10, 0.76)
#     # params.plv_release = max(0.0, params.plv_p1 - 0.02)
    
#     # fig, phases, traces = plot_ignition_window_report(_records,PackProvider(pack).slice(556,581),
#     #                                                   params=params, title="Ignition 556–581s")
#     # print(phases)
