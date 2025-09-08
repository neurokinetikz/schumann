"""
Cross-frequency coupling expansions tied to Schumann harmonics (0.1–60 Hz)
==========================================================================
Implements simple tests + graphs for:
  1) CF-PLV across the harmonic ladder: PLV between φ_7.83 and φ_{m·7.83} using 1:m locking
     (i.e., |<e^{i(m φ1 − φm)}>|). Baseline vs ignition with surrogate p-values.
  2) Band-limited PAC per order: MVL & Tort MI with low-phase fixed at each Schumann line and
     high-frequency amplitude at the corresponding harmonic band; ignition vs baseline;
     **shape-controlled surrogates** via within-cycle phase shuffles.
  3) Cross-frequency directionality (CFDC-like): lagged regression ΔR² where phase@7.83 predicts
     A_m at t+τ beyond A_m(t); reversed model estimates A→phase; surrogate p-values via
     circular time-shifts. Reports peak ΔR² and lag.

Outputs
-------
• PNGs: CF-PLV bars, PAC bars (MVL/MI), directionality lag curves (+ reverse), per harmonic.
• CSV: cfc_harmonics_summary.csv with metrics & p-values per window and order.

Deps: numpy, scipy, matplotlib, pandas.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal

# ---------------------------- helpers ----------------------------

def ensure_dir(d):
    if d: os.makedirs(d, exist_ok=True)
    return d

def ensure_timestamp_column(df, time_col='Timestamp', default_fs=128.0):
    if time_col in df.columns:
        s = df[time_col]
        if np.issubdtype(s.dtype, np.datetime64) or 'datetime' in str(s.dtype).lower():
            tsec=(pd.to_datetime(s)-pd.to_datetime(s).iloc[0]).dt.total_seconds().astype(float)
            df[time_col] = tsec.values; return time_col
        sn = pd.to_numeric(s, errors='coerce').astype(float)
        if sn.notna().sum()>max(50,0.5*len(df)):
            sn = sn - np.nanmin(sn[np.isfinite(sn)])
            df[time_col] = sn.values; return time_col
    df[time_col] = np.arange(len(df), dtype=float)/default_fs
    return time_col

def infer_fs(df, time_col='Timestamp'):
    t = np.asarray(df[time_col].values, float)
    dt = np.diff(t); dt = dt[(dt>0)&np.isfinite(dt)]
    if dt.size==0: raise ValueError('Cannot infer fs from time column.')
    return float(1.0/np.median(dt))

def get_series(df, name):
    if name in df.columns:
        return pd.to_numeric(df[name], errors='coerce').fillna(0.0).values.astype(float)
    alt = 'EEG.'+name
    if alt in df.columns:
        return pd.to_numeric(df[alt], errors='coerce').fillna(0.0).values.astype(float)
    raise ValueError(f'{name} not found in dataframe columns.')

# Filters & analytic

def _bandpass(x, fs, lo, hi, order=4):
    ny=0.5*fs; lo=max(1e-6, min(lo, 0.99*ny)); hi=max(lo+1e-6, min(hi, 0.999*ny))
    b,a=signal.butter(order, [lo/ny, hi/ny], btype='band'); return signal.filtfilt(b,a,x)

def phase_at(x, fs, f0, half):
    xb = _bandpass(x, fs, f0-half, f0+half)
    z  = signal.hilbert(xb)
    return np.angle(z)  # radians

def amp_envelope(x, fs, f0, half):
    xb = _bandpass(x, fs, f0-half, f0+half)
    return np.abs(signal.hilbert(xb))

# Windows utilities

def windows_to_samples(wins, fs, N):
    segs=[]
    for (a,b) in (wins or []):
        i0,i1=int(round(a*fs)), int(round(b*fs))
        i0=max(0,i0); i1=min(N,i1)
        if i1>i0: segs.append((i0,i1))
    return segs

def concat_segments(x, segs):
    return np.concatenate([x[i0:i1] for (i0,i1) in segs]) if segs else np.array([])

# CF-PLV (1:m locking)

def cf_plv(phi1, phim, m, wins_samp):
    phi1_w = concat_segments(phi1, wins_samp)
    phim_w = concat_segments(phim, wins_samp)
    L = min(len(phi1_w), len(phim_w))
    if L < 50:
        return np.nan, np.nan
    dphi = (m*phi1_w[:L] - phim_w[:L])
    plv = np.abs(np.mean(np.exp(1j*dphi)))
    # Surrogates: circular shift phim relative to phi1
    rng = np.random.default_rng(7)
    null=[]
    for _ in range(200):
        s = int(rng.integers(1, L-1))
        d = (m*phi1_w[:L] - np.r_[phim_w[-s:], phim_w[:-s]][:L])
        null.append(np.abs(np.mean(np.exp(1j*d))))
    p = float((np.sum(np.asarray(null) >= plv) + 1) / (len(null) + 1))
    return float(plv), p

# PAC metrics

def mvl(phase, amp):
    amp = np.asarray(amp, float)
    phase = np.asarray(phase, float)
    return float(np.abs(np.nanmean(amp * np.exp(1j*phase))) / (np.nanmean(amp) + 1e-12))

def tort_mi(phase, amp, nbins=18):
    edges = np.linspace(-np.pi, np.pi, nbins+1)
    bins = np.digitize(phase, edges) - 1
    bins = np.clip(bins, 0, nbins-1)
    m = np.zeros(nbins)
    for k in range(nbins):
        sel = (bins==k)
        m[k] = np.mean(amp[sel]) if np.any(sel) else 0.0
    if m.sum() <= 0:
        return 0.0
    p = m / m.sum(); eps=1e-12
    return float(np.sum(p*np.log((p+eps)/(1.0/nbins))) / np.log(nbins))

# Within-cycle phase-shuffle surrogate for PAC

def pac_within_cycle_surrogate(phase_slow, amp_fast, fs, n_perm=200):
    # detect cycles by zero-crossings of the slow signal (phase unwrap is tricky)
    slow_sig = np.sin(phase_slow)  # proxy signal at the slow band
    zc = np.where(np.diff(np.signbit(slow_sig).astype(int)) != 0)[0]
    if zc.size < 4:
        # fallback: simple circular shift surrogate
        rng = np.random.default_rng(11)
        null=[]
        for _ in range(n_perm):
            s = int(rng.integers(1, len(amp_fast)-1))
            null.append((phase_slow, np.r_[amp_fast[-s:], amp_fast[:-s]]))
        return null
    segs = [(zc[i], zc[i+1]) for i in range(len(zc)-1)]
    rng = np.random.default_rng(11)
    null=[]
    for _ in range(n_perm):
        af = amp_fast.copy()
        for (i0,i1) in segs:
            L = i1 - i0
            if L <= 3: continue
            sh = int(rng.integers(0, L))
            af[i0:i1] = np.r_[af[i0:i1][-sh:], af[i0:i1][:-sh]]
        null.append((phase_slow, af))
    return null

# Directionality (CFDC-like ΔR²)

def directionality_phase_to_amp(phi1, amp_m, fs, lags=np.arange(0.0, 0.51, 0.02)):
    s = np.sin(phi1); c = np.cos(phi1)
    A = (amp_m - np.nanmean(amp_m)) / (np.nanstd(amp_m) + 1e-12)
    out = []
    for tau in lags:
        shift = int(round(tau * fs))
        if shift <= 0: y = A
        else:
            y = A[shift:]
        xA  = A[:len(y)]
        xs  = s[:len(y)]; xc = c[:len(y)]
        # baseline: y ~ xA
        X0 = np.column_stack([np.ones(len(y)), xA])
        # full: y ~ xA + sin + cos
        X1 = np.column_stack([np.ones(len(y)), xA, xs, xc])
        # solve by least squares
        b0, *_ = np.linalg.lstsq(X0, y, rcond=None)
        b1, *_ = np.linalg.lstsq(X1, y, rcond=None)
        y0 = X0 @ b0; y1 = X1 @ b1
        SS = np.sum((y - np.mean(y))**2)
        R2_0 = 1 - np.sum((y - y0)**2)/(SS + 1e-12)
        R2_1 = 1 - np.sum((y - y1)**2)/(SS + 1e-12)
        out.append(R2_1 - R2_0)
    return np.asarray(out), lags

def directionality_amp_to_phase(amp_m, phi1, fs, lags=np.arange(0.0, 0.51, 0.02)):
    # Predict sin(phi1_{t+tau}) from [sin(phi1_t), cos(phi1_t), A_m(t)]
    s = np.sin(phi1); c = np.cos(phi1)
    A = (amp_m - np.nanmean(amp_m)) / (np.nanstd(amp_m) + 1e-12)
    out = []
    for tau in lags:
        shift = int(round(tau * fs))
        if shift <= 0:
            y = s
            s0, c0, a0 = s, c, A
        else:
            y = s[shift:]
            s0, c0, a0 = s[:len(y)], c[:len(y)], A[:len(y)]
        # baseline: y ~ s0 + c0
        X0 = np.column_stack([np.ones(len(y)), s0, c0])
        # full: y ~ s0 + c0 + a0
        X1 = np.column_stack([np.ones(len(y)), s0, c0, a0])
        b0, *_ = np.linalg.lstsq(X0, y, rcond=None)
        b1, *_ = np.linalg.lstsq(X1, y, rcond=None)
        y0 = X0 @ b0; y1 = X1 @ b1
        SS = np.sum((y - np.mean(y))**2)
        R2_0 = 1 - np.sum((y - y0)**2)/(SS + 1e-12)
        R2_1 = 1 - np.sum((y - y1)**2)/(SS + 1e-12)
        out.append(R2_1 - R2_0)
    return np.asarray(out), lags

# ---------------------------- main orchestrator ----------------------------

def analyze_cfc_harmonics(
    RECORDS,
    eeg_channel: str,
    windows: dict,    # {'baseline':[(t0,t1),...], 'ignition':[(..)], ...}
    time_col='Timestamp',
    fundamental=7.83,
    harmonics=(14.3,20.8,27.3,33.8,40.3,46.8,53.3,59.8),
    half_bw=0.6,
    out_dir='exports_cfc_harm', show=True
):
    """Compute CF-PLV (1:m), PAC per order (MVL & MI with within-cycle surrogates),
    and directionality ΔR² curves per harmonic. Saves PNGs + CSV summary.
    """
    ensure_dir(out_dir)
    ensure_timestamp_column(RECORDS, time_col=time_col)
    fs = infer_fs(RECORDS, time_col)

    x = get_series(RECORDS, eeg_channel)
    N = len(x)

    # Precompute phases & amplitudes for all needed bands
    phi1 = phase_at(x, fs, fundamental, half_bw)
    A = {}
    PHI = {}
    for fm in harmonics:
        if fm > min(60.0, 0.999*0.5*fs):
            continue
        PHI[fm] = phase_at(x, fs, fm, half_bw)
        A[fm]   = amp_envelope(x, fs, fm, half_bw)

    # Windows in samples
    W = {k: windows_to_samples(v, fs, N) for k,v in (windows or {}).items()}

    # Summary rows
    rows=[]

    # ---------- CF-PLV ----------
    for fm in PHI:
        m = int(round(fm / fundamental))
        for wname, segs in W.items():
            plv, p = cf_plv(phi1, PHI[fm], m, segs)
            rows.append({'metric':'CF-PLV', 'window':wname, 'order':m, 'f_hz':fm, 'value':plv, 'p_value':p})
        # Bar plot across windows for this fm
        vals=[]; labels=[]; errs=[]
        for wname, segs in W.items():
            plv, p = cf_plv(phi1, PHI[fm], m, segs)
            vals.append(plv); labels.append(wname); errs.append(0)
        fig, ax = plt.subplots(figsize=(6,3.0))
        ax.bar(np.arange(len(vals)), vals, width=0.6)
        ax.set_xticks(np.arange(len(vals))); ax.set_xticklabels(labels)
        ax.set_ylim(0,1); ax.set_ylabel('CF-PLV (1:m)'); ax.set_title(f'CF-PLV  φ1↔φ{m}  ({fundamental:.2f}↔{fm:.2f} Hz)')
        ax.grid(True, axis='y', alpha=0.25, linestyle=':')
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, f'cfplv_m{m}_{fm:.2f}Hz.png'), dpi=160)
        if show: plt.show(); plt.close()

    # ---------- PAC per order (MVL & MI) ----------
    for fm in A:
        m = int(round(fm / fundamental))
        for wname, segs in W.items():
            phi_w = concat_segments(phi1, segs)
            amp_w = concat_segments(A[fm], segs)
            if len(phi_w) < 200 or len(amp_w) < 200:
                rows.append({'metric':'PAC-MVL', 'window':wname, 'order':m, 'f_hz':fm, 'value':np.nan, 'p_value':np.nan})
                rows.append({'metric':'PAC-MI',  'window':wname, 'order':m, 'f_hz':fm, 'value':np.nan, 'p_value':np.nan})
                continue
            mv = mvl(phi_w, amp_w); mi = tort_mi(phi_w, amp_w)
            # within-cycle surrogates
            null = pac_within_cycle_surrogate(phi_w, amp_w, fs)
            mv_null = []; mi_null=[]
            for (ph_s, af_s) in null:
                mv_null.append(mvl(ph_s, af_s)); mi_null.append(tort_mi(ph_s, af_s))
            p_mv = float((np.sum(np.asarray(mv_null) >= mv) + 1) / (len(mv_null)+1))
            p_mi = float((np.sum(np.asarray(mi_null) >= mi) + 1) / (len(mi_null)+1))
            rows.append({'metric':'PAC-MVL', 'window':wname, 'order':m, 'f_hz':fm, 'value':mv, 'p_value':p_mv})
            rows.append({'metric':'PAC-MI',  'window':wname, 'order':m, 'f_hz':fm, 'value':mi, 'p_value':p_mi})
        # plot
        S = [r for r in rows if r['metric']=='PAC-MI' and r['f_hz']==fm]
        labs=[r['window'] for r in S]; vals=[r['value'] for r in S]
        fig, ax = plt.subplots(figsize=(6,3.0))
        ax.bar(np.arange(len(vals)), vals, width=0.6)
        ax.set_xticks(np.arange(len(vals))); ax.set_xticklabels(labs)
        ax.set_ylabel('PAC (MI)'); ax.set_title(f'PAC MI: phase {fundamental:.2f} Hz → amp {fm:.2f} Hz')
        ax.grid(True, axis='y', alpha=0.25, linestyle=':')
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, f'pac_mi_{fm:.2f}Hz.png'), dpi=160)
        if show: plt.show(); plt.close()

    # ---------- Directionality (ΔR² curves) ----------
    for fm in A:
        m = int(round(fm / fundamental))
        for wname, segs in W.items():
            phi_w = concat_segments(phi1, segs)
            amp_w = concat_segments(A[fm], segs)
            if len(phi_w) < 400 or len(amp_w) < 400:
                rows.append({'metric':'DIR-P2A-peak', 'window':wname, 'order':m, 'f_hz':fm, 'value':np.nan, 'p_value':np.nan})
                rows.append({'metric':'DIR-A2P-peak', 'window':wname, 'order':m, 'f_hz':fm, 'value':np.nan, 'p_value':np.nan})
                continue
            dP2A, lags = directionality_phase_to_amp(phi_w, amp_w, fs)
            dA2P, _    = directionality_amp_to_phase(amp_w, phi_w, fs)
            # surrogate: circular shift amplitude relative to phase
            rng = np.random.default_rng(19)
            nullP=[]; nullA=[]
            for _ in range(200):
                s = int(rng.integers(1, len(amp_w)-1))
                amp_s = np.r_[amp_w[-s:], amp_w[:-s]]
                dp, _ = directionality_phase_to_amp(phi_w, amp_s, fs)
                da, _ = directionality_amp_to_phase(amp_s, phi_w, fs)
                nullP.append(np.nanmax(dp)); nullA.append(np.nanmax(da))
            peakP = float(np.nanmax(dP2A)); peakA = float(np.nanmax(dA2P))
            pP = float((np.sum(np.asarray(nullP) >= peakP) + 1) / (len(nullP)+1))
            pA = float((np.sum(np.asarray(nullA) >= peakA) + 1) / (len(nullA)+1))
            rows.append({'metric':'DIR-P2A-peak', 'window':wname, 'order':m, 'f_hz':fm, 'value':peakP, 'p_value':pP})
            rows.append({'metric':'DIR-A2P-peak', 'window':wname, 'order':m, 'f_hz':fm, 'value':peakA, 'p_value':pA})
            # plot lag curves
            fig, ax = plt.subplots(figsize=(6.8,3.0))
            ax.plot(lags, dP2A, lw=1.6, label='phase→amp ΔR²')
            ax.plot(lags, dA2P, lw=1.2, ls='--', label='amp→phase ΔR²')
            ax.set_xlabel('Lag τ (s)'); ax.set_ylabel('ΔR²'); ax.set_title(f'Directionality (φ1 ↔ A_{m})  {fundamental:.2f}↔{fm:.2f} Hz')
            ax.legend(); ax.grid(True, alpha=0.25, linestyle=':')
            plt.tight_layout(); plt.savefig(os.path.join(out_dir, f'dir_m{m}_{fm:.2f}Hz.png'), dpi=160)
            if show: plt.show(); plt.close()

    # Save summary
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(out_dir, 'cfc_harmonics_summary.csv'), index=False)
    return summary

# ---------------------------- Example usage ----------------------------
# windows = {
#   'baseline': [(0, 290)],
#   'ignition': [(290, 310), (580, 600)],
#   'rebound':  [(325, 580)]
# }
# summary = analyze_cfc_harmonics(
#     RECORDS,
#     eeg_channel='EEG.O1',  # or 'EEG.VIRT' if you built a virtual posterior ROI
#     windows=windows,
#     fundamental=7.83,
#     harmonics=(14.3,20.8,27.3,33.8,40.3,46.8,53.3,59.8),
#     half_bw=0.6,
#     out_dir='exports_cfc_harm', show=True
# )
