"""
Lead/Lag Quantification Among SR Families (Temporal Dynamics)
=============================================================
Implements simple graphs and validation tests for:
  1) Envelope cross‑correlation lag (minute‑scale 0.003–0.03 Hz “breathing”).
  2) Phase‑lead probability between slow envelopes.
  3) Consensus ordering score across families: SubH(2–5) → 7.83 → (14–34) → (40–60).

Per window (pre‑ignition, ignition, rebound) and per band/family, outputs:
  • τ̂_env (s) with block‑bootstrap 95% CI.
  • P_lead with block‑bootstrap 95% CI.
  • Summary CSV + quick bar plots.

Usage example at bottom.
Dependencies: numpy, scipy, matplotlib, pandas.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal

# ------------------ basic helpers (reuse yours if present) ------------------

def ensure_dir(d):
    if d:
        os.makedirs(d, exist_ok=True)
    return d

def ensure_timestamp_column(df, time_col='Timestamp', default_fs=128.0):
    if time_col in df.columns:
        s = df[time_col]
        # datetime → seconds
        if np.issubdtype(s.dtype, np.datetime64) or 'datetime' in str(s.dtype).lower():
            tsec=(pd.to_datetime(s)-pd.to_datetime(s).iloc[0]).dt.total_seconds().astype(float)
            df[time_col] = tsec.values
            return time_col
        # numeric
        sn = pd.to_numeric(s, errors='coerce').astype(float)
        if sn.notna().sum()>max(50,0.5*len(df)):
            sn = sn - np.nanmin(sn[np.isfinite(sn)])
            df[time_col] = sn.values
            return time_col
    # fallback: synth time
    df[time_col] = np.arange(len(df), dtype=float)/default_fs
    return time_col

def infer_fs(df, time_col='Timestamp'):
    t = np.asarray(df[time_col].values, float)
    dt = np.diff(t); dt = dt[(dt>0)&np.isfinite(dt)]
    if dt.size==0: raise ValueError('Cannot infer fs from time column.')
    return float(1.0/np.median(dt))

# filtering

def _butter_bandpass(x, fs, lo, hi, order=4):
    ny=0.5*fs
    lo=max(1e-6,min(lo,0.99*ny)); hi=max(lo+1e-6,min(hi,0.999*ny))
    b,a=signal.butter(order,[lo/ny, hi/ny], btype='band')
    return signal.filtfilt(b,a,x)

def _butter_lowpass(x, fs, hi, order=4):
    ny=0.5*fs; hi=max(1e-6, min(hi, 0.999*ny))
    b,a=signal.butter(order, hi/ny, btype='low')
    return signal.filtfilt(b,a,x)

# narrowband analytic + slow envelope & its phase

def band_envelope_and_slow_phase(x, fs, f0, half=0.6, slow_band=(0.003,0.03)):
    """Return: slow_env (band‑passed envelope in slow_band), slow_phase (Hilbert angle)."""
    xnb = _butter_bandpass(x, fs, f0-half, f0+half)
    env = np.abs(signal.hilbert(xnb))           # amplitude envelope
    # band‑pass the envelope in minute‑scale band
    slo = _butter_bandpass(env, fs, slow_band[0], slow_band[1])
    z = signal.hilbert(slo)
    return slo, np.angle(z)

# --------------- windowing & bootstrap ---------------

def windows_to_samples(wins, fs, N):
    segs = []
    for (a,b) in wins:
        i0,i1 = int(round(a*fs)), int(round(b*fs))
        i0=max(0,i0); i1=min(N,i1)
        if i1>i0: segs.append((i0,i1))
    return segs

# block bootstrap (circular)

def block_bootstrap_series(x, block_N, out_N, rng):
    idx=[]; filled=0
    while filled<out_N:
        start=int(rng.integers(0, len(x)))
        end=start+block_N
        if end<=len(x): idx.extend(range(start,end))
        else: idx.extend(list(range(start,len(x)))+list(range(0,end-len(x))))
        filled+=block_N
    idx=idx[:out_N]
    return x[idx]

# --------------- core metrics ---------------



def xcorr_lag(env_f, env_ref, fs, max_lag_s=30.0):
    # drop any NaNs pairwise
    env_f = np.asarray(env_f, float)
    env_ref = np.asarray(env_ref, float)
    m = np.isfinite(env_f) & np.isfinite(env_ref)
    env_f = env_f[m]; env_ref = env_ref[m]
    # bail if too short or flat
    if env_f.size < 8 or env_ref.size < 8:
        return np.nan, np.nan, np.array([0.0]), np.array([np.nan])
    if np.nanstd(env_f) < 1e-12 or np.nanstd(env_ref) < 1e-12:
        return np.nan, np.nan, np.array([0.0]), np.array([np.nan])

    # z-score (robust to constant signals)
    x = (env_f - np.nanmean(env_f)) / (np.nanstd(env_f) + 1e-12)
    y = (env_ref - np.nanmean(env_ref)) / (np.nanstd(env_ref) + 1e-12)

    c = signal.correlate(x, y, mode='full', method='auto')
    lags = signal.correlation_lags(len(x), len(y), mode='full')

    # restrict to ±max_lag
    L = int(round(max_lag_s * fs))
    sel = (lags >= -L) & (lags <= L)
    c = c[sel]; lags = lags[sel]

    if not np.any(np.isfinite(c)):
        return np.nan, np.nan, lags/fs, c

    # argmax ignoring NaNs
    k = int(np.nanargmax(np.nan_to_num(c, nan=-np.inf)))
    tau_s = float(lags[k]) / fs           # +τ => env_f leads env_ref
    rmax  = float(c[k]) / float(len(x))   # simple scale-stable norm
    return tau_s, rmax, lags/fs, c




def lag_ci_bootstrap(env_f, env_ref, fs, max_lag_s=30.0, n_boot=500, block_sec=10.0, seed=23):
    rng = np.random.default_rng(seed)
    N = min(len(env_f), len(env_ref))
    bN = max(8, int(round(block_sec*fs)))
    taus = []
    for _ in range(int(n_boot)):
        xf = block_bootstrap_series(env_f, bN, N, rng)
        xr = block_bootstrap_series(env_ref, bN, N, rng)
        tau,_r, _L,_C = xcorr_lag(xf, xr, fs, max_lag_s=max_lag_s)
        taus.append(tau)
    lo, hi = np.nanpercentile(taus, [2.5, 97.5])
    return float(lo), float(hi)


def phase_lead_probability(phi_f, phi_ref, n_boot=1000, block_len=200, seed=7):
    """Return P_lead = Pr(Δϕ ∈ (0,π)) with block‑bootstrap 95% CI.
    Δϕ = wrap(φ_f − φ_ref) to (−π,π].
    """
    rng = np.random.default_rng(seed)
    dphi = np.angle(np.exp(1j*(phi_f - phi_ref)))  # wrapped
    p = float(np.nanmean((dphi>0) & (dphi<np.pi)))
    N = len(dphi)
    B = max(20, int(block_len))
    ps=[]
    for _ in range(int(n_boot)):
        idx=[]; filled=0
        while filled<N:
            start=int(rng.integers(0,N))
            end=start+B
            if end<=N: idx.extend(range(start,end))
            else: idx.extend(list(range(start,N))+list(range(0,end-N)))
            filled+=B
        idx=idx[:N]
        d = dphi[idx]
        ps.append(float(np.nanmean((d>0) & (d<np.pi))))
    lo, hi = np.nanpercentile(ps, [2.5,97.5])
    return p, float(lo), float(hi)

# --------------- main analysis ---------------

def analyze_lead_lag_temporal(
    RECORDS,
    eeg_channel: str,
    windows: dict,  # {'pre':[(t0,t1),...], 'ignition':[(..)], 'rebound':[(..)]}
    time_col='Timestamp',
    # frequency families
    fundamental=7.83,
    family_subh=(3.915, 2.61, 1.9575, 1.566),      # SubH(2–5)
    family_low=(14.3, 20.8, 27.3, 33.8),           # (14–34)
    family_high=(40.3, 46.8, 53.3, 59.8),          # (40–60)
    half_bw=0.6, slow_band=(0.003,0.03),
    max_lag_s=30.0, n_boot=500, block_sec=10.0,
    coincident_tol_s=2.0,
    out_dir='exports_leadlag', show=True
):
    """Compute envelope xcorr lags & envelope‑phase lead probabilities per window and family.
    Saves summary CSV and simple bar plots with 95% CIs. Returns summary DataFrame.
    """
    ensure_dir(out_dir)
    ensure_timestamp_column(RECORDS, time_col=time_col)
    fs = infer_fs(RECORDS, time_col)

    x = RECORDS[eeg_channel].astype(float).values if eeg_channel in RECORDS.columns else pd.to_numeric(RECORDS['EEG.'+eeg_channel], errors='coerce').fillna(0.0).values.astype(float)
    N = len(x)

    # Precompute slow envelope + phase for fundamental and all freqs
    def env_phase_for(f):
        slo, phi = band_envelope_and_slow_phase(x, fs, f, half=half_bw, slow_band=slow_band)
        return slo, phi

    env_phi = {}
    all_freqs = {'fund':(fundamental,), 'subh':tuple(family_subh), 'low':tuple(family_low), 'high':tuple(family_high)}
    for fam, freqs in all_freqs.items():
        for f in freqs:
            env_phi[f] = env_phase_for(f)

    env_fund, phi_fund = env_phi[fundamental]

    # helper to slice windows and compute stats
    def analyze_window(win_name, segs):
        rows=[]
        # family containers for consensus ordering (use median lag across members)
        fam_lags = {'subh':[], 'fund':[], 'low':[], 'high':[]}
        for fam, freqs in [('subh', family_subh), ('fund',(fundamental,)), ('low', family_low), ('high', family_high)]:
            for f in freqs:
                env_f, phi_f = env_phi[f]
                # concatenate window segments
                idxs = windows_to_samples(segs, fs, N)
                ef = np.concatenate([env_f[i0:i1] for (i0,i1) in idxs]) if idxs else np.array([])
                er = np.concatenate([env_fund[i0:i1] for (i0,i1) in idxs]) if idxs else np.array([])
                pf = np.concatenate([phi_f[i0:i1] for (i0,i1) in idxs]) if idxs else np.array([])
                pr = np.concatenate([phi_fund[i0:i1] for (i0,i1) in idxs]) if idxs else np.array([])
                if ef.size<10 or er.size<10:
                    continue
                tau, rmax, _lags,_c = xcorr_lag(ef, er, fs, max_lag_s=max_lag_s)
                lo, hi = lag_ci_bootstrap(ef, er, fs, max_lag_s=max_lag_s, n_boot=n_boot, block_sec=block_sec)
                P, Plo, Phi = phase_lead_probability(pf, pr, n_boot=n_boot, block_len=int(block_sec*fs))
                fam_lags[fam].append(tau)
                rows.append({'window':win_name, 'family':fam, 'f0':f, 'tau_env_s':tau, 'tau_lo':lo, 'tau_hi':hi,
                             'P_lead':P, 'P_lead_lo':Plo, 'P_lead_hi':Phi, 'rmax':rmax,
                             'n_samples': int(ef.size)})
        # consensus ordering score
        score, n_tests = consensus_order_score(fam_lags, tol=coincident_tol_s)
        rows.append({'window':win_name, 'family':'CONSENSUS', 'f0':np.nan, 'tau_env_s':np.nan,
                     'tau_lo':np.nan, 'tau_hi':np.nan, 'P_lead':np.nan,
                     'P_lead_lo':np.nan, 'P_lead_hi':np.nan, 'rmax':np.nan,
                     'n_samples':int(sum(len(v) for v in fam_lags.values())),
                     'consensus_score_pct': 100.0*score, 'n_tests': n_tests})
        return rows

    def consensus_order_score(fam_lags, tol=2.0):
        """Return fraction of pairwise constraints satisfied for ordering:
        SubH → Fund → Low → High. We use median lag per family; Fund≈Low treated as coincident
        if |median(Fund) − median(Low)| ≤ tol.
        Positive τ means family leads FUND; negative means lags (by our xcorr sign choice).
        Constraints:
          median(SubH) >= median(Fund) + tol  (SubH leads)
          |median(Low) − median(Fund)| ≤ tol  (coincident)
          median(High) ≤ median(Fund) − tol  (High lags)
          median(Low) ≥ median(High) + tol   (Low ahead of High)
          median(SubH) ≥ median(Low) + tol   (SubH ahead of Low)
          median(SubH) ≥ median(High) + tol  (SubH ahead of High)
        """
        import math
        # median per family; if empty, set nan
        med = {k: (np.nan if len(v)==0 else float(np.nanmedian(v))) for k,v in fam_lags.items()}
        tests=0; ok=0
        def inc(test):
            nonlocal tests, ok
            tests += 1
            if test: ok += 1
        if not math.isnan(med['subh']) and not math.isnan(med['fund']):
            inc(med['subh'] >= med['fund'] + tol)
        if not math.isnan(med['low']) and not math.isnan(med['fund']):
            inc(abs(med['low'] - med['fund']) <= tol)
        if not math.isnan(med['high']) and not math.isnan(med['fund']):
            inc(med['high'] <= med['fund'] - tol)
        if not math.isnan(med['low']) and not math.isnan(med['high']):
            inc(med['low'] >= med['high'] + tol)
        if not math.isnan(med['subh']) and not math.isnan(med['low']):
            inc(med['subh'] >= med['low'] + tol)
        if not math.isnan(med['subh']) and not math.isnan(med['high']):
            inc(med['subh'] >= med['high'] + tol)
        return (ok / max(1, tests)), tests

    # run per window
    all_rows=[]
    for wname, segs in windows.items():
        if not segs: continue
        all_rows.extend(analyze_window(wname, segs))

    summary = pd.DataFrame(all_rows)
    csv_path = os.path.join(out_dir, 'leadlag_summary.csv')
    summary.to_csv(csv_path, index=False)

    # ------------------ simple graphs ------------------
    # Bar: τ̂_env by family (per window) with CI whiskers
    fam_order = ['subh','fund','low','high']
    for wname in sorted(set(summary['window'])):
        S = summary[(summary['window']==wname) & (summary['family'].isin(fam_order))]
        if S.empty: continue
        # aggregate per family using median across f0s
        agg = S.groupby('family').agg({'tau_env_s':'median','tau_lo':'median','tau_hi':'median'}).reindex(fam_order)
        x = np.arange(len(agg)); y = agg['tau_env_s'].values
        err_lo = y - agg['tau_lo'].values; err_hi = agg['tau_hi'].values - y
        fig, ax = plt.subplots(figsize=(8,3.2))
        ax.bar(x, y, yerr=[err_lo, err_hi], width=0.6, capsize=3)
        ax.axhline(0, color='k', lw=0.7, alpha=0.6)
        ax.set_xticks(x); ax.set_xticklabels(['SubH(2–5)','7.83','14–34','40–60'])
        ax.set_ylabel('τ̂_env (s)'); ax.set_title(f'Envelope lag by family — {wname}')
        ax.grid(True, axis='y', alpha=0.25, linestyle=':')
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, f'lag_bars_{wname}.png'), dpi=160)
        if show: plt.show(); plt.close()

    # Bar: P_lead by family (per window)
    for wname in sorted(set(summary['window'])):
        S = summary[(summary['window']==wname) & (summary['family'].isin(fam_order))]
        if S.empty: continue
        agg = S.groupby('family').agg({'P_lead':'median','P_lead_lo':'median','P_lead_hi':'median'}).reindex(fam_order)
        x = np.arange(len(agg)); y = agg['P_lead'].values
        err_lo = y - agg['P_lead_lo'].values; err_hi = agg['P_lead_hi'].values - y
        fig, ax = plt.subplots(figsize=(8,3.2))
        ax.bar(x, y, yerr=[err_lo, err_hi], width=0.6, capsize=3)
        ax.set_ylim(0,1)
        ax.axhline(0.5, color='k', lw=0.7, ls='--', alpha=0.5)
        ax.set_xticks(x); ax.set_xticklabels(['SubH(2–5)','7.83','14–34','40–60'])
        ax.set_ylabel('P_lead'); ax.set_title(f'Phase‑lead probability (slow envelope) — {wname}')
        ax.grid(True, axis='y', alpha=0.25, linestyle=':')
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, f'plead_bars_{wname}.png'), dpi=160)
        if show: plt.show(); plt.close()

    # Consensus score table saved separately
    CONS = summary[summary['family']=='CONSENSUS'][['window','consensus_score_pct','n_tests']]
    if not CONS.empty:
        CONS.to_csv(os.path.join(out_dir, 'consensus_scores.csv'), index=False)

    return summary

# ------------------ Example usage ------------------
# windows = {
#   'pre': [(0, 290)],
#   'ignition': [(290, 310), (580, 600)],
#   'rebound': [(325, 580)]
# }
# summary = analyze_lead_lag_temporal(
#     RECORDS,
#     eeg_channel='EEG.O1',           # or 'EEG.VIRT' if you built a virtual posterior
#     windows=windows,
#     fundamental=7.83,
#     family_subh=(3.915, 2.61, 1.9575, 1.566),
#     family_low=(14.3, 20.8, 27.3, 33.8),
#     family_high=(40.3, 46.8, 53.3, 59.8),
#     half_bw=0.6, slow_band=(0.003,0.03),
#     max_lag_s=30.0, n_boot=500, block_sec=10.0,
#     coincident_tol_s=2.0,
#     out_dir='exports_leadlag', show=True
# )


def virtual_eeg_snr_weighted(RECORDS, channels, fs, f0, half=0.6, time_col='Timestamp'):
    """Return (v_sig, weights) where v_sig is SNR‑weighted sum of channels normalized to unit gain."""
    X = []
    for ch in channels:
        if ch in RECORDS.columns:
            x = pd.to_numeric(RECORDS[ch], errors='coerce').fillna(0.0).values.astype(float)
        elif ('EEG.'+ch) in RECORDS.columns:
            x = pd.to_numeric(RECORDS['EEG.'+ch], errors='coerce').fillna(0.0).values.astype(float)
        else:
            raise ValueError(f"{ch} not in dataframe")
    X.append(x)
    X = np.vstack(X) # shape (C, N)
    # SNR per channel
    snrs = np.array([snr_at_f0(x, fs, f0, half=half) for x in X])
    w = snrs / (np.sum(snrs) + 1e-12)
    v = np.dot(w, X) # (N,)
    return v, w

def snr_at_f0(x, fs, f0, half=0.6, flank=2.0):
    """SNR = band power at f0±half / average flank power at [f0±(half+δ) .. f0±(half+δ+flank)]."""
    sig = _bandpass(x, fs, f0-half, f0+half)
    p_sig = np.mean(sig**2)
    # two flanks: below and above
    lo1, lo2 = max(0.01, f0-(half+flank+0.5)), f0-(half+0.5)
    hi1, hi2 = f0+(half+0.5), f0+(half+flank+0.5)
    if lo2>lo1:
        fl = _bandpass(x, fs, lo1, lo2); p_fl = np.mean(fl**2)
    else:
        p_fl = 0.0
    if hi2>hi1:
        fh = _bandpass(x, fs, hi1, hi2); p_fh = np.mean(fh**2)
    else:
        p_fh = 0.0
    p_noise = np.mean([p for p in [p_fl, p_fh] if np.isfinite(p)]) or 1e-12
    return float(p_sig / p_noise)



def _bandpass(x, fs, lo, hi, order=4):
    ny=0.5*fs; lo=max(1e-6, min(lo, 0.99*ny)); hi=max(lo+1e-6, min(hi, 0.999*ny))
    b,a=signal.butter(order, [lo/ny, hi/ny], btype='band');
    return signal.filtfilt(b,a,x)
