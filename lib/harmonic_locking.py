"""
Harmonic locking metrics for SR fundamentals & harmonics (0.1–60 Hz)
====================================================================
Implements:
  • H‑PLI_k(t): |< exp(i[φ_k^EEG(t) − φ_k^SR(t)]) >_{t∈W}|
  • XH‑PLI_m(t): |< exp(i[φ_m^EEG(t) − m φ_1^EEG(t)]) >_{t∈W}|
  • SubH‑PLI_n(t): |< exp(i[n φ_{1/n}^EEG(t) − φ_1^EEG(t)]) >_{t∈W}|
  • HCS(t) = Σ_k w_k · H‑PLI_k(t), default w_k=1/k

Also provides:
  • Simple surrogate tests with smooth‑aware nulls (circular time shift of phases)
  • Block‑bootstrap CIs for time‑average metrics
  • Faceted time‑series plots (PLI vs null) and bar charts with CIs
  • Summary CSV of per‑order metrics and HCS

Dependencies: numpy, scipy, matplotlib (and pandas for summaries)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# --------------------------- helpers ---------------------------

def ensure_dir(d):
    if d:
        os.makedirs(d, exist_ok=True)
    return d

def ensure_timestamp_column(df, time_col='Timestamp', default_fs=128.0):
    import pandas as pd
    if time_col in df.columns:
        s = df[time_col]
        if np.issubdtype(s.dtype, np.datetime64) or 'datetime' in str(s.dtype).lower():
            tsec=(pd.to_datetime(s)-pd.to_datetime(s).iloc[0]).dt.total_seconds().astype(float)
            df[time_col] = tsec.values
            return time_col
        sn = pd.to_numeric(s, errors='coerce').astype(float)
        if sn.notna().sum()>max(50,0.5*len(df)):
            sn = sn - np.nanmin(sn[np.isfinite(sn)])
            df[time_col] = sn.values
            return time_col
    df[time_col] = np.arange(len(df), dtype=float)/default_fs
    return time_col

def infer_fs(df, time_col='Timestamp'):
    t = np.asarray(df[time_col].values, float)
    dt = np.diff(t); dt = dt[(dt>0)&np.isfinite(dt)]
    if dt.size==0: raise ValueError('Cannot infer fs from time column.')
    return float(1.0/np.median(dt))

def get_series(df, name):
    import pandas as pd
    if name in df.columns:
        return pd.to_numeric(df[name], errors='coerce').fillna(0.0).values.astype(float)
    alt='EEG.'+name
    if alt in df.columns:
        return pd.to_numeric(df[alt], errors='coerce').fillna(0.0).values.astype(float)
    raise ValueError(f'{name} not found in dataframe columns.')

# bandpass + Hilbert phase

def bandpass(x, fs, f1, f2, order=4):
    ny = 0.5*fs
    f1 = max(1e-6, min(f1, 0.99*ny))
    f2 = max(f1+1e-6, min(f2, 0.999*ny))
    b,a = signal.butter(order, [f1/ny, f2/ny], btype='band')
    return signal.filtfilt(b,a,x)

def phase_series(x, fs, f0, half):
    xb = bandpass(x, fs, float(f0)-float(half), float(f0)+float(half))
    z  = signal.hilbert(xb)
    return np.angle(z)  # in radians, wrapped

# sliding windows

def sliding_centers(N, fs, win_sec, step_sec):
    win = int(round(win_sec*fs)); step=int(round(step_sec*fs))
    if win < 8: raise ValueError('win_sec too small')
    return np.arange(win//2, N - win//2, step, dtype=int)

# smoothing (optional visual only)

def _auto_savgol(y, max_window=31):
    y = np.asarray(y, float)
    n = y.size
    if n < 7: return y
    try:
        from scipy.signal import savgol_filter
        w = max(5, min(max_window, int(round(n/15))))
        if w % 2 == 0: w += 1
        if w >= n: w = n-1 if (n % 2 == 0) else n
        if w < 5: return y
        return savgol_filter(y, w, polyorder=2, mode='interp')
    except Exception:
        w = max(5, min(max_window, int(round(n/15))))
        if w < 2: return y
        return np.convolve(np.nan_to_num(y), np.ones(w)/w, mode='same')

# block bootstrap for CI of means

def block_bootstrap_ci(x, n_boot=1000, alpha=0.05, block_len=None, seed=13):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    N = x.size
    if N == 0: return (np.nan, np.nan)
    if block_len is None:
        block_len = max(5, int(round(N/20)))
    B = int(block_len)
    means = []
    for _ in range(int(n_boot)):
        idx = []
        filled = 0
        while filled < N:
            start = int(rng.integers(0, N))
            end = start + B
            if end <= N:
                idx.extend(range(start, end))
            else:
                idx.extend(list(range(start, N)) + list(range(0, end - N)))
            filled += B
        idx = np.asarray(idx[:N], int)
        means.append(float(np.nanmean(x[idx])))
    lo, hi = np.nanpercentile(means, [100*alpha/2, 100*(1-alpha/2)])
    return float(lo), float(hi)

# surrogate builder for PLI curves (smooth‑aware)

def pli_surrogates(phi_a, phi_b, centers, win, step, n_perm=200, smoother=None, seed=7):
    rng = np.random.default_rng(seed)
    N = len(phi_a)
    out = []
    for _ in range(int(n_perm)):
        s = int(rng.integers(win, N-1))  # shift at least one window
        phi_b_sh = np.r_[phi_b[-s:], phi_b[:-s]]
        vals = []
        for c in centers:
            sl = slice(c - win//2, c + win//2)
            dphi = phi_a[sl] - phi_b_sh[sl]
            vals.append(np.abs(np.mean(np.exp(1j*dphi))))
        v = np.asarray(vals, float)
        if smoother is not None:
            v = np.asarray(smoother(v), float)
        out.append(v)
    return np.asarray(out)  # shape (n_perm, n_centers)

# --------------------------- core metrics ---------------------------

def compute_H_PLI(phi_eeg, phi_sr, fs, win_sec=8.0, step_sec=1.0, smoother=None, n_perm=200):
    win = int(round(win_sec*fs)); step=int(round(step_sec*fs))
    centers = sliding_centers(len(phi_eeg), fs, win_sec, step_sec)
    pli = []
    for c in centers:
        sl = slice(c - win//2, c + win//2)
        dphi = phi_eeg[sl] - phi_sr[sl]
        pli.append(np.abs(np.mean(np.exp(1j*dphi))))
    pli = np.asarray(pli, float)
    if smoother is not None:
        pli_plot = np.asarray(smoother(pli), float)
    else:
        pli_plot = pli
    # smooth‑aware null (pointwise 95th percentile) + mean p‑value
    sur = pli_surrogates(phi_eeg, phi_sr, centers, win, step, n_perm=n_perm, smoother=smoother)
    null95_curve = np.nanpercentile(sur, 95, axis=0)
    p_mean = float((np.sum(np.nanmean(sur, axis=1) >= np.nanmean(pli_plot)) + 1) / (n_perm + 1))
    return centers/fs, pli_plot, null95_curve, p_mean, pli


def compute_XH_PLI(phi_m_eeg, phi1_eeg, m, fs, win_sec=8.0, step_sec=1.0, smoother=None, n_perm=200):
    win = int(round(win_sec*fs)); step=int(round(step_sec*fs))
    centers = sliding_centers(len(phi_m_eeg), fs, win_sec, step_sec)
    pli = []
    for c in centers:
        sl = slice(c - win//2, c + win//2)
        dphi = phi_m_eeg[sl] - m*phi1_eeg[sl]
        pli.append(np.abs(np.mean(np.exp(1j*dphi))))
    pli = np.asarray(pli, float)
    pli_plot = np.asarray(smoother(pli), float) if smoother is not None else pli
    # surrogates: circular shift the fundamental phase
    sur = []
    rng = np.random.default_rng(11)
    N = len(phi1_eeg)
    for _ in range(int(n_perm)):
        s = int(rng.integers(win, N-1))
        phi1_sh = np.r_[phi1_eeg[-s:], phi1_eeg[:-s]]
        vals = []
        for c in centers:
            sl = slice(c - win//2, c + win//2)
            dphi = phi_m_eeg[sl] - m*phi1_sh[sl]
            vals.append(np.abs(np.mean(np.exp(1j*dphi))))
        v = np.asarray(vals, float)
        if smoother is not None: v = np.asarray(smoother(v), float)
        sur.append(v)
    sur = np.asarray(sur)
    null95_curve = np.nanpercentile(sur, 95, axis=0)
    p_mean = float((np.sum(np.nanmean(sur, axis=1) >= np.nanmean(pli_plot)) + 1) / (n_perm + 1))
    return centers/fs, pli_plot, null95_curve, p_mean, pli


def compute_SubH_PLI(phi_s_eeg, phi1_eeg, n, fs, win_sec=8.0, step_sec=1.0, smoother=None, n_perm=200):
    win = int(round(win_sec*fs)); step=int(round(step_sec*fs))
    centers = sliding_centers(len(phi_s_eeg), fs, win_sec, step_sec)
    pli = []
    for c in centers:
        sl = slice(c - win//2, c + win//2)
        dphi = n*phi_s_eeg[sl] - phi1_eeg[sl]
        pli.append(np.abs(np.mean(np.exp(1j*dphi))))
    pli = np.asarray(pli, float)
    pli_plot = np.asarray(smoother(pli), float) if smoother is not None else pli
    # surrogates: shift the subharmonic phase
    sur = []
    rng = np.random.default_rng(17)
    N = len(phi_s_eeg)
    for _ in range(int(n_perm)):
        s = int(rng.integers(win, N-1))
        phi_s_sh = np.r_[phi_s_eeg[-s:], phi_s_eeg[:-s]]
        vals = []
        for c in centers:
            sl = slice(c - win//2, c + win//2)
            dphi = n*phi_s_sh[sl] - phi1_eeg[sl]
            vals.append(np.abs(np.mean(np.exp(1j*dphi))))
        v = np.asarray(vals, float)
        if smoother is not None: v = np.asarray(smoother(v), float)
        sur.append(v)
    sur = np.asarray(sur)
    null95_curve = np.nanpercentile(sur, 95, axis=0)
    p_mean = float((np.sum(np.nanmean(sur, axis=1) >= np.nanmean(pli_plot)) + 1) / (n_perm + 1))
    return centers/fs, pli_plot, null95_curve, p_mean, pli

# --------------------------- orchestration ---------------------------

def win_for_f0(f0, cycles=8, min_win=8.0, max_win=120.0):
    if f0 <= 1e-9: return max_win
    return float(np.clip(cycles/float(f0), min_win, max_win))

def analyze_locking(RECORDS,
                    eeg_channel: str,
                    sr_channel: str,
                    fundamental=7.83,
                    harmonics=(7.83,14.3,20.8,27.3,33.8),
                    subharmonics=(3.915, 2.61, 1.9575, 1.566, 1.305, 1.11857, 0.9788, 1.2, 0.783),
                    time_col='Timestamp',
                    half_bw=0.6,
                    cycles=8, min_win=8.0, max_win=120.0,
                    step_sec=1.0,
                    n_perm=200,
                    limit_high_hz=60.0,
                    smooth=True,
                    weights_scheme='inverse_k',    # or 'equal'
                    out_dir='exports_locking',
                    show=True):
    """
    Compute H‑PLI per harmonic (EEG vs SR), cross‑order XH‑PLI_m (EEG harmonics vs EEG fundamental),
    and SubH‑PLI_n for provided subharmonics. Plot simple graphs and save a summary CSV.
    """
    import pandas as pd
    ensure_timestamp_column(RECORDS, time_col=time_col)
    fs = infer_fs(RECORDS, time_col)
    ensure_dir(out_dir)

    x_eeg = get_series(RECORDS, eeg_channel).astype(float)
    x_sr  = get_series(RECORDS, sr_channel).astype(float)

    harmonics = tuple([f for f in harmonics if f <= float(limit_high_hz)+1e-9])

    smoother = _auto_savgol if smooth else None

    rows = []
    traces_H = []

    # Fundamental phase (EEG) for cross‑order comparisons
    phi1_eeg = phase_series(x_eeg, fs, fundamental, half_bw)

    # ----- H‑PLI per harmonic -----
    for fk in harmonics:
        win_sec = win_for_f0(fk, cycles=cycles, min_win=min_win, max_win=max_win)
        phi_k_eeg = phase_series(x_eeg, fs, fk, half_bw)
        phi_k_sr  = phase_series(x_sr,  fs, fk, half_bw)
        t, pli_curve, null95_curve, p_mean, pli_raw = compute_H_PLI(phi_k_eeg, phi_k_sr, fs,
                                                                    win_sec=win_sec, step_sec=step_sec,
                                                                    smoother=smoother, n_perm=n_perm)
        # CI for mean (block bootstrap)
        ci_lo, ci_hi = block_bootstrap_ci(pli_curve, n_boot=1000, alpha=0.05)
        mean_pli = float(np.nanmean(pli_curve))
        med_pli  = float(np.nanmedian(pli_curve))
        cover = float(100.0*np.nanmean((pli_curve > null95_curve).astype(float)))
        # store
        traces_H.append((fk, t, pli_curve, null95_curve, win_sec))
        k_order = max(1, int(round(fk / fundamental)))
        rows.append({'metric':'H-PLI', 'order':k_order, 'f0':fk,
                     'mean':mean_pli, 'median':med_pli, 'ci_lo':ci_lo, 'ci_hi':ci_hi,
                     'p_mean':p_mean, 'coverage_pct':cover, 'win_sec':win_sec})

    # ----- XH‑PLI vs fundamental -----
    traces_XH = []
    for fk in harmonics:
        m = max(1, int(round(fk / fundamental)))
        if m == 1:  # skip trivial
            continue
        win_sec = win_for_f0(fk, cycles=cycles, min_win=min_win, max_win=max_win)
        phi_m_eeg = phase_series(x_eeg, fs, fk, half_bw)
        t, pli_curve, null95_curve, p_mean, pli_raw = compute_XH_PLI(phi_m_eeg, phi1_eeg, m, fs,
                                                                     win_sec=win_sec, step_sec=step_sec,
                                                                     smoother=smoother, n_perm=n_perm)
        ci_lo, ci_hi = block_bootstrap_ci(pli_curve, n_boot=1000, alpha=0.05)
        mean_pli = float(np.nanmean(pli_curve))
        med_pli  = float(np.nanmedian(pli_curve))
        cover = float(100.0*np.nanmean((pli_curve > null95_curve).astype(float)))
        traces_XH.append((m, fk, t, pli_curve, null95_curve))
        rows.append({'metric':'XH-PLI', 'order':m, 'f0':fk,
                     'mean':mean_pli, 'median':med_pli, 'ci_lo':ci_lo, 'ci_hi':ci_hi,
                     'p_mean':p_mean, 'coverage_pct':cover, 'win_sec':win_sec})

    # ----- SubH‑PLI for subharmonics s=1/n -----
    traces_SH = []
    for fsb in subharmonics:
        if fsb < 0.1:  # ignore too low
            continue
        n = int(round(fundamental / fsb)) if fsb > 0 else None
        if n is None or n < 2:  # only true subharmonics
            continue
        win_sec = win_for_f0(fsb, cycles=cycles, min_win=min_win, max_win=max_win)
        phi_s_eeg = phase_series(x_eeg, fs, fsb, half_bw)
        t, pli_curve, null95_curve, p_mean, pli_raw = compute_SubH_PLI(phi_s_eeg, phi1_eeg, n, fs,
                                                                       win_sec=win_sec, step_sec=step_sec,
                                                                       smoother=smoother, n_perm=n_perm)
        ci_lo, ci_hi = block_bootstrap_ci(pli_curve, n_boot=1000, alpha=0.05)
        mean_pli = float(np.nanmean(pli_curve))
        med_pli  = float(np.nanmedian(pli_curve))
        cover = float(100.0*np.nanmean((pli_curve > null95_curve).astype(float)))
        traces_SH.append((n, fsb, t, pli_curve, null95_curve))
        rows.append({'metric':'SubH-PLI', 'order':n, 'f0':fsb,
                     'mean':mean_pli, 'median':med_pli, 'ci_lo':ci_lo, 'ci_hi':ci_hi,
                     'p_mean':p_mean, 'coverage_pct':cover, 'win_sec':win_sec})

    # ----- HCS (weighted sum over harmonics per time) -----
    # Align to common time grid (use intersections of centers)
    if traces_H:
        t_common = traces_H[0][1]
        # ensure all H traces share same centers; if not, resample by nearest
        def _resample_to(t_src, y_src, t_ref):
            idx = np.searchsorted(t_src, t_ref)
            idx = np.clip(idx, 0, len(t_src)-1)
            return y_src[idx]
        # weights
        if weights_scheme == 'inverse_k':
            weights = []
            for fk, t_k, y_k, _, _ in traces_H:
                k = max(1, int(round(fk/fundamental)))
                weights.append(1.0/float(k))
        else:
            weights = [1.0 for _ in traces_H]
        weights = np.asarray(weights, float)
        weights /= np.sum(weights)
        Y = []
        for (fk, t_k, y_k, _, _) in traces_H:
            if len(t_k) != len(t_common) or np.any(np.abs(t_k - t_common) > 1e-6):
                y_k = _resample_to(t_k, y_k, t_common)
            Y.append(y_k)
        Y = np.asarray(Y)  # shape (K, T)
        HCS_curve = np.tensordot(weights, Y, axes=(0,0))  # (T,)
        HCS_mean = float(np.nanmean(HCS_curve))
        HCS_lo, HCS_hi = block_bootstrap_ci(HCS_curve, n_boot=1000, alpha=0.05)
        rows.append({'metric':'HCS', 'order':0, 'f0':np.nan,
                     'mean':HCS_mean, 'median':float(np.nanmedian(HCS_curve)),
                     'ci_lo':HCS_lo, 'ci_hi':HCS_hi, 'p_mean':np.nan,
                     'coverage_pct':np.nan, 'win_sec':np.nan})
    else:
        HCS_curve = None; t_common = None

    summary = pd.DataFrame(rows)

    # --------------------------- plots ---------------------------
    # (1) Faceted H‑PLI traces with null curves
    if traces_H:
        nrows = len(traces_H)
        fig, axes = plt.subplots(nrows, 1, sharex=True, figsize=(10, max(6, 2.0*nrows)))
        if not isinstance(axes, np.ndarray): axes = np.array([axes])
        for i, (fk, t, y, thr, win_sec) in enumerate(traces_H):
            ax = axes[i]
            ax.plot(t, y, lw=1.8, label=f'H-PLI @ {fk:.3g} Hz')
            ax.plot(t, thr, lw=1.0, ls='--', alpha=0.7, label='null95 (smooth-aware)')
            ax.set_ylabel('PLI')
            ax.grid(True, alpha=0.25, linestyle=':')
            ax.legend(loc='upper right', fontsize=8)
        axes[-1].set_xlabel('Time (s)')
        fig.suptitle('H‑PLI (EEG vs SR) per harmonic'); fig.tight_layout()
        plt.savefig(os.path.join(out_dir, 'H_PLI_traces.png'), dpi=160, bbox_inches='tight')
        if show: plt.show(); plt.close()

    # (2) Bar charts: H‑PLI means with 95% CIs; compare to XH‑PLI means
    if len(summary):
        import pandas as pd
        S = summary
        Hbars = S[S['metric']=='H-PLI'].copy()
        if not Hbars.empty:
            x = np.arange(len(Hbars))
            fig, ax = plt.subplots(figsize=(10,3.2))
            ax.bar(x, Hbars['mean'].values, yerr=[Hbars['mean']-Hbars['ci_lo'], Hbars['ci_hi']-Hbars['mean']],
                   width=0.6, capsize=3)
            ax.set_xticks(x); ax.set_xticklabels([f"k≈{int(round(f/ fundamental))}\n{f:.2f} Hz" for f in Hbars['f0']], rotation=0)
            ax.set_ylabel('Mean H‑PLI (±CI)'); ax.set_title('H‑PLI by order')
            ax.grid(True, axis='y', alpha=0.25, linestyle=':')
            plt.tight_layout(); plt.savefig(os.path.join(out_dir, 'H_PLI_bars.png'), dpi=160)
            if show: plt.show(); plt.close()
        Xbars = S[S['metric']=='XH-PLI'].copy()
        if not Xbars.empty:
            # group by order m
            orders = Xbars['order'].values; means = Xbars['mean'].values
            ci_lo = Xbars['ci_lo'].values; ci_hi = Xbars['ci_hi'].values
            x = np.arange(len(orders))
            fig, ax = plt.subplots(figsize=(10,3.2))
            ax.bar(x, means, yerr=[means-ci_lo, ci_hi-means], width=0.6, capsize=3)
            ax.set_xticks(x); ax.set_xticklabels([f"m={m}" for m in orders])
            ax.set_ylabel('Mean XH‑PLI (±CI)'); ax.set_title('Cross‑order locking (waveform shape)')
            ax.grid(True, axis='y', alpha=0.25, linestyle=':')
            plt.tight_layout(); plt.savefig(os.path.join(out_dir, 'XH_PLI_bars.png'), dpi=160)
            if show: plt.show(); plt.close()

    # (3) HCS curve
    if HCS_curve is not None and t_common is not None:
        fig, ax = plt.subplots(figsize=(10,3.2))
        ax.plot(t_common, HCS_curve, lw=1.8)
        ax.set_xlabel('Time (s)'); ax.set_ylabel('HCS'); ax.grid(True, alpha=0.25, linestyle=':')
        ax.set_title('Harmonic Coherence Score (weighted sum of H‑PLI)')
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, 'HCS_curve.png'), dpi=160)
        if show: plt.show(); plt.close()

    # Save summary
    summary.to_csv(os.path.join(out_dir, 'locking_summary.csv'), index=False)

    return {'summary': summary, 'H_traces': traces_H, 'XH_traces': traces_XH, 'SH_traces': traces_SH,
            'HCS': (t_common, HCS_curve) if HCS_curve is not None else None,
            'out_dir': out_dir}

# --------------------------- example usage ---------------------------
# res = analyze_locking(
#     RECORDS,
#     eeg_channel='EEG.O1',
#     sr_channel='EEG.Pz',   # or magnetometer channel if available
#     fundamental=7.83,
#     harmonics=(7.83,14.3,20.8,27.3,33.8,40.3,46.8,53.3,59.8),
#     subharmonics=(3.915, 2.61, 1.9575, 1.566, 1.305, 1.11857, 0.9788, 1.2, 0.783),
#     half_bw=0.6,
#     cycles=8, min_win=8.0, max_win=120.0,
#     step_sec=1.0, n_perm=200, limit_high_hz=60.0,
#     smooth=True, weights_scheme='inverse_k',
#     out_dir='exports_locking', show=True
# )
