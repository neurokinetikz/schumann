"""
Disentangling waveform shape vs. true multi‑mode resonance (0.1–60 Hz)
=====================================================================
Implements simple tests, graphs, and summary stats for:
  (A) Cycle‑by‑cycle morphology at ~7–9 Hz (rise/decay ratio, peak/trough sharpness,
      zero‑crossing asymmetry) to quantify non‑sinusoidal shape.
  (B) IRASA‑cleaned oscillations (remove 1/f fractal) and re‑inspect harmonic peaks.
  (C) Polyspectra (auto‑bicoherence EEG; cross‑bicoherence SR→EEG) on a discrete
      SR frequency set (fundamental + harmonics up to 60 Hz).

Rule‑of‑thumb interpretation:
  • shape‑only → high auto‑bicoherence without cross‑bicoherence; harmonics shrink after IRASA;
    morphology metrics high; harmonic power correlates with sharpness/asymmetry.
  • true resonance → both auto‑ and cross‑bicoherence significant (e.g., (7.83,7.83)→15.66),
    IRASA‑oscillatory peaks persist; morphology may be neutral.

Outputs:
  • PNGs: morphology histograms/scatter, IRASA PSD plots, auto‑/cross‑bicoherence heatmaps.
  • CSV: summary table with key metrics & simple classification.

Dependencies: numpy, scipy, matplotlib, pandas.
"""

import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy import signal

# ----------------------------- I/O helpers -----------------------------

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
    alt='EEG.'+name
    if alt in df.columns:
        return pd.to_numeric(df[alt], errors='coerce').fillna(0.0).values.astype(float)
    raise ValueError(f'{name} not in dataframe')

# ----------------------------- Filters -----------------------------

def _bandpass(x, fs, lo, hi, order=4):
    ny=0.5*fs; lo=max(1e-6, min(lo, 0.99*ny)); hi=max(lo+1e-6, min(hi, 0.999*ny))
    b,a=signal.butter(order, [lo/ny, hi/ny], btype='band'); return signal.filtfilt(b,a,x)

def _lowpass(x, fs, hi, order=4):
    ny=0.5*fs; hi=max(1e-6, min(hi, 0.999*ny))
    b,a=signal.butter(order, hi/ny, btype='low'); return signal.filtfilt(b,a,x)

# ----------------------------- (A) Cycle‑by‑cycle morphology -----------------------------

def cycles_morphology(x, fs, f0=7.83, half=1.0, sharp_win=0.02):
    """Compute cycle features from bandpassed x around f0±half.
    Returns DataFrame with period, rise/decay times, ratio, peak/trough sharpness,
    zero‑crossing asymmetry (ZC asym), and timestamps of cycles (center time).
    """
    xb = _bandpass(x, fs, f0-half, f0+half)
    # find zero crossings for cycle boundaries
    s = np.signbit(xb)
    zc_idx = np.where(np.diff(s.astype(int)) != 0)[0]
    # peaks & troughs
    peaks,_ = signal.find_peaks(xb)
    troughs,_ = signal.find_peaks(-xb)
    # helper to nearest peak after a trough etc.
    def next_idx(arr, i):
        j = arr.searchsorted(i, side='left')
        return arr[j] if j < len(arr) else None
    peaks = np.asarray(peaks); troughs = np.asarray(troughs)
    rows=[]; w = int(round(sharp_win*fs))
    for i in range(len(troughs)-1):
        t0 = troughs[i]; t1 = troughs[i+1]
        # ensure one peak between troughs
        pk = peaks[(peaks>t0)&(peaks<t1)]
        if pk.size==0: continue
        pk = pk[0]
        # zero‑crossings inside the cycle
        z = zc_idx[(zc_idx>=t0)&(zc_idx<=t1)]
        # features
        period = (t1 - t0)/fs
        rise   = (pk - t0)/fs
        decay  = (t1 - pk)/fs
        rz     = rise/(decay+1e-12)
        # sharpness via local slope/curvature proxy around extrema
        a = max(0, pk - w); b = min(len(xb), pk + w + 1)
        peak_sharp = float(xb[pk] - 0.5*(xb[a] + xb[b-1])) if (b-a)>=3 else float('nan')
        a = max(0, t0 - w); b = min(len(xb), t0 + w + 1)
        trough_sharp = float(0.5*(xb[a] + xb[b-1]) - xb[t0]) if (b-a)>=3 else float('nan')
        # ZC asymmetry: fraction of cycle spent above zero vs below
        if z.size >= 2:
            # first crossing after t0 and next crossing
            z1, z2 = z[0], z[1]
            above = (z2 - z1)/fs
            zc_asym = above / (period + 1e-12)
        else:
            zc_asym = float('nan')
        rows.append({'t_center': (t0+t1)/(2*fs), 'period_s': period, 'rise_s': rise, 'decay_s': decay,
                     'rise_decay_ratio': rz, 'peak_sharp': peak_sharp, 'trough_sharp': trough_sharp,
                     'zc_asym': zc_asym})
    return pd.DataFrame(rows)

# ----------------------------- (B) IRASA‑like background removal -----------------------------

def irasa_psd(x, fs, hset=(1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9), nperseg=None, fmax=60.0):
    """Approximate IRASA: resample by h and 1/h, compute PSDs, map freqs back by /h and *h,
    take geometric mean and then median over h to estimate fractal; subtract (in linear power)
    to get oscillatory. Returns f, Pxx, P_frac, P_osc (clipped >=0).
    """
    x = np.asarray(x, float)
    if nperseg is None:
        nperseg = int(max(2*fs, 1024))
    # base PSD
    f, P = signal.welch(x, fs=fs, window='hann', nperseg=nperseg, noverlap=nperseg//2,
                        nfft=int(2**np.ceil(np.log2(nperseg*2))), scaling='density')
    mask = (f>0) & (f<=min(fmax, 0.999*0.5*fs))
    f = f[mask]; P = P[mask]
    # surrogate fractal estimate
    def res_psd(y, up, down):
        z = signal.resample_poly(y, up, down)
        fs2 = fs*(up/float(down))
        f2, P2 = signal.welch(z, fs=fs2, window='hann', nperseg=nperseg, noverlap=nperseg//2,
                              nfft=int(2**np.ceil(np.log2(nperseg*2))), scaling='density')
        return f2, P2
    P_fracs=[]
    for h in hset:
        # up/down factors via rational approximation
        from fractions import Fraction
        frac = Fraction(str(h)).limit_denominator(64)
        up, down = frac.numerator, frac.denominator
        f_h, Ph = res_psd(x, up, down)
        f_hm, Phm = res_psd(x, down, up)  # 1/h
        # map both to base freq grid
        # resampling by h compresses time → expands freq by h; to map back divide freqs by h
        fh_map = f_h/float(h);  Ph_map = np.interp(f, fh_map, Ph, left=np.nan, right=np.nan)
        fhm_map = f_hm*float(h); Phm_map = np.interp(f, fhm_map, Phm, left=np.nan, right=np.nan)
        # geometric mean
        G = np.sqrt(np.maximum(Ph_map, 1e-20) * np.maximum(Phm_map, 1e-20))
        P_fracs.append(G)
    P_frac = np.nanmedian(np.vstack(P_fracs), axis=0)
    P_osc = np.clip(P - P_frac, 0, None)
    return f, P, P_frac, P_osc

# ----------------------------- (C) Discrete (cross‑)bicoherence -----------------------------

def _fft_segments(x, fs, nperseg, step):
    w = signal.hann(nperseg, sym=False)
    hop = nperseg - step
    nseg = 1 + max(0, (len(x)-nperseg)//hop)
    Xs=[]
    for i in range(nseg):
        s = i*hop; e = s + nperseg
        seg = x[s:e]
        if len(seg) < nperseg: break
        seg = seg - np.mean(seg)
        X = np.fft.rfft(w*seg, n=nperseg)
        Xs.append(X)
    Xs = np.asarray(Xs)
    freqs = np.fft.rfftfreq(nperseg, d=1.0/fs)
    return freqs, Xs


def bicoherence_discrete_auto(x, fs, f_list, nperseg=None, step=None):
    """Auto‑bicoherence on a discrete frequency list f_list (Hz). Returns matrix B[i,j] for f1=f_list[i], f2=f_list[j] with f1+f2 in grid.
    """
    if nperseg is None:
        nperseg = int(max(4*fs, 2048))
    if step is None:
        step = nperseg//2
    freqs, Xs = _fft_segments(x, fs, nperseg, step)
    # map desired freqs to bins
    def bin_idx(f): return int(np.argmin(np.abs(freqs - f)))
    idxs = [bin_idx(f) for f in f_list]
    B = np.full((len(f_list), len(f_list)), np.nan, float)
    for i, fi in enumerate(f_list):
        for j, fj in enumerate(f_list):
            fk = fi + fj
            if fk > freqs[-1]:
                continue
            ik = bin_idx(fk); ii = idxs[i]; jj = idxs[j]
            num = np.mean(Xs[:, ii] * Xs[:, jj] * np.conj(Xs[:, ik]))
            den = np.sqrt(np.mean(np.abs(Xs[:, ii]*Xs[:, jj])**2) * np.mean(np.abs(Xs[:, ik])**2) + 1e-20)
            B[i,j] = np.abs(num) / (den + 1e-20)
    return np.asarray(f_list), B


def bicoherence_discrete_cross(sr, eeg, fs, f_list, nperseg=None, step=None):
    """Cross‑bicoherence variant: B_sse(f1,f2) = <S(f1) S(f2) E*(f1+f2)> / sqrt(<|S(f1)S(f2)|^2><|E(f1+f2)|^2>).
    """
    if nperseg is None:
        nperseg = int(max(4*fs, 2048))
    if step is None:
        step = nperseg//2
    freqs, Ss = _fft_segments(sr, fs, nperseg, step)
    _,    Es = _fft_segments(eeg, fs, nperseg, step)
    def bin_idx(f): return int(np.argmin(np.abs(freqs - f)))
    idxs = [bin_idx(f) for f in f_list]
    B = np.full((len(f_list), len(f_list)), np.nan, float)
    for i, fi in enumerate(f_list):
        for j, fj in enumerate(f_list):
            fk = fi + fj
            if fk > freqs[-1]:
                continue
            ik = bin_idx(fk); ii = idxs[i]; jj = idxs[j]
            num = np.mean(Ss[:, ii] * Ss[:, jj] * np.conj(Es[:, ik]))
            den = np.sqrt(np.mean(np.abs(Ss[:, ii]*Ss[:, jj])**2) * np.mean(np.abs(Es[:, ik])**2) + 1e-20)
            B[i,j] = np.abs(num) / (den + 1e-20)
    return np.asarray(f_list), B

# ----------------------------- Orchestrator -----------------------------

def analyze_shape_vs_resonance(
    RECORDS,
    eeg_channel: str,
    sr_channel: str,
    time_col='Timestamp',
    fundamental=7.83,
    harmonics=(7.83,14.3,20.8,27.3,33.8,40.3,46.8,53.3,59.8),
    half_bw=0.6,
    slow_band=(0.003,0.03),
    fmax=60.0,
    nperseg_irasa=None,
    nperseg_bico=None,
    out_dir='exports_shape_vs_res', show=True,
    n_perm=200
):
    """Run morphology, IRASA, and (cross‑)bicoherence tests; save figures and a summary CSV."""
    ensure_dir(out_dir)
    ensure_timestamp_column(RECORDS, time_col=time_col)
    fs = infer_fs(RECORDS, time_col)

    x = get_series(RECORDS, eeg_channel)
    s = get_series(RECORDS, sr_channel)

    # ---- (A) Morphology at ~fundamental ----
    morph = cycles_morphology(x, fs, f0=fundamental, half=half_bw, sharp_win=0.02)

    # Relation to harmonic power: compute PSD and ratio (sum harmonics 14–60 / power at 7.8)
    nper = int(max(4*fs, 2048))
    f_psd, P = signal.welch(x, fs=fs, window='hann', nperseg=nper, noverlap=nper//2,
                            nfft=int(2**np.ceil(np.log2(nper*2))), scaling='density')
    mask = (f_psd>0) & (f_psd<=min(fmax, 0.999*0.5*fs))
    f_psd = f_psd[mask]; P = P[mask]
    def at(freq, bw=0.4):
        m = (f_psd>=freq-bw) & (f_psd<=freq+bw)
        return float(np.trapz(P[m], f_psd[m])) if np.any(m) else 0.0
    fund_pow = at(fundamental, bw=half_bw)
    harm_bw = 0.5
    harm_list = [f for f in harmonics if f>fundamental and f<=fmax]
    harm_pow = sum(at(f, bw=harm_bw) for f in harm_list)
    harm_ratio = float(harm_pow / (fund_pow + 1e-12))

    # ---- (B) IRASA ----
    fI, Pxx, Pfrac, Posc = irasa_psd(x, fs, nperseg=nperseg_irasa, fmax=fmax)

    # ---- (C) Bicoherence on discrete SR set ----
    f_list = [f for f in harmonics if f <= min(fmax, 0.999*0.5*fs)]
    nper_b = nperseg_bico or int(max(4*fs, 4096))
    step_b = nper_b//2
    fgrid, Bauto = bicoherence_discrete_auto(x, fs, f_list, nperseg=nper_b, step=step_b)
    _,     Bcross = bicoherence_discrete_cross(s, x, fs, f_list, nperseg=nper_b, step=step_b)

    # Simple surrogates: circularly shift SR vs EEG for cross; sign‑flip segments for auto
    rng = np.random.default_rng(7)
    def sur_cross(nr=200):
        vals=[]
        _, Es = _fft_segments(x, fs, nper_b, step_b)
        freqs, Ss = _fft_segments(s, fs, nper_b, step_b)
        for _ in range(int(nr)):
            # circular shift SR in time domain by random samples
            sh = int(rng.integers(1, len(s)-1))
            s_sh = np.r_[s[-sh:], s[:-sh]]
            _,     Bc = bicoherence_discrete_cross(s_sh, x, fs, f_list, nperseg=nper_b, step=step_b)
            vals.append(Bc)
        return np.stack(vals, axis=0)  # (nr, F, F)
    def sur_auto(nr=200):
        vals=[]
        for _ in range(int(nr)):
            # randomly invert segments to break consistent triple phase
            x_sh = x.copy()
            segN = nper_b
            for start in range(0, len(x_sh)-segN, segN):
                if rng.random()<0.5:
                    x_sh[start:start+segN] *= -1
            _, Ba = bicoherence_discrete_auto(x_sh, fs, f_list, nperseg=nper_b, step=step_b)
            vals.append(Ba)
        return np.stack(vals, axis=0)

    SC = sur_cross(n_perm)
    SA = sur_auto(n_perm)
    thr_cross = np.nanpercentile(SC, 95, axis=0)
    thr_auto  = np.nanpercentile(SA, 95, axis=0)

    # Key cells near (7.83,7.83)->15.66
    def nearest_idx(arr, val):
        return int(np.argmin(np.abs(np.asarray(arr)-val)))
    i7 = nearest_idx(fgrid, fundamental)
    cross_7_7 = float(Bcross[i7, i7]) if i7 < len(fgrid) else np.nan
    auto_7_7  = float(Bauto[i7, i7])  if i7 < len(fgrid) else np.nan
    cross_thr = float(thr_cross[i7, i7]) if i7 < len(fgrid) else np.nan
    auto_thr  = float(thr_auto[i7, i7])  if i7 < len(fgrid) else np.nan

    # ----- simple classification -----
    # thresholds can be tuned; start conservative with surrogate 95th percentile
    shape_only = (auto_7_7 > auto_thr) and not (cross_7_7 > cross_thr)
    true_res   = (auto_7_7 > auto_thr) and (cross_7_7 > cross_thr)
    label = 'shape_only' if shape_only else ('true_resonance' if true_res else 'ambiguous')

    # ----------------------------- plots -----------------------------
    # Morphology distributions & scatter vs harmonic ratio
    if not morph.empty:
        fig, axes = plt.subplots(1,3, figsize=(12,3.2))
        axes[0].hist(morph['rise_decay_ratio'].dropna(), bins=30, alpha=0.9)
        axes[0].set_title('Rise/Decay ratio'); axes[0].set_xlabel('ratio'); axes[0].set_ylabel('count')
        axes[1].hist(morph['peak_sharp'].dropna(), bins=30, alpha=0.9)
        axes[1].set_title('Peak sharpness')
        axes[2].hist(morph['zc_asym'].dropna(), bins=30, alpha=0.9)
        axes[2].set_title('ZC asymmetry'); axes[2].set_xlabel('fraction of cycle above 0')
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, 'morph_hist.png'), dpi=160)
        if show: plt.show(); plt.close()
        # scatter sharpness vs harmonic ratio
        fig, ax = plt.subplots(figsize=(5.2,3.2))
        ax.scatter(morph['peak_sharp'], np.full(len(morph), harm_ratio), s=12, alpha=0.4)
        ax.set_xlabel('Peak sharpness (a.u.)'); ax.set_ylabel('Harmonic ratio (14–60 / 7.8)')
        ax.set_title('Harmonics vs shape (quick view)')
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, 'morph_scatter.png'), dpi=160)
        if show: plt.show(); plt.close()

    # IRASA plots
    fig, ax = plt.subplots(figsize=(7.8,3.2))
    ax.plot(fI, 10*np.log10(Pxx+1e-20), label='PSD (Welch)')
    ax.plot(fI, 10*np.log10(Pfrac+1e-20), label='Fractal (IRASA≈)')
    ax.plot(fI, 10*np.log10(Posc+1e-20), label='Oscillatory (PSD−Fractal)')
    ax.set_xlim(0, fmax); ax.set_xlabel('Hz'); ax.set_ylabel('dB')
    ax.set_title('IRASA‑cleaned oscillations'); ax.legend(); ax.grid(True, alpha=0.25, linestyle=':')
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, 'irasa_psd.png'), dpi=160)
    if show: plt.show(); plt.close()

    # Bicoherence heatmaps
    extent=[fgrid[0], fgrid[-1], fgrid[0], fgrid[-1]]
    def heat(M, thr, title, fname):
        plt.figure(figsize=(6.4,5.4))
        plt.imshow(M, origin='lower', aspect='equal', extent=extent, vmin=0, vmax=np.nanmax(M))
        cb=plt.colorbar(); cb.set_label('bicoherence')
        yy, xx = np.where(M > thr)
        if yy.size:
            plt.scatter(fgrid[xx], fgrid[yy], s=10, c='cyan', alpha=0.7, label='> null95')
            plt.legend(loc='upper right', fontsize=8)
        plt.xlabel('f1 (Hz)'); plt.ylabel('f2 (Hz)'); plt.title(title)
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, fname), dpi=160)
        if show: plt.show(); plt.close()
    heat(Bauto, thr_auto,  'Auto‑bicoherence EEG',  'bico_auto.png')
    heat(Bcross, thr_cross,'Cross‑bicoherence SR→EEG','bico_cross.png')

    # ----------------------------- summary table -----------------------------
    rows=[{
        'fundamental': fundamental,
        'harm_ratio_14_60_over_7_8': harm_ratio,
        'auto_bico_7.83,7.83': auto_7_7,
        'auto_null95_7.83,7.83': auto_thr,
        'cross_bico_7.83,7.83': cross_7_7,
        'cross_null95_7.83,7.83': cross_thr,
        'morph_median_rise_decay': float(morph['rise_decay_ratio'].median()) if not morph.empty else np.nan,
        'morph_median_peak_sharp': float(morph['peak_sharp'].median()) if not morph.empty else np.nan,
        'morph_median_zc_asym': float(morph['zc_asym'].median()) if not morph.empty else np.nan,
        'classification': label
    }]
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(out_dir, 'shape_vs_res_summary.csv'), index=False)

    return {
        'morphology': morph,
        'irasa': (fI, Pxx, Pfrac, Posc),
        'bicoherence': {'fgrid': fgrid, 'auto': Bauto, 'cross': Bcross,
                        'thr_auto': thr_auto, 'thr_cross': thr_cross},
        'summary': summary,
        'label': label,
        'out_dir': out_dir
    }

# ----------------------------- Example usage -----------------------------
# res = analyze_shape_vs_resonance(
#     RECORDS,
#     eeg_channel='EEG.O1',
#     sr_channel='EEG.Pz',        # or a magnetometer channel
#     time_col='Timestamp',
#     fundamental=7.83,
#     harmonics=(7.83,14.3,20.8,27.3,33.8,40.3,46.8,53.3,59.8),
#     half_bw=0.6,
#     slow_band=(0.003,0.03),
#     fmax=60.0,
#     n_perm=200,
#     out_dir='exports_shape_vs_res', show=True
# )
