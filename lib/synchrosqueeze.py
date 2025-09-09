"""
Synchrosqueezed time–frequency analysis for SR fundamentals & harmonics (0.1–60 Hz)
===================================================================================

What you get
------------
1) ssq‑CWT heatmaps (EEG & SR) with ridge‑sharp energy (uses ssqueezepy if available).
   • Fallback: high‑res STFT spectrogram if ssqueezepy is not installed.
2) Ridge extraction per target f0 (fundamental + harmonics/subharmonics), with:
   • Ridge frequency track f̂(t) and ridge power p̂(t) inside ±bw around f0.
   • Alignment error |f̂(t) − f0| statistics; coverage above null.
3) Simple validation tests (per f0):
   • Within‑band coverage vs off‑band surrogate (circular shift) → p‑value.
   • EEG↔SR ridge‑power correlation with circular‑shift null → p‑value.
4) Ready‑made plotting: heatmaps + overlaid ridge tracks + bar charts of metrics.

Limits: All analyses clamp to ≤ 60 Hz (or ≤ Nyquist).

Dependencies: numpy, scipy, matplotlib. Optional: ssqueezepy (preferred).
Install (optional): pip install ssqueezepy
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# ---------- small helpers (reuse your own if already defined) ----------

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

# ---------- synchrosqueezed CWT (preferred) or STFT (fallback) ----------
try:
    from ssqueezepy import ssq_cwt, cwt, Wavelet
    _HAS_SSQ = True
except Exception:
    _HAS_SSQ = False


def _ssq_cwt_TFR(x, fs, fmin=0.1, fmax=60.0):
    """Return (t, f, power) using synchrosqueezed CWT if available; else STFT fallback.
    power has shape (n_freqs, n_times). t in seconds, f in Hz.
    """
    N = len(x); t = np.arange(N)/float(fs)
    ny = 0.5*fs
    fmax = float(min(fmax, 0.999*ny))
    if _HAS_SSQ:
        wv = Wavelet('morlet')  # good narrow ridges
        Wx, scales = cwt(x, wavelet=wv, fs=fs)
        Tx, fs_ssq, *_ = ssq_cwt(Wx, scales, wavelet=wv, fs=fs)
        freqs = np.asarray(fs_ssq, float)
        mask = (freqs >= float(fmin)) & (freqs <= float(fmax))
        P = np.abs(Tx[mask, :])**2
        f = freqs[mask]
        return t, f, P
    # Fallback: high‑res STFT spectrogram
    nper = int(max(fs*4, 256))      # 4 s window for LF detail
    nover = int(0.75*nper)
    nfft = int(2**np.ceil(np.log2(nper*2)))
    f, tt, Sxx = signal.spectrogram(x, fs=fs, window='hann', nperseg=nper,
                                    noverlap=nover, nfft=nfft, scaling='spectrum', mode='magnitude')
    mask = (f >= float(fmin)) & (f <= float(fmax))
    P = (Sxx[mask, :])**2
    # Interpolate STFT time grid to sample grid for comparable t axis
    if tt.size and tt[0] > 0:
        t_coarse = tt
        # return coarse t to avoid misleading densification
        return t_coarse, f[mask], P
    return t, f[mask], P

# ---------- ridge extraction around a target frequency ----------

def ridge_in_band(P, f, t, f0, bw=0.6):
    """Within [f0-bw, f0+bw], take per‑time max → ridge freq & power.
    Returns dict with arrays: f_hat(t), p_hat(t), and simple stats.
    """
    f0 = float(f0); bw = float(bw)
    band = np.where((f >= max(0.0, f0-bw)) & (f <= f0+bw))[0]
    if band.size == 0:
        # fallback: nearest single bin
        idx = int(np.argmin(np.abs(f - f0)))
        return {
            'f_hat': np.full(t.shape, f[idx]),
            'p_hat': P[idx, :].astype(float),
            'band_idx': np.array([idx]),
            'band_freqs': np.array([f[idx]])
        }
    sub = P[band, :]
    jmax = np.argmax(sub, axis=0)               # argmax over band per time
    idxs = band[jmax]
    f_hat = f[idxs]
    p_hat = sub[jmax, np.arange(sub.shape[1])]
    return {'f_hat': f_hat, 'p_hat': p_hat, 'band_idx': band, 'band_freqs': f[band]}

# ---------- simple surrogates & validation ----------

def circular_shift(a, s):
    s = int(s) % len(a)
    if s == 0: return a
    return np.r_[a[-s:], a[:-s]]


def validate_ridge(p_hat, offband_ref=None, n_perm=200, rng=None):
    """Coverage & p‑value: how often is ridge power above off‑band reference?
    offband_ref: 1D array representing background power (same length or pooled).
    If None, uses the ridge power itself with circular‑shift surrogates.
    Returns dict with coverage_pct, p_value, thr95.
    """
    rng = np.random.default_rng() if rng is None else rng
    x = np.asarray(p_hat, float)
    if offband_ref is None:
        # build null by circular shift → max distribution
        maxima = []
        for _ in range(int(n_perm)):
            s = rng.integers(1, len(x)-1)
            xs = circular_shift(x, s)
            maxima.append(float(np.nanmax(xs)))
        thr95 = float(np.nanpercentile(maxima, 95))
    else:
        ref = np.asarray(offband_ref, float)
        thr95 = float(np.nanpercentile(ref, 95))
    coverage = float(100.0*np.nanmean((x > thr95).astype(float)))
    # simple p: percentile of observed median vs surrogate medians
    meds = []
    for _ in range(int(n_perm)):
        s = rng.integers(1, len(x)-1)
        xs = circular_shift(x, s)
        meds.append(float(np.nanmedian(xs)))
    pval = float((np.sum(np.asarray(meds) >= np.nanmedian(x)) + 1) / (n_perm + 1))
    return {'coverage_pct': coverage, 'thr95': thr95, 'p_value': pval}


def validate_eeg_sr_coupling(p_eeg, p_sr, n_perm=200, rng=None):
    """Correlation between EEG & SR ridge power with circular‑shift null."""
    rng = np.random.default_rng() if rng is None else rng
    x = np.asarray(p_eeg, float); y = np.asarray(p_sr, float)
    L = min(len(x), len(y))
    x = x[:L]; y = y[:L]
    r_obs = float(np.corrcoef(x, y)[0,1]) if np.std(x)>0 and np.std(y)>0 else 0.0
    null = []
    for _ in range(int(n_perm)):
        s = rng.integers(1, L-1)
        yn = circular_shift(y, s)
        r = float(np.corrcoef(x, yn)[0,1]) if np.std(yn)>0 else 0.0
        null.append(r)
    p = float((np.sum(np.asarray(null) >= r_obs) + 1) / (n_perm + 1))
    return {'r': r_obs, 'p_value': p}

# --- helper: always make parent dir before saving ---
def _safe_savefig(path, dpi=160, **kwargs):
    if not path:
        return
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
        plt.savefig(path, dpi=dpi, **kwargs)

# ---------- main user‑facing routine ----------

def ssq_sr_validate(RECORDS,
                    eeg_channel: str,
                    sr_channel: str,
                    time_col='Timestamp',
                    freq_groups=None,
                    fmin=0.1, fmax=60.0,
                    bw=0.6, n_perm=200,
                    show=True, out_dir=None):
    """
    Compute synchrosqueezed (or STFT fallback) T–F for EEG & SR; extract ridges at
    requested frequencies; run simple validations; and draw simple graphs.

    freq_groups: dict name -> tuple/list of target f0s. If None, a default set
                 using typical SR harmonics up to 60 Hz is used.
    """
    import os, pandas as pd
    ensure_timestamp_column(RECORDS, time_col=time_col)
    fs = infer_fs(RECORDS, time_col)
    ny = 0.5*fs
    fmax = float(min(fmax, 0.999*ny))

    if freq_groups is None:
        freq_groups = {
            'SR_UpTo33': (7.83, 14.3, 20.8, 27.3, 33.8),
            'SR_40to60': (7.83, 40.3, 46.8, 53.3, 59.8),
            'SR_Sub_2to5': (7.83, 3.915, 2.61, 1.9575, 1.566),
            'SR_Sub_6to10': (7.83, 1.305, 1.11857, 0.9788, 1.2, 0.783),
        }

    xe = get_series(RECORDS, eeg_channel).astype(float)
    xs = get_series(RECORDS, sr_channel).astype(float)

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)

    # TFRs
    t_eeg, f_eeg, P_eeg = _ssq_cwt_TFR(xe, fs, fmin=fmin, fmax=fmax)
    t_sr,  f_sr,  P_sr  = _ssq_cwt_TFR(xs, fs, fmin=fmin, fmax=fmax)

    # heatmaps
    # --- replace your _plot_heatmap with this version ---
    def _plot_heatmap(t, f, P, title, lines=None, out_png=None, show=True):
        plt.figure(figsize=(10, 4.2))
        extent=[t[0], t[-1], f[0], f[-1]]
        plt.imshow(10*np.log10(P + 1e-20), aspect='auto', origin='lower',
        extent=extent, cmap='magma')
        cb=plt.colorbar(); cb.set_label('Power (dB)')
        if lines:
            for (freq, col) in lines:
                if f[0] <= freq <= f[-1]:
                    plt.plot([t[0], t[-1]],[freq,freq], color=col, lw=1.0, ls='--', alpha=0.7)
                    plt.xlabel('Time (s)'); plt.ylabel('Frequency (Hz)')
                    plt.title(title)
                    plt.tight_layout()
        if out_png:
            _safe_savefig(out_png, dpi=160)
        if show:
            plt.show()
            plt.close()

    # overlay expected harmonics with colors
    COLORS = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b']

    # Prepare lines for all f0s
    all_f0s = sorted(set([round(float(f),6) for vals in freq_groups.values() for f in vals]))
    line_list = [(f0, COLORS[i % len(COLORS)]) for i, f0 in enumerate(all_f0s) if fmin <= f0 <= fmax]

    if show:
        _plot_heatmap(t_eeg, f_eeg, P_eeg, f'EEG T–F (ssqCWT {"on" if _HAS_SSQ else "STFT"})',
                      lines=line_list,
                      out_png=(None if out_dir is None else os.path.join(out_dir, 'EEG_TF.png')),show=show)
        _plot_heatmap(t_sr, f_sr, P_sr, f'SR T–F (ssqCWT {"on" if _HAS_SSQ else "STFT"})',
                      lines=line_list,
                      out_png=(None if out_dir is None else os.path.join(out_dir, 'SR_TF.png')),show=show)

    # For each group/f0: ridges + validations
    rows = []
    for gname, f0s in freq_groups.items():
        for idx, f0 in enumerate(f0s):
            if not (fmin <= f0 <= fmax):
                continue
            col = COLORS[idx % len(COLORS)]
            eeg_r = ridge_in_band(P_eeg, f_eeg, t_eeg, f0, bw=bw)
            sr_r  = ridge_in_band(P_sr,  f_sr,  t_sr,  f0, bw=bw)
            # alignment error (median absolute deviation in Hz)
            align_eeg = float(np.nanmedian(np.abs(eeg_r['f_hat'] - f0)))
            align_sr  = float(np.nanmedian(np.abs(sr_r['f_hat']  - f0)))
            # validations
            val_eeg = validate_ridge(eeg_r['p_hat'], n_perm=n_perm)
            val_sr  = validate_ridge(sr_r['p_hat'],  n_perm=n_perm)
            val_xy  = validate_eeg_sr_coupling(eeg_r['p_hat'], sr_r['p_hat'], n_perm=n_perm)
            rows.append({
                'group': gname, 'f0': float(f0), 'bw': float(bw),
                'EEG_align_Hz_med': align_eeg,
                'EEG_coverage_pct': val_eeg['coverage_pct'], 'EEG_pval': val_eeg['p_value'],
                'SR_align_Hz_med': align_sr,
                'SR_coverage_pct': val_sr['coverage_pct'], 'SR_pval': val_sr['p_value'],
                'EEGxSR_r': val_xy['r'], 'EEGxSR_pval': val_xy['p_value']
            })
            # quick line plots of ridges over time
            if show:
                plt.figure(figsize=(10, 2.6))
                plt.plot(t_eeg, eeg_r['f_hat'], color=col, lw=1.5, label=f'EEG ridge @ {f0:.3g} Hz')
                plt.plot(t_sr,  sr_r['f_hat'],  color=col, lw=1.0, ls='--', alpha=0.7, label='SR ridge')
                plt.axhline(f0, color=col, lw=1.0, ls=':', alpha=0.7)
                plt.ylabel('Freq (Hz)'); plt.xlabel('Time (s)'); plt.ylim(max(fmin,0.05), fmax)
                plt.title(f'Ridge tracks around {f0:.3g} Hz (±{bw:.2f} Hz)')
                plt.legend(loc='upper right', fontsize=8); plt.tight_layout()
                if out_dir: plt.savefig(os.path.join(out_dir, f'ridge_{gname}_{f0:.3g}Hz.png'), dpi=160)
                plt.show(); plt.close()

    import pandas as pd
    summary = pd.DataFrame(rows)

    # bar plot: coverage per f0 (EEG & SR)
    if show and len(rows):
        fig, ax = plt.subplots(figsize=(10,3.2))
        f0s_plot = summary['f0'].values
        ax.bar(f0s_plot-0.15, summary['EEG_coverage_pct'].values, width=0.3, label='EEG')
        ax.bar(f0s_plot+0.15, summary['SR_coverage_pct'].values, width=0.3, label='SR')
        ax.set_xlabel('Frequency (Hz)'); ax.set_ylabel('Coverage > null95 (%)')
        ax.set_title('Ridge coverage vs null (EEG vs SR)')
        ax.legend(); plt.tight_layout()
        if out_dir: plt.savefig(os.path.join(out_dir, 'coverage_bar.png'), dpi=160)
        plt.show(); plt.close()

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        summary.to_csv(os.path.join(out_dir, 'sr_ssq_summary.csv'), index=False)

    return {'summary': summary, 'EEG_TF': (t_eeg, f_eeg, P_eeg), 'SR_TF': (t_sr, f_sr, P_sr)}


# ---------------- Example usage ----------------
# res = ssq_sr_validate(
#     RECORDS,
#     eeg_channel='EEG.O1',
#     sr_channel='EEG.Pz',   # or a magnetometer if you have it
#     time_col='Timestamp',
#     freq_groups={
#        'Harmonics_UpTo33': (7.83, 14.3, 20.8, 27.3, 33.8),
#        'HighHarmonics_40to60': (7.83, 40.3, 46.8, 53.3, 59.8),
#        'Subharmonics_2to5': (7.83, 3.915, 2.61, 1.9575, 1.566),
#        'Subharmonics_6to10_Mixed': (7.83, 1.305, 1.11857, 0.9788, 1.2, 0.783),
#     },
#     fmin=0.1, fmax=60.0, bw=0.6, n_perm=200, show=True,
#     out_dir='exports_ssq'
# )
