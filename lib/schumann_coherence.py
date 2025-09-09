"""
EEG–Schumann Coherence Testing — Simple Graphs & Validation
===========================================================

What it does
------------
1) Per-channel Welch coherence vs SR at Schumann harmonics with 95% shift-null.
2) Time–frequency wavelet coherence (WTC) with cluster-based permutation.
3) Sliding-window coherence time series at 7.83 Hz with a global 95% null line.

Inputs
------
RECORDS: pandas.DataFrame with a time column (default 'Timestamp') and EEG.* columns.
eeg_channels: list of EEG channel names (e.g., ['EEG.O1','EEG.O2',...]).
sr_channel: the Schumann/ELF reference column (e.g., magnetometer). If you don’t
            have one, you can use a posterior EEG (Oz/O1/O2) as a proxy.

Usage
-----
res = run_eeg_schumann_coherence(
    RECORDS,
    eeg_channels=['EEG.O1','EEG.O2','EEG.P7','EEG.P8'],
    sr_channel='EEG.O1',                         # use your magnetometer if you have it
    ignition_windows=[(290,310),(580,600)],
    baseline_windows=[(0,290),(325,580)],
    time_col='Timestamp',
    out_dir='exports_eeg_sr/S01',
    show=False
)
print(res['summary'])
"""
from __future__ import annotations
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from scipy import signal
import networkx as nx

# ---------------- small utilities ----------------
def _ensure_dir(d): os.makedirs(d, exist_ok=True); return d

def infer_fs(df: pd.DataFrame, time_col='Timestamp')->float:
    t = np.asarray(pd.to_numeric(df[time_col], errors='coerce').values, float)
    dt = np.diff(t); dt = dt[(dt>0)&np.isfinite(dt)]
    if dt.size==0: raise ValueError("Cannot infer fs from time column.")
    return float(1.0/np.median(dt))

def get_series(df: pd.DataFrame, name: str)->np.ndarray:
    if name in df.columns:
        x = pd.to_numeric(df[name], errors='coerce').fillna(0.0).values
        return np.asarray(x, float)
    alt = 'EEG.'+name
    if alt in df.columns:
        x = pd.to_numeric(df[alt], errors='coerce').fillna(0.0).values
        return np.asarray(x, float)
    raise ValueError(f"Series '{name}' not in DataFrame.")

def slice_concat(x: np.ndarray, fs: float, wins: Optional[List[Tuple[float,float]]]):
    if not wins: return x.copy()
    segs=[]; n=len(x)
    for (a,b) in wins:
        i0,i1 = int(round(a*fs)), int(round(b*fs))
        i0=max(0,i0); i1=min(n,i1)
        if i1>i0: segs.append(x[i0:i1])
    return np.concatenate(segs) if segs else x.copy()

def bandpass(x, fs, f1, f2, order=4):
    ny=0.5*fs; f1=max(1e-6,min(f1,0.99*ny)); f2=max(f1+1e-6,min(f2,0.999*ny))
    b,a=signal.butter(order,[f1/ny,f2/ny],btype='band'); return signal.filtfilt(b,a,x)

# ---------------- Welch MSC @ harmonics with shift-null ----------------
def msc_harmonics_table(df, eeg_channels, sr_channel, wins, time_col='Timestamp',
                        harmonics=(7.83,14.3,20.8,27.3,33.8), nperseg_sec=4.0, n_null=200)->pd.DataFrame:
    fs = infer_fs(df, time_col)
    sr = slice_concat(get_series(df, sr_channel), fs, wins)
    nperseg = int(round(nperseg_sec*fs)); noverlap=nperseg//2
    rows=[]
    rng = np.random.default_rng(7)
    for ch in eeg_channels:
        x = slice_concat(get_series(df, ch), fs, wins)
        f, C = signal.coherence(x, sr, fs=fs, nperseg=nperseg, noverlap=noverlap)
        # null by circularly shifting SR
        null_vals = {h:[] for h in harmonics}
        for _ in range(n_null):
            s = int(rng.integers(1, len(sr)-1))
            sr_sh = np.r_[sr[-s:], sr[:-s]]
            _, C0 = signal.coherence(x, sr_sh, fs=fs, nperseg=nperseg, noverlap=noverlap)
            for h in harmonics:
                idx = int(np.argmin(np.abs(f - h)))
                null_vals[h].append(float(C0[idx]))
        for h in harmonics:
            idx = int(np.argmin(np.abs(f - h)))
            coh = float(C[idx])
            thr = float(np.nanpercentile(null_vals[h], 95)) if null_vals[h] else np.nan
            rows.append({'channel':ch, 'freq':float(f[idx]), 'MSC':coh, 'null95':thr})
    return pd.DataFrame(rows)

# ---------------- Wavelet coherence (TF) with cluster permutation ----------------
def wavelet_coherence_tf(df, x_name, y_name, time_col='Timestamp',
                         fmin=4, fmax=40, n_freq=64, w0=6.0,
                         n_perm=200, alpha=0.05, wins=None, show=True, out_png=None)->Dict[str,object]:
    fs = infer_fs(df, time_col)
    x = slice_concat(get_series(df, x_name), fs, wins)
    y = slice_concat(get_series(df, y_name), fs, wins)
    N = len(x)
    if N < 64:
        raise ValueError("Window too short for wavelet coherence.")

    # log-spaced frequencies
    freqs = np.exp(np.linspace(np.log(fmin), np.log(fmax), n_freq))

    # ---- Complex CWT via linear convolution (complex FFT) ----
    def cwt_linear(sig: np.ndarray) -> np.ndarray:
        sig = np.asarray(sig, float)
        Wx = []
        for f0 in freqs:
            # build complex Morlet in time
            dur = max(2.0, 8.0/f0)            # a few cycles
            L = int(np.ceil(dur*fs))
            if L % 2 == 0: L += 1
            tt = (np.arange(-(L//2), L//2+1))/fs
            sigma_t = w0/(2*np.pi*f0)
            mw = np.exp(-0.5*(tt/sigma_t)**2) * np.exp(1j*2*np.pi*f0*tt)
            mw -= mw.mean()
            mw /= (np.sqrt(np.sum(np.abs(mw)**2)) + 1e-24)

            # linear convolution via FFT, center-trim to length N
            n_lin = N + L - 1
            n_fft = int(2**np.ceil(np.log2(n_lin)))
            S = np.fft.fft(sig, n=n_fft)
            H = np.fft.fft(mw,  n=n_fft)
            conv = np.fft.ifft(S*H)[:n_lin]      # complex
            start = (L - 1)//2
            Wx.append(conv[start:start+N])
        return np.array(Wx)                      # (n_freq, N)

    Wx = cwt_linear(x)
    Wy = cwt_linear(y)

    # spectral smoothing along time (small Hann)
    def smooth(A: np.ndarray, wlen: int = 9) -> np.ndarray:
        if wlen <= 1: return A
        w = np.hanning(wlen); w /= w.sum()
        return np.apply_along_axis(lambda m: np.convolve(m, w, mode='same'), axis=1, arr=A)

    Sxx = np.abs(Wx)**2; Syy = np.abs(Wy)**2; Sxy = Wx * np.conj(Wy)
    Sxx_s, Syy_s, Sxy_s = smooth(Sxx), smooth(Syy), smooth(Sxy)
    WTC = (np.abs(Sxy_s)**2) / (Sxx_s * Syy_s + 1e-24)

    # ---- Circular-shift null on y ----
    rng = np.random.default_rng(11)
    null_max = []
    for _ in range(n_perm):
        sh = int(rng.integers(max(1, int(0.1*fs)), N - max(1, int(0.1*fs))))
        y_sh = np.r_[y[-sh:], y[:-sh]]
        Wy_s = cwt_linear(y_sh)
        Sxy0 = Wx * np.conj(Wy_s)
        WTC0 = (np.abs(smooth(Sxy0))**2) / (Sxx_s * (np.abs(smooth(Wy_s))**2) + 1e-24)
        null_max.append(np.nanmax(WTC0))
    thresh = float(np.nanpercentile(null_max, 95))
    sig = WTC >= thresh

    # ---- Plot (optional) ----
    if show or out_png:
        t = np.arange(N)/fs
        plt.figure(figsize=(10, 4))
        extent = [t[0], t[-1], freqs[0], freqs[-1]]
        plt.imshow(WTC, aspect='auto', origin='lower', extent=extent, cmap='magma',
                   vmin=0, vmax=np.nanmax(WTC))
        cb = plt.colorbar(); cb.set_label('Wavelet coherence')
        # highlight significant pixels (cluster visual)
        G = nx.grid_2d_graph(WTC.shape[0], WTC.shape[1])
        mask_idx = set(zip(*np.where(sig)))
        visited=set()
        for node in list(mask_idx):
            if node in visited: continue
            stack=[node]; comp=[]
            while stack:
                u=stack.pop()
                if u in visited or u not in mask_idx: continue
                visited.add(u); comp.append(u)
                for v in G.neighbors(u):
                    if v in mask_idx and v not in visited: stack.append(v)
            if comp:
                yy, xx = zip(*comp)
                plt.scatter(t[np.array(xx)], freqs[np.array(yy)], s=2, c='cyan', alpha=0.6)
        plt.xlabel('Time (s)'); plt.ylabel('Frequency (Hz)')
        plt.title('EEG–SR Wavelet Coherence (cyan: > null 95%)')
        plt.tight_layout()
        if out_png: plt.savefig(out_png, dpi=140)
        if show: plt.show()
        plt.close()

    return {'WTC': WTC, 'freqs': freqs, 'thresh': thresh, 'sig_mask': sig}


# ---------------- Sliding coherence @ 7.83 Hz ----------------
def sliding_coherence_f0(df, eeg_channel, sr_channel, ignition_windows,
    f0, half, time_col='Timestamp',
    win_sec=8.0, step_sec=1.0,
    n_null=200, show=False,
    fast_mode=False, max_sec=None, max_windows=None):

    # --- Convert time to seconds from start ---
    tcol = df[time_col]
    if np.issubdtype(tcol.dtype, np.number):
        t_sec = tcol.values.astype(float)
        t_sec = t_sec - t_sec[0]
    else:
        t_dt = pd.to_datetime(tcol)
        t_sec = (t_dt - t_dt.iloc[0]).dt.total_seconds().values
    df['t_sec'] = t_sec


    # --- REMOVE hidden caps. Only crop if explicitly requested ---
    if fast_mode and (max_sec is None and max_windows is None):
        max_sec = 40.0 # opt-in fast behavior; otherwise no crop


    if max_sec is not None:
        df = df[df['t_sec'] <= max_sec]
    if max_windows is not None:
        # subsample evenly to at most max_windows centers later
        pass # (implement only if you *really* need it)


    # --- compute sliding coherence over the WHOLE (possibly cropped) df ---
    # estimate fs robustly
    dt = np.median(np.diff(df['t_sec'].values))
    fs = 1.0/float(dt)


    # centers from win/2 to T-win/2, step = step_sec
    T = df['t_sec'].iloc[-1]
    centers = np.arange(win_sec/2, max(0, T - win_sec/2) + 1e-9, step_sec)


    # compute coherence at f0 for each center (pseudo)
    coh_vals = []
    for c in centers:
        t0, t1 = c - win_sec/2, c + win_sec/2
        w = (df['t_sec'] >= t0) & (df['t_sec'] < t1)
        xe = df.loc[w, eeg_channel].values
        xs = df.loc[w, sr_channel].values
        if len(xe) < int(0.8*win_sec*fs) or len(xs) < int(0.8*win_sec*fs):
            coh_vals.append(np.nan); continue
        # bandpass around f0 ± half (your own filter or multitaper)
        # compute magnitude-squared coherence at f0 (your method)
        coh_vals.append(compute_coherence_at_f0(xe, xs, fs, f0, half))


    coh = np.asarray(coh_vals)
    # build null via surrogates (existing code)
    null95 = build_null_threshold(coh, n_null=n_null) # your existing routine


    return {
    't': centers,
    'coh': coh,
    'null95': null95,
    'fs': fs
    }

def compute_coherence_at_f0(xe, xs, fs, f0, half):
    """
    Magnitude-squared coherence between signals xe and xs at target frequency f0.

    Uses Welch autospectra (Pxx, Pyy) and cross-spectrum (Pxy) on the provided
    windowed segments, then returns a scalar coherence value aggregated within
    the band [f0 - half, f0 + half]. If that band contains no frequency bin,
    returns the coherence at the nearest available bin to f0.

    Parameters
    ----------
    xe : array_like
        First signal segment (e.g., EEG), 1-D.
    xs : array_like
        Second signal segment (e.g., SR proxy / magnetometer), 1-D.
    fs : float
        Sampling rate in Hz.
    f0 : float
        Target center frequency in Hz.
    half : float
        Half-bandwidth in Hz; analyze [f0 - half, f0 + half].

    Returns
    -------
    float
        Coherence (0..1) at/around f0.
    """
    xe = np.asarray(xe, dtype=float)
    xs = np.asarray(xs, dtype=float)
    N = int(min(len(xe), len(xs)))
    if N < 32:
        raise ValueError("Window too short for coherence (N < 32 samples)")
    xe = xe[:N]
    xs = xs[:N]

    # Choose segment length for spectral estimates: use half the window (typical)
    nperseg = int(max(32, min(N, N // 2)))
    noverlap = int(nperseg // 2)

    # Autospectra and cross-spectrum (Welch / CSD)
    f, Pxx = signal.welch(
        xe, fs=fs, window='hann', nperseg=nperseg, noverlap=noverlap,
        detrend='constant', return_onesided=True, scaling='density'
    )
    _, Pyy = signal.welch(
        xs, fs=fs, window='hann', nperseg=nperseg, noverlap=noverlap,
        detrend='constant', return_onesided=True, scaling='density'
    )
    _, Pxy = signal.csd(
        xe, xs, fs=fs, window='hann', nperseg=nperseg, noverlap=noverlap,
        detrend='constant', return_onesided=True, scaling='density'
    )

    # Magnitude-squared coherence
    eps = 1e-20
    Cxy = (np.abs(Pxy) ** 2) / (Pxx * Pyy + eps)

    # Aggregate within the f0 ± half band; fallback to nearest bin if empty
    f_lo = max(0.0, float(f0) - float(half))
    f_hi = float(f0) + float(half)
    band = (f >= f_lo) & (f <= f_hi) & np.isfinite(Cxy)
    if np.any(band):
        c_val = float(np.nanmean(Cxy[band]))
    else:
        idx = int(np.argmin(np.abs(f - float(f0))))
        c_val = float(Cxy[idx]) if np.isfinite(Cxy[idx]) else float('nan')

    # Clamp numeric noise into [0, 1]
    if np.isfinite(c_val):
        c_val = float(np.clip(c_val, 0.0, 1.0))
    return c_val

def build_null_threshold(coh, n_null=200, method='block', block_len=None, alpha=0.05, random_state=13):
    """
    Estimate a null threshold for a sliding coherence trace by resampling the
    coherence sequence itself.

    For each surrogate, we create a bootstrap replica of the coherence
    time-series and take its maximum. The (1-alpha) percentile of these maxima
    is returned. By default, we use a block-bootstrap that preserves local
    autocorrelation structure; set method='iid' for simple i.i.d. resampling.

    Parameters
    ----------
    coh : array_like
        Coherence values (0..1) across sliding-window centers; NaNs allowed.
    n_null : int, optional
        Number of surrogate replicates (default 200).
    method : {'block','iid'}, optional
        Resampling strategy. 'block' preserves short-range correlations.
    block_len : int or None, optional
        Block length (in samples) for block-bootstrap. If None, uses ~5% of N
        (at least 5 samples).
    alpha : float, optional
        Significance level (default 0.05 → 95th percentile).
    random_state : int, optional
        RNG seed for reproducibility.

    Returns
    -------
    float
        Estimated (1 - alpha) percentile of the null maxima; clipped to [0, 1].
    """

    c = np.asarray(coh, dtype=float)
    # Keep only finite values
    c = c[np.isfinite(c)]
    if c.size == 0:
        return float('nan')

    rng = np.random.default_rng(random_state)
    N = int(c.size)

    maxima = []
    if method == 'iid':
        for _ in range(int(n_null)):
            samp = rng.choice(c, size=N, replace=True)
            maxima.append(float(np.nanmax(samp)))
    else:
        # Block bootstrap with circular wrap to preserve local structure
        if block_len is None:
            block_len = max(5, int(round(N / 20)))  # ~5% of the series
        B = int(block_len)
        for _ in range(int(n_null)):
            idx = []
            filled = 0
            while filled < N:
                start = int(rng.integers(0, N))
                end = start + B
                if end <= N:
                    idx.extend(range(start, end))
                else:
                    # circular wrap
                    idx.extend(list(range(start, N)) + list(range(0, end - N)))
                filled += B
            idx = np.asarray(idx[:N], dtype=int)
            surrogate = c[idx]
            maxima.append(float(np.nanmax(surrogate)))

    q = 100.0 * (1.0 - float(alpha))
    thr = float(np.nanpercentile(maxima, q))
    # numeric safety
    return float(np.clip(thr, 0.0, 1.0))

# ---------------- Orchestrator ----------------
def run_eeg_schumann_coherence(
    RECORDS: pd.DataFrame,
    eeg_channels: List[str],
    sr_channel: str,
    ignition_windows: Optional[List[Tuple[float,float]]] = None,
    baseline_windows: Optional[List[Tuple[float,float]]] = None,
    time_col: str = 'Timestamp',
    out_dir: str = 'exports_eeg_sr/session',
    show: bool = True,
    harmonics: Tuple[float,...] = (7.83,14.3,20.8,27.3,33.8)
) -> Dict[str, object]:
    _ensure_dir(out_dir)
    # ---------- 1) Harmonic MSC bars with 95% null ----------
    tbl_ign = msc_harmonics_table(RECORDS, eeg_channels, sr_channel, ignition_windows,
                                  time_col=time_col, harmonics=harmonics, nperseg_sec=4.0, n_null=200)
    tbl_ign.to_csv(os.path.join(out_dir,'msc_harmonics_ign.csv'), index=False)
    if baseline_windows:
        tbl_bas = msc_harmonics_table(RECORDS, eeg_channels, sr_channel, baseline_windows,
                                      time_col=time_col, harmonics=harmonics, nperseg_sec=4.0, n_null=200)
        tbl_bas.to_csv(os.path.join(out_dir,'msc_harmonics_base.csv'), index=False)
    else:
        tbl_bas = None

    # bar plot (ignition)
    if show or True:
    # ---------- 1b) Build pivot & pick nearest-to-7.83 column ONCE ----------
        pivot = tbl_ign.pivot(index='channel', columns='freq', values='MSC')
        pthr  = tbl_ign.pivot(index='channel', columns='freq', values='null95')

        # convert column labels to float array & find nearest to target
        cols_raw = list(pivot.columns)
        try:
            cols = np.array(cols_raw, dtype=float)
        except Exception:
            cols = np.array([float(c) for c in cols_raw])

        if cols.size == 0:
            raise ValueError("No frequency columns in pivot; check input data / windows.")

        target = float(harmonics[0])
        j_near = int(np.nanargmin(np.abs(cols - target)))
        col_near_val = float(cols[j_near])  # for labelling only

        # Best channel at the nearest-to-7.83 bin (by POSITION, not label)
        ch_best = pivot.iloc[:, j_near].idxmax()

        # ---------- 1c) Bar plot (ignition) ----------
        if show or True:
            x = np.arange(len(cols))
            chans = list(pivot.index)
            w = 0.8 / max(1, len(chans))

            plt.figure(figsize=(min(10, 2.0 + 0.5*len(chans)), 3.2))
            for i, ch in enumerate(chans):
                vals = pivot.loc[ch, :].to_numpy()
                plt.bar(x + (i - (len(chans)-1)/2.0)*w, vals, width=w, label=ch)

            # overlay null95 (black dots) for each frequency column (by POSITION)
            for j in range(len(cols)):
                thr = pthr.iloc[:, j].to_numpy()
                plt.scatter(np.full_like(thr, x[j]), thr, s=12, c='k', zorder=5)

            plt.xticks(x, [f"{c:.2f}" for c in cols])
            plt.ylabel('MSC')
            plt.title('EEG–SR Coherence at Schumann Harmonics (Ignition)\n(black dots: shift-null 95%)')
            plt.legend(fontsize=8, ncol=min(4, len(chans)))
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, 'msc_harmonics_bars_ign.png'), dpi=140)
            if show: plt.show()
            plt.close()

    # ---------- 2) WTC using the nearest-to-7.83 best channel ----------
    wtc = wavelet_coherence_tf(RECORDS, ch_best, sr_channel, time_col=time_col,
                               fmin=4, fmax=40, n_freq=64, w0=6.0,
                               n_perm=200, alpha=0.05,
                               wins=ignition_windows, show=show,
                               out_png=os.path.join(out_dir, f'wtc_{ch_best}_ign.png'))

    # ---------- 3) Sliding coherence at ~7.83 Hz ----------
    sl = sliding_coherence_f0(RECORDS, ch_best, sr_channel, ignition_windows,
                              f0=harmonics[0], half=0.6, time_col=time_col,
                              win_sec=8.0, step_sec=1.0, n_null=200, show=show)

    # ---------- 4) Summary ----------
    summary = {
        'best_channel': ch_best,
        'nearest_bin_to_7p83_Hz': col_near_val,
        'mean_MSC_nearest_bin_ign': float(pivot.iloc[:, j_near].mean()),
        'WTC_thresh': float(wtc['thresh']),
        'sliding_null95': float(sl['null95'])
    }
    pd.DataFrame([summary]).to_csv(os.path.join(out_dir,'summary.csv'), index=False)
    return {'summary': summary, 'msc_ign': tbl_ign, 'msc_base': tbl_bas, 'wtc': wtc, 'sliding': sl, 'out_dir': out_dir}
