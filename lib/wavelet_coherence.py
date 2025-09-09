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

# ---- Complex CWT via linear convolution (complex FFT) ----
def cwt_linear(freqs,sig: np.ndarray,w0,N,fs=128) -> np.ndarray:
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

# spectral smoothing along time (small Hann)
def smooth(A: np.ndarray, wlen: int = 9) -> np.ndarray:
    if wlen <= 1: return A
    w = np.hanning(wlen); w /= w.sum()
    return np.apply_along_axis(lambda m: np.convolve(m, w, mode='same'), axis=1, arr=A)

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

    

    Wx = cwt_linear(freqs,x,w0,N)
    Wy = cwt_linear(freqs,y,w0,N)

    

    Sxx = np.abs(Wx)**2; Syy = np.abs(Wy)**2; Sxy = Wx * np.conj(Wy)
    Sxx_s, Syy_s, Sxy_s = smooth(Sxx), smooth(Syy), smooth(Sxy)
    WTC = (np.abs(Sxy_s)**2) / (Sxx_s * Syy_s + 1e-24)

    # ---- Circular-shift null on y ----
    rng = np.random.default_rng(11)
    null_max = []
    for _ in range(n_perm):
        sh = int(rng.integers(max(1, int(0.1*fs)), N - max(1, int(0.1*fs))))
        y_sh = np.r_[y[-sh:], y[:-sh]]
        Wy_s = cwt_linear(freqs,y_sh,w0,N)
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


def plot_sr_ignition_wtc_strip(RECORDS,
    eeg_channel: str,                 # e.g., 'EEG.O1' (or your best posterior)
    sr_channel: str,                  # magnetometer if you have one; else posterior proxy
    ignition_windows: list,           # [(t0, t1), ...] in seconds
    time_col: str = 'Timestamp',
    fmin: float = 0.5, fmax: float = 59.8, n_freq: int = 64,
    harmonics=(7.83, 14.3, 20.8, 27.3, 33.8),   # show bands for these (you can add more)
    half_band: float = 0.6,           # ±Hz shading around each harmonic
    w0: float = 6.0,                  # Morlet parameter
    n_perm: int = 200, alpha: float = 0.05,
    out_png: str = 'sr_wtc_strip.png',
    show: bool = True
):

    # 1) Run WTC on the *full* record (so ignition spans align to absolute time)
    wtc = wavelet_coherence_tf(
        RECORDS, eeg_channel, sr_channel,
        time_col=time_col, fmin=fmin, fmax=fmax, n_freq=n_freq, w0=w0,
        n_perm=n_perm, alpha=alpha, wins=None, show=True, out_png=None
    )

    # 2) Build the time axis from the DataFrame
    t_all = np.asarray(pd.to_numeric(RECORDS[time_col], errors='coerce').values, float)
    # If wavelet_coherence_tf returned a trimmed/sliced length, map to the last N samples
    N = wtc['WTC'].shape[1]
    if len(t_all) != N:
        # use the last N timestamps to match WTC length
        t = t_all[-N:]
    else:
        t = t_all

    # 3) Plot WTC with cyan significant pixels and ignition shading
    plt.figure(figsize=(11, 4))
    extent = [t[0], t[-1], wtc['freqs'][0], wtc['freqs'][-1]]

    plt.imshow(
        wtc['WTC'], aspect='auto', origin='lower', extent=extent,
        cmap='magma', vmin=0, vmax=np.nanmax(wtc['WTC'])
    )
    cb = plt.colorbar(); cb.set_label('Wavelet coherence')

    # Cyan significant pixels (cluster-based, mask already thresholded vs shift-null)
    sig = wtc['sig_mask']
    if sig is not None and np.any(sig):
        yy, xx = np.where(sig)
        plt.scatter(t[xx], wtc['freqs'][yy], s=4, c='cyan', alpha=0.7, label='> null 95%')

    # Horizontal harmonic bands (±half_band)
    for h in harmonics:
        plt.axhspan(h - half_band, h + half_band, color='white', alpha=0.08)
        plt.axhline(h, color='white', lw=0.8, alpha=0.6)

    # Shade ignition windows
    for (t0, t1) in ignition_windows:
        plt.axvspan(t0, t1, color='k', alpha=0.08)

    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title(f'EEG–SR Wavelet Coherence (cyan = > null 95%; shaded = ignition)')
    if sig is not None and np.any(sig):
        plt.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    if out_png:
        plt.savefig(out_png, dpi=140)
    if show:
        plt.show()
    plt.close()
