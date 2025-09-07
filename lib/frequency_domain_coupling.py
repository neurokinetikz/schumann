"""
Frequency-domain coupling (core): Multi-taper MSC + Wavelet Coherence (WTC)
fs is inferred from RECORDS[time_col] (or override easily).
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import signal
from scipy.signal.windows import dpss
import matplotlib.pyplot as plt
import networkx as nx
from scipy.signal.windows import hann

# ----------------- small utils -----------------

def infer_fs(RECORDS: pd.DataFrame, time_col: str = 'Timestamp') -> float:
    t = np.asarray(pd.to_numeric(RECORDS[time_col], errors='coerce').values, float)
    dt = np.diff(t)
    dt = dt[(dt > 0) & np.isfinite(dt)]
    if dt.size == 0:
        raise ValueError("Cannot infer sampling rate from time column.")
    return float(1.0 / np.median(dt))

def get_series(RECORDS: pd.DataFrame, name: str) -> np.ndarray:
    """Return signal column. Accepts 'EEG.O1' or bare 'O1' (will try 'EEG.O1')."""
    if name in RECORDS.columns:
        s = pd.to_numeric(RECORDS[name], errors='coerce').fillna(0.0).values
        return np.asarray(s, float)
    alt = 'EEG.' + name
    if alt in RECORDS.columns:
        s = pd.to_numeric(RECORDS[alt], errors='coerce').fillna(0.0).values
        return np.asarray(s, float)
    raise ValueError(f"Signal '{name}' not found in RECORDS.")

def slice_concat(x: np.ndarray, fs: float, windows: Optional[List[Tuple[float,float]]]) -> np.ndarray:
    """Concatenate [t0,t1] windows (sec) from x; if windows is None/empty, return x."""
    if not windows:
        return x.copy()
    segs = []
    n = x.size
    for (t0, t1) in windows:
        i0, i1 = int(round(t0*fs)), int(round(t1*fs))
        i0 = max(0, i0); i1 = min(n, i1)
        if i1 > i0:
            segs.append(x[i0:i1])
    return np.concatenate(segs) if segs else x.copy()

# ----------------- (a) Multi-taper MSC -----------------

def _mtm_cross_spectra(x: np.ndarray, y: np.ndarray, fs: float, half_bw_hz: float,
                       n_fft: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Thomson multi-taper auto/cross spectra with jackknife (leave-one-taper):
    returns (f, Coh, lo, hi, Sxx_all, Syy_all, Sxy_all).
    """
    N = min(x.size, y.size)
    x = x[:N]
    y = y[:N]
    if n_fft is None:
        n_fft = int(2 ** np.ceil(np.log2(N)))

    T = N / fs
    # Thomson: W = NW/T, we target W ≈ half_bw_hz ⇒ NW ≈ half_bw_hz * T
    NW = max(2.0, half_bw_hz * T)
    K = int(max(3, np.floor(2 * NW - 1)))  # K ≈ 2NW−1
    tapers, eigs = dpss(N, NW=NW, Kmax=K, return_ratios=True)
    w = eigs / np.sum(eigs)

    Xk = np.fft.rfft((tapers * x[:, None]).T, n=n_fft)  # (K, n_freq)
    Yk = np.fft.rfft((tapers * y[:, None]).T, n=n_fft)

    Sxxk = (np.abs(Xk) ** 2)
    Syyk = (np.abs(Yk) ** 2)
    Sxyk = (Xk * np.conj(Yk))

    # Weighted averages (full estimate)
    Sxx = np.tensordot(w, Sxxk, axes=(0, 0))
    Syy = np.tensordot(w, Syyk, axes=(0, 0))
    Sxy = np.tensordot(w, Sxyk, axes=(0, 0))
    Coh = (np.abs(Sxy) ** 2) / (Sxx * Syy + 1e-24)

    # Jackknife leave-one-taper for Coh CIs
    K = int(K)
    jk = []
    for k in range(K):
        idx = [i for i in range(K) if i != k]
        ww = w[idx] / np.sum(w[idx])
        Sxx_l = np.tensordot(ww, Sxxk[idx], axes=(0, 0))
        Syy_l = np.tensordot(ww, Syyk[idx], axes=(0, 0))
        Sxy_l = np.tensordot(ww, Sxyk[idx], axes=(0, 0))
        jk.append((np.abs(Sxy_l) ** 2) / (Sxx_l * Syy_l + 1e-24))
    jk = np.stack(jk, axis=0)
    mu = np.mean(jk, axis=0)
    se = np.sqrt((K - 1) * np.mean((jk - mu) ** 2, axis=0))
    lo = np.clip(Coh - 1.96 * se, 0, 1)
    hi = np.clip(Coh + 1.96 * se, 0, 1)

    f = np.fft.rfftfreq(n_fft, d=1 / fs)
    return f, Coh, lo, hi, Sxx, Syy, Sxy

def run_multitaper_msc_harmonics(RECORDS: pd.DataFrame,x_channels: List[str],y_channel: str,windows: Optional[List[Tuple[float, float]]] = None,time_col: str = 'Timestamp',half_bw_hz: float = 3.0,harmonics: List[float] = (7.83, 14.3, 20.8, 27.3, 33.8),n_fft: Optional[int] = None,) -> Dict[str, object]:
    """
    MSC between mean of x_channels and y_channel over 'windows'.
    Reports coherence at given Schumann harmonics + jackknife CIs and phase/lag.
    """
    fs = infer_fs(RECORDS, time_col)
    X = np.nanmean(np.vstack([get_series(RECORDS, ch) for ch in x_channels]), axis=0)
    Y = get_series(RECORDS, y_channel)

    Xw = slice_concat(X, fs, windows)
    Yw = slice_concat(Y, fs, windows)
    N = min(Xw.size, Yw.size)
    Xw = Xw[:N]
    Yw = Yw[:N]

    f, Coh, lo, hi, Sxx, Syy, Sxy = _mtm_cross_spectra(Xw, Yw, fs, half_bw_hz, n_fft=n_fft)

    phase = np.angle(Sxy)                         # radians
    lag_sec = np.where(f > 0, phase / (2 * np.pi * f + 1e-24), np.nan)

    rows = []
    for hf in harmonics:
        idx = int(np.argmin(np.abs(f - hf)))
        rows.append({
            'freq': float(f[idx]),
            'MSC': float(Coh[idx]),
            'lo': float(lo[idx]),
            'hi': float(hi[idx]),
            'phase_rad': float(phase[idx]),
            'lag_ms': float(lag_sec[idx] * 1000.0),
        })
    return {'f': f, 'MSC': Coh, 'lo': lo, 'hi': hi, 'phase': phase, 'lag_sec': lag_sec,
            'harmonics_table': pd.DataFrame(rows)}

# ----------------- (b) Wavelet coherence (WTC) -----------------

def run_wavelet_coherence(
    RECORDS: pd.DataFrame,
    x_channel: str,
    y_channel: str,
    time_col: str = 'Timestamp',
    fmin: float = 4.0,
    fmax: float = 40.0,
    n_freq: int = 64,
    w0: float = 6.0,
    n_perm: int = 200,
    p_cluster: float = 0.05,
    show: bool = True,
) -> Dict[str, object]:
    """
    Morlet wavelet coherence with cluster-based permutation correction.
    Returns WTC map, threshold, and significant clusters.
    """
    fs = infer_fs(RECORDS, time_col)
    x = get_series(RECORDS, x_channel)
    y = get_series(RECORDS, y_channel)
    t = np.asarray(pd.to_numeric(RECORDS[time_col], errors='coerce').values, float)

    freqs = np.exp(np.linspace(np.log(fmin), np.log(fmax), n_freq))

    def cwt(sig: np.ndarray) -> np.ndarray:
        N = sig.size
        n_fft = int(2 ** np.ceil(np.log2(N * 2)))
        S = np.fft.rfft(sig, n=n_fft)
        Wx = []
        for f0 in freqs:
            dur = max(2.0, 10.0 / f0)  # several cycles
            L = int(np.ceil(dur * fs))
            L += (L % 2 == 0)
            tt = (np.arange(-(L // 2), L // 2 + 1)) / fs
            sigma_t = w0 / (2 * np.pi * f0)
            mw = np.exp(-0.5 * (tt / sigma_t) ** 2) * np.exp(1j * 2 * np.pi * f0 * tt)
            mw -= np.mean(mw)
            mw /= np.sqrt(np.sum(np.abs(mw) ** 2))
            H = np.fft.rfft(mw, n=n_fft)
            conv = np.fft.irfft(S * H, n=n_fft)[:N]
            Wx.append(conv)
        return np.array(Wx)  # (n_freq, N)

    Wx = cwt(x)
    Wy = cwt(y)
    Sxx = np.abs(Wx) ** 2
    Syy = np.abs(Wy) ** 2
    Sxy = Wx * np.conj(Wy)

    def smooth(A: np.ndarray, wlen: int = 7) -> np.ndarray:
        if wlen <= 1:
            return A
        w = np.hanning(wlen)
        w /= w.sum()
        return np.apply_along_axis(lambda m: np.convolve(m, w, mode='same'), axis=1, arr=A)

    Sxx_s = smooth(Sxx)
    Syy_s = smooth(Syy)
    Sxy_s = smooth(Sxy)
    WTC = np.abs(Sxy_s) ** 2 / (Sxx_s * Syy_s + 1e-24)

    # Cluster-based permutation: circularly shift y
    thresh = None
    clusters = None
    if n_perm > 0:
        rng = np.random.default_rng(123)
        null_max = []
        for _ in range(n_perm):
            shift = rng.integers(low=int(0.1 * fs), high=x.size - int(0.1 * fs))
            y_sh = np.r_[y[-shift:], y[:-shift]]
            Wy_s = cwt(y_sh)
            Sxy0 = Wx * np.conj(Wy_s)
            Syy0 = np.abs(Wy_s) ** 2
            Sxy0_s = smooth(Sxy0)
            Syy0_s = smooth(Syy0)
            WTC0 = np.abs(Sxy0_s) ** 2 / (Sxx_s * Syy0_s + 1e-24)
            null_max.append(np.nanmax(WTC0))
        thresh = np.percentile(null_max, 100 * (1 - p_cluster))
        sig = WTC >= thresh
        # 8-connectivity clustering on TF grid
        G = nx.grid_2d_graph(WTC.shape[0], WTC.shape[1])
        mask_idx = set(zip(*np.where(sig)))
        clusters = []
        visited = set()
        for node in mask_idx:
            if node in visited:
                continue
            stack = [node]
            c = []
            while stack:
                u = stack.pop()
                if u in visited or u not in mask_idx:
                    continue
                visited.add(u)
                c.append(u)
                for v in G.neighbors(u):
                    if v in mask_idx and v not in visited:
                        stack.append(v)
            clusters.append(c)

    if show:
        plt.figure(figsize=(10, 4))
        extent = [t[0], t[-1], freqs[0], freqs[-1]]
        plt.imshow(WTC, aspect='auto', origin='lower', extent=extent, cmap='magma')
        cbar = plt.colorbar()
        cbar.set_label('Wavelet coherence')
        plt.ylabel('Frequency (Hz)')
        plt.xlabel('Time (s)')
        plt.title('WTC (Morlet)')
        plt.tight_layout()
        plt.show()

    return {'t': t, 'freqs': freqs, 'WTC': WTC, 'thresh': thresh, 'clusters': clusters}

# # ----------------- example usage -----------------
# if __name__ == "__main__":
#     # Example (requires a RECORDS DataFrame in scope)
#     # 1) MSC at harmonics:
#     # msc = run_multitaper_msc_harmonics(RECORDS,
#     #         x_channels=['EEG.O1','EEG.O2'], y_channel='EEG.O1',
#     #         windows=[(290,310),(580,600)], time_col='Timestamp',
#     #         half_bw_hz=3.0)
#     # print(msc['harmonics_table'])

#     # 2) Wavelet coherence map:
#     # wtc = run_wavelet_coherence(RECORDS, 'EEG.O1', 'EEG.O2',
#     #         time_col='Timestamp', fmin=4, fmax=40, n_freq=64, w0=6.0,
#     #         n_perm=200, p_cluster=0.05, show=True)
#     pass


def _mtm_cross_spectra(x: np.ndarray,
                       y: np.ndarray,
                       fs: float,
                       half_bw_hz: float,
                       n_fft: Optional[int] = None
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Thomson multi-taper auto/cross spectra with jackknife CIs.
    Returns (f, Coh, lo, hi, Sxx_all, Syy_all, Sxy_all).
    """
    # --- prep ---
    N = min(x.size, y.size)
    x = x[:N]
    y = y[:N]
    if n_fft is None:
        n_fft = int(2 ** np.ceil(np.log2(N)))

    T = N / fs
    # Target half-bandwidth W (Hz) -> NW = W*T
    NW = max(2.0, half_bw_hz * T)
    K = int(max(3, np.floor(2 * NW - 1)))   # K ≈ 2NW - 1

    # DPSS returns (K, N)
    tapers, eigs = dpss(N, NW=NW, Kmax=K, return_ratios=True)
    if tapers.ndim == 1:                     # guard if K==1 edge case
        tapers = tapers[None, :]
        eigs = np.array([1.0])
        K = 1
    w = eigs / np.sum(eigs)

    # --- taper & FFT along time axis ---
    # Shapes: (K, N); broadcast x[None,:] to (1, N) => (K, N)
    Xk = np.fft.rfft(tapers * x[None, :], n=n_fft, axis=1)   # (K, n_freq)
    Yk = np.fft.rfft(tapers * y[None, :], n=n_fft, axis=1)

    # Per-taper spectra
    Sxxk = (np.abs(Xk) ** 2)                # (K, n_freq)
    Syyk = (np.abs(Yk) ** 2)
    Sxyk = (Xk * np.conj(Yk))

    # Weighted (eigenvalue) averages over K tapers
    Sxx = np.tensordot(w, Sxxk, axes=(0, 0))     # (n_freq,)
    Syy = np.tensordot(w, Syyk, axes=(0, 0))
    Sxy = np.tensordot(w, Sxyk, axes=(0, 0))
    Coh = (np.abs(Sxy) ** 2) / (Sxx * Syy + 1e-24)

    # --- jackknife leave-one-taper CI on coherence ---
    jk = []
    for k in range(K):
        idx = [i for i in range(K) if i != k]
        ww = w[idx] / np.sum(w[idx])
        Sxx_l = np.tensordot(ww, Sxxk[idx], axes=(0, 0))
        Syy_l = np.tensordot(ww, Syyk[idx], axes=(0, 0))
        Sxy_l = np.tensordot(ww, Sxyk[idx], axes=(0, 0))
        C_l = (np.abs(Sxy_l) ** 2) / (Sxx_l * Syy_l + 1e-24)
        jk.append(C_l)
    jk = np.stack(jk, axis=0)               # (K, n_freq)
    mu = np.mean(jk, axis=0)
    se = np.sqrt((K - 1) * np.mean((jk - mu) ** 2, axis=0))
    lo = np.clip(Coh - 1.96 * se, 0, 1)
    hi = np.clip(Coh + 1.96 * se, 0, 1)

    f = np.fft.rfftfreq(n_fft, d=1 / fs)
    return f, Coh, lo, hi, Sxx, Syy, Sxy



def plot_msc_harmonics_table(msc_table: pd.DataFrame,
                             ax: Optional[plt.Axes] = None,
                             label: str = 'state',
                             color: str = 'tab:blue',
                             title: Optional[str] = None) -> plt.Axes:
    """
    Plot MSC with jackknife CIs at harmonic lines from a single run_multitaper_msc_harmonics result.
    Expects columns: ['freq','MSC','lo','hi'] (others are ignored for plotting).
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 3.2))
    df = msc_table.copy()
    df = df.sort_values('freq')
    x = np.arange(len(df))
    y = df['MSC'].values
    lo = df['lo'].values
    hi = df['hi'].values
    yerr = np.vstack([y - lo, hi - y])

    ax.bar(x, y, width=0.6, color=color, alpha=0.85, label=label)
    ax.errorbar(x, y, yerr=yerr, fmt='none', ecolor='k', capsize=2, lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{f:.2f}" for f in df['freq'].values])
    ax.set_ylabel('MSC')
    ax.set_xlabel('Frequency (Hz)')
    if title:
        ax.set_title(title)
    ax.set_ylim(0, 1.0)
    ax.legend()
    ax.grid(axis='y', alpha=0.25)
    return ax

def plot_msc_harmonics_compare(msc_ign: Dict[str, object],
                               msc_base: Dict[str, object],
                               title: str = 'Multi-taper MSC at Schumann Harmonics (Ignition vs Baseline)') -> None:
    """
    Side-by-side bars with jackknife CIs comparing ignition vs baseline.
    Pass the dicts returned by run_multitaper_msc_harmonics (for ignition and baseline).
    """
    # Join on (rounded) frequency to align bins cleanly
    t_ign = msc_ign['harmonics_table'].copy()
    t_base = msc_base['harmonics_table'].copy()
    t_ign['f_round'] = t_ign['freq'].round(2)
    t_base['f_round'] = t_base['freq'].round(2)
    freqs = sorted(np.unique(np.r_[t_ign['f_round'].values, t_base['f_round'].values]))

    # extract in aligned order
    ign_vals, ign_lo, ign_hi = [], [], []
    bas_vals, bas_lo, bas_hi = [], [], []
    for f in freqs:
        if f in t_ign['f_round'].values:
            r = t_ign.set_index('f_round').loc[f]
            ign_vals.append(float(r['MSC'])); ign_lo.append(float(r['lo'])); ign_hi.append(float(r['hi']))
        else:
            ign_vals.append(np.nan); ign_lo.append(np.nan); ign_hi.append(np.nan)
        if f in t_base['f_round'].values:
            r = t_base.set_index('f_round').loc[f]
            bas_vals.append(float(r['MSC'])); bas_lo.append(float(r['lo'])); bas_hi.append(float(r['hi']))
        else:
            bas_vals.append(np.nan); bas_lo.append(np.nan); bas_hi.append(np.nan)

    x = np.arange(len(freqs))
    w = 0.38
    fig, ax = plt.subplots(1, 1, figsize=(9, 3.2))
    # ignition bars + CIs
    ax.bar(x - w/2, ign_vals, width=w, label='Ignition', color='tab:blue', alpha=0.9)
    ign_yerr = np.vstack([np.array(ign_vals) - np.array(ign_lo),
                          np.array(ign_hi) - np.array(ign_vals)])
    ax.errorbar(x - w/2, ign_vals, yerr=ign_yerr, fmt='none', ecolor='k', capsize=2, lw=1)

    # baseline bars + CIs
    ax.bar(x + w/2, bas_vals, width=w, label='Baseline', color='tab:orange', alpha=0.9)
    bas_yerr = np.vstack([np.array(bas_vals) - np.array(bas_lo),
                          np.array(bas_hi) - np.array(bas_vals)])
    ax.errorbar(x + w/2, bas_vals, yerr=bas_yerr, fmt='none', ecolor='k', capsize=2, lw=1)

    ax.set_xticks(x); ax.set_xticklabels([f"{f:.2f}" for f in freqs])
    ax.set_ylabel('MSC')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_title(title)
    ax.set_ylim(0, 1.0)
    ax.legend()
    ax.grid(axis='y', alpha=0.25)
    plt.tight_layout()
    plt.show()

# ----------------- example of using the helpers -----------------
# msc_ign = run_multitaper_msc_harmonics(RECORDS, ['EEG.O1','EEG.O2'], 'EEG.O1',
#                                        windows=[(290,310),(580,600)], time_col='Timestamp', half_bw_hz=3.0)
# msc_base = run_multitaper_msc_harmonics(RECORDS, ['EEG.O1','EEG.O2'], 'EEG.O1',
#                                         windows=None, time_col='Timestamp', half_bw_hz=3.0)
# plot_msc_harmonics_compare(msc_ign, msc_base)
# plot_msc_harmonics_table(msc_ign['harmonics_table'], title='Ignition only', label='Ignition', color='tab:blue')


"""
Frequency-domain coupling (add-on):
(a) Phase-Locking Value (PLV) at Schumann harmonics with phase lead/lag and optional topography
(b) Cross-correlograms of band-limited envelopes (EEG θ≈7.8 Hz vs Schumann amplitude) with bootstrap CIs
(c) Cyclostationary Spectral Correlation (SCF) at Schumann cyclic frequencies (robust to noise)

Assumes a pandas.DataFrame RECORDS with a time column (default 'Timestamp')
and signal columns like 'EEG.O1','EEG.O2', or a magnetometer reference.
"""


# -------------------- generic helpers --------------------

def infer_fs(RECORDS: pd.DataFrame, time_col: str = 'Timestamp') -> float:
    t = np.asarray(pd.to_numeric(RECORDS[time_col], errors='coerce').values, float)
    dt = np.diff(t); dt = dt[(dt > 0) & np.isfinite(dt)]
    if dt.size == 0:
        raise ValueError("Cannot infer sampling rate from time column.")
    return float(1.0 / np.median(dt))

def get_series(RECORDS: pd.DataFrame, name: str) -> np.ndarray:
    """Return a numeric signal array. Accepts 'EEG.O1' or bare 'O1'."""
    if name in RECORDS.columns:
        x = pd.to_numeric(RECORDS[name], errors='coerce').fillna(0.0).values
        return np.asarray(x, float)
    alt = 'EEG.' + name
    if alt in RECORDS.columns:
        x = pd.to_numeric(RECORDS[alt], errors='coerce').fillna(0.0).values
        return np.asarray(x, float)
    raise ValueError(f"Signal '{name}' not found in RECORDS.")

def slice_concat(x: np.ndarray, fs: float, windows: Optional[List[Tuple[float, float]]]) -> np.ndarray:
    """Concatenate [t0,t1] windows in seconds; if None/empty, return full signal."""
    if not windows:
        return x.copy()
    segs = []
    n = x.size
    for (t0, t1) in windows:
        i0, i1 = int(round(t0 * fs)), int(round(t1 * fs))
        i0 = max(0, i0); i1 = min(n, i1)
        if i1 > i0:
            segs.append(x[i0:i1])
    return np.concatenate(segs) if segs else x.copy()

def bandpass(x: np.ndarray, fs: float, f1: float, f2: float, order: int = 4) -> np.ndarray:
    ny = 0.5 * fs
    f1 = max(1e-6, min(f1, ny * 0.99))
    f2 = max(f1 + 1e-6, min(f2, ny * 0.999))
    b, a = signal.butter(order, [f1 / ny, f2 / ny], btype='band')
    return signal.filtfilt(b, a, x)

# -------------------- (a) PLV at harmonics --------------------

def plv_and_mean_phase(eeg: np.ndarray,
                       sr: np.ndarray,
                       fs: float,
                       center_hz: float,
                       half_bw_hz: float = 0.6) -> Tuple[float, float]:
    """
    Compute PLV and circular mean phase difference between EEG and SR at a narrow band centered at center_hz.
    Returns (PLV, mean_phase_rad), where mean_phase_rad is arg(E[exp(i*Δφ)]).
    """
    x = bandpass(eeg, fs, center_hz - half_bw_hz, center_hz + half_bw_hz)
    y = bandpass(sr,  fs, center_hz - half_bw_hz, center_hz + half_bw_hz)
    zx = signal.hilbert(x); zy = signal.hilbert(y)
    dphi = np.angle(zx) - np.angle(zy)
    plv = np.abs(np.mean(np.exp(1j * dphi)))
    mean_phase = np.angle(np.mean(np.exp(1j * dphi)))
    return float(plv), float(mean_phase)

def run_plv_harmonics_topography(RECORDS: pd.DataFrame,
                                 eeg_channels: List[str],
                                 sr_channel: str,
                                 harmonics: List[float] = (7.83, 14.3, 20.8, 27.3, 33.8),
                                 half_bw_hz: float = 0.6,
                                 windows: Optional[List[Tuple[float, float]]] = None,
                                 time_col: str = 'Timestamp'
                                 ) -> Dict[str, object]:
    """
    Compute PLV and mean phase for each EEG channel vs Schumann reference at Schumann harmonics.
    Returns per-harmonic DataFrame with columns: ['channel','freq','PLV','mean_phase_rad','lag_ms'].
    lag_ms is derived from mean_phase / (2π f).
    """
    fs = infer_fs(RECORDS, time_col)
    sr = get_series(RECORDS, sr_channel)
    sr = slice_concat(sr, fs, windows)
    rows = []
    for ch in eeg_channels:
        x = get_series(RECORDS, ch)
        x = slice_concat(x, fs, windows)
        for f0 in harmonics:
            plv, mean_ph = plv_and_mean_phase(x, sr, fs, f0, half_bw_hz=half_bw_hz)
            lag_ms = (mean_ph / (2 * np.pi * f0 + 1e-24)) * 1000.0
            rows.append({'channel': ch if ch in RECORDS.columns else 'EEG.' + ch,
                         'freq': float(f0),
                         'PLV': plv,
                         'mean_phase_rad': mean_ph,
                         'lag_ms': lag_ms})
    table = pd.DataFrame(rows)
    return {'table': table}

def plot_plv_topography(plv_table: pd.DataFrame,
                        chan_pos: Dict[str, Tuple[float, float]],
                        freq: float,
                        vmin: float = 0.0, vmax: float = 1.0,
                        title: Optional[str] = None) -> None:
    """
    Simple 2D scatter topography for a single harmonic:
    chan_pos: dict like {'O1':(x,y),'O2':(x,y), ...}. Keys may be 'O1' or 'EEG.O1' — function handles both.
    """
    df = plv_table[plv_table['freq'] == freq]
    xs, ys, vals = [], [], []
    for _, r in df.iterrows():
        label = r['channel'].split('.', 1)[-1]
        if label in chan_pos:
            xs.append(chan_pos[label][0])
            ys.append(chan_pos[label][1])
            vals.append(r['PLV'])
    if not xs:
        raise ValueError("No channels from plv_table found in chan_pos.")
    plt.figure(figsize=(4.5, 4))
    sc = plt.scatter(xs, ys, c=vals, s=200, cmap='viridis', vmin=vmin, vmax=vmax, edgecolor='k')
    plt.colorbar(sc, label='PLV')
    plt.title(title or f'PLV topography @ {freq:.2f} Hz')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# -------------------- (b) Cross-correlograms of band-limited envelopes --------------------

def xcorr_envelopes_peaklag(eeg: np.ndarray,
                            sr: np.ndarray,
                            fs: float,
                            center_hz: float = 7.83,
                            half_bw_hz: float = 0.6,
                            max_lag_sec: float = 2.0) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Band-limit both to [center±half_bw], take Hilbert envelopes and compute normalized cross-correlogram.
    Returns (peak_lag_ms, lags_ms, xcorr).
    """
    xe = bandpass(eeg, fs, center_hz - half_bw_hz, center_hz + half_bw_hz)
    ye = bandpass(sr,  fs, center_hz - half_bw_hz, center_hz + half_bw_hz)
    xe = np.abs(signal.hilbert(xe))
    ye = np.abs(signal.hilbert(ye))
    xe = (xe - xe.mean()) / (xe.std() + 1e-12)
    ye = (ye - ye.mean()) / (ye.std() + 1e-12)
    max_lag = int(round(max_lag_sec * fs))
    xcorr = signal.correlate(xe, ye, mode='full') / len(xe)
    lags = signal.correlation_lags(len(xe), len(ye), mode='full')
    keep = (np.abs(lags) <= max_lag)
    lags = lags[keep]; xcorr = xcorr[keep]
    k = int(np.argmax(xcorr))
    peak_lag_ms = (lags[k] / fs) * 1000.0
    return float(peak_lag_ms), (lags / fs) * 1000.0, xcorr

def bootstrap_peaklag_ci(eeg: np.ndarray,
                         sr: np.ndarray,
                         fs: float,
                         center_hz: float = 7.83,
                         half_bw_hz: float = 0.6,
                         max_lag_sec: float = 2.0,
                         n_boot: int = 500,
                         rng_seed: int = 3) -> Dict[str, object]:
    """
    Circular-shift bootstrap null for peak lag. Returns {'peak_ms':..., 'ci':(lo,hi), 'lags_ms':..., 'xcorr':...}.
    """
    peak_ms, lags_ms, xcorr = xcorr_envelopes_peaklag(eeg, sr, fs, center_hz, half_bw_hz, max_lag_sec)
    rng = np.random.default_rng(rng_seed)
    xe = bandpass(eeg, fs, center_hz - half_bw_hz, center_hz + half_bw_hz)
    ye = bandpass(sr,  fs, center_hz - half_bw_hz, center_hz + half_bw_hz)
    xe = np.abs(signal.hilbert(xe))
    ye = np.abs(signal.hilbert(ye))
    n = len(ye)
    null_peaks = []
    for _ in range(n_boot):
        s = int(rng.integers(low=int(0.1 * fs), high=n - int(0.1 * fs)))
        ye_sh = np.r_[ye[-s:], ye[:-s]]
        p, _, _ = xcorr_envelopes_peaklag(xe, ye_sh, fs, center_hz, half_bw_hz, max_lag_sec)
        null_peaks.append(p)
    lo, hi = np.nanpercentile(null_peaks, [2.5, 97.5])
    return {'peak_ms': peak_ms, 'ci': (float(lo), float(hi)),
            'lags_ms': lags_ms, 'xcorr': xcorr, 'null_peaks': np.asarray(null_peaks)}

# -------------------- (c) Cyclostationary Spectral Correlation (SCF) --------------------

def scf_cyclic_periodogram(x: np.ndarray,
                           fs: float,
                           alpha_hz: float,
                           nperseg: int = 2048,
                           noverlap: int = 1024) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple cyclic periodogram SCF estimator at cyclic frequency alpha (Hz):
    SCF(f; alpha) ~ E[ X(f+alpha/2) * X*(f-alpha/2) ] across segments.

    Returns (f, SCF_complex). Use |SCF| or normalize if desired.
    """
    x = np.asarray(x, float)
    step = nperseg - noverlap
    n_fft = int(2 ** np.ceil(np.log2(nperseg)))
    df = fs / n_fft
    k_alpha = int(round(alpha_hz / df))
    # segment & window
    win = signal.hann(nperseg, sym=False)
    segments = []
    for start in range(0, len(x) - nperseg + 1, step):
        seg = x[start:start + nperseg] * win
        X = np.fft.rfft(seg, n=n_fft)  # bins 0..n_fft/2
        segments.append(X)
    if not segments:
        raise ValueError("Signal too short for SCF windowing.")
    Xall = np.stack(segments, axis=0)  # (n_seg, n_freq)
    n_freq = Xall.shape[1]
    # freq indices for f±alpha/2 need half-bin offsets; use nearest bins
    # For each usable bin i with i-k_alpha/2 >=0 and i+k_alpha/2 < n_freq
    # To keep integer indexing, require even k_alpha; relax by rounding half-steps:
    # Build pairs (i_plus, i_minus) s.t. i_plus - i_minus ≈ k_alpha
    pairs = []
    for i in range(k_alpha, n_freq - k_alpha):
        im = i - k_alpha // 2
        ip = i + (k_alpha - k_alpha // 2)
        if 0 <= im < n_freq and 0 <= ip < n_freq:
            pairs.append((im, ip))
    if not pairs:
        raise ValueError("alpha too small/large for current FFT resolution; increase nperseg.")
    # SCF(f; alpha) as average over segments: X(f+alpha/2) * conj(X(f-alpha/2))
    SCF = np.zeros(len(pairs), dtype=complex)
    for idx, (im, ip) in enumerate(pairs):
        SCF[idx] = np.mean(Xall[:, ip] * np.conj(Xall[:, im]))
    f = np.fft.rfftfreq(n_fft, d=1 / fs)[:len(pairs)]
    return f, SCF

def scf_at_harmonics(RECORDS: pd.DataFrame,
                     channel: str,
                     harmonics: List[float] = (7.83, 14.3, 20.8, 27.3, 33.8),
                     windows: Optional[List[Tuple[float, float]]] = None,
                     time_col: str = 'Timestamp',
                     nperseg: int = 4096,
                     noverlap: int = 2048
                     ) -> Dict[str, object]:
    """
    Compute SCF magnitude integrated over frequency for each Schumann cyclic frequency alpha.
    Returns table with |SCF|_int and peak |SCF(f; alpha)| frequency.
    """
    fs = infer_fs(RECORDS, time_col)
    x = get_series(RECORDS, channel)
    x = slice_concat(x, fs, windows)
    rows = []
    scf_maps = {}
    for a in harmonics:
        f, SCF = scf_cyclic_periodogram(x, fs, a, nperseg=nperseg, noverlap=noverlap)
        mag = np.abs(SCF)
        peak_idx = int(np.argmax(mag))
        rows.append({'alpha_hz': a,
                     'SCF_int': float(np.trapz(mag, f)),
                     'SCF_peak_f': float(f[peak_idx]),
                     'SCF_peak_mag': float(mag[peak_idx])})
        scf_maps[a] = {'f': f, 'SCF': SCF}
    table = pd.DataFrame(rows)
    return {'table': table, 'maps': scf_maps}

# -------------------- example usage --------------------
if __name__ == "__main__":
    # 1) PLV at harmonics (with optional topography if you provide chan_pos)
    # plv_res = run_plv_harmonics_topography(RECORDS,
    #             eeg_channels=['EEG.O1','EEG.O2','EEG.Pz'],
    #             sr_channel='EEG.O1',    # or a magnetometer reference
    #             harmonics=[7.83,14.3,20.8,27.3,33.8],
    #             half_bw_hz=0.6,
    #             windows=[(290,310),(580,600)])
    # print(plv_res['table'])
    # # Optional topography plot for a single harmonic:
    # # chan_pos = {'O1':(-0.4,-0.8),'O2':(0.4,-0.8),'Pz':(0, -0.4)}  # example positions
    # # plot_plv_topography(plv_res['table'], chan_pos, freq=7.83, title='PLV @ 7.83 Hz')

    # 2) Cross-correlogram of envelopes (θ≈7.8 Hz)
    # fs = infer_fs(RECORDS)
    # eeg = get_series(RECORDS, 'EEG.O1')
    # sr  = get_series(RECORDS, 'EEG.O2')
    # boot = bootstrap_peaklag_ci(eeg, sr, fs, center_hz=7.83, half_bw_hz=0.6, max_lag_sec=2.0, n_boot=500)
    # print('Peak lag (ms):', boot['peak_ms'], '95% CI:', boot['ci'])

    # 3) Cyclostationary spectral correlation at harmonics
    # scf = scf_at_harmonics(RECORDS, 'EEG.O1', harmonics=[7.83,14.3,20.8,27.3,33.8],
    #                        windows=[(290,310),(580,600)])
    # print(scf['table'])
    pass

def scf_cyclic_periodogram_demod(x: np.ndarray,
                                 fs: float,
                                 alpha_hz: float,
                                 nperseg: int = 4096,
                                 noverlap: int = 2048,
                                 n_fft: Optional[int] = None
                                 ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cyclostationary spectral correlation via complex demodulation:
      x_plus(t)  = x(t) * exp(-j 2π α/2 t)
      x_minus(t) = x(t) * exp(+j 2π α/2 t)
      SCF(f; α) ~= E[ X_plus(f) * conj( X_minus(f) ) ]  (Welch average over segments)

    Returns:
      f   : frequency axis (Hz)
      SCF : complex SCF(f; α) (use np.abs / np.angle as needed)
    """
    x = np.asarray(x, float)
    N = len(x)
    if n_fft is None:
        n_fft = int(2**np.ceil(np.log2(nperseg)))

    # Complex demodulation (global, not per-seg) – avoids segment phase offsets
    t = np.arange(N)/fs
    half = alpha_hz / 2.0
    x_plus  = x * np.exp(-1j * 2*np.pi*half * t)
    x_minus = x * np.exp(+1j * 2*np.pi*half * t)

    # Welch-style segmentation
    step = nperseg - noverlap
    win  = signal.hann(nperseg, sym=False)
    Wnorm = np.sum(win**2) + 1e-24

    S_acc = None
    n_seg = 0
    for start in range(0, N - nperseg + 1, step):
        seg_p = x_plus[start:start+nperseg]  * win
        seg_m = x_minus[start:start+nperseg] * win
        Xp = np.fft.rfft(seg_p, n=n_fft)
        Xm = np.fft.rfft(seg_m, n=n_fft)
        # segment SCF contribution
        Sseg = Xp * np.conj(Xm)
        if S_acc is None:
            S_acc = Sseg
        else:
            S_acc += Sseg
        n_seg += 1

    if n_seg == 0:
        raise ValueError("Signal too short for chosen nperseg/noverlap.")

    SCF = S_acc / (n_seg * Wnorm)          # normalize by window power and #segments
    f = np.fft.rfftfreq(n_fft, d=1/fs)
    return f, SCF


def scf_at_harmonics(RECORDS: pd.DataFrame,
                     channel: str,
                     harmonics: List[float] = (7.83, 14.3, 20.8, 27.3, 33.8),
                     windows: Optional[List[Tuple[float, float]]] = None,
                     time_col: str = 'Timestamp',
                     nperseg: int = 4096,
                     noverlap: int = 2048
                     ) -> Dict[str, object]:
    """
    Compute SCF magnitude integrated over f for each Schumann cyclic α using demodulation SCF.
    Returns a table with integrated |SCF|, and per-α SCF maps.
    """
    fs = infer_fs(RECORDS, time_col)
    x = get_series(RECORDS, channel)
    # if you used windows previously, keep it – it just trims x
    x = slice_concat(x, fs, windows)
    rows = []
    scf_maps = {}
    for a in harmonics:
        f, SCF = scf_cyclic_periodogram_demod(x, fs, a, nperseg=nperseg, noverlap=noverlap)
        mag = np.abs(SCF)
        peak_idx = int(np.argmax(mag))
        rows.append({'alpha_hz': a,
                     'SCF_int': float(np.trapz(mag, f)),
                     'SCF_peak_f': float(f[peak_idx]),
                     'SCF_peak_mag': float(mag[peak_idx])})
        scf_maps[a] = {'f': f, 'SCF': SCF}
    table = pd.DataFrame(rows)
    return {'table': table, 'maps': scf_maps}

def scf_cyclic_periodogram_demod(x: np.ndarray,
                                 fs: float,
                                 alpha_hz: float,
                                 nperseg: int = 4096,
                                 noverlap: int = 2048,
                                 n_fft: Optional[int] = None
                                 ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cyclostationary spectral correlation via complex demodulation:
      x_plus(t)  = x(t) * exp(-j 2π α/2 t)
      x_minus(t) = x(t) * exp(+j 2π α/2 t)
      SCF(f; α) ≈ Welch average of X_plus(f) * conj(X_minus(f))
    Returns (f>=0, SCF_complex(f; α)).
    """
    x = np.asarray(x, float)
    N = len(x)
    if n_fft is None:
        n_fft = int(2**np.ceil(np.log2(nperseg)))

    t = np.arange(N) / fs
    half = alpha_hz / 2.0
    x_plus  = x * np.exp(-1j * 2*np.pi * half * t)  # complex
    x_minus = x * np.exp(+1j * 2*np.pi * half * t)  # complex

    step = nperseg - noverlap
    win  = hann(nperseg, sym=False)
    Wnorm = np.sum(win**2) + 1e-24

    S_acc = None
    n_seg = 0
    for start in range(0, N - nperseg + 1, step):
        seg_p = x_plus[start:start+nperseg]  * win
        seg_m = x_minus[start:start+nperseg] * win
        # Complex FFT (not rfft)
        Xp = np.fft.fft(seg_p, n=n_fft)
        Xm = np.fft.fft(seg_m, n=n_fft)
        Sseg = Xp * np.conj(Xm)            # segment SCF contribution
        if S_acc is None:
            S_acc = Sseg
        else:
            S_acc += Sseg
        n_seg += 1

    if n_seg == 0:
        raise ValueError("Signal too short for chosen nperseg/noverlap.")

    SCF_full = S_acc / (n_seg * Wnorm)
    f_full   = np.fft.fftfreq(n_fft, d=1/fs)

    # Keep nonnegative frequencies for a one-sided view
    pos = f_full >= 0
    return f_full[pos], SCF_full[pos]

