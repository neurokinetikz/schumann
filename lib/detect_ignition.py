# Key patches applied to your last version:
# 1) Harmonics can be passed explicitly (harmonics_hz + harmonic_bw_hz).
# 2) Safer R(t): edge-window skip + low-power rejection + t0_net-centered zR_max.
# 3) SR envelope outputs per event: sr_z_max, sr_z_mean_pm5, sr_z_mean_post5.
# 4) Gamma PEL band clamped to Nyquist; per-session valid_harmonics filtered.
# 5) ETA aligned to t0_net with robust SEM.
# 6) Print block: original summary preserved + new metrics appended.
#
# Paste the code below into your working module to replace the function
# and helpers. If you already integrated earlier patches, this version
# is drop-in compatible.

from __future__ import annotations
import os, json, re
from typing import Optional, List, Tuple, Dict, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import welch
from scipy.stats import zscore

# FOOOF-based harmonic detection
try:
    from lib.fooof_harmonics import detect_harmonics_fooof
    FOOOF_AVAILABLE = True
except ImportError:
    FOOOF_AVAILABLE = False
    print("Warning: fooof_harmonics not available. Using PSD-only harmonic detection.")

# ---------- small utilities ----------

def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def _merge_intervals_int(it: List[Tuple[int,int]]) -> List[Tuple[int,int]]:
    if not it:
        return []
    it = sorted(it)
    out = [it[0]]
    for a,b in it[1:]:
        la, lb = out[-1]
        if a <= lb:
            out[-1] = (la, max(lb, b))
        else:
            out.append((a,b))
    return out


def ensure_timestamp_column(df: pd.DataFrame, time_col: str = 'Timestamp', default_fs: float = 128.0) -> str:
    if time_col in df.columns:
        return time_col
    n = len(df)
    df[time_col] = np.arange(n)/float(default_fs)
    return time_col


def infer_fs(df: pd.DataFrame, time_col: str) -> float:
    t = pd.to_numeric(df[time_col], errors='coerce').values.astype(float)
    dt = np.diff(t[np.isfinite(t)])
    dt = dt[dt > 0]
    if dt.size == 0:
        return 128.0
    return float(np.round(1.0/np.median(dt)))


def get_series(df: pd.DataFrame, col: str) -> np.ndarray:
    return pd.to_numeric(df[col], errors='coerce').values.astype(float)

# --- bandpass with safety ---

def _safe_band(f_lo, f_hi, fs, pad_frac=1e-3):
    nyq = fs/2.0
    pad = pad_frac*nyq
    lo = max(pad, min(f_lo, nyq - 2*pad))
    hi = max(lo + pad, min(f_hi, nyq - pad))
    return lo, hi


def bandpass_safe(x: np.ndarray, fs: float, f1: float, f2: float, order=4) -> np.ndarray:
    f1, f2 = _safe_band(f1, f2, fs)
    ny = 0.5*fs
    b,a = signal.butter(order, [f1/ny, f2/ny], btype='band')
    return signal.filtfilt(b,a,x,axis=-1)


def _sr_envelope_z_series(y: np.ndarray, fs: float, f0: float,
                          half_bw: float, smooth_sec: float) -> np.ndarray:
    """Envelope z-score for a monaural SR band."""
    yb = bandpass_safe(y, fs, f0 - half_bw, f0 + half_bw)
    env = np.abs(signal.hilbert(yb))
    n_smooth = max(1, int(round(smooth_sec * fs)))
    if n_smooth > 1:
        w = np.hanning(n_smooth)
        w /= w.sum()
        env = np.convolve(env, w, mode='same')
    return zscore(env, nan_policy='omit')


def _scalar_bandwidth(val, default: float = 0.5) -> float:
    if val is None:
        return float(default)
    arr = np.atleast_1d(val)
    if arr.size == 0:
        return float(default)
    try:
        return float(arr.flat[0])
    except (TypeError, ValueError):
        return float(default)

# --- MSC helper (time-resolved coherence at f0) ---

def _msc_f0_series(x: np.ndarray, y: np.ndarray, fs: float, f0: float, win: float = 1.0, step: float = 0.1) -> np.ndarray:
    """Return a short-window MSC time series at f0 for signals x,y.
    x,y: 1-D arrays; win/step in seconds.
    """
    nwin = int(round(win*fs)); nstep = int(round(step*fs))
    if nwin <= 1 or len(x) < nwin or len(y) < nwin:
        return np.array([])
    vals = []
    for i in range(0, min(len(x),len(y)) - nwin + 1, nstep):
        segx = x[i:i+nwin]; segy = y[i:i+nwin]
        f, C = signal.coherence(segx, segy, fs=fs, nperseg=nwin)
        vals.append(C[np.argmin(np.abs(f - f0))])
    return np.array(vals, float) if vals else np.array([])

# ---------- Virtual SR reference builders ----------

def _ssd_weights(X: np.ndarray, fs: float, f0: float, bw: float = 0.4, flank: float = 1.0) -> np.ndarray:
    Bs = bandpass_safe(X, fs, f0-bw, f0+bw)
    N1 = bandpass_safe(X, fs, max(0.1, f0-bw-flank), f0-bw)
    N2 = bandpass_safe(X, fs, f0+bw, f0+bw+flank)
    Cs = np.cov(Bs)
    Cn = np.cov(np.hstack([N1, N2]))
    eigvals, eigvecs = np.linalg.eig(np.linalg.pinv(Cn) @ Cs)
    w = eigvecs[:, np.argmax(eigvals.real)].real
    w /= (np.linalg.norm(w) + 1e-12)
    return w


def _plv_weights(X: np.ndarray, fs: float, f_lo: float, f_hi: float) -> np.ndarray:
    Xb = bandpass_safe(X, fs, f_lo, f_hi)
    ph = np.angle(signal.hilbert(Xb, axis=-1))
    n = X.shape[0]
    plv = np.zeros((n, n))
    for i in range(n):
        dphi = ph[i:i+1] - ph
        plv[i] = np.abs(np.mean(np.exp(1j*dphi), axis=1))
    w = plv.mean(axis=1)
    w = w / (w.sum() + 1e-12)
    return w


def _pca_reference(X: np.ndarray, fs: float, f_lo: float, f_hi: float) -> Tuple[np.ndarray, np.ndarray]:
    Xb = bandpass_safe(X, fs, f_lo, f_hi)
    U, S, Vt = np.linalg.svd(Xb.T, full_matrices=False)
    c1 = (U[:, 0] * S[0])
    w = Vt[0, :]
    w = w / (np.linalg.norm(w) + 1e-12)
    return w, c1


def _build_virtual_sr(X: np.ndarray, fs: float, f0: float, bw: float, mode: str = 'auto-SSD') -> Tuple[np.ndarray, np.ndarray]:
    f_lo, f_hi = f0-bw, f0+bw
    if mode == 'auto-SSD':
        w = _ssd_weights(X, fs, f0, bw)
        v = w @ X
    elif mode == 'auto-PLV':
        w = _plv_weights(X, fs, f_lo, f_hi)
        v = (w[:, None] * X).sum(axis=0)
    elif mode == 'auto-PCA':
        w, v = _pca_reference(X, fs, f_lo, f_hi)
    else:
        raise ValueError("sr_reference must be 'auto-SSD' | 'auto-PLV' | 'auto-PCA'")
    return v, w


def _msc_per_channel_vs_median(X: np.ndarray, fs: float, freqs: List[float], bw: float) -> List[float]:
    """
    Compute mean MSC across channels using chart-style method:
    Each channel vs median reference, averaged across channels.

    This matches the visualization MSC calculation in test.py::_msc_matrix().
    """
    n_ch = X.shape[0]
    ref = np.nanmedian(X, axis=0)  # Median across channels as reference

    msc_modes: List[float] = []
    for f0 in freqs:
        # Bandpass filter at target frequency
        f_lo, f_hi = f0 - bw, f0 + bw
        Xb = bandpass_safe(X, fs, f_lo, f_hi)
        ref_b = bandpass_safe(ref, fs, f_lo, f_hi)

        # Compute MSC for each channel vs median reference
        msc_vals = []
        for ci in range(n_ch):
            ch = Xb[ci]
            # Hilbert-based MSC (same as test.py::_msc_channel_to_reference)
            z_ch = signal.hilbert(ch)
            z_ref = signal.hilbert(ref_b)
            num = np.abs(np.mean(z_ch * np.conj(z_ref))) ** 2
            den = (np.mean(np.abs(z_ch) ** 2) * np.mean(np.abs(z_ref) ** 2)) + 1e-12
            msc_vals.append(float(num / den))

        # Average MSC across channels
        mean_msc = float(np.nanmean(msc_vals)) if msc_vals else np.nan
        msc_modes.append(mean_msc)

    return msc_modes


def _spectral_slope_during_event(X: np.ndarray, fs: float,
                                   band: Tuple[float, float] = (3, 45),
                                   exclude_centers: Optional[List[float]] = None,
                                   exclude_bw: float = 0.6) -> float:
    """
    Compute mean 1/f spectral slope across channels during event.

    Returns the spectral slope (beta) where Power ~ f^(-beta).
    More negative values indicate steeper 1/f falloff.

    Broadband artifacts flatten the slope (less negative).
    True SR ignitions maintain or steepen the slope.
    """
    slopes = []
    for ch in range(X.shape[0]):
        f, P = signal.welch(X[ch], fs=fs, nperseg=min(4096, int(2*fs)))
        mask = (f >= band[0]) & (f <= band[1])

        # Exclude SR frequencies to isolate background 1/f
        if exclude_centers:
            for fc in exclude_centers:
                mask &= ~((f >= fc - exclude_bw) & (f <= fc + exclude_bw))

        if np.sum(mask) < 6:
            continue

        x = np.log10(f[mask] + 1e-12)
        y = 10 * np.log10(P[mask] + 1e-20)

        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            continue

        slope, _ = np.polyfit(x, y, 1)
        slopes.append(slope)

    return float(np.nanmean(slopes)) if slopes else np.nan


def _frequency_specificity_index(X: np.ndarray, fs: float,
                                   sr_freqs: List[float],
                                   bw: float = 0.5,
                                   band: Tuple[float, float] = (3, 45)) -> float:
    """
    Compute Frequency Specificity Index (FSI).

    FSI = power_in_SR_bands / total_power

    High FSI (>0.4) → power concentrated at SR frequencies (true ignition)
    Low FSI (<0.3) → broadband contamination (artifact)
    """
    X_mean = np.mean(X, axis=0)

    # Total power in band
    X_band = bandpass_safe(X_mean, fs, band[0], band[1])
    P_total = np.mean(X_band**2)

    # Power in SR bands
    P_sr = 0.0
    for f0 in sr_freqs:
        X_sr = bandpass_safe(X_mean, fs, f0 - bw, f0 + bw)
        P_sr += np.mean(X_sr**2)

    FSI = P_sr / (P_total + 1e-12)
    return float(np.clip(FSI, 0, 1))


def _msc_bandwidth_specificity(X: np.ndarray, fs: float,
                                 sr_freqs: List[float],
                                 bw: float = 0.5,
                                 offset_hz: float = 2.0) -> float:
    """
    Compute MSC Bandwidth Specificity Ratio (BSR).

    BSR = msc_at_SR_freqs / msc_at_offset_freqs

    High BSR (>2) → MSC specific to SR frequencies (true ignition)
    Low BSR (<1.5) → MSC elevated across all frequencies (artifact)
    """
    ref = np.nanmedian(X, axis=0)

    msc_sr_list = []
    msc_off_list = []

    for f0 in sr_freqs:  # Use all specified harmonics
        # MSC at SR frequency
        X_sr = bandpass_safe(X, fs, f0 - bw, f0 + bw)
        ref_sr = bandpass_safe(ref, fs, f0 - bw, f0 + bw)
        msc_sr_ch = []
        for ch in range(X.shape[0]):
            z_ch = signal.hilbert(X_sr[ch])
            z_ref = signal.hilbert(ref_sr)
            num = np.abs(np.mean(z_ch * np.conj(z_ref))) ** 2
            den = (np.mean(np.abs(z_ch)**2) * np.mean(np.abs(z_ref)**2)) + 1e-12
            msc_sr_ch.append(num / den)
        msc_sr_list.append(np.nanmean(msc_sr_ch))

        # MSC at offset frequency (f0 + offset_hz)
        f_off = f0 + offset_hz
        if f_off + bw < fs / 2:  # Check Nyquist limit
            X_off = bandpass_safe(X, fs, f_off - bw, f_off + bw)
            ref_off = bandpass_safe(ref, fs, f_off - bw, f_off + bw)
            msc_off_ch = []
            for ch in range(X.shape[0]):
                z_ch = signal.hilbert(X_off[ch])
                z_ref = signal.hilbert(ref_off)
                num = np.abs(np.mean(z_ch * np.conj(z_ref))) ** 2
                den = (np.mean(np.abs(z_ch)**2) * np.mean(np.abs(z_ref)**2)) + 1e-12
                msc_off_ch.append(num / den)
            msc_off_list.append(np.nanmean(msc_off_ch))

    if not msc_sr_list or not msc_off_list:
        return np.nan

    BSR = np.nanmean(msc_sr_list) / (np.nanmean(msc_off_list) + 1e-6)
    return float(BSR)


# ---------- Kuramoto R(t) & t0 detection (safer) ----------

def _kuramoto_R_timeseries(X, fs, f_lo, f_hi, win_sec=1.0, step_sec=0.25, edge_sec=2.0, min_rms=1e-7):
    Xb = bandpass_safe(X, fs, f_lo, f_hi)
    ph = np.angle(signal.hilbert(Xb, axis=-1))
    n = X.shape[1]
    w = max(1, int(round(win_sec*fs)))
    s = max(1, int(round(step_sec*fs)))
    edge = int(round(edge_sec*fs))
    times, R = [], []
    for i0 in range(edge, n - w - edge, s):
        seg_ph = ph[:, i0:i0+w]
        rms = np.sqrt(np.mean(Xb[:, i0:i0+w]**2))
        if rms < min_rms:
            Rt = np.nan
        else:
            Rt = np.abs(np.mean(np.exp(1j*seg_ph), axis=0)).mean()
        R.append(Rt)
        times.append((i0 + w//2)/fs)
    return np.array(times), np.array(R, float)


def _detect_t0_from_R(times: np.ndarray, R: np.ndarray, thresh: float = 0.6) -> float:
    if times.size == 0:
        return np.nan
    dR = np.gradient(R, times)
    mask = R >= thresh
    if np.any(mask):
        idxs = np.where(mask)[0]
        idx = idxs[np.nanargmax(dR[idxs])]
    else:
        idx = int(np.nanargmax(dR))
    return float(times[idx])

# ---------- Latencies / propagation ----------

def _channel_latencies(
    X: np.ndarray,
    fs: float,
    f_lo: float,
    f_hi: float,
    t0: float,
    pre: float = 2.0,
    post: float = 1.0,
    z_th: float = 2.0,
    *,
    min_run_sec: float = 0.08,
    smooth_sec: float = 0.10,
    return_details: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Return per-channel latency estimates (and optional diagnostics) near t0.

    min_run_sec: require the z-threshold to hold for at least this duration.
    smooth_sec : moving-average smoothing of the Hilbert envelope to suppress spikes.
    """
    Xb = bandpass_safe(X, fs, f_lo, f_hi)
    amp = np.abs(signal.hilbert(Xb, axis=-1))

    if smooth_sec and smooth_sec > 0:
        n_smooth = max(1, int(round(smooth_sec * fs)))
        if n_smooth > 1:
            kernel = np.ones(n_smooth, dtype=float)
            kernel /= kernel.sum()
            amp = np.vstack([np.convolve(row, kernel, mode='same') for row in amp])

    n = X.shape[1]
    t0_idx = int(np.clip(round(t0 * fs), 0, max(0, n - 1)))
    pre_samp = int(round(pre * fs))
    post_samp = int(round(post * fs))
    i0 = max(0, t0_idx - pre_samp)
    i1 = min(n, t0_idx + post_samp)
    if i1 <= i0:
        i1 = min(n, i0 + max(1, pre_samp + post_samp))

    base = amp[:, i0:t0_idx] if t0_idx > i0 else amp[:, max(0, t0_idx - pre_samp):t0_idx]
    if base.size == 0 or base.shape[1] < max(5, int(0.2 * fs)):
        alt_start = max(0, t0_idx - max(pre_samp, int(0.5 * fs)))
        base = amp[:, alt_start:t0_idx] if t0_idx > alt_start else amp[:, :max(1, t0_idx)]

    mu = np.nanmedian(base, axis=1, keepdims=True)
    mad = np.nanmedian(np.abs(base - mu), axis=1, keepdims=True)
    sigma = 1.4826 * mad
    fallback_sd = np.nanstd(base, axis=1, keepdims=True)
    sigma = np.where(
        (sigma <= 1e-9) | (~np.isfinite(sigma)),
        fallback_sd + 1e-12,
        sigma + 1e-12
    )

    window = amp[:, i0:i1]
    z = (window - mu) / sigma

    lats = np.full(X.shape[0], np.nan)
    rise_idx = np.full(X.shape[0], -1, dtype=int)
    run_samples = max(1, int(round(min_run_sec * fs)))
    ones = np.ones(run_samples, dtype=int)

    for ch in range(X.shape[0]):
        row = z[ch]
        if not np.any(np.isfinite(row)):
            continue
        mask = np.isfinite(row) & (row >= z_th)
        idx_candidate = None
        if np.any(mask):
            if run_samples > 1 and mask.size >= run_samples:
                conv = np.convolve(mask.astype(int), ones, mode='valid')
                hits = np.flatnonzero(conv >= run_samples)
                if hits.size:
                    idx_candidate = int(hits[0])
            if idx_candidate is None:
                hits = np.flatnonzero(mask)
                if hits.size:
                    idx_candidate = int(hits[0])
        if idx_candidate is not None:
            lats[ch] = (i0 + idx_candidate) / fs
            rise_idx[ch] = idx_candidate

    peak_z = np.full(X.shape[0], np.nan)
    rise_z = np.full(X.shape[0], np.nan)
    for ch in range(X.shape[0]):
        row = z[ch]
        if np.any(np.isfinite(row)):
            peak_z[ch] = float(np.nanmax(row))
        if rise_idx[ch] >= 0 and rise_idx[ch] < row.size and np.isfinite(row[rise_idx[ch]]):
            rise_z[ch] = float(row[rise_idx[ch]])

    if return_details:
        return lats, peak_z, rise_z
    return lats


def _granger_bivariate_matrix(X: np.ndarray, maxlag: int = 6) -> np.ndarray:
    """
    Pairwise (time-domain) Granger causality using simple VAR fits.
    Returns matrix F_{i<-j} normalized to 0..1.
    """
    n, T = X.shape
    if T < 8 or n < 2:
        return np.zeros((n, n))
    maxlag = int(max(1, min(maxlag, T // 4)))
    F = np.zeros((n, n))
    for i in range(n):
        yi = X[i]
        for j in range(n):
            if i == j:
                continue
            yj = X[j]
            bestF = 0.0
            for p in range(1, maxlag + 1):
                Y = yi[p:]
                if len(Y) <= (2 * p + 1):
                    break
                Phi_i = np.column_stack([yi[p - k:-k] for k in range(1, p + 1)])
                Phi_ij = np.column_stack([Phi_i] + [yj[p - k:-k] for k in range(1, p + 1)])
                try:
                    beta_i = np.linalg.lstsq(Phi_i, Y, rcond=None)[0]
                    beta_ij = np.linalg.lstsq(Phi_ij, Y, rcond=None)[0]
                except np.linalg.LinAlgError:
                    continue
                rss_i = np.sum((Y - Phi_i @ beta_i) ** 2)
                rss_ij = np.sum((Y - Phi_ij @ beta_ij) ** 2)
                k_num = p
                k_den = len(Y) - 2 * p
                if k_den <= 0 or rss_ij <= 0:
                    continue
                Fp = ((rss_i - rss_ij) / k_num) / (rss_ij / k_den)
                if np.isfinite(Fp) and Fp > bestF:
                    bestF = Fp
            F[i, j] = bestF
    maxF = np.nanmax(F)
    if maxF <= 0 or not np.isfinite(maxF):
        return np.zeros((n, n))
    return F / maxF


def _directed_flow_scores(
    X: np.ndarray,
    fs: float,
    f_lo: float,
    f_hi: float,
    *,
    maxlag: int = 6
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per-channel directed flow (in/out) using bivariate Granger within [f_lo,f_hi].
    """
    n, T = X.shape
    if n < 2 or T < 8:
        nan_vec = np.full(n, np.nan)
        return nan_vec, nan_vec, np.zeros((n, n))

    pad = max(0.2, 0.5 * (f_hi - f_lo))
    f1 = max(0.1, f_lo - pad)
    f2 = min(0.45 * fs, f_hi + pad)
    X_band = bandpass_safe(X, fs, f1, f2)
    X_band -= X_band.mean(axis=1, keepdims=True)
    std = X_band.std(axis=1, keepdims=True) + 1e-12
    X_norm = X_band / std

    F = _granger_bivariate_matrix(X_norm, maxlag=maxlag)
    denom = float(max(1, n - 1))
    flow_out = np.nansum(F, axis=1) / denom
    flow_in = np.nansum(F, axis=0) / denom
    return flow_in, flow_out, F


def _phase_gradient_directionality(X: np.ndarray, fs: float, f_lo: float, f_hi: float,
                                   t0: float, xy: Dict[str, Tuple[float,float]],
                                   ch_names: List[str]) -> Tuple[float, float]:
    Xb = bandpass_safe(X, fs, f_lo, f_hi)
    ph = np.angle(signal.hilbert(Xb, axis=-1))
    ti = int(round(t0*fs))
    phi = ph[:, ti]
    phi = np.unwrap(phi - np.mean(phi))
    coords = []
    for name in ch_names:
        if name in xy:
            coords.append([xy[name][0], xy[name][1], 1.0])
        else:
            coords.append([0.0, 0.0, 1.0])
    coords = np.asarray(coords)
    a, b, c = np.linalg.lstsq(coords, phi, rcond=None)[0]
    direction_deg = (np.degrees(np.arctan2(b, a)) + 360.0) % 360.0
    grad_mag = np.hypot(a, b) + 1e-9
    freq = 0.5*(f_lo+f_hi)
    speed = (2*np.pi*freq) / grad_mag
    return float(direction_deg), float(speed)

# ---------- Harmonics & cascade (flexible list) ----------

def _harmonic_stack_index_flexible(
    x: np.ndarray, fs: float,
    base_hz: float, base_bw_hz: float,
    harmonic_centers_hz: List[float], harmonic_bw_hz: float
) -> Tuple[float, float]:
    """
    Compute Harmonic Stack Index (HSI) and MaxH (strongest overtone frequency).

    HSI = sum(overtone powers) / fundamental_power

    Only overtones (not the fundamental) are included in the numerator to give
    a true measure of harmonic vs fundamental dominance.
    """
    pf = np.mean(bandpass_safe(x, fs, base_hz-base_bw_hz, base_hz+base_bw_hz)**2)
    powers = []
    centers = []
    for f0 in harmonic_centers_hz:
        # Skip the fundamental frequency (only include overtones in HSI)
        if abs(f0 - base_hz) < 1.0:  # If within 1 Hz of base, it's the fundamental
            continue
        if f0 + harmonic_bw_hz >= fs/2.0:
            continue
        bh = bandpass_safe(x, fs, f0 - harmonic_bw_hz, f0 + harmonic_bw_hz)
        pb = np.mean(bh**2)
        powers.append(pb)
        centers.append(f0)
    ph_sum = np.nansum(powers) if len(powers) else 0.0
    HSI = ph_sum / (pf + 1e-12)
    if len(powers):
        MaxH = centers[int(np.nanargmax(powers))]
    else:
        MaxH = np.nan
    return float(HSI), float(MaxH)

# ---------- Main ----------

def detect_ignitions_session(
    RECORDZ: pd.DataFrame,
    sr_channel: Optional[str] = "EEG.F4",
    eeg_channels: Optional[List[str]] = None,
    time_col: str = 'Timestamp',
    out_dir: str = 'exports_ignitions/S01',
    center_hz: float = 7.83,
    half_bw_hz: float = 0.5,
    smooth_sec: float = 0.25,
    z_thresh: float = 2.5,
    min_isi_sec: float = 2.0, window_sec: float = 20.0, merge_gap_sec: float = 5.0,
    R_band: Tuple[float, float] = (8,13), R_win_sec: float = 1.0, R_step_sec: float = 0.25,
    eta_pre_sec: float = 10.0, eta_post_sec: float = 10.0,
    sr_reference: str = 'auto-SSD',
    seed_method: str = 'latency',
    pel_band: Tuple[float,float] = (60, 90),
    electrode_xy: Optional[Dict[str, Tuple[float,float]]] = None,
    harmonics: Tuple[int,...] = (2,3,4,5,6,7),
    harmonics_hz: Optional[List[float]] = None,
    labels: Optional[List[str]] = None,  # Custom labels for harmonics (e.g., ['sr1', 'sr1.5', 'sr2', ...])
    harmonic_bw_hz: Optional[float] = None,
    harmonic_method: str = 'psd',  # 'psd', 'fooof_session', 'fooof_event', 'fooof_hybrid'
    fooof_freq_range: Optional[Tuple[float, float]] = None,
    fooof_freq_ranges: Optional[List[List[float]]] = None,
    fooof_max_n_peaks: int = 15,
    fooof_peak_threshold: float = 2.0,
    fooof_min_peak_height: float = 0.05,
    fooof_peak_width_limits: Tuple[float, float] = (1.0, 8.0),
    fooof_match_method: str = 'power',  # 'distance', 'power', 'average'
    nperseg_sec: float = 4.0,  # Spectral resolution control (seconds)
    additional_windows: Optional[List[Tuple[float, float]]] = None,  # Extra windows to analyze
    make_passport: bool = True,
    show: bool = True,
    verbose: bool = True,
    session_name: str = "SESSION_NAME"
) -> Tuple[Dict[str, object], List[Tuple[int,int]]]:

    if eeg_channels is None:
        eeg_channels = [c for c in RECORDZ.columns if c.startswith('EEG.')]

    _ensure_dir(out_dir)
    time_col = ensure_timestamp_column(RECORDZ, time_col=time_col, default_fs=128.0)
    fs = infer_fs(RECORDZ, time_col)

    # Handle half_bw_hz as scalar or list - extract first element for initial detection
    half_bw_arr = np.atleast_1d(half_bw_hz)
    half_bw_scalar = float(half_bw_arr[0])

    t = pd.to_numeric(RECORDZ[time_col], errors='coerce').values.astype(float)
    # Convert to relative time (start from 0)
    t = t - t[0]

    # --- 1) SR envelope z(t) & onsets (proposal via all eeg_channels) ---
    Y = np.vstack([get_series(RECORDZ, ch) for ch in eeg_channels])
    y = Y.mean(axis=0)  # or use median: np.median(Y, axis=0)

    sr_env_cache: Dict[Tuple[float, float], np.ndarray] = {}

    def _get_sr_env_z(f0: float, bw: float) -> np.ndarray:
        key = (float(np.round(f0, 5)), float(np.round(bw, 5)))
        if key not in sr_env_cache:
            sr_env_cache[key] = _sr_envelope_z_series(y, fs, f0, bw, smooth_sec)
        return sr_env_cache[key]

    z = _get_sr_env_z(center_hz, half_bw_scalar)
    mask = z >= z_thresh
    on_idx = np.where(np.diff(mask.astype(int)) == 1)[0] + 1

    onsets, last_t = [], -np.inf
    for i in on_idx:
        if t[i] - last_t >= min_isi_sec:
            onsets.append(t[i]); last_t = t[i]
    onsets = np.array(onsets, float)

    # --- 2) ignition windows (merge) ---
    ign: List[Tuple[float,float]] = []
    for s in onsets:
        a = s - window_sec/2.0
        b = s + window_sec/2.0
        if ign and a <= ign[-1][1] + merge_gap_sec:
            ign[-1] = (ign[-1][0], b)
        else:
            ign.append((a, b))
    t0s, t1s = float(t[0]), float(t[-1])
    ign = [(max(t0s,a), min(t1s,b)) for (a,b) in ign if (b-a) > 1.0]

    # Add additional windows (e.g., random windows for null control tests)
    if additional_windows is not None and len(additional_windows) > 0:
        for (a, b) in additional_windows:
            # Clip to recording bounds
            a_clip = max(t0s, a)
            b_clip = min(t1s, b)
            if b_clip - a_clip > 1.0:
                ign.append((a_clip, b_clip))

    # --- 2b) rounded windows
    rounded = []
    for a,b in ign:
        sa, sb = int(np.floor(a)), int(np.ceil(b))
        if sb > sa:
            rounded.append((sa, sb))
    ignition_windows_rounded = _merge_intervals_int(rounded)
    ign_json_path = os.path.join(out_dir, 'ignition_windows.json')
    with open(ign_json_path, 'w') as f:
        json.dump(ignition_windows_rounded, f)
    #if verbose:
        # print(f"Ignition windows (rounded, whole seconds): {ignition_windows_rounded}")
        # print(f"Saved → {ign_json_path}")

    
    # --- 3) EEG matrix & session R(t)
    X = np.vstack([get_series(RECORDZ, ch) for ch in eeg_channels])
    L = min(map(len, X))
    X, y, t = X[:, :L], y[:L], t[:L]

    t_cent, Rt = _kuramoto_R_timeseries(X, fs, R_band[0], R_band[1], R_win_sec, R_step_sec)
    zR = (Rt - np.nanmean(Rt)) / (np.nanstd(Rt) + 1e-12)

    # --- 4) per-event characterization ---
    rows = []

    # resolve harmonic centers
    if harmonics_hz and len(harmonics_hz):
        harmonic_centers = list(harmonics_hz)
    else:
        harmonic_centers = [k*center_hz for k in harmonics]

    # Save canonical harmonic centers for per-event FOOOF seed frequencies
    # (harmonic_centers may be updated by session FOOOF, but per-event should use canonical)
    canonical_harmonic_centers = list(harmonic_centers)

    # --- FOOOF session-level harmonic detection (if requested) ---
    use_event_fooof = False
    use_event_psd = True
    session_fooof_result = None

    if harmonic_method in ['fooof', 'fooof_session', 'fooof_event', 'fooof_hybrid']:
        if not FOOOF_AVAILABLE:
            if verbose:
                print(f"⚠️  harmonic_method='{harmonic_method}' requires fooof_harmonics, falling back to 'psd'")
            harmonic_method = 'psd'
        else:
            if harmonic_method in ['fooof_session', 'fooof_hybrid']:
                # Run FOOOF once on entire session
                try:
                    if verbose:
                        print(f"Running session-level FOOOF harmonic detection...")

                    # Prepare canonical frequencies for FOOOF seeding
                    # If using freq_ranges, harmonic_centers already has all harmonics
                    # Otherwise, add center_hz to harmonics
                    if fooof_freq_ranges is not None:
                        f_can = harmonic_centers
                    else:
                        f_can = [center_hz] + harmonic_centers

                    # Determine freq_range for FOOOF
                    if fooof_freq_range is not None:
                        freq_range_use = fooof_freq_range
                    else:
                        freq_range_use = (1.0, min(50.0, fs/2.0 - 1.0))

                    # Use full bandwidth array if available, otherwise use scalar
                    search_hb = half_bw_arr.tolist() if half_bw_arr.size > 1 else half_bw_scalar

                    session_harmonics, session_fooof_result = detect_harmonics_fooof(
                        records=RECORDZ,
                        channels=eeg_channels,
                        fs=fs,
                        time_col=time_col,
                        f_can=f_can,
                        freq_range=freq_range_use,
                        freq_ranges=fooof_freq_ranges,
                        combine='median',
                        search_halfband=search_hb,
                        per_harmonic_fits=(fooof_freq_ranges is not None),
                        nperseg_sec=nperseg_sec,
                        max_n_peaks=fooof_max_n_peaks,
                        peak_threshold=fooof_peak_threshold,
                        min_peak_height=fooof_min_peak_height,
                        peak_width_limits=fooof_peak_width_limits,
                        match_method=fooof_match_method
                    )

                    # Override harmonic_centers with FOOOF-detected frequencies
                    harmonic_centers = list(session_harmonics)

                    if verbose:
                        print(f"  FOOOF harmonics: {[f'{h:.2f}' for h in harmonic_centers]}")
                        # Handle both scalar and list aperiodic values
                        if isinstance(session_fooof_result.aperiodic_exponent, list):
                            beta_str = f"[{', '.join([f'{b:.3f}' for b in session_fooof_result.aperiodic_exponent])}]"
                        else:
                            beta_str = f"{session_fooof_result.aperiodic_exponent:.3f}"
                        print(f"  Aperiodic β: {beta_str}")

                        if isinstance(session_fooof_result.r_squared, list):
                            r2_str = f"[{', '.join([f'{r:.3f}' for r in session_fooof_result.r_squared])}]"
                        else:
                            r2_str = f"{session_fooof_result.r_squared:.3f}"
                        print(f"  R²: {r2_str}")

                    # For hybrid mode, also use per-event FOOOF
                    if harmonic_method == 'fooof_hybrid':
                        use_event_fooof = True
                        use_event_psd = False
                        if verbose:
                            print(f"  Using per-event FOOOF for refinement...")
                    else:
                        use_event_fooof = False
                        use_event_psd = False

                except Exception as e:
                    if verbose:
                        print(f"⚠️  Session FOOOF failed: {e}")
                        print(f"  Falling back to canonical harmonics")
                    use_event_psd = True

            elif harmonic_method in ['fooof', 'fooof_event']:
                # Use per-event FOOOF (handled in event loop)
                use_event_fooof = True
                use_event_psd = False
                if verbose:
                    print(f"Using per-event FOOOF harmonic detection...")

    if harmonic_bw_hz is None:
        harmonic_bw_list = [half_bw_scalar] * len(harmonic_centers)
    else:
        hb_arr = np.atleast_1d(harmonic_bw_hz)
        if hb_arr.size == 1:
            harmonic_bw_list = [float(hb_arr.flat[0])] * len(harmonic_centers)
        else:
            if hb_arr.size != len(harmonic_centers):
                raise ValueError("harmonic_bw_hz must match harmonic centers length")
            harmonic_bw_list = [float(x) for x in hb_arr]

    # per-session valid harmonics (below Nyquist with small margin)
    valid_pairs = [
        (f0, bw)
        for f0, bw in zip(harmonic_centers, harmonic_bw_list)
        if (f0 + bw) < (fs/2.0 - 1e-3)
    ]
    if not valid_pairs:
        valid_pairs = [(2 * center_hz, half_bw_scalar)]
    valid_harmonics = [f for f, _ in valid_pairs]
    valid_bandwidths = [bw for _, bw in valid_pairs]
    hbw = valid_bandwidths[0] if valid_bandwidths else half_bw_scalar

    # Generate or validate harmonic labels
    if labels is not None:
        # Filter labels same way harmonics are filtered (keep those below Nyquist)
        if len(labels) != len(harmonic_centers):
            raise ValueError(f"labels length ({len(labels)}) must match harmonics_hz length ({len(harmonic_centers)})")
        valid_mask = [(f0 + bw) < (fs/2.0 - 1e-3) for f0, bw in zip(harmonic_centers, harmonic_bw_list)]
        harmonic_labels = [lbl for lbl, keep in zip(labels, valid_mask) if keep]
        if not harmonic_labels:
            harmonic_labels = [labels[0]] if labels else ['sr1']
    else:
        # Default: sr1, sr2, sr3, ...
        harmonic_labels = [f'sr{i+1}' for i in range(len(valid_harmonics))]

    base_idx = None
    base_tolerance = max(half_bw_scalar, 0.8)
    for k, f0 in enumerate(valid_harmonics):
        if abs(f0 - center_hz) <= base_tolerance:
            base_idx = k
            break

    def _bandwidth_for_harmonic(idx: int) -> float:
        if idx == 0:
            if base_idx is not None and base_idx < len(valid_bandwidths):
                return valid_bandwidths[base_idx]
            return half_bw_scalar
        if not valid_bandwidths:
            return hbw
        if base_idx is None:
            target_idx = max(0, min(len(valid_bandwidths) - 1, idx - 1))
        else:
            target_idx = max(0, min(len(valid_bandwidths) - 1, base_idx + idx))
        return valid_bandwidths[target_idx]

    # --- Determine base guess from custom list (if provided) ---
    if harmonics_hz and len(harmonics_hz) and any(f < 10.0 for f in harmonics_hz):
        base_guess = float(min([f for f in harmonics_hz if f < 10.0]))
    else:
        base_guess = center_hz
    base_margin = max(hbw, 0.8)  # widen margin to catch detuned base (e.g., 7.03 Hz)

    # exclude fundamental neighborhood from harmonic set (overtones only)
    valid_harmonics_ot = [f0 for f0 in valid_harmonics if abs(f0 - base_guess) > (base_margin + 1e-6)]
    if not valid_harmonics_ot:
        # fallback: if user list only had base, synthesize multiples below Nyquist
        hbw_global = max(valid_bandwidths) if valid_bandwidths else hbw
        valid_harmonics_ot = [k*center_hz for k in (2,3,4,5,6,7) if (k*center_hz + hbw_global) < (fs/2.0 - 1e-3)]

    # clamp gamma band to Nyquist
    g_lo, g_hi = pel_band
    g_lo, g_hi = _safe_band(g_lo, g_hi, fs)
    gamma_band = (g_lo, g_hi)

    ch_short = [c.split('.',1)[-1] for c in eeg_channels]

    event_windows: List[Tuple[float, float]] = []

    for (a, b) in ign:

        i0 = max(0, int(round(a*fs)))
        i1 = min(L, int(round(b*fs)))
        if i1 - i0 < int(2*fs):
            continue
        Xw = X[:, i0:i1]

        # Per-event harmonic estimation: FOOOF or PSD method
        event_beta = np.nan
        event_r2 = np.nan

        if use_event_fooof:
            # FOOOF per-event detection
            try:
                # Convert sample indices to time bounds
                t_start = a
                t_end = b

                # Determine freq_range for FOOOF
                if fooof_freq_range is not None:
                    freq_range_use = fooof_freq_range
                else:
                    freq_range_use = (1.0, min(50.0, fs/2.0 - 1.0))

                # Use full bandwidth array if available, otherwise use scalar
                search_hb = half_bw_arr.tolist() if half_bw_arr.size > 1 else half_bw_scalar

                # Run FOOOF on this event window
                event_harmonics, event_fooof_result = detect_harmonics_fooof(
                    records=RECORDZ,
                    channels=eeg_channels,
                    fs=fs,
                    time_col=time_col,
                    window=[t_start, t_end],
                    f_can=canonical_harmonic_centers,  # Use CANONICAL as seeds (not session-detected)
                    freq_range=freq_range_use,
                    freq_ranges=fooof_freq_ranges,
                    combine='median',
                    search_halfband=search_hb,
                    per_harmonic_fits=(fooof_freq_ranges is not None) or True,
                    nperseg_sec=nperseg_sec,
                    peak_width_limits=fooof_peak_width_limits,
                    min_peak_height=fooof_min_peak_height,
                    max_n_peaks=fooof_max_n_peaks,
                    peak_threshold=fooof_peak_threshold,
                    match_method=fooof_match_method
                )

                # Store both frequencies and powers for output generation
                ignition_freqs = list(event_harmonics)
                ignition_powers = list(event_fooof_result.harmonic_powers)

                # Store FOOOF metrics
                if isinstance(event_fooof_result.aperiodic_exponent, list):
                    event_beta = np.mean(event_fooof_result.aperiodic_exponent)
                else:
                    event_beta = event_fooof_result.aperiodic_exponent

                if isinstance(event_fooof_result.r_squared, list):
                    event_r2 = np.mean(event_fooof_result.r_squared)
                else:
                    event_r2 = event_fooof_result.r_squared

            except Exception as e:
                if verbose:
                    print(f"⚠️  Event FOOOF failed at t={a:.1f}s: {e}, using canonical")
                ignition_freqs = list(harmonic_centers)
                ignition_powers = [np.nan] * len(harmonic_centers)

        elif use_event_psd or (not use_event_fooof and harmonic_method == 'psd'):
            # Two-pass Welch PSD refinement (original method)
            # Compute average PSD across channels for this event window
            psds = []
            for ch_idx in range(Xw.shape[0]):
                freqs, psd = welch(Xw[ch_idx, :], fs, nperseg=min(4096, Xw.shape[1]))
                psds.append(psd)
            avg_psd = np.mean(psds, axis=0)

            # PASS 1: Search around canonical ladder centers with full bandwidth
            first_pass_freqs = []
            for h_idx, canonical_freq in enumerate(harmonic_centers):
                # Get search bandwidth for this harmonic
                bw = half_bw_arr[h_idx] if h_idx < half_bw_arr.size else half_bw_arr[-1]

                # Search for peak around canonical frequency
                mask = (freqs >= canonical_freq - bw) & (freqs <= canonical_freq + bw)
                if np.any(mask):
                    peak_idx = np.argmax(avg_psd[mask])
                    peak_freq = freqs[mask][peak_idx]
                    first_pass_freqs.append(peak_freq)
                else:
                    # Fallback to canonical if no data in range
                    first_pass_freqs.append(canonical_freq)

            # PASS 2: Refine around first-pass estimates with half bandwidth
            ignition_freqs = []
            ignition_powers = []  # Track powers for PSD method (all valid)
            for h_idx, first_pass_freq in enumerate(first_pass_freqs):
                # Get original bandwidth and halve it for refinement
                bw = half_bw_arr[h_idx] if h_idx < half_bw_arr.size else half_bw_arr[-1]
                bw_refined = 0.1

                # Search for peak around first-pass estimate
                mask = (freqs >= first_pass_freq - bw_refined) & (freqs <= first_pass_freq + bw_refined)
                if np.any(mask):
                    peak_idx = np.argmax(avg_psd[mask])
                    peak_freq = freqs[mask][peak_idx]
                    peak_power = avg_psd[mask][peak_idx]
                    ignition_freqs.append(peak_freq)
                    ignition_powers.append(peak_power)
                else:
                    # Fallback to first-pass result if no data in refined range
                    ignition_freqs.append(first_pass_freq)
                    ignition_powers.append(np.nan)  # Mark as not found

        else:
            # fooof_session mode: use session-level harmonics directly
            ignition_freqs = list(harmonic_centers)
            ignition_powers = [1.0] * len(harmonic_centers)  # Assume all valid for session mode

        max_modes = len(ignition_freqs)

        # virtual SR (use first bandwidth from refinement list/scalar)
        bw0 = half_bw_arr[0] if half_bw_arr.size > 0 else half_bw_scalar
        v_sr, w_sr = _build_virtual_sr(Xw, fs, ignition_freqs[0], bw0, mode=sr_reference)

        # t0 from SR1 band
        f_lo, f_hi = ignition_freqs[0] - bw0, ignition_freqs[0] + bw0
        tR_ev, R_ev = _kuramoto_R_timeseries(Xw, fs, f_lo, f_hi, win_sec=2.5, step_sec=0.001)
        tR_ev = tR_ev + a
        t0_net = _detect_t0_from_R(tR_ev, R_ev, thresh=0.6)
        if not np.isfinite(t0_net):
            t0_net = 0.5*(a+b)

        # t0-centered zR maxima
        mskR_ev = (t_cent >= (t0_net - 2.5)) & (t_cent <= (t0_net + 2.5))
        zR_max_ev = float(np.nanmax(zR[mskR_ev])) if np.any(mskR_ev) else np.nan
        zR_peak_5s = zR_max_ev

        # latencies & spread
        lats, peak_z_per_ch, rise_z_per_ch = _channel_latencies(
            X,
            fs,
            f_lo,
            f_hi,
            t0_net,
            pre=2.0,
            post=1.0,
            z_th=2.0,
            min_run_sec=0.12,
            smooth_sec=0.12,
            return_details=True
        )

        flow_in, flow_out, flow_mat = _directed_flow_scores(Xw, fs, f_lo, f_hi, maxlag=6)
        flow_net = flow_out - flow_in
        flow_roles = np.array(['ambiguous'] * len(eeg_channels), dtype=object)
        if np.any(np.isfinite(flow_out)) or np.any(np.isfinite(flow_in)):
            max_out = np.nanmax(flow_out) if np.any(np.isfinite(flow_out)) else np.nan
            max_in = np.nanmax(flow_in) if np.any(np.isfinite(flow_in)) else np.nan
            if not np.isfinite(max_out) or max_out <= 0:
                out_norm = np.zeros_like(flow_out)
            else:
                out_norm = flow_out / (max_out + 1e-12)
            if not np.isfinite(max_in) or max_in <= 0:
                in_norm = np.zeros_like(flow_in)
            else:
                in_norm = flow_in / (max_in + 1e-12)
            gen_out_min = 0.55
            gen_in_max = 0.35
            hub_out_min = 0.50
            hub_in_margin = 0.05
            prop_in_min = 0.45
            prop_out_max = 0.25
            for ch_idx in range(len(flow_roles)):
                if not np.isfinite(out_norm[ch_idx]) and not np.isfinite(in_norm[ch_idx]):
                    continue
                o = out_norm[ch_idx] if np.isfinite(out_norm[ch_idx]) else 0.0
                i = in_norm[ch_idx] if np.isfinite(in_norm[ch_idx]) else 0.0
                if (o >= gen_out_min) and (i <= gen_in_max):
                    flow_roles[ch_idx] = 'generator'
                elif (o >= hub_out_min) and (i >= max(0.0, o - hub_in_margin)):
                    flow_roles[ch_idx] = 'network-hub'
                elif (i >= prop_in_min) and (o <= prop_out_max):
                    flow_roles[ch_idx] = 'propagation'
                else:
                    flow_roles[ch_idx] = 'ambiguous'
        else:
            out_norm = np.zeros_like(flow_out)
            in_norm = np.zeros_like(flow_in)

        # Composite seed scoring approach: weight multiple metrics
        def _compute_seed_score(idx: int) -> float:
            """
            Composite seed score combining temporal, causal, signal strength, and dynamics.
            Higher score = more likely to be the true generator/source channel.
            """
            score = 0.0

            # 1. Temporal component: earlier latency is better (40% weight)
            if np.isfinite(lats[idx]):
                delay = max(0.0, lats[idx] - t0_net)
                # Exponential decay: 0.5s time constant
                lat_score = np.exp(-delay / 0.5)
            else:
                lat_score = 0.0

            # 2. Causal flow component: high outflow, low inflow (35% weight)
            if np.isfinite(flow_out[idx]) and np.isfinite(flow_in[idx]):
                flow_net = flow_out[idx] - flow_in[idx]
                flow_sum = flow_out[idx] + flow_in[idx] + 1e-9
                flow_net_norm = flow_net / flow_sum  # ranges [-1, 1]
                flow_score = (1.0 + flow_net_norm) / 2.0  # map to [0, 1]
            elif np.isfinite(flow_out[idx]):
                # Only outflow known: use normalized outflow
                flow_score = min(1.0, flow_out[idx] / (np.nanmax(flow_out) + 1e-9))
            else:
                flow_score = 0.0

            # 3. Signal strength component: peak z-score (15% weight)
            if np.isfinite(peak_z_per_ch[idx]):
                # Normalize to [0,1] with saturation at z=10
                signal_score = min(1.0, peak_z_per_ch[idx] / 10.0)
            else:
                signal_score = 0.0

            # 4. Dynamics component: steep rise indicates generator (10% weight)
            if np.isfinite(rise_z_per_ch[idx]):
                # Normalize to [0,1] with saturation at rise_z=5
                rise_score = min(1.0, rise_z_per_ch[idx] / 5.0)
            else:
                rise_score = 0.0

            # Weighted combination
            composite = (0.40 * lat_score +
                        0.35 * flow_score +
                        0.15 * signal_score +
                        0.10 * rise_score)

            return composite

        # Compute scores for all channels
        seed_scores = np.array([_compute_seed_score(i) for i in range(len(eeg_channels))])

        # Select channel with highest composite score
        if np.any(np.isfinite(seed_scores)) and np.nanmax(seed_scores) > 0:
            seed_idx = int(np.nanargmax(seed_scores))
        else:
            # Fallback: use channel 0
            seed_idx = 0

        seed_flow_role = flow_roles[seed_idx]
        seed_score_value = float(seed_scores[seed_idx]) if seed_idx < len(seed_scores) else 0.0

        seed_ch = eeg_channels[seed_idx]
        seed_upper = seed_ch.upper()
        hemis = 'central'
        if any(seed_upper.startswith(prefix) for prefix in ('EEG.FP1','EEG.AF3','EEG.F3','EEG.F5','EEG.F7',
                                                             'EEG.FC5','EEG.FC3','EEG.C3','EEG.C5','EEG.T7','EEG.T9',
                                                             'EEG.CP5','EEG.P7','EEG.P9','EEG.P3','EEG.O1','EEG.FP3','EEG.PO7')):
            hemis = 'left'
        elif any(seed_upper.startswith(prefix) for prefix in ('EEG.FP2','EEG.AF4','EEG.F4','EEG.F6','EEG.F8',
                                                               'EEG.FC6','EEG.FC4','EEG.C4','EEG.C6','EEG.T8','EEG.T10',
                                                               'EEG.CP6','EEG.P8','EEG.P10','EEG.P4','EEG.O2','EEG.FP4','EEG.PO8')):
            hemis = 'right'
        seed_roi = ('occipital' if seed_ch.upper().startswith(('EEG.O','EEG.PO')) else
                    'parietal'  if seed_ch.upper().startswith(('EEG.P','EEG.CP')) else
                    'temporal'  if seed_ch.upper().startswith(('EEG.T','EEG.TP')) else
                    'frontal'   if seed_ch.upper().startswith(('EEG.F','EEG.AF','EEG.FP')) else
                    'central')
        spread = float(np.nanmedian(lats) - np.nanmin(lats)) if np.any(np.isfinite(lats)) else np.nan
        SF = float(np.mean((lats >= (t0_net-1e-6)) & (lats <= (t0_net+1.0)))) if np.any(np.isfinite(lats)) else np.nan

        # direction/speed (optional)
        if (seed_method.upper() == 'PGD') and (electrode_xy is not None):
            dir_deg, speed_cms = _phase_gradient_directionality(X, fs, f_lo, f_hi, t0_net, electrode_xy, ch_short)
        else:
            dir_deg, speed_cms = np.nan, np.nan

        # harmonics (flexible) — use overtones only (exclude base)
        HSI, MaxH = _harmonic_stack_index_flexible(
            v_sr, fs,
            base_hz=ignition_freqs[0], base_bw_hz=bw0,
            harmonic_centers_hz=ignition_freqs, harmonic_bw_hz=hbw
        )

        # HSI_canonical: using only sr1, sr3, sr5 (canonical Schumann harmonics)
        canonical_labels_hsi = ['sr1', 'sr3', 'sr5']
        canonical_freqs_hsi = []
        for clbl in canonical_labels_hsi:
            if clbl in harmonic_labels:
                idx = harmonic_labels.index(clbl)
                if idx < len(ignition_freqs):
                    canonical_freqs_hsi.append(ignition_freqs[idx])
        if len(canonical_freqs_hsi) >= 2:  # Need at least sr1 + one overtone
            HSI_canonical, _ = _harmonic_stack_index_flexible(
                v_sr, fs,
                base_hz=canonical_freqs_hsi[0], base_bw_hz=bw0,
                harmonic_centers_hz=canonical_freqs_hsi, harmonic_bw_hz=hbw
            )
        else:
            HSI_canonical = np.nan

        # Estimate per-event fundamental (base) to sanitize MaxH against local base
        try:
            fw, Pw = signal.welch(v_sr, fs=fs, nperseg=min(4096,int(2*fs)))
            # search around base_guess with expanded window
            base_win_lo = max(0.1, base_guess - max(1.2, hbw))
            base_win_hi = base_guess + max(1.2, hbw)
            base_mask = (fw >= base_win_lo) & (fw <= base_win_hi)
            if np.any(base_mask):
                base_est_hz = float(fw[base_mask][np.argmax(Pw[base_mask])])
            else:
                base_est_hz = base_guess
        except Exception:
            base_est_hz = base_guess
        # If MaxH sits within the base neighborhood for this event, drop it
        if np.isfinite(MaxH) and (abs(MaxH - base_est_hz) <= (base_margin + 1e-6)):
            MaxH_ov = np.nan
        else:
            MaxH_ov = MaxH

        # PEL (gamma→theta), using legal band
        PEL = (lambda x: (np.nan if x.size==0 else x))(np.array([0.0]))  # placeholder init
        # compute via peak-minus-peak around t0_net
        i0p = max(0, int(round((t0_net-2.0)*fs)))
        i1p = min(len(v_sr), int(round((t0_net+2.0)*fs)))
        seg = v_sr[i0p:i1p]
        # pac_mvl = np.nan
        # if seg.size > 10:
        #     th = bandpass_safe(seg, fs, ignition_freqs[0]-0.3, ignition_freqs[0]+0.3)
        #     ga = bandpass_safe(seg, fs, gamma_band[0], gamma_band[1])
        #     env_th = np.abs(signal.hilbert(th))
        #     env_ga = np.abs(signal.hilbert(ga))
        #     tt = np.arange(seg.size)/fs + (t0_net-2.0)
        #     k0 = np.argmin(np.abs(tt - t0_net))
        #     p_th = np.argmax(env_th[:k0]) if k0>0 else 0
        #     p_ga = np.argmax(env_ga[:k0]) if k0>0 else 0
        #     PEL = float(tt[p_th] - tt[p_ga])
        #     try:
        #         phase_th = np.angle(signal.hilbert(th))
        #         amp_ga = np.abs(signal.hilbert(ga))
        #         pac_mvl = float(np.abs(np.mean(amp_ga * np.exp(1j * phase_th))) /
        #                          (np.mean(amp_ga) + 1e-12))
        #     except Exception:
        #         pac_mvl = np.nan
        # else:
        #     PEL = np.nan

        # FS metrics from v_sr around t0_net
        v_f = bandpass_safe(v_sr, fs, f_lo, f_hi)
        env_v = np.abs(signal.hilbert(v_f))
        b0 = max(0, int(round((t0_net - a - 2.0)*fs)))
        b1 = max(1, int(round((t0_net - a)*fs)))
        mu = np.mean(env_v[b0:b1]) if b1>b0 else np.mean(env_v)
        sd = np.std(env_v[b0:b1]) + 1e-12 if b1>b0 else (np.std(env_v)+1e-12)
        z_env = (env_v - mu)/sd
        fs_z = float(np.nanmax(z_env))
        k0 = int(round((t0_net - a)*fs))
        kL = max(0, k0 - int(1.0*fs)); kR = min(len(z_env), k0 + int(1.0*fs))
        fs_auc = float(np.trapz(z_env[kL:kR], dx=1/fs)) if kR>kL else np.nan


        # Chart-style MSC: per-channel vs median reference, averaged across channels
        # This matches the visualization in six_panel_2() coherence chart
        msc_modes = _msc_per_channel_vs_median(Xw, fs, ignition_freqs[:max_modes], bw0)
        # Extract for backward compatibility
        msc_v = msc_modes[0] if msc_modes else np.nan
        msc_sr2_v = msc_modes[1] if len(msc_modes) > 1 else np.nan
        msc_sr3_v = msc_modes[2] if len(msc_modes) > 2 else np.nan

        # --- Artifact detection metrics ---
        # Spectral slope (1/f exponent) during ignition
        spectral_slope = _spectral_slope_during_event(
            Xw, fs, band=(3, 45),
            exclude_centers=ignition_freqs[:min(3, len(ignition_freqs))],
            exclude_bw=bw0
        )

        # Frequency Specificity Index (power concentration at SR freqs)
        freq_specificity = _frequency_specificity_index(
            Xw, fs,
            sr_freqs=ignition_freqs[:min(3, len(ignition_freqs))],
            bw=bw0
        )

        # FSI_canonical: using only sr1, sr3, sr5 (canonical Schumann harmonics)
        canonical_labels_fsi = ['sr1', 'sr3', 'sr5']
        canonical_freqs_fsi = []
        for clbl in canonical_labels_fsi:
            if clbl in harmonic_labels:
                idx = harmonic_labels.index(clbl)
                if idx < len(ignition_freqs):
                    canonical_freqs_fsi.append(ignition_freqs[idx])
        if canonical_freqs_fsi:
            freq_specificity_canonical = _frequency_specificity_index(
                Xw, fs,
                sr_freqs=canonical_freqs_fsi,
                bw=bw0
            )
        else:
            freq_specificity_canonical = np.nan

        # MSC Bandwidth Specificity Ratio (MSC sharpness at SR freqs)
        msc_bandwidth_ratio = _msc_bandwidth_specificity(
            Xw, fs,
            sr_freqs=ignition_freqs[:min(3, len(ignition_freqs))],
            bw=bw0
        )

        # --- MSC peak vs average: compute around t0_net (±2.5 s) ---
        x_mean = Xw.mean(axis=0)
        y_ref  = v_sr
        def _slice_idx(t0, left, right):
            i0s = max(0, int(round((t0 + left - a)*fs)))
            i1s = min(len(x_mean), int(round((t0 + right - a)*fs)))
            return i0s, i1s
        i0_loc, i1_loc   = _slice_idx(t0_net, -2.5, +2.5)
        i0_base, i1_base = _slice_idx(t0_net, -5.0, -2.0)
        msc_loc_vals  = _msc_f0_series(x_mean[i0_loc:i1_loc],  y_ref[i0_loc:i1_loc],  fs, ignition_freqs[0], win=1.0, step=0.1)
        msc_base_vals = _msc_f0_series(x_mean[i0_base:i1_base], y_ref[i0_base:i1_base], fs, ignition_freqs[0], win=1.0, step=0.1)
        msc_peak      = float(np.nanmax(msc_loc_vals))   if msc_loc_vals.size  else np.nan
        msc_mean_loc  = float(np.nanmean(msc_loc_vals))  if msc_loc_vals.size  else np.nan
        msc_base      = float(np.nanmean(msc_base_vals)) if msc_base_vals.size else np.nan
        msc_auc_loc   = float(msc_mean_loc * max(0.0, (i1_loc - i0_loc)/fs)) if np.isfinite(msc_mean_loc) else np.nan

        # SR envelope summaries from reference channel z(t)
        i0w = max(0, int(np.floor(a*fs)))
        i1w = min(len(z), int(np.ceil(b*fs)))
        sr_z_max = sr_z_peak_t = sr_z_mean_pm5 = sr_z_mean_post5 = np.nan
        # Initialize z_max for all harmonics (will populate sr2_z_max, sr3_z_max, sr4_z_max, etc.)
        harmonic_z_max = []
        if i1w - i0w > 0:
            seg_z = z[i0w:i1w]
            if np.all(np.isnan(seg_z)):
                k_rel = 0
                k_peak = i0w
            else:
                k_rel = int(np.nanargmax(seg_z))
                k_peak = i0w + k_rel
            sr_z_max = float(seg_z[k_rel]) if seg_z.size else np.nan
            sr_z_peak_t = float(t[k_peak]) if k_peak < len(t) else np.nan
            t_on = a + window_sec/2.0
            k_on = int(np.argmin(np.abs(t - t_on)))
            kL2 = max(0, k_on - int(5*fs))
            kR2 = min(len(z), k_on + int(5*fs))
            sr_z_mean_pm5 = float(np.nanmean(z[kL2:kR2])) if kR2>kL2 else np.nan
            k_postR = min(len(z), k_peak + int(5*fs))
            sr_z_mean_post5 = float(np.nanmean(z[k_peak:k_postR])) if k_postR>k_peak else np.nan

            # Compute envelope z_max for all harmonics (SR2, SR3, SR4, ...)
            for mode_idx in range(1, max_modes):
                bw_mode = _bandwidth_for_harmonic(mode_idx)
                z_mode = _get_sr_env_z(ignition_freqs[mode_idx], bw_mode)
                seg_mode = z_mode[i0w:i1w]
                if seg_mode.size and np.any(np.isfinite(seg_mode)):
                    peak_val = float(np.nanmax(seg_mode))
                else:
                    peak_val = np.nan
                harmonic_z_max.append(peak_val)

        # Extract for backward compatibility
        sr2_z_max = harmonic_z_max[0] if len(harmonic_z_max) > 0 else np.nan
        sr3_z_max = harmonic_z_max[1] if len(harmonic_z_max) > 1 else np.nan

        # PLV around ignition (±5 s) for all SR bands
        plv_modes: List[float] = []
        try:
            t_event = t[i0:i1]
            if t_event.size:
                plv_mask = (t_event >= (t0_net - 5.0)) & (t_event <= (t0_net + 5.0))
            else:
                plv_mask = np.array([], dtype=bool)
            for mode_idx in range(max_modes):
                bw_mode = _bandwidth_for_harmonic(mode_idx)
                f_center = ignition_freqs[mode_idx]
                Xw_band = bandpass_safe(Xw, fs, f_center - bw_mode, f_center + bw_mode)
                phases = np.angle(signal.hilbert(Xw_band, axis=-1))
                plv_inst = np.abs(np.nanmean(np.exp(1j * phases), axis=0))
                if plv_inst.size:
                    if plv_mask.size and np.any(plv_mask):
                        val = float(np.nanmean(plv_inst[plv_mask]))
                    else:
                        val = float(np.nanmean(plv_inst))
                else:
                    val = np.nan
                plv_modes.append(val)
        except Exception:
            pass

        # Extract for backward compatibility
        plv_mean_pm5 = plv_modes[0] if len(plv_modes) > 0 else np.nan
        plv_sr2_pm5 = plv_modes[1] if len(plv_modes) > 1 else np.nan
        plv_sr3_pm5 = plv_modes[2] if len(plv_modes) > 2 else np.nan

        # label: classify based on which harmonic has the strongest envelope
        if np.isfinite(sr_z_max) and np.isfinite(sr2_z_max) and np.isfinite(sr3_z_max):
            # Direct comparison of harmonic z-scores
            z_scores = [sr_z_max, sr2_z_max, sr3_z_max]
            max_idx = int(np.nanargmax(z_scores))

            if max_idx == 0:  # SR1 (fundamental) is strongest
                type_label = 'fundamental-led'
            elif max_idx in (1, 2):  # SR2 or SR3 (overtones) are strongest
                type_label = 'overtone-led'
            else:
                type_label = 'fundamental-led'  # fallback
        else:
            # Fallback to old logic if harmonic z-scores not available
            if (fs_z >= 3.0) and (HSI >= 0.2):
                type_label = 'fundamental-led'
            elif (fs_z < 2.0) and (HSI >= 0.5 or (np.isfinite(MaxH) and MaxH >= 6*center_hz-1.0)):
                type_label = 'overtone-led'
            else:
                pks, _ = signal.find_peaks(z_env, distance=int(1.0*fs), height=0.6*np.nanmax(z_env))
                type_label = 'two-phase' if len(pks) >= 2 else 'fundamental-led'

        


        rows.append({
            'session_name': session_name, 
            't_start': a, 
            'sr_z_peak_t': sr_z_peak_t,
            't_end': b,
            'duration_s': float(b-a),
            'seed_ch': seed_ch, 
            'seed_roi': seed_roi, 
            'seed_hemisphere': hemis,
            'seed_latency_s': float(lats[seed_idx]) if np.isfinite(lats[seed_idx]) else np.nan,
            'seed_latency_offset_s': float(lats[seed_idx] - t0_net) if np.isfinite(lats[seed_idx]) and np.isfinite(t0_net) else np.nan,
            'seed_peak_z': float(peak_z_per_ch[seed_idx]) if np.isfinite(peak_z_per_ch[seed_idx]) else np.nan,
            'seed_rise_z': float(rise_z_per_ch[seed_idx]) if np.isfinite(rise_z_per_ch[seed_idx]) else np.nan,
            'seed_flow_role': seed_flow_role,
            'seed_flow_out': float(flow_out[seed_idx]) if np.isfinite(flow_out[seed_idx]) else np.nan,
            'seed_flow_in': float(flow_in[seed_idx]) if np.isfinite(flow_in[seed_idx]) else np.nan,
            'seed_flow_net': float(flow_net[seed_idx]) if np.isfinite(flow_net[seed_idx]) else np.nan,
            'seed_flow_validation': bool(seed_flow_role == 'generator'),
            'seed_score': seed_score_value,
            'type_label': type_label,
            # Add all harmonic frequencies dynamically using labels
            # Show "NaN" if power is NaN (no peak detected in CANON window)
            **{harmonic_labels[i]: ("NaN" if (i < len(ignition_powers) and np.isnan(ignition_powers[i]))
                            else f"{ignition_freqs[i]:.2f}" if i < len(ignition_freqs) else "nan")
               for i in range(min(max_modes, len(harmonic_labels)))},
            # Add harmonic z_max values dynamically using labels
            **{f'{harmonic_labels[0]}_z_max': sr_z_max},
            **{f'{harmonic_labels[i]}_z_max': harmonic_z_max[i-1] if (i-1) < len(harmonic_z_max) else np.nan
               for i in range(1, len(harmonic_labels))},
            # Add harmonic MSC values dynamically using labels
            **{f'msc_{harmonic_labels[i]}_v': msc_modes[i] if i < len(msc_modes) else np.nan
               for i in range(len(harmonic_labels))},
            # Add harmonic PLV values dynamically using labels
            **{f'plv_{harmonic_labels[i]}_pm5': plv_modes[i] if i < len(plv_modes) else np.nan
               for i in range(len(harmonic_labels))},
            'HSI': HSI,
            'HSI_canonical': HSI_canonical,
            'spectral_slope': spectral_slope,
            'freq_specificity': freq_specificity,
            'FSI_canonical': freq_specificity_canonical,
            'msc_bandwidth_ratio': msc_bandwidth_ratio,

            't0_net': t0_net, 'zR_max': zR_max_ev, 'zR_peak_±5s': zR_peak_5s,
            'fs_z': fs_z, 'fs_auc': fs_auc, 'MaxH': MaxH, 'MaxH_overtone': MaxH_ov, 'PEL_sec': PEL,
            'spread_time_sec': spread, 'SF': SF,
            # 'msc_7p83_v_peak': msc_peak, 'msc_7p83_v_mean_local': msc_mean_loc,
            # 'msc_7p83_v_base': msc_base, 'msc_7p83_v_auc_loc': msc_auc_loc,
            'sr_z_mean_pm5': sr_z_mean_pm5, 'sr_z_mean_post5': sr_z_mean_post5,
            # 'pac_mvl': pac_mvl,
             'base_est_hz': base_est_hz,

            'ignition_freqs': ignition_freqs,
            'harmonic_method': harmonic_method,
            'fooof_beta': event_beta,
            'fooof_r2': event_r2,
        })
        event_windows.append((a, b))

    events = pd.DataFrame(rows)

    raw_event_count = int(len(events)) if not events.empty else 0
    filtered_out_count = 0
    if not events.empty:
        mask = pd.Series(True, index=events.index)
        # if 'msc_7p83_v' in events.columns:
        #     msc_vals = pd.to_numeric(events['msc_7p83_v'], errors='coerce')
        #     mask &= msc_vals.isna() | (msc_vals >= 0.2)
        # if 'HSI' in events.columns:
        #     hsi_vals = pd.to_numeric(events['HSI'], errors='coerce')
        #     mask &= hsi_vals.isna() | (hsi_vals <= 2.0)
        kept = int(mask.sum())
        filtered_out_count = raw_event_count - kept
        events = events.loc[mask].copy()
        event_windows = [event_windows[i] for i, keep in enumerate(mask.to_numpy()) if keep]
        events.reset_index(drop=True, inplace=True)
    else:
        event_windows = []
    # event_windows = []

    ignition_windows_final = event_windows
    ignition_windows_rounded_final = []
    for (a, b) in ignition_windows_final:
        sa = int(np.floor(a))
        sb = int(np.ceil(b))
        if sb > sa:
            ignition_windows_rounded_final.append((sa, sb))
    with open(ign_json_path, 'w') as f:
        json.dump(ignition_windows_rounded_final, f)

    # --- 5) ETA of zR(t) aligned to t0_net ---
    if onsets.size and not events.empty and t_cent.size and zR.size:
        dt_R = np.median(np.diff(t_cent)) if t_cent.size > 1 else R_step_sec
        tau = np.arange(-eta_pre_sec, eta_post_sec + dt_R/2, dt_R)
        ETA = []
        for t0 in events['t0_net'].dropna().to_numpy():
            ETA.append(np.interp(t0 + tau, t_cent, zR, left=np.nan, right=np.nan))
        ETA = np.vstack(ETA) if len(ETA) else np.empty((0, len(tau)))
        if ETA.size:
            eta_mean = np.nanmean(ETA, axis=0)
            counts = np.sum(np.isfinite(ETA), axis=0)
            den = np.sqrt(np.maximum(1, counts))
            eta_sem  = np.nanstd(ETA, axis=0) / den
        else:
            eta_mean = np.full_like(tau, np.nan)
            eta_sem  = np.full_like(tau, np.nan)
    else:
        tau = np.array([]); eta_mean = np.array([]); eta_sem = np.array([])

    # --- 6) Plots ---
    fig = plt.figure(figsize=(18,4))
    plt.plot(t[:len(z)], z, lw=1.0, label='SR env z (ref)')
    plt.axhline(z_thresh, color='k', ls='--', lw=1, label='z-thresh')
    for (aa,bb) in ignition_windows_final:
        plt.axvspan(aa,bb, color='tab:orange', alpha=0.15)
    plt.xlabel('Time (s)'); plt.ylabel('SR z')
    fig.suptitle('SR envelope z(t) with detected ignitions', fontsize=14, y=0.98)
    plt.title(session_name, fontsize=10, pad=10)
    plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(out_dir,'sr_env_z.png'), dpi=140)
    if show: plt.show();
    plt.close()

    # --- 7) summaries & files ---
    if events.empty:
        summary = {'n_events': 0}
    else:
        summary = {
            'n_events': int(len(events)),
            'median_duration_s': float(events['duration_s'].mean()),
            'median_fs_z': float(events['fs_z'].mean()),
            'median_HSI': float(events['HSI'].mean()),
            'median_PEL_sec': float(events['PEL_sec'].mean()),
            'coverage_pct': float(100.0*np.sum(events['duration_s'])/max(1e-9, t[-1]-t[0]))
        }

    # Compute sr_score before export
    if not events.empty:
        def _col_vals(name: str, fill: float = np.nan) -> np.ndarray:
            if name in events.columns:
                return pd.to_numeric(events[name], errors='coerce').to_numpy()
            return np.full(len(events), fill, dtype=float)

        # Phi-weighting function: parse exponent from label, return weight = φ^(-exponent)
        def _phi_weight_from_label(label: str) -> float:
            """
            Parse phi exponent from harmonic label and return weight.
            Labels: sr1, sr1.5, sr2, sr2o, sr2.5, sr3, sr4, sr5, sr6
            Weight = φ^(-exponent)
            """
            PHI = 1.618033988749895
            # Special case: sr2o (observed 2nd harmonic) → exponent 1.25
            if label == 'sr2o':
                return PHI ** (-1.25)
            # Parse numeric part from label (e.g., "sr1.5" → 1.5, "sr3" → 3)
            match = re.search(r'sr(\d+\.?\d*)', label)
            if match:
                num = float(match.group(1))
                # Exponent mapping: sr1→0, sr1.5→0.5, sr2→1, sr2.5→1.5, sr3→2, etc.
                exponent = num - 1
                return PHI ** (-exponent)
            return 1.0  # Default weight if label doesn't match

        n_harmonics = len(harmonic_labels)
        weights = np.array([_phi_weight_from_label(lbl) for lbl in harmonic_labels])

        # Weighted scores across all available harmonics
        _z_vals = np.column_stack([_col_vals(f'{lbl}_z_max', fill=0.0) for lbl in harmonic_labels]) if n_harmonics > 0 else np.zeros((len(events), 1))
        _msc_vals = np.column_stack([_col_vals(f'msc_{lbl}_v', fill=0.0) for lbl in harmonic_labels]) if n_harmonics > 0 else np.zeros((len(events), 1))
        _plv_vals = np.column_stack([_col_vals(f'plv_{lbl}_pm5', fill=0.0) for lbl in harmonic_labels]) if n_harmonics > 0 else np.zeros((len(events), 1))

        _z_score = np.sum(_z_vals * weights, axis=1)
        _msc_score = np.sum(_msc_vals * weights, axis=1)
        _plv_score = np.sum(_plv_vals * weights, axis=1)

        _HSIv = _col_vals('HSI', fill=0.0)
        events['sr_score'] = _z_score**0.7 * _msc_score**1.2 * _plv_score / (1 + _HSIv)

        # Canonical score: only sr1, sr3, sr5 (observed Schumann harmonics)
        # Uses FSI_canonical and HSI_canonical for full consistency
        canonical_labels = ['sr1', 'sr3', 'sr5']
        canonical_present = [lbl for lbl in canonical_labels if lbl in harmonic_labels]
        if canonical_present:
            # Fixed canonical weights for sr1, sr3, sr5
            _canonical_weight_map = {'sr1': 0.618, 'sr3': 0.326, 'sr5': 0.146}
            canonical_weights = np.array([_canonical_weight_map[lbl] for lbl in canonical_present])
            _z_canon = np.column_stack([_col_vals(f'{lbl}_z_max', fill=0.0) for lbl in canonical_present])
            _msc_canon = np.column_stack([_col_vals(f'msc_{lbl}_v', fill=0.0) for lbl in canonical_present])
            _plv_canon = np.column_stack([_col_vals(f'plv_{lbl}_pm5', fill=0.0) for lbl in canonical_present])
            _z_score_canon = np.sum(_z_canon * canonical_weights, axis=1)
            _msc_score_canon = np.sum(_msc_canon * canonical_weights, axis=1)
            _plv_score_canon = np.sum(_plv_canon * canonical_weights, axis=1)
            # Use HSI_canonical for canonical sr_score
            _HSIv_canon = _col_vals('HSI_canonical', fill=0.0)
            events['sr_score_canonical'] = _z_score_canon**0.7 * _msc_score_canon**1.2 * _plv_score_canon / (1 + _HSIv_canon)
        else:
            events['sr_score_canonical'] = np.nan

    # Build export columns (same as display)
    export_base_cols = ['session_name','t_start','t0_net','sr_z_peak_t','t_end','duration_s']
    export_freq_cols = harmonic_labels
    export_zmax_cols = [f'{lbl}_z_max' for lbl in harmonic_labels]
    export_msc_cols = [f'msc_{lbl}_v' for lbl in harmonic_labels]
    export_plv_cols = [f'plv_{lbl}_pm5' for lbl in harmonic_labels]
    export_other_cols = ['HSI', 'HSI_canonical', 'freq_specificity', 'FSI_canonical', 'sr_score', 'sr_score_canonical', 'ignition_freqs']
    export_cols = [c for c in export_base_cols + export_freq_cols + export_zmax_cols + export_msc_cols + export_plv_cols + export_other_cols
                   if c in events.columns]
    events_export = events[export_cols].copy()
    events_export.rename(columns={'freq_specificity': 'FSI'}, inplace=True)

    events_export.to_csv(os.path.join(out_dir,'events.csv'), index=False)
    pd.DataFrame([summary]).to_csv(os.path.join(out_dir,'summary.csv'), index=False)
    if make_passport:
        events_export.to_csv(os.path.join(out_dir,'event_passport.csv'), index=False)

    if verbose:
        print("\n=== Ignition Detection — Session Summary ===\n")
        print(f"Session: {session_name}")

        # For FOOOF methods, show actual detected harmonics (not canonical)
        if harmonic_method in ['fooof', 'fooof_event', 'fooof_session', 'fooof_hybrid']:
            if not events.empty and 'ignition_freqs' in events.columns:
                # Extract actual FOOOF-detected harmonics from events
                all_freqs = []
                for freqs in events['ignition_freqs']:
                    if isinstance(freqs, str):
                        freqs = eval(freqs)
                    if freqs is not None and len(freqs) > 0:
                        all_freqs.append(freqs)

                if all_freqs:
                    # Compute median across events for each harmonic
                    n_harmonics = len(all_freqs[0])
                    median_harmonics = []
                    for i in range(n_harmonics):
                        h_values = [f[i] for f in all_freqs if len(f) > i]
                        if h_values:
                            median_harmonics.append(np.median(h_values))

                    # For hybrid mode, show both session-level and per-event harmonics
                    if harmonic_method == 'fooof_hybrid':
                        print("Session SR (FOOOF session):    ", np.round(harmonic_centers, 2))
                        print("Per-Event SR (FOOOF median):   ", np.round(median_harmonics, 2))
                        if harmonics_hz is not None:
                            print("Canonical SR (input):          ", np.round(harmonics_hz, 2))
                    else:
                        # For session or event-only methods
                        if harmonic_method == 'fooof_session':
                            print("Estimated SR (FOOOF session):  ", np.round(median_harmonics, 2))
                        else:
                            print("Estimated SR (FOOOF per-event):", np.round(median_harmonics, 2))
                        if harmonics_hz is not None:
                            print("Canonical SR (input):          ", np.round(harmonics_hz, 2))
                else:
                    print("Canonical SR (no events detected): ", np.round(harmonics_hz, 2) if harmonics_hz is not None else np.round(harmonic_centers, 2))
            else:
                print("Canonical SR (no events detected): ", np.round(harmonics_hz, 2) if harmonics_hz is not None else np.round(harmonic_centers, 2))
        else:
            # PSD method: use canonical harmonics
            if harmonics_hz is not None:
                print("Estimated SR: ", np.round(harmonics_hz, 2))
            else:
                print("Harmonic Centers: ", np.round(harmonic_centers, 2))
        print(f"\nIgnition windows: {ignition_windows_rounded_final}")
        print(f"EEG channels (n={len(eeg_channels)}): {', '.join([c.split('.',1)[-1] for c in eeg_channels])}")
        # Format bandwidth display (handle both scalar and list)
        if half_bw_arr.size == 1:
            bw_str = f"{half_bw_scalar:.2f}"
        else:
            bw_str = f"[{', '.join([f'{x:.2f}' for x in half_bw_arr])}]"
        print(f"Detection band: {center_hz:.2f}±{bw_str} Hz; z-thresh={z_thresh:.2f}; window={window_sec:.1f}s; min_ISI={min_isi_sec:.1f}s")
        # print(f"R(t) band: {R_band[0]:.1f}–{R_band[1]:.1f} Hz, win={R_win_sec:.2f}s, step={R_step_sec:.2f}s")
        print(f"Event SR mode: {sr_reference}")
        harm_src = 'custom' if (harmonics_hz and len(harmonics_hz)) else 'multiples'
        # print(f"PEL gamma band: {gamma_band[0]:.1f}–{gamma_band[1]:.1f} Hz; Harmonics (valid, {harm_src}): {np.round(valid_harmonics,3)}")

        def fmt_iqr(x: np.ndarray) -> str:
            x = np.asarray(x, float)
            x = x[np.isfinite(x)]
            if x.size == 0: return "n/a"
            q1, med, q3 = np.nanpercentile(x, [25, 50, 75])
            return f"{med:.2f} [{q1:.2f}, {q3:.2f}]"

        

        total_events = int(len(events)) if not events.empty else 0
        print(f"\nEvents detected: {total_events}")
        if filtered_out_count > 0:
            print(f"Filtered out {filtered_out_count} candidate ignition(s) with msc_7p83_v < 0.2 or HSI > 2.0.")

        if total_events > 0:
            def col_vals(name: str, fill: float = np.nan) -> np.ndarray:
                if name in events.columns:
                    return pd.to_numeric(events[name], errors='coerce').to_numpy()
                return np.full(total_events, fill, dtype=float)

            dur   = col_vals('duration_s')
            # Use harmonic_labels for column lookups
            srmax = col_vals(f'{harmonic_labels[0]}_z_max', fill=0.0) if len(harmonic_labels) > 0 else np.zeros(total_events)
            sr2max = col_vals(f'{harmonic_labels[1]}_z_max', fill=0.0) if len(harmonic_labels) > 1 else np.zeros(total_events)
            s3rmax = col_vals(f'{harmonic_labels[2]}_z_max', fill=0.0) if len(harmonic_labels) > 2 else np.zeros(total_events)

            srpm5 = col_vals('sr_z_mean_pm5')

            plvpm5 = col_vals(f'plv_{harmonic_labels[0]}_pm5', fill=0.0) if len(harmonic_labels) > 0 else np.zeros(total_events)
            plv_sr2_pm5 = col_vals(f'plv_{harmonic_labels[1]}_pm5', fill=0.0) if len(harmonic_labels) > 1 else np.zeros(total_events)
            plv_sr3_pm5 = col_vals(f'plv_{harmonic_labels[2]}_pm5', fill=0.0) if len(harmonic_labels) > 2 else np.zeros(total_events)

            msc_v  = col_vals(f'msc_{harmonic_labels[0]}_v', fill=0.0) if len(harmonic_labels) > 0 else np.zeros(total_events)
            msc_sr2_v  = col_vals(f'msc_{harmonic_labels[1]}_v', fill=0.0) if len(harmonic_labels) > 1 else np.zeros(total_events)
            msc_sr3_v  = col_vals(f'msc_{harmonic_labels[2]}_v', fill=0.0) if len(harmonic_labels) > 2 else np.zeros(total_events)

            rec_cov = (100.0*np.nansum(dur)/max(1e-9, t[-1]-t[0])) if dur.size else np.nan
            
            fsz  = col_vals('fs_z')
            HSIv = col_vals('HSI', fill=0.0)
            HSIv_canon = col_vals('HSI_canonical', fill=0.0)
            FSIv_canon = col_vals('FSI_canonical', fill=0.0)
            PELv = col_vals('PEL_sec')
            spread= col_vals('spread_time_sec')
            SFv   = col_vals('SF')
            seed_counts = events['seed_roi'].value_counts(dropna=True)
            type_counts = events['type_label'].value_counts(dropna=True)
            flow_role_counts = events['seed_flow_role'].value_counts(dropna=True)

            z_score = (0.618*srmax + 0.236*sr2max + 0.146*s3rmax)
            msc_score = (0.618*msc_v + 0.236*msc_sr2_v + 0.146*msc_sr3_v)
            plv_score = (0.618*plvpm5 + 0.236*plv_sr2_pm5 + 0.146*plv_sr3_pm5)

            sr_score = z_score**0.7 * msc_score**1.2 * plv_score / (1 + HSIv)

            # Canonical score: only sr1, sr3, sr5 (observed Schumann harmonics)
            # Uses FSI_canonical and HSI_canonical for full consistency
            canonical_labels = ['sr1', 'sr3', 'sr5']
            canonical_present = [lbl for lbl in canonical_labels if lbl in harmonic_labels]
            if canonical_present:
                # Fixed canonical weights for sr1, sr3, sr5
                _canonical_weight_map = {'sr1': 0.618, 'sr3': 0.326, 'sr5': 0.146}
                canon_weights = np.array([_canonical_weight_map[lbl] for lbl in canonical_present])
                z_canon = np.column_stack([col_vals(f'{lbl}_z_max', fill=0.0) for lbl in canonical_present])
                msc_canon = np.column_stack([col_vals(f'msc_{lbl}_v', fill=0.0) for lbl in canonical_present])
                plv_canon = np.column_stack([col_vals(f'plv_{lbl}_pm5', fill=0.0) for lbl in canonical_present])
                z_score_canon = np.sum(z_canon * canon_weights, axis=1)
                msc_score_canon = np.sum(msc_canon * canon_weights, axis=1)
                plv_score_canon = np.sum(plv_canon * canon_weights, axis=1)
                # Use HSI_canonical for canonical sr_score
                HSIv_canon = col_vals('HSI_canonical', fill=0.0)
                sr_score_canonical = z_score_canon**0.7 * msc_score_canon**1.2 * plv_score_canon / (1 + HSIv_canon)
            else:
                sr_score_canonical = np.full(total_events, np.nan)

            events['sr_score'] = sr_score
            events['sr_score_canonical'] = sr_score_canonical
            # events['t_score'] = t_score
            # events['score'] = score

            # print(f"  Duration (s)           — median [IQR]: {fmt_iqr(dur)}")
            lbl0 = harmonic_labels[0] if len(harmonic_labels) > 0 else 'SR1'
            lbl1 = harmonic_labels[1] if len(harmonic_labels) > 1 else 'SR2'
            lbl2 = harmonic_labels[2] if len(harmonic_labels) > 2 else 'SR3'
            print(f"  {lbl0} z max (ref)      — median [IQR]: {fmt_iqr(srmax)}")
            print(f"  {lbl1} z max (ref)      — median [IQR]: {fmt_iqr(sr2max)}")
            print(f"  {lbl2} z max (ref)      — median [IQR]: {fmt_iqr(s3rmax)}")
            print(f"  MSC@{lbl0} (virtual)    — median [IQR]: {fmt_iqr(msc_v)}")
            print(f"  PLV@{lbl0} (±5 s)       — median [IQR]: {fmt_iqr(plvpm5)}")
            # print(f"  Seed composite score   — median [IQR]: {fmt_iqr(events['seed_score'].to_numpy(dtype=float))}")
            # print(f"  Seed outflow score     — median [IQR]: {fmt_iqr(events['seed_flow_out'].to_numpy(dtype=float))}")
            # if not flow_role_counts.empty:
            #     flow_summary = ", ".join([f"{role}: {cnt} ({100.0*cnt/total_events:.0f}%)" for role, cnt in flow_role_counts.items()])
            #     print(f"  Seed flow roles        — {flow_summary}")
            # if 'seed_flow_validation' in events.columns:
            #     val_mean = float(np.nanmean(events['seed_flow_validation'].astype(float)))
            #     if np.isfinite(val_mean):
            #         print(f"  Flow-validated seeds    — {100.0*val_mean:.0f}% generator-classified")
            
            # print(f"  PAC MVL (θ–γ)          — median [IQR]: {fmt_iqr(pacv)}")
            print(f"  HSI (harmonic stack)   — median [IQR]: {fmt_iqr(HSIv)}")
            print(f"  HSI (canonical)        — median [IQR]: {fmt_iqr(HSIv_canon)}")
            print(f"  FSI (canonical)        — median [IQR]: {fmt_iqr(FSIv_canon)}")
            print(f"  SR-Score               — median [IQR]: {fmt_iqr(sr_score)}")
            print(f"  SR-Score (canonical)   — median [IQR]: {fmt_iqr(sr_score_canonical)}")
            # print(f"  T-Score                — median [IQR]: {fmt_iqr(t_score)}")
            # print(f"  Score                  — median [IQR]: {fmt_iqr(score)}")
            print(f"  Coverage of recording  — {rec_cov:.2f}%")
            
            # print("\n— Event-centric metrics —")
            # print(f"  FS z (SR1)             — median [IQR]: {fmt_iqr(fsz)}")
            # # print(f"  PEL Γ→θ lag (s)        — median [IQR]: {fmt_iqr(PELv)}")
            # print(f"  Seed ROI distribution  — ", ",".join([f"{k}: {int(v)} ({100.0*v/total_events:.0f}%)" for k,v in seed_counts.items()]))
            # print(f"  Spread time (s)        — median [IQR]: {fmt_iqr(spread)}")
            # print(f"  Synchronized fraction  — median [IQR]: {fmt_iqr(SFv)}")
            
            try:
                top_by_srz = events.sort_values('t_start', ascending=True)
                # Build column list dynamically using harmonic_labels
                base_cols = ['t_start','t0_net','sr_z_peak_t','t_end','duration_s']

                # Use harmonic_labels for column names
                sr_freq_cols = harmonic_labels
                sr_zmax_cols = [f'{lbl}_z_max' for lbl in harmonic_labels]
                msc_cols = [f'msc_{lbl}_v' for lbl in harmonic_labels]
                plv_cols = [f'plv_{lbl}_pm5' for lbl in harmonic_labels]

                other_cols = ['HSI', 'freq_specificity', 'sr_score', 'sr_score_canonical']

                # Combine all columns and filter by what exists in the dataframe
                cols2 = [c for c in base_cols + sr_freq_cols + sr_zmax_cols + msc_cols + plv_cols + other_cols
                        if c in events.columns]
                df_display = top_by_srz[cols2].copy()
                df_display.rename(columns={'freq_specificity': 'FSI'}, inplace=True)
                num_cols = df_display.select_dtypes(include=[np.number]).columns
                for col in num_cols:
                    if col in ['t_start','t_end','duration_s']:
                        df_display[col] = df_display[col].apply(lambda v: f"{v:.0f}" if pd.notnull(v) else "nan")
                    else:
                        df_display[col] = df_display[col].apply(lambda v: f"{v:.3f}" if pd.notnull(v) else "nan")
                print(f"\nTop events by Score ({harmonic_labels[0]}_z_max * msc * plv / (1 + HSI)):")
                print(df_display.to_string(index=False, justify='center'))
            except Exception:
                pass

            print("")
        else:
            if raw_event_count > 0:
                print("No events pass the display filter.")
            print("")
        # print(f"\nFiles written to: {out_dir}")
        # print("  - sr_env_z.png, R_timeseries.png, ETA_zR.png, MaxH_hz_distribution.png")
        # print("  - events.csv, summary.csv, event_passport.csv")

    result = {
        'events': events_export,
        'summary': summary,
        'ignition_windows': ignition_windows_final,
        'ignition_windows_rounded': ignition_windows_rounded_final,
        'ignition_windows_path': ign_json_path,
        'fs': fs,
        't_R': t_cent,
        'zR': zR,
        'z_sr': z,  # SR envelope z-score for debugging
        't_sr': t,  # time array for SR envelope
        'ETA_tau': tau,
        'ETA_mean': eta_mean,
        'ETA_sem': eta_sem,
        'out_dir': out_dir,
        'figs': {
            'sr_env': os.path.join(out_dir,'sr_env_z.png'),
            'R_timeseries': os.path.join(out_dir,'R_timeseries.png'),
            'ETA_zR': os.path.join(out_dir,'ETA_zR.png'),
            'MaxH_hz_distribution': os.path.join(out_dir,'MaxH_hz_distribution.png')
        },
        'harmonics_used_hz': np.array(valid_harmonics, dtype=float),
        'harmonics_source': ('custom' if (harmonics_hz and len(harmonics_hz)) else 'multiples'),
        'harmonic_labels': harmonic_labels
    }
    return result, ignition_windows_rounded_final

# -----------------------------
# Plotting helpers: PSD & RBP
# -----------------------------


def _extract_eeg_matrix(RECORDZ: pd.DataFrame, eeg_channels: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray, float, List[str]]:
    """Return X (n_ch,n_samp), time vector t, fs, and channel list from RECORDZ."""
    time_col = ensure_timestamp_column(RECORDZ, time_col='Timestamp', default_fs=128.0)
    fs = infer_fs(RECORDZ, time_col)
    t = pd.to_numeric(RECORDZ[time_col], errors='coerce').values.astype(float)
    if eeg_channels is None:
        eeg_channels = [c for c in RECORDZ.columns if c.startswith('EEG.')]
    X = np.vstack([get_series(RECORDZ, ch) for ch in eeg_channels])
    L = min(map(len, X))
    return X[:, :L], t[:L], fs, eeg_channels


def _welch_psd(x: np.ndarray, fs: float, nperseg: Optional[int] = None, noverlap: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Welch PSD for a 1-D signal x."""
    if nperseg is None:
        nperseg = int(round(2.0*fs))
    if noverlap is None:
        noverlap = nperseg//2
    f, Pxx = signal.welch(x, fs=fs, nperseg=nperseg, noverlap=noverlap)
    return f, Pxx


def _band_power_from_psd(f: np.ndarray, Pxx: np.ndarray, f_lo: float, f_hi: float) -> float:
    mask = (f >= f_lo) & (f <= f_hi)
    if not np.any(mask):
        return 0.0
    return float(np.trapz(Pxx[mask], f[mask]))


def plot_psd_pre_peak_post(
    RECORDZ: pd.DataFrame,
    events_df: pd.DataFrame,
    event_index: int,
    eeg_channels: Optional[List[str]] = None,
    center_hz: float = 7.83,
    harmonics_hz: Optional[List[float]] = None,
    harmonic_bw_hz: float = 0.35,
    out_path: str = 'psd_pre_peak_post.png'
) -> str:
    """
    Make the hero PSD overlay (baseline, crest, afterglow) for a single event with harmonic annotations.
    Windows (relative to t0_net): baseline [-5,-2] s, crest [-1.5,+1.5] s, afterglow [+2,+5] s.
    """
    assert 0 <= event_index < len(events_df)
    row = events_df.iloc[event_index]
    t0 = float(row['t0_net'])

    X, t, fs, chans = _extract_eeg_matrix(RECORDZ, eeg_channels)
    x_mean = X.mean(axis=0)

    def _slice(left, right):
        i0 = max(0, int(round((t0+left - t[0])*fs)))
        i1 = min(len(x_mean), int(round((t0+right - t[0])*fs)))
        return i0, i1

    # segments
    i0_b, i1_b = _slice(-5.0, -2.0)
    i0_c, i1_c = _slice(-1.5, +1.5)
    i0_a, i1_a = _slice(+2.0, +5.0)

    segs = {
        'Baseline': x_mean[i0_b:i1_b],
        'Crest':    x_mean[i0_c:i1_c],
        'Afterglow':x_mean[i0_a:i1_a]
    }

    plt.figure(figsize=(8,4))
    colors = {'Baseline':'#888888','Crest':'#d62728','Afterglow':'#1f77b4'}
    f_peak = {}
    for label, seg in segs.items():
        if seg.size < int(fs):
            continue
        f, Pxx = _welch_psd(seg, fs)
        # dB scale for readability
        Pxx_db = 10.0*np.log10(Pxx + 1e-18)
        plt.plot(f, Pxx_db, lw=1.6, label=label, color=colors[label])
        f_peak[label] = (f, Pxx_db)

    # Harmonic lines
    if harmonics_hz is None or len(harmonics_hz) == 0:
        harmonics_hz = [center_hz * k for k in (1,2,3,4,5,6) if center_hz*k < fs/2.0]
    for f0 in harmonics_hz:
        plt.axvspan(f0-harmonic_bw_hz, f0+harmonic_bw_hz, color='orange', alpha=0.1)
        plt.axvline(f0, color='orange', alpha=0.6, lw=0.8)

    plt.xlim(2, 60)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('PSD (dB)')
    plt.title('PSD: Baseline vs Crest vs Afterglow (Event #{})'.format(event_index))
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()
    return out_path


def plot_harmonic_rbp_bar(
    RECORDZ: pd.DataFrame,
    events_df: pd.DataFrame,
    event_index: int,
    eeg_channels: Optional[List[str]] = None,
    center_hz: float = 7.83,
    harmonics_hz: Optional[List[float]] = None,
    harmonic_bw_hz: float = 0.35,
    total_band: Tuple[float,float] = (4.0, 60.0),
    out_path: str = 'harmonic_rbp_bar.png'
) -> str:
    """Barplot of Relative Band Power for each harmonic at crest vs baseline for one event."""
    assert 0 <= event_index < len(events_df)
    row = events_df.iloc[event_index]
    t0 = float(row['t0_net'])

    X, t, fs, chans = _extract_eeg_matrix(RECORDZ, eeg_channels)
    x_mean = X.mean(axis=0)

    def _slice(left, right):
        i0 = max(0, int(round((t0+left - t[0])*fs)))
        i1 = min(len(x_mean), int(round((t0+right - t[0])*fs)))
        return i0, i1

    # Define windows
    i0_b, i1_b = _slice(-5.0, -2.0)   # baseline
    i0_c, i1_c = _slice(-1.5, +1.5)   # crest

    seg_b = x_mean[i0_b:i1_b]; seg_c = x_mean[i0_c:i1_c]
    f_b, P_b = _welch_psd(seg_b, fs); f_c, P_c = _welch_psd(seg_c, fs)

    if harmonics_hz is None or len(harmonics_hz) == 0:
        harmonics_hz = [center_hz * k for k in (1,2,3,4,5,6) if center_hz*k < fs/2.0]

    # total power for RBP normalization
    Ptot_b = _band_power_from_psd(f_b, P_b, total_band[0], total_band[1])
    Ptot_c = _band_power_from_psd(f_c, P_c, total_band[0], total_band[1])

    rbp_b, rbp_c = [], []
    for f0 in harmonics_hz:
        rbp_b.append(_band_power_from_psd(f_b, P_b, f0-harmonic_bw_hz, f0+harmonic_bw_hz) / (Ptot_b + 1e-18))
        rbp_c.append(_band_power_from_psd(f_c, P_c, f0-harmonic_bw_hz, f0+harmonic_bw_hz) / (Ptot_c + 1e-18))

    inds = np.arange(len(harmonics_hz))
    width = 0.38
    plt.figure(figsize=(8,4))
    plt.bar(inds - width/2, rbp_b, width=width, color='#888888', label='Baseline')
    plt.bar(inds + width/2, rbp_c, width=width, color='#d62728', label='Crest')
    for i,(b,c) in enumerate(zip(rbp_b, rbp_c)):
        plt.text(i - 0.25, max(b,c)+0.005, f"Δ={100*(c-b):.1f}%", fontsize=8)
    plt.xticks(inds, [f"{f0:.2f}" for f0 in harmonics_hz], rotation=0)
    plt.xlabel('Harmonic center (Hz)')
    plt.ylabel('Relative Band Power (fraction of {}–{} Hz)'.format(*total_band))
    plt.title('Harmonic RBP: Baseline vs Crest (Event #{})'.format(event_index))
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()
    return out_path


def make_ignition_hero_figures(
    RECORDZ: pd.DataFrame,
    events_csv_path: str,
    event_index: int,
    eeg_channels: Optional[List[str]] = None,
    center_hz: float = 7.83,
    harmonics_hz: Optional[List[float]] = None,
    harmonic_bw_hz: float = 0.35,
    out_dir: str = '.'
) -> Dict[str,str]:
    """Convenience wrapper: generates the PSD overlay and the harmonic RBP barplot for one event."""
    events_df = pd.read_csv(events_csv_path)
    os.makedirs(out_dir, exist_ok=True)
    psd_path = os.path.join(out_dir, f'psd_pre_peak_post_evt{event_index}.png')
    rbp_path = os.path.join(out_dir, f'harmonic_rbp_bar_evt{event_index}.png')
    plot_psd_pre_peak_post(RECORDZ, events_df, event_index, eeg_channels, center_hz, harmonics_hz, harmonic_bw_hz, psd_path)
    plot_harmonic_rbp_bar(RECORDZ, events_df, event_index, eeg_channels, center_hz, harmonics_hz, harmonic_bw_hz, (4,60), rbp_path)
    return {'psd': psd_path, 'rbp': rbp_path}


# -----------------------------
# Animation: Relative Band Power (per band)
# -----------------------------

def _compute_rbp_timeseries(
    RECORDZ: pd.DataFrame,
    eeg_channels: Optional[List[str]] = None,
    time_col: str = 'Timestamp',
    t_range: Optional[Tuple[float,float]] = None,
    bands: Optional[List[Tuple[str,Tuple[float,float]]]] = None,
    total_band: Tuple[float,float] = (4.0, 60.0),
    win_sec: float = 1.0,
    step_sec: float = 0.1,
    combine: str = 'mean'
) -> Tuple[np.ndarray, np.ndarray, List[str], float]:
    """
    Compute sliding-window Relative Band Power (RBP) time series.
    Returns times (t_mid), RBP array (n_bands x n_times), band labels, fs.
    """
    if bands is None:
        bands = [
            ('Delta', (0.5, 4.0)),
            ('Theta', (4.0, 8.0)),
            ('Alpha', (8.0, 12.0)),
            ('BetaL', (12.0, 20.0)),
            ('BetaH', (20.0, 35.0)),
            ('Gamma', (35.0, 60.0)),
        ]
    time_col = ensure_timestamp_column(RECORDZ, time_col=time_col, default_fs=128.0)
    fs = infer_fs(RECORDZ, time_col)
    t = pd.to_numeric(RECORDZ[time_col], errors='coerce').values.astype(float)
    if eeg_channels is None:
        eeg_channels = [c for c in RECORDZ.columns if c.startswith('EEG.')]
    X = np.vstack([get_series(RECORDZ, ch) for ch in eeg_channels])
    L = min(map(len, X))
    X = X[:, :L]; t = t[:L]

    if t_range is None:
        t0, t1 = float(t[0]), float(t[-1])
    else:
        t0, t1 = t_range
    i0 = max(0, int(round((t0 - t[0])*fs)))
    i1 = min(L, int(round((t1 - t[0])*fs)))

    if combine == 'mean':
        x = X[:, i0:i1].mean(axis=0)
    elif combine == 'median':
        x = np.median(X[:, i0:i1], axis=0)
    else:
        # assume combine is a single channel name
        if combine in RECORDZ.columns:
            x = get_series(RECORDZ, combine)[i0:i1]
        else:
            # fallback to first requested channel
            x = X[0, i0:i1]

    nwin = int(round(win_sec*fs)); nstep = int(round(step_sec*fs))
    t_mids, rbp_list = [], []

    # Precompute FFT frequency grid via Welch
    for s in range(i0, i1 - nwin + 1, nstep):
        seg = x[s - i0 : s - i0 + nwin]
        f, Pxx = signal.welch(seg, fs=fs, nperseg=nwin)
        # Compute power per band and normalize so each time window sums to 1.0
        powers = []
        for _, (lo, hi) in bands:
            powers.append(_band_power_from_psd(f, Pxx, lo, hi))
        Psum = float(np.sum(powers)) + 1e-18
        rbp = [p / Psum for p in powers]
        rbp_list.append(rbp)
        t_mid = t[0] + (s + nwin/2)/fs
        t_mids.append(t_mid)

    RBP = np.array(rbp_list).T  # n_bands x n_times
    return np.array(t_mids), RBP, [b[0] for b in bands], fs


def animate_rbp(
    RECORDZ: pd.DataFrame,
    eeg_channels: Optional[List[str]] = None,
    combine: str = 'mean',  # 'mean' | 'median' | channel name like 'EEG.F4'
    time_col: str = 'Timestamp',
    t_range: Optional[Tuple[float,float]] = None,
    bands: Optional[List[Tuple[str,Tuple[float,float]]]] = None,
    total_band: Tuple[float,float] = (4.0, 60.0),
    win_sec: float = 1.0,
    step_sec: float = 0.1,
    fps: int = 20,
    out_path: str = 'rbp_animation.mp4',
    show_inline: bool = False,
    view_sec: Optional[float] = None,  # <-- fixed-width sliding window (e.g., 20.0),
    fill_alpha: float = 0.0
):
    """
    Create an animation of the stacked Relative Band Power over time for a selected electrode or
    a combined group (mean/median). Saves to MP4/GIF based on extension. If show_inline=True,
    returns the matplotlib.animation object for Jupyter display.
    """
    import matplotlib.animation as animation

    t_mid, RBP, labels, fs = _compute_rbp_timeseries(
        RECORDZ, eeg_channels=eeg_channels, time_col=time_col, t_range=t_range,
        bands=bands, total_band=total_band, win_sec=win_sec, step_sec=step_sec, combine=combine)

    if RBP.size == 0:
        raise ValueError('No RBP data to animate (check t_range or window settings).')

    color_map = {
        'Delta':'#1f77b4', 'Theta':'#ff7f0e', 'Alpha':'#2ca02c',
        'BetaL':'#d62728', 'BetaH':'#9467bd', 'Gamma':'#8c564b'
    }
    colors = [color_map.get(lbl, None) for lbl in labels]

    fig, ax = plt.subplots(figsize=(8,3.2))
    # leave room on the right for an outside legend
    fig.subplots_adjust(right=0.78)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Relative band power (fraction)')
    ax.set_title(f'RBP (per band) — {combine} of {len(eeg_channels) if eeg_channels else "all EEG"} channels')

    # helper to get left index for a fixed view window ending at t_mid[k]
    import bisect
    def _left_index(k):
        if view_sec is None:
            return 0
        t_right = t_mid[k]
        t_left  = t_right - view_sec
        i_left  = bisect.bisect_left(t_mid, t_left)
        return max(0, i_left)

    def _draw_frame(k):
        ax.clear()
        ax.set_ylim(0, 1)
        ax.set_xlabel('Time (s)'); ax.set_ylabel('Relative band power (fraction)')
        ax.set_title(f'RBP (per band) — {combine} of {len(eeg_channels) if eeg_channels else "all EEG"} channels')
        i_left = _left_index(k)
        t_slice = t_mid[i_left:k+1]
        rbp_slice = RBP[:, i_left:k+1]
        # Set x-limits to fixed sliding window if requested
        if view_sec is not None:
            ax.set_xlim(t_mid[k] - view_sec, t_mid[k])
        else:
            ax.set_xlim(t_mid[0], t_mid[-1])
        # Plot each band as its own line (not stacked)
        for i, (lbl, col) in enumerate(zip(labels, colors)):
            y = rbp_slice[i]
            ax.plot(t_slice, y, color=col, lw=1.8, label=lbl, zorder=3)
            if fill_alpha and fill_alpha > 0:
                ax.fill_between(t_slice, 0, y, color=col, alpha=fill_alpha, zorder=2)
        # Legend outside to the right (always visible during animation)
        ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=8, ncol=1)
        return []

    anim = animation.FuncAnimation(fig, _draw_frame, frames=len(t_mid), interval=1000/fps, blit=False)

    # Save using writer based on extension
    ext = out_path.split('.')[-1].lower()
    if ext in ('mp4','m4v','mov'):
        try:
            writer = animation.FFMpegWriter(fps=fps, bitrate=2000)
            anim.save(out_path, writer=writer, dpi=140)
        except Exception:
            # fallback if ffmpeg missing → save GIF instead
            from matplotlib.animation import PillowWriter
            gif_path = out_path.rsplit('.', 1)[0] + '.gif'
            anim.save(gif_path, writer=PillowWriter(fps=fps))
            out_path = gif_path
    elif ext in ('gif',):
        from matplotlib.animation import PillowWriter
        anim.save(out_path, writer=PillowWriter(fps=fps))
    else:
        # default to mp4
        try:
            writer = animation.FFMpegWriter(fps=fps, bitrate=2000)
            anim.save(out_path, writer=writer, dpi=140)
        except Exception:
            from matplotlib.animation import PillowWriter
            gif_path = out_path + '.gif'
            anim.save(gif_path, writer=PillowWriter(fps=fps))
            out_path = gif_path

    plt.close(fig)

    if show_inline:
        return anim, out_path
    return out_path

# -----------------------------
# Delta-band power surge scanner (micro-spectrogram)
# -----------------------------

def plot_delta_spectrogram(
    RECORDZ: pd.DataFrame,
    eeg_channels: Optional[List[str]] = None,
    combine: str = 'mean',      # 'mean' | 'median' | channel name e.g. 'EEG.F4'
    time_col: str = 'Timestamp',
    t_range: Optional[Tuple[float,float]] = None,
    baseline_range: Optional[Tuple[float,float]] = None,
    f_lo: float = 0.5,
    f_hi: float = 4.0,
    win_sec: float = 12.0,
    step_sec: float = 1.0,
    out_path: str = 'delta_spectrogram.png',
    show: bool = True,
    return_peaks: bool = False
):
    """
    Fast view of *which delta frequencies* (0.5–4 Hz) show surges.
    Computes sliding-window Welch PSD → z-scores vs baseline per frequency → heatmap (time × freq).

    Returns `out_path` (and optional peaks DataFrame if `return_peaks=True`).
    """
    # Extract data
    time_col = ensure_timestamp_column(RECORDZ, time_col=time_col, default_fs=128.0)
    fs = infer_fs(RECORDZ, time_col)
    t = pd.to_numeric(RECORDZ[time_col], errors='coerce').values.astype(float)
    if eeg_channels is None:
        eeg_channels = [c for c in RECORDZ.columns if c.startswith('EEG.')]
    X = np.vstack([get_series(RECORDZ, ch) for ch in eeg_channels])
    L = min(map(len, X)); X = X[:, :L]; t = t[:L]

    # Time indices
    if t_range is None:
        t0, t1 = float(t[0]), float(t[-1])
    else:
        t0, t1 = t_range
    i0 = max(0, int(round((t0 - t[0])*fs)))
    i1 = min(L, int(round((t1 - t[0])*fs)))

    # Combine channels
    if combine == 'mean':
        x = X[:, i0:i1].mean(axis=0)
    elif combine == 'median':
        x = np.median(X[:, i0:i1], axis=0)
    else:
        x = get_series(RECORDZ, combine)[i0:i1] if combine in RECORDZ.columns else X[0, i0:i1]

    nwin = int(round(win_sec*fs)); nstep = int(round(step_sec*fs))
    t_mids, rows = [], []
    f_ref, mask = None, None

    # Slide and compute PSD rows
    for s in range(0, len(x) - nwin + 1, nstep):
        seg = x[s:s+nwin]
        f, Pxx = signal.welch(seg, fs=fs, nperseg=nwin)
        if f_ref is None:
            f_ref = f
            mask = (f_ref >= f_lo) & (f_ref <= f_hi)
        rows.append(Pxx[mask])
        t_mids.append(t0 + (s + nwin/2)/fs)

    if not rows:
        raise ValueError('No windows for given settings; decrease win_sec or expand t_range.')

    P = np.vstack(rows)              # shape: n_times × n_freqs(delta)
    f_delta = f_ref[mask]
    t_mid = np.array(t_mids)

    # Baseline (first N windows or given range)
    if baseline_range is not None:
        b0 = max(0, int(round((baseline_range[0] - t0)*fs)))
        b1 = min(len(x), int(round((baseline_range[1] - t0)*fs)))
        # windows overlapping baseline
        b_inds = [k for k in range(0, len(x) - nwin + 1, nstep)
                  if (k >= b0) and (k+nwin <= b1)]
        if not b_inds:
            # fallback: first 10 windows
            nB = min(10, P.shape[0])
            B = P[:nB]
        else:
            B = np.vstack([P[(k//nstep)] for k in b_inds if (k//nstep) < P.shape[0]])
    else:
        nB = min(10, P.shape[0])
        B = P[:nB]

    mu = np.nanmean(B, axis=0)
    sd = np.nanstd(B, axis=0) + 1e-12
    Z = (P - mu) / sd   # z-score per frequency over time

    # Plot heatmap (time × freq)
    plt.figure(figsize=(8, 3.2))
    plt.pcolormesh(t_mid, f_delta, Z.T, shading='auto', cmap='magma')
    plt.colorbar(label='Δ Power (z)')
    plt.xlabel('Time (s)'); plt.ylabel('Frequency (Hz)')
    plt.title(f'Delta micro-spectrogram (win={win_sec:.1f}s, step={step_sec:.1f}s) — {combine}')
    plt.tight_layout(); plt.savefig(out_path, dpi=140)
    if show: plt.show()
    plt.close()

    if not return_peaks:
        return out_path

    # Peak tracker: for each time, the max-z delta freq and its z
    pk_idx = np.nanargmax(Z, axis=1)
    pk_freq = f_delta[pk_idx]
    pk_z = Z[np.arange(Z.shape[0]), pk_idx]
    df_peaks = pd.DataFrame({'t_mid': t_mid, 'f_peak_hz': pk_freq, 'z_peak': pk_z})
    return out_path, df_peaks

# -----------------------------
# Delta peak extraction & cohort hotspots
# -----------------------------

def _parabolic_peak_refine(f: np.ndarray, y: np.ndarray, i: int) -> float:
    """Quadratic (parabolic) interpolation around bin i (use log power).
    Returns refined frequency in Hz.
    """
    if i <= 0 or i >= len(y)-1:
        return float(f[i])
    y0, y1, y2 = np.log(y[i-1]+1e-18), np.log(y[i]+1e-18), np.log(y[i+1]+1e-18)
    denom = (y0 - 2*y1 + y2)
    if abs(denom) < 1e-18:
        return float(f[i])
    delta = 0.5*(y0 - y2)/denom
    df = f[1]-f[0]
    return float(f[i] + delta*df)


def delta_peaks_for_event(
    RECORDZ: pd.DataFrame,
    t0_net: float,
    eeg_channels: Optional[List[str]] = None,
    combine: str = 'mean',        # 'mean' | 'median' | channel name
    time_col: str = 'Timestamp',
    crest_win: float = 3.0,       # PSD window centered at t0
    baseline_range: Optional[Tuple[float,float]] = None,
    f_lo: float = 0.5,
    f_hi: float = 4.0,
    top_n: int = 3
) -> pd.DataFrame:
    """Return up to top_n delta-frequency peaks at the ignition crest with z vs baseline.
    Columns: f_hz (refined), z_surge, raw_power, baseline_mu, baseline_sd.
    """
    # Extract signal
    time_col = ensure_timestamp_column(RECORDZ, time_col=time_col, default_fs=128.0)
    fs = infer_fs(RECORDZ, time_col)
    t = pd.to_numeric(RECORDZ[time_col], errors='coerce').values.astype(float)
    if eeg_channels is None:
        eeg_channels = [c for c in RECORDZ.columns if c.startswith('EEG.')]
    X = np.vstack([get_series(RECORDZ, ch) for ch in eeg_channels])
    L = min(map(len, X)); X = X[:, :L]; t = t[:L]

    # Combine
    if combine == 'mean':
        x = X.mean(axis=0)
    elif combine == 'median':
        x = np.median(X, axis=0)
    else:
        x = get_series(RECORDZ, combine)

    # Crest window PSD
    i0 = max(0, int(round((t0_net - crest_win/2 - t[0])*fs)))
    i1 = min(L, int(round((t0_net + crest_win/2 - t[0])*fs)))
    seg = x[i0:i1]
    f, P = signal.welch(seg, fs=fs, nperseg=int(round(crest_win*fs)))
    mask = (f >= f_lo) & (f <= f_hi)
    fD, PD = f[mask], P[mask]

    # Baseline PSD
    if baseline_range is not None:
        b0 = max(0, int(round((baseline_range[0]-t[0])*fs)))
        b1 = min(L, int(round((baseline_range[1]-t[0])*fs)))
        bseg = x[b0:b1]
    else:
        # default: equally sized pre-crest chunk
        j1 = max(0, i0 - (i1 - i0))
        bseg = x[j1:i0]
    fb, Pb = signal.welch(bseg, fs=fs, nperseg=int(round(crest_win*fs)))
    PbD = Pb[(fb >= f_lo) & (fb <= f_hi)]
    mu, sd = float(np.mean(PbD)), float(np.std(PbD) + 1e-18)

    # Find peaks in delta band
    from scipy.signal import find_peaks
    peak_idx, _ = find_peaks(PD, distance=max(1, int(0.2/((fD[1]-fD[0]) or 1e-6))))
    if peak_idx.size == 0:
        return pd.DataFrame(columns=['f_hz','z_surge','raw_power','baseline_mu','baseline_sd'])

    # Rank by z and keep top_n
    zvals = (PD[peak_idx] - mu) / sd
    order = np.argsort(zvals)[::-1][:top_n]

    rows = []
    for j in order:
        i = int(peak_idx[j])
        f_refined = _parabolic_peak_refine(fD, PD, i)
        rows.append({'f_hz': f_refined,
                     'z_surge': float(zvals[j]),
                     'raw_power': float(PD[i]),
                     'baseline_mu': mu,
                     'baseline_sd': sd})
    return pd.DataFrame(rows)


def summarize_delta_hotspots(
    RECORDZ: pd.DataFrame,
    events_df: pd.DataFrame,
    eeg_channels: Optional[List[str]] = None,
    combine: str = 'mean',
    time_col: str = 'Timestamp',
    crest_win: float = 3.0,
    baseline_offset: Tuple[float,float] = (-10.0, -5.0),
    f_lo: float = 0.5, f_hi: float = 4.0,
    top_n: int = 2,
    out_path: str = 'delta_hotspots.png'
) -> Tuple[pd.DataFrame, str]:
    """Scan all events for delta crest peaks, aggregate, and plot KDE+hist of hotspots.
    Returns (peaks_dataframe, figure_path).
    """
    all_rows = []
    for _, row in events_df.iterrows():
        t0 = float(row['t0_net'])
        peaks = delta_peaks_for_event(
            RECORDZ, t0, eeg_channels=eeg_channels, combine=combine, time_col=time_col,
            crest_win=crest_win,
            baseline_range=(t0+baseline_offset[0], t0+baseline_offset[1]),
            f_lo=f_lo, f_hi=f_hi, top_n=top_n
        )
        if not peaks.empty:
            peaks = peaks.assign(t0_net=t0)
            all_rows.append(peaks)
    if not all_rows:
        return pd.DataFrame(columns=['f_hz','z_surge','t0_net']), ''

    DF = pd.concat(all_rows, ignore_index=True)

    # Plot histogram + KDE
    import seaborn as sns
    plt.figure(figsize=(7.5,3.2))
    sns.histplot(DF['f_hz'], bins=np.linspace(f_lo, f_hi, 30), stat='count', color='#2ca02c', alpha=0.35, edgecolor='k')
    try:
        sns.kdeplot(DF['f_hz'], bw_adjust=0.5, color='#d62728', lw=2)
    except Exception:
        pass
    plt.xlabel('Delta peak frequency at crest (Hz)')
    plt.ylabel('Event count')
    plt.title('Delta hotspots across ignitions')
    plt.tight_layout(); plt.savefig(out_path, dpi=140)
    plt.close()
    return DF[['t0_net','f_hz','z_surge']], out_path

# -----------------------------
# Option A: MeanShift clustering of delta surge frequencies (with safe fallback)
# -----------------------------

def cluster_delta_hotspots_meanshift(
    DF: pd.DataFrame,
    z_thresh: float = 2.0,
    bandwidth_quantile: float = 0.2,
    fallback_bw: float = 0.05,
    B: int = 2000,
    alpha: float = 0.05
) -> pd.DataFrame:
    """Cluster crest delta peaks (Hz) using MeanShift with robust bandwidth fallback,
    and return cluster centers with 95% bootstrap CIs and counts.

    Parameters
    ----------
    DF : DataFrame with columns ['t0_net','f_hz','z_surge'] as returned by summarize_delta_hotspots.
    z_thresh : keep only peaks with z_surge >= z_thresh (default 2.0) for surges.
    bandwidth_quantile : quantile passed to sklearn.estimate_bandwidth.
    fallback_bw : minimal bandwidth (Hz) if estimate_bandwidth returns <= 0.
    B : bootstrap iterations for CI.
    alpha : CI level (default 0.05 → 95% CI).

    Returns
    -------
    DataFrame with columns:
      center_hz, ci_low, ci_high, n_events, mean_z, median_z, bandwidth
    Sorted by n_events then mean_z.
    """
    import numpy as np
    import pandas as pd
    from sklearn.cluster import MeanShift, estimate_bandwidth

    surge = DF.loc[DF['z_surge'] >= z_thresh].copy()
    if surge.empty:
        return pd.DataFrame(columns=['center_hz','ci_low','ci_high','n_events','mean_z','median_z','bandwidth'])

    X = surge[['f_hz']].values.astype(float)

    # bandwidth estimation with safe fallback
    try:
        bw = estimate_bandwidth(X, quantile=bandwidth_quantile, n_samples=min(len(X), 500))
    except Exception:
        bw = 0.0
    if (not np.isfinite(bw)) or (bw <= 1e-6):
        rng = float(np.max(X) - np.min(X))
        bw = max(fallback_bw, 0.10 * (rng + 1e-6))

    ms = MeanShift(bandwidth=bw, bin_seeding=True).fit(X)
    surge['cluster'] = ms.labels_

    # bootstrap CI for weighted mean center
    rng = np.random.default_rng(0)
    def boot_ci(vals, weights):
        vals = np.asarray(vals)
        w = np.asarray(weights)
        w = w / (w.sum() + 1e-18)
        boots = []
        for _ in range(B):
            idx = rng.choice(len(vals), size=len(vals), replace=True, p=w)
            boots.append(np.average(vals[idx], weights=w[idx]))
        lo, hi = np.percentile(boots, [100*alpha/2, 100*(1-alpha/2)])
        return float(lo), float(hi)

    rows = []
    for k in sorted(surge['cluster'].unique()):
        sub = surge.loc[surge['cluster']==k]
        center = float(np.average(sub['f_hz'], weights=sub['z_surge']))
        lo, hi = boot_ci(sub['f_hz'].values, sub['z_surge'].values)
        rows.append({
            'center_hz': center,
            'ci_low': lo,
            'ci_high': hi,
            'n_events': int(len(sub)),
            'mean_z': float(sub['z_surge'].mean()),
            'median_z': float(sub['z_surge'].median()),
            'bandwidth': float(bw)
        })

    out = (pd.DataFrame(rows)
           .sort_values(['n_events','mean_z'], ascending=[False, False])
           .reset_index(drop=True))
    return out

# -----------------------------
# Delta PSD animation (frequency sweep over time) — v2 with dynamic y-limits + fill shading
# -----------------------------

def animate_delta_psd(
    RECORDZ: pd.DataFrame,
    eeg_channels: Optional[List[str]] = None,
    combine: str = 'mean',          # 'mean' | 'median' | channel name like 'EEG.F4'
    time_col: str = 'Timestamp',
    t_range: Optional[Tuple[float,float]] = None,   # (t0, t1) seconds
    f_lo: float = 0.5,
    f_hi: float = 4.0,
    win_sec: float = 12.0,
    step_sec: float = 0.25,
    detrend: bool = True,
    norm: str = 'z',                # 'z' | 'rel' | None
    baseline_range: Optional[Tuple[float,float]] = None,
    fps: int = 15,
    out_path: str = 'delta_psd_anim.mp4',
    show_inline: bool = False,
    title: Optional[str] = None,
    fill_alpha: float = 0.15,       # <-- shaded fill under curve
    dyn_ylim: bool = True,          # <-- update y-limits per frame to avoid clipping
    ylim_pad: float = 1.10          # <-- multiplier pad for headroom
):
    """
    Animate delta-band PSD over time for selected electrodes.
    x-axis: frequency in [f_lo, f_hi]; y-axis: PSD response per frequency.

    dyn_ylim = True  → compute per-frame max and update ax.set_ylim each frame
    fill_alpha > 0   → draw filled area under the curve for visibility
    """
    import matplotlib.animation as animation

    # --- 1) Extract & combine data
    time_col = ensure_timestamp_column(RECORDZ, time_col=time_col, default_fs=128.0)
    fs = infer_fs(RECORDZ, time_col)
    t = pd.to_numeric(RECORDZ[time_col], errors='coerce').values.astype(float)
    if eeg_channels is None:
        eeg_channels = [c for c in RECORDZ.columns if c.startswith('EEG.')]
    X = np.vstack([get_series(RECORDZ, ch) for ch in eeg_channels])
    L = min(map(len, X)); X = X[:, :L]; t = t[:L]

    if t_range is None:
        t0, t1 = float(t[0]), float(t[-1])
    else:
        t0, t1 = t_range
    i0 = max(0, int(round((t0 - t[0])*fs)))
    i1 = min(L, int(round((t1 - t[0])*fs)))

    if combine == 'mean':
        x = X[:, i0:i1].mean(axis=0)
    elif combine == 'median':
        x = np.median(X[:, i0:i1], axis=0)
    else:
        x = get_series(RECORDZ, combine)[i0:i1] if combine in RECORDZ.columns else X[0, i0:i1]

    if detrend:
        x = signal.detrend(x)

    # --- 2) Sliding-window PSDs
    nwin = int(round(win_sec*fs)); nstep = int(round(step_sec*fs))
    frames, t_mids = [], []
    f_ref = None

    for s in range(0, len(x) - nwin + 1, nstep):
        seg = x[s:s+nwin]
        f, Pxx = signal.welch(seg, fs=fs, nperseg=nwin)
        if f_ref is None:
            f_ref = f
        frames.append(Pxx)
        t_mids.append(t0 + (s + nwin/2)/fs)

    if not frames:
        raise ValueError('No frames to animate; adjust t_range/win_sec/step_sec.')

    F = np.vstack(frames)           # n_frames × n_freqs
    mask = (f_ref >= f_lo) & (f_ref <= f_hi)
    f_delta = f_ref[mask]
    F = F[:, mask]

    # --- 3) Normalize
    if norm == 'z':
        if baseline_range is not None:
            b0 = max(0, int(round((baseline_range[0]-t0)*fs)))
            b1 = min(len(x), int(round((baseline_range[1]-t0)*fs)))
            b_idx = [k for k in range(0, len(x) - nwin + 1, nstep) if (k >= b0) and (k+nwin <= b1)]
            if b_idx:
                B = F[[k//nstep for k in b_idx if (k//nstep) < F.shape[0]]]
            else:
                B = F[:min(10, F.shape[0])]
        else:
            B = F[:min(10, F.shape[0])]
        mu = np.nanmean(B, axis=0); sd = np.nanstd(B, axis=0) + 1e-12
        Fn = (F - mu) / sd
    elif norm == 'rel':
        A = np.trapz(F, f_delta, axis=1)[:, None] + 1e-18
        Fn = F / A
    else:
        Fn = F

    # --- 4) Build animation
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    ax.set_xlim(f_lo, f_hi)
    base_ylim = float(np.nanpercentile(Fn, 95))
    ax.set_ylim(0, base_ylim*ylim_pad)
    if title:
        ax.set_title(title)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel({'z':'PSD z-score','rel':'Relative PSD','None':'PSD'}[str(norm)])

    line, = ax.plot([], [], lw=2, color='#1f77b4', zorder=3)
    fill_poly = [None]  # holder for the current PolyCollection

    def _init():
        line.set_data([], [])
        return (line,)

    def _update(i):
        y = Fn[i]
        if dyn_ylim:
            ytop = float(np.nanmax(y)) * ylim_pad
            if ytop > 0:
                ax.set_ylim(0, ytop)
        line.set_data(f_delta, y)
        # shaded fill under the curve (remove previous PolyCollection safely)
        if fill_poly[0] is not None:
            try:
                fill_poly[0].remove()
            except Exception:
                pass
        if fill_alpha > 0:
            fill_poly[0] = ax.fill_between(f_delta, 0, y, color='#1f77b4', alpha=fill_alpha, zorder=2)
        ax.set_title((title or 'Delta PSD') + f"Window center: t = {t_mids[i]:.2f} s")
        return (line,)

    anim = animation.FuncAnimation(fig, _update, init_func=_init,
                                   frames=Fn.shape[0], interval=1000/fps, blit=False)

    # --- 5) Save only if not showing inline (avoid ffmpeg requirement when embedding)
    if not show_inline:
        ext = out_path.split('.')[-1].lower()
        try:
            if ext in ('mp4','m4v','mov'):
                writer = animation.FFMpegWriter(fps=fps, bitrate=2000)
                anim.save(out_path, writer=writer, dpi=140)
            elif ext in ('gif',):
                from matplotlib.animation import PillowWriter
                anim.save(out_path, writer=PillowWriter(fps=fps))
            else:
                writer = animation.FFMpegWriter(fps=fps, bitrate=2000)
                anim.save(out_path, writer=writer, dpi=140)
        except Exception:
            from matplotlib.animation import PillowWriter
            gif_path = out_path.rsplit('.', 1)[0] + '.gif'
            anim.save(gif_path, writer=PillowWriter(fps=fps))
            out_path = gif_path
        plt.close(fig)
        return out_path

    # Inline case: return the animation object (no saving)
    plt.close(fig)
    return anim, out_path



# -----------------------------
# PSD animation — Stacked ABSOLUTE power over time (all bands or custom) — v3
#  * cumulative stacked area over time
#  * saves movie (MP4 if ffmpeg, else GIF fallback)
#  * optionally returns inline anim
#  * NEW: saves a static PNG of the **last frame** (full ignition window) for one-look overview
# -----------------------------

# -----------------------------------------------------------------------------
# Optional presets helpful for notebook discoverability. The function below also
# defines these internally if bands is None.
BAND_PRESETS = {
    'canonical': [
        ('Delta', (0.5, 4.0)),
        ('Theta', (4.0, 8.0)),
        ('Alpha', (8.0, 12.0)),
        ('BetaL', (12.0, 20.0)),
        ('BetaH', (20.0, 35.0)),
        ('Gamma', (35.0, 60.0)),
    ],
    'schumann': [
        ('SR1', (7.45, 8.15)), ('2x', (13.0, 15.0)), ('3x', (19.0, 21.0)), ('4x', (25.0, 28.0)),
        ('5x', (31.0, 35.0)), ('6x', (38.0, 42.0)), ('7x', (45.0, 48.0)), ('8x', (52.0, 54.0))
    ],
}


def animate_psd_stacked(
    RECORDZ: pd.DataFrame,
    eeg_channels: Optional[List[str]] = None,
    combine: str = 'mean',                  # 'mean' | 'median' | 'EEG.F4' etc.
    time_col: str = 'Timestamp',
    t_range: Optional[Tuple[float,float]] = None,   # (t0, t1) seconds
    bands: Optional[List[Tuple[str, Tuple[float,float]]]] = None,  # list of (label,(flo,fhi))
    default_bands: str = 'canonical',       # 'canonical' | 'schumann' | 'delta6' | 'custom'
    win_sec: float = 12.0,
    step_sec: float = 0.25,
    detrend: bool = True,
    fps: int = 15,
    out_path: str = 'psd_stacked.mp4',
    show_inline: bool = False,
    title: Optional[str] = None,
    dyn_ylim: bool = True,
    ylim_pad: float = 1.10,
    legend_outside: bool = True,
    save_last_frame: bool = True,
    last_frame_path: Optional[str] = None
):
    """
    Animate a *stacked area* of **absolute band power** (integrated PSD) over time across bands.

    • x-axis: time (sliding window centers).  • y-axis: stacked absolute power per band (μV²).
    • Saves to `out_path` (MP4 if ffmpeg is available; otherwise GIF fallback) and optionally returns inline anim.
    • If `save_last_frame=True`, also saves a static PNG of the **final stacked area** (full window) to `last_frame_path`.

    Presets for `bands` via `default_bands`:
      - 'canonical' → Delta (0.5–4), Theta (4–8), Alpha (8–12), BetaL (12–20), BetaH (20–35), Gamma (35–60)
      - 'schumann'  → SR1±0.35 (~7.45–8.15), 2× (~13–15), 3× (~19–21), 4× (~25–28), 5× (~31–35), 6× (~38–42), 7× (~45–48), 8× (~52–54)
      - 'delta6'    → six equal slices within 0.5–4.0 Hz
      - 'custom'    → pass explicit `bands=[('Label',(flo,fhi)), ...]`
    """
    import matplotlib.animation as animation

    # ---- 1) Extract & combine data
    # Ensure we have a timestamp column (helper provided by caller's environment)
    time_col = ensure_timestamp_column(RECORDZ, time_col=time_col, default_fs=128.0)
    fs = infer_fs(RECORDZ, time_col)

    # Time vector
    t = pd.to_numeric(RECORDZ[time_col], errors='coerce').values.astype(float)

    # Channels
    if eeg_channels is None:
        eeg_channels = [c for c in RECORDZ.columns if c.startswith('EEG.')]
        if not eeg_channels:
            raise ValueError("No EEG.* columns found. Provide `eeg_channels` explicitly.")

    X = np.vstack([get_series(RECORDZ, ch) for ch in eeg_channels])

    # Guard shapes
    L = min(map(len, X))
    X = X[:, :L]
    t = t[:L]

    # Time range indices
    if t_range is None:
        t0, t1 = float(t[0]), float(t[-1])
    else:
        t0, t1 = t_range
    i0 = max(0, int(round((t0 - t[0]) * fs)))
    i1 = min(L, int(round((t1 - t[0]) * fs)))
    if i1 - i0 <= 1:
        raise ValueError("t_range yields empty slice. Adjust t_range or check timestamp units.")

    # Combine signals
    if combine == 'mean':
        x = X[:, i0:i1].mean(axis=0)
    elif combine == 'median':
        x = np.median(X[:, i0:i1], axis=0)
    else:
        x = get_series(RECORDZ, combine)[i0:i1] if combine in RECORDZ.columns else X[0, i0:i1]

    x = np.asarray(x, dtype=float)
    if detrend:
        x = signal.detrend(x)

    # ---- 2) Bands
    if bands is None:
        if default_bands == 'canonical':
            bands = [
                ('Delta', (0.5, 4.0)), ('Theta', (4.0, 8.0)), ('Alpha', (8.0, 12.0)),
                ('BetaL', (12.0, 20.0)), ('BetaH', (20.0, 35.0)), ('Gamma', (35.0, 60.0))
            ]
        elif default_bands == 'schumann':
            bands = [
                ('SR1', (7.45, 8.15)), ('2x', (13.0, 15.0)), ('3x', (19.0, 21.0)), ('4x', (25.0, 28.0)),
                ('5x', (31.0, 35.0)), ('6x', (38.0, 42.0)), ('7x', (45.0, 48.0)), ('8x', (52.0, 54.0))
            ]
        elif default_bands == 'delta6':
            w = (4.0 - 0.5) / 6.0
            bands = [(f"{0.5 + k*w:.2f}-{0.5 + (k+1)*w:.2f}", (0.5 + k*w, 0.5 + (k+1)*w)) for k in range(6)]
        else:
            if not bands:
                raise ValueError("When default_bands='custom', pass non-empty `bands`.")

    # ---- 3) Sliding-window PSD → absolute band powers
    nwin = int(round(win_sec * fs))
    nstep = int(round(step_sec * fs))
    if nwin <= 1 or nstep < 1:
        raise ValueError("win_sec/step_sec too small relative to fs.")

    frames_bp, t_mids = [], []

    for s in range(0, len(x) - nwin + 1, nstep):
        seg = x[s:s + nwin]
        # Welch PSD using full window as segment for smooth band integration
        f, Pxx = signal.welch(seg, fs=fs, nperseg=nwin)
        bp = []
        for _, (blo, bhi) in bands:
            mask = (f >= blo) & (f <= bhi)
            bp.append(float(np.trapz(Pxx[mask], f[mask])) if np.any(mask) else 0.0)
        frames_bp.append(bp)
        t_mids.append(t0 + (s + nwin / 2) / fs)

    if not frames_bp:
        raise ValueError('No frames to animate; adjust t_range/win_sec/step_sec.')

    BP = np.asarray(frames_bp)  # [n_frames × n_bands]
    labels = [lbl for (lbl, _) in bands]

    # ---- 4) Build animation (cumulative stacked area over time)
    fig, ax = plt.subplots(figsize=(9.0, 3.6))
    ax.set_xlim(t_mids[0], t_mids[-1])
    total_power = BP.sum(axis=1)
    base_ylim = float(np.nanpercentile(total_power, 95))
    ax.set_ylim(0, max(base_ylim * ylim_pad, 1e-9))

    ttl = title or f"Stacked absolute power — {combine}"
    ax.set_title(ttl)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Absolute band power (μV²)')

    palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
               '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    colors = [palette[i % len(palette)] for i in range(len(labels))]

    stack_poly = [None]

    def _draw_frame(k):
        x_t = np.asarray(t_mids[:k + 1])
        Y = BP[:k + 1, :].T  # n_bands × (k+1)
        if dyn_ylim:
            ytop = float(np.nanmax(np.sum(Y, axis=0))) * ylim_pad
            if ytop > 0:
                ax.set_ylim(0, ytop)
        # remove previous stack safely
        if stack_poly[0] is not None:
            for coll in stack_poly[0]:
                try:
                    coll.remove()
                except Exception:
                    pass
        stack_poly[0] = ax.stackplot(x_t, *Y, labels=labels, colors=colors, alpha=0.95)
        ax.set_title(f"{ttl}\nWindow center: t = {t_mids[k]:.2f} s")
        return stack_poly[0]

    anim = animation.FuncAnimation(fig, _draw_frame, frames=BP.shape[0], interval=1000 / fps, blit=False)

    # Legend with static proxies
    from matplotlib.patches import Patch
    proxies = [Patch(facecolor=colors[i], label=labels[i]) for i in range(len(labels))]
    if legend_outside:
        fig.subplots_adjust(right=0.78)
        ax.legend(handles=proxies, loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=8, ncol=1)
    else:
        ax.legend(handles=proxies, frameon=False, fontsize=8, ncol=min(3, len(labels)))

    # ---- 5) Save movie (MP4 if available, else GIF fallback)
    ext = out_path.split('.')[-1].lower() if '.' in out_path else 'mp4'
    try:
        if ext in ('mp4', 'm4v', 'mov'):
            writer = animation.FFMpegWriter(fps=fps, bitrate=2000)
            anim.save(out_path, writer=writer, dpi=140)
        elif ext in ('gif',):
            from matplotlib.animation import PillowWriter
            anim.save(out_path, writer=PillowWriter(fps=fps))
        else:
            writer = animation.FFMpegWriter(fps=fps, bitrate=2000)
            anim.save(out_path, writer=writer, dpi=140)
    except Exception:
        from matplotlib.animation import PillowWriter
        gif_path = out_path.rsplit('.', 1)[0] + '.gif'
        anim.save(gif_path, writer=PillowWriter(fps=fps))
        out_path = gif_path

    plt.close(fig)

    # ---- 6) Save final stacked image (whole window) for one-look overview
    saved_last = None
    if save_last_frame:
        if last_frame_path is None:
            base, _ext = os.path.splitext(out_path)
            last_frame_path = base + '_last.png'
        fig2, ax2 = plt.subplots(figsize=(9.0, 3.6))
        ax2.set_xlim(t_mids[0], t_mids[-1])
        # full-window Y and ylim
        Y_full = BP.T  # n_bands × n_frames
        ytop = float(np.nanmax(np.sum(Y_full, axis=0))) * ylim_pad
        ax2.set_ylim(0, ytop if ytop > 0 else 1.0)
        ax2.stackplot(np.asarray(t_mids), *Y_full, labels=labels, colors=colors, alpha=0.95)
        # proxies for legend, title, labels
        ax2.set_title((title or 'Stacked absolute power') + ' — last frame (full window)')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Absolute band power (μV²)')
        if legend_outside:
            fig2.subplots_adjust(right=0.78)
            ax2.legend(handles=proxies, loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=8, ncol=1)
        else:
            ax2.legend(handles=proxies, frameon=False, fontsize=8, ncol=min(3, len(labels)))
        fig2.tight_layout()
        fig2.savefig(last_frame_path, dpi=140)
        plt.close(fig2)
        saved_last = last_frame_path

    if show_inline:
        return (anim, out_path, saved_last) if save_last_frame else (anim, out_path)
    return (out_path, saved_last) if save_last_frame else out_path


__all__ = [
    'animate_psd_stacked', 'BAND_PRESETS'
]


# -----------------------------------------------------------------------------
# Phase/Delay + WTC-ridge + Bicoherence helper
# -----------------------------------------------------------------------------

def _fit_group_delay(f: np.ndarray, phase: np.ndarray, fit_range=(1.0, 45.0)) -> Tuple[float, float, float]:
    """Fit a line to unwrapped phase(f) over `fit_range` to estimate group delay.
    Returns (tau_sec, slope, intercept) where phase ≈ slope * f + intercept.
    Group delay τ = - slope / (2π).
    """
    f = np.asarray(f, float)
    ph = np.unwrap(np.asarray(phase, float))
    mask = (f >= fit_range[0]) & (f <= fit_range[1])
    if not np.any(mask):
        raise ValueError("fit_range excludes all frequency samples")
    m, b = np.polyfit(f[mask], ph[mask], 1)
    tau = -m / (2.0 * np.pi)
    return float(tau), float(m), float(b)


def plot_phase_delay_wtc_bico(
    x: np.ndarray,
    y: np.ndarray,
    fs: float,
    bands: Optional[List[Tuple[str, Tuple[float, float]]]] = None,
    freq_fit: Tuple[float, float] = (1.0, 45.0),
    detrend: bool = True,
    stft_win_sec: float = 2.0,
    stft_step_sec: float = 0.25,
    max_coh_freq: float = 50.0,
    bico_fmax: float = 40.0,
    bico_bins: int = 36,
    title: Optional[str] = None,
):
    """
    Small diagnostic panel that:
      1) overlays cross-spectral phase φ(f) with a fitted group-delay line;
      2) prints per-band lag as % of cycle (using fitted τ);
      3) renders a compact WTC-like ridge (STFT coherence ridge) + bicoherence heatmap for the same window.

    Notes:
    - The WTC panel uses an STFT-based magnitude-squared coherence proxy C(f,t) = |X*conj(Y)|^2 / (|X|^2 |Y|^2).
      If `pycwt` is installed, you may swap in a true wavelet coherence implementation.
    - The bicoherence panel computes a normalized third-order coupling on `x` only to reveal quadratic phase coupling.

    Returns: (fig, axes_dict) with keys {'phase': ax0, 'coh': ax1, 'bico': ax2}.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if detrend:
        x = signal.detrend(x)
        y = signal.detrend(y)

    # -------------------- Cross-spectrum phase & group delay fit --------------------
    nper = int(round(fs * 4.0))
    nover = int(round(nper * 0.5))
    f, Pxy = signal.csd(x, y, fs=fs, nperseg=nper, noverlap=nover)
    phase = np.angle(Pxy)
    tau, m, b = _fit_group_delay(f, phase, fit_range=freq_fit)

    # Per-band lag as % cycle
    if bands is None:
        bands = BAND_PRESETS.get('canonical', [('Delta', (0.5, 4.0)), ('Theta', (4.0, 8.0)), ('Alpha', (8.0, 12.0))])

    def _band_center(blo, bhi):
        # geometric mean is less biased on log-frequency scales
        return float(np.sqrt(blo * bhi)) if blo > 0 else (blo + bhi) / 2.0

    band_rows = []
    for lbl, (blo, bhi) in bands:
        fc = _band_center(blo, bhi)
        cyc = tau * fc
        # wrap to [-0.5, 0.5) cycles for readability
        cyc_wrapped = ((cyc + 0.5) % 1.0) - 0.5
        pct = 100.0 * cyc_wrapped
        band_rows.append((lbl, fc, tau, pct))

    # -------------------- STFT coherence & ridge --------------------
    nper_stft = int(round(stft_win_sec * fs))
    nover_stft = nper_stft - int(round(stft_step_sec * fs))
    nover_stft = max(0, min(nover_stft, nper_stft - 1))

    f_stft, t_stft, Zx = signal.stft(x, fs=fs, nperseg=nper_stft, noverlap=nover_stft, boundary=None)
    _, _, Zy = signal.stft(y, fs=fs, nperseg=nper_stft, noverlap=nover_stft, boundary=None)

    eps = 1e-12
    C = (np.abs(Zx * np.conj(Zy)) ** 2) / (np.maximum(np.abs(Zx) ** 2 * np.abs(Zy) ** 2, eps))
    fmask = f_stft <= max_coh_freq
    C = C[fmask, :]
    f_coh = f_stft[fmask]

    ridge_idx = np.argmax(C, axis=0)
    ridge_freq = f_coh[ridge_idx]

    # -------------------- Bicoherence (on x) --------------------
    # Downselect frequencies up to bico_fmax and to ~bico_bins points
    fmask_b = f_stft <= bico_fmax
    Zx_b = Zx[fmask_b, :]
    f_b = f_stft[fmask_b]
    if len(f_b) == 0:
        raise ValueError("No STFT frequency bins under bico_fmax.")
    step = max(1, len(f_b) // bico_bins)
    Zb = Zx_b[::step, :]
    fb = f_b[::step]

    Nb, T = Zb.shape
    B = np.zeros((Nb, Nb), dtype=float)

    # Simple normalized bicoherence estimator
    for i in range(Nb):
        Zi = Zb[i, :]
        for j in range(Nb - i):  # ensure i+j within range
            k = i + j
            Zij = Zi * Zb[j, :]
            Zk = Zb[k, :]
            num = np.sum(Zij * np.conj(Zk))
            den = np.sqrt(np.sum(np.abs(Zij) ** 2) * np.sum(np.abs(Zk) ** 2)) + eps
            B[i, j] = np.abs(num) / den
        # optional: mask upper triangle beyond Nyquist sum region
        for j in range(Nb - i, Nb):
            B[i, j] = np.nan

    # -------------------- Plot layout --------------------
    fig = plt.figure(figsize=(12, 3.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.3, 1.2], wspace=0.32)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])

    if title:
        fig.suptitle(title, y=1.02, fontsize=11)

    # (0) Phase & fit
    ax0.plot(f, np.unwrap(phase), lw=1.2, label='phase(f)')
    ax0.plot(f, m * f + b, lw=1.2, linestyle='--', label=f'fit → τ={tau*1e3:.1f} ms')
    ax0.set_xlim(freq_fit[0], max(freq_fit[1], min(max(f), freq_fit[1])))
    ax0.set_xlabel('Frequency (Hz)')
    ax0.set_ylabel('Phase (rad)')
    ax0.legend(frameon=False, fontsize=8)

    # Band lag text box (% of cycle)
    lines = ["Band  fc(Hz)  τ(ms)  lag% (wrapped)"]
    for lbl, fc, tau_s, pct in band_rows:
        lines.append(f"{lbl:>6}  {fc:5.2f}  {tau_s*1e3:6.1f}  {pct:7.1f}")
    txt = "".join(lines)
    ax0.text(0.98, 0.05, txt, transform=ax0.transAxes, ha='right', va='bottom', fontsize=8,
             family='monospace', bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.8, lw=0.0))

    # (1) WTC-like coherence map + ridge
    pcm = ax1.pcolormesh(t_stft, f_coh, C, shading='auto')
    ax1.plot(t_stft, ridge_freq, lw=1.2, color='k', alpha=0.9, label='ridge')
    ax1.set_ylim(0, max_coh_freq)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Freq (Hz)')
    ax1.set_title('STFT coherence (proxy) + ridge')
    cb = fig.colorbar(pcm, ax=ax1, pad=0.02)
    cb.ax.set_ylabel('C(f,t)')

    # (2) Bicoherence heatmap
    im = ax2.imshow(B, origin='lower', aspect='auto', extent=[fb[0], fb[-1], fb[0], fb[-1]])
    ax2.set_xlabel('f₁ (Hz)')
    ax2.set_ylabel('f₂ (Hz)')
    ax2.set_title('Bicoherence |⟨X(f1)X(f2)X*(f1+f2)⟩|')
    cb2 = fig.colorbar(im, ax=ax2, pad=0.02)
    cb2.ax.set_ylabel('bicoherence')

    fig.tight_layout()
    return fig, {'phase': ax0, 'coh': ax1, 'bico': ax2}


def phase_wtc_bico_from_df(
    RECORDZ: pd.DataFrame,
    x_col: str,
    y_col: str,
    time_col: str = 'Timestamp',
    t_range: Optional[Tuple[float, float]] = None,
    bands: Optional[List[Tuple[str, Tuple[float, float]]]] = None,
    **kwargs,
):
    """
    Convenience wrapper: slice a time window from RECORDZ and call `plot_phase_delay_wtc_bico`.
    Requires helper utilities: ensure_timestamp_column, infer_fs, get_series.
    """
    time_col = ensure_timestamp_column(RECORDZ, time_col=time_col, default_fs=128.0)
    fs = infer_fs(RECORDZ, time_col)
    t = pd.to_numeric(RECORDZ[time_col], errors='coerce').values.astype(float)
    x_full = get_series(RECORDZ, x_col)
    y_full = get_series(RECORDZ, y_col)

    if t_range is None:
        i0, i1 = 0, len(t)
    else:
        t0, t1 = t_range
        i0 = max(0, int(round((t0 - t[0]) * fs)))
        i1 = min(len(t), int(round((t1 - t[0]) * fs)))
        if i1 - i0 <= 1:
            raise ValueError("t_range yields empty slice. Adjust t_range or check timestamp units.")

    x = np.asarray(x_full[i0:i1], float)
    y = np.asarray(y_full[i0:i1], float)
    return plot_phase_delay_wtc_bico(x, y, fs=fs, bands=bands, title=f"{x_col} vs {y_col}", **kwargs)


__all__ += [
    'plot_phase_delay_wtc_bico',
    'phase_wtc_bico_from_df',
]
