"""
Schumann Spike Detector — Morlet Wavelet (fs=128)
-------------------------------------------------
Wavelet-based variant for detecting transient Schumann spikes up to the 5th harmonic, with
sharper time–frequency ridges than IIR band-pass + Hilbert.

Approach
- Use complex Morlet (CWT) at center frequencies f_k = k * f0 (k=1..n_harmonics)
- Extract per-harmonic amplitude |W_k(t)|, smooth lightly, robust baseline removal → z-scores
- Threshold + min-duration → spike intervals per harmonic
- Coincidence grouping across harmonics
- Plots: harmonic heatmap (z), piano-roll intervals, Schumann Activity Index (sum of z>0)

Dependencies: numpy, pandas, scipy, matplotlib

Note: This implementation builds its own Morlet at each harmonic with bandwidth parameter 'w'
(omega0) and time-domain convolution using FFT (scipy.signal.fftconvolve). It avoids extra
packages; if PyWavelets is available, this can be swapped to pywt.cwt for convenience.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from scipy import signal

# ------------- utilities -------------

def _get_fs(RECORDS: pd.DataFrame, time_col: str) -> float:
    if 'infer_fs_from_records' in globals():
        try:
            return float(infer_fs_from_records(RECORDS, time_col=time_col))
        except Exception:
            pass
    t = np.asarray(RECORDS[time_col].values, dtype=float)
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        raise ValueError('Cannot infer fs from time column')
    return 1.0 / np.median(dt)


def _smooth(x: np.ndarray, fs: float, smooth_sec: float) -> np.ndarray:
    if smooth_sec <= 0:
        return x
    n = max(1, int(round(fs * smooth_sec)))
    if n <= 1:
        return x
    w = np.hanning(n)
    w = w / w.sum()
    return np.convolve(x, w, mode='same')


def _rolling_median_mad(x: np.ndarray, win: int) -> Tuple[np.ndarray, np.ndarray]:
    if win < 3:
        med = signal.medfilt(x, kernel_size=3)
        mad = np.median(np.abs(x - med)) + 1e-12
        return med, np.full_like(x, mad)
    pad = win // 2
    xp = np.pad(x, (pad, pad), mode='reflect')
    med = np.zeros_like(x)
    mad = np.zeros_like(x)
    for i in range(x.size):
        s = slice(i, i + win)
        m = np.median(xp[s])
        med[i] = m
        mad[i] = np.median(np.abs(xp[s] - m)) + 1e-12
    return med, mad


# ------------- Morlet wavelet core -------------

def _morlet_kernel(fs: float, f0: float, w: float = 6.0, dur_sec: float = 2.0) -> np.ndarray:
    """Build a complex Morlet wavelet centered at f0 (Hz).
    w (omega0) controls trade-off: larger w → better frequency, worse time; typical 5–7.
    dur_sec controls temporal support of kernel window.
    """
    N = int(round(fs * dur_sec))
    if N % 2 == 0:
        N += 1
    t = np.arange(-(N//2), N//2 + 1) / fs
    sigma_t = w / (2 * np.pi * f0)  # time std from central freq & w
    gauss = np.exp(-0.5 * (t / sigma_t)**2)
    carrier = np.exp(1j * 2 * np.pi * f0 * t)
    # zero-mean correction to reduce DC leakage
    psi = gauss * (carrier - np.exp(-(2*np.pi**2) * (sigma_t**2) * (f0**2)))
    # L2-normalize
    psi /= np.sqrt(np.sum(np.abs(psi)**2)) + 1e-12
    return psi


def _cwt_morlet(x: np.ndarray, fs: float, freqs: np.ndarray, w: float = 6.0, dur_sec: float = 2.0) -> np.ndarray:
    """Compute CWT magnitudes for requested freqs via FFT convolution per frequency.
    Returns array shape (len(freqs), len(x)) of |W_x(f,t)|.
    """
    mags = np.zeros((len(freqs), len(x)), dtype=float)
    for i, f0 in enumerate(freqs):
        psi = _morlet_kernel(fs, f0, w=w, dur_sec=dur_sec)
        conv = signal.fftconvolve(x, np.conj(psi[::-1]), mode='same')  # analytic response
        mags[i] = np.abs(conv)
    return mags


# ------------- detection -------------

def detect_schumann_spikes_wavelet(
    RECORDS: pd.DataFrame,
    signal_col: str,
    time_col: str = 'Timestamp',
    f0: float = 7.83,
    n_harmonics: int = 5,
    w: float = 6.0,
    kernel_dur_sec: float = 2.0,
    baseline_win_sec: float = 120.0,
    smooth_sec: float = 0.20,
    z_thresh: float = 3.5,
    min_dur_sec: float = 0.25,
) -> Dict[str, object]:
    fs = _get_fs(RECORDS, time_col)
    t = np.asarray(RECORDS[time_col].values, dtype=float)
    x = np.asarray(pd.to_numeric(RECORDS[signal_col], errors='coerce').fillna(0.0).values, dtype=float)
    x = signal.detrend(x, type='linear')

    harms = np.arange(1, n_harmonics + 1)
    freqs = f0 * harms

    mags = _cwt_morlet(x, fs, freqs, w=w, dur_sec=kernel_dur_sec)
    # smooth & baseline z-score per harmonic
    z_spec = np.zeros_like(mags)
    events: List[List[Dict[str, float]]] = []
    min_len = max(1, int(round(fs * min_dur_sec)))
    win = max(3, int(round(fs * baseline_win_sec)))

    for hi in range(harms.size):
        amp = _smooth(mags[hi], fs, smooth_sec)
        med, mad = _rolling_median_mad(amp, win)
        z = (amp - med) / (1.4826 * mad)
        z_spec[hi] = z
        # events
        mask = z >= z_thresh
        # find intervals
        diff = np.diff(mask.astype(int))
        starts = np.where(diff == 1)[0] + 1
        ends = np.where(diff == -1)[0]
        if mask[0]:
            starts = np.r_[0, starts]
        if mask[-1]:
            ends = np.r_[ends, mask.size - 1]
        evs = []
        for s, e in zip(starts, ends):
            if e - s + 1 < min_len:
                continue
            seg = z[s:e+1]
            pk = int(s + np.argmax(seg))
            evs.append({
                'harmonic': int(harms[hi]),
                'start_idx': int(s), 'end_idx': int(e), 'peak_idx': int(pk),
                'start_time': float(t[s]), 'end_time': float(t[e]), 'peak_time': float(t[pk]),
                'peak_z': float(z[pk])
            })
        events.append(evs)

    return {
        'index': t,
        'freqs': freqs,
        'z_spec': z_spec,
        'events': events,
        'params': {
            'fs': fs, 'f0': f0, 'n_harmonics': n_harmonics,
            'w': w, 'kernel_dur_sec': kernel_dur_sec,
            'baseline_win_sec': baseline_win_sec,
            'smooth_sec': smooth_sec,
            'z_thresh': z_thresh,
            'min_dur_sec': min_dur_sec,
            'signal_col': signal_col,
        }
    }


def group_coincident(events: List[List[Dict[str, float]]], tol_sec: float = 0.10) -> List[Dict[str, object]]:
    flat = []
    for row in events:
        flat.extend(row)
    if not flat:
        return []
    flat.sort(key=lambda d: d['start_time'])
    merged = []
    cur = { 'start_time': flat[0]['start_time'], 'end_time': flat[0]['end_time'],
            'harmonics': [flat[0]['harmonic']], 'peaks': {flat[0]['harmonic']: flat[0]['peak_z']} }
    for e in flat[1:]:
        if e['start_time'] <= cur['end_time'] + tol_sec:
            cur['end_time'] = max(cur['end_time'], e['end_time'])
            if e['harmonic'] not in cur['harmonics']:
                cur['harmonics'].append(e['harmonic'])
            cur['peaks'][e['harmonic']] = e['peak_z']
        else:
            cur['harmonics'].sort()
            merged.append(cur)
            cur = { 'start_time': e['start_time'], 'end_time': e['end_time'],
                    'harmonics': [e['harmonic']], 'peaks': {e['harmonic']: e['peak_z']} }
    cur['harmonics'].sort()
    merged.append(cur)
    return merged


def schumann_activity_index(z_spec: np.ndarray) -> np.ndarray:
    zp = np.clip(z_spec, 0.0, None)
    return np.nansum(zp, axis=0)


# ------------- plotting -------------

def plot_harmonic_heatmap(t: np.ndarray, z_spec: np.ndarray, f0: float, title: str = '') -> None:
    plt.figure(figsize=(10, 3.6))
    im = plt.imshow(z_spec, aspect='auto', origin='lower',
                    extent=[t[0], t[-1], 1, z_spec.shape[0]], cmap='magma')
    cb = plt.colorbar(im, pad=0.01)
    cb.set_label('z-score')
    yticks = np.arange(1, z_spec.shape[0] + 1)
    ylabels = [f'{k}×{f0:.2f} Hz' for k in yticks]
    plt.yticks(yticks, ylabels)
    plt.xlabel('Time (s)')
    plt.ylabel('Harmonic')
    plt.title(title or 'Schumann Harmonics — z-score heatmap (Morlet)')
    plt.tight_layout()


def plot_piano_roll(t: np.ndarray, events: List[List[Dict[str, float]]], f0: float, title: str = '') -> None:
    plt.figure(figsize=(10, 2.8))
    for hi, evs in enumerate(events, start=1):
        for e in evs:
            plt.plot([e['start_time'], e['end_time']], [hi, hi], lw=6, solid_capstyle='butt')
    yticks = np.arange(1, len(events) + 1)
    ylabels = [f'{k}×{f0:.2f} Hz' for k in yticks]
    plt.yticks(yticks, ylabels)
    plt.xlabel('Time (s)')
    plt.ylabel('Harmonic')
    plt.title(title or 'Detected Schumann spike intervals (Morlet piano roll)')
    plt.tight_layout()


def plot_sai(t: np.ndarray, sai: np.ndarray, title: str = '') -> None:
    plt.figure(figsize=(10, 2.4))
    plt.plot(t, sai, lw=1.2)
    plt.ylabel('Activity index (Σ z⁺)')
    plt.xlabel('Time (s)')
    plt.title(title or 'Schumann Activity Index (Morlet)')
    plt.tight_layout()


# ------------- orchestration -------------

def detect_and_plot_schumann_wavelet(
    RECORDS: pd.DataFrame,
    signal_col: str,
    time_col: str = 'Timestamp',
    f0: float = 7.83,
    n_harmonics: int = 5,
    w: float = 6.0,
    kernel_dur_sec: float = 2.0,
    baseline_win_sec: float = 120.0,
    smooth_sec: float = 0.20,
    z_thresh: float = 3.5,
    min_dur_sec: float = 0.25,
    show: bool = True,
) -> Dict[str, object]:
    out = detect_schumann_spikes_wavelet(
        RECORDS, signal_col, time_col=time_col, f0=f0, n_harmonics=n_harmonics,
        w=w, kernel_dur_sec=kernel_dur_sec,
        baseline_win_sec=baseline_win_sec, smooth_sec=smooth_sec,
        z_thresh=z_thresh, min_dur_sec=min_dur_sec)

    t, z_spec, events = out['index'], out['z_spec'], out['events']
    coinc = group_coincident(events, tol_sec=0.10)
    sai = schumann_activity_index(z_spec)

    out.update({'coincidence': coinc, 'sai': sai})

    if show:
        plot_harmonic_heatmap(t, z_spec, f0)
        plot_piano_roll(t, events, f0)
        plot_sai(t, sai)
    return out
