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
from typing import Dict, List, Tuple, Optional
from scipy import signal
import networkx as nx
from scipy.signal import butter, filtfilt, hilbert, welch, csd
from scipy.stats import circvar
from scipy.interpolate import griddata

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

def find_channel_series(records: pd.DataFrame, ch_name: str) -> Optional[pd.Series]:
    # ---------------- Basics: fs + channel access ----------------
    _DEF_TIME_COL = 'Timestamp'
    _DEF_CH_PATTERNS = ("EEG.{ch}", "eeg.{ch}", "{ch}", "RAW.{ch}", "CHAN.{ch}")
    for pat in _DEF_CH_PATTERNS:
        col = pat.format(ch=ch_name)
        if col in records.columns:
            return pd.to_numeric(records[col], errors='coerce').astype(float)
    return None
    
def _get_channel_vector(RECORDS, ch: str) -> np.ndarray:
    """Return 1D array for a channel; accepts 'F4' or 'EEG.F4' etc."""
    candidates = {ch, f'EEG.{ch}', ch.upper(), f'EEG.{ch.upper()}'}
    if isinstance(RECORDS, dict):
        data = RECORDS.get('data', RECORDS.get('eeg', RECORDS.get('EEG')))
        # pandas DataFrame?
        if hasattr(data, 'columns'):
            cols = list(map(str, data.columns))
            for nm in candidates:
                if nm in cols:
                    return np.asarray(data[nm]).astype(float)
        # numpy + channel list?
        ch_names = RECORDS.get('channel_names') or RECORDS.get('channels')
        if ch_names is not None and data is not None:
            ch_names = list(map(str, ch_names))
            for nm in candidates:
                if nm in ch_names:
                    i = ch_names.index(nm)
                    return np.asarray(data[:, i]).astype(float)
    # attributes fallback
    if hasattr(RECORDS, ch):
        return np.asarray(getattr(RECORDS, ch)).astype(float)
    if hasattr(RECORDS, 'data') and hasattr(RECORDS, 'channel_names'):
        ch_names = list(map(str, getattr(RECORDS, 'channel_names')))
        for nm in candidates:
            if nm in ch_names:
                i = ch_names.index(nm)
                return np.asarray(getattr(RECORDS, 'data')[:, i]).astype(float)
    # last try: strip EEG. prefix
    base = ch.replace('EEG.', '')
    if hasattr(RECORDS, base):
        return np.asarray(getattr(RECORDS, base)).astype(float)
    raise KeyError(f"Channel {ch} not found. Tried {sorted(candidates)}")

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

"""
Schumann Micro-grid + Heatmap Ridge Overlay (fs=128)
----------------------------------------------------
Adds fused TF heatmaps with ridge overlays to the drift tracker. For each harmonic k:
- computes CWT magnitude on a micro-grid around k*f0
- overlays the tracked ridge f_k(t) on the grid heatmap
- collects figures + returns event/coincidence tables

Entry point:
    fused = detect_and_plot_schumann_microgrid_with_heatmaps(...)
Outputs:
    fused['per_harmonic'][k-1] has: 'grid', 'mags', 'ridge_hz', plus figs displayed when show=True
"""


# ---------- reuse util from prior modules ----------

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
    w = np.hanning(n); w /= w.sum()
    return np.convolve(x, w, mode='same')


def _rolling_median_mad(x: np.ndarray, win: int):
    if win < 3:
        med = signal.medfilt(x, kernel_size=3)
        mad = np.median(np.abs(x - med)) + 1e-12
        return med, np.full_like(x, mad)
    pad = win // 2
    xp = np.pad(x, (pad, pad), mode='reflect')
    med = np.zeros_like(x); mad = np.zeros_like(x)
    for i in range(x.size):
        s = slice(i, i + win)
        m = np.median(xp[s])
        med[i] = m
        mad[i] = np.median(np.abs(xp[s] - m)) + 1e-12
    return med, mad


def _morlet_kernel(fs: float, f0: float, w: float = 6.0, dur_sec: float = 2.0) -> np.ndarray:
    N = int(round(fs * dur_sec)); N += (N % 2 == 0)
    t = np.arange(-(N//2), N//2 + 1) / fs
    sigma_t = w / (2 * np.pi * f0)
    gauss = np.exp(-0.5 * (t / sigma_t)**2)
    carrier = np.exp(1j * 2 * np.pi * f0 * t)
    psi = gauss * (carrier - np.exp(-(2*np.pi**2) * (sigma_t**2) * (f0**2)))
    psi /= np.sqrt(np.sum(np.abs(psi)**2)) + 1e-12
    return psi


def _cwt_grid_morlet(x: np.ndarray, fs: float, grid: np.ndarray, w: float = 6.0, dur_sec: float = 2.0) -> np.ndarray:
    mags = np.zeros((len(grid), len(x)), dtype=float)
    for i, f0 in enumerate(grid):
        psi = _morlet_kernel(fs, f0, w=w, dur_sec=dur_sec)
        conv = signal.fftconvolve(x, np.conj(psi[::-1]), mode='same')
        mags[i] = np.abs(conv)
    return mags


def _find_intervals(mask: np.ndarray) -> List[Tuple[int, int]]:
    if mask.size == 0:
        return []
    diff = np.diff(mask.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0]
    if mask[0]:
        starts = np.r_[0, starts]
    if mask[-1]:
        ends = np.r_[ends, mask.size - 1]
    return list(zip(starts, ends))


def group_coincident(events: List[List[Dict[str, float]]], tol_sec: float = 0.10) -> List[Dict[str, object]]:
    flat = []
    for evs in events:
        flat.extend(evs)
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
            cur['harmonics'].sort(); merged.append(cur)
            cur = { 'start_time': e['start_time'], 'end_time': e['end_time'],
                    'harmonics': [e['harmonic']], 'peaks': {e['harmonic']: e['peak_z']} }
    cur['harmonics'].sort(); merged.append(cur)
    return merged


def schumann_activity_index(z: np.ndarray) -> np.ndarray:
    zp = np.clip(z, 0.0, None)
    return np.nansum(zp, axis=0)

# ---------- fused drift + heatmap ----------

def detect_and_plot_schumann_microgrid_with_heatmaps(
    RECORDS: pd.DataFrame,
    signal_col: str,
    time_col: str = 'Timestamp',
    f0: float = 7.83,
    n_harmonics: int = 5,
    delta_hz: float = 0.45,
    step_hz: float = 0.05,
    w: float = 6.0,
    kernel_dur_sec: float = 2.0,
    baseline_win_sec: float = 120.0,
    smooth_sec: float = 0.20,
    z_thresh: float = 3.5,
    min_dur_sec: float = 0.25,
    ridge_medfilt_sec: float = 0.5,
    show: bool = True,
) -> Dict[str, object]:
    fs = _get_fs(RECORDS, time_col)
    t = np.asarray(RECORDS[time_col].values, dtype=float)
    x = np.asarray(pd.to_numeric(RECORDS[signal_col], errors='coerce').fillna(0.0).values, dtype=float)
    x = signal.detrend(x, type='linear')

    harms = np.arange(1, n_harmonics+1)
    events: List[List[Dict[str, float]]] = []
    ridge_hz = []; ridge_amp = []; z_ridge = []
    per_harmonic = []

    min_len = max(1, int(round(fs * min_dur_sec)))
    base_win = max(3, int(round(fs * baseline_win_sec)))
    rid_med_n = max(1, int(round(fs * ridge_medfilt_sec))) | 1

    for k in harms:
        nominal = f0 * k
        grid = np.arange(nominal - delta_hz, nominal + delta_hz + 1e-9, step_hz)
        mags = _cwt_grid_morlet(x, fs, grid, w=w, dur_sec=kernel_dur_sec)  # [n_f, n_t]
        idx = np.argmax(mags, axis=0)
        idx_s = signal.medfilt(idx, kernel_size=rid_med_n)
        fh = grid[idx_s]
        ah = mags[idx_s, np.arange(mags.shape[1])]
        ah_s = _smooth(ah, fs, smooth_sec)
        med, mad = _rolling_median_mad(ah_s, base_win)
        zh = (ah_s - med) / (1.4826 * mad)
        # spike events along ridge
        mask = zh >= z_thresh
        ints = _find_intervals(mask)
        evs = []
        for s, e in ints:
            if e - s + 1 < min_len:
                continue
            seg = zh[s:e+1]; pk = int(s + np.argmax(seg))
            evs.append({
                'harmonic': int(k),
                'start_idx': int(s), 'end_idx': int(e), 'peak_idx': int(pk),
                'start_time': float(t[s]), 'end_time': float(t[e]), 'peak_time': float(t[pk]),
                'peak_z': float(zh[pk]), 'peak_freq_hz': float(fh[pk])
            })
        events.append(evs)
        ridge_hz.append(fh); ridge_amp.append(ah_s); z_ridge.append(zh)
        per_harmonic.append({'grid': grid, 'mags': mags, 'ridge_hz': fh})

        # ---- plot heatmap with ridge overlay ----
        if show:
            plt.figure(figsize=(10, 3.6))
            im = plt.imshow(mags, aspect='auto', origin='lower',
                            extent=[t[0], t[-1], grid[0], grid[-1]], cmap='viridis')
            plt.plot(t, fh, color='w', lw=1.2)
            cb = plt.colorbar(im, pad=0.01); cb.set_label('|CWT|')
            plt.xlabel('Time (s)'); plt.ylabel('Frequency (Hz)')
            plt.title(f'Harmonic {k}: micro-grid TF heatmap + ridge')
            plt.tight_layout()

    ridge_hz = np.vstack(ridge_hz)
    ridge_amp = np.vstack(ridge_amp)
    z_ridge = np.vstack(z_ridge)

    coinc = group_coincident(events, tol_sec=0.10)
    sai = schumann_activity_index(z_ridge)

    # summary plots
    if show:
        # drift panels
        for i, k in enumerate(harms):
            plt.figure(figsize=(10, 2.6))
            plt.plot(t, ridge_hz[i] - (f0*k), lw=1.2)
            plt.axhline(0.0, color='k', lw=0.8)
            plt.ylabel('Drift (Hz)'); plt.xlabel('Time (s)')
            plt.title(f'Harmonic {k} drift')
            plt.tight_layout()
        # piano roll + SAI
        plt.figure(figsize=(10, 2.8))
        for hi, evs in enumerate(events, start=1):
            for e in evs:
                plt.plot([e['start_time'], e['end_time']], [hi, hi], lw=6, solid_capstyle='butt')
        yticks = np.arange(1, len(events) + 1)
        ylabels = [f'{k}×{f0:.2f} Hz' for k in yticks]
        plt.yticks(yticks, ylabels)
        plt.xlabel('Time (s)'); plt.ylabel('Harmonic')
        plt.title('Detected spikes along drift-tracked ridges'); plt.tight_layout()

        plt.figure(figsize=(10, 2.4))
        plt.plot(t, sai, lw=1.2)
        plt.ylabel('Activity index (Σ z⁺)'); plt.xlabel('Time (s)')
        plt.title('Schumann Activity Index (ridge-based)'); plt.tight_layout()

    return {
        'index': t,
        'per_harmonic': per_harmonic,
        'ridge_hz': ridge_hz,
        'ridge_amp': ridge_amp,
        'z_ridge': z_ridge,
        'events': events,
        'coincidence': coinc,
        'sai': sai,
        'params': {
            'fs': fs, 'f0': f0, 'n_harmonics': n_harmonics,
            'delta_hz': delta_hz, 'step_hz': step_hz,
            'w': w, 'kernel_dur_sec': kernel_dur_sec,
            'baseline_win_sec': baseline_win_sec, 'smooth_sec': smooth_sec,
            'z_thresh': z_thresh, 'min_dur_sec': min_dur_sec,
            'ridge_medfilt_sec': ridge_medfilt_sec,
            'signal_col': signal_col,
        }
    }


"""
Schumann Global TF Pane + Ridge Overlay (fs=128)
------------------------------------------------
Adds a global time–frequency (TF) panel that stacks all five harmonic micro-grids into a
single figure and overlays each harmonic’s drift-tracked ridge. Designed to slot into the
micro-grid drift tracker pipeline.

Entry point:
    fused = detect_and_plot_schumann_microgrid_with_global_tf(...)

Outputs:
    - Keeps per-harmonic heatmaps with ridge overlays (like before)
    - Adds one global TF figure composed by vertically concatenating each harmonic grid
      (freq axis is absolute Hz), with ridge traces drawn in unique colors.
"""

# ---------- utilities (same as prior modules) ----------

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
    w = np.hanning(n); w /= w.sum()
    return np.convolve(x, w, mode='same')


def _rolling_median_mad(x: np.ndarray, win: int):
    if win < 3:
        med = signal.medfilt(x, kernel_size=3)
        mad = np.median(np.abs(x - med)) + 1e-12
        return med, np.full_like(x, mad)
    pad = win // 2
    xp = np.pad(x, (pad, pad), mode='reflect')
    med = np.zeros_like(x); mad = np.zeros_like(x)
    for i in range(x.size):
        s = slice(i, i + win)
        m = np.median(xp[s])
        med[i] = m
        mad[i] = np.median(np.abs(xp[s] - m)) + 1e-12
    return med, mad


def _morlet_kernel(fs: float, f0: float, w: float = 6.0, dur_sec: float = 2.0) -> np.ndarray:
    N = int(round(fs * dur_sec)); N += (N % 2 == 0)
    t = np.arange(-(N//2), N//2 + 1) / fs
    sigma_t = w / (2 * np.pi * f0)
    gauss = np.exp(-0.5 * (t / sigma_t)**2)
    carrier = np.exp(1j * 2 * np.pi * f0 * t)
    psi = gauss * (carrier - np.exp(-(2*np.pi**2) * (sigma_t**2) * (f0**2)))
    psi /= np.sqrt(np.sum(np.abs(psi)**2)) + 1e-12
    return psi


def _cwt_grid_morlet(x: np.ndarray, fs: float, grid: np.ndarray, w: float = 6.0, dur_sec: float = 2.0) -> np.ndarray:
    mags = np.zeros((len(grid), len(x)), dtype=float)
    for i, f0 in enumerate(grid):
        psi = _morlet_kernel(fs, f0, w=w, dur_sec=dur_sec)
        conv = signal.fftconvolve(x, np.conj(psi[::-1]), mode='same')
        mags[i] = np.abs(conv)
    return mags


def _find_intervals(mask: np.ndarray) -> List[Tuple[int, int]]:
    if mask.size == 0:
        return []
    diff = np.diff(mask.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0]
    if mask[0]:
        starts = np.r_[0, starts]
    if mask[-1]:
        ends = np.r_[ends, mask.size - 1]
    return list(zip(starts, ends))


def group_coincident(events: List[List[Dict[str, float]]], tol_sec: float = 0.10) -> List[Dict[str, object]]:
    flat = []
    for evs in events:
        flat.extend(evs)
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
            cur['harmonics'].sort(); merged.append(cur)
            cur = { 'start_time': e['start_time'], 'end_time': e['end_time'],
                    'harmonics': [e['harmonic']], 'peaks': {e['harmonic']: e['peak_z']} }
    cur['harmonics'].sort(); merged.append(cur)
    return merged


def schumann_activity_index(z: np.ndarray) -> np.ndarray:
    zp = np.clip(z, 0.0, None)
    return np.nansum(zp, axis=0)

# ---------- global TF with ridge overlay ----------

def detect_and_plot_schumann_microgrid_with_global_tf(
    RECORDS: pd.DataFrame,
    signal_col: str,
    time_col: str = 'Timestamp',
    f0: float = 7.83,
    n_harmonics: int = 5,
    delta_hz: float = 0.45,
    step_hz: float = 0.05,
    w: float = 6.0,
    kernel_dur_sec: float = 2.0,
    baseline_win_sec: float = 120.0,
    smooth_sec: float = 0.20,
    z_thresh: float = 3.5,
    min_dur_sec: float = 0.25,
    ridge_medfilt_sec: float = 0.5,
    show: bool = True,
) -> Dict[str, object]:
    fs = _get_fs(RECORDS, time_col)
    t = np.asarray(RECORDS[time_col].values, dtype=float)
    x = np.asarray(pd.to_numeric(RECORDS[signal_col], errors='coerce').fillna(0.0).values, dtype=float)
    x = signal.detrend(x, type='linear')

    harms = np.arange(1, n_harmonics+1)
    events: List[List[Dict[str, float]]] = []
    ridge_hz = []; ridge_amp = []; z_ridge = []

    min_len = max(1, int(round(fs * min_dur_sec)))
    base_win = max(3, int(round(fs * baseline_win_sec)))
    rid_med_n = max(1, int(round(fs * ridge_medfilt_sec))) | 1

    # stash for global panel
    grids = []
    mags_list = []
    ridges = []

    for k in harms:
        nominal = f0 * k
        grid = np.arange(nominal - delta_hz, nominal + delta_hz + 1e-9, step_hz)
        mags = _cwt_grid_morlet(x, fs, grid, w=w, dur_sec=kernel_dur_sec)
        idx = np.argmax(mags, axis=0)
        idx_s = signal.medfilt(idx, kernel_size=rid_med_n)
        fh = grid[idx_s]
        ah = mags[idx_s, np.arange(mags.shape[1])]
        ah_s = _smooth(ah, fs, smooth_sec)
        med, mad = _rolling_median_mad(ah_s, base_win)
        zh = (ah_s - med) / (1.4826 * mad)
        # events along ridge
        mask = zh >= z_thresh
        ints = _find_intervals(mask)
        evs = []
        for s, e in ints:
            if e - s + 1 < min_len:
                continue
            seg = zh[s:e+1]; pk = int(s + np.argmax(seg))
            evs.append({
                'harmonic': int(k),
                'start_idx': int(s), 'end_idx': int(e), 'peak_idx': int(pk),
                'start_time': float(t[s]), 'end_time': float(t[e]), 'peak_time': float(t[pk]),
                'peak_z': float(zh[pk]), 'peak_freq_hz': float(fh[pk])
            })
        events.append(evs)
        ridge_hz.append(fh); ridge_amp.append(ah_s); z_ridge.append(zh)
        grids.append(grid); mags_list.append(mags); ridges.append(fh)

    ridge_hz = np.vstack(ridge_hz)
    ridge_amp = np.vstack(ridge_amp)
    z_ridge = np.vstack(z_ridge)

    coinc = group_coincident(events, tol_sec=0.10)
    sai = schumann_activity_index(z_ridge)

    # ----- global TF figure: stack grids vertically -----
    if show:
        # Build a unified frequency axis from min(grid_1) to max(grid_last)
        fmin = min(g[0] for g in grids)
        fmax = max(g[-1] for g in grids)
        # create an empty canvas with enough rows to place each band’s grid block
        # We’ll space the blocks at their true frequency ranges with small gaps
        fig, ax = plt.subplots(1, 1, figsize=(10, 6), constrained_layout=True)
        # draw each harmonic block as an image in its own frequency slice
        colors = ['w', 'y', 'c', 'm', 'r']  # ridge colors per harmonic
        for i, (grid, mags, fh) in enumerate(zip(grids, mags_list, ridges)):
            # extent uses absolute Hz so blocks are placed realistically
            ext = [t[0], t[-1], grid[0], grid[-1]]
            ax.imshow(mags, aspect='auto', origin='lower', extent=ext, cmap='viridis')
            ax.plot(t, fh, color=colors[i % len(colors)], lw=1.2)
        cb = plt.colorbar(ax.images[0], ax=ax, pad=0.01)
        cb.set_label('|CWT|')
        ax.set_xlabel('Time (s)'); ax.set_ylabel('Frequency (Hz)')
        ax.set_title('Global TF pane (all harmonics) + ridge overlays')
        plt.show()

        # summary plots as before: piano roll + SAI
        plt.figure(figsize=(10, 2.8))
        for hi, evs in enumerate(events, start=1):
            for e in evs:
                plt.plot([e['start_time'], e['end_time']], [hi, hi], lw=6, solid_capstyle='butt')
        yticks = np.arange(1, len(events) + 1)
        ylabels = [f'{k}×{f0:.2f} Hz' for k in range(1, n_harmonics+1)]
        plt.yticks(yticks, ylabels)
        plt.xlabel('Time (s)'); plt.ylabel('Harmonic')
        plt.title('Detected spikes along drift-tracked ridges'); plt.tight_layout()

        plt.figure(figsize=(10, 2.4))
        plt.plot(t, sai, lw=1.2)
        plt.ylabel('Activity index (Σ z⁺)'); plt.xlabel('Time (s)')
        plt.title('Schumann Activity Index (ridge-based)'); plt.tight_layout()

    return {
        'index': t,
        'ridge_hz': ridge_hz,
        'ridge_amp': ridge_amp,
        'z_ridge': z_ridge,
        'events': events,
        'coincidence': coinc,
        'sai': sai,
        'grids': grids,
        'mags_list': mags_list,
        'params': {
            'fs': fs, 'f0': f0, 'n_harmonics': n_harmonics,
            'delta_hz': delta_hz, 'step_hz': step_hz,
            'w': w, 'kernel_dur_sec': kernel_dur_sec,
            'baseline_win_sec': baseline_win_sec, 'smooth_sec': smooth_sec,
            'z_thresh': z_thresh, 'min_dur_sec': min_dur_sec,
            'ridge_medfilt_sec': ridge_medfilt_sec,
            'signal_col': signal_col,
        }
    }

"""
Schumann Overlap Score (fs=128)
-------------------------------
Computes and visualizes a harmonic **overlap score**: at each time t, how many harmonics
are simultaneously active (z >= threshold) along their drift-tracked ridges.

This slots into the micro-grid drift tracker pipeline. Use either a fused result from:
- detect_and_plot_schumann_microgrid_with_global_tf(...)
- detect_and_plot_schumann_microgrid_with_heatmaps(...)
- detect_and_plot_schumann_microgrid(...)

API:
    out = compute_and_plot_overlap_from_fused(
        fused_dict,
        z_thresh=None,       # falls back to fused_dict['params']['z_thresh']
        min_len_sec=0.25,    # min overlap interval duration
        show=True
    )

Returns `out` with keys:
    'overlap_series' : np.ndarray shape [n_times], integer counts [0..n_harm]
    'intervals'      : dict mapping K -> list of (start_idx, end_idx, start_time, end_time)
    'params'         : metadata

Also provides:
    plot_overlap_series(t, overlap)
    plot_overlap_intervals(t, intervals) — piano-roll of K>=2..n_harm
    plot_overlap_hist(overlap)
"""


# ---- small utilities re-used ----

def _find_intervals(mask: np.ndarray) -> List[Tuple[int, int]]:
    if mask.size == 0:
        return []
    diff = np.diff(mask.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0]
    if mask[0]:
        starts = np.r_[0, starts]
    if mask[-1]:
        ends = np.r_[ends, mask.size - 1]
    return list(zip(starts, ends))

# ---- core overlap computation ----

def compute_overlap_series(z_ridge: np.ndarray, z_thresh: float) -> np.ndarray:
    """Count of harmonics with z >= z_thresh at each time point.
    z_ridge: [n_harm, n_times]
    """
    active = (z_ridge >= float(z_thresh)).astype(int)
    overlap = np.sum(active, axis=0)
    return overlap


def summarize_overlap_intervals(
    t: np.ndarray,
    overlap: np.ndarray,
    n_harm: int,
    min_len_sec: float,
    fs: float
) -> Dict[int, List[Tuple[int, int, float, float]]]:
    """For K=2..n_harm, create intervals where overlap>=K, enforcing min length.
    Returns dict K -> list of (start_idx, end_idx, start_time, end_time).
    """
    intervals: Dict[int, List[Tuple[int, int, float, float]]] = {}
    min_len = max(1, int(round(fs * min_len_sec)))
    for K in range(2, n_harm + 1):
        mask = overlap >= K
        ints = []
        for s, e in _find_intervals(mask):
            if e - s + 1 < min_len:
                continue
            ints.append((s, e, float(t[s]), float(t[e])))
        intervals[K] = ints
    return intervals

# ---- plots ----

def plot_overlap_series(t: np.ndarray, overlap: np.ndarray, title: str = '') -> None:
    plt.figure(figsize=(10, 2.4))
    plt.step(t, overlap, where='mid', lw=1.2)
    plt.ylim(-0.5, np.max(overlap) + 0.5)
    plt.ylabel('# Harmonics ≥ z')
    plt.xlabel('Time (s)')
    plt.title(title or 'Harmonic overlap (count active)')
    plt.tight_layout()


def plot_overlap_hist(overlap: np.ndarray, title: str = '') -> None:
    plt.figure(figsize=(4.5, 3.2))
    bins = np.arange(-0.5, np.max(overlap) + 1.5, 1)
    plt.hist(overlap, bins=bins, rwidth=0.9)
    plt.xlabel('Simultaneously active harmonics')
    plt.ylabel('Samples')
    plt.title(title or 'Overlap histogram')
    plt.tight_layout()


def plot_overlap_intervals(t: np.ndarray, intervals: Dict[int, List[Tuple[int, int, float, float]]], title: str = '') -> None:
    plt.figure(figsize=(10, 3.2))
    # K goes from 2 to max key
    Ks = sorted(intervals.keys())
    for i, K in enumerate(Ks, start=1):
        for s, e, ts, te in intervals[K]:
            plt.plot([ts, te], [K, K], lw=6, solid_capstyle='butt')
    plt.yticks(Ks, [f'K≥{K}' for K in Ks])
    plt.xlabel('Time (s)'); plt.ylabel('Overlap tiers')
    plt.title(title or 'Overlap intervals (K≥2..N)')
    plt.tight_layout()

# ---- wrapper for fused pipeline dict ----

def compute_and_plot_overlap_from_fused(
    fused: Dict[str, object],
    z_thresh: float | None = None,
    min_len_sec: float = 0.25,
    show: bool = True
) -> Dict[str, object]:
    """Compute and (optionally) plot overlap score from a fused micro-grid result.
    Expects fused to contain: 'index', 'z_ridge', and 'params' with 'fs' and 'n_harmonics'.
    """
    if 'index' not in fused or 'z_ridge' not in fused:
        raise ValueError('Fused dict must contain index and z_ridge')
    t = fused['index']
    z_ridge = fused['z_ridge']
    params = fused.get('params', {})
    fs = float(params.get('fs', 128.0))
    n_harm = int(params.get('n_harmonics', z_ridge.shape[0]))
    if z_thresh is None:
        z_thresh = float(params.get('z_thresh', 3.5))

    overlap = compute_overlap_series(z_ridge, z_thresh)
    intervals = summarize_overlap_intervals(t, overlap, n_harm, min_len_sec, fs)

    if show:
        plot_overlap_series(t, overlap)
        plot_overlap_intervals(t, intervals)
        plot_overlap_hist(overlap)

    return {
        'overlap_series': overlap,
        'intervals': intervals,
        'params': {
            'fs': fs,
            'n_harmonics': n_harm,
            'z_thresh': z_thresh,
            'min_len_sec': min_len_sec
        }
    }

"""
Schumann Overlap → Field Coherence ETAs (fs=128)
-------------------------------------------------
Hypothesis: Multi-harmonic co-activation (K≥3) marks episodes of global field order.

This module takes a fused micro-grid result (with ridge z and index), computes the
**overlap series** (count of harmonics active z≥thr), detects **K≥3 onsets**, and
computes **event‑triggered averages (ETAs)** for:
  • PLV in a chosen EEG band
  • PAC (one or more phase→amp pairs)
  • Graph min-cut (wPLI in a chosen band)
  • 1/f slope β (Welch fit)

It also supports **pseudo‑onset bootstraps** to generate a null band.

Usage
-----
# 1) Run micro-grid (no plots needed)
fused = detect_and_plot_schumann_microgrid_with_global_tf(
    RECORDS, signal_col='EEG.O1', time_col='Timestamp', show=False)

# 2) Run ETAs
etas = run_overlap_coherence_etas(
    RECORDS,
    fused=fused,
    electrodes=['F4','O1','O2'],      # or omit to autodetect EEG.*
    time_col='Timestamp',
    K=3,
    win_sec=2.0, step_sec=0.25,       # sliding window + step for ETA
    span_sec=5.0,                     # ETA span: ±5 s around onsets
    plv_band=(8,13),                  # EEG band for PLV
    pac_pairs={'theta→gamma':((4,8),(30,80))},
    mincut_band=(8,13),               # wPLI/min-cut band
    beta_band=(1,40),                 # 1/f fit band
    n_boot=200,                       # pseudo-onset bootstraps
    show=True
)

# 3) Inspect summary
print(etas['n_onsets'], 'K≥3 onsets')
print(etas['eta_time'])                # ETA time axis (s)
print(etas['eta_plv'].shape)          # (T,)
print(etas['eta_beta'].shape)         # (T,)
print(etas['eta_mincut'].shape)       # (T,)
print({k: v.shape for k,v in etas['eta_pac'].items()})
"""

# ---------- utilities ----------

def _get_fs(RECORDS: pd.DataFrame, time_col: str) -> float:
    if 'infer_fs_from_records' in globals():
        try:
            return float(infer_fs_from_records(RECORDS, time_col=time_col))
        except Exception:
            pass
    t = np.asarray(RECORDS[time_col].values, dtype=float)
    dt = np.diff(t); dt = dt[np.isfinite(dt) & (dt>0)]
    if dt.size == 0: raise ValueError('Cannot infer fs')
    return 1.0/np.median(dt)


def _autoelectrodes(RECORDS: pd.DataFrame, time_col: str) -> List[str]:
    els = []
    for col in RECORDS.columns:
        if col == time_col: continue
        if col.startswith('EEG.'):
            ch = col.split('.',1)[1]
            if ch and ch not in els: els.append(ch)
    return els or ['F4','O1','O2']


def _bandpass(X: np.ndarray, fs: float, f1: float, f2: float, order: int=4) -> np.ndarray:
    ny = 0.5*fs; f1 = max(1e-6, min(f1, ny*0.99)); f2 = max(f1+1e-6, min(f2, ny*0.999))
    b,a = signal.butter(order, [f1/ny, f2/ny], btype='band')
    return signal.filtfilt(b,a,X,axis=1)


# ---------- metrics in a window ----------

def _plv_mean_block(Xb: np.ndarray) -> float:
    Z = signal.hilbert(Xb, axis=1); ang = np.angle(Z)
    n = Xb.shape[0]; vals=[]
    for i in range(n):
        for j in range(i+1,n):
            d = ang[i]-ang[j]
            vals.append(np.abs(np.mean(np.exp(1j*d))))
    return float(np.nanmean(vals)) if vals else np.nan


def _mincut_wpli_block(Xb: np.ndarray) -> float:
    # wPLI adjacency (fallback to simple analytic)
    Z = signal.hilbert(Xb, axis=1); n = Xb.shape[0]
    A = np.zeros((n,n), float)
    for i in range(n):
        zi = Z[i]
        for j in range(i+1,n):
            im = np.imag(zi*np.conj(Z[j]))
            num = np.abs(np.mean(im)); den = np.mean(np.abs(im))+1e-12
            A[i,j]=A[j,i]=num/den
    np.fill_diagonal(A, 0.0)
    # global min cut
    G = nx.from_numpy_array(A)
    try:
        cut_val, _ = nx.algorithms.connectivity.stoer_wagner(G)
        return float(cut_val)
    except Exception:
        return float('nan')


def _pac_mi_block(x_phase: np.ndarray, x_amp: np.ndarray, nbins: int=18) -> float:
    # Tort MI per channel averaged
    phase = np.angle(signal.hilbert(x_phase, axis=1))
    amp   = np.abs(signal.hilbert(x_amp,   axis=1))
    vals=[]
    for ch in range(phase.shape[0]):
        ph = phase[ch]; am = amp[ch]
        edges = np.linspace(-np.pi, np.pi, nbins+1)
        digit = np.digitize(ph, edges)-1; digit = np.clip(digit,0,nbins-1)
        m = np.zeros(nbins)
        for k in range(nbins):
            sel = (digit==k); m[k]=np.mean(am[sel]) if np.any(sel) else 0.0
        if m.sum()<=0: continue
        p = m/(m.sum()); eps=1e-12
        kl = np.sum(p*np.log((p+eps)/(1.0/nbins)))
        vals.append(kl/np.log(nbins))
    return float(np.nanmean(vals)) if vals else 0.0


def _beta_welch_block(x: np.ndarray, fs: float, fmin: float, fmax: float) -> float:
    f,p = signal.welch(x, fs=fs, nperseg=4*int(fs))
    sel = (f>=fmin)&(f<=fmax)&np.isfinite(p)
    if np.count_nonzero(sel)<6: return np.nan
    X = np.vstack([np.ones(np.sum(sel)), -np.log(f[sel])]).T
    y = np.log(p[sel]+1e-24)
    b,*_ = np.linalg.lstsq(X,y,rcond=None)
    return float(b[1])

# ---------- event-triggered averaging ----------

def _eta_time_series(onsets: np.ndarray, tvec: np.ndarray, y: np.ndarray, span_sec: float) -> Tuple[np.ndarray,np.ndarray]:
    # returns (tau, mean_eta), tau in seconds
    fs_eff = 1.0/np.median(np.diff(tvec))
    half = int(round(fs_eff*span_sec))
    ets = []
    for i0 in onsets:
        s = max(0, i0-half); e = min(len(tvec), i0+half)
        seg = y[s:e]
        if seg.size < 2*half:
            pad = np.full(2*half, np.nan); pad[:seg.size]=seg; seg=pad
        ets.append(seg)
    if not ets:
        return np.linspace(-span_sec, span_sec, 2*half), np.full(2*half, np.nan)
    M = np.nanmean(np.vstack(ets), axis=0)
    tau = np.linspace(-span_sec, span_sec, 2*half)
    return tau, M

# ---------- main orchestration ----------

def run_overlap_coherence_etas(
    RECORDS: pd.DataFrame,
    fused: Dict[str, object],
    electrodes: Optional[List[str]] = None,
    time_col: str = 'Timestamp',
    K: int = 3,
    win_sec: float = 2.0,
    step_sec: float = 0.25,
    span_sec: float = 5.0,
    plv_band: Tuple[float,float] = (8,13),
    pac_pairs: Optional[Dict[str, Tuple[Tuple[float,float], Tuple[float,float]]]] = None,
    mincut_band: Tuple[float,float] = (8,13),
    beta_band: Tuple[float,float] = (1,40),
    n_boot: int = 200,
    show: bool = True,
) -> Dict[str, object]:
    fs = _get_fs(RECORDS, time_col)
    if electrodes is None:
        electrodes = _autoelectrodes(RECORDS, time_col)
    pac_pairs = pac_pairs or {'theta→gamma':((4,8),(30,80))}

    # Build sensor matrix X
    series=[]
    for ch in electrodes:
        s = find_channel_series(RECORDS, ch)
        if s is None: continue
        series.append(np.asarray(s.values, dtype=float))
    X = np.vstack(series)

    # Overlap series on fused timebase
    z_ridge = np.asarray(fused['z_ridge'])
    sch_t   = np.asarray(fused['index'])
    z_thr   = float(fused.get('params',{}).get('z_thresh', 3.5))
    overlap = np.sum((z_ridge >= z_thr).astype(int), axis=0)

    # PAC/PLV/mincut/beta time-series sampled on a uniform timebase
    n = X.shape[1]; win = max(16, int(round(fs*win_sec))); step = max(1, int(round(fs*step_sec)))
    idxs=[]; tvec=[]; plv_ts=[]; beta_ts=[]; mincut_ts=[]
    pac_ts = {k: [] for k in pac_pairs.keys()}

    for s0 in range(0, n-win+1, step):
        e0 = s0+win; tmid = (s0+e0)/(2*fs)
        blk = X[:, s0:e0]
        # PLV
        xb = _bandpass(blk, fs, plv_band[0], plv_band[1])
        plv_ts.append(_plv_mean_block(xb))
        # mincut via wPLI in mincut_band
        xm = _bandpass(blk, fs, mincut_band[0], mincut_band[1])
        mincut_ts.append(_mincut_wpli_block(xm))
        # beta on broadband
        beta_ts.append(_beta_welch_block(blk.mean(axis=0), fs, beta_band[0], beta_band[1]))
        # PAC pairs
        for name,(pb,ab) in pac_pairs.items():
            xp = _bandpass(blk, fs, pb[0], pb[1])
            xa = _bandpass(blk, fs, ab[0], ab[1])
            pac_ts[name].append(_pac_mi_block(xp, xa))
        tvec.append(tmid); idxs.append((s0,e0))

    tvec = np.asarray(tvec)
    # Interpolate overlap to this metric timebase
    ov_p = np.interp(tvec, sch_t, overlap)

    # Detect K-onsets on metric timebase
    mask = ov_p >= K
    d = np.diff(mask.astype(int)); onsets = np.where(d==1)[0]+1

    # Build ETAs
    def eta(y): return _eta_time_series(onsets, tvec, np.asarray(y, float), span_sec)
    tau, eta_plv = eta(plv_ts)
    _,   eta_beta= eta(beta_ts)
    _,   eta_mincut= eta(mincut_ts)
    eta_pac = {}
    for name,y in pac_ts.items():
        _, ety = eta(y); eta_pac[name]=ety

    # Bootstraps (pseudo onsets)
    boot = {'plv':[], 'beta':[], 'mincut':[], 'pac':{k:[] for k in pac_ts}}
    rng = np.random.default_rng(17)
    if n_boot>0 and onsets.size>0:
        m = len(tvec)
        for _ in range(n_boot):
            # sample the same number of onsets uniformly, avoid edges
            cand = rng.integers(low=int((span_sec)*1.0/step_sec), high=m-int((span_sec)*1.0/step_sec), size=onsets.size)
            cand.sort()
            def eta_c(y):
                return _eta_time_series(cand, tvec, np.asarray(y,float), span_sec)[1]
            boot['plv'].append(eta_c(plv_ts))
            boot['beta'].append(eta_c(beta_ts))
            boot['mincut'].append(eta_c(mincut_ts))
            for name,y in pac_ts.items():
                boot['pac'][name].append(eta_c(y))
        # summarize mean±95% band
        boot['plv'] = np.vstack(boot['plv']);     boot['plv_ci'] = (np.nanpercentile(boot['plv'], 2.5, axis=0), np.nanpercentile(boot['plv'], 97.5, axis=0))
        boot['beta']= np.vstack(boot['beta']);    boot['beta_ci']= (np.nanpercentile(boot['beta'],2.5, axis=0), np.nanpercentile(boot['beta'],97.5, axis=0))
        boot['mincut']=np.vstack(boot['mincut']); boot['mincut_ci']=(np.nanpercentile(boot['mincut'],2.5,axis=0), np.nanpercentile(boot['mincut'],97.5,axis=0))
        for name in pac_ts:
            arr = np.vstack(boot['pac'][name])
            boot['pac'][name] = arr
            boot['pac_ci_'+name] = (np.nanpercentile(arr, 2.5, axis=0), np.nanpercentile(arr, 97.5, axis=0))

    # Plots
    if show:
        def bandplot(ax, y, ci, label, color):
            ax.plot(tau, y, color=color, lw=1.8, label=label)
            if ci is not None:
                ax.fill_between(tau, ci[0], ci[1], color=color, alpha=0.2, linewidth=0)
        # PLV
        fig, axs = plt.subplots(2,2, figsize=(12,6), constrained_layout=True)
        bandplot(axs[0,0], eta_plv, boot['plv_ci'] if 'plv_ci' in boot else None, f'PLV {plv_band[0]}–{plv_band[1]} Hz', 'tab:blue')
        axs[0,0].axvline(0, color='k', lw=0.8); axs[0,0].set_title('ETA: PLV around K≥%d onsets' % K); axs[0,0].set_xlabel('Time (s)'); axs[0,0].set_ylabel('PLV')
        # β
        bandplot(axs[0,1], eta_beta, boot['beta_ci'] if 'beta_ci' in boot else None, f'β {beta_band[0]}–{beta_band[1]} Hz', 'tab:orange')
        axs[0,1].axvline(0, color='k', lw=0.8); axs[0,1].set_title('ETA: 1/f slope β (lower=flatter)'); axs[0,1].set_xlabel('Time (s)'); axs[0,1].set_ylabel('β')
        # min-cut
        bandplot(axs[1,0], eta_mincut, boot['mincut_ci'] if 'mincut_ci' in boot else None, f'min-cut {mincut_band[0]}–{mincut_band[1]} Hz', 'tab:green')
        axs[1,0].axvline(0, color='k', lw=0.8); axs[1,0].set_title('ETA: global min-cut'); axs[1,0].set_xlabel('Time (s)'); axs[1,0].set_ylabel('cut')
        # PAC (first pair)
        name0 = list(eta_pac.keys())[0]
        ci = boot.get('pac_ci_'+name0)
        bandplot(axs[1,1], eta_pac[name0], ci if ci is not None else None, f'PAC {name0}', 'tab:red')
        axs[1,1].axvline(0, color='k', lw=0.8); axs[1,1].set_title('ETA: PAC'); axs[1,1].set_xlabel('Time (s)'); axs[1,1].set_ylabel('MI')
        plt.show()

    return {
        'eta_time': tau,
        'eta_plv': eta_plv,
        'eta_beta': eta_beta,
        'eta_mincut': eta_mincut,
        'eta_pac': eta_pac,
        'n_onsets': int(len(np.where(np.diff((ov_p>=K).astype(int))==1)[0]))
    }

def _infer_fs(df: pd.DataFrame, time_col: str)->float:
    t = np.asarray(pd.to_numeric(df[time_col], errors='coerce').values, float)
    dt = np.diff(t); dt = dt[(dt>0)&np.isfinite(dt)]
    if dt.size==0: raise ValueError("Cannot infer fs from time column.")
    return float(1.0/np.median(dt))

def estimate_sr_harmonics(RECORDS, sr_channel='EEG.F4', fs=None,
                          f_can=(7.83, 14.3, 20.8, 27.3, 33.8),
                          search_halfband=0.8, nperseg_sec=32.0, overlap=0.5, time_col="Timestamp"):
    if fs is None:
        fs = _infer_fs(RECORDS,time_col)
    x = _get_channel_vector(RECORDS, sr_channel)
    nper = int(round(nperseg_sec*fs)); nover = int(round(overlap*nper))
    f, Pxx = welch(x, fs=fs, nperseg=nper, noverlap=nover, window='hann', detrend='constant', scaling='density')
    def _peak_near(f0, half=0.8):
        m = (f >= max(0.1, f0-half)) & (f <= (f0+half))
        if not np.any(m): return f0
        ff, pp = f[m], Pxx[m]
        k = int(np.nanargmax(pp))
        # small parabolic refine in log power
        if 0 < k < len(pp)-1 and pp[k-1]>0 and pp[k]>0 and pp[k+1]>0:
            y1,y2,y3 = np.log(pp[k-1]), np.log(pp[k]), np.log(pp[k+1])
            denom = (y1 - 2*y2 + y3)
            delta = 0.5*(y1 - y3)/denom if denom != 0 else 0.0
            step = ff[1]-ff[0]
            return float(np.clip(ff[k] + delta*step, ff[0], ff[-1]))
        return float(ff[k])
    return [_peak_near(f0, half=search_halfband) for f0 in f_can]
