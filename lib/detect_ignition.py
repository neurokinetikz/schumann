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
import os, json
from typing import Optional, List, Tuple, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import zscore

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

def _channel_latencies(X: np.ndarray, fs: float, f_lo: float, f_hi: float,
                       t0: float, pre: float = 2.0, post: float = 1.0, z_th: float = 2.0) -> np.ndarray:
    Xb = bandpass_safe(X, fs, f_lo, f_hi)
    amp = np.abs(signal.hilbert(Xb, axis=-1))
    n = X.shape[1]
    t0_idx = int(round(t0*fs))
    i0 = max(0, t0_idx - int(round(pre*fs)))
    i1 = min(n, t0_idx + int(round(post*fs)))
    base = amp[:, i0:t0_idx]
    mu = base.mean(axis=1, keepdims=True)
    sd = base.std(axis=1, keepdims=True) + 1e-12
    z = (amp[:, i0:i1] - mu)/sd
    lats = np.full(X.shape[0], np.nan)
    for ch in range(X.shape[0]):
        idx = np.where(z[ch] >= z_th)[0]
        if idx.size:
            lats[ch] = (i0 + idx[0])/fs
    return lats


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
    pf = np.mean(bandpass_safe(x, fs, base_hz-base_bw_hz, base_hz+base_bw_hz)**2)
    powers = []
    centers = []
    for f0 in harmonic_centers_hz:
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
    center_hz: float = 7.83, half_bw_hz: float = 0.6,
    smooth_sec: float = 0.25, z_thresh: float = 2.5,
    min_isi_sec: float = 2.0, window_sec: float = 20.0, merge_gap_sec: float = 5.0,
    R_band: Tuple[float, float] = (8,13), R_win_sec: float = 1.0, R_step_sec: float = 0.25,
    eta_pre_sec: float = 10.0, eta_post_sec: float = 10.0,
    sr_reference: str = 'auto-SSD',
    seed_method: str = 'latency',
    pel_band: Tuple[float,float] = (60, 90),
    electrode_xy: Optional[Dict[str, Tuple[float,float]]] = None,
    harmonics: Tuple[int,...] = (2,3,4,5,6,7),
    harmonics_hz: Optional[List[float]] = None,
    harmonic_bw_hz: Optional[float] = None,
    make_passport: bool = True,
    show: bool = True,
    verbose: bool = True
) -> Tuple[Dict[str, object], List[Tuple[int,int]]]:

    if eeg_channels is None:
        eeg_channels = [c for c in RECORDZ.columns if c.startswith('EEG.')]

    _ensure_dir(out_dir)
    time_col = ensure_timestamp_column(RECORDZ, time_col=time_col, default_fs=128.0)
    fs = infer_fs(RECORDZ, time_col)
    t = pd.to_numeric(RECORDZ[time_col], errors='coerce').values.astype(float)

    # --- 1) SR envelope z(t) & onsets (proposal via sr_channel) ---
    y = get_series(RECORDZ, sr_channel)
    yb = bandpass_safe(y, fs, center_hz - half_bw_hz, center_hz + half_bw_hz)
    env = np.abs(signal.hilbert(yb))
    n_smooth = max(1, int(round(smooth_sec*fs)))
    if n_smooth > 1:
        w = np.hanning(n_smooth); w /= w.sum()
        env_s = np.convolve(env, w, mode='same')
    else:
        env_s = env
    z = zscore(env_s, nan_policy='omit')
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

    # per-session valid harmonics (below Nyquist with small margin)
    valid_harmonics = [f0 for f0 in harmonic_centers if (f0 + (harmonic_bw_hz or half_bw_hz)) < (fs/2.0 - 1e-3)]
    if not valid_harmonics:
        valid_harmonics = [2*center_hz]
    hbw = harmonic_bw_hz if harmonic_bw_hz is not None else half_bw_hz

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
        valid_harmonics_ot = [k*center_hz for k in (2,3,4,5,6,7) if (k*center_hz + hbw) < (fs/2.0 - 1e-3)]

    # clamp gamma band to Nyquist
    g_lo, g_hi = pel_band
    g_lo, g_hi = _safe_band(g_lo, g_hi, fs)
    gamma_band = (g_lo, g_hi)


    if verbose:
        print("\n=== Ignition Detection — Session Summary ===\n")
        # print(f"SR reference: {sr_channel}")
        print(f"Ignition windows: {ignition_windows_rounded}")
        print("Estimated SR: ", np.round(valid_harmonics,2))
        print(f"EEG channels (n={len(eeg_channels)}): {', '.join([c.split('.',1)[-1] for c in eeg_channels])}")
        print(f"Detection band: {center_hz:.2f}±{half_bw_hz:.2f} Hz; z-thresh={z_thresh:.2f}; window={window_sec:.1f}s; min_ISI={min_isi_sec:.1f}s")
        print(f"R(t) band: {R_band[0]:.1f}–{R_band[1]:.1f} Hz, win={R_win_sec:.2f}s, step={R_step_sec:.2f}s")
        print(f"Event SR mode: {sr_reference}")

    

    ch_short = [c.split('.',1)[-1] for c in eeg_channels]

    for (a, b) in ign:
        i0 = max(0, int(round(a*fs)))
        i1 = min(L, int(round(b*fs)))
        if i1 - i0 < int(2*fs):
            continue
        Xw = X[:, i0:i1]

        # virtual SR
        if sr_reference.upper() == 'F4' and 'EEG.F4' in eeg_channels:
            idx = eeg_channels.index('EEG.F4')
            v_sr = Xw[idx]
            w_sr = np.zeros(len(eeg_channels)); w_sr[idx] = 1.0
        else:
            v_sr, w_sr = _build_virtual_sr(Xw, fs, center_hz, half_bw_hz, mode=sr_reference)

        # t0 from SR1 band
        f_lo, f_hi = center_hz - half_bw_hz, center_hz + half_bw_hz
        tR_ev, R_ev = _kuramoto_R_timeseries(Xw, fs, f_lo, f_hi, win_sec=0.5, step_sec=0.05)
        tR_ev = tR_ev + a
        t0_net = _detect_t0_from_R(tR_ev, R_ev, thresh=0.6)
        if not np.isfinite(t0_net):
            t0_net = 0.5*(a+b)

        # t0-centered zR maxima
        mskR_ev = (t_cent >= (t0_net - 2.5)) & (t_cent <= (t0_net + 2.5))
        zR_max_ev = float(np.nanmax(zR[mskR_ev])) if np.any(mskR_ev) else np.nan
        zR_peak_5s = zR_max_ev

        # latencies & spread
        lats = _channel_latencies(X, fs, f_lo, f_hi, t0_net, pre=2.0, post=1.0, z_th=2.0)
        seed_idx = int(np.nanargmin(lats)) if np.any(np.isfinite(lats)) else 0
        seed_ch = eeg_channels[seed_idx]
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
            base_hz=center_hz, base_bw_hz=half_bw_hz,
            harmonic_centers_hz=valid_harmonics_ot, harmonic_bw_hz=hbw
        )

        # Estimate per-event fundamental (base) to sanitize MaxH against local base
        try:
            fw, Pw = signal.welch(v_sr, fs=fs, nperseg=int(2*fs))
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
        if seg.size > 10:
            th = bandpass_safe(seg, fs, center_hz-0.3, center_hz+0.3)
            ga = bandpass_safe(seg, fs, gamma_band[0], gamma_band[1])
            env_th = np.abs(signal.hilbert(th))
            env_ga = np.abs(signal.hilbert(ga))
            tt = np.arange(seg.size)/fs + (t0_net-2.0)
            k0 = np.argmin(np.abs(tt - t0_net))
            p_th = np.argmax(env_th[:k0]) if k0>0 else 0
            p_ga = np.argmax(env_ga[:k0]) if k0>0 else 0
            PEL = float(tt[p_th] - tt[p_ga])
        else:
            PEL = np.nan

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

        # per-window coherence vs SR channel and vs vSR
        fE, CE = signal.coherence(Xw.mean(axis=0), y[i0:i1], fs=fs, nperseg=int(2*fs))
        idxF = int(np.argmin(np.abs(fE - center_hz)))
        msc_sr = float(CE[idxF])
        fEv, CEv = signal.coherence(Xw.mean(axis=0), v_sr, fs=fs, nperseg=int(2*fs))
        idxV = int(np.argmin(np.abs(fEv - center_hz)))
        msc_v = float(CEv[idxV])

        # --- MSC peak vs average: compute around t0_net (±2.5 s) ---
        x_mean = Xw.mean(axis=0)
        y_ref  = v_sr
        def _slice_idx(t0, left, right):
            i0s = max(0, int(round((t0 + left - a)*fs)))
            i1s = min(len(x_mean), int(round((t0 + right - a)*fs)))
            return i0s, i1s
        i0_loc, i1_loc   = _slice_idx(t0_net, -2.5, +2.5)
        i0_base, i1_base = _slice_idx(t0_net, -5.0, -2.0)
        msc_loc_vals  = _msc_f0_series(x_mean[i0_loc:i1_loc],  y_ref[i0_loc:i1_loc],  fs, center_hz, win=1.0, step=0.1)
        msc_base_vals = _msc_f0_series(x_mean[i0_base:i1_base], y_ref[i0_base:i1_base], fs, center_hz, win=1.0, step=0.1)
        msc_peak      = float(np.nanmax(msc_loc_vals))   if msc_loc_vals.size  else np.nan
        msc_mean_loc  = float(np.nanmean(msc_loc_vals))  if msc_loc_vals.size  else np.nan
        msc_base      = float(np.nanmean(msc_base_vals)) if msc_base_vals.size else np.nan
        msc_auc_loc   = float(msc_mean_loc * max(0.0, (i1_loc - i0_loc)/fs)) if np.isfinite(msc_mean_loc) else np.nan

        # SR envelope summaries from reference channel z(t)
        i0w = max(0, int(np.floor(a*fs)))
        i1w = min(len(z), int(np.ceil(b*fs)))
        if i1w - i0w > 0:
            seg_z = z[i0w:i1w]
            k_rel = int(np.nanargmax(seg_z))
            k_peak = i0w + k_rel
            sr_z_max = float(seg_z[k_rel])
            sr_z_peak_t = float(t[k_peak])
            t_on = a + window_sec/2.0
            k_on = int(np.argmin(np.abs(t - t_on)))
            kL2 = max(0, k_on - int(5*fs))
            kR2 = min(len(z), k_on + int(5*fs))
            sr_z_mean_pm5 = float(np.nanmean(z[kL2:kR2])) if kR2>kL2 else np.nan
            k_postR = min(len(z), k_peak + int(5*fs))
            sr_z_mean_post5 = float(np.nanmean(z[k_peak:k_postR])) if k_postR>k_peak else np.nan
        else:
            sr_z_max = sr_z_peak_t = sr_z_mean_pm5 = sr_z_mean_post5 = np.nan

        # label
        if (fs_z >= 3.0) and (HSI >= 0.2):
            type_label = 'fundamental-led'
        elif (fs_z < 2.0) and (HSI >= 0.5 or (np.isfinite(MaxH) and MaxH >= 6*center_hz-1.0)):
            type_label = 'overtone-led'
        else:
            pks, _ = signal.find_peaks(z_env, distance=int(1.0*fs), height=0.6*np.nanmax(z_env))
            type_label = 'two-phase' if len(pks) >= 2 else 'fundamental-led'

        rows.append({
            't_start': a, 't_end': b, 'duration_s': float(b-a),
            't0_net': t0_net, 'zR_max': zR_max_ev, 'zR_peak_±5s': zR_peak_5s,
            'fs_z': fs_z, 'fs_auc': fs_auc, 'HSI': HSI, 'MaxH': MaxH, 'MaxH_overtone': MaxH_ov, 'PEL_sec': PEL,
            'seed_ch': seed_ch, 'seed_roi': seed_roi, 'spread_time_sec': spread, 'SF': SF,
            'msc_7p83_sr': msc_sr, 'msc_7p83_v': msc_v,
            'msc_7p83_v_peak': msc_peak, 'msc_7p83_v_mean_local': msc_mean_loc,
            'msc_7p83_v_base': msc_base, 'msc_7p83_v_auc_loc': msc_auc_loc,
            'sr_z_max': sr_z_max, 'sr_z_peak_t': sr_z_peak_t,
            'sr_z_mean_pm5': sr_z_mean_pm5, 'sr_z_mean_post5': sr_z_mean_post5,
            'type_label': type_label,
        })

    events = pd.DataFrame(rows)

    # --- 5) ETA of zR(t) aligned to t0_net ---
    if onsets.size and not events.empty:
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
    plt.figure(figsize=(11,3))
    plt.plot(t[:len(z)], z, lw=1.0, label='SR env z (ref)')
    plt.axhline(z_thresh, color='k', ls='--', lw=1, label='z-thresh')
    for (aa,bb) in ign: plt.axvspan(aa,bb, color='tab:orange', alpha=0.15)
    plt.xlabel('Time (s)'); plt.ylabel('SR z'); plt.title('SR envelope z(t) with detected ignitions')
    plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(out_dir,'sr_env_z.png'), dpi=140)
    if show: plt.show();
    plt.close()

    # plt.figure(figsize=(11,3))
    # plt.plot(t_cent, zR, lw=1.0, label=f'zR(t) {R_band[0]}–{R_band[1]} Hz')
    # for (aa,bb) in ign: plt.axvspan(aa,bb, color='tab:orange', alpha=0.15)
    # plt.xlabel('Time (s)'); plt.ylabel('zR'); plt.title('Global synchrony R(t)')
    # plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(out_dir,'R_timeseries.png'), dpi=140)
    # if show: plt.show();
    # plt.close()

    # if tau.size:
    #     plt.figure(figsize=(7.5,3))
    #     plt.plot(tau, eta_mean, lw=1.6, label='mean zR (aligned to t0_net)')
    #     if np.any(np.isfinite(eta_sem)):
    #         plt.fill_between(tau, eta_mean-eta_sem, eta_mean+eta_sem, alpha=0.2)
    #     plt.axvline(0, color='k', lw=1)
    #     plt.xlabel('Time from t0_net (s)'); plt.ylabel('zR'); plt.title('Event-triggered zR(t)')
    #     plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(out_dir,'ETA_zR.png'), dpi=140)
    #     if show: plt.show();
    #     plt.close()

    # # MaxH_hz distribution across events (use sanitized overtone-only values)
    # if not events.empty and ('MaxH_overtone' in events.columns):
    #     mh = pd.to_numeric(events['MaxH_overtone'], errors='coerce').to_numpy()
    #     mh = mh[np.isfinite(mh)]
    #     # remove any residual base-neighborhood values
    #     if mh.size:
    #         mh = mh[np.abs(mh - base_guess) > (base_margin + 1e-6)]
    #     if mh.size and len(valid_harmonics_ot):
    #         plt.figure(figsize=(7.5,3))
    #         lo = float(min(valid_harmonics_ot))
    #         hi = float(min(max(valid_harmonics_ot), fs/2.0-1e-3))
    #         nb = max(8, min(30, len(valid_harmonics_ot)*3))
    #         bins = np.linspace(lo, hi, nb)
    #         plt.hist(mh, bins=bins, alpha=0.75, edgecolor='k')
    #         for f0 in valid_harmonics_ot:
    #             plt.axvline(f0, color='tab:orange', alpha=0.5, lw=1)
    #         plt.xlabel('MaxH (overtone) frequency (Hz)')
    #         plt.ylabel('Event count')
    #         plt.title('MaxH_overtone distribution across events')
    #         plt.tight_layout(); plt.savefig(os.path.join(out_dir,'MaxH_hz_distribution.png'), dpi=140)
    #         if show: plt.show();
    #         plt.close()

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

    events.to_csv(os.path.join(out_dir,'events.csv'), index=False)
    pd.DataFrame([summary]).to_csv(os.path.join(out_dir,'summary.csv'), index=False)
    if make_passport:
        events.to_csv(os.path.join(out_dir,'event_passport.csv'), index=False)

    if verbose:
        # print("\n=== Ignition Detection — Session Summary ===")
        # print(f"Ignition windows (rounded, whole seconds): {ignition_windows_rounded}")
        # print(f"SR reference: {sr_channel}")
        # print(f"EEG channels (n={len(eeg_channels)}): {', '.join([c.split('.',1)[-1] for c in eeg_channels])}")
        # print(f"Detection band: {center_hz:.2f}±{half_bw_hz:.2f} Hz; z-thresh={z_thresh:.2f}; window={window_sec:.1f}s; min_ISI={min_isi_sec:.1f}s")
        # print(f"R(t) band: {R_band[0]:.1f}–{R_band[1]:.1f} Hz, win={R_win_sec:.2f}s, step={R_step_sec:.2f}s")
        # print(f"Event SR mode: {sr_reference}")
        harm_src = 'custom' if (harmonics_hz and len(harmonics_hz)) else 'multiples'
        # print(f"PEL gamma band: {gamma_band[0]:.1f}–{gamma_band[1]:.1f} Hz; Harmonics (valid, {harm_src}): {np.round(valid_harmonics,3)}")

        def fmt_iqr(x: np.ndarray) -> str:
            x = np.asarray(x, float)
            x = x[np.isfinite(x)]
            if x.size == 0: return "n/a"
            q1, med, q3 = np.nanpercentile(x, [25, 50, 75])
            return f"{med:.2f} [{q1:.2f}, {q3:.2f}]"

        

        n_events = int(len(events)) if not events.empty else 0
        print(f"\nEvents detected: {n_events}")
        if n_events > 0:
            dur   = events['duration_s'].to_numpy()
            srmax = events['sr_z_max'].to_numpy()
            srpm5 = events['sr_z_mean_pm5'].to_numpy()
            msc_v  = events['msc_7p83_v'].to_numpy()  if 'msc_7p83_v'  in events.columns else np.array([])
            msc_pk = events['msc_7p83_v_peak'].to_numpy() if 'msc_7p83_v_peak' in events.columns else np.array([])


            rec_cov = (100.0*np.nansum(dur)/max(1e-9, t[-1]-t[0])) if dur.size else np.nan
            

            # event-centric
            fsz  = events['fs_z'].to_numpy()
            HSIv = events['HSI'].to_numpy()
            PELv = events['PEL_sec'].to_numpy()
            spread= events['spread_time_sec'].to_numpy()
            SFv   = events['SF'].to_numpy()
            seed_counts = events['seed_roi'].value_counts(dropna=True)
            type_counts = events['type_label'].value_counts(dropna=True)

            score = srmax * msc_v / (1+HSIv)
            events['score'] = score

            print(f"  Duration (s)           — median [IQR]: {fmt_iqr(dur)}")
            print(f"  SR z max (ref)         — median [IQR]: {fmt_iqr(srmax)}")
            print(f"  SR z mean (±5 s)       — median [IQR]: {fmt_iqr(srpm5)}")
            print(f"  MSC@~7.83 (virtual)    — median [IQR]: {fmt_iqr(msc_v)}")
            print(f"  HSI (harmonic stack)   — median [IQR]: {fmt_iqr(HSIv)}")
            print(f"  Score                  — median [IQR]: {fmt_iqr(score)}")
            # print(f"  MSC@~7.83 peak         — median [IQR]: {fmt_iqr(msc_pk)}")
            print(f"  Coverage of recording  — {rec_cov:.2f}%")
            # print("\n— Event-centric metrics —")
            # print(f"  FS z (SR1)             — median [IQR]: {fmt_iqr(fsz)}")
            
            # # print(f"  PEL Γ→θ lag (s)        — median [IQR]: {fmt_iqr(PELv)}")
            # print(f"  Seed ROI distribution  — ", ",".join([f"{k}: {int(v)} ({100.0*v/n_events:.0f}%)" for k,v in seed_counts.items()]))
            # print(f"  Spread time (s)        — median [IQR]: {fmt_iqr(spread)}")
            # print(f"  Synchronized fraction  — median [IQR]: {fmt_iqr(SFv)}")
            

            # Top tables (SR z, FS z, HSI)
            try:
                top_by_srz = events.sort_values('score', ascending=False)
                cols2 = [c for c in ['t_start','t_end','duration_s','sr_z_max','sr_z_mean_pm5','msc_7p83_v',
                                        'HSI','fs_z','type_label','seed_roi','seed_ch','score'] if c in events.columns]
                print("\nTop events by Score (sr_z_max * msc_7p83_v * HSI):")
                print(top_by_srz[cols2].to_string(index=False, justify='center'))
            except Exception:
                pass

            print("")
        # print(f"\nFiles written to: {out_dir}")
        # print("  - sr_env_z.png, R_timeseries.png, ETA_zR.png, MaxH_hz_distribution.png")
        # print("  - events.csv, summary.csv, event_passport.csv")

    result = {
        'events': events,
        'summary': summary,
        'ignition_windows': ign,
        'ignition_windows_rounded': ignition_windows_rounded,
        'ignition_windows_path': ign_json_path,
        'fs': fs,
        't_R': t_cent,
        'zR': zR,
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
        'harmonics_source': ('custom' if (harmonics_hz and len(harmonics_hz)) else 'multiples')
    }
    return result, ignition_windows_rounded

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