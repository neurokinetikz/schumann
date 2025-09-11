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
    if verbose:
        print(f"Ignition windows (rounded, whole seconds): {ignition_windows_rounded}")
        print(f"Saved → {ign_json_path}")

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

    plt.figure(figsize=(11,3))
    plt.plot(t_cent, zR, lw=1.0, label=f'zR(t) {R_band[0]}–{R_band[1]} Hz')
    for (aa,bb) in ign: plt.axvspan(aa,bb, color='tab:orange', alpha=0.15)
    plt.xlabel('Time (s)'); plt.ylabel('zR'); plt.title('Global synchrony R(t)')
    plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(out_dir,'R_timeseries.png'), dpi=140)
    if show: plt.show();
    plt.close()

    if tau.size:
        plt.figure(figsize=(7.5,3))
        plt.plot(tau, eta_mean, lw=1.6, label='mean zR (aligned to t0_net)')
        if np.any(np.isfinite(eta_sem)):
            plt.fill_between(tau, eta_mean-eta_sem, eta_mean+eta_sem, alpha=0.2)
        plt.axvline(0, color='k', lw=1)
        plt.xlabel('Time from t0_net (s)'); plt.ylabel('zR'); plt.title('Event-triggered zR(t)')
        plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(out_dir,'ETA_zR.png'), dpi=140)
        if show: plt.show();
        plt.close()

    # MaxH_hz distribution across events (use sanitized overtone-only values)
    if not events.empty and ('MaxH_overtone' in events.columns):
        mh = pd.to_numeric(events['MaxH_overtone'], errors='coerce').to_numpy()
        mh = mh[np.isfinite(mh)]
        # remove any residual base-neighborhood values
        if mh.size:
            mh = mh[np.abs(mh - base_guess) > (base_margin + 1e-6)]
        if mh.size and len(valid_harmonics_ot):
            plt.figure(figsize=(7.5,3))
            lo = float(min(valid_harmonics_ot))
            hi = float(min(max(valid_harmonics_ot), fs/2.0-1e-3))
            nb = max(8, min(30, len(valid_harmonics_ot)*3))
            bins = np.linspace(lo, hi, nb)
            plt.hist(mh, bins=bins, alpha=0.75, edgecolor='k')
            for f0 in valid_harmonics_ot:
                plt.axvline(f0, color='tab:orange', alpha=0.5, lw=1)
            plt.xlabel('MaxH (overtone) frequency (Hz)')
            plt.ylabel('Event count')
            plt.title('MaxH_overtone distribution across events')
            plt.tight_layout(); plt.savefig(os.path.join(out_dir,'MaxH_hz_distribution.png'), dpi=140)
            if show: plt.show();
            plt.close()

    # --- 7) summaries & files ---
    if events.empty:
        summary = {'n_events': 0}
    else:
        summary = {
            'n_events': int(len(events)),
            'median_duration_s': float(events['duration_s'].median()),
            'median_fs_z': float(events['fs_z'].median()),
            'median_HSI': float(events['HSI'].median()),
            'median_PEL_sec': float(events['PEL_sec'].median()),
            'coverage_pct': float(100.0*np.sum(events['duration_s'])/max(1e-9, t[-1]-t[0]))
        }

    events.to_csv(os.path.join(out_dir,'events.csv'), index=False)
    pd.DataFrame([summary]).to_csv(os.path.join(out_dir,'summary.csv'), index=False)
    if make_passport:
        events.to_csv(os.path.join(out_dir,'event_passport.csv'), index=False)

    if verbose:
        print("\n=== Ignition Detection — Session Summary ===")
        print(f"SR reference: {sr_channel}")
        print(f"EEG channels (n={len(eeg_channels)}): {', '.join([c.split('.',1)[-1] for c in eeg_channels])}")
        print(f"Detection band: {center_hz:.2f}±{half_bw_hz:.2f} Hz; z-thresh={z_thresh:.2f}; window={window_sec:.1f}s; min_ISI={min_isi_sec:.1f}s")
        print(f"R(t) band: {R_band[0]:.1f}–{R_band[1]:.1f} Hz, win={R_win_sec:.2f}s, step={R_step_sec:.2f}s")
        print(f"Event SR mode: {sr_reference}")
        harm_src = 'custom' if (harmonics_hz and len(harmonics_hz)) else 'multiples'
        print(f"PEL gamma band: {gamma_band[0]:.1f}–{gamma_band[1]:.1f} Hz; Harmonics (valid, {harm_src}): {np.round(valid_harmonics,3)}")

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
            msc_v = events['msc_7p83_v'].to_numpy() if 'msc_7p83_v' in events.columns else np.array([])

            rec_cov = (100.0*np.nansum(dur)/max(1e-9, t[-1]-t[0])) if dur.size else np.nan
            print(f"  Duration (s)           — median [IQR]: {fmt_iqr(dur)}")
            print(f"  SR z max (ref)         — median [IQR]: {fmt_iqr(srmax)}")
            print(f"  SR z mean (±5 s)       — median [IQR]: {fmt_iqr(srpm5)}")
            print(f"  MSC@~7.83 (virtual)    — median [IQR]: {fmt_iqr(msc_v)}")
            print(f"  Coverage of recording  — {rec_cov:.2f}%")

            # event-centric
            fsz  = events['fs_z'].to_numpy()
            HSIv = events['HSI'].to_numpy()
            PELv = events['PEL_sec'].to_numpy()
            spread= events['spread_time_sec'].to_numpy()
            SFv   = events['SF'].to_numpy()
            seed_counts = events['seed_roi'].value_counts(dropna=True)
            type_counts = events['type_label'].value_counts(dropna=True)

            print("\n— Event-centric metrics —")
            print(f"  FS z (SR1)             — median [IQR]: {fmt_iqr(fsz)}")
            print(f"  HSI (harmonic stack)   — median [IQR]: {fmt_iqr(HSIv)}")
            print(f"  PEL Γ→θ lag (s)        — median [IQR]: {fmt_iqr(PELv)}")
            print(f"  Seed ROI distribution  — ", ", ".join([f"{k}: {int(v)} ({100.0*v/n_events:.0f}%)" for k,v in seed_counts.items()]))
            print(f"  Spread time (s)        — median [IQR]: {fmt_iqr(spread)}")
            print(f"  Synchronized fraction  — median [IQR]: {fmt_iqr(SFv)}")

            # Top tables (SR z, FS z, HSI)
            try:
                top_by_srz = events.sort_values('sr_z_max', ascending=False)
                cols2 = [c for c in ['t_start','t_end','duration_s','sr_z_max','sr_z_mean_pm5','msc_7p83_v'] if c in events.columns]
                print("\nTop events by SR z:")
                print(top_by_srz[cols2].to_string(index=False, justify='center'))
            except Exception:
                pass
            try:
                top_by_srz = events.sort_values('msc_7p83_v', ascending=False)
                cols2 = [c for c in ['t_start','t_end','duration_s','sr_z_max','sr_z_mean_pm5','msc_7p83_v'] if c in events.columns]
                print("\nTop events by MSC:")
                print(top_by_srz[cols2].to_string(index=False, justify='center'))
            except Exception:
                pass
            try:
                top_by_fsz = events.sort_values('fs_z', ascending=False)
                cols2 = [c for c in ['t_start','t_end','duration_s','fs_z','HSI','MaxH','seed_ch','seed_roi'] if c in events.columns]
                print("\nTop events by FS z (SR1):")
                print(top_by_fsz[cols2].to_string(index=False, justify='center'))
            except Exception:
                pass
            try:
                top_by_hsi = events.sort_values('HSI', ascending=False)
                cols3 = [c for c in ['t_start','t_end','duration_s','HSI','MaxH','fs_z','type_label'] if c in events.columns]
                print("\nTop events by HSI (harmonics):")
                print(top_by_hsi[cols3].to_string(index=False, justify='center'))
            except Exception:
                pass

        print(f"Files written to: {out_dir}")
        print("  - sr_env_z.png, R_timeseries.png, ETA_zR.png, MaxH_hz_distribution.png")
        print("  - events.csv, summary.csv, event_passport.csv")

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
