import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
from scipy import signal
from scipy.signal.windows import hann

# Reuse-safe filters with guardrails
def _butter_bandpass(x: np.ndarray, fs: float, f1: float, f2: float, order: int = 4) -> np.ndarray:
    ny = 0.5 * fs
    # Clamp frequencies to valid range
    if f1 <= 0: f1 = 1e-6
    if f2 >= ny: f2 = ny - 1e-6
    if not (0 < f1 < f2 < ny):
        raise ValueError(f"Invalid bandpass range: ({f1}, {f2}) with fs={fs}")
    b, a = signal.butter(order, [f1/ny, f2/ny], btype='band')
    return signal.filtfilt(b, a, x.astype(float))


def _phase_amp(x: np.ndarray, fs: float, f_phase: Tuple[float, float], f_amp: Tuple[float, float]):
    xp = _butter_bandpass(x, fs, *f_phase)
    xa = _butter_bandpass(x, fs, *f_amp)
    ph = np.angle(signal.hilbert(xp))
    amp = np.abs(signal.hilbert(xa))
    return ph, amp

# -----------------------------
# Helpers for RECORDS (DataFrame)
# -----------------------------

_DEF_TIME_COL = 'Timestamp'

def infer_fs_from_records(RECORDS: pd.DataFrame, time_col: str = _DEF_TIME_COL) -> float:
    t = RECORDS[time_col].values
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if len(dt) == 0:
        raise ValueError("Cannot infer fs: Timestamp spacing invalid.")
    return float(1.0 / np.median(dt))

_DEF_CH_CAND_PATTERNS = (
    "EEG.{ch}", "eeg.{ch}", "{ch}", "RAW.{ch}", "CHAN.{ch}"
)

def _find_channel_series(RECORDS: pd.DataFrame, ch_name: str) -> Optional[pd.Series]:
    for pat in _DEF_CH_CAND_PATTERNS:
        col = pat.format(ch=ch_name)
        if col in RECORDS.columns:
            s = pd.to_numeric(RECORDS[col], errors='coerce').astype(float)
            return s
    return None

def _sanitize_band(band: Tuple[float,float], fs: float, min_bw: float = 0.5) -> Optional[Tuple[float,float]]:
    lo, hi = float(band[0]), float(band[1])
    ny = 0.5*fs
    if not np.isfinite(lo) or not np.isfinite(hi):
        return None
    lo = max(lo, 1e-6)              # >0
    hi = min(hi, ny - 1e-6)         # < Nyquist
    if hi - lo < min_bw:            # ensure a tiny bandwidth
        lo = max(1e-6, hi - min_bw)
    if not (0 < lo < hi < ny):
        return None
    return (lo, hi)

def _sanitize_band_list(bands: List[Tuple[float,float]], fs: float, label: str) -> List[Tuple[float,float]]:
    out = []
    for b in bands:
        sb = _sanitize_band(b, fs)
        if sb is not None:
            out.append(sb)
        else:
            print(f"[warn] Skipping invalid {label} band {b} for fs={fs:.2f} Hz (Nyquist={0.5*fs:.2f} Hz)")
    if not out:
        print(f"[warn] No valid {label} bands remain after sanitization for fs={fs:.2f} Hz")
    return out



# -----------------------------
# PAC (Tort MI / GLM-R2) working on np.array + fs
# -----------------------------

def pac_tort_mi(phase: np.ndarray, amp: np.ndarray, nbins: int = 18) -> float:
    bins = np.linspace(-np.pi, np.pi, nbins+1)
    idx = np.digitize(((phase + np.pi) % (2*np.pi)) - np.pi, bins) - 1
    idx = np.clip(idx, 0, nbins-1)
    mean_amp = np.array([amp[idx == k].mean() if np.any(idx == k) else 0. for k in range(nbins)])
    if mean_amp.sum() <= 0:
        return 0.0
    p = mean_amp / mean_amp.sum()
    p = np.where(p > 0, p, 1e-12)
    H = -np.sum(p*np.log(p))
    Hmax = np.log(nbins)
    return float((Hmax - H) / Hmax)


def pac_glm_r2(phase: np.ndarray, amp: np.ndarray) -> float:
    X = np.c_[np.ones_like(phase), np.sin(phase), np.cos(phase)]
    beta, *_ = np.linalg.lstsq(X, amp, rcond=None)
    yhat = X @ beta
    ss_res = np.sum((amp - yhat)**2)
    ss_tot = np.sum((amp - amp.mean())**2) + 1e-12
    return float(1. - (ss_res/ss_tot))


def pac_surrogate_z(phase: np.ndarray, amp: np.ndarray, method: str = 'mi', n: int = 200) -> Tuple[float, float]:
    rng = np.random.RandomState(0)
    stat = pac_glm_r2(phase, amp) if method == 'glm' else pac_tort_mi(phase, amp)
    sur = []
    for _ in range(n):
        shift = rng.randint(len(amp))
        amp_s = np.roll(amp, shift)
        s = pac_glm_r2(phase, amp_s) if method == 'glm' else pac_tort_mi(phase, amp_s)
        sur.append(s)
    sur = np.array(sur)
    mu, sd = np.mean(sur), np.std(sur) + 1e-12
    return float(stat), float((stat - mu)/sd)


def pac_comodulogram_array(x: np.ndarray, fs: float,
                            phase_bands: List[Tuple[float, float]],
                            amp_bands: List[Tuple[float, float]],
                            method: str = 'mi', n_sur: int = 0, nbins: int = 18) -> Tuple[np.ndarray, Optional[np.ndarray]]:

    # Sanitize bands against Nyquist
    phase_bands = _sanitize_band_list(phase_bands, fs, label='phase')
    amp_bands   = _sanitize_band_list(amp_bands,   fs, label='amplitude')
    if not phase_bands or not amp_bands:
        raise ValueError(f"No valid bands for PAC with fs={fs:.2f} Hz. Reduce high bands or increase fs.")
    M = np.zeros((len(phase_bands), len(amp_bands)))


    M = np.zeros((len(phase_bands), len(amp_bands)))
    Z = np.zeros_like(M) if n_sur > 0 else None
    for i, pband in enumerate(phase_bands):
        for j, aband in enumerate(amp_bands):
            ph, amp = _phase_amp(x, fs, pband, aband)
            if method == 'glm':
                stat = pac_glm_r2(ph, amp); M[i,j] = stat
                if n_sur > 0: _, z = pac_surrogate_z(ph, amp, method='glm', n=n_sur); Z[i,j] = z
            else:
                mi = pac_tort_mi(ph, amp, nbins); M[i,j] = mi
                if n_sur > 0: _, z = pac_surrogate_z(ph, amp, method='mi', n=n_sur); Z[i,j] = z
    return M, Z


def pac_event_windows_records(RECORDS: pd.DataFrame, ch_name: str, windows: List[Tuple[float,float]],
                              phase_bands: List[Tuple[float,float]]=[(4,8),(8,13)],
                              amp_bands: List[Tuple[float,float]]=[(13,30),(30,80)],
                              method: str='mi', n_sur: int=0,
                              time_col: str=_DEF_TIME_COL) -> Tuple[np.ndarray, Optional[np.ndarray], List[str], List[str]]:
    fs = infer_fs_from_records(RECORDS, time_col=time_col)
    s = _find_channel_series(RECORDS, ch_name)
    if s is None:
        raise ValueError(f"Channel '{ch_name}' not found in RECORDS.")
    x = s.values.astype(float)
    t = RECORDS[time_col].values
    mats, zmaps = [], []
    for (t0, t1) in windows:
        sel = (t >= t0) & (t <= t1)
        if not np.any(sel):
            continue
        M, Z = pac_comodulogram_array(x[sel], fs, phase_bands, amp_bands, method=method, n_sur=n_sur)
        mats.append(M)
        if Z is not None: zmaps.append(Z)
    Mavg = np.nanmean(mats, axis=0) if mats else np.zeros((len(phase_bands), len(amp_bands)))
    Zavg = (np.nanmean(zmaps, axis=0) if zmaps else None)
    plabs = [f"{a}-{b} Hz" for (a,b) in phase_bands]
    alabs = [f"{a}-{b} Hz" for (a,b) in amp_bands]
    return Mavg, Zavg, plabs, alabs

# -----------------------------
# Bicoherence (records)
# -----------------------------

def bicoherence_array(x: np.ndarray, fs: float, nperseg: int=1024, noverlap: int=512, fmax: Optional[float]=None):
    ny = 0.5*fs
#   if fmax is None: fmax = fs/3
    if fmax is None or fmax > ny: fmax = ny - 1e-6
    f, t, Z = signal.stft(x, fs=fs, nperseg=nperseg, noverlap=noverlap, window='hann', padded=False, boundary=None)
    keep = f <= fmax; f = f[keep]; Z = Z[keep,:]
    F = len(f)
    Bnum = np.zeros((F,F), dtype=complex); Bden1 = np.zeros((F,F)); Bden2 = np.zeros((F,F))
    for i in range(F):
        for j in range(F-i):
            k = i + j
            prod = Z[i,:]*Z[j,:]*np.conj(Z[k,:])
            Bnum[i,j] = prod.mean()
            Bden1[i,j] = np.mean(np.abs(Z[i,:]*Z[j,:])**2)
            Bden2[i,j] = np.mean(np.abs(Z[k,:])**2)
    B = np.abs(Bnum) / (np.sqrt(Bden1*Bden2) + 1e-12)
    return B, f, f


def bicoherence_event_windows_records(RECORDS: pd.DataFrame, ch_name: str, windows: List[Tuple[float,float]],
                                      nperseg: int=1024, noverlap: int=512, fmax: Optional[float]=None,
                                      time_col: str=_DEF_TIME_COL):
    fs = infer_fs_from_records(RECORDS, time_col=time_col)
    s = _find_channel_series(RECORDS, ch_name)
    if s is None:
        raise ValueError(f"Channel '{ch_name}' not found in RECORDS.")
    x = s.values.astype(float); t = RECORDS[time_col].values

    # Enforce a single fmax for all windows (≤ Nyquist)
    ny = 0.5*fs
    if fmax is None or fmax > ny:
        fmax = ny - 1e-6

    mats = []
    f_list = []
    for (t0,t1) in windows:
        sel = (t>=t0) & (t<=t1)
        if not np.any(sel):
            continue
        B, f1, f2 = bicoherence_array(x[sel], fs, nperseg=nperseg, noverlap=noverlap, fmax=fmax)
        mats.append(B)
        f_list.append(f1)  # f1 and f2 are identical arrays here

    if not mats:
        return None, None, None

    # Crop all to the smallest shape to make averaging safe
    minF = min(m.shape[0] for m in mats)
    mats_cropped = [m[:minF, :minF] for m in mats]
    Bmean = np.nanmean(np.stack(mats_cropped, axis=0), axis=0)

    # Return a matching frequency axis
    f_common = f_list[0][:minF]
    return Bmean, f_common, f_common


# -----------------------------
# Waveform shape (records)
# -----------------------------

def waveform_shape_metrics_array(x: np.ndarray, fs: float, band: Tuple[float,float]=(4,8), neighborhood_ms: float=5.0):
    xf = _butter_bandpass(x, fs, *band)
    peaks, _ = signal.find_peaks(xf)
    troughs, _ = signal.find_peaks(-xf)
    if len(peaks) < 2 or len(troughs) < 2:
        return {'sharp_asym': np.nan, 'steep_asym': np.nan, 'rise_decay_ratio': np.nan}
    k = int(max(1, neighborhood_ms*1e-3*fs))
    def sharpness(idxs):
        vals=[]
        for i in idxs:
            i0,i1 = max(0, i-k), min(len(xf)-1, i+k)
            vals.append(xf[i] - 0.5*(xf[i0]+xf[i1]))
        return np.nanmean(vals)
    peak_sharp = sharpness(peaks)
    trough_sharp = sharpness(troughs)
    sharp_asym = (peak_sharp - abs(trough_sharp)) / (abs(peak_sharp)+abs(trough_sharp)+1e-12)
    dx = np.gradient(xf)
    rises, decays = [], []
    all_ext = np.sort(np.r_[peaks, troughs])
    for a,b in zip(all_ext[:-1], all_ext[1:]):
        seg = dx[a:b]
        if xf[b] > xf[a]:
            rises.append(np.max(seg))
        else:
            decays.append(np.min(seg))
    steep_asym = (np.nanmean(rises) - abs(np.nanmean(decays))) / (abs(np.nanmean(rises))+abs(np.nanmean(decays))+1e-12)
    trs = np.sort(troughs); cycles=[]
    for a,b in zip(trs[:-1], trs[1:]):
        pk = a + int(np.argmax(xf[a:b]))
        rise = (pk - a)/fs; decay=(b - pk)/fs
        cycles.append(rise/(decay+1e-12))
    r_d_ratio = np.nanmedian(cycles) if cycles else np.nan
    return {'sharp_asym': float(sharp_asym), 'steep_asym': float(steep_asym), 'rise_decay_ratio': float(r_d_ratio)}


def waveform_shape_event_windows_records(RECORDS: pd.DataFrame, ch_name: str, windows: List[Tuple[float,float]],
                                          band: Tuple[float,float]=(4,8), time_col: str=_DEF_TIME_COL) -> Dict[str,float]:
    fs = infer_fs_from_records(RECORDS, time_col=time_col)
    s = _find_channel_series(RECORDS, ch_name)
    if s is None:
        raise ValueError(f"Channel '{ch_name}' not found in RECORDS.")
    x = s.values.astype(float); t = RECORDS[time_col].values
    mets=[]
    for (t0,t1) in windows:
        sel=(t>=t0)&(t<=t1)
        if not np.any(sel): continue
        mets.append(waveform_shape_metrics_array(x[sel], fs, band))
    keys=['sharp_asym','steep_asym','rise_decay_ratio']
    out = {k: float(np.nanmean([m.get(k,np.nan) for m in mets])) for k in keys} if mets else {k: np.nan for k in keys}
    return out

# -----------------------------
# High-level driver for RECORDS
# -----------------------------

def run_crossfreq_suite_records(RECORDS: pd.DataFrame,
                                ignition_windows: List[Tuple[float,float]],
                                rebound_windows: Optional[List[Tuple[float,float]]]=None,
                                sensor_phase_ch: str='F4',
                                sensor_amp_chs: Tuple[str,...]=('O1','O2','P7','P8','T7','T8'),
                                phase_bands: List[Tuple[float,float]]=[(4,8),(8,13)],
                                amp_bands: List[Tuple[float,float]]=[(13,30),(30,80)],
                                method: str='mi', n_sur: int=0,
                                time_col: str=_DEF_TIME_COL) -> Dict[str, object]:
    fs = infer_fs_from_records(RECORDS, time_col=time_col)

    # Optional pre-sanitize (will also sanitize again inside PAC)
    phase_bands = _sanitize_band_list(phase_bands, fs, label='phase')
    amp_bands   = _sanitize_band_list(amp_bands,   fs, label='amplitude')


    results: Dict[str, object] = {}

    # PAC — IGNITION
    pac_ign=[]; z_ign=[]; pl=None; al=None
    for tgt in sensor_amp_chs:
        if _find_channel_series(RECORDS, tgt) is None: continue
        M, Z, pl, al = pac_event_windows_records(RECORDS, tgt, ignition_windows, phase_bands, amp_bands, method, n_sur, time_col)
        pac_ign.append(M)
        if Z is not None: z_ign.append(Z)
    if pac_ign:
        pac_ign_mean = np.nanmean(np.stack(pac_ign, axis=0), axis=0)
        results['pac_ign_mean']=pac_ign_mean; results['pac_phase_labels']=pl; results['pac_amp_labels']=al
        # quick plot
        fig, ax = plt.subplots(figsize=(6,5)); im=ax.imshow(pac_ign_mean,origin='lower',aspect='auto');
        ax.set_title('PAC (Ignition): mean over posterior sensors'); plt.colorbar(im,ax=ax); plt.tight_layout(); plt.show()
        if z_ign:
            z_ign_mean=np.nanmean(np.stack(z_ign,axis=0),axis=0); results['pac_ign_z']=z_ign_mean
            fig, ax = plt.subplots(figsize=(6,5)); im=ax.imshow(z_ign_mean,origin='lower',aspect='auto',cmap='magma');
            ax.set_title('PAC z (Ignition): surrogate z'); plt.colorbar(im,ax=ax); plt.tight_layout(); plt.show()

    # PAC — REBOUND (optional)
    if rebound_windows:
        pac_reb=[]; z_reb=[]
        for tgt in sensor_amp_chs:
            if _find_channel_series(RECORDS, tgt) is None: continue
            M, Z, _, _ = pac_event_windows_records(RECORDS, tgt, rebound_windows, phase_bands, amp_bands, method, n_sur, time_col)
            pac_reb.append(M)
            if Z is not None: z_reb.append(Z)
        if pac_reb:
            pac_reb_mean=np.nanmean(np.stack(pac_reb,axis=0),axis=0); results['pac_reb_mean']=pac_reb_mean
            fig, ax = plt.subplots(figsize=(6,5)); im=ax.imshow(pac_reb_mean,origin='lower',aspect='auto');
            ax.set_title('PAC (Rebound): mean over posterior sensors'); plt.colorbar(im,ax=ax); plt.tight_layout(); plt.show()
            if z_reb:
                z_reb_mean=np.nanmean(np.stack(z_reb,axis=0),axis=0); results['pac_reb_z']=z_reb_mean
                fig, ax = plt.subplots(figsize=(6,5)); im=ax.imshow(z_reb_mean,origin='lower',aspect='auto',cmap='magma');
                ax.set_title('PAC z (Rebound): surrogate z'); plt.colorbar(im,ax=ax); plt.tight_layout(); plt.show()

    # Bicoherence & waveform shape at first available posterior
        ch0 = next((c for c in sensor_amp_chs if _find_channel_series(RECORDS, c) is not None), None)
        if ch0 is not None:
            # choose a consistent fmax for all windows (e.g. 0.45*fs)
            fs = infer_fs_from_records(RECORDS, time_col=time_col)
            fmax_common = 0.45 * fs

            B, f1, f2 = bicoherence_event_windows_records(
                RECORDS, ch0, ignition_windows, nperseg=1024, noverlap=512, fmax=fmax_common, time_col=time_col
            )
            if B is not None:
                results['bicoherence'] = B
                results['bicoherence_f1'] = f1
                results['bicoherence_f2'] = f2
                fig, ax = plt.subplots(figsize=(6,5))
                im = ax.imshow(B, origin='lower', aspect='auto', extent=[f2[0], f2[-1], f1[0], f1[-1]])
                ax.set_title(f'Bicoherence (Ignition) @ {ch0}')
                plt.colorbar(im, ax=ax); plt.tight_layout(); plt.show()
        mets = waveform_shape_event_windows_records(RECORDS, ch0, ignition_windows, band=(4,8), time_col=time_col)
        results['waveform_metrics']=mets
        fig, ax = plt.subplots(figsize=(6,4)); ax.bar(['sharp_asym','steep_asym','rise_decay_ratio'], [mets.get('sharp_asym',np.nan), mets.get('steep_asym',np.nan), mets.get('rise_decay_ratio',np.nan)])
        ax.set_title(f'Waveform shape (θ 4–8 Hz) @ {ch0} (Ignition)'); plt.tight_layout(); plt.show()

    # Verdict summary
    verdict=[]
    if 'pac_ign_mean' in results:
        p_idx = next((i for i,(a,b) in enumerate(phase_bands) if (a,b)==(4,8)), 0)
        a_idx = next((j for j,(a,b) in enumerate(amp_bands) if a>=30), len(amp_bands)-1)
        val = results['pac_ign_mean'][p_idx,a_idx]; verdict.append(f"θ→γ PAC mean = {val:.3f}")
        if 'pac_ign_z' in results: verdict.append(f"θ→γ PAC z = {results['pac_ign_z'][p_idx,a_idx]:.2f}")
    if 'bicoherence' in results:
        v = float(np.nanpercentile(results['bicoherence'],95)); verdict.append(f"Bicoherence 95th pct = {v:.3f}")
    if 'waveform_metrics' in results:
        wa=results['waveform_metrics']; verdict.append(f"Shape: sharp={wa['sharp_asym']:.3f}, steep={wa['steep_asym']:.3f}, rise/decay={wa['rise_decay_ratio']:.3f}")
    results['verdict_notes']='; '.join(verdict)
    return results

# -----------------------------
# Example (RECORDS)
# -----------------------------
# windows_ign = [(12.0, 17.0), (33.0, 38.0)]
# windows_reb = [(18.0, 23.0), (39.0, 44.0)]
# res = run_crossfreq_suite_records(RECORDS, ignition_windows=windows_ign, rebound_windows=windows_reb,
#                                   sensor_phase_ch='F4', sensor_amp_chs=('O1','O2','P7','P8','T7','T8'),
#                                   phase_bands=[(4,8),(8,13)], amp_bands=[(13,30),(30,80)], method='mi', n_sur=0)
# print(res['verdict_notes'])

# -------------------- basic helpers --------------------

def infer_fs(RECORDS: pd.DataFrame, time_col: str = 'Timestamp') -> float:
    t = np.asarray(pd.to_numeric(RECORDS[time_col], errors='coerce').values, float)
    dt = np.diff(t); dt = dt[(dt > 0) & np.isfinite(dt)]
    if dt.size == 0:
        raise ValueError("Cannot infer sampling rate from time column.")
    return float(1.0 / np.median(dt))

def get_series(RECORDS: pd.DataFrame, name: str) -> np.ndarray:
    """Return a numeric signal array. Accepts 'EEG.O1' or bare 'O1' (will try 'EEG.O1')."""
    if name in RECORDS.columns:
        x = pd.to_numeric(RECORDS[name], errors='coerce').fillna(0.0).values
        return np.asarray(x, float)
    alt = 'EEG.' + name
    if alt in RECORDS.columns:
        x = pd.to_numeric(RECORDS[alt], errors='coerce').fillna(0.0).values
        return np.asarray(x, float)
    raise ValueError(f"Signal '{name}' not found in RECORDS.")

def bandpass(x: np.ndarray, fs: float, f1: float, f2: float, order: int = 4) -> np.ndarray:
    ny = 0.5 * fs
    f1 = max(1e-6, min(f1, ny * 0.99))
    f2 = max(f1 + 1e-6, min(f2, ny * 0.999))
    b, a = signal.butter(order, [f1 / ny, f2 / ny], btype='band')
    return signal.filtfilt(b, a, x)

def slice_epoch(x: np.ndarray, idx0: int, idx1: int) -> Optional[np.ndarray]:
    idx0 = max(0, idx0); idx1 = min(len(x), idx1)
    if idx1 <= idx0:
        return None
    return x[idx0:idx1]

# -------------------- 3a) Schumann-locked, event-related PAC --------------------

def detect_schumann_bursts(RECORDS: pd.DataFrame,
                           sr_channel: str,
                           time_col: str = 'Timestamp',
                           center_hz: float = 7.83,
                           half_bw_hz: float = 0.6,
                           smooth_sec: float = 0.25,
                           thresh_mode: str = 'z',
                           z_thresh: float = 2.5,
                           perc_thresh: float = 95.0,
                           min_isi_sec: float = 2.0
                           ) -> Dict[str, object]:
    """
    Detect Schumann bursts on a reference signal by thresholding the narrowband envelope.
    Returns {'onsets_sec': [...], 'env': env, 't': t}.
    """
    fs = infer_fs(RECORDS, time_col)
    t = np.asarray(pd.to_numeric(RECORDS[time_col], errors='coerce').values, float)
    y = get_series(RECORDS, sr_channel)
    yb = bandpass(y, fs, center_hz - half_bw_hz, center_hz + half_bw_hz)
    env = np.abs(signal.hilbert(yb))
    # smooth envelope
    n = max(1, int(round(fs * smooth_sec)))
    if n > 1:
        w = np.hanning(n) / np.sum(np.hanning(n))
        env = np.convolve(env, w, mode='same')
    # threshold
    if thresh_mode == 'z':
        z = (env - env.mean()) / (env.std() + 1e-12)
        mask = z >= z_thresh
    else:
        thr = np.percentile(env, perc_thresh)
        mask = env >= thr
    # rising edges
    on_idx = np.where(np.diff(mask.astype(int)) == 1)[0] + 1
    # enforce minimum ISI
    on = []
    last_t = -np.inf
    for i in on_idx:
        if t[i] - last_t >= min_isi_sec:
            on.append(t[i]); last_t = t[i]
    return {'onsets_sec': on, 'env': env, 't': t}

def pac_mi_phase_amp(x_phase: np.ndarray,
                     x_amp: np.ndarray,
                     nbins: int = 18) -> float:
    """Tort MI: KL divergence of phase-binned amplitude from uniform."""
    ph = np.angle(signal.hilbert(x_phase))
    am = np.abs(signal.hilbert(x_amp))
    edges = np.linspace(-np.pi, np.pi, nbins + 1)
    digit = np.digitize(ph, edges) - 1
    digit = np.clip(digit, 0, nbins - 1)
    m = np.zeros(nbins)
    for k in range(nbins):
        sel = (digit == k)
        m[k] = np.mean(am[sel]) if np.any(sel) else 0.0
    if m.sum() <= 0:
        return 0.0
    p = m / m.sum()
    eps = 1e-12
    mi = np.sum(p * np.log((p + eps) / (1.0 / nbins))) / np.log(nbins)
    return float(mi)

def epochwise_pac_timecourse(RECORDS: pd.DataFrame,
                             eeg_channels: List[str],
                             time_col: str,
                             onsets_sec: List[float],
                             win_sec: Tuple[float, float] = (-10.0, 10.0),
                             pac_phase_band: Tuple[float, float] = (4, 8),
                             pac_amp_band: Tuple[float, float] = (30, 80),
                             step_sec: float = 0.25,
                             win_pac_sec: float = 1.0,
                             nbins: int = 18) -> Dict[str, object]:
    """
    Build trial x time PAC(t) around onsets, averaged over channels.
    Sliding window (win_pac_sec) with step (step_sec).
    """
    fs = infer_fs(RECORDS, time_col)
    n_step = max(1, int(round(step_sec * fs)))
    L = int(round((win_sec[1] - win_sec[0]) * fs))
    # assemble channel matrix
    X = []
    for ch in eeg_channels:
        X.append(get_series(RECORDS, ch))
    X = np.vstack(X)  # (n_ch, N)
    N = X.shape[1]
    # time vector relative to onset for centers of windows
    centers = np.arange(int(round(win_sec[0] * fs + win_pac_sec * fs / 2)),
                        int(round(win_sec[1] * fs - win_pac_sec * fs / 2)) + 1,
                        n_step)
    t_rel = centers / fs
    # compute PAC per trial
    PAC = []  # trials x time
    keep_onsets = []
    for on in onsets_sec:
        i_on = int(round(on * fs))
        i0 = i_on + int(round(win_sec[0] * fs))
        i1 = i_on + int(round(win_sec[1] * fs))
        if i0 < 0 or i1 > N or (i1 - i0) < int(round(win_pac_sec * fs)):
            continue
        trial = []
        for c in centers:
            s = i_on + int(round(c - win_pac_sec * fs / 2))
            e = s + int(round(win_pac_sec * fs))
            if s < 0 or e > N:
                trial.append(np.nan)
                continue
            # band-limit per channel and average MI across channels
            mis = []
            for x in X:
                xp = bandpass(x[s:e], fs, pac_phase_band[0], pac_phase_band[1])
                xa = bandpass(x[s:e], fs, pac_amp_band[0], pac_amp_band[1])
                mis.append(pac_mi_phase_amp(xp, xa, nbins=nbins))
            trial.append(np.nanmean(mis))
        PAC.append(trial)
        keep_onsets.append(on)
    PAC = np.array(PAC, float)  # (n_trials, n_time)
    return {'t_rel': t_rel, 'PAC_trials': PAC, 'onsets_used': keep_onsets}

def cluster_permutation_1d(mean_tc: np.ndarray,
                           trials_tc: np.ndarray,
                           alpha: float = 0.05,
                           n_perm: int = 200,
                           rng_seed: int = 7) -> Dict[str, object]:
    """
    Simple 1D cluster-based permutation along time for ERPAC curve.
    - Observed: mean_tc (T,) from trials_tc (N,T) relative to baseline 0
    - Null: sign-flip trials randomly to build max-cluster distribution
    Returns significant mask and cluster boundaries.
    """
    rng = np.random.default_rng(rng_seed)
    T = mean_tc.size
    # Threshold = percentile of permuted means at each time (one-sided)
    null_means = []
    for _ in range(n_perm):
        signs = rng.choice([-1, 1], size=trials_tc.shape[0])
        perm = np.nanmean(signs[:, None] * trials_tc, axis=0)
        null_means.append(perm)
    null_means = np.array(null_means)
    thr = np.nanpercentile(null_means, 100 * (1 - alpha), axis=0)  # timepoint-wise threshold

    # observed clusters
    sig = mean_tc > thr
    # cluster mass = sum over contiguous sig points
    clusters = []
    start = None
    for i in range(T):
        if sig[i] and start is None:
            start = i
        elif (not sig[i]) and start is not None:
            clusters.append((start, i - 1))
            start = None
    if start is not None:
        clusters.append((start, T - 1))

    # Null cluster masses
    null_max = []
    for p in null_means:
        s = p > thr  # reuse same threshold
        maxmass = 0.0; run = 0.0
        for i in range(T):
            if s[i]:
                run += p[i]
                maxmass = max(maxmass, run)
            else:
                run = 0.0
        null_max.append(maxmass)
    thresh_mass = np.nanpercentile(null_max, 95)

    # Which observed clusters exceed mass threshold?
    sig_clusters = []
    for (a, b) in clusters:
        mass = np.nansum(mean_tc[a:b+1])
        if mass >= thresh_mass:
            sig_clusters.append((a, b))
    mask = np.zeros(T, dtype=bool)
    for (a, b) in sig_clusters:
        mask[a:b+1] = True
    return {'sig_mask': mask, 'sig_clusters': sig_clusters, 'thr_point': thr, 'thr_mass': thresh_mass}

def run_schumann_locked_erpac(RECORDS: pd.DataFrame,
                              sr_channel: str,
                              eeg_channels: List[str],
                              time_col: str = 'Timestamp',
                              detect_params: Dict = None,
                              erpac_params: Dict = None,
                              baseline_window: Tuple[float, float] = (-10.0, -2.0),
                              do_permutation: bool = True) -> Dict[str, object]:
    """
    Full ERPAC:
      1) detect Schumann bursts on sr_channel
      2) build trial x time PAC around onsets
      3) baseline-correct per trial by subtracting mean PAC in baseline_window
      4) cluster-based permutation along time (optional)
    """
    detect_params = detect_params or {}
    erpac_params = erpac_params or {}
    fs = infer_fs(RECORDS, time_col)
    # 1) detect bursts
    det = detect_schumann_bursts(RECORDS, sr_channel, time_col=time_col, **detect_params)
    onsets = det['onsets_sec']
    if len(onsets) == 0:
        raise ValueError("No Schumann bursts detected with current threshold.")

    # 2) epochwise PAC
    ep = epochwise_pac_timecourse(RECORDS, eeg_channels, time_col, onsets, **erpac_params)
    PAC = ep['PAC_trials']  # (n_trials, T)
    t_rel = ep['t_rel']

    # 3) baseline-correct per trial
    bsel = (t_rel >= baseline_window[0]) & (t_rel <= baseline_window[1])
    PAC_bc = PAC - np.nanmean(PAC[:, bsel], axis=1, keepdims=True)
    mean_tc = np.nanmean(PAC_bc, axis=0)

    out = {'t_rel': t_rel, 'PAC_trials': PAC, 'PAC_bc': PAC_bc, 'mean_tc': mean_tc, 'onsets': ep['onsets_used']}
    if do_permutation:
        perm = cluster_permutation_1d(mean_tc, PAC_bc)
        out.update({'perm': perm})
    return out

# -------------------- 3b) Cross-bicoherence / cross-bispectrum --------------------

def segment_fft(sig: np.ndarray, fs: float, nperseg: int, noverlap: int) -> np.ndarray:
    """Return STFT-like complex spectra array (n_seg, n_freq) using Hann windows."""
    step = nperseg - noverlap
    win = signal.windows.hann(nperseg, sym=False)
    n_fft = int(2 ** np.ceil(np.log2(nperseg)))
    segs = []
    for start in range(0, len(sig) - nperseg + 1, step):
        seg = sig[start:start+nperseg] * win
        S = np.fft.rfft(seg, n=n_fft)   # (n_freq,)
        segs.append(S)
    return np.array(segs), np.fft.rfftfreq(n_fft, d=1/fs)

def cross_bicoherence(RECORDS: pd.DataFrame,
                      x_sr: str,              # Schumann reference channel (for f1 ≈ 7.8 Hz)
                      y_eeg: str,             # EEG channel for gamma (f2)
                      z_eeg: Optional[str] = None,  # EEG channel for f1+f2 (default = y_eeg)
                      time_col: str = 'Timestamp',
                      f1_list: List[float] = (7.83,),    # cyclic base(s)
                      f2_min: float = 30.0, f2_max: float = 80.0, n_f2: int = 40,
                      nperseg: int = 2048, noverlap: int = 1024,
                      do_surrogate: bool = True, n_surr: int = 200, rng_seed: int = 11
                      ) -> Dict[str, object]:
    """
    Compute cross-bicoherence b_xy(f1,f2) predicting Z at f1+f2:
      b_xy(f1,f2) = E[X(f1)Y(f2)Z*(f1+f2)] / sqrt( E|X(f1)Y(f2)|^2 * E|Z(f1+f2)|^2 )
    Returns matrix over (f1_list,f2_grid) and a simple circular-shift surrogate null for y/z.
    """
    fs = infer_fs(RECORDS, time_col)
    x = get_series(RECORDS, x_sr)
    y = get_series(RECORDS, y_eeg)
    if z_eeg is None:
        z = y
    else:
        z = get_series(RECORDS, z_eeg)

    X, f = segment_fft(x, fs, nperseg, noverlap)   # (n_seg, n_freq)
    Y, _ = segment_fft(y, fs, nperseg, noverlap)
    Z, _ = segment_fft(z, fs, nperseg, noverlap)
    if X.size == 0 or Y.size == 0 or Z.size == 0:
        raise ValueError("Not enough data for the chosen nperseg/noverlap.")

    # f2 grid and indexing helpers
    f2_grid = np.linspace(f2_min, f2_max, n_f2)
    def idx_of(freq):
        return int(np.argmin(np.abs(f - freq)))

    B = np.zeros((len(f1_list), n_f2), float)
    for i, f1 in enumerate(f1_list):
        i1 = idx_of(f1)
        for j, f2 in enumerate(f2_grid):
            i2 = idx_of(f2)
            i12 = idx_of(f1 + f2)
            num = np.mean(X[:, i1] * Y[:, i2] * np.conj(Z[:, i12]))
            den = np.sqrt(np.mean(np.abs(X[:, i1] * Y[:, i2])**2) * np.mean(np.abs(Z[:, i12])**2) + 1e-24)
            B[i, j] = np.abs(num) / (den + 1e-24)

    out = {'f1_list': np.array(f1_list), 'f2_grid': f2_grid, 'bicoherence': B, 'freqs': f}
    # Simple surrogate: circularly shift Y (gamma) segments per realization
    if do_surrogate:
        rng = np.random.default_rng(rng_seed)
        null_max = []
        for _ in range(n_surr):
            # circular shift each epoch spectrum by a random small amount in time domain
            # (approximate by reordering epochs; simpler robust null)
            Y_perm = Y.copy()
#             rng.shuffle(Y_perm, axis=0)
            Y_perm = Y_perm[rng.permutation(Y_perm.shape[0]), :]
            B0 = np.zeros_like(B)
            for i, f1 in enumerate(f1_list):
                i1 = idx_of(f1)
                for j, f2 in enumerate(f2_grid):
                    i2 = idx_of(f2); i12 = idx_of(f1 + f2)
                    num = np.mean(X[:, i1] * Y_perm[:, i2] * np.conj(Z[:, i12]))
                    den = np.sqrt(np.mean(np.abs(X[:, i1] * Y_perm[:, i2])**2) * np.mean(np.abs(Z[:, i12])**2) + 1e-24)
                    B0[i, j] = np.abs(num) / (den + 1e-24)
            null_max.append(np.nanmax(B0))
        out['null_thresh95'] = float(np.nanpercentile(null_max, 95))
    return out

def plot_bicoherence(out: Dict[str, object], i_f1: int = 0, title: Optional[str] = None) -> None:
    """Heatmap of cross-bicoherence at a fixed f1 index across f2_grid."""
    f2 = out['f2_grid']; B = out['bicoherence']; f1_list = out['f1_list']
    plt.figure(figsize=(7, 3))
    plt.plot(f2, B[i_f1], lw=1.8)
    if 'null_thresh95' in out:
        plt.axhline(out['null_thresh95'], color='k', ls='--', lw=1, label='null 95%')
    plt.xlabel('f2 (Hz, EEG γ)'); plt.ylabel('cross-bicoherence |b_xy|')
    plt.title(title or f'Cross-bicoherence at f1={f1_list[i_f1]:.2f} Hz')
    plt.grid(alpha=0.2)
    if 'null_thresh95' in out:
        plt.legend()
    plt.tight_layout(); plt.show()
