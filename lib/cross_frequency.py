import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
from scipy import signal

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
