"""
Toroidal Phase–Torus Analysis (fs=128)
-------------------------------------
Detect and quantify toroidal phase structure (S1×S1) from two reference bands
(e.g., theta & alpha phases), compare ignition vs baseline, and visualize the
phase torus. Includes null via phase-scrambling.

Key ideas
- Build reference signal (mean of chosen electrodes), band-limit to band1 & band2,
  take analytic phases φ1(t), φ2(t).
- Torus metrics per state:
   • Circular mean resultant lengths R1,R2 (uniform → low R, high circular variance)
   • Circular–circular correlation ρ_c (Jammalamadaka–Sarma) — low if independent
   • Joint phase occupancy entropy H2D on (φ1,φ2) grid (high when torus is well-filled)
   • Winding numbers W1,W2 = Δunwrap(φ1,φ2)/(2π), and ratio r=W2/W1
   • Intrinsic dimension (ID) on embedding E=[cosφ1,sinφ1,cosφ2,sinφ2] via TwoNN
     (torus ≈ 2D manifold → ID~2)
- Controls: Fourier phase-randomized nulls for the reference signal.
- Plots: φ1×φ2 heatmap (phase torus occupancy), 3D torus scatter (optional radius∝amp).

Usage
-----
res = run_toroidal_phase_analysis(
    RECORDS,
    time_col='Timestamp',
    ignition_windows=[(290,310),(580,600)],
    rebound_windows=[(310,325)],
    ref_electrodes=['O1','O2'],
    band1=(4,8), band2=(8,13),
    amp_band=(30,80),             # optional for 3D radius (gamma envelope)
    n_bins=36, n_surr=200,
    show=True
)

print(res['delta_table'])         # ignition − baseline for key torus metrics
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from scipy import signal

# ----------------- helpers -----------------

def _get_fs(RECORDS: pd.DataFrame, time_col: str) -> float:
    if 'infer_fs_from_records' in globals():
        try:
            return float(infer_fs_from_records(RECORDS, time_col=time_col))
        except Exception:
            pass
    t = np.asarray(RECORDS[time_col].values, float)
    dt = np.diff(t); dt = dt[np.isfinite(dt) & (dt>0)]
    if dt.size==0: raise ValueError('Cannot infer fs')
    return 1.0/np.median(dt)


def _bandpass(x: np.ndarray, fs: float, f1: float, f2: float, order: int=4) -> np.ndarray:
    ny = 0.5*fs
    f1 = max(1e-6, min(f1, ny*0.99)); f2 = max(f1+1e-6, min(f2, ny*0.999))
    b,a = signal.butter(order, [f1/ny, f2/ny], btype='band')
    return signal.filtfilt(b,a,x)


def _mean_reference(RECORDS: pd.DataFrame, time_col: str, electrodes: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    series=[]
    for ch in electrodes:
        s = find_channel_series(RECORDS, ch)
        if s is None: continue
        series.append(np.asarray(s.values, float))
    if not series:
        raise ValueError('No reference electrodes found in RECORDS')
    X = np.vstack(series)
    ref = np.nanmean(X, axis=0)
    t = np.asarray(RECORDS[time_col].values, float)
    return t, ref

def find_channel_series(records: pd.DataFrame, ch_name: str) -> Optional[pd.Series]:
    # ---------------- Basics: fs + channel access ----------------
    _DEF_TIME_COL = 'Timestamp'
    _DEF_CH_PATTERNS = ("EEG.{ch}", "eeg.{ch}", "{ch}", "RAW.{ch}", "CHAN.{ch}")
    for pat in _DEF_CH_PATTERNS:
        col = pat.format(ch=ch_name)
        if col in records.columns:
            return pd.to_numeric(records[col], errors='coerce').astype(float)
    return None

# circular metrics

def _circ_R(phi: np.ndarray) -> float:
    return float(np.hypot(np.mean(np.cos(phi)), np.mean(np.sin(phi))))


def _circ_corr(phi1: np.ndarray, phi2: np.ndarray) -> float:
    # Jammalamadaka–Sarma circular correlation
    mu1 = np.angle(np.mean(np.exp(1j*phi1)))
    mu2 = np.angle(np.mean(np.exp(1j*phi2)))
    s1 = np.sin(phi1 - mu1); s2 = np.sin(phi2 - mu2)
    num = np.mean(s1*s2)
    den = np.sqrt(np.mean(s1**2)*np.mean(s2**2)) + 1e-12
    return float(num/den)


def _two_nn_id(pts: np.ndarray) -> float:
    """Levina–Bickel TwoNN intrinsic dimension estimate on points (N×d)."""
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=3).fit(pts)
    dists, _ = nbrs.kneighbors(pts)
    r1 = dists[:,1]; r2 = dists[:,2]
    ratio = r2/(r1+1e-12)
    m = np.mean(np.log(ratio + 1e-12))
    return float(1.0/(m+1e-12))

# occupancy entropy

def _joint_phase_entropy(phi1: np.ndarray, phi2: np.ndarray, n_bins: int=36) -> float:
    H, xedges, yedges = np.histogram2d(phi1, phi2, bins=n_bins, range=[[-np.pi, np.pi],[-np.pi, np.pi]], density=False)
    P = H / (np.sum(H) + 1e-12)
    nz = P[P>0]
    H2 = -np.sum(nz*np.log(nz)) / np.log(n_bins*n_bins)
    return float(H2)

# winding numbers

def _winding(phi: np.ndarray) -> float:
    uw = np.unwrap(phi)
    return float((uw[-1]-uw[0])/(2*np.pi))

# surrogate

def _phase_scramble(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    X = np.fft.rfft(x)
    mag = np.abs(X)
    k = X.size
    ph = rng.uniform(-np.pi, np.pi, size=k)
    ph[0] = np.angle(X[0])
    if k % 2 == 0:
        ph[-1] = np.angle(X[-1])
    Xs = mag * np.exp(1j*ph)
    return np.fft.irfft(Xs, n=x.size).astype(float)

# ----------------- core analysis -----------------

def _compute_torus_metrics(sig: np.ndarray, fs: float, band1: Tuple[float,float], band2: Tuple[float,float],
                           amp_band: Optional[Tuple[float,float]] = None, n_bins: int = 36,
                           show: bool = True, title: str = '') -> Dict[str, object]:
    # phases
    x1 = _bandpass(sig, fs, band1[0], band1[1]); z1 = signal.hilbert(x1); phi1 = np.angle(z1)
    x2 = _bandpass(sig, fs, band2[0], band2[1]); z2 = signal.hilbert(x2); phi2 = np.angle(z2)
    # amplitude for optional 3D radius
    r = None
    if amp_band is not None:
        xa = _bandpass(sig, fs, amp_band[0], amp_band[1]); ra = np.abs(signal.hilbert(xa))
        r = (ra - np.nanmin(ra)) / (np.nanmax(ra) - np.nanmin(ra) + 1e-12)

    # metrics
    R1 = _circ_R(phi1); R2 = _circ_R(phi2)                 # low if uniform
    rho = _circ_corr(phi1, phi2)                           # ~0 if independent (torus fill)
    H2 = _joint_phase_entropy(phi1, phi2, n_bins=n_bins)   # high if well-filled torus
    W1 = _winding(phi1); W2 = _winding(phi2); ratio = W2/(W1+1e-12)
    E = np.column_stack([np.cos(phi1), np.sin(phi1), np.cos(phi2), np.sin(phi2)])
    try:
        ID = _two_nn_id(E)
    except Exception:
        ID = np.nan

    if show:
        # 2D occupancy
        plt.figure(figsize=(4.8,4))
        plt.hist2d(phi1, phi2, bins=n_bins, range=[[-np.pi, np.pi],[-np.pi, np.pi]], cmap='magma')
        plt.xlabel('φ₁ (rad)'); plt.ylabel('φ₂ (rad)'); plt.title(f'Phase torus occupancy {title}'); plt.tight_layout(); plt.show()
        # 3D torus-like embedding
        from mpl_toolkits.mplot3d import Axes3D  # noqa
        R0 = 1.0; r0 = 0.25 if r is None else 0.15 + 0.25*r
        # map torus coords: big circle φ1, small circle φ2
        X = (R0 + r0*np.cos(phi2)) * np.cos(phi1)
        Y = (R0 + r0*np.cos(phi2)) * np.sin(phi1)
        Z = r0*np.sin(phi2)
        fig = plt.figure(figsize=(4.8,4.2))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X, Y, Z, s=2, alpha=0.6)
        ax.set_title(f'3D torus embedding {title}')
        plt.tight_layout(); plt.show()

    return {'R1':R1,'R2':R2,'rho':rho,'H2':H2,'W1':W1,'W2':W2,'ratio':ratio,'ID':ID}


def run_toroidal_phase_analysis(
    RECORDS: pd.DataFrame,
    time_col: str = 'Timestamp',
    ignition_windows: List[Tuple[float,float]] = None,
    rebound_windows: Optional[List[Tuple[float,float]]] = None,
    ref_electrodes: Optional[List[str]] = None,
    band1: Tuple[float,float] = (4,8),
    band2: Tuple[float,float] = (8,13),
    amp_band: Optional[Tuple[float,float]] = (30,80),
    n_bins: int = 36,
    n_surr: int = 200,
    show: bool = True,
) -> Dict[str, object]:
    fs = _get_fs(RECORDS, time_col)
    ref_electrodes = ref_electrodes or [e for e in ['O1','O2','Oz','Pz'] if ('EEG.'+e) in RECORDS.columns] or [c.split('.',1)[1] for c in RECORDS.columns if c.startswith('EEG.')][:1]
    t, ref = _mean_reference(RECORDS, time_col, ref_electrodes)

    # concatenate ignition windows; baseline = complement
    mask_base = np.ones_like(ref, dtype=bool)
    ign_segs = []
    for (t0,t1) in ignition_windows or []:
        i0,i1 = int(t0*fs), int(t1*fs)
        i0=max(0,i0); i1=min(len(ref),i1)
        ign_segs.append(ref[i0:i1])
        mask_base[i0:i1] = False
    base_seg = ref[mask_base]

    # compute metrics for ignition (concat) and baseline
    ref_ign = np.concatenate(ign_segs) if len(ign_segs)>0 else ref
    m_ign = _compute_torus_metrics(ref_ign, fs, band1, band2, amp_band=amp_band, n_bins=n_bins, show=show, title='(ignition)')
    m_base= _compute_torus_metrics(base_seg, fs, band1, band2, amp_band=amp_band, n_bins=n_bins, show=show, title='(baseline)')

    # surrogates on ignition for nulls
    rng = np.random.default_rng(23)
    surr = []
    for _ in range(n_surr):
        xs = _phase_scramble(ref_ign, rng)
        surr.append(_compute_torus_metrics(xs, fs, band1, band2, amp_band=None, n_bins=n_bins, show=False))
    surr_df = pd.DataFrame(surr)

    delta = {
        'd_H2': float(m_ign['H2'] - m_base['H2']),
        'd_rho': float(m_ign['rho'] - m_base['rho']),      # expect negative (more independence) if torus fills
        'd_ID': float(m_ign['ID'] - m_base['ID']),         # expect → + (toward ~2)
        'd_R1': float(m_ign['R1'] - m_base['R1']),         # expect negative (more uniform)
        'd_R2': float(m_ign['R2'] - m_base['R2'])
    }
    delta_table = pd.DataFrame([delta])

    return {
        'ignition': m_ign,
        'baseline': m_base,
        'delta_table': delta_table,
        'surrogates': surr_df,
        'params': {'ref_electrodes':ref_electrodes,'band1':band1,'band2':band2,'n_bins':n_bins}
    }
