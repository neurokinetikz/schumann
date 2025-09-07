"""
Connectome Harmonics Engagement Breadth (fs=128)
-----------------------------------------------
Hypothesis: Conscious expansion recruits a broader spectrum of connectome harmonics.

This module projects sensor-space EEG onto a provided set of **harmonic spatial modes**
(e.g., connectome Laplacian eigenvectors) and compares the **mode-power distributions**
between ignition and baseline (and optional rebound). It returns:

- Spectral entropy across modes (H)
- Participation ratio (PR)
- Tail heaviness metrics (e.g., top-decile mass, optional power-law tail fit)
- Surrogate controls via per-channel Fourier phase randomization

Inputs you supply
- `H`: (n_elec × n_modes) matrix of spatial harmonics in **sensor space** aligned to your
  electrode names (rows correspond to EEG.<name> order you use here). If your harmonics
  are in parcel/source space, first map sensors→parcels (leadfield/inverse) to build H.
- `orthonormal`: True if columns of H are orthonormal in the sensor metric; if False, we
  use a ridge-regularized least-squares projection.

Key metrics
- **Spectral entropy** H = −Σ p_k log p_k / log(K), where p_k = P_k / Σ P_k
- **Participation ratio** PR = (Σ P_k)^2 / Σ (P_k^2)
- **Tail heaviness**: fraction of power in top decile modes; optional Pareto tail fit

Usage
-----
H = np.load('harmonics_sensor_space.npy')  # (n_elec × n_modes), rows match your electrodes

res = run_connectome_harmonics_breadth(
    RECORDS,
    H=H,
    electrodes=['F4','O1','O2', ...],      # MUST align to H rows
    ignition_windows=[(120,150)],
    rebound_windows=[(300,330)],
    time_col='Timestamp',
    orthonormal=True,
    do_surrogate=True, n_surr=200,
)

print(res['delta_table'])
plot_harmonics_power_spectra(res['spectra'])
plot_harmonics_breadth_deltas(res['delta_table'])
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from scipy import signal
from numpy.linalg import lstsq

# ----------------- helpers -----------------

def _get_fs(RECORDS: pd.DataFrame, time_col: str) -> float:
    if 'infer_fs_from_records' in globals():
        try:
            return float(infer_fs_from_records(RECORDS, time_col=time_col))
        except Exception:
            pass
    t = np.asarray(RECORDS[time_col].values, dtype=float)
    dt = np.diff(t); dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0: raise ValueError('Cannot infer fs')
    return 1.0 / np.median(dt)


def _slice_blocks(RECORDS: pd.DataFrame, time_col: str, X: np.ndarray, fs: float, winlist: List[Tuple[float,float]]) -> List[np.ndarray]:
    t = np.asarray(RECORDS[time_col].values, dtype=float)
    blocks = []
    for (t0,t1) in winlist or []:
        sel = (t>=t0) & (t<=t1)
        idx = np.where(sel)[0]
        if idx.size < fs*1.0:  # at least 1 s
            continue
        blocks.append(X[:, idx[0]:idx[-1]+1])
    return blocks


def _fourier_phase_randomize(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    X = np.fft.rfft(x)
    mag = np.abs(X); ph = np.angle(X)
    k = ph.size; rand = rng.uniform(-np.pi, np.pi, size=k)
    rand[0] = ph[0]
    if k % 2 == 0: rand[-1] = ph[-1]
    Xs = mag * np.exp(1j*rand)
    xs = np.fft.irfft(Xs, n=x.size)
    return xs.astype(float)


def _make_surrogate_block(data: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = np.zeros_like(data)
    for i in range(data.shape[0]):
        out[i] = _fourier_phase_randomize(data[i], rng)
    return out

def find_channel_series(records: pd.DataFrame, ch_name: str) -> Optional[pd.Series]:
    # ---------------- Basics: fs + channel access ----------------
    _DEF_TIME_COL = 'Timestamp'
    _DEF_CH_PATTERNS = ("EEG.{ch}", "eeg.{ch}", "{ch}", "RAW.{ch}", "CHAN.{ch}")
    for pat in _DEF_CH_PATTERNS:
        col = pat.format(ch=ch_name)
        if col in records.columns:
            return pd.to_numeric(records[col], errors='coerce').astype(float)
    return None

# ----------------- projections -----------------

def project_to_harmonics(X: np.ndarray, H: np.ndarray, orthonormal: bool=True, ridge: float=1e-3) -> np.ndarray:
    """Project sensor data X (n_elec × n_times) onto spatial harmonics H (n_elec × n_modes).
    Returns A (n_modes × n_times) of mode coefficients.
    If orthonormal=True, uses A = H^T X; else ridge-regularized least-squares.
    """
    if orthonormal:
        return H.T.dot(X)
    # ridge LS: (H^T H + λI)^{-1} H^T X
    HT = H.T
    G = HT.dot(H)
    G.flat[::G.shape[0]+1] += ridge
    A = np.linalg.solve(G, HT.dot(X))
    return A


def mode_power(A: np.ndarray) -> np.ndarray:
    """Return per-mode average power across time: P_k = mean_t A_k(t)^2."""
    return np.mean(A**2, axis=1)

# ----------------- metrics -----------------

def spectral_entropy(P: np.ndarray) -> float:
    P = np.asarray(P, float); P = np.clip(P, 0, None)
    s = P.sum();
    if s <= 0: return np.nan
    p = P / s
    H = -np.sum(p * np.log(p + 1e-12)) / np.log(len(p))
    return float(H)


def participation_ratio(P: np.ndarray) -> float:
    P = np.asarray(P, float); s = P.sum()
    if s <= 0: return np.nan
    return float((s**2) / (np.sum(P**2) + 1e-12))


def top_decile_mass(P: np.ndarray) -> float:
    P = np.asarray(P, float)
    K = P.size; k = max(1, int(np.ceil(0.1*K)))
    idx = np.argsort(P)[-k:]
    return float(np.sum(P[idx]) / (np.sum(P) + 1e-12))

# ----------------- orchestration -----------------

def run_connectome_harmonics_breadth(
    RECORDS: pd.DataFrame,
    H: np.ndarray,
    electrodes: List[str],           # order matches rows of H
    ignition_windows: List[Tuple[float,float]],
    rebound_windows: Optional[List[Tuple[float,float]]] = None,
    control_windows: Optional[List[Tuple[float,float]]] = None,
    time_col: str = 'Timestamp',
    orthonormal: bool = True,
    do_surrogate: bool = True,
    n_surr: int = 200,
    rng_seed: int = 23,
) -> Dict[str, object]:
    fs = _get_fs(RECORDS, time_col)
    # Build sensor matrix X in the same electrode order as H rows
    series = []
    for ch in electrodes:
        s = find_channel_series(RECORDS, ch)
        if s is None:
            raise ValueError(f'Missing channel {ch} in RECORDS')
        series.append(np.asarray(s.values, dtype=float))
    X = np.vstack(series)  # (n_elec × n_times)

    # Slice windows
    ign_blocks = _slice_blocks(RECORDS, time_col, X, fs, ignition_windows)
    # Baseline complement
    t = np.asarray(RECORDS[time_col].values, dtype=float)
    mask = np.ones(X.shape[1], dtype=bool)
    for (t0,t1) in ignition_windows:
        i0,i1 = int(t0*fs), int(t1*fs)
        mask[max(0,i0):min(X.shape[1],i1)] = False
    base_blocks = [X[:, mask]] if np.any(mask) else []
    reb_blocks  = _slice_blocks(RECORDS, time_col, X, fs, rebound_windows) if rebound_windows else []
    ctrl_blocks = _slice_blocks(RECORDS, time_col, X, fs, control_windows) if control_windows else []

    def analyze_state(blocks: List[np.ndarray]) -> Dict[str, object]:
        if not blocks:
            return {'P': np.array([]), 'H': np.nan, 'PR': np.nan, 'Top10': np.nan}
        # concat
        concat = np.hstack(blocks)
        A = project_to_harmonics(concat, H, orthonormal=orthonormal)
        P = mode_power(A)
        return {
            'P': P,
            'H': spectral_entropy(P),
            'PR': participation_ratio(P),
            'Top10': top_decile_mass(P),
        }

    ign = analyze_state(ign_blocks)
    base= analyze_state(base_blocks)
    reb = analyze_state(reb_blocks)
    ctrl= analyze_state(ctrl_blocks)

    # Surrogates
    surr = None
    if do_surrogate:
        rng = np.random.default_rng(rng_seed)
        sh = []
        for _ in range(n_surr):
            Xs = _make_surrogate_block(X, rng)
            # use same baseline mask to approximate null
            Ab = project_to_harmonics(Xs[:, mask], H, orthonormal=orthonormal)
            Pb = mode_power(Ab)
            sh.append({'H': spectral_entropy(Pb), 'PR': participation_ratio(Pb), 'Top10': top_decile_mass(Pb)})
        surr = pd.DataFrame(sh)

    # Delta table
    delta_table = pd.DataFrame([{
        'd_H': float(ign['H'] - base['H']),
        'd_PR': float(ign['PR'] - base['PR']),
        'd_Top10': float(ign['Top10'] - base['Top10'])
    }])

    spectra = {
        'ignition': ign['P'],
        'baseline': base['P'],
        'rebound' : reb['P'],
        'control' : ctrl['P'],
    }

    return {
        'delta_table': delta_table,
        'metrics': {
            'ignition': {'H':ign['H'],'PR':ign['PR'],'Top10':ign['Top10']},
            'baseline': {'H':base['H'],'PR':base['PR'],'Top10':base['Top10']},
            'rebound' : {'H':reb['H'],'PR':reb['PR'],'Top10':reb['Top10']},
            'control' : {'H':ctrl['H'],'PR':ctrl['PR'],'Top10':ctrl['Top10']},
        },
        'spectra': spectra,
        'surrogates': surr,
        'params': {
            'electrodes': electrodes,
            'orthonormal': orthonormal,
            'n_surr': n_surr,
        }
    }

# ----------------- plots -----------------

def plot_harmonics_power_spectra(spectra: Dict[str,np.ndarray], top_k: int = 60) -> None:
    plt.figure(figsize=(10,4))
    for k,P in spectra.items():
        if P is None or len(P)==0: continue
        idx = np.argsort(P)[::-1][:top_k]
        plt.plot(np.arange(1, len(idx)+1), P[idx]/(P.sum()+1e-12), label=k)
    plt.xlabel('Mode rank (power-desc)'); plt.ylabel('Relative power');
    plt.title('Connectome harmonic power spectra (top modes)'); plt.legend(); plt.tight_layout(); plt.show()


def plot_harmonics_breadth_deltas(delta_df: pd.DataFrame) -> None:
    plt.figure(figsize=(6,3.2))
    labels = ['ΔH (entropy)','ΔPR (participation)','ΔTop10 (tail mass)']
    vals = [float(delta_df['d_H'].values[0]), float(delta_df['d_PR'].values[0]), float(delta_df['d_Top10'].values[0])]
    x = np.arange(len(vals)); plt.bar(x, vals, width=0.6)
    plt.xticks(x, labels, rotation=0); plt.ylabel('Ign − Base');
    plt.title('Harmonics engagement breadth — deltas'); plt.tight_layout(); plt.show()

"""
Functional Harmonics Builder from Baseline (fs=128)
--------------------------------------------------
When a connectome harmonics matrix `H` (sensor space) is not available,
this builds a **functional harmonics** basis from your *baseline* EEG by:

1) Band-limiting (default 4–40 Hz)
2) Estimating **pseudo-wPLI** adjacency across electrodes
3) Computing the graph **Laplacian eigenvectors** (low→high) as spatial modes

Returns an orthonormal H (n_elec × n_modes) aligned to the electrode order you supply.
You can save it and reuse:
    np.save('harmonics_sensor_space.npy', H)

Usage
-----
ELECTRODES = ['F4','O1','O2', ...]  # order you’ll use in the breadth analysis (must match H rows)
H = build_functional_harmonics_from_baseline(
    RECORDS,
    electrodes=ELECTRODES,
    ignition_windows=[(120,150)],
    time_col='Timestamp',
    fband=(4,40),
    n_modes=64
)
np.save('harmonics_sensor_space.npy', H)

# Then run the breadth analysis with the generated H
res = run_connectome_harmonics_breadth(
    RECORDS,
    H=H,
    electrodes=ELECTRODES,
    ignition_windows=[(120,150)],
    rebound_windows=[(300,330)],
    time_col='Timestamp',
    orthonormal=True,
    do_surrogate=True, n_surr=200
)
print(res['delta_table'])
"""

# -------- helpers --------

def _get_fs(RECORDS: pd.DataFrame, time_col: str) -> float:
    if 'infer_fs_from_records' in globals():
        try:
            return float(infer_fs_from_records(RECORDS, time_col=time_col))
        except Exception:
            pass
    t = np.asarray(RECORDS[time_col].values, dtype=float)
    dt = np.diff(t); dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0: raise ValueError('Cannot infer fs')
    return 1.0 / np.median(dt)


def _slice_baseline_mask(RECORDS: pd.DataFrame, time_col: str, fs: float, n_cols: int,
                         ignition_windows: List[Tuple[float,float]]) -> np.ndarray:
    t = np.asarray(RECORDS[time_col].values, dtype=float)
    mask = np.ones(n_cols, dtype=bool)
    for (t0,t1) in ignition_windows or []:
        i0,i1 = int(t0*fs), int(t1*fs)
        mask[max(0,i0):min(n_cols,i1)] = False
    return mask


def _bandpass(X: np.ndarray, fs: float, f1: float, f2: float) -> np.ndarray:
    ny = 0.5*fs
    f1 = max(1e-6, min(f1, ny*0.99))
    f2 = max(f1+1e-6, min(f2, ny*0.999))
    b,a = signal.butter(4, [f1/ny, f2/ny], btype='band')
    return signal.filtfilt(b,a,X,axis=1)


def _pseudo_wpli(Xb: np.ndarray) -> np.ndarray:
    """Pseudo-wPLI (Hilbert analytic) on band-limited data.
    Xb: (n_ch, n_times)
    """
    Z = signal.hilbert(Xb, axis=1)
    n = Xb.shape[0]
    A = np.zeros((n,n), dtype=float)
    for i in range(n):
        zi = Z[i]
        for j in range(i+1, n):
            im = np.imag(zi * np.conj(Z[j]))
            num = np.abs(np.mean(im))
            den = np.mean(np.abs(im)) + 1e-12
            w = num/den
            A[i,j] = A[j,i] = w
    np.fill_diagonal(A, 0.0)
    return A


def build_functional_harmonics_from_baseline(
    RECORDS: pd.DataFrame,
    electrodes: List[str],
    ignition_windows: Optional[List[Tuple[float,float]]] = None,
    time_col: str = 'Timestamp',
    fband: Tuple[float,float] = (4,40),
    n_modes: int = 64,
) -> np.ndarray:
    """Build sensor-space functional harmonics from baseline data.
    Returns H (n_elec × K) with orthonormal columns (graph Laplacian eigenvectors).
    """
    fs = _get_fs(RECORDS, time_col)
    # Build data matrix X in electrode order
    series = []
    for ch in electrodes:
        s = find_channel_series(RECORDS, ch)
        if s is None:
            raise ValueError(f'Missing channel {ch} in RECORDS')
        series.append(np.asarray(s.values, dtype=float))
    X = np.vstack(series)  # (n_elec × n_times)

    # Baseline mask (complement of ignition windows)
    mask = _slice_baseline_mask(RECORDS, time_col, fs, X.shape[1], ignition_windows)
    Xb = X[:, mask]

    # Band-limit and compute pseudo-wPLI adjacency
    Xf = _bandpass(Xb, fs, fband[0], fband[1])
    A = _pseudo_wpli(Xf)

    # Graph Laplacian eigenvectors
    D = np.diag(A.sum(axis=1))
    L = D - A
    # ensure symmetric positive semidefinite
    L = 0.5*(L + L.T)
    evals, evecs = np.linalg.eigh(L)
    # sort ascending by eigenvalue (low-frequency first)
    idx = np.argsort(evals)
    evecs = evecs[:, idx]
    K = min(n_modes, evecs.shape[1])
    H = evecs[:, :K]
    # columns are orthonormal for symmetric L; normalize just in case
    for k in range(H.shape[1]):
        H[:,k] /= (np.linalg.norm(H[:,k]) + 1e-12)
    return H
