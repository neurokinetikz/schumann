"""
Cross-Frequency & Cross-Region Coupling — Simple Graphs & Validation
====================================================================

What this module does
---------------------
1) Cross-channel PAC (phase→amplitude): slow phase in chA modulates fast amplitude in chB.
   • Comodulogram over f_slow ∈ [4..12] Hz and f_fast ∈ [30..80] Hz.
   • Surrogate null (circular shift of the **fast** channel) → cell-wise 95% threshold.
2) n:m Phase locking (PLV_{n:m}): |<exp(i*(n*phi1 − m*phi2))>|
   • Grid over f1 ∈ [4..15] Hz, f2 ∈ [20..80] Hz, ratios m∈{2..6}, n=1 (default), cross-channel.
   • Surrogate null (shift one phase) → significance map.
3) Summaries: top PAC pairs (chA→chB, f1*, f2*, MI, p), top n:m pairs (ch1↔ch2, f1*, f2*, m, PLV, p).
4) Optional Ignition vs Baseline comparison (re-run on each window set).

Outputs
-------
• PNGs: comodulograms with significant cells (cyan), n:m PLV maps, pairwise summaries.
• CSV: cfc_summary.csv (top finds & p-values) in out_dir.

Usage
-----
res = run_cfc_cross_region(
    RECORDS,
    pairs=[('EEG.F3','EEG.P8'), ('EEG.F4','EEG.P7')],  # phase@F → amp@P examples
    ignition_windows=[(290,310),(580,600)],
    baseline_windows=[(0,290),(325,580)],
    time_col='Timestamp',
    out_dir='exports_cfc/S01',
    show=False
)
print(res['summary'])
"""
from __future__ import annotations
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from scipy import signal
import networkx as nx

# ---------------- I/O & time helpers ----------------
def _ensure_dir(d): os.makedirs(d, exist_ok=True); return d

def detect_time_col(df,
    candidates=('Timestamp','Time','time','t','seconds','sec','ms','datetime','DateTime','Datetime')) -> Optional[str]:
    for c in candidates:
        if c in df.columns: return c
    # numeric, roughly monotonic
    for c in df.columns:
        s = pd.to_numeric(df[c], errors='coerce')
        if s.notna().sum()>max(50,0.5*len(df)):
            x=s.values.astype(float); dt=np.diff(x[np.isfinite(x)])
            if dt.size and np.nanmedian(dt)>0: return c
    # datetime
    for c in df.columns:
        try:
            _ = pd.to_datetime(df[c], errors='raise'); return c
        except Exception: pass
    return None

def ensure_timestamp_column(df: pd.DataFrame, time_col: Optional[str]=None, default_fs: float = 128.0, out_name='Timestamp')->str:
    col = time_col or detect_time_col(df)
    if col is None:
        df[out_name] = np.arange(len(df), dtype=float)/default_fs; return out_name
    s = df[col]
    if np.issubdtype(s.dtype, np.datetime64) or 'datetime' in str(s.dtype).lower():
        tsec=(pd.to_datetime(s)-pd.to_datetime(s).iloc[0]).dt.total_seconds().astype(float)
        df[out_name]=tsec.values; return out_name
    sn = pd.to_numeric(s, errors='coerce').astype(float)
    if sn.notna().sum()<max(50,0.5*len(df)):
        df[out_name]=np.arange(len(df), dtype=float)/default_fs; return out_name
    sn = sn - np.nanmin(sn[np.isfinite(sn)]); df[out_name]=sn.values; return out_name

def infer_fs(df: pd.DataFrame, time_col: str)->float:
    t = np.asarray(pd.to_numeric(df[time_col], errors='coerce').values, float)
    dt=np.diff(t); dt=dt[(dt>0)&np.isfinite(dt)]
    if dt.size==0: raise ValueError("Cannot infer fs."); return 1.0
    return float(1.0/np.median(dt))

def get_series(df: pd.DataFrame, name: str)->np.ndarray:
    if name in df.columns:
        return pd.to_numeric(df[name], errors='coerce').fillna(0.0).values.astype(float)
    alt='EEG.'+name
    if alt in df.columns:
        return pd.to_numeric(df[alt], errors='coerce').fillna(0.0).values.astype(float)
    raise ValueError(f"{name} not found.")

def slice_concat(x: np.ndarray, fs: float, wins: Optional[List[Tuple[float,float]]])->np.ndarray:
    if not wins: return x.copy()
    segs=[]; n=len(x)
    for (a,b) in wins:
        i0,i1=int(round(a*fs)), int(round(b*fs))
        i0=max(0,i0); i1=min(n,i1)
        if i1>i0: segs.append(x[i0:i1])
    return np.concatenate(segs) if segs else x.copy()

def zscore(x): x=np.asarray(x,float); return (x-np.mean(x))/(np.std(x)+1e-12)

# ---------------- Filtering & analytic ----------------
def bandpass(x, fs, f1, f2, order=4):
    ny=0.5*fs; f1=max(1e-6,min(f1,0.99*ny)); f2=max(f1+1e-6,min(f2,0.999*ny))
    b,a=signal.butter(order,[f1/ny,f2/ny],btype='band'); return signal.filtfilt(b,a,x)

def analytic_phase_amp(x, fs, f1, f2):
    xb = bandpass(x, fs, f1, f2)
    z  = signal.hilbert(xb)
    return np.angle(z), np.abs(z)

# ---------------- PAC (Tort MI) ----------------
def pac_mi_phase_amp(phase: np.ndarray, amp: np.ndarray, nbins: int = 18) -> float:
    edges = np.linspace(-np.pi, np.pi, nbins+1)
    digit = np.digitize(phase, edges) - 1
    digit = np.clip(digit, 0, nbins-1)
    m = np.zeros(nbins)
    for k in range(nbins):
        sel = (digit==k)
        m[k] = np.mean(amp[sel]) if np.any(sel) else 0.0
    if m.sum()<=0: return 0.0
    p = m / m.sum()
    eps=1e-12
    mi = np.sum(p*np.log((p+eps)/(1.0/nbins))) / np.log(nbins)
    return float(mi)

# ---------------- n:m phase locking ----------------
def n_m_plv(phi1: np.ndarray, phi2: np.ndarray, n: int = 1, m: int = 2) -> float:
    return float(np.abs(np.mean(np.exp(1j*(n*phi1 - m*phi2)))))

# ---------------- Comodulogram & PLV grids with nulls ----------------


"""
Patch: clamp all fast-frequency scans and plots to ≤ 60 Hz (or to Nyquist, if lower).

Why you were seeing >60 Hz: the original defaults used fast bands up to 80 Hz:
- PAC:   fast_range=(30, 80)
- n:m PLV: f2_range=(20, 80)

This patch parameterizes the high cutoff and clamps it to min(user_limit, 0.999*fs/2).
It also propagates the limit through plots so color maps never show >60 Hz.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# ---------------- PAC (Tort MI) ----------------
def comodulogram_pair(x_phase: np.ndarray, y_amp: np.ndarray, fs: float,
                      slow_range=(4,12), slow_bw=1.0, slow_step=1.0,
                      fast_range=(30,80), fast_bw=5.0, fast_step=2.0,
                      n_perm: int = 200,
                      max_fast_hz: float = 60.0):
    """Phase@x vs Amp@y PAC with an explicit upper clamp for fast frequencies.

    max_fast_hz: hard upper limit for fast band (default 60 Hz). The true upper
    limit applied is min(max_fast_hz, 0.999*fs/2, fast_range[1]).
    """
    fslow = np.arange(slow_range[0], slow_range[1] + 1e-6, slow_step)
    hi_cap = min(float(max_fast_hz), 0.999*float(fs)/2.0, float(fast_range[1]))
    if hi_cap <= fast_range[0] + 1e-9:
        raise ValueError(f"fast upper limit ({hi_cap:.2f} Hz) must exceed lower bound ({fast_range[0]} Hz)")
    ffast = np.arange(fast_range[0], hi_cap + 1e-6, fast_step)

    PAC = np.zeros((len(fslow), len(ffast)), float)

    def bandpass(x, f1, f2, order=4):
        ny = 0.5*fs
        f1 = max(1e-6, min(f1, 0.99*ny))
        f2 = max(f1 + 1e-6, min(f2, 0.999*ny))
        b, a = signal.butter(order, [f1/ny, f2/ny], btype='band')
        return signal.filtfilt(b, a, x)

    def analytic_phase_amp(x, f1, f2):
        xb = bandpass(x, f1, f2)
        z  = signal.hilbert(xb)
        return np.angle(z), np.abs(z)

    # compute PAC
    for i, f1 in enumerate(fslow):
        ph, _ = analytic_phase_amp(x_phase, f1 - slow_bw/2, f1 + slow_bw/2)
        for j, f2 in enumerate(ffast):
            _, amp = analytic_phase_amp(y_amp, f2 - fast_bw/2, f2 + fast_bw/2)
            # Tort MI (simple implementation)
            nbins = 18
            edges = np.linspace(-np.pi, np.pi, nbins+1)
            digit = np.digitize(ph, edges) - 1
            digit = np.clip(digit, 0, nbins-1)
            m = np.zeros(nbins)
            for k in range(nbins):
                sel = (digit == k)
                m[k] = np.mean(amp[sel]) if np.any(sel) else 0.0
            if m.sum() <= 0:
                PAC[i, j] = 0.0
            else:
                p = m / m.sum()
                eps = 1e-12
                PAC[i, j] = float(np.sum(p * np.log((p + eps) / (1.0/nbins))) / np.log(nbins))

    # null via circular shift of amp
    rng = np.random.default_rng(7)
    null95 = np.zeros_like(PAC)
    for i, f1 in enumerate(fslow):
        ph, _ = analytic_phase_amp(x_phase, f1 - slow_bw/2, f1 + slow_bw/2)
        for j, f2 in enumerate(ffast):
            _, amp = analytic_phase_amp(y_amp, f2 - fast_bw/2, f2 + fast_bw/2)
            maxima = []
            for _ in range(n_perm):
                s = int(rng.integers(1, len(amp)-1))
                amp_sh = np.r_[amp[-s:], amp[:-s]]
                # MI again
                nbins = 18
                edges = np.linspace(-np.pi, np.pi, nbins+1)
                digit = np.digitize(ph, edges) - 1
                digit = np.clip(digit, 0, nbins-1)
                m = np.zeros(nbins)
                for k in range(nbins):
                    sel = (digit == k)
                    m[k] = np.mean(amp_sh[sel]) if np.any(sel) else 0.0
                if m.sum() <= 0:
                    maxima.append(0.0)
                else:
                    p = m / m.sum()
                    eps = 1e-12
                    mi = float(np.sum(p * np.log((p + eps) / (1.0/nbins))) / np.log(nbins))
                    maxima.append(mi)
            null95[i, j] = float(np.nanpercentile(maxima, 95))

    return fslow, ffast, PAC, null95


# ---------------- n:m phase locking ----------------
def nm_plv_grid(x1: np.ndarray, x2: np.ndarray, fs: float,
                f1_range=(4,15), f1_bw=1.0, f1_step=1.0,
                f2_range=(20,80), f2_bw=2.0, f2_step=2.0,
                m_vals=(2,3,4,5,6), n: int = 1, n_perm: int = 200,
                max_f2_hz: float = 60.0):
    """Phase locking grid with an explicit upper clamp for f2 frequencies.

    max_f2_hz: hard upper limit for f2 (default 60 Hz). True upper limit is
    min(max_f2_hz, 0.999*fs/2, f2_range[1]).
    """
    f1s = np.arange(f1_range[0], f1_range[1] + 1e-6, f1_step)
    hi_cap = min(float(max_f2_hz), 0.999*float(fs)/2.0, float(f2_range[1]))
    if hi_cap <= f2_range[0] + 1e-9:
        raise ValueError(f"f2 upper limit ({hi_cap:.2f} Hz) must exceed lower bound ({f2_range[0]} Hz)")
    f2s = np.arange(f2_range[0], hi_cap + 1e-6, f2_step)

    def bandpass(x, f1, f2, order=4):
        ny = 0.5*fs
        f1 = max(1e-6, min(f1, 0.99*ny))
        f2 = max(f1 + 1e-6, min(f2, 0.999*ny))
        b, a = signal.butter(order, [f1/ny, f2/ny], btype='band')
        return signal.filtfilt(b, a, x)

    def analytic_phase(x, f1, f2):
        xb = bandpass(x, f1, f2)
        z  = signal.hilbert(xb)
        return np.angle(z)

    PLV = np.zeros((len(f1s), len(f2s), len(m_vals)), float)
    rng = np.random.default_rng(11)
    null95 = np.zeros_like(PLV)

    for i, f1 in enumerate(f1s):
        phi1 = analytic_phase(x1, f1 - f1_bw/2, f1 + f1_bw/2)
        for j, f2 in enumerate(f2s):
            phi2 = analytic_phase(x2, f2 - f2_bw/2, f2 + f2_bw/2)
            for k, m in enumerate(m_vals):
                plv = np.abs(np.mean(np.exp(1j * (n*phi1 - m*phi2))))
                PLV[i, j, k] = float(plv)
                # Null by circular phase shift of phi2
                maxima = []
                for _ in range(n_perm):
                    s = int(rng.integers(1, len(phi2)-1))
                    phi2_sh = np.r_[phi2[-s:], phi2[:-s]]
                    plv0 = np.abs(np.mean(np.exp(1j * (n*phi1 - m*phi2_sh))))
                    maxima.append(float(plv0))
                null95[i, j, k] = float(np.nanpercentile(maxima, 95))

    return f1s, f2s, m_vals, PLV, null95


# ---------------- Orchestrator (adds a high‑freq clamp knob) ----------------
def run_cfc_cross_region(
    RECORDS,
    pairs,
    ignition_windows=None,
    baseline_windows=None,
    time_col='Timestamp',
    out_dir='exports_cfc/session',
    show=False,
    n_perm=200,
    limit_high_hz: float = 60.0  # NEW: global clamp for fast/f2 frequencies
):
    """Same orchestrator as before, but enforces a ≤limit_high_hz scan in PAC/PLV.

    Note: plotting automatically respects the truncated frequency arrays, so the
    axes will end at ≤ limit_high_hz as well.
    """
    import os, pandas as pd
    from typing import Optional, List, Tuple

    def _ensure_dir(d): os.makedirs(d, exist_ok=True); return d

    def ensure_timestamp_column(df, time_col=None, default_fs: float = 128.0, out_name='Timestamp'):
        col = time_col if time_col is not None else 'Timestamp'
        if col not in df.columns:
            df[out_name] = np.arange(len(df), dtype=float)/default_fs; return out_name
        s = df[col]
        if np.issubdtype(s.dtype, np.datetime64) or 'datetime' in str(s.dtype).lower():
            tsec=(pd.to_datetime(s)-pd.to_datetime(s).iloc[0]).dt.total_seconds().astype(float)
            df[out_name]=tsec.values; return out_name
        sn = pd.to_numeric(s, errors='coerce').astype(float)
        if sn.notna().sum() < max(50, 0.5*len(df)):
            df[out_name]=np.arange(len(df), dtype=float)/default_fs; return out_name
        sn = sn - np.nanmin(sn[np.isfinite(sn)])
        df[out_name]=sn.values; return out_name

    def infer_fs(df, time_col: str) -> float:
        t = np.asarray(pd.to_numeric(df[time_col], errors='coerce').values, float)
        dt = np.diff(t); dt = dt[(dt > 0) & np.isfinite(dt)]
        if dt.size == 0:
            raise ValueError("Cannot infer fs.")
        return float(1.0/np.median(dt))

    def get_series(df, name: str) -> np.ndarray:
        if name in df.columns:
            return pd.to_numeric(df[name], errors='coerce').fillna(0.0).values.astype(float)
        alt = 'EEG.' + name
        if alt in df.columns:
            return pd.to_numeric(df[alt], errors='coerce').fillna(0.0).values.astype(float)
        raise ValueError(f"{name} not found.")

    _ensure_dir(out_dir)
    time_col = ensure_timestamp_column(RECORDS, time_col=time_col, default_fs=128.0)
    fs = infer_fs(RECORDS, time_col)

    def zscore(x):
        x = np.asarray(x, float)
        return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-12)

    results = {}
    rows = []

    def get_sig(name, wins):
        x = get_series(RECORDS, name)
        if not wins:
            return zscore(x)
        # slice & concat windows
        segs = []
        n = len(x)
        for (a,b) in wins:
            i0,i1 = int(round(a*fs)), int(round(b*fs))
            i0 = max(0, i0); i1 = min(n, i1)
            if i1 > i0:
                segs.append(x[i0:i1])
        return zscore(np.concatenate(segs) if segs else x)

    for st, wins in {'ignition': ignition_windows, 'baseline': baseline_windows}.items():
        if wins is None:
            continue
        for (ch_phase, ch_other) in pairs:
            a = ch_phase if ch_phase in RECORDS.columns else ('EEG.'+ch_phase if ('EEG.'+ch_phase) in RECORDS.columns else ch_phase)
            b = ch_other if ch_other in RECORDS.columns else ('EEG.'+ch_other if ('EEG.'+ch_other) in RECORDS.columns else ch_other)

            x = get_sig(a, wins)   # phase carrier
            y = get_sig(b, wins)   # amplitude or phase carrier

            # (1) PAC with clamp
            fslow, ffast, PAC, PAC_null95 = comodulogram_pair(
                x, y, fs,
                slow_range=(4,12), slow_bw=1.0, slow_step=1.0,
                fast_range=(30,80), fast_bw=5.0, fast_step=2.0,
                n_perm=n_perm,
                max_fast_hz=float(limit_high_hz)
            )

            imax = np.unravel_index(np.argmax(PAC), PAC.shape)
            pac_best = float(PAC[imax]); pac_thr = float(PAC_null95[imax])
            f1_best = float(fslow[imax[0]]); f2_best = float(ffast[imax[1]])
            pac_sig  = bool(pac_best > pac_thr)

            plt.figure(figsize=(7.8,3.2))
            extent=[ffast[0], ffast[-1], fslow[0], fslow[-1]]
            plt.imshow(PAC, aspect='auto', origin='lower', extent=extent, cmap='magma', vmin=0, vmax=np.nanmax(PAC))
            cb=plt.colorbar(); cb.set_label('PAC (MI)')
            sig_mask = PAC > PAC_null95
            yy, xx = np.where(sig_mask)
            if yy.size:
                plt.scatter(ffast[xx], fslow[yy], s=6, c='cyan', alpha=0.6, label='> null95')
            plt.xlabel('Fast freq (Hz)'); plt.ylabel('Slow freq (Hz)')
            plt.title(f'PAC: {a} phase → {b} amplitude — {st}')
            if yy.size: plt.legend(loc='upper right', fontsize=8)
            plt.tight_layout(); plt.savefig(os.path.join(out_dir, f'pac_{a}_to_{b}_{st}.png'), dpi=140)
            if show: plt.show()
            plt.close()

            # (2) n:m PLV with clamp
            f1s, f2s, m_vals, PLV, PLV_null95 = nm_plv_grid(
                x, y, fs,
                f1_range=(4,15), f1_bw=1.0, f1_step=1.0,
                f2_range=(20,80), f2_bw=2.0, f2_step=2.0,
                m_vals=(2,3,4,5,6), n=1, n_perm=n_perm,
                max_f2_hz=float(limit_high_hz)
            )

            imax_plv = np.unravel_index(np.argmax(PLV), PLV.shape)
            plv_best = float(PLV[imax_plv]); plv_thr = float(PLV_null95[imax_plv])
            f1_star = float(f1s[imax_plv[0]]); f2_star = float(f2s[imax_plv[1]]); m_star = int(m_vals[imax_plv[2]])
            plv_sig = bool(plv_best > plv_thr)

            k = imax_plv[2]
            plt.figure(figsize=(7.8,3.2))
            extent=[f2s[0], f2s[-1], f1s[0], f1s[-1]]
            plt.imshow(PLV[:,:,k], aspect='auto', origin='lower', extent=extent, cmap='viridis', vmin=0, vmax=np.nanmax(PLV[:,:,k]))
            cb=plt.colorbar(); cb.set_label(f'PLV (1:{m_vals[k]})')
            sig = PLV[:,:,k] > PLV_null95[:,:,k]
            yy, xx = np.where(sig)
            if yy.size:
                plt.scatter(f2s[xx], f1s[yy], s=6, c='cyan', alpha=0.6, label='> null95')
            plt.xlabel('f2 (Hz)'); plt.ylabel('f1 (Hz)')
            plt.title(f'n:m PLV (1:{m_vals[k]}): {a} ↔ {b} — {st}')
            if yy.size: plt.legend(loc='upper right', fontsize=8)
            plt.tight_layout(); plt.savefig(os.path.join(out_dir, f'plv_{a}_to_{b}_m{m_vals[k]}_{st}.png'), dpi=140)
            if show: plt.show()
            plt.close()

            rows.append({'state':st, 'pair':f'{a}->{b}',
                         'PAC_best':pac_best, 'PAC_thr95':pac_thr, 'PAC_sig':pac_sig,
                         'PAC_fslow':f1_best, 'PAC_ffast':f2_best,
                         'PLV_best':plv_best, 'PLV_thr95':plv_thr, 'PLV_sig':plv_sig,
                         'PLV_f1':f1_star, 'PLV_f2':f2_star, 'PLV_m':m_star})

            results.setdefault(st, {})[f'{a}->{b}'] = {
                'fslow':fslow, 'ffast':ffast, 'PAC':PAC, 'PAC_null95':PAC_null95,
                'f1s':f1s, 'f2s':f2s, 'm_vals':m_vals, 'PLV':PLV, 'PLV_null95':PLV_null95
            }

    summary = pd.DataFrame(rows)
    _ensure_dir(out_dir)
    summary.to_csv(os.path.join(out_dir,'cfc_summary.csv'), index=False)
    return {'summary': summary, 'results': results, 'out_dir': out_dir}


# ---------------- Usage example ----------------
# res = run_cfc_cross_region(
#     RECORDS,
#     pairs=[('EEG.F3','EEG.P8'), ('EEG.F4','EEG.P7')],
#     ignition_windows=[(290,310),(580,600)],
#     baseline_windows=[(0,290),(325,580)],
#     time_col='Timestamp',
#     out_dir='exports_cfc/S01',
#     show=False,
#     n_perm=200,
#     limit_high_hz=60.0  # ← clamp to ≤60 Hz
# )
