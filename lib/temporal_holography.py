"""
Temporal Holography — Multiplexed References (fs=128)
-----------------------------------------------------
Hypothesis: Different reference phases (theta vs alpha) “select” different content traces
(holographic multiplexing). Trials binned by **reference phase at event onset** yield
content-specific spectral/PAC signatures and better decoding.

This module:
1) Bins trials by **phase at onset** in a chosen reference band (theta or alpha) from a
   chosen **reference signal** (electrode or cluster average).
2) Extracts per-trial **spectral features** (band powers across electrodes) and **PAC
   fingerprints** (e.g., theta→gamma, alpha→gamma) in an analysis window around onset.
3) Computes **between-bin separability** via LDA **AUC** for supplied content labels.
4) Returns per-bin **PAC fingerprints** and runs **random-bin controls** for AUC.

Inputs you provide
- `event_onsets`: list/array of onset times (s) per trial
- `labels`: optional trial labels (binary/multi-class); if omitted, AUC is skipped
- `ref_electrodes`: reference channel(s) for phase at onset (e.g., occipital/parietal)

Usage
-----
res = run_temporal_holography_multiplexed(
    RECORDS,
    event_onsets=ONSETS,           # list/array of onset times
    labels=LABELS,                 # optional (len==len(ONSETS))
    time_col='Timestamp',
    electrodes=['F4','O1','O2', ...],   # feature electrodes
    ref_electrodes=['O1','O2'],         # reference for phase at onset
    ref_band='theta',                   # 'theta' or 'alpha'
    n_bins=6,
    feat_bands={'theta':(4,8),'alpha':(8,13),'beta':(13,30),'gamma':(30,80)},
    feat_window=(-0.5, 1.0),           # seconds around onset
    pac_pairs={'theta→gamma':((4,8),(30,80)), 'alpha→gamma':((8,13),(30,80))},
    n_shuffle=200,
    show=True
)

print(res['auc_table'])           # AUC per phase bin (and null bands if labels provided)
print(res['pac_fingerprints'].keys())
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from scipy import signal

# sklearn for LDA/AUC
try:
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    _HAS_SK = True
except Exception:
    _HAS_SK = False

# ----------------- helpers -----------------

def _get_fs(RECORDS: pd.DataFrame, time_col: str) -> float:
    if 'infer_fs_from_records' in globals():
        try: return float(infer_fs_from_records(RECORDS, time_col=time_col))
        except Exception: pass
    t = np.asarray(RECORDS[time_col].values, dtype=float)
    dt = np.diff(t); dt = dt[np.isfinite(dt) & (dt>0)]
    if dt.size == 0: raise ValueError('Cannot infer fs')
    return 1.0/np.median(dt)


def _bandpass(x: np.ndarray, fs: float, f1: float, f2: float, order: int=4) -> np.ndarray:
    ny = 0.5*fs
    f1 = max(1e-6, min(f1, ny*0.99)); f2 = max(f1+1e-6, min(f2, ny*0.999))
    b,a = signal.butter(order, [f1/ny, f2/ny], btype='band')
    return signal.filtfilt(b,a,x)


def _get_series_matrix(RECORDS, electrodes, time_col):
    series=[]
    for ch in electrodes:
        s = find_channel_series(RECORDS, ch)
        if s is None: raise ValueError(f'Missing channel {ch}')
        series.append(np.asarray(s.values, float))
    X = np.vstack(series)
    t = np.asarray(RECORDS[time_col].values, float)
    return t, X


def _phase_at_time(ref_sig: np.ndarray, tvec: np.ndarray, fs: float, band: Tuple[float,float], onset: float) -> float:
    # find nearest index
    idx = int(round(onset*fs))
    if idx < 0 or idx >= ref_sig.size: return np.nan
    xb = _bandpass(ref_sig, fs, band[0], band[1])
    ang = np.angle(signal.hilbert(xb))
    return float(ang[idx])


def _pac_mi_single(x_phase: np.ndarray, x_amp: np.ndarray, nbins: int=18) -> float:
    ph = np.angle(signal.hilbert(x_phase))
    am = np.abs(signal.hilbert(x_amp))
    edges = np.linspace(-np.pi, np.pi, nbins+1)
    digit = np.digitize(ph, edges)-1; digit = np.clip(digit,0,nbins-1)
    m = np.zeros(nbins)
    for k in range(nbins):
        sel = (digit==k); m[k] = np.mean(am[sel]) if np.any(sel) else 0.0
    if m.sum() <= 0: return 0.0
    p = m / m.sum(); eps=1e-12
    kl = np.sum(p*np.log((p+eps)/(1.0/nbins)))
    return float(kl/np.log(nbins))

def find_channel_series(records: pd.DataFrame, ch_name: str) -> Optional[pd.Series]:
    # ---------------- Basics: fs + channel access ----------------
    _DEF_TIME_COL = 'Timestamp'
    _DEF_CH_PATTERNS = ("EEG.{ch}", "eeg.{ch}", "{ch}", "RAW.{ch}", "CHAN.{ch}")
    for pat in _DEF_CH_PATTERNS:
        col = pat.format(ch=ch_name)
        if col in records.columns:
            return pd.to_numeric(records[col], errors='coerce').astype(float)
    return None

# ----------------- core -----------------

def run_temporal_holography_multiplexed(
    RECORDS: pd.DataFrame,
    event_onsets: List[float],
    labels: Optional[List] = None,          # optional labels per trial
    time_col: str = 'Timestamp',
    electrodes: Optional[List[str]] = None,
    ref_electrodes: Optional[List[str]] = None,
    ref_band: str = 'theta',                # 'theta' or 'alpha'
    n_bins: int = 6,
    feat_bands: Dict[str,Tuple[float,float]] = None,
    feat_window: Tuple[float,float] = (-0.5, 1.0),
    pac_pairs: Optional[Dict[str, Tuple[Tuple[float,float],Tuple[float,float]]]] = None,
    n_shuffle: int = 200,
    show: bool = True,
    rng_seed: int = 17,
) -> Dict[str, object]:
    fs = _get_fs(RECORDS, time_col)
    if electrodes is None:
        # autodiscover EEG.*
        electrodes = [c.split('.',1)[1] for c in RECORDS.columns if c.startswith('EEG.')]
    if ref_electrodes is None:
        # fallback: use O1/O2 if present else first electrode
        ref_electrodes = [e for e in ['O1','O2','Oz','Pz'] if e in electrodes] or electrodes[:1]
    feat_bands = feat_bands or {'theta':(4,8),'alpha':(8,13),'beta':(13,30),'gamma':(30,80)}
    pac_pairs = pac_pairs or {'theta→gamma':((4,8),(30,80)), 'alpha→gamma':((8,13),(30,80))}

    # data
    t, X = _get_series_matrix(RECORDS, electrodes, time_col)
    # reference signal = mean of ref_electrodes
    _, Xref = _get_series_matrix(RECORDS, ref_electrodes, time_col)
    ref_sig = np.nanmean(Xref, axis=0)

    # choose band for phase binning
    if ref_band == 'theta':  ref_bp = (4,8)
    elif ref_band == 'alpha': ref_bp = (8,13)
    else: raise ValueError("ref_band must be 'theta' or 'alpha'")

    # compute phase & bin per trial
    phases = []
    for onset in event_onsets:
        phases.append(_phase_at_time(ref_sig, t, fs, ref_bp, onset))
    phases = np.asarray(phases)
    # edges from -pi..pi into n_bins
    edges = np.linspace(-np.pi, np.pi, n_bins+1)
    bins = np.digitize(phases, edges) - 1
    bins = np.clip(bins, 0, n_bins-1)

    # feature extraction per trial
    rng = np.random.default_rng(rng_seed)
    win_samp = (int(round(feat_window[0]*fs)), int(round(feat_window[1]*fs)))

    def trial_segment(on):
        i0 = int(round(on*fs)) + win_samp[0]
        i1 = int(round(on*fs)) + win_samp[1]
        i0 = max(0, i0); i1 = min(X.shape[1], i1)
        if i1 <= i0: return None
        return X[:, i0:i1]

    # spectral features: band power per electrode × band
    feat_list = []
    keep_idx = []
    for ti,on in enumerate(event_onsets):
        seg = trial_segment(on)
        if seg is None: continue
        fv = []
        for _,(f1,f2) in feat_bands.items():
            xb = _bandpass(seg, fs, f1,f2)
            fv.extend(list(np.mean(xb**2, axis=1)))  # mean power per electrode
        feat_list.append(np.asarray(fv, float))
        keep_idx.append(ti)
    if len(feat_list)==0:
        raise ValueError('No valid trial segments in feature window')
    F = np.vstack(feat_list)
    bins = bins[keep_idx]
    phases = phases[keep_idx]
    y = np.asarray(labels)[keep_idx] if labels is not None else None

    # PAC fingerprints: average within each phase bin
    pac_fingerprints: Dict[str, np.ndarray] = {}
    for name,(pb,ab) in pac_pairs.items():
        vals = np.zeros(n_bins, float)
        for b in range(n_bins):
            # concatenate all trials in bin b
            idx = np.where(bins==b)[0]
            if idx.size==0: vals[b]=np.nan; continue
            # build one long segment by concatenation over trials
            cat = np.hstack([trial_segment(event_onsets[keep_idx[i]]) for i in idx])
            xp = _bandpass(cat, fs, pb[0], pb[1])
            xa = _bandpass(cat, fs, ab[0], ab[1])
            # average per-channel MI then mean across channels
            mis = [_pac_mi_single(xp[ch], xa[ch]) for ch in range(xp.shape[0])]
            vals[b] = float(np.nanmean(mis)) if mis else np.nan
        pac_fingerprints[name] = vals

    # Between-bin separability: LDA AUC by bin (if labels provided and sklearn present)
    auc_rows = []
    if (y is not None) and _HAS_SK and (len(np.unique(y))>=2):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=rng.integers(1e9))
        for b in range(n_bins):
            idx = np.where(bins==b)[0]
            if idx.size < 8: 
                auc_rows.append({'bin':b, 'AUC':np.nan});
                continue
            Xb = F[idx]
            yb = y[idx]
            try:
                # binary or one-vs-rest macro AUC
                if len(np.unique(yb))==2:
                    auc_cv = []
                    for tr,te in skf.split(Xb, yb):
                        clf = LDA(); clf.fit(Xb[tr], yb[tr])
                        prob = clf.predict_proba(Xb[te])[:,1]
                        auc_cv.append(roc_auc_score(yb[te], prob))
                    auc = float(np.mean(auc_cv))
                else:
                    # macro-averaged OVR AUC
                    from sklearn.preprocessing import label_binarize
                    classes = np.unique(yb)
                    Yb = label_binarize(yb, classes=classes)
                    auc_cv=[]
                    for tr,te in skf.split(Xb, yb):
                        clf=LDA(); clf.fit(Xb[tr], yb[tr])
                        pro=clf.predict_proba(Xb[te])
                        auc_cv.append(np.mean([roc_auc_score(Yb[te][:,k], pro[:,k]) for k in range(Yb.shape[1])]))
                    auc = float(np.mean(auc_cv))
            except Exception:
                auc = np.nan
            auc_rows.append({'bin':b, 'AUC':auc})
        auc_table = pd.DataFrame(auc_rows)
    else:
        auc_table = pd.DataFrame([{'bin':b, 'AUC':np.nan} for b in range(n_bins)])

    # Random-bin controls for AUC
    control = None
    if (y is not None) and _HAS_SK and n_shuffle>0 and len(np.unique(y))>=2:
        null = []
        for _ in range(n_shuffle):
            b_rand = bins.copy(); rng.shuffle(b_rand)
            # compute mean AUC over bins (only where ≥8 trials)
            aucs=[]
            for b in range(n_bins):
                idx = np.where(b_rand==b)[0]
                if idx.size < 8: continue
                Xb = F[idx]; yb = y[idx]
                try:
                    if len(np.unique(yb))==2:
                        cv = []
                        for tr,te in skf.split(Xb, yb):
                            clf=LDA(); clf.fit(Xb[tr], yb[tr])
                            pro=clf.predict_proba(Xb[te])[:,1]
                            cv.append(roc_auc_score(yb[te], pro))
                        aucs.append(np.mean(cv))
                    else:
                        from sklearn.preprocessing import label_binarize
                        classes=np.unique(yb); Yb=label_binarize(yb, classes=classes)
                        cv=[]
                        for tr,te in skf.split(Xb, yb):
                            clf=LDA(); clf.fit(Xb[tr], yb[tr])
                            pro=clf.predict_proba(Xb[te])
                            cv.append(np.mean([roc_auc_score(Yb[te][:,k], pro[:,k]) for k in range(Yb.shape[1])]))
                        aucs.append(np.mean(cv))
                except Exception:
                    pass
            null.append(np.nanmean(aucs) if len(aucs)>0 else np.nan)
        control = {'null_auc_mean': np.nanmean(null), 'null_auc_ci': (np.nanpercentile(null,2.5), np.nanpercentile(null,97.5))}

    # Plots
    if show:
        # AUC by bin (if available)
        plt.figure(figsize=(8,3))
        plt.bar(auc_table['bin'].values, auc_table['AUC'].values, width=0.8)
        plt.xlabel(f'{ref_band} phase bin'); plt.ylabel('AUC'); plt.title('Between-bin separability (LDA AUC)')
        if control is not None and np.isfinite(control['null_auc_mean']):
            mu, (lo,hi) = control['null_auc_mean'], control['null_auc_ci']
            plt.axhline(mu, color='k', lw=1.0)
            plt.fill_between([-0.5, n_bins-0.5], lo, hi, color='k', alpha=0.15)
        plt.show()

        # PAC fingerprints across bins
        for name,vals in pac_fingerprints.items():
            plt.figure(figsize=(8,3))
            plt.bar(np.arange(n_bins), vals, width=0.8)
            plt.xlabel(f'{ref_band} phase bin'); plt.ylabel('PAC MI'); plt.title(f'PAC fingerprint: {name}')
            plt.show()

    return {
        'bins': bins,
        'phases': phases,
        'auc_table': auc_table,
        'pac_fingerprints': pac_fingerprints,
        'control': control,
        'params': {'ref_band':ref_band,'n_bins':n_bins,'feat_window':feat_window}
    }
