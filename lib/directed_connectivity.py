"""
DLPFC → Sensory Top-Down Connectivity (Source-space)
----------------------------------------------------
Beamform/sLORETA per event window, then compute:
  • dPLI (directional phase-lag index)  — volume-conduction-resilient
  • Conditional Granger (VAR)          — causal predictability
  • Phase Transfer Entropy (stub)      — ready to wire IDTxl/JIDT

Includes a SENSOR fallback (F4 → posterior sensors) while you prepare MRI assets.

Requirements
------------
python >= 3.9
pip install mne mne-connectivity statsmodels
(For PTE: IDTxl or JIDT+JPype — not used by default here)

Usage sketch
------------
from DLPFC_TopDown_SourceConnectivity import (
    run_topdown_ignition_pipeline,
)

windows = [(12.0, 17.0), (33.0, 38.0), (85.5, 90.5)]

# SOURCE space (main analysis)
df_src = run_topdown_ignition_pipeline(
    records=RECORDS, electrodes=ELECTRODES, fs=FS,  # or pass raw directly
    raw=None,
    windows=windows,
    mode='source',
    subjects_dir='/path/to/subjects_dir',
    subject='fsaverage',
    trans='/path/to/trans.fif',
    method='lcmv',             # or 'sloreta'
    granger_maxlags=10
)
print(df_src.head())

# SENSOR fallback (fast validation; no MRI)
df_sensor = run_topdown_ignition_pipeline(
    records=RECORDS, electrodes=ELECTRODES, fs=FS,
    raw=None, windows=windows, mode='sensor'
)
print(df_sensor.head())
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# --- Dependencies & soft-checks ---
try:
    import mne
    from mne.beamformer import make_lcmv, apply_lcmv_raw
    from mne.minimum_norm import make_inverse_operator, apply_inverse_raw
    from mne.label import read_labels_from_annot
    from mne import extract_label_time_course   # correct import path in modern MNE
    from mne.filter import filter_data
except Exception as e:
    raise ImportError("This module requires MNE-Python. Install with: pip install mne\n" + str(e))

try:
    import mne_connectivity
    _HAS_MNE_CONN = True
except Exception:
    _HAS_MNE_CONN = False
    warnings.warn("mne-connectivity not found — dPLI will return NaN.\n"
                  "Install with: pip install mne-connectivity")

try:
    from statsmodels.tsa.api import VAR
    _HAS_STATSMODELS = True
except Exception:
    _HAS_STATSMODELS = False
    warnings.warn("statsmodels not found — Granger causality will return NaN.\n"
                  "Install with: pip install statsmodels")

# ----------------- Parameters -----------------
BANDS: Dict[str, Tuple[float, float]] = {
    'theta': (4.0, 8.0),
    'alpha': (8.0, 13.0),
    'beta':  (13.0, 30.0),
}

@dataclass
class ROIMap:
    """Aparc label lists for each ROI (Right DLPFC + sensory)."""
    dlpfc_r: List[str]
    occipital: List[str]
    temporal: List[str]
    parietal: List[str]

# Default ROI mapping using aparc (Desikan-Killiany)
DEFAULT_ROIMAP = ROIMap(
    dlpfc_r=[
        'rh_caudalmiddlefrontal', 'rh_rostralmiddlefrontal', 'rh_superiorfrontal'
    ],
    occipital=[
        'lh_lateraloccipital', 'lh_cuneus', 'lh_lingual',
        'rh_lateraloccipital', 'rh_cuneus', 'rh_lingual'
    ],
    temporal=[
        'lh_superiortemporal', 'lh_middletemporal', 'lh_inferiortemporal',
        'rh_superiortemporal', 'rh_middletemporal', 'rh_inferiortemporal'
    ],
    parietal=[
        'lh_superiorparietal', 'lh_inferiorparietal', 'lh_precuneus', 'lh_supramarginal',
        'rh_superiorparietal', 'rh_inferiorparietal', 'rh_precuneus', 'rh_supramarginal'
    ]
)

# ----------------- Helpers -----------------

def df_to_raw(records: pd.DataFrame, ch_names: List[str], sfreq: float, montage: str = 'standard_1020') -> mne.io.Raw:
    """Convert a (Timestamp + EEG.<ch>.FILTERED) DataFrame into MNE RawArray."""
    eeg_cols = [f"EEG.{c}.FILTERED" for c in ch_names]
    X = records[eeg_cols].to_numpy(dtype=float).T
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
    raw = mne.io.RawArray(X, info)
    try:
        raw.set_montage(mne.channels.make_standard_montage(montage))
    except Exception:
        warnings.warn("Could not set montage — check channel names/montage string.")
    raw.set_eeg_reference('average', projection=False)
    return raw

# ----------------- Connectivity metrics -----------------

def _compute_dpli(data2xT: np.ndarray, sfreq: float, fmin: float, fmax: float) -> float:
    """dPLI(source→target) for a 2×T array (row0=src, row1=tgt)."""
    if not _HAS_MNE_CONN:
        return np.nan
    arr = data2xT[np.newaxis, ...]  # (1, 2, T)
    res = mne_connectivity.spectral_connectivity(
        data=arr, method='dpli', mode='multitaper', sfreq=sfreq,
        fmin=fmin, fmax=fmax, faverage=True, verbose=False
    )
    return float(res.get_data()[0, 1, 0])


def _conditional_granger(source: np.ndarray, target: np.ndarray, conditioners: Optional[np.ndarray], maxlags: int = 10) -> Dict:
    """Conditional Granger: does src cause tgt given other series? Returns dict with F, p, lags."""
    if not _HAS_STATSMODELS:
        return {'F': np.nan, 'pval': np.nan, 'lags': np.nan}
    # Build matrix [src, tgt, conders...]
    mat = np.vstack([source, target]).T
    cols = ['src', 'tgt']
    if conditioners is not None and conditioners.size:
        mat = np.concatenate([mat, conditioners.T], axis=1)
        cols += [f'c{i}' for i in range(conditioners.shape[0])]
    df_sig = pd.DataFrame(mat, columns=cols)
    try:
        model = VAR(df_sig.values)
        fitted = model.fit(maxlags=maxlags, ic='aic')
        res = fitted.test_causality(caused=1, causing=[0])  # 1=tgt, 0=src column index
        return {'F': float(res.test_statistic), 'pval': float(res.pvalue), 'lags': int(fitted.k_ar)}
    except Exception:
        return {'F': np.nan, 'pval': np.nan, 'lags': np.nan}


def phase_transfer_entropy_stub(*args, **kwargs) -> float:
    """Stub for PTE (wire IDTxl/JIDT here if desired)."""
    warnings.warn("Phase Transfer Entropy not implemented — install IDTxl/JIDT and integrate here.")
    return np.nan

# ----------------- SENSOR fallback -----------------

def sensor_directed_connectivity(
    raw: mne.io.BaseRaw,
    windows: List[Tuple[float, float]],
    bands: Dict[str, Tuple[float, float]] = BANDS,
    source_ch: str = 'F4',
    posterior_chs: Tuple[str, ...] = ('O1', 'O2', 'P7', 'P8', 'T7', 'T8'),
    granger_maxlags: int = 10,
) -> pd.DataFrame:
    """Compute dPLI + (pairwise) Granger from F4→posterior sensors per event window & band."""
    out = []
    for (t0, t1) in windows:
        seg = raw.copy().crop(tmin=t0, tmax=t1, include_tmax=False)
        try:
            s_full = seg.copy().pick_channels([source_ch]).get_data()[0]
        except Exception:
            warnings.warn(f"Source channel {source_ch} not found — skipping window {(t0,t1)}")
            continue
        for tgt_ch in posterior_chs:
            try:
                t_full = seg.copy().pick_channels([tgt_ch]).get_data()[0]
            except Exception:
                continue
            for band, (f1, f2) in bands.items():
                s = filter_data(s_full, raw.info['sfreq'], f1, f2, verbose=False)
                t = filter_data(t_full, raw.info['sfreq'], f1, f2, verbose=False)
                dpli = _compute_dpli(np.vstack([s, t]), raw.info['sfreq'], f1, f2)
                if _HAS_STATSMODELS:
                    try:
                        fitted = VAR(pd.DataFrame({'s': s, 't': t}).values).fit(maxlags=granger_maxlags, ic='aic')
                        cres = fitted.test_causality(caused=1, causing=[0])
                        F, p = float(cres.test_statistic), float(cres.pvalue)
                    except Exception:
                        F, p = np.nan, np.nan
                else:
                    F, p = np.nan, np.nan
                out.append({'window': (t0, t1), 'pair': f'{source_ch}->{tgt_ch}', 'band': band,
                            'dpli': dpli, 'granger_F': F, 'granger_p': p})
    return pd.DataFrame(out)

# ----------------- SOURCE space -----------------

def source_directed_connectivity(
    raw: mne.io.BaseRaw,
    windows: List[Tuple[float, float]],
    subjects_dir: str,
    subject: str,
    trans: str,
    method: str = 'lcmv',        # 'lcmv' or 'sloreta'
    parc: str = 'aparc',
    roimap: ROIMap = DEFAULT_ROIMAP,
    bands: Dict[str, Tuple[float, float]] = BANDS,
    reg: float = 0.05,
    granger_maxlags: int = 10,
) -> pd.DataFrame:
    """Compute dPLI + conditional Granger: DLPFC_R→(occipital/temporal/parietal) per window & band."""
    # Forward model
    try:
        src = mne.setup_source_space(subject=subject, spacing='oct6', subjects_dir=subjects_dir, add_dist=False)
        bem = mne.make_bem_model(subject=subject, ico=4, subjects_dir=subjects_dir)
        bem_sol = mne.make_bem_solution(bem)
        fwd = mne.make_forward_solution(raw.info, trans=trans, src=src, bem=bem_sol, eeg=True, mindist=5.0)
    except Exception as e:
        raise RuntimeError(f"Forward model creation failed: {e}. Provide precomputed fwd/src/bem.")

    # Labels
    labels = read_labels_from_annot(subject=subject, parc=parc, subjects_dir=subjects_dir)
    label_dict = {lab.name: lab for lab in labels}
    def _pick(labels_list: List[str]):
        return [label_dict[n] for n in labels_list if n in label_dict]

    out_rows = []
    sfreq = raw.info['sfreq']

    for (t0, t1) in windows:
        seg = raw.copy().crop(tmin=t0, tmax=t1, include_tmax=False)
        # Covariances per window
        data_cov = mne.compute_raw_covariance(seg, verbose=False)
        noise_cov = mne.compute_raw_covariance(seg.copy().filter(45., None, verbose=False), verbose=False)

        if method.lower() == 'lcmv':
            filters = make_lcmv(raw.info, fwd, data_cov=data_cov, noise_cov=noise_cov,
                                reg=reg, pick_ori='max-power', weight_norm='unit-noise-gain')
            stc = apply_lcmv_raw(seg, filters, max_ori_out='signed')
        elif method.lower() == 'sloreta':
            inv = make_inverse_operator(raw.info, fwd, noise_cov, loose=0.2, depth=0.8)
            stc = apply_inverse_raw(seg, inv, lambda2=1./9., method='sLORETA')
        else:
            raise ValueError("method must be 'lcmv' or 'sloreta'")

        # ROI time-courses
        roi_tc: Dict[str, np.ndarray] = {}
        for roi_name, roi_list in {
            'dlpfc_r': roimap.dlpfc_r,
            'occipital': roimap.occipital,
            'temporal': roimap.temporal,
            'parietal': roimap.parietal,
        }.items():
            labs = _pick(roi_list)
            if not labs:
                warnings.warn(f"No labels found for ROI {roi_name} — check atlas/parc.")
                continue
            tcs = extract_label_time_course(stc, labels=labs, src=fwd['src'], mode='mean_flip')
            roi_tc[roi_name] = np.mean(tcs, axis=0)   # average across labels

        # DLPFC→ROI per band
        for band, (f1, f2) in bands.items():
            if 'dlpfc_r' not in roi_tc:
                continue
            s = filter_data(roi_tc['dlpfc_r'], sfreq, f1, f2, verbose=False)
            for tgt in ['occipital', 'temporal', 'parietal']:
                if tgt not in roi_tc:
                    continue
                t = filter_data(roi_tc[tgt], sfreq, f1, f2, verbose=False)
                dpli = _compute_dpli(np.vstack([s, t]), sfreq, f1, f2)
                # conditional Granger: condition on other sensory ROIs
                conders = []
                for c in ['occipital', 'temporal', 'parietal']:
                    if c != tgt and c in roi_tc:
                        conders.append(filter_data(roi_tc[c], sfreq, f1, f2, verbose=False))
                conders = np.vstack(conders) if len(conders) else None
                cg = _conditional_granger(s, t, conders, maxlags=granger_maxlags)
                out_rows.append({
                    'window': (t0, t1),
                    'band': band,
                    'edge': f'DLPFC_R→{tgt}',
                    'dpli': dpli,
                    'granger_F': cg.get('F', np.nan),
                    'granger_p': cg.get('pval', np.nan),
                    'lags': cg.get('lags', np.nan),
                })

    return pd.DataFrame(out_rows)

# ----------------- Top-level wrapper -----------------

def run_topdown_ignition_pipeline(
    records: Optional[pd.DataFrame],
    electrodes: Optional[List[str]],
    fs: Optional[float],
    raw: Optional[mne.io.BaseRaw],
    windows: List[Tuple[float, float]],
    subjects_dir: Optional[str] = None,
    subject: Optional[str] = None,
    trans: Optional[str] = None,
    mode: str = 'sensor',   # 'sensor' or 'source'
    method: str = 'lcmv',
    **kwargs,
) -> pd.DataFrame:
    """Run SENSOR or SOURCE pipeline.

    SENSOR: F4→(O1/O2/P7/P8/T7/T8) dPLI & pairwise Granger per band/window.
    SOURCE: DLPFC_R→(occipital/temporal/parietal) dPLI & **conditional** Granger per band/window.
    """
    # build raw if needed
    if raw is None:
        if records is None or electrodes is None or fs is None:
            raise ValueError("Provide either raw, or (records + electrodes + fs).")
        raw = df_to_raw(records, electrodes, fs)

    if mode == 'sensor':
        return sensor_directed_connectivity(raw=raw, windows=windows, **kwargs)

    if mode == 'source':
        if subjects_dir is None or subject is None or trans is None:
            raise ValueError("SOURCE mode requires subjects_dir, subject, and trans.")
        return source_directed_connectivity(
            raw=raw, windows=windows, subjects_dir=subjects_dir, subject=subject,
            trans=trans, method=method, **kwargs,
        )

    raise ValueError("mode must be 'sensor' or 'source'")
