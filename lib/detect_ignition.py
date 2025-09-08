# ============================
# Ignition detector (one-file)
# ============================
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from typing import Dict, List, Tuple, Optional

# --------- small helpers ---------
def _ensure_dir(d): os.makedirs(d, exist_ok=True); return d

def detect_time_col(df, candidates=('Timestamp','Time','time','t','seconds','sec','ms','datetime','DateTime','Datetime')):
    for c in candidates:
        if c in df.columns: return c
    for c in df.columns:
        s = pd.to_numeric(df[c], errors='coerce')
        if s.notna().sum() > max(50, 0.5*len(df)):
            x = s.values.astype(float); dt = np.diff(x[np.isfinite(x)])
            if dt.size and np.nanmedian(dt)>0: return c
    for c in df.columns:
        try:
            _ = pd.to_datetime(df[c], errors='raise'); return c
        except Exception:
            pass
    return None

def ensure_timestamp_column(df, time_col=None, default_fs=128.0, out_name='Timestamp'):
    col = time_col or detect_time_col(df)
    if col is None:
        df[out_name] = np.arange(len(df), dtype=float)/default_fs; return out_name
    s = df[col]
    if np.issubdtype(s.dtype, np.datetime64) or 'datetime' in str(s.dtype).lower():
        tsec = (pd.to_datetime(s) - pd.to_datetime(s).iloc[0]).dt.total_seconds().astype(float)
        df[out_name] = tsec.values; return out_name
    sn = pd.to_numeric(s, errors='coerce').astype(float)
    if sn.notna().sum() < max(50, 0.5*len(df)):
        df[out_name] = np.arange(len(df), dtype=float)/default_fs; return out_name
    sn = sn - np.nanmin(sn[np.isfinite(sn)])
    df[out_name] = sn.values
    return out_name

def infer_fs(df, time_col):
    t = np.asarray(pd.to_numeric(df[time_col], errors='coerce').values, float)
    dt = np.diff(t); dt = dt[(dt>0)&np.isfinite(dt)]
    if dt.size==0: raise ValueError("Cannot infer fs from time column.")
    return float(1.0/np.median(dt))

def zscore(x):
    x = np.asarray(x, float)
    return (x - np.mean(x)) / (np.std(x) + 1e-12)

def bandpass(x, fs, f1, f2, order=4):
    ny=0.5*fs; f1=max(1e-6, min(f1, 0.99*ny)); f2=max(f1+1e-6, min(f2, 0.999*ny))
    b,a = signal.butter(order, [f1/ny, f2/ny], btype='band')
    return signal.filtfilt(b,a,np.asarray(x,float))



def get_series(df: pd.DataFrame, name: str) -> np.ndarray:
    """
    Robustly fetch a channel by name (accepts 'EEG.X', 'X', case-insensitive).
    Raises ValueError with suggestions if not found.
    """

#     real = _resolve_channel_name(df, name)
    if name is None:
        # suggest a few candidates that contain the token
        token = (name or '').replace('EEG.', '')
        sugg = [c for c in df.columns if token.lower() in c.lower()][:8]
        raise ValueError(f"{name} not found. Suggestions: {sugg}" if sugg else f"{name} not found.")

    return pd.to_numeric(df[name], errors='coerce').fillna(0.0).values.astype(float)

def slice_concat(x, fs, wins):
    if not wins: return x.copy()
    segs=[]; n=len(x)
    for (a,b) in wins:
        i0,i1 = int(round(a*fs)), int(round(b*fs))
        i0=max(0,i0); i1=min(n,i1)
        if i1>i0: segs.append(x[i0:i1])
    return np.concatenate(segs) if segs else x.copy()

# --------- sliding Kuramoto R(t) (global synchrony) ---------
def kuramoto_R_timeseries(X, fs, band=(8,13), win_sec=1.0, step_sec=0.25):
    n,T = X.shape
    win = int(round(win_sec*fs)); step=int(round(step_sec*fs))
    tcent=[]; Rt=[]
    for c in range(win//2, T-win//2, step):
        Xw = X[:, c-win//2:c+win//2]
        Xb = np.vstack([bandpass(x, fs, band[0], band[1]) for x in Xw])
        Z = signal.hilbert(Xb, axis=1)
        phi = np.angle(Z)
        R = np.abs(np.mean(np.exp(1j*phi), axis=0)).mean()
        Rt.append(R); tcent.append(c/fs)
    return np.array(tcent,float), np.array(Rt,float)

def _merge_intervals_int(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Merge overlapping/touching integer-second intervals in-place and return a new list.
    Each interval is a half-open-like [start, end] in whole seconds where start < end.
    Touching intervals (e.g. (180, 200) and (200, 220)) are merged into (180, 220).
    """
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda ab: (ab[0], ab[1]))
    out = [intervals[0]]
    for a, b in intervals[1:]:
        sa, sb = out[-1]
        if a <= sb: # overlap or touch
            out[-1] = (sa, max(sb, b))
        else:
            out.append((a, b))
    return out


def detect_ignitions_session(
    RECORDZ: pd.DataFrame,
    sr_channel: Optional[str] = "EEG.F4",                 # magnetometer or posterior proxy
    eeg_channels: Optional[List[str]] = None,              # channels to build R(t) and EEG power/coh
    time_col: str = 'Timestamp',
    out_dir: str = 'exports_ignitions/S01',
    # detection params
    center_hz: float = 7.83, half_bw_hz: float = 0.6,
    smooth_sec: float = 0.25, z_thresh: float = 2.5,
    min_isi_sec: float = 2.0, window_sec: float = 20.0, merge_gap_sec: float = 5.0,
    # validation params
    R_band: Tuple[float, float] = (8,13), R_win_sec: float = 1.0, R_step_sec: float = 0.25,
    eta_pre_sec: float = 10.0, eta_post_sec: float = 10.0,
    show: bool = True,
    verbose: bool = True
) -> Dict[str, object]:
    """
    Detect SR-band bursts → ignition windows, and compute validation graphs + stats.

    Returns
    -------
    dict with keys including (non-exhaustive):
      - events: DataFrame of per-event stats
      - ignition_windows: list of (float_start, float_end) seconds
      - ignition_windows_rounded: list of (int_start, int_end) seconds (merged), suitable for downstream params
      - ignition_windows_path: path to a JSON file with the rounded list
    """

    if eeg_channels is None:
        # Fallback to all EEG.* columns if not provided
        eeg_channels = [c for c in RECORDZ.columns if c.startswith('EEG.')]

    _ensure_dir(out_dir)
    time_col = ensure_timestamp_column(RECORDZ, time_col=time_col, default_fs=128.0)
    fs = infer_fs(RECORDZ, time_col)
    t = pd.to_numeric(RECORDZ[time_col], errors='coerce').values.astype(float)
    T = len(t)

    # --- 1) SR envelope z(t) & burst onsets ---
    y = get_series(RECORDZ, sr_channel)

    yb = bandpass(y, fs, center_hz - half_bw_hz, center_hz + half_bw_hz)
    env = np.abs(signal.hilbert(yb))
    # smooth
    n_smooth = max(1, int(round(smooth_sec*fs)))
    if n_smooth > 1:
        w = np.hanning(n_smooth); w /= w.sum()
        env_s = np.convolve(env, w, mode='same')
    else:
        env_s = env
    z = zscore(env_s)
    # threshold + rising edges
    mask = z >= z_thresh
    on_idx = np.where(np.diff(mask.astype(int)) == 1)[0] + 1
    # min ISI
    onsets = []
    last_t = -np.inf
    for i in on_idx:
        if t[i] - last_t >= min_isi_sec:
            onsets.append(t[i]); last_t = t[i]
    onsets = np.array(onsets, float)

    # --- 2) Build ignition windows (merge close onsets) ---
    ign = []
    last_end = -np.inf
    for s in onsets:
        a = s - window_sec/2.0
        b = s + window_sec/2.0
        if ign and a <= ign[-1][1] + merge_gap_sec:
            ign[-1] = (ign[-1][0], b)
        else:
            ign.append((a, b))
    # clip to recording & keep > 1s
    t0, t1 = float(t[0]), float(t[-1])
    ign = [(max(t0, a), min(t1, b)) for (a, b) in ign if (b - a) > 1.0]

    # --- 2b) NEW: Rounded whole-second ignition windows (+ merge) ---
    # Use floor for starts and ceil for ends to cover the full detected window
    rounded = []
    for a, b in ign:
        sa = int(np.floor(a))
        sb = int(np.ceil(b))
        if sb > sa:
            rounded.append((sa, sb))
    ignition_windows_rounded = _merge_intervals_int(rounded)

    # Persist and optionally print
    ign_json_path = os.path.join(out_dir, 'ignition_windows.json')
    with open(ign_json_path, 'w') as f:
        json.dump(ignition_windows_rounded, f)

    if verbose:
        print(f"Ignition windows (rounded, whole seconds): {ignition_windows_rounded}")
        print(f"Saved → {ign_json_path}")

    # --- 3) Build EEG matrix & R(t) (global synchrony) ---
    X = []
    for ch in eeg_channels:
        X.append(get_series(RECORDZ, ch))
    L = min(map(len, X)); X = np.vstack([x[:L] for x in X]); t = t[:L]
    # full-series R(t)
    t_cent, Rt = kuramoto_R_timeseries(X, fs, band=R_band, win_sec=R_win_sec, step_sec=R_step_sec)
    zR = (Rt - Rt.mean()) / (Rt.std() + 1e-12)

    # --- 4) Per-event validation stats ---
    rows = []
    fC, C = signal.coherence(np.mean(X, axis=0), y[:L], fs=fs, nperseg=int(4*fs), noverlap=int(2*fs))
    for (a, b) in ign:
        # SR z at onset (nearest sample)
        i_on = int(np.argmin(np.abs(t - (a + window_sec/2.0))))
        i_end = int(np.argmin(np.abs(t - (b + window_sec/2.0))))
        z_at_on = float(z[i_on]) if 0 <= i_on < len(z) else np.nan
        # R(t) peak within ±5 s of onset
        t_on = a + window_sec/2.0
        msk = (t_cent >= (t_on - 5.0)) & (t_cent <= (t_on + 5.0))
        zR_peak = float(np.nanmax(zR[msk])) if np.any(msk) else np.nan
        # zR_max over window
        zR_max = float(np.nanmax(z[i_on:i_end]))
        # MSC at 7.83 (nearest Welch bin) within this window
        i0, i1 = int(max(0, round(a*fs))), int(min(L, round(b*fs)))
        if i1 - i0 > int(2*fs):
            fE, CE = signal.coherence(np.mean(X[:, i0:i1], axis=0), y[i0:i1], fs=fs, nperseg=int(2*fs))
            idx = int(np.argmin(np.abs(fE - center_hz)))
            msc_win = float(CE[idx])
        else:
            idx = int(np.argmin(np.abs(fC - center_hz)))
            msc_win = float(C[idx])
        rows.append({'t_start': a, 't_end': b, 'duration_s': float(b-a),
                     'zR_peak_±5s': zR_peak, 'msc_7p83': msc_win, 'zR_max': zR_max})
    events = pd.DataFrame(rows)

    # --- 5) Event-triggered average (ETA) of zR(t) ---
    if len(onsets):
        pre = eta_pre_sec; post = eta_post_sec
        dt_R = np.median(np.diff(t_cent)) if t_cent.size > 1 else R_step_sec
        tau = np.arange(-pre, post + dt_R/2, dt_R)
        ETA = []
        for s in onsets:
            sel = (t_cent >= s - pre) & (t_cent <= s + post)
            if not np.any(sel):
                continue
            z_seg = np.interp(s + tau, t_cent, zR, left=np.nan, right=np.nan)
            ETA.append(z_seg)
        ETA = np.vstack(ETA) if ETA else np.empty((0, len(tau)))
        eta_mean = np.nanmean(ETA, axis=0) if ETA.size else np.full_like(tau, np.nan)
        eta_sem  = (np.nanstd(ETA, axis=0) / np.sqrt(np.sum(np.isfinite(ETA), axis=0))
                    if ETA.size else np.full_like(tau, np.nan))
    else:
        tau = np.array([]); eta_mean = np.array([]); eta_sem = np.array([])

    # Package outputs
    out = {
        'events': events,
        'ignition_windows': ign,
        'ignition_windows_rounded': ignition_windows_rounded,
        'ignition_windows_path': ign_json_path,
        'fs': fs,
        't_R': t_cent,
        'zR': zR,
        'ETA_tau': tau,
        'ETA_mean': eta_mean,
        'ETA_sem': eta_sem,
    }

    # (Optional) additional saving/plotting controlled by `show` could be added here
#     return out


    # --- 6) Plots ---
    # (A) SR envelope with shaded ignitions
    plt.figure(figsize=(11,3))
    plt.plot(pd.to_numeric(RECORDZ[time_col], errors='coerce').values.astype(float)[:len(env)], z, lw=1.0, label='SR env z')
    plt.axhline(z_thresh, color='k', ls='--', lw=1, label='z-thresh')
    for (a,b) in ign: plt.axvspan(a,b, color='tab:orange', alpha=0.15)
    plt.xlabel('Time (s)'); plt.ylabel('SR z'); plt.title('SR envelope z(t) with detected ignitions')
    plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(out_dir,'sr_env_z.png'), dpi=140)
    if show: plt.show();
    plt.close()

    # (B) zR(t) with ignitions
    plt.figure(figsize=(11,3))
    plt.plot(t_cent, zR, lw=1.0, label='zR(t) (global synchrony)')
    for (a,b) in ign: plt.axvspan(a,b, color='tab:orange', alpha=0.15)
    plt.xlabel('Time (s)'); plt.ylabel('zR'); plt.title(f'Global synchrony R(t) in {R_band[0]}–{R_band[1]} Hz')
    plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(out_dir,'R_timeseries.png'), dpi=140)
    if show: plt.show();
    plt.close()

    # (C) ETA of zR around onsets
    if tau.size:
        plt.figure(figsize=(7.5,3))
        plt.plot(tau, eta_mean, lw=1.6, label='mean zR')
        if np.any(np.isfinite(eta_sem)):
            plt.fill_between(tau, eta_mean-eta_sem, eta_mean+eta_sem, alpha=0.2)
        plt.axvline(0, color='k', lw=1)
        plt.xlabel('Time from onset (s)'); plt.ylabel('zR'); plt.title('Event-triggered zR(t)')
        plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(out_dir,'ETA_zR.png'), dpi=140)
        if show: plt.show();
        plt.close()

    # (D) Per-event MSC@7.83 boxplot
    if not events.empty:
        plt.figure(figsize=(4,3))
        plt.boxplot(events['msc_7p83'].dropna(), vert=True)
        plt.ylabel('MSC@~7.83 Hz'); plt.title('Per-event EEG↔SR coherence')
        plt.tight_layout(); plt.savefig(os.path.join(out_dir,'events_msc_box.png'), dpi=140)
        if show: plt.show();
        plt.close()

    # --- 7) Summary CSVs ---
    if events.empty:
        summary = {'n_events':0}
    else:
        summary = {
            'n_events': int(len(events)),
#             'median_duration_s': float(events['duration_s'].median()),
#             'median_sr_z_onset': float(events['sr_z_onset'].median()),
            'median_zR_max': float(events['zR_max'].median()),
            'median_zR_peak': float(events['zR_peak_±5s'].median()),
            'median_msc_7p83': float(events['msc_7p83'].median()),
            'coverage_pct': float(100.0*np.sum(events['duration_s'])/max(1e-9, t[-1]-t[0]))
        }
    events.to_csv(os.path.join(out_dir,'events.csv'), index=False)
    pd.DataFrame([summary]).to_csv(os.path.join(out_dir,'summary.csv'), index=False)

    # ---------- NEW: pretty print summary to the notebook ----------
    if verbose:
        print("\n=== Ignition Detection — Session Summary ===")
        print(f"SR reference: {sr_channel}")
        print(f"EEG channels (n={len(eeg_channels)}): {', '.join([c.split('.',1)[-1] for c in eeg_channels])}")
        print(f"Detection band: {center_hz:.2f}±{half_bw_hz:.2f} Hz; z-thresh={z_thresh:.2f}; "
              f"window={window_sec:.1f}s; min_ISI={min_isi_sec:.1f}s")
        print(f"R(t) band: {R_band[0]:.1f}–{R_band[1]:.1f} Hz, win={R_win_sec:.2f}s, step={R_step_sec:.2f}s")

        def fmt_iqr(x: np.ndarray) -> str:
            if x.size == 0 or np.all(~np.isfinite(x)): return "n/a"
            q1, med, q3 = np.nanpercentile(x, [25, 50, 75])
            return f"{med:.2f} [{q1:.2f}, {q3:.2f}]"

        n_events = summary.get('n_events', 0)
        print(f"\nEvents detected: {n_events}")
        if n_events > 0:
            dur  = events['duration_s'].to_numpy()
#             z_on = events['sr_z_onset'].to_numpy()
            zpk  = events['zR_peak_±5s'].to_numpy()
            msc  = events['msc_7p83'].to_numpy()
            z_max = events['zR_max'].to_numpy()


#             print(f"  Duration (s)           — median [IQR]: {fmt_iqr(dur)}")
#             print(f"  SR z at onset          — median [IQR]: {fmt_iqr(z_on)}")
            print(f"  zR max                 — median [IQR]: {fmt_iqr(z_max)}")
            print(f"  zR peak (±5s)          — median [IQR]: {fmt_iqr(zpk)}")
            print(f"  MSC@~7.83 Hz           — median [IQR]: {fmt_iqr(msc)}")
            print(f"  Coverage of recording  — {summary['coverage_pct']:.2f}%")

            # ETA peak, if computed
            try:
                # eta_mean, tau were defined earlier in your function (ETA_zR section)
                if tau.size and np.any(np.isfinite(eta_mean)):
                    i_max = int(np.nanargmax(eta_mean))
                    print(f"  ETA zR peak            — {eta_mean[i_max]:.2f} at {tau[i_max]:.2f}s")
            except Exception:
                pass

            # Show top 5 by MSC or zR peak for a quick feel
            top_by_msc = events.sort_values('zR_max', ascending=False)
            print("\nEvents by MSC@~7.83 Hz:")
            print(top_by_msc[['t_start','t_end','msc_7p83','zR_max','zR_peak_±5s']]
                  .to_string(index=False, justify='center'))

        print(f"\nFiles written to: {out_dir}")
        print("  - sr_env_z.png, R_timeseries.png, ETA_zR.png, events_msc_box.png")
        print("  - events.csv (per event) and summary.csv (session summary)")

    return {'events': events, 'summary': summary, 'out_dir': out_dir,
            'figs': {'sr_env': os.path.join(out_dir,'sr_env_z.png'),
                     'R_timeseries': os.path.join(out_dir,'R_timeseries.png'),
                     'ETA_zR': os.path.join(out_dir,'ETA_zR.png'),
                     'events_msc_box': os.path.join(out_dir,'events_msc_box.png')}}
