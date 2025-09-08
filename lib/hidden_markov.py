"""
Event-related & HMM approaches — simple validity tests & graphs
===============================================================

6a) Schumann-burst ERP/ERSP/ITC (with simple cluster-perm tests)
    • Detect Schumann bursts on a reference channel (7.83 ± half_bw Hz envelope).
    • Time-lock EEG trials to burst onsets; build:
        – ERP (trial-average time-domain)
        – ERSP (time–frequency power via Morlet CWT)
        – ITC (inter-trial coherence)
    • Simple permutation tests:
        – ERP: sign-flip trials → time-wise threshold + max-cluster mass along time.
        – ERSP/ITC: sign-flip trials → TF threshold + TF cluster mass (4-connectivity).
    • Plots: ERP with significant time mask; ERSP & ITC maps with TF clusters.

6b) HMM-like state analysis on EEG spectrograms (GaussianMixture fallback)
    • Build sliding-window band-power features (θ/α/β/γ) from EEG (mean over channels).
    • Fit GaussianMixture (K states) to features; decode state(t).
    • Validate vs Schumann amplitude:
        – Event-triggered state occupancy around Schumann peaks (ETA + null by circular shift)
        – Logistic regression of state transitions vs Schumann amplitude (ROC-AUC + null)

Only depends on NumPy/SciPy/matplotlib/scikit-learn (+NetworkX for clustering).
Copy/paste into your notebook. See usage at bottom.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, List, Tuple, Optional
from scipy import signal
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# ------------------------------- generic helpers -------------------------------

def infer_fs(RECORDS: pd.DataFrame, time_col: str = 'Timestamp') -> float:
    t = np.asarray(pd.to_numeric(RECORDS[time_col], errors='coerce').values, float)
    dt = np.diff(t); dt = dt[(dt > 0) & np.isfinite(dt)]
    if dt.size == 0: raise ValueError("Cannot infer sampling rate from time column.")
    return float(1.0 / np.median(dt))

def get_series(RECORDS: pd.DataFrame, name: str) -> np.ndarray:
    if name in RECORDS.columns:
        x = pd.to_numeric(RECORDS[name], errors='coerce').fillna(0.0).values
        return np.asarray(x, float)
    alt = 'EEG.' + name
    if alt in RECORDS.columns:
        x = pd.to_numeric(RECORDS[alt], errors='coerce').fillna(0.0).values
        return np.asarray(x, float)
    raise ValueError(f"Signal '{name}' not found in RECORDS.")

def bandpass(x: np.ndarray, fs: float, f1: float, f2: float, order: int = 4) -> np.ndarray:
    ny = 0.5*fs
    f1 = max(1e-6, min(f1, 0.99*ny)); f2 = max(f1+1e-6, min(f2, 0.999*ny))
    b,a = signal.butter(order, [f1/ny, f2/ny], btype='band')
    return signal.filtfilt(b,a,x)

def slice_epoch(x: np.ndarray, i0: int, i1: int) -> Optional[np.ndarray]:
    i0 = max(0, i0); i1 = min(len(x), i1)
    if i1 <= i0: return None
    return x[i0:i1]

# --------------------- Schumann burst detection (re-used) ----------------------

def detect_schumann_bursts(RECORDS: pd.DataFrame, sr_channel: str,
                           time_col: str = 'Timestamp',
                           center_hz: float = 7.83, half_bw_hz: float = 0.6,
                           smooth_sec: float = 0.25,
                           thresh_mode: str = 'z', z_thresh: float = 2.5,
                           perc_thresh: float = 95.0,
                           min_isi_sec: float = 2.0) -> Dict[str, object]:
    fs = infer_fs(RECORDS, time_col)
    t = np.asarray(pd.to_numeric(RECORDS[time_col], errors='coerce').values, float)
    y = get_series(RECORDS, sr_channel)
    yb = bandpass(y, fs, center_hz-half_bw_hz, center_hz+half_bw_hz)
    env = np.abs(signal.hilbert(yb))
    # smooth
    n = max(1, int(round(fs*smooth_sec)))
    if n>1:
        w = np.hanning(n); w /= w.sum()
        env = np.convolve(env, w, mode='same')
    # threshold
    if thresh_mode == 'z':
        z = (env - env.mean())/(env.std()+1e-12)
        mask = z >= z_thresh
    else:
        thr = np.percentile(env, perc_thresh)
        mask = env >= thr
    on_idx = np.where(np.diff(mask.astype(int))==1)[0] + 1
    on = []
    last = -np.inf
    for i in on_idx:
        if t[i] - last >= min_isi_sec:
            on.append(t[i]); last = t[i]
    return {'onsets_sec': on, 'env': env, 't': t}

# --------------------------- 6a) ERP / ERSP / ITC ------------------------------

def morlet_cwt(sig: np.ndarray, fs: float, freqs: np.ndarray, w0: float = 6.0) -> np.ndarray:
    """
    Complex Morlet CWT via FFT-convolution.
    Returns array of shape (n_freq, N) with complex coefficients.
    """
    sig = np.asarray(sig, float)
    N = sig.size
    Wx = []

    # Precompute FFT of the (padded) signal once per maximum kernel length
    # We build each kernel with length L depending on f0; use linear convolution length N+L-1
    for f0 in freqs:
        # Build complex Morlet kernel in time
        # duration: a few cycles at f0 (wider at low f to keep frequency resolution)
        dur = max(2.0, 8.0 / f0)                 # seconds
        L = int(np.ceil(dur * fs))
        if L % 2 == 0:
            L += 1
        tt = (np.arange(-(L // 2), L // 2 + 1)) / fs
        sigma_t = w0 / (2 * np.pi * f0)
        mw = np.exp(-0.5 * (tt / sigma_t) ** 2) * np.exp(1j * 2 * np.pi * f0 * tt)
        # zero-mean correction + L2 normalize
        mw = mw - np.mean(mw)
        mw = mw / (np.sqrt(np.sum(np.abs(mw) ** 2)) + 1e-24)

        # Linear convolution via FFT (complex)
        n_lin = N + L - 1
        n_fft = int(2 ** np.ceil(np.log2(n_lin)))   # next power of two
        S = np.fft.fft(sig, n=n_fft)
        H = np.fft.fft(mw,  n=n_fft)
        conv = np.fft.ifft(S * H)[:n_lin]          # complex result

        # center-trim to length N (align kernel center at each t)
        # shift so kernel center aligns with signal sample
        start = (L - 1) // 2
        end = start + N
        Wx.append(conv[start:end])

    return np.array(Wx)  # (n_freq, N) complex


def erp_ersp_itc(RECORDS: pd.DataFrame, eeg_channels: List[str], sr_channel: str,
                 time_col: str = 'Timestamp',
                 win_sec: Tuple[float,float] = (-5.0, 5.0),
                 baseline_sec: Tuple[float,float] = (-4.0,-1.0),
                 center_hz: float = 7.83, half_bw_hz: float = 0.6,
                 detect_kwargs: Dict = None,
                 fmin: float = 4.0, fmax: float = 40.0, n_freq: int = 48,
                 w0: float = 6.0,
                 n_perm: int = 200, alpha: float = 0.05,
                 show: bool = True) -> Dict[str, object]:
    """
    Build ERP/ERSP/ITC time-locked to Schumann bursts and run simple cluster-perm tests.
    Returns dict with curves/maps and significance masks.
    """
    detect_kwargs = detect_kwargs or {}
    fs = infer_fs(RECORDS, time_col)
    det = detect_schumann_bursts(RECORDS, sr_channel, time_col=time_col,
                                 center_hz=center_hz, half_bw_hz=half_bw_hz, **detect_kwargs)
    onsets = det['onsets_sec']
    if len(onsets)==0:
        raise ValueError("No Schumann bursts detected. Loosen threshold or check channel.")

    # build trials for each EEG channel -> ERP first (avg all channels at the end)
    t_axis = np.arange(int(win_sec[0]*fs), int(win_sec[1]*fs)) / fs
    trials = []        # (n_trials, n_time)
    tf_trials = []     # list of (n_trials, n_freq, n_time) per channel (we'll average across channels)
    freqs = np.exp(np.linspace(np.log(fmin), np.log(fmax), n_freq))
    for ch in eeg_channels:
        x = get_series(RECORDS, ch)
        # time-lock segments
        segs=[]
        tf_segs=[]
        for on in onsets:
            i_on = int(round(on*fs))
            i0 = i_on + int(round(win_sec[0]*fs))
            i1 = i_on + int(round(win_sec[1]*fs))
            seg = slice_epoch(x, i0, i1)
            if seg is None or len(seg) != len(t_axis):
                continue
            segs.append(seg)
            # TF (power)
            W = morlet_cwt(seg, fs, freqs, w0=w0)
            tf_segs.append(np.abs(W)**2)
        if segs:
            arr = np.vstack(segs)                  # (n_trials, n_time)
            tf_arr = np.stack(tf_segs, axis=0)     # (n_trials, n_freq, n_time)
            trials.append(arr)
            tf_trials.append(tf_arr)

    if not trials:
        raise ValueError("No valid trials formed (edge effects or too few bursts).")
    # average across channels → trial x time (ERP) and trial x freq x time (ERSP base)
    ERP_trials = np.nanmean(np.stack(trials, axis=0), axis=0)            # (n_trials, n_time)
    ERSP_trials = np.nanmean(np.stack(tf_trials, axis=0), axis=0)        # (n_trials, n_freq, n_time)

    # baseline correction for ERSP (dB)
    bsel = (t_axis>=baseline_sec[0]) & (t_axis<=baseline_sec[1])
    ERSP_db = 10*np.log10(ERSP_trials / (np.nanmean(ERSP_trials[:,:,bsel], axis=2, keepdims=True)+1e-24))
    # ITC = |mean(W/|W|)| across trials (use channel-avg phase: we already averaged power across channels;
    # compute ITC from first channel's complex CWT to keep it simple)
    # Use the first channel's TF trials to get phases:
    Wphase = np.stack(tf_trials, axis=0)[0]  # (n_trials, n_freq, n_time) power, not phase — recompute phases from one channel
    # recompute from first EEG channel to keep code consistent:
    x0 = get_series(RECORDS, eeg_channels[0])
    W_trials=[]
    for on in onsets[:ERP_trials.shape[0]]:
        i_on = int(round(on*fs))
        seg = slice_epoch(x0, i_on+int(round(win_sec[0]*fs)), i_on+int(round(win_sec[1]*fs)))
        if seg is None or len(seg)!=len(t_axis): continue
        W_trials.append(morlet_cwt(seg, fs, freqs, w0=w0))
    W_trials = np.stack(W_trials, axis=0)           # (n_trials, n_freq, n_time)
    ITC = np.abs(np.nanmean(W_trials/np.maximum(np.abs(W_trials),1e-24), axis=0))  # (n_freq, n_time)

    # ---------------- permutation tests ----------------
    rng = np.random.default_rng(11)
    # ERP sign-flip → time threshold + cluster mass (1D)
    mean_erp = np.nanmean(ERP_trials, axis=0)
    null_erp=[]
    for _ in range(n_perm):
        signs = rng.choice([-1,1], size=ERP_trials.shape[0])
        null_erp.append(np.nanmean(signs[:,None]*ERP_trials, axis=0))
    null_erp = np.stack(null_erp, axis=0)
    thr_erp = np.nanpercentile(null_erp, 100*(1-alpha), axis=0)
    sig_time = mean_erp > thr_erp
    # cluster mass (1D)
    max_mass=0.0; cur=0.0
    for i,flag in enumerate(sig_time):
        if flag: cur += mean_erp[i]
        else: max_mass=max(max_mass,cur); cur=0.0
    erp_mass = max(max_mass, cur)
    null_mass=[]
    for p in null_erp:
        cur=0.0; mm=0.0
        for i in range(len(p)):
            if p[i]>thr_erp[i]:
                cur+=p[i]; mm=max(mm,cur)
            else:
                cur=0.0
        null_mass.append(mm)
    erp_sig = erp_mass >= np.nanpercentile(null_mass, 95)

    # ERSP/ITC TF clustering (4-connectivity, one-sided)
    # ERSP: positive deviations (power increases)
    ERSP_mean = np.nanmean(ERSP_db, axis=0)         # (n_freq, n_time)
    # null by sign-flip trials
    null_tf=[]
    for _ in range(n_perm):
        signs = rng.choice([-1,1], size=ERSP_db.shape[0])[:,None,None]
        null_tf.append(np.nanmean(signs*ERSP_db, axis=0))
    null_tf = np.stack(null_tf, axis=0)
    thr_tf = np.nanpercentile(null_tf, 100*(1-alpha), axis=0)  # per-TF threshold
    tf_sig = ERSP_mean > thr_tf

    # cluster mass on TF grid
    G = nx.grid_2d_graph(tf_sig.shape[0], tf_sig.shape[1])
    def max_cluster_mass(mask, value_map):
        mask_idx = set(zip(*np.where(mask)))
        visited=set(); best=0.0
        for node in list(mask_idx):
            if node in visited: continue
            stack=[node]; mass=0.0
            while stack:
                u=stack.pop()
                if u in visited or u not in mask_idx: continue
                visited.add(u); mass += float(value_map[u[0], u[1]])
                for v in G.neighbors(u):
                    if v in mask_idx and v not in visited:
                        stack.append(v)
            best=max(best,mass)
        return best
    ersp_mass = max_cluster_mass(tf_sig, ERSP_mean)
    # null TF cluster mass
    null_mass_tf=[]
    for p in null_tf:
        null_mass_tf.append(max_cluster_mass(p>thr_tf, p))
    ersp_sig = ersp_mass >= np.nanpercentile(null_mass_tf, 95)

    # ITC: same procedure
    # Build null by random sign of trials' unit-phase (approximate)
    # (We already averaged trials → approximate null by time-circular shift of phase map)
    def circ_shift_2d(A, sh0, sh1):
        return np.roll(np.roll(A, sh0, axis=0), sh1, axis=1)
    null_itc=[]
    for _ in range(n_perm):
        sh0 = int(rng.integers(1, ITC.shape[0]-1))
        sh1 = int(rng.integers(1, ITC.shape[1]-1))
        null_itc.append(circ_shift_2d(ITC, sh0, sh1))
    null_itc = np.stack(null_itc, axis=0)
    thr_itc = np.nanpercentile(null_itc, 100*(1-alpha), axis=0)
    itc_sig = ITC > thr_itc
    itc_mass = max_cluster_mass(itc_sig, ITC)
    itc_sig_global = itc_mass >= np.nanpercentile([max_cluster_mass(n>thr_itc, n) for n in null_itc], 95)

    # ---------------- plots ----------------
    if show:
        # ERP
        plt.figure(figsize=(9,3))
        plt.plot(t_axis, mean_erp, lw=1.8, label='ERP (avg EEG)')
        plt.fill_between(t_axis, 0, mean_erp, where=sig_time, color='tab:red', alpha=0.25, step='pre', label='sig (time-wise)')
        plt.axvline(0, color='k', lw=1); plt.axhline(0, color='k', lw=0.5, alpha=0.4)
        plt.title(f"Schumann-locked ERP  (cluster-sig={erp_sig})"); plt.xlabel('Time (s)'); plt.ylabel('uV (a.u.)')
        plt.legend(); plt.tight_layout(); plt.show()

        # ERSP
        plt.figure(figsize=(9,3.2))
        extent = [t_axis[0], t_axis[-1], freqs[0], freqs[-1]]
        plt.imshow(ERSP_mean, aspect='auto', origin='lower', extent=extent, cmap='magma')
        plt.colorbar(label='ERSP (dB vs baseline)')
        # overlay significant TF mask
        yy, xx = np.where(tf_sig)
        plt.scatter(t_axis[xx], freqs[yy], s=2, c='cyan', alpha=0.6, label='sig TF')
        plt.title(f"Schumann-locked ERSP (TF cluster-sig={ersp_sig})")
        plt.xlabel('Time (s)'); plt.ylabel('Frequency (Hz)'); plt.legend(loc='upper right', fontsize=8)
        plt.tight_layout(); plt.show()

        # ITC
        plt.figure(figsize=(9,3.2))
        plt.imshow(ITC, aspect='auto', origin='lower', extent=extent, cmap='viridis', vmin=0, vmax=1)
        plt.colorbar(label='ITC')
        yi, xi = np.where(itc_sig)
        plt.scatter(t_axis[xi], freqs[yi], s=2, c='white', alpha=0.7, label='sig TF')
        plt.title(f"Schumann-locked ITC (TF cluster-sig={itc_sig_global})")
        plt.xlabel('Time (s)'); plt.ylabel('Frequency (Hz)'); plt.legend(loc='upper right', fontsize=8)
        plt.tight_layout(); plt.show()

    return {'t': t_axis, 'freqs': freqs,
            'ERP_mean': mean_erp, 'ERP_sig_time': sig_time, 'ERP_cluster_sig': erp_sig,
            'ERSP_mean_db': ERSP_mean, 'ERSP_sig_tf': tf_sig, 'ERSP_cluster_sig': ersp_sig,
            'ITC': ITC, 'ITC_sig_tf': itc_sig, 'ITC_cluster_sig': itc_sig_global}

# --------------------------- 6b) HMM-like state analysis -----------------------

def bandpower_features(RECORDS: pd.DataFrame, eeg_channels: List[str],
                       time_col: str = 'Timestamp',
                       bands: Dict[str, Tuple[float,float]] = None,
                       win_sec: float = 2.0, step_sec: float = 0.25) -> Dict[str, object]:
    """
    Sliding-window mean band power per band, averaged over EEG channels.
    Returns {'t': t_centers, 'X': feature_matrix (T, nbands)}.
    """
    bands = bands or {'theta':(4,8),'alpha':(8,13),'beta':(13,30),'gamma':(30,80)}
    fs = infer_fs(RECORDS, time_col)
    Xsig = [get_series(RECORDS, ch) for ch in eeg_channels]
    Xsig = np.vstack(Xsig)  # (n_ch, N)
    N = Xsig.shape[1]
    win = int(round(win_sec*fs)); step = int(round(step_sec*fs))
    centers = np.arange(win//2, N - win//2, step)
    feats=[]
    for c in centers:
        s = c - win//2; e = c + win//2
        seg = Xsig[:, s:e]                      # (n_ch, win)
        row=[]
        for (f1,f2) in bands.values():
            bp = []
            for ch in range(seg.shape[0]):
                xb = bandpass(seg[ch], fs, f1, f2)
                bp.append(np.mean(xb**2))
            row.append(np.mean(bp))
        feats.append(row)
    feats = np.asarray(feats)     # (T, nbands)
    return {'t': centers/fs, 'X': feats, 'bands': list(bands.keys())}

def schumann_amplitude(RECORDS: pd.DataFrame, sr_channel: str,
                       time_col: str = 'Timestamp',
                       center_hz: float = 7.83, half_bw_hz: float = 0.6,
                       win_sec: float = 2.0, step_sec: float = 0.25) -> Dict[str, object]:
    """Sliding-window Schumann envelope mean in the same grid as bandpower_features."""
    fs = infer_fs(RECORDS, time_col)
    y = get_series(RECORDS, sr_channel)
    env = np.abs(signal.hilbert(bandpass(y, fs, center_hz-half_bw_hz, center_hz+half_bw_hz)))
    N = len(env); win = int(round(win_sec*fs)); step = int(round(step_sec*fs))
    centers = np.arange(win//2, N-win//2, step)
    amp=[]
    for c in centers:
        s=c-win//2; e=c+win//2
        amp.append(np.mean(env[s:e]))
    return {'t': centers/fs, 'amp': np.asarray(amp)}

def hmm_states_gmm(features: np.ndarray, K: int = 3, random_state: int = 0) -> Dict[str, object]:
    """Fit GaussianMixture as a simple HMM surrogate; decode state(t)."""
    gmm = GaussianMixture(n_components=K, covariance_type='full', random_state=random_state)
    gmm.fit(features)
    gamma = gmm.predict_proba(features)        # (T, K)
    z = np.argmax(gamma, axis=1)               # hard states
    return {'model': gmm, 'states': z, 'post': gamma}

def eta_state_occupancy(states: np.ndarray, T: np.ndarray,
                        event_times: np.ndarray, span_sec: float = 10.0, n_states: int = None) -> Dict[str, object]:
    """Event-triggered state occupancy around event_times (ETA)."""
    if n_states is None: n_states = int(states.max()+1)
    # time index mapping
    tau = np.arange(-span_sec, span_sec, np.median(np.diff(T)))
    occ = np.zeros((n_states, tau.size))
    counts = np.zeros(tau.size)
    for et in event_times:
        i0 = np.argmin(np.abs(T - et))
        # build relative samples
        for k,dt in enumerate(tau):
            idx = i0 + int(round(dt / np.median(np.diff(T))))
            if 0 <= idx < len(states):
                s = states[idx]
                occ[s, k] += 1
                counts[k] += 1
    occ = occ / np.maximum(counts, 1e-12)
    return {'tau': tau, 'occ': occ, 'counts': counts}

def logistic_state_transition_vs_amp(states: np.ndarray, amp: np.ndarray) -> Dict[str, float]:
    """Binary transition Y: 1 if state changes at t+1; regress on Schumann amp(t)."""
    y = (np.diff(states) != 0).astype(int)
    X = amp[:-1][:,None]
    if np.all(y==y[0]):
        return {'auc': np.nan, 'coef': np.nan}
    clf = LogisticRegression(class_weight='balanced', max_iter=1000)
    clf.fit(X, y)
    p = clf.predict_proba(X)[:,1]
    auc = roc_auc_score(y, p)
    return {'auc': float(auc), 'coef': float(clf.coef_[0,0])}

def run_hmm_state_tests(RECORDS: pd.DataFrame, eeg_channels: List[str], sr_channel: str,
                        time_col: str = 'Timestamp',
                        K: int = 3, bands: Dict[str, Tuple[float,float]] = None,
                        win_sec: float = 2.0, step_sec: float = 0.25,
                        span_sec: float = 10.0,
                        peak_perc: float = 95.0,
                        n_perm: int = 200, rng_seed: int = 23,
                        show: bool = True) -> Dict[str, object]:
    """
    1) Build band-power features; fit GMM(K) → state(t).
    2) Build Schumann amplitude; find peaks (> percentile).
    3) ETA of state occupancies around peaks + circular-shift null.
    4) Logistic regression: state transitions vs Schumann amplitude (AUC + null).
    """
    feat = bandpower_features(RECORDS, eeg_channels, time_col=time_col, bands=bands,
                              win_sec=win_sec, step_sec=step_sec)
    amp  = schumann_amplitude(RECORDS, sr_channel, time_col=time_col,
                              center_hz=7.83, half_bw_hz=0.6,
                              win_sec=win_sec, step_sec=step_sec)

    # align to common time grid
    t  = feat['t']; X = feat['X']
    ta = amp['t'];  A = amp['amp']
    if len(ta) != len(t):
        # interpolate Schumann amp to feature grid
        A = np.interp(t, ta, A)

    hmm = hmm_states_gmm(X, K=K, random_state=0)
    z = hmm['states']

    # peak events
    thr = np.percentile(A, peak_perc)
    peaks = np.where((A[1:-1]>A[:-2]) & (A[1:-1]>A[2:]) & (A[1:-1]>=thr))[0] + 1
    events = t[peaks]

    eta = eta_state_occupancy(z, t, events, span_sec=span_sec, n_states=K)
    tau, occ = eta['tau'], eta['occ']

    # ETA null by circular shift of A
    rng = np.random.default_rng(rng_seed)
    null_occ = []
    for _ in range(n_perm):
        s = int(rng.integers(1, len(A)-1))
        Ap = np.r_[A[-s:], A[:-s]]
        pk = np.where((Ap[1:-1]>Ap[:-2]) & (Ap[1:-1]>Ap[2:]) & (Ap[1:-1]>=thr))[0] + 1
        ev = t[pk]
        et = eta_state_occupancy(z, t, ev, span_sec=span_sec, n_states=K)
        null_occ.append(et['occ'])
    null_occ = np.stack(null_occ, axis=0)   # (n_perm, K, Ttau)

    # summarize: max occupancy boost per state vs null 95%
    obs_boost = np.max(occ - np.nanmean(null_occ, axis=0), axis=1)  # (K,)
    thr95 = np.nanpercentile(np.max(null_occ - np.nanmean(null_occ, axis=0), axis=2), 95, axis=0)  # (K,)

    # logistic regression: transitions vs A
    lr = logistic_state_transition_vs_amp(z, A)
    # null AUC by circular shift
    auc_null=[]
    for _ in range(n_perm):
        s = int(rng.integers(1, len(A)-1))
        Ap = np.r_[A[-s:], A[:-s]]
        lr0 = logistic_state_transition_vs_amp(z, Ap)
        if not np.isnan(lr0['auc']):
            auc_null.append(lr0['auc'])
    auc_thr95 = np.nanpercentile(auc_null, 95) if auc_null else np.nan

    if show:
        # plot occupancy ETAs
        plt.figure(figsize=(9, 3.2))
        for k in range(K):
            plt.plot(tau, occ[k], lw=1.6, label=f'State {k}')
        plt.axvline(0, color='k', lw=1)
        plt.title('ETA: state occupancy around Schumann peaks')
        plt.xlabel('Time (s)'); plt.ylabel('Occupancy')
        plt.legend(); plt.tight_layout(); plt.show()

        # bar: max boost vs null
        plt.figure(figsize=(6,3))
        x = np.arange(K)
        plt.bar(x, obs_boost, color='tab:blue', alpha=0.9)
        for i,thr in enumerate(thr95):
            plt.plot([i-0.35,i+0.35],[thr,thr],'k--',lw=1)
        plt.xticks(x, [f'S{k}' for k in range(K)])
        plt.ylabel('Max occupancy boost'); plt.title('ETA boost vs null (95% dashed)')
        plt.tight_layout(); plt.show()

        # logistic regression summary
        print(f"LogReg transitions ~ Schumann amp: AUC={lr['auc']:.3f}, coef={lr['coef']:.3f}, null95={auc_thr95:.3f}")

    return {'t': t, 'states': z, 'post': hmm['post'], 'amp': A,
            'events': events,
            'eta_tau': tau, 'eta_occ': occ, 'eta_null': null_occ,
            'eta_boost': obs_boost, 'eta_boost_thr95': thr95,
            'lr_auc': lr['auc'], 'lr_coef': lr['coef'], 'lr_auc_thr95': auc_thr95}

def erp_ersp_itc_safe(RECORDS, eeg_channels, sr_channel,
                      time_col='Timestamp',
                      win_sec=(-3.0, 3.0),           # shorter default
                      baseline_sec=(-2.0, -0.5),     # inside window
                      center_hz=7.83, half_bw_hz=0.6,
                      detect_kwargs=None,
                      fmin=4.0, fmax=40.0, n_freq=48, w0=6.0,
                      n_perm=200, alpha=0.05,
                      edge_policy='pad',              # 'pad' or 'drop'
                      pad_mode='reflect',             # or 'constant'
                      show=True):
    """Schumann-locked ERP/ERSP/ITC with edge padding and TF cluster-perm."""
    detect_kwargs = detect_kwargs or {}
    fs = infer_fs(RECORDS, time_col)
    det = detect_schumann_bursts(RECORDS, sr_channel, time_col=time_col,
                                 center_hz=center_hz, half_bw_hz=half_bw_hz, **detect_kwargs)
    onsets = np.array(det['onsets_sec'], float)
    if onsets.size == 0:
        raise ValueError("No Schumann bursts detected.")

    t_axis = np.arange(int(win_sec[0]*fs), int(win_sec[1]*fs)) / fs
    L = len(t_axis)
    freqs = np.exp(np.linspace(np.log(fmin), np.log(fmax), n_freq))

    # helper: pad segment to length L
    def take_segment(x, i_on):
        i0 = i_on + int(round(win_sec[0]*fs))
        i1 = i_on + int(round(win_sec[1]*fs))
        if 0 <= i0 and i1 <= len(x):
            seg = x[i0:i1]
            if len(seg) != L: return None
            return seg
        if edge_policy == 'drop':
            return None
        # pad
        left = max(0, -i0)
        right = max(0, i1 - len(x))
        s = max(i0, 0); e = min(i1, len(x))
        seg = x[s:e]
        if left or right:
            seg = np.pad(seg, (left, right), mode=pad_mode)
        if len(seg) != L:  # last resort
            return None
        return seg

    # collect trials per channel
    trials_by_ch = []
    cwt_by_ch   = []
    for ch in eeg_channels:
        x = get_series(RECORDS, ch)
        segs=[]; cplx=[]
        for on in onsets:
            i_on = int(round(on*fs))
            seg = take_segment(x, i_on)
            if seg is None: continue
            seg_hilb = morlet_cwt(seg, fs, freqs, w0=w0)   # (n_freq, L) complex
            segs.append(seg)
            cplx.append(seg_hilb)
        if segs:
            trials_by_ch.append(np.vstack(segs))                   # (n_trials, L)
            cwt_by_ch.append(np.stack(cplx, axis=0))               # (n_trials, n_freq, L)

    if not trials_by_ch:
        raise ValueError("No valid trials formed after padding.")
    # average across channels
    ERP_trials  = np.nanmean(np.stack(trials_by_ch, axis=0), axis=0)      # (n_trials, L)
    CWT_trials  = np.nanmean(np.stack(cwt_by_ch,   axis=0), axis=0)       # (n_trials, n_freq, L)
    ERSP_trials = np.abs(CWT_trials)**2                                   # power

    # ERSP baseline (dB)
    bsel = (t_axis >= baseline_sec[0]) & (t_axis <= baseline_sec[1])
    ERSP_db = 10*np.log10(ERSP_trials / (np.nanmean(ERSP_trials[:,:,bsel], axis=2, keepdims=True)+1e-24))

    # ITC = |mean(exp(i*phase))| across trials
    phases = CWT_trials / np.maximum(np.abs(CWT_trials), 1e-24)
    ITC = np.abs(np.nanmean(phases, axis=0))                              # (n_freq, L)

    # ERP permutation (sign-flip)
    mean_erp = np.nanmean(ERP_trials, axis=0)
    rng = np.random.default_rng(11)
    null_erp=[]
    for _ in range(n_perm):
        signs = rng.choice([-1,1], size=ERP_trials.shape[0])[:,None]
        null_erp.append(np.nanmean(signs*ERP_trials, axis=0))
    null_erp = np.stack(null_erp, axis=0)
    thr_erp  = np.nanpercentile(null_erp, 100*(1-alpha), axis=0)
    sig_time = mean_erp > thr_erp

    # ERSP TF cluster
    ERSP_mean = np.nanmean(ERSP_db, axis=0)                               # (n_freq, L)
    null_tf=[]
    for _ in range(n_perm):
        signs = rng.choice([-1,1], size=ERSP_db.shape[0])[:,None,None]
        null_tf.append(np.nanmean(signs*ERSP_db, axis=0))
    null_tf = np.stack(null_tf, axis=0)
    thr_tf  = np.nanpercentile(null_tf, 100*(1-alpha), axis=0)
    tf_sig  = ERSP_mean > thr_tf

    # cluster mass (4-connectivity)
    G = nx.grid_2d_graph(*ERSP_mean.shape)
    def max_cluster_mass(mask, val):
        idx = set(zip(*np.where(mask))); seen=set(); best=0.0
        for u in list(idx):
            if u in seen: continue
            stack=[u]; mass=0.0
            while stack:
                v = stack.pop()
                if v in seen or v not in idx: continue
                seen.add(v); mass += float(val[v[0], v[1]])
                for w in G.neighbors(v):
                    if w in idx and w not in seen:
                        stack.append(w)
            best = max(best, mass)
        return best
    ersp_mass = max_cluster_mass(tf_sig, ERSP_mean)
    null_mass = [max_cluster_mass(p>thr_tf, p) for p in null_tf]
    ersp_sig  = ersp_mass >= np.nanpercentile(null_mass, 95)

    # ITC TF cluster
    null_itc=[]
    for _ in range(n_perm):
        sh0 = int(rng.integers(1, ITC.shape[0]-1)); sh1 = int(rng.integers(1, ITC.shape[1]-1))
        null_itc.append(np.roll(np.roll(ITC, sh0, axis=0), sh1, axis=1))
    null_itc = np.stack(null_itc, axis=0)
    thr_itc  = np.nanpercentile(null_itc, 100*(1-alpha), axis=0)
    itc_sig  = ITC > thr_itc
    itc_mass = max_cluster_mass(itc_sig, ITC)
    itc_sig_global = itc_mass >= np.nanpercentile([max_cluster_mass(n>thr_itc, n) for n in null_itc], 95)

    # plots (same as before)
    if show:
        plt.figure(figsize=(9,3))
        plt.plot(t_axis, mean_erp, lw=1.8, label='ERP (avg EEG)')
        plt.fill_between(t_axis, 0, mean_erp, where=sig_time, color='tab:red', alpha=0.25, step='pre', label='sig (time-wise)')
        plt.axvline(0, color='k', lw=1); plt.axhline(0, color='k', lw=0.5, alpha=0.4)
        plt.title(f"Schumann-locked ERP  (cluster-sig={ersp_sig})"); plt.xlabel('Time (s)'); plt.ylabel('uV (a.u.)')
        plt.legend(); plt.tight_layout(); plt.show()

        extent = [t_axis[0], t_axis[-1], freqs[0], freqs[-1]]
        plt.figure(figsize=(9,3.2))
        plt.imshow(ERSP_mean, aspect='auto', origin='lower', extent=extent, cmap='magma')
        plt.colorbar(label='ERSP (dB vs baseline)')
        yx = np.where(tf_sig); plt.scatter(t_axis[yx[1]], freqs[yx[0]], s=2, c='cyan', alpha=0.6, label='sig TF')
        plt.title(f"Schumann-locked ERSP (TF cluster-sig={ersp_sig})")
        plt.xlabel('Time (s)'); plt.ylabel('Frequency (Hz)'); plt.legend(loc='upper right', fontsize=8); plt.tight_layout(); plt.show()

        plt.figure(figsize=(9,3.2))
        plt.imshow(ITC, aspect='auto', origin='lower', extent=extent, cmap='viridis', vmin=0, vmax=1)
        plt.colorbar(label='ITC')
        yi, xi = np.where(itc_sig)
        plt.scatter(t_axis[xi], freqs[yi], s=2, c='white', alpha=0.7, label='sig TF')
        plt.title(f"Schumann-locked ITC (TF cluster-sig={itc_sig_global})")
        plt.xlabel('Time (s)'); plt.ylabel('Frequency (Hz)'); plt.legend(loc='upper right', fontsize=8)
        plt.tight_layout(); plt.show()

    return {'t': t_axis, 'freqs': freqs,
            'ERP_mean': mean_erp, 'ERP_sig_time': sig_time,
            'ERSP_mean_db': ERSP_mean, 'ERSP_sig_tf': tf_sig, 'ERSP_cluster_sig': ersp_sig,
            'ITC': ITC, 'ITC_sig_tf': itc_sig, 'ITC_cluster_sig': itc_sig_global}
