# ===== Directionality across systems & harmonics — self-contained loader =====
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy import signal

# --- helpers ---
def ensure_dir(d):
    if d: os.makedirs(d, exist_ok=True)
    return d

def ensure_timestamp_column(df, time_col='Timestamp', default_fs=128.0):
    if time_col in df.columns:
        s = df[time_col]
        if np.issubdtype(s.dtype, np.datetime64) or 'datetime' in str(s.dtype).lower():
            tsec=(pd.to_datetime(s)-pd.to_datetime(s).iloc[0]).dt.total_seconds().astype(float)
            df[time_col] = tsec.values; return time_col
        sn = pd.to_numeric(s, errors='coerce').astype(float)
        if sn.notna().sum()>max(50,0.5*len(df)):
            sn = sn - np.nanmin(sn[np.isfinite(sn)])
            df[time_col] = sn.values; return time_col
    df[time_col] = np.arange(len(df), dtype=float)/default_fs
    return time_col

def infer_fs(df, time_col='Timestamp'):
    t = np.asarray(df[time_col].values, float)
    dt = np.diff(t); dt = dt[(dt>0)&np.isfinite(dt)]
    if dt.size==0: raise ValueError('Cannot infer fs from time column.')
    return float(1.0/np.median(dt))

def _bandpass(x, fs, lo, hi, order=4):
    ny=0.5*fs; lo=max(1e-6,min(lo,0.99*ny)); hi=max(lo+1e-6,min(hi,0.999*ny))
    b,a=signal.butter(order,[lo/ny, hi/ny],btype='band'); return signal.filtfilt(b,a,x)

def narrowband_pair(x_eeg, x_sr, fs, f0, half):
    lo=max(0.01, f0-half); hi=f0+half
    return _bandpass(x_eeg, fs, lo, hi), _bandpass(x_sr, fs, lo, hi)

def windows_to_samples(wins, fs, N):
    segs=[]
    for (a,b) in (wins or []):
        i0,i1=int(round(a*fs)), int(round(b*fs))
        i0=max(0,i0); i1=min(N,i1)
        if i1>i0: segs.append((i0,i1))
    return segs

def concat_segments(x, segs):
    return np.concatenate([x[i0:i1] for (i0,i1) in segs]) if segs else np.array([])

# --- bivariate MVAR + PDC/GC (LS) ---
def fit_mvar_2d(x, y, p=6):
    X = np.column_stack([x, y]).astype(float)
    N, k = X.shape
    if N <= p: raise ValueError('Too few samples for MVAR')
    Y = X[p:]
    Z = np.hstack([X[p-i:-i] for i in range(1, p+1)])  # (N-p) x (2p)
    B, *_ = np.linalg.lstsq(Z, Y, rcond=None)          # (2p) x 2
    A = [B[2*(i-1):2*i,:].T for i in range(1, p+1)]    # list of 2x2
    res = Y - Z @ B
    Sigma = (res.T @ res) / (len(res) - 1)
    return A, Sigma

def pdc_from_mvar(A_list, fs, f_hz):
    dt = 1.0/fs
    z = np.exp(-1j*2*np.pi*f_hz*dt)
    A_f = np.eye(2, dtype=complex)
    for k, Ak in enumerate(A_list, start=1):
        A_f = A_f - Ak * (z**k)
    denom = np.sqrt(np.sum(np.abs(A_f)**2, axis=0)) + 1e-12
    return np.abs(A_f) / denom  # 2x2

def granger_2d_refit(x, y, p=6):
    # full
    A_full, Sigma_full = fit_mvar_2d(x, y, p=p)
    sig_x_full = float(Sigma_full[0,0]); sig_y_full = float(Sigma_full[1,1])
    # Build design matrices
    X = np.column_stack([x, y]).astype(float)
    N = len(X)
    Y = X[p:]
    Z_full = np.hstack([X[p-i:-i] for i in range(1,p+1)])  # (N-p) x (2p)
    # Restricted x_t: only x lags
    Zx = np.hstack([X[p-i:-i, 0:1] for i in range(1,p+1)])
    bx, *_ = np.linalg.lstsq(Zx, Y[:,0], rcond=None)
    res_x = Y[:,0] - Zx @ bx
    sig_x_restr = float(np.var(res_x, ddof=1))
    # Restricted y_t: only y lags
    Zy = np.hstack([X[p-i:-i, 1:2] for i in range(1,p+1)])
    by, *_ = np.linalg.lstsq(Zy, Y[:,1], rcond=None)
    res_y = Y[:,1] - Zy @ by
    sig_y_restr = float(np.var(res_y, ddof=1))
    F_y_to_x = np.log((sig_x_restr + 1e-12)/(sig_x_full + 1e-12))
    F_x_to_y = np.log((sig_y_restr + 1e-12)/(sig_y_full + 1e-12))
    return F_y_to_x, F_x_to_y, A_full, Sigma_full

# --- FFT segs for bispectrum ---
def _fft_segments(x, fs, nperseg, step):
    w = signal.hann(nperseg, sym=False)
    hop = nperseg - step
    nseg = 1 + max(0, (len(x)-nperseg)//hop)
    Xs=[]
    for i in range(nseg):
        s = i*hop; e = s + nperseg
        seg = x[s:e]
        if len(seg) < nperseg: break
        seg = seg - np.mean(seg)
        X = np.fft.rfft(w*seg, n=nperseg)
        Xs.append(X)
    Xs = np.asarray(Xs)
    freqs = np.fft.rfftfreq(nperseg, d=1.0/fs)
    return freqs, Xs

# --- main orchestrator ---
def analyze_directionality_harmonics(
    RECORDS,
    eeg_channel: str,
    sr_channel: str,
    windows: dict,                 # {'baseline':[(t0,t1),...], 'ignition':[(..)], ...}
    time_col='Timestamp',
    fundamental=7.83,
    harmonics=(14.3,20.8,27.3,33.8,40.3,46.8,53.3,59.8),
    half_bw=0.6,
    mvar_order=6,
    win_sec=10.0,
    step_sec=2.0,
    out_dir='exports_directionality', show=True
):
    """
    (1) TV-Granger/PDC per harmonic, (2) MIMO ARX over harmonic envelopes, (3) bispectral directionality.
    Saves figures and returns a summary dict of DataFrames.
    """
    ensure_dir(out_dir)
    ensure_timestamp_column(RECORDS, time_col=time_col)
    fs = infer_fs(RECORDS, time_col)

    # get channels robustly
    if eeg_channel in RECORDS.columns:
        x_eeg = pd.to_numeric(RECORDS[eeg_channel], errors='coerce').fillna(0.0).values.astype(float)
    elif ('EEG.'+eeg_channel) in RECORDS.columns:
        x_eeg = pd.to_numeric(RECORDS['EEG.'+eeg_channel], errors='coerce').fillna(0.0).values.astype(float)
    else:
        raise ValueError(f"{eeg_channel} not found.")

    if sr_channel in RECORDS.columns:
        x_sr = pd.to_numeric(RECORDS[sr_channel], errors='coerce').fillna(0.0).values.astype(float)
    elif ('EEG.'+sr_channel) in RECORDS.columns:
        x_sr = pd.to_numeric(RECORDS['EEG.'+sr_channel], errors='coerce').fillna(0.0).values.astype(float)
    else:
        raise ValueError(f"{sr_channel} not found.")

    # windows→samples
    W = {k: windows_to_samples(v, fs, len(RECORDS)) for k,v in (windows or {}).items()}

    # ----- (1) TV-Granger / PDC per harmonic -----
    rows=[]
    for fm in harmonics:
        if fm > min(60.0, 0.999*0.5*fs):  # clamp to ≤60 Hz
            continue
        xe_nb, xs_nb = narrowband_pair(x_eeg, x_sr, fs, fm, half_bw)
        win = int(round(win_sec*fs)); step=int(round(step_sec*fs))
        centers = np.arange(win//2, len(xe_nb)-win//2, step, dtype=int)
        gc_sr2eeg=[]; gc_eeg2sr=[]; pdc_sr2eeg=[]; pdc_eeg2sr=[]
        for c in centers:
            sl = slice(c-win//2, c+win//2)
            xw = xe_nb[sl]; yw = xs_nb[sl]
            if np.std(xw)<1e-12 or np.std(yw)<1e-12:
                continue
            try:
                Fy2x, Fx2y, A_full, Sigma = granger_2d_refit(xw, yw, p=mvar_order)
                gc_sr2eeg.append(float(Fy2x)); gc_eeg2sr.append(float(Fx2y))
                P = pdc_from_mvar(A_full, fs, fm)
                pdc_sr2eeg.append(float(P[0,1])); pdc_eeg2sr.append(float(P[1,0]))
            except Exception:
                continue
        for wname, segs in W.items():
            # (simplify) average across all valid centers
            rows.append({'metric':'GC_SR→EEG','window':wname,'f_hz':fm,'value':float(np.nanmean(gc_sr2eeg)) if gc_sr2eeg else np.nan})
            rows.append({'metric':'GC_EEG→SR','window':wname,'f_hz':fm,'value':float(np.nanmean(gc_eeg2sr)) if gc_eeg2sr else np.nan})
            rows.append({'metric':'PDC_SR→EEG','window':wname,'f_hz':fm,'value':float(np.nanmean(pdc_sr2eeg)) if pdc_sr2eeg else np.nan})
            rows.append({'metric':'PDC_EEG→SR','window':wname,'f_hz':fm,'value':float(np.nanmean(pdc_eeg2sr)) if pdc_eeg2sr else np.nan})

        # bar plots per fm
        def _bar_for(metric_prefix):
            labs=list(W.keys())
            y=[float(np.nanmean([r['value'] for r in rows if r['metric']==metric_prefix and r['f_hz']==fm and r['window']==w])) for w in labs]
            fig, ax = plt.subplots(figsize=(6,3.0))
            ax.bar(np.arange(len(y)), y, width=0.6)
            ax.set_xticks(np.arange(len(y))); ax.set_xticklabels(labs)
            ax.set_title(f'{metric_prefix} @ {fm:.2f} Hz'); ax.grid(True, axis='y', alpha=0.25, linestyle=':')
            plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"{metric_prefix.replace('→','to').replace(':','')}_{fm:.2f}Hz.png"), dpi=160)
            if show: plt.show(); plt.close()
        _bar_for('GC_SR→EEG'); _bar_for('GC_EEG→SR'); _bar_for('PDC_SR→EEG'); _bar_for('PDC_EEG→SR')

    dir_df = pd.DataFrame(rows)
    dir_df.to_csv(os.path.join(out_dir, 'directionality_summary.csv'), index=False)

    # ----- (2) MIMO ARX over harmonic envelopes -----
    freqs = [fundamental] + [f for f in harmonics if f<=min(60.0,0.999*0.5*fs)]
    K = len(freqs)
    A_eeg=[]; A_sr=[]
    for f0 in freqs:
        xe, xs = narrowband_pair(x_eeg, x_sr, fs, f0, half_bw)
        A_eeg.append(np.abs(signal.hilbert(xe)))
        A_sr.append(np.abs(signal.hilbert(xs)))
    A_eeg = np.vstack(A_eeg); A_sr = np.vstack(A_sr)  # K x N
    L = 1
    out_rows=[]
    for wname, segs in W.items():
        idxs=[]
        for (i0,i1) in segs: idxs.extend(list(range(i0+L, i1)))
        idxs = np.array(idxs, int)
        if idxs.size < 50: continue
        # SR→EEG
        Y = A_eeg[:, idxs]; X = A_sr[:, idxs-L]
        XX = X @ X.T + 1e-9*np.eye(K); G = (Y @ X.T) @ np.linalg.inv(XX)
        # surrogates for threshold
        rng = np.random.default_rng(31); null_abs=[]
        for _ in range(200):
            Xs = np.zeros_like(X)
            for k in range(K):
                s = int(rng.integers(1, X.shape[1]-1))
                Xs[k] = np.r_[X[k,-s:], X[k,:-s]]
            Gs = (Y @ Xs.T) @ np.linalg.inv(Xs @ Xs.T + 1e-9*np.eye(K))
            null_abs.append(np.abs(Gs))
        thr = np.nanpercentile(np.stack(null_abs,0), 95, axis=0); sig = (np.abs(G) > thr)
        for i in range(K):
            for j in range(K):
                out_rows.append({'window':wname,'direction':'SR→EEG','target_idx':i,'source_idx':j,
                                 'f_target':freqs[i],'f_source':freqs[j],
                                 'G':float(G[i,j]),'absG':float(abs(G[i,j])),'sig':bool(sig[i,j])})
        # EEG→SR
        Y = A_sr[:, idxs]; X = A_eeg[:, idxs-L]
        XX = X @ X.T + 1e-9*np.eye(K); G = (Y @ X.T) @ np.linalg.inv(XX)
        null_abs=[]
        for _ in range(200):
            Xs = np.zeros_like(X)
            for k in range(K):
                s = int(rng.integers(1, X.shape[1]-1))
                Xs[k] = np.r_[X[k,-s:], X[k,:-s]]
            Gs = (Y @ Xs.T) @ np.linalg.inv(Xs @ Xs.T + 1e-9*np.eye(K))
            null_abs.append(np.abs(Gs))
        thr = np.nanpercentile(np.stack(null_abs,0), 95, axis=0); sig = (np.abs(G) > thr)
        for i in range(K):
            for j in range(K):
                out_rows.append({'window':wname,'direction':'EEG→SR','target_idx':i,'source_idx':j,
                                 'f_target':freqs[i],'f_source':freqs[j],
                                 'G':float(G[i,j]),'absG':float(abs(G[i,j])),'sig':bool(sig[i,j])})
    arx_df = pd.DataFrame(out_rows); arx_df.to_csv(os.path.join(out_dir, 'arx_couplings.csv'), index=False)

    # ----- (3) Bispectral directionality -----
    nper = int(max(4*fs, 4096)); step = nper//2
    freqs_fft, Se = _fft_segments(x_eeg, fs, nper, step); _, Ss = _fft_segments(x_sr, fs, nper, step)
    def bin_idx(f): return int(np.argmin(np.abs(freqs_fft - f)))
    rows_bi=[]
    for f0 in [fundamental] + list(harmonics):
        if f0*2 > min(60.0, 0.999*0.5*fs): continue
        i = bin_idx(f0); k = bin_idx(2*f0)
        num_sse = np.mean(Ss[:, i] * Ss[:, i] * np.conj(Se[:, k]))
        num_ess = np.mean(Se[:, i] * Se[:, i] * np.conj(Ss[:, k]))
        B_sse = np.abs(num_sse); B_ess = np.abs(num_ess)
        rng = np.random.default_rng(21); null_diff=[]
        for _ in range(200):
            sh = int(rng.integers(1, len(x_sr)-1))
            xsr = np.r_[x_sr[-sh:], x_sr[:-sh]]
            _, Ssh = _fft_segments(xsr, fs, nper, step)
            num_sse_n = np.mean(Ssh[:, i] * Ssh[:, i] * np.conj(Se[:, k]))
            num_ess_n = np.mean(Se[:, i] * Se[:, i] * np.conj(Ssh[:, k]))
            null_diff.append(np.abs(num_sse_n) - np.abs(num_ess_n))
        null_diff = np.asarray(null_diff, float)
        diff = float(B_sse - B_ess)
        p = float((np.sum(null_diff >= diff) + 1) / (len(null_diff)+1))
        rows_bi.append({'metric':'BISPEC_DIR','f_hz':f0,'B_sse':float(B_sse),'B_ess':float(B_ess),'diff':diff,'p_value':p})
    bi_df = pd.DataFrame(rows_bi); bi_df.to_csv(os.path.join(out_dir, 'bispec_directionality.csv'), index=False)

    # done
    return {'gc_pdc': dir_df, 'arx': arx_df, 'bispec': bi_df, 'out_dir': out_dir}
