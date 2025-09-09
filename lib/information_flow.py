"""
Directionality & Information Flow (stand-alone)

4a) Frequency-domain Granger / Partial Directed Coherence (PDC) / DTF
    • Fit VAR (bivariate or multivariate) on EEG + Schumann reference
    • Diagnostics: order selection (AIC/BIC), stability (roots<1), residual whiteness (Ljung-Box)
    • Spectral DTF/PDC and (optional) time-domain Granger tests
    • Report values at Schumann harmonics (≈7.83, 14.3, 20.8, 27.3, 33.8 Hz)

4b) Transfer Entropy (TE) / Conditional TE (lag-resolved)
    • kNN (Kraskov-style) estimator for TE X→Y at specified lags
    • Surrogate significance via circular time-shift

4c) Time-varying (state-space flavored) AR “Kalman-RLS”
    • Recursive least-squares with forgetting to track AR coefficients over time
    • Extract directed coupling gains X→Y(t), Y→X(t) and a time-varying DTF at 7.83 Hz

Assumes a pandas.DataFrame RECORDS with a time column (default 'Timestamp') and
columns like 'EEG.O1', 'EEG.O2', or a Schumann reference (EEG or magnetometer).
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import signal, stats
import matplotlib.pyplot as plt
from scipy.special import gamma as Gamma, digamma
from sklearn.neighbors import NearestNeighbors

import statsmodels.api as sm
from statsmodels.tsa.api import VAR


# optional: VAR & Ljung-Box from statsmodels
try:
    from statsmodels.tsa.api import VAR
    from statsmodels.stats.diagnostic import acorr_ljungbox
    _HAS_SM = True
except Exception:
    _HAS_SM = False

# ------------------ generic helpers ------------------

def infer_fs(RECORDS: pd.DataFrame, time_col: str = 'Timestamp') -> float:
    t = np.asarray(pd.to_numeric(RECORDS[time_col], errors='coerce').values, float)
    dt = np.diff(t); dt = dt[(dt > 0) & np.isfinite(dt)]
    if dt.size == 0:
        raise ValueError("Cannot infer sampling rate from time column.")
    return float(1.0 / np.median(dt))

def get_series(RECORDS: pd.DataFrame, name: str) -> np.ndarray:
    """Return numeric signal. Accepts 'EEG.O1' or bare 'O1' (tries 'EEG.O1')."""
    if name in RECORDS.columns:
        x = pd.to_numeric(RECORDS[name], errors='coerce').fillna(0.0).values
        return np.asarray(x, float)
    alt = 'EEG.' + name
    if alt in RECORDS.columns:
        x = pd.to_numeric(RECORDS[alt], errors='coerce').fillna(0.0).values
        return np.asarray(x, float)
    raise ValueError(f"Signal '{name}' not found.")

def slice_concat(x: np.ndarray, fs: float, windows: Optional[List[Tuple[float,float]]]) -> np.ndarray:
    if not windows: return x.copy()
    segs=[]; n=len(x)
    for (t0,t1) in windows:
        i0,i1 = int(round(t0*fs)), int(round(t1*fs))
        i0=max(0,i0); i1=min(n,i1)
        if i1>i0: segs.append(x[i0:i1])
    return np.concatenate(segs) if segs else x.copy()

def stack_channels(RECORDS: pd.DataFrame, channels: List[str],
                   fs: float, windows: Optional[List[Tuple[float,float]]],
                   demean: bool=True) -> np.ndarray:
    X = []
    for ch in channels:
        x = get_series(RECORDS, ch)
        x = slice_concat(x, fs, windows)
        if demean: x = x - np.mean(x)
        X.append(x)
    # truncate to min length
    L = min(map(len, X))
    X = np.vstack([x[:L] for x in X])  # (n_ch, L)
    return X

# ------------------ 4a) VAR / DTF / PDC ------------------

def fit_var_model(X: np.ndarray,
                  order_max: int = 20,
                  crit: str = 'bic') -> Dict[str, object]:
    """
    Fit VAR to (n_ch, L) array X.T using statsmodels.
    Returns chosen order p, A matrices (p, n_ch, n_ch), noise cov Sigma_u,
    stability flag, Ljung-Box p-values (per channel), and model object (if available).
    """
    if not _HAS_SM:
        raise RuntimeError("statsmodels is required for VAR fitting.")

    Y = X.T  # (L, n_ch)
    model = VAR(Y)

    # --- order selection: support both attribute and dict-like outputs
    sel = model.select_order(maxlags=order_max)
    if hasattr(sel, crit):
        p = int(getattr(sel, crit))
    else:
        # older statsmodels returns a dict-like
        p = int(sel[crit])

    # fit VAR(p)
    res = model.fit(p)
    A = np.array(res.coefs)            # shape (p, n_ch, n_ch)
    Sigma_u = np.array(res.sigma_u)    # (n_ch, n_ch)

    # stability: all roots inside unit circle
    stable = bool(np.all(np.abs(res.roots) < 1.0))

    # --- residual whiteness: Ljung–Box, compatible with old/new statsmodels
    def _lb_last_pvalue(x: np.ndarray, max_lag: int) -> float:
        lag = max(1, min(20, max_lag))
        try:
            # newer API (return_df=True)
            lb_df = acorr_ljungbox(x, lags=[lag], return_df=True)
            return float(lb_df['lb_pvalue'].iloc[-1])
        except TypeError:
            # older API: returns (stat, pvalue)
            stat, p = acorr_ljungbox(x, lags=lag)
            # ensure we return the last lag's p-value
            p = np.atleast_1d(p)
            return float(p[-1])

    resid = res.resid  # (L-p, n_ch)
    lb_pvals = []
    for j in range(resid.shape[1]):
        lb_pvals.append(_lb_last_pvalue(resid[:, j], max_lag=len(resid)//5))

    return {
        'order': p,
        'A': A,
        'Sigma_u': Sigma_u,
        'stable': stable,
        'lb_pvals': lb_pvals,
        'res': res
    }

def _A_of_f(A: np.ndarray, f: np.ndarray, fs: float) -> np.ndarray:
    """
    A(f) = I - sum_{k=1..p} A_k e^{-i 2π f k / fs}, shape: (n_freq, n_ch, n_ch).
    """
    p, n, _ = A.shape
    I = np.eye(n)
    Af = []
    for ff in f:
        z = np.zeros((n, n), dtype=complex)
        for k in range(1, p+1):
            z += A[k-1] * np.exp(-1j * 2*np.pi*ff * k / fs)
        Af.append(I - z)
    return np.array(Af)  # (n_freq, n, n)

def _H_of_f(Af: np.ndarray) -> np.ndarray:
    """
    Transfer matrix H(f) = A(f)^{-1}, for each frequency. Af shape (n_freq, n, n).
    """
    Hf = np.zeros_like(Af, dtype=complex)
    for i in range(Af.shape[0]):
        Hf[i] = np.linalg.inv(Af[i])
    return Hf

def spectral_dtf_pdc(A: np.ndarray, Sigma_u: np.ndarray, fs: float,
                     fmin: float = 0.0, fmax: float = 50.0, n_freq: int = 256) -> Dict[str, np.ndarray]:
    """
    Compute DTF and PDC spectra from VAR(A, Sigma_u).
    DTF_{i<-j}(f) = |H_{ij}(f)| / sqrt(sum_k |H_{ik}(f)|^2)  (outflow from j to i)
    PDC_{i<-j}(f) = |A_{ij}(f)| / sqrt(sum_k |A_{kj}(f)|^2)  (column-normalized A(f))
    """
    n = A.shape[1]
    f = np.linspace(fmin, fmax, n_freq)
    Af = _A_of_f(A, f, fs)                # (n_f, n, n)
    Hf = _H_of_f(Af)                      # (n_f, n, n)

    # DTF
    DTF = np.zeros((n, n, n_freq))
    for k in range(n_freq):
        H = Hf[k]
        num = np.abs(H)**2
        den = np.sum(num, axis=1, keepdims=True) + 1e-24  # row sum: to targets i
        DTF[:, :, k] = np.sqrt(num / den)                 # sqrt often used; use num/den if you prefer

    # PDC
    PDC = np.zeros((n, n, n_freq))
    for k in range(n_freq):
        A_k = Af[k]
        num = np.abs(A_k)**2
        den = np.sum(num, axis=0, keepdims=True) + 1e-24  # column sum: out of source j
        PDC[:, :, k] = np.sqrt(num / den)

    return {'f': f, 'DTF': DTF, 'PDC': PDC}

def summarize_dtf_pdc_at_harmonics(spec: Dict[str, np.ndarray],
                                   harm: List[float]) -> pd.DataFrame:
    f = spec['f']; DTF = spec['DTF']; PDC = spec['PDC']
    n = DTF.shape[0]
    rows=[]
    for hf in harm:
        idx = int(np.argmin(np.abs(f - hf)))
        for i in range(n):
            for j in range(n):
                rows.append({'freq': float(f[idx]),
                             'target_i': i, 'source_j': j,
                             'DTF': float(DTF[i, j, idx]),
                             'PDC': float(PDC[i, j, idx])})
    return pd.DataFrame(rows)

def run_freq_granger_pdc_dtf(RECORDS: pd.DataFrame,channels: List[str], windows: Optional[List[Tuple[float,float]]] = None,time_col: str = 'Timestamp',order_max: int = 20,crit: str = 'bic',fmin: float = 0.0, fmax: float = 45.0, n_freq: int = 256,harmonics: List[float] = (7.83,14.3,20.8,27.3,33.8),run_granger_tests: bool = True) -> Dict[str, object]:
    """
    Fit VAR, compute DTF/PDC spectra, and report harmonic values.
    Optionally run time-domain Granger tests (F-tests) if statsmodels available.
    """
    fs = infer_fs(RECORDS, time_col)
    X = stack_channels(RECORDS, channels, fs, windows)   # (n_ch, L)
    var = fit_var_model(X, order_max=order_max, crit=crit)

    spec = spectral_dtf_pdc(var['A'], var['Sigma_u'], fs, fmin=fmin, fmax=fmax, n_freq=n_freq)
    table_hp = summarize_dtf_pdc_at_harmonics(spec, list(harmonics))

    gtests = None
    if run_granger_tests and _HAS_SM:
        # test_causality on fitted VAR
        res = var['res']
        pairs=[]
        for i in range(len(channels)):
            for j in range(len(channels)):
                if i==j: continue
                try:
                    # does j cause i?
                    out = res.test_causality(caused=i, causing=[j], kind='f')
                    pairs.append({'target_i':i,'source_j':j,'F':float(out.statistic),'p':float(out.pvalue)})
                except Exception:
                    pairs.append({'target_i':i,'source_j':j,'F':np.nan,'p':np.nan})
        gtests = pd.DataFrame(pairs)

    return {'order': var['order'], 'stable': var['stable'], 'lb_pvals': var['lb_pvals'],'A': var['A'], 'Sigma_u': var['Sigma_u'],'spec': spec, 'harmonics_table': table_hp, 'granger_tests': gtests}

# ------------------ 4b) Transfer Entropy (kNN) ------------------

def _knn_entropy(points: np.ndarray, k: int = 4) -> float:
    """
    Shannon differential entropy via Kozachenko–Leonenko (Euclidean metric).
    H ≈ ψ(n) − ψ(k) + log(c_d) + d * mean(log r_k)
    where r_k is the distance to the k-th NN, c_d = π^{d/2} / Γ(d/2 + 1).
    """
    from sklearn.neighbors import NearestNeighbors
    from scipy.special import gamma as Gamma, digamma

    points = np.asarray(points, float)
    n, dim = points.shape
    if n <= k:
        return np.nan

    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(points)
    dists, _ = nbrs.kneighbors(points)          # (n, k+1) includes self at col 0
    rk = dists[:, -1]                           # distance to k-th neighbor

    c_d = (np.pi ** (dim / 2.0)) / Gamma(dim / 2.0 + 1.0)   # unit-ball volume
    H = digamma(n) - digamma(k) + np.log(c_d) + dim * np.mean(np.log(rk + 1e-24))
    return float(H) + (dim*np.log(2)) - stats.digamma(k) + stats.digamma(n)

def transfer_entropy_knn(x: np.ndarray, y: np.ndarray,lag: int,k_embed_x: int = 1, k_embed_y: int = 1,k: int = 4) -> float:
    """
    TE X→Y at a given *positive* lag (samples): predicts y_{t+lag} from [y_t^(k_embed_y), x_t^(k_embed_x)].
    Uses kNN differential entropy approximation (simple, small-sample friendly).
    """
    assert lag > 0
    N = min(len(x), len(y)) - lag
    if N <= max(k_embed_x, k_embed_y):
        return np.nan
    # build delay vectors
    def embed(sig, kdim):
        if kdim <= 0: return None
        X = []
        for d in range(kdim):
            X.append(sig[d:N+d])
        return np.column_stack(X)
    x0 = x[:N]
    y0 = y[:N]
    y_future = y[lag:lag+N]

    Xy = embed(y0, k_embed_y)          # past of Y
    Xx = embed(x0, k_embed_x)          # past of X
    if Xy is None and Xx is None:
        return np.nan
    # assemble joint vectors
    if Xy is None:
        YF_X = np.column_stack([y_future, Xx])
        YF = y_future[:,None]
        H_yf_x = _knn_entropy(YF_X, k=k)
        H_yf   = _knn_entropy(YF, k=k)
        return float(H_yf - H_yf_x)
    if Xx is None:
        YF_Y = np.column_stack([y_future, Xy])
        H_yf_y = _knn_entropy(YF_Y, k=k)
        H_yf   = _knn_entropy(y_future[:,None], k=k)
        return float(H_yf - H_yf_y)
    YF_YX = np.column_stack([y_future, Xy, Xx])
    YF_Y  = np.column_stack([y_future, Xy])
    H_yf_yx = _knn_entropy(YF_YX, k=k)
    H_yf_y  = _knn_entropy(YF_Y,  k=k)
    return float(H_yf_y - H_yf_yx)

def run_transfer_entropy(RECORDS: pd.DataFrame,x_channel: str, y_channel: str,windows: Optional[List[Tuple[float,float]]] = None,time_col: str = 'Timestamp',lags_ms: List[float] = (10, 20, 40, 80, 160, 320),k_embed_x: int = 1, k_embed_y: int = 1,k: int = 4,n_surr: int = 200,rng_seed: int = 13) -> Dict[str, object]:
    """
    Compute TE(X→Y) and TE(Y→X) across lags (ms). Returns arrays and surrogate 95% thresholds.
    """
    fs = infer_fs(RECORDS, time_col)
    x = get_series(RECORDS, x_channel)
    y = get_series(RECORDS, y_channel)
    x = slice_concat(x, fs, windows)
    y = slice_concat(y, fs, windows)

    lags = [max(1, int(round(fs * lm / 1000.0))) for lm in lags_ms]
    te_xy, te_yx = [], []
    for L in lags:
        te_xy.append(transfer_entropy_knn(x, y, lag=L, k_embed_x=k_embed_x, k_embed_y=k_embed_y, k=k))
        te_yx.append(transfer_entropy_knn(y, x, lag=L, k_embed_x=k_embed_y, k_embed_y=k_embed_x, k=k))
    te_xy = np.array(te_xy, float)
    te_yx = np.array(te_yx, float)

    # surrogate null via circular shift of X and Y independently
    rng = np.random.default_rng(rng_seed)
    null_xy = []
    null_yx = []
    n = len(x)
    for _ in range(n_surr):
        sx = int(rng.integers(1, n-1))
        sy = int(rng.integers(1, n-1))
        xs = np.r_[x[-sx:], x[:-sx]]
        ys = np.r_[y[-sy:], y[:-sy]]
        row_xy=[]; row_yx=[]
        for L in lags:
            row_xy.append(transfer_entropy_knn(xs, y, lag=L, k_embed_x=k_embed_x, k_embed_y=k_embed_y, k=k))
            row_yx.append(transfer_entropy_knn(ys, x, lag=L, k_embed_x=k_embed_y, k_embed_y=k_embed_x, k=k))
        null_xy.append(row_xy); null_yx.append(row_yx)
    null_xy = np.array(null_xy, float)
    null_yx = np.array(null_yx, float)
    thr_xy = np.nanpercentile(null_xy, 95, axis=0)
    thr_yx = np.nanpercentile(null_yx, 95, axis=0)

    return {'lags_ms': np.array(lags_ms, float), 'TE_xy': te_xy, 'TE_yx': te_yx,'thr_xy_95': thr_xy, 'thr_yx_95': thr_yx,'null_xy': null_xy, 'null_yx': null_yx}

# ------------------ 4c) Time-varying AR via Kalman-RLS ------------------

def kalman_rls_tvar_ar(X: np.ndarray, order: int = 4, lam: float = 0.995) -> Dict[str, object]:
    """
    Track time-varying AR coefficients for a multivariate series X (n_ch, L)
    using recursive least squares with forgetting factor lam.
    State vector stacks AR mats row-wise: for n_ch and order p, size = n_ch*n_ch*p.
    Returns coeffs[t] shaped (p, n_ch, n_ch).
    """
    n, L = X.shape
    p = order
    d = n*n*p
    theta = np.zeros(d)                     # initial coeff vector
    P = np.eye(d) * 1e3                     # large initial covariance
    coeffs = []

    def phi_t(t_idx):
        # design vector for y_t = sum_k A_k y_{t-k}
        rows=[]
        for k in range(1, p+1):
            rows.append(X[:, t_idx-k])      # shape (n,)
        Ypast = np.concatenate(rows, axis=0)  # (n*p,)
        # Build block-diag kron for all rows (for each output dim)
        Phi = np.zeros((n, d))
        # For output i, its row params live at offsets
        for i in range(n):
            # coefficients for output i are contiguous blocks of length n across lags
            # index mapping: offset = i + n*j + n*n*(k-1) over (k,j)
            col = 0
            for kk in range(p):
                for j in range(n):
                    idx = i + j*n + kk*n*n   # row-major per-lag
                    Phi[i, idx] = Ypast[kk*n + j]
                    col += 1
        return Phi

    for t in range(p, L):
        Phi = phi_t(t)               # (n, d)
        y  = X[:, t]                 # (n,)
        # RLS update for each output eq combined (matrix form)
        # flatten to a big observation by stacking rows
        H = Phi                      # (n, d)
        R = np.eye(n) * 1e-3
        # predict
        P = P / lam
        # Kalman gain
        S = H @ P @ H.T + R
        K = (P @ H.T) @ np.linalg.pinv(S)
        # residual
        y_hat = H @ theta
        err = y - y_hat
        # update
        theta = theta + K @ err
        P = (np.eye(d) - K @ H) @ P
        # store reshaped coeffs at time t
        A_t = np.zeros((p, n, n))
        for kk in range(p):
            for j in range(n):
                for i in range(n):
                    idx = i + j*n + kk*n*n
                    A_t[kk, i, j] = theta[idx]
        coeffs.append(A_t)
    coeffs = np.array(coeffs)  # (L-p, p, n, n)
    return {'A_t': coeffs, 'order': p}

def dtf_at_freq_from_A(A: np.ndarray, fs: float, f0: float) -> np.ndarray:
    """
    DTF at a single frequency f0 from AR matrices A (p,n,n).
    """
    f = np.array([f0])
    Af = _A_of_f(A, f, fs)            # (1, n, n)
    Hf = _H_of_f(Af)                  # (1, n, n)
    H = Hf[0]
    num = np.abs(H)**2
    den = np.sum(num, axis=1, keepdims=True) + 1e-24
    return np.sqrt(num/den)           # (n, n)

def run_tvar_dtf(RECORDS: pd.DataFrame,channels: List[str],windows: Optional[List[Tuple[float,float]]] = None,time_col: str = 'Timestamp',order: int = 4, lam: float = 0.995,f0: float = 7.83) -> Dict[str, object]:

    """
    Time-varying AR (Kalman-RLS) and DTF(t, i<-j) at f0.
    Returns dict with A_t (T,p,n,n) and DTF_t (T,n,n).
    """
    fs = infer_fs(RECORDS, time_col)
    X = stack_channels(RECORDS, channels, fs, windows)  # (n, L)
    tv = kalman_rls_tvar_ar(X, order=order, lam=lam)
    A_t = tv['A_t']                      # (T, p, n, n)
    T, p, n, _ = A_t.shape
    D = np.zeros((T, n, n))
    for t in range(T):
        D[t] = dtf_at_freq_from_A(A_t[t], fs, f0)
    return {'A_t': A_t, 'DTF_t': D, 'fs': fs, 'f0': f0, 'channels': channels}

# ===== Transfer Entropy helpers (drop-in replacement) =====


def _knn_entropy(points: np.ndarray, k: int = 4) -> float:
    """
    Differential entropy via Kozachenko–Leonenko (Euclidean).
    H ≈ ψ(n) − ψ(k) + log(c_d) + d * mean(log r_k),
    where r_k is distance to the k-th NN; c_d = π^{d/2} / Γ(d/2 + 1).
    """
    P = np.asarray(points, float)
    if P.ndim == 1:
        P = P[:, None]
    n, d = P.shape
    if n <= k or d < 1:
        return np.nan

    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(P)
    dists, _ = nbrs.kneighbors(P)           # includes self at [:,0]
    rk = dists[:, -1]                       # k-th neighbor distance

    c_d = (np.pi ** (d / 2.0)) / Gamma(d / 2.0 + 1.0)  # unit-ball volume
    H = digamma(n) - digamma(k) + np.log(c_d) + d * np.mean(np.log(rk + 1e-24))
    return float(H)

def _embed(sig: np.ndarray, kdim: int) -> Optional[np.ndarray]:
    """Simple delay embedding with unit delay: [x_t, x_{t+1}, ..., x_{t+kdim-1}] aligned to t."""
    if kdim <= 0:
        return None
    N = sig.size - (kdim - 1)
    if N <= 0:
        return None
    cols = [sig[i:i+N] for i in range(kdim)]
    return np.column_stack(cols)

def transfer_entropy_knn(x: np.ndarray, y: np.ndarray,
                         lag: int,
                         k_embed_x: int = 1, k_embed_y: int = 1,
                         k: int = 4) -> float:
    """
    TE X→Y at positive sample lag.
    TE = H(Y_{t+lag}, Y_t^(k)) - H(Y_t^(k)) - H(Y_{t+lag}, Y_t^(k), X_t^(l)) + H(Y_t^(k), X_t^(l))
    """
    assert lag > 0
    L = min(x.size, y.size)
    # Align so that future y is available
    y_future = y[lag:L]
    y_past = y[:L-lag]
    x_past = x[:L-lag]

    Yp = _embed(y_past, k_embed_y)    # (N, k_y) or None
    Xp = _embed(x_past, k_embed_x)    # (N, k_x) or None
    if Yp is None and Xp is None:
        return np.nan

    # Trim y_future to match embedded rows
    N = None
    if Yp is not None:
        N = Yp.shape[0]
    if Xp is not None:
        N = Xp.shape[0] if N is None else min(N, Xp.shape[0])
    if N is None or N <= k+1:
        return np.nan
    yF = y_future[:N, None]
    if Yp is not None: Yp = Yp[:N, :]
    if Xp is not None: Xp = Xp[:N, :]

    # Build joint vectors
    if Yp is None:    # TE reduces to H(yF) - H(yF, Xp)
        H1 = _knn_entropy(yF, k=k)
        H2 = _knn_entropy(np.column_stack([yF, Xp]), k=k)
        return float(H1 - H2)
    if Xp is None:    # TE reduces to H(yF, Yp) - H(Yp) - [H(yF, Yp) - H(Yp)] = 0
        H_yF_Yp = _knn_entropy(np.column_stack([yF, Yp]), k=k)
        H_Yp    = _knn_entropy(Yp, k=k)
        return float((H_yF_Yp - H_Yp) - (H_yF_Yp - H_Yp))

    H_yF_Yp     = _knn_entropy(np.column_stack([yF, Yp]), k=k)
    H_Yp        = _knn_entropy(Yp, k=k)
    H_yF_Yp_Xp  = _knn_entropy(np.column_stack([yF, Yp, Xp]), k=k)
    H_Yp_Xp     = _knn_entropy(np.column_stack([Yp, Xp]), k=k)

    TE = (H_yF_Yp - H_Yp) - (H_yF_Yp_Xp - H_Yp_Xp)
    return float(TE)

"""
DTF grid + text report
----------------------
Create a grid of DTF(t, target <- source) plots for ALL electrodes (targets),
and print a text table with:
  • mean DTF in ignition windows
  • Δ = mean_ign − mean_base
  • Δ threshold (95% null) from a simple permutation (circular shift) null

USAGE
=====
# 1) Compute time-varying DTF separately for ignition and baseline:
tv_ign = run_tvar_dtf(
    RECORDS,
    channels=['EEG.Oz','EEG.O1','EEG.O2'],  # include your chosen source channel too
    windows=[(290,310),(580,600)],
    time_col='Timestamp',
    order=4, lam=0.995, f0=7.83
)
tv_base = run_tvar_dtf(
    RECORDS,
    channels=['EEG.Oz','EEG.O1','EEG.O2'],
    windows=[(0, 280), (325, 575)],         # supply your baseline windows explicitly
    time_col='Timestamp',
    order=4, lam=0.995, f0=7.83
)

# 2) Make the grid and printed report for source='EEG.Oz' (or whichever)
plot_dtf_grid_and_report(
    tv_ign, tv_base,
    src_channel='EEG.Oz',
    smooth_sec=0.5,
    n_cols=3,
    session_name='session_A'
)
"""


def _moving_average(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1: return x
    w = np.ones(win) / win
    return np.convolve(x, w, mode='same')

def _mean_in_seconds(x: np.ndarray, fs: float, windows: Optional[List[Tuple[float,float]]]) -> float:
    if not windows:
        return float(np.nanmean(x))
    vals = []
    T = len(x)
    for (t0, t1) in windows:
        s = int(round(t0 * fs)); e = int(round(t1 * fs))
        s = max(0, min(T-1, s)); e = max(s+1, min(T, e))
        vals.append(np.nanmean(x[s:e]))
    return float(np.nanmean(vals)) if vals else np.nan

def plot_dtf_grid_bidir_like_single(tv_ign: Dict[str, object],
                                    tv_base: Optional[Dict[str, object]],
                                    src_channel: str,
                                    smooth_sec: float = 0.5,
                                    n_cols: int = 3,
                                    ign_windows_sec: Optional[List[Tuple[float,float]]] = None,
                                    base_windows_sec: Optional[List[Tuple[float,float]]] = None,
                                    show_baseline: bool = False,
                                    n_perm: int = 500,
                                    rng_seed: int = 7,
                                    session_name: Optional[str] = None) -> None:
    """
    Grid of DTF(t, target <- source) for all targets (all electrodes except source).
    For each target:
      - plot smoothed DTF time series
      - compute mean ignition (over ign_windows_sec or whole ign series)
      - compute baseline mean (over base_windows_sec or whole base series)
      - permutation null on Δ = mean_ign − mean_base  -> threshold (95%)
    Print a text summary of mean_ign and Δ threshold.

    tv_ign / tv_base: dicts from run_tvar_dtf; must be computed with the SAME 'channels' ordering.
    src_channel: e.g., 'EEG.Oz' — must be present in tv_ign['channels'].
    """
    chs = tv_ign['channels']
    if src_channel not in chs:
        # allow bare names
        src_channel = ('EEG.' + src_channel) if ('EEG.' + src_channel) in chs else src_channel
    assert src_channel in chs, f"{src_channel} not found in {chs}"
    if tv_base is not None:
        assert chs == tv_base['channels'], "Channel lists differ between tv_ign and tv_base."

    src_idx = chs.index(src_channel)
    fs  = float(tv_ign['fs'])
    f0  = float(tv_ign['f0'])
    D_I = np.asarray(tv_ign['DTF_t'])    # (T_ign, n, n)
    D_B = np.asarray(tv_base['DTF_t']) if tv_base is not None else None

    # smoothing window
    win = max(1, int(round(smooth_sec * fs)))

    # targets = all except source
    targets = [i for i in range(len(chs)) if i != src_idx]
    n_t = len(targets)
    n_cols = max(1, n_cols)
    n_rows = int(np.ceil(n_t / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.8*n_cols, 2.4*n_rows), squeeze=False)
    axes = axes.ravel()

    rng = np.random.default_rng(rng_seed)
    text_lines = []

    for m, tgt in enumerate(targets):
        ax = axes[m]

        # ignition: src→tgt and tgt→src
        ij_I = _moving_average(D_I[:, tgt, src_idx], win)
        ji_I = _moving_average(D_I[:, src_idx, tgt], win)
        di_I = ij_I - ji_I

        # baseline (optional overlay)
        if show_baseline and D_B is not None:
            ij_B = _moving_average(D_B[:, tgt, src_idx], win)
            ji_B = _moving_average(D_B[:, src_idx, tgt], win)
        else:
            ij_B = ji_B = None

        # means for stats (per windows if provided)
        mean_ign = _mean_in_seconds(ij_I, fs, ign_windows_sec)
        if D_B is not None:
            mean_base = _mean_in_seconds(ij_B if ij_B is not None else D_B[:, tgt, src_idx], fs, base_windows_sec)
        else:
            mean_base = np.nan
        delta_obs = mean_ign - mean_base if np.isfinite(mean_base) else np.nan

        # permutation null for Δ: circularly shift ignition ij series
        thr95 = np.nan
        if np.isfinite(delta_obs):
            null = []
            T = len(ij_I)
            for _ in range(n_perm):
                s = int(rng.integers(1, max(2, T-1)))
                ij_perm = np.r_[ij_I[-s:], ij_I[:-s]]
                perm_mean_ign = _mean_in_seconds(ij_perm, fs, ign_windows_sec)
                null.append(perm_mean_ign - mean_base)
            thr95 = float(np.nanpercentile(null, 95))

        # ---- plotting (match single-link style) ----
        ax.plot(ij_I, color='tab:blue',  lw=1.6, label=f'{src_channel}→{chs[tgt]} @ {f0:.2f} Hz')
        ax.plot(ji_I, color='tab:orange', lw=1.2, label=f'{chs[tgt]}→{src_channel}')
        ax.plot(di_I, color='tab:green', lw=1.2, label='Directionality Index (Δ)')

        if show_baseline and D_B is not None:
            ax.plot(np.linspace(0, len(D_B)-1, len(D_B)), ij_B, color='tab:blue',  alpha=0.25, lw=0.8)
            ax.plot(np.linspace(0, len(D_B)-1, len(D_B)), ji_B, color='tab:orange', alpha=0.25, lw=0.8)

#         ax.set_ylim(-1.0, 1.0)
        # dynamic limits that include src→tgt, tgt→src, and Δ, with a small pad
        ymin = float(np.nanmin([ij_I.min(), ji_I.min(), di_I.min()]))
        ymax = float(np.nanmax([ij_I.max(), ji_I.max(), di_I.max()]))

        pad  = 0.05 * (ymax - ymin + 1e-9)
        lo   = max(ymin - pad, -0.2)          # don’t let it go crazy low
        hi   = min(max(ymax + pad, 0.2), 1.05)  # cap near 1

        ax.set_ylim(lo, hi)
        ax.grid(alpha=0.25)
        ax.set_title(f"{chs[tgt]}  ({src_channel}↔{chs[tgt]}) @ {f0:.2f} Hz", fontsize=10)
        ax.set_xlabel('Samples (after AR warm-up)')
        ax.set_ylabel('DTF')
        ax.legend(loc='upper right', fontsize=8, ncol=1, framealpha=0.8)

        # text box
        txt = [f"mean_ign={mean_ign:.3f}"]
        if np.isfinite(delta_obs):
            txt.append(f"Δ={delta_obs:.3f}")
            txt.append(f"thr95={thr95:.3f}")
            txt.append(f"Sig={delta_obs>thr95}")
        ax.text(0.01, 0.02, "\n".join(txt), transform=ax.transAxes, fontsize=8,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, lw=0.5))

        text_lines.append(f"{chs[tgt]}: mean_ign={mean_ign:.4f}" +
                          (f", Δ={delta_obs:.4f}, thr95={thr95:.4f}, Sig={delta_obs>thr95}"
                           if np.isfinite(delta_obs) else ""))

    # drop unused axes
    for k in range(n_t, n_rows*n_cols):
        fig.delaxes(axes[k])

    supt = f"DTF grid: {src_channel}↔targets @ {f0:.2f} Hz"
    if session_name: supt = f"{session_name} — " + supt
    fig.suptitle(supt, y=1.02, fontsize=12)
    plt.tight_layout()
    plt.show()

    print("\n=== DTF mean ignition and Δ thresholds ===")
    for line in text_lines:
        print(line)
