import numpy as np
import matplotlib.pyplot as plt

# ---------- helpers ----------
def _auto_savgol(y, max_window=31):
    """Light, safe smoothing for visibility (odd window; poly=2)."""
    y = np.asarray(y, float)
    n = y.size
    if n < 7:
        return y
    try:
        from scipy.signal import savgol_filter
        w = max(5, min(max_window, int(round(n/15))))
        if w % 2 == 0:
            w += 1
        if w >= n:
            w = n - 1 if (n % 2 == 0) else n
        if w < 5:
            return y
        return savgol_filter(y, w, polyorder=2, mode='interp')
    except Exception:
        # moving average fallback
        w = max(5, min(max_window, int(round(n/15))))
        if w < 2:
            return y
        return np.convolve(np.nan_to_num(y), np.ones(w)/w, mode='same')

def _clip_shading(ax, windows, tmin, tmax, **kwargs):
    if not windows:
        return
    for (t0, t1) in windows:
        if t1 < tmin or t0 > tmax:
            continue
        ax.axvspan(max(t0, tmin), min(t1, tmax), **kwargs)

# ---------- main ----------

def plot_sr_ignition_signature(records,
                               eeg_channel: str,            # e.g., 'EEG.O1'
                               sr_channel: str,             # magnetometer or posterior proxy
                               ignition_windows,            # [(t0,t1), ...] in *absolute seconds*
                               time_col='Timestamp',
                               harmonics=(7.83, 14.3, 20.8, 27.3, 33.8),
                               half_bw=0.6, win_sec=8.0, step_sec=1.0,
                               n_null=200, out_png='sr_ignition_signature.png',
                               *,
                               facet=False,                 # one subplot per harmonic
                               direct_labels=True,          # write labels at line ends
                               legend_outside=True,         # legend outside if used
                               smooth=False,                # light Savitzky–Golay smoothing
                               linewidth=1.8,
                               grid=True,
                               clip_to_data=True,
                               figsize_overlay=(10, 4.6),
                               figsize_facet=(10, 8),
                               dpi=160,
                               return_fig_ax=False):
    """
    Enhanced sliding-coherence plotter: clearer lines, non-overlapping labels/legend,
    faceting option, shaded ignition clipped to data range, and optional smoothing.

    Notes
    -----
    - Assumes `sliding_coherence_f0` returns dict with keys: 't' (sec), 'coh', 'null95'.
    - If your `sliding_coherence_f0` uses *absolute seconds* for 't', shaded windows
      will align. If your 't' is relative to a slice, align your windows accordingly.
    """



    # 1) compute a sliding coherence trace for each harmonic
    traces = []
    for f0 in harmonics:
        sl = sliding_coherence_f0(
            records, eeg_channel, sr_channel,
            f0=f0, half=half_bw, time_col=time_col,
            win_sec=win_sec, step_sec=step_sec, n_null=n_null, show=False
        )
        if sl['t'] is None or len(sl['t']) == 0:
            continue
        coh_raw = sl['coh']
        if smooth:
            # Use the SAME callable you use to smooth the plotted trace
            smoother_fn = _auto_savgol # <-- callable, not a bool
            coh_plot = smoother_fn(coh_raw)
            thr95_s = build_null_threshold_smooth(coh_raw, n_null=n_null, method='block',smoother=smoother_fn)
            zcoh = (coh_plot - np.nanmean(coh_plot)) / (np.nanstd(coh_plot) + 1e-12)
            zthr = zscore_with_series(thr95_s, coh_plot)
        else:
            smoother_fn = None
            coh_plot = coh_raw
            thr95 = build_null_threshold_smooth(coh_raw, n_null=n_null, method='block',smoother=None)
            zcoh = (coh_plot - np.nanmean(coh_plot)) / (np.nanstd(coh_plot) + 1e-12)
            zthr = zscore_with_series(thr95, coh_plot)
        traces.append({'f0': f0, 't': np.asarray(sl['t']), 'zcoh': zcoh, 'zthr': zthr})

    if not traces:
        raise ValueError('No coherence points to plot — check inputs/windows')

    # common time extent across harmonics (for shading and xlim)
    t_all = np.concatenate([tr['t'] for tr in traces])
    tmin, tmax = float(np.nanmin(t_all)), float(np.nanmax(t_all))

    colors = plt.get_cmap('tab10').colors
    null_alpha = 0.35


    coh_raw = sl['coh']
    if smooth:
        # Use the SAME callable you use to smooth the plotted trace
        smoother_fn = _auto_savgol # <-- callable, not a bool
        coh_plot = smoother_fn(coh_raw)
        thr95_s = build_null_threshold_smooth(coh_raw, n_null=n_null, method='block',smoother=smoother_fn)
        zcoh = (coh_plot - np.nanmean(coh_plot)) / (np.nanstd(coh_plot) + 1e-12)
        zthr = zscore_with_series(thr95_s, coh_plot)
    else:
        smoother_fn = None
        coh_plot = coh_raw
        thr95 = build_null_threshold_smooth(coh_raw, n_null=n_null, method='block',smoother=None)
        zcoh = (coh_plot - np.nanmean(coh_plot)) / (np.nanstd(coh_plot) + 1e-12)
        zthr = zscore_with_series(thr95, coh_plot)

    # ---------- Faceted mode ----------
    if facet:
        fig, axes = plt.subplots(len(traces), 1, sharex=True, sharey=True,
                                 figsize=figsize_facet)
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])
        for i, (ax, tr) in enumerate(zip(axes, traces)):
            col = colors[i % len(colors)]
            ax.plot(tr['t'], tr['zcoh'], lw=linewidth, color=col)
            ax.hlines(tr['zthr'], tmin, tmax, colors=col, linestyles='--', lw=1.0, alpha=null_alpha)
            _clip_shading(ax, ignition_windows, tmin, tmax, color='k', alpha=0.08, zorder=0)
            if grid:
                ax.grid(True, alpha=1, linestyle=':')
            ax.set_ylabel(f"{tr['f0']:.2f} Hz")
        axes[-1].set_xlabel('Time (s)')
        fig.suptitle('EEG–SR sliding coherence at Schumann harmonics (faceted)', y=0.995)
        fig.tight_layout()
        if out_png:
            fig.savefig(out_png, dpi=dpi, bbox_inches='tight')
        if return_fig_ax:
            return fig, axes
        plt.show()
        return

    # ---------- Overlay mode ----------
    fig, ax = plt.subplots(figsize=figsize_overlay)

    lines = []
    for i, tr in enumerate(traces):
        col = colors[i % len(colors)]
        ln, = ax.plot(tr['t'], tr['zcoh'], lw=linewidth, color=col, label=f"{tr['f0']:.2f} Hz")
        lines.append((ln, tr))
        ax.hlines(tr['zthr'], tmin, tmax, colors=col, linestyles='--', lw=1.0, alpha=null_alpha)

    _clip_shading(ax, ignition_windows, tmin, tmax, color='k', alpha=0.08, zorder=0)

    if grid:
        ax.grid(True, alpha=0.25, linestyle=':')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('z-coherence')
    ax.set_title('EEG–SR sliding coherence at Schumann harmonics (shaded = ignition)')

    # tidy legend/labels
    if clip_to_data:
        ax.set_xlim(tmin, tmax)

    used_legend = False
    if direct_labels:
        # label at the right-most finite sample for each line
        xpad = 0.01 * (tmax - tmin)
        for ln, tr in lines:
            x = tr['t']; y = tr['zcoh']
            idx = np.where(np.isfinite(y))[0]
            if idx.size == 0:
                continue
            j = idx[-1]
            ax.text(x[j] + xpad, y[j], f"{tr['f0']:.2f}", color=ln.get_color(),
                    fontsize=9, va='center', ha='left', clip_on=False)
    else:
        if legend_outside:
            ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, frameon=False)
            fig.subplots_adjust(right=0.78)
        else:
            ax.legend(loc='lower right', frameon=False)
        used_legend = True

    fig.tight_layout()
    if out_png:
        fig.savefig(out_png, dpi=dpi, bbox_inches='tight')
    if return_fig_ax:
        return fig, ax
    plt.show()

import numpy as np
from scipy import signal

def compute_coherence_at_f0(xe, xs, fs, f0, half):
    """
    Magnitude-squared coherence between signals xe and xs at target frequency f0.

    Uses Welch autospectra (Pxx, Pyy) and cross-spectrum (Pxy) on the provided
    windowed segments, then returns a scalar coherence value aggregated within
    the band [f0 - half, f0 + half]. If that band contains no frequency bin,
    returns the coherence at the nearest available bin to f0.

    Parameters
    ----------
    xe : array_like
        First signal segment (e.g., EEG), 1-D.
    xs : array_like
        Second signal segment (e.g., SR proxy / magnetometer), 1-D.
    fs : float
        Sampling rate in Hz.
    f0 : float
        Target center frequency in Hz.
    half : float
        Half-bandwidth in Hz; analyze [f0 - half, f0 + half].

    Returns
    -------
    float
        Coherence (0..1) at/around f0.
    """
    xe = np.asarray(xe, dtype=float)
    xs = np.asarray(xs, dtype=float)
    N = int(min(len(xe), len(xs)))
    if N < 32:
        raise ValueError("Window too short for coherence (N < 32 samples)")
    xe = xe[:N]
    xs = xs[:N]

    # Choose segment length for spectral estimates: use half the window (typical)
    nperseg = int(max(32, min(N, N // 2)))
    noverlap = int(nperseg // 2)

    # Autospectra and cross-spectrum (Welch / CSD)
    f, Pxx = signal.welch(
        xe, fs=fs, window='hann', nperseg=nperseg, noverlap=noverlap,
        detrend='constant', return_onesided=True, scaling='density'
    )
    _, Pyy = signal.welch(
        xs, fs=fs, window='hann', nperseg=nperseg, noverlap=noverlap,
        detrend='constant', return_onesided=True, scaling='density'
    )
    _, Pxy = signal.csd(
        xe, xs, fs=fs, window='hann', nperseg=nperseg, noverlap=noverlap,
        detrend='constant', return_onesided=True, scaling='density'
    )

    # Magnitude-squared coherence
    eps = 1e-20
    Cxy = (np.abs(Pxy) ** 2) / (Pxx * Pyy + eps)

    # Aggregate within the f0 ± half band; fallback to nearest bin if empty
    f_lo = max(0.0, float(f0) - float(half))
    f_hi = float(f0) + float(half)
    band = (f >= f_lo) & (f <= f_hi) & np.isfinite(Cxy)
    if np.any(band):
        c_val = float(np.nanmean(Cxy[band]))
    else:
        idx = int(np.argmin(np.abs(f - float(f0))))
        c_val = float(Cxy[idx]) if np.isfinite(Cxy[idx]) else float('nan')

    # Clamp numeric noise into [0, 1]
    if np.isfinite(c_val):
        c_val = float(np.clip(c_val, 0.0, 1.0))
    return c_val



def build_null_threshold(coh, n_null=200, method='block', block_len=None, alpha=0.05, random_state=13):
    """
    Estimate a null threshold for a sliding coherence trace by resampling the
    coherence sequence itself.

    For each surrogate, we create a bootstrap replica of the coherence
    time-series and take its maximum. The (1-alpha) percentile of these maxima
    is returned. By default, we use a block-bootstrap that preserves local
    autocorrelation structure; set method='iid' for simple i.i.d. resampling.

    Parameters
    ----------
    coh : array_like
        Coherence values (0..1) across sliding-window centers; NaNs allowed.
    n_null : int, optional
        Number of surrogate replicates (default 200).
    method : {'block','iid'}, optional
        Resampling strategy. 'block' preserves short-range correlations.
    block_len : int or None, optional
        Block length (in samples) for block-bootstrap. If None, uses ~5% of N
        (at least 5 samples).
    alpha : float, optional
        Significance level (default 0.05 → 95th percentile).
    random_state : int, optional
        RNG seed for reproducibility.

    Returns
    -------
    float
        Estimated (1 - alpha) percentile of the null maxima; clipped to [0, 1].
    """
    import numpy as np

    c = np.asarray(coh, dtype=float)
    # Keep only finite values
    c = c[np.isfinite(c)]
    if c.size == 0:
        return float('nan')

    rng = np.random.default_rng(random_state)
    N = int(c.size)

    maxima = []
    if method == 'iid':
        for _ in range(int(n_null)):
            samp = rng.choice(c, size=N, replace=True)
            maxima.append(float(np.nanmax(samp)))
    else:
        # Block bootstrap with circular wrap to preserve local structure
        if block_len is None:
            block_len = max(5, int(round(N / 20)))  # ~5% of the series
        B = int(block_len)
        for _ in range(int(n_null)):
            idx = []
            filled = 0
            while filled < N:
                start = int(rng.integers(0, N))
                end = start + B
                if end <= N:
                    idx.extend(range(start, end))
                else:
                    # circular wrap
                    idx.extend(list(range(start, N)) + list(range(0, end - N)))
                filled += B
            idx = np.asarray(idx[:N], dtype=int)
            surrogate = c[idx]
            maxima.append(float(np.nanmax(surrogate)))

    q = 100.0 * (1.0 - float(alpha))
    thr = float(np.nanpercentile(maxima, q))
    # numeric safety
    return float(np.clip(thr, 0.0, 1.0))

import numpy as np

def build_null_threshold_smooth(coh_raw, n_null=200, method='block', block_len=None,
                                alpha=0.05, random_state=13, smoother=None):
    """
    Compute a (1-alpha) null threshold compatible with *smoothed* plotting.

    We generate surrogate coherence sequences from the *raw* coherence trace
    (coh_raw) via block bootstrap (or IID), then optionally apply the same
    smoothing used for the plotted trace to each surrogate before taking the
    maximum. This keeps the null line comparable to what you actually plot.

    Parameters
    ----------
    coh_raw : array_like
        UnsMoothed coherence values (0..1) across time.
    n_null : int
        Number of surrogate replicates.
    method : {'block','iid'}
        Resampling strategy for temporal dependence.
    block_len : int or None
        Block length in samples for block bootstrap. If None, ~5% of N (>=5).
    alpha : float
        Significance level (default 0.05 → 95th percentile of maxima).
    random_state : int
        RNG seed.
    smoother : callable or None
        A function y -> y_s that applies the *same* smoothing as used on the
        plotted trace. If None, no smoothing is applied to surrogates.

    Returns
    -------
    thr95_s : float
        Null threshold after applying `smoother` to surrogates (if provided).
    """
    c = np.asarray(coh_raw, float)
    c = c[np.isfinite(c)]
    if c.size == 0:
        return float('nan')

    rng = np.random.default_rng(random_state)
    N = int(c.size)

    maxima = []
    if method == 'iid':
        for _ in range(int(n_null)):
            samp = rng.choice(c, size=N, replace=True)
            if smoother is not None:
                samp = np.asarray(smoother(samp), float)
            maxima.append(float(np.nanmax(samp)))
    else:
        if block_len is None:
            block_len = max(5, int(round(N / 20)))
        B = int(block_len)
        for _ in range(int(n_null)):
            idx = []
            filled = 0
            while filled < N:
                start = int(rng.integers(0, N))
                end = start + B
                if end <= N:
                    idx.extend(range(start, end))
                else:
                    idx.extend(list(range(start, N)) + list(range(0, end - N)))
                filled += B
            idx = np.asarray(idx[:N], int)
            surrogate = c[idx]
            if smoother is not None:
                surrogate = np.asarray(smoother(surrogate), float)
            maxima.append(float(np.nanmax(surrogate)))

    q = 100.0 * (1.0 - float(alpha))
    thr95_s = float(np.nanpercentile(maxima, q))
    return float(np.clip(thr95_s, 0.0, 1.0))


def zscore_with_series(value, series, eps=1e-12):
    """Convert a scalar threshold `value` into z-units using the mean/std of `series`."""
    series = np.asarray(series, float)
    m = float(np.nanmean(series))
    s = float(np.nanstd(series)) + eps
    return float((float(value) - m) / s)

# ---- Example wiring inside your plotting code ----
# coh_raw = sl['coh']
# if smooth:
#     coh_plot = _auto_savgol(coh_raw)
#     thr95_s = build_null_threshold_smooth(coh_raw, n_null=n_null, method='block',
#                                           smoother=_auto_savgol)
#     zcoh = (coh_plot - np.nanmean(coh_plot)) / (np.nanstd(coh_plot) + 1e-12)
#     zthr = zscore_with_series(thr95_s, coh_plot)
# else:
#     zcoh = (coh_raw - np.nanmean(coh_raw)) / (np.nanstd(coh_raw) + 1e-12)
#     thr95 = build_null_threshold_smooth(coh_raw, n_null=n_null, method='block', smoother=None)
#     zthr = zscore_with_series(thr95, coh_raw)
