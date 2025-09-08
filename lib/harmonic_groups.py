# SR harmonic groups supplied by user + helpers to analyze/plot each group
# Requires: numpy, matplotlib, and your existing sliding_coherence_f0(...)
# Optional (already in your notebook/canvas): build_null_threshold, build_null_threshold_smooth, _auto_savgol

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# ----------------------------
# 1) Group definitions (exactly as provided)
# ----------------------------

def sr_groups():
    return {
        # H1–H5
        'Harmonics_UpTo33': (7.83, 14.3, 20.8, 27.3, 33.8),
        # H6–H10 (mind 60 Hz mains; you may cap at 58 Hz in plots)
        'HighHarmonics_40to60': (7.83, 40.3, 46.8, 53.3, 59.8),
        # Subharmonics /2..5
        'Subharmonics_2to5': (7.83, 3.915, 2.61, 1.9575, 1.566),
        # Subharmonics /6..10 (+ 1.2 as given)
        'Subharmonics_6to10_Mixed': (7.83, 1.305, 1.11857, 0.9788, 1.2, 0.783),
    }

# ----------------------------
# 2) Adaptive parameter rules per f0
# ----------------------------

def win_for_f0(f0, cycles=8, min_win=8.0, max_win=120.0):
    """Adaptive window: ensure ≥cycles of f0, clipped to [min_win, max_win]."""
    if f0 <= 1e-9:
        return max_win
    return float(np.clip(cycles/float(f0), min_win, max_win))

def half_bw_for_win(win_sec, mult=2.5, min_bw=0.1):
    """Choose half-bandwidth from spectral resolution Δf≈2/win_sec, scaled by mult."""
    df = 2.0 / max(win_sec, 1e-6)
    return float(max(min_bw, mult*df/2.0))  # ≈ mult * (Δf/2)

# Fallbacks if the smoother/null helpers aren't in scope

def _maybe_smoother():
    try:
        return _auto_savgol  # defined in your plotting utils
    except Exception:
        return None

def _build_null_for_series(coh_raw, n_null=200, smoother=None):
    try:
        return build_null_threshold_smooth(coh_raw, n_null=n_null, smoother=smoother)
    except Exception:
        return build_null_threshold(coh_raw, n_null=n_null)


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

def sliding_coherence_f0(df, eeg_channel, sr_channel, ignition_windows,
    f0, half, time_col='Timestamp',
    win_sec=8.0, step_sec=1.0,
    n_null=200, show=False,
    fast_mode=False, max_sec=None, max_windows=None):

    # --- Convert time to seconds from start ---
    tcol = df[time_col]
    if np.issubdtype(tcol.dtype, np.number):
        t_sec = tcol.values.astype(float)
        t_sec = t_sec - t_sec[0]
    else:
        t_dt = pd.to_datetime(tcol)
        t_sec = (t_dt - t_dt.iloc[0]).dt.total_seconds().values
    df['t_sec'] = t_sec


    # --- REMOVE hidden caps. Only crop if explicitly requested ---
    if fast_mode and (max_sec is None and max_windows is None):
        max_sec = 40.0 # opt-in fast behavior; otherwise no crop


    if max_sec is not None:
        df = df[df['t_sec'] <= max_sec]
    if max_windows is not None:
        # subsample evenly to at most max_windows centers later
        pass # (implement only if you *really* need it)


    # --- compute sliding coherence over the WHOLE (possibly cropped) df ---
    # estimate fs robustly
    dt = np.median(np.diff(df['t_sec'].values))
    fs = 1.0/float(dt)


    # centers from win/2 to T-win/2, step = step_sec
    T = df['t_sec'].iloc[-1]
    centers = np.arange(win_sec/2, max(0, T - win_sec/2) + 1e-9, step_sec)


    # compute coherence at f0 for each center (pseudo)
    coh_vals = []
    for c in centers:
        t0, t1 = c - win_sec/2, c + win_sec/2
        w = (df['t_sec'] >= t0) & (df['t_sec'] < t1)
        xe = df.loc[w, eeg_channel].values
        xs = df.loc[w, sr_channel].values
        if len(xe) < int(0.8*win_sec*fs) or len(xs) < int(0.8*win_sec*fs):
            coh_vals.append(np.nan); continue
        # bandpass around f0 ± half (your own filter or multitaper)
        # compute magnitude-squared coherence at f0 (your method)
        coh_vals.append(compute_coherence_at_f0(xe, xs, fs, f0, half))


    coh = np.asarray(coh_vals)
    # build null via surrogates (existing code)
    null95 = build_null_threshold(coh, n_null=n_null) # your existing routine


    return {
    't': centers,
    'coh': coh,
    'null95': null95,
    'fs': fs
    }

# ----------------------------
# 3) Plot a single group with adaptive windows and smooth-aware nulls
# ----------------------------

def plot_sr_group_adaptive(records,
                            eeg_channel: str,
                            sr_channel: str,
                            group_name: str,
                            ignition_windows=None,
                            time_col='Timestamp',
                            cycles=8, min_win=8.0, max_win=120.0,
                            step_sec=1.0, n_null=200, smooth=True,
                            facet=True, linewidth=1.8, grid=True,
                            limit_high_hz=59.8,
                            out_png=None, dpi=160):
    """
    Compute & plot sliding z-coherence for all f0 in the chosen group with
    frequency-adaptive window lengths and (if available) smooth-aware nulls.
    """
    GROUPS = sr_groups()
    if group_name not in GROUPS:
        raise ValueError(f"Unknown group '{group_name}'. Available: {list(GROUPS)}")

    f0s = [f for f in GROUPS[group_name] if (f <= float(limit_high_hz) + 1e-9)]
    if not f0s:
        raise ValueError("No frequencies ≤ limit_high_hz in this group.")

    traces = []
    smoother_fn = _maybe_smoother() if smooth else None

    # Try to detect whether sliding_coherence_f0 accepts 'wins' kw
    import inspect
    sig = inspect.signature(sliding_coherence_f0)
    has_wins = ('wins' in sig.parameters)

    for f0 in f0s:
        win_sec = win_for_f0(f0, cycles=cycles, min_win=min_win, max_win=max_win)
        half_bw = half_bw_for_win(win_sec)
        # call sliding_coherence_f0 with or without wins
        kwargs = dict(f0=f0, half=half_bw, time_col=time_col,
                      win_sec=win_sec, step_sec=step_sec, n_null=n_null, show=False)
        if has_wins:
            kwargs['wins'] = None
        sl = sliding_coherence_f0(records, eeg_channel, sr_channel,ignition_windows, **kwargs)
        t = np.asarray(sl['t'])
        coh_raw = np.asarray(sl['coh'], float)
        if t.size == 0 or coh_raw.size == 0:
            continue
        # smoothing for display
        if smoother_fn is not None:
            try:
                coh_plot = np.asarray(smoother_fn(coh_raw), float)
            except Exception:
                coh_plot = coh_raw
                smoother_fn = None
        else:
            coh_plot = coh_raw
        # smooth-aware null if available
        thr95 = _build_null_for_series(coh_raw, n_null=n_null, smoother=smoother_fn)
        # z-score relative to plotted series
        m = float(np.nanmean(coh_plot)); s = float(np.nanstd(coh_plot) + 1e-12)
        zcoh = (coh_plot - m)/s
        zthr = (thr95 - m)/s
        traces.append({'f0': f0, 't': t, 'zcoh': zcoh, 'zthr': zthr, 'win_sec': win_sec, 'half_bw': half_bw})

    if not traces:
        raise ValueError("No valid traces to plot.")

    # Common time range
    t_all = np.concatenate([tr['t'] for tr in traces])
    tmin, tmax = float(np.nanmin(t_all)), float(np.nanmax(t_all))

    colors = plt.get_cmap('tab10').colors
    null_alpha = 0.35

    if facet:
        fig, axes = plt.subplots(len(traces), 1, sharex=True, sharey=True,
                                 figsize=(10, max(6, 2.0*len(traces))))
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])
        for i, (ax, tr) in enumerate(zip(axes, traces)):
            col = colors[i % len(colors)]
            ax.plot(tr['t'], tr['zcoh'], lw=linewidth, color=col)
            ax.hlines(tr['zthr'], tmin, tmax, colors=col, linestyles='--', lw=1.0, alpha=null_alpha)
            # Shade ignitions (clipped)
            if ignition_windows:
                for (t0, t1) in ignition_windows:
                    if t1 < tmin or t0 > tmax: continue
                    ax.axvspan(max(t0, tmin), min(t1, tmax), color='k', alpha=0.08, zorder=0)
            if grid: ax.grid(True, alpha=0.25, linestyle=':')
            ax.set_ylabel(f"{tr['f0']:.3g} Hz\n(w={tr['win_sec']:.0f}s, h={tr['half_bw']:.2f})")
        axes[-1].set_xlabel('Time (s)')
        fig.suptitle(f"SR z-coherence — {group_name} (adaptive windows)")
        fig.tight_layout()
        if out_png: fig.savefig(out_png, dpi=dpi, bbox_inches='tight')
        plt.show()
        return traces
    else:
        fig, ax = plt.subplots(figsize=(10, 4.6))
        for i, tr in enumerate(traces):
            col = colors[i % len(colors)]
            ax.plot(tr['t'], tr['zcoh'], lw=linewidth, color=col, label=f"{tr['f0']:.3g} Hz")
            ax.hlines(tr['zthr'], tmin, tmax, colors=col, linestyles='--', lw=1.0, alpha=null_alpha)
        if ignition_windows:
            for (t0, t1) in ignition_windows:
                if t1 < tmin or t0 > tmax: continue
                ax.axvspan(max(t0, tmin), min(t1, tmax), color='k', alpha=0.08, zorder=0)
        if grid: ax.grid(True, alpha=0.25, linestyle=':')
        ax.set_xlim(tmin, tmax)
        ax.set_xlabel('Time (s)'); ax.set_ylabel('z-coherence')
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), frameon=False)
        ax.set_title(f"SR z-coherence — {group_name} (adaptive windows)")
        fig.tight_layout()
        if out_png: fig.savefig(out_png, dpi=dpi, bbox_inches='tight')
        plt.show()
        return traces

# ----------------------------
# 4) Convenience: run all groups & return a tidy summary
# ----------------------------

def summarize_sr_groups(records, eeg_channel, sr_channel, ignition_windows=None,
                        groups=None, **plot_kwargs):
    groups = sr_groups() if groups is None else groups
    all_rows = []
    for name in groups:
        traces = plot_sr_group_adaptive(records, eeg_channel, sr_channel,
                                        group_name=name, ignition_windows=ignition_windows,
                                        **plot_kwargs)
        for tr in traces:
            z = tr['zcoh']; m = float(np.nanmedian(z)); mx = float(np.nanmax(z))
            cover = float(100.0*np.nanmean((z > tr['zthr']).astype(float)))
            all_rows.append({'group': name, 'f0': tr['f0'], 'win_sec': tr['win_sec'], 'half_bw': tr['half_bw'],
                             'median_z': m, 'max_z': mx, 'coverage_pct': cover})
    import pandas as pd
    return pd.DataFrame(all_rows)

# ----------------------------
# Example usage (uncomment):
# GROUPS = sr_groups()
# traces = plot_sr_group_adaptive(RECORDS, 'EEG.O1', 'EEG.Pz',
#                                 group_name='Harmonics_UpTo33',
#                                 ignition_windows=[(290,310),(580,600)],
#                                 facet=True, smooth=True, step_sec=1.0, n_null=200,
#                                 limit_high_hz=60.0)
# summary_df = summarize_sr_groups(RECORDS, 'EEG.O1', 'EEG.Pz',
#                                  ignition_windows=[(290,310),(580,600)],
#                                  facet=False, smooth=True)
# print(summary_df)
