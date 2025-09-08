# SR harmonic groups supplied by user + helpers to analyze/plot each group
# Requires: numpy, matplotlib, and your existing sliding_coherence_f0(...)
# Optional (already in your notebook/canvas): build_null_threshold, build_null_threshold_smooth, _auto_savgol

import numpy as np
import matplotlib.pyplot as plt

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
        sl = sliding_coherence_f0(records, eeg_channel, sr_channel, **kwargs)
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
