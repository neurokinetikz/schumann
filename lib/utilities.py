# ==== CONFIG ==============================
import warnings
warnings.filterwarnings("ignore")

import math
import numpy as np
import pandas as pd
from pathlib import Path

from scipy import signal
from mne.time_frequency import psd_array_multitaper

# Optional metrics (install if you have these packages)
try:
    from entropy import sample_entropy, permutation_entropy
except Exception:
    sample_entropy = None
    permutation_entropy = None

try:
    from lempel_ziv_complexity import lempel_ziv_complexity as lzc
except Exception:
    lzc = None

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import seaborn as sns

from mne.time_frequency import psd_array_welch

# Sampling rate and channel definitions
FS = 128  # Hz (change if your CSV has a different rate)
ELECTRODES = ['AF3','AF4','F7','F8','F3','F4','FC5','FC6','P7','P8','T7','T8','O1','O2']
BRAINWAVES = ['Delta','Theta','Alpha','BetaL','BetaH','Gamma']
RANGES  = {'Delta':[1,4],'Theta':[4,8],'Alpha':[8,12],'BetaL':[12,16], 'BetaH':[16,25],'Gamma':[25,45]}

# Helper to make full column names used in your CSV (e.g., "EEG.AF3")
col = lambda e: f"EEG.{e}"



def load_eeg_csv(csv_path, electrodes):
    """Load CSV as in user's snippet and return a pre-processed DataFrame.
    Expects columns: 'Timestamp' (seconds or ms) and 'EEG.<electrode>' per channel.
    """
    df = pd.read_csv(csv_path, low_memory=False, header=1).sort_values(by=['Timestamp']).reset_index(drop=True)

    # Normalize time to start at 0
    df['Timestamp'] = df['Timestamp'] - df['Timestamp'].iloc[0]
    # If Timestamp looks like milliseconds, convert to seconds
    if df['Timestamp'].iloc[-1] > 1e6:
        df['Timestamp'] = df['Timestamp'] / 1000.0

    # High-pass each electrode at 1 Hz and compute squared power
    for e in electrodes:
        ch = col(e)
        if ch not in df.columns:
            continue
        s = df[ch].astype(float)
        # Handle NaNs by forward/backward filling before filtering
        s = s.interpolate(limit_direction='both')
        filt = butter_highpass(s.values, cutoff_hz=1.0, fs=FS, order=2)
        df[f"{ch}.FILTERED"] = filt
        df[f"{ch}.FILTERED.POW"] = filt * filt

        # Band-pass per range, absolute & relative power, binary bins
        pow_cols = []
        for w, (f1, f2) in RANGES.items():
            band_sig = butter_bandpass(filt, f1, f2, fs=FS, order=4)
            band_col = f"{ch}.{w}"
            pow_col = f"POW.{ch}.{w}"
            df[band_col] = band_sig
            df[pow_col] = band_sig * band_sig
            pow_cols.append(pow_col)
            # complexity bins (threshold at band mean)
            df[f"{ch}.{w}.BIN"] = (band_sig < np.nanmean(band_sig)).astype(int)

        # Relative power across this channel's bands
        pow_sum = df[pow_cols].sum(axis=1).replace(0, np.nan)
        for w in RANGES.keys():
            pc = f"POW.{ch}.{w}"
            df[f"{pc}.REL"] = df[pc] / pow_sum

    return df

# ==== FILTER DESIGN HELPERS ===============
def butter_highpass(sig, cutoff_hz, fs=FS, order=2):
    nyq = fs/2.0
    b, a = signal.butter(order, cutoff_hz/nyq, btype='highpass')
    return signal.filtfilt(b, a, sig)

def butter_bandpass(sig, f_lo, f_hi, fs=FS, order=4):
    nyq = fs/2.0
    b, a = signal.butter(order, [f_lo/nyq, f_hi/nyq], btype='band')
    return signal.filtfilt(b, a, sig)

# ==== FEATURES ============================
def compute_psd_multitaper(sig, fs=FS, fmin=1, fmax=45):
    """Return freqs (Hz) and PSD using MNE multitaper on a 1D numpy array."""
    psd, freqs = psd_array_multitaper(sig, sfreq=fs, fmin=fmin, fmax=fmax, adaptive=True, normalization='full', verbose=False)
    return freqs, psd

def bandpowers_from_psd(freqs, psd, ranges=RANGES):
    """Integrate PSD over frequency bands (absolute) and compute relative shares."""
    bp_abs = {}
    for w, (f1, f2) in ranges.items():
        idx = np.logical_and(freqs >= f1, freqs <= f2)
        bp_abs[w] = np.trapz(psd[idx], freqs[idx]) if np.any(idx) else np.nan
    total = np.nansum(list(bp_abs.values()))
    bp_rel = {w: (bp_abs[w]/total if total and not np.isnan(bp_abs[w]) else np.nan) for w in bp_abs}
    return bp_abs, bp_rel

def binary_lzc(series):
    if lzc is None:
        return np.nan
    b = ''.join(series.astype(int).astype(str).tolist())
    return lzc(b)

def series_entropy(x):
    """Sample entropy wrapper (returns NaN if package missing)."""
    if sample_entropy is None:
        return np.nan
    try:
        return float(sample_entropy(x, order=2, metric='chebyshev'))
    except Exception:
        return np.nan

# ==== QUICKLOOK PLOTS =====================
def plot_channel_overview(df, electrode='AF3', seconds=10):
    ch = col(electrode)
    if f"{ch}.FILTERED" not in df.columns:
        print(f"Channel {ch} not found.")
        return
    n = int(seconds * FS)
    s = df[f"{ch}.FILTERED"].values[:n]
    t = df['Timestamp'].values[:n]

    fig, axes = plt.subplots(2, 2, figsize=(12, 6))
    axes = axes.ravel()

    # Time series
    axes[0].plot(t, s, lw=0.8)
    axes[0].set_title(f"{electrode} filtered (first {seconds}s)")
    axes[0].set_xlabel("Time (s)")

    # PSD
    freqs, psd = compute_psd_multitaper(s, fs=FS)
    axes[1].semilogy(freqs, psd)
    axes[1].set_xlim(1, 45)
    axes[1].set_title("PSD (multitaper)")
    axes[1].set_xlabel("Hz")

    # Band time series (Theta/Alpha)
    for i, w in enumerate(['Theta','Alpha']):
        if f"{ch}.{w}" in df.columns:
            axes[2+i].plot(t, df[f"{ch}.{w}"].values[:n], lw=0.8, label=w)
            axes[2+i].set_title(f"{w} band")
            axes[2+i].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.show()

def compute_relpower_table(df, electrodes=ELECTRODES, bands=list(RANGES.keys())):
    rows = []
    for e in electrodes:
        ch = col(e)
        # collect time-averaged relative power per band for this electrode
        band_vals = {}
        for w in bands:
            rel_col = f"POW.{ch}.{w}.REL"
            if rel_col in df.columns:
                band_vals[w] = np.nanmean(df[rel_col].values)
            else:
                band_vals[w] = np.nan
        rows.append({"electrode": e, **band_vals})
    rp = pd.DataFrame(rows)
    return rp


def plot_stacked_relpower(rp_df, bands=list(RANGES.keys())):
    # Sort electrodes in the order provided
    rp_df = rp_df.set_index('electrode').loc[[e for e in ELECTRODES if e in rp_df['electrode'].tolist()]] if 'electrode' in rp_df.columns else rp_df
    ax = rp_df[bands].plot(kind='bar', stacked=True, figsize=(12,6))
    ax.set_ylabel('Relative Power (fraction)')
    ax.set_xlabel('Electrode')
    ax.set_title('Relative Band Power per Electrode (stacked)')
    ax.legend(title='Band', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def plot_stacked_relpower_timeseries(
    df, electrodes=ELECTRODES, bands=list(RANGES.keys()),
    start_sec=0, end_sec=None, x_interval=None, fs=128,
    smooth_sec=1.0, psd_fft_win=8.0
):
    """
    For each electrode, plot the stacked relative power time series and the PSD in one row (two subplots).

    Parameters
    ----------
    df : DataFrame with relative power columns
    electrodes : list of str, electrode names
    bands : list of str, frequency band keys
    start_sec : float, start time (seconds)
    end_sec : float, end time (seconds). If None, goes to end of data.
    x_interval : float, tick interval on x-axis (seconds). If None, auto.
    fs : int, sampling frequency (Hz)
    smooth_sec : float, smoothing window duration in seconds for moving average
    psd_fft_win : float, FFT window length in seconds for PSD calculation
    """
    if 'Timestamp' not in df:
        raise ValueError("DataFrame must contain a 'Timestamp' column")

    t = df['Timestamp'].values
    if end_sec is None:
        end_sec = t[-1]
    if end_sec < start_sec:
        start_sec, end_sec = end_sec, start_sec
    mask = (t >= start_sec) & (t <= end_sec)

    win = int(fs * smooth_sec)

    for e in electrodes:
        ch = f"EEG.{e}"
        rel_cols = [f"POW.{ch}.{w}.REL" for w in bands if f"POW.{ch}.{w}.REL" in df.columns]
        if not rel_cols:
            continue
        needed_cols = ['Timestamp'] + rel_cols
        if f"{ch}.FILTERED" in df.columns:
            needed_cols.append(f"{ch}.FILTERED")
        sub = df.loc[mask, needed_cols].dropna()
        if sub.empty:
            continue

        smoothed = {}
        for c in rel_cols:
            smoothed[c] = sub[c].rolling(win, center=True, min_periods=1).mean().values

        fig, axes = plt.subplots(1, 2, figsize=(16, 3))

        # Left: stacked relative power
        axes[0].stackplot(sub['Timestamp'], [smoothed[c] for c in rel_cols], labels=bands)
        axes[0].set_title(f"{e} - Relative Band Power ({smooth_sec}s MA)")
        axes[0].set_xlabel("Time (s)")
        axes[0].set_ylabel("Fraction")
        axes[0].set_ylim(0, 1.0)
        if x_interval is not None and x_interval > 0:
            ticks = np.arange(start_sec, end_sec + 1e-9, x_interval)
            axes[0].set_xticks(ticks)
            axes[0].set_xlim(start_sec, end_sec)
        else:
            axes[0].set_xlim(start_sec, end_sec)
        axes[0].legend(loc='upper right', ncol=len(bands))

        # Right: PSD
        if f"{ch}.FILTERED" in sub.columns:
            sig = sub[f"{ch}.FILTERED"].values
            if sig.size > 0:
                fmin, fmax = RANGES['Theta']
                fmax = 10
                n_fft = int(fs * psd_fft_win)
                n_overlap = n_fft // 2
                psds, freqs = psd_array_welch(
                    sig, sfreq=fs, fmin=fmin, fmax=fmax,
                    n_fft=n_fft, n_overlap=n_overlap, verbose=False
                )
                axes[1].semilogy(freqs, psds)
                axes[1].set_xlim(fmin, fmax)
                axes[1].axvline(7.8, color='red', linestyle='--')
                # Add x-axis ticks including 7.8 Hz and highlight it in red
                xticks = list(axes[1].get_xticks()) + [7.8]
                axes[1].set_xticks(sorted(set(xticks)))
                labels = []
                for tick in axes[1].get_xticks():
                    if abs(tick - 7.8) < 1e-6:
                        labels.append("7.8")
                    else:
                        labels.append(str(round(tick, 1)))
                axes[1].set_xticklabels(labels, color='black')
                for label in axes[1].get_xticklabels():
                    if label.get_text() == '7.8':
                        label.set_color('red')
                        label.set_fontweight('bold')
                axes[1].set_xlabel("Frequency (Hz)")
                axes[1].set_ylabel("PSD (V^2/Hz)")
                axes[1].set_title(f"{e} - PSD (Theta {fmin}-{fmax} Hz, {psd_fft_win}s FFT)")

        plt.tight_layout()
        plt.show()

def _moving_average(x, k):
    k = int(max(1, k))
    if k == 1:
        return x
    w = np.ones(k, dtype=float) / k
    return np.convolve(x, w, mode='same')


def compute_iaf(
    df,
    electrodes,
    fs=128,
    start_sec=0.0,
    end_sec=None,
    alpha_band=(8.0, 13.0),
    roi=("O1","O2","POz","Pz","P3","P4"),
    psd_fft_win=4.0,
    psd_overlap=0.5,
    smooth_hz=0.5,
    return_psd=False,
):
    """
    Compute Individual Alpha Frequency (IAF) per electrode and ROI summary.

    Methods implemented:
      - PAF (peak alpha frequency): argmax of smoothed PSD within alpha band
      - CoG (center of gravity): spectral centroid within alpha band

    Parameters
    ----------
    df : pandas.DataFrame
        Must include 'Timestamp' and per-electrode filtered columns named f"EEG.<ELECTRODE>.FILTERED".
    electrodes : list[str]
        Electrode labels to consider (e.g., ELECTRODES).
    fs : float
        Sampling rate (Hz).
    start_sec, end_sec : float
        Time window to analyze (seconds). If end_sec is None, uses end of data.
    alpha_band : tuple(float,float)
        Alpha range (Hz) used for IAF estimation (default 8–13 Hz).
    roi : tuple[str]
        Occipital/parietal electrodes used for ROI summaries.
    psd_fft_win : float
        FFT window length in seconds for Welch PSD.
    psd_overlap : float
        Fractional overlap (0–1) between Welch windows.
    smooth_hz : float
        Width (Hz) of a simple moving-average smoother applied to PSD before peak picking.
    return_psd : bool
        If True, also return (freqs, psd_dict) for debugging/plotting.

    Returns
    -------
    results : pd.DataFrame
        Columns: electrode, PAF_Hz, CoG_Hz, alpha_power
        (alpha_power is total power in alpha band for context/quality).
    summaries : dict
        ROI-based summaries: median/mean PAF and CoG over roi electrodes that were found.
    (optional) freqs, psd_dict :
        Only if return_psd=True. psd_dict maps electrode -> PSD array (full band 1–45 Hz).
    """
    if 'Timestamp' not in df:
        raise ValueError("DataFrame must contain 'Timestamp'.")

    t = df['Timestamp'].values
    if end_sec is None:
        end_sec = float(t[-1])
    if end_sec < start_sec:
        start_sec, end_sec = end_sec, start_sec
    mask = (t >= start_sec) & (t <= end_sec)

    # Welch params
    n_fft = int(fs * psd_fft_win)
    n_overlap = int(n_fft * np.clip(psd_overlap, 0.0, 0.95))

    # Global PSD settings
    fmin_all, fmax_all = 1.0, 45.0
    fa, fb = alpha_band

    # To compute smoothing kernel length in bins, we need freq resolution after Welch.
    # We'll approximate with df ~ fs/n_fft (fine for large n_fft)
    approx_df = fs / n_fft if n_fft > 0 else 0.5
    k_smooth = int(max(1, round(smooth_hz / max(approx_df, 1e-6))))
    # Force odd length for symmetric smoothing
    if k_smooth % 2 == 0:
        k_smooth += 1

    rows = []
    psd_dict = {}
    freqs_all = None

    for e in electrodes:
        ch = f"EEG.{e}.FILTERED"
        if ch not in df.columns:
            continue
        x = df.loc[mask, ch].dropna().values.astype(float)
        if x.size < n_fft:
            # Not enough data to compute stable PSD
            continue

        psd, freqs = psd_array_welch(
            x, sfreq=fs, fmin=fmin_all, fmax=fmax_all,
            n_fft=n_fft, n_overlap=n_overlap, verbose=False
        )
        if freqs_all is None:
            freqs_all = freqs
        # Smooth PSD for robust peak finding
        psd_s = _moving_average(psd, k_smooth)

        # Alpha band indices
        idx_a = (freqs >= fa) & (freqs <= fb)
        if np.count_nonzero(idx_a) < 3:
            continue

        # PAF: location of maximum in smoothed PSD within alpha
        paf_idx = np.argmax(psd_s[idx_a])
        paf_hz = float(freqs[idx_a][paf_idx])

        # CoG: spectral centroid within alpha
        P = psd[idx_a]
        F = freqs[idx_a]
        cog_hz = float(np.sum(F * P) / np.sum(P)) if np.sum(P) > 0 else np.nan

        # Alpha power (area) for context/quality
        alpha_power = float(np.trapz(P, F))

        rows.append({
            'electrode': e,
            'PAF_Hz': paf_hz,
            'CoG_Hz': cog_hz,
            'alpha_power': alpha_power,
        })
        psd_dict[e] = psd

    if not rows:
        results = pd.DataFrame(columns=['electrode','PAF_Hz','CoG_Hz','alpha_power'])
    else:
        results = pd.DataFrame(rows)

    # ROI summaries
    roi_set = set(roi)
    res_roi = results[results['electrode'].isin(roi_set)]
    summaries = {}
    if not res_roi.empty:
        summaries['ROI_electrodes_used'] = list(res_roi['electrode'])
        summaries['PAF_median_Hz'] = float(res_roi['PAF_Hz'].median())
        summaries['PAF_mean_Hz']   = float(res_roi['PAF_Hz'].mean())
        summaries['CoG_median_Hz'] = float(res_roi['CoG_Hz'].median())
        summaries['CoG_mean_Hz']   = float(res_roi['CoG_Hz'].mean())
    else:
        summaries['ROI_electrodes_used'] = []
        summaries['PAF_median_Hz'] = np.nan
        summaries['PAF_mean_Hz']   = np.nan
        summaries['CoG_median_Hz'] = np.nan
        summaries['CoG_mean_Hz']   = np.nan

    if return_psd:
        return results, summaries, freqs_all, psd_dict
    return results, summaries

def animate_theta_alpha_psd(
    df,
    electrode,
    fs=128,
    start_sec=0.0,
    end_sec=None,
    win_sec=10.0,
    step_sec=1.0,
    psd_fft_win=4.0,
    psd_overlap=0.5,
    band=(4.0, 12.0),
    dpi=90,
    show_legend=True,
    save_path=None,
):
    """
    Create a timelapse animation of the combined theta+alpha PSD for a single electrode.
    Adds extra space above the chart so lines are not clipped.
    Exports as a looping animated GIF if save_path ends with .gif.

    Parameters
    ----------
    df : pandas.DataFrame
        Must include 'Timestamp' and column f"EEG.<electrode>.FILTERED".
    electrode : str
        Electrode label, e.g., 'O1'.
    fs : float
        Sampling rate (Hz).
    start_sec, end_sec : float
        Overall analysis window in seconds. If end_sec is None, uses end of data.
    win_sec : float
        Length (s) of each sliding window for PSD.
    step_sec : float
        Step (s) between successive windows.
    psd_fft_win : float
        Welch FFT segment length in seconds.
    psd_overlap : float
        Fractional overlap (0–1) between Welch segments.
    band : tuple(float,float)
        Frequency band (Hz) for PSD (default 4–13 Hz for theta+alpha).
    dpi : int
        Figure DPI for animation.
    show_legend : bool
        Whether to show the legend (7.8 Hz marker).
    save_path : str or None
        If provided, saves animation to this path (e.g., 'theta_alpha_psd_O1.mp4' or '.gif').

    Returns
    -------
    anim : matplotlib.animation.FuncAnimation
    """
    if 'Timestamp' not in df:
        raise ValueError("DataFrame must contain 'Timestamp'.")
    ch = f"EEG.{electrode}.FILTERED"
    if ch not in df.columns:
        raise ValueError(f"Column '{ch}' not found in DataFrame.")

    t = df['Timestamp'].values.astype(float)
    if end_sec is None:
        end_sec = float(t[-1])
    if end_sec < start_sec:
        start_sec, end_sec = end_sec, start_sec

    mask_all = (t >= start_sec) & (t <= end_sec)
    x_all = df.loc[mask_all, ch].dropna().values.astype(float)
    t_all = df.loc[mask_all, 'Timestamp'].values.astype(float)
    if x_all.size < int(fs * win_sec):
        raise ValueError("Not enough samples in the selected window for the requested win_sec.")

    hop = int(round(step_sec * fs))
    win_n = int(round(win_sec * fs))
    starts = np.arange(0, x_all.size - win_n + 1, hop, dtype=int)

    n_fft = int(round(psd_fft_win * fs))
    n_overlap = int(round(n_fft * np.clip(psd_overlap, 0.0, 0.95)))

    fmin, fmax = band

    from mne.time_frequency import psd_array_welch as _welch
    psd0, freqs = _welch(x_all[:win_n], sfreq=fs, fmin=fmin, fmax=fmax,
                         n_fft=n_fft, n_overlap=n_overlap, verbose=False)

    fig, ax = plt.subplots(figsize=(8, 4), dpi=dpi)  # Increased height for more top space
    (psd_line,) = ax.plot(freqs, np.maximum(psd0, 1e-20), lw=1.5)
    ax.set_xlim(fmin, fmax)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (V$^2$/Hz)")
    vline = ax.axvline(7.8, color='red', linestyle='--')
    vline = ax.axvline(14.3, color='red', linestyle='--')
    vline = ax.axvline(20.8, color='red', linestyle='--')
    vline = ax.axvline(27.3, color='red', linestyle='--')
    vline = ax.axvline(30.8, color='red', linestyle='--')

    xt = list(ax.get_xticks())
    if 7.8 < fmax and 7.8 > fmin:
        xt += [7.8]
    ax.set_xticks(sorted(set(xt)))
    labels = []
    for tick in ax.get_xticks():
        if abs(tick - 7.8) < 1e-6:
            labels.append('7.8')
        else:
            labels.append(str(round(float(tick), 1)))
    ax.set_xticklabels(labels, color='black')
    for lab in ax.get_xticklabels():
        if lab.get_text() == '7.8':
            lab.set_color('red')
            lab.set_fontweight('bold')
    if show_legend:
        ax.legend([vline], ["7.8 Hz"], loc='upper right')

    title = ax.set_title("")
    plt.subplots_adjust(top=0.85)  # More space at top to prevent clipping

    ax.set_ylim(0,1750)

    def _update(i):
        s = starts[i]
        seg = x_all[s : s + win_n]
        psd, _ = _welch(seg, sfreq=fs, fmin=fmin, fmax=fmax,
                        n_fft=n_fft, n_overlap=n_overlap, verbose=False)
        psd = np.maximum(psd, 1e-20)
        psd_line.set_ydata(psd)
        t0 = t_all[s] - t_all[0] + start_sec
        t1 = t0 + win_sec
        title.set_text(f"{electrode}  |  Theta+Alpha PSD  |  window: {t0:.1f}–{t1:.1f} s  (win={win_sec:.1f}s, step={step_sec:.1f}s)")
        return (psd_line, title)

    anim = FuncAnimation(fig, _update, frames=len(starts), interval=200, blit=False, repeat=True)

    if save_path:
        if save_path.lower().endswith('.gif'):
            writer = PillowWriter(fps=5)
            anim.save(save_path, writer=writer, dpi=dpi)
        else:
            try:
                anim.save(save_path, writer='ffmpeg', dpi=dpi)
            except Exception:
                writer = PillowWriter(fps=5)
                anim.save(save_path, writer=writer, dpi=dpi)
    return anim

def plot_gfp_and_theta_alpha(df, electrodes, fs=128, bands=None, smooth_window=1.0):
    """
    Plot Global Field Power (GFP) and theta/alpha band power over the entire recording.

    Parameters
    ----------
    df : DataFrame
        EEG data with columns EEG.<electrode>.FILTERED and EEG.<electrode>.<band> if available.
    electrodes : list[str]
        Electrode labels.
    fs : int
        Sampling frequency in Hz.
    bands : dict
        Frequency bands, defaults to {'theta': (4,8), 'alpha': (8,13)}.
    smooth_window : float
        Moving average smoothing window in seconds.
    """
    if bands is None:
        bands = {'theta': (4, 8), 'alpha': (8, 13)}

    t = df['Timestamp'].values
    n_times = len(t)
    win_samples = max(1, int(round(smooth_window * fs)))

    # Stack electrode signals
    data = []
    for e in electrodes:
        col = f"EEG.{e}.FILTERED"
        if col in df.columns:
            data.append(df[col].values.astype(float))
    X = np.vstack(data)

    # Compute GFP
    mean_across = X.mean(axis=0)
    gfp = ((X - mean_across) ** 2).mean(axis=0)
    gfp_smooth = pd.Series(gfp).rolling(win_samples, center=True, min_periods=1).mean().values

    # Compute per-band GFP (theta, alpha)
    band_gfp = {}
    for band in bands.keys():
        band_data = []
        for e in electrodes:
            col = f"EEG.{e}.{band}"
            if col in df.columns:
                band_data.append(df[col].values.astype(float))
        if band_data:
            Xb = np.vstack(band_data)
            mean_across = Xb.mean(axis=0)
            band_gfp[band] = ((Xb - mean_across) ** 2).mean(axis=0)
            band_gfp[band] = pd.Series(band_gfp[band]).rolling(win_samples, center=True, min_periods=1).mean().values

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(t/60.0, gfp_smooth, label='GFP', color='black', linewidth=1.5)
    for band, vals in band_gfp.items():
        plt.plot(t/60.0, vals, label=f'{band} GFP')

    plt.xlabel('Time (minutes)')
    plt.ylabel('Power (a.u.)')
    plt.title('Global Field Power and Theta/Alpha GFP Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def compute_gfp(data):
    """
    Compute Global Field Power (GFP) across electrodes.

    Parameters
    ----------
    data : np.ndarray, shape (n_channels, n_times)
        EEG data array.

    Returns
    -------
    gfp : np.ndarray, shape (n_times,)
        Global Field Power time series.
    """
    mean_across = np.mean(data, axis=0)
    gfp = np.mean((data - mean_across) ** 2, axis=0)
    return gfp


def detect_power_spike_events(
    df,
    electrodes,
    fs=128,
    bands=((4,8),(8,13),(13,30)),
    threshold=3.0,
    min_duration=5.0
):
    """
    Detect meditation-specific spectral bursts defined as epochs where GFP exceeds baseline mean by >3 SDs
    across ≥3 frequency bands simultaneously, persisting ≥5 s.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing EEG.<electrode>.FILTERED columns.
    electrodes : list of str
        Electrode names to include.
    fs : int
        Sampling rate (Hz).
    bands : list of tuple
        Frequency bands to consider.
    threshold : float
        Threshold in SDs above baseline mean.
    min_duration : float
        Minimum duration (seconds) to persist.

    Returns
    -------
    events : list of dict
        Each dict contains start_time, end_time, and involved bands.
    """
    # Build data array
    data = []
    for e in electrodes:
        ch = f"EEG.{e}.FILTERED"
        if ch not in df.columns:
            continue
        data.append(df[ch].values)
    data = np.vstack(data)

    # Compute GFP
    gfp = compute_gfp(data)
    t = df['Timestamp'].values

    # Baseline mean/std from first 30s
    baseline_mask = t <= (t[0] + 30.0)
    baseline_mean = np.mean(gfp[baseline_mask])
    baseline_std = np.std(gfp[baseline_mask])

    # Compute band-limited GFP signals
    band_gfps = {}
    for fmin, fmax in bands:
        psds, freqs = psd_array_welch(data, sfreq=fs, fmin=fmin, fmax=fmax, n_fft=fs*2, n_overlap=fs, verbose=False)
        # GFP from PSD across channels
        band_gfps[(fmin,fmax)] = np.mean(psds, axis=0)

    # Identify threshold crossings
    over_thresh = {}
    for band, vals in band_gfps.items():
        over_thresh[band] = vals > (baseline_mean + threshold*baseline_std)

    # Combine criteria: ≥3 bands simultaneously
    over_matrix = np.array(list(over_thresh.values()))  # shape (n_bands, n_freqs)
    simultaneous = np.sum(over_matrix, axis=0) >= 3

    # Find contiguous segments lasting ≥min_duration
    events = []
    min_samples = int(min_duration * fs)
    in_event = False
    start_idx = None
    for i, val in enumerate(simultaneous):
        if val and not in_event:
            in_event = True
            start_idx = i
        elif not val and in_event:
            if i - start_idx >= min_samples:
                events.append({
                    'start_time': t[start_idx],
                    'end_time': t[i-1],
                    'bands': [band for band, mask in over_thresh.items() if np.any(mask[start_idx:i])]
                })
            in_event = False

    return events

# =============================
# Event Detection: Meditation-Specific Spectral Bursts via GFP
# =============================
import numpy as np
import pandas as pd
from scipy import signal


def _butter_bandpass(sig, fs, f_lo, f_hi, order=4):
    nyq = fs / 2.0
    b, a = signal.butter(order, [f_lo / nyq, f_hi / nyq], btype='band')
    return signal.filtfilt(b, a, sig)


def compute_gfp_multichannel(X):
    """Compute Global Field Power (GFP) over time.
    X: 2D array, shape (n_channels, n_times)
    Returns: 1D array gfp(t) = (1/N) * sum_i (x_i(t) - mean_i(t))^2
    """
    if X.ndim != 2:
        raise ValueError("X must be 2D: (n_channels, n_times)")
    xbar = X.mean(axis=0, keepdims=True)
    return ((X - xbar) ** 2).mean(axis=0)


def compute_band_gfp(df, electrodes, fs, bands, use_existing_cols=True):
    """Compute band-limited GFP time series per band.

    Parameters
    ----------
    df : DataFrame with 'Timestamp' and EEG channel columns
    electrodes : list[str] electrode labels (e.g., ['AF3','AF4',...])
    fs : float sampling rate (Hz)
    bands : dict like {'theta':(4,8), 'alpha':(8,13), ...}
    use_existing_cols : bool
        If True, will try to use columns named f"EEG.{e}.{band}" if present.
        Otherwise, bandpass-filter from f"EEG.{e}.FILTERED".

    Returns
    -------
    band_gfp : dict[band] -> np.ndarray of shape (n_times,)
    channels_used : dict[band] -> list of channel column names used
    """
    t = df['Timestamp'].values
    n_times = t.size
    band_gfp = {}
    channels_used = {}

    for band, (f1, f2) in bands.items():
        chan_cols = []
        X_list = []
        for e in electrodes:
            if use_existing_cols:
                col = f"EEG.{e}.{band}"
                if col in df.columns:
                    x = df[col].values.astype(float)
                    X_list.append(x)
                    chan_cols.append(col)
                    continue
            # fallback: filter from FILTERED column
            base_col = f"EEG.{e}.FILTERED"
            if base_col in df.columns:
                x = df[base_col].values.astype(float)
                x = _butter_bandpass(x, fs, f1, f2, order=4)
                X_list.append(x)
                chan_cols.append(base_col)
        if not X_list:
            # no channels found for this band
            continue
        X = np.vstack(X_list)
        band_gfp[band] = compute_gfp_multichannel(X)
        channels_used[band] = chan_cols

    return band_gfp, channels_used


def detect_power_spike_events(
    band_gfp,
    fs,
    baseline_slice=None,
    z_thresh=3.0,
    min_bands=3,
    min_duration_s=5.0,
):
    """Detect power spike events where GFP exceeds baseline mean+z*std
    simultaneously in >= min_bands bands, persisting >= min_duration_s.

    Parameters
    ----------
    band_gfp : dict[band] -> 1D array of GFP over time (same length across bands)
    fs : float sampling rate (Hz)
    baseline_slice : slice or (start_idx, end_idx) or None
        Region used to compute baseline mean/std for each band.
        If None, uses the first 60 seconds by default.
    z_thresh : float, threshold in SD units (>3 SD per spec)
    min_bands : int, number of bands simultaneously exceeding threshold
    min_duration_s : float, minimum contiguous duration in seconds

    Returns
    -------
    events_df : DataFrame with columns ['start_sec','end_sec','duration_s','max_bands_overlap']
    over_threshold_mask : 1D boolean array for combined criterion (>=min_bands)
    per_band_masks : dict[band] -> boolean mask of timepoints over threshold
    baselines : DataFrame with per-band mean and std used
    """
    if not band_gfp:
        raise ValueError("band_gfp is empty")

    bands = list(band_gfp.keys())
    n_times = len(next(iter(band_gfp.values())))

    # Normalize baseline indices
    if baseline_slice is None:
        # default: first 60 s
        n_base = int(60 * fs)
        baseline_idx = np.arange(min(n_times, n_base))
    elif isinstance(baseline_slice, slice):
        baseline_idx = np.arange(n_times)[baseline_slice]
    elif isinstance(baseline_slice, (tuple, list)) and len(baseline_slice) == 2:
        s0, s1 = baseline_slice
        baseline_idx = np.arange(s0, min(n_times, s1))
    else:
        raise ValueError("Invalid baseline_slice")

    # Compute per-band thresholds
    baseline_stats = []
    per_band_masks = {}
    for b in bands:
        g = np.asarray(band_gfp[b])
        mu = g[baseline_idx].mean()
        sd = g[baseline_idx].std(ddof=1) if baseline_idx.size > 1 else g.std(ddof=1)
        thr = mu + z_thresh * sd
        per_band_masks[b] = g > thr
        baseline_stats.append({'band': b, 'mean': mu, 'std': sd, 'threshold': thr})
    baselines = pd.DataFrame(baseline_stats)

    # Combine across bands: at each timepoint, count how many bands exceed thr
    counts = np.zeros(n_times, dtype=int)
    for b in bands:
        counts += per_band_masks[b].astype(int)
    over_threshold_mask = counts >= int(min_bands)

    # Find contiguous segments meeting min duration
    min_len = int(round(min_duration_s * fs))
    events = []
    in_seg = False
    seg_start = 0
    for i, flag in enumerate(over_threshold_mask):
        if flag and not in_seg:
            in_seg = True
            seg_start = i
        elif not flag and in_seg:
            seg_end = i  # exclusive
            if seg_end - seg_start >= min_len:
                duration_s = (seg_end - seg_start) / fs
                max_overlap = counts[seg_start:seg_end].max()
                events.append((seg_start, seg_end, duration_s, max_overlap))
            in_seg = False
    # Handle case where mask ends in True
    if in_seg:
        seg_end = n_times
        if seg_end - seg_start >= min_len:
            duration_s = (seg_end - seg_start) / fs
            max_overlap = counts[seg_start:seg_end].max()
            events.append((seg_start, seg_end, duration_s, max_overlap))

    # Build event table
    if events:
        events_df = pd.DataFrame(
            [
                {
                    'start_idx': s0,
                    'end_idx': s1,
                    'start_sec': s0 / fs,
                    'end_sec': s1 / fs,
                    'duration_s': dur,
                    'max_bands_overlap': maxov,
                }
                for (s0, s1, dur, maxov) in events
            ]
        )
    else:
        events_df = pd.DataFrame(columns=['start_idx','end_idx','start_sec','end_sec','duration_s','max_bands_overlap'])

    return events_df, over_threshold_mask, per_band_masks, baselines


def run_event_detection_pipeline(
    df,
    electrodes,
    fs,
    bands=None,
    baseline_slice=None,
    z_thresh=3.0,
    min_bands=3,
    min_duration_s=5.0,
    use_existing_cols=True,
):
    """High-level helper that computes band GFP and detects events.

    bands default covers common ranges: delta/theta/alpha/beta/gamma (up to 45 Hz).
    """
    if bands is None:
        bands = {
            'delta': (1, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 45),
        }
    band_gfp, channels_used = compute_band_gfp(df, electrodes, fs, bands, use_existing_cols=use_existing_cols)
    events_df, combo_mask, per_band_masks, baselines = detect_power_spike_events(
        band_gfp=band_gfp,
        fs=fs,
        baseline_slice=baseline_slice,
        z_thresh=z_thresh,
        min_bands=min_bands,
        min_duration_s=min_duration_s,
    )
    return {
        'events': events_df,
        'combo_mask': combo_mask,
        'per_band_masks': per_band_masks,
        'band_gfp': band_gfp,
        'baselines': baselines,
        'channels_used': channels_used,
    }


# =============================
# Plot GFP (theta/alpha) over 10 minutes and detect recurring "mountains"
# =============================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths


def _moving_average(x, win):
    if win <= 1:
        return x
    w = np.ones(int(win), dtype=float)
    w /= w.sum()
    return np.convolve(x, w, mode='same')


def plot_pps_mountains(
    df,
    electrodes,
    fs,
    start_sec=0.0,
    duration_sec=600.0,  # 10 minutes default
    bands=("theta", "alpha"),
    smooth_sec=2.0,
    baseline_first_sec=60.0,
    z_thresh=2.5,
    prominence=0.5,
    min_width_sec=5.0,
    min_distance_sec=10.0,
    use_existing_cols=True,
):
    """
    Plot band-limited GFP for theta/alpha over a 10-minute window and mark
    recurring "mountains" (candidate PPS boundary expansions) as prominent peaks
    in the combined theta+alpha GFP (z-scored vs. local baseline).

    Returns a dict with the figure and a DataFrame of detected peaks.
    """
    # Build band dict from requested bands
    band_defs = {}
    canonical = {
        'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30), 'gamma': (30, 45)
    }
    for b in bands:
        if b not in canonical:
            raise ValueError(f"Unknown band '{b}'")
        band_defs[b] = canonical[b]

    # Compute band-limited GFP
    band_gfp, channels_used = compute_band_gfp(
        df=df, electrodes=electrodes, fs=fs, bands=band_defs, use_existing_cols=use_existing_cols
    )
    if not band_gfp:
        raise ValueError("No band GFP computed (check column names/electrodes)")

    t = df['Timestamp'].values.astype(float)
    t0 = start_sec
    t1 = min(t[-1], start_sec + duration_sec)
    mask = (t >= t0) & (t <= t1)
    if not np.any(mask):
        raise ValueError("Selected time window has no samples")

    # Extract series and optionally smooth
    win = max(1, int(round(smooth_sec * fs)))
    series = {}
    for b, g in band_gfp.items():
        g = np.asarray(g)
        gw = g[mask]
        if smooth_sec > 0:
            gw = _moving_average(gw, win)
        series[b] = gw

    # Combine theta+alpha GFP (sum) for peak finding
    combo = np.zeros_like(next(iter(series.values())))
    for b in bands:
        if b in series:
            combo += series[b]
    twin = t[mask]

    # Baseline within the *window*
    base_end = min(t1, t0 + baseline_first_sec)
    bmask = (twin >= t0) & (twin <= base_end)
    if not np.any(bmask):
        # fallback to first 60s of recording
        bmask = twin <= (t[0] + baseline_first_sec)
    mu, sd = float(np.mean(combo[bmask])), float(np.std(combo[bmask], ddof=1))
    z = (combo - mu) / (sd + 1e-12)

    # Peak finding on z-scored combined GFP
    distance = int(round(min_distance_sec * fs))
    width = int(round(min_width_sec * fs))
    peaks, props = find_peaks(
        z, height=z_thresh, prominence=prominence, distance=distance, width=width
    )
    # Peak widths at half-prominence to estimate mountain spans
    widths, w_h, left_ips, right_ips = peak_widths(z, peaks, rel_height=0.5)

    # Prepare plotting
    fig, ax = plt.subplots(figsize=(12, 5))
    # Plot per-band GFP
    for b in bands:
        if b in series:
            ax.plot(twin - t0, series[b], label=f"{b} GFP")
    # Overplot combined z (scaled to band units for visibility)
    ax2 = ax.twinx()
    ax2.plot(twin - t0, z, color='k', alpha=0.35, lw=1.0, label='Combined z')
    ax2.set_ylabel('z-score (θ+α GFP)')

    # Shade mountains
    for k, p in enumerate(peaks):
        l = max(0, int(np.floor(left_ips[k])))
        r = min(len(z)-1, int(np.ceil(right_ips[k])))
        ax.axvspan((twin[l]-t0), (twin[r]-t0), color='orange', alpha=0.20)
        ax.axvline((twin[p]-t0), color='orange', lw=1.0)

    ax.set_xlabel('Time in window (s)')
    ax.set_ylabel('GFP (band-limited)')
    ax.set_title('Theta/Alpha GFP over 10 min with PPS "mountains" (peaks in combined z)')
    ax.set_xlim(0, t1 - t0)

    # Legends
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    if h2:
        ax2.legend(h1 + h2, l1 + l2, loc='upper right')
    elif h1:
        ax.legend(loc='upper right')

    # Collect peak table
    rows = []
    for k, p in enumerate(peaks):
        rows.append({
            'peak_time_sec': float(twin[p] - t0),
            'peak_z': float(z[p]),
            'left_sec': float(twin[max(0, int(np.floor(left_ips[k])))] - t0),
            'right_sec': float(twin[min(len(z)-1, int(np.ceil(right_ips[k])))] - t0),
            'width_sec': float(widths[k] / fs),
            'prominence': float(props['prominences'][k]),
        })
    peaks_df = pd.DataFrame(rows)

    return {
        'fig': fig,
        'peaks': peaks_df,
        'window': (t0, t1),
        'baseline': {'mu': mu, 'sd': sd, 'z_thresh': z_thresh},
    }

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mne.time_frequency import psd_array_welch


def compute_aperiodic_slope_timeseries(
    df,
    electrodes,
    fs,
    start_sec=0.0,
    end_sec=None,
    win_sec=4.0,
    step_sec=2.0,
    fmin=1.0,
    fmax=45.0,
    exclude_bands_for_fit=((8, 13), (18, 25)),
    channel_suffix='.FILTERED',
):
    """Compute time series of the aperiodic 1/f slope (beta exponent) per electrode."""
    t = df['Timestamp'].values.astype(float)
    if end_sec is None:
        end_sec = float(t[-1])
    if end_sec < start_sec:
        start_sec, end_sec = end_sec, start_sec

    mask = (t >= start_sec) & (t <= end_sec)
    if not np.any(mask):
        raise ValueError('Selected time window has no samples')

    x_all = {}
    for e in electrodes:
        col = f"EEG.{e}{channel_suffix}"
        if col in df.columns:
            x_all[e] = df[col].values.astype(float)
    if not x_all:
        raise ValueError('No matching EEG.<electrode>.FILTERED columns found')

    hop = int(round(step_sec * fs))
    wlen = int(round(win_sec * fs))
    idx = np.where(mask)[0]
    start_idx = idx[0]
    end_idx = idx[-1] + 1
    starts = np.arange(start_idx, max(start_idx, end_idx - wlen + 1), hop)
    if starts.size == 0:
        starts = np.array([start_idx])
    centers = starts + wlen // 2
    times = (centers / fs).astype(float)

    n_fft = max(256, int(2 ** np.ceil(np.log2(wlen))))
    n_overlap = int(0.5 * wlen)

    slope_df = pd.DataFrame(index=times, columns=sorted(x_all.keys()), dtype=float)

    def _fit_mask(freqs):
        m = (freqs >= fmin) & (freqs <= fmax)
        for (a, b) in (exclude_bands_for_fit or ()):
            m &= ~((freqs >= a) & (freqs <= b))
        return m

    for e, sig in x_all.items():
        for wi, s0 in enumerate(starts):
            seg = sig[s0:s0 + wlen]
            if seg.size < wlen:
                pad = np.full(wlen - seg.size, seg[-1] if seg.size else 0.0)
                seg = np.concatenate([seg, pad])
            psd, freqs = psd_array_welch(seg, sfreq=fs, fmin=fmin, fmax=fmax,
                                         n_fft=n_fft, n_overlap=min(n_overlap, wlen - 1), verbose=False)
            psd = np.asarray(psd).ravel()
            freqs = np.asarray(freqs).ravel()

            fit_m = _fit_mask(freqs)
            X = np.log10(freqs[fit_m])
            Y = np.log10(np.maximum(psd[fit_m], 1e-20))
            if X.size >= 5:
                A = np.vstack([np.ones_like(X), X]).T
                beta_lin, _, _, _ = np.linalg.lstsq(A, Y, rcond=None)
                a, b = beta_lin[0], beta_lin[1]
                beta_exp = -b
            else:
                beta_exp = np.nan
            slope_df.loc[times[wi], e] = beta_exp

    return {'times': times, 'slope': slope_df, 'freqs': freqs}


def plot_aperiodic_slope_timeseries(
    df,
    electrodes,
    fs,
    start_sec=0.0,
    end_sec=None,
    win_sec=4.0,
    step_sec=2.0,
    fmin=1.0,
    fmax=45.0,
    exclude_bands_for_fit=((8, 13), (18, 25)),
    ylim=None,
    figsize=(10, 2.5)
):
    """
    Plot the aperiodic 1/f slope (beta exponent) over time for each electrode.
    """
    res = compute_aperiodic_slope_timeseries(
        df=df,
        electrodes=electrodes,
        fs=fs,
        start_sec=start_sec,
        end_sec=end_sec,
        win_sec=win_sec,
        step_sec=step_sec,
        fmin=fmin,
        fmax=fmax,
        exclude_bands_for_fit=exclude_bands_for_fit,
    )

    T = np.asarray(res['times'])
    slopes = res['slope'].astype(float)

    n_elec = len(electrodes)
    fig, axes = plt.subplots(n_elec, 1, figsize=(figsize[0], figsize[1]*n_elec), sharex=True)
    if n_elec == 1:
        axes = [axes]

    for ax, e in zip(axes, electrodes):
        y = slopes[e].values
        ax.plot(T, y, lw=1.8)
        ax.set_ylabel(f"{e} β")
        ax.grid(alpha=0.3)
        if ylim:
            ax.set_ylim(*ylim)
    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.show()

    return fig, axes

# =========================================
# GRAPH EEG TIMELINE BY BAND FOR EACH ELECTRODE
# =========================================
def graph_eeg_timeline(df, electrodes, ranges=RANGES, time_col='Timestamp', start_time=None, end_time=None):
    """
    Create a timeline plot of EEG power for each frequency band by electrode.

    Parameters:
    - df: DataFrame containing EEG data with band power columns like 'POW.EEG.F3.alpha'.
    - electrodes: list of electrode names (e.g. ['F3','F4','O1','O2']).
    - ranges: dict of band ranges (keys are band names like 'delta','theta','alpha','beta','gamma').
    - time_col: name of the time column (default 'Timestamp').
    - start_time: float | None, lower bound of time window in seconds.
    - end_time: float | None, upper bound of time window in seconds.
    """

    bands = list(ranges.keys())
    n_bands = len(bands)
    n_elec = len(electrodes)

    fig, axes = plt.subplots(n_bands, n_elec, figsize=(4*n_elec, 3*n_bands), sharex=True, sharey=False)
    if n_bands == 1:
        axes = [axes]
    if n_elec == 1:
        axes = [[ax] for ax in axes]

    for i, b in enumerate(bands):
        for j, e in enumerate(electrodes):
            ax = axes[i][j]
            pow_col = f"POW.EEG.{e}.{b}"
            if pow_col not in df.columns:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                ax.set_axis_off()
                continue

            t = df[time_col]
            y = df[pow_col]

            # Apply time window mask if specified
            mask = pd.Series(True, index=df.index)
            if start_time is not None:
                mask &= (t >= start_time)
            if end_time is not None:
                mask &= (t <= end_time)

            sns.lineplot(x=t[mask], y=y[mask], ax=ax, color='blue')
            ax.set_title(f"{e} — {b}")
            ax.set_ylabel("Power (µV²)")

    for ax in axes[-1]:
        ax.set_xlabel("Time (s)")

    plt.tight_layout()
    plt.show()

# ==========================
# EEG TIMELINE GRID BY BAND × ELECTRODE
# ==========================
def plot_eeg_timeline_grid(df, electrodes, ranges=RANGES, time_col='Timestamp',
                           start_time=None, end_time=None, sharey=False,
                           figsize_per_cell=(4.0, 2.2)):
    """
    Create a timeline plot for each frequency band by electrode (grid of subplots).

    Grid layout: rows = bands (order of keys in `ranges`), columns = electrodes.
    Each cell shows the time-series for that electrode's band power ('POW.EEG.{e}.{band}').

    Parameters
    ----------
    df : DataFrame
        Must include a time column and band-power columns such as 'POW.EEG.F3.alpha'.
    electrodes : list[str]
        Electrode names, e.g. ['F3','F4','O1','O2'].
    ranges : dict[str, (low, high)]
        Band definitions; keys define the row order in the grid.
    time_col : str
        Timestamp column name in seconds.
    start_time, end_time : float | None
        Optional time window.
    sharey : bool
        Share y-axis across subplots (per column) to facilitate comparison.
    figsize_per_cell : (float, float)
        Size per subplot cell; overall figure scales with rows × cols.
    """
    bands = list(ranges.keys())
    n_rows = len(bands)
    n_cols = len(electrodes)

    fig_w = figsize_per_cell[0] * max(1, n_cols)
    fig_h = figsize_per_cell[1] * max(1, n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), sharex=True, sharey=sharey)
    # Normalize axes indexing to 2D
    if n_rows == 1 and n_cols == 1:
        axes = [[axes]]
    elif n_rows == 1:
        axes = [axes]
    elif n_cols == 1:
        axes = [[ax] for ax in axes]

    t_all = df[time_col]
    mask = pd.Series(True, index=df.index)
    if start_time is not None:
        mask &= (t_all >= start_time)
    if end_time is not None:
        mask &= (t_all <= end_time)

    t = t_all[mask]

    for r, band in enumerate(bands):
        for c, e in enumerate(electrodes):
            ax = axes[r][c]
            col = f"POW.EEG.{e}.{band}"
            if col in df.columns:
                y = df[col][mask]
                ax.plot(t, y, linewidth=1.0)
            else:
                ax.text(0.5, 0.5, 'missing', ha='center', va='center', fontsize=9, color='crimson', transform=ax.transAxes)
            if r == 0:
                ax.set_title(e)
            if c == 0:
                ax.set_ylabel(f"{band}\nPower (µV²)")
            if r == n_rows - 1:
                ax.set_xlabel("Time (s)")

    fig.suptitle("EEG Band Timelines by Electrode", y=0.995)
    fig.tight_layout()
    plt.show()
