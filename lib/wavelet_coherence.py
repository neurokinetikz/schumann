def plot_sr_ignition_wtc_strip(
    RECORDS,
    eeg_channel: str,                 # e.g., 'EEG.O1' (or your best posterior)
    sr_channel: str,                  # magnetometer if you have one; else posterior proxy
    ignition_windows: list,           # [(t0, t1), ...] in seconds
    time_col: str = 'Timestamp',
    fmin: float = 0.5, fmax: float = 59.8, n_freq: int = 64,
    harmonics=(7.83, 14.3, 20.8, 27.3, 33.8),   # show bands for these (you can add more)
    half_band: float = 0.6,           # ±Hz shading around each harmonic
    w0: float = 6.0,                  # Morlet parameter
    n_perm: int = 200, alpha: float = 0.05,
    out_png: str = 'sr_wtc_strip.png',
    show: bool = True
):
    import numpy as np
    import matplotlib.pyplot as plt

    # 1) Run WTC on the *full* record (so ignition spans align to absolute time)
    wtc = wavelet_coherence_tf(
        RECORDS, eeg_channel, sr_channel,
        time_col=time_col, fmin=fmin, fmax=fmax, n_freq=n_freq, w0=w0,
        n_perm=n_perm, alpha=alpha, wins=None, show=True, out_png=None
    )

    # 2) Build the time axis from the DataFrame
    t_all = np.asarray(pd.to_numeric(RECORDS[time_col], errors='coerce').values, float)
    # If wavelet_coherence_tf returned a trimmed/sliced length, map to the last N samples
    N = wtc['WTC'].shape[1]
    if len(t_all) != N:
        # use the last N timestamps to match WTC length
        t = t_all[-N:]
    else:
        t = t_all

    # 3) Plot WTC with cyan significant pixels and ignition shading
    plt.figure(figsize=(11, 4))
    extent = [t[0], t[-1], wtc['freqs'][0], wtc['freqs'][-1]]

    plt.imshow(
        wtc['WTC'], aspect='auto', origin='lower', extent=extent,
        cmap='magma', vmin=0, vmax=np.nanmax(wtc['WTC'])
    )
    cb = plt.colorbar(); cb.set_label('Wavelet coherence')

    # Cyan significant pixels (cluster-based, mask already thresholded vs shift-null)
    sig = wtc['sig_mask']
    if sig is not None and np.any(sig):
        yy, xx = np.where(sig)
        plt.scatter(t[xx], wtc['freqs'][yy], s=4, c='cyan', alpha=0.7, label='> null 95%')

    # Horizontal harmonic bands (±half_band)
    for h in harmonics:
        plt.axhspan(h - half_band, h + half_band, color='white', alpha=0.08)
        plt.axhline(h, color='white', lw=0.8, alpha=0.6)

    # Shade ignition windows
    for (t0, t1) in ignition_windows:
        plt.axvspan(t0, t1, color='k', alpha=0.08)

    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title(f'EEG–SR Wavelet Coherence (cyan = > null 95%; shaded = ignition)')
    if sig is not None and np.any(sig):
        plt.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    if out_png:
        plt.savefig(out_png, dpi=140)
    if show:
        plt.show()
    plt.close()
