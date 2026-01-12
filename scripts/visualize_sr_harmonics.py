"""
Schumann Resonance Harmonics Visualization
==========================================
Clear, publication-quality visualizations of SR harmonic detection results.

Includes:
1. Power Spectral Density with harmonic markers
2. Time-frequency heatmap of harmonic activity
3. Schumann Activity Index over time
4. Harmonic coherence/phase relationships
5. Comparison of detected vs canonical harmonics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap
from scipy import signal
from scipy.signal import welch, coherence
import sys
sys.path.insert(0, './lib')

# Schumann Resonance canonical frequencies
SR_CANONICAL = {
    'f0': 7.83,   # Fundamental
    'f1': 14.3,   # 2nd harmonic
    'f2': 20.8,   # 3rd harmonic
    'f3': 27.3,   # 4th harmonic
    'f4': 33.8,   # 5th harmonic
    'f5': 40.3,   # 6th harmonic
}

# Subharmonics
SR_SUBHARMONICS = {
    'f0/2': 3.915,
    'f0/3': 2.61,
    'f0/4': 1.9575,
}

# EEG band definitions
EEG_BANDS = {
    'Delta': (1, 4),
    'Theta': (4, 8),
    'Alpha': (8, 12),
    'Beta-L': (12, 16),
    'Beta-H': (16, 25),
    'Gamma': (25, 45),
}


def generate_synthetic_eeg_with_sr(duration_sec=60, fs=128, sr_strength=0.3,
                                   noise_level=1.0, seed=42):
    """
    Generate synthetic EEG data with embedded Schumann Resonance harmonics.

    This creates realistic-looking EEG with:
    - 1/f background noise
    - Alpha oscillations (~10 Hz)
    - Embedded SR harmonics with controlled strength
    """
    np.random.seed(seed)
    n_samples = int(duration_sec * fs)
    t = np.arange(n_samples) / fs

    # 1/f (pink) noise background
    freqs = np.fft.rfftfreq(n_samples, 1/fs)
    freqs[0] = 1e-10  # Avoid division by zero
    pink_spectrum = 1 / np.sqrt(freqs)
    pink_spectrum[0] = 0
    phases = np.random.uniform(0, 2*np.pi, len(pink_spectrum))
    pink_noise = np.fft.irfft(pink_spectrum * np.exp(1j * phases), n_samples)
    pink_noise = pink_noise / np.std(pink_noise) * noise_level

    # Alpha oscillations (8-12 Hz, dominant around 10 Hz)
    alpha = 0.5 * np.sin(2 * np.pi * 10 * t + np.random.uniform(0, 2*np.pi))
    alpha += 0.2 * np.sin(2 * np.pi * 9.5 * t + np.random.uniform(0, 2*np.pi))
    alpha += 0.2 * np.sin(2 * np.pi * 10.5 * t + np.random.uniform(0, 2*np.pi))

    # Embed Schumann Resonance harmonics
    sr_signal = np.zeros(n_samples)
    sr_components = {}
    for name, freq in SR_CANONICAL.items():
        # Each harmonic has decreasing amplitude and slight frequency jitter
        amplitude = sr_strength * (0.8 ** list(SR_CANONICAL.keys()).index(name))
        # Add slow amplitude modulation
        modulation = 1 + 0.3 * np.sin(2 * np.pi * 0.1 * t + np.random.uniform(0, 2*np.pi))
        component = amplitude * modulation * np.sin(2 * np.pi * freq * t + np.random.uniform(0, 2*np.pi))
        sr_signal += component
        sr_components[name] = component

    # Combine
    eeg = pink_noise + alpha + sr_signal

    # Create DataFrame
    records = pd.DataFrame({
        'Timestamp': t,
        'EEG.F4': eeg,
        'EEG.O1': eeg + np.random.randn(n_samples) * 0.2,  # Slightly different
        'EEG.P7': eeg + np.random.randn(n_samples) * 0.3,
    })

    return records, sr_components


def compute_psd_with_harmonics(records, channel='EEG.F4', fs=128, nperseg_sec=8.0):
    """Compute PSD and identify SR harmonics."""
    x = records[channel].values
    nperseg = int(nperseg_sec * fs)
    f, Pxx = welch(x, fs=fs, nperseg=nperseg, noverlap=nperseg//2)

    # Convert to dB
    Pxx_db = 10 * np.log10(Pxx + 1e-12)

    # Find peaks near canonical frequencies
    detected_harmonics = {}
    for name, f_can in SR_CANONICAL.items():
        mask = (f >= f_can - 1.0) & (f <= f_can + 1.0)
        if np.any(mask):
            idx_peak = np.argmax(Pxx[mask])
            f_detected = f[mask][idx_peak]
            power_detected = Pxx_db[mask][idx_peak]
            detected_harmonics[name] = {
                'canonical': f_can,
                'detected': f_detected,
                'offset': f_detected - f_can,
                'power_db': power_detected
            }

    return f, Pxx, Pxx_db, detected_harmonics


def compute_time_frequency(records, channel='EEG.F4', fs=128,
                          freq_range=(1, 45), n_freqs=100):
    """Compute time-frequency representation using Morlet wavelets."""
    x = records[channel].values
    t = records['Timestamp'].values

    freqs = np.linspace(freq_range[0], freq_range[1], n_freqs)

    # Simple STFT-based approach for speed
    nperseg = int(2.0 * fs)  # 2-second windows
    noverlap = int(1.5 * fs)  # 75% overlap

    f_stft, t_stft, Sxx = signal.spectrogram(
        x, fs=fs, nperseg=nperseg, noverlap=noverlap,
        window='hann', mode='magnitude'
    )

    # Convert to dB
    Sxx_db = 10 * np.log10(Sxx + 1e-12)

    return f_stft, t_stft, Sxx_db


def compute_schumann_activity_index(records, channel='EEG.F4', fs=128,
                                    harmonics=None, bandwidth=1.0):
    """
    Compute Schumann Activity Index (SAI) over time.

    SAI = sum of z-scored power in SR harmonic bands.
    """
    if harmonics is None:
        harmonics = list(SR_CANONICAL.values())[:5]

    x = records[channel].values
    t = records['Timestamp'].values
    n_samples = len(x)

    # Sliding window parameters
    win_sec = 4.0
    step_sec = 0.5
    win_samples = int(win_sec * fs)
    step_samples = int(step_sec * fs)

    n_windows = (n_samples - win_samples) // step_samples + 1
    sai = np.zeros(n_windows)
    t_sai = np.zeros(n_windows)
    harmonic_powers = {f'h{i}': np.zeros(n_windows) for i in range(len(harmonics))}

    for i in range(n_windows):
        start = i * step_samples
        end = start + win_samples
        segment = x[start:end]
        t_sai[i] = t[start + win_samples // 2]

        # Compute PSD for this window
        f, Pxx = welch(segment, fs=fs, nperseg=min(win_samples, 256))

        # Extract power at each harmonic
        total_power = 0
        for j, h_freq in enumerate(harmonics):
            mask = (f >= h_freq - bandwidth/2) & (f <= h_freq + bandwidth/2)
            if np.any(mask):
                h_power = np.mean(Pxx[mask])
                harmonic_powers[f'h{j}'][i] = h_power
                total_power += h_power

        sai[i] = total_power

    # Z-score the SAI
    sai_z = (sai - np.mean(sai)) / (np.std(sai) + 1e-12)

    return t_sai, sai_z, harmonic_powers


def plot_sr_harmonics_comprehensive(records, channel='EEG.F4', fs=128,
                                    title="Schumann Resonance Harmonic Analysis",
                                    save_path=None):
    """
    Create comprehensive 6-panel visualization of SR harmonic results.
    """
    fig = plt.figure(figsize=(16, 12))

    # Compute all analyses
    f, Pxx, Pxx_db, detected = compute_psd_with_harmonics(records, channel, fs)
    f_tf, t_tf, Sxx_db = compute_time_frequency(records, channel, fs)
    t_sai, sai_z, h_powers = compute_schumann_activity_index(records, channel, fs)

    # Custom colormap for time-frequency
    colors = ['#000033', '#000066', '#0000CC', '#0066FF', '#00CCFF',
              '#66FFCC', '#CCFF66', '#FFCC00', '#FF6600', '#FF0000']
    cmap_tf = LinearSegmentedColormap.from_list('sr_tf', colors)

    # Panel 1: Power Spectral Density with harmonic markers
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.plot(f, Pxx_db, 'k-', linewidth=1, alpha=0.8)
    ax1.set_xlim(0, 50)
    ax1.set_xlabel('Frequency (Hz)', fontsize=11)
    ax1.set_ylabel('Power (dB)', fontsize=11)
    ax1.set_title('Power Spectral Density', fontsize=12, fontweight='bold')

    # Mark SR harmonics
    colors_h = plt.cm.rainbow(np.linspace(0, 1, len(SR_CANONICAL)))
    for (name, freq), color in zip(SR_CANONICAL.items(), colors_h):
        ax1.axvline(freq, color=color, linestyle='--', alpha=0.7, linewidth=1.5)
        if name in detected:
            det = detected[name]
            ax1.axvline(det['detected'], color=color, linestyle='-', alpha=0.9, linewidth=2)
            ax1.scatter([det['detected']], [det['power_db']], color=color, s=80, zorder=5,
                       edgecolor='black', linewidth=1)

    # Add EEG band shading
    for band_name, (lo, hi) in EEG_BANDS.items():
        ax1.axvspan(lo, hi, alpha=0.1, color='gray')

    ax1.grid(True, alpha=0.3)
    ax1.legend([f'{n}: {f:.2f} Hz' for n, f in SR_CANONICAL.items()],
               loc='upper right', fontsize=8, title='SR Harmonics')

    # Panel 2: Zoomed PSD around fundamental
    ax2 = fig.add_subplot(2, 3, 2)
    mask = (f >= 5) & (f <= 12)
    ax2.plot(f[mask], Pxx_db[mask], 'b-', linewidth=2)
    ax2.axvline(SR_CANONICAL['f0'], color='red', linestyle='--', linewidth=2,
                label=f"SR f₀ = {SR_CANONICAL['f0']:.2f} Hz")
    if 'f0' in detected:
        ax2.axvline(detected['f0']['detected'], color='green', linestyle='-', linewidth=2,
                   label=f"Detected = {detected['f0']['detected']:.2f} Hz")
    ax2.fill_between(f[mask], Pxx_db[mask], alpha=0.3)
    ax2.set_xlabel('Frequency (Hz)', fontsize=11)
    ax2.set_ylabel('Power (dB)', fontsize=11)
    ax2.set_title('Fundamental (f₀ ≈ 7.83 Hz)', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Detected vs Canonical comparison
    ax3 = fig.add_subplot(2, 3, 3)
    names = list(detected.keys())
    canonical = [detected[n]['canonical'] for n in names]
    detected_f = [detected[n]['detected'] for n in names]
    offsets = [detected[n]['offset'] for n in names]

    x_pos = np.arange(len(names))
    width = 0.35

    bars1 = ax3.bar(x_pos - width/2, canonical, width, label='Canonical', color='steelblue', alpha=0.8)
    bars2 = ax3.bar(x_pos + width/2, detected_f, width, label='Detected', color='coral', alpha=0.8)

    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([f'{n}\n({o:+.2f})' for n, o in zip(names, offsets)], fontsize=9)
    ax3.set_ylabel('Frequency (Hz)', fontsize=11)
    ax3.set_title('Detected vs Canonical Frequencies', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')

    # Panel 4: Time-Frequency Spectrogram
    ax4 = fig.add_subplot(2, 3, 4)
    extent = [t_tf[0], t_tf[-1], f_tf[0], f_tf[-1]]
    im = ax4.imshow(Sxx_db, aspect='auto', origin='lower', extent=extent,
                    cmap=cmap_tf, vmin=np.percentile(Sxx_db, 5),
                    vmax=np.percentile(Sxx_db, 98))

    # Overlay SR harmonic lines
    for name, freq in list(SR_CANONICAL.items())[:5]:
        ax4.axhline(freq, color='white', linestyle='--', alpha=0.7, linewidth=1)

    ax4.set_xlim(t_tf[0], t_tf[-1])
    ax4.set_ylim(1, 45)
    ax4.set_xlabel('Time (s)', fontsize=11)
    ax4.set_ylabel('Frequency (Hz)', fontsize=11)
    ax4.set_title('Time-Frequency Spectrogram', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax4, label='Power (dB)')

    # Panel 5: Schumann Activity Index
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.plot(t_sai, sai_z, 'b-', linewidth=1.5)
    ax5.fill_between(t_sai, 0, sai_z, where=sai_z > 0, alpha=0.3, color='green', label='Above mean')
    ax5.fill_between(t_sai, 0, sai_z, where=sai_z < 0, alpha=0.3, color='red', label='Below mean')
    ax5.axhline(0, color='gray', linestyle='-', linewidth=1)
    ax5.axhline(2, color='orange', linestyle='--', alpha=0.7, label='z = 2')
    ax5.axhline(-2, color='orange', linestyle='--', alpha=0.7)
    ax5.set_xlabel('Time (s)', fontsize=11)
    ax5.set_ylabel('SAI (z-score)', fontsize=11)
    ax5.set_title('Schumann Activity Index', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=8, loc='upper right')
    ax5.grid(True, alpha=0.3)

    # Panel 6: Harmonic power time series (stacked)
    ax6 = fig.add_subplot(2, 3, 6)
    colors_stack = plt.cm.viridis(np.linspace(0.2, 0.9, 5))

    # Normalize and stack
    h_names = [f'h{i}' for i in range(5)]
    h_data = np.array([h_powers[h] for h in h_names])
    h_data_norm = h_data / (np.max(h_data, axis=1, keepdims=True) + 1e-12)

    for i, (h_name, color) in enumerate(zip(h_names, colors_stack)):
        offset = i * 1.2
        ax6.fill_between(t_sai, offset, offset + h_data_norm[i], alpha=0.7, color=color)
        ax6.plot(t_sai, offset + h_data_norm[i], color='black', linewidth=0.5)

    ax6.set_yticks([i * 1.2 + 0.5 for i in range(5)])
    ax6.set_yticklabels([f"f{i} ({list(SR_CANONICAL.values())[i]:.1f} Hz)" for i in range(5)])
    ax6.set_xlabel('Time (s)', fontsize=11)
    ax6.set_title('Individual Harmonic Power', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='x')

    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()
    return fig, detected


def plot_harmonic_phase_coherence(records, channels=['EEG.F4', 'EEG.O1'], fs=128,
                                  save_path=None):
    """
    Visualize phase coherence between channels at SR harmonic frequencies.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    ch1, ch2 = channels[0], channels[1]
    x1 = records[ch1].values
    x2 = records[ch2].values

    # Compute coherence
    f_coh, Cxy = coherence(x1, x2, fs=fs, nperseg=int(4*fs))

    # Panel 1: Full coherence spectrum
    ax = axes[0, 0]
    ax.plot(f_coh, Cxy, 'b-', linewidth=2)
    for name, freq in list(SR_CANONICAL.items())[:5]:
        ax.axvline(freq, color='red', linestyle='--', alpha=0.5)
    ax.set_xlim(0, 50)
    ax.set_xlabel('Frequency (Hz)', fontsize=11)
    ax.set_ylabel('Coherence', fontsize=11)
    ax.set_title(f'Coherence: {ch1} vs {ch2}', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Panels 2-6: Zoomed coherence around each harmonic
    for i, (name, freq) in enumerate(list(SR_CANONICAL.items())[:5]):
        ax = axes[(i+1)//3, (i+1)%3]
        mask = (f_coh >= freq - 2) & (f_coh <= freq + 2)
        ax.plot(f_coh[mask], Cxy[mask], 'b-', linewidth=2)
        ax.axvline(freq, color='red', linestyle='--', linewidth=2, label=f'Canonical: {freq:.2f} Hz')
        ax.fill_between(f_coh[mask], 0, Cxy[mask], alpha=0.3)

        # Find peak coherence in this band
        if np.any(mask):
            peak_idx = np.argmax(Cxy[mask])
            peak_f = f_coh[mask][peak_idx]
            peak_c = Cxy[mask][peak_idx]
            ax.scatter([peak_f], [peak_c], color='green', s=100, zorder=5,
                      label=f'Peak: {peak_f:.2f} Hz, C={peak_c:.2f}')

        ax.set_xlabel('Frequency (Hz)', fontsize=10)
        ax.set_ylabel('Coherence', fontsize=10)
        ax.set_title(f'{name}: {freq:.2f} Hz', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

    plt.suptitle('Phase Coherence at Schumann Resonance Frequencies', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()
    return fig


def plot_sr_summary_table(detected_harmonics, save_path=None):
    """
    Create a summary table visualization of detected harmonics.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')

    # Prepare table data
    col_labels = ['Harmonic', 'Canonical (Hz)', 'Detected (Hz)', 'Offset (Hz)', 'Power (dB)']
    cell_data = []

    for name, data in detected_harmonics.items():
        cell_data.append([
            name,
            f"{data['canonical']:.2f}",
            f"{data['detected']:.2f}",
            f"{data['offset']:+.3f}",
            f"{data['power_db']:.1f}"
        ])

    # Color cells based on offset magnitude
    cell_colors = []
    for row in cell_data:
        offset = float(row[3])
        if abs(offset) < 0.1:
            color = '#90EE90'  # Light green - excellent match
        elif abs(offset) < 0.3:
            color = '#FFFFE0'  # Light yellow - good match
        else:
            color = '#FFB6C1'  # Light pink - notable offset
        cell_colors.append(['white', 'white', 'white', color, 'white'])

    table = ax.table(cellText=cell_data, colLabels=col_labels, cellColours=cell_colors,
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    # Style header
    for j in range(len(col_labels)):
        table[(0, j)].set_facecolor('#4472C4')
        table[(0, j)].set_text_props(color='white', fontweight='bold')

    ax.set_title('Schumann Resonance Harmonic Detection Summary', fontsize=14,
                 fontweight='bold', pad=20)

    # Add legend
    legend_text = "Offset colors:  ■ <0.1 Hz (excellent)  ■ <0.3 Hz (good)  ■ ≥0.3 Hz (notable)"
    fig.text(0.5, 0.05, legend_text, ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()
    return fig


def run_sr_visualization_demo(save_dir='sr_figures'):
    """
    Run complete SR harmonics visualization demo.
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 60)
    print("SCHUMANN RESONANCE HARMONICS VISUALIZATION")
    print("=" * 60)

    # Generate synthetic data
    print("\n[1/4] Generating synthetic EEG with embedded SR harmonics...")
    records, sr_components = generate_synthetic_eeg_with_sr(
        duration_sec=120, fs=128, sr_strength=0.4, seed=42
    )
    print(f"      Generated {len(records)} samples ({len(records)/128:.1f} seconds)")

    # Main comprehensive visualization
    print("\n[2/4] Creating comprehensive SR harmonic analysis...")
    fig1, detected = plot_sr_harmonics_comprehensive(
        records, channel='EEG.F4', fs=128,
        title="Schumann Resonance Harmonic Analysis",
        save_path=f'{save_dir}/sr_harmonics_comprehensive.png'
    )

    # Phase coherence visualization
    print("\n[3/4] Creating phase coherence visualization...")
    fig2 = plot_harmonic_phase_coherence(
        records, channels=['EEG.F4', 'EEG.O1'], fs=128,
        save_path=f'{save_dir}/sr_phase_coherence.png'
    )

    # Summary table
    print("\n[4/4] Creating summary table...")
    fig3 = plot_sr_summary_table(
        detected,
        save_path=f'{save_dir}/sr_summary_table.png'
    )

    # Print summary
    print("\n" + "=" * 60)
    print("DETECTION SUMMARY")
    print("=" * 60)
    print(f"\n{'Harmonic':<10} {'Canonical':>12} {'Detected':>12} {'Offset':>10} {'Power':>10}")
    print("-" * 55)
    for name, data in detected.items():
        print(f"{name:<10} {data['canonical']:>12.2f} {data['detected']:>12.2f} "
              f"{data['offset']:>+10.3f} {data['power_db']:>10.1f} dB")

    print(f"\nFigures saved to {save_dir}/")
    print("✓ Visualization complete!")

    return records, detected


if __name__ == "__main__":
    records, detected = run_sr_visualization_demo(save_dir='sr_figures')
