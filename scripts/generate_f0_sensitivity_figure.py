#!/usr/bin/env python3
"""
Generate publication-quality f₀ sensitivity figure for the paper.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PHI = (1 + np.sqrt(5)) / 2

def compute_lattice_coordinate(freq, f0):
    n = np.log(freq / f0) / np.log(PHI)
    return n - np.floor(n)

def compute_all_enrichments(freqs, f0):
    u = compute_lattice_coordinate(freqs, f0)
    u_window = 0.05
    
    results = {}
    positions = [('boundary', 0.0), ('noble_2', 0.382), 
                 ('attractor', 0.5), ('noble_1', 0.618)]
    
    for pos_name, pos_center in positions:
        in_window = np.abs(u - pos_center) < u_window
        if pos_name == 'boundary':
            in_window |= np.abs(u - 1.0) < u_window
            expected_frac = 4 * u_window
        else:
            expected_frac = 2 * u_window
        
        observed_frac = in_window.sum() / len(u)
        enrichment = (observed_frac / expected_frac - 1) * 100
        results[pos_name] = enrichment
    
    return results

def alignment_score(enrichments):
    """Returns weighted alignment score matching Emotions dataset style."""
    return (
        -enrichments['boundary'] +
        0.5 * enrichments['noble_2'] +
        enrichments['attractor'] +
        1.5 * enrichments['noble_1']
    )

def main():
    # Load data
    peaks_df = pd.read_csv('papers/golden_ratio_peaks_ALL.csv')
    freqs = peaks_df['freq'].values
    freqs = freqs[(freqs >= 4) & (freqs <= 50)]
    
    # Sweep f0 (matching Emotions dataset range)
    f0_values = np.arange(6.0, 9.0 + 0.1, 0.1)

    metrics = {
        'alignment': [],
        'noble_1': [],
        'attractor': [],
        'noble_2': [],
        'boundary': [],
    }

    for f0 in f0_values:
        enrichments = compute_all_enrichments(freqs, f0)
        metrics['alignment'].append(alignment_score(enrichments))
        metrics['noble_1'].append(enrichments['noble_1'])
        metrics['attractor'].append(enrichments['attractor'])
        metrics['noble_2'].append(enrichments['noble_2'])
        metrics['boundary'].append(enrichments['boundary'])
    
    # Create figure (matching Emotions dataset style)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Top panel: Alignment score
    ax1.plot(f0_values, metrics['alignment'], 'b-', linewidth=2, marker='o', markersize=4)

    # Find optimal and max alignment
    opt_idx = np.argmax(metrics['alignment'])
    opt_f0 = f0_values[opt_idx]
    max_alignment = metrics['alignment'][opt_idx]

    # Add 95% threshold line
    ax1.axhline(max_alignment * 0.95, color='red', linestyle='--',
                linewidth=1.5, label='95% threshold')

    # Shade 95% plateau (green, matching Emotions style)
    threshold = 0.95 * max_alignment
    plateau_mask = np.array(metrics['alignment']) >= threshold
    plateau_idx = np.where(plateau_mask)[0]
    if len(plateau_idx) > 0:
        p_start, p_end = f0_values[plateau_idx[0]], f0_values[plateau_idx[-1]]
        ax1.axvspan(p_start, p_end, alpha=0.2, color='green', label='Plateau')
        ax1.axvline(7.6, color='orange', linestyle=':', linewidth=2, label='f₀=7.6 Hz')

    ax1.set_ylabel('Alignment Score', fontsize=12)
    ax1.set_title('f₀ Sensitivity Analysis', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Bottom panel: Position enrichments (4 types, matching Emotions colors)
    ax2.plot(f0_values, metrics['boundary'], 'r-', linewidth=2, label='Boundary')
    ax2.plot(f0_values, metrics['noble_2'], 'purple', linewidth=2, label='2° Noble')
    ax2.plot(f0_values, metrics['attractor'], 'orange', linewidth=2, label='Attractor')
    ax2.plot(f0_values, metrics['noble_1'], 'g-', linewidth=2, label='1° Noble')

    ax2.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax2.set_xlabel('Base Frequency f₀ (Hz)', fontsize=12)
    ax2.set_ylabel('Enrichment (%)', fontsize=12)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('papers/images/phi_f0_sensitivity.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: papers/images/phi_f0_sensitivity.png")

if __name__ == '__main__':
    main()
