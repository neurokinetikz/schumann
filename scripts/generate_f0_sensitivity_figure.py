#!/usr/bin/env python3
"""
Generate publication-quality f₀ sensitivity figure for the paper.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

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

def ranking_metric(enrichments):
    """Returns score based on whether ranking matches theory."""
    expected_order = ['boundary', 'noble_2', 'attractor', 'noble_1']
    values = [enrichments[p] for p in expected_order]
    
    n_pairs = 0
    n_correct = 0
    for i in range(len(expected_order)):
        for j in range(i+1, len(expected_order)):
            n_pairs += 1
            if values[i] < values[j]:
                n_correct += 1
    
    rank_score = n_correct / n_pairs * 100
    magnitude = (enrichments['noble_1'] - enrichments['boundary'])
    return rank_score + magnitude

def main():
    # Load data
    peaks_df = pd.read_csv('golden_ratio_peaks_ALL.csv')
    freqs = peaks_df['freq'].values
    freqs = freqs[(freqs >= 4) & (freqs <= 50)]
    
    # Sweep f0
    f0_values = np.arange(6.5, 8.5 + 0.025, 0.025)
    
    metrics = {
        'ranking': [],
        'noble_1': [],
        'attractor': [],
        'boundary': [],
    }
    
    for f0 in f0_values:
        enrichments = compute_all_enrichments(freqs, f0)
        metrics['ranking'].append(ranking_metric(enrichments))
        metrics['noble_1'].append(enrichments['noble_1'])
        metrics['attractor'].append(enrichments['attractor'])
        metrics['boundary'].append(enrichments['boundary'])
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), height_ratios=[2, 1])
    
    # Top panel: Ranking metric
    ax1 = axes[0]
    ax1.plot(f0_values, metrics['ranking'], 'b-', linewidth=2.5, label='Ranking + Magnitude')
    
    # Find optimal
    opt_idx = np.argmax(metrics['ranking'])
    opt_f0 = f0_values[opt_idx]
    opt_val = metrics['ranking'][opt_idx]
    
    # Mark optimal
    ax1.axvline(opt_f0, color='green', linestyle='--', linewidth=2, alpha=0.8)
    ax1.plot(opt_f0, opt_val, 'go', markersize=12, zorder=5)
    ax1.annotate(f'Optimal: {opt_f0:.2f} Hz', xy=(opt_f0, opt_val), 
                xytext=(opt_f0 + 0.3, opt_val - 15),
                fontsize=11, fontweight='bold', color='green',
                arrowprops=dict(arrowstyle='->', color='green', lw=1.5))
    
    # Mark geophysical values
    ax1.axvline(7.6, color='purple', linestyle=':', linewidth=2, alpha=0.7)
    ax1.axvline(7.83, color='orange', linestyle=':', linewidth=2, alpha=0.7)
    
    # Shade plateau
    threshold = 0.7 * opt_val
    plateau_mask = np.array(metrics['ranking']) >= threshold
    plateau_idx = np.where(plateau_mask)[0]
    if len(plateau_idx) > 0:
        p_start, p_end = f0_values[plateau_idx[0]], f0_values[plateau_idx[-1]]
        ax1.axvspan(p_start, p_end, alpha=0.15, color='blue', 
                   label=f'Plateau (>70%): {p_start:.2f}--{p_end:.2f} Hz')
    
    ax1.set_ylabel('Alignment Metric\n(Ranking + Magnitude)', fontsize=12)
    ax1.set_title('$f_0$ Sensitivity Analysis: Comprehensive Metric', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(6.5, 8.5)
    
    # Custom legend
    legend_elements = [
        Line2D([0], [0], color='blue', linewidth=2.5, label='Ranking + Magnitude metric'),
        Line2D([0], [0], color='green', linestyle='--', linewidth=2, label=f'Optimal: {opt_f0:.2f} Hz'),
        Line2D([0], [0], color='purple', linestyle=':', linewidth=2, label='Geophysical (7.6 Hz)'),
        Line2D([0], [0], color='orange', linestyle=':', linewidth=2, label='Canonical (7.83 Hz)'),
    ]
    ax1.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    # Bottom panel: Individual enrichments
    ax2 = axes[1]
    ax2.plot(f0_values, metrics['noble_1'], 'purple', linewidth=2, label='1° Noble')
    ax2.plot(f0_values, metrics['attractor'], 'green', linewidth=2, label='Attractor')
    ax2.plot(f0_values, metrics['boundary'], 'red', linewidth=2, label='Boundary')
    
    ax2.axhline(0, color='gray', linestyle='-', alpha=0.5)
    ax2.axvline(7.6, color='purple', linestyle=':', linewidth=2, alpha=0.7)
    ax2.axvline(7.83, color='orange', linestyle=':', linewidth=2, alpha=0.7)
    
    # Mark the collapse of 1° Noble at 8.05 Hz
    idx_805 = np.argmin(np.abs(f0_values - 8.05))
    ax2.annotate('1° Noble collapses\nat higher $f_0$', 
                xy=(8.05, metrics['noble_1'][idx_805]),
                xytext=(8.2, 25),
                fontsize=9, color='purple',
                arrowprops=dict(arrowstyle='->', color='purple', lw=1))
    
    ax2.set_xlabel('Base Frequency $f_0$ (Hz)', fontsize=12)
    ax2.set_ylabel('Enrichment (%)', fontsize=12)
    ax2.set_title('Position-Type Enrichment vs $f_0$', fontsize=12, fontweight='bold')
    ax2.legend(loc='center right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(6.5, 8.5)
    ax2.set_ylim(-70, 50)
    
    plt.tight_layout()
    plt.savefig('phi_f0_sensitivity.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: phi_f0_sensitivity.png")

if __name__ == '__main__':
    main()
