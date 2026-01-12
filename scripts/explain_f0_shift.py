#!/usr/bin/env python3
"""
Visualize how changing f₀ affects which frequencies land at special positions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PHI = (1 + np.sqrt(5)) / 2

def get_special_frequencies(f0, n_range=(-2, 4)):
    """Get the actual frequencies that land at special positions for a given f₀."""
    positions = {}
    
    for n in range(n_range[0], n_range[1] + 1):
        # Boundary: integer n
        positions[f'boundary_n{n}'] = f0 * (PHI ** n)
        # Attractor: n + 0.5
        positions[f'attractor_n{n}'] = f0 * (PHI ** (n + 0.5))
        # 1° Noble: n + 0.618
        positions[f'noble1_n{n}'] = f0 * (PHI ** (n + 0.618))
    
    return positions

def main():
    # Compare special frequencies at different f₀
    f0_values = [7.6, 7.83, 8.05]
    
    print("=" * 80)
    print("SPECIAL FREQUENCIES AT DIFFERENT f₀ VALUES")
    print("=" * 80)
    
    # Focus on the EEG-relevant range (4-50 Hz)
    for f0 in f0_values:
        print(f"\n{'='*40}")
        print(f"f₀ = {f0} Hz")
        print(f"{'='*40}")
        
        freqs = get_special_frequencies(f0)
        
        # Group by position type
        for pos_type in ['boundary', 'attractor', 'noble1']:
            relevant = [(k, v) for k, v in freqs.items() 
                       if k.startswith(pos_type) and 4 < v < 50]
            print(f"\n{pos_type.upper()} positions in 4-50 Hz:")
            for name, freq in sorted(relevant, key=lambda x: x[1]):
                n = name.split('_n')[1]
                print(f"  n={n}: {freq:.2f} Hz")
    
    # Key comparison: Where do the major EEG bands land?
    print("\n" + "=" * 80)
    print("KEY INSIGHT: ALPHA BAND (8-12 Hz) PLACEMENT")
    print("=" * 80)
    
    alpha_center = 10.0  # Approximate alpha peak
    
    for f0 in f0_values:
        n = np.log(alpha_center / f0) / np.log(PHI)
        u = n - np.floor(n)
        
        # Determine position type
        if abs(u) < 0.1 or abs(u - 1) < 0.1:
            pos_type = "BOUNDARY"
        elif abs(u - 0.5) < 0.1:
            pos_type = "ATTRACTOR"
        elif abs(u - 0.618) < 0.1:
            pos_type = "1° NOBLE"
        elif abs(u - 0.382) < 0.1:
            pos_type = "2° NOBLE"
        else:
            pos_type = "between positions"
        
        print(f"\nf₀ = {f0} Hz:")
        print(f"  Alpha (10 Hz) → n = {n:.3f}, u = {u:.3f}")
        print(f"  Position type: {pos_type}")
    
    # Create visualization
    print("\n" + "=" * 80)
    print("CREATING VISUALIZATION")
    print("=" * 80)
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    # Load actual peak data
    peaks_df = pd.read_csv('golden_ratio_peaks_ALL_EMOTIV.csv')
    freqs = peaks_df['freq'].values
    freqs = freqs[(freqs >= 4) & (freqs <= 50)]
    
    for ax, f0 in zip(axes, f0_values):
        # Compute lattice coordinates
        n = np.log(freqs / f0) / np.log(PHI)
        u = n - np.floor(n)
        
        # Histogram
        counts, bins, patches = ax.hist(u, bins=50, range=(0, 1), 
                                         color='steelblue', alpha=0.7, 
                                         edgecolor='white')
        
        # Mark special positions
        positions = {'Boundary': 0.0, '2° Noble': 0.382, 
                    'Attractor': 0.5, '1° Noble': 0.618}
        colors = {'Boundary': 'red', '2° Noble': 'orange', 
                 'Attractor': 'green', '1° Noble': 'purple'}
        
        for name, pos in positions.items():
            ax.axvline(pos, color=colors[name], linestyle='--', 
                      linewidth=2, alpha=0.8, label=name)
            # Shade window
            ax.axvspan(pos - 0.05, pos + 0.05, alpha=0.15, color=colors[name])
        
        # Also mark boundary at 1.0 (wraparound)
        ax.axvline(1.0, color='red', linestyle='--', linewidth=2, alpha=0.8)
        ax.axvspan(0.95, 1.0, alpha=0.15, color='red')
        
        ax.set_ylabel('Peak Count')
        ax.set_title(f'f₀ = {f0} Hz', fontsize=12, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.3)
        
        if f0 == 7.6:
            ax.legend(loc='upper right', fontsize=8)
    
    axes[-1].set_xlabel('Lattice Coordinate u = [log_φ(f/f₀)] mod 1')
    
    plt.tight_layout()
    plt.savefig('f0_shift_visualization.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: f0_shift_visualization.png")
    
    # Quantitative breakdown
    print("\n" + "=" * 80)
    print("WHY 1° NOBLE COLLAPSES AT f₀ = 8.05 Hz")
    print("=" * 80)
    
    # Show which actual frequencies fall into the 1° noble window at each f0
    for f0 in f0_values:
        n = np.log(freqs / f0) / np.log(PHI)
        u = n - np.floor(n)
        
        # 1° Noble window: 0.568 to 0.668
        in_noble = (u >= 0.568) & (u <= 0.668)
        noble_freqs = freqs[in_noble]
        
        print(f"\nf₀ = {f0} Hz:")
        print(f"  Peaks in 1° Noble window: {len(noble_freqs):,}")
        print(f"  Frequency range landing in Noble: {noble_freqs.min():.1f} - {noble_freqs.max():.1f} Hz")
        
        # What bands are these?
        bands = {'Delta': (1, 4), 'Theta': (4, 8), 'Alpha': (8, 12), 
                'Beta': (12, 30), 'Gamma': (30, 50)}
        print(f"  Band breakdown:")
        for band, (lo, hi) in bands.items():
            count = ((noble_freqs >= lo) & (noble_freqs < hi)).sum()
            if count > 0:
                print(f"    {band}: {count:,} peaks ({100*count/len(noble_freqs):.1f}%)")

if __name__ == '__main__':
    main()
