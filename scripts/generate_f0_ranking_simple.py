#!/usr/bin/env python3
"""
Generate a simple, intuitive f₀ ranking validation figure.

Shows horizontal bar charts at 3 key f₀ values to make the ranking
concept immediately obvious.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PHI = (1 + np.sqrt(5)) / 2

def compute_lattice_coordinate(freq, f0):
    n = np.log(freq / f0) / np.log(PHI)
    return n - np.floor(n)

def compute_position_enrichments(freqs, f0, u_window=0.05):
    u = compute_lattice_coordinate(freqs, f0)

    results = {}
    positions = [
        ('Boundary', 0.0),
        ('2° Noble', 0.382),
        ('Attractor', 0.5),
        ('1° Noble', 0.618)
    ]

    for pos_name, pos_center in positions:
        in_window = np.abs(u - pos_center) < u_window
        if pos_name == 'Boundary':
            in_window |= np.abs(u - 1.0) < u_window
            expected_frac = 4 * u_window
        else:
            expected_frac = 2 * u_window

        observed_frac = in_window.sum() / len(u)
        enrichment = (observed_frac / expected_frac - 1) * 100
        results[pos_name] = enrichment

    return results

def check_ranking(e):
    """Check if ranking is satisfied: Boundary < 2° Noble < Attractor < 1° Noble"""
    return (e['Boundary'] < e['2° Noble'] < e['Attractor'] < e['1° Noble'])

def main():
    # Load data
    peaks_df = pd.read_csv('golden_ratio_peaks_ALL_EMOTIV.csv')
    freqs = peaks_df['freq'].values
    freqs = freqs[(freqs >= 4) & (freqs <= 50)]

    # Three key f₀ values to compare
    f0_values = [7.6, 7.83, 8.05]
    labels = ['7.6 Hz\n(Geophysical)', '7.83 Hz\n(Canonical)', '8.05 Hz\n(Higher)']

    # Compute enrichments at each
    all_enrichments = []
    for f0 in f0_values:
        e = compute_position_enrichments(freqs, f0)
        all_enrichments.append(e)

    # Create figure - 3 panels side by side
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)

    position_order = ['Boundary', '2° Noble', 'Attractor', '1° Noble']
    colors = ['#d62728', '#9467bd', '#2ca02c', '#1f77b4']  # red, purple, green, blue

    for idx, (ax, e, f0, label) in enumerate(zip(axes, all_enrichments, f0_values, labels)):
        values = [e[p] for p in position_order]
        y_pos = np.arange(len(position_order))

        # Create horizontal bars
        bars = ax.barh(y_pos, values, color=colors, alpha=0.8, height=0.6)

        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, values)):
            if val >= 0:
                ax.text(val + 1, i, f'+{val:.0f}%', va='center', ha='left', fontsize=11, fontweight='bold')
            else:
                ax.text(val - 1, i, f'{val:.0f}%', va='center', ha='right', fontsize=11, fontweight='bold')

        # Add vertical line at 0
        ax.axvline(0, color='gray', linestyle='-', linewidth=1)

        # Check if ranking is satisfied
        ranking_ok = check_ranking(e)

        # Title with pass/fail indicator
        if ranking_ok:
            title_color = 'green'
            status = '✓ Ranking Satisfied'
            # Add green border
            for spine in ax.spines.values():
                spine.set_edgecolor('green')
                spine.set_linewidth(3)
        else:
            title_color = 'red'
            status = '✗ Ranking Violated'
            # Add red border
            for spine in ax.spines.values():
                spine.set_edgecolor('red')
                spine.set_linewidth(3)

        ax.set_title(f'$f_0$ = {label}\n{status}', fontsize=13, fontweight='bold', color=title_color)

        ax.set_yticks(y_pos)
        if idx == 0:
            ax.set_yticklabels(position_order, fontsize=11)
        else:
            ax.set_yticklabels([])

        ax.set_xlabel('Enrichment (%)', fontsize=11)
        ax.set_xlim(-80, 60)
        ax.grid(True, axis='x', alpha=0.3)

        # Add arrows showing the expected ordering
        if idx == 0:
            # Add "Theory predicts" annotation
            ax.annotate('', xy=(0.02, 0.95), xytext=(0.02, 0.05),
                       xycoords='axes fraction',
                       arrowprops=dict(arrowstyle='->', color='black', lw=2))
            ax.text(-0.15, 0.5, 'Theory\npredicts\nthis order\n↑', transform=ax.transAxes,
                   fontsize=9, ha='center', va='center', style='italic')

    # Add overall title
    fig.suptitle('$f_0$ Validation: Does the Theoretical Ranking Hold?\n'
                 'Expected order: Boundary < 2° Noble < Attractor < 1° Noble',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig('phi_f0_ranking_simple.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: phi_f0_ranking_simple.png")

    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for f0, e, label in zip(f0_values, all_enrichments, labels):
        ranking_ok = check_ranking(e)
        print(f"\nAt f₀ = {f0} Hz ({label.split(chr(10))[1].strip('()')}:")
        print(f"  Boundary:  {e['Boundary']:+.1f}%")
        print(f"  2° Noble:  {e['2° Noble']:+.1f}%")
        print(f"  Attractor: {e['Attractor']:+.1f}%")
        print(f"  1° Noble:  {e['1° Noble']:+.1f}%")
        print(f"  Ranking: {'✓ SATISFIED' if ranking_ok else '✗ VIOLATED'}")

        if not ranking_ok:
            # Explain why it fails
            if e['1° Noble'] < e['Attractor']:
                print(f"  → FAILS because 1° Noble ({e['1° Noble']:+.1f}%) < Attractor ({e['Attractor']:+.1f}%)")

if __name__ == '__main__':
    main()
