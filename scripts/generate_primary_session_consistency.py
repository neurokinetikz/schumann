#!/usr/bin/env python3
"""Generate session-level consistency chart for the primary dataset.

Matches the style of Figure 13 (EEGEmotions-27 session consistency).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from typing import Dict

# Constants
PHI = 1.618033988749895
F0_DEFAULT = 7.6

# Position types for lattice coordinate analysis
POSITION_TYPES = {
    'boundary': {'center': 0.0, 'ranges': [(0.0, 0.1), (0.9, 1.0)]},
    'noble_2': {'center': 0.382, 'ranges': [(0.332, 0.432)]},
    'attractor': {'center': 0.5, 'ranges': [(0.45, 0.55)]},
    'noble_1': {'center': 0.618, 'ranges': [(0.568, 0.668)]},
}

# Paths
INPUT_CSV = 'csv/golden_ratio_peaks_ALL.csv'
OUTPUT_PNG = 'papers/images/phi_session_consistency.png'


def compute_lattice_coordinate(freqs: np.ndarray, f0: float = F0_DEFAULT) -> np.ndarray:
    """Compute phi-lattice coordinate u = [log_phi(f/f0)] mod 1."""
    return (np.log(freqs / f0) / np.log(PHI)) % 1


def session_consistency_analysis(peaks_df: pd.DataFrame, f0: float = F0_DEFAULT,
                                  min_peaks: int = 50) -> Dict:
    """For each session: compute position enrichment and test ordering."""
    sessions = peaks_df['session'].unique()
    session_results = []

    for session in sessions:
        session_peaks = peaks_df[peaks_df['session'] == session]

        if len(session_peaks) < min_peaks:
            continue

        freqs = session_peaks['freq'].values
        u = compute_lattice_coordinate(freqs, f0)

        enrichments = {}
        for pos_name, config in POSITION_TYPES.items():
            in_pos = np.zeros(len(u), dtype=bool)
            width = 0.0
            for u_low, u_high in config['ranges']:
                in_pos |= (u >= u_low) & (u < u_high)
                width += (u_high - u_low)
            observed_frac = in_pos.sum() / len(u) if len(u) > 0 else 0
            enrichments[pos_name] = ((observed_frac / width) - 1) * 100 if width > 0 else 0

        correct = (enrichments['noble_1'] > enrichments['attractor'] >
                   enrichments['noble_2'] > enrichments['boundary'])
        attractor_gt_boundary = enrichments['attractor'] > enrichments['boundary']

        session_results.append({
            'session': session,
            'n_peaks': len(session_peaks),
            'correct_ordering': correct,
            'attractor_gt_boundary': attractor_gt_boundary,
            **{f'{k}_enrich': v for k, v in enrichments.items()}
        })

    df = pd.DataFrame(session_results)

    n_sessions = len(df)
    pct_correct = 100 * df['correct_ordering'].sum() / n_sessions if n_sessions > 0 else 0
    pct_attractor_gt = 100 * df['attractor_gt_boundary'].sum() / n_sessions if n_sessions > 0 else 0

    diffs = df['attractor_enrich'] - df['boundary_enrich']
    cohens_d = diffs.mean() / diffs.std() if diffs.std() > 0 else 0

    return {
        'session_df': df,
        'n_sessions': n_sessions,
        'pct_correct_ordering': pct_correct,
        'pct_attractor_gt_boundary': pct_attractor_gt,
        'cohens_d': cohens_d,
    }


def plot_session_consistency(session_results: Dict, output_path: str):
    """Create 2-panel session-level consistency figure matching Figure 13 style."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    df = session_results['session_df']

    # Panel A: histogram of attractor - boundary difference
    diffs = df['attractor_enrich'] - df['boundary_enrich']
    ax1.hist(diffs, bins=40, color='steelblue', alpha=0.7, edgecolor='white')
    ax1.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero (no effect)')
    ax1.axvline(diffs.mean(), color='green', linestyle='-', linewidth=2,
                label=f'Mean: {diffs.mean():.1f}%')
    ax1.set_xlabel('Attractor - Boundary Enrichment (%)', fontsize=11)
    ax1.set_ylabel('Number of Sessions', fontsize=11)
    ax1.set_title(f"Session-Level Alignment Distribution\n{session_results['pct_attractor_gt_boundary']:.1f}% show attractor > boundary",
                  fontsize=12, fontweight='bold')
    ax1.legend()

    # Panel B: scatter of n_peaks vs alignment
    colors = ['green' if c else 'red' for c in df['attractor_gt_boundary']]
    ax2.scatter(df['n_peaks'], diffs, c=colors, alpha=0.5, s=20)
    ax2.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax2.set_xlabel('Peaks per Session', fontsize=11)
    ax2.set_ylabel('Attractor - Boundary (%)', fontsize=11)
    ax2.set_title(f"Alignment vs Session Size\nCohen's d = {session_results['cohens_d']:.2f}",
                  fontsize=12, fontweight='bold')

    legend_elements = [Patch(facecolor='green', alpha=0.5, label='Attractor > Boundary'),
                       Patch(facecolor='red', alpha=0.5, label='Attractor ≤ Boundary')]
    ax2.legend(handles=legend_elements, loc='upper right')

    plt.suptitle(f"PRIMARY: Session-Level Consistency (N = {session_results['n_sessions']:,} sessions)",
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    print("Loading primary dataset peaks...")
    peaks_df = pd.read_csv(INPUT_CSV)
    print(f"  Loaded {len(peaks_df):,} peaks from {peaks_df['session'].nunique()} sessions")

    print("\nAnalyzing session-level consistency...")
    session_results = session_consistency_analysis(peaks_df)
    print(f"  Sessions analyzed: {session_results['n_sessions']:,}")
    print(f"  Correct ordering: {session_results['pct_correct_ordering']:.1f}%")
    print(f"  Attractor > Boundary: {session_results['pct_attractor_gt_boundary']:.1f}%")
    print(f"  Cohen's d = {session_results['cohens_d']:.2f}")

    print("\nGenerating figure...")
    plot_session_consistency(session_results, OUTPUT_PNG)

    print("\nDone!")


if __name__ == '__main__':
    main()
