#!/usr/bin/env python3
"""
Per-Subject φⁿ Ratio Analysis

Computes per-subject harmonic ratio precision from SIE-PAPER-FINAL.csv
to demonstrate that φⁿ relationships hold within individual subjects,
not just in pooled data.

Output:
- per_subject_phi_ratios.csv: Per-subject statistics
- per_subject_phi_ratios.png: Histogram of per-subject mean ratio errors
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Constants
PHI = (1 + np.sqrt(5)) / 2  # 1.618034
PHI_SQUARED = PHI ** 2       # 2.618034
PHI_CUBED = PHI ** 3         # 4.236068

# File paths
INPUT_FILE = 'papers/SIE-PAPER-FINAL.csv'
OUTPUT_CSV = 'per_subject_phi_ratios.csv'
OUTPUT_FIG = 'papers/images/per_subject_phi_ratios.png'

def main():
    print("=" * 70)
    print("Per-Subject φⁿ Ratio Analysis")
    print("=" * 70)

    # Load data
    print(f"\nLoading {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df)} events from {df['subject'].nunique()} subjects")

    # Filter out rows with missing ratios
    df_valid = df.dropna(subset=['sr3/sr1', 'sr5/sr1', 'sr5/sr3'])
    print(f"Valid events with all ratios: {len(df_valid)}")

    # Group by subject and compute statistics
    print("\nComputing per-subject statistics...")

    per_subject = df_valid.groupby('subject').agg({
        'sr3/sr1': ['mean', 'std', 'count'],
        'sr5/sr1': ['mean', 'std'],
        'sr5/sr3': ['mean', 'std'],
        'dataset': 'first',
        'device': 'first',
        'context': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0]
    }).reset_index()

    # Flatten column names
    per_subject.columns = [
        'subject',
        'mean_sr3_sr1', 'std_sr3_sr1', 'n_events',
        'mean_sr5_sr1', 'std_sr5_sr1',
        'mean_sr5_sr3', 'std_sr5_sr3',
        'dataset', 'device', 'context'
    ]

    # Compute percentage errors vs theoretical φⁿ values
    per_subject['error_sr3_sr1_pct'] = np.abs(per_subject['mean_sr3_sr1'] - PHI_SQUARED) / PHI_SQUARED * 100
    per_subject['error_sr5_sr1_pct'] = np.abs(per_subject['mean_sr5_sr1'] - PHI_CUBED) / PHI_CUBED * 100
    per_subject['error_sr5_sr3_pct'] = np.abs(per_subject['mean_sr5_sr3'] - PHI) / PHI * 100

    # Mean of the three ratio errors
    per_subject['mean_ratio_error_pct'] = (
        per_subject['error_sr3_sr1_pct'] +
        per_subject['error_sr5_sr1_pct'] +
        per_subject['error_sr5_sr3_pct']
    ) / 3

    # Sort by event count (descending)
    per_subject = per_subject.sort_values('n_events', ascending=False)

    # Print summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    print(f"Total subjects analyzed: {len(per_subject)}")
    print(f"Total events analyzed: {per_subject['n_events'].sum()}")
    print(f"\nTheoretical φⁿ values:")
    print(f"  φ² (SR3/SR1 expected): {PHI_SQUARED:.4f}")
    print(f"  φ³ (SR5/SR1 expected): {PHI_CUBED:.4f}")
    print(f"  φ  (SR5/SR3 expected): {PHI:.4f}")

    print(f"\nPer-subject mean ratio error:")
    print(f"  Mean:   {per_subject['mean_ratio_error_pct'].mean():.2f}%")
    print(f"  Median: {per_subject['mean_ratio_error_pct'].median():.2f}%")
    print(f"  Std:    {per_subject['mean_ratio_error_pct'].std():.2f}%")
    print(f"  Min:    {per_subject['mean_ratio_error_pct'].min():.2f}%")
    print(f"  Max:    {per_subject['mean_ratio_error_pct'].max():.2f}%")

    # Count subjects by error threshold
    n_under_1pct = (per_subject['mean_ratio_error_pct'] < 1).sum()
    n_under_2pct = (per_subject['mean_ratio_error_pct'] < 2).sum()
    n_under_5pct = (per_subject['mean_ratio_error_pct'] < 5).sum()

    print(f"\nSubjects by mean ratio error threshold:")
    print(f"  <1%: {n_under_1pct}/{len(per_subject)} ({100*n_under_1pct/len(per_subject):.1f}%)")
    print(f"  <2%: {n_under_2pct}/{len(per_subject)} ({100*n_under_2pct/len(per_subject):.1f}%)")
    print(f"  <5%: {n_under_5pct}/{len(per_subject)} ({100*n_under_5pct/len(per_subject):.1f}%)")

    # By-ratio statistics
    print(f"\nPer-ratio error statistics (across all subjects):")
    print(f"  SR3/SR1 error: {per_subject['error_sr3_sr1_pct'].mean():.2f}% ± {per_subject['error_sr3_sr1_pct'].std():.2f}%")
    print(f"  SR5/SR1 error: {per_subject['error_sr5_sr1_pct'].mean():.2f}% ± {per_subject['error_sr5_sr1_pct'].std():.2f}%")
    print(f"  SR5/SR3 error: {per_subject['error_sr5_sr3_pct'].mean():.2f}% ± {per_subject['error_sr5_sr3_pct'].std():.2f}%")

    # Print top 15 subjects by event count
    print("\n" + "=" * 70)
    print("TOP 15 SUBJECTS BY EVENT COUNT")
    print("=" * 70)
    print(f"{'Subject':<15} {'Dataset':<10} {'Events':>6} {'SR3/SR1':>8} {'SR5/SR1':>8} {'SR5/SR3':>8} {'Mean Err':>8}")
    print("-" * 70)

    for _, row in per_subject.head(15).iterrows():
        print(f"{row['subject']:<15} {row['dataset']:<10} {row['n_events']:>6} "
              f"{row['error_sr3_sr1_pct']:>7.2f}% {row['error_sr5_sr1_pct']:>7.2f}% "
              f"{row['error_sr5_sr3_pct']:>7.2f}% {row['mean_ratio_error_pct']:>7.2f}%")

    # Save CSV
    print(f"\n" + "=" * 70)
    print("SAVING OUTPUT FILES")
    print("=" * 70)

    per_subject.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved: {OUTPUT_CSV}")

    # Generate figure
    os.makedirs(os.path.dirname(OUTPUT_FIG), exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left panel: Histogram of mean ratio errors
    ax1 = axes[0]
    bins = np.arange(0, per_subject['mean_ratio_error_pct'].max() + 1, 0.5)
    n, bins_out, patches = ax1.hist(per_subject['mean_ratio_error_pct'], bins=bins,
                                     color='steelblue', edgecolor='white', alpha=0.8)

    # Add vertical lines for thresholds
    ax1.axvline(1, color='green', linestyle='--', linewidth=2, label=f'1% ({n_under_1pct} subjects)')
    ax1.axvline(2, color='orange', linestyle='--', linewidth=2, label=f'2% ({n_under_2pct} subjects)')
    ax1.axvline(5, color='red', linestyle='--', linewidth=2, label=f'5% ({n_under_5pct} subjects)')

    ax1.set_xlabel('Mean φⁿ Ratio Error (%)', fontsize=11)
    ax1.set_ylabel('Number of Subjects', fontsize=11)
    ax1.set_title('Distribution of Per-Subject φⁿ Ratio Errors\n(N = {} subjects)'.format(len(per_subject)), fontsize=12)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')

    # Right panel: Error by ratio type (box plot)
    ax2 = axes[1]
    error_data = [
        per_subject['error_sr3_sr1_pct'].values,
        per_subject['error_sr5_sr1_pct'].values,
        per_subject['error_sr5_sr3_pct'].values
    ]
    bp = ax2.boxplot(error_data, labels=['SR3/SR1\n(vs φ²)', 'SR5/SR1\n(vs φ³)', 'SR5/SR3\n(vs φ)'],
                     patch_artist=True)

    colors = ['#4ECDC4', '#FF6B6B', '#45B7D1']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax2.axhline(1, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
    ax2.axhline(2, color='orange', linestyle='--', linewidth=1.5, alpha=0.7)

    ax2.set_ylabel('Ratio Error (%)', fontsize=11)
    ax2.set_title('Per-Subject Ratio Errors by Harmonic Pair', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(OUTPUT_FIG, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {OUTPUT_FIG}")

    # Print LaTeX table for paper (top 10 by event count)
    print("\n" + "=" * 70)
    print("LATEX TABLE (for paper)")
    print("=" * 70)
    print(r"""
\begin{table}[H]
\centering
\caption{Per-Subject $\phisym^n$ Ratio Precision (Top 10 by Event Count)}
\label{tab:per_subject_phi}
\begin{tabular}{@{}llrrrrrr@{}}
\toprule
Subject & Dataset & Events & SR3/SR1 & Err\% & SR5/SR1 & Err\% & Mean Err\% \\
\midrule""")

    for _, row in per_subject.head(10).iterrows():
        subj = row['subject'].replace('_', r'\_')
        dset = row['dataset']
        n = int(row['n_events'])
        r1 = row['mean_sr3_sr1']
        e1 = row['error_sr3_sr1_pct']
        r2 = row['mean_sr5_sr1']
        e2 = row['error_sr5_sr1_pct']
        me = row['mean_ratio_error_pct']
        print(f"{subj} & {dset} & {n} & {r1:.3f} & {e1:.1f} & {r2:.3f} & {e2:.1f} & {me:.1f} \\\\")

    print(r"""\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item Theoretical values: $\phisym^2 = 2.618$ (SR3/SR1), $\phisym^3 = 4.236$ (SR5/SR1).
Error percentages show absolute deviation from theoretical.
\end{tablenotes}
\end{table}""")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
