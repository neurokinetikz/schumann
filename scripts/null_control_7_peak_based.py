#!/usr/bin/env python3
"""
Null Control 7: Peak-Based Random Triplets from EEG Bands Data
==============================================================

Tests if SIE events show better φ-convergence than random triplets
sampled from the actual distribution of FOOOF peaks in EEG data.

Uses golden_ratio_peaks_ALL.csv from eeg_bands paper analysis.

Key difference from previous null controls:
- Previous: Uniform sampling from frequency ranges (biased - ranges centered on φⁿ)
- This: Sample from ACTUAL detected FOOOF peaks (controls for EEG spectral structure)

Pass Criteria:
- p < 0.05 (SIE triplets significantly more φ-precise than random peak triplets)
- Cohen's d > 0.5 (substantial effect size)

Note: If p > 0.05, this indicates φⁿ organization is inherent to EEG peaks,
and SIEs represent amplification of existing structure (consistent with
the independence-convergence finding).
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional


# ============================================================================
# Constants
# ============================================================================

PHI = (1 + np.sqrt(5)) / 2  # 1.618033988749895
PHI_SQ = PHI ** 2            # 2.618033988749895
PHI_CUBE = PHI ** 3          # 4.236067977499790

# Band definitions for filtering peaks (Hz)
# These should bracket the SR harmonics but not be too tight
SR1_RANGE = (6.5, 9.5)    # Fundamental (~7.6 Hz) - avoid alpha contamination
SR3_RANGE = (18.0, 22.0)  # ~φ² harmonic (~20 Hz)
SR5_RANGE = (30.0, 36.0)  # ~φ³ harmonic (~32 Hz)

# Alternative wider bands for sensitivity analysis
SR1_RANGE_WIDE = (5.0, 11.0)
SR3_RANGE_WIDE = (16.0, 24.0)
SR5_RANGE_WIDE = (28.0, 38.0)

# Number of random triplets
N_RANDOM_TRIPLETS = 10000

# File paths
PEAKS_FILE = 'golden_ratio_peaks_ALL.csv'
SIE_FILE = 'PAPER-3-sie-analysis.csv'


# ============================================================================
# φ-Error Computation
# ============================================================================

def compute_phi_error(f1: float, f2: float, f3: float) -> float:
    """
    Compute mean φ-error for a frequency triplet.

    Expected ratios:
    - f2/f1 ≈ φ² (2.618) - SR3/SR1
    - f3/f1 ≈ φ³ (4.236) - SR5/SR1
    - f3/f2 ≈ φ  (1.618) - SR5/SR3

    Returns percentage error.
    """
    r1 = f2 / f1  # Should be ~φ²
    r2 = f3 / f1  # Should be ~φ³
    r3 = f3 / f2  # Should be ~φ

    # Relative errors
    e1 = abs(r1 - PHI_SQ) / PHI_SQ
    e2 = abs(r2 - PHI_CUBE) / PHI_CUBE
    e3 = abs(r3 - PHI) / PHI

    return np.mean([e1, e2, e3]) * 100  # Percentage


def compute_phi_error_detailed(f1: float, f2: float, f3: float) -> Dict:
    """Compute detailed φ-error breakdown for a triplet."""
    r1 = f2 / f1
    r2 = f3 / f1
    r3 = f3 / f2

    e1 = abs(r1 - PHI_SQ) / PHI_SQ * 100
    e2 = abs(r2 - PHI_CUBE) / PHI_CUBE * 100
    e3 = abs(r3 - PHI) / PHI * 100

    return {
        'ratio_sr3_sr1': r1,
        'ratio_sr5_sr1': r2,
        'ratio_sr5_sr3': r3,
        'error_sr3_sr1': e1,
        'error_sr5_sr1': e2,
        'error_sr5_sr3': e3,
        'mean_error': np.mean([e1, e2, e3])
    }


# ============================================================================
# Data Loading
# ============================================================================

def load_peaks_data(filepath: str = PEAKS_FILE) -> pd.DataFrame:
    """Load FOOOF peaks from eeg_bands analysis."""
    print(f"Loading peaks from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"  Loaded {len(df):,} peaks")
    return df


def load_sie_events(filepath: str = SIE_FILE) -> pd.DataFrame:
    """Load SIE events with harmonic frequencies."""
    print(f"Loading SIE events from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"  Loaded {len(df):,} events")
    return df


def filter_peaks_by_band(peaks_df: pd.DataFrame,
                         freq_range: Tuple[float, float]) -> np.ndarray:
    """Filter peaks to a frequency band and return frequencies."""
    mask = (peaks_df['freq'] >= freq_range[0]) & (peaks_df['freq'] <= freq_range[1])
    return peaks_df.loc[mask, 'freq'].values


# ============================================================================
# Triplet Generation
# ============================================================================

def generate_random_triplets(sr1_pool: np.ndarray,
                            sr3_pool: np.ndarray,
                            sr5_pool: np.ndarray,
                            n: int = N_RANDOM_TRIPLETS,
                            seed: int = None) -> np.ndarray:
    """
    Generate random triplets by sampling from peak pools.

    Returns array of φ-errors for each triplet.
    """
    if seed is not None:
        np.random.seed(seed)

    errors = []
    for _ in range(n):
        f1 = np.random.choice(sr1_pool)
        f2 = np.random.choice(sr3_pool)
        f3 = np.random.choice(sr5_pool)
        errors.append(compute_phi_error(f1, f2, f3))

    return np.array(errors)


def extract_sie_errors(sie_df: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Extract φ-errors from SIE events.

    Returns (errors_array, valid_events_df)
    """
    # Filter to events with all three harmonics
    valid_mask = (
        sie_df['sr1'].notna() &
        sie_df['sr3'].notna() &
        sie_df['sr5'].notna()
    )
    valid_df = sie_df[valid_mask].copy()

    errors = []
    for _, row in valid_df.iterrows():
        errors.append(compute_phi_error(row['sr1'], row['sr3'], row['sr5']))

    return np.array(errors), valid_df


# ============================================================================
# Statistical Analysis
# ============================================================================

def compute_statistics(sie_errors: np.ndarray,
                       random_errors: np.ndarray) -> Dict:
    """Compute comprehensive comparison statistics."""

    sie_mean = np.mean(sie_errors)
    sie_std = np.std(sie_errors)
    sie_median = np.median(sie_errors)

    random_mean = np.mean(random_errors)
    random_std = np.std(random_errors)
    random_median = np.median(random_errors)

    # Effect size (Cohen's d) - positive means SIE is better (lower error)
    pooled_std = np.sqrt((sie_std**2 + random_std**2) / 2)
    cohens_d = (random_mean - sie_mean) / pooled_std

    # Percentile rank (what % of random triplets have HIGHER error than SIE mean)
    percentile = 100 * np.mean(random_errors > sie_mean)

    # P-value (proportion of random triplets with error <= SIE mean)
    p_value = np.mean(random_errors <= sie_mean)

    # Mann-Whitney U test (non-parametric)
    u_stat, p_mw = stats.mannwhitneyu(sie_errors, random_errors, alternative='less')

    # Permutation test
    observed_diff = sie_mean - random_mean
    combined = np.concatenate([sie_errors, random_errors])
    n_sie = len(sie_errors)

    n_perms = 10000
    perm_diffs = []
    for _ in range(n_perms):
        np.random.shuffle(combined)
        perm_sie = combined[:n_sie]
        perm_random = combined[n_sie:]
        perm_diffs.append(np.mean(perm_sie) - np.mean(perm_random))

    p_perm = np.mean(np.array(perm_diffs) <= observed_diff)

    # Bootstrap CI for SIE mean
    n_boot = 10000
    boot_means = []
    for _ in range(n_boot):
        boot_sample = np.random.choice(sie_errors, len(sie_errors), replace=True)
        boot_means.append(np.mean(boot_sample))
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])

    return {
        'sie_mean': sie_mean,
        'sie_std': sie_std,
        'sie_median': sie_median,
        'sie_ci_low': ci_low,
        'sie_ci_high': ci_high,
        'random_mean': random_mean,
        'random_std': random_std,
        'random_median': random_median,
        'cohens_d': cohens_d,
        'percentile': percentile,
        'p_value_empirical': p_value,
        'p_value_mannwhitney': p_mw,
        'p_value_permutation': p_perm,
        'n_sie': len(sie_errors),
        'n_random': len(random_errors)
    }


# ============================================================================
# Visualization
# ============================================================================

def create_visualizations(sie_errors: np.ndarray,
                         random_errors: np.ndarray,
                         stats_results: Dict,
                         output_dir: Path):
    """Create comprehensive visualizations."""

    sns.set_style("whitegrid")
    plt.rcParams['font.size'] = 10

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # 1. Histogram comparison
    ax1 = fig.add_subplot(gs[0, 0])
    bins = np.linspace(0, max(np.max(sie_errors), np.percentile(random_errors, 99)), 50)

    ax1.hist(random_errors, bins=bins, alpha=0.6, label=f'Random Peaks (n={len(random_errors):,})',
             color='gray', density=True)
    ax1.hist(sie_errors, bins=bins, alpha=0.8, label=f'SIE Events (n={len(sie_errors)})',
             color='red', density=True)

    ax1.axvline(stats_results['sie_mean'], color='red', linestyle='--', linewidth=2,
                label=f"SIE mean: {stats_results['sie_mean']:.2f}%")
    ax1.axvline(stats_results['random_mean'], color='gray', linestyle='--', linewidth=2,
                label=f"Random mean: {stats_results['random_mean']:.2f}%")

    ax1.set_xlabel('Mean φ-Error (%)')
    ax1.set_ylabel('Density')
    ax1.set_title('φ-Error Distributions: SIE vs Random Peak Triplets')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # 2. Box plot comparison
    ax2 = fig.add_subplot(gs[0, 1])
    bp = ax2.boxplot([random_errors, sie_errors], labels=['Random\nPeaks', 'SIE\nEvents'],
                     patch_artist=True, widths=0.6)
    bp['boxes'][0].set_facecolor('gray')
    bp['boxes'][0].set_alpha(0.6)
    bp['boxes'][1].set_facecolor('red')
    bp['boxes'][1].set_alpha(0.8)

    ax2.set_ylabel('Mean φ-Error (%)')
    ax2.set_title('φ-Error Comparison')
    ax2.grid(True, alpha=0.3, axis='y')

    # Add statistics annotation
    textstr = f"Cohen's d = {stats_results['cohens_d']:.3f}\n"
    textstr += f"p (perm) = {stats_results['p_value_permutation']:.4f}\n"
    textstr += f"p (M-W) = {stats_results['p_value_mannwhitney']:.4f}"
    ax2.text(0.95, 0.95, textstr, transform=ax2.transAxes, ha='right', va='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=9)

    # 3. Cumulative distribution
    ax3 = fig.add_subplot(gs[1, 0])
    random_sorted = np.sort(random_errors)
    sie_sorted = np.sort(sie_errors)
    random_cdf = np.arange(1, len(random_sorted) + 1) / len(random_sorted)
    sie_cdf = np.arange(1, len(sie_sorted) + 1) / len(sie_sorted)

    ax3.plot(random_sorted, random_cdf, label='Random Peaks', color='gray', linewidth=2)
    ax3.plot(sie_sorted, sie_cdf, label='SIE Events', color='red', linewidth=2)
    ax3.axvline(stats_results['sie_mean'], color='red', linestyle='--', alpha=0.5)
    ax3.axhline(stats_results['percentile']/100, color='gray', linestyle=':', alpha=0.5)

    ax3.set_xlabel('Mean φ-Error (%)')
    ax3.set_ylabel('Cumulative Probability')
    ax3.set_title('Cumulative Distribution Functions')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Difference from expected
    ax4 = fig.add_subplot(gs[1, 1])

    # Show how far each distribution is from perfect φ
    percentiles = [5, 10, 25, 50, 75, 90, 95]
    random_pcts = np.percentile(random_errors, percentiles)
    sie_pcts = np.percentile(sie_errors, percentiles)

    x = np.arange(len(percentiles))
    width = 0.35
    ax4.bar(x - width/2, random_pcts, width, label='Random Peaks', color='gray', alpha=0.6)
    ax4.bar(x + width/2, sie_pcts, width, label='SIE Events', color='red', alpha=0.8)

    ax4.set_xlabel('Percentile')
    ax4.set_ylabel('Mean φ-Error (%)')
    ax4.set_title('Percentile Comparison')
    ax4.set_xticks(x)
    ax4.set_xticklabels([f'{p}th' for p in percentiles])
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    # Overall status
    is_significant = stats_results['p_value_permutation'] < 0.05
    status = "SIGNIFICANT (p < 0.05)" if is_significant else "NOT SIGNIFICANT (p > 0.05)"
    color = 'green' if is_significant else 'orange'
    fig.suptitle(f'Peak-Based Null Control: {status}', fontsize=14, fontweight='bold', color=color)

    plt.tight_layout()
    plt.savefig(output_dir / 'nc7_peak_based_results.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'nc7_peak_based_results.pdf', bbox_inches='tight')
    plt.close()

    print(f"Visualizations saved to {output_dir}")


def create_supplementary_figure(sie_df: pd.DataFrame,
                                peaks_df: pd.DataFrame,
                                sr1_range: Tuple,
                                sr3_range: Tuple,
                                sr5_range: Tuple,
                                output_dir: Path):
    """Create supplementary figure showing peak distributions."""

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    bands = [
        ('SR1', sr1_range, 'sr1', 'C0'),
        ('SR3', sr3_range, 'sr3', 'C1'),
        ('SR5', sr5_range, 'sr5', 'C2')
    ]

    for ax, (name, freq_range, col, color) in zip(axes, bands):
        # Peak distribution
        pool = filter_peaks_by_band(peaks_df, freq_range)
        ax.hist(pool, bins=50, alpha=0.6, color='gray', density=True,
                label=f'All peaks (n={len(pool):,})')

        # SIE frequencies
        sie_freqs = sie_df[col].dropna().values
        ax.hist(sie_freqs, bins=30, alpha=0.8, color=color, density=True,
                label=f'SIE events (n={len(sie_freqs)})')

        ax.axvline(np.mean(pool), color='gray', linestyle='--',
                   label=f'Peak mean: {np.mean(pool):.2f}')
        ax.axvline(np.mean(sie_freqs), color=color, linestyle='--',
                   label=f'SIE mean: {np.mean(sie_freqs):.2f}')

        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Density')
        ax.set_title(f'{name} Band ({freq_range[0]}-{freq_range[1]} Hz)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'nc7_peak_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()


# ============================================================================
# Results Report
# ============================================================================

def generate_report(stats_results: Dict,
                    pool_sizes: Dict,
                    output_dir: Path) -> str:
    """Generate markdown results report."""

    is_sig = stats_results['p_value_permutation'] < 0.05
    status = "SIGNIFICANT" if is_sig else "NOT SIGNIFICANT"

    report = f"""# Null Control 7: Peak-Based Random Triplets - Results

**Status: {status}** (p = {stats_results['p_value_permutation']:.4f})

## Summary

This test compares SIE frequency triplets against random triplets sampled from
**actual FOOOF peaks** detected in EEG recordings. Unlike the uniform sampling
null (which samples uniformly from frequency ranges), this test controls for
the actual spectral structure of EEG data.

## Peak Pool Sizes

| Band | Frequency Range | N Peaks |
|------|----------------|---------|
| SR1 | {SR1_RANGE[0]}-{SR1_RANGE[1]} Hz | {pool_sizes['sr1']:,} |
| SR3 | {SR3_RANGE[0]}-{SR3_RANGE[1]} Hz | {pool_sizes['sr3']:,} |
| SR5 | {SR5_RANGE[0]}-{SR5_RANGE[1]} Hz | {pool_sizes['sr5']:,} |

## Statistical Results

| Metric | SIE Events | Random Peaks |
|--------|-----------|--------------|
| N | {stats_results['n_sie']} | {stats_results['n_random']:,} |
| Mean φ-error | {stats_results['sie_mean']:.2f}% | {stats_results['random_mean']:.2f}% |
| Std | {stats_results['sie_std']:.2f}% | {stats_results['random_std']:.2f}% |
| Median | {stats_results['sie_median']:.2f}% | {stats_results['random_median']:.2f}% |

### Statistical Tests

| Test | Value | Interpretation |
|------|-------|----------------|
| Cohen's d | {stats_results['cohens_d']:.3f} | {'Large' if abs(stats_results['cohens_d']) > 0.8 else 'Medium' if abs(stats_results['cohens_d']) > 0.5 else 'Small'} effect |
| Percentile | {stats_results['percentile']:.1f}% | {stats_results['percentile']:.1f}% of random triplets worse than SIE mean |
| p (permutation) | {stats_results['p_value_permutation']:.4f} | {'Significant' if stats_results['p_value_permutation'] < 0.05 else 'Not significant'} |
| p (Mann-Whitney) | {stats_results['p_value_mannwhitney']:.4f} | {'Significant' if stats_results['p_value_mannwhitney'] < 0.05 else 'Not significant'} |

### SIE Mean 95% CI

{stats_results['sie_mean']:.2f}% [{stats_results['sie_ci_low']:.2f}%, {stats_results['sie_ci_high']:.2f}%]

## Interpretation

"""

    if is_sig:
        report += """SIE events show **significantly better φ-precision** than random triplets
sampled from actual EEG peaks. This indicates that the SIE detection algorithm
preferentially selects frequency combinations that are more φ-aligned than the
typical EEG spectral structure would produce by chance.

**Implication**: The φⁿ organization in SIE events is not simply an artifact of
EEG spectral structure; SIEs represent genuinely exceptional frequency combinations.
"""
    else:
        report += """SIE events do **not** show significantly better φ-precision than random
triplets sampled from actual EEG peaks. This indicates that **φⁿ organization is
inherent to EEG spectral peaks** - when you sample peaks from the actual frequency
distributions in EEG data, they naturally produce good φ-ratios.

**Implication**: SIEs are not "spectrally exceptional" in terms of their frequency
ratios. Instead, they represent **high-power, high-coherence amplification** of a
φⁿ architecture that exists continuously in EEG. This is consistent with the
independence-convergence finding: individual harmonic frequencies vary independently,
yet ratios are preserved because the marginal frequency distributions are already
constrained to φⁿ values.

**This is actually the expected result** given the paper's main finding that φⁿ
relationships are encoded at the population level rather than through event-level
coordination.
"""

    report += f"""

## Comparison to Uniform Null

| Null Model | p-value | Cohen's d | Interpretation |
|------------|---------|-----------|----------------|
| Uniform sampling | 0.074 | 1.71 | Marginal (biased null) |
| **Peak-based** | **{stats_results['p_value_permutation']:.3f}** | **{stats_results['cohens_d']:.2f}** | {'Significant' if is_sig else 'Not significant'} (proper null) |

The peak-based null is more appropriate because it controls for the actual
spectral structure of EEG data rather than assuming uniform frequency distributions.

---

*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

    report_path = output_dir / 'nc7_results_summary.md'
    with open(report_path, 'w') as f:
        f.write(report)

    print(f"Report saved to {report_path}")
    return report


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main execution function."""

    print("=" * 70)
    print("NULL CONTROL 7: Peak-Based Random Triplets")
    print("=" * 70)
    print()

    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(f'results/null_control_7_{timestamp}')
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    print()

    # Load data
    peaks_df = load_peaks_data()
    sie_df = load_sie_events()
    print()

    # Filter peaks into SR bands
    print("Filtering peaks by SR band...")
    sr1_pool = filter_peaks_by_band(peaks_df, SR1_RANGE)
    sr3_pool = filter_peaks_by_band(peaks_df, SR3_RANGE)
    sr5_pool = filter_peaks_by_band(peaks_df, SR5_RANGE)

    pool_sizes = {
        'sr1': len(sr1_pool),
        'sr3': len(sr3_pool),
        'sr5': len(sr5_pool)
    }

    print(f"  SR1 ({SR1_RANGE[0]}-{SR1_RANGE[1]} Hz): {len(sr1_pool):,} peaks")
    print(f"  SR3 ({SR3_RANGE[0]}-{SR3_RANGE[1]} Hz): {len(sr3_pool):,} peaks")
    print(f"  SR5 ({SR5_RANGE[0]}-{SR5_RANGE[1]} Hz): {len(sr5_pool):,} peaks")
    print()

    # Check for sufficient peaks
    min_peaks = 100
    if len(sr1_pool) < min_peaks or len(sr3_pool) < min_peaks or len(sr5_pool) < min_peaks:
        print(f"ERROR: Insufficient peaks in one or more bands (minimum {min_peaks} required)")
        return

    # Extract SIE errors
    print("Computing SIE φ-errors...")
    sie_errors, valid_sie_df = extract_sie_errors(sie_df)
    print(f"  Valid SIE events (with all harmonics): {len(sie_errors)}")
    print(f"  Mean SIE φ-error: {np.mean(sie_errors):.2f}%")
    print()

    # Generate random triplets
    print(f"Generating {N_RANDOM_TRIPLETS:,} random triplets from peak pools...")
    random_errors = generate_random_triplets(sr1_pool, sr3_pool, sr5_pool,
                                             n=N_RANDOM_TRIPLETS, seed=42)
    print(f"  Mean random φ-error: {np.mean(random_errors):.2f}%")
    print()

    # Compute statistics
    print("Computing statistics...")
    stats_results = compute_statistics(sie_errors, random_errors)
    print()

    # Print results
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"SIE mean φ-error:    {stats_results['sie_mean']:.2f}% ± {stats_results['sie_std']:.2f}%")
    print(f"Random mean φ-error: {stats_results['random_mean']:.2f}% ± {stats_results['random_std']:.2f}%")
    print()
    print(f"Cohen's d:           {stats_results['cohens_d']:.3f}")
    print(f"Percentile:          {stats_results['percentile']:.1f}% of random triplets worse than SIE")
    print(f"P-value (perm):      {stats_results['p_value_permutation']:.4f}")
    print(f"P-value (M-W):       {stats_results['p_value_mannwhitney']:.4f}")
    print()

    is_sig = stats_results['p_value_permutation'] < 0.05
    if is_sig:
        print("STATUS: SIGNIFICANT - SIE events ARE more φ-precise than typical EEG peaks")
    else:
        print("STATUS: NOT SIGNIFICANT - φⁿ organization is inherent to EEG spectral peaks")
        print("        (consistent with population-level encoding hypothesis)")
    print("=" * 70)
    print()

    # Create visualizations
    print("Creating visualizations...")
    create_visualizations(sie_errors, random_errors, stats_results, output_dir)
    create_supplementary_figure(valid_sie_df, peaks_df, SR1_RANGE, SR3_RANGE, SR5_RANGE, output_dir)
    print()

    # Generate report
    print("Generating report...")
    generate_report(stats_results, pool_sizes, output_dir)
    print()

    # Save raw data
    print("Saving raw data...")

    # Save statistics
    stats_df = pd.DataFrame([stats_results])
    stats_df.to_csv(output_dir / 'statistics.csv', index=False)

    # Save error distributions (sample for random)
    errors_df = pd.DataFrame({
        'sie_errors': np.concatenate([sie_errors, [np.nan] * (N_RANDOM_TRIPLETS - len(sie_errors))]),
        'random_errors': random_errors
    })
    errors_df.to_csv(output_dir / 'error_distributions.csv', index=False)

    print(f"Raw data saved to {output_dir}")
    print()

    print("=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"All results saved to: {output_dir}")
    print()

    return stats_results


if __name__ == "__main__":
    results = main()
