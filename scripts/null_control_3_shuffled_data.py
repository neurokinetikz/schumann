#!/usr/bin/env python3
"""
Null Control Test 3: Shuffled Data Bootstrap

Tests if φ-ratios depend on specific SR1-SR3-SR5 pairings or just the frequency
distributions by randomly shuffling the values across events.

This test takes ACTUAL detected SIE events and independently shuffles the SR1,
SR3, and SR5 values across events. If φ-ratios are genuine harmonic relationships,
shuffling should destroy them. If they arise only from frequency distributions,
shuffling won't matter.

Pass Criteria:
- Observed composite error < 5th percentile of shuffled
- p < 0.05 for composite error
- All three individual ratios significant
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from scipy import stats
from dataclasses import dataclass

# Golden ratio and powers
PHI = (1 + np.sqrt(5)) / 2  # 1.618033988749895
PHI_SQ = PHI ** 2            # 2.618033988749895
PHI_CUBE = PHI ** 3          # 4.236067977499790

# Bootstrap parameters
N_BOOTSTRAP_ITERATIONS = 10000

# Default CSV path
DEFAULT_CSV_PATH = "data/SIE.csv"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class PhiErrors:
    """φ-ratio errors for a single triplet"""
    error_f2_f1: float  # |f2/f1 - φ²|
    error_f3_f1: float  # |f3/f1 - φ³|
    error_f3_f2: float  # |f3/f2 - φ|
    composite_error: float  # Mean of all three errors


# ============================================================================
# Data Loading
# ============================================================================

def load_sie_events(csv_path: str) -> pd.DataFrame:
    """
    Load SIE event data from CSV.

    Args:
        csv_path: Path to CSV file with sr1, sr3, sr5 columns

    Returns:
        DataFrame with sr1, sr3, sr5 columns
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Check for required columns
    required_cols = ['sr1', 'sr3', 'sr5']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}. Found: {df.columns.tolist()}")

    # Remove rows with NaN values in SR columns
    df = df.dropna(subset=['sr1', 'sr3', 'sr5'])

    if len(df) == 0:
        raise ValueError("No valid events found (all rows have NaN in SR columns)")

    print(f"Loaded {len(df)} SIE events from {csv_path}")

    return df


# ============================================================================
# φ-Ratio Calculations
# ============================================================================

def calculate_phi_errors(sr1: np.ndarray, sr3: np.ndarray, sr5: np.ndarray) -> PhiErrors:
    """
    Calculate φ-ratio errors for frequency triplets.

    Always divides larger by smaller to match pipeline convention:
    - Sort frequencies (f1 < f2 < f3)
    - Calculate f2/f1, f3/f1, f3/f2
    - Compare to φ², φ³, φ

    Args:
        sr1, sr3, sr5: Arrays of SR frequencies

    Returns:
        PhiErrors with individual and composite errors
    """
    # Stack and sort to ensure f1 < f2 < f3
    triplets = np.column_stack([sr1, sr3, sr5])
    sorted_triplets = np.sort(triplets, axis=1)

    f1 = sorted_triplets[:, 0]
    f2 = sorted_triplets[:, 1]
    f3 = sorted_triplets[:, 2]

    # Calculate ratios (always > 1)
    ratio_f2_f1 = f2 / f1
    ratio_f3_f1 = f3 / f1
    ratio_f3_f2 = f3 / f2

    # Calculate errors from golden ratio expectations
    error_f2_f1 = np.abs(ratio_f2_f1 - PHI_SQ)
    error_f3_f1 = np.abs(ratio_f3_f1 - PHI_CUBE)
    error_f3_f2 = np.abs(ratio_f3_f2 - PHI)

    # Composite error (mean of all three)
    composite_error = (error_f2_f1 + error_f3_f1 + error_f3_f2) / 3

    return PhiErrors(
        error_f2_f1=error_f2_f1,
        error_f3_f1=error_f3_f1,
        error_f3_f2=error_f3_f2,
        composite_error=composite_error
    )


# ============================================================================
# Bootstrap Shuffling
# ============================================================================

def shuffle_bootstrap(sr1: np.ndarray, sr3: np.ndarray, sr5: np.ndarray,
                     n_iterations: int = 10000) -> Dict:
    """
    Perform bootstrap shuffling of SR values.

    Independently shuffles sr1, sr3, sr5 arrays across events to test if
    φ-ratios depend on specific pairings or just frequency distributions.

    Args:
        sr1, sr3, sr5: Original SR frequency arrays
        n_iterations: Number of bootstrap shuffles

    Returns:
        Dictionary with shuffled error distributions
    """
    print(f"Performing {n_iterations} bootstrap shuffles...")

    n_events = len(sr1)

    # Storage for shuffled errors
    shuffled_composite_errors = []
    shuffled_errors_f2_f1 = []
    shuffled_errors_f3_f1 = []
    shuffled_errors_f3_f2 = []

    for i in range(n_iterations):
        if (i + 1) % 1000 == 0:
            print(f"  Iteration {i + 1}/{n_iterations}")

        # Independently shuffle each SR column
        shuffled_sr1 = np.random.permutation(sr1)
        shuffled_sr3 = np.random.permutation(sr3)
        shuffled_sr5 = np.random.permutation(sr5)

        # Calculate φ-errors for shuffled triplets
        errors = calculate_phi_errors(shuffled_sr1, shuffled_sr3, shuffled_sr5)

        # Store mean errors across all events
        shuffled_composite_errors.append(np.mean(errors.composite_error))
        shuffled_errors_f2_f1.append(np.mean(errors.error_f2_f1))
        shuffled_errors_f3_f1.append(np.mean(errors.error_f3_f1))
        shuffled_errors_f3_f2.append(np.mean(errors.error_f3_f2))

    return {
        'composite_errors': np.array(shuffled_composite_errors),
        'errors_f2_f1': np.array(shuffled_errors_f2_f1),
        'errors_f3_f1': np.array(shuffled_errors_f3_f1),
        'errors_f3_f2': np.array(shuffled_errors_f3_f2),
        'n_iterations': n_iterations,
        'n_events': n_events
    }


# ============================================================================
# Statistical Testing
# ============================================================================

def perform_statistical_test(observed_errors: PhiErrors,
                            shuffled_results: Dict) -> Dict:
    """
    Compare observed vs shuffled error distributions.

    Args:
        observed_errors: PhiErrors from actual event pairings
        shuffled_results: Bootstrap shuffle results

    Returns:
        Dictionary with statistical test results
    """
    # Calculate mean observed errors
    obs_composite = np.mean(observed_errors.composite_error)
    obs_f2_f1 = np.mean(observed_errors.error_f2_f1)
    obs_f3_f1 = np.mean(observed_errors.error_f3_f1)
    obs_f3_f2 = np.mean(observed_errors.error_f3_f2)

    # Shuffled distributions
    shuf_composite = shuffled_results['composite_errors']
    shuf_f2_f1 = shuffled_results['errors_f2_f1']
    shuf_f3_f1 = shuffled_results['errors_f3_f1']
    shuf_f3_f2 = shuffled_results['errors_f3_f2']

    # Calculate percentile ranks (lower is better)
    percentile_composite = stats.percentileofscore(shuf_composite, obs_composite)
    percentile_f2_f1 = stats.percentileofscore(shuf_f2_f1, obs_f2_f1)
    percentile_f3_f1 = stats.percentileofscore(shuf_f3_f1, obs_f3_f1)
    percentile_f3_f2 = stats.percentileofscore(shuf_f3_f2, obs_f3_f2)

    # Calculate p-values (one-tailed: observed < shuffled)
    p_composite = np.sum(shuf_composite <= obs_composite) / len(shuf_composite)
    p_f2_f1 = np.sum(shuf_f2_f1 <= obs_f2_f1) / len(shuf_f2_f1)
    p_f3_f1 = np.sum(shuf_f3_f1 <= obs_f3_f1) / len(shuf_f3_f1)
    p_f3_f2 = np.sum(shuf_f3_f2 <= obs_f3_f2) / len(shuf_f3_f2)

    # Pass criteria
    passes_percentile = percentile_composite <= 5  # Observed < 5th percentile
    passes_pvalue = p_composite < 0.05  # p < 0.05
    passes_individual = (p_f2_f1 < 0.05) and (p_f3_f1 < 0.05) and (p_f3_f2 < 0.05)

    overall_pass = passes_percentile and passes_pvalue and passes_individual

    return {
        'n_events': shuffled_results['n_events'],
        'n_iterations': shuffled_results['n_iterations'],

        # Observed errors
        'obs_composite': obs_composite,
        'obs_f2_f1': obs_f2_f1,
        'obs_f3_f1': obs_f3_f1,
        'obs_f3_f2': obs_f3_f2,

        # Shuffled error statistics
        'shuf_composite_mean': np.mean(shuf_composite),
        'shuf_composite_std': np.std(shuf_composite),
        'shuf_f2_f1_mean': np.mean(shuf_f2_f1),
        'shuf_f3_f1_mean': np.mean(shuf_f3_f1),
        'shuf_f3_f2_mean': np.mean(shuf_f3_f2),

        # Percentiles
        'percentile_composite': percentile_composite,
        'percentile_f2_f1': percentile_f2_f1,
        'percentile_f3_f1': percentile_f3_f1,
        'percentile_f3_f2': percentile_f3_f2,

        # P-values
        'p_composite': p_composite,
        'p_f2_f1': p_f2_f1,
        'p_f3_f1': p_f3_f1,
        'p_f3_f2': p_f3_f2,

        # Pass/fail
        'passes_percentile': passes_percentile,
        'passes_pvalue': passes_pvalue,
        'passes_individual': passes_individual,
        'overall_pass': overall_pass
    }


# ============================================================================
# Visualization
# ============================================================================

def create_visualizations(observed_errors: PhiErrors,
                         shuffled_results: Dict,
                         stats_results: Dict,
                         output_dir: Path):
    """
    Create comprehensive 4-panel visualization.

    Args:
        observed_errors: Observed φ-errors
        shuffled_results: Bootstrap shuffle results
        stats_results: Statistical test results
        output_dir: Directory to save plots
    """
    obs_composite = stats_results['obs_composite']
    shuf_composite = shuffled_results['composite_errors']

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['font.size'] = 10

    # Create figure with 4 subplots
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # 1. Histogram with distributions
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(shuf_composite, bins=50, alpha=0.6, label='Shuffled', color='gray', density=True)
    ax1.axvline(obs_composite, color='red', linestyle='--', linewidth=2,
                label=f'Observed: {obs_composite:.4f}')
    ax1.axvline(stats_results['shuf_composite_mean'], color='gray', linestyle='--', linewidth=2,
                label=f'Shuffled mean: {stats_results["shuf_composite_mean"]:.4f}')
    ax1.set_xlabel('Composite φ-Error')
    ax1.set_ylabel('Density')
    ax1.set_title('Composite Error: Observed vs Shuffled')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. CDF comparison
    ax2 = fig.add_subplot(gs[0, 1])
    shuf_sorted = np.sort(shuf_composite)
    cdf = np.arange(1, len(shuf_sorted) + 1) / len(shuf_sorted)
    ax2.plot(shuf_sorted, cdf, label='Shuffled CDF', color='gray', linewidth=2)
    ax2.axvline(obs_composite, color='red', linestyle='--', linewidth=2,
                label=f'Observed ({stats_results["percentile_composite"]:.1f}th percentile)')
    ax2.axhline(0.05, color='orange', linestyle=':', linewidth=2, label='5th percentile threshold')
    ax2.set_xlabel('Composite φ-Error')
    ax2.set_ylabel('Cumulative Probability')
    ax2.set_title('Cumulative Distribution Function')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Individual ratio comparison
    ax3 = fig.add_subplot(gs[1, 0])
    ratios = ['f2/f1\n(≈φ²)', 'f3/f1\n(≈φ³)', 'f3/f2\n(≈φ)']
    obs_values = [stats_results['obs_f2_f1'], stats_results['obs_f3_f1'], stats_results['obs_f3_f2']]
    shuf_means = [stats_results['shuf_f2_f1_mean'], stats_results['shuf_f3_f1_mean'],
                  stats_results['shuf_f3_f2_mean']]

    x = np.arange(len(ratios))
    width = 0.35
    ax3.bar(x - width/2, obs_values, width, label='Observed', color='red', alpha=0.8)
    ax3.bar(x + width/2, shuf_means, width, label='Shuffled Mean', color='gray', alpha=0.6)
    ax3.set_xlabel('Ratio')
    ax3.set_ylabel('Mean φ-Error')
    ax3.set_title('Individual Ratio Errors')
    ax3.set_xticks(x)
    ax3.set_xticklabels(ratios)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    # 4. P-value summary
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')

    # Create summary table
    summary_text = f"""
STATISTICAL SUMMARY

Composite Error:
  Observed: {obs_composite:.4f}
  Shuffled: {stats_results['shuf_composite_mean']:.4f} ± {stats_results['shuf_composite_std']:.4f}
  Percentile: {stats_results['percentile_composite']:.1f}th
  P-value: {stats_results['p_composite']:.4f}

Individual Ratios:
  f2/f1 (φ²): p = {stats_results['p_f2_f1']:.4f} {'✓' if stats_results['p_f2_f1'] < 0.05 else '✗'}
  f3/f1 (φ³): p = {stats_results['p_f3_f1']:.4f} {'✓' if stats_results['p_f3_f1'] < 0.05 else '✗'}
  f3/f2 (φ):  p = {stats_results['p_f3_f2']:.4f} {'✓' if stats_results['p_f3_f2'] < 0.05 else '✗'}

Pass Criteria:
  Percentile ≤ 5th:  {'✓ PASS' if stats_results['passes_percentile'] else '✗ FAIL'}
  P-value < 0.05:    {'✓ PASS' if stats_results['passes_pvalue'] else '✗ FAIL'}
  All ratios sig.:   {'✓ PASS' if stats_results['passes_individual'] else '✗ FAIL'}

OVERALL: {'✓ PASS' if stats_results['overall_pass'] else '✗ FAIL'}
"""

    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes,
             fontsize=10, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    # Add overall pass/fail annotation
    status = "PASS ✓" if stats_results['overall_pass'] else "FAIL ✗"
    color = 'green' if stats_results['overall_pass'] else 'red'
    fig.text(0.5, 0.98, f'Null Control Test 3 (Shuffled Bootstrap): {status}',
             ha='center', va='top', fontsize=14, fontweight='bold', color=color)

    # Save figure
    plt.savefig(output_dir / 'phi_bootstrap_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'phi_bootstrap_analysis.pdf', bbox_inches='tight')
    plt.close()

    print(f"Visualizations saved to {output_dir}")


# ============================================================================
# Results Report
# ============================================================================

def generate_results_report(stats_results: Dict, output_dir: Path):
    """
    Generate 200-word results summary in markdown format.

    Args:
        stats_results: Statistical test results
        output_dir: Directory to save report
    """
    status = "PASSED" if stats_results['overall_pass'] else "FAILED"

    report = f"""# Null Control Test 3: Shuffled Data Bootstrap - Results

**Status: {status}**

## Summary

We tested whether φ-ratios in Schumann Ignition Events (SIE) depend on specific SR1-SR3-SR5 pairings or arise merely from the frequency distributions. Using {stats_results['n_events']} detected SIE events, we performed {stats_results['n_iterations']:,} bootstrap iterations where SR1, SR3, and SR5 values were independently shuffled across events. If φ-ratios reflect genuine harmonic relationships, shuffling should destroy them; if they arise only from frequency distributions, shuffling should have no effect.

## Statistical Results

- **Observed composite error**: {stats_results['obs_composite']:.4f}
- **Shuffled composite error**: {stats_results['shuf_composite_mean']:.4f} ± {stats_results['shuf_composite_std']:.4f}
- **Observed percentile**: {stats_results['percentile_composite']:.1f}th {'✓' if stats_results['passes_percentile'] else '✗'} (criterion: ≤ 5th)
- **P-value (composite)**: {stats_results['p_composite']:.4f} {'✓' if stats_results['passes_pvalue'] else '✗'} (criterion: < 0.05)

### Individual Ratios
- **f2/f1 (φ²)**: p = {stats_results['p_f2_f1']:.4f} {'✓' if stats_results['p_f2_f1'] < 0.05 else '✗'}
- **f3/f1 (φ³)**: p = {stats_results['p_f3_f1']:.4f} {'✓' if stats_results['p_f3_f1'] < 0.05 else '✗'}
- **f3/f2 (φ)**: p = {stats_results['p_f3_f2']:.4f} {'✓' if stats_results['p_f3_f2'] < 0.05 else '✗'}

## Interpretation

{'Observed SIE events show significantly better φ-convergence than shuffled events, indicating that φ-ratios depend on specific SR1-SR3-SR5 pairings. This demonstrates genuine harmonic coupling rather than artifacts of frequency distributions.' if stats_results['overall_pass'] else 'Shuffling SR values does not significantly affect φ-convergence, suggesting φ-ratios may arise from frequency distributions alone rather than specific harmonic pairings. This indicates potential artifacts rather than genuine coupling.'}

## Pass Criteria

- [{'x' if stats_results['passes_percentile'] else ' '}] Observed < 5th percentile of shuffled
- [{'x' if stats_results['passes_pvalue'] else ' '}] p < 0.05 for composite error
- [{'x' if stats_results['passes_individual'] else ' '}] All three ratios individually significant

**Overall: {'PASS' if stats_results['overall_pass'] else 'FAIL'}**

---

*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

    # Save report
    report_path = output_dir / 'results_summary.md'
    with open(report_path, 'w') as f:
        f.write(report)

    print(f"Results summary saved to {report_path}")

    return report


# ============================================================================
# Main Execution
# ============================================================================

def main(csv_path: Optional[str] = None, n_iterations: Optional[int] = None):
    """
    Main execution function.

    Args:
        csv_path: Path to CSV file with SIE events (default: DEFAULT_CSV_PATH)
        n_iterations: Number of bootstrap iterations (default: N_BOOTSTRAP_ITERATIONS)
    """
    if csv_path is None:
        csv_path = DEFAULT_CSV_PATH

    if n_iterations is None:
        n_iterations = N_BOOTSTRAP_ITERATIONS

    print("="*80)
    print("NULL CONTROL TEST 3: Shuffled Data Bootstrap")
    print("="*80)
    print()

    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(f'results/null_control_shuffled_{timestamp}')
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    print()

    # Load SIE events
    print("Loading SIE events...")
    df = load_sie_events(csv_path)
    print()

    # Extract SR values
    sr1 = df['sr1'].values
    sr3 = df['sr3'].values
    sr5 = df['sr5'].values

    print(f"SR1 range: {sr1.min():.2f} - {sr1.max():.2f} Hz")
    print(f"SR3 range: {sr3.min():.2f} - {sr3.max():.2f} Hz")
    print(f"SR5 range: {sr5.min():.2f} - {sr5.max():.2f} Hz")
    print()

    # Calculate observed φ-errors
    print("Calculating observed φ-errors...")
    observed_errors = calculate_phi_errors(sr1, sr3, sr5)
    print()

    # Perform bootstrap shuffling
    print(f"Performing {n_iterations} bootstrap shuffles...")
    shuffled_results = shuffle_bootstrap(sr1, sr3, sr5, n_iterations)
    print()

    # Statistical testing
    print("Running statistical tests...")
    stats_results = perform_statistical_test(observed_errors, shuffled_results)
    print()

    # Print summary
    print("="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"Events: {stats_results['n_events']}")
    print(f"Bootstrap iterations: {stats_results['n_iterations']}")
    print()
    print(f"Observed composite error: {stats_results['obs_composite']:.4f}")
    print(f"Shuffled composite error: {stats_results['shuf_composite_mean']:.4f} ± {stats_results['shuf_composite_std']:.4f}")
    print()
    print(f"Observed percentile: {stats_results['percentile_composite']:.1f}th {'✓ PASS' if stats_results['passes_percentile'] else '✗ FAIL'}")
    print(f"P-value (composite): {stats_results['p_composite']:.4f} {'✓ PASS' if stats_results['passes_pvalue'] else '✗ FAIL'}")
    print()
    print("Individual ratios:")
    print(f"  f2/f1 (φ²): p = {stats_results['p_f2_f1']:.4f} {'✓' if stats_results['p_f2_f1'] < 0.05 else '✗'}")
    print(f"  f3/f1 (φ³): p = {stats_results['p_f3_f1']:.4f} {'✓' if stats_results['p_f3_f1'] < 0.05 else '✗'}")
    print(f"  f3/f2 (φ):  p = {stats_results['p_f3_f2']:.4f} {'✓' if stats_results['p_f3_f2'] < 0.05 else '✗'}")
    print()
    print(f"OVERALL: {'✓ PASS' if stats_results['overall_pass'] else '✗ FAIL'}")
    print("="*80)
    print()

    # Create visualizations
    print("Creating visualizations...")
    create_visualizations(observed_errors, shuffled_results, stats_results, output_dir)
    print()

    # Generate report
    print("Generating results summary...")
    generate_results_report(stats_results, output_dir)
    print()

    # Save raw data
    print("Saving raw data...")

    # Save observed errors
    obs_df = pd.DataFrame({
        'sr1': sr1,
        'sr3': sr3,
        'sr5': sr5,
        'error_f2_f1': observed_errors.error_f2_f1,
        'error_f3_f1': observed_errors.error_f3_f1,
        'error_f3_f2': observed_errors.error_f3_f2,
        'composite_error': observed_errors.composite_error
    })
    obs_df.to_csv(output_dir / 'observed_errors.csv', index=False)

    # Save shuffled error distribution (sample for file size)
    shuf_sample_size = min(1000, len(shuffled_results['composite_errors']))
    shuf_df = pd.DataFrame({
        'composite_error': shuffled_results['composite_errors'][:shuf_sample_size],
        'error_f2_f1': shuffled_results['errors_f2_f1'][:shuf_sample_size],
        'error_f3_f1': shuffled_results['errors_f3_f1'][:shuf_sample_size],
        'error_f3_f2': shuffled_results['errors_f3_f2'][:shuf_sample_size]
    })
    shuf_df.to_csv(output_dir / 'shuffled_errors_sample.csv', index=False)

    # Save statistics
    stats_df = pd.DataFrame([stats_results])
    stats_df.to_csv(output_dir / 'statistics.csv', index=False)

    print(f"Raw data saved to {output_dir}")
    print()

    print("="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"Results directory: {output_dir}")
    print()
    print("Output files:")
    print("  - phi_bootstrap_analysis.png/.pdf")
    print("  - results_summary.md")
    print("  - observed_errors.csv")
    print("  - shuffled_errors_sample.csv")
    print("  - statistics.csv")


if __name__ == '__main__':
    import sys

    # Allow command-line arguments
    csv_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV_PATH
    n_iterations = int(sys.argv[2]) if len(sys.argv) > 2 else N_BOOTSTRAP_ITERATIONS

    main(csv_path=csv_path, n_iterations=n_iterations)
