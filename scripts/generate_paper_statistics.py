#!/usr/bin/env python3
"""
Generate Paper Statistics for SIE Analysis

This script computes ALL statistics needed for the SIE paper from PAPER-3-sie-analysis.csv.
Outputs are formatted for direct inclusion in LaTeX.

Usage:
    python generate_paper_statistics.py
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr, spearmanr, ttest_ind, f_oneway
import warnings
warnings.filterwarnings('ignore')

# Constants
PHI = (1 + np.sqrt(5)) / 2  # Golden ratio ≈ 1.618034
CSV_PATH = "ALL-SCORES-CANON-3-sie-analysis.csv"

# Harmonic columns in the CSV
HARMONIC_COLS = ['sr1', 'sr1.5', 'sr2', 'sr2o', 'sr2.5', 'sr3', 'sr4', 'sr5', 'sr6']
HARMONIC_NAMES = ['SR1', 'SR1.5', 'SR2', 'SR2o', 'SR2.5', 'SR3', 'SR4', 'SR5', 'SR6']
HARMONIC_N_VALUES = [0.0, 0.5, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 3.5]  # φⁿ exponents

# Synchronization metric columns
Z_COLS = ['sr1_z_max', 'sr1.5_z_max', 'sr2_z_max', 'sr2o_z_max', 'sr2.5_z_max',
          'sr3_z_max', 'sr4_z_max', 'sr5_z_max', 'sr6_z_max']
MSC_COLS = ['msc_sr1_v', 'msc_ sr1.5_v', 'msc_sr2_v', 'msc_sr2o_v', 'msc_sr2.5_v',
            'msc_sr3_v', 'msc_sr4_v', 'msc_sr5_v', 'msc_sr6_v']
PLV_COLS = ['plv_sr1_pm5', 'plv_ sr1.5_pm5', 'plv_sr2_pm5', 'plv_sr2o_pm5', 'plv_sr2.5_pm5',
            'plv_sr3_pm5', 'plv_sr4_pm5', 'plv_sr5_pm5', 'plv_sr6_pm5']

def load_data():
    """Load and prepare the dataset."""
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} events from {CSV_PATH}")
    return df

def compute_dataset_overview(df):
    """Compute dataset overview statistics."""
    print("\n" + "="*80)
    print("SECTION A: DATASET OVERVIEW")
    print("="*80)

    n_events = len(df)
    n_sessions = df['session_name'].nunique()
    n_subjects = df['subject'].nunique()

    print(f"\nTotal Events: {n_events}")
    print(f"Total Sessions: {n_sessions}")
    print(f"Total Subjects: {n_subjects}")

    # Events per session
    events_per_session = df.groupby('session_name').size()
    print(f"\nEvents per Session: {events_per_session.mean():.1f} ± {events_per_session.std():.1f}")
    print(f"  Range: {events_per_session.min()} - {events_per_session.max()}")

    # Duration statistics
    if 'duration_s' in df.columns:
        dur = df['duration_s'].dropna()
        print(f"\nEvent Duration: {dur.mean():.1f} ± {dur.std():.1f} seconds")
        print(f"  Median: {dur.median():.1f} seconds")

    # By device
    print("\n--- BY DEVICE ---")
    device_counts = df['device'].value_counts()
    for device, count in device_counts.items():
        print(f"  {device}: {count} events ({100*count/n_events:.1f}%)")

    # By context
    print("\n--- BY CONTEXT ---")
    context_counts = df['context'].value_counts()
    for context, count in context_counts.items():
        print(f"  {context}: {count} events ({100*count/n_events:.1f}%)")

    # By dataset
    print("\n--- BY DATASET ---")
    dataset_counts = df['dataset'].value_counts()
    for dataset, count in dataset_counts.items():
        n_sess = df[df['dataset'] == dataset]['session_name'].nunique()
        n_subj = df[df['dataset'] == dataset]['subject'].nunique()
        eps = count / n_sess if n_sess > 0 else 0
        print(f"  {dataset}: {count} events, {n_sess} sessions, {n_subj} subjects, {eps:.1f} events/session")

    return {
        'n_events': n_events,
        'n_sessions': n_sessions,
        'n_subjects': n_subjects,
        'events_per_session_mean': events_per_session.mean(),
        'events_per_session_std': events_per_session.std(),
    }

def compute_harmonic_statistics(df):
    """Compute harmonic frequency statistics."""
    print("\n" + "="*80)
    print("SECTION B: HARMONIC FREQUENCY ARCHITECTURE")
    print("="*80)

    # Empirical f0 for predictions
    f0 = 7.6  # Empirical fundamental

    results = []
    print(f"\nUsing f₀ = {f0} Hz for φⁿ predictions")
    print("\n{:<8} {:<6} {:<12} {:<16} {:<10} {:<8} {:<6}".format(
        "Harmonic", "n", "Predicted", "Measured", "Error%", "CV%", "N"))
    print("-" * 70)

    for col, name, n in zip(HARMONIC_COLS, HARMONIC_NAMES, HARMONIC_N_VALUES):
        if col not in df.columns:
            continue

        vals = df[col].dropna()
        if len(vals) == 0:
            continue

        predicted = f0 * (PHI ** n)
        measured_mean = vals.mean()
        measured_std = vals.std()
        error_pct = 100 * (measured_mean - predicted) / predicted
        cv_pct = 100 * measured_std / measured_mean

        print(f"{name:<8} {n:<6.2f} {predicted:<12.2f} {measured_mean:.2f} ± {measured_std:.2f}   {error_pct:+.2f}%    {cv_pct:.2f}%   {len(vals)}")

        results.append({
            'harmonic': name,
            'n': n,
            'predicted': predicted,
            'measured_mean': measured_mean,
            'measured_std': measured_std,
            'error_pct': error_pct,
            'cv_pct': cv_pct,
            'N': len(vals)
        })

    # Mean absolute error
    errors = [abs(r['error_pct']) for r in results]
    print(f"\nMean Absolute Error: {np.mean(errors):.2f}%")
    print(f"Median Absolute Error: {np.median(errors):.2f}%")

    # Errors < 1%
    n_sub1 = sum(1 for e in errors if e < 1)
    print(f"Harmonics with |error| < 1%: {n_sub1} of {len(errors)}")

    return results

def compute_ratio_statistics(df):
    """Compute golden ratio precision statistics."""
    print("\n" + "="*80)
    print("SECTION C: GOLDEN RATIO PRECISION IN HARMONIC RATIOS")
    print("="*80)

    # Compute ratios
    df = df.copy()
    df['ratio_sr3_sr1'] = df['sr3'] / df['sr1']
    df['ratio_sr5_sr1'] = df['sr5'] / df['sr1']
    df['ratio_sr5_sr3'] = df['sr5'] / df['sr3']
    df['ratio_sr6_sr4'] = df['sr6'] / df['sr4']

    ratios = [
        ('SR3/SR1', 'ratio_sr3_sr1', PHI**2),
        ('SR5/SR1', 'ratio_sr5_sr1', PHI**3),
        ('SR5/SR3', 'ratio_sr5_sr3', PHI**1),
        ('SR6/SR4', 'ratio_sr6_sr4', PHI**1),
    ]

    print("\n{:<12} {:<10} {:<18} {:<10} {:<20}".format(
        "Ratio", "Predicted", "Measured", "Error%", "95% CI"))
    print("-" * 70)

    results = []
    for name, col, predicted in ratios:
        vals = df[col].dropna()
        vals = vals[np.isfinite(vals)]

        if len(vals) == 0:
            continue

        mean_val = vals.mean()
        std_val = vals.std()
        error_pct = 100 * (mean_val - predicted) / predicted

        # Bootstrap 95% CI
        n_boot = 10000
        boot_means = [vals.sample(len(vals), replace=True).mean() for _ in range(n_boot)]
        ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])

        print(f"{name:<12} {predicted:<10.4f} {mean_val:.4f} ± {std_val:.3f}   {error_pct:+.2f}%    [{ci_low:.3f}, {ci_high:.3f}]")

        results.append({
            'ratio': name,
            'predicted': predicted,
            'measured_mean': mean_val,
            'measured_std': std_val,
            'error_pct': error_pct,
            'ci_low': ci_low,
            'ci_high': ci_high,
            'N': len(vals)
        })

    # Mean absolute ratio error
    errors = [abs(r['error_pct']) for r in results]
    print(f"\nMean Absolute Ratio Error: {np.mean(errors):.2f}%")

    return results, df

def compute_independence_analysis(df):
    """Compute frequency independence analysis."""
    print("\n" + "="*80)
    print("SECTION D: INDEPENDENCE-CONVERGENCE ANALYSIS")
    print("="*80)

    print("\n--- HARMONIC FREQUENCY INDEPENDENCE ---")
    pairs = [('sr1', 'sr3'), ('sr1', 'sr5'), ('sr3', 'sr5')]

    for h1, h2 in pairs:
        mask = df[h1].notna() & df[h2].notna()
        x, y = df.loc[mask, h1], df.loc[mask, h2]
        r, p = pearsonr(x, y)
        print(f"  {h1.upper()} vs {h2.upper()}: r = {r:+.3f}, p = {p:.2e}")

    # Compensatory error mechanism
    print("\n--- COMPENSATORY ERROR MECHANISM ---")

    # Compute ratio errors
    df = df.copy()
    df['error_sr3_sr1'] = (df['sr3'] / df['sr1']) - PHI**2
    df['error_sr5_sr1'] = (df['sr5'] / df['sr1']) - PHI**3
    df['error_sr5_sr3'] = (df['sr5'] / df['sr3']) - PHI**1

    error_pairs = [
        ('error_sr3_sr1', 'error_sr5_sr3'),
        ('error_sr3_sr1', 'error_sr5_sr1'),
        ('error_sr5_sr1', 'error_sr5_sr3'),
    ]

    for e1, e2 in error_pairs:
        mask = df[e1].notna() & df[e2].notna() & np.isfinite(df[e1]) & np.isfinite(df[e2])
        x, y = df.loc[mask, e1], df.loc[mask, e2]
        r, p = pearsonr(x, y)
        print(f"  {e1} vs {e2}: r = {r:+.3f}, p = {p:.2e}")

    return df

def compute_shuffled_bootstrap(df, n_iterations=1000):
    """Compute shuffled bootstrap null control."""
    print("\n--- SHUFFLED BOOTSTRAP ANALYSIS ---")

    # Get valid triplets
    mask = df['sr1'].notna() & df['sr3'].notna() & df['sr5'].notna()
    sr1 = df.loc[mask, 'sr1'].values
    sr3 = df.loc[mask, 'sr3'].values
    sr5 = df.loc[mask, 'sr5'].values

    # Compute observed composite error
    def compute_composite_error(s1, s3, s5):
        r1 = s3 / s1  # Should be φ²
        r2 = s5 / s1  # Should be φ³
        r3 = s5 / s3  # Should be φ

        e1 = np.abs(r1 - PHI**2) / PHI**2
        e2 = np.abs(r2 - PHI**3) / PHI**3
        e3 = np.abs(r3 - PHI**1) / PHI**1

        return np.mean([e1, e2, e3])

    observed_errors = [compute_composite_error(sr1[i], sr3[i], sr5[i]) for i in range(len(sr1))]
    observed_mean = np.mean(observed_errors)

    # Shuffled null
    shuffled_means = []
    for _ in range(n_iterations):
        s1_shuf = np.random.permutation(sr1)
        s3_shuf = np.random.permutation(sr3)
        s5_shuf = np.random.permutation(sr5)

        shuf_errors = [compute_composite_error(s1_shuf[i], s3_shuf[i], s5_shuf[i]) for i in range(len(sr1))]
        shuffled_means.append(np.mean(shuf_errors))

    shuffled_mean = np.mean(shuffled_means)
    shuffled_std = np.std(shuffled_means)

    # Statistics
    p_value = np.mean([s <= observed_mean for s in shuffled_means])
    cohens_d = (shuffled_mean - observed_mean) / shuffled_std if shuffled_std > 0 else 0

    print(f"  Observed mean ratio error: {observed_mean:.4f}")
    print(f"  Shuffled null mean error: {shuffled_mean:.4f} ± {shuffled_std:.4f}")
    print(f"  p-value: {p_value:.4f}")
    print(f"  Cohen's d: {cohens_d:.2f}")

    return {
        'observed_mean': observed_mean,
        'shuffled_mean': shuffled_mean,
        'shuffled_std': shuffled_std,
        'p_value': p_value,
        'cohens_d': cohens_d
    }

def compute_random_triplet_null(df, n_triplets=10000):
    """Compare to random frequency triplets."""
    print("\n--- RANDOM FREQUENCY TRIPLET NULL CONTROL ---")

    # Get observed errors
    mask = df['sr1'].notna() & df['sr3'].notna() & df['sr5'].notna()
    sr1 = df.loc[mask, 'sr1'].values
    sr3 = df.loc[mask, 'sr3'].values
    sr5 = df.loc[mask, 'sr5'].values

    def compute_phi_error(f1, f2, f3):
        """Compute deviation from nearest φⁿ for triplet ratios."""
        r1 = f2 / f1
        r2 = f3 / f1
        r3 = f3 / f2

        # Find closest φⁿ for each ratio
        phi_powers = [PHI**n for n in range(-3, 5)]

        e1 = min(abs(r1 - p) / p for p in phi_powers)
        e2 = min(abs(r2 - p) / p for p in phi_powers)
        e3 = min(abs(r3 - p) / p for p in phi_powers)

        return np.mean([e1, e2, e3])

    # Observed error
    observed_errors = [compute_phi_error(sr1[i], sr3[i], sr5[i]) for i in range(len(sr1))]
    observed_mean = np.mean(observed_errors) * 100  # Convert to percentage

    # Random triplets (physiologically plausible ranges)
    random_errors = []
    for _ in range(n_triplets):
        f1 = np.random.uniform(5, 10)   # SR1 range
        f2 = np.random.uniform(17, 23)  # SR3 range
        f3 = np.random.uniform(28, 38)  # SR5 range
        random_errors.append(compute_phi_error(f1, f2, f3) * 100)

    random_mean = np.mean(random_errors)
    random_std = np.std(random_errors)

    # Statistics
    percentile = 100 * np.mean([r >= observed_mean for r in random_errors])
    cohens_d = (random_mean - observed_mean) / random_std if random_std > 0 else 0

    print(f"  Observed mean φ-error: {observed_mean:.2f}%")
    print(f"  Random triplet mean error: {random_mean:.1f}% ± {random_std:.1f}%")
    print(f"  Cohen's d: {cohens_d:.2f}")
    print(f"  Percentile rank: {100-percentile:.1f}% of random triplets are worse")
    print(f"  p-value (one-tailed): {percentile/100:.4f}")

    return {
        'observed_mean': observed_mean,
        'random_mean': random_mean,
        'random_std': random_std,
        'cohens_d': cohens_d,
        'percentile': percentile
    }

def compute_synchronization_statistics(df):
    """Compute synchronization metric statistics."""
    print("\n" + "="*80)
    print("SECTION E: SYNCHRONIZATION METRICS")
    print("="*80)

    # Z-score statistics
    print("\n--- POWER Z-SCORES BY HARMONIC ---")
    print("{:<8} {:<12} {:<10} {:<8} {:<8} {:<8} {:<8}".format(
        "Harmonic", "Mean z", "Median", "SD", ">2σ%", ">3σ%", ">5σ%"))
    print("-" * 70)

    for col, name in zip(Z_COLS, HARMONIC_NAMES):
        if col not in df.columns:
            continue
        vals = df[col].dropna()
        if len(vals) == 0:
            continue

        pct_2 = 100 * (vals > 2).mean()
        pct_3 = 100 * (vals > 3).mean()
        pct_5 = 100 * (vals > 5).mean()

        print(f"{name:<8} {vals.mean():.2f} ± {vals.std():.2f}  {vals.median():.2f}     {vals.std():.2f}   {pct_2:.1f}%   {pct_3:.1f}%   {pct_5:.1f}%")

    # MSC statistics
    print("\n--- MAGNITUDE SQUARED COHERENCE BY HARMONIC ---")
    print("{:<8} {:<12} {:<10}".format("Harmonic", "Mean MSC", ">0.5%"))
    print("-" * 40)

    for col, name in zip(MSC_COLS, HARMONIC_NAMES):
        if col not in df.columns:
            continue
        vals = df[col].dropna()
        if len(vals) == 0:
            continue

        pct_05 = 100 * (vals > 0.5).mean()
        print(f"{name:<8} {vals.mean():.3f} ± {vals.std():.3f}   {pct_05:.1f}%")

    # PLV statistics
    print("\n--- PHASE LOCKING VALUE BY HARMONIC ---")
    print("{:<8} {:<12} {:<10}".format("Harmonic", "Mean PLV", ">0.6%"))
    print("-" * 40)

    for col, name in zip(PLV_COLS, HARMONIC_NAMES):
        if col not in df.columns:
            continue
        vals = df[col].dropna()
        if len(vals) == 0:
            continue

        pct_06 = 100 * (vals > 0.6).mean()
        print(f"{name:<8} {vals.mean():.3f} ± {vals.std():.3f}   {pct_06:.1f}%")

    # Key threshold percentages for abstract
    if 'plv_sr1_pm5' in df.columns and 'msc_sr1_v' in df.columns:
        plv_sr1 = df['plv_sr1_pm5'].dropna()
        msc_sr1 = df['msc_sr1_v'].dropna()
        print(f"\n=== KEY STATISTICS FOR ABSTRACT ===")
        print(f"PLV > 0.6 at SR1: {100*(plv_sr1 > 0.6).mean():.1f}%")
        print(f"MSC > 0.5 at SR1: {100*(msc_sr1 > 0.5).mean():.1f}%")

def compute_quality_statistics(df):
    """Compute quality metric statistics."""
    print("\n" + "="*80)
    print("SECTION F: QUALITY METRICS")
    print("="*80)

    # SR-Score distribution
    if 'sr_score' in df.columns:
        score = df['sr_score'].dropna()
        print("\n--- SR-SCORE DISTRIBUTION ---")
        print(f"  Mean: {score.mean():.3f} ± {score.std():.3f}")
        print(f"  Median: {score.median():.3f}")
        print(f"  Range: {score.min():.3f} - {score.max():.3f}")
        print(f"  Quartiles: Q1={score.quantile(0.25):.3f}, Q2={score.quantile(0.5):.3f}, Q3={score.quantile(0.75):.3f}")

    # HSI distribution
    if 'HSI' in df.columns:
        hsi = df['HSI'].dropna()
        print("\n--- HSI DISTRIBUTION ---")
        print(f"  Mean: {hsi.mean():.2f} ± {hsi.std():.2f}")
        print(f"  Median: {hsi.median():.2f}")

    # φ-error distribution
    if 'sr3' in df.columns and 'sr1' in df.columns and 'sr5' in df.columns:
        df = df.copy()
        # Compute composite φ-error per event
        r1 = df['sr3'] / df['sr1']  # Should be φ²
        r2 = df['sr5'] / df['sr1']  # Should be φ³
        r3 = df['sr5'] / df['sr3']  # Should be φ

        e1 = 100 * np.abs(r1 - PHI**2) / PHI**2
        e2 = 100 * np.abs(r2 - PHI**3) / PHI**3
        e3 = 100 * np.abs(r3 - PHI**1) / PHI**1

        composite_error = (e1 + e2 + e3) / 3
        composite_error = composite_error.dropna()

        print("\n--- COMPOSITE φ-ERROR DISTRIBUTION ---")
        print(f"  Mean: {composite_error.mean():.2f}%")
        print(f"  Median: {composite_error.median():.2f}%")
        print(f"  SD: {composite_error.std():.2f}%")
        print(f"  Min: {composite_error.min():.2f}%")
        print(f"  Max: {composite_error.max():.2f}%")
        print(f"\n  Events < 1% error: {100*(composite_error < 1).mean():.1f}%")
        print(f"  Events < 3% error: {100*(composite_error < 3).mean():.1f}%")
        print(f"  Events < 5% error: {100*(composite_error < 5).mean():.1f}%")

def compute_device_anova(df):
    """Compute device independence ANOVA."""
    print("\n" + "="*80)
    print("SECTION G: DEVICE INDEPENDENCE (ANOVA)")
    print("="*80)

    devices = df['device'].dropna().unique()
    print(f"\nDevices: {devices}")

    for col, name in [('sr1', 'SR1'), ('sr3', 'SR3'), ('sr5', 'SR5')]:
        if col not in df.columns:
            continue

        groups = [df[df['device'] == d][col].dropna().values for d in devices]
        groups = [g for g in groups if len(g) > 0]

        if len(groups) < 2:
            continue

        f_stat, p_val = f_oneway(*groups)

        # Per-device means
        means = []
        for d in devices:
            vals = df[df['device'] == d][col].dropna()
            if len(vals) > 0:
                means.append(f"{d}: {vals.mean():.2f} ± {vals.std():.2f}")

        print(f"\n{name}:")
        print(f"  {', '.join(means)}")
        print(f"  F = {f_stat:.2f}, p = {p_val:.4f}")

def compute_context_anova(df):
    """Compute context independence ANOVA."""
    print("\n" + "="*80)
    print("SECTION H: CONTEXT INDEPENDENCE (ANOVA)")
    print("="*80)

    contexts = df['context'].dropna().unique()
    print(f"\nContexts: {contexts}")

    for col, name in [('sr1', 'SR1'), ('sr3', 'SR3'), ('sr5', 'SR5')]:
        if col not in df.columns:
            continue

        groups = [df[df['context'] == c][col].dropna().values for c in contexts]
        groups = [g for g in groups if len(g) > 0]

        if len(groups) < 2:
            continue

        f_stat, p_val = f_oneway(*groups)

        # Per-context means
        means = {c: df[df['context'] == c][col].dropna().mean() for c in contexts}

        print(f"\n{name}:")
        for c in contexts:
            val = means.get(c, np.nan)
            print(f"  {c}: {val:.2f}")
        print(f"  F = {f_stat:.2f}, p = {p_val:.4f}")

def compute_cross_frequency_coupling(df):
    """Compute cross-frequency coupling statistics."""
    print("\n" + "="*80)
    print("SECTION I: CROSS-FREQUENCY COUPLING")
    print("="*80)

    # Z-score correlations
    print("\n--- Z-SCORE CORRELATIONS ---")
    z_pairs = [
        ('sr1_z_max', 'sr3_z_max'),
        ('sr3_z_max', 'sr5_z_max'),
        ('sr5_z_max', 'sr6_z_max'),
        ('sr1_z_max', 'sr6_z_max'),
    ]

    for c1, c2 in z_pairs:
        if c1 in df.columns and c2 in df.columns:
            mask = df[c1].notna() & df[c2].notna()
            x, y = df.loc[mask, c1], df.loc[mask, c2]
            r, p = pearsonr(x, y)
            print(f"  {c1} vs {c2}: r = {r:.2f}")

    # MSC correlations
    print("\n--- MSC CORRELATION MATRIX ---")
    msc_cols_clean = [c for c in MSC_COLS if c in df.columns]
    if len(msc_cols_clean) >= 2:
        msc_data = df[msc_cols_clean].dropna()
        corr_matrix = msc_data.corr()
        print(corr_matrix.round(3))

def compute_quality_comparison(df):
    """Compare high vs low quality events."""
    print("\n" + "="*80)
    print("SECTION J: HIGH VS LOW QUALITY COMPARISON")
    print("="*80)

    if 'sr_score' not in df.columns:
        print("  sr_score column not found")
        return

    score = df['sr_score']
    q25 = score.quantile(0.25)
    q75 = score.quantile(0.75)

    low_q = df[score <= q25]
    high_q = df[score >= q75]

    print(f"\nLow quality (bottom 25%): {len(low_q)} events, SR-Score <= {q25:.3f}")
    print(f"High quality (top 25%): {len(high_q)} events, SR-Score >= {q75:.3f}")

    metrics = [
        ('msc_sr1_v', 'MSC (SR1)'),
        ('plv_sr1_pm5', 'PLV (SR1)'),
        ('sr1_z_max', 'z-score (SR1)'),
        ('HSI', 'HSI'),
    ]

    print("\n{:<15} {:<12} {:<12} {:<10} {:<10}".format(
        "Metric", "High-Q", "Low-Q", "t", "p"))
    print("-" * 60)

    for col, name in metrics:
        if col not in df.columns:
            continue

        high_vals = high_q[col].dropna()
        low_vals = low_q[col].dropna()

        if len(high_vals) < 2 or len(low_vals) < 2:
            continue

        t_stat, p_val = ttest_ind(high_vals, low_vals)

        print(f"{name:<15} {high_vals.mean():.3f}        {low_vals.mean():.3f}        {t_stat:.2f}      {p_val:.2e}")

def compute_subject_level_analysis(df):
    """Compute subject-level statistics."""
    print("\n" + "="*80)
    print("SECTION K: SUBJECT-LEVEL ANALYSIS")
    print("="*80)

    n_subjects = df['subject'].nunique()
    print(f"\nN = {n_subjects} subjects")

    # Subject-level ratio means
    subject_ratios = df.groupby('subject').agg({
        'sr1': 'mean',
        'sr3': 'mean',
        'sr5': 'mean',
    }).dropna()

    subject_ratios['ratio_sr3_sr1'] = subject_ratios['sr3'] / subject_ratios['sr1']
    subject_ratios['ratio_sr5_sr3'] = subject_ratios['sr5'] / subject_ratios['sr3']

    r1 = subject_ratios['ratio_sr3_sr1']
    r2 = subject_ratios['ratio_sr5_sr3']

    print(f"\nSubject-level SR3/SR1: {r1.mean():.4f} ± {r1.std():.3f} (predicted: {PHI**2:.4f}, error: {100*(r1.mean() - PHI**2)/PHI**2:+.2f}%)")
    print(f"Subject-level SR5/SR3: {r2.mean():.4f} ± {r2.std():.3f} (predicted: {PHI:.4f}, error: {100*(r2.mean() - PHI)/PHI:+.2f}%)")


def main():
    """Main execution."""
    print("="*80)
    print("SIE PAPER STATISTICS GENERATOR")
    print("="*80)

    # Load data
    df = load_data()

    # Run all analyses
    overview = compute_dataset_overview(df)
    harmonic_stats = compute_harmonic_statistics(df)
    ratio_stats, df = compute_ratio_statistics(df)
    df = compute_independence_analysis(df)
    shuffled = compute_shuffled_bootstrap(df, n_iterations=1000)
    random_triplet = compute_random_triplet_null(df, n_triplets=10000)
    compute_synchronization_statistics(df)
    compute_quality_statistics(df)
    compute_device_anova(df)
    compute_context_anova(df)
    compute_cross_frequency_coupling(df)
    compute_quality_comparison(df)
    compute_subject_level_analysis(df)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

    return df

if __name__ == "__main__":
    main()
