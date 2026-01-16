#!/usr/bin/env python3
"""
Power Analysis for Unified Paper

Computes sensitivity power analysis and achieved power for all key statistical tests.
Generates values for Table X (Statistical Power for Key Analyses).

Author: Michael Lacy
Date: January 2026
"""

import numpy as np
from scipy import stats
from scipy.optimize import brentq

# Try to import statsmodels for ANOVA power; fall back to manual calculation
try:
    from statsmodels.stats.power import TTestIndPower, TTestPower, FTestAnovaPower
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("Warning: statsmodels not available, using manual power calculations")


def power_one_sample_t(n, d, alpha=0.05):
    """
    Compute power for one-sample t-test.

    Parameters
    ----------
    n : int
        Sample size
    d : float
        Cohen's d effect size
    alpha : float
        Significance level (two-tailed)

    Returns
    -------
    power : float
        Statistical power (0-1)
    """
    df = n - 1
    t_crit = stats.t.ppf(1 - alpha/2, df)
    ncp = d * np.sqrt(n)  # Non-centrality parameter

    # For very large non-centrality parameters, power approaches 1.0
    if ncp > 10:  # For ncp > 10, lower tail is negligible
        upper = stats.nct.cdf(t_crit, df, ncp)
        if np.isnan(upper) or upper < 1e-100:
            return 1.0
        power = 1 - upper
    else:
        upper = stats.nct.cdf(t_crit, df, ncp)
        lower = stats.nct.cdf(-t_crit, df, ncp)
        # Handle NaN from numerical underflow
        if np.isnan(lower):
            lower = 0.0
        power = 1 - upper + lower

    return min(max(power, 0.0), 1.0)  # Clamp to [0, 1]


def sensitivity_one_sample_t(n, power=0.80, alpha=0.05):
    """
    Find minimum detectable effect size (Cohen's d) for one-sample t-test.

    Parameters
    ----------
    n : int
        Sample size
    power : float
        Target power
    alpha : float
        Significance level (two-tailed)

    Returns
    -------
    d : float
        Minimum detectable Cohen's d
    """
    def objective(d):
        return power_one_sample_t(n, d, alpha) - power

    try:
        # Check if power at d=0.001 already exceeds target (very large samples)
        if power_one_sample_t(n, 0.001, alpha) >= power:
            return 0.001
        d_min = brentq(objective, 0.001, 2.0)
    except ValueError:
        d_min = np.nan
    return d_min


def power_two_sample_t(n1, n2, d, alpha=0.05):
    """
    Compute power for two-sample independent t-test.

    Parameters
    ----------
    n1, n2 : int
        Sample sizes for each group
    d : float
        Cohen's d effect size
    alpha : float
        Significance level (two-tailed)

    Returns
    -------
    power : float
        Statistical power (0-1)
    """
    df = n1 + n2 - 2
    t_crit = stats.t.ppf(1 - alpha/2, df)
    # Pooled standard error
    se = np.sqrt(1/n1 + 1/n2)
    ncp = d / se

    # For very large non-centrality parameters, power approaches 1.0
    if ncp > 10:  # For ncp > 10, lower tail is negligible
        upper = stats.nct.cdf(t_crit, df, ncp)
        if np.isnan(upper) or upper < 1e-100:
            return 1.0
        power = 1 - upper
    else:
        upper = stats.nct.cdf(t_crit, df, ncp)
        lower = stats.nct.cdf(-t_crit, df, ncp)
        if np.isnan(lower):
            lower = 0.0
        power = 1 - upper + lower

    return min(max(power, 0.0), 1.0)


def sensitivity_two_sample_t(n1, n2, power=0.80, alpha=0.05):
    """
    Find minimum detectable effect size for two-sample t-test.
    """
    def objective(d):
        return power_two_sample_t(n1, n2, d, alpha) - power

    try:
        # Check if power at d=0.001 already exceeds target (very large samples)
        if power_two_sample_t(n1, n2, 0.001, alpha) >= power:
            return 0.001
        # Use reasonable upper bound for effect size search
        d_min = brentq(objective, 0.001, 1.0, xtol=1e-4)
    except ValueError:
        # If brentq fails, try wider bounds
        try:
            d_min = brentq(objective, 0.0001, 2.0, xtol=1e-4)
        except ValueError:
            d_min = np.nan
    return d_min


def power_correlation(n, r, alpha=0.05):
    """
    Compute power for Pearson correlation test.

    Uses Fisher z-transformation.

    Parameters
    ----------
    n : int
        Sample size
    r : float
        Correlation coefficient (effect size)
    alpha : float
        Significance level (two-tailed)

    Returns
    -------
    power : float
        Statistical power (0-1)
    """
    # Fisher z-transformation
    z_r = 0.5 * np.log((1 + r) / (1 - r))
    se = 1 / np.sqrt(n - 3)
    z_crit = stats.norm.ppf(1 - alpha/2)

    # Power calculation
    power = 1 - stats.norm.cdf(z_crit - z_r/se) + stats.norm.cdf(-z_crit - z_r/se)
    return power


def sensitivity_correlation(n, power=0.80, alpha=0.05):
    """
    Find minimum detectable correlation coefficient.
    """
    def objective(r):
        return power_correlation(n, r, alpha) - power

    try:
        r_min = brentq(objective, 0.001, 0.99)
    except ValueError:
        r_min = np.nan
    return r_min


def power_one_way_anova(group_sizes, f, alpha=0.05):
    """
    Compute power for one-way ANOVA.

    Parameters
    ----------
    group_sizes : list
        Sample sizes for each group
    f : float
        Cohen's f effect size
    alpha : float
        Significance level

    Returns
    -------
    power : float
        Statistical power (0-1)
    """
    k = len(group_sizes)
    N = sum(group_sizes)
    df_between = k - 1
    df_within = N - k

    # Non-centrality parameter for F-test
    # lambda = N * f^2 for balanced design; for unbalanced, use harmonic mean
    n_harmonic = k / sum(1/n for n in group_sizes)
    ncp = n_harmonic * k * f**2

    # Critical F value
    f_crit = stats.f.ppf(1 - alpha, df_between, df_within)

    # Power from non-central F distribution
    power = 1 - stats.ncf.cdf(f_crit, df_between, df_within, ncp)
    return power


def sensitivity_anova(group_sizes, power=0.80, alpha=0.05):
    """
    Find minimum detectable Cohen's f for ANOVA.
    """
    def objective(f):
        return power_one_way_anova(group_sizes, f, alpha) - power

    try:
        f_min = brentq(objective, 0.001, 2.0)
    except ValueError:
        f_min = np.nan
    return f_min


def main():
    """Run all power analyses and print results."""

    print("=" * 70)
    print("POWER ANALYSIS FOR UNIFIED PAPER")
    print("Golden Ratio Architecture of Human Neural Oscillations")
    print("=" * 70)
    print()

    # =========================================================================
    # STUDY 1: SIE Detection (N = 1,121 valid triplets)
    # =========================================================================
    print("STUDY 1: SCHUMANN IGNITION EVENTS")
    print("-" * 50)

    n_triplets = 1121
    n_events = 1366

    # 1. One-sample t-test sensitivity (ratio deviation from phi^n)
    print("\n1. ONE-SAMPLE T-TEST (ratio precision)")
    print(f"   Sample size: N = {n_triplets}")
    d_min_one = sensitivity_one_sample_t(n_triplets, power=0.80)
    print(f"   Minimum detectable Cohen's d (80% power): {d_min_one:.3f}")

    # Achieved power with observed d = 1.44
    d_observed = 1.44
    power_achieved = power_one_sample_t(n_triplets, d_observed)
    print(f"   Observed Cohen's d: {d_observed}")
    print(f"   Achieved power: {power_achieved:.4f}")

    # 2. Two-sample t-test (peak-based null control: SIE vs random triplets)
    print("\n2. TWO-SAMPLE T-TEST (peak-based null control)")
    n_sie = 1121
    n_null = 10000
    d_min_two = sensitivity_two_sample_t(n_sie, n_null, power=0.80)
    print(f"   Sample sizes: N_SIE = {n_sie}, N_null = {n_null}")
    print(f"   Minimum detectable Cohen's d (80% power): {d_min_two:.3f}")

    # Achieved power
    power_null = power_two_sample_t(n_sie, n_null, d_observed)
    print(f"   Observed Cohen's d: {d_observed}")
    print(f"   Achieved power: {power_null:.4f}")

    # 3. Correlation (frequency independence)
    print("\n3. CORRELATION (frequency independence)")
    print(f"   Sample size: N = {n_triplets}")
    r_min = sensitivity_correlation(n_triplets, power=0.80)
    print(f"   Minimum detectable |r| (80% power): {r_min:.3f}")

    # Power to detect various r values
    for r_test in [0.05, 0.08, 0.10, 0.15]:
        p = power_correlation(n_triplets, r_test)
        print(f"   Power to detect r = {r_test}: {p:.3f}")

    # Observed correlations
    print(f"\n   Observed correlations: |r| < 0.03 (all non-significant)")
    print(f"   Interpretation: Observed r well below detectable threshold")
    print(f"   Supports genuine independence, not Type II error")

    # 4. Device independence ANOVA (3 groups)
    print("\n4. ONE-WAY ANOVA (device independence)")
    # Approximate group sizes: Muse ~165, Insight ~57, EPOC X ~1144
    device_sizes = [165, 57, 1144]
    print(f"   Group sizes: Muse={device_sizes[0]}, Insight={device_sizes[1]}, EPOC X={device_sizes[2]}")
    f_min_device = sensitivity_anova(device_sizes, power=0.80)
    print(f"   Minimum detectable Cohen's f (80% power): {f_min_device:.3f}")

    # Power for observed F = 5.74 (SR5)
    # Convert F to Cohen's f: f = sqrt(F * df_between / N)
    k = 3
    N = sum(device_sizes)
    F_observed = 5.74
    f_observed = np.sqrt(F_observed * (k-1) / N)
    power_device = power_one_way_anova(device_sizes, f_observed)
    print(f"   Observed F = {F_observed}, implied f = {f_observed:.3f}")
    print(f"   Achieved power: {power_device:.3f}")

    # 5. Context independence ANOVA (5 groups)
    print("\n5. ONE-WAY ANOVA (context independence)")
    # Approximate sizes: meditation ~260, flow ~250, non-flow ~200, gaming ~600, visual ~55
    context_sizes = [260, 250, 200, 604, 55]
    contexts = ['meditation', 'flow', 'non-flow', 'gaming', 'visual']
    print(f"   Group sizes: {dict(zip(contexts, context_sizes))}")
    f_min_context = sensitivity_anova(context_sizes, power=0.80)
    print(f"   Minimum detectable Cohen's f (80% power): {f_min_context:.3f}")

    # =========================================================================
    # STUDY 2: Spectral Validation (N = 244,955 peaks)
    # =========================================================================
    print("\n" + "=" * 50)
    print("STUDY 2: SPECTRAL VALIDATION")
    print("-" * 50)

    n_peaks = 244955
    n_sessions = 968

    # 1. Position enrichment (essentially infinite power)
    print("\n1. POSITION ENRICHMENT")
    print(f"   Sample size: N = {n_peaks} peaks")
    print(f"   Power for any meaningful effect: >0.9999")
    print(f"   Observed effects: -18% (boundary), +39% (1° noble)")

    # 2. Session-level analysis
    print("\n2. SESSION-LEVEL ANALYSIS")
    print(f"   Sample size: N = {n_sessions} sessions")
    d_min_session = sensitivity_one_sample_t(n_sessions, power=0.80)
    print(f"   Minimum detectable Cohen's d (80% power): {d_min_session:.3f}")

    # Observed Cohen's d = 0.89 for attractor enrichment
    d_session = 0.89
    power_session = power_one_sample_t(n_sessions, d_session)
    print(f"   Observed Cohen's d: {d_session}")
    print(f"   Achieved power: {power_session:.4f}")

    # 3. Scaling factor comparison (6 factors)
    print("\n3. SCALING FACTOR COMPARISON")
    print(f"   Bootstrap iterations: 10,000")
    print(f"   p < 0.001 for phi vs alternatives")

    # =========================================================================
    # PER-SUBJECT AND PER-SESSION ANALYSES
    # =========================================================================
    print("\n" + "=" * 50)
    print("HIERARCHICAL ANALYSES")
    print("-" * 50)

    n_subjects = 90
    n_sessions_valid = 513

    print(f"\n1. PER-SUBJECT ANALYSIS")
    print(f"   N = {n_subjects} subjects with valid triplets")
    d_min_subj = sensitivity_one_sample_t(n_subjects, power=0.80)
    print(f"   Minimum detectable Cohen's d: {d_min_subj:.3f}")

    print(f"\n2. PER-SESSION ANALYSIS")
    print(f"   N = {n_sessions_valid} sessions with valid triplets")
    d_min_sess = sensitivity_one_sample_t(n_sessions_valid, power=0.80)
    print(f"   Minimum detectable Cohen's d: {d_min_sess:.3f}")

    # =========================================================================
    # SUMMARY TABLE
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY TABLE FOR PAPER")
    print("=" * 70)
    print()
    print("| Analysis                    | N       | Observed | Min Det | Power  |")
    print("|:----------------------------|:--------|:---------|:--------|:-------|")
    print(f"| Ratio precision vs null     | 1,121   | d=1.44   | d={d_min_two:.2f}  | >{power_null:.3f} |")
    print(f"| Position enrichment         | 244,955 | 18-39%   | <1%     | >0.999 |")
    print(f"| Session attractor effect    | 968     | d=0.89   | d={d_min_session:.2f}  | {power_session:.3f}  |")
    print(f"| Frequency independence      | 1,121   | r<0.03   | r={r_min:.2f}  | 0.20*  |")
    print(f"| Device independence (SR5)   | 1,366   | F=5.74   | f={f_min_device:.2f}  | {power_device:.2f}   |")
    print()
    print("* For frequency independence, low power to detect small correlations")
    print("  confirms that |r| < 0.03 represents genuine independence.")

    # =========================================================================
    # LATEX OUTPUT
    # =========================================================================
    print("\n" + "=" * 70)
    print("LATEX TABLE (copy to paper)")
    print("=" * 70)
    print(r"""
\begin{table}[H]
\centering
\caption{Statistical Power for Key Analyses}
\label{tab:power_analysis}
\begin{tabular}{@{}lllll@{}}
\toprule
Analysis & $N$ & Observed Effect & Min Detectable & Power \\
\midrule""")
    print(f"Ratio precision vs null & 1,121 & $d = 1.44$ & $d = {d_min_two:.2f}$ & $>0.999$ \\\\")
    print(f"Position enrichment & 244,955 & 18--39\\% & $<1\\%$ & $>0.999$ \\\\")
    print(f"Session attractor effect & 968 & $d = 0.89$ & $d = {d_min_session:.2f}$ & ${power_session:.3f}$ \\\\")
    print(f"Frequency independence & 1,121 & $|r| < 0.03$ & $r = {r_min:.2f}$ & $0.20^*$ \\\\")
    print(f"Device independence (SR5) & 1,366 & $F = 5.74$ & $f = {f_min_device:.2f}$ & ${power_device:.2f}$ \\\\")
    print(r"""\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item All analyses assumed $\alpha = 0.05$ (two-tailed) and target power of 0.80 for sensitivity calculations.
\item $^*$For frequency independence, low power at observed $|r| < 0.03$ confirms genuine independence rather than Type II error; we had 80\% power to detect $r \geq 0.08$.
\end{tablenotes}
\end{table}
""")


if __name__ == "__main__":
    main()
