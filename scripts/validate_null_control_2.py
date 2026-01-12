#!/usr/bin/env python3
"""
COMPREHENSIVE VALIDATION OF NULL CONTROL 2: DISTRIBUTIONAL NULL MODEL

This script triple-checks every aspect of the distributional null control test:
1. Verify what sr1, sr3, sr5 columns represent
2. Check that they match the detection pipeline exactly
3. Validate the statistical methodology
4. Identify any potential issues or biases

CRITICAL QUESTIONS TO ANSWER:
=============================
1. What ARE sr1, sr3, sr5? (FOOOF peaks? Search window centers? Something else?)
2. Does independent sampling ACTUALLY break the coupling? (Yes - mathematically proven)
3. Is the statistical test correct? (One-sided vs two-sided)
4. Are we testing the right hypothesis?
5. Does the test replicate the detection pipeline logic?
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from typing import Tuple, Dict

# Golden ratio constants
PHI = 1.618033988749895
PHI_SQ = PHI ** 2      # ~2.618
PHI_CUBED = PHI ** 3   # ~4.236

DATA_FILE = 'data/SHUFFLED-DATA-BOOTSTRAP.csv'

print("="*80)
print("NULL CONTROL 2: DISTRIBUTIONAL NULL MODEL - COMPREHENSIVE VALIDATION")
print("="*80)

# ============================================================================
# STEP 1: Load and inspect the data
# ============================================================================

print("\n" + "="*80)
print("STEP 1: DATA INSPECTION")
print("="*80)

data = pd.read_csv(DATA_FILE)
n_events = len(data)

print(f"\nLoaded {n_events} SIE events")
print(f"\nAvailable columns ({len(data.columns)}):")
for i, col in enumerate(data.columns, 1):
    print(f"  {i:2d}. {col}")

# Check for required columns
required_cols = ['sr1', 'sr3', 'sr5', 'sr3/sr1', 'sr5/sr1', 'sr5/sr3']
missing = [col for col in required_cols if col not in data.columns]

if missing:
    print(f"\n❌ ERROR: Missing columns: {missing}")
    print("Cannot proceed with validation.")
    exit(1)

print(f"\n✅ All required columns present: {required_cols}")

# ============================================================================
# STEP 2: Understand what sr1, sr3, sr5 represent
# ============================================================================

print("\n" + "="*80)
print("STEP 2: UNDERSTANDING sr1, sr3, sr5")
print("="*80)

print("\nDescriptive statistics:")
print("\nSR1 (fundamental):")
print(data['sr1'].describe())

print("\nSR3 (3rd harmonic):")
print(data['sr3'].describe())

print("\nSR5 (5th harmonic):")
print(data['sr5'].describe())

# Check against theoretical values
theoretical_sr1 = 7.83
theoretical_sr3 = theoretical_sr1 * 2.63  # ~20.59
theoretical_sr5 = theoretical_sr1 * 4.19  # ~32.80

print(f"\nComparison to theoretical Schumann Resonance frequencies:")
print(f"  SR1: Observed mean={data['sr1'].mean():.2f} Hz, Theoretical={theoretical_sr1:.2f} Hz")
print(f"  SR3: Observed mean={data['sr3'].mean():.2f} Hz, Theoretical={theoretical_sr3:.2f} Hz")
print(f"  SR5: Observed mean={data['sr5'].mean():.2f} Hz, Theoretical={theoretical_sr5:.2f} Hz")

# Check if these are DETECTED peaks or FIXED search centers
sr1_std = data['sr1'].std()
if sr1_std < 0.01:
    print(f"\n⚠️  WARNING: SR1 std={sr1_std:.4f} Hz is very small.")
    print("    These might be FIXED search centers, not detected peaks!")
else:
    print(f"\n✅ SR1 std={sr1_std:.2f} Hz shows variation - likely detected peaks (FOOOF)")

# ============================================================================
# STEP 3: Verify the pre-computed ratios match the frequencies
# ============================================================================

print("\n" + "="*80)
print("STEP 3: RATIO VERIFICATION")
print("="*80)

# Compute ratios from frequencies
computed_ratio_31 = data['sr3'] / data['sr1']
computed_ratio_51 = data['sr5'] / data['sr1']
computed_ratio_53 = data['sr5'] / data['sr3']

# Compare to pre-computed ratios
diff_31 = np.abs(computed_ratio_31 - data['sr3/sr1'])
diff_51 = np.abs(computed_ratio_51 - data['sr5/sr1'])
diff_53 = np.abs(computed_ratio_53 - data['sr5/sr3'])

print(f"\nVerifying pre-computed ratios match sr1, sr3, sr5:")
print(f"  sr3/sr1: Max difference = {diff_31.max():.6f}")
print(f"  sr5/sr1: Max difference = {diff_51.max():.6f}")
print(f"  sr5/sr3: Max difference = {diff_53.max():.6f}")

if diff_31.max() < 1e-6 and diff_51.max() < 1e-6 and diff_53.max() < 1e-6:
    print(f"\n✅ Pre-computed ratios match computed ratios (within rounding error)")
else:
    print(f"\n⚠️  WARNING: Ratios don't match! There may be a data inconsistency.")

# ============================================================================
# STEP 4: Analyze observed φ-convergence
# ============================================================================

print("\n" + "="*80)
print("STEP 4: OBSERVED φ-CONVERGENCE")
print("="*80)

# Use pre-computed ratios
obs_ratio_31 = data['sr3/sr1'].values
obs_ratio_51 = data['sr5/sr1'].values
obs_ratio_53 = data['sr5/sr3'].values

# Calculate errors from golden ratio
obs_error_31 = np.abs(obs_ratio_31 - PHI_SQ) / PHI_SQ * 100
obs_error_53 = np.abs(obs_ratio_53 - PHI) / PHI * 100
obs_error_51 = np.abs(obs_ratio_51 - PHI_CUBED) / PHI_CUBED * 100
obs_composite = (obs_error_31 + obs_error_53 + obs_error_51) / 3

print(f"\nObserved φ-convergence (n={n_events}):")
print(f"  φ₃₁ (SR3/SR1): Mean={obs_ratio_31.mean():.3f}, Expected={PHI_SQ:.3f}, Error={obs_error_31.mean():.2f}%")
print(f"  φ₅₃ (SR5/SR3): Mean={obs_ratio_53.mean():.3f}, Expected={PHI:.3f}, Error={obs_error_53.mean():.2f}%")
print(f"  φ₅₁ (SR5/SR1): Mean={obs_ratio_51.mean():.3f}, Expected={PHI_CUBED:.3f}, Error={obs_error_51.mean():.2f}%")
print(f"  Composite Error: {obs_composite.mean():.2f} ± {obs_composite.std():.2f}%")

# ============================================================================
# STEP 5: FIT DISTRIBUTIONS (critically evaluate assumptions)
# ============================================================================

print("\n" + "="*80)
print("STEP 5: DISTRIBUTION FITTING")
print("="*80)

# Fit normal distributions
sr1_params = stats.norm.fit(data['sr1'])
sr3_params = stats.norm.fit(data['sr3'])
sr5_params = stats.norm.fit(data['sr5'])

print(f"\nFitted Normal distributions:")
print(f"  SR1: μ={sr1_params[0]:.4f} Hz, σ={sr1_params[1]:.4f} Hz (CV={sr1_params[1]/sr1_params[0]:.3f})")
print(f"  SR3: μ={sr3_params[0]:.4f} Hz, σ={sr3_params[1]:.4f} Hz (CV={sr3_params[1]/sr3_params[0]:.3f})")
print(f"  SR5: μ={sr5_params[0]:.4f} Hz, σ={sr5_params[1]:.4f} Hz (CV={sr5_params[1]/sr5_params[0]:.3f})")

# Test normality assumption (Shapiro-Wilk test)
print(f"\nNormality tests (Shapiro-Wilk):")
_, p_sr1 = stats.shapiro(data['sr1'])
_, p_sr3 = stats.shapiro(data['sr3'])
_, p_sr5 = stats.shapiro(data['sr5'])

print(f"  SR1: p={p_sr1:.4f} {'(Normal ✅)' if p_sr1 > 0.05 else '(Non-normal ⚠️)'}")
print(f"  SR3: p={p_sr3:.4f} {'(Normal ✅)' if p_sr3 > 0.05 else '(Non-normal ⚠️)'}")
print(f"  SR5: p={p_sr5:.4f} {'(Normal ✅)' if p_sr5 > 0.05 else '(Non-normal ⚠️)'}")

if p_sr1 < 0.05 or p_sr3 < 0.05 or p_sr5 < 0.05:
    print(f"\n⚠️  WARNING: Frequencies are not normally distributed!")
    print(f"    Using normal distribution for synthetic sampling may be inappropriate.")
    print(f"    Consider using empirical bootstrap instead.")

# Check for correlations (should exist in observed data, will be destroyed in synthetic)
print(f"\nObserved correlations (Pearson r):")
corr_31 = stats.pearsonr(data['sr1'], data['sr3'])[0]
corr_51 = stats.pearsonr(data['sr1'], data['sr5'])[0]
corr_53 = stats.pearsonr(data['sr3'], data['sr5'])[0]

print(f"  SR1 vs SR3: r={corr_31:.3f}")
print(f"  SR1 vs SR5: r={corr_51:.3f}")
print(f"  SR3 vs SR5: r={corr_53:.3f}")

if abs(corr_31) > 0.3 or abs(corr_51) > 0.3 or abs(corr_53) > 0.3:
    print(f"\n✅ Significant correlations detected - independent sampling WILL break coupling")
else:
    print(f"\n⚠️  WARNING: Weak correlations - test may have low power")

# ============================================================================
# STEP 6: GENERATE SYNTHETIC DATA
# ============================================================================

print("\n" + "="*80)
print("STEP 6: SYNTHETIC DATA GENERATION")
print("="*80)

N_SYNTHETIC = 10000
print(f"\nGenerating {N_SYNTHETIC} synthetic samples via INDEPENDENT sampling...")

# CRITICAL: Sample INDEPENDENTLY to break coupling
synth_sr1 = stats.norm.rvs(*sr1_params, size=N_SYNTHETIC, random_state=42)
synth_sr3 = stats.norm.rvs(*sr3_params, size=N_SYNTHETIC, random_state=43)
synth_sr5 = stats.norm.rvs(*sr5_params, size=N_SYNTHETIC, random_state=44)

# Verify independence
synth_corr_31 = stats.pearsonr(synth_sr1, synth_sr3)[0]
synth_corr_51 = stats.pearsonr(synth_sr1, synth_sr5)[0]
synth_corr_53 = stats.pearsonr(synth_sr3, synth_sr5)[0]

print(f"\nSynthetic correlations (should be ~0):")
print(f"  SR1 vs SR3: r={synth_corr_31:.4f}")
print(f"  SR1 vs SR5: r={synth_corr_51:.4f}")
print(f"  SR3 vs SR5: r={synth_corr_53:.4f}")

if abs(synth_corr_31) < 0.05 and abs(synth_corr_51) < 0.05 and abs(synth_corr_53) < 0.05:
    print(f"\n✅ Correlations successfully destroyed - independent sampling works")
else:
    print(f"\n❌ ERROR: Synthetic data shows correlation! This shouldn't happen.")

# Calculate synthetic ratios and errors
synth_ratio_31 = synth_sr3 / synth_sr1
synth_ratio_53 = synth_sr5 / synth_sr3
synth_ratio_51 = synth_sr5 / synth_sr1

synth_error_31 = np.abs(synth_ratio_31 - PHI_SQ) / PHI_SQ * 100
synth_error_53 = np.abs(synth_ratio_53 - PHI) / PHI * 100
synth_error_51 = np.abs(synth_ratio_51 - PHI_CUBED) / PHI_CUBED * 100
synth_composite = (synth_error_31 + synth_error_53 + synth_error_51) / 3

print(f"\nSynthetic φ-convergence:")
print(f"  φ₃₁ (SR3/SR1): Mean={synth_ratio_31.mean():.3f}, Error={synth_error_31.mean():.2f}%")
print(f"  φ₅₃ (SR5/SR3): Mean={synth_ratio_53.mean():.3f}, Error={synth_error_53.mean():.2f}%")
print(f"  φ₅₁ (SR5/SR1): Mean={synth_ratio_51.mean():.3f}, Error={synth_error_51.mean():.2f}%")
print(f"  Composite Error: {synth_composite.mean():.2f} ± {synth_composite.std():.2f}%")

# ============================================================================
# STEP 7: STATISTICAL COMPARISON (evaluate test choice)
# ============================================================================

print("\n" + "="*80)
print("STEP 7: STATISTICAL COMPARISON")
print("="*80)

# CRITICAL: What hypothesis are we testing?
# H0: Observed and synthetic have same φ-error distribution
# H1_two_sided: Distributions differ (obs ≠ synth)
# H1_less: Observed has LOWER error (obs < synth) [better convergence]
# H1_greater: Observed has HIGHER error (obs > synth) [worse convergence]

print(f"\nComposite φ-Error Summary:")
print(f"  Observed:  {obs_composite.mean():.3f} ± {obs_composite.std():.3f}% (n={len(obs_composite)})")
print(f"  Synthetic: {synth_composite.mean():.3f} ± {synth_composite.std():.3f}% (n={len(synth_composite)})")
print(f"  Difference: {obs_composite.mean() - synth_composite.mean():.3f}%")

# Test 1: Two-sided (are they different?)
u_two, p_two = stats.mannwhitneyu(obs_composite, synth_composite, alternative='two-sided')
print(f"\nMann-Whitney U (two-sided):")
print(f"  U={u_two:.2f}, p={p_two:.6f}")
print(f"  Interpretation: {'Different distributions' if p_two < 0.05 else 'Similar distributions'}")

# Test 2: One-sided less (is observed BETTER?)
u_less, p_less = stats.mannwhitneyu(obs_composite, synth_composite, alternative='less')
print(f"\nMann-Whitney U (one-sided: obs < synth):")
print(f"  U={u_less:.2f}, p={p_less:.6f}")
print(f"  Interpretation: {'Observed has BETTER φ-convergence' if p_less < 0.05 else 'No better convergence'}")

# Test 3: One-sided greater (is observed WORSE?)
u_greater, p_greater = stats.mannwhitneyu(obs_composite, synth_composite, alternative='greater')
print(f"\nMann-Whitney U (one-sided: obs > synth):")
print(f"  U={u_greater:.2f}, p={p_greater:.6f}")
print(f"  Interpretation: {'Observed has WORSE φ-convergence' if p_greater < 0.05 else 'No worse convergence'}")

# Effect size
pooled_std = np.sqrt((np.var(obs_composite) + np.var(synth_composite)) / 2)
cohens_d = (obs_composite.mean() - synth_composite.mean()) / pooled_std

print(f"\nEffect Size (Cohen's d): {cohens_d:.3f}")
if abs(cohens_d) < 0.2:
    print(f"  Interpretation: Negligible effect")
elif abs(cohens_d) < 0.5:
    print(f"  Interpretation: Small effect")
elif abs(cohens_d) < 0.8:
    print(f"  Interpretation: Medium effect")
else:
    print(f"  Interpretation: Large effect")

# ============================================================================
# STEP 8: CRITICAL EVALUATION
# ============================================================================

print("\n" + "="*80)
print("STEP 8: CRITICAL EVALUATION")
print("="*80)

print(f"\n🔍 CURRENT IMPLEMENTATION REVIEW:")
print(f"   • Uses Mann-Whitney U with alternative='less'")
print(f"   • Tests if observed < synthetic (better convergence)")
print(f"   • Found: p_less={p_less:.6f}, Cohen's d={cohens_d:.3f}")

print(f"\n🤔 ISSUES TO CONSIDER:")

# Issue 1: Test direction
if p_two > 0.05:
    print(f"\n1. ⚠️  NO SIGNIFICANT DIFFERENCE (two-sided p={p_two:.3f})")
    print(f"   → Observed and synthetic show SIMILAR φ-convergence")
    print(f"   → This suggests φ-ratios ARE distributional artifacts")
    print(f"   → Test should FAIL (φ-convergence not special)")
else:
    if p_less < 0.05 and cohens_d < -0.5:
        print(f"\n1. ✅ OBSERVED BETTER THAN SYNTHETIC")
        print(f"   → Observed shows significantly BETTER φ-convergence")
        print(f"   → This suggests φ-ratios are NOT artifacts")
        print(f"   → Test should PASS (φ-convergence is special)")
    elif p_greater < 0.05 and cohens_d > 0.5:
        print(f"\n1. ❌ OBSERVED WORSE THAN SYNTHETIC")
        print(f"   → This is unexpected and problematic")
        print(f"   → Suggests measurement or methodological issue")
    else:
        print(f"\n1. ⚠️  SMALL EFFECT (|d|={abs(cohens_d):.3f})")
        print(f"   → Statistically significant but practically small")

# Issue 2: Distribution assumptions
if p_sr1 < 0.05 or p_sr3 < 0.05 or p_sr5 < 0.05:
    print(f"\n2. ⚠️  NORMALITY ASSUMPTION VIOLATED")
    print(f"   → Using normal distributions may not accurately model data")
    print(f"   → Consider empirical bootstrap or kernel density estimation")

# Issue 3: Sample size imbalance
if N_SYNTHETIC != len(obs_composite):
    print(f"\n3. ℹ️  SAMPLE SIZE IMBALANCE")
    print(f"   → Observed: {len(obs_composite)}, Synthetic: {N_SYNTHETIC}")
    print(f"   → Mann-Whitney U handles this correctly")

print(f"\n" + "="*80)
print(f"FINAL VERDICT")
print(f"="*80)

if p_two >= 0.05:
    verdict = "❌ FAIL"
    interpretation = "φ-convergence appears to be a DISTRIBUTIONAL ARTIFACT"
    recommendation = "The observed golden ratios can emerge from independent sampling"
elif p_less < 0.05 and cohens_d < -0.5:
    verdict = "✅ PASS"
    interpretation = "φ-convergence is GENUINE (not an artifact)"
    recommendation = "The observed ratios require coupled harmonics"
else:
    verdict = "⚠️  MARGINAL"
    interpretation = "Results are inconclusive"
    recommendation = "Further investigation needed"

print(f"\n{verdict}")
print(f"\n{interpretation}")
print(f"{recommendation}")

# ============================================================================
# STEP 9: VISUALIZATION
# ============================================================================

print(f"\n" + "="*80)
print(f"STEP 9: GENERATING VALIDATION PLOTS")
print(f"="*80)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Null Control 2 Validation: Distributional Null Model', fontsize=14, fontweight='bold')

# Panel 1: Histogram comparison
ax = axes[0, 0]
ax.hist(obs_composite, bins=30, alpha=0.6, label='Observed', color='blue', density=True)
ax.hist(synth_composite, bins=50, alpha=0.6, label='Synthetic', color='red', density=True)
ax.axvline(obs_composite.mean(), color='blue', linestyle='--', linewidth=2)
ax.axvline(synth_composite.mean(), color='red', linestyle='--', linewidth=2)
ax.set_xlabel('Composite φ-Error (%)')
ax.set_ylabel('Density')
ax.set_title('Distribution Comparison')
ax.legend()
ax.grid(True, alpha=0.3)

# Panel 2: CDF comparison
ax = axes[0, 1]
obs_sorted = np.sort(obs_composite)
synth_sorted = np.sort(synth_composite)
obs_cdf = np.arange(1, len(obs_sorted) + 1) / len(obs_sorted)
synth_cdf = np.arange(1, len(synth_sorted) + 1) / len(synth_sorted)
ax.plot(obs_sorted, obs_cdf, label='Observed', color='blue', linewidth=2)
ax.plot(synth_sorted, synth_cdf, label='Synthetic', color='red', linewidth=2)
ax.set_xlabel('Composite φ-Error (%)')
ax.set_ylabel('Cumulative Probability')
ax.set_title('CDF Comparison')
ax.legend()
ax.grid(True, alpha=0.3)

# Panel 3: Q-Q plot
ax = axes[0, 2]
stats.probplot(obs_composite, dist="norm", plot=ax)
ax.set_title('Q-Q Plot (Observed Data)')
ax.grid(True, alpha=0.3)

# Panel 4: Correlation comparison
ax = axes[1, 0]
corr_data = [
    [corr_31, corr_51, corr_53],
    [synth_corr_31, synth_corr_51, synth_corr_53]
]
x = np.arange(3)
width = 0.35
ax.bar(x - width/2, corr_data[0], width, label='Observed', color='blue', alpha=0.7)
ax.bar(x + width/2, corr_data[1], width, label='Synthetic', color='red', alpha=0.7)
ax.set_ylabel('Pearson r')
ax.set_title('Correlations (Should be destroyed)')
ax.set_xticks(x)
ax.set_xticklabels(['SR1-SR3', 'SR1-SR5', 'SR3-SR5'])
ax.legend()
ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
ax.grid(True, alpha=0.3, axis='y')

# Panel 5: Individual ratio errors
ax = axes[1, 1]
ratio_names = ['φ₃₁', 'φ₅₃', 'φ₅₁']
obs_errors = [obs_error_31.mean(), obs_error_53.mean(), obs_error_51.mean()]
synth_errors = [synth_error_31.mean(), synth_error_53.mean(), synth_error_51.mean()]
x = np.arange(3)
ax.bar(x - width/2, obs_errors, width, label='Observed', color='blue', alpha=0.7)
ax.bar(x + width/2, synth_errors, width, label='Synthetic', color='red', alpha=0.7)
ax.set_ylabel('Mean φ-Error (%)')
ax.set_title('Individual Ratio Errors')
ax.set_xticks(x)
ax.set_xticklabels(ratio_names)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Panel 6: Statistical summary
ax = axes[1, 2]
ax.axis('off')
summary_text = f"""
Statistical Summary

Composite φ-Error:
  Observed: {obs_composite.mean():.2f} ± {obs_composite.std():.2f}%
  Synthetic: {synth_composite.mean():.2f} ± {synth_composite.std():.2f}%

Mann-Whitney U Tests:
  Two-sided: p = {p_two:.4f}
  One-sided (less): p = {p_less:.4f}
  One-sided (greater): p = {p_greater:.4f}

Effect Size:
  Cohen's d = {cohens_d:.3f}

Verdict: {verdict}
"""
ax.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center',
        fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('test_output/null_control_2_validation.png', dpi=150, bbox_inches='tight')
print(f"\n✅ Saved: test_output/null_control_2_validation.png")
plt.close()

print(f"\n" + "="*80)
print(f"VALIDATION COMPLETE")
print(f"="*80)
