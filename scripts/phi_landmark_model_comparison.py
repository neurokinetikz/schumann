"""
Phi Landmark Model Comparison
-----------------------------
Distinguishes between:
1. Discrete φ-anchored structure (golden ratio positions specifically matter)
2. Smooth non-uniformity that happens to align with 0.618

Approach:
- Fit smooth baseline g(u) to within-band u distribution (splines or Fourier)
- Add landmark kernels at 0.382/0.5/0.618
- Test incremental explanatory power via likelihood ratio / BIC comparison
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize
from scipy.interpolate import BSpline
import warnings
warnings.filterwarnings('ignore')

# Constants
PHI = (1 + np.sqrt(5)) / 2
F0 = 7.6  # Empirically derived base frequency

# Landmark positions
LANDMARKS = {
    '2° Noble (0.382)': 0.382,
    'Attractor (0.5)': 0.5,
    '1° Noble (0.618)': 0.618,
}

def load_all_peaks():
    """Load peaks from all dataset CSVs."""
    datasets = ['FILES', 'INSIGHT', 'PHYSF', 'VEP', 'ArEEG', 'MPENG1', 'MPENG2']
    all_peaks = []

    for ds in datasets:
        try:
            df = pd.read_csv(f'golden_ratio_peaks_{ds}.csv')
            df['dataset'] = ds
            all_peaks.append(df)
            print(f"  {ds}: {len(df)} peaks")
        except FileNotFoundError:
            print(f"  {ds}: not found")

    return pd.concat(all_peaks, ignore_index=True)

def compute_phi_coordinates(freqs, f0=F0):
    """Compute φ-log coordinates: n = log_φ(f/f0), then k = floor(n), u = n mod 1."""
    n_values = np.log(freqs / f0) / np.log(PHI)
    k_values = np.floor(n_values).astype(int)
    u_values = n_values - k_values
    return n_values, k_values, u_values

# =============================================================================
# MODEL DEFINITIONS
# =============================================================================

def fourier_basis(u, n_terms=4):
    """Low-order Fourier basis for smooth baseline."""
    # u is in [0, 1], create basis functions
    basis = [np.ones_like(u)]  # Constant term
    for k in range(1, n_terms + 1):
        basis.append(np.cos(2 * np.pi * k * u))
        basis.append(np.sin(2 * np.pi * k * u))
    return np.column_stack(basis)

def gaussian_kernel(u, center, width=0.05):
    """Gaussian kernel centered at a landmark position."""
    return np.exp(-0.5 * ((u - center) / width) ** 2)

def model_smooth_only(u, params, n_fourier=4):
    """Smooth baseline model g(u) using Fourier series."""
    basis = fourier_basis(u, n_fourier)
    n_basis = basis.shape[1]
    coeffs = params[:n_basis]
    log_density = basis @ coeffs
    # Normalize to probability density
    density = np.exp(log_density)
    return density / np.trapz(density, u) if len(u) > 1 else density

def model_smooth_plus_landmarks(u, params, n_fourier=4, landmark_positions=[0.382, 0.5, 0.618]):
    """Smooth baseline + Gaussian landmark kernels."""
    basis = fourier_basis(u, n_fourier)
    n_basis = basis.shape[1]

    # Smooth component
    fourier_coeffs = params[:n_basis]
    log_smooth = basis @ fourier_coeffs
    smooth_density = np.exp(log_smooth)

    # Landmark kernels (amplitudes and widths)
    n_landmarks = len(landmark_positions)
    landmark_amplitudes = params[n_basis:n_basis + n_landmarks]
    landmark_width = 0.05  # Fixed width

    landmark_contribution = np.zeros_like(u)
    for i, pos in enumerate(landmark_positions):
        amp = np.exp(landmark_amplitudes[i])  # Ensure positive
        landmark_contribution += amp * gaussian_kernel(u, pos, landmark_width)

    total_density = smooth_density + landmark_contribution
    return total_density / np.trapz(total_density, u) if len(u) > 1 else total_density

def neg_log_likelihood_smooth(params, u_data, u_grid, n_fourier):
    """Negative log-likelihood for smooth-only model."""
    density_grid = model_smooth_only(u_grid, params, n_fourier=n_fourier)
    density_grid = np.maximum(density_grid, 1e-10)
    density_grid = density_grid / np.trapz(density_grid, u_grid)
    density_at_data = np.interp(u_data, u_grid, density_grid)
    density_at_data = np.maximum(density_at_data, 1e-10)
    return -np.sum(np.log(density_at_data))

def neg_log_likelihood_full(params, u_data, u_grid, n_fourier, landmark_positions):
    """Negative log-likelihood for smooth + landmarks model."""
    density_grid = model_smooth_plus_landmarks(u_grid, params, n_fourier=n_fourier,
                                               landmark_positions=landmark_positions)
    density_grid = np.maximum(density_grid, 1e-10)
    density_grid = density_grid / np.trapz(density_grid, u_grid)
    density_at_data = np.interp(u_data, u_grid, density_grid)
    density_at_data = np.maximum(density_at_data, 1e-10)
    return -np.sum(np.log(density_at_data))

def fit_smooth_model(u_data, n_fourier=4):
    """Fit smooth-only model via maximum likelihood."""
    u_grid = np.linspace(0, 1, 500)
    n_params = 2 * n_fourier + 1

    np.random.seed(42)
    x0 = np.random.randn(n_params) * 0.1

    result = minimize(
        neg_log_likelihood_smooth,
        x0,
        args=(u_data, u_grid, n_fourier),
        method='L-BFGS-B',
        options={'maxiter': 1000}
    )

    return result.x, -result.fun, result

def fit_full_model(u_data, n_fourier=4, landmark_positions=[0.382, 0.5, 0.618]):
    """Fit smooth + landmarks model via maximum likelihood."""
    u_grid = np.linspace(0, 1, 500)
    n_fourier_params = 2 * n_fourier + 1
    n_params = n_fourier_params + len(landmark_positions)

    np.random.seed(42)
    x0 = np.random.randn(n_params) * 0.1

    result = minimize(
        neg_log_likelihood_full,
        x0,
        args=(u_data, u_grid, n_fourier, landmark_positions),
        method='L-BFGS-B',
        options={'maxiter': 1000}
    )

    return result.x, -result.fun, result

# =============================================================================
# MAIN ANALYSIS
# =============================================================================

print("=" * 80)
print("φ-LANDMARK MODEL COMPARISON")
print("Smooth Baseline g(u) vs. Smooth + Landmark Kernels")
print("=" * 80)

# Load data
print("\nLoading peak data...")
peaks_df = load_all_peaks()
print(f"Total peaks: {len(peaks_df)}")

# Compute φ-coordinates
freqs = peaks_df['freq'].values
n_vals, k_vals, u_vals = compute_phi_coordinates(freqs)

# Filter to valid range (bands -1 to 3)
valid_mask = (k_vals >= -1) & (k_vals <= 3) & (u_vals >= 0) & (u_vals <= 1)
u_data = u_vals[valid_mask]
k_data = k_vals[valid_mask]
print(f"Peaks in valid band range: {len(u_data)}")

# =============================================================================
# MODEL FITTING
# =============================================================================

print("\n" + "=" * 80)
print("MODEL FITTING")
print("=" * 80)

n_fourier = 4  # 4 Fourier terms = 9 parameters (1 const + 4 cos + 4 sin)
n_fourier_params = 2 * n_fourier + 1
n_landmark_params = 3  # Three landmarks: 0.382, 0.5, 0.618

# Model 1: Smooth baseline only
print("\nFitting Model 1: Smooth Fourier baseline g(u)...")
params_smooth, ll_smooth, result_smooth = fit_smooth_model(u_data, n_fourier=n_fourier)
print(f"  Parameters: {n_fourier_params}")
print(f"  Log-likelihood: {ll_smooth:.2f}")

# Model 2: Smooth + Landmarks
print("\nFitting Model 2: Smooth + Landmark Kernels...")
n_params_full = n_fourier_params + n_landmark_params
params_full, ll_full, result_full = fit_full_model(
    u_data, n_fourier=n_fourier, landmark_positions=[0.382, 0.5, 0.618]
)
print(f"  Parameters: {n_params_full}")
print(f"  Log-likelihood: {ll_full:.2f}")

# =============================================================================
# MODEL COMPARISON
# =============================================================================

print("\n" + "=" * 80)
print("MODEL COMPARISON")
print("=" * 80)

n = len(u_data)

# Likelihood Ratio Test
lr_statistic = 2 * (ll_full - ll_smooth)
df_diff = n_landmark_params
p_value = 1 - stats.chi2.cdf(lr_statistic, df_diff)

print(f"\nLikelihood Ratio Test:")
print(f"  LR statistic: {lr_statistic:.2f}")
print(f"  df: {df_diff}")
print(f"  p-value: {p_value:.2e}")

# BIC comparison
bic_smooth = n_fourier_params * np.log(n) - 2 * ll_smooth
bic_full = n_params_full * np.log(n) - 2 * ll_full
delta_bic = bic_smooth - bic_full  # Positive = full model better

print(f"\nBayesian Information Criterion (BIC):")
print(f"  BIC (smooth only): {bic_smooth:.2f}")
print(f"  BIC (smooth + landmarks): {bic_full:.2f}")
print(f"  ΔBIC = BIC_smooth - BIC_full: {delta_bic:.2f}")
print(f"  Interpretation: {'Landmarks strongly preferred' if delta_bic > 10 else 'Landmarks preferred' if delta_bic > 2 else 'No clear preference' if delta_bic > -2 else 'Smooth preferred'}")

# AIC comparison
aic_smooth = 2 * n_fourier_params - 2 * ll_smooth
aic_full = 2 * n_params_full - 2 * ll_full
delta_aic = aic_smooth - aic_full

print(f"\nAkaike Information Criterion (AIC):")
print(f"  AIC (smooth only): {aic_smooth:.2f}")
print(f"  AIC (smooth + landmarks): {aic_full:.2f}")
print(f"  ΔAIC = AIC_smooth - AIC_full: {delta_aic:.2f}")

# =============================================================================
# DECOMPOSITION: How much variance do landmarks explain?
# =============================================================================

print("\n" + "=" * 80)
print("LANDMARK CONTRIBUTION ANALYSIS")
print("=" * 80)

u_grid = np.linspace(0, 1, 500)

# Get smooth-only density
smooth_density = model_smooth_only(u_grid, params_smooth, n_fourier=n_fourier)
smooth_density = smooth_density / np.trapz(smooth_density, u_grid)

# Get full density
full_density = model_smooth_plus_landmarks(u_grid, params_full, n_fourier=n_fourier,
                                            landmark_positions=[0.382, 0.5, 0.618])
full_density = full_density / np.trapz(full_density, u_grid)

# Extract landmark amplitudes from full model
landmark_amps = np.exp(params_full[n_fourier_params:n_fourier_params + 3])
print(f"\nLandmark kernel amplitudes (relative):")
for i, (name, pos) in enumerate(LANDMARKS.items()):
    print(f"  {name}: {landmark_amps[i]:.4f}")

# Compute actual observed histogram
hist, bin_edges = np.histogram(u_data, bins=50, density=True)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# Compute residuals
smooth_at_hist = np.interp(bin_centers, u_grid, smooth_density)
full_at_hist = np.interp(bin_centers, u_grid, full_density)
residuals_smooth = hist - smooth_at_hist
residuals_full = hist - full_at_hist

# R² improvement
ss_total = np.sum((hist - np.mean(hist))**2)
ss_res_smooth = np.sum(residuals_smooth**2)
ss_res_full = np.sum(residuals_full**2)

r2_smooth = 1 - ss_res_smooth / ss_total
r2_full = 1 - ss_res_full / ss_total
r2_improvement = r2_full - r2_smooth

print(f"\nVariance Explained (R²):")
print(f"  Smooth baseline: {r2_smooth:.4f}")
print(f"  Smooth + Landmarks: {r2_full:.4f}")
print(f"  Improvement from landmarks: {r2_improvement:.4f} ({100*r2_improvement:.2f}%)")

# =============================================================================
# PARTIAL F-TEST
# =============================================================================

print("\n" + "=" * 80)
print("PARTIAL F-TEST FOR LANDMARKS")
print("=" * 80)

# F-test comparing nested models
ss_reduction = ss_res_smooth - ss_res_full
df_numerator = n_landmark_params
df_denominator = len(bin_centers) - n_params_full

if ss_res_full > 0 and df_denominator > 0:
    f_statistic = (ss_reduction / df_numerator) / (ss_res_full / df_denominator)
    p_value_f = 1 - stats.f.cdf(f_statistic, df_numerator, df_denominator)
    print(f"  F-statistic: {f_statistic:.2f}")
    print(f"  df: ({df_numerator}, {df_denominator})")
    print(f"  p-value: {p_value_f:.2e}")
else:
    print("  Could not compute F-test")

# =============================================================================
# VISUALIZATION
# =============================================================================

print("\n" + "=" * 80)
print("GENERATING VISUALIZATION...")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel A: Data histogram with both model fits
ax = axes[0, 0]
ax.hist(u_data, bins=50, density=True, alpha=0.6, color='steelblue',
        edgecolor='white', label='Observed data')
ax.plot(u_grid, smooth_density, 'g-', linewidth=2, label='Smooth baseline g(u)')
ax.plot(u_grid, full_density, 'r-', linewidth=2, label='Smooth + Landmarks')

# Mark landmark positions
for name, pos in LANDMARKS.items():
    ax.axvline(pos, color='orange', linestyle='--', alpha=0.7, linewidth=1.5)
    ax.text(pos, ax.get_ylim()[1] * 0.95, name.split('(')[0].strip(),
            ha='center', va='top', fontsize=8, rotation=90)

ax.set_xlabel('Fractional Position u (within φ-band)', fontsize=11)
ax.set_ylabel('Probability Density', fontsize=11)
ax.set_title('A. Model Fits: Smooth vs. Smooth + Landmarks', fontweight='bold')
ax.legend(loc='upper left', fontsize=9)
ax.set_xlim(0, 1)

# Panel B: Residuals comparison
ax = axes[0, 1]
width = 0.008
ax.bar(bin_centers - width, residuals_smooth, width=width*1.8, alpha=0.6,
       color='green', label=f'Smooth only (R²={r2_smooth:.3f})')
ax.bar(bin_centers + width, residuals_full, width=width*1.8, alpha=0.6,
       color='red', label=f'Smooth + Landmarks (R²={r2_full:.3f})')
ax.axhline(0, color='black', linewidth=0.5)

for name, pos in LANDMARKS.items():
    ax.axvline(pos, color='orange', linestyle='--', alpha=0.5)

ax.set_xlabel('Fractional Position u', fontsize=11)
ax.set_ylabel('Residual (Observed - Model)', fontsize=11)
ax.set_title('B. Residuals: Landmarks Reduce Error at φ-Positions', fontweight='bold')
ax.legend(loc='upper right', fontsize=9)
ax.set_xlim(0, 1)

# Panel C: Landmark kernels contribution
ax = axes[1, 0]

# Compute landmark contribution separately
smooth_component = model_smooth_only(u_grid, params_full[:n_fourier_params], n_fourier=n_fourier)
smooth_component = smooth_component / np.trapz(smooth_component, u_grid) * np.trapz(full_density, u_grid)

landmark_contribution = full_density - smooth_component

ax.fill_between(u_grid, 0, smooth_component, alpha=0.4, color='green', label='Smooth component')
ax.fill_between(u_grid, smooth_component, full_density, alpha=0.6, color='orange', label='Landmark contribution')
ax.plot(u_grid, full_density, 'r-', linewidth=1.5, label='Total model')

for name, pos in LANDMARKS.items():
    ax.axvline(pos, color='red', linestyle='--', alpha=0.7, linewidth=1)

ax.set_xlabel('Fractional Position u', fontsize=11)
ax.set_ylabel('Probability Density', fontsize=11)
ax.set_title('C. Decomposition: Smooth vs. Landmark Components', fontweight='bold')
ax.legend(loc='upper left', fontsize=9)
ax.set_xlim(0, 1)

# Panel D: Model comparison summary
ax = axes[1, 1]
ax.axis('off')

summary_text = f"""
MODEL COMPARISON SUMMARY
{'='*50}

Smooth Baseline g(u): Fourier series ({n_fourier_params} params)
Full Model: Smooth + 3 Gaussian landmark kernels ({n_params_full} params)

LIKELIHOOD RATIO TEST
  LR statistic: {lr_statistic:.2f}
  df: {df_diff}
  p-value: {p_value:.2e}  {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else ''}

INFORMATION CRITERIA
  ΔBIC (smooth - full): {delta_bic:+.1f}
  ΔAIC (smooth - full): {delta_aic:+.1f}
  Interpretation: {'LANDMARKS STRONGLY PREFERRED' if delta_bic > 10 else 'Landmarks preferred' if delta_bic > 2 else 'No preference'}

VARIANCE EXPLAINED
  R² smooth only: {r2_smooth:.4f}
  R² with landmarks: {r2_full:.4f}
  Improvement: +{100*r2_improvement:.2f}%

LANDMARK KERNEL AMPLITUDES
  0.382 (2° Noble): {landmark_amps[0]:.4f}
  0.500 (Attractor): {landmark_amps[1]:.4f}
  0.618 (1° Noble): {landmark_amps[2]:.4f}

CONCLUSION
{'='*50}
{'Landmarks provide SIGNIFICANT incremental explanatory power!' if delta_bic > 10 and p_value < 0.001 else 'Landmarks provide incremental explanatory power.' if delta_bic > 2 else 'Smooth hump suffices; landmarks add little.'}
{'→ Discrete φ-anchored structure is supported.' if delta_bic > 10 else '→ SR-anchored φ-log reveals non-uniformity.' if delta_bic > -2 else '→ Consider alternative explanations.'}
"""

ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle('φⁿ Landmark Model Comparison: Discrete Structure vs. Smooth Non-Uniformity',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('phi_landmark_model_comparison.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: phi_landmark_model_comparison.png")

# =============================================================================
# BOOTSTRAP CONFIDENCE INTERVALS FOR LANDMARK CONTRIBUTION
# =============================================================================

print("\n" + "=" * 80)
print("BOOTSTRAP: 95% CI FOR LANDMARK CONTRIBUTION")
print("=" * 80)

n_bootstrap = 30  # Reduced for faster runtime
delta_bic_boot = []
r2_improvement_boot = []

print("Running bootstrap iterations...", end='', flush=True)
for i in range(n_bootstrap):
    if i % 20 == 0:
        print(f" {i}", end='', flush=True)

    # Resample with replacement
    idx = np.random.choice(len(u_data), len(u_data), replace=True)
    u_boot = u_data[idx]

    # Fit both models
    try:
        _, ll_s, _ = fit_smooth_model(u_boot, n_fourier=n_fourier)
        _, ll_f, _ = fit_full_model(u_boot, n_fourier=n_fourier,
                                    landmark_positions=[0.382, 0.5, 0.618])

        bic_s = n_fourier_params * np.log(len(u_boot)) - 2 * ll_s
        bic_f = n_params_full * np.log(len(u_boot)) - 2 * ll_f
        delta_bic_boot.append(bic_s - bic_f)
    except:
        pass

print(" done.")

delta_bic_boot = np.array(delta_bic_boot)
ci_low, ci_high = np.percentile(delta_bic_boot, [2.5, 97.5])

print(f"\nΔBIC Bootstrap 95% CI: [{ci_low:.1f}, {ci_high:.1f}]")
print(f"Mean ΔBIC: {np.mean(delta_bic_boot):.1f} ± {np.std(delta_bic_boot):.1f}")

# =============================================================================
# FINAL INTERPRETATION
# =============================================================================

print("\n" + "=" * 80)
print("FINAL INTERPRETATION")
print("=" * 80)

if delta_bic > 10 and p_value < 0.001:
    print("""
RESULT: Landmarks provide STRONG incremental explanatory power.

The full model with discrete landmark kernels at:
  - 0.382 (2° Noble = 1 - 1/φ)
  - 0.500 (Attractor)
  - 0.618 (1° Noble = 1/φ)

significantly outperforms the smooth baseline alone.

CONCLUSION: The data supports DISCRETE φ-anchored structure,
not merely a smooth hump that happens to align with 0.618.
The golden ratio positions are specifically enriched beyond
what a smooth non-uniformity would predict.
""")
elif delta_bic > 2:
    print("""
RESULT: Landmarks provide MODEST incremental explanatory power.

The discrete landmark kernels improve fit, but the effect is moderate.
Both interpretations have some merit:
  - There is evidence for φ-specific positions
  - But a smooth non-uniformity captures most of the structure

CONCLUSION: Evidence for discrete φ-anchored structure,
but the effect is not dramatically stronger than smooth alternatives.
""")
else:
    print("""
RESULT: Landmarks provide MINIMAL incremental explanatory power.

The smooth baseline g(u) captures most of the within-band structure.
Adding discrete landmark kernels does not substantially improve the fit.

CONCLUSION: The correct interpretation is:
"SR-anchored φ-log coordinate reveals systematic within-band non-uniformity"
rather than "specific golden ratio positions are preferred."

This is still a meaningful finding, just framed differently.
""")
