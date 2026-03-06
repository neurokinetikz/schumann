"""
Continuous Phi-Lattice Compliance Metrics
==========================================

Replaces binary window-based enrichment with Gaussian kernel density
estimates at lattice positions. Same structural score formula:

    SS = -E(boundary) + E(attractor) + mean(E(nobles))

but with continuous inputs instead of binary counts.

Also provides mean minimum distance (MMD) as a parameter-free metric.
"""

import os
import sys
import numpy as np
import pandas as pd
from functools import lru_cache

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from ratio_specificity import lattice_coordinate, PHI, NAMED_RATIOS
from structural_phi_specificity import natural_positions

# Try importing make_phi_bands from lemon_utils (may not always be available)
try:
    from lemon_utils import make_phi_bands
except ImportError:
    def make_phi_bands(f0=8.5, freq_ceil=45.0):
        return {
            'theta':     (f0 / PHI, f0),
            'alpha':     (f0, f0 * PHI),
            'beta_low':  (f0 * PHI, f0 * PHI ** 2),
            'beta_high': (f0 * PHI ** 2, f0 * PHI ** 3),
            'gamma':     (f0 * PHI ** 3, freq_ceil),
        }

# Default sigma for kernel density
SIGMA_DEFAULT = 0.035

# Sweep range
SIGMA_SWEEP_VALUES = np.arange(0.015, 0.085, 0.005)


# ============================================================================
# CORE: Circular distance and kernel density
# ============================================================================

def circular_distance(u, v):
    """Circular distance on [0, 1). Handles wrap-around at boundary.

    d(u, v) = min(|u - v|, 1 - |u - v|)

    Parameters
    ----------
    u, v : array_like or scalar
        Points on the unit interval [0, 1).

    Returns
    -------
    ndarray
        Circular distances, always in [0, 0.5].
    """
    d = np.abs(np.asarray(u, dtype=float) - np.asarray(v, dtype=float))
    return np.minimum(d, 1.0 - d)


def kernel_density_at_positions(u_coords, positions, sigma):
    """Gaussian kernel density at each lattice position.

    For position p_j:
        D(p_j) = (1/N) * sum_i exp(-d_circ(u_i, p_j)^2 / (2*sigma^2))

    Parameters
    ----------
    u_coords : ndarray
        Lattice coordinates in [0, 1).
    positions : dict
        {name: offset} for each lattice position.
    sigma : float
        Kernel bandwidth.

    Returns
    -------
    dict : {name: density_value}
    """
    u = np.asarray(u_coords, dtype=float)
    n = len(u)
    if n == 0:
        return {name: 0.0 for name in positions}

    two_sigma_sq = 2.0 * sigma ** 2
    densities = {}
    for name, pos in positions.items():
        d = circular_distance(u, pos)
        densities[name] = np.mean(np.exp(-d ** 2 / two_sigma_sq))
    return densities


@lru_cache(maxsize=128)
def null_kernel_density(sigma, n_grid=100_000):
    """Expected kernel density at any position under uniform distribution.

    D_null(sigma) = integral_0^1 exp(-d_circ(u, 0)^2 / (2*sigma^2)) du

    By circular symmetry, this is the same for all positions.
    Computed via dense numerical integration and cached.
    """
    u_grid = np.linspace(0, 1, n_grid, endpoint=False)
    d = circular_distance(u_grid, 0.0)
    return np.mean(np.exp(-d ** 2 / (2.0 * sigma ** 2)))


def kernel_enrichment(u_coords, positions, sigma):
    """Enrichment (%) at each position using Gaussian kernel.

    E_kernel(p_j) = (D(p_j) / D_null(sigma) - 1) * 100

    Under uniformity: all enrichments = 0%.

    Parameters
    ----------
    u_coords : ndarray
        Lattice coordinates in [0, 1).
    positions : dict
        {name: offset} for each lattice position.
    sigma : float
        Kernel bandwidth.

    Returns
    -------
    dict : {name: enrichment_percent}
    """
    densities = kernel_density_at_positions(u_coords, positions, sigma)
    null_d = null_kernel_density(sigma)

    if null_d == 0:
        return {name: 0.0 for name in positions}

    return {name: (d / null_d - 1.0) * 100.0 for name, d in densities.items()}


# ============================================================================
# STRUCTURAL SCORE (continuous)
# ============================================================================

def continuous_structural_score(u_coords, positions, sigma):
    """Structural score with kernel enrichments.

    SS = -E(boundary) + E(attractor) + mean(E(nobles))

    Drop-in replacement for compute_structural_score().

    Returns
    -------
    score : float
    enrichments : dict
    """
    u = np.asarray(u_coords, dtype=float)
    if len(u) == 0:
        return 0.0, {}

    enrichments = kernel_enrichment(u, positions, sigma)

    boundary_e = enrichments.get('boundary', 0.0)
    attractor_e = enrichments.get('attractor', 0.0)
    noble_keys = [k for k in enrichments if k not in ('boundary', 'attractor')]
    noble_mean = np.mean([enrichments[k] for k in noble_keys]) if noble_keys else 0.0

    score = -boundary_e + attractor_e + noble_mean
    return score, enrichments


# ============================================================================
# COMPLIANCE SCORE (continuous, per-subject)
# ============================================================================

def continuous_compliance_score(peak_freqs, f0=8.5, sigma=SIGMA_DEFAULT,
                                 base=PHI, freq_ceil=45.0):
    """Full compliance scoring with continuous kernel.

    Drop-in replacement for lemon_utils.compute_compliance_score().

    Parameters
    ----------
    peak_freqs : array_like
        Peak frequencies in Hz.
    f0 : float
        Anchor frequency.
    sigma : float
        Kernel bandwidth.
    base : float
        Lattice base (default: golden ratio).
    freq_ceil : float
        Upper frequency limit for gamma band.

    Returns
    -------
    dict with keys: compliance, E_boundary, E_noble_2, E_attractor, E_noble_1,
                    n_peaks, lattice_coords, mmd, compliance_<band>
    """
    peak_freqs = np.asarray(peak_freqs, dtype=float)
    peak_freqs = peak_freqs[peak_freqs > 0]

    if len(peak_freqs) == 0:
        return {
            'compliance': np.nan, 'n_peaks': 0,
            'E_boundary': np.nan, 'E_noble_2': np.nan,
            'E_attractor': np.nan, 'E_noble_1': np.nan,
            'lattice_coords': np.array([]), 'mmd': np.nan,
        }

    # Lattice coordinates
    u = lattice_coordinate(peak_freqs, f0, base)
    u_valid = u[np.isfinite(u)]

    if len(u_valid) == 0:
        return {
            'compliance': np.nan, 'n_peaks': len(peak_freqs),
            'E_boundary': np.nan, 'E_noble_2': np.nan,
            'E_attractor': np.nan, 'E_noble_1': np.nan,
            'lattice_coords': np.array([]), 'mmd': np.nan,
        }

    # Natural positions for this base
    positions = natural_positions(base)

    # Structural score
    score, enrichments = continuous_structural_score(u_valid, positions, sigma)

    # MMD
    mmd_val, _, _ = mean_min_distance(u_valid, positions)

    result = {
        'compliance': score,
        'n_peaks': len(peak_freqs),
        'E_boundary': enrichments.get('boundary', np.nan),
        'E_noble_2': enrichments.get('noble_2', np.nan),
        'E_attractor': enrichments.get('attractor', np.nan),
        'E_noble_1': enrichments.get('noble', np.nan),
        'lattice_coords': u_valid,
        'mmd': mmd_val,
    }

    # Band-specific compliance
    bands = make_phi_bands(f0, freq_ceil)
    for band_name, (lo, hi) in bands.items():
        band_freqs = peak_freqs[(peak_freqs >= lo) & (peak_freqs < hi)]
        if len(band_freqs) >= 3:
            u_band = lattice_coordinate(band_freqs, f0, base)
            u_band = u_band[np.isfinite(u_band)]
            if len(u_band) >= 3:
                band_score, _ = continuous_structural_score(
                    u_band, positions, sigma)
                result[f'compliance_{band_name}'] = band_score
            else:
                result[f'compliance_{band_name}'] = np.nan
        else:
            result[f'compliance_{band_name}'] = np.nan

    return result


# ============================================================================
# MEAN MINIMUM DISTANCE (parameter-free)
# ============================================================================

def mean_min_distance(u_coords, positions):
    """Mean minimum circular distance to any lattice position.

    MMD = (1/N) * sum_i min_j d_circ(u_i, p_j)

    Also computes the expected MMD under uniform distribution (analytical)
    and a z-score.

    Returns
    -------
    mmd : float
    expected_mmd : float
    z_score : float (negative = more compliant than chance)
    """
    u = np.asarray(u_coords, dtype=float)
    if len(u) == 0:
        return np.nan, np.nan, np.nan

    pos_values = np.array(list(positions.values()))

    # Compute min distance for each peak
    # d_matrix: (n_peaks, n_positions)
    d_matrix = np.column_stack([circular_distance(u, p) for p in pos_values])
    min_dists = d_matrix.min(axis=1)
    mmd = np.mean(min_dists)

    # Expected MMD under uniform distribution (analytical)
    # Voronoi cells: each position owns the region of the circle closest to it
    # Expected value within a cell of width w centered on a position = w/4
    # Overall = sum_j (w_j * w_j/4)
    expected_mmd = _expected_mmd_uniform(pos_values)

    # Variance under uniform for z-score
    # Var(min_dist) under uniform — computed via second moment
    expected_var = _expected_mmd_variance(pos_values)
    se = np.sqrt(expected_var / len(u)) if expected_var > 0 else 1e-10
    z_score = (mmd - expected_mmd) / se

    return mmd, expected_mmd, z_score


def _expected_mmd_uniform(pos_values):
    """Analytical expected MMD under uniform distribution on [0, 1).

    For Voronoi cell of width w_j:
        E[d | in cell j] = w_j / 4
        P(in cell j) = w_j
    E[MMD] = sum_j w_j^2 / 4
    """
    pos_sorted = np.sort(pos_values % 1.0)
    # Voronoi boundaries: midpoints between adjacent positions (circular)
    n = len(pos_sorted)
    if n == 0:
        return 0.5  # uniform on [0, 0.5]

    # Circular gaps
    gaps = np.diff(pos_sorted)
    # Add the wrap-around gap
    wrap_gap = 1.0 - pos_sorted[-1] + pos_sorted[0]
    all_gaps = np.append(gaps, wrap_gap)

    # Voronoi cell width for position i = (gap_before + gap_after) / 2
    # gap_before[i] = all_gaps[i-1], gap_after[i] = all_gaps[i]
    widths = np.zeros(n)
    for i in range(n):
        gap_before = all_gaps[(i - 1) % n]
        gap_after = all_gaps[i]
        widths[i] = (gap_before + gap_after) / 2.0

    return np.sum(widths ** 2) / 4.0


def _expected_mmd_variance(pos_values):
    """Approximate variance of MMD under uniform distribution.

    Uses E[d^2] - E[d]^2 for each Voronoi cell.
    For a uniform on [0, w/2]: E[d^2] = w^2/12, E[d] = w/4
    Var(d | cell j) = w_j^2/12 - w_j^2/16 = w_j^2/48
    Var(d) = sum_j P(j) * [Var(d|j) + (E[d|j] - E[d])^2]
    """
    pos_sorted = np.sort(np.asarray(pos_values) % 1.0)
    n = len(pos_sorted)
    if n == 0:
        return 1.0 / 12.0

    gaps = np.diff(pos_sorted)
    wrap_gap = 1.0 - pos_sorted[-1] + pos_sorted[0]
    all_gaps = np.append(gaps, wrap_gap)

    widths = np.zeros(n)
    for i in range(n):
        widths[i] = (all_gaps[(i - 1) % n] + all_gaps[i]) / 2.0

    # E[d^2] = sum_j w_j * (w_j^2 / 12)
    e_d2 = np.sum(widths * widths ** 2 / 12.0)
    e_d = np.sum(widths ** 2) / 4.0
    return e_d2 - e_d ** 2


# ============================================================================
# SIGMA SWEEP
# ============================================================================

def sigma_sweep(u_coords, positions, sigmas=None):
    """Compute structural score across range of sigma values.

    Parameters
    ----------
    u_coords : ndarray
        Lattice coordinates.
    positions : dict
        Lattice positions.
    sigmas : array_like, optional
        Sigma values to test. Default: 0.015 to 0.080 in steps of 0.005.

    Returns
    -------
    DataFrame with columns: sigma, score, + one column per position enrichment
    """
    if sigmas is None:
        sigmas = SIGMA_SWEEP_VALUES

    rows = []
    for s in sigmas:
        score, enrichments = continuous_structural_score(u_coords, positions, s)
        row = {'sigma': s, 'score': score}
        row.update(enrichments)
        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================================
# RATIO SPECIFICITY (continuous)
# ============================================================================

# 9-base set matching Paper 2 convention
SPECIFICITY_BASES = {
    '1.4':   1.4,
    '√2':    np.sqrt(2),
    '3/2':   1.5,
    'φ':     PHI,
    '1.7':   1.7,
    '1.8':   1.8,
    '2':     2.0,
    'e':     np.e,
    'π':     np.pi,
}


def continuous_ratio_specificity(freqs, f0, sigma=SIGMA_DEFAULT,
                                   bases=None, n_perm=1000, seed=42):
    """Ratio specificity test with continuous kernel enrichments.

    For each base b:
      1. u = lattice_coordinate(freqs, f0, b)
      2. positions = natural_positions(b)
      3. score = continuous_structural_score(u, positions, sigma)
      4. Phase-rotation null for z-score and p-value
      5. Also compute MMD (parameter-free)

    Parameters
    ----------
    freqs : array_like
        Peak frequencies in Hz.
    f0 : float
        Anchor frequency.
    sigma : float
        Kernel bandwidth.
    bases : dict, optional
        {name: value}. Default: 9-base Paper 2 set.
    n_perm : int
        Number of phase-rotation permutations.
    seed : int
        Random seed.

    Returns
    -------
    DataFrame sorted by SS descending.
    """
    if bases is None:
        bases = SPECIFICITY_BASES

    freqs = np.asarray(freqs, dtype=float)
    freqs = freqs[freqs > 0]
    rng = np.random.default_rng(seed)

    rows = []
    for name, value in bases.items():
        if value <= 1.0:
            continue

        u = lattice_coordinate(freqs, f0, value)
        u = u[np.isfinite(u)]
        if len(u) == 0:
            continue

        positions = natural_positions(value)

        # Observed score
        obs_score, obs_enrichments = continuous_structural_score(
            u, positions, sigma)

        # MMD
        mmd_val, mmd_expected, mmd_z = mean_min_distance(u, positions)

        # Phase-rotation null
        null_scores = np.zeros(n_perm)
        for pi in range(n_perm):
            theta = rng.uniform(0, 1)
            u_rot = (u + theta) % 1.0
            null_scores[pi], _ = continuous_structural_score(
                u_rot, positions, sigma)

        null_mean = null_scores.mean()
        null_std = null_scores.std()
        z_score = (obs_score - null_mean) / null_std if null_std > 0 else 0.0
        p_value = (null_scores >= obs_score).sum() / n_perm

        rows.append({
            'base_name': name,
            'base_value': value,
            'SS_kernel': obs_score,
            'z_score': z_score,
            'p_value': p_value,
            'null_mean': null_mean,
            'null_std': null_std,
            'MMD': mmd_val,
            'MMD_expected': mmd_expected,
            'MMD_z': mmd_z,
            'n_positions': len(positions),
            **{f'E_{k}': v for k, v in obs_enrichments.items()},
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values('SS_kernel', ascending=False).reset_index(drop=True)
        df['rank_SS'] = range(1, len(df) + 1)
        df = df.sort_values('MMD', ascending=True).reset_index(drop=True)
        df['rank_MMD'] = range(1, len(df) + 1)
        df = df.sort_values('SS_kernel', ascending=False).reset_index(drop=True)

    return df


# ============================================================================
# WEIGHTED COMPLIANCE (amplitude-weighted kernel density)
# ============================================================================

def _apply_weight_transform(powers, transform='rank'):
    """Convert raw FOOOF peak powers to weights.

    Parameters
    ----------
    powers : array_like
        FOOOF peak heights (log power over aperiodic).
    transform : str
        'rank'   — weight = rank / N  (primary, nonparametric)
        'zscore' — z-score, clamp negatives to 0
        'linear' — 10**power  (linear from dB)
        'raw'    — power as-is (log-scale)

    Returns
    -------
    ndarray of non-negative weights, normalized to sum to 1.
    """
    p = np.asarray(powers, dtype=float)
    n = len(p)
    if n == 0:
        return np.array([])

    if transform == 'rank':
        order = np.argsort(np.argsort(p))  # 0-based ranks
        w = (order + 1.0) / n
    elif transform == 'zscore':
        mu, sd = p.mean(), p.std()
        w = (p - mu) / sd if sd > 0 else np.ones(n)
        w = np.maximum(w, 0.0)
    elif transform == 'linear':
        w = 10.0 ** p
    elif transform == 'raw':
        w = np.maximum(p, 0.0)
    else:
        raise ValueError(f"Unknown weight_transform: {transform}")

    total = w.sum()
    if total > 0:
        w = w / total
    else:
        w = np.ones(n) / n
    return w


def weighted_kernel_density_at_positions(u_coords, positions, sigma, weights):
    """Gaussian kernel density at lattice positions, amplitude-weighted.

    D_w(p_j) = sum_i w_i * exp(-d_circ(u_i, p_j)^2 / (2*sigma^2))

    where weights sum to 1. Under the null (weights independent of position),
    D_w has the same expectation as unweighted D.
    """
    u = np.asarray(u_coords, dtype=float)
    w = np.asarray(weights, dtype=float)
    n = len(u)
    if n == 0:
        return {name: 0.0 for name in positions}

    two_sigma_sq = 2.0 * sigma ** 2
    densities = {}
    for name, pos in positions.items():
        d = circular_distance(u, pos)
        densities[name] = np.sum(w * np.exp(-d ** 2 / two_sigma_sq))
    return densities


def weighted_kernel_enrichment(u_coords, positions, sigma, weights):
    """Enrichment (%) at each position, amplitude-weighted.

    E_w(p_j) = (D_w(p_j) / D_null(sigma) - 1) * 100
    """
    densities = weighted_kernel_density_at_positions(
        u_coords, positions, sigma, weights)
    null_d = null_kernel_density(sigma)
    if null_d == 0:
        return {name: 0.0 for name in positions}
    return {name: (d / null_d - 1.0) * 100.0 for name, d in densities.items()}


def weighted_structural_score(u_coords, positions, sigma, weights):
    """Structural score with amplitude-weighted kernel enrichments.

    SS_w = -E_w(boundary) + E_w(attractor) + mean(E_w(nobles))
    """
    u = np.asarray(u_coords, dtype=float)
    if len(u) == 0:
        return 0.0, {}

    enrichments = weighted_kernel_enrichment(u, positions, sigma, weights)

    boundary_e = enrichments.get('boundary', 0.0)
    attractor_e = enrichments.get('attractor', 0.0)
    noble_keys = [k for k in enrichments if k not in ('boundary', 'attractor')]
    noble_mean = (np.mean([enrichments[k] for k in noble_keys])
                  if noble_keys else 0.0)

    score = -boundary_e + attractor_e + noble_mean
    return score, enrichments


def weighted_compliance_score(peak_freqs, peak_powers, f0=8.5,
                               sigma=SIGMA_DEFAULT, base=PHI,
                               freq_ceil=45.0, weight_transform='rank'):
    """Full weighted compliance pipeline.

    freqs → lattice coords + weights → weighted SS + per-position enrichments.

    Parameters
    ----------
    peak_freqs : array_like
        Peak frequencies in Hz.
    peak_powers : array_like
        FOOOF peak heights (same length as peak_freqs).
    f0 : float
        Anchor frequency.
    sigma : float
        Kernel bandwidth.
    base : float
        Lattice base (default: phi).
    freq_ceil : float
        Upper frequency limit.
    weight_transform : str
        'rank', 'zscore', 'linear', 'raw'.

    Returns
    -------
    dict with compliance_weighted, per-position enrichments, band-specific.
    """
    peak_freqs = np.asarray(peak_freqs, dtype=float)
    peak_powers = np.asarray(peak_powers, dtype=float)

    # Filter valid
    mask = (peak_freqs > 0) & np.isfinite(peak_freqs) & np.isfinite(peak_powers)
    peak_freqs = peak_freqs[mask]
    peak_powers = peak_powers[mask]

    nan_result = {
        'compliance_weighted': np.nan, 'n_peaks': 0,
        'E_boundary_w': np.nan, 'E_noble_2_w': np.nan,
        'E_attractor_w': np.nan, 'E_noble_1_w': np.nan,
        'mean_weight': np.nan,
    }

    if len(peak_freqs) == 0:
        return nan_result

    # Lattice coordinates
    u = lattice_coordinate(peak_freqs, f0, base)
    valid = np.isfinite(u)
    u = u[valid]
    powers_valid = peak_powers[valid]
    freqs_valid = peak_freqs[valid]

    if len(u) == 0:
        nan_result['n_peaks'] = len(peak_freqs)
        return nan_result

    # Weights
    w = _apply_weight_transform(powers_valid, weight_transform)

    # Natural positions
    positions = natural_positions(base)

    # Weighted structural score
    score, enrichments = weighted_structural_score(u, positions, sigma, w)

    result = {
        'compliance_weighted': score,
        'n_peaks': len(freqs_valid),
        'E_boundary_w': enrichments.get('boundary', np.nan),
        'E_noble_2_w': enrichments.get('noble_2', np.nan),
        'E_attractor_w': enrichments.get('attractor', np.nan),
        'E_noble_1_w': enrichments.get('noble', np.nan),
        'mean_weight': w.mean(),
    }

    # Band-specific weighted compliance
    bands = make_phi_bands(f0, freq_ceil)
    for band_name, (lo, hi) in bands.items():
        band_mask = (freqs_valid >= lo) & (freqs_valid < hi)
        if band_mask.sum() >= 3:
            u_band = u[band_mask]
            w_band = w[band_mask]
            # Re-normalize band weights
            w_sum = w_band.sum()
            if w_sum > 0:
                w_band = w_band / w_sum
            else:
                w_band = np.ones(len(w_band)) / len(w_band)
            band_score, _ = weighted_structural_score(
                u_band, positions, sigma, w_band)
            result[f'compliance_{band_name}_weighted'] = band_score
        else:
            result[f'compliance_{band_name}_weighted'] = np.nan

    return result


def weighted_within_band_shuffle(peak_freqs, peak_powers, f0=8.5,
                                  sigma=SIGMA_DEFAULT, n_perm=1000,
                                  seed=42, weight_transform='rank',
                                  freq_ceil=45.0, base=PHI):
    """Within-band shuffle null for weighted compliance.

    For each permutation: shuffle u-coordinates uniformly within their
    phi-octave band, keeping weights paired with new positions.
    A gamma peak stays in gamma, never lands in alpha.

    Returns
    -------
    dict with SS_obs, null_mean, null_std, z_score, p_value.
    """
    peak_freqs = np.asarray(peak_freqs, dtype=float)
    peak_powers = np.asarray(peak_powers, dtype=float)

    mask = (peak_freqs > 0) & np.isfinite(peak_freqs) & np.isfinite(peak_powers)
    peak_freqs = peak_freqs[mask]
    peak_powers = peak_powers[mask]

    if len(peak_freqs) == 0:
        return {'SS_obs': np.nan, 'null_mean': np.nan, 'null_std': np.nan,
                'z_score': np.nan, 'p_value': np.nan}

    # Lattice coords
    u = lattice_coordinate(peak_freqs, f0, base)
    valid = np.isfinite(u)
    u = u[valid]
    powers_valid = peak_powers[valid]
    freqs_valid = peak_freqs[valid]

    if len(u) == 0:
        return {'SS_obs': np.nan, 'null_mean': np.nan, 'null_std': np.nan,
                'z_score': np.nan, 'p_value': np.nan}

    w = _apply_weight_transform(powers_valid, weight_transform)
    positions = natural_positions(base)

    # Observed
    ss_obs, _ = weighted_structural_score(u, positions, sigma, w)

    # Assign each peak to a phi-octave band
    bands = make_phi_bands(f0, freq_ceil)
    band_labels = np.full(len(freqs_valid), -1, dtype=int)
    band_list = list(bands.items())
    for bi, (bname, (lo, hi)) in enumerate(band_list):
        in_band = (freqs_valid >= lo) & (freqs_valid < hi)
        band_labels[in_band] = bi

    # Peaks outside all bands — assign to nearest
    outside = band_labels == -1
    if outside.any():
        for idx in np.where(outside)[0]:
            f = freqs_valid[idx]
            best_dist = np.inf
            best_bi = 0
            for bi, (bname, (lo, hi)) in enumerate(band_list):
                mid = (lo + hi) / 2
                d = abs(f - mid)
                if d < best_dist:
                    best_dist = d
                    best_bi = bi
            band_labels[idx] = best_bi

    # Null distribution
    rng = np.random.default_rng(seed)
    null_scores = np.zeros(n_perm)
    for pi in range(n_perm):
        u_shuf = u.copy()
        for bi in range(len(band_list)):
            in_band = band_labels == bi
            if in_band.sum() > 0:
                u_shuf[in_band] = rng.uniform(0, 1, size=in_band.sum())
        null_scores[pi], _ = weighted_structural_score(
            u_shuf, positions, sigma, w)

    null_mean = null_scores.mean()
    null_std = null_scores.std()
    z_score = (ss_obs - null_mean) / null_std if null_std > 0 else 0.0
    p_value = (null_scores >= ss_obs).sum() / n_perm

    return {
        'SS_obs': ss_obs,
        'null_mean': null_mean,
        'null_std': null_std,
        'z_score': z_score,
        'p_value': p_value,
    }
