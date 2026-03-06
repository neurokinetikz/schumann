#!/usr/bin/env python3
"""
Investigation: Golden Ratio Architecture in EEG Spectral Peaks

Tests the Pletzer-Klimesch-Lacy hypothesis that EEG frequency bands follow
φⁿ (golden ratio) organization anchored at f₀ ≈ 7.5 Hz.

Data: PhysioNet EEG Motor Movement/Imagery Dataset (eegmmidb v1.0.0)
  109 subjects, 64 channels, 160 Hz, eyes-closed resting baseline (run 02)

The claim:
  f(n) = f₀ × φⁿ,  f₀ = c/(2πr) ≈ 7.5 Hz
  Integer n     → band boundaries (peaks depleted)
  Half-integer n → band attractors  (peaks enriched)
  Zero free parameters.

Eight directions:
  D1: φⁿ lattice fit with phase-rotation permutation null
  D2: Ratio specificity — is φ uniquely preferred? (with null envelope)
  D3: f₀ anchoring — is 7.5 Hz genuinely optimal? (with null envelope)
  D4: 2D (f₀, r) joint heatmap — the money plot
  D5: Per-band decomposition — does the lattice work outside alpha?
  D6: Per-subject enrichment distribution
  D7: Peak frequency distribution (diagnostic — where are the peaks?)
  D8: Surrogate validation — does φ structure survive IAAFT?

Key methodological improvement over v1: phase-rotation null.
  The raw enrichment score is biased by the non-uniform marginal peak
  frequency distribution (alpha dominance). The phase-rotation null adds
  a random offset δ ~ U(0,1) to all lattice phases, preserving the shape
  of the distribution but destroying alignment with f₀. This properly
  tests whether f₀ places peaks at attractors vs chance alignment.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import numpy as np
from scipy import stats
from scipy.signal import welch, find_peaks, medfilt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import mne

# =========================================================================
# PARAMETERS
# =========================================================================
N_SUBJECTS = 109
PHI = (1 + np.sqrt(5)) / 2
F0_CLAIMED = 7.5
SFREQ = 160.0
FREQ_LO, FREQ_HI = 2.0, 50.0
NPERSEG = 512
N_PERM = 5000          # phase-rotation null iterations
N_PERM_SWEEP = 1000    # for per-point sweep nulls
N_SURR = 20
IAAFT_ITER = 100
EEG_PATH = '/tmp/eeg_phi_probe'

# φⁿ band definitions (boundaries at integer n, attractors at half-integer)
PHI_BANDS = {
    'Theta':     (-1, 0),     # φ⁻¹·f₀ to φ⁰·f₀  = 4.63–7.50 Hz
    'Alpha':     (0, 1),      # φ⁰·f₀  to φ¹·f₀  = 7.50–12.13 Hz
    'Low Beta':  (1, 2),      # φ¹·f₀  to φ²·f₀  = 12.13–19.63 Hz
    'High Beta': (2, 3),      # φ²·f₀  to φ³·f₀  = 19.63–31.76 Hz
    'Gamma':     (3, 4),      # φ³·f₀  to φ⁴·f₀  = 31.76–51.38 Hz
}

# Named ratios for D2
NAMED_RATIOS = {
    'φ': PHI, '2': 2.0, 'e': np.e, 'π': np.pi,
    '√2': np.sqrt(2), 'δ_S': 1 + np.sqrt(2),
    '2+√3': 2 + np.sqrt(3), '√3': np.sqrt(3),
    '3/2': 1.5, '5/3': 5/3, '7/4': 1.75, '√5': np.sqrt(5),
}

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'figures')


# =========================================================================
# DATA LOADING
# =========================================================================
def load_eeg_data():
    """Download and load eyes-closed resting EEG from PhysioNet eegmmidb."""
    print(f"Loading eyes-closed baseline for {N_SUBJECTS} subjects...")
    all_data = []

    for subj in range(1, N_SUBJECTS + 1):
        try:
            files = mne.datasets.eegbci.load_data(
                subjects=subj, runs=[2],
                path=EEG_PATH, update_path=False, verbose=False)
            raw = mne.io.read_raw_edf(files[0], preload=True, verbose=False)
            raw.filter(1.0, 55.0, verbose=False)
            data = raw.get_data()
            for ch in range(data.shape[0]):
                all_data.append((subj, ch, data[ch]))
            if subj % 20 == 0:
                print(f"  {subj}/{N_SUBJECTS}")
        except Exception as e:
            print(f"  Subject {subj}: FAILED ({e})")

    print(f"  Total: {len(all_data)} time series")
    return all_data


# =========================================================================
# SPECTRAL PEAK EXTRACTION
# =========================================================================
def extract_peaks(signal_data, sfreq=SFREQ, return_prominence=False):
    """Extract spectral peaks after 1/f background subtraction."""
    freqs, psd = welch(signal_data, fs=sfreq, nperseg=NPERSEG,
                       noverlap=NPERSEG // 2)
    mask = (freqs >= FREQ_LO) & (freqs <= FREQ_HI)
    freqs, psd = freqs[mask], psd[mask]

    log_psd = np.log10(psd + 1e-30)
    kernel = min(len(log_psd) // 3, 51)
    if kernel % 2 == 0:
        kernel += 1
    if kernel < 3:
        return (np.array([]), np.array([])) if return_prominence else np.array([])
    background = medfilt(log_psd, kernel_size=kernel)
    residual = log_psd - background

    mad = np.median(np.abs(residual - np.median(residual)))
    if mad < 1e-10:
        return (np.array([]), np.array([])) if return_prominence else np.array([])

    peak_idx, props = find_peaks(residual, prominence=2 * mad, height=0)
    if return_prominence:
        return freqs[peak_idx], props['prominences']
    return freqs[peak_idx]


def extract_peaks_fooof(signal_data, sfreq=SFREQ):
    """Extract spectral peaks via specparam (FOOOF) parametric decomposition.
    Fits explicit Gaussians against a parametric aperiodic component."""
    from specparam import SpectralModel
    freqs, psd = welch(signal_data, fs=sfreq, nperseg=NPERSEG,
                       noverlap=NPERSEG // 2)
    mask = (freqs >= FREQ_LO) & (freqs <= FREQ_HI)
    freqs, psd = freqs[mask], psd[mask]
    if len(freqs) < 10 or np.all(psd < 1e-30):
        return np.array([])
    try:
        sm = SpectralModel(
            peak_width_limits=[1.0, 8.0],
            max_n_peaks=8,
            min_peak_height=0.05,
            verbose=False,
        )
        sm.fit(freqs, psd)
        peaks = sm.get_params('peak')
        if peaks is None or (hasattr(peaks, 'size') and peaks.size == 0):
            return np.array([])
        if peaks.ndim == 1:
            return np.array([peaks[0]])
        return peaks[:, 0]  # center frequencies
    except Exception:
        return np.array([])


# =========================================================================
# LATTICE PHASE + ENRICHMENT
# =========================================================================
def lattice_phase(freqs, f0, ratio):
    """Map frequencies to lattice phase u ∈ [0, 1).
    u = 0 → boundary, u = 0.5 → attractor."""
    return (np.log(freqs / f0) / np.log(ratio)) % 1.0


def enrichment_score(u_values, width=0.15):
    """(attractor density - boundary density) / expected.
    Positive supports hypothesis. Zero = uniform."""
    n = len(u_values)
    if n < 10:
        return 0.0
    near_bnd = np.sum((u_values < width) | (u_values > 1 - width))
    near_att = np.sum(np.abs(u_values - 0.5) < width)
    expected = n * 2 * width
    if expected < 1:
        return 0.0
    return (near_att - near_bnd) / expected


def phase_rotation_null(u_values, n_perm=N_PERM, rng=None):
    """Phase-rotation permutation null for enrichment score.
    Adds random offset to all phases, preserving shape but destroying
    alignment with f₀. Returns null distribution of enrichment scores."""
    if rng is None:
        rng = np.random.default_rng(0)
    null = np.empty(n_perm)
    for i in range(n_perm):
        delta = rng.uniform()
        u_shifted = (u_values + delta) % 1.0
        null[i] = enrichment_score(u_shifted)
    return null


def enrichment_at(u_values, target, width):
    """Density enrichment at a specific phase position.
    Returns (observed - expected) / expected. Zero = uniform."""
    n = len(u_values)
    if n < 10:
        return 0.0
    # Circular distance on [0, 1)
    d = np.abs(u_values - target)
    d = np.minimum(d, 1.0 - d)
    near = np.sum(d < width)
    expected = n * 2 * width
    return (near - expected) / expected if expected > 0 else 0.0


def kuiper_v(u_values):
    """Kuiper's V test for circular non-uniformity.
    Returns (V, p_value) using asymptotic approximation."""
    n = len(u_values)
    u_sorted = np.sort(u_values)
    i = np.arange(1, n + 1)
    D_plus = np.max(i / n - u_sorted)
    D_minus = np.max(u_sorted - (i - 1) / n)
    V = D_plus + D_minus
    # Stephens (1970) finite-sample correction
    Vstar = V * (np.sqrt(n) + 0.155 + 0.24 / np.sqrt(n))
    # Asymptotic survival function
    p = 0.0
    for j in range(1, 100):
        p += (4 * j**2 * Vstar**2 - 1) * np.exp(-2 * j**2 * Vstar**2)
    p = 2.0 * p
    return V, min(max(p, 0.0), 1.0)


# =========================================================================
# CONTROLS
# =========================================================================
def gen_pink_noise(rng, size):
    white = rng.standard_normal(size)
    fft = np.fft.rfft(white)
    f = np.fft.rfftfreq(size)
    f[0] = 1
    fft /= np.sqrt(f)
    return np.fft.irfft(fft, n=size)


def iaaft_surrogate(data, rng, n_iter=IAAFT_ITER):
    x = data.astype(np.float64)
    n = len(x)
    target_amp = np.abs(np.fft.rfft(x))
    sorted_x = np.sort(x)
    surr = x.copy()
    rng.shuffle(surr)
    for _ in range(n_iter):
        s_fft = np.fft.rfft(surr)
        s_fft = target_amp * np.exp(1j * np.angle(s_fft))
        surr = np.fft.irfft(s_fft, n=n)
        surr = sorted_x[np.argsort(np.argsort(surr))]
    return surr


# =========================================================================
# D1: φⁿ LATTICE FIT WITH PERMUTATION NULL
# =========================================================================
def direction_1(all_peaks):
    print("\n" + "=" * 78)
    print("D1: φⁿ LATTICE FIT (with phase-rotation null)")
    print("=" * 78)

    u = lattice_phase(all_peaks, F0_CLAIMED, PHI)
    observed = enrichment_score(u)

    rng = np.random.default_rng(42)
    null = phase_rotation_null(u, N_PERM, rng)
    p_value = np.mean(null >= observed)
    null_mean = null.mean()
    null_std = null.std()

    print(f"  Peaks: {len(u)}")
    print(f"  Observed enrichment: {observed:+.3f}")
    print(f"  Null (phase-rotated): {null_mean:+.3f} ± {null_std:.3f}")
    print(f"  Excess over null: {observed - null_mean:+.3f}")
    print(f"  p-value: {p_value:.4f}")
    print(f"  95th pctl null: {np.percentile(null, 95):.3f}")

    return u, observed, null


# =========================================================================
# D2: RATIO SPECIFICITY WITH NULL ENVELOPES
# =========================================================================
def direction_2(all_peaks):
    print("\n" + "=" * 78)
    print("D2: RATIO SPECIFICITY (with null envelopes)")
    print("=" * 78)

    rng = np.random.default_rng(100)

    # Named ratios with null CIs
    named_results = {}
    for name, r in NAMED_RATIOS.items():
        u = lattice_phase(all_peaks, F0_CLAIMED, r)
        obs = enrichment_score(u)
        null = phase_rotation_null(u, N_PERM_SWEEP, rng)
        p = np.mean(null >= obs)
        named_results[name] = (r, obs, null.mean(), np.percentile(null, 95), p)
        flag = " ***" if p < 0.01 else " *" if p < 0.05 else ""
        print(f"  {name:8s} r={r:.4f}: score={obs:+.3f}  "
              f"null={null.mean():+.3f}  p={p:.4f}{flag}")

    # Dense sweep (raw scores — null expensive at every point)
    sweep_r = np.linspace(1.2, 4.0, 300)
    sweep_obs = np.array([enrichment_score(lattice_phase(all_peaks, F0_CLAIMED, r))
                          for r in sweep_r])

    # Null envelope: compute null at a subset of ratios
    null_r_idx = np.linspace(0, len(sweep_r) - 1, 30, dtype=int)
    null_95 = np.full(len(sweep_r), np.nan)
    null_mean = np.full(len(sweep_r), np.nan)
    for idx in null_r_idx:
        r = sweep_r[idx]
        u = lattice_phase(all_peaks, F0_CLAIMED, r)
        nl = phase_rotation_null(u, N_PERM_SWEEP, rng)
        null_95[idx] = np.percentile(nl, 95)
        null_mean[idx] = nl.mean()
    # Interpolate
    valid = ~np.isnan(null_95)
    null_95 = np.interp(np.arange(len(sweep_r)),
                        np.where(valid)[0], null_95[valid])
    null_mean = np.interp(np.arange(len(sweep_r)),
                          np.where(valid)[0], null_mean[valid])

    # Excess = observed - null_mean
    sweep_excess = sweep_obs - null_mean

    phi_obs = named_results['φ'][1]
    phi_excess = named_results['φ'][1] - named_results['φ'][2]
    n_better = sum(1 for v in named_results.values() if (v[1] - v[2]) > phi_excess)
    print(f"\n  φ excess over null: {phi_excess:+.3f}")
    print(f"  φ rank (excess): #{n_better + 1}/{len(named_results)}")

    return named_results, sweep_r, sweep_obs, null_95, null_mean


# =========================================================================
# D3: f₀ ANCHORING WITH NULL ENVELOPE
# =========================================================================
def direction_3(all_peaks):
    print("\n" + "=" * 78)
    print("D3: f₀ ANCHORING (with null envelope)")
    print("=" * 78)

    rng = np.random.default_rng(200)
    f0_vals = np.linspace(3.0, 15.0, 300)
    scores = np.array([enrichment_score(lattice_phase(all_peaks, f0, PHI))
                       for f0 in f0_vals])

    # Null envelope at subset
    null_idx = np.linspace(0, len(f0_vals) - 1, 30, dtype=int)
    null_95 = np.full(len(f0_vals), np.nan)
    null_mean_arr = np.full(len(f0_vals), np.nan)
    for idx in null_idx:
        f0 = f0_vals[idx]
        u = lattice_phase(all_peaks, f0, PHI)
        nl = phase_rotation_null(u, N_PERM_SWEEP, rng)
        null_95[idx] = np.percentile(nl, 95)
        null_mean_arr[idx] = nl.mean()
    valid = ~np.isnan(null_95)
    null_95 = np.interp(np.arange(len(f0_vals)),
                        np.where(valid)[0], null_95[valid])
    null_mean_arr = np.interp(np.arange(len(f0_vals)),
                              np.where(valid)[0], null_mean_arr[valid])

    excess = scores - null_mean_arr
    best_idx = np.argmax(excess)
    best_f0 = f0_vals[best_idx]
    claimed_idx = np.argmin(np.abs(f0_vals - F0_CLAIMED))
    schumann_idx = np.argmin(np.abs(f0_vals - 7.83))

    print(f"  Best f₀ (excess):  {best_f0:.2f} Hz  (excess={excess[best_idx]:+.3f})")
    print(f"  Claimed f₀ 7.5:    excess={excess[claimed_idx]:+.3f}  "
          f"(above 95% null: {scores[claimed_idx] > null_95[claimed_idx]})")
    print(f"  Schumann 7.83:     excess={excess[schumann_idx]:+.3f}")

    return f0_vals, scores, null_95, null_mean_arr, best_f0


# =========================================================================
# D4: 2D (f₀, r) JOINT HEATMAP
# =========================================================================
def direction_4(all_peaks):
    print("\n" + "=" * 78)
    print("D4: 2D (f₀, r) JOINT HEATMAP")
    print("=" * 78)

    f0_grid = np.linspace(3.0, 15.0, 100)
    r_grid = np.linspace(1.2, 4.0, 100)
    heatmap = np.zeros((len(r_grid), len(f0_grid)))

    for i, r in enumerate(r_grid):
        for j, f0 in enumerate(f0_grid):
            u = lattice_phase(all_peaks, f0, r)
            heatmap[i, j] = enrichment_score(u)

    # Find global optimum
    best_ij = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    best_r = r_grid[best_ij[0]]
    best_f0 = f0_grid[best_ij[1]]

    print(f"  Grid: {len(f0_grid)} × {len(r_grid)} = {len(f0_grid)*len(r_grid)} points")
    print(f"  Global optimum: f₀={best_f0:.2f} Hz, r={best_r:.3f}  "
          f"(score={heatmap[best_ij]:+.3f})")
    print(f"  At claimed (7.5, φ={PHI:.3f}): "
          f"score={heatmap[np.argmin(np.abs(r_grid - PHI)), np.argmin(np.abs(f0_grid - 7.5))]:+.3f}")

    return f0_grid, r_grid, heatmap, best_f0, best_r


# =========================================================================
# D5: PER-BAND DECOMPOSITION
# =========================================================================
def direction_5(all_peaks):
    """Test enrichment within each φⁿ-defined frequency band."""
    print("\n" + "=" * 78)
    print("D5: PER-BAND DECOMPOSITION")
    print("=" * 78)

    rng = np.random.default_rng(500)
    band_results = {}

    for band_name, (n_lo, n_hi) in PHI_BANDS.items():
        f_lo = F0_CLAIMED * PHI ** n_lo
        f_hi = F0_CLAIMED * PHI ** n_hi
        f_att = F0_CLAIMED * PHI ** ((n_lo + n_hi) / 2)

        in_band = all_peaks[(all_peaks >= f_lo) & (all_peaks < f_hi)]
        n_peaks = len(in_band)

        if n_peaks < 10:
            print(f"  {band_name:12s} [{f_lo:5.1f}–{f_hi:5.1f} Hz] "
                  f"att={f_att:5.1f}:  n={n_peaks:5d} (too few)")
            band_results[band_name] = (f_lo, f_hi, f_att, n_peaks, 0.0, 1.0)
            continue

        u = lattice_phase(in_band, F0_CLAIMED, PHI)
        obs = enrichment_score(u)
        null = phase_rotation_null(u, N_PERM_SWEEP, rng)
        p = np.mean(null >= obs)

        # Also compute: fraction of peaks in inner half (|u-0.5|<0.25)
        inner_frac = np.mean(np.abs(u - 0.5) < 0.25)

        print(f"  {band_name:12s} [{f_lo:5.1f}–{f_hi:5.1f} Hz] "
              f"att={f_att:5.1f}:  n={n_peaks:5d}  score={obs:+.3f}  "
              f"p={p:.4f}  inner={inner_frac:.2f}")
        band_results[band_name] = (f_lo, f_hi, f_att, n_peaks, obs, p)

    return band_results


# =========================================================================
# D6: PER-SUBJECT ENRICHMENT
# =========================================================================
def direction_6(per_subject):
    print("\n" + "=" * 78)
    print("D6: PER-SUBJECT ENRICHMENT")
    print("=" * 78)

    rng = np.random.default_rng(600)
    subj_obs = []
    subj_null_means = []

    for subj in sorted(per_subject.keys()):
        peaks = np.array(per_subject[subj])
        if len(peaks) < 20:
            continue
        u = lattice_phase(peaks, F0_CLAIMED, PHI)
        obs = enrichment_score(u)
        null = phase_rotation_null(u, 500, rng)
        subj_obs.append(obs)
        subj_null_means.append(null.mean())

    subj_obs = np.array(subj_obs)
    subj_null = np.array(subj_null_means)
    excess = subj_obs - subj_null

    t, p = stats.ttest_1samp(excess, 0)
    print(f"  Subjects: {len(excess)}")
    print(f"  Mean excess enrichment: {excess.mean():+.3f} ± {excess.std():.3f}")
    print(f"  t={t:.2f}, p={p:.4f}")
    print(f"  Subjects with positive excess: {np.sum(excess > 0)}/{len(excess)}")

    return subj_obs, subj_null, excess


# =========================================================================
# D8: SURROGATE VALIDATION
# =========================================================================
def direction_8(eeg_signals):
    print("\n" + "=" * 78)
    print("D8: SURROGATE VALIDATION")
    print("=" * 78)

    n_test = min(40, len(eeg_signals))
    rng = np.random.default_rng(42)

    orig_scores, surr_scores = [], []
    for i in range(n_test):
        sig = eeg_signals[i]
        peaks = extract_peaks(sig)
        if len(peaks) < 3:
            continue
        u = lattice_phase(peaks, F0_CLAIMED, PHI)
        orig_scores.append(enrichment_score(u))

        s_list = []
        for _ in range(N_SURR):
            surr = iaaft_surrogate(sig, rng)
            sp = extract_peaks(surr)
            if len(sp) >= 3:
                u_s = lattice_phase(sp, F0_CLAIMED, PHI)
                s_list.append(enrichment_score(u_s))
        if s_list:
            surr_scores.append(np.mean(s_list))

        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{n_test}")

    orig_scores = np.array(orig_scores)
    surr_scores = np.array(surr_scores)

    if len(orig_scores) >= 3 and len(surr_scores) >= 3:
        t, p = stats.ttest_ind(orig_scores, surr_scores)
        pooled = np.sqrt((orig_scores.std()**2 + surr_scores.std()**2) / 2)
        d = (orig_scores.mean() - surr_scores.mean()) / max(pooled, 1e-15)
        print(f"  Original:  {orig_scores.mean():.3f} ± {orig_scores.std():.3f}")
        print(f"  Surrogate: {surr_scores.mean():.3f} ± {surr_scores.std():.3f}")
        print(f"  d={d:.2f}, p={p:.4f}")

    return orig_scores, surr_scores


# =========================================================================
# D9: NOBLE-POSITION ENRICHMENT (response to u=1/φ critique)
# =========================================================================
def direction_9(all_peaks):
    """D9: Test enrichment at u=1/φ=0.618 vs u=0.5.
    Addresses critique that phi-lattice theory predicts noble-position
    enrichment at u=1/φ, not band-center enrichment at u=0.5."""
    print("\n" + "=" * 78)
    print("D9: NOBLE-POSITION ENRICHMENT (u=1/φ vs u=0.5)")
    print("=" * 78)

    u = lattice_phase(all_peaks, F0_CLAIMED, PHI)
    rng = np.random.default_rng(900)
    inv_phi = 1.0 / PHI  # 0.6180339...

    # ── Part A: Targeted tests ──────────────────────────────────────────
    print(f"\n  Peaks: {len(u)}")
    print(f"  1/φ = {inv_phi:.6f}")
    tests = [
        ('u=0.500 w=0.15', 0.500, 0.15),   # our original
        ('u=0.618 w=0.15', inv_phi, 0.15),  # their position, our width
        ('u=0.618 w=0.05', inv_phi, 0.05),  # their exact claim
        ('u=0.500 w=0.05', 0.500, 0.05),    # our position, their width
    ]

    targeted = {}
    for label, target, width in tests:
        obs = enrichment_at(u, target, width)
        null = np.empty(N_PERM)
        for i in range(N_PERM):
            delta = rng.uniform()
            u_shifted = (u + delta) % 1.0
            null[i] = enrichment_at(u_shifted, target, width)
        p = np.mean(null >= obs)
        targeted[label] = (target, width, obs, null.mean(), null.std(), p)
        sig = " ***" if p < 0.01 else " *" if p < 0.05 else ""
        print(f"  {label}:  obs={obs:+.4f}  null={null.mean():+.4f}±{null.std():.4f}  p={p:.4f}{sig}")

    # ── Part B: Phase target sweep ──────────────────────────────────────
    # Where does enrichment actually peak? Agnostic to theory.
    print("\n  Phase target sweep (width=0.05):")
    targets = np.linspace(0, 1, 200, endpoint=False)
    sweep_obs = np.array([enrichment_at(u, t, 0.05) for t in targets])

    # Null envelope at subset of targets
    null_idx = np.linspace(0, len(targets) - 1, 40, dtype=int)
    sweep_null_mean = np.full(len(targets), np.nan)
    sweep_null_95 = np.full(len(targets), np.nan)
    for idx in null_idx:
        t = targets[idx]
        nl = np.empty(N_PERM_SWEEP)
        for i in range(N_PERM_SWEEP):
            delta = rng.uniform()
            u_shifted = (u + delta) % 1.0
            nl[i] = enrichment_at(u_shifted, t, 0.05)
        sweep_null_mean[idx] = nl.mean()
        sweep_null_95[idx] = np.percentile(nl, 95)
    valid = ~np.isnan(sweep_null_mean)
    sweep_null_mean = np.interp(np.arange(len(targets)),
                                np.where(valid)[0], sweep_null_mean[valid])
    sweep_null_95 = np.interp(np.arange(len(targets)),
                              np.where(valid)[0], sweep_null_95[valid])

    sweep_excess = sweep_obs - sweep_null_mean
    best_target = targets[np.argmax(sweep_excess)]
    idx_05 = np.argmin(np.abs(targets - 0.5))
    idx_618 = np.argmin(np.abs(targets - inv_phi))
    print(f"    Best target: u={best_target:.3f} (excess={sweep_excess.max():+.4f})")
    print(f"    At u=0.500:  excess={sweep_excess[idx_05]:+.4f}")
    print(f"    At u=0.618:  excess={sweep_excess[idx_618]:+.4f}")

    # ── Part C: Width sensitivity at u=0.618 ────────────────────────────
    print("\n  Width sensitivity at u=0.618:")
    widths = np.array([0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20])
    width_results = []
    for w in widths:
        obs_w = enrichment_at(u, inv_phi, w)
        nl = np.empty(N_PERM_SWEEP)
        for i in range(N_PERM_SWEEP):
            delta = rng.uniform()
            u_shifted = (u + delta) % 1.0
            nl[i] = enrichment_at(u_shifted, inv_phi, w)
        p_w = np.mean(nl >= obs_w)
        width_results.append((w, obs_w, nl.mean(), p_w))
        sig = " ***" if p_w < 0.01 else " *" if p_w < 0.05 else ""
        print(f"    w={w:.2f}: obs={obs_w:+.4f}  null={nl.mean():+.4f}  p={p_w:.4f}{sig}")

    # ── Part D: Kuiper's V omnibus ──────────────────────────────────────
    # Raw Kuiper (vs uniform)
    V_raw, p_raw = kuiper_v(u)
    # Phase-rotation Kuiper (controls for peak frequency distribution)
    V_null = np.empty(N_PERM_SWEEP)
    for i in range(N_PERM_SWEEP):
        delta = rng.uniform()
        u_shifted = (u + delta) % 1.0
        V_null[i] = kuiper_v(u_shifted)[0]
    p_rot = np.mean(V_null >= V_raw)
    print(f"\n  Kuiper's V omnibus:")
    print(f"    V={V_raw:.4f}, p(asymptotic)={p_raw:.6f}")
    print(f"    p(phase-rotation null)={p_rot:.4f}")
    if p_raw < 0.01:
        print(f"    Phase distribution IS non-uniform (p={p_raw:.2e})")
        if p_rot > 0.05:
            print(f"    But non-uniformity is NOT f₀-specific (phase-rot p={p_rot:.4f})")
        else:
            print(f"    And non-uniformity IS f₀-specific (phase-rot p={p_rot:.4f})")

    # ── Part E: D2 re-ranking with u=0.618 metric ──────────────────────
    print(f"\n  D2 re-ranking (u=0.618, w=0.05):")
    ratio_618 = {}
    for name, r in NAMED_RATIOS.items():
        u_r = lattice_phase(all_peaks, F0_CLAIMED, r)
        obs_r = enrichment_at(u_r, inv_phi, 0.05)
        nl = np.empty(N_PERM_SWEEP)
        for i in range(N_PERM_SWEEP):
            delta = rng.uniform()
            u_shifted = (u_r + delta) % 1.0
            nl[i] = enrichment_at(u_shifted, inv_phi, 0.05)
        p_r = np.mean(nl >= obs_r)
        excess_r = obs_r - nl.mean()
        ratio_618[name] = (r, obs_r, nl.mean(), excess_r, p_r)

    phi_excess = ratio_618['φ'][3]
    n_better = sum(1 for v in ratio_618.values() if v[3] > phi_excess)
    print(f"    φ excess: {phi_excess:+.4f}, rank: #{n_better + 1}/{len(ratio_618)}")
    for name in sorted(ratio_618, key=lambda n: -ratio_618[n][3]):
        v = ratio_618[name]
        sig = " ***" if v[4] < 0.01 else " *" if v[4] < 0.05 else ""
        print(f"    {name:8s}: excess={v[3]:+.4f}  p={v[4]:.4f}{sig}")

    # ── Part F: D4 heatmap with u=0.618 metric ─────────────────────────
    print(f"\n  D4 re-scan (u=0.618, w=0.05):")
    f0_g = np.linspace(3.0, 15.0, 100)
    r_g = np.linspace(1.2, 4.0, 100)
    heatmap_618 = np.zeros((len(r_g), len(f0_g)))
    for i, r in enumerate(r_g):
        for j, f0 in enumerate(f0_g):
            u_tmp = lattice_phase(all_peaks, f0, r)
            heatmap_618[i, j] = enrichment_at(u_tmp, inv_phi, 0.05)
    best_ij = np.unravel_index(np.argmax(heatmap_618), heatmap_618.shape)
    best_r_618 = r_g[best_ij[0]]
    best_f0_618 = f0_g[best_ij[1]]
    claimed_ij = (np.argmin(np.abs(r_g - PHI)), np.argmin(np.abs(f0_g - 7.5)))
    print(f"    Global optimum: f₀={best_f0_618:.2f}, r={best_r_618:.3f} "
          f"(score={heatmap_618[best_ij]:+.4f})")
    print(f"    At claimed (7.5, φ): score={heatmap_618[claimed_ij]:+.4f}")

    return (targeted, targets, sweep_obs, sweep_null_mean, sweep_null_95,
            width_results, V_raw, p_raw, p_rot, ratio_618,
            f0_g, r_g, heatmap_618, best_f0_618, best_r_618)


# =========================================================================
# D10: FOOOF EXTRACTION COMPARISON
# =========================================================================
def direction_10(all_data):
    """D10: Re-run D9 key tests with FOOOF (specparam) peak extraction.
    Tests whether the noble-position result depends on extraction method."""
    print("\n" + "=" * 78)
    print("D10: FOOOF EXTRACTION COMPARISON")
    print("=" * 78)

    # ── Extract peaks with FOOOF ────────────────────────────────────────
    print("  Extracting peaks via specparam (FOOOF)...")
    fooof_peaks = []
    fooof_per_subject = {}
    medfilt_peaks = []
    medfilt_per_subject = {}

    for idx, (subj, ch, sig) in enumerate(all_data):
        fp = extract_peaks_fooof(sig)
        fooof_peaks.extend(fp)
        fooof_per_subject.setdefault(subj, []).extend(fp)

        mp = extract_peaks(sig)
        medfilt_peaks.extend(mp)
        medfilt_per_subject.setdefault(subj, []).extend(mp)

        if (idx + 1) % 1000 == 0:
            print(f"    {idx + 1}/{len(all_data)}")

    fooof_peaks = np.array(fooof_peaks)
    medfilt_peaks = np.array(medfilt_peaks)

    print(f"\n  Peak counts:")
    print(f"    Median-filter: {len(medfilt_peaks)}")
    print(f"    FOOOF:         {len(fooof_peaks)}")
    print(f"    Ratio:         {len(fooof_peaks) / max(len(medfilt_peaks), 1):.2f}x")

    if len(fooof_peaks) < 100:
        print("  FOOOF: too few peaks, aborting D10")
        return None

    # ── Band-wise comparison ────────────────────────────────────────────
    print(f"\n  Band-wise peak counts (FOOOF vs medfilt):")
    for band_name, (n_lo, n_hi) in PHI_BANDS.items():
        f_lo = F0_CLAIMED * PHI ** n_lo
        f_hi = F0_CLAIMED * PHI ** n_hi
        n_f = np.sum((fooof_peaks >= f_lo) & (fooof_peaks < f_hi))
        n_m = np.sum((medfilt_peaks >= f_lo) & (medfilt_peaks < f_hi))
        ratio = n_f / max(n_m, 1)
        print(f"    {band_name:12s} [{f_lo:5.1f}–{f_hi:5.1f}]:  FOOOF={n_f:6d}  medfilt={n_m:6d}  ratio={ratio:.2f}x")

    inv_phi = 1.0 / PHI
    rng = np.random.default_rng(1000)

    # ── D9 key tests on FOOOF peaks ─────────────────────────────────────
    print(f"\n  D9 tests on FOOOF peaks:")
    u_f = lattice_phase(fooof_peaks, F0_CLAIMED, PHI)
    u_m = lattice_phase(medfilt_peaks, F0_CLAIMED, PHI)

    tests = [
        ('u=0.500 w=0.15', 0.500, 0.15),
        ('u=0.618 w=0.15', inv_phi, 0.15),
        ('u=0.618 w=0.05', inv_phi, 0.05),
        ('u=0.500 w=0.05', 0.500, 0.05),
    ]

    fooof_results = {}
    medfilt_results = {}
    for label, target, width in tests:
        # FOOOF
        obs_f = enrichment_at(u_f, target, width)
        null_f = np.empty(N_PERM)
        for i in range(N_PERM):
            delta = rng.uniform()
            null_f[i] = enrichment_at((u_f + delta) % 1.0, target, width)
        p_f = np.mean(null_f >= obs_f)
        fooof_results[label] = (target, width, obs_f, null_f.mean(), null_f.std(), p_f)

        # Medfilt (re-run for fair comparison with same RNG state pattern)
        obs_m = enrichment_at(u_m, target, width)
        null_m = np.empty(N_PERM)
        for i in range(N_PERM):
            delta = rng.uniform()
            null_m[i] = enrichment_at((u_m + delta) % 1.0, target, width)
        p_m = np.mean(null_m >= obs_m)
        medfilt_results[label] = (target, width, obs_m, null_m.mean(), null_m.std(), p_m)

        sig_f = " ***" if p_f < 0.01 else " *" if p_f < 0.05 else ""
        sig_m = " ***" if p_m < 0.01 else " *" if p_m < 0.05 else ""
        print(f"    {label}:")
        print(f"      FOOOF:   obs={obs_f:+.4f}  null={null_f.mean():+.4f}±{null_f.std():.4f}  p={p_f:.4f}{sig_f}")
        print(f"      medfilt: obs={obs_m:+.4f}  null={null_m.mean():+.4f}±{null_m.std():.4f}  p={p_m:.4f}{sig_m}")

    # ── Kuiper omnibus ──────────────────────────────────────────────────
    V_f, p_f_raw = kuiper_v(u_f)
    V_null_f = np.empty(N_PERM_SWEEP)
    for i in range(N_PERM_SWEEP):
        delta = rng.uniform()
        V_null_f[i] = kuiper_v((u_f + delta) % 1.0)[0]
    p_f_rot = np.mean(V_null_f >= V_f)

    V_m, p_m_raw = kuiper_v(u_m)
    V_null_m = np.empty(N_PERM_SWEEP)
    for i in range(N_PERM_SWEEP):
        delta = rng.uniform()
        V_null_m[i] = kuiper_v((u_m + delta) % 1.0)[0]
    p_m_rot = np.mean(V_null_m >= V_m)

    print(f"\n  Kuiper's V omnibus:")
    print(f"    FOOOF:   V={V_f:.4f}, p(asymp)={p_f_raw:.2e}, p(phase-rot)={p_f_rot:.4f}")
    print(f"    medfilt: V={V_m:.4f}, p(asymp)={p_m_raw:.2e}, p(phase-rot)={p_m_rot:.4f}")

    # ── D2 ratio ranking with FOOOF ─────────────────────────────────────
    print(f"\n  D2 re-ranking (u=0.618, w=0.05) with FOOOF peaks:")
    ratio_618_f = {}
    for name, r in NAMED_RATIOS.items():
        u_r = lattice_phase(fooof_peaks, F0_CLAIMED, r)
        obs_r = enrichment_at(u_r, inv_phi, 0.05)
        nl = np.empty(N_PERM_SWEEP)
        for i in range(N_PERM_SWEEP):
            delta = rng.uniform()
            nl[i] = enrichment_at((u_r + delta) % 1.0, inv_phi, 0.05)
        p_r = np.mean(nl >= obs_r)
        excess_r = obs_r - nl.mean()
        ratio_618_f[name] = (r, obs_r, nl.mean(), excess_r, p_r)

    phi_excess = ratio_618_f['φ'][3]
    n_better = sum(1 for v in ratio_618_f.values() if v[3] > phi_excess)
    print(f"    φ excess: {phi_excess:+.4f}, rank: #{n_better + 1}/{len(ratio_618_f)}")
    for name in sorted(ratio_618_f, key=lambda n: -ratio_618_f[n][3]):
        v = ratio_618_f[name]
        sig = " ***" if v[4] < 0.01 else " *" if v[4] < 0.05 else ""
        print(f"    {name:8s}: excess={v[3]:+.4f}  p={v[4]:.4f}{sig}")

    # ── Per-subject comparison ──────────────────────────────────────────
    print(f"\n  Per-subject enrichment at u=0.618 (w=0.05):")
    subj_excess_f = []
    subj_excess_m = []
    for subj in sorted(fooof_per_subject.keys()):
        fp = np.array(fooof_per_subject.get(subj, []))
        mp = np.array(medfilt_per_subject.get(subj, []))
        if len(fp) >= 20:
            u_s = lattice_phase(fp, F0_CLAIMED, PHI)
            obs_s = enrichment_at(u_s, inv_phi, 0.05)
            null_s = np.empty(500)
            for i in range(500):
                null_s[i] = enrichment_at((u_s + rng.uniform()) % 1.0, inv_phi, 0.05)
            subj_excess_f.append(obs_s - null_s.mean())
        if len(mp) >= 20:
            u_s = lattice_phase(mp, F0_CLAIMED, PHI)
            obs_s = enrichment_at(u_s, inv_phi, 0.05)
            null_s = np.empty(500)
            for i in range(500):
                null_s[i] = enrichment_at((u_s + rng.uniform()) % 1.0, inv_phi, 0.05)
            subj_excess_m.append(obs_s - null_s.mean())

    subj_excess_f = np.array(subj_excess_f)
    subj_excess_m = np.array(subj_excess_m)

    if len(subj_excess_f) >= 3:
        t_f, p_f_subj = stats.ttest_1samp(subj_excess_f, 0)
        print(f"    FOOOF:   {subj_excess_f.mean():+.4f} ± {subj_excess_f.std():.4f}, "
              f"t={t_f:.2f}, p={p_f_subj:.4f}, {np.sum(subj_excess_f > 0)}/{len(subj_excess_f)} positive")
    if len(subj_excess_m) >= 3:
        t_m, p_m_subj = stats.ttest_1samp(subj_excess_m, 0)
        print(f"    medfilt: {subj_excess_m.mean():+.4f} ± {subj_excess_m.std():.4f}, "
              f"t={t_m:.2f}, p={p_m_subj:.4f}, {np.sum(subj_excess_m > 0)}/{len(subj_excess_m)} positive")

    # ── Phase target sweep with FOOOF ───────────────────────────────────
    print(f"\n  Phase target sweep (FOOOF, w=0.05):")
    targets = np.linspace(0, 1, 200, endpoint=False)
    sweep_f = np.array([enrichment_at(u_f, t, 0.05) for t in targets])
    null_idx = np.linspace(0, len(targets) - 1, 40, dtype=int)
    sweep_null_f = np.full(len(targets), np.nan)
    for idx in null_idx:
        t = targets[idx]
        nl = np.empty(N_PERM_SWEEP)
        for i in range(N_PERM_SWEEP):
            nl[i] = enrichment_at((u_f + rng.uniform()) % 1.0, t, 0.05)
        sweep_null_f[idx] = nl.mean()
    valid = ~np.isnan(sweep_null_f)
    sweep_null_f = np.interp(np.arange(len(targets)),
                             np.where(valid)[0], sweep_null_f[valid])
    sweep_excess_f = sweep_f - sweep_null_f
    best_t_f = targets[np.argmax(sweep_excess_f)]
    idx_05 = np.argmin(np.abs(targets - 0.5))
    idx_618 = np.argmin(np.abs(targets - inv_phi))
    print(f"    Best target: u={best_t_f:.3f} (excess={sweep_excess_f.max():+.4f})")
    print(f"    At u=0.500:  excess={sweep_excess_f[idx_05]:+.4f}")
    print(f"    At u=0.618:  excess={sweep_excess_f[idx_618]:+.4f}")

    # ── Rebuttal A: Subject-averaged PSD → FOOOF ──────────────────────
    from specparam import SpectralModel
    print(f"\n  REBUTTAL A: Subject-averaged PSD → FOOOF")
    by_subject = {}
    for subj, ch, sig in all_data:
        by_subject.setdefault(subj, []).append(sig)

    avg_fooof_peaks = []
    for subj in sorted(by_subject.keys()):
        sigs = by_subject[subj]
        psds = []
        for sig in sigs:
            freqs_w, psd_w = welch(sig, fs=SFREQ, nperseg=NPERSEG,
                                   noverlap=NPERSEG // 2)
            psds.append(psd_w)
        avg_psd = np.mean(psds, axis=0)
        mask = (freqs_w >= FREQ_LO) & (freqs_w <= FREQ_HI)
        try:
            sm = SpectralModel(peak_width_limits=[1.0, 8.0], max_n_peaks=8,
                               min_peak_height=0.05, verbose=False)
            sm.fit(freqs_w[mask], avg_psd[mask])
            p = sm.get_params('peak')
            if p is not None and hasattr(p, 'size') and p.size > 0:
                cfs = p[:, 0] if p.ndim == 2 else [p[0]]
                avg_fooof_peaks.extend(cfs)
        except Exception:
            pass
    avg_fooof_peaks = np.array(avg_fooof_peaks)
    print(f"    Peaks: {len(avg_fooof_peaks)} ({len(avg_fooof_peaks)/len(by_subject):.1f}/subject)")

    avg_results = {}
    if len(avg_fooof_peaks) >= 50:
        u_avg = lattice_phase(avg_fooof_peaks, F0_CLAIMED, PHI)
        for label, target, width in tests:
            obs_a = enrichment_at(u_avg, target, width)
            null_a = np.empty(N_PERM)
            for i in range(N_PERM):
                null_a[i] = enrichment_at((u_avg + rng.uniform()) % 1.0, target, width)
            p_a = np.mean(null_a >= obs_a)
            avg_results[label] = p_a
            sig = " ***" if p_a < 0.01 else " *" if p_a < 0.05 else ""
            print(f"    {label}: obs={obs_a:+.4f}  p={p_a:.4f}{sig}")

        V_avg, _ = kuiper_v(u_avg)
        V_null_a = np.empty(N_PERM_SWEEP)
        for i in range(N_PERM_SWEEP):
            V_null_a[i] = kuiper_v((u_avg + rng.uniform()) % 1.0)[0]
        p_avg_rot = np.mean(V_null_a >= V_avg)
        avg_results['kuiper_rot'] = p_avg_rot
        print(f"    Kuiper phase-rot: p={p_avg_rot:.4f}")

    # ── Rebuttal B: Alpha-band only ─────────────────────────────────────
    print(f"\n  REBUTTAL B: Alpha-band only [{F0_CLAIMED:.1f}–{F0_CLAIMED * PHI:.1f} Hz]")
    alpha_lo = F0_CLAIMED * PHI ** 0
    alpha_hi = F0_CLAIMED * PHI ** 1
    fooof_alpha = fooof_peaks[(fooof_peaks >= alpha_lo) & (fooof_peaks < alpha_hi)]
    medfilt_alpha = medfilt_peaks[(medfilt_peaks >= alpha_lo) & (medfilt_peaks < alpha_hi)]
    print(f"    FOOOF alpha: {len(fooof_alpha)}, medfilt alpha: {len(medfilt_alpha)}")

    alpha_results = {}
    for method_name, alpha_set in [('FOOOF', fooof_alpha), ('medfilt', medfilt_alpha)]:
        if len(alpha_set) < 50:
            continue
        u_a = lattice_phase(alpha_set, F0_CLAIMED, PHI)
        for label, target, width in tests:
            obs_a = enrichment_at(u_a, target, width)
            null_a = np.empty(N_PERM)
            for i in range(N_PERM):
                null_a[i] = enrichment_at((u_a + rng.uniform()) % 1.0, target, width)
            p_a = np.mean(null_a >= obs_a)
            key = f"{method_name}:{label}"
            alpha_results[key] = p_a
            sig = " ***" if p_a < 0.01 else " *" if p_a < 0.05 else ""
            print(f"    {method_name} {label}: obs={obs_a:+.4f}  p={p_a:.4f}{sig}")

        V_a, _ = kuiper_v(u_a)
        V_null_a = np.empty(N_PERM_SWEEP)
        for i in range(N_PERM_SWEEP):
            V_null_a[i] = kuiper_v((u_a + rng.uniform()) % 1.0)[0]
        p_a_rot = np.mean(V_null_a >= V_a)
        alpha_results[f"{method_name}:kuiper_rot"] = p_a_rot
        print(f"    {method_name} Kuiper phase-rot: p={p_a_rot:.4f}")

    # ── Rebuttal C: FOOOF parameter sensitivity (averaged PSDs) ─────────
    print(f"\n  REBUTTAL C: FOOOF parameter sensitivity (averaged PSDs)")
    param_results = {}
    for mph in [0.01, 0.05, 0.10]:
        peaks_p = []
        for subj in sorted(by_subject.keys()):
            sigs = by_subject[subj]
            psds = []
            for sig in sigs:
                freqs_w, psd_w = welch(sig, fs=SFREQ, nperseg=NPERSEG,
                                       noverlap=NPERSEG // 2)
                psds.append(psd_w)
            avg_psd = np.mean(psds, axis=0)
            mask = (freqs_w >= FREQ_LO) & (freqs_w <= FREQ_HI)
            try:
                sm = SpectralModel(peak_width_limits=[0.5, 12.0], max_n_peaks=12,
                                   min_peak_height=mph, verbose=False)
                sm.fit(freqs_w[mask], avg_psd[mask])
                p = sm.get_params('peak')
                if p is not None and hasattr(p, 'size') and p.size > 0:
                    cfs = p[:, 0] if p.ndim == 2 else [p[0]]
                    peaks_p.extend(cfs)
            except Exception:
                pass
        peaks_p = np.array(peaks_p)
        if len(peaks_p) < 50:
            print(f"    mph={mph}: {len(peaks_p)} peaks (too few)")
            continue
        u_p = lattice_phase(peaks_p, F0_CLAIMED, PHI)
        obs_p = enrichment_at(u_p, inv_phi, 0.05)
        null_p = np.empty(N_PERM)
        for i in range(N_PERM):
            null_p[i] = enrichment_at((u_p + rng.uniform()) % 1.0, inv_phi, 0.05)
        p_val = np.mean(null_p >= obs_p)
        param_results[mph] = (len(peaks_p), p_val)
        sig = " ***" if p_val < 0.01 else " *" if p_val < 0.05 else ""
        print(f"    mph={mph}: {len(peaks_p)} peaks, u=0.618 w=0.05 p={p_val:.4f}{sig}")

    # ── Rebuttal D: Matched peak count ──────────────────────────────────
    print(f"\n  REBUTTAL D: Matched peak count (subsample medfilt → {len(fooof_peaks)})")
    rng_d = np.random.default_rng(5000)
    matched_p = []
    for trial in range(50):
        idx = rng_d.choice(len(medfilt_peaks), size=len(fooof_peaks), replace=False)
        sub = medfilt_peaks[idx]
        u_s = lattice_phase(sub, F0_CLAIMED, PHI)
        obs_s = enrichment_at(u_s, inv_phi, 0.05)
        null_s = np.empty(500)
        for i in range(500):
            null_s[i] = enrichment_at((u_s + rng_d.uniform()) % 1.0, inv_phi, 0.05)
        matched_p.append(np.mean(null_s >= obs_s))
    matched_p = np.array(matched_p)
    print(f"    50 subsamples: p={matched_p.mean():.4f} ± {matched_p.std():.4f}, "
          f"{np.mean(matched_p < 0.05):.0%} reach p<0.05")

    return {
        'fooof_peaks': fooof_peaks,
        'medfilt_peaks': medfilt_peaks,
        'fooof_results': fooof_results,
        'medfilt_results': medfilt_results,
        'kuiper_fooof': (V_f, p_f_raw, p_f_rot),
        'kuiper_medfilt': (V_m, p_m_raw, p_m_rot),
        'ratio_618_fooof': ratio_618_f,
        'subj_excess_fooof': subj_excess_f,
        'subj_excess_medfilt': subj_excess_m,
        'targets': targets,
        'sweep_fooof': sweep_f,
        'sweep_null_fooof': sweep_null_f,
        'avg_results': avg_results,
        'alpha_results': alpha_results,
        'param_results': param_results,
        'matched_p': matched_p,
    }


# =========================================================================
# FIGURE
# =========================================================================
def make_figure(all_peaks, u_d1, obs_d1, null_d1,
                named_d2, sweep_r, sweep_obs, null_95_d2, null_mean_d2,
                f0_vals, f0_scores, null_95_d3, null_mean_d3, best_f0_d3,
                f0_grid, r_grid, heatmap, best_f0_d4, best_r_d4,
                band_results,
                subj_obs, subj_null, subj_excess,
                d8_orig, d8_surr,
                example_signal):

    fig = plt.figure(figsize=(20, 10), facecolor='#111111')
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.38, wspace=0.35,
                           left=0.04, right=0.97, top=0.92, bottom=0.07)
    fig.suptitle("Golden Ratio Architecture in EEG — Deep Probe",
                 color='white', fontsize=14, fontweight='bold')

    def dax(pos):
        ax = fig.add_subplot(pos)
        ax.set_facecolor('#181818')
        for s in ax.spines.values():
            s.set_color('#444444')
        ax.tick_params(colors='#cccccc', labelsize=7)
        return ax

    # ── P1: PSD + φⁿ grid + peak freq distribution ─────────────────────
    ax1 = dax(gs[0, 0])
    freqs, psd = welch(example_signal, fs=SFREQ, nperseg=NPERSEG,
                       noverlap=NPERSEG // 2)
    m = (freqs >= FREQ_LO) & (freqs <= FREQ_HI)
    ax1.semilogy(freqs[m], psd[m], color='#66bbff', lw=0.8, alpha=0.9)

    for n in range(-2, 6):
        fb = F0_CLAIMED * PHI ** n
        fa = F0_CLAIMED * PHI ** (n + 0.5)
        if FREQ_LO < fb < FREQ_HI:
            ax1.axvline(fb, color='#ff6666', alpha=0.4, lw=0.7, ls='--')
        if FREQ_LO < fa < FREQ_HI:
            ax1.axvline(fa, color='#66ff66', alpha=0.4, lw=0.7, ls='--')

    # Peak freq histogram on twin axis
    ax1t = ax1.twinx()
    ax1t.hist(all_peaks, bins=80, range=(FREQ_LO, FREQ_HI),
              color='#ffaa44', alpha=0.25, density=True)
    ax1t.set_ylabel('Peak density', color='#ffaa44', fontsize=7)
    ax1t.tick_params(colors='#ffaa44', labelsize=6)
    ax1t.set_ylim(bottom=0)

    ax1.set_xlabel('Frequency (Hz)', color='#ccc', fontsize=8)
    ax1.set_ylabel('PSD', color='#ccc', fontsize=8)
    ax1.set_title('PSD + φⁿ grid + peak distribution', color='white', fontsize=9)
    ax1.set_zorder(ax1t.get_zorder() + 1)
    ax1.patch.set_visible(False)

    # ── P2: Lattice phase with null CI ──────────────────────────────────
    ax2 = dax(gs[0, 1])
    bins = np.linspace(0, 1, 41)
    ax2.hist(u_d1, bins=bins, color='#44aaff', alpha=0.7, density=True,
             edgecolor='#333', lw=0.4, label=f'EEG (n={len(u_d1)})')

    null_mean = null_d1.mean()
    null_p95 = np.percentile(null_d1, 97.5)
    null_p05 = np.percentile(null_d1, 2.5)
    ax2.axhline(1.0, color='#666', lw=1, ls=':', label='Uniform')
    ax2.axvspan(0, 0.15, alpha=0.12, color='red')
    ax2.axvspan(0.85, 1.0, alpha=0.12, color='red')
    ax2.axvspan(0.35, 0.65, alpha=0.12, color='green')

    p_val = np.mean(null_d1 >= obs_d1)
    ax2.set_xlabel('u = log_φ(f/f₀) mod 1', color='#ccc', fontsize=8)
    ax2.set_ylabel('Density', color='#ccc', fontsize=8)
    ax2.set_title(f'D1: Lattice phase  (obs={obs_d1:+.3f}, p={p_val:.4f})',
                  color='white', fontsize=9)
    ax2.legend(fontsize=6, facecolor='#222', edgecolor='#444', labelcolor='#ccc')

    # ── P3: 2D heatmap ──────────────────────────────────────────────────
    ax3 = dax(gs[0, 2])
    extent = [f0_grid[0], f0_grid[-1], r_grid[0], r_grid[-1]]
    im = ax3.imshow(heatmap, aspect='auto', origin='lower', extent=extent,
                    cmap='RdBu_r', vmin=-0.6, vmax=0.6)
    ax3.plot(F0_CLAIMED, PHI, 'w*', markersize=12, label=f'Claimed ({F0_CLAIMED}, φ)')
    ax3.plot(best_f0_d4, best_r_d4, 'g+', markersize=10, mew=2,
             label=f'Best ({best_f0_d4:.1f}, {best_r_d4:.2f})')
    ax3.set_xlabel('f₀ (Hz)', color='#ccc', fontsize=8)
    ax3.set_ylabel('Ratio r', color='#ccc', fontsize=8)
    ax3.set_title('D4: Joint (f₀, r) enrichment', color='white', fontsize=9)
    ax3.legend(fontsize=6, facecolor='#222', edgecolor='#444', labelcolor='#ccc',
               loc='upper right')
    cb = fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=6, colors='#ccc')

    # ── P4: Ratio sweep with null ───────────────────────────────────────
    ax4 = dax(gs[0, 3])
    ax4.fill_between(sweep_r, null_mean_d2, null_95_d2,
                     color='#ff6666', alpha=0.15, label='Null 95% CI')
    ax4.plot(sweep_r, null_mean_d2, color='#ff6666', lw=0.5, alpha=0.5)
    ax4.plot(sweep_r, sweep_obs, color='#44aaff', lw=0.7, alpha=0.7)

    # Highlight ratios with excess above null
    for name, (r, obs, nm, n95, p) in named_d2.items():
        c = '#ff4444' if name == 'φ' else '#aaaaaa'
        sz = 40 if name == 'φ' else 15
        marker = '*' if p < 0.05 else 'o'
        ax4.scatter(r, obs, color=c, s=sz, zorder=5, marker=marker,
                    edgecolors='white' if name == 'φ' else 'none',
                    linewidths=0.5 if name == 'φ' else 0)

    ax4.axhline(0, color='#444', lw=0.5)
    ax4.set_xlabel('Ratio r', color='#ccc', fontsize=8)
    ax4.set_ylabel('Enrichment', color='#ccc', fontsize=8)
    ax4.set_title('D2: Ratio specificity (red=null 95%)', color='white', fontsize=9)

    # ── P5: f₀ sweep with null ──────────────────────────────────────────
    ax5 = dax(gs[1, 0])
    ax5.fill_between(f0_vals, null_mean_d3, null_95_d3,
                     color='#ff6666', alpha=0.15, label='Null 95% CI')
    ax5.plot(f0_vals, null_mean_d3, color='#ff6666', lw=0.5, alpha=0.5)
    ax5.plot(f0_vals, f0_scores, color='#44aaff', lw=1)
    ax5.axvline(F0_CLAIMED, color='#ff6666', lw=1.5, ls='--',
                label=f'Claimed {F0_CLAIMED}')
    ax5.axvline(best_f0_d3, color='#66ff66', lw=1.5, ls=':',
                label=f'Best {best_f0_d3:.1f}')
    ax5.axvline(7.83, color='#ffaa00', lw=1, ls='--', alpha=0.5,
                label='Schumann 7.83')
    ax5.set_xlabel('f₀ (Hz)', color='#ccc', fontsize=8)
    ax5.set_ylabel('Enrichment', color='#ccc', fontsize=8)
    ax5.set_title('D3: f₀ anchoring', color='white', fontsize=9)
    ax5.legend(fontsize=5.5, facecolor='#222', edgecolor='#444', labelcolor='#ccc')

    # ── P6: Per-band decomposition ──────────────────────────────────────
    ax6 = dax(gs[1, 1])
    band_names = list(band_results.keys())
    band_scores = [band_results[b][4] for b in band_names]
    band_ps = [band_results[b][5] for b in band_names]
    band_n = [band_results[b][3] for b in band_names]
    colors6 = ['#44ff44' if p < 0.05 else '#ff6666' if s < 0 else '#888888'
               for s, p in zip(band_scores, band_ps)]
    bars = ax6.bar(range(len(band_names)), band_scores, color=colors6, alpha=0.7)
    ax6.set_xticks(range(len(band_names)))
    ax6.set_xticklabels(band_names, fontsize=7, color='#ccc', rotation=20)
    ax6.axhline(0, color='#444', lw=0.5)
    ax6.set_ylabel('Enrichment', color='#ccc', fontsize=8)
    ax6.set_title('D5: Per-band (green=p<.05)', color='white', fontsize=9)
    # Annotate counts
    for i, (n, p) in enumerate(zip(band_n, band_ps)):
        ax6.text(i, band_scores[i] + 0.02 * np.sign(band_scores[i]),
                 f'n={n}\np={p:.3f}', ha='center', fontsize=5.5, color='#ccc')

    # ── P7: Per-subject excess ──────────────────────────────────────────
    ax7 = dax(gs[1, 2])
    if len(subj_excess) > 0:
        ax7.hist(subj_excess, bins=25, color='#44aaff', alpha=0.7,
                 edgecolor='#333', lw=0.4, density=True)
        ax7.axvline(0, color='#ff6666', lw=1.5, ls='--', label='Null=0')
        ax7.axvline(subj_excess.mean(), color='#66ff66', lw=1.5,
                    label=f'Mean={subj_excess.mean():+.3f}')
        t, p = stats.ttest_1samp(subj_excess, 0)
        ax7.set_xlabel('Excess enrichment (obs − null)', color='#ccc', fontsize=8)
        ax7.set_ylabel('Density', color='#ccc', fontsize=8)
        ax7.set_title(f'D6: Per-subject (t={t:.1f}, p={p:.4f})',
                      color='white', fontsize=9)
        ax7.legend(fontsize=6, facecolor='#222', edgecolor='#444', labelcolor='#ccc')

    # ── P8: Surrogate ───────────────────────────────────────────────────
    ax8 = dax(gs[1, 3])
    if len(d8_orig) > 0 and len(d8_surr) > 0:
        bp = ax8.boxplot([d8_orig, d8_surr], positions=[1, 2], widths=0.5,
                         patch_artist=True, showfliers=True,
                         flierprops=dict(marker='.', ms=3, mfc='#666'))
        for patch, c in zip(bp['boxes'], ['#44aaff', '#ff8844']):
            patch.set_facecolor(c); patch.set_alpha(0.6); patch.set_edgecolor('#888')
        for el in ['whiskers', 'caps', 'medians']:
            for line in bp[el]:
                line.set_color('#ccc')
        ax8.set_xticks([1, 2])
        ax8.set_xticklabels(['Original', 'IAAFT'], color='#ccc', fontsize=8)
        ax8.set_ylabel('Enrichment', color='#ccc', fontsize=8)
        if len(d8_orig) >= 3 and len(d8_surr) >= 3:
            _, p = stats.ttest_ind(d8_orig, d8_surr)
            verdict = 'spectral' if p > 0.05 else 'nonlinear'
            ax8.set_title(f'D8: Surrogate (p={p:.3f}, {verdict})',
                          color='white', fontsize=9)
    else:
        ax8.set_title('D8: Surrogate', color='white', fontsize=9)

    out = os.path.join(FIG_DIR, 'eeg_phi.png')
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"\nFigure saved: {out}")


def make_figure_d9(u_d1, targeted, targets, sweep_obs, sweep_null_mean,
                   sweep_null_95, width_results, V_raw, p_raw, p_rot,
                   ratio_618, f0_g, r_g, heatmap_618, best_f0_618,
                   best_r_618):
    """D9 response figure: noble-position enrichment tests."""
    inv_phi = 1.0 / PHI
    fig = plt.figure(figsize=(20, 10), facecolor='#111111')
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35,
                           left=0.05, right=0.96, top=0.91, bottom=0.08)
    fig.suptitle("D9: Noble-Position Enrichment — u=1/φ vs u=0.5",
                 color='white', fontsize=14, fontweight='bold')

    def dax(pos):
        ax = fig.add_subplot(pos)
        ax.set_facecolor('#181818')
        for s in ax.spines.values():
            s.set_color('#444444')
        ax.tick_params(colors='#cccccc', labelsize=7)
        return ax

    # ── P1: Phase histogram with both targets marked ────────────────────
    ax1 = dax(gs[0, 0])
    bins = np.linspace(0, 1, 51)
    ax1.hist(u_d1, bins=bins, color='#44aaff', alpha=0.7, density=True,
             edgecolor='#333', lw=0.4)
    ax1.axhline(1.0, color='#666', lw=1, ls=':', label='Uniform')
    ax1.axvline(0.5, color='#ff6666', lw=2, ls='--', label='u=0.5 (PKL)')
    ax1.axvline(inv_phi, color='#44ff44', lw=2, ls='--', label=f'u=1/φ={inv_phi:.3f}')
    # Shade both windows (width=0.05)
    ax1.axvspan(0.5 - 0.05, 0.5 + 0.05, alpha=0.10, color='red')
    ax1.axvspan(inv_phi - 0.05, inv_phi + 0.05, alpha=0.10, color='green')
    # Annotate targeted test results
    for label, (tgt, w, obs, nm, ns, p) in targeted.items():
        if 'w=0.05' in label:
            x = tgt
            y_off = 1.15 if tgt > 0.55 else 1.25
            ax1.annotate(f'p={p:.4f}', xy=(x, y_off), fontsize=7,
                         color='#44ff44' if tgt > 0.55 else '#ff6666',
                         ha='center', va='bottom')
    ax1.set_xlabel('u = log_φ(f/f₀) mod 1', color='#ccc', fontsize=8)
    ax1.set_ylabel('Density', color='#ccc', fontsize=8)

    V_str = f'Kuiper V={V_raw:.3f}'
    k_color = '#44ff44' if p_rot < 0.05 else '#ff6666'
    ax1.set_title(f'Phase distribution  ({V_str}, p_rot={p_rot:.4f})',
                  color='white', fontsize=9)
    ax1.legend(fontsize=6, facecolor='#222', edgecolor='#444', labelcolor='#ccc',
               loc='upper left')

    # ── P2: Phase target sweep ──────────────────────────────────────────
    ax2 = dax(gs[0, 1])
    ax2.fill_between(targets, sweep_null_mean, sweep_null_95,
                     color='#ff6666', alpha=0.15, label='Null 95% CI')
    ax2.plot(targets, sweep_null_mean, color='#ff6666', lw=0.5, alpha=0.5)
    ax2.plot(targets, sweep_obs, color='#44aaff', lw=1, label='Observed')
    ax2.axvline(0.5, color='#ff6666', lw=1.5, ls='--', alpha=0.7, label='u=0.5')
    ax2.axvline(inv_phi, color='#44ff44', lw=1.5, ls='--', alpha=0.7,
                label=f'u=1/φ')
    best_t = targets[np.argmax(sweep_obs - sweep_null_mean)]
    ax2.axvline(best_t, color='#ffaa00', lw=1.5, ls=':', alpha=0.7,
                label=f'Best u={best_t:.3f}')
    ax2.set_xlabel('Target phase position u', color='#ccc', fontsize=8)
    ax2.set_ylabel('Enrichment (w=0.05)', color='#ccc', fontsize=8)
    ax2.set_title('Phase target sweep', color='white', fontsize=9)
    ax2.legend(fontsize=5.5, facecolor='#222', edgecolor='#444', labelcolor='#ccc')

    # ── P3: D4 heatmap with u=0.618 metric ─────────────────────────────
    ax3 = dax(gs[0, 2])
    extent = [f0_g[0], f0_g[-1], r_g[0], r_g[-1]]
    vmax = max(abs(heatmap_618.min()), abs(heatmap_618.max()))
    im = ax3.imshow(heatmap_618, aspect='auto', origin='lower', extent=extent,
                    cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax3.plot(F0_CLAIMED, PHI, 'w*', markersize=12,
             label=f'Claimed ({F0_CLAIMED}, φ)')
    ax3.plot(best_f0_618, best_r_618, 'g+', markersize=10, mew=2,
             label=f'Best ({best_f0_618:.1f}, {best_r_618:.2f})')
    ax3.set_xlabel('f₀ (Hz)', color='#ccc', fontsize=8)
    ax3.set_ylabel('Ratio r', color='#ccc', fontsize=8)
    ax3.set_title('D4 rescan (u=0.618, w=0.05)', color='white', fontsize=9)
    ax3.legend(fontsize=6, facecolor='#222', edgecolor='#444', labelcolor='#ccc',
               loc='upper right')
    cb = fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=6, colors='#ccc')

    # ── P4: Width sensitivity ───────────────────────────────────────────
    ax4 = dax(gs[1, 0])
    ws = [wr[0] for wr in width_results]
    ps = [wr[3] for wr in width_results]
    obs_vals = [wr[1] for wr in width_results]
    null_vals = [wr[2] for wr in width_results]
    ax4.semilogy(ws, ps, 'o-', color='#44aaff', lw=1.5, ms=5, label='p-value')
    ax4.axhline(0.05, color='#ff6666', lw=1, ls='--', label='p=0.05')
    ax4.axhline(0.01, color='#ff4444', lw=1, ls=':', alpha=0.5, label='p=0.01')
    ax4.set_xlabel('Window half-width', color='#ccc', fontsize=8)
    ax4.set_ylabel('p-value (log scale)', color='#ccc', fontsize=8)
    ax4.set_title('Width sensitivity at u=0.618', color='white', fontsize=9)
    ax4.legend(fontsize=6, facecolor='#222', edgecolor='#444', labelcolor='#ccc')
    ax4.set_ylim(bottom=1e-4, top=1.1)

    # ── P5: D2 re-ranking comparison ────────────────────────────────────
    ax5 = dax(gs[1, 1])
    sorted_names = sorted(ratio_618, key=lambda n: -ratio_618[n][3])
    excesses = [ratio_618[n][3] for n in sorted_names]
    p_vals = [ratio_618[n][4] for n in sorted_names]
    colors5 = ['#ff4444' if n == 'φ' else '#44ff44' if p < 0.05
                else '#888888' for n, p in zip(sorted_names, p_vals)]
    y_pos = range(len(sorted_names))
    ax5.barh(y_pos, excesses, color=colors5, alpha=0.7, edgecolor='#333', lw=0.4)
    ax5.set_yticks(y_pos)
    ax5.set_yticklabels(sorted_names, fontsize=7, color='#ccc')
    ax5.axvline(0, color='#444', lw=0.5)
    ax5.set_xlabel('Excess enrichment over null', color='#ccc', fontsize=8)
    ax5.set_title('D2 re-ranking (u=0.618, w=0.05)', color='white', fontsize=9)
    # Annotate p-values
    for i, (name, ex, p) in enumerate(zip(sorted_names, excesses, p_vals)):
        ax5.text(max(ex, 0) + 0.001, i, f'p={p:.3f}', va='center',
                 fontsize=6, color='#ccc')

    # ── P6: Summary ────────────────────────────────────────────────────
    ax6 = dax(gs[1, 2])
    ax6.axis('off')
    lines = [
        "SUMMARY — D9: Noble-Position Test",
        "",
        "Critique: enrichment should be tested",
        f"at u=1/φ={inv_phi:.4f}, not u=0.5",
        "",
    ]
    for label, (tgt, w, obs, nm, ns, p) in targeted.items():
        status = "SIG" if p < 0.05 else "n.s."
        lines.append(f"{label}: p={p:.4f} [{status}]")
    lines.append("")

    phi_rank = sum(1 for v in ratio_618.values() if v[3] > ratio_618['φ'][3]) + 1
    lines.append(f"φ rank (u=0.618 metric): #{phi_rank}/{len(ratio_618)}")
    lines.append(f"D4 optimum: f₀={best_f0_618:.1f}, r={best_r_618:.2f}")
    lines.append(f"  vs claimed: f₀=7.5, r=φ={PHI:.3f}")
    lines.append("")
    lines.append(f"Kuiper V={V_raw:.4f}")
    lines.append(f"  p(asymptotic)={p_raw:.2e}")
    lines.append(f"  p(phase-rot)={p_rot:.4f}")

    text = '\n'.join(lines)
    ax6.text(0.05, 0.95, text, transform=ax6.transAxes, fontsize=8,
             color='#cccccc', fontfamily='monospace', verticalalignment='top')

    out = os.path.join(FIG_DIR, 'eeg_phi_d9.png')
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"\nD9 figure saved: {out}")


# =========================================================================
# MAIN
# =========================================================================
def main():
    print("=" * 78)
    print("INVESTIGATION: GOLDEN RATIO ARCHITECTURE IN EEG (deep)")
    print("=" * 78)
    print(f"φ = {PHI:.10f}")
    print(f"Claimed f₀ = {F0_CLAIMED} Hz")
    print(f"Range: {FREQ_LO}–{FREQ_HI} Hz")
    print(f"Subjects: {N_SUBJECTS}")

    # ── Load ──
    all_data = load_eeg_data()
    if not all_data:
        return

    # ── Extract peaks ──
    print("\nExtracting spectral peaks...")
    all_peaks = []
    eeg_signals = []
    per_subject = {}

    for subj, ch, sig in all_data:
        eeg_signals.append(sig)
        peaks = extract_peaks(sig)
        all_peaks.extend(peaks)
        per_subject.setdefault(subj, []).extend(peaks)

    all_peaks = np.array(all_peaks)
    print(f"  Total peaks: {len(all_peaks)}")
    print(f"  Per channel: {len(all_peaks) / len(all_data):.1f}")
    if len(all_peaks) > 0:
        print(f"  Range: {all_peaks.min():.1f}–{all_peaks.max():.1f} Hz")
        print(f"  Median: {np.median(all_peaks):.1f} Hz")

        # Peak frequency quantiles
        for q in [10, 25, 50, 75, 90]:
            print(f"    {q}th pctl: {np.percentile(all_peaks, q):.1f} Hz")

    if len(all_peaks) < 100:
        print("FATAL: too few peaks"); return

    # ── D1 ──
    u_d1, obs_d1, null_d1 = direction_1(all_peaks)

    # ── D2 ──
    named_d2, sweep_r, sweep_obs, null_95_d2, null_mean_d2 = direction_2(all_peaks)

    # ── D3 ──
    f0_vals, f0_scores, null_95_d3, null_mean_d3, best_f0 = direction_3(all_peaks)

    # ── D4 ──
    f0_grid, r_grid, heatmap, best_f0_4, best_r_4 = direction_4(all_peaks)

    # ── D5 ──
    band_results = direction_5(all_peaks)

    # ── D6 ──
    subj_obs, subj_null, subj_excess = direction_6(per_subject)

    # ── D8 ──
    d8_orig, d8_surr = direction_8(eeg_signals)

    # ── D9 ──
    d9 = direction_9(all_peaks)
    (d9_targeted, d9_targets, d9_sweep_obs, d9_sweep_null_mean,
     d9_sweep_null_95, d9_width_results, d9_V, d9_p_raw, d9_p_rot,
     d9_ratio_618, d9_f0_g, d9_r_g, d9_heatmap, d9_best_f0,
     d9_best_r) = d9

    # ── Figure ──
    make_figure(all_peaks, u_d1, obs_d1, null_d1,
                named_d2, sweep_r, sweep_obs, null_95_d2, null_mean_d2,
                f0_vals, f0_scores, null_95_d3, null_mean_d3, best_f0,
                f0_grid, r_grid, heatmap, best_f0_4, best_r_4,
                band_results,
                subj_obs, subj_null, subj_excess,
                d8_orig, d8_surr,
                eeg_signals[0])

    # ── Summary ──
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    p_d1 = np.mean(null_d1 >= obs_d1)
    print(f"  Peaks: {len(all_peaks)} from {len(all_data)} channels")
    print(f"  D1 enrichment: {obs_d1:+.3f} (p={p_d1:.4f} vs phase-rotation null)")

    phi_excess = named_d2['φ'][1] - named_d2['φ'][2]
    better = sum(1 for v in named_d2.values() if (v[1] - v[2]) > phi_excess)
    print(f"  D2 φ excess: {phi_excess:+.3f}, rank #{better + 1}/{len(named_d2)}")

    print(f"  D3 best f₀: {best_f0:.2f} Hz (claimed 7.5)")
    print(f"  D4 global optimum: f₀={best_f0_4:.2f}, r={best_r_4:.3f}")

    sig_bands = [b for b, v in band_results.items() if v[5] < 0.05]
    print(f"  D5 significant bands: {sig_bands if sig_bands else 'none'}")

    if len(subj_excess) > 0:
        t, p = stats.ttest_1samp(subj_excess, 0)
        print(f"  D6 per-subject excess: {subj_excess.mean():+.3f} (p={p:.4f})")

    if len(d8_orig) >= 3 and len(d8_surr) >= 3:
        _, p = stats.ttest_ind(d8_orig, d8_surr)
        print(f"  D8 surrogate: p={p:.4f}")

    # D9 summary
    inv_phi = 1.0 / PHI
    print(f"\n  D9 noble-position (u=1/φ={inv_phi:.4f}):")
    for label, (tgt, w, obs, nm, ns, p) in d9_targeted.items():
        status = "SIG" if p < 0.05 else "n.s."
        print(f"    {label}: p={p:.4f} [{status}]")
    phi_rank = sum(1 for v in d9_ratio_618.values()
                   if v[3] > d9_ratio_618['φ'][3]) + 1
    print(f"    φ rank (u=0.618 metric): #{phi_rank}/{len(d9_ratio_618)}")
    print(f"    D4 optimum: f₀={d9_best_f0:.1f}, r={d9_best_r:.2f}")
    print(f"    Kuiper V={d9_V:.4f}, p(phase-rot)={d9_p_rot:.4f}")

    # ── D9 Figure ──
    make_figure_d9(u_d1, d9_targeted, d9_targets, d9_sweep_obs,
                   d9_sweep_null_mean, d9_sweep_null_95, d9_width_results,
                   d9_V, d9_p_raw, d9_p_rot, d9_ratio_618,
                   d9_f0_g, d9_r_g, d9_heatmap, d9_best_f0, d9_best_r)

    # ── D10 ──
    d10 = direction_10(all_data)
    if d10:
        print(f"\n  D10 FOOOF comparison:")
        for label in ['u=0.618 w=0.05', 'u=0.500 w=0.15']:
            f = d10['fooof_results'].get(label)
            m = d10['medfilt_results'].get(label)
            if f and m:
                print(f"    {label}: FOOOF p={f[5]:.4f}, medfilt p={m[5]:.4f}")
        kf = d10['kuiper_fooof']
        km = d10['kuiper_medfilt']
        print(f"    Kuiper phase-rot: FOOOF p={kf[2]:.4f}, medfilt p={km[2]:.4f}")


if __name__ == '__main__':
    main()
