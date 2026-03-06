"""
Phi Frequency Model: Complete φⁿ Position Framework
====================================================

Implements the complete phi-based frequency prediction model:
    f(n) = F0 × φⁿ  where φ = 1.6180339887, F0 = 7.60 Hz

Position Types (8 per φ-octave):
- Boundary (n + 0.000): Integer φ exponents, band edges
- 4° Noble (n + 0.146): φ⁻⁴ offset
- 3° Noble (n + 0.236): φ⁻³ offset
- 2° Noble (n + 0.382): 1 - φ⁻¹ offset
- Attractor (n + 0.500): Mid-octave stable point
- 1° Noble (n + 0.618): φ⁻¹ offset (most noble)
- 3° Inverse (n + 0.764): 1 - φ⁻³ offset
- 4° Inverse (n + 0.854): 1 - φ⁻⁴ offset

Dependencies: numpy, pandas
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field

# ============================================================================
# CONSTANTS
# ============================================================================

# Golden ratio and fundamental frequency
PHI = 1.6180339887
PHI_INV = 1.0 / PHI  # 0.6180339887
F0 = 7.60  # Fundamental frequency (Hz)

# Position offsets within each φ-octave (ordered by offset value)
POSITION_OFFSETS = {
    'boundary':    0.000,
    'noble_4':     0.146,  # φ⁻⁴ ≈ 0.1459
    'noble_3':     0.236,  # φ⁻³ ≈ 0.2361
    'noble_2':     0.382,  # 1 - φ⁻¹ = 0.3820
    'attractor':   0.500,
    'noble_1':     0.618,  # φ⁻¹ = 0.6180
    'inv_noble_3': 0.764,  # 1 - φ⁻³ ≈ 0.7639
    'inv_noble_4': 0.854,  # 1 - φ⁻⁴ ≈ 0.8541
}

# Exact mathematical offsets (for validation)
POSITION_OFFSETS_EXACT = {
    'boundary':    0.0,
    'noble_4':     PHI_INV ** 4,           # 0.14589...
    'noble_3':     PHI_INV ** 3,           # 0.23606...
    'noble_2':     1 - PHI_INV,            # 0.38196...
    'attractor':   0.5,
    'noble_1':     PHI_INV,                # 0.61803...
    'inv_noble_3': 1 - PHI_INV ** 3,       # 0.76393...
    'inv_noble_4': 1 - PHI_INV ** 4,       # 0.85410...
}

# Position hierarchy for statistical grouping
POSITION_HIERARCHY = {
    'edges':       ['boundary'],
    'centers':     ['attractor'],
    'nobles':      ['noble_1', 'noble_2', 'noble_3', 'noble_4'],
    'inv_nobles':  ['inv_noble_3', 'inv_noble_4'],
    'all_nobles':  ['noble_1', 'noble_2', 'noble_3', 'noble_4', 'inv_noble_3', 'inv_noble_4'],
}

# Traditional EEG band definitions mapped to φ-octaves
BANDS = {
    'delta':     {'octave_range': (-6, -1), 'freq_range': (0.42, 4.70)},
    'theta':     {'octave_range': (-1, 0),  'freq_range': (4.70, 7.60)},
    'alpha':     {'octave_range': (0, 1),   'freq_range': (7.60, 12.30)},
    'beta_low':  {'octave_range': (1, 2),   'freq_range': (12.30, 19.90)},
    'beta_high': {'octave_range': (2, 3),   'freq_range': (19.90, 32.19)},
    'gamma':     {'octave_range': (3, 4),   'freq_range': (32.19, 52.09)},
}

# Recommended GED search bandwidths by band (Hz)
BAND_SEARCH_BW = {
    'delta':     0.3,
    'theta':     0.4,
    'alpha':     0.5,
    'beta_low':  0.7,
    'beta_high': 1.0,
    'gamma':     1.5,
}


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class PhiPrediction:
    """
    Single phi-based frequency prediction.

    Attributes
    ----------
    label : str
        Unique identifier, e.g., 'alpha_noble_1', 'theta_attractor'
    n : float
        φ exponent (e.g., 0.618 for alpha band 1° noble)
    frequency : float
        Predicted frequency in Hz = F0 × φⁿ
    position_type : str
        Position type: 'boundary', 'attractor', 'noble_1', etc.
    octave : int
        Which φ-octave this position is in (lower boundary)
    band : str
        Traditional EEG band name
    search_bw : float
        Recommended GED search bandwidth (Hz)
    offset : float
        Position offset within octave (0.0 to <1.0)
    """
    label: str
    n: float
    frequency: float
    position_type: str
    octave: int
    band: str
    search_bw: float
    offset: float = 0.0

    def __post_init__(self):
        """Compute offset from n if not provided."""
        if self.offset == 0.0 and self.n != int(self.n):
            self.offset = self.n - int(self.n) if self.n >= 0 else self.n - int(self.n) + 1

    @property
    def phi_power(self) -> float:
        """Return φⁿ (the multiplier)."""
        return PHI ** self.n

    def __repr__(self) -> str:
        return f"PhiPrediction({self.label}: {self.frequency:.2f} Hz, n={self.n:.3f}, {self.position_type})"


@dataclass
class PhiTable:
    """
    Complete table of phi frequency predictions.

    Provides dict-like access plus helper methods for filtering,
    searching, and statistical analysis.
    """
    predictions: Dict[str, PhiPrediction] = field(default_factory=dict)
    f0: float = F0
    octave_range: Tuple[int, int] = (-1, 4)
    freq_limits: Tuple[float, float] = (3.0, 55.0)

    def __getitem__(self, key: str) -> PhiPrediction:
        return self.predictions[key]

    def __iter__(self):
        return iter(self.predictions)

    def __len__(self) -> int:
        return len(self.predictions)

    def keys(self):
        return self.predictions.keys()

    def values(self):
        return self.predictions.values()

    def items(self):
        return self.predictions.items()

    def get(self, key: str, default=None) -> Optional[PhiPrediction]:
        return self.predictions.get(key, default)

    def by_position_type(self, position_type: str) -> List[PhiPrediction]:
        """Filter predictions by position type."""
        return [p for p in self.predictions.values() if p.position_type == position_type]

    def by_band(self, band: str) -> List[PhiPrediction]:
        """Filter predictions by EEG band."""
        return [p for p in self.predictions.values() if p.band == band]

    def by_octave(self, octave: int) -> List[PhiPrediction]:
        """Filter predictions by φ-octave."""
        return [p for p in self.predictions.values() if p.octave == octave]

    def in_freq_range(self, f_lo: float, f_hi: float) -> List[PhiPrediction]:
        """Filter predictions by frequency range."""
        return [p for p in self.predictions.values() if f_lo <= p.frequency <= f_hi]

    def nearest(self, freq: float) -> PhiPrediction:
        """Find the prediction nearest to a given frequency."""
        return min(self.predictions.values(), key=lambda p: abs(p.frequency - freq))

    def nearest_within(self, freq: float, tolerance: float = 0.05) -> Optional[PhiPrediction]:
        """
        Find nearest prediction within relative tolerance.

        Parameters
        ----------
        freq : float
            Query frequency (Hz)
        tolerance : float
            Relative tolerance (e.g., 0.05 = 5%)

        Returns
        -------
        PhiPrediction or None if no prediction within tolerance
        """
        nearest = self.nearest(freq)
        rel_diff = abs(nearest.frequency - freq) / freq
        return nearest if rel_diff <= tolerance else None

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame for analysis."""
        records = []
        for p in self.predictions.values():
            records.append({
                'label': p.label,
                'n': p.n,
                'frequency': p.frequency,
                'position_type': p.position_type,
                'octave': p.octave,
                'band': p.band,
                'search_bw': p.search_bw,
                'offset': p.offset,
            })
        return pd.DataFrame(records).sort_values('frequency').reset_index(drop=True)

    @property
    def frequencies(self) -> np.ndarray:
        """Array of all predicted frequencies (sorted)."""
        return np.array(sorted([p.frequency for p in self.predictions.values()]))

    @property
    def position_types(self) -> List[str]:
        """List of unique position types."""
        return list(set(p.position_type for p in self.predictions.values()))


# ============================================================================
# TABLE GENERATION
# ============================================================================

def _get_band_for_n(n: float) -> str:
    """Determine EEG band name for a given φ exponent."""
    for band_name, info in BANDS.items():
        lo, hi = info['octave_range']
        if lo <= n < hi:
            return band_name
    # Edge cases
    if n < -6:
        return 'sub_delta'
    if n >= 4:
        return 'high_gamma'
    return 'unknown'


def _get_search_bw(band: str, position_type: str) -> float:
    """
    Get recommended search bandwidth for GED.

    Returns uniform bandwidth for all position types to avoid
    circular reasoning in deviation comparisons.
    """
    # Uniform bandwidth - no position-type adjustments
    # This prevents artificial constraint of attractor search ranges
    # that would bias deviation comparisons
    return BAND_SEARCH_BW.get(band, 0.5)


def generate_phi_table(
    f0: float = F0,
    octave_range: Tuple[int, int] = (-1, 4),
    freq_limits: Tuple[float, float] = (3.0, 55.0),
    position_types: Optional[List[str]] = None,
    use_exact_offsets: bool = False
) -> PhiTable:
    """
    Generate complete phi-based frequency prediction table.

    Parameters
    ----------
    f0 : float
        Fundamental frequency (Hz), default 7.60
    octave_range : tuple
        (min_octave, max_octave) inclusive on lower, exclusive on upper
    freq_limits : tuple
        (min_freq, max_freq) to filter predictions
    position_types : list or None
        Specific position types to include, or None for all
    use_exact_offsets : bool
        If True, use mathematically exact offsets; if False, use rounded values

    Returns
    -------
    PhiTable
        Table of all predictions within range
    """
    offsets = POSITION_OFFSETS_EXACT if use_exact_offsets else POSITION_OFFSETS

    if position_types is None:
        position_types = list(offsets.keys())

    predictions = {}

    for octave in range(octave_range[0], octave_range[1]):
        band = _get_band_for_n(octave)

        for pos_type in position_types:
            if pos_type not in offsets:
                continue

            offset = offsets[pos_type]
            n = octave + offset
            freq = f0 * (PHI ** n)

            # Skip if outside frequency limits
            if freq < freq_limits[0] or freq > freq_limits[1]:
                continue

            # Generate unique label
            label = f"{band}_{pos_type}"

            # Handle duplicate labels (multiple octaves in same band)
            if label in predictions:
                label = f"{band}_n{octave}_{pos_type}"

            search_bw = _get_search_bw(band, pos_type)

            predictions[label] = PhiPrediction(
                label=label,
                n=n,
                frequency=round(freq, 2),
                position_type=pos_type,
                octave=octave,
                band=band,
                search_bw=search_bw,
                offset=offset,
            )

    return PhiTable(
        predictions=predictions,
        f0=f0,
        octave_range=octave_range,
        freq_limits=freq_limits,
    )


def generate_phi_table_detailed(
    f0: float = F0,
    octave_range: Tuple[int, int] = (-6, 5),
    freq_limits: Tuple[float, float] = (0.4, 60.0),
) -> pd.DataFrame:
    """
    Generate detailed prediction table matching supplemental format.

    Returns DataFrame with columns matching the supplemental tables:
    - Position Type, Offset, n Value, φⁿ, Frequency (Hz)
    """
    rows = []

    for octave in range(octave_range[0], octave_range[1]):
        for pos_type, offset in POSITION_OFFSETS.items():
            n = octave + offset
            phi_n = PHI ** n
            freq = f0 * phi_n

            if freq < freq_limits[0] or freq > freq_limits[1]:
                continue

            # Map position type to display name
            display_names = {
                'boundary': 'Boundary' if offset == 0 else 'Upper Boundary',
                'noble_4': '4° Noble',
                'noble_3': '3° Noble',
                'noble_2': '2° Noble',
                'attractor': 'Attractor',
                'noble_1': '1° Noble',
                'inv_noble_3': '3° Inverse',
                'inv_noble_4': '4° Inverse',
            }

            rows.append({
                'Position Type': display_names.get(pos_type, pos_type),
                'Offset': round(offset, 3),
                'n Value': round(n, 3),
                'phi_n': round(phi_n, 4),
                'Frequency (Hz)': round(freq, 2),
                'Band': _get_band_for_n(n),
                'Octave': octave,
                'position_key': pos_type,
            })

    df = pd.DataFrame(rows)
    return df.sort_values(['Octave', 'Offset']).reset_index(drop=True)


# ============================================================================
# POSITION MATCHING AND DISTANCE FUNCTIONS
# ============================================================================

def phi_distance(freq: float, f0: float = F0) -> Tuple[float, float, str]:
    """
    Compute distance from frequency to nearest φⁿ position.

    Parameters
    ----------
    freq : float
        Query frequency (Hz)
    f0 : float
        Fundamental frequency (Hz)

    Returns
    -------
    n_nearest : float
        Nearest φ exponent
    distance : float
        Absolute distance in n-space
    position_type : str
        Position type of nearest prediction
    """
    # Convert frequency to n-space
    n_observed = np.log(freq / f0) / np.log(PHI)

    # Find nearest position
    octave = int(np.floor(n_observed))
    fractional = n_observed - octave

    # Find nearest offset
    min_dist = float('inf')
    nearest_type = 'boundary'
    nearest_offset = 0.0

    for pos_type, offset in POSITION_OFFSETS.items():
        dist = abs(fractional - offset)
        # Also check wrap-around (fractional near 1.0 vs offset near 0.0)
        dist_wrap = min(dist, abs(fractional - offset - 1), abs(fractional - offset + 1))
        if dist_wrap < min_dist:
            min_dist = dist_wrap
            nearest_type = pos_type
            nearest_offset = offset

    n_nearest = octave + nearest_offset
    distance = abs(n_observed - n_nearest)

    return n_nearest, distance, nearest_type


def assign_position(
    freq: float,
    phi_table: PhiTable,
    tolerance: float = 0.05
) -> Tuple[Optional[PhiPrediction], float, bool]:
    """
    Assign a discovered frequency to its nearest phi position.

    Parameters
    ----------
    freq : float
        Discovered frequency (Hz)
    phi_table : PhiTable
        Table of predictions
    tolerance : float
        Relative tolerance for "aligned" classification

    Returns
    -------
    prediction : PhiPrediction or None
        Nearest prediction (None if outside tolerance)
    rel_distance : float
        Relative distance (|observed - predicted| / predicted)
    is_aligned : bool
        True if within tolerance
    """
    nearest = phi_table.nearest(freq)
    rel_distance = abs(freq - nearest.frequency) / nearest.frequency
    is_aligned = rel_distance <= tolerance

    return (nearest if is_aligned else None, rel_distance, is_aligned)


def batch_assign_positions(
    frequencies: np.ndarray,
    phi_table: PhiTable,
    tolerance: float = 0.05
) -> pd.DataFrame:
    """
    Assign multiple frequencies to phi positions.

    Returns DataFrame with columns:
    - observed_freq, nearest_label, predicted_freq, rel_distance, is_aligned, position_type
    """
    rows = []
    for freq in frequencies:
        nearest = phi_table.nearest(freq)
        rel_dist = abs(freq - nearest.frequency) / nearest.frequency
        aligned = rel_dist <= tolerance

        rows.append({
            'observed_freq': freq,
            'nearest_label': nearest.label,
            'predicted_freq': nearest.frequency,
            'rel_distance': rel_dist,
            'is_aligned': aligned,
            'position_type': nearest.position_type,
            'band': nearest.band,
            'n': nearest.n,
        })

    return pd.DataFrame(rows)


# ============================================================================
# RATIO VALIDATION
# ============================================================================

# Expected ratios between positions (powers of φ)
EXPECTED_RATIOS = {
    # Adjacent octave boundaries
    ('beta_low_boundary', 'alpha_boundary'): PHI ** 1,
    ('beta_high_boundary', 'beta_low_boundary'): PHI ** 1,
    ('gamma_boundary', 'beta_high_boundary'): PHI ** 1,
    ('alpha_boundary', 'theta_boundary'): PHI ** 1,

    # Within-octave ratios
    ('alpha_noble_1', 'alpha_boundary'): PHI ** 0.618,
    ('alpha_attractor', 'alpha_boundary'): PHI ** 0.5,

    # Cross-octave noble ratios
    ('beta_low_noble_1', 'alpha_noble_1'): PHI ** 1,
}


def validate_ratio(
    freq1: float,
    freq2: float,
    expected_phi_exp: float,
    tolerance: float = 0.03
) -> Tuple[float, float, bool]:
    """
    Validate if ratio of two frequencies matches φⁿ expectation.

    Parameters
    ----------
    freq1, freq2 : float
        Frequencies (Hz), freq1 > freq2 expected
    expected_phi_exp : float
        Expected φ exponent for the ratio
    tolerance : float
        Relative tolerance

    Returns
    -------
    observed_ratio : float
    expected_ratio : float
    is_valid : bool
    """
    observed_ratio = freq1 / freq2
    expected_ratio = PHI ** expected_phi_exp
    rel_error = abs(observed_ratio - expected_ratio) / expected_ratio

    return observed_ratio, expected_ratio, rel_error <= tolerance


def compute_ratio_matrix(frequencies: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute all pairwise frequency ratios and their φⁿ distances.

    Returns
    -------
    ratio_matrix : np.ndarray
        Pairwise ratios (upper triangular meaningful)
    phi_exp_matrix : np.ndarray
        Equivalent φ exponents for each ratio
    """
    n = len(frequencies)
    ratio_matrix = np.zeros((n, n))
    phi_exp_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if frequencies[j] > 0:
                ratio = frequencies[i] / frequencies[j]
                ratio_matrix[i, j] = ratio
                if ratio > 0:
                    phi_exp_matrix[i, j] = np.log(ratio) / np.log(PHI)

    return ratio_matrix, phi_exp_matrix


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def get_default_phi_table() -> PhiTable:
    """Get default phi table for EEG analysis (3-55 Hz)."""
    return generate_phi_table(
        f0=F0,
        octave_range=(-1, 4),
        freq_limits=(3.0, 55.0),
    )


def print_phi_table_summary(phi_table: PhiTable) -> None:
    """Print human-readable summary of phi table."""
    df = phi_table.to_dataframe()

    print(f"Phi Frequency Table (f0={phi_table.f0} Hz)")
    print("=" * 70)

    for band in df['band'].unique():
        band_df = df[df['band'] == band].sort_values('frequency')
        print(f"\n{band.upper()} Band ({len(band_df)} positions):")
        print("-" * 50)

        for _, row in band_df.iterrows():
            print(f"  {row['position_type']:12s}  n={row['n']:+.3f}  "
                  f"f={row['frequency']:6.2f} Hz  (bw={row['search_bw']:.2f})")


# ============================================================================
# VALIDATION: Verify against supplemental tables
# ============================================================================

def verify_against_supplemental() -> bool:
    """
    Verify generated frequencies match the supplemental tables exactly.

    Returns True if all frequencies match within 0.01 Hz.
    """
    # Expected values from supplemental Table S3 (Alpha Band, n=0 to 1)
    expected_alpha = {
        'boundary': 7.60,
        'noble_4': 8.15,
        'noble_3': 8.51,
        'noble_2': 9.13,
        'attractor': 9.67,
        'noble_1': 10.23,
        'inv_noble_3': 10.98,
        'inv_noble_4': 11.46,
    }

    phi_table = generate_phi_table(octave_range=(0, 1))

    all_match = True
    for pos_type, expected_freq in expected_alpha.items():
        label = f"alpha_{pos_type}"
        if label in phi_table:
            observed = phi_table[label].frequency
            if abs(observed - expected_freq) > 0.01:
                print(f"MISMATCH: {label} expected {expected_freq}, got {observed}")
                all_match = False
        else:
            print(f"MISSING: {label}")
            all_match = False

    return all_match


if __name__ == "__main__":
    # Verify against supplemental tables
    print("Verifying against supplemental tables...")
    if verify_against_supplemental():
        print("All frequencies match!")
    else:
        print("Some frequencies don't match - check implementation")

    print("\n")

    # Print full table
    phi_table = get_default_phi_table()
    print_phi_table_summary(phi_table)

    print("\n\nDetailed table (CSV format):")
    df = generate_phi_table_detailed(octave_range=(-1, 4), freq_limits=(3.0, 55.0))
    print(df.to_string())
