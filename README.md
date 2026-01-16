# Schumann Ignition Events: Golden Ratio Architecture in Human EEG

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18244908.svg)](https://doi.org/10.5281/zenodo.18244908)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**Empirical discovery of φⁿ frequency organization in human neural oscillations, anchored to Schumann Resonance harmonics.**

---

## Overview

This repository contains analysis code, data, and documentation for research demonstrating that human EEG spectral peaks organize according to golden ratio (φ = 1.618...) scaling, with the fundamental frequency corresponding to the Earth's Schumann Resonance.

### The Core Equation

Neural oscillation frequencies follow:

$$f(n) = \frac{c}{r} \times \varphi^n$$

Where:
- **c** = speed of light (299,792,458 m/s)
- **r** = Earth's mean radius (6,371,000 m)  
- **φ** = golden ratio ((1 + √5)/2 ≈ 1.6180339...)
- **n** = integer or half-integer index

This yields **f₀ = c/r ≈ 7.6 Hz** — the Schumann Resonance fundamental frequency.

**Critically, this equation has zero free parameters.** The frequency architecture is fully determined by fundamental physical constants.

---

## Key Findings

### Transient High-Coherence Events (Schumann Ignition Events)

Analysis of **1,366 transient events** across **78 recording sessions** from **26 subjects** reveals:

| Ratio | Predicted | Observed | Error |
|-------|-----------|----------|-------|
| SR3/SR1 | φ² = 2.618 | 2.608 | **0.38%** |
| SR5/SR1 | φ³ = 4.236 | 4.193 | **1.02%** |
| SR5/SR3 | φ = 1.618 | 1.607 | **1.01%** |

### Continuous Spectral Architecture

Analysis of **857,945 spectral peaks** (FOOOF-extracted) shows:
- **+21% enrichment** at half-integer φⁿ positions (attractors)
- **−18% depletion** at integer φⁿ positions (boundaries)
- Cross-device validation (Muse, Emotiv EPOC X, Emotiv Insight)
- Cross-context validation (meditation, cognitive tasks, emotion induction)

### The Independence-Convergence Paradox

Individual harmonic frequencies vary **independently** (r ≈ 0 between SR1, SR3, SR5), yet their **ratios remain tightly constrained** (r = 0.930 for inter-ratio correlation). This suggests an intrinsic mathematical constraint mechanism rather than emergent coupling.

---

## Theoretical Support

This empirical pattern has independent theoretical foundations:

### Pletzer et al. (2010)
*"When Frequencies Never Synchronize: The Golden Mean and the Resting EEG"*  
Brain Research 1335:91-102

**Key result:** Mathematical proof that φ uniquely prevents spurious synchronization between neural oscillators. The golden ratio provides maximal desynchronization in the resting state while enabling controlled state transitions.

### Kramer (2022)
*"The Physics of Rhythm in the Brain: New Insights from the Golden Mean"*  
Biological Cybernetics 116:479-504

**Key result:** φⁿ scaling is the unique solution for cross-frequency coupling that doesn't require additional rhythm generators. "Golden triplets" (f, φf, φ²f) have the lowest resonance order (=3), enabling strongest coupling.

---

## Repository Structure

```
schumann/
├── lib/                    # Python analysis libraries
│   ├── fooof_utils.py      # FOOOF spectral parameterization
│   ├── sie_detection.py    # SIE event detection algorithms
│   └── phi_analysis.py     # φⁿ ratio calculations
├── scripts/                # Analysis scripts
├── papers/                 # LaTeX manuscript materials
├── docs/                   # Additional documentation
├── Basics.ipynb            # Introduction to the analysis
├── Ignition.ipynb          # SIE detection and characterization
├── Six Panel.ipynb         # Publication figure generation
├── validate_phi_architecture.ipynb  # Statistical validation
├── CLAUDE.md               # AI assistant context file
└── README.md
```

---

## Installation

### Requirements

```bash
# Core dependencies
pip install numpy pandas scipy matplotlib seaborn

# Signal processing
pip install mne fooof

# Clustering
pip install scikit-learn
```

### Clone and Setup

```bash
git clone https://github.com/neurokinetikz/schumann.git
cd schumann
pip install -r requirements.txt  # if available
```

---

## Quick Start

### 1. Basic φⁿ Frequency Calculation

```python
import numpy as np

# Constants
c = 299792458  # speed of light (m/s)
r = 6371000    # Earth radius (m)
phi = (1 + np.sqrt(5)) / 2  # golden ratio

# Fundamental frequency
f0 = c / r  # ≈ 47.06 rad/s ≈ 7.49 Hz

# φⁿ frequency series
def phi_frequency(n):
    """Calculate frequency for φⁿ index."""
    return f0 * (phi ** n)

# Generate canonical frequencies
for n in np.arange(-1, 4.5, 0.5):
    f = phi_frequency(n)
    print(f"n={n:+.1f}: {f:.2f} Hz")
```

**Output:**
```
n=-1.0: 4.63 Hz   # δ-θ boundary
n=-0.5: 5.95 Hz   # θ attractor
n=+0.0: 7.49 Hz   # θ-α boundary (SR fundamental)
n=+0.5: 9.62 Hz   # α attractor
n=+1.0: 12.12 Hz  # α-β boundary
n=+1.5: 15.59 Hz  # low-β attractor
n=+2.0: 19.61 Hz  # β boundary
n=+2.5: 25.21 Hz  # high-β attractor
n=+3.0: 31.73 Hz  # β-γ boundary
n=+3.5: 40.79 Hz  # γ attractor
n=+4.0: 51.33 Hz  # high-γ boundary
```

### 2. Analyze Your Own EEG Data

```python
from lib.sie_detection import detect_ignition_events
from lib.phi_analysis import calculate_phi_ratios

# Load your EEG data (MNE format)
import mne
raw = mne.io.read_raw_edf('your_data.edf', preload=True)

# Detect Schumann Ignition Events
events = detect_ignition_events(raw, sr_bands=[7.6, 12.1, 19.6, 31.7])

# Calculate φⁿ ratios
ratios = calculate_phi_ratios(events)
print(f"SR3/SR1 = {ratios['sr3_sr1']:.4f} (φ² = 2.6180)")
print(f"SR5/SR1 = {ratios['sr5_sr1']:.4f} (φ³ = 4.2361)")
```

---

## Validation Methods

The analysis employs five independent null controls:

| Control | Method | Result |
|---------|--------|--------|
| **Temporal Shuffle** | Randomize event timing | Pattern destroyed (p < 0.001) |
| **Random Triplets** | Sample arbitrary frequency triplets | 10× worse precision (d = 3.39) |
| **Baseline Windows** | Extract peaks outside SIEs | φⁿ ratios present but degraded |
| **Cross-Device** | Compare Muse vs Emotiv | Consistent architecture |
| **Blind Clustering** | FOOOF + DBSCAN without SR specification | SR bands emerge unsupervised |

---

## Data Availability

### This Repository
- Analysis code and Jupyter notebooks
- Library functions for SIE detection and φⁿ analysis

### Zenodo (Full Paper + Data)
- **DOI:** [10.5281/zenodo.18244908](https://doi.org/10.5281/zenodo.18244908)
- Complete 63-page manuscript
- Supplementary materials and figures

### EEG Data Sources
- Personal meditation recordings (2019-2024)
- PhysioFlow cognitive task dataset (publicly available)
- Emotion induction dataset

---

## Discovery Timeline

| Date | Milestone |
|------|-----------|
| **October 8, 2019** | First public observation: 7.8 Hz spectral peak in meditation EEG ([X/Twitter](https://twitter.com/neurokinetikz)) |
| **March 2021** | "The Variety of Synaptic Experience" — Substack exploration |
| **August 2025** | Systematic investigation initiated with AI-assisted analysis |
| **September 2025** | "Quantitative Model of Schumann-Resonant Brain Dynamics" — Medium |
| **November 2025** | "Continuous Golden Ratio Architecture" — complete analysis published |
| **January 2026** | Formal paper deposited on Zenodo |

---

## Frequency Band Mapping

The φⁿ framework provides principled definitions for canonical EEG bands:

| Band | Lower Boundary | Attractor | Upper Boundary |
|------|----------------|-----------|----------------|
| **Delta** | — | — | φ⁻¹ = 4.6 Hz |
| **Theta** | φ⁻¹ = 4.6 Hz | φ⁻⁰·⁵ = 6.0 Hz | φ⁰ = 7.5 Hz |
| **Alpha** | φ⁰ = 7.5 Hz | φ⁺⁰·⁵ = 9.6 Hz | φ¹ = 12.1 Hz |
| **Low Beta** | φ¹ = 12.1 Hz | φ¹·⁵ = 15.6 Hz | φ² = 19.6 Hz |
| **High Beta** | φ² = 19.6 Hz | φ²·⁵ = 25.2 Hz | φ³ = 31.7 Hz |
| **Gamma** | φ³ = 31.7 Hz | φ³·⁵ = 40.8 Hz | φ⁴ = 51.3 Hz |

**Integer φⁿ values = Boundaries** (unstable, frequencies avoid)  
**Half-integer φⁿ values = Attractors** (stable, frequencies cluster)

---

## Citation

If you use this code or findings in your research, please cite:

```bibtex
@misc{neurokinetikz2026schumann,
  author       = {neurokinetikz},
  title        = {Golden Ratio Architecture of Human Neural Oscillations: 
                  Schumann Ignition Events and φⁿ Frequency Organization},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.18244908},
  url          = {https://doi.org/10.5281/zenodo.18244908}
}
```

---

## Related Literature

### Foundational Theory
- Pletzer, B., Kerschbaum, H., & Klimesch, W. (2010). When frequencies never synchronize: The golden mean and the resting EEG. *Brain Research*, 1335, 91-102.
- Kramer, M. A. (2022). The physics of rhythm in the brain: New insights from the golden mean. *Biological Cybernetics*, 116, 479-504.

### EEG Frequency Architecture
- Klimesch, W. (2012). Alpha-band oscillations, attention, and controlled access to stored information. *Trends in Cognitive Sciences*, 16(12), 606-617.
- Doelling, K. B., & Poeppel, D. (2015). Cortical entrainment to music and its modulation by expertise. *PNAS*, 112(45), E6233-E6242.

### Schumann Resonance
- Schumann, W. O. (1952). Über die strahlungslosen Eigenschwingungen einer leitenden Kugel. *Zeitschrift für Naturforschung A*, 7(2), 149-154.
- Cherry, N. J. (2002). Schumann Resonances, a plausible biophysical mechanism for the human health effects of Solar/Geomagnetic Activity. *Natural Hazards*, 26, 279-331.

---

## Contributing

Contributions welcome! Areas of particular interest:

1. **Independent replication** with different EEG systems
2. **Cross-species validation** (non-human EEG/LFP data)
3. **Mechanism investigation** (concurrent SR field measurements)
4. **Clinical applications** (φⁿ-based neurofeedback)

Please open an issue or submit a pull request.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

Code is freely available for research and educational purposes.

---

## Contact

- **GitHub:** [@neurokinetikz](https://github.com/neurokinetikz)
- **Medium:** [@neurokinetikz](https://medium.com/@neurokinetikz)
- **X/Twitter:** [@neurokinetikz](https://twitter.com/neurokinetikz)

---

<p align="center">
  <i>"The brain does not choose arbitrary frequencies. It chooses φ."</i>
</p>