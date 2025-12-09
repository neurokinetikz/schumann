# Library Module Documentation

Auto-generated documentation for all modules in the `lib/` directory.

## Summary

- **Total Modules:** 48
- **Total Functions:** 957
- **Public Functions:** 580+
- **Private Helpers:** 370+

## Module Categories

### Core Analysis
| Module | Description | Functions |
|--------|-------------|-----------|
| [utilities](utilities.md) | Data loading, filtering, PSD computation, visualization | 34 |
| [harmonics](harmonics.md) | Schumann spike detection using Morlet wavelets | 57 |
| [detect_ignition](detect_ignition.md) | Neural ignition event detection | 59 |

### Cross-Frequency Analysis
| Module | Description | Functions |
|--------|-------------|-----------|
| [cross_frequency](cross_frequency.md) | PAC, bicoherence, waveform shape | 30 |
| [cross_frequency_harmonics](cross_frequency_harmonics.md) | CFC at Schumann harmonic frequencies | 16 |
| [cross_frequency_region_coupling](cross_frequency_region_coupling.md) | Cross-region cross-frequency coupling | 24 |
| [pac_multiplexing](pac_multiplexing.md) | PAC vs Schumann activity index | 21 |

### Criticality & Complexity
| Module | Description | Functions |
|--------|-------------|-----------|
| [criticality](criticality.md) | 1/f slope, DFA exponents, avalanche statistics | 13 |
| [chaos_metrics](chaos_metrics.md) | RQA and chaos metrics | 18 |
| [multiscale_entropy_and_fractal_scaling](multiscale_entropy_and_fractal_scaling.md) | MSE and DFA multiscale analysis | 15 |

### Network & Connectivity
| Module | Description | Functions |
|--------|-------------|-----------|
| [network_geometry](network_geometry.md) | State-space embeddings (UMAP) | 19 |
| [network_coupling](network_coupling.md) | Cross-domain graph alignment | 12 |
| [network_graph_hubs](network_graph_hubs.md) | Graph metrics and hub analysis | 20 |
| [dynamic_connectivity_metastability](dynamic_connectivity_metastability.md) | Time-varying connectivity | 18 |

### Information Flow & Directionality
| Module | Description | Functions |
|--------|-------------|-----------|
| [information_flow](information_flow.md) | VAR, PDC, DTF, transfer entropy | 25 |
| [directed_connectivity](directed_connectivity.md) | Top-down ignition pipeline | 8 |
| [directional_coupling](directional_coupling.md) | dPLI and Granger causality | 11 |
| [causal_routing](causal_routing.md) | Directed connectivity routing | 24 |
| [directionality_harmonics](directionality_harmonics.md) | Directionality at harmonics | 14 |

### Schumann Coherence & Harmonics
| Module | Description | Functions |
|--------|-------------|-----------|
| [harmonic_coherence](harmonic_coherence.md) | SR ignition signatures | 8 |
| [harmonic_groups](harmonic_groups.md) | SR harmonic group analysis | 10 |
| [harmonic_locking](harmonic_locking.md) | Harmonic phase locking | 16 |
| [harmonic_resonance](harmonic_resonance.md) | Spectral mode analysis | 11 |
| [schumann_coherence](schumann_coherence.md) | EEG-Schumann coherence testing | 13 |
| [wavelet_coherence](wavelet_coherence.md) | Wavelet coherence (WTC) | 10 |
| [synchrosqueeze](synchrosqueeze.md) | Synchrosqueeze validation | 11 |

### Frequency Domain
| Module | Description | Functions |
|--------|-------------|-----------|
| [frequency_domain_coupling](frequency_domain_coupling.md) | Multi-taper MSC, PLV, SCF | 25 |
| [psd_waterfall](psd_waterfall.md) | PSD waterfall visualization | 24 |

### Attractor & Topology
| Module | Description | Functions |
|--------|-------------|-----------|
| [attractor_geometry](attractor_geometry.md) | TDA using persistent homology | 16 |
| [attractor_topology](attractor_topology.md) | Lyapunov exponents, correlation dimension | 19 |
| [toroidal_phase](toroidal_phase.md) | Toroidal phase-torus analysis | 12 |
| [emergent_geometry](emergent_geometry.md) | Phase metric embeddings | 16 |

### Connectome & Spatial
| Module | Description | Functions |
|--------|-------------|-----------|
| [connectome](connectome.md) | Connectome utilities | 2 |
| [connectome_harmonics](connectome_harmonics.md) | Connectome harmonics breadth | 19 |
| [resonant_modes](resonant_modes.md) | Resonant mode analysis | 15 |
| [spatial_source_harmonics](spatial_source_harmonics.md) | Spatial/source-level harmonics | 14 |
| [surface_cuts](surface_cuts.md) | Multi-seed surface cuts | 13 |

### Information & Entropy
| Module | Description | Functions |
|--------|-------------|-----------|
| [informational_geometry](informational_geometry.md) | Information geometry on manifolds | 18 |
| [entanglement_entropy](entanglement_entropy.md) | Integration analogs | 16 |
| [entanglement_geometry](entanglement_geometry.md) | Min-cut and PLV measures | 13 |

### State Analysis
| Module | Description | Functions |
|--------|-------------|-----------|
| [hidden_markov](hidden_markov.md) | ERP/ERSP/ITC, HMM states | 18 |
| [microstate_segmentation](microstate_segmentation.md) | EEG microstate analysis | 18 |
| [ignition_rebound](ignition_rebound.md) | Ignition vs rebound comparison | 6 |

### Temporal Analysis
| Module | Description | Functions |
|--------|-------------|-----------|
| [temporal_dynamics](temporal_dynamics.md) | Lead/lag temporal dynamics | 19 |
| [temporal_holography](temporal_holography.md) | Temporal holography | 8 |
| [shape_vs_resonance](shape_vs_resonance.md) | Waveform shape vs resonance | 21 |

### Other
| Module | Description | Functions |
|--------|-------------|-----------|
| [test](test.md) | Test utilities | 128 |
| [extra](extra.md) | Additional utilities | 0 |

## Usage

```python
import sys
sys.path.insert(0, './lib')

# Import specific modules
import utilities
import harmonics
import detect_ignition
import cross_frequency
import criticality
# ... etc
```

## Common Function Patterns

Most analysis functions follow this signature:
```python
def run_<analysis>(
    RECORDS: pd.DataFrame,           # EEG data
    ignition_windows: List[Tuple],   # [(start_sec, end_sec), ...]
    baseline_windows: List[Tuple],   # Optional baseline
    electrodes: List[str],           # Channel list
    time_col: str = 'Timestamp',
    out_dir: str = None,             # Export directory
    show: bool = True,               # Show plots
    **kwargs
) -> Dict[str, object]:
```
