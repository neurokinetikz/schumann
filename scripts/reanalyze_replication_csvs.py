#!/usr/bin/env python
"""Re-analyze existing replication dominant-peak CSVs through the unified protocol.

Loads per_subject_dominant_peaks.csv files from prior extraction runs and
re-runs statistics + 14-position enrichment using the current lib/phi_replication.py
code (degree-2 cross-base with null-adjusted z-scores).

No FOOOF extraction needed — just statistics on already-computed peaks.
"""
import sys
sys.path.insert(0, './lib')

import pandas as pd
from phi_replication import run_full_protocol

# ── Datasets to re-analyze ──
DATASETS = {
    # LEMON
    'LEMON EC [OT]': 'exports_lemon/replication/EC/per_subject_dominant_peaks.csv',
    'LEMON EO [OT]': 'exports_lemon/replication/EO/per_subject_dominant_peaks.csv',
    'LEMON EC [STD]': 'exports_lemon/replication/EC/sensitivity_standard/per_subject_dominant_peaks.csv',

    # EEGMMIDB
    'EEGMMIDB Combined [OT]': 'exports_eegmmidb/replication/combined/per_subject_dominant_peaks.csv',
    'EEGMMIDB EC [OT]': 'exports_eegmmidb/replication/EC/per_subject_dominant_peaks.csv',
    'EEGMMIDB EO [OT]': 'exports_eegmmidb/replication/EO/per_subject_dominant_peaks.csv',
    'EEGMMIDB EC [STD]': 'exports_eegmmidb/replication/EC/sensitivity_standard/per_subject_dominant_peaks.csv',

    # Dortmund
    'Dortmund EC-pre [OT]': '/Volumes/T9/dortmund_data/lattice_results_replication/EyesClosed_pre/per_subject_dominant_peaks.csv',
    'Dortmund EC-pre [STD]': '/Volumes/T9/dortmund_data/lattice_results_replication/EyesClosed_pre/sensitivity_standard/per_subject_dominant_peaks.csv',
}

# ── Run each ──
all_results = {}
for label, path in DATASETS.items():
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"  {path}")
    print(f"{'='*70}")

    df = pd.read_csv(path)
    print(f"  N = {len(df)}")

    result = run_full_protocol(df, label=label)
    all_results[label] = result

# ── Summary comparison table ──
print(f"\n\n{'='*70}")
print(f"  CROSS-DATASET COMPARISON")
print(f"{'='*70}\n")

header = f"{'Dataset':35s} {'N':>5s} {'d':>6s} {'p_t':>10s} {'p_pop':>8s} {'phi_rank':>9s} {'noble%':>7s}"
print(header)
print('-' * len(header))

for label, result in all_results.items():
    s = result.get('stats', {})
    d = s.get('cohen_d', float('nan'))
    pt = s.get('p_ttest', float('nan'))
    pp = s.get('p_pop', float('nan'))
    pr = s.get('phi_rank', '?')
    nc = s.get('noble_contrib', float('nan'))
    print(f"{label:35s} {d:6.3f} {pt:10.2e} {pp:8.4f} {str(pr):>9} {nc:6.1f}%")
