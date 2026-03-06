#!/usr/bin/env python
"""Re-analyze all replication CSVs with current code (degree-3 cross-base).

Reports the degree-3 coverage-normalized d̄/d̄_null and phi rank for each dataset.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))

import pandas as pd
from phi_replication import run_statistics

# ── All datasets ──
DATASETS = {
    # LEMON
    'LEMON EC [OT]': 'exports_lemon/replication/EC/per_subject_dominant_peaks.csv',
    'LEMON EO [OT]': 'exports_lemon/replication/EO/per_subject_dominant_peaks.csv',
    'LEMON EC [STD]': 'exports_lemon/replication/EC/sensitivity_standard/per_subject_dominant_peaks.csv',

    # EEGMMIDB
    'EEGMMIDB Comb [OT]': 'exports_eegmmidb/replication/combined/per_subject_dominant_peaks.csv',
    'EEGMMIDB EC [OT]': 'exports_eegmmidb/replication/EC/per_subject_dominant_peaks.csv',
    'EEGMMIDB EO [OT]': 'exports_eegmmidb/replication/EO/per_subject_dominant_peaks.csv',
    'EEGMMIDB EC [STD]': 'exports_eegmmidb/replication/EC/sensitivity_standard/per_subject_dominant_peaks.csv',

    # Dortmund v2 (fresh extraction)
    'Dort EC-pre [OT]': '/Volumes/T9/dortmund_data/lattice_results_replication_v2/EyesClosed_pre/per_subject_dominant_peaks.csv',
    'Dort EO-pre [OT]': '/Volumes/T9/dortmund_data/lattice_results_replication_v2/EyesOpen_pre/per_subject_dominant_peaks.csv',
    'Dort EC-post [OT]': '/Volumes/T9/dortmund_data/lattice_results_replication_v2/EyesClosed_post/per_subject_dominant_peaks.csv',
    'Dort EO-post [OT]': '/Volumes/T9/dortmund_data/lattice_results_replication_v2/EyesOpen_post/per_subject_dominant_peaks.csv',
    'Dort EC-pre [STD]': '/Volumes/T9/dortmund_data/lattice_results_replication_v2/EyesClosed_pre/sensitivity_standard/per_subject_dominant_peaks.csv',
}

os.chdir(os.path.join(os.path.dirname(__file__), '..'))

all_results = {}
for label, path in DATASETS.items():
    if not os.path.exists(path):
        print(f"\n  SKIP: {label} — file not found: {path}")
        continue

    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")

    df = pd.read_csv(path)
    print(f"  N = {len(df)}")

    stats = run_statistics(df, label=label)
    all_results[label] = stats

# ── Summary table ──
print(f"\n\n{'='*80}")
print(f"  DEGREE-3 CROSS-BASE COMPARISON — ALL DATASETS")
print(f"{'='*80}\n")

header = f"{'Dataset':35s} {'N':>5s} {'d(deg2)':>8s} {'phi_z_rank':>11s} {'phi_raw_rank':>13s} {'phi d/d_null':>12s} {'phi_z':>7s}"
print(header)
print('-' * len(header))

for label, s in all_results.items():
    d = s.get('cohen_d', float('nan'))
    pzr = s.get('phi_rank', '?')
    prr = s.get('phi_rank_raw', '?')

    # Get d/d_null ratio
    br = s.get('base_results', {})
    phi_br = br.get('phi', {})
    mean_d = phi_br.get('mean_d', float('nan'))
    null_mean = phi_br.get('null_mean', float('nan'))
    ratio = mean_d / null_mean if null_mean > 0 else float('nan')
    z = phi_br.get('z_score', float('nan'))
    n_pos = phi_br.get('n_positions', '?')

    n = s.get('n_complete', '?')

    print(f"{label:35s} {str(n):>5s} {d:8.3f} {str(pzr)+'/9':>11s} {str(prr)+'/9':>13s} {ratio:12.3f} {z:7.2f}")
