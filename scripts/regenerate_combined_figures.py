#!/usr/bin/env python3
"""
Regenerate GED validation combined figures from existing CSV data.
Uses v3 exports for physf, vep, and mpeng datasets.
"""

import sys
import os

# Add lib to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))

import pandas as pd
import numpy as np
from pathlib import Path

# Import the plotting function
from ged_validation_pipeline import plot_ged_validation_summary, plot_ignition_baseline_comparison

def main():
    base_dir = Path(__file__).parent.parent

    # Datasets to combine
    datasets = ['physf_v3', 'vep_v3', 'mpeng_v3']

    # Output directory
    output_dir = base_dir / 'exports_ged_validation_combined_v3'
    output_dir.mkdir(exist_ok=True)
    (output_dir / 'aggregate').mkdir(exist_ok=True)
    (output_dir / 'figures').mkdir(exist_ok=True)

    # Load and combine canonical data
    print("Loading canonical data...")
    canonical_dfs = []
    for ds in datasets:
        csv_path = base_dir / f'exports_ged_validation_{ds}' / 'aggregate' / 'all_windows_canonical.csv'
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            print(f"  {ds}: {len(df)} windows")
            canonical_dfs.append(df)
        else:
            print(f"  WARNING: {csv_path} not found")

    if not canonical_dfs:
        print("ERROR: No canonical data found!")
        return

    canonical_df = pd.concat(canonical_dfs, ignore_index=True)
    print(f"Combined canonical: {len(canonical_df)} total windows")

    # Load and combine blind peaks data
    print("\nLoading blind peaks data...")
    blind_dfs = []
    for ds in datasets:
        csv_path = base_dir / f'exports_ged_validation_{ds}' / 'aggregate' / 'all_blind_peaks.csv'
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            print(f"  {ds}: {len(df)} peaks")
            blind_dfs.append(df)
        else:
            print(f"  WARNING: {csv_path} not found")

    blind_df = pd.concat(blind_dfs, ignore_index=True) if blind_dfs else None
    if blind_df is not None:
        print(f"Combined blind peaks: {len(blind_df)} total peaks")

    # Save combined CSVs
    print("\nSaving combined CSVs...")
    canonical_df.to_csv(output_dir / 'aggregate' / 'all_windows_canonical.csv', index=False)
    if blind_df is not None:
        blind_df.to_csv(output_dir / 'aggregate' / 'all_blind_peaks.csv', index=False)

    # Generate the 8-panel summary figure
    print("\nGenerating 8-panel GED validation summary figure...")
    fig_path = output_dir / 'figures' / 'ged_validation_summary.png'
    pdf_path = output_dir / 'figures' / 'ged_validation_summary.pdf'

    fig = plot_ged_validation_summary(
        canonical_df=canonical_df,
        blind_df=blind_df,
        output_path=str(fig_path),
        show=False,
        dataset_name='PhySF + VEP + MPeng'
    )

    # Save PDF version
    fig.savefig(str(pdf_path), format='pdf', dpi=300, bbox_inches='tight')
    print(f"  Saved: {fig_path}")
    print(f"  Saved: {pdf_path}")

    # Check for baseline data
    print("\nChecking for baseline data...")
    baseline_dfs = []
    for ds in datasets:
        csv_path = base_dir / f'exports_ged_validation_{ds}' / 'aggregate' / 'all_windows_baseline.csv'
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            print(f"  {ds}: {len(df)} baseline windows")
            baseline_dfs.append(df)

    if baseline_dfs:
        baseline_df = pd.concat(baseline_dfs, ignore_index=True)
        print(f"Combined baseline: {len(baseline_df)} total windows")
        baseline_df.to_csv(output_dir / 'aggregate' / 'all_windows_baseline.csv', index=False)

        # Generate ignition vs baseline comparison figure
        print("\nGenerating ignition vs baseline comparison figure...")
        comparison_path = output_dir / 'figures' / 'ignition_vs_baseline_comparison.png'
        comparison_pdf = output_dir / 'figures' / 'ignition_vs_baseline_comparison.pdf'

        try:
            fig2 = plot_ignition_baseline_comparison(
                ignition_df=canonical_df,
                baseline_df=baseline_df,
                output_path=str(comparison_path),
                show=False,
                dataset_name='PhySF + VEP + MPeng'
            )
            fig2.savefig(str(comparison_pdf), format='pdf', dpi=300, bbox_inches='tight')
            print(f"  Saved: {comparison_path}")
            print(f"  Saved: {comparison_pdf}")
        except Exception as e:
            print(f"  WARNING: Could not generate ignition vs baseline figure: {e}")
    else:
        print("  No baseline data found")

    print("\nDone!")
    print(f"Output directory: {output_dir}")

if __name__ == '__main__':
    main()
