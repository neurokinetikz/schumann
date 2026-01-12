#!/usr/bin/env python3
"""
Null Control 5: Per-Subject Blind Cluster Analysis

Runs blind clustering analysis for each subject in PHYSF dataset separately,
then generates a collective summary of findings across all subjects.

Subject ID extraction: Characters before first underscore in filename
Example: "s2_session1.csv" -> subject "s2"
         "s12_baseline.csv" -> subject "s12"
"""

import os
import sys
import glob
from collections import defaultdict
import numpy as np
import pandas as pd
from datetime import datetime

# Import the main NC5 functions
sys.path.insert(0, os.path.dirname(__file__))
from null_control_5_blind_clustering import (
    process_session, blind_cluster_analysis, compare_to_sr_bands,
    compare_to_phi_predictions, test_sr_frequency_privilege,
    create_visualization, generate_report,
    SR_SCAN_BANDS, FUNDAMENTAL_HZ, DBSCAN_EPS, DBSCAN_MIN_SAMPLES
)

# ============================================================================
# Configuration
# ============================================================================

PHYSF_DIR = 'data/PhySF'
OUTPUT_BASE_DIR = 'nc5_per_subject'

# ============================================================================
# Subject File Grouping
# ============================================================================

def extract_subject_id(filename):
    """
    Extract subject ID from filename (characters before first underscore).

    Examples:
        s2_session1.csv -> s2
        s12_baseline.csv -> s12
        s8_eyes_open.csv -> s8
    """
    basename = os.path.basename(filename)
    subject_id = basename.split('_')[0]
    return subject_id


def group_files_by_subject(directory):
    """
    Scan directory and group CSV files by subject ID.

    Returns:
        dict: {subject_id: [list of file paths]}
    """
    if not os.path.exists(directory):
        print(f"ERROR: Directory not found: {directory}")
        return {}

    # Find all CSV files
    csv_files = glob.glob(os.path.join(directory, '**', '*.csv'), recursive=True)

    # Group by subject
    subject_files = defaultdict(list)
    for file_path in csv_files:
        subject_id = extract_subject_id(file_path)
        subject_files[subject_id].append(file_path)

    return dict(subject_files)


# ============================================================================
# Per-Subject Analysis
# ============================================================================

def process_subject(subject_id, file_paths, output_dir):
    """
    Process all files for a single subject and generate outputs.

    Args:
        subject_id: Subject identifier (e.g., "s2", "s12")
        file_paths: List of file paths for this subject
        output_dir: Output directory for this subject's results

    Returns:
        dict: Summary statistics for this subject
    """
    print("\n" + "=" * 80)
    print(f"PROCESSING SUBJECT: {subject_id}")
    print("=" * 80)
    print(f"Files: {len(file_paths)}")
    for fp in file_paths:
        print(f"  - {os.path.basename(fp)}")

    # Create subject output directory
    os.makedirs(output_dir, exist_ok=True)

    # Collect all peaks across all files for this subject
    all_peaks_list = []

    for file_path in file_paths:
        print(f"\nProcessing file: {os.path.basename(file_path)}")
        try:
            peaks_df = process_session(file_path)
            if len(peaks_df) > 0:
                all_peaks_list.append(peaks_df)
                print(f"  Extracted {len(peaks_df)} peaks")
            else:
                print(f"  No peaks found")
        except Exception as e:
            print(f"  ERROR: Failed to process file: {e}")
            continue

    # Combine all peaks for this subject
    if len(all_peaks_list) == 0:
        print(f"\n⚠️  No peaks extracted for subject {subject_id}")
        return {
            'subject_id': subject_id,
            'n_files': len(file_paths),
            'n_events': 0,
            'n_peaks': 0,
            'n_clusters': 0,
            'n_sr_aligned': 0,
            'chi2_pvalue': np.nan,
            'chi2_pass': False,
            'obs_sr_pct': np.nan,
            'status': 'NO_PEAKS'
        }

    all_peaks_df = pd.concat(all_peaks_list, ignore_index=True)

    print(f"\n{'=' * 80}")
    print(f"SUBJECT {subject_id} SUMMARY")
    print(f"{'=' * 80}")
    print(f"Total files processed: {len(file_paths)}")
    print(f"Total events analyzed: {all_peaks_df['event_idx'].nunique()}")
    print(f"Total FOOOF peaks: {len(all_peaks_df)}")

    # Run blind clustering
    peak_frequencies = all_peaks_df['peak_freq'].values

    if len(peak_frequencies) < 10:
        print(f"\n⚠️  Too few peaks ({len(peak_frequencies)}) for clustering")
        return {
            'subject_id': subject_id,
            'n_files': len(file_paths),
            'n_events': all_peaks_df['event_idx'].nunique(),
            'n_peaks': len(peak_frequencies),
            'n_clusters': 0,
            'n_sr_aligned': 0,
            'chi2_pvalue': np.nan,
            'chi2_pass': False,
            'obs_sr_pct': np.nan,
            'status': 'INSUFFICIENT_PEAKS'
        }

    print(f"\nRunning blind clustering on {len(peak_frequencies)} peaks...")
    clusters, labels = blind_cluster_analysis(peak_frequencies, DBSCAN_EPS, DBSCAN_MIN_SAMPLES)

    print(f"Discovered clusters: {len(clusters)}")

    if len(clusters) == 0:
        print(f"\n⚠️  No clusters found for subject {subject_id}")
        return {
            'subject_id': subject_id,
            'n_files': len(file_paths),
            'n_events': all_peaks_df['event_idx'].nunique(),
            'n_peaks': len(peak_frequencies),
            'n_clusters': 0,
            'n_sr_aligned': 0,
            'chi2_pvalue': np.nan,
            'chi2_pass': False,
            'obs_sr_pct': np.nan,
            'status': 'NO_CLUSTERS'
        }

    # Compare to SR bands
    sr_comparison = compare_to_sr_bands(clusters)
    n_sr_aligned = sr_comparison['within_band'].sum()

    # Compare to φ-predictions
    phi_comparison = compare_to_phi_predictions(clusters, FUNDAMENTAL_HZ)

    # Test SR frequency privilege
    chi2_result = test_sr_frequency_privilege(all_peaks_df, sr_comparison, labels)

    print(f"SR-aligned clusters: {n_sr_aligned}/{len(clusters)}")
    print(f"χ² test p-value: {chi2_result['p_value']:.4e}")
    print(f"SR enrichment: {chi2_result['obs_sr_pct']:.1f}% (expected {chi2_result['exp_sr_pct']:.1f}%)")

    # Generate outputs for this subject
    try:
        fig_path = os.path.join(output_dir, f'{subject_id}_figure.png')
        create_visualization(all_peaks_df, clusters, labels, sr_comparison,
                           phi_comparison, fig_path)
    except Exception as e:
        print(f"Warning: Failed to generate figure: {e}")

    try:
        report_path = os.path.join(output_dir, f'{subject_id}_report.md')
        test_passes = False  # Determined by pass/fail criteria
        generate_report(clusters, sr_comparison, phi_comparison, all_peaks_df,
                       labels, test_passes, chi2_result, report_path)
    except Exception as e:
        print(f"Warning: Failed to generate report: {e}")

    # Save subject-specific CSVs
    try:
        all_peaks_df.to_csv(os.path.join(output_dir, f'{subject_id}_peaks.csv'), index=False)

        clusters_df = pd.DataFrame(clusters)
        clusters_df.to_csv(os.path.join(output_dir, f'{subject_id}_clusters.csv'), index=False)

        sr_comparison.to_csv(os.path.join(output_dir, f'{subject_id}_sr_comparison.csv'), index=False)
    except Exception as e:
        print(f"Warning: Failed to save CSVs: {e}")

    # Return summary statistics
    return {
        'subject_id': subject_id,
        'n_files': len(file_paths),
        'n_events': all_peaks_df['event_idx'].nunique(),
        'n_peaks': len(peak_frequencies),
        'n_clusters': len(clusters),
        'n_sr_aligned': n_sr_aligned,
        'chi2_statistic': chi2_result['chi2'],
        'chi2_pvalue': chi2_result['p_value'],
        'chi2_pass': chi2_result['pass'],
        'obs_sr_pct': chi2_result['obs_sr_pct'],
        'exp_sr_pct': chi2_result['exp_sr_pct'],
        'effect_size': chi2_result['effect_size'],
        'status': 'SUCCESS'
    }


# ============================================================================
# Collective Summary
# ============================================================================

def generate_collective_summary(subject_results, output_path='nc5_collective_summary.md'):
    """
    Generate collective summary report across all subjects.

    Args:
        subject_results: List of subject result dicts
        output_path: Path for collective summary report
    """
    df = pd.DataFrame(subject_results)

    # Filter successful subjects
    df_success = df[df['status'] == 'SUCCESS'].copy()

    n_total = len(df)
    n_success = len(df_success)
    n_failed = n_total - n_success

    report = f"""# Null Control 5: Collective Summary Across PHYSF Subjects

**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Overview

| Metric | Value |
|--------|-------|
| **Total subjects** | {n_total} |
| **Successful analyses** | {n_success} |
| **Failed analyses** | {n_failed} |

---

## Subject-Level Results

### All Subjects

| Subject | Files | Events | Peaks | Clusters | SR-Aligned | χ² p-value | SR Enrichment | Status |
|---------|-------|--------|-------|----------|------------|------------|---------------|--------|
"""

    for _, row in df.iterrows():
        if row['status'] == 'SUCCESS':
            report += f"| {row['subject_id']} | {row['n_files']} | {row['n_events']} | {row['n_peaks']} | {row['n_clusters']} | {row['n_sr_aligned']} | {row['chi2_pvalue']:.4e} | {row['obs_sr_pct']:.1f}% | ✅ |\n"
        else:
            report += f"| {row['subject_id']} | {row['n_files']} | {row.get('n_events', 0)} | {row.get('n_peaks', 0)} | - | - | - | - | ❌ {row['status']} |\n"

    if n_success > 0:
        report += f"""

---

## Aggregate Statistics (Successful Subjects Only)

### Peak Detection

| Metric | Mean | Median | Std | Min | Max |
|--------|------|--------|-----|-----|-----|
| **Peaks per subject** | {df_success['n_peaks'].mean():.1f} | {df_success['n_peaks'].median():.1f} | {df_success['n_peaks'].std():.1f} | {df_success['n_peaks'].min():.0f} | {df_success['n_peaks'].max():.0f} |
| **Events per subject** | {df_success['n_events'].mean():.1f} | {df_success['n_events'].median():.1f} | {df_success['n_events'].std():.1f} | {df_success['n_events'].min():.0f} | {df_success['n_events'].max():.0f} |
| **Clusters per subject** | {df_success['n_clusters'].mean():.1f} | {df_success['n_clusters'].median():.1f} | {df_success['n_clusters'].std():.1f} | {df_success['n_clusters'].min():.0f} | {df_success['n_clusters'].max():.0f} |

### SR Alignment

| Metric | Mean | Median | Std | Min | Max |
|--------|------|--------|-----|-----|-----|
| **SR-aligned clusters** | {df_success['n_sr_aligned'].mean():.1f} | {df_success['n_sr_aligned'].median():.1f} | {df_success['n_sr_aligned'].std():.1f} | {df_success['n_sr_aligned'].min():.0f} | {df_success['n_sr_aligned'].max():.0f} |
| **SR alignment rate** | {(df_success['n_sr_aligned'] / df_success['n_clusters']).mean() * 100:.1f}% | {(df_success['n_sr_aligned'] / df_success['n_clusters']).median() * 100:.1f}% | - | - | - |

### SR Frequency Privilege Test

| Metric | Value |
|--------|-------|
| **Subjects passing χ² test (p < 0.001)** | {df_success['chi2_pass'].sum()}/{n_success} ({df_success['chi2_pass'].sum()/n_success*100:.1f}%) |
| **Mean observed SR enrichment** | {df_success['obs_sr_pct'].mean():.1f}% |
| **Mean expected SR enrichment (uniform)** | {df_success['exp_sr_pct'].mean():.1f}% |
| **Mean effect size** | {df_success['effect_size'].mean():.2f} |

---

## SR Band Detection Frequency

Across {n_success} subjects, SR bands detected:
"""

        # Calculate SR band detection frequency (would need to aggregate from individual reports)
        # For now, provide summary statistics

        report += f"""

---

## Collective Findings

### 1. Peak Detection Consistency

**Observation:** Blind FOOOF peak detection extracted an average of {df_success['n_peaks'].mean():.0f} peaks per subject (range: {df_success['n_peaks'].min():.0f}-{df_success['n_peaks'].max():.0f}).

### 2. Cluster Discovery Patterns

**Observation:** DBSCAN discovered an average of {df_success['n_clusters'].mean():.0f} frequency clusters per subject (range: {df_success['n_clusters'].min():.0f}-{df_success['n_clusters'].max():.0f}).

**Interpretation:** High variability in cluster count suggests individual differences in spectral architecture, though this may also reflect data quality and recording duration.

### 3. SR Frequency Alignment

**Observation:** An average of {(df_success['n_sr_aligned'] / df_success['n_clusters']).mean() * 100:.1f}% of clusters aligned with predefined SR scan bands.

**Subjects with majority SR-aligned clusters (>50%):** {(df_success['n_sr_aligned'] / df_success['n_clusters'] > 0.5).sum()}/{n_success} ({(df_success['n_sr_aligned'] / df_success['n_clusters'] > 0.5).sum()/n_success*100:.1f}%)

### 4. SR Frequency Privilege Test

**χ² Goodness-of-Fit Results:**

- **Pass rate (p < 0.001):** {df_success['chi2_pass'].sum()}/{n_success} subjects ({df_success['chi2_pass'].sum()/n_success*100:.1f}%)
- **Mean p-value:** {df_success['chi2_pvalue'].mean():.4e}
- **Median p-value:** {df_success['chi2_pvalue'].median():.4e}

**Interpretation:**
"""

        if df_success['chi2_pass'].sum() / n_success > 0.5:
            report += f"✅ Majority of subjects ({df_success['chi2_pass'].sum()}/{n_success}) show significant SR frequency privilege (p < 0.001), indicating peaks concentrate in SR bands beyond chance expectation.\n"
        else:
            report += f"❌ Minority of subjects ({df_success['chi2_pass'].sum()}/{n_success}) show significant SR frequency privilege (p < 0.001), suggesting peak distribution is largely consistent with uniform distribution.\n"

        report += f"""

### 5. SR Enrichment Analysis

**Observed SR enrichment:** {df_success['obs_sr_pct'].mean():.1f}% ± {df_success['obs_sr_pct'].std():.1f}%
**Expected (uniform distribution):** {df_success['exp_sr_pct'].mean():.1f}%
**Mean effect size:** {df_success['effect_size'].mean():.2f}

**Subjects showing SR enrichment (obs > exp):** {(df_success['obs_sr_pct'] > df_success['exp_sr_pct']).sum()}/{n_success} ({(df_success['obs_sr_pct'] > df_success['exp_sr_pct']).sum()/n_success*100:.1f}%)

---

## Conclusions

"""

        # Generate conclusions based on aggregate statistics
        pass_rate = df_success['chi2_pass'].sum() / n_success

        if pass_rate >= 0.7:
            report += "✅ **STRONG EVIDENCE FOR SR FREQUENCY PRIVILEGE**\n\n"
            report += f"Across {n_success} subjects, {df_success['chi2_pass'].sum()} ({pass_rate*100:.0f}%) showed significant concentration of spectral peaks in SR frequency bands (χ² test p < 0.001). This collective finding provides strong evidence that naturally occurring EEG spectral peaks preferentially cluster around Schumann Resonance frequencies.\n"
        elif pass_rate >= 0.5:
            report += "⚠️ **MODERATE EVIDENCE FOR SR FREQUENCY PRIVILEGE**\n\n"
            report += f"Across {n_success} subjects, {df_success['chi2_pass'].sum()} ({pass_rate*100:.0f}%) showed significant SR frequency privilege. Results are mixed, suggesting SR alignment may be present in some individuals but not universal.\n"
        else:
            report += "❌ **LIMITED EVIDENCE FOR SR FREQUENCY PRIVILEGE**\n\n"
            report += f"Across {n_success} subjects, only {df_success['chi2_pass'].sum()} ({pass_rate*100:.0f}%) showed significant SR frequency privilege (χ² test p < 0.001). The majority of subjects show peak distributions consistent with uniform distribution across the 5-40 Hz range.\n"

        report += f"""

**Key Statistics:**
- Mean SR enrichment: {df_success['obs_sr_pct'].mean():.1f}% (expected: {df_success['exp_sr_pct'].mean():.1f}%)
- Mean effect size: {df_success['effect_size'].mean():.2f}
- SR-aligned cluster rate: {(df_success['n_sr_aligned'] / df_success['n_clusters']).mean() * 100:.1f}%

---

## Individual Subject Reports

Detailed reports for each subject are available in: `{OUTPUT_BASE_DIR}/s<SUBJECT_ID>/`

Each subject folder contains:
- `s<SUBJECT_ID>_figure.png` - 4-panel visualization
- `s<SUBJECT_ID>_report.md` - Detailed analysis report
- `s<SUBJECT_ID>_peaks.csv` - All extracted FOOOF peaks
- `s<SUBJECT_ID>_clusters.csv` - Discovered clusters
- `s<SUBJECT_ID>_sr_comparison.csv` - SR band alignment results

---

*Generated by null_control_5_per_subject.py*
"""

    # Save report
    with open(output_path, 'w') as f:
        f.write(report)

    print(f"\n{'=' * 80}")
    print(f"Collective summary saved: {output_path}")
    print(f"{'=' * 80}")

    # Also save CSV summary
    csv_path = output_path.replace('.md', '.csv')
    df.to_csv(csv_path, index=False)
    print(f"Summary CSV saved: {csv_path}")


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main execution function."""
    print("\n" + "=" * 80)
    print("NULL CONTROL 5: PER-SUBJECT BLIND CLUSTER ANALYSIS")
    print("=" * 80)
    print(f"Dataset: PHYSF ({PHYSF_DIR})")
    print(f"Output directory: {OUTPUT_BASE_DIR}")

    # Group files by subject
    print("\nScanning for subject files...")
    subject_files = group_files_by_subject(PHYSF_DIR)

    if len(subject_files) == 0:
        print(f"ERROR: No subjects found in {PHYSF_DIR}")
        return

    print(f"\nFound {len(subject_files)} subjects:")
    for subject_id, files in sorted(subject_files.items()):
        print(f"  {subject_id}: {len(files)} file(s)")

    # Create base output directory
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

    # Process each subject
    subject_results = []

    for subject_id, file_paths in sorted(subject_files.items()):
        subject_output_dir = os.path.join(OUTPUT_BASE_DIR, subject_id)

        try:
            result = process_subject(subject_id, file_paths, subject_output_dir)
            subject_results.append(result)
        except Exception as e:
            print(f"\n❌ ERROR processing subject {subject_id}: {e}")
            import traceback
            traceback.print_exc()
            subject_results.append({
                'subject_id': subject_id,
                'n_files': len(file_paths),
                'status': f'ERROR: {str(e)[:50]}'
            })

    # Generate collective summary
    print("\n" + "=" * 80)
    print("GENERATING COLLECTIVE SUMMARY")
    print("=" * 80)

    summary_path = os.path.join(OUTPUT_BASE_DIR, 'collective_summary.md')
    generate_collective_summary(subject_results, summary_path)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nTotal subjects: {len(subject_results)}")
    print(f"Successful: {sum(1 for r in subject_results if r.get('status') == 'SUCCESS')}")
    print(f"Failed: {sum(1 for r in subject_results if r.get('status') != 'SUCCESS')}")
    print(f"\nOutputs saved to: {OUTPUT_BASE_DIR}/")
    print(f"Collective summary: {summary_path}")


if __name__ == '__main__':
    main()
