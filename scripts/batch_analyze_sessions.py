#!/usr/bin/env python3
"""
Batch Analysis Script with AI-Powered Session Summaries

This script:
1. Runs analyze_ignition_custom.py on all configured EEG data files
2. Uses Claude API to analyze all output charts (7 PNGs)
3. Parses the markdown console output for numerical metrics
4. Generates comprehensive session summary reports
5. Saves reports to each session's output directory

Usage:
    python batch_analyze_sessions.py --output exports --analyze
    python batch_analyze_sessions.py --analyze-only  # Skip analysis, just generate reports
    python batch_analyze_sessions.py --limit 5       # Process only first 5 files
"""

import os
import sys
import glob
import base64
import argparse
import traceback
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Optional

# Add lib to path for imports
sys.path.insert(0, './lib')

# Import the main analysis function
from analyze_ignition_custom import main as analyze_main

# Import cross-session non-SR peak collector
try:
    from lib.non_sr_clustering import CrossSessionNonSRCollector
    NON_SR_CLUSTERING_AVAILABLE = True
except ImportError:
    NON_SR_CLUSTERING_AVAILABLE = False
    CrossSessionNonSRCollector = None


def list_csv_files(directory: str) -> List[str]:
    """List all CSV files in a directory."""
    pattern = os.path.join(directory, "*.csv")
    return sorted(glob.glob(pattern))


def get_datasets() -> List[Tuple[str, str, List[str], int, float]]:
    """
    Get all configured datasets - replicates run_batch() configuration exactly.

    Returns list of (filepath, device, electrodes, header, fs) tuples.
    """
    # Electrode configurations
    EPOC_ELECTRODES = ['EEG.AF3', 'EEG.F7', 'EEG.F3', 'EEG.FC5', 'EEG.P7',
                       'EEG.O1', 'EEG.O2', 'EEG.P8', 'EEG.FC6', 'EEG.F4',
                       'EEG.F8', 'EEG.AF4']
    INSIGHT_ELECTRODES = ['EEG.AF3', 'EEG.T7', 'EEG.Pz', 'EEG.T8', 'EEG.AF4']
    MUSE_ELECTRODES = ['EEG.AF7', 'EEG.AF8', 'EEG.TP9', 'EEG.TP10']

    datasets = []

    # FILES (5 EPOC files, header=1, 128 Hz)
    for filepath in [
        'data/test schumann_EPOCX_111270_2023.04.23T14.50.35.05.00.md.pm.bp.csv',
        'data/20201229_29.12.20_11.27.57.md.pm.bp.csv',
        'data/hyp_02.01.21_13.51.16.md.pm.bp.csv',
        'data/med_EPOCX_111270_2021.06.12T09.50.52.04.00.md.bp.csv',
        'data/binaural_EPOCX_111270_2021.06.17T10.04.52.04.00.md.bp.csv',
    ]:
        datasets.append((filepath, 'emotiv', EPOC_ELECTRODES, 1, 128.0))

    # PHYSF (data/PhySF/*.csv, EPOC format, header=0, 128 Hz)
    for filepath in list_csv_files("data/PhySF"):
        datasets.append((filepath, 'emotiv', EPOC_ELECTRODES, 0, 128.0))

    # EMOTIONS (data/emotions/*.csv, EPOC format, header=0, 256 Hz)
    for filepath in list_csv_files("data/emotions"):
        datasets.append((filepath, 'emotiv', EPOC_ELECTRODES, 0, 256.0))

    # INSIGHT (data/insight/*.csv, different electrodes, header=1, 128 Hz)
    for filepath in list_csv_files("data/insight"):
        datasets.append((filepath, 'emotiv', INSIGHT_ELECTRODES, 1, 128.0))

    # MUSE (data/muse/*.csv, muse format, header=0, 128 Hz)
    for filepath in list_csv_files("data/muse"):
        datasets.append((filepath, 'muse', MUSE_ELECTRODES, 0, 128.0))

    return datasets


def run_analysis_on_file(filepath: str, device: str, electrodes: List[str],
                         header: int, output_dir: str, fs: float = 128.0) -> Tuple[bool, str]:
    """
    Run analyze_ignition_custom.main() on a single file.

    Returns (success, session_name or error_message)
    """
    session_name = Path(filepath).stem

    try:
        analyze_main(
            data_file=filepath,
            device=device,
            electrodes=electrodes,
            header=header,
            session_name=session_name,
            output_dir=output_dir,
            show_plots=False,  # Don't show plots in batch mode
            fs=fs
        )
        return True, session_name
    except Exception as e:
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        return False, error_msg


def get_session_output_files(session_dir: str, session_name: str) -> Dict[str, Optional[str]]:
    """
    Get paths to all expected output files for a session.
    Supports both NEW format (separate charts) and OLD format (combined charts).

    Returns dict with keys: raw_eeg, f0_filtered, z_envelope, msc, plv,
                           bicoherence, effect_sizes, markdown,
                           detection, extended (old format)
    """
    files = {
        'raw_eeg': None,
        'f0_filtered': None,
        'z_envelope': None,
        'msc': None,
        'plv': None,
        'bicoherence': None,
        'effect_sizes': None,
        'markdown': None,
        # Old format files
        'detection': None,
        'extended': None,
    }

    # NEW format: separate charts (from recent analyze_ignition_custom.py)
    new_png_patterns = [
        ('raw_eeg', f'{session_name}_1_raw_eeg.png'),
        ('f0_filtered', f'{session_name}_2_f0_filtered.png'),
        ('z_envelope', f'{session_name}_3_z_envelope.png'),
        ('msc', f'{session_name}_4_msc.png'),
        ('plv', f'{session_name}_5_plv.png'),
        ('bicoherence', f'{session_name}_6_bicoherence.png'),
        ('effect_sizes', f'{session_name}_7_effect_sizes.png'),
    ]

    # OLD format: combined charts (from earlier versions)
    old_png_patterns = [
        ('detection', f'{session_name}_detection.png'),
        ('bicoherence', f'{session_name}_bicoherence.png'),
        ('extended', f'{session_name}_extended.png'),
    ]

    # Check new format first
    for key, filename in new_png_patterns:
        path = os.path.join(session_dir, filename)
        if os.path.exists(path):
            files[key] = path

    # Check old format as fallback
    for key, filename in old_png_patterns:
        path = os.path.join(session_dir, filename)
        if os.path.exists(path) and files[key] is None:
            files[key] = path

    # Markdown file - new format first, then old format
    md_path_new = os.path.join(session_dir, f'{session_name}.md')
    md_path_old = os.path.join(session_dir, 'console_output.md')

    if os.path.exists(md_path_new):
        files['markdown'] = md_path_new
    elif os.path.exists(md_path_old):
        files['markdown'] = md_path_old

    return files


def encode_image_base64(image_path: str) -> str:
    """Read an image file and return base64 encoded string."""
    with open(image_path, 'rb') as f:
        return base64.standard_b64encode(f.read()).decode('utf-8')


def analyze_session_with_claude(session_dir: str, session_name: str,
                                client) -> Optional[str]:
    """
    Use Claude API to analyze all session outputs and generate a summary.

    Args:
        session_dir: Path to session output directory
        session_name: Name of the session
        client: Anthropic client instance

    Returns:
        Generated summary report as markdown string, or None on failure
    """
    files = get_session_output_files(session_dir, session_name)

    # Check if we have enough files to analyze
    available_files = [k for k, v in files.items() if v is not None]
    if len(available_files) < 3:
        print(f"  Insufficient output files for analysis ({len(available_files)} found)")
        return None

    # Read markdown content if available
    markdown_content = ""
    if files['markdown']:
        try:
            with open(files['markdown'], 'r', encoding='utf-8') as f:
                markdown_content = f.read()
        except Exception as e:
            print(f"  Warning: Could not read markdown file: {e}")

    # Build message content with images and prompt
    content = []

    # Add each available image - supports both NEW and OLD format
    chart_names = {
        # New format (separate charts)
        'raw_eeg': '1. Raw EEG Signal',
        'f0_filtered': '2. F0 Filtered Signal (Schumann fundamental ~7.6 Hz)',
        'z_envelope': '3. Z-Scored Detection Envelope',
        'msc': '4. Mean Squared Coherence (MSC)',
        'plv': '5. Phase Locking Value (PLV)',
        'bicoherence': '6. Bicoherence Triads',
        'effect_sizes': '7. Effect Sizes (Cohen\'s d)',
        # Old format (combined charts)
        'detection': 'Detection Overview (combined: raw EEG, filtered, z-envelope, MSC, PLV)',
        'extended': 'Extended Analysis (band powers, harmonic activity, spectral metrics)',
    }

    # Check which format we have
    new_format_keys = ['raw_eeg', 'f0_filtered', 'z_envelope', 'msc', 'plv', 'bicoherence', 'effect_sizes']
    old_format_keys = ['detection', 'bicoherence', 'extended']

    # Determine which keys to use based on what files exist
    has_new_format = any(files[k] for k in new_format_keys[:5])  # Check first 5 new format keys
    has_old_format = files['detection'] is not None

    if has_new_format:
        keys_to_use = new_format_keys
    else:
        keys_to_use = old_format_keys

    image_descriptions = []
    for key in keys_to_use:
        if files[key]:
            try:
                img_b64 = encode_image_base64(files[key])
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": img_b64
                    }
                })
                image_descriptions.append(chart_names[key])
            except Exception as e:
                print(f"  Warning: Could not encode {key} image: {e}")

    # Create the analysis prompt
    prompt = f"""You are an expert neuroscience researcher analyzing EEG data for Schumann Resonance coherence patterns.

I'm providing you with the analysis outputs for session: {session_name}

The images provided are (in order):
{chr(10).join(f"- {desc}" for desc in image_descriptions)}

The markdown console output contains detailed numerical metrics from the analysis:

<markdown_output>
{markdown_content[:15000] if markdown_content else "No markdown output available"}
</markdown_output>

Please provide a comprehensive session analysis report with the following structure:

## Overview
Provide a brief summary of the session including:
- Total duration and number of ignition events
- Overall signal quality assessment
- Key harmonic frequencies detected

## Chart-by-Chart Analysis

For each chart provided, give a detailed interpretation:
- What patterns are visible
- Key timepoints or features of interest
- How it relates to the Schumann Resonance hypothesis

## Key Metrics Summary
Create a table of the most important metrics extracted from the markdown output:
- Event statistics (count, duration, peak z-scores)
- Harmonic frequencies (f0, f1, f2)
- Coherence measures (MSC, PLV)
- Effect sizes (Cohen's d for ignition vs baseline)

## Session Assessment
Provide an overall assessment:
- Strength of evidence for brain-Schumann coupling
- Quality of ignition events detected
- Notable patterns or anomalies
- Comparison to typical session characteristics

## Recommendations
Suggest any:
- Follow-up analyses that might be informative
- Features worth investigating in more detail
- Caveats or limitations of this session's data

Be specific, quantitative, and scientific in your analysis."""

    content.append({
        "type": "text",
        "text": prompt
    })

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=6000,
            messages=[{
                "role": "user",
                "content": content
            }]
        )
        return response.content[0].text
    except Exception as e:
        print(f"  Claude API error: {e}")
        return None


def generate_summary_report(session_name: str, analysis_text: str) -> str:
    """
    Wrap the Claude analysis in a formatted report with metadata.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report = f"""# Session Summary Report: {session_name}

*Generated: {timestamp}*

---

{analysis_text}

---

*This report was automatically generated using Claude AI analysis of the session outputs.*
*For questions or issues, review the raw output files in the session directory.*
"""
    return report


def main():
    parser = argparse.ArgumentParser(
        description='Batch EEG Analysis with AI-Powered Session Summaries'
    )
    parser.add_argument('--output', type=str, default='exports',
                        help='Output directory (default: exports)')
    parser.add_argument('--analyze', action='store_true',
                        help='Enable AI analysis and summary generation')
    parser.add_argument('--analyze-only', action='store_true',
                        help='Skip EEG analysis, only generate AI summaries for existing outputs')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip sessions that already have summary reports')
    parser.add_argument('--limit', type=int, default=None,
                        help='Process only first N files (for testing)')

    args = parser.parse_args()

    # Get datasets
    datasets = get_datasets()
    print(f"\nFound {len(datasets)} datasets to process")

    if args.limit:
        datasets = datasets[:args.limit]
        print(f"Limited to first {args.limit} datasets")

    # Initialize Claude client if analysis is enabled
    client = None
    if args.analyze or args.analyze_only:
        # Check for API key first
        api_key = os.environ.get('ANTHROPIC_API_KEY')
        if not api_key:
            print("ERROR: ANTHROPIC_API_KEY environment variable not set")
            print("\nTo set it, run:")
            print("  export ANTHROPIC_API_KEY='your-api-key-here'")
            print("\nOr add it to your shell profile (~/.zshrc or ~/.bashrc)")
            sys.exit(1)

        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            print("Claude API client initialized")
        except ImportError:
            print("ERROR: anthropic package not installed. Run: pip install anthropic")
            sys.exit(1)
        except Exception as e:
            print(f"ERROR: Could not initialize Claude client: {e}")
            sys.exit(1)

    # Track results
    results = []
    errors = []
    summaries_generated = 0
    session_analysis_results = {}  # Store analysis results for cross-session processing

    # Initialize cross-session non-SR peak collector
    cross_session_collector = None
    if NON_SR_CLUSTERING_AVAILABLE:
        cross_session_collector = CrossSessionNonSRCollector(freq_range=(1.0, 50.0))

    # Process each dataset
    for i, (filepath, device, electrodes, header, fs) in enumerate(datasets, 1):
        session_name = Path(filepath).stem
        session_dir = os.path.join(args.output, session_name)

        print(f"\n[{i}/{len(datasets)}] Processing: {session_name}")

        # Check if session already has summary report
        summary_path = os.path.join(session_dir, f'{session_name}_summary_report.md')
        if args.skip_existing and os.path.exists(summary_path):
            print(f"  Skipping (summary exists): {summary_path}")
            results.append(session_name)
            continue

        # Run EEG analysis (unless analyze-only mode)
        if not args.analyze_only:
            if not os.path.exists(filepath):
                print(f"  SKIP (file not found): {filepath}")
                continue

            print(f"  Running EEG analysis...")
            success, result = run_analysis_on_file(
                filepath, device, electrodes, header, args.output, fs
            )

            if not success:
                print(f"  ERROR: {result[:200]}")
                errors.append((session_name, result))
                continue

            results.append(session_name)
            print(f"  Analysis complete")

            # Load non-SR peaks for cross-session aggregation
            if cross_session_collector:
                non_sr_csv = os.path.join(session_dir, f'{session_name}_non_sr_peaks.csv')
                if os.path.exists(non_sr_csv):
                    try:
                        import pandas as pd
                        peaks_df = pd.read_csv(non_sr_csv)
                        n_loaded = 0
                        for _, row in peaks_df.iterrows():
                            from lib.non_sr_clustering import NonSRPeak
                            peak = NonSRPeak(
                                freq_hz=row['freq_hz'],
                                power_log10=row['power_log10'],
                                bandwidth_hz=row['bandwidth_hz'],
                                session_id=row['session_id'],
                                window_type=row.get('window_type', 'session'),
                                window_index=int(row.get('window_index', 0)),
                                window_start_sec=row.get('window_start_sec', float('nan')),
                                window_end_sec=row.get('window_end_sec', float('nan')),
                                cluster_label=int(row.get('cluster_label', -1))
                            )
                            cross_session_collector._collector._peaks.append(peak)
                            cross_session_collector._collector._session_ids.add(peak.session_id)
                            n_loaded += 1
                        cross_session_collector._session_results[session_name] = {'n_peaks': n_loaded}
                        print(f"  Loaded {n_loaded} non-SR peaks for cross-session analysis")
                    except Exception as e:
                        print(f"  Warning: Could not load non-SR peaks: {e}")
        else:
            # In analyze-only mode, check if session directory exists
            if not os.path.exists(session_dir):
                print(f"  SKIP (no output directory): {session_dir}")
                continue
            results.append(session_name)

            # Also try to load existing non-SR peaks in analyze-only mode
            if cross_session_collector:
                non_sr_csv = os.path.join(session_dir, f'{session_name}_non_sr_peaks.csv')
                if os.path.exists(non_sr_csv):
                    try:
                        import pandas as pd
                        peaks_df = pd.read_csv(non_sr_csv)
                        n_loaded = 0
                        for _, row in peaks_df.iterrows():
                            from lib.non_sr_clustering import NonSRPeak
                            peak = NonSRPeak(
                                freq_hz=row['freq_hz'],
                                power_log10=row['power_log10'],
                                bandwidth_hz=row['bandwidth_hz'],
                                session_id=row['session_id'],
                                window_type=row.get('window_type', 'session'),
                                window_index=int(row.get('window_index', 0)),
                                window_start_sec=row.get('window_start_sec', float('nan')),
                                window_end_sec=row.get('window_end_sec', float('nan')),
                                cluster_label=int(row.get('cluster_label', -1))
                            )
                            cross_session_collector._collector._peaks.append(peak)
                            cross_session_collector._collector._session_ids.add(peak.session_id)
                            n_loaded += 1
                        cross_session_collector._session_results[session_name] = {'n_peaks': n_loaded}
                        if n_loaded > 0:
                            print(f"  Loaded {n_loaded} non-SR peaks for cross-session analysis")
                    except Exception as e:
                        print(f"  Warning: Could not load non-SR peaks: {e}")

        # Generate AI summary if enabled
        if client and (args.analyze or args.analyze_only):
            print(f"  Generating AI summary...")

            analysis_text = analyze_session_with_claude(session_dir, session_name, client)

            if analysis_text:
                report = generate_summary_report(session_name, analysis_text)

                # Save report
                with open(summary_path, 'w', encoding='utf-8') as f:
                    f.write(report)

                print(f"  Summary saved: {summary_path}")
                summaries_generated += 1
            else:
                print(f"  Could not generate summary")

    # ==========================================================================
    # CROSS-SESSION NON-SR PEAK CLUSTERING
    # ==========================================================================
    if cross_session_collector and cross_session_collector.n_peaks > 0:
        print("\n" + "="*60)
        print("CROSS-SESSION NON-SR PEAK ANALYSIS")
        print("="*60)
        print(f"Total non-SR peaks across all sessions: {cross_session_collector.n_peaks}")
        print(f"Sessions with peaks: {cross_session_collector.n_sessions}")

        # Run grand clustering
        if cross_session_collector.n_peaks >= 10:
            print("\nRunning cross-session clustering...")
            try:
                grand_results = cross_session_collector.run_grand_clustering(
                    method='kmeans',
                    n_clusters='auto'
                )
                print(f"  Clusters found: {grand_results['n_clusters']}")

                if grand_results['cluster_centers_hz']:
                    print("  Cluster centers (Hz):")
                    for stats in grand_results['cluster_stats']:
                        print(f"    Cluster {stats['cluster_id']}: {stats['center_freq_hz']:.2f} Hz "
                              f"(n={stats['n_peaks']}, {stats['n_sessions']} sessions, "
                              f"{stats['session_prevalence']:.0%} prevalence)")

                # Generate visualization
                grand_fig_path = os.path.join(args.output, 'non_sr_grand_clusters.png')
                cross_session_collector.plot_grand_clusters(
                    grand_results,
                    title=f'Cross-Session Non-SR Peak Clusters ({cross_session_collector.n_peaks} peaks)',
                    out_path=grand_fig_path,
                    show=False
                )
                print(f"\n  Grand cluster visualization saved: {grand_fig_path}")

                # Export summary files
                output_files = cross_session_collector.export_summary(
                    grand_results, out_dir=args.output
                )
                for file_type, path in output_files.items():
                    print(f"  Exported: {path}")

            except Exception as e:
                print(f"  Error in cross-session clustering: {e}")
        else:
            print(f"  Skipping clustering (need >= 10 peaks, got {cross_session_collector.n_peaks})")

    # Print final summary
    print("\n" + "="*60)
    print("BATCH PROCESSING COMPLETE")
    print("="*60)
    print(f"Total datasets:       {len(datasets)}")
    print(f"Successfully analyzed: {len(results)}")
    print(f"Errors:               {len(errors)}")
    if client:
        print(f"Summaries generated:  {summaries_generated}")
    if cross_session_collector:
        print(f"Cross-session peaks:  {cross_session_collector.n_peaks}")

    if errors:
        print("\nErrors occurred in:")
        for session_name, error in errors:
            print(f"  - {session_name}: {error[:100]}...")

        # Save error log
        error_log_path = os.path.join(args.output, 'batch_errors.log')
        with open(error_log_path, 'w') as f:
            for session_name, error in errors:
                f.write(f"\n{'='*60}\n{session_name}\n{'='*60}\n{error}\n")
        print(f"\nFull error log saved to: {error_log_path}")


if __name__ == '__main__':
    main()
