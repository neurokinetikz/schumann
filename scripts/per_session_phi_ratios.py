#!/usr/bin/env python3
"""
Per-Session and Individual Differences Analysis for phi^n Ratios

Computes:
1. Per-session harmonic ratio precision from SIE-PAPER-FINAL.csv
2. Three-level variance decomposition (subject/session/event)
3. Intraclass correlation coefficients (ICC)

Addresses reviewer concerns:
- Per-session analysis of ratio precision (not just aggregate)
- Individual differences in phi^n adherence

Output:
- per_session_phi_ratios.csv: Per-session statistics
- papers/images/per_session_phi_ratios.png: 4-panel visualization
- Console output with LaTeX tables for paper
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Constants
PHI = (1 + np.sqrt(5)) / 2  # 1.618034
PHI_SQUARED = PHI ** 2       # 2.618034
PHI_CUBED = PHI ** 3         # 4.236068

# File paths
INPUT_FILE = 'papers/SIE-PAPER-FINAL.csv'
OUTPUT_CSV = 'per_session_phi_ratios.csv'
OUTPUT_FIG = 'papers/images/per_session_phi_ratios.png'


def compute_icc(df, group_col, value_col):
    """
    Compute ICC(1,1) - one-way random effects, single measurement.

    ICC = (MSB - MSW) / (MSB + (k-1)*MSW)
    where MSB = mean squares between groups, MSW = mean squares within groups
    """
    groups = df.groupby(group_col)[value_col].apply(list)
    groups = [g for g in groups if len(g) > 1]  # Need at least 2 obs per group

    if len(groups) < 2:
        return np.nan, (np.nan, np.nan)

    # Compute ANOVA components
    n_groups = len(groups)
    group_means = [np.mean(g) for g in groups]
    grand_mean = np.mean([x for g in groups for x in g])

    # Mean squares between
    n_per_group = [len(g) for g in groups]
    k = np.mean(n_per_group)  # Average group size
    ssb = sum(len(g) * (m - grand_mean)**2 for g, m in zip(groups, group_means))
    msb = ssb / (n_groups - 1)

    # Mean squares within
    ssw = sum(sum((x - m)**2 for x in g) for g, m in zip(groups, group_means))
    df_within = sum(len(g) - 1 for g in groups)
    msw = ssw / df_within if df_within > 0 else np.nan

    # ICC(1,1)
    if msw == 0 or np.isnan(msw):
        return np.nan, (np.nan, np.nan)

    icc = (msb - msw) / (msb + (k - 1) * msw)

    # Bootstrap 95% CI
    n_bootstrap = 1000
    icc_boots = []
    group_list = list(groups)
    for _ in range(n_bootstrap):
        boot_idx = np.random.choice(len(group_list), size=len(group_list), replace=True)
        boot_groups = [group_list[i] for i in boot_idx]

        boot_means = [np.mean(g) for g in boot_groups]
        boot_grand = np.mean([x for g in boot_groups for x in g])

        boot_ssb = sum(len(g) * (m - boot_grand)**2 for g, m in zip(boot_groups, boot_means))
        boot_msb = boot_ssb / (len(boot_groups) - 1)

        boot_ssw = sum(sum((x - m)**2 for x in g) for g, m in zip(boot_groups, boot_means))
        boot_df_w = sum(len(g) - 1 for g in boot_groups)
        boot_msw = boot_ssw / boot_df_w if boot_df_w > 0 else np.nan

        if boot_msw > 0 and not np.isnan(boot_msw):
            boot_icc = (boot_msb - boot_msw) / (boot_msb + (k - 1) * boot_msw)
            icc_boots.append(boot_icc)

    ci = (np.percentile(icc_boots, 2.5), np.percentile(icc_boots, 97.5)) if icc_boots else (np.nan, np.nan)

    return icc, ci


def compute_variance_decomposition(df, subject_col, session_col, value_col):
    """
    Three-level variance decomposition:
    - Between-subject variance
    - Between-session (within-subject) variance
    - Within-session (event-level) variance
    """
    # Grand mean
    grand_mean = df[value_col].mean()

    # Subject means
    subject_means = df.groupby(subject_col)[value_col].mean()

    # Session means
    session_means = df.groupby(session_col)[value_col].mean()
    session_to_subject = df.groupby(session_col)[subject_col].first()

    # Between-subject variance
    subject_effects = df[subject_col].map(subject_means) - grand_mean
    var_subject = subject_effects.var()

    # Between-session (within-subject) variance
    session_subject_means = df[session_col].map(lambda s: subject_means[session_to_subject[s]])
    session_effects = df[session_col].map(session_means) - session_subject_means
    var_session = session_effects.var()

    # Within-session (event-level) variance
    event_effects = df[value_col] - df[session_col].map(session_means)
    var_event = event_effects.var()

    # Total variance
    total_var = var_subject + var_session + var_event

    return {
        'var_subject': var_subject,
        'var_session': var_session,
        'var_event': var_event,
        'total_var': total_var,
        'pct_subject': 100 * var_subject / total_var if total_var > 0 else 0,
        'pct_session': 100 * var_session / total_var if total_var > 0 else 0,
        'pct_event': 100 * var_event / total_var if total_var > 0 else 0
    }


def main():
    print("=" * 70)
    print("Per-Session and Individual Differences Analysis")
    print("=" * 70)

    # Load data
    print(f"\nLoading {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df)} events from {df['session_name'].nunique()} sessions, {df['subject'].nunique()} subjects")

    # Filter out rows with missing ratios
    ratio_cols = ['sr3/sr1', 'sr5/sr1', 'sr5/sr3']
    df_valid = df.dropna(subset=ratio_cols)
    print(f"Valid events with all ratios: {len(df_valid)}")

    # =========================================================================
    # PART 1: Per-Session Statistics
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 1: PER-SESSION RATIO PRECISION")
    print("=" * 70)

    # Group by session and compute statistics
    per_session = df_valid.groupby('session_name').agg({
        'sr3/sr1': ['mean', 'std', 'count'],
        'sr5/sr1': ['mean', 'std'],
        'sr5/sr3': ['mean', 'std'],
        'subject': 'first',
        'dataset': 'first',
        'device': 'first',
        'context': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0]
    }).reset_index()

    # Flatten column names
    per_session.columns = [
        'session_name',
        'mean_sr3_sr1', 'std_sr3_sr1', 'n_events',
        'mean_sr5_sr1', 'std_sr5_sr1',
        'mean_sr5_sr3', 'std_sr5_sr3',
        'subject', 'dataset', 'device', 'context'
    ]

    # Compute percentage errors vs theoretical phi^n values
    per_session['error_sr3_sr1_pct'] = np.abs(per_session['mean_sr3_sr1'] - PHI_SQUARED) / PHI_SQUARED * 100
    per_session['error_sr5_sr1_pct'] = np.abs(per_session['mean_sr5_sr1'] - PHI_CUBED) / PHI_CUBED * 100
    per_session['error_sr5_sr3_pct'] = np.abs(per_session['mean_sr5_sr3'] - PHI) / PHI * 100

    # Mean of the three ratio errors
    per_session['mean_ratio_error_pct'] = (
        per_session['error_sr3_sr1_pct'] +
        per_session['error_sr5_sr1_pct'] +
        per_session['error_sr5_sr3_pct']
    ) / 3

    # Filter to sessions with at least 2 events (for meaningful within-session stats)
    per_session_multi = per_session[per_session['n_events'] >= 2].copy()

    # Print summary statistics
    print(f"\nTotal sessions analyzed: {len(per_session)}")
    print(f"Sessions with 2+ events: {len(per_session_multi)}")
    print(f"Mean events per session: {per_session['n_events'].mean():.1f} (range: {per_session['n_events'].min()}-{per_session['n_events'].max()})")

    print(f"\nTheoretical phi^n values:")
    print(f"  phi^2 (SR3/SR1 expected): {PHI_SQUARED:.4f}")
    print(f"  phi^3 (SR5/SR1 expected): {PHI_CUBED:.4f}")
    print(f"  phi   (SR5/SR3 expected): {PHI:.4f}")

    print(f"\nPer-session mean ratio error (all sessions):")
    print(f"  Mean:   {per_session['mean_ratio_error_pct'].mean():.2f}%")
    print(f"  Median: {per_session['mean_ratio_error_pct'].median():.2f}%")
    print(f"  Std:    {per_session['mean_ratio_error_pct'].std():.2f}%")
    print(f"  Min:    {per_session['mean_ratio_error_pct'].min():.2f}%")
    print(f"  Max:    {per_session['mean_ratio_error_pct'].max():.2f}%")

    # Count sessions by error threshold
    n_under_1pct = (per_session['mean_ratio_error_pct'] < 1).sum()
    n_under_2pct = (per_session['mean_ratio_error_pct'] < 2).sum()
    n_under_5pct = (per_session['mean_ratio_error_pct'] < 5).sum()
    n_total = len(per_session)

    print(f"\nSessions by mean ratio error threshold:")
    print(f"  <1%: {n_under_1pct}/{n_total} ({100*n_under_1pct/n_total:.1f}%)")
    print(f"  <2%: {n_under_2pct}/{n_total} ({100*n_under_2pct/n_total:.1f}%)")
    print(f"  <5%: {n_under_5pct}/{n_total} ({100*n_under_5pct/n_total:.1f}%)")

    # Per-ratio statistics
    print(f"\nPer-ratio error statistics (across all sessions):")
    print(f"  SR3/SR1 error: {per_session['error_sr3_sr1_pct'].mean():.2f}% +/- {per_session['error_sr3_sr1_pct'].std():.2f}%")
    print(f"  SR5/SR1 error: {per_session['error_sr5_sr1_pct'].mean():.2f}% +/- {per_session['error_sr5_sr1_pct'].std():.2f}%")
    print(f"  SR5/SR3 error: {per_session['error_sr5_sr3_pct'].mean():.2f}% +/- {per_session['error_sr5_sr3_pct'].std():.2f}%")

    # Within-session variability (for sessions with 2+ events)
    print(f"\nWithin-session variability (sessions with 2+ events, n={len(per_session_multi)}):")
    print(f"  SR3/SR1 mean within-session SD: {per_session_multi['std_sr3_sr1'].mean():.4f}")
    print(f"  SR5/SR1 mean within-session SD: {per_session_multi['std_sr5_sr1'].mean():.4f}")
    print(f"  SR5/SR3 mean within-session SD: {per_session_multi['std_sr5_sr3'].mean():.4f}")

    # =========================================================================
    # PART 2: Variance Decomposition
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 2: VARIANCE DECOMPOSITION")
    print("=" * 70)

    print("\nThree-level variance decomposition for each ratio:")

    for ratio_name, theoretical in [('sr3/sr1', PHI_SQUARED), ('sr5/sr1', PHI_CUBED), ('sr5/sr3', PHI)]:
        var_decomp = compute_variance_decomposition(
            df_valid, 'subject', 'session_name', ratio_name
        )
        print(f"\n{ratio_name.upper()} (theoretical = {theoretical:.4f}):")
        print(f"  Between-subject variance:  {var_decomp['var_subject']:.6f} ({var_decomp['pct_subject']:.1f}%)")
        print(f"  Between-session variance:  {var_decomp['var_session']:.6f} ({var_decomp['pct_session']:.1f}%)")
        print(f"  Within-session variance:   {var_decomp['var_event']:.6f} ({var_decomp['pct_event']:.1f}%)")
        print(f"  Total variance:            {var_decomp['total_var']:.6f}")

    # =========================================================================
    # PART 3: Intraclass Correlation Coefficients
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 3: INTRACLASS CORRELATION COEFFICIENTS")
    print("=" * 70)

    print("\nICC(1,1) - Subject-level consistency (events nested within subjects):")
    for ratio_name in ratio_cols:
        icc, ci = compute_icc(df_valid, 'subject', ratio_name)
        print(f"  {ratio_name}: ICC = {icc:.3f} [95% CI: {ci[0]:.3f}, {ci[1]:.3f}]")

    print("\nICC(1,1) - Session-level consistency (events nested within sessions):")
    for ratio_name in ratio_cols:
        icc, ci = compute_icc(df_valid, 'session_name', ratio_name)
        print(f"  {ratio_name}: ICC = {icc:.3f} [95% CI: {ci[0]:.3f}, {ci[1]:.3f}]")

    # =========================================================================
    # PART 4: Save Outputs
    # =========================================================================
    print("\n" + "=" * 70)
    print("SAVING OUTPUT FILES")
    print("=" * 70)

    # Sort by event count descending
    per_session = per_session.sort_values('n_events', ascending=False)
    per_session.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved: {OUTPUT_CSV}")

    # =========================================================================
    # PART 5: Generate Figure
    # =========================================================================
    os.makedirs(os.path.dirname(OUTPUT_FIG), exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel A: Histogram of per-session mean errors
    ax1 = axes[0, 0]
    bins = np.arange(0, min(per_session['mean_ratio_error_pct'].max() + 1, 20), 0.5)
    ax1.hist(per_session['mean_ratio_error_pct'], bins=bins, color='steelblue',
             edgecolor='white', alpha=0.8)
    ax1.axvline(1, color='green', linestyle='--', linewidth=2, label=f'1% ({n_under_1pct} sessions)')
    ax1.axvline(2, color='orange', linestyle='--', linewidth=2, label=f'2% ({n_under_2pct} sessions)')
    ax1.axvline(5, color='red', linestyle='--', linewidth=2, label=f'5% ({n_under_5pct} sessions)')
    ax1.set_xlabel('Mean phi^n Ratio Error (%)', fontsize=11)
    ax1.set_ylabel('Number of Sessions', fontsize=11)
    ax1.set_title(f'A. Distribution of Per-Session phi^n Ratio Errors\n(N = {len(per_session)} sessions)', fontsize=12)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')

    # Panel B: Sessions per subject vs mean error
    ax2 = axes[0, 1]
    subject_stats = per_session.groupby('subject').agg({
        'mean_ratio_error_pct': 'mean',
        'session_name': 'count'
    }).reset_index()
    subject_stats.columns = ['subject', 'mean_error', 'n_sessions']
    ax2.scatter(subject_stats['n_sessions'], subject_stats['mean_error'],
                alpha=0.6, c='steelblue', edgecolors='white', s=50)
    ax2.axhline(2, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label='2% threshold')
    ax2.set_xlabel('Number of Sessions per Subject', fontsize=11)
    ax2.set_ylabel('Mean phi^n Ratio Error (%)', fontsize=11)
    ax2.set_title('B. Subject Consistency: Sessions vs Error', fontsize=12)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Panel C: Box plots by ratio type (session-level)
    ax3 = axes[1, 0]
    error_data = [
        per_session['error_sr3_sr1_pct'].values,
        per_session['error_sr5_sr1_pct'].values,
        per_session['error_sr5_sr3_pct'].values
    ]
    bp = ax3.boxplot(error_data, labels=['SR3/SR1\n(vs phi^2)', 'SR5/SR1\n(vs phi^3)', 'SR5/SR3\n(vs phi)'],
                     patch_artist=True)
    colors = ['#4ECDC4', '#FF6B6B', '#45B7D1']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax3.axhline(1, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
    ax3.axhline(2, color='orange', linestyle='--', linewidth=1.5, alpha=0.7)
    ax3.set_ylabel('Ratio Error (%)', fontsize=11)
    ax3.set_title('C. Per-Session Ratio Errors by Harmonic Pair', fontsize=12)
    ax3.grid(True, alpha=0.3, axis='y')

    # Panel D: Variance decomposition bar chart
    ax4 = axes[1, 1]

    # Compute variance decomposition for all three ratios
    var_data = {}
    for ratio_name in ratio_cols:
        var_decomp = compute_variance_decomposition(df_valid, 'subject', 'session_name', ratio_name)
        var_data[ratio_name] = var_decomp

    x = np.arange(3)
    width = 0.25

    pct_subject = [var_data[r]['pct_subject'] for r in ratio_cols]
    pct_session = [var_data[r]['pct_session'] for r in ratio_cols]
    pct_event = [var_data[r]['pct_event'] for r in ratio_cols]

    ax4.bar(x - width, pct_subject, width, label='Between-subject', color='#2E86AB')
    ax4.bar(x, pct_session, width, label='Between-session', color='#A23B72')
    ax4.bar(x + width, pct_event, width, label='Within-session', color='#F18F01')

    ax4.set_ylabel('Percentage of Total Variance', fontsize=11)
    ax4.set_title('D. Three-Level Variance Decomposition', fontsize=12)
    ax4.set_xticks(x)
    ax4.set_xticklabels(['SR3/SR1', 'SR5/SR1', 'SR5/SR3'])
    ax4.legend(loc='upper right', fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(OUTPUT_FIG, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {OUTPUT_FIG}")

    # =========================================================================
    # PART 6: LaTeX Tables for Paper
    # =========================================================================
    print("\n" + "=" * 70)
    print("LATEX TABLES FOR PAPER")
    print("=" * 70)

    # Table A: Per-Session Ratio Precision Summary
    print(r"""
\begin{table}[H]
\centering
\caption{Per-Session $\phisym^n$ Ratio Precision Summary}
\label{tab:per_session_precision}
\begin{tabular}{@{}lccc@{}}
\toprule
Metric & SR3/SR1 (vs $\phisym^2$) & SR5/SR1 (vs $\phisym^3$) & SR5/SR3 (vs $\phisym$) \\
\midrule""")

    print(f"Sessions analyzed & {len(per_session)} & {len(per_session)} & {len(per_session)} \\\\")
    print(f"Mean error (\\%) & {per_session['error_sr3_sr1_pct'].mean():.2f} & {per_session['error_sr5_sr1_pct'].mean():.2f} & {per_session['error_sr5_sr3_pct'].mean():.2f} \\\\")
    print(f"Median error (\\%) & {per_session['error_sr3_sr1_pct'].median():.2f} & {per_session['error_sr5_sr1_pct'].median():.2f} & {per_session['error_sr5_sr3_pct'].median():.2f} \\\\")

    pct_under_2_sr3 = 100 * (per_session['error_sr3_sr1_pct'] < 2).sum() / len(per_session)
    pct_under_2_sr5_1 = 100 * (per_session['error_sr5_sr1_pct'] < 2).sum() / len(per_session)
    pct_under_2_sr5_3 = 100 * (per_session['error_sr5_sr3_pct'] < 2).sum() / len(per_session)
    print(f"Sessions $<$2\\% error & {pct_under_2_sr3:.1f}\\% & {pct_under_2_sr5_1:.1f}\\% & {pct_under_2_sr5_3:.1f}\\% \\\\")

    print(f"Within-session SD & {per_session_multi['std_sr3_sr1'].mean():.3f} & {per_session_multi['std_sr5_sr1'].mean():.3f} & {per_session_multi['std_sr5_sr3'].mean():.3f} \\\\")

    print(r"""\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item Within-session SD computed for sessions with $\geq$2 events (N=""" + f"{len(per_session_multi)}" + r""").
\end{tablenotes}
\end{table}""")

    # Table B: Variance Decomposition
    print(r"""
\begin{table}[H]
\centering
\caption{Variance Decomposition of $\phisym^n$ Ratio Precision}
\label{tab:variance_decomposition}
\begin{tabular}{@{}llccc@{}}
\toprule
Level & Description & SR3/SR1 & SR5/SR1 & SR5/SR3 \\
\midrule""")

    print(f"Between-subject & Subject differences in mean adherence & {var_data['sr3/sr1']['pct_subject']:.1f}\\% & {var_data['sr5/sr1']['pct_subject']:.1f}\\% & {var_data['sr5/sr3']['pct_subject']:.1f}\\% \\\\")
    print(f"Between-session & Session variation within subjects & {var_data['sr3/sr1']['pct_session']:.1f}\\% & {var_data['sr5/sr1']['pct_session']:.1f}\\% & {var_data['sr5/sr3']['pct_session']:.1f}\\% \\\\")
    print(f"Within-session & Event variation within sessions & {var_data['sr3/sr1']['pct_event']:.1f}\\% & {var_data['sr5/sr1']['pct_event']:.1f}\\% & {var_data['sr5/sr3']['pct_event']:.1f}\\% \\\\")

    print(r"""\bottomrule
\end{tabular}
\end{table}""")

    # Table C: ICC
    print(r"""
\begin{table}[H]
\centering
\caption{Intraclass Correlation Coefficients for $\phisym^n$ Ratio Precision}
\label{tab:icc}
\begin{tabular}{@{}llll@{}}
\toprule
Ratio & ICC (Subject) & 95\% CI & Interpretation \\
\midrule""")

    for ratio_name, display_name in [('sr3/sr1', 'SR3/SR1'), ('sr5/sr1', 'SR5/SR1'), ('sr5/sr3', 'SR5/SR3')]:
        icc, ci = compute_icc(df_valid, 'subject', ratio_name)
        if icc >= 0.75:
            interp = "Excellent"
        elif icc >= 0.5:
            interp = "Good"
        elif icc >= 0.25:
            interp = "Moderate"
        else:
            interp = "Poor"
        print(f"{display_name} & {icc:.3f} & [{ci[0]:.3f}, {ci[1]:.3f}] & {interp} \\\\")

    print(r"""\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item ICC(1,1) computed with bootstrap 95\% confidence intervals (N=1000).
Interpretation: $<$0.25 poor, 0.25--0.50 moderate, 0.50--0.75 good, $>$0.75 excellent.
\end{tablenotes}
\end{table}""")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
