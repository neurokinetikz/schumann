#!/usr/bin/env python3
"""
Test comprehensive alignment metric that includes all position types.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PHI = (1 + np.sqrt(5)) / 2

def compute_lattice_coordinate(freq, f0):
    """Compute u = [log_φ(f/f0)] mod 1"""
    n = np.log(freq / f0) / np.log(PHI)
    return n - np.floor(n)

def compute_all_enrichments(freqs, f0):
    """Compute enrichment for all 4 position types."""
    u = compute_lattice_coordinate(freqs, f0)
    u_window = 0.05
    
    results = {}
    positions = [('boundary', 0.0), ('noble_2', 0.382), 
                 ('attractor', 0.5), ('noble_1', 0.618)]
    
    for pos_name, pos_center in positions:
        in_window = np.abs(u - pos_center) < u_window
        if pos_name == 'boundary':
            in_window |= np.abs(u - 1.0) < u_window
            expected_frac = 4 * u_window  # 0.2 (both ends)
        else:
            expected_frac = 2 * u_window  # 0.1
        
        observed_frac = in_window.sum() / len(u)
        enrichment = (observed_frac / expected_frac - 1) * 100
        results[pos_name] = enrichment
    
    return results

def simple_metric(enrichments):
    """Original metric: attractor - boundary"""
    return enrichments['attractor'] - enrichments['boundary']

def comprehensive_metric_v1(enrichments):
    """Weighted sum of all positions"""
    return (-1.0 * enrichments['boundary'] +      # Depletion is good
             1.0 * enrichments['attractor'] +
             1.5 * enrichments['noble_1'] +        # Weight strongest signal
             0.5 * enrichments['noble_2'])

def comprehensive_metric_v2(enrichments):
    """Sum of deviations from uniform, weighted by sign correctness"""
    # Theory: boundary should be negative, others positive
    score = 0
    if enrichments['boundary'] < 0:
        score += abs(enrichments['boundary'])  # Reward depletion
    else:
        score -= abs(enrichments['boundary'])  # Penalize enrichment
    
    for pos in ['attractor', 'noble_1', 'noble_2']:
        if enrichments[pos] > 0:
            score += enrichments[pos]  # Reward enrichment
        else:
            score -= abs(enrichments[pos])  # Penalize depletion
    
    return score

def comprehensive_metric_v3(enrichments):
    """Pattern fidelity: how well ranking matches theory"""
    # Theory predicts: boundary < noble_2 < attractor < noble_1
    expected_order = ['boundary', 'noble_2', 'attractor', 'noble_1']
    actual_values = [enrichments[p] for p in expected_order]
    
    # Count correct pairwise orderings
    n_pairs = 0
    n_correct = 0
    for i in range(len(expected_order)):
        for j in range(i+1, len(expected_order)):
            n_pairs += 1
            if actual_values[i] < actual_values[j]:
                n_correct += 1
    
    # Return ranking score + magnitude of pattern
    rank_score = n_correct / n_pairs * 100
    magnitude = (enrichments['noble_1'] - enrichments['boundary'])
    return rank_score + magnitude

def f0_sweep(freqs, f0_range=(6.5, 8.5), step=0.05):
    """Sweep f0 and compute all metrics."""
    f0_values = np.arange(f0_range[0], f0_range[1] + step/2, step)
    
    results = {
        'f0': f0_values,
        'simple': [],
        'comprehensive_v1': [],
        'comprehensive_v2': [],
        'comprehensive_v3': [],
    }
    
    for f0 in f0_values:
        enrichments = compute_all_enrichments(freqs, f0)
        results['simple'].append(simple_metric(enrichments))
        results['comprehensive_v1'].append(comprehensive_metric_v1(enrichments))
        results['comprehensive_v2'].append(comprehensive_metric_v2(enrichments))
        results['comprehensive_v3'].append(comprehensive_metric_v3(enrichments))
    
    return results

def main():
    # Load peak data
    print("Loading peak data...")
    peaks_df = pd.read_csv('golden_ratio_peaks_ALL_EMOTIV.csv')
    freqs = peaks_df['freq'].values
    freqs = freqs[(freqs >= 4) & (freqs <= 50)]
    print(f"  {len(freqs):,} peaks in 4-50 Hz range")
    
    # Run sweep
    print("\nRunning f₀ sensitivity sweep with multiple metrics...")
    results = f0_sweep(freqs)
    
    # Find optima for each metric
    print("\n" + "=" * 70)
    print("OPTIMAL f₀ BY METRIC")
    print("=" * 70)
    
    metrics = ['simple', 'comprehensive_v1', 'comprehensive_v2', 'comprehensive_v3']
    metric_names = {
        'simple': 'Simple (attractor - boundary)',
        'comprehensive_v1': 'Weighted sum (all positions)',
        'comprehensive_v2': 'Sign-correct deviations',
        'comprehensive_v3': 'Ranking + magnitude',
    }
    
    optimal_f0s = {}
    for metric in metrics:
        values = np.array(results[metric])
        opt_idx = np.argmax(values)
        opt_f0 = results['f0'][opt_idx]
        opt_val = values[opt_idx]
        optimal_f0s[metric] = opt_f0
        
        # Find plateau (>70% of optimal)
        threshold = 0.7 * opt_val if opt_val > 0 else opt_val * 1.3
        plateau_mask = values >= threshold
        plateau_idx = np.where(plateau_mask)[0]
        if len(plateau_idx) > 0:
            plateau = (results['f0'][plateau_idx[0]], results['f0'][plateau_idx[-1]])
        else:
            plateau = (opt_f0, opt_f0)
        
        print(f"\n{metric_names[metric]}:")
        print(f"  Optimal f₀: {opt_f0:.2f} Hz")
        print(f"  Optimal value: {opt_val:.1f}")
        print(f"  Plateau (>70%): {plateau[0]:.2f} - {plateau[1]:.2f} Hz")
        
        # Value at key f0 points
        for test_f0 in [7.6, 7.83, 8.05]:
            idx = np.argmin(np.abs(results['f0'] - test_f0))
            print(f"  Value at {test_f0} Hz: {values[idx]:.1f}")
    
    # Detailed enrichments at key f0 values
    print("\n" + "=" * 70)
    print("ENRICHMENT BREAKDOWN AT KEY f₀ VALUES")
    print("=" * 70)
    
    for test_f0 in [7.6, 7.83, 8.05]:
        enrichments = compute_all_enrichments(freqs, test_f0)
        print(f"\nf₀ = {test_f0} Hz:")
        print(f"  Boundary:  {enrichments['boundary']:+.1f}%")
        print(f"  2° Noble:  {enrichments['noble_2']:+.1f}%")
        print(f"  Attractor: {enrichments['attractor']:+.1f}%")
        print(f"  1° Noble:  {enrichments['noble_1']:+.1f}%")
    
    # Create visualization
    print("\nGenerating comparison plot...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    colors = {'simple': 'blue', 'comprehensive_v1': 'green', 
              'comprehensive_v2': 'orange', 'comprehensive_v3': 'purple'}
    
    for ax, metric in zip(axes.flat, metrics):
        values = np.array(results[metric])
        ax.plot(results['f0'], values, color=colors[metric], linewidth=2)
        
        opt_f0 = optimal_f0s[metric]
        opt_idx = np.argmax(values)
        ax.axvline(opt_f0, color='red', linestyle='--', alpha=0.7, 
                   label=f'Optimal: {opt_f0:.2f} Hz')
        ax.axvline(7.6, color='gray', linestyle=':', alpha=0.7, label='7.6 Hz')
        ax.axvline(7.83, color='black', linestyle=':', alpha=0.7, label='7.83 Hz')
        
        ax.set_xlabel('f₀ (Hz)')
        ax.set_ylabel('Metric Value')
        ax.set_title(metric_names[metric])
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('f0_metric_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: f0_metric_comparison.png")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Simple metric optimal: {optimal_f0s['simple']:.2f} Hz")
    print(f"  Comprehensive v1 optimal: {optimal_f0s['comprehensive_v1']:.2f} Hz")
    print(f"  Comprehensive v2 optimal: {optimal_f0s['comprehensive_v2']:.2f} Hz")
    print(f"  Comprehensive v3 optimal: {optimal_f0s['comprehensive_v3']:.2f} Hz")
    print(f"\n  Mean optimal across metrics: {np.mean(list(optimal_f0s.values())):.2f} Hz")
    print(f"  Std of optima: {np.std(list(optimal_f0s.values())):.2f} Hz")

if __name__ == '__main__':
    main()
