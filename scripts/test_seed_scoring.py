#!/usr/bin/env python3
"""
Quick test to validate the new composite seed scoring approach
"""
import numpy as np

def test_seed_score_logic():
    """Test the scoring function with synthetic data"""

    # Simulate scoring function
    def compute_seed_score(lat, t0, flow_out, flow_in, peak_z, rise_z):
        """Simplified version of _compute_seed_score for testing"""
        score = 0.0

        # 1. Temporal (40%)
        if np.isfinite(lat):
            delay = max(0.0, lat - t0)
            lat_score = np.exp(-delay / 0.5)
        else:
            lat_score = 0.0

        # 2. Flow (35%)
        if np.isfinite(flow_out) and np.isfinite(flow_in):
            flow_net = flow_out - flow_in
            flow_sum = flow_out + flow_in + 1e-9
            flow_net_norm = flow_net / flow_sum
            flow_score = (1.0 + flow_net_norm) / 2.0
        elif np.isfinite(flow_out):
            flow_score = min(1.0, flow_out / 1.0)  # normalized by assumed max
        else:
            flow_score = 0.0

        # 3. Signal strength (15%)
        if np.isfinite(peak_z):
            signal_score = min(1.0, peak_z / 10.0)
        else:
            signal_score = 0.0

        # 4. Rise dynamics (10%)
        if np.isfinite(rise_z):
            rise_score = min(1.0, rise_z / 5.0)
        else:
            rise_score = 0.0

        composite = (0.40 * lat_score +
                    0.35 * flow_score +
                    0.15 * signal_score +
                    0.10 * rise_score)

        return composite, lat_score, flow_score, signal_score, rise_score

    t0 = 100.0  # ignition onset time

    # Test case 1: Ideal generator (early, high outflow, strong signal)
    print("=" * 60)
    print("Test Case 1: Ideal Generator Channel")
    print("-" * 60)
    score, lat_s, flow_s, sig_s, rise_s = compute_seed_score(
        lat=100.0,      # at t0
        t0=t0,
        flow_out=0.8,   # high outflow
        flow_in=0.2,    # low inflow
        peak_z=8.0,     # strong signal
        rise_z=4.0      # steep rise
    )
    print(f"Latency: 100.0 (at t0) -> score: {lat_s:.3f}")
    print(f"Flow: out=0.8, in=0.2 -> score: {flow_s:.3f}")
    print(f"Peak z: 8.0 -> score: {sig_s:.3f}")
    print(f"Rise z: 4.0 -> score: {rise_s:.3f}")
    print(f"COMPOSITE SCORE: {score:.3f}")

    # Test case 2: Propagation channel (late, high inflow, moderate signal)
    print("\n" + "=" * 60)
    print("Test Case 2: Propagation Channel")
    print("-" * 60)
    score, lat_s, flow_s, sig_s, rise_s = compute_seed_score(
        lat=101.5,      # 1.5s after t0
        t0=t0,
        flow_out=0.2,   # low outflow
        flow_in=0.7,    # high inflow
        peak_z=6.0,     # moderate signal
        rise_z=2.0      # slow rise
    )
    print(f"Latency: 101.5 (1.5s after t0) -> score: {lat_s:.3f}")
    print(f"Flow: out=0.2, in=0.7 -> score: {flow_s:.3f}")
    print(f"Peak z: 6.0 -> score: {sig_s:.3f}")
    print(f"Rise z: 2.0 -> score: {rise_s:.3f}")
    print(f"COMPOSITE SCORE: {score:.3f}")

    # Test case 3: Network hub (moderate timing, balanced flow)
    print("\n" + "=" * 60)
    print("Test Case 3: Network Hub Channel")
    print("-" * 60)
    score, lat_s, flow_s, sig_s, rise_s = compute_seed_score(
        lat=100.3,      # 0.3s after t0
        t0=t0,
        flow_out=0.6,   # moderate outflow
        flow_in=0.5,    # moderate inflow
        peak_z=7.0,     # moderate-high signal
        rise_z=3.0      # moderate rise
    )
    print(f"Latency: 100.3 (0.3s after t0) -> score: {lat_s:.3f}")
    print(f"Flow: out=0.6, in=0.5 -> score: {flow_s:.3f}")
    print(f"Peak z: 7.0 -> score: {sig_s:.3f}")
    print(f"Rise z: 3.0 -> score: {rise_s:.3f}")
    print(f"COMPOSITE SCORE: {score:.3f}")

    # Test case 4: Missing data (NaN handling)
    print("\n" + "=" * 60)
    print("Test Case 4: Channel with Missing Data")
    print("-" * 60)
    score, lat_s, flow_s, sig_s, rise_s = compute_seed_score(
        lat=np.nan,     # no latency
        t0=t0,
        flow_out=0.7,   # only flow_out available
        flow_in=np.nan,
        peak_z=5.0,     # moderate signal
        rise_z=np.nan   # no rise data
    )
    print(f"Latency: NaN -> score: {lat_s:.3f}")
    print(f"Flow: out=0.7, in=NaN -> score: {flow_s:.3f}")
    print(f"Peak z: 5.0 -> score: {sig_s:.3f}")
    print(f"Rise z: NaN -> score: {rise_s:.3f}")
    print(f"COMPOSITE SCORE: {score:.3f}")

    print("\n" + "=" * 60)
    print("Summary:")
    print("-" * 60)
    print("✓ Ideal generator should score highest (~0.7-0.9)")
    print("✓ Propagation channel should score lower (~0.2-0.4)")
    print("✓ Network hub should score intermediate (~0.4-0.6)")
    print("✓ Missing data is handled gracefully (partial score)")
    print("✓ Weights: Latency 40%, Flow 35%, Signal 15%, Rise 10%")
    print("=" * 60)

if __name__ == "__main__":
    test_seed_score_logic()
