#!/usr/bin/env python3
"""
Add this debug code to your actual script to see what values are being used.
"""

# Add this RIGHT BEFORE your FOOOF calls:

print("\n" + "=" * 70)
print("DEBUG: Actual Parameter Values")
print("=" * 70)

# Print your configuration
print(f"\nYour configuration:")
print(f"  CANON = {CANON}")
print(f"  _half_bw = {_half_bw}")
print(f"  type(_half_bw) = {type(_half_bw)}")

# Show which values map to which
if isinstance(_half_bw, (list, tuple, np.ndarray)):
    print(f"\nPer-harmonic windows:")
    for i, (canon, hb) in enumerate(zip(CANON, _half_bw)):
        lo, hi = canon - hb, canon + hb
        print(f"  sr{i+1}: CANON={canon:6.2f} Hz, half_bw={hb}, window=[{lo:6.2f}, {hi:6.2f}]")
        if lo <= 34.27 <= hi:
            print(f"       ⚠️  34.27 Hz IS within this window!")
else:
    print(f"\n⚠️  WARNING: _half_bw is SCALAR!")
    print(f"  All harmonics will use window of ±{_half_bw} Hz")

# If you have the results already, check what was actually stored
if 'events' in _ign_out and not _ign_out['events'].empty:
    events = _ign_out['events']
    print(f"\nStored in DataFrame:")
    for idx, row in events.head(3).iterrows():
        freqs = row['ignition_freqs']
        if isinstance(freqs, str):
            freqs = eval(freqs)
        print(f"  Event {idx}: {freqs}")

        # Check which harmonic has 34.27
        if isinstance(freqs, (list, tuple)):
            for i, f in enumerate(freqs):
                if abs(f - 34.27) < 0.1:
                    print(f"       ^ sr{i+1} = {f:.4f} Hz (this is the 34.27 Hz peak)")

print("=" * 70 + "\n")
