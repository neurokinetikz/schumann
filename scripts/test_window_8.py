#!/usr/bin/env python3
"""Test script to render ignition window index 8"""

# This is the CORRECT way to render a single ignition window:
# If IGNITION_WINDOWS is a list of tuples, and you want window 8:

# WRONG (what you were doing):
# for idx, IGN_WIN in enumerate(IGNITION_WINDOWS[8]):  # This enumerates the tuple's elements!
#     test.six_panel(...)

# RIGHT (what you should do):
idx = 8
IGN_WIN = IGNITION_WINDOWS[idx]  # Get the 8th window tuple (start, end)

# Configure for this window
CFG = test.PanelConfig(
    palette='synthwave',  # or 'professional', 'nature', 'cyberpunk'
    show_phase_labels=True,
    show_phase_shading=True
)

# Render just this one window
test.six_panel(
    RECORDS,
    ELECTRODES,
    IGN_WIN,  # Single window tuple
    IGN_OUT,
    IGN_OUT['events']['ignition_freqs'][idx],
    CFG,
    SESSION_NAME
)

# If you want to render ALL windows, use this instead:
# for idx, IGN_WIN in enumerate(IGNITION_WINDOWS):  # Enumerate the full list
#     CFG = test.PanelConfig(palette='synthwave', show_phase_labels=True, show_phase_shading=True)
#     test.six_panel(RECORDS, ELECTRODES, IGN_WIN, IGN_OUT, IGN_OUT['events']['ignition_freqs'][idx], CFG, SESSION_NAME)
