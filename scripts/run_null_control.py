#!/usr/bin/env python3
"""
Quick runner for null control test with dataset selection.
Can be imported in notebooks or run from command line.
"""

from null_control_1_random_triplets import main

# ============================================================================
# Predefined dataset configurations
# ============================================================================

ALL_DATASETS = {
    'FILES': True,
    'KAGGLE': True,
    'MPENG': True,
    'MPENG1': True,
    'MPENG2': True,
    'VEP': True,
    'PHYSF': True,
    'INSIGHT': True,
    'MUSE': True,
}

TEST_ONLY = {
    'FILES': True,
    'KAGGLE': False,
    'MPENG': False,
    'MPENG1': False,
    'MPENG2': False,
    'VEP': False,
    'PHYSF': False,
    'INSIGHT': False,
    'MUSE': False,
}

PHYSF_ONLY = {
    'FILES': False,
    'KAGGLE': False,
    'MPENG': False,
    'MPENG1': False,
    'MPENG2': False,
    'VEP': False,
    'PHYSF': True,
    'INSIGHT': False,
    'MUSE': False,
}

ALL_MPENG = {
    'FILES': False,
    'KAGGLE': False,
    'MPENG': True,
    'MPENG1': True,
    'MPENG2': True,
    'VEP': False,
    'PHYSF': False,
    'INSIGHT': False,
    'MUSE': False,
}

NO_MPENG2 = {  # Exclude large MPENG2 dataset
    'FILES': True,
    'KAGGLE': True,
    'MPENG': True,
    'MPENG1': True,
    'MPENG2': False,  # Skip the largest dataset
    'VEP': True,
    'PHYSF': True,
    'INSIGHT': True,
    'MUSE': True,
}


# ============================================================================
# Convenience functions
# ============================================================================

def run_all():
    """Run null control test with all datasets"""
    print("Running with ALL datasets...")
    main(dataset_selection=ALL_DATASETS)


def run_test_only():
    """Run null control test with only test files (quick validation)"""
    print("Running with TEST files only...")
    main(dataset_selection=TEST_ONLY)


def run_physf_only():
    """Run null control test with only PHYSF dataset"""
    print("Running with PHYSF dataset only...")
    main(dataset_selection=PHYSF_ONLY)


def run_all_mpeng():
    """Run null control test with all MPENG datasets"""
    print("Running with all MPENG datasets...")
    main(dataset_selection=ALL_MPENG)


def run_no_mpeng2():
    """Run null control test excluding MPENG2 (the largest dataset)"""
    print("Running with all datasets EXCEPT MPENG2...")
    main(dataset_selection=NO_MPENG2)


def run_custom(**kwargs):
    """
    Run null control test with custom dataset selection.

    Args:
        **kwargs: Dataset names as keyword arguments (e.g., FILES=True, KAGGLE=False)

    Example:
        run_custom(FILES=True, PHYSF=True, KAGGLE=False)
    """
    # Start with all False
    custom_selection = {k: False for k in ALL_DATASETS.keys()}

    # Update with provided arguments
    for key, value in kwargs.items():
        if key.upper() in custom_selection:
            custom_selection[key.upper()] = value
        else:
            print(f"Warning: Unknown dataset '{key}' - valid options: {list(custom_selection.keys())}")

    print(f"Running with custom selection: {custom_selection}")
    main(dataset_selection=custom_selection)


# ============================================================================
# Main (for command-line usage)
# ============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()

        if mode == 'all':
            run_all()
        elif mode == 'test':
            run_test_only()
        elif mode == 'physf':
            run_physf_only()
        elif mode == 'mpeng':
            run_all_mpeng()
        elif mode == 'no-mpeng2':
            run_no_mpeng2()
        else:
            print(f"Unknown mode: {mode}")
            print("Valid modes: all, test, physf, mpeng, no-mpeng2")
            sys.exit(1)
    else:
        # Default: show usage
        print("="*80)
        print("Null Control Test - Quick Runner")
        print("="*80)
        print()
        print("Usage:")
        print("  python3 run_null_control.py <mode>")
        print()
        print("Modes:")
        print("  all        - Run with all datasets (1059 files)")
        print("  test       - Run with test files only (3 files) - QUICK TEST")
        print("  physf      - Run with PHYSF dataset only (46 files)")
        print("  mpeng      - Run with all MPENG datasets (900 files)")
        print("  no-mpeng2  - Run without MPENG2, rest included (367 files)")
        print()
        print("From Python/Notebook:")
        print("  from run_null_control import run_all, run_test_only, run_custom")
        print("  run_test_only()  # Quick test")
        print("  run_custom(FILES=True, PHYSF=True)  # Custom selection")
        print()
