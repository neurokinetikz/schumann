#!/usr/bin/env python3
"""
Examples of how to run null control test with different file selections
"""

from null_control_1_random_triplets import get_all_files, DATASET_SELECTION

# ============================================================================
# Example 1: Use default selection (all datasets)
# ============================================================================
print("Example 1: Default selection (all datasets)")
print("=" * 80)
files = get_all_files()
print(f"Total files: {len(files)}")
print()


# ============================================================================
# Example 2: Only specific datasets
# ============================================================================
print("Example 2: Only FILES and PHYSF datasets")
print("=" * 80)
custom_selection = {
    'FILES': True,
    'KAGGLE': False,
    'MPENG': False,
    'MPENG1': False,
    'MPENG2': False,
    'VEP': False,
    'PHYSF': True,
    'INSIGHT': False,
    'MUSE': False,
}
files = get_all_files(custom_selection)
print(f"Total files: {len(files)}")
print()


# ============================================================================
# Example 3: Only test files for quick testing
# ============================================================================
print("Example 3: Only test FILES for quick testing")
print("=" * 80)
test_only = {
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
files = get_all_files(test_only)
print(f"Total files: {len(files)}")
print()


# ============================================================================
# Example 4: All MPENG datasets
# ============================================================================
print("Example 4: All MPENG datasets combined")
print("=" * 80)
mpeng_only = {
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
files = get_all_files(mpeng_only)
print(f"Total files: {len(files)}")
print()


# ============================================================================
# To run the full test with a custom selection, you can:
# ============================================================================
print("To run the null control test with custom selection:")
print("=" * 80)
print("""
Option 1: Edit DATASET_SELECTION at the top of null_control_1_random_triplets.py

Option 2: In a notebook or script:
    from null_control_1_random_triplets import main, DATASET_SELECTION

    # Modify selection
    DATASET_SELECTION['KAGGLE'] = False
    DATASET_SELECTION['MUSE'] = False

    # Run test
    main()

Option 3: Pass custom selection programmatically (requires modifying main() slightly)

Option 4: Use command-line arguments (would require adding argparse)
""")
