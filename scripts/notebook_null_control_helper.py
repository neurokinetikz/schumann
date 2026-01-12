"""
Notebook helper for running null control tests.
Import this at the top of your Jupyter notebook.

Example usage in notebook:
    from notebook_null_control_helper import *

    # Quick test
    quick_test()

    # Custom selection
    run_with(FILES=True, PHYSF=True)

    # Show available datasets
    show_datasets()
"""

from run_null_control import (
    run_all, run_test_only, run_physf_only,
    run_all_mpeng, run_no_mpeng2, run_custom,
    ALL_DATASETS, TEST_ONLY, PHYSF_ONLY, ALL_MPENG, NO_MPENG2
)
from null_control_1_random_triplets import get_all_files


# ============================================================================
# Simplified aliases for notebook use
# ============================================================================

def quick_test():
    """
    Quick validation test with 3 files (~5 minutes).
    Run this first to make sure everything works!
    """
    print("🚀 Quick Test Mode (3 files)")
    print("This should take about 5 minutes")
    print()
    run_test_only()


def run_with(**kwargs):
    """
    Run null control test with custom dataset selection.

    Usage:
        run_with(FILES=True, PHYSF=True)
        run_with(MPENG=True, MPENG1=True)
        run_with(VEP=True, KAGGLE=True, PHYSF=True)

    Available datasets:
        FILES, KAGGLE, MPENG, MPENG1, MPENG2, VEP, PHYSF, INSIGHT, MUSE
    """
    run_custom(**kwargs)


def full_analysis():
    """
    Run complete analysis with all 3,574 files.
    WARNING: This will take several hours!
    """
    print("⚠️  FULL ANALYSIS MODE (3,574 files)")
    print("This will take several hours to complete.")
    response = input("Continue? (yes/no): ")
    if response.lower() in ['yes', 'y']:
        run_all()
    else:
        print("Cancelled.")


def medium_test():
    """
    Medium test excluding MPENG2 (365 files, ~1-2 hours).
    Good balance between runtime and sample size.
    """
    print("📊 Medium Test Mode (365 files)")
    print("This should take 1-2 hours")
    print()
    run_no_mpeng2()


def show_datasets():
    """Show all available datasets and file counts."""
    print("=" * 80)
    print("Available Datasets")
    print("=" * 80)
    print()

    # Get counts for each dataset (MPENG2 reduced after moving non-EEG files to other/)
    datasets_info = {
        'FILES': ('Individual test files', 5),
        'KAGGLE': ('data/mainData', 15),
        'MPENG': ('data/mpeng', 76),
        'MPENG1': ('data/mpeng1', 132),
        'MPENG2': ('data/mpeng2', 692),  # Was 3209, non-EEG files moved to other/
        'VEP': ('data/vep', 64),
        'PHYSF': ('data/PhySF', 46),
        'INSIGHT': ('data/insight', 12),
        'MUSE': ('data/muse', 17),
    }

    for name, (location, count) in datasets_info.items():
        print(f"  {name:10} - {location:20} - {count:5,} files")

    print()
    print(f"  {'TOTAL':10} - {'':20} - {sum(c for _, c in datasets_info.values()):5,} files")
    print()
    print("Usage:")
    print("  run_with(FILES=True, PHYSF=True)  # Select specific datasets")
    print()


def list_presets():
    """Show all available preset configurations."""
    print("=" * 80)
    print("Available Presets")
    print("=" * 80)
    print()
    print("quick_test()     - Test files only (3 files, ~5 min)")
    print("medium_test()    - All except MPENG2 (367 files, ~1-2 hours)")
    print("full_analysis()  - All datasets (1,059 files, several hours)")
    print()
    print("run_physf_only() - PHYSF dataset (46 files)")
    print("run_all_mpeng()  - All MPENG datasets (900 files)")
    print()
    print("Or create custom selection:")
    print("  run_with(FILES=True, PHYSF=True, VEP=True)")
    print()


def preview_selection(**kwargs):
    """
    Preview how many files would be selected without running the test.

    Usage:
        preview_selection(FILES=True, PHYSF=True)
    """
    # Create selection dict
    custom_selection = {k: False for k in ALL_DATASETS.keys()}
    for key, value in kwargs.items():
        if key.upper() in custom_selection:
            custom_selection[key.upper()] = value

    print("=" * 80)
    print("Preview: File Selection")
    print("=" * 80)
    print()

    files = get_all_files(custom_selection)
    print(f"\nTotal files selected: {len(files)}")
    print(f"Estimated runtime: {estimate_runtime(len(files))}")
    print()
    print("To run this selection:")
    print(f"  run_with({', '.join(f'{k}=True' for k, v in custom_selection.items() if v)})")
    print()


def estimate_runtime(n_files):
    """Estimate runtime based on number of files."""
    # Assume ~1-3 minutes per file average
    min_minutes = n_files * 1
    max_minutes = n_files * 3

    if max_minutes < 60:
        return f"{min_minutes}-{max_minutes} minutes"
    else:
        min_hours = min_minutes / 60
        max_hours = max_minutes / 60
        return f"{min_hours:.1f}-{max_hours:.1f} hours"


# ============================================================================
# Print helpful message on import
# ============================================================================

print("=" * 80)
print("Null Control Test - Notebook Helper Loaded ✓")
print("=" * 80)
print()
print("Quick commands:")
print("  show_datasets()   - List all available datasets")
print("  quick_test()      - Run quick validation (3 files, ~5 min)")
print("  medium_test()     - Run medium test (365 files, ~1-2 hours)")
print("  full_analysis()   - Run complete analysis (3,574 files)")
print()
print("Custom selection:")
print("  run_with(FILES=True, PHYSF=True)")
print("  preview_selection(FILES=True, PHYSF=True)")
print()
print("For more info:")
print("  list_presets()    - Show all preset configurations")
print()
