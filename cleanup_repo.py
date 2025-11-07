#!/usr/bin/env python
"""
Clean up repository by removing unnecessary files.
"""

import subprocess
import sys

# Files and directories to remove
FILES_TO_REMOVE = [
    # Old notebooks
    "LifespanPredict.ipynb",
    "LifespanPredictClass.ipynb",
    "run_featurizer_lifespan.ipynb",
    "run_fingerprint_lifespan_simplified.ipynb",
    "Download_HerbalMDB_10000.ipynb",
    "pubmed anti-aging compound search key word.ipynb",
    
    # Data files
    "train.csv",
    "train - original.csv",
    "train 20250530.csv",
    "train_with_canonical.csv",
    "train_with_common_names.csv",
    "train_with_pubchem.csv",
    "test.csv",
    "TrainSample.csv",
    "QuerySample.csv",
    "bace.csv",
    "elixirseeker_trainset.csv",
    "geroprotectors_database2025-07-23.csv",
    "csv-anti-aging-set_20250909_with_urls_with_download_results.csv",
    "WOS_Aging_MAJR_Drug_Effects.csv",
    "foodb_2020_04_07_csv/",
    
    # Task summaries
    "TASK_1_SUMMARY.md",
    "TASK_3_SUMMARY.md",
    "TASK_4_SUMMARY.md",
    "TASK_5_SUMMARY.md",
    "TASK_6_SUMMARY.md",
    "TASK_10_SUMMARY.md",
    "TASK_12_SUMMARY.md",
    "TASK_13_SUMMARY.md",
    "TASK_14_SUMMARY.md",
    
    # Other unnecessary files
    "CODE_REVIEW_AND_OPTIMIZATION.md",
    "GITHUB_RELEASE_GUIDE.md",
    "QUICK_START_GITHUB.md",
    "verify_structure.py",
    "push_to_github.bat",
    
    # Spec files (internal development)
    ".kiro/",
    ".flake8",
]


def run_command(cmd):
    """Run a git command."""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
    return result.returncode == 0


def main():
    """Main cleanup function."""
    print("=" * 80)
    print("Repository Cleanup Script")
    print("=" * 80)
    print()
    
    print(f"Will remove {len(FILES_TO_REMOVE)} files/directories")
    print()
    
    # Confirm
    response = input("Continue? (y/n): ").strip().lower()
    if response != 'y':
        print("Cancelled.")
        sys.exit(0)
    
    print()
    print("Removing files...")
    print()
    
    removed_count = 0
    failed_count = 0
    
    for file_path in FILES_TO_REMOVE:
        if run_command(f'git rm -rf "{file_path}"'):
            print(f"✓ Removed: {file_path}")
            removed_count += 1
        else:
            print(f"✗ Failed or not found: {file_path}")
            failed_count += 1
    
    print()
    print("=" * 80)
    print(f"Removed: {removed_count} files/directories")
    print(f"Failed/Not found: {failed_count}")
    print("=" * 80)
    print()
    
    # Commit changes
    print("Committing changes...")
    if run_command('git commit -m "Clean up repository: remove unnecessary files"'):
        print("✓ Changes committed")
    else:
        print("✗ Commit failed")
        sys.exit(1)
    
    print()
    print("=" * 80)
    print("Cleanup complete!")
    print("=" * 80)
    print()
    print("Next step: Push to GitHub")
    print("  git push origin main")
    print()


if __name__ == "__main__":
    main()
