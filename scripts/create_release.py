#!/usr/bin/env python
"""
Script to create a release package.

This script:
1. Validates the package structure
2. Runs tests
3. Builds the distribution
4. Creates a git tag (if in a git repository)
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run_command(cmd, check=True):
    """Run a shell command and return the result."""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"Error: {result.stderr}")
        sys.exit(1)
    return result


def validate_package():
    """Validate package structure."""
    print("\n" + "=" * 80)
    print("VALIDATING PACKAGE STRUCTURE")
    print("=" * 80)

    required_files = [
        "setup.py",
        "README.md",
        "CHANGELOG.md",
        "VERSION",
        "requirements.txt",
        "lifespan_predictor/__init__.py",
    ]

    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)

    if missing_files:
        print(f"Error: Missing required files: {missing_files}")
        sys.exit(1)

    print("✓ All required files present")


def run_tests():
    """Run the test suite."""
    print("\n" + "=" * 80)
    print("RUNNING TESTS")
    print("=" * 80)

    result = run_command("python -m pytest tests/ -v", check=False)
    if result.returncode != 0:
        print("Warning: Some tests failed. Continue anyway? (y/n)")
        response = input().strip().lower()
        if response != "y":
            sys.exit(1)
    else:
        print("✓ All tests passed")


def run_linting():
    """Run code quality checks."""
    print("\n" + "=" * 80)
    print("RUNNING CODE QUALITY CHECKS")
    print("=" * 80)

    # Run flake8
    result = run_command(
        "python -m flake8 lifespan_predictor/ --count --max-line-length=100",
        check=False,
    )
    if result.returncode != 0:
        print("Warning: Linting issues found. Continue anyway? (y/n)")
        response = input().strip().lower()
        if response != "y":
            sys.exit(1)
    else:
        print("✓ No linting issues")


def build_distribution():
    """Build the distribution packages."""
    print("\n" + "=" * 80)
    print("BUILDING DISTRIBUTION")
    print("=" * 80)

    # Clean previous builds
    run_command("python -m pip install --upgrade build", check=True)

    # Remove old dist files
    if Path("dist").exists():
        import shutil

        shutil.rmtree("dist")
    if Path("build").exists():
        import shutil

        shutil.rmtree("build")

    # Build
    run_command("python -m build", check=True)

    print("✓ Distribution built successfully")
    print("\nDistribution files:")
    for file in Path("dist").glob("*"):
        print(f"  - {file}")


def create_git_tag(version):
    """Create a git tag for the release."""
    print("\n" + "=" * 80)
    print("CREATING GIT TAG")
    print("=" * 80)

    # Check if we're in a git repository
    result = run_command("git status", check=False)
    if result.returncode != 0:
        print("Not in a git repository, skipping tag creation")
        return

    # Check for uncommitted changes
    result = run_command("git status --porcelain", check=False)
    if result.stdout.strip():
        print("Warning: Uncommitted changes detected")
        print("Commit changes before creating a release? (y/n)")
        response = input().strip().lower()
        if response == "y":
            run_command("git add -A")
            run_command(f'git commit -m "Release v{version}"')

    # Create tag
    tag_name = f"v{version}"
    result = run_command(f'git tag -a {tag_name} -m "Release {tag_name}"', check=False)

    if result.returncode == 0:
        print(f"✓ Created git tag: {tag_name}")
        print(f"\nTo push the tag, run: git push origin {tag_name}")
    else:
        print(f"Warning: Could not create tag (may already exist)")


def main():
    """Main release creation function."""
    parser = argparse.ArgumentParser(description="Create a release package")
    parser.add_argument(
        "--skip-tests", action="store_true", help="Skip running tests"
    )
    parser.add_argument(
        "--skip-lint", action="store_true", help="Skip linting checks"
    )
    parser.add_argument(
        "--skip-tag", action="store_true", help="Skip creating git tag"
    )
    args = parser.parse_args()

    # Read version
    version_file = Path("VERSION")
    if not version_file.exists():
        print("Error: VERSION file not found")
        sys.exit(1)

    version = version_file.read_text().strip()
    print(f"\nCreating release for version: {version}")

    # Validate package
    validate_package()

    # Run tests
    if not args.skip_tests:
        run_tests()

    # Run linting
    if not args.skip_lint:
        run_linting()

    # Build distribution
    build_distribution()

    # Create git tag
    if not args.skip_tag:
        create_git_tag(version)

    print("\n" + "=" * 80)
    print("RELEASE CREATION COMPLETE")
    print("=" * 80)
    print(f"\nVersion {version} is ready for release!")
    print("\nNext steps:")
    print("1. Review the distribution files in dist/")
    print("2. Test installation: pip install dist/lifespan_predictor-*.whl")
    print("3. Push git tag: git push origin v" + version)
    print("4. Upload to PyPI: python -m twine upload dist/*")


if __name__ == "__main__":
    main()
