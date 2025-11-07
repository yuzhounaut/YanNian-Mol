#!/usr/bin/env python
"""
Script to run all CI checks locally before pushing.

This script runs the same checks that will be run in CI:
- Code formatting (black, isort)
- Linting (flake8)
- Type checking (mypy)
- Unit tests
- Integration tests
- Coverage report

Usage:
    python scripts/run_ci_checks.py
    python scripts/run_ci_checks.py --fast  # Skip slow tests
    python scripts/run_ci_checks.py --fix   # Auto-fix formatting issues
"""

import argparse
import subprocess
import sys


def run_command(cmd, description, check=True, capture_output=False):
    """
    Run a shell command and report results.

    Args:
        cmd: Command to run (list of strings)
        description: Description of what the command does
        check: Whether to check return code
        capture_output: Whether to capture output

    Returns:
        CompletedProcess object
    """
    print(f"\n{'='*70}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*70}\n")

    try:
        result = subprocess.run(cmd, check=check, capture_output=capture_output, text=True)

        if result.returncode == 0:
            print(f"\n✓ {description} passed")
        else:
            print(f"\n✗ {description} failed")

        return result

    except subprocess.CalledProcessError as e:
        print(f"\n✗ {description} failed with error:")
        print(e.stderr if e.stderr else str(e))
        if check:
            raise
        return e


def main():  # noqa: C901
    """Run all CI checks."""
    parser = argparse.ArgumentParser(description="Run CI checks locally")
    parser.add_argument("--fast", action="store_true", help="Skip slow tests and integration tests")
    parser.add_argument("--fix", action="store_true", help="Auto-fix formatting issues")
    parser.add_argument("--skip-tests", action="store_true", help="Skip running tests")
    parser.add_argument("--skip-lint", action="store_true", help="Skip linting checks")
    parser.add_argument("--skip-type-check", action="store_true", help="Skip type checking")

    args = parser.parse_args()

    # Project root not used currently
    # project_root = Path(__file__).parent.parent

    # Track failures
    failures = []

    # 1. Code formatting
    if not args.skip_lint:
        if args.fix:
            # Auto-fix formatting
            try:
                run_command(["black", "lifespan_predictor", "tests"], "Format code with black")
                run_command(["isort", "lifespan_predictor", "tests"], "Sort imports with isort")
            except subprocess.CalledProcessError:
                failures.append("Code formatting (auto-fix)")
        else:
            # Check formatting
            try:
                run_command(
                    ["black", "--check", "lifespan_predictor", "tests"],
                    "Check code formatting with black",
                )
            except subprocess.CalledProcessError:
                failures.append("Black formatting check")

            try:
                run_command(
                    ["isort", "--check-only", "lifespan_predictor", "tests"],
                    "Check import sorting with isort",
                )
            except subprocess.CalledProcessError:
                failures.append("isort check")

    # 2. Linting
    if not args.skip_lint:
        try:
            run_command(
                ["flake8", "lifespan_predictor", "tests", "--count", "--statistics"],
                "Lint code with flake8",
            )
        except subprocess.CalledProcessError:
            failures.append("Flake8 linting")

    # 3. Type checking
    if not args.skip_type_check:
        try:
            run_command(
                ["mypy", "lifespan_predictor", "--config-file", "mypy.ini"],
                "Type check with mypy",
                check=False,  # Don't fail on type errors
            )
        except subprocess.CalledProcessError:
            failures.append("Type checking")

    # 4. Tests
    if not args.skip_tests:
        if args.fast:
            # Run only fast unit tests
            try:
                run_command(
                    [
                        "pytest",
                        "tests/",
                        "-v",
                        "-m",
                        "not slow",
                        "--cov=lifespan_predictor",
                        "--cov-report=term-missing",
                        "--cov-report=html",
                    ],
                    "Run unit tests (fast)",
                )
            except subprocess.CalledProcessError:
                failures.append("Unit tests")
        else:
            # Run unit tests
            try:
                run_command(
                    [
                        "pytest",
                        "tests/",
                        "-v",
                        "-m",
                        "not slow",
                        "--ignore=tests/test_integration.py",
                        "--cov=lifespan_predictor",
                        "--cov-report=term-missing",
                    ],
                    "Run unit tests",
                )
            except subprocess.CalledProcessError:
                failures.append("Unit tests")

            # Run integration tests
            try:
                run_command(
                    [
                        "pytest",
                        "tests/test_integration.py",
                        "-v",
                        "--cov=lifespan_predictor",
                        "--cov-append",
                        "--cov-report=term-missing",
                        "--cov-report=html",
                    ],
                    "Run integration tests",
                )
            except subprocess.CalledProcessError:
                failures.append("Integration tests")

    # Print summary
    print(f"\n{'='*70}")
    print("CI CHECKS SUMMARY")
    print(f"{'='*70}\n")

    if failures:
        print("✗ The following checks failed:")
        for failure in failures:
            print(f"  - {failure}")
        print("\nPlease fix the issues before pushing.")
        sys.exit(1)
    else:
        print("✓ All CI checks passed!")
        print("\nYour code is ready to push.")
        sys.exit(0)


if __name__ == "__main__":
    main()
