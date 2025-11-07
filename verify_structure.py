"""Verify the project structure is set up correctly."""

import os
from pathlib import Path


def check_file_exists(path: str) -> bool:
    """Check if a file exists."""
    exists = os.path.exists(path)
    status = "✓" if exists else "✗"
    print(f"{status} {path}")
    return exists


def main():
    """Verify project structure."""
    print("Verifying project structure...\n")
    
    required_files = [
        # Package structure
        "lifespan_predictor/__init__.py",
        "lifespan_predictor/config/__init__.py",
        "lifespan_predictor/config/config.py",
        "lifespan_predictor/config/default_config.yaml",
        "lifespan_predictor/data/__init__.py",
        "lifespan_predictor/models/__init__.py",
        "lifespan_predictor/training/__init__.py",
        "lifespan_predictor/utils/__init__.py",
        
        # Tests
        "tests/__init__.py",
        "tests/test_config.py",
        
        # Notebooks
        "notebooks/README.md",
        
        # Project files
        "setup.py",
        "requirements.txt",
        "requirements-dev.txt",
        "README.md",
        ".gitignore",
        "MANIFEST.in",
    ]
    
    print("Required files:")
    all_exist = all(check_file_exists(f) for f in required_files)
    
    print("\n" + "="*50)
    if all_exist:
        print("✓ All required files are present!")
        print("\nProject structure is set up correctly.")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Install package in dev mode: pip install -e .")
        print("3. Run tests: pytest tests/test_config.py")
        return 0
    else:
        print("✗ Some required files are missing!")
        return 1


if __name__ == "__main__":
    exit(main())
