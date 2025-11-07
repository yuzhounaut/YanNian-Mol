"""Setup script for lifespan_predictor package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_file = Path(__file__).parent / "README.md"
long_description = ""
if readme_file.exists():
    long_description = readme_file.read_text(encoding="utf-8")

setup(
    name="lifespan_predictor",
    version="0.1.0",
    author="Lifespan Predictor Team",
    description="A modular deep learning pipeline for predicting compound effects on C. elegans lifespan",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=["tests", "notebooks"]),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torch-geometric>=2.3.0",
        "dgl>=1.0.0",
        "dgllife>=0.3.0",
        "rdkit>=2022.09.1",
        "deepchem>=2.7.0",
        "numpy>=1.23.0",
        "pandas>=1.5.0",
        "scikit-learn>=1.2.0",
        "pydantic>=2.0.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "tensorboard>=2.12.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "joblib>=1.2.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
        ],
        "docs": [
            "sphinx>=6.0.0",
            "sphinx-rtd-theme>=1.2.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    include_package_data=True,
    package_data={
        "lifespan_predictor": ["config/default_config.yaml"],
    },
)
