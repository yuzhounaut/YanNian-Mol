# CI/CD Setup and Testing Guide

This document describes the continuous integration and continuous deployment (CI/CD) setup for the lifespan predictor project.

## Overview

The project uses GitHub Actions for automated testing, linting, and code quality checks. Tests are run on multiple operating systems and Python versions to ensure compatibility.

## Test Structure

### Test Categories

Tests are organized into several categories using pytest markers:

- **Unit tests** (`@pytest.mark.unit`): Fast tests for individual functions and classes
- **Integration tests** (`@pytest.mark.integration`): Tests for full pipelines and workflows
- **Slow tests** (`@pytest.mark.slow`): Long-running tests (excluded from CI by default)
- **GPU tests** (`@pytest.mark.gpu`): Tests requiring GPU (run manually)

### Test Files

```
tests/
├── data/                      # Test datasets
│   ├── test_molecules.csv     # Small test dataset (15 molecules)
│   ├── test_edge_cases.csv    # Edge cases and invalid data
│   └── README.md              # Test data documentation
├── test_config.py             # Configuration tests
├── test_preprocessing.py      # Data preprocessing tests
├── test_featurizers.py        # Featurization tests
├── test_fingerprints.py       # Fingerprint generation tests
├── test_dataset.py            # Dataset and data loader tests
├── test_models.py             # Model architecture tests
├── test_metrics.py            # Metrics computation tests
├── test_training.py           # Training loop tests
├── test_utils.py              # Utility function tests
└── test_integration.py        # End-to-end integration tests
```

## Running Tests Locally

### Quick Start

```bash
# Install development dependencies
make install-dev

# Run unit tests (fast)
make test

# Run integration tests
make test-integration

# Run all tests including slow tests
make test-all
```

### Using pytest directly

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_integration.py

# Run specific test class
pytest tests/test_integration.py::TestPreprocessingPipeline

# Run specific test function
pytest tests/test_integration.py::TestPreprocessingPipeline::test_load_and_clean_pipeline

# Run tests with specific marker
pytest -m "unit"                    # Only unit tests
pytest -m "integration"             # Only integration tests
pytest -m "not slow"                # Exclude slow tests

# Run with coverage report
pytest --cov=lifespan_predictor --cov-report=html

# Run with verbose output
pytest -v

# Stop on first failure
pytest -x

# Show local variables in tracebacks
pytest -l
```

## Code Quality Checks

### Linting with flake8

```bash
# Run linting
make lint

# Or directly
flake8 lifespan_predictor tests
```

Configuration is in `.flake8` file.

### Code Formatting

```bash
# Format code with black and isort
make format

# Check formatting without modifying files
black --check lifespan_predictor tests
isort --check-only lifespan_predictor tests
```

Configuration is in `pyproject.toml`.

### Type Checking with mypy

```bash
# Run type checking
make type-check

# Or directly
mypy lifespan_predictor --config-file mypy.ini
```

Configuration is in `mypy.ini`.

## GitHub Actions Workflows

### Main CI Workflow (`.github/workflows/ci.yml`)

Runs on every push and pull request to `main` and `develop` branches.

#### Jobs

1. **test**: Runs unit tests on multiple OS and Python versions
   - OS: Ubuntu, Windows, macOS
   - Python: 3.9, 3.10, 3.11
   - Steps:
     - Install dependencies
     - Lint with flake8
     - Type check with mypy
     - Run unit tests with coverage
     - Upload coverage to Codecov

2. **integration-test**: Runs integration tests
   - OS: Ubuntu
   - Python: 3.10
   - Steps:
     - Install dependencies
     - Run integration tests
     - Upload coverage to Codecov

3. **code-quality**: Checks code formatting and style
   - OS: Ubuntu
   - Python: 3.10
   - Steps:
     - Check formatting with black
     - Check import sorting with isort
     - Lint with flake8

### Workflow Status

You can view the status of CI runs in the GitHub Actions tab of your repository.

## Test Data

### test_molecules.csv

Small dataset for quick testing:
- 15 molecules total
- 6 positive class (Life_extended=1)
- 7 negative class (Life_extended=0)
- 2 invalid SMILES for error handling testing
- Mix of simple and complex molecules

### test_edge_cases.csv

Edge cases for robustness testing:
- Single atoms
- Empty/invalid SMILES
- Very long SMILES
- Special characters
- Stereochemistry
- Ring systems

## Coverage Reports

### Viewing Coverage

After running tests with coverage:

```bash
# Generate HTML coverage report
pytest --cov=lifespan_predictor --cov-report=html

# Open report in browser
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```

### Coverage Goals

- Overall coverage: > 80%
- Core modules (models, training): > 90%
- Utility modules: > 70%

## Troubleshooting

### Common Issues

#### Tests fail with import errors

```bash
# Make sure package is installed in development mode
pip install -e .
```

#### Tests fail with missing dependencies

```bash
# Install all development dependencies
pip install -r requirements-dev.txt
```

#### GPU tests fail on CPU-only machine

```bash
# Skip GPU tests
pytest -m "not gpu"
```

#### Integration tests are slow

```bash
# Run only unit tests
pytest -m "unit and not slow"
```

#### Coverage report not generated

```bash
# Install pytest-cov
pip install pytest-cov

# Run with coverage
pytest --cov=lifespan_predictor
```

## Best Practices

### Writing Tests

1. **Keep tests focused**: Each test should verify one specific behavior
2. **Use fixtures**: Share setup code using pytest fixtures
3. **Use markers**: Mark tests appropriately (unit, integration, slow, gpu)
4. **Test edge cases**: Include tests for invalid inputs and error conditions
5. **Use descriptive names**: Test names should clearly describe what they test
6. **Avoid test interdependencies**: Tests should be independent and runnable in any order

### Example Test Structure

```python
import pytest
from lifespan_predictor.data.preprocessing import clean_smiles

class TestSmilesProcessing:
    """Tests for SMILES processing functions."""
    
    @pytest.fixture
    def valid_smiles(self):
        """Fixture providing valid SMILES strings."""
        return ["CCO", "CC(C)O", "c1ccccc1"]
    
    @pytest.mark.unit
    def test_clean_smiles_valid(self, valid_smiles):
        """Test cleaning of valid SMILES strings."""
        for smiles in valid_smiles:
            result = clean_smiles(smiles)
            assert result is not None
            assert isinstance(result, str)
    
    @pytest.mark.unit
    def test_clean_smiles_invalid(self):
        """Test handling of invalid SMILES strings."""
        invalid_smiles = ["INVALID", "", None, 123]
        for smiles in invalid_smiles:
            result = clean_smiles(smiles)
            assert result is None
```

## Continuous Improvement

### Adding New Tests

When adding new features:

1. Write tests first (TDD approach)
2. Ensure tests cover both success and failure cases
3. Add appropriate markers
4. Update this documentation if needed

### Monitoring Test Performance

```bash
# Show slowest tests
pytest --durations=10

# Profile test execution
pytest --profile
```

### Updating CI Configuration

When modifying CI workflows:

1. Test changes locally first
2. Use branch protection rules
3. Require CI to pass before merging
4. Monitor CI run times and optimize if needed

## Additional Resources

- [pytest documentation](https://docs.pytest.org/)
- [GitHub Actions documentation](https://docs.github.com/en/actions)
- [flake8 documentation](https://flake8.pycqa.org/)
- [black documentation](https://black.readthedocs.io/)
- [mypy documentation](https://mypy.readthedocs.io/)
