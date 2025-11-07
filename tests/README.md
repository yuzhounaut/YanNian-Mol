# Tests

This directory contains all tests for the lifespan predictor project.

## Test Structure

### Unit Tests

Unit tests verify individual components in isolation:

- `test_config.py` - Configuration management tests
- `test_preprocessing.py` - Data preprocessing tests
- `test_featurizers.py` - Molecular featurization tests
- `test_fingerprints.py` - Fingerprint generation tests
- `test_dataset.py` - Dataset and data loader tests
- `test_models.py` - Model architecture tests
- `test_metrics.py` - Metrics computation tests
- `test_training.py` - Training loop tests
- `test_utils.py` - Utility function tests

### Integration Tests

Integration tests verify end-to-end workflows:

- `test_integration.py` - Full pipeline tests
  - Preprocessing pipeline
  - Training pipeline
  - Inference pipeline
  - Edge cases and error handling

### Test Data

The `data/` directory contains small test datasets:

- `test_molecules.csv` - 15 molecules for quick testing
- `test_edge_cases.csv` - Edge cases and invalid data
- `README.md` - Test data documentation

## Running Tests

### Quick Start

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_integration.py

# Run with coverage
pytest --cov=lifespan_predictor
```

### Test Markers

Tests are marked with categories:

```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Exclude slow tests
pytest -m "not slow"

# Run GPU tests (requires GPU)
pytest -m gpu
```

### Common Options

```bash
# Verbose output
pytest -v

# Stop on first failure
pytest -x

# Show local variables in tracebacks
pytest -l

# Run tests in parallel
pytest -n auto

# Generate HTML coverage report
pytest --cov=lifespan_predictor --cov-report=html
```

## Writing Tests

### Test Structure

```python
import pytest
from lifespan_predictor.module import function

class TestFeature:
    """Tests for specific feature."""
    
    @pytest.fixture
    def sample_data(self):
        """Fixture providing test data."""
        return {"key": "value"}
    
    @pytest.mark.unit
    def test_basic_functionality(self, sample_data):
        """Test basic functionality."""
        result = function(sample_data)
        assert result is not None
    
    @pytest.mark.unit
    def test_error_handling(self):
        """Test error handling."""
        with pytest.raises(ValueError):
            function(invalid_input)
```

### Best Practices

1. **One test, one assertion**: Each test should verify one specific behavior
2. **Use fixtures**: Share setup code using pytest fixtures
3. **Use markers**: Mark tests appropriately (unit, integration, slow, gpu)
4. **Test edge cases**: Include tests for invalid inputs and error conditions
5. **Descriptive names**: Test names should clearly describe what they test
6. **Independent tests**: Tests should be independent and runnable in any order

### Fixtures

Common fixtures are defined in `conftest.py`:

```python
@pytest.fixture
def temp_dir():
    """Create temporary directory for test outputs."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def config():
    """Create test configuration."""
    return Config()
```

## Coverage

### Viewing Coverage

```bash
# Generate HTML coverage report
pytest --cov=lifespan_predictor --cov-report=html

# Open report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```

### Coverage Goals

- Overall coverage: > 80%
- Core modules (models, training): > 90%
- Utility modules: > 70%

## Continuous Integration

Tests are automatically run on GitHub Actions for:

- Multiple Python versions (3.9, 3.10, 3.11)
- Multiple operating systems (Ubuntu, Windows, macOS)
- Pull requests and pushes to main/develop branches

See `.github/workflows/ci.yml` for CI configuration.

## Troubleshooting

### Import Errors

```bash
# Install package in development mode
pip install -e .
```

### Missing Dependencies

```bash
# Install development dependencies
pip install -r requirements-dev.txt
```

### Slow Tests

```bash
# Skip slow tests
pytest -m "not slow"

# Run tests in parallel
pytest -n auto
```

### GPU Tests on CPU

```bash
# Skip GPU tests
pytest -m "not gpu"
```

## Additional Resources

- [pytest documentation](https://docs.pytest.org/)
- [pytest-cov documentation](https://pytest-cov.readthedocs.io/)
- [Testing best practices](https://docs.python-guide.org/writing/tests/)
