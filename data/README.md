# Sample Data

This directory should contain small sample datasets for testing and demonstration.

## Expected Files

- `sample_train.csv` - Small training dataset (10-20 compounds)
- `sample_test.csv` - Small test dataset (5-10 compounds)

## Format

CSV files should have columns:
- `SMILES` - Molecular SMILES string
- `Life_extended` - Binary label (0 or 1)

## Usage

```python
from lifespan_predictor.data.preprocessing import load_and_clean_csv

# Load sample data
df = load_and_clean_csv('data/sample/sample_train.csv')
```

## Note

Full datasets are not included in the repository due to size.
Please refer to the main README for data sources and preparation instructions.
