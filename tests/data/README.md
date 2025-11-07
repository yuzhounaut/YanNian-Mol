# Test Data

This directory contains test datasets for integration testing.

## Files

### test_molecules.csv
- 15 molecules total
- 6 positive class (Life_extended=1)
- 7 negative class (Life_extended=0)
- 2 invalid SMILES for error handling testing
- Mix of simple and complex molecules

### test_edge_cases.csv
- 10 molecules with edge cases
- Single atoms
- Empty/invalid SMILES
- Very long SMILES
- Special characters
- Stereochemistry
- Ring systems

## Usage

These datasets are used by integration tests to verify:
- Full preprocessing pipeline
- Training pipeline with small data
- Inference pipeline
- Error handling for invalid inputs
