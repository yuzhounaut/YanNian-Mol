# Validation Guide

This document describes how to validate that the refactored implementation produces results consistent with the original notebooks.

## Overview

The validation process compares three key aspects:
1. **Preprocessing outputs** - Graph features and fingerprints
2. **Training metrics** - Model performance metrics
3. **Inference predictions** - Model predictions on test data

## Prerequisites

- Both original and refactored code installed
- Original notebooks: `LifespanPredictClass.ipynb`, `run_featurizer_lifespan.ipynb`, `run_fingerprint_lifespan_simplified.ipynb`
- New notebooks: `notebooks/01_data_preprocessing.ipynb`, `notebooks/02_model_training.ipynb`, `notebooks/03_inference.ipynb`
- Training data: `train.csv`, `test.csv`

## Validation Steps

### 1. Preprocessing Validation

**Objective**: Verify that graph features and fingerprints match between implementations.

**Steps**:

1. Run original preprocessing:
   ```bash
   # Run original featurizer
   jupyter nbconvert --execute run_featurizer_lifespan.ipynb
   
   # Run original fingerprint generator
   jupyter nbconvert --execute run_fingerprint_lifespan_simplified.ipynb
   ```

2. Run new preprocessing:
   ```bash
   jupyter nbconvert --execute notebooks/01_data_preprocessing.ipynb
   ```

3. Compare outputs:
   ```bash
   python scripts/validate_notebooks.py
   ```

**Expected Results**:
- Adjacency matrices should match exactly
- Node features should match within numerical tolerance (1e-6)
- Fingerprints should match exactly

**Troubleshooting**:
- If features differ, check SMILES canonicalization
- Verify RDKit version matches (>= 2022.09)
- Check random seed settings

### 2. Training Metrics Validation

**Objective**: Verify that model training produces similar performance metrics.

**Steps**:

1. Run original training:
   ```bash
   jupyter nbconvert --execute LifespanPredictClass.ipynb
   ```
   
   Record the final metrics:
   - Training AUC
   - Validation AUC
   - Training Accuracy
   - Validation Accuracy
   - Training Loss
   - Validation Loss

2. Run new training:
   ```bash
   jupyter nbconvert --execute notebooks/02_model_training.ipynb
   ```
   
   Record the same metrics from the output.

3. Compare metrics:
   - Metrics should be within 1-2% of each other
   - Loss curves should follow similar patterns
   - Best epoch should be similar (±2 epochs)

**Expected Results**:
- AUC difference < 0.02
- Accuracy difference < 0.02
- Loss values within 10%

**Note**: Due to randomness in training (even with fixed seeds), exact reproduction is not expected. The key is that performance is comparable.

**Troubleshooting**:
- Verify same random seed used
- Check learning rate and optimizer settings
- Ensure same data split used
- Verify model architecture matches

### 3. Inference Validation

**Objective**: Verify that model predictions are consistent.

**Steps**:

1. Save predictions from original model:
   ```python
   # In LifespanPredictClass.ipynb, add at the end:
   import numpy as np
   np.save('original_predictions.npy', test_predictions)
   np.save('original_labels.npy', test_labels)
   ```

2. Run new inference:
   ```bash
   jupyter nbconvert --execute notebooks/03_inference.ipynb
   ```

3. Compare predictions:
   ```python
   import numpy as np
   from scripts.validate_notebooks import NotebookValidator
   
   validator = NotebookValidator(tolerance=1e-3)
   
   original_preds = np.load('original_predictions.npy')
   new_preds = np.load('new_predictions.npy')
   labels = np.load('original_labels.npy')
   
   results = validator.validate_predictions(original_preds, new_preds, labels)
   print(results)
   ```

**Expected Results**:
- Predictions should be highly correlated (r > 0.99)
- Mean absolute difference < 0.01
- Same samples classified correctly/incorrectly

**Troubleshooting**:
- Verify same model checkpoint loaded
- Check preprocessing consistency
- Ensure same device (CPU/GPU) used

## Automated Validation

Run the complete validation script:

```bash
python scripts/validate_notebooks.py
```

This will:
1. Check preprocessing outputs
2. Provide instructions for manual metric comparison
3. Generate a validation report: `validation_report.json`

## Validation Report

The validation report (`validation_report.json`) contains:

```json
{
  "preprocessing": {
    "status": "PASS|FAIL|SKIP",
    "checks": [
      {
        "name": "adjacency_matrices",
        "status": "PASS",
        "message": "Adjacency matrices match"
      },
      ...
    ]
  },
  "training": {
    "status": "PASS|FAIL|SKIP",
    "checks": [...]
  },
  "inference": {
    "status": "PASS|FAIL|SKIP",
    "checks": [...]
  },
  "overall_status": "PASS|FAIL|PARTIAL|INCOMPLETE"
}
```

## Success Criteria

The refactored implementation is considered validated if:

1. ✓ Preprocessing outputs match within numerical tolerance
2. ✓ Training metrics are within acceptable range (±2%)
3. ✓ Predictions are highly consistent (correlation > 0.99)
4. ✓ No degradation in model performance
5. ✓ Code runs without errors

## Known Differences

Some expected differences between implementations:

1. **Training time**: Refactored code may be faster due to optimizations
2. **Memory usage**: Should be lower due to better memory management
3. **Exact metric values**: May differ slightly due to randomness
4. **Checkpoint format**: Different but compatible

## Reporting Issues

If validation fails:

1. Check the validation report for specific failures
2. Review the troubleshooting sections above
3. Verify all prerequisites are met
4. Check package versions match requirements
5. Report issues with:
   - Validation report
   - Environment details
   - Steps to reproduce

## References

- Requirements: 9.3, 9.4
- Original notebooks: `LifespanPredictClass.ipynb`
- New notebooks: `notebooks/`
- Validation script: `scripts/validate_notebooks.py`
