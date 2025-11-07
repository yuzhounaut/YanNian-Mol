.. _troubleshooting:

Troubleshooting
===============

This page provides solutions to common issues you may encounter.

Installation Issues
-------------------

CUDA Out of Memory
~~~~~~~~~~~~~~~~~~

**Problem**: ``RuntimeError: CUDA out of memory``

**Solutions**:

1. Reduce batch size:

.. code-block:: yaml

   training:
     batch_size: 16  # Reduce from 32

2. Enable mixed precision training:

.. code-block:: yaml

   training:
     use_mixed_precision: true

3. Reduce model size:

.. code-block:: yaml

   model:
     gnn_graph_embed_dim: 64  # Reduce from 128
     fp_dnn_layers: [128, 64]  # Smaller layers

4. Disable branches:

.. code-block:: yaml

   model:
     enable_fp_cnn: false  # Disable CNN branch

RDKit Installation Failed
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: ``pip install rdkit`` fails

**Solution**: Use conda instead:

.. code-block:: bash

   conda install -c conda-forge rdkit

Or use the PyPI version:

.. code-block:: bash

   pip install rdkit-pypi

DGL Installation Issues
~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: DGL not compatible with CUDA version

**Solution**: Install DGL with specific CUDA version:

.. code-block:: bash

   # For CUDA 11.8
   pip install dgl -f https://data.dgl.ai/wheels/cu118/repo.html
   
   # For CUDA 12.1
   pip install dgl -f https://data.dgl.ai/wheels/cu121/repo.html
   
   # For CPU only
   pip install dgl

Data Issues
-----------

Invalid SMILES Strings
~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Many molecules fail to parse

**Solutions**:

1. Enable SMILES cleaning:

.. code-block:: python

   df = load_and_clean_csv(
       "data.csv",
       clean=True,
       drop_invalid=True
   )

2. Check input data quality:

.. code-block:: python

   from lifespan_predictor.data import clean_smiles
   
   # Test individual SMILES
   for smiles in df["SMILES"]:
       cleaned = clean_smiles(smiles)
       if cleaned is None:
           print(f"Invalid: {smiles}")

3. Review preprocessing logs:

.. code-block:: python

   import logging
   logging.basicConfig(level=logging.DEBUG)

Slow Featurization
~~~~~~~~~~~~~~~~~~

**Problem**: Molecular featurization takes too long

**Solutions**:

1. Enable caching:

.. code-block:: yaml

   featurization:
     use_cache: true

2. Increase parallel jobs:

.. code-block:: yaml

   featurization:
     n_jobs: -1  # Use all CPU cores

3. Reduce max_atoms:

.. code-block:: yaml

   featurization:
     max_atoms: 150  # Skip large molecules

Cache Not Working
~~~~~~~~~~~~~~~~~

**Problem**: Features are recomputed every time

**Solutions**:

1. Check cache directory exists and is writable:

.. code-block:: python

   import os
   cache_dir = "cache/features"
   os.makedirs(cache_dir, exist_ok=True)

2. Verify cache files:

.. code-block:: bash

   ls -lh cache/features/

3. Force cache rebuild if corrupted:

.. code-block:: python

   adj, feat, sim, labels = featurizer.featurize(
       smiles_list,
       labels=labels,
       force_recompute=True
   )

Training Issues
---------------

Model Not Learning
~~~~~~~~~~~~~~~~~~

**Problem**: Training loss not decreasing

**Solutions**:

1. Check learning rate:

.. code-block:: yaml

   training:
     learning_rate: 0.001  # Increase if too slow

2. Verify data labels:

.. code-block:: python

   print(df["label"].value_counts())
   print(df["label"].describe())

3. Check for data leakage:

.. code-block:: python

   # Ensure train/val split is correct
   from sklearn.model_selection import train_test_split
   
   train_idx, val_idx = train_test_split(
       range(len(dataset)),
       test_size=0.3,
       stratify=labels,
       random_state=42
   )

4. Try different model configurations:

.. code-block:: yaml

   model:
     gnn_num_layers: 3  # Increase capacity
     gnn_dropout: 0.3   # Reduce dropout

Training Unstable
~~~~~~~~~~~~~~~~~

**Problem**: Loss oscillates or explodes

**Solutions**:

1. Enable gradient clipping:

.. code-block:: yaml

   training:
     gradient_clip: 1.0

2. Reduce learning rate:

.. code-block:: yaml

   training:
     learning_rate: 0.0001

3. Check for NaN values:

.. code-block:: python

   # Add to training loop
   if torch.isnan(loss):
       print("NaN loss detected!")
       break

4. Use learning rate scheduler:

.. code-block:: python

   from lifespan_predictor.training.callbacks import LearningRateScheduler
   import torch.optim.lr_scheduler as lr_scheduler
   
   callbacks = [
       LearningRateScheduler(
           scheduler_class=lr_scheduler.ReduceLROnPlateau,
           scheduler_params={"factor": 0.5, "patience": 10}
       )
   ]

Overfitting
~~~~~~~~~~~

**Problem**: Validation loss increases while training loss decreases

**Solutions**:

1. Increase dropout:

.. code-block:: yaml

   model:
     gnn_dropout: 0.7
     fp_dropout: 0.7

2. Add weight decay:

.. code-block:: yaml

   training:
     weight_decay: 0.001

3. Use early stopping:

.. code-block:: python

   from lifespan_predictor.training.callbacks import EarlyStopping
   
   callbacks = [
       EarlyStopping(patience=10, monitor="val_loss")
   ]

4. Reduce model complexity:

.. code-block:: yaml

   model:
     gnn_num_layers: 2  # Reduce from 3
     fp_dnn_layers: [128]  # Smaller network

Runtime Issues
--------------

Import Errors
~~~~~~~~~~~~~

**Problem**: ``ModuleNotFoundError``

**Solutions**:

1. Reinstall dependencies:

.. code-block:: bash

   pip install -r requirements.txt

2. Check Python version:

.. code-block:: bash

   python --version  # Should be >= 3.9

3. Verify virtual environment:

.. code-block:: bash

   which python  # Should point to venv

Configuration Errors
~~~~~~~~~~~~~~~~~~~~

**Problem**: ``ValidationError`` when loading config

**Solutions**:

1. Check YAML syntax:

.. code-block:: python

   import yaml
   with open("config.yaml") as f:
       config_dict = yaml.safe_load(f)
       print(config_dict)

2. Verify parameter types:

.. code-block:: yaml

   training:
     batch_size: 32  # Must be int, not "32"
     learning_rate: 0.0001  # Must be float

3. Check required parameters:

.. code-block:: python

   from lifespan_predictor.config import Config
   
   try:
       config = Config.from_yaml("config.yaml")
   except Exception as e:
       print(f"Config error: {e}")

Memory Leaks
~~~~~~~~~~~~

**Problem**: Memory usage keeps increasing

**Solutions**:

1. Clear GPU cache:

.. code-block:: python

   import torch
   
   # After each epoch
   torch.cuda.empty_cache()

2. Delete unused variables:

.. code-block:: python

   # After training
   del model, optimizer, trainer
   torch.cuda.empty_cache()

3. Use gradient accumulation:

.. code-block:: python

   # In training loop
   accumulation_steps = 4
   
   for i, batch in enumerate(train_loader):
       loss = loss / accumulation_steps
       loss.backward()
       
       if (i + 1) % accumulation_steps == 0:
           optimizer.step()
           optimizer.zero_grad()

Performance Issues
------------------

Slow Training
~~~~~~~~~~~~~

**Problem**: Training is too slow

**Solutions**:

1. Use GPU:

.. code-block:: yaml

   device:
     use_cuda: true

2. Enable mixed precision:

.. code-block:: yaml

   training:
     use_mixed_precision: true

3. Increase batch size:

.. code-block:: yaml

   training:
     batch_size: 64  # If GPU memory allows

4. Use DataLoader workers:

.. code-block:: python

   train_loader = create_dataloader(
       dataset,
       batch_size=32,
       num_workers=4  # Parallel data loading
   )

Slow Inference
~~~~~~~~~~~~~~

**Problem**: Predictions are slow

**Solutions**:

1. Use batch inference:

.. code-block:: python

   # Instead of one-by-one
   predictions = []
   for batch in test_loader:
       pred = model(batch)
       predictions.append(pred)

2. Disable gradient computation:

.. code-block:: python

   with torch.no_grad():
       predictions = model(batch)

3. Use model.eval():

.. code-block:: python

   model.eval()  # Disables dropout and batch norm updates

Getting Help
------------

If you still have issues:

1. Check the documentation thoroughly
2. Review example notebooks
3. Search existing GitHub issues
4. Open a new issue with:
   
   - Error message and full traceback
   - Configuration file
   - Python and package versions (``pip list``)
   - Steps to reproduce
   - Minimal code example

Debugging Tips
--------------

Enable Debug Logging
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import logging
   logging.basicConfig(level=logging.DEBUG)

Check Tensor Shapes
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Add to model forward pass
   print(f"Input shape: {batch.x.shape}")
   print(f"Output shape: {output.shape}")

Validate Data
~~~~~~~~~~~~~

.. code-block:: python

   # Check for NaN or inf
   assert not torch.isnan(batch.x).any()
   assert not torch.isinf(batch.x).any()

Profile Code
~~~~~~~~~~~~

.. code-block:: python

   import cProfile
   import pstats
   
   profiler = cProfile.Profile()
   profiler.enable()
   
   # Your code here
   trainer.train()
   
   profiler.disable()
   stats = pstats.Stats(profiler)
   stats.sort_stats('cumulative')
   stats.print_stats(20)
