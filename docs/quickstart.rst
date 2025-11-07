.. _quickstart:

Quick Start Guide
=================

This guide will help you get started with the Lifespan Predictor in just a few minutes.

Basic Workflow
--------------

The typical workflow consists of three steps:

1. **Data Preprocessing**: Load and featurize molecules
2. **Model Training**: Train the predictor model
3. **Inference**: Make predictions on new molecules

Step 1: Prepare Your Data
--------------------------

Create a CSV file with SMILES strings and labels:

.. code-block:: text

   SMILES,Life_extended
   CCO,0
   c1ccccc1,1
   CC(C)C,0

Step 2: Configure the Pipeline
-------------------------------

Create a configuration file ``config.yaml``:

.. code-block:: yaml

   data:
     train_csv: "train.csv"
     test_csv: "test.csv"
     smiles_column: "SMILES"
     label_column: "Life_extended"
   
   training:
     task: "classification"
     batch_size: 32
     max_epochs: 100
     learning_rate: 0.0001
   
   model:
     enable_gnn: true
     enable_fp_dnn: true
     enable_fp_cnn: true

Step 3: Run the Pipeline
-------------------------

Using Notebooks
~~~~~~~~~~~~~~~

The easiest way to get started is using the provided Jupyter notebooks:

1. ``notebooks/01_data_preprocessing.ipynb`` - Data preparation
2. ``notebooks/02_model_training.ipynb`` - Model training
3. ``notebooks/03_inference.ipynb`` - Making predictions

Using Python API
~~~~~~~~~~~~~~~~

Alternatively, use the Python API directly:

.. code-block:: python

   from lifespan_predictor.config import Config
   from lifespan_predictor.data import (
       load_and_clean_csv,
       CachedGraphFeaturizer,
       FingerprintGenerator,
       LifespanDataset,
       create_dataloader
   )
   from lifespan_predictor.models import LifespanPredictor
   from lifespan_predictor.training import Trainer, EarlyStopping
   from lifespan_predictor.training.metrics import MetricCollection, AUC
   import torch
   
   # Load configuration
   config = Config.from_yaml("config.yaml")
   
   # Load and clean data
   df = load_and_clean_csv(
       config.data.train_csv,
       smiles_column=config.data.smiles_column,
       label_column=config.data.label_column
   )
   
   # Featurize molecules
   featurizer = CachedGraphFeaturizer(
       cache_dir=config.data.graph_features_dir
   )
   adj, feat, sim, labels = featurizer.featurize(
       df[config.data.smiles_column].tolist(),
       labels=df[config.data.label_column].values
   )
   
   # Generate fingerprints
   fp_gen = FingerprintGenerator()
   hashed_fps, non_hashed_fps = fp_gen.generate_fingerprints(
       df[config.data.smiles_column].tolist(),
       cache_dir=config.data.fingerprints_dir
   )
   
   # Create dataset
   dataset = LifespanDataset(
       root="data/processed",
       smiles_list=df[config.data.smiles_column].tolist(),
       graph_features=(adj, feat, sim),
       fingerprints=(hashed_fps, non_hashed_fps),
       labels=labels
   )
   
   # Create data loader
   train_loader = create_dataloader(
       dataset,
       batch_size=config.training.batch_size
   )
   
   # Initialize model
   model = LifespanPredictor(config)
   
   # Setup training
   optimizer = torch.optim.Adam(
       model.parameters(),
       lr=config.training.learning_rate
   )
   criterion = torch.nn.BCEWithLogitsLoss()
   metrics = MetricCollection([AUC()])
   
   # Train model
   trainer = Trainer(
       model=model,
       config=config,
       train_loader=train_loader,
       val_loader=train_loader,  # Use separate validation set in practice
       optimizer=optimizer,
       criterion=criterion,
       metrics=metrics,
       callbacks=[EarlyStopping(patience=15)]
   )
   
   history = trainer.train()

Step 4: Make Predictions
-------------------------

.. code-block:: python

   # Load trained model
   checkpoint = torch.load("checkpoints/best_model.pt")
   model.load_state_dict(checkpoint['model_state_dict'])
   model.eval()
   
   # Prepare new molecules
   new_smiles = ["CCCC", "c1ccc(O)cc1"]
   # ... featurize and create dataset ...
   
   # Make predictions
   with torch.no_grad():
       for batch in test_loader:
           predictions = model(batch)
           probabilities = torch.sigmoid(predictions)
           print(probabilities)

Next Steps
----------

* Read the :ref:`configuration-guide` to customize the pipeline
* Explore the :ref:`api-reference` for detailed API documentation
* Check out example notebooks for more advanced usage
* See :ref:`troubleshooting` if you encounter issues
