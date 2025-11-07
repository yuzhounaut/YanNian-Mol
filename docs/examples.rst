Examples
========

This page provides practical examples for common use cases.

Basic Usage
-----------

Loading and Preprocessing Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from lifespan_predictor.data import load_and_clean_csv
   
   # Load CSV with automatic SMILES cleaning
   df = load_and_clean_csv(
       "data.csv",
       smiles_column="SMILES",
       label_column="Activity",
       clean=True,
       drop_invalid=True
   )
   
   print(f"Loaded {len(df)} valid molecules")

Molecular Featurization
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from lifespan_predictor.data import CachedGraphFeaturizer
   import numpy as np
   
   # Initialize featurizer
   featurizer = CachedGraphFeaturizer(
       cache_dir="cache/features",
       max_atoms=200,
       n_jobs=-1
   )
   
   # Featurize molecules
   smiles_list = df["SMILES"].tolist()
   labels = df["Activity"].values
   
   adj, feat, sim, labels = featurizer.featurize(
       smiles_list,
       labels=labels,
       force_recompute=False
   )
   
   print(f"Adjacency shape: {adj.shape}")
   print(f"Features shape: {feat.shape}")

Fingerprint Generation
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from lifespan_predictor.data import FingerprintGenerator
   
   # Initialize generator
   fp_gen = FingerprintGenerator(
       morgan_radius=2,
       morgan_nbits=2048,
       rdkit_fp_nbits=2048,
       n_jobs=-1
   )
   
   # Generate fingerprints
   hashed_fps, non_hashed_fps = fp_gen.generate_fingerprints(
       smiles_list,
       cache_dir="cache/fingerprints"
   )
   
   print(f"Hashed FPs shape: {hashed_fps.shape}")
   print(f"Non-hashed FPs shape: {non_hashed_fps.shape}")

Advanced Usage
--------------

Custom Configuration
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from lifespan_predictor.config import Config
   
   # Create custom configuration
   config = Config.from_dict({
       "data": {
           "train_csv": "my_data.csv",
           "smiles_column": "mol",
           "label_column": "target"
       },
       "model": {
           "enable_gnn": True,
           "enable_fp_dnn": True,
           "enable_fp_cnn": False,
           "gnn_num_layers": 3,
           "gnn_graph_embed_dim": 256
       },
       "training": {
           "task": "regression",
           "batch_size": 64,
           "learning_rate": 0.001,
           "main_metric": "RMSE"
       }
   })
   
   # Save configuration
   config.save("my_config.yaml")

Custom Training Loop
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from lifespan_predictor.training import Trainer
   from lifespan_predictor.training.callbacks import (
       EarlyStopping,
       ModelCheckpoint,
       LearningRateScheduler
   )
   from lifespan_predictor.training.metrics import MetricCollection, AUC, Accuracy
   import torch
   
   # Setup callbacks
   callbacks = [
       EarlyStopping(
           patience=20,
           monitor="val_AUC",
           mode="max"
       ),
       ModelCheckpoint(
           save_dir="checkpoints",
           monitor="val_AUC",
           mode="max",
           save_best_only=True
       ),
       LearningRateScheduler(
           scheduler_class=torch.optim.lr_scheduler.ReduceLROnPlateau,
           scheduler_params={"factor": 0.5, "patience": 10}
       )
   ]
   
   # Setup metrics
   metrics = MetricCollection([
       AUC(),
       Accuracy()
   ])
   
   # Create trainer
   trainer = Trainer(
       model=model,
       config=config,
       train_loader=train_loader,
       val_loader=val_loader,
       optimizer=optimizer,
       criterion=criterion,
       metrics=metrics,
       callbacks=callbacks
   )
   
   # Train
   history = trainer.train()

Custom Callback
~~~~~~~~~~~~~~~

.. code-block:: python

   from lifespan_predictor.training.callbacks import Callback
   
   class LogToFile(Callback):
       def __init__(self, filepath):
           self.filepath = filepath
           self.file = None
       
       def on_train_begin(self, logs=None):
           self.file = open(self.filepath, 'w')
           self.file.write("epoch,train_loss,val_loss\\n")
       
       def on_epoch_end(self, epoch, logs):
           self.file.write(
               f"{epoch},{logs['train_loss']},{logs['val_loss']}\\n"
           )
           self.file.flush()
       
       def on_train_end(self, logs=None):
           if self.file:
               self.file.close()
   
   # Use custom callback
   trainer = Trainer(
       model=model,
       config=config,
       train_loader=train_loader,
       val_loader=val_loader,
       optimizer=optimizer,
       criterion=criterion,
       metrics=metrics,
       callbacks=[LogToFile("training_log.csv")]
   )

Inference
~~~~~~~~~

.. code-block:: python

   import torch
   from lifespan_predictor.models import LifespanPredictor
   from lifespan_predictor.data import create_dataloader
   
   # Load trained model
   model = LifespanPredictor(config)
   checkpoint = torch.load("checkpoints/best_model.pt")
   model.load_state_dict(checkpoint['model_state_dict'])
   model.eval()
   
   # Prepare test data
   test_loader = create_dataloader(
       test_dataset,
       batch_size=32,
       shuffle=False
   )
   
   # Make predictions
   all_predictions = []
   all_labels = []
   
   with torch.no_grad():
       for batch in test_loader:
           batch = batch.to(config.get_device())
           predictions = model(batch)
           
           # For classification, get probabilities
           if config.training.task == "classification":
               predictions = torch.sigmoid(predictions)
           
           all_predictions.append(predictions.cpu().numpy())
           if hasattr(batch, 'y'):
               all_labels.append(batch.y.cpu().numpy())
   
   # Concatenate results
   predictions = np.concatenate(all_predictions)
   if all_labels:
       labels = np.concatenate(all_labels)

Visualization
~~~~~~~~~~~~~

.. code-block:: python

   from lifespan_predictor.utils.visualization import (
       plot_training_curves,
       plot_predictions,
       plot_roc_curve
   )
   
   # Plot training curves
   plot_training_curves(
       history,
       save_path="plots/training_curves.png"
   )
   
   # Plot predictions vs true values
   plot_predictions(
       y_true=labels,
       y_pred=predictions,
       save_path="plots/predictions.png",
       task="classification"
   )
   
   # Plot ROC curve (for classification)
   plot_roc_curve(
       y_true=labels,
       y_pred=predictions,
       save_path="plots/roc_curve.png"
   )

Batch Processing
~~~~~~~~~~~~~~~~

.. code-block:: python

   from lifespan_predictor.data import CachedGraphFeaturizer
   from tqdm import tqdm
   
   # Process large dataset in chunks
   chunk_size = 1000
   all_features = []
   
   for i in tqdm(range(0, len(smiles_list), chunk_size)):
       chunk = smiles_list[i:i+chunk_size]
       
       adj, feat, sim, _ = featurizer.featurize(
           chunk,
           force_recompute=False
       )
       
       all_features.append((adj, feat, sim))
   
   # Concatenate all chunks
   all_adj = np.concatenate([f[0] for f in all_features])
   all_feat = np.concatenate([f[1] for f in all_features])
   all_sim = np.concatenate([f[2] for f in all_features])

Multi-GPU Training
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch.nn as nn
   
   # Wrap model for multi-GPU
   if torch.cuda.device_count() > 1:
       print(f"Using {torch.cuda.device_count()} GPUs")
       model = nn.DataParallel(model)
   
   model = model.to(config.get_device())
   
   # Train as usual
   trainer = Trainer(
       model=model,
       config=config,
       train_loader=train_loader,
       val_loader=val_loader,
       optimizer=optimizer,
       criterion=criterion,
       metrics=metrics
   )

Hyperparameter Tuning
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from itertools import product
   
   # Define hyperparameter grid
   learning_rates = [1e-4, 5e-4, 1e-3]
   batch_sizes = [16, 32, 64]
   dropout_rates = [0.3, 0.5, 0.7]
   
   best_score = 0
   best_params = None
   
   # Grid search
   for lr, bs, dropout in product(learning_rates, batch_sizes, dropout_rates):
       print(f"Testing lr={lr}, batch_size={bs}, dropout={dropout}")
       
       # Update config
       config.training.learning_rate = lr
       config.training.batch_size = bs
       config.model.gnn_dropout = dropout
       
       # Create new model and trainer
       model = LifespanPredictor(config)
       optimizer = torch.optim.Adam(model.parameters(), lr=lr)
       
       trainer = Trainer(
           model=model,
           config=config,
           train_loader=train_loader,
           val_loader=val_loader,
           optimizer=optimizer,
           criterion=criterion,
           metrics=metrics
       )
       
       # Train
       history = trainer.train()
       
       # Get best validation score
       val_score = max(history['val_AUC'])
       
       if val_score > best_score:
           best_score = val_score
           best_params = (lr, bs, dropout)
   
   print(f"Best params: lr={best_params[0]}, bs={best_params[1]}, dropout={best_params[2]}")
   print(f"Best score: {best_score}")
