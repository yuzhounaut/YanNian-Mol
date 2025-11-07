Data Module
===========

.. automodule:: lifespan_predictor.data
   :members:
   :undoc-members:
   :show-inheritance:

Preprocessing
-------------

.. automodule:: lifespan_predictor.data.preprocessing
   :members:
   :undoc-members:
   :show-inheritance:

Functions
~~~~~~~~~

.. autofunction:: lifespan_predictor.data.preprocessing.clean_smiles
.. autofunction:: lifespan_predictor.data.preprocessing.validate_smiles_list
.. autofunction:: lifespan_predictor.data.preprocessing.load_and_clean_csv

Featurizers
-----------

.. automodule:: lifespan_predictor.data.featurizers
   :members:
   :undoc-members:
   :show-inheritance:

CachedGraphFeaturizer
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: lifespan_predictor.data.featurizers.CachedGraphFeaturizer
   :members:
   :undoc-members:
   :show-inheritance:

Fingerprints
------------

.. automodule:: lifespan_predictor.data.fingerprints
   :members:
   :undoc-members:
   :show-inheritance:

FingerprintGenerator
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: lifespan_predictor.data.fingerprints.FingerprintGenerator
   :members:
   :undoc-members:
   :show-inheritance:

Dataset
-------

.. automodule:: lifespan_predictor.data.dataset
   :members:
   :undoc-members:
   :show-inheritance:

LifespanDataset
~~~~~~~~~~~~~~~

.. autoclass:: lifespan_predictor.data.dataset.LifespanDataset
   :members:
   :undoc-members:
   :show-inheritance:

GraphDataBuilder
~~~~~~~~~~~~~~~~

.. autoclass:: lifespan_predictor.data.dataset.GraphDataBuilder
   :members:
   :undoc-members:
   :show-inheritance:

Functions
~~~~~~~~~

.. autofunction:: lifespan_predictor.data.dataset.collate_lifespan_data
.. autofunction:: lifespan_predictor.data.dataset.create_dataloader
