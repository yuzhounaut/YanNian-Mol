.. _installation:

Installation
============

Prerequisites
-------------

* Python >= 3.9
* CUDA-capable GPU (optional, but recommended for training)

Install from Source
-------------------

1. Clone the repository:

.. code-block:: bash

   git clone <repository-url>
   cd lifespan_predictor

2. Create a virtual environment (recommended):

.. code-block:: bash

   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install dependencies:

.. code-block:: bash

   pip install -r requirements.txt

4. Install the package in development mode:

.. code-block:: bash

   pip install -e .

Dependencies
------------

Core Dependencies
~~~~~~~~~~~~~~~~~

* PyTorch >= 2.0
* PyTorch Geometric >= 2.3
* DGL >= 1.0
* DGL-LifeSci >= 0.3
* RDKit >= 2022.09
* DeepChem >= 2.7
* NumPy >= 1.23
* Pandas >= 1.5
* scikit-learn >= 1.2

Additional Dependencies
~~~~~~~~~~~~~~~~~~~~~~~

* Pydantic >= 2.0 (configuration validation)
* PyYAML >= 6.0 (configuration files)
* tqdm >= 4.65 (progress bars)
* tensorboard >= 2.12 (logging)
* matplotlib >= 3.7 (visualization)
* seaborn >= 0.12 (visualization)
* joblib >= 1.2 (parallel processing)

Development Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~

* pytest >= 7.3
* pytest-cov >= 4.0
* black >= 23.0 (code formatting)
* flake8 >= 6.0 (linting)
* mypy >= 1.0 (type checking)
* sphinx >= 6.0 (documentation)
* sphinx-rtd-theme (documentation theme)

Verifying Installation
----------------------

To verify the installation, run:

.. code-block:: python

   from lifespan_predictor.config import Config
   from lifespan_predictor.models import LifespanPredictor
   
   config = Config()
   model = LifespanPredictor(config)
   print("Installation successful!")

Troubleshooting
---------------

CUDA Issues
~~~~~~~~~~~

If you encounter CUDA-related errors:

1. Verify CUDA is installed: ``nvidia-smi``
2. Install PyTorch with CUDA support: Visit `pytorch.org <https://pytorch.org>`_
3. Check CUDA version compatibility with PyTorch

RDKit Installation
~~~~~~~~~~~~~~~~~~

If RDKit installation fails:

.. code-block:: bash

   # Using conda (recommended)
   conda install -c conda-forge rdkit
   
   # Or using pip
   pip install rdkit-pypi

DGL Installation
~~~~~~~~~~~~~~~~

For GPU support with DGL:

.. code-block:: bash

   # For CUDA 11.8
   pip install dgl -f https://data.dgl.ai/wheels/cu118/repo.html
   
   # For CPU only
   pip install dgl
