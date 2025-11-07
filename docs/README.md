# Documentation

This directory contains the Sphinx documentation for the Lifespan Predictor project.

## Building the Documentation

### Prerequisites

Install Sphinx and required extensions:

```bash
pip install -r docs/requirements.txt
```

### Build HTML Documentation

```bash
cd docs
make html
```

The generated HTML documentation will be in `docs/_build/html/`.

### View Documentation

Open `docs/_build/html/index.html` in your browser, or serve it locally:

```bash
cd docs
make html
make serve
```

Then visit http://localhost:8000 in your browser.

### Clean Build

To remove all generated files:

```bash
cd docs
make clean
```

### Fast Build

For faster builds during development (uses parallel processing):

```bash
cd docs
make html-fast
```

## Documentation Structure

```
docs/
├── conf.py                 # Sphinx configuration
├── index.rst              # Main documentation page
├── installation.rst       # Installation guide
├── quickstart.rst         # Quick start guide
├── configuration.md       # Configuration guide (Markdown)
├── examples.rst           # Usage examples
├── troubleshooting.rst    # Troubleshooting guide
├── api/                   # API reference
│   ├── index.rst
│   ├── config.rst
│   ├── data.rst
│   ├── models.rst
│   ├── training.rst
│   └── utils.rst
├── Makefile              # Build commands
└── requirements.txt      # Sphinx dependencies
```

## Writing Documentation

### reStructuredText (RST) Format

Most documentation files use reStructuredText format. Key syntax:

```rst
Section Header
==============

Subsection
----------

**Bold text**
*Italic text*
``Code``

.. code-block:: python

   # Python code block
   import lifespan_predictor

.. note::
   This is a note.

.. warning::
   This is a warning.
```

### Docstrings

All Python code should have docstrings in NumPy or Google style:

```python
def my_function(param1: int, param2: str) -> bool:
    """
    Brief description.
    
    Longer description with more details.
    
    Parameters
    ----------
    param1 : int
        Description of param1
    param2 : str
        Description of param2
    
    Returns
    -------
    bool
        Description of return value
    
    Examples
    --------
    >>> my_function(42, "hello")
    True
    """
    pass
```

### Adding New Pages

1. Create a new `.rst` file in `docs/`
2. Add it to the `toctree` in `index.rst`:

```rst
.. toctree::
   :maxdepth: 2
   
   installation
   quickstart
   your_new_page
```

### Adding API Documentation

API documentation is automatically generated from docstrings. To add a new module:

1. Create a new `.rst` file in `docs/api/`
2. Add autodoc directives:

```rst
My Module
=========

.. automodule:: lifespan_predictor.my_module
   :members:
   :undoc-members:
   :show-inheritance:
```

3. Add it to `docs/api/index.rst`

## Continuous Integration

The documentation can be built automatically in CI/CD:

```yaml
# .github/workflows/docs.yml
name: Documentation

on: [push, pull_request]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.9
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r docs/requirements.txt
      - name: Build documentation
        run: |
          cd docs
          make html
```

## Hosting Documentation

### GitHub Pages

1. Build documentation:
   ```bash
   cd docs
   make html
   ```

2. Copy to gh-pages branch:
   ```bash
   git checkout gh-pages
   cp -r docs/_build/html/* .
   git add .
   git commit -m "Update documentation"
   git push
   ```

### Read the Docs

1. Create account at https://readthedocs.org
2. Import your repository
3. Configure build settings (uses `docs/conf.py` automatically)

## Troubleshooting

### Module Import Errors

If Sphinx can't import modules:

```python
# In docs/conf.py
import sys
import os
sys.path.insert(0, os.path.abspath('..'))
```

### Missing Dependencies

Install all project dependencies:

```bash
pip install -r requirements.txt
pip install -r docs/requirements.txt
```

### Build Warnings

Fix warnings to ensure documentation quality:

```bash
cd docs
make html 2>&1 | grep WARNING
```

Common warnings:
- Missing docstrings
- Broken cross-references
- Invalid RST syntax
