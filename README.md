# Urban Residential Classification using Explainable Machine Learning
## Automatic Detection of Residential Typologies and Enclosure Patterns: A Comparison Between Interpretability and Performance

## Project Overview

TO COMPLETE...

---

## Project Structure

```text
project_root/
│
├── data/               # Raw and processed datasets
├── images/             # Figures, plots and documentation assets
├── notebooks/          # Jupyter notebooks for exploration and analysis
├── output/             # Generated results, models and reports
├── src/                # Reusable Python source code
├── requirements.txt    # Python dependencies
├── pyproject.toml      # Project packaging configuration
└── .gitignore
```

---

## Environment Setup

### 1. Create a Virtual Environment

You may use either Conda or Python virtual environments.

#### Option A: Conda

```bash
conda create -n urban-analysis python=3.11.15
conda activate urban-analysis
```

#### Option B: venv

```bash
python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

---

### 2. Install Dependencies

Install all project dependencies using:

```bash
pip install -r requirements.txt
```

---

## Why `pip install -e .` Is Required

The project source code is located inside the `src/` directory.

To allow notebooks to import modules directly from `src/`, the project is installed in **editable mode** using:

```text
-e .
```

included in `requirements.txt`.

This enables imports such as:

```python
from src.EDA_utils import plot_fix_map
from src.nested_cv_fold import generate_folds
```

without manually modifying `PYTHONPATH` or adding custom path manipulation code inside notebooks.

Editable installation ensures that any modifications made to files inside `src/` are immediately available without reinstalling the package.

---

## Packaging Configuration

To support editable installation, the project includes the following configuration in `pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=61.0.0"]
build-backend = "setuptools.build_meta"

[project]
name = "tfm"
version = "0.1.0"
description = "Urban Analysis Using Supervised Machine Learning"
readme = "README.md"
requires-python = ">=3.10"

[tool.setuptools]
packages = ["src"]    
```

This configuration explicitly registers the `src` package so that Python can discover and import it correctly across different environments and operating systems.

---

## Development Environment

The project has been developed and tested using:

* Python 3.11.15
* Anaconda
* Visual Studio Code

However, any environment capable of installing the dependencies listed in `requirements.txt` should work correctly.

---

## Reproducibility

To ensure reproducibility:

1. Create a clean virtual environment.
2. Install dependencies using:

```bash
pip install -r requirements.txt
```

3. Verify that the editable package installation completes successfully.
4. Run notebooks from the repository root or through the configured Jupyter environment.

---

## License

This project was developed as part of the Master’s Degree in Data Science and Computer Engineering at the University of Granada.

It is released under the Apache License 2.0. See the `LICENSE` file for details.