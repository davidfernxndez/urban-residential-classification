# Explainable Artificial Intelligence for the Study of Urban Fragmentation

### A Comparative Analysis of the Performance and Interpretability of Supervised Learning Models for the Classification of Residential Enclosure Patterns

<p align="left">
  <img src="https://img.shields.io/badge/Language-Python-3776AB?logo=python&logoColor=white" alt="Language: Python">
  <img src="https://img.shields.io/badge/Model-XGBoost-EC6C00" alt="Model: XGBoost">
  <img src="https://img.shields.io/badge/Explainability-SHAP-purple" alt="Explainability: SHAP">
  <img src="https://img.shields.io/badge/Application-Streamlit-FF4B4B?logo=streamlit&logoColor=white" alt="Application: Streamlit">
</p>

<p align="left">
  <img src="images/app_demo/demo.gif" alt="Application Demo" width="900">
</p>

<p align="center">
  <a href="https://urban-tipology-xai.streamlit.app/">
    <img src="https://img.shields.io/badge/Try_the_Interactive_Application-24292F?style=for-the-badge&logo=streamlit&logoColor=white" alt="Interactive Application" width="400">
  </a>
</p>


## Project Overview

Urban typologies provide a framework for identifying and characterizing recurring patterns in the structure and functioning of cities. However, their identification and classification traditionally rely on manual analysis and domain expertise, limiting the scalability of these approaches to larger urban areas.

This project explores the use of **supervised machine learning** to support the systematic and scalable characterization of urban space, focusing  **urban fragmentation**, a sociological phenomenon that has attracted increasing attention due to its implications for patterns of spatial organization and social interaction in contemporary cities. 

The analysis is based on the «*Complejos residenciales del área metropolitana de Granada*» dataset, which provides information on one of the main manifestations of this phenomenon: **the degree of enclosure of residential compounds**. This urban typology characterizes how residential developments with shared communal spaces are physically delimited and relate to their surrounding urban environment, shaping interactions between residents and non-residents and, consequently, processes of urban integration and fragmentation.

The project follows a **comparative experimental approach**, evaluating supervised learning models from two complementary perspectives:
* **Predictive performance** — assessing the ability of different models to classify residential enclosure patterns.
* **Interpretability** — analysing how models arrive at their predictions and their ability to generate and transfer knowledge to the domain of study.

Model performance is evaluated using a common experimental protocol designed according to the characteristics of the dataset and the specific objectives of the study. While for interpretability, mechanisms are used according to the nature of the model:

| Model                               | Type        | Explainability Mechanism              |
| ----------------------------------- | ----------- | ------------------------------------- |
| *Multinomial Logistic Regression* | Transparent | Model coefficients                    |
| *Decision Tree*                  | Transparent | Decision rules and impurity reduction |
| *Support Vector Machine (SVM)*    | Black-box   | SHAP                                  |
| *Random Forest*                   | Black-box   | SHAP                                  |
| *XGBoost*                         | Black-box   | SHAP                                  |


The final outcome combines the predictive and interpretability analyses to select the most suitable model and explanation mechanism for the problem. These components are integrated into an **interactive Streamlit application**, providing a practical interface for domain experts to explore the dataset, classify new residential developments, and analyse the factors underlying individual predictions.

> **[Dataset](https://hdl.handle.net/10481/105464):** Baldán Lozano, H., & Susino Arbucias, J. (2024). *Base de datos de complejos residenciales en el área metropolitana de Granada*. Universidad de Granada.

---

## Project Structure

The repository is organized into separate directories for **data, experimentation, results, source code, and the interactive application**. The workflow is primarily developed and documented through a sequence of *Jupyter notebooks*, while reusable functionality is centralized in the `src/` package.

```text
urban-typology-xai/
│
├── data/
│   ├── raw/                    # Original dataset in Excel format
│   ├── processed/              # Processed datasets used for modelling
│   └── folds/                  # Pre-generated CV folds for reproducibility
│
├── notebooks/                  # Project development and experimental workflow
│   ├── 1_EDA.ipynb
│   ├── 2_Folds_generation.ipynb
│   ├── 3.1_Performance_methodology.ipynb
│   ├── 3.2_Performance_results.ipynb
│   ├── 4.1_Global_interp_methodology.ipynb
│   ├── 4.2_Global_interp_analysis.ipynb
│   ├── 5_Discussion.ipynb
│   ├── 6_Local_interp_analysis.ipynb
│   └── 7_App_demo.ipynb
│
├── output/
│   ├── 5x5_NCV/                # Main Nested Cross-Validation results
│   ├── 5x3_NCV/                # Alternative 5×3 Nested CV results
│   ├── 5x10_NCV/               # Alternative 5×10 Nested CV results
│   ├── models/                 # Trained models for interpretability analysis
│   ├── SHAP_global/            # Global SHAP values for black-box models
│   └── SHAP_local/             # Local SHAP values for selected models
│
├── src/                        # Reusable source code
│   ├── config.py
│   ├── EDA_utils.py
│   ├── nested_cv_utils.py
│   ├── performance_utils.py
│   ├── balanced_xgb.py
│   ├── production_training.py
│   └── XAI_utils.py
│
├── app/                        # Interactive Streamlit application
│   ├── .streamlit/             # Streamlit configuration
│   │
│   ├── app_src/
│   │   ├── ui/                 # Application views
│   │   ├── utils/              # Application utility functions
│   │   └── app_config.py       # Application configuration
│   │
│   ├── assets/                 # Custom application styles
│   │
│   └── app.py                  # Application entry point
│
├── requirements-full.txt       # Complete development and experimentation dependencies
├── requirements.txt            # Deployment dependencies used by Streamlit Cloud
├── pyproject.toml
└── README.md
```

### Data

The `data/` directory contains the different stages of data preparation and the pre-generated partitions required to reproduce the experiments:

* **`raw/`** — Original dataset in its source Excel format.
* **`processed/`** — Processed datasets prepared for modelling and experimentation.
* **`folds/`** — Pre-generated cross-validation partitions stored to ensure consistent and reproducible experiments.

### Notebooks

The `notebooks/` directory contains the complete experimental workflow, progressing from data exploration to model evaluation, interpretability analysis, discussion, and application deployment.

| Notebook                              | Description                                                                                                                            |
| ------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| `1_EDA.ipynb`                         | Introduction to the database through Exploratory Data Analysis (EDA).                                                         |
| `2_Folds_generation.ipynb`            | Generation of the Nested Cross-Validation folds and the experimental partitions required for reproducibility.                          |
| `3.1_Performance_methodology.ipynb`   | Definition and execution of the common methodology used to evaluate all classification models under homogeneous conditions.            |
| `3.2_Performance_results.ipynb`       | Analysis of the predictive performance results obtained from the Nested Cross-Validation experiments.                                  |
| `4.1_Global_interp_methodology.ipynb` | Definition of the methodology used for global interpretability analysis of transparent and black-box models.                           |
| `4.2_Global_interp_analysis.ipynb`    | Analysis of the global interpretability results and their consistency with domain knowledge.                                           |
| `5_Discussion.ipynb`                  | Joint discussion of predictive performance and interpretability to identify the most suitable model for deployment.                    |
| `6_Local_interp_analysis.ipynb`       | Local interpretability analysis of the selected model and development of the SHAP-based explanation mechanism used by the application. |
| `7_App_demo.ipynb`                    | Demonstration of the interactive platform integrating the selected model and its explainability mechanism.                             |

### Source Code

The `src/` directory contains the reusable Python functionality used throughout the notebooks:

| File                     | Description                                                                                    |
| ------------------------ | ---------------------------------------------------------------------------------------------- |
| `config.py`              | Global configuration, paths, random seed, and experiment settings.                             |
| `EDA_utils.py`           | Utility functions for exploratory data analysis.                                               |
| `nested_cv_utils.py`     | Functions for generating and managing Nested Cross-Validation folds.                           |
| `performance_utils.py`   | Functions for model training, evaluation, and performance analysis.                            |
| `balanced_xgb.py`        | Custom wrapper implementing cost-sensitive XGBoost for multiclass classification.              |
| `production_training.py` | Functions for training and optimizing the selected model using the complete available dataset. |
| `XAI_utils.py`           | Functions supporting global and local model interpretability analyses.                         |

### Output

The `output/` directory stores the results generated throughout the experimental workflow.

The main performance results are stored in **`5x5_NCV/`**, with a separate directory for each evaluated model:

```text
5x5_NCV/
├── Decision_Tree/
├── Logistic_regression/
├── SVM/
├── Random_Forest/
└── XGBoost/
```

Each model directory contains its global metrics, class-specific out-of-fold metrics, and out-of-fold confusion matrix.

The repository also includes alternative Nested Cross-Validation configurations (`5x3_NCV/` and `5x10_NCV/`) that were evaluated but were not used for the final experimental analysis.

The remaining output directories contain the trained models and the SHAP values generated for global and local interpretability analyses.

### Application

The `app/` directory contains the **interactive Streamlit application** developed as the final knowledge-transfer component of the project. It integrates the selected classification model and its explainability mechanism into an accessible environment for exploring the dataset, simulating new residential developments, and analysing individual predictions.

The application is structured as a **self-contained component** with its own configuration, user interface, utility functions, and visual assets:

| Component               | Description                                                                                |
| ----------------------- | ------------------------------------------------------------------------------------------ |
| `.streamlit/`           | Streamlit-specific configuration, including application settings defined in `config.toml`. |
| `app_src/ui/`           | User interface views implementing the main application screens.                            |
| `app_src/utils/`        | Utility functions for loading and caching the resources required by the application.       |
| `app_src/app_config.py` | Application-specific configuration.                                                        |
| `assets/`               | Static assets used by the application, including custom CSS styles.                        |
| `app.py`                | Main application entry point.                                                              |

The three main views provide complementary functionality:

* **Context view** — presents the context and background information required to understand the analysis.
* **Explorative view** — enables exploration of the available residential compounds and their characteristics.
* **Simulator view** — allows users to configure a residential development, obtain a prediction from the selected model, and analyse the corresponding local explanation.

## Installation & Usage

### 1. Clone the Repository

Clone the repository and move into the project directory:

```bash
git clone https://github.com/davidfernxndez/urban-typology-xai.git
cd urban-typology-xai
```

### 2. Create the Environment

The project was developed and tested using **Python 3.11.15** within an **Anaconda** environment.

Create and activate a dedicated environment:

```bash
conda create -n urban-typology-xai python=3.11.15
conda activate urban-typology-xai
```

Install all required dependencies using the provided `requirements-full.txt` file:

```bash
pip install -r requirements-full.txt
```

### 3. Install the Project in Editable Mode

The reusable source code is located in the `src/` directory. The repository includes a `pyproject.toml` file that defines the project as a Python package and allows it to be installed in **editable mode**.

The following entry in `requirements-full.txt`:

```text
-e .
```

installs the local project in editable mode.

This allows the notebooks to import modules from `src/` directly:

```python
from src.EDA_utils import plot_univariate_distribution
from src.nested_cv_utils import generate_folds
```

without manually modifying `PYTHONPATH` or adding custom path manipulation code.

The editable installation also means that changes made to files inside `src/` are immediately available to the notebooks without requiring the package to be reinstalled.

### 4. Run the Application Locally

The interactive application can be launched from the **root directory of the repository** using:

```bash
streamlit run app/app.py
```

Streamlit will start a local server and provide the URL required to access the application in a web browser.

### Online Demo

The application is also available as a public deployment through **Streamlit Community Cloud**: [urban-tipology-xai.streamlit.app](https://urban-tipology-xai.streamlit.app/)

## License

This project is licensed under the **Apache License 2.0**. See the [`LICENSE`](LICENSE) file for the full license text.

This project was developed as part of the **Master's Degree in Data Science and Computer Engineering at the University of Granada**.