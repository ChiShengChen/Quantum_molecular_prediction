# Quantum and Classical Regression for Molecular Properties

This project trains and evaluates various quantum and classical machine learning models for predicting molecular properties based on input features from an Excel dataset.

## Description

The primary script, `quantum_regression.py`, implements several regression models, including:

*   **Dataset:**
    * Dataset is from https://www.pnas.org/doi/10.1073/pnas.2214357120   

*   **Quantum Models:**
    *   `QuantumRegressionNet`: A Quantum Neural Network (QNN) featuring an `EnhancedQuantumLayer` with data re-uploading.
    *   `QASARegression`: A model inspired by the Quantum Attention Self-Attention (QASA) architecture, adapted for tabular regression.
*   **Classical Deep Learning Models:**
    *   `MLPBaseline`: Multi-Layer Perceptron.
    *   `TransformerBaseline`: Transformer Encoder based model.
    *   `CNNBaseline`: 1D Convolutional Neural Network.
    *   `ResNetBaseline`: Residual Network.
    *   `LSTMBaseline`: Long Short-Term Memory network.
*   **Classical Machine Learning Models:**
    *   `SVR`: Support Vector Regressor (via scikit-learn).
    *   `KNeighborsRegressor`: K-Nearest Neighbors Regressor (via scikit-learn).
    *   `XGBoost`: XGBoost Regressor.
    *   `LightGBM`: LightGBM Regressor.

The script performs the following steps:
1.  Loads data from a specified Excel sheet.
2.  Preprocesses the data (handles missing values, scales features).
3.  Splits data into training, validation (for DL models), and test sets.
4.  Optionally trains selected models using command-line flags.
5.  Evaluates each trained model on the test set, calculating metrics (MSE, RMSE, MAE, R2) for each target property.
6.  Saves results, including metrics, predictions, and scatter plots, into separate directories for each model.
7.  Generates summary plots comparing the performance of all successfully trained models across key metrics.

## Installation

1.  **Clone the repository (if applicable):**
    ```bash
    git clone https://github.com/ChiShengChen/Quantum_molecular_prediction.git
    cd Quantum_molecular_prediction
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: Some dependencies like `pennylane-lightning` might require specific system libraries or compilers. Refer to their official documentation if you encounter installation issues.*

## Usage

Run the main script from the command line:

```bash
python quantum_regression.py [OPTIONS]
```

**Key Options:**

*   `--excel-path`: Path to the input Excel file (default: `pnas.2214357120.sd01.xlsx`).
*   `--sheet-name`: Name of the sheet containing the data (default: `Dataset`).
*   `--save-dir-prefix`: Prefix for the directories where results will be saved (default: `results`). Model-specific results will be in `<prefix>_<ModelName>`.
*   `--epochs`: Number of training epochs for deep learning models (default: 50).
*   `--batch-size`: Batch size for deep learning models (default: 32).
*   `--lr`: Learning rate for deep learning models (default: 0.001).
*   `--debug`: Run in debug mode with reduced parameters for faster testing.
*   **Model Flags:** Use flags like `--run-qnn`, `--no-run-mlp`, `--run-transformer`, etc., to select which models to train and evaluate. By default, most models are set to run. Use `--no-<flag>` to disable a specific model (e.g., `--no-run-svr`). See `python quantum_regression.py --help` for all model flags and other hyperparameters.
*   **Model Hyperparameters:** Arguments like `--qnn-hidden-dim`, `--n-qubits`, `--transformer-nhead`, `--svr-c`, etc., allow tuning specific model parameters.

**Example:** Run only the QNN and MLP models with specific hyperparameters for 100 epochs:

```bash
python quantum_regression.py --run-qnn --run-mlp --no-run-transformer --no-run-cnn --no-run-resnet --no-run-lstm --no-run-qasa --no-run-svr --no-run-knn --no-run-xgboost --no-run-lightgbm --epochs 100 --qnn-hidden-dim 512 --mlp-hidden-dim 512
```

**Or, more simply, disable all models first (assuming a future flag is added) and then enable the ones you want, or just disable the ones you *don't* want:**

```bash
# Assuming defaults run most models, just disable the ones you don't need
python quantum_regression.py --no-run-transformer --no-run-cnn --no-run-resnet --no-run-lstm --no-run-qasa --no-run-svr --no-run-knn --no-run-xgboost --no-run-lightgbm --epochs 100
```

## Data

The script expects an Excel file (`.xlsx`) with a sheet containing the data.
*   The target columns to be predicted are hardcoded as `['LCE', 'sC', 'aC', 'C']`.
*   All other columns with numerical data types (float, int) will be used as input features.
*   The script includes preprocessing steps like handling missing values (median imputation) and feature scaling (RobustScaler, falling back to StandardScaler).

## Results

*   For each model run, a directory named `<save-dir-prefix>_<ModelName>` (e.g., `results_QNN`) is created.
*   Inside each model directory, you'll find:
    *   `loss_curves.png`: Training and validation loss curves (for DL models).
    *   `best_model.pt` / `final_model.pt`: Saved model checkpoints (for DL models).
    *   `scatter_<TargetName>.png`: Scatter plot comparing true vs. predicted values for each target.
    *   `metrics.csv`: CSV file containing MSE, RMSE, MAE, and R2 scores for each target.
    *   `predictions.csv`: CSV file with the model's predictions on the test set.
*   A summary directory named `<save-dir-prefix>_summary` (e.g., `results_summary`) is created if multiple models are run.
*   This directory contains comparison bar plots (`summary_<MetricName>.png`) for key metrics (RMSE, MAE, R2) across all models.
![image](https://github.com/user-attachments/assets/65526350-bb12-4347-8734-ef6f0f0a2ce1)


