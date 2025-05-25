# Tensor Neural Networks For Hyperspectral Data Classification

This project implements and compares several machine learning models—including Logistic Regression (LR), Multi-Layer Perceptron (MLP), Convolutional Neural Network (CNN), and Tensor Neural Network (TNN)—for hyperspectral image classification. The code supports hyperparameter tuning using Optuna and provides scripts for result aggregation and visualization.

## Project Structure

```
.
├── CNN_TNN_Final.ipynb                # Main notebook for CNN and TNN experiments
├── Hyperparamater Tuning.ipynb        # Hyperparameter tuning for LR, MLP and TNN
├── results.py                         # Aggregates results and generates plots
├── Datasets/                          # Hyperspectral datasets (.mat files)
├── plots/                             # Output accuracy plots
├── results/                           # (Optional) Additional results
├── README.md                          # Project documentation
└── .gitignore
```

## Datasets

The `Datasets/` folder contains the following hyperspectral datasets:
- Botswana
- Indian Pines (corrected)
- Kennedy Space Center (KSC)
- Pavia
- Salinas (corrected)

Each dataset includes both the data and ground truth labels in `.mat` format.

## Usage

### 1. Hyperparameter Tuning

Use [Hyperparamater Tuning.ipynb](Hyperparamater%20Tuning.ipynb) to tune hyperparameters for each model using Optuna. The notebook includes:
- Logistic Regression tuning
- MLP (FCNN) tuning
- CNN and TNN tuning (with PyTorch)

### 2. Running Experiments

You can run experiments and save results in Excel format. Each run should be saved as a separate sheet (e.g., "Run 1", "Run 2", ...).

### 3. Aggregating Results and Plotting

Use [results.py](results.py) to aggregate results from multiple runs and generate accuracy plots for each dataset and model. The script expects an Excel file with results from multiple runs.

**To run:**
```sh
python results.py
```
Plots will be saved in the `plots/` directory.

## Requirements

- Python 3.x
- numpy
- pandas
- matplotlib
- scikit-learn
- torch
- optuna
- scipy (for loading .mat files)

Install dependencies with:
```sh
pip install numpy pandas matplotlib scikit-learn torch optuna scipy
```

## Results

Accuracy plots for each dataset and model are saved in the `plots/` directory, e.g.:
- `Botswana_accuracy_vs_samples.png`
- `indian_pines_corrected_accuracy_vs_samples.png`
- ...

## References

- [results.py](results.py): Aggregates and plots results.
- [Hyperparamater Tuning.ipynb](Hyperparamater%20Tuning.ipynb): Hyperparameter optimization.
- [CNN_TNN_Final.ipynb](CNN_TNN_Final.ipynb): Main experiments.

---

**Note:** Update the Excel file path in `results.py` as needed for your results file.
