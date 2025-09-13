# Previsions Forecast

This project provides a simple forecast tool for incoming call volumes using a Streamlit dashboard. It includes:

- **app.py** &ndash; an interactive Streamlit interface to visualise historical data and future predictions for Front Office (FO) and Social Networks (RRSS).
- **train_models_FO.py** &ndash; trains a RandomForest model from the provided Excel files, performing a grid search over an expanded range of hyperparameters before saving the resulting pickled model.
- **BBDD_calls2.xlsx** and **BBDD_calls_RRSS.xlsx** &ndash; example data sets containing the columns `dat` (date), `con` (calls) and `cyb`.

Pre‑trained models (`*.pkl`) are already included.

## Installation

1. Create a virtual environment (optional but recommended).
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the dashboard
```bash
streamlit run app.py
```

The dashboard uses responsive styles so the layout looks consistent on
desktop and mobile browsers (Android and iOS).

### Training a model
```bash
python train_models_FO.py
```
The training script now uses a grid search with time-series cross-validation to optimise parameters such as the number of estimators, tree depth and feature sampling, alongside the existing time-based features (e.g. quarter, day of year, cyclical encodings and additional lag/rolling statistics). It reports R2, MAE and RMSE scores for each dataset. Adjust the file paths inside `train_models_FO.py` if your data is stored elsewhere.

## Requirements
All required Python packages are listed in `requirements.txt`.
