# Predictive Maintenance MSc Project

This folder contains the completed working project created from the uploaded `ai4i2020.csv` dataset.

## What is included

- Trained Random Forest, SVM and exploratory one-step LSTM models
- Genuine test-set metrics and predictions
- Training-time, inference-time and memory measurements
- Confusion matrices and ROC curves
- Random Forest feature importance
- SHAP summary plots for Random Forest, SVM and LSTM
- A Streamlit dashboard
- A completed Chapter 3 draft
- A Chapter 4 results draft
- A small Chapter 2 correction required because the final LSTM uses PyTorch

## Easiest way to open the dashboard on Windows

1. Extract the ZIP folder.
2. Open the extracted `predictive_maintenance_project` folder.
3. Double-click `setup_windows.bat` once.
   - This creates a private Python environment and installs the required packages.
   - Installation can take several minutes, especially PyTorch.
4. After setup finishes, double-click `run_dashboard.bat`.
5. A browser window should open with the dashboard.

If Windows says that `py` is not recognised, install 64-bit Python and make sure **Add Python to PATH** is selected during installation.

## Files for the dissertation

- Chapter 3: `report/chapter3_completed.md`
- Chapter 4: `report/chapter4_results_draft.md`
- Required Chapter 2 edit: `report/chapter2_required_correction.md`
- Results summary: `PROJECT_RESULTS_SUMMARY.md`
- Figures: `outputs/figures/`
- Tables and genuine values: `outputs/tables/`
- Trained models: `models/`
- Source code: `src/`

## Screenshot needed for Chapter 3

After the dashboard opens:

1. Open the **Model comparison** tab.
2. Press `Windows + Shift + S`.
3. Drag around the dashboard window.
4. Save the screenshot as `figure_3_streamlit_dashboard.png`.
5. Insert it into Chapter 3 where Figure 3 is discussed.

## Important academic note

Read the code, results and chapter drafts before submission so that the implementation can be explained during supervision or assessment. The LSTM is deliberately described as exploratory because AI4I does not contain genuine machine-specific temporal histories.


## Streamlit dependency
The dashboard dependency is pinned to Streamlit 1.49.1 in requirements.txt for reproducible deployment. Model training results do not depend on Streamlit.
