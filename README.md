# AI-Powered Student Performance Prediction System

## Overview
A command-line machine-learning project that predicts one of three academic performance categories from student information.

## Dataset
This project uses the **Student Performance** dataset by Paulo Cortez (UCI Machine Learning Repository), specifically `student-mat.csv` (Mathematics course).

- Official source: https://archive.ics.uci.edu/dataset/320/student%2Bperformance
- Citation: Cortez, P. (2008). Student Performance [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5TG7T.
- License: Creative Commons Attribution 4.0 (CC BY 4.0).
- The original data contains 395 mathematics-course records and 33 attributes.

**Important dataset limitation:** The original UCI dataset does not contain attendance percentage, daily study hours, assignment completion, or classroom participation. This implementation uses available related fields such as absences, weekly study-time category, previous period grades, support, and activities. Do not describe these proxy fields as the exact requested fields.

## Performance categories
The original final grade (`G3`) is on a 0–20 scale. This project groups it as:
- 0–9: Needs Improvement
- 10–13: Average Performance
- 14–20: Good Performance

These thresholds are project-defined categories, not official UCI labels.

## Algorithms
- Logistic Regression
- Decision Tree
- Random Forest

The script compares accuracy, weighted precision, weighted recall, weighted F1-score and confusion matrices. It also performs 5-fold cross-validation for the selected model. Actual results are written to `outputs/evaluation_results.json` when you run training.

## Installation (Windows)
1. Install Python 3.10 or newer and tick **Add Python to PATH** during setup.
2. Open this folder in VS Code.
3. Open Terminal > New Terminal.
4. Run:
   ```bash
   python -m pip install -r requirements.txt
   python download_data.py
   python eda.py
   python train_model.py
   python predict.py
   ```

Alternatively, double-click `run_project.bat`.

## Project files
- `download_data.py`: downloads the official UCI dataset.
- `preprocess.py`: removes duplicate rows and creates the target classes.
- `eda.py`: creates three graphs.
- `train_model.py`: preprocesses features, trains and compares three classifiers, evaluates them and saves the best pipeline.
- `predict.py`: accepts terminal inputs and predicts a category.
- `models/`: saved model.
- `outputs/`: evaluation results and graphs.

## Expected output
The terminal asks for student details and prints one predicted category with observations and suggestions. Predictions are estimates and are not guaranteed outcomes.

## Limitations
The dataset is from secondary-school students in Portugal, not a representative dataset of all colleges or countries. Some fields requested in the project brief are not present in the source dataset. The model should be treated as an educational demonstration, not a high-stakes student assessment tool.
