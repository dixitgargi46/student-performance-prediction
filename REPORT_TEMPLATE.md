# Project Report: AI-Powered Student Performance Prediction System

**Student Name:** Gargi Dixit  
**Course / Semester:** BA Programme, 1st Semester  
**College:** Maitreyi College, University of Delhi  

## 1. Introduction

The AI-Powered Student Performance Prediction System is a machine learning project designed to predict the broad academic performance category of a student. The system uses historical student data and machine learning classification algorithms to estimate performance.

The project uses Python libraries such as Pandas, NumPy, Matplotlib, Seaborn and Scikit-learn for data processing, visualization and model training.

## 2. Problem Statement

It can be difficult to identify students who may need additional academic support. This project demonstrates how historical student information can be used to estimate broad performance categories using machine learning techniques.

## 3. Objectives

- Clean and explore a public student dataset.
- Train three classification algorithms.
- Compare models using standard evaluation metrics.
- Save the best-performing model.
- Build an interactive command-line prediction program.

## 4. Dataset Description

The dataset used in this project is the UCI Student Performance dataset by Paulo Cortez (2008). The Mathematics course file, named `student-mat.csv`, was selected.

The dataset contains 395 student records and 33 attributes. It includes information related to students' previous grades, study time, absences, family support, educational support and other student-related factors.

**Data limitation:** The source dataset does not contain attendance percentage, daily study hours, assignment completion or classroom participation. This project uses available fields such as absences, weekly study-time category, previous grades and support/activity indicators.

## 5. Methodology

The project was completed using the following steps:

1. Downloaded the dataset from the UCI Machine Learning Repository.
2. Removed duplicate records.
3. Separated input features and final grade.
4. Converted final grade (G3) into three performance categories.
5. Imputed missing values, scaled numerical features and one-hot encoded categorical features.
6. Split the data into training and testing sets.
7. Trained Logistic Regression, Decision Tree and Random Forest models.
8. Compared the evaluation metrics of all models.
9. Performed 5-fold cross-validation for the selected model.
10. Saved the trained model and created a terminal-based prediction system.

## 6. System Workflow

Dataset → Data Cleaning → Exploratory Data Analysis → Preprocessing → Model Training → Evaluation → Saved Model → Terminal Prediction.

## 7. Implementation Details

### download_data.py
This file downloads the official student performance dataset from the UCI Machine Learning Repository and extracts the required files.

### preprocess.py
This file loads the dataset, removes duplicate records, selects input features and converts final grades into three performance categories.

### eda.py
This file performs exploratory data analysis and generates graphs to understand the student data and relationships between selected variables.

### train_model.py
This file preprocesses the data, trains three classification algorithms, compares their performance, performs cross-validation and saves the best-performing trained model.

### predict.py
This file provides an interactive command-line interface. It accepts student-related inputs and predicts the student's broad performance category using the saved model.

## 8. Model Evaluation

The dataset was divided into training and testing sets. Three machine learning classification models were trained and evaluated.

| Model | Accuracy | Weighted Precision | Weighted Recall | Weighted F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 81.01% | 81.16% | 81.01% | 80.73% |
| Decision Tree | 84.81% | 85.04% | 84.81% | 84.83% |
| Random Forest | 84.81% | 85.18% | 84.81% | 84.90% |

Random Forest was selected as the best-performing model because it achieved the highest weighted F1-score among the three models.

### Random Forest Confusion Matrix

The confusion matrix is shown in the following class order: Needs Improvement, Average Performance, Good Performance.

| Actual / Predicted | Needs Improvement | Average Performance | Good Performance |
|---|---:|---:|---:|
| Needs Improvement | 22 | 4 | 0 |
| Average Performance | 4 | 28 | 1 |
| Good Performance | 0 | 3 | 17 |

### Cross-validation

The selected model was evaluated using 5-fold cross-validation.

The F1-scores obtained were approximately:

- Fold 1: 87.20%
- Fold 2: 82.23%
- Fold 3: 86.21%
- Fold 4: 77.40%
- Fold 5: 91.20%

The average cross-validation F1-score was approximately 84.85%.

The evaluation shows that Random Forest performed well on the selected test data. However, the model's predictions are estimates and may not generalize to other student populations.

## 9. Sample Outputs

The following screenshots should be added to this section:

- Dataset preview and performance category counts.
- Exploratory data analysis graphs.
- Model evaluation and comparison output.
- A complete terminal prediction showing the input values and predicted category.

## 10. Limitations

- The dataset represents secondary-school students in Portugal.
- The dataset does not include every feature mentioned in the original project brief.
- Performance categories are defined for this project based on final grades.
- The model's predictions are estimates and should not be treated as definitive academic outcomes.
- The model may not generalize to other student populations without further testing.

## 11. Conclusion

The AI-Powered Student Performance Prediction System was successfully developed using Python and machine learning. The project involved dataset preprocessing, exploratory data analysis, training and comparison of three classification algorithms, model evaluation and development of an interactive terminal prediction system.

Among the three models, Random Forest achieved the highest weighted F1-score of approximately 84.90% and an accuracy of 84.81%. The trained model was saved and used in the terminal-based prediction program.

This project demonstrates how machine learning can be applied to student data to estimate broad academic performance categories.

## 12. Future Enhancements

A future version could use a suitable dataset containing attendance, daily study hours, assignment completion and classroom participation. The system could also be tested on a broader student population and include additional features to improve its usefulness.

## References

Cortez, P. (2008). *Student Performance* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5TG7T
