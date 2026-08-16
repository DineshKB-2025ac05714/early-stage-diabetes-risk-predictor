# Early Stage Diabetes Risk Predictor

## 1. Problem Statement

The goal of this project is to build, evaluate, and deploy an end-to-end machine learning classification system to predict early-stage diabetes risk based on demographic information and reported symptoms.

## 2. Dataset Description
* **Source:** UCI Machine Learning Repository (Early Stage Diabetes Risk Prediction Dataset)
* **Dataset link:** https://archive.ics.uci.edu/dataset/529/early+stage+diabetes+risk+prediction+dataset
* **Dataset shape:** 520 x 17
* **Instance Count:** 520 instances
* **Feature Count:** 16 clinical features

   (Gender, Polyuria, Polydipsia, sudden weight loss, weakness, Polyphagia, Genital thrush, visual blurring, Itching, Irritability, delayed healing, partial paresis, muscle stiffness, Alopecia, Obesity)
* **Target Label:** Binary class (`Positive` = 1, `Negative` = 0)
* **Target Distribution** `Positive` =  320 (61.54%), `Negative` = 200 (38.46%)

## 3. GitHub Repository Link
https://github.com/DineshKB-2025ac05714/early-stage-diabetes-risk-predictor/tree/master

## 4. Models Used & Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.9135 | 0.9762 | 0.9365 | 0.9219 | 0.9291 | 0.8182 |
| **Decision Tree** | 0.9808 | 0.9797 | 0.9844 | 0.9844 | 0.9844 | 0.9594 |
| **K-Nearest Neighbor** | 0.9423 | 0.9832 | 0.9677 | 0.9375 | 0.9524 | 0.8800 |
| **Naive Bayes** | 0.9038 | 0.9535 | 0.9219 | 0.9219 | 0.9219 | 0.7969 |
| **Random Forest (Ensemble)** | 0.9808 | 0.9988 | 0.9844 | 0.9844 | 0.9844 | 0.9594 |

## 5. Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Achieved a baseline accuracy of 91.35% and strong AUC (0.9762). Features like Polyuria and Polydipsia provide strong linear separability for early detection. |
| **Decision Tree** | Outstanding overall accuracy (98.08%) and high MCC (0.9594), successfully capturing multi-symptom decision paths with high precision and recall. |
| **K-Nearest Neighbor** | Solid accuracy of 94.23% with high precision (0.9677) post-scaling, successfully grouping similar clinical risk profiles into local clusters. |
| **Naive Bayes** | Reliable 90.38% accuracy. Performance is slightly lower than tree models due to feature correlation among co-occurring diabetic symptoms. |
| **Random Forest (Ensemble)** | Exceptional performance across all parameters, matching Decision Tree's top accuracy (98.08%) while reaching the highest AUC score (0.9988). |

## 6.Overall Winner for dataset:

**Random Forest (Ensemble)** achieved the highest AUC score of **0.9988** on the held-out test dataset.

**Decision Tree** and **Random Forest** achieved the highest Accuracy (**98.08%**), Precision (**98.44%**), Recall (**98.44%**), F1 Score (**98.44%**), and MCC (**0.9594**).

Therefore, based on the evaluated metrics,

**Random Forest** provides the strongest overall performance, particularly due to its highest AUC.


> **Disclaimer:** This application is intended for educational and academic purposes only. It is not a medical diagnostic tool and should not be used as a substitute for professional medical advice.

--------------
## Project Structure
```
early-stage-diabetes-risk-predictor/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
├── test_data.csv
│
├── model/
│   ├── logistic_regression.py
│   ├── logistic_regression.pkl
│   ├── knn.py
│   ├── knn.pkl
│   ├── decision_tree.py
│   ├── decision_tree.pkl
│   ├── naive_bayes.py
│   ├── naive_bayes.pkl
│   ├── random_forest.py
│   └── random_forest.pkl
│
└── utils/
    ├── data_validator_and_preprocessor.py
    └── model_evaluation_and_display_results.py
```
## Installation Guidelines

**Clone Repositry:**

`git clone https://github.com/DineshKB-2025ac05714/early-stage-diabetes-risk-predictor.git`

**Navigate to the project directory:**

`cd early-stage-diabetes-risk-predictor`

**Create a virtual environment:**

`python3 -m venv .venv`

**Activate the virtual environment:**

`source .venv/bin/activate`

**Install the required dependencies:**

`pip install -r requirements.txt`

## Running the Models

**Logistic Regression**

`python3 model/logistic_regression.py`

**KNN**

`python3 model/knn.py`

**Decision Tree**

`python3 model/decision_tree.py`

**Naive Bayes**

`python3 model/naive_bayes.py`

**Random Forest**

`python3 model/random_forest.py`

> **Note:** The trained models are saved as `.pkl` files in the model/ directory.

--------------
