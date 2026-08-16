# --------------------------------------------------------
# Import block
# --------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --------------------------------------------------------
# Configuration block
# --------------------------------------------------------

# Configuration specific to  early diabetic prediction dataset


DATA_PATH = "data/diabetes_data.csv" # Dataset file location
TEST_FILE = "test_data.csv"
R_SEED = 11  # Fixed seed for reproducible train-test split
TEST_SIZE = 0.20 # 80/20 split
ITERATION=1500 # No. of iterations

# Target Column
target_column = "class"

# Categorical columns
categorical_columns = [
  "Gender",
  "Polyuria",
  "Polydipsia",
  "sudden weight loss",
  "weakness",
  "Polyphagia",
  "Genital thrush",
  "visual blurring",
  "Itching",
  "Irritability",
  "delayed healing",
  "partial paresis",
  "muscle stiffness",
  "Alopecia",
  "Obesity"
]

# Numerical column
numerical_columns = ["Age"]

# --------------------------------------------------------
# Function defintions
# --------------------------------------------------------


# Function to load dataset

def load_dataset(file_path):
    return pd.read_csv(file_path)

# Function to validate dataset and display overview
def validate_dataset(df):
    print("=" * 60)
    print("DATASET VALIDATION")
    print("=" * 60)

    # 1. Shape
    print("\nDataset Shape:")
    print(df.shape)

    # 2. Column names
    print("\nColumns:")
    for i, col in enumerate(df.columns, start=1):
        print(f"{i:2}. {col}")

    # 3. Data types
    print("\nData Types:")
    print(df.dtypes)

    # 4. Missing values
    print("\nMissing Values:")
    print(df.isnull().sum())

    # 5. Target distribution
    print("\nTarget Distribution:")
    print(df["class"].value_counts())

    print("\nTarget Distribution (%):")
    print(df["class"].value_counts(normalize=True) * 100)

    # 6. Unique values
    print("\nUnique Values:")
    for col in df.columns:
        print(f"\n{col}:")
        print(df[col].unique())


# Function to seperate input features and target
def prepare_features_target(df):

    X = df.drop(columns=[target_column])
    y = df[target_column].map({
        "Negative": 0,
        "Positive": 1
    })

    return X, y

# Function to save test data
def save_test_data(X_test, y_test, file_path):
    """
    Save the test dataset including the original target labels.
    """
    test_data = X_test.copy()

    test_data["class"] = y_test.map({
        0: "Negative",
        1: "Positive"
    })

    test_data.to_csv(
        file_path,
        index=False
    )

# Function to perform 80/20 stratified split
# Performing 80/20 Stratified split.

def split_dataset(X, y):
    """
    Perform an 80/20 stratified train-test split.
	Stratified because of the proportion of the  target class
	Positive class: ~ 61.5%
	Negative class: ~ 38.5%
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=R_SEED,
        stratify=y
    )
    # Save the test data into CSV file
    save_test_data(X_test, y_test, TEST_FILE)
    return X_train, X_test, y_train, y_test

# Function to pre-process dataset
# Handles suitable encoding
def create_preprocessor(scale_numeric=False):
    """
    Categorical variables: One-Hot Encoding

    Numerical variables: Age

    scale_numeric=True:
        - Used for models such as Logistic Regression and kNN.

    scale_numeric=False:
        - Used for tree-based models and Naive Bayes.
    """

    if scale_numeric:
        numerical_transformer = StandardScaler()
    else:
        numerical_transformer = "passthrough"

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numerical_transformer,
                numerical_columns
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    drop="if_binary"
                ),
                categorical_columns
            )
        ]
    )

    return preprocessor
