import os
import sys
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# Allow importing files from utils/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import variables
from utils.data_validator_and_preprocessor import (
    R_SEED,
    ITERATION,
    DATA_PATH
)

# Import custom functions
from utils.data_validator_and_preprocessor import (
    load_dataset,
    prepare_features_target,
    split_dataset,
    create_preprocessor
)

from utils.model_evaluation_and_display_results import (
        evaluate_model_display_results
)

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
MODEL_NAME="logistic_regression"
MODEL_PATH= f"model/{MODEL_NAME}.pkl"

# ---------------------------------------------------------
# Load and prepare data
# ---------------------------------------------------------

# Load the dataset
df = load_dataset(DATA_PATH)
# Split input features and target
X, y = prepare_features_target(df)
# Split data into train and test - 80/20 ratio - Stratified.
X_train, X_test, y_train, y_test = split_dataset(X, y)


# ---------------------------------------------------------
# Create preprocessing + model pipeline
# ---------------------------------------------------------

# Scalar number is set for Logistic regression to perform
# standard scalar for "Age" feature
preprocessor = create_preprocessor(scale_numeric=True)

# Creating the Logistic Regression model
model = LogisticRegression(
    max_iter=ITERATION,
    random_state=R_SEED
)

# Creating the pipleline for  preprocessing and classification
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", model)
    ]
)


# ---------------------------------------------------------
# Model training
# ---------------------------------------------------------

pipeline.fit(X_train, y_train)


# ---------------------------------------------------------
# Predictions
# ---------------------------------------------------------

y_pred = pipeline.predict(X_test)

y_prob = pipeline.predict_proba(X_test)[:, 1]


# ---------------------------------------------------------
# Evaluate model and display results
# ---------------------------------------------------------
evaluate_model_display_results(MODEL_NAME, y_test, y_pred, y_prob)

# ---------------------------------------------------------
# Save complete pipeline
# ---------------------------------------------------------

joblib.dump(pipeline, MODEL_PATH)

print(f"\nModel saved to: {MODEL_PATH}")
print("=" * 60)
