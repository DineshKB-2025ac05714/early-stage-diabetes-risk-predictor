import os
import sys
import joblib

from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

# Allow importing files from utils/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import variables
from utils.data_validator_and_preprocessor import (
    R_SEED,
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
MODEL_NAME="decision_tree"
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

# Standard Scalar is not needed for Age (Numberic data) in decision tree
preprocessor = create_preprocessor(scale_numeric=False)

# Creating Decision tree classifier model
model = DecisionTreeClassifier(
    random_state=R_SEED
)
# Creating pipeline for preprocessing and classification
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", model)
    ]
)

# ---------------------------------------------------------
# Model Training
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
