import os

import joblib
import pandas as pd
import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Early-Stage Diabetes Risk Predictor",
    page_icon="🩺",
    layout="wide"
)


# ============================================================
# BASE DIRECTORY
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_PATHS = {
    "Logistic Regression": os.path.join(
        BASE_DIR,
        "model",
        "logistic_regression.pkl"
    ),

    "Decision Tree": os.path.join(
        BASE_DIR,
        "model",
        "decision_tree.pkl"
    ),

    "K-Nearest Neighbors": os.path.join(
        BASE_DIR,
        "model",
        "knn.pkl"
    ),

    "Gaussian Naive Bayes": os.path.join(
        BASE_DIR,
        "model",
        "naive_bayes.pkl"
    ),

    "Random Forest": os.path.join(
        BASE_DIR,
        "model",
        "random_forest.pkl"
    )
}


# ============================================================
# EXPECTED DATASET COLUMNS
# ============================================================

EXPECTED_COLUMNS = [
    "Age",
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
    "Obesity",
    "class"
]


FEATURE_COLUMNS = [
    column
    for column in EXPECTED_COLUMNS
    if column != "class"
]


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model(model_path):
    """
    Load a saved preprocessing + machine learning pipeline.
    """
    return joblib.load(model_path)


# ============================================================
# CUSTOM UI STYLING
# ============================================================

st.markdown(
    """
    <style>

    /* Main tab text */
    button[data-baseweb="tab"] {
        font-size: 20px !important;
        font-weight: 600 !important;
        padding: 14px 28px !important;
    }

    button[data-baseweb="tab"] p {
        font-size: 20px !important;
        font-weight: 600 !important;
    }

    /* Model selection label */
    div[data-testid="stSelectbox"] label {
        font-size: 17px !important;
        font-weight: 600 !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.title("🩺 Early-Stage Diabetes Risk Predictor")

st.markdown(
    """
    A machine learning application for predicting **early-stage diabetes
    risk** using demographic information and reported symptoms.
    """
)

st.warning(
    "⚠️ This application is for educational and demonstration purposes "
    "only. It is not a medical diagnostic tool."
)


# ============================================================
# MODEL SELECTION
# ============================================================

st.markdown("### 🤖 Select Machine Learning Model")

st.caption(
    "The selected model is used for patient risk prediction. "
    "The test-data evaluation section evaluates all available models."
)

model_col1, model_col2, model_col3 = st.columns([1, 2, 1])

with model_col2:

    selected_model = st.selectbox(
        "Choose a Machine Learning Model",
        list(MODEL_PATHS.keys()),
        label_visibility="collapsed"
    )


selected_model_path = MODEL_PATHS[selected_model]


# ============================================================
# CHECK SELECTED MODEL
# ============================================================

if not os.path.exists(selected_model_path):

    st.error(
        f"Model file not found:\n\n"
        f"`{selected_model_path}`\n\n"
        "Please run the corresponding model training script first."
    )

    st.stop()


selected_model_object = load_model(
    selected_model_path
)


# ============================================================
# TABS
# ============================================================

prediction_tab, evaluation_tab = st.tabs(
    [
        "🔍 Risk Prediction",
        "📊 Test Data Evaluation"
    ]
)


# ################################################################
# TAB 1 — RISK PREDICTION
# ################################################################

with prediction_tab:

    st.header("🔍 Early-Stage Diabetes Risk Prediction")

    st.write(
        f"Prediction using **{selected_model}**."
    )


    # ------------------------------------------------------------
    # PATIENT INFORMATION
    # ------------------------------------------------------------

    st.subheader("Patient Information")

    col1, col2 = st.columns(2)


    # ------------------------------------------------------------
    # LEFT COLUMN
    # ------------------------------------------------------------

    with col1:

        age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=40,
            step=1
        )

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        polyuria = st.selectbox(
            "Polyuria",
            ["No", "Yes"]
        )

        polydipsia = st.selectbox(
            "Polydipsia",
            ["No", "Yes"]
        )

        sudden_weight_loss = st.selectbox(
            "Sudden Weight Loss",
            ["No", "Yes"]
        )

        weakness = st.selectbox(
            "Weakness",
            ["No", "Yes"]
        )

        polyphagia = st.selectbox(
            "Polyphagia",
            ["No", "Yes"]
        )

        genital_thrush = st.selectbox(
            "Genital Thrush",
            ["No", "Yes"]
        )


    # ------------------------------------------------------------
    # RIGHT COLUMN
    # ------------------------------------------------------------

    with col2:

        visual_blurring = st.selectbox(
            "Visual Blurring",
            ["No", "Yes"]
        )

        itching = st.selectbox(
            "Itching",
            ["No", "Yes"]
        )

        irritability = st.selectbox(
            "Irritability",
            ["No", "Yes"]
        )

        delayed_healing = st.selectbox(
            "Delayed Healing",
            ["No", "Yes"]
        )

        partial_paresis = st.selectbox(
            "Partial Paresis",
            ["No", "Yes"]
        )

        muscle_stiffness = st.selectbox(
            "Muscle Stiffness",
            ["No", "Yes"]
        )

        alopecia = st.selectbox(
            "Alopecia",
            ["No", "Yes"]
        )

        obesity = st.selectbox(
            "Obesity",
            ["No", "Yes"]
        )


    # ------------------------------------------------------------
    # CREATE INPUT DATAFRAME
    # ------------------------------------------------------------

    input_data = pd.DataFrame(
        {
            "Age": [age],
            "Gender": [gender],
            "Polyuria": [polyuria],
            "Polydipsia": [polydipsia],
            "sudden weight loss": [sudden_weight_loss],
            "weakness": [weakness],
            "Polyphagia": [polyphagia],
            "Genital thrush": [genital_thrush],
            "visual blurring": [visual_blurring],
            "Itching": [itching],
            "Irritability": [irritability],
            "delayed healing": [delayed_healing],
            "partial paresis": [partial_paresis],
            "muscle stiffness": [muscle_stiffness],
            "Alopecia": [alopecia],
            "Obesity": [obesity]
        }
    )


    # ------------------------------------------------------------
    # PREDICTION BUTTON
    # ------------------------------------------------------------

    st.divider()

    predict_button = st.button(
        "🔍 Predict Diabetes Risk",
        type="primary",
        use_container_width=True
    )


    if predict_button:

        prediction = selected_model_object.predict(
            input_data
        )[0]

        probabilities = selected_model_object.predict_proba(
            input_data
        )[0]

        negative_probability = probabilities[0]

        positive_probability = probabilities[1]


        # --------------------------------------------------------
        # PREDICTION RESULT
        # --------------------------------------------------------

        st.subheader("Prediction Result")


        if prediction == 1:

            st.error(
                "⚠️ **Positive — Early-Stage Diabetes Risk Predicted**"
            )

        else:

            st.success(
                "✅ **Negative — Early-Stage Diabetes Risk Not Predicted**"
            )


        # --------------------------------------------------------
        # PROBABILITIES
        # --------------------------------------------------------

        st.subheader("Prediction Probability")

        probability_col1, probability_col2 = st.columns(2)


        with probability_col1:

            st.metric(
                "Negative Probability",
                f"{negative_probability:.2%}"
            )


        with probability_col2:

            st.metric(
                "Positive Probability",
                f"{positive_probability:.2%}"
            )


        st.caption(
            f"Prediction generated using **{selected_model}**."
        )


        # --------------------------------------------------------
        # INPUT SUMMARY
        # --------------------------------------------------------

        with st.expander("View Patient Input"):

            st.dataframe(
                input_data.T.rename(
                    columns={0: "Value"}
                ),
                use_container_width=True
            )


# ################################################################
# TAB 2 — TEST DATA EVALUATION
# ################################################################

with evaluation_tab:

    st.header("📊 Test Data Evaluation")

    st.write(
        """
        Upload the provided **test_data.csv** file to evaluate all
        implemented machine learning models on the same held-out
        test dataset.
        """
    )

    st.info(
        "📁 Upload only the test dataset. "
        "The training dataset should not be uploaded to the application."
    )


    # ------------------------------------------------------------
    # CSV UPLOAD
    # ------------------------------------------------------------

    uploaded_file = st.file_uploader(
        "Upload Test Dataset (CSV)",
        type=["csv"],
        help=(
            "Upload the provided test_data.csv containing "
            "the held-out test samples."
        )
    )


    # ------------------------------------------------------------
    # WAIT FOR UPLOAD
    # ------------------------------------------------------------

    if uploaded_file is None:

        st.info(
            "Please upload the provided `test_data.csv` "
            "file to begin model evaluation."
        )

    else:

        # --------------------------------------------------------
        # READ CSV
        # --------------------------------------------------------

        try:

            test_data = pd.read_csv(
                uploaded_file
            )

        except Exception as error:

            st.error(
                f"Unable to read the uploaded CSV file: {error}"
            )

            st.stop()


        # --------------------------------------------------------
        # VALIDATE COLUMNS
        # --------------------------------------------------------

        missing_columns = [
            column
            for column in EXPECTED_COLUMNS
            if column not in test_data.columns
        ]


        extra_columns = [
            column
            for column in test_data.columns
            if column not in EXPECTED_COLUMNS
        ]


        if missing_columns:

            st.error(
                "❌ Invalid test dataset.\n\n"
                "Missing columns:\n\n"
                + "\n".join(
                    f"- {column}"
                    for column in missing_columns
                )
            )

            st.stop()


        if extra_columns:

            st.warning(
                "The uploaded dataset contains additional columns: "
                + ", ".join(extra_columns)
            )


        # --------------------------------------------------------
        # CHECK REQUIRED FEATURE COLUMNS
        # --------------------------------------------------------

        if not all(
            column in test_data.columns
            for column in FEATURE_COLUMNS
        ):

            st.error(
                "The uploaded dataset does not contain all "
                "required feature columns."
            )

            st.stop()


        # --------------------------------------------------------
        # VALIDATE TARGET VALUES
        # --------------------------------------------------------

        valid_targets = {
            "Positive",
            "Negative"
        }


        actual_targets = set(
            test_data["class"]
            .dropna()
            .unique()
        )


        invalid_targets = (
            actual_targets - valid_targets
        )


        if invalid_targets:

            st.error(
                "❌ Invalid values found in the `class` column:\n\n"
                + ", ".join(
                    str(value)
                    for value in invalid_targets
                )
                + "\n\nExpected values are `Positive` and `Negative`."
            )

            st.stop()


        # --------------------------------------------------------
        # CHECK MISSING VALUES
        # --------------------------------------------------------

        missing_values = test_data.isnull().sum()

        columns_with_missing_values = (
            missing_values[
                missing_values > 0
            ]
        )


        if not columns_with_missing_values.empty:

            st.error(
                "❌ The uploaded test dataset contains missing values."
            )

            st.dataframe(
                columns_with_missing_values,
                use_container_width=True
            )

            st.stop()


        # --------------------------------------------------------
        # SUCCESS MESSAGE
        # --------------------------------------------------------

        st.success(
            f"✅ Test dataset uploaded successfully — "
            f"{len(test_data)} samples."
        )


        # --------------------------------------------------------
        # TEST DATA INFORMATION
        # --------------------------------------------------------

        info_col1, info_col2, info_col3 = st.columns(3)


        with info_col1:

            st.metric(
                "Test Samples",
                len(test_data)
            )


        with info_col2:

            st.metric(
                "Features",
                len(FEATURE_COLUMNS)
            )


        with info_col3:

            st.metric(
                "Target Column",
                "class"
            )


        # --------------------------------------------------------
        # TARGET DISTRIBUTION
        # --------------------------------------------------------

        st.subheader("Test Data Target Distribution")

        target_distribution = (
            test_data["class"]
            .value_counts()
            .rename_axis("Class")
            .reset_index(name="Count")
        )


        st.dataframe(
            target_distribution,
            hide_index=True,
            use_container_width=True
        )


        # --------------------------------------------------------
        # VIEW TEST DATA
        # --------------------------------------------------------

        with st.expander("View Uploaded Test Data"):

            st.dataframe(
                test_data,
                use_container_width=True
            )


        # --------------------------------------------------------
        # PREPARE TEST DATA
        # --------------------------------------------------------

        X_test = test_data[
            FEATURE_COLUMNS
        ]

        y_test = test_data[
            "class"
        ].map(
            {
                "Negative": 0,
                "Positive": 1
            }
        )


        # --------------------------------------------------------
        # EVALUATE ALL MODELS
        # --------------------------------------------------------

        st.divider()

        st.subheader(
            "Model Performance Comparison"
        )

        results = []


        for model_name, model_path in MODEL_PATHS.items():

            if not os.path.exists(model_path):

                st.warning(
                    f"Model file not found: `{model_path}`"
                )

                continue


            current_model = load_model(
                model_path
            )


            # ----------------------------------------------------
            # PREDICTIONS
            # ----------------------------------------------------

            predictions = current_model.predict(
                X_test
            )


            probabilities = current_model.predict_proba(
                X_test
            )[:, 1]


            # ----------------------------------------------------
            # METRICS
            # ----------------------------------------------------

            accuracy = accuracy_score(
                y_test,
                predictions
            )

            auc = roc_auc_score(
                y_test,
                probabilities
            )

            precision = precision_score(
                y_test,
                predictions,
                zero_division=0
            )

            recall = recall_score(
                y_test,
                predictions,
                zero_division=0
            )

            f1 = f1_score(
                y_test,
                predictions,
                zero_division=0
            )

            mcc = matthews_corrcoef(
                y_test,
                predictions
            )


            results.append(
                {
                    "Model": model_name,
                    "Accuracy": accuracy,
                    "AUC": auc,
                    "Precision": precision,
                    "Recall": recall,
                    "F1 Score": f1,
                    "MCC": mcc
                }
            )


        # --------------------------------------------------------
        # CREATE RESULTS DATAFRAME
        # --------------------------------------------------------

        results_df = pd.DataFrame(
            results
        )


        # --------------------------------------------------------
        # DISPLAY COMPARISON
        # --------------------------------------------------------

        st.dataframe(
            results_df.style.format(
                {
                    "Accuracy": "{:.4f}",
                    "AUC": "{:.4f}",
                    "Precision": "{:.4f}",
                    "Recall": "{:.4f}",
                    "F1 Score": "{:.4f}",
                    "MCC": "{:.4f}"
                }
            ),
            use_container_width=True,
            hide_index=True
        )


        # --------------------------------------------------------
        # BEST MODEL BASED ON F1
        # --------------------------------------------------------

        if not results_df.empty:

            best_model_row = results_df.loc[
                results_df["F1 Score"].idxmax()
            ]

            st.success(
                f"🏆 Highest F1 Score: "
                f"**{best_model_row['Model']}** "
                f"({best_model_row['F1 Score']:.4f})"
            )


        # --------------------------------------------------------
        # DETAILED MODEL RESULTS
        # --------------------------------------------------------

        st.divider()

        st.subheader(
            "Detailed Model Evaluation"
        )


        for model_name, model_path in MODEL_PATHS.items():

            if not os.path.exists(model_path):

                continue


            current_model = load_model(
                model_path
            )


            predictions = current_model.predict(
                X_test
            )


            probabilities = current_model.predict_proba(
                X_test
            )[:, 1]


            # ----------------------------------------------------
            # METRICS
            # ----------------------------------------------------

            accuracy = accuracy_score(
                y_test,
                predictions
            )

            auc = roc_auc_score(
                y_test,
                probabilities
            )

            precision = precision_score(
                y_test,
                predictions,
                zero_division=0
            )

            recall = recall_score(
                y_test,
                predictions,
                zero_division=0
            )

            f1 = f1_score(
                y_test,
                predictions,
                zero_division=0
            )

            mcc = matthews_corrcoef(
                y_test,
                predictions
            )


            # ----------------------------------------------------
            # EXPANDER
            # ----------------------------------------------------

            with st.expander(
                f"📌 {model_name}"
            ):


                # ------------------------------------------------
                # METRICS
                # ------------------------------------------------

                metric_col1, metric_col2, metric_col3 = (
                    st.columns(3)
                )


                with metric_col1:

                    st.metric(
                        "Accuracy",
                        f"{accuracy:.4f}"
                    )

                    st.metric(
                        "Precision",
                        f"{precision:.4f}"
                    )


                with metric_col2:

                    st.metric(
                        "AUC",
                        f"{auc:.4f}"
                    )

                    st.metric(
                        "Recall",
                        f"{recall:.4f}"
                    )


                with metric_col3:

                    st.metric(
                        "F1 Score",
                        f"{f1:.4f}"
                    )

                    st.metric(
                        "MCC",
                        f"{mcc:.4f}"
                    )


                # ------------------------------------------------
                # CONFUSION MATRIX
                # ------------------------------------------------

                st.markdown(
                    "#### Confusion Matrix"
                )


                cm = confusion_matrix(
                    y_test,
                    predictions
                )


                cm_df = pd.DataFrame(
                    cm,
                    index=[
                        "Actual Negative",
                        "Actual Positive"
                    ],
                    columns=[
                        "Predicted Negative",
                        "Predicted Positive"
                    ]
                )


                st.dataframe(
                    cm_df,
                    use_container_width=True
                )


                # ------------------------------------------------
                # CLASSIFICATION REPORT
                # ------------------------------------------------

                st.markdown(
                    "#### Classification Report"
                )


                report = classification_report(
                    y_test,
                    predictions,
                    target_names=[
                        "Negative",
                        "Positive"
                    ],
                    output_dict=True,
                    zero_division=0
                )


                report_df = pd.DataFrame(
                    report
                ).transpose()


                st.dataframe(
                    report_df.style.format(
                        "{:.4f}"
                    ),
                    use_container_width=True
                )
