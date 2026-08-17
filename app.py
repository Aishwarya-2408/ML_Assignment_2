import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    classification_report,
    confusion_matrix
)


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Dry Bean Classification",
    page_icon="🌱",
    layout="wide"
)


# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("🌱 Dry Bean Classification using Machine Learning")

st.write(
    """
    This application predicts Dry Bean classes using trained
    machine learning classification models.

    The application supports:
    - Multiple ML model selection
    - Test CSV upload
    - Performance evaluation
    - Classification report
    - Confusion matrix visualization
    """
)


# --------------------------------------------------
# Paths
# --------------------------------------------------

MODEL_FOLDER = "model"


MODEL_FILES = {

    "Logistic Regression":
        "logistic_regression.pkl",

    "Decision Tree":
        "decision_tree.pkl",

    "KNN":
        "knn.pkl",

    "Naive Bayes":
        "naive_bayes.pkl",

    "Random Forest":
        "random_forest.pkl"
}



# --------------------------------------------------
# Load Models
# --------------------------------------------------

@st.cache_resource
def load_models():

    models = {}

    for name, file in MODEL_FILES.items():

        path = os.path.join(
            MODEL_FOLDER,
            file
        )

        if os.path.exists(path):

            models[name] = joblib.load(path)

    return models



models = load_models()



if len(models) == 0:

    st.error(
        "No trained models found. "
        "Please place .pkl files inside model folder."
    )

    st.stop()



# --------------------------------------------------
# Load Label Encoder
# --------------------------------------------------

encoder_path = os.path.join(
    MODEL_FOLDER,
    "label_encoder.pkl"
)


if os.path.exists(encoder_path):

    label_encoder = joblib.load(
        encoder_path
    )

    class_names = label_encoder.classes_

else:

    class_names = [
        "BARBUNYA",
        "BOMBAY",
        "CALI",
        "DERMASON",
        "HOROZ",
        "SEKER",
        "SIRA"
    ]



# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.header(
    "Model Configuration"
)


selected_model = st.sidebar.selectbox(

    "Select Machine Learning Model",

    list(models.keys())

)



uploaded_file = st.sidebar.file_uploader(

    "Upload Test CSV File",

    type=["csv"]

)



# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

if uploaded_file is not None:

    df = pd.read_csv(
        uploaded_file
    )

    st.sidebar.success(
        "CSV uploaded successfully"
    )


else:

    default_file = "test_data.csv"


    if os.path.exists(default_file):

        df = pd.read_csv(
            default_file
        )

        st.sidebar.info(
            "Using default test_data.csv"
        )

    else:

        st.warning(
            "Please upload test CSV file"
        )

        st.stop()



# --------------------------------------------------
# Dataset Preview
# --------------------------------------------------

st.subheader(
    "Dataset Preview"
)

st.dataframe(
    df.head(),
    use_container_width=True
)



# --------------------------------------------------
# Separate Features and Target
# --------------------------------------------------

if "Class" not in df.columns:

    st.error(
        "CSV must contain target column 'Class'"
    )

    st.stop()



X_test = df.drop(
    "Class",
    axis=1
)


y_test = df["Class"]



# Convert labels if encoded

try:

    y_test_encoded = y_test.astype(int)


except:

    y_test_encoded = label_encoder.transform(
        y_test
    )



# --------------------------------------------------
# Prediction
# --------------------------------------------------

model = models[selected_model]


predictions = model.predict(
    X_test
)



# Probability for AUC

try:

    probabilities = model.predict_proba(
        X_test
    )


except:

    probabilities = None



# --------------------------------------------------
# Evaluation Metrics
# --------------------------------------------------

st.subheader(
    f"Evaluation Results - {selected_model}"
)


accuracy = accuracy_score(
    y_test_encoded,
    predictions
)


precision = precision_score(
    y_test_encoded,
    predictions,
    average="weighted",
    zero_division=0
)


recall = recall_score(
    y_test_encoded,
    predictions,
    average="weighted",
    zero_division=0
)


f1 = f1_score(
    y_test_encoded,
    predictions,
    average="weighted",
    zero_division=0
)


mcc = matthews_corrcoef(
    y_test_encoded,
    predictions
)



if probabilities is not None:

    auc = roc_auc_score(

        y_test_encoded,

        probabilities,

        multi_class="ovr",

        average="weighted"

    )

else:

    auc = np.nan



# Display metrics

col1, col2, col3 = st.columns(3)

col4, col5, col6 = st.columns(3)



col1.metric(
    "Accuracy",
    round(accuracy,4)
)


col2.metric(
    "AUC",
    round(auc,4)
)


col3.metric(
    "Precision",
    round(precision,4)
)


col4.metric(
    "Recall",
    round(recall,4)
)


col5.metric(
    "F1 Score",
    round(f1,4)
)


col6.metric(
    "MCC",
    round(mcc,4)
)



# --------------------------------------------------
# Classification Report
# --------------------------------------------------

st.subheader(
    "Classification Report"
)


report = classification_report(

    y_test_encoded,

    predictions,

    target_names=class_names,

    output_dict=True,

    zero_division=0

)


report_df = pd.DataFrame(
    report
).transpose()


st.dataframe(
    report_df.round(4),
    use_container_width=True
)



# --------------------------------------------------
# Confusion Matrix
# --------------------------------------------------

st.subheader(
    "Confusion Matrix"
)



cm = confusion_matrix(

    y_test_encoded,

    predictions

)



fig, ax = plt.subplots(
    figsize=(8,6)
)



sns.heatmap(

    cm,

    annot=True,

    fmt="d",

    cmap="Blues",

    xticklabels=class_names,

    yticklabels=class_names,

    ax=ax

)



ax.set_xlabel(
    "Predicted Class"
)


ax.set_ylabel(
    "Actual Class"
)


ax.set_title(

    selected_model + " Confusion Matrix"

)



plt.tight_layout()



st.pyplot(
    fig
)



# --------------------------------------------------
# Footer
# --------------------------------------------------

st.success(
    "Model evaluation completed successfully."
)