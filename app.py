import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef, confusion_matrix
)
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Bank Marketing Classification", layout="wide")

st.title("Bank Marketing Classification App")

st.write("Upload test dataset and select a trained model to evaluate performance.")

# Load models
MODELS = {
    "Logistic Regression": joblib.load("model/logistic_regression.pkl"),
    "Decision Tree": joblib.load("model/decision_tree.pkl"),
    "kNN": joblib.load("model/knn.pkl"),
    "Random Forest": joblib.load("model/random_forest.pkl"),
    "XGBoost": joblib.load("model/xgboost.pkl")
}

preprocessor = joblib.load("model/preprocessor.pkl")
naive_bayes = joblib.load("model/naive_bayes.pkl")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

model_name = st.selectbox("Select Model", list(MODELS.keys()) + ["Naive Bayes"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file, sep=";")
    
    if "y" not in data.columns:
        st.error("Target column 'y' not found in uploaded data")
    else:
        X = data.drop(columns=["y"])
        y = data["y"].map({"yes": 1, "no": 0})

        if model_name == "Naive Bayes":
            X_transformed = preprocessor.transform(X)
            model = naive_bayes
            y_pred = model.predict(X_transformed)
            y_proba = model.predict_proba(X_transformed)[:, 1]
        else:
            model = MODELS[model_name]
            y_pred = model.predict(X)
            y_proba = model.predict_proba(X)[:, 1]

        st.subheader("Evaluation Metrics")

        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{accuracy_score(y, y_pred):.4f}")
        col2.metric("AUC", f"{roc_auc_score(y, y_proba):.4f}")
        col3.metric("MCC", f"{matthews_corrcoef(y, y_pred):.4f}")

        col4, col5, col6 = st.columns(3)
        col4.metric("Precision", f"{precision_score(y, y_pred):.4f}")
        col5.metric("Recall", f"{recall_score(y, y_pred):.4f}")
        col6.metric("F1 Score", f"{f1_score(y, y_pred):.4f}")

        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y, y_pred)

        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)