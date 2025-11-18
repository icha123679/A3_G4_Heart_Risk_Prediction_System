
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc

from lifelines import CoxPHFitter   # Survival model

st.title("Heart Disease Model Trainer (Classification + Survival)📊")
st.write("This tool trains Logistic, Decision Tree, SVM, XGBoost, and Cox Survival Model.")



have_xgb = True
try:
    from xgboost import XGBClassifier
except Exception:
    have_xgb = False
    st.warning("xgboost not installed. XGBoost model will not be trained.")


uploaded_file = st.file_uploader("Upload dataset containing: output, time, event", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:")
    st.dataframe(df.head())

 
    # CHECK REQUIRED COLUMNS
    required_cols = ["output", "event", "time"]

    for col in required_cols:
        if col not in df.columns:
            st.error(f"Dataset must contain column: {col}")
            st.stop()

    
    # CLASSIFICATION TARGET (output)
    y_raw = df["output"].astype(str).str.lower()

    y = y_raw.map({
        "presence": 1,
        "absence": 0,
        "yes": 1,
        "no": 0,
        "positive": 1,
        "negative": 0,
        "1": 1,
        "0": 0
    })


    if y.isnull().any():
        y = pd.factorize(y_raw)[0]


    
    # (exclude output, event, time)
   
    X = df.drop(columns=["output", "event", "time"])

    # Handle categorical columns
    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = X[col].astype(str).factorize()[0]

    # Fill missing values
    X = X.fillna(X.median(numeric_only=True))

  
    # Feature Scaling

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

 
    # Train-Test Split
   
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train Logistic Regression
    log_model = LogisticRegression(max_iter=1000)
    log_model.fit(X_train, y_train)
    y_pred_log = log_model.predict(X_test)
    y_prob_log = log_model.predict_proba(X_test)[:, 1]
    acc_log = accuracy_score(y_test, y_pred_log)


    # Train Decision Tree
    tree_model = DecisionTreeClassifier(max_depth=5)
    tree_model.fit(X_train, y_train)
    y_pred_tree = tree_model.predict(X_test)
    y_prob_tree = tree_model.predict_proba(X_test)[:, 1]
    acc_tree = accuracy_score(y_test, y_pred_tree)


    # Train SVM (with probability)
    svm_model = SVC(probability=True, kernel="rbf")
    svm_model.fit(X_train, y_train)
    y_pred_svm = svm_model.predict(X_test)
    y_prob_svm = svm_model.predict_proba(X_test)[:, 1]
    acc_svm = accuracy_score(y_test, y_pred_svm)

    
    # Train XGBoost (optional)
    xgb_model = None
    if have_xgb:
        xgb_model = XGBClassifier(
            use_label_encoder=False,
            eval_metric="logloss",
            verbosity=0
        )
        xgb_model.fit(X_train, y_train)
        y_pred_xgb = xgb_model.predict(X_test)
        y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]
        acc_xgb = accuracy_score(y_test, y_pred_xgb)
    else:
        acc_xgb = None

    
    # Display Accuracy Scores
    st.subheader("Classifier Accuracy Comparison")
    st.write(f"Logistic Regression: `{acc_log:.4f}`")
    st.write(f"Decision Tree: `{acc_tree:.4f}`")
    st.write(f"SVM: `{acc_svm:.4f}`")
    if acc_xgb is not None:
        st.write(f"XGBoost: `{acc_xgb:.4f}`")


    # Confusion Matrix Plot Function

    def plot_cm(cm, title):
        fig, ax = plt.subplots()
        ax.imshow(cm, cmap="Blues")
        ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i, j], ha="center", va="center")
        st.pyplot(fig)

    st.subheader("Confusion Matrices")
    plot_cm(confusion_matrix(y_test, y_pred_log), "Logistic Regression")
    plot_cm(confusion_matrix(y_test, y_pred_tree), "Decision Tree")
    plot_cm(confusion_matrix(y_test, y_pred_svm), "SVM")
    if acc_xgb is not None:
        plot_cm(confusion_matrix(y_test, y_pred_xgb), "XGBoost")


    # ROC Curves
   
    def plot_roc(fpr, tpr, auc_score, title):
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"AUC = {auc_score:.3f}")
        ax.plot([0, 1], [0, 1], linestyle="--")
        ax.set_title(title)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend()
        st.pyplot(fig)

    fpr_log, tpr_log, _ = roc_curve(y_test, y_prob_log)
    fpr_tree, tpr_tree, _ = roc_curve(y_test, y_prob_tree)
    fpr_svm, tpr_svm, _ = roc_curve(y_test, y_prob_svm)

    auc_log = auc(fpr_log, tpr_log)
    auc_tree = auc(fpr_tree, tpr_tree)
    auc_svm = auc(fpr_svm, tpr_svm)

    st.subheader("ROC Curves")
    plot_roc(fpr_log, tpr_log, auc_log, "Logistic Regression")
    plot_roc(fpr_tree, tpr_tree, auc_tree, "Decision Tree")
    plot_roc(fpr_svm, tpr_svm, auc_svm, "SVM")

    if acc_xgb is not None:
        fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_prob_xgb)
        auc_xgb = auc(fpr_xgb, tpr_xgb)
        plot_roc(fpr_xgb, tpr_xgb, auc_xgb, "XGBoost")



    # Train Cox Survival Model (ALL columns must be numeric) 
    st.subheader("Training Cox Survival Model...")

    cox_df = df.copy()
    cox_df = cox_df.drop(columns=["output"])  # Remove non-numeric output column

    # Remove any non-numeric columns to avoid Cox errors
    for col in cox_df.columns:
        if cox_df[col].dtype == "object":
            st.error(f"Cox model cannot use non-numeric column: {col}")
            st.stop()

    cph = CoxPHFitter()

    try:
        cph.fit(cox_df, duration_col="time", event_col="event")
        st.success("Cox Survival Model trained successfully.")
    except Exception as e:
        st.error(f"Error training Cox model: {e}")
        st.stop()


   
    # SAVE MODELS
 
    joblib.dump(log_model, "logistic_model.pkl")
    joblib.dump(tree_model, "tree_model.pkl")
    joblib.dump(svm_model, "svm_model.pkl")

    if xgb_model is not None:
        joblib.dump(xgb_model, "xgb_model.pkl")

    joblib.dump(cph, "cox_model.pkl")
    joblib.dump(scaler, "scaler.pkl")

    st.success("All models saved successfully.")
    st.write("Saved files:")
    st.write("- logistic_model.pkl")
    st.write("- tree_model.pkl")
    st.write("- svm_model.pkl")
    if have_xgb:
        st.write("- xgb_model.pkl")
    st.write("- cox_model.pkl")
    st.write("- scaler.pkl")

else:
    st.info("Upload dataset to begin training.")

