
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(layout="centered")
st.title("Heart Disease Prediction and Future Risk💗🚑")

# Load models if present
models = {}
model_files = {
    "Logistic Regression": "logistic_model.pkl",
    "Decision Tree": "tree_model.pkl",
    "SVM": "svm_model.pkl",
    "XGBoost": "xgb_model.pkl"
}

for name, fname in model_files.items():
    if os.path.exists(fname):
        try:
            models[name] = joblib.load(fname)
        except Exception as e:
            st.warning(f"Could not load {fname}: {e}")

# Load scaler
scaler = None
if os.path.exists("scaler.pkl"):
    scaler = joblib.load("scaler.pkl")
else:
    st.warning("scaler.pkl not found. Classifier predictions will be unavailable until training is completed.")

# Load Cox model
cox_model = None
if os.path.exists("cox_model.pkl"):
    try:
        cox_model = joblib.load("cox_model.pkl")
    except Exception as e:
        st.warning(f"Could not load cox_model.pkl: {e}")
else:
    st.info("Cox survival model not found. Train models to enable survival-based future risk predictions.")


# Input form — fields must match training feature set
st.header("Patient details")

age = st.number_input("Age", 1, 120, 50)
sex = st.selectbox("Sex (1 = Male, 0 = Female)", [1, 0])
chest_pain_type = st.selectbox("Chest pain type", [0, 1, 2, 3, 4])  
bp = st.number_input("Resting blood pressure (BP)", 50, 250, 120)
cholesterol = st.number_input("Cholesterol", 50, 700, 200)
fbs_over_120 = st.selectbox("Fasting blood sugar > 120 mg/dl", [0, 1])
ekg_results = st.selectbox("EKG results", [0, 1, 2])
max_hr = st.number_input("Max heart rate achieved", 40, 250, 140)
exercise_angina = st.selectbox("Exercise induced angina (1 = yes, 0 = no)", [1, 0])
st_depression = st.number_input("ST depression", 0.0, 10.0, 1.0, step=0.1)
slope_of_st = st.selectbox("Slope of ST segment", [0, 1, 2])
number_of_vessels_fluro = st.selectbox("Number of vessels fluro (0-4)", [0, 1, 2, 3, 4])
thallium = st.number_input("Thallium", 0, 10, 3)

# Build raw input dataframe for Cox model (MUST match training column names exactly)
cox_input_df = pd.DataFrame([{
    "age": age,
    "sex": sex,
    "chest pain type": chest_pain_type,
    "bp": bp,
    "cholesterol": cholesterol,
    "fbs over 120": fbs_over_120,
    "ekg results": ekg_results,
    "max hr": max_hr,
    "exercise angina": exercise_angina,
    "st depression": st_depression,
    "slope of st": slope_of_st,
    "number of vessels fluro": number_of_vessels_fluro,
    "thallium": thallium
}])



# Build scaled input for classifiers
input_array = np.array([[age, sex, chest_pain_type, bp, cholesterol, fbs_over_120,
                         ekg_results, max_hr, exercise_angina, st_depression,
                         slope_of_st, number_of_vessels_fluro, thallium]])

if scaler is not None:
    try:
        scaled_input = scaler.transform(input_array)
    except Exception as e:
        st.error("Scaling failed: ensure feature order and count match training. Error: " + str(e))
        st.stop()
else:
    scaled_input = input_array   #validation if data is not trained yet


st.markdown("---")
st.subheader("Settings")
critical_threshold = st.slider("Critical threshold for alerts (%)", 1, 100, 70)
selected_models = st.multiselect("Select models to include in evaluation",
                                 list(models.keys()), default=list(models.keys()))

st.markdown("---")

# Predict button
if st.button("Predict"):

    if not models and cox_model is None:
        st.error("No models available. Train models first (run train_model.py).")
        st.stop()

    
    results = {}
    for name, mdl in models.items():
        try:
            prob = float(mdl.predict_proba(scaled_input)[0][1])
            pred = int(mdl.predict(scaled_input)[0])
        except Exception:
            
            try:
                score = mdl.decision_function(scaled_input)[0]
                prob = float(1.0 / (1.0 + np.exp(-score)))
                pred = int(prob >= 0.5)
            except Exception:
                st.warning(f"{name} cannot produce probability. Skipping.")
                continue
        results[name] = {"pred": pred, "prob": prob}

    # Survival predictions (1/5/10 years)  CoxFitter :    
    survival_results = {}
    if cox_model is not None:
        
        times_days = [365, 365 * 5, 365 * 10]
        try:
            surv_funcs = cox_model.predict_survival_function(cox_input_df, times=times_days)
            
            if isinstance(surv_funcs, pd.DataFrame):
            
                surv_1 = float(surv_funcs.loc[times_days[0]].iloc[0])
                surv_5 = float(surv_funcs.loc[times_days[1]].iloc[0])
                surv_10 = float(surv_funcs.loc[times_days[2]].iloc[0])
            else:
                # For other possible return shapes, attempt conversion
                arr = surv_funcs.values
                surv_1 = float(arr[0][0])
                surv_5 = float(arr[1][0])
                surv_10 = float(arr[2][0])
            survival_results = {
                "survival_1yr": surv_1,
                "survival_5yr": surv_5,
                "survival_10yr": surv_10,
                "risk_1yr": 1.0 - surv_1,
                "risk_5yr": 1.0 - surv_5,
                "risk_10yr": 1.0 - surv_10
            }
        except Exception as e:
            st.warning("Cox survival prediction failed: " + str(e))
            survival_results = {}

    # Display per-model classifier results
    st.subheader("Classifier Results")
    for name in selected_models:
        if name not in results:
            st.write(f"{name}: not available")
            continue
        row = results[name]
        label = "High risk" if row["pred"] == 1 else "Low risk"
        st.write(f"{name}: {label} (Probability: {row['prob']*100:.2f}%)")

    # Display survival results if available
    st.markdown("---")
    st.subheader("Survival-based Future Risk (CoxPH)")
    if survival_results:
        st.write(f"1-year survival probability: {survival_results['survival_1yr']*100:.2f}%")
        st.write(f"1-year risk (event): {survival_results['risk_1yr']*100:.2f}%")
        st.write(f"5-year survival probability: {survival_results['survival_5yr']*100:.2f}%")
        st.write(f"5-year risk (event): {survival_results['risk_5yr']*100:.2f}%")
        st.write(f"10-year survival probability: {survival_results['survival_10yr']*100:.2f}%")
        st.write(f"10-year risk (event): {survival_results['risk_10yr']*100:.2f}%")
    else:
        st.info("Survival predictions are not available. Ensure Cox model is trained and cox_model.pkl is present.")

    # Critical alert logic:
    
    critical_flag = False
    critical_reasons = []

    # classifier-based checks
    for name, info in results.items():
        if info["prob"] * 100 >= critical_threshold:
            critical_flag = True
            critical_reasons.append(f"{name} classifier probability {info['prob']*100:.2f}% >= threshold")

    # survival-based check (1-year)
    if survival_results:
        if survival_results["risk_1yr"] * 100 >= critical_threshold:
            critical_flag = True
            critical_reasons.append(f"1-year survival-based risk {survival_results['risk_1yr']*100:.2f}% >= threshold")

    # Final majority voting among selected classifiers (only classifiers considered)
    valid_models = [m for m in selected_models if m in results]
    final_decision = None

    if len(valid_models) == 0:
        st.write("No valid classifier predictions to perform majority voting.")
    else:
        high_votes = sum(1 for m in valid_models if results[m]["pred"] == 1)
        low_votes = sum(1 for m in valid_models if results[m]["pred"] == 0)

        # CASE 1: More high-risk votes
        if high_votes > low_votes:
            final_decision = ("High risk", high_votes, low_votes)

        # CASE 2: More low-risk votes
        elif low_votes > high_votes:
            final_decision = ("Low risk", low_votes, high_votes)

        # CASE 3: Tie → Use average probability
        else:
            avg_prob = float(np.mean([results[m]["prob"] for m in valid_models]))
            if avg_prob >= 0.5:
                final_decision = ("High risk (tie-break by average probability)", avg_prob, None)
            else:
                final_decision = ("Low risk (tie-break by average probability)", avg_prob, None)


    st.markdown("---")
    st.subheader("Final Decision and Alerts")

    if final_decision is not None:
        if isinstance(final_decision[1], int):
            st.write(f"Final decision (majority voting): {final_decision[0]} ({final_decision[1]} vs {final_decision[2]})")
        else:
            st.write(f"Final decision (tie-break by average probability): {final_decision[0]} (average probability = {final_decision[1]*100:.2f}%)")

    if critical_flag:
        st.write("Critical alert: this patient meets one or more criteria for critical risk.")
        st.write("Reasons:")
        for r in critical_reasons:
            st.write(f"- {r}")
        st.write("Recommendation: escalate to clinical evaluation.")
    else:
        st.write("No critical-level alert raised under the selected threshold.")

    # Risk comparison chart for classifiers
    if len(valid_models) > 0:
        st.markdown("---")
        st.subheader("Classifier Probability Comparison")

        names = valid_models
        probs = [results[n]["prob"] * 100 for n in names]

  
        fig, ax = plt.subplots(figsize=(10, 6)) 

        bars = ax.bar(names, probs)

        ax.set_ylim(0, 110)  
        ax.set_ylabel("Predicted probability (%)", fontsize=12)
        ax.set_title("Classifiers: Predicted Probability of Current Heart Disease Risk", fontsize=14)

        # Rotate names slightly if many models
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")

        # Add value labels above bars
        for i, v in enumerate(probs):
            ax.text(
                i,
                v + 3,  # moved higher to avoid touching bar
                f"{v:.1f}%",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold"
            )

        plt.tight_layout()  # prevents overlap on edges
        st.pyplot(fig)
    st.success("Prediction completed.")