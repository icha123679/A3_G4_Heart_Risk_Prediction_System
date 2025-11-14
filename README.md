=========Heart Risk Prediction System=================


**1. Project Title & Objective**
Project Title:
Heart Disease Risk Prediction Using Machine Learning

Objective:
The goal of this project is to construct a simple and reliable system for the prediction of a person's heart disease based on medical information.
Two machine learning models, Logistic Regression and Decision Tree, have been employed for the classification of patients as High Risk or Low Risk.
The system also provides probability scores and visual insights that help users and healthcare professionals understand the predictions better.

**2. Dataset Details**
The system works with any commonly used heart disease dataset in CSV format.

Typical input features include:
Age
Gender
Chest pain type
Resting blood pressure
Serum cholesterol
Fasting blood sugar
Resting ECG result
Maximum heart rate
Exercise-induced angina
Oldpeak (ST depression)
ST segment slope
Number of major vessels
Thalassemia type

Target Column:
These are typically labeled as 0 (no disease) and 1 (disease present).
It automatically detects whichever target label the dataset uses.

**3. Algorithms/Models Used**

1. Logistic Regression
Ideal for binary prediction problems.
Turns inputs into probabilities using the sigmoid function.
Produces smooth, realistic probability scores.
Works well for the prediction of medical risk, as sensitivity and specificity are well-balanced.

2. Decision Tree Classifier
Predicts, utilizing a tree-like structure of rules.
Very easy to interpret; it's clear how the model derived a decision.
Depth limited to 5 levels to prevent overfitting.
May be sensitive to noise and sometimes gives extreme probability values.

**4. System Workflow**

A. Data Preprocessing
The system performs data cleaning and provides consistent results.
Removing rows containing missing values
Automatic identification of the target column
Conversion of text labels into numeric values
Feature scaling using StandardScaler (necessary for Logistic Regression)

B. Model Training
Dataset split into 80% training and 20% testing
Both models have been trained on the processed data.
Pre-trained models saved by joblib to be reused

C. Evaluation

The system assesses model performance according to:
Accuracy
Confusion matrix
Precision, recall, and F1-score
ROC curve and AUC score
Probability distribution comparison

D. Prediction Module

The Streamlit interface allows users to input patient data manually.
The system then returns
High Risk or Low Risk
Probability score
Model-specific insights

**5. Results**

• Accuracy
Logistic Regression gives slightly higher test accuracy.
Decision Tree does fairly well but shows moderate overfitting on smaller sets.

• Sensitivity & Specificity

Logistic Regression → Balanced and reliable
Decision Tree → High sensitivity but more false positives

• ROC–AUC

Logistic Regression: Smooth curve, Higher AUC
Decision Tree: more jumps and a lower AUC

• Probability Behavior

Logistic Regression → Gradual and realistic values
Decision Tree → Often extreme 0% or 100%

Overall Finding:

Logistic Regression is more dependable in terms of risk prediction, while the Decision Tree is better for interpretability.

**6. Conclusion**

The project successfully develops a machine learning-based system that predicts heart disease risk.
It automates the pre-processing of data, trains two models, and compares their performance; it also delivers understandable predictions via Streamlit.

Key points:

Logistic Regression is better for probability-based medical decisions.
Decision Tree is great in explaining how the prediction was done.
The system is user-friendly, interactive, and useful for early risk screening.
This can serve as a supportive tool for workers in the health field or as a learning platform for researchers and students.

**7. Future Scope**
   
 Some enhancements that can improve the system are: Adding ensemble models, such as Random Forest or XGBoost Using deep learning networks
 Integrating patient history with time-series data Applying explainability tools like SHAP Detecting data imbalance automatically
 Connect to electronic health record systems (EHRs) These improvements can make the system more accurate, transparent,
 and suitable for real clinical environments.

**8. References**

A collection of research articles on machine learning techniques, heart disease prediction, logistic regression, 
ROC analysis, and AI in healthcare was consulted to support this project
