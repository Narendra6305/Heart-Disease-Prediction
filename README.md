# Heart Disease Prediction 💓

Predicting the presence of heart disease using clinical data and machine learning.

---

## 📖 Overview

This project builds a classification model to determine whether a patient has heart disease based on various medical attributes. It employs a full machine learning pipeline including:

1. **Data Loading & Cleaning**  
   Load the clinical dataset and handle missing values or irrelevant columns.

2. **Exploratory Data Analysis (EDA)**  
   Visualize distributions and relationships among features like age, cholesterol, blood pressure, and diagnostic results.

3. **Feature Engineering & Preprocessing**  
   - Apply transformations (e.g., scaling numeric features using `StandardScaler`)  
   - Encode categorical variables (e.g., chest pain types, resting ECG, exercise-induced angina).

4. **Train/Test Split**  
   Divide data into training and test sets (commonly 70/30 or 80/20 split).

5. **Model Training**  
   Evaluate classifiers such as Logistic Regression, Decision Tree, Random Forest, K‑Nearest Neighbors, Naive Bayes, and optionally Neural Networks.

6. **Model Evaluation**  
   Assess models using metrics like accuracy, precision, recall, F1-score, and ROC-AUC. Use confusion matrices for class-level insights.

7. **Model Selection & Saving**  
   Choose the best-performing model, optionally tune hyperparameters, and serialize it (e.g., using `pickle` or `joblib`).

---

## 🗂️ File Structure

├── data/
│ └── heart.csv # cleaned dataset (or raw CSV)
├── notebooks/
│ └── Heart Disease Prediction.ipynb # full analysis pipeline
├── models/
│ └── best_model.pkl # serialized final model (optional)
├── requirements.txt
└── README.md


---

## 🧩 Sample Modeling Code

Here’s a quick Python snippet illustrating training a Logistic Regression classifier:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

# Load data
df = pd.read_csv('data/heart.csv')

# Prepare features and target
X = df.drop(columns=['target'])
y = df['target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Standardize numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred))
print('ROC-AUC:', roc_auc_score(y_test, y_proba))

🔧 Requirements & Setup
Clone the repo:
git clone https://github.com/Narendra6305/Heart-Disease-Prediction.git

Install dependencies:
Typical packages: pandas, numpy, scikit-learn, matplotlib, seaborn.

Launch the notebook:
jupyter notebook notebooks/Heart\ Disease\ Prediction.ipynb

🔍 Findings & Observations
Key predictive features often include: chest pain type, exercise-induced angina, serum cholesterol, resting blood pressure, and maximum heart rate.

Model performance typically ranges from 80–90% accuracy, with ROC-AUC values around 0.85–0.95 depending on the algorithm and tuning.

Feature preprocessing (like scaling and encoding) improves model stability and performance.

📈 Advanced Steps
Compare multiple algorithms (e.g., Random Forest, SVM, XGBoost) and tune them via GridSearchCV or RandomizedSearchCV.

Employ cross-validation for more reliable evaluation.

Visualize ROC curves and confusion matrices.

Deploy model via a web app (Streamlit, Flask) or expose as an API.

Explain model behavior using tools like SHAP or LIME.

✨ Credits to Narendra6305
Based on  notebook analysis. Built using pandas, scikit-learn, and seaborn. Inspired by similar end-to-end heart disease prediction tutorials.

