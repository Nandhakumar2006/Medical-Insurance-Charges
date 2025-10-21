#### 🩺 Medical Insurance Cost Prediction



#### 📘 Overview

This project predicts medical insurance charges based on patient demographics and lifestyle factors. It uses a Random Forest Regressor to model complex non-linear relationships in the dataset and is deployed via Gradio for easy web-based interaction.

#### 📂 Dataset

Source: insurance.csv
Target Variable: charges (medical insurance cost)

#### Features Used:

age: Age of the individual

sex: Gender (encoded)

bmi: Body Mass Index

children: Number of dependents

smoker: Smoking status

region: Geographical region

#### ⚙️ Data Preprocessing

Removed duplicates and checked for missing values.

Explored skewness and distribution of numeric features.

Used one-hot encoding (pd.get_dummies) for categorical variables (sex, smoker, region, children).

Split dataset into 80% train / 20% test using train_test_split.

#### 🧠 Model — Random Forest Regressor
rf = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    max_depth=10,
    min_samples_split=5
)

#### 🤔 Why Random Forest?
Algorithm	Pros	Cons	Why Not Used
Linear Regression	Simple,
interpretable	Struggles with non-linear relationships	Insurance cost data is highly non-linear

Decision Tree	Handles non-linearities	Prone to overfitting	High variance

Support Vector Regression (SVR)	Good for small datasets	Computationally expensive on larger datasets	Slower for grid search

Random Forest (Chosen)	Combines many decision trees → low variance, handles non-linear patterns well, robust to outliers	Slightly less interpretable	✅ Best balance of accuracy, stability, and scalability

# Random Forest was chosen because:

It captures complex feature interactions (like smoker × BMI).

Provides excellent generalization and robustness to noise.

Consistently achieves high R² and low RMSE on unseen data.

#### 📈 Model Performance
Metric	Score

MSE: 20023009.46
RMSE: 4474.71
MAE: 2491.33
R²: 0.891

#### 🧾 Model Saving

The trained model is stored using pickle for deployment:

with open("rf_model.pkl", "wb") as f:
    pickle.dump((rf, training_columns), f)

#### 🚀 Deployment — Gradio App

File: app.py


#### 📊 Visualizations

Distribution plots for numeric features (age, bmi, children, charges).

Correlation heatmap to identify feature influence on cost.

#### 🧩 Technologies Used

Python 3.9+

Pandas, NumPy, Seaborn, Matplotlib

Scikit-learn (modeling)

Gradio (deployment)


###### APP LINK 

https://huggingface.co/spaces/nandha-01/InsurancePricePrediction-Medical
