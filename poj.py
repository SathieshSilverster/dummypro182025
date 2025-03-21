import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Step 1: Load Dataset
df = pd.read_csv(r'D:\vscode\dummypro182025\messy_dataset_100k.csv')  # Replace with actual dataset

# Step 2: Data Cleaning & Processing
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

# Step 3: Handling High-Cardinality Categorical Features
high_cardinality_threshold = 50  # Limit to top 50 unique categories

for col in df.select_dtypes(include=['object']).columns:
    if df[col].nunique() > high_cardinality_threshold:
        top_categories = df[col].value_counts().nlargest(high_cardinality_threshold).index
        df[col] = df[col].where(df[col].isin(top_categories), "Other")

# Step 4: Convert Categorical Variables Efficiently
label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])  # Convert categorical to numeric
    label_encoders[col] = le  # Store encoders if needed later

# Optional: Use One-Hot Encoding for Low-Cardinality Categorical Features
df = pd.get_dummies(df, dtype="uint8", sparse=True)

# Step 5: Exploratory Data Analysis (EDA)
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()

# Step 6: Feature Selection & Engineering
target_column = 'Department'  # Replace with actual target variable
X = df.drop(columns=[target_column])
y = df[target_column]

# Standardizing Numerical Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 7: Splitting the Data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 8: Model Selection & Training
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Step 9: Model Evaluation & Validation
y_pred = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Step 10: Hyperparameter Tuning
param_grid = {'n_estimators': [100, 200, 300], 'max_depth': [None, 10, 20]}
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2')
grid_search.fit(X_train, y_train)
print("Best Parameters:", grid_search.best_params_)

# Step 11: Model Deployment & Monitoring
joblib.dump(grid_search.best_estimator_, 'trained_model.pkl')
print("Model saved successfully!")
