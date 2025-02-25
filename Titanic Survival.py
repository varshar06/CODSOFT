import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

# Ensure dataset file exists
dataset_path = "Titanic-Dataset.csv"
if not os.path.isfile(dataset_path):
    raise FileNotFoundError(f"Dataset file '{dataset_path}' not found. Please check the file path.")

# Load Dataset
df = pd.read_csv(dataset_path)

# Display first few rows
print("Dataset Sample:")
print(df.head())

# Data Preprocessing
def preprocess_data(df):
    required_columns = {"Survived", "Age", "Fare", "Embarked", "Sex"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")
    
    # Drop irrelevant columns
    df = df.drop(["Name", "Ticket", "Cabin"], axis=1, errors='ignore')
    
    # Fill missing values
    df["Age"].fillna(df["Age"].median(), inplace=True)
    df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)
    df["Fare"].fillna(df["Fare"].median(), inplace=True)
    
    # Encode categorical variables
    label_enc = LabelEncoder()
    df["Sex"] = label_enc.fit_transform(df["Sex"])
    df["Embarked"] = label_enc.fit_transform(df["Embarked"])
    
    return df

df = preprocess_data(df)

# Data Visualization
sns.countplot(x="Survived", data=df, palette="coolwarm")
plt.title("Survival Count")
plt.show()

sns.heatmap(df.corr(), annot=True, cmap="Blues")
plt.title("Feature Correlations")
plt.show()

# Splitting Data
X = df.drop("Survived", axis=1)
y = df["Survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize Features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Model
model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model Evaluation
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))

